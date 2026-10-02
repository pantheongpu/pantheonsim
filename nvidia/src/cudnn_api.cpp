// libvgpucudnn — VirtualGPU's implementation of the classic cuDNN API.
//
// Same boundary as cuBLAS: cuDNN is a library, so its math runs on the host
// and reads/writes virtual device memory rather than being interpreted. That
// keeps convolution and the pointwise layers off the ~10^8 lane-ops/s path.
//
// Scope is the legacy (descriptor) API that frameworks still use, forward and
// backward: convolution (data, filter and bias gradients, the fused
// bias-activation form, every algorithm cuDNN enumerates), activation,
// pooling, softmax, LRN, batch normalization, dropout, and tensor arithmetic
// (add, op-tensor, reduce, transform, set, scale). Tensors are 1-8
// dimensional with any strides -- NCHW, NHWC or neither -- of float, double,
// half or bfloat16, with the compute types cuDNN documents for each, and INT8
// convolution in NHWC. What cuDNN would compute that this library does not
// returns CUDNN_STATUS_NOT_SUPPORTED, by name, so a caller can fall back
// rather than receive a plausible wrong answer. The graph/backend API is in
// cudnn_backend.cpp, RNNs in cudnn_rnn.cpp.
//
// What the hardware answers was measured on an RTX 3060 with cuDNN 9.27
// (algorithm lists, status codes, which operand each gradient reads, tie
// breaking, rounding); nvidia/docs/libraries.md says where the arithmetic
// order makes results differ in the last bits.
//
// Compiled against the real cuDNN headers (nvidia/third_party/cudnn_include) so the
// ABI is the vendor's, not a guess.
#include "cudnn_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace vgpu_cudnn;

namespace {

struct Handle { cudaStream_t stream = nullptr; };

struct TensorDesc {
  Layout l;
  bool nd = false;            // set through an Nd setter: GetTensorNdDescriptor reports its own rank
  bool channels_last = false; // NHWC by format, or by strides with C innermost
};

struct FilterDesc {
  Layout l;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
};

struct ConvDesc {
  int nsp = 2;
  int64_t pad[3] = {0, 0, 0}, str[3] = {1, 1, 1}, dil[3] = {1, 1, 1};
  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  cudnnDataType_t type = CUDNN_DATA_FLOAT;  // the compute type
  int groups = 1;
  cudnnMathType_t math = CUDNN_DEFAULT_MATH;
  cudnnReorderType_t reorder = CUDNN_DEFAULT_REORDER;
};

struct ActDesc {
  cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU;
  cudnnNanPropagation_t nan = CUDNN_NOT_PROPAGATE_NAN;
  double coef = 0.0;
  double swish_beta = 1.0;  // separate setter in cuDNN, so it survives SetActivationDescriptor
};

struct PoolDesc {
  cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
  cudnnNanPropagation_t nan = CUDNN_NOT_PROPAGATE_NAN;
  int nsp = 2;
  int win[3] = {1, 1, 1}, pad[3] = {0, 0, 0}, str[3] = {1, 1, 1};
};

struct OpDesc {
  cudnnOpTensorOp_t op = CUDNN_OP_TENSOR_ADD;
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  cudnnNanPropagation_t nan = CUDNN_NOT_PROPAGATE_NAN;
};

struct ReduceDesc {
  cudnnReduceTensorOp_t op = CUDNN_REDUCE_TENSOR_ADD;
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  cudnnNanPropagation_t nan = CUDNN_NOT_PROPAGATE_NAN;
  cudnnReduceTensorIndices_t indices = CUDNN_REDUCE_TENSOR_NO_INDICES;
  cudnnIndicesType_t index_type = CUDNN_32BIT_INDICES;
};

struct LrnDesc {
  unsigned n = 5;
  double alpha = 1e-4, beta = 0.75, k = 2.0;  // cuDNN's defaults
};

template <class T, class D> T* as(D d) { return reinterpret_cast<T*>(d); }

// A tensor descriptor that exists and has been given a shape.
const TensorDesc* tdesc(const void* d) {
  if (!known(d)) return nullptr;
  const auto* t = static_cast<const TensorDesc*>(d);
  return t->l.rank ? t : nullptr;
}
const FilterDesc* fdesc(const void* d) {
  if (!known(d)) return nullptr;
  const auto* f = static_cast<const FilterDesc*>(d);
  return f->l.rank ? f : nullptr;
}

#define BAD(fn, why) fail(CUDNN_STATUS_BAD_PARAM, fn, why)
#define UNSUPPORTED(fn, why) fail(CUDNN_STATUS_NOT_SUPPORTED, fn, why)

// Whether every dimension of a is c's or 1: what may be broadcast onto c.
bool broadcastable(const Layout& a, const Layout& c) {
  if (a.rank != c.rank) return false;
  for (int i = 0; i < a.rank; ++i)
    if (a.dims[i] != c.dims[i] && a.dims[i] != 1) return false;
  return true;
}

// For each element of c (logical order), the logical index of the element of
// a broadcast onto it.
std::vector<size_t> broadcast_index(const Layout& a, const Layout& c) {
  std::vector<size_t> idx(c.count());
  int64_t at[kMaxRank] = {};
  for (size_t i = 0; i < idx.size(); ++i) {
    size_t j = 0;
    for (int d = 0; d < c.rank; ++d) j = j * a.dims[d] + (a.dims[d] == 1 ? 0 : at[d]);
    idx[i] = j;
    for (int d = c.rank; d-- > 0;) {
      if (++at[d] < c.dims[d]) break;
      at[d] = 0;
    }
  }
  return idx;
}

// The scaling factors' type for an operation writing `out`.
inline double sc(const void* p, const Layout& out) { return scale_of(p, out.type); }

// N, C and the product of the remaining (spatial) extents.
void ncs(const Layout& t, int64_t* N, int64_t* C, int64_t* S) {
  *N = t.rank > 0 ? t.dims[0] : 1;
  *C = t.rank > 1 ? t.dims[1] : 1;
  *S = 1;
  for (int i = 2; i < t.rank; ++i) *S *= t.dims[i];
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- lifecycle ---- */

VGPU_EXPORT cudnnStatus_t cudnnCreate(cudnnHandle_t* h) {
  if (!h) return CUDNN_STATUS_BAD_PARAM;
  *h = reinterpret_cast<cudnnHandle_t>(track(new Handle()));
  if (!quiet())
    std::fprintf(stderr, "[vgpu] cuDNN handle created (host-computed; see nvidia/docs/libraries.md)\n");
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroy(cudnnHandle_t h) {
  if (!known(h)) return CUDNN_STATUS_BAD_PARAM;
  untrack(h); delete as<Handle>(h);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetStream(cudnnHandle_t h, cudaStream_t s) {
  if (!known(h)) return CUDNN_STATUS_BAD_PARAM;
  as<Handle>(h)->stream = s;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetStream(cudnnHandle_t h, cudaStream_t* s) {
  if (!known(h) || !s) return CUDNN_STATUS_BAD_PARAM;
  *s = as<Handle>(h)->stream;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetProperty(libraryPropertyType type, int* value) {
  if (!value) return CUDNN_STATUS_BAD_PARAM;
  switch (type) {
    // The release whose headers this is built against, as cudnnGetVersion
    // reports it: PyTorch refuses a runtime older than the one it was built
    // with, and compares these.
    case MAJOR_VERSION: *value = CUDNN_MAJOR; break;
    case MINOR_VERSION: *value = CUDNN_MINOR; break;
    case PATCH_LEVEL: *value = CUDNN_PATCHLEVEL; break;
    default: return CUDNN_STATUS_BAD_PARAM;
  }
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT size_t cudnnGetVersion(void) { return CUDNN_VERSION; }
VGPU_EXPORT size_t cudnnGetMaxDeviceVersion(void) { return CUDNN_MAX_DEVICE_VERSION; }
VGPU_EXPORT size_t cudnnGetCudartVersion(void) { return 13000; }
// The sub-libraries are all this one library, so always the same version.
VGPU_EXPORT cudnnStatus_t cudnnGraphVersionCheck(void) { return CUDNN_STATUS_SUCCESS; }
VGPU_EXPORT cudnnStatus_t cudnnOpsVersionCheck(void) { return CUDNN_STATUS_SUCCESS; }
VGPU_EXPORT cudnnStatus_t cudnnCnnVersionCheck(void) { return CUDNN_STATUS_SUCCESS; }

VGPU_EXPORT const char* cudnnGetErrorString(cudnnStatus_t s) {
  switch (s) {
#define NAME(x) case x: return #x;
    NAME(CUDNN_STATUS_SUCCESS)
    NAME(CUDNN_STATUS_NOT_INITIALIZED)
    NAME(CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH)
    NAME(CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH)
    NAME(CUDNN_STATUS_DEPRECATED)
    NAME(CUDNN_STATUS_LICENSE_ERROR)
    NAME(CUDNN_STATUS_RUNTIME_IN_PROGRESS)
    NAME(CUDNN_STATUS_RUNTIME_FP_OVERFLOW)
    NAME(CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED)
    NAME(CUDNN_STATUS_BAD_PARAM)
    NAME(CUDNN_STATUS_BAD_PARAM_NULL_POINTER)
    NAME(CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER)
    NAME(CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED)
    NAME(CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND)
    NAME(CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT)
    NAME(CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH)
    NAME(CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH)
    NAME(CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES)
    NAME(CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE)
    NAME(CUDNN_STATUS_BAD_PARAM_CUDA_GRAPH_MISMATCH)
    NAME(CUDNN_STATUS_BAD_PARAM_DESCRIPTOR_TYPE)
    NAME(CUDNN_STATUS_NOT_SUPPORTED)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_SHAPE)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_LAYOUT)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_PADDING)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM)
    NAME(CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API)
    NAME(CUDNN_STATUS_INTERNAL_ERROR)
    NAME(CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED)
    NAME(CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE)
    NAME(CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED)
    NAME(CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED)
    NAME(CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM)
    NAME(CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED)
    NAME(CUDNN_STATUS_EXECUTION_FAILED)
    NAME(CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER)
    NAME(CUDNN_STATUS_EXECUTION_FAILED_CUBLAS)
    NAME(CUDNN_STATUS_EXECUTION_FAILED_CUDART)
    NAME(CUDNN_STATUS_EXECUTION_FAILED_CURAND)
#undef NAME
    default: return "CUDNN_UNKNOWN_STATUS";
  }
}

// The reason the last failing call on this thread gave.
VGPU_EXPORT void cudnnGetLastErrorString(char* message, size_t max_size) {
  if (!message || !max_size) return;
  const std::string s = last_error();
  const size_t n = std::min(s.size(), max_size - 1);
  std::memcpy(message, s.data(), n);
  message[n] = '\0';
}

// Nothing runs asynchronously inside this library, so nothing is pending.
VGPU_EXPORT cudnnStatus_t cudnnQueryRuntimeError(cudnnHandle_t h, cudnnStatus_t* rstatus, cudnnErrQueryMode_t,
                                                 cudnnRuntimeTag_t*) {
  if (!known(h) || !rstatus) return CUDNN_STATUS_BAD_PARAM;
  *rstatus = CUDNN_STATUS_SUCCESS;
  return CUDNN_STATUS_SUCCESS;
}

// The logging callback: kept and handed back; this library logs nothing
// through it.
namespace { cudnnCallback_t g_callback = nullptr; void* g_udata = nullptr; unsigned g_mask = 0; }
VGPU_EXPORT cudnnStatus_t cudnnSetCallback(unsigned mask, void* udata, cudnnCallback_t fptr) {
  g_mask = mask, g_udata = udata, g_callback = fptr;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetCallback(unsigned* mask, void** udata, cudnnCallback_t* fptr) {
  if (mask) *mask = g_mask;
  if (udata) *udata = g_udata;
  if (fptr) *fptr = g_callback;
  return CUDNN_STATUS_SUCCESS;
}

/* ---- tensor descriptors ---- */

VGPU_EXPORT cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnTensorDescriptor_t>(track(new TensorDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete as<TensorDesc>(d);
  return CUDNN_STATUS_SUCCESS;
}

namespace {
// Shared by every tensor setter: the type must be one this library stores
// element by element, and every extent and stride positive.
cudnnStatus_t set_tensor(const char* fn, TensorDesc* t, cudnnDataType_t type, int nb, const int* dims,
                         const int* strides, int fmt) {
  if (nb > kMaxRank) return UNSUPPORTED(fn, "more than " + std::to_string(kMaxRank) + " dimensions");
  if (nb < 1) return BAD(fn, "a tensor needs at least one dimension");
  if (fmt == CUDNN_TENSOR_NCHW_VECT_C)
    return UNSUPPORTED(fn, "the vectorized CUDNN_TENSOR_NCHW_VECT_C layout is not supported");
  if (!storable(type) || type == CUDNN_DATA_BOOLEAN)
    return UNSUPPORTED(fn, std::string("data type ") + type_name(type) + " is not supported");
  Layout l;
  l.type = type;
  l.rank = nb;
  for (int i = 0; i < nb; ++i) {
    if (dims[i] <= 0) return BAD(fn, "dimension " + std::to_string(i) + " is " + std::to_string(dims[i]));
    l.dims[i] = dims[i];
  }
  if (strides) {
    for (int i = 0; i < nb; ++i) {
      if (strides[i] <= 0) return BAD(fn, "stride " + std::to_string(i) + " is " + std::to_string(strides[i]));
      l.strides[i] = strides[i];
    }
  } else {
    packed_strides(nb, l.dims, fmt == CUDNN_TENSOR_NHWC, l.strides);
  }
  t->l = l;
  t->channels_last = fmt == CUDNN_TENSOR_NHWC ||
                     (fmt < 0 && nb >= 3 && l.strides[1] == 1 && l.dims[1] > 1);
  return CUDNN_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d, cudnnTensorFormat_t fmt,
                                                     cudnnDataType_t type, int n, int c, int h, int w) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  if (fmt != CUDNN_TENSOR_NCHW && fmt != CUDNN_TENSOR_NHWC && fmt != CUDNN_TENSOR_NCHW_VECT_C)
    return BAD("cudnnSetTensor4dDescriptor", "unknown format");
  const int dims[4] = {n, c, h, w};
  cudnnStatus_t s = set_tensor("cudnnSetTensor4dDescriptor", as<TensorDesc>(d), type, 4, dims, nullptr, fmt);
  if (s == CUDNN_STATUS_SUCCESS) as<TensorDesc>(d)->nd = false;
  return s;
}
VGPU_EXPORT cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t d, cudnnDataType_t type, int n, int c,
                                                       int h, int w, int sn, int sc_, int sh, int sw) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const int dims[4] = {n, c, h, w}, strides[4] = {sn, sc_, sh, sw};
  cudnnStatus_t s = set_tensor("cudnnSetTensor4dDescriptorEx", as<TensorDesc>(d), type, 4, dims, strides, -1);
  if (s == CUDNN_STATUS_SUCCESS) as<TensorDesc>(d)->nd = false;
  return s;
}
VGPU_EXPORT cudnnStatus_t cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t d, cudnnDataType_t* type, int* n,
                                                     int* c, int* h, int* w, int* sn, int* sc_, int* sh, int* sw) {
  const TensorDesc* t = tdesc(d);
  if (!t) return CUDNN_STATUS_BAD_PARAM;
  // A tensor of another rank answers with its first four dimensions, as
  // cuDNN does; missing ones read as extent and stride 1.
  auto dim = [&](int i) { return i < t->l.rank ? (int)t->l.dims[i] : 1; };
  auto str = [&](int i) { return i < t->l.rank ? (int)t->l.strides[i] : 1; };
  if (type) *type = t->l.type;
  if (n) *n = dim(0);
  if (c) *c = dim(1);
  if (h) *h = dim(2);
  if (w) *w = dim(3);
  if (sn) *sn = str(0);
  if (sc_) *sc_ = str(1);
  if (sh) *sh = str(2);
  if (sw) *sw = str(3);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t d, cudnnDataType_t type, int nb,
                                                     const int dims[], const int strides[]) {
  if (!known(d) || !dims || !strides) return CUDNN_STATUS_BAD_PARAM;
  cudnnStatus_t s = set_tensor("cudnnSetTensorNdDescriptor", as<TensorDesc>(d), type, nb, dims, strides, -1);
  if (s == CUDNN_STATUS_SUCCESS) as<TensorDesc>(d)->nd = true;
  return s;
}
VGPU_EXPORT cudnnStatus_t cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t d, cudnnTensorFormat_t fmt,
                                                       cudnnDataType_t type, int nb, const int dims[]) {
  if (!known(d) || !dims) return CUDNN_STATUS_BAD_PARAM;
  if (fmt != CUDNN_TENSOR_NCHW && fmt != CUDNN_TENSOR_NHWC && fmt != CUDNN_TENSOR_NCHW_VECT_C)
    return BAD("cudnnSetTensorNdDescriptorEx", "unknown format");
  cudnnStatus_t s = set_tensor("cudnnSetTensorNdDescriptorEx", as<TensorDesc>(d), type, nb, dims, nullptr, fmt);
  if (s == CUDNN_STATUS_SUCCESS) as<TensorDesc>(d)->nd = true;
  return s;
}
VGPU_EXPORT cudnnStatus_t cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t d, int requested,
                                                     cudnnDataType_t* type, int* nb, int dims[], int strides[]) {
  const TensorDesc* t = tdesc(d);
  if (!t) return CUDNN_STATUS_BAD_PARAM;
  if (type) *type = t->l.type;
  if (nb) *nb = t->l.rank;
  for (int i = 0; i < std::min(requested, t->l.rank); ++i) {
    if (dims) dims[i] = static_cast<int>(t->l.dims[i]);
    if (strides) strides[i] = static_cast<int>(t->l.strides[i]);
  }
  return CUDNN_STATUS_SUCCESS;
}
// Bytes from the first element to one past the last, as the strides lay them
// out -- what has to be allocated.
VGPU_EXPORT cudnnStatus_t cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t d, size_t* size) {
  const TensorDesc* t = tdesc(d);
  if (!t || !size) return CUDNN_STATUS_BAD_PARAM;
  *size = t->l.span() * type_bytes(t->l.type);
  return CUDNN_STATUS_SUCCESS;
}

/* ---- filter descriptors ---- */

VGPU_EXPORT cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnFilterDescriptor_t>(track(new FilterDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete as<FilterDesc>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t d, cudnnDataType_t type,
                                                     cudnnTensorFormat_t fmt, int nb, const int dims[]) {
  static const char* fn = "cudnnSetFilterNdDescriptor";
  if (!known(d) || !dims) return CUDNN_STATUS_BAD_PARAM;
  if (fmt == CUDNN_TENSOR_NCHW_VECT_C) return UNSUPPORTED(fn, "the vectorized CUDNN_TENSOR_NCHW_VECT_C layout is not supported");
  if (fmt != CUDNN_TENSOR_NCHW && fmt != CUDNN_TENSOR_NHWC) return BAD(fn, "unknown format");
  if (nb < 3 || nb > kMaxRank) return BAD(fn, "a filter has 3 to 8 dimensions");
  if (!storable(type) || type == CUDNN_DATA_BOOLEAN)
    return UNSUPPORTED(fn, std::string("data type ") + type_name(type) + " is not supported");
  FilterDesc f;
  f.format = fmt;
  f.l.type = type;
  f.l.rank = nb;
  for (int i = 0; i < nb; ++i) {
    if (dims[i] <= 0) return BAD(fn, "dimension " + std::to_string(i) + " is " + std::to_string(dims[i]));
    f.l.dims[i] = dims[i];
  }
  // NHWC filters are KRSC in memory: input channels innermost.
  packed_strides(nb, f.l.dims, fmt == CUDNN_TENSOR_NHWC, f.l.strides);
  *as<FilterDesc>(d) = f;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t d, cudnnDataType_t type,
                                                     cudnnTensorFormat_t fmt, int k, int c, int h, int w) {
  const int dims[4] = {k, c, h, w};
  return cudnnSetFilterNdDescriptor(d, type, fmt, 4, dims);
}
VGPU_EXPORT cudnnStatus_t cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t d, cudnnDataType_t* type,
                                                     cudnnTensorFormat_t* fmt, int* k, int* c, int* h, int* w) {
  const FilterDesc* f = fdesc(d);
  if (!f) return CUDNN_STATUS_BAD_PARAM;
  auto dim = [&](int i) { return i < f->l.rank ? (int)f->l.dims[i] : 1; };
  if (type) *type = f->l.type;
  if (fmt) *fmt = f->format;
  if (k) *k = dim(0);
  if (c) *c = dim(1);
  if (h) *h = dim(2);
  if (w) *w = dim(3);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t d, int requested,
                                                     cudnnDataType_t* type, cudnnTensorFormat_t* fmt, int* nb,
                                                     int dims[]) {
  const FilterDesc* f = fdesc(d);
  if (!f) return CUDNN_STATUS_BAD_PARAM;
  if (type) *type = f->l.type;
  if (fmt) *fmt = f->format;
  if (nb) *nb = f->l.rank;
  for (int i = 0; dims && i < std::min(requested, f->l.rank); ++i) dims[i] = static_cast<int>(f->l.dims[i]);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t d, size_t* size) {
  const FilterDesc* f = fdesc(d);
  if (!f || !size) return CUDNN_STATUS_BAD_PARAM;
  *size = f->l.count() * type_bytes(f->l.type);
  return CUDNN_STATUS_SUCCESS;
}

/* ---- convolution descriptors ---- */

VGPU_EXPORT cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnConvolutionDescriptor_t>(track(new ConvDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete as<ConvDesc>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t d, int n, const int pads[],
                                                          const int strides[], const int dils[],
                                                          cudnnConvolutionMode_t mode, cudnnDataType_t type) {
  static const char* fn = "cudnnSetConvolutionNdDescriptor";
  if (!known(d) || !pads || !strides || !dils || n < 1) return CUDNN_STATUS_BAD_PARAM;
  if (n > 3) return UNSUPPORTED(fn, "more than 3 spatial dimensions");
  if (mode != CUDNN_CONVOLUTION && mode != CUDNN_CROSS_CORRELATION) return BAD(fn, "unknown mode");
  auto* c = as<ConvDesc>(d);
  ConvDesc nc = *c;  // groups, math type and reorder type survive
  nc.nsp = n, nc.mode = mode, nc.type = type;
  for (int i = 0; i < n; ++i) {
    if (pads[i] < 0 || strides[i] <= 0 || dils[i] <= 0) return BAD(fn, "a padding, stride or dilation is out of range");
    nc.pad[i] = pads[i], nc.str[i] = strides[i], nc.dil[i] = dils[i];
  }
  *c = nc;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t d, int ph, int pw, int sh,
                                                          int sw, int dh, int dw, cudnnConvolutionMode_t mode,
                                                          cudnnDataType_t type) {
  const int pads[2] = {ph, pw}, strides[2] = {sh, sw}, dils[2] = {dh, dw};
  return cudnnSetConvolutionNdDescriptor(d, 2, pads, strides, dils, mode, type);
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t d, int* ph, int* pw,
                                                          int* sh, int* sw, int* dh, int* dw,
                                                          cudnnConvolutionMode_t* mode, cudnnDataType_t* type) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const auto* c = as<const ConvDesc>(d);
  if (ph) *ph = (int)c->pad[0];
  if (pw) *pw = (int)c->pad[1];
  if (sh) *sh = (int)c->str[0];
  if (sw) *sw = (int)c->str[1];
  if (dh) *dh = (int)c->dil[0];
  if (dw) *dw = (int)c->dil[1];
  if (mode) *mode = c->mode;
  if (type) *type = c->type;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t d, int requested, int* n,
                                                          int pads[], int strides[], int dils[],
                                                          cudnnConvolutionMode_t* mode, cudnnDataType_t* type) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const auto* c = as<const ConvDesc>(d);
  if (n) *n = c->nsp;
  for (int i = 0; i < std::min(requested, c->nsp); ++i) {
    if (pads) pads[i] = (int)c->pad[i];
    if (strides) strides[i] = (int)c->str[i];
    if (dils) dils[i] = (int)c->dil[i];
  }
  if (mode) *mode = c->mode;
  if (type) *type = c->type;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t d, int g) {
  if (!known(d) || g < 1) return CUDNN_STATUS_BAD_PARAM;
  as<ConvDesc>(d)->groups = g;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t d, int* g) {
  if (!known(d) || !g) return CUDNN_STATUS_BAD_PARAM;
  *g = as<ConvDesc>(d)->groups;
  return CUDNN_STATUS_SUCCESS;
}
// Tensor-core math changes nothing here: every configuration computes in its
// documented compute type. The setting is kept and reported back.
VGPU_EXPORT cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t d, cudnnMathType_t m) {
  if (!known(d) || m < CUDNN_DEFAULT_MATH || m > CUDNN_FMA_MATH) return CUDNN_STATUS_BAD_PARAM;
  as<ConvDesc>(d)->math = m;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t d, cudnnMathType_t* m) {
  if (!known(d) || !m) return CUDNN_STATUS_BAD_PARAM;
  *m = as<ConvDesc>(d)->math;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t d, cudnnReorderType_t r) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  as<ConvDesc>(d)->reorder = r;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t d, cudnnReorderType_t* r) {
  if (!known(d) || !r) return CUDNN_STATUS_BAD_PARAM;
  *r = as<ConvDesc>(d)->reorder;
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t cd,
                                                                const cudnnTensorDescriptor_t xd,
                                                                const cudnnFilterDescriptor_t wd, int nb, int out[]) {
  static const char* fn = "cudnnGetConvolutionNdForwardOutputDim";
  const TensorDesc* x = tdesc(xd);
  const FilterDesc* w = fdesc(wd);
  if (!known(cd) || !x || !w || !out) return CUDNN_STATUS_BAD_PARAM;
  const auto* c = as<const ConvDesc>(cd);
  if (x->l.rank != c->nsp + 2 || w->l.rank != x->l.rank || nb < x->l.rank)
    return BAD(fn, "the tensor, filter and convolution ranks do not agree");
  if (x->l.dims[1] != w->l.dims[1] * c->groups)
    return BAD(fn, "input channels are not the filter's times the group count");
  out[0] = (int)x->l.dims[0];
  out[1] = (int)w->l.dims[0];
  for (int i = 0; i < c->nsp; ++i) {
    const int64_t o = conv_out(x->l.dims[2 + i], c->pad[i], w->l.dims[2 + i], c->str[i], c->dil[i]);
    if (o <= 0) return BAD(fn, "the filter is larger than the padded input");
    out[2 + i] = (int)o;
  }
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t cd,
                                                                const cudnnTensorDescriptor_t xd,
                                                                const cudnnFilterDescriptor_t wd, int* n, int* c,
                                                                int* h, int* w) {
  int out[4];
  cudnnStatus_t s = cudnnGetConvolutionNdForwardOutputDim(cd, xd, wd, 4, out);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  if (n) *n = out[0];
  if (c) *c = out[1];
  if (h) *h = out[2];
  if (w) *w = out[3];
  return CUDNN_STATUS_SUCCESS;
}

/* ---- convolution: configurations, algorithms, workspace ---- */

namespace {

// One convolution call's operands, checked: their shapes fit together
// (else BAD_PARAM) and their types are one of cuDNN's documented
// configurations (else NOT_SUPPORTED).
struct ConvCall {
  ConvGeom g;
  Accum acc = Accum::Exact;
  const Layout *x = nullptr, *w = nullptr, *y = nullptr;
  cudnnDataType_t scale_type = CUDNN_DATA_FLOAT;
};

cudnnStatus_t conv_check(const char* fn, ConvDir dir, const void* xd, const void* wd, const void* cd, const void* yd,
                         ConvCall* call) {
  const TensorDesc* x = tdesc(xd);
  const FilterDesc* w = fdesc(wd);
  const TensorDesc* y = tdesc(yd);
  if (!x || !w || !y || !known(cd)) return BAD(fn, "a descriptor is null, destroyed or never set");
  const auto* c = static_cast<const ConvDesc*>(cd);
  std::string why;
  if (!conv_geometry(x->l, w->l, y->l, c->nsp, c->pad, c->str, c->dil, c->mode == CUDNN_CONVOLUTION, &call->g, &why))
    return BAD(fn, why);
  if (call->g.G != c->groups)
    return BAD(fn, "input channels " + std::to_string(call->g.C) + " are not the filter's " +
                       std::to_string(call->g.Cg) + " times the group count " + std::to_string(c->groups));
  call->x = &x->l, call->w = &w->l, call->y = &y->l;
  const cudnnDataType_t xt = x->l.type, wt = w->l.type, yt = y->l.type, ct = c->type;
  auto config = [&]() -> std::string {
    return std::string("x ") + type_name(xt) + ", w " + type_name(wt) + ", y " + type_name(yt) + " with compute type " +
           type_name(ct) + " is not one of cuDNN's convolution configurations";
  };
  call->scale_type = CUDNN_DATA_FLOAT;
  if (xt == CUDNN_DATA_INT8 && wt == CUDNN_DATA_INT8) {
    // INT8_CONFIG and INT8_EXT_CONFIG: forward only, channels-last only.
    if (dir != ConvDir::Forward || ct != CUDNN_DATA_INT32 || (yt != CUDNN_DATA_INT8 && yt != CUDNN_DATA_FLOAT))
      return UNSUPPORTED(fn, config());
    if (!x->channels_last || w->format != CUDNN_TENSOR_NHWC || !y->channels_last)
      return UNSUPPORTED(fn, "INT8 convolution needs NHWC tensors and filters");
    call->acc = Accum::Int;
    return CUDNN_STATUS_SUCCESS;
  }
  if (xt != wt || xt != yt) return UNSUPPORTED(fn, config());
  switch (xt) {
    case CUDNN_DATA_HALF:  // TRUE_HALF_CONFIG and PSEUDO_HALF_CONFIG
      if (ct != CUDNN_DATA_HALF && ct != CUDNN_DATA_FLOAT) return UNSUPPORTED(fn, config());
      call->acc = ct == CUDNN_DATA_HALF ? Accum::Half : Accum::Exact;
      return CUDNN_STATUS_SUCCESS;
    case CUDNN_DATA_BFLOAT16:  // PSEUDO_BFLOAT16_CONFIG
    case CUDNN_DATA_FLOAT:     // FLOAT_CONFIG
      if (ct != CUDNN_DATA_FLOAT) return UNSUPPORTED(fn, config());
      call->acc = Accum::Exact;
      return CUDNN_STATUS_SUCCESS;
    case CUDNN_DATA_DOUBLE:  // DOUBLE_CONFIG
      if (ct != CUDNN_DATA_DOUBLE) return UNSUPPORTED(fn, config());
      call->acc = Accum::Exact;
      call->scale_type = CUDNN_DATA_DOUBLE;
      return CUDNN_STATUS_SUCCESS;
    default:
      return UNSUPPORTED(fn, config());
  }
}

// cuDNN's algorithms, in the order this library lists them: the one cuDNN's
// heuristics put first for most shapes leads. Every one computes the same
// convolution here, exactly and deterministically, with no workspace, except
// the two cuDNN itself has never implemented -- forward DIRECT and
// backward-filter WINOGRAD -- which are listed last as NOT_SUPPORTED, as the
// hardware lists them.
constexpr int kFwdAlgos[] = {CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                             CUDNN_CONVOLUTION_FWD_ALGO_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_FFT,
                             CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                             CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT};
constexpr int kDataAlgos[] = {CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                              CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
                              CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
                              CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED};
constexpr int kFilterAlgos[] = {CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
                                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
                                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
                                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD};
constexpr int kFwdCount = sizeof kFwdAlgos / sizeof kFwdAlgos[0];
constexpr int kDataCount = sizeof kDataAlgos / sizeof kDataAlgos[0];
constexpr int kFilterCount = sizeof kFilterAlgos / sizeof kFilterAlgos[0];

bool algo_runs(ConvDir dir, int algo) {
  switch (dir) {
    case ConvDir::Forward: return algo >= 0 && algo < kFwdCount && algo != CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
    case ConvDir::Data: return algo >= 0 && algo < kDataCount;
    default: return algo >= 0 && algo < kFilterCount && algo != CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
  }
}

const char* dir_name(ConvDir dir) {
  return dir == ConvDir::Forward ? "forward" : dir == ConvDir::Data ? "backward-data" : "backward-filter";
}

// Fills cuDNN's perf structs (all three have the same members). `timed` is
// the Find form, which reports a time for each algorithm that ran (here, a
// nominal one in list order) and -1 for one that did not.
template <class Perf>
void list_algos(ConvDir dir, const int* order, int count, int requested, int* returned, Perf* perf, bool config_ok,
                cudnnMathType_t math, bool timed) {
  const int n = std::min(requested, count);
  // Runnable ones first, then the rest, as the hardware sorts them.
  int k = 0;
  for (int pass = 0; pass < 2; ++pass)
    for (int i = 0; i < count && k < n; ++i) {
      const bool runs = config_ok && algo_runs(dir, order[i]);
      if (runs != (pass == 0)) continue;
      Perf& p = perf[k];
      std::memset(&p, 0, sizeof p);
      p.algo = static_cast<decltype(p.algo)>(order[i]);
      p.status = runs ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_NOT_SUPPORTED;
      p.time = timed ? (runs ? 0.001f * (k + 1) : -1.0f) : -1.0f;
      p.memory = 0;
      p.determinism = CUDNN_DETERMINISTIC;
      p.mathType = math;
      ++k;
    }
  *returned = k;
}

cudnnStatus_t run_conv(const char* fn, ConvDir dir, cudnnHandle_t h, int algo, const void* alpha, const void* a,
                       const void* b, const void* beta, void* out, const ConvCall& call) {
  if (!algo_runs(dir, algo))
    return UNSUPPORTED(fn, std::string(dir_name(dir)) + " algorithm " + std::to_string(algo) + " is not one cuDNN implements");
  if (!alpha || !beta || !a || !b || !out) return BAD(fn, "a scaling factor or data pointer is null");
  sync_handle(h);
  std::vector<double> va, vb, r;
  const Layout* la = dir == ConvDir::Forward ? call.x : call.y;
  const Layout* lb = dir == ConvDir::Filter ? call.x : call.w;
  const Layout* lo = dir == ConvDir::Forward ? call.y : dir == ConvDir::Data ? call.x : call.w;
  if (!read(*la, a, &va) || !read(*lb, b, &vb)) return CUDNN_STATUS_EXECUTION_FAILED;
  convolve(call.g, dir, va, vb, &r, call.acc);
  const double al = scale_of(alpha, call.scale_type), be = scale_of(beta, call.scale_type);
  return blend_write(*lo, out, r, al, be) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

}  // namespace

// The input as the matrix an explicit-GEMM convolution multiplies: one row
// per (input channel, filter tap) of a group-0 filter, row-major over the
// taps, and one column per (image, output position) -- stored row by row
// (measured: [C*R*S][N*P*Q]), zero where a tap falls in the padding, and the
// taps flipped for CUDNN_CONVOLUTION.
VGPU_EXPORT cudnnStatus_t cudnnIm2Col(cudnnHandle_t h, const cudnnTensorDescriptor_t xd, const void* x,
                                      const cudnnFilterDescriptor_t wd, const cudnnConvolutionDescriptor_t cd,
                                      void* col) {
  static const char* fn = "cudnnIm2Col";
  const TensorDesc* X = tdesc(xd);
  const FilterDesc* W = fdesc(wd);
  if (!known(h) || !X || !W || !known(cd) || !x || !col) return BAD(fn, "invalid handle, descriptor or pointer");
  const auto* C = as<const ConvDesc>(cd);
  if (X->l.rank != C->nsp + 2 || W->l.rank != X->l.rank) return BAD(fn, "the tensor, filter and convolution ranks differ");
  ConvGeom g;
  const int nsp = C->nsp;
  int64_t out[3];
  for (int i = 0; i < nsp; ++i) out[i] = conv_out(X->l.dims[2 + i], C->pad[i], W->l.dims[2 + i], C->str[i], C->dil[i]);
  // Geometry for one channel group the filter's width: what the GEMM sees.
  Layout yl;
  yl.rank = X->l.rank;
  yl.dims[0] = X->l.dims[0], yl.dims[1] = W->l.dims[0];
  for (int i = 0; i < nsp; ++i) yl.dims[2 + i] = out[i];
  Layout wl = W->l;
  wl.dims[1] = X->l.dims[1];  // all input channels
  std::string why;
  if (!conv_geometry(X->l, wl, yl, nsp, C->pad, C->str, C->dil, C->mode == CUDNN_CONVOLUTION, &g, &why))
    return BAD(fn, why);
  sync_handle(h);
  std::vector<double> vx;
  if (!read(X->l, x, &vx)) return CUDNN_STATUS_EXECUTION_FAILED;
  const int64_t taps = g.flt[0] * g.flt[1] * g.flt[2], positions = g.out[0] * g.out[1] * g.out[2];
  const int64_t cols = g.N * positions, isz = g.in[0] * g.in[1] * g.in[2];
  std::vector<double> m(static_cast<size_t>(g.C * taps * cols), 0.0);
  for (int64_t c = 0; c < g.C; ++c)
    for (int64_t t = 0; t < taps; ++t) {
      const int64_t f[3] = {t / (g.flt[1] * g.flt[2]), (t / g.flt[2]) % g.flt[1], t % g.flt[2]};
      for (int64_t n = 0; n < g.N; ++n)
        for (int64_t o = 0; o < positions; ++o) {
          const int64_t q[3] = {o / (g.out[1] * g.out[2]), (o / g.out[2]) % g.out[1], o % g.out[2]};
          int64_t p[3];
          bool inside = true;
          for (int d = 0; d < 3; ++d) {
            const int64_t tap = g.flip ? g.flt[d] - 1 - f[d] : f[d];
            p[d] = q[d] * g.str[d] - g.pad[d] + tap * g.dil[d];
            inside &= p[d] >= 0 && p[d] < g.in[d];
          }
          if (inside)
            m[static_cast<size_t>((c * taps + t) * cols + n * positions + o)] =
                vx[static_cast<size_t>((n * g.C + c) * isz + (p[0] * g.in[1] + p[1]) * g.in[2] + p[2])];
        }
    }
  Layout ml;
  ml.type = X->l.type;
  ml.rank = 1;
  ml.dims[0] = static_cast<int64_t>(m.size());
  ml.strides[0] = 1;
  return write(ml, col, m) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

// cuDNN reports two more than it lists; programs size their arrays by this.
VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t h, int* count) {
  if (!known(h) || !count) return CUDNN_STATUS_BAD_PARAM;
  *count = kFwdCount + 2;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t h, int* count) {
  if (!known(h) || !count) return CUDNN_STATUS_BAD_PARAM;
  *count = kDataCount + 2;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t h, int* count) {
  if (!known(h) || !count) return CUDNN_STATUS_BAD_PARAM;
  *count = kFilterCount + 2;
  return CUDNN_STATUS_SUCCESS;
}

namespace {
template <class Perf>
cudnnStatus_t enumerate(const char* fn, ConvDir dir, cudnnHandle_t h, const void* xd, const void* wd, const void* cd,
                        const void* yd, int requested, int* returned, Perf* perf, bool timed) {
  if (!known(h) || !returned || !perf || requested < 1) return BAD(fn, "a handle, count or result array is invalid");
  ConvCall call;
  cudnnStatus_t s = conv_check(fn, dir, xd, wd, cd, yd, &call);
  if (s == CUDNN_STATUS_BAD_PARAM) return s;
  const int* order = dir == ConvDir::Forward ? kFwdAlgos : dir == ConvDir::Data ? kDataAlgos : kFilterAlgos;
  const int count = dir == ConvDir::Forward ? kFwdCount : dir == ConvDir::Data ? kDataCount : kFilterCount;
  list_algos(dir, order, count, requested, returned, perf, s == CUDNN_STATUS_SUCCESS,
             static_cast<const ConvDesc*>(cd)->math, timed);
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t workspace(const char* fn, ConvDir dir, const void* xd, const void* wd, const void* cd, const void* yd,
                        int algo, size_t* bytes) {
  if (!bytes) return BAD(fn, "the size pointer is null");
  ConvCall call;
  cudnnStatus_t s = conv_check(fn, dir, xd, wd, cd, yd, &call);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  if (!algo_runs(dir, algo))
    return UNSUPPORTED(fn, std::string(dir_name(dir)) + " algorithm " + std::to_string(algo) + " is not one cuDNN implements");
  *bytes = 0;  // computed on the host
  return CUDNN_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t h, const cudnnTensorDescriptor_t xd, const cudnnFilterDescriptor_t wd,
    const cudnnConvolutionDescriptor_t cd, const cudnnTensorDescriptor_t yd, int requested, int* returned,
    cudnnConvolutionFwdAlgoPerf_t* perf) {
  return enumerate("cudnnGetConvolutionForwardAlgorithm_v7", ConvDir::Forward, h, xd, wd, cd, yd, requested, returned,
                   perf, false);
}
VGPU_EXPORT cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t h, const cudnnTensorDescriptor_t xd, const cudnnFilterDescriptor_t wd,
    const cudnnConvolutionDescriptor_t cd, const cudnnTensorDescriptor_t yd, int requested, int* returned,
    cudnnConvolutionFwdAlgoPerf_t* perf) {
  return enumerate("cudnnFindConvolutionForwardAlgorithm", ConvDir::Forward, h, xd, wd, cd, yd, requested, returned,
                   perf, true);
}
VGPU_EXPORT cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(
    cudnnHandle_t h, const cudnnTensorDescriptor_t xd, const void*, const cudnnFilterDescriptor_t wd, const void*,
    const cudnnConvolutionDescriptor_t cd, const cudnnTensorDescriptor_t yd, void*, int requested, int* returned,
    cudnnConvolutionFwdAlgoPerf_t* perf, void*, size_t) {
  return enumerate("cudnnFindConvolutionForwardAlgorithmEx", ConvDir::Forward, h, xd, wd, cd, yd, requested, returned,
                   perf, true);
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t h, const cudnnFilterDescriptor_t wd, const cudnnTensorDescriptor_t dyd,
    const cudnnConvolutionDescriptor_t cd, const cudnnTensorDescriptor_t dxd, int requested, int* returned,
    cudnnConvolutionBwdDataAlgoPerf_t* perf) {
  return enumerate("cudnnGetConvolutionBackwardDataAlgorithm_v7", ConvDir::Data, h, dxd, wd, cd, dyd, requested,
                   returned, perf, false);
}
VGPU_EXPORT cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(
    cudnnHandle_t h, const cudnnFilterDescriptor_t wd, const cudnnTensorDescriptor_t dyd,
    const cudnnConvolutionDescriptor_t cd, const cudnnTensorDescriptor_t dxd, int requested, int* returned,
    cudnnConvolutionBwdDataAlgoPerf_t* perf) {
  return enumerate("cudnnFindConvolutionBackwardDataAlgorithm", ConvDir::Data, h, dxd, wd, cd, dyd, requested,
                   returned, perf, true);
}
VGPU_EXPORT cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(
    cudnnHandle_t h, const cudnnFilterDescriptor_t wd, const void*, const cudnnTensorDescriptor_t dyd, const void*,
    const cudnnConvolutionDescriptor_t cd, const cudnnTensorDescriptor_t dxd, void*, int requested, int* returned,
    cudnnConvolutionBwdDataAlgoPerf_t* perf, void*, size_t) {
  return enumerate("cudnnFindConvolutionBackwardDataAlgorithmEx", ConvDir::Data, h, dxd, wd, cd, dyd, requested,
                   returned, perf, true);
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t h, const cudnnTensorDescriptor_t xd, const cudnnTensorDescriptor_t dyd,
    const cudnnConvolutionDescriptor_t cd, const cudnnFilterDescriptor_t dwd, int requested, int* returned,
    cudnnConvolutionBwdFilterAlgoPerf_t* perf) {
  return enumerate("cudnnGetConvolutionBackwardFilterAlgorithm_v7", ConvDir::Filter, h, xd, dwd, cd, dyd, requested,
                   returned, perf, false);
}
VGPU_EXPORT cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t h, const cudnnTensorDescriptor_t xd, const cudnnTensorDescriptor_t dyd,
    const cudnnConvolutionDescriptor_t cd, const cudnnFilterDescriptor_t dwd, int requested, int* returned,
    cudnnConvolutionBwdFilterAlgoPerf_t* perf) {
  return enumerate("cudnnFindConvolutionBackwardFilterAlgorithm", ConvDir::Filter, h, xd, dwd, cd, dyd, requested,
                   returned, perf, true);
}
VGPU_EXPORT cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(
    cudnnHandle_t h, const cudnnTensorDescriptor_t xd, const void*, const cudnnTensorDescriptor_t dyd, const void*,
    const cudnnConvolutionDescriptor_t cd, const cudnnFilterDescriptor_t dwd, void*, int requested, int* returned,
    cudnnConvolutionBwdFilterAlgoPerf_t* perf, void*, size_t) {
  return enumerate("cudnnFindConvolutionBackwardFilterAlgorithmEx", ConvDir::Filter, h, xd, dwd, cd, dyd, requested,
                   returned, perf, true);
}

VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t h, const cudnnTensorDescriptor_t xd,
                                                                  const cudnnFilterDescriptor_t wd,
                                                                  const cudnnConvolutionDescriptor_t cd,
                                                                  const cudnnTensorDescriptor_t yd,
                                                                  cudnnConvolutionFwdAlgo_t algo, size_t* bytes) {
  if (!known(h)) return CUDNN_STATUS_BAD_PARAM;
  return workspace("cudnnGetConvolutionForwardWorkspaceSize", ConvDir::Forward, xd, wd, cd, yd, algo, bytes);
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t h, const cudnnFilterDescriptor_t wd,
                                                                       const cudnnTensorDescriptor_t dyd,
                                                                       const cudnnConvolutionDescriptor_t cd,
                                                                       const cudnnTensorDescriptor_t dxd,
                                                                       cudnnConvolutionBwdDataAlgo_t algo,
                                                                       size_t* bytes) {
  if (!known(h)) return CUDNN_STATUS_BAD_PARAM;
  return workspace("cudnnGetConvolutionBackwardDataWorkspaceSize", ConvDir::Data, dxd, wd, cd, dyd, algo, bytes);
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t h, const cudnnTensorDescriptor_t xd, const cudnnTensorDescriptor_t dyd,
    const cudnnConvolutionDescriptor_t cd, const cudnnFilterDescriptor_t dwd, cudnnConvolutionBwdFilterAlgo_t algo,
    size_t* bytes) {
  if (!known(h)) return CUDNN_STATUS_BAD_PARAM;
  return workspace("cudnnGetConvolutionBackwardFilterWorkspaceSize", ConvDir::Filter, xd, dwd, cd, dyd, algo, bytes);
}

/* ---- convolution: the operations ---- */

VGPU_EXPORT cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t h, const void* alpha, const cudnnTensorDescriptor_t xd,
                                                  const void* x, const cudnnFilterDescriptor_t wd, const void* w,
                                                  const cudnnConvolutionDescriptor_t cd, cudnnConvolutionFwdAlgo_t algo,
                                                  void*, size_t, const void* beta, const cudnnTensorDescriptor_t yd,
                                                  void* y) {
  static const char* fn = "cudnnConvolutionForward";
  if (!known(h)) return BAD(fn, "invalid handle");
  ConvCall call;
  cudnnStatus_t s = conv_check(fn, ConvDir::Forward, xd, wd, cd, yd, &call);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  return run_conv(fn, ConvDir::Forward, h, algo, alpha, x, w, beta, y, call);
}

VGPU_EXPORT cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t h, const void* alpha,
                                                       const cudnnFilterDescriptor_t wd, const void* w,
                                                       const cudnnTensorDescriptor_t dyd, const void* dy,
                                                       const cudnnConvolutionDescriptor_t cd,
                                                       cudnnConvolutionBwdDataAlgo_t algo, void*, size_t,
                                                       const void* beta, const cudnnTensorDescriptor_t dxd, void* dx) {
  static const char* fn = "cudnnConvolutionBackwardData";
  if (!known(h)) return BAD(fn, "invalid handle");
  ConvCall call;
  cudnnStatus_t s = conv_check(fn, ConvDir::Data, dxd, wd, cd, dyd, &call);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  return run_conv(fn, ConvDir::Data, h, algo, alpha, dy, w, beta, dx, call);
}

VGPU_EXPORT cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t h, const void* alpha,
                                                         const cudnnTensorDescriptor_t xd, const void* x,
                                                         const cudnnTensorDescriptor_t dyd, const void* dy,
                                                         const cudnnConvolutionDescriptor_t cd,
                                                         cudnnConvolutionBwdFilterAlgo_t algo, void*, size_t,
                                                         const void* beta, const cudnnFilterDescriptor_t dwd,
                                                         void* dw) {
  static const char* fn = "cudnnConvolutionBackwardFilter";
  if (!known(h)) return BAD(fn, "invalid handle");
  ConvCall call;
  cudnnStatus_t s = conv_check(fn, ConvDir::Filter, xd, dwd, cd, dyd, &call);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  return run_conv(fn, ConvDir::Filter, h, algo, alpha, dy, x, beta, dw, call);
}

// db[k] = alpha * (sum of dy over the batch and every spatial position) + beta * db[k].
VGPU_EXPORT cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t h, const void* alpha,
                                                       const cudnnTensorDescriptor_t dyd, const void* dy,
                                                       const void* beta, const cudnnTensorDescriptor_t dbd, void* db) {
  static const char* fn = "cudnnConvolutionBackwardBias";
  const TensorDesc* Y = tdesc(dyd);
  const TensorDesc* B = tdesc(dbd);
  if (!known(h) || !Y || !B || !alpha || !beta || !dy || !db) return BAD(fn, "invalid handle, descriptor or pointer");
  bool shape = Y->l.rank == B->l.rank && Y->l.rank >= 2 && B->l.dims[1] == Y->l.dims[1] && B->l.type == Y->l.type;
  for (int i = 0; shape && i < B->l.rank; ++i)
    if (i != 1 && B->l.dims[i] != 1) shape = false;
  if (!shape) return BAD(fn, "the bias must be [1, C, 1, ...] of dy's rank, channels and type");
  if (!floating(Y->l.type)) return UNSUPPORTED(fn, std::string("data type ") + type_name(Y->l.type));
  sync_handle(h);
  std::vector<double> vy;
  if (!read(Y->l, dy, &vy)) return CUDNN_STATUS_EXECUTION_FAILED;
  int64_t N, C, S;
  ncs(Y->l, &N, &C, &S);
  std::vector<double> r(static_cast<size_t>(C), 0.0);
  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < C; ++c)
      for (int64_t s = 0; s < S; ++s) r[c] += vy[(n * C + c) * S + s];
  return blend_write(B->l, db, r, sc(alpha, B->l), sc(beta, B->l)) ? CUDNN_STATUS_SUCCESS
                                                                     : CUDNN_STATUS_EXECUTION_FAILED;
}

/* ---- activation ---- */

VGPU_EXPORT cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnActivationDescriptor_t>(track(new ActDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete as<ActDesc>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t d, cudnnActivationMode_t mode,
                                                       cudnnNanPropagation_t nan, double coef) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  auto* a = as<ActDesc>(d);
  a->mode = mode, a->nan = nan, a->coef = coef;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t d, cudnnActivationMode_t* mode,
                                                       cudnnNanPropagation_t* nan, double* coef) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const auto* a = as<const ActDesc>(d);
  if (mode) *mode = a->mode;
  if (nan) *nan = a->nan;
  if (coef) *coef = a->coef;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t d, double beta) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  as<ActDesc>(d)->swish_beta = beta;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t d, double* beta) {
  if (!known(d) || !beta) return CUDNN_STATUS_BAD_PARAM;
  *beta = as<ActDesc>(d)->swish_beta;
  return CUDNN_STATUS_SUCCESS;
}

namespace {
double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

// The activation itself; false for a mode with no forward function here.
bool activate(const ActDesc& a, double v, double* r) {
  switch (a.mode) {
    case CUDNN_ACTIVATION_SIGMOID: *r = sigmoid(v); return true;
    // coef is the ELU alpha and the CLIPPED_RELU ceiling; plain RELU ignores it.
    case CUDNN_ACTIVATION_RELU: *r = v > 0.0 ? v : 0.0; return true;
    case CUDNN_ACTIVATION_TANH: *r = std::tanh(v); return true;
    case CUDNN_ACTIVATION_CLIPPED_RELU: *r = std::min(std::max(v, 0.0), a.coef); return true;
    case CUDNN_ACTIVATION_ELU: *r = v > 0.0 ? v : a.coef * (std::exp(v) - 1.0); return true;
    case CUDNN_ACTIVATION_SWISH: *r = v * sigmoid(a.swish_beta * v); return true;
    case CUDNN_ACTIVATION_IDENTITY: *r = v; return true;
    default: return false;
  }
}
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnActivationForward(cudnnHandle_t h, cudnnActivationDescriptor_t ad, const void* alpha,
                                                 const cudnnTensorDescriptor_t xd, const void* x, const void* beta,
                                                 const cudnnTensorDescriptor_t yd, void* y) {
  static const char* fn = "cudnnActivationForward";
  const TensorDesc* X = tdesc(xd);
  const TensorDesc* Y = tdesc(yd);
  if (!known(h) || !known(ad) || !X || !Y || !alpha || !beta || !x || !y)
    return BAD(fn, "invalid handle, descriptor or pointer");
  const ActDesc& A = *as<const ActDesc>(ad);
  // Hardware rejects IDENTITY here: real cuDNN only accepts it through
  // cudnnConvolutionBiasActivationForward, and returns BAD_PARAM from this
  // entry point. Matching that keeps callers' fallbacks intact.
  if (A.mode == CUDNN_ACTIVATION_IDENTITY) return BAD(fn, "CUDNN_ACTIVATION_IDENTITY is only for the fused convolution");
  if (!X->l.same_dims(Y->l)) return BAD(fn, "x and y differ in shape");
  if (!floating(X->l.type) || !floating(Y->l.type)) return UNSUPPORTED(fn, "non-floating-point data");
  double r;
  if (!activate(A, 0.0, &r)) return UNSUPPORTED(fn, "activation mode " + std::to_string(A.mode));
  sync_handle(h);
  std::vector<double> v;
  if (!read(X->l, x, &v)) return CUDNN_STATUS_EXECUTION_FAILED;
  for (double& e : v) activate(A, e, &e);
  return blend_write(Y->l, y, v, sc(alpha, Y->l), sc(beta, Y->l)) ? CUDNN_STATUS_SUCCESS
                                                                   : CUDNN_STATUS_EXECUTION_FAILED;
}

// dx = alpha * f'(x) dy + beta * dx. Which of x and y each mode's gradient
// reads is the hardware's choice, measured: RELU, CLIPPED_RELU and SWISH read
// x (RELU passes dy where x > 0, CLIPPED_RELU where 0 < x <= coef); SIGMOID,
// TANH and ELU read y (y(1 - y), 1 - y^2, and dy where y >= 0 else
// dy (y + coef)).
VGPU_EXPORT cudnnStatus_t cudnnActivationBackward(cudnnHandle_t h, cudnnActivationDescriptor_t ad, const void* alpha,
                                                  const cudnnTensorDescriptor_t yd, const void* y,
                                                  const cudnnTensorDescriptor_t dyd, const void* dy,
                                                  const cudnnTensorDescriptor_t xd, const void* x, const void* beta,
                                                  const cudnnTensorDescriptor_t dxd, void* dx) {
  static const char* fn = "cudnnActivationBackward";
  const TensorDesc *Y = tdesc(yd), *DY = tdesc(dyd), *X = tdesc(xd), *DX = tdesc(dxd);
  if (!known(h) || !known(ad) || !Y || !DY || !X || !DX || !alpha || !beta || !dy || !dx)
    return BAD(fn, "invalid handle, descriptor or pointer");
  const ActDesc& A = *as<const ActDesc>(ad);
  if (A.mode == CUDNN_ACTIVATION_IDENTITY) return BAD(fn, "CUDNN_ACTIVATION_IDENTITY is only for the fused convolution");
  if (!Y->l.same_dims(DY->l) || !Y->l.same_dims(X->l) || !Y->l.same_dims(DX->l))
    return BAD(fn, "y, dy, x and dx differ in shape");
  for (const TensorDesc* t : {Y, DY, X, DX})
    if (!floating(t->l.type)) return UNSUPPORTED(fn, "non-floating-point data");
  const bool reads_x = A.mode == CUDNN_ACTIVATION_RELU || A.mode == CUDNN_ACTIVATION_CLIPPED_RELU ||
                       A.mode == CUDNN_ACTIVATION_SWISH;
  const bool reads_y = A.mode == CUDNN_ACTIVATION_SIGMOID || A.mode == CUDNN_ACTIVATION_TANH ||
                       A.mode == CUDNN_ACTIVATION_ELU;
  if (!reads_x && !reads_y) return UNSUPPORTED(fn, "activation mode " + std::to_string(A.mode));
  if ((reads_x && !x) || (reads_y && !y)) return BAD(fn, "the operand this mode's gradient reads is null");
  sync_handle(h);
  std::vector<double> vin, g;
  if (!read(reads_x ? X->l : Y->l, reads_x ? x : y, &vin) || !read(DY->l, dy, &g)) return CUDNN_STATUS_EXECUTION_FAILED;
  for (size_t i = 0; i < g.size(); ++i) {
    const double v = vin[i];
    switch (A.mode) {
      case CUDNN_ACTIVATION_RELU: g[i] = v > 0.0 ? g[i] : 0.0; break;
      case CUDNN_ACTIVATION_CLIPPED_RELU: g[i] = v > 0.0 && v <= A.coef ? g[i] : 0.0; break;
      case CUDNN_ACTIVATION_SWISH: {
        const double s = sigmoid(A.swish_beta * v);
        g[i] *= s + A.swish_beta * v * s * (1.0 - s);
        break;
      }
      case CUDNN_ACTIVATION_SIGMOID: g[i] *= v * (1.0 - v); break;
      case CUDNN_ACTIVATION_TANH: g[i] *= 1.0 - v * v; break;
      case CUDNN_ACTIVATION_ELU: g[i] = v >= 0.0 ? g[i] : g[i] * (v + A.coef); break;
      default: break;
    }
  }
  return blend_write(DX->l, dx, g, sc(alpha, DX->l), sc(beta, DX->l)) ? CUDNN_STATUS_SUCCESS
                                                                      : CUDNN_STATUS_EXECUTION_FAILED;
}

// y = act(alpha1 * conv(x, w) + alpha2 * z + bias), with act RELU or IDENTITY
// (the only two cuDNN fuses; the others are NOT_SUPPORTED, as on the hardware).
VGPU_EXPORT cudnnStatus_t cudnnConvolutionBiasActivationForward(
    cudnnHandle_t h, const void* alpha1, const cudnnTensorDescriptor_t xd, const void* x,
    const cudnnFilterDescriptor_t wd, const void* w, const cudnnConvolutionDescriptor_t cd,
    cudnnConvolutionFwdAlgo_t algo, void*, size_t, const void* alpha2, const cudnnTensorDescriptor_t zd, const void* z,
    const cudnnTensorDescriptor_t bd, const void* bias, const cudnnActivationDescriptor_t ad,
    const cudnnTensorDescriptor_t yd, void* y) {
  static const char* fn = "cudnnConvolutionBiasActivationForward";
  if (!known(h) || !known(ad)) return BAD(fn, "invalid handle or activation descriptor");
  ConvCall call;
  cudnnStatus_t s = conv_check(fn, ConvDir::Forward, xd, wd, cd, yd, &call);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  const TensorDesc *Z = tdesc(zd), *B = tdesc(bd);
  if (!Z || !B || !alpha1 || !alpha2 || !x || !w || !z || !bias || !y)
    return BAD(fn, "a descriptor or pointer is null");
  if (!Z->l.same_dims(*call.y)) return BAD(fn, "z differs from y in shape");
  bool bshape = B->l.rank == call.y->rank && B->l.dims[1] == call.y->dims[1];
  for (int i = 0; bshape && i < B->l.rank; ++i)
    if (i != 1 && B->l.dims[i] != 1) bshape = false;
  if (!bshape) return BAD(fn, "the bias must be [1, K, 1, ...]");
  const ActDesc& A = *as<const ActDesc>(ad);
  if (A.mode != CUDNN_ACTIVATION_RELU && A.mode != CUDNN_ACTIVATION_IDENTITY)
    return UNSUPPORTED(fn, "only RELU and IDENTITY activations are fused");
  if (!algo_runs(ConvDir::Forward, algo)) return UNSUPPORTED(fn, "forward algorithm " + std::to_string(algo));
  sync_handle(h);
  std::vector<double> vx, vw, vz, vb, r;
  if (!read(*call.x, x, &vx) || !read(*call.w, w, &vw) || !read(Z->l, z, &vz) || !read(B->l, bias, &vb))
    return CUDNN_STATUS_EXECUTION_FAILED;
  convolve(call.g, ConvDir::Forward, vx, vw, &r, call.acc);
  const double a1 = scale_of(alpha1, call.scale_type), a2 = scale_of(alpha2, call.scale_type);
  int64_t N, K, S;
  ncs(*call.y, &N, &K, &S);
  for (int64_t n = 0; n < N; ++n)
    for (int64_t k = 0; k < K; ++k)
      for (int64_t i = 0; i < S; ++i) {
        const size_t at = static_cast<size_t>((n * K + k) * S + i);
        double v = a1 * r[at] + (a2 != 0.0 ? a2 * vz[at] : 0.0) + vb[k];
        if (A.mode == CUDNN_ACTIVATION_RELU && !(v > 0.0)) v = 0.0;
        r[at] = v;
      }
  return write(*call.y, y, r) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

/* ---- pooling ---- */

VGPU_EXPORT cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnPoolingDescriptor_t>(track(new PoolDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete as<PoolDesc>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t d, const cudnnPoolingMode_t mode,
                                                      const cudnnNanPropagation_t nan, int nb, const int win[],
                                                      const int pad[], const int str[]) {
  static const char* fn = "cudnnSetPoolingNdDescriptor";
  if (!known(d) || !win || !pad || !str || nb < 1) return CUDNN_STATUS_BAD_PARAM;
  if (nb > 3) return UNSUPPORTED(fn, "more than 3 pooled dimensions");
  if (mode != CUDNN_POOLING_MAX && mode != CUDNN_POOLING_MAX_DETERMINISTIC &&
      mode != CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING && mode != CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
    return BAD(fn, "unknown pooling mode");
  PoolDesc p;
  p.mode = mode, p.nan = nan, p.nsp = nb;
  for (int i = 0; i < nb; ++i) {
    if (win[i] <= 0 || str[i] <= 0 || pad[i] < 0) return BAD(fn, "a window, stride or padding is out of range");
    p.win[i] = win[i], p.pad[i] = pad[i], p.str[i] = str[i];
  }
  *as<PoolDesc>(d) = p;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t d, cudnnPoolingMode_t mode,
                                                      cudnnNanPropagation_t nan, int wh, int ww, int ph, int pw,
                                                      int sh, int sw) {
  const int win[2] = {wh, ww}, pad[2] = {ph, pw}, str[2] = {sh, sw};
  return cudnnSetPoolingNdDescriptor(d, mode, nan, 2, win, pad, str);
}
VGPU_EXPORT cudnnStatus_t cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t d, cudnnPoolingMode_t* mode,
                                                      cudnnNanPropagation_t* nan, int* wh, int* ww, int* ph, int* pw,
                                                      int* sh, int* sw) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const auto* p = as<const PoolDesc>(d);
  if (mode) *mode = p->mode;
  if (nan) *nan = p->nan;
  if (wh) *wh = p->win[0];
  if (ww) *ww = p->win[1];
  if (ph) *ph = p->pad[0];
  if (pw) *pw = p->pad[1];
  if (sh) *sh = p->str[0];
  if (sw) *sw = p->str[1];
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t d, int requested,
                                                      cudnnPoolingMode_t* mode, cudnnNanPropagation_t* nan, int* nb,
                                                      int win[], int pad[], int str[]) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const auto* p = as<const PoolDesc>(d);
  if (mode) *mode = p->mode;
  if (nan) *nan = p->nan;
  if (nb) *nb = p->nsp;
  for (int i = 0; i < std::min(requested, p->nsp); ++i) {
    if (win) win[i] = p->win[i];
    if (pad) pad[i] = p->pad[i];
    if (str) str[i] = p->str[i];
  }
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t pd,
                                                            const cudnnTensorDescriptor_t xd, int nb, int out[]) {
  const TensorDesc* X = tdesc(xd);
  if (!known(pd) || !X || !out) return CUDNN_STATUS_BAD_PARAM;
  const auto* p = as<const PoolDesc>(pd);
  if (X->l.rank != p->nsp + 2 || nb < X->l.rank)
    return BAD("cudnnGetPoolingNdForwardOutputDim", "the tensor's rank is not the window's plus two");
  out[0] = (int)X->l.dims[0];
  out[1] = (int)X->l.dims[1];
  for (int i = 0; i < p->nsp; ++i) out[2 + i] = (int)(1 + (X->l.dims[2 + i] + 2 * p->pad[i] - p->win[i]) / p->str[i]);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t pd,
                                                            const cudnnTensorDescriptor_t xd, int* n, int* c, int* h,
                                                            int* w) {
  int out[4];
  cudnnStatus_t s = cudnnGetPoolingNdForwardOutputDim(pd, xd, 4, out);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  if (n) *n = out[0];
  if (c) *c = out[1];
  if (h) *h = out[2];
  if (w) *w = out[3];
  return CUDNN_STATUS_SUCCESS;
}

namespace {
// The pooling windows of x (logical order) onto y: for every (n, c, output
// position), the input positions of the window that fall inside x, in
// row-major window order, and how many window taps there are in all.
struct PoolGeom {
  int64_t NC = 0, in[3] = {1, 1, 1}, out[3] = {1, 1, 1}, win[3] = {1, 1, 1}, pad[3] = {0, 0, 0}, str[3] = {1, 1, 1};
  int64_t isz() const { return in[0] * in[1] * in[2]; }
  int64_t osz() const { return out[0] * out[1] * out[2]; }
  int64_t taps() const { return win[0] * win[1] * win[2]; }
  template <class F>
  void window(int64_t o, F&& f) const {  // f(input offset within the plane)
    const int64_t o0 = o / (out[1] * out[2]), o1 = (o / out[2]) % out[1], o2 = o % out[2];
    for (int64_t i = 0; i < win[0]; ++i) {
      const int64_t p0 = o0 * str[0] - pad[0] + i;
      if (p0 < 0 || p0 >= in[0]) continue;
      for (int64_t j = 0; j < win[1]; ++j) {
        const int64_t p1 = o1 * str[1] - pad[1] + j;
        if (p1 < 0 || p1 >= in[1]) continue;
        for (int64_t k = 0; k < win[2]; ++k) {
          const int64_t p2 = o2 * str[2] - pad[2] + k;
          if (p2 < 0 || p2 >= in[2]) continue;
          f((p0 * in[1] + p1) * in[2] + p2);
        }
      }
    }
  }
};

cudnnStatus_t pool_geometry(const char* fn, const PoolDesc& p, const Layout& x, const Layout& y, PoolGeom* g) {
  if (x.rank != p.nsp + 2 || y.rank != x.rank) return BAD(fn, "the tensors' rank is not the window's plus two");
  if (x.dims[0] != y.dims[0] || x.dims[1] != y.dims[1]) return BAD(fn, "x and y differ in batch or channels");
  g->NC = x.dims[0] * x.dims[1];
  for (int i = 0; i < p.nsp; ++i) {
    const int k = 3 - p.nsp + i;
    g->in[k] = x.dims[2 + i], g->out[k] = y.dims[2 + i];
    g->win[k] = p.win[i], g->pad[k] = p.pad[i], g->str[k] = p.str[i];
    if (g->out[k] != 1 + (g->in[k] + 2 * g->pad[k] - g->win[k]) / g->str[k])
      return BAD(fn, "y's extent in pooled dimension " + std::to_string(i) + " is not what the window gives");
  }
  return CUDNN_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnPoolingForward(cudnnHandle_t h, const cudnnPoolingDescriptor_t pd, const void* alpha,
                                              const cudnnTensorDescriptor_t xd, const void* x, const void* beta,
                                              const cudnnTensorDescriptor_t yd, void* y) {
  static const char* fn = "cudnnPoolingForward";
  const TensorDesc *X = tdesc(xd), *Y = tdesc(yd);
  if (!known(h) || !known(pd) || !X || !Y || !alpha || !beta || !x || !y)
    return BAD(fn, "invalid handle, descriptor or pointer");
  const PoolDesc& P = *as<const PoolDesc>(pd);
  PoolGeom g;
  cudnnStatus_t s = pool_geometry(fn, P, X->l, Y->l, &g);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  if (!floating(X->l.type) || !floating(Y->l.type)) return UNSUPPORTED(fn, "non-floating-point data");
  sync_handle(h);
  std::vector<double> vx;
  if (!read(X->l, x, &vx)) return CUDNN_STATUS_EXECUTION_FAILED;
  std::vector<double> r(static_cast<size_t>(g.NC * g.osz()));
  const bool is_max = P.mode == CUDNN_POOLING_MAX || P.mode == CUDNN_POOLING_MAX_DETERMINISTIC;
  for (int64_t nc = 0; nc < g.NC; ++nc) {
    const double* plane = vx.data() + nc * g.isz();
    for (int64_t o = 0; o < g.osz(); ++o) {
      double best = -INFINITY, sum = 0.0;
      int64_t count = 0;
      g.window(o, [&](int64_t at) { best = std::max(best, plane[at]); sum += plane[at]; ++count; });
      double v;
      if (is_max) v = best;
      else if (P.mode == CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) v = sum / static_cast<double>(g.taps());
      else v = count ? sum / static_cast<double>(count) : 0.0;
      r[static_cast<size_t>(nc * g.osz() + o)] = v;
    }
  }
  return blend_write(Y->l, y, r, sc(alpha, Y->l), sc(beta, Y->l)) ? CUDNN_STATUS_SUCCESS
                                                                   : CUDNN_STATUS_EXECUTION_FAILED;
}

// dx = alpha * (dy routed back through each window) + beta * dx. Measured:
// MAX gives each window's dy to its first largest x (row-major window order),
// recomputing it from x; MAX_DETERMINISTIC to the first x equal to y; the
// averages spread dy over the window's inputs, divided as the forward pass
// divided.
VGPU_EXPORT cudnnStatus_t cudnnPoolingBackward(cudnnHandle_t h, const cudnnPoolingDescriptor_t pd, const void* alpha,
                                               const cudnnTensorDescriptor_t yd, const void* y,
                                               const cudnnTensorDescriptor_t dyd, const void* dy,
                                               const cudnnTensorDescriptor_t xd, const void* x, const void* beta,
                                               const cudnnTensorDescriptor_t dxd, void* dx) {
  static const char* fn = "cudnnPoolingBackward";
  const TensorDesc *Y = tdesc(yd), *DY = tdesc(dyd), *X = tdesc(xd), *DX = tdesc(dxd);
  if (!known(h) || !known(pd) || !Y || !DY || !X || !DX || !alpha || !beta || !dy || !dx)
    return BAD(fn, "invalid handle, descriptor or pointer");
  const PoolDesc& P = *as<const PoolDesc>(pd);
  if (!Y->l.same_dims(DY->l) || !X->l.same_dims(DX->l)) return BAD(fn, "y and dy, or x and dx, differ in shape");
  PoolGeom g;
  cudnnStatus_t s = pool_geometry(fn, P, X->l, Y->l, &g);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  for (const TensorDesc* t : {Y, DY, X, DX})
    if (!floating(t->l.type)) return UNSUPPORTED(fn, "non-floating-point data");
  const bool is_max = P.mode == CUDNN_POOLING_MAX || P.mode == CUDNN_POOLING_MAX_DETERMINISTIC;
  if (is_max && !x) return BAD(fn, "max pooling's gradient reads x, which is null");
  if (P.mode == CUDNN_POOLING_MAX_DETERMINISTIC && !y) return BAD(fn, "y is null");
  sync_handle(h);
  std::vector<double> vx, vy, vdy;
  if (!read(DY->l, dy, &vdy)) return CUDNN_STATUS_EXECUTION_FAILED;
  if (is_max && !read(X->l, x, &vx)) return CUDNN_STATUS_EXECUTION_FAILED;
  if (P.mode == CUDNN_POOLING_MAX_DETERMINISTIC && !read(Y->l, y, &vy)) return CUDNN_STATUS_EXECUTION_FAILED;
  std::vector<double> r(static_cast<size_t>(g.NC * g.isz()), 0.0);
  for (int64_t nc = 0; nc < g.NC; ++nc) {
    double* out = r.data() + nc * g.isz();
    for (int64_t o = 0; o < g.osz(); ++o) {
      const size_t yo = static_cast<size_t>(nc * g.osz() + o);
      const double grad = vdy[yo];
      if (P.mode == CUDNN_POOLING_MAX) {
        const double* plane = vx.data() + nc * g.isz();
        int64_t arg = -1;
        double best = 0.0;
        g.window(o, [&](int64_t at) { if (arg < 0 || plane[at] > best) arg = at, best = plane[at]; });
        if (arg >= 0) out[arg] += grad;
      } else if (P.mode == CUDNN_POOLING_MAX_DETERMINISTIC) {
        const double* plane = vx.data() + nc * g.isz();
        int64_t arg = -1;
        g.window(o, [&](int64_t at) { if (arg < 0 && plane[at] == vy[yo]) arg = at; });
        if (arg >= 0) out[arg] += grad;
      } else {
        int64_t count = 0;
        g.window(o, [&](int64_t) { ++count; });
        const double div = P.mode == CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING ? (double)g.taps() : (double)count;
        if (count) g.window(o, [&](int64_t at) { out[at] += grad / div; });
      }
    }
  }
  return blend_write(DX->l, dx, r, sc(alpha, DX->l), sc(beta, DX->l)) ? CUDNN_STATUS_SUCCESS
                                                                      : CUDNN_STATUS_EXECUTION_FAILED;
}

/* ---- softmax ---- */

namespace {
// The groups softmax normalizes over, as lists of logical indexes: per n
// (INSTANCE) or per (n, spatial position) over C (CHANNEL).
template <class F>
void softmax_groups(const Layout& t, cudnnSoftmaxMode_t mode, F&& f) {
  int64_t N, C, S;
  ncs(t, &N, &C, &S);
  std::vector<size_t> idx;
  if (mode == CUDNN_SOFTMAX_MODE_INSTANCE) {
    for (int64_t n = 0; n < N; ++n) {
      idx.clear();
      for (int64_t i = 0; i < C * S; ++i) idx.push_back(static_cast<size_t>(n * C * S + i));
      f(idx);
    }
  } else {
    for (int64_t n = 0; n < N; ++n)
      for (int64_t s = 0; s < S; ++s) {
        idx.clear();
        for (int64_t c = 0; c < C; ++c) idx.push_back(static_cast<size_t>((n * C + c) * S + s));
        f(idx);
      }
  }
}
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t h, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
                                              const void* alpha, const cudnnTensorDescriptor_t xd, const void* x,
                                              const void* beta, const cudnnTensorDescriptor_t yd, void* y) {
  static const char* fn = "cudnnSoftmaxForward";
  const TensorDesc *X = tdesc(xd), *Y = tdesc(yd);
  if (!known(h) || !X || !Y || !alpha || !beta || !x || !y) return BAD(fn, "invalid handle, descriptor or pointer");
  if (!X->l.same_dims(Y->l)) return BAD(fn, "x and y differ in shape");
  if (algo < CUDNN_SOFTMAX_FAST || algo > CUDNN_SOFTMAX_LOG || mode < 0 || mode > CUDNN_SOFTMAX_MODE_CHANNEL)
    return BAD(fn, "unknown algorithm or mode");
  if (!floating(X->l.type) || !floating(Y->l.type)) return UNSUPPORTED(fn, "non-floating-point data");
  sync_handle(h);
  std::vector<double> v;
  if (!read(X->l, x, &v)) return CUDNN_STATUS_EXECUTION_FAILED;
  std::vector<double> r(v.size());
  softmax_groups(X->l, mode, [&](const std::vector<size_t>& idx) {
    double mx = -INFINITY, sum = 0.0;
    for (size_t i : idx) mx = std::max(mx, v[i]);
    for (size_t i : idx) sum += std::exp(v[i] - mx);
    for (size_t i : idx) r[i] = algo == CUDNN_SOFTMAX_LOG ? v[i] - mx - std::log(sum) : std::exp(v[i] - mx) / sum;
  });
  return blend_write(Y->l, y, r, sc(alpha, Y->l), sc(beta, Y->l)) ? CUDNN_STATUS_SUCCESS
                                                                   : CUDNN_STATUS_EXECUTION_FAILED;
}

// dx = alpha * y (dy - sum(y dy)) + beta * dx; for LOG, dy - exp(y) sum(dy).
VGPU_EXPORT cudnnStatus_t cudnnSoftmaxBackward(cudnnHandle_t h, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
                                               const void* alpha, const cudnnTensorDescriptor_t yd, const void* y,
                                               const cudnnTensorDescriptor_t dyd, const void* dy, const void* beta,
                                               const cudnnTensorDescriptor_t dxd, void* dx) {
  static const char* fn = "cudnnSoftmaxBackward";
  const TensorDesc *Y = tdesc(yd), *DY = tdesc(dyd), *DX = tdesc(dxd);
  if (!known(h) || !Y || !DY || !DX || !alpha || !beta || !y || !dy || !dx)
    return BAD(fn, "invalid handle, descriptor or pointer");
  if (!Y->l.same_dims(DY->l) || !Y->l.same_dims(DX->l)) return BAD(fn, "y, dy and dx differ in shape");
  if (algo < CUDNN_SOFTMAX_FAST || algo > CUDNN_SOFTMAX_LOG || mode < 0 || mode > CUDNN_SOFTMAX_MODE_CHANNEL)
    return BAD(fn, "unknown algorithm or mode");
  for (const TensorDesc* t : {Y, DY, DX})
    if (!floating(t->l.type)) return UNSUPPORTED(fn, "non-floating-point data");
  sync_handle(h);
  std::vector<double> vy, vdy;
  if (!read(Y->l, y, &vy) || !read(DY->l, dy, &vdy)) return CUDNN_STATUS_EXECUTION_FAILED;
  std::vector<double> r(vy.size());
  softmax_groups(Y->l, mode, [&](const std::vector<size_t>& idx) {
    double sum = 0.0;
    if (algo == CUDNN_SOFTMAX_LOG) {
      for (size_t i : idx) sum += vdy[i];
      for (size_t i : idx) r[i] = vdy[i] - std::exp(vy[i]) * sum;
    } else {
      for (size_t i : idx) sum += vy[i] * vdy[i];
      for (size_t i : idx) r[i] = vy[i] * (vdy[i] - sum);
    }
  });
  return blend_write(DX->l, dx, r, sc(alpha, DX->l), sc(beta, DX->l)) ? CUDNN_STATUS_SUCCESS
                                                                      : CUDNN_STATUS_EXECUTION_FAILED;
}

/* ---- local response normalization (across channels) ---- */

VGPU_EXPORT cudnnStatus_t cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnLRNDescriptor_t>(track(new LrnDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete as<LrnDesc>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetLRNDescriptor(cudnnLRNDescriptor_t d, unsigned n, double alpha, double beta,
                                                double k) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  if (n < CUDNN_LRN_MIN_N || n > CUDNN_LRN_MAX_N || k < CUDNN_LRN_MIN_K || beta < CUDNN_LRN_MIN_BETA)
    return BAD("cudnnSetLRNDescriptor", "n, k or beta is out of cuDNN's range");
  *as<LrnDesc>(d) = LrnDesc{n, alpha, beta, k};
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetLRNDescriptor(cudnnLRNDescriptor_t d, unsigned* n, double* alpha, double* beta,
                                                double* k) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const auto* l = as<const LrnDesc>(d);
  if (n) *n = l->n;
  if (alpha) *alpha = l->alpha;
  if (beta) *beta = l->beta;
  if (k) *k = l->k;
  return CUDNN_STATUS_SUCCESS;
}

namespace {
// The channels in channel c's window: c - floor((n-1)/2) to c + ceil((n-1)/2),
// clipped to the tensor (measured: an even n reaches one further up).
inline void lrn_window(const LrnDesc& L, int64_t c, int64_t C, int64_t* lo, int64_t* hi) {
  const int64_t n = L.n;
  *lo = std::max<int64_t>(0, c - (n - 1) / 2);
  *hi = std::min<int64_t>(C - 1, c + n / 2);
}
// s[i] = k + alpha / n * (sum of x^2 over i's window).
std::vector<double> lrn_scale(const LrnDesc& L, const Layout& t, const std::vector<double>& x) {
  int64_t N, C, S;
  ncs(t, &N, &C, &S);
  std::vector<double> s(x.size());
  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < C; ++c) {
      int64_t lo, hi;
      lrn_window(L, c, C, &lo, &hi);
      for (int64_t i = 0; i < S; ++i) {
        double sum = 0.0;
        for (int64_t j = lo; j <= hi; ++j) {
          const double v = x[static_cast<size_t>((n * C + j) * S + i)];
          sum += v * v;
        }
        s[static_cast<size_t>((n * C + c) * S + i)] = L.k + L.alpha / L.n * sum;
      }
    }
  return s;
}
}  // namespace

// y = x * s^-beta, s = k + alpha/n * (sum of x^2 over the channel window).
VGPU_EXPORT cudnnStatus_t cudnnLRNCrossChannelForward(cudnnHandle_t h, cudnnLRNDescriptor_t ld, cudnnLRNMode_t mode,
                                                      const void* alpha, const cudnnTensorDescriptor_t xd,
                                                      const void* x, const void* beta,
                                                      const cudnnTensorDescriptor_t yd, void* y) {
  static const char* fn = "cudnnLRNCrossChannelForward";
  const TensorDesc *X = tdesc(xd), *Y = tdesc(yd);
  if (!known(h) || !known(ld) || !X || !Y || !alpha || !beta || !x || !y)
    return BAD(fn, "invalid handle, descriptor or pointer");
  if (mode != CUDNN_LRN_CROSS_CHANNEL_DIM1) return BAD(fn, "unknown LRN mode");
  if (!X->l.same_dims(Y->l) || X->l.rank < 3) return BAD(fn, "x and y differ in shape");
  if (!floating(X->l.type) || !floating(Y->l.type)) return UNSUPPORTED(fn, "non-floating-point data");
  const LrnDesc& L = *as<const LrnDesc>(ld);
  sync_handle(h);
  std::vector<double> v;
  if (!read(X->l, x, &v)) return CUDNN_STATUS_EXECUTION_FAILED;
  const std::vector<double> s = lrn_scale(L, X->l, v);
  for (size_t i = 0; i < v.size(); ++i) v[i] *= std::pow(s[i], -L.beta);
  return blend_write(Y->l, y, v, sc(alpha, Y->l), sc(beta, Y->l)) ? CUDNN_STATUS_SUCCESS
                                                                   : CUDNN_STATUS_EXECUTION_FAILED;
}

// dx_i = dy_i s_i^-beta - 2 alpha beta / n * x_i * sum over the windows j that
// contain i of dy_j y_j / s_j.
VGPU_EXPORT cudnnStatus_t cudnnLRNCrossChannelBackward(cudnnHandle_t h, cudnnLRNDescriptor_t ld, cudnnLRNMode_t mode,
                                                       const void* alpha, const cudnnTensorDescriptor_t yd,
                                                       const void* y, const cudnnTensorDescriptor_t dyd,
                                                       const void* dy, const cudnnTensorDescriptor_t xd, const void* x,
                                                       const void* beta, const cudnnTensorDescriptor_t dxd, void* dx) {
  static const char* fn = "cudnnLRNCrossChannelBackward";
  const TensorDesc *Y = tdesc(yd), *DY = tdesc(dyd), *X = tdesc(xd), *DX = tdesc(dxd);
  if (!known(h) || !known(ld) || !Y || !DY || !X || !DX || !alpha || !beta || !y || !dy || !x || !dx)
    return BAD(fn, "invalid handle, descriptor or pointer");
  if (mode != CUDNN_LRN_CROSS_CHANNEL_DIM1) return BAD(fn, "unknown LRN mode");
  if (!X->l.same_dims(Y->l) || !X->l.same_dims(DY->l) || !X->l.same_dims(DX->l) || X->l.rank < 3)
    return BAD(fn, "y, dy, x and dx differ in shape");
  for (const TensorDesc* t : {Y, DY, X, DX})
    if (!floating(t->l.type)) return UNSUPPORTED(fn, "non-floating-point data");
  const LrnDesc& L = *as<const LrnDesc>(ld);
  sync_handle(h);
  std::vector<double> vx, vy, vdy;
  if (!read(X->l, x, &vx) || !read(Y->l, y, &vy) || !read(DY->l, dy, &vdy)) return CUDNN_STATUS_EXECUTION_FAILED;
  const std::vector<double> s = lrn_scale(L, X->l, vx);
  int64_t N, C, S;
  ncs(X->l, &N, &C, &S);
  std::vector<double> r(vx.size());
  const double coef = 2.0 * L.alpha * L.beta / L.n;
  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < C; ++c)
      for (int64_t i = 0; i < S; ++i) {
        const size_t at = static_cast<size_t>((n * C + c) * S + i);
        double sum = 0.0;
        // The windows containing c are those of channels j with c in [lo(j), hi(j)].
        for (int64_t j = 0; j < C; ++j) {
          int64_t lo, hi;
          lrn_window(L, j, C, &lo, &hi);
          if (c < lo || c > hi) continue;
          const size_t jt = static_cast<size_t>((n * C + j) * S + i);
          sum += vdy[jt] * vy[jt] / s[jt];
        }
        r[at] = vdy[at] * std::pow(s[at], -L.beta) - coef * vx[at] * sum;
      }
  return blend_write(DX->l, dx, r, sc(alpha, DX->l), sc(beta, DX->l)) ? CUDNN_STATUS_SUCCESS
                                                                      : CUDNN_STATUS_EXECUTION_FAILED;
}

/* ---- tensor arithmetic ---- */

// C = alpha * A + beta * C, A broadcast over any dimension where it has
// extent 1 -- the usual bias add has A shaped [1, C, 1, 1].
VGPU_EXPORT cudnnStatus_t cudnnAddTensor(cudnnHandle_t h, const void* alpha, const cudnnTensorDescriptor_t ad,
                                         const void* A, const void* beta, const cudnnTensorDescriptor_t cd, void* C) {
  static const char* fn = "cudnnAddTensor";
  const TensorDesc *Ad = tdesc(ad), *Cd = tdesc(cd);
  if (!known(h) || !Ad || !Cd || !alpha || !beta || !A || !C) return BAD(fn, "invalid handle, descriptor or pointer");
  if (!broadcastable(Ad->l, Cd->l)) return BAD(fn, "every dimension of A must be C's or 1");
  if (!floating(Ad->l.type) || !floating(Cd->l.type)) return UNSUPPORTED(fn, "non-floating-point data");
  sync_handle(h);
  std::vector<double> va;
  if (!read(Ad->l, A, &va)) return CUDNN_STATUS_EXECUTION_FAILED;
  const std::vector<size_t> bi = broadcast_index(Ad->l, Cd->l);
  std::vector<double> r(bi.size());
  for (size_t i = 0; i < r.size(); ++i) r[i] = va[bi[i]];
  return blend_write(Cd->l, C, r, sc(alpha, Cd->l), sc(beta, Cd->l)) ? CUDNN_STATUS_SUCCESS
                                                                     : CUDNN_STATUS_EXECUTION_FAILED;
}

VGPU_EXPORT cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnOpTensorDescriptor_t>(track(new OpDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete as<OpDesc>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t d, cudnnOpTensorOp_t op,
                                                     cudnnDataType_t type, cudnnNanPropagation_t nan) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  if (op < CUDNN_OP_TENSOR_ADD || op > CUDNN_OP_TENSOR_NOT) return BAD("cudnnSetOpTensorDescriptor", "unknown op");
  *as<OpDesc>(d) = OpDesc{op, type, nan};
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t d, cudnnOpTensorOp_t* op,
                                                     cudnnDataType_t* type, cudnnNanPropagation_t* nan) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const auto* o = as<const OpDesc>(d);
  if (op) *op = o->op;
  if (type) *type = o->type;
  if (nan) *nan = o->nan;
  return CUDNN_STATUS_SUCCESS;
}

// C = op(alpha1 * A, alpha2 * B) + beta * C, B broadcast onto C (A may not
// be: measured, BAD_PARAM). The compute type is FLOAT for float, half and
// bfloat16 data and DOUBLE for double, the only combinations cuDNN runs.
VGPU_EXPORT cudnnStatus_t cudnnOpTensor(cudnnHandle_t h, const cudnnOpTensorDescriptor_t od, const void* alpha1,
                                        const cudnnTensorDescriptor_t ad, const void* A, const void* alpha2,
                                        const cudnnTensorDescriptor_t bd, const void* Bp, const void* beta,
                                        const cudnnTensorDescriptor_t cd, void* C) {
  static const char* fn = "cudnnOpTensor";
  const TensorDesc *Ad = tdesc(ad), *Bd = tdesc(bd), *Cd = tdesc(cd);
  if (!known(h) || !known(od) || !Ad || !Bd || !Cd || !alpha1 || !alpha2 || !beta || !A || !Bp || !C)
    return BAD(fn, "invalid handle, descriptor or pointer");
  const OpDesc& O = *as<const OpDesc>(od);
  if (!Ad->l.same_dims(Cd->l)) return BAD(fn, "A must have C's shape");
  if (!broadcastable(Bd->l, Cd->l)) return BAD(fn, "every dimension of B must be C's or 1");
  for (const TensorDesc* t : {Ad, Bd, Cd}) {
    const cudnnDataType_t want = t->l.type == CUDNN_DATA_DOUBLE ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    if (!floating(t->l.type) || O.type != want)
      return UNSUPPORTED(fn, std::string(type_name(t->l.type)) + " data with compute type " + type_name(O.type));
  }
  const double a1 = sc(alpha1, Cd->l), a2 = sc(alpha2, Cd->l), b = sc(beta, Cd->l);
  sync_handle(h);
  std::vector<double> va, vb;
  if (!read(Ad->l, A, &va) || !read(Bd->l, Bp, &vb)) return CUDNN_STATUS_EXECUTION_FAILED;
  const std::vector<size_t> bi = broadcast_index(Bd->l, Cd->l);
  std::vector<double> r(va.size());
  for (size_t i = 0; i < r.size(); ++i) {
    const double av = a1 * va[i], bv = a2 * vb[bi[i]];
    switch (O.op) {
      case CUDNN_OP_TENSOR_ADD: r[i] = av + bv; break;
      case CUDNN_OP_TENSOR_MUL: r[i] = av * bv; break;
      case CUDNN_OP_TENSOR_MIN: r[i] = std::min(av, bv); break;
      case CUDNN_OP_TENSOR_MAX: r[i] = std::max(av, bv); break;
      case CUDNN_OP_TENSOR_SQRT: r[i] = std::sqrt(av); break;  // unary: B is not read
      case CUDNN_OP_TENSOR_NOT: r[i] = 1.0 - av; break;
      default: return UNSUPPORTED(fn, "op " + std::to_string(O.op));
    }
  }
  return blend_write(Cd->l, C, r, 1.0, b) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

VGPU_EXPORT cudnnStatus_t cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnReduceTensorDescriptor_t>(track(new ReduceDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete as<ReduceDesc>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t d, cudnnReduceTensorOp_t op,
                                                         cudnnDataType_t type, cudnnNanPropagation_t nan,
                                                         cudnnReduceTensorIndices_t indices,
                                                         cudnnIndicesType_t index_type) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  if (op < CUDNN_REDUCE_TENSOR_ADD || op > CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS)
    return BAD("cudnnSetReduceTensorDescriptor", "unknown reduction");
  *as<ReduceDesc>(d) = ReduceDesc{op, type, nan, indices, index_type};
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t d,
                                                         cudnnReduceTensorOp_t* op, cudnnDataType_t* type,
                                                         cudnnNanPropagation_t* nan,
                                                         cudnnReduceTensorIndices_t* indices,
                                                         cudnnIndicesType_t* index_type) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const auto* r = as<const ReduceDesc>(d);
  if (op) *op = r->op;
  if (type) *type = r->type;
  if (nan) *nan = r->nan;
  if (indices) *indices = r->indices;
  if (index_type) *index_type = r->index_type;
  return CUDNN_STATUS_SUCCESS;
}

namespace {
bool returns_indices(const ReduceDesc& r) {
  return r.indices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES &&
         (r.op == CUDNN_REDUCE_TENSOR_MIN || r.op == CUDNN_REDUCE_TENSOR_MAX || r.op == CUDNN_REDUCE_TENSOR_AMAX);
}
// C must have A's rank, and each of its dimensions A's or 1.
cudnnStatus_t reduce_check(const char* fn, const void* rd, const TensorDesc* A, const TensorDesc* C) {
  if (!known(rd) || !A || !C) return BAD(fn, "a descriptor is null, destroyed or never set");
  if (!broadcastable(C->l, A->l)) return BAD(fn, "every dimension of C must be A's or 1");
  const ReduceDesc& R = *static_cast<const ReduceDesc*>(rd);
  if (returns_indices(R) && R.index_type != CUDNN_32BIT_INDICES)
    return UNSUPPORTED(fn, "only 32-bit indices are supported");
  return CUDNN_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnGetReductionIndicesSize(cudnnHandle_t h, const cudnnReduceTensorDescriptor_t rd,
                                                       const cudnnTensorDescriptor_t ad,
                                                       const cudnnTensorDescriptor_t cd, size_t* size) {
  static const char* fn = "cudnnGetReductionIndicesSize";
  if (!known(h) || !size) return CUDNN_STATUS_BAD_PARAM;
  const TensorDesc *A = tdesc(ad), *C = tdesc(cd);
  cudnnStatus_t s = reduce_check(fn, rd, A, C);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  *size = returns_indices(*as<const ReduceDesc>(rd)) ? C->l.count() * sizeof(uint32_t) : 0;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetReductionWorkspaceSize(cudnnHandle_t h, const cudnnReduceTensorDescriptor_t rd,
                                                         const cudnnTensorDescriptor_t ad,
                                                         const cudnnTensorDescriptor_t cd, size_t* size) {
  if (!known(h) || !size) return CUDNN_STATUS_BAD_PARAM;
  cudnnStatus_t s = reduce_check("cudnnGetReductionWorkspaceSize", rd, tdesc(ad), tdesc(cd));
  if (s != CUDNN_STATUS_SUCCESS) return s;
  *size = 0;  // computed on the host
  return CUDNN_STATUS_SUCCESS;
}

// C = alpha * reduce(A) + beta * C over the dimensions where C has extent 1.
// MIN, MAX and AMAX can also give each result's index: its position among the
// elements reduced, row-major over the reduced dimensions, first one on a tie.
VGPU_EXPORT cudnnStatus_t cudnnReduceTensor(cudnnHandle_t h, const cudnnReduceTensorDescriptor_t rd, void* indices,
                                            size_t indices_bytes, void*, size_t, const void* alpha,
                                            const cudnnTensorDescriptor_t ad, const void* A, const void* beta,
                                            const cudnnTensorDescriptor_t cd, void* C) {
  static const char* fn = "cudnnReduceTensor";
  const TensorDesc *Ad = tdesc(ad), *Cd = tdesc(cd);
  if (!known(h) || !alpha || !beta || !A || !C) return BAD(fn, "invalid handle or pointer");
  cudnnStatus_t s = reduce_check(fn, rd, Ad, Cd);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  // Measured: the sizes are answered for any shapes, but the reduction itself
  // refuses one that reduces nothing.
  bool reduces = false;
  for (int i = 0; i < Ad->l.rank; ++i) reduces |= Cd->l.dims[i] != Ad->l.dims[i];
  if (!reduces) return BAD(fn, "C has A's shape: nothing is reduced");
  const ReduceDesc& R = *as<const ReduceDesc>(rd);
  const cudnnDataType_t want = Cd->l.type == CUDNN_DATA_DOUBLE ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
  if (!floating(Ad->l.type) || !floating(Cd->l.type) || R.type != want)
    return UNSUPPORTED(fn, std::string(type_name(Ad->l.type)) + " data with compute type " + type_name(R.type));
  const bool want_idx = returns_indices(R);
  if (want_idx && (!indices || indices_bytes < Cd->l.count() * sizeof(uint32_t)))
    return BAD(fn, "the indices buffer is too small");
  sync_handle(h);
  std::vector<double> va;
  if (!read(Ad->l, A, &va)) return CUDNN_STATUS_EXECUTION_FAILED;
  const size_t nc = Cd->l.count();
  std::vector<double> acc(nc), r(nc);
  std::vector<uint32_t> idx(nc, 0);
  std::vector<int64_t> cnt(nc, 0);
  // Walk A in logical order; each element's output and its position among
  // that output's reduced elements.
  const Layout& L = Ad->l;
  int64_t at[kMaxRank] = {};
  for (size_t i = 0; i < va.size(); ++i) {
    size_t o = 0, pos = 0;
    for (int d = 0; d < L.rank; ++d) {
      const bool reduced = Cd->l.dims[d] == 1 && L.dims[d] != 1;
      o = o * Cd->l.dims[d] + (reduced ? 0 : at[d]);
      if (reduced) pos = pos * L.dims[d] + at[d];
    }
    const double v = va[i];
    const int64_t k = cnt[o]++;
    double& a = acc[o];
    switch (R.op) {
      case CUDNN_REDUCE_TENSOR_ADD:
      case CUDNN_REDUCE_TENSOR_AVG: a = k ? a + v : v; break;
      case CUDNN_REDUCE_TENSOR_MUL: a = k ? a * v : v; break;
      case CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS: if (v != 0.0) a = (k && a != 0.0) ? a * v : (k ? a : v); else if (!k) a = 0.0; break;
      case CUDNN_REDUCE_TENSOR_MIN: if (!k || v < a) a = v, idx[o] = (uint32_t)pos; break;
      case CUDNN_REDUCE_TENSOR_MAX: if (!k || v > a) a = v, idx[o] = (uint32_t)pos; break;
      case CUDNN_REDUCE_TENSOR_AMAX: if (!k || std::fabs(v) > a) a = std::fabs(v), idx[o] = (uint32_t)pos; break;
      case CUDNN_REDUCE_TENSOR_NORM1: a = (k ? a : 0.0) + std::fabs(v); break;
      case CUDNN_REDUCE_TENSOR_NORM2: a = (k ? a : 0.0) + v * v; break;
    }
    for (int d = L.rank; d-- > 0;) {
      if (++at[d] < L.dims[d]) break;
      at[d] = 0;
    }
  }
  for (size_t o = 0; o < nc; ++o) {
    r[o] = acc[o];
    if (R.op == CUDNN_REDUCE_TENSOR_AVG) r[o] /= static_cast<double>(cnt[o]);
    if (R.op == CUDNN_REDUCE_TENSOR_NORM2) r[o] = std::sqrt(r[o]);
  }
  if (!blend_write(Cd->l, C, r, sc(alpha, Cd->l), sc(beta, Cd->l))) return CUDNN_STATUS_EXECUTION_FAILED;
  if (want_idx && cudaMemcpy(indices, idx.data(), nc * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess)
    return CUDNN_STATUS_EXECUTION_FAILED;
  return CUDNN_STATUS_SUCCESS;
}

// y = alpha * x + beta * y, between any two layouts and element types of the
// same shape: how a program converts NCHW to NHWC, or float to half.
VGPU_EXPORT cudnnStatus_t cudnnTransformTensor(cudnnHandle_t h, const void* alpha, const cudnnTensorDescriptor_t xd,
                                               const void* x, const void* beta, const cudnnTensorDescriptor_t yd,
                                               void* y) {
  static const char* fn = "cudnnTransformTensor";
  const TensorDesc *X = tdesc(xd), *Y = tdesc(yd);
  if (!known(h) || !X || !Y || !alpha || !beta || !x || !y) return BAD(fn, "invalid handle, descriptor or pointer");
  if (!X->l.same_dims(Y->l)) return BAD(fn, "x and y differ in shape");
  sync_handle(h);
  std::vector<double> v;
  if (!read(X->l, x, &v)) return CUDNN_STATUS_EXECUTION_FAILED;
  return blend_write(Y->l, y, v, sc(alpha, Y->l), sc(beta, Y->l)) ? CUDNN_STATUS_SUCCESS
                                                                   : CUDNN_STATUS_EXECUTION_FAILED;
}

// Measured: the value is a float for float, half and bfloat16 tensors and a
// double for double ones; integer tensors are NOT_SUPPORTED.
VGPU_EXPORT cudnnStatus_t cudnnSetTensor(cudnnHandle_t h, const cudnnTensorDescriptor_t yd, void* y,
                                         const void* value) {
  static const char* fn = "cudnnSetTensor";
  const TensorDesc* Y = tdesc(yd);
  if (!known(h) || !Y || !y || !value) return BAD(fn, "invalid handle, descriptor or pointer");
  if (!floating(Y->l.type)) return UNSUPPORTED(fn, std::string(type_name(Y->l.type)) + " tensors");
  sync_handle(h);
  std::vector<double> v(Y->l.count(), sc(value, Y->l));
  return write(Y->l, y, v) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}
VGPU_EXPORT cudnnStatus_t cudnnScaleTensor(cudnnHandle_t h, const cudnnTensorDescriptor_t yd, void* y,
                                           const void* alpha) {
  static const char* fn = "cudnnScaleTensor";
  const TensorDesc* Y = tdesc(yd);
  if (!known(h) || !Y || !y || !alpha) return BAD(fn, "invalid handle, descriptor or pointer");
  if (!floating(Y->l.type)) return UNSUPPORTED(fn, std::string(type_name(Y->l.type)) + " tensors");
  sync_handle(h);
  std::vector<double> v;
  if (!read(Y->l, y, &v)) return CUDNN_STATUS_EXECUTION_FAILED;
  const double a = sc(alpha, Y->l);
  for (double& e : v) e *= a;
  return write(Y->l, y, v) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

/* ---- batch normalization ---- */

namespace {
bool bn_spatial(cudnnBatchNormMode_t m) {
  return m == CUDNN_BATCHNORM_SPATIAL || m == CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
}
// The parameter slot each element of x uses: its channel (spatial), or its
// (c, spatial position) (per activation). Logical, packed.
struct BnGeom {
  int64_t N = 0, C = 0, S = 0;
  bool spatial = true;
  size_t params() const { return static_cast<size_t>(spatial ? C : C * S); }
  size_t slot(int64_t c, int64_t s) const { return static_cast<size_t>(spatial ? c : c * S + s); }
};
cudnnStatus_t bn_check(const char* fn, cudnnBatchNormMode_t mode, const TensorDesc* X, const TensorDesc* B,
                       BnGeom* g) {
  if (!X || !B) return BAD(fn, "a descriptor is null, destroyed or never set");
  if (mode != CUDNN_BATCHNORM_SPATIAL && mode != CUDNN_BATCHNORM_SPATIAL_PERSISTENT &&
      mode != CUDNN_BATCHNORM_PER_ACTIVATION)
    return BAD(fn, "unknown batch normalization mode");
  g->spatial = bn_spatial(mode);
  ncs(X->l, &g->N, &g->C, &g->S);
  if (B->l.count() != g->params() || B->l.rank != X->l.rank || B->l.dims[1] != g->C)
    return BAD(fn, "the parameter tensor's shape does not match the mode");
  if (!floating(X->l.type) || !floating(B->l.type)) return UNSUPPORTED(fn, "non-floating-point data");
  return CUDNN_STATUS_SUCCESS;
}
}  // namespace

// [1, C, 1, ...] (spatial) or [1, C, spatial...] (per activation), of float --
// double for double data -- whatever x's own type, as cuDNN derives it.
VGPU_EXPORT cudnnStatus_t cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t d, const cudnnTensorDescriptor_t xd,
                                                        cudnnBatchNormMode_t mode) {
  const TensorDesc* X = tdesc(xd);
  if (!known(d) || !X) return CUDNN_STATUS_BAD_PARAM;
  if (X->l.rank < 3) return BAD("cudnnDeriveBNTensorDescriptor", "x needs at least three dimensions");
  TensorDesc t;
  t.l.type = X->l.type == CUDNN_DATA_DOUBLE ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
  t.l.rank = X->l.rank;
  t.nd = X->nd;
  for (int i = 0; i < t.l.rank; ++i) t.l.dims[i] = (i == 1 || (!bn_spatial(mode) && i >= 2)) ? X->l.dims[i] : 1;
  packed_strides(t.l.rank, t.l.dims, false, t.l.strides);
  *as<TensorDesc>(d) = t;
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t h, cudnnBatchNormMode_t mode, const void* alpha, const void* beta, const cudnnTensorDescriptor_t xd,
    const void* x, const cudnnTensorDescriptor_t yd, void* y, const cudnnTensorDescriptor_t bnd, const void* scale,
    const void* bias, const void* mean, const void* var, double eps);

namespace {
// Batch normalization's fused forms (the Ex calls): y = act(bn(x) + z), the
// add and the activation each optional. The activation is RELU or SWISH, the
// two cuDNN fuses forward, and RELU alone backward (measured: the others are
// NOT_SUPPORTED, a missing descriptor BAD_PARAM). The pre-activation value
// goes to the reserve space, one float per element, for the backward pass,
// which can also do without it, from y.
struct BnFusion {
  bool add = false;
  const ActDesc* act = nullptr;  // null: no activation
  const TensorDesc* Z = nullptr;
  const void* z = nullptr;
  void* reserve = nullptr;
  size_t reserve_bytes = 0;
};

cudnnStatus_t bn_fusion(const char* fn, cudnnBatchNormOps_t ops, const void* zd, const void* z, const void* ad,
                        const Layout& x, void* reserve, size_t reserve_bytes, BnFusion* f) {
  if (ops == CUDNN_BATCHNORM_OPS_BN) return CUDNN_STATUS_SUCCESS;
  if (ops != CUDNN_BATCHNORM_OPS_BN_ACTIVATION && ops != CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION)
    return BAD(fn, "unknown batch normalization ops");
  if (!known(ad)) return BAD(fn, "a fused activation needs an activation descriptor");
  f->act = static_cast<const ActDesc*>(ad);
  if (f->act->mode != CUDNN_ACTIVATION_RELU && f->act->mode != CUDNN_ACTIVATION_SWISH)
    return UNSUPPORTED(fn, "only RELU and SWISH are fused into batch normalization");
  if (ops == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
    f->add = true;
    f->Z = tdesc(zd);
    if (!f->Z || !z) return BAD(fn, "the added tensor z is missing");
    if (!f->Z->l.same_dims(x)) return BAD(fn, "z differs from x in shape");
    f->z = z;
  }
  if (reserve && reserve_bytes >= x.count() * sizeof(float)) f->reserve = reserve, f->reserve_bytes = reserve_bytes;
  return CUDNN_STATUS_SUCCESS;
}

double fused_act(const ActDesc& a, double t) {
  return a.mode == CUDNN_ACTIVATION_RELU ? (t > 0.0 ? t : 0.0) : t * sigmoid(a.swish_beta * t);
}
double fused_act_grad(const ActDesc& a, double t) {
  if (a.mode == CUDNN_ACTIVATION_RELU) return t > 0.0 ? 1.0 : 0.0;
  const double s = sigmoid(a.swish_beta * t);
  return s + a.swish_beta * t * s * (1.0 - s);
}

cudnnStatus_t bn_forward_training(const char* fn, cudnnHandle_t h, cudnnBatchNormMode_t mode, const void* alpha,
                                  const void* beta, const void* xd, const void* x, const void* yd, void* y,
                                  const void* bnd, const void* scale, const void* bias, double exp_avg_factor,
                                  void* running_mean, void* running_var, double eps, void* save_mean,
                                  void* save_inv_var, const BnFusion& fuse) {
  const TensorDesc *X = tdesc(xd), *Y = tdesc(yd), *B = tdesc(bnd);
  if (!known(h) || !Y || !alpha || !beta || !x || !y || !scale || !bias)
    return BAD(fn, "invalid handle, descriptor or pointer");
  if (eps < CUDNN_BN_MIN_EPSILON) return BAD(fn, "epsilon is below CUDNN_BN_MIN_EPSILON");
  BnGeom g;
  cudnnStatus_t s = bn_check(fn, mode, X, B, &g);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  if (!X->l.same_dims(Y->l)) return BAD(fn, "x and y differ in shape");
  sync_handle(h);
  std::vector<double> vx, hs, hb, vz;
  if (!read(X->l, x, &vx) || !read(B->l, scale, &hs) || !read(B->l, bias, &hb)) return CUDNN_STATUS_EXECUTION_FAILED;
  if (fuse.add && !read(fuse.Z->l, fuse.z, &vz)) return CUDNN_STATUS_EXECUTION_FAILED;
  const size_t np = g.params();
  std::vector<double> sum(np, 0.0), sq(np, 0.0), cnt(np, 0.0);
  for (int64_t n = 0; n < g.N; ++n)
    for (int64_t c = 0; c < g.C; ++c)
      for (int64_t i = 0; i < g.S; ++i) {
        const size_t p = g.slot(c, i);
        const double v = vx[static_cast<size_t>((n * g.C + c) * g.S + i)];
        sum[p] += v, sq[p] += v * v, cnt[p] += 1;
      }
  std::vector<double> mean(np), var(np), inv(np);
  for (size_t p = 0; p < np; ++p) {
    mean[p] = cnt[p] ? sum[p] / cnt[p] : 0.0;
    var[p] = cnt[p] ? std::max(0.0, sq[p] / cnt[p] - mean[p] * mean[p]) : 0.0;
    inv[p] = 1.0 / std::sqrt(var[p] + eps);
  }
  std::vector<double> r(vx.size());
  std::vector<float> pre(fuse.act ? vx.size() : 0);
  for (int64_t n = 0; n < g.N; ++n)
    for (int64_t c = 0; c < g.C; ++c)
      for (int64_t i = 0; i < g.S; ++i) {
        const size_t p = g.slot(c, i), at = static_cast<size_t>((n * g.C + c) * g.S + i);
        double t = hs[p] * (vx[at] - mean[p]) * inv[p] + hb[p];
        if (fuse.add) t += vz[at];
        if (fuse.act) pre[at] = static_cast<float>(t), t = fused_act(*fuse.act, t);
        r[at] = t;
      }
  if (!blend_write(Y->l, y, r, sc(alpha, Y->l), sc(beta, Y->l))) return CUDNN_STATUS_EXECUTION_FAILED;
  if (fuse.reserve && cudaMemcpy(fuse.reserve, pre.data(), pre.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return CUDNN_STATUS_EXECUTION_FAILED;
  if (save_mean && !write(B->l, save_mean, mean)) return CUDNN_STATUS_EXECUTION_FAILED;
  if (save_inv_var && !write(B->l, save_inv_var, inv)) return CUDNN_STATUS_EXECUTION_FAILED;
  if (running_mean && running_var) {
    std::vector<double> rm, rv;
    if (!read(B->l, running_mean, &rm) || !read(B->l, running_var, &rv)) return CUDNN_STATUS_EXECUTION_FAILED;
    for (size_t p = 0; p < np; ++p) {
      // Running variance tracks the *unbiased* estimate, unlike the biased one
      // used to normalize this batch.
      const double unbiased = cnt[p] > 1 ? var[p] * cnt[p] / (cnt[p] - 1) : var[p];
      rm[p] = exp_avg_factor * mean[p] + (1.0 - exp_avg_factor) * rm[p];
      rv[p] = exp_avg_factor * unbiased + (1.0 - exp_avg_factor) * rv[p];
    }
    if (!write(B->l, running_mean, rm) || !write(B->l, running_var, rv)) return CUDNN_STATUS_EXECUTION_FAILED;
  }
  return CUDNN_STATUS_SUCCESS;
}

// The gradient of y = scale * (x - mean) * inv + bias, per channel (spatial)
// or per activation:
//   dbias  = sum g
//   dscale = sum g * xhat
//   dx     = scale * inv / m * (m * g - dbias - xhat * dscale)
// with the saved mean and inverse deviation when given, else recomputed. g is
// dy, or for a fused form dy through the activation's derivative (and dz = g
// when z was added).
cudnnStatus_t bn_backward(const char* fn, cudnnHandle_t h, cudnnBatchNormMode_t mode, const void* alpha_data,
                          const void* beta_data, const void* alpha_param, const void* beta_param, const void* xd,
                          const void* x, const void* yd, const void* y, const void* dyd, const void* dy,
                          const void* dzd, void* dz, const void* dxd, void* dx, const void* bnd, const void* scale,
                          const void* bias, void* dscale_out, void* dbias_out, double eps, const void* saved_mean,
                          const void* saved_inv, const BnFusion& fuse) {
  const TensorDesc *X = tdesc(xd), *DY = tdesc(dyd), *DX = tdesc(dxd), *B = tdesc(bnd);
  if (!known(h) || !DY || !DX || !alpha_data || !beta_data || !alpha_param || !beta_param || !x || !dy || !dx ||
      !scale)
    return BAD(fn, "invalid handle, descriptor or pointer");
  BnGeom g;
  cudnnStatus_t s = bn_check(fn, mode, X, B, &g);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  if (!X->l.same_dims(DY->l) || !X->l.same_dims(DX->l)) return BAD(fn, "x, dy and dx differ in shape");
  const TensorDesc *Y = tdesc(yd), *DZ = tdesc(dzd);
  // Measured: the hardware's fused backward pass takes RELU only.
  if (fuse.act && fuse.act->mode != CUDNN_ACTIVATION_RELU)
    return UNSUPPORTED(fn, "only RELU is fused into batch normalization's backward pass");
  if (fuse.act && !fuse.reserve && (!Y || !y || !Y->l.same_dims(X->l)))
    return BAD(fn, "a fused activation's backward pass needs y or the reserve space");
  if (fuse.add && (!DZ || !dz || !DZ->l.same_dims(X->l))) return BAD(fn, "the fused add's gradient dz is missing");
  const double ad = sc(alpha_data, DX->l), bd = sc(beta_data, DX->l);
  const double ap = sc(alpha_param, B->l), bp = sc(beta_param, B->l);
  sync_handle(h);
  std::vector<double> vx, vdy, hs;
  if (!read(X->l, x, &vx) || !read(DY->l, dy, &vdy) || !read(B->l, scale, &hs)) return CUDNN_STATUS_EXECUTION_FAILED;
  if (fuse.act) {
    std::vector<double> t;
    if (fuse.reserve) {
      std::vector<float> pre(vx.size());
      if (cudaMemcpy(pre.data(), fuse.reserve, pre.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return CUDNN_STATUS_EXECUTION_FAILED;
      t.assign(pre.begin(), pre.end());
    } else if (!read(Y->l, y, &t)) {  // RELU: y > 0 exactly where the pre-activation was
      return CUDNN_STATUS_EXECUTION_FAILED;
    }
    for (size_t i = 0; i < vdy.size(); ++i) vdy[i] *= fused_act_grad(*fuse.act, t[i]);
    if (fuse.add && !blend_write(DZ->l, dz, vdy, ad, bd)) return CUDNN_STATUS_EXECUTION_FAILED;
  }
  (void)bias;  // the gradient does not depend on it
  const size_t np = g.params();
  std::vector<double> mean(np, 0.0), inv(np, 0.0), cnt(np, 0.0);
  for (int64_t n = 0; n < g.N; ++n)
    for (int64_t c = 0; c < g.C; ++c)
      for (int64_t i = 0; i < g.S; ++i) cnt[g.slot(c, i)] += 1;
  if (saved_mean && saved_inv) {
    if (!read(B->l, saved_mean, &mean) || !read(B->l, saved_inv, &inv)) return CUDNN_STATUS_EXECUTION_FAILED;
  } else {
    std::vector<double> sum(np, 0.0), sq(np, 0.0);
    for (int64_t n = 0; n < g.N; ++n)
      for (int64_t c = 0; c < g.C; ++c)
        for (int64_t i = 0; i < g.S; ++i) {
          const size_t p = g.slot(c, i);
          const double v = vx[static_cast<size_t>((n * g.C + c) * g.S + i)];
          sum[p] += v, sq[p] += v * v;
        }
    for (size_t p = 0; p < np; ++p) {
      mean[p] = cnt[p] ? sum[p] / cnt[p] : 0.0;
      const double var = cnt[p] ? std::max(0.0, sq[p] / cnt[p] - mean[p] * mean[p]) : 0.0;
      inv[p] = 1.0 / std::sqrt(var + eps);
    }
  }
  std::vector<double> dbias(np, 0.0), dscale(np, 0.0);
  for (int64_t n = 0; n < g.N; ++n)
    for (int64_t c = 0; c < g.C; ++c)
      for (int64_t i = 0; i < g.S; ++i) {
        const size_t p = g.slot(c, i), at = static_cast<size_t>((n * g.C + c) * g.S + i);
        dbias[p] += vdy[at];
        dscale[p] += vdy[at] * (vx[at] - mean[p]) * inv[p];
      }
  std::vector<double> r(vx.size());
  for (int64_t n = 0; n < g.N; ++n)
    for (int64_t c = 0; c < g.C; ++c)
      for (int64_t i = 0; i < g.S; ++i) {
        const size_t p = g.slot(c, i), at = static_cast<size_t>((n * g.C + c) * g.S + i);
        const double xhat = (vx[at] - mean[p]) * inv[p];
        r[at] = hs[p] * inv[p] / cnt[p] * (cnt[p] * vdy[at] - dbias[p] - xhat * dscale[p]);
      }
  if (!blend_write(DX->l, dx, r, ad, bd)) return CUDNN_STATUS_EXECUTION_FAILED;
  if (dscale_out && !blend_write(B->l, dscale_out, dscale, ap, bp)) return CUDNN_STATUS_EXECUTION_FAILED;
  if (dbias_out && !blend_write(B->l, dbias_out, dbias, ap, bp)) return CUDNN_STATUS_EXECUTION_FAILED;
  return CUDNN_STATUS_SUCCESS;
}
}  // namespace

namespace {
cudnnStatus_t bn_inference(const char* fn, cudnnHandle_t h, cudnnBatchNormMode_t mode, const void* alpha,
                           const void* beta, const void* xd, const void* x, const void* yd, void* y, const void* bnd,
                           const void* scale, const void* bias, const void* mvd, const void* mean, const void* var,
                           double eps, const BnFusion& fuse) {
  const TensorDesc *X = tdesc(xd), *Y = tdesc(yd), *B = tdesc(bnd), *M = tdesc(mvd);
  if (!known(h) || !Y || !M || !alpha || !beta || !x || !y || !scale || !bias || !mean || !var)
    return BAD(fn, "invalid handle, descriptor or pointer");
  if (eps < CUDNN_BN_MIN_EPSILON) return BAD(fn, "epsilon is below CUDNN_BN_MIN_EPSILON");
  BnGeom g;
  cudnnStatus_t s = bn_check(fn, mode, X, B, &g);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  if (!X->l.same_dims(Y->l)) return BAD(fn, "x and y differ in shape");
  if (!M->l.same_dims(B->l)) return BAD(fn, "the mean and variance's shape is not the scale's");
  sync_handle(h);
  std::vector<double> vx, hs, hb, hm, hv, vz;
  if (!read(X->l, x, &vx) || !read(B->l, scale, &hs) || !read(B->l, bias, &hb) || !read(M->l, mean, &hm) ||
      !read(M->l, var, &hv))
    return CUDNN_STATUS_EXECUTION_FAILED;
  if (fuse.add && !read(fuse.Z->l, fuse.z, &vz)) return CUDNN_STATUS_EXECUTION_FAILED;
  for (int64_t n = 0; n < g.N; ++n)
    for (int64_t c = 0; c < g.C; ++c)
      for (int64_t i = 0; i < g.S; ++i) {
        const size_t p = g.slot(c, i), at = static_cast<size_t>((n * g.C + c) * g.S + i);
        double t = hs[p] * (vx[at] - hm[p]) / std::sqrt(eps + hv[p]) + hb[p];
        if (fuse.add) t += vz[at];
        vx[at] = fuse.act ? fused_act(*fuse.act, t) : t;
      }
  return blend_write(Y->l, y, vx, sc(alpha, Y->l), sc(beta, Y->l)) ? CUDNN_STATUS_SUCCESS
                                                                    : CUDNN_STATUS_EXECUTION_FAILED;
}
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t h, cudnnBatchNormMode_t mode, const void* alpha, const void* beta, const cudnnTensorDescriptor_t xd,
    const void* x, const cudnnTensorDescriptor_t yd, void* y, const cudnnTensorDescriptor_t bnd, const void* scale,
    const void* bias, const void* mean, const void* var, double eps) {
  return bn_inference("cudnnBatchNormalizationForwardInference", h, mode, alpha, beta, xd, x, yd, y, bnd, scale, bias,
                      bnd, mean, var, eps, BnFusion{});
}

VGPU_EXPORT cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t h, cudnnBatchNormMode_t mode, const void* alpha, const void* beta, const cudnnTensorDescriptor_t xd,
    const void* x, const cudnnTensorDescriptor_t yd, void* y, const cudnnTensorDescriptor_t bnd, const void* scale,
    const void* bias, double exp_avg_factor, void* running_mean, void* running_var, double eps, void* save_mean,
    void* save_inv_var) {
  return bn_forward_training("cudnnBatchNormalizationForwardTraining", h, mode, alpha, beta, xd, x, yd, y, bnd, scale,
                             bias, exp_avg_factor, running_mean, running_var, eps, save_mean, save_inv_var, BnFusion{});
}

// Workspace: none, since the work is done on the host. Reserve space: the
// fused forms keep the pre-activation value, a float per element.
VGPU_EXPORT cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const cudnnTensorDescriptor_t,
    const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
    const cudnnActivationDescriptor_t, size_t* size) {
  if (size) *size = 0;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const cudnnTensorDescriptor_t,
    const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
    const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const cudnnActivationDescriptor_t, size_t* size) {
  if (size) *size = 0;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t ops, const cudnnActivationDescriptor_t,
    const cudnnTensorDescriptor_t xd, size_t* size) {
  if (!size) return CUDNN_STATUS_BAD_PARAM;
  const TensorDesc* X = tdesc(xd);
  *size = ops != CUDNN_BATCHNORM_OPS_BN && X ? X->l.count() * sizeof(float) : 0;
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t h, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t ops, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t xd, const void* x, const cudnnTensorDescriptor_t zd, const void* z,
    const cudnnTensorDescriptor_t yd, void* y, const cudnnTensorDescriptor_t bnd, const void* scale, const void* bias,
    double factor, void* running_mean, void* running_var, double eps, void* save_mean, void* save_inv_var,
    cudnnActivationDescriptor_t ad, void*, size_t, void* reserve, size_t reserve_bytes) {
  static const char* fn = "cudnnBatchNormalizationForwardTrainingEx";
  const TensorDesc* X = tdesc(xd);
  if (!X) return BAD(fn, "x's descriptor is null, destroyed or never set");
  BnFusion fuse;
  cudnnStatus_t s = bn_fusion(fn, ops, zd, z, ad, X->l, reserve, reserve_bytes, &fuse);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  return bn_forward_training(fn, h, mode, alpha, beta, xd, x, yd, y, bnd, scale, bias, factor, running_mean,
                             running_var, eps, save_mean, save_inv_var, fuse);
}

VGPU_EXPORT cudnnStatus_t cudnnBatchNormalizationBackward(
    cudnnHandle_t h, cudnnBatchNormMode_t mode, const void* alpha_data, const void* beta_data, const void* alpha_param,
    const void* beta_param, const cudnnTensorDescriptor_t xd, const void* x, const cudnnTensorDescriptor_t dyd,
    const void* dy, const cudnnTensorDescriptor_t dxd, void* dx, const cudnnTensorDescriptor_t bnd, const void* scale,
    void* dscale_out, void* dbias_out, double eps, const void* saved_mean, const void* saved_inv) {
  return bn_backward("cudnnBatchNormalizationBackward", h, mode, alpha_data, beta_data, alpha_param, beta_param, xd, x,
                     nullptr, nullptr, dyd, dy, nullptr, nullptr, dxd, dx, bnd, scale, nullptr, dscale_out, dbias_out,
                     eps, saved_mean, saved_inv, BnFusion{});
}

VGPU_EXPORT cudnnStatus_t cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t h, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t ops, const void* alpha_data, const void* beta_data,
    const void* alpha_param, const void* beta_param, const cudnnTensorDescriptor_t xd, const void* x,
    const cudnnTensorDescriptor_t yd, const void* y, const cudnnTensorDescriptor_t dyd, const void* dy,
    const cudnnTensorDescriptor_t dzd, void* dz, const cudnnTensorDescriptor_t dxd, void* dx,
    const cudnnTensorDescriptor_t bnd, const void* scale, const void* bias, void* dscale, void* dbias, double eps,
    const void* saved_mean, const void* saved_inv, cudnnActivationDescriptor_t ad, void*, size_t, void* reserve,
    size_t reserve_bytes) {
  static const char* fn = "cudnnBatchNormalizationBackwardEx";
  const TensorDesc* X = tdesc(xd);
  if (!X) return BAD(fn, "x's descriptor is null, destroyed or never set");
  BnFusion fuse;
  // The added tensor's gradient stands in for z when checking the fusion.
  cudnnStatus_t s = bn_fusion(fn, ops, dzd, dz, ad, X->l, reserve, reserve_bytes, &fuse);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  return bn_backward(fn, h, mode, alpha_data, beta_data, alpha_param, beta_param, xd, x, yd, y, dyd, dy, dzd, dz, dxd,
                     dx, bnd, scale, bias, dscale, dbias, eps, saved_mean, saved_inv, fuse);
}

/* ---- the cuDNN 8 normalization API: batch normalization by another name ---- */

namespace {
// PER_CHANNEL is batch normalization's SPATIAL mode, PER_ACTIVATION its own;
// the ops map one to one. One group only (measured: more is NOT_SUPPORTED).
cudnnStatus_t norm_mode(const char* fn, cudnnNormMode_t mode, int groups, cudnnBatchNormMode_t* bn) {
  if (groups != 1) return UNSUPPORTED(fn, "only one group (groupCnt 1) is supported");
  if (mode == CUDNN_NORM_PER_CHANNEL) *bn = CUDNN_BATCHNORM_SPATIAL;
  else if (mode == CUDNN_NORM_PER_ACTIVATION) *bn = CUDNN_BATCHNORM_PER_ACTIVATION;
  else return BAD(fn, "unknown normalization mode");
  return CUDNN_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnDeriveNormTensorDescriptor(cudnnTensorDescriptor_t scale_bias,
                                                          cudnnTensorDescriptor_t mean_var,
                                                          const cudnnTensorDescriptor_t xd, cudnnNormMode_t mode,
                                                          int groups) {
  cudnnBatchNormMode_t bn;
  cudnnStatus_t s = norm_mode("cudnnDeriveNormTensorDescriptor", mode, groups, &bn);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  if ((s = cudnnDeriveBNTensorDescriptor(scale_bias, xd, bn)) != CUDNN_STATUS_SUCCESS) return s;
  return cudnnDeriveBNTensorDescriptor(mean_var, xd, bn);
}

VGPU_EXPORT cudnnStatus_t cudnnNormalizationForwardInference(
    cudnnHandle_t h, cudnnNormMode_t mode, cudnnNormOps_t ops, cudnnNormAlgo_t, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t xd, const void* x, const cudnnTensorDescriptor_t sbd, const void* scale,
    const void* bias, const cudnnTensorDescriptor_t mvd, const void* mean, const void* var,
    const cudnnTensorDescriptor_t zd, const void* z, cudnnActivationDescriptor_t ad, const cudnnTensorDescriptor_t yd,
    void* y, double eps, int groups) {
  static const char* fn = "cudnnNormalizationForwardInference";
  cudnnBatchNormMode_t bn;
  cudnnStatus_t s = norm_mode(fn, mode, groups, &bn);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  const TensorDesc* X = tdesc(xd);
  if (!X) return BAD(fn, "x's descriptor is null, destroyed or never set");
  BnFusion fuse;
  if ((s = bn_fusion(fn, static_cast<cudnnBatchNormOps_t>(ops), zd, z, ad, X->l, nullptr, 0, &fuse)) != CUDNN_STATUS_SUCCESS)
    return s;
  return bn_inference(fn, h, bn, alpha, beta, xd, x, yd, y, sbd, scale, bias, mvd, mean, var, eps, fuse);
}

VGPU_EXPORT cudnnStatus_t cudnnGetNormalizationForwardTrainingWorkspaceSize(
    cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, const cudnnTensorDescriptor_t,
    const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
    const cudnnActivationDescriptor_t, const cudnnTensorDescriptor_t, size_t* size, int groups) {
  if (!size) return CUDNN_STATUS_BAD_PARAM;
  if (groups != 1) return CUDNN_STATUS_NOT_SUPPORTED;
  *size = 0;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetNormalizationBackwardWorkspaceSize(
    cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t, const cudnnTensorDescriptor_t,
    const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
    const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const cudnnActivationDescriptor_t,
    const cudnnTensorDescriptor_t, size_t* size, int groups) {
  if (!size) return CUDNN_STATUS_BAD_PARAM;
  if (groups != 1) return CUDNN_STATUS_NOT_SUPPORTED;
  *size = 0;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetNormalizationTrainingReserveSpaceSize(cudnnHandle_t h, cudnnNormMode_t,
                                                                        cudnnNormOps_t ops, cudnnNormAlgo_t,
                                                                        const cudnnActivationDescriptor_t ad,
                                                                        const cudnnTensorDescriptor_t xd, size_t* size,
                                                                        int groups) {
  if (groups != 1) return CUDNN_STATUS_NOT_SUPPORTED;
  return cudnnGetBatchNormalizationTrainingExReserveSpaceSize(h, CUDNN_BATCHNORM_SPATIAL,
                                                              static_cast<cudnnBatchNormOps_t>(ops), ad, xd, size);
}

VGPU_EXPORT cudnnStatus_t cudnnNormalizationForwardTraining(
    cudnnHandle_t h, cudnnNormMode_t mode, cudnnNormOps_t ops, cudnnNormAlgo_t, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t xd, const void* x, const cudnnTensorDescriptor_t sbd, const void* scale,
    const void* bias, double factor, const cudnnTensorDescriptor_t mvd, void* running_mean, void* running_var,
    double eps, void* save_mean, void* save_inv_var, cudnnActivationDescriptor_t ad, const cudnnTensorDescriptor_t zd,
    const void* z, const cudnnTensorDescriptor_t yd, void* y, void*, size_t, void* reserve, size_t reserve_bytes,
    int groups) {
  static const char* fn = "cudnnNormalizationForwardTraining";
  cudnnBatchNormMode_t bn;
  cudnnStatus_t s = norm_mode(fn, mode, groups, &bn);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  const TensorDesc *X = tdesc(xd), *SB = tdesc(sbd), *MV = tdesc(mvd);
  if (!X || !SB || !MV) return BAD(fn, "a descriptor is null, destroyed or never set");
  if (!MV->l.same_dims(SB->l) || MV->l.type != SB->l.type)
    return UNSUPPORTED(fn, "mean and variance shaped or typed unlike the scale and bias");
  BnFusion fuse;
  if ((s = bn_fusion(fn, static_cast<cudnnBatchNormOps_t>(ops), zd, z, ad, X->l, reserve, reserve_bytes, &fuse)) !=
      CUDNN_STATUS_SUCCESS)
    return s;
  return bn_forward_training(fn, h, bn, alpha, beta, xd, x, yd, y, sbd, scale, bias, factor, running_mean,
                             running_var, eps, save_mean, save_inv_var, fuse);
}

VGPU_EXPORT cudnnStatus_t cudnnNormalizationBackward(
    cudnnHandle_t h, cudnnNormMode_t mode, cudnnNormOps_t ops, cudnnNormAlgo_t, const void* alpha_data,
    const void* beta_data, const void* alpha_param, const void* beta_param, const cudnnTensorDescriptor_t xd,
    const void* x, const cudnnTensorDescriptor_t yd, const void* y, const cudnnTensorDescriptor_t dyd, const void* dy,
    const cudnnTensorDescriptor_t dzd, void* dz, const cudnnTensorDescriptor_t dxd, void* dx,
    const cudnnTensorDescriptor_t sbd, const void* scale, const void* bias, void* dscale, void* dbias, double eps,
    const cudnnTensorDescriptor_t mvd, const void* saved_mean, const void* saved_inv, cudnnActivationDescriptor_t ad,
    void*, size_t, void* reserve, size_t reserve_bytes, int groups) {
  static const char* fn = "cudnnNormalizationBackward";
  cudnnBatchNormMode_t bn;
  cudnnStatus_t s = norm_mode(fn, mode, groups, &bn);
  if (s != CUDNN_STATUS_SUCCESS) return s;
  const TensorDesc *X = tdesc(xd), *SB = tdesc(sbd), *MV = tdesc(mvd);
  if (!X || !SB || !MV) return BAD(fn, "a descriptor is null, destroyed or never set");
  if (!MV->l.same_dims(SB->l) || MV->l.type != SB->l.type)
    return UNSUPPORTED(fn, "mean and variance shaped or typed unlike the scale and bias");
  BnFusion fuse;
  if ((s = bn_fusion(fn, static_cast<cudnnBatchNormOps_t>(ops), dzd, dz, ad, X->l, reserve, reserve_bytes, &fuse)) !=
      CUDNN_STATUS_SUCCESS)
    return s;
  return bn_backward(fn, h, bn, alpha_data, beta_data, alpha_param, beta_param, xd, x, yd, y, dyd, dy, dzd, dz, dxd,
                     dx, sbd, scale, bias, dscale, dbias, eps, saved_mean, saved_inv, fuse);
}

/* ---- spatial transformer: affine grid and bilinear sampling ---- */

namespace {
struct StDesc {
  cudnnSamplerType_t sampler = CUDNN_SAMPLER_BILINEAR;
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  int n = 0, c = 0, h = 0, w = 0;  // the output's dimensions
  bool set = false;
};
// theta [N, 2, 3] and grid [N, H, W, 2], packed, of the descriptor's type.
Layout st_array(const StDesc& d, std::initializer_list<int64_t> dims) {
  Layout l;
  l.type = d.type;
  l.rank = static_cast<int>(dims.size());
  int i = 0;
  for (int64_t v : dims) l.dims[i++] = v;
  packed_strides(l.rank, l.dims, false, l.strides);
  return l;
}
// The output's normalized coordinates: -1 to 1 across the pixel centres, the
// corners included (measured: a 4-wide row is -1, -1/3, 1/3, 1).
inline double st_coord(int i, int n) { return n > 1 ? -1.0 + 2.0 * i / (n - 1) : 0.0; }
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnSpatialTransformerDescriptor_t>(track(new StDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete as<StDesc>(d);
  return CUDNN_STATUS_SUCCESS;
}
// The output's [N, C, H, W]; a fifth dimension is accepted and ignored here,
// as only 2-D transforms exist.
VGPU_EXPORT cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t d,
                                                                 cudnnSamplerType_t sampler, cudnnDataType_t type,
                                                                 const int nb, const int dims[]) {
  static const char* fn = "cudnnSetSpatialTransformerNdDescriptor";
  if (!known(d) || !dims || nb < 4) return CUDNN_STATUS_BAD_PARAM;
  if (sampler != CUDNN_SAMPLER_BILINEAR) return BAD(fn, "only the bilinear sampler exists");
  if (!floating(type)) return UNSUPPORTED(fn, std::string(type_name(type)) + " data");
  for (int i = 0; i < 4; ++i)
    if (dims[i] <= 0) return BAD(fn, "a dimension is not positive");
  *as<StDesc>(d) = StDesc{sampler, type, dims[0], dims[1], dims[2], dims[3], true};
  return CUDNN_STATUS_SUCCESS;
}

// grid[n, h, w] = theta[n] * (x_t, y_t, 1) over the output's normalized coordinates.
VGPU_EXPORT cudnnStatus_t cudnnSpatialTfGridGeneratorForward(cudnnHandle_t h,
                                                             const cudnnSpatialTransformerDescriptor_t d,
                                                             const void* theta, void* grid) {
  if (!known(h) || !known(d) || !theta || !grid || !as<const StDesc>(d)->set) return CUDNN_STATUS_BAD_PARAM;
  const StDesc& D = *as<const StDesc>(d);
  const Layout tl = st_array(D, {D.n, 2, 3}), gl = st_array(D, {D.n, D.h, D.w, 2});
  sync_handle(h);
  std::vector<double> th, g(gl.count());
  if (!read(tl, theta, &th)) return CUDNN_STATUS_EXECUTION_FAILED;
  for (int n = 0; n < D.n; ++n)
    for (int i = 0; i < D.h; ++i)
      for (int j = 0; j < D.w; ++j) {
        const double xt = st_coord(j, D.w), yt = st_coord(i, D.h), *t = th.data() + n * 6;
        double* o = g.data() + ((static_cast<size_t>(n) * D.h + i) * D.w + j) * 2;
        o[0] = t[0] * xt + t[1] * yt + t[2];
        o[1] = t[3] * xt + t[4] * yt + t[5];
      }
  return write(gl, grid, g) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

// dtheta[n] = sum over the grid of dgrid (x) (x_t, y_t, 1).
VGPU_EXPORT cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t h,
                                                              const cudnnSpatialTransformerDescriptor_t d,
                                                              const void* dgrid, void* dtheta) {
  if (!known(h) || !known(d) || !dgrid || !dtheta || !as<const StDesc>(d)->set) return CUDNN_STATUS_BAD_PARAM;
  const StDesc& D = *as<const StDesc>(d);
  const Layout tl = st_array(D, {D.n, 2, 3}), gl = st_array(D, {D.n, D.h, D.w, 2});
  sync_handle(h);
  std::vector<double> g, th(tl.count(), 0.0);
  if (!read(gl, dgrid, &g)) return CUDNN_STATUS_EXECUTION_FAILED;
  for (int n = 0; n < D.n; ++n)
    for (int i = 0; i < D.h; ++i)
      for (int j = 0; j < D.w; ++j) {
        const double xt = st_coord(j, D.w), yt = st_coord(i, D.h);
        const double* q = g.data() + ((static_cast<size_t>(n) * D.h + i) * D.w + j) * 2;
        double* t = th.data() + n * 6;
        t[0] += q[0] * xt, t[1] += q[0] * yt, t[2] += q[0];
        t[3] += q[1] * xt, t[4] += q[1] * yt, t[5] += q[1];
      }
  return write(tl, dtheta, th) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

namespace {
// Bilinear sampling of one plane at normalized (gx, gy): the source pixel is
// ((gx + 1)(W - 1) / 2, (gy + 1)(H - 1) / 2), the four around it weighted by
// distance, and those outside the plane read as zero (measured). f(at, weight)
// is called for each corner inside; the derivatives of the result with
// respect to the pixel coordinates come back in dx, dy when asked.
template <class F>
void bilinear(int64_t H, int64_t W, double gx, double gy, F&& f) {
  const double px = (gx + 1.0) * (W - 1) / 2.0, py = (gy + 1.0) * (H - 1) / 2.0;
  const double x0 = std::floor(px), y0 = std::floor(py), fx = px - x0, fy = py - y0;
  for (int dyi = 0; dyi < 2; ++dyi)
    for (int dxi = 0; dxi < 2; ++dxi) {
      const int64_t xi = static_cast<int64_t>(x0) + dxi, yi = static_cast<int64_t>(y0) + dyi;
      if (xi < 0 || xi >= W || yi < 0 || yi >= H) continue;
      const double wx = dxi ? fx : 1.0 - fx, wy = dyi ? fy : 1.0 - fy;
      // d(weight)/d(px), d(weight)/d(py)
      const double dwx = (dxi ? 1.0 : -1.0) * wy, dwy = (dyi ? 1.0 : -1.0) * wx;
      f(yi * W + xi, wx * wy, dwx, dwy);
    }
}
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnSpatialTfSamplerForward(cudnnHandle_t h, cudnnSpatialTransformerDescriptor_t d,
                                                       const void* alpha, const cudnnTensorDescriptor_t xd,
                                                       const void* x, const void* grid, const void* beta,
                                                       cudnnTensorDescriptor_t yd, void* y) {
  static const char* fn = "cudnnSpatialTfSamplerForward";
  const TensorDesc *X = tdesc(xd), *Y = tdesc(yd);
  if (!known(h) || !known(d) || !X || !Y || !alpha || !beta || !x || !grid || !y)
    return BAD(fn, "invalid handle, descriptor or pointer");
  const StDesc& D = *as<const StDesc>(d);
  if (!D.set || X->l.rank != 4 || Y->l.rank != 4 || Y->l.dims[0] != D.n || Y->l.dims[1] != D.c ||
      Y->l.dims[2] != D.h || Y->l.dims[3] != D.w || X->l.dims[0] != D.n || X->l.dims[1] != D.c)
    return BAD(fn, "x, y and the transformer's dimensions do not agree");
  const Layout gl = st_array(D, {D.n, D.h, D.w, 2});
  const int64_t Hi = X->l.dims[2], Wi = X->l.dims[3];
  sync_handle(h);
  std::vector<double> vx, g;
  if (!read(X->l, x, &vx) || !read(gl, grid, &g)) return CUDNN_STATUS_EXECUTION_FAILED;
  std::vector<double> r(Y->l.count(), 0.0);
  for (int n = 0; n < D.n; ++n)
    for (int i = 0; i < D.h; ++i)
      for (int j = 0; j < D.w; ++j) {
        const double* q = g.data() + ((static_cast<size_t>(n) * D.h + i) * D.w + j) * 2;
        for (int c = 0; c < D.c; ++c) {
          const double* plane = vx.data() + (static_cast<size_t>(n) * D.c + c) * Hi * Wi;
          double v = 0.0;
          bilinear(Hi, Wi, q[0], q[1], [&](int64_t at, double wt, double, double) { v += wt * plane[at]; });
          r[((static_cast<size_t>(n) * D.c + c) * D.h + i) * D.w + j] = v;
        }
      }
  return blend_write(Y->l, y, r, sc(alpha, Y->l), sc(beta, Y->l)) ? CUDNN_STATUS_SUCCESS
                                                                   : CUDNN_STATUS_EXECUTION_FAILED;
}

// dx = alpha * (dy spread back over each sample's four inputs) + beta * dx;
// dgrid = alphaDgrid * (dy times the sample's slope in each coordinate) +
// betaDgrid * dgrid.
VGPU_EXPORT cudnnStatus_t cudnnSpatialTfSamplerBackward(cudnnHandle_t h, cudnnSpatialTransformerDescriptor_t d,
                                                        const void* alpha, const cudnnTensorDescriptor_t xd,
                                                        const void* x, const void* beta,
                                                        const cudnnTensorDescriptor_t dxd, void* dx,
                                                        const void* alpha_grid, const cudnnTensorDescriptor_t dyd,
                                                        const void* dy, const void* grid, const void* beta_grid,
                                                        void* dgrid) {
  static const char* fn = "cudnnSpatialTfSamplerBackward";
  const TensorDesc *X = tdesc(xd), *DX = tdesc(dxd), *DY = tdesc(dyd);
  if (!known(h) || !known(d) || !X || !DX || !DY || !alpha || !beta || !alpha_grid || !beta_grid || !x || !dx ||
      !dy || !grid || !dgrid)
    return BAD(fn, "invalid handle, descriptor or pointer");
  const StDesc& D = *as<const StDesc>(d);
  if (!D.set || X->l.rank != 4 || !X->l.same_dims(DX->l) || DY->l.rank != 4 || DY->l.dims[0] != D.n ||
      DY->l.dims[1] != D.c || DY->l.dims[2] != D.h || DY->l.dims[3] != D.w || X->l.dims[0] != D.n ||
      X->l.dims[1] != D.c)
    return BAD(fn, "x, dx, dy and the transformer's dimensions do not agree");
  const Layout gl = st_array(D, {D.n, D.h, D.w, 2});
  const int64_t Hi = X->l.dims[2], Wi = X->l.dims[3];
  sync_handle(h);
  std::vector<double> vx, vdy, g;
  if (!read(X->l, x, &vx) || !read(DY->l, dy, &vdy) || !read(gl, grid, &g)) return CUDNN_STATUS_EXECUTION_FAILED;
  std::vector<double> rdx(vx.size(), 0.0), rdg(g.size(), 0.0);
  for (int n = 0; n < D.n; ++n)
    for (int i = 0; i < D.h; ++i)
      for (int j = 0; j < D.w; ++j) {
        const size_t gi = ((static_cast<size_t>(n) * D.h + i) * D.w + j) * 2;
        for (int c = 0; c < D.c; ++c) {
          const size_t pbase = (static_cast<size_t>(n) * D.c + c) * Hi * Wi;
          const double gy = vdy[((static_cast<size_t>(n) * D.c + c) * D.h + i) * D.w + j];
          bilinear(Hi, Wi, g[gi], g[gi + 1], [&](int64_t at, double wt, double dwx, double dwy) {
            rdx[pbase + at] += wt * gy;
            rdg[gi] += gy * vx[pbase + at] * dwx * (Wi - 1) / 2.0;
            rdg[gi + 1] += gy * vx[pbase + at] * dwy * (Hi - 1) / 2.0;
          });
        }
      }
  if (!blend_write(DX->l, dx, rdx, sc(alpha, DX->l), sc(beta, DX->l))) return CUDNN_STATUS_EXECUTION_FAILED;
  return blend_write(gl, dgrid, rdg, scale_of(alpha_grid, D.type), scale_of(beta_grid, D.type))
             ? CUDNN_STATUS_SUCCESS
             : CUDNN_STATUS_EXECUTION_FAILED;
}

/* ---- CTC loss ---- */

namespace {
struct CtcDesc {
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  cudnnLossNormalizationMode_t norm = CUDNN_LOSS_NORMALIZATION_SOFTMAX;
  cudnnCTCGradMode_t grad = CUDNN_CTC_ZERO_OOB_GRADIENTS;
  int max_label = 255;
};

inline double log_add(double a, double b) {
  if (a == -INFINITY) return b;
  if (b == -INFINITY) return a;
  const double m = std::max(a, b);
  return m + std::log1p(std::exp(-std::fabs(a - b)));
}

// The loss and its gradient for every sequence, by the forward-backward
// recursion over the label with blanks between and around (label 0 is the
// blank), in log space. SOFTMAX mode takes activations and gives the gradient
// with respect to them, y - posterior; NONE takes probabilities and gives the
// gradient with respect to them, -posterior / y where some path passes and,
// as measured on the hardware, y where none does. A label the input is too
// short for costs 0, with its gradient zeroed (up to the batch's longest
// input) or left alone as the descriptor says; otherwise steps past a
// sequence's length are left alone.
cudnnStatus_t ctc(const char* fn, cudnnHandle_t h, const CtcDesc& D, const void* pd, const void* probs,
                  const std::vector<int>& labels, const std::vector<int>& label_len, const std::vector<int>& input_len,
                  void* costs, const void* gd, void* grads) {
  const TensorDesc *P = tdesc(pd), *G = tdesc(gd);
  if (!known(h) || !P || !probs || !costs) return BAD(fn, "invalid handle, descriptor or pointer");
  if (P->l.rank != 3) return BAD(fn, "probabilities must be [T, N, A]");
  if (grads && (!G || !G->l.same_dims(P->l))) return BAD(fn, "the gradients' shape is not the probabilities'");
  if (!floating(P->l.type)) return UNSUPPORTED(fn, std::string(type_name(P->l.type)) + " data");
  const int T = static_cast<int>(P->l.dims[0]), N = static_cast<int>(P->l.dims[1]), A = static_cast<int>(P->l.dims[2]);
  if (static_cast<int>(label_len.size()) < N || static_cast<int>(input_len.size()) < N) return BAD(fn, "lengths missing");
  size_t total = 0;
  for (int n = 0; n < N; ++n) {
    if (label_len[n] < 0 || label_len[n] > D.max_label || input_len[n] < 0 || input_len[n] > T)
      return BAD(fn, "a label or input length is out of range");
    total += static_cast<size_t>(label_len[n]);
  }
  if (labels.size() < total) return BAD(fn, "fewer labels than the label lengths add up to");
  for (size_t i = 0; i < total; ++i)
    if (labels[i] < 1 || labels[i] >= A) return BAD(fn, "a label is outside [1, A - 1]");
  sync_handle(h);
  std::vector<double> in, g;
  if (!read(P->l, probs, &in)) return CUDNN_STATUS_EXECUTION_FAILED;
  if (grads && !read(G->l, grads, &g)) return CUDNN_STATUS_EXECUTION_FAILED;  // what is left alone stays
  // log y[t][n][a]
  std::vector<double> ly(in.size());
  for (int t = 0; t < T; ++t)
    for (int n = 0; n < N; ++n) {
      const double* row = in.data() + (static_cast<size_t>(t) * N + n) * A;
      double* out = ly.data() + (static_cast<size_t>(t) * N + n) * A;
      if (D.norm == CUDNN_LOSS_NORMALIZATION_SOFTMAX) {
        double m = -INFINITY, sum = 0.0;
        for (int a = 0; a < A; ++a) m = std::max(m, row[a]);
        for (int a = 0; a < A; ++a) sum += std::exp(row[a] - m);
        for (int a = 0; a < A; ++a) out[a] = row[a] - m - std::log(sum);
      } else {
        for (int a = 0; a < A; ++a) out[a] = std::log(row[a]);
      }
    }
  auto LY = [&](int t, int n, int a) { return ly[(static_cast<size_t>(t) * N + n) * A + a]; };
  std::vector<double> cost(N, 0.0);
  const int max_len = *std::max_element(input_len.begin(), input_len.begin() + N);
  size_t off = 0;
  for (int n = 0; n < N; ++n) {
    const int L = label_len[n], Tn = input_len[n], S = 2 * L + 1;
    std::vector<int> lp(S, 0);
    for (int i = 0; i < L; ++i) lp[2 * i + 1] = labels[off + i];
    off += static_cast<size_t>(L);
    std::vector<double> al(static_cast<size_t>(Tn) * S, -INFINITY), be(static_cast<size_t>(Tn) * S, -INFINITY);
    auto at = [&](int t, int s) { return static_cast<size_t>(t) * S + s; };
    double logp = -INFINITY;
    if (Tn > 0) {
      al[at(0, 0)] = LY(0, n, lp[0]);
      if (S > 1) al[at(0, 1)] = LY(0, n, lp[1]);
      for (int t = 1; t < Tn; ++t)
        for (int s = 0; s < S; ++s) {
          double v = al[at(t - 1, s)];
          if (s > 0) v = log_add(v, al[at(t - 1, s - 1)]);
          if (s > 1 && lp[s] != 0 && lp[s] != lp[s - 2]) v = log_add(v, al[at(t - 1, s - 2)]);
          al[at(t, s)] = v + LY(t, n, lp[s]);
        }
      be[at(Tn - 1, S - 1)] = LY(Tn - 1, n, lp[S - 1]);
      if (S > 1) be[at(Tn - 1, S - 2)] = LY(Tn - 1, n, lp[S - 2]);
      for (int t = Tn - 2; t >= 0; --t)
        for (int s = 0; s < S; ++s) {
          double v = be[at(t + 1, s)];
          if (s + 1 < S) v = log_add(v, be[at(t + 1, s + 1)]);
          if (s + 2 < S && lp[s] != 0 && lp[s] != lp[s + 2]) v = log_add(v, be[at(t + 1, s + 2)]);
          be[at(t, s)] = v + LY(t, n, lp[s]);
        }
      logp = al[at(Tn - 1, S - 1)];
      if (S > 1) logp = log_add(logp, al[at(Tn - 1, S - 2)]);
    }
    const bool feasible = logp > -INFINITY;
    cost[n] = feasible ? -logp : 0.0;
    if (!grads) continue;
    if (!feasible) {
      // Measured: up to the longest input of the batch, past this one's too.
      if (D.grad == CUDNN_CTC_ZERO_OOB_GRADIENTS)
        for (int t = 0; t < max_len; ++t)
          for (int a = 0; a < A; ++a) g[(static_cast<size_t>(t) * N + n) * A + a] = 0.0;
      continue;
    }
    for (int t = 0; t < Tn; ++t) {
      // log of sum over s with lp[s] = a of alpha * beta (each includes y once too many)
      std::vector<double> lab(A, -INFINITY);
      for (int s = 0; s < S; ++s) lab[lp[s]] = log_add(lab[lp[s]], al[at(t, s)] + be[at(t, s)]);
      for (int a = 0; a < A; ++a) {
        const double lyta = LY(t, n, a), yv = std::exp(lyta);
        double& out = g[(static_cast<size_t>(t) * N + n) * A + a];
        if (D.norm == CUDNN_LOSS_NORMALIZATION_SOFTMAX)
          out = yv - std::exp(lab[a] - logp - lyta);
        else
          out = lab[a] == -INFINITY ? yv : -std::exp(lab[a] - logp - 2.0 * lyta);
      }
    }
  }
  Layout cl;
  cl.type = P->l.type;
  cl.rank = 1;
  cl.dims[0] = N;
  cl.strides[0] = 1;
  if (!write(cl, costs, cost)) return CUDNN_STATUS_EXECUTION_FAILED;
  if (grads && !write(G->l, grads, g)) return CUDNN_STATUS_EXECUTION_FAILED;
  return CUDNN_STATUS_SUCCESS;
}

std::vector<int> device_ints(const int* p, size_t n) {
  std::vector<int> v(n, 0);
  if (p && n) cudaMemcpy(v.data(), p, n * sizeof(int), cudaMemcpyDeviceToHost);
  return v;
}
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnCTCLossDescriptor_t>(track(new CtcDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete as<CtcDesc>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetCTCLossDescriptor_v9(cudnnCTCLossDescriptor_t d, cudnnDataType_t type,
                                                       cudnnLossNormalizationMode_t norm, cudnnCTCGradMode_t grad,
                                                       int max_label) {
  if (!known(d) || max_label < 0) return CUDNN_STATUS_BAD_PARAM;
  if (norm != CUDNN_LOSS_NORMALIZATION_NONE && norm != CUDNN_LOSS_NORMALIZATION_SOFTMAX)
    return BAD("cudnnSetCTCLossDescriptor_v9", "unknown normalization mode");
  *as<CtcDesc>(d) = CtcDesc{type, norm, grad, max_label};
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetCTCLossDescriptor_v8(cudnnCTCLossDescriptor_t d, cudnnDataType_t type,
                                                       cudnnLossNormalizationMode_t norm, cudnnNanPropagation_t,
                                                       int max_label) {
  return cudnnSetCTCLossDescriptor_v9(d, type, norm, CUDNN_CTC_ZERO_OOB_GRADIENTS, max_label);
}
VGPU_EXPORT cudnnStatus_t cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t d, cudnnDataType_t type,
                                                      cudnnLossNormalizationMode_t norm, cudnnNanPropagation_t) {
  return cudnnSetCTCLossDescriptor_v9(d, type, norm, CUDNN_CTC_ZERO_OOB_GRADIENTS, 255);
}
VGPU_EXPORT cudnnStatus_t cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t d, cudnnDataType_t type) {
  return cudnnSetCTCLossDescriptor_v9(d, type, CUDNN_LOSS_NORMALIZATION_NONE, CUDNN_CTC_ZERO_OOB_GRADIENTS, 255);
}
VGPU_EXPORT cudnnStatus_t cudnnGetCTCLossDescriptor_v9(cudnnCTCLossDescriptor_t d, cudnnDataType_t* type,
                                                       cudnnLossNormalizationMode_t* norm, cudnnCTCGradMode_t* grad,
                                                       int* max_label) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const auto* c = as<const CtcDesc>(d);
  if (type) *type = c->type;
  if (norm) *norm = c->norm;
  if (grad) *grad = c->grad;
  if (max_label) *max_label = c->max_label;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetCTCLossDescriptor_v8(cudnnCTCLossDescriptor_t d, cudnnDataType_t* type,
                                                       cudnnLossNormalizationMode_t* norm, cudnnNanPropagation_t* nan,
                                                       int* max_label) {
  if (nan) *nan = CUDNN_NOT_PROPAGATE_NAN;
  return cudnnGetCTCLossDescriptor_v9(d, type, norm, nullptr, max_label);
}
VGPU_EXPORT cudnnStatus_t cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t d, cudnnDataType_t* type,
                                                      cudnnLossNormalizationMode_t* norm, cudnnNanPropagation_t* nan) {
  if (nan) *nan = CUDNN_NOT_PROPAGATE_NAN;
  return cudnnGetCTCLossDescriptor_v9(d, type, norm, nullptr, nullptr);
}
VGPU_EXPORT cudnnStatus_t cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t d, cudnnDataType_t* type) {
  return cudnnGetCTCLossDescriptor_v9(d, type, nullptr, nullptr, nullptr);
}

// Workspace: none, since the work is done on the host.
VGPU_EXPORT cudnnStatus_t cudnnGetCTCLossWorkspaceSize(cudnnHandle_t h, const cudnnTensorDescriptor_t pd,
                                                       const cudnnTensorDescriptor_t, const int*, const int*,
                                                       const int*, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t d,
                                                       size_t* size) {
  if (!known(h) || !tdesc(pd) || !known(d) || !size) return CUDNN_STATUS_BAD_PARAM;
  *size = 0;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetCTCLossWorkspaceSize_v8(cudnnHandle_t h, cudnnCTCLossAlgo_t,
                                                          cudnnCTCLossDescriptor_t d, const cudnnTensorDescriptor_t pd,
                                                          const cudnnTensorDescriptor_t, size_t* size) {
  if (!known(h) || !tdesc(pd) || !known(d) || !size) return CUDNN_STATUS_BAD_PARAM;
  *size = 0;
  return CUDNN_STATUS_SUCCESS;
}

// Labels and lengths in host memory.
VGPU_EXPORT cudnnStatus_t cudnnCTCLoss(cudnnHandle_t h, const cudnnTensorDescriptor_t pd, const void* probs,
                                       const int labels[], const int label_len[], const int input_len[], void* costs,
                                       const cudnnTensorDescriptor_t gd, void* grads, cudnnCTCLossAlgo_t algo,
                                       cudnnCTCLossDescriptor_t d, void*, size_t) {
  static const char* fn = "cudnnCTCLoss";
  const TensorDesc* P = tdesc(pd);
  if (!known(d) || !P || P->l.rank != 3 || !labels || !label_len || !input_len) return BAD(fn, "invalid descriptor or pointer");
  if (algo != CUDNN_CTC_LOSS_ALGO_DETERMINISTIC && algo != CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC)
    return BAD(fn, "unknown algorithm");
  const int N = static_cast<int>(P->l.dims[1]);
  std::vector<int> ll(label_len, label_len + N), il(input_len, input_len + N);
  size_t total = 0;
  for (int v : ll) total += v > 0 ? static_cast<size_t>(v) : 0;
  return ctc(fn, h, *as<const CtcDesc>(d), pd, probs, std::vector<int>(labels, labels + total), ll, il, costs, gd,
             grads);
}
// Labels and lengths in device memory.
VGPU_EXPORT cudnnStatus_t cudnnCTCLoss_v8(cudnnHandle_t h, cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t d,
                                          const cudnnTensorDescriptor_t pd, const void* probs, const int labels[],
                                          const int label_len[], const int input_len[], void* costs,
                                          const cudnnTensorDescriptor_t gd, void* grads, size_t, void*) {
  static const char* fn = "cudnnCTCLoss_v8";
  const TensorDesc* P = tdesc(pd);
  if (!known(h) || !known(d) || !P || P->l.rank != 3 || !labels || !label_len || !input_len)
    return BAD(fn, "invalid handle, descriptor or pointer");
  if (algo != CUDNN_CTC_LOSS_ALGO_DETERMINISTIC && algo != CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC)
    return BAD(fn, "unknown algorithm");
  sync_handle(h);
  const int N = static_cast<int>(P->l.dims[1]);
  const std::vector<int> ll = device_ints(label_len, N), il = device_ints(input_len, N);
  size_t total = 0;
  for (int v : ll) total += v > 0 ? static_cast<size_t>(v) : 0;
  return ctc(fn, h, *as<const CtcDesc>(d), pd, probs, device_ints(labels, total), ll, il, costs, gd, grads);
}

/* ---- dropout ---- */

namespace {
// One bit per element, least significant first, 1 for kept; the size rounded
// up to whole 32-bit words (measured on the hardware: 10000 elements, 1252 bytes).
size_t dropout_reserve(size_t elements) { return (elements + 31) / 32 * 4; }
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnDropoutDescriptor_t>(track(new DropoutDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete as<DropoutDesc>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t h, size_t* size) {
  if (!known(h) || !size) return CUDNN_STATUS_BAD_PARAM;
  *size = 256;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xd, size_t* size) {
  const TensorDesc* X = tdesc(xd);
  if (!X || !size) return CUDNN_STATUS_BAD_PARAM;
  *size = dropout_reserve(X->l.count());
  return CUDNN_STATUS_SUCCESS;
}
// Starts the generator over from the seed, in the states buffer if one is given.
VGPU_EXPORT cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t d, cudnnHandle_t h, float p,
                                                    void* states, size_t bytes, unsigned long long seed) {
  if (!known(d) || !(p >= 0.0f && p <= 1.0f)) return CUDNN_STATUS_BAD_PARAM;
  if (states && bytes < sizeof(DropoutState))
    return BAD("cudnnSetDropoutDescriptor", "the states buffer is smaller than cudnnDropoutGetStatesSize says");
  auto* D = as<DropoutDesc>(d);
  D->p = p, D->states = states, D->state_bytes = bytes, D->seed = seed, D->drawn = 0;
  if (states) {
    if (known(h)) sync_handle(h);
    const DropoutState st{seed, 0};
    if (cudaMemcpy(states, &st, sizeof st, cudaMemcpyHostToDevice) != cudaSuccess) return CUDNN_STATUS_EXECUTION_FAILED;
  }
  return CUDNN_STATUS_SUCCESS;
}
// Picks up the generator wherever the states buffer says it is.
VGPU_EXPORT cudnnStatus_t cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t d, cudnnHandle_t, float p,
                                                        void* states, size_t bytes, unsigned long long seed) {
  if (!known(d) || !(p >= 0.0f && p <= 1.0f)) return CUDNN_STATUS_BAD_PARAM;
  auto* D = as<DropoutDesc>(d);
  D->p = p, D->states = states, D->state_bytes = bytes, D->seed = seed, D->drawn = 0;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t d, cudnnHandle_t, float* p,
                                                    void** states, unsigned long long* seed) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const auto* D = as<const DropoutDesc>(d);
  if (p) *p = D->p;
  if (states) *states = D->states;
  if (seed) *seed = D->seed;
  return CUDNN_STATUS_SUCCESS;
}

// y = x / (1 - p) where the mask keeps it, else 0; the mask goes to the
// reserve space for the backward pass.
VGPU_EXPORT cudnnStatus_t cudnnDropoutForward(cudnnHandle_t h, const cudnnDropoutDescriptor_t dd,
                                              const cudnnTensorDescriptor_t xd, const void* x,
                                              const cudnnTensorDescriptor_t yd, void* y, void* reserve,
                                              size_t reserve_bytes) {
  static const char* fn = "cudnnDropoutForward";
  const TensorDesc *X = tdesc(xd), *Y = tdesc(yd);
  if (!known(h) || !known(dd) || !X || !Y || !x || !y || !reserve) return BAD(fn, "invalid handle, descriptor or pointer");
  if (!X->l.same_dims(Y->l)) return BAD(fn, "x and y differ in shape");
  if (!floating(X->l.type) || !floating(Y->l.type)) return UNSUPPORTED(fn, "non-floating-point data");
  const size_t n = X->l.count();
  if (reserve_bytes < dropout_reserve(n)) return BAD(fn, "the reserve space is smaller than cudnnDropoutGetReserveSpaceSize says");
  auto* D = as<DropoutDesc>(dd);
  sync_handle(h);
  std::vector<double> v;
  std::vector<uint8_t> keep;
  if (!read(X->l, x, &v) || !dropout_draw(D, n, &keep)) return CUDNN_STATUS_EXECUTION_FAILED;
  std::vector<uint8_t> mask(dropout_reserve(n), 0);
  const double p = D->p;
  for (size_t i = 0; i < n; ++i) {
    if (keep[i]) mask[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    v[i] = keep[i] ? v[i] * (1.0 / (1.0 - p)) : 0.0;
  }
  if (cudaMemcpy(reserve, mask.data(), mask.size(), cudaMemcpyHostToDevice) != cudaSuccess)
    return CUDNN_STATUS_EXECUTION_FAILED;
  return write(Y->l, y, v) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

// dx = dy / (1 - p) where the forward pass's mask kept the element, else 0.
VGPU_EXPORT cudnnStatus_t cudnnDropoutBackward(cudnnHandle_t h, const cudnnDropoutDescriptor_t dd,
                                               const cudnnTensorDescriptor_t dyd, const void* dy,
                                               const cudnnTensorDescriptor_t dxd, void* dx, void* reserve,
                                               size_t reserve_bytes) {
  static const char* fn = "cudnnDropoutBackward";
  const TensorDesc *DY = tdesc(dyd), *DX = tdesc(dxd);
  if (!known(h) || !known(dd) || !DY || !DX || !dy || !dx || !reserve)
    return BAD(fn, "invalid handle, descriptor or pointer");
  if (!DY->l.same_dims(DX->l)) return BAD(fn, "dy and dx differ in shape");
  if (!floating(DY->l.type) || !floating(DX->l.type)) return UNSUPPORTED(fn, "non-floating-point data");
  const size_t n = DY->l.count();
  if (reserve_bytes < dropout_reserve(n)) return BAD(fn, "the reserve space is smaller than the forward pass's");
  const double p = as<const DropoutDesc>(dd)->p;
  sync_handle(h);
  std::vector<uint8_t> mask(dropout_reserve(n));
  std::vector<double> v;
  if (cudaMemcpy(mask.data(), reserve, mask.size(), cudaMemcpyDeviceToHost) != cudaSuccess || !read(DY->l, dy, &v))
    return CUDNN_STATUS_EXECUTION_FAILED;
  for (size_t i = 0; i < n; ++i)
    v[i] = (mask[i / 8] >> (i % 8)) & 1 ? v[i] * (1.0 / (1.0 - p)) : 0.0;
  return write(DX->l, dx, v) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}
