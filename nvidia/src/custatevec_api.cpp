// libvgpucustatevec -- VirtualGPU's cuStateVec, presented as libcustatevec.so.1.
//
// NVIDIA's libcustatevec cannot run on a simulated GPU: it carries a
// statically linked CUDA runtime, which reaches the driver through NVIDIA's
// undocumented internal interface, so its first call (custatevecCreate) fails.
// This is the documented API implemented the way libvgpucublas implements
// cuBLAS: a library call reads its operands out of simulated device memory,
// does the arithmetic on the host in double precision, and writes the result
// back. Application kernels are simulated; vendor library calls are
// implemented.
//
// The subset is what QuEST's cuQuantum backend calls: dense and diagonal
// gates with any controls, controlled index-bit swaps, probabilities,
// projection, and Pauli-string expectation values. Anything outside it --
// a permutation in a generalized permutation matrix, a call made while the
// handle's stream is capturing a graph -- returns NOT_SUPPORTED with a
// message rather than a wrong answer.
//
// Bit conventions, from the documentation: bit k of an amplitude's index is
// index bit k. In a gate, targets[0] is the least significant bit of the
// matrix's row and column index, and a control is satisfied when its index
// bit equals controlBitValues[k] (1 when that array is null). In
// Abs2SumArray, bit k of the output's index is index bit bitOrdering[k].
#include "../include/vgpu_custatevec.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct custatevecContext {
  cudaStream_t stream = nullptr;
  custatevecDeviceMemHandler_t mem{};
  bool has_mem = false;
};

namespace {

using cd = std::complex<double>;
using Status = custatevecStatus_t;

bool quiet() { const char* q = std::getenv("VGPU_QUIET"); return q && q[0] == '1'; }

Status refuse(const char* api, const char* why) {
  if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", api, why);
  return CUSTATEVEC_STATUS_NOT_SUPPORTED;
}

bool complex_type(cudaDataType_t t) { return t == CUDA_C_64F || t == CUDA_C_32F; }

size_t elem_bytes(cudaDataType_t t) { return t == CUDA_C_64F ? 16 : 8; }

// Work issued on the handle's stream must have finished before its memory is
// read. A capturing stream has nothing to read yet, so its calls are refused.
Status settle(custatevecHandle_t h, const char* api) {
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(h->stream, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone)
    return refuse(api, "the handle's stream is capturing a graph, which this simulator's "
                       "cuStateVec does not record; call it outside capture");
  cudaGetLastError();
  return cudaStreamSynchronize(h->stream) == cudaSuccess ? CUSTATEVEC_STATUS_SUCCESS
                                                         : CUSTATEVEC_STATUS_CUDA_ERROR;
}

// Raw bytes in, whether `p` is host or device memory (cuStateVec takes gate
// matrices, diagonals and Abs2SumArray's output from either).
bool on_device(const void* p) {
  cudaPointerAttributes a{};
  const bool dev = cudaPointerGetAttributes(&a, p) == cudaSuccess &&
                   (a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged);
  cudaGetLastError();
  return dev;
}

bool read_bytes(void* dst, const void* src, size_t n) {
  if (!n) return true;
  if (on_device(src)) return cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost) == cudaSuccess;
  std::memcpy(dst, src, n);
  return true;
}

bool write_bytes(void* dst, const void* src, size_t n) {
  if (!n) return true;
  if (on_device(dst)) return cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice) == cudaSuccess;
  std::memcpy(dst, src, n);
  return true;
}

// Complex elements of either precision, widened to double.
bool read_complex(const void* p, cudaDataType_t t, size_t count, std::vector<cd>& out) {
  out.assign(count, cd{});
  if (t == CUDA_C_64F) return read_bytes(out.data(), p, count * sizeof(cd));
  std::vector<std::complex<float>> f(count);
  if (!read_bytes(f.data(), p, count * sizeof f[0])) return false;
  for (size_t i = 0; i < count; ++i) out[i] = cd(f[i].real(), f[i].imag());
  return true;
}

bool write_complex(void* p, cudaDataType_t t, const std::vector<cd>& v) {
  if (t == CUDA_C_64F) return write_bytes(p, v.data(), v.size() * sizeof(cd));
  std::vector<std::complex<float>> f(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    f[i] = std::complex<float>((float)v[i].real(), (float)v[i].imag());
  return write_bytes(p, f.data(), f.size() * sizeof f[0]);
}

// The state vector is always device memory, read and written whole.
Status load_sv(const void* sv, cudaDataType_t t, uint32_t bits, std::vector<cd>& out) {
  return read_complex(sv, t, size_t{1} << bits, out) ? CUSTATEVEC_STATUS_SUCCESS
                                                     : CUSTATEVEC_STATUS_CUDA_ERROR;
}

Status store_sv(void* sv, cudaDataType_t t, const std::vector<cd>& v) {
  return write_complex(sv, t, v) ? CUSTATEVEC_STATUS_SUCCESS : CUSTATEVEC_STATUS_CUDA_ERROR;
}

// A list of index bits: each in range, none repeated, and none already taken
// by `used` (targets, controls and masks may not share a bit).
bool take_bits(const int32_t* bits, uint32_t n, uint32_t nIndexBits, uint64_t& used, uint64_t* mask) {
  if (n && !bits) return false;
  uint64_t m = 0;
  for (uint32_t k = 0; k < n; ++k) {
    if (bits[k] < 0 || (uint32_t)bits[k] >= nIndexBits) return false;
    const uint64_t b = uint64_t{1} << bits[k];
    if ((used | m) & b) return false;
    m |= b;
  }
  used |= m;
  if (mask) *mask = m;
  return true;
}

// The value the listed bits must hold: values[k] for bits[k], 1 for every bit
// when values is null (a control's default).
uint64_t required_value(const int32_t* bits, const int32_t* values, uint32_t n) {
  uint64_t v = 0;
  for (uint32_t k = 0; k < n; ++k)
    if (!values || values[k]) v |= uint64_t{1} << bits[k];
  return v;
}

// Index bits bits[0..n) of i, packed into the low n bits (bits[0] lowest).
uint64_t gather(uint64_t i, const int32_t* bits, uint32_t n) {
  uint64_t k = 0;
  for (uint32_t j = 0; j < n; ++j) k |= ((i >> bits[j]) & 1) << j;
  return k;
}

// The inverse: the low n bits of k placed at index bits bits[0..n).
uint64_t scatter(uint64_t k, const int32_t* bits, uint32_t n) {
  uint64_t i = 0;
  for (uint32_t j = 0; j < n; ++j) i |= ((k >> j) & 1) << bits[j];
  return i;
}

bool sane_bits(uint32_t nIndexBits) { return nIndexBits >= 1 && nIndexBits <= 40; }

}  // namespace

extern "C" {

custatevecStatus_t custatevecCreate(custatevecHandle_t* handle) {
  if (!handle) return CUSTATEVEC_STATUS_INVALID_VALUE;
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    cudaGetLastError();
    return CUSTATEVEC_STATUS_NOT_SUPPORTED;
  }
  *handle = new custatevecContext{};
  return CUSTATEVEC_STATUS_SUCCESS;
}

custatevecStatus_t custatevecDestroy(custatevecHandle_t handle) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  delete handle;
  return CUSTATEVEC_STATUS_SUCCESS;
}

const char* custatevecGetErrorName(custatevecStatus_t status) {
  switch (status) {
    case CUSTATEVEC_STATUS_SUCCESS: return "CUSTATEVEC_STATUS_SUCCESS";
    case CUSTATEVEC_STATUS_NOT_INITIALIZED: return "CUSTATEVEC_STATUS_NOT_INITIALIZED";
    case CUSTATEVEC_STATUS_ALLOC_FAILED: return "CUSTATEVEC_STATUS_ALLOC_FAILED";
    case CUSTATEVEC_STATUS_INVALID_VALUE: return "CUSTATEVEC_STATUS_INVALID_VALUE";
    case CUSTATEVEC_STATUS_ARCH_MISMATCH: return "CUSTATEVEC_STATUS_ARCH_MISMATCH";
    case CUSTATEVEC_STATUS_EXECUTION_FAILED: return "CUSTATEVEC_STATUS_EXECUTION_FAILED";
    case CUSTATEVEC_STATUS_INTERNAL_ERROR: return "CUSTATEVEC_STATUS_INTERNAL_ERROR";
    case CUSTATEVEC_STATUS_NOT_SUPPORTED: return "CUSTATEVEC_STATUS_NOT_SUPPORTED";
    case CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE: return "CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE";
    case CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED: return "CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED";
    case CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR: return "CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR";
    case CUSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR: return "CUSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR";
    case CUSTATEVEC_STATUS_COMMUNICATOR_ERROR: return "CUSTATEVEC_STATUS_COMMUNICATOR_ERROR";
    case CUSTATEVEC_STATUS_LOADING_LIBRARY_FAILED: return "CUSTATEVEC_STATUS_LOADING_LIBRARY_FAILED";
    case CUSTATEVEC_STATUS_INVALID_CONFIGURATION: return "CUSTATEVEC_STATUS_INVALID_CONFIGURATION";
    case CUSTATEVEC_STATUS_ALREADY_INITIALIZED: return "CUSTATEVEC_STATUS_ALREADY_INITIALIZED";
    case CUSTATEVEC_STATUS_INVALID_WIRE: return "CUSTATEVEC_STATUS_INVALID_WIRE";
    case CUSTATEVEC_STATUS_SYSTEM_ERROR: return "CUSTATEVEC_STATUS_SYSTEM_ERROR";
    case CUSTATEVEC_STATUS_CUDA_ERROR: return "CUSTATEVEC_STATUS_CUDA_ERROR";
    case CUSTATEVEC_STATUS_NUMERICAL_ERROR: return "CUSTATEVEC_STATUS_NUMERICAL_ERROR";
    case CUSTATEVEC_STATUS_RESOURCES_NOT_ACCESSIBLE: return "CUSTATEVEC_STATUS_RESOURCES_NOT_ACCESSIBLE";
    default: return "CUSTATEVEC_STATUS_UNKNOWN";
  }
}

const char* custatevecGetErrorString(custatevecStatus_t status) {
  switch (status) {
    case CUSTATEVEC_STATUS_SUCCESS: return "success";
    case CUSTATEVEC_STATUS_NOT_INITIALIZED: return "the library handle was not initialized";
    case CUSTATEVEC_STATUS_INVALID_VALUE: return "invalid value";
    case CUSTATEVEC_STATUS_NOT_SUPPORTED: return "not supported";
    case CUSTATEVEC_STATUS_CUDA_ERROR: return "a CUDA call failed";
    default: return custatevecGetErrorName(status);
  }
}

custatevecStatus_t custatevecGetProperty(libraryPropertyType type, int32_t* value) {
  if (!value) return CUSTATEVEC_STATUS_INVALID_VALUE;
  switch (type) {
    case MAJOR_VERSION: *value = CUSTATEVEC_VER_MAJOR; return CUSTATEVEC_STATUS_SUCCESS;
    case MINOR_VERSION: *value = CUSTATEVEC_VER_MINOR; return CUSTATEVEC_STATUS_SUCCESS;
    case PATCH_LEVEL: *value = CUSTATEVEC_VER_PATCH; return CUSTATEVEC_STATUS_SUCCESS;
    default: return CUSTATEVEC_STATUS_INVALID_VALUE;
  }
}

size_t custatevecGetVersion(void) { return CUSTATEVEC_VERSION; }

custatevecStatus_t custatevecSetStream(custatevecHandle_t handle, cudaStream_t streamId) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  handle->stream = streamId;
  return CUSTATEVEC_STATUS_SUCCESS;
}

custatevecStatus_t custatevecGetStream(custatevecHandle_t handle, cudaStream_t* streamId) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  if (!streamId) return CUSTATEVEC_STATUS_INVALID_VALUE;
  *streamId = handle->stream;
  return CUSTATEVEC_STATUS_SUCCESS;
}

// The handler is kept and handed back. This implementation needs no device
// workspace, so it never allocates through it.
custatevecStatus_t custatevecSetDeviceMemHandler(custatevecHandle_t handle,
                                                 const custatevecDeviceMemHandler_t* handler) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  if (!handler || !handler->device_alloc || !handler->device_free) return CUSTATEVEC_STATUS_INVALID_VALUE;
  handle->mem = *handler;
  handle->has_mem = true;
  return CUSTATEVEC_STATUS_SUCCESS;
}

custatevecStatus_t custatevecGetDeviceMemHandler(custatevecHandle_t handle,
                                                 custatevecDeviceMemHandler_t* handler) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  if (!handler) return CUSTATEVEC_STATUS_INVALID_VALUE;
  if (!handle->has_mem) return CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR;
  *handler = handle->mem;
  return CUSTATEVEC_STATUS_SUCCESS;
}

custatevecStatus_t custatevecApplyMatrixGetWorkspaceSize(
    custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits,
    const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout,
    const int32_t, const uint32_t nTargets, const uint32_t, custatevecComputeType_t,
    size_t* extraWorkspaceSizeInBytes) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  if (!extraWorkspaceSizeInBytes || !matrix || !complex_type(svDataType) ||
      !complex_type(matrixDataType) || !sane_bits(nIndexBits) || nTargets < 1 ||
      nTargets > nIndexBits ||
      (layout != CUSTATEVEC_MATRIX_LAYOUT_ROW && layout != CUSTATEVEC_MATRIX_LAYOUT_COL))
    return CUSTATEVEC_STATUS_INVALID_VALUE;
  *extraWorkspaceSizeInBytes = 0;
  return CUSTATEVEC_STATUS_SUCCESS;
}

custatevecStatus_t custatevecApplyMatrix(
    custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits,
    const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout,
    const int32_t adjoint, const int32_t* targets, const uint32_t nTargets,
    const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls,
    custatevecComputeType_t, void*, size_t) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  uint64_t used = 0, tmask = 0, cmask = 0;
  if (!sv || !matrix || !complex_type(svDataType) || !complex_type(matrixDataType) ||
      !sane_bits(nIndexBits) || nTargets < 1 ||
      (layout != CUSTATEVEC_MATRIX_LAYOUT_ROW && layout != CUSTATEVEC_MATRIX_LAYOUT_COL) ||
      !take_bits(targets, nTargets, nIndexBits, used, &tmask) ||
      !take_bits(controls, nControls, nIndexBits, used, &cmask))
    return CUSTATEVEC_STATUS_INVALID_VALUE;
  if (Status s = settle(handle, "custatevecApplyMatrix")) return s;

  const size_t dim = size_t{1} << nTargets;
  std::vector<cd> m;
  if (!read_complex(matrix, matrixDataType, dim * dim, m)) return CUSTATEVEC_STATUS_CUDA_ERROR;
  // M[r][c], row-major, after the layout and the adjoint are applied.
  std::vector<cd> M(dim * dim);
  for (size_t r = 0; r < dim; ++r)
    for (size_t c = 0; c < dim; ++c) {
      const cd e = layout == CUSTATEVEC_MATRIX_LAYOUT_ROW ? m[r * dim + c] : m[c * dim + r];
      if (adjoint) M[c * dim + r] = std::conj(e);
      else M[r * dim + c] = e;
    }

  std::vector<cd> a;
  if (Status s = load_sv(sv, svDataType, nIndexBits, a)) return s;
  const uint64_t cval = required_value(controls, controlBitValues, nControls);
  std::vector<uint64_t> offset(dim);
  for (size_t k = 0; k < dim; ++k) offset[k] = scatter(k, targets, nTargets);
  std::vector<cd> in(dim);
  const uint64_t n = uint64_t{1} << nIndexBits;
  for (uint64_t base = 0; base < n; ++base) {
    if ((base & tmask) || (base & cmask) != cval) continue;
    for (size_t k = 0; k < dim; ++k) in[k] = a[base | offset[k]];
    for (size_t r = 0; r < dim; ++r) {
      cd acc = 0;
      for (size_t c = 0; c < dim; ++c) acc += M[r * dim + c] * in[c];
      a[base | offset[r]] = acc;
    }
  }
  return store_sv(sv, svDataType, a);
}

custatevecStatus_t custatevecApplyGeneralizedPermutationMatrix(
    custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits,
    custatevecIndex_t* permutation, const void* diagonals, cudaDataType_t diagonalsDataType,
    const int32_t adjoint, const int32_t* targets, const uint32_t nTargets,
    const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls,
    void*, size_t) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  uint64_t used = 0, tmask = 0, cmask = 0;
  if (!sv || !complex_type(svDataType) || !sane_bits(nIndexBits) || nTargets < 1 ||
      (diagonals && !complex_type(diagonalsDataType)) ||
      !take_bits(targets, nTargets, nIndexBits, used, &tmask) ||
      !take_bits(controls, nControls, nIndexBits, used, &cmask))
    return CUSTATEVEC_STATUS_INVALID_VALUE;
  if (permutation)
    return refuse("custatevecApplyGeneralizedPermutationMatrix",
                  "a permutation table is not supported by this simulator's cuStateVec; "
                  "a diagonal (permutation = NULL) is");
  if (!diagonals) return CUSTATEVEC_STATUS_SUCCESS;  // the identity
  if (Status s = settle(handle, "custatevecApplyGeneralizedPermutationMatrix")) return s;

  const size_t dim = size_t{1} << nTargets;
  std::vector<cd> d;
  if (!read_complex(diagonals, diagonalsDataType, dim, d)) return CUSTATEVEC_STATUS_CUDA_ERROR;
  if (adjoint)
    for (cd& e : d) e = std::conj(e);
  std::vector<cd> a;
  if (Status s = load_sv(sv, svDataType, nIndexBits, a)) return s;
  const uint64_t cval = required_value(controls, controlBitValues, nControls);
  const uint64_t n = uint64_t{1} << nIndexBits;
  for (uint64_t i = 0; i < n; ++i)
    if ((i & cmask) == cval) a[i] *= d[gather(i, targets, nTargets)];
  (void)tmask;
  return store_sv(sv, svDataType, a);
}

custatevecStatus_t custatevecSwapIndexBits(
    custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits,
    const int2* bitSwaps, const uint32_t nBitSwaps, const int32_t* maskBitString,
    const int32_t* maskOrdering, const uint32_t maskLen) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  uint64_t used = 0, swapped = 0, mmask = 0;
  if (!sv || !complex_type(svDataType) || !sane_bits(nIndexBits) || (nBitSwaps && !bitSwaps) ||
      (maskLen && !maskBitString))
    return CUSTATEVEC_STATUS_INVALID_VALUE;
  for (uint32_t k = 0; k < nBitSwaps; ++k) {
    const int x = bitSwaps[k].x, y = bitSwaps[k].y;
    if (x < 0 || y < 0 || (uint32_t)x >= nIndexBits || (uint32_t)y >= nIndexBits || x == y)
      return CUSTATEVEC_STATUS_INVALID_VALUE;
    swapped |= (uint64_t{1} << x) | (uint64_t{1} << y);
  }
  used = swapped;
  if (!take_bits(maskOrdering, maskLen, nIndexBits, used, &mmask)) return CUSTATEVEC_STATUS_INVALID_VALUE;
  if (Status s = settle(handle, "custatevecSwapIndexBits")) return s;

  std::vector<cd> a;
  if (Status s = load_sv(sv, svDataType, nIndexBits, a)) return s;
  const uint64_t mval = required_value(maskOrdering, maskBitString, maskLen);
  std::vector<cd> out = a;
  const uint64_t n = uint64_t{1} << nIndexBits;
  for (uint64_t i = 0; i < n; ++i) {
    if ((i & mmask) != mval) continue;
    uint64_t j = i;
    for (uint32_t k = 0; k < nBitSwaps; ++k) {
      const uint64_t bx = (j >> bitSwaps[k].x) & 1, by = (j >> bitSwaps[k].y) & 1;
      if (bx != by) j ^= (uint64_t{1} << bitSwaps[k].x) | (uint64_t{1} << bitSwaps[k].y);
    }
    out[j] = a[i];
  }
  return store_sv(sv, svDataType, out);
}

custatevecStatus_t custatevecAbs2SumArray(
    custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType,
    const uint32_t nIndexBits, double* abs2sum, const int32_t* bitOrdering,
    const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering,
    const uint32_t maskLen) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  uint64_t used = 0, mmask = 0;
  if (!sv || !abs2sum || !complex_type(svDataType) || !sane_bits(nIndexBits) ||
      (maskLen && !maskBitString) || !take_bits(bitOrdering, bitOrderingLen, nIndexBits, used, nullptr) ||
      !take_bits(maskOrdering, maskLen, nIndexBits, used, &mmask))
    return CUSTATEVEC_STATUS_INVALID_VALUE;
  if (Status s = settle(handle, "custatevecAbs2SumArray")) return s;

  std::vector<cd> a;
  if (Status s = load_sv(sv, svDataType, nIndexBits, a)) return s;
  const uint64_t mval = required_value(maskOrdering, maskBitString, maskLen);
  std::vector<double> out(size_t{1} << bitOrderingLen, 0.0);
  const uint64_t n = uint64_t{1} << nIndexBits;
  for (uint64_t i = 0; i < n; ++i)
    if ((i & mmask) == mval) out[gather(i, bitOrdering, bitOrderingLen)] += std::norm(a[i]);
  return write_bytes(abs2sum, out.data(), out.size() * sizeof(double)) ? CUSTATEVEC_STATUS_SUCCESS
                                                                       : CUSTATEVEC_STATUS_CUDA_ERROR;
}

custatevecStatus_t custatevecAbs2SumOnZBasis(
    custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType,
    const uint32_t nIndexBits, double* abs2sum0, double* abs2sum1, const int32_t* basisBits,
    const uint32_t nBasisBits) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  uint64_t used = 0, zmask = 0;
  if (!sv || (!abs2sum0 && !abs2sum1) || !complex_type(svDataType) || !sane_bits(nIndexBits) ||
      nBasisBits < 1 || !take_bits(basisBits, nBasisBits, nIndexBits, used, &zmask))
    return CUSTATEVEC_STATUS_INVALID_VALUE;
  if (Status s = settle(handle, "custatevecAbs2SumOnZBasis")) return s;

  std::vector<cd> a;
  if (Status s = load_sv(sv, svDataType, nIndexBits, a)) return s;
  double sum[2] = {0, 0};
  const uint64_t n = uint64_t{1} << nIndexBits;
  for (uint64_t i = 0; i < n; ++i) sum[__builtin_parityll(i & zmask)] += std::norm(a[i]);
  if (abs2sum0) *abs2sum0 = sum[0];
  if (abs2sum1) *abs2sum1 = sum[1];
  return CUSTATEVEC_STATUS_SUCCESS;
}

custatevecStatus_t custatevecCollapseByBitString(
    custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits,
    const int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen,
    double norm) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  uint64_t used = 0, bmask = 0;
  if (!sv || !complex_type(svDataType) || !sane_bits(nIndexBits) || (bitStringLen && !bitString) ||
      !take_bits(bitOrdering, bitStringLen, nIndexBits, used, &bmask) || !(norm > 0))
    return CUSTATEVEC_STATUS_INVALID_VALUE;
  if (Status s = settle(handle, "custatevecCollapseByBitString")) return s;

  std::vector<cd> a;
  if (Status s = load_sv(sv, svDataType, nIndexBits, a)) return s;
  const uint64_t want = required_value(bitOrdering, bitString, bitStringLen);
  const double scale = 1.0 / std::sqrt(norm);
  const uint64_t n = uint64_t{1} << nIndexBits;
  for (uint64_t i = 0; i < n; ++i) a[i] = (i & bmask) == want ? a[i] * scale : cd{};
  return store_sv(sv, svDataType, a);
}

// <psi| P |psi> for each Pauli string P. P|i> = phase(i) |i ^ flip>, where
// flip has the X and Y bits; Z contributes (-1)^bit and Y contributes
// i * (-1)^bit (Y|0> = i|1>, Y|1> = -i|0>).
custatevecStatus_t custatevecComputeExpectationsOnPauliBasis(
    custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType,
    const uint32_t nIndexBits, double* expectationValues,
    const custatevecPauli_t** pauliOperatorsArray, const uint32_t nPauliOperatorArrays,
    const int32_t** basisBitsArray, const uint32_t* nBasisBitsArray) {
  if (!handle) return CUSTATEVEC_STATUS_NOT_INITIALIZED;
  if (!sv || !expectationValues || !complex_type(svDataType) || !sane_bits(nIndexBits) ||
      (nPauliOperatorArrays && (!pauliOperatorsArray || !basisBitsArray || !nBasisBitsArray)))
    return CUSTATEVEC_STATUS_INVALID_VALUE;
  struct Term { uint64_t flip = 0, zmask = 0, ymask = 0; };
  std::vector<Term> terms(nPauliOperatorArrays);
  for (uint32_t t = 0; t < nPauliOperatorArrays; ++t) {
    const uint32_t len = nBasisBitsArray[t];
    const custatevecPauli_t* ops = pauliOperatorsArray[t];
    const int32_t* bits = basisBitsArray[t];
    uint64_t used = 0;
    if (len && !ops) return CUSTATEVEC_STATUS_INVALID_VALUE;
    if (!take_bits(bits, len, nIndexBits, used, nullptr)) return CUSTATEVEC_STATUS_INVALID_VALUE;
    for (uint32_t k = 0; k < len; ++k) {
      const uint64_t b = uint64_t{1} << bits[k];
      switch (ops[k]) {
        case CUSTATEVEC_PAULI_I: break;
        case CUSTATEVEC_PAULI_X: terms[t].flip |= b; break;
        case CUSTATEVEC_PAULI_Y: terms[t].flip |= b; terms[t].ymask |= b; break;
        case CUSTATEVEC_PAULI_Z: terms[t].zmask |= b; break;
        default: return CUSTATEVEC_STATUS_INVALID_VALUE;
      }
    }
  }
  if (Status s = settle(handle, "custatevecComputeExpectationsOnPauliBasis")) return s;

  std::vector<cd> a;
  if (Status s = load_sv(sv, svDataType, nIndexBits, a)) return s;
  const uint64_t n = uint64_t{1} << nIndexBits;
  static const cd kIpow[4] = {cd(1, 0), cd(0, 1), cd(-1, 0), cd(0, -1)};
  for (uint32_t t = 0; t < nPauliOperatorArrays; ++t) {
    const Term& T = terms[t];
    const int ny = __builtin_popcountll(T.ymask);
    cd acc = 0;
    for (uint64_t i = 0; i < n; ++i) {
      // Each Z and each Y on a set bit contributes -1; each Y contributes i.
      const int sign = __builtin_parityll(i & (T.zmask | T.ymask)) ? -1 : 1;
      acc += std::conj(a[i ^ T.flip]) * (double)sign * kIpow[ny & 3] * a[i];
    }
    expectationValues[t] = acc.real();
  }
  return CUSTATEVEC_STATUS_SUCCESS;
}

}  // extern "C"
