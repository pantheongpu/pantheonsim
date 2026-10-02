// libvgpucudnn's backend (graph) API: cudnnBackendCreateDescriptor, Set/Get
// Attribute, Finalize, Execute -- what cuDNN 8 and 9 build every convolution
// out of, and what PyTorch's cuDNN path (through cudnn-frontend) calls for
// each one.
//
// A program describes its work as descriptors: tensors, a convolution, an
// operation on them, a graph of operations. It asks the heuristics for
// engine configurations, builds an execution plan from one, and executes the
// plan with a variant pack that says where each tensor is. Here:
//
//   - Every descriptor keeps its attributes as it was given them, and hands
//     them back on GetAttribute, so a caller reads what it wrote.
//   - One engine, global index 0, runs any graph this library can compute:
//     any number of convolution (forward, backward-data, backward-filter),
//     matmul, pointwise, reduction, normalization (layer, instance, batch,
//     RMS; forward and backward), pooling (resampling: max and average,
//     forward and backward) and concatenation operations, joined through
//     virtual tensors -- cudnn-frontend's conv-bias-activation, dgrad-drelu,
//     matmul-epilogue, reduction, norm and pooling patterns, and PyTorch's
//     single convolutions. A graph with any other operation is refused when
//     it is finalized, by name, rather than accepted and run wrongly.
//   - Execute computes on the host, as the classic API here does (see
//     cudnn_api.cpp): the operations run in an order that respects their
//     data, inputs copied out of device memory and outputs copied back after
//     the handle's stream has finished what it was given; intermediates stay
//     on the host, rounded to their declared types. Convolutions take one to
//     three spatial dimensions, groups, dilation, padding and any strides;
//     tensors are float, half, bfloat16, double or integer, and a HALF
//     compute type accumulates in half precision, as the hardware does.
//
// Declarations and values are cuDNN's own headers (nvidia/third_party/
// cudnn_include); nothing here is NVIDIA's code.
#include "cudnn_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}
bool trace() {
  const char* t = std::getenv("VGPU_TRACE");
  return t && t[0] == '1';
}

// The size of one element of each attribute type, as cuDNN's header defines
// the types: enums are ints, BOOLEAN is a C++ bool, FRACTION two int64s.
size_t elem_size(cudnnBackendAttributeType_t t) {
  switch (t) {
    case CUDNN_TYPE_HANDLE:
    case CUDNN_TYPE_VOID_PTR:
    case CUDNN_TYPE_BACKEND_DESCRIPTOR: return sizeof(void*);
    case CUDNN_TYPE_BOOLEAN: return sizeof(bool);
    case CUDNN_TYPE_INT64: return sizeof(int64_t);
    case CUDNN_TYPE_FLOAT: return sizeof(float);
    case CUDNN_TYPE_DOUBLE: return sizeof(double);
    case CUDNN_TYPE_CHAR: return sizeof(char);
    case CUDNN_TYPE_FRACTION: return sizeof(cudnnFraction_t);
    default: return sizeof(int32_t);  // the enums, and INT32
  }
}

struct Attr {
  cudnnBackendAttributeType_t type = CUDNN_TYPE_INT64;
  int64_t count = 0;
  std::vector<uint8_t> bytes;
};

struct Desc {
  cudnnBackendDescriptorType_t type;
  bool finalized = false;
  std::map<cudnnBackendAttributeName_t, Attr> attrs;
  // Descriptors this one made itself and hands out by copy (an engine
  // configuration's engine): kept for as long as this one lives.
  std::vector<std::unique_ptr<Desc>> owned;

  const Attr* get(cudnnBackendAttributeName_t n) const {
    auto it = attrs.find(n);
    return it == attrs.end() ? nullptr : &it->second;
  }
  int64_t i64(cudnnBackendAttributeName_t n, int64_t dflt = 0) const {
    const Attr* a = get(n);
    if (!a || a->count < 1) return dflt;
    int64_t v = 0;
    std::memcpy(&v, a->bytes.data(), std::min<size_t>(a->bytes.size(), sizeof v));
    if (a->type != CUDNN_TYPE_INT64) v = static_cast<int32_t>(v);  // an enum or INT32
    return v;
  }
  std::vector<int64_t> i64s(cudnnBackendAttributeName_t n) const {
    const Attr* a = get(n);
    std::vector<int64_t> v;
    if (!a) return v;
    v.resize(static_cast<size_t>(a->count));
    if (a->type == CUDNN_TYPE_INT64) std::memcpy(v.data(), a->bytes.data(), v.size() * 8);
    return v;
  }
  Desc* desc(cudnnBackendAttributeName_t n, size_t i = 0) const {
    const Attr* a = get(n);
    if (!a || a->type != CUDNN_TYPE_BACKEND_DESCRIPTOR || static_cast<size_t>(a->count) <= i) return nullptr;
    Desc* d = nullptr;
    std::memcpy(&d, a->bytes.data() + i * sizeof(void*), sizeof(void*));
    return d;
  }
  void set(cudnnBackendAttributeName_t n, cudnnBackendAttributeType_t t, int64_t count, const void* p) {
    Attr& a = attrs[n];
    a.type = t;
    a.count = count;
    a.bytes.assign(static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + elem_size(t) * count);
  }
};

// A descriptor stored in another's attribute is copied, as cuDNN copies it:
// cudnn-frontend destroys its tensor and operation descriptors as soon as the
// graph that uses them is built, so a pointer kept to the caller's would
// dangle. The copy is deep and owned by the descriptor that holds it.
std::unique_ptr<Desc> clone(const Desc* src);

void assign(Desc* dst, const Desc* src) {
  dst->type = src->type;
  dst->finalized = src->finalized;
  dst->attrs.clear();
  std::vector<std::unique_ptr<Desc>> keep;
  keep.swap(dst->owned);  // src may be one of dst's own children
  for (const auto& [name, a] : src->attrs) {
    Attr copy = a;
    if (a.type == CUDNN_TYPE_BACKEND_DESCRIPTOR) {
      for (int64_t i = 0; i < a.count; ++i) {
        Desc* child = nullptr;
        std::memcpy(&child, a.bytes.data() + i * sizeof(void*), sizeof(void*));
        Desc* mine = nullptr;
        if (child) {
          dst->owned.push_back(clone(child));
          mine = dst->owned.back().get();
        }
        std::memcpy(copy.bytes.data() + i * sizeof(void*), &mine, sizeof(void*));
      }
    }
    dst->attrs[name] = std::move(copy);
  }
}

std::unique_ptr<Desc> clone(const Desc* src) {
  auto d = std::make_unique<Desc>();
  assign(d.get(), src);
  return d;
}

std::mutex g_mu;
std::set<const Desc*> g_live;
bool live(const void* p) {
  std::lock_guard<std::mutex> l(g_mu);
  return p && g_live.count(static_cast<const Desc*>(p));
}

cudnnStatus_t refuse(const char* what, const std::string& why) {
  if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", what, why.c_str());
  return CUDNN_STATUS_NOT_SUPPORTED;
}

// ---- the graph: tensors, operations, and running them ------------------------

namespace vc = vgpu_cudnn;

// A tensor as an operation names it. Virtual tensors live only inside the
// graph (here, on the host between operations); by-value tensors are scalars
// whose variant-pack pointer is host memory.
struct GTensor {
  vc::Layout l;  // logical dims (a vectorized dimension counts elements, not vectors) and strides
  int64_t uid = 0;
  bool is_virtual = false, by_value = false;
  // A vectorized tensor (CUDNN_ATTR_TENSOR_VECTOR_COUNT): dimension vdim
  // holds vectors of vcount elements, each vector contiguous, and the
  // descriptor's dimensions and strides count vectors -- the graph API's form
  // of NCHW_VECT_C (INT8x4, INT8x32). l has the element count; mem_dims and
  // mem_strides are the descriptor's own.
  int64_t vcount = 1;
  int vdim = -1;
  int64_t mem_dims[vc::kMaxRank] = {}, mem_strides[vc::kMaxRank] = {};
  // A ragged tensor (CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC): batch b's first
  // element is at offset[b] * mult elements rather than b * strides[0] --
  // packed variable-length sequences (THD) in attention.
  std::shared_ptr<GTensor> ragged;
  int64_t ragged_mult = 1;
  // A by-value tensor's compile-time constant (CUDNN_ATTR_TENSOR_CONSTANT_VALUE),
  // used when the variant pack gives it no pointer.
  std::vector<uint8_t> constant;
};

bool tensor_of(const Desc* d, GTensor* t, std::string* why) {
  if (!d || d->type != CUDNN_BACKEND_TENSOR_DESCRIPTOR) { *why = "a tensor is missing"; return false; }
  const std::vector<int64_t> dims = d->i64s(CUDNN_ATTR_TENSOR_DIMENSIONS), strides = d->i64s(CUDNN_ATTR_TENSOR_STRIDES);
  t->uid = d->i64(CUDNN_ATTR_TENSOR_UNIQUE_ID);
  t->l.type = static_cast<cudnnDataType_t>(d->i64(CUDNN_ATTR_TENSOR_DATA_TYPE));
  const Attr* v = d->get(CUDNN_ATTR_TENSOR_IS_VIRTUAL);
  t->is_virtual = v && !v->bytes.empty() && v->bytes[0];
  const Attr* bv = d->get(CUDNN_ATTR_TENSOR_IS_BY_VALUE);
  t->by_value = bv && !bv->bytes.empty() && bv->bytes[0];
  if (dims.empty() || dims.size() != strides.size() || dims.size() > static_cast<size_t>(vc::kMaxRank)) {
    *why = "tensor " + std::to_string(t->uid) + " has " + std::to_string(dims.size()) + " dimensions and " +
           std::to_string(strides.size()) + " strides";
    return false;
  }
  t->vcount = d->i64(CUDNN_ATTR_TENSOR_VECTOR_COUNT, 1);
  if (t->vcount != 1) {
    t->vdim = static_cast<int>(d->i64(CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION, -1));
    if (t->vcount < 1 || t->vdim < 0 || t->vdim >= static_cast<int>(dims.size()) || t->is_virtual) {
      *why = "tensor " + std::to_string(t->uid) + ": a vectorized tensor needs a vectorized dimension inside it, and "
             "cannot be virtual";
      return false;
    }
  }
  if (const Desc* rd = d->desc(CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC)) {
    t->ragged = std::make_shared<GTensor>();
    std::string w;
    if (!tensor_of(rd, t->ragged.get(), &w) || t->ragged->ragged || t->ragged->is_virtual ||
        (t->ragged->l.type != CUDNN_DATA_INT32 && t->ragged->l.type != CUDNN_DATA_INT64) ||
        t->ragged->l.count() < static_cast<size_t>(dims[0]) + 1) {
      *why = "tensor " + std::to_string(t->uid) + ": its ragged offsets must be an INT32 or INT64 device tensor with " +
             "one more element than the first dimension" + (w.empty() ? "" : " (" + w + ")");
      return false;
    }
    t->ragged_mult = d->i64(CUDNN_ATTR_TENSOR_RAGGED_OFFSET_MULTIPLIER, 1);
    if (t->ragged_mult < 1) { *why = "tensor " + std::to_string(t->uid) + ": the ragged offset multiplier is below 1"; return false; }
  }
  if (const Attr* cv = d->get(CUDNN_ATTR_TENSOR_CONSTANT_VALUE)) t->constant = cv->bytes;
  // A reordered filter (INT8x32's interleaving for IMMA) is in a layout cuDNN
  // does not document.
  if (d->i64(CUDNN_ATTR_TENSOR_REORDERING_MODE, CUDNN_TENSOR_REORDERING_NONE) != CUDNN_TENSOR_REORDERING_NONE) {
    *why = "tensor " + std::to_string(t->uid) + " is reordered (INT8x32 or F16x16 interleaving), which is not supported";
    return false;
  }
  if (!vc::storable(t->l.type)) {
    *why = std::string("tensor ") + std::to_string(t->uid) + " has data type " + vc::type_name(t->l.type) +
           ", which is not supported";
    return false;
  }
  t->l.rank = static_cast<int>(dims.size());
  for (int i = 0; i < t->l.rank; ++i) {
    // A stride of 0 broadcasts (a by-value or [1..] scalar may say so).
    if (dims[i] <= 0 || strides[i] < 0) { *why = "tensor " + std::to_string(t->uid) + " has a nonpositive extent or negative stride"; return false; }
    t->mem_dims[i] = dims[i], t->mem_strides[i] = strides[i];
    t->l.dims[i] = dims[i] * (i == t->vdim ? t->vcount : 1), t->l.strides[i] = strides[i];
  }
  if (t->vcount != 1) vc::packed_strides(t->l.rank, t->l.dims, false, t->l.strides);
  return true;
}

// Each logical element's offset, in elements, from the tensor's base pointer:
// strided, vectorized or ragged (offs: the ragged offsets, read already).
//
// A ragged tensor's elements past its last offset (the end of the packed
// data) are -1: not read (0) and not written. Others past a batch's own
// length land in the next batch's rows, which come later in logical order
// and so overwrite them, as the hardware's writes of only the valid rows
// leave them.
std::vector<int64_t> element_offsets(const GTensor& t, const std::vector<double>* offs = nullptr) {
  const vc::Layout& L = t.l;
  std::vector<int64_t> o(L.count());
  const int64_t end = offs && offs->size() > static_cast<size_t>(L.dims[0])
                          ? static_cast<int64_t>((*offs)[static_cast<size_t>(L.dims[0])]) * t.ragged_mult : -1;
  int64_t at[vc::kMaxRank] = {};
  for (size_t i = 0; i < o.size(); ++i) {
    int64_t off = 0;
    for (int d = 0; d < L.rank; ++d) {
      int64_t idx = at[d];
      if (d == t.vdim) {
        off += (idx / t.vcount) * t.mem_strides[d] * t.vcount + idx % t.vcount;
        continue;
      }
      if (d == 0 && offs) { off += static_cast<int64_t>((*offs)[static_cast<size_t>(idx)]) * t.ragged_mult; continue; }
      off += idx * t.mem_strides[d] * (t.vdim >= 0 ? t.vcount : 1);
    }
    o[i] = end >= 0 && off >= end ? -1 : off;
    for (int d = L.rank; d-- > 0;) {
      if (++at[d] < L.dims[d]) break;
      at[d] = 0;
    }
  }
  return o;
}

// Reads (gathers) or writes (scatters) a tensor at the given element offsets.
bool gather(cudnnDataType_t type, const void* dev, const std::vector<int64_t>& offs, std::vector<double>* out) {
  out->assign(offs.size(), 0.0);
  if (offs.empty()) return true;
  if (!dev) return false;
  const size_t eb = vc::type_bytes(type);
  int64_t hi = 0;
  for (int64_t o : offs) hi = std::max(hi, o + 1);
  std::vector<uint8_t> raw(static_cast<size_t>(hi) * eb);
  if (hi && cudaMemcpy(raw.data(), dev, raw.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
  for (size_t i = 0; i < offs.size(); ++i)
    if (offs[i] >= 0) (*out)[i] = vc::decode(type, raw.data() + offs[i] * eb);
  return true;
}
bool scatter(cudnnDataType_t type, void* dev, const std::vector<int64_t>& offs, const std::vector<double>& v) {
  if (offs.empty()) return true;
  if (!dev || v.size() < offs.size()) return false;
  const size_t eb = vc::type_bytes(type);
  int64_t hi = 0;
  for (int64_t o : offs) hi = std::max(hi, o + 1);
  if (!hi) return true;
  std::vector<uint8_t> raw(static_cast<size_t>(hi) * eb);
  // What lies between the elements belongs to someone else: keep it.
  if (cudaMemcpy(raw.data(), dev, raw.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
  for (size_t i = 0; i < offs.size(); ++i)
    if (offs[i] >= 0) vc::encode(type, v[i], raw.data() + offs[i] * eb);
  return cudaMemcpy(dev, raw.data(), raw.size(), cudaMemcpyHostToDevice) == cudaSuccess;
}

double scalar(const Desc* op, cudnnBackendAttributeName_t n, double dflt) {
  const Attr* a = op->get(n);
  if (!a || a->count < 1) return dflt;
  if (a->type == CUDNN_TYPE_DOUBLE) { double d; std::memcpy(&d, a->bytes.data(), 8); return d; }
  if (a->type == CUDNN_TYPE_FLOAT) { float f; std::memcpy(&f, a->bytes.data(), 4); return f; }
  return dflt;
}

// What an operation is, which tensors it reads and writes, and its settings,
// checked: everything a graph's finalization needs to accept it and its
// execution needs to run it.
enum class Kind { ConvFwd, ConvData, ConvFilter, Matmul, Pointwise, Reduction, NormFwd, NormBwd, PoolFwd, PoolBwd, Concat,
                  Reshape, Transpose, Slice, Rng, GenStats, Softmax, BandMask, SdpaFwd, SdpaBwd, PagedLoad };

// An operation's tensors, by role: which of an Op's inputs (or, for an
// output role, which of {out, more...}) each is; -1 when the graph does not
// give it.
enum Role { kX, kScale, kBias, kEps, kMean, kInv, kFactor, kRunMeanIn, kRunVarIn, kDy,
            kY, kMeanOut, kInvOut, kRunMeanOut, kRunVarOut, kDx, kDscale, kDbias,
            // matmul overrides; resampling's index tensor
            kMOverride, kNOverride, kKOverride, kIdx,
            // RNG
            kSeed, kOffset,
            // softmax and attention
            kSink, kStats, kMax, kSumExp, kFill, kSeqQ, kSeqKV, kLeft, kShift,
            kQ, kK, kV, kO, kDO, kDQ, kDK, kDV, kRngDump, kPageK, kPageV, kDSink,
            // statistics generation; paged cache load
            kSum, kSqSum, kContainer, kPageTable, kSeqLen,
            kRoles };

struct Op {
  Kind kind;
  const Desc* desc;
  std::vector<GTensor> in;   // the roles below, in order
  GTensor out;
  // Convolutions: in = {x, w} (forward), {dy, w} (data), {dy, x} (filter).
  vc::ConvGeom geom;
  double alpha = 1.0, beta = 0.0, alpha2 = 1.0;
  vc::Accum acc = vc::Accum::Exact;
  // Pointwise: the mode and its parameters; in = {x}, {x, b}, {x, b, t} or,
  // for a backward mode, {dy, x}.
  cudnnPointwiseMode_t pw = CUDNN_POINTWISE_ADD;
  double lower = 0.0, upper = 0.0, slope = 0.0, elu_alpha = 1.0, softplus_beta = 1.0, swish_beta = 1.0;
  bool has_upper = false;
  int64_t axis = -1;
  // Reduction: the operator. Matmul: in = {a, b}.
  cudnnReduceTensorOp_t red = CUDNN_REDUCE_TENSOR_ADD;
  // Normalization: the mode, whether it trains, the outputs beyond `out`,
  // and where each role is (an index into in, or for an output role into
  // {out, more...}).
  // Resampling (pooling): the mode, how padding reads, and the window.
  cudnnResampleMode_t resample = CUDNN_RESAMPLE_MAXPOOL;
  cudnnPaddingMode_t padding = CUDNN_ZERO_PAD;
  int nsp = 0;
  int64_t win[3] = {1, 1, 1}, pre[3] = {0, 0, 0}, post[3] = {0, 0, 0}, pstr[3] = {1, 1, 1};
  cudnnBackendNormMode_t norm = CUDNN_LAYER_NORM;
  bool training = false;
  // Resampling by interpolation: fractional window, strides and paddings.
  double fwin[3] = {1, 1, 1}, fpre[3] = {0, 0, 0}, fpost[3] = {0, 0, 0}, fstr[3] = {1, 1, 1};
  // Matmul: the value an element outside an overridden M x N is given.
  double pad_value = 0.0;
  // Reshape: view-only (the same memory through other dims and strides) or
  // logical (row-major order kept). Transpose: the permutation. Slice: per
  // dimension start, limit and stride.
  bool view_only = true;
  std::vector<int64_t> perm, start, limit, sstride;
  // RNG: the distribution and its parameters; the seed when it is a value.
  cudnnRngDistribution_t dist = CUDNN_RNG_DISTRIBUTION_BERNOULLI;
  double prob = 0.5, umin = 0.0, umax = 1.0, nmean = 0.0, nstd = 1.0;
  int64_t seed = 0;
  // Diagonal band mask: the comparison.
  cudnnPointwiseMode_t cmp = CUDNN_POINTWISE_CMP_GE;
  // Attention: dropout probability; the pre-softmax subgraph (its schedule,
  // input and output) and the softmax's own operation, when given.
  double dropout = 0.0;
  std::shared_ptr<std::vector<struct Op>> sub;
  int64_t sub_in = 0, sub_out = 0;
  std::shared_ptr<struct Op> softmax;
  std::vector<GTensor> more;
  int role[kRoles];
  Op() { for (int& r : role) r = -1; }
};

const char* op_name(cudnnBackendDescriptorType_t t) {
  switch (t) {
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR: return "convolution forward";
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR: return "convolution backward-data";
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR: return "convolution backward-filter";
    case CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR: return "matmul";
    case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR: return "pointwise";
    case CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR: return "reduction";
    case CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR: return "generate-statistics";
    case CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR: return "BN finalize-statistics";
    case CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR: return "BN backward-weights";
    case CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR: return "resample forward";
    case CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR: return "resample backward";
    case CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR: return "concatenate";
    case CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR: return "signal";
    case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR: return "normalization forward";
    case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR: return "normalization backward";
    case CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR: return "reshape";
    case CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR: return "random number generation";
    case CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR: return "paged cache load";
    case CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR: return "block-scale quantize";
    case CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR: return "block-scale dequantize";
    case CUDNN_BACKEND_OPERATION_EXPAND_BAND_MATRIX_DESCRIPTOR: return "expand band matrix";
    case CUDNN_BACKEND_OPERATION_CONTRACT_BAND_MATRIX_DESCRIPTOR: return "contract band matrix";
    case CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR: return "scaled dot-product attention forward";
    case CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR: return "scaled dot-product attention backward";
    case CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR: return "MoE grouped matmul";
    case CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR: return "diagonal band mask";
    case CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR: return "softmax";
    case CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR: return "transpose";
    case CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR: return "slice";
    case CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR: return "MoE grouped matmul backward";
    case CUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR: return "rotary position embedding forward";
    case CUDNN_BACKEND_OPERATION_ROPE_BWD_DESCRIPTOR: return "rotary position embedding backward";
    default: return nullptr;
  }
}

enum class Arity { Unary, Binary, Ternary, Backward, Unknown };
Arity arity(cudnnPointwiseMode_t m) {
  switch (m) {
    case CUDNN_POINTWISE_ADD: case CUDNN_POINTWISE_ADD_SQUARE: case CUDNN_POINTWISE_DIV: case CUDNN_POINTWISE_MAX:
    case CUDNN_POINTWISE_MIN: case CUDNN_POINTWISE_MOD: case CUDNN_POINTWISE_MUL: case CUDNN_POINTWISE_POW:
    case CUDNN_POINTWISE_SUB: case CUDNN_POINTWISE_ATAN2: case CUDNN_POINTWISE_CMP_EQ: case CUDNN_POINTWISE_CMP_NEQ:
    case CUDNN_POINTWISE_CMP_GT: case CUDNN_POINTWISE_CMP_GE: case CUDNN_POINTWISE_CMP_LT: case CUDNN_POINTWISE_CMP_LE:
    case CUDNN_POINTWISE_LOGICAL_AND: case CUDNN_POINTWISE_LOGICAL_OR:
      return Arity::Binary;
    case CUDNN_POINTWISE_ABS: case CUDNN_POINTWISE_CEIL: case CUDNN_POINTWISE_COS: case CUDNN_POINTWISE_EXP:
    case CUDNN_POINTWISE_FLOOR: case CUDNN_POINTWISE_LOG: case CUDNN_POINTWISE_NEG: case CUDNN_POINTWISE_RSQRT:
    case CUDNN_POINTWISE_SIN: case CUDNN_POINTWISE_SQRT: case CUDNN_POINTWISE_TAN: case CUDNN_POINTWISE_ERF:
    case CUDNN_POINTWISE_IDENTITY: case CUDNN_POINTWISE_RECIPROCAL: case CUDNN_POINTWISE_LOGICAL_NOT:
    case CUDNN_POINTWISE_GEN_INDEX:
    case CUDNN_POINTWISE_RELU_FWD: case CUDNN_POINTWISE_TANH_FWD: case CUDNN_POINTWISE_SIGMOID_FWD:
    case CUDNN_POINTWISE_ELU_FWD: case CUDNN_POINTWISE_GELU_FWD: case CUDNN_POINTWISE_SOFTPLUS_FWD:
    case CUDNN_POINTWISE_SWISH_FWD: case CUDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return Arity::Unary;
    case CUDNN_POINTWISE_RELU_BWD: case CUDNN_POINTWISE_TANH_BWD: case CUDNN_POINTWISE_SIGMOID_BWD:
    case CUDNN_POINTWISE_ELU_BWD: case CUDNN_POINTWISE_GELU_BWD: case CUDNN_POINTWISE_SOFTPLUS_BWD:
    case CUDNN_POINTWISE_SWISH_BWD: case CUDNN_POINTWISE_GELU_APPROX_TANH_BWD:
      return Arity::Backward;
    case CUDNN_POINTWISE_BINARY_SELECT:
      return Arity::Ternary;
    default:
      return Arity::Unknown;
  }
}

// a at c's rank: a lower-rank tensor (cudnn-frontend's scalars are [1])
// lines up with c's last dimensions, the leading ones of extent 1.
vc::Layout at_rank(const vc::Layout& a, int rank) {
  if (a.rank >= rank) return a;
  vc::Layout r = a;
  r.rank = rank;
  const int k = rank - a.rank;
  for (int i = rank; i-- > 0;) {
    r.dims[i] = i >= k ? a.dims[i - k] : 1;
    r.strides[i] = i >= k ? a.strides[i - k] : 0;
  }
  return r;
}

// Every dimension of a is c's or 1, at c's rank.
bool broadcasts(const vc::Layout& a0, const vc::Layout& c) {
  const vc::Layout a = at_rank(a0, c.rank);
  if (a.rank != c.rank) return false;
  for (int i = 0; i < a.rank; ++i)
    if (a.dims[i] != c.dims[i] && a.dims[i] != 1) return false;
  return true;
}

// Every dimension of a is c's or divides it (a group of c's entries per
// entry of a: grouped-query heads, group normalization's statistics).
bool divides(const vc::Layout& a, const vc::Layout& c) {
  if (a.rank != c.rank) return false;
  for (int i = 0; i < a.rank; ++i)
    if (a.dims[i] < 1 || c.dims[i] % a.dims[i]) return false;
  return true;
}

bool schedule_ops(const Desc* graph, std::vector<Op>* order, std::string* why, const std::set<int64_t>& free_inputs);

bool op_of(const Desc* d, Op* op, std::string* why) {
  op->desc = d;
  auto tensor = [&](cudnnBackendAttributeName_t n, const char* role, GTensor* t) {
    std::string w;
    if (tensor_of(d->desc(n), t, &w)) return true;
    *why = std::string(op_name(d->type)) + ": " + role + ": " + w;
    return false;
  };
  auto conv = [&](cudnnBackendAttributeName_t cn, const vc::Layout& x, const vc::Layout& w, const vc::Layout& y) {
    const Desc* c = d->desc(cn);
    if (!c || c->type != CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR) { *why = "the convolution's settings are missing"; return false; }
    const int nsp = static_cast<int>(c->i64(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS));
    const auto pre = c->i64s(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS), post = c->i64s(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS);
    const auto str = c->i64s(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES), dil = c->i64s(CUDNN_ATTR_CONVOLUTION_DILATIONS);
    const size_t s = static_cast<size_t>(nsp);
    if (nsp < 1 || nsp > 3 || pre.size() != s || str.size() != s || dil.size() != s) {
      *why = "the convolution's settings are incomplete (spatial " + std::to_string(nsp) + ", pads " +
             std::to_string(pre.size()) + ", strides " + std::to_string(str.size()) + ", dilations " +
             std::to_string(dil.size()) + ")";
      return false;
    }
    // Post paddings may differ from pre paddings (TensorFlow's SAME padding):
    // the output's extent follows from both, positions from the pre ones.
    if (!post.empty() && post.size() != s) { *why = "the convolution's post paddings are incomplete"; return false; }
    const bool flip = c->i64(CUDNN_ATTR_CONVOLUTION_CONV_MODE) == CUDNN_CONVOLUTION;
    if (!vc::conv_geometry(x, w, y, nsp, pre.data(), str.data(), dil.data(), flip, &op->geom, why,
                           post.empty() ? nullptr : post.data()))
      return false;
    const cudnnDataType_t ct = static_cast<cudnnDataType_t>(c->i64(CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_DATA_FLOAT));
    op->acc = ct == CUDNN_DATA_HALF ? vc::Accum::Half : vc::Accum::Exact;
    return true;
  };
  // Each role's tensor, if set: inputs go to in, outputs to out (the
  // first) and then more.
  bool have_out = false;
  auto take = [&](cudnnBackendAttributeName_t n, int r, bool required, bool output) {
    if (!d->desc(n)) {
      if (required) *why = std::string(op_name(d->type)) + ": a required tensor is missing";
      return !required;
    }
    GTensor t;
    if (!tensor(n, "a tensor", &t)) return false;
    if (output) {
      if (!have_out) {
        op->out = t;
        op->role[r] = 0;
        have_out = true;
      } else {
        op->more.push_back(t);
        op->role[r] = static_cast<int>(op->more.size());
      }
    } else {
      op->in.push_back(t);
      op->role[r] = static_cast<int>(op->in.size() - 1);
    }
    return true;
  };
  switch (d->type) {
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR: {
      GTensor x, w;
      op->kind = Kind::ConvFwd;
      if (!tensor(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, "x", &x) ||
          !tensor(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, "w", &w) ||
          !tensor(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, "y", &op->out))
        return false;
      op->in = {x, w};
      op->alpha = scalar(d, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, 1.0);
      op->beta = scalar(d, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, 0.0);
      return conv(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, x.l, w.l, op->out.l);
    }
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR: {
      GTensor dy, w;
      op->kind = Kind::ConvData;
      if (!tensor(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, "dy", &dy) ||
          !tensor(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, "w", &w) ||
          !tensor(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, "dx", &op->out))
        return false;
      op->in = {dy, w};
      op->alpha = scalar(d, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, 1.0);
      op->beta = scalar(d, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, 0.0);
      return conv(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, op->out.l, w.l, dy.l);
    }
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR: {
      GTensor dy, x;
      op->kind = Kind::ConvFilter;
      if (!tensor(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, "dy", &dy) ||
          !tensor(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, "x", &x) ||
          !tensor(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, "dw", &op->out))
        return false;
      op->in = {dy, x};
      op->alpha = scalar(d, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, 1.0);
      op->beta = scalar(d, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, 0.0);
      return conv(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, x.l, op->out.l, dy.l);
    }
    case CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR: {
      GTensor a, b;
      op->kind = Kind::Matmul;
      if (!tensor(CUDNN_ATTR_OPERATION_MATMUL_ADESC, "a", &a) || !tensor(CUDNN_ATTR_OPERATION_MATMUL_BDESC, "b", &b) ||
          !tensor(CUDNN_ATTR_OPERATION_MATMUL_CDESC, "c", &op->out))
        return false;
      const Desc* md = d->desc(CUDNN_ATTR_OPERATION_MATMUL_DESC);
      if (!md || md->type != CUDNN_BACKEND_MATMUL_DESCRIPTOR) { *why = "matmul: the matmul descriptor is missing"; return false; }
      const auto ct = static_cast<cudnnDataType_t>(md->i64(CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_DATA_FLOAT));
      op->acc = ct == CUDNN_DATA_HALF ? vc::Accum::Half : vc::Accum::Exact;
      op->pad_value = scalar(md, CUDNN_ATTR_MATMUL_PADDING_VALUE, 0.0);
      const vc::Layout &A = a.l, &B = b.l, &C = op->out.l;
      const int r = C.rank;
      if (r < 2 || A.rank != r || B.rank != r || A.dims[r - 1] != B.dims[r - 2] || A.dims[r - 2] != C.dims[r - 2] ||
          B.dims[r - 1] != C.dims[r - 1]) {
        *why = "matmul: a [.., M, K], b [.., K, N] and c [.., M, N] do not fit together";
        return false;
      }
      // A batch dimension of a or b is c's, 1 (broadcast) or a divisor of
      // c's, each of its entries then serving a consecutive group of c's
      // (grouped-query attention's K and V heads).
      for (int i = 0; i < r - 2; ++i)
        if (C.dims[i] % A.dims[i] || C.dims[i] % B.dims[i]) {
          *why = "matmul: a batch dimension of a or b neither is c's nor divides it";
          return false;
        }
      op->in = {a, b};
      // Per-batch M, N and K: the leading rows, columns and terms each batch
      // really has (padded sequences); the rest of c is the padding value.
      if (!take(CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC, kMOverride, false, false) ||
          !take(CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC, kNOverride, false, false) ||
          !take(CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC, kKOverride, false, false))
        return false;
      for (int ro : {kMOverride, kNOverride, kKOverride}) {
        if (op->role[ro] < 0) continue;
        const vc::Layout& L = op->in[op->role[ro]].l;
        bool fits = L.rank == r;
        for (int i = 0; fits && i < r; ++i) fits = i < r - 2 ? C.dims[i] % L.dims[i] == 0 : L.dims[i] == 1;
        if (!fits) { *why = "matmul: an M/N/K override is not [batch.., 1, 1] over c's batch"; return false; }
      }
      return true;
    }
    case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR: {
      op->kind = Kind::Pointwise;
      const Desc* pd = d->desc(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR);
      if (!pd || pd->type != CUDNN_BACKEND_POINTWISE_DESCRIPTOR) { *why = "pointwise: the pointwise descriptor is missing"; return false; }
      op->pw = static_cast<cudnnPointwiseMode_t>(pd->i64(CUDNN_ATTR_POINTWISE_MODE));
      op->lower = scalar(pd, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, 0.0);
      op->has_upper = pd->get(CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP) != nullptr;
      op->upper = scalar(pd, CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP, 0.0);
      op->slope = scalar(pd, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE, 0.0);
      op->elu_alpha = scalar(pd, CUDNN_ATTR_POINTWISE_ELU_ALPHA, 1.0);
      op->softplus_beta = scalar(pd, CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA, 1.0);
      op->swish_beta = scalar(pd, CUDNN_ATTR_POINTWISE_SWISH_BETA, 1.0);
      op->axis = pd->i64(CUDNN_ATTR_POINTWISE_AXIS, -1);
      op->alpha = scalar(d, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, 1.0);
      op->alpha2 = scalar(d, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, 1.0);
      GTensor x, b, t;
      switch (arity(op->pw)) {
        case Arity::Unary:
          if (!tensor(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, "x", &x) || !tensor(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, "y", &op->out)) return false;
          op->in = {x};
          break;
        case Arity::Binary:
          if (!tensor(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, "x", &x) || !tensor(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, "b", &b) ||
              !tensor(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, "y", &op->out))
            return false;
          op->in = {x, b};
          break;
        case Arity::Ternary:
          if (!tensor(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, "x", &x) || !tensor(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, "b", &b) ||
              !tensor(CUDNN_ATTR_OPERATION_POINTWISE_TDESC, "t", &t) || !tensor(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, "y", &op->out))
            return false;
          op->in = {x, b, t};
          break;
        case Arity::Backward:
          if (!tensor(CUDNN_ATTR_OPERATION_POINTWISE_DYDESC, "dy", &b) || !tensor(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, "x", &x) ||
              !tensor(CUDNN_ATTR_OPERATION_POINTWISE_DXDESC, "dx", &op->out))
            return false;
          op->in = {b, x};
          break;
        default:
          *why = "pointwise mode " + std::to_string(op->pw) + " is not supported";
          return false;
      }
      for (const GTensor& g : op->in)
        if (!broadcasts(g.l, op->out.l)) { *why = "pointwise: an input's dimensions are neither the output's nor 1"; return false; }
      if (op->pw == CUDNN_POINTWISE_GEN_INDEX && (op->axis < 0 || op->axis >= op->out.l.rank)) {
        *why = "pointwise GEN_INDEX: the axis is outside the tensor";
        return false;
      }
      return true;
    }
    case CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR: {
      GTensor x;
      op->kind = Kind::Reduction;
      if (!tensor(CUDNN_ATTR_OPERATION_REDUCTION_XDESC, "x", &x) || !tensor(CUDNN_ATTR_OPERATION_REDUCTION_YDESC, "y", &op->out))
        return false;
      const Desc* rd = d->desc(CUDNN_ATTR_OPERATION_REDUCTION_DESC);
      if (!rd || rd->type != CUDNN_BACKEND_REDUCTION_DESCRIPTOR) { *why = "reduction: the reduction descriptor is missing"; return false; }
      op->red = static_cast<cudnnReduceTensorOp_t>(rd->i64(CUDNN_ATTR_REDUCTION_OPERATOR));
      if (op->red < CUDNN_REDUCE_TENSOR_ADD || op->red > CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS) {
        *why = "reduction operator " + std::to_string(op->red) + " is not supported";
        return false;
      }
      // y's extents are x's, 1, or divisors of x's: each y entry then sums
      // a consecutive group (grouped-query attention's head reduction).
      if (!divides(op->out.l, x.l)) {
        *why = "reduction: every dimension of y must be x's, 1 or a divisor of x's";
        return false;
      }
      op->in = {x};
      return true;
    }
    case CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR: {
      op->kind = Kind::Concat;
      if (!tensor(CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC, "output", &op->out)) return false;
      const Attr* ins = d->get(CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS);
      if (!ins || ins->count < 1) { *why = "concatenate: no inputs"; return false; }
      op->axis = d->i64(CUDNN_ATTR_OPERATION_CONCAT_AXIS, -1);
      const vc::Layout& Y = op->out.l;
      if (op->axis < 0 || op->axis >= Y.rank) { *why = "concatenate: the axis is outside the output"; return false; }
      int64_t total = 0;
      for (int64_t i = 0; i < ins->count; ++i) {
        GTensor t;
        if (!tensor_of(d->desc(CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS, static_cast<size_t>(i)), &t, why)) return false;
        bool fits = t.l.rank == Y.rank;
        for (int k = 0; fits && k < Y.rank; ++k) fits = k == op->axis || t.l.dims[k] == Y.dims[k];
        if (!fits) { *why = "concatenate: an input differs from the output outside the axis"; return false; }
        total += t.l.dims[op->axis];
        op->in.push_back(t);
      }
      if (total != Y.dims[op->axis]) { *why = "concatenate: the inputs' extents along the axis do not add up to the output's"; return false; }
      // The in-place input (already where it belongs in the output) is
      // copied over itself like the others: the same values.
      return true;
    }
    case CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR: {
      const bool fwd = d->type == CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR;
      op->kind = fwd ? Kind::PoolFwd : Kind::PoolBwd;
      const Desc* rd = d->desc(fwd ? CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC : CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC);
      if (!rd || rd->type != CUDNN_BACKEND_RESAMPLE_DESCRIPTOR) { *why = "resample: the resample descriptor is missing"; return false; }
      op->resample = static_cast<cudnnResampleMode_t>(rd->i64(CUDNN_ATTR_RESAMPLE_MODE, CUDNN_RESAMPLE_NEAREST));
      if (op->resample != CUDNN_RESAMPLE_MAXPOOL && op->resample != CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING &&
          op->resample != CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING && op->resample != CUDNN_RESAMPLE_NEAREST &&
          op->resample != CUDNN_RESAMPLE_BILINEAR) {
        *why = "resample mode " + std::to_string(op->resample) + " is not defined";
        return false;
      }
      const bool interp = op->resample == CUDNN_RESAMPLE_NEAREST || op->resample == CUDNN_RESAMPLE_BILINEAR;
      // Interpolation: cuDNN documents neither how the window, strides and
      // paddings place the samples nor how bilinear weights them, and an RTX
      // 3060 with cuDNN 9.27 offers no engine for either mode (forward or
      // backward, NCHW or NHWC, upsampling or downsampling -- measured), so
      // there is nothing to match: refused by name, as that hardware does.
      if (interp) {
        *why = "resample: nearest and bilinear interpolation are not supported (no engine on the hardware, semantics "
               "undocumented)";
        return false;
      }
      op->padding = static_cast<cudnnPaddingMode_t>(rd->i64(CUDNN_ATTR_RESAMPLE_PADDING_MODE, CUDNN_ZERO_PAD));
      op->nsp = static_cast<int>(rd->i64(CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS));
      // Each setting: integers, or fractions (interpolation's strides and
      // paddings are fractional when it upsamples).
      auto fracs = [&](cudnnBackendAttributeName_t n, double* out) {
        const Attr* a = rd->get(n);
        if (!a || a->count != op->nsp) return false;
        for (int i = 0; i < op->nsp; ++i) {
          if (a->type == CUDNN_TYPE_INT64) {
            int64_t v;
            std::memcpy(&v, a->bytes.data() + i * sizeof v, sizeof v);
            out[i] = static_cast<double>(v);
          } else if (a->type == CUDNN_TYPE_FRACTION) {
            cudnnFraction_t f;
            std::memcpy(&f, a->bytes.data() + i * sizeof f, sizeof f);
            if (f.denominator == 0) return false;
            out[i] = static_cast<double>(f.numerator) / static_cast<double>(f.denominator);
          } else {
            return false;
          }
        }
        return true;
      };
      if (op->nsp < 1 || op->nsp > 3 || !fracs(CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, op->fwin) ||
          !fracs(CUDNN_ATTR_RESAMPLE_STRIDES, op->fstr) || !fracs(CUDNN_ATTR_RESAMPLE_PRE_PADDINGS, op->fpre) ||
          !fracs(CUDNN_ATTR_RESAMPLE_POST_PADDINGS, op->fpost)) {
        *why = "resample: the window, strides and paddings must be lists of 1 to 3 integers or fractions";
        return false;
      }
      for (int i = 0; i < op->nsp; ++i) {
        const bool integral = op->fwin[i] == std::floor(op->fwin[i]) && op->fstr[i] == std::floor(op->fstr[i]) &&
                              op->fpre[i] == std::floor(op->fpre[i]) && op->fpost[i] == std::floor(op->fpost[i]);
        if (!interp && !integral) {
          *why = "resample: pooling takes integer windows, strides and paddings";
          return false;
        }
        op->win[i] = static_cast<int64_t>(op->fwin[i]), op->pstr[i] = static_cast<int64_t>(op->fstr[i]);
        op->pre[i] = static_cast<int64_t>(op->fpre[i]), op->post[i] = static_cast<int64_t>(op->fpost[i]);
        if (op->resample == CUDNN_RESAMPLE_BILINEAR && op->fwin[i] != 2.0) {
          *why = "resample: bilinear interpolation's window is 2";  // cuDNN's documented rule
          return false;
        }
      }
      // The index tensor: max pooling's argmax (or nearest's source) within
      // each window, which the backward pass may read instead of x.
      if (!take(fwd ? CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC : CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC, kIdx, false, false))
        return false;
      if (op->role[kIdx] >= 0) {
        const GTensor idx = op->in.back();
        op->in.pop_back();
        op->role[kIdx] = -1;
        op->more.push_back(idx);  // placed below: an output forward, an input backward
      }
      op->alpha = scalar(d, fwd ? CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA : CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA, 1.0);
      op->beta = scalar(d, fwd ? CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA : CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA, 0.0);
      GTensor x, y;
      std::vector<GTensor> idx;
      idx.swap(op->more);
      const vc::Layout *X, *Y;
      if (fwd) {
        if (!tensor(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, "x", &x) || !tensor(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, "y", &op->out))
          return false;
        op->in = {x};
        if (!idx.empty()) op->more = idx, op->role[kIdx] = 1;
        X = &op->in[0].l, Y = &op->out.l;
      } else {
        GTensor dy;
        if (!tensor(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC, "dy", &dy) ||
            !tensor(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC, "dx", &op->out))
          return false;
        op->in = {dy};
        // x (which max pooling's gradient reads, unless the index tensor
        // says where each maximum was) and y may be given.
        if (d->desc(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC)) {
          if (!tensor(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC, "x", &x)) return false;
          op->in.push_back(x);
          op->role[kX] = 1;
        } else if (op->resample == CUDNN_RESAMPLE_MAXPOOL && idx.empty()) {
          *why = "resample backward: max pooling's gradient needs x or the index tensor";
          return false;
        }
        if (d->desc(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC)) {
          if (!tensor(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC, "y", &y)) return false;
          op->in.push_back(y);
        }
        if (!idx.empty()) {
          op->in.push_back(idx[0]);
          op->role[kIdx] = static_cast<int>(op->in.size() - 1);
        }
        X = &op->out.l, Y = &op->in[0].l;
      }
      if (X->rank != op->nsp + 2 || Y->rank != X->rank || X->dims[0] != Y->dims[0] || X->dims[1] != Y->dims[1]) {
        *why = "resample: x and y do not have the window's rank, batch and channels";
        return false;
      }
      // y_i = 1 + (x_i + pre_i + post_i - w_i) / s_i, as cuDNN documents it
      // for every mode (the division truncating).
      for (int i = 0; i < op->nsp; ++i) {
        const double span = static_cast<double>(X->dims[2 + i]) + op->fpre[i] + op->fpost[i] - op->fwin[i];
        if (op->fwin[i] <= 0 || op->fstr[i] <= 0 || op->fpre[i] < 0 || op->fpost[i] < 0 || span < 0 ||
            Y->dims[2 + i] != 1 + static_cast<int64_t>(std::floor(span / op->fstr[i] + 1e-9))) {
          *why = "resample: y's extent in spatial dimension " + std::to_string(i) + " is not what the window gives";
          return false;
        }
      }
      if (op->role[kIdx] >= 0) {
        const vc::Layout& I = fwd ? op->more[0].l : op->in[op->role[kIdx]].l;
        if (!I.same_dims(*Y) || (op->resample != CUDNN_RESAMPLE_MAXPOOL && op->resample != CUDNN_RESAMPLE_NEAREST)) {
          *why = "resample: the index tensor must have y's shape, for max pooling or nearest resampling";
          return false;
        }
      }
      if (!fwd && op->role[kX] >= 0 && !op->in[op->role[kX]].l.same_dims(*X)) {
        *why = "resample backward: x's shape is not dx's";
        return false;
      }
      return true;
    }
    case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR: {
      const bool fwd = d->type == CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR;
      op->kind = fwd ? Kind::NormFwd : Kind::NormBwd;
      op->norm = static_cast<cudnnBackendNormMode_t>(d->i64(fwd ? CUDNN_ATTR_OPERATION_NORM_FWD_MODE : CUDNN_ATTR_OPERATION_NORM_BWD_MODE));
      op->training = fwd && d->i64(CUDNN_ATTR_OPERATION_NORM_FWD_PHASE) == CUDNN_NORM_FWD_TRAINING;
      const char* mode = op->norm == CUDNN_LAYER_NORM ? "layer" : op->norm == CUDNN_INSTANCE_NORM ? "instance"
                       : op->norm == CUDNN_BATCH_NORM ? "batch" : op->norm == CUDNN_RMS_NORM ? "RMS"
                       : op->norm == CUDNN_GROUP_NORM ? "group" : nullptr;
      if (!mode) { *why = "normalization mode " + std::to_string(op->norm) + " is not defined"; return false; }
      if (fwd && op->norm == CUDNN_BATCH_NORM && !op->training) {
        *why = "batch normalization's inference phase is not supported (nor is it by cuDNN's graph API)";
        return false;
      }
      if (d->get(fwd ? CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS : CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS)) {
        *why = "multi-GPU normalization (peer statistics) is not supported";
        return false;
      }
      const bool rms = op->norm == CUDNN_RMS_NORM;
      bool ok;
      if (fwd) {
        ok = take(CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, kY, true, true) &&
             take(CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, kX, true, false) &&
             take(CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, kScale, true, false) &&
             take(CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, kBias, !rms, false) &&
             take(CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, kEps, true, false);
        if (ok && op->training)
          ok = take(CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, kMeanOut, false, true) &&
               take(CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, kInvOut, false, true) &&
               take(CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC, kFactor, false, false) &&
               take(CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC, kRunMeanIn, false, false) &&
               take(CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC, kRunVarIn, false, false) &&
               take(CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC, kRunMeanOut, false, true) &&
               take(CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC, kRunVarOut, false, true);
        else if (ok)
          ok = take(CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, kMean, !rms, false) &&
               take(CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, kInv, true, false);
        if (ok && op->role[kRunMeanOut] >= 0 && (op->role[kRunMeanIn] < 0 || op->role[kFactor] < 0 || op->role[kRunVarOut] < 0 || op->role[kRunVarIn] < 0)) {
          *why = "normalization forward: running statistics need both inputs, both outputs and the factor";
          return false;
        }
      } else {
        ok = take(CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, kDx, true, true) &&
             take(CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, kX, true, false) &&
             take(CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, kDy, true, false) &&
             take(CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, kScale, true, false) &&
             take(CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC, kMean, false, false) &&
             take(CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, kInv, false, false) &&
             take(CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC, kEps, false, false) &&
             take(CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, kDscale, false, true) &&
             take(CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC, kDbias, false, true);
        // Without the saved statistics they are recomputed from x, which
        // needs epsilon.
        const bool saved = op->role[kInv] >= 0 && (rms || op->role[kMean] >= 0);
        if (ok && !saved && op->role[kEps] < 0) {
          *why = "normalization backward: without the saved mean and inverse variance, epsilon is required";
          return false;
        }
      }
      if (!ok) return false;
      // Shapes: y (dx) is x's; scale, bias and the statistics broadcast onto x.
      const vc::Layout& X = op->in[op->role[kX]].l;
      if (!op->out.l.same_dims(X)) { *why = std::string(mode) + " normalization: the output's shape is not x's"; return false; }
      // Statistics may also cover groups of x's entries (group
      // normalization's [N, G, 1, 1] over [N, C, ...]).
      for (int r : {kScale, kBias, kRunMeanIn, kRunVarIn, kDy})
        if (op->role[r] >= 0 && !broadcasts(op->in[op->role[r]].l, X)) {
          *why = std::string(mode) + " normalization: a parameter does not broadcast onto x";
          return false;
        }
      for (int r : {kMean, kInv})
        if (op->role[r] >= 0 && !divides(op->in[op->role[r]].l, X)) {
          *why = std::string(mode) + " normalization: a statistic does not cover x";
          return false;
        }
      for (const GTensor& t : op->more)
        if (!divides(t.l, X)) { *why = std::string(mode) + " normalization: an output does not cover x"; return false; }
      if (op->norm == CUDNN_GROUP_NORM) {
        // The groups are the statistics' second dimension: they must be
        // given (by the mean or inverse-variance tensor) to say how many.
        const vc::Layout* st = nullptr;
        for (int r : {kMean, kInv})
          if (op->role[r] >= 0) st = &op->in[op->role[r]].l;
        for (int r : {kMeanOut, kInvOut})
          if (op->role[r] > 0) st = &op->more[op->role[r] - 1].l;
        if (!st || X.rank < 3 || st->dims[0] != X.dims[0] || X.dims[1] % st->dims[1]) {
          *why = "group normalization: the mean or inverse-variance tensor, [N, G, 1, ...] with G dividing C, is required";
          return false;
        }
        for (int i = 2; i < st->rank; ++i)
          if (st->dims[i] != 1) { *why = "group normalization: the statistics must be 1 past the group dimension"; return false; }
      }
      if (op->role[kEps] >= 0 && op->in[op->role[kEps]].l.count() != 1) {
        *why = std::string(mode) + " normalization: epsilon is not a scalar";
        return false;
      }
      return true;
    }
    case CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR: {
      // The same elements under other dims and strides: view-only keeps
      // their memory (y is x's bytes read through y's strides -- a
      // transpose, when the strides are permuted), logical keeps their
      // row-major order.
      op->kind = Kind::Reshape;
      if (!take(CUDNN_ATTR_OPERATION_RESHAPE_XDESC, kX, true, false) || !take(CUDNN_ATTR_OPERATION_RESHAPE_YDESC, kY, true, true))
        return false;
      op->view_only = d->i64(CUDNN_ATTR_OPERATION_RESHAPE_MODE, CUDNN_RESHAPE_VIEW_ONLY) == CUDNN_RESHAPE_VIEW_ONLY;
      if (op->in[0].l.count() != op->out.l.count()) { *why = "reshape: x and y hold different numbers of elements"; return false; }
      return true;
    }
    case CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR: {
      op->kind = Kind::Transpose;
      if (!take(CUDNN_ATTR_OPERATION_TRANSPOSE_XDESC, kX, true, false) || !take(CUDNN_ATTR_OPERATION_TRANSPOSE_YDESC, kY, true, true))
        return false;
      op->perm = d->i64s(CUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION);
      const vc::Layout &X = op->in[0].l, &Y = op->out.l;
      std::vector<bool> seen(static_cast<size_t>(X.rank), false);
      bool ok = static_cast<int>(op->perm.size()) == X.rank && Y.rank == X.rank;
      for (int i = 0; ok && i < X.rank; ++i) {
        const int64_t q = op->perm[i];
        ok = q >= 0 && q < X.rank && !seen[q] && Y.dims[i] == X.dims[q];
        if (ok) seen[q] = true;
      }
      if (!ok) { *why = "transpose: the permutation does not map x's dimensions onto y's"; return false; }
      return true;
    }
    case CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR: {
      op->kind = Kind::Slice;
      if (!take(CUDNN_ATTR_OPERATION_SLICE_XDESC, kX, true, false) || !take(CUDNN_ATTR_OPERATION_SLICE_YDESC, kY, true, true))
        return false;
      op->start = d->i64s(CUDNN_ATTR_OPERATION_SLICE_START_INDICES);
      op->limit = d->i64s(CUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES);
      op->sstride = d->i64s(CUDNN_ATTR_OPERATION_SLICE_STRIDES);
      const vc::Layout &X = op->in[0].l, &Y = op->out.l;
      const size_t r = static_cast<size_t>(X.rank);
      if (op->sstride.empty()) op->sstride.assign(r, 1);
      bool ok = op->start.size() == r && op->limit.size() == r && op->sstride.size() == r && Y.rank == X.rank;
      for (size_t i = 0; ok && i < r; ++i)
        ok = op->sstride[i] >= 1 && op->start[i] >= 0 && op->limit[i] <= X.dims[i] && op->start[i] < op->limit[i] &&
             Y.dims[i] == (op->limit[i] - op->start[i] + op->sstride[i] - 1) / op->sstride[i];
      if (!ok) { *why = "slice: the start, limit and strides do not give y's shape inside x"; return false; }
      return true;
    }
    case CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR: {
      op->kind = Kind::Rng;
      const Desc* rd = d->desc(CUDNN_ATTR_OPERATION_RNG_DESC);
      if (!rd || rd->type != CUDNN_BACKEND_RNG_DESCRIPTOR) { *why = "random number generation: the RNG descriptor is missing"; return false; }
      op->dist = static_cast<cudnnRngDistribution_t>(rd->i64(CUDNN_ATTR_RNG_DISTRIBUTION, CUDNN_RNG_DISTRIBUTION_BERNOULLI));
      op->prob = scalar(rd, CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY, 0.5);
      op->umin = scalar(rd, CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM, 0.0);
      op->umax = scalar(rd, CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM, 1.0);
      op->nmean = scalar(rd, CUDNN_ATTR_RNG_NORMAL_DIST_MEAN, -1.0);
      op->nstd = scalar(rd, CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION, -1.0);
      if (!take(CUDNN_ATTR_OPERATION_RNG_YDESC, kY, true, true) || !take(CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC, kOffset, true, false))
        return false;
      const Attr* sa = d->get(CUDNN_ATTR_OPERATION_RNG_SEED);
      if (sa && sa->type == CUDNN_TYPE_BACKEND_DESCRIPTOR) {
        if (!take(CUDNN_ATTR_OPERATION_RNG_SEED, kSeed, true, false)) return false;
      } else {
        op->seed = d->i64(CUDNN_ATTR_OPERATION_RNG_SEED, 0);
      }
      for (int r : {kSeed, kOffset})
        if (op->role[r] >= 0 && op->in[op->role[r]].l.count() != 1) {
          *why = "random number generation: the seed and offset must be single-element tensors";
          return false;
        }
      return true;
    }
    case CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR: {
      // Per-channel sum and sum of squares: every output dimension 1 but C.
      op->kind = Kind::GenStats;
      if (!take(CUDNN_ATTR_OPERATION_GENSTATS_XDESC, kX, true, false) ||
          !take(CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC, kSum, true, true) ||
          !take(CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC, kSqSum, true, true))
        return false;
      if (d->i64(CUDNN_ATTR_OPERATION_GENSTATS_MODE, CUDNN_GENSTATS_SUM_SQSUM) != CUDNN_GENSTATS_SUM_SQSUM) {
        *why = "generate-statistics: only the SUM_SQSUM mode is defined";
        return false;
      }
      const vc::Layout& X = op->in[0].l;
      for (const vc::Layout* l : {&op->out.l, &op->more[0].l}) {
        bool ok = l->rank == X.rank && X.rank >= 2;
        for (int i = 0; ok && i < X.rank; ++i) ok = l->dims[i] == (i == 1 ? X.dims[1] : 1);
        if (!ok) { *why = "generate-statistics: the outputs must be 1 in every dimension but C, which is x's"; return false; }
      }
      return true;
    }
    case CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR: {
      // Softmax over the last dimension, with its row maximum, sum of
      // exponentials and log-sum-exp (stats) as optional outputs, and an
      // optional sink: one more logit per head that joins the denominator.
      op->kind = Kind::Softmax;
      if (!take(CUDNN_ATTR_OPERATION_SOFTMAX_YDESC, kY, true, true) || !take(CUDNN_ATTR_OPERATION_SOFTMAX_XDESC, kX, true, false) ||
          !take(CUDNN_ATTR_OPERATION_SOFTMAX_SINK_DESC, kSink, false, false) ||
          !take(CUDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC, kStats, false, true) ||
          !take(CUDNN_ATTR_OPERATION_SOFTMAX_MAX_DESC, kMax, false, true) ||
          !take(CUDNN_ATTR_OPERATION_SOFTMAX_SUM_EXP_DESC, kSumExp, false, true))
        return false;
      const vc::Layout& X = op->in[op->role[kX]].l;
      if (!op->out.l.same_dims(X) || X.rank < 2) { *why = "softmax: y's shape is not x's"; return false; }
      vc::Layout row = X;
      row.dims[X.rank - 1] = 1;
      for (int r : {kStats, kMax, kSumExp})
        if (op->role[r] > 0 && !op->more[op->role[r] - 1].l.same_dims(row)) {
          *why = "softmax: the statistics are not x's shape with a last dimension of 1";
          return false;
        }
      if (op->role[kSink] >= 0 && !broadcasts(op->in[op->role[kSink]].l, row)) {
        *why = "softmax: the sink does not broadcast onto x's rows";
        return false;
      }
      return true;
    }
    case CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR: {
      // y = x where the element (row = dimension -2, column = dimension -1)
      // lies inside the band, else b (minus infinity, typically):
      //   right bound (CMP_GE):  row + shift [+ s_kv - s_q] >= col
      //   left bound (CMP_GT):   col + left [- s_kv + s_q] > row
      // the sequence lengths, when given, aligning the diagonal bottom-right.
      op->kind = Kind::BandMask;
      if (!take(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC, kY, true, true) ||
          !take(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC, kX, true, false) ||
          !take(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC, kFill, false, false) ||
          !take(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC, kSeqQ, false, false) ||
          !take(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC, kSeqKV, false, false) ||
          !take(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC, kLeft, false, false) ||
          !take(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SHIFT_RIGHT_BOUND_DESC, kShift, false, false))
        return false;
      if (d->desc(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_QDESC) ||
          d->desc(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_KVDESC)) {
        *why = "diagonal band mask: cumulative sequence lengths are not supported";
        return false;
      }
      op->cmp = static_cast<cudnnPointwiseMode_t>(d->i64(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE, CUDNN_POINTWISE_CMP_GE));
      const vc::Layout& X = op->in[op->role[kX]].l;
      if (!op->out.l.same_dims(X) || X.rank < 2) { *why = "diagonal band mask: y's shape is not x's"; return false; }
      for (int r : {kFill, kSeqQ, kSeqKV, kLeft, kShift})
        if (op->role[r] >= 0 && !broadcasts(op->in[op->role[r]].l, X)) {
          *why = "diagonal band mask: a bound or sequence length does not broadcast onto x";
          return false;
        }
      return true;
    }
    case CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR: {
      // y[b, h, s, d] = container[table[b, s / bs], h, s % bs, d] for s below
      // the batch's sequence length (y may be K's transpose, through its strides).
      op->kind = Kind::PagedLoad;
      if (!take(CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC, kY, true, true) ||
          !take(CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC, kContainer, true, false) ||
          !take(CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC, kPageTable, true, false) ||
          !take(CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC, kSeqLen, true, false))
        return false;
      const vc::Layout &C = op->in[op->role[kContainer]].l, &T = op->in[op->role[kPageTable]].l, &Y = op->out.l;
      if (C.rank != 4 || T.rank != 4 || Y.rank != 4 || Y.dims[1] != C.dims[1] || T.dims[0] != Y.dims[0]) {
        *why = "paged cache load: the container [blocks, H, block size, D], page table [B, 1, pages, 1] and y do not fit";
        return false;
      }
      return true;
    }
    case CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR: {
      const bool fwd = d->type == CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR;
      op->kind = fwd ? Kind::SdpaFwd : Kind::SdpaBwd;
      bool ok;
      if (fwd) {
        if (d->desc(CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC)) {
          *why = "scaled dot-product attention: block masks are not supported";
          return false;
        }
        for (cudnnBackendAttributeName_t n : {CUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_QDESC, CUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_KVDESC,
                                              CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_QDESC, CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_KDESC,
                                              CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_VDESC, CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_SDESC,
                                              CUDNN_ATTR_OPERATION_SDPA_FWD_SCALE_SDESC, CUDNN_ATTR_OPERATION_SDPA_FWD_SCALE_ODESC,
                                              CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_SDESC, CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_ODESC})
          if (d->desc(n)) {
            *why = "scaled dot-product attention: cumulative sequence lengths and FP8 scaling are not supported";
            return false;
          }
        ok = take(CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, kO, true, true) && take(CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, kQ, true, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, kK, true, false) && take(CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, kV, true, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC, kStats, false, true) &&
             take(CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC, kScale, false, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC, kSeqQ, false, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC, kSeqKV, false, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC, kPageK, false, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC, kPageV, false, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_SEED_DESC, kSeed, false, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_OFFSET_DESC, kOffset, false, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_RNG_DUMP_DESC, kRngDump, false, true);
        op->dropout = scalar(d, CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_PROBABILITY, 0.0);
        if (ok && op->dropout != 0.0 && (op->role[kSeed] < 0 || op->role[kOffset] < 0 || op->dropout < 0 || op->dropout >= 1)) {
          *why = "scaled dot-product attention: dropout needs a seed, an offset and a probability in [0, 1)";
          return false;
        }
        // The softmax's own descriptor (cuDNN 9.21+): its statistics, row
        // maximum, sum of exponentials and sink join this operation's
        // tensors.
        if (ok && d->desc(CUDNN_ATTR_OPERATION_SDPA_FWD_SOFTMAX_DESC)) {
          auto sm = std::make_shared<Op>();
          if (!op_of(d->desc(CUDNN_ATTR_OPERATION_SDPA_FWD_SOFTMAX_DESC), sm.get(), why)) return false;
          if (sm->kind != Kind::Softmax) { *why = "scaled dot-product attention: the softmax descriptor is not a softmax"; return false; }
          if (sm->role[kSink] >= 0) {
            op->in.push_back(sm->in[sm->role[kSink]]);
            op->role[kSink] = static_cast<int>(op->in.size() - 1);
          }
          for (int r : {kStats, kMax, kSumExp})
            if (sm->role[r] > 0) {
              if (op->role[r] >= 0) { *why = "scaled dot-product attention: statistics given twice"; return false; }
              op->more.push_back(sm->more[sm->role[r] - 1]);
              op->role[r] = static_cast<int>(op->more.size());
            }
          op->softmax = sm;
        }
      } else {
        if (d->desc(CUDNN_ATTR_OPERATION_SDPA_BWD_SINK_DESC) || d->desc(CUDNN_ATTR_OPERATION_SDPA_BWD_DSINK_DESC)) {
          *why = "scaled dot-product attention backward: sinks are not supported";
          return false;
        }
        ok = take(CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC, kDQ, true, true) && take(CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC, kDK, true, true) &&
             take(CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC, kDV, true, true) && take(CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC, kQ, true, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC, kK, true, false) && take(CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC, kV, true, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC, kO, true, false) && take(CUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC, kDO, true, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC, kStats, true, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC, kScale, false, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC, kSeqQ, false, false) &&
             take(CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC, kSeqKV, false, false);
      }
      if (!ok) return false;
      // The score modifier subgraph: a graph run on the scaled scores
      // (bias, masks, ALiBi, soft-capping) whose input and output uids are
      // given; its other tensors come from this graph's variant pack.
      const Desc* sg = d->desc(fwd ? CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH : CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH);
      if (sg) {
        op->sub_in = d->i64(fwd ? CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID : CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_INPUT_UID, -1);
        op->sub_out = d->i64(fwd ? CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID : CUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_OUTPUT_UID, -1);
        op->sub = std::make_shared<std::vector<Op>>();
        if (!schedule_ops(sg, op->sub.get(), why, {op->sub_in})) return false;
        // Its device inputs are this operation's too (for the schedule and
        // the variant pack); virtual ones stay inside it.
        for (const Op& so : *op->sub)
          for (const GTensor& t : so.in)
            if (!t.is_virtual && !t.by_value) op->in.push_back(t);
      }
      // Shapes: Q [B, Hq, Sq, D], K [B, Hk, Skv, D], V [B, Hv, Skv, Dv], O [B, Hq, Sq, Dv],
      // Hk and Hv dividing Hq; or K and V page containers.
      const vc::Layout &Q = op->in[op->role[kQ]].l, &K = op->in[op->role[kK]].l, &V = op->in[op->role[kV]].l;
      const vc::Layout& O = fwd ? op->out.l : op->in[op->role[kO]].l;
      const bool paged = op->role[kPageK] >= 0 || op->role[kPageV] >= 0;
      if (paged && (op->role[kPageK] < 0 || op->role[kPageV] < 0 || op->role[kSeqKV] < 0)) {
        *why = "scaled dot-product attention: paged K and V need both page tables and the K/V sequence lengths";
        return false;
      }
      bool shapes = Q.rank == 4 && K.rank == 4 && V.rank == 4 && O.rank == 4 && K.dims[3] == Q.dims[3] &&
                    O.dims[0] == Q.dims[0] && O.dims[1] == Q.dims[1] && O.dims[2] == Q.dims[2] && O.dims[3] == V.dims[3] &&
                    K.dims[1] >= 1 && V.dims[1] >= 1 && Q.dims[1] % K.dims[1] == 0 && Q.dims[1] % V.dims[1] == 0;
      if (shapes && !paged)
        shapes = K.dims[0] == Q.dims[0] && V.dims[0] == Q.dims[0] && K.dims[2] == V.dims[2];
      if (!shapes) { *why = "scaled dot-product attention: Q, K, V and O do not fit together"; return false; }
      for (int r : {kSeqQ, kSeqKV})
        if (op->role[r] >= 0 && op->in[op->role[r]].l.count() != static_cast<size_t>(Q.dims[0])) {
          *why = "scaled dot-product attention: a sequence-length tensor does not have one element per batch";
          return false;
        }
      if (op->role[kScale] >= 0 && op->in[op->role[kScale]].l.count() != 1) {
        *why = "scaled dot-product attention: the scale is not a single element";
        return false;
      }
      return true;
    }
    default: {
      const char* n = op_name(d->type);
      *why = n ? std::string("the ") + n + " operation is not implemented"
               : "operation descriptor type " + std::to_string(d->type) + " is not implemented";
      return false;
    }
  }
}

// A graph's operations in an order that runs: each after the ones whose
// outputs it reads. Every tensor no operation writes must be one the variant
// pack supplies (not virtual) -- or, for a subgraph (an attention operation's
// score modifiers), one of free_inputs, which its operation provides.
bool schedule_ops(const Desc* graph, std::vector<Op>* order, std::string* why, const std::set<int64_t>& free_inputs) {
  const Attr* ops = graph->get(CUDNN_ATTR_OPERATIONGRAPH_OPS);
  if (!ops || ops->count < 1) { *why = "the graph has no operations"; return false; }
  std::vector<Op> all(static_cast<size_t>(ops->count));
  std::map<int64_t, size_t> producer;
  for (size_t i = 0; i < all.size(); ++i) {
    const Desc* d = graph->desc(CUDNN_ATTR_OPERATIONGRAPH_OPS, i);
    if (!d || !op_of(d, &all[i], why)) return false;
    std::vector<int64_t> outs{all[i].out.uid};
    for (const GTensor& t : all[i].more) outs.push_back(t.uid);
    for (int64_t uid : outs)
      if (!producer.emplace(uid, i).second) {
        *why = "tensor " + std::to_string(uid) + " is written by two operations";
        return false;
      }
  }
  for (const Op& o : all)
    for (const GTensor& t : o.in)
      if (t.is_virtual && !producer.count(t.uid) && !free_inputs.count(t.uid)) {
        *why = "virtual tensor " + std::to_string(t.uid) + " is read but no operation writes it";
        return false;
      }
  std::vector<bool> done(all.size(), false);
  std::set<int64_t> ready;
  order->clear();
  while (order->size() < all.size()) {
    bool progress = false;
    for (size_t i = 0; i < all.size(); ++i) {
      if (done[i]) continue;
      bool ok = true;
      for (const GTensor& t : all[i].in) ok &= !producer.count(t.uid) || ready.count(t.uid);
      if (!ok) continue;
      done[i] = true, progress = true;
      ready.insert(all[i].out.uid);
      for (const GTensor& t : all[i].more) ready.insert(t.uid);
      order->push_back(all[i]);
    }
    if (!progress) { *why = "the graph's operations depend on each other in a cycle"; return false; }
  }
  return true;
}
bool schedule(const Desc* graph, std::vector<Op>* order, std::string* why) {
  return schedule_ops(graph, order, why, {});
}

// Whether this library can run a graph, and if not, why. A graph made
// without a handle is a subgraph (an attention operation's score modifiers):
// its virtual inputs come from the operation that holds it.
bool runnable(const Desc* graph, std::string* why) {
  std::vector<Op> order;
  if (!graph->get(CUDNN_ATTR_OPERATIONGRAPH_HANDLE)) {
    std::set<int64_t> any;
    const Attr* ops = graph->get(CUDNN_ATTR_OPERATIONGRAPH_OPS);
    for (int64_t i = 0; ops && i < ops->count; ++i) {
      Op o;
      const Desc* d = graph->desc(CUDNN_ATTR_OPERATIONGRAPH_OPS, static_cast<size_t>(i));
      if (d && op_of(d, &o, why))
        for (const GTensor& t : o.in)
          if (t.is_virtual) any.insert(t.uid);
    }
    return schedule_ops(graph, &order, why, any);
  }
  return schedule(graph, &order, why);
}

/* ---- running one operation ---- */

double erf_gelu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }
double tanh_gelu_inner(double x) { return std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x); }
double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double pointwise(const Op& op, double x, double b, double t, int64_t index) {
  switch (op.pw) {
    case CUDNN_POINTWISE_ADD: return x + b;
    case CUDNN_POINTWISE_ADD_SQUARE: return x + b * b;
    case CUDNN_POINTWISE_DIV: return x / b;
    case CUDNN_POINTWISE_MAX: return std::max(x, b);
    case CUDNN_POINTWISE_MIN: return std::min(x, b);
    case CUDNN_POINTWISE_MOD: return std::fmod(x, b);
    case CUDNN_POINTWISE_MUL: return x * b;
    case CUDNN_POINTWISE_POW: return std::pow(x, b);
    case CUDNN_POINTWISE_SUB: return x - b;
    case CUDNN_POINTWISE_ATAN2: return std::atan2(x, b);
    case CUDNN_POINTWISE_ABS: return std::fabs(x);
    case CUDNN_POINTWISE_CEIL: return std::ceil(x);
    case CUDNN_POINTWISE_COS: return std::cos(x);
    case CUDNN_POINTWISE_EXP: return std::exp(x);
    case CUDNN_POINTWISE_FLOOR: return std::floor(x);
    case CUDNN_POINTWISE_LOG: return std::log(x);
    case CUDNN_POINTWISE_NEG: return -x;
    case CUDNN_POINTWISE_RSQRT: return 1.0 / std::sqrt(x);
    case CUDNN_POINTWISE_SIN: return std::sin(x);
    case CUDNN_POINTWISE_SQRT: return std::sqrt(x);
    case CUDNN_POINTWISE_TAN: return std::tan(x);
    case CUDNN_POINTWISE_ERF: return std::erf(x);
    case CUDNN_POINTWISE_IDENTITY: return x;
    case CUDNN_POINTWISE_RECIPROCAL: return 1.0 / x;
    case CUDNN_POINTWISE_CMP_EQ: return x == b;
    case CUDNN_POINTWISE_CMP_NEQ: return x != b;
    case CUDNN_POINTWISE_CMP_GT: return x > b;
    case CUDNN_POINTWISE_CMP_GE: return x >= b;
    case CUDNN_POINTWISE_CMP_LT: return x < b;
    case CUDNN_POINTWISE_CMP_LE: return x <= b;
    case CUDNN_POINTWISE_LOGICAL_AND: return x != 0.0 && b != 0.0;
    case CUDNN_POINTWISE_LOGICAL_OR: return x != 0.0 || b != 0.0;
    case CUDNN_POINTWISE_LOGICAL_NOT: return x == 0.0;
    case CUDNN_POINTWISE_GEN_INDEX: return static_cast<double>(index);
    case CUDNN_POINTWISE_BINARY_SELECT: return t != 0.0 ? x : b;
    case CUDNN_POINTWISE_RELU_FWD:
      // Below the lower clip: lower + slope * (x - lower); above the upper: upper.
      if (x < op.lower) return op.lower + op.slope * (x - op.lower);
      if (op.has_upper && x > op.upper) return op.upper;
      return x;
    case CUDNN_POINTWISE_TANH_FWD: return std::tanh(x);
    case CUDNN_POINTWISE_SIGMOID_FWD: return sigm(x);
    case CUDNN_POINTWISE_ELU_FWD: return x > 0.0 ? x : op.elu_alpha * (std::exp(x) - 1.0);
    case CUDNN_POINTWISE_GELU_FWD: return erf_gelu(x);
    case CUDNN_POINTWISE_SOFTPLUS_FWD: return std::log1p(std::exp(op.softplus_beta * x)) / op.softplus_beta;
    case CUDNN_POINTWISE_SWISH_FWD: return x * sigm(op.swish_beta * x);
    case CUDNN_POINTWISE_GELU_APPROX_TANH_FWD: return 0.5 * x * (1.0 + std::tanh(tanh_gelu_inner(x)));
    // The backward modes: b is dy, x the forward pass's input.
    case CUDNN_POINTWISE_RELU_BWD:
      if (x <= op.lower) return b * op.slope;
      if (op.has_upper && x >= op.upper) return 0.0;
      return b;
    case CUDNN_POINTWISE_TANH_BWD: { const double th = std::tanh(x); return b * (1.0 - th * th); }
    case CUDNN_POINTWISE_SIGMOID_BWD: { const double s = sigm(x); return b * s * (1.0 - s); }
    case CUDNN_POINTWISE_ELU_BWD: return x > 0.0 ? b : b * op.elu_alpha * std::exp(x);
    case CUDNN_POINTWISE_GELU_BWD: {
      const double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
      const double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
      return b * (cdf + x * pdf);
    }
    case CUDNN_POINTWISE_SOFTPLUS_BWD: return b * sigm(op.softplus_beta * x);
    case CUDNN_POINTWISE_SWISH_BWD: {
      const double s = sigm(op.swish_beta * x);
      return b * (s + op.swish_beta * x * s * (1.0 - s));
    }
    case CUDNN_POINTWISE_GELU_APPROX_TANH_BWD: {
      const double u = tanh_gelu_inner(x), th = std::tanh(u);
      const double du = std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * x * x);
      return b * (0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du);
    }
    default: return 0.0;
  }
}

// For each element of c (logical order), the logical index of a's element
// broadcast onto it: a's extent is c's, 1, or a divisor of c's (each of a's
// entries then covering a consecutive group of c's).
std::vector<size_t> broadcast_index(const vc::Layout& a0, const vc::Layout& c) {
  const vc::Layout a = at_rank(a0, c.rank);
  std::vector<size_t> idx(c.count());
  int64_t at[vc::kMaxRank] = {};
  for (size_t i = 0; i < idx.size(); ++i) {
    size_t j = 0;
    for (int d = 0; d < c.rank; ++d) j = j * a.dims[d] + at[d] / (c.dims[d] / a.dims[d]);
    idx[i] = j;
    for (int d = c.rank; d-- > 0;) {
      if (++at[d] < c.dims[d]) break;
      at[d] = 0;
    }
  }
  return idx;
}

// Batched c = a b. a's and b's batch dimensions broadcast or group onto c's;
// per-batch M, N and K overrides (the leading rows, columns and terms a
// padded batch really has) leave the rest of c at the padding value.
void run_matmul(const Op& op, const std::vector<const std::vector<double>*>& in, std::vector<double>* c) {
  const vc::Layout &A = op.in[0].l, &B = op.in[1].l, &C = op.out.l;
  const std::vector<double> &a = *in[0], &b = *in[1];
  const int r = C.rank;
  const int64_t M = C.dims[r - 2], N = C.dims[r - 1], K = A.dims[r - 1];
  // The batch: C's leading dimensions, with a's and b's broadcast onto them.
  vc::Layout cb, ab, bb;
  cb.rank = ab.rank = bb.rank = r - 2;
  for (int i = 0; i < r - 2; ++i) cb.dims[i] = C.dims[i], ab.dims[i] = A.dims[i], bb.dims[i] = B.dims[i];
  const size_t batches = r > 2 ? cb.count() : 1;
  const std::vector<size_t> ai = r > 2 ? broadcast_index(ab, cb) : std::vector<size_t>{0};
  const std::vector<size_t> bi = r > 2 ? broadcast_index(bb, cb) : std::vector<size_t>{0};
  // An override's value for each batch.
  auto per_batch = [&](int role, int64_t full) {
    std::vector<int64_t> v(batches, full);
    if (op.role[role] < 0) return v;
    vc::Layout ob;
    ob.rank = r - 2;
    const vc::Layout& L = op.in[op.role[role]].l;
    for (int i = 0; i < r - 2; ++i) ob.dims[i] = L.dims[i];
    const std::vector<size_t> oi = r > 2 ? broadcast_index(ob, cb) : std::vector<size_t>{0};
    for (size_t q = 0; q < batches; ++q)
      v[q] = std::max<int64_t>(0, std::min<int64_t>(full, static_cast<int64_t>((*in[op.role[role]])[oi[q]])));
    return v;
  };
  const std::vector<int64_t> Mq = per_batch(kMOverride, M), Nq = per_batch(kNOverride, N), Kq = per_batch(kKOverride, K);
  c->assign(batches * M * N, op.pad_value);
  for (size_t q = 0; q < batches; ++q) {
    const double* pa = a.data() + ai[q] * M * K;
    const double* pb = b.data() + bi[q] * K * N;
    double* pc = c->data() + q * M * N;
    for (int64_t i = 0; i < Mq[q]; ++i)
      for (int64_t j = 0; j < Nq[q]; ++j) {
        double s = 0.0;
        for (int64_t k = 0; k < Kq[q]; ++k) {
          if (op.acc == vc::Accum::Half)
            s = vc::half_to_float(vc::float_to_half(static_cast<float>(s) + static_cast<float>(pa[i * K + k]) * static_cast<float>(pb[k * N + j])));
          else
            s += pa[i * K + k] * pb[k * N + j];
        }
        pc[i * N + j] = s;
      }
  }
}

void run_reduction(const Op& op, const std::vector<double>& x, std::vector<double>* y) {
  const vc::Layout &X = op.in[0].l, &Y = op.out.l;
  const size_t ny = Y.count();
  std::vector<double> acc(ny, 0.0);
  std::vector<int64_t> cnt(ny, 0);
  int64_t at[vc::kMaxRank] = {};
  for (size_t i = 0; i < x.size(); ++i) {
    size_t o = 0;
    for (int d = 0; d < X.rank; ++d) o = o * Y.dims[d] + at[d] / (X.dims[d] / Y.dims[d]);
    const double v = x[i];
    const int64_t k = cnt[o]++;
    double& a = acc[o];
    switch (op.red) {
      case CUDNN_REDUCE_TENSOR_ADD: case CUDNN_REDUCE_TENSOR_AVG: a = k ? a + v : v; break;
      case CUDNN_REDUCE_TENSOR_MUL: a = k ? a * v : v; break;
      case CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS: if (v != 0.0) a = (k && a != 0.0) ? a * v : (k ? a : v); else if (!k) a = 0.0; break;
      case CUDNN_REDUCE_TENSOR_MIN: a = k ? std::min(a, v) : v; break;
      case CUDNN_REDUCE_TENSOR_MAX: a = k ? std::max(a, v) : v; break;
      case CUDNN_REDUCE_TENSOR_AMAX: a = k ? std::max(a, std::fabs(v)) : std::fabs(v); break;
      case CUDNN_REDUCE_TENSOR_NORM1: a += std::fabs(v); break;
      case CUDNN_REDUCE_TENSOR_NORM2: a += v * v; break;
    }
    for (int d = X.rank; d-- > 0;) {
      if (++at[d] < X.dims[d]) break;
      at[d] = 0;
    }
  }
  for (size_t o = 0; o < ny; ++o) {
    if (op.red == CUDNN_REDUCE_TENSOR_AVG && cnt[o]) acc[o] /= static_cast<double>(cnt[o]);
    if (op.red == CUDNN_REDUCE_TENSOR_NORM2) acc[o] = std::sqrt(acc[o]);
  }
  *y = std::move(acc);
}

// Layer, instance, batch and RMS normalization: statistics over the
// dimensions where the statistics tensor has extent 1 (by default those the
// mode names: all but N; all but N and C; all but C; all but N), then
// y = scale * (x - mean) * inv + bias, with inv = 1 / sqrt(var + eps) and,
// for RMS, no mean. The backward pass, from the saved statistics:
//   dx = inv * (g - mean(g) - xhat * mean(g * xhat)),  g = dy * scale
// (no mean(g) term for RMS), and dscale, dbias summed over the dimensions
// where they have extent 1. outs[0] is op.out; outs[k] is op.more[k - 1].
void run_norm(const Op& op, const std::vector<const std::vector<double>*>& in, std::vector<std::vector<double>>* outs) {
  const vc::Layout& X = op.in[op.role[kX]].l;
  const std::vector<double>& x = *in[op.role[kX]];
  const size_t n = x.size();
  auto input = [&](int r) -> const std::vector<double>* { return op.role[r] >= 0 ? in[op.role[r]] : nullptr; };
  auto layout_in = [&](int r) -> const vc::Layout* { return op.role[r] >= 0 ? &op.in[op.role[r]].l : nullptr; };
  auto layout_out = [&](int r) -> const vc::Layout* {
    return op.role[r] < 0 ? nullptr : op.role[r] == 0 ? &op.out.l : &op.more[op.role[r] - 1].l;
  };
  // The statistics' shape.
  vc::Layout st;
  const vc::Layout* given = nullptr;
  for (const vc::Layout* l : {layout_out(kMeanOut), layout_out(kInvOut), layout_in(kMean), layout_in(kInv)})
    if (l && !given) given = l;
  if (given) {
    st = *given;
  } else {
    st.rank = X.rank;
    for (int i = 0; i < X.rank; ++i) {
      const bool keep = op.norm == CUDNN_BATCH_NORM ? i == 1 : op.norm == CUDNN_INSTANCE_NORM ? i <= 1 : i == 0;
      st.dims[i] = keep ? X.dims[i] : 1;
    }
  }
  const std::vector<size_t> si = broadcast_index(st, X);
  const size_t ns = st.count();
  std::vector<double> cnt(ns, 0.0);
  for (size_t i = 0; i < n; ++i) cnt[si[i]] += 1;
  const bool rms = op.norm == CUDNN_RMS_NORM;
  const double eps = input(kEps) ? (*input(kEps))[0] : 0.0;
  auto bcast = [&](int r, double dflt) {  // a parameter, per element of x
    std::vector<double> v(n, dflt);
    if (op.role[r] < 0) return v;
    const std::vector<size_t> bi = broadcast_index(op.in[op.role[r]].l, X);
    for (size_t i = 0; i < n; ++i) v[i] = (*in[op.role[r]])[bi[i]];
    return v;
  };
  std::vector<double> mean(ns, 0.0), inv(ns, 0.0), var(ns, 0.0);
  const bool saved = op.role[kInv] >= 0 && (rms || op.role[kMean] >= 0);
  if ((op.kind == Kind::NormFwd && op.training) || (op.kind == Kind::NormBwd && !saved)) {
    for (size_t i = 0; i < n; ++i) mean[si[i]] += rms ? 0.0 : x[i];
    for (size_t k = 0; k < ns; ++k) mean[k] /= cnt[k];
    for (size_t i = 0; i < n; ++i) { const double c = x[i] - mean[si[i]]; var[si[i]] += c * c; }
    for (size_t k = 0; k < ns; ++k) var[k] /= cnt[k], inv[k] = 1.0 / std::sqrt(var[k] + eps);
  } else {
    // From the given statistics, read in st's own order.
    const std::vector<size_t> mi = input(kMean) ? broadcast_index(*layout_in(kMean), st) : std::vector<size_t>{};
    const std::vector<size_t> ii = broadcast_index(*layout_in(kInv), st);
    for (size_t k = 0; k < ns; ++k) {
      mean[k] = input(kMean) && !rms ? (*input(kMean))[mi[k]] : 0.0;
      inv[k] = (*input(kInv))[ii[k]];
    }
  }
  const std::vector<double> scale = bcast(kScale, 1.0);
  outs->assign(1 + op.more.size(), {});
  auto put = [&](int r, std::vector<double> v) { if (op.role[r] >= 0) (*outs)[op.role[r]] = std::move(v); };
  // Statistics in an output tensor's own shape (it is st's, or broadcast-equal).
  auto stats_out = [&](int r, const std::vector<double>& v) {
    const vc::Layout* l = layout_out(r);
    if (!l) return;
    const std::vector<size_t> bi = broadcast_index(st, *l);
    std::vector<double> o(l->count());
    for (size_t k = 0; k < o.size(); ++k) o[k] = v[bi[k]];
    put(r, std::move(o));
  };
  if (op.kind == Kind::NormFwd) {
    const std::vector<double> bias = bcast(kBias, 0.0);
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i) y[i] = scale[i] * (x[i] - mean[si[i]]) * inv[si[i]] + bias[i];
    put(kY, std::move(y));
    if (op.training) {
      stats_out(kMeanOut, mean);
      stats_out(kInvOut, inv);
      if (op.role[kRunMeanOut] >= 0) {
        // Running variance tracks the unbiased estimate.
        const double f = (*input(kFactor))[0];
        const std::vector<size_t> rmi = broadcast_index(*layout_in(kRunMeanIn), st);
        const std::vector<size_t> rvi = broadcast_index(*layout_in(kRunVarIn), st);
        std::vector<double> rm(ns), rv(ns);
        for (size_t k = 0; k < ns; ++k) {
          const double unbiased = cnt[k] > 1 ? var[k] * cnt[k] / (cnt[k] - 1) : var[k];
          rm[k] = (1.0 - f) * (*input(kRunMeanIn))[rmi[k]] + f * mean[k];
          rv[k] = (1.0 - f) * (*input(kRunVarIn))[rvi[k]] + f * unbiased;
        }
        stats_out(kRunMeanOut, rm);
        stats_out(kRunVarOut, rv);
      }
    }
    return;
  }
  // Backward.
  const std::vector<double> dy = bcast(kDy, 0.0);
  std::vector<double> xhat(n), g(n), mg(ns, 0.0), mgx(ns, 0.0);
  for (size_t i = 0; i < n; ++i) {
    xhat[i] = (x[i] - mean[si[i]]) * inv[si[i]];
    g[i] = dy[i] * scale[i];
    mg[si[i]] += g[i];
    mgx[si[i]] += g[i] * xhat[i];
  }
  std::vector<double> dx(n);
  for (size_t i = 0; i < n; ++i) {
    const size_t k = si[i];
    dx[i] = inv[k] * (g[i] - (rms ? 0.0 : mg[k] / cnt[k]) - xhat[i] * mgx[k] / cnt[k]);
  }
  put(kDx, std::move(dx));
  for (int r : {kDscale, kDbias}) {
    const vc::Layout* l = layout_out(r);
    if (!l) continue;
    const std::vector<size_t> bi = broadcast_index(*l, X);
    std::vector<double> o(l->count(), 0.0);
    for (size_t i = 0; i < n; ++i) o[bi[i]] += r == kDscale ? dy[i] * xhat[i] : dy[i];
    put(r, std::move(o));
  }
}

// Pooling as the graph API's resampling: per (n, c), each output's window
// over the input, the padding read as zero, minus infinity or the nearest
// edge value as the padding mode says. Average pooling includes padded taps
// in its divisor or not as its mode says; the maximum's gradient goes to the
// first largest tap, recomputed from x.
// The index tensor (max pooling's, measured on an RTX 3060 with cuDNN 9.27):
// the maximum's position in its window, row-major over the window's taps
// with padded taps counted, as INT8; the backward pass may take it in place
// of x.
void run_pool(const Op& op, const std::vector<const std::vector<double>*>& in, std::vector<std::vector<double>>* outs) {
  std::vector<double>* r = &(*outs)[0];
  const bool fwd = op.kind == Kind::PoolFwd;
  const vc::Layout& X = fwd ? op.in[0].l : op.out.l;
  const vc::Layout& Y = fwd ? op.out.l : op.in[0].l;
  int64_t I[3] = {1, 1, 1}, O[3] = {1, 1, 1}, Wn[3] = {1, 1, 1}, P[3] = {0, 0, 0}, S[3] = {1, 1, 1};
  for (int i = 0; i < op.nsp; ++i) {
    const int k = 3 - op.nsp + i;
    I[k] = X.dims[2 + i], O[k] = Y.dims[2 + i], Wn[k] = op.win[i], P[k] = op.pre[i], S[k] = op.pstr[i];
  }
  const int64_t NC = X.dims[0] * X.dims[1], isz = I[0] * I[1] * I[2], osz = O[0] * O[1] * O[2];
  const bool is_max = op.resample == CUDNN_RESAMPLE_MAXPOOL;
  const std::vector<double>* xin = fwd ? in[0] : (is_max && op.role[kX] >= 0 ? in[op.role[kX]] : nullptr);
  const std::vector<double>* idx_in = !fwd && op.role[kIdx] >= 0 ? in[op.role[kIdx]] : nullptr;
  std::vector<double>* idx_out = fwd && op.role[kIdx] > 0 ? &(*outs)[op.role[kIdx]] : nullptr;
  if (idx_out) idx_out->assign(static_cast<size_t>(NC * osz), 0.0);
  r->assign(static_cast<size_t>(fwd ? NC * osz : NC * isz), 0.0);
  for (int64_t nc = 0; nc < NC; ++nc)
    for (int64_t o = 0; o < osz; ++o) {
      const int64_t o0 = o / (O[1] * O[2]), o1 = (o / O[2]) % O[1], o2 = o % O[2];
      double best = -INFINITY, sum = 0.0;
      int64_t arg = -1, valid = 0, taps = 0, warg = 0;
      for (int64_t a = 0; a < Wn[0]; ++a)
        for (int64_t b = 0; b < Wn[1]; ++b)
          for (int64_t c = 0; c < Wn[2]; ++c) {
            int64_t p[3] = {o0 * S[0] - P[0] + a, o1 * S[1] - P[1] + b, o2 * S[2] - P[2] + c};
            ++taps;
            bool inside = true;
            for (int d = 0; d < 3; ++d) inside &= p[d] >= 0 && p[d] < I[d];
            if (!inside && op.padding == CUDNN_EDGE_VAL_PAD) {
              for (int d = 0; d < 3; ++d) p[d] = std::min(std::max<int64_t>(p[d], 0), I[d] - 1);
              inside = true;
            }
            double v;
            int64_t at = -1;
            if (inside) {
              at = (p[0] * I[1] + p[1]) * I[2] + p[2];
              v = xin ? (*xin)[static_cast<size_t>(nc * isz + at)] : 0.0;
              ++valid;
            } else {
              if (op.padding == CUDNN_NEG_INF_PAD) continue;
              v = 0.0;  // zero padding
            }
            if (is_max && (arg == -1 || v > best)) best = v, arg = at, warg = (a * Wn[1] + b) * Wn[2] + c;
            sum += v;
          }
      const double div = op.resample == CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING ? static_cast<double>(valid)
                                                                              : static_cast<double>(taps);
      if (fwd) {
        (*r)[static_cast<size_t>(nc * osz + o)] = is_max ? best : (div ? sum / div : 0.0);
        if (idx_out) (*idx_out)[static_cast<size_t>(nc * osz + o)] = static_cast<double>(warg);
        continue;
      }
      const double g = (*in[0])[static_cast<size_t>(nc * osz + o)];
      if (is_max && idx_in) {
        // The window position the forward pass recorded.
        const int64_t w = static_cast<int64_t>((*idx_in)[static_cast<size_t>(nc * osz + o)]);
        if (w < 0 || w >= Wn[0] * Wn[1] * Wn[2]) continue;
        int64_t p[3] = {o0 * S[0] - P[0] + w / (Wn[1] * Wn[2]), o1 * S[1] - P[1] + (w / Wn[2]) % Wn[1], o2 * S[2] - P[2] + w % Wn[2]};
        bool inside = true;
        for (int d = 0; d < 3; ++d) inside &= p[d] >= 0 && p[d] < I[d];
        if (!inside && op.padding == CUDNN_EDGE_VAL_PAD) {
          for (int d = 0; d < 3; ++d) p[d] = std::min(std::max<int64_t>(p[d], 0), I[d] - 1);
          inside = true;
        }
        if (inside) (*r)[static_cast<size_t>(nc * isz + (p[0] * I[1] + p[1]) * I[2] + p[2])] += g;
        continue;
      }
      if (is_max) {
        if (arg >= 0) (*r)[static_cast<size_t>(nc * isz + arg)] += g;  // a padded maximum takes it nowhere
        continue;
      }
      for (int64_t a = 0; a < Wn[0]; ++a)
        for (int64_t b = 0; b < Wn[1]; ++b)
          for (int64_t c = 0; c < Wn[2]; ++c) {
            int64_t p[3] = {o0 * S[0] - P[0] + a, o1 * S[1] - P[1] + b, o2 * S[2] - P[2] + c};
            bool inside = true;
            for (int d = 0; d < 3; ++d) inside &= p[d] >= 0 && p[d] < I[d];
            if (!inside && op.padding == CUDNN_EDGE_VAL_PAD) {
              for (int d = 0; d < 3; ++d) p[d] = std::min(std::max<int64_t>(p[d], 0), I[d] - 1);
              inside = true;
            }
            if (inside && div) (*r)[static_cast<size_t>(nc * isz + (p[0] * I[1] + p[1]) * I[2] + p[2])] += g / div;
          }
    }
}

// Concatenation: each input's elements at their offset along the axis.
void run_concat(const Op& op, const std::vector<const std::vector<double>*>& in, std::vector<double>* r) {
  const vc::Layout& Y = op.out.l;
  const int ax = static_cast<int>(op.axis);
  int64_t outer = 1, inner = 1;
  for (int k = 0; k < ax; ++k) outer *= Y.dims[k];
  for (int k = ax + 1; k < Y.rank; ++k) inner *= Y.dims[k];
  r->assign(Y.count(), 0.0);
  int64_t at = 0;
  for (size_t i = 0; i < op.in.size(); ++i) {
    const int64_t e = op.in[i].l.dims[ax];
    for (int64_t o = 0; o < outer; ++o)
      for (int64_t j = 0; j < e * inner; ++j)
        (*r)[static_cast<size_t>((o * Y.dims[ax] + at) * inner + j)] = (*in[i])[static_cast<size_t>(o * e * inner + j)];
    at += e;
  }
}

/* ---- Philox: the RNG operation's generator and attention's dropout ---- */

// Philox4x32-10 (Salmon et al., "Parallel random numbers: as easy as 1, 2,
// 3"), keyed and countered as PyTorch's PhiloxRNGEngine, which cuDNN's RNG
// operation documents it follows: key = seed, counter = {offset, subsequence}
// (each a 64-bit pair of words).
struct Philox4 {
  uint32_t v[4];
};
Philox4 philox(uint64_t seed, uint64_t subsequence, uint64_t offset) {
  uint32_t c[4] = {static_cast<uint32_t>(offset), static_cast<uint32_t>(offset >> 32), static_cast<uint32_t>(subsequence),
                   static_cast<uint32_t>(subsequence >> 32)};
  uint32_t k[2] = {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
  for (int r = 0; r < 10; ++r) {
    const uint64_t p0 = 0xD2511F53ull * c[0], p1 = 0xCD9E8D57ull * c[2];
    const uint32_t hi0 = static_cast<uint32_t>(p0 >> 32), lo0 = static_cast<uint32_t>(p0);
    const uint32_t hi1 = static_cast<uint32_t>(p1 >> 32), lo1 = static_cast<uint32_t>(p1);
    const uint32_t n0 = hi1 ^ c[1] ^ k[0], n2 = hi0 ^ c[3] ^ k[1];
    c[0] = n0, c[1] = lo1, c[2] = n2, c[3] = lo0;
    k[0] += 0x9E3779B9u, k[1] += 0xBB67AE85u;
  }
  return {{c[0], c[1], c[2], c[3]}};
}
// A 32-bit draw as a uniform value in [0, 1).
double unit(uint32_t x) { return static_cast<double>(x) * (1.0 / 4294967296.0); }

// The i-th draw of a stream: the i-th 32-bit output of subsequence 0 from
// the offset on, as PyTorch's engine hands them out.
uint32_t draw(uint64_t seed, uint64_t offset, uint64_t i) {
  return philox(seed, 0, offset + i / 4).v[i % 4];
}

/* ---- data movement: reshape, transpose, slice, paged cache load ---- */

void run_reshape(const Op& op, const std::vector<double>& x, std::vector<double>* y) {
  const GTensor &X = op.in[0], &Y = op.out;
  if (!op.view_only) {  // the same row-major order
    *y = x;
    return;
  }
  // y is x's memory read through y's strides: find, for each of y's
  // element offsets, the element of x that lives there.
  const std::vector<int64_t> xo = element_offsets(X), yo = element_offsets(Y);
  int64_t hi = 0;
  for (int64_t o : xo) hi = std::max(hi, o + 1);
  std::vector<int64_t> at(static_cast<size_t>(hi), -1);
  for (size_t i = 0; i < xo.size(); ++i) at[static_cast<size_t>(xo[i])] = static_cast<int64_t>(i);
  y->assign(yo.size(), 0.0);
  for (size_t i = 0; i < yo.size(); ++i)
    if (yo[i] >= 0 && yo[i] < hi && at[static_cast<size_t>(yo[i])] >= 0) (*y)[i] = x[static_cast<size_t>(at[static_cast<size_t>(yo[i])])];
}

void run_transpose(const Op& op, const std::vector<double>& x, std::vector<double>* y) {
  const vc::Layout &X = op.in[0].l, &Y = op.out.l;
  y->assign(Y.count(), 0.0);
  int64_t at[vc::kMaxRank] = {}, xs[vc::kMaxRank] = {};
  for (int d = X.rank, s = 1; d-- > 0;) xs[d] = s, s *= static_cast<int>(X.dims[d]);
  for (size_t i = 0; i < y->size(); ++i) {
    int64_t j = 0;
    for (int d = 0; d < Y.rank; ++d) j += at[d] * xs[op.perm[d]];  // y index d is x index perm[d]
    (*y)[i] = x[static_cast<size_t>(j)];
    for (int d = Y.rank; d-- > 0;) {
      if (++at[d] < Y.dims[d]) break;
      at[d] = 0;
    }
  }
}

void run_slice(const Op& op, const std::vector<double>& x, std::vector<double>* y) {
  const vc::Layout &X = op.in[0].l, &Y = op.out.l;
  y->assign(Y.count(), 0.0);
  int64_t at[vc::kMaxRank] = {}, xs[vc::kMaxRank] = {};
  for (int d = X.rank, s = 1; d-- > 0;) xs[d] = s, s *= static_cast<int>(X.dims[d]);
  for (size_t i = 0; i < y->size(); ++i) {
    int64_t j = 0;
    for (int d = 0; d < Y.rank; ++d) j += (op.start[d] + at[d] * op.sstride[d]) * xs[d];
    (*y)[i] = x[static_cast<size_t>(j)];
    for (int d = Y.rank; d-- > 0;) {
      if (++at[d] < Y.dims[d]) break;
      at[d] = 0;
    }
  }
}

// Gathers [B, H, S, D] from a container of pages [blocks, H, block size, D]:
// position s of batch b is in page table[b, s / block size].
void page_gather(const vc::Layout& C, const std::vector<double>& cont, const std::vector<double>& table, int64_t pages,
                 int64_t B, int64_t S, const std::vector<double>* seq, std::vector<double>* out) {
  const int64_t H = C.dims[1], bs = C.dims[2], D = C.dims[3];
  out->assign(static_cast<size_t>(B * H * S * D), 0.0);
  for (int64_t b = 0; b < B; ++b) {
    const int64_t len = seq ? std::min<int64_t>(S, static_cast<int64_t>((*seq)[static_cast<size_t>(b)])) : S;
    for (int64_t s = 0; s < len; ++s) {
      const int64_t page = static_cast<int64_t>(table[static_cast<size_t>(b * pages + s / bs)]);
      if (page < 0 || page >= C.dims[0]) continue;
      for (int64_t h = 0; h < H; ++h)
        for (int64_t d = 0; d < D; ++d)
          (*out)[static_cast<size_t>(((b * H + h) * S + s) * D + d)] =
              cont[static_cast<size_t>(((page * H + h) * bs + s % bs) * D + d)];
    }
  }
}

void run_paged_load(const Op& op, const std::vector<const std::vector<double>*>& in, std::vector<double>* y) {
  const vc::Layout &C = op.in[op.role[kContainer]].l, &T = op.in[op.role[kPageTable]].l, &Y = op.out.l;
  // y may be declared transposed ([B, H, D, S], K's form for Q K^T): its
  // strides say which; the gather is done in [B, H, S, D] and laid out.
  const bool kt = Y.dims[3] != C.dims[3];
  const int64_t S = kt ? Y.dims[3] : Y.dims[2];
  std::vector<double> g;
  page_gather(C, *in[op.role[kContainer]], *in[op.role[kPageTable]], T.dims[2], Y.dims[0], S, in[op.role[kSeqLen]], &g);
  if (!kt) {
    *y = std::move(g);
    return;
  }
  const int64_t B = Y.dims[0], H = Y.dims[1], D = Y.dims[2];
  y->assign(g.size(), 0.0);
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t s = 0; s < S; ++s)
        for (int64_t d = 0; d < D; ++d)
          (*y)[static_cast<size_t>(((b * H + h) * D + d) * S + s)] = g[static_cast<size_t>(((b * H + h) * S + s) * D + d)];
}

/* ---- random numbers, statistics, softmax and masks ---- */

// The RNG operation: one draw per element of y in its logical order.
// Bernoulli gives 1 with the given probability; uniform covers [min, max);
// normal pairs draws through Box-Muller.
void run_rng(const Op& op, const std::vector<const std::vector<double>*>& in, std::vector<double>* y) {
  const uint64_t seed = op.role[kSeed] >= 0 ? static_cast<uint64_t>(static_cast<int64_t>((*in[op.role[kSeed]])[0]))
                                            : static_cast<uint64_t>(op.seed);
  const uint64_t offset = static_cast<uint64_t>(static_cast<int64_t>((*in[op.role[kOffset]])[0]));
  const size_t n = op.out.l.count();
  y->assign(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const double u = unit(draw(seed, offset, i));
    switch (op.dist) {
      case CUDNN_RNG_DISTRIBUTION_BERNOULLI: (*y)[i] = u < op.prob ? 1.0 : 0.0; break;
      case CUDNN_RNG_DISTRIBUTION_UNIFORM: (*y)[i] = op.umin + u * (op.umax - op.umin); break;
      default: {
        const double u2 = unit(draw(seed, offset ^ 0x8000000000000000ull, i));
        (*y)[i] = op.nmean + op.nstd * std::sqrt(-2.0 * std::log(1.0 - u)) * std::cos(2.0 * M_PI * u2);
        break;
      }
    }
  }
}

void run_genstats(const Op& op, const std::vector<double>& x, std::vector<std::vector<double>>* outs) {
  const vc::Layout& X = op.in[0].l;
  const int64_t C = X.dims[1];
  int64_t inner = 1;
  for (int d = 2; d < X.rank; ++d) inner *= X.dims[d];
  std::vector<double> sum(static_cast<size_t>(C), 0.0), sq(static_cast<size_t>(C), 0.0);
  for (size_t i = 0; i < x.size(); ++i) {
    const size_t c = static_cast<size_t>((static_cast<int64_t>(i) / inner) % C);
    sum[c] += x[i], sq[c] += x[i] * x[i];
  }
  outs->assign(2, {});
  (*outs)[op.role[kSum]] = std::move(sum);
  (*outs)[op.role[kSqSum]] = std::move(sq);
}

// Softmax along each row (the last dimension) of x, with an optional sink
// logit per row joining the denominator: m = max(row, sink), e = exp(x - m),
// z = sum(e) + exp(sink - m), y = e / z; stats = m + log z (the log-sum-exp),
// and m and z themselves. A row with nothing unmasked gives y = 0.
struct SoftmaxRows {
  std::vector<double> y, stats, max, sum;
};
SoftmaxRows softmax_rows(const std::vector<double>& x, int64_t cols, const std::vector<double>* sink_per_row) {
  SoftmaxRows r;
  const size_t rows = x.size() / static_cast<size_t>(cols);
  r.y.assign(x.size(), 0.0);
  r.stats.assign(rows, 0.0), r.max.assign(rows, 0.0), r.sum.assign(rows, 0.0);
  for (size_t i = 0; i < rows; ++i) {
    const double* xr = x.data() + i * static_cast<size_t>(cols);
    double m = -INFINITY;
    for (int64_t j = 0; j < cols; ++j) m = std::max(m, xr[j]);
    const double sink = sink_per_row ? (*sink_per_row)[i] : -INFINITY;
    m = std::max(m, sink);
    double z = 0.0;
    if (m != -INFINITY) {
      for (int64_t j = 0; j < cols; ++j) z += std::exp(xr[j] - m);
      if (sink != -INFINITY) z += std::exp(sink - m);
      for (int64_t j = 0; j < cols; ++j) r.y[i * static_cast<size_t>(cols) + static_cast<size_t>(j)] = std::exp(xr[j] - m) / z;
    }
    r.max[i] = m, r.sum[i] = z, r.stats[i] = m + std::log(z);
  }
  return r;
}

void run_softmax(const Op& op, const std::vector<const std::vector<double>*>& in, std::vector<std::vector<double>>* outs) {
  const vc::Layout& X = op.in[op.role[kX]].l;
  const int64_t cols = X.dims[X.rank - 1];
  std::vector<double> sink;
  if (op.role[kSink] >= 0) {
    vc::Layout row = X;
    row.dims[X.rank - 1] = 1;
    const std::vector<size_t> si = broadcast_index(op.in[op.role[kSink]].l, row);
    sink.resize(si.size());
    for (size_t i = 0; i < si.size(); ++i) sink[i] = (*in[op.role[kSink]])[si[i]];
  }
  SoftmaxRows r = softmax_rows(*in[op.role[kX]], cols, sink.empty() ? nullptr : &sink);
  outs->assign(1 + op.more.size(), {});
  (*outs)[0] = std::move(r.y);
  if (op.role[kStats] > 0) (*outs)[op.role[kStats]] = std::move(r.stats);
  if (op.role[kMax] > 0) (*outs)[op.role[kMax]] = std::move(r.max);
  if (op.role[kSumExp] > 0) (*outs)[op.role[kSumExp]] = std::move(r.sum);
}

// One input of a band mask or an attention operation, broadcast onto x and
// read at element i (or a default when absent).
struct Bcast {
  const std::vector<double>* v = nullptr;
  std::vector<size_t> idx;
  double dflt = 0.0;
  double operator[](size_t i) const { return v ? (*v)[idx[i]] : dflt; }
};
Bcast bcast_of(const Op& op, const std::vector<const std::vector<double>*>& in, int role, const vc::Layout& onto, double dflt) {
  Bcast b;
  b.dflt = dflt;
  if (op.role[role] < 0) return b;
  b.v = in[op.role[role]];
  b.idx = broadcast_index(op.in[op.role[role]].l, onto);
  return b;
}

// Whether element (row, col) of batch b's scores is inside a band; see the
// diagonal band mask's description in op_of.
bool in_band(bool left_mode, cudnnPointwiseMode_t cmp, double row, double col, double bound, double sq, double skv,
             bool have_sq, bool have_skv) {
  double a, b;
  if (!left_mode) {
    a = row + bound + (have_skv ? skv : 0.0) - (have_sq ? sq : 0.0);
    b = col;
  } else {
    a = col + bound - (have_skv ? skv : 0.0) + (have_sq ? sq : 0.0);
    b = row;
  }
  switch (cmp) {
    case CUDNN_POINTWISE_CMP_GT: return a > b;
    case CUDNN_POINTWISE_CMP_GE: return a >= b;
    case CUDNN_POINTWISE_CMP_LT: return a < b;
    case CUDNN_POINTWISE_CMP_LE: return a <= b;
    case CUDNN_POINTWISE_CMP_EQ: return a == b;
    case CUDNN_POINTWISE_CMP_NEQ: return a != b;
    default: return a >= b;
  }
}

void run_bandmask(const Op& op, const std::vector<const std::vector<double>*>& in, std::vector<double>* y) {
  const vc::Layout& X = op.in[op.role[kX]].l;
  const std::vector<double>& x = *in[op.role[kX]];
  const bool left_mode = op.role[kLeft] >= 0;
  const Bcast fill = bcast_of(op, in, kFill, X, 0.0), sq = bcast_of(op, in, kSeqQ, X, 0.0), skv = bcast_of(op, in, kSeqKV, X, 0.0);
  const Bcast bound = bcast_of(op, in, left_mode ? kLeft : kShift, X, 0.0);
  y->assign(x.size(), 0.0);
  int64_t at[vc::kMaxRank] = {};
  for (size_t i = 0; i < x.size(); ++i) {
    const double row = static_cast<double>(at[X.rank - 2]), col = static_cast<double>(at[X.rank - 1]);
    const bool keep = in_band(left_mode, op.cmp, row, col, bound[i], sq[i], skv[i], op.role[kSeqQ] >= 0, op.role[kSeqKV] >= 0);
    (*y)[i] = keep ? x[i] : fill[i];
    for (int d = X.rank; d-- > 0;) {
      if (++at[d] < X.dims[d]) break;
      at[d] = 0;
    }
  }
}

/* ---- running a graph ---- */

// Runs a scheduled graph: inputs from the variant pack (device memory, or
// host memory for a by-value scalar), intermediates on the host rounded to
// their declared types, outputs written back. An attention operation's
// score-modifier subgraph runs in a Runner of its own over the same pack.
struct Runner {
  cudnnHandle_t handle;
  const std::map<int64_t, void*>& ptrs;
  std::map<int64_t, std::vector<double>> values;
  std::set<int64_t> consumed;  // tensors some operation reads

  Runner(cudnnHandle_t h, const std::map<int64_t, void*>& p) : handle(h), ptrs(p) {}

  static constexpr const char* fn = "cudnnBackendExecute";

  void* ptr_of(int64_t uid) const {
    auto p = ptrs.find(uid);
    return p == ptrs.end() ? nullptr : p->second;
  }

  // The element offsets of a tensor that is ragged or vectorized (empty:
  // plainly strided, which read()/write() handle themselves).
  cudnnStatus_t offsets_of(const GTensor& t, std::vector<int64_t>* offs) {
    offs->clear();
    if (t.ragged) {
      const std::vector<double>* ro = nullptr;
      cudnnStatus_t s = input(*t.ragged, &ro);
      if (s != CUDNN_STATUS_SUCCESS) return s;
      *offs = element_offsets(t, ro);
    } else if (t.vdim >= 0) {
      *offs = element_offsets(t);
    }
    return CUDNN_STATUS_SUCCESS;
  }

  cudnnStatus_t input(const GTensor& t, const std::vector<double>** out) {
    auto it = values.find(t.uid);
    if (it == values.end()) {
      void* p = ptr_of(t.uid);
      std::vector<double> v;
      if (t.by_value && !p && !t.constant.empty()) {
        v.assign(t.l.count(), vc::decode(t.l.type, t.constant.data()));
      } else if (!p) {
        return refuse(fn, "no data pointer for tensor " + std::to_string(t.uid));
      } else if (t.by_value) {
        v.assign(t.l.count(), vc::decode(t.l.type, static_cast<const uint8_t*>(p)));
      } else {
        std::vector<int64_t> offs;
        cudnnStatus_t s = offsets_of(t, &offs);
        if (s != CUDNN_STATUS_SUCCESS) return s;
        if (offs.empty() ? !vc::read(t.l, p, &v) : !gather(t.l.type, p, offs, &v)) return CUDNN_STATUS_EXECUTION_FAILED;
      }
      it = values.emplace(t.uid, std::move(v)).first;
    }
    *out = &it->second;
    return CUDNN_STATUS_SUCCESS;
  }

  // An operation's result: kept on the host if virtual, else written out
  // (and read back, rounded to its type, if a later operation reads it).
  cudnnStatus_t store(const GTensor& o, std::vector<double> r, double alpha, double beta) {
    if (o.is_virtual) {
      for (double& v : r) v = vc::round_to(o.l.type, alpha * v);
      values[o.uid] = std::move(r);
      return CUDNN_STATUS_SUCCESS;
    }
    void* p = ptr_of(o.uid);
    if (!p) return refuse(fn, "no data pointer for tensor " + std::to_string(o.uid));
    std::vector<int64_t> offs;
    cudnnStatus_t s = offsets_of(o, &offs);
    if (s != CUDNN_STATUS_SUCCESS) return s;
    if (offs.empty()) {
      if (!vc::blend_write(o.l, p, r, alpha, beta)) return CUDNN_STATUS_EXECUTION_FAILED;
    } else {
      std::vector<double> prior;
      if (beta != 0.0 && !gather(o.l.type, p, offs, &prior)) return CUDNN_STATUS_EXECUTION_FAILED;
      for (size_t i = 0; i < r.size(); ++i) r[i] = beta != 0.0 ? alpha * r[i] + beta * prior[i] : alpha * r[i];
      if (!scatter(o.l.type, p, offs, r)) return CUDNN_STATUS_EXECUTION_FAILED;
    }
    if (!consumed.count(o.uid)) return CUDNN_STATUS_SUCCESS;
    std::vector<double> back;
    if (offs.empty() ? !vc::read(o.l, p, &back) : !gather(o.l.type, p, offs, &back)) return CUDNN_STATUS_EXECUTION_FAILED;
    values[o.uid] = std::move(back);
    return CUDNN_STATUS_SUCCESS;
  }

  cudnnStatus_t run(const std::vector<Op>& order) {
    for (const Op& op : order) {
      for (const GTensor& t : op.in) consumed.insert(t.uid);
      if (op.sub)
        for (const Op& so : *op.sub)
          for (const GTensor& t : so.in) consumed.insert(t.uid);
    }
    for (const Op& op : order) {
      cudnnStatus_t s = run_op(op);
      if (s != CUDNN_STATUS_SUCCESS) return s;
    }
    return CUDNN_STATUS_SUCCESS;
  }

  // Applies an attention operation's score modifiers to s ([B, H, Sq, Skv]).
  cudnnStatus_t modify_scores(const Op& op, std::vector<double>* s) {
    if (!op.sub) return CUDNN_STATUS_SUCCESS;
    Runner sub(handle, ptrs);
    sub.values[op.sub_in] = *s;
    cudnnStatus_t st = sub.run(*op.sub);
    if (st != CUDNN_STATUS_SUCCESS) return st;
    auto it = sub.values.find(op.sub_out);
    if (it == sub.values.end() || it->second.size() != s->size())
      return refuse(fn, "the attention score subgraph did not produce its output tensor");
    *s = std::move(it->second);
    return CUDNN_STATUS_SUCCESS;
  }

  cudnnStatus_t run_sdpa(const Op& op, const std::vector<const std::vector<double>*>& in, std::vector<std::vector<double>>* outs);
  cudnnStatus_t run_op(const Op& op);
};

// Attention's dropout decision for element (b, h, i, j) of the [B, H, Sq,
// Skv] probabilities: kept (1) with probability 1 - p.
double dropout_keep(uint64_t seed, uint64_t offset, size_t linear, double p) {
  return unit(draw(seed, offset, linear)) >= p ? 1.0 : 0.0;
}

// Scaled dot-product attention, forward and backward, as cuDNN's fused
// operations define it:
//   S = scale * Q K^T (then the score subgraph: bias, masks, ...), padded
//   rows and columns (sequence lengths) masked; P = softmax(S) with its
//   log-sum-exp kept as the statistics; dropout keeps each P element with
//   probability 1 - p and scales it by 1 / (1 - p); O = P V, P taken in the
//   I/O type as the hardware's second matmul takes it.
// Backward, from O, dO and the statistics: P = exp(S - stats),
//   dV = P^T dO, dP = dO V^T, dS = P (dP - rowsum(dO O)),
//   dQ = scale dS K, dK = scale dS^T Q,
// grouped K/V heads summing their query heads' gradients.
cudnnStatus_t Runner::run_sdpa(const Op& op, const std::vector<const std::vector<double>*>& in,
                               std::vector<std::vector<double>>* outs) {
  const bool fwd = op.kind == Kind::SdpaFwd;
  const vc::Layout &QL = op.in[op.role[kQ]].l, &KL = op.in[op.role[kK]].l, &VL = op.in[op.role[kV]].l;
  const int64_t B = QL.dims[0], Hq = QL.dims[1], Sq = QL.dims[2], D = QL.dims[3], Dv = VL.dims[3];
  const int64_t Hk = KL.dims[1], Hv = VL.dims[1];
  const std::vector<double>& Q = *in[op.role[kQ]];
  std::vector<double> K = *in[op.role[kK]], V = *in[op.role[kV]];
  int64_t Skv = KL.dims[2];
  if (op.role[kPageK] >= 0) {
    // Paged K and V: gather each batch's sequence from its pages.
    const vc::Layout &TK = op.in[op.role[kPageK]].l, &TV = op.in[op.role[kPageV]].l;
    Skv = TK.dims[2] * KL.dims[2];
    const int64_t SkvV = TV.dims[2] * VL.dims[2];
    if (SkvV < Skv) return refuse(fn, "scaled dot-product attention: V's pages cover fewer positions than K's");
    std::vector<double> k2, v2;
    page_gather(KL, K, *in[op.role[kPageK]], TK.dims[2], B, Skv, in[op.role[kSeqKV]], &k2);
    page_gather(VL, V, *in[op.role[kPageV]], TV.dims[2], B, Skv, in[op.role[kSeqKV]], &v2);
    K.swap(k2), V.swap(v2);
  }
  const double scale = op.role[kScale] >= 0 ? (*in[op.role[kScale]])[0] : 1.0;
  const bool padded = op.role[kSeqQ] >= 0 && op.role[kSeqKV] >= 0;
  auto len = [&](int r, int64_t b, int64_t full) {
    return op.role[r] >= 0 ? std::min<int64_t>(full, static_cast<int64_t>((*in[op.role[r]])[static_cast<size_t>(b)])) : full;
  };
  const cudnnDataType_t io = op.in[op.role[kQ]].l.type;
  const size_t nS = static_cast<size_t>(B * Hq * Sq * Skv);
  // S, scaled, in float as the hardware accumulates it, then modified.
  std::vector<double> S(nS, 0.0);
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < Hq; ++h) {
      const int64_t hk = h / (Hq / Hk);
      for (int64_t i = 0; i < Sq; ++i)
        for (int64_t j = 0; j < Skv; ++j) {
          double acc = 0.0;
          for (int64_t d = 0; d < D; ++d)
            acc += Q[static_cast<size_t>(((b * Hq + h) * Sq + i) * D + d)] * K[static_cast<size_t>(((b * Hk + hk) * Skv + j) * D + d)];
          S[static_cast<size_t>(((b * Hq + h) * Sq + i) * Skv + j)] = static_cast<float>(acc * scale);
        }
    }
  cudnnStatus_t st = modify_scores(op, &S);
  if (st != CUDNN_STATUS_SUCCESS) return st;
  std::vector<bool> row_live(static_cast<size_t>(B * Hq * Sq), true);
  if (padded)
    for (int64_t b = 0; b < B; ++b) {
      const int64_t lq = len(kSeqQ, b, Sq), lk = len(kSeqKV, b, Skv);
      for (int64_t h = 0; h < Hq; ++h)
        for (int64_t i = 0; i < Sq; ++i) {
          row_live[static_cast<size_t>((b * Hq + h) * Sq + i)] = i < lq;
          for (int64_t j = 0; j < Skv; ++j)
            if (i >= lq || j >= lk) S[static_cast<size_t>(((b * Hq + h) * Sq + i) * Skv + j)] = -INFINITY;
        }
    }
  outs->assign(1 + op.more.size(), {});
  if (fwd) {
    std::vector<double> sink;
    if (op.role[kSink] >= 0) {
      vc::Layout rows;
      rows.rank = 4, rows.dims[0] = B, rows.dims[1] = Hq, rows.dims[2] = Sq, rows.dims[3] = 1;
      const std::vector<size_t> si = broadcast_index(op.in[op.role[kSink]].l, rows);
      sink.resize(si.size());
      for (size_t i = 0; i < si.size(); ++i) sink[i] = (*in[op.role[kSink]])[si[i]];
    }
    SoftmaxRows sm = softmax_rows(S, Skv, sink.empty() ? nullptr : &sink);
    std::vector<double> P = std::move(sm.y);
    std::vector<double> mask;
    if (op.dropout > 0.0) {
      const uint64_t seed = static_cast<uint64_t>(static_cast<int64_t>((*in[op.role[kSeed]])[0]));
      const uint64_t offset = static_cast<uint64_t>(static_cast<int64_t>((*in[op.role[kOffset]])[0]));
      mask.resize(nS);
      for (size_t e = 0; e < nS; ++e) {
        mask[e] = dropout_keep(seed, offset, e, op.dropout);
        P[e] *= mask[e] / (1.0 - op.dropout);
      }
    }
    std::vector<double> O(static_cast<size_t>(B * Hq * Sq * Dv), 0.0);
    for (int64_t b = 0; b < B; ++b)
      for (int64_t h = 0; h < Hq; ++h) {
        const int64_t hv = h / (Hq / Hv);
        for (int64_t i = 0; i < Sq; ++i) {
          if (!row_live[static_cast<size_t>((b * Hq + h) * Sq + i)]) continue;  // padded rows stay 0
          for (int64_t j = 0; j < Skv; ++j) {
            const double p = vc::round_to(io, P[static_cast<size_t>(((b * Hq + h) * Sq + i) * Skv + j)]);
            if (p == 0.0) continue;
            for (int64_t e = 0; e < Dv; ++e)
              O[static_cast<size_t>(((b * Hq + h) * Sq + i) * Dv + e)] += p * V[static_cast<size_t>(((b * Hv + hv) * Skv + j) * Dv + e)];
          }
        }
      }
    (*outs)[0] = std::move(O);
    if (op.role[kStats] > 0) (*outs)[op.role[kStats]] = std::move(sm.stats);
    if (op.role[kMax] > 0) (*outs)[op.role[kMax]] = std::move(sm.max);
    if (op.role[kSumExp] > 0) (*outs)[op.role[kSumExp]] = std::move(sm.sum);
    if (op.role[kRngDump] > 0) (*outs)[op.role[kRngDump]] = mask.empty() ? std::vector<double>(nS, 1.0) : mask;
    return CUDNN_STATUS_SUCCESS;
  }
  // Backward.
  const std::vector<double> &O = *in[op.role[kO]], &dO = *in[op.role[kDO]], &stats = *in[op.role[kStats]];
  std::vector<double> dQ(Q.size(), 0.0), dK(K.size(), 0.0), dV(V.size(), 0.0);
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < Hq; ++h) {
      const int64_t hk = h / (Hq / Hk), hv = h / (Hq / Hv);
      for (int64_t i = 0; i < Sq; ++i) {
        const size_t row = static_cast<size_t>((b * Hq + h) * Sq + i);
        if (!row_live[row]) continue;
        double Drow = 0.0;
        for (int64_t e = 0; e < Dv; ++e) Drow += dO[row * Dv + e] * O[row * Dv + e];
        for (int64_t j = 0; j < Skv; ++j) {
          const double s = S[row * Skv + j];
          const double p = s == -INFINITY ? 0.0 : std::exp(s - stats[row]);
          if (p == 0.0) continue;
          const double pio = vc::round_to(io, p);
          double dp = 0.0;
          for (int64_t e = 0; e < Dv; ++e) {
            const double g = dO[row * Dv + e];
            dp += g * V[static_cast<size_t>(((b * Hv + hv) * Skv + j) * Dv + e)];
            dV[static_cast<size_t>(((b * Hv + hv) * Skv + j) * Dv + e)] += pio * g;
          }
          const double ds = vc::round_to(io, p * (dp - Drow) * scale);
          for (int64_t d = 0; d < D; ++d) {
            dQ[row * D + d] += ds * K[static_cast<size_t>(((b * Hk + hk) * Skv + j) * D + d)];
            dK[static_cast<size_t>(((b * Hk + hk) * Skv + j) * D + d)] += ds * Q[row * D + d];
          }
        }
      }
    }
  (*outs)[0] = std::move(dQ);
  (*outs)[op.role[kDK]] = std::move(dK);
  (*outs)[op.role[kDV]] = std::move(dV);
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t Runner::run_op(const Op& op) {
  std::vector<const std::vector<double>*> in(op.in.size());
  for (size_t i = 0; i < op.in.size(); ++i) {
    cudnnStatus_t s = input(op.in[i], &in[i]);
    if (s != CUDNN_STATUS_SUCCESS) return s;
  }
  // outs[0] is op.out; outs[k] is op.more[k - 1] (left empty: not written).
  std::vector<std::vector<double>> outs(1 + op.more.size());
  std::vector<double>& r = outs[0];
  double alpha = 1.0, beta = 0.0;
  switch (op.kind) {
    case Kind::ConvFwd: vc::convolve(op.geom, vc::ConvDir::Forward, *in[0], *in[1], &r, op.acc); alpha = op.alpha, beta = op.beta; break;
    case Kind::ConvData: vc::convolve(op.geom, vc::ConvDir::Data, *in[0], *in[1], &r, op.acc); alpha = op.alpha, beta = op.beta; break;
    case Kind::ConvFilter: vc::convolve(op.geom, vc::ConvDir::Filter, *in[0], *in[1], &r, op.acc); alpha = op.alpha, beta = op.beta; break;
    case Kind::Matmul: run_matmul(op, in, &r); break;
    case Kind::Reduction: run_reduction(op, *in[0], &r); break;
    case Kind::PoolFwd:
    case Kind::PoolBwd:
      run_pool(op, in, &outs);
      alpha = op.alpha, beta = op.beta;
      break;
    case Kind::Concat: run_concat(op, in, &r); break;
    case Kind::NormFwd:
    case Kind::NormBwd: run_norm(op, in, &outs); break;
    case Kind::Reshape: run_reshape(op, *in[0], &r); break;
    case Kind::Transpose: run_transpose(op, *in[0], &r); break;
    case Kind::Slice: run_slice(op, *in[0], &r); break;
    case Kind::Rng: run_rng(op, in, &r); break;
    case Kind::GenStats: run_genstats(op, *in[0], &outs); break;
    case Kind::Softmax: run_softmax(op, in, &outs); break;
    case Kind::BandMask: run_bandmask(op, in, &r); break;
    case Kind::PagedLoad: run_paged_load(op, in, &r); break;
    case Kind::SdpaFwd:
    case Kind::SdpaBwd: {
      cudnnStatus_t s = run_sdpa(op, in, &outs);
      if (s != CUDNN_STATUS_SUCCESS) return s;
      break;
    }
    case Kind::Pointwise: {
      const vc::Layout& Y = op.out.l;
      r.resize(Y.count());
      std::vector<std::vector<size_t>> bi;
      for (const GTensor& t : op.in) bi.push_back(broadcast_index(t.l, Y));
      const bool backward = arity(op.pw) == Arity::Backward;
      int64_t at[vc::kMaxRank] = {};
      for (size_t i = 0; i < r.size(); ++i) {
        // Backward modes: in = {dy, x}; the others: {x, b, t}.
        const double x = backward ? (*in[1])[bi[1][i]] : op.alpha * (*in[0])[bi[0][i]];
        const double b = backward ? (*in[0])[bi[0][i]] : in.size() > 1 ? op.alpha2 * (*in[1])[bi[1][i]] : 0.0;
        const double t = in.size() > 2 ? (*in[2])[bi[2][i]] : 0.0;
        r[i] = pointwise(op, x, b, t, op.axis >= 0 && op.axis < Y.rank ? at[op.axis] : 0);
        for (int d = Y.rank; d-- > 0;) {
          if (++at[d] < Y.dims[d]) break;
          at[d] = 0;
        }
      }
      break;
    }
  }
  // Every output after the first, then the first (which alpha and beta blend).
  for (size_t k = 0; k < op.more.size(); ++k) {
    if (outs[k + 1].empty()) continue;
    cudnnStatus_t s = store(op.more[k], std::move(outs[k + 1]), 1.0, 0.0);
    if (s != CUDNN_STATUS_SUCCESS) return s;
  }
  return store(op.out, std::move(r), alpha, beta);
}

cudnnStatus_t execute_graph(cudnnHandle_t handle, const std::vector<Op>& order, const std::map<int64_t, void*>& ptrs) {
  // The work the program queued before this call comes first.
  vc::sync_handle(handle);
  Runner runner(handle, ptrs);
  return runner.run(order);
}

// The one engine: global index 0, for the graph it was made for, owned (with
// its own copy of the graph) by `owner`.
Desc* make_engine(Desc* owner, const Desc* graph) {
  auto e = std::make_unique<Desc>();
  e->type = CUDNN_BACKEND_ENGINE_DESCRIPTOR;
  e->owned.push_back(clone(graph));
  Desc* g = e->owned.back().get();
  e->set(CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &g);
  const int64_t index = 0;
  e->set(CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &index);
  e->finalized = true;
  Desc* raw = e.get();
  owner->owned.push_back(std::move(e));
  return raw;
}

// Copies a descriptor into one the caller made, as GetAttribute does for an
// attribute that is a descriptor.
void copy_into(Desc* dst, const Desc* src) { assign(dst, src); }

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

VGPU_EXPORT cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t type,
                                                      cudnnBackendDescriptor_t* out) {
  if (!out) return CUDNN_STATUS_BAD_PARAM;
  auto* d = new Desc();
  d->type = type;
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_live.insert(d);
  }
  *out = reinterpret_cast<cudnnBackendDescriptor_t>(d);
  if (trace()) std::fprintf(stderr, "[vgpu][trace] cudnnBackendCreateDescriptor(%d) -> %p\n", type, (void*)d);
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t desc) {
  if (!live(desc)) return CUDNN_STATUS_BAD_PARAM;
  auto* d = reinterpret_cast<Desc*>(desc);
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_live.erase(d);
  }
  delete d;
  return CUDNN_STATUS_SUCCESS;
}

// Clears a descriptor back to what it was when it was created.
VGPU_EXPORT cudnnStatus_t cudnnBackendInitialize(cudnnBackendDescriptor_t desc) {
  if (!live(desc)) return CUDNN_STATUS_BAD_PARAM;
  auto* d = reinterpret_cast<Desc*>(desc);
  d->finalized = false;
  d->attrs.clear();
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnBackendSetAttribute(cudnnBackendDescriptor_t desc,
                                                  cudnnBackendAttributeName_t name,
                                                  cudnnBackendAttributeType_t type, int64_t count,
                                                  const void* values) {
  if (!live(desc) || count < 0 || (count && !values)) return CUDNN_STATUS_BAD_PARAM;
  auto* d = reinterpret_cast<Desc*>(desc);
  if (d->finalized) return CUDNN_STATUS_BAD_PARAM;  // a finalized descriptor is read-only
  if (type == CUDNN_TYPE_BACKEND_DESCRIPTOR) {
    // Copies of the descriptors given, which the caller may destroy.
    std::vector<Desc*> mine(static_cast<size_t>(count), nullptr);
    for (int64_t i = 0; i < count; ++i) {
      Desc* given = nullptr;
      std::memcpy(&given, static_cast<const uint8_t*>(values) + i * sizeof(void*), sizeof(void*));
      if (!given) continue;
      if (!live(given)) return CUDNN_STATUS_BAD_PARAM;
      d->owned.push_back(clone(given));
      mine[static_cast<size_t>(i)] = d->owned.back().get();
    }
    d->set(name, type, count, mine.data());
  } else {
    d->set(name, type, count, values);
  }
  if (trace()) {
    std::fprintf(stderr, "[vgpu][trace] cudnnBackendSetAttribute(%p, %d, type %d, %lld)", (void*)d, name, type, (long long)count);
    if (type == CUDNN_TYPE_BACKEND_DESCRIPTOR && count > 0) {
      void* v0 = nullptr;
      std::memcpy(&v0, values, sizeof v0);
      std::fprintf(stderr, " first=%p live=%d", v0, live(v0) ? 1 : 0);
    }
    std::fprintf(stderr, "\n");
  }
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t desc) {
  if (!live(desc)) return CUDNN_STATUS_BAD_PARAM;
  auto* d = reinterpret_cast<Desc*>(desc);
  if (trace()) std::fprintf(stderr, "[vgpu][trace] cudnnBackendFinalize(%p, type %d)\n", (void*)d, d->type);
  std::string why;
  switch (d->type) {
    case CUDNN_BACKEND_TENSOR_DESCRIPTOR: {
      const size_t nd = d->i64s(CUDNN_ATTR_TENSOR_DIMENSIONS).size();
      if (!nd || nd != d->i64s(CUDNN_ATTR_TENSOR_STRIDES).size() || !d->get(CUDNN_ATTR_TENSOR_UNIQUE_ID))
        return CUDNN_STATUS_BAD_PARAM;
      break;
    }
    case CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
      if (!runnable(d, &why)) return refuse("cudnnBackendFinalize(operation graph)", why);
      break;
    case CUDNN_BACKEND_ENGINE_DESCRIPTOR:
      if (d->i64(CUDNN_ATTR_ENGINE_GLOBAL_INDEX, -1) != 0 || !d->desc(CUDNN_ATTR_ENGINE_OPERATION_GRAPH))
        return CUDNN_STATUS_NOT_SUPPORTED;  // only engine 0 exists
      break;
    case CUDNN_BACKEND_ENGINECFG_DESCRIPTOR:
      if (!d->desc(CUDNN_ATTR_ENGINECFG_ENGINE)) return CUDNN_STATUS_BAD_PARAM;
      break;
    case CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
      if (!d->desc(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH)) return CUDNN_STATUS_BAD_PARAM;
      break;
    case CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: {
      const Desc* cfg = d->desc(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
      const Desc* eng = cfg ? cfg->desc(CUDNN_ATTR_ENGINECFG_ENGINE) : nullptr;
      if (!eng || !eng->desc(CUDNN_ATTR_ENGINE_OPERATION_GRAPH)) return CUDNN_STATUS_BAD_PARAM;
      break;
    }
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR: {
      // Shapes that do not fit are the caller's error; a setting this
      // library does not compute is refused by name.
      Op op;
      if (!op_of(d, &op, &why)) {
        if (why.find("not supported") != std::string::npos || why.find("not implemented") != std::string::npos)
          return refuse("cudnnBackendFinalize", why);
        if (trace()) std::fprintf(stderr, "[vgpu] cudnnBackendFinalize: %s\n", why.c_str());
        vgpu_cudnn::set_last_error("cudnnBackendFinalize: " + why);
        return CUDNN_STATUS_BAD_PARAM;
      }
      break;
    }
    case CUDNN_BACKEND_POINTWISE_DESCRIPTOR:
      if (!d->get(CUDNN_ATTR_POINTWISE_MODE) || !d->get(CUDNN_ATTR_POINTWISE_MATH_PREC)) return CUDNN_STATUS_BAD_PARAM;
      if (arity(static_cast<cudnnPointwiseMode_t>(d->i64(CUDNN_ATTR_POINTWISE_MODE))) == Arity::Unknown)
        return refuse("cudnnBackendFinalize", "pointwise mode " + std::to_string(d->i64(CUDNN_ATTR_POINTWISE_MODE)) +
                                                  " is not supported");
      break;
    case CUDNN_BACKEND_MATMUL_DESCRIPTOR:
      if (!d->get(CUDNN_ATTR_MATMUL_COMP_TYPE)) return CUDNN_STATUS_BAD_PARAM;
      break;
    case CUDNN_BACKEND_RNG_DESCRIPTOR: {
      // cuDNN's documented refusals: a negative standard deviation or
      // probability, a uniform range whose maximum is below its minimum.
      const int64_t dist = d->i64(CUDNN_ATTR_RNG_DISTRIBUTION, CUDNN_RNG_DISTRIBUTION_BERNOULLI);
      if ((dist == CUDNN_RNG_DISTRIBUTION_NORMAL && scalar(d, CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION, -1.0) < 0) ||
          (dist == CUDNN_RNG_DISTRIBUTION_UNIFORM && scalar(d, CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM, 1.0) < scalar(d, CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM, 0.0)) ||
          (dist == CUDNN_RNG_DISTRIBUTION_BERNOULLI && scalar(d, CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY, 0.5) < 0) ||
          dist < CUDNN_RNG_DISTRIBUTION_BERNOULLI || dist > CUDNN_RNG_DISTRIBUTION_NORMAL)
        return CUDNN_STATUS_BAD_PARAM;
      break;
    }
    case CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: {
      if (!d->get(CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS)) return CUDNN_STATUS_BAD_PARAM;
      // Interpolation's window is 2 in every dimension: cuDNN documents it
      // for bilinear, and an RTX 3060 (cuDNN 9.27) refuses a window of 1 for
      // nearest too, with NOT_SUPPORTED.
      const int64_t mode = d->i64(CUDNN_ATTR_RESAMPLE_MODE, CUDNN_RESAMPLE_NEAREST);
      if (mode == CUDNN_RESAMPLE_NEAREST || mode == CUDNN_RESAMPLE_BILINEAR) {
        const Attr* w = d->get(CUDNN_ATTR_RESAMPLE_WINDOW_DIMS);
        for (int64_t i = 0; w && i < w->count; ++i) {
          double v = 0;
          if (w->type == CUDNN_TYPE_INT64) { int64_t x; std::memcpy(&x, w->bytes.data() + i * 8, 8); v = static_cast<double>(x); }
          else if (w->type == CUDNN_TYPE_FRACTION) {
            cudnnFraction_t f; std::memcpy(&f, w->bytes.data() + i * sizeof f, sizeof f);
            v = f.denominator ? static_cast<double>(f.numerator) / static_cast<double>(f.denominator) : 0.0;
          }
          if (v != 2.0) return refuse("cudnnBackendFinalize", "resample: interpolation's window must be 2");
        }
      }
      break;
    }
    case CUDNN_BACKEND_REDUCTION_DESCRIPTOR: {
      if (!d->get(CUDNN_ATTR_REDUCTION_OPERATOR) || !d->get(CUDNN_ATTR_REDUCTION_COMP_TYPE)) return CUDNN_STATUS_BAD_PARAM;
      const int64_t r = d->i64(CUDNN_ATTR_REDUCTION_OPERATOR);
      if (r < CUDNN_REDUCE_TENSOR_ADD || r > CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS)
        return refuse("cudnnBackendFinalize", "reduction operator " + std::to_string(r) + " is not supported");
      break;
    }
    case CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR:
    case CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
    case CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR:
    case CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR:
      break;
    default: {
      // Norms, resampling, attention and the rest: the graph they would join
      // is refused anyway, but say which one arrived first.
      const char* n = op_name(d->type);
      return refuse("cudnnBackendFinalize", n ? std::string("the ") + n + " operation is not implemented"
                                              : "descriptor type " + std::to_string(d->type) + " is not implemented");
    }
  }
  d->finalized = true;
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnBackendGetAttribute(cudnnBackendDescriptor_t const desc,
                                                  cudnnBackendAttributeName_t name,
                                                  cudnnBackendAttributeType_t type, int64_t requested,
                                                  int64_t* count, void* values) {
  if (!live(desc) || requested < 0) return CUDNN_STATUS_BAD_PARAM;
  auto* d = reinterpret_cast<Desc*>(desc);
  if (trace()) std::fprintf(stderr, "[vgpu][trace] cudnnBackendGetAttribute(%p, %d, type %d, %lld)\n", (void*)d, name, type, (long long)requested);
  auto give_i64 = [&](int64_t v) {
    if (count) *count = 1;
    if (requested >= 1 && values) std::memcpy(values, &v, sizeof v);
    return CUDNN_STATUS_SUCCESS;
  };
  auto give_none = [&]() {
    if (count) *count = 0;
    return CUDNN_STATUS_SUCCESS;
  };
  switch (name) {
    // What this library computes rather than stores.
    case CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT: {
      std::string why;
      return give_i64(runnable(d, &why) ? 1 : 0);
    }
    case CUDNN_ATTR_ENGINEHEUR_RESULTS: {
      // One configuration: engine 0, no knobs. The caller made the ENGINECFG
      // descriptors it passes; they are filled in.
      Desc* graph = d->desc(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH);
      std::string why;
      if (!graph || !runnable(graph, &why)) {
        if (trace()) std::fprintf(stderr, "[vgpu][trace] engine heuristics: no engine: %s\n", why.c_str());
        return give_none();
      }
      if (count) *count = 1;
      if (requested >= 1 && values) {
        Desc* cfg = nullptr;
        std::memcpy(&cfg, values, sizeof(void*));
        if (!live(cfg)) return CUDNN_STATUS_BAD_PARAM;
        cfg->type = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
        cfg->attrs.clear();
        cfg->owned.clear();
        Desc* engine = make_engine(cfg, graph);
        cfg->set(CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine);
        cfg->finalized = true;
      }
      return CUDNN_STATUS_SUCCESS;
    }
    case CUDNN_ATTR_ENGINECFG_ENGINE: {
      const Desc* engine = d->desc(CUDNN_ATTR_ENGINECFG_ENGINE);
      if (!engine) return CUDNN_STATUS_BAD_PARAM;
      if (count) *count = 1;
      if (requested >= 1 && values) {
        Desc* dst = nullptr;
        std::memcpy(&dst, values, sizeof(void*));
        if (!live(dst)) return CUDNN_STATUS_BAD_PARAM;
        dst->type = CUDNN_BACKEND_ENGINE_DESCRIPTOR;
        copy_into(dst, engine);
      }
      return CUDNN_STATUS_SUCCESS;
    }
    case CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG: {
      const Desc* cfg = d->desc(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
      if (!cfg) return CUDNN_STATUS_BAD_PARAM;
      if (count) *count = 1;
      if (requested >= 1 && values) {
        Desc* dst = nullptr;
        std::memcpy(&dst, values, sizeof(void*));
        if (!live(dst)) return CUDNN_STATUS_BAD_PARAM;
        dst->type = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
        copy_into(dst, cfg);
      }
      return CUDNN_STATUS_SUCCESS;
    }
    case CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
    case CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE:
    case CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED:
      return give_i64(0);  // computed on the host: no workspace
    // Engine 0 has no knobs, numerical notes or behaviour notes to report,
    // and a plan computes no intermediates.
    case CUDNN_ATTR_ENGINE_KNOB_INFO:
    case CUDNN_ATTR_ENGINE_NUMERICAL_NOTE:
    case CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
    case CUDNN_ATTR_ENGINE_LAYOUT_INFO:
    case CUDNN_ATTR_ENGINECFG_KNOB_CHOICES:
    case CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS:
    case CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS:
      if (!d->get(name)) return give_none();
      break;
    case CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION: {
      // A plan's serialized form: enough to name it (engine 0), not to
      // rebuild it -- deserializing plans is not supported.
      static const char kJson[] = "{\"engine\":0,\"library\":\"VirtualGPU\"}";
      if (count) *count = sizeof kJson;
      if (values && requested > 0) {
        const size_t n = std::min<size_t>(static_cast<size_t>(requested), sizeof kJson);
        std::memcpy(values, kJson, n);
      }
      return CUDNN_STATUS_SUCCESS;
    }
    default:
      break;
  }
  // Anything else: what the caller set, handed back.
  const Attr* a = d->get(name);
  if (!a) {
    if (trace() || !quiet())
      std::fprintf(stderr, "[vgpu] cudnnBackendGetAttribute: attribute %d of descriptor type %d is not answered\n",
                   name, d->type);
    return CUDNN_STATUS_NOT_SUPPORTED;
  }
  if (count) *count = a->count;
  if (values && requested > 0) {
    if (a->type == CUDNN_TYPE_BACKEND_DESCRIPTOR && type == CUDNN_TYPE_BACKEND_DESCRIPTOR) {
      // Descriptor-valued: copy each into the caller's own.
      for (int64_t i = 0; i < std::min(requested, a->count); ++i) {
        Desc* src = d->desc(name, static_cast<size_t>(i));
        Desc* dst = nullptr;
        std::memcpy(&dst, static_cast<uint8_t*>(values) + i * sizeof(void*), sizeof(void*));
        if (!src || !live(dst)) return CUDNN_STATUS_BAD_PARAM;
        dst->type = src->type;
        copy_into(dst, src);
      }
    } else {
      const size_t n = std::min<size_t>(static_cast<size_t>(requested) * elem_size(type), a->bytes.size());
      std::memcpy(values, a->bytes.data(), n);
    }
  }
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnBackendExecute(cudnnHandle_t handle, cudnnBackendDescriptor_t plan,
                                             cudnnBackendDescriptor_t pack) {
  if (!live(plan) || !live(pack)) return CUDNN_STATUS_BAD_PARAM;
  const auto* p = reinterpret_cast<const Desc*>(plan);
  const auto* v = reinterpret_cast<const Desc*>(pack);
  if (!p->finalized || !v->finalized) return CUDNN_STATUS_BAD_PARAM;
  const Desc* cfg = p->desc(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
  const Desc* eng = cfg ? cfg->desc(CUDNN_ATTR_ENGINECFG_ENGINE) : nullptr;
  const Desc* graph = eng ? eng->desc(CUDNN_ATTR_ENGINE_OPERATION_GRAPH) : nullptr;
  if (!graph) return CUDNN_STATUS_BAD_PARAM;
  std::string why;
  std::vector<Op> order;
  if (!schedule(graph, &order, &why)) return refuse("cudnnBackendExecute", why);
  // The variant pack: unique ids and the device pointers that go with them.
  const std::vector<int64_t> uids = v->i64s(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS);
  const Attr* ptr_attr = v->get(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS);
  if (!ptr_attr || static_cast<size_t>(ptr_attr->count) != uids.size()) return CUDNN_STATUS_BAD_PARAM;
  std::map<int64_t, void*> ptrs;
  for (size_t i = 0; i < uids.size(); ++i) {
    void* ptr = nullptr;
    std::memcpy(&ptr, ptr_attr->bytes.data() + i * sizeof(void*), sizeof(void*));
    ptrs[uids[i]] = ptr;
  }
  return execute_graph(handle, order, ptrs);
}

// cudnn-frontend references cudnnReorderFilterAndBias (for INT8x32 filters),
// so a program built with it needs the symbol to load. Weak: the classic
// API's own definition, where there is one, takes its place.
extern "C" __attribute__((visibility("default"), weak)) cudnnStatus_t cudnnReorderFilterAndBias(
    cudnnHandle_t, const cudnnFilterDescriptor_t, cudnnReorderType_t, const void*, void*, int, const void*, void*) {
  return vgpu_cudnn::fail(CUDNN_STATUS_NOT_SUPPORTED, "cudnnReorderFilterAndBias", "filter reordering is not supported");
}
