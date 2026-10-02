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
//     matmul, pointwise, reduction and normalization (layer, instance, batch,
//     RMS; forward and backward) operations, joined through virtual tensors --
//     cudnn-frontend's conv-bias-activation, dgrad-drelu, matmul-epilogue,
//     reduction and norm patterns, and PyTorch's single convolutions. A graph
//     with any other operation is refused when it is finalized, by name,
//     rather than accepted and run wrongly.
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
  vc::Layout l;
  int64_t uid = 0;
  bool is_virtual = false, by_value = false;
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
  if (d->i64(CUDNN_ATTR_TENSOR_VECTOR_COUNT, 1) != 1) {
    *why = "tensor " + std::to_string(t->uid) + " is vectorized, which is not supported";
    return false;
  }
  if (d->get(CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC)) {
    *why = "tensor " + std::to_string(t->uid) + " is ragged, which is not supported";
    return false;
  }
  if (!vc::storable(t->l.type)) {
    *why = std::string("tensor ") + std::to_string(t->uid) + " has data type " + vc::type_name(t->l.type) +
           ", which is not supported";
    return false;
  }
  t->l.rank = static_cast<int>(dims.size());
  for (int i = 0; i < t->l.rank; ++i) {
    if (dims[i] <= 0 || strides[i] <= 0) { *why = "tensor " + std::to_string(t->uid) + " has a nonpositive extent or stride"; return false; }
    t->l.dims[i] = dims[i], t->l.strides[i] = strides[i];
  }
  return true;
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
enum class Kind { ConvFwd, ConvData, ConvFilter, Matmul, Pointwise, Reduction, NormFwd, NormBwd };

// A normalization's tensors, by role: which of an Op's inputs and outputs
// each is (-1 when the graph does not give it).
enum NormRole { kX, kScale, kBias, kEps, kMean, kInv, kFactor, kRunMeanIn, kRunVarIn, kDy,
                kY, kMeanOut, kInvOut, kRunMeanOut, kRunVarOut, kDx, kDscale, kDbias, kRoles };

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
  cudnnBackendNormMode_t norm = CUDNN_LAYER_NORM;
  bool training = false;
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

// Every dimension of a is c's or 1, at c's rank.
bool broadcasts(const vc::Layout& a, const vc::Layout& c) {
  if (a.rank != c.rank) return false;
  for (int i = 0; i < a.rank; ++i)
    if (a.dims[i] != c.dims[i] && a.dims[i] != 1) return false;
  return true;
}

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
    // Asymmetric padding: the output extent would follow from both sides.
    if (!post.empty() && post != pre) { *why = "asymmetric padding (post-paddings differ from pre-paddings) is not supported"; return false; }
    const bool flip = c->i64(CUDNN_ATTR_CONVOLUTION_CONV_MODE) == CUDNN_CONVOLUTION;
    if (!vc::conv_geometry(x, w, y, nsp, pre.data(), str.data(), dil.data(), flip, &op->geom, why)) return false;
    const cudnnDataType_t ct = static_cast<cudnnDataType_t>(c->i64(CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_DATA_FLOAT));
    op->acc = ct == CUDNN_DATA_HALF ? vc::Accum::Half : vc::Accum::Exact;
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
      if (d->desc(CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC) || d->desc(CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC) ||
          d->desc(CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC)) {
        *why = "matmul: per-batch M/N/K overrides (ragged batches) are not supported";
        return false;
      }
      const Desc* md = d->desc(CUDNN_ATTR_OPERATION_MATMUL_DESC);
      if (!md || md->type != CUDNN_BACKEND_MATMUL_DESCRIPTOR) { *why = "matmul: the matmul descriptor is missing"; return false; }
      const auto ct = static_cast<cudnnDataType_t>(md->i64(CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_DATA_FLOAT));
      op->acc = ct == CUDNN_DATA_HALF ? vc::Accum::Half : vc::Accum::Exact;
      const vc::Layout &A = a.l, &B = b.l, &C = op->out.l;
      const int r = C.rank;
      if (r < 2 || A.rank != r || B.rank != r || A.dims[r - 1] != B.dims[r - 2] || A.dims[r - 2] != C.dims[r - 2] ||
          B.dims[r - 1] != C.dims[r - 1]) {
        *why = "matmul: a [.., M, K], b [.., K, N] and c [.., M, N] do not fit together";
        return false;
      }
      for (int i = 0; i < r - 2; ++i)
        if ((A.dims[i] != C.dims[i] && A.dims[i] != 1) || (B.dims[i] != C.dims[i] && B.dims[i] != 1)) {
          *why = "matmul: a batch dimension of a or b is neither c's nor 1";
          return false;
        }
      op->in = {a, b};
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
      if (!broadcasts(op->out.l, x.l)) {
        *why = "reduction: every dimension of y must be x's or 1 (grouped reductions are not supported)";
        return false;
      }
      op->in = {x};
      return true;
    }
    case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR: {
      const bool fwd = d->type == CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR;
      op->kind = fwd ? Kind::NormFwd : Kind::NormBwd;
      op->norm = static_cast<cudnnBackendNormMode_t>(d->i64(fwd ? CUDNN_ATTR_OPERATION_NORM_FWD_MODE : CUDNN_ATTR_OPERATION_NORM_BWD_MODE));
      op->training = fwd && d->i64(CUDNN_ATTR_OPERATION_NORM_FWD_PHASE) == CUDNN_NORM_FWD_TRAINING;
      const char* mode = op->norm == CUDNN_LAYER_NORM ? "layer" : op->norm == CUDNN_INSTANCE_NORM ? "instance"
                       : op->norm == CUDNN_BATCH_NORM ? "batch" : op->norm == CUDNN_RMS_NORM ? "RMS" : nullptr;
      if (!mode) { *why = "normalization mode " + std::to_string(op->norm) + " (group norm) is not supported"; return false; }
      if (fwd && op->norm == CUDNN_BATCH_NORM && !op->training) {
        *why = "batch normalization's inference phase is not supported (nor is it by cuDNN's graph API)";
        return false;
      }
      if (d->get(fwd ? CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS : CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS)) {
        *why = "multi-GPU normalization (peer statistics) is not supported";
        return false;
      }
      // Each role's tensor, if set: inputs go to in, outputs to out (the
      // first, y or dx) and then more.
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
        if (ok && (op->role[kInv] < 0 || (!rms && op->role[kMean] < 0))) {
          *why = "normalization backward: only the form with the saved mean and inverse variance is supported";
          return false;
        }
      }
      if (!ok) return false;
      // Shapes: y (dx) is x's; scale, bias and the statistics broadcast onto x.
      const vc::Layout& X = op->in[op->role[kX]].l;
      if (!op->out.l.same_dims(X)) { *why = std::string(mode) + " normalization: the output's shape is not x's"; return false; }
      for (int r : {kScale, kBias, kMean, kInv, kRunMeanIn, kRunVarIn, kDy})
        if (op->role[r] >= 0 && !broadcasts(op->in[op->role[r]].l, X)) {
          *why = std::string(mode) + " normalization: a parameter or statistic does not broadcast onto x";
          return false;
        }
      for (const GTensor& t : op->more)
        if (!broadcasts(t.l, X)) { *why = std::string(mode) + " normalization: an output does not broadcast onto x"; return false; }
      if (op->role[kEps] >= 0 && op->in[op->role[kEps]].l.count() != 1) {
        *why = std::string(mode) + " normalization: epsilon is not a scalar";
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
// pack supplies (not virtual).
bool schedule(const Desc* graph, std::vector<Op>* order, std::string* why) {
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
      if (t.is_virtual && !producer.count(t.uid)) {
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

// Whether this library can run a graph, and if not, why.
bool runnable(const Desc* graph, std::string* why) {
  std::vector<Op> order;
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
// broadcast onto it.
std::vector<size_t> broadcast_index(const vc::Layout& a, const vc::Layout& c) {
  std::vector<size_t> idx(c.count());
  int64_t at[vc::kMaxRank] = {};
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

void run_matmul(const Op& op, const std::vector<double>& a, const std::vector<double>& b, std::vector<double>* c) {
  const vc::Layout &A = op.in[0].l, &B = op.in[1].l, &C = op.out.l;
  const int r = C.rank;
  const int64_t M = C.dims[r - 2], N = C.dims[r - 1], K = A.dims[r - 1];
  // The batch: C's leading dimensions, with a's and b's broadcast onto them.
  vc::Layout cb, ab, bb;
  cb.rank = ab.rank = bb.rank = r - 2;
  for (int i = 0; i < r - 2; ++i) cb.dims[i] = C.dims[i], ab.dims[i] = A.dims[i], bb.dims[i] = B.dims[i];
  const size_t batches = r > 2 ? cb.count() : 1;
  const std::vector<size_t> ai = r > 2 ? broadcast_index(ab, cb) : std::vector<size_t>{0};
  const std::vector<size_t> bi = r > 2 ? broadcast_index(bb, cb) : std::vector<size_t>{0};
  c->assign(batches * M * N, 0.0);
  for (size_t q = 0; q < batches; ++q) {
    const double* pa = a.data() + ai[q] * M * K;
    const double* pb = b.data() + bi[q] * K * N;
    double* pc = c->data() + q * M * N;
    for (int64_t i = 0; i < M; ++i)
      for (int64_t j = 0; j < N; ++j) {
        double s = 0.0;
        for (int64_t k = 0; k < K; ++k) {
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
    for (int d = 0; d < X.rank; ++d) o = o * Y.dims[d] + (Y.dims[d] == 1 ? 0 : at[d]);
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
  if (op.kind == Kind::NormFwd && op.training) {
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

// Runs a scheduled graph: inputs from the variant pack (device memory, or
// host memory for a by-value scalar), intermediates on the host rounded to
// their declared types, outputs written back.
cudnnStatus_t execute_graph(cudnnHandle_t handle, const std::vector<Op>& order, const std::map<int64_t, void*>& ptrs) {
  static const char* fn = "cudnnBackendExecute";
  std::map<int64_t, std::vector<double>> values;
  std::set<int64_t> consumed;  // tensors some operation reads
  for (const Op& op : order)
    for (const GTensor& t : op.in) consumed.insert(t.uid);
  // The work the program queued before this call comes first.
  vc::sync_handle(handle);
  auto input = [&](const GTensor& t, const std::vector<double>** out) -> cudnnStatus_t {
    auto it = values.find(t.uid);
    if (it == values.end()) {
      auto p = ptrs.find(t.uid);
      if (p == ptrs.end() || !p->second) return refuse(fn, "no data pointer for tensor " + std::to_string(t.uid));
      std::vector<double> v;
      if (t.by_value) {
        v.assign(t.l.count(), vc::decode(t.l.type, static_cast<const uint8_t*>(p->second)));
      } else if (!vc::read(t.l, p->second, &v)) {
        return CUDNN_STATUS_EXECUTION_FAILED;
      }
      it = values.emplace(t.uid, std::move(v)).first;
    }
    *out = &it->second;
    return CUDNN_STATUS_SUCCESS;
  };
  // An operation's result: kept on the host if virtual, else written out
  // (and read back, rounded to its type, if a later operation reads it).
  auto store = [&](const GTensor& o, std::vector<double> r, double alpha, double beta) -> cudnnStatus_t {
    if (o.is_virtual) {
      for (double& v : r) v = vc::round_to(o.l.type, alpha * v);
      values[o.uid] = std::move(r);
      return CUDNN_STATUS_SUCCESS;
    }
    auto p = ptrs.find(o.uid);
    if (p == ptrs.end() || !p->second) return refuse(fn, "no data pointer for tensor " + std::to_string(o.uid));
    if (!vc::blend_write(o.l, p->second, r, alpha, beta)) return CUDNN_STATUS_EXECUTION_FAILED;
    if (!consumed.count(o.uid)) return CUDNN_STATUS_SUCCESS;
    std::vector<double> back;
    if (!vc::read(o.l, p->second, &back)) return CUDNN_STATUS_EXECUTION_FAILED;
    values[o.uid] = std::move(back);
    return CUDNN_STATUS_SUCCESS;
  };
  for (const Op& op : order) {
    std::vector<const std::vector<double>*> in(op.in.size());
    for (size_t i = 0; i < op.in.size(); ++i) {
      cudnnStatus_t s = input(op.in[i], &in[i]);
      if (s != CUDNN_STATUS_SUCCESS) return s;
    }
    std::vector<double> r;
    double alpha = 1.0, beta = 0.0;
    switch (op.kind) {
      case Kind::ConvFwd: vc::convolve(op.geom, vc::ConvDir::Forward, *in[0], *in[1], &r, op.acc); alpha = op.alpha, beta = op.beta; break;
      case Kind::ConvData: vc::convolve(op.geom, vc::ConvDir::Data, *in[0], *in[1], &r, op.acc); alpha = op.alpha, beta = op.beta; break;
      case Kind::ConvFilter: vc::convolve(op.geom, vc::ConvDir::Filter, *in[0], *in[1], &r, op.acc); alpha = op.alpha, beta = op.beta; break;
      case Kind::Matmul: run_matmul(op, *in[0], *in[1], &r); break;
      case Kind::Reduction: run_reduction(op, *in[0], &r); break;
      case Kind::NormFwd:
      case Kind::NormBwd: {
        std::vector<std::vector<double>> outs;
        run_norm(op, in, &outs);
        // Every output after the first is stored here; the first goes the
        // common way below.
        for (size_t k = 0; k < op.more.size(); ++k) {
          cudnnStatus_t s = store(op.more[k], std::move(outs[k + 1]), 1.0, 0.0);
          if (s != CUDNN_STATUS_SUCCESS) return s;
        }
        r = std::move(outs[0]);
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
    cudnnStatus_t s = store(op.out, std::move(r), alpha, beta);
    if (s != CUDNN_STATUS_SUCCESS) return s;
  }
  return CUDNN_STATUS_SUCCESS;
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
    case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR: {
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
