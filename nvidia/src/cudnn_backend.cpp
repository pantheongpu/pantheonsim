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
//     a single convolution forward, backward-data or backward-filter. A graph
//     with anything else is refused when it is finalized, by name, rather
//     than accepted and run wrongly.
//   - Execute computes on the host, as the classic API here does (see
//     cudnn_api.cpp): the tensors are copied out of device memory, the
//     convolution is done in float, and the result copied back, after the
//     handle's stream has finished what it was given. Any number of spatial
//     dimensions from one to three, groups, dilation, padding and any
//     strides; float, half, bfloat16 and double data.
//
// Declarations and values are cuDNN's own headers (nvidia/third_party/
// cudnn_include); nothing here is NVIDIA's code.
#include <cudnn.h>

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

bool is_conv_op(cudnnBackendDescriptorType_t t) {
  return t == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR ||
         t == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR ||
         t == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
}

// ---- tensors on the host ------------------------------------------------------

struct Tensor {
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  std::vector<int64_t> dims, strides;  // in elements
  int64_t uid = 0;
  size_t count() const {
    size_t n = 1;
    for (int64_t d : dims) n *= static_cast<size_t>(d);
    return n;
  }
  // Elements from the first to one past the last, as the strides lay them out.
  size_t span() const {
    size_t s = 1;
    for (size_t i = 0; i < dims.size(); ++i) s += static_cast<size_t>((dims[i] - 1) * strides[i]);
    return s;
  }
};

bool tensor_of(const Desc* d, Tensor* t) {
  if (!d || d->type != CUDNN_BACKEND_TENSOR_DESCRIPTOR) return false;
  t->type = static_cast<cudnnDataType_t>(d->i64(CUDNN_ATTR_TENSOR_DATA_TYPE));
  t->dims = d->i64s(CUDNN_ATTR_TENSOR_DIMENSIONS);
  t->strides = d->i64s(CUDNN_ATTR_TENSOR_STRIDES);
  t->uid = d->i64(CUDNN_ATTR_TENSOR_UNIQUE_ID);
  return t->dims.size() >= 3 && t->dims.size() == t->strides.size();
}

size_t type_bytes(cudnnDataType_t t) {
  switch (t) {
    case CUDNN_DATA_HALF:
    case CUDNN_DATA_BFLOAT16: return 2;
    case CUDNN_DATA_DOUBLE: return 8;
    default: return 4;
  }
}
bool supported_type(cudnnDataType_t t) {
  return t == CUDNN_DATA_FLOAT || t == CUDNN_DATA_HALF || t == CUDNN_DATA_BFLOAT16 || t == CUDNN_DATA_DOUBLE;
}

float half_to_float(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) bits = sign;
    else {  // subnormal: normalize
      int e = -1;
      uint32_t m = man;
      do { ++e; m <<= 1; } while (!(m & 0x400));
      bits = sign | ((127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
    }
  } else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
  else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
uint16_t float_to_half(float f) {  // round to nearest even
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
  uint32_t man = x & 0x7fffff;
  if (((x >> 23) & 0xff) == 0xff) return static_cast<uint16_t>(sign | 0x7c00u | (man ? 0x200u : 0));
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    man |= 0x800000;
    const int shift = 14 - exp;
    uint32_t h = man >> shift;
    const uint32_t rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) ++h;
    return static_cast<uint16_t>(sign | h);
  }
  uint32_t h = (static_cast<uint32_t>(exp) << 10) | (man >> 13);
  const uint32_t rem = man & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
  return static_cast<uint16_t>(sign | h);
}
float bf16_to_float(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
uint16_t float_to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7f800000u) == 0x7f800000u && (x & 0x7fffff)) return static_cast<uint16_t>((x >> 16) | 0x40);
  x += 0x7fff + ((x >> 16) & 1);
  return static_cast<uint16_t>(x >> 16);
}

// A tensor's elements in logical order (dims[0] outermost), as float.
struct Host {
  Tensor t;
  void* dev = nullptr;
  std::vector<uint8_t> raw;   // the span, as it is in device memory
  std::vector<float> v;       // logical order
};

// Walks every element's logical index and its offset in the span.
template <class F>
void each(const Tensor& t, F&& f) {
  const size_t rank = t.dims.size();
  std::vector<int64_t> idx(rank, 0);
  const size_t n = t.count();
  for (size_t i = 0; i < n; ++i) {
    size_t off = 0;
    for (size_t d = 0; d < rank; ++d) off += static_cast<size_t>(idx[d] * t.strides[d]);
    f(i, off);
    for (size_t d = rank; d-- > 0;) {
      if (++idx[d] < t.dims[d]) break;
      idx[d] = 0;
    }
  }
}

bool fetch(Host* h, bool read_values) {
  const size_t eb = type_bytes(h->t.type);
  h->raw.assign(h->t.span() * eb, 0);
  if (cudaMemcpy(h->raw.data(), h->dev, h->raw.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
  h->v.assign(h->t.count(), 0.0f);
  if (!read_values) return true;
  each(h->t, [&](size_t i, size_t off) {
    const uint8_t* p = h->raw.data() + off * eb;
    switch (h->t.type) {
      case CUDNN_DATA_HALF: { uint16_t b; std::memcpy(&b, p, 2); h->v[i] = half_to_float(b); break; }
      case CUDNN_DATA_BFLOAT16: { uint16_t b; std::memcpy(&b, p, 2); h->v[i] = bf16_to_float(b); break; }
      case CUDNN_DATA_DOUBLE: { double d; std::memcpy(&d, p, 8); h->v[i] = static_cast<float>(d); break; }
      default: std::memcpy(&h->v[i], p, 4); break;
    }
  });
  return true;
}

// Writes the logical values back over the span, leaving any gaps the strides
// leave as they were.
bool flush(Host* h) {
  const size_t eb = type_bytes(h->t.type);
  each(h->t, [&](size_t i, size_t off) {
    uint8_t* p = h->raw.data() + off * eb;
    switch (h->t.type) {
      case CUDNN_DATA_HALF: { const uint16_t b = float_to_half(h->v[i]); std::memcpy(p, &b, 2); break; }
      case CUDNN_DATA_BFLOAT16: { const uint16_t b = float_to_bf16(h->v[i]); std::memcpy(p, &b, 2); break; }
      case CUDNN_DATA_DOUBLE: { const double d = h->v[i]; std::memcpy(p, &d, 8); break; }
      default: std::memcpy(p, &h->v[i], 4); break;
    }
  });
  return cudaMemcpy(h->dev, h->raw.data(), h->raw.size(), cudaMemcpyHostToDevice) == cudaSuccess;
}

// ---- convolution --------------------------------------------------------------

struct Conv {
  int spatial = 2;
  std::vector<int64_t> pad, stride, dilation;
  bool flip = false;  // CUDNN_CONVOLUTION flips the filter; cross-correlation does not
};

bool conv_of(const Desc* d, Conv* c) {
  if (!d || d->type != CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR) return false;
  c->spatial = static_cast<int>(d->i64(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS));
  c->pad = d->i64s(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS);
  c->stride = d->i64s(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES);
  c->dilation = d->i64s(CUDNN_ATTR_CONVOLUTION_DILATIONS);
  c->flip = d->i64(CUDNN_ATTR_CONVOLUTION_CONV_MODE) == CUDNN_CONVOLUTION;
  const size_t s = static_cast<size_t>(c->spatial);
  return c->spatial >= 1 && c->spatial <= 3 && c->pad.size() == s && c->stride.size() == s &&
         c->dilation.size() == s;
}

// The three convolutions share one loop: every (batch, output channel, output
// position, input channel, filter tap) that lands inside the input. Up to
// three spatial dimensions; fewer are padded with extent 1.
struct Geometry {
  int64_t N, C, K, G, Cg, Kg;
  int64_t in[3] = {1, 1, 1}, out[3] = {1, 1, 1}, flt[3] = {1, 1, 1};
  int64_t pad[3] = {0, 0, 0}, str[3] = {1, 1, 1}, dil[3] = {1, 1, 1};
  bool flip = false;
};

bool geometry(const Tensor& x, const Tensor& w, const Tensor& y, const Conv& c, Geometry* g, std::string* why) {
  const size_t s = static_cast<size_t>(c.spatial);
  if (x.dims.size() != s + 2 || w.dims.size() != s + 2 || y.dims.size() != s + 2) {
    *why = "tensor ranks do not match the convolution's spatial dimensions";
    return false;
  }
  g->N = x.dims[0], g->C = x.dims[1], g->K = w.dims[0], g->Cg = w.dims[1];
  if (g->Cg <= 0 || g->C % g->Cg) { *why = "input channels are not a multiple of the filter's"; return false; }
  g->G = g->C / g->Cg;
  if (g->K % g->G) { *why = "output channels are not a multiple of the group count"; return false; }
  g->Kg = g->K / g->G;
  if (y.dims[0] != g->N || y.dims[1] != g->K) { *why = "the output's batch or channels do not match"; return false; }
  for (size_t i = 0; i < s; ++i) {
    const size_t k = 3 - s + i;  // right-align into three
    g->in[k] = x.dims[2 + i], g->out[k] = y.dims[2 + i], g->flt[k] = w.dims[2 + i];
    g->pad[k] = c.pad[i], g->str[k] = c.stride[i], g->dil[k] = c.dilation[i];
  }
  g->flip = c.flip;
  return true;
}

enum class Dir { Forward, Data, Filter };

// Logical, packed indexes into the tensors' value arrays.
inline size_t xi(const Geometry& g, int64_t n, int64_t c, const int64_t p[3]) {
  return static_cast<size_t>(((n * g.C + c) * g.in[0] + p[0]) * g.in[1] + p[1]) * g.in[2] + p[2];
}
inline size_t wi(const Geometry& g, int64_t k, int64_t c, const int64_t f[3]) {
  return static_cast<size_t>(((k * g.Cg + c) * g.flt[0] + f[0]) * g.flt[1] + f[1]) * g.flt[2] + f[2];
}
inline size_t yi(const Geometry& g, int64_t n, int64_t k, const int64_t o[3]) {
  return static_cast<size_t>(((n * g.K + k) * g.out[0] + o[0]) * g.out[1] + o[1]) * g.out[2] + o[2];
}

// acc is the result tensor's size; the caller scales it into the output.
void convolve(const Geometry& g, Dir dir, const std::vector<float>& a, const std::vector<float>& b,
              std::vector<double>& acc) {
  int64_t o[3], f[3], p[3];
  for (int64_t n = 0; n < g.N; ++n)
    for (int64_t k = 0; k < g.K; ++k) {
      const int64_t grp = k / g.Kg;
      for (o[0] = 0; o[0] < g.out[0]; ++o[0])
        for (o[1] = 0; o[1] < g.out[1]; ++o[1])
          for (o[2] = 0; o[2] < g.out[2]; ++o[2])
            for (int64_t ci = 0; ci < g.Cg; ++ci)
              for (f[0] = 0; f[0] < g.flt[0]; ++f[0])
                for (f[1] = 0; f[1] < g.flt[1]; ++f[1])
                  for (f[2] = 0; f[2] < g.flt[2]; ++f[2]) {
                    bool inside = true;
                    for (int d = 0; d < 3 && inside; ++d) {
                      const int64_t tap = g.flip ? g.flt[d] - 1 - f[d] : f[d];
                      p[d] = o[d] * g.str[d] - g.pad[d] + tap * g.dil[d];
                      inside = p[d] >= 0 && p[d] < g.in[d];
                    }
                    if (!inside) continue;
                    const int64_t c = grp * g.Cg + ci;
                    switch (dir) {
                      case Dir::Forward:  // a = x, b = w, acc = y
                        acc[yi(g, n, k, o)] += static_cast<double>(a[xi(g, n, c, p)]) * b[wi(g, k, ci, f)];
                        break;
                      case Dir::Data:     // a = dy, b = w, acc = dx
                        acc[xi(g, n, c, p)] += static_cast<double>(a[yi(g, n, k, o)]) * b[wi(g, k, ci, f)];
                        break;
                      case Dir::Filter:   // a = dy, b = x, acc = dw
                        acc[wi(g, k, ci, f)] += static_cast<double>(a[yi(g, n, k, o)]) * b[xi(g, n, c, p)];
                        break;
                    }
                  }
    }
}

double scalar(const Desc* op, cudnnBackendAttributeName_t n, double dflt) {
  const Attr* a = op->get(n);
  if (!a || a->count < 1) return dflt;
  if (a->type == CUDNN_TYPE_DOUBLE) { double d; std::memcpy(&d, a->bytes.data(), 8); return d; }
  if (a->type == CUDNN_TYPE_FLOAT) { float f; std::memcpy(&f, a->bytes.data(), 4); return f; }
  return dflt;
}

struct ConvOp {
  Dir dir;
  const Desc *x, *w, *y, *conv;  // for Data: y is dy, x is dx; for Filter: w is dw
  double alpha, beta;
};

bool conv_op_of(const Desc* op, ConvOp* c) {
  switch (op->type) {
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
      *c = {Dir::Forward, op->desc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X),
            op->desc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W), op->desc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y),
            op->desc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC),
            scalar(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, 1.0),
            scalar(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, 0.0)};
      break;
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
      *c = {Dir::Data, op->desc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX),
            op->desc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W), op->desc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY),
            op->desc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC),
            scalar(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, 1.0),
            scalar(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, 0.0)};
      break;
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
      *c = {Dir::Filter, op->desc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X),
            op->desc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW),
            op->desc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY),
            op->desc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC),
            scalar(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, 1.0),
            scalar(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, 0.0)};
      break;
    default: return false;
  }
  return c->x && c->w && c->y && c->conv;
}

// Whether this library can run a graph, and if not, why.
bool runnable(const Desc* graph, std::string* why) {
  const Attr* ops = graph->get(CUDNN_ATTR_OPERATIONGRAPH_OPS);
  if (!ops || ops->count != 1) {
    *why = "only a graph of one convolution is supported (this one has " +
           std::to_string(ops ? ops->count : 0) + " operations)";
    return false;
  }
  const Desc* op = graph->desc(CUDNN_ATTR_OPERATIONGRAPH_OPS, 0);
  ConvOp c;
  if (!op || !conv_op_of(op, &c)) {
    *why = "only convolution forward, backward-data and backward-filter operations are supported";
    return false;
  }
  Tensor x, w, y;
  Conv cv;
  Geometry g;
  const bool tx = tensor_of(c.x, &x), tw = tensor_of(c.w, &w), ty = tensor_of(c.y, &y), tc = conv_of(c.conv, &cv);
  if (!tx || !tw || !ty || !tc) {
    *why = std::string("the convolution's ") + (!tx ? "x " : "") + (!tw ? "w " : "") + (!ty ? "y " : "") +
           (!tc ? "settings " : "") + "are incomplete";
    if (!tc && c.conv)
      *why += " (spatial " + std::to_string(cv.spatial) + ", pads " + std::to_string(cv.pad.size()) + ", strides " +
              std::to_string(cv.stride.size()) + ", dilations " + std::to_string(cv.dilation.size()) + ", type " +
              std::to_string(c.conv->type) + ")";
    return false;
  }
  for (const Tensor* t : {&x, &w, &y})
    if (!supported_type(t->type)) { *why = "tensor data type " + std::to_string(t->type) + " is not supported"; return false; }
  return geometry(x, w, y, cv, &g, why);
}

cudnnStatus_t execute_conv(cudnnHandle_t handle, const Desc* op, const std::map<int64_t, void*>& ptrs) {
  ConvOp c;
  conv_op_of(op, &c);
  Host x, w, y;
  Conv cv;
  tensor_of(c.x, &x.t), tensor_of(c.w, &w.t), tensor_of(c.y, &y.t), conv_of(c.conv, &cv);
  for (Host* h : {&x, &w, &y}) {
    auto it = ptrs.find(h->t.uid);
    if (it == ptrs.end() || !it->second) return refuse("cudnnBackendExecute", "no data pointer for tensor " + std::to_string(h->t.uid));
    h->dev = it->second;
  }
  Geometry g;
  std::string why;
  if (!geometry(x.t, w.t, y.t, cv, &g, &why)) return refuse("cudnnBackendExecute", why);
  // The work the program queued before this call comes first.
  cudaStream_t stream = nullptr;
  cudnnGetStream(handle, &stream);
  cudaStreamSynchronize(stream);
  Host* out = c.dir == Dir::Forward ? &y : c.dir == Dir::Data ? &x : &w;
  Host* in1 = c.dir == Dir::Forward ? &x : &y;
  Host* in2 = c.dir == Dir::Filter ? &x : &w;
  if (!fetch(in1, true) || !fetch(in2, true) || !fetch(out, c.beta != 0.0)) return CUDNN_STATUS_EXECUTION_FAILED;
  std::vector<double> acc(out->v.size(), 0.0);
  convolve(g, c.dir, in1->v, in2->v, acc);
  for (size_t i = 0; i < acc.size(); ++i)
    out->v[i] = static_cast<float>(c.alpha * acc[i] + (c.beta != 0.0 ? c.beta * out->v[i] : 0.0));
  return flush(out) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
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
      Tensor t;
      if (!tensor_of(d, &t) && d->i64s(CUDNN_ATTR_TENSOR_DIMENSIONS).size() != d->i64s(CUDNN_ATTR_TENSOR_STRIDES).size())
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
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR: {
      ConvOp c;
      if (!conv_op_of(d, &c)) return CUDNN_STATUS_BAD_PARAM;
      break;
    }
    case CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR:
    case CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
    case CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR:
    case CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR:
      break;
    default:
      // Pointwise, matmul, norms, reductions and the rest: the graph they
      // would join is refused anyway, but say which one arrived first.
      return refuse("cudnnBackendFinalize", "descriptor type " + std::to_string(d->type) + " is not implemented");
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
  if (!runnable(graph, &why)) return refuse("cudnnBackendExecute", why);
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
  return execute_conv(handle, graph->desc(CUDNN_ATTR_OPERATIONGRAPH_OPS, 0), ptrs);
}
