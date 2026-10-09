// libvgpucudnn's shared pieces (cudnn_common.hpp): element conversion, tensors
// on the host, and the convolution every API path runs.
#include "cudnn_common.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <set>

#include "vgpu/runtime/capture.hpp"

namespace vgpu_cudnn {

/* ---- registry and errors ---- */

namespace {
std::mutex g_mu;
struct Entry {
  CloneFn clone;
  void (*del)(void*);
};
std::map<const void*, Entry> g_live;
thread_local std::string t_last_error;
thread_local bool t_probe = false;
thread_local bool t_replay = false;
}  // namespace

void* track_raw(void* p, CloneFn clone, void (*del)(void*)) {
  std::lock_guard<std::mutex> l(g_mu);
  g_live[p] = Entry{clone, del};
  return p;
}
bool known(const void* p) {
  std::lock_guard<std::mutex> l(g_mu);
  return p && g_live.count(p);
}
void untrack(const void* p) {
  std::lock_guard<std::mutex> l(g_mu);
  g_live.erase(p);
}

/* ---- stream capture ---- */

Snapshot::~Snapshot() {
  for (auto& [orig, c] : done_) {
    untrack(c.p);
    if (c.del) c.del(c.p);
  }
}

void* Snapshot::of(const void* p) {
  if (!p) return nullptr;
  for (auto& [orig, c] : done_)
    if (orig == p) return c.p;
  Entry e{};
  {
    std::lock_guard<std::mutex> l(g_mu);
    auto it = g_live.find(p);
    if (it == g_live.end()) return const_cast<void*>(p);   // device memory, a function, an index
    e = it->second;
  }
  if (!e.clone || !e.del) return const_cast<void*>(p);   // not copyable: used as it is
  void* c = e.clone(p, *this);
  track_raw(c, e.clone, e.del);
  done_.push_back({p, Copy{c, e.del}});
  return c;
}

const void* Snapshot::keep(const void* p, size_t bytes) {
  if (!p) return nullptr;
  const auto* b = static_cast<const uint8_t*>(p);
  bufs_.push_back(std::make_shared<std::vector<uint8_t>>(b, b + bytes));
  return bufs_.back()->data();
}

bool replaying() { return t_replay; }

// Layout is the first member of tensor and filter descriptors alike.
Host scalar(const void* p, const void* owner) {
  return Host{p, [owner] {
                return known(owner) && reinterpret_cast<const Layout*>(owner)->type == CUDNN_DATA_DOUBLE
                           ? sizeof(double)
                           : sizeof(float);
              }};
}
namespace detail {
bool probe_active() { return t_probe; }
void set_probe(bool on) { t_probe = on; }
void set_replaying(bool on) { t_replay = on; }
bool capturing_stream(cudnnHandle_t h, cudaStream_t* stream) {
  cudaStream_t s = nullptr;
  if (!h || !known(h) || cudnnGetStream(h, &s) != CUDNN_STATUS_SUCCESS || !s) return false;
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(s, &st) != cudaSuccess || st != cudaStreamCaptureStatusActive) return false;
  *stream = s;
  return true;
}
bool record_closure(cudaStream_t stream, std::function<void()> op) { return vgpu_record_host_op_if_capturing(stream, std::move(op)); }
}  // namespace detail

void set_last_error(const std::string& msg) { t_last_error = msg; }
std::string last_error() { return t_last_error; }

cudnnStatus_t fail(cudnnStatus_t s, const char* fn, const std::string& why) {
  set_last_error(std::string(fn) + ": " + why);
  if (s == CUDNN_STATUS_NOT_SUPPORTED ? !quiet() : trace())
    std::fprintf(stderr, "[vgpu] %s: %s\n", fn, why.c_str());
  return s;
}

const char* type_name(cudnnDataType_t t) {
  switch (t) {
    case CUDNN_DATA_FLOAT: return "FLOAT";
    case CUDNN_DATA_DOUBLE: return "DOUBLE";
    case CUDNN_DATA_HALF: return "HALF";
    case CUDNN_DATA_BFLOAT16: return "BFLOAT16";
    case CUDNN_DATA_INT8: return "INT8";
    case CUDNN_DATA_UINT8: return "UINT8";
    case CUDNN_DATA_INT32: return "INT32";
    case CUDNN_DATA_INT64: return "INT64";
    case CUDNN_DATA_BOOLEAN: return "BOOLEAN";
    case CUDNN_DATA_INT8x4: return "INT8x4";
    case CUDNN_DATA_UINT8x4: return "UINT8x4";
    case CUDNN_DATA_INT8x32: return "INT8x32";
    case CUDNN_DATA_FP8_E4M3: return "FP8_E4M3";
    case CUDNN_DATA_FP8_E5M2: return "FP8_E5M2";
    default: return "an unknown type";
  }
}

/* ---- element conversion ---- */

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
uint16_t float_to_half(float f) {
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

namespace {
// To nearest, ties to even, then clamped to [lo, hi]; NaN becomes 0.
template <class I>
I saturate(double v) {
  if (std::isnan(v)) return 0;
  const double r = std::nearbyint(v);
  if (r <= static_cast<double>(std::numeric_limits<I>::min())) return std::numeric_limits<I>::min();
  if (r >= static_cast<double>(std::numeric_limits<I>::max())) return std::numeric_limits<I>::max();
  return static_cast<I>(r);
}
}  // namespace

namespace {
// FP8 by the formats' definitions: E4M3 has no infinity and one NaN, E5M2 is a truncated half. Conversion to FP8
// rounds to nearest even and saturates to the largest finite value (the cvt .satfinite every FP8 store uses), a NaN
// becomes 0x7F (measured on an L4: both formats).
double fp8_decode(uint8_t b, bool e4m3) {
  const int ebits = e4m3 ? 4 : 5, mbits = e4m3 ? 3 : 2, bias = e4m3 ? 7 : 15;
  const int e = (b >> mbits) & ((1 << ebits) - 1), m = b & ((1 << mbits) - 1);
  double v;
  if (e4m3 && e == 15 && m == 7) v = std::numeric_limits<double>::quiet_NaN();
  else if (!e4m3 && e == 31) v = m ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
  else if (e == 0) v = std::ldexp(static_cast<double>(m), 1 - bias - mbits);
  else v = std::ldexp(static_cast<double>(m | (1 << mbits)), e - bias - mbits);
  return (b & 0x80) ? -v : v;
}
uint8_t fp8_encode(double v, bool e4m3) {
  const int mbits = e4m3 ? 3 : 2, bias = e4m3 ? 7 : 15;
  const double maxv = e4m3 ? 448.0 : 57344.0;
  const uint8_t maxcode = e4m3 ? 0x7e : 0x7b;
  if (std::isnan(v)) return 0x7f;
  const uint8_t sign = std::signbit(v) ? 0x80 : 0;
  const double a = std::fabs(v);
  if (a >= maxv) return sign | maxcode;
  int e;
  std::frexp(a, &e);
  int exp = e - 1;
  const int emin = 1 - bias;
  if (exp < emin) exp = emin;
  const double quantum = std::ldexp(1.0, exp - mbits);
  const double r = std::nearbyint(a / quantum) * quantum;
  if (r >= maxv) return sign | maxcode;
  if (r == 0) return sign;
  int re;
  std::frexp(r, &re);
  const int rexp = re - 1;
  if (rexp < emin) return sign | static_cast<uint8_t>(static_cast<int>(r / std::ldexp(1.0, emin - mbits)));
  return sign | static_cast<uint8_t>(((rexp + bias) << mbits) | (static_cast<int>(r / std::ldexp(1.0, rexp - mbits)) - (1 << mbits)));
}
}  // namespace

double decode(cudnnDataType_t t, const uint8_t* p) {
  switch (t) {
    case CUDNN_DATA_FLOAT: { float f; std::memcpy(&f, p, 4); return f; }
    case CUDNN_DATA_DOUBLE: { double d; std::memcpy(&d, p, 8); return d; }
    case CUDNN_DATA_HALF: { uint16_t b; std::memcpy(&b, p, 2); return half_to_float(b); }
    case CUDNN_DATA_BFLOAT16: { uint16_t b; std::memcpy(&b, p, 2); return bf16_to_float(b); }
    case CUDNN_DATA_INT8: return static_cast<int8_t>(*p);
    case CUDNN_DATA_FP8_E4M3: return fp8_decode(*p, true);
    case CUDNN_DATA_FP8_E5M2: return fp8_decode(*p, false);
    case CUDNN_DATA_UINT8: return *p;
    case CUDNN_DATA_BOOLEAN: return *p ? 1.0 : 0.0;
    case CUDNN_DATA_INT32: { int32_t i; std::memcpy(&i, p, 4); return i; }
    case CUDNN_DATA_INT64: { int64_t i; std::memcpy(&i, p, 8); return static_cast<double>(i); }
    default: return 0.0;
  }
}
void encode(cudnnDataType_t t, double v, uint8_t* p) {
  switch (t) {
    case CUDNN_DATA_FLOAT: { const float f = static_cast<float>(v); std::memcpy(p, &f, 4); break; }
    case CUDNN_DATA_DOUBLE: std::memcpy(p, &v, 8); break;
    // Through float first: double -> float -> half can round twice, but every
    // value this library stores as half was computed in float on the
    // hardware too, so the float is the value being converted.
    case CUDNN_DATA_HALF: { const uint16_t b = float_to_half(static_cast<float>(v)); std::memcpy(p, &b, 2); break; }
    case CUDNN_DATA_BFLOAT16: { const uint16_t b = float_to_bf16(static_cast<float>(v)); std::memcpy(p, &b, 2); break; }
    // 8-bit results come out of a float epilogue on the hardware: the value is
    // a float before it is rounded (measured: alpha 0.05f times 50 is 2.5f,
    // which rounds to 2, where the exact product would round to 3).
    case CUDNN_DATA_INT8: { const int8_t i = saturate<int8_t>(static_cast<float>(v)); std::memcpy(p, &i, 1); break; }
    case CUDNN_DATA_UINT8: *p = saturate<uint8_t>(static_cast<float>(v)); break;
    case CUDNN_DATA_FP8_E4M3: *p = fp8_encode(static_cast<float>(v), true); break;
    case CUDNN_DATA_FP8_E5M2: *p = fp8_encode(static_cast<float>(v), false); break;
    case CUDNN_DATA_BOOLEAN: *p = v != 0.0; break;
    case CUDNN_DATA_INT32: { const int32_t i = saturate<int32_t>(v); std::memcpy(p, &i, 4); break; }
    case CUDNN_DATA_INT64: { const int64_t i = saturate<int64_t>(v); std::memcpy(p, &i, 8); break; }
    default: break;
  }
}
double round_to(cudnnDataType_t t, double v) {
  uint8_t b[8];
  encode(t, v, b);
  return decode(t, b);
}

/* ---- tensors ---- */

void packed_strides(int rank, const int64_t* dims, bool channels_last, int64_t* strides) {
  if (!channels_last || rank < 3) {
    int64_t s = 1;
    for (int i = rank; i-- > 0;) strides[i] = s, s *= dims[i];
    return;
  }
  // N, then spatial outer to inner, then C innermost.
  int64_t s = 1;
  strides[1] = s;
  s *= dims[1];
  for (int i = rank; i-- > 2;) strides[i] = s, s *= dims[i];
  strides[0] = s;
}

bool read(const Layout& t, const void* dev, std::vector<double>* out) {
  const size_t eb = type_bytes(t.type), n = t.count();
  out->assign(n, 0.0);
  if (!n) return true;
  if (!dev) return false;
  std::vector<uint8_t> raw(t.span() * eb);
  if (cudaMemcpy(raw.data(), dev, raw.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
  double* v = out->data();
  each(t, [&](size_t i, size_t off) { v[i] = decode(t.type, raw.data() + off * eb); });
  return true;
}

bool write(const Layout& t, void* dev, const std::vector<double>& v) {
  const size_t eb = type_bytes(t.type), n = t.count();
  if (!n) return true;
  if (!dev || v.size() < n) return false;
  std::vector<uint8_t> raw(t.span() * eb);
  // A strided tensor's gaps belong to someone else: keep them.
  if (raw.size() != n * eb && cudaMemcpy(raw.data(), dev, raw.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
    return false;
  each(t, [&](size_t i, size_t off) { encode(t.type, v[i], raw.data() + off * eb); });
  return cudaMemcpy(dev, raw.data(), raw.size(), cudaMemcpyHostToDevice) == cudaSuccess;
}

bool blend_write(const Layout& t, void* dev, const std::vector<double>& r, double alpha, double beta) {
  std::vector<double> out(t.count(), 0.0);
  if (beta != 0.0 && !read(t, dev, &out)) return false;
  for (size_t i = 0; i < out.size() && i < r.size(); ++i)
    out[i] = beta != 0.0 ? alpha * r[i] + beta * out[i] : alpha * r[i];
  return write(t, dev, out);
}

/* ---- convolution ---- */

bool conv_geometry(const Layout& x, const Layout& w, const Layout& y, int nsp, const int64_t* pad,
                   const int64_t* str, const int64_t* dil, bool flip, ConvGeom* g, std::string* why,
                   const int64_t* post) {
  if (nsp < 1 || nsp > 3) { *why = "only 1 to 3 spatial dimensions are supported"; return false; }
  if (x.rank != nsp + 2 || w.rank != nsp + 2 || y.rank != nsp + 2) {
    *why = "tensor ranks (" + std::to_string(x.rank) + ", " + std::to_string(w.rank) + ", " + std::to_string(y.rank) +
           ") do not match the convolution's " + std::to_string(nsp) + " spatial dimensions";
    return false;
  }
  g->N = x.dims[0], g->C = x.dims[1], g->K = w.dims[0], g->Cg = w.dims[1];
  if (g->Cg <= 0 || g->C % g->Cg) { *why = "input channels are not a multiple of the filter's"; return false; }
  g->G = g->C / g->Cg;
  if (g->K % g->G) { *why = "output channels are not a multiple of the group count"; return false; }
  g->Kg = g->K / g->G;
  if (y.dims[0] != g->N || y.dims[1] != g->K) { *why = "the output's batch or channels do not match"; return false; }
  for (int i = 0; i < nsp; ++i) {
    const int k = 3 - nsp + i;
    g->in[k] = x.dims[2 + i], g->out[k] = y.dims[2 + i], g->flt[k] = w.dims[2 + i];
    g->pad[k] = pad[i], g->str[k] = str[i], g->dil[k] = dil[i];
    const int64_t after = post ? post[i] : pad[i];
    if (str[i] < 1 || dil[i] < 1 || pad[i] < 0 || after < 0) { *why = "a stride, dilation or padding is out of range"; return false; }
    const int64_t want = conv_out(g->in[k], g->pad[k], after, g->flt[k], g->str[k], g->dil[k]);
    if (g->out[k] != want) {
      *why = "output extent " + std::to_string(g->out[k]) + " in spatial dimension " + std::to_string(i) +
             " is not the " + std::to_string(want) + " the convolution gives";
      return false;
    }
  }
  g->flip = flip;
  return true;
}

namespace {
inline double half_fma(double acc, double a, double b) {
  // a and b are halves, so their product is exact in float; one rounding.
  return half_to_float(float_to_half(static_cast<float>(acc) + static_cast<float>(a) * static_cast<float>(b)));
}
}  // namespace

void convolve(const ConvGeom& g, ConvDir dir, const std::vector<double>& a, const std::vector<double>& b,
              std::vector<double>* out, Accum acc) {
  const int64_t I0 = g.in[0], I1 = g.in[1], I2 = g.in[2];
  const int64_t O0 = g.out[0], O1 = g.out[1], O2 = g.out[2];
  const int64_t F0 = g.flt[0], F1 = g.flt[1], F2 = g.flt[2];
  const int64_t isz = I0 * I1 * I2, osz = O0 * O1 * O2, fsz = F0 * F1 * F2;
  // The input position tap f of output position o reads, per dimension.
  auto in_pos = [&](int d, int64_t o, int64_t f) {
    const int64_t tap = g.flip ? g.flt[d] - 1 - f : f;
    return o * g.str[d] - g.pad[d] + tap * g.dil[d];
  };
  auto add = [&](double s, double x, double y) {
    switch (acc) {
      case Accum::Half: return half_fma(s, x, y);
      default: return s + x * y;
    }
  };
  switch (dir) {
    case ConvDir::Forward: {  // a = x [N,C,I], b = w [K,Cg,F] -> y [N,K,O]
      out->assign(static_cast<size_t>(g.N * g.K * osz), 0.0);
      for (int64_t n = 0; n < g.N; ++n)
        for (int64_t k = 0; k < g.K; ++k) {
          const int64_t c0 = (k / g.Kg) * g.Cg;
          for (int64_t o0 = 0; o0 < O0; ++o0)
            for (int64_t o1 = 0; o1 < O1; ++o1)
              for (int64_t o2 = 0; o2 < O2; ++o2) {
                double s = 0.0;
                for (int64_t ci = 0; ci < g.Cg; ++ci) {
                  const double* xb = a.data() + ((n * g.C + c0 + ci) * isz);
                  const double* wb = b.data() + ((k * g.Cg + ci) * fsz);
                  for (int64_t f0 = 0; f0 < F0; ++f0) {
                    const int64_t p0 = in_pos(0, o0, f0);
                    if (p0 < 0 || p0 >= I0) continue;
                    for (int64_t f1 = 0; f1 < F1; ++f1) {
                      const int64_t p1 = in_pos(1, o1, f1);
                      if (p1 < 0 || p1 >= I1) continue;
                      for (int64_t f2 = 0; f2 < F2; ++f2) {
                        const int64_t p2 = in_pos(2, o2, f2);
                        if (p2 < 0 || p2 >= I2) continue;
                        s = add(s, xb[(p0 * I1 + p1) * I2 + p2], wb[(f0 * F1 + f1) * F2 + f2]);
                      }
                    }
                  }
                }
                (*out)[static_cast<size_t>((n * g.K + k) * osz + (o0 * O1 + o1) * O2 + o2)] = s;
              }
        }
      break;
    }
    case ConvDir::Data: {  // a = dy [N,K,O], b = w [K,Cg,F] -> dx [N,C,I]
      out->assign(static_cast<size_t>(g.N * g.C * isz), 0.0);
      // The output position o that tap f of position p came from, if any.
      auto out_pos = [&](int d, int64_t p, int64_t f, int64_t* o) {
        const int64_t tap = g.flip ? g.flt[d] - 1 - f : f;
        const int64_t num = p + g.pad[d] - tap * g.dil[d];
        if (num < 0 || num % g.str[d]) return false;
        *o = num / g.str[d];
        return *o < g.out[d];
      };
      for (int64_t n = 0; n < g.N; ++n)
        for (int64_t c = 0; c < g.C; ++c) {
          const int64_t grp = c / g.Cg, ci = c % g.Cg;
          for (int64_t p0 = 0; p0 < I0; ++p0)
            for (int64_t p1 = 0; p1 < I1; ++p1)
              for (int64_t p2 = 0; p2 < I2; ++p2) {
                double s = 0.0;
                for (int64_t kk = 0; kk < g.Kg; ++kk) {
                  const int64_t k = grp * g.Kg + kk;
                  const double* yb = a.data() + ((n * g.K + k) * osz);
                  const double* wb = b.data() + ((k * g.Cg + ci) * fsz);
                  int64_t o0, o1, o2;
                  for (int64_t f0 = 0; f0 < F0; ++f0) {
                    if (!out_pos(0, p0, f0, &o0)) continue;
                    for (int64_t f1 = 0; f1 < F1; ++f1) {
                      if (!out_pos(1, p1, f1, &o1)) continue;
                      for (int64_t f2 = 0; f2 < F2; ++f2) {
                        if (!out_pos(2, p2, f2, &o2)) continue;
                        s = add(s, yb[(o0 * O1 + o1) * O2 + o2], wb[(f0 * F1 + f1) * F2 + f2]);
                      }
                    }
                  }
                }
                (*out)[static_cast<size_t>((n * g.C + c) * isz + (p0 * I1 + p1) * I2 + p2)] = s;
              }
        }
      break;
    }
    case ConvDir::Filter: {  // a = dy [N,K,O], b = x [N,C,I] -> dw [K,Cg,F]
      out->assign(static_cast<size_t>(g.K * g.Cg * fsz), 0.0);
      for (int64_t k = 0; k < g.K; ++k) {
        const int64_t c0 = (k / g.Kg) * g.Cg;
        for (int64_t ci = 0; ci < g.Cg; ++ci)
          for (int64_t f0 = 0; f0 < F0; ++f0)
            for (int64_t f1 = 0; f1 < F1; ++f1)
              for (int64_t f2 = 0; f2 < F2; ++f2) {
                double s = 0.0;
                for (int64_t n = 0; n < g.N; ++n) {
                  const double* yb = a.data() + ((n * g.K + k) * osz);
                  const double* xb = b.data() + ((n * g.C + c0 + ci) * isz);
                  for (int64_t o0 = 0; o0 < O0; ++o0) {
                    const int64_t p0 = in_pos(0, o0, f0);
                    if (p0 < 0 || p0 >= I0) continue;
                    for (int64_t o1 = 0; o1 < O1; ++o1) {
                      const int64_t p1 = in_pos(1, o1, f1);
                      if (p1 < 0 || p1 >= I1) continue;
                      for (int64_t o2 = 0; o2 < O2; ++o2) {
                        const int64_t p2 = in_pos(2, o2, f2);
                        if (p2 < 0 || p2 >= I2) continue;
                        s = add(s, yb[(o0 * O1 + o1) * O2 + o2], xb[(p0 * I1 + p1) * I2 + p2]);
                      }
                    }
                  }
                }
                (*out)[static_cast<size_t>((k * g.Cg + ci) * fsz + (f0 * F1 + f1) * F2 + f2)] = s;
              }
      }
      break;
    }
  }
}

// ---- dropout's generators: cuRAND's XORWOW, one state per thread ----
//
// XORWOW is Marsaglia's xorshift generator with a Weyl sequence added. Each
// state is a Weyl counter d and five words v0..v4; a step is
//   t = v0 ^ (v0 >> 2); v0..v3 = v1..v4; v4 = (v4 ^ (v4 << 4)) ^ (t ^ (t << 1));
//   d += 362437; output = d + v4.
// The seeding below, and cuDNN's use of it, were measured on an RTX 3060 and
// match its states buffer word for word: thread t starts at the seed's state
// moved forward by t * 2^67 steps (the v words are a linear recurrence over
// GF(2), so the move is a matrix power; d does not change, since 2^67 times
// the Weyl increment is 0 mod 2^32).

namespace {

constexpr int kStateWords = 12;  // 48 bytes: d, v0..v4, then cuRAND's Box-Muller fields; word 9 is padding

// The recurrence as a 160 x 160 matrix over GF(2), by columns: column k is the
// image of bit k (word k / 32, bit k % 32).
struct Gf2 {
  uint32_t col[160][5];
};

void xorwow_v_step(uint32_t v[5]) {
  const uint32_t t = v[0] ^ (v[0] >> 2);
  const uint32_t n4 = (v[4] ^ (v[4] << 4)) ^ (t ^ (t << 1));
  v[0] = v[1], v[1] = v[2], v[2] = v[3], v[3] = v[4], v[4] = n4;
}

void apply(const Gf2& m, const uint32_t in[5], uint32_t out[5]) {
  uint32_t r[5] = {0, 0, 0, 0, 0};
  for (int w = 0; w < 5; ++w)
    for (uint32_t bits = in[w]; bits; bits &= bits - 1) {
      const int k = w * 32 + __builtin_ctz(bits);
      for (int j = 0; j < 5; ++j) r[j] ^= m.col[k][j];
    }
  for (int j = 0; j < 5; ++j) out[j] = r[j];
}

// 2^67 steps: the recurrence's matrix squared 67 times.
const Gf2& subsequence_jump() {
  static const Gf2 jump = [] {
    Gf2 m;
    for (int k = 0; k < 160; ++k) {
      uint32_t v[5] = {0, 0, 0, 0, 0};
      v[k / 32] = 1u << (k % 32);
      xorwow_v_step(v);
      for (int j = 0; j < 5; ++j) m.col[k][j] = v[j];
    }
    for (int i = 0; i < 67; ++i) {
      Gf2 sq;
      for (int k = 0; k < 160; ++k) apply(m, m.col[k], sq.col[k]);
      m = sq;
    }
    return m;
  }();
  return jump;
}

// curand_init(seed, 0, 0)'s state.
void xorwow_first_state(unsigned long long seed, uint32_t* d, uint32_t v[5]) {
  const uint32_t s0 = static_cast<uint32_t>(seed) ^ 0xaad26b49u, s1 = static_cast<uint32_t>(seed >> 32) ^ 0xf7dcefddu;
  const uint32_t t0 = 1099087573u * s0, t1 = 2591861531u * s1;
  *d = 6615241u + t1 + t0;
  v[0] = 123456789u + t0, v[1] = 362436069u ^ t0, v[2] = 521288629u + t1, v[3] = 88675123u ^ t1, v[4] = 5783321u + t0;
}

// curand_uniform: (0, 1], in float, the multiply and the add each rounded.
inline float curand_uniform_of(uint32_t x) {
  volatile float scaled = static_cast<float>(x) * 2.3283064365386963e-10f;
  return scaled + 1.1641532182693481e-10f;
}

}  // namespace

size_t dropout_threads() {
  int dev = 0, sms = 28;
  if (cudaGetDevice(&dev) == cudaSuccess) {
    int n = 0;
    if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && n > 0) sms = n;
  }
  return static_cast<size_t>(sms) * 768;
}

bool dropout_seed(DropoutDesc* d) {
  const size_t T = d->threads;
  std::vector<uint32_t> words(T * kStateWords, 0);
  // Keep whatever sits in a buffer already (the padding word).
  if (d->states && cudaMemcpy(words.data(), d->states, T * 48, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
  uint32_t dw, v[5];
  xorwow_first_state(d->seed, &dw, v);
  const Gf2& jump = subsequence_jump();
  for (size_t t = 0; t < T; ++t) {
    uint32_t* w = &words[t * kStateWords];
    w[0] = dw;
    for (int j = 0; j < 5; ++j) w[1 + j] = v[j];
    w[6] = w[7] = w[8] = w[10] = w[11] = 0;
    uint32_t next[5];
    apply(jump, v, next);
    for (int j = 0; j < 5; ++j) v[j] = next[j];
  }
  if (!d->states) {
    d->local = std::move(words);
    return true;
  }
  return cudaMemcpy(d->states, words.data(), T * 48, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool dropout_draw(DropoutDesc* d, size_t n, std::vector<uint8_t>* keep) {
  const size_t T = d->threads ? d->threads : dropout_threads();
  std::vector<uint32_t> words;
  if (d->states) {
    if (d->state_bytes < T * 48) return false;  // a buffer too small for these generators (set on another device)
    words.resize(T * kStateWords);
    if (cudaMemcpy(words.data(), d->states, T * 48, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
  } else {
    if (d->local.size() != T * kStateWords) {
      d->threads = T;
      if (!dropout_seed(d)) return false;
    }
    words = d->local;
  }
  keep->resize(n);
  const float p = d->p;
  for (size_t i = 0; i < n; ++i) {
    uint32_t* w = &words[(i % T) * kStateWords];
    xorwow_v_step(w + 1);
    w[0] += 362437u;
    (*keep)[i] = curand_uniform_of(w[0] + w[5]) > p;
  }
  if (!d->states) {
    d->local = std::move(words);
    return true;
  }
  return cudaMemcpy(d->states, words.data(), T * 48, cudaMemcpyHostToDevice) == cudaSuccess;
}

void sync_handle(cudnnHandle_t h) {
  if (t_probe) throw ProbeCommit{};   // see defer_call
  if (t_replay) return;               // a graph runs its nodes in order
  cudaStream_t s = nullptr;
  if (cudnnGetStream(h, &s) == CUDNN_STATUS_SUCCESS) cudaStreamSynchronize(s);
}

}  // namespace vgpu_cudnn
