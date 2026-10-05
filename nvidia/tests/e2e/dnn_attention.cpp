// cuDNN's scaled dot-product attention as cudnn-frontend builds it -- the
// graph PyTorch's and Transformer Engine's cuDNN attention hand the library:
//
//   - forward: O = softmax(scale * Q K^T + bias, masked) V, with the
//     log-sum-exp statistics a training pass keeps, a causal (top-left or
//     bottom-right) or sliding-window mask, per-batch padding (sequence
//     lengths, or packed variable-length sequences through ragged offsets),
//     grouped-query heads, paged K/V caches, and Philox dropout;
//   - backward: dQ, dK, dV (and dBias) from O, dO and the statistics;
//
// both as the frontend's UNIFIED node (one SDPA operation, cuDNN 9.13+) and
// as its COMPOSITE node (matmul, pointwise, softmax, diagonal-band mask, RNG
// and reshape operations joined by virtual tensors), in half, bfloat16 and,
// for the composite form, float. Each result is checked against a reference
// computed here in double from the same rounded inputs; dropout's mask is the
// one the library reports through the RNG dump, checked to drop about the
// requested fraction, and the backward pass regenerates it from the seed.
//
// The same program runs against NVIDIA's libcudnn.so.9: where its heuristics
// offer no engine for a configuration on that GPU, the check says so and is
// skipped (the library declining is not a wrong answer); on VirtualGPU every
// configuration must run.
// The frontend resolves cuDNN, the runtime and NVRTC at run time, as PyTorch
// builds it, rather than linking the CUDA 12+ runtime entry points its
// optional OSS engines name.
#define NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#include <cudnn_frontend.h>
#include <cuda_runtime.h>

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;
namespace cudnn_frontend {
void* cudnn_dlhandle = dlopen("libcudnn.so.9", RTLD_NOW | RTLD_GLOBAL);
}

static int fails = 0;
static void expect(const std::string& what, bool ok, double detail = 0) {
  std::printf("%s %s", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok && detail != 0) std::printf(" (%g)", detail);
  std::printf("\n");
  fails += !ok;
}
static const bool kOnSim = std::getenv("VGPU_GPU") != nullptr;

// ---- element types -----------------------------------------------------------

static uint16_t f2h(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t e = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = x & 0x7fffffu;
  if (((x >> 23) & 0xff) == 0xff) return static_cast<uint16_t>(sign | 0x7c00u | (m ? 0x200u : 0));
  if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if (e <= 0) {
    if (e < -10) return static_cast<uint16_t>(sign);
    m |= 0x800000u;
    const int shift = 14 - e;
    uint32_t r = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (r & 1))) ++r;
    return static_cast<uint16_t>(sign | r);
  }
  uint32_t r = (static_cast<uint32_t>(e) << 10) | (m >> 13);
  const uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
  return static_cast<uint16_t>(sign | r);
}
static float h2f(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  int32_t e = (h >> 10) & 0x1f;
  uint32_t m = h & 0x3ffu, x;
  if (e == 0) {
    if (!m) x = sign;
    else {
      e = 1;
      while (!(m & 0x400u)) m <<= 1, --e;
      m &= 0x3ffu;
      x = sign | (static_cast<uint32_t>(e + 127 - 15) << 23) | (m << 13);
    }
  } else if (e == 31) {
    x = sign | 0x7f800000u | (m << 13);
  } else {
    x = sign | (static_cast<uint32_t>(e + 127 - 15) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}
static uint16_t f2bf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  x += 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>(x >> 16);
}
static float bf2f(uint16_t b) {
  const uint32_t x = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

struct Dev {
  void* p = nullptr;
  size_t bytes = 0;
  explicit Dev(size_t n) : bytes(n ? n : 1) { cudaMalloc(&p, bytes); cudaMemset(p, 0, bytes); }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
};

// A tensor's values in its element type: stored to the device, and read
// back, as doubles in logical order (the strides are packed BHSD here, or
// what the caller says).
struct Buf {
  fe::DataType_t t;
  std::vector<double> v;  // logical values, already rounded to t
  std::unique_ptr<Dev> d;
  size_t esize() const { return t == fe::DataType_t::FLOAT || t == fe::DataType_t::INT32 ? 4 : t == fe::DataType_t::INT64 ? 8 : 2; }
  // phys[i]: the element offset of logical element i.
  std::vector<int64_t> phys;
};
static double round_to(fe::DataType_t t, double x) {
  if (t == fe::DataType_t::HALF) return h2f(f2h(static_cast<float>(x)));
  if (t == fe::DataType_t::BFLOAT16) return bf2f(f2bf(static_cast<float>(x)));
  if (t == fe::DataType_t::FLOAT) return static_cast<float>(x);
  return x;
}
static std::vector<int64_t> physical(const std::vector<int64_t>& dim, const std::vector<int64_t>& str) {
  size_t n = 1;
  for (int64_t x : dim) n *= static_cast<size_t>(x);
  std::vector<int64_t> p(n);
  std::vector<int64_t> at(dim.size(), 0);
  for (size_t i = 0; i < n; ++i) {
    int64_t o = 0;
    for (size_t k = 0; k < dim.size(); ++k) o += at[k] * str[k];
    p[i] = o;
    for (size_t k = dim.size(); k-- > 0;) {
      if (++at[k] < dim[k]) break;
      at[k] = 0;
    }
  }
  return p;
}
// phys, when given, says where each logical element lives (-1: nowhere, a
// ragged tensor's padding).
static std::unique_ptr<Buf> make_buf(fe::DataType_t t, const std::vector<int64_t>& dim, const std::vector<int64_t>& str,
                                     std::vector<double> vals, const std::vector<int64_t>* phys = nullptr) {
  auto b = std::make_unique<Buf>();
  b->t = t;
  b->phys = phys ? *phys : physical(dim, str);
  int64_t span = 0;
  for (int64_t o : b->phys) span = std::max(span, o + 1);
  b->d = std::make_unique<Dev>(static_cast<size_t>(span) * b->esize());
  b->v.resize(b->phys.size());
  std::vector<uint8_t> host(static_cast<size_t>(span) * b->esize(), 0);
  for (size_t i = 0; i < b->phys.size(); ++i) {
    const double x = i < vals.size() ? vals[i] : 0.0;
    b->v[i] = round_to(t, x);
    if (b->phys[i] < 0) continue;
    uint8_t* at = host.data() + b->phys[i] * b->esize();
    if (t == fe::DataType_t::HALF) { uint16_t h = f2h(static_cast<float>(x)); std::memcpy(at, &h, 2); }
    else if (t == fe::DataType_t::BFLOAT16) { uint16_t h = f2bf(static_cast<float>(x)); std::memcpy(at, &h, 2); }
    else if (t == fe::DataType_t::FLOAT) { float f = static_cast<float>(x); std::memcpy(at, &f, 4); }
    else if (t == fe::DataType_t::INT32) { int32_t k = static_cast<int32_t>(x); std::memcpy(at, &k, 4); }
    else { int64_t k = static_cast<int64_t>(x); std::memcpy(at, &k, 8); }
  }
  cudaMemcpy(b->d->p, host.data(), host.size(), cudaMemcpyHostToDevice);
  return b;
}
static std::vector<double> read_buf(const Buf& b) {
  std::vector<uint8_t> host(b.d->bytes);
  cudaMemcpy(host.data(), b.d->p, host.size(), cudaMemcpyDeviceToHost);
  std::vector<double> out(b.phys.size());
  for (size_t i = 0; i < out.size(); ++i) {
    if (b.phys[i] < 0) continue;
    const uint8_t* at = host.data() + b.phys[i] * b.esize();
    if (b.t == fe::DataType_t::HALF) { uint16_t h; std::memcpy(&h, at, 2); out[i] = h2f(h); }
    else if (b.t == fe::DataType_t::BFLOAT16) { uint16_t h; std::memcpy(&h, at, 2); out[i] = bf2f(h); }
    else if (b.t == fe::DataType_t::FLOAT) { float f; std::memcpy(&f, at, 4); out[i] = f; }
    else if (b.t == fe::DataType_t::INT32) { int32_t k; std::memcpy(&k, at, 4); out[i] = k; }
    else { int64_t k; std::memcpy(&k, at, 8); out[i] = static_cast<double>(k); }
  }
  return out;
}
static std::vector<double> randoms(size_t n, unsigned seed, double lo = -1.0, double hi = 1.0) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<double> u(lo, hi);
  std::vector<double> v(n);
  for (double& x : v) x = u(g);
  return v;
}
// Largest |a - b| / (1 + |b|).
static double err_of(const std::vector<double>& a, const std::vector<double>& b) {
  double e = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]) / (1.0 + std::fabs(b[i]));
    if (!(d <= e)) e = std::isnan(d) ? INFINITY : d;
  }
  return a.size() == b.size() ? e : INFINITY;
}

// ---- the reference --------------------------------------------------------------

struct Cfg {
  std::string name;
  fe::DataType_t io = fe::DataType_t::HALF;
  fe::AttentionImplementation_t impl = fe::AttentionImplementation_t::AUTO;
  int64_t b = 2, hq = 2, hk = 2, sq = 16, skv = 16, d = 16, dv = 16;
  float scale = 0.25f;
  bool causal = false, bottom_right = false;
  int left = -1;           // sliding window: keep column > row - left (top-left), when >= 0
  bool padding = false;    // per-batch sequence lengths
  bool bias = false;       // an additive [1, hq, sq, skv] bias
  float dropout = 0.0f;
  bool backward = true;
  bool bhsd_interleaved = false;  // Q/K/V/O as [b, s, h, d] in memory
  bool ragged = false;            // packed sequences (THD), with ragged offsets; implies padding
  int64_t page = 0;               // paged K/V caches of this block size (forward only; implies padding)
};

// A ragged (THD) tensor: batch b's tokens start at offset[b] = (tokens
// before it) * H * D elements, each token's heads then dims; positions past
// a batch's length are not stored.
static std::vector<int64_t> ragged_phys(int64_t B, int64_t H, int64_t S, int64_t D, const std::vector<int>& len,
                                        std::vector<double>* offsets) {
  std::vector<int64_t> p(static_cast<size_t>(B * H * S * D), -1);
  offsets->assign(static_cast<size_t>(B + 1), 0.0);
  int64_t tok = 0;
  for (int64_t b = 0; b < B; ++b) {
    (*offsets)[b] = static_cast<double>(tok * H * D);
    for (int64_t h = 0; h < H; ++h)
      for (int64_t s = 0; s < len[b]; ++s)
        for (int64_t d = 0; d < D; ++d) p[((b * H + h) * S + s) * D + d] = (tok + s) * H * D + h * D + d;
    tok += len[b];
  }
  (*offsets)[B] = static_cast<double>(tok * H * D);
  return p;
}

// Scores' mask for one (batch, row, col): whether the element is kept.
static bool kept(const Cfg& c, int64_t row, int64_t col, int64_t sq_b, int64_t skv_b) {
  if (c.padding && (row >= sq_b || col >= skv_b)) return false;
  const int64_t shift = c.bottom_right ? skv_b - sq_b : 0;
  if (c.causal && col > row + shift) return false;
  if (c.left >= 0 && !(col > row + shift - c.left)) return false;
  return true;
}

struct Ref {
  std::vector<double> O, stats, P;  // P: the softmax (before dropout), [b, hq, sq, skv]
};
static Ref reference_fwd(const Cfg& c, const std::vector<double>& Q, const std::vector<double>& K,
                         const std::vector<double>& V, const std::vector<double>* bias,
                         const std::vector<int>& seq_q, const std::vector<int>& seq_kv,
                         const std::vector<double>* mask) {
  Ref r;
  r.O.assign(static_cast<size_t>(c.b * c.hq * c.sq * c.dv), 0.0);
  r.stats.assign(static_cast<size_t>(c.b * c.hq * c.sq), 0.0);
  r.P.assign(static_cast<size_t>(c.b * c.hq * c.sq * c.skv), 0.0);
  const double keep_scale = c.dropout > 0 ? 1.0 / (1.0 - c.dropout) : 1.0;
  for (int64_t bb = 0; bb < c.b; ++bb)
    for (int64_t h = 0; h < c.hq; ++h) {
      const int64_t hk = h / (c.hq / c.hk), hv = h / (c.hq / c.hk);
      const int64_t sqb = c.padding ? seq_q[bb] : c.sq, skvb = c.padding ? seq_kv[bb] : c.skv;
      for (int64_t i = 0; i < c.sq; ++i) {
        std::vector<double> s(static_cast<size_t>(c.skv));
        double mx = -INFINITY;
        for (int64_t j = 0; j < c.skv; ++j) {
          double acc = 0;
          for (int64_t k = 0; k < c.d; ++k)
            acc += Q[((bb * c.hq + h) * c.sq + i) * c.d + k] * K[((bb * c.hk + hk) * c.skv + j) * c.d + k];
          acc *= c.scale;
          if (bias) acc += (*bias)[(h * c.sq + i) * c.skv + j];
          s[j] = kept(c, i, j, sqb, skvb) ? acc : -INFINITY;
          mx = std::max(mx, s[j]);
        }
        if (c.padding && i >= sqb) continue;  // padded rows: not compared
        double sum = 0;
        for (int64_t j = 0; j < c.skv; ++j) sum += s[j] == -INFINITY ? 0.0 : std::exp(s[j] - mx);
        r.stats[(bb * c.hq + h) * c.sq + i] = mx + std::log(sum);
        for (int64_t j = 0; j < c.skv; ++j) {
          const double p = s[j] == -INFINITY ? 0.0 : std::exp(s[j] - mx) / sum;
          const size_t pi = static_cast<size_t>(((bb * c.hq + h) * c.sq + i) * c.skv + j);
          r.P[pi] = p;
          const double pd = mask ? p * (*mask)[pi] * keep_scale : p;
          for (int64_t e = 0; e < c.dv; ++e)
            r.O[((bb * c.hq + h) * c.sq + i) * c.dv + e] += pd * V[((bb * c.hk + hv) * c.skv + j) * c.dv + e];
        }
      }
    }
  return r;
}

struct Grads {
  std::vector<double> dQ, dK, dV, dBias;
};
static Grads reference_bwd(const Cfg& c, const std::vector<double>& Q, const std::vector<double>& K,
                           const std::vector<double>& V, const Ref& f, const std::vector<double>& dO,
                           const std::vector<int>& seq_q, const std::vector<double>* mask) {
  Grads g;
  g.dQ.assign(Q.size(), 0.0);
  g.dK.assign(K.size(), 0.0);
  g.dV.assign(V.size(), 0.0);
  g.dBias.assign(static_cast<size_t>(c.hq * c.sq * c.skv), 0.0);
  const double ks = c.dropout > 0 ? 1.0 / (1.0 - c.dropout) : 1.0;
  for (int64_t bb = 0; bb < c.b; ++bb)
    for (int64_t h = 0; h < c.hq; ++h) {
      const int64_t hk = h / (c.hq / c.hk);
      const int64_t sqb = c.padding ? seq_q[bb] : c.sq;
      for (int64_t i = 0; i < sqb; ++i) {
        // D = rowsum(dO * O)
        double D = 0;
        for (int64_t e = 0; e < c.dv; ++e)
          D += dO[((bb * c.hq + h) * c.sq + i) * c.dv + e] * f.O[((bb * c.hq + h) * c.sq + i) * c.dv + e];
        for (int64_t j = 0; j < c.skv; ++j) {
          const size_t pi = static_cast<size_t>(((bb * c.hq + h) * c.sq + i) * c.skv + j);
          const double p = f.P[pi], m = mask ? (*mask)[pi] * ks : 1.0;
          double dp = 0;
          for (int64_t e = 0; e < c.dv; ++e) {
            const double dov = dO[((bb * c.hq + h) * c.sq + i) * c.dv + e];
            dp += dov * V[((bb * c.hk + hk) * c.skv + j) * c.dv + e];
            g.dV[((bb * c.hk + hk) * c.skv + j) * c.dv + e] += p * m * dov;
          }
          const double ds = p * (dp * m - D);
          g.dBias[(h * c.sq + i) * c.skv + j] += ds;
          for (int64_t k = 0; k < c.d; ++k) {
            g.dQ[((bb * c.hq + h) * c.sq + i) * c.d + k] += c.scale * ds * K[((bb * c.hk + hk) * c.skv + j) * c.d + k];
            g.dK[((bb * c.hk + hk) * c.skv + j) * c.d + k] += c.scale * ds * Q[((bb * c.hq + h) * c.sq + i) * c.d + k];
          }
        }
      }
    }
  return g;
}

// ---- running a graph -------------------------------------------------------------

using Var = std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*>;

// 1 ran, 0 no engine offered, -1 an error.
static int build_and_run(const char* what, fe::graph::Graph& g, cudnnHandle_t h, Var& pack) {
  auto st = g.validate();
  if (st.is_good()) st = g.build_operation_graph(h);
  if (st.is_good()) st = g.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
  if (st.is_good()) st = g.check_support();
  if (st.is_good()) st = g.build_plans();
  if (!st.is_good()) {
    std::printf("     %s: %s\n", what, st.get_message().c_str());
    return 0;
  }
  int64_t ws = 0;
  if (!g.get_workspace_size(ws).is_good()) return -1;
  Dev work(static_cast<size_t>(ws));
  st = g.execute(h, pack, work.p);
  cudaDeviceSynchronize();
  if (!st.is_good()) {
    std::printf("     %s: execute: %s\n", what, st.get_message().c_str());
    return -1;
  }
  return 1;
}

static std::vector<int64_t> strides_of(const Cfg& c, int64_t h, int64_t s, int64_t d) {
  if (c.ragged) return {h * d, d, h * d, 1};                 // THD: the batch stride is the ragged offset's
  if (c.bhsd_interleaved) return {s * h * d, d, h * d, 1};  // memory [b, s, h, d]
  return {h * s * d, s * d, d, 1};
}

static void run(const Cfg& c, cudnnHandle_t handle) {
  const fe::DataType_t T = c.io;
  // Tolerances: the inputs are rounded alike, so what differs is the output
  // rounding and the order of accumulation.
  const double tol = T == fe::DataType_t::FLOAT ? 2e-5 : T == fe::DataType_t::HALF ? 4e-3 : 3e-2;
  const auto Qv = randoms(static_cast<size_t>(c.b * c.hq * c.sq * c.d), 1);
  const auto Kv = randoms(static_cast<size_t>(c.b * c.hk * c.skv * c.d), 2);
  const auto Vv = randoms(static_cast<size_t>(c.b * c.hk * c.skv * c.dv), 3);
  const auto dOv = randoms(static_cast<size_t>(c.b * c.hq * c.sq * c.dv), 4);
  const auto Bv = randoms(static_cast<size_t>(c.hq * c.sq * c.skv), 5, -0.5, 0.5);
  std::vector<int> seq_q(static_cast<size_t>(c.b)), seq_kv(static_cast<size_t>(c.b));
  for (int64_t i = 0; i < c.b; ++i) {
    seq_q[i] = static_cast<int>(c.sq - 3 * i - 1 > 1 ? c.sq - 3 * i - 1 : 1);
    seq_kv[i] = static_cast<int>(c.skv - 2 * i - 2 > 1 ? c.skv - 2 * i - 2 : 1);
  }
  // Ragged (THD) tensors: where each element lives, and the offsets.
  std::vector<double> rq_off, rkv_off;
  const std::vector<int64_t> pq = c.ragged ? ragged_phys(c.b, c.hq, c.sq, c.d, seq_q, &rq_off) : std::vector<int64_t>{};
  const std::vector<int64_t> pkv = c.ragged ? ragged_phys(c.b, c.hk, c.skv, c.d, seq_kv, &rkv_off) : std::vector<int64_t>{};
  const std::vector<int64_t>* PQ = c.ragged ? &pq : nullptr;
  const std::vector<int64_t>* PKV = c.ragged ? &pkv : nullptr;
  auto RagQ = make_buf(fe::DataType_t::INT32, {c.b + 1, 1, 1, 1}, {1, 1, 1, 1}, rq_off);
  auto RagKV = make_buf(fe::DataType_t::INT32, {c.b + 1, 1, 1, 1}, {1, 1, 1, 1}, rkv_off);
  auto Q = make_buf(T, {c.b, c.hq, c.sq, c.d}, strides_of(c, c.hq, c.sq, c.d), Qv, PQ);
  auto K = make_buf(T, {c.b, c.hk, c.skv, c.d}, strides_of(c, c.hk, c.skv, c.d), Kv, PKV);
  auto V = make_buf(T, {c.b, c.hk, c.skv, c.dv}, strides_of(c, c.hk, c.skv, c.dv), Vv, PKV);
  auto O = make_buf(T, {c.b, c.hq, c.sq, c.dv}, strides_of(c, c.hq, c.sq, c.dv), {}, PQ);
  auto dO = make_buf(T, {c.b, c.hq, c.sq, c.dv}, strides_of(c, c.hq, c.sq, c.dv), dOv, PQ);
  auto Bias = make_buf(T, {1, c.hq, c.sq, c.skv}, {c.hq * c.sq * c.skv, c.sq * c.skv, c.skv, 1}, Bv);
  auto Stats = make_buf(fe::DataType_t::FLOAT, {c.b, c.hq, c.sq, 1}, {c.hq * c.sq, c.sq, 1, 1}, {});
  auto Mask = make_buf(fe::DataType_t::FLOAT, {c.b, c.hq, c.sq, c.skv},
                       {c.hq * c.sq * c.skv, c.sq * c.skv, c.skv, 1}, {});
  std::vector<double> sq_d(seq_q.begin(), seq_q.end()), skv_d(seq_kv.begin(), seq_kv.end());
  auto SeqQ = make_buf(fe::DataType_t::INT32, {c.b, 1, 1, 1}, {1, 1, 1, 1}, sq_d);
  auto SeqKV = make_buf(fe::DataType_t::INT32, {c.b, 1, 1, 1}, {1, 1, 1, 1}, skv_d);
  int64_t seed_v = 1234567, offset_v = 89;
  auto Seed = make_buf(fe::DataType_t::INT64, {1, 1, 1, 1}, {1, 1, 1, 1}, {static_cast<double>(seed_v)});
  auto Offset = make_buf(fe::DataType_t::INT64, {1, 1, 1, 1}, {1, 1, 1, 1}, {static_cast<double>(offset_v)});

  enum : int64_t { kQ = 1, kK, kV, kO, kStats, kBias, kSeqQ, kSeqKV, kSeed, kOffset, kMask, kdO, kdQ, kdK, kdV, kdBias, kRagQ, kRagKV,
                   kTableK, kTableV };
  // Paged K and V: each batch's sequence in pages of c.page positions,
  // scattered over a container of blocks in a shuffled order the page table
  // records.
  const int64_t pages = c.page ? c.skv / c.page : 0, nblocks = c.b * pages;
  std::vector<double> table(static_cast<size_t>(nblocks));
  for (int64_t i = 0; i < nblocks; ++i) table[i] = static_cast<double>((i * 5 + 3) % nblocks);
  auto paged = [&](const std::vector<double>& logical, int64_t D) {
    std::vector<double> cont(static_cast<size_t>(nblocks * c.hk * c.page * D), 0.0);
    for (int64_t b = 0; b < c.b; ++b)
      for (int64_t h = 0; h < c.hk; ++h)
        for (int64_t s = 0; s < c.skv; ++s)
          for (int64_t d = 0; d < D; ++d) {
            const int64_t blk = static_cast<int64_t>(table[b * pages + s / c.page]);
            cont[((blk * c.hk + h) * c.page + s % c.page) * D + d] = logical[((b * c.hk + h) * c.skv + s) * D + d];
          }
    return cont;
  };
  auto KC = c.page ? make_buf(T, {nblocks, c.hk, c.page, c.d}, {c.hk * c.page * c.d, c.page * c.d, c.d, 1}, paged(K->v, c.d)) : nullptr;
  auto VC = c.page ? make_buf(T, {nblocks, c.hk, c.page, c.dv}, {c.hk * c.page * c.dv, c.page * c.dv, c.dv, 1}, paged(V->v, c.dv)) : nullptr;
  auto Table = make_buf(fe::DataType_t::INT32, {c.b, 1, pages ? pages : 1, 1}, {pages ? pages : 1, pages ? pages : 1, 1, 1}, table);
  auto common = [&](fe::graph::Graph& g) {
    g.set_io_data_type(T).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
  };
  auto tensor = [&](fe::graph::Graph& g, int64_t uid, const char* n, std::vector<int64_t> dim, std::vector<int64_t> str,
                    fe::DataType_t t = fe::DataType_t::NOT_SET) {
    auto a = fe::graph::Tensor_attributes().set_name(n).set_uid(uid).set_dim(dim).set_stride(str);
    if (t != fe::DataType_t::NOT_SET) a.set_data_type(t);
    return g.tensor(a);
  };
  // The ragged offsets of a graph's Q-side and K/V-side tensors.
  auto ragged = [&](fe::graph::Graph& g, bool q_side) {
    return tensor(g, q_side ? kRagQ : kRagKV, q_side ? "ragged_q" : "ragged_kv", {c.b + 1, 1, 1, 1}, {1, 1, 1, 1},
                  fe::DataType_t::INT32);
  };
  const bool dropout = c.dropout > 0;

  // Forward.
  fe::graph::Graph fg;
  common(fg);
  auto q = tensor(fg, kQ, "Q", {c.b, c.hq, c.sq, c.d}, strides_of(c, c.hq, c.sq, c.d));
  auto k = c.page ? tensor(fg, kK, "K", {nblocks, c.hk, c.page, c.d}, {c.hk * c.page * c.d, c.page * c.d, c.d, 1})
                   : tensor(fg, kK, "K", {c.b, c.hk, c.skv, c.d}, strides_of(c, c.hk, c.skv, c.d));
  auto v = c.page ? tensor(fg, kV, "V", {nblocks, c.hk, c.page, c.dv}, {c.hk * c.page * c.dv, c.page * c.dv, c.dv, 1})
                   : tensor(fg, kV, "V", {c.b, c.hk, c.skv, c.dv}, strides_of(c, c.hk, c.skv, c.dv));
  std::shared_ptr<fe::graph::Tensor_attributes> frq, frkv;
  if (c.ragged) {
    frq = ragged(fg, true), frkv = ragged(fg, false);
    q->set_ragged_offset(frq), k->set_ragged_offset(frkv), v->set_ragged_offset(frkv);
  }
  auto opts = fe::graph::SDPA_attributes().set_name("sdpa").set_generate_stats(c.backward).set_attn_scale(c.scale);
  opts.set_implementation(c.impl);
  if (c.causal) {
    opts.set_diagonal_alignment(c.bottom_right ? fe::DiagonalAlignment_t::BOTTOM_RIGHT : fe::DiagonalAlignment_t::TOP_LEFT)
        .set_diagonal_band_right_bound(0);
  }
  if (c.left >= 0) opts.set_diagonal_band_left_bound(c.left);
  if (c.page) {
    opts.set_paged_attention_k_table(tensor(fg, kTableK, "table_k", {c.b, 1, pages, 1}, {pages, pages, 1, 1}, fe::DataType_t::INT32));
    opts.set_paged_attention_v_table(tensor(fg, kTableV, "table_v", {c.b, 1, pages, 1}, {pages, pages, 1, 1}, fe::DataType_t::INT32));
    opts.set_paged_attention_max_seq_len_kv(static_cast<int>(c.skv));
  }
  if (c.bias) opts.set_bias(tensor(fg, kBias, "bias", {1, c.hq, c.sq, c.skv}, {c.hq * c.sq * c.skv, c.sq * c.skv, c.skv, 1}));
  if (c.padding)
    opts.set_padding_mask(true)
        .set_seq_len_q(tensor(fg, kSeqQ, "seq_q", {c.b, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT32))
        .set_seq_len_kv(tensor(fg, kSeqKV, "seq_kv", {c.b, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT32));
  if (dropout) {
    auto sd = tensor(fg, kSeed, "seed", {1, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT64);
    auto of = tensor(fg, kOffset, "offset", {1, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT64);
    opts.set_dropout(c.dropout, sd, of);
    opts.set_rng_dump(tensor(fg, kMask, "rng_dump", {c.b, c.hq, c.sq, c.skv}, {c.hq * c.sq * c.skv, c.sq * c.skv, c.skv, 1},
                             fe::DataType_t::FLOAT));
  }
  auto [o, stats] = fg.sdpa(q, k, v, opts);
  o->set_output(true).set_uid(kO).set_dim({c.b, c.hq, c.sq, c.dv}).set_stride(strides_of(c, c.hq, c.sq, c.dv));
  if (stats) stats->set_output(true).set_uid(kStats).set_data_type(fe::DataType_t::FLOAT);
  if (c.ragged) o->set_ragged_offset(frq);
  Var pack{{kQ, Q->d->p}, {kK, K->d->p}, {kV, V->d->p}, {kO, O->d->p}};
  if (c.page) pack[kK] = KC->d->p, pack[kV] = VC->d->p, pack[kTableK] = Table->d->p, pack[kTableV] = Table->d->p;
  if (c.ragged) pack[kRagQ] = RagQ->d->p, pack[kRagKV] = RagKV->d->p;
  if (c.backward) pack[kStats] = Stats->d->p;
  if (c.bias) pack[kBias] = Bias->d->p;
  if (c.padding) pack[kSeqQ] = SeqQ->d->p, pack[kSeqKV] = SeqKV->d->p;
  if (dropout) pack[kSeed] = Seed->d->p, pack[kOffset] = Offset->d->p, pack[kMask] = Mask->d->p;
  const std::string fwd_name = c.name + " forward";
  const int ran = build_and_run(fwd_name.c_str(), fg, handle, pack);
  if (ran == 0 && !kOnSim) {
    std::printf("skip %s (no engine offered on this GPU)\n", fwd_name.c_str());
    return;
  }
  if (ran != 1) { expect(fwd_name + " runs", false); return; }

  std::vector<double> maskv;
  if (dropout) {
    maskv = read_buf(*Mask);
    size_t kept_n = 0, bad = 0;
    for (double m : maskv) kept_n += m == 1.0, bad += m != 0.0 && m != 1.0;
    const double frac = static_cast<double>(kept_n) / static_cast<double>(maskv.size());
    expect(c.name + ": the dropout mask is 0/1 and keeps about 1 - p", !bad && std::fabs(frac - (1.0 - c.dropout)) < 0.08,
           frac);
  }
  const Ref ref = reference_fwd(c, Q->v, K->v, V->v, c.bias ? &Bias->v : nullptr, seq_q, seq_kv, dropout ? &maskv : nullptr);
  std::vector<double> got = read_buf(*O), want = ref.O;
  if (c.padding)  // rows past a batch's sequence length are the library's own
    for (int64_t bb = 0; bb < c.b; ++bb)
      for (int64_t h = 0; h < c.hq; ++h)
        for (int64_t i = seq_q[bb]; i < c.sq; ++i)
          for (int64_t e = 0; e < c.dv; ++e) got[((bb * c.hq + h) * c.sq + i) * c.dv + e] = want[((bb * c.hq + h) * c.sq + i) * c.dv + e];
  double e = err_of(got, want);
  expect(fwd_name + ": O", e < tol, e);
  if (c.backward) {
    std::vector<double> gs = read_buf(*Stats), ws = ref.stats;
    if (c.padding)
      for (int64_t bb = 0; bb < c.b; ++bb)
        for (int64_t h = 0; h < c.hq; ++h)
          for (int64_t i = seq_q[bb]; i < c.sq; ++i) gs[(bb * c.hq + h) * c.sq + i] = ws[(bb * c.hq + h) * c.sq + i];
    e = err_of(gs, ws);
    expect(fwd_name + ": the log-sum-exp statistics", e < 1e-4 + (T == fe::DataType_t::FLOAT ? 0 : 1e-3), e);
  }
  if (!c.backward) return;

  // Backward, from the library's own O and statistics.
  auto Og = make_buf(T, {c.b, c.hq, c.sq, c.dv}, strides_of(c, c.hq, c.sq, c.dv), read_buf(*O), PQ);
  Ref fref = ref;
  fref.O = Og->v;
  auto dQ = make_buf(T, {c.b, c.hq, c.sq, c.d}, strides_of(c, c.hq, c.sq, c.d), {}, PQ);
  auto dK = make_buf(T, {c.b, c.hk, c.skv, c.d}, strides_of(c, c.hk, c.skv, c.d), {}, PKV);
  auto dV = make_buf(T, {c.b, c.hk, c.skv, c.dv}, strides_of(c, c.hk, c.skv, c.dv), {}, PKV);
  auto dBias = make_buf(T, {1, c.hq, c.sq, c.skv}, {c.hq * c.sq * c.skv, c.sq * c.skv, c.skv, 1}, {});
  auto Mask2 = make_buf(fe::DataType_t::FLOAT, {c.b, c.hq, c.sq, c.skv}, {c.hq * c.sq * c.skv, c.sq * c.skv, c.skv, 1}, {});
  fe::graph::Graph bg;
  common(bg);
  auto bq = tensor(bg, kQ, "Q", {c.b, c.hq, c.sq, c.d}, strides_of(c, c.hq, c.sq, c.d));
  auto bk = tensor(bg, kK, "K", {c.b, c.hk, c.skv, c.d}, strides_of(c, c.hk, c.skv, c.d));
  auto bv = tensor(bg, kV, "V", {c.b, c.hk, c.skv, c.dv}, strides_of(c, c.hk, c.skv, c.dv));
  auto bo = tensor(bg, kO, "O", {c.b, c.hq, c.sq, c.dv}, strides_of(c, c.hq, c.sq, c.dv));
  auto bdo = tensor(bg, kdO, "dO", {c.b, c.hq, c.sq, c.dv}, strides_of(c, c.hq, c.sq, c.dv));
  auto bst = tensor(bg, kStats, "stats", {c.b, c.hq, c.sq, 1}, {c.hq * c.sq, c.sq, 1, 1}, fe::DataType_t::FLOAT);
  std::shared_ptr<fe::graph::Tensor_attributes> brq, brkv;
  if (c.ragged) {
    brq = ragged(bg, true), brkv = ragged(bg, false);
    bq->set_ragged_offset(brq), bo->set_ragged_offset(brq), bdo->set_ragged_offset(brq);
    bk->set_ragged_offset(brkv), bv->set_ragged_offset(brkv);
  }
  auto bopts = fe::graph::SDPA_backward_attributes().set_name("sdpa_backward").set_attn_scale(c.scale);
  if (c.ragged) {
    int64_t tq = 0, tkv = 0;
    for (int64_t i = 0; i < c.b; ++i) tq += seq_q[i], tkv += seq_kv[i];
    bopts.set_max_total_seq_len_q(tq).set_max_total_seq_len_kv(tkv);
  }
  if (c.causal) {
    bopts.set_diagonal_alignment(c.bottom_right ? fe::DiagonalAlignment_t::BOTTOM_RIGHT : fe::DiagonalAlignment_t::TOP_LEFT)
        .set_diagonal_band_right_bound(0);
  }
  if (c.left >= 0) bopts.set_diagonal_band_left_bound(c.left);
  if (c.bias) {
    bopts.set_bias(tensor(bg, kBias, "bias", {1, c.hq, c.sq, c.skv}, {c.hq * c.sq * c.skv, c.sq * c.skv, c.skv, 1}));
    bopts.set_dbias(tensor(bg, kdBias, "dbias", {1, c.hq, c.sq, c.skv}, {c.hq * c.sq * c.skv, c.sq * c.skv, c.skv, 1}));
  }
  if (c.padding)
    bopts.set_padding_mask(true)
        .set_seq_len_q(tensor(bg, kSeqQ, "seq_q", {c.b, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT32))
        .set_seq_len_kv(tensor(bg, kSeqKV, "seq_kv", {c.b, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT32));
  if (dropout) {
    auto sd = tensor(bg, kSeed, "seed", {1, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT64);
    auto of = tensor(bg, kOffset, "offset", {1, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT64);
    bopts.set_dropout(c.dropout, sd, of);
    bopts.set_rng_dump(tensor(bg, kMask, "rng_dump", {c.b, c.hq, c.sq, c.skv}, {c.hq * c.sq * c.skv, c.sq * c.skv, c.skv, 1},
                              fe::DataType_t::FLOAT));
  }
  auto [dq, dk, dv] = bg.sdpa_backward(bq, bk, bv, bo, bdo, bst, bopts);
  dq->set_output(true).set_uid(kdQ).set_dim({c.b, c.hq, c.sq, c.d}).set_stride(strides_of(c, c.hq, c.sq, c.d));
  dk->set_output(true).set_uid(kdK).set_dim({c.b, c.hk, c.skv, c.d}).set_stride(strides_of(c, c.hk, c.skv, c.d));
  dv->set_output(true).set_uid(kdV).set_dim({c.b, c.hk, c.skv, c.dv}).set_stride(strides_of(c, c.hk, c.skv, c.dv));
  if (c.ragged) dq->set_ragged_offset(brq), dk->set_ragged_offset(brkv), dv->set_ragged_offset(brkv);
  Var bpack{{kQ, Q->d->p}, {kK, K->d->p}, {kV, V->d->p}, {kO, Og->d->p}, {kdO, dO->d->p}, {kStats, Stats->d->p},
            {kdQ, dQ->d->p}, {kdK, dK->d->p}, {kdV, dV->d->p}};
  if (c.bias) bpack[kBias] = Bias->d->p, bpack[kdBias] = dBias->d->p;
  if (c.padding) bpack[kSeqQ] = SeqQ->d->p, bpack[kSeqKV] = SeqKV->d->p;
  if (c.ragged) bpack[kRagQ] = RagQ->d->p, bpack[kRagKV] = RagKV->d->p;
  if (dropout) bpack[kSeed] = Seed->d->p, bpack[kOffset] = Offset->d->p, bpack[kMask] = Mask2->d->p;
  const std::string bwd_name = c.name + " backward";
  const int bran = build_and_run(bwd_name.c_str(), bg, handle, bpack);
  if (bran == 0 && !kOnSim) {
    std::printf("skip %s (no engine offered on this GPU)\n", bwd_name.c_str());
    return;
  }
  if (bran != 1) { expect(bwd_name + " runs", false); return; }
  if (dropout) {
    const auto m2 = read_buf(*Mask2);
    expect(c.name + ": the backward pass regenerates the forward pass's dropout mask", m2 == maskv);
  }
  const Grads gr = reference_bwd(c, Q->v, K->v, V->v, fref, dO->v, seq_q, dropout ? &maskv : nullptr);
  // Gradients sum over a sequence: allow for that in the tolerance.
  const double gtol = tol * 4;
  auto check = [&](const char* n, const Buf& b, const std::vector<double>& want_all, bool q_rows) {
    std::vector<double> got_all = read_buf(b), w = want_all;
    if (c.padding && q_rows)
      for (int64_t bb = 0; bb < c.b; ++bb)
        for (int64_t h = 0; h < c.hq; ++h)
          for (int64_t i = seq_q[bb]; i < c.sq; ++i)
            for (int64_t x = 0; x < c.d; ++x) got_all[((bb * c.hq + h) * c.sq + i) * c.d + x] = w[((bb * c.hq + h) * c.sq + i) * c.d + x];
    const double er = err_of(got_all, w);
    expect(bwd_name + ": " + n, er < gtol, er);
  };
  check("dQ", *dQ, gr.dQ, true);
  check("dK", *dK, gr.dK, false);
  check("dV", *dV, gr.dV, false);
  if (c.bias) check("dBias", *dBias, gr.dBias, false);
}

int main() {
  cudnnHandle_t handle;
  if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
    std::printf("SKIP: cudnnCreate failed\n");
    return 0;
  }
  std::printf("cuDNN %zu, cudnn-frontend %d.%d.%d\n", cudnnGetVersion(), CUDNN_FRONTEND_MAJOR_VERSION,
              CUDNN_FRONTEND_MINOR_VERSION, CUDNN_FRONTEND_PATCH_VERSION);
  using I = fe::AttentionImplementation_t;
  std::vector<Cfg> cfgs;
  auto add = [&](const char* name, auto f) {
    Cfg c;
    c.name = name;
    f(c);
    cfgs.push_back(c);
  };
  add("half, unified", [](Cfg& c) { c.impl = I::UNIFIED; c.backward = false; });
  add("half, unified, causal", [](Cfg& c) { c.impl = I::UNIFIED; c.causal = true; });
  add("half, composite", [](Cfg& c) { c.impl = I::COMPOSITE; });
  add("half, composite, causal", [](Cfg& c) { c.impl = I::COMPOSITE; c.causal = true; });
  add("bfloat16, composite, causal, s_q != s_kv", [](Cfg& c) {
    c.io = fe::DataType_t::BFLOAT16; c.impl = I::COMPOSITE; c.causal = true; c.sq = 8; c.skv = 24;
  });
  add("half, composite, bottom-right causal, s_q < s_kv", [](Cfg& c) {
    c.impl = I::COMPOSITE; c.causal = true; c.bottom_right = true; c.sq = 8; c.skv = 24;
  });
  add("half, composite, sliding window", [](Cfg& c) { c.impl = I::COMPOSITE; c.causal = true; c.left = 4; });
  add("half, composite, grouped-query heads", [](Cfg& c) { c.impl = I::COMPOSITE; c.hq = 4; c.hk = 2; c.causal = true; });
  add("half, composite, bias", [](Cfg& c) { c.impl = I::COMPOSITE; c.bias = true; });
  add("half, composite, padding", [](Cfg& c) { c.impl = I::COMPOSITE; c.padding = true; c.causal = true; });
  add("half, composite, ragged (packed THD) sequences", [](Cfg& c) { c.impl = I::COMPOSITE; c.ragged = c.padding = true; c.causal = true; });
  add("bfloat16, unified, ragged (packed THD) sequences", [](Cfg& c) {
    c.io = fe::DataType_t::BFLOAT16; c.impl = I::UNIFIED; c.ragged = c.padding = true; c.backward = false;
  });
  add("half, unified, padding", [](Cfg& c) { c.impl = I::UNIFIED; c.padding = true; c.backward = false; });
  add("half, composite, paged K/V caches", [](Cfg& c) { c.impl = I::COMPOSITE; c.page = 4; c.padding = true; c.backward = false; });
  add("half, unified, paged K/V caches", [](Cfg& c) { c.impl = I::UNIFIED; c.page = 4; c.padding = true; c.backward = false; });
  add("half, composite, BSHD layout, d = 32", [](Cfg& c) { c.impl = I::COMPOSITE; c.bhsd_interleaved = true; c.d = c.dv = 32; });
  add("half, composite, dropout", [](Cfg& c) { c.impl = I::COMPOSITE; c.dropout = 0.25f; c.sq = c.skv = 32; });
  add("float, composite", [](Cfg& c) { c.io = fe::DataType_t::FLOAT; c.impl = I::COMPOSITE; c.causal = true; });
  add("half, auto", [](Cfg& c) { c.impl = I::AUTO; c.causal = true; });
  for (const Cfg& c : cfgs) {
    try {
      run(c, handle);
    } catch (const std::exception& ex) {
      if (kOnSim) expect(c.name + " (exception: " + ex.what() + ")", false);
      else std::printf("skip %s (frontend: %s)\n", c.name.c_str(), ex.what());
    }
  }
  cudnnDestroy(handle);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
