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
  bool sink = false;              // a per-head sink logit that joins each softmax denominator (forward and backward)
  bool rope = false;              // rotary embeddings on Q and K feed the attention (fused, as cuDNN's engines run them); dQ and dK go back through the inverse rotation
  int64_t rope_dim = 0;           // the width rotated (0: the whole head dimension), the rest scaled only
  float rope_scale = 1.0f;        // the embeddings' output scale
  int cu_seq = 0;                 // sequence lengths as cumulative sums (cu_seq_len_q/kv; unified, forward only; implies padding): 1 both sides, 2 only Q's, 3 only K/V's
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
                         const std::vector<double>* mask, const std::vector<double>* sink = nullptr) {
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
        // A sink is one more logit in the row's softmax whose probability goes nowhere.
        if (sink) mx = std::max(mx, (*sink)[h]);
        double sum = sink ? std::exp((*sink)[h] - mx) : 0.0;
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
  std::vector<double> dQ, dK, dV, dBias, dSink;
};
static Grads reference_bwd(const Cfg& c, const std::vector<double>& Q, const std::vector<double>& K,
                           const std::vector<double>& V, const Ref& f, const std::vector<double>& dO,
                           const std::vector<int>& seq_q, const std::vector<double>* mask,
                           const std::vector<double>* sink = nullptr) {
  Grads g;
  g.dSink.assign(static_cast<size_t>(c.hq), 0.0);
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
        // The sink's probability exp(sink - stats) takes no part in O, so its gradient is -p_sink * D.
        if (sink) g.dSink[h] -= std::exp((*sink)[h] - f.stats[(bb * c.hq + h) * c.sq + i]) * D;
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

// Rotary embedding (non-interleaved) of x [B*H, S, D] with angles fr [S, r] (r = the rotated
// width): the last r elements of a row rotate in pairs (j, j + r/2) by fr[s][j], all scaled; the
// leading D - r are scaled only. bwd: the inverse rotation. Measured on an RTX 3060.
static std::vector<double> rope_ref(const std::vector<double>& x, const std::vector<double>& fr, int64_t BH, int64_t S, int64_t D,
                                    int64_t r, double scale, bool bwd) {
  std::vector<double> y(x.size());
  const int64_t nope = D - r, half = r / 2;
  for (int64_t bh = 0; bh < BH; ++bh)
    for (int64_t s = 0; s < S; ++s) {
      const size_t row = static_cast<size_t>((bh * S + s) * D);
      for (int64_t d = 0; d < nope; ++d) y[row + d] = x[row + d] * scale;
      for (int64_t j = 0; j < half; ++j) {
        const double a = fr[static_cast<size_t>(s * r + j)], c = std::cos(a) * scale, sn = std::sin(a) * scale;
        const double x1 = x[row + nope + j], x2 = x[row + nope + half + j];
        y[row + nope + j] = bwd ? x1 * c + x2 * sn : x1 * c - x2 * sn;
        y[row + nope + half + j] = bwd ? x2 * c - x1 * sn : x2 * c + x1 * sn;
      }
    }
  return y;
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
  // Cumulative sequence lengths: cu[b + 1] = cu[b] + length[b], in tokens.
  std::vector<double> cuq_d(static_cast<size_t>(c.b + 1), 0.0), cukv_d(static_cast<size_t>(c.b + 1), 0.0);
  for (int64_t i = 0; i < c.b; ++i) cuq_d[i + 1] = cuq_d[i] + seq_q[i], cukv_d[i + 1] = cukv_d[i] + seq_kv[i];
  auto CuQ = make_buf(fe::DataType_t::INT32, {c.b + 1, 1, 1, 1}, {1, 1, 1, 1}, cuq_d);
  auto CuKV = make_buf(fe::DataType_t::INT32, {c.b + 1, 1, 1, 1}, {1, 1, 1, 1}, cukv_d);
  const auto Sinkv = randoms(static_cast<size_t>(c.hq), 6, -1.0, 1.0);
  auto Sink = make_buf(fe::DataType_t::FLOAT, {1, c.hq, 1, 1}, {c.hq, 1, 1, 1}, Sinkv);
  auto dSink = make_buf(fe::DataType_t::FLOAT, {1, c.hq, 1, 1}, {c.hq, 1, 1, 1}, {});
  // Rotary embeddings: angles [max(S_q, S_kv), 1, 1, r], the rotated Q and K the library hands back.
  const int64_t rope_r = c.rope_dim ? c.rope_dim : c.d, rope_s = std::max(c.sq, c.skv);
  const auto Freqv = [&] {
    std::vector<double> f(static_cast<size_t>(rope_s * rope_r));
    for (int64_t s2 = 0; s2 < rope_s; ++s2)
      for (int64_t j = 0; j < rope_r; ++j) f[s2 * rope_r + j] = static_cast<double>(static_cast<float>(0.07 * (s2 + 1) * (1 + (j % (rope_r / 2)) * 0.31)));
    return f;
  }();
  auto Freqs = make_buf(fe::DataType_t::FLOAT, {rope_s, 1, 1, rope_r}, {rope_r, rope_r, rope_r, 1}, Freqv);
  auto QRot = make_buf(T, {c.b, c.hq, c.sq, c.d}, strides_of(c, c.hq, c.sq, c.d), {});
  auto KRot = make_buf(T, {c.b, c.hk, c.skv, c.d}, strides_of(c, c.hk, c.skv, c.d), {});
  int64_t seed_v = 1234567, offset_v = 89;
  auto Seed = make_buf(fe::DataType_t::INT64, {1, 1, 1, 1}, {1, 1, 1, 1}, {static_cast<double>(seed_v)});
  auto Offset = make_buf(fe::DataType_t::INT64, {1, 1, 1, 1}, {1, 1, 1, 1}, {static_cast<double>(offset_v)});

  enum : int64_t { kQ = 1, kK, kV, kO, kStats, kBias, kSeqQ, kSeqKV, kSeed, kOffset, kMask, kdO, kdQ, kdK, kdV, kdBias, kRagQ, kRagKV,
                   kTableK, kTableV, kSink, kdSink, kCuQ, kCuKV, kFreqs, kQRot, kKRot, kdQRaw };
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
  if (c.padding) {
    opts.set_padding_mask(true);
    if (c.cu_seq == 1 || c.cu_seq == 2)
      opts.set_cu_seq_len_q(tensor(fg, kCuQ, "cu_seq_q", {c.b + 1, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT32));
    else
      opts.set_seq_len_q(tensor(fg, kSeqQ, "seq_q", {c.b, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT32));
    if (c.cu_seq == 1 || c.cu_seq == 3)
      opts.set_cu_seq_len_kv(tensor(fg, kCuKV, "cu_seq_kv", {c.b + 1, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT32));
    else
      opts.set_seq_len_kv(tensor(fg, kSeqKV, "seq_kv", {c.b, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT32));
  }
  if (c.sink) opts.set_sink_token(tensor(fg, kSink, "sink", {1, c.hq, 1, 1}, {c.hq, 1, 1, 1}, fe::DataType_t::FLOAT));
  if (dropout) {
    auto sd = tensor(fg, kSeed, "seed", {1, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT64);
    auto of = tensor(fg, kOffset, "offset", {1, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT64);
    opts.set_dropout(c.dropout, sd, of);
    opts.set_rng_dump(tensor(fg, kMask, "rng_dump", {c.b, c.hq, c.sq, c.skv}, {c.hq * c.sq * c.skv, c.sq * c.skv, c.skv, 1},
                             fe::DataType_t::FLOAT));
  }
  std::shared_ptr<fe::graph::Tensor_attributes> sq_in = q, sk_in = k;
  if (c.rope) {
    auto fr = tensor(fg, kFreqs, "freqs", {rope_s, 1, 1, rope_r}, {rope_r, rope_r, rope_r, 1}, fe::DataType_t::FLOAT);
    auto ra = fe::graph::RoPE_attributes().set_name("rope_q").set_output_scale(c.rope_scale);
    auto rb = fe::graph::RoPE_attributes().set_name("rope_k").set_output_scale(c.rope_scale);
    if (c.rope_dim) ra.set_rope_dim(c.rope_dim), rb.set_rope_dim(c.rope_dim);
    sq_in = fg.rope(q, fr, ra);
    sq_in->set_output(true).set_uid(kQRot).set_data_type(T).set_dim({c.b, c.hq, c.sq, c.d}).set_stride(strides_of(c, c.hq, c.sq, c.d));
    sk_in = fg.rope(k, fr, rb);
    sk_in->set_output(true).set_uid(kKRot).set_data_type(T).set_dim({c.b, c.hk, c.skv, c.d}).set_stride(strides_of(c, c.hk, c.skv, c.d));
  }
  auto [o, stats] = fg.sdpa(sq_in, sk_in, v, opts);
  o->set_output(true).set_uid(kO).set_dim({c.b, c.hq, c.sq, c.dv}).set_stride(strides_of(c, c.hq, c.sq, c.dv));
  if (stats) stats->set_output(true).set_uid(kStats).set_data_type(fe::DataType_t::FLOAT);
  if (c.ragged) o->set_ragged_offset(frq);
  Var pack{{kQ, Q->d->p}, {kK, K->d->p}, {kV, V->d->p}, {kO, O->d->p}};
  if (c.page) pack[kK] = KC->d->p, pack[kV] = VC->d->p, pack[kTableK] = Table->d->p, pack[kTableV] = Table->d->p;
  if (c.ragged) pack[kRagQ] = RagQ->d->p, pack[kRagKV] = RagKV->d->p;
  if (c.backward) pack[kStats] = Stats->d->p;
  if (c.bias) pack[kBias] = Bias->d->p;
  if (c.padding) {
    if (c.cu_seq == 1 || c.cu_seq == 2) pack[kCuQ] = CuQ->d->p; else pack[kSeqQ] = SeqQ->d->p;
    if (c.cu_seq == 1 || c.cu_seq == 3) pack[kCuKV] = CuKV->d->p; else pack[kSeqKV] = SeqKV->d->p;
  }
  if (c.sink) pack[kSink] = Sink->d->p;
  if (c.rope) pack[kFreqs] = Freqs->d->p, pack[kQRot] = QRot->d->p, pack[kKRot] = KRot->d->p;
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
  // The queries and keys the attention saw: the rotated ones, which the library hands back and
  // which are checked against the reference rotation (to the rounding of the output type).
  std::vector<double> Qeff = Q->v, Keff = K->v;
  if (c.rope) {
    const std::vector<double> qr = rope_ref(Q->v, Freqv, c.b * c.hq, c.sq, c.d, rope_r, c.rope_scale, false);
    const std::vector<double> kr = rope_ref(K->v, Freqv, c.b * c.hk, c.skv, c.d, rope_r, c.rope_scale, false);
    Qeff = read_buf(*QRot), Keff = read_buf(*KRot);
    expect(c.name + ": the rotated queries", err_of(Qeff, qr) < tol, err_of(Qeff, qr));
    expect(c.name + ": the rotated keys", err_of(Keff, kr) < tol, err_of(Keff, kr));
  }
  const Ref ref = reference_fwd(c, Qeff, Keff, V->v, c.bias ? &Bias->v : nullptr, seq_q, seq_kv, dropout ? &maskv : nullptr,
                                c.sink ? &Sinkv : nullptr);
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
  // With rotary embeddings the backward pass is given the rotated Q and K, and its dQ goes back
  // through the inverse rotation.
  auto QRB = c.rope ? make_buf(T, {c.b, c.hq, c.sq, c.d}, strides_of(c, c.hq, c.sq, c.d), Qeff) : nullptr;
  auto KRB = c.rope ? make_buf(T, {c.b, c.hk, c.skv, c.d}, strides_of(c, c.hk, c.skv, c.d), Keff) : nullptr;
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
  if (c.sink) {
    bopts.set_sink_token(tensor(bg, kSink, "sink", {1, c.hq, 1, 1}, {c.hq, 1, 1, 1}, fe::DataType_t::FLOAT));
    bopts.set_dsink_token(tensor(bg, kdSink, "dsink", {1, c.hq, 1, 1}, {c.hq, 1, 1, 1}, fe::DataType_t::FLOAT));
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
  if (c.rope) {
    auto fr = tensor(bg, kFreqs, "freqs", {rope_s, 1, 1, rope_r}, {rope_r, rope_r, rope_r, 1}, fe::DataType_t::FLOAT);
    auto ra = fe::graph::RoPE_backward_attributes().set_name("rope_q_bwd").set_output_scale(c.rope_scale);
    if (c.rope_dim) ra.set_rope_dim(c.rope_dim);
    // dQ, before its rotation, is an output too: the library reads its pointer.
    dq->set_output(true).set_uid(kdQRaw).set_data_type(T).set_dim({c.b, c.hq, c.sq, c.d}).set_stride(strides_of(c, c.hq, c.sq, c.d));
    dq = bg.rope_backward(dq, fr, ra);  // (cuDNN's engines take one inverse rotation, dQ's, in this graph)
  }
  dq->set_output(true).set_uid(kdQ).set_dim({c.b, c.hq, c.sq, c.d}).set_stride(strides_of(c, c.hq, c.sq, c.d));
  dk->set_output(true).set_uid(kdK).set_dim({c.b, c.hk, c.skv, c.d}).set_stride(strides_of(c, c.hk, c.skv, c.d));
  dv->set_output(true).set_uid(kdV).set_dim({c.b, c.hk, c.skv, c.dv}).set_stride(strides_of(c, c.hk, c.skv, c.dv));
  if (c.ragged) dq->set_ragged_offset(brq), dk->set_ragged_offset(brkv), dv->set_ragged_offset(brkv);
  Var bpack{{kQ, c.rope ? QRB->d->p : Q->d->p}, {kK, c.rope ? KRB->d->p : K->d->p}, {kV, V->d->p}, {kO, Og->d->p}, {kdO, dO->d->p}, {kStats, Stats->d->p},
            {kdQ, dQ->d->p}, {kdK, dK->d->p}, {kdV, dV->d->p}};
  if (c.bias) bpack[kBias] = Bias->d->p, bpack[kdBias] = dBias->d->p;
  if (c.padding) bpack[kSeqQ] = SeqQ->d->p, bpack[kSeqKV] = SeqKV->d->p;
  if (c.sink) bpack[kSink] = Sink->d->p, bpack[kdSink] = dSink->d->p;
  auto dQRaw = c.rope ? make_buf(T, {c.b, c.hq, c.sq, c.d}, strides_of(c, c.hq, c.sq, c.d), {}) : nullptr;
  if (c.rope) bpack[kFreqs] = Freqs->d->p, bpack[kdQRaw] = dQRaw->d->p;
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
  Grads gr = reference_bwd(c, Qeff, Keff, V->v, fref, dO->v, seq_q, dropout ? &maskv : nullptr,
                                 c.sink ? &Sinkv : nullptr);
  if (c.rope) {
    gr.dQ = rope_ref(gr.dQ, Freqv, c.b * c.hq, c.sq, c.d, rope_r, c.rope_scale, true);
  }
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
  if (c.sink) {
    const double er = err_of(read_buf(*dSink), gr.dSink);
    expect(bwd_name + ": dSink", er < gtol, er);
  }
}

// ---- the backend's attention backward operation, with a sink --------------------------------------

// cudnn-frontend never emits CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR (its backward is the composite
// graph), so this one is built by hand. cuDNN's only engines for it are Blackwell's: an RTX 3060 offers
// none and the check is skipped there. Its sink semantics are those of the composite graph's, which the
// card runs and the checks above compare with the same reference: the sink is one more logit of every
// softmax row whose probability exp(sink - stats) goes to no value, and its gradient is
// -exp(sink - stats) * rowsum(dO * O), summed over the rows that share it.
using Desc = cudnnBackendDescriptor_t;
static Desc make_desc(cudnnBackendDescriptorType_t t) {
  Desc d = nullptr;
  cudnnBackendCreateDescriptor(t, &d);
  return d;
}
static void set_attr(Desc d, cudnnBackendAttributeName_t n, cudnnBackendAttributeType_t t, int64_t count, const void* v) {
  cudnnBackendSetAttribute(d, n, t, count, v);
}
static Desc raw_tensor(int64_t uid, cudnnDataType_t t, const std::vector<int64_t>& dim, bool by_value = false) {
  Desc d = make_desc(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  std::vector<int64_t> str(dim.size(), 1);
  for (size_t i = dim.size() - 1; i-- > 0;) str[i] = str[i + 1] * dim[i + 1];
  const int64_t align = 16;
  set_attr(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &t);
  set_attr(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, static_cast<int64_t>(dim.size()), dim.data());
  set_attr(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, static_cast<int64_t>(str.size()), str.data());
  set_attr(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  set_attr(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
  if (by_value) set_attr(d, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &by_value);
  cudnnBackendFinalize(d);
  return d;
}

static void raw_sdpa_backward(cudnnHandle_t handle) {
  Cfg c;
  c.name = "raw SDPA backward operation, half, sink";
  c.impl = fe::AttentionImplementation_t::UNIFIED;
  c.sink = true;
  const fe::DataType_t T = c.io;
  const auto Qv = randoms(static_cast<size_t>(c.b * c.hq * c.sq * c.d), 1), Kv = randoms(static_cast<size_t>(c.b * c.hk * c.skv * c.d), 2),
             Vv = randoms(static_cast<size_t>(c.b * c.hk * c.skv * c.dv), 3), dOv = randoms(static_cast<size_t>(c.b * c.hq * c.sq * c.dv), 4),
             Sinkv = randoms(static_cast<size_t>(c.hq), 6, -1.0, 1.0);
  const std::vector<int> none;
  auto Q = make_buf(T, {c.b, c.hq, c.sq, c.d}, {c.hq * c.sq * c.d, c.sq * c.d, c.d, 1}, Qv);
  auto K = make_buf(T, {c.b, c.hk, c.skv, c.d}, {c.hk * c.skv * c.d, c.skv * c.d, c.d, 1}, Kv);
  auto V = make_buf(T, {c.b, c.hk, c.skv, c.dv}, {c.hk * c.skv * c.dv, c.skv * c.dv, c.dv, 1}, Vv);
  Ref ref = reference_fwd(c, Q->v, K->v, V->v, nullptr, none, none, nullptr, &Sinkv);
  for (double& x : ref.O) x = round_to(T, x);
  auto O = make_buf(T, {c.b, c.hq, c.sq, c.dv}, {c.hq * c.sq * c.dv, c.sq * c.dv, c.dv, 1}, ref.O);
  auto dO = make_buf(T, {c.b, c.hq, c.sq, c.dv}, {c.hq * c.sq * c.dv, c.sq * c.dv, c.dv, 1}, dOv);
  auto Stats = make_buf(fe::DataType_t::FLOAT, {c.b, c.hq, c.sq, 1}, {c.hq * c.sq, c.sq, 1, 1}, ref.stats);
  auto Sink = make_buf(fe::DataType_t::FLOAT, {1, c.hq, 1, 1}, {c.hq, 1, 1, 1}, Sinkv);
  auto dQ = make_buf(T, {c.b, c.hq, c.sq, c.d}, {c.hq * c.sq * c.d, c.sq * c.d, c.d, 1}, {});
  auto dK = make_buf(T, {c.b, c.hk, c.skv, c.d}, {c.hk * c.skv * c.d, c.skv * c.d, c.d, 1}, {});
  auto dV = make_buf(T, {c.b, c.hk, c.skv, c.dv}, {c.hk * c.skv * c.dv, c.skv * c.dv, c.dv, 1}, {});
  auto dSink = make_buf(fe::DataType_t::FLOAT, {1, c.hq, 1, 1}, {c.hq, 1, 1, 1}, {});
  const cudnnDataType_t h16 = CUDNN_DATA_HALF, f32 = CUDNN_DATA_FLOAT;
  std::vector<Desc> keep;
  auto tens = [&](int64_t uid, cudnnDataType_t t, std::vector<int64_t> dim, bool bv = false) {
    keep.push_back(raw_tensor(uid, t, dim, bv));
    return keep.back();
  };
  Desc op = make_desc(CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR);
  auto attach = [&](cudnnBackendAttributeName_t n, Desc d) { set_attr(op, n, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &d); };
  attach(CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC, tens(1, h16, {c.b, c.hq, c.sq, c.d}));
  attach(CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC, tens(2, h16, {c.b, c.hk, c.skv, c.d}));
  attach(CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC, tens(3, h16, {c.b, c.hk, c.skv, c.dv}));
  attach(CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC, tens(4, h16, {c.b, c.hq, c.sq, c.dv}));
  attach(CUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC, tens(5, h16, {c.b, c.hq, c.sq, c.dv}));
  attach(CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC, tens(6, f32, {c.b, c.hq, c.sq, 1}));
  attach(CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC, tens(7, h16, {c.b, c.hq, c.sq, c.d}));
  attach(CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC, tens(8, h16, {c.b, c.hk, c.skv, c.d}));
  attach(CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC, tens(9, h16, {c.b, c.hk, c.skv, c.dv}));
  attach(CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC, tens(10, f32, {1, 1, 1, 1}, true));
  attach(CUDNN_ATTR_OPERATION_SDPA_BWD_SINK_DESC, tens(11, f32, {1, c.hq, 1, 1}));
  attach(CUDNN_ATTR_OPERATION_SDPA_BWD_DSINK_DESC, tens(12, f32, {1, c.hq, 1, 1}));
  if (cudnnBackendFinalize(op) != CUDNN_STATUS_SUCCESS) {
    std::printf("skip %s (the operation does not finalize here)\n", c.name.c_str());
    return;
  }
  Desc graph = make_desc(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  set_attr(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
  set_attr(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op);
  Desc plan = nullptr;
  if (cudnnBackendFinalize(graph) == CUDNN_STATUS_SUCCESS) {
    for (cudnnBackendHeurMode_t mode : {CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK}) {
      Desc heur = make_desc(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
      set_attr(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
      set_attr(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode);
      if (cudnnBackendFinalize(heur) == CUDNN_STATUS_SUCCESS) {
        int64_t n = 0;
        Desc cfg = make_desc(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &n, &cfg);
        if (n > 0 && !plan) {
          Desc p = make_desc(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
          set_attr(p, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
          set_attr(p, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfg);
          if (cudnnBackendFinalize(p) == CUDNN_STATUS_SUCCESS) plan = p;
        }
      }
    }
  }
  if (!plan) {
    if (!kOnSim) {
      std::printf("skip %s (no engine offered on this GPU)\n", c.name.c_str());
      return;
    }
    expect(c.name + " plans", false);
    return;
  }
  float scale = c.scale;
  std::vector<int64_t> uids{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<void*> ptrs{Q->d->p, K->d->p, V->d->p, O->d->p, dO->d->p, Stats->d->p, dQ->d->p, dK->d->p, dV->d->p, &scale, Sink->d->p, dSink->d->p};
  Desc vp = make_desc(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
  void* ws = nullptr;
  set_attr(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, static_cast<int64_t>(ptrs.size()), ptrs.data());
  set_attr(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, static_cast<int64_t>(uids.size()), uids.data());
  set_attr(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ws);
  cudnnBackendFinalize(vp);
  const cudnnStatus_t st = cudnnBackendExecute(handle, plan, vp);
  cudaDeviceSynchronize();
  expect(c.name + " runs", st == CUDNN_STATUS_SUCCESS);
  if (st == CUDNN_STATUS_SUCCESS) {
    Ref fref = ref;
    fref.O = O->v;
    const Grads gr = reference_bwd(c, Q->v, K->v, V->v, fref, dO->v, none, nullptr, &Sinkv);
    const double gtol = 4e-3 * 4;
    const std::vector<std::pair<const char*, std::pair<const Buf*, const std::vector<double>*>>> checks{
        {"dQ", {dQ.get(), &gr.dQ}}, {"dK", {dK.get(), &gr.dK}}, {"dV", {dV.get(), &gr.dV}}, {"dSink", {dSink.get(), &gr.dSink}}};
    for (const auto& [n, bw] : checks) {
      const double er = err_of(read_buf(*bw.first), *bw.second);
      expect(c.name + ": " + n, er < gtol, er);
    }
  }
  for (Desc d : keep) cudnnBackendDestroyDescriptor(d);
  for (Desc d : {op, graph, plan, vp}) if (d) cudnnBackendDestroyDescriptor(d);
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
  add("half, unified, sink", [](Cfg& c) { c.impl = I::UNIFIED; c.sink = true; c.causal = true; });
  add("half, composite, sink", [](Cfg& c) { c.impl = I::COMPOSITE; c.sink = true; c.causal = true; });
  add("half, unified, cumulative sequence lengths", [](Cfg& c) {
    c.impl = I::UNIFIED; c.padding = true; c.cu_seq = 1; c.backward = false; c.causal = true;
  });
  add("half, unified, cumulative query lengths, plain K/V lengths", [](Cfg& c) {
    c.impl = I::UNIFIED; c.padding = true; c.cu_seq = 2; c.backward = false;
  });
  add("half, unified, plain query lengths, cumulative K/V lengths", [](Cfg& c) {
    c.impl = I::UNIFIED; c.padding = true; c.cu_seq = 3; c.backward = false; c.sq = 8; c.skv = 24;
  });
  add("half, unified, cumulative sequence lengths, ragged", [](Cfg& c) {
    c.impl = I::UNIFIED; c.padding = c.ragged = true; c.cu_seq = 1; c.backward = false; c.causal = true;
  });
  add("half, auto, rotary embeddings", [](Cfg& c) { c.impl = I::AUTO; c.rope = true; c.d = c.dv = 64; c.causal = true; });
  add("bfloat16, auto, rotary embeddings, rotated width 32, scaled", [](Cfg& c) {
    c.io = fe::DataType_t::BFLOAT16; c.impl = I::AUTO; c.rope = true; c.rope_dim = 32; c.rope_scale = 0.5f; c.d = c.dv = 64;
  });
  for (const Cfg& c : cfgs) {
    try {
      run(c, handle);
    } catch (const std::exception& ex) {
      if (kOnSim) expect(c.name + " (exception: " + ex.what() + ")", false);
      else std::printf("skip %s (frontend: %s)\n", c.name.c_str(), ex.what());
    }
  }
  raw_sdpa_backward(handle);
  cudnnDestroy(handle);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
