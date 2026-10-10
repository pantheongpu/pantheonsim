// cuDNN graph operations beyond attention, built as cudnn-frontend builds them:
//
//   - the mixture-of-experts grouped matmul, forward, in its three routing modes
//     (tokens already grouped by expert, gathered through an index, scattered
//     back through an index and a top-k position), in half, bfloat16 and
//     float, with expert groups given as first-token offsets;
//   - graphs cuDNN's engines on an RTX 3060 (sm_86, cuDNN 9.27) do not run,
//     which this library refuses as the card does: the graph builds, the
//     heuristics offer no engine, and cudnn-frontend's create_execution_plans
//     fails with "No valid engine configs" (its HEURISTIC_QUERY_FAILED). They
//     are rotary embeddings outside a matrix-multiplication graph, the MoE
//     backward pass, attention with a block mask, nearest and bilinear
//     resampling outside its one documented configuration; FP8 attention is
//     refused by cudnn-frontend itself below compute capability 9.
//   - bilinear upsampling by 2, the one resampling cuDNN has an engine for
//     (NHWC, float, window 2, strides 1/2, pre-padding 1/2, post-padding 1):
//     checked against its formula, and the configurations around it that have
//     no engine, or an engine whose plan cannot be built (half data).
//
// Each result is checked against a reference computed here from the same
// rounded inputs. The same program runs against NVIDIA's libcudnn.so.9: where
// its heuristics offer no engine for a configuration on that GPU, the check
// says so and is skipped; the refusals above are checked on both.
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
using TA = fe::graph::Tensor_attributes;

struct Dev {
  void* p = nullptr;
  size_t n;
  explicit Dev(size_t bytes) : n(bytes ? bytes : 1) { cudaMalloc(&p, n); cudaMemset(p, 0, n); }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
};

// ---- element types ---------------------------------------------------------------

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
// A float as a TF32 operand: the low 13 mantissa bits rounded away to nearest, ties away from zero.
static double tf32(double v) {
  float f = static_cast<float>(v);
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7f800000u) == 0x7f800000u) return f;
  x = (x + 0x1000u) & 0xffffe000u;
  std::memcpy(&f, &x, 4);
  return f;
}

enum class Ty { Half, Bf16, Float };
static fe::DataType_t fe_type(Ty t) { return t == Ty::Half ? fe::DataType_t::HALF : t == Ty::Bf16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::FLOAT; }
static size_t esize(Ty t) { return t == Ty::Float ? 4 : 2; }
static double round_to(Ty t, double x) {
  return t == Ty::Half ? h2f(f2h(static_cast<float>(x))) : t == Ty::Bf16 ? bf2f(f2bf(static_cast<float>(x))) : static_cast<float>(x);
}
static void encode(Ty t, double x, uint8_t* at) {
  if (t == Ty::Half) { const uint16_t h = f2h(static_cast<float>(x)); std::memcpy(at, &h, 2); }
  else if (t == Ty::Bf16) { const uint16_t h = f2bf(static_cast<float>(x)); std::memcpy(at, &h, 2); }
  else { const float f = static_cast<float>(x); std::memcpy(at, &f, 4); }
}
static double decode(Ty t, const uint8_t* at) {
  if (t == Ty::Float) { float f; std::memcpy(&f, at, 4); return f; }
  uint16_t h;
  std::memcpy(&h, at, 2);
  return t == Ty::Half ? h2f(h) : bf2f(h);
}
static std::vector<uint8_t> pack(Ty t, const std::vector<double>& v) {
  std::vector<uint8_t> b(v.size() * esize(t));
  for (size_t i = 0; i < v.size(); ++i) encode(t, v[i], b.data() + i * esize(t));
  return b;
}
static std::vector<double> unpack(Ty t, const std::vector<uint8_t>& b) {
  std::vector<double> v(b.size() / esize(t));
  for (size_t i = 0; i < v.size(); ++i) v[i] = decode(t, b.data() + i * esize(t));
  return v;
}
static std::vector<double> randoms(size_t n, unsigned seed, double lo = -1.0, double hi = 1.0) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<double> u(lo, hi);
  std::vector<double> v(n);
  for (double& x : v) x = u(g);
  return v;
}

// ---- building and running a graph ----------------------------------------------------

enum class Stage { Ran, NoEngine, Failed };
struct Outcome {
  Stage stage = Stage::Failed;
  fe::error_code_t code = fe::error_code_t::OK;
  std::string at, message;
};
// Builds the graph in cudnn-frontend's stages; a graph that does not plan reports where and with which code.
static Outcome plan(fe::graph::Graph& g, cudnnHandle_t h) {
  Outcome o;
  auto step = [&](const char* at, fe::error_t st) {
    if (st.is_good()) return true;
    o.at = at, o.code = st.get_code(), o.message = st.get_message();
    return false;
  };
  if (step("validate", g.validate()) && step("build_operation_graph", g.build_operation_graph(h)) &&
      step("create_execution_plans", g.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK})) &&
      step("check_support", g.check_support()) && step("build_plans", g.build_plans()))
    o.stage = Stage::Ran;
  else
    o.stage = o.at == "create_execution_plans" || o.at == "check_support" || o.at == "build_plans" ? Stage::NoEngine : Stage::Failed;
  return o;
}
using Var = std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*>;
static bool run_plan(fe::graph::Graph& g, cudnnHandle_t h, Var& v) {
  int64_t ws = 0;
  if (!g.get_workspace_size(ws).is_good()) return false;
  Dev work(static_cast<size_t>(ws));
  auto st = g.execute(h, v, work.p);
  cudaDeviceSynchronize();
  if (!st.is_good()) std::printf("     execute: %s\n", st.get_message().c_str());
  return st.is_good();
}

// A graph this library refuses as the card does: the engine search finds nothing, with cudnn-frontend's
// own error for it, on the card and here alike.
static void expect_no_engine(const std::string& what, fe::graph::Graph& g, cudnnHandle_t h) {
  const Outcome o = plan(g, h);
  const bool right = o.stage == Stage::NoEngine && o.at == "create_execution_plans" &&
                     o.code == fe::error_code_t::HEURISTIC_QUERY_FAILED && o.message.rfind("No valid engine configs for", 0) == 0;
  if (!right) std::printf("     %s: %s at %s (code %d): %s\n", what.c_str(), o.stage == Stage::Ran ? "planned" : "refused", o.at.c_str(), static_cast<int>(o.code), o.message.substr(0, 120).c_str());
  expect(what + ": no engine (create_execution_plans fails with \"No valid engine configs\")", right);
}

// ---- the MoE grouped matmul ------------------------------------------------------------

struct Moe {
  std::string name;
  Ty ty = Ty::Half;
  bool float_math = true;
  int mode = 0;  // 0 none, 1 gather, 2 scatter
  int64_t E = 3, K = 16, N = 8, M = 12, S = 12, top_k = 1, groups_per_batch = 3;
  std::vector<int32_t> offsets;  // B * E first-token offsets
  std::vector<int32_t> index, ks;
};

static void moe_case(const Moe& c, cudnnHandle_t h) {
  const fe::DataType_t T = fe_type(c.ty), CT = c.float_math ? fe::DataType_t::FLOAT : T;
  const int64_t rows_in = c.mode == 1 ? c.S : c.M;
  fe::graph::Graph g;
  g.set_intermediate_data_type(CT).set_compute_data_type(CT);
  auto t = [&](const char* n, std::vector<int64_t> d, std::vector<int64_t> s, fe::DataType_t ty, int uid) {
    return g.tensor(TA().set_name(n).set_dim(d).set_stride(s).set_data_type(ty).set_uid(uid));
  };
  auto tok = t("token", {1, rows_in, c.K}, {rows_in * c.K, c.K, 1}, T, 1);
  auto w = t("weight", {c.E, c.K, c.N}, {c.K * c.N, 1, c.K}, T, 2);  // each expert's [K, N] column-major
  auto off = t("first_token_offset", {static_cast<int64_t>(c.offsets.size()), 1, 1}, {1, 1, 1}, fe::DataType_t::INT32, 3);
  std::shared_ptr<TA> idx, kk;
  if (c.mode >= 1) idx = t("token_index", {1, c.M, 1}, {c.M, 1, 1}, fe::DataType_t::INT32, 5);
  if (c.mode == 2) kk = t("token_ks", {1, c.M, 1}, {c.M, 1, 1}, fe::DataType_t::INT32, 6);
  auto a = fe::graph::Moe_grouped_matmul_attributes()
               .set_name("moe")
               .set_mode(c.mode == 0 ? fe::MoeGroupedMatmulMode_t::NONE : c.mode == 1 ? fe::MoeGroupedMatmulMode_t::GATHER : fe::MoeGroupedMatmulMode_t::SCATTER)
               .set_compute_data_type(CT)
               .set_top_k(static_cast<int32_t>(c.top_k));
  auto y = g.moe_grouped_matmul(tok, w, off, idx, kk, a);
  y->set_output(true).set_data_type(T).set_uid(4);
  if (c.ty == Ty::Bf16 && !c.float_math) {
    // The card has no engine for a bfloat16 product accumulated in bfloat16.
    expect_no_engine(c.name, g, h);
    return;
  }
  const Outcome o = plan(g, h);
  if (o.stage != Stage::Ran) {
    if (!kOnSim && o.stage == Stage::NoEngine) {
      std::printf("skip %s (no engine offered on this GPU)\n", c.name.c_str());
      return;
    }
    std::printf("     %s: %s: %s\n", c.name.c_str(), o.at.c_str(), o.message.substr(0, 160).c_str());
    expect(c.name + " plans", false);
    return;
  }
  // Data: the operands as the type holds them; weights laid out per expert [K, N] column-major.
  std::vector<double> tokv = randoms(static_cast<size_t>(rows_in * c.K), 11), wv = randoms(static_cast<size_t>(c.E * c.K * c.N), 12);
  for (double& x : tokv) x = round_to(c.ty, x);
  for (double& x : wv) x = round_to(c.ty, x);
  std::vector<double> wmem(wv.size());
  for (int64_t e = 0; e < c.E; ++e)
    for (int64_t k = 0; k < c.K; ++k)
      for (int64_t n = 0; n < c.N; ++n) wmem[static_cast<size_t>(e * c.K * c.N + k + n * c.K)] = wv[static_cast<size_t>((e * c.K + k) * c.N + n)];
  const std::vector<uint8_t> btok = pack(c.ty, tokv), bw = pack(c.ty, wmem);
  Dev dtok(btok.size()), dw(bw.size()), doff(c.offsets.size() * 4), dout(static_cast<size_t>(c.M * c.N) * esize(c.ty)),
      didx(static_cast<size_t>(c.M) * 4), dks(static_cast<size_t>(c.M) * 4);
  cudaMemcpy(dtok.p, btok.data(), btok.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dw.p, bw.data(), bw.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(doff.p, c.offsets.data(), c.offsets.size() * 4, cudaMemcpyHostToDevice);
  if (c.mode >= 1) cudaMemcpy(didx.p, c.index.data(), c.index.size() * 4, cudaMemcpyHostToDevice);
  if (c.mode == 2) cudaMemcpy(dks.p, c.ks.data(), c.ks.size() * 4, cudaMemcpyHostToDevice);
  // The output starts as a sentinel, so that rows nothing writes show.
  const double kSentinel = 5.0;
  const std::vector<uint8_t> bsent = pack(c.ty, std::vector<double>(static_cast<size_t>(c.M * c.N), kSentinel));
  cudaMemcpy(dout.p, bsent.data(), bsent.size(), cudaMemcpyHostToDevice);
  Var v{{1, dtok.p}, {2, dw.p}, {3, doff.p}, {4, dout.p}};
  if (c.mode >= 1) v[5] = didx.p;
  if (c.mode == 2) v[6] = dks.p;
  if (!run_plan(g, h, v)) {
    expect(c.name + " runs", false);
    return;
  }
  std::vector<uint8_t> raw(bsent.size());
  cudaMemcpy(raw.data(), dout.p, raw.size(), cudaMemcpyDeviceToHost);
  const std::vector<double> got = unpack(c.ty, raw);
  // The reference: groups g = 0..B*E-1 own rows from their offset (the first from 0) to the next one's
  // (the last to the end), never overlapping what an earlier group took; group g uses expert g % E.
  std::vector<double> want(static_cast<size_t>(c.M * c.N), kSentinel);
  const int64_t routed = c.mode == 1 ? c.M : rows_in, groups = static_cast<int64_t>(c.offsets.size());
  auto clampi = [&](int64_t x) { return std::max<int64_t>(0, std::min<int64_t>(routed, x)); };
  int64_t cursor = 0;
  for (int64_t gr = 0; gr < groups; ++gr) {
    const int64_t start = std::max(cursor, gr == 0 ? int64_t{0} : clampi(c.offsets[static_cast<size_t>(gr)]));
    const int64_t end = gr + 1 < groups ? clampi(c.offsets[static_cast<size_t>(gr + 1)]) : routed;
    if (end <= start) continue;
    cursor = end;
    for (int64_t i = start; i < end; ++i) {
      const int64_t src = c.mode == 1 ? c.index[static_cast<size_t>(i)] : i;
      const int64_t dst = c.mode == 2 ? static_cast<int64_t>(c.index[static_cast<size_t>(i)]) * c.top_k + c.ks[static_cast<size_t>(i)] : i;
      if (dst < 0 || dst >= c.M) continue;
      for (int64_t n = 0; n < c.N; ++n) {
        double acc = 0;
        for (int64_t k = 0; k < c.K; ++k) {
          const double a2 = tokv[static_cast<size_t>(src * c.K + k)], b2 = wv[static_cast<size_t>(((gr % c.E) * c.K + k) * c.N + n)];
          acc += c.ty == Ty::Float ? tf32(a2) * tf32(b2) : a2 * b2;
        }
        want[static_cast<size_t>(dst * c.N + n)] = round_to(c.ty, acc);
      }
    }
  }
  double err = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::fabs(got[i] - want[i]) / (1.0 + std::fabs(want[i]));
    err = std::max(err, std::isnan(d) ? INFINITY : d);
  }
  // fp16 and bf16 results differ by an output rounding at most (the sums are exact in float to a few ulps).
  const double tol = c.ty == Ty::Bf16 ? 1.6e-2 : c.ty == Ty::Half ? 2e-3 : 2e-5;
  expect(c.name, err < tol, err);
}

static void moe_cases(cudnnHandle_t h) {
  std::vector<Moe> cs;
  auto add = [&](const char* name, auto f) {
    Moe c;
    c.name = std::string("MoE grouped matmul, ") + name;
    f(c);
    cs.push_back(c);
  };
  for (Ty ty : {Ty::Half, Ty::Bf16, Ty::Float}) {
    const char* tn = ty == Ty::Half ? "half" : ty == Ty::Bf16 ? "bfloat16" : "float";
    add((std::string(tn) + ", tokens grouped by expert").c_str(), [&](Moe& c) { c.ty = ty; c.offsets = {0, 3, 8}; });
    add((std::string(tn) + ", gather").c_str(), [&](Moe& c) {
      c.ty = ty; c.mode = 1; c.S = 6; c.M = 12; c.offsets = {0, 4, 8};
      c.index = {5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0};
    });
    add((std::string(tn) + ", scatter, top-2").c_str(), [&](Moe& c) {
      c.ty = ty; c.mode = 2; c.top_k = 2; c.M = 16; c.offsets = {0, 5, 13};
      // Slots naming a token outside [0, S) land outside the output: dropped.
      c.index = {1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12};
      c.ks = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    });
  }
  add("half, a second batch of groups (B = 2)", [](Moe& c) { c.M = 16; c.offsets = {0, 3, 8, 10, 12, 14}; });
  add("float, an empty expert and rows before the first offset", [](Moe& c) { c.ty = Ty::Float; c.offsets = {2, 5, 5}; });
  add("bfloat16 accumulated in bfloat16", [](Moe& c) { c.ty = Ty::Bf16; c.float_math = false; c.offsets = {0, 3, 8}; });
  for (const Moe& c : cs) {
    try {
      moe_case(c, h);
    } catch (const std::exception& ex) {
      if (kOnSim) expect(c.name + " (exception: " + ex.what() + ")", false);
      else std::printf("skip %s (frontend: %s)\n", c.name.c_str(), ex.what());
    }
  }
}

// ---- bilinear upsampling ---------------------------------------------------------------------

// cuDNN documents one configuration of bilinear resampling for its runtime-fusion engines: NHWC, float, window 2, strides
// 1/2, pre-padding 1/2, post-padding 1, so that the output is twice the input; nearest has no configuration at all.
// Measured on an RTX 3060 (cuDNN 9.27): output i samples the input at s = i * stride - pre + window / 2 - 1/2 in each
// dimension (that is i / 2 here), clamped to the input, between the pixels either side of s by distance; zero or
// edge-value padding give the same result. Half and bfloat16 data get an engine whose plan cannot be built; every
// other parameter set tried has no engine.
static void bilinear_cases(cudnnHandle_t h) {
  const cudnnFraction_t half_{1, 2}, one{1, 1}, two{2, 1}, zero{0, 1}, third{1, 3};
  const fe::DataType_t fl = fe::DataType_t::FLOAT;
  struct Cfg {
    std::string name;
    int64_t N, C, H, W;
    fe::PaddingMode_t pm;
  };
  auto build = [&](fe::graph::Graph& g, const Cfg& c, fe::DataType_t ty, bool nhwc, std::vector<cudnnFraction_t> win,
                   std::vector<cudnnFraction_t> str, std::vector<cudnnFraction_t> pre, std::vector<cudnnFraction_t> post,
                   int64_t OH, int64_t OW, fe::ResampleMode_t mode) {
    g.set_io_data_type(ty).set_intermediate_data_type(fl).set_compute_data_type(fl);
    const int64_t C = c.C, H = c.H, W = c.W;
    std::vector<int64_t> xs = nhwc ? std::vector<int64_t>{C * H * W, 1, C * W, C} : std::vector<int64_t>{C * H * W, H * W, W, 1};
    std::vector<int64_t> ys = nhwc ? std::vector<int64_t>{C * OH * OW, 1, C * OW, C} : std::vector<int64_t>{C * OH * OW, OH * OW, OW, 1};
    auto x = g.tensor(TA().set_name("x").set_dim({c.N, C, H, W}).set_stride(xs).set_uid(1));
    auto [y, idx] = g.resample(x, fe::graph::Resample_attributes().set_generate_index(false).set_resampling_mode(mode)
                                      .set_padding_mode(c.pm).set_window(win).set_stride(str).set_pre_padding(pre).set_post_padding(post));
    (void)idx;
    y->set_output(true).set_uid(2).set_dim({c.N, C, OH, OW}).set_stride(ys);
  };
  const std::vector<Cfg> cases = {{"batch 2, 8 channels, 5 x 7", 2, 8, 5, 7, fe::PaddingMode_t::EDGE_VAL_PAD},
                                  {"one pixel", 1, 4, 1, 1, fe::PaddingMode_t::EDGE_VAL_PAD},
                                  {"one channel, 4 x 4", 1, 1, 4, 4, fe::PaddingMode_t::EDGE_VAL_PAD},
                                  {"zero padding mode, 4 x 6", 1, 3, 4, 6, fe::PaddingMode_t::ZERO_PAD}};
  for (const Cfg& c : cases) {
    fe::graph::Graph g;
    build(g, c, fl, true, {two, two}, {half_, half_}, {half_, half_}, {one, one}, 2 * c.H, 2 * c.W, fe::ResampleMode_t::BILINEAR);
    const Outcome o = plan(g, h);
    const std::string name = "bilinear upsampling by 2, " + c.name;
    if (o.stage != Stage::Ran) {
      std::printf("     %s: %s: %s\n", name.c_str(), o.at.c_str(), o.message.substr(0, 120).c_str());
      expect(name + " plans", false);
      continue;
    }
    const int64_t OH = 2 * c.H, OW = 2 * c.W;
    std::vector<double> xv(static_cast<size_t>(c.N * c.C * c.H * c.W));
    for (int64_t n = 0; n < c.N; ++n)
      for (int64_t ch = 0; ch < c.C; ++ch)
        for (int64_t i = 0; i < c.H; ++i)
          for (int64_t j = 0; j < c.W; ++j)
            xv[static_cast<size_t>(((n * c.H + i) * c.W + j) * c.C + ch)] =
                static_cast<double>(n * 100 + ch * 7 + i * 10 + j) + 0.125 * static_cast<double>((i * 7 + j * 3) % 5);
    Dev dx(xv.size() * 4), dy(static_cast<size_t>(c.N * c.C * OH * OW) * 4);
    const std::vector<uint8_t> bx = pack(Ty::Float, xv);
    cudaMemcpy(dx.p, bx.data(), bx.size(), cudaMemcpyHostToDevice);
    Var v{{1, dx.p}, {2, dy.p}};
    if (!run_plan(g, h, v)) {
      expect(name + " runs", false);
      continue;
    }
    std::vector<uint8_t> raw(dy.n);
    cudaMemcpy(raw.data(), dy.p, raw.size(), cudaMemcpyDeviceToHost);
    const std::vector<double> got = unpack(Ty::Float, raw);
    auto at = [&](int64_t n, int64_t ch, int64_t i, int64_t j) { return xv[static_cast<size_t>(((n * c.H + i) * c.W + j) * c.C + ch)]; };
    double err = 0;
    for (int64_t n = 0; n < c.N; ++n)
      for (int64_t ch = 0; ch < c.C; ++ch)
        for (int64_t i = 0; i < OH; ++i)
          for (int64_t j = 0; j < OW; ++j) {
            // s = i / 2 here: i * (1/2) - 1/2 + 2/2 - 1/2, clamped to [0, size - 1].
            const double sy = std::min(std::max(i * 0.5, 0.0), (double)(c.H - 1)), sx = std::min(std::max(j * 0.5, 0.0), (double)(c.W - 1));
            const int64_t y0 = (int64_t)std::floor(sy), x0 = (int64_t)std::floor(sx), y1 = std::min(y0 + 1, c.H - 1), x1 = std::min(x0 + 1, c.W - 1);
            const double fy = sy - (double)y0, fx = sx - (double)x0;
            const double want = (1 - fy) * ((1 - fx) * at(n, ch, y0, x0) + fx * at(n, ch, y0, x1)) +
                                fy * ((1 - fx) * at(n, ch, y1, x0) + fx * at(n, ch, y1, x1));
            err = std::max(err, std::fabs(got[static_cast<size_t>(((n * OH + i) * OW + j) * c.C + ch)] - want));
          }
    expect(name, err < 1e-5, err);
  }
  // What has no engine, and what has one whose plan cannot be built.
  const Cfg base{"", 1, 8, 4, 4, fe::PaddingMode_t::EDGE_VAL_PAD};
  {
    fe::graph::Graph g;
    build(g, base, fl, false, {two, two}, {half_, half_}, {half_, half_}, {one, one}, 8, 8, fe::ResampleMode_t::BILINEAR);
    expect_no_engine("bilinear upsampling in NCHW", g, h);
  }
  {
    fe::graph::Graph g;
    build(g, base, fl, true, {two, two}, {half_, half_}, {half_, half_}, {one, one}, 8, 8, fe::ResampleMode_t::NEAREST);
    expect_no_engine("nearest upsampling, with the bilinear parameters", g, h);
  }
  {
    fe::graph::Graph g;
    build(g, base, fl, true, {two, two}, {half_, half_}, {half_, half_}, {half_, half_}, 7, 7, fe::ResampleMode_t::BILINEAR);
    expect_no_engine("bilinear upsampling with post-padding 1/2", g, h);
  }
  {
    fe::graph::Graph g;
    build(g, base, fl, true, {two, two}, {half_, one}, {half_, zero}, {one, zero}, 8, 3, fe::ResampleMode_t::BILINEAR);
    expect_no_engine("bilinear upsampling of the height alone", g, h);
  }
  for (fe::DataType_t ty : {fe::DataType_t::HALF, fe::DataType_t::BFLOAT16}) {
    fe::graph::Graph g;
    build(g, base, ty, true, {two, two}, {half_, half_}, {half_, half_}, {one, one}, 8, 8, fe::ResampleMode_t::BILINEAR);
    const Outcome o = plan(g, h);
    expect(std::string("bilinear upsampling in ") + (ty == fe::DataType_t::HALF ? "half" : "bfloat16") +
               ": an engine is offered but its plan cannot be built",
           o.stage == Stage::NoEngine && o.at == "build_plans", 0);
  }
  (void)third;
}

// ---- graphs without an engine ----------------------------------------------------------------

static void refusals(cudnnHandle_t h) {
  const int64_t B = 1, H = 2, S = 16, D = 64;
  const std::vector<int64_t> st{H * S * D, S * D, D, 1};
  auto tensor = [&](fe::graph::Graph& g, const char* n, int uid, fe::DataType_t t) {
    return g.tensor(TA().set_name(n).set_dim({B, H, S, D}).set_stride(st).set_data_type(t).set_uid(uid));
  };
  const fe::DataType_t bf = fe::DataType_t::BFLOAT16, fl = fe::DataType_t::FLOAT;
  {  // Rotary embeddings alone: cuDNN runs them only inside a graph with a matrix multiplication or attention.
    fe::graph::Graph g;
    g.set_intermediate_data_type(fl).set_compute_data_type(fl);
    auto x = tensor(g, "x", 1, bf);
    auto f = g.tensor(TA().set_name("freqs").set_dim({S, 1, 1, D}).set_stride({D, D, D, 1}).set_data_type(fl).set_uid(2));
    auto y = g.rope(x, f, fe::graph::RoPE_attributes().set_name("rope"));
    y->set_output(true).set_data_type(bf).set_dim({B, H, S, D}).set_stride(st).set_uid(3);
    expect_no_engine("rotary embedding alone", g, h);
  }
  {
    fe::graph::Graph g;
    g.set_intermediate_data_type(fl).set_compute_data_type(fl);
    auto x = tensor(g, "dy", 1, bf);
    auto f = g.tensor(TA().set_name("freqs").set_dim({S, 1, 1, D}).set_stride({D, D, D, 1}).set_data_type(fl).set_uid(2));
    auto y = g.rope_backward(x, f, fe::graph::RoPE_backward_attributes().set_name("rope_bwd"));
    y->set_output(true).set_data_type(bf).set_dim({B, H, S, D}).set_stride(st).set_uid(3);
    expect_no_engine("rotary embedding backward alone", g, h);
  }
  {  // The MoE backward pass.
    fe::graph::Graph g;
    g.set_intermediate_data_type(fl).set_compute_data_type(fl);
    const int64_t E = 3, T = 64, K = 32, N = 16;
    auto dout = g.tensor(TA().set_name("dout").set_dim({1, T, N}).set_stride({T * N, N, 1}).set_data_type(bf).set_uid(1));
    auto tok = g.tensor(TA().set_name("tok").set_dim({1, T, K}).set_stride({T * K, K, 1}).set_data_type(bf).set_uid(2));
    auto off = g.tensor(TA().set_name("off").set_dim({E, 1, 1}).set_stride({1, 1, 1}).set_data_type(fe::DataType_t::INT32).set_uid(3));
    auto dw = g.moe_grouped_matmul_bwd(dout, tok, off, fe::graph::Moe_grouped_matmul_bwd_attributes().set_name("moe_bwd").set_compute_data_type(fl));
    dw->set_output(true).set_data_type(bf).set_uid(4);
    expect_no_engine("MoE grouped matmul backward", g, h);
  }
  {  // Attention with a block mask: cuDNN's engines for it are Blackwell's and Rubin's.
    fe::graph::Graph g;
    g.set_io_data_type(bf).set_intermediate_data_type(fl).set_compute_data_type(fl);
    const int64_t s = 256, d = 64;
    auto q = g.tensor(TA().set_name("Q").set_dim({B, H, s, d}).set_stride({H * s * d, s * d, d, 1}).set_uid(1));
    auto k = g.tensor(TA().set_name("K").set_dim({B, H, s, d}).set_stride({H * s * d, s * d, d, 1}).set_uid(2));
    auto v = g.tensor(TA().set_name("V").set_dim({B, H, s, d}).set_stride({H * s * d, s * d, d, 1}).set_uid(3));
    auto mask = g.tensor(TA().set_name("block_mask").set_dim({B, H, 2, 1}).set_stride({H * 2, 2, 1, 1}).set_data_type(fe::DataType_t::UINT8).set_uid(9));
    auto [o, stats] = g.sdpa(q, k, v,
                             fe::graph::SDPA_attributes().set_name("sdpa").set_generate_stats(true).set_attn_scale(0.125f).set_block_mask(mask)
                                 .set_implementation(fe::AttentionImplementation_t::UNIFIED));
    o->set_output(true).set_dim({B, H, s, d}).set_stride({H * s * d, s * d, d, 1}).set_uid(4);
    stats->set_output(true).set_data_type(fl).set_uid(5);
    expect_no_engine("attention with a block mask", g, h);
  }
  // Nearest and bilinear resampling (window 2, stride 1): no engine either.
  for (bool bilinear : {false, true}) {
    fe::graph::Graph g;
    g.set_io_data_type(fl).set_compute_data_type(fl);
    auto x = g.tensor(TA().set_name("x").set_uid(1).set_dim({1, 2, 4, 4}).set_stride({32, 16, 4, 1}));
    auto [y, idx] = g.resample(x, fe::graph::Resample_attributes()
                                       .set_generate_index(false)
                                       .set_resampling_mode(bilinear ? fe::ResampleMode_t::BILINEAR : fe::ResampleMode_t::NEAREST)
                                       .set_padding_mode(fe::PaddingMode_t::EDGE_VAL_PAD)
                                       .set_window({2, 2}).set_stride({1, 1}).set_pre_padding({0, 0}).set_post_padding({0, 0}));
    y->set_output(true).set_uid(2);
    expect_no_engine(bilinear ? "bilinear resampling" : "nearest resampling", g, h);
  }
  // FP8 attention: cudnn-frontend itself refuses it below compute capability 9 (before the library is asked).
  int major = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  if (major < 9) {
    fe::graph::Graph g;
    g.set_io_data_type(fe::DataType_t::FP8_E4M3).set_intermediate_data_type(fl).set_compute_data_type(fl);
    const int64_t s = 128, d = 64;
    auto q = g.tensor(TA().set_name("Q").set_dim({B, H, s, d}).set_stride({H * s * d, s * d, d, 1}).set_uid(1));
    auto k = g.tensor(TA().set_name("K").set_dim({B, H, s, d}).set_stride({H * s * d, s * d, d, 1}).set_uid(2));
    auto v = g.tensor(TA().set_name("V").set_dim({B, H, s, d}).set_stride({H * s * d, s * d, d, 1}).set_uid(3));
    auto sc = [&](const char* n, int uid) {
      return g.tensor(TA().set_name(n).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fl).set_uid(uid));
    };
    auto r = g.sdpa_fp8(q, k, v, sc("dq", 11), sc("dk", 12), sc("dv", 13), sc("ds", 14), sc("sS", 15), sc("sO", 16),
                        fe::graph::SDPA_fp8_attributes().set_name("sdpa_fp8").set_generate_stats(false).set_attn_scale(0.125f));
    r[0]->set_output(true).set_dim({B, H, s, d}).set_stride({H * s * d, s * d, d, 1}).set_uid(4);
    r[2]->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fl).set_uid(5);
    r[3]->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fl).set_uid(6);
    const Outcome o = plan(g, h);
    expect("FP8 attention below compute capability 9: refused when the graph is validated (GRAPH_NOT_SUPPORTED)",
           o.stage == Stage::Failed && o.at == "validate" && o.code == fe::error_code_t::GRAPH_NOT_SUPPORTED, static_cast<double>(static_cast<int>(o.code)));
  } else {
    std::printf("skip FP8 attention's refusal (this GPU has compute capability %d)\n", major);
  }
}

int main() {
  cudnnHandle_t handle;
  if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
    std::printf("SKIP: cudnnCreate failed\n");
    return 0;
  }
  std::printf("cuDNN %zu, cudnn-frontend %d.%d.%d\n", cudnnGetVersion(), CUDNN_FRONTEND_MAJOR_VERSION,
              CUDNN_FRONTEND_MINOR_VERSION, CUDNN_FRONTEND_PATCH_VERSION);
  moe_cases(handle);
  try {
    bilinear_cases(handle);
    refusals(handle);
  } catch (const std::exception& ex) {
    expect(std::string("refusals (exception: ") + ex.what() + ")", false);
  }
  cudnnDestroy(handle);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
