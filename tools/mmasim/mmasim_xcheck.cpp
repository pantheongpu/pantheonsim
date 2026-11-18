// Cross-check of the simulator's matrix-instruction sums against MMA-Sim (Xie et al., arXiv 2511.10909, github.com/microsoft/MMA-Sim, MIT).
//
// The same vectors go to two sides and the outputs are compared bit for bit, per architecture and per instruction form:
//   ours    NVIDIA: tc86::mma_wide (include/vgpu/exec/tc_ampere.hpp) with the rule the simulator would use for that architecture
//                   (Ampere, Hopper) or, where it has none (Volta, and Turing's m8n8k4), the same rule family with its numbers;
//           AMD:    the simulator's default (a double sum, rounded once: mfma_dot of amd/src/mfma_math.hpp) and, for gfx90a f16 /
//                   bf16, the opt-in FTTN rule (cdna2_dot_seq32).
//   theirs  either "transcription": a line-by-line C++ transcription of MMA-Sim's arithmetic (arithmetic/fdpa.py, ftz_mul_add.py,
//           fma.py and the per-architecture tables of nv_ptx/sim.py and amd/sim.py at the pinned commit), run here; or "file": the
//           outputs of the real MMA-Sim Python package, made by tools/mmasim/mmasim_ref.py from vectors this program exports.
//           The transcription is NOT MMA-Sim: it is this file's reading of it. Only "file" results are MMA-Sim's own.
//
//   mmasim_xcheck run     [--n N] [--seed S] [--filter text] [--show K]   ours against the transcription (needs no Python)
//   mmasim_xcheck export  <dir> [--n N] [--seed S] [--filter text]        write the vectors: <dir>/manifest.tsv and <dir>/<id>.in
//   mmasim_xcheck compare <dir> [--show K] [--filter text]                ours against <dir>/<id>.out (written by mmasim_ref.py)
//
// Build (one file; from the repo root):
//   g++ -std=c++20 -O2 -I include -I amd/src tools/mmasim/mmasim_xcheck.cpp -o mmasim_xcheck
// It links nothing: only tc86::mma_wide and the header-only AMD sums are called, never the dispatcher that reads simd state.
//
// Vector files: <id>.in is uint32 words, n vectors of [a(K), b(K), c], every value as the float32 bits of the (exactly representable)
// f16 / bf16 / tf32 / f32 value; <id>.out is n uint32 words, the result as the bits of its format (a 16-bit result in the low half).
// A NaN result is compared as "a NaN" for AMD and by its canonical bits (0x7fffffff, f16 0x7fff) for NVIDIA, as MMA-Sim writes them.
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "mfma_math.hpp"
#include "vgpu/exec/tc_ampere.hpp"

namespace tc = vgpu::exec::tc86;
using u32 = uint32_t;
using u64 = uint64_t;

// Deliberately wrong transcriptions, to show each corner is seen (--mut-theirs ezero|rne|sign|accpos): the harness's own power test.
static int g_mut_theirs = 0;   // 1 no floor under a zero accumulator (R2), 2 round instead of cut a subnormal f32 result (R1), 3 keep a negative zero (R4), 4 subnormal accumulator at its own leading bit (R3)
static float f32of(u32 u) { float f; std::memcpy(&f, &u, 4); return f; }
static u32 bitsof(float f) { u32 u; std::memcpy(&u, &f, 4); return u; }
static std::string hexs(const char* fmt, unsigned long v) { char b[32]; std::snprintf(b, sizeof b, fmt, v); return b; }

// ---------------------------------------------------------------------------------------------------------------------------
// Formats and vector generation
// ---------------------------------------------------------------------------------------------------------------------------
enum class Fmt { F16, BF16, TF32, F32 };
struct FmtInfo { int mbits, bias, maxexp; };
static FmtInfo info(Fmt f) {
  switch (f) {
    case Fmt::F16: return {10, 15, 31};
    case Fmt::BF16: return {7, 127, 255};
    case Fmt::TF32: return {10, 127, 255};
    default: return {23, 127, 255};
  }
}
static float make(Fmt f, u32 sign, u32 expf, u32 mant) {
  switch (f) {
    case Fmt::F16: return vgpu::simd::f16_to_f32(static_cast<uint16_t>((sign << 15) | (expf << 10) | mant));
    case Fmt::BF16: return f32of((sign << 31) | (expf << 23) | (mant << 16));
    case Fmt::TF32: return f32of((sign << 31) | (expf << 23) | (mant << 13));
    default: return f32of((sign << 31) | (expf << 23) | mant);
  }
}
static std::string hexval(Fmt f, float v) {
  if (f == Fmt::F16) return hexs("%04lx", vgpu::simd::f64_to_f16_bits(static_cast<double>(v)));
  if (f == Fmt::BF16) return hexs("%04lx", bitsof(v) >> 16);
  return hexs("%08lx", bitsof(v));
}

struct Rng {
  u64 s;
  explicit Rng(u64 x) : s(x) {}
  u64 next() {
    s += 0x9E3779B97F4A7C15ull;
    u64 z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  u32 below(u32 n) { return static_cast<u32>(next() % n); }
};

enum Cls { kBits, kFinite, kWindow, kTinyA, kTinyB, kTinyC, kTinyD, kTinyE };
static float elem(Rng& r, Fmt f, Cls c) {
  const FmtInfo fi = info(f);
  const u32 sign = r.below(2);
  u32 e;
  switch (c) {
    case kBits: e = r.below(fi.maxexp + 1); break;
    case kFinite: e = r.below(fi.maxexp); break;
    case kWindow: e = fi.bias - 4 + r.below(9); break;
    case kTinyA: e = f == Fmt::F16 ? r.below(11) : 40 + r.below(36); break;
    case kTinyB: e = f == Fmt::F16 ? 4 + r.below(13) : r.below(15); break;
    case kTinyC: e = f == Fmt::F16 ? r.below(6) : 48 + r.below(19); break;   // products around the smallest subnormal of the result
    case kTinyD: e = f == Fmt::F16 ? r.below(12) : 50 + r.below(23); break;
    default: e = f == Fmt::F16 ? r.below(7) : 48 + r.below(17); break;      // kTinyE
  }
  u32 m = r.below(1u << fi.mbits);
  if (r.below(4) == 0) m = 0;   // powers of two: ties and exact cancellations
  return make(f, sign, e, m);
}
static float special(Rng& r, Fmt f) {
  const FmtInfo fi = info(f);
  const u32 sign = r.below(2);
  switch (r.below(4)) {
    case 0: return make(f, sign, 0, 0);
    case 1: case 2: return make(f, sign, fi.maxexp, 0);
    default: return make(f, sign, fi.maxexp, 1 + r.below((1u << fi.mbits) - 1));
  }
}
// accumulator kinds: 0 +0, 1 -0, 2 window, 3 tiny / subnormal, 4 finite random, 5 any bit pattern
static float accum(Rng& r, Fmt f, int kind) {
  switch (kind) {
    case 0: return 0.0f;
    case 1: return -0.0f;
    case 2: return elem(r, f, kWindow);
    case 6: {   // a subnormal accumulator
      const FmtInfo fi = info(f);
      return make(f, r.below(2), 0, 1 + r.below((1u << fi.mbits) - 1));
    }
    case 3: {
      const FmtInfo fi = info(f);
      return make(f, r.below(2), r.below(f == Fmt::F16 ? 8 : 30), r.below(1u << fi.mbits));
    }
    case 4: return elem(r, f, kFinite);
    default: return elem(r, f, kBits);
  }
}

struct Vec { float a[16], b[16], c; };
static constexpr int kClasses = 11;
static Vec gen(Rng& r, Fmt f, Fmt accf, unsigned K, int vc) {
  Vec v{};
  auto fill = [&](Cls c) { for (unsigned k = 0; k < K; ++k) v.a[k] = elem(r, f, c), v.b[k] = elem(r, f, c); };
  switch (vc) {
    case 0: fill(kBits); v.c = accum(r, accf, 5); break;
    case 1: fill(kFinite); v.c = accum(r, accf, 4); break;
    case 2: fill(kWindow); v.c = accum(r, accf, r.below(2) ? 2 : 0); break;
    case 3: fill(kTinyA); v.c = accum(r, accf, r.below(3)); break;   // 0, -0, window
    case 4: fill(kTinyB); { static const int k3[] = {0, 3, 2}; v.c = accum(r, accf, k3[r.below(3)]); } break;
    case 5: {   // the second half cancels the first, maybe off by one operand's mantissa
      fill(kWindow);
      const unsigned h = K / 2;
      for (unsigned k = 0; k < h; ++k) v.a[k + h] = v.a[k], v.b[k + h] = -v.b[k];
      if (r.below(2)) v.a[r.below(K)] = elem(r, f, kWindow);
      v.c = accum(r, accf, r.below(2) ? 2 : 0);
      break;
    }
    case 6:
      fill(kWindow);
      for (unsigned i = 1 + r.below(3); i > 0; --i) (r.below(2) ? v.a : v.b)[r.below(K)] = special(r, f);
      v.c = r.below(8) == 0 ? special(r, accf) : accum(r, accf, 2);
      break;
    case 8:   // tiny products all of one sign (a positive, b negative), so the sum is negative and may cut to zero
      fill(kTinyC);
      for (unsigned k = 0; k < K; ++k) v.a[k] = std::fabs(v.a[k]), v.b[k] = -std::fabs(v.b[k]);
      { static const int k3[] = {0, 1, 6}; v.c = accum(r, accf, k3[r.below(3)]); }
      break;
    case 9: fill(kTinyD); v.c = accum(r, accf, 6); break;   // a subnormal accumulator among tiny products
    case 10: {   // two to four tiny products of either sign, nothing else, a zero accumulator: where the floor under a zero accumulator shows
      for (unsigned k = 0; k < K; ++k) v.a[k] = make(f, 0, 0, 0), v.b[k] = make(f, 0, 0, 0);
      for (unsigned i = 2 + r.below(3); i > 0; --i) {
        const unsigned k = r.below(K);
        v.a[k] = elem(r, f, kTinyE);
        v.b[k] = elem(r, f, kTinyE);
      }
      v.c = r.below(2) ? 0.0f : -0.0f;
      break;
    }
    default:
      for (unsigned k = 0; k < K; ++k) {
        static const Cls cs[] = {kWindow, kFinite, kTinyA, kTinyB};
        v.a[k] = elem(r, f, cs[r.below(4)]);
        v.b[k] = elem(r, f, cs[r.below(4)]);
      }
      v.c = accum(r, accf, r.below(6));
      break;
  }
  return v;
}

// ---------------------------------------------------------------------------------------------------------------------------
// The groups: one per (architecture, instruction)
// ---------------------------------------------------------------------------------------------------------------------------
struct NvTheirs { int F; int L; int e_zero; int esub; bool acc16; };   // MMA_T_FDPA's numbers (rho follows acc16: RNE-FP16 or RZ-FP32)
struct Group {
  std::string id, side, arch, api, instr;   // side: nv | amd
  Fmt fmt = Fmt::F16;
  bool acc16 = false;
  unsigned K = 0;
  // ours, NVIDIA
  std::string our_model;
  unsigned our_chunk = 0;
  int our_grid = 0;
  int emin = -14;
  // theirs, NVIDIA
  NvTheirs nv{};
  // AMD: theirs kind: 'e' CDNA1 exact blocks (L), 'p' CDNA2 pairs (P), 't' CDNA3 TR (L), 'f' f32 fma chain
  char amd_kind = 0;
  int amd_L = 0;
  bool amd_optin = false;   // also run our gfx90a opt-in rule
};

static const char* const kArch[] = {"Volta", "Turing", "Ampere", "Ada Lovelace", "Hopper", "Blackwell", "RTX Blackwell"};
static const int kF[] = {23, 24, 24, 24, 25, 25, 25};
static const int kLbytes[] = {8, 16, 16, 16, 32, 32, 32};
static const int kEz16[] = {-19, -20, -20, -20, -21, -21, -21};
static const int kEz32[] = {-131, -132, -132, -132, -133, -133, -133};

static void add_nv(std::vector<Group>& out, int ai, const char* api, const char* instr, Fmt fmt, bool acc16, unsigned K) {
  Group g;
  g.side = "nv";
  g.arch = kArch[ai];
  g.api = api;
  g.instr = instr;
  g.fmt = fmt;
  g.acc16 = acc16;
  g.K = K;
  g.emin = fmt == Fmt::F16 ? -14 : -126;
  const int item = fmt == Fmt::TF32 ? 4 : 2;
  const bool is_mma = g.api == "mma";
  const bool volta_form = is_mma && g.instr.rfind("m8n8k4", 0) == 0;
  // theirs: nv_ptx/sim.py
  if (is_mma) {
    g.nv = {kF[ai], kLbytes[ai] / item, acc16 ? kEz16[ai] : kEz32[ai], g.emin, acc16};
    if (volta_form) g.nv = {23, 4, acc16 ? -19 : -131, g.emin, acc16};   // Turing falls back to Volta's numbers for m8n8k4
  } else {
    g.nv = {25, 32 / item, acc16 ? -21 : -133, g.emin, acc16};   // wgmma, tcgen05.mma
  }
  // ours: the rule the simulator would use for this architecture
  const bool hopper_like = ai >= 4;
  if (volta_form || ai == 0) {
    g.our_model = "Volta numbers (chunk 4, G 24): not a shipped model";
    g.our_chunk = 4;
    g.our_grid = 24;
  } else if (hopper_like) {
    g.our_model = "Hopper (chunk 16 / tf32 8, G 26)";
    g.our_chunk = fmt == Fmt::TF32 ? tc::kHopperRule.chunk_tf32 : tc::kHopperRule.chunk;
    g.our_grid = tc::kHopperRule.grid;
  } else {
    g.our_model = "Ampere (chunk 8 / tf32 4, G 25)";
    g.our_chunk = fmt == Fmt::TF32 ? tc::kAmpereRule.chunk_tf32 : tc::kAmpereRule.chunk;
    g.our_grid = tc::kAmpereRule.grid;
  }
  if (g.our_chunk > K) g.our_chunk = K;
  g.id = std::string("nv_") + std::to_string(out.size());
  out.push_back(g);
}

static void add_amd(std::vector<Group>& out, const char* arch, const char* suffix, Fmt fmt, unsigned K, char kind, int L, bool optin) {
  Group g;
  g.side = "amd";
  g.arch = arch;
  g.api = "mfma";
  g.instr = suffix;
  g.fmt = fmt;
  g.K = K;
  g.amd_kind = kind;
  g.amd_L = L;
  g.amd_optin = optin;
  g.our_model = "double sum, rounded once (default)";
  g.id = std::string("amd_") + std::to_string(out.size());
  out.push_back(g);
}

static std::vector<Group> all_groups() {
  std::vector<Group> g;
  const struct { const char* instr; Fmt f; bool acc16; unsigned K; } sm70[] = {
      {"m8n8k4.f32.f16.f16.f32", Fmt::F16, false, 4}, {"m8n8k4.f16.f16.f16.f16", Fmt::F16, true, 4}};
  const struct { const char* instr; Fmt f; bool acc16; unsigned K; } sm75[] = {
      {"m16n8k8.f32.f16.f16.f32", Fmt::F16, false, 8}, {"m16n8k8.f16.f16.f16.f16", Fmt::F16, true, 8}};
  const struct { const char* instr; Fmt f; bool acc16; unsigned K; } sm80[] = {
      {"m16n8k16.f32.f16.f16.f32", Fmt::F16, false, 16}, {"m16n8k16.f16.f16.f16.f16", Fmt::F16, true, 16},
      {"m16n8k16.f32.bf16.bf16.f32", Fmt::BF16, false, 16}, {"m16n8k8.f32.bf16.bf16.f32", Fmt::BF16, false, 8},
      {"m16n8k8.f32.tf32.tf32.f32", Fmt::TF32, false, 8}, {"m16n8k4.f32.tf32.tf32.f32", Fmt::TF32, false, 4}};
  for (const auto& s : sm70) add_nv(g, 0, "mma", s.instr, s.f, s.acc16, s.K);
  for (const auto& s : sm70) add_nv(g, 1, "mma", s.instr, s.f, s.acc16, s.K);
  for (int ai = 1; ai <= 6; ++ai) {
    for (const auto& s : sm75) add_nv(g, ai, "mma", s.instr, s.f, s.acc16, s.K);
    if (ai >= 2)
      for (const auto& s : sm80) add_nv(g, ai, "mma", s.instr, s.f, s.acc16, s.K);
    if (ai == 4) {   // Hopper wgmma
      add_nv(g, ai, "wgmma", "m64n8k16.f32.f16.f16", Fmt::F16, false, 16);
      add_nv(g, ai, "wgmma", "m64n8k16.f16.f16.f16", Fmt::F16, true, 16);
      add_nv(g, ai, "wgmma", "m64n8k16.f32.bf16.bf16", Fmt::BF16, false, 16);
      add_nv(g, ai, "wgmma", "m64n8k8.f32.tf32.tf32", Fmt::TF32, false, 8);
    }
    if (ai == 5) {   // Blackwell tcgen05.mma
      add_nv(g, ai, "tcgen05", "m64n8k16.kind::f16.f32.f16.f16", Fmt::F16, false, 16);
      add_nv(g, ai, "tcgen05", "m64n8k16.kind::f16.f16.f16.f16", Fmt::F16, true, 16);
      add_nv(g, ai, "tcgen05", "m64n8k16.kind::f16.f32.bf16.bf16", Fmt::BF16, false, 16);
      add_nv(g, ai, "tcgen05", "m64n8k8.kind::tf32.f32.tf32.tf32", Fmt::TF32, false, 8);
    }
  }
  // AMD
  add_amd(g, "CDNA1", "f32_16x16x16f16", Fmt::F16, 16, 'e', 4, false);
  add_amd(g, "CDNA1", "f32_16x16x8bf16", Fmt::BF16, 8, 'e', 2, false);
  add_amd(g, "CDNA1", "f32_16x16x4f32", Fmt::F32, 4, 'f', 0, false);
  add_amd(g, "CDNA2", "f32_16x16x16f16", Fmt::F16, 16, 'p', 4, true);
  add_amd(g, "CDNA2", "f32_16x16x16bf16_1k", Fmt::BF16, 16, 'p', 4, true);
  add_amd(g, "CDNA2", "f32_16x16x8bf16", Fmt::BF16, 8, 'p', 2, true);
  add_amd(g, "CDNA2", "f32_16x16x4f32", Fmt::F32, 4, 'f', 0, false);
  add_amd(g, "CDNA3", "f32_16x16x16_f16", Fmt::F16, 16, 't', 8, false);
  add_amd(g, "CDNA3", "f32_16x16x16_bf16", Fmt::BF16, 16, 't', 8, false);
  add_amd(g, "CDNA3", "f32_16x16x8_xf32", Fmt::TF32, 8, 't', 4, false);
  add_amd(g, "CDNA3", "f32_16x16x4_f32", Fmt::F32, 4, 'f', 0, false);
  return g;
}

// ---------------------------------------------------------------------------------------------------------------------------
// Ours
// ---------------------------------------------------------------------------------------------------------------------------
static u32 f16_result_bits(float d) { return d != d ? 0x7fffu : vgpu::simd::f64_to_f16_bits(static_cast<double>(d)); }

static int g_mut_grid = 0, g_mut_chunk = 0;   // --mut-grid / --mut-chunk: a deliberately wrong rule, to show the harness can see a difference
static u32 ours_nv(const Group& g, const Vec& v) {
  float D = 0;
  tc::mma_wide(1, 1, g.K, std::max(1u, std::min(g.K, static_cast<unsigned>(static_cast<int>(g.our_chunk) + g_mut_chunk))), g.our_grid + g_mut_grid, g.emin, g.acc16, v.a, g.K, v.b, 1, &v.c, 1, &D, 1);
  return g.acc16 ? f16_result_bits(D) : bitsof(D);
}

static u32 ours_amd(const Group& g, const Vec& v, bool optin) {
  double a[16], b[16];
  for (unsigned k = 0; k < g.K; ++k) a[k] = v.a[k], b[k] = v.b[k];
  double c = v.c;
  if (!optin) return bitsof(static_cast<float>(vgpu::amd::mfma_dot(a, b, g.K, c)));
  const double mn = g.fmt == Fmt::F16 ? 0x1p-14 : 0x1p-126;
  vgpu::amd::cdna2_flush_inputs(a, g.K, mn);
  vgpu::amd::cdna2_flush_inputs(b, g.K, mn);
  vgpu::amd::cdna2_flush_inputs(&c, 1, 0x1p-126);
  float r = static_cast<float>(vgpu::amd::cdna2_dot_seq32(a, b, 1, g.K, c));
  if (r != r) r = static_cast<float>(vgpu::amd::mfma_dot(a, b, g.K, c));   // (the simulator takes the earlier NaN)
  return bitsof(r);
}

// ---------------------------------------------------------------------------------------------------------------------------
// Theirs: a transcription of MMA-Sim's arithmetic (commit pinned in docs/numerics/mmasim-crosscheck.md)
// ---------------------------------------------------------------------------------------------------------------------------
// frexp_and_normalize: x = s * 2^e with 1 <= |s| < 2, a subnormal (e below e_sub) held at e_sub with s scaled down, a zero at e_sub.
static void frexp_norm(double x, int esub, double& s, int& e) {
  int ee = 0;
  const double m = std::frexp(x, &ee);
  if (!std::isfinite(x)) ee = 0;   // (std::frexp leaves it unspecified; glibc, and so torch.frexp, gives 0)
  s = m * 2;
  e = ee - 1;
  if (e < esub) {
    s *= std::ldexp(1.0, e - esub);
    e = esub;
  }
  if (s == 0.0) e = esub;
}
// truncated_fused_sum: every term cut toward zero at 2^(e_max - F), then added.
static double trunc_fused(const double* s, const int* e, int n, int F, int& e_max) {
  e_max = e[0];
  for (int i = 1; i < n; ++i) e_max = std::max(e_max, e[i]);
  double sum = 0;
  for (int i = 0; i < n; ++i) sum += std::trunc(s[i] * std::ldexp(1.0, e[i] - e_max + F)) * std::ldexp(1.0, -F);
  return sum;
}
// ldexp_and_normalize with rho RNE-FP16 or RZ-FP32 (acc16 chooses); the result bits of the format, +0.0 added by the caller's choice.
static u32 finish(double s, int e, bool acc16, bool plus_zero) {
  if (acc16) {
    double s2;
    int e2;
    frexp_norm(s * std::ldexp(1.0, e), -14, s2, e2);
    s2 = std::nearbyint(s2 * 1024.0) / 1024.0;   // round half even (default rounding mode)
    const double x = s2 * std::ldexp(1.0, e2);
    if (x != x) return 0x7fffu;
    u32 h = vgpu::simd::f64_to_f16_bits(x);
    if (plus_zero && h == 0x8000u && g_mut_theirs != 3) h = 0;
    return h;
  }
  double s2;
  int e2;
  frexp_norm(s * std::ldexp(1.0, e), -126, s2, e2);
  s2 = (g_mut_theirs == 2 && std::fabs(s2) < 1.0 ? std::nearbyint(s2 * 8388608.0) : std::trunc(s2 * 8388608.0)) / 8388608.0;
  float r = static_cast<float>(s2 * std::ldexp(1.0, e2));
  if (r != r) return 0x7fffffffu;
  if (plus_zero && r == 0 && g_mut_theirs != 3) r = 0.0f;
  return bitsof(r);
}

// MMA_T_FDPA (mma.sync, wgmma, tcgen05.mma f16 / bf16 / tf32)
static u32 theirs_nv(const Group& g, const Vec& v) {
  const NvTheirs& p = g.nv;
  float c = v.c;
  const unsigned L = std::min<unsigned>(g.K, p.L);
  for (unsigned i = 0; i < g.K; i += L) {
    double s[17];
    int e[17];
    for (unsigned t = 0; t < L; ++t) {
      double sa, sb;
      int ea, eb;
      frexp_norm(v.a[i + t], p.esub, sa, ea);
      frexp_norm(v.b[i + t], p.esub, sb, eb);
      s[t] = static_cast<double>(static_cast<float>(sa) * static_cast<float>(sb));   // (float32 tensors in MMA-Sim)
      e[t] = ea + eb;
    }
    double sc;
    int ec;
    frexp_norm(c, g_mut_theirs == 4 ? -100000 : p.acc16 ? -14 : -126, sc, ec);
    s[L] = sc;
    e[L] = ec;
    for (unsigned t = 0; t <= L; ++t)
      if (s[t] == 0.0) e[t] = g_mut_theirs == 1 ? -100000 : p.e_zero;
    int e_max;
    const double sum = trunc_fused(s, e, static_cast<int>(L) + 1, p.F, e_max);
    const u32 bitsres = finish(sum, e_max, p.acc16, true);
    c = p.acc16 ? vgpu::simd::f16_to_f32(static_cast<uint16_t>(bitsres)) : f32of(bitsres);
    if (i + L >= g.K) return bitsres;
  }
  return 0;
}

// MMA_FMA (f32 forms): a chain of fused multiply-adds in float
static u32 theirs_fma(const Group& g, const Vec& v) {
  float c = v.c;
  for (unsigned k = 0; k < g.K; ++k) c = std::fmaf(v.a[k], v.b[k], c);
  return bitsof(c);
}

// MMA_FTZ_MUL_ADD (CDNA2 f16 / bf16): products flushed, added in pairs (P = 2) or pairs of pairs (P = 4), then to the accumulator
static float flush_f(float x, float lo, bool keep_sign) {
  if (std::fabs(x) < lo) return keep_sign ? x * 0.0f : 0.0f;
  return x;
}
static u32 theirs_ftz(const Group& g, const Vec& v) {
  const float lo16 = 0x1p-14f, lo32 = 0x1p-126f;
  const float lo_in = g.fmt == Fmt::F16 ? lo16 : lo32;
  float p[16];
  for (unsigned k = 0; k < g.K; ++k) {
    const float x = flush_f(v.a[k], lo_in, false), y = flush_f(v.b[k], lo_in, false);
    p[k] = flush_f(x * y, lo32, true);
  }
  auto add = [&](float x, float y) { return flush_f(x + y, lo32, true); };
  float c = flush_f(v.c, lo32, false);
  const int P = g.amd_L;
  for (unsigned i = 0; i < g.K; i += P) {
    float s = add(p[i], p[i + 1]);
    if (P == 4) s = add(s, add(p[i + 2], p[i + 3]));
    c = add(c, s);
  }
  return bitsof(c);
}

// MMA_E_FDPA (CDNA1): blocks of L products and the accumulator summed exactly, rounded once to nearest even.
// Done in 128-bit fixed point; a vector whose terms span more than 120 bits is skipped (returns false).
static bool theirs_exact(const Group& g, const Vec& v, u32& out) {
  float c = v.c;
  const unsigned L = std::min<unsigned>(g.K, g.amd_L);
  for (unsigned i = 0; i < g.K; i += L) {
    double dsum = c;
    bool finite = std::isfinite(c);
    for (unsigned t = 0; t < L; ++t) {
      dsum += static_cast<double>(v.a[i + t]) * static_cast<double>(v.b[i + t]);
      finite &= std::isfinite(v.a[i + t]) && std::isfinite(v.b[i + t]);
    }
    const float special = static_cast<float>(dsum);
    if (!finite || std::isnan(special) || std::isinf(special)) {
      c = special;
      continue;
    }
    struct Term { long long sig; int lsb; };
    Term terms[5];
    int n = 0;
    auto dec = [](float x, long long& sig, int& lsb) {
      int ee;
      const double m = std::frexp(static_cast<double>(x), &ee);
      sig = static_cast<long long>(std::ldexp(m, 24));
      lsb = ee - 24;
    };
    bool all_neg_zero = true;
    for (unsigned t = 0; t < L; ++t) {
      const float x = v.a[i + t], y = v.b[i + t];
      if (x == 0 || y == 0) {
        all_neg_zero &= std::signbit(x) != std::signbit(y);
        continue;
      }
      all_neg_zero = false;
      long long sa, sb;
      int la, lb;
      dec(x, sa, la);
      dec(y, sb, lb);
      terms[n++] = {sa * sb, la + lb};
    }
    if (c != 0) {
      all_neg_zero = false;
      long long sc;
      int lc;
      dec(c, sc, lc);
      terms[n++] = {sc, lc};
    } else {
      all_neg_zero &= std::signbit(c);
    }
    if (n == 0) {
      c = all_neg_zero ? -0.0f : 0.0f;
      continue;
    }
    int lo = terms[0].lsb, hi = -100000;
    for (int t = 0; t < n; ++t) {
      lo = std::min(lo, terms[t].lsb);
      const long long m = terms[t].sig < 0 ? -terms[t].sig : terms[t].sig;
      hi = std::max(hi, terms[t].lsb + 64 - __builtin_clzll(static_cast<unsigned long long>(m)));
    }
    if (hi - lo > 120 || hi > 120 || lo < -250) return false;
    __int128 S = 0;
    for (int t = 0; t < n; ++t) S += static_cast<__int128>(terms[t].sig) << (terms[t].lsb - lo);
    if (S == 0) {
      c = 0.0f;
      continue;
    }
    const bool neg = S < 0;
    unsigned __int128 mag = neg ? -static_cast<unsigned __int128>(S) : static_cast<unsigned __int128>(S);
    int nb = 0;
    for (unsigned __int128 t = mag; t; t >>= 1) ++nb;
    int e = lo;
    if (nb > 24) {
      const int drop = nb - 24;
      unsigned __int128 q = mag >> drop, rem = mag & ((static_cast<unsigned __int128>(1) << drop) - 1), half = static_cast<unsigned __int128>(1) << (drop - 1);
      if (rem > half || (rem == half && (q & 1))) ++q;
      mag = q;
      e += drop;
    }
    if (e + 24 < -120) return false;   // (subnormal results are outside the window)
    const float r = std::ldexp(static_cast<float>(static_cast<u32>(mag)), e);
    c = neg ? -r : r;
  }
  out = bitsof(c);
  return true;
}

// MMA_TR_FDPA (CDNA3 f16 / bf16 / xf32): products truncated to F = 24 bits around the largest, the accumulator floored, a floor to F2 = 31
static u32 theirs_tr(const Group& g, const Vec& v) {
  const int F = 24, F2 = 31;
  const int esub = g.fmt == Fmt::F16 ? -14 : -126;
  float c = v.c;
  const unsigned L = std::min<unsigned>(g.K, g.amd_L);
  for (unsigned i = 0; i < g.K; i += L) {
    double s[16];
    int e[16];
    for (unsigned t = 0; t < L; ++t) {
      double sa, sb;
      int ea, eb;
      frexp_norm(v.a[i + t], esub, sa, ea);
      frexp_norm(v.b[i + t], esub, sb, eb);
      double p = static_cast<double>(static_cast<float>(sa) * static_cast<float>(sb));
      int ee = ea + eb;
      if (((std::fabs(p) >= 2.0 ? 1 : 0) + ee) >= 128) p *= HUGE_VAL;   // (s[overflow] *= inf; a zero times inf is a NaN)
      if (p == 0.0) ee = -999;
      s[t] = p;
      e[t] = ee;
    }
    int e_dot;
    double s_dot = trunc_fused(s, e, static_cast<int>(L), F, e_dot);
    double s_c;
    int e_c;
    frexp_norm(c, -126, s_c, e_c);
    const int e_max = std::max(e_dot, e_c);
    s_dot = std::floor(s_dot * std::ldexp(1.0, e_dot - e_max + (F2 + 1))) * std::ldexp(1.0, -(F2 + 1));
    s_c = std::floor(s_c * std::ldexp(1.0, e_c - e_max + F)) * std::ldexp(1.0, -F);
    double sum;
    int e_sum;
    frexp_norm((s_dot + s_c) * std::ldexp(1.0, e_max), -126, sum, e_sum);
    sum = std::floor(sum * std::ldexp(1.0, F2)) * std::ldexp(1.0, -F2);
    float r = static_cast<float>(sum * std::ldexp(1.0, e_sum));
    if (r != r) r = f32of(0x7fffffffu);
    c = r;
  }
  return bitsof(c);
}

// ---------------------------------------------------------------------------------------------------------------------------
// Comparison and report
// ---------------------------------------------------------------------------------------------------------------------------
struct Col {
  std::string name;
  std::function<u32(const Group&, const Vec&)> f;
};
struct Stat {
  unsigned long n = 0, bad = 0, skipped = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, other = 0;
  unsigned long cov[4] = {0, 0, 0, 0};   // vectors in each corner's neighbourhood, matching or not (how well the corners were exercised)
  std::vector<std::string> first, first_other;
};

static bool both_nan_amd(u32 a, u32 b) { return (a & 0x7fffffffu) > 0x7f800000u && (b & 0x7fffffffu) > 0x7f800000u; }

// Which of the four subnormal corners (docs/matrix-numerics.md, R1-R4) a mismatching vector is in, by predicates on the inputs and the two results.
static unsigned corners(const Group& g, const Vec& v, u32 ours, u32 theirs) {
  unsigned m = 0;
  const bool a16 = g.acc16;
  const float cabs = std::fabs(v.c);
  // R3: a subnormal accumulator
  if (v.c != 0 && cabs == cabs && cabs < (a16 ? 0x1p-14f : 0x1p-126f)) m |= 4;
  // R2: a zero accumulator and every product below the floor under it (the floor is `zero_acc_floor` of the grid)
  if (v.c == 0) {
    int top = -100000;
    const int emin = g.emin;
    for (unsigned k = 0; k < g.K; ++k) {
      const float x = v.a[k], y = v.b[k];
      if (x == 0 || y == 0 || !std::isfinite(x) || !std::isfinite(y)) continue;
      int ea, eb;
      std::frexp(x, &ea);
      std::frexp(y, &eb);
      top = std::max(top, std::max(ea - 1, emin) + std::max(eb - 1, emin) + 1);
    }
    if (top > -100000 && top < (a16 ? 6 : -106) - g.our_grid) m |= 2;
  }
  // R1: an f32 result in the subnormal range, on either side
  if (!a16) {
    for (u32 r : {ours, theirs})
      if ((r & 0x7fffffffu) != 0 && (r & 0x7fffffffu) < 0x00800000u) m |= 1;
  }
  // R4: a zero result of the other sign
  const u32 z = a16 ? 0x8000u : 0x80000000u;
  if (ours == z || theirs == z) m |= 8;
  return m;
}

// R4's neighbourhood (for the coverage count only): our result is zero although the real-number sum of the finite terms is negative.
static bool r4_neighbourhood(const Group& g, const Vec& v, u32 ours) {
  const u32 mag = g.acc16 ? (ours & 0x7fffu) : (ours & 0x7fffffffu);
  if (mag != 0) return false;
  double sum = v.c;
  for (unsigned k = 0; k < g.K; ++k) {
    if (!std::isfinite(v.a[k]) || !std::isfinite(v.b[k])) return false;
    sum += static_cast<double>(v.a[k]) * static_cast<double>(v.b[k]);
  }
  return std::isfinite(sum) && sum < 0;
}

static std::string vec_str(const Group& g, const Vec& v) {
  const Fmt accf = g.acc16 ? Fmt::F16 : Fmt::F32;
  std::string s = "a=[";
  for (unsigned k = 0; k < g.K; ++k) s += (k ? " " : "") + hexval(g.fmt, v.a[k]);
  s += "] b=[";
  for (unsigned k = 0; k < g.K; ++k) s += (k ? " " : "") + hexval(g.fmt, v.b[k]);
  s += "] c=" + hexval(accf, v.c);
  return s;
}

static void score(const Group& g, const Vec& v, u32 ours, u32 theirs, bool nan_any, Stat& st, int show) {
  ++st.n;
  const unsigned m = g.side == "nv" ? corners(g, v, ours, theirs) : 0u;
  for (int i = 0; i < 3; ++i) st.cov[i] += (m >> i) & 1;
  if (g.side == "nv" && r4_neighbourhood(g, v, ours)) ++st.cov[3];
  if (ours == theirs || (nan_any && both_nan_amd(ours, theirs))) return;
  ++st.bad;
  if (m & 1) ++st.r1;
  if (m & 2) ++st.r2;
  if (m & 4) ++st.r3;
  if (m & 8) ++st.r4;
  if (!m) ++st.other;
  const char* w = g.acc16 ? "%04x" : "%08x";
  char b1[16], b2[16];
  std::snprintf(b1, sizeof b1, w, ours);
  std::snprintf(b2, sizeof b2, w, theirs);
  std::string line = vec_str(g, v) + "  ours=" + b1 + " mmasim=" + b2 + "  corners:" + ((m & 1) ? " R1" : "") + ((m & 2) ? " R2" : "") + ((m & 4) ? " R3" : "") +
                     ((m & 8) ? " R4" : "") + (m ? "" : " none");
  if (static_cast<int>(st.first.size()) < show) st.first.push_back(line);
  if (!m && static_cast<int>(st.first_other.size()) < show) st.first_other.push_back(line);
}

static std::vector<Col> columns(const Group& g) {
  std::vector<Col> c;
  if (g.side == "nv") {
    c.push_back({"ours: " + g.our_model, [](const Group& gg, const Vec& v) { return ours_nv(gg, v); }});
  } else {
    c.push_back({"ours: double sum, rounded once (default)", [](const Group& gg, const Vec& v) { return ours_amd(gg, v, false); }});
    if (g.amd_optin) c.push_back({"ours: gfx90a opt-in (VGPU_MFMA_CDNA2=fttn)", [](const Group& gg, const Vec& v) { return ours_amd(gg, v, true); }});
  }
  return c;
}

struct Totals { unsigned long n = 0, bad = 0; };

static void report(const Group& g, const std::vector<Col>& cols, const std::vector<Stat>& st, const char* source) {
  std::printf("\n### %s | %s %s (K=%u, %s accumulator) [%s]\n", g.arch.c_str(), g.api.c_str(), g.instr.c_str(), g.K, g.acc16 ? "f16" : "f32", source);
  for (size_t i = 0; i < cols.size(); ++i) {
    const Stat& s = st[i];
    std::printf("  %-52s vectors %lu, mismatches %lu (R1 %lu, R2 %lu, R3 %lu, R4 %lu, none of the four %lu)%s\n", cols[i].name.c_str(), s.n, s.bad, s.r1, s.r2, s.r3, s.r4,
                s.other, s.skipped ? (", skipped outside the exact window: " + std::to_string(s.skipped)).c_str() : "");
    if (g.side == "nv") std::printf("    corner coverage (vectors in each corner's neighbourhood): R1 %lu, R2 %lu, R3 %lu, R4 %lu\n", s.cov[0], s.cov[1], s.cov[2], s.cov[3]);
    for (const std::string& l : s.first) std::printf("    first: %s\n", l.c_str());
    for (const std::string& l : s.first_other) std::printf("    non-corner: %s\n", l.c_str());
  }
}

static std::string theirs_name(const Group& g) {
  if (g.side == "nv") return "MMA-Sim T_FDPA transcription";
  switch (g.amd_kind) {
    case 'e': return "MMA-Sim E_FDPA transcription";
    case 'p': return "MMA-Sim FTZ_MUL_ADD transcription";
    case 't': return "MMA-Sim TR_FDPA transcription";
    default: return "MMA-Sim FMA chain transcription";
  }
}

// theirs for the transcription; false: the vector is outside what the transcription handles
static bool theirs_run(const Group& g, const Vec& v, u32& out) {
  if (g.side == "nv") return out = theirs_nv(g, v), true;
  switch (g.amd_kind) {
    case 'e': return theirs_exact(g, v, out);
    case 'p': return out = theirs_ftz(g, v), true;
    case 't': return out = theirs_tr(g, v), true;
    default: return out = theirs_fma(g, v), true;
  }
}

static Fmt acc_fmt(const Group& g) { return g.acc16 ? Fmt::F16 : Fmt::F32; }

static std::vector<Vec> make_vectors(const Group& g, unsigned long n, u64 seed, size_t index) {
  Rng r(seed * 1000003ull + index * 7919ull + 17);
  std::vector<Vec> v;
  v.reserve(n);
  for (unsigned long i = 0; i < n; ++i) v.push_back(gen(r, g.fmt, acc_fmt(g), g.K, static_cast<int>(i % kClasses)));
  return v;
}

int main(int argc, char** argv) {
  if (argc < 2) return std::fprintf(stderr, "usage: %s run|export <dir>|compare <dir> [--n N] [--seed S] [--filter text] [--show K]\n", argv[0]), 2;
  const std::string mode = argv[1];
  std::string dir, filter;
  unsigned long n = 20000;
  u64 seed = 1;
  int show = 3;
  int ai = 2;
  if (mode != "run") {
    if (argc < 3) return std::fprintf(stderr, "a directory is needed\n"), 2;
    dir = argv[2];
    ai = 3;
  }
  for (; ai + 1 < argc; ai += 2) {
    const std::string k = argv[ai];
    if (k == "--n") n = std::strtoul(argv[ai + 1], nullptr, 10);
    else if (k == "--seed") seed = std::strtoull(argv[ai + 1], nullptr, 10);
    else if (k == "--filter") filter = argv[ai + 1];
    else if (k == "--show") show = std::atoi(argv[ai + 1]);
    else if (k == "--mut-theirs") g_mut_theirs = std::atoi(argv[ai + 1]);
    else if (k == "--mut-grid") g_mut_grid = std::atoi(argv[ai + 1]);
    else if (k == "--mut-chunk") g_mut_chunk = std::atoi(argv[ai + 1]);
    else return std::fprintf(stderr, "unknown option %s\n", k.c_str()), 2;
  }
  const std::vector<Group> groups = all_groups();
  FILE* manifest = nullptr;
  if (mode == "export") manifest = std::fopen((dir + "/manifest.tsv").c_str(), "w");
  Totals nvt, amdt;
  std::printf("mmasim_xcheck %s: %lu vectors per form, seed %llu\n", mode.c_str(), n, static_cast<unsigned long long>(seed));
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    const Group& g = groups[gi];
    const std::string label = g.arch + " " + g.api + " " + g.instr;
    if (!filter.empty() && label.find(filter) == std::string::npos) continue;
    const std::vector<Col> cols = columns(g);
    if (mode == "export") {
      const std::vector<Vec> vs = make_vectors(g, n, seed, gi);
      std::vector<u32> words;
      for (const Vec& v : vs) {
        for (unsigned k = 0; k < g.K; ++k) words.push_back(bitsof(v.a[k]));
        for (unsigned k = 0; k < g.K; ++k) words.push_back(bitsof(v.b[k]));
        words.push_back(bitsof(v.c));
      }
      if (FILE* f = std::fopen((dir + "/" + g.id + ".in").c_str(), "wb")) {
        std::fwrite(words.data(), 4, words.size(), f);
        std::fclose(f);
      }
      const char* fm = g.fmt == Fmt::F16 ? "f16" : g.fmt == Fmt::BF16 ? "bf16" : g.fmt == Fmt::TF32 ? "tf32" : "f32";
      if (manifest) std::fprintf(manifest, "%s\t%s\t%s\t%s\t%s\t%u\t%lu\t%s\t%s\n", g.id.c_str(), g.side.c_str(), g.arch.c_str(), g.api.c_str(), g.instr.c_str(), g.K, n, fm, g.acc16 ? "f16" : "f32");
      continue;
    }
    std::vector<Stat> st(cols.size());
    if (mode == "run") {
      const std::vector<Vec> vs = make_vectors(g, n, seed, gi);
      for (const Vec& v : vs) {
        u32 t;
        const bool ok = theirs_run(g, v, t);
        for (size_t ci = 0; ci < cols.size(); ++ci) {
          if (!ok) { ++st[ci].skipped; continue; }
          score(g, v, cols[ci].f(g, v), t, g.side == "amd", st[ci], show);
        }
      }
    } else {   // compare: the file the real MMA-Sim wrote
      FILE* fi = std::fopen((dir + "/" + g.id + ".in").c_str(), "rb");
      FILE* fo = std::fopen((dir + "/" + g.id + ".out").c_str(), "rb");
      if (!fi || !fo) {
        if (fi) std::fclose(fi);
        if (fo) std::fclose(fo);
        std::printf("\n### %s: no %s.in / %s.out in %s (not run)\n", label.c_str(), g.id.c_str(), g.id.c_str(), dir.c_str());
        continue;
      }
      const size_t per = 2 * g.K + 1;
      std::vector<u32> w(per);
      u32 t;
      while (std::fread(w.data(), 4, per, fi) == per && std::fread(&t, 4, 1, fo) == 1) {
        Vec v{};
        for (unsigned k = 0; k < g.K; ++k) v.a[k] = f32of(w[k]), v.b[k] = f32of(w[g.K + k]);
        v.c = f32of(w[2 * g.K]);
        for (size_t ci = 0; ci < cols.size(); ++ci) score(g, v, cols[ci].f(g, v), t, g.side == "amd", st[ci], show);
      }
      std::fclose(fi);
      std::fclose(fo);
    }
    report(g, cols, st, mode == "run" ? theirs_name(g).c_str() : "real MMA-Sim output");
    for (const Stat& s : st) (g.side == "nv" ? nvt : amdt).n += s.n, (g.side == "nv" ? nvt : amdt).bad += s.bad;
  }
  if (manifest) std::fclose(manifest);
  if (mode != "export") std::printf("\nTOTAL NVIDIA: %lu mismatches of %lu comparisons; AMD (every column): %lu of %lu\n", nvt.bad, nvt.n, amdt.bad, amdt.n);
  return 0;
}
