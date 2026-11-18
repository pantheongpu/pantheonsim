// The e4m3 / e5m2 tensor-core sums (include/vgpu/exec/tc_fp8.hpp, docs/matrix-numerics.md):
//
//   * card_vectors: 40 vectors recorded on an RTX 1000 Ada by Khattak and Mikaitis (tc_fp8_card_vectors.inc), each one a result the
//     sequential sum gets wrong, f32 and f16 accumulators, e4m3 and e5m2;
//   * agrees_with_an_independent_form: the rule written the other way round (terms cut by value in double, not by integer alignment)
//     over random operands, subnormal operands, huge and tiny accumulators;
//   * special_values: NaN, infinities of e5m2, 0 x infinity, the sign of a zero, an f16 accumulator that overflows;
//   * the_corners: the two ways a zero term differs from an absent one;
//   * profiles_name_their_fp8_sum: every NVIDIA profile in the binary has the FP8 sum its compute capability's evidence gives it, and
//     none claims a card measurement it does not have.
// The engines' own use of it (PTX and SASS) is held to the recorded vectors by nvidia/tests/e2e/run_tcpapers.sh.
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "vgpu/exec/tc_fp8.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using namespace vgpu::exec::tc86;

namespace {

float f(uint32_t u) { return std::bit_cast<float>(u); }
uint32_t bits(double x) { return std::bit_cast<uint32_t>(static_cast<float>(x)); }

struct Vec {
  int fmt, acc16;
  uint32_t a[32], b[32], c, d;
};
const Vec kVectors[] = {
#include "tc_fp8_card_vectors.inc"
};

int emin_of(int fmt) { return fmt ? kEminE5M2 : kEminE4M3; }

// A float32 holding a value, rounded to binary16 (nearest even), as the recorded vectors' f16 accumulators went in: the paper's program
// converts c to f16 before the instruction.
double half_of(double x) {
  if (x == 0 || std::isinf(x) || std::isnan(x)) return x;
  const double lsb = std::ldexp(1.0, std::max(std::ilogb(x), -14) - 10);
  return std::nearbyint(x / lsb) * lsb;
}

double dot(const Fp8SumRule& r, bool acc16, int fa, int fb, const double* a, const double* b, unsigned K, double c) {
  return fp8_dot(r, acc16, emin_of(fa), emin_of(fb), a, 1, b, 1, K, c);
}

// The rule by value. Terms are exact in double; each is cut toward zero to a multiple of 2^(e_max - F) (exact: the scaled terms are
// integers below 2^30), summed (exact), and the sum is cut to the accumulator's format with frexp.
double reference(const Fp8SumRule& r, bool acc16, int fa, int fb, const double* a, const double* b, unsigned K, double c) {
  const int F = r.grid - 1;
  const auto ex = [](double v, int emin) { return std::max(std::ilogb(v), emin); };
  double acc = c;
  for (unsigned k0 = 0; k0 < K; k0 += r.chunk) {
    if (std::isinf(acc)) break;   // (an f16 accumulator that overflowed; operands here are finite)
    int e_max = INT32_MIN;
    for (unsigned k = k0; k < std::min(K, k0 + r.chunk); ++k)
      if (a[k] != 0 && b[k] != 0) e_max = std::max(e_max, ex(a[k], emin_of(fa)) + ex(b[k], emin_of(fb)));
    if (acc != 0) e_max = std::max(e_max, ex(acc, acc16 ? -14 : -126));
    if (e_max == INT32_MIN) { acc = 0; continue; }
    const double unit = std::ldexp(1.0, e_max - F);
    double sum = std::trunc(acc / unit) * unit;
    for (unsigned k = k0; k < std::min(K, k0 + r.chunk); ++k) sum += std::trunc(a[k] * b[k] / unit) * unit;
    if (sum == 0) { acc = 0; continue; }
    int e;
    std::frexp(sum, &e);   // |sum| in [2^(e-1), 2^e)
    if (acc16) {
      const double lsb = std::ldexp(1.0, std::max(e - 1, -14) - 10);
      acc = std::nearbyint(sum / lsb) * lsb;   // the host rounds to nearest even
    } else {
      const double lsb = std::ldexp(1.0, std::max(e - 1, -126) - r.out_bits);
      acc = std::trunc(sum / lsb) * lsb;
    }
    if (acc16 && std::fabs(acc) > 65504.0) acc = std::copysign(INFINITY, acc);
  }
  return acc;
}

// An e4m3 or e5m2 value from its byte.
double fp8_value(unsigned byte, int fmt) {
  const int sign = byte >> 7 ? -1 : 1;
  const int eb = fmt ? 5 : 4, mb = fmt ? 2 : 3, bias = fmt ? 15 : 7;
  const int e = (byte >> mb) & ((1 << eb) - 1), m = byte & ((1 << mb) - 1);
  if (fmt && e == 31) return m ? NAN : sign * INFINITY;
  if (!fmt && e == 15 && m == 7) return NAN;
  return sign * (e ? std::ldexp(1.0 + m / double(1 << mb), e - bias) : std::ldexp(m / double(1 << mb), 1 - bias));
}

}  // namespace

VTEST(card_vectors) {
  int seq_differs = 0;
  for (const Vec& v : kVectors) {
    double a[32], b[32];
    for (int k = 0; k < 32; ++k) a[k] = f(v.a[k]), b[k] = f(v.b[k]);
    const double c = v.acc16 ? half_of(f(v.c)) : f(v.c);
    const double got = dot(kAdaFp8Rule, v.acc16, v.fmt, v.fmt, a, b, 32, c);
    if (bits(got) != v.d) throw vtest::Failure("vector " + std::to_string(&v - kVectors) + ": rule " + std::to_string(got) + ", recorded " + std::to_string(f(v.d)));
    // the sequential sum (what the engines did before) is not the card's on these
    float acc = static_cast<float>(c);
    for (int k = 0; k < 32; ++k) acc += static_cast<float>(a[k] * b[k]);
    if (!v.acc16 && bits(acc) != v.d) ++seq_differs;
  }
  VCHECK(seq_differs >= 18);
  // the rule is one that the vectors tell from its neighbours: a grid bit more or less, chunks of 8 or 32, a full 23-bit result
  int rejected[4] = {};
  for (const Vec& v : kVectors) {
    double a[32], b[32];
    for (int k = 0; k < 32; ++k) a[k] = f(v.a[k]), b[k] = f(v.b[k]);
    Fp8SumRule r[4] = {kAdaFp8Rule, kAdaFp8Rule, kAdaFp8Rule, kAdaFp8Rule};
    r[0].grid = 15;
    r[1].grid = 13;
    r[2].chunk = 32;
    r[3].out_bits = 23;
    for (int i = 0; i < 4; ++i)
      if (i != 3 || !v.acc16) rejected[i] += bits(dot(r[i], v.acc16, v.fmt, v.fmt, a, b, 32, v.acc16 ? half_of(f(v.c)) : f(v.c))) != v.d;
  }
  for (int i = 0; i < 4; ++i)
    if (rejected[i] < (i == 3 ? 2 : 4)) throw vtest::Failure("neighbour " + std::to_string(i) + " rejected by " + std::to_string(rejected[i]) + " vectors only");
}

VTEST(agrees_with_an_independent_form) {
  std::mt19937_64 rng(664);
  for (int iter = 0; iter < 20000; ++iter) {
    const int fa = static_cast<int>(rng() & 1), fb = static_cast<int>(rng() & 1);
    const bool acc16 = rng() % 4 == 0;
    const unsigned K = (rng() & 1) ? 32 : 16;
    double a[32], b[32];
    // (the exponent range is varied so that terms sit at every distance from the largest; a byte of 0 or a subnormal comes often)
    const unsigned narrow = static_cast<unsigned>(rng() % 3);
    for (unsigned k = 0; k < K; ++k) {
      for (int side = 0; side < 2; ++side) {
        unsigned byte = static_cast<unsigned>(rng() & 0x7f) | (static_cast<unsigned>(rng() & 1) << 7);
        if (narrow == 1 && (byte & 0x7f) > 0x30) byte &= 0x8f;   // small values
        if (narrow == 2 && rng() % 4 == 0) byte &= 0x80;          // zeros
        double v = fp8_value(byte, side ? fb : fa);
        if (std::isnan(v) || std::isinf(v)) v = 1.0;
        (side ? b : a)[k] = v;
      }
    }
    double c;
    switch (rng() % 5) {
      case 0: c = 0; break;
      case 1: c = std::ldexp(1.0, static_cast<int>(rng() % 60) - 30) * (rng() & 1 ? 1 : -1); break;
      case 2: c = std::ldexp(static_cast<double>(rng() & 0xffffff), -static_cast<int>(rng() % 40) - 20); break;
      case 3: c = std::ldexp(1.0, -120 - static_cast<int>(rng() % 24)); break;   // a subnormal f32 (f16 below)
      default: c = std::ldexp(1.0, static_cast<int>(rng() % 100) - 40) * (static_cast<int>(rng() % 7) - 3); break;
    }
    c = static_cast<float>(c);
    if (acc16) c = std::ldexp(static_cast<double>(rng() % 2048), static_cast<int>(rng() % 30) - 24) * (rng() & 1 ? 1 : -1);   // a binary16 value (11 bits)
    const double want = reference(kAdaFp8Rule, acc16, fa, fb, a, b, K, c), got = dot(kAdaFp8Rule, acc16, fa, fb, a, b, K, c);
    if (std::isinf(want) ? !std::isinf(got) : bits(want) != bits(got) && !(want == 0 && got == 0))
      throw vtest::Failure("iteration " + std::to_string(iter) + ": reference " + std::to_string(want) + ", rule " + std::to_string(got));
  }
}

VTEST(special_values) {
  const double nan = NAN, inf = INFINITY;
  double a[32], b[32];
  const auto reset = [&] { for (int k = 0; k < 32; ++k) a[k] = b[k] = 1.0; };
  const auto run = [&](bool acc16, double c) { return dot(kAdaFp8Rule, acc16, 1, 1, a, b, 32, c); };
  reset();
  VCHECK_EQ(run(false, 0.0), 32.0);
  a[3] = nan;
  VCHECK(std::isnan(run(false, 0.0)));
  a[3] = inf;
  VCHECK(run(false, 0.0) == inf);
  b[3] = 0.0;   // 0 x infinity
  VCHECK(std::isnan(run(false, 0.0)));
  b[3] = -1.0;   // -infinity
  VCHECK(run(false, 0.0) == -inf);
  a[20] = inf;   // +infinity in the second chunk, where the first chunk's -infinity is the accumulator
  VCHECK(std::isnan(run(false, 0.0)));
  reset();
  VCHECK(std::isnan(run(false, nan)));
  VCHECK(run(false, inf) == inf);
  // the zero's sign: every sum that cancels is +0
  for (int k = 0; k < 32; ++k) a[k] = (k & 1) ? -1.0 : 1.0;
  VCHECK(!std::signbit(run(false, -0.0)));
  VCHECK(!std::signbit(run(false, 0.0)));
  // an f16 accumulator that overflows is an infinity
  reset();
  a[0] = 57344.0;
  VCHECK(run(true, 60000.0) == inf);
}

VTEST(the_corners) {
  // 2^-18 x 2^-18 products far under an f32 accumulator of 1: cut away (grid 13 below 2^0).
  double a[32] = {}, b[32] = {};
  a[0] = std::ldexp(1.0, -9);
  b[0] = std::ldexp(1.0, -9);
  VCHECK_EQ(dot(kAdaFp8Rule, false, 0, 0, a, b, 32, 1.0), 1.0);
  // ... but alone they are exact: the zero accumulator takes the position of the largest term
  VCHECK_EQ(dot(kAdaFp8Rule, false, 0, 0, a, b, 32, 0.0), std::ldexp(1.0, -18));
  // a result keeps 13 fraction bits: 1 + 2^-13 is kept, 1 + 2^-14 is cut
  a[0] = 1.0;
  b[0] = 1.0 + 0.0;
  VCHECK_EQ(dot(kAdaFp8Rule, false, 0, 0, a, b, 32, std::ldexp(1.0, -13)), 1.0 + std::ldexp(1.0, -13));
  VCHECK_EQ(dot(kAdaFp8Rule, false, 0, 0, a, b, 32, std::ldexp(1.0, -14)), 1.0);
}

VTEST(profiles_name_their_fp8_sum) {
  int ada = 0, via = 0, seq = 0;
  for (const std::string& id : available_gpus()) {
    const DeviceProfile p = load_gpu(id);
    if (p.vendor != "nvidia") {
      VCHECK(p.numerics.tensor_sum_fp8 == Fp8Sum::Sequential);
      continue;
    }
    const int cc = p.cc_major * 10 + p.cc_minor;
    const Fp8Sum want = cc == 89 ? Fp8Sum::Ada : (cc == 90 || cc == 100) ? Fp8Sum::ViaF16 : Fp8Sum::Sequential;
    if (p.numerics.tensor_sum_fp8 != want) throw vtest::Failure(id + ": numerics.tensor_sum_fp8 is not what its compute capability's evidence gives");
    (want == Fp8Sum::Ada ? ada : want == Fp8Sum::ViaF16 ? via : seq)++;
    if (want == Fp8Sum::Sequential) {
      VCHECK(!p.provenance_of("numerics.tensor_sum_fp8"));   // not stated, so no class is claimed
      continue;
    }
    // no card of ours ran an FP8 form: nothing may say measured
    for (const char* key : {"numerics.tensor_sum_fp8", "numerics.tensor_sum_fp8_chunk", "numerics.tensor_sum_fp8_grid", "numerics.tensor_sum_fp8_out_bits"}) {
      const auto c = p.provenance_of(key);
      if (want == Fp8Sum::ViaF16 && std::string(key) != "numerics.tensor_sum_fp8") continue;
      if (!c) throw vtest::Failure(id + ": " + key + " has no class");
      VCHECK(*c == Provenance::Inferred);
      VCHECK(!p.evidence_of(key).empty());
    }
    if (const Fp8SumRule* r = fp8_sum_rule(want)) {
      VCHECK(p.numerics.tensor_sum_fp8_chunk == r->chunk && p.numerics.tensor_sum_fp8_grid == r->grid && p.numerics.tensor_sum_fp8_out_bits == r->out_bits);
    } else {
      VCHECK(p.numerics.tensor_sum_fp8_chunk == 0 && p.numerics.tensor_sum_fp8_grid == 0 && p.numerics.tensor_sum_fp8_out_bits == 0);
      VCHECK(applies(p));   // via_f16 sums as the f16 rule does
    }
  }
  VCHECK_EQ(ada, 6);   // l4 l40 l40s rtx4080 rtx4090 rtx6000-ada
  VCHECK_EQ(via, 8);   // h100 h100-pcie h100-nvl h200 h200-nvl gh200-480gb b200 gb200
  VCHECK(seq > 0);
  // VGPU_TC_EXACT=0 turns both off
  const DeviceProfile l = load_gpu("nvidia/l40s"), h = load_gpu("nvidia/h100");
  VCHECK(fp8_rule(l) != nullptr && !fp8_via_f16(l) && fp8_rule(h) == nullptr && fp8_via_f16(h));
  refresh_exact("0");
  VCHECK(fp8_rule(l) == nullptr && !fp8_via_f16(h));
  refresh_exact(nullptr);
}

VTEST_MAIN
