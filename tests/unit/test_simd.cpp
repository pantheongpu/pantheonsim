// The SIMD kernels (include/vgpu/simd, src/simd) against plain scalar code,
// at every level this CPU can run: same bits for every input that can break
// arithmetic -- zeros of both signs, subnormals, the extremes, infinities, NaNs
// with different payloads, rounding boundaries -- and for random ones.
//
// The references here are written independently of the kernels: the compiler's
// own _Float16 conversions (libgcc on a baseline build) for the narrowing and
// widening, and for the tensor-core kernel the SASS executor's original
// element-at-a-time form (decode to double, products rounded to f32, summed in
// k order). Levels the host cannot run are reported as skipped, not passed.
#pragma GCC optimize("fp-contract=off")

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <atomic>
#include <random>
#include <thread>
#include <string>
#include <vector>

#include "vgpu/exec/numerics.hpp"
#include "vgpu/simd/fp16.hpp"
#include "vgpu/simd/simd.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

uint32_t fb(float f) { return std::bit_cast<uint32_t>(f); }
uint64_t db(double d) { return std::bit_cast<uint64_t>(d); }

// The reference conversions: the compiler's.
uint32_t ref_f16_to_f32(uint16_t h) { return fb(static_cast<float>(std::bit_cast<_Float16>(h))); }
uint16_t ref_f32_to_f16(float f) { return std::bit_cast<uint16_t>(static_cast<_Float16>(f)); }
uint16_t ref_f64_to_f16(double d) { return std::bit_cast<uint16_t>(static_cast<_Float16>(d)); }

std::vector<simd::Level> levels() {
  std::vector<simd::Level> v;
  for (int i = 0; i < simd::kLevels; ++i) {
    const auto l = static_cast<simd::Level>(i);
    if (simd::supported(l)) v.push_back(l);
    else std::printf("  [ SKIP ] simd level %s: this CPU cannot run it\n", simd::level_name(l));
  }
  return v;
}

std::string hex(uint64_t v) {
  char b[24];
  std::snprintf(b, sizeof b, "%llx", static_cast<unsigned long long>(v));
  return b;
}

// Floats that sit on the rounding boundaries of binary16: each half, the
// midpoints between neighbours, and one float ulp either side of each.
std::vector<float> f16_boundaries() {
  std::vector<float> out;
  for (uint32_t h = 0; h < 0x7C00; ++h) {  // finite non-negative halves
    const float a = std::bit_cast<float>(ref_f16_to_f32(static_cast<uint16_t>(h)));
    const float b = std::bit_cast<float>(ref_f16_to_f32(static_cast<uint16_t>(h + 1)));
    const float mid = static_cast<float>((static_cast<double>(a) + static_cast<double>(b)) / 2);
    for (float x : {a, mid, b}) {
      const uint32_t u = fb(x);
      for (uint32_t d : {u - 1, u, u + 1}) {
        out.push_back(std::bit_cast<float>(d));
        out.push_back(-std::bit_cast<float>(d));
      }
    }
  }
  return out;
}

}  // namespace

// The F16C check is made on the first use of a kernel that needs F16C, once, from whichever thread gets there first.
VTEST(f16c_check_is_lazy_and_selects_what_it_did) {
  simd::testing::reset_f16c_check();
  VCHECK(simd::f16c_state() == simd::F16cState::Unchecked);
  VCHECK_EQ(simd::testing::f16c_checks_run(), 0);
  // Kernels that do not use F16C do not make the check.
  uint32_t a[32] = {1, 2, 3}, b[32] = {4, 5, 6}, c[32] = {7, 8, 9}, r[32];
  simd::kernels().iadd3(r, a, b, c, 0);
  VCHECK_EQ(r[0], 12u);
  VCHECK_EQ(r[2], 18u);
  VCHECK(simd::f16c_state() == simd::F16cState::Unchecked);
  const uint16_t halves[8] = {0x3c00, 0x4000, 0x7bff, 0x0001, 0xfc00, 0x7e00, 0x8000, 0x3555};
  float got[8];
  simd::kernels().f16_to_f32(halves, got, 8);
  for (int i = 0; i < 8; ++i) VCHECK_EQ(fb(got[i]), ref_f16_to_f32(halves[i]));
  if (simd::level() == simd::Level::Avx2) {
    VCHECK(simd::f16c_state() == simd::F16cState::Agrees);
    VCHECK_EQ(simd::testing::f16c_checks_run(), 1);
    // Afterwards the engines read the same table kernels_for gives, with the F16C kernels in it.
    const auto& k = simd::kernels();
    const auto& full = simd::kernels_for(simd::Level::Avx2);
    VCHECK(k.hmma == full.hmma && k.hfma2_m == full.hfma2_m && k.f16_to_f32 == full.f16_to_f32 && k.f32_to_f16 == full.f32_to_f16);
    VCHECK(k.hmma != simd::kernels_for(simd::Level::Base).hmma);
    VCHECK(k.iadd3 == full.iadd3 && k.ffma == full.ffma);
    simd::kernels().f16_to_f32(halves, got, 8);   // a second use does not check again
    VCHECK_EQ(simd::testing::f16c_checks_run(), 1);
  } else {
    VCHECK(simd::f16c_state() == simd::F16cState::Unchecked);   // the baseline never needs it
    VCHECK_EQ(simd::testing::f16c_checks_run(), 0);
  }
}

// Eight threads reaching the first F16C kernel together: one check, every thread's result right.
VTEST(f16c_check_with_concurrent_first_use) {
  std::mt19937 rng(77);
  uint32_t a[128], b[64], c[128], want[128];
  for (auto& w : a) w = 0x3c003c00u + (rng() & 0x03ff03ffu);
  for (auto& w : b) w = 0x3c003c00u + (rng() & 0x03ff03ffu);
  for (auto& w : c) w = rng() & 0x3fffffffu;
  simd::kernels_for(simd::Level::Base).hmma(0, 16, false, a, b, c, want);
  for (int round = 0; round < 40; ++round) {
    simd::testing::reset_f16c_check();
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<int> bad{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
      threads.emplace_back([&, t] {
        ready.fetch_add(1);
        while (!go.load()) {}
        uint32_t out[128] = {};
        if (t % 2) {
          simd::kernels().hmma(0, 16, false, a, b, c, out);
          for (int i = 0; i < 128; ++i) if (out[i] != want[i]) bad.fetch_add(1);
        } else {
          const uint16_t h[8] = {0x3c00, 0x4000, 0x7bff, 0x0001, 0xfc00, 0x7e00, 0x8000, 0x3555};
          float f[8];
          simd::kernels().f16_to_f32(h, f, 8);
          for (int i = 0; i < 8; ++i) if (fb(f[i]) != ref_f16_to_f32(h[i])) bad.fetch_add(1);
        }
      });
    while (ready.load() < 8) {}
    go.store(true);
    for (auto& th : threads) th.join();
    VCHECK_EQ(bad.load(), 0);
    if (simd::level() == simd::Level::Avx2) {
      VCHECK_EQ(simd::testing::f16c_checks_run(), 1);
      VCHECK(simd::f16c_state() == simd::F16cState::Agrees);
    }
  }
}

VTEST(f16_to_f32_all_inputs) {
  for (uint32_t h = 0; h < 65536; ++h)
    if (simd::f16_to_f32_bits(static_cast<uint16_t>(h)) != ref_f16_to_f32(static_cast<uint16_t>(h)))
      throw vtest::Failure("f16_to_f32_bits(0x" + hex(h) + ") = 0x" + hex(simd::f16_to_f32_bits(h)) + ", want 0x" +
                           hex(ref_f16_to_f32(h)));
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    std::vector<uint16_t> in(65536 + 3);   // an odd length: the tail past the vector loop
    for (uint32_t h = 0; h < in.size(); ++h) in[h] = static_cast<uint16_t>(h * 40503u);
    std::vector<float> out(in.size());
    k.f16_to_f32(in.data(), out.data(), in.size());
    for (size_t i = 0; i < in.size(); ++i)
      if (fb(out[i]) != ref_f16_to_f32(in[i]))
        throw vtest::Failure(std::string(simd::level_name(l)) + ": f16_to_f32 of 0x" + hex(in[i]) + " = 0x" +
                             hex(fb(out[i])) + ", want 0x" + hex(ref_f16_to_f32(in[i])));
  }
}

VTEST(f32_to_f16_edges_and_random) {
  std::vector<float> xs = f16_boundaries();
  for (float f : {0.0f, -0.0f, 1.0f, 65504.0f, 65519.99f, 65520.0f, 65536.0f, 1e30f, 6.1e-5f, 5.96e-8f, 2.98e-8f,
                  2.98023224e-8f, 1e-30f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max()})
    xs.push_back(f), xs.push_back(-f);
  for (uint32_t p : {0x7f800001u, 0x7fc00000u, 0x7fffffffu, 0xffc00001u, 0x7fa00000u, 0x7f801000u, 0x7fe01234u})
    xs.push_back(std::bit_cast<float>(p));
  std::mt19937 rng(7);
  for (int i = 0; i < 600000; ++i) {
    uint32_t u = rng();
    if (i & 1) u = (u & 0x80000000u) | (0x30000000u + (u & 0x0FFFFFFFu));   // around the half range
    xs.push_back(std::bit_cast<float>(u));
  }
  for (float x : xs)
    if (simd::f32_to_f16_bits(x) != ref_f32_to_f16(x))
      throw vtest::Failure("f32_to_f16_bits(0x" + hex(fb(x)) + ") = 0x" + hex(simd::f32_to_f16_bits(x)) + ", want 0x" +
                           hex(ref_f32_to_f16(x)));
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    std::vector<uint16_t> out(xs.size());
    k.f32_to_f16(xs.data(), out.data(), xs.size());
    for (size_t i = 0; i < xs.size(); ++i)
      if (out[i] != ref_f32_to_f16(xs[i]))
        throw vtest::Failure(std::string(simd::level_name(l)) + ": f32_to_f16 of 0x" + hex(fb(xs[i])) + " = 0x" +
                             hex(out[i]) + ", want 0x" + hex(ref_f32_to_f16(xs[i])));
  }
}

VTEST(f64_to_f16_edges_and_random) {
  std::vector<double> xs;
  for (uint32_t h = 0; h < 0x7C00; ++h) {
    const double a = static_cast<double>(std::bit_cast<float>(ref_f16_to_f32(static_cast<uint16_t>(h))));
    const double b = static_cast<double>(std::bit_cast<float>(ref_f16_to_f32(static_cast<uint16_t>(h + 1))));
    for (double x : {a, (a + b) / 2, b}) {
      const uint64_t u = db(x);
      for (uint64_t d : {u - 1, u, u + 1}) {
        xs.push_back(std::bit_cast<double>(d));
        xs.push_back(-std::bit_cast<double>(d));
      }
    }
  }
  for (double d : {0.0, 1.0, 65504.0, 65519.999999, 65520.0, 1e300, 1e-300, 2.9802322387695312e-08,
                   2.9802322387695316e-08, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::denorm_min(),
                   std::numeric_limits<double>::max(), std::numeric_limits<double>::min()})
    xs.push_back(d), xs.push_back(-d);
  for (uint64_t p : {0x7ff0000000000001ull, 0x7ff8000000000000ull, 0x7fffffffffffffffull, 0xfff8000000000001ull,
                     0x7ff4000000000000ull, 0x7ff0040000000000ull, 0x7ff0000000000400ull, 0x7ffc123456789abcull})
    xs.push_back(std::bit_cast<double>(p));
  std::mt19937_64 rng(11);
  for (int i = 0; i < 600000; ++i) {
    uint64_t u = rng();
    if (i & 1) u = (u & 0x8000000000000000ull) | (0x3D00000000000000ull + (u & 0x01FFFFFFFFFFFFFFull));   // around the half range
    xs.push_back(std::bit_cast<double>(u));
  }
  for (double x : xs)
    if (simd::f64_to_f16_bits(x) != ref_f64_to_f16(x))
      throw vtest::Failure("f64_to_f16_bits(0x" + hex(db(x)) + ") = 0x" + hex(simd::f64_to_f16_bits(x)) + ", want 0x" +
                           hex(ref_f64_to_f16(x)));
}

VTEST(bf16_helpers) {
  for (uint32_t h = 0; h < 65536; ++h) VCHECK_EQ(fb(simd::bf16_to_f32(static_cast<uint16_t>(h))), h << 16);
  // f32_to_bf16_bits: nearest even on the dropped half; a NaN keeps its top bits and is quieted.
  std::mt19937 rng(3);
  for (int i = 0; i < 1000000; ++i) {
    const uint32_t u = rng();
    const float f = std::bit_cast<float>(u);
    uint32_t want;
    if (f != f) want = (u >> 16) | 0x40;
    else want = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    VCHECK_EQ(static_cast<uint32_t>(simd::f32_to_bf16_bits(f)), want & 0xffffu);
  }
}

namespace {

// The SASS executor's original HMMA form (src/sass/exec_ops.inc exec_mma
// before the kernel): element-at-a-time decode into doubles, the fragment
// layouts of PTX ISA 9.7.16, products rounded to f32 and summed in k order.
double ref_elem(unsigned kind, uint32_t v, unsigned e) {
  if (kind == 0) return static_cast<double>(static_cast<float>(std::bit_cast<_Float16>(static_cast<uint16_t>(v >> (16 * e)))));
  if (kind == 1) return static_cast<double>(std::bit_cast<float>(static_cast<uint32_t>(static_cast<uint16_t>(v >> (16 * e))) << 16));
  return static_cast<double>(std::bit_cast<float>(v & 0xffffe000u));
}

void ref_hmma(unsigned kind, unsigned K, bool acc_f16, const uint32_t* a, const uint32_t* b, const uint32_t* c,
              uint32_t* out) {
  const unsigned per_reg = kind == 2 ? 1 : 2, M = 16, kN = 8;
  const unsigned a_regs = M * K / (32 * per_reg), b_regs = K * kN / (32 * per_reg);
  std::vector<double> A(M * K, 0.0), B(K * kN, 0.0);
  double C[16 * 8] = {};
  for (unsigned r = 0; r < a_regs; ++r)
    for (unsigned l = 0; l < 32; ++l) {
      const unsigned g = l / 4, t = l % 4, row = g + (r % 2) * 8, col0 = t * per_reg + (r / 2) * (per_reg * 4);
      for (unsigned e = 0; e < per_reg; ++e) A[row * K + col0 + e] = ref_elem(kind, a[r * 32 + l], e);
    }
  for (unsigned r = 0; r < b_regs; ++r)
    for (unsigned l = 0; l < 32; ++l) {
      const unsigned g = l / 4, t = l % 4, row0 = t * per_reg + r * (per_reg * 4);
      for (unsigned e = 0; e < per_reg; ++e) B[(row0 + e) * kN + g] = ref_elem(kind, b[r * 32 + l], e);
    }
  const auto cd = [&](unsigned l, unsigned slot) { return (l / 4 + (slot / 2) * 8) * kN + (l % 4) * 2 + slot % 2; };
  for (unsigned l = 0; l < 32; ++l)
    for (unsigned s = 0; s < 4; ++s) {
      double v;
      if (acc_f16) {
        const uint32_t word = c ? c[(s / 2) * 32 + l] : 0;
        v = static_cast<float>(std::bit_cast<_Float16>(static_cast<uint16_t>(word >> (16 * (s % 2)))));
      } else {
        v = std::bit_cast<float>(c ? c[s * 32 + l] : 0u);
      }
      C[cd(l, s)] = v;
    }
  double D[16 * 8];
  for (unsigned i = 0; i < M; ++i)
    for (unsigned j = 0; j < kN; ++j) {
      float acc = static_cast<float>(C[i * kN + j]);
      for (unsigned k = 0; k < K; ++k) {
        volatile float prod = static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * kN + j]);   // rounded here, never fused
        acc = acc + prod;
      }
      D[i * kN + j] = acc;
    }
  for (unsigned l = 0; l < 32; ++l)
    for (unsigned s = 0; s < 4; ++s) {
      const double v = D[cd(l, s)];
      if (acc_f16) {
        const uint16_t h = v != v ? uint16_t{0x7fff} : std::bit_cast<uint16_t>(static_cast<_Float16>(v));
        uint32_t& word = out[(s / 2) * 32 + l];
        if (s % 2 == 0) word = h;
        else word |= static_cast<uint32_t>(h) << 16;
      } else {
        out[s * 32 + l] = v != v ? 0x7fffffffu : fb(static_cast<float>(v));
      }
    }
}

// A word for an operand register: two 16-bit halves (or one f32 for tf32) drawn from the values that break arithmetic.
uint32_t operand_word(std::mt19937& rng, unsigned kind, int style) {
  static const uint16_t h16[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF,
                                 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE01, 0x3555, 0x4248, 0x0200, 0x5640, 0xD640};
  static const uint16_t b16[] = {0x0000, 0x8000, 0x0001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F,
                                 0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0x3EAB, 0x4049, 0x1F00, 0x6F00};
  static const uint32_t f32w[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x007FFFFFu, 0x00800000u, 0x3F800000u,
                                  0xBF800000u, 0x7F7FFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7F800001u,
                                  0x3F801000u, 0x3F800FFFu, 0x40490FDBu, 0x1F000000u};
  if (style == 0) return rng();
  if (kind == 2) return f32w[rng() % (sizeof f32w / sizeof f32w[0])];
  const uint16_t* t = kind == 0 ? h16 : b16;
  const unsigned n = kind == 0 ? sizeof h16 / sizeof h16[0] : sizeof b16 / sizeof b16[0];
  return uint32_t{t[rng() % n]} | (uint32_t{t[rng() % n]} << 16);
}

uint32_t acc_word(std::mt19937& rng, bool acc_f16, int style) {
  static const uint32_t f32w[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
                                  0x3F800000u, 0xBF800000u, 0x4B800000u, 0xCB800000u, 0x7F7FFFFFu, 0x33800000u};
  if (style == 0) return rng();
  if (acc_f16) return operand_word(rng, 0, 1);
  return f32w[rng() % (sizeof f32w / sizeof f32w[0])];
}

}  // namespace

VTEST(hmma_matches_the_element_form_at_every_level) {
  std::mt19937 rng(99);
  struct Form { unsigned kind, K; };
  const Form forms[] = {{0, 8}, {0, 16}, {1, 8}, {1, 16}, {2, 4}, {2, 8}};
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    for (const Form& f : forms)
      for (bool acc_f16 : {false, true})
        for (int round = 0; round < 400; ++round) {
          const unsigned per_reg = f.kind == 2 ? 1 : 2;
          const unsigned a_regs = 16 * f.K / (32 * per_reg), b_regs = f.K * 8 / (32 * per_reg), c_regs = acc_f16 ? 2 : 4;
          const int style = round % 3 == 0 ? 0 : 1;
          std::vector<uint32_t> a(a_regs * 32), b(b_regs * 32), c(c_regs * 32);
          for (auto& w : a) w = operand_word(rng, f.kind, style);
          for (auto& w : b) w = operand_word(rng, f.kind, style);
          for (auto& w : c) w = acc_word(rng, acc_f16, style);
          const bool zero_c = round % 17 == 5;   // RZ as the accumulator
          std::vector<uint32_t> got(c_regs * 32, 0xdeadbeefu), want(c_regs * 32, 0xdeadbeefu);
          k.hmma(f.kind, f.K, acc_f16, a.data(), b.data(), zero_c ? nullptr : c.data(), got.data());
          ref_hmma(f.kind, f.K, acc_f16, a.data(), b.data(), zero_c ? nullptr : c.data(), want.data());
          for (size_t i = 0; i < got.size(); ++i)
            if (got[i] != want[i])
              throw vtest::Failure(std::string(simd::level_name(l)) + " hmma kind " + std::to_string(f.kind) + " K " +
                                   std::to_string(f.K) + (acc_f16 ? " f16acc" : " f32acc") + " round " +
                                   std::to_string(round) + ": word " + std::to_string(i) + " = 0x" + hex(got[i]) +
                                   ", want 0x" + hex(want[i]));
        }
  }
}

namespace {

// Words that break integer and float arithmetic: zeros of both signs, one, the extremes, subnormals, infinities,
// NaNs with several payloads, values at the rounding edges of an f32 product, and a sign bit alone.
const uint32_t kWords[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007FFFFFu, 0x807FFFFFu, 0x00800000u,
                           0x80800000u, 0x3F800000u, 0xBF800000u, 0x3F800001u, 0x3F7FFFFFu, 0x40000000u, 0xC0000000u,
                           0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x7F800001u,
                           0x7FBFFFFFu, 0xFFA00000u, 0x7FC12345u, 0x33800000u, 0x4B000000u, 0x4B800001u, 0x1F800000u,
                           0x5F800000u, 0x3E800000u, 0xFFFFFFFFu, 0x7FFFFFFFu, 0xDEADBEEFu, 0x12345678u, 0x0000FFFFu,
                           0x00010000u, 0x55555555u, 0xAAAAAAAAu};
constexpr size_t kNW = sizeof kWords / sizeof kWords[0];

void fill(std::mt19937& rng, uint32_t* x, int style) {
  for (int l = 0; l < 32; ++l) x[l] = style == 0 ? static_cast<uint32_t>(rng()) : kWords[rng() % kNW];
}

uint32_t ref_lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t lut) {   // the executor's own, bit by bit
  uint32_t r = 0;
  for (int i = 0; i < 8; ++i)
    if ((lut >> i) & 1) r |= ((i & 4) ? a : ~a) & ((i & 2) ? b : ~b) & ((i & 1) ? c : ~c);
  return r;
}
uint32_t ref_out(float v) { return v != v ? 0x7fffffffu : fb(v); }
float asf(uint32_t b) { return std::bit_cast<float>(b); }

}  // namespace

VTEST(lane_kernels_match_the_scalar_forms_at_every_level) {
  std::mt19937 rng(5);
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    const auto check = [&](const char* what, const uint32_t* got, const uint32_t* want, const uint32_t* a,
                           const uint32_t* b, const uint32_t* c) {
      for (int i = 0; i < 32; ++i)
        if (got[i] != want[i])
          throw vtest::Failure(lv + " " + what + " lane " + std::to_string(i) + ": a=0x" + hex(a[i]) + " b=0x" +
                               hex(b[i]) + " c=0x" + hex(c ? c[i] : 0) + " got 0x" + hex(got[i]) + " want 0x" + hex(want[i]));
    };
    for (int round = 0; round < 60000; ++round) {
      uint32_t a[32], b[32], c[32], got[32], want[32];
      const int style = round % 3 == 0 ? 0 : 1;
      fill(rng, a, style);
      fill(rng, b, round % 5 == 0 ? 0 : style);
      fill(rng, c, style);
      const uint32_t add = round % 4;
      k.iadd3(got, a, b, c, add);
      for (int i = 0; i < 32; ++i) want[i] = a[i] + b[i] + c[i] + add;
      check("iadd3", got, want, a, b, c);
      k.imad(got, a, b, c);
      for (int i = 0; i < 32; ++i) want[i] = a[i] * b[i] + c[i];
      check("imad", got, want, a, b, c);
      const uint32_t lut = round < 256 ? static_cast<uint32_t>(round) : static_cast<uint32_t>(rng() & 0xff);
      k.lop3(got, a, b, c, lut);
      for (int i = 0; i < 32; ++i) want[i] = ref_lop3(a[i], b[i], c[i], lut);
      check("lop3", got, want, a, b, c);
      k.ffma(got, a, b, c);
      for (int i = 0; i < 32; ++i) want[i] = ref_out(std::fma(asf(a[i]), asf(b[i]), asf(c[i])));
      check("ffma", got, want, a, b, c);
      k.fadd(got, a, b);
      for (int i = 0; i < 32; ++i) {
        volatile float s = asf(a[i]) + asf(b[i]);
        want[i] = ref_out(s);
      }
      check("fadd", got, want, a, b, nullptr);
      k.fmul(got, a, b);
      for (int i = 0; i < 32; ++i) {
        volatile float p = asf(a[i]) * asf(b[i]);
        want[i] = ref_out(p);
      }
      check("fmul", got, want, a, b, nullptr);
    }
  }
}

VTEST(f64_to_f16_kernel_at_every_level) {
  std::vector<double> xs;
  for (uint32_t h = 0; h < 0x7C00; ++h) {
    const double a = static_cast<double>(std::bit_cast<float>(ref_f16_to_f32(static_cast<uint16_t>(h))));
    const double b = static_cast<double>(std::bit_cast<float>(ref_f16_to_f32(static_cast<uint16_t>(h + 1))));
    for (double x : {a, (a + b) / 2, b}) {
      const uint64_t u = db(x);
      for (uint64_t d : {u - 1, u, u + 1}) {
        xs.push_back(std::bit_cast<double>(d));
        xs.push_back(-std::bit_cast<double>(d));
      }
    }
  }
  for (uint64_t p : {0x7ff0000000000001ull, 0x7ff8000000000000ull, 0xfff8000000000001ull, 0x7fffffffffffffffull,
                     0x7ff0000000000000ull, 0xfff0000000000000ull, 0x0000000000000001ull, 0x8000000000000000ull})
    xs.push_back(std::bit_cast<double>(p));
  std::mt19937_64 rng(21);
  for (int i = 0; i < 400000; ++i) {
    uint64_t u = rng();
    if (i & 1) u = (u & 0x8000000000000000ull) | (0x3D00000000000000ull + (u & 0x01FFFFFFFFFFFFFFull));
    xs.push_back(std::bit_cast<double>(u));
  }
  xs.push_back(1.0), xs.push_back(2.0), xs.push_back(3.0);   // an odd count: the tail past the vector loop
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    std::vector<uint16_t> out(xs.size());
    k.f64_to_f16_canon(xs.data(), out.data(), xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
      const uint16_t want = xs[i] != xs[i] ? uint16_t{0x7fff} : ref_f64_to_f16(xs[i]);
      if (out[i] != want)
        throw vtest::Failure(std::string(simd::level_name(l)) + ": f64_to_f16 of 0x" + hex(db(xs[i])) + " = 0x" +
                             hex(out[i]) + ", want 0x" + hex(want));
    }
  }
}

VTEST(masked_kernels_at_every_level) {
  std::mt19937 rng(17);
  static const uint16_t h16[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x7C00,
                                 0xFC00, 0x7E00, 0x7C01, 0xFE01, 0x3555, 0x4248, 0x0200, 0x5640, 0xD640, 0x0100, 0x0010};
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    for (int round = 0; round < 40000; ++round) {
      const unsigned n = round & 1 ? 64 : 32;
      uint32_t a[64], b[64], c[64], d[64], d0[64], want[64];
      for (unsigned i = 0; i < n; ++i) {
        if (round % 3 == 0) {
          a[i] = rng(), b[i] = rng(), c[i] = rng();
        } else {
          constexpr size_t N = sizeof h16 / sizeof h16[0];
          a[i] = uint32_t{h16[rng() % N]} | (uint32_t{h16[rng() % N]} << 16);
          b[i] = uint32_t{h16[rng() % N]} | (uint32_t{h16[rng() % N]} << 16);
          c[i] = uint32_t{h16[rng() % N]} | (uint32_t{h16[rng() % N]} << 16);
        }
        d0[i] = rng();
      }
      uint64_t mask = (static_cast<uint64_t>(rng()) << 32) | rng();
      if (round % 7 == 0) mask = ~uint64_t{0};
      if (round % 11 == 0) mask = 0;
      const unsigned halves = round % 5 == 0 ? 1 : 2;
      const bool alias = round % 4 == 1;
      std::memcpy(d, d0, sizeof d);
      uint32_t* src_a = a;
      if (alias) {   // d is also the first source
        std::memcpy(d, a, sizeof d);
        std::memcpy(d0, a, sizeof d0);
      }
      k.hfma2_m(d, src_a, b, c, mask, n, halves);
      for (unsigned i = 0; i < n; ++i) {
        uint32_t out = 0;
        for (unsigned h = 0; h < halves; ++h) {
          const auto cv = [&](uint32_t w) { return static_cast<double>(static_cast<float>(std::bit_cast<_Float16>(static_cast<uint16_t>(w >> (16 * h))))); };
          const double v = std::fma(cv(a[i]), cv(b[i]), cv(c[i]));
          const uint16_t hb = v != v ? uint16_t{0x7fff} : std::bit_cast<uint16_t>(static_cast<_Float16>(v));
          out |= uint32_t{hb} << (16 * h);
        }
        want[i] = (mask >> i) & 1 ? out : d0[i];
        if (d[i] != want[i])
          throw vtest::Failure(lv + " hfma2_m round " + std::to_string(round) + " lane " + std::to_string(i) + ": got 0x" +
                               hex(d[i]) + " want 0x" + hex(want[i]) + " (a=0x" + hex(a[i]) + " b=0x" + hex(b[i]) +
                               " c=0x" + hex(c[i]) + ")");
      }
      // ffma_m on the same words.
      std::memcpy(d, d0, sizeof d);
      k.ffma_m(d, a, b, c, mask, n);
      for (unsigned i = 0; i < n; ++i) {
        const uint32_t w = (mask >> i) & 1 ? ref_out(std::fma(asf(a[i]), asf(b[i]), asf(c[i]))) : d0[i];
        if (d[i] != w)
          throw vtest::Failure(lv + " ffma_m round " + std::to_string(round) + " lane " + std::to_string(i) + ": got 0x" +
                               hex(d[i]) + " want 0x" + hex(w));
      }
    }
  }
}

namespace {

// A double to bfloat16 as the SASS executor's general form does it: to binary32 first (to nearest even), then to the nearest
// bfloat16 (ties to the even one), a NaN the canonical 0x7fff. Written by choosing between the two neighbours by their distance,
// not by the add-and-truncate the kernel uses.
uint16_t ref_f64_to_bf16(double v) {
  if (v != v) return 0x7fff;
  const float f = static_cast<float>(v);
  if (f != f) return 0x7fff;
  const uint32_t u = fb(f);
  const uint32_t sign = u & 0x80000000u, mag = u & 0x7fffffffu;
  if (mag >= 0x7f800000u) return static_cast<uint16_t>(u >> 16);   // an infinity
  const uint32_t lo = mag & 0xffff0000u, hi = lo + 0x10000u;
  const double dlo = static_cast<double>(asf(lo));
  const double dhi = hi >= 0x7f800000u ? std::ldexp(1.0, 128) : static_cast<double>(asf(hi));
  const double m = std::fabs(static_cast<double>(f));
  uint32_t pick;
  if (m - dlo < dhi - m) pick = lo;
  else if (m - dlo > dhi - m) pick = hi;
  else pick = ((lo >> 16) & 1) ? hi : lo;
  return static_cast<uint16_t>((sign | pick) >> 16);
}

}  // namespace

// hfma2_bf_m: both bfloat16 halves of each word, a * b + c with the exact binary64 result rounded to binary32 and then to
// bfloat16 (the executor's general form), on zeros, subnormals, the extremes, infinities, NaN payloads, cancellations and
// random words, with part of the lanes masked off and the destination aliasing a source.
VTEST(hfma2_bf_m_at_every_level) {
  std::mt19937 rng(23);
  static const uint16_t b16[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80,
                                 0x7fc0, 0x7f81, 0xffc1, 0x4049, 0x0040, 0x4f80, 0x5f00, 0x1f80, 0x3fc0, 0x7f00, 0x0100, 0x3f81,
                                 0x3f7f, 0x7e80, 0x2000, 0xa000};
  constexpr size_t N = sizeof b16 / sizeof b16[0];
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    for (int round = 0; round < 40000; ++round) {
      const unsigned n = round & 1 ? 64 : 32;
      uint32_t a[64], b[64], c[64], d[64], d0[64];
      for (unsigned i = 0; i < n; ++i) {
        if (round % 3 == 0) {
          a[i] = rng(), b[i] = rng(), c[i] = rng();
        } else {
          a[i] = uint32_t{b16[rng() % N]} | (uint32_t{b16[rng() % N]} << 16);
          b[i] = uint32_t{b16[rng() % N]} | (uint32_t{b16[rng() % N]} << 16);
          c[i] = uint32_t{b16[rng() % N]} | (uint32_t{b16[rng() % N]} << 16);
          if (round % 5 == 0) {   // c cancels the product, or is far below it: the rounding of the sum, not of the product
            for (int h = 0; h < 2; ++h) {
              const double p = static_cast<double>(asf(((a[i] >> (16 * h)) & 0xffffu) << 16)) *
                               static_cast<double>(asf(((b[i] >> (16 * h)) & 0xffffu) << 16));
              const float pf = static_cast<float>(-p);
              const uint32_t cb = fb(pf) >> 16;
              c[i] = (c[i] & ~(0xffffu << (16 * h))) | ((round % 10 == 0 ? cb : cb + 1) << (16 * h));
            }
          }
        }
        d0[i] = rng();
      }
      uint64_t mask = (static_cast<uint64_t>(rng()) << 32) | rng();
      if (round % 7 == 0) mask = ~uint64_t{0};
      if (round % 11 == 0) mask = 0;
      std::memcpy(d, d0, sizeof d);
      const uint32_t* src_a = a;
      if (round % 4 == 1) {   // d is also the first source
        std::memcpy(d, a, sizeof d);
        std::memcpy(d0, a, sizeof d0);
      }
      k.hfma2_bf_m(d, src_a, b, c, mask, n);
      for (unsigned i = 0; i < n; ++i) {
        uint32_t out = 0;
        for (unsigned h = 0; h < 2; ++h) {
          const auto cv = [&](uint32_t w) { return static_cast<double>(asf(((w >> (16 * h)) & 0xffffu) << 16)); };
          out |= uint32_t{ref_f64_to_bf16(std::fma(cv(a[i]), cv(b[i]), cv(c[i])))} << (16 * h);
        }
        const uint32_t want = (mask >> i) & 1 ? out : d0[i];
        if (d[i] != want)
          throw vtest::Failure(lv + " hfma2_bf_m round " + std::to_string(round) + " lane " + std::to_string(i) + ": got 0x" +
                               hex(d[i]) + " want 0x" + hex(want) + " (a=0x" + hex(a[i]) + " b=0x" + hex(b[i]) + " c=0x" +
                               hex(c[i]) + ")");
      }
    }
  }
}

namespace {

// The narrow-float decode and the exact sum as they were before the tables and the integer form (copied from
// src/exec/interpreter.cpp at the commit before this one): the references.
double ref_fp8(uint32_t byte, int exp_bits, int man_bits, int bias, bool has_inf) {
  const uint32_t man_mask = (1u << man_bits) - 1u;
  const uint32_t exp_mask = (1u << exp_bits) - 1u;
  const bool sign = (byte >> (exp_bits + man_bits)) & 1u;
  const uint32_t exp = (byte >> man_bits) & exp_mask;
  const uint32_t man = byte & man_mask;
  double v;
  if (exp == exp_mask) {
    if (has_inf) {
      v = man ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
    } else {
      v = man == man_mask ? std::numeric_limits<double>::quiet_NaN()
                          : std::ldexp(1.0 + static_cast<double>(man) / (man_mask + 1), static_cast<int>(exp) - bias);
    }
  } else if (exp == 0) {
    v = std::ldexp(static_cast<double>(man) / (man_mask + 1), 1 - bias);
  } else {
    v = std::ldexp(1.0 + static_cast<double>(man) / (man_mask + 1), static_cast<int>(exp) - bias);
  }
  return sign ? -v : v;
}
double ref_small_float(uint32_t code, int eb, int mb, int bias) {
  const uint32_t mant = code & ((1u << mb) - 1), exp = (code >> mb) & ((1u << eb) - 1);
  const double mag = exp ? std::ldexp(1.0 + mant / double(1u << mb), int(exp) - bias)
                         : std::ldexp(mant / double(1u << mb), 1 - bias);
  return (code >> (eb + mb)) & 1 ? -mag : mag;
}

float ref_narrow_sum(const double* terms, size_t n, float c) {
  bool nan = std::isnan(c), pinf = std::isinf(c) && c > 0, ninf = std::isinf(c) && c < 0;
  bool all_neg_zero = c == 0.0f && std::signbit(c);
  int max_e = std::numeric_limits<int>::min();
  auto look = [&](double t) {
    if (std::isnan(t)) { nan = true; all_neg_zero = false; return; }
    if (std::isinf(t)) { (t > 0 ? pinf : ninf) = true; all_neg_zero = false; return; }
    if (t != 0) { max_e = std::max(max_e, std::ilogb(t)); all_neg_zero = false; }
    else if (!std::signbit(t)) all_neg_zero = false;
  };
  for (size_t i = 0; i < n; ++i) look(terms[i]);
  if (std::isfinite(c) && c != 0.0f) max_e = std::max(max_e, std::ilogb(static_cast<double>(c)));
  if (nan || (pinf && ninf)) return std::bit_cast<float>(0x7fffffffu);
  if (pinf) return std::numeric_limits<float>::infinity();
  if (ninf) return -std::numeric_limits<float>::infinity();
  if (max_e == std::numeric_limits<int>::min()) return all_neg_zero ? -0.0f : 0.0f;
  const int lsb = max_e - 118;
  __int128 acc = 0;
  auto add = [&](double t) {
    if (t == 0) return;
    acc += static_cast<__int128>(std::floor(std::ldexp(t, -lsb)));
  };
  for (size_t i = 0; i < n; ++i) add(terms[i]);
  add(static_cast<double>(c));
  if (acc == 0) return 0.0f;
  const bool neg = acc < 0;
  unsigned __int128 mag = neg ? -static_cast<unsigned __int128>(acc) : static_cast<unsigned __int128>(acc);
  const uint64_t hi = static_cast<uint64_t>(mag >> 64), lo = static_cast<uint64_t>(mag);
  const int top = hi ? 127 - std::countl_zero(hi) : 63 - std::countl_zero(lo);
  const int shift = std::max(top - 23, -149 - lsb);
  const uint64_t m = shift >= 128 ? 0 : shift >= 0 ? static_cast<uint64_t>(mag >> shift) : static_cast<uint64_t>(mag << -shift);
  if (m == 0) return 0.0f;
  const double v = std::ldexp(static_cast<double>(m), lsb + shift);
  if (v >= 0x1p128) return neg ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
  return neg ? -static_cast<float>(v) : static_cast<float>(v);
}

}  // namespace

VTEST(narrow_float_decode_tables) {
  for (uint32_t i = 0; i < 256; ++i) {
    for (bool e5m2 : {false, true}) {
      const double want = e5m2 ? ref_fp8(i, 5, 2, 15, true) : ref_fp8(i, 4, 3, 7, false);
      const double got = exec::fp8_value(i, e5m2);
      VCHECK_EQ(db(got), db(want));
      VCHECK_EQ(db(exec::fp8_value(i + 256 * 3, e5m2)), db(want));   // only the low byte is the code
    }
  }
  const struct { int eb, mb, bias; } fmts[] = {{3, 2, 3}, {2, 3, 1}, {2, 1, 1}, {4, 3, 7}};
  for (const auto& f : fmts)
    for (uint32_t code = 0; code < 256; ++code)
      VCHECK_EQ(db(exec::mx_float_value(code, f.eb, f.mb, f.bias)), db(ref_small_float(code, f.eb, f.mb, f.bias)));
}

VTEST(narrow_sum_matches_the_libm_form) {
  std::mt19937_64 rng(31);
  std::vector<double> fp8;   // every finite fp8 value of both formats, and a few scales
  for (uint32_t i = 0; i < 256; ++i) {
    for (double v : {ref_fp8(i, 4, 3, 7, false), ref_fp8(i, 5, 2, 15, true)})
      if (std::isfinite(v)) fp8.push_back(v);
  }
  const double specials[] = {0.0, -0.0, std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN(), 1.0, -1.0, 0x1p-127, 0x1p127, 0x1p-149,
                             std::numeric_limits<double>::denorm_min(), 3.0 * 0x1p-1074, 0x1p-1022, 0x1p300, -0x1p-300};
  const float cs[] = {0.0f, -0.0f, 1.0f, -1.0f, 0x1p-149f, -0x1p-149f, 3.4e38f, -3.4e38f, std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 0x1p-126f, 1e-20f, 123456.789f};
  for (int round = 0; round < 200000; ++round) {
    const size_t n = round % 3 == 0 ? 32 : round % 3 == 1 ? 64 : 1 + rng() % 200;
    double terms[256];
    const int style = round % 5;
    for (size_t i = 0; i < n; ++i) {
      const uint64_t r = rng();
      double t;
      if (style == 0) {   // products of two fp8 values: the real shape
        t = fp8[r % fp8.size()] * fp8[(r >> 20) % fp8.size()];
      } else if (style == 1) {   // with block scales (powers of two over the whole ue8m0 range)
        t = fp8[r % fp8.size()] * fp8[(r >> 20) % fp8.size()] * std::ldexp(1.0, static_cast<int>((r >> 40) % 250) - 125) *
            std::ldexp(1.0, static_cast<int>((r >> 50) % 250) - 125);
      } else if (style == 2) {   // a few specials among them
        t = (r & 15) == 0 ? specials[(r >> 8) % (sizeof specials / sizeof specials[0])] : fp8[r % fp8.size()] * fp8[(r >> 20) % fp8.size()];
      } else if (style == 3) {   // wild doubles
        t = std::bit_cast<double>((r & 0x800FFFFFFFFFFFFFull) | (((r >> 52) % 400 + 800) << 52));
      } else {   // zeros and tiny ones mostly
        t = (r & 3) == 0 ? 0.0 : (r & 4) ? -0.0 : fp8[r % fp8.size()] * 0x1p-100;
      }
      terms[i] = t;
    }
    const float c = round % 4 == 0 ? cs[rng() % (sizeof cs / sizeof cs[0])] : static_cast<float>(fp8[rng() % fp8.size()]) * 0.37f;
    const float got = exec::mma_narrow_sum(terms, n, c), want = ref_narrow_sum(terms, n, c);
    if (fb(got) != fb(want))
      throw vtest::Failure("mma_narrow_sum round " + std::to_string(round) + " (n " + std::to_string(n) + ", style " +
                           std::to_string(style) + "): got 0x" + hex(fb(got)) + ", want 0x" + hex(fb(want)));
  }
}

VTEST(integer_lane_kernels_at_every_level) {
  std::mt19937 rng(41);
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    for (int round = 0; round < 40000; ++round) {
      const unsigned n = round & 1 ? 64 : 32;
      uint32_t a[64], b[64], c[64], d[64], d0[64];
      const int style = round % 3;
      for (unsigned i = 0; i < n; ++i) {
        a[i] = style == 0 ? static_cast<uint32_t>(rng()) : kWords[rng() % kNW];
        b[i] = style == 2 ? static_cast<uint32_t>(rng() % 70) : style == 0 ? static_cast<uint32_t>(rng()) : kWords[rng() % kNW];
        c[i] = style == 0 ? static_cast<uint32_t>(rng()) : kWords[rng() % kNW];
        d0[i] = rng();
      }
      uint64_t mask = (static_cast<uint64_t>(rng()) << 32) | rng();
      if (round % 7 == 0) mask = ~uint64_t{0};
      if (round % 11 == 0) mask = 0;
      const uint64_t all = n == 64 ? ~uint64_t{0} : 0xffffffffull;
      // comparisons
      for (unsigned cmp = 0; cmp < 6; ++cmp)
        for (bool sgn : {false, true}) {
          uint64_t want = 0;
          for (unsigned i = 0; i < n; ++i) {
            const int32_t sx = static_cast<int32_t>(a[i]), sy = static_cast<int32_t>(b[i]);
            bool r = false;
            switch (cmp) {
              case 0: r = a[i] == b[i]; break;
              case 1: r = a[i] != b[i]; break;
              case 2: r = sgn ? sx < sy : a[i] < b[i]; break;
              case 3: r = sgn ? sx <= sy : a[i] <= b[i]; break;
              case 4: r = sgn ? sx > sy : a[i] > b[i]; break;
              default: r = sgn ? sx >= sy : a[i] >= b[i]; break;
            }
            want |= uint64_t{r} << i;
          }
          const uint64_t got = k.icmp_mask(cmp, sgn, a, b, n);
          if (got != want)
            throw vtest::Failure(lv + " icmp_mask cmp " + std::to_string(cmp) + (sgn ? " signed" : " unsigned") + ": got 0x" +
                                 hex(got) + " want 0x" + hex(want));
        }
      // binary operations
      for (unsigned op = 0; op < 10; ++op)
        for (bool sgn : {false, true}) {
          std::memcpy(d, d0, sizeof d);
          k.ibin_m(op, sgn, d, a, b, mask, n);
          for (unsigned i = 0; i < n; ++i) {
            const uint32_t x = a[i], y = b[i];
            const int32_t sx = static_cast<int32_t>(x), sy = static_cast<int32_t>(y);
            uint32_t r = 0;
            switch (op) {
              case 0: r = x + y; break;
              case 1: r = x - y; break;
              case 2: r = x * y; break;
              case 3: r = x & y; break;
              case 4: r = x | y; break;
              case 5: r = x ^ y; break;
              case 6: r = sgn ? static_cast<uint32_t>(std::min(sx, sy)) : std::min(x, y); break;
              case 7: r = sgn ? static_cast<uint32_t>(std::max(sx, sy)) : std::max(x, y); break;
              case 8: r = y >= 32 ? 0u : x << y; break;
              default: r = sgn ? static_cast<uint32_t>(y >= 32 ? (sx < 0 ? -1 : 0) : sx >> y) : (y >= 32 ? 0u : x >> y); break;
            }
            const uint32_t want = (mask >> i) & 1 ? r : d0[i];
            if (d[i] != want)
              throw vtest::Failure(lv + " ibin_m op " + std::to_string(op) + (sgn ? " signed" : " unsigned") + " lane " +
                                   std::to_string(i) + ": a=0x" + hex(x) + " b=0x" + hex(y) + " got 0x" + hex(d[i]) +
                                   " want 0x" + hex(want));
          }
        }
      // d aliasing a source
      std::memcpy(d, a, n * 4);
      k.imad_m(d, d, b, c, mask, n);
      for (unsigned i = 0; i < n; ++i) {
        const uint32_t want = (mask >> i) & 1 ? a[i] * b[i] + c[i] : a[i];
        if (d[i] != want)
          throw vtest::Failure(lv + " imad_m lane " + std::to_string(i) + ": got 0x" + hex(d[i]) + " want 0x" + hex(want));
      }
      (void)all;
    }
  }
}

VTEST(double_lanes_and_popcount) {
  std::mt19937_64 rng(53);
  // popcount32 against the bit-by-bit count: every word with one or two bits set, every byte pattern in each
  // position, and random ones.
  const auto slow = [](uint32_t x) { unsigned n = 0; for (; x; x &= x - 1) ++n; return n; };
  for (int i = 0; i < 32; ++i)
    for (int j = 0; j < 32; ++j) VCHECK_EQ(simd::popcount32((1u << i) | (1u << j)), slow((1u << i) | (1u << j)));
  for (uint32_t b = 0; b < 256; ++b)
    for (int pos = 0; pos < 4; ++pos) VCHECK_EQ(simd::popcount32(b << (8 * pos)), slow(b << (8 * pos)));
  for (int i = 0; i < 4000000; ++i) {
    const uint32_t x = static_cast<uint32_t>(rng());
    if (simd::popcount32(x) != slow(x)) throw vtest::Failure("popcount32 of 0x" + hex(x));
  }
  VCHECK_EQ(simd::popcount32(0xffffffffu), 32u);
  VCHECK_EQ(simd::popcount32(0), 0u);
  static const uint64_t special[] = {0x0000000000000000ull, 0x8000000000000000ull, 0x3ff0000000000000ull,
                                     0xbff0000000000000ull, 0x7ff0000000000000ull, 0xfff0000000000000ull,
                                     0x7ff8000000000000ull, 0x7ff0000000000001ull, 0x0000000000000001ull,
                                     0x000fffffffffffffull, 0x0010000000000000ull, 0x7fefffffffffffffull,
                                     0x4000000000000000ull, 0x3ca0000000000000ull, 0x4340000000000000ull};
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    for (int round = 0; round < 60000; ++round) {
      uint32_t lo[3][32], hi[3][32], dlo[32], dhi[32], olo[32], ohi[32];
      for (int s = 0; s < 3; ++s)
        for (int i = 0; i < 32; ++i) {
          const uint64_t v = round % 3 == 0 ? rng() : round % 3 == 1 ? special[rng() % (sizeof special / sizeof special[0])]
                                                                      : (rng() & 0x800fffffffffffffull) | ((rng() % 40 + 1000) << 52);
          lo[s][i] = static_cast<uint32_t>(v);
          hi[s][i] = static_cast<uint32_t>(v >> 32);
        }
      for (int i = 0; i < 32; ++i) { olo[i] = rng(); ohi[i] = rng(); }
      const uint32_t mask = round % 5 == 0 ? ~0u : static_cast<uint32_t>(rng());
      const unsigned kind = round % 3, mods = static_cast<unsigned>(rng() & 63);
      std::memcpy(dlo, olo, sizeof dlo);
      std::memcpy(dhi, ohi, sizeof dhi);
      k.dop_m(kind, mods, dlo, dhi, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], mask);
      for (int i = 0; i < 32; ++i) {
        const auto get = [&](int s) {
          double d = std::bit_cast<double>(uint64_t{lo[s][i]} | (uint64_t{hi[s][i]} << 32));
          if (mods & (1u << (2 * s))) d = std::fabs(d);
          if (mods & (2u << (2 * s))) d = -d;
          return d;
        };
        volatile double x = get(0), y = get(1);
        double r;
        if (kind == 0) { volatile double t = x + y; r = t; }
        else if (kind == 1) { volatile double t = x * y; r = t; }
        else r = std::fma(x, y, get(2));
        const uint64_t u = (mask >> i) & 1 ? std::bit_cast<uint64_t>(r) : uint64_t{olo[i]} | (uint64_t{ohi[i]} << 32);
        const uint64_t g = uint64_t{dlo[i]} | (uint64_t{dhi[i]} << 32);
        // A NaN operand decides the result by the kernel's rule, not by the host instruction the compiler picked: the
        // first NaN of (b, a) for a sum or product (as the scalar code and an RTX 3060 do) and of (b, a, c) for an fma,
        // with its quiet bit set.
        uint64_t u2 = u;
        {
          const auto raw = [&](int s) {
            uint64_t v = uint64_t{lo[s][i]} | (uint64_t{hi[s][i]} << 32);
            if (mods & (1u << (2 * s))) v &= 0x7fffffffffffffffull;
            if (mods & (2u << (2 * s))) v ^= 0x8000000000000000ull;
            return v;
          };
          const auto nan = [](uint64_t v) { return (v & 0x7fffffffffffffffull) > 0x7ff0000000000000ull; };
          const uint64_t ra = raw(0), rb = raw(1), rc = kind == 2 ? raw(2) : 0;
          if (nan(ra) || nan(rb) || nan(rc)) {
            const uint64_t pick = kind == 2 ? (nan(rb) ? rb : nan(ra) ? ra : rc) : (nan(rb) ? rb : ra);
            if ((mask >> i) & 1) u2 = pick | 0x0008000000000000ull;
          }
        }
        if (g != u2)
          throw vtest::Failure(lv + " dop_m kind " + std::to_string(kind) + " mods " + std::to_string(mods) + " lane " +
                               std::to_string(i) + ": got 0x" + hex(g) + " want 0x" + hex(u2));
      }
    }
  }
}

// The FFMA and DFMA coverage of the SASS executor's own lane-math test (tests/unit/test_sass_lane_math.cpp, retired when
// its hardware-FMA loops gave way to these kernels): every triple of special values, and random bit patterns with
// the addend often set to cancel the product, so the rounding of the fused operation shows.
VTEST(fma_kernels_special_triples_and_cancellation) {
  static const uint32_t sp[] = {0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000,
                                0x80800000, 0x3f800000, 0xbf800000, 0x40000000, 0x3effffff, 0x7f7fffff, 0xff7fffff,
                                0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7fc12345, 0x7f800001, 0xffbfffff,
                                0x7fffffff, 0x4b000000, 0x4b800001, 0x33800000, 0x3f800001, 0x0f800000, 0x7e800000,
                                0x1e800000, 0x61800000};
  static const uint64_t dsp[] = {0x0000000000000000ull, 0x8000000000000000ull, 0x0000000000000001ull,
                                 0x000fffffffffffffull, 0x0010000000000000ull, 0x3ff0000000000000ull,
                                 0xbff0000000000000ull, 0x7fefffffffffffffull, 0x7ff0000000000000ull,
                                 0xfff0000000000000ull, 0x7ff8000000000000ull, 0x7ff0000000000001ull,
                                 0x4340000000000000ull, 0x3ca0000000000000ull, 0x3ff0000000000001ull};
  constexpr size_t kS = sizeof sp / sizeof sp[0], kD = sizeof dsp / sizeof dsp[0];
  std::mt19937_64 rng(77);
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    const auto check_f = [&](const uint32_t* a, const uint32_t* b, const uint32_t* c) {
      uint32_t got[32];
      k.ffma(got, a, b, c);
      for (int i = 0; i < 32; ++i) {
        const uint32_t want = ref_out(std::fma(asf(a[i]), asf(b[i]), asf(c[i])));
        if (got[i] != want)
          throw vtest::Failure(lv + " ffma lane " + std::to_string(i) + " fma(" + hex(a[i]) + ", " + hex(b[i]) + ", " +
                               hex(c[i]) + ") = " + hex(got[i]) + ", the fused operation says " + hex(want));
      }
    };
    const auto check_d = [&](const uint64_t* a, const uint64_t* b, const uint64_t* c) {
      uint32_t lo[3][32], hi[3][32], dlo[32] = {}, dhi[32] = {};
      for (int i = 0; i < 32; ++i) {
        const uint64_t v[3] = {a[i], b[i], c[i]};
        for (int s = 0; s < 3; ++s) {
          lo[s][i] = static_cast<uint32_t>(v[s]);
          hi[s][i] = static_cast<uint32_t>(v[s] >> 32);
        }
      }
      k.dop_m(2, 0, dlo, dhi, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], ~0u);
      for (int i = 0; i < 32; ++i) {
        const double want = std::fma(std::bit_cast<double>(a[i]), std::bit_cast<double>(b[i]), std::bit_cast<double>(c[i]));
        const uint64_t g = uint64_t{dlo[i]} | (uint64_t{dhi[i]} << 32);
        // Which NaN a NaN operand gives is the kernel's own rule (double_lanes_and_popcount checks it); here any NaN is a
        // NaN, and every other result is exact.
        const bool both_nan = want != want && std::bit_cast<double>(g) != std::bit_cast<double>(g);
        if (!both_nan && g != std::bit_cast<uint64_t>(want))
          throw vtest::Failure(lv + " dfma lane " + std::to_string(i) + " fma(" + hex(a[i]) + ", " + hex(b[i]) + ", " +
                               hex(c[i]) + ") = " + hex(g) + ", the fused operation says " + hex(std::bit_cast<uint64_t>(want)));
      }
    };
    uint32_t a[32], b[32], c[32];
    size_t n = 0;
    for (size_t x = 0; x < kS; ++x)
      for (size_t y = 0; y < kS; ++y)
        for (size_t z = 0; z < kS; ++z) {
          a[n % 32] = sp[x], b[n % 32] = sp[y], c[n % 32] = sp[z];
          if (++n % 32 == 0) check_f(a, b, c);
        }
    for (; n % 32 != 0; ++n) a[n % 32] = b[n % 32] = c[n % 32] = 0;
    check_f(a, b, c);
    for (int iter = 0; iter < 20000; ++iter) {
      for (unsigned i = 0; i < 32; ++i) {
        a[i] = static_cast<uint32_t>(rng()) ^ (static_cast<uint32_t>(rng()) << 20);
        b[i] = static_cast<uint32_t>(rng()) ^ (static_cast<uint32_t>(rng()) << 20);
        c[i] = static_cast<uint32_t>(rng()) ^ (static_cast<uint32_t>(rng()) << 20);
        if (iter % 3 == 0)   // the addend and the product cancel
          c[i] = (a[i] & 0x80000000u ? 0u : 0x80000000u) ^ (static_cast<uint32_t>(rng()) & 0x7f800000u) ^ (b[i] & 0x007fffffu);
      }
      check_f(a, b, c);
    }
    uint64_t da[32], dbv[32], dc[32];
    n = 0;
    for (size_t x = 0; x < kD; ++x)
      for (size_t y = 0; y < kD; ++y)
        for (size_t z = 0; z < kD; ++z) {
          da[n % 32] = dsp[x], dbv[n % 32] = dsp[y], dc[n % 32] = dsp[z];
          if (++n % 32 == 0) check_d(da, dbv, dc);
        }
    for (; n % 32 != 0; ++n) da[n % 32] = dbv[n % 32] = dc[n % 32] = 0;
    check_d(da, dbv, dc);
    for (int iter = 0; iter < 20000; ++iter) {
      for (unsigned i = 0; i < 32; ++i) {
        da[i] = rng();
        dbv[i] = rng();
        dc[i] = iter % 3 == 0 ? db(-(std::bit_cast<double>(da[i]) * std::bit_cast<double>(dbv[i]))) : rng();
      }
      check_d(da, dbv, dc);
    }
  }
}

// The wider integer kernels the SASS executor's plain-form ALU path uses (blend_m, iadd3x, imad_wide, shf), and the
// double kernel with its destination in the same registers as its sources. The references are written out here
// independently: 64-bit and 128-bit integer arithmetic, and the funnel-shift rule as the general path states it.
VTEST(wide_integer_kernels_at_every_level) {
  std::mt19937_64 rng(2026);
  static const uint32_t edge[] = {0u, 1u, 2u, 3u, 31u, 32u, 33u, 63u, 64u, 65u, 0x7fffffffu, 0x80000000u, 0x80000001u,
                                  0xffffffffu, 0xfffffffeu, 0xffffu, 0x10000u, 0x55555555u, 0xaaaaaaaau, 0xdeadbeefu};
  constexpr unsigned kE = sizeof edge / sizeof edge[0];
  const auto word = [&](int style) -> uint32_t {
    return style == 0 ? static_cast<uint32_t>(rng()) : style == 1 ? edge[rng() % kE] : static_cast<uint32_t>(rng() % 80);
  };
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    for (int round = 0; round < 30000; ++round) {
      const int style = round % 3;
      uint32_t a[32], b[32], c[32], c2[32], d0[32], d[32];
      for (int i = 0; i < 32; ++i) {
        a[i] = word(style);
        b[i] = word((style + round / 3) % 3);
        c[i] = word(style == 2 ? 1 : style);
        c2[i] = word((style + 1) % 3);
        d0[i] = static_cast<uint32_t>(rng());
      }
      uint32_t m1 = static_cast<uint32_t>(rng()), m2 = static_cast<uint32_t>(rng());
      if (round % 7 == 0) m1 = ~0u;
      if (round % 11 == 0) m2 = 0;
      if (round % 13 == 0) m1 = 0;
      if (round % 17 == 0) m2 = ~0u;

      // blend_m: the lanes of the mask from r, the others untouched; d may be what r was copied from.
      std::memcpy(d, d0, sizeof d);
      k.blend_m(d, a, m1);
      for (int i = 0; i < 32; ++i)
        if (d[i] != ((m1 >> i) & 1 ? a[i] : d0[i])) throw vtest::Failure(lv + " blend_m lane " + std::to_string(i));

      // iadd3x
      {
        const uint32_t add = static_cast<uint32_t>(rng() % 4);
        uint32_t r[32], g1 = 0xdeadbeef, g2 = 0xdeadbeef, w1 = 0, w2 = 0;
        k.iadd3x(r, a, b, c, add, m1, m2, &g1, &g2);
        for (int i = 0; i < 32; ++i) {
          const uint64_t sum = uint64_t{a[i]} + b[i] + c[i] + add + ((m1 >> i) & 1) + ((m2 >> i) & 1);
          w1 |= static_cast<uint32_t>(sum >> 32 >= 1) << i;
          w2 |= static_cast<uint32_t>(sum >> 32 >= 2) << i;
          if (r[i] != static_cast<uint32_t>(sum))
            throw vtest::Failure(lv + " iadd3x lane " + std::to_string(i) + ": got 0x" + hex(r[i]) + " want 0x" + hex(static_cast<uint32_t>(sum)));
        }
        if (g1 != w1 || g2 != w2)
          throw vtest::Failure(lv + " iadd3x carries: got 0x" + hex(g1) + "/0x" + hex(g2) + " want 0x" + hex(w1) + "/0x" + hex(w2));
      }

      // imad_wide, signed and unsigned
      for (bool sgn : {false, true}) {
        uint32_t rlo[32], rhi[32], carry = 0xdeadbeef, want_carry = 0;
        k.imad_wide(rlo, rhi, a, b, c, c2, sgn, m1, &carry);
        for (int i = 0; i < 32; ++i) {
          const unsigned __int128 prod = sgn ? static_cast<unsigned __int128>(static_cast<__int128>(static_cast<int64_t>(static_cast<int32_t>(a[i])) * static_cast<int32_t>(b[i])) & ((static_cast<__int128>(1) << 64) - 1))
                                             : static_cast<unsigned __int128>(uint64_t{a[i]} * b[i]);
          const unsigned __int128 full = prod + (uint64_t{c[i]} | (uint64_t{c2[i]} << 32)) + ((m1 >> i) & 1);
          want_carry |= static_cast<uint32_t>((full >> 64) != 0) << i;
          const uint64_t lo64 = static_cast<uint64_t>(full);
          if (rlo[i] != static_cast<uint32_t>(lo64) || rhi[i] != static_cast<uint32_t>(lo64 >> 32))
            throw vtest::Failure(lv + (sgn ? " imad_wide signed" : " imad_wide unsigned") + " lane " + std::to_string(i) + ": a=0x" +
                                 hex(a[i]) + " b=0x" + hex(b[i]) + " c=0x" + hex(c[i] | (uint64_t{c2[i]} << 32)) + " got 0x" +
                                 hex(uint64_t{rlo[i]} | (uint64_t{rhi[i]} << 32)) + " want 0x" + hex(lo64));
        }
        if (carry != want_carry)
          throw vtest::Failure(lv + " imad_wide carry out: got 0x" + hex(carry) + " want 0x" + hex(want_carry));
      }

      // shf, every form
      for (unsigned type = 0; type < 4; ++type)
        for (int form = 0; form < 8; ++form) {
          const bool wrap = form & 1, right = form & 2, hi = form & 4;
          uint32_t r[32];
          k.shf(r, a, b, c, type, wrap, right, hi);
          const bool is64 = type < 2, sgn = type == 0 || type == 2;
          const uint32_t width = is64 ? 64 : 32;
          for (int i = 0; i < 32; ++i) {
            const uint64_t v = a[i] | (uint64_t{c[i]} << 32);
            const uint32_t n = wrap ? b[i] % width : std::min(b[i], width);
            uint64_t x;
            if (!right) x = n >= 64 ? 0 : v << n;
            else if (sgn) x = static_cast<uint64_t>(static_cast<int64_t>(v) >> std::min(n, 63u));
            else x = n >= 64 ? 0 : v >> n;
            const uint32_t want = static_cast<uint32_t>(hi ? x >> 32 : x);
            if (r[i] != want)
              throw vtest::Failure(lv + " shf type " + std::to_string(type) + (wrap ? " wrap" : " clamp") + (right ? " right" : " left") +
                                   (hi ? " hi" : " lo") + " lane " + std::to_string(i) + ": lo=0x" + hex(a[i]) + " top=0x" + hex(c[i]) +
                                   " n=" + std::to_string(b[i]) + " got 0x" + hex(r[i]) + " want 0x" + hex(want));
          }
        }
    }
  }
}

// The float kernels the SASS executor's fast path calls for flush and saturation, compares, MUFU, conversions and min/max,
// against the executor's own scalar rules written out here (the general rule of exec_float, on doubles and libm), over special
// values (zeros of both signs, subnormals, the extremes, infinities, NaNs of both signs and payloads) and random bits.
namespace {
float ftz_ref(float f) { return std::fpclassify(f) == FP_SUBNORMAL ? std::copysign(0.0f, f) : f; }
float sat_ref(float f) { return f != f ? 0.0f : f < 0.0f ? 0.0f : f > 1.0f ? 1.0f : f; }
bool fcmp_ref(unsigned cmp, double a, double b) {
  const bool un = a != a || b != b;
  switch (cmp) {
    case 0: return false;
    case 1: return !un && a < b;
    case 2: return !un && a == b;
    case 3: return !un && a <= b;
    case 4: return !un && a > b;
    case 5: return !un && a != b;
    case 6: return !un && a >= b;
    case 7: return !un;
    case 8: return un;
    case 9: return un || a < b;
    case 10: return un || a == b;
    case 11: return un || a <= b;
    case 12: return un || a > b;
    case 13: return un || a != b;
    case 14: return un || a >= b;
    default: return true;
  }
}
const uint32_t kFloatBits[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu, 0x807fffffu, 0x00800000u, 0x80800000u,
                               0x3f800000u, 0xbf800000u, 0x3f000000u, 0x40000000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
                               0x7fc00000u, 0xffc00000u, 0x7fc12345u, 0x7f800001u, 0xffbfffffu, 0x3effffffu, 0x4b000000u, 0x4b800001u,
                               0x33800000u, 0x3f800001u, 0x2f800000u, 0x4f000000u, 0xcf000000u, 0x4f800000u, 0x4effffffu, 0xbf000000u,
                               0x3f7fffffu, 0x40490fdbu, 0x0f800000u, 0x7e800000u, 0xc0000000u};
constexpr size_t kNFB = sizeof kFloatBits / sizeof kFloatBits[0];
}  // namespace

VTEST(float_forms_at_every_level) {
  std::mt19937 rng(9090);
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    const auto fail = [&](const char* what, unsigned lane, uint32_t a, uint32_t b, uint32_t c, uint32_t got, uint32_t want, int info) {
      throw vtest::Failure(lv + " " + what + " info " + std::to_string(info) + " lane " + std::to_string(lane) + ": a=0x" + hex(a) +
                           " b=0x" + hex(b) + " c=0x" + hex(c) + " got 0x" + hex(got) + " want 0x" + hex(want));
    };
    for (int round = 0; round < 20000; ++round) {
      uint32_t a[32], b[32], c[32], r[32];
      const int style = round % 3;
      for (int i = 0; i < 32; ++i) {
        a[i] = style == 0 ? static_cast<uint32_t>(rng()) : kFloatBits[rng() % kNFB];
        b[i] = style == 1 ? static_cast<uint32_t>(rng()) : kFloatBits[rng() % kNFB];
        c[i] = round % 5 == 0 ? static_cast<uint32_t>(rng()) : kFloatBits[rng() % kNFB];
      }
      // ffma_x, fadd_x, fmul_x
      for (unsigned flags = 0; flags < 4; ++flags) {
        const auto in = [&](uint32_t v) { return (flags & 1) ? ftz_ref(std::bit_cast<float>(v)) : std::bit_cast<float>(v); };
        const auto fin = [&](float f) {
          if (flags & 1) f = ftz_ref(f);
          if (flags & 2) f = sat_ref(f);
          return f != f ? 0x7fffffffu : std::bit_cast<uint32_t>(f);
        };
        k.ffma_x(r, a, b, c, flags);
        for (int i = 0; i < 32; ++i) {
          volatile float v = std::fma(in(a[i]), in(b[i]), in(c[i]));
          if (r[i] != fin(v)) fail("ffma_x", i, a[i], b[i], c[i], r[i], fin(v), flags);
        }
        k.fadd_x(r, a, b, flags);
        for (int i = 0; i < 32; ++i) {
          volatile float v = in(a[i]) + in(b[i]);
          if (r[i] != fin(v)) fail("fadd_x", i, a[i], b[i], 0, r[i], fin(v), flags);
        }
        k.fmul_x(r, a, b, flags);
        for (int i = 0; i < 32; ++i) {
          volatile float v = in(a[i]) * in(b[i]);
          if (r[i] != fin(v)) fail("fmul_x", i, a[i], b[i], 0, r[i], fin(v), flags);
        }
      }
      // fcmp_mask
      for (unsigned cmp = 0; cmp < 16; ++cmp)
        for (bool ftz : {false, true}) {
          const uint32_t got = k.fcmp_mask(cmp, ftz, a, b);
          uint32_t want = 0;
          for (int i = 0; i < 32; ++i) {
            const float x = ftz ? ftz_ref(std::bit_cast<float>(a[i])) : std::bit_cast<float>(a[i]);
            const float y = ftz ? ftz_ref(std::bit_cast<float>(b[i])) : std::bit_cast<float>(b[i]);
            want |= static_cast<uint32_t>(fcmp_ref(cmp, x, y)) << i;
          }
          if (got != want)
            throw vtest::Failure(lv + " fcmp_mask cmp " + std::to_string(cmp) + (ftz ? " ftz" : "") + ": got 0x" + hex(got) + " want 0x" + hex(want));
        }
      // mufu: RCP, RSQ, SQRT
      for (unsigned kind : {4u, 5u, 8u}) {
        k.mufu(kind, r, a);
        for (int i = 0; i < 32; ++i) {
          volatile double x = ftz_ref(std::bit_cast<float>(a[i]));
          volatile double q = kind == 4 ? 1.0 / x : kind == 5 ? 1.0 / std::sqrt(x) : std::sqrt(x);
          const float f = ftz_ref(static_cast<float>(q));
          const uint32_t want = std::bit_cast<uint32_t>(f);
          if (r[i] != want) fail("mufu", i, a[i], 0, 0, r[i], want, static_cast<int>(kind));
        }
      }
      // i2f
      for (bool sgn : {false, true}) {
        k.i2f(sgn, r, a);
        for (int i = 0; i < 32; ++i) {
          const int64_t sv = sgn ? static_cast<int64_t>(static_cast<int32_t>(a[i])) : static_cast<int64_t>(a[i]);
          const uint32_t want = std::bit_cast<uint32_t>(static_cast<float>(sv));
          if (r[i] != want) fail("i2f", i, a[i], 0, 0, r[i], want, sgn);
        }
      }
      // f2i
      for (bool sgn : {false, true})
        for (unsigned rnd = 0; rnd < 4; ++rnd)
          for (bool ftz : {false, true}) {
            k.f2i(sgn, rnd, ftz, r, a);
            for (int i = 0; i < 32; ++i) {
              double x = ftz ? ftz_ref(std::bit_cast<float>(a[i])) : std::bit_cast<float>(a[i]);
              switch (rnd) {
                case 0: x = std::nearbyint(x); break;
                case 1: x = std::floor(x); break;
                case 2: x = std::ceil(x); break;
                default: x = std::trunc(x); break;
              }
              uint64_t want64;
              if (x != x) want64 = 0;
              else if (sgn) {
                const double lo = -std::ldexp(1.0, 31), hi = std::ldexp(1.0, 31) - 1;
                want64 = static_cast<uint64_t>(static_cast<int64_t>(x < lo ? lo : x > hi ? hi : x));
              } else {
                const double hi = std::ldexp(1.0, 32) - 1;
                want64 = static_cast<uint64_t>(x < 0 ? 0 : x > hi ? hi : x);
              }
              const uint32_t want = static_cast<uint32_t>(want64);
              if (r[i] != want) fail("f2i", i, a[i], 0, 0, r[i], want, static_cast<int>(sgn) * 100 + static_cast<int>(rnd) * 10 + ftz);
            }
          }
      // fmnmx
      {
        const uint32_t mn = round % 7 == 0 ? ~0u : round % 11 == 0 ? 0u : static_cast<uint32_t>(rng());
        for (bool ftz : {false, true})
          for (bool nan : {false, true}) {
            k.fmnmx(r, a, b, mn, ftz, nan);
            for (int i = 0; i < 32; ++i) {
              const float x = ftz ? ftz_ref(std::bit_cast<float>(a[i])) : std::bit_cast<float>(a[i]);
              const float y = ftz ? ftz_ref(std::bit_cast<float>(b[i])) : std::bit_cast<float>(b[i]);
              const bool m = (mn >> i) & 1;
              uint32_t want;
              if (nan && (x != x || y != y)) {
                want = 0x7fffffffu;
              } else {
                float v;
                if (x != x) v = y;
                else if (y != y) v = x;
                else if (x == 0 && y == 0) v = m ? (std::signbit(x) ? x : y) : (std::signbit(x) ? y : x);
                else v = m ? std::min(x, y) : std::max(x, y);
                want = std::bit_cast<uint32_t>(v);
              }
              if (r[i] != want) fail("fmnmx", i, a[i], b[i], 0, r[i], want, (m ? 4 : 0) + (ftz ? 2 : 0) + (nan ? 1 : 0));
            }
          }
      }
    }
  }
}

// PRMT's default mode against the byte-by-byte rule: every selector nibble (all 16 values in each of the four positions), over
// words of special and random bytes.
namespace {
// A 32-bit integer to float by the compiler's own conversion, in whatever rounding mode the host is in (out of line, and the
// result stored through a volatile, so that nothing is evaluated before the mode is set).
[[gnu::noinline]] uint32_t ref_i2f_in_mode(bool sgn, uint32_t a) {
  volatile float f = sgn ? static_cast<float>(static_cast<int64_t>(static_cast<int32_t>(a))) : static_cast<float>(static_cast<int64_t>(a));
  return std::bit_cast<uint32_t>(f);
}
}  // namespace

// i2f runs inside a RoundingScope in the SASS executor (I2F.RM / .RP / .RZ): under every host rounding mode it must give the bits the
// compiler's conversion gives -- including the sign of a zero (the unsigned conversion once made -0 from 0 when rounding down).
VTEST(i2f_under_directed_rounding_at_every_level) {
  std::mt19937 rng(77);
  static const uint32_t edge[] = {0u, 1u, 2u, 0x7fffffffu, 0x80000000u, 0x80000001u, 0xffffffffu, 0xfffffffeu, 0x01000001u, 0x01000003u,
                                  0x00ffffffu, 0xfeffffffu, 0xff000001u, 0x7f000001u, 0x40000001u, 0x3fffffffu, 0xc0000001u};
  constexpr size_t N = sizeof edge / sizeof edge[0];
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      for (int round = 0; round < 2000; ++round) {
        uint32_t a[32], r[32], want[32];
        for (int i = 0; i < 32; ++i) a[i] = round % 4 == 0 ? static_cast<uint32_t>(rng()) : edge[rng() % N] + (round % 3 == 0 ? static_cast<uint32_t>(rng() % 5) : 0u);
        for (bool sgn : {false, true}) {
          const int saved = std::fegetround();
          std::fesetround(mode);
          k.i2f(sgn, r, a);
          for (int i = 0; i < 32; ++i) want[i] = ref_i2f_in_mode(sgn, a[i]);
          std::fesetround(saved);
          for (int i = 0; i < 32; ++i)
            if (r[i] != want[i])
              throw vtest::Failure(lv + " i2f mode " + std::to_string(mode) + (sgn ? " signed" : " unsigned") + " lane " + std::to_string(i) +
                                   ": a=0x" + hex(a[i]) + " got 0x" + hex(r[i]) + " want 0x" + hex(want[i]));
        }
      }
    }
  }
}

VTEST(prmt0_at_every_level) {
  std::mt19937 rng(246);
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    for (int round = 0; round < 40000; ++round) {
      uint32_t a[32], b[32], sel[32], r[32];
      for (int i = 0; i < 32; ++i) {
        a[i] = round % 3 == 0 ? static_cast<uint32_t>(rng()) : kWords[rng() % kNW];
        b[i] = round % 3 == 1 ? static_cast<uint32_t>(rng()) : kWords[rng() % kNW];
        sel[i] = round < 20 ? (0x1111u * static_cast<uint32_t>(round % 16)) : static_cast<uint32_t>(rng() >> (rng() % 3 == 0 ? 16 : 0));
      }
      k.prmt0(r, a, b, sel);
      for (int i = 0; i < 32; ++i) {
        uint32_t want = 0;
        for (unsigned j = 0; j < 4; ++j) {
          const unsigned n = (sel[i] >> (4 * j)) & 0xf;
          const uint32_t byte = n & 4 ? (b[i] >> (8 * (n & 3))) & 0xff : (a[i] >> (8 * (n & 3))) & 0xff;
          want |= (n & 8 ? (byte & 0x80 ? 0xffu : 0u) : byte) << (8 * j);
        }
        if (r[i] != want)
          throw vtest::Failure(lv + " prmt0 lane " + std::to_string(i) + ": a=0x" + hex(a[i]) + " b=0x" + hex(b[i]) + " sel=0x" +
                               hex(sel[i]) + " got 0x" + hex(r[i]) + " want 0x" + hex(want));
      }
    }
  }
}

// dcmp: DSETP's compares and its MIN/MAX against the executor's rule on doubles, over special values and random bit patterns,
// with |x| and -x on either source.
VTEST(dcmp_at_every_level) {
  std::mt19937_64 rng(135);
  static const uint64_t special[] = {0x0000000000000000ull, 0x8000000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull,
                                     0x7ff0000000000000ull, 0xfff0000000000000ull, 0x7ff8000000000000ull, 0xfff8000000000001ull,
                                     0x7ff0000000000001ull, 0x0000000000000001ull, 0x800fffffffffffffull, 0x7fefffffffffffffull,
                                     0x4000000000000000ull, 0xc000000000000000ull};
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    for (int round = 0; round < 20000; ++round) {
      uint32_t lo[2][32], hi[2][32];
      double v[2][32];
      const unsigned mods = round % 4 == 0 ? static_cast<unsigned>(rng() & 15) : 0;
      for (int s = 0; s < 2; ++s)
        for (int i = 0; i < 32; ++i) {
          uint64_t u = round % 3 == 0 ? rng() : special[rng() % 14];
          if (s == 1 && round % 5 == 0 && i % 2 == 0) u = std::bit_cast<uint64_t>(v[0][i]);   // equal operands
          lo[s][i] = static_cast<uint32_t>(u);
          hi[s][i] = static_cast<uint32_t>(u >> 32);
          double d = std::bit_cast<double>(u);
          if (mods & (1u << (2 * s))) d = std::fabs(d);
          if (mods & (2u << (2 * s))) d = -d;
          v[s][i] = d;
        }
      for (unsigned cmp = 0; cmp < 16; ++cmp) {
        uint32_t got = 0, got_nan = 0xdeadbeef, want = 0, want_nan = 0;
        k.dcmp(cmp, mods, lo[0], hi[0], lo[1], hi[1], &got, &got_nan);
        for (int i = 0; i < 32; ++i) {
          const double a = v[0][i], b = v[1][i];
          bool r;
          if (cmp == 0 || cmp == 15) {
            const bool an = a != a, bn = b != b;
            if (an || bn) r = !an;
            else if (a == b) r = cmp == 0 ? std::signbit(a) : !std::signbit(a);
            else r = cmp == 0 ? a < b : a > b;
            want_nan |= static_cast<uint32_t>(an && bn) << i;
          } else {
            r = fcmp_ref(cmp, a, b);
          }
          want |= static_cast<uint32_t>(r) << i;
        }
        if (got != want || got_nan != want_nan)
          throw vtest::Failure(lv + " dcmp cmp " + std::to_string(cmp) + " mods " + std::to_string(mods) + ": got 0x" + hex(got) + "/0x" +
                               hex(got_nan) + " want 0x" + hex(want) + "/0x" + hex(want_nan));
      }
    }
  }
}

// lane_unary: the population count and the bit reversal against bit-by-bit loops, and FLO against a scan, for every word
// with one or two bits set, every byte pattern in each byte, the edge words and random ones.
VTEST(lane_unary_at_every_level) {
  std::mt19937 rng(515);
  std::vector<uint32_t> words;
  for (int i = 0; i < 32; ++i)
    for (int j = 0; j < 32; ++j) words.push_back((1u << i) | (1u << j));
  for (uint32_t b = 0; b < 256; ++b)
    for (int pos = 0; pos < 4; ++pos) words.push_back(b << (8 * pos));
  for (uint32_t v : {0u, ~0u, 0x80000000u, 0x7fffffffu, 0x80000001u, 0xdeadbeefu, 0x55555555u, 0xaaaaaaaau, 0xffffu, 0xffff0000u})
    words.push_back(v);
  for (int i = 0; i < 200000; ++i) words.push_back(static_cast<uint32_t>(rng()));
  for (int i = 0; i < 100000; ++i) words.push_back(static_cast<uint32_t>(rng()) >> (rng() % 32));   // (high bits clear)
  while (words.size() % 32) words.push_back(0);
  const auto flo = [](uint32_t v, bool sgn, bool sh) {
    if (sgn && (v >> 31)) v = ~v;
    uint32_t r = ~0u;
    for (int b = 31; b >= 0; --b)
      if ((v >> b) & 1) {
        r = static_cast<uint32_t>(b);
        break;
      }
    if (sh && r != ~0u) r = 31 - r;
    return r;
  };
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    for (size_t at = 0; at < words.size(); at += 32) {
      uint32_t r[32];
      for (unsigned kind = 0; kind < 6; ++kind) {
        k.lane_unary(kind, r, &words[at]);
        for (int i = 0; i < 32; ++i) {
          const uint32_t v = words[at + i];
          uint32_t want = 0;
          if (kind == 0) {
            for (int b = 0; b < 32; ++b) want += (v >> b) & 1;
          } else if (kind == 1) {
            for (int b = 0; b < 32; ++b) want |= ((v >> b) & 1u) << (31 - b);
          } else {
            want = flo(v, (kind - 2) & 1, ((kind - 2) & 2) != 0);
          }
          if (r[i] != want)
            throw vtest::Failure(lv + " lane_unary kind " + std::to_string(kind) + " of 0x" + hex(v) + ": got 0x" + hex(r[i]) +
                                 " want 0x" + hex(want));
        }
      }
    }
  }
}

// dop_m with the destination registers being the sources' (the SASS executor writes a result over an operand), and a
// register of the destination pair being the other half of an operand pair: the result of a lane depends on that
// lane's operands alone, all read before the lane is written.
VTEST(double_lanes_in_place_at_every_level) {
  std::mt19937_64 rng(88);
  static const uint64_t special[] = {0x0000000000000000ull, 0x8000000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull,
                                     0x7ff0000000000000ull, 0xfff0000000000000ull, 0x7ff8000000000000ull, 0x7ff0000000000001ull,
                                     0x0000000000000001ull, 0x000fffffffffffffull, 0x4000000000000000ull, 0x7fefffffffffffffull};
  for (simd::Level l : levels()) {
    const auto& k = simd::kernels_for(l);
    const std::string lv = simd::level_name(l);
    for (int round = 0; round < 20000; ++round) {
      // A file of 8 registers of 32 lanes; operands are register pairs (r, r + 1) at start registers 0..6.
      uint32_t file[8][32], ref[8][32];
      for (auto& row : file)
        for (uint32_t& v : row) v = round % 2 ? static_cast<uint32_t>(rng()) : static_cast<uint32_t>(special[rng() % 12] >> (rng() % 2 ? 32 : 0));
      std::memcpy(ref, file, sizeof file);
      const unsigned ra = rng() % 7, rb = rng() % 7, rc = rng() % 7, rd = rng() % 7;
      const unsigned kind = round % 3, mods = static_cast<unsigned>(rng() & 63);
      const uint32_t mask = round % 5 == 0 ? ~0u : static_cast<uint32_t>(rng());
      // The reference: operands copied out first, then the kernel on separate arrays, then the pair written.
      uint32_t lo[3][32], hi[3][32], olo[32], ohi[32];
      for (int i = 0; i < 32; ++i) {
        lo[0][i] = ref[ra][i]; hi[0][i] = ref[ra + 1][i];
        lo[1][i] = ref[rb][i]; hi[1][i] = ref[rb + 1][i];
        lo[2][i] = ref[rc][i]; hi[2][i] = ref[rc + 1][i];
        olo[i] = ref[rd][i];
        ohi[i] = ref[rd + 1][i];
      }
      // (the destination's own words are what is not overwritten; a lane of the destination pair that overlaps an
      // operand reads the old value because the operands were copied above)
      k.dop_m(kind, mods, olo, ohi, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], mask);
      for (int i = 0; i < 32; ++i) {
        ref[rd][i] = olo[i];
        ref[rd + 1][i] = ohi[i];
      }
      // The same, in place in the file. (When the destination pair is a register apart from an operand pair, the
      // halves overlap: rd == ra + 1 and so on.)
      k.dop_m(kind, mods, file[rd], file[rd + 1], file[ra], file[ra + 1], file[rb], file[rb + 1], file[rc], file[rc + 1], mask);
      for (unsigned r = 0; r < 8; ++r)
        for (int i = 0; i < 32; ++i)
          if (file[r][i] != ref[r][i] && !(rd == r || rd + 1 == r))
            throw vtest::Failure(lv + " dop_m in place changed register " + std::to_string(r));
      for (int i = 0; i < 32; ++i)
        if (file[rd][i] != ref[rd][i] || file[rd + 1][i] != ref[rd + 1][i]) {
          // A destination that overlaps an operand is read before it is written only when the registers are the same
          // or the lanes independent, which they are: any difference is a fault of the kernel.
          throw vtest::Failure(lv + " dop_m in place kind " + std::to_string(kind) + " regs a" + std::to_string(ra) + " b" +
                               std::to_string(rb) + " c" + std::to_string(rc) + " d" + std::to_string(rd) + " lane " + std::to_string(i));
        }
    }
  }
}

VTEST(level_choice) {
  VCHECK(simd::supported(simd::Level::Base));
  VCHECK(simd::supported(simd::level()));
  VCHECK_EQ(static_cast<int>(simd::kernels().level), static_cast<int>(simd::level()));
}

VTEST_MAIN
