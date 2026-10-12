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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

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

VTEST(level_choice) {
  VCHECK(simd::supported(simd::Level::Base));
  VCHECK(simd::supported(simd::level()));
  VCHECK_EQ(static_cast<int>(simd::kernels().level), static_cast<int>(simd::level()));
}

VTEST_MAIN
