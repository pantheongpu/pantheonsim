// OpLibm (patch E6): the OpenCL.std host-library math functions at f16, f32 and f64.
//
// Three layers, all against references written here from the C library and from the properties each function has:
//  1. the evaluator (vgpu::exec::libm_eval) over edge values (zeros, infinities, NaN, subnormals, the extremes,
//     integers and halves) and random bit patterns, bit for bit with the host function; the exact operations
//     (fmod, remainder, fdim, ldexp, nextafter, round, logb, ilogb) against their definitions, not against libm;
//     NaN results always the canonical NaN of the width; OpenCL's special cases (powr, pown, ilogb, fdim, pow);
//  2. f16 over all 65536 inputs (unary) and a dense grid (binary);
//  3. the instruction through the interpreter: a kernel built with the AST builder applies OpLibm to arrays at warp
//     widths 8 and 32 with a partial last warp, and every lane equals libm_eval (register widths, integer operand
//     and result, .ftz, a guard).
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "vgpu/exec/libm.hpp"
#include "ast_run.hpp"
#include "vtest.hpp"

using namespace astrun;
using vgpu::exec::libm_eval;
using vgpu::ptx::LibmFn;

namespace {

constexpr uint64_t kNaN16 = 0x7FFF, kNaN32 = 0x7FFFFFFFull, kNaN64 = 0x7FFFFFFFFFFFFFFFull;

float F(uint32_t b) { return std::bit_cast<float>(b); }
uint32_t B(float f) { return std::bit_cast<uint32_t>(f); }
double D(uint64_t b) { return std::bit_cast<double>(b); }
uint64_t DB(double d) { return std::bit_cast<uint64_t>(d); }

// The references: the host's double function for the inexact ones.
double ref_d(LibmFn fn, double x, double y, int n) {
  switch (fn) {
    case LibmFn::Acos: return std::acos(x);
    case LibmFn::Acosh: return std::acosh(x);
    case LibmFn::Asin: return std::asin(x);
    case LibmFn::Asinh: return std::asinh(x);
    case LibmFn::Atan: return std::atan(x);
    case LibmFn::Atan2: return std::atan2(x, y);
    case LibmFn::Atanh: return std::atanh(x);
    case LibmFn::Cbrt: return std::cbrt(x);
    case LibmFn::Cos: return std::cos(x);
    case LibmFn::Cosh: return std::cosh(x);
    case LibmFn::Erf: return std::erf(x);
    case LibmFn::Erfc: return std::erfc(x);
    case LibmFn::Exp: return std::exp(x);
    case LibmFn::Exp2: return std::exp2(x);
    case LibmFn::Exp10: return std::pow(10.0, x);
    case LibmFn::Expm1: return std::expm1(x);
    case LibmFn::Hypot: return std::hypot(x, y);
    case LibmFn::Lgamma: { int s; return ::lgamma_r(x, &s); }
    case LibmFn::Log: return std::log(x);
    case LibmFn::Log2: return std::log2(x);
    case LibmFn::Log10: return std::log10(x);
    case LibmFn::Log1p: return std::log1p(x);
    case LibmFn::Pow: return std::pow(x, y);
    case LibmFn::Pown: return std::pow(x, static_cast<double>(n));
    case LibmFn::Sin: return std::sin(x);
    case LibmFn::Sinh: return std::sinh(x);
    case LibmFn::Tan: return std::tan(x);
    case LibmFn::Tanh: return std::tanh(x);
    case LibmFn::Tgamma: return std::tgamma(x);
    default: return 0;
  }
}
// OpenCL's powr, from its definition: pow with the domain restrictions.
double ref_powr(double x, double y) {
  if (std::isnan(x) || std::isnan(y) || x < 0) return std::nan("");
  if (x == 0 && y == 0) return std::nan("");
  if (std::isinf(x) && y == 0) return std::nan("");
  if (x == 1 && std::isinf(y)) return std::nan("");
  return std::pow(std::fabs(x), y);
}

// A result the evaluator must give bit for bit: NaN is the canonical one.
bool same64(uint64_t got, double want) {
  return std::isnan(want) ? got == kNaN64 : got == DB(want);
}
bool same32(uint64_t got, float want) {
  return std::isnan(want) ? got == kNaN32 : got == B(want);
}

// ---- the values ------------------------------------------------------------------------------------------------

std::vector<uint32_t> edge32() {
  std::vector<uint32_t> v = {0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000,
                             0x80800000, 0x3F800000, 0xBF800000, 0x3F000000, 0xBF000000, 0x40000000, 0xC0000000,
                             0x40490FDB, 0x3FC90FDB, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
                             0xFFC00000, 0x7F800001, 0x3F800001, 0x3F7FFFFF, 0x41200000, 0x42C80000, 0xC1200000,
                             0x4B800000, 0x4B000000, 0x3DCCCCCD, 0x33800000, 0x5F000000, 0x1F800000, 0x40400000,
                             0x40A00000, 0xC0A00000, 0x42FE0000, 0xC2FE0000, 0x43000000};
  return v;
}
std::vector<uint64_t> edge64() {
  std::vector<uint64_t> v = {0, 1ull << 63, 1, (1ull << 63) | 1, 0x000FFFFFFFFFFFFFull, 0x0010000000000000ull, 0x3FF0000000000000ull,
                             0xBFF0000000000000ull, 0x3FE0000000000000ull, 0xBFE0000000000000ull, 0x4000000000000000ull,
                             0xC000000000000000ull, 0x400921FB54442D18ull, 0x7FEFFFFFFFFFFFFFull, 0xFFEFFFFFFFFFFFFFull,
                             0x7FF0000000000000ull, 0xFFF0000000000000ull, 0x7FF8000000000000ull, 0xFFF8000000000000ull,
                             0x3FF0000000000001ull, 0x3FEFFFFFFFFFFFFFull, 0x4024000000000000ull, 0x4059000000000000ull,
                             0xC024000000000000ull, 0x4008000000000000ull, 0x4014000000000000ull, 0xC014000000000000ull,
                             0x4086200000000000ull, 0xC086200000000000ull, 0x3E70000000000000ull, 0x7FF0000000000001ull};
  return v;
}
// Random f32 bit patterns, half of them confined to a moderate range so the functions do not all overflow.
std::vector<uint32_t> random32(uint32_t count, uint64_t seed) {
  Rng r(seed);
  std::vector<uint32_t> v;
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t x = static_cast<uint32_t>(r.next() >> 16);
    if (i & 1) {   // exponent in [-12, 12]
      const uint32_t e = 127 - 12 + static_cast<uint32_t>(r.next() % 25);
      x = (x & 0x807FFFFFu) | (e << 23);
    }
    v.push_back(x);
  }
  return v;
}
std::vector<uint64_t> random64(uint32_t count, uint64_t seed) {
  Rng r(seed);
  std::vector<uint64_t> v;
  for (uint32_t i = 0; i < count; ++i) {
    uint64_t x = r.next();
    if (i & 1) {
      const uint64_t e = 1023 - 12 + r.next() % 25;
      x = (x & 0x800FFFFFFFFFFFFFull) | (e << 52);
    }
    v.push_back(x);
  }
  return v;
}
const int kInts[] = {0, 1, -1, 2, -2, 3, -3, 7, -8, 10, 100, -100, 127, -127, 128, -149, 1000, -1000, 2147483647, -2147483647 - 1};

const LibmFn kAll[] = {LibmFn::Acos, LibmFn::Acosh, LibmFn::Asin, LibmFn::Asinh, LibmFn::Atan, LibmFn::Atan2, LibmFn::Atanh,
                       LibmFn::Cbrt, LibmFn::Cos, LibmFn::Cosh, LibmFn::Erf, LibmFn::Erfc, LibmFn::Exp, LibmFn::Exp2,
                       LibmFn::Exp10, LibmFn::Expm1, LibmFn::Fdim, LibmFn::Fmod, LibmFn::Hypot, LibmFn::Ilogb, LibmFn::Ldexp,
                       LibmFn::Lgamma, LibmFn::Log, LibmFn::Log2, LibmFn::Log10, LibmFn::Log1p, LibmFn::Logb,
                       LibmFn::Nextafter, LibmFn::Pow, LibmFn::Pown, LibmFn::Powr, LibmFn::Remainder, LibmFn::Sin,
                       LibmFn::Sinh, LibmFn::Tan, LibmFn::Tanh, LibmFn::Tgamma, LibmFn::Round};

// The expected f32 result for the functions the evaluator is specified to compute in a way libm names.
uint64_t expect32(LibmFn fn, uint32_t a, uint32_t b) {
  const float x = F(a), y = F(b);
  const int n = static_cast<int32_t>(b);
  float r = 0;
  switch (fn) {
    case LibmFn::Fdim: r = std::isnan(x) || std::isnan(y) ? std::nanf("") : (x > y ? x - y : 0.0f); break;
    case LibmFn::Fmod: r = std::fmod(x, y); break;
    case LibmFn::Remainder: r = std::remainder(x, y); break;
    case LibmFn::Ldexp: r = std::ldexp(x, n); break;
    case LibmFn::Nextafter: r = std::nextafter(x, y); break;
    case LibmFn::Round: r = std::round(x); break;
    case LibmFn::Logb: r = std::logb(x); break;
    case LibmFn::Ilogb:
      return static_cast<uint32_t>(std::isnan(x) || std::isinf(x) ? INT32_MAX : x == 0 ? INT32_MIN : std::ilogb(x));
    case LibmFn::Powr: r = static_cast<float>(ref_powr(x, y)); break;
    default: r = static_cast<float>(ref_d(fn, x, y, n)); break;
  }
  return std::isnan(r) ? kNaN32 : B(r);
}
uint64_t expect64(LibmFn fn, uint64_t a, uint64_t b) {
  const double x = D(a), y = D(b);
  const int n = static_cast<int32_t>(b);
  double r = 0;
  switch (fn) {
    case LibmFn::Fdim: r = std::isnan(x) || std::isnan(y) ? std::nan("") : (x > y ? x - y : 0.0); break;
    case LibmFn::Fmod: r = std::fmod(x, y); break;
    case LibmFn::Remainder: r = std::remainder(x, y); break;
    case LibmFn::Ldexp: r = std::ldexp(x, n); break;
    case LibmFn::Nextafter: r = std::nextafter(x, y); break;
    case LibmFn::Round: r = std::round(x); break;
    case LibmFn::Logb: r = std::logb(x); break;
    case LibmFn::Ilogb:
      return static_cast<uint32_t>(std::isnan(x) || std::isinf(x) ? INT32_MAX : x == 0 ? INT32_MIN : std::ilogb(x));
    case LibmFn::Powr: r = ref_powr(x, y); break;
    default: r = ref_d(fn, x, y, n); break;
  }
  return std::isnan(r) ? kNaN64 : DB(r);
}

}  // namespace

// The names: 38 functions, the 37 of the OpenCL.std table of host-library math plus round, none repeated.
VTEST(every_function_has_its_opencl_name) {
  VCHECK_EQ(vgpu::ptx::kLibmFnCount, 38u);
  VCHECK_EQ(sizeof(kAll) / sizeof(kAll[0]), size_t{38});
  std::vector<std::string> seen;
  for (LibmFn fn : kAll) {
    const std::string nm = vgpu::exec::libm_name(fn);
    for (const auto& s : seen) VCHECK(s != nm);
    seen.push_back(nm);
  }
  VCHECK_EQ(std::string(vgpu::exec::libm_name(LibmFn::Exp10)), std::string("exp10"));
  VCHECK_EQ(std::string(vgpu::exec::libm_name(LibmFn::Round)), std::string("round"));
  VCHECK(vgpu::exec::libm_binary(LibmFn::Atan2) && vgpu::exec::libm_binary(LibmFn::Ldexp) && !vgpu::exec::libm_binary(LibmFn::Sin));
  VCHECK(vgpu::exec::libm_int_operand(LibmFn::Pown) && vgpu::exec::libm_int_result(LibmFn::Ilogb));
}

// f32: edge values (all pairs) and random patterns, bit for bit with the host function; every NaN is canonical.
VTEST(f32_matches_the_host_libm_bit_for_bit) {
  const auto edge = edge32();
  const auto rnd = random32(3000, 11);
  uint64_t checked = 0;
  for (LibmFn fn : kAll) {
    const bool bin = vgpu::exec::libm_binary(fn), ints = vgpu::exec::libm_int_operand(fn);
    auto one = [&](uint32_t a, uint32_t b) {
      const uint64_t got = libm_eval(fn, 32, a, b);
      const uint64_t want = expect32(fn, a, b);
      if (got != want)
        throw vtest::Failure(std::string("f32 ") + vgpu::exec::libm_name(fn) + "(" + std::to_string(a) + ", " + std::to_string(b) +
                             "): got " + std::to_string(got) + " want " + std::to_string(want));
      ++checked;
    };
    for (uint32_t a : edge) {
      if (!bin) one(a, 0);
      else if (ints) for (int n : kInts) one(a, static_cast<uint32_t>(n));
      else for (uint32_t b : edge) one(a, b);
    }
    for (size_t i = 0; i < rnd.size(); ++i) {
      if (!bin) one(rnd[i], 0);
      else if (ints) one(rnd[i], static_cast<uint32_t>(kInts[i % 20]));
      else one(rnd[i], rnd[(i * 7 + 1) % rnd.size()]);
    }
  }
  VCHECK(checked > 100000);
}

VTEST(f64_matches_the_host_libm_bit_for_bit) {
  const auto edge = edge64();
  const auto rnd = random64(3000, 12);
  uint64_t checked = 0;
  for (LibmFn fn : kAll) {
    const bool bin = vgpu::exec::libm_binary(fn), ints = vgpu::exec::libm_int_operand(fn);
    auto one = [&](uint64_t a, uint64_t b) {
      const uint64_t got = libm_eval(fn, 64, a, b);
      const uint64_t want = expect64(fn, a, b);
      if (got != want)
        throw vtest::Failure(std::string("f64 ") + vgpu::exec::libm_name(fn) + "(" + std::to_string(a) + ", " + std::to_string(b) +
                             "): got " + std::to_string(got) + " want " + std::to_string(want));
      ++checked;
    };
    for (uint64_t a : edge) {
      if (!bin) one(a, 0);
      else if (ints) for (int n : kInts) one(a, static_cast<uint32_t>(n));
      else for (uint64_t b : edge) one(a, b);
    }
    for (size_t i = 0; i < rnd.size(); ++i) {
      if (!bin) one(rnd[i], 0);
      else if (ints) one(rnd[i], static_cast<uint32_t>(kInts[i % 20]));
      else one(rnd[i], rnd[(i * 7 + 1) % rnd.size()]);
    }
  }
  VCHECK(checked > 100000);
}

// The values a correctly rounded implementation must return, which do not depend on which libm is underneath.
VTEST(known_values) {
  auto f32of = [](LibmFn fn, float x, float y = 0) { return static_cast<uint32_t>(libm_eval(fn, 32, B(x), B(y))); };
  VCHECK_EQ(f32of(LibmFn::Exp, 0.0f), B(1.0f));
  VCHECK_EQ(f32of(LibmFn::Exp, 1.0f), 0x402DF854u);          // e
  VCHECK_EQ(f32of(LibmFn::Exp2, 10.0f), B(1024.0f));
  VCHECK_EQ(f32of(LibmFn::Exp10, 3.0f), B(1000.0f));
  VCHECK_EQ(f32of(LibmFn::Log2, 8.0f), B(3.0f));
  VCHECK_EQ(f32of(LibmFn::Log10, 1000.0f), B(3.0f));
  VCHECK_EQ(f32of(LibmFn::Cbrt, 27.0f), B(3.0f));
  VCHECK_EQ(f32of(LibmFn::Cbrt, -8.0f), B(-2.0f));
  VCHECK_EQ(f32of(LibmFn::Hypot, 3.0f, 4.0f), B(5.0f));
  VCHECK_EQ(f32of(LibmFn::Pow, 2.0f, 10.0f), B(1024.0f));
  VCHECK_EQ(f32of(LibmFn::Tgamma, 5.0f), B(24.0f));
  VCHECK_EQ(f32of(LibmFn::Lgamma, 1.0f), B(0.0f));
  VCHECK_EQ(f32of(LibmFn::Erf, 0.0f), B(0.0f));
  VCHECK_EQ(f32of(LibmFn::Atan2, 1.0f, 1.0f), 0x3F490FDBu);  // pi/4
  VCHECK_EQ(f32of(LibmFn::Acos, -1.0f), 0x40490FDBu);        // pi
  VCHECK_EQ(f32of(LibmFn::Asin, 1.0f), 0x3FC90FDBu);         // pi/2
  VCHECK_EQ(f32of(LibmFn::Cosh, 0.0f), B(1.0f));
  VCHECK_EQ(f32of(LibmFn::Sinh, -0.0f), B(-0.0f));
  VCHECK_EQ(f32of(LibmFn::Tanh, 0.0f), B(0.0f));
  VCHECK_EQ(f32of(LibmFn::Log1p, 0.0f), B(0.0f));
  VCHECK_EQ(f32of(LibmFn::Expm1, -0.0f), B(-0.0f));
  VCHECK_EQ(libm_eval(LibmFn::Exp, 64, DB(0.0), 0), DB(1.0));
  VCHECK_EQ(libm_eval(LibmFn::Exp, 64, DB(1.0), 0), 0x4005BF0A8B145769ull);   // e
}

// The exact operations, against their definitions rather than a reference implementation.
VTEST(exact_operations_follow_their_definitions) {
  const auto rnd = random32(20000, 21);
  for (size_t i = 0; i + 1 < rnd.size(); ++i) {
    const float x = F(rnd[i]), y = F(rnd[i + 1]);
    // fmod and remainder are exact in any precision: the double computation gives the same value
    if (std::isfinite(x) && std::isfinite(y) && y != 0) {
      const float r = F(static_cast<uint32_t>(libm_eval(LibmFn::Fmod, 32, rnd[i], rnd[i + 1])));
      VCHECK(std::fabs(r) < std::fabs(y));
      VCHECK(r == 0 || std::signbit(r) == std::signbit(x));
      VCHECK(static_cast<double>(r) == std::fmod(static_cast<double>(x), static_cast<double>(y)));
      const float rr = F(static_cast<uint32_t>(libm_eval(LibmFn::Remainder, 32, rnd[i], rnd[i + 1])));
      VCHECK(std::fabs(static_cast<double>(rr)) <= std::fabs(static_cast<double>(y)) / 2);
      VCHECK(static_cast<double>(rr) == std::remainder(static_cast<double>(x), static_cast<double>(y)));
    }
    // nextafter steps the bit pattern by one toward y
    if (!std::isnan(x) && !std::isnan(y) && x != y && x != 0) {
      const uint32_t got = static_cast<uint32_t>(libm_eval(LibmFn::Nextafter, 32, rnd[i], rnd[i + 1]));
      const bool up = y > x;
      const bool pos = x > 0;
      const uint32_t want = (up == pos) ? rnd[i] + 1 : rnd[i] - 1;
      VCHECK_EQ(got, want);
    }
    // round: nearest integer, halfway away from zero
    if (std::isfinite(x)) {
      const float r = F(static_cast<uint32_t>(libm_eval(LibmFn::Round, 32, rnd[i], 0)));
      const double ax = std::fabs(static_cast<double>(x)), fl = std::floor(ax), fr = ax - fl;
      const double want = fr >= 0.5 ? fl + 1 : fl;
      VCHECK(std::fabs(static_cast<double>(r)) == want);
      VCHECK(std::signbit(r) == std::signbit(x));
    }
    // ldexp scales exactly
    if (std::isfinite(x) && x != 0) {
      const int n = static_cast<int>(i % 41) - 20;
      const float r = F(static_cast<uint32_t>(libm_eval(LibmFn::Ldexp, 32, rnd[i], static_cast<uint32_t>(n))));
      const double want = std::ldexp(static_cast<double>(x), n);
      VCHECK(static_cast<double>(r) == static_cast<double>(static_cast<float>(want)));
    }
    // ilogb / logb: the unbiased exponent, also for subnormals
    if (std::isfinite(x) && x != 0) {
      int e;
      std::frexp(static_cast<double>(x), &e);
      VCHECK_EQ(static_cast<int32_t>(libm_eval(LibmFn::Ilogb, 32, rnd[i], 0)), e - 1);
      VCHECK_EQ(F(static_cast<uint32_t>(libm_eval(LibmFn::Logb, 32, rnd[i], 0))), static_cast<float>(e - 1));
    }
  }
}

// OpenCL's special cases for the functions whose C counterparts differ or are not named in C.
VTEST(opencl_special_cases) {
  const float inf = std::numeric_limits<float>::infinity(), nan = std::nanf("");
  auto r32 = [](LibmFn fn, float x, float y = 0) { return static_cast<uint32_t>(libm_eval(fn, 32, B(x), B(y))); };
  auto ri = [](LibmFn fn, float x, int n) { return static_cast<uint32_t>(libm_eval(fn, 32, B(x), static_cast<uint32_t>(n))); };
  // powr: NaN for a negative base, 0^0, inf^0, 1^inf and a NaN operand; a zero base is +0
  VCHECK_EQ(r32(LibmFn::Powr, -2.0f, 2.0f), 0x7FFFFFFFu);
  VCHECK_EQ(r32(LibmFn::Powr, 0.0f, 0.0f), 0x7FFFFFFFu);
  VCHECK_EQ(r32(LibmFn::Powr, inf, 0.0f), 0x7FFFFFFFu);
  VCHECK_EQ(r32(LibmFn::Powr, 1.0f, inf), 0x7FFFFFFFu);
  VCHECK_EQ(r32(LibmFn::Powr, 1.0f, nan), 0x7FFFFFFFu);
  VCHECK_EQ(r32(LibmFn::Powr, nan, 2.0f), 0x7FFFFFFFu);
  VCHECK_EQ(r32(LibmFn::Powr, -0.0f, 3.0f), B(0.0f));
  VCHECK_EQ(r32(LibmFn::Powr, 0.0f, -1.0f), B(inf));
  VCHECK_EQ(r32(LibmFn::Powr, 4.0f, 0.5f), B(2.0f));
  VCHECK_EQ(r32(LibmFn::Powr, 5.0f, 0.0f), B(1.0f));
  // pow keeps C's cases: pow(x, 0) = 1 even for NaN, pow(1, y) = 1 even for NaN, a negative base with a non-integer exponent is NaN
  VCHECK_EQ(r32(LibmFn::Pow, nan, 0.0f), B(1.0f));
  VCHECK_EQ(r32(LibmFn::Pow, 1.0f, nan), B(1.0f));
  VCHECK_EQ(r32(LibmFn::Pow, -8.0f, 1.0f / 3.0f), 0x7FFFFFFFu);
  VCHECK_EQ(r32(LibmFn::Pow, -2.0f, 3.0f), B(-8.0f));
  VCHECK_EQ(r32(LibmFn::Pow, -0.0f, -3.0f), B(-inf));
  // pown: an integer exponent; pown(x, 0) = 1 for any x, a negative base with an odd exponent stays negative
  VCHECK_EQ(ri(LibmFn::Pown, nan, 0), B(1.0f));
  VCHECK_EQ(ri(LibmFn::Pown, -2.0f, 3), B(-8.0f));
  VCHECK_EQ(ri(LibmFn::Pown, -2.0f, 4), B(16.0f));
  VCHECK_EQ(ri(LibmFn::Pown, 2.0f, -2), B(0.25f));
  VCHECK_EQ(ri(LibmFn::Pown, -0.0f, -3), B(-inf));
  // ilogb: the OpenCL values, not the platform's
  VCHECK_EQ(ri(LibmFn::Ilogb, 0.0f, 0), 0x80000000u);
  VCHECK_EQ(ri(LibmFn::Ilogb, -0.0f, 0), 0x80000000u);
  VCHECK_EQ(ri(LibmFn::Ilogb, inf, 0), 0x7FFFFFFFu);
  VCHECK_EQ(ri(LibmFn::Ilogb, -inf, 0), 0x7FFFFFFFu);
  VCHECK_EQ(ri(LibmFn::Ilogb, nan, 0), 0x7FFFFFFFu);
  VCHECK_EQ(ri(LibmFn::Ilogb, 8.0f, 0), 3u);
  VCHECK_EQ(ri(LibmFn::Ilogb, 0.75f, 0), 0xFFFFFFFFu);          // -1
  VCHECK_EQ(ri(LibmFn::Ilogb, F(1), 0), 0xFFFFFF6Bu);            // the smallest subnormal: -149
  VCHECK_EQ(libm_eval(LibmFn::Ilogb, 64, DB(0.0), 0), 0x80000000u);
  VCHECK_EQ(libm_eval(LibmFn::Ilogb, 64, 1, 0), static_cast<uint32_t>(-1074));
  // fdim: x - y if x > y, else +0, NaN if either is NaN
  VCHECK_EQ(r32(LibmFn::Fdim, 5.0f, 3.0f), B(2.0f));
  VCHECK_EQ(r32(LibmFn::Fdim, 3.0f, 5.0f), B(0.0f));
  VCHECK_EQ(r32(LibmFn::Fdim, -0.0f, 0.0f), B(0.0f));
  VCHECK_EQ(r32(LibmFn::Fdim, nan, 1.0f), 0x7FFFFFFFu);
  VCHECK_EQ(r32(LibmFn::Fdim, inf, inf), B(0.0f));
  // zeros keep their sign where the function is odd; signed-zero results of the rest
  VCHECK_EQ(r32(LibmFn::Sin, -0.0f), B(-0.0f));
  VCHECK_EQ(r32(LibmFn::Tan, -0.0f), B(-0.0f));
  VCHECK_EQ(r32(LibmFn::Atan, -0.0f), B(-0.0f));
  VCHECK_EQ(r32(LibmFn::Asinh, -0.0f), B(-0.0f));
  VCHECK_EQ(r32(LibmFn::Cbrt, -0.0f), B(-0.0f));
  VCHECK_EQ(r32(LibmFn::Fmod, -3.0f, 3.0f), B(-0.0f));
  VCHECK_EQ(r32(LibmFn::Log, 0.0f), B(-inf));
  VCHECK_EQ(r32(LibmFn::Log, -1.0f), 0x7FFFFFFFu);
  VCHECK_EQ(r32(LibmFn::Tgamma, -1.0f), 0x7FFFFFFFu);
  VCHECK_EQ(r32(LibmFn::Tgamma, 0.0f), B(inf));
  VCHECK_EQ(r32(LibmFn::Tgamma, -0.0f), B(-inf));
  VCHECK_EQ(r32(LibmFn::Lgamma, 0.0f), B(inf));
  VCHECK_EQ(r32(LibmFn::Exp, inf), B(inf));
  VCHECK_EQ(r32(LibmFn::Exp, -inf), B(0.0f));
  VCHECK_EQ(r32(LibmFn::Exp, 89.0f), B(inf));
  VCHECK_EQ(r32(LibmFn::Exp, -104.0f), B(0.0f));
  VCHECK_EQ(r32(LibmFn::Acos, 2.0f), 0x7FFFFFFFu);
  VCHECK_EQ(r32(LibmFn::Atanh, 1.0f), B(inf));
  // every kind of NaN result is the canonical one, whatever the host's sign or payload
  const uint64_t nans64[] = {0x7FF8000000000000ull, 0xFFF8000000000000ull, 0x7FF0000000000001ull, 0xFFFFFFFFFFFFFFFFull,
                             0x7FF123456789ABCDull};
  for (LibmFn fn : kAll) {
    if (fn == LibmFn::Ilogb || fn == LibmFn::Pow || fn == LibmFn::Pown) continue;   // pow(NaN, 0) = 1, pown(NaN, 0) = 1
    const bool ints = vgpu::exec::libm_int_operand(fn);
    for (uint32_t nb : {0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0xFFFFFFFFu, 0x7FA12345u})
      VCHECK_EQ(libm_eval(fn, 32, nb, ints ? 1u : nb), kNaN32);
    for (uint64_t nb : nans64) VCHECK_EQ(libm_eval(fn, 64, nb, ints ? 1u : nb), kNaN64);
    for (uint32_t nb : {0x7E00u, 0xFE00u, 0x7C01u, 0xFFFFu}) VCHECK_EQ(libm_eval(fn, 16, nb, ints ? 1u : nb), kNaN16);
  }
  VCHECK_EQ(libm_eval(LibmFn::Sin, 32, 0xFFFFFFFFu, 0), kNaN32);
  VCHECK_EQ(libm_eval(LibmFn::Log, 64, DB(-1.0), 0), kNaN64);
  VCHECK_EQ(libm_eval(LibmFn::Log, 16, 0xBC00, 0), kNaN16);   // log(-1) in half
}

// .ftz: f32 subnormal operands and results to zero of the same sign.
VTEST(ftz_flushes_subnormals) {
  const uint32_t sub = 0x00400000u, nsub = 0x80400000u;
  VCHECK_EQ(libm_eval(LibmFn::Sin, 32, sub, 0, false), sub);                 // sin(x) = x for tiny x
  VCHECK_EQ(libm_eval(LibmFn::Sin, 32, sub, 0, true), 0u);
  VCHECK_EQ(libm_eval(LibmFn::Sin, 32, nsub, 0, true), 0x80000000u);
  VCHECK_EQ(libm_eval(LibmFn::Exp, 32, sub, 0, true), B(1.0f));
  VCHECK_EQ(libm_eval(LibmFn::Ldexp, 32, B(1.0f), static_cast<uint32_t>(-127), true), 0u);            // a subnormal result
  VCHECK_EQ(libm_eval(LibmFn::Ldexp, 32, B(-1.0f), static_cast<uint32_t>(-127), true), 0x80000000u);
  VCHECK_EQ(libm_eval(LibmFn::Ldexp, 32, B(1.0f), static_cast<uint32_t>(-127), false), 0x00400000u);
  VCHECK_EQ(libm_eval(LibmFn::Fmod, 32, sub, B(3.0f), true), 0u);
  VCHECK_EQ(libm_eval(LibmFn::Ilogb, 32, sub, 0, true), 0x80000000u);       // flushed to zero first
  VCHECK_EQ(libm_eval(LibmFn::Ilogb, 32, sub, 0, false), static_cast<uint32_t>(-127));
}

// f16: every input of every unary function, and a grid for the binary ones, against the f32 result rounded once to half
// with a converter written here.
VTEST(f16_is_the_f32_result_rounded_once) {
  uint64_t checked = 0;
  for (LibmFn fn : kAll) {
    const bool bin = vgpu::exec::libm_binary(fn), ints = vgpu::exec::libm_int_operand(fn);
    if (fn == LibmFn::Nextafter) continue;   // below
    auto want_of = [&](uint32_t a, uint32_t b) -> uint64_t {
      const float x = static_cast<float>(half_value(a));
      const float y = ints ? 0.0f : static_cast<float>(half_value(b));
      const uint64_t r32 = expect32(fn, B(x), ints ? b : B(y));
      if (fn == LibmFn::Ilogb) return r32;
      const float r = F(static_cast<uint32_t>(r32));
      return std::isnan(r) ? kNaN16 : half_from(static_cast<double>(r));
    };
    if (!bin) {
      for (uint32_t a = 0; a < 65536; ++a) {
        const uint64_t got = libm_eval(fn, 16, a, 0), want = want_of(a, 0);
        if (got != want)
          throw vtest::Failure(std::string("f16 ") + vgpu::exec::libm_name(fn) + "(" + std::to_string(a) + "): got " +
                               std::to_string(got) + " want " + std::to_string(want));
        ++checked;
      }
    } else {
      for (uint32_t a = 0; a < 65536; a += 97)
        for (uint32_t k = 0; k < 40; ++k) {
          const uint32_t b = ints ? static_cast<uint32_t>(kInts[k % 20]) : (k * 1597u + 13u) & 0xFFFFu;
          const uint64_t got = libm_eval(fn, 16, a, b), want = want_of(a, b);
          if (got != want)
            throw vtest::Failure(std::string("f16 ") + vgpu::exec::libm_name(fn) + "(" + std::to_string(a) + ", " +
                                 std::to_string(b) + "): got " + std::to_string(got) + " want " + std::to_string(want));
          ++checked;
        }
    }
  }
  VCHECK(checked > 1000000);
  // nextafter steps in half precision
  VCHECK_EQ(libm_eval(LibmFn::Nextafter, 16, 0x3C00, 0x4000), 0x3C01u);   // 1 -> up
  VCHECK_EQ(libm_eval(LibmFn::Nextafter, 16, 0x3C00, 0x0000), 0x3BFFu);   // 1 -> down
  VCHECK_EQ(libm_eval(LibmFn::Nextafter, 16, 0x0000, 0x3C00), 0x0001u);   // 0 -> smallest subnormal
  VCHECK_EQ(libm_eval(LibmFn::Nextafter, 16, 0x0000, 0xBC00), 0x8001u);
  VCHECK_EQ(libm_eval(LibmFn::Nextafter, 16, 0x7BFF, 0x7C00), 0x7C00u);   // max -> infinity
  VCHECK_EQ(libm_eval(LibmFn::Nextafter, 16, 0xBC00, 0xBC00), 0xBC00u);
  VCHECK_EQ(libm_eval(LibmFn::Nextafter, 16, 0x7E00, 0x3C00), kNaN16);
  // the half conversion itself, against exhaustive round trips: every half value is its own result for round()
  for (uint32_t a = 0; a < 65536; ++a) {
    const double v = half_value(a);
    if (std::isnan(v) || std::fabs(v) < 1024) continue;
    VCHECK_EQ(libm_eval(LibmFn::Round, 16, a, 0), static_cast<uint64_t>(a));   // integers stay put
  }
}

// ---- the instruction through the interpreter -----------------------------------------------------------------------

namespace {
// out[i] = libm(fn, ty)(a[i], b[i]) for i < n; a and b are `bits`-wide values (b is a 32-bit integer for ldexp/pown).
std::vector<uint8_t> run_libm(LibmFn fn, uint32_t bits, bool ftz, uint32_t W, uint32_t n, const std::vector<uint64_t>& a,
                              const std::vector<uint64_t>& b) {
  const Type fty = tf(bits);
  const bool bin = vgpu::exec::libm_binary(fn), ints = vgpu::exec::libm_int_operand(fn), ires = vgpu::exec::libm_int_result(fn);
  const Type bty = ints ? ts(32) : fty, rty = ires ? ts(32) : fty;
  vgpu::ptx::KernelBuilder kb("libm");
  Prologue p = prologue(kb, 3);
  const Reg ok = kb.pred();
  const Reg nreg = kb.temp(tu(32));
  kb.emit(mov(tu(32), nreg, IM(n)), "mov");
  kb.emit(setp(CmpOp::Ge, tu(32), ok, RO(p.gid), RO(nreg)), "setp");
  const vgpu::ptx::Label done = kb.make_label();
  kb.bra(done, ok);
  const Reg x = load_elem(kb, p.ptr[0], p.gid, fty);
  Reg y;
  if (bin) y = load_elem(kb, p.ptr[1], p.gid, bty);
  const Reg r = kb.temp(rty);
  OpLibm op;
  op.fn = fn;
  op.ty = fty;
  op.ftz = ftz;
  op.dst = r;
  op.a = RO(x);
  if (bin) op.b = RO(y);
  kb.emit(op, "libm");
  store_elem(kb, p.ptr[2], p.gid, rty, r);
  kb.bind(done);
  kb.emit(OpRet{}, "ret");

  auto pack = [&](const std::vector<uint64_t>& v, uint32_t w) {
    std::vector<uint8_t> o(v.size() * (w / 8));
    for (size_t i = 0; i < v.size(); ++i) std::memcpy(&o[i * (w / 8)], &v[i], w / 8);
    return o;
  };
  vgpu::exec::LaunchConfig cfg;
  cfg.block = {40, 1, 1};     // not a whole number of warps at 8 or 32 lanes
  cfg.grid = {(n + 39) / 40, 1, 1};
  cfg.warp_size = W;
  std::vector<uint8_t> out(static_cast<size_t>(n) * (ires ? 4 : bits / 8), 0xCD);
  auto res = run(kb.finish(), cfg, {pack(a, bits), pack(b, ints ? 32 : bits), out});
  return res[2];
}
}  // namespace

VTEST(the_instruction_matches_the_evaluator_at_every_width_and_warp_size) {
  const uint32_t n = 300;
  for (uint32_t bits : {16u, 32u, 64u}) {
    for (LibmFn fn : kAll) {
      const bool ints = vgpu::exec::libm_int_operand(fn), ires = vgpu::exec::libm_int_result(fn);
      std::vector<uint64_t> a(n), b(n);
      Rng r(1000 + static_cast<uint64_t>(fn) * 31 + bits);
      for (uint32_t i = 0; i < n; ++i) {
        if (bits == 16) a[i] = r.next() & 0xFFFF, b[i] = ints ? static_cast<uint32_t>(kInts[i % 20]) : (r.next() & 0xFFFF);
        else if (bits == 32) a[i] = i % 3 ? random32(1, r.next())[0] : edge32()[i % edge32().size()], b[i] = ints ? static_cast<uint32_t>(kInts[i % 20]) : random32(1, r.next())[0];
        else a[i] = i % 3 ? random64(1, r.next())[0] : edge64()[i % edge64().size()], b[i] = ints ? static_cast<uint32_t>(kInts[i % 20]) : random64(1, r.next())[0];
      }
      for (bool ftz : {false, true}) {
        if (ftz && bits != 32) continue;
        for (uint32_t W : {8u, 32u}) {
          const auto out = run_libm(fn, bits, ftz, W, n, a, b);
          const uint32_t w = ires ? 32 : bits;
          for (uint32_t i = 0; i < n; ++i) {
            uint64_t got = 0;
            std::memcpy(&got, &out[static_cast<size_t>(i) * (w / 8)], w / 8);
            const uint64_t want = libm_eval(fn, bits, a[i], b[i], ftz);
            if (got != want)
              throw vtest::Failure(std::string("OpLibm ") + vgpu::exec::libm_name(fn) + " f" + std::to_string(bits) + " W=" +
                                   std::to_string(W) + " lane " + std::to_string(i) + ": got " + std::to_string(got) + " want " +
                                   std::to_string(want));
          }
        }
      }
    }
  }
}

// An odd launch: a guarded OpLibm (only some lanes of a warp take it) leaves the others' register alone.
VTEST(a_guarded_libm_writes_only_the_lanes_that_take_it) {
  vgpu::ptx::KernelBuilder kb("g");
  Prologue p = prologue(kb, 2);
  const Reg x = load_elem(kb, p.ptr[0], p.gid, tf(32));
  const Reg r = kb.temp(tf(32));
  kb.emit(mov(tf(32), r, F32(-5.0f)), "mov");
  const Reg odd = kb.temp(tu(32));
  kb.emit(ibin(IntBinOp::And, tu(32), odd, RO(p.gid), IM(1)), "and");
  const Reg q = kb.pred();
  kb.emit(setp(CmpOp::Ne, tu(32), q, RO(odd), IM(0)), "setp");
  OpLibm op;
  op.fn = LibmFn::Exp2;
  op.ty = tf(32);
  op.dst = r;
  op.a = RO(x);
  kb.emit_if(q, false, op, "libm");
  store_elem(kb, p.ptr[1], p.gid, tf(32), r);
  kb.emit(OpRet{}, "ret");
  std::vector<float> in(64);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i) * 0.25f;
  vgpu::exec::LaunchConfig cfg;
  cfg.block = {64, 1, 1};
  cfg.warp_size = 16;
  auto res = run(kb.finish(), cfg, {bytes_of(in), std::vector<uint8_t>(64 * 4, 0)});
  const auto out = from_bytes<float>(res[1]);
  for (size_t i = 0; i < in.size(); ++i) VCHECK_EQ(out[i], (i & 1) ? static_cast<float>(std::exp2(static_cast<double>(in[i]))) : -5.0f);
}

VTEST_MAIN
