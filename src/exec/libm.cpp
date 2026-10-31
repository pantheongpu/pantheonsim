// OpLibm's evaluator: see include/vgpu/exec/libm.hpp for what is called and what is promised.
#include "vgpu/exec/libm.hpp"

#include <bit>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>

namespace vgpu::exec {

using ptx::LibmFn;

namespace {

constexpr uint64_t kNaN16 = 0x7FFF;
constexpr uint64_t kNaN32 = 0x7FFFFFFFull;
constexpr uint64_t kNaN64 = 0x7FFFFFFFFFFFFFFFull;

inline float f32of(uint64_t v) { return std::bit_cast<float>(static_cast<uint32_t>(v)); }
inline double f64of(uint64_t v) { return std::bit_cast<double>(v); }

// binary16 -> binary32, exact.
float half_to_float(uint32_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
  uint32_t out;
  if (e == 0x1F) {
    out = sign | 0x7F800000u | (m << 13);
  } else if (e != 0) {
    out = sign | ((e + 112) << 23) | (m << 13);
  } else if (m == 0) {
    out = sign;
  } else {
    // a subnormal half: normalise it
    int sh = 0;
    while (!(m & 0x400u)) { m <<= 1; ++sh; }
    out = sign | ((113u - static_cast<uint32_t>(sh)) << 23) | ((m & 0x3FFu) << 13);
  }
  return std::bit_cast<float>(out);
}

// binary32 -> binary16, round to nearest even; every NaN is the canonical one.
uint32_t float_to_half(float f) {
  const uint32_t x = std::bit_cast<uint32_t>(f);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t mag = x & 0x7FFFFFFFu;
  if (mag > 0x7F800000u) return static_cast<uint32_t>(kNaN16);
  if (mag >= 0x477FF000u) return sign | 0x7C00u;     // 65520 and up rounds to infinity
  if (mag <= 0x33000000u) return sign;               // 2^-25 and below: half of the smallest subnormal ties to 0
  if (mag < 0x38800000u) {                           // below 2^-14: a subnormal half
    const uint32_t e = mag >> 23;
    const uint32_t m = (mag & 0x7FFFFFu) | 0x800000u;
    const uint32_t shift = 126u - e;                 // 14 .. 24
    uint32_t r = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (r & 1u))) ++r;
    return sign | r;
  }
  const uint32_t v = mag - 0x38000000u;
  uint32_t r = v >> 13;
  const uint32_t rem = v & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
  return sign | r;
}

// nextafter in half precision, on the bit pattern.
uint32_t nextafter_half(uint32_t a, uint32_t b) {
  const float fa = half_to_float(a), fb = half_to_float(b);
  if (std::isnan(fa) || std::isnan(fb)) return static_cast<uint32_t>(kNaN16);
  if (fa == fb) return b & 0xFFFFu;
  if (fa == 0.0f) return (b & 0x8000u) | 1u;          // the smallest subnormal toward b
  const bool grow = (fb > fa) == (fa > 0.0f);          // away from zero?
  return grow ? (a + 1u) & 0xFFFFu : (a - 1u) & 0xFFFFu;
}

int ilogb_ocl(double x) {
  if (std::isnan(x) || std::isinf(x)) return INT_MAX;
  if (x == 0.0) return INT_MIN;
  return std::ilogb(x);
}

double powr_d(double x, double y) {
  if (std::isnan(x) || std::isnan(y)) return std::numeric_limits<double>::quiet_NaN();
  if (x < 0.0) return std::numeric_limits<double>::quiet_NaN();
  if (x == 0.0 && y == 0.0) return std::numeric_limits<double>::quiet_NaN();
  if (std::isinf(x) && y == 0.0) return std::numeric_limits<double>::quiet_NaN();
  if (x == 1.0 && std::isinf(y)) return std::numeric_limits<double>::quiet_NaN();
  if (x == 0.0) x = 0.0;   // -0 is a zero base like +0
  return std::pow(x, y);
}

double lgamma_d(double x) {
#if defined(__GLIBC__) || defined(__APPLE__)
  int sign = 0;
  return ::lgamma_r(x, &sign);
#else
  return std::lgamma(x);
#endif
}

// The double-precision functions; `n` is the integer operand of Ldexp and Pown.
double eval_d(LibmFn fn, double x, double y, int n) {
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
    case LibmFn::Fdim: return std::isnan(x) || std::isnan(y) ? std::numeric_limits<double>::quiet_NaN() : (x > y ? x - y : 0.0);
    case LibmFn::Fmod: return std::fmod(x, y);
    case LibmFn::Hypot: return std::hypot(x, y);
    case LibmFn::Ilogb: return static_cast<double>(ilogb_ocl(x));
    case LibmFn::Ldexp: return std::ldexp(x, n);
    case LibmFn::Lgamma: return lgamma_d(x);
    case LibmFn::Log: return std::log(x);
    case LibmFn::Log2: return std::log2(x);
    case LibmFn::Log10: return std::log10(x);
    case LibmFn::Log1p: return std::log1p(x);
    case LibmFn::Logb: return std::logb(x);
    case LibmFn::Nextafter: return std::nextafter(x, y);
    case LibmFn::Pow: return std::pow(x, y);
    case LibmFn::Pown: return std::pow(x, static_cast<double>(n));
    case LibmFn::Powr: return powr_d(x, y);
    case LibmFn::Remainder: return std::remainder(x, y);
    case LibmFn::Sin: return std::sin(x);
    case LibmFn::Sinh: return std::sinh(x);
    case LibmFn::Tan: return std::tan(x);
    case LibmFn::Tanh: return std::tanh(x);
    case LibmFn::Tgamma: return std::tgamma(x);
    case LibmFn::Round: return std::round(x);
  }
  return 0.0;
}

// The single-precision functions: the ones exact at any width at float, the rest through double.
float eval_f(LibmFn fn, float x, float y, int n) {
  switch (fn) {
    case LibmFn::Fdim: return std::isnan(x) || std::isnan(y) ? std::numeric_limits<float>::quiet_NaN() : (x > y ? x - y : 0.0f);
    case LibmFn::Fmod: return std::fmod(x, y);
    case LibmFn::Remainder: return std::remainder(x, y);
    case LibmFn::Ldexp: return std::ldexp(x, n);
    case LibmFn::Nextafter: return std::nextafter(x, y);
    default: return static_cast<float>(eval_d(fn, static_cast<double>(x), static_cast<double>(y), n));
  }
}

inline float flush_f(float x) { return std::fpclassify(x) == FP_SUBNORMAL ? std::copysign(0.0f, x) : x; }

}  // namespace

bool libm_binary(LibmFn fn) {
  switch (fn) {
    case LibmFn::Atan2: case LibmFn::Fdim: case LibmFn::Fmod: case LibmFn::Hypot: case LibmFn::Ldexp:
    case LibmFn::Nextafter: case LibmFn::Pow: case LibmFn::Pown: case LibmFn::Powr: case LibmFn::Remainder:
      return true;
    default: return false;
  }
}
bool libm_int_operand(LibmFn fn) { return fn == LibmFn::Ldexp || fn == LibmFn::Pown; }
bool libm_int_result(LibmFn fn) { return fn == LibmFn::Ilogb; }

const char* libm_name(LibmFn fn) {
  static const char* const names[] = {"acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "cos", "cosh",
                                      "erf", "erfc", "exp", "exp2", "exp10", "expm1", "fdim", "fmod", "hypot", "ilogb",
                                      "ldexp", "lgamma", "log", "log2", "log10", "log1p", "logb", "nextafter", "pow",
                                      "pown", "powr", "remainder", "sin", "sinh", "tan", "tanh", "tgamma", "round"};
  static_assert(sizeof(names) / sizeof(names[0]) == ptx::kLibmFnCount);
  return names[static_cast<unsigned>(fn)];
}

uint64_t libm_eval(LibmFn fn, uint32_t bits, uint64_t a, uint64_t b, bool ftz) {
  const bool int_op = libm_int_operand(fn);
  const int n = int_op ? static_cast<int32_t>(static_cast<uint32_t>(b)) : 0;
  if (bits == 64) {
    const double x = f64of(a), y = int_op ? 0.0 : f64of(b);
    if (fn == LibmFn::Ilogb) return static_cast<uint32_t>(ilogb_ocl(x));
    const double r = eval_d(fn, x, y, n);
    return std::isnan(r) ? kNaN64 : std::bit_cast<uint64_t>(r);
  }
  const bool half = bits == 16;
  float x = half ? half_to_float(static_cast<uint32_t>(a) & 0xFFFFu) : f32of(a);
  float y = int_op ? 0.0f : half ? half_to_float(static_cast<uint32_t>(b) & 0xFFFFu) : f32of(b);
  if (ftz && !half) {
    x = flush_f(x);
    y = flush_f(y);
  }
  if (fn == LibmFn::Ilogb) return static_cast<uint32_t>(ilogb_ocl(static_cast<double>(x)));
  if (half && fn == LibmFn::Nextafter)
    return nextafter_half(static_cast<uint32_t>(a) & 0xFFFFu, static_cast<uint32_t>(b) & 0xFFFFu);
  float r = eval_f(fn, x, y, n);
  if (half) return std::isnan(r) ? kNaN16 : float_to_half(r);
  if (ftz) r = flush_f(r);
  return std::isnan(r) ? kNaN32 : std::bit_cast<uint32_t>(r);
}

}  // namespace vgpu::exec
