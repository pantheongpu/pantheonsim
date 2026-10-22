// The payload and sign of a NaN that comes out of an operation with two or three NaN operands is the operand order's: the
// hardware (and x86's SSE, and libm's fma) hand on the first NaN of their operand order, quieted. C++ leaves the order of the
// operands of + , * and fma() to the compiler, and GCC swaps them as it likes, differently at -O0, -O1 and -O3 and in two copies
// of one loop, so the NaN a plain `x + y` leaves was the compiler's choice, and the recorded digests of
// amd/tests/unit/test_gcn_same_bits.cpp held one compiler's. These give the result of the plain operation unless it is a NaN with a
// NaN operand, and then the first NaN operand in the order the template names, quieted: the same bits in every build.
//
//   Order for two operands: 0 = x then y, 1 = y then x. The order each call site uses is the one that reproduces the recorded digests
//   (found by trying them, as for the matrix instructions in mfma_math.hpp), so it is not the same at every site.
//   Order for three operands (fma): the permutation of (a, b, c): 0 abc, 1 acb, 2 bac, 3 bca, 4 cab, 5 cba.
#pragma once
#include <bit>
#include <cmath>
#include <cstdint>

namespace vgpu::amd {

template <typename T>
inline T nan_quieted(T x) {
  if constexpr (sizeof(T) == 4) return std::bit_cast<T>(std::bit_cast<uint32_t>(x) | 0x400000u);
  else return std::bit_cast<T>(std::bit_cast<uint64_t>(x) | (1ull << 51));
}

template <int Order, typename T>
inline T nan_add(T x, T y) {
  const T r = x + y;
  if (r == r) return r;
  if constexpr (Order == 0) return x != x ? nan_quieted(x) : y != y ? nan_quieted(y) : r;
  else return y != y ? nan_quieted(y) : x != x ? nan_quieted(x) : r;
}
template <int Order, typename T>
inline T nan_mul(T x, T y) {
  const T r = x * y;
  if (r == r) return r;
  if constexpr (Order == 0) return x != x ? nan_quieted(x) : y != y ? nan_quieted(y) : r;
  else return y != y ? nan_quieted(y) : x != x ? nan_quieted(x) : r;
}
template <int Order, typename T>
inline T nan_sub(T x, T y) {   // x - y; the NaN that comes out is an operand's as it is, not negated
  const T r = x - y;
  if (r == r) return r;
  if constexpr (Order == 0) return x != x ? nan_quieted(x) : y != y ? nan_quieted(y) : r;
  else return y != y ? nan_quieted(y) : x != x ? nan_quieted(x) : r;
}
template <int Order, typename T>
inline T nan_fma(T a, T b, T c) {
  const T r = std::fma(a, b, c);
  if (r == r) return r;
  static constexpr int kPerm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  const T v[3] = {a, b, c};
  for (int i : kPerm[Order])
    if (v[i] != v[i]) return nan_quieted(v[i]);
  return r;
}

// C's fmin and fmax as the libm of the CI image (glibc 2.39) gives them, which the recorded digests hold: the comparison where there
// is one (+0 with -0 gives the second operand); a signaling NaN beside a number gives that NaN, quieted; a quiet NaN beside a number
// gives the number; two NaNs give the first, quieted. GCC treats fmin and fmax as commutative and passes the operands to the library
// in whichever order it likes (differently at -O0 and -O3, and from one call to the next), so which of the two NaNs, or which zero,
// a plain std::fmax(x, y) gave was the compiler's choice. Order 1 here is the library's answer for (y, x) instead of (x, y).
template <typename T>
inline bool nan_signaling(T x) {
  if constexpr (sizeof(T) == 4) return x != x && !(std::bit_cast<uint32_t>(x) & 0x400000u);
  else return x != x && !(std::bit_cast<uint64_t>(x) & (1ull << 51));
}
template <typename T>
inline T libm_fmax(T a, T b) {
  if (a > b) return a;
  if (a < b) return b;
  if (a == b) return b;
  const bool an = a != a, bn = b != b;
  if (an && bn) return nan_quieted(a);
  if (nan_signaling(a)) return nan_quieted(a);
  if (nan_signaling(b)) return nan_quieted(b);
  return an ? b : a;
}
template <typename T>
inline T libm_fmin(T a, T b) {
  if (a < b) return a;
  if (a > b) return b;
  if (a == b) return b;
  const bool an = a != a, bn = b != b;
  if (an && bn) return nan_quieted(a);
  if (nan_signaling(a)) return nan_quieted(a);
  if (nan_signaling(b)) return nan_quieted(b);
  return an ? b : a;
}
template <int Order, typename T>
inline T nan_fmax(T x, T y) {
  if constexpr (Order & 1) return libm_fmax(y, x);
  else return libm_fmax(x, y);
}
template <int Order, typename T>
inline T nan_fmin(T x, T y) {
  if constexpr (Order & 1) return libm_fmin(y, x);
  else return libm_fmin(x, y);
}

}  // namespace vgpu::amd
