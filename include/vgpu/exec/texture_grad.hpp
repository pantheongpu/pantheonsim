// The level of detail an RTX 3060's texture unit derives from explicit gradients (tex.grad), bit for bit.
//
// Measured with a mipmapped float texture whose level k is the constant k, trilinear in the mip chain, so the
// fetched value is the LOD in 1/256ths of a level itself (gradprobe in the round-4 notes: ~5 million fetches of
// 1D and 2D gradients, power-of-two sizes). What the card does, in the units of texels of the base level:
//
//  1. Each gradient component is cut to 10 significant bits (the rest truncated toward zero): the "10-bit float".
//  2. A vector's length is approximated by the larger of its components plus 11/32 of the smaller: the smaller
//     one's 10-bit mantissa M (an integer 512..1023, the value M * 2^kb) is multiplied by 11/8 and rounded half up
//     to an integer, which counts in units of 2^(kb-2); the sum is truncated to 10 bits again.
//  3. The levels of detail come from four such lengths: those of dPdx and dPdy, and 11/16 of those of dPdx + dPdy
//     and dPdx - dPdy (the diagonals of the quad the gradients span), the largest of the four being taken. The
//     components of a diagonal are added as the card's adder does (the smaller operand is cut to one bit below
//     the larger's last bit before the add); 11/16 of a diagonal's larger component is round(11 M / 8) in units
//     of 2^(k-1) (truncated to 10 bits), and 11/16 of its smaller component's 11/32 share is round(121 M / 64)
//     in units of 2^(kb-3).
//  4. log2 of the result, as a 256-entry table indexed by the top 8 bits of the mantissa and added to 256 times
//     the exponent -- the table is not 256 log2: it is the card's (the same table for every size).
//
// 1D textures are 2D ones of height 1 (the gradients' second components are 0), as the hardware fetches them.
// A gradient is scaled by the texture's size first; that is exact for a power of two and is all that is
// reproduced (for other sizes the card's product is rounded in a way that no tried pipeline of truncations and
// roundings reproduces: about one fetch in ten differs by one or two 256ths), so other sizes are refused by the
// caller. Three components (3D, cube) are not reproduced either: the card's 3D length is max + 11/32 mid +
// 1/4 min to within a part in 1000, but its rounding was not recovered.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vgpu::exec::tex_grad {

inline constexpr uint16_t kLog2Table[256] = {
    0, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 18, 19, 20, 22,
    23, 24, 26, 27, 28, 30, 31, 32, 34, 35, 36, 38, 39, 40, 42, 43,
    44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 60, 62, 63,
    64, 65, 66, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 81, 82,
    83, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100,
    101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
    135, 136, 137, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 162, 163, 164,
    165, 166, 167, 168, 169, 170, 171, 172, 173, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 181, 182, 183, 184, 185, 186, 187, 188, 188, 189, 190, 191, 192, 193,
    194, 195, 195, 196, 197, 198, 199, 200, 200, 201, 202, 203, 204, 205, 205, 206,
    207, 208, 209, 210, 210, 211, 212, 213, 214, 214, 215, 216, 217, 218, 218, 219,
    220, 221, 222, 222, 223, 224, 225, 226, 226, 227, 228, 229, 229, 230, 231, 232,
    233, 233, 234, 235, 236, 236, 237, 238, 239, 239, 240, 241, 242, 242, 243, 244,
    245, 245, 246, 247, 248, 248, 249, 250, 251, 251, 252, 253, 253, 254, 255, 255,
};

namespace detail {

// Truncates |v| to `bits` significant bits (toward zero); v is finite.
inline double trunc_bits(double v, int bits) {
  if (v == 0) return 0;
  const int e = std::ilogb(v);
  const double s = std::ldexp(1.0, e - bits + 1);
  return std::floor(v / s) * s;
}
inline double t10(double v) { return std::copysign(trunc_bits(std::fabs(v), 10), v); }

// The 10-bit mantissa of a positive 10-bit value (an integer 512..1023) and the exponent of its last bit.
inline void mant10(double v, double* m, int* k) {
  *k = std::ilogb(v) - 9;
  *m = std::ldexp(v, -*k);
}
inline double round_half_up(double x) { return std::floor(x + 0.5); }

// Approximate length of the vector (u, v): larger + 11/32 smaller, on 10-bit values.
inline double length_unit(double u, double v) {
  const double au = trunc_bits(std::fabs(u), 10), av = trunc_bits(std::fabs(v), 10);
  const double a = std::max(au, av), b = std::min(au, av);
  double p = 0;
  if (b > 0) {
    double bm;
    int kb;
    mant10(b, &bm, &kb);
    p = std::ldexp(round_half_up(11.0 * bm / 8.0), kb - 2);
  }
  return trunc_bits(a + p, 10);
}

// 11/16 of a 10-bit value, as the card rounds it: round(11 M / 8) in units of 2^(k-1).
inline double scale_c(double s) {
  if (s <= 0) return 0;
  double m;
  int k;
  mant10(s, &m, &k);
  return std::ldexp(round_half_up(11.0 * m / 8.0), k - 1);
}

// 11/16 of the length of a diagonal (u, v).
inline double diagonal_unit(double u, double v) {
  const double au = trunc_bits(std::fabs(u), 10), av = trunc_bits(std::fabs(v), 10);
  const double a = std::max(au, av), b = std::min(au, av);
  const double big = trunc_bits(scale_c(a), 10);
  double p = 0;
  if (b > 0) {
    double bm;
    int kb;
    mant10(b, &bm, &kb);
    p = std::ldexp(round_half_up(121.0 * bm / 64.0), kb - 3);
  }
  return trunc_bits(big + p, 10);
}

// The card's adder on two 10-bit floats: the smaller operand is cut to one bit below the larger's last bit.
inline double add_signed(double x, double y) {
  const double big = std::fabs(x) >= std::fabs(y) ? x : y, small = std::fabs(x) >= std::fabs(y) ? y : x;
  if (big == 0) return 0;
  const double gg = std::ldexp(1.0, std::ilogb(big) - 10);
  return big + std::copysign(std::floor(std::fabs(small) / gg) * gg, small);
}

}  // namespace detail

// Sanitizes a gradient component the way the card reads it: NaN is 0, infinities are huge.
inline double component(float g) {
  if (std::isnan(g)) return 0;
  if (std::isinf(g)) return g > 0 ? 1e300 : -1e300;
  return static_cast<double>(g);
}

// 256 * log2 of a positive length, as the card's table gives it; the lowest int64 for a length of 0.
inline int64_t log2_q(double rho) {
  if (!(rho > 0)) return INT64_MIN / 4;
  const int e = std::ilogb(rho);
  const double m = std::ldexp(rho, -e);
  const int k = std::min(255, static_cast<int>(std::floor((m - 1.0) * 256.0)));
  return int64_t{256} * e + kLog2Table[k];
}

// The LOD in 1/256ths of a level for a 2D fetch (or a 1D one: v components 0) from the gradients in texels of the
// base level: dPdx = (dudx, dvdx), dPdy = (dudy, dvdy).
inline int64_t lod_q_2d(double dudx, double dvdx, double dudy, double dvdy) {
  using namespace detail;
  const double a1 = t10(dudx), a2 = t10(dvdx), a3 = t10(dudy), a4 = t10(dvdy);
  const double rho = std::max({length_unit(dudx, dvdx), length_unit(dudy, dvdy),
                               diagonal_unit(add_signed(a1, a3), add_signed(a2, a4)),
                               diagonal_unit(add_signed(a1, -a3), add_signed(a2, -a4))});
  return log2_q(rho);
}

}  // namespace vgpu::exec::tex_grad
