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
// A gradient is scaled by the texture's size first. For a power of two that is exact (an exponent change). For any
// other size the card multiplies the gradient's 10-bit significand by the size's significand (cut to 10
// significant bits) as a sum of shifted copies, each shifted-out bit dropped at 12 fractional bits, and converts
// the sum back to a 10-bit float with "add 7, then shift" -- by 3 bits when the sum is below 2, by 4 bits when it
// is 2 or more (see scale_by_size). Measured on an RTX 3060 against ~1.5 million fetches of 14 sizes from 3 to
// 16383, single components first and then every pair: no differences.
// Three components (3D, cube) are not reproduced: the card's 3D length is max + 11/32 mid + 1/4 min to within a
// part in 1000, but its rounding was not recovered.
#pragma once

#include <algorithm>
#include <bit>
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

// Sorts three magnitudes, largest first.
inline void sort3(double* v) {
  if (v[0] < v[1]) std::swap(v[0], v[1]);
  if (v[1] < v[2]) std::swap(v[1], v[2]);
  if (v[0] < v[1]) std::swap(v[0], v[1]);
}

// Approximate length of a vector of three components: larger + 11/32 middle + 1/4 smallest. The middle one's share
// is built as in length_unit and cut to 10 bits; the smallest's quarter is cut to the grid of that share's 10 bits
// before the three are added and cut again.
inline double length3(const double* v3) {
  double v[3] = {std::fabs(v3[0]), std::fabs(v3[1]), std::fabs(v3[2])};
  sort3(v);
  const double a = trunc_bits(v[0], 10), b = trunc_bits(v[1], 10), c = trunc_bits(v[2], 10);
  if (b == 0) return a;
  double bm;
  int kb;
  mant10(b, &bm, &kb);
  const double p = trunc_bits(std::ldexp(round_half_up(11.0 * bm / 8.0), kb - 2), 10);
  double q = 0;
  if (c > 0) {
    const double gr = std::ldexp(1.0, std::ilogb(p) - 9);
    q = std::floor(c / 4 / gr) * gr;
  }
  return trunc_bits(a + p + q, 10);
}

// 11/16 of the length of a vector of three components (a diagonal of the quad), the same way: each share is
// rounded from the 10-bit mantissa: round(11 M / 8) in units of 2^(k-1) for the largest, round(121 M / 64) in
// units of 2^(k-3) for the middle one's share, round(11 M / 8) in units of 2^(k-3) for the smallest's.
inline double diagonal3(const double* v3) {
  double v[3] = {trunc_bits(std::fabs(v3[0]), 10), trunc_bits(std::fabs(v3[1]), 10),
                 trunc_bits(std::fabs(v3[2]), 10)};
  sort3(v);
  if (v[0] == 0) return 0;
  double m;
  int k;
  mant10(v[0], &m, &k);
  const double ta = trunc_bits(std::ldexp(round_half_up(11.0 * m / 8.0), k - 1), 10);
  double tb = 0, tc = 0;
  if (v[1] > 0) {
    mant10(v[1], &m, &k);
    tb = trunc_bits(std::ldexp(round_half_up(121.0 * m / 64.0), k - 3), 10);
  }
  if (v[2] > 0) {
    mant10(v[2], &m, &k);
    tc = std::ldexp(round_half_up(11.0 * m / 8.0), k - 3);
    if (tb > 0) {
      const double gr = std::ldexp(1.0, std::ilogb(tb) - 9);
      tc = std::floor(tc / gr) * gr;
    }
  }
  return trunc_bits(ta + tb + tc, 10);
}

}  // namespace detail

// Sanitizes a gradient component the way the card reads it: NaN is 0, infinities are huge.
inline double component(float g) {
  if (std::isnan(g)) return 0;
  if (std::isinf(g)) return g > 0 ? 1e300 : -1e300;
  return static_cast<double>(g);
}

// A gradient component (a double; an infinite one already mapped to +-1e300) times a texture size in texels, as
// the card's multiplier gives it: a 10-bit float (see the header comment).
inline double scale_by_size(double g, uint32_t size) {
  if (g == 0 || size == 0 || (size & (size - 1)) == 0) return g * static_cast<double>(size);
  int e;
  const double fr = std::frexp(std::fabs(g), &e);   // fr in [0.5, 1): the significand 512..1023 is its top 10 bits
  e -= 1;
  const uint32_t sig = static_cast<uint32_t>(fr * 1024.0);
  const int wexp = static_cast<int>(std::bit_width(size)) - 1;
  uint32_t acc = sig << 3;   // the size's leading one: the gradient itself, with 12 fractional bits
  for (int i = 0; i < std::min(wexp, 9); ++i)   // the size's significand has 9 fractional bits at most
    if ((size >> (wexp - 1 - i)) & 1u) acc += (sig << 3) >> (i + 1);
  const int f = e + wexp;
  const double v = acc < (1u << 13) ? std::ldexp(static_cast<double>((acc + 7) >> 3), f - 9)
                                    : std::ldexp(static_cast<double>((acc + 7) >> 4), f - 8);
  return std::copysign(v, g);
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

// The LOD in 1/256ths of a level for a 3D fetch from the gradients in texels of the base level, dPdx = dx[0..2] and
// dPdy = dy[0..2]. The vectors, their sum and their difference are measured as for a 2D fetch, with the
// three-component lengths above.
inline int64_t lod_q_3d(const double* dx, const double* dy) {
  using namespace detail;
  double sum[3], dif[3];
  for (int i = 0; i < 3; ++i) {
    const double a = t10(dx[i]), b = t10(dy[i]);
    sum[i] = add_signed(a, b);
    dif[i] = add_signed(a, -b);
  }
  const double rho = std::max({length3(dx), length3(dy), diagonal3(sum), diagonal3(dif)});
  return log2_q(rho);
}

// What the card's texture unit sees as a gradient of a 3D (or cube) fetch. ptxas builds those fetches from the
// coordinates of a quad -- P, P + dPdx, P + dPdy -- computed in single precision (FSWZADD) and handed to the unit,
// which takes their differences: the gradient is c1 - c0, not d. A coordinate that is NaN reads as 0 (measured: a
// NaN gradient component makes the level that of a jump from the coordinate to 0), and one of 2^97 or more,
// infinite included, makes the difference overflow: the fetch reads the last level whatever the other components
// are.
inline double lane_difference(float c0, float c1) {
  constexpr float kOverflow = 0x1p97f;
  if ((!std::isnan(c0) && std::fabs(c0) >= kOverflow) || (!std::isnan(c1) && std::fabs(c1) >= kOverflow)) return 1e300;
  return (std::isnan(c1) ? 0.0 : static_cast<double>(c1)) - (std::isnan(c0) ? 0.0 : static_cast<double>(c0));
}

// The same from the position and the gradient component (the PTX instruction's operands).
inline double quad_difference(float p, float d) {
  volatile float c = p + d;   // one single-precision add, as FSWZADD does it
  return lane_difference(p, c);
}

}  // namespace vgpu::exec::tex_grad
