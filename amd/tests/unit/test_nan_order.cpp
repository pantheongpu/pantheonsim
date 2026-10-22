// The NaN-order helpers (amd/src/nan_order.hpp): the same bits whatever the optimizer does with the operands. The tests compare the
// helpers with what the same operation gives where the answer does not depend on the operand order (no NaN, or one), and spell out
// the order where it does. The values are the ones the same-bits corpora draw: zeros, infinities, quiet and signaling NaNs of both
// signs, and numbers.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../../src/nan_order.hpp"
#include "vtest.hpp"

namespace {

using vgpu::amd::nan_add;
using vgpu::amd::nan_fma;
using vgpu::amd::nan_fmax;
using vgpu::amd::nan_fmin;
using vgpu::amd::nan_mul;

uint32_t bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}
float from(uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}
const std::vector<float>& values() {
  static const std::vector<float> v = {from(0), from(0x80000000), 1.0f, -2.5f, from(0x7f800000), from(0xff800000), from(0x7fc00001),
                                       from(0xffc00002), from(0x7f800001), from(0xff800002), 3.0e38f, from(1)};
  return v;
}
bool plain(float x, float y) { return !(x != x) && !(y != y); }

VTEST(a_sum_a_product_and_a_fused_sum_of_numbers_are_the_plain_operation) {
  for (float x : values())
    for (float y : values())
      if (plain(x, y)) {
        const volatile float vx = x, vy = y;
        const float px = vx, py = vy;
        VCHECK_EQ(bits(nan_add<0>(px, py)), bits(nan_add<1>(px, py)));
        if (!std::isnan(px + py)) VCHECK_EQ(bits(nan_add<0>(px, py)), bits(px + py));
        if (!std::isnan(px * py)) VCHECK_EQ(bits(nan_mul<1>(px, py)), bits(px * py));
        for (float z : values())
          if (!std::isnan(std::fma(px, py, z)) && z == z) VCHECK_EQ(bits(nan_fma<2>(px, py, z)), bits(std::fma(px, py, z)));
      }
}

VTEST(two_nans_in_a_sum_or_a_product_give_the_first_in_the_order_named_quieted) {
  const float a = from(0x7f800001), b = from(0xffc00002);   // a signaling NaN, a quiet one
  VCHECK_EQ(bits(nan_add<0>(a, b)), 0x7fc00001u);
  VCHECK_EQ(bits(nan_add<1>(a, b)), 0xffc00002u);
  VCHECK_EQ(bits(nan_mul<0>(a, b)), 0x7fc00001u);
  VCHECK_EQ(bits(nan_mul<1>(a, b)), 0xffc00002u);
  // A NaN beside a number is the NaN, quieted, in either order; a NaN made by the operation (infinity minus infinity) is the default one.
  VCHECK_EQ(bits(nan_add<0>(1.0f, a)), 0x7fc00001u);
  VCHECK_EQ(bits(nan_add<1>(a, 1.0f)), 0x7fc00001u);
  VCHECK(std::isnan(nan_add<0>(from(0x7f800000), from(0xff800000))));
}

VTEST(an_fma_with_several_nans_takes_the_first_in_the_permutation_named) {
  const float a = from(0x7fc00001), b = from(0xffc00002), c = from(0x7fc00003);
  static const uint32_t kWant[6] = {0x7fc00001, 0x7fc00001, 0xffc00002, 0xffc00002, 0x7fc00003, 0x7fc00003};
  VCHECK_EQ(bits(nan_fma<0>(a, b, c)), kWant[0]);   // a b c
  VCHECK_EQ(bits(nan_fma<1>(a, b, c)), kWant[1]);   // a c b
  VCHECK_EQ(bits(nan_fma<2>(a, b, c)), kWant[2]);   // b a c
  VCHECK_EQ(bits(nan_fma<3>(a, b, c)), kWant[3]);   // b c a
  VCHECK_EQ(bits(nan_fma<4>(a, b, c)), kWant[4]);   // c a b
  VCHECK_EQ(bits(nan_fma<5>(a, b, c)), kWant[5]);   // c b a
  VCHECK_EQ(bits(nan_fma<2>(a, 1.0f, c)), 0x7fc00001u);
  VCHECK_EQ(bits(nan_fma<2>(0.0f, from(0x7f800000), 1.0f)), 0xffc00000u);   // zero times infinity: the default NaN
}

// C's fmin and fmax, where the library's answer is not a matter of the operand order the compiler passed (it is, for two NaNs and for
// +0 against -0, and that is what the order template argument spells).
VTEST(fmin_and_fmax_are_the_librarys_where_it_does_not_depend_on_the_order) {
  for (float x : values())
    for (float y : values()) {
      const volatile float vx = x, vy = y;
      const float px = vx, py = vy;
      const bool zeros = px == py;   // (+0 against -0)
      const bool nans = (px != px) && (py != py);
      const bool signaling_one = (px != px) != (py != py);
      if (nans || zeros || signaling_one) continue;
      VCHECK_EQ(bits(nan_fmax<0>(px, py)), bits(std::fmax(px, py)));
      VCHECK_EQ(bits(nan_fmin<1>(px, py)), bits(std::fmin(px, py)));
    }
  // A quiet NaN beside a number is the number; a signaling one is itself, quieted.
  VCHECK_EQ(bits(nan_fmax<0>(from(0x7fc00001), 2.0f)), bits(2.0f));
  VCHECK_EQ(bits(nan_fmin<0>(2.0f, from(0x7fc00001))), bits(2.0f));
  VCHECK_EQ(bits(nan_fmax<0>(from(0x7f800001), 2.0f)), 0x7fc00001u);
  // Two NaNs: the first, quieted, or with the order flipped the second.
  VCHECK_EQ(bits(nan_fmax<0>(from(0x7f800001), from(0xffc00002))), 0x7fc00001u);
  VCHECK_EQ(bits(nan_fmax<1>(from(0x7f800001), from(0xffc00002))), 0xffc00002u);
  // +0 against -0: the second operand of the library call.
  VCHECK_EQ(bits(nan_fmax<0>(from(0), from(0x80000000))), 0x80000000u);
  VCHECK_EQ(bits(nan_fmax<1>(from(0), from(0x80000000))), 0u);
  VCHECK_EQ(bits(nan_fmin<0>(from(0x80000000), from(0))), 0u);
}

}  // namespace

VTEST_MAIN
