// v_rcp_f32, v_rsq_f32 and v_sqrt_f32 as an RX 6800 and an RX 6700 XT (gfx1030, gfx1031) compute them, bit for bit over all 2^32 operands
// (docs/numerics/math-functions.md, "AMD"; issues #729 and #741). The card's results are not the correctly rounded ones (10.7, 11.0 and
// 15.4 percent of the operands differ by one unit in the last place); no closed form is known, so each function is a cubic per 2^18
// mantissas, the card's squarer operand with its low bits cleared, plus the list of the mantissas where the card is one unit away from the
// cubic (about 0.8 percent). The data is amd/src/rdna2_trans_data.inc (tools/numerics/amd_trans_table_fit.py, from the card's dumps of one
// binade of operands); the model is the card's on every normal operand (the hashes of all 2^32 operands are tested), and the special
// operands (zero, denormal, infinity, NaN, a negative operand of rsq and sqrt) are the host rule of transcendental.hpp, which the card's
// vectors confirm.
#pragma once
#include <cstdint>

namespace vgpu::amd::rdna2 {

float rcp(float x);
float rsq(float x);
float sqrt(float x);

}  // namespace vgpu::amd::rdna2
