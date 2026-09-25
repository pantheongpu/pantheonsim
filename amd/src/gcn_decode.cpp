#include <cstdio>
#include <string>
#include <map>

#include "vgpu/amd_gcn.hpp"
#include "vgpu/error.hpp"

namespace vgpu::amd::gcn {
namespace {

// What an instruction does with its operands, which is what the decoder needs
// beyond the opcode: how many sources it reads, and how many 32-bit registers
// its destination and each source cover.
struct Shape {
  const char* name;
  uint32_t dst_width = 1;       // 0: it writes no register
  uint32_t srcs = 2;            // how many sources the assembler prints
  uint32_t w0 = 1, w1 = 1, w2 = 1;
  // VOP3b: the instruction also writes a scalar pair (a carry-out, or the
  // condition v_div_scale reports). VOP3 with a scalar destination instead
  // (v_readlane, and the comparisons in their long form).
  bool sdst = false;
  bool scalar_dst = false;
  uint32_t src_width(uint32_t i) const { return i == 0 ? w0 : i == 1 ? w1 : w2; }
};

// The instructions decoded so far: those the compiler emits for the kernels
// this runs. Each is <encoding, opcode> -> its name and operand shape, from
// the CDNA3 ISA reference guide's opcode tables.
const std::map<std::pair<Enc, uint32_t>, Shape>& table() {
  static const std::map<std::pair<Enc, uint32_t>, Shape> t = {
      // SOP1: one scalar source, one scalar destination.
      {{Enc::Sop1, 0x00}, {"s_mov_b32", 1, 1}},
      {{Enc::Sop1, 0x01}, {"s_mov_b64", 2, 1, 2}},
      {{Enc::Sop1, 0x08}, {"s_brev_b32", 1, 1}},
      {{Enc::Sop1, 0x0d}, {"s_bcnt1_i32_b64", 1, 1, 2}},
      {{Enc::Sop1, 0x11}, {"s_ff1_i32_b64", 1, 1, 2}},
      // Where the wave is: what a kernel adds a constant to, to reach a global.
      {{Enc::Sop1, 0x1c}, {"s_getpc_b64", 2, 0}},
      // A call, and the return from it: one saves where to come back to
      // while it jumps, the other jumps back.
      {{Enc::Sop1, 0x1d}, {"s_setpc_b64", 0, 1, 2}},
      {{Enc::Sop1, 0x1e}, {"s_swappc_b64", 2, 1, 2}},
      {{Enc::Sop1, 0x20}, {"s_and_saveexec_b64", 2, 1, 2}},
      {{Enc::Sop1, 0x21}, {"s_or_saveexec_b64", 2, 1, 2}},
      {{Enc::Sop1, 0x23}, {"s_andn2_saveexec_b64", 2, 1, 2}},
      // SOP2: two scalar sources.
      {{Enc::Sop2, 0x00}, {"s_add_u32", 1, 2}},
      {{Enc::Sop2, 0x01}, {"s_sub_u32", 1, 2}},
      {{Enc::Sop2, 0x02}, {"s_add_i32", 1, 2}},
      {{Enc::Sop2, 0x03}, {"s_sub_i32", 1, 2}},
      {{Enc::Sop2, 0x04}, {"s_addc_u32", 1, 2}},
      {{Enc::Sop2, 0x05}, {"s_subb_u32", 1, 2}},
      {{Enc::Sop2, 0x0a}, {"s_cselect_b32", 1, 2}},
      {{Enc::Sop2, 0x0b}, {"s_cselect_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x0c}, {"s_and_b32", 1, 2}},
      {{Enc::Sop2, 0x0d}, {"s_and_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x0e}, {"s_or_b32", 1, 2}},
      {{Enc::Sop2, 0x0f}, {"s_or_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x10}, {"s_xor_b32", 1, 2}},
      {{Enc::Sop2, 0x11}, {"s_xor_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x13}, {"s_andn2_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x15}, {"s_orn2_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x1c}, {"s_lshl_b32", 1, 2}},
      {{Enc::Sop2, 0x1d}, {"s_lshl_b64", 2, 2, 2, 1}},
      {{Enc::Sop2, 0x1e}, {"s_lshr_b32", 1, 2}},
      {{Enc::Sop2, 0x1f}, {"s_lshr_b64", 2, 2, 2, 1}},
      {{Enc::Sop2, 0x20}, {"s_ashr_i32", 1, 2}},
      {{Enc::Sop2, 0x24}, {"s_mul_i32", 1, 2}},
      {{Enc::Sop2, 0x2c}, {"s_mul_hi_u32", 1, 2}},
      // SOPK: a 16-bit immediate.
      {{Enc::Sopk, 0x00}, {"s_movk_i32", 1, 0}},
      {{Enc::Sopk, 0x0f}, {"s_mulk_i32", 1, 0}},
      // A comparison against the instruction's own constant: the register
      // field names what is compared, not where a result goes.
      {{Enc::Sopk, 0x02}, {"s_cmpk_eq_i32", 1, 0}},
      // SOPP: an immediate, and no registers.
      {{Enc::Sopp, 0x00}, {"s_nop", 0, 0}},
      {{Enc::Sopp, 0x01}, {"s_endpgm", 0, 0}},
      {{Enc::Sopp, 0x02}, {"s_branch", 0, 0}},
      {{Enc::Sopp, 0x04}, {"s_cbranch_scc0", 0, 0}},
      {{Enc::Sopp, 0x05}, {"s_cbranch_scc1", 0, 0}},
      {{Enc::Sopp, 0x06}, {"s_cbranch_vccz", 0, 0}},
      {{Enc::Sopp, 0x07}, {"s_cbranch_vccnz", 0, 0}},
      {{Enc::Sopp, 0x08}, {"s_cbranch_execz", 0, 0}},
      {{Enc::Sopp, 0x09}, {"s_cbranch_execnz", 0, 0}},
      {{Enc::Sopp, 0x0a}, {"s_barrier", 0, 0}},
      {{Enc::Sopp, 0x0c}, {"s_waitcnt", 0, 0}},
      {{Enc::Sopp, 0x0e}, {"s_sleep", 0, 0}},
      // A message to the host: MSG_INTERRUPT is how a kernel wakes the host
      // for a hostcall, which is what device-side printf is built on.
      {{Enc::Sopp, 0x10}, {"s_sendmsg", 0, 0}},
      // SOPC: a scalar comparison, which sets SCC.
      {{Enc::Sopc, 0x02}, {"s_cmp_gt_i32", 0, 2}},
      {{Enc::Sopc, 0x03}, {"s_cmp_ge_i32", 0, 2}},
      {{Enc::Sopc, 0x04}, {"s_cmp_lt_i32", 0, 2}},
      {{Enc::Sopc, 0x06}, {"s_cmp_eq_u32", 0, 2}},
      {{Enc::Sopc, 0x07}, {"s_cmp_lg_u32", 0, 2}},
      {{Enc::Sopc, 0x08}, {"s_cmp_gt_u32", 0, 2}},
      {{Enc::Sopc, 0x09}, {"s_cmp_ge_u32", 0, 2}},
      {{Enc::Sopc, 0x0a}, {"s_cmp_lt_u32", 0, 2}},
      {{Enc::Sopc, 0x12}, {"s_cmp_eq_u64", 0, 2, 2, 2}},
      {{Enc::Sopc, 0x13}, {"s_cmp_lg_u64", 0, 2, 2, 2}},
      // SMEM: a scalar load through a 64-bit base address.
      {{Enc::Smem, 0x00}, {"s_load_dword", 1, 1, 2}},
      {{Enc::Smem, 0x01}, {"s_load_dwordx2", 2, 1, 2}},
      {{Enc::Smem, 0x02}, {"s_load_dwordx4", 4, 1, 2}},
      {{Enc::Smem, 0x03}, {"s_load_dwordx8", 8, 1, 2}},
      // A counter the wave reads: no address, and nothing but the pair it
      // writes.
      {{Enc::Smem, 0x24}, {"s_memtime", 2, 0}},
      {{Enc::Smem, 0x25}, {"s_memrealtime", 2, 0}},
      // VOP1 and VOP2, the vector ALU's short forms.
      {{Enc::Vop1, 0x01}, {"v_mov_b32_e32", 1, 1}},
      {{Enc::Vop2, 0x00}, {"v_cndmask_b32_e32", 1, 2}},
      {{Enc::Vop2, 0x01}, {"v_add_f32_e32", 1, 2}},
      {{Enc::Vop2, 0x02}, {"v_sub_f32_e32", 1, 2}},
      {{Enc::Vop2, 0x04}, {"v_fmac_f64_e32", 2, 2, 2, 2}},
      {{Enc::Vop2, 0x05}, {"v_mul_f32_e32", 1, 2}},
      {{Enc::Vop2, 0x10}, {"v_lshrrev_b32_e32", 1, 2}},
      {{Enc::Vop2, 0x12}, {"v_lshlrev_b32_e32", 1, 2}},
      {{Enc::Vop2, 0x11}, {"v_ashrrev_i32_e32", 1, 2}},
      // VOPC: a comparison, writing VCC.
      {{Enc::Vopc, 0xc4}, {"v_cmp_gt_i32_e32", 2, 2}},
      {{Enc::Vopc, 0xca}, {"v_cmp_eq_u32_e32", 2, 2}},
      {{Enc::Vopc, 0xcc}, {"v_cmp_gt_u32_e32", 2, 2}},
      {{Enc::Vop1, 0x005}, {"v_cvt_f32_i32_e32", 1, 1}},
      {{Enc::Vop1, 0x006}, {"v_cvt_f32_u32_e32", 1, 1}},
      {{Enc::Vop1, 0x002}, {"v_readfirstlane_b32", 1, 1, 1, 1, 1, false, true}},
      {{Enc::Vop1, 0x007}, {"v_cvt_u32_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x008}, {"v_cvt_i32_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x003}, {"v_cvt_i32_f64_e32", 1, 1, 2}},
      {{Enc::Vop1, 0x004}, {"v_cvt_f64_i32_e32", 2, 1, 1}},
      {{Enc::Vop1, 0x00a}, {"v_cvt_f16_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x00b}, {"v_cvt_f32_f16_e32", 1, 1}},
      {{Enc::Vop1, 0x00f}, {"v_cvt_f32_f64_e32", 1, 1, 2}},
      {{Enc::Vop1, 0x01c}, {"v_trunc_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x010}, {"v_cvt_f64_f32_e32", 2, 1, 1}},
      {{Enc::Vop1, 0x016}, {"v_cvt_f64_u32_e32", 2, 1, 1}},
      // The same rounding over a double, which a register pair holds.
      {{Enc::Vop1, 0x017}, {"v_trunc_f64_e32", 2, 1, 2}},
      {{Enc::Vop1, 0x018}, {"v_ceil_f64_e32", 2, 1, 2}},
      {{Enc::Vop1, 0x019}, {"v_rndne_f64_e32", 2, 1, 2}},
      {{Enc::Vop1, 0x01a}, {"v_floor_f64_e32", 2, 1, 2}},
      {{Enc::Vop1, 0x011}, {"v_cvt_f32_ubyte0_e32", 1, 1}},
      {{Enc::Vop1, 0x01d}, {"v_ceil_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x01e}, {"v_rndne_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x01f}, {"v_floor_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x020}, {"v_exp_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x021}, {"v_log_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x022}, {"v_rcp_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x023}, {"v_rcp_iflag_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x025}, {"v_rcp_f64_e32", 2, 1, 2}},
      {{Enc::Vop1, 0x026}, {"v_rsq_f64_e32", 2, 1, 2}},
      {{Enc::Vop1, 0x024}, {"v_rsq_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x027}, {"v_sqrt_f32_e32", 1, 1}},
      // The sine and the cosine of a turn: the argument is in turns, not
      // radians, which is why the compiler multiplies by 1/2pi first.
      {{Enc::Vop1, 0x029}, {"v_sin_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x02a}, {"v_cos_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x02b}, {"v_not_b32_e32", 1, 1}},
      {{Enc::Vop1, 0x02c}, {"v_bfrev_b32_e32", 1, 1}},
      {{Enc::Vop1, 0x02d}, {"v_ffbh_u32_e32", 1, 1}},
      {{Enc::Vop1, 0x033}, {"v_frexp_exp_i32_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x034}, {"v_frexp_mant_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x038}, {"v_mov_b64_e32", 2, 1, 2}},
      {{Enc::Vop2, 0x006}, {"v_mul_i32_i24_e32", 1, 2}},
      {{Enc::Vop2, 0x008}, {"v_mul_u32_u24_e32", 1, 2}},
      {{Enc::Vop2, 0x00a}, {"v_min_f32_e32", 1, 2}},
      {{Enc::Vop2, 0x00b}, {"v_max_f32_e32", 1, 2}},
      {{Enc::Vop2, 0x00f}, {"v_max_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x00e}, {"v_min_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x013}, {"v_and_b32_e32", 1, 2}},
      {{Enc::Vop2, 0x014}, {"v_or_b32_e32", 1, 2}},
      {{Enc::Vop2, 0x015}, {"v_xor_b32_e32", 1, 2}},
      // 16-bit arithmetic, which a kernel gets for a short or a char. The
      // result is the low half of the destination, and the high half is
      // zeroed: the compiler drops the mask a widening would otherwise need
      // after one of these, which it could not do if the half were kept.
      {{Enc::Vop2, 0x026}, {"v_add_u16_e32", 1, 2}},
      {{Enc::Vop2, 0x027}, {"v_sub_u16_e32", 1, 2}},
      {{Enc::Vop2, 0x029}, {"v_mul_lo_u16_e32", 1, 2}},
      {{Enc::Vop2, 0x02a}, {"v_lshlrev_b16_e32", 1, 2}},
      // Half precision one value at a time, where the packed form does two.
      {{Enc::Vop2, 0x01f}, {"v_add_f16_e32", 1, 2}},
      {{Enc::Vop2, 0x020}, {"v_sub_f16_e32", 1, 2}},
      {{Enc::Vop2, 0x022}, {"v_mul_f16_e32", 1, 2}},
      // A constant of the instruction's own, between its two sources.
      {{Enc::Vop2, 0x017}, {"v_fmamk_f32", 1, 2}},
      // The carry forms: each writes a mask of the lanes that carried beside
      // its result, and the two that take one read it back.
      // ...and v_fmaak_f32, whose constant comes last.
      {{Enc::Vop2, 0x018}, {"v_fmaak_f32", 1, 2}},
      {{Enc::Vop2, 0x019}, {"v_add_co_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x01a}, {"v_sub_co_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x01c}, {"v_addc_co_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x01d}, {"v_subb_co_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x034}, {"v_add_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x035}, {"v_sub_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x036}, {"v_subrev_u32_e32", 1, 2}},
      // The dot products, which add into their destination as fmac does.
      {{Enc::Vop2, 0x037}, {"v_dot2c_f32_f16_e32", 1, 2}},
      {{Enc::Vop2, 0x039}, {"v_dot4c_i32_i8_e32", 1, 2}},
      {{Enc::Vop2, 0x03b}, {"v_fmac_f32_e32", 1, 2}},
      // The float comparisons share their opcodes with the long forms above.
      // A class test: the second source is a mask of the kinds of float
      // (NaN, infinity, normal, denormal, zero, each with a sign) it asks about.
      {{Enc::Vopc, 0x010}, {"v_cmp_class_f32_e32", 2, 2}},
      {{Enc::Vopc, 0x012}, {"v_cmp_class_f64_e32", 2, 2, 2, 1}},
      {{Enc::Vopc, 0x021}, {"v_cmp_lt_f16_e32", 2, 2}},
      {{Enc::Vopc, 0x022}, {"v_cmp_eq_f16_e32", 2, 2}},
      {{Enc::Vopc, 0x024}, {"v_cmp_gt_f16_e32", 2, 2}},
      {{Enc::Vopc, 0x026}, {"v_cmp_ge_f16_e32", 2, 2}},
      {{Enc::Vopc, 0x02b}, {"v_cmp_ngt_f16_e32", 2, 2}},
      {{Enc::Vopc, 0x02d}, {"v_cmp_neq_f16_e32", 2, 2}},
      // The comparisons over a double.
      {{Enc::Vopc, 0x061}, {"v_cmp_lt_f64_e32", 2, 2, 2, 2}},
      {{Enc::Vopc, 0x062}, {"v_cmp_eq_f64_e32", 2, 2, 2, 2}},
      {{Enc::Vopc, 0x064}, {"v_cmp_gt_f64_e32", 2, 2, 2, 2}},
      {{Enc::Vopc, 0x066}, {"v_cmp_ge_f64_e32", 2, 2, 2, 2}},
      {{Enc::Vopc, 0x06d}, {"v_cmp_neq_f64_e32", 2, 2, 2, 2}},
      {{Enc::Vopc, 0x041}, {"v_cmp_lt_f32_e32", 2, 2}},
      {{Enc::Vopc, 0x042}, {"v_cmp_eq_f32_e32", 2, 2}},
      // The negated forms, which a NaN answers "yes" to.
      {{Enc::Vopc, 0x049}, {"v_cmp_nge_f32_e32", 2, 2}},
      {{Enc::Vopc, 0x04b}, {"v_cmp_ngt_f32_e32", 2, 2}},
      {{Enc::Vopc, 0x04d}, {"v_cmp_neq_f32_e32", 2, 2}},
      {{Enc::Vopc, 0x04e}, {"v_cmp_nlt_f32_e32", 2, 2}},
      {{Enc::Vopc, 0x0aa}, {"v_cmp_eq_u16_e32", 2, 2}},
      {{Enc::Vopc, 0x0ad}, {"v_cmp_ne_u16_e32", 2, 2}},
      {{Enc::Vopc, 0x0c9}, {"v_cmp_lt_u32_e32", 2, 2}},
      {{Enc::Vopc, 0x0eb}, {"v_cmp_le_u64_e32", 2, 2, 2, 2}},
      {{Enc::Vopc, 0x0ec}, {"v_cmp_gt_u64_e32", 2, 2, 2, 2}},
      {{Enc::Vopc, 0x0ee}, {"v_cmp_ge_u64_e32", 2, 2, 2, 2}},
      {{Enc::Vopc, 0x044}, {"v_cmp_gt_f32_e32", 2, 2}},
      {{Enc::Vopc, 0x046}, {"v_cmp_ge_f32_e32", 2, 2}},
      {{Enc::Vopc, 0x0c1}, {"v_cmp_lt_i32_e32", 2, 2}},
      {{Enc::Vopc, 0x0cb}, {"v_cmp_le_u32_e32", 2, 2}},
      {{Enc::Vopc, 0x0cd}, {"v_cmp_ne_u32_e32", 2, 2}},
      {{Enc::Vopc, 0x0ce}, {"v_cmp_ge_u32_e32", 2, 2}},
      // The same comparisons over a register pair.
      {{Enc::Vopc, 0x0e9}, {"v_cmp_lt_u64_e32", 2, 2, 2, 2}},
      {{Enc::Vopc, 0x0ea}, {"v_cmp_eq_u64_e32", 2, 2, 2, 2}},
      {{Enc::Vopc, 0x0ed}, {"v_cmp_ne_u64_e32", 2, 2, 2, 2}},
      // VOP3: the long form. v_lshl_add shifts its first source and adds the
      // third; the shift itself is always 32-bit.
      {{Enc::Vop3, 0x041}, {"v_cmp_lt_f32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x044}, {"v_cmp_gt_f32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x0aa}, {"v_cmp_eq_u16_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x010}, {"v_cmp_class_f32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x02b}, {"v_cmp_ngt_f16_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x04b}, {"v_cmp_ngt_f32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x04e}, {"v_cmp_nlt_f32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x06b}, {"v_cmp_ngt_f64_e64", 2, 2, 2, 2, 1, false, true}},
      {{Enc::Vop3, 0x0c9}, {"v_cmp_lt_u32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x0ca}, {"v_cmp_eq_u32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x0cd}, {"v_cmp_ne_u32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x0ce}, {"v_cmp_ge_u32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x0cb}, {"v_cmp_le_u32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x0cc}, {"v_cmp_gt_u32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x0e9}, {"v_cmp_lt_u64_e64", 2, 2, 2, 2, 1, false, true}},
      {{Enc::Vop3, 0x0eb}, {"v_cmp_le_u64_e64", 2, 2, 2, 2, 1, false, true}},
      {{Enc::Vop3, 0x0ec}, {"v_cmp_gt_u64_e64", 2, 2, 2, 2, 1, false, true}},
      {{Enc::Vop3, 0x101}, {"v_add_f32_e64", 1, 2}},
      // The long forms of the carry arithmetic, which name the pair they
      // write the carry to rather than always using VCC.
      {{Enc::Vop3, 0x11a}, {"v_sub_co_u32_e64", 1, 2, 1, 1, 1, true}},
      // The long forms of the 16-bit arithmetic, which name a source the
      // short form could not reach.
      {{Enc::Vop3, 0x126}, {"v_add_u16_e64", 1, 2}},
      {{Enc::Vop3, 0x12a}, {"v_lshlrev_b16_e64", 1, 2}},
      {{Enc::Vop3, 0x11d}, {"v_subb_co_u32_e64", 1, 3, 1, 1, 2, true}},
      {{Enc::Vop3, 0x11e}, {"v_subbrev_co_u32_e64", 1, 3, 1, 1, 2, true}},
      {{Enc::Vop3, 0x10b}, {"v_max_f32_e64", 1, 2}},
      {{Enc::Vop3, 0x100}, {"v_cndmask_b32_e64", 1, 3, 1, 1, 2}},
      {{Enc::Vop3, 0x102}, {"v_sub_f32_e64", 1, 2}},
      {{Enc::Vop3, 0x105}, {"v_mul_f32_e64", 1, 2}},
      {{Enc::Vop3, 0x112}, {"v_lshlrev_b32_e64", 1, 2}},
      {{Enc::Vop3, 0x11b}, {"v_subrev_co_u32_e64", 1, 2, 1, 1, 1, true}},
      {{Enc::Vop3, 0x1c3}, {"v_mad_u32_u24", 1, 3}},
      {{Enc::Vop3, 0x1c8}, {"v_bfe_u32", 1, 3}},
      {{Enc::Vop3, 0x1d0}, {"v_min3_f32", 1, 3}},
      {{Enc::Vop3, 0x1d1}, {"v_min3_i32", 1, 3}},
      {{Enc::Vop3, 0x1d3}, {"v_max3_f32", 1, 3}},
      {{Enc::Vop3, 0x1d4}, {"v_max3_i32", 1, 3}},
      {{Enc::Vop3, 0x1d7}, {"v_med3_i32", 1, 3}},
      {{Enc::Vop3, 0x1c9}, {"v_bfe_i32", 1, 3}},
      {{Enc::Vop3, 0x1ca}, {"v_bfi_b32", 1, 3}},
      {{Enc::Vop3, 0x1cb}, {"v_fma_f32", 1, 3}},
      {{Enc::Vop3, 0x1ce}, {"v_alignbit_b32", 1, 3}},
      {{Enc::Vop3, 0x1cc}, {"v_fma_f64", 2, 3, 2, 2, 2}},
      {{Enc::Vop3, 0x1de}, {"v_div_fixup_f32", 1, 3}},
      {{Enc::Vop3, 0x1e0}, {"v_div_scale_f32", 1, 3, 1, 1, 1, true}},
      {{Enc::Vop3, 0x1df}, {"v_div_fixup_f64", 2, 3, 2, 2, 2}},
      {{Enc::Vop3, 0x1e1}, {"v_div_scale_f64", 2, 3, 2, 2, 2, true}},
      {{Enc::Vop3, 0x1e2}, {"v_div_fmas_f32", 1, 3}},
      {{Enc::Vop3, 0x1e3}, {"v_div_fmas_f64", 2, 3, 2, 2, 2}},
      {{Enc::Vop3, 0x1e8}, {"v_mad_u64_u32", 2, 3, 1, 1, 2, true}},
      {{Enc::Vop3, 0x1eb}, {"v_mad_legacy_u16", 1, 3}},
      {{Enc::Vop3, 0x1fd}, {"v_lshl_add_u32", 1, 3}},
      {{Enc::Vop3, 0x1ed}, {"v_perm_b32", 1, 3}},
      {{Enc::Vop3, 0x1f3}, {"v_xad_u32", 1, 3}},
      {{Enc::Vop3, 0x1fe}, {"v_add_lshl_u32", 1, 3}},
      {{Enc::Vop3, 0x1ff}, {"v_add3_u32", 1, 3}},
      {{Enc::Vop3, 0x201}, {"v_and_or_b32", 1, 3}},
      {{Enc::Vop3, 0x200}, {"v_lshl_or_b32", 1, 3}},
      {{Enc::Vop3, 0x202}, {"v_or3_b32", 1, 3}},
      {{Enc::Vop3, 0x206}, {"v_fma_f16", 1, 3}},
      {{Enc::Vop3, 0x208}, {"v_lshl_add_u64", 2, 3, 2, 1, 2}},
      {{Enc::Vop3, 0x280}, {"v_add_f64", 2, 2, 2, 2}},
      {{Enc::Vop3, 0x281}, {"v_mul_f64", 2, 2, 2, 2}},
      {{Enc::Vop3, 0x282}, {"v_min_f64", 2, 2, 2, 2}},
      {{Enc::Vop3, 0x283}, {"v_max_f64", 2, 2, 2, 2}},
      {{Enc::Vop3, 0x284}, {"v_ldexp_f64", 2, 2, 2, 1}},
      {{Enc::Vop3, 0x285}, {"v_mul_lo_u32", 1, 2}},
      {{Enc::Vop3, 0x286}, {"v_mul_hi_u32", 1, 2}},
      {{Enc::Vop3, 0x287}, {"v_mul_hi_i32", 1, 2}},
      {{Enc::Vop3, 0x288}, {"v_ldexp_f32", 1, 2}},
      {{Enc::Vop3, 0x289}, {"v_readlane_b32", 1, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x28a}, {"v_writelane_b32", 1, 2}},
      {{Enc::Vop3, 0x28b}, {"v_bcnt_u32_b32", 1, 2}},
      {{Enc::Vop3, 0x28c}, {"v_mbcnt_lo_u32_b32", 1, 2}},
      {{Enc::Vop3, 0x28d}, {"v_mbcnt_hi_u32_b32", 1, 2}},
      {{Enc::Vop3, 0x28f}, {"v_lshlrev_b64", 2, 2, 1, 2}},
      {{Enc::Vop3, 0x296}, {"v_cvt_pkrtz_f16_f32", 1, 2}},
      {{Enc::Vop3, 0x290}, {"v_lshrrev_b64", 2, 2, 1, 2}},
      {{Enc::Vop3, 0x291}, {"v_ashrrev_i64", 2, 2, 1, 2}},
      {{Enc::Vop3, 0x2a0}, {"v_pack_b32_f16", 1, 2}},
      // DS: LDS reads and writes. A read takes the address; a write takes the
      // address and the data.
      {{Enc::Ds, 0x00}, {"ds_add_u32", 0, 2}},
      {{Enc::Ds, 0x06}, {"ds_max_i32", 0, 2}},
      {{Enc::Ds, 0x0b}, {"ds_xor_b32", 0, 2}},
      {{Enc::Ds, 0x0d}, {"ds_write_b32", 0, 2}},
      {{Enc::Ds, 0x0e}, {"ds_write2_b32", 0, 3}},
      {{Enc::Ds, 0x15}, {"ds_add_f32", 0, 2}},
      // Narrower than a word, and wider: a byte, a half, and two or four
      // words at once.
      {{Enc::Ds, 0x1e}, {"ds_write_b8", 0, 2}},
      {{Enc::Ds, 0x1f}, {"ds_write_b16", 0, 2}},
      {{Enc::Ds, 0x37}, {"ds_read2_b32", 2, 1}},
      {{Enc::Ds, 0x36}, {"ds_read_b32", 1, 1}},
      {{Enc::Ds, 0x38}, {"ds_read2st64_b32", 2, 1}},
      {{Enc::Ds, 0x39}, {"ds_read_i8", 1, 1}},
      {{Enc::Ds, 0x3a}, {"ds_read_u8", 1, 1}},
      {{Enc::Ds, 0x3c}, {"ds_read_u16", 1, 1}},
      {{Enc::Ds, 0x4d}, {"ds_write_b64", 0, 2, 1, 2}},
      {{Enc::Ds, 0x76}, {"ds_read_b64", 2, 1}},
      {{Enc::Ds, 0xdf}, {"ds_write_b128", 0, 2, 1, 4}},
      {{Enc::Ds, 0xff}, {"ds_read_b128", 4, 1}},
      // Lanes trading values without touching LDS at all: the offset is a
      // pattern saying which lane each one reads, and the data register sits
      // where an address would.
      {{Enc::Ds, 0x3d}, {"ds_swizzle_b32", 1, 1}},
      // A lane reads the value another lane holds: the address says which.
      {{Enc::Ds, 0x3f}, {"ds_bpermute_b32", 1, 2}},
      // FLAT and the two segments that share its opcodes: global (an address
      // in a register pair, or a scalar base and a per-lane offset) and
      // scratch (each work-item's private memory). The name carries the
      // segment, which the encoding keeps separately, so the table holds what
      // follows it.
      {{Enc::Flat, 0x10}, {"load_ubyte", 1, 1}},
      {{Enc::Flat, 0x11}, {"load_sbyte", 1, 1}},
      {{Enc::Flat, 0x12}, {"load_ushort", 1, 1}},
      {{Enc::Flat, 0x13}, {"load_sshort", 1, 1}},
      {{Enc::Flat, 0x14}, {"load_dword", 1, 1}},
      {{Enc::Flat, 0x15}, {"load_dwordx2", 2, 1}},
      {{Enc::Flat, 0x16}, {"load_dwordx3", 3, 1}},
      {{Enc::Flat, 0x17}, {"load_dwordx4", 4, 1}},
      {{Enc::Flat, 0x18}, {"store_byte", 0, 2}},
      {{Enc::Flat, 0x1a}, {"store_short", 0, 2}},
      {{Enc::Flat, 0x1c}, {"store_dword", 0, 2}},
      {{Enc::Flat, 0x1d}, {"store_dwordx2", 0, 2, 1, 2}},
      {{Enc::Flat, 0x1e}, {"store_dwordx3", 0, 2, 1, 3}},
      {{Enc::Flat, 0x1f}, {"store_dwordx4", 0, 2, 1, 4}},
      {{Enc::Flat, 0x40}, {"atomic_swap", 0, 2}},
      {{Enc::Flat, 0x41}, {"atomic_cmpswap", 0, 2, 1, 2}},
      {{Enc::Flat, 0x42}, {"atomic_add", 0, 2}},
      {{Enc::Flat, 0x43}, {"atomic_sub", 0, 2}},
      {{Enc::Flat, 0x48}, {"atomic_and", 0, 2}},
      {{Enc::Flat, 0x49}, {"atomic_or", 0, 2}},
      {{Enc::Flat, 0x4a}, {"atomic_xor", 0, 2}},
      {{Enc::Flat, 0x4d}, {"atomic_add_f32", 0, 2}},
      {{Enc::Flat, 0x61}, {"atomic_cmpswap_x2", 0, 2, 1, 4}},
      {{Enc::Flat, 0x62}, {"atomic_add_x2", 0, 2, 1, 2}},
      // MUBUF: the buffer instructions. Only the two that act on the caches
      // are decoded -- a write-back of L2 and an invalidate, which a memory
      // fence compiles to -- and they take no operands.
      {{Enc::Mubuf, 0x28}, {"buffer_wbl2", 0, 0}},
      {{Enc::Mubuf, 0x29}, {"buffer_inv", 0, 0}},
      // VOP3P: a packed pair of halves in one register, both computed at once.
      {{Enc::Vop3p, 0x0e}, {"v_pk_fma_f16", 1, 3}},
      // And the packed float form, where each of the two is a whole register.
      // A multiply-add over sources that may each be a float or a half,
      // rounded to a half and written into one half of the destination.
      {{Enc::Vop3p, 0x21}, {"v_fma_mixlo_f16", 1, 3}},
      {{Enc::Vop3p, 0x22}, {"v_fma_mixhi_f16", 1, 3}},
      {{Enc::Vop3p, 0x30}, {"v_pk_fma_f32", 2, 3, 2, 2, 2}},
      {{Enc::Vop3p, 0x32}, {"v_pk_add_f32", 2, 2, 2, 2}},
      {{Enc::Vop3p, 0x31}, {"v_pk_mul_f32", 2, 2, 2, 2}},
      // Between a vector register and an accumulation register, which is
      // where a kernel puts what will not fit in the vector ones. They share
      // the packed forms' encoding without being packed.
      // A matrix multiply-add over the whole wave: a 16x16 by 16x16 product of
      // halves, added into a 16x16 block of floats spread across the lanes.
      {{Enc::Vop3p, 0x4d}, {"v_mfma_f32_16x16x16_f16", 4, 3, 2, 2, 4}},
      {{Enc::Vop3p, 0x58}, {"v_accvgpr_read_b32", 1, 1}},
      {{Enc::Vop3p, 0x59}, {"v_accvgpr_write_b32", 1, 1}},
  };
  return t;
}

uint32_t word(const std::vector<uint8_t>& code, uint64_t at) {
  if (at + 4 > code.size()) throw Error::make(Err::Unsupported, "code ends inside an instruction");
  return code[at] | code[at + 1] << 8 | code[at + 2] << 16 | static_cast<uint32_t>(code[at + 3]) << 24;
}

// The ISA numbers scalar registers, the special registers and the inline
// constants in one space, with 255 meaning "a literal follows" and 256 and up
// the vector registers.
Operand operand(uint32_t code, uint32_t width) {
  Operand o;
  o.width = width;
  if (code <= 101) {
    o.kind = OperandKind::Sgpr;
    o.index = code;
  } else if (code == 106) {
    // A 32-bit instruction that reads VCC reads its low half, and the
    // assembler says so.
    o.kind = OperandKind::Vcc;
    o.width = width >= 2 ? 2 : 1;
  } else if (code == 124) {
    o.kind = OperandKind::M0;
  } else if (code == 126) {
    o.kind = width >= 2 ? OperandKind::Exec : OperandKind::ExecLo;
    o.width = width >= 2 ? 2 : 1;
  } else if (code == 127) {
    o.kind = OperandKind::ExecHi;
  } else if (code == 235) {
    // Where LDS sits in the one address space a flat access uses.
    o.kind = OperandKind::SharedBase;
    o.width = 2;
  } else if (code >= 240 && code <= 248) {
    // The inline float constants, in the ISA's order, ending with the one the
    // sine and the cosine need: a turn is 2pi radians, so a kernel that asked
    // for radians multiplies by this first.
    static const double kFloats[] = {0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 0.15915494309189532};
    o.kind = OperandKind::InlineFloat;
    o.fvalue = kFloats[code - 240];
  } else if (code >= 128 && code <= 192) {
    o.kind = OperandKind::Inline;
    o.value = code - 128;   // 0 through 64
  } else if (code >= 193 && code <= 208) {
    o.kind = OperandKind::Inline;
    o.value = -static_cast<int64_t>(code - 192);   // -1 through -16
  } else if (code == 255) {
    o.kind = OperandKind::Literal;
  } else if (code >= 256) {
    o.kind = OperandKind::Vgpr;
    o.index = code - 256;
  } else {
    throw Error::make(Err::Unsupported, "operand ", code, " is one this does not decode yet");
  }
  return o;
}

// A scalar destination: the same numbering as a source, so exec and vcc read
// as themselves rather than as the registers they are numbered with.
Operand sgpr(uint32_t index, uint32_t width = 1) {
  Operand o = operand(index, width);
  return o;
}
Operand vgpr(uint32_t index, uint32_t width = 1) {
  Operand o;
  o.kind = OperandKind::Vgpr;
  o.index = index;
  o.width = width;
  return o;
}

const Shape& shape(Enc e, uint32_t opcode) {
  const auto it = table().find({e, opcode});
  if (it == table().end())
    throw Error::make(Err::Unsupported, enc_name(e), " opcode 0x", [&] {
      char b[8];
      std::snprintf(b, sizeof b, "%x", opcode);
      return std::string(b);
    }(), " is not decoded yet");
  return it->second;
}

}  // namespace

const char* enc_name(Enc e) {
  switch (e) {
    case Enc::Sop1: return "SOP1";
    case Enc::Sop2: return "SOP2";
    case Enc::Sopk: return "SOPK";
    case Enc::Sopc: return "SOPC";
    case Enc::Sopp: return "SOPP";
    case Enc::Smem: return "SMEM";
    case Enc::Vop1: return "VOP1";
    case Enc::Vop2: return "VOP2";
    case Enc::Vop3: return "VOP3";
    case Enc::Vop3p: return "VOP3P";
    case Enc::Vopc: return "VOPC";
    case Enc::Ds: return "DS";
    case Enc::Flat: return "FLAT";
    case Enc::Mubuf: return "MUBUF";
    case Enc::Unknown: return "unknown";
  }
  return "unknown";
}

// How the assembler spells a swizzle's pattern. Four lanes choosing among
// their own four is a quad permute. Otherwise each lane of a group of 32
// reads the lane its own number becomes once it is ANDed with one mask, ORed
// with a second and XORed with a third, and the assembler names the common
// shapes of that -- a swap of groups, a reversal, a broadcast -- before
// falling back to spelling each of the five bits out.
std::string swizzle_text(uint32_t imm) {
  char b[64];
  if (imm & 0x8000) {
    std::snprintf(b, sizeof b, "swizzle(QUAD_PERM,%u,%u,%u,%u)", imm & 3, (imm >> 2) & 3, (imm >> 4) & 3,
                  (imm >> 6) & 3);
    return b;
  }
  const uint32_t and_mask = imm & 0x1F, or_mask = (imm >> 5) & 0x1F, xor_mask = (imm >> 10) & 0x1F;
  if (and_mask == 0x1F && or_mask == 0 && xor_mask && !(xor_mask & (xor_mask - 1))) {
    std::snprintf(b, sizeof b, "swizzle(SWAP,%u)", xor_mask);
    return b;
  }
  if (and_mask == 0x1F && or_mask == 0 && xor_mask && !((xor_mask + 1) & xor_mask)) {
    std::snprintf(b, sizeof b, "swizzle(REVERSE,%u)", xor_mask + 1);
    return b;
  }
  if (xor_mask == 0) {
    for (uint32_t size = 2; size <= 32; size *= 2)
      if (and_mask == (0x1Fu & ~(size - 1)) && or_mask < size) {
        std::snprintf(b, sizeof b, "swizzle(BROADCAST,%u,%u)", size, or_mask);
        return b;
      }
  }
  std::string bits = "swizzle(BITMASK_PERM,\"";
  for (int k = 4; k >= 0; --k) {
    const bool a = (and_mask >> k) & 1, o = (or_mask >> k) & 1, x = (xor_mask >> k) & 1;
    bits += a && !o ? (x ? 'i' : 'p') : ((o ^ x) ? '1' : '0');
  }
  return bits + "\")";
}

// The second word of a sub-dword instruction: which part of each source it
// reads, with or without the sign, and which part of the destination it
// writes. A source is a vector register unless its own bit says the field
// names one of the scalars instead.
void read_sdwa(Inst& in, const Shape& s, uint32_t w0, uint32_t w1, uint32_t srcs) {
  in.sdwa = true;
  if (const size_t at_e32 = in.name.rfind("_e32"); at_e32 != std::string::npos) in.name.resize(at_e32);
  in.name += "_sdwa";
  in.size = 8;
  in.dst_sel = (w1 >> 8) & 0x7;
  in.dst_unused = (w1 >> 11) & 0x3;
  in.clamp = ((w1 >> 13) & 1) != 0;
  if (const uint32_t omod = (w1 >> 14) & 0x3; omod)
    throw Error::make(Err::Unsupported, in.name, " uses an output multiplier (omod ", omod,
                      "), which this does not model");
  in.dst.push_back(vgpr((w0 >> 17) & 0xFF, s.dst_width));
  // Each source has its own group of bits: the part of the register, the
  // sign, the two modifiers, and whether it is a scalar register at all.
  for (uint32_t k = 0; k < srcs; ++k) {
    const uint32_t bits = k == 0 ? (w1 >> 16) & 0xFF : (w1 >> 24) & 0xFF;
    const uint32_t field = k == 0 ? w1 & 0xFF : (w0 >> 9) & 0xFF;
    const bool scalar = (bits >> 7) & 1;
    Operand o = scalar ? operand(field, s.src_width(k)) : vgpr(field, s.src_width(k));
    o.sel = bits & 0x7;
    o.sext = (bits >> 3) & 1;
    o.neg = (bits >> 4) & 1;
    o.abs = (bits >> 5) & 1;
    in.src.push_back(o);
  }
}

// The second word of a DPP instruction: which lane each lane reads its first
// source from, which lanes are written, and what happens where the lane to
// read is not there. The first source's register is in this word too, since
// the field that would have held it says "DPP" instead.
Operand read_dpp(Inst& in, uint32_t w1) {
  in.dpp = true;
  in.dpp_ctrl = (w1 >> 8) & 0x1FF;
  in.bound_ctrl = ((w1 >> 19) & 1) != 0;
  in.bank_mask = static_cast<uint8_t>((w1 >> 24) & 0xF);
  in.row_mask = static_cast<uint8_t>((w1 >> 28) & 0xF);
  Operand o;
  o.kind = OperandKind::Vgpr;
  o.index = w1 & 0xFF;
  o.neg = ((w1 >> 20) & 1) != 0;
  o.abs = ((w1 >> 21) & 1) != 0;
  return o;
}

// How the assembler spells a control.
std::string dpp_control_text(uint32_t ctrl) {
  char b[64];
  if (ctrl <= 0xFF) {
    std::snprintf(b, sizeof b, "quad_perm:[%u,%u,%u,%u]", ctrl & 3, (ctrl >> 2) & 3, (ctrl >> 4) & 3,
                  (ctrl >> 6) & 3);
    return b;
  }
  if (ctrl >= 0x101 && ctrl <= 0x10F) {
    std::snprintf(b, sizeof b, "row_shl:%u", ctrl - 0x100);
    return b;
  }
  if (ctrl >= 0x111 && ctrl <= 0x11F) {
    std::snprintf(b, sizeof b, "row_shr:%u", ctrl - 0x110);
    return b;
  }
  if (ctrl >= 0x121 && ctrl <= 0x12F) {
    std::snprintf(b, sizeof b, "row_ror:%u", ctrl - 0x120);
    return b;
  }
  switch (ctrl) {
    case 0x130: return "wave_shl:1";
    case 0x134: return "wave_rol:1";
    case 0x138: return "wave_shr:1";
    case 0x13C: return "wave_ror:1";
    case 0x140: return "row_mirror";
    case 0x141: return "row_half_mirror";
    case 0x142: return "row_bcast:15";
    case 0x143: return "row_bcast:31";
    default: break;
  }
  throw Error::make(Err::Unsupported, "a DPP control this does not decode yet (", ctrl, ")");
}

Inst decode(const std::vector<uint8_t>& code, uint64_t at, uint64_t pc) {
  const uint32_t w0 = word(code, at);
  Inst in;
  in.pc = pc;
  in.size = 4;
  // A literal follows the instruction where an operand asked for one.
  bool literal = false;
  const auto take = [&](uint32_t field, uint32_t width) {
    Operand o = operand(field, width);
    if (o.kind == OperandKind::Literal) literal = true;
    return o;
  };

  if ((w0 >> 23) == 0x17d) {          // SOP1
    in.enc = Enc::Sop1;
    in.opcode = (w0 >> 8) & 0xFF;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    if (s.dst_width) in.dst.push_back(sgpr((w0 >> 16) & 0x7F, s.dst_width));
    if (s.srcs > 0) in.src.push_back(take(w0 & 0xFF, s.src_width(0)));
  } else if ((w0 >> 23) == 0x17f) {   // SOPP
    in.enc = Enc::Sopp;
    in.opcode = (w0 >> 16) & 0x7F;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    in.simm = static_cast<int16_t>(w0 & 0xFFFF);
    if (in.name.rfind("s_branch", 0) == 0 || in.name.rfind("s_cbranch", 0) == 0)
      in.target = pc + 4 + static_cast<uint64_t>(static_cast<int64_t>(in.simm) * 4);
  } else if ((w0 >> 23) == 0x17e) {   // SOPC
    in.enc = Enc::Sopc;
    in.opcode = (w0 >> 16) & 0x7F;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    in.src.push_back(take(w0 & 0xFF, s.src_width(0)));
    in.src.push_back(take((w0 >> 8) & 0xFF, s.src_width(1)));
  } else if ((w0 >> 28) == 0xb) {     // SOPK
    in.enc = Enc::Sopk;
    in.opcode = (w0 >> 23) & 0x1F;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    if (s.dst_width) in.dst.push_back(sgpr((w0 >> 16) & 0x7F, s.dst_width));
    in.simm = static_cast<int16_t>(w0 & 0xFFFF);
  } else if ((w0 >> 30) == 0x2) {     // SOP2
    in.enc = Enc::Sop2;
    in.opcode = (w0 >> 23) & 0x7F;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    if (s.dst_width) in.dst.push_back(sgpr((w0 >> 16) & 0x7F, s.dst_width));
    in.src.push_back(take(w0 & 0xFF, s.src_width(0)));
    in.src.push_back(take((w0 >> 8) & 0xFF, s.src_width(1)));
  } else if ((w0 >> 26) == 0x30) {    // SMEM
    in.enc = Enc::Smem;
    in.opcode = (w0 >> 18) & 0xFF;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    in.size = 8;
    const uint32_t w1 = word(code, at + 4);
    in.dst.push_back(sgpr((w0 >> 6) & 0x7F, s.dst_width));
    if (s.srcs > 0) {
      in.src.push_back(sgpr(((w0 & 0x3F) << 1), s.src_width(0)));   // sbase counts register pairs
      in.offset = static_cast<int32_t>(w1 & 0x1FFFFF);
    }
  } else if ((w0 >> 25) == 0x3f) {    // VOP1
    in.enc = Enc::Vop1;
    in.opcode = (w0 >> 9) & 0xFF;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    if (s.scalar_dst) in.dst.push_back(sgpr((w0 >> 17) & 0xFF, s.dst_width));
    else in.dst.push_back(vgpr((w0 >> 17) & 0xFF, s.dst_width));
    if ((w0 & 0x1FF) == 250) {   // the first source comes from another lane
      in.size = 8;
      in.name = in.name.substr(0, in.name.size() - 4) + "_dpp";
      in.src.push_back(read_dpp(in, word(code, at + 4)));
    } else if ((w0 & 0x1FF) == 249) {   // the sub-dword form, which has one source here
      in.dst.clear();
      read_sdwa(in, s, w0, word(code, at + 4), 1);
    } else {
      in.src.push_back(take(w0 & 0x1FF, s.src_width(0)));
    }
  } else if ((w0 >> 25) == 0x3e) {    // VOPC
    in.enc = Enc::Vopc;
    in.opcode = (w0 >> 17) & 0xFF;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    Operand vcc;
    vcc.kind = OperandKind::Vcc;
    vcc.width = 2;
    in.dst.push_back(vcc);
    in.src.push_back(take(w0 & 0x1FF, s.src_width(0)));
    in.src.push_back(vgpr((w0 >> 9) & 0xFF, s.src_width(1)));
  } else if ((w0 >> 23) == 0x1a7) {   // VOP3P: the packed forms
    in.enc = Enc::Vop3p;
    in.opcode = (w0 >> 16) & 0x7F;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    in.size = 8;
    const uint32_t w1 = word(code, at + 4);
    in.clamp = (w0 >> 15) & 1;
    // The matrix instructions share the packed encoding and read its bits
    // their own way: which of the accumulation registers the result and the
    // addend live in, and three ways of broadcasting parts of the sources,
    // which are refused rather than guessed at.
    if (in.name.rfind("v_mfma", 0) == 0) {
      in.clamp = false;
      const uint32_t cbsz = (w0 >> 8) & 0x7, abid = (w0 >> 11) & 0xF, acc_cd = (w0 >> 15) & 1;
      const uint32_t acc = (w1 >> 27) & 0x3, blgp = (w1 >> 29) & 0x7;
      if (cbsz || abid || blgp)
        throw Error::make(Err::Unsupported, in.name, " broadcasts part of a source (cbsz ", cbsz, ", abid ", abid,
                          ", blgp ", blgp, "), which this does not model");
      in.dst.push_back(vgpr(w0 & 0xFF, s.dst_width));
      if (acc_cd) in.dst[0].kind = OperandKind::Agpr;
      for (uint32_t k = 0; k < s.srcs; ++k) {
        Operand o = take((w1 >> (9 * k)) & 0x1FF, s.src_width(k));
        if (k < 2 && ((acc >> k) & 1) && o.kind == OperandKind::Vgpr) o.kind = OperandKind::Agpr;
        if (k == 2 && acc_cd && o.kind == OperandKind::Vgpr) o.kind = OperandKind::Agpr;
        in.src.push_back(o);
      }
      return in;
    }
    // op_sel picks which half of each source feeds which half of the result.
    // Plain packed work is op_sel 0 with op_sel_hi all ones, and anything
    // else is a shuffle this does not model.
    const uint32_t op_sel = (w0 >> 11) & 0x7, neg_hi = (w0 >> 8) & 0x7;
    const uint32_t op_sel_hi = ((w0 >> 14) & 1) << 2 | ((w1 >> 27) & 0x3);
    in.op_sel = static_cast<uint8_t>(op_sel);
    in.op_sel_hi = static_cast<uint8_t>(op_sel_hi);
    // Taking the halves in their own order is a shuffle, and only the packed
    // float form is modelled beyond the plain arrangement: there a zero in
    // op_sel_hi means one value serves both halves, which is what the
    // compiler does with a constant.
    const bool mix = in.name.rfind("v_fma_mix", 0) == 0;
    if (mix && (op_sel != 0 || in.clamp))
      throw Error::make(Err::Unsupported, in.name, " takes a half from the top of a register or clamps its result",
                        ", which this does not model");
    if (!mix && (op_sel != 0 || (op_sel_hi != 0x7 && in.name.find("_f32") == std::string::npos)))
      throw Error::make(Err::Unsupported, in.name, " selects halves (op_sel ", op_sel, ", op_sel_hi ", op_sel_hi,
                        "), which this does not model");
    in.dst.push_back(vgpr(w0 & 0xFF, s.dst_width));
    const uint32_t neg = (w1 >> 29) & 0x7;
    for (uint32_t k = 0; k < s.srcs; ++k) {
      Operand o = take((w1 >> (9 * k)) & 0x1FF, s.src_width(k));
      o.neg = (neg >> k) & 1;
      if ((neg_hi >> k) & 1)
        throw Error::make(Err::Unsupported, in.name, " negates one half only, which this does not model");
      in.src.push_back(o);
    }
    // The two that move a value to or from an accumulation register: the
    // encoding numbers those in the vector registers' space, and which bank
    // is meant is the instruction itself.
    if (in.name == "v_accvgpr_read_b32") in.src[0].kind = OperandKind::Agpr;
    if (in.name == "v_accvgpr_write_b32") in.dst[0].kind = OperandKind::Agpr;
  } else if ((w0 >> 26) == 0x34) {    // VOP3
    in.enc = Enc::Vop3;
    in.opcode = (w0 >> 16) & 0x3FF;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    in.size = 8;
    const uint32_t w1 = word(code, at + 4);
    // The modifiers: a source's absolute value and negation, and a clamp of
    // the result. An output multiplier is refused rather than dropped, since
    // dropping one gives a wrong answer with nothing to show for it.
    const uint32_t abs = s.sdst ? 0 : (w0 >> 8) & 0x7;
    in.clamp = s.sdst ? false : ((w0 >> 15) & 1) != 0;
    if (const uint32_t omod = (w1 >> 27) & 0x3; omod)
      throw Error::make(Err::Unsupported, in.name, " uses an output multiplier (omod ", omod,
                        "), which this does not model");
    if (s.scalar_dst) in.dst.push_back(sgpr(w0 & 0xFF, s.dst_width));
    else in.dst.push_back(vgpr(w0 & 0xFF, s.dst_width));
    // VOP3b also writes a scalar pair: a carry out, or the condition
    // v_div_scale reports.
    if (s.sdst) in.dst.push_back(sgpr((w0 >> 8) & 0x7F, 2));
    const uint32_t neg = (w1 >> 29) & 0x7;
    for (uint32_t k = 0; k < s.srcs; ++k) {
      Operand o = take((w1 >> (9 * k)) & 0x1FF, s.src_width(k));
      o.neg = (neg >> k) & 1;
      o.abs = (abs >> k) & 1;
      in.src.push_back(o);
    }

  } else if ((w0 >> 26) == 0x36) {    // DS
    in.enc = Enc::Ds;
    in.opcode = (w0 >> 17) & 0xFF;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    in.size = 8;
    const uint32_t w1 = word(code, at + 4);
    // A two-address form (ds_read2, ds_write2) has an offset per address;
    // every other DS instruction has one 16-bit offset.
    if (in.name.find("read2") != std::string::npos || in.name.find("write2") != std::string::npos) {
      in.offset = w0 & 0xFF;
      in.offset1 = (w0 >> 8) & 0xFF;
    } else {
      in.offset = w0 & 0xFFFF;
    }
    if (s.dst_width) in.dst.push_back(vgpr((w1 >> 24) & 0xFF, s.dst_width));
    in.src.push_back(vgpr(w1 & 0xFF));                              // the address
    if (s.srcs > 1) in.src.push_back(vgpr((w1 >> 8) & 0xFF, s.src_width(1)));   // the data written
    if (s.srcs > 2) in.src.push_back(vgpr((w1 >> 16) & 0xFF));      // and the second, for a two-address write
  } else if ((w0 >> 26) == 0x38) {    // MUBUF: here only the cache operations
    in.enc = Enc::Mubuf;
    in.opcode = (w0 >> 18) & 0x7F;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    in.size = 8;
    in.cache = ((w0 >> 14) & 1) | ((w0 >> 17) & 1) << 1 | ((w0 >> 15) & 1) << 2;   // sc0, nt, sc1
  } else if ((w0 >> 26) == 0x37) {    // FLAT, and its global and scratch forms
    in.enc = Enc::Flat;
    in.opcode = (w0 >> 18) & 0x7F;
    const Shape& s = shape(in.enc, in.opcode);
    // The segment is in the encoding, and the assembler puts it in the name.
    const uint32_t seg = (w0 >> 14) & 0x3;
    in.segment = seg == 0 ? Inst::Segment::Flat : seg == 1 ? Inst::Segment::Scratch : Inst::Segment::Global;
    in.name = (seg == 0 ? "flat_" : seg == 1 ? "scratch_" : "global_") + std::string(s.name);
    in.size = 8;
    const uint32_t w1 = word(code, at + 4);
    in.offset = seg == 2 ? static_cast<int32_t>(w0 & 0x1FFF) << 19 >> 19   // global: signed
                         : static_cast<int32_t>(w0 & 0xFFF);               // flat and scratch: unsigned
    // The scope bits: sc0 and nt beside the offset, sc1 above the opcode.
    // They say how far a write is published; every access here is already
    // visible to every wave, so they change nothing and are kept for the
    // listing.
    in.cache = ((w0 >> 16) & 1) | ((w0 >> 17) & 1) << 1 | ((w0 >> 25) & 1) << 2;
    const uint32_t saddr = (w1 >> 16) & 0x7F;
    in.has_saddr = saddr != 0x7F;                    // 0x7f: the address is the VGPR pair's
    in.saddr = saddr;
    // A scratch access may have neither an address register nor a scalar
    // base: then the offset alone says where in the work-item's own memory.
    in.has_vaddr = in.segment != Inst::Segment::Scratch || ((w0 >> 13) & 1) != 0;
    // An atomic gives back the value it replaced where sc0 asks for it, and
    // the register it gives it back in is the one a load would write. A
    // compare-and-swap takes a pair and gives back one of them.
    const bool returns = std::string(s.name).rfind("atomic", 0) == 0 && (in.cache & 1);
    if (s.dst_width) in.dst.push_back(vgpr((w1 >> 24) & 0xFF, s.dst_width));
    else if (returns)
      in.dst.push_back(vgpr((w1 >> 24) & 0xFF,
                            std::string(s.name).find("cmpswap") != std::string::npos ? s.src_width(1) / 2
                                                                                     : s.src_width(1)));
    // A flat address is 64-bit; a global one is 64-bit unless a scalar base
    // carries the top of it; a scratch one is a 32-bit offset.
    const uint32_t addr_width = in.segment == Inst::Segment::Global ? (in.has_saddr ? 1 : 2)
                                                                   : in.segment == Inst::Segment::Flat ? 2 : 1;
    in.src.push_back(vgpr(w1 & 0xFF, addr_width));
    if (s.srcs > 1) in.src.push_back(vgpr((w1 >> 8) & 0xFF, s.src_width(1)));   // the data written
  } else if ((w0 >> 31) == 0) {       // VOP2
    in.enc = Enc::Vop2;
    in.opcode = (w0 >> 25) & 0x3F;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    const auto vcc = [] {
      Operand o;
      o.kind = OperandKind::Vcc;
      o.width = 2;
      return o;
    };
    if ((w0 & 0x1FF) == 249) {   // the sub-dword form
      read_sdwa(in, s, w0, word(code, at + 4), 2);
      return in;
    }
    const bool carry_out = in.name == "v_add_co_u32_e32" || in.name == "v_sub_co_u32_e32" ||
                           in.name == "v_addc_co_u32_e32" || in.name == "v_subb_co_u32_e32";
    const bool carry_in = in.name == "v_addc_co_u32_e32" || in.name == "v_subb_co_u32_e32";
    in.dst.push_back(vgpr((w0 >> 17) & 0xFF, s.dst_width));
    // The short forms of the carry arithmetic write VCC beside their result,
    // which the encoding does not spell out and the assembler does.
    if (carry_out) in.dst.push_back(vcc());
    if ((w0 & 0x1FF) == 250) {   // the first source comes from another lane
      in.size = 8;
      in.name = in.name.substr(0, in.name.size() - 4) + "_dpp";
      in.src.push_back(read_dpp(in, word(code, at + 4)));
      in.src.push_back(vgpr((w0 >> 9) & 0xFF, s.src_width(1)));
      if (carry_out || in.name.rfind("v_cndmask", 0) == 0)
        throw Error::make(Err::Unsupported, in.name,
                          " reads a lane of its own beside writing VCC, which this does not decode yet");
      return in;
    }
    in.src.push_back(take(w0 & 0x1FF, s.src_width(0)));
    // v_fmamk_f32 carries a constant of its own, which sits between the two
    // sources rather than taking one of their places.
    if (in.name == "v_fmamk_f32") {
      Operand k;
      k.kind = OperandKind::Literal;
      in.src.push_back(k);
      literal = true;
    }
    in.src.push_back(vgpr((w0 >> 9) & 0xFF, s.src_width(1)));
    if (in.name == "v_fmaak_f32") {
      Operand k;
      k.kind = OperandKind::Literal;
      in.src.push_back(k);
      literal = true;
    }
    // The short form of the select reads its condition from VCC, as the two
    // that take a carry in read theirs.
    if (in.name == "v_cndmask_b32_e32" || carry_in) in.src.push_back(vcc());
  } else {
    throw Error::make(Err::Unsupported, "an instruction encoding this does not decode yet");
  }

  if (literal) {
    const uint32_t value = word(code, at + in.size);
    // A 16-bit instruction reads the low half of the word, and the assembler
    // prints that half. A packed instruction is 16-bit twice over and takes
    // the whole word.
    const bool half = in.name.rfind("v_pk_", 0) != 0 &&
                      (in.name.find("_u16") != std::string::npos || in.name.find("_i16") != std::string::npos ||
                       in.name.find("_b16") != std::string::npos || in.name.find("_f16") != std::string::npos);
    for (Operand& o : in.src)
      if (o.kind == OperandKind::Literal) o.value = half ? (value & 0xFFFF) : value;
    in.size += 4;
  }
  return in;
}

std::string operand_text(const Operand& o) {
  char b[64];
  const std::string neg = o.neg ? "-" : "";
  // The assembler writes an absolute value as |v1|, with the negation outside it.
  const auto wrap = [&](const std::string& text) { return o.abs ? neg + "|" + text + "|" : neg + text; };
  const auto range = [&](const char* kind) {
    if (o.width <= 1) std::snprintf(b, sizeof b, "%s%u", kind, o.index);
    else std::snprintf(b, sizeof b, "%s[%u:%u]", kind, o.index, o.index + o.width - 1);
    return std::string(b);
  };
  switch (o.kind) {
    case OperandKind::Sgpr: return wrap(range("s"));
    case OperandKind::Vgpr: return wrap(o.sext ? "sext(" + range("v") + ")" : range("v"));
    case OperandKind::Agpr: return wrap(range("a"));
    case OperandKind::Vcc: return wrap(o.width >= 2 ? "vcc" : "vcc_lo");
    case OperandKind::Exec: return wrap("exec");
    case OperandKind::ExecLo: return wrap("exec_lo");
    case OperandKind::ExecHi: return wrap("exec_hi");
    case OperandKind::M0: return wrap("m0");
    case OperandKind::SharedBase: return wrap("src_shared_base");
    case OperandKind::InlineFloat:
      // As the assembler writes them: 1.0, -0.5, and, for one over two pi,
      // the digits of the float it stands for.
      if (o.fvalue > 0.159 && o.fvalue < 0.16) return neg + "0.15915494";
      std::snprintf(b, sizeof b, "%.1f", o.fvalue);
      return neg + b;
    case OperandKind::Inline:
      std::snprintf(b, sizeof b, "%lld", static_cast<long long>(o.value));
      return neg + b;
    case OperandKind::Literal: {
      // A literal is a word of the instruction stream, and the assembler
      // prints it in hex -- unless its value is one an inline constant could
      // have carried, which it writes as the number itself. (The compiler
      // spends a literal on such a value when the word is going to be
      // rewritten, as the address of a global is.)
      const int32_t v = static_cast<int32_t>(static_cast<uint32_t>(o.value));
      if (v >= -16 && v <= 64) std::snprintf(b, sizeof b, "%d", v);
      else std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(static_cast<uint32_t>(o.value)));
      return neg + b;
    }
    case OperandKind::None: break;
  }
  return "?";
}

std::string to_text(const Inst& i) {
  std::string s = i.name;
  std::string sep = " ";
  // A scratch access with no address register: the assembler writes "off"
  // where the register would be.
  const bool scratch_no_addr = i.enc == Enc::Flat && i.segment == Inst::Segment::Scratch && !i.has_vaddr;
  char b[64];
  for (const Operand& o : i.dst) {
    s += sep + operand_text(o);
    sep = ", ";
  }
  for (size_t k = 0; k < i.src.size(); ++k) {
    s += sep + (scratch_no_addr && k == 0 ? "off" : operand_text(i.src[k]));
    sep = ", ";
  }
  if (i.dpp) {
    char m[64];
    std::snprintf(m, sizeof m, " row_mask:0x%x bank_mask:0x%x", i.row_mask, i.bank_mask);
    s += " " + dpp_control_text(i.dpp_ctrl) + m;
    if (i.bound_ctrl) s += " bound_ctrl:1";
  }
  if (i.enc == Enc::Vop3p) {
    // The assembler prints these only where they are not the plain
    // arrangement: every source's low half to the low result, every source's
    // high half to the high one.
    const auto bits = [&](uint8_t v) {
      std::string out = "[";
      for (size_t k = 0; k < i.src.size(); ++k) out += (k ? "," : "") + std::to_string((v >> k) & 1);
      return out + "]";
    };
    // The mixed-precision forms read every source as a float unless op_sel_hi
    // says otherwise, so for them the plain arrangement is all zeroes.
    const uint8_t plain_hi = i.name.rfind("v_fma_mix", 0) == 0 ? 0 : 0x7;
    if (i.op_sel) s += " op_sel:" + bits(i.op_sel);
    if (i.op_sel_hi != plain_hi) s += " op_sel_hi:" + bits(i.op_sel_hi);
  }
  if (i.clamp) s += " clamp";
  if (i.sdwa) {
    static const char* kParts[8] = {"BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD", "?"};
    static const char* kUnused[4] = {"UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE", "?"};
    s += std::string(" dst_sel:") + kParts[i.dst_sel & 7] + " dst_unused:" + kUnused[i.dst_unused & 3];
    for (size_t k = 0; k < i.src.size(); ++k)
      s += " src" + std::to_string(k) + "_sel:" + kParts[i.src[k].sel & 7];
  }
  if (i.enc == Enc::Smem) {
    if (!i.src.empty()) {
      std::snprintf(b, sizeof b, ", 0x%x", i.offset);
      s += b;
    }
  } else if (i.enc == Enc::Ds) {
    const bool two = i.name.find("read2") != std::string::npos || i.name.find("write2") != std::string::npos;
    if (i.name == "ds_swizzle_b32") {
      if (i.offset) s += " offset:" + swizzle_text(static_cast<uint32_t>(i.offset));
    } else if (i.offset) {
      std::snprintf(b, sizeof b, two ? " offset0:%d" : " offset:%d", i.offset);
      s += b;
    }
    if (two && i.offset1) {
      std::snprintf(b, sizeof b, " offset1:%d", i.offset1);
      s += b;
    }
  } else if (i.enc == Enc::Flat) {
    // flat prints only its address and data; global and scratch print the
    // scalar base, or "off" where there is none.
    if (i.segment != Inst::Segment::Flat)
      s += i.has_saddr ? ", s[" + std::to_string(i.saddr) + ":" + std::to_string(i.saddr + 1) + "]" : ", off";
    if (i.offset) {
      std::snprintf(b, sizeof b, " offset:%d", i.offset);
      s += b;
    }
    if (i.cache & 1) s += " sc0";
    if (i.cache & 2) s += " nt";
    if (i.cache & 4) s += " sc1";
  } else if (i.enc == Enc::Mubuf) {
    if (i.cache & 1) s += " sc0";
    if (i.cache & 2) s += " nt";
    if (i.cache & 4) s += " sc1";
  } else if (i.name == "s_sendmsg") {
    const uint32_t msg = static_cast<uint32_t>(i.simm) & 0xF;
    if (msg != 1 || (static_cast<uint32_t>(i.simm) & 0xFFF0))
      throw Error::make(Err::Unsupported, "s_sendmsg with message ", static_cast<uint32_t>(i.simm) & 0xFFFF,
                        " is not decoded yet");
    s += " sendmsg(MSG_INTERRUPT)";
  } else if (i.name == "s_nop" || i.name == "s_sleep") {
    std::snprintf(b, sizeof b, " %d", i.simm);   // how many cycles to wait
    s += b;
  } else if (i.name == "s_waitcnt") {
    // vmcnt in bits 3:0 and 15:14, expcnt in 6:4, lgkmcnt in 11:8; a counter
    // at its maximum is not waited on, and is not printed.
    const uint32_t imm = static_cast<uint32_t>(i.simm) & 0xFFFF;
    const uint32_t vm = (imm & 0xF) | ((imm >> 14) & 0x3) << 4, exp = (imm >> 4) & 0x7, lgkm = (imm >> 8) & 0xF;
    if (vm != 0x3F) {
      std::snprintf(b, sizeof b, " vmcnt(%u)", vm);
      s += b;
    }
    if (exp != 0x7) {
      std::snprintf(b, sizeof b, " expcnt(%u)", exp);
      s += b;
    }
    if (lgkm != 0xF) {
      std::snprintf(b, sizeof b, " lgkmcnt(%u)", lgkm);
      s += b;
    }
  } else if (i.enc == Enc::Sopp && i.target) {
    // As the assembler writes it: the immediate itself, unsigned.
    std::snprintf(b, sizeof b, " %u", static_cast<uint32_t>(i.simm) & 0xFFFF);
    s += b;
  } else if (i.enc == Enc::Sopk) {
    std::snprintf(b, sizeof b, ", 0x%x", (unsigned)(i.simm & 0xFFFF));
    s += b;
  }
  return s;
}

}  // namespace vgpu::amd::gcn
