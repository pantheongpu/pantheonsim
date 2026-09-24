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
      {{Enc::Sop1, 0x11}, {"s_ff1_i32_b64", 1, 1, 2}},
      // Where the wave is: what a kernel adds a constant to, to reach a global.
      {{Enc::Sop1, 0x1c}, {"s_getpc_b64", 2, 0}},
      {{Enc::Sop1, 0x20}, {"s_and_saveexec_b64", 2, 1, 2}},
      {{Enc::Sop1, 0x23}, {"s_andn2_saveexec_b64", 2, 1, 2}},
      // SOP2: two scalar sources.
      {{Enc::Sop2, 0x00}, {"s_add_u32", 1, 2}},
      {{Enc::Sop2, 0x02}, {"s_add_i32", 1, 2}},
      {{Enc::Sop2, 0x03}, {"s_sub_i32", 1, 2}},
      {{Enc::Sop2, 0x04}, {"s_addc_u32", 1, 2}},
      {{Enc::Sop2, 0x0b}, {"s_cselect_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x0c}, {"s_and_b32", 1, 2}},
      {{Enc::Sop2, 0x0d}, {"s_and_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x0e}, {"s_or_b32", 1, 2}},
      {{Enc::Sop2, 0x0f}, {"s_or_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x10}, {"s_xor_b32", 1, 2}},
      {{Enc::Sop2, 0x11}, {"s_xor_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x13}, {"s_andn2_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x1d}, {"s_lshl_b64", 2, 2, 2, 1}},
      {{Enc::Sop2, 0x20}, {"s_ashr_i32", 1, 2}},
      {{Enc::Sop2, 0x24}, {"s_mul_i32", 1, 2}},
      // SOPK: a 16-bit immediate.
      {{Enc::Sopk, 0x00}, {"s_movk_i32", 1, 0}},
      {{Enc::Sopk, 0x0f}, {"s_mulk_i32", 1, 0}},
      // SOPP: an immediate, and no registers.
      {{Enc::Sopp, 0x00}, {"s_nop", 0, 0}},
      {{Enc::Sopp, 0x01}, {"s_endpgm", 0, 0}},
      {{Enc::Sopp, 0x02}, {"s_branch", 0, 0}},
      {{Enc::Sopp, 0x04}, {"s_cbranch_scc0", 0, 0}},
      {{Enc::Sopp, 0x05}, {"s_cbranch_scc1", 0, 0}},
      {{Enc::Sopp, 0x08}, {"s_cbranch_execz", 0, 0}},
      {{Enc::Sopp, 0x09}, {"s_cbranch_execnz", 0, 0}},
      {{Enc::Sopp, 0x0a}, {"s_barrier", 0, 0}},
      {{Enc::Sopp, 0x0c}, {"s_waitcnt", 0, 0}},
      // SOPC: a scalar comparison, which sets SCC.
      {{Enc::Sopc, 0x04}, {"s_cmp_lt_i32", 0, 2}},
      {{Enc::Sopc, 0x06}, {"s_cmp_eq_u32", 0, 2}},
      {{Enc::Sopc, 0x13}, {"s_cmp_lg_u64", 0, 2, 2, 2}},
      // SMEM: a scalar load through a 64-bit base address.
      {{Enc::Smem, 0x00}, {"s_load_dword", 1, 1, 2}},
      {{Enc::Smem, 0x01}, {"s_load_dwordx2", 2, 1, 2}},
      {{Enc::Smem, 0x02}, {"s_load_dwordx4", 4, 1, 2}},
      {{Enc::Smem, 0x03}, {"s_load_dwordx8", 8, 1, 2}},
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
      {{Enc::Vop1, 0x00a}, {"v_cvt_f16_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x00b}, {"v_cvt_f32_f16_e32", 1, 1}},
      {{Enc::Vop1, 0x00f}, {"v_cvt_f32_f64_e32", 1, 1, 2}},
      {{Enc::Vop1, 0x01c}, {"v_trunc_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x020}, {"v_exp_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x021}, {"v_log_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x022}, {"v_rcp_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x023}, {"v_rcp_iflag_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x025}, {"v_rcp_f64_e32", 2, 1, 2}},
      {{Enc::Vop1, 0x027}, {"v_sqrt_f32_e32", 1, 1}},
      {{Enc::Vop1, 0x02c}, {"v_bfrev_b32_e32", 1, 1}},
      {{Enc::Vop1, 0x02d}, {"v_ffbh_u32_e32", 1, 1}},
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
      {{Enc::Vop2, 0x019}, {"v_add_co_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x01a}, {"v_sub_co_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x01c}, {"v_addc_co_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x01d}, {"v_subb_co_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x034}, {"v_add_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x035}, {"v_sub_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x036}, {"v_subrev_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x03b}, {"v_fmac_f32_e32", 1, 2}},
      // The float comparisons share their opcodes with the long forms above.
      // A class test: the second source is a mask of the kinds of float
      // (NaN, infinity, normal, denormal, zero, each with a sign) it asks about.
      {{Enc::Vopc, 0x010}, {"v_cmp_class_f32_e32", 2, 2}},
      {{Enc::Vopc, 0x021}, {"v_cmp_lt_f16_e32", 2, 2}},
      {{Enc::Vopc, 0x022}, {"v_cmp_eq_f16_e32", 2, 2}},
      {{Enc::Vopc, 0x024}, {"v_cmp_gt_f16_e32", 2, 2}},
      {{Enc::Vopc, 0x026}, {"v_cmp_ge_f16_e32", 2, 2}},
      {{Enc::Vopc, 0x02b}, {"v_cmp_ngt_f16_e32", 2, 2}},
      {{Enc::Vopc, 0x02d}, {"v_cmp_neq_f16_e32", 2, 2}},
      {{Enc::Vopc, 0x041}, {"v_cmp_lt_f32_e32", 2, 2}},
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
      {{Enc::Vop3, 0x0c9}, {"v_cmp_lt_u32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x0ca}, {"v_cmp_eq_u32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x0cd}, {"v_cmp_ne_u32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x0ce}, {"v_cmp_ge_u32_e64", 2, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x101}, {"v_add_f32_e64", 1, 2}},
      // The long forms of the carry arithmetic, which name the pair they
      // write the carry to rather than always using VCC.
      {{Enc::Vop3, 0x11a}, {"v_sub_co_u32_e64", 1, 2, 1, 1, 1, true}},
      {{Enc::Vop3, 0x11d}, {"v_subb_co_u32_e64", 1, 3, 1, 1, 2, true}},
      {{Enc::Vop3, 0x11e}, {"v_subbrev_co_u32_e64", 1, 3, 1, 1, 2, true}},
      {{Enc::Vop3, 0x10b}, {"v_max_f32_e64", 1, 2}},
      {{Enc::Vop3, 0x100}, {"v_cndmask_b32_e64", 1, 3, 1, 1, 2}},
      {{Enc::Vop3, 0x1c8}, {"v_bfe_u32", 1, 3}},
      {{Enc::Vop3, 0x1c9}, {"v_bfe_i32", 1, 3}},
      {{Enc::Vop3, 0x1cb}, {"v_fma_f32", 1, 3}},
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
      {{Enc::Vop3, 0x1f3}, {"v_xad_u32", 1, 3}},
      {{Enc::Vop3, 0x1ff}, {"v_add3_u32", 1, 3}},
      {{Enc::Vop3, 0x200}, {"v_lshl_or_b32", 1, 3}},
      {{Enc::Vop3, 0x202}, {"v_or3_b32", 1, 3}},
      {{Enc::Vop3, 0x206}, {"v_fma_f16", 1, 3}},
      {{Enc::Vop3, 0x208}, {"v_lshl_add_u64", 2, 3, 2, 1, 2}},
      {{Enc::Vop3, 0x280}, {"v_add_f64", 2, 2, 2, 2}},
      {{Enc::Vop3, 0x281}, {"v_mul_f64", 2, 2, 2, 2}},
      {{Enc::Vop3, 0x285}, {"v_mul_lo_u32", 1, 2}},
      {{Enc::Vop3, 0x286}, {"v_mul_hi_u32", 1, 2}},
      {{Enc::Vop3, 0x287}, {"v_mul_hi_i32", 1, 2}},
      {{Enc::Vop3, 0x289}, {"v_readlane_b32", 1, 2, 1, 1, 1, false, true}},
      {{Enc::Vop3, 0x28a}, {"v_writelane_b32", 1, 2}},
      {{Enc::Vop3, 0x28b}, {"v_bcnt_u32_b32", 1, 2}},
      {{Enc::Vop3, 0x28c}, {"v_mbcnt_lo_u32_b32", 1, 2}},
      {{Enc::Vop3, 0x28d}, {"v_mbcnt_hi_u32_b32", 1, 2}},
      {{Enc::Vop3, 0x28f}, {"v_lshlrev_b64", 2, 2, 1, 2}},
      {{Enc::Vop3, 0x291}, {"v_ashrrev_i64", 2, 2, 1, 2}},
      // DS: LDS reads and writes. A read takes the address; a write takes the
      // address and the data.
      {{Enc::Ds, 0x00}, {"ds_add_u32", 0, 2}},
      {{Enc::Ds, 0x06}, {"ds_max_i32", 0, 2}},
      {{Enc::Ds, 0x0b}, {"ds_xor_b32", 0, 2}},
      {{Enc::Ds, 0x0d}, {"ds_write_b32", 0, 2}},
      {{Enc::Ds, 0x0e}, {"ds_write2_b32", 0, 3}},
      {{Enc::Ds, 0x37}, {"ds_read2_b32", 2, 1}},
      {{Enc::Ds, 0x36}, {"ds_read_b32", 1, 1}},
      {{Enc::Ds, 0x38}, {"ds_read2st64_b32", 2, 1}},
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
      {{Enc::Flat, 0x62}, {"atomic_add_x2", 0, 2, 1, 2}},
      // VOP3P: a packed pair of halves in one register, both computed at once.
      {{Enc::Vop3p, 0x0e}, {"v_pk_fma_f16", 1, 3}},
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
    o.kind = OperandKind::Vcc;
    o.width = 2;
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
  } else if (code >= 240 && code <= 247) {
    // The inline float constants, in the ISA's order.
    static const double kFloats[] = {0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0};
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
    case Enc::Unknown: return "unknown";
  }
  return "unknown";
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
    in.src.push_back(sgpr(((w0 & 0x3F) << 1), s.src_width(0)));   // sbase counts register pairs
    in.offset = static_cast<int32_t>(w1 & 0x1FFFFF);
  } else if ((w0 >> 25) == 0x3f) {    // VOP1
    in.enc = Enc::Vop1;
    in.opcode = (w0 >> 9) & 0xFF;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    if (s.scalar_dst) in.dst.push_back(sgpr((w0 >> 17) & 0xFF, s.dst_width));
    else in.dst.push_back(vgpr((w0 >> 17) & 0xFF, s.dst_width));
    in.src.push_back(take(w0 & 0x1FF, s.src_width(0)));
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
    // op_sel picks which half of each source feeds which half of the result.
    // Plain packed work is op_sel 0 with op_sel_hi all ones, and anything
    // else is a shuffle this does not model.
    const uint32_t op_sel = (w0 >> 11) & 0x7, neg_hi = (w0 >> 8) & 0x7;
    const uint32_t op_sel_hi = ((w0 >> 14) & 1) << 2 | ((w1 >> 27) & 0x3);
    if (op_sel != 0 || op_sel_hi != 0x7)
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
    if (s.srcs > 1) in.src.push_back(vgpr((w1 >> 8) & 0xFF));       // the data written
    if (s.srcs > 2) in.src.push_back(vgpr((w1 >> 16) & 0xFF));      // and the second, for a two-address write
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
    // The sub-dword form: the first source field says 249 instead of naming a
    // register, and a second word says which part of each register the
    // instruction reads and which part of the destination it writes.
    if ((w0 & 0x1FF) == 249) {
      in.sdwa = true;
      in.name = std::string(s.name);
      if (const size_t at_e32 = in.name.rfind("_e32"); at_e32 != std::string::npos) in.name.resize(at_e32);
      in.name += "_sdwa";
      in.size = 8;
      const uint32_t w1 = word(code, at + 4);
      in.dst_sel = (w1 >> 8) & 0x7;
      in.dst_unused = (w1 >> 11) & 0x3;
      in.clamp = ((w1 >> 13) & 1) != 0;
      if (const uint32_t omod = (w1 >> 14) & 0x3; omod)
        throw Error::make(Err::Unsupported, in.name, " uses an output multiplier (omod ", omod,
                          "), which this does not model");
      if (((w1 >> 22) & 1) || ((w1 >> 30) & 1))
        throw Error::make(Err::Unsupported, in.name,
                          " reads a scalar register through the sub-dword form, which this has never seen the "
                          "compiler emit and so does not decode");
      in.dst.push_back(vgpr((w0 >> 17) & 0xFF, s.dst_width));
      Operand src0 = vgpr(w1 & 0xFF, s.src_width(0));
      src0.sel = (w1 >> 16) & 0x7;
      src0.sext = ((w1 >> 19) & 1) != 0;
      src0.neg = ((w1 >> 20) & 1) != 0;
      src0.abs = ((w1 >> 21) & 1) != 0;
      Operand src1 = vgpr((w0 >> 9) & 0xFF, s.src_width(1));
      src1.sel = (w1 >> 24) & 0x7;
      src1.sext = ((w1 >> 27) & 1) != 0;
      src1.neg = ((w1 >> 28) & 1) != 0;
      src1.abs = ((w1 >> 29) & 1) != 0;
      in.src.push_back(src0);
      in.src.push_back(src1);
      return in;
    }
    const bool carry_out = in.name == "v_add_co_u32_e32" || in.name == "v_sub_co_u32_e32" ||
                           in.name == "v_addc_co_u32_e32" || in.name == "v_subb_co_u32_e32";
    const bool carry_in = in.name == "v_addc_co_u32_e32" || in.name == "v_subb_co_u32_e32";
    in.dst.push_back(vgpr((w0 >> 17) & 0xFF, s.dst_width));
    // The short forms of the carry arithmetic write VCC beside their result,
    // which the encoding does not spell out and the assembler does.
    if (carry_out) in.dst.push_back(vcc());
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
    case OperandKind::Vcc: return wrap("vcc");
    case OperandKind::Exec: return wrap("exec");
    case OperandKind::ExecLo: return wrap("exec_lo");
    case OperandKind::ExecHi: return wrap("exec_hi");
    case OperandKind::M0: return wrap("m0");
    case OperandKind::SharedBase: return wrap("src_shared_base");
    case OperandKind::InlineFloat:
      // As the assembler writes them: 1.0, -0.5, and so on.
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
  if (i.clamp) s += " clamp";
  if (i.sdwa) {
    static const char* kParts[8] = {"BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD", "?"};
    static const char* kUnused[4] = {"UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE", "?"};
    s += std::string(" dst_sel:") + kParts[i.dst_sel & 7] + " dst_unused:" + kUnused[i.dst_unused & 3];
    for (size_t k = 0; k < i.src.size(); ++k)
      s += " src" + std::to_string(k) + "_sel:" + kParts[i.src[k].sel & 7];
  }
  if (i.enc == Enc::Smem) {
    std::snprintf(b, sizeof b, ", 0x%x", i.offset);
    s += b;
  } else if (i.enc == Enc::Ds) {
    const bool two = i.name.find("read2") != std::string::npos || i.name.find("write2") != std::string::npos;
    if (i.offset) {
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
  } else if (i.name == "s_nop") {
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
