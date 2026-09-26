#include <cstdio>
#include <string>
#include <map>
#include <vector>

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
  // A long form made from a short one's entry (add_long_forms), which runs as
  // the short form does.
  bool promoted = false;
  uint32_t src_width(uint32_t i) const { return i == 0 ? w0 : i == 1 ? w1 : w2; }
};

// Every comparison VOPC and VOP3 have, generated rather than listed: the
// number says the type and the test, as the ISA's tables lay them out (the
// executor reads them the same way), in the short form writing VCC and the
// long form a scalar pair it names. The X forms, sixteen on from each, write
// EXEC as well.
void add_comparisons(std::map<std::pair<Enc, uint32_t>, Shape>& t) {
  static const char* kFloatTests[16] = {"f", "lt", "eq", "le", "gt", "lg", "ge", "o",
                                        "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"};
  static const char* kIntTests[8] = {"f", "lt", "eq", "le", "gt", "ne", "ge", "t"};
  static std::vector<std::string> names;   // the shapes point at their names
  names.reserve(4 * (3 * 16 + 6 * 8 + 3));
  const auto add_one = [&](uint32_t op, const std::string& name, uint32_t w0, uint32_t w1) {
    names.push_back(name + "_e32");
    t.emplace(std::make_pair(Enc::Vopc, op), Shape{names.back().c_str(), 2, 2, w0, w1});
    names.push_back(name + "_e64");
    t.emplace(std::make_pair(Enc::Vop3, op), Shape{names.back().c_str(), 2, 2, w0, w1, 1, false, true});
  };
  const auto add = [&](uint32_t op, const std::string& name, uint32_t w0, uint32_t w1) {
    add_one(op, name, w0, w1);
    // The class tests' X forms follow each directly; every other's sixteen on.
    add_one(op + (op < 0x20 ? 1 : 0x10), "v_cmpx" + name.substr(5), w0, w1);
  };
  const struct { uint32_t base; const char* type; uint32_t width; } floats[] = {
      {0x20, "f16", 1}, {0x40, "f32", 1}, {0x60, "f64", 2}};
  for (const auto& f : floats)
    for (uint32_t k = 0; k < 16; ++k) add(f.base + k, std::string("v_cmp_") + kFloatTests[k] + "_" + f.type, f.width, f.width);
  const struct { uint32_t base; const char* type; uint32_t width; } ints[] = {
      {0xa0, "i16", 1}, {0xa8, "u16", 1}, {0xc0, "i32", 1}, {0xc8, "u32", 1}, {0xe0, "i64", 2}, {0xe8, "u64", 2}};
  for (const auto& i : ints)
    for (uint32_t k = 0; k < 8; ++k) add(i.base + k, std::string("v_cmp_") + kIntTests[k] + "_" + i.type, i.width, i.width);
  add(0x10, "v_cmp_class_f32", 1, 1);
  add(0x12, "v_cmp_class_f64", 2, 1);
  add(0x14, "v_cmp_class_f16", 1, 1);
}

// The instructions decoded so far: those the compiler emits for the kernels
// this runs. Each is <encoding, opcode> -> its name and operand shape, from
// the CDNA3 ISA reference guide's opcode tables.
// The VOP3 form of every VOP1 and VOP2 instruction that has no entry of its
// own: the ISA puts them at 0x140 plus the VOP1 opcode and 0x100 plus the
// VOP2 one, doing the same arithmetic with VOP3's modifiers and any source in
// any slot. Optimized code rarely needs the long form of these, since the
// short one fits; unoptimized code uses it all the time. Anything that reads
// or writes VCC implicitly in its short form (v_cndmask, the carries) does
// something else in its long one, and those are listed by hand or refused.
void add_long_forms(std::map<std::pair<Enc, uint32_t>, Shape>& t) {
  static std::vector<std::string> names;   // the shapes point at their names
  names.reserve(256);
  std::vector<std::pair<uint32_t, Shape>> add;
  for (const auto& [key, sh] : t) {
    const auto [enc, op] = key;
    if (enc != Enc::Vop1 && enc != Enc::Vop2) continue;
    const std::string name = sh.name;
    if (name.size() < 4 || name.compare(name.size() - 4, 4, "_e32") != 0) continue;
    if (name.find("cndmask") != std::string::npos || name.find("_co_") != std::string::npos) continue;
    add.emplace_back((enc == Enc::Vop1 ? 0x140 : 0x100) + op, sh);
    names.push_back(name.substr(0, name.size() - 4) + "_e64");
    add.back().second.name = names.back().c_str();
    add.back().second.promoted = true;
  }
  for (const auto& [op, sh] : add) t.emplace(std::make_pair(Enc::Vop3, op), sh);   // a listed entry stays
}

const std::map<std::pair<Enc, uint32_t>, Shape>& table() {
  static const std::map<std::pair<Enc, uint32_t>, Shape> t = [] {
    std::map<std::pair<Enc, uint32_t>, Shape> m = {
      // SOP1: one scalar source, one scalar destination.
      {{Enc::Sop1, 0x00}, {"s_mov_b32", 1, 1}},
      {{Enc::Sop1, 0x01}, {"s_mov_b64", 2, 1, 2}},
      {{Enc::Sop1, 0x04}, {"s_not_b32", 1, 1}},
      {{Enc::Sop1, 0x05}, {"s_not_b64", 2, 1, 2}},
      // A move only where the last comparison held.
      {{Enc::Sop1, 0x02}, {"s_cmov_b32", 1, 1}},
      {{Enc::Sop1, 0x30}, {"s_abs_i32", 1, 1}},
      // Clear or set one bit of the destination, the rest of it kept.
      {{Enc::Sop1, 0x18}, {"s_bitset0_b32", 1, 1}},
      {{Enc::Sop1, 0x1a}, {"s_bitset1_b32", 1, 1}},
      {{Enc::Sop1, 0x08}, {"s_brev_b32", 1, 1}},
      {{Enc::Sop1, 0x0d}, {"s_bcnt1_i32_b64", 1, 1, 2}},
      {{Enc::Sop1, 0x11}, {"s_ff1_i32_b64", 1, 1, 2}},
      {{Enc::Sop1, 0x0c}, {"s_bcnt1_i32_b32", 1, 1}},
      {{Enc::Sop1, 0x10}, {"s_ff1_i32_b32", 1, 1}},
      // The first set bit from the top (flbit), or for the signed forms the
      // first bit that differs from the sign; -1 when there is none.
      {{Enc::Sop1, 0x12}, {"s_flbit_i32_b32", 1, 1}},
      {{Enc::Sop1, 0x13}, {"s_flbit_i32_b64", 1, 1, 2}},
      {{Enc::Sop1, 0x14}, {"s_flbit_i32", 1, 1}},
      {{Enc::Sop1, 0x16}, {"s_sext_i32_i8", 1, 1}},
      {{Enc::Sop1, 0x17}, {"s_sext_i32_i16", 1, 1}},
      // Where the wave is: what a kernel adds a constant to, to reach a global.
      {{Enc::Sop1, 0x1c}, {"s_getpc_b64", 2, 0}},
      // A call, and the return from it: one saves where to come back to
      // while it jumps, the other jumps back.
      {{Enc::Sop1, 0x1d}, {"s_setpc_b64", 0, 1, 2}},
      {{Enc::Sop1, 0x1e}, {"s_swappc_b64", 2, 1, 2}},
      {{Enc::Sop1, 0x20}, {"s_and_saveexec_b64", 2, 1, 2}},
      {{Enc::Sop1, 0x21}, {"s_or_saveexec_b64", 2, 1, 2}},
      {{Enc::Sop1, 0x22}, {"s_xor_saveexec_b64", 2, 1, 2}},
      {{Enc::Sop1, 0x23}, {"s_andn2_saveexec_b64", 2, 1, 2}},
      // SOP2: two scalar sources.
      {{Enc::Sop2, 0x00}, {"s_add_u32", 1, 2}},
      {{Enc::Sop2, 0x01}, {"s_sub_u32", 1, 2}},
      {{Enc::Sop2, 0x02}, {"s_add_i32", 1, 2}},
      {{Enc::Sop2, 0x03}, {"s_sub_i32", 1, 2}},
      {{Enc::Sop2, 0x04}, {"s_addc_u32", 1, 2}},
      {{Enc::Sop2, 0x05}, {"s_subb_u32", 1, 2}},
      {{Enc::Sop2, 0x06}, {"s_min_i32", 1, 2}},
      {{Enc::Sop2, 0x07}, {"s_min_u32", 1, 2}},
      {{Enc::Sop2, 0x08}, {"s_max_i32", 1, 2}},
      {{Enc::Sop2, 0x09}, {"s_max_u32", 1, 2}},
      {{Enc::Sop2, 0x0a}, {"s_cselect_b32", 1, 2}},
      {{Enc::Sop2, 0x0b}, {"s_cselect_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x0c}, {"s_and_b32", 1, 2}},
      {{Enc::Sop2, 0x0d}, {"s_and_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x0e}, {"s_or_b32", 1, 2}},
      {{Enc::Sop2, 0x0f}, {"s_or_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x10}, {"s_xor_b32", 1, 2}},
      {{Enc::Sop2, 0x11}, {"s_xor_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x12}, {"s_andn2_b32", 1, 2}},
      {{Enc::Sop2, 0x13}, {"s_andn2_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x15}, {"s_orn2_b64", 2, 2, 2, 2}},
      {{Enc::Sop2, 0x1c}, {"s_lshl_b32", 1, 2}},
      {{Enc::Sop2, 0x1d}, {"s_lshl_b64", 2, 2, 2, 1}},
      {{Enc::Sop2, 0x1e}, {"s_lshr_b32", 1, 2}},
      {{Enc::Sop2, 0x1f}, {"s_lshr_b64", 2, 2, 2, 1}},
      {{Enc::Sop2, 0x20}, {"s_ashr_i32", 1, 2}},
      {{Enc::Sop2, 0x21}, {"s_ashr_i64", 2, 2, 2, 1}},
      // A bit field: the second source's low bits say where it starts, its
      // bits 16 and up how wide it is.
      {{Enc::Sop2, 0x22}, {"s_bfm_b32", 1, 2}},
      {{Enc::Sop2, 0x25}, {"s_bfe_u32", 1, 2}},
      {{Enc::Sop2, 0x26}, {"s_bfe_i32", 1, 2}},
      {{Enc::Sop2, 0x28}, {"s_bfe_i64", 2, 2, 2, 1}},
      {{Enc::Sop2, 0x24}, {"s_mul_i32", 1, 2}},
      {{Enc::Sop2, 0x2c}, {"s_mul_hi_u32", 1, 2}},
      {{Enc::Sop2, 0x2d}, {"s_mul_hi_i32", 1, 2}},
      {{Enc::Sop2, 0x32}, {"s_pack_ll_b32_b16", 1, 2}},
      {{Enc::Sop2, 0x33}, {"s_pack_lh_b32_b16", 1, 2}},
      {{Enc::Sop2, 0x34}, {"s_pack_hh_b32_b16", 1, 2}},
      // The first source shifted left by one to four, and the second added.
      {{Enc::Sop2, 0x2e}, {"s_lshl1_add_u32", 1, 2}},
      {{Enc::Sop2, 0x2f}, {"s_lshl2_add_u32", 1, 2}},
      {{Enc::Sop2, 0x30}, {"s_lshl3_add_u32", 1, 2}},
      {{Enc::Sop2, 0x31}, {"s_lshl4_add_u32", 1, 2}},
      // SOPK: a 16-bit immediate.
      {{Enc::Sopk, 0x00}, {"s_movk_i32", 1, 0}},
      {{Enc::Sopk, 0x0e}, {"s_addk_i32", 1, 0}},
      {{Enc::Sopk, 0x0f}, {"s_mulk_i32", 1, 0}},
      // A hardware register read, or written from a constant that follows:
      // the immediate names the register and the field.
      {{Enc::Sopk, 0x11}, {"s_getreg_b32", 1, 0}},
      {{Enc::Sopk, 0x14}, {"s_setreg_imm32_b32", 0, 0}},
      // A call: where to come back to goes into the pair, and the immediate
      // says how far to jump.
      {{Enc::Sopk, 0x15}, {"s_call_b64", 2, 0}},
      // A comparison against the instruction's own constant: the register
      // field names what is compared, not where a result goes.
      {{Enc::Sopk, 0x02}, {"s_cmpk_eq_i32", 1, 0}},
      {{Enc::Sopk, 0x03}, {"s_cmpk_lg_i32", 1, 0}},
      {{Enc::Sopk, 0x04}, {"s_cmpk_gt_i32", 1, 0}},
      {{Enc::Sopk, 0x05}, {"s_cmpk_ge_i32", 1, 0}},
      {{Enc::Sopk, 0x06}, {"s_cmpk_lt_i32", 1, 0}},
      {{Enc::Sopk, 0x07}, {"s_cmpk_le_i32", 1, 0}},
      // The unsigned ones take their constant without its sign.
      {{Enc::Sopk, 0x08}, {"s_cmpk_eq_u32", 1, 0}},
      {{Enc::Sopk, 0x09}, {"s_cmpk_lg_u32", 1, 0}},
      {{Enc::Sopk, 0x0a}, {"s_cmpk_gt_u32", 1, 0}},
      {{Enc::Sopk, 0x0b}, {"s_cmpk_ge_u32", 1, 0}},
      {{Enc::Sopk, 0x0c}, {"s_cmpk_lt_u32", 1, 0}},
      {{Enc::Sopk, 0x0d}, {"s_cmpk_le_u32", 1, 0}},
      // SOPP: an immediate, and no registers.
      {{Enc::Sopp, 0x00}, {"s_nop", 0, 0}},
      {{Enc::Sopp, 0x01}, {"s_endpgm", 0, 0}},
      {{Enc::Sopp, 0x02}, {"s_branch", 0, 0}},
      // Wakes the work-group's waves that sleep: none here sleeps past what
      // s_sleep asked, so it does nothing.
      {{Enc::Sopp, 0x03}, {"s_wakeup", 0, 0}},
      // Invalidates the instruction cache: there is none.
      {{Enc::Sopp, 0x13}, {"s_icache_inv", 0, 0}},
      // The end of VGPR indexing (see s_set_gpr_idx_on).
      {{Enc::Sopp, 0x1c}, {"s_set_gpr_idx_off", 0, 0}},
      {{Enc::Sopp, 0x04}, {"s_cbranch_scc0", 0, 0}},
      {{Enc::Sopp, 0x05}, {"s_cbranch_scc1", 0, 0}},
      {{Enc::Sopp, 0x06}, {"s_cbranch_vccz", 0, 0}},
      {{Enc::Sopp, 0x07}, {"s_cbranch_vccnz", 0, 0}},
      {{Enc::Sopp, 0x08}, {"s_cbranch_execz", 0, 0}},
      {{Enc::Sopp, 0x09}, {"s_cbranch_execnz", 0, 0}},
      {{Enc::Sopp, 0x0a}, {"s_barrier", 0, 0}},
      {{Enc::Sopp, 0x0c}, {"s_waitcnt", 0, 0}},
      {{Enc::Sopp, 0x0e}, {"s_sleep", 0, 0}},
      // A wave's priority against the others: nothing here schedules by it.
      {{Enc::Sopp, 0x0f}, {"s_setprio", 0, 0}},
      // A message to the host: MSG_INTERRUPT is how a kernel wakes the host
      // for a hostcall, which is what device-side printf is built on.
      {{Enc::Sopp, 0x10}, {"s_sendmsg", 0, 0}},
      // A trap into the handler: 2 is llvm.trap, what __builtin_trap and
      // abort() compile to, and what unoptimized device library code keeps on
      // the paths optimization proves unreachable.
      {{Enc::Sopp, 0x12}, {"s_trap", 0, 0}},
      // SOPC: a scalar comparison, which sets SCC.
      {{Enc::Sopc, 0x00}, {"s_cmp_eq_i32", 0, 2}},
      {{Enc::Sopc, 0x01}, {"s_cmp_lg_i32", 0, 2}},
      {{Enc::Sopc, 0x02}, {"s_cmp_gt_i32", 0, 2}},
      {{Enc::Sopc, 0x03}, {"s_cmp_ge_i32", 0, 2}},
      {{Enc::Sopc, 0x04}, {"s_cmp_lt_i32", 0, 2}},
      {{Enc::Sopc, 0x05}, {"s_cmp_le_i32", 0, 2}},
      {{Enc::Sopc, 0x06}, {"s_cmp_eq_u32", 0, 2}},
      {{Enc::Sopc, 0x07}, {"s_cmp_lg_u32", 0, 2}},
      {{Enc::Sopc, 0x08}, {"s_cmp_gt_u32", 0, 2}},
      {{Enc::Sopc, 0x09}, {"s_cmp_ge_u32", 0, 2}},
      {{Enc::Sopc, 0x0a}, {"s_cmp_lt_u32", 0, 2}},
      {{Enc::Sopc, 0x0b}, {"s_cmp_le_u32", 0, 2}},
      // Whether one bit of the first source is clear, or set.
      {{Enc::Sopc, 0x0c}, {"s_bitcmp0_b32", 0, 2}},
      {{Enc::Sopc, 0x0d}, {"s_bitcmp1_b32", 0, 2}},
      {{Enc::Sopc, 0x12}, {"s_cmp_eq_u64", 0, 2, 2, 2}},
      // VGPR indexing: from here to s_set_gpr_idx_off, the vector registers
      // the named operands name are offset by the first source.
      {{Enc::Sopc, 0x11}, {"s_set_gpr_idx_on", 0, 1}},
      {{Enc::Sopc, 0x13}, {"s_cmp_lg_u64", 0, 2, 2, 2}},
      // SMEM: a scalar load through a 64-bit base address.
      {{Enc::Smem, 0x00}, {"s_load_dword", 1, 1, 2}},
      {{Enc::Smem, 0x01}, {"s_load_dwordx2", 2, 1, 2}},
      {{Enc::Smem, 0x02}, {"s_load_dwordx4", 4, 1, 2}},
      {{Enc::Smem, 0x03}, {"s_load_dwordx8", 8, 1, 2}},
      {{Enc::Smem, 0x04}, {"s_load_dwordx16", 16, 1, 2}},
      // The same through a buffer resource (V#), four registers: what a
      // kernel reads its constants through when they are in a buffer.
      {{Enc::Smem, 0x08}, {"s_buffer_load_dword", 1, 1, 4}},
      {{Enc::Smem, 0x09}, {"s_buffer_load_dwordx2", 2, 1, 4}},
      {{Enc::Smem, 0x0a}, {"s_buffer_load_dwordx4", 4, 1, 4}},
      {{Enc::Smem, 0x0b}, {"s_buffer_load_dwordx8", 8, 1, 4}},
      {{Enc::Smem, 0x0c}, {"s_buffer_load_dwordx16", 16, 1, 4}},
      // A counter the wave reads: no address, and nothing but the pair it
      // writes.
      {{Enc::Smem, 0x24}, {"s_memtime", 2, 0}},
      // Scalar stores, and the scalar cache's write-back and invalidate.
      {{Enc::Smem, 0x10}, {"s_store_dword", 1, 1, 2}},
      {{Enc::Smem, 0x11}, {"s_store_dwordx2", 2, 1, 2}},
      {{Enc::Smem, 0x12}, {"s_store_dwordx4", 4, 1, 2}},
      {{Enc::Smem, 0x20}, {"s_dcache_inv", 0, 0}},
      {{Enc::Smem, 0x21}, {"s_dcache_wb", 0, 0}},
      // Scalar atomics: the data register is the value, and with glc gets
      // back what memory held. hipBLASLt's kernels count work-groups with them.
      {{Enc::Smem, 0x80}, {"s_atomic_swap", 1, 1, 2}},
      {{Enc::Smem, 0x81}, {"s_atomic_cmpswap", 2, 1, 2}},
      {{Enc::Smem, 0x82}, {"s_atomic_add", 1, 1, 2}},
      {{Enc::Smem, 0x83}, {"s_atomic_sub", 1, 1, 2}},
      {{Enc::Smem, 0x84}, {"s_atomic_smin", 1, 1, 2}},
      {{Enc::Smem, 0x85}, {"s_atomic_umin", 1, 1, 2}},
      {{Enc::Smem, 0x86}, {"s_atomic_smax", 1, 1, 2}},
      {{Enc::Smem, 0x87}, {"s_atomic_umax", 1, 1, 2}},
      {{Enc::Smem, 0x88}, {"s_atomic_and", 1, 1, 2}},
      {{Enc::Smem, 0x89}, {"s_atomic_or", 1, 1, 2}},
      {{Enc::Smem, 0x8a}, {"s_atomic_xor", 1, 1, 2}},
      {{Enc::Smem, 0x8b}, {"s_atomic_inc", 1, 1, 2}},
      {{Enc::Smem, 0x8c}, {"s_atomic_dec", 1, 1, 2}},
      {{Enc::Smem, 0xa0}, {"s_atomic_swap_x2", 2, 1, 2}},
      {{Enc::Smem, 0xa1}, {"s_atomic_cmpswap_x2", 4, 1, 2}},
      {{Enc::Smem, 0xa2}, {"s_atomic_add_x2", 2, 1, 2}},
      {{Enc::Smem, 0xa3}, {"s_atomic_sub_x2", 2, 1, 2}},
      {{Enc::Smem, 0xa4}, {"s_atomic_smin_x2", 2, 1, 2}},
      {{Enc::Smem, 0xa5}, {"s_atomic_umin_x2", 2, 1, 2}},
      {{Enc::Smem, 0xa6}, {"s_atomic_smax_x2", 2, 1, 2}},
      {{Enc::Smem, 0xa7}, {"s_atomic_umax_x2", 2, 1, 2}},
      {{Enc::Smem, 0xa8}, {"s_atomic_and_x2", 2, 1, 2}},
      {{Enc::Smem, 0xa9}, {"s_atomic_or_x2", 2, 1, 2}},
      {{Enc::Smem, 0xaa}, {"s_atomic_xor_x2", 2, 1, 2}},
      {{Enc::Smem, 0xab}, {"s_atomic_inc_x2", 2, 1, 2}},
      {{Enc::Smem, 0xac}, {"s_atomic_dec_x2", 2, 1, 2}},
      {{Enc::Smem, 0x25}, {"s_memrealtime", 2, 0}},
      // VOP1 and VOP2, the vector ALU's short forms.
      {{Enc::Vop1, 0x00}, {"v_nop", 0, 0}},
      {{Enc::Vop1, 0x01}, {"v_mov_b32_e32", 1, 1}},
      {{Enc::Vop1, 0x0d}, {"v_cvt_flr_i32_f32_e32", 1, 1}},
      // One byte of the source, widened to a float.
      {{Enc::Vop1, 0x12}, {"v_cvt_f32_ubyte1_e32", 1, 1}},
      {{Enc::Vop1, 0x13}, {"v_cvt_f32_ubyte2_e32", 1, 1}},
      {{Enc::Vop1, 0x14}, {"v_cvt_f32_ubyte3_e32", 1, 1}},
      {{Enc::Vop1, 0x2e}, {"v_ffbl_b32_e32", 1, 1}},
      {{Enc::Vop1, 0x2f}, {"v_ffbh_i32_e32", 1, 1}},
      {{Enc::Vop1, 0x30}, {"v_frexp_exp_i32_f64_e32", 1, 1, 2}},
      {{Enc::Vop1, 0x31}, {"v_frexp_mant_f64_e32", 2, 1, 2}},
      {{Enc::Vop1, 0x32}, {"v_fract_f64_e32", 2, 1, 2}},
      // Halves: to and from 16-bit integers, and one value at a time through
      // the transcendental and rounding units.
      {{Enc::Vop1, 0x39}, {"v_cvt_f16_u16_e32", 1, 1}},
      {{Enc::Vop1, 0x3a}, {"v_cvt_f16_i16_e32", 1, 1}},
      {{Enc::Vop1, 0x3b}, {"v_cvt_u16_f16_e32", 1, 1}},
      {{Enc::Vop1, 0x3c}, {"v_cvt_i16_f16_e32", 1, 1}},
      {{Enc::Vop1, 0x3d}, {"v_rcp_f16_e32", 1, 1}},
      {{Enc::Vop1, 0x3e}, {"v_sqrt_f16_e32", 1, 1}},
      {{Enc::Vop1, 0x3f}, {"v_rsq_f16_e32", 1, 1}},
      {{Enc::Vop1, 0x40}, {"v_log_f16_e32", 1, 1}},
      {{Enc::Vop1, 0x41}, {"v_exp_f16_e32", 1, 1}},
      {{Enc::Vop1, 0x44}, {"v_floor_f16_e32", 1, 1}},
      {{Enc::Vop1, 0x45}, {"v_ceil_f16_e32", 1, 1}},
      {{Enc::Vop1, 0x46}, {"v_trunc_f16_e32", 1, 1}},
      {{Enc::Vop1, 0x47}, {"v_rndne_f16_e32", 1, 1}},
      // Two registers trade places, lane by lane.
      {{Enc::Vop1, 0x51}, {"v_swap_b32", 1, 1}},
      // gfx950's: a pseudo-random step, two swaps across halves of the wave
      // (each writes both registers), and a bfloat16 widened to a float.
      {{Enc::Vop1, 0x58}, {"v_prng_b32_e32", 1, 1}},
      {{Enc::Vop1, 0x59}, {"v_permlane16_swap_b32_e32", 1, 1}},
      {{Enc::Vop1, 0x5a}, {"v_permlane32_swap_b32_e32", 1, 1}},
      {{Enc::Vop1, 0x5b}, {"v_cvt_f32_bf16_e32", 1, 1}},
      // One accumulation register copied to another.
      {{Enc::Vop1, 0x52}, {"v_accvgpr_mov_b32", 1, 1}},
      {{Enc::Vop1, 0x15}, {"v_cvt_u32_f64_e32", 1, 1, 2}},
      {{Enc::Vop2, 0x00}, {"v_cndmask_b32_e32", 1, 2}},
      {{Enc::Vop2, 0x01}, {"v_add_f32_e32", 1, 2}},
      {{Enc::Vop2, 0x02}, {"v_sub_f32_e32", 1, 2}},
      {{Enc::Vop2, 0x03}, {"v_subrev_f32_e32", 1, 2}},
      {{Enc::Vop2, 0x09}, {"v_mul_hi_u32_u24_e32", 1, 2}},
      {{Enc::Vop2, 0x21}, {"v_subrev_f16_e32", 1, 2}},
      {{Enc::Vop2, 0x28}, {"v_subrev_u16_e32", 1, 2}},
      {{Enc::Vop2, 0x2c}, {"v_ashrrev_i16_e32", 1, 2}},
      {{Enc::Vop2, 0x2f}, {"v_max_u16_e32", 1, 2}},
      {{Enc::Vop2, 0x30}, {"v_max_i16_e32", 1, 2}},
      {{Enc::Vop2, 0x31}, {"v_min_u16_e32", 1, 2}},
      {{Enc::Vop2, 0x32}, {"v_min_i16_e32", 1, 2}},
      {{Enc::Vop2, 0x3d}, {"v_xnor_b32_e32", 1, 2}},
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
      // An 8-bit float (fp8, or bf8 with a wider exponent) widened to a
      // float, or two of them to a pair.
      {{Enc::Vop1, 0x054}, {"v_cvt_f32_fp8_e32", 1, 1}},
      {{Enc::Vop1, 0x055}, {"v_cvt_f32_bf8_e32", 1, 1}},
      {{Enc::Vop1, 0x056}, {"v_cvt_pk_f32_fp8_e32", 2, 1}},
      {{Enc::Vop1, 0x057}, {"v_cvt_pk_f32_bf8_e32", 2, 1}},
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
      {{Enc::Vop2, 0x01b}, {"v_subrev_co_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x01e}, {"v_subbrev_co_u32_e32", 1, 2}},
      {{Enc::Vop2, 0x00c}, {"v_min_i32_e32", 1, 2}},
      {{Enc::Vop2, 0x00d}, {"v_max_i32_e32", 1, 2}},
      {{Enc::Vop2, 0x02b}, {"v_lshrrev_b16_e32", 1, 2}},
      {{Enc::Vop2, 0x02d}, {"v_max_f16_e32", 1, 2}},
      {{Enc::Vop2, 0x02e}, {"v_min_f16_e32", 1, 2}},
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
      {{Enc::Vop3, 0x114}, {"v_or_b32_e64", 1, 2}},
      {{Enc::Vop3, 0x119}, {"v_add_co_u32_e64", 1, 2, 1, 1, 1, true}},
      {{Enc::Vop3, 0x11c}, {"v_addc_co_u32_e64", 1, 3, 1, 1, 2, true}},
      {{Enc::Vop3, 0x134}, {"v_add_u32_e64", 1, 2}},
      {{Enc::Vop3, 0x135}, {"v_sub_u32_e64", 1, 2}},
      {{Enc::Vop3, 0x13b}, {"v_fmac_f32_e64", 1, 2}},
      {{Enc::Vop3, 0x1c2}, {"v_mad_i32_i24", 1, 3}},
      {{Enc::Vop3, 0x1d2}, {"v_min3_u32", 1, 3}},
      {{Enc::Vop3, 0x1e9}, {"v_mad_i64_i32", 2, 3, 1, 1, 2, true}},
      {{Enc::Vop3, 0x1c3}, {"v_mad_u32_u24", 1, 3}},
      {{Enc::Vop3, 0x1c8}, {"v_bfe_u32", 1, 3}},
      {{Enc::Vop3, 0x1d0}, {"v_min3_f32", 1, 3}},
      {{Enc::Vop3, 0x1d1}, {"v_min3_i32", 1, 3}},
      {{Enc::Vop3, 0x1d3}, {"v_max3_f32", 1, 3}},
      {{Enc::Vop3, 0x1d4}, {"v_max3_i32", 1, 3}},
      {{Enc::Vop3, 0x1d6}, {"v_med3_f32", 1, 3}},
      {{Enc::Vop3, 0x1d7}, {"v_med3_i32", 1, 3}},
      {{Enc::Vop3, 0x1d8}, {"v_med3_u32", 1, 3}},
      // The 16-bit forms of the three-way minimum and maximum, and a 16-bit
      // multiply added into 32 bits.
      {{Enc::Vop3, 0x1f1}, {"v_mad_u32_u16", 1, 3}},
      {{Enc::Vop3, 0x1f5}, {"v_min3_i16", 1, 3}},
      {{Enc::Vop3, 0x1f6}, {"v_min3_u16", 1, 3}},
      {{Enc::Vop3, 0x1f8}, {"v_max3_i16", 1, 3}},
      {{Enc::Vop3, 0x1f9}, {"v_max3_u16", 1, 3}},
      {{Enc::Vop3, 0x207}, {"v_div_fixup_f16", 1, 3}},
      // 53 bits of 2/pi, for reducing a large argument of sine or cosine.
      {{Enc::Vop3, 0x292}, {"v_trig_preop_f64", 2, 2, 2, 1}},
      {{Enc::Vop3, 0x293}, {"v_bfm_b32", 1, 2}},
      {{Enc::Vop3, 0x297}, {"v_cvt_pk_u16_u32", 1, 2}},
      // Signed 32-bit add and subtract, which clamp where asked.
      {{Enc::Vop3, 0x29c}, {"v_add_i32", 1, 2}},
      {{Enc::Vop3, 0x29d}, {"v_sub_i32", 1, 2}},
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
      // gfx950's: any function of three inputs, bit by bit, by its truth
      // table; and two floats to a packed pair of halves or bfloat16s,
      // rounded to nearest.
      {{Enc::Vop3, 0x233}, {"v_bitop3_b16", 1, 3}},
      {{Enc::Vop3, 0x234}, {"v_bitop3_b32", 1, 3}},
      {{Enc::Vop3, 0x267}, {"v_cvt_pk_f16_f32", 1, 2}},
      {{Enc::Vop3, 0x268}, {"v_cvt_pk_bf16_f32", 1, 2}},
      {{Enc::Vop3, 0x290}, {"v_lshrrev_b64", 2, 2, 1, 2}},
      {{Enc::Vop3, 0x291}, {"v_ashrrev_i64", 2, 2, 1, 2}},
      {{Enc::Vop3, 0x2a0}, {"v_pack_b32_f16", 1, 2}},
      // And floats narrowed to them: two into one half of the destination,
      // rounded to nearest, or one into one byte, rounded as its second
      // source's random bits say.
      {{Enc::Vop3, 0x2a2}, {"v_cvt_pk_fp8_f32", 1, 2}},
      {{Enc::Vop3, 0x2a3}, {"v_cvt_pk_bf8_f32", 1, 2}},
      {{Enc::Vop3, 0x2a4}, {"v_cvt_sr_fp8_f32", 1, 2}},
      {{Enc::Vop3, 0x2a5}, {"v_cvt_sr_bf8_f32", 1, 2}},
      // DS: LDS reads and writes. A read takes the address; a write takes the
      // address and the data.
      {{Enc::Ds, 0x00}, {"ds_add_u32", 0, 2}},
      {{Enc::Ds, 0x05}, {"ds_min_i32", 0, 2}},
      {{Enc::Ds, 0x06}, {"ds_max_i32", 0, 2}},
      {{Enc::Ds, 0x07}, {"ds_min_u32", 0, 2}},
      {{Enc::Ds, 0x08}, {"ds_max_u32", 0, 2}},
      {{Enc::Ds, 0x09}, {"ds_and_b32", 0, 2}},
      {{Enc::Ds, 0x0a}, {"ds_or_b32", 0, 2}},
      // The forms that hand back what they found: an add, and a compare
      // that stores its second value where the first matched.
      {{Enc::Ds, 0x20}, {"ds_add_rtn_u32", 1, 2}},
      {{Enc::Ds, 0x30}, {"ds_cmpst_rtn_b32", 1, 3}},
      {{Enc::Ds, 0x3b}, {"ds_read_i16", 1, 1}},
      // 64-bit atomics over a register pair.
      {{Enc::Ds, 0x40}, {"ds_add_u64", 0, 2, 1, 2}},
      {{Enc::Ds, 0x45}, {"ds_min_i64", 0, 2, 1, 2}},
      {{Enc::Ds, 0x46}, {"ds_max_i64", 0, 2, 1, 2}},
      {{Enc::Ds, 0x47}, {"ds_min_u64", 0, 2, 1, 2}},
      {{Enc::Ds, 0x48}, {"ds_max_u64", 0, 2, 1, 2}},
      {{Enc::Ds, 0x5c}, {"ds_add_f64", 0, 2, 1, 2}},
      {{Enc::Ds, 0x70}, {"ds_cmpst_rtn_b64", 2, 3, 1, 2, 2}},
      {{Enc::Ds, 0xde}, {"ds_write_b96", 0, 2, 1, 3}},
      {{Enc::Ds, 0x0b}, {"ds_xor_b32", 0, 2}},
      {{Enc::Ds, 0x0d}, {"ds_write_b32", 0, 2}},
      {{Enc::Ds, 0x0e}, {"ds_write2_b32", 0, 3}},
      {{Enc::Ds, 0x0f}, {"ds_write2st64_b32", 0, 3}},
      {{Enc::Ds, 0x4e}, {"ds_write2_b64", 0, 3, 1, 2, 2}},
      {{Enc::Ds, 0x4f}, {"ds_write2st64_b64", 0, 3, 1, 2, 2}},
      {{Enc::Ds, 0x77}, {"ds_read2_b64", 4, 1}},
      {{Enc::Ds, 0x78}, {"ds_read2st64_b64", 4, 1}},
      {{Enc::Ds, 0xfe}, {"ds_read_b96", 3, 1}},
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
      // A byte or a short into one half of a register, the other half kept,
      // and the top half of one stored.
      {{Enc::Ds, 0x54}, {"ds_write_b8_d16_hi", 0, 2}},
      {{Enc::Ds, 0x55}, {"ds_write_b16_d16_hi", 0, 2}},
      {{Enc::Ds, 0x56}, {"ds_read_u8_d16", 1, 1}},
      {{Enc::Ds, 0x57}, {"ds_read_u8_d16_hi", 1, 1}},
      {{Enc::Ds, 0x58}, {"ds_read_i8_d16", 1, 1}},
      {{Enc::Ds, 0x59}, {"ds_read_i8_d16_hi", 1, 1}},
      {{Enc::Ds, 0x5a}, {"ds_read_u16_d16", 1, 1}},
      {{Enc::Ds, 0x5b}, {"ds_read_u16_d16_hi", 1, 1}},
      {{Enc::Ds, 0x4d}, {"ds_write_b64", 0, 2, 1, 2}},
      {{Enc::Ds, 0x76}, {"ds_read_b64", 2, 1}},
      {{Enc::Ds, 0xdf}, {"ds_write_b128", 0, 2, 1, 4}},
      {{Enc::Ds, 0xff}, {"ds_read_b128", 4, 1}},
      // Lanes trading values without touching LDS at all: the offset is a
      // pattern saying which lane each one reads, and the data register sits
      // where an address would.
      {{Enc::Ds, 0x3d}, {"ds_swizzle_b32", 1, 1}},
      // A lane reads the value another lane holds: the address says which.
      {{Enc::Ds, 0x3e}, {"ds_permute_b32", 1, 2}},
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
      // The top half of the register, stored as a short.
      {{Enc::Flat, 0x1b}, {"store_short_d16_hi", 0, 2}},
      {{Enc::Flat, 0x19}, {"store_byte_d16_hi", 0, 2}},
      {{Enc::Flat, 0x20}, {"load_ubyte_d16", 1, 1}},
      {{Enc::Flat, 0x21}, {"load_ubyte_d16_hi", 1, 1}},
      {{Enc::Flat, 0x22}, {"load_sbyte_d16", 1, 1}},
      {{Enc::Flat, 0x23}, {"load_sbyte_d16_hi", 1, 1}},
      {{Enc::Flat, 0x24}, {"load_short_d16", 1, 1}},
      {{Enc::Flat, 0x25}, {"load_short_d16_hi", 1, 1}},
      {{Enc::Flat, 0x1c}, {"store_dword", 0, 2}},
      {{Enc::Flat, 0x1d}, {"store_dwordx2", 0, 2, 1, 2}},
      {{Enc::Flat, 0x1e}, {"store_dwordx3", 0, 2, 1, 3}},
      {{Enc::Flat, 0x1f}, {"store_dwordx4", 0, 2, 1, 4}},
      {{Enc::Flat, 0x40}, {"atomic_swap", 0, 2}},
      {{Enc::Flat, 0x41}, {"atomic_cmpswap", 0, 2, 1, 2}},
      {{Enc::Flat, 0x42}, {"atomic_add", 0, 2}},
      {{Enc::Flat, 0x4f}, {"atomic_add_f64", 0, 2, 1, 2}},
      {{Enc::Flat, 0x50}, {"atomic_min_f64", 0, 2, 1, 2}},
      {{Enc::Flat, 0x51}, {"atomic_max_f64", 0, 2, 1, 2}},
      {{Enc::Flat, 0x43}, {"atomic_sub", 0, 2}},
      {{Enc::Flat, 0x48}, {"atomic_and", 0, 2}},
      {{Enc::Flat, 0x49}, {"atomic_or", 0, 2}},
      {{Enc::Flat, 0x4a}, {"atomic_xor", 0, 2}},
      {{Enc::Flat, 0x4d}, {"atomic_add_f32", 0, 2}},
      {{Enc::Flat, 0x44}, {"atomic_smin", 0, 2}},
      {{Enc::Flat, 0x45}, {"atomic_umin", 0, 2}},
      {{Enc::Flat, 0x46}, {"atomic_smax", 0, 2}},
      {{Enc::Flat, 0x47}, {"atomic_umax", 0, 2}},
      // Two halves, or two bfloat16s, added into memory at once.
      {{Enc::Flat, 0x4e}, {"atomic_pk_add_f16", 0, 2}},
      {{Enc::Flat, 0x52}, {"atomic_pk_add_bf16", 0, 2}},
      {{Enc::Flat, 0x60}, {"atomic_swap_x2", 0, 2, 1, 2}},
      {{Enc::Flat, 0x63}, {"atomic_sub_x2", 0, 2, 1, 2}},
      {{Enc::Flat, 0x64}, {"atomic_smin_x2", 0, 2, 1, 2}},
      {{Enc::Flat, 0x65}, {"atomic_umin_x2", 0, 2, 1, 2}},
      {{Enc::Flat, 0x66}, {"atomic_smax_x2", 0, 2, 1, 2}},
      {{Enc::Flat, 0x67}, {"atomic_umax_x2", 0, 2, 1, 2}},
      {{Enc::Flat, 0x68}, {"atomic_and_x2", 0, 2, 1, 2}},
      {{Enc::Flat, 0x69}, {"atomic_or_x2", 0, 2, 1, 2}},
      {{Enc::Flat, 0x6a}, {"atomic_xor_x2", 0, 2, 1, 2}},
      {{Enc::Flat, 0x61}, {"atomic_cmpswap_x2", 0, 2, 1, 4}},
      {{Enc::Flat, 0x62}, {"atomic_add_x2", 0, 2, 1, 2}},
      // MUBUF: the buffer instructions. Only the two that act on the caches
      // are decoded -- a write-back of L2 and an invalidate, which a memory
      // fence compiles to -- and they take no operands.
      {{Enc::Mubuf, 0x28}, {"buffer_wbl2", 0, 0}},
      {{Enc::Mubuf, 0x29}, {"buffer_inv", 0, 0}},
      // Through a buffer resource: the data, then the address register, the
      // resource and the scalar offset.
      {{Enc::Mubuf, 0x10}, {"buffer_load_ubyte", 1, 3}},
      {{Enc::Mubuf, 0x11}, {"buffer_load_sbyte", 1, 3}},
      {{Enc::Mubuf, 0x12}, {"buffer_load_ushort", 1, 3}},
      {{Enc::Mubuf, 0x13}, {"buffer_load_sshort", 1, 3}},
      {{Enc::Mubuf, 0x14}, {"buffer_load_dword", 1, 3}},
      {{Enc::Mubuf, 0x15}, {"buffer_load_dwordx2", 2, 3}},
      {{Enc::Mubuf, 0x16}, {"buffer_load_dwordx3", 3, 3}},
      {{Enc::Mubuf, 0x17}, {"buffer_load_dwordx4", 4, 3}},
      {{Enc::Mubuf, 0x18}, {"buffer_store_byte", 0, 4, 1}},
      {{Enc::Mubuf, 0x1a}, {"buffer_store_short", 0, 4, 1}},
      {{Enc::Mubuf, 0x1c}, {"buffer_store_dword", 0, 4, 1}},
      {{Enc::Mubuf, 0x1d}, {"buffer_store_dwordx2", 0, 4, 2}},
      {{Enc::Mubuf, 0x1e}, {"buffer_store_dwordx3", 0, 4, 3}},
      {{Enc::Mubuf, 0x1f}, {"buffer_store_dwordx4", 0, 4, 4}},
      {{Enc::Mubuf, 0x19}, {"buffer_store_byte_d16_hi", 0, 4, 1}},
      {{Enc::Mubuf, 0x1b}, {"buffer_store_short_d16_hi", 0, 4, 1}},
      {{Enc::Mubuf, 0x20}, {"buffer_load_ubyte_d16", 1, 3}},
      {{Enc::Mubuf, 0x21}, {"buffer_load_ubyte_d16_hi", 1, 3}},
      {{Enc::Mubuf, 0x22}, {"buffer_load_sbyte_d16", 1, 3}},
      {{Enc::Mubuf, 0x23}, {"buffer_load_sbyte_d16_hi", 1, 3}},
      {{Enc::Mubuf, 0x24}, {"buffer_load_short_d16", 1, 3}},
      {{Enc::Mubuf, 0x25}, {"buffer_load_short_d16_hi", 1, 3}},
      {{Enc::Mubuf, 0x40}, {"buffer_atomic_swap", 0, 4, 1}},
      {{Enc::Mubuf, 0x41}, {"buffer_atomic_cmpswap", 0, 4, 2}},
      {{Enc::Mubuf, 0x42}, {"buffer_atomic_add", 0, 4, 1}},
      {{Enc::Mubuf, 0x4d}, {"buffer_atomic_add_f32", 0, 4, 1}},
      {{Enc::Mubuf, 0x4e}, {"buffer_atomic_pk_add_f16", 0, 4, 1}},
      {{Enc::Mubuf, 0x61}, {"buffer_atomic_cmpswap_x2", 0, 4, 4}},
      // VOP3P: a packed pair of halves in one register, both computed at once.
      {{Enc::Vop3p, 0x00}, {"v_pk_mad_i16", 1, 3}},
      {{Enc::Vop3p, 0x01}, {"v_pk_mul_lo_u16", 1, 2}},
      {{Enc::Vop3p, 0x02}, {"v_pk_add_i16", 1, 2}},
      {{Enc::Vop3p, 0x03}, {"v_pk_sub_i16", 1, 2}},
      {{Enc::Vop3p, 0x04}, {"v_pk_lshlrev_b16", 1, 2}},
      {{Enc::Vop3p, 0x05}, {"v_pk_lshrrev_b16", 1, 2}},
      {{Enc::Vop3p, 0x06}, {"v_pk_ashrrev_i16", 1, 2}},
      {{Enc::Vop3p, 0x07}, {"v_pk_max_i16", 1, 2}},
      {{Enc::Vop3p, 0x08}, {"v_pk_min_i16", 1, 2}},
      {{Enc::Vop3p, 0x09}, {"v_pk_mad_u16", 1, 3}},
      {{Enc::Vop3p, 0x0a}, {"v_pk_add_u16", 1, 2}},
      {{Enc::Vop3p, 0x0b}, {"v_pk_sub_u16", 1, 2}},
      {{Enc::Vop3p, 0x0c}, {"v_pk_max_u16", 1, 2}},
      {{Enc::Vop3p, 0x0d}, {"v_pk_min_u16", 1, 2}},
      {{Enc::Vop3p, 0x0e}, {"v_pk_fma_f16", 1, 3}},
      {{Enc::Vop3p, 0x0f}, {"v_pk_add_f16", 1, 2}},
      {{Enc::Vop3p, 0x10}, {"v_pk_mul_f16", 1, 2}},
      {{Enc::Vop3p, 0x11}, {"v_pk_min_f16", 1, 2}},
      {{Enc::Vop3p, 0x12}, {"v_pk_max_f16", 1, 2}},
      // And the packed float form, where each of the two is a whole register.
      // A multiply-add over sources that may each be a float or a half,
      // rounded to a half and written into one half of the destination.
      {{Enc::Vop3p, 0x20}, {"v_fma_mix_f32", 1, 3}},
      // Four signed bytes of each source multiplied pairwise and added to the
      // third, clamped where asked.
      {{Enc::Vop3p, 0x28}, {"v_dot4_i32_i8", 1, 3}},
      // Two pairs of halves multiplied and added into a float.
      {{Enc::Vop3p, 0x23}, {"v_dot2_f32_f16", 1, 3}},
      {{Enc::Vop3p, 0x21}, {"v_fma_mixlo_f16", 1, 3}},
      {{Enc::Vop3p, 0x22}, {"v_fma_mixhi_f16", 1, 3}},
      {{Enc::Vop3p, 0x30}, {"v_pk_fma_f32", 2, 3, 2, 2, 2}},
      {{Enc::Vop3p, 0x32}, {"v_pk_add_f32", 2, 2, 2, 2}},
      {{Enc::Vop3p, 0x31}, {"v_pk_mul_f32", 2, 2, 2, 2}},
      // Two registers, each chosen from either source by op_sel.
      {{Enc::Vop3p, 0x33}, {"v_pk_mov_b32", 2, 2, 2, 2}},
      // Between a vector register and an accumulation register, which is
      // where a kernel puts what will not fit in the vector ones. They share
      // the packed forms' encoding without being packed.
      // A matrix multiply-add over the whole wave: a 16x16 by 16x16 product of
      // halves, added into a 16x16 block of floats spread across the lanes.
      {{Enc::Vop3p, 0x4d}, {"v_mfma_f32_16x16x16_f16", 4, 3, 2, 2, 4}},
      // And the others Tensile's GEMMs are built on: floats and doubles, one
      // block, or several smaller ones side by side.
      {{Enc::Vop3p, 0x40}, {"v_mfma_f32_32x32x1_2b_f32", 32, 3, 1, 1, 32}},
      {{Enc::Vop3p, 0x41}, {"v_mfma_f32_16x16x1_4b_f32", 16, 3, 1, 1, 16}},
      {{Enc::Vop3p, 0x42}, {"v_mfma_f32_4x4x1_16b_f32", 4, 3, 1, 1, 4}},
      {{Enc::Vop3p, 0x44}, {"v_mfma_f32_32x32x2_f32", 16, 3, 1, 1, 16}},
      {{Enc::Vop3p, 0x45}, {"v_mfma_f32_16x16x4_f32", 4, 3, 1, 1, 4}},
      {{Enc::Vop3p, 0x6e}, {"v_mfma_f64_16x16x4_f64", 8, 3, 2, 2, 8}},
      // Halves, bfloat16s and bytes into floats and ints: the forms the half,
      // bfloat16 and int8 GEMMs use.
      {{Enc::Vop3p, 0x49}, {"v_mfma_f32_16x16x4_4b_f16", 16, 3, 2, 2, 16}},
      {{Enc::Vop3p, 0x4a}, {"v_mfma_f32_4x4x4_16b_f16", 4, 3, 2, 2, 4}},
      {{Enc::Vop3p, 0x4c}, {"v_mfma_f32_32x32x8_f16", 16, 3, 2, 2, 16}},
      {{Enc::Vop3p, 0x56}, {"v_mfma_i32_32x32x16_i8", 16, 3, 2, 2, 16}},
      {{Enc::Vop3p, 0x57}, {"v_mfma_i32_16x16x32_i8", 4, 3, 2, 2, 4}},
      {{Enc::Vop3p, 0x5e}, {"v_mfma_f32_16x16x4_4b_bf16", 16, 3, 2, 2, 16}},
      {{Enc::Vop3p, 0x5f}, {"v_mfma_f32_4x4x4_16b_bf16", 4, 3, 2, 2, 4}},
      {{Enc::Vop3p, 0x60}, {"v_mfma_f32_32x32x8_bf16", 16, 3, 2, 2, 16}},
      {{Enc::Vop3p, 0x61}, {"v_mfma_f32_16x16x16_bf16", 4, 3, 2, 2, 4}},
      // gfx950's: the same with K doubled, eight halves or bfloat16s a lane.
      {{Enc::Vop3p, 0x54}, {"v_mfma_f32_16x16x32_f16", 4, 3, 4, 4, 4}},
      {{Enc::Vop3p, 0x55}, {"v_mfma_f32_32x32x16_f16", 16, 3, 4, 4, 16}},
      {{Enc::Vop3p, 0x35}, {"v_mfma_f32_16x16x32_bf16", 4, 3, 4, 4, 4}},
      {{Enc::Vop3p, 0x37}, {"v_mfma_f32_32x32x16_bf16", 16, 3, 4, 4, 16}},
      // 8-bit floats, A's and B's each fp8 or bf8, eight to a register pair.
      {{Enc::Vop3p, 0x70}, {"v_mfma_f32_16x16x32_bf8_bf8", 4, 3, 2, 2, 4}},
      {{Enc::Vop3p, 0x71}, {"v_mfma_f32_16x16x32_bf8_fp8", 4, 3, 2, 2, 4}},
      {{Enc::Vop3p, 0x72}, {"v_mfma_f32_16x16x32_fp8_bf8", 4, 3, 2, 2, 4}},
      {{Enc::Vop3p, 0x73}, {"v_mfma_f32_16x16x32_fp8_fp8", 4, 3, 2, 2, 4}},
      {{Enc::Vop3p, 0x74}, {"v_mfma_f32_32x32x16_bf8_bf8", 16, 3, 2, 2, 16}},
      {{Enc::Vop3p, 0x75}, {"v_mfma_f32_32x32x16_bf8_fp8", 16, 3, 2, 2, 16}},
      {{Enc::Vop3p, 0x76}, {"v_mfma_f32_32x32x16_fp8_bf8", 16, 3, 2, 2, 16}},
      {{Enc::Vop3p, 0x77}, {"v_mfma_f32_32x32x16_fp8_fp8", 16, 3, 2, 2, 16}},
      // The reduced-precision float forms decode, and are refused where they
      // run: what a card rounds their inputs to is not modelled.
      {{Enc::Vop3p, 0x3e}, {"v_mfma_f32_16x16x8_xf32", 4, 3, 2, 2, 4}},
      {{Enc::Vop3p, 0x3f}, {"v_mfma_f32_32x32x4_xf32", 16, 3, 2, 2, 16}},
      {{Enc::Vop3p, 0x58}, {"v_accvgpr_read_b32", 1, 1}},
      {{Enc::Vop3p, 0x59}, {"v_accvgpr_write_b32", 1, 1}},
    };
    add_comparisons(m);
    add_long_forms(m);
    return m;
  }();
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
  } else if (code == 107) {
    // VCC's high half, as a register of its own: wave64 code keeps a spare
    // 32-bit value there.
    o.kind = OperandKind::VccHi;
    o.width = 1;
  } else if (code == 124) {
    o.kind = OperandKind::M0;
  } else if (code == 126) {
    o.kind = width >= 2 ? OperandKind::Exec : OperandKind::ExecLo;
    o.width = width >= 2 ? 2 : 1;
  } else if (code == 127) {
    o.kind = OperandKind::ExecHi;
  } else if (code == 235 || code == 237) {
    // Where LDS, or a work-item's private memory, sits in the one address
    // space a flat access uses: the aperture's base as a pair, or its high
    // half as one register.
    o.kind = code == 235 ? OperandKind::SharedBase : OperandKind::PrivateBase;
    o.width = width;
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
  const auto& t = table();
  const auto it = t.find({e, opcode});
  if (it == t.end())
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
  if ((imm & 0xFF00) == 0x8000) {
    std::snprintf(b, sizeof b, "swizzle(QUAD_PERM,%u,%u,%u,%u)", imm & 3, (imm >> 2) & 3, (imm >> 4) & 3,
                  (imm >> 6) & 3);
    return b;
  }
  // The extended modes, each with bit 15 set: a rotation within 32 lanes
  // (the direction, then by how many), and the FFT patterns.
  if ((imm & 0xE000) == 0xC000) {
    std::snprintf(b, sizeof b, "swizzle(ROTATE,%u,%u)", (imm >> 10) & 1, (imm >> 5) & 0x1F);
    return b;
  }
  if (imm & 0x8000) throw Error::make(Err::Unsupported, "a ds_swizzle pattern this does not decode yet (", imm, ")");
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
// names one of the scalars instead. A comparison has no vector destination:
// the bits that would say which part of it to write say instead which scalar
// pair takes the result, when it is not VCC.
void read_sdwa(Inst& in, const Shape& s, uint32_t w0, uint32_t w1, uint32_t srcs) {
  in.sdwa = true;
  if (const size_t at_e32 = in.name.rfind("_e32"); at_e32 != std::string::npos) in.name.resize(at_e32);
  in.name += "_sdwa";
  in.size = 8;
  if (in.enc == Enc::Vopc) {
    if ((w1 >> 15) & 1) in.dst[0] = operand((w1 >> 8) & 0x7F, 2);
  } else {
    in.dst_sel = (w1 >> 8) & 0x7;
    in.dst_unused = (w1 >> 11) & 0x3;
    in.clamp = ((w1 >> 13) & 1) != 0;
    in.omod = static_cast<uint8_t>((w1 >> 14) & 0x3);
    in.dst.push_back(vgpr((w0 >> 17) & 0xFF, s.dst_width));
  }
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
    // s_set_gpr_idx_on's second field is which operands are indexed.
    if (in.name == "s_set_gpr_idx_on") in.simm = static_cast<int32_t>((w0 >> 8) & 0xF);
    else in.src.push_back(take((w0 >> 8) & 0xFF, s.src_width(1)));
  } else if ((w0 >> 28) == 0xb) {     // SOPK
    in.enc = Enc::Sopk;
    in.opcode = (w0 >> 23) & 0x1F;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    if (s.dst_width) in.dst.push_back(sgpr((w0 >> 16) & 0x7F, s.dst_width));
    in.simm = static_cast<int16_t>(w0 & 0xFFFF);
    if (in.name == "s_setreg_imm32_b32") {   // the value comes after, as a literal
      Operand k;
      k.kind = OperandKind::Literal;
      k.value = word(code, at + 4);
      in.src.push_back(k);
      in.size = 8;
    }
    if (in.name == "s_call_b64") in.target = pc + 4 + static_cast<uint64_t>(static_cast<int64_t>(in.simm) * 4);
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
    if (s.dst_width) in.dst.push_back(sgpr((w0 >> 6) & 0x7F, s.dst_width));
    in.cache = (w0 >> 16) & 1;   // glc: an atomic hands back what it found
    if (s.srcs > 0) {
      in.src.push_back(sgpr(((w0 & 0x3F) << 1), s.src_width(0)));   // sbase counts register pairs
      // The offset is the instruction's own (IMM set), a scalar register's
      // (IMM clear: the field names the register), or both (SOE with IMM,
      // the register named above the offset).
      const bool imm = (w0 >> 17) & 1, soe = (w0 >> 14) & 1;
      if (imm) in.offset = static_cast<int32_t>(w1 & 0x1FFFFF);
      in.smem_both_offsets = imm && soe;
      if (!imm || soe) {
        in.has_saddr = true;
        in.saddr = imm ? (w1 >> 25) & 0x7F : w1 & 0x7F;
      }
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
    if (!s.dst_width && !s.srcs) {   // v_nop names nothing
      in.dst.clear();
      in.src.clear();
    }
    // Both of its registers are accumulation registers, in the fields that
    // name vector ones.
    if (in.name == "v_accvgpr_mov_b32") in.dst[0].kind = in.src[0].kind = OperandKind::Agpr;
  } else if ((w0 >> 25) == 0x3e) {    // VOPC
    in.enc = Enc::Vopc;
    in.opcode = (w0 >> 17) & 0xFF;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    Operand vcc;
    vcc.kind = OperandKind::Vcc;
    vcc.width = 2;
    in.dst.push_back(vcc);
    if ((w0 & 0x1FF) == 249) {   // the sub-dword form
      read_sdwa(in, s, w0, word(code, at + 4), 2);
    } else {
      in.src.push_back(take(w0 & 0x1FF, s.src_width(0)));
      in.src.push_back(vgpr((w0 >> 9) & 0xFF, s.src_width(1)));
    }
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
    // op_sel says which half of each source feeds the low result, and
    // op_sel_hi which feeds the high one; neg_lo and neg_hi which sources are
    // negated on the way into each. For the mixed-precision forms, op_sel_hi
    // says which sources are halves at all, op_sel which half, and the two
    // negation fields are each source's negation and absolute value.
    const uint32_t op_sel = (w0 >> 11) & 0x7, neg_hi = (w0 >> 8) & 0x7, neg = (w1 >> 29) & 0x7;
    const uint32_t op_sel_hi = ((w0 >> 14) & 1) << 2 | ((w1 >> 27) & 0x3);
    in.op_sel = static_cast<uint8_t>(op_sel);
    in.op_sel_hi = static_cast<uint8_t>(op_sel_hi);
    const bool mix = in.name.rfind("v_fma_mix", 0) == 0;
    if (!mix) {
      in.neg_lo = static_cast<uint8_t>(neg);
      in.neg_hi = static_cast<uint8_t>(neg_hi);
    }
    in.dst.push_back(vgpr(w0 & 0xFF, s.dst_width));
    for (uint32_t k = 0; k < s.srcs; ++k) {
      Operand o = take((w1 >> (9 * k)) & 0x1FF, s.src_width(k));
      if (mix) {
        o.neg = (neg >> k) & 1;
        o.abs = (neg_hi >> k) & 1;
      }
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
    in.promoted = s.promoted;
    in.size = 8;
    const uint32_t w1 = word(code, at + 4);
    // The modifiers: a source's absolute value and negation, and a clamp of
    // the result. An output multiplier is refused rather than dropped, since
    // dropping one gives a wrong answer with nothing to show for it.
    const uint32_t abs = s.sdst ? 0 : (w0 >> 8) & 0x7;
    in.clamp = s.sdst ? false : ((w0 >> 15) & 1) != 0;
    in.omod = static_cast<uint8_t>((w1 >> 27) & 0x3);
    if (s.scalar_dst) in.dst.push_back(sgpr(w0 & 0xFF, s.dst_width));
    else in.dst.push_back(vgpr(w0 & 0xFF, s.dst_width));
    // VOP3b also writes a scalar pair: a carry out, or the condition
    // v_div_scale reports.
    if (s.sdst) in.dst.push_back(sgpr((w0 >> 8) & 0x7F, 2));
    // op_sel: which half of each 16-bit source is read, and (bit 3) which
    // half of the destination is written. The 8-bit float conversions use
    // the bits to name a half of the destination, or a byte of it. VOP3b
    // keeps its scalar destination in these bits instead.
    if (!s.sdst) in.op_sel = static_cast<uint8_t>((w0 >> 11) & 0xF);
    const uint32_t neg = (w1 >> 29) & 0x7;
    for (uint32_t k = 0; k < s.srcs; ++k) {
      Operand o = take((w1 >> (9 * k)) & 0x1FF, s.src_width(k));
      o.neg = (neg >> k) & 1;
      o.abs = (abs >> k) & 1;
      in.src.push_back(o);
    }
    // v_bitop3's truth table is in those modifier bits instead.
    if (in.name.rfind("v_bitop3_", 0) == 0) {
      in.bitop3 = static_cast<uint8_t>(neg | abs << 3 | in.omod << 6);
      in.omod = 0;
      for (Operand& o : in.src) o.neg = o.abs = false;
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
    // The data, read or written, may be accumulation registers instead.
    const bool acc = (w0 >> 25) & 1;
    const auto data = [&](uint32_t index, uint32_t width) {
      Operand o = vgpr(index, width);
      if (acc) o.kind = OperandKind::Agpr;
      return o;
    };
    if (s.dst_width) in.dst.push_back(data((w1 >> 24) & 0xFF, s.dst_width));
    in.src.push_back(vgpr(w1 & 0xFF));                              // the address
    if (s.srcs > 1) in.src.push_back(data((w1 >> 8) & 0xFF, s.src_width(1)));   // the data written
    if (s.srcs > 2) in.src.push_back(data((w1 >> 16) & 0xFF, s.src_width(2)));  // and the second, for a two-address write
  } else if ((w0 >> 26) == 0x38) {    // MUBUF: through a buffer resource, and the cache operations
    in.enc = Enc::Mubuf;
    in.opcode = (w0 >> 18) & 0x7F;
    const Shape& s = shape(in.enc, in.opcode);
    in.name = s.name;
    in.size = 8;
    in.cache = ((w0 >> 14) & 1) | ((w0 >> 17) & 1) << 1 | ((w0 >> 15) & 1) << 2;   // sc0, nt, sc1
    if (s.srcs) {
      const uint32_t w1 = word(code, at + 4);
      if ((w0 >> 16) & 1)
        throw Error::make(Err::Unsupported, in.name, " loads straight into LDS, which this does not model");
      in.offset = static_cast<int32_t>(w0 & 0xFFF);
      in.offen = (w0 >> 12) & 1;
      in.idxen = (w0 >> 13) & 1;
      // The data: what a load writes, what a store or an atomic reads (and
      // an atomic gives back into, where sc0 asks). Then the address register
      // -- an offset, an index, or the two -- the resource's four scalar
      // registers, and the scalar offset.
      Operand data = vgpr((w1 >> 8) & 0xFF, s.dst_width ? s.dst_width : s.src_width(0));
      if ((w1 >> 23) & 1) data.kind = OperandKind::Agpr;
      if (s.dst_width) in.dst.push_back(data);
      else in.src.push_back(data);
      in.src.push_back(vgpr(w1 & 0xFF, in.offen && in.idxen ? 2 : 1));
      in.src.push_back(sgpr(((w1 >> 16) & 0x1F) << 2, 4));
      in.src.push_back(take((w1 >> 24) & 0xFF, 1));
    }
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
    // 0x7f: the address is the VGPR pair's. A flat access has no scalar base
    // at all, whatever the field holds.
    in.has_saddr = in.segment != Inst::Segment::Flat && saddr != 0x7F;
    in.saddr = saddr;
    // A scratch access may have neither an address register nor a scalar
    // base: then the offset alone says where in the work-item's own memory.
    in.has_vaddr = in.segment != Inst::Segment::Scratch || ((w0 >> 13) & 1) != 0;
    // An atomic gives back the value it replaced where sc0 asks for it, and
    // the register it gives it back in is the one a load would write. A
    // compare-and-swap takes a pair and gives back one of them.
    const bool returns = std::string(s.name).rfind("atomic", 0) == 0 && (in.cache & 1);
    // The data, loaded or stored, may be accumulation registers instead.
    const bool acc = (w1 >> 23) & 1;
    const auto data = [&](uint32_t index, uint32_t width) {
      Operand o = vgpr(index, width);
      if (acc) o.kind = OperandKind::Agpr;
      return o;
    };
    if (s.dst_width) in.dst.push_back(data((w1 >> 24) & 0xFF, s.dst_width));
    else if (returns)
      in.dst.push_back(data((w1 >> 24) & 0xFF,
                            std::string(s.name).find("cmpswap") != std::string::npos ? s.src_width(1) / 2
                                                                                     : s.src_width(1)));
    // A flat address is 64-bit; a global one is 64-bit unless a scalar base
    // carries the top of it; a scratch one is a 32-bit offset.
    const uint32_t addr_width = in.segment == Inst::Segment::Global ? (in.has_saddr ? 1 : 2)
                                                                   : in.segment == Inst::Segment::Flat ? 2 : 1;
    in.src.push_back(vgpr(w1 & 0xFF, addr_width));
    if (s.srcs > 1) in.src.push_back(data((w1 >> 8) & 0xFF, s.src_width(1)));   // the data written
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
    const bool carry_in = in.name == "v_addc_co_u32_e32" || in.name == "v_subb_co_u32_e32" ||
                          in.name == "v_subbrev_co_u32_e32";
    const bool carry_out = carry_in || in.name == "v_add_co_u32_e32" || in.name == "v_sub_co_u32_e32" ||
                           in.name == "v_subrev_co_u32_e32";
    const bool select = in.name == "v_cndmask_b32_e32";
    if ((w0 & 0x1FF) == 249) {   // the sub-dword form, VCC where the short form has it
      read_sdwa(in, s, w0, word(code, at + 4), 2);
      if (carry_out) in.dst.push_back(vcc());
      if (carry_in || select) in.src.push_back(vcc());
      return in;
    }
    in.dst.push_back(vgpr((w0 >> 17) & 0xFF, s.dst_width));
    // The short forms of the carry arithmetic write VCC beside their result,
    // which the encoding does not spell out and the assembler does.
    if (carry_out) in.dst.push_back(vcc());
    if ((w0 & 0x1FF) == 250) {   // the first source comes from another lane
      in.size = 8;
      in.name = in.name.substr(0, in.name.size() - 4) + "_dpp";
      in.src.push_back(read_dpp(in, word(code, at + 4)));
      in.src.push_back(vgpr((w0 >> 9) & 0xFF, s.src_width(1)));
      if (carry_in || select) in.src.push_back(vcc());
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
    case OperandKind::VccHi: return wrap("vcc_hi");
    case OperandKind::M0: return wrap("m0");
    case OperandKind::SharedBase: return wrap("src_shared_base");
    case OperandKind::PrivateBase: return wrap("src_private_base");
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
      // A 64-bit operand zero-extends its literal, so 0xffffffff there is
      // not the -1 an inline constant would sign-extend to.
      const int64_t v = o.width == 2 ? static_cast<int64_t>(static_cast<uint32_t>(o.value))
                                     : static_cast<int32_t>(static_cast<uint32_t>(o.value));
      if (v >= -16 && v <= 64) std::snprintf(b, sizeof b, "%lld", static_cast<long long>(v));
      else std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(static_cast<uint32_t>(o.value)));
      return neg + b;
    }
    case OperandKind::None: break;
  }
  return "?";
}

// A hardware register field as the assembler names it: hwreg(HW_REG_MODE),
// with the offset and the width where they are not the whole register.
std::string hwreg_text(uint32_t simm) {
  static const std::map<uint32_t, const char*> kNames = {
      {1, "HW_REG_MODE"},      {2, "HW_REG_STATUS"},      {3, "HW_REG_TRAPSTS"},     {4, "HW_REG_HW_ID"},
      {5, "HW_REG_GPR_ALLOC"}, {6, "HW_REG_LDS_ALLOC"},   {7, "HW_REG_IB_STS"},      {15, "HW_REG_SH_MEM_BASES"},
      {20, "HW_REG_FLAT_SCR_LO"}, {21, "HW_REG_FLAT_SCR_HI"}, {22, "HW_REG_XNACK_MASK"}};
  const uint32_t id = simm & 0x3F, offset = (simm >> 6) & 0x1F, size = ((simm >> 11) & 0x1F) + 1;
  const auto it = kNames.find(id);
  std::string out = "hwreg(" + (it != kNames.end() ? std::string(it->second) : std::to_string(id));
  if (offset || size != 32) out += ", " + std::to_string(offset) + ", " + std::to_string(size);
  return out + ")";
}

std::string to_text(const Inst& i) {
  std::string s = i.name;
  std::string sep = " ";
  if (i.name == "s_getreg_b32") return s + " " + operand_text(i.dst[0]) + ", " + hwreg_text(static_cast<uint32_t>(i.simm));
  if (i.name == "s_setreg_imm32_b32") {
    const uint32_t v = static_cast<uint32_t>(i.src[0].value);
    char k[32];
    std::snprintf(k, sizeof k, v <= 64 ? "%u" : "0x%x", v);
    return s + " " + hwreg_text(static_cast<uint32_t>(i.simm)) + ", " + k;
  }
  if (i.name == "s_call_b64")
    return s + " " + operand_text(i.dst[0]) + ", " + std::to_string(static_cast<uint32_t>(i.simm) & 0xFFFF);
  if (i.name == "s_set_gpr_idx_on") {
    static const char* kModes[4] = {"SRC0", "SRC1", "SRC2", "DST"};
    std::string modes;
    for (int k = 0; k < 4; ++k)
      if ((i.simm >> k) & 1) modes += (modes.empty() ? "" : ",") + std::string(kModes[k]);
    return s + " " + operand_text(i.src[0]) + ", gpr_idx(" + modes + ")";
  }
  // A literal in a 16-bit float instruction that is one of the float
  // constants is written as that constant, as the assembler reads it back.
  const bool f16 = i.name.find("_f16") != std::string::npos && i.enc != Enc::Vop3p;
  const auto half_constant = [](int64_t v) -> const char* {
    switch (v & 0xFFFF) {
      case 0x3800: return "0.5";
      case 0xB800: return "-0.5";
      case 0x3C00: return "1.0";
      case 0xBC00: return "-1.0";
      case 0x4000: return "2.0";
      case 0xC000: return "-2.0";
      case 0x4400: return "4.0";
      case 0xC400: return "-4.0";
      case 0x3118: return "0.15915494";
      default: return nullptr;
    }
  };
  // A scratch access with no address register: the assembler writes "off"
  // where the register would be.
  const bool scratch_no_addr = i.enc == Enc::Flat && i.segment == Inst::Segment::Scratch && !i.has_vaddr;
  char b[64];
  for (const Operand& o : i.dst) {
    s += sep + operand_text(o);
    sep = ", ";
  }
  for (size_t k = 0; k < i.src.size(); ++k) {
    // A buffer access that takes neither an offset nor an index from a
    // register prints "off" where the register would be.
    const bool no_vaddr = i.enc == Enc::Mubuf && !i.offen && !i.idxen && k == (i.dst.empty() ? 1u : 0u);
    const char* as_half = f16 && i.src[k].kind == OperandKind::Literal && (i.src[k].value & ~0xFFFFll) == 0
                              ? half_constant(i.src[k].value) : nullptr;
    s += sep + ((scratch_no_addr && k == 0) || no_vaddr ? "off" : as_half ? as_half : operand_text(i.src[k]));
    sep = ", ";
  }
  if (i.dpp) {
    char m[64];
    std::snprintf(m, sizeof m, " row_mask:0x%x bank_mask:0x%x", i.row_mask, i.bank_mask);
    s += " " + dpp_control_text(i.dpp_ctrl) + m;
    if (i.bound_ctrl) s += " bound_ctrl:1";
  }
  if (i.bitop3) {
    char t[24];
    std::snprintf(t, sizeof t, " bitop3:0x%x", i.bitop3);
    s += t;
  }
  if (i.enc == Enc::Vop3 && i.op_sel) {
    // A source's bit each, then the destination's (bit 3); the stochastic
    // conversions print bit 2 in a place of its own, though they have two
    // sources.
    const bool byte = i.name.find("_sr_") != std::string::npos;
    std::string out = " op_sel:[";
    for (uint32_t k = 0; k < i.src.size() + (byte ? 1 : 0); ++k) out += std::to_string((i.op_sel >> k) & 1) + ",";
    s += out + std::to_string((i.op_sel >> 3) & 1) + "]";
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
    if (i.neg_lo) s += " neg_lo:" + bits(i.neg_lo);
    if (i.neg_hi) s += " neg_hi:" + bits(i.neg_hi);
  }
  if (i.clamp) s += " clamp";
  if (i.omod) s += i.omod == 1 ? " mul:2" : i.omod == 2 ? " mul:4" : " div:2";
  if (i.sdwa) {
    static const char* kParts[8] = {"BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD", "?"};
    static const char* kUnused[4] = {"UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE", "?"};
    // A comparison has no destination part to name, and neither has an
    // 8-bit float widened: the assembler takes none for those.
    const bool f8 = i.name.find("fp8") != std::string::npos || i.name.find("bf8") != std::string::npos;
    if (i.enc != Enc::Vopc && !f8)
      s += std::string(" dst_sel:") + kParts[i.dst_sel & 7] + " dst_unused:" + kUnused[i.dst_unused & 3];
    for (size_t k = 0; k < i.src.size(); ++k)
      if (i.src[k].kind != OperandKind::Vcc) s += " src" + std::to_string(k) + "_sel:" + kParts[i.src[k].sel & 7];
  }
  if (i.enc == Enc::Smem) {
    if (!i.src.empty() && i.has_saddr) {
      s += ", " + operand_text(operand(i.saddr, 1));
      if (i.offset || i.smem_both_offsets) {
        std::snprintf(b, sizeof b, " offset:0x%x", i.offset);
        s += b;
      }
    } else if (!i.src.empty()) {
      std::snprintf(b, sizeof b, ", 0x%x", i.offset);
      s += b;
    }
    if (i.cache & 1) s += " glc";
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
      // A global access's scalar base is a 64-bit address, a scratch one's a
      // 32-bit offset.
      s += !i.has_saddr ? ", off"
                        : ", " + operand_text(operand(i.saddr, i.segment == Inst::Segment::Scratch ? 1 : 2));
    if (i.offset) {
      std::snprintf(b, sizeof b, " offset:%d", i.offset);
      s += b;
    }
    if (i.cache & 1) s += " sc0";
    if (i.cache & 2) s += " nt";
    if (i.cache & 4) s += " sc1";
  } else if (i.enc == Enc::Mubuf) {
    if (i.idxen) s += " idxen";
    if (i.offen) s += " offen";
    if (i.offset) {
      std::snprintf(b, sizeof b, " offset:%d", i.offset);
      s += b;
    }
    if (i.cache & 1) s += " sc0";
    if (i.cache & 2) s += " nt";
    if (i.cache & 4) s += " sc1";
  } else if (i.name == "s_sendmsg") {
    const uint32_t msg = static_cast<uint32_t>(i.simm) & 0xF;
    if (msg != 1 || (static_cast<uint32_t>(i.simm) & 0xFFF0))
      throw Error::make(Err::Unsupported, "s_sendmsg with message ", static_cast<uint32_t>(i.simm) & 0xFFFF,
                        " is not decoded yet");
    s += " sendmsg(MSG_INTERRUPT)";
  } else if (i.name == "s_nop" || i.name == "s_sleep" || i.name == "s_setprio" || i.name == "s_trap") {
    std::snprintf(b, sizeof b, " %d", i.simm);   // how many cycles to wait
    s += b;
  } else if (i.name == "s_waitcnt") {
    // vmcnt in bits 3:0 and 15:14, expcnt in 6:4, lgkmcnt in 11:8; a counter
    // at its maximum is not waited on, and is not printed.
    const uint32_t imm = static_cast<uint32_t>(i.simm) & 0xFFFF;
    const uint32_t vm = (imm & 0xF) | ((imm >> 14) & 0x3) << 4, exp = (imm >> 4) & 0x7, lgkm = (imm >> 8) & 0xF;
    // Waiting on nothing at all, the assembler prints every counter.
    if (vm == 0x3F && exp == 0x7 && lgkm == 0xF) s += " vmcnt(63) expcnt(7) lgkmcnt(15)";
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
