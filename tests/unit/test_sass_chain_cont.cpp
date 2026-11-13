// The run loop of the SASS tier goes on past an instruction that is not straight only for the scheduler's sake (S2R, ULDC, UMOV, USEL,
// PLOP3, SEL: they write registers and nothing else), and past a BRA that does not split the group,, where it used to end the call and let the chain start the next one at once
// (Runner::tier_run_t in src/sass/tier_ops.inc, "cont"). It may do so only when that chain would have started exactly there: the warp has not
// diverged, nothing waits or has yielded, and the caller's turn has calls left; the call it saved is still counted against the turn. Run
// with the tier and with VGPU_FASTPATH=0 (the interpreter, one call per instruction group), the kernels here must leave the same words,
// the same instruction counts and the same clock readings (SR_CLOCKLO / CS2R count the instructions the warp has run).
//
//   * a diverged warp whose two halves meet at an S2R: the lanes that arrive first must wait one instruction for the others and run it
//     once together (one instruction counted, one clock tick), not run it alone and again for the others;
//   * a branch that the whole group takes, or none of it, and one that a guard lets only some lanes run;
//   * two warps that each load, add and store one word in a loop of S2Rs, 3000 times: where the 4096-call turns end decides which updates
//     are lost, so the final word and the clocks agree only if the saved calls are counted as the interpreter's are.
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

#include "sass/jit.hpp"   // (the statistics of tier 2, for the linking check)

using namespace vgpu;

namespace {

struct Insn {
  sass::Word w;
  std::string text;
};

std::string reg(unsigned r) { return r == 255 ? "RZ" : "R" + std::to_string(r); }
std::string hex(uint32_t v) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%x", v);
  return b;
}
Insn make(uint64_t hi, uint64_t lo, std::string text) {
  Insn i;
  i.w.hi = hi, i.w.lo = lo;
  i.text = std::move(text);
  return i;
}
// lo: opcode | rd << 16 | ra << 24 | rb or immediate << 32; hi's low byte is the third register of the forms that have one.
uint64_t lo_of(uint64_t op, unsigned rd, unsigned ra, uint64_t b) { return op | uint64_t{rd} << 16 | uint64_t{ra} << 24 | b << 32; }

constexpr unsigned kRZ = 255;
// guard: 7 none (PT), 0 @P0, 8 @!P0
Insn iadd3i(unsigned d, unsigned a, uint32_t imm, unsigned c, unsigned guard = 7) {
  return make(0x000fc80007ffe000ull | c, lo_of(0x0810 | (guard == 7 ? 0x7000 : guard << 12), d, a, imm),
              std::string(guard == 7 ? "" : guard == 8 ? "@!P0 " : "@P0 ") + "IADD3 " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}
Insn movi(unsigned d, uint32_t imm) { return make(0x000fe20000000f00ull, lo_of(0x7802, d, 0, imm), "MOV " + reg(d) + ", " + hex(imm)); }
Insn isetp_ge(unsigned a, uint32_t imm) {   // P0 = (Ra >= imm), signed
  return make(0x000fc40003f06270ull, lo_of(0x780c, 0, a, imm), "ISETP.GE.AND P0, PT, " + reg(a) + ", " + hex(imm) + ", PT");
}
Insn s2r_tid(unsigned d) { return make(0x000e220000002100ull, lo_of(0x7919, d, 0, 0), "S2R " + reg(d) + ", SR_TID.X"); }
Insn exit_() { return make(0x000fea0003800000ull, 0x000000000000794dull, "EXIT"); }
// STG.E [Ra.64 + off], Rk
Insn stg(unsigned a, unsigned k, unsigned off) {
  return make(0x000fe2000c101904ull, lo_of(0x7986, 0, a, k) | uint64_t{off} << 40,
              "STG.E [" + reg(a) + ".64" + (off ? "+" + hex(off) : "") + "], " + reg(k));
}
// IMAD.WIDE.U32 Rd, Ra, imm, Rc
Insn imad_wide_i(unsigned d, unsigned a, uint32_t imm, unsigned c) {
  return make(0x000fe200078e0000ull | c, lo_of(0x7825, d, a, imm),
              "IMAD.WIDE.U32 " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm_86 with one kernel taking one 8-byte pointer.
std::string make_cubin(const std::vector<Insn>& code) {
  const std::string names("\0.shstrtab\0.text.k\0.nv.info.k\0", 31);
  std::string info;   // KPARAM_INFO: index, ordinal | offset << 16, size << 18
  put<uint8_t>(info, 4), put<uint8_t>(info, 0x17), put<uint16_t>(info, 12);
  put<uint32_t>(info, 0), put<uint32_t>(info, 0), put<uint32_t>(info, 8u << 18);
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  while (out.size() % 16) out.push_back('\0');
  const uint64_t text_off = out.size();
  for (const Insn& i : code) put(out, i.w.lo), put(out, i.w.hi);
  const uint64_t info_off = out.size();
  out += info;
  while (out.size() % 8) out.push_back('\0');
  const uint64_t shoff = out.size();
  const auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size) {
    put<uint32_t>(out, name), put<uint32_t>(out, type);
    put<uint64_t>(out, 0), put<uint64_t>(out, 0), put<uint64_t>(out, off), put<uint64_t>(out, size);
    put<uint32_t>(out, 0), put<uint32_t>(out, 0), put<uint64_t>(out, 16), put<uint64_t>(out, 0);
  };
  shdr(0, 0, 0, 0);
  shdr(1, 3, names_off, names.size());
  shdr(11, 1, text_off, 16 * code.size());
  shdr(19, 0x70000000, info_off, info.size());
  out[0] = 0x7f, out[1] = 'E', out[2] = 'L', out[3] = 'F', out[4] = 2, out[5] = 1, out[6] = 1, out[7] = 0x41, out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = 4, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (86u << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

// S2R Rd, SR_LANEID
Insn s2r_lane(unsigned d) { return make(0x000e220000000000ull, lo_of(0x7919, d, 0, 0), "S2R " + reg(d) + ", SR_LANEID"); }
// CS2R Rd, SR_CLOCKLO
Insn cs2r_clock(unsigned d) { return make(0x000fca0000015000ull, lo_of(0x7805, d, 0, 0), "CS2R " + reg(d) + ", SR_CLOCKLO"); }
// LDG.E Rd, [Ra.64]
Insn ldg(unsigned d, unsigned a) { return make(0x000ea2000c1e1900ull, lo_of(0x7981, d, a, 4), "LDG.E " + reg(d) + ", [" + reg(a) + ".64]"); }
// @P0 BRA to the instruction at index `to`, from the one at index `from` (the offset is from the next instruction)
Insn bra_p0(int from, int to) {
  const int64_t off = int64_t{16} * (to - from - 1);
  const uint64_t hi = 0x000fc00003800000ull | (off < 0 ? 0x3ffffull : 0ull);
  char buf[32];
  std::snprintf(buf, sizeof buf, "@P0 BRA 0x%llx", static_cast<unsigned long long>(16 * to));
  return make(hi, 0x0947ull | (static_cast<uint64_t>(off) & 0xffffffffull) << 32, buf);
}

// @P0 BRA P1, to the instruction at index `to`: taken by the lanes the guard P0 lets run and P1 holds for
Insn bra_p0_p1(int from, int to) {
  const int64_t off = int64_t{16} * (to - from - 1);
  const uint64_t hi = 0x000fea0000800000ull | (off < 0 ? 0x3ffffull : 0ull);
  char buf[40];
  std::snprintf(buf, sizeof buf, "@P0 BRA P1, 0x%llx", static_cast<unsigned long long>(16 * to));
  return make(hi, 0x0947ull | (static_cast<uint64_t>(off) & 0xffffffffull) << 32, buf);
}

constexpr uint32_t kStride = 0x10;   // bytes a thread's results take

// One branch over two S2Rs: P0 = (tid >= `ge`). With `cond`, the branch also needs P1, which nothing sets, so no lane takes it: the guard lets
// the lanes of tid >= `ge` run it (a branch the group does not split, but one that only some of its lanes ran: counted as divergent).
std::vector<Insn> branch_kernel(unsigned ge, bool cond) {
  std::vector<Insn> c;
  c.push_back(s2r_tid(0));
  c.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  c.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  c.push_back(imad_wide_i(14, 0, kStride, 12));
  c.push_back(isetp_ge(0, ge));
  const int from = static_cast<int>(c.size());
  c.push_back(cond ? bra_p0_p1(from, from + 3) : bra_p0(from, from + 3));
  c.push_back(s2r_lane(1));
  c.push_back(s2r_lane(1));
  c.push_back(s2r_lane(2));
  c.push_back(cs2r_clock(3));
  c.push_back(stg(14, 1, 0));
  c.push_back(stg(14, 2, 4));
  c.push_back(stg(14, 3, 8));
  c.push_back(stg(14, 0, 12));
  c.push_back(exit_());
  return c;
}

// A diverged warp: lanes 0..15 fall through S2Rs, lanes 16..31 jump to the last of them, so the halves meet there.
std::vector<Insn> meeting_kernel(unsigned lead, unsigned tail) {
  std::vector<Insn> c;
  c.push_back(s2r_tid(0));
  c.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  c.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  c.push_back(imad_wide_i(14, 0, kStride, 12));
  c.push_back(isetp_ge(0, 16));
  const int from = static_cast<int>(c.size());
  const int meet = from + 1 + static_cast<int>(lead);
  c.push_back(bra_p0(from, meet));
  for (unsigned i = 0; i < lead; ++i) c.push_back(s2r_lane(1));
  for (unsigned i = 0; i < tail; ++i) c.push_back(s2r_lane(2));   // (the first of these is where the halves meet)
  c.push_back(cs2r_clock(3));
  c.push_back(stg(14, 1, 0));
  c.push_back(stg(14, 2, 4));
  c.push_back(stg(14, 3, 8));
  c.push_back(stg(14, 0, 12));
  c.push_back(exit_());
  return c;
}

// Every warp: `iters` times, load the word, run `s2rs` S2Rs, add one, store it.
std::vector<Insn> loop_kernel(unsigned iters, unsigned s2rs) {
  std::vector<Insn> c;
  c.push_back(s2r_tid(0));
  c.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  c.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  c.push_back(imad_wide_i(14, 0, kStride, 12));
  c.push_back(movi(5, iters));
  const int top = static_cast<int>(c.size());
  c.push_back(ldg(1, 12));
  for (unsigned i = 0; i < s2rs; ++i) c.push_back(s2r_lane(2));
  c.push_back(iadd3i(1, 1, 1, kRZ));
  c.push_back(stg(12, 1, 0));
  Insn dec = iadd3i(5, 5, 0xffffffffu, kRZ);
  dec.text = "IADD3 R5, R5, -0x1, RZ";   // (the decoder prints a negative immediate with its sign)
  c.push_back(dec);
  c.push_back(isetp_ge(5, 1));
  c.push_back(bra_p0(static_cast<int>(c.size()), top));
  c.push_back(cs2r_clock(3));
  c.push_back(stg(14, 3, 4));
  c.push_back(exit_());
  return c;
}

// A warp in two groups that loop apart (round sp52: the run loop's chain between the passes of a diverged warp, which tier 2 with VGPU_JIT_LINK=1 takes over
// inside generated code). Lanes 16..31 jump ahead to their own loop (B) and wait there while lanes 0..15, at the lower address, run theirs (A): every pass of
// A's ends at an S2R or at the branch back, and the chain picks A again because it stands below B; each taken branch back counts towards the spin rule, and
// the 64th lets B in. After its loop A jumps over B's code to the tail (above B: B runs first, and the two meet at the tail), or, with `meet`, to B's first
// instruction (the groups meet there and run loop B together). Both loops update one shared word, so with two warps the place where a 4096-call turn ends
// decides which updates are lost; every thread stores the clock and its lane.
std::vector<Insn> two_loops_kernel(unsigned iters_a, unsigned iters_b, unsigned s2rs, bool meet) {
  std::vector<Insn> c;
  c.push_back(s2r_tid(0));
  c.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  c.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  c.push_back(imad_wide_i(14, 0, kStride, 12));
  c.push_back(s2r_lane(4));
  c.push_back(movi(5, iters_a));
  c.push_back(isetp_ge(4, 16));
  const size_t split = c.size();
  c.push_back(exit_());   // (placeholder: the branch to loop B)
  Insn dec = iadd3i(5, 5, 0xffffffffu, kRZ);
  dec.text = "IADD3 R5, R5, -0x1, RZ";
  const auto loop = [&] {
    const int top = static_cast<int>(c.size());
    c.push_back(ldg(1, 12));
    for (unsigned i = 0; i < s2rs; ++i) c.push_back(s2r_lane(2));
    c.push_back(iadd3i(1, 1, 1, kRZ));
    c.push_back(stg(12, 1, 0));
    c.push_back(dec);
    c.push_back(isetp_ge(5, 1));
    c.push_back(bra_p0(static_cast<int>(c.size()), top));
  };
  loop();   // A
  c.push_back(isetp_ge(4, 0));   // (every lane: the branch below is taken by the whole group)
  const size_t leave_a = c.size();
  c.push_back(exit_());   // (placeholder: A's branch over B's code, or into it)
  const int setup_b = static_cast<int>(c.size());
  c.push_back(movi(5, iters_b));
  loop();   // B
  const int tail = static_cast<int>(c.size());
  c.push_back(cs2r_clock(3));
  c.push_back(stg(14, 3, 4));
  c.push_back(stg(14, 2, 8));
  c.push_back(exit_());
  c[split] = bra_p0(static_cast<int>(split), setup_b);
  c[leave_a] = bra_p0(static_cast<int>(leave_a), meet ? setup_b : tail);
  return c;
}

// BSSY Bb, <the instruction at index `to`> and BSYNC Bb (no text: the test checks what they decode to).
Insn bssy(unsigned b, int from, int to) {
  const uint64_t off = static_cast<uint64_t>(int64_t{16} * (to - from - 1)) / 4;
  return make(0x000fe20003800000ull, 0x7945ull | uint64_t{b} << 16 | (off & 0x3fffffffull) << 34, "");
}
Insn bsync(unsigned b) { return make(0x000fea0003800000ull, 0x7941ull | uint64_t{b} << 16, ""); }

// The same two loops inside a BSSY / BSYNC pair, as a compiler lays out an if and its else: every lane joins barrier B0 before the warp splits, loop A
// (lanes 0..15) falls into the join, a BSYNC B0, and loop B (lanes 16..31, at the higher addresses) branches back to it. The group that gets there first
// waits at the BSYNC for the other: it may not run the tail alone, which it would if the BSSY had not entered the lanes in the barrier (with linking the BSSY
// is an op of a generated region, and entering them is the host's call).
std::vector<Insn> joined_loops_kernel(unsigned iters_a, unsigned iters_b, unsigned s2rs) {
  std::vector<Insn> c;
  c.push_back(s2r_tid(0));
  c.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  c.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  c.push_back(imad_wide_i(14, 0, kStride, 12));
  c.push_back(s2r_lane(4));
  c.push_back(movi(5, iters_a));
  const size_t enter = c.size();
  c.push_back(exit_());   // (placeholder: the BSSY)
  c.push_back(isetp_ge(4, 16));
  const size_t split = c.size();
  c.push_back(exit_());   // (placeholder: the branch to loop B)
  Insn dec = iadd3i(5, 5, 0xffffffffu, kRZ);
  dec.text = "IADD3 R5, R5, -0x1, RZ";
  const auto loop = [&] {
    const int top = static_cast<int>(c.size());
    c.push_back(ldg(1, 12));
    for (unsigned i = 0; i < s2rs; ++i) c.push_back(s2r_lane(2));
    c.push_back(iadd3i(1, 1, 1, kRZ));
    c.push_back(stg(12, 1, 0));
    c.push_back(dec);
    c.push_back(isetp_ge(5, 1));
    c.push_back(bra_p0(static_cast<int>(c.size()), top));
  };
  loop();   // A
  const int join = static_cast<int>(c.size());
  c.push_back(bsync(0));
  c.push_back(cs2r_clock(3));
  c.push_back(stg(14, 3, 4));
  c.push_back(stg(14, 2, 8));
  c.push_back(exit_());
  const int setup_b = static_cast<int>(c.size());
  c.push_back(movi(5, iters_b));
  loop();   // B
  c.push_back(isetp_ge(4, 0));   // (every lane: the branch back to the join is taken by the whole group)
  c.push_back(bra_p0(static_cast<int>(c.size()), join));
  c.push_back(exit_());   // (never reached: a section's last instruction is not a straight one)
  c[enter] = bssy(0, static_cast<int>(enter), join);
  c[split] = bra_p0(static_cast<int>(split), setup_b);
  return c;
}

// An if and its else inside a loop (round sp53; the BSYNC is the run loop's op, where generated code stops, with linking too). Every iteration the warp joins barrier B0, the
// lanes of lane id >= `ge` branch to the else part (above the tail), the others fall through the then part into the join, a BSYNC B0, and wait there; the
// chain then runs the else group, which branches back to the join, completes the barrier and releases everyone: the joined warp counts the iteration and
// branches back. So a diverged warp runs the BSYNC twice an iteration: once to wait (the next group is another one, standing elsewhere) and once to release
// (the next group is the joined lanes at the next op). With `ge` 0 or above 31 the warp never splits and the BSYNC releases at once. Both parts update one
// shared word, so with two warps the place where a 4096-call turn ends decides which updates are lost: the BSYNC's call must be charged as the chain's.
// With `side` below 32 the lanes of lane id >= `side` never enter the loop: they branch to a loop of their own above everything else (`side_iters` times
// the same update) before the first BSSY, so they are no lanes of the barrier, and while the others go round the diamond they are runnable lanes that stand
// elsewhere: after every BSYNC the group that goes on has other lanes above it (the link bit, their lowest place), each of its branches back counts
// towards the spin rule, and the 64th lets the side loop in.
std::vector<Insn> diamond_loop_kernel(unsigned iters, unsigned ge, unsigned s2rs, unsigned side = 99, unsigned side_iters = 0) {
  std::vector<Insn> c;
  c.push_back(s2r_tid(0));
  c.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  c.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  c.push_back(imad_wide_i(14, 0, kStride, 12));
  c.push_back(s2r_lane(4));
  c.push_back(movi(5, iters));
  c.push_back(isetp_ge(4, side));
  const size_t aside = c.size();
  c.push_back(exit_());   // (placeholder: the branch to the side loop)
  const int top = static_cast<int>(c.size());
  const size_t enter = c.size();
  c.push_back(exit_());   // (placeholder: the BSSY)
  c.push_back(isetp_ge(4, ge));
  const size_t split = c.size();
  c.push_back(exit_());   // (placeholder: the branch to the else part)
  const auto part = [&](uint32_t add) {
    c.push_back(ldg(1, 12));
    for (unsigned i = 0; i < s2rs; ++i) c.push_back(s2r_lane(2));
    c.push_back(iadd3i(1, 1, add, kRZ));
    c.push_back(stg(12, 1, 0));
  };
  part(1);   // then
  const int join = static_cast<int>(c.size());
  c.push_back(bsync(0));
  Insn dec = iadd3i(5, 5, 0xffffffffu, kRZ);
  dec.text = "IADD3 R5, R5, -0x1, RZ";
  c.push_back(dec);
  c.push_back(isetp_ge(5, 1));
  c.push_back(bra_p0(static_cast<int>(c.size()), top));
  const int tail = static_cast<int>(c.size());
  c.push_back(cs2r_clock(3));
  c.push_back(stg(14, 3, 4));
  c.push_back(stg(14, 2, 8));
  c.push_back(exit_());
  const int else_ = static_cast<int>(c.size());
  part(2);   // else
  c.push_back(isetp_ge(4, 0));   // (every lane: the branch back to the join is taken by the whole group)
  c.push_back(bra_p0(static_cast<int>(c.size()), join));
  const int side_setup = static_cast<int>(c.size());
  c.push_back(movi(5, side_iters));
  {
    const int side_top = static_cast<int>(c.size());
    part(3);
    c.push_back(dec);
    c.push_back(isetp_ge(5, 1));
    c.push_back(bra_p0(static_cast<int>(c.size()), side_top));
  }
  c.push_back(isetp_ge(4, 0));
  c.push_back(bra_p0(static_cast<int>(c.size()), tail));
  c.push_back(exit_());   // (never reached: a section's last instruction is not a straight one)
  c[aside] = bra_p0(static_cast<int>(aside), side_setup);
  c[enter] = bssy(0, static_cast<int>(enter), join);
  c[split] = bra_p0(static_cast<int>(split), else_);
  return c;
}

// CALL.REL.NOINC <the instruction at index `to`> and RET.REL.NODEC Rr (no text: the test checks what they decode to).
Insn call_rel(int from, int to) {
  const int64_t off = int64_t{16} * (to - from - 1);
  return make(0x000fc00003800000ull | (off < 0 ? 0x3ffffull : 0ull), 0x7944ull | (static_cast<uint64_t>(off) & 0xffffffffull) << 32, "");
}
Insn ret_rel(unsigned r) { return make(0x000fc00003800000ull, 0x7950ull | uint64_t{r} << 24, ""); }

// A loop that calls a function (round sp52: CALL.REL and RET.REL, which tier 2 with linking runs inside generated code). The function loads the shared
// word, runs S2Rs, adds one and stores it; it returns through R8:R9, which the caller sets before each call as a compiler does. Lanes 16..31 make `more`
// more iterations, so the branch back splits the warp once: the half that goes on loops, calls and returns while the other half waits at the tail.
// With `fn_low` the function lies below the loop (the calls and returns of the half that goes on stay below the waiting half: the chain picks it again);
// without, it lies above the tail (the chain picks the waiting half first, at every call).
std::vector<Insn> call_kernel(unsigned iters, unsigned more, unsigned s2rs, bool fn_low) {
  std::vector<Insn> c;
  c.push_back(s2r_tid(0));
  c.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  c.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  c.push_back(imad_wide_i(14, 0, kStride, 12));
  c.push_back(s2r_lane(4));
  c.push_back(movi(5, iters));
  c.push_back(movi(9, 0));
  c.push_back(isetp_ge(4, 16));
  c.push_back(iadd3i(5, 5, more, kRZ, 0));   // @P0: lanes 16..31
  int fn = -1;
  const auto function = [&] {
    fn = static_cast<int>(c.size());
    c.push_back(ldg(1, 12));
    for (unsigned i = 0; i < s2rs; ++i) c.push_back(s2r_lane(2));
    c.push_back(iadd3i(1, 1, 1, kRZ));
    c.push_back(stg(12, 1, 0));
    c.push_back(ret_rel(8));
  };
  size_t skip = 0;
  if (fn_low) {
    c.push_back(isetp_ge(4, 0));   // (every lane: the branch over the function is taken by the whole warp)
    skip = c.size();
    c.push_back(exit_());   // (placeholder)
    function();
    c[skip] = bra_p0(static_cast<int>(skip), static_cast<int>(c.size()));
  }
  const int top = static_cast<int>(c.size());
  const size_t ra = c.size();
  c.push_back(exit_());   // (placeholder: MOV R8, <the address after the call>)
  const size_t call = c.size();
  c.push_back(exit_());   // (placeholder: the call)
  c[ra] = movi(8, 16u * static_cast<uint32_t>(c.size()));
  Insn dec = iadd3i(5, 5, 0xffffffffu, kRZ);
  dec.text = "IADD3 R5, R5, -0x1, RZ";
  c.push_back(dec);
  c.push_back(isetp_ge(5, 1));
  c.push_back(bra_p0(static_cast<int>(c.size()), top));
  c.push_back(cs2r_clock(3));
  c.push_back(stg(14, 3, 4));
  c.push_back(stg(14, 2, 8));
  c.push_back(exit_());
  if (!fn_low) function();
  c.push_back(exit_());   // (never reached: a section's last instruction is not a straight one)
  c[call] = call_rel(static_cast<int>(call), fn);
  return c;
}

struct Result {
  std::vector<uint32_t> out;
  exec::LaunchStats stats;
};

struct Env {
  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu("nvidia/a10");   // sm_86
  std::shared_ptr<sass::Module> load(const std::vector<Insn>& code) {
    for (size_t i = 0; i < code.size(); ++i) {
      const sass::Instr ins = sass::decode(code[i].w, 16 * i, 86);
      if (!code[i].text.empty() && sass::to_text(ins) != code[i].text)
        throw vtest::Failure("word " + std::to_string(i) + " is \"" + sass::to_text(ins) + "\", not \"" + code[i].text + "\"");
    }
    const std::string c = make_cubin(code);
    return sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), mem, prof);
  }
  Result run(sass::Module& m, uint32_t threads) {
    const uint32_t words = threads * kStride / 4 + 4;   // (the shared word first: a thread's results start at byte 0 of its own stride)
    const uint64_t out = mem.alloc(words * 4);
    const std::vector<uint32_t> zero(words, 0);
    mem.write(out, zero.data(), words * 4);
    exec::LaunchConfig cfg;
    cfg.grid = {1, 1, 1};
    cfg.block = {threads, 1, 1};
    std::vector<uint8_t> arg(8);
    std::memcpy(arg.data(), &out, 8);
    Result r;
    r.stats = sass::launch(m, "k", cfg, {arg}, mem, prof);
    r.out.resize(words);
    mem.read(out, r.out.data(), words * 4);
    mem.free(out);
    return r;
  }
};

struct Scoped {
  Scoped() { ::setenv("VGPU_THREADS", "1", 1); }
  ~Scoped() {
    ::unsetenv("VGPU_THREADS");
    ::unsetenv("VGPU_FASTPATH");
  }
};

void expect_same(const Result& tier, const Result& ref, const std::string& what) {
  if (tier.out != ref.out) {
    size_t i = 0;
    while (i < tier.out.size() && tier.out[i] == ref.out[i]) ++i;
    throw vtest::Failure(what + ": word " + std::to_string(i) + " is " + std::to_string(tier.out[i]) + " with the tier and " + std::to_string(ref.out[i]) + " without");
  }
  if (std::memcmp(&tier.stats, &ref.stats, exec::kCounterWords * sizeof(uint64_t)) != 0) {
    const uint64_t* a = reinterpret_cast<const uint64_t*>(&tier.stats);
    const uint64_t* b = reinterpret_cast<const uint64_t*>(&ref.stats);
    size_t i = 0;
    while (a[i] == b[i]) ++i;
    throw vtest::Failure(what + ": counter word " + std::to_string(i) + " is " + std::to_string(a[i]) + " with the tier and " + std::to_string(b[i]) + " without");
  }
}

Result compare(const std::vector<Insn>& code, uint32_t threads, const std::string& name) {
  Scoped scoped;
  Env e;
  const auto m = e.load(code);
  const Result tier = e.run(*m, threads);
  ::setenv("VGPU_FASTPATH", "0", 1);
  const Result ref = e.run(*m, threads);
  ::unsetenv("VGPU_FASTPATH");
  expect_same(tier, ref, name + ", " + std::to_string(threads) + " threads");
  return tier;
}

}  // namespace

VTEST(halves_of_a_diverged_warp_meet_at_an_s2r_and_run_it_once_together) {
  for (unsigned lead : {1u, 2u, 5u})
    for (unsigned tail : {1u, 3u}) compare(meeting_kernel(lead, tail), 32, "lead " + std::to_string(lead) + ", tail " + std::to_string(tail));
  compare(meeting_kernel(3, 2), 64, "two warps");
  compare(meeting_kernel(3, 2), 20, "a part of a warp");
}

VTEST(the_calls_saved_by_going_on_past_an_s2r_count_against_the_warps_turn) {
  // Two warps update one word 3000 times each, with S2Rs in the loop: which updates are lost depends on where the 4096-call turns end.
  for (unsigned s2rs : {1u, 2u, 3u, 5u}) compare(loop_kernel(3000, s2rs), 64, std::to_string(s2rs) + " S2Rs per iteration");
  compare(loop_kernel(1500, 2), 128, "four warps");
}

VTEST(a_branch_the_whole_group_takes_or_none_of_it_lands_where_the_chain_would_start) {
  compare(branch_kernel(0, false), 32, "taken by every lane");      // (P0 holds for all: a forward jump over two S2Rs)
  compare(branch_kernel(99, false), 32, "taken by none");
  compare(branch_kernel(16, false), 32, "taken by half");           // (diverges: the general path)
  compare(branch_kernel(16, true), 32, "guard on half, condition false");
  compare(branch_kernel(0, true), 32, "guard on all, condition false");
  compare(branch_kernel(0, false), 40, "a warp and a part of one");
}

VTEST(the_groups_of_a_diverged_warp_loop_apart_and_meet_where_the_chain_picks_them) {
  for (const bool meet : {false, true}) {
    const std::string m = meet ? ", the groups meet at loop B" : ", the groups meet at the tail";
    for (unsigned s2rs : {1u, 3u}) {
      compare(two_loops_kernel(5, 7, s2rs, meet), 32, "5 and 7 iterations, " + std::to_string(s2rs) + " S2Rs" + m);
      compare(two_loops_kernel(150, 3, s2rs, meet), 32, "150 and 3 iterations (the spin rule lets the other group in), " + std::to_string(s2rs) + " S2Rs" + m);
      compare(two_loops_kernel(3, 150, s2rs, meet), 32, "3 and 150 iterations, " + std::to_string(s2rs) + " S2Rs" + m);
    }
    compare(two_loops_kernel(700, 900, 2, meet), 64, "two warps, turns end inside the loops" + m);
    compare(two_loops_kernel(40, 90, 1, meet), 40, "a warp and a part of one" + m);
    compare(two_loops_kernel(63, 64, 1, meet), 32, "63 and 64 iterations" + m);
    compare(two_loops_kernel(64, 65, 1, meet), 32, "64 and 65 iterations" + m);
  }
  // The same loops between a BSSY and the BSYNC both groups come to.
  {
    const std::vector<Insn> k = joined_loops_kernel(5, 7, 1);
    unsigned bssys = 0, bsyncs = 0;
    for (size_t i = 0; i < k.size(); ++i) {
      const sass::Instr ins = sass::decode(k[i].w, 16 * i, 86);
      bssys += ins.op == sass::Op::BSSY;
      bsyncs += ins.op == sass::Op::BSYNC;
    }
    VCHECK_EQ(bssys, 1u);
    VCHECK_EQ(bsyncs, 1u);
  }
  for (unsigned s2rs : {1u, 3u}) {
    compare(joined_loops_kernel(5, 7, s2rs), 32, "joined, 5 and 7 iterations, " + std::to_string(s2rs) + " S2Rs");
    compare(joined_loops_kernel(150, 3, s2rs), 32, "joined, 150 and 3 iterations, " + std::to_string(s2rs) + " S2Rs");
    compare(joined_loops_kernel(3, 150, s2rs), 32, "joined, 3 and 150 iterations, " + std::to_string(s2rs) + " S2Rs");
  }
  compare(joined_loops_kernel(700, 900, 2), 64, "joined, two warps, turns end inside the loops");
  compare(joined_loops_kernel(40, 90, 1), 40, "joined, a warp and a part of one");
  // A loop that calls a function and returns from it, the warp split once by the loop's branch back.
  for (const bool fn_low : {false, true}) {
    const std::vector<Insn> k = call_kernel(5, 7, 1, fn_low);
    unsigned calls = 0, rets = 0;
    for (size_t i = 0; i < k.size(); ++i) {
      const sass::Instr ins = sass::decode(k[i].w, 16 * i, 86);
      calls += ins.op == sass::Op::CALL;
      rets += ins.op == sass::Op::RET;
    }
    VCHECK_EQ(calls, 1u);
    VCHECK_EQ(rets, 1u);
    const std::string m = fn_low ? ", the function below the loop" : ", the function above the tail";
    for (unsigned s2rs : {1u, 3u}) {
      compare(call_kernel(5, 7, s2rs, fn_low), 32, "calls, 5 and 12 iterations, " + std::to_string(s2rs) + " S2Rs" + m);
      compare(call_kernel(100, 150, s2rs, fn_low), 32, "calls, 100 and 250 iterations, " + std::to_string(s2rs) + " S2Rs" + m);
    }
    compare(call_kernel(900, 300, 2, fn_low), 64, "calls, two warps, turns end inside the loops" + m);
    compare(call_kernel(40, 0, 1, fn_low), 32, "calls, the warp never splits" + m);
    compare(call_kernel(40, 30, 1, fn_low), 40, "calls, a warp and a part of one" + m);
  }
  // An if and its else in a loop: the BSYNC waits and releases every iteration (round sp53).
  for (unsigned s2rs : {1u, 2u}) {
    compare(diamond_loop_kernel(40, 16, s2rs), 32, "diamond, the warp in halves, " + std::to_string(s2rs) + " S2Rs");
    compare(diamond_loop_kernel(40, 0, s2rs), 32, "diamond, every lane in the else part, " + std::to_string(s2rs) + " S2Rs");
    compare(diamond_loop_kernel(40, 99, s2rs), 32, "diamond, every lane in the then part, " + std::to_string(s2rs) + " S2Rs");
    compare(diamond_loop_kernel(40, 31, s2rs), 32, "diamond, one lane in the else part, " + std::to_string(s2rs) + " S2Rs");
  }
  compare(diamond_loop_kernel(900, 16, 2), 64, "diamond, two warps, turns end inside the loop");
  compare(diamond_loop_kernel(900, 5, 1), 128, "diamond, four warps");
  compare(diamond_loop_kernel(60, 16, 1), 40, "diamond, a warp and a part of one");
  compare(diamond_loop_kernel(60, 4, 1), 20, "diamond, a part of a warp");
  // ... with a third group in a loop of its own above the diamond: other lanes can run after every BSYNC.
  for (unsigned s2rs : {1u, 2u}) {
    compare(diamond_loop_kernel(40, 8, s2rs, 24, 30), 32, "diamond beside a side loop, " + std::to_string(s2rs) + " S2Rs");
    compare(diamond_loop_kernel(200, 8, s2rs, 24, 150), 32, "diamond beside a side loop, the spin rule lets it in, " + std::to_string(s2rs) + " S2Rs");
    compare(diamond_loop_kernel(200, 0, s2rs, 16, 150), 32, "diamond never split, beside a side loop, " + std::to_string(s2rs) + " S2Rs");
  }
  compare(diamond_loop_kernel(700, 8, 2, 24, 500), 64, "diamond beside a side loop, two warps");
  compare(diamond_loop_kernel(90, 8, 1, 24, 60), 40, "diamond beside a side loop, a warp and a part of one");
  // With tier 2 and linking asked for (ctest test_sass_chain_cont_jit_link), the generated code must have taken the chain's part: a run in which nothing was
  // linked would compare equal too. (Counted only with VGPU_JIT_STATS; no compiler on PATH: nothing ran as generated code.)
  if (sass::jit::requested() && sass::jit::link_requested() && sass::jit::stats_on() && sass::jit::stats().regions_run.load() > 0) {
    std::printf("  linked inside generated code: %llu branches and register-only ops\n", static_cast<unsigned long long>(sass::jit::stats().linked.load()));
    VCHECK(sass::jit::stats().linked.load() > 1000);
    // The loop rule (round sp53): a section with a loop and no BSSY, BSYNC, CALL or RET is generated without linking. The kernels above that loop without a
    // barrier are such sections (two_loops_kernel, which diverges by a plain branch as no compiler's code does, among them: the legs that check linking in
    // it run with VGPU_JIT_LINK_RULE=0); the joined, the calling and the diamond kernels are not, and must still have been linked (the checks around this one).
    {
      const char* rule = std::getenv("VGPU_JIT_LINK_RULE");
      const unsigned long long declined = sass::jit::stats().link_declined.load();
      std::printf("  sections left unlinked by the loop rule: %llu\n", declined);
      if (rule && rule[0] == '0') VCHECK_EQ(declined, 0ull);
      else VCHECK(declined >= 10);
    }
  }
}

VTEST(one_warp_alone_gives_the_same_words_and_counters) { compare(loop_kernel(2000, 3), 32, "one warp"); }

// Round sp57 (ctest test_sass_chain_cont_jit_wait: a compile threshold that is far, a short profile). A section that asks the cache and gets nothing goes to
// the plain loop (jit::State::Waiting) and is counted there by what each call of the loop ran; short of the threshold it comes back to the hook loop and is
// compiled. The words and counters of kernels that cross all four stretches are the interpreter's (compare), and since a run in which no section waited, or
// none came back, would compare equal too, the counts are checked. The iteration counts differ from run to run so that the cache never has these sections.
VTEST(a_section_waits_for_a_far_threshold_on_the_plain_loop_and_is_compiled_after_it) {
  if (!std::getenv("VGPU_JIT_EXPECT_WAIT") || !sass::jit::requested()) return;
  const auto& s = sass::jit::stats();
  const unsigned long long w0 = s.waits.load(), e0 = s.wait_ends.load(), r0 = s.units_ready.load(), f0 = s.failures.load();
  const unsigned salt = static_cast<unsigned>(::getpid()) % 1000u;
  compare(loop_kernel(20000 + salt, 3), 32, "one warp, the threshold far");
  compare(loop_kernel(20000 + salt, 1), 64, "two warps, the threshold far");
  compare(diamond_loop_kernel(9000 + salt, 16, 2), 64, "diamond, two warps, the threshold far");
  if (s.failures.load() != f0) return;   // (no compiler on PATH: the lookup failed and the tier ran it all)
  std::printf("  sections that waited on the plain loop: %llu, came back to count: %llu, compiled after it: %llu\n",
              static_cast<unsigned long long>(s.waits.load()) - w0, static_cast<unsigned long long>(s.wait_ends.load()) - e0,
              static_cast<unsigned long long>(s.units_ready.load()) - r0);
  VCHECK(s.waits.load() >= w0 + 3);
  VCHECK(s.wait_ends.load() >= e0 + 3);
  VCHECK(s.units_ready.load() >= r0 + 3);
}

// The wait before the lookup (round sp63): with the share rule on, a kept section waits on the plain loop for its lookup too and is hooked again when all of
// its instructions make up the rest of VGPU_JIT_LOOKUP_AFTER; then the lookup misses, it waits for the threshold, is hooked again and compiled. Five
// stretches inside one kernel, the interpreter's words and counters (compare), and the counts that say it happened (ctest test_sass_chain_cont_jit_prewait).
VTEST(a_kept_section_waits_for_its_lookup_and_its_threshold_on_the_plain_loop_and_is_compiled_after_them) {
  if (!std::getenv("VGPU_JIT_EXPECT_PREWAIT") || !sass::jit::requested()) return;
  const auto& s = sass::jit::stats();
  const unsigned long long p0 = s.prewaits.load(), w0 = s.waits.load(), e0 = s.wait_ends.load(), r0 = s.units_ready.load(), f0 = s.failures.load(),
                           a0 = s.accepted.load();
  const unsigned salt = static_cast<unsigned>(::getpid()) % 1000u + 7u;
  compare(loop_kernel(20000 + salt, 3), 32, "one warp, a wait for the lookup and one for the threshold");
  compare(loop_kernel(20000 + salt, 1), 64, "two warps, likewise");
  compare(diamond_loop_kernel(9000 + salt, 16, 2), 64, "diamond, two warps, likewise");
  if (s.failures.load() != f0) return;   // (no compiler on PATH)
  std::printf("  kept by the share rule: %llu, waited for the lookup: %llu, for the threshold: %llu, waits ended: %llu, compiled: %llu\n",
              static_cast<unsigned long long>(s.accepted.load()) - a0, static_cast<unsigned long long>(s.prewaits.load()) - p0,
              static_cast<unsigned long long>(s.waits.load()) - w0, static_cast<unsigned long long>(s.wait_ends.load()) - e0,
              static_cast<unsigned long long>(s.units_ready.load()) - r0);
  VCHECK(s.prewaits.load() >= p0 + 3);
  VCHECK(s.waits.load() >= w0 + 3);
  VCHECK(s.wait_ends.load() >= e0 + 6);
  VCHECK(s.units_ready.load() >= r0 + 3);
}

VTEST_MAIN
