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
#include <cstdint>
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

VTEST(one_warp_alone_gives_the_same_words_and_counters) { compare(loop_kernel(2000, 3), 32, "one warp"); }

VTEST_MAIN
