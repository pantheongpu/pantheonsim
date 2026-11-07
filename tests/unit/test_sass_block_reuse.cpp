// The SASS executor keeps a thread's Block (the warps' register files, the shared and local memory, the tensor memory) from block to
// block and from launch to launch, and sets it up again for each block (Runner::init_block). Every block must find what a new one
// does: a warp that is neither finished nor waiting, and registers that read zero.
//
// The kernels are SASS written out word by word (sm_86; nvdisasm's text for each word is checked, so a mistaken encoding fails here
// and not as a strange result): each stores one register to out[blockIdx.x] from every thread. One stores a register it never wrote,
// one a register it set to a number first; run one after the other on this thread (VGPU_THREADS=1), the second gets the first's block.
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/tcgen05.hpp"
#include "vgpu/exec/worker_pool.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

struct Insn {
  sass::Word w;
  const char* text;
  Insn(uint64_t hi, uint64_t lo, const char* t) : text(t) { w.hi = hi, w.lo = lo; }   // (the corpora print the high half first)
};

// S2R R0, SR_CTAID.X; MOV R6/R7, the out pointer (bank 0, 0x160); IMAD.WIDE.U32 R10, R0, 0x4, R6 (the pair R10:R11 is the address, R9 stays free).
const Insn kCtaid = {0x000ea20000002500ull, 0x0000000000007919ull, "S2R R0, SR_CTAID.X"};
const Insn kPtrLo = {0x000fc60000000f00ull, 0x0000580000067a02ull, "MOV R6, c[0x0][0x160]"};
const Insn kPtrHi = {0x000fc60000000f00ull, 0x0000590000077a02ull, "MOV R7, c[0x0][0x164]"};
const Insn kAddr = {0x000fe200078e0006ull, 0x00000004000a7825ull, "IMAD.WIDE.U32 R10, R0, 0x4, R6"};
const Insn kSetR9 = {0x000fc80000000f00ull, 0x1234567800097802ull, "MOV R9, 0x12345678"};
const Insn kStoreR9 = {0x000fe2000c101904ull, 0x000000090a007986ull, "STG.E [R10.64], R9"};
const Insn kStoreR0 = {0x000fe2000c101904ull, 0x000000000a007986ull, "STG.E [R10.64], R0"};
// A uniform register: UMOV UR62, 0x1234 (the highest a program can write below sm_100: UR63 is URZ), and MOV R9, UR62.
const Insn kSetUr62 = {0x000fe20000000000ull, 0x00001234003e7882ull, "UMOV UR62, 0x1234"};
const Insn kSetUr0 = {0x000fe20000000000ull, 0x0000abcd00007882ull, "UMOV UR0, 0xabcd"};
const Insn kR9FromUr62 = {0x000fc80008000f00ull, 0x0000003e00097c02ull, "MOV R9, UR62"};
const Insn kR9FromUr0 = {0x000fc80008000f00ull, 0x0000000000097c02ull, "MOV R9, UR0"};
const Insn kExit = {0x000fea0003800000ull, 0x000000000000794dull, "EXIT"};

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

struct Kernel {
  std::string name;
  std::vector<Insn> code;
};

// A cubin for sm_86 (or `sm`) with the kernels given, each taking one 8-byte pointer.
std::string make_cubin(const std::vector<Kernel>& kernels, int sm = 86) {
  std::string names("\0.shstrtab\0", 11);
  std::vector<uint32_t> text_name, info_name;
  for (const Kernel& k : kernels) {
    text_name.push_back(static_cast<uint32_t>(names.size()));
    names += ".text." + k.name + '\0';
    info_name.push_back(static_cast<uint32_t>(names.size()));
    names += ".nv.info." + k.name + '\0';
  }
  std::string info;   // KPARAM_INFO: index, ordinal | offset << 16, size << 18
  put<uint8_t>(info, 4), put<uint8_t>(info, 0x17), put<uint16_t>(info, 12);
  put<uint32_t>(info, 0), put<uint32_t>(info, 0), put<uint32_t>(info, 8u << 18);
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  struct Place {
    uint64_t text_off, text_size, info_off;
  };
  std::vector<Place> places;
  for (const Kernel& k : kernels) {
    while (out.size() % 16) out.push_back('\0');
    Place p{out.size(), 16 * k.code.size(), 0};
    for (const Insn& i : k.code) put(out, i.w.lo), put(out, i.w.hi);
    p.info_off = out.size();
    out += info;
    places.push_back(p);
  }
  while (out.size() % 8) out.push_back('\0');
  const uint64_t shoff = out.size();
  const auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size) {
    put<uint32_t>(out, name), put<uint32_t>(out, type);
    put<uint64_t>(out, 0), put<uint64_t>(out, 0), put<uint64_t>(out, off), put<uint64_t>(out, size);
    put<uint32_t>(out, 0), put<uint32_t>(out, 0), put<uint64_t>(out, 16), put<uint64_t>(out, 0);
  };
  shdr(0, 0, 0, 0);
  shdr(1, 3, names_off, names.size());
  for (size_t i = 0; i < kernels.size(); ++i) {
    shdr(text_name[i], 1, places[i].text_off, places[i].text_size);
    shdr(info_name[i], 0x70000000, places[i].info_off, info.size());
  }
  out[0] = 0x7f, out[1] = 'E', out[2] = 'L', out[3] = 'F', out[4] = 2, out[5] = 1, out[6] = 1, out[7] = 0x41, out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = static_cast<uint16_t>(2 + 2 * kernels.size()), shstrndx = 1;
  const uint32_t flags = 0x06000002u | (static_cast<uint32_t>(sm) << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

struct Env {
  int sm;
  MemoryManager mem{1 << 24};
  DeviceProfile prof;
  explicit Env(int sm_ = 86) : sm(sm_), prof(load_gpu(sm_ == 86 ? "nvidia/a10" : sm_ == 90 ? "nvidia/h100" : "nvidia/b200")) {}   // sm_86, sm_90, sm_100
  std::shared_ptr<sass::Module> load(const std::vector<Kernel>& kernels) {
    for (const Kernel& k : kernels)
      for (size_t i = 0; i < k.code.size(); ++i) {
        const sass::Instr ins = sass::decode(k.code[i].w, 16 * i, sm);
        std::string want = k.code[i].text;
        if (sm >= 90 && want.rfind("STG.E [", 0) == 0) want.insert(6, "desc[UR4]");   // (from sm_90 a store names its cache-policy descriptor; UR4 reads zero)
        if (sass::to_text(ins) != want)
          throw vtest::Failure(k.name + " word " + std::to_string(i) + " is \"" + sass::to_text(ins) + "\", not \"" + want + "\"");
      }
    const std::string c = make_cubin(kernels, sm);
    return sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), mem, prof);
  }
  std::shared_ptr<sass::Module> load(const std::vector<Insn>& code) { return load(std::vector<Kernel>{{"k", code}}); }
  // Runs the module's kernel over `blocks` blocks of `threads`, each storing to out[block]; returns out.
  std::vector<uint32_t> run(sass::Module& m, uint32_t blocks, uint32_t threads, const char* kernel = "k", exec::LaunchStats* stats = nullptr,
                              uint32_t shared_bytes = 0) {
    const uint64_t out = mem.alloc(blocks * 4);
    const std::vector<uint32_t> poison(blocks, 0x5A5A5A5Au);
    mem.write(out, poison.data(), blocks * 4);
    exec::LaunchConfig cfg;
    cfg.grid = {blocks, 1, 1};
    cfg.block = {threads, 1, 1};
    cfg.shared_bytes = shared_bytes;
    std::vector<uint8_t> arg(8);
    std::memcpy(arg.data(), &out, 8);
    const exec::LaunchStats st = sass::launch(m, kernel, cfg, {arg}, mem, prof);
    if (stats) *stats = st;
    std::vector<uint32_t> got(blocks);
    mem.read(out, got.data(), blocks * 4);
    return got;
  }
};

// One block of `threads`, each thread storing to out[threadIdx.x]; returns out.
std::vector<uint32_t> run_words(Env& e, sass::Module& m, uint32_t threads) {
  const uint64_t out = e.mem.alloc(threads * 4);
  const std::vector<uint32_t> poison(threads, 0x5A5A5A5Au);
  e.mem.write(out, poison.data(), threads * 4);
  exec::LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {threads, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &out, 8);
  sass::launch(m, "k", cfg, {arg}, e.mem, e.prof);
  std::vector<uint32_t> got(threads);
  e.mem.read(out, got.data(), threads * 4);
  return got;
}

struct OneThread {
  OneThread() { ::setenv("VGPU_THREADS", "1", 1); }
  ~OneThread() { ::unsetenv("VGPU_THREADS"); }
};

}  // namespace

VTEST(every_block_of_a_grid_runs_on_a_warp_that_has_not_finished) {
  OneThread one;
  Env e;
  const auto m = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kStoreR0, kExit});
  for (uint32_t threads : {1u, 32u, 64u, 100u}) {
    const std::vector<uint32_t> got = e.run(*m, 5, threads);
    for (uint32_t b = 0; b < 5; ++b) VCHECK_EQ(got[b], b);
  }
}

VTEST(a_register_a_kernel_sets_reads_back_as_set) {
  OneThread one;
  Env e;
  const auto m = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kSetR9, kStoreR9, kExit});
  for (uint32_t v : e.run(*m, 3, 64)) VCHECK_EQ(v, 0x12345678u);
}

VTEST(registers_start_at_zero_in_the_next_block_and_the_next_launch) {
  OneThread one;
  Env e;
  const auto sets = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kSetR9, kStoreR9, kExit});
  const auto reads = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kStoreR9, kExit});
  for (int round = 0; round < 3; ++round) {
    for (uint32_t v : e.run(*sets, 4, 128)) VCHECK_EQ(v, 0x12345678u);
    for (uint32_t v : e.run(*reads, 4, 128)) VCHECK_EQ(v, 0u);
    for (uint32_t v : e.run(*reads, 4, 32)) VCHECK_EQ(v, 0u);
  }
}

// The uniform registers are cleared by what the module can write (Warp::ur_extent), not by what the last block did: a kernel that sets the first and
// the last register a program can write, then the next block, launch and kernel, read zero in both.
VTEST(uniform_registers_start_at_zero_in_the_next_block_and_the_next_launch) {
  OneThread one;
  Env e;
  const auto sets62 = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kSetUr62, kR9FromUr62, kStoreR9, kExit});
  const auto sets0 = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kSetUr0, kR9FromUr0, kStoreR9, kExit});
  const auto reads62 = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kR9FromUr62, kStoreR9, kExit});
  const auto reads0 = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kR9FromUr0, kStoreR9, kExit});
  for (int round = 0; round < 3; ++round) {
    for (uint32_t v : e.run(*sets62, 4, 128)) VCHECK_EQ(v, 0x1234u);
    for (uint32_t v : e.run(*reads62, 4, 128)) VCHECK_EQ(v, 0u);
    for (uint32_t v : e.run(*sets0, 4, 64)) VCHECK_EQ(v, 0xabcdu);
    for (uint32_t v : e.run(*reads0, 4, 64)) VCHECK_EQ(v, 0u);
    for (uint32_t v : e.run(*reads62, 3, 32)) VCHECK_EQ(v, 0u);
    // (and the block itself: a warp's second block, after a first that set it, reads zero)
    for (uint32_t v : e.run(*sets62, 1, 32)) VCHECK_EQ(v, 0x1234u);
    for (uint32_t v : e.run(*reads62, 5, 32)) VCHECK_EQ(v, 0u);
  }
}

// A block's counts start at zero in every block, in the Block a thread keeps: a launch's counts are the sum of its blocks', whatever ran there before.
VTEST(a_blocks_counts_start_at_zero_in_the_next_block_and_the_next_launch) {
  OneThread one;
  Env e;
  const auto m = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kStoreR0, kExit});
  const auto longer = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kSetR9, kSetR9, kSetR9, kStoreR9, kExit});
  exec::LaunchStats first;
  e.run(*m, 6, 64, "k", &first);
  VCHECK_EQ(first.blocks, uint64_t{6});
  VCHECK_EQ(first.warps, uint64_t{12});
  VCHECK_EQ(first.instructions, uint64_t{6 * 2 * 6});   // six instructions in each of two warps of each of six blocks
  for (int round = 0; round < 2; ++round) {
    exec::LaunchStats long_run, again;
    e.run(*longer, 3, 32, "k", &long_run);
    VCHECK_EQ(long_run.instructions, uint64_t{3 * 9});
    e.run(*m, 6, 64, "k", &again);
    VCHECK_EQ(again.instructions, first.instructions);
    VCHECK_EQ(again.global_stores, first.global_stores);
    VCHECK_EQ(again.thread_instructions, first.thread_instructions);
    VCHECK_EQ(again.warps, first.warps);
  }
}

VTEST(each_kernel_of_a_module_starts_at_its_own_code_and_keeps_its_constant_banks) {
  OneThread one;
  Env e;
  // j ends at once; k is the second kernel of the module, so its code does not start at the section's base.
  const auto m = e.load({{"j", {kExit}}, {"k", {kCtaid, kPtrLo, kPtrHi, kAddr, kStoreR0, kExit}}});
  for (int round = 0; round < 2; ++round) {
    for (uint32_t v : e.run(*m, 4, 32, "j")) VCHECK_EQ(v, 0x5A5A5A5Au);
    const std::vector<uint32_t> got = e.run(*m, 4, 32, "k");
    for (uint32_t b = 0; b < 4; ++b) VCHECK_EQ(got[b], b);
  }
}

// The tier's run loop ends a run of instructions at a VOTE, a branch or a BSSY and goes on into the next group itself. These kernels write the
// lane (S2R SR_TID.X) to the address R10 = out + 4 * tid, so every lane's word is checked. tid >= 8 is P0.
const Insn kTid = {0x000e220000002100ull, 0x0000000000007919ull, "S2R R0, SR_TID.X"};
const Insn kSetR9One = {0x000fc60000000f00ull, 0x0000000100097802ull, "MOV R9, 0x1"};
const Insn kP0TidGe8 = {0x000fe40003f06070ull, 0x000000080000780cull, "ISETP.GE.U32.AND P0, PT, R0, 0x8, PT"};
const Insn kVoteAnyP0 = {0x000fe400000e0100ull, 0x0000000000097806ull, "VOTE.ANY R9, PT, P0"};
const Insn kVoteAnyNotP0 = {0x000fc800040e0100ull, 0x0000000000097806ull, "VOTE.ANY R9, PT, !P0"};
const Insn kBraP0Skip1 = {0x000fea0003800000ull, 0x0000001000000947ull, "@P0 BRA 0x80"};

// A full warp votes: every lane gets the ballot of the 24 lanes with tid >= 8 (a converged run that ends at the VOTE).
VTEST(a_full_warp_vote_gives_every_lane_the_ballot) {
  OneThread one;
  Env e;
  const auto m = e.load({kTid, kPtrLo, kPtrHi, kAddr, kP0TidGe8, kVoteAnyP0, kStoreR9, kExit});
  for (const uint32_t v : run_words(e, *m, 32)) VCHECK_EQ(v, 0xffffff00u);
}

// Lanes below 8 take no branch and vote among themselves (a partial warp: the ballot is over the 8 lanes, its register written for those lanes only);
// the others branch over the vote and keep R9 = 1. The branch splits the warp, and the lowest address runs first.
VTEST(a_diverged_warp_votes_among_the_lanes_that_run_it) {
  OneThread one;
  Env e;
  const auto m = e.load({kTid, kPtrLo, kPtrHi, kAddr, kSetR9One, kP0TidGe8, kBraP0Skip1, kVoteAnyNotP0, kStoreR9, kExit});
  const std::vector<uint32_t> got = run_words(e, *m, 32);
  for (uint32_t l = 0; l < 32; ++l) VCHECK_EQ(got[l], l < 8 ? 0xffu : 1u);
  // (a block of 20 threads: the lanes past the block never exist)
  const std::vector<uint32_t> few = run_words(e, *m, 20);
  for (uint32_t l = 0; l < 20; ++l) VCHECK_EQ(few[l], l < 8 ? 0xffu : 1u);
}


// ---- state beyond the warps: block barriers, mbarriers, warpgroup MMAs, Tensor Memory, and the uniform registers of sm_100 ----
// Runner::init_block sets these again only when a block may have changed them (a flag on the barriers, an empty() test, TensorMemory::pristine()).
// Each test leaves one of them in a state a new block must not see, then runs a block that would misbehave if it did. In a debug build init_block
// also compares every warp, barrier, mbarrier, warpgroup MMA and Tensor Memory with a new one's, so reaching the state is enough there.

// The out pointer is at c[0x0][0x160] for any architecture here: the cubins have no PARAM_CBANK record, so the parameters start at the default base
// (a real cubin from sm_90 says 0x210).
Insn ptr_lo(int) { return kPtrLo; }
Insn ptr_hi(int) { return kPtrHi; }
std::vector<Insn> head(int sm) { return {kCtaid, ptr_lo(sm), ptr_hi(sm), kAddr}; }
std::vector<Insn> cat(std::vector<Insn> a, const std::vector<Insn>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
std::string hex(uint32_t v) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%x", v);
  return b;
}
// UMOV URn, imm and MOV R9, URn for any uniform register (8 bits from sm_100).
Insn umov(unsigned ur, uint32_t imm) {
  static std::vector<std::string> keep;   // (the text outlives the Insn)
  keep.reserve(64);
  keep.push_back("UMOV UR" + std::to_string(ur) + ", " + hex(imm));
  return {0x000fe20000000000ull, 0x0000000000007882ull | uint64_t{ur} << 16 | uint64_t{imm} << 32, keep.back().c_str()};
}
Insn r9_from_ur(unsigned ur) {
  static std::vector<std::string> keep;
  keep.reserve(64);
  keep.push_back("MOV R9, UR" + std::to_string(ur));
  return {0x000fc80008000f00ull, 0x0000000000097c02ull | uint64_t{ur} << 32, keep.back().c_str()};
}

// A barrier some of whose threads have arrived is not there in the next block, launch or kernel: a block of one warp arrives at barrier 1, which
// waits for 128 threads (BAR.ARV does not wait, so the block ends with the barrier a quarter full); the next kernel's four warps then wait on it
// for 128 threads, and complete only if it starts empty. (A barrier that kept the earlier warp would complete early, and the fourth warp would wait
// for ever.)
const Insn kBarArv = {0x000fec0000002000ull, 0x0042000000007b1dull, "BAR.ARV 0x1, 0x80"};
const Insn kBarSync = {0x000fe20000010000ull, 0x0042000000007b1dull, "BAR.SYNC.DEFER_BLOCKING 0x1, 0x80"};

VTEST(a_barrier_left_part_full_by_a_block_is_empty_in_the_next_block_and_the_next_launch) {
  OneThread one;
  for (int sm : {86, 90, 100}) {
    Env e(sm);
    const auto arrives = e.load({kBarArv, kExit});
    const auto syncs = e.load(cat(head(sm), {kBarSync, kStoreR0, kExit}));
    for (int round = 0; round < 3; ++round) {
      e.run(*arrives, 1, 32);
      const std::vector<uint32_t> got = e.run(*syncs, 1, 128);
      VCHECK_EQ(got[0], 0u);
      // (a grid of blocks of one warp each: the barrier has one block's arrival at a time, and never completes)
      e.run(*arrives, 3, 32);
      const std::vector<uint32_t> again = e.run(*syncs, 2, 128);
      VCHECK_EQ(again[0], 0u);
      VCHECK_EQ(again[1], 1u);
    }
  }
}

// An mbarrier a block touched (a SYNCS.ARRIVE makes the side record its transaction count lives in) is gone in the next block, launch and kernel.
// One thread of one block expects 64 transaction bytes on the mbarrier at shared offset 0 and ends; a thread of the next block makes the barrier's
// last arrival (0x100000 of them) and reads the word the arrival left. The phase flips only if no transaction is still due: a record kept from the
// earlier block would hold it back, and the word's phase bit would read 0 and not 1.
const Insn kSetR0To64 = {0x000fc60000000f00ull, 0x0000004000007802ull, "MOV R0, 0x40"};
const Insn kSetR6ToCount = {0x000fc60000000f00ull, 0x0010000000067802ull, "MOV R6, 0x100000"};
const Insn kMovR9R11 = {0x000fc80000000f00ull, 0x0000000b00097202ull, "MOV R9, R11"};
const Insn kExpectTx = {0x000fe20008300404ull, 0x00000000ffff79a7ull, "SYNCS.ARRIVE.TRANS64.RED.A0TR RZ, [UR4], R0"};
const Insn kArriveAll = {0x000fe20008500004ull, 0x00000006ffff79a7ull, "SYNCS.ARRIVE.TRANS64.ART0 RZ, [UR4], R6"};
const Insn kArriveOnce = {0x000e62000810000aull, 0x000000ffff0a79a7ull, "SYNCS.ARRIVE.TRANS64.A1T0 R10, [UR10], RZ"};

VTEST(an_mbarrier_a_block_used_is_gone_in_the_next_block_and_the_next_launch) {
  OneThread one;
  for (int sm : {90, 100}) {
    Env e(sm);
    const auto expects = e.load({kSetR0To64, kExpectTx, kExit});
    const auto probes = e.load(cat({kSetR6ToCount, kArriveAll, kArriveOnce, kMovR9R11}, cat(head(sm), {kStoreR9, kExit})));
    const uint32_t shared = 64;
    for (int round = 0; round < 3; ++round) {
      for (uint32_t v : e.run(*probes, 2, 1, "k", nullptr, shared)) VCHECK_EQ(v, 0x80000000u);
      e.run(*expects, 1, 1, "k", nullptr, shared);
      for (uint32_t v : e.run(*probes, 3, 1, "k", nullptr, shared)) VCHECK_EQ(v, 0x80000000u);
      e.run(*expects, 2, 1, "k", nullptr, shared);   // (a grid: each block leaves its record, the next block finds none)
      for (uint32_t v : e.run(*probes, 1, 1, "k", nullptr, shared)) VCHECK_EQ(v, 0x80000000u);
    }
  }
}

// A warpgroup MMA only some of the warpgroup's warps issued leaves its operands snapshot (the warp that comes last takes the last share): a one-warp
// block issues HGMMA and ends, and the next block's first HGMMA must be the warpgroup's first again.
const Insn kHgmma = {0x000fe20008000804ull, 0x00000000040479f0ull, "HGMMA.64x8x16.F32 R4, gdesc[UR4], R4, gsb0"};

const Insn kTid32 = {0x000e220000002100ull, 0x0000000000007919ull, "S2R R0, SR_TID.X"};
const Insn kP0TidGe32 = {0x000fe40003f06070ull, 0x000000200000780cull, "ISETP.GE.U32.AND P0, PT, R0, 0x20, PT"};
const Insn kP0Exit = {0x000fea0003800000ull, 0x000000000000094dull, "@P0 EXIT"};

VTEST(a_warpgroup_mma_only_some_warps_issued_is_not_there_in_the_next_block_and_the_next_launch) {
  OneThread one;
  Env e(90);
  const auto mma = e.load({kTid32, kP0TidGe32, kP0Exit, kHgmma, kExit});   // warp 0 of a warpgroup issues it; warps 1 to 3 leave
  for (int round = 0; round < 3; ++round) {
    exec::LaunchStats st;
    e.run(*mma, 3, 128, "k", &st, 1 << 16);
    VCHECK_EQ(st.blocks, uint64_t{3});
  }
}

// Tensor Memory a block allocated and did not free is free in the next: the same kernel, run in every block, is given the same granule.
// UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR5, UR5 asks for UR5 granules and answers with the first.
const Insn kTmemFind = {0x000e640008000800ull, 0x00000005000575e3ull, "UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR5, UR5"};

VTEST(tensor_memory_a_block_allocated_is_free_in_the_next_block_and_the_next_launch) {
  OneThread one;
  Env e(100);
  const auto allocs = e.load(cat(head(100), {umov(5, 1), kTmemFind, r9_from_ur(5), kStoreR9, kExit}));
  for (int round = 0; round < 3; ++round)
    for (uint32_t v : e.run(*allocs, 5, 32)) VCHECK_EQ(v, 0u);   // (never freed: a block that kept it would be given granule 1, 2, ...)
}

// From sm_100 a uniform register is named by 8 bits, so a program can write UR200: the next block, launch and kernel read zero in it.
VTEST(uniform_registers_past_63_start_at_zero_from_sm_100) {
  OneThread one;
  Env e(100);
  const auto sets = e.load(cat(head(100), {umov(200, 0x5678), r9_from_ur(200), kStoreR9, kExit}));
  const auto sets_hi = e.load(cat(head(100), {umov(255 - 1, 0x9abc), r9_from_ur(254), kStoreR9, kExit}));
  const auto reads = e.load(cat(head(100), {r9_from_ur(200), kStoreR9, kExit}));
  const auto reads_hi = e.load(cat(head(100), {r9_from_ur(254), kStoreR9, kExit}));
  const auto reads_64 = e.load(cat(head(100), {r9_from_ur(64), kStoreR9, kExit}));
  for (int round = 0; round < 3; ++round) {
    for (uint32_t v : e.run(*sets, 3, 64)) VCHECK_EQ(v, 0x5678u);
    for (uint32_t v : e.run(*reads, 3, 64)) VCHECK_EQ(v, 0u);
    for (uint32_t v : e.run(*sets_hi, 3, 32)) VCHECK_EQ(v, 0x9abcu);
    for (uint32_t v : e.run(*reads_hi, 3, 32)) VCHECK_EQ(v, 0u);
    for (uint32_t v : e.run(*reads_64, 2, 32)) VCHECK_EQ(v, 0u);
    for (uint32_t v : e.run(*sets, 1, 32)) VCHECK_EQ(v, 0x5678u);
    for (uint32_t v : e.run(*reads, 2, 32)) VCHECK_EQ(v, 0u);
  }
}

// The return-address register (RPCMOV, kept per lane out of the warp's hot members: Warp::Rare) is zero in the next block, launch and kernel; the
// block that sets it and the one that reads it are different kernels, so only the warp's reset can carry it over.
const Insn kSetR2 = {0x000fc60000000f00ull, 0x1234567800027802ull, "MOV R2, 0x12345678"};
const Insn kRpcLoFromR2 = {0x000fcc0000000000ull, 0x0000000200007352ull, "RPCMOV.32 Rpc.LO, R2"};
const Insn kR9FromRpcLo = {0x000fe20000000000ull, 0x0000000000097353ull, "RPCMOV.32 R9, Rpc.LO"};

VTEST(the_return_address_register_starts_at_zero_in_the_next_block_and_the_next_launch) {
  OneThread one;
  Env e(90);
  const auto sets = e.load(cat(head(90), {kSetR2, kRpcLoFromR2, kR9FromRpcLo, kStoreR9, kExit}));
  const auto reads = e.load(cat(head(90), {kR9FromRpcLo, kStoreR9, kExit}));
  for (int round = 0; round < 3; ++round) {
    for (uint32_t v : e.run(*sets, 3, 64)) VCHECK_EQ(v, 0x12345678u);
    for (uint32_t v : e.run(*reads, 3, 64)) VCHECK_EQ(v, 0u);
    for (uint32_t v : e.run(*sets, 1, 32)) VCHECK_EQ(v, 0x12345678u);
    for (uint32_t v : e.run(*reads, 2, 32)) VCHECK_EQ(v, 0u);
  }
}

// TensorMemory::pristine() is what init_block asks before it skips reset(): it must say no for every member reset() puts back, and yes after reset().
VTEST(tensor_memory_is_pristine_only_when_every_member_is_what_reset_leaves) {
  using TM = exec::TensorMemory;
  VCHECK(TM{}.pristine());
  const std::vector<std::function<void(TM&)>> dirty = {
      [](TM& t) { t.cells.assign(8, 0); },
      [](TM& t) { t.live[0] = 32; },
      [](TM& t) { t.exclusive = true; },
      [](TM& t) { t.relinquished = true; },
      [](TM& t) { t.peer_alloc.emplace_back(0, 32); },
      [](TM& t) { t.peer_dealloc.emplace_back(0, 0); },
      [](TM& t) { t.collector_b[0].valid = true; },
      [](TM& t) { t.collector_b[1].desc = 7; },
      [](TM& t) { t.collector_b[2].b_fields = 7; },
      [](TM& t) { t.collector_b[3].valid = true; },
      [](TM& t) { t.collector_b[3].desc = 1; },
      [](TM& t) { t.collector_b[3].b_fields = 1; },
  };
  for (size_t i = 0; i < dirty.size(); ++i) {
    TM t;
    dirty[i](t);
    VCHECK(!t.pristine());
    t.reset();
    VCHECK(t.pristine());
    VCHECK(t.cells.empty() && t.live.empty() && t.peer_alloc.empty() && t.peer_dealloc.empty() && !t.exclusive && !t.relinquished);
  }
  TM all;
  for (const auto& d : dirty) d(all);
  all.reset();
  VCHECK(all.pristine());
}

// AddressSanitizer makes a block cost more than the pool's break-even (100 us), so the launches below are genuinely worth the pool there.
#if defined(__SANITIZE_THREAD__) || defined(__SANITIZE_ADDRESS__)
#define VGPU_SASS_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer) || __has_feature(address_sanitizer)
#define VGPU_SASS_TSAN 1
#endif
#endif

#ifndef VGPU_SASS_TSAN
// The SASS twin of test_sched's a_pool_slowed_by_a_crowded_cpu_...: four threads on one CPU make every hand-off wait for a time
// slice, and a launch that was slow for that reason must not be costed as one with much work (the cost is the time the threads
// spent running the launch's bodies, run_parallel's tally), or the kernel stays on the pool and stays slow for good. Eight blocks of a few
// instructions are far below the work that pays for a second thread. A child, so the pool's threads are born on the one CPU.
VTEST(a_crowded_cpu_does_not_keep_a_small_sass_launch_on_the_pool) {
  const pid_t pid = fork();
  VCHECK(pid >= 0);
  if (pid == 0) {
    alarm(120);
    int bad = 0;
    cpu_set_t all, one;
    CPU_ZERO(&all);
    CPU_ZERO(&one);
    if (sched_getaffinity(0, sizeof all, &all) != 0) _exit(2);
    for (int c = 0; c < CPU_SETSIZE; ++c)
      if (CPU_ISSET(c, &all)) { CPU_SET(c, &one); break; }
    if (sched_setaffinity(0, sizeof one, &one) != 0) _exit(2);
    ::unsetenv("VGPU_POOL_MIN_WORK_US");
    ::setenv("VGPU_THREADS", "4", 1);
    try {
      Env e;
      const auto m = e.load({kCtaid, kPtrLo, kPtrHi, kAddr, kStoreR0, kExit});
      for (int i = 0; i < 2; ++i) e.run(*m, 8, 32);   // the first launch of a kernel has no cost to go by
      const uint64_t calls = exec::worker_pool_stats().calls;
      for (int i = 0; i < 400; ++i) {
        const std::vector<uint32_t> got = e.run(*m, 8, 32);
        for (uint32_t b = 0; b < 8; ++b) bad += got[b] != b;
      }
      bad += exec::worker_pool_stats().calls - calls > 40;   // a few may read a cost taken while preempted; not most
    } catch (...) {
      bad = 1;
    }
    _exit(bad ? 1 : 0);
  }
  int status = 0;
  VCHECK_EQ(waitpid(pid, &status, 0), pid);
  VCHECK(WIFEXITED(status));
  VCHECK_EQ(WEXITSTATUS(status), 0);
}
#endif

VTEST_MAIN
