// Blocks (VGPU_AMD_BLOCKS, Machine::run_block in amd/src/gcn_exec.cpp) against the tier's one-instruction-at-a-time loop, bit for bit:
// programs made of the instruction shapes of the ISA corpora are run both ways by the executor's own loop over rounds, and the digest of
// everything a run leaves -- every wave's registers, masks and program counter, LDS, scratch, memory, the instruction counts, the number of
// rounds, the error an instruction threw and where, and after every round where each wave was and how many steps its turn had counted --
// must agree.
//
// A program is 6 to 40 instructions: mostly register-only shapes the tier has handlers for (so that runs of them form blocks), forward
// and backward branches between its own instructions (a backward one makes a loop; the run ends at a bound on rounds), and now and then any
// shape at all -- memory accesses, instructions with no handler, instructions that change the wave's mode -- which end a block, throw, or
// hand the rest of the program to the interpreter. It ends in s_endpgm. Each program runs on several kinds of state (digest_instruction's
// flavours: small integers, floats with and without NaNs, half the lanes off, any bits, the integers' edges; one or two waves, with loads
// in flight) and with turns of several lengths, so that a turn ends inside a block, on its last instruction and just after it.
//
// The test also checks that it tested something: most instructions ran inside blocks, some blocks were left early because a handler handed
// an instruction back (a NaN), some runs threw and some ended. The cases after the first aim at what random programs seldom do: a wait for
// loads inside a block, a wave in a state the tier leaves to the interpreter, a barrier between two blocks.
// VGPU_BLOCKS_PROGRAMS=n for more programs a target (default 400); VGPU_BLOCKS_DUMP=1 prints the first differing program.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_gcn.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

struct Target {
  amd::gcn::Target target;
  std::vector<const char*> files;
};
const Target kTargets[] = {
    {amd::gcn::Target::Gfx942, {"isa_corpus.txt", "isa_corpus_sopp_gfx942.txt", "isa_corpus_tier_gfx942.txt", "isa_corpus_matrix_gfx942.txt"}},
    {amd::gcn::Target::Gfx1030, {"isa_corpus_gfx1030.txt", "isa_corpus_tier_gfx1030.txt"}},
    {amd::gcn::Target::Gfx1100, {"isa_corpus_gfx1100.txt", "isa_corpus_tier_gfx1100.txt"}},
    {amd::gcn::Target::Gfx1200, {"isa_corpus_gfx1201.txt", "isa_corpus_tier_gfx1201.txt"}},
};

struct Rng {
  uint64_t s;
  uint64_t next() {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  uint64_t below(uint64_t n) { return next() % n; }
};

bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }
bool is_branch(const amd::gcn::Inst& in) { return starts(in.name, "s_branch") || starts(in.name, "s_cbranch"); }
// Instructions that send the program counter somewhere the program is not, or end the wave: kept out of the common pool (a program of them
// would be over at once), still drawn among "any shape".
bool leaves(const amd::gcn::Inst& in) {
  return starts(in.name, "s_endpgm") || starts(in.name, "s_setpc") || starts(in.name, "s_swappc") || starts(in.name, "s_call") ||
         starts(in.name, "s_rfe") || starts(in.name, "s_trap") || starts(in.name, "s_code_end");
}
bool register_only(const amd::gcn::Inst& in) {
  using amd::gcn::Enc;
  switch (in.enc) {
    case Enc::Vop1: case Enc::Vop2: case Enc::Vop3: case Enc::Vop3p: case Enc::Vopc: case Enc::Vopd:
    case Enc::Sop1: case Enc::Sop2: case Enc::Sopk: case Enc::Sopc: case Enc::Sopp:
      return true;
    default: return false;
  }
}

struct Pools {
  std::vector<amd::gcn::Inst> blockable;   // has a tier handler, not the Machine's, register-only, not a branch
  std::vector<amd::gcn::Inst> branches;    // has a tier handler
  std::vector<amd::gcn::Inst> any;         // everything that decodes
  std::vector<amd::gcn::Inst> endpgm;
};

Pools pools_of(const Target& t) {
  Pools p;
  for (const char* file : t.files) {
    std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + file, std::ios::binary);
    if (!in) throw vtest::Failure(std::string("no fixture ") + file);
    for (std::string line; std::getline(in, line);) {
      if (line.empty() || line[0] == '#') continue;
      const size_t tab = line.find('\t');
      std::vector<uint8_t> code;
      std::istringstream words(line.substr(0, tab));
      for (std::string w; words >> w;) {
        const uint32_t v = static_cast<uint32_t>(std::stoul(w, nullptr, 16));
        for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
      }
      amd::gcn::Inst inst;
      try {
        inst = amd::gcn::decode(code, 0, 0, t.target);
      } catch (const std::exception&) {
        continue;
      }
      if (inst.name == "s_endpgm") {
        if (p.endpgm.empty()) p.endpgm.push_back(inst);
        continue;
      }
      // (The host's clock is not the same in two runs.)
      if (inst.name.find("realtime") != std::string::npos || starts(inst.name, "s_sendmsg_rtn")) continue;
      p.any.push_back(inst);
      if (!inst.fast.run) continue;
      if (is_branch(inst)) p.branches.push_back(inst);
      else if (!inst.fast.mach && register_only(inst) && !leaves(inst)) p.blockable.push_back(inst);
    }
  }
  // The corpora hold no branches: s_branch and the six s_cbranch_*, encoded here (SOPP: 0xBF800000, the opcode in 22:16, the offset in 15:0; the
  // opcodes are 2 and 4 to 9 on gfx9 and gfx10, 0x20 to 0x26 from gfx11). The test aims each at an instruction of its program (Inst::target).
  const bool gfx11 = t.target == amd::gcn::Target::Gfx1100 || t.target == amd::gcn::Target::Gfx1200;
  static constexpr uint32_t kOld[] = {2, 4, 5, 6, 7, 8, 9}, kNew[] = {0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26};
  for (int i = 0; i < 7; ++i) {
    const uint32_t word = 0xBF800000u | (gfx11 ? kNew[i] : kOld[i]) << 16 | 1;
    std::vector<uint8_t> code;
    for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(word >> (8 * b)));
    amd::gcn::Inst inst = amd::gcn::decode(code, 0, 0, t.target);
    if (!is_branch(inst)) throw vtest::Failure("not a branch: " + inst.name);
    if (!inst.fast.run) throw vtest::Failure("a branch with no tier handler: " + inst.name);
    p.branches.push_back(inst);
  }
  return p;
}

// Shapes of the pool whose handler hands the instruction back on a state of floats with NaNs among them (flavour 2), found by running each.
std::vector<amd::gcn::Inst> handing_back(const Pools& pool, amd::gcn::Target target, size_t want) {
  std::vector<amd::gcn::Inst> out;
  amd::gcn::tier::set_blocks(false);
  amd::gcn::tier::count_bails(true);
  for (const amd::gcn::Inst& in : pool.blockable) {
    if (in.enc == amd::gcn::Enc::Sop1 || in.enc == amd::gcn::Enc::Sop2 || in.enc == amd::gcn::Enc::Sopk || in.enc == amd::gcn::Enc::Sopc) continue;
    const uint64_t before = amd::gcn::tier::bails();
    amd::digest_program({in, pool.endpgm[0]}, target, 1, 2, 1, 0);
    if (amd::gcn::tier::bails() != before) out.push_back(in);
    if (out.size() >= want) break;
  }
  amd::gcn::tier::count_bails(false);
  return out;
}

std::string listing(const std::vector<amd::gcn::Inst>& program) {
  std::string out;
  uint64_t at = 0;
  for (const amd::gcn::Inst& in : program) {
    char head[64];
    std::snprintf(head, sizeof head, "\n      +0x%llx%s  ", static_cast<unsigned long long>(at), in.fast.run ? (in.fast.mach ? " m" : " t") : "  ");
    out += head + amd::gcn::to_text(in);
    if (is_branch(in)) {
      std::snprintf(head, sizeof head, "  -> +0x%llx", static_cast<unsigned long long>(in.target));
      out += head;
    }
    at += in.size;
  }
  return out;
}

}  // namespace

VTEST(blocks_give_the_bits_of_one_instruction_at_a_time) {
  const bool tier_was = amd::gcn::tier::enabled(), blocks_were = amd::gcn::tier::blocks();
  amd::gcn::tier::set_enabled(true);
  const uint64_t programs = std::getenv("VGPU_BLOCKS_PROGRAMS") ? std::strtoull(std::getenv("VGPU_BLOCKS_PROGRAMS"), nullptr, 0) : 400;
  static constexpr uint32_t kFlavours[] = {1, 10, 11, 2, 0, 3, 20, 12, 13};
  static constexpr uint64_t kSlices[] = {0, 1, 2, 3, 5, 8, 13, 64};
  uint64_t runs = 0, wrong = 0, instructions = 0, in_blocks = 0, threw = 0, ended = 0, with_blocks = 0, left_early = 0;
  std::string report;
  for (const Target& t : kTargets) {
    const Pools pool = pools_of(t);
    VCHECK(pool.blockable.size() > 100);
    VCHECK(!pool.branches.empty());
    VCHECK(pool.endpgm.size() == 1);
    Rng r{0x5eed0000 + static_cast<uint64_t>(t.target)};
    for (uint64_t pn = 0; pn < programs; ++pn) {
      const size_t len = 6 + r.below(35);
      std::vector<amd::gcn::Inst> program;
      bool loops = false;
      for (size_t i = 0; i < len; ++i) {
        const uint64_t pick = r.below(100);
        if (pick < 84) {
          program.push_back(pool.blockable[r.below(pool.blockable.size())]);
          // No instruction with a tier handler counts matrix operations today (the matrix instructions are the interpreter's): one in
          // thirty is given some, so that a block's sum of them is held to the tier's.
          if (r.below(30) == 0) {
            program.back().mops = 1 + static_cast<uint32_t>(r.below(7));
            program.back().mops_type = static_cast<amd::gcn::MopsType>(1 + r.below(static_cast<uint64_t>(amd::gcn::MopsType::Count) - 1));
          }
        } else if (pick < 94) {
          program.push_back(pool.branches[r.below(pool.branches.size())]);
        } else {
          program.push_back(pool.any[r.below(pool.any.size())]);
        }
      }
      program.push_back(pool.endpgm[0]);
      // Aim the branches at the program's own instructions: most forward, one in six backward.
      std::vector<uint64_t> at(program.size());
      for (size_t i = 0, a = 0; i < program.size(); a += program[i].size, ++i) at[i] = a;
      for (size_t i = 0; i + 1 < program.size(); ++i) {
        if (!is_branch(program[i])) continue;
        const bool back = i > 0 && r.below(6) == 0;
        const size_t to = back ? r.below(i) : i + 1 + r.below(program.size() - i - 1);
        program[i].target = at[to];
        loops = loops || back;
      }
      for (const uint32_t flavour : kFlavours) {
        const uint32_t waves = flavour == 12 || flavour == 13 || r.below(4) == 0 ? 2 : 1;
        // (A program with a loop may never end: it runs in turns, and the bound on rounds ends it.)
        uint64_t slice = kSlices[r.below(std::size(kSlices))];
        if (loops && slice == 0) slice = 64;
        const uint64_t seed = pn * 7919 + flavour;
        amd::ProgramRun off_run, on_run;
        amd::gcn::tier::set_blocks(false);
        const uint64_t off = amd::digest_program(program, t.target, seed, flavour, waves, slice, &off_run);
        amd::gcn::tier::set_blocks(true);
        const uint64_t on = amd::digest_program(program, t.target, seed, flavour, waves, slice, &on_run);
        ++runs;
        instructions += on_run.instructions, in_blocks += on_run.in_blocks;
        threw += !on_run.error.empty(), ended += on_run.ended, with_blocks += on_run.in_blocks != 0;
        VCHECK(off_run.in_blocks == 0);
        if (off != on || off_run.error != on_run.error || off_run.instructions != on_run.instructions) {
          if (++wrong <= 4) {
            report += "\n  target " + std::to_string(static_cast<int>(t.target)) + " program " + std::to_string(pn) + " flavour " +
                      std::to_string(flavour) + " waves " + std::to_string(waves) + " slice " + std::to_string(slice) + ": " +
                      std::to_string(off_run.instructions) + " instructions [" + off_run.error + "] one at a time, " +
                      std::to_string(on_run.instructions) + " [" + on_run.error + "] in blocks";
            if (wrong == 1 && std::getenv("VGPU_BLOCKS_DUMP")) report += listing(program);
          }
        }
      }
    }
  }
  // Blocks left before their last instruction because a handler handed the instruction back: a program of shapes that do, between shapes that do not.
  {
    const Pools pool = pools_of(kTargets[0]);
    const std::vector<amd::gcn::Inst> back = handing_back(pool, kTargets[0].target, 6);
    VCHECK(!back.empty());
    std::vector<amd::gcn::Inst> program;
    for (size_t i = 0; i < 24; ++i) program.push_back(i % 4 == 3 ? back[i / 4 % back.size()] : pool.blockable[(i * 37) % pool.blockable.size()]);
    program.push_back(pool.endpgm[0]);
    amd::gcn::tier::count_bails(true);
    amd::gcn::tier::set_blocks(true);
    const uint64_t before = amd::gcn::tier::bails();
    amd::ProgramRun run;
    amd::digest_program(program, kTargets[0].target, 1, 2, 1, 0, &run);
    left_early = amd::gcn::tier::bails() - before;
    VCHECK(run.in_blocks > 0);
    amd::gcn::tier::count_bails(false);
  }
  amd::gcn::tier::set_blocks(blocks_were);
  amd::gcn::tier::set_enabled(tier_was);
  std::fprintf(stderr,
               "blocks: %llu runs each way, %llu instructions of which %llu in blocks, %llu runs with a block, %llu threw, %llu ended, %llu handed back\n",
               static_cast<unsigned long long>(runs), static_cast<unsigned long long>(instructions), static_cast<unsigned long long>(in_blocks),
               static_cast<unsigned long long>(with_blocks), static_cast<unsigned long long>(threw), static_cast<unsigned long long>(ended),
               static_cast<unsigned long long>(left_early));
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(runs) + " runs differ with blocks on:" + report);
  // That the runs were worth making.
  VCHECK(in_blocks * 2 > instructions);
  VCHECK(with_blocks * 10 > runs * 8);
  VCHECK(threw > 0 && threw < runs);
  VCHECK(ended > runs / 10);
  VCHECK(left_early > 0);
}

// The number of bails is the same both ways: a block that meets an instruction its handler hands back leaves it to the plain step, which
// notes it once.
VTEST(a_handed_back_instruction_is_noted_once) {
  const bool blocks_were = amd::gcn::tier::blocks();
  const Pools pool = pools_of(kTargets[0]);
  const std::vector<amd::gcn::Inst> back = handing_back(pool, kTargets[0].target, 6);
  VCHECK(!back.empty());
  std::vector<amd::gcn::Inst> program;
  for (size_t i = 0; i < 30; ++i) program.push_back(i % 3 == 2 ? back[i / 3 % back.size()] : pool.blockable[(i * 53) % pool.blockable.size()]);
  program.push_back(pool.endpgm[0]);
  amd::gcn::tier::count_bails(true);
  uint64_t bails[2] = {0, 0};
  for (int blocks = 0; blocks < 2; ++blocks) {
    amd::gcn::tier::set_blocks(blocks != 0);
    const uint64_t before = amd::gcn::tier::bails();
    for (uint64_t seed = 0; seed < 8; ++seed) amd::digest_program(program, kTargets[0].target, seed, 2, 1, seed % 3 ? 0 : 5);
    bails[blocks] = amd::gcn::tier::bails() - before;
  }
  amd::gcn::tier::count_bails(false);
  amd::gcn::tier::set_blocks(blocks_were);
  VCHECK(bails[0] > 0);
  VCHECK(bails[0] == bails[1]);
}

// A wait for loads in the middle of a block (s_waitcnt with loads in flight, in a work-group of two waves) ends the wave's turn after it has
// run: the block stops there, the wait is counted, and it is not one of the turn's steps. Programs of three plain instructions and a wait,
// over and over; with no turn limit the whole program is one round, so the number of steps the round counted is in the digest.
VTEST(a_wait_inside_a_block_ends_the_turn_there) {
  const bool tier_was = amd::gcn::tier::enabled(), blocks_were = amd::gcn::tier::blocks();
  amd::gcn::tier::set_enabled(true);
  const Pools pool = pools_of(kTargets[0]);
  std::vector<amd::gcn::Inst> waits, plain;
  for (const amd::gcn::Inst& in : pool.blockable) {
    if (starts(in.name, "s_waitcnt") && !starts(in.name, "s_waitcnt_depctr")) waits.push_back(in);
    else if (in.enc == amd::gcn::Enc::Vop2 && plain.size() < 40) plain.push_back(in);
  }
  VCHECK(!waits.empty());
  VCHECK(plain.size() >= 8);
  Rng r{0x77a17};
  uint64_t runs = 0, wrong = 0, in_blocks = 0;
  for (uint64_t pn = 0; pn < 200; ++pn) {
    std::vector<amd::gcn::Inst> program;
    for (int i = 0; i < 24; ++i) program.push_back(i % 4 == 3 ? waits[r.below(waits.size())] : plain[r.below(plain.size())]);
    program.push_back(pool.endpgm[0]);
    for (const uint32_t flavour : {12u, 13u})
      for (const uint64_t slice : {uint64_t{0}, uint64_t{5}}) {
        amd::ProgramRun off_run, on_run;
        amd::gcn::tier::set_blocks(false);
        const uint64_t off = amd::digest_program(program, kTargets[0].target, pn, flavour, 2, slice, &off_run);
        amd::gcn::tier::set_blocks(true);
        const uint64_t on = amd::digest_program(program, kTargets[0].target, pn, flavour, 2, slice, &on_run);
        ++runs, in_blocks += on_run.in_blocks;
        wrong += off != on;
      }
  }
  amd::gcn::tier::set_blocks(blocks_were);
  amd::gcn::tier::set_enabled(tier_was);
  VCHECK(in_blocks > runs * 20);
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(runs) + " runs with waits differ with blocks on");
}

// A wave in a state the tier does not model (VSKIP set, VGPR indexing on, a round mode that is not the default) runs no block: the tier's
// handlers are not called for it one at a time either (tier_step's first line), so a block of them would compute something else.
VTEST(a_wave_the_tier_leaves_to_the_interpreter_runs_no_block) {
  const bool tier_was = amd::gcn::tier::enabled(), blocks_were = amd::gcn::tier::blocks();
  amd::gcn::tier::set_enabled(true);
  const Pools pool = pools_of(kTargets[0]);
  Rng r{0x57a7e};
  uint64_t runs = 0, wrong = 0, differ_from_plain = 0;
  for (uint64_t pn = 0; pn < 60; ++pn) {
    std::vector<amd::gcn::Inst> program;
    for (int i = 0; i < 16; ++i) {
      const amd::gcn::Inst& in = pool.blockable[r.below(pool.blockable.size())];
      // (Vector instructions on registers low enough that an index of up to four still names a register of the file.)
      if (in.enc != amd::gcn::Enc::Vop2 && in.enc != amd::gcn::Enc::Vop1 && in.enc != amd::gcn::Enc::Vop3) {
        --i;
        continue;
      }
      program.push_back(in);
    }
    program.push_back(pool.endpgm[0]);
    for (const uint32_t state : {amd::kProgramVskip, amd::kProgramGprIdx, amd::kProgramRoundToZero}) {
      amd::ProgramRun off_run, on_run;
      amd::gcn::tier::set_blocks(false);
      const uint64_t off = amd::digest_program(program, kTargets[0].target, pn, 10, 1, 0, &off_run, state);
      const uint64_t plain = amd::digest_program(program, kTargets[0].target, pn, 10, 1, 0);
      amd::gcn::tier::set_blocks(true);
      const uint64_t on = amd::digest_program(program, kTargets[0].target, pn, 10, 1, 0, &on_run, state);
      ++runs;
      wrong += off != on || off_run.error != on_run.error;
      differ_from_plain += off != plain;
      VCHECK(on_run.in_blocks == 0);
    }
  }
  amd::gcn::tier::set_blocks(blocks_were);
  amd::gcn::tier::set_enabled(tier_was);
  VCHECK(differ_from_plain * 2 > runs);   // (the states change what the programs compute, so running blocks in them would show)
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(runs) + " runs in a state the tier does not model differ with blocks on");
}

// A barrier is the Machine's: a block ends before it, and the waves of the group meet there as they do one instruction at a time.
VTEST(a_barrier_ends_a_block) {
  const bool tier_was = amd::gcn::tier::enabled(), blocks_were = amd::gcn::tier::blocks();
  amd::gcn::tier::set_enabled(true);
  const Pools pool = pools_of(kTargets[0]);
  std::vector<amd::gcn::Inst> barrier, plain;
  for (const amd::gcn::Inst& in : pool.any)
    if (in.name == "s_barrier" && barrier.empty()) barrier.push_back(in);
  for (const amd::gcn::Inst& in : pool.blockable)
    if (in.enc == amd::gcn::Enc::Vop2 && plain.size() < 40) plain.push_back(in);
  VCHECK(barrier.size() == 1);
  VCHECK(barrier[0].fast.mach != 0);
  Rng r{0xba221e2};
  uint64_t runs = 0, wrong = 0, in_blocks = 0;
  for (uint64_t pn = 0; pn < 100; ++pn) {
    std::vector<amd::gcn::Inst> program;
    for (int i = 0; i < 20; ++i) program.push_back(i % 5 == 4 ? barrier[0] : plain[r.below(plain.size())]);
    program.push_back(pool.endpgm[0]);
    for (const uint32_t waves : {1u, 2u, 3u}) {
      amd::ProgramRun off_run, on_run;
      amd::gcn::tier::set_blocks(false);
      const uint64_t off = amd::digest_program(program, kTargets[0].target, pn, 1, waves, pn % 2 ? 0 : 7, &off_run);
      amd::gcn::tier::set_blocks(true);
      const uint64_t on = amd::digest_program(program, kTargets[0].target, pn, 1, waves, pn % 2 ? 0 : 7, &on_run);
      ++runs, in_blocks += on_run.in_blocks;
      wrong += off != on;
      VCHECK(off_run.ended && on_run.ended);
    }
  }
  amd::gcn::tier::set_blocks(blocks_were);
  amd::gcn::tier::set_enabled(tier_was);
  VCHECK(in_blocks > runs * 10);
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(runs) + " runs with barriers differ with blocks on");
}

// Memory instructions inside blocks: global loads and stores whose addresses are on a memory region (until an instruction before them writes
// the address register: then the access faults, in the middle of a block, and the error says where), scalar memory (s_memtime among it, which
// reads the instruction count: everything before it has to be counted when it runs), LDS reads and writes at small addresses, between plain
// instructions; one to three waves, so that a wave's first store of a round waits for the others inside a block and the wave goes on in the block.
VTEST(memory_instructions_inside_blocks) {
  const bool tier_was = amd::gcn::tier::enabled(), blocks_were = amd::gcn::tier::blocks();
  amd::gcn::tier::set_enabled(true);
  uint64_t runs = 0, wrong = 0, in_blocks = 0, machine = 0, instructions = 0, threw = 0, ended = 0;
  std::string report;
  for (const Target& t : {kTargets[0], kTargets[1]}) {
    const Pools pool = pools_of(t);
    std::vector<amd::gcn::Inst> global, scalar, lds, plain;
    for (const amd::gcn::Inst& in : pool.any) {
      if (!in.fast.run) continue;
      if (in.enc == amd::gcn::Enc::Flat && in.fast.mach && (starts(in.name, "global_load") || starts(in.name, "global_store"))) global.push_back(in);
      else if (in.enc == amd::gcn::Enc::Smem && in.fast.mach) scalar.push_back(in);
      else if (in.enc == amd::gcn::Enc::Ds && !in.fast.mach) lds.push_back(in);
    }
    for (const amd::gcn::Inst& in : pool.blockable)
      if ((in.enc == amd::gcn::Enc::Vop2 || in.enc == amd::gcn::Enc::Sop2) && plain.size() < 80) plain.push_back(in);
    VCHECK(global.size() >= 4);
    VCHECK(!scalar.empty());
    VCHECK(lds.size() >= 4);
    Rng r{0x3e3021 + static_cast<uint64_t>(t.target)};
    for (uint64_t pn = 0; pn < 150; ++pn) {
      // The memory accesses first, so that the other instructions can be chosen among those that leave their address registers alone (a
      // program whose first access faults tests one fault and nothing after it; one access in five may still be aimed elsewhere by a load before it).
      const size_t len = 8 + r.below(24);
      std::vector<amd::gcn::Inst> program(len);
      std::vector<bool> taken(len, false), vreg(260, false), sreg(110, false);
      for (size_t i = 0; i < len; ++i) {
        const uint64_t pick = r.below(100);
        if (pick >= 45) continue;
        program[i] = pick < 25 ? global[r.below(global.size())] : pick < 38 ? lds[r.below(lds.size())] : scalar[r.below(scalar.size())];
        taken[i] = true;
        const amd::gcn::Inst& in = program[i];
        if (in.enc == amd::gcn::Enc::Flat && !in.src.empty() && in.src[0].kind == amd::gcn::OperandKind::Vgpr && in.src[0].index < 256)
          vreg[in.src[0].index] = vreg[in.src[0].index + 1] = true;
        if (in.enc == amd::gcn::Enc::Flat && in.has_saddr && in.saddr < 105) sreg[in.saddr] = sreg[in.saddr + 1] = true;
        if (in.enc == amd::gcn::Enc::Smem && !in.src.empty() && in.src[0].kind == amd::gcn::OperandKind::Sgpr && in.src[0].index < 105)
          sreg[in.src[0].index] = sreg[in.src[0].index + 1] = true;
      }
      const auto leaves_addresses = [&](const amd::gcn::Inst& in) {
        for (const amd::gcn::Operand& o : in.dst)
          for (uint32_t k = 0; k < o.width + 1; ++k) {
            if (o.kind == amd::gcn::OperandKind::Vgpr && o.index + k < 260 && vreg[o.index + k]) return false;
            if (o.kind == amd::gcn::OperandKind::Sgpr && o.index + k < 110 && sreg[o.index + k]) return false;
          }
        return true;
      };
      for (size_t i = 0; i < len; ++i) {
        if (taken[i]) continue;
        do program[i] = plain[r.below(plain.size())];
        while (!leaves_addresses(program[i]));
      }
      program.push_back(pool.endpgm[0]);
      for (const uint32_t waves : {1u, 2u, 3u}) {
        const uint64_t slice = r.below(3) == 0 ? 0 : 1 + r.below(9);
        amd::ProgramRun off_run, on_run;
        amd::gcn::tier::set_blocks(false);
        const uint64_t off = amd::digest_program(program, t.target, pn, 1, waves, slice, &off_run, amd::kProgramAddresses);
        amd::gcn::tier::set_blocks(true);
        const uint64_t on = amd::digest_program(program, t.target, pn, 1, waves, slice, &on_run, amd::kProgramAddresses);
        ++runs, in_blocks += on_run.in_blocks, machine += on_run.machine_in_blocks, instructions += on_run.instructions;
        threw += !on_run.error.empty(), ended += on_run.ended;
        if (off != on || off_run.error != on_run.error) {
          if (++wrong <= 3) {
            report += "\n  target " + std::to_string(static_cast<int>(t.target)) + " program " + std::to_string(pn) + " waves " + std::to_string(waves) + " slice " +
                      std::to_string(slice) + ": [" + off_run.error + "] one at a time, [" + on_run.error + "] in blocks";
            if (wrong == 1 && std::getenv("VGPU_BLOCKS_DUMP")) report += listing(program);
          }
        }
      }
    }
  }
  amd::gcn::tier::set_blocks(blocks_were);
  amd::gcn::tier::set_enabled(tier_was);
  std::fprintf(stderr, "memory in blocks: %llu runs each way, %llu instructions, %llu in blocks, %llu of them the Machine's, %llu threw, %llu ended\n",
               static_cast<unsigned long long>(runs), static_cast<unsigned long long>(instructions), static_cast<unsigned long long>(in_blocks),
               static_cast<unsigned long long>(machine), static_cast<unsigned long long>(threw), static_cast<unsigned long long>(ended));
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(runs) + " runs with memory instructions differ with blocks on:" + report);
  VCHECK(in_blocks * 2 > instructions);
  VCHECK(machine * 20 > instructions);
  VCHECK(threw * 10 > runs && ended * 10 > runs);
}

VTEST_MAIN
