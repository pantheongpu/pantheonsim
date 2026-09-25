// What each profiler count should read for counted_mix, the first kernel of
// amd/tests/data/counters.c, from llvm-objdump's listing of it: the kernel has
// no branches, so every wave issues each listed instruction once. Sorted by
// the assembler's mnemonics, not by VirtualGPU's decoder, so the two are
// independent. Shared by test_amd_gcn_counters (the interpreter's counts) and
// test_amd_rocprofiler (the same counts as a profiling tool is handed them).
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "vtest.hpp"

inline bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

// One wave's counts, from the listing.
struct Expected {
  uint64_t all = 0, valu = 0, mfma = 0, salu = 0, smem = 0, vmem = 0, flat = 0, generic = 0, lds = 0, branch = 0;
  uint64_t reads = 0, writes = 0, atomics = 0;
};

inline Expected from_listing(const std::string& dir) {
  std::ifstream in(dir + "counters.gfx942.dis");
  if (!in) throw vtest::Failure("no listing at " + dir + "counters.gfx942.dis");
  Expected e;
  // counted_mix, the first kernel in the listing, up to its s_endpgm.
  // The sequencer's own instructions: they go to no execution unit.
  const std::vector<std::string> sequencer = {"s_nop", "s_waitcnt", "s_endpgm", "s_barrier", "s_sleep",
                                              "s_setprio", "s_sendmsg", "s_trap", "s_icache_inv"};
  std::string line;
  while (std::getline(in, line)) {
    const std::string m = line.substr(0, line.find(' '));
    ++e.all;
    if (starts(m, "v_")) {
      ++e.valu;
      e.mfma += starts(m, "v_mfma") || starts(m, "v_smfmac");
    } else if (starts(m, "ds_")) {
      ++e.lds;
    } else if (starts(m, "global_") || starts(m, "flat_") || starts(m, "scratch_")) {
      ++e.vmem;
      ++e.flat;
      e.generic += starts(m, "flat_");
      if (m.find("_atomic") != std::string::npos) ++e.atomics;
      else if (m.find("_store") != std::string::npos) ++e.writes;
      else ++e.reads;
    } else if (starts(m, "buffer_")) {
      ++e.vmem;
    } else if (starts(m, "s_load") || starts(m, "s_buffer_load") || starts(m, "s_store") || starts(m, "s_dcache") ||
               starts(m, "s_memtime") || starts(m, "s_memrealtime")) {
      ++e.smem;
    } else if (m == "s_branch" || starts(m, "s_cbranch_")) {
      ++e.branch;
    } else if (starts(m, "s_")) {
      bool seq = false;
      for (const std::string& s : sequencer) seq = seq || m == s;
      e.salu += !seq;
    }
    if (m == "s_endpgm") break;
  }
  return e;
}
