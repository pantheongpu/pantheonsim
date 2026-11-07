// Global and flat accesses whose lanes make no run, or make a run and then gather (amd/src/gcn_tier_machine.inc tier_global,
// Machine::wave_scalars, MemoryManager::load_batch), against the definition: the same instruction run lane by lane (the tier off and
// the memory manager's whole-wave calls off, amd::set_fast_paths) must leave the same registers, memory, counts and error as with the
// tier on and the whole-wave calls on, and with each of the two on alone.
//
// Every flat / global instruction of the ISA corpora (wave64 for CDNA, wave32 for RDNA) is run on a wave whose addresses are inside an
// allocation: consecutive lanes (flavour 14), scattered and half the lanes off (15), a run of consecutive lanes and then a gather from
// a few places so that some lanes read one address twice, every lane on (21) and one of the gathering lanes off (22), lanes a word apart that move two or three words each so that neighbours overlap, as a stencil's do (23, and 24 with a few lanes on the same address and one off), and every lane at one address (25: Machine::wave_scalars reads such a load once), the second half of a wave at the first half's addresses (34) and groups of neighbouring lanes at one address each (35: Machine::load_wave_repeats reads each address once; one lane off half the time). Where the
// gathering starts is every lane of the wave, over the seeds, so that a gather from lane 0 (the whole wave), from lane k (the rest of
// the wave) and from the last lane are all met. No digest is recorded: each run is held to its own lane-by-lane twin.
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
struct Corpus {
  const char* file;
  amd::gcn::Target target;
};
const Corpus kCorpora[] = {
    {"isa_corpus.txt", amd::gcn::Target::Gfx942},   {"isa_corpus_gfx950.txt", amd::gcn::Target::Gfx950},
    {"isa_corpus_gfx90a.txt", amd::gcn::Target::Gfx90a}, {"isa_corpus_gfx1100.txt", amd::gcn::Target::Gfx1100},
    {"isa_corpus_gfx1030.txt", amd::gcn::Target::Gfx1030},
};

std::vector<std::string> lines_of(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no fixture at " + path);
  std::vector<std::string> out;
  for (std::string line; std::getline(in, line);)
    if (!line.empty()) out.push_back(line);
  return out;
}
}  // namespace

VTEST(a_gather_or_a_run_then_a_gather_gives_the_bits_lane_by_lane_gives) {
  const char* only = std::getenv("VGPU_GATHER_ONLY");   // (a substring of the instructions to run)
  const bool tier_was = amd::gcn::tier::enabled();
  size_t shapes = 0, runs = 0, wrong = 0, nonzero = 0;
  std::string report;
  for (const Corpus& c : kCorpora) {
    for (const std::string& line : lines_of(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + c.file)) {
      if (line[0] == '#') continue;
      const size_t tab = line.find('\t');
      std::vector<uint8_t> code;
      std::istringstream words(line.substr(0, tab));
      for (std::string w; words >> w;) {
        const uint32_t v = static_cast<uint32_t>(std::stoul(w, nullptr, 16));
        for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
      }
      const std::string text = line.substr(tab + 1);
      if (only && text.find(only) == std::string::npos) continue;
      amd::gcn::Inst in;
      try {
        in = amd::gcn::decode(code, 0, 0, c.target);
      } catch (const std::exception&) {
        continue;
      }
      if (in.enc != amd::gcn::Enc::Flat) continue;
      ++shapes;
      for (uint64_t seed = 0; seed < 12; ++seed) {
        for (const uint32_t flavour : {14u, 15u, 21u, 22u, 23u, 24u, 25u, 34u, 35u}) {
          amd::gcn::tier::set_enabled(false);
          amd::set_fast_paths(true, false);
          const uint64_t def = amd::digest_instruction(in, c.target, seed * 7919 + shapes, flavour);
          const uint64_t twin[3] = {[&] { amd::gcn::tier::set_enabled(true), amd::set_fast_paths(true, true); return amd::digest_instruction(in, c.target, seed * 7919 + shapes, flavour); }(),
                                    [&] { amd::gcn::tier::set_enabled(false), amd::set_fast_paths(true, true); return amd::digest_instruction(in, c.target, seed * 7919 + shapes, flavour); }(),
                                    [&] { amd::gcn::tier::set_enabled(true), amd::set_fast_paths(true, false); return amd::digest_instruction(in, c.target, seed * 7919 + shapes, flavour); }()};
          ++runs;
          nonzero += def != 0;
          for (int t = 0; t < 3; ++t)
            if (twin[t] != def && ++wrong <= 12)
              report += std::string("\n  ") + c.file + ": " + text + " (flavour " + std::to_string(flavour) + ", seed " + std::to_string(seed) + ", variant " + std::to_string(t) + ")";
        }
      }
    }
  }
  amd::gcn::tier::set_enabled(tier_was);
  amd::set_fast_paths(true, true);
  std::fprintf(stderr, "gather: %zu flat shapes, %zu runs\n", shapes, runs);
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " runs differ from the lane-by-lane bits:" + report);
  VCHECK(only || shapes > 60);
  VCHECK(only || nonzero > 1000);
}

// A load into LDS whose LDS address is below zero (M0 0, offset -16, lane 0: 2^64 - 16) must be refused, not written. The bound was
// `at + bytes > size`, which wraps to 0 here, so the 16 bytes went in front of the LDS buffer and the heap's next free() aborted
// ("double free or corruption", seen as VGPU_GATHER_ONLY=dwordx4 aborting the test above). Every register is zero (flavour 16), so
// M0 is 0 and lane 0 is the lane that wraps. The refusal is read from the error itself: the corruption is silent until a free().
VTEST(a_load_into_lds_below_address_zero_is_refused) {
  size_t seen = 0;
  for (const Corpus& c : kCorpora) {
    for (const std::string& line : lines_of(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + c.file)) {
      if (line[0] == '#') continue;
      const size_t tab = line.find('\t');
      const std::string text = line.substr(tab + 1);
      if (text.find("load_lds") == std::string::npos || text.find("offset:-") == std::string::npos) continue;
      std::vector<uint8_t> code;
      std::istringstream words(line.substr(0, tab));
      for (std::string w; words >> w;) {
        const uint32_t v = static_cast<uint32_t>(std::stoul(w, nullptr, 16));
        for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
      }
      amd::gcn::Inst in;
      try {
        in = amd::gcn::decode(code, 0, 0, c.target);
      } catch (const std::exception&) {
        continue;
      }
      std::string error;
      amd::digest_instruction(in, c.target, 1, 16, &error);
      ++seen;
      if (error.find("writes LDS") == std::string::npos)
        throw vtest::Failure(std::string(c.file) + ": " + text + " was not refused (error: '" + error + "')");
    }
  }
  VCHECK(seen >= 1);
}

VTEST_MAIN
