// The AMD tier (amd/src/gcn_tier.cpp) against the interpreter, bit for bit: every instruction shape of the ISA corpora
// that has a tier handler is run on pseudo-random state both ways (tier off, tier on) and the digests of everything the
// instruction changed -- registers, masks, SCC, the program counter, LDS, memory, the instruction counts, or the error it
// threw -- must agree. Flavours of state: 0 any bits, 1 small integers, 2 floats with the specials and NaNs, 3 half the lanes
// off, 4 (the matrix probe's) small finite numbers; and the tier's six, 10 and 11 NaN-free floats (11 with half the lanes off, so
// the fast path runs rather than hands the instruction back), 12 and 13 a second wave in the work-group and loads in flight, for
// the store gate and the waits (any bits; small integers with half the lanes off), and, for the flat and global accesses only, 14
// and 15, which point the address registers into memory that exists, in consecutive lanes and scattered (5 to 9 do not exist).
// Three seeds each (VGPU_TIER_SEEDS=n for more).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
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
    {"isa_corpus.txt", amd::gcn::Target::Gfx942},          {"isa_corpus_sopp_gfx942.txt", amd::gcn::Target::Gfx942},
    {"isa_corpus_gfx950.txt", amd::gcn::Target::Gfx950},   {"isa_corpus_gfx90a.txt", amd::gcn::Target::Gfx90a},
    {"isa_corpus_gfx1100.txt", amd::gcn::Target::Gfx1100}, {"isa_corpus_gfx1201.txt", amd::gcn::Target::Gfx1200},
    {"isa_corpus_gfx1250.txt", amd::gcn::Target::Gfx1250}, {"isa_corpus_gfx1030.txt", amd::gcn::Target::Gfx1030},
    // The forms of the handlers written by hand (amd/tools/tier-corpus.py): every operand kind, the VOP3 modifiers, SDWA selectors, constants.
    {"isa_corpus_tier_gfx942.txt", amd::gcn::Target::Gfx942}, {"isa_corpus_tier_gfx90a.txt", amd::gcn::Target::Gfx90a},
    {"isa_corpus_tier_gfx950.txt", amd::gcn::Target::Gfx950},
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

VTEST(every_tier_handler_gives_the_interpreters_bits) {
  size_t shapes = 0, with_handler = 0, runs = 0, wrong = 0;
  std::string report;
  std::map<std::string, size_t> failing;   // (by mnemonic)
  std::map<std::string, std::string> example;   // (the first failing run of each)
  const bool was_on = amd::gcn::tier::enabled();
  amd::gcn::tier::count_bails(true);
  const char* only = std::getenv("VGPU_TIER_ONLY");   // (a substring of the instructions to run)
#if defined(__SANITIZE_THREAD__) || defined(__SANITIZE_ADDRESS__)
  constexpr uint64_t kDefaultSeeds = 1;   // (a sanitizer runs it ten to twenty times slower; one seed still covers every shape)
#else
  constexpr uint64_t kDefaultSeeds = 3;
#endif
  const uint64_t seeds = std::getenv("VGPU_TIER_SEEDS") ? std::strtoull(std::getenv("VGPU_TIER_SEEDS"), nullptr, 0) : kDefaultSeeds;
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
      ++shapes;
      if (!in.fast.run) continue;
      ++with_handler;
      // (A branch the corpus holds goes where it was: send it elsewhere, so that taking it shows.)
      if (in.name.rfind("s_branch", 0) == 0 || in.name.rfind("s_cbranch", 0) == 0) in.target = 0x1000;
      for (uint64_t seed = 0; seed < seeds; ++seed)
        for (uint32_t flavour = 0; flavour < 20; ++flavour) {
          if (flavour >= 5 && flavour <= 9) continue;
          if (flavour == 16) continue;   // (every register zero: not a state worth running)
          if ((flavour == 14 || flavour == 15) && in.enc != amd::gcn::Enc::Flat) continue;   // (the flavours that give a global access real addresses)
          if ((flavour == 18 || flavour == 19) && in.enc != amd::gcn::Enc::Ds) continue;   // (and an LDS access consecutive ones)
          amd::gcn::tier::set_enabled(false);
          const uint64_t off = amd::digest_instruction(in, c.target, seed * 7919 + with_handler, flavour);
          amd::gcn::tier::set_enabled(true);
          const uint64_t on = amd::digest_instruction(in, c.target, seed * 7919 + with_handler, flavour);
          ++runs;
          if (off != on && ++failing[in.name] <= 2)
            example[in.name] += std::string("\n    ") + c.file + ": " + text + " (flavour " + std::to_string(flavour) + ", seed " + std::to_string(seed) + ")";
          if (off != on && ++wrong <= 12)
            report += std::string("\n  ") + c.file + ": " + text + " (flavour " + std::to_string(flavour) + ", seed " + std::to_string(seed) + ")";
        }
    }
  }
  amd::gcn::tier::set_enabled(was_on);
  std::fprintf(stderr, "tier handlers: %zu of %zu shapes, %zu runs each way, %llu handed back to the interpreter\n", with_handler, shapes,
               runs, static_cast<unsigned long long>(amd::gcn::tier::bails()));
  if (wrong) {
    report += "\n  differing mnemonics:";
    for (const auto& [name, count] : failing) report += " " + name + "(" + std::to_string(count) + ")";
    for (const auto& [name, text] : example) report += "\n  " + name + ":" + text;
  }
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(runs) + " runs differ with the tier on:" + report);
  VCHECK(with_handler > 300 || only);   // (a run restricted to some instructions finds few)
}

VTEST_MAIN
