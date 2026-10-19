// The matrix instructions (v_mfma_*, v_smfmac_*, RDNA's v_wmma_*) worked out a whole tile at a time, against the way they were
// worked out before (an element at a time, through lambdas that look at the operand's type for each one): every matrix instruction
// of the ISA corpora, run on a wave of pseudo-random registers with set_matrix_fast_path on and off, must leave the same
// bits behind -- registers, LDS, memory, counters, or the text of the error. Five kinds of registers: any bits (NaNs and infinities
// everywhere), small integers, floats with the special values among them, half the lanes off (an error, the same on both roads),
// and small finite numbers whatever type they are read as (no NaNs: the numbers a real program multiplies).
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
    {"isa_corpus.txt", amd::gcn::Target::Gfx942},          {"isa_corpus_gfx950.txt", amd::gcn::Target::Gfx950},
    {"isa_corpus_gfx90a.txt", amd::gcn::Target::Gfx90a},   {"isa_corpus_gfx1100.txt", amd::gcn::Target::Gfx1100},
    {"isa_corpus_gfx1201.txt", amd::gcn::Target::Gfx1200}, {"isa_corpus_matrix_gfx942.txt", amd::gcn::Target::Gfx942},
    {"isa_corpus_matrix_gfx950.txt", amd::gcn::Target::Gfx950}, {"isa_corpus_matrix_gfx1100.txt", amd::gcn::Target::Gfx1100},
    {"isa_corpus_matrix_gfx1201.txt", amd::gcn::Target::Gfx1200},
};

bool is_matrix(const std::string& text) {
  return text.rfind("v_mfma", 0) == 0 || text.rfind("v_smfmac", 0) == 0 || text.rfind("v_wmma", 0) == 0;
}

// Runs every matrix instruction of the corpora (those that decode) under `seeds` seeds; returns how many digests it compared.
// `mutated` is set when a digest differs, with the text of the first such instruction.
size_t compare_all(uint32_t seeds, std::string* first_wrong, size_t* wrong_total) {
  size_t compared = 0;
  *wrong_total = 0;
  for (const Corpus& c : kCorpora) {
    std::ifstream f(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + c.file, std::ios::binary);
    if (!f) throw vtest::Failure(std::string("no fixture ") + c.file);
    for (std::string line; std::getline(f, line);) {
      if (line.empty() || line[0] == '#') continue;
      const size_t tab = line.find('\t');
      const std::string text = line.substr(tab + 1);
      if (!is_matrix(text)) continue;
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
      for (uint32_t seed = 0; seed < seeds; ++seed)
        for (uint32_t flavour = 0; flavour < 5; ++flavour) {
          amd::set_matrix_fast_path(true);
          const uint64_t fast = amd::digest_instruction(in, c.target, 1000 + seed, flavour);
          amd::set_matrix_fast_path(false);
          const uint64_t slow = amd::digest_instruction(in, c.target, 1000 + seed, flavour);
          ++compared;
          if (fast != slow && ++*wrong_total == 1) *first_wrong = text + " (" + c.file + ", seed " + std::to_string(seed) + ", flavour " + std::to_string(flavour) + ")";
        }
    }
  }
  amd::set_matrix_fast_path(true);
  return compared;
}

}  // namespace

VTEST(every_matrix_instruction_gives_the_same_bits_on_the_tile_path_as_on_the_reference_path) {
  std::string first;
  size_t wrong = 0;
  const size_t compared = compare_all(8, &first, &wrong);
  VCHECK(compared > 10000);
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(compared) + " differ; first: " + first);
}

VTEST_MAIN
