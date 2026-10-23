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
    {"isa_corpus_matrix_gfx1201.txt", amd::gcn::Target::Gfx1200}, {"isa_corpus_matrix_gfx1250.txt", amd::gcn::Target::Gfx1250},
    {"isa_corpus_gfx1250.txt", amd::gcn::Target::Gfx1250},
};

bool is_matrix(const std::string& text) {
  return text.rfind("v_mfma", 0) == 0 || text.rfind("v_smfmac", 0) == 0 || text.rfind("v_wmma", 0) == 0 || text.rfind("v_swmmac", 0) == 0;
}

// Runs every matrix instruction of the corpora (those that decode) under `seeds` seeds; returns how many digests it compared.
// `mutated` is set when a digest differs, with the text of the first such instruction.
size_t compare_all(uint32_t seeds, bool shuffle_registers, std::string* first_wrong, size_t* wrong_total) {
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
          amd::gcn::Inst shuffled = in;
          if (shuffle_registers) {
            // The same instruction with its registers elsewhere, overlapping or not, in either bank: the destination over a source, a
            // source over another, the accumulator in the destination's registers or half over them.
            uint64_t x = (seed * 5 + flavour + 1) * 0x9E3779B97F4A7C15ull + text.size();
            const auto next = [&] { return x ^= x << 13, x ^= x >> 7, x ^= x << 17; };
            const size_t movable = shuffled.src.size() < 3 ? shuffled.src.size() : 3;
            const auto move = [&](amd::gcn::Operand& o) {
              if (o.kind != amd::gcn::OperandKind::Vgpr && o.kind != amd::gcn::OperandKind::Agpr) return;
              o.index = static_cast<uint32_t>(next() % 24) * (next() % 2 ? 1 : 2);
              if (next() % 4 == 0) o.kind = o.kind == amd::gcn::OperandKind::Vgpr ? amd::gcn::OperandKind::Agpr : amd::gcn::OperandKind::Vgpr;
            };
            move(shuffled.dst[0]);
            for (size_t k = 0; k < movable; ++k) move(shuffled.src[k]);
          }
          amd::set_matrix_fast_path(true);
          const uint64_t fast = amd::digest_instruction(shuffled, c.target, 1000 + seed, flavour);
          amd::set_matrix_fast_path(false);
          const uint64_t slow = amd::digest_instruction(shuffled, c.target, 1000 + seed, flavour);
          ++compared;
          if (fast != slow && ++*wrong_total == 1) *first_wrong = text + " (" + c.file + ", seed " + std::to_string(seed) + ", flavour " + std::to_string(flavour) + ")";
        }
    }
  }
  amd::set_matrix_fast_path(true);
  return compared;
}

// The modifier bits a corpus line does not carry: BLGP's lane permutations of B (and the doubles' negations of A, B and C, which are
// BLGP's bits there), the small-float formats CBSZ and BLGP name, and the scale selectors of the scaled forms. The tile path must give
// the reference's bits for each, with the registers as they are and moved over one another.
size_t compare_modifiers(uint32_t seeds, std::string* first_wrong, size_t* wrong_total, size_t* variants, size_t* never_tiled,
                         std::string* first_slow) {
  size_t compared = 0;
  *wrong_total = 0;
  for (const Corpus& c : kCorpora) {
    if (c.target == amd::gcn::Target::Gfx1100 || c.target == amd::gcn::Target::Gfx1200 || c.target == amd::gcn::Target::Gfx1250) continue;
    std::ifstream f(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + c.file, std::ios::binary);
    if (!f) throw vtest::Failure(std::string("no fixture ") + c.file);
    for (std::string line; std::getline(f, line);) {
      if (line.empty() || line[0] == '#') continue;
      const size_t tab = line.find('\t');
      const std::string text = line.substr(tab + 1);
      if (text.rfind("v_mfma", 0) != 0) continue;
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
      const bool small = text.find("f8f6f4") != std::string::npos;
      const bool scaled = text.rfind("v_mfma_scale", 0) == 0;
      for (uint32_t a = 0; a < (small ? 5u : 1u); ++a)          // CBSZ: A's format
        for (uint32_t b = 0; b < (small ? 5u : 8u); ++b)       // BLGP: B's format, or the permutation, or the doubles' signs
          for (uint32_t sel = 0; sel < (scaled ? 16u : 1u); sel += 3) {
            amd::gcn::Inst v = in;
            if (small) v.cbsz = static_cast<uint8_t>(a);
            v.blgp = static_cast<uint8_t>(b);
            if (scaled) v.scale_sel = static_cast<uint8_t>(sel);
            const uint64_t tiles_before = amd::matrix_tile_count();
            for (uint32_t seed = 0; seed < seeds; ++seed)
              for (uint32_t flavour = 0; flavour < 5; ++flavour) {
                amd::set_matrix_fast_path(true);
                const uint64_t fast = amd::digest_instruction(v, c.target, 2000 + seed, flavour);
                amd::set_matrix_fast_path(false);
                const uint64_t slow = amd::digest_instruction(v, c.target, 2000 + seed, flavour);
                ++compared;
                if (fast != slow && ++*wrong_total == 1)
                  *first_wrong = text + " cbsz " + std::to_string(a) + " blgp " + std::to_string(b) + " sel " + std::to_string(sel) + " (" + c.file +
                                 ", seed " + std::to_string(seed) + ", flavour " + std::to_string(flavour) + ")";
              }
            // (the forms this piece taught the tile path -- small floats, scaled, doubles with BLGP's signs, B permuted -- must take it)
            ++*variants;
            const bool taught = small || scaled || b > 0;
            if (taught && amd::matrix_tile_count() == tiles_before && ++*never_tiled == 1)
              *first_slow = text + " cbsz " + std::to_string(a) + " blgp " + std::to_string(b) + " sel " + std::to_string(sel) + " (" + c.file + ")";
          }
    }
  }
  amd::set_matrix_fast_path(true);
  return compared;
}

}  // namespace

VTEST(the_modifier_bits_BLGP_CBSZ_and_the_scale_selectors_give_the_same_bits_on_both_paths) {
  std::string first;
  size_t wrong = 0;
  size_t variants = 0, never_tiled = 0;
  std::string slow;
  const size_t compared = compare_modifiers(2, &first, &wrong, &variants, &never_tiled, &slow);
  VCHECK(compared > 10000);
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(compared) + " differ; first: " + first);
  if (never_tiled) throw vtest::Failure(std::to_string(never_tiled) + " of " + std::to_string(variants) + " variants never took the tile path; first: " + slow);
}

VTEST(every_matrix_instruction_gives_the_same_bits_on_the_tile_path_as_on_the_reference_path) {
  std::string first;
  size_t wrong = 0;
  const size_t compared = compare_all(8, false, &first, &wrong);
  VCHECK(compared > 10000);
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(compared) + " differ; first: " + first);
}

VTEST(the_same_holds_with_the_registers_moved_over_one_another) {
  std::string first;
  size_t wrong = 0;
  const size_t compared = compare_all(8, true, &first, &wrong);
  VCHECK(compared > 10000);
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(compared) + " differ; first: " + first);
}

VTEST_MAIN
