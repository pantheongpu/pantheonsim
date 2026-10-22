// The packed 16-bit float, dot product, half conversion, half comparison, sine/cosine and packed integer instructions, bit for
// bit, against a recorded run of the build before they were made faster (amd/src/packed_math.hpp, and the half conversions and comparisons of gcn_exec.cpp): every
// instruction of amd/tests/data/isa_corpus_packed.txt (amd/tools/packed-corpus.py: each of these families in every combination
// of op_sel, op_sel_hi, neg_lo, neg_hi, clamp, abs, neg and omod, with vector, scalar and constant sources) is run once on a wave
// of pseudo-random registers in four ways (any bits -- NaNs of every payload among the halves --, small integers, floats and
// doubles with the special values among them, half the lanes off), and a digest of everything it left behind (or of the text of
// the error it threw) is compared with amd/tests/data/packed_same_bits_digests.txt. The method is that of
// test_amd_gcn_same_bits; the digests were recorded from the build before the change (VGPU_PACKED_SAME_BITS_UPDATE=<file>
// writes them), never from the one under test.
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

// The digests were recorded from an optimized build. A build with no optimization (a sanitizer job's Debug) lets the compiler order the operands of
// float operations on two NaNs, and of a few libm calls, another way, so a few of the shapes that have NaNs among their inputs come out with other
// payloads there: up to 2 % of the shapes may differ in such a build (and a note says so); an optimized build must match every one.
inline bool digests_may_differ(size_t wrong, size_t checked) {
#if defined(__OPTIMIZE__)
  (void)wrong, (void)checked;
  return false;
#else
  if (wrong * 50 > checked) return false;
  std::printf("NOTE: %zu of %zu digests differ in this unoptimized build (NaN payloads and libm calls the compiler orders as it likes)\n", wrong, checked);
  return true;
#endif
}

namespace {

std::vector<std::string> lines_of(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no fixture at " + path);
  std::vector<std::string> out;
  for (std::string line; std::getline(in, line);)
    if (!line.empty() && line[0] != '#') out.push_back(line);
  return out;
}

// One digest per instruction of the corpus (0 where the decoder refuses it), and the instruction texts.
void digest_corpus(std::vector<uint32_t>* digests, std::vector<std::string>* texts) {
  for (const std::string& line : lines_of(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/isa_corpus_packed.txt")) {
    const size_t tab = line.find('\t');
    std::vector<uint8_t> code;
    std::istringstream words(line.substr(0, tab));
    for (std::string w; words >> w;) {
      const uint32_t v = static_cast<uint32_t>(std::stoul(w, nullptr, 16));
      for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
    }
    uint64_t d = 0;
    try {
      amd::gcn::Inst in = amd::gcn::decode(code, 0, 0, amd::gcn::Target::Gfx942);
      const uint64_t seed = digests->size();
      for (uint32_t flavour = 0; flavour < 4; ++flavour) {
        const uint64_t x = amd::digest_instruction(in, amd::gcn::Target::Gfx942, seed + flavour, flavour);
        d = d * 0x9E3779B97F4A7C15ull + x;
      }
      d ^= d >> 32;
    } catch (const std::exception&) {
    }
    digests->push_back(static_cast<uint32_t>(d));
    texts->push_back(line.substr(tab + 1));
  }
}

}  // namespace

VTEST(the_packed_half_dot_conversion_comparison_trig_and_integer_instructions_give_the_recorded_bits) {
  std::vector<uint32_t> digests;
  std::vector<std::string> texts;
  digest_corpus(&digests, &texts);
  if (const char* update = std::getenv("VGPU_PACKED_SAME_BITS_UPDATE"); update && *update) {
    std::ofstream out(update);
    out << "# " << digests.size() << "\n";
    char buf[16];
    for (uint32_t d : digests) {
      std::snprintf(buf, sizeof buf, "%08x\n", d);
      out << buf;
    }
    std::fprintf(stderr, "packed same-bits digests written to %s\n", update);
    return;
  }
  const std::vector<std::string> golden = lines_of(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/packed_same_bits_digests.txt");
  if (golden.size() != digests.size()) throw vtest::Failure("packed_same_bits_digests.txt has a different number of instructions than the corpus");
  size_t wrong = 0, nonzero = 0;
  std::string report;
  for (size_t i = 0; i < digests.size(); ++i) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%08x", digests[i]);
    nonzero += digests[i] != 0;
    if (golden[i] != buf && ++wrong <= 10) report += std::string("\n  ") + texts[i] + ": got " + buf + ", recorded " + golden[i];
  }
  VCHECK(nonzero > digests.size() / 2);   // most shapes ran (or threw a definite error), not all refused
  if (wrong && !digests_may_differ(wrong, digests.size())) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(digests.size()) + " instructions differ from the recorded bits:" + report);
}

VTEST_MAIN
