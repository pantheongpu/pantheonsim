// The interpreter's results, bit for bit, instruction by instruction, against a recorded run of the
// build before the dispatch of its instructions was rewritten (amd/tools/ladder-to-switch.py, decode-time
// flags, hoisted lane-invariant operands): every instruction shape of the ISA corpora
// (amd/tests/data/isa_corpus*.txt, the ones test_gcn.cpp checks against llvm-objdump) is decoded and run
// once on a wave of pseudo-random registers, LDS and memory in four ways (any bits, small integers,
// floats with the special values among them, half the lanes off), and a digest of everything it left
// behind (registers, masks, LDS, memory, the instruction counts, or the text of the error it threw) is
// compared with amd/tests/data/same_bits_digests.txt.
//
// The digests were recorded from the baseline build (VGPU_SAME_BITS_UPDATE=<file> writes them). They
// depend on the host's libm for the transcendental instructions; if a different libm makes them differ
// on the instructions whose names say sin, cos, exp, log, rcp, rsq or sqrt, record them again from the
// commit before the change under test, never from the one being tested.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
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
    {"isa_corpus.txt", amd::gcn::Target::Gfx942},          {"isa_corpus_sopp_gfx942.txt", amd::gcn::Target::Gfx942},          {"isa_corpus_gfx950.txt", amd::gcn::Target::Gfx950},
    {"isa_corpus_gfx90a.txt", amd::gcn::Target::Gfx90a},  {"isa_corpus_gfx1100.txt", amd::gcn::Target::Gfx1100},
    {"isa_corpus_gfx1201.txt", amd::gcn::Target::Gfx1200}, {"isa_corpus_gfx1250.txt", amd::gcn::Target::Gfx1250},
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

// Shapes the gfx1250 fuzz corpus holds that no assembler would write, which the digest does not cover: on the base
// commit (f81cc0c1) they took the probe down (the executor read the address operand of an LDS instruction that has
// none, off the end of an empty vector), so no base digest exists to record. They run, and are checked, in
// ds_instructions_without_an_address_fail_gracefully below.
bool skipped(const amd::gcn::Inst& in) {
  return in.name == "ds_nop" || in.name == "ds_consume" || in.name == "ds_append";
}

// One digest per instruction of the corpus (0 where the decoder refuses it), and the instruction texts.
void digest_corpus(const Corpus& c, std::vector<uint32_t>* digests, std::vector<std::string>* texts) {
  for (const std::string& line : lines_of(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + c.file)) {
    if (line[0] == '#') continue;
    const size_t tab = line.find('\t');
    std::vector<uint8_t> code;
    std::istringstream words(line.substr(0, tab));
    for (std::string w; words >> w;) {
      const uint32_t v = static_cast<uint32_t>(std::stoul(w, nullptr, 16));
      for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
    }
    uint64_t d = 0;
    std::string text = line.substr(tab + 1);
    if (const char* only = std::getenv("VGPU_SAME_BITS_ONLY"); only && text.find(only) == std::string::npos) {
      digests->push_back(0);
      texts->push_back(text);
      continue;
    }
    if (std::getenv("VGPU_SAME_BITS_TRACE")) std::fprintf(stderr, "%s: %s\n", c.file, text.c_str());
    try {
      amd::gcn::Inst in = amd::gcn::decode(code, 0, 0, c.target);
      // (A branch the corpus holds goes where it was: send it elsewhere, so that taking it shows.)
      if (in.name.rfind("s_branch", 0) == 0 || in.name.rfind("s_cbranch", 0) == 0) in.target = 0x1000;
      if (skipped(in)) {
        digests->push_back(0);
        texts->push_back(text);
        continue;
      }
      const uint64_t seed = digests->size();
      // (An instruction that reads the clock gives a different digest every time: recorded as such, not compared.)
      bool stable = true;
      for (uint32_t flavour = 0; flavour < 4; ++flavour) {
        // (An LDS access with a wild address is a crash this probe must not meet: small integers only.)
        const uint32_t f = in.enc == amd::gcn::Enc::Ds ? 1 : flavour;
        const uint64_t x = amd::digest_instruction(in, c.target, seed + flavour, f);
        stable = stable && x == amd::digest_instruction(in, c.target, seed + flavour, f);
        d = d * 0x9E3779B97F4A7C15ull + x;
      }
      d ^= d >> 32;
      if (!stable) d = 0xFFFFFFFFu;
    } catch (const std::exception& e) {
      if (std::getenv("VGPU_SAME_BITS_TRACE")) std::fprintf(stderr, "   refused: %s\n", e.what());
    }
    digests->push_back(static_cast<uint32_t>(d));
    texts->push_back(text);
  }
}

}  // namespace

VTEST(every_instruction_of_the_isa_corpora_gives_the_recorded_bits) {
  const std::string golden_path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/same_bits_digests.txt";
  if (const char* update = std::getenv("VGPU_SAME_BITS_UPDATE"); update && *update) {
    std::ofstream out(update);
    for (const Corpus& c : kCorpora) {
      std::vector<uint32_t> digests;
      std::vector<std::string> texts;
      digest_corpus(c, &digests, &texts);
      out << "# " << c.file << " " << digests.size() << "\n";
      char buf[16];
      for (uint32_t d : digests) {
        std::snprintf(buf, sizeof buf, "%08x\n", d);
        out << buf;
      }
    }
    std::fprintf(stderr, "same-bits digests written to %s\n", update);
    return;
  }
  const std::vector<std::string> golden = lines_of(golden_path);
  size_t at = 0, checked = 0, wrong_total = 0, nonzero = 0;
  std::string report;
  for (const Corpus& c : kCorpora) {
    if (at >= golden.size() || golden[at].rfind(std::string("# ") + c.file, 0) != 0)
      throw vtest::Failure(std::string("same_bits_digests.txt has no section for ") + c.file);
    const size_t n = std::stoul(golden[at].substr(golden[at].rfind(' ') + 1));
    ++at;
    std::vector<uint32_t> digests;
    std::vector<std::string> texts;
    digest_corpus(c, &digests, &texts);
    if (digests.size() != n) throw vtest::Failure(std::string(c.file) + " has a different number of instructions than the digests");
    size_t wrong = 0;
    for (size_t i = 0; i < n; ++i, ++at) {
      char buf[16];
      std::snprintf(buf, sizeof buf, "%08x", digests[i]);
      ++checked;
      nonzero += digests[i] != 0;
      if (golden[at] != buf && golden[at] != "ffffffff" && digests[i] != 0xFFFFFFFFu && ++wrong <= (std::getenv("VGPU_SAME_BITS_ALL") ? 100000u : 8u)) report += std::string("\n  ") + c.file + ": " + texts[i] + ": got " + buf + ", recorded " + golden[at];
    }
    wrong_total += wrong;
  }
  VCHECK(nonzero > 10000);   // most shapes ran (or threw a definite error), not all refused
  if (wrong_total)
    throw vtest::Failure(std::to_string(wrong_total) + " of " + std::to_string(checked) + " instructions differ from the recorded bits:" + report);
}

// ds_nop does nothing; ds_append and ds_consume (the wave-wide counters) are not modelled and say so. Neither may read an
// address operand the decoder did not give them (a segfault in the build this test was added to).
VTEST(ds_instructions_without_an_address_fail_gracefully) {
  struct Case {
    const char* name;
    uint32_t word0;
    bool runs;
  };
  const Case cases[] = {{"ds_nop", 0xD8500000u, true}, {"ds_consume", 0xD8F40000u, false}, {"ds_append", 0xD8F80000u, false}};
  const amd::gcn::Target targets[] = {amd::gcn::Target::Gfx1030, amd::gcn::Target::Gfx1100, amd::gcn::Target::Gfx1200,
                                      amd::gcn::Target::Gfx1250};
  size_t checked = 0;
  for (amd::gcn::Target t : targets)
    for (const Case& c : cases)
      for (uint32_t vdst : {0u, 5u, 0xC9u}) {
        const uint32_t word1 = c.runs ? 0u : vdst << 24;
        std::vector<uint8_t> code;
        for (uint32_t v : {c.word0, word1})
          for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
        amd::gcn::Inst in;
        try {
          in = amd::gcn::decode(code, 0, 0, t);
        } catch (const std::exception&) {
          continue;   // (an architecture without the instruction)
        }
        if (in.name != c.name) continue;
        for (uint32_t flavour = 0; flavour < 4; ++flavour) {
          std::string error;
          amd::digest_instruction(in, t, 7 + flavour, flavour, &error);
          if (c.runs) VCHECK(error.empty());
          else VCHECK(error.find("not modelled") != std::string::npos);
          ++checked;
        }
      }
  VCHECK(checked >= 36);   // (the three instructions, three registers, four flavours, on at least one architecture)
}

VTEST_MAIN
