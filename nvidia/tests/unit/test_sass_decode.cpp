// The SASS decoder against NVIDIA's own disassembler. Each corpus line
// (nvidia/tools/sass-corpus.py) is one instruction's 128 bits, its address,
// and nvdisasm's text for it; each must decode and print exactly as nvdisasm
// printed it. Every mismatch is counted by opcode and the first few listed.
//
// VGPU_SASS_REPORT=1 prints the per-opcode counts even when all is well;
// VGPU_SASS_CORPUS=<file> checks one file instead (a full local corpus);
// VGPU_SASS_OP=<opcode> lists every mismatch of that opcode.
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

struct Tally {
  size_t ok = 0, wrong = 0, unknown = 0;
  std::string first;   // one example of a mismatch
};

std::string opcode_of(const std::string& text) {
  std::string t = text;
  if (!t.empty() && t[0] == '@') t = t.substr(t.find(' ') + 1);
  return t.substr(0, t.find_first_of(" ."));
}

void check_corpus(const std::string& file, int sm) {
  const char* over = std::getenv("VGPU_SASS_CORPUS");
  const std::string path =
      over && *over ? std::string(over) : std::string(VGPU_SOURCE_DIR) + "/nvidia/tests/data/sass/" + file;
  if (over && *over) {
    // The architecture from the file's name: .../sm_90a.txt is sm_90.
    const size_t at = path.rfind("sm_");
    if (at != std::string::npos) sm = std::atoi(path.c_str() + at + 3);
  }
  std::ifstream in(path);
  if (!in) throw vtest::Failure("no corpus at " + path);
  std::map<std::string, Tally> by_op;
  size_t checked = 0, bad = 0;
  std::string listed;
  int shown = 0;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    const size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1);
    const std::string hexw = line.substr(0, t1);
    sass::Word w;
    w.hi = std::stoull(hexw.substr(0, 16), nullptr, 16);
    w.lo = std::stoull(hexw.substr(16, 16), nullptr, 16);
    const uint64_t pc = std::stoull(line.substr(t1 + 1, t2 - t1 - 1), nullptr, 16);
    const std::string want = line.substr(t2 + 1);
    Tally& t = by_op[opcode_of(want)];
    ++checked;
    std::string got;
    bool unknown = false;
    try {
      got = sass::to_text(sass::decode(w, pc, sm));
    } catch (const std::exception& e) {
      got = std::string("(refused: ") + e.what() + ")";
      unknown = true;
    }
    if (got == want) {
      ++t.ok;
      continue;
    }
    ++bad;
    (unknown ? t.unknown : t.wrong)++;
    if (t.first.empty()) t.first = "\n    " + hexw + "\n    want \"" + want + "\"\n    got  \"" + got + "\"";
    const char* only = std::getenv("VGPU_SASS_OP");
    const bool pick = only && *only && opcode_of(want) == only;
    if ((pick || (!only && !unknown)) && shown++ < (pick ? 60 : 25))
      listed += "\n  " + hexw + "\n  want \"" + want + "\"\n  got  \"" + got + "\"";
  }
  const bool report = std::getenv("VGPU_SASS_REPORT") != nullptr;
  if (report || bad) {
    std::string table;
    for (const auto& [op, t] : by_op) {
      if (!report && t.wrong + t.unknown == 0) continue;
      table += "\n  " + op + ": " + std::to_string(t.ok) + " ok, " + std::to_string(t.wrong) + " wrong, " +
               std::to_string(t.unknown) + " unknown" + (t.wrong ? t.first : "");
    }
    std::fprintf(stderr, "%s (sm_%d): %zu of %zu match%s\n", path.substr(path.rfind('/') + 1).c_str(), sm, checked - bad, checked,
                 table.c_str());
  }
  VCHECK(checked > 1000);
  if (bad)
    throw vtest::Failure(std::to_string(bad) + " of " + std::to_string(checked) + " differ" + listed);
}

}  // namespace

VTEST(sm_86_decodes_as_nvdisasm_prints_it) { check_corpus("sm_86.txt", 86); }

VTEST(fields_straddle_the_two_words) {
  sass::Word w;
  w.lo = 0xf000000000000000ull;
  w.hi = 0x5;
  VCHECK_EQ(w.field(60, 8), 0x5full);
  VCHECK_EQ(w.sfield(60, 8), static_cast<int64_t>(0x5f));
  VCHECK_EQ(w.sfield(64, 3), -3);
}

VTEST_MAIN
