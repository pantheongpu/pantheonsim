// Memory instructions, bit for bit against the build before their names were parsed once at decode time
// (Inst::mem, amd/src/gcn_decode.cpp mem_facts_of) instead of at every execution.
//
// Every memory instruction shape of the ISA corpora (DS, FLAT / global / scratch, MUBUF, MTBUF and scalar memory)
// is decoded and run on a wave whose address registers are aimed at three allocations in twelve ways (amd::digest_memory_instruction:
// aligned, misaligned, on the end of an allocation, past it, wild, in the shared and the private aperture, runs
// of consecutive lanes, the same address in every lane, mixed per lane; buffer resources valid, invalid, swizzled,
// too small; LDS accesses past the work-group's LDS; half the lanes off), and the digest of everything the
// instruction left behind (registers, LDS, scratch, the three allocations, the counts, or the text of the error it
// threw) is compared with amd/tests/data/mem_same_bits_digests.txt, which was recorded from the base build
// (VGPU_MEM_SAME_BITS_UPDATE=<file> writes it: only ever from the commit before the change under test).
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
    {"isa_corpus.txt", amd::gcn::Target::Gfx942},          {"isa_corpus_sopp_gfx942.txt", amd::gcn::Target::Gfx942},          {"isa_corpus_gfx950.txt", amd::gcn::Target::Gfx950},
    {"isa_corpus_gfx90a.txt", amd::gcn::Target::Gfx90a},  {"isa_corpus_gfx1100.txt", amd::gcn::Target::Gfx1100},
    {"isa_corpus_gfx1201.txt", amd::gcn::Target::Gfx1200}, {"isa_corpus_gfx1250.txt", amd::gcn::Target::Gfx1250},
    {"isa_corpus_gfx1030.txt", amd::gcn::Target::Gfx1030},
};
constexpr uint32_t kModes = 12;

std::vector<std::string> lines_of(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no fixture at " + path);
  std::vector<std::string> out;
  for (std::string line; std::getline(in, line);)
    if (!line.empty()) out.push_back(line);
  return out;
}

bool is_memory(const amd::gcn::Inst& in) {
  using amd::gcn::Enc;
  return in.enc == Enc::Ds || in.enc == Enc::Flat || in.enc == Enc::Mubuf || in.enc == Enc::Mtbuf || in.enc == Enc::Smem;
}

// One digest per instruction of the corpus (0 where the decoder refuses it or it is not a memory instruction).
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
    try {
      amd::gcn::Inst in = amd::gcn::decode(code, 0, 0, c.target);
      if (std::getenv("VGPU_MEM_SAME_BITS_TRACE")) std::fprintf(stderr, "%s: %s\n", c.file, line.substr(tab + 1).c_str());
      if (is_memory(in) && in.name != "ds_nop" && in.name != "ds_consume" &&
          in.name != "ds_append" && in.name != "s_memtime" && in.name != "s_memrealtime") {   // (the last two read a clock)
        const uint64_t seed = digests->size();
        for (uint32_t mode = 0; mode < kModes; ++mode) d = d * 0x9E3779B97F4A7C15ull + amd::digest_memory_instruction(in, c.target, seed, mode);
        d ^= d >> 32;
      }
    } catch (const std::exception&) {
    }
    digests->push_back(static_cast<uint32_t>(d));
    texts->push_back(line.substr(tab + 1));
  }
}

}  // namespace

VTEST(every_memory_instruction_of_the_isa_corpora_gives_the_recorded_bits) {
  const std::string golden_path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/mem_same_bits_digests.txt";
  if (const char* update = std::getenv("VGPU_MEM_SAME_BITS_UPDATE"); update && *update) {
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
    std::fprintf(stderr, "memory same-bits digests written to %s\n", update);
    return;
  }
  const std::vector<std::string> golden = lines_of(golden_path);
  size_t at = 0, checked = 0, wrong_total = 0, nonzero = 0;
  std::string report;
  for (const Corpus& c : kCorpora) {
    if (at >= golden.size() || golden[at].rfind(std::string("# ") + c.file, 0) != 0)
      throw vtest::Failure(std::string("mem_same_bits_digests.txt has no section for ") + c.file);
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
      if (golden[at] != buf && ++wrong <= (std::getenv("VGPU_SAME_BITS_ALL") ? 100000u : 8u))
        report += std::string("\n  ") + c.file + ": " + texts[i] + ": got " + buf + ", recorded " + golden[at];
    }
    wrong_total += wrong;
  }
  VCHECK(nonzero > 1000);   // the memory shapes ran (or threw a definite error)
  if (wrong_total)
    throw vtest::Failure(std::to_string(wrong_total) + " of " + std::to_string(checked) + " instructions differ from the recorded bits:" + report);
}

VTEST_MAIN
