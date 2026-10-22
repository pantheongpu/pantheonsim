// What one matrix instruction costs the interpreter: every instruction of the ISA corpora whose text contains the
// filter (default: the MFMA, SMFMAC, WMMA and SWMMAC forms) is run `iters` times on a wave of pseudo-random registers
// and its mean time reported, slowest first.
//
//   amd_matrix_bench <corpus dir> [filter] [iters] [flavour]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_gcn.hpp"

using namespace vgpu;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <amd/tests/data> [filter] [iters] [flavour]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1], filter = argc > 2 ? argv[2] : "";
  const uint32_t iters = argc > 3 ? static_cast<uint32_t>(std::atoi(argv[3])) : 2000;
  const uint32_t flavour = argc > 4 ? static_cast<uint32_t>(std::atoi(argv[4])) : 2;
  struct C {
    const char* file;
    amd::gcn::Target t;
  };
  const C corpora[] = {{"isa_corpus.txt", amd::gcn::Target::Gfx942},       {"isa_corpus_gfx950.txt", amd::gcn::Target::Gfx950},
                       {"isa_corpus_gfx90a.txt", amd::gcn::Target::Gfx90a}, {"isa_corpus_gfx1100.txt", amd::gcn::Target::Gfx1100},
                       {"isa_corpus_gfx1201.txt", amd::gcn::Target::Gfx1200}, {"isa_corpus_gfx1250.txt", amd::gcn::Target::Gfx1250},   // (slow: BENCH_1250=1)
                       {"isa_corpus_matrix_gfx942.txt", amd::gcn::Target::Gfx942}, {"isa_corpus_matrix_gfx950.txt", amd::gcn::Target::Gfx950},
                       {"isa_corpus_matrix_gfx1100.txt", amd::gcn::Target::Gfx1100}, {"isa_corpus_matrix_gfx1201.txt", amd::gcn::Target::Gfx1200},
                       {"isa_corpus_matrix_gfx1250.txt", amd::gcn::Target::Gfx1250}};
  struct Row {
    std::string text, file;
    double ns;
  };
  std::vector<Row> rows;
  for (const C& c : corpora) {
    if (c.t == amd::gcn::Target::Gfx1250 && !std::getenv("BENCH_1250")) continue;
    std::ifstream f(dir + "/" + c.file);
    for (std::string line; std::getline(f, line);) {
      if (line.empty() || line[0] == '#') continue;
      const size_t tab = line.find('\t');
      const std::string text = line.substr(tab + 1);
      if (filter.empty() ? !(text.find("mfma") != std::string::npos || text.find("mmac") != std::string::npos || text.find("wmma") != std::string::npos)
                         : text.find(filter) == std::string::npos)
        continue;
      std::vector<uint8_t> code;
      std::istringstream words(line.substr(0, tab));
      for (std::string w; words >> w;) {
        const uint32_t v = static_cast<uint32_t>(std::stoul(w, nullptr, 16));
        for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
      }
      try {
        const amd::gcn::Inst in = amd::gcn::decode(code, 0, 0, c.t);
        rows.push_back({text, c.file, amd::time_instruction(in, c.t, 1, flavour, iters)});
      } catch (const std::exception& e) {
        std::fprintf(stderr, "skip %s: %s\n", text.c_str(), e.what());
      }
    }
  }
  std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.ns > b.ns; });
  for (const Row& r : rows) std::printf("%10.0f ns  %-24s %s\n", r.ns, r.file.c_str(), r.text.c_str());
}
