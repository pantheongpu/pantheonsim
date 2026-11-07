// What the AMD tier saves on the LDS instructions with two offsets: every instruction of a tier corpus whose text contains the filter
// is run `iters` times on a wave aimed at the LDS (amd::time_instruction, flavour 18 consecutive words or 27 eight bytes apart), once with the tier
// off (the interpreter) and once on, and the two mean times are printed.
//
//   amd_lds_bench <amd/tests/data> [filter] [iters] [flavour]
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
  const std::string dir = argv[1], filter = argc > 2 ? argv[2] : "ds_read2_b64";
  const uint32_t iters = argc > 3 ? static_cast<uint32_t>(std::atoi(argv[3])) : 200000;
  const uint32_t flavour = argc > 4 ? static_cast<uint32_t>(std::atoi(argv[4])) : 27;
  std::ifstream f(dir + "/isa_corpus_tier_gfx942.txt");
  for (std::string line; std::getline(f, line);) {
    const size_t tab = line.find('\t');
    if (line.empty() || tab == std::string::npos) continue;
    const std::string text = line.substr(tab + 1);
    if (text.find(filter) == std::string::npos) continue;
    std::vector<uint8_t> code;
    std::istringstream words(line.substr(0, tab));
    for (std::string w; words >> w;) {
      const uint32_t v = static_cast<uint32_t>(std::stoul(w, nullptr, 16));
      for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
    }
    try {
      const amd::gcn::Inst in = amd::gcn::decode(code, 0, 0, amd::gcn::Target::Gfx942);
      amd::gcn::tier::set_enabled(false);
      const double off = amd::time_instruction(in, amd::gcn::Target::Gfx942, 1, flavour, iters);
      amd::gcn::tier::set_enabled(true);
      const double on = amd::time_instruction(in, amd::gcn::Target::Gfx942, 1, flavour, iters);
      std::printf("%9.1f ns interpreter %9.1f ns tier  x%-6.2f %s\n", off, on, off / on, text.c_str());
    } catch (const std::exception& e) {
      std::fprintf(stderr, "skip %s: %s\n", text.c_str(), e.what());
    }
  }
}
