// The instruction forms the AMD probe workloads (tools/bench-amd/kernels/bench_amd_probe.hip) found the interpreter lacking, run on one wave each:
// RDNA2's VOP3 spellings of v_ashrrev_i16, v_max_i16 and v_min_i16 (decoded as ..._e64) must run, not stop at "decoded but not implemented", and
// give what the 16-bit arithmetic of the operands' low halves says. The tier is on and off: the same lanes either way.
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_gcn.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {
struct Form {
  const char* text;
  uint32_t words[2];
  size_t n;
};
std::vector<uint8_t> bytes_of(const std::string& hex_words) {
  std::vector<uint8_t> code;
  std::istringstream words(hex_words);
  for (std::string w; words >> w;) {
    const uint32_t v = static_cast<uint32_t>(std::stoul(w, nullptr, 16));
    for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
  }
  return code;
}
}  // namespace

VTEST(rdna2_vop3_sixteen_bit_integer_forms_run) {
  // (llvm-mc encodings, gfx1030: v_ashrrev_i16 v1, v2, v3 / v_max_i16 v1, v2, v3 / v_min_i16 v1, v2, v3 / v_ashrrev_i16 v1, 3, v3)
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/isa_corpus_tier_gfx1030.txt", std::ios::binary);
  VCHECK(in.good());
  size_t checked = 0;
  for (std::string line; std::getline(in, line);) {
    const size_t tab = line.find('\t');
    if (line.empty() || line[0] == '#' || tab == std::string::npos) continue;
    const std::string text = line.substr(tab + 1);
    if (text.rfind("v_ashrrev_i16", 0) != 0 && text.rfind("v_max_i16", 0) != 0 && text.rfind("v_min_i16", 0) != 0) continue;
    const amd::gcn::Inst inst = amd::gcn::decode(bytes_of(line.substr(0, tab)), 0, 0, amd::gcn::Target::Gfx1030);
    for (uint32_t flavour : {0u, 1u, 3u}) {
      std::string error_off, error_on;
      amd::gcn::tier::set_enabled(false);
      const uint64_t off = amd::digest_instruction(inst, amd::gcn::Target::Gfx1030, 7, flavour, &error_off);
      amd::gcn::tier::set_enabled(true);
      const uint64_t on = amd::digest_instruction(inst, amd::gcn::Target::Gfx1030, 7, flavour, &error_on);
      VCHECK(error_off.empty());
      VCHECK(error_on.empty());
      VCHECK(off == on);
    }
    ++checked;
  }
  VCHECK(checked >= 6);
}

VTEST_MAIN
