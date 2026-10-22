// A scratch access whose instruction offset is negative and whose register offsets are zero (`scratch_store_b128 v9, v[1:4], off offset:-16`
// with v9 = 0). The offset is unsigned inside the interpreter, so the range check `offset + bytes > per_lane` wrapped past 2^64 for such an
// access and passed it: the access landed before the work-item's scratch, on the heap (a crash of the memory digest probe, and a write
// outside the allocation under AddressSanitizer). It is now refused as every other access past the work-item's scratch is, with the same
// error. Shapes of the ISA corpora of gfx942, gfx1030, gfx1100 and gfx1250 with their instruction offset set to minus the access's size and
// every register zero (digest flavour 16).
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_gcn.hpp"
#include "vtest.hpp"

namespace {

// The bytes a scratch instruction moves, from its name; 0 for a name this does not know.
uint32_t bytes_of(const std::string& n) {
  static const std::pair<const char*, uint32_t> kSuffix[] = {{"dwordx4", 16}, {"dwordx3", 12}, {"dwordx2", 8}, {"dword", 4}, {"b128", 16},
                                                              {"b96", 12},    {"b64", 8},      {"b32", 4},     {"b16", 2},   {"b8", 1},
                                                              {"ubyte", 1},   {"sbyte", 1},    {"byte", 1},    {"ushort", 2}, {"sshort", 2},
                                                              {"short", 2},   {"u8", 1},       {"i8", 1},      {"u16", 2},   {"i16", 2}};
  for (const auto& [s, bytes] : kSuffix)
    if (n.size() > std::string(s).size() && n.compare(n.size() - std::string(s).size(), std::string(s).size(), s) == 0) return bytes;
  return 0;
}

struct Source {
  const char* file;
  vgpu::amd::gcn::Target target;
};

}  // namespace

VTEST(a_scratch_access_at_a_negative_offset_from_a_zero_register_is_refused_not_wrapped) {
  using namespace vgpu;
  const Source sources[] = {{"isa_corpus.txt", amd::gcn::Target::Gfx942},     {"isa_corpus_gfx1030.txt", amd::gcn::Target::Gfx1030},
                            {"isa_corpus_gfx1100.txt", amd::gcn::Target::Gfx1100}, {"isa_corpus_gfx1250.txt", amd::gcn::Target::Gfx1250}};
  for (const Source& src : sources) {
    std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + src.file);
    if (!in) throw vtest::Failure(std::string("no corpus ") + src.file);
    size_t checked = 0, stores = 0, loads = 0;
    for (std::string line; std::getline(in, line);) {
      if (line.empty() || line[0] == '#') continue;
      const size_t tab = line.find('\t');
      std::vector<uint8_t> code;
      std::istringstream words(line.substr(0, tab));
      for (std::string w; words >> w;) {
        const uint32_t v = static_cast<uint32_t>(std::stoul(w, nullptr, 16));
        for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
      }
      // gfx1030's corpus has no scratch instruction: its global loads and stores with the SEG field (bits 15:14 of the first word) set to scratch are.
      if (src.target == amd::gcn::Target::Gfx1030 && (line.find("\tglobal_store_dword") != std::string::npos || line.find("\tglobal_load_dword") != std::string::npos))
        code[1] = static_cast<uint8_t>((code[1] & 0x3F) | 0x40);
      amd::gcn::Inst inst;
      try {
        inst = amd::gcn::decode(code, 0, 0, src.target);
      } catch (const std::exception&) {
        continue;
      }
      if (inst.name.rfind("scratch_", 0) != 0 || inst.name.find("block") != std::string::npos || inst.name.find("lds") != std::string::npos ||
          inst.name.find("d16") != std::string::npos)
        continue;
      const uint32_t bytes = bytes_of(inst.name);
      const bool is_store = inst.name.rfind("scratch_store", 0) == 0;
      if (!bytes || bytes > 16 || (is_store ? stores : loads) >= 12) continue;
      inst.offset = -static_cast<int32_t>(bytes);
      std::string error;
      amd::digest_instruction(inst, src.target, 11, 16, &error);
      if (error.find("scalar register") != std::string::npos) continue;   // (a gfx1250 shape whose scalar offset names a register past the file: refused first, for that)
      if (error.find("a scratch access at") == std::string::npos || error.find("bytes a work-item reserved") == std::string::npos)
        throw vtest::Failure(std::string(src.file) + ": " + inst.name + " at offset -" + std::to_string(bytes) + " gave [" + error + "]");
      ++checked;
      (is_store ? stores : loads)++;
    }
    std::printf("%s: %zu scratch accesses refused (%zu stores, %zu loads)\n", src.file, checked, stores, loads);
    VCHECK(stores >= 3);
    VCHECK(loads >= 3);
  }
}

VTEST_MAIN
