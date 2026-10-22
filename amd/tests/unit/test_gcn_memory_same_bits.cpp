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
#include <iterator>
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

// The names an instruction of the corpora can be run under besides its own, for the families whose members take the same
// operands and differ in what the name says: the atomic operations, the byte and short loads and stores, the LDS accesses
// of each width. (The first few instructions of a family in a corpus are run under every name of the family.)
const char* const kFamilies[][24] = {
    {"s_atomic_swap", "s_atomic_cmpswap", "s_atomic_add", "s_atomic_sub", "s_atomic_smin", "s_atomic_umin", "s_atomic_smax", "s_atomic_umax",
     "s_atomic_and", "s_atomic_or", "s_atomic_xor", "s_atomic_inc", "s_atomic_dec", "s_atomic_add_x2", "s_atomic_cmpswap_x2", "s_atomic_umin_x2",
     "s_atomic_bogus", nullptr},
    {"buffer_atomic_add", "buffer_atomic_sub", "buffer_atomic_swap", "buffer_atomic_cmpswap", "buffer_atomic_cmpswap_x2", "buffer_atomic_add_f32",
     "buffer_atomic_pk_add_f16", "buffer_atomic_smin", "buffer_atomic_smax", "buffer_atomic_umin", "buffer_atomic_umax", "buffer_atomic_and",
     "buffer_atomic_or", "buffer_atomic_xor", "buffer_atomic_inc", "buffer_atomic_dec", "buffer_atomic_add_x2", "buffer_atomic_bogus", nullptr},
    {"global_atomic_add", "global_atomic_sub", "global_atomic_swap", "global_atomic_cmpswap", "global_atomic_smin", "global_atomic_umin",
     "global_atomic_smax", "global_atomic_umax", "global_atomic_and", "global_atomic_or", "global_atomic_xor", "global_atomic_inc",
     "global_atomic_dec", "global_atomic_add_f32", "global_atomic_pk_add_f16", "global_atomic_pk_add_bf16", "global_atomic_add_x2",
     "global_atomic_cmpswap_x2", "global_atomic_swap_x2", "global_atomic_add_f64", "global_atomic_min_f64", "global_atomic_max_f64", "global_atomic_bogus", nullptr},
    {"flat_atomic_add", "flat_atomic_swap", "flat_atomic_cmpswap", "flat_atomic_smin", "flat_atomic_umax", "flat_atomic_xor", "flat_atomic_inc",
     "flat_atomic_dec", "flat_atomic_add_f32", "flat_atomic_add_x2", "flat_atomic_cmpswap_x2", "flat_atomic_fmin", "flat_atomic_fmax_x2",
     "flat_atomic_min_f64", "flat_atomic_bogus", nullptr},
    {"global_load_ubyte", "global_load_sbyte", "global_load_ushort", "global_load_sshort", "global_load_dword", "global_load_ubyte_d16",
     "global_load_sbyte_d16_hi", "global_load_short_d16", nullptr},
    {"flat_load_ubyte", "flat_load_sbyte", "flat_load_ushort", "flat_load_sshort", "flat_load_dword", "flat_load_ubyte_d16_hi", nullptr},
    {"buffer_load_ubyte", "buffer_load_sbyte", "buffer_load_ushort", "buffer_load_sshort", "buffer_load_dword", "buffer_load_ubyte_d16",
     "buffer_load_sbyte_d16_hi", "buffer_load_short_d16", nullptr},
    {"scratch_load_ubyte", "scratch_load_sbyte", "scratch_load_ushort", "scratch_load_sshort", "scratch_load_dword", nullptr},
    {"global_store_byte", "global_store_short", "global_store_dword", "global_store_byte_d16_hi", "global_store_short_d16_hi", nullptr},
    {"flat_store_byte", "flat_store_short", "flat_store_dword", "flat_store_byte_d16_hi", nullptr},
    {"buffer_store_byte", "buffer_store_short", "buffer_store_dword", "buffer_store_byte_d16_hi", "buffer_store_short_d16_hi", nullptr},
    {"scratch_store_byte", "scratch_store_short", "scratch_store_dword", nullptr},
    {"ds_read_b32", "ds_read_u8", "ds_read_i8", "ds_read_u16", "ds_read_u8_d16", "ds_read_u8_d16_hi", "ds_read_i8_d16", "ds_read_u16_d16",
     "ds_read_u16_d16_hi", nullptr},
    {"ds_read_b64", "ds_read_b96", "ds_read_b128", "ds_read2_b64", "ds_read2st64_b64", nullptr},
    {"ds_write_b32", "ds_write_b8", "ds_write_b16", "ds_write_b8_d16_hi", "ds_write_b16_d16_hi", nullptr},
    {"ds_write_b64", "ds_write_b128", nullptr},
    {"ds_read2_b32", "ds_read2st64_b32", nullptr},
    {"ds_write2_b32", "ds_write2st64_b32", "ds_write2_b64", nullptr},
    {"ds_add_u32", "ds_sub_u32", "ds_and_b32", "ds_or_b32", "ds_max_i32", nullptr},
};

std::vector<amd::gcn::Inst> variants(const amd::gcn::Inst& in, std::vector<int>* seen) {
  std::vector<amd::gcn::Inst> out;
  for (size_t f = 0; f < std::size(kFamilies); ++f) {
    bool member = false;
    for (const char* const* n = kFamilies[f]; *n && !member; ++n) member = in.name == *n;
    if (!member) continue;
    if (const char* only = std::getenv("VGPU_MEM_SAME_BITS_FAMILY"); only && std::atoi(only) != static_cast<int>(f)) continue;
    if (seen->size() <= f) seen->resize(f + 1, 0);
    if ((*seen)[f]++ >= 3) break;
    for (const char* const* n = kFamilies[f]; *n; ++n) {
      if (in.name == *n) continue;
      amd::gcn::Inst copy = in;
      copy.name = *n;
      amd::gcn::finalize(copy);
      out.push_back(std::move(copy));
    }
    break;
  }
  return out;
}

// One digest per instruction of the corpus (0 where the decoder refuses it or it is not a memory instruction).
void digest_corpus(const Corpus& c, std::vector<uint32_t>* digests, std::vector<std::string>* texts) {
  std::vector<int> seen;   // how many of each family have been run under all its names
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
      if (const char* only = std::getenv("VGPU_MEM_SAME_BITS_ONLY"); only && in.name.find(only) == std::string::npos) {
        digests->push_back(0);
        texts->push_back(line.substr(tab + 1));
        continue;
      }
      if (std::getenv("VGPU_MEM_SAME_BITS_TRACE")) std::fprintf(stderr, "%s: %s\n", c.file, line.substr(tab + 1).c_str());
      if (is_memory(in) && in.name != "ds_nop" && in.name != "ds_consume" &&
          in.name != "ds_append" && in.name != "s_memtime" && in.name != "s_memrealtime") {   // (the last two read a clock)
        const uint64_t seed = digests->size();
        for (uint32_t mode = 0; mode < kModes; ++mode) d = d * 0x9E3779B97F4A7C15ull + amd::digest_memory_instruction(in, c.target, seed, mode);
        for (const amd::gcn::Inst& v : variants(in, &seen))
          for (uint32_t mode = 0; mode < kModes; ++mode) d = d * 0x9E3779B97F4A7C15ull + amd::digest_memory_instruction(v, c.target, seed, mode);
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
