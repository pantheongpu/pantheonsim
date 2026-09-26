// Decoding CDNA machine code: every instruction of the fixture's two kernels,
// compared with what the assembler wrote.
//
// amd/tests/data/vector_add.gfx942.dis is that code as llvm-objdump prints
// it, written beside the object by amd/tests/data/build.sh. Decoding the
// object has to produce the same instructions, in the same order, with the
// same operands -- which catches an opcode read from the wrong bits, an
// operand of the wrong width, and an instruction whose length is wrong (that
// one shifts every instruction after it).
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_decode_cache.hpp"
#include "vgpu/amd_gcn.hpp"
#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

// The fixture, or another object and listing: VGPU_GCN_OBJECT and
// VGPU_GCN_LISTING point these checks at a code object built here and now
// (amd/tests/e2e/run_gcn_disasm.sh), so the decoder is checked against the
// assembler's own output rather than only against what was checked in.
std::string read(const std::string& name, bool allow_override = true) {
  const char* over =
      allow_override ? std::getenv(name.find(".dis") != std::string::npos ? "VGPU_GCN_LISTING" : "VGPU_GCN_OBJECT")
                     : nullptr;
  const std::string path = over && *over ? std::string(over) : std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + name;
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no fixture at " + path);
  return std::string((std::istreambuf_iterator<char>(in)), {});
}

// The fixture itself: the checks below name its instructions, so they read it
// whatever object the listing comparison was pointed at.
amd::CodeObject object() {
  return amd::load_code_object(read("vector_add.gfx942.o", false), "vector_add.gfx942.o");
}

// Every instruction in the object's .text, in order, as this decodes them --
// the kernels and the padding the assembler puts between them, which is what
// the listing covers too.
std::vector<std::string> decoded(const amd::CodeObject& o) {
  std::vector<std::string> out;
  for (uint64_t at = 0; at < o.text.size();) {
    const amd::gcn::Inst in = amd::gcn::decode(o.text, at, o.text_addr + at);
    out.push_back(amd::gcn::to_text(in));
    at += in.size;
  }
  return out;
}

std::vector<std::string> lines(const std::string& text) {
  std::vector<std::string> out;
  std::istringstream in(text);
  for (std::string line; std::getline(in, line);)
    if (!line.empty()) out.push_back(line);
  return out;
}

}  // namespace

// One object against its listing, instruction by instruction.
void check_against_assembler(const std::string& fixture, const std::string& listing_name) {
  // Named in what it reports as the object it is: the fixture, or the one
  // VGPU_GCN_OBJECT points at.
  const char* over = std::getenv("VGPU_GCN_OBJECT");
  const std::string object_name = over && *over ? std::string(over) : fixture;
  const amd::CodeObject o = amd::load_code_object(read(fixture), object_name);
  const std::vector<std::string> mine = decoded(o), theirs = lines(read(listing_name));
  if (theirs.size() < 20) throw vtest::Failure(listing_name + " has too few instructions to be the listing");
  if (mine.size() != theirs.size())
    throw vtest::Failure(object_name + ": decoded " + std::to_string(mine.size()) + " instructions, the assembler " +
                         "wrote " + std::to_string(theirs.size()));
  for (size_t i = 0; i < mine.size(); ++i)
    if (mine[i] != theirs[i])
      throw vtest::Failure(object_name + ", instruction " + std::to_string(i) + ": decoded \"" + mine[i] +
                           "\", assembler wrote \"" + theirs[i] + "\"");
}

VTEST(every_instruction_decodes_as_the_assembler_wrote_it) {
  check_against_assembler("vector_add.gfx942.o", "vector_add.gfx942.dis");
  // The other fixtures exercise what a real kernel uses: integer and float
  // math, a division, a loop and an atomic in one, and doubles, packed
  // halves, the modifiers and the transcendentals in the other.
  if (!std::getenv("VGPU_GCN_OBJECT")) {
    check_against_assembler("ops.gfx942.o", "ops.gfx942.dis");
    check_against_assembler("math.gfx942.o", "math.gfx942.dis");
    check_against_assembler("memory.gfx942.o", "memory.gfx942.dis");
    check_against_assembler("globals.gfx942.o", "globals.gfx942.dis");
    check_against_assembler("grid.gfx942.o", "grid.gfx942.dis");
    // What PyTorch's ROCm libraries use beyond a compiler's usual output.
    check_against_assembler("asm_libs.gfx942.o", "asm_libs.gfx942.dis");
    // And the hand-written kernels the executor's tests run, instruction by
    // instruction, so a decoder change that misreads one shows up here too.
    for (const char* name : {"asm_sopk", "asm_scalar", "asm_memory", "asm_vector"})
      check_against_assembler(std::string(name) + ".gfx942.o", std::string(name) + ".gfx942.dis");
  }
}

// The corpus drawn from ROCm's libraries (amd/tools/isa-corpus.py): one of
// every shape of instruction their gfx942 code holds, each decoded from its
// own bytes and printed as llvm-objdump printed it. Every mismatch is listed.
VTEST(every_instruction_shape_rocms_libraries_use_decodes_as_llvm_prints_it) {
  const std::vector<std::string> corpus = lines(read("isa_corpus.txt", false));
  size_t checked = 0;
  std::string wrong;
  int wrong_count = 0;
  for (const std::string& line : corpus) {
    if (line[0] == '#') continue;
    const size_t tab = line.find('\t');
    std::vector<uint8_t> code;
    std::istringstream words(line.substr(0, tab));
    for (std::string w; words >> w;) {
      const uint32_t v = static_cast<uint32_t>(std::stoul(w, nullptr, 16));
      for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
    }
    const std::string want = line.substr(tab + 1);
    std::string got;
    try {
      got = amd::gcn::to_text(amd::gcn::decode(code, 0, 0));
    } catch (const std::exception& e) {
      got = std::string("(refused: ") + e.what() + ")";
    }
    ++checked;
    if (got != want && wrong_count++ < 10) wrong += "\n  want \"" + want + "\"\n  got  \"" + got + "\"";
  }
  VCHECK(checked > 1000);
  if (wrong_count)
    throw vtest::Failure(std::to_string(wrong_count) + " of " + std::to_string(checked) + " differ:" + wrong);
}

// The cache every launch of a module shares: an instruction is decoded once,
// however many threads reach it at the same moment, and all of them get the
// same copy; a word past the end of the code is not cached.
VTEST(the_shared_decode_cache_decodes_each_instruction_once) {
  std::vector<uint8_t> code;
  for (int i = 0; i < 3 * 4096 + 5; ++i)   // s_nop 0, across more than one page
    for (uint8_t b : {0x00, 0x00, 0x80, 0xbf}) code.push_back(b);
  amd::DecodeCache cache(code.size());
  std::atomic<int> decodes{0};
  const auto get = [&](uint64_t word) {
    return cache.get(word, [&] {
      ++decodes;
      return amd::gcn::decode(code, word * 4, 0x1000 + word * 4);
    });
  };
  const amd::gcn::Inst* first = get(4096 + 7);
  VCHECK(first != nullptr);
  VCHECK_EQ(first->name, std::string("s_nop"));
  VCHECK_EQ(first->pc, uint64_t{0x1000 + (4096 + 7) * 4});
  VCHECK(get(4096 + 7) == first);
  VCHECK_EQ(decodes.load(), 1);
  VCHECK(get(code.size() / 4 + 4096) == nullptr);   // past the last page

  // Threads racing for the same words: each word ends up with one copy.
  std::vector<std::thread> threads;
  std::vector<const amd::gcn::Inst*> seen(8 * 64);
  for (int t = 0; t < 8; ++t)
    threads.emplace_back([&, t] {
      for (int w = 0; w < 64; ++w) seen[t * 64 + w] = get(uint64_t{2} * 4096 + w);
    });
  for (auto& th : threads) th.join();
  for (int t = 1; t < 8; ++t)
    for (int w = 0; w < 64; ++w) VCHECK(seen[t * 64 + w] == seen[w]);
  for (int w = 0; w < 64; ++w) VCHECK_EQ(seen[w]->pc, uint64_t{0x1000} + uint64_t(2 * 4096 + w) * 4);
}

VTEST(an_instruction_says_where_its_operands_are) {
  const amd::CodeObject o = object();
  // s_load_dword s3, s[0:1], 0x18: the kernel's first instruction reads its
  // fourth argument out of the kernarg segment, whose address the hardware
  // left in s[0:1].
  const amd::gcn::Inst load = amd::gcn::decode(o.text, 0, 0);
  VCHECK_EQ(load.name, std::string("s_load_dword"));
  VCHECK_EQ(load.size, 8u);                       // SMEM is two words
  VCHECK_EQ(load.offset, 0x18);
  VCHECK_EQ(load.dst.size(), 1u);
  VCHECK(load.dst[0].kind == amd::gcn::OperandKind::Sgpr);
  VCHECK_EQ(load.dst[0].index, 3u);
  VCHECK(load.src[0].kind == amd::gcn::OperandKind::Sgpr);
  VCHECK_EQ(load.src[0].width, 2u);               // a 64-bit address

  // v_lshl_add_u32 v0, s2, 8, v0: a vector instruction reading a scalar, an
  // inline constant and a vector register.
  const amd::gcn::Inst mad = amd::gcn::decode(o.text, load.size, load.size);
  VCHECK_EQ(mad.name, std::string("v_lshl_add_u32"));
  VCHECK_EQ(mad.src.size(), 3u);
  VCHECK(mad.src[0].kind == amd::gcn::OperandKind::Sgpr);
  VCHECK(mad.src[1].kind == amd::gcn::OperandKind::Inline);
  VCHECK_EQ(mad.src[1].value, 8);
  VCHECK(mad.src[2].kind == amd::gcn::OperandKind::Vgpr);
  VCHECK(mad.dst[0].kind == amd::gcn::OperandKind::Vgpr);
}

VTEST(a_branch_says_where_it_goes) {
  const amd::CodeObject o = object();
  // The first kernel guards its work with a comparison and a branch over it,
  // and the branch lands on the s_endpgm that ends the kernel.
  bool checked = false;
  uint64_t at = 0;
  while (at < o.kernels[0].size) {
    const amd::gcn::Inst in = amd::gcn::decode(o.text, at, o.text_addr + at);
    if (in.name == "s_cbranch_execz") {
      VCHECK(in.target > in.pc);
      const amd::gcn::Inst landed = amd::gcn::decode(o.text, in.target - o.text_addr, in.target);
      VCHECK_EQ(landed.name, std::string("s_endpgm"));
      checked = true;
    }
    at += in.size;
  }
  VCHECK(checked);
}

VTEST(an_instruction_this_does_not_know_is_refused_by_name) {
  std::string what;
  try {
    // VOP3 with an opcode nothing uses.
    const std::vector<uint8_t> code = {0x00, 0x00, 0xff, 0xd3, 0x00, 0x00, 0x00, 0x00};
    amd::gcn::decode(code, 0, 0);
  } catch (const std::exception& e) {
    what = e.what();
  }
  VCHECK_CONTAINS(what, "VOP3");
  VCHECK_CONTAINS(what, "not decoded yet");

  // Code that stops in the middle of an instruction.
  what.clear();
  try {
    amd::gcn::decode({0x00, 0x00}, 0, 0);
  } catch (const std::exception& e) {
    what = e.what();
  }
  VCHECK_CONTAINS(what, "code ends inside an instruction");
}

VTEST_MAIN
