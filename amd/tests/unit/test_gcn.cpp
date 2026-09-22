// Decoding CDNA machine code: every instruction of the fixture's two kernels,
// compared with what the assembler wrote.
//
// amd/tests/data/vector_add.gfx942.dis is that code as llvm-objdump prints
// it, written beside the object by amd/tests/data/build.sh. Decoding the
// object has to produce the same instructions, in the same order, with the
// same operands -- which catches an opcode read from the wrong bits, an
// operand of the wrong width, and an instruction whose length is wrong (that
// one shifts every instruction after it).
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_gcn.hpp"
#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

std::string read(const std::string& name) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + name;
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no fixture at " + path);
  return std::string((std::istreambuf_iterator<char>(in)), {});
}

amd::CodeObject object() { return amd::load_code_object(read("vector_add.gfx942.o"), "vector_add.gfx942.o"); }

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

VTEST(every_instruction_decodes_as_the_assembler_wrote_it) {
  const amd::CodeObject o = object();
  const std::vector<std::string> mine = decoded(o), theirs = lines(read("vector_add.gfx942.dis"));
  VCHECK(theirs.size() > 400);
  VCHECK_EQ(mine.size(), theirs.size());
  for (size_t i = 0; i < mine.size() && i < theirs.size(); ++i)
    if (mine[i] != theirs[i])
      throw vtest::Failure("instruction " + std::to_string(i) + ": decoded \"" + mine[i] + "\", assembler wrote \"" +
                           theirs[i] + "\"");
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
