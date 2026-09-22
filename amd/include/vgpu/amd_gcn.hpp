// Decoding CDNA (GCN) machine code: one instruction at a time, from the
// .text of a code object (vgpu/amd_codeobject.hpp).
//
// An AMD GPU's instructions come in a handful of encodings, each a 32-bit
// word and some a second one, with a 32-bit literal after either where an
// operand asks for one. What an encoding means -- which bits are the opcode,
// which the operands -- is the public ISA (AMD's CDNA3 Instruction Set
// Architecture reference guide); the names are the assembler's.
//
// An instruction this does not know is an error naming its encoding and
// opcode, never a guess: a wrong guess would run silently and give a wrong
// answer.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vgpu::amd::gcn {

// The encodings, by the ISA's names for them.
enum class Enc { Sop1, Sop2, Sopk, Sopc, Sopp, Smem, Vop1, Vop2, Vop3, Vopc, Ds, Flat, Unknown };
const char* enc_name(Enc e);

// Where an operand lives. The ISA numbers scalar registers, vector registers
// and the inline constants in one 9-bit space; this splits them apart.
enum class OperandKind { Sgpr, Vgpr, Vcc, Exec, M0, Inline, Literal, None };
struct Operand {
  OperandKind kind = OperandKind::None;
  uint32_t index = 0;    // the register's number
  int64_t value = 0;     // an inline constant's or a literal's value
  uint32_t width = 1;    // how many 32-bit registers it covers
};
std::string operand_text(const Operand& o);

struct Inst {
  Enc enc = Enc::Unknown;
  uint32_t opcode = 0;
  std::string name;              // "v_add_f32_e32", as the assembler spells it
  uint32_t size = 4;             // bytes, this instruction and its literal
  uint64_t pc = 0;               // where it is, in the code object's addresses

  std::vector<Operand> dst, src;
  // Memory instructions: the offset added to the address, and for DS the
  // second offset a two-address instruction uses.
  int32_t offset = 0, offset1 = 0;
  uint32_t saddr = 0;            // a global_* instruction's scalar base, when it has one
  bool has_saddr = false;
  // s_waitcnt's counts, and the branch target of a branch, as the ISA's
  // immediate gives it.
  int32_t simm = 0;
  uint64_t target = 0;
};

// Decodes the instruction at `at` in `code`. Throws Err::Unsupported naming
// the encoding and opcode when it is one this does not know yet.
Inst decode(const std::vector<uint8_t>& code, uint64_t at, uint64_t pc);

// The instruction as the assembler writes it, for tests and for a
// disassembly listing.
std::string to_text(const Inst& i);

}  // namespace vgpu::amd::gcn
