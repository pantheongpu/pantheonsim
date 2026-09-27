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
// Mtbuf and Vopd are RDNA's: a typed buffer access, and VOPD, two vector
// instructions issued as one.
enum class Enc { Sop1, Sop2, Sopk, Sopc, Sopp, Smem, Vop1, Vop2, Vop3, Vop3p, Vopc, Ds, Flat, Mubuf, Mtbuf, Vopd, Unknown };
const char* enc_name(Enc e);

// Where an operand lives. The ISA numbers scalar registers, vector registers
// and the inline constants in one 9-bit space; this splits them apart.
// Agpr: the accumulation registers, which a kernel with more values than it
// has vector registers keeps the rest of them in.
// RDNA adds the trap handler's registers (Ttmp), a null register that reads
// zero and drops what is written to it, SCC as a vector operand, and the
// apertures' limits beside their bases.
enum class OperandKind {
  Sgpr, Vgpr, Agpr, Vcc, VccHi, Exec, ExecLo, ExecHi, M0, SharedBase, PrivateBase, Inline, InlineFloat, Literal,
  Ttmp, Null, Scc, SharedLimit, PrivateLimit, None
};
struct Operand {
  OperandKind kind = OperandKind::None;
  uint32_t index = 0;    // the register's number
  int64_t value = 0;     // an inline constant's or a literal's value
  double fvalue = 0;     // an inline float constant's value (0.5, 1.0, 2.0, 4.0 and their negatives)
  uint32_t width = 1;    // how many 32-bit registers it covers
  bool neg = false;      // VOP3: the source is negated
  bool abs = false;      // VOP3: its absolute value is taken, before the negation
  // SDWA: which part of the register the instruction reads (0 to 3 a byte,
  // 4 and 5 a half, 6 the whole of it), and whether what it reads keeps its
  // sign on the way into a 32-bit operation.
  uint8_t sel = 6;
  bool sext = false;
  // RDNA's 16-bit operands (true16): half a vector register, the high half
  // where `hi` is set -- v1.h -- and the low one otherwise -- v1.l.
  bool half = false;
  bool hi = false;
  bool bits16 = false;   // a 16-bit operand, whatever it is
  bool hidden = false;   // read by the instruction, not written by the assembler (VOPD's vcc_lo)
  // A literal in a 64-bit float operand: the word is the double's high half,
  // the low half zero (a 64-bit integer operand's is zero-extended instead).
  bool literal_high = false;
  bool constant_k = false;   // RDNA: an instruction's own constant (v_fmaak_f32's K), always written in hex
};
std::string operand_text(const Operand& o);

// Which processor the code is for. gfx90a (CDNA2) numbers some instructions
// differently from gfx940 and later -- its matrix instructions above all --
// and has a few they dropped (v_mad_f32, v_mac_f32); gfx950 adds to gfx942's.
// What a vector instruction counts as for the performance counters
// (SQ_INSTS_VALU_ADD_F32 and the rest): its operation and type. Worked out
// once, when the instruction is decoded.
enum class Mix : uint8_t {
  None, AddF16, AddF32, AddF64, MulF16, MulF32, MulF64, FmaF16, FmaF32, FmaF64,
  TransF16, TransF32, TransF64, Cvt, Int32, Int64, Count
};
// And a matrix instruction's input type, for SQ_INSTS_VALU_MFMA_MOPS_*.
enum class MopsType : uint8_t { None, I8, F16, BF16, F32, F64, F8, Count };

// gfx1100 stands for RDNA3 (gfx11), whose encodings are its own: every
// gfx11 GPU decodes alike. gfx1200 for RDNA4 (gfx12) likewise.
enum class Target { Gfx942, Gfx90a, Gfx950, Gfx1100, Gfx1200 };
inline bool is_rdna(Target t) { return t == Target::Gfx1100 || t == Target::Gfx1200; }

struct Inst {
  Enc enc = Enc::Unknown;
  uint32_t opcode = 0;
  std::string name;              // "v_add_f32_e32", as the assembler spells it
  // RDNA: the assembler's name, where `name` is the gfx9 instruction's that
  // does the same (global_load_b32 runs as global_load_dword).
  std::string asm_name;
  uint32_t size = 4;             // bytes, this instruction and its literal
  uint64_t pc = 0;               // where it is, in the code object's addresses

  std::vector<Operand> dst, src;
  // Memory instructions: the offset added to the address, and for DS the
  // second offset a two-address instruction uses.
  int32_t offset = 0, offset1 = 0;
  uint32_t saddr = 0;            // a global_* instruction's scalar base, when it has one
  bool has_saddr = false;
  bool has_vaddr = true;         // a scratch_* instruction may address by offset alone
  // Which of the one address space a memory instruction reaches: flat (the
  // address decides), global, or a work-item's private memory.
  enum class Segment { Flat, Scratch, Global } segment = Segment::Global;
  // s_waitcnt's counts, and the branch target of a branch, as the ISA's
  // immediate gives it.
  int32_t simm = 0;
  uint64_t target = 0;
  // VOP3's clamp: a float result is held to [0, 1].
  bool clamp = false;
  // VOP3's (and SDWA's) output multiplier: 1 doubles a float result, 2
  // quadruples it, 3 halves it.
  uint8_t omod = 0;
  // gfx950's v_bitop3: the truth table of three inputs, kept in the bits
  // VOP3 otherwise uses for negation (bits 0-2), absolute value (3-5) and
  // the output multiplier (6-7).
  uint8_t bitop3 = 0;
  // gfx950's f8f6f4 matrix instructions: A's and B's formats, which the
  // encoding keeps where the others' broadcast controls are (CBSZ, BLGP):
  // 0 fp8, 1 bf8, 2 fp6 (E2M3), 3 bf6 (E3M2), 4 fp4 (E2M1).
  uint8_t cbsz = 0, blgp = 0;
  // gfx950's scaled matrix instructions (v_mfma_scale_*): 16 bytes, a
  // load-scale prefix then the product. Sources 3 and 4 are A's and B's E8M0
  // scales, and scale_sel which byte of each register (bits 0-1 A's, 2-3 B's):
  // {OP_SEL_HI, OP_SEL} of the prefix, per source.
  bool scaled = false;
  // The sparse matrix instructions (v_smfmac_*): which set of indices in the
  // index register -- ABID, where CBSZ is 0; the first set otherwise.
  uint8_t abid = 0;
  uint8_t scale_sel = 0;
  // The processor it was decoded for, which names some instructions and
  // cache bits otherwise when it is printed (gfx90a's glc, slc and scc).
  Target arch = Target::Gfx942;
  // SMEM: the offset is the instruction's own and a register's both, which
  // the assembler writes as "offset:" even when the constant is zero.
  bool smem_both_offsets = false;
  // The VOP3 form of a VOP1 or VOP2 instruction, decoded from the short
  // form's entry: it runs as the short form does.
  bool promoted = false;
  // SDWA: the instruction reads part of a register rather than all of it,
  // which is how the compiler mixes widths. The destination has the same
  // choice, and says what becomes of the rest of the register.
  // VOP3P: which half of each source feeds the low result and which feeds the
  // high one, a bit per source. For the packed float form a "half" is one
  // register of a pair, and a zero in op_sel_hi means the high result reads
  // the low register, which is how a single value is used by both.
  uint8_t op_sel = 0;
  uint8_t op_sel_hi = 7;
  // VOP3P: which sources are negated on their way into the low result, and
  // into the high one, a bit per source. (The mixed-precision forms use
  // these bits as each source's negation and absolute value instead, and the
  // decoder puts them on the operands.)
  uint8_t neg_lo = 0;
  uint8_t neg_hi = 0;
  // MUBUF: whether the address register holds an offset into the buffer, an
  // index into it, or (both set) the index then the offset.
  bool offen = false, idxen = false;
  // DPP: the instruction reads its first source from another lane. The
  // control says which lane, the two masks say which lanes are written at
  // all, and bound_ctrl says what a lane gets when the lane it would read
  // is not there -- zero, or nothing written.
  bool dpp = false;
  uint32_t dpp_ctrl = 0;
  uint8_t row_mask = 0xF;
  uint8_t bank_mask = 0xF;
  bool bound_ctrl = false;
  bool sdwa = false;
  uint8_t dst_sel = 6;
  uint8_t dst_unused = 0;   // 0 pads the rest with zeroes, 1 with the sign, 2 keeps it
  // RDNA: DPP8 (each of eight lanes picks one of the eight, dpp_ctrl holding
  // the eight 3-bit choices), DPP's fetch-inactive bit, DS's GDS bit, a typed
  // buffer access's format, and VOPD's two halves.
  bool dpp8 = false;
  bool fi = false;
  bool gfx12_cache = false;   // RDNA4: `cache` is TH | SCOPE << 3, not glc/slc/dlc
  uint8_t printed_op_sel = 0;  // RDNA VOP3: the op_sel the assembler writes (gfx11 folds 16-bit halves into v1.h)
  bool gds = false;
  uint32_t format = 0;
  std::vector<Inst> dual;
  // For the performance counters: what the instruction counts as, and a
  // matrix instruction's work in units of 512 floating-point (or integer)
  // operations, its shape's 2 x M x N x K (x blocks).
  Mix mix = Mix::None;
  MopsType mops_type = MopsType::None;
  uint32_t mops = 0;
  // A memory instruction's scope bits (global_atomic_*'s sc0/sc1/nt), which
  // say how far a write is published. Every access here is already visible to
  // every wave, so they change nothing and are kept for the listing.
  uint32_t cache = 0;
};

// The target a code object's e_flags machine field (EF_AMDGPU_MACH) names.
inline Target target_of_mach(uint32_t mach) {
  switch (mach) {
    case 0x3f: return Target::Gfx90a;
    case 0x4f: return Target::Gfx950;
    // gfx1100, 1101, 1102, 1103, 1150, 1151, 1152: RDNA3 and 3.5.
    case 0x41: case 0x46: case 0x47: case 0x44: case 0x43: case 0x4a: case 0x55: return Target::Gfx1100;
    // gfx1200, 1201: RDNA4.
    case 0x48: case 0x4e: return Target::Gfx1200;
    default: return Target::Gfx942;
  }
}

// Decodes the instruction at `at` in `code`, for `target`. Throws
// Err::Unsupported naming the encoding and opcode when it is one this does
// not know yet.
// `wave64`: RDNA code built for 64-lane waves, whose lane masks (VCC, a
// comparison's result, a carry) are register pairs rather than one register.
Inst decode(const std::vector<uint8_t>& code, uint64_t at, uint64_t pc, Target target = Target::Gfx942,
            bool wave64 = false);

// The instruction as the assembler writes it, for tests and for a
// disassembly listing.
std::string to_text(const Inst& i);

}  // namespace vgpu::amd::gcn
