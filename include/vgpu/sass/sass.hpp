// NVIDIA SASS: the machine code of Volta and later GPUs (sm_70 onward), which
// shares one 128-bit instruction format across generations. See
// nvidia/docs/sass.md for where the encodings come from and how the decoder
// is checked (every instruction of a corpus against NVIDIA's nvdisasm).
//
// An instruction's low 64 bits carry the opcode (bits 0-11), the guard
// predicate (12-15), the destination (16-23) and the sources; the high 64
// bits carry more operand and modifier bits and, from bit 105, the control
// bits the scheduler reads (stall count, yield, dependency barriers, operand
// reuse). Control bits are timing, which VirtualGPU does not model; only the
// reuse flags are kept, because nvdisasm prints them.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vgpu::sass {

// One instruction's 128 bits.
struct Word {
  uint64_t lo = 0, hi = 0;

  // `len` bits starting at bit `pos` (0-127) as an unsigned value.
  uint64_t field(unsigned pos, unsigned len) const {
    if (len == 0) return 0;
    const auto take = [](uint64_t w, unsigned p, unsigned n) {
      return n >= 64 ? (w >> p) : ((w >> p) & ((uint64_t{1} << n) - 1));
    };
    if (pos >= 64) return take(hi, pos - 64, len);
    if (pos + len <= 64) return take(lo, pos, len);
    const unsigned low_bits = 64 - pos;
    return take(lo, pos, low_bits) | (take(hi, 0, len - low_bits) << low_bits);
  }
  bool bit(unsigned pos) const { return field(pos, 1) != 0; }
  // The same field read as a two's-complement signed value.
  int64_t sfield(unsigned pos, unsigned len) const {
    const uint64_t v = field(pos, len);
    return len < 64 && (v >> (len - 1)) ? static_cast<int64_t>(v | (~uint64_t{0} << len))
                                        : static_cast<int64_t>(v);
  }
};

// The register numbers that mean "zero" / "true" rather than a register.
constexpr unsigned kRZ = 255;   // RZ
constexpr unsigned kPT = 7;     // PT, UPT

// What an operand is. The printer turns each into nvdisasm's spelling.
enum class Kind : uint8_t {
  Reg,      // R<n>, RZ; `width` registers from n for 64/128-bit values
  UReg,     // UR<n>, URZ
  Pred,     // P<n>, PT (with ! for not)
  UPred,    // UP<n>, UPT
  Imm,      // integer immediate
  FImm,     // floating-point immediate, printed as nvdisasm prints floats
  CBank,    // c[bank][offset]
  UCBank,   // cx[UR<n>][offset]: a bank chosen by a uniform register
  Mem,      // [base.64 + UR + imm] and its variants; `text` holds the form
  SReg,     // SR_TID.X and the rest; `text` holds the name
  Label,    // a branch target (an absolute code address)
  Bar,      // B<n>: a convergence barrier
  Text,     // anything printed verbatim (`text`)
};

struct Operand {
  Kind kind = Kind::Reg;
  uint32_t reg = 0;         // register/predicate number, or c[] bank
  int64_t imm = 0;          // immediate, c[] offset, label address
  uint8_t width = 1;        // registers read or written (R2.64 is 2)
  bool neg = false;         // -R, and ! on predicates
  bool abs = false;         // |R|
  bool bnot = false;        // ~R (bitwise not, integer ops)
  bool reuse = false;       // .reuse (the operand-reuse cache: timing only)
  std::string suffix;       // printed after the operand: .H1, .B2, .ROW, ...
  std::string text;         // Mem/SReg/Text: the operand as nvdisasm prints it
  uint32_t fbits = 0;       // FImm: the float's bits
  uint8_t fwidth = 32;      // FImm: 16, 32 or 64
  uint8_t slot = 0;         // ALU sources: the encoding field it came from
};

// Every instruction VirtualGPU decodes. The executor switches on this.
enum class Op : uint16_t {
  Unknown,
#define VGPU_SASS_OP(name) name,
#include "vgpu/sass/ops.inc"
#undef VGPU_SASS_OP
};

const char* op_name(Op op);

struct Instr {
  Word w;
  uint64_t pc = 0;          // byte address of the instruction in its section
  int sm = 0;               // architecture it was decoded for: 75, 86, 90, ...
  Op op = Op::Unknown;
  std::string mnemonic;     // as printed: IMAD, ISETP, ...
  std::vector<std::string> mods;   // printed after the mnemonic, dot-joined
  uint8_t guard = kPT;      // the predicate that guards it (PT: always)
  bool guard_not = false;
  bool guard_uniform = false;      // @UP<n> rather than @P<n>
  std::vector<Operand> dst;        // written operands, printed first
  std::vector<Operand> src;        // read operands, printed after
  // Op-specific decoded fields the executor reads (what each means is set by
  // the op's decoder, and documented there).
  uint32_t f[8] = {};
};

// Decodes one instruction. `pc` is its address (branch targets are relative to
// the next instruction). Throws vgpu::Error naming the opcode when it is not
// one VirtualGPU knows.
Instr decode(const Word& w, uint64_t pc, int sm);

// The instruction as nvdisasm prints it, without the trailing " ;".
std::string to_text(const Instr& ins);

}  // namespace vgpu::sass
