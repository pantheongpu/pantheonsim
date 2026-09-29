// SASS decoder for the sm_70-and-later 128-bit format. Field positions come
// from Mesa NAK's encoder (src/nouveau/compiler/nak/sm70_encode.rs, MIT) and,
// where NAK is silent, from NVIDIA's own disassembler: every decoder here is
// checked against nvdisasm's text for a corpus of real instructions
// (nvidia/tests/data/sass/, test_sass_decode).
#include "vgpu/sass/sass.hpp"

#include <cstdio>
#include <cstring>
#include <unordered_map>

#include "vgpu/error.hpp"

namespace vgpu::sass {

const char* op_name(Op op) {
  static const char* const names[] = {
      "(unknown)",
#define VGPU_SASS_OP(name) #name,
#include "vgpu/sass/ops.inc"
#undef VGPU_SASS_OP
  };
  return names[static_cast<size_t>(op)];
}

namespace {

// ---- operand builders -------------------------------------------------------

Operand R(unsigned n, unsigned width = 1) {
  Operand o;
  o.kind = Kind::Reg;
  o.reg = n;
  o.width = static_cast<uint8_t>(width);
  return o;
}
Operand UR(unsigned n, int sm) {
  Operand o;
  o.kind = Kind::UReg;
  // Before sm_100 a uniform register is 6 bits and 63 is URZ; from sm_100 it
  // is 8 bits and 255 is URZ.
  o.reg = n;
  (void)sm;
  return o;
}
Operand P(unsigned n, bool neg = false) {
  Operand o;
  o.kind = Kind::Pred;
  o.reg = n;
  o.neg = neg;
  return o;
}
Operand UP(unsigned n, bool neg = false) {
  Operand o;
  o.kind = Kind::UPred;
  o.reg = n;
  o.neg = neg;
  return o;
}
Operand Imm(int64_t v) {
  Operand o;
  o.kind = Kind::Imm;
  o.imm = v;
  return o;
}
Operand CB(unsigned bank, unsigned offset) {
  Operand o;
  o.kind = Kind::CBank;
  o.reg = bank;
  o.imm = offset;
  return o;
}
Operand Txt(std::string s) {
  Operand o;
  o.kind = Kind::Text;
  o.text = std::move(s);
  return o;
}

unsigned ureg_bits(int sm) { return sm >= 100 ? 8 : 6; }
unsigned urz(int sm) { return sm >= 100 ? 255 : 63; }

// Reuse flags: bits 122-125 say which of source slots a, b, c (24, 32, 64)
// the operand-reuse cache keeps. nvdisasm prints them as .reuse.
bool reuse(const Word& w, int slot) { return w.bit(122 + slot); }

// ---- the ALU operand forms ---------------------------------------------------
//
// Bits 9-11 of the opcode are the form: which of the second and third sources
// is an immediate, a constant-bank value or a uniform register. src0 is always
// the register at 24-31. Where the immediate or constant is the third source,
// the second source moves to the register field at 64-71.
enum class Slot { R24, R32, R64, Imm32, CB, UR32 };

struct AluForm {
  Slot s1, s2;
};

AluForm alu_form(unsigned form) {
  switch (form) {
    case 1: return {Slot::R32, Slot::R64};
    case 2: return {Slot::R64, Slot::Imm32};
    case 3: return {Slot::R64, Slot::CB};
    case 4: return {Slot::Imm32, Slot::R64};
    case 5: return {Slot::CB, Slot::R64};
    case 6: return {Slot::UR32, Slot::R64};
    case 7: return {Slot::R64, Slot::UR32};
    default: throw Error(Err::UnsupportedPtx, "SASS: ALU form " + std::to_string(form));
  }
}

// How a source reads: signed immediates print negative (IADD3, IMAD), logic
// ops print them as raw bits.
enum ImmStyle { kSigned, kUnsigned, kFloat };

// A 32-bit float immediate as nvdisasm prints it: its exact value to twenty
// significant digits (0.25, 1, 1.4426950216293334961), infinities and NaNs
// by name.
Operand FImm32(uint32_t bits) {
  Operand o;
  o.kind = Kind::FImm;
  o.fbits = bits;
  o.fwidth = 32;
  float f;
  std::memcpy(&f, &bits, 4);
  char b[48];
  if (f != f) {
    std::snprintf(b, sizeof b, "%sNAN", (bits >> 31) ? "-" : "+");
  } else if (f == __builtin_inff() || f == -__builtin_inff()) {
    std::snprintf(b, sizeof b, "%sINF", (bits >> 31) ? "-" : "+");
  } else {
    std::snprintf(b, sizeof b, "%.20g", static_cast<double>(f));
  }
  o.text = b;
  return o;
}

// One source in the given slot, with the neg/abs bits that slot carries.
Operand alu_src(const Word& w, Slot s, int sm, ImmStyle imm, bool uniform_op = false) {
  Operand o;
  switch (s) {
    case Slot::R24:
      // Bits 72 and 73 are src0's negate and absolute value only for float
      // ops; integer ops use them for other things (IMAD's signedness,
      // ISETP's .EX), so each op's decoder sets src0's modifiers itself.
      o = uniform_op ? UR(static_cast<unsigned>(w.field(24, ureg_bits(sm))), sm)
                     : R(static_cast<unsigned>(w.field(24, 8)));
      o.slot = static_cast<uint8_t>(s);
      return o;
    case Slot::R32:
      o = uniform_op ? UR(static_cast<unsigned>(w.field(32, ureg_bits(sm))), sm)
                     : R(static_cast<unsigned>(w.field(32, 8)));
      o.slot = static_cast<uint8_t>(s);
      return o;
    case Slot::R64:
      o = uniform_op ? UR(static_cast<unsigned>(w.field(64, ureg_bits(sm))), sm)
                     : R(static_cast<unsigned>(w.field(64, 8)));
      o.slot = static_cast<uint8_t>(s);
      return o;
    case Slot::Imm32: {
      const uint32_t v = static_cast<uint32_t>(w.field(32, 32));
      o = imm == kFloat ? FImm32(v)
                        : Imm(imm == kSigned ? static_cast<int64_t>(static_cast<int32_t>(v)) : static_cast<int64_t>(v));
      o.slot = static_cast<uint8_t>(s);
      return o;
    }
    case Slot::CB:
      if (w.bit(91)) {   // bindless: cx[UR][offset]
        o.kind = Kind::UCBank;
        o.reg = static_cast<unsigned>(w.field(32, 6));
        o.imm = static_cast<int64_t>(w.field(38, 16));
      } else {
        o = CB(static_cast<unsigned>(w.field(54, 5)), static_cast<unsigned>(w.field(38, 16)));
      }
      o.slot = static_cast<uint8_t>(s);
      return o;
    case Slot::UR32:
      o = UR(static_cast<unsigned>(w.field(32, ureg_bits(sm))), sm);
      o.slot = static_cast<uint8_t>(s);
      return o;
  }
  return o;
}

// The reuse flags belong to the operand's position among the sources (a, b,
// c), not to the encoding field it came from.
void mark_reuse(Instr& ins, const Word& w) {
  for (size_t i = 0; i < ins.src.size() && i < 3; ++i)
    if (ins.src[i].kind == Kind::Reg) ins.src[i].reuse = reuse(w, static_cast<int>(i));
}

// The three sources of a three-source ALU instruction, in printed order.
void alu3(Instr& ins, const Word& w, ImmStyle imm, bool uniform = false) {
  const AluForm f = alu_form(static_cast<unsigned>(w.field(9, 3)));
  ins.src.push_back(alu_src(w, Slot::R24, ins.sm, imm, uniform));
  ins.src.push_back(alu_src(w, f.s1, ins.sm, imm, uniform));
  ins.src.push_back(alu_src(w, f.s2, ins.sm, imm, uniform));
  mark_reuse(ins, w);
}

// Two sources: src0 and whichever of the form's slots holds the second (a
// two-source op keeps its second source in the second slot, except that an
// immediate, constant or uniform register takes the third slot's place when
// the form says the third is that kind).
void alu2(Instr& ins, const Word& w, ImmStyle imm, bool uniform = false) {
  const AluForm f = alu_form(static_cast<unsigned>(w.field(9, 3)));
  ins.src.push_back(alu_src(w, Slot::R24, ins.sm, imm, uniform));
  const Slot s = f.s1 == Slot::R64 ? f.s2 : f.s1;
  ins.src.push_back(alu_src(w, s, ins.sm, imm, uniform));
  mark_reuse(ins, w);
}

// Source modifiers, which live in bits tied to the field a source came from.
// Float ops have negate and absolute value on every source; integer ops that
// negate (IADD3, IMAD's addend) have the negate bits only. Which ops have
// which is the op's own business: several reuse these bits for other fields.
void src_neg_abs(Operand& o, const Word& w, bool with_abs) {
  switch (static_cast<Slot>(o.slot)) {
    case Slot::R24:
      o.neg = w.bit(72);
      if (with_abs) o.abs = w.bit(73);
      break;
    case Slot::R64:
      o.neg = w.bit(75);
      if (with_abs) o.abs = w.bit(74);
      break;
    case Slot::R32:
    case Slot::CB:
    case Slot::UR32:
      o.neg = w.bit(63);
      if (with_abs) o.abs = w.bit(62);
      break;
    case Slot::Imm32: break;
  }
}
void float_srcs(Instr& ins, const Word& w) {
  for (Operand& o : ins.src) src_neg_abs(o, w, true);
}
void int_neg(Operand& o, const Word& w) { src_neg_abs(o, w, false); }

Operand dst_reg(const Word& w, bool uniform, int sm) {
  return uniform ? UR(static_cast<unsigned>(w.field(16, ureg_bits(sm))), sm)
                 : R(static_cast<unsigned>(w.field(16, 8)));
}

// A predicate source at [pos, pos+3) with its not bit.
Operand pred_src(const Word& w, unsigned pos, unsigned not_bit, bool uniform = false) {
  return uniform ? UP(static_cast<unsigned>(w.field(pos, 3)), w.bit(not_bit))
                 : P(static_cast<unsigned>(w.field(pos, 3)), w.bit(not_bit));
}

const char* const kIntCmp[] = {"F", "LT", "EQ", "LE", "GT", "NE", "GE", "T"};
const char* const kBoolOp[] = {"AND", "OR", "XOR", "(bool 3)"};

// ---- integer ALU -------------------------------------------------------------

void dec_imad(Instr& ins, const Word& w, bool uniform) {
  const unsigned low9 = static_cast<unsigned>(w.field(0, 9) & 0x7f);   // 0x24 IMAD, 0x25 WIDE, 0x27 HI
  ins.op = uniform ? Op::UIMAD : Op::IMAD;
  ins.mnemonic = uniform ? "UIMAD" : "IMAD";
  const bool is_signed = w.bit(73);
  const bool x = w.bit(74);   // .X: add the carry-in predicate
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  alu3(ins, w, kSigned, uniform);
  int_neg(ins.src[1], w);
  int_neg(ins.src[2], w);
  for (Operand& o : ins.src)
    if (x && o.neg) {   // with a carry-in, the complement: see IADD3
      o.neg = false;
      o.bnot = true;
    }
  ins.f[0] = low9;
  ins.f[1] = is_signed;
  ins.f[2] = x;
  // The carry-out predicate (81-83) and carry-in (87-89, not 90).
  const unsigned pdst = static_cast<unsigned>(w.field(81, 3));
  if (low9 == 0x25) {
    ins.mods.push_back("WIDE");
    if (!is_signed) ins.mods.push_back("U32");
  } else if (low9 == 0x27) {
    ins.mods.push_back("HI");
    if (!is_signed) ins.mods.push_back("U32");
  } else {
    // nvdisasm's aliases for IMAD a, b, c: a move (a*b is 0*0), an add (b is
    // 1), and a shift (b is a power of two, c is zero).
    const Operand& a = ins.src[0];
    const Operand& b = ins.src[1];
    const Operand& c = ins.src[2];
    const bool a_rz = a.kind == Kind::Reg && a.reg == kRZ && !a.neg;
    const bool b_rz = b.kind == Kind::Reg && b.reg == kRZ && !b.neg;
    const bool c_rz = c.kind == Kind::Reg && c.reg == kRZ && !c.neg;
    // Only the mnemonic changes; every operand is still printed.
    const bool c_uniform = c.kind == Kind::UReg;
    if (!uniform && a_rz && b_rz && !x && !c_uniform) {
      ins.mods.push_back("MOV");
      if (!is_signed) ins.mods.push_back("U32");
    } else if (!uniform && b.kind == Kind::Imm && b.imm == 1 && !x && is_signed) {
      ins.mods.push_back("IADD");
    } else if (!uniform && b.kind == Kind::Imm && c_rz && !is_signed && !x && b.imm > 0 &&
               b.imm < 0x10000 && (b.imm & (b.imm - 1)) == 0) {
      ins.mods.push_back("SHL");
      ins.mods.push_back("U32");
    } else if (!is_signed && !x) {
      ins.mods.push_back("U32");
    }
  }
  if (x) ins.mods.push_back("X");
  if (pdst != kPT) ins.dst.push_back(P(pdst));
  if (x) ins.src.push_back(pred_src(w, 87, 90, uniform));
}

void dec_iadd3(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::UIADD3 : Op::IADD3;
  ins.mnemonic = uniform ? "UIADD3" : "IADD3";
  const bool x = w.bit(74);
  if (x) ins.mods.push_back("X");
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  const unsigned p0 = static_cast<unsigned>(w.field(81, 3)), p1 = static_cast<unsigned>(w.field(84, 3));
  if (!x) {
    if (p0 != kPT || p1 != kPT) ins.dst.push_back(uniform ? UP(p0) : P(p0));
    if (p1 != kPT) ins.dst.push_back(uniform ? UP(p1) : P(p1));
  }
  alu3(ins, w, kSigned, uniform);
  for (Operand& o : ins.src) {
    int_neg(o, w);
    // With a carry-in, "negation" is the one's complement: -a is ~a + 1,
    // and the + 1 comes in as the carry.
    if (x && o.neg) {
      o.neg = false;
      o.bnot = true;
    }
  }
  if (x) {
    ins.src.push_back(pred_src(w, 87, 90, uniform));
    ins.src.push_back(pred_src(w, 77, 80, uniform));
  }
  ins.f[0] = x;
}

std::string hex(uint64_t v) {
  char b[24];
  std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(v));
  return b;
}

void dec_lop3(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::ULOP3 : Op::LOP3;
  ins.mnemonic = uniform ? "ULOP3" : "LOP3";
  ins.mods.push_back("LUT");
  const unsigned pdst = static_cast<unsigned>(w.field(81, 3));
  if (pdst != kPT) ins.dst.push_back(uniform ? UP(pdst) : P(pdst));
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  alu3(ins, w, kUnsigned, uniform);
  const unsigned lut = static_cast<unsigned>(w.field(72, 8));
  ins.src.push_back(Imm(lut));
  ins.src.push_back(pred_src(w, 87, 90, uniform));
  ins.f[0] = lut;
  ins.f[1] = w.bit(80);   // .PAND: the predicate is the AND of the bits, not the OR
  if (ins.f[1]) ins.mods.push_back("PAND");
}

void dec_shf(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::USHF : Op::SHF;
  ins.mnemonic = uniform ? "USHF" : "SHF";
  static const char* const types[] = {"S64", "U64", "S32", "U32"};
  const unsigned type = static_cast<unsigned>(w.field(73, 2));
  const bool wrap = w.bit(75), right = w.bit(76), hi = w.bit(80);
  ins.mods.push_back(right ? "R" : "L");
  if (wrap) ins.mods.push_back("W");
  ins.mods.push_back(types[type]);
  if (hi) ins.mods.push_back("HI");
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  alu3(ins, w, kUnsigned, uniform);
  ins.f[0] = type;
  ins.f[1] = wrap;
  ins.f[2] = right;
  ins.f[3] = hi;
}

void dec_isetp(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::UISETP : Op::ISETP;
  ins.mnemonic = uniform ? "UISETP" : "ISETP";
  const unsigned cmp = static_cast<unsigned>(w.field(76, 3));
  const bool is_signed = w.bit(73);
  const unsigned bop = static_cast<unsigned>(w.field(74, 2));
  const bool ex = w.bit(72);
  ins.mods.push_back(kIntCmp[cmp]);
  if (!is_signed) ins.mods.push_back("U32");
  ins.mods.push_back(kBoolOp[bop]);
  if (ex) ins.mods.push_back("EX");
  const unsigned p0 = static_cast<unsigned>(w.field(81, 3)), p1 = static_cast<unsigned>(w.field(84, 3));
  ins.dst.push_back(uniform ? UP(p0) : P(p0));
  ins.dst.push_back(uniform ? UP(p1) : P(p1));
  alu2(ins, w, kSigned, uniform);
  ins.src.push_back(pred_src(w, 87, 90, uniform));
  if (ex) ins.src.push_back(pred_src(w, 68, 71, uniform));
  ins.f[0] = cmp;
  ins.f[1] = is_signed;
  ins.f[2] = bop;
  ins.f[3] = ex;
}

void dec_lea(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::ULEA : Op::LEA;
  ins.mnemonic = uniform ? "ULEA" : "LEA";
  const bool hi = w.bit(80), x = w.bit(74), sx32 = w.bit(73);
  const unsigned shift = static_cast<unsigned>(w.field(75, 5));
  if (hi) ins.mods.push_back("HI");
  if (x) ins.mods.push_back("X");
  if (sx32) ins.mods.push_back("SX32");
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  const unsigned pdst = static_cast<unsigned>(w.field(81, 3));
  if (pdst != kPT) ins.dst.push_back(uniform ? UP(pdst) : P(pdst));
  // .HI shifts a 64-bit value (the third source is its high word), except
  // with .SX32, where the high word is the first source's sign.
  if (hi && !sx32) {
    alu3(ins, w, kUnsigned, uniform);
  } else {
    alu2(ins, w, kUnsigned, uniform);
  }
  int_neg(ins.src[1], w);
  ins.src[0].neg = w.bit(72);   // the shifted operand's negation
  if (x && ins.src[1].neg) {
    ins.src[1].neg = false;
    ins.src[1].bnot = true;
  }
  ins.src.push_back(Imm(shift));
  if (x) ins.src.push_back(pred_src(w, 87, 90, uniform));
  ins.f[0] = hi;
  ins.f[1] = x;
  ins.f[2] = sx32;
  ins.f[3] = shift;
}

void dec_prmt(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::UPRMT : Op::PRMT;
  ins.mnemonic = uniform ? "UPRMT" : "PRMT";
  static const char* const modes[] = {"", "F4E", "B4E", "RC8", "ECL", "ECR", "RC16", "(7)"};
  const unsigned mode = static_cast<unsigned>(w.field(72, 3));
  if (mode) ins.mods.push_back(modes[mode]);
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  alu3(ins, w, kUnsigned, uniform);
  ins.f[0] = mode;
}

void dec_sel(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::USEL : Op::SEL;
  ins.mnemonic = uniform ? "USEL" : "SEL";
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  alu2(ins, w, kUnsigned, uniform);
  ins.src.push_back(pred_src(w, 87, 90, uniform));
}

void dec_mov(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::UMOV : Op::MOV;
  ins.mnemonic = uniform ? "UMOV" : "MOV";
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  const AluForm f = alu_form(static_cast<unsigned>(w.field(9, 3)));
  ins.src.push_back(alu_src(w, f.s1 == Slot::R64 ? f.s2 : f.s1, ins.sm, kUnsigned, uniform));
  mark_reuse(ins, w);
  const unsigned lanes = static_cast<unsigned>(w.field(72, 4));
  if (!uniform && lanes != 0xf) ins.src.push_back(Imm(lanes));
  ins.f[0] = lanes;
}

void dec_imnmx(Instr& ins, const Word& w, bool uniform) {
  ins.op = Op::IMNMX;
  ins.mnemonic = uniform ? "UIMNMX" : "IMNMX";
  if (!w.bit(73)) ins.mods.push_back("U32");
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  alu2(ins, w, kSigned, uniform);
  ins.src.push_back(pred_src(w, 87, 90, uniform));
  ins.f[0] = w.bit(73);
}

// ---- float ALU ---------------------------------------------------------------

const char* const kRnd[] = {"", "RM", "RP", "RZ"};

void float_mods(Instr& ins, const Word& w, bool has_dnz) {
  const unsigned rnd = static_cast<unsigned>(w.field(78, 2));
  if (w.bit(80)) ins.mods.push_back("FTZ");
  if (has_dnz && w.bit(76)) ins.mods.push_back("DNZ");
  if (rnd) ins.mods.push_back(kRnd[rnd]);
  if (w.bit(77)) ins.mods.push_back("SAT");
  ins.f[0] = rnd;
  ins.f[1] = w.bit(80);
  ins.f[2] = w.bit(77);
  ins.f[3] = has_dnz && w.bit(76);
}

void dec_ffma(Instr& ins, const Word& w) {
  ins.op = Op::FFMA;
  ins.mnemonic = "FFMA";
  float_mods(ins, w, true);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu3(ins, w, kFloat);
  float_srcs(ins, w);
}

void dec_fmul(Instr& ins, const Word& w) {
  ins.op = Op::FMUL;
  ins.mnemonic = "FMUL";
  float_mods(ins, w, true);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu2(ins, w, kFloat);
  float_srcs(ins, w);
}

void dec_fadd(Instr& ins, const Word& w) {
  ins.op = Op::FADD;
  ins.mnemonic = "FADD";
  float_mods(ins, w, false);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  // FADD's second source sits in the third slot when it is a register... in
  // every form; the second slot is unused (RZ).
  const AluForm f = alu_form(static_cast<unsigned>(w.field(9, 3)));
  ins.src.push_back(alu_src(w, Slot::R24, ins.sm, kFloat));
  ins.src.push_back(alu_src(w, f.s1 == Slot::R64 ? f.s2 : f.s1, ins.sm, kFloat));
  mark_reuse(ins, w);
  float_srcs(ins, w);
}

// ---- control -----------------------------------------------------------------

// A relative branch target: a signed offset in bytes from the next
// instruction, held in bits 34-81 (the low two bits are always zero).
uint64_t branch_target(const Word& w, uint64_t pc) {
  const int64_t off = w.sfield(34, 48) * 4;
  return static_cast<uint64_t>(static_cast<int64_t>(pc) + 16 + off);
}

Operand label(uint64_t target) {
  Operand o;
  o.kind = Kind::Label;
  o.imm = static_cast<int64_t>(target);
  return o;
}

void dec_bra(Instr& ins, const Word& w) {
  ins.op = Op::BRA;
  ins.mnemonic = "BRA";
  // Bits 32-33: 0 a plain branch; 2 .DIV and 3 .CONV, which branch only
  // when the warp is (not) converged, per the uniform register at 24-29
  // (bit 30 inverts it).
  const unsigned mode = static_cast<unsigned>(w.field(32, 2));
  if (mode == 2) ins.mods.push_back("DIV");
  if (mode == 3) ins.mods.push_back("CONV");
  const unsigned cond = static_cast<unsigned>(w.field(87, 3));
  if (cond != kPT || w.bit(90)) ins.src.push_back(pred_src(w, 87, 90));
  if (mode >= 2) {
    Operand u = UR(static_cast<unsigned>(w.field(24, ureg_bits(ins.sm))), ins.sm);
    u.bnot = w.bit(30);
    ins.src.push_back(u);
  }
  ins.src.push_back(label(branch_target(w, ins.pc)));
  ins.f[0] = mode;
}

void dec_exit(Instr& ins, const Word& w) {
  ins.op = Op::EXIT;
  ins.mnemonic = "EXIT";
  const unsigned cond = static_cast<unsigned>(w.field(87, 3));
  if (cond != kPT || w.bit(90)) ins.src.push_back(pred_src(w, 87, 90));
}

void dec_bssy(Instr& ins, const Word& w) {
  ins.op = Op::BSSY;
  ins.mnemonic = "BSSY";
  Operand b;
  b.kind = Kind::Bar;
  b.reg = static_cast<unsigned>(w.field(16, 4));
  ins.dst.push_back(b);
  ins.src.push_back(label(branch_target(w, ins.pc)));
}

void dec_bsync(Instr& ins, const Word& w) {
  ins.op = Op::BSYNC;
  ins.mnemonic = "BSYNC";
  Operand b;
  b.kind = Kind::Bar;
  b.reg = static_cast<unsigned>(w.field(16, 4));
  ins.src.push_back(b);
}

void dec_nop(Instr& ins, const Word&) {
  ins.op = Op::NOP;
  ins.mnemonic = "NOP";
}

// ---- special registers ---------------------------------------------------------

std::string sreg_name(unsigned idx) {
  switch (idx) {
    case 0x00: return "SR_LANEID";
    case 0x01: return "SR_CLOCK";
    case 0x02: return "SR_VIRTCFG";
    case 0x03: return "SR_VIRTID";
    case 0x21: return "SR_TID.X";
    case 0x22: return "SR_TID.Y";
    case 0x23: return "SR_TID.Z";
    case 0x25: return "SR_CTAID.X";
    case 0x26: return "SR_CTAID.Y";
    case 0x27: return "SR_CTAID.Z";
    case 0x28: return "SR_NTID";
    case 0x29: return "SR_CirQueueIncrMinusOne";
    case 0x2a: return "SR_NLATC";
    case 0x2c: return "SR_SM_SPA_VERSION";
    case 0x2d: return "SR_MULTIPASSSHADERINFO";
    case 0x2e: return "SR_LWINHI";
    case 0x2f: return "SR_SWINHI";
    case 0x30: return "SR_SWINLO";
    case 0x31: return "SR_SWINSZ";
    case 0x32: return "SR_SMEMSZ";
    case 0x33: return "SR_SMEMBANKS";
    case 0x34: return "SR_LWINLO";
    case 0x35: return "SR_LWINSZ";
    case 0x36: return "SR_LMEMLOSZ";
    case 0x37: return "SR_LMEMHIOFF";
    case 0x38: return "SR_EQMASK";
    case 0x39: return "SR_LTMASK";
    case 0x3a: return "SR_LEMASK";
    case 0x3b: return "SR_GTMASK";
    case 0x3c: return "SR_GEMASK";
    case 0x3d: return "SR_REGALLOC";
    case 0x3f: return "SR_GLOBALERRORSTATUS";
    case 0x41: return "SR_WARPERRORSTATUS";
    case 0x50: return "SR_CLOCKLO";
    case 0x51: return "SR_CLOCKHI";
    case 0x52: return "SR_GLOBALTIMERLO";
    case 0x53: return "SR_GLOBALTIMERHI";
    case 0x80: return "SR_PM0";
    default: {
      char b[16];
      std::snprintf(b, sizeof b, "SR%u", idx);
      return b;
    }
  }
}

void dec_s2r(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::S2UR : Op::S2R;
  ins.mnemonic = uniform ? "S2UR" : "S2R";
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  const unsigned idx = static_cast<unsigned>(w.field(72, 8));
  Operand s;
  s.kind = Kind::SReg;
  s.reg = idx;
  s.text = sreg_name(idx);
  ins.src.push_back(s);
  ins.f[0] = idx;
}

void dec_uldc(Instr& ins, const Word& w) {
  ins.op = Op::ULDC;
  ins.mnemonic = "ULDC";
  static const char* const sizes[] = {"U8", "S8", "U16", "S16", "", "64", "(6)", "(7)"};
  const unsigned size = static_cast<unsigned>(w.field(73, 3));
  if (size != 4) ins.mods.push_back(sizes[size]);
  ins.dst.push_back(UR(static_cast<unsigned>(w.field(16, ureg_bits(ins.sm))), ins.sm));
  ins.src.push_back(CB(static_cast<unsigned>(w.field(54, 5)), static_cast<unsigned>(w.field(38, 16))));
  ins.f[0] = size;
}

// ---- memory ------------------------------------------------------------------
//
// Common fields (Mesa NAK's set_mem_access, and nvdisasm for the spelling):
//   73-75  size: U8, S8, U16, S16, 32, 64, 128
//   77-80  order and scope (sm_80+): 0 weak, 4 .CONSTANT, 5 .STRONG.SM,
//          7 .STRONG.GPU, 0xa .STRONG.SYS
//   84-86  eviction priority: 0 .EF, 1 normal, 2 .EL, 3 .LU, 4 .EU, 5 .NA
//   40-63  a signed 24-bit byte offset
//   72     .E: the address is 64 bits and the uniform register at 32 is the
//          memory descriptor (which nvdisasm leaves out before sm_90);
//          without it that uniform register is added to the address
//   90     the register address is 64 bits (R2.64)

const char* const kMemSize[] = {"U8", "S8", "U16", "S16", "", "64", "128", "(size 7)"};

unsigned mem_regs(unsigned size) { return size == 6 ? 4 : size == 5 ? 2 : 1; }

void mem_order_mods(Instr& ins, const Word& w) {
  const unsigned order = static_cast<unsigned>(w.field(77, 4));
  switch (order) {
    case 0x0: break;
    case 0x4: ins.mods.push_back("CONSTANT"); break;
    case 0x5: ins.mods.push_back("STRONG"); ins.mods.push_back("SM"); break;
    case 0x7: ins.mods.push_back("STRONG"); ins.mods.push_back("GPU"); break;
    case 0xa: ins.mods.push_back("STRONG"); ins.mods.push_back("SYS"); break;
    case 0xb: ins.mods.push_back("MMIO"); ins.mods.push_back("SYS"); break;
    default: ins.mods.push_back("ORDER" + std::to_string(order)); break;
  }
  ins.f[1] = order;
}

void evict_mods(Instr& ins, const Word& w) {
  static const char* const names[] = {"EF", "", "EL", "LU", "EU", "NA", "(ev 6)", "(ev 7)"};
  const unsigned ev = static_cast<unsigned>(w.field(84, 3));
  if (ev != 1) ins.mods.push_back(names[ev]);
  ins.f[2] = ev;
}

std::string signed_hex(int64_t v) {
  char b[32];
  if (v < 0) std::snprintf(b, sizeof b, "-0x%llx", static_cast<unsigned long long>(-v));
  else std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(v));
  return b;
}

// [Ra(.64)(.Xn) + URb + off] as nvdisasm prints it: parts that are zero
// (RZ, URZ, a zero offset) are left out, and an address of nothing but an
// offset prints as that offset. `force_ur` keeps a URZ (ATOMS prints it).
Operand mem_addr(unsigned ra, bool ra64, const std::string& ra_suffix, int ur, int64_t off, int sm,
                 bool force_ur = false) {
  Operand o;
  o.kind = Kind::Mem;
  std::string t;
  if (ra != kRZ) t = "R" + std::to_string(ra) + (ra64 ? ".64" : "") + ra_suffix;
  if (ur >= 0 && (static_cast<unsigned>(ur) != urz(sm) || force_ur)) {
    if (!t.empty()) t += "+";
    t += static_cast<unsigned>(ur) == urz(sm) ? "URZ" : "UR" + std::to_string(ur);
  }
  if (off != 0 || t.empty()) {
    if (t.empty()) t = signed_hex(off);
    else t += "+" + signed_hex(off);   // nvdisasm writes a negative one as +-0x80
  }
  o.text = "[" + t + "]";
  o.reg = ra;
  o.imm = off;
  return o;
}

// Global and generic loads/stores: LDG, STG, LD, ST.
void dec_gmem(Instr& ins, const Word& w, Op op, const char* name, bool store) {
  ins.op = op;
  ins.mnemonic = name;
  const bool e = w.bit(72);
  const unsigned size = static_cast<unsigned>(w.field(73, 3));
  if (e) ins.mods.push_back("E");
  if (*kMemSize[size]) ins.mods.push_back(kMemSize[size]);
  mem_order_mods(ins, w);
  evict_mods(ins, w);
  ins.f[0] = size;
  const unsigned ra = static_cast<unsigned>(w.field(24, 8));
  const bool ra64 = w.bit(90);
  // Bit 91 says the uniform register field is in use (the 0x3xx forms of
  // these opcodes have none).
  const int ur = w.bit(91) ? static_cast<int>(w.field(store ? 64 : 32, ureg_bits(ins.sm))) : -1;
  const int64_t off = w.sfield(40, 24);
  // With .E the uniform register is the descriptor, not part of the address.
  const Operand addr = mem_addr(ra, ra64, "", e ? -1 : ur, off, ins.sm);
  if (store) {
    ins.src.push_back(addr);
    ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8)), mem_regs(size)));
  } else {
    ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), mem_regs(size)));
    ins.src.push_back(addr);
  }
}

// Shared and local: LDS, STS, LDL, STL. The address is 32 bits.
void dec_smem(Instr& ins, const Word& w, Op op, const char* name, bool store, bool shared) {
  ins.op = op;
  ins.mnemonic = name;
  const unsigned size = static_cast<unsigned>(w.field(73, 3));
  if (*kMemSize[size]) ins.mods.push_back(kMemSize[size]);
  if (!shared) evict_mods(ins, w);
  ins.f[0] = size;
  const unsigned ra = static_cast<unsigned>(w.field(24, 8));
  static const char* const strides[] = {"", ".X4", ".X8", ".X16"};
  const unsigned stride = shared ? static_cast<unsigned>(w.field(78, 2)) : 0;
  const int ur = w.bit(91) ? static_cast<int>(w.field(store ? 64 : 32, ureg_bits(ins.sm))) : -1;
  const Operand addr = mem_addr(ra, false, strides[stride], ur, w.sfield(40, 24), ins.sm);
  ins.f[3] = stride;
  if (store) {
    ins.src.push_back(addr);
    ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8)), mem_regs(size)));
  } else {
    ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), mem_regs(size)));
    ins.src.push_back(addr);
  }
}

// LDC: a constant bank indexed by a register, c[bank][Ra + offset].
void dec_ldc(Instr& ins, const Word& w) {
  ins.op = Op::LDC;
  ins.mnemonic = "LDC";
  const unsigned size = static_cast<unsigned>(w.field(73, 3));
  if (*kMemSize[size]) ins.mods.push_back(kMemSize[size]);
  ins.f[0] = size;
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), mem_regs(size)));
  const unsigned ra = static_cast<unsigned>(w.field(24, 8));
  const unsigned bank = static_cast<unsigned>(w.field(54, 5));
  const int64_t off = w.sfield(38, 16);
  std::string t = "c[" + signed_hex(bank) + "][";
  if (ra != kRZ) {
    t += "R" + std::to_string(ra);
    if (off) t += "+" + signed_hex(off);
  } else {
    t += signed_hex(off);
  }
  Operand o = Txt(t + "]");
  o.kind = Kind::Mem;
  o.reg = ra;
  o.imm = off;
  ins.src.push_back(o);
  ins.f[4] = bank;
}

// ---- dispatch ------------------------------------------------------------------

using AluDec = void (*)(Instr&, const Word&, bool uniform);
using Dec = void (*)(Instr&, const Word&);

// ALU ops by the low nine opcode bits; forms come from bits 9-11. The uniform
// datapath's ops are the same with 0x80 set in the low nine bits.
const std::unordered_map<unsigned, AluDec>& alu_table() {
  static const std::unordered_map<unsigned, AluDec> t = {
      {0x02, dec_mov},   {0x07, dec_sel},   {0x0c, dec_isetp}, {0x10, dec_iadd3}, {0x11, dec_lea},
      {0x12, dec_lop3},  {0x16, dec_prmt},  {0x17, dec_imnmx}, {0x19, dec_shf},   {0x24, dec_imad},
      {0x25, dec_imad},  {0x27, dec_imad},
  };
  return t;
}

const std::unordered_map<unsigned, Dec>& float_table() {
  static const std::unordered_map<unsigned, Dec> t = {
      {0x20, dec_fmul},
      {0x21, dec_fadd},
      {0x23, dec_ffma},
  };
  return t;
}

// Everything else by its full twelve-bit opcode.
const std::unordered_map<unsigned, Dec>& fixed_table() {
  static const std::unordered_map<unsigned, Dec> t = {
      {0x918, dec_nop},  {0x919, [](Instr& i, const Word& w) { dec_s2r(i, w, false); }},
      {0x9c3, [](Instr& i, const Word& w) { dec_s2r(i, w, true); }},
      {0x941, dec_bsync}, {0x945, dec_bssy}, {0x947, dec_bra}, {0x94d, dec_exit},
      {0xab9, dec_uldc},
      {0x980, [](Instr& i, const Word& w) { dec_gmem(i, w, Op::LD, "LD", false); }},
      {0x981, [](Instr& i, const Word& w) { dec_gmem(i, w, Op::LDG, "LDG", false); }},
      {0x985, [](Instr& i, const Word& w) { dec_gmem(i, w, Op::ST, "ST", true); }},
      {0x986, [](Instr& i, const Word& w) { dec_gmem(i, w, Op::STG, "STG", true); }},
      {0x983, [](Instr& i, const Word& w) { dec_smem(i, w, Op::LDL, "LDL", false, false); }},
      {0x984, [](Instr& i, const Word& w) { dec_smem(i, w, Op::LDS, "LDS", false, true); }},
      {0x987, [](Instr& i, const Word& w) { dec_smem(i, w, Op::STL, "STL", true, false); }},
      {0x988, [](Instr& i, const Word& w) { dec_smem(i, w, Op::STS, "STS", true, true); }},
      {0xb82, dec_ldc},
      // The same without the uniform-register field.
      {0x380, [](Instr& i, const Word& w) { dec_gmem(i, w, Op::LD, "LD", false); }},
      {0x381, [](Instr& i, const Word& w) { dec_gmem(i, w, Op::LDG, "LDG", false); }},
      {0x385, [](Instr& i, const Word& w) { dec_gmem(i, w, Op::ST, "ST", true); }},
      {0x386, [](Instr& i, const Word& w) { dec_gmem(i, w, Op::STG, "STG", true); }},
      {0x383, [](Instr& i, const Word& w) { dec_smem(i, w, Op::LDL, "LDL", false, false); }},
      {0x384, [](Instr& i, const Word& w) { dec_smem(i, w, Op::LDS, "LDS", false, true); }},
      {0x387, [](Instr& i, const Word& w) { dec_smem(i, w, Op::STL, "STL", true, false); }},
      {0x388, [](Instr& i, const Word& w) { dec_smem(i, w, Op::STS, "STS", true, true); }},
  };
  return t;
}

}  // namespace

Instr decode(const Word& w, uint64_t pc, int sm) {
  Instr ins;
  ins.w = w;
  ins.pc = pc;
  ins.sm = sm;
  ins.guard = static_cast<uint8_t>(w.field(12, 3));
  ins.guard_not = w.bit(15);
  const unsigned opc = static_cast<unsigned>(w.field(0, 12));
  if (const auto it = fixed_table().find(opc); it != fixed_table().end()) {
    it->second(ins, w);
    return ins;
  }
  const unsigned low9 = opc & 0x1ff;
  // ALU ops keep bit 8 clear; the form in bits 9-11 is never zero for them.
  if (w.field(9, 3) != 0 && (low9 & 0x100) == 0) {
    const bool uniform = (low9 & 0x80) != 0;
    if (const auto it = alu_table().find(low9 & 0x7f); it != alu_table().end()) {
      it->second(ins, w, uniform);
      ins.guard_uniform = uniform;   // the uniform datapath's guards are UP<n>
      return ins;
    }
    if (const auto it = float_table().find(low9); it != float_table().end()) {
      it->second(ins, w);
      return ins;
    }
  }
  char b[64];
  std::snprintf(b, sizeof b, "SASS: unknown opcode 0x%03x (sm_%d)", opc, sm);
  throw Error(Err::UnsupportedPtx, b);
}

// ---- printing ----------------------------------------------------------------

namespace {

std::string reg_text(const Operand& o, int sm) {
  switch (o.kind) {
    case Kind::Reg: return o.reg == kRZ ? "RZ" : "R" + std::to_string(o.reg);
    case Kind::UReg: return o.reg == urz(sm) ? "URZ" : "UR" + std::to_string(o.reg);
    case Kind::Pred: return o.reg == kPT ? "PT" : "P" + std::to_string(o.reg);
    case Kind::UPred: return o.reg == kPT ? "UPT" : "UP" + std::to_string(o.reg);
    default: return "";
  }
}

std::string imm_text(int64_t v) {
  char b[32];
  if (v < 0) {
    std::snprintf(b, sizeof b, "-0x%llx", static_cast<unsigned long long>(-v));
  } else {
    std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(v));
  }
  return b;
}

std::string operand_text(const Operand& o, int sm) {
  std::string s;
  switch (o.kind) {
    case Kind::Reg:
    case Kind::UReg:
      s = reg_text(o, sm);
      if (o.reuse) s += ".reuse";
      break;
    case Kind::Pred:
    case Kind::UPred:
      return (o.neg ? "!" : "") + reg_text(o, sm) + o.suffix;
    case Kind::Imm: s = imm_text(o.imm); break;
    case Kind::FImm: s = o.text; break;
    case Kind::CBank: s = "c[" + imm_text(o.reg) + "][" + imm_text(o.imm) + "]"; break;
    case Kind::UCBank: s = "cx[UR" + std::to_string(o.reg) + "][" + imm_text(o.imm) + "]"; break;
    case Kind::Label: s = imm_text(o.imm); break;
    case Kind::Bar: s = "B" + std::to_string(o.reg); break;
    case Kind::Mem:
    case Kind::SReg:
    case Kind::Text: s = o.text; break;
  }
  s += o.suffix;
  if (o.abs) s = "|" + s + "|";
  if (o.neg && o.kind != Kind::Pred && o.kind != Kind::UPred) s = "-" + s;
  if (o.bnot) s = "~" + s;
  return s;
}

}  // namespace

std::string to_text(const Instr& ins) {
  std::string s;
  if (ins.guard != kPT || ins.guard_not) {
    s += "@";
    if (ins.guard_not) s += "!";
    const std::string file = ins.guard_uniform ? "UP" : "P";
    s += (ins.guard == kPT ? file + "T" : file + std::to_string(ins.guard)) + " ";
  }
  s += ins.mnemonic;
  for (const std::string& m : ins.mods) s += "." + m;
  bool first = true;
  for (const auto* list : {&ins.dst, &ins.src})
    for (const Operand& o : *list) {
      s += first ? " " : ", ";
      first = false;
      s += operand_text(o, ins.sm);
    }
  return s;
}

}  // namespace vgpu::sass
