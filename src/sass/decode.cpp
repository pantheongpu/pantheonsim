// SASS decoder for the sm_70-and-later 128-bit format. Field positions come
// from Mesa NAK's encoder (src/nouveau/compiler/nak/sm70_encode.rs, MIT) and,
// where NAK is silent, from NVIDIA's own disassembler: every decoder here is
// checked against nvdisasm's text for a corpus of real instructions
// (nvidia/tests/data/sass/, test_sass_decode).
#include "vgpu/sass/sass.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <cstring>
#include <unordered_map>
#include <vector>

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
// A float or double as nvdisasm prints it: twenty significant digits, and
// from 2^31 up in exponent form with twenty after the point.
std::string float_text(double d) {
  char b[48];
  if (d == __builtin_inf() || d == -__builtin_inf()) return d < 0 ? "-INF" : "+INF";
  if (d == 0 && std::signbit(d)) return "-0.0";
  if (std::fabs(d) >= 2147483648.0) std::snprintf(b, sizeof b, "%.20e", d);   // past int32's range
  else std::snprintf(b, sizeof b, "%.20g", d);
  return b;
}

Operand FImm32(uint32_t bits) {
  Operand o;
  o.kind = Kind::FImm;
  o.fbits = bits;
  o.fwidth = 32;
  float f;
  std::memcpy(&f, &bits, 4);
  if (f != f) o.text = std::string((bits >> 31) ? "-" : "+") + "QNAN";   // nvdisasm's word for any NaN
  else o.text = float_text(f);
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
  unsigned low9 = static_cast<unsigned>(w.field(0, 9) & 0x7f);   // 0x24 IMAD, 0x25 WIDE, 0x27 HI
  if (low9 == 0x26) low9 = 0x27;   // sm_120's UIMAD.HI
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
    } else if (!uniform && b.kind == Kind::Imm && b.imm == 1 && !x) {
      ins.mods.push_back("IADD");
      if (!is_signed) ins.mods.push_back("U32");
    } else if (!uniform && b.kind == Kind::Imm && c_rz && !is_signed && !x && b.imm > 0 &&
               b.imm != 0x10000 && (b.imm & (b.imm - 1)) == 0) {   // nvdisasm's one exception
      ins.mods.push_back("SHL");
      ins.mods.push_back("U32");
    } else if (!is_signed) {
      ins.mods.push_back("U32");
    }
  }
  if (x) ins.mods.push_back("X");
  if (pdst != kPT) ins.dst.push_back(uniform ? UP(pdst) : P(pdst));
  if (x) ins.src.push_back(pred_src(w, 87, 90, uniform));
}

void dec_iadd3(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::UIADD3 : Op::IADD3;
  ins.mnemonic = uniform ? "UIADD3" : "IADD3";
  const bool x = w.bit(74);
  if (x) ins.mods.push_back("X");
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  const unsigned p0 = static_cast<unsigned>(w.field(81, 3)), p1 = static_cast<unsigned>(w.field(84, 3));
  // The carry-outs, when not PT (with .X too: a chained wider add).
  // From sm_100 nvdisasm prints both even when PT.
  if (p0 != kPT || p1 != kPT || ins.sm >= 100) ins.dst.push_back(uniform ? UP(p0) : P(p0));
  if (p1 != kPT || ins.sm >= 100) ins.dst.push_back(uniform ? UP(p1) : P(p1));
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
  const bool wide = ins.sm >= 120 && w.bit(80);   // sm_120: a 64-bit compare
  if (wide) ins.mods.push_back(is_signed ? "S64" : "U64");
  else if (!is_signed) ins.mods.push_back("U32");
  ins.mods.push_back(kBoolOp[bop]);
  if (ex) ins.mods.push_back("EX");
  ins.f[4] = wide;
  const unsigned p0 = static_cast<unsigned>(w.field(81, 3)), p1 = static_cast<unsigned>(w.field(84, 3));
  ins.dst.push_back(uniform ? UP(p0) : P(p0));
  ins.dst.push_back(uniform ? UP(p1) : P(p1));
  alu2(ins, w, kSigned, uniform);
  if (wide)
    for (Operand& o : ins.src)
      if (o.kind == Kind::Reg || o.kind == Kind::UReg) o.width = 2;
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
  // sm_120: the forms naming the third slot select 64-bit values (vector
  // 2, 3, 7; uniform 2, 6), with b an immediate (sign-extended), a
  // register at 32 or a uniform one there.
  const unsigned form = static_cast<unsigned>(w.field(9, 3));
  const bool wide = ins.sm >= 120 && (uniform ? form == 2 || form == 6 : form == 2 || form == 3 || form == 7);
  if (wide) ins.mods.push_back("64");
  ins.f[0] = wide;
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  if (wide) {
    ins.src.push_back(alu_src(w, Slot::R24, ins.sm, kSigned, uniform));
    const Slot b = form == 2 ? Slot::Imm32 : (form == 3 && !uniform) ? Slot::R32 : Slot::UR32;
    ins.src.push_back(alu_src(w, b, ins.sm, kSigned, uniform));
    mark_reuse(ins, w);
  } else {
    alu2(ins, w, kUnsigned, uniform);
  }
  if (wide)
    for (Operand& o : ins.src) {
      if (o.kind == Kind::Imm) o.fwidth = 64;   // sign-extended to 64 bits
      else o.width = 2;
    }
  ins.src.push_back(pred_src(w, 87, 90, uniform));
}

void dec_mov(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::UMOV : Op::MOV;
  ins.mnemonic = uniform ? "UMOV" : "MOV";
  if (!uniform && ins.sm >= 100 && w.bit(88)) ins.mods.push_back("SPILL");   // a uniform register spilled to a vector one
  // sm_120: form 2 moves a 64-bit immediate, held at 24-87; 80 a register pair.
  if (ins.sm >= 120 && (w.field(9, 3) == 2 || w.bit(80))) {
    ins.mods.push_back("64");
    ins.f[1] = 1;
    Operand d = dst_reg(w, uniform, ins.sm);
    d.width = 2;
    ins.dst.push_back(d);
    if (w.field(9, 3) == 2) {
      const uint64_t v = w.field(24, 64);
      Operand o = Imm(static_cast<int64_t>(v));
      o.width = 2;
      o.fwidth = 64;
      ins.src.push_back(o);
    } else {
      const AluForm f = alu_form(static_cast<unsigned>(w.field(9, 3)));
      ins.src.push_back(alu_src(w, f.s1 == Slot::R64 ? f.s2 : f.s1, ins.sm, kUnsigned, uniform));
      ins.src.back().width = 2;
      if (ins.src.back().kind == Kind::Reg) ins.src.back().reuse = reuse(w, 1);
    }
    return;
  }
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  const AluForm f = alu_form(static_cast<unsigned>(w.field(9, 3)));
  ins.src.push_back(alu_src(w, f.s1 == Slot::R64 ? f.s2 : f.s1, ins.sm, kUnsigned, uniform));
  if (ins.src[0].kind == Kind::Reg) ins.src[0].reuse = reuse(w, 1);   // it is operand b
  const unsigned lanes = static_cast<unsigned>(w.field(72, 4));
  if (!uniform && lanes != 0xf) ins.src.push_back(Imm(lanes));
  ins.f[0] = lanes;
}

// sm_120's IMNMX (0x17) and UIMNMX (0x85): 64-bit, with two predicate
// results (81, 84) and two predicate sources (87-90 choosing the min, and
// 77-80).
void dec_imnmx64(Instr& ins, const Word& w, bool uniform) {
  ins.op = Op::IMNMX;
  ins.mnemonic = uniform ? "UIMNMX" : "IMNMX";
  ins.mods.push_back(w.bit(73) ? "S64" : "U64");
  ins.f[0] = w.bit(73);
  ins.f[1] = 1;   // 64-bit
  const auto p = [&](unsigned n) { return uniform ? UP(n) : P(n); };
  ins.dst.push_back(p(static_cast<unsigned>(w.field(81, 3))));
  ins.dst.push_back(p(static_cast<unsigned>(w.field(84, 3))));
  Operand d = dst_reg(w, uniform, ins.sm);
  d.width = 2;
  ins.dst.push_back(d);
  alu2(ins, w, kSigned, uniform);
  for (Operand& o : ins.src)
    if (o.kind == Kind::Reg || o.kind == Kind::UReg) o.width = 2;
  ins.src.push_back(pred_src(w, 87, 90, uniform));
  ins.src.push_back(pred_src(w, 77, 80, uniform));
}

void dec_imnmx(Instr& ins, const Word& w, bool uniform) {
  if (ins.sm >= 120) {
    dec_imnmx64(ins, w, uniform);
    return;
  }
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
  // 84-86: the product times 2^(field - 4): 4 is none, 1-3 .D8 .D4 .D2
  // (seen), 5-7 .M2 .M4 .M8 by symmetry.
  static const char* const scales[] = {"(0)", "D8", "D4", "D2", "", "M2", "M4", "M8"};
  const unsigned scale = static_cast<unsigned>(w.field(84, 3));
  if (scale != 4) ins.mods.push_back(scales[scale]);
  ins.f[4] = scale;
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
  // FADD is a*1 + b: its second register reuses as operand c.
  if (ins.src[1].kind == Kind::Reg) ins.src[1].reuse = reuse(w, 2);
  float_srcs(ins, w);
}

// ---- control -----------------------------------------------------------------

// A relative branch target: a signed offset in bytes from the next
// instruction, held in bits 34-81 (the low two bits are always zero).
// A branch offset in bytes. From sm_90 its low eight bits (in words of
// four bytes) moved to 16-23, below the rest at 34-81.
int64_t branch_offset(const Word& w, int sm) {
  if (sm >= 90) return static_cast<int64_t>(static_cast<uint64_t>(w.sfield(34, 48)) << 8 | w.field(16, 8)) * 4;
  return w.sfield(34, 48) * 4;
}

uint64_t branch_target(const Word& w, uint64_t pc, int sm) {
  return static_cast<uint64_t>(static_cast<int64_t>(pc) + 16 + branch_offset(w, sm));
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
  if (mode == 1) ins.mods.push_back("U");   // the warp is known to agree
  if (w.bit(84)) ins.mods.push_back("ANY");  // taken by all when any thread's guard holds
  ins.f[1] = w.bit(84);
  if (mode == 2) ins.mods.push_back("DIV");
  if (mode == 3) ins.mods.push_back("CONV");
  const unsigned cond = static_cast<unsigned>(w.field(87, 3));
  if (cond != kPT || w.bit(90)) ins.src.push_back(pred_src(w, 87, 90));
  if (mode >= 2 && ins.sm >= 80) {   // Turing's take no uniform register
    Operand u = UR(static_cast<unsigned>(w.field(24, ureg_bits(ins.sm))), ins.sm);
    u.bnot = w.bit(30);
    ins.src.push_back(u);
  }
  ins.src.push_back(label(branch_target(w, ins.pc, ins.sm)));
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
  if (ins.sm >= 100 && w.bit(72)) ins.mods.push_back("RELIABLE");
  if (ins.sm >= 100 && w.bit(73)) ins.mods.push_back("RECONVERGENT");
  Operand b;
  b.kind = Kind::Bar;
  b.reg = static_cast<unsigned>(w.field(16, 4));
  ins.dst.push_back(b);
  // The old layout, ending at 71 from sm_100 (73 is a flag there).
  const int64_t off = ins.sm >= 100 ? w.sfield(34, 38) * 4 : w.sfield(34, 48) * 4;
  ins.src.push_back(label(static_cast<uint64_t>(static_cast<int64_t>(ins.pc) + 16 + off)));
}

void dec_bsync(Instr& ins, const Word& w) {
  ins.op = Op::BSYNC;
  ins.mnemonic = "BSYNC";
  if (ins.sm >= 100 && w.bit(72)) ins.mods.push_back("RELIABLE");
  if (ins.sm >= 100 && w.bit(73)) ins.mods.push_back("RECONVERGENT");
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
    case 0x88: return "SR_CgaCtaId";   // sm_90: the block's rank in its cluster
    case 0x8a: return "SR_CgaSize";
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

const char* const kMemSize[] = {"U8", "S8", "U16", "S16", "", "64", "128", "U.128"};

unsigned mem_regs(unsigned size) { return size >= 6 ? 4 : size == 5 ? 2 : 1; }

void mem_order_mods(Instr& ins, const Word& w) {
  const unsigned order = static_cast<unsigned>(w.field(77, 4));
  ins.f[1] = order;
  if (ins.sm < 80) {
    // Turing always names the scope; a weak access is .SYS.
    switch (order) {
      case 0x3: ins.mods.push_back("CONSTANT"); ins.mods.push_back("SYS"); break;
      case 0x7: ins.mods.push_back("SYS"); break;
      case 0x8: ins.mods.push_back("STRONG"); ins.mods.push_back("CTA"); break;
      case 0xa: ins.mods.push_back("STRONG"); ins.mods.push_back("GPU"); break;
      case 0xb: ins.mods.push_back("STRONG"); ins.mods.push_back("SYS"); break;
      default: ins.mods.push_back("ORDER" + std::to_string(order)); break;
    }
    return;
  }
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
    if (t.empty()) t = off == 0 && ra == kRZ ? "RZ" : signed_hex(off);
    else t += "+" + signed_hex(off);   // nvdisasm writes a negative one as +-0x80
  }
  o.text = "[" + t + "]";
  o.reg = ra;
  o.imm = off;
  return o;
}

// Turing's global addresses: a uniform register (91) is part of the
// address, the register a 64-bit one (90) or a 32-bit offset (.U32) beside
// it.
Operand sm75_gaddr(const Word& w, unsigned ur_pos, int64_t off, int sm) {
  const unsigned ra = static_cast<unsigned>(w.field(24, 8));
  const bool has_ur = w.bit(91);
  const int ur = has_ur ? static_cast<int>(w.field(ur_pos, ureg_bits(sm))) : -1;
  const char* suffix = !has_ur ? "" : w.bit(90) ? ".64" : ".U32";
  return mem_addr(ra, false, suffix, ur, off, sm);
}

// An atomic's offset: 40-63, from sm_100 40-62 (63 is something else).
int64_t atom_off(const Word& w, int sm) { return sm >= 100 ? w.sfield(40, 23) : w.sfield(40, 24); }

// From sm_90 nvdisasm prints an .E access's memory descriptor, the uniform
// register pair holding it (earlier architectures have it too, unprinted).
void add_desc(Operand& a, const Word& w, unsigned pos, int sm) {
  if (sm < 90) return;
  a.text = "desc[UR" + std::to_string(w.field(pos, ureg_bits(sm))) + "]" + a.text;
}

// Global and generic loads/stores: LDG, STG, LD, ST.
void dec_gmem(Instr& ins, const Word& w, Op op, const char* name, bool store) {
  ins.op = op;
  ins.mnemonic = name;
  const bool e = w.bit(72);
  const unsigned size = static_cast<unsigned>(w.field(73, 3));
  if (e) ins.mods.push_back("E");
  evict_mods(ins, w);
  // 68-69: an L2 prefetch of the surrounding 64, 128 or 256 bytes.
  static const char* const ltc[] = {"", "LTC64B", "LTC128B", "LTC256B"};
  if (!store && w.field(68, 2)) ins.mods.push_back(ltc[w.field(68, 2)]);
  if (*kMemSize[size]) ins.mods.push_back(kMemSize[size]);
  mem_order_mods(ins, w);
  ins.f[0] = size;
  const unsigned ra = static_cast<unsigned>(w.field(24, 8));
  const bool ra64 = w.bit(90);
  // Bit 91 says the uniform register field is in use (the 0x3xx forms of
  // these opcodes have none).
  const int ur = w.bit(91) ? static_cast<int>(w.field(store ? 64 : 32, ureg_bits(ins.sm))) : -1;
  // Turing's generic LD/ST keep the offset at 32 unless a uniform register
  // is there (91), and a store's data at 64.
  const bool t_generic = op == Op::LD || op == Op::ST;   // generic LD/ST, every architecture
  const int64_t off = t_generic && !w.bit(91) ? w.sfield(32, 24) : w.sfield(40, 24);
  // With .E the uniform register is the descriptor, not part of the address.
  const bool t_moved = t_generic && !w.bit(91);   // offset at 32, data at 64
  Operand addr = ins.sm < 80 ? sm75_gaddr(w, store ? 64 : 32, off, ins.sm)
                             : mem_addr(ra, ra64, "", e ? -1 : ur, off, ins.sm);
  if (e && w.bit(91)) add_desc(addr, w, store ? 64 : 32, ins.sm);
  if (store) {
    ins.src.push_back(addr);
    ins.src.push_back(R(static_cast<unsigned>(w.field(t_moved ? 64 : 32, 8)), mem_regs(size)));
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
  if (!shared) evict_mods(ins, w);
  if (shared && w.bit(76)) ins.mods.push_back("U");   // Turing: a uniform-address load
  if (*kMemSize[size]) ins.mods.push_back(kMemSize[size]);
  ins.f[0] = size;
  const unsigned ra = static_cast<unsigned>(w.field(24, 8));
  static const char* const strides[] = {"", ".X4", ".X8", ".X16"};
  const unsigned stride = shared ? static_cast<unsigned>(w.field(78, 2)) : 0;
  const int ur = w.bit(91) ? static_cast<int>(w.field(store ? 64 : 32, ureg_bits(ins.sm))) : -1;
  const Operand addr = mem_addr(ra, false, strides[stride], ur, w.sfield(40, 24), ins.sm, ins.sm >= 100 && ur >= 0);
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
    t += off == 0 && ins.sm >= 90 ? "RZ" : signed_hex(off);   // Hopper names the zero register
  }
  Operand o = Txt(t + "]");
  o.kind = Kind::Mem;
  o.reg = ra;
  o.imm = off;
  ins.src.push_back(o);
  ins.f[4] = bank;
}

void one_src(Instr& ins, const Word& w, ImmStyle imm, bool uniform);

// ---- half precision ----------------------------------------------------------------
//
// Packed-half ops read each source as two halves; a register source can be
// swizzled (.H0_H0 takes the low half for both, .H1_H1 the high, .F32 reads a
// 32-bit float). Swizzle fields: src0 74-75, the second slot 60-61, the third
// 81-82; the third slot's abs/neg move to 83/84 for these ops. An immediate
// holds two halves, printed as two operands, high half first.

// A half as nvdisasm prints it: its exact value to twenty significant digits.
std::string half_text(uint16_t h) {
  const unsigned sign = h >> 15, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
  char b[48];
  if (exp == 0x1f) {
    std::snprintf(b, sizeof b, "%s%s", sign ? "-" : "+", man ? "QNAN" : "INF");
    return b;
  }
  double v = exp ? (1.0 + man / 1024.0) * std::ldexp(1.0, static_cast<int>(exp) - 15)
                 : man / 1024.0 * std::ldexp(1.0, -14);
  if (sign) v = -v;
  return float_text(v);
}

const char* const kSwz[] = {"", "F32", "H0_H0", "H1_H1"};

void half_src_mods(Instr& ins, const Word& w) {
  for (Operand& o : ins.src) {
    switch (static_cast<Slot>(o.slot)) {
      case Slot::R24:
        o.neg = w.bit(72);
        o.abs = w.bit(73);
        if (o.kind == Kind::Reg && w.field(74, 2)) o.suffix = std::string(".") + kSwz[w.field(74, 2)];
        break;
      case Slot::R32:
      case Slot::CB:
      case Slot::UR32:
        o.abs = w.bit(62);
        o.neg = w.bit(63);
        if (w.field(60, 2)) o.suffix = std::string(".") + kSwz[w.field(60, 2)];
        break;
      case Slot::R64:
        o.abs = w.bit(83);
        o.neg = w.bit(84);
        if (w.field(81, 2)) o.suffix = std::string(".") + kSwz[w.field(81, 2)];
        break;
      case Slot::Imm32: break;
    }
  }
}

std::string bf16_text(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  char b[48];
  if (f != f) std::snprintf(b, sizeof b, "%sQNAN", (h >> 15) ? "-" : "+");
  else return float_text(f);
  return b;
}

// Replaces a 32-bit immediate source with its two halves (bf16s with
// .BF16_V2).
void split_half_imm(Instr& ins, bool bf16 = false) {
  for (size_t i = 0; i < ins.src.size(); ++i) {
    if (static_cast<Slot>(ins.src[i].slot) != Slot::Imm32) continue;
    const uint32_t v = static_cast<uint32_t>(ins.src[i].kind == Kind::FImm ? ins.src[i].fbits : ins.src[i].imm);
    const auto text = [&](uint16_t h) { return bf16 ? bf16_text(h) : half_text(h); };
    Operand hi = Txt(text(static_cast<uint16_t>(v >> 16)));
    Operand lo = Txt(text(static_cast<uint16_t>(v)));
    hi.kind = lo.kind = Kind::FImm;
    hi.fbits = v >> 16;
    lo.fbits = v & 0xffff;
    hi.fwidth = lo.fwidth = 16;
    hi.slot = lo.slot = static_cast<uint8_t>(Slot::Imm32);
    ins.src[i] = hi;
    ins.src.insert(ins.src.begin() + static_cast<long>(i) + 1, lo);
    ++i;
  }
}

void half_common(Instr& ins, const Word& w, bool dnz, bool relu) {
  if (w.bit(85)) ins.mods.push_back("BF16_V2");
  if (w.bit(78)) ins.mods.push_back("F32");
  if (w.bit(80)) ins.mods.push_back("FTZ");
  if (dnz && w.bit(76)) ins.mods.push_back("DNZ");
  if (w.bit(77)) ins.mods.push_back("SAT");
  if (relu && w.bit(79)) ins.mods.push_back("RELU");
  ins.f[0] = w.bit(85);
  ins.f[1] = w.bit(78);
  ins.f[2] = w.bit(80);
  ins.f[3] = w.bit(77);
  ins.f[5] = relu && w.bit(79);
}

void dec_hadd2(Instr& ins, const Word& w) {
  ins.op = Op::HADD2;
  ins.mnemonic = "HADD2";
  half_common(ins, w, false, false);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu2(ins, w, kUnsigned);
  // HADD2 computes a*1 + b, and its second register is operand c for reuse.
  if (ins.src[1].kind == Kind::Reg) ins.src[1].reuse = reuse(w, 2);
  half_src_mods(ins, w);
  split_half_imm(ins, w.bit(85));
}

void dec_hmul2(Instr& ins, const Word& w) {
  ins.op = Op::HMUL2;
  ins.mnemonic = "HMUL2";
  half_common(ins, w, true, true);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu2(ins, w, kUnsigned);
  half_src_mods(ins, w);
  split_half_imm(ins, w.bit(85));
}

void dec_hfma2(Instr& ins, const Word& w) {
  ins.op = Op::HFMA2;
  ins.mnemonic = "HFMA2";
  half_common(ins, w, true, true);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu3(ins, w, kUnsigned);
  half_src_mods(ins, w);
  split_half_imm(ins, w.bit(85));
}

const char* const kFloatCmp[] = {"F",  "LT",  "EQ",  "LE",  "GT",  "NE",  "GE",  "NUM",
                                 "NAN", "LTU", "EQU", "LEU", "GTU", "NEU", "GEU", "T"};

void dec_hset2(Instr& ins, const Word& w, bool setp) {
  ins.op = setp ? Op::HSETP2 : Op::HSET2;
  ins.mnemonic = setp ? "HSETP2" : "HSET2";
  if (w.bit(65)) ins.mods.push_back("BF16_V2");
  if (!setp && w.bit(71)) ins.mods.push_back("BF");
  ins.mods.push_back(kFloatCmp[w.field(76, 4)]);
  if (w.bit(80)) ins.mods.push_back("FTZ");
  ins.mods.push_back(kBoolOp[w.field(69, 2)]);
  ins.f[0] = static_cast<uint32_t>(w.field(76, 4));
  ins.f[1] = static_cast<uint32_t>(w.field(69, 2));
  ins.f[2] = !setp && w.bit(71);   // .BF: 1.0 for true, else a mask of ones
  ins.f[3] = w.bit(65);            // bfloat16 pairs
  ins.f[4] = w.bit(80);
  if (setp) {
    ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
    ins.dst.push_back(P(static_cast<unsigned>(w.field(84, 3))));
  } else {
    ins.dst.push_back(dst_reg(w, false, ins.sm));
  }
  alu2(ins, w, kUnsigned);
  // Like HADD2, the second register reuses as operand c.
  if (ins.src[1].kind == Kind::Reg) ins.src[1].reuse = reuse(w, 2);
  half_src_mods(ins, w);
  split_half_imm(ins, w.bit(85));
  ins.src.push_back(pred_src(w, 87, 90));
}

// ---- double precision ------------------------------------------------------------

// A 64-bit float immediate is the top 32 bits of the double.
void double_imm(Instr& ins) {
  for (Operand& o : ins.src) {
    if (static_cast<Slot>(o.slot) != Slot::Imm32) continue;
    const uint64_t bits = static_cast<uint64_t>(o.kind == Kind::FImm ? o.fbits : static_cast<uint32_t>(o.imm)) << 32;
    double d;
    std::memcpy(&d, &bits, 8);
    o.kind = Kind::FImm;
    if (d != d) o.text = std::string((bits >> 63) ? "-" : "+") + "QNAN";
    else o.text = float_text(d);
    o.fwidth = 64;
  }
}

void dec_dop(Instr& ins, const Word& w, Op op, const char* name, int nsrc) {
  ins.op = op;
  ins.mnemonic = name;
  const unsigned rnd = static_cast<unsigned>(w.field(78, 2));
  if (rnd) ins.mods.push_back(kRnd[rnd]);
  ins.f[0] = rnd;
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), 2));
  if (nsrc == 3) alu3(ins, w, kFloat);
  else if (op == Op::DADD) {
    // DADD is a + c: a register second source is in the third slot (Mesa
    // NAK encodes it so, and nvdisasm agrees).
    const AluForm f = alu_form(static_cast<unsigned>(w.field(9, 3)));
    ins.src.push_back(alu_src(w, Slot::R24, ins.sm, kFloat));
    ins.src.push_back(alu_src(w, f.s1 == Slot::R32 || f.s1 == Slot::R64 ? f.s2 : f.s1, ins.sm, kFloat));
    mark_reuse(ins, w);
  } else alu2(ins, w, kFloat);
  // DADD is a + c: its second register reuses as operand c.
  if (op == Op::DADD && ins.src[1].kind == Kind::Reg) ins.src[1].reuse = reuse(w, 2);
  float_srcs(ins, w);
  double_imm(ins);
}

void dec_dsetp(Instr& ins, const Word& w) {
  ins.op = Op::DSETP;
  ins.mnemonic = "DSETP";
  // Comparisons 0 and 15 (never, always) are MIN and MAX here: the first
  // predicate picks a over b, the second says both are NaN.
  const unsigned cmp = static_cast<unsigned>(w.field(76, 4));
  ins.mods.push_back(cmp == 0 ? "MIN" : cmp == 15 ? "MAX" : kFloatCmp[cmp]);
  ins.mods.push_back(kBoolOp[w.field(74, 2)]);
  ins.f[0] = static_cast<uint32_t>(w.field(76, 4));
  ins.f[1] = static_cast<uint32_t>(w.field(74, 2));
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
  ins.dst.push_back(P(static_cast<unsigned>(w.field(84, 3))));
  alu2(ins, w, kFloat);
  if (ins.src[1].kind == Kind::Reg) ins.src[1].reuse = reuse(w, 2);   // as DADD
  float_srcs(ins, w);
  double_imm(ins);
  ins.src.push_back(pred_src(w, 87, 90));
}

void dec_fsetp(Instr& ins, const Word& w) {
  ins.op = Op::FSETP;
  ins.mnemonic = "FSETP";
  ins.mods.push_back(kFloatCmp[w.field(76, 4)]);
  if (w.bit(80)) ins.mods.push_back("FTZ");
  ins.mods.push_back(kBoolOp[w.field(74, 2)]);
  ins.f[0] = static_cast<uint32_t>(w.field(76, 4));
  ins.f[1] = static_cast<uint32_t>(w.field(74, 2));
  ins.f[2] = w.bit(80);
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
  ins.dst.push_back(P(static_cast<unsigned>(w.field(84, 3))));
  alu2(ins, w, kFloat);
  float_srcs(ins, w);
  ins.src.push_back(pred_src(w, 87, 90));
}

// FSET: 1.0 or 0.0 (.BF, the only form ptxas emits) from a comparison.
void dec_fset(Instr& ins, const Word& w) {
  ins.op = Op::FSET;
  ins.mnemonic = "FSET";
  ins.mods.push_back("BF");
  ins.mods.push_back(kFloatCmp[w.field(76, 4)]);
  if (w.bit(80)) ins.mods.push_back("FTZ");
  ins.mods.push_back(kBoolOp[w.field(74, 2)]);
  ins.f[0] = static_cast<uint32_t>(w.field(76, 4));
  ins.f[1] = static_cast<uint32_t>(w.field(74, 2));
  ins.f[2] = w.bit(80);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu2(ins, w, kFloat);
  float_srcs(ins, w);
  ins.src.push_back(pred_src(w, 87, 90));
}

void dec_fsel(Instr& ins, const Word& w) {
  ins.op = Op::FSEL;
  ins.mnemonic = "FSEL";
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu2(ins, w, kFloat);
  float_srcs(ins, w);
  ins.src.push_back(pred_src(w, 87, 90));
}

// FRND (0x107, 32-bit; 0x113 with a 64-bit side): round to an integral
// value. 78-79 the rounding (none: to nearest even), 75-76 the result's
// size and 84-85 the source's (log2 bytes), 80 FTZ.
void dec_frnd(Instr& ins, const Word& w) {
  ins.op = Op::FRND;
  ins.mnemonic = "FRND";
  const unsigned dsz = static_cast<unsigned>(w.field(75, 2)), ssz = static_cast<unsigned>(w.field(84, 2));
  if (dsz == 3 && ssz == 3) ins.mods.push_back("F64");
  else if (dsz != 2 || ssz != 2) {
    ins.mods.push_back(dsz == 3 ? "F64" : dsz == 1 ? "F16" : "F32");
    ins.mods.push_back(ssz == 3 ? "F64" : ssz == 1 ? "F16" : "F32");
  }
  static const char* const rnd[] = {"", "FLOOR", "CEIL", "TRUNC"};
  const unsigned r = static_cast<unsigned>(w.field(78, 2));
  if (w.bit(80)) ins.mods.push_back("FTZ");
  if (r) ins.mods.push_back(rnd[r]);
  ins.f[0] = r;
  ins.f[1] = w.bit(80);
  ins.f[2] = dsz;
  ins.f[3] = ssz;
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), dsz == 3 ? 2 : 1));
  one_src(ins, w, kFloat, false);
  if (ssz == 3 && ins.src[0].kind == Kind::Reg) ins.src[0].width = 2;
  src_neg_abs(ins.src[0], w, true);
  if (ssz == 3) double_imm(ins);
}

// FCHK: whether a / b needs the division's slow path.
void dec_fchk(Instr& ins, const Word& w) {
  ins.op = Op::FCHK;
  ins.mnemonic = "FCHK";
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
  alu2(ins, w, kFloat);
  float_srcs(ins, w);
}

// ---- conversions -----------------------------------------------------------------

// Integer types by their 2-bit size (8, 16, 32, 64) and a signed bit.
std::string int_type(unsigned size, bool is_signed) {
  static const char* const bits[] = {"8", "16", "32", "64"};
  return std::string(is_signed ? "S" : "U") + bits[size];
}
std::string float_type(unsigned size) {
  static const char* const bits[] = {"F8", "F16", "F32", "F64"};
  return bits[size];
}

// I2F: integer (84-85 size, 74 signed) to float (75-76 size). nvdisasm leaves
// out the defaults, F32 and S32.
void dec_i2f(Instr& ins, const Word& w) {
  ins.op = Op::I2F;
  ins.mnemonic = "I2F";
  const unsigned dsize = w.field(75, 3) == 4 ? 4 : static_cast<unsigned>(w.field(75, 2));   // 4: BF16
  const unsigned ssize = static_cast<unsigned>(w.field(84, 2));
  const bool sgn = w.bit(74);
  if (dsize != 2) ins.mods.push_back(dsize == 4 ? "BF16" : float_type(dsize));
  if (!(ssize == 2 && sgn)) ins.mods.push_back(int_type(ssize, sgn));
  const unsigned rnd = static_cast<unsigned>(w.field(78, 2));
  if (rnd) ins.mods.push_back(kRnd[rnd]);
  ins.f[0] = dsize;
  ins.f[1] = ssize;
  ins.f[2] = sgn;
  ins.f[3] = rnd;
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), dsize == 3 ? 2 : 1));
  one_src(ins, w, kUnsigned, false);
  if (ssize == 3) ins.src[0].width = 2;
  // A narrow source's part of the register, 60-61: .H1, or .B1-.B3.
  const unsigned part = static_cast<unsigned>(w.field(60, 2));
  if (part && ins.src[0].kind == Kind::Reg) ins.src[0].suffix = (ssize == 1 ? ".H" : ".B") + std::to_string(part);
  ins.f[4] = part;
}

// F2I: float (84-85) to integer (75-76 size, 72 signed); 78-79 the rounding
// (.TRUNC etc.), 77 .NTZ, 80 .FTZ.
void dec_f2i(Instr& ins, const Word& w) {
  ins.op = Op::F2I;
  ins.mnemonic = "F2I";
  const unsigned dsize = static_cast<unsigned>(w.field(75, 2)), ssize = static_cast<unsigned>(w.field(84, 2));
  const bool sgn = w.bit(72);
  if (w.bit(80)) ins.mods.push_back("FTZ");
  if (!(dsize == 2 && sgn)) ins.mods.push_back(int_type(dsize, sgn));
  if (ssize != 2) ins.mods.push_back(float_type(ssize));
  static const char* const rnds[] = {"", "FLOOR", "CEIL", "TRUNC"};
  const unsigned rnd = static_cast<unsigned>(w.field(78, 2));
  if (rnd) ins.mods.push_back(rnds[rnd]);
  if (w.bit(77)) ins.mods.push_back("NTZ");
  ins.f[0] = dsize;
  ins.f[1] = ssize;
  ins.f[2] = sgn;
  ins.f[3] = rnd;
  ins.f[4] = w.bit(80);
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), dsize == 3 ? 2 : 1));
  one_src(ins, w, kFloat, false);
  src_neg_abs(ins.src[0], w, true);
  if (ssize == 3) ins.src[0].width = 2;
}

// F2F: float (84-85) to float (75-76).
void dec_f2f(Instr& ins, const Word& w) {
  ins.op = Op::F2F;
  ins.mnemonic = "F2F";
  const unsigned dsize = static_cast<unsigned>(w.field(75, 2)), ssize = static_cast<unsigned>(w.field(84, 2));
  if (w.bit(80)) ins.mods.push_back("FTZ");
  // The result's type is three bits, 75-77: 4 is BF16.
  const bool bf16 = w.field(75, 3) == 4;
  ins.mods.push_back(bf16 ? "BF16" : float_type(dsize));
  ins.mods.push_back(float_type(ssize));
  const unsigned rnd = static_cast<unsigned>(w.field(78, 2));
  if (rnd) ins.mods.push_back(kRnd[rnd]);
  ins.f[0] = bf16 ? 4 : dsize;
  ins.f[1] = ssize;
  ins.f[3] = rnd;
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), dsize == 3 ? 2 : 1));
  one_src(ins, w, kFloat, false);
  src_neg_abs(ins.src[0], w, true);
  if (ssize == 3) ins.src[0].width = 2;
}

// I2I: saturating integer narrowing (sm_75+): 32-bit source to U8/S8/U16/S16.
void dec_i2i(Instr& ins, const Word& w) {
  ins.op = Op::I2I;
  ins.mnemonic = "I2I";
  static const char* const types[] = {"U8", "S8", "U16", "S16"};
  const unsigned t = static_cast<unsigned>(w.field(76, 2));
  ins.mods.push_back(types[t]);
  ins.mods.push_back("S32");
  ins.mods.push_back("SAT");
  ins.f[0] = t;
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  one_src(ins, w, kUnsigned, false);
}

// F2FP: two floats packed into one register, or (sm_89) FP8 pairs packed
// and unpacked. 76-78 the result's type (0 F16, 1 BF16, 6 E5M2, 7 E4M3),
// 73-74 the source's (0 F32, 2 E5M2, 3 E4M3), 87-89 the form (0 PACK_AB,
// 1 PACK_AB_MERGE_C: the pair into c's low half, 4 UNPACK_B: b's two bytes
// to two halves), 75 .RELU, 79-80 the rounding, 90 .SATFINITE. From sm_89
// nvdisasm names both types.
void dec_f2fp(Instr& ins, const Word& w) {
  ins.op = Op::F2FP;
  ins.mnemonic = "F2FP";
  static const char* const dts[] = {"F16", "BF16", "(2)", "(3)", "(4)", "TF32", "E5M2", "E4M3"};
  static const char* const sts[] = {"F32", "(1)", "E5M2", "E4M3"};
  const unsigned dt = static_cast<unsigned>(w.field(76, 3)), st = static_cast<unsigned>(w.field(73, 2));
  // 89: the one-operand forms (88: its source's upper half), 87: MERGE_C.
  const unsigned mode = w.bit(89) ? 4 : w.bit(87) ? 1 : 0;
  if (w.bit(75)) ins.mods.push_back("RELU");   // negatives become zero
  if (w.bit(90)) ins.mods.push_back("SATFINITE");
  if (ins.sm >= 89) {
    ins.mods.push_back(dts[dt]);
    ins.mods.push_back(sts[st]);
  } else if (dt != 0) {
    ins.mods.push_back(dts[dt]);
  }
  // Form 4 is one operand: b packed from F32 (PACK_B: tf32), else unpacked.
  ins.mods.push_back(mode == 4 ? (st == 0 ? "PACK_B" : "UNPACK_B") : mode == 1 ? "PACK_AB_MERGE_C" : "PACK_AB");
  const unsigned rnd = static_cast<unsigned>(w.field(79, 2));
  if (rnd) ins.mods.push_back(kRnd[rnd]);
  ins.f[0] = dt == 1;
  ins.f[1] = rnd;
  ins.f[2] = w.bit(75);
  ins.f[3] = dt;
  ins.f[4] = st;
  ins.f[5] = mode;
  ins.f[6] = w.bit(90);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  if (mode == 4) {
    one_src(ins, w, kUnsigned, false);
    if (w.bit(88)) ins.src[0].suffix = ".H1";
    ins.f[7] = w.bit(88);
  } else {
    alu2(ins, w, kFloat);
    if (mode == 1) ins.src.push_back(R(static_cast<unsigned>(w.field(64, 8))));
  }
}

// ---- tensor cores ------------------------------------------------------------------

// A sparse MMA also reads the metadata register (40-47) and a selector
// (48-49) saying which threads' metadata the instruction uses.
void sparse_meta(Instr& ins, const Word& w, bool sp) {
  if (!sp) return;
  ins.src.push_back(R(static_cast<unsigned>(w.field(40, 8))));
  ins.src.back().reuse = w.bit(50);
  ins.src.push_back(Imm(w.field(48, 2)));
  ins.f[7] = static_cast<uint32_t>(w.field(48, 2));
}

void mma_regs(Instr& ins, const Word& w, const char* sa = "", const char* sb = "") {
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  Operand a = R(static_cast<unsigned>(w.field(24, 8)));
  Operand b = R(static_cast<unsigned>(w.field(32, 8)));
  Operand c = R(static_cast<unsigned>(w.field(64, 8)));
  a.suffix = sa;
  b.suffix = sb;
  ins.src = {a, b, c};
  mark_reuse(ins, w);
}

// HMMA: 75+78 the shape (0 16x8x8, 1 16x8x16, 2 16x8x4), 76 f32 result, 82-83
// the input type (0 f16, 1 bf16, 2 tf32); 77 .SP (sparse, sm_80+).
void dec_hmma(Instr& ins, const Word& w) {
  ins.op = Op::HMMA;
  ins.mnemonic = "HMMA";
  const unsigned shape = static_cast<unsigned>(w.field(75, 1) | (w.field(78, 1) << 1));
  const unsigned in_type = static_cast<unsigned>(w.field(82, 2));
  const bool sp = w.bit(73);   // .SP: A is 2:4 structured-sparse (sm_80+)
  static const char* const shapes[] = {"1688", "16816", "1684", "(3)"};
  static const char* const sp_shapes[] = {"1688", "16816", "(2)", "16832"};
  if (sp) ins.mods.push_back("SP");
  ins.mods.push_back(sp ? sp_shapes[shape] : shapes[shape]);
  ins.mods.push_back(w.bit(76) ? "F32" : "F16");
  if (in_type == 1) ins.mods.push_back("BF16");
  if (in_type == 2) ins.mods.push_back("TF32");
  ins.f[0] = shape;
  ins.f[1] = w.bit(76);
  ins.f[2] = in_type;
  ins.f[3] = sp;
  mma_regs(ins, w);
  sparse_meta(ins, w, sp);
}

// IMMA: the shape from 75 and 85-86, operand signedness 76/78, 4-bit 83/84,
// .SAT 82.
void dec_imma(Instr& ins, const Word& w) {
  ins.op = Op::IMMA;
  ins.mnemonic = "IMMA";
  const unsigned shape = static_cast<unsigned>(w.field(75, 1) | (w.field(85, 2) << 1));
  const bool sp = w.bit(72);
  static const char* const shapes[] = {"8816", "(1)", "8832", "(3)", "16816", "16832", "16864", "168128"};
  if (sp) ins.mods.push_back("SP");
  ins.mods.push_back(shapes[shape]);
  const bool a4 = w.bit(83), b4 = w.bit(84);
  ins.mods.push_back(std::string(w.bit(76) ? "S" : "U") + (a4 ? "4" : "8"));
  ins.mods.push_back(std::string(w.bit(78) ? "S" : "U") + (b4 ? "4" : "8"));
  if (w.bit(82)) ins.mods.push_back("SAT");
  ins.f[0] = shape;
  ins.f[1] = w.bit(76);
  ins.f[2] = w.bit(78);
  ins.f[3] = sp;
  ins.f[4] = a4;
  ins.f[5] = b4;
  ins.f[6] = w.bit(82);
  mma_regs(ins, w, ".ROW", ".COL");
  sparse_meta(ins, w, sp);
}

// QMMA (sm_89): FP8 A and B. 77 an F32 accumulator (else F16), 78 and 79
// A and B are E5M2 (else E4M3).
void dec_qmma(Instr& ins, const Word& w) {
  ins.op = Op::QMMA;
  ins.mnemonic = "QMMA";
  ins.mods.push_back(w.field(74, 2) == 3 ? "16832" : "(shape " + std::to_string(w.field(74, 2)) + ")");
  ins.mods.push_back(w.bit(77) ? "F32" : "F16");
  ins.mods.push_back(w.bit(78) ? "E5M2" : "E4M3");
  ins.mods.push_back(w.bit(79) ? "E5M2" : "E4M3");
  ins.f[0] = w.bit(77);
  ins.f[1] = w.bit(78);
  ins.f[2] = w.bit(79);
  if (ins.sm >= 120) mma_regs(ins, w);   // sm_120 prints no layouts
  else mma_regs(ins, w, ".ROW", ".COL");
}

// BMMA: 1-bit matrices; 76 .AND (else .XOR), shape from 75 and 85-86.
void dec_bmma(Instr& ins, const Word& w) {
  ins.op = Op::BMMA;
  ins.mnemonic = "BMMA";
  // 75-76 the shape, 78 .AND (else .XOR).
  const unsigned shape = static_cast<unsigned>(w.field(75, 2));
  static const char* const shapes[] = {"88128", "168128", "168256", "(3)"};
  ins.mods.push_back(shapes[shape]);
  ins.mods.push_back(w.bit(78) ? "AND" : "XOR");
  ins.mods.push_back("POPC");
  ins.f[0] = shape;
  ins.f[1] = w.bit(78);
  mma_regs(ins, w, ".ROW", ".COL");
}

// DMMA.884: double precision, 78-79 the rounding.
void dec_dmma(Instr& ins, const Word& w) {
  ins.op = Op::DMMA;
  ins.mnemonic = "DMMA";
  ins.mods.push_back(ins.sm >= 90 ? "8x8x4" : "884");
  const unsigned rnd = static_cast<unsigned>(w.field(78, 2));
  if (rnd) ins.mods.push_back(kRnd[rnd]);
  ins.f[0] = rnd;
  mma_regs(ins, w);
}

void dec_movm(Instr& ins, const Word& w) {
  ins.op = Op::MOVM;
  ins.mnemonic = "MOVM";
  if (w.bit(75) && w.bit(78)) {   // sm_100: 4-bit elements widened to 8
    ins.mods.push_back("U4TO8");
    ins.mods.push_back("M832");
    ins.f[0] = 1;
  } else {
    ins.mods.push_back("16");
    ins.mods.push_back("MT88");
  }
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), ins.f[0] ? 2 : 1));
  ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8))));
}

// ---- warp-wide ------------------------------------------------------------------

void dec_shfl(Instr& ins, const Word& w) {
  ins.op = Op::SHFL;
  ins.mnemonic = "SHFL";
  static const char* const modes[] = {"IDX", "UP", "DOWN", "BFLY"};
  const unsigned mode = static_cast<unsigned>(w.field(58, 2));
  ins.mods.push_back(modes[mode]);
  ins.f[0] = mode;
  // 0x389 lane and clamp registers; 0x589 clamp immediate; 0x989 lane
  // immediate; 0xf89 both immediate.
  const unsigned form = static_cast<unsigned>(w.field(9, 3));
  const bool lane_imm = form == 4 || form == 7, c_imm = form == 2 || form == 7;
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8))));
  ins.src.push_back(lane_imm ? Imm(w.field(53, 5)) : R(static_cast<unsigned>(w.field(32, 8))));
  ins.src.push_back(c_imm ? Imm(w.field(40, 13)) : R(static_cast<unsigned>(w.field(64, 8))));
  mark_reuse(ins, w);
}

void dec_vote(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::VOTEU : Op::VOTE;
  ins.mnemonic = uniform ? "VOTEU" : "VOTE";
  static const char* const modes[] = {"ALL", "ANY", "EQ", "(3)"};
  const unsigned mode = static_cast<unsigned>(w.field(72, 2));
  ins.mods.push_back(modes[mode]);
  ins.f[0] = mode;
  const unsigned d = static_cast<unsigned>(w.field(16, uniform ? ureg_bits(ins.sm) : 8));
  if (uniform ? d != urz(ins.sm) : d != kRZ) ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  const unsigned pd = static_cast<unsigned>(w.field(81, 3));
  ins.dst.push_back(uniform ? UP(pd) : P(pd));
  ins.src.push_back(pred_src(w, 87, 90));
}

void dec_match(Instr& ins, const Word& w) {
  ins.op = Op::MATCH;
  ins.mnemonic = "MATCH";
  ins.mods.push_back(w.bit(79) ? "ANY" : "ALL");
  if (w.bit(73)) ins.mods.push_back("U64");
  ins.f[0] = w.bit(79);
  ins.f[1] = w.bit(73);
  const unsigned pd = static_cast<unsigned>(w.field(81, 3));
  if (!w.bit(79) && pd != kPT) ins.dst.push_back(P(pd));
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8)), w.bit(73) ? 2 : 1));
}

void dec_redux(Instr& ins, const Word& w) {
  ins.op = Op::REDUX;
  ins.mnemonic = "REDUX";
  static const char* const ops[] = {"AND", "OR", "XOR", "SUM", "MIN", "MAX", "(6)", "(7)"};
  const unsigned op = static_cast<unsigned>(w.field(78, 3));
  if (op) ins.mods.push_back(ops[op]);   // nvdisasm leaves AND unnamed
  if (op >= 3 && w.bit(73)) ins.mods.push_back("S32");
  ins.f[0] = op;
  ins.f[1] = w.bit(73);
  ins.dst.push_back(UR(static_cast<unsigned>(w.field(16, ureg_bits(ins.sm))), ins.sm));
  ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8))));
}

// ---- predicates ---------------------------------------------------------------

void dec_plop3(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::UPLOP3 : Op::PLOP3;
  ins.mnemonic = uniform ? "UPLOP3" : "PLOP3";
  ins.mods.push_back("LUT");
  const unsigned p0 = static_cast<unsigned>(w.field(81, 3)), p1 = static_cast<unsigned>(w.field(84, 3));
  ins.dst.push_back(uniform ? UP(p0) : P(p0));
  ins.dst.push_back(uniform ? UP(p1) : P(p1));
  ins.src.push_back(pred_src(w, 87, 90, uniform));
  ins.src.push_back(pred_src(w, 77, 80, uniform));
  // The third may be a uniform predicate in a non-uniform PLOP3 (bit 67).
  ins.src.push_back(pred_src(w, 68, 71, uniform || w.bit(67)));
  const unsigned lut0 = static_cast<unsigned>(w.field(64, 3) | (w.field(72, 5) << 3));
  const unsigned lut1 = static_cast<unsigned>(w.field(16, 8));
  ins.src.push_back(Imm(lut0));
  ins.src.push_back(Imm(lut1));
  ins.f[0] = lut0;
  ins.f[1] = lut1;
}

// PLOP3 on registers' sign bits (sm_90, 0x21f): a, b, c at 24, 32, 64;
// the first LUT half moves to 72-79.
void dec_plop3_sign(Instr& ins, const Word& w) {
  ins.op = Op::PLOP3;
  ins.mnemonic = "PLOP3";
  ins.mods.push_back("LUT");
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
  ins.dst.push_back(P(static_cast<unsigned>(w.field(84, 3))));
  for (unsigned pos : {24u, 32u, 64u}) {
    Operand r = R(static_cast<unsigned>(w.field(pos, 8)));
    r.suffix = ".SIGN";
    r.reuse = reuse(w, static_cast<int>(ins.src.size()));
    ins.src.push_back(r);
  }
  const unsigned lut0 = static_cast<unsigned>(w.field(72, 8)), lut1 = static_cast<unsigned>(w.field(16, 8));
  ins.src.push_back(Imm(lut0));
  ins.src.push_back(Imm(lut1));
  ins.f[0] = lut0;
  ins.f[1] = lut1;
  ins.f[2] = 1;   // sources are sign bits
}

// sm_90's integer ops: VIADD a + b; VIMNMX min/max(a, b); VIADDMNMX
// min/max(a + b, c); VIMNMX3 min/max(a, b, c) (.S16x2 at 73: per halves).
// 72 signed (else .U32), 87-90 the predicate choosing min, 63 b negated.
void dec_viadd(Instr& ins, const Word& w) {
  ins.op = Op::VIADD;
  ins.mnemonic = "VIADD";
  // sm_120: 74-75 the lanes (1 S32, 2 U8x4), 80 saturating (.ISAT).
  const unsigned lanes = static_cast<unsigned>(w.field(74, 2));
  if (lanes == 1) ins.mods.push_back("S32");
  if (lanes == 2) ins.mods.push_back(w.bit(73) ? "S8x4" : "U8x4");
  if (w.bit(80)) ins.mods.push_back("ISAT");
  ins.f[0] = lanes;
  ins.f[1] = w.bit(80);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu2(ins, w, kUnsigned);
  int_neg(ins.src[1], w);
}

void dec_vimnmx(Instr& ins, const Word& w, int nsrc, bool add) {
  ins.op = add ? Op::VIADDMNMX : nsrc == 3 ? Op::VIMNMX3 : Op::VIMNMX;
  ins.mnemonic = add ? "VIADDMNMX" : nsrc == 3 ? "VIMNMX3" : "VIMNMX";
  const bool s16x2 = !add && w.bit(73);   // per 16-bit half
  const bool u8x4 = !add && w.bit(74);    // per byte (sm_120)
  if (s16x2) ins.mods.push_back(w.bit(72) ? "S16x2" : "U16x2");
  else if (u8x4) ins.mods.push_back(w.bit(72) ? "S8x4" : "U8x4");
  else if (!w.bit(72)) ins.mods.push_back("U32");
  else if (ins.sm >= 120 && !add && nsrc == 2 && !u8x4) ins.mods.push_back("S32");   // sm_120 names signed too
  if (w.bit(76)) ins.mods.push_back("RELU");   // a negative result becomes 0
  ins.f[0] = w.bit(72);
  ins.f[1] = s16x2 ? 1 : u8x4 ? 2 : 0;   // lanes: 32-bit, 16x2, 8x4
  ins.f[2] = w.bit(76);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  if (ins.sm >= 100 && ins.sm < 120 && !add && nsrc == 2) {   // sm_100 prints two predicate results (PT so far)
    ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
    ins.dst.push_back(P(static_cast<unsigned>(w.field(84, 3))));
  }
  // VIMNMX prints a signed immediate as signed; the others print the bits.
  const ImmStyle imm = !add && nsrc == 2 ? kSigned : kUnsigned;
  if (nsrc == 3) alu3(ins, w, imm);
  else alu2(ins, w, imm);
  if (add) int_neg(ins.src[1], w);
  ins.src.push_back(pred_src(w, 87, 90));
}

void dec_p2r(Instr& ins, const Word& w) {
  ins.op = Op::P2R;
  ins.mnemonic = "P2R";
  if (w.bit(72) || w.field(73, 2)) {   // .B1-.B3: which byte of the register takes the bits
    ins.mods.push_back("B" + std::to_string(w.field(72, 2)));
  }
  ins.f[0] = static_cast<uint32_t>(w.field(72, 2));
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  ins.src.push_back(Txt("PR"));
  alu2(ins, w, kUnsigned);   // the register to merge into, then the mask
}

void dec_r2p(Instr& ins, const Word& w) {
  ins.op = Op::R2P;
  ins.mnemonic = "R2P";
  // Which byte of the register: 72-73, printed as a modifier; Turing keeps
  // it at 76-77 and prints it on the register.
  const unsigned byte = static_cast<unsigned>(ins.sm < 80 ? w.field(76, 2) : w.field(72, 2));
  if (byte && ins.sm >= 80) ins.mods.push_back("B" + std::to_string(byte));
  ins.f[0] = byte;
  ins.dst.push_back(Txt("PR"));
  alu2(ins, w, kUnsigned);
  if (byte && ins.sm < 80) ins.src[0].suffix = ".B" + std::to_string(byte);
}

// ---- one-source integer ops -----------------------------------------------------

// POPC, FLO, BREV, IABS: the one source sits where a second source would.
void one_src(Instr& ins, const Word& w, ImmStyle imm, bool uniform) {
  const AluForm f = alu_form(static_cast<unsigned>(w.field(9, 3)));
  ins.src.push_back(alu_src(w, f.s1 == Slot::R64 ? f.s2 : f.s1, ins.sm, imm, uniform));
  if (ins.src[0].kind == Kind::Reg) ins.src[0].reuse = reuse(w, 1);
}

void dec_popc(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::UPOPC : Op::POPC;
  ins.mnemonic = uniform ? "UPOPC" : "POPC";
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  one_src(ins, w, kUnsigned, uniform);
  ins.src[0].bnot = w.bit(63);
}

void dec_flo(Instr& ins, const Word& w, bool uniform) {
  ins.op = uniform ? Op::UFLO : Op::FLO;
  ins.mnemonic = uniform ? "UFLO" : "FLO";
  if (!w.bit(73)) ins.mods.push_back("U32");
  if (w.bit(74)) ins.mods.push_back("SH");
  ins.f[0] = w.bit(73);
  ins.f[1] = w.bit(74);
  const unsigned pd = static_cast<unsigned>(w.field(81, 3));
  if (pd != kPT) ins.dst.push_back(uniform ? UP(pd) : P(pd));
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  one_src(ins, w, kUnsigned, uniform);
  ins.src[0].bnot = w.bit(63);
}

void dec_brev(Instr& ins, const Word& w, bool uniform) {
  ins.op = Op::BREV;
  ins.mnemonic = uniform ? "UBREV" : "BREV";
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  one_src(ins, w, kUnsigned, uniform);
}

void dec_iabs(Instr& ins, const Word& w) {
  ins.op = Op::IABS;
  ins.mnemonic = "IABS";
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  one_src(ins, w, kSigned, false);
}

// ---- float and conversions --------------------------------------------------------

void dec_mufu(Instr& ins, const Word& w) {
  ins.op = Op::MUFU;
  ins.mnemonic = "MUFU";
  static const char* const ops[] = {"COS", "SIN", "EX2", "LG2", "RCP", "RSQ", "RCP64H", "RSQ64H",
                                    "SQRT", "TANH", "(10)", "(11)", "(12)", "(13)", "(14)", "(15)"};
  const unsigned op = static_cast<unsigned>(w.field(74, 4));
  ins.mods.push_back(ops[op]);
  if (w.bit(73)) ins.mods.push_back("F16");
  ins.f[0] = op;
  ins.f[1] = w.bit(73);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  one_src(ins, w, kFloat, false);
  src_neg_abs(ins.src[0], w, true);
  if (op == 6 || op == 7) double_imm(ins);   // an immediate is a double's high word
}

void dec_fmnmx(Instr& ins, const Word& w) {
  ins.op = Op::FMNMX;
  ins.mnemonic = "FMNMX";
  if (w.bit(80)) ins.mods.push_back("FTZ");
  if (w.bit(81)) ins.mods.push_back("NAN");   // a NaN operand gives NaN
  ins.f[0] = w.bit(80);
  ins.f[1] = w.bit(81);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu2(ins, w, kFloat);
  float_srcs(ins, w);
  ins.src.push_back(pred_src(w, 87, 90));
}

void dec_i2fp(Instr& ins, const Word& w) {
  // I2FP (sm_80+): a 32-bit integer to a 32-bit float, the ALU's own form.
  ins.op = Op::I2FP;
  ins.mnemonic = "I2FP";
  ins.mods.push_back("F32");
  ins.mods.push_back(w.bit(74) ? "S32" : "U32");
  const unsigned rnd = static_cast<unsigned>(w.field(78, 2));
  if (rnd) ins.mods.push_back(kRnd[rnd]);
  ins.f[0] = w.bit(74);
  ins.f[1] = rnd;
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  one_src(ins, w, kUnsigned, false);
}

// ---- control -------------------------------------------------------------------

void dec_warpsync(Instr& ins, const Word& w) {
  ins.op = Op::WARPSYNC;
  ins.mnemonic = "WARPSYNC";
  const unsigned cond = static_cast<unsigned>(w.field(87, 3));
  if (cond != kPT || w.bit(90)) ins.src.push_back(pred_src(w, 87, 90));
  if (ins.sm >= 90 && w.field(9, 3) == 1) {
    ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8))));   // moved to 24
    if (w.bit(85)) ins.mods.push_back("EXCLUSIVE");
    // .COLLECTIVE (86): with a branch target, for collectives in divergent
    // code (paired with ENDCOLLECTIVE).
    if (w.bit(86)) {
      ins.mods.push_back("COLLECTIVE");
      ins.src.push_back(label(branch_target(w, ins.pc, ins.sm)));
      ins.f[1] = 1;
    }
  } else if (ins.sm >= 90 && w.field(32, 32) == 0) {
    ins.mods.push_back("ALL");   // Hopper spells the full mask so
    ins.f[0] = 1;
  } else {
    one_src(ins, w, kUnsigned, false);
  }
}

void dec_nanosleep(Instr& ins, const Word& w) {
  ins.op = Op::NANOSLEEP;
  ins.mnemonic = "NANOSLEEP";
  if (w.bit(84)) ins.mods.push_back("SYNCS");   // sm_90: wakes on a transaction barrier too
  if (w.bit(85)) ins.mods.push_back("WARP");
  one_src(ins, w, kUnsigned, false);
}

void dec_bar(Instr& ins, const Word& w) {
  ins.op = Op::BAR;
  ins.mnemonic = "BAR";
  // Bits 77-79 the mode (0 .SYNC, 1 .ARV, 2 .RED, 3 .SCAN, 4 .SYNCALL),
  // 74-75 the reduction (0 .POPC, 1 .AND, 2 .OR), 80 .DEFER_BLOCKING.
  static const char* const modes[] = {"SYNC", "ARV", "RED", "SCAN", "SYNCALL", "(5)", "(6)", "(7)"};
  static const char* const reds[] = {"POPC", "AND", "OR", "(3)"};
  const unsigned mode = static_cast<unsigned>(w.field(77, 3));
  const unsigned red = static_cast<unsigned>(w.field(74, 2));
  ins.mods.push_back(modes[mode]);
  if (mode == 2) ins.mods.push_back(reds[red]);
  if (w.bit(80)) ins.mods.push_back("DEFER_BLOCKING");
  ins.f[0] = mode;
  ins.f[1] = red;
  // The barrier: an immediate at 54-57 unless bit 90 says it is a register
  // (for .RED, bit 90 is the predicate's not).
  const bool breg = mode != 2 && w.bit(90);
  if (breg) ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8))));
  else ins.src.push_back(Imm(w.field(54, 4)));
  // A thread count at 42-53 (.ARV always has one).
  const unsigned count = static_cast<unsigned>(w.field(42, 12));
  if (mode == 1 || count) ins.src.push_back(Imm(count));
  ins.f[2] = count;
  if (mode == 2) ins.src.push_back(pred_src(w, 87, 90));
}

// CALL.ABS R: to the address in a register; CALL.REL: to a relative target,
// or (0x344) to the target plus a register, printed "R2 0x3d40" like RET.
void dec_call(Instr& ins, const Word& w, bool abs, bool reg_rel = false) {
  ins.op = Op::CALL;
  ins.mnemonic = "CALL";
  ins.mods.push_back(abs ? "ABS" : "REL");
  ins.mods.push_back("NOINC");
  if (abs) {
    ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8))));
  } else if (reg_rel) {
    const unsigned r = static_cast<unsigned>(w.field(24, 8));
    const int64_t target = static_cast<int64_t>(branch_target(w, ins.pc, ins.sm));
    ins.src.push_back(Txt("R" + std::to_string(r) + " " + signed_hex(target)));
    ins.src.back().reg = r;
    ins.src.back().imm = target;
  } else {
    ins.src.push_back(label(branch_target(w, ins.pc, ins.sm)));
  }
  ins.f[0] = abs;
  ins.f[1] = reg_rel;
}

// RET and BRX print their register and target separated by a space, not a
// comma, so the pair is one operand.
void dec_ret(Instr& ins, const Word& w) {
  ins.op = Op::RET;
  ins.mnemonic = "RET";
  ins.mods.push_back("REL");
  ins.mods.push_back("NODEC");
  const unsigned r = static_cast<unsigned>(w.field(24, 8));
  ins.src.push_back(Txt("R" + std::to_string(r) + " " + signed_hex(static_cast<int64_t>(branch_target(w, ins.pc, ins.sm)))));
  ins.src.back().reg = r;
  ins.src.back().imm = static_cast<int64_t>(branch_target(w, ins.pc, ins.sm));
}

// BRX jumps to the next instruction's address plus the register plus the
// offset; nvdisasm prints the offset itself, not a resolved target.
void dec_brx(Instr& ins, const Word& w) {
  ins.op = Op::BRX;
  ins.mnemonic = "BRX";
  const unsigned r = static_cast<unsigned>(w.field(24, 8));
  const int64_t off = branch_offset(w, ins.sm);
  ins.src.push_back(Txt("R" + std::to_string(r) + " " + signed_hex(off)));
  ins.src.back().reg = r;
  ins.src.back().imm = off;
}

void dec_yield(Instr& ins, const Word&) {
  ins.op = Op::YIELD;
  ins.mnemonic = "YIELD";
}

void dec_membar(Instr& ins, const Word& w) {
  ins.op = Op::MEMBAR;
  ins.mnemonic = "MEMBAR";
  static const char* const scopes[] = {"CTA", "SM", "GPU", "SYS", "(4)", "(5)", "(6)", "(7)"};
  ins.mods.push_back(w.bit(79) ? "ALL" : "SC");
  ins.mods.push_back(scopes[w.field(76, 3)]);
  ins.f[0] = static_cast<uint32_t>(w.field(76, 3));
  ins.f[1] = w.bit(79);
}

void dec_depbar(Instr& ins, const Word& w) {
  ins.op = Op::DEPBAR;
  ins.mnemonic = "DEPBAR";
  // .LE SBn, count: wait until scoreboard n is at most count. 32-37: more
  // scoreboards to wait for entirely, printed as a list.
  if (w.bit(47)) {
    ins.mods.push_back("LE");
    ins.src.push_back(Txt("SB" + std::to_string(w.field(44, 3))));
    ins.src.push_back(Imm(w.field(38, 6)));
  }
  const unsigned list = static_cast<unsigned>(w.field(32, 6));
  if (list) {
    std::string t = "{";
    for (int i = 5; i >= 0; --i)
      if (list >> i & 1) t += (t.size() > 1 ? "," : "") + std::to_string(i);
    ins.src.push_back(Txt(t + "}"));
  }
}

void dec_cs2r(Instr& ins, const Word& w) {
  ins.op = Op::CS2R;
  ins.mnemonic = "CS2R";
  if (!w.bit(80)) ins.mods.push_back("32");
  const unsigned idx = static_cast<unsigned>(w.field(72, 8));
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), w.bit(80) ? 2 : 1));
  Operand sr;
  sr.kind = Kind::SReg;
  sr.reg = idx;
  sr.text = idx == 0xff ? "SRZ" : sreg_name(idx);
  ins.src.push_back(sr);
  ins.f[0] = idx;
}

void dec_r2ur(Instr& ins, const Word& w) {
  ins.op = Op::R2UR;
  ins.mnemonic = "R2UR";
  const unsigned pd = static_cast<unsigned>(w.field(81, 3));
  if (w.bit(84)) ins.mods.push_back("OR");
  if (ins.sm >= 100 && w.bit(86)) ins.mods.push_back("FILL");   // refill a spilled uniform register
  if (ins.sm >= 100 && w.bit(87)) ins.mods.push_back("BROADCAST");
  ins.f[0] = pd != kPT;
  if (pd != kPT) ins.dst.push_back(P(pd));
  ins.dst.push_back(UR(static_cast<unsigned>(w.field(16, ureg_bits(ins.sm))), ins.sm));
  ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8))));
  ins.src.back().reuse = reuse(w, 0);
}

// ---- atomics -------------------------------------------------------------------

const char* const kAtomOp[] = {"ADD", "MIN", "MAX", "INC", "DEC", "AND", "OR", "XOR", "EXCH",
                               "(9)", "(10)", "(11)", "(12)", "(13)", "(14)", "(15)"};
// Before sm_90: 73-75 the operand type. nvdisasm prints no type for .U32.
const char* const kAtomType[] = {"", "S32", "64", "F32.FTZ.RN", "F16x2.RN", "S64", "F64.RN", "(7)"};

void atom_type_mods(Instr& ins, const Word& w) {
  const unsigned t = static_cast<unsigned>(w.field(73, 3));
  if (*kAtomType[t]) ins.mods.push_back(kAtomType[t]);
  ins.f[4] = t;
}

// ATOMG and ATOM (generic): a returned value, global/generic address.
void dec_atom_g(Instr& ins, const Word& w, Op op, const char* name) {
  ins.op = op;
  ins.mnemonic = name;
  if (w.bit(72)) ins.mods.push_back("E");
  const unsigned aop = static_cast<unsigned>(w.field(87, 4));
  ins.mods.push_back(kAtomOp[aop]);
  atom_type_mods(ins, w);
  mem_order_mods(ins, w);
  evict_mods(ins, w);
  ins.f[0] = aop;
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), w.field(73, 3) == 2 || w.field(73, 3) == 5 ? 2 : 1));
  const int ur = w.bit(91) && !w.bit(72) ? static_cast<int>(w.field(64, ureg_bits(ins.sm))) : -1;
  // The 0x3xx forms (bit 11 clear) have no uniform-register field and print
  // a bare register.
  const bool full = w.bit(11);
  ins.src.push_back(ins.sm < 80 ? sm75_gaddr(w, 64, atom_off(w, ins.sm), ins.sm)
                                : mem_addr(static_cast<unsigned>(w.field(24, 8)), full && (w.bit(90) || w.bit(72)), "", ur,
                                           atom_off(w, ins.sm), ins.sm));
  if (full && w.bit(72)) add_desc(ins.src.back(), w, 64, ins.sm);
  ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8))));
}

// sm_90's float atomics, opcodes of their own: ATOM (0x9a2, generic),
// ATOMG (0x9a3), REDG (0x9a6). 73-76 the type (9 F32, 15 F64, 0 F16x2,
// 3 BF16x2), mapped onto the older types' numbering (BF16x2 is 7).
void dec_atom_f(Instr& ins, const Word& w, Op op, const char* name) {
  ins.op = op;
  ins.mnemonic = name;
  const bool red = op == Op::RED;
  if (w.bit(72)) ins.mods.push_back("E");
  const unsigned aop = static_cast<unsigned>(w.field(87, 3));   // 90 is something else from sm_100
  ins.mods.push_back(kAtomOp[aop]);
  unsigned t;
  const char* tname;
  switch (w.field(73, 4)) {
    case 9: t = 3; tname = "F32.FTZ.RN"; break;
    case 15: t = 6; tname = "F64.RN"; break;
    case 0: t = 4; tname = "F16x2.RN"; break;
    case 3: t = 7; tname = "BF16x2.RN"; break;
    default: throw Error(Err::UnsupportedPtx, "SASS: float atomic type " + std::to_string(w.field(73, 4)));
  }
  ins.mods.push_back(tname);
  mem_order_mods(ins, w);
  evict_mods(ins, w);
  ins.f[0] = aop;
  ins.f[4] = t;
  const unsigned width = t == 6 ? 2 : 1;
  if (!red) {
    ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
    ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), width));
  }
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), w.bit(90) || w.bit(72), "", -1, atom_off(w, ins.sm), ins.sm));
  if (w.bit(72)) add_desc(ins.src.back(), w, 64, ins.sm);
  ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8)), width));
}

// RED: an atomic with no result.
void dec_red(Instr& ins, const Word& w) {
  ins.op = Op::RED;
  ins.mnemonic = ins.sm >= 90 ? "REDG" : "RED";
  if (w.bit(72)) ins.mods.push_back("E");
  const unsigned aop = static_cast<unsigned>(w.field(87, 3));
  ins.mods.push_back(kAtomOp[aop]);
  atom_type_mods(ins, w);
  mem_order_mods(ins, w);
  evict_mods(ins, w);
  ins.f[0] = aop;
  const int ur = w.bit(91) && !w.bit(72) ? static_cast<int>(w.field(64, ureg_bits(ins.sm))) : -1;
  ins.src.push_back(ins.sm < 80 ? sm75_gaddr(w, 64, atom_off(w, ins.sm), ins.sm)
                                : mem_addr(static_cast<unsigned>(w.field(24, 8)), w.bit(90), "", ur, atom_off(w, ins.sm), ins.sm));
  if (w.bit(72)) add_desc(ins.src.back(), w, 64, ins.sm);
  ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8))));
}

// ATOMS: shared memory.
void dec_atoms(Instr& ins, const Word& w, bool cas) {
  ins.op = Op::ATOMS;
  ins.mnemonic = "ATOMS";
  if (cas) {
    // 87-88 = 3: CAST.SPIN, a compare-and-store that reports whether it
    // stored (the spin-lock form).
    if (w.field(87, 2) == 3) {
      ins.mods.push_back("CAST");
      ins.mods.push_back("SPIN");
    } else {
      ins.mods.push_back("CAS");
    }
    ins.f[1] = static_cast<uint32_t>(w.field(87, 2));
  } else {
    const unsigned aop = static_cast<unsigned>(w.field(87, 4));
    ins.mods.push_back(kAtomOp[aop]);
    ins.f[0] = aop;
  }
  atom_type_mods(ins, w);
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  static const char* const strides[] = {"", ".X4", ".X8", ".X16"};
  const int ur = w.bit(91) ? static_cast<int>(w.field(64, ureg_bits(ins.sm))) : -1;
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), false, strides[w.field(78, 2)], ur,
                             w.sfield(40, 24), ins.sm, ur >= 0));
  ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8))));
  if (cas) ins.src.push_back(R(static_cast<unsigned>(w.field(64, 8))));
}

// ATOMS.POPC.INC: each active thread's increment of one shared word, summed
// across the warp (a warp-aggregated atomic). It returns the old value.
// 87-90 = 9 is instead ATOMS.ARRIVE.64 Rd, [addr]: sm_80's mbarrier.arrive
// (exec_ops.inc, exec_arrive80), returning the word before.
void dec_atoms_popc(Instr& ins, const Word& w) {
  ins.op = Op::ATOMS;
  ins.mnemonic = "ATOMS";
  if (w.field(87, 4) == 9) {
    ins.mods = {"ARRIVE", "64"};
    ins.f[0] = 9;
    ins.f[2] = 2;   // the arrive form
    const unsigned rd = static_cast<unsigned>(w.field(16, 8));
    ins.dst.push_back(R(rd, rd == kRZ ? 1 : 2));
    const int ur = static_cast<int>(w.field(64, ureg_bits(ins.sm)));
    ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), false, "", ur, w.sfield(40, 24), ins.sm, true));
    return;
  }
  ins.mods.push_back("POPC");
  ins.mods.push_back("INC");
  ins.mods.push_back("32");
  ins.f[0] = 3;
  ins.f[2] = 1;   // popc form
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  static const char* const strides[] = {"", ".X4", ".X8", ".X16"};
  const int ur = static_cast<int>(w.field(64, ureg_bits(ins.sm)));
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), false, strides[w.field(78, 2)], ur,
                             w.sfield(40, 24), ins.sm, true));
}

// ---- textures and surfaces -------------------------------------------------------
//
// TEX/TLD/TLD4 read a texture through its header in a constant bank: 40-53
// the header's offset, 54-58 the bank (printed as two operands, 0x0, 0x58).
// 61-63 the dimensionality, 72-75 the channel mask, 64-71 a second
// destination (RZ when none), 81-83 a fault predicate.

const char* const kTexDim[] = {"1D", "2D", "3D", "CUBE", "ARRAY_1D", "ARRAY_2D", "(6)", "ARRAY_CUBE"};

// The object: a constant-bank word (bank 54-58, word 40-53) in the 0xbXX
// forms; in the 0xfXX forms (sm_90+) a uniform register at 40 holding the
// handle's low word, then an immediate (sm_90) or a second uniform register
// at 48 (sm_100, unprinted when URZ).
void tex_object(Instr& ins, const Word& w) {
  if (!w.bit(10)) {
    ins.src.push_back(Imm(w.field(54, 5)));
    ins.src.push_back(Imm(w.field(40, 14)));   // the header's word index in the bank
    return;
  }
  ins.f[6] = 1;   // handle in a uniform register
  ins.f[7] = static_cast<uint32_t>(w.field(40, ureg_bits(ins.sm)));
  ins.src.push_back(UR(static_cast<unsigned>(w.field(40, ureg_bits(ins.sm))), ins.sm));
  if (ins.sm < 100) {
    ins.src.push_back(Imm(w.field(48, 6)));
  } else if (w.field(48, 8) != urz(ins.sm)) {
    ins.src.push_back(UR(static_cast<unsigned>(w.field(48, 8)), ins.sm));
  }
}

void tex_common(Instr& ins, const Word& w, bool with_mask) {
  ins.dst.push_back(R(static_cast<unsigned>(w.field(64, 8))));
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8))));
  // A second coordinate register of RZ (a 1D lookup) is not printed.
  if (w.field(32, 8) != kRZ) ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8))));
  tex_object(ins, w);
  ins.src.push_back(Txt(kTexDim[w.field(61, 3)]));
  if (with_mask && w.field(72, 4) != 0xf) ins.src.push_back(Imm(w.field(72, 4)));   // all four: the default
  ins.f[0] = static_cast<uint32_t>(w.field(61, 3));
  ins.f[1] = static_cast<uint32_t>(w.field(72, 4));
}

// The LOD mode, 87-89: nothing (auto), .LZ, .LB, .LL, .LC? (checked against
// nvdisasm as they appear).
const char* const kLod[] = {"", "LZ", "LB", "LL", "LBA", "LLA", "(6)", "(7)"};

// The LOD mode: 87-89, or from sm_100 split with its low bit at 59 (Mesa
// NAK's set_tex_lod_mode2).
unsigned tex_lod(const Instr& ins, const Word& w) {
  if (ins.sm >= 100) return static_cast<unsigned>(w.field(59, 1) | (w.field(87, 2) << 1));
  return static_cast<unsigned>(w.field(87, 3));
}

void dec_tex(Instr& ins, const Word& w) {
  ins.op = Op::TEX;
  ins.mnemonic = "TEX";
  if (w.bit(60)) ins.mods.push_back("SCR");
  const unsigned lod = tex_lod(ins, w);
  if (lod) ins.mods.push_back(kLod[lod]);
  ins.f[2] = lod;
  tex_common(ins, w, true);
}

void dec_tld(Instr& ins, const Word& w) {
  ins.op = Op::TLD;
  ins.mnemonic = "TLD";
  if (w.bit(60)) ins.mods.push_back("SCR");
  unsigned lod = tex_lod(ins, w);
  if (ins.sm >= 100 && lod == 0) lod = 1;   // a fetch's LOD is explicit: 0 reads as .LZ
  static const char* const lods[] = {"", "LZ", "(2)", "LL", "(4)", "(5)", "(6)", "(7)"};
  if (lod) ins.mods.push_back(lods[lod]);
  ins.f[2] = lod;
  tex_common(ins, w, true);
}

void dec_tld4(Instr& ins, const Word& w) {
  ins.op = Op::TLD4;
  ins.mnemonic = "TLD4";
  if (w.bit(60)) ins.mods.push_back("SCR");
  static const char* const comps[] = {"R", "G", "B", "A"};
  ins.mods.push_back(comps[w.field(87, 2)]);
  ins.f[2] = static_cast<uint32_t>(w.field(87, 2));
  tex_common(ins, w, false);
}

// SULD/SUST: surfaces by header (40-58 as for textures), 61-63 the
// dimensionality, 73-75 the size (.D binary form); .STRONG.SM etc. as memory,
// 84-86 the out-of-bounds mode? (.IGN, .TRAP).
const char* const kImgDim[] = {"1D", "1D_BUFFER", "1D_ARRAY", "2D", "2D_ARRAY", "3D", "(6)", "(7)"};

void surf_mods(Instr& ins, const Word& w) {
  ins.mods.push_back("D");
  ins.mods.push_back("BA");
  ins.mods.push_back(kImgDim[w.field(61, 3)]);
  const unsigned size = static_cast<unsigned>(w.field(73, 3));
  if (*kMemSize[size]) ins.mods.push_back(kMemSize[size]);
  mem_order_mods(ins, w);
  // 59-60: what an out-of-bounds access does: 1 the default, 0 .IGN
  // (ignored), 2 .TRAP.
  static const char* const oob[] = {"IGN", "", "TRAP", "(3)"};
  const unsigned mode = static_cast<unsigned>(w.field(59, 2));
  if (*oob[mode]) ins.mods.push_back(oob[mode]);
  ins.f[0] = static_cast<uint32_t>(w.field(61, 3));
  ins.f[3] = size;
  ins.f[5] = mode;
}

void dec_suld(Instr& ins, const Word& w) {
  ins.op = Op::SULD;
  ins.mnemonic = "SULD";
  surf_mods(ins, w);
  const unsigned size = static_cast<unsigned>(w.field(73, 3));
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), mem_regs(size)));
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), false, "", -1, 0, ins.sm));
  tex_object(ins, w);
}

void dec_sust(Instr& ins, const Word& w) {
  ins.op = Op::SUST;
  ins.mnemonic = "SUST";
  surf_mods(ins, w);
  const unsigned size = static_cast<unsigned>(w.field(73, 3));
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), false, "", -1, 0, ins.sm));
  ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8)), mem_regs(size)));
  tex_object(ins, w);
}

// ---- the rest ------------------------------------------------------------------

// ATOMG.CAS (0x3a9) and ATOM.CAS (0x38b, generic): compare with the
// register at 32, store the one at 64 (register pairs for .64). The
// predicate (81-83) says the hardware did it: a generic CAS of a shared
// address leaves it false and the code takes a software path.
void dec_atom_cas(Instr& ins, const Word& w, Op op, const char* name) {
  ins.op = op;
  ins.mnemonic = name;
  if (w.bit(72)) ins.mods.push_back("E");
  const bool cast = w.field(87, 2) == 3;   // CAST.SPIN: reports whether it stored
  if (cast) {
    ins.mods.push_back("CAST");
    ins.mods.push_back("SPIN");
  } else {
    ins.mods.push_back("CAS");
  }
  ins.f[6] = cast;
  atom_type_mods(ins, w);
  mem_order_mods(ins, w);
  evict_mods(ins, w);
  ins.f[5] = 1;   // CAS
  const unsigned t = static_cast<unsigned>(w.field(73, 3));
  const unsigned width = t == 2 || t == 5 || t == 6 ? 2 : 1;
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), width));
  const int ur = w.bit(91) && !w.bit(72) ? static_cast<int>(w.field(64, ureg_bits(ins.sm))) : -1;
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), w.bit(90), "", ur, atom_off(w, ins.sm), ins.sm));
  ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8)), width));
  ins.src.push_back(R(static_cast<unsigned>(w.field(64, 8)), width));
}

// BMOV.32 Bn, R: set a convergence barrier's thread mask.
void dec_bmov(Instr& ins, const Word& w) {
  ins.op = Op::BMOV;
  ins.mnemonic = "BMOV";
  ins.mods.push_back("32");
  Operand b;
  b.kind = Kind::Bar;
  b.reg = static_cast<unsigned>(w.field(24, 4));
  ins.dst.push_back(b);
  const AluForm f = alu_form(static_cast<unsigned>(w.field(9, 3)));
  ins.src.push_back(alu_src(w, f.s1 == Slot::R64 ? f.s2 : f.s1, ins.sm, kUnsigned));
}

// BMOV.32(.CLEAR) R, Bn: read a convergence barrier's mask (84: and clear it).
void dec_bmov_r(Instr& ins, const Word& w) {
  ins.op = Op::BMOV;
  ins.mnemonic = "BMOV";
  ins.mods.push_back("32");
  if (w.bit(84)) ins.mods.push_back("CLEAR");
  ins.f[0] = 1;   // to a register
  ins.f[1] = w.bit(84);
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  Operand b;
  b.kind = Kind::Bar;
  b.reg = static_cast<unsigned>(w.field(24, 4));
  ins.src.push_back(b);
}

// BRXU URn offset: BRX with the index in a uniform register.
void dec_brxu(Instr& ins, const Word& w) {
  ins.op = Op::BRX;
  ins.mnemonic = "BRXU";
  const unsigned r = static_cast<unsigned>(w.field(24, ureg_bits(ins.sm)));
  const int64_t off = branch_offset(w, ins.sm);
  ins.src.push_back(Txt("UR" + std::to_string(r) + " " + signed_hex(off)));
  ins.src.back().reg = r;
  ins.src.back().imm = off;
  ins.f[0] = 1;   // uniform index
}

// LDGSTS [shared], [global]: an asynchronous copy of 4, 8 or 16 bytes from
// global to shared memory. 16-23 the shared address and 44-63 its offset,
// 24-31 the global one (64-bit with .E) and 32-43 its offset. 81 clear is
// .BYPASS (not kept in L1), 72 .LTC128B, 82 .ZFILL (the global address's
// low bits count the bytes to zero-fill at the end); 87-90 a predicate,
// false to zero-fill it all.
void dec_ldgsts(Instr& ins, const Word& w) {
  ins.op = Op::LDGSTS;
  ins.mnemonic = "LDGSTS";
  ins.mods.push_back("E");
  if (!w.bit(81)) ins.mods.push_back("BYPASS");
  // sm_100 moves these up a bit: .LTC128B to 73, the size to 74-76.
  if (w.bit(ins.sm >= 100 ? 73 : 72)) ins.mods.push_back("LTC128B");
  const unsigned size = static_cast<unsigned>(w.field(ins.sm >= 100 ? 74 : 73, 3));
  if (*kMemSize[size]) ins.mods.push_back(kMemSize[size]);
  if (w.bit(82)) ins.mods.push_back("ZFILL");
  ins.f[0] = size;
  ins.f[1] = w.bit(82);
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(16, 8)), false, "", -1,
                             static_cast<int64_t>(w.field(44, 20)), ins.sm));
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), true, "", -1, w.sfield(32, 12), ins.sm));
  add_desc(ins.src.back(), w, 64, ins.sm);
  const unsigned p = static_cast<unsigned>(w.field(87, 3));
  if (p != kPT || w.bit(90)) ins.src.push_back(pred_src(w, 87, 90));
}

// LDSM.16.M88(.2/.4) (ldmatrix): 72-73 the matrix count, 78 .MT88
// (transposed); the address as for LDS.
void dec_ldsm(Instr& ins, const Word& w) {
  ins.op = Op::LDSM;
  ins.mnemonic = "LDSM";
  ins.mods.push_back("16");
  ins.mods.push_back(w.bit(78) ? "MT88" : "M88");
  const unsigned n = static_cast<unsigned>(w.field(72, 2));
  if (n == 1) ins.mods.push_back("2");
  if (n == 2) ins.mods.push_back("4");
  ins.f[0] = 1u << n;
  ins.f[1] = w.bit(78);
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), 1u << n));
  const int ur = w.bit(91) ? static_cast<int>(w.field(32, ureg_bits(ins.sm))) : -1;
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), false, "", ur, w.sfield(40, 24), ins.sm));
}

// SGXT(.U32) d, a, n: the low n bits of a, sign- (or zero-) extended.
void dec_sgxt(Instr& ins, const Word& w, bool uniform) {
  ins.op = Op::SGXT;
  ins.mnemonic = uniform ? "USGXT" : "SGXT";
  if (w.bit(75)) ins.mods.push_back("W");
  if (!w.bit(73)) ins.mods.push_back("U32");
  ins.f[0] = w.bit(73);
  ins.f[1] = w.bit(75);
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  alu2(ins, w, kUnsigned, uniform);
}

// I2IP (cvt.pack.sat): b and a saturated to 8 (or 16) bits, packed low to
// high, over c. 76-77 the type as I2I's.
void dec_i2ip(Instr& ins, const Word& w) {
  ins.op = Op::I2IP;
  ins.mnemonic = "I2IP";
  static const char* const types[] = {"U8", "S8", "U16", "S16"};
  const unsigned t = static_cast<unsigned>(w.field(76, 2));
  ins.mods.push_back(types[t]);
  ins.mods.push_back("S32");
  ins.mods.push_back("SAT");
  ins.f[0] = t;
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu3(ins, w, kUnsigned);
}

void dec_ubmsk(Instr& ins, const Word& w, bool uniform) {
  ins.op = Op::BMSK;
  ins.mnemonic = uniform ? "UBMSK" : "BMSK";
  if (w.bit(75)) ins.mods.push_back("W");
  ins.f[0] = w.bit(75);
  ins.dst.push_back(dst_reg(w, uniform, ins.sm));
  alu2(ins, w, kUnsigned, uniform);
}

void dec_ldgdepbar(Instr& ins, const Word&) {
  ins.op = Op::LDGDEPBAR;
  ins.mnemonic = "LDGDEPBAR";
}

// ULDC URd, c[bank][URa + offset]: the constant bank indexed by a uniform
// register.
void dec_uldc_idx(Instr& ins, const Word& w) {
  dec_uldc(ins, w);
  const unsigned ur = static_cast<unsigned>(w.field(24, ureg_bits(ins.sm)));
  const unsigned bank = static_cast<unsigned>(w.field(54, 5)), off = static_cast<unsigned>(w.field(38, 16));
  Operand c = Txt("c[" + hex(bank) + "][UR" + std::to_string(ur) + (off ? "+" + hex(off) : "") + "]");
  c.reg = ur;
  c.imm = off;
  ins.src[0] = c;
  ins.f[1] = 1;   // indexed
  ins.f[2] = bank;
}

// ---- Hopper: transaction barriers, cluster and warpgroup housekeeping ----------
//
// SYNCS works on an mbarrier, a 64-bit word in shared memory whose layout
// ptxas spells out (the constants it emits for mbarrier.init and
// pending_count): the low word holds 0x100000 - count at bits 1-20, the high
// word counts arrivals up from the same value at bits 11-30 and holds the
// phase at bit 31. The address is [Ra + URb + offset]: Ra at 24, URb at 64,
// the offset at 40-63.
Operand syncs_addr(const Word& w, int sm) {
  const unsigned ra = static_cast<unsigned>(w.field(24, 8));
  const int ur = static_cast<int>(w.field(64, ureg_bits(sm)));
  return mem_addr(ra, false, "", ur, w.sfield(40, 24), sm, ra != kRZ);
}

// SYNCS.EXCH.64 URd, [URa + offset], URb: store b, returning the old word
// (mbarrier.init).
void dec_syncs_exch(Instr& ins, const Word& w) {
  ins.op = Op::SYNCS;
  ins.mnemonic = "SYNCS";
  ins.mods.push_back("EXCH");
  if (w.bit(72)) ins.mods.push_back("64");
  ins.f[0] = 0;   // EXCH
  ins.dst.push_back(UR(static_cast<unsigned>(w.field(16, ureg_bits(ins.sm))), ins.sm));
  ins.src.push_back(mem_addr(kRZ, false, "", static_cast<int>(w.field(24, ureg_bits(ins.sm))), w.sfield(40, 24), ins.sm));
  ins.src.push_back(UR(static_cast<unsigned>(w.field(32, ureg_bits(ins.sm))), ins.sm));
}

// SYNCS.ARRIVE.TRANS64[.RED][.OPTOUT][.mode] Rd, [addr], Rb. 84-86 the
// mode: 0 arrive once and expect Rb more transaction bytes, 1 .A1T0 arrive
// once, 3 .A0TR expect Rb bytes, 4 .A0TX complete Rb bytes, 5 .ART0 arrive
// Rb times. 74 .RED returns nothing; 75 .OPTOUT also leaves the barrier
// for later phases (arrive_drop). Rd (a pair) gets the word before.
void dec_syncs_arrive(Instr& ins, const Word& w) {
  ins.op = Op::SYNCS;
  ins.mnemonic = "SYNCS";
  ins.mods.push_back("ARRIVE");
  ins.mods.push_back("TRANS64");
  if (w.bit(74)) ins.mods.push_back("RED");
  if (w.bit(75)) ins.mods.push_back("OPTOUT");
  if (w.bit(73)) ins.mods.push_back("TMASK");   // arrive.noComplete
  static const char* const modes[] = {"", "A1T0", "A0T1", "A0TR", "A0TX", "ART0", "(6)", "(7)"};
  const unsigned mode = static_cast<unsigned>(w.field(84, 3));
  if (mode) ins.mods.push_back(modes[mode]);
  ins.f[0] = 1;   // ARRIVE
  ins.f[1] = mode;
  ins.f[2] = w.bit(75);
  const unsigned rd = static_cast<unsigned>(w.field(16, 8));
  ins.dst.push_back(R(rd, rd == kRZ ? 1 : 2));
  ins.src.push_back(syncs_addr(w, ins.sm));
  ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8))));
}

// SYNCS.PHASECHK.TRANS64[.TRYWAIT] Pd, [addr], Rb: whether the phase whose
// parity is Rb's bit 31 has completed (72: .TRYWAIT, the waiting form).
void dec_syncs_phasechk(Instr& ins, const Word& w) {
  ins.op = Op::SYNCS;
  ins.mnemonic = "SYNCS";
  ins.mods.push_back("PHASECHK");
  ins.mods.push_back("TRANS64");
  if (w.bit(72)) ins.mods.push_back("TRYWAIT");
  ins.f[0] = 2;   // PHASECHK
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
  ins.src.push_back(syncs_addr(w, ins.sm));
  ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8))));
}

// SYNCS.CCTL.IV [addr]: invalidate an mbarrier (.IVALL: all of them).
void dec_syncs_cctl(Instr& ins, const Word& w) {
  ins.op = Op::SYNCS;
  ins.mnemonic = "SYNCS";
  ins.mods.push_back("CCTL");
  ins.f[0] = 3;
  if (!w.bit(91)) {   // no address: all of them
    ins.mods.push_back("IVALL");
    return;
  }
  ins.mods.push_back("IV");
  ins.src.push_back(syncs_addr(w, ins.sm));
}

// ARRIVES.LDGSTSBAR.64[.TRANSCNT/.ARVCNT] [Ra + URb + offset]: cp.async's
// mbarrier arrive, taken when the thread's earlier cp.asyncs land. Plain
// before sm_90 (an arrival, sm_80's layout); from sm_90 .TRANSCNT completes
// the transaction A0T1 declared and .ARVCNT arrives (the .noinc form); at
// 71/70 on sm_90, 77/76 from sm_100.
void dec_arrives(Instr& ins, const Word& w) {
  ins.op = Op::ARRIVES;
  ins.mnemonic = "ARRIVES";
  ins.mods = {"LDGSTSBAR", "64"};
  const bool bw = ins.sm >= 100;
  const bool trans = w.bit(bw ? 77 : 71), arv = w.bit(bw ? 76 : 70);
  if (trans) ins.mods.push_back("TRANSCNT");
  if (arv) ins.mods.push_back("ARVCNT");
  ins.f[0] = trans ? 1 : arv ? 2 : 0;
  const int ur = static_cast<int>(w.field(64, ureg_bits(ins.sm)));
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), false, "", ur, w.sfield(40, 24), ins.sm, true));
}

// STSM.16.M88(.2/.4) (stmatrix): LDSM's layout, stored: 72-73 the count,
// 78 .MT88; the address at 24 (+ offset 40-63), the data at 32.
void dec_stsm(Instr& ins, const Word& w) {
  ins.op = Op::STSM;
  ins.mnemonic = "STSM";
  ins.mods.push_back("16");
  ins.mods.push_back(w.bit(78) ? "MT88" : "M88");
  const unsigned n = static_cast<unsigned>(w.field(72, 2));
  if (n == 1) ins.mods.push_back("2");
  if (n == 2) ins.mods.push_back("4");
  ins.f[0] = 1u << n;
  ins.f[1] = w.bit(78);
  const int ur = w.bit(91) ? static_cast<int>(w.field(64, ureg_bits(ins.sm))) : -1;
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), false, "", ur, w.sfield(40, 24), ins.sm));
  ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8)), 1u << n));
}

// LEPC Rd, target (sm_90's 0x94e): the address of a code location, for a
// return address; the offset keeps the old layout (Rd is at 16).
void dec_lepc_target(Instr& ins, const Word& w) {
  ins.op = Op::MOV;
  ins.mnemonic = "LEPC";
  ins.f[7] = 0x4e;
  ins.f[6] = 1;   // with a target
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), 2));
  // The offset's low byte is at 24-31 here (seen: 0x10, one instruction).
  const int64_t off = static_cast<int64_t>(static_cast<uint64_t>(w.sfield(34, 48)) << 8 | w.field(24, 8));
  ins.src.push_back(label(static_cast<uint64_t>(static_cast<int64_t>(ins.pc) + 16 + off)));
}

// ELECT Pd, URd, Ps: one leader among the threads with Ps; Pd says which,
// URd gets its lane.
void dec_elect(Instr& ins, const Word& w) {
  ins.op = Op::ELECT;
  ins.mnemonic = "ELECT";
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
  ins.dst.push_back(UR(static_cast<unsigned>(w.field(16, ureg_bits(ins.sm))), ins.sm));
  ins.src.push_back(pred_src(w, 87, 90));
}

// Operand-less housekeeping the executor treats as no-ops or waits.
void dec_plain(Instr& ins, Op op, const char* name) {
  ins.op = op;
  ins.mnemonic = name;
}

// FENCE.VIEW.ASYNC.S: order generic-proxy shared memory accesses before
// async-proxy ones (fence.proxy.async).
void dec_fence(Instr& ins, const Word& w) {
  ins.op = Op::MEMBAR;
  ins.mnemonic = "FENCE";
  ins.mods = {"VIEW", "ASYNC", w.bit(73) ? "T" : "S"};   // .T: tcgen05's fences
}

// UTMACCTL.PF/.IV [URa]: prefetch or invalidate a TMA descriptor (82: PF).
void dec_utmacctl(Instr& ins, const Word& w) {
  ins.op = Op::CCTL;
  ins.mnemonic = "UTMACCTL";
  ins.mods.push_back(w.bit(82) ? "PF" : "IV");
  ins.f[0] = 0xff;   // not a data-cache op
  ins.src.push_back(mem_addr(kRZ, false, "", static_cast<int>(w.field(24, ureg_bits(ins.sm))), 0, ins.sm));
}

// USETMAXREG.DEALLOC/.TRY_ALLOC.CTAPOOL [UPd,] n (setmaxnreg): 72-73 the
// direction, 74 .CTAPOOL, the count at 32.
void dec_usetmaxreg(Instr& ins, const Word& w) {
  ins.op = Op::USETMAXREG;
  ins.mnemonic = "USETMAXREG";
  static const char* const dirs[] = {"(0)", "DEALLOC", "TRY_ALLOC", "(3)"};
  ins.mods.push_back(dirs[w.field(72, 2)]);
  if (w.bit(74)) ins.mods.push_back("CTAPOOL");
  const unsigned up = static_cast<unsigned>(w.field(81, 3));
  if (up != kPT) ins.dst.push_back(UP(up));
  ins.src.push_back(Imm(w.field(32, 8)));
}

// WARPGROUP.ARRIVE (wgmma.fence) and WARPGROUP.DEPBAR.LE gsb0, n
// (wgmma.wait_group n; 80 selects it).
void dec_warpgroup(Instr& ins, const Word& w) {
  ins.op = Op::WARPGROUP;
  ins.mnemonic = "WARPGROUP";
  if (!w.bit(80)) {
    ins.mods.push_back("ARRIVE");
    return;
  }
  ins.mods.push_back("DEPBAR");
  if (w.bit(47)) ins.mods.push_back("LE");
  ins.src.push_back(Txt("gsb0"));
  ins.src.push_back(Imm(w.field(72, 6)));
}

// LDGMC (multimem.ld_reduce): a reduction over a multicast object's copies.
// Decoded for the record; the executor does not model multicast memory.
void dec_ldgmc(Instr& ins, const Word& w) {
  ins.op = Op::LDGMC;
  ins.mnemonic = "LDGMC";
  if (w.bit(72)) ins.mods.push_back("E");
  const unsigned t = static_cast<unsigned>(w.field(73, 4));
  ins.mods.push_back(w.field(88, 2) != 3 ? "ADD" : w.bit(81) ? "HPADD" : "F32ADD");
  static const std::map<unsigned, const char*> types = {
      {12, "F32.RN"}, {13, "F32x2.RN"}, {14, "F32x4.RN"}, {3, "BF16x2.RN"}, {4, "BF16x4.RN"}, {5, "BF16x8.RN"}};
  const auto ty = types.find(t);
  ins.mods.push_back(ty != types.end() ? ty->second : "(t" + std::to_string(t) + ")");
  mem_order_mods(ins, w);
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), true, "", static_cast<int>(w.field(64, ureg_bits(ins.sm))),
                             w.sfield(40, 24), ins.sm, true));
}

// ---- Hopper: warpgroup MMA and the tensor memory accelerator ------------------
//
// [HIQB]GMMA (wgmma.mma_async; 0x9f0-0x9f3, 0xdf0 with A in registers):
//   16 D, 24 A's registers or the descriptors' uniform registers (gdesc:
//   A's then B's 64-bit smem descriptors), 32 the descriptors when A is in
//   registers, 64 C (RZ: none), 53-58 N (below), 61/62 A/B transposed,
//   72/63 A/B negated, 73 sparse (metadata register at 40, selector at 48),
//   72-82 the types (below), 84-86 the scoreboard (0 gsb0, 7 none), 87-89
//   the scale-d predicate stored as 7 - UPn, 90 its not.
//
// N: HGMMA and QGMMA take any multiple of 8 and hold N/8 - 1 at 53-57;
// the integer forms take 8, 16, 24 and then multiples of 16, and hold the
// place in that list -- BGMMA at 53-57, IGMMA three times it (plus one
// when sparse) at 53-58.
unsigned gmma_int_n(unsigned idx) { return idx < 4 ? (idx + 1) * 8 : 32 + 16 * (idx - 3); }

enum GmmaKind { kGmmaH, kGmmaI, kGmmaB, kGmmaQ };

void dec_gmma(Instr& ins, const Word& w, GmmaKind kind) {
  static const char* const names[] = {"HGMMA", "IGMMA", "BGMMA", "QGMMA"};
  static const Op ops[] = {Op::HGMMA, Op::IGMMA, Op::BGMMA, Op::QGMMA};
  ins.op = ops[kind];
  ins.mnemonic = names[kind];
  const bool reg_a = w.bit(10), sp = w.bit(73);
  const unsigned n = kind == kGmmaI   ? gmma_int_n(static_cast<unsigned>(w.field(53, 6)) / 3)
                     : kind == kGmmaB ? gmma_int_n(static_cast<unsigned>(w.field(53, 5)))
                                      : (static_cast<unsigned>(w.field(53, 5)) + 1) * 8;
  unsigned k = 0;
  std::vector<std::string> types;
  switch (kind) {
    case kGmmaH:
      k = w.bit(77) ? 8 : 16;
      types.push_back(w.bit(75) ? "F32" : "F16");
      if (w.bit(76)) types.push_back("BF16");
      if (w.bit(77)) types.push_back("TF32");
      break;
    case kGmmaI:
      k = 32;
      types.push_back(w.bit(76) ? "S8" : "U8");   // A signed
      types.push_back(w.bit(82) ? "S8" : "U8");   // B signed
      if (w.bit(75)) types.push_back("SAT");
      break;
    case kGmmaB:
      k = 256;
      types.push_back(w.bit(78) ? "AND" : "XOR");
      types.push_back("POPC");
      break;
    case kGmmaQ:
      k = 32;
      types.push_back(w.bit(75) ? "F32" : "F16");
      types.push_back(w.bit(76) ? "E5M2" : "E4M3");
      types.push_back(w.bit(77) ? "E5M2" : "E4M3");
      break;
  }
  if (sp) {
    ins.mods.push_back("SP");
    k *= 2;
  }
  ins.mods.push_back("64x" + std::to_string(n) + "x" + std::to_string(k));
  for (const std::string& t : types) ins.mods.push_back(t);
  ins.f[0] = n;
  ins.f[1] = k;
  ins.f[2] = static_cast<uint32_t>(w.field(72, 11));   // the type bits, as above
  ins.f[3] = reg_a;
  ins.f[4] = sp;
  ins.f[5] = (w.bit(61) ? 1u : 0u) | (w.bit(62) ? 2u : 0u) | (w.bit(72) && kind == kGmmaH ? 4u : 0u) |
             (w.bit(63) ? 8u : 0u) | (kind == kGmmaI && w.bit(75) ? 16u : 0u);   // tnspA, tnspB, negA, negB, SAT
  ins.f[7] = sp ? static_cast<uint32_t>(w.field(40, 8) | (kind == kGmmaH ? w.field(48, 2) << 8 : 0)) : 0;   // metadata, selector
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  const unsigned desc_at = reg_a ? 32 : 24;
  if (reg_a) ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8))));
  std::string g = "gdesc[UR" + std::to_string(w.field(desc_at, ureg_bits(ins.sm))) + "]";
  if (kind == kGmmaH && w.bit(72)) g += ".negA";
  if (w.bit(63)) g += ".negB";
  if (w.bit(61)) g += ".tnspA";
  if (w.bit(62)) g += ".tnspB";
  Operand d = Txt(g);
  d.reg = static_cast<unsigned>(w.field(desc_at, ureg_bits(ins.sm)));
  ins.src.push_back(d);
  ins.src.push_back(R(static_cast<unsigned>(w.field(64, 8))));
  const unsigned up = 7 - static_cast<unsigned>(w.field(87, 3));
  ins.f[6] = up | (w.bit(90) ? 8u : 0u);
  if (up != kPT || w.bit(90)) ins.src.push_back(UP(up, w.bit(90)));
  if (sp) {   // the metadata, and for 16-bit types the selector (8-bit ones have none)
    ins.src.push_back(R(static_cast<unsigned>(w.field(40, 8))));
    if (kind == kGmmaH) ins.src.push_back(Imm(w.field(48, 2)));
  }
  if (w.field(84, 3) == 0) ins.src.push_back(Txt("gsb0"));
}

// UTMALDG / UTMASTG / UTMAREDG (cp.async.bulk.tensor): 79-81 the
// dimensions less one, 82 .IM2COL (offsets in the uniform register at 64),
// 75 .MULTICAST (the CTA mask there), 76 an L2 policy descriptor at 40.
// [URa] at 32: a load's shared destination, its mbarrier (URa+1) and the
// coordinates (URa+2...); a store's shared source and the coordinates
// (URa+1...). [URb] at 24: the tensor map's address.
void dec_tma(Instr& ins, const Word& w, Op op, const char* name) {
  ins.op = op;
  ins.mnemonic = name;
  const unsigned dims = static_cast<unsigned>(w.field(79, 3)) + 1;
  ins.mods.push_back(std::to_string(dims) + "D");
  if (op == Op::UTMAREDG) ins.mods.push_back(kAtomOp[w.field(87, 3)]);   // 87-89: ADD MIN MAX INC DEC AND OR XOR
  if (w.bit(82)) ins.mods.push_back("IM2COL");
  if (w.bit(75)) ins.mods.push_back("MULTICAST");
  if (ins.sm >= 100 && w.bit(85)) ins.mods.push_back("2CTA");   // a CTA pair's shared memory
  ins.f[0] = dims;
  ins.f[1] = w.bit(82);
  ins.f[2] = w.bit(75);
  ins.f[3] = static_cast<uint32_t>(w.field(87, 3));   // UTMAREDG's op
  const int ub = static_cast<int>(ureg_bits(ins.sm));
  ins.src.push_back(mem_addr(kRZ, false, "", static_cast<int>(w.field(32, ub)), 0, ins.sm));
  ins.src.push_back(mem_addr(kRZ, false, "", static_cast<int>(w.field(24, ub)), 0, ins.sm));
  if (w.bit(82) || w.bit(75)) ins.src.push_back(UR(static_cast<unsigned>(w.field(64, ub)), ins.sm));
  if (w.bit(76)) ins.src.push_back(Txt("desc[UR" + std::to_string(w.field(40, ub)) + "]"));
}

// UBLKCP (cp.async.bulk) and UBLKRED (cp.reduce.async.bulk): 73-74 the
// direction -- 1 .S.G (shared <- global), 2 .G.S, 3 .S.S (to another CTA's
// shared memory) -- and 75 .MULTICAST (the CTA mask in the uniform register
// after the count). [URa] at 32 the destination (a shared one's mbarrier at
// URa+1), [URb] at 24 the source, the byte count in the uniform register at
// 64. UBLKRED: 87-89 the op, 81-84 the type (0 U32, 1 S32, 2 U64, 3 S64,
// 4 F16.RN, 5 F32.RN, 7 F64.RN, 8 BF16.RN).
void dec_ublk(Instr& ins, const Word& w, bool red) {
  ins.op = red ? Op::UBLKRED : Op::UBLKCP;
  ins.mnemonic = red ? "UBLKRED" : "UBLKCP";
  const unsigned dir = static_cast<unsigned>(w.field(73, 2));
  ins.mods.push_back(dir == 1 ? "S" : dir == 2 ? "G" : "S");
  ins.mods.push_back(dir == 1 ? "G" : "S");
  if (w.bit(75)) ins.mods.push_back("MULTICAST");
  ins.f[0] = dir;
  ins.f[1] = w.bit(75);
  if (red) {
    static const char* const types[] = {"",   "S32",  "U64",     "S64",  "F16.RN", "F32.RN", "(6)",  "F64.RN",
                                        "BF16.RN", "(9)", "(10)", "(11)", "(12)", "(13)", "(14)", "(15)"};
    const unsigned t = static_cast<unsigned>(w.field(81, 4));
    ins.mods.push_back(kAtomOp[w.field(87, 3)]);
    if (*types[t]) ins.mods.push_back(types[t]);
    ins.f[2] = static_cast<uint32_t>(w.field(87, 3));
    ins.f[3] = t;
  }
  const int ub = static_cast<int>(ureg_bits(ins.sm));
  ins.src.push_back(mem_addr(kRZ, false, "", static_cast<int>(w.field(32, ub)), 0, ins.sm));
  ins.src.push_back(mem_addr(kRZ, false, "", static_cast<int>(w.field(24, ub)), 0, ins.sm));
  ins.src.push_back(UR(static_cast<unsigned>(w.field(64, ub)), ins.sm));
}

// UBLKPF.L2 [URa], URb (cp.async.bulk.prefetch.L2) and UTMAPF.L2.nD [URa],
// [URb] (its tensor form: the coordinates' registers at 32, the tensor map
// at 24): hints, with nothing to do here.
void dec_ublkpf(Instr& ins, const Word& w) {
  ins.op = Op::NOP;
  ins.mnemonic = "UBLKPF";
  ins.mods.push_back("L2");
  const int ub = static_cast<int>(ureg_bits(ins.sm));
  ins.src.push_back(mem_addr(kRZ, false, "", static_cast<int>(w.field(24, ub)), 0, ins.sm));
  ins.src.push_back(UR(static_cast<unsigned>(w.field(64, ub)), ins.sm));
}

void dec_utmapf(Instr& ins, const Word& w) {
  ins.op = Op::NOP;
  ins.mnemonic = "UTMAPF";
  ins.mods.push_back("L2");
  ins.mods.push_back(std::to_string(w.field(79, 3) + 1) + "D");
  const int ub = static_cast<int>(ureg_bits(ins.sm));
  ins.src.push_back(mem_addr(kRZ, false, "", static_cast<int>(w.field(32, ub)), 0, ins.sm));
  ins.src.push_back(mem_addr(kRZ, false, "", static_cast<int>(w.field(24, ub)), 0, ins.sm));
}

// ---- Blackwell ------------------------------------------------------------------

// sm_120's uniform float and conversion ops share their vector twins'
// encodings (with their own opcodes): decode as the twin, then move every
// register onto the uniform datapath -- the fields are the same width from
// sm_100 (eight bits), and RZ/PT become URZ/UPT.
void uniformize(Instr& ins, Op op, const char* name) {
  const auto conv = [&](Operand& o) {
    if (o.kind == Kind::Reg) {
      o.kind = Kind::UReg;
    } else if (o.kind == Kind::Pred) {
      o.kind = Kind::UPred;
    }
  };
  for (Operand& o : ins.dst) conv(o);
  for (Operand& o : ins.src) conv(o);
  ins.op = op;
  ins.mnemonic = name;
  ins.guard_uniform = true;
}

template <void (*Dec)(Instr&, const Word&)>
void dec_uniform_float(Instr& ins, const Word& w, Op op, const char* name) {
  Dec(ins, w);
  uniformize(ins, op, name);
}

// IADD(.64) d, a, b (sm_120): 73 64-bit, 72 a negated, 63 b negated.
void dec_iadd(Instr& ins, const Word& w) {
  ins.op = Op::IADD;
  ins.mnemonic = "IADD";
  const bool wide = w.bit(73);
  if (wide) ins.mods.push_back("64");
  ins.f[0] = wide;
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), wide ? 2 : 1));
  alu2(ins, w, kSigned);
  ins.src[0].neg = w.bit(72);
  if (wide && ins.src[1].kind == Kind::Imm) ins.src[1].fwidth = 64;
  int_neg(ins.src[1], w);
  if (wide)
    for (Operand& o : ins.src)
      if (o.kind == Kind::Reg) o.width = 2;
}

// FMNMX3 d, a, b, c, p (sm_100): the min (p) or max of three.
void dec_fmnmx3(Instr& ins, const Word& w) {
  ins.op = Op::FMNMX3;
  ins.mnemonic = "FMNMX3";
  if (w.bit(80)) ins.mods.push_back("FTZ");
  if (w.bit(81)) ins.mods.push_back("NAN");
  ins.f[0] = w.bit(80);
  ins.f[1] = w.bit(81);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu3(ins, w, kFloat);
  float_srcs(ins, w);
  ins.src.push_back(pred_src(w, 87, 90));
}

// LDCU(.size) URd, c[bank][URa + offset]: a uniform constant load; the
// offset at 37-53 (in bytes), the bank at 54-58.
void dec_ldcu(Instr& ins, const Word& w) {
  ins.op = Op::ULDC;
  ins.mnemonic = "LDCU";
  static const char* const sizes[] = {"U8", "S8", "U16", "S16", "", "64", "128", "(7)"};
  const unsigned size = static_cast<unsigned>(w.field(73, 3));
  if (size != 4) ins.mods.push_back(sizes[size]);
  ins.f[0] = size;
  ins.dst.push_back(UR(static_cast<unsigned>(w.field(16, 8)), ins.sm));
  const unsigned ur = static_cast<unsigned>(w.field(24, 8));
  const unsigned bank = static_cast<unsigned>(w.field(54, 5)), off = static_cast<unsigned>(w.field(37, 17));
  std::string t = "c[" + hex(bank) + "][";
  if (ur != urz(ins.sm)) t += "UR" + std::to_string(ur) + (off ? "+" + hex(off) : "");
  else t += off ? hex(off) : "URZ";
  Operand c = Txt(t + "]");
  c.reg = ur;
  c.imm = off;
  ins.src.push_back(c);
  ins.f[1] = 2;   // LDCU's own addressing: f[2] bank, the register may be URZ
  ins.f[2] = bank;
}

// CREDUX.MIN/MAX(.S32) URd, Ra (sm_100): a warp reduction to a uniform
// register. 78-80 the op (0 MAX, 2 MIN), 73 signed.
void dec_credux(Instr& ins, const Word& w) {
  ins.op = Op::REDUX;
  ins.mnemonic = "CREDUX";
  const unsigned op = static_cast<unsigned>(w.field(78, 3));
  ins.mods.push_back(op == 2 ? "MIN" : op == 0 ? "MAX" : "(op" + std::to_string(op) + ")");
  if (w.bit(73)) ins.mods.push_back("S32");
  ins.f[0] = op == 2 ? 4 : op == 0 ? 5 : 0xff;   // REDUX's numbering: 4 MIN, 5 MAX
  ins.f[1] = w.bit(73);
  ins.dst.push_back(UR(static_cast<unsigned>(w.field(16, 8)), ins.sm));
  ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8))));
  ins.src.back().reuse = reuse(w, 0);
}

// RPCMOV.32 Rpc.LO, Ra / Ra, Rpc.LO (sm_90+): the return address register,
// a half at a time (31: .HI).
void dec_rpcmov(Instr& ins, const Word& w, bool to_reg) {
  ins.op = Op::RPCMOV;
  ins.mnemonic = "RPCMOV";
  ins.mods.push_back("32");
  ins.f[0] = to_reg;
  ins.f[1] = w.bit(31);   // the high word
  const char* half = w.bit(31) ? "Rpc.HI" : "Rpc.LO";
  if (to_reg) {
    ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
    ins.src.push_back(Txt(half));
  } else {
    ins.dst.push_back(Txt(half));
    ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8))));
  }
}

// UP2UR URd, UPR, URa, mask: the uniform predicates into a register.
void dec_up2ur(Instr& ins, const Word& w) {
  ins.op = Op::UP2UR;
  ins.mnemonic = "UP2UR";
  ins.dst.push_back(UR(static_cast<unsigned>(w.field(16, 8)), ins.sm));
  ins.src.push_back(Txt("UPR"));
  ins.src.push_back(UR(static_cast<unsigned>(w.field(24, 8)), ins.sm));
  ins.src.push_back(Imm(w.field(32, 8)));
}

// CS2UR URd, SR: a special register into a uniform register (pair, 80).
void dec_cs2ur(Instr& ins, const Word& w) {
  ins.op = Op::S2UR;
  ins.mnemonic = "CS2UR";
  if (!w.bit(80)) ins.mods.push_back("32");
  ins.f[3] = w.bit(80);
  ins.dst.push_back(UR(static_cast<unsigned>(w.field(16, 8)), ins.sm));
  Operand sr;
  sr.kind = Kind::SReg;
  sr.reg = static_cast<unsigned>(w.field(72, 8));
  sr.text = sreg_name(sr.reg);
  ins.src.push_back(sr);
}

// LDG/STG.E.ENL2.256: 32 bytes in two register quads (sm_100): the first
// quad at 64, the second at 16 (a load's destinations, a store's data with
// the first at 32); the offset at 40-55.
void dec_gmem256(Instr& ins, const Word& w, bool store) {
  ins.op = store ? Op::STG : Op::LDG;
  ins.mnemonic = store ? "STG" : "LDG";
  ins.mods.push_back("E");
  ins.mods.push_back("ENL2");
  ins.mods.push_back("256");
  mem_order_mods(ins, w);
  evict_mods(ins, w);
  ins.f[0] = 8;   // 32 bytes
  const unsigned q0 = static_cast<unsigned>(w.field(store ? 32 : 64, 8)), q1 = static_cast<unsigned>(w.field(16, 8));
  Operand a = mem_addr(static_cast<unsigned>(w.field(24, 8)), true, "", -1, w.sfield(40, 16) * 32, ins.sm);   // in 32-byte units
  add_desc(a, w, store ? 64 : 32, ins.sm);
  if (store) {
    ins.src.push_back(a);
    ins.src.push_back(R(q0, 4));
    ins.src.push_back(R(q1, 4));
  } else {
    ins.dst.push_back(R(q0, 4));
    ins.dst.push_back(R(q1, 4));
    ins.src.push_back(a);
  }
  // 57-63: a load's hint, printed when not the default 0x7f.
  if (!store && w.field(57, 7) != 0x7f) ins.src.push_back(Imm(w.field(57, 7)));
}

// CCTL.E.C.LDCU.IV.DEEP [URa]: invalidate the uniform constant cache for an
// address (sm_100). Nothing to do with no cache modelled.
void dec_cctl_ldcu(Instr& ins, const Word& w) {
  ins.op = Op::CCTL;
  ins.mnemonic = "CCTL";
  ins.mods = {"E", "C", "LDCU", "IV", "DEEP"};
  ins.f[0] = 0xff;
  ins.src.push_back(mem_addr(kRZ, false, "", static_cast<int>(w.field(24, 8)), 0, ins.sm));
}

// ATOMS.CAST.SPIN Pd, [addr], Rb, Rc (sm_100's 0x58d): the result is the
// predicate at 81 instead of a register.
void dec_atoms_cast_p(Instr& ins, const Word& w) {
  dec_atoms(ins, w, true);
  ins.dst[0] = P(static_cast<unsigned>(w.field(81, 3)));
  ins.f[6] = 1;   // predicate result
}

// BRA with a uniform predicate condition (sm_100's 0x547): UPn at 24-26,
// its not at 27.
void dec_bra_up(Instr& ins, const Word& w) {
  dec_bra(ins, w);
  Operand up = UP(static_cast<unsigned>(w.field(24, 3)), w.bit(27));
  ins.src.insert(ins.src.end() - 1, up);
  ins.f[2] = 1;   // the condition is src[size-2]
}

// UVIMNMX (sm_120): VIMNMX on uniform registers, printing .S32 for signed.
void dec_uvimnmx(Instr& ins, const Word& w) {
  dec_vimnmx(ins, w, 2, false);
  // No predicate results on the uniform form.
  ins.dst.resize(1);
  uniformize(ins, Op::VIMNMX, "UVIMNMX");
}

// ---- Blackwell's tensor cores (tcgen05) ------------------------------------------
//
// Tensor memory is addressed as tmem[URa + offset]. 85 is .2CTA throughout
// (the pair of CTAs sharing an MMA).
std::string tmem_text(unsigned ur, int64_t off, int sm) {
  std::string t = "tmem[" + (ur == urz(sm) ? std::string("URZ") : "UR" + std::to_string(ur));
  if (off) t += "+" + signed_hex(off);
  return t + "]";
}

// LDTM (tcgen05.ld) Rd, tmem[URa + offset] and STTM (tcgen05.st)
// tmem[URa + offset], Rb: 80 .PACK16BIT/.EXPAND16BIT; 81-82 the shape -- 0
// 16x128b, 1 16x256b, 2 32x32b (unnamed), 3 16x64b -- or with 87 one half of
// 16x32bx2 (81: lanes 16-31); 83-85 log2 of the count. The data register is
// at 16 (LDTM) or 32 (STTM), the uniform register at 32 (LDTM) or 64 (STTM),
// the offset (lane << 16 | column) at 40-63.
void dec_tmem_ldst(Instr& ins, const Word& w, bool store) {
  ins.op = store ? Op::STTM : Op::LDTM;
  ins.mnemonic = store ? "STTM" : "LDTM";
  static const char* const shapes[] = {"16dp128bit", "16dp256bit", "", "16dp64bit"};
  const bool half = w.bit(87);
  const unsigned shape = half ? 4 + static_cast<unsigned>(w.bit(81)) : static_cast<unsigned>(w.field(81, 2));
  const unsigned n = static_cast<unsigned>(w.field(83, 3));
  if (half) ins.mods.push_back(w.bit(81) ? "16dp32bit_t16_t31" : "16dp32bit_t0_t15");
  else if (*shapes[shape]) ins.mods.push_back(shapes[shape]);
  if (n) ins.mods.push_back("x" + std::to_string(1u << n));
  if (w.bit(80)) ins.mods.push_back(store ? "EXPAND16BIT" : "PACK16BIT");
  ins.f[0] = shape;   // 0-3 as above, 4/5 16x32bx2's halves
  ins.f[1] = 1u << n;
  ins.f[2] = w.bit(80);
  const unsigned ur = static_cast<unsigned>(w.field(store ? 64 : 32, 8));
  const int64_t off = static_cast<int64_t>(w.field(40, 24));
  Operand t = Txt(tmem_text(ur, off, ins.sm));
  t.reg = ur;
  t.imm = off;
  const unsigned r = static_cast<unsigned>(w.field(store ? 32 : 16, 8));
  if (store) {
    ins.src.push_back(t);
    ins.src.push_back(R(r));
  } else {
    ins.dst.push_back(R(r));
    ins.src.push_back(t);
  }
}

// UTC[HIQO]MMA (tcgen05.mma). 10-11 of the opcode: 1 A in shared memory, 2
// A in Tensor Memory, 3 A in shared memory with UTCQMMA's scale factors. A's
// descriptor (or Tensor Memory address) at 24, B's descriptor at 32, D in
// Tensor Memory at 64; the pair at 40 is the sparsity metadata's Tensor
// Memory address (0 when dense) and the instruction descriptor; at 48 the
// scale factors' Tensor Memory address (block-scaled kinds), or a register
// with .ws's zero-column mask or the disable-output-lane masks. The kind:
// 72-73 = 1 IMMA, 3 QMMA; 62-63 OMMA (63 alone: .4X, sm_103's .BLOCK16);
// else HMMA. 75-78 scale-input-d; 83 .WS; 85 .2CTA; 87-89 the enable-input-D
// predicate, 90 its not. Collectors: A's 84 .A_KEEP and 86 .A_REUSE; .ws
// B's 81 .B_KEEP, 82 .B_REUSE and 79-80 the buffer.
void dec_utcmma(Instr& ins, const Word& w) {
  const unsigned form = static_cast<unsigned>(w.field(10, 2));
  const unsigned kind72 = static_cast<unsigned>(w.field(72, 2));
  const bool o = w.bit(63), q = !o && kind72 == 3, i8 = !o && kind72 == 1;
  ins.op = Op::UTCMMA;
  ins.mnemonic = o ? "UTCOMMA" : q ? "UTCQMMA" : i8 ? "UTCIMMA" : "UTCHMMA";
  const bool two = w.bit(85), ws = w.bit(83);
  if (two) ins.mods.push_back("2CTA");
  if (ws) ins.mods.push_back("WS");
  if (o && !w.bit(62)) ins.mods.push_back(ins.sm == 103 ? "BLOCK16" : "4X");
  const bool a_tmem = form == 2, scaled = o || form == 3;
  ins.f[0] = o ? 3 : q ? 1 : i8 ? 2 : 0;   // HMMA (f16/tf32), QMMA, IMMA, OMMA
  ins.f[1] = two;
  ins.f[2] = a_tmem;
  ins.f[3] = scaled;
  ins.f[4] = ws;
  ins.f[5] = static_cast<uint32_t>(w.field(75, 4));   // scale-input-d
  ins.f[6] = static_cast<uint32_t>(w.field(79, 8));   // collector controls, as above (from bit 79)
  ins.f[7] = o && !w.bit(62);                          // 4X
  const auto u = [&](unsigned pos) { return static_cast<unsigned>(w.field(pos, 8)); };
  const auto tmem = [&](unsigned r) {
    Operand d = Txt(tmem_text(r, 0, ins.sm));
    d.reg = r;
    return d;
  };
  const auto gdesc = [&](unsigned r, std::string suffix) {
    Operand d = Txt("gdesc[UR" + std::to_string(r) + "]" + suffix);
    d.reg = r;
    return d;
  };
  if (a_tmem) {
    Operand a = tmem(u(24));
    if (w.bit(86)) a.text += ".A_REUSE";
    if (w.bit(84)) a.text += ".A_KEEP";
    ins.src.push_back(a);
  } else {
    ins.src.push_back(gdesc(u(24), ""));
  }
  std::string bs;
  if (ws) {
    if (w.bit(82)) bs += ".B_REUSE";
    if (w.bit(81)) bs += ".B_KEEP";
    if (const unsigned buf = static_cast<unsigned>(w.field(79, 2))) bs += ".BUFFER" + std::to_string(buf);
  }
  ins.src.push_back(gdesc(u(32), bs));
  ins.src.push_back(tmem(u(64)));
  const unsigned p = u(40);
  ins.src.push_back(tmem(p));
  Operand id = Txt("idesc[UR" + std::to_string(p + 1) + "]");
  id.reg = p + 1;
  ins.src.push_back(id);
  const unsigned x = u(48);
  if (scaled) ins.src.push_back(tmem(x));
  else if (x != urz(ins.sm)) ins.src.push_back(UR(x, ins.sm));
  ins.src.push_back(pred_src(w, 87, 90, true));
  if (ins.f[5]) ins.src.push_back(Imm(ins.f[5]));
}

// UTCBAR[.2CTA][.MULTICAST] [URa], URb[, URc] (tcgen05.commit): arrive on
// an mbarrier when the MMAs so far complete; 75 multicast (the CTA mask at
// 64).
void dec_utcbar(Instr& ins, const Word& w) {
  ins.op = Op::UTCBAR;
  ins.mnemonic = "UTCBAR";
  if (w.bit(85)) ins.mods.push_back("2CTA");
  if (w.bit(75)) ins.mods.push_back("MULTICAST");
  ins.f[0] = w.bit(75);
  ins.src.push_back(mem_addr(kRZ, false, "", static_cast<int>(w.field(24, 8)), 0, ins.sm));
  ins.src.push_back(UR(static_cast<unsigned>(w.field(32, 8)), ins.sm));
  if (w.bit(75)) ins.src.push_back(UR(static_cast<unsigned>(w.field(64, 8)), ins.sm));
}

// UTCCP.T.S[.2CTA][.shape][.decompress] tmem[URa + offset], gdesc[URb]
// (tcgen05.cp): shared memory into Tensor Memory. The shape from 83, 84 and
// 88: 0 128x256b (unnamed), 2 4x256b, 3 128x128b, 4 64x128b.warpx2::02_13,
// 5 its 01_23, 6 32x128b.warpx4; 80-81 the decompression (1 b4x16_p64, 2
// b6x16_p32 into 8 bits); 85 .2CTA. Tensor Memory at 24 (offset 40-55), the
// descriptor at 32.
void dec_utccp(Instr& ins, const Word& w) {
  ins.op = Op::UTCCP;
  ins.mnemonic = "UTCCP";
  ins.mods = {"T", "S"};
  if (w.bit(85)) ins.mods.push_back("2CTA");
  const unsigned shape = static_cast<unsigned>(w.field(83, 2) | (w.field(88, 1) << 2));
  static const char* const shapes[] = {"", "(1)", "4dp256bit", "128dp128bit", "2x64dp128bit_lw02_lw13",
                                       "2x64dp128bit_lw01_lw23", "4x32dp128bit", "(7)"};
  if (*shapes[shape]) ins.mods.push_back(shapes[shape]);
  const unsigned dec = static_cast<unsigned>(w.field(80, 2));
  if (dec == 1) ins.mods.push_back("U4x16P64");
  if (dec == 2) ins.mods.push_back("U6x16P32");
  ins.f[0] = shape;
  ins.f[1] = w.bit(85);
  ins.f[2] = dec;
  const unsigned ur = static_cast<unsigned>(w.field(24, 8));
  const int64_t off = static_cast<int64_t>(w.field(40, 16));
  ins.src.push_back(Txt(tmem_text(ur, off, ins.sm)));
  ins.src.back().reg = ur;
  ins.src.back().imm = off;
  ins.src.push_back(Txt("gdesc[UR" + std::to_string(w.field(32, 8)) + "]"));
  ins.src.back().reg = static_cast<unsigned>(w.field(32, 8));
}

// UTCSHIFT[.2CTA].DOWN tmem[URa + offset] (tcgen05.shift.down): each row of
// the 32 lanes' region down one lane; the offset at 40-55.
void dec_utcshift(Instr& ins, const Word& w) {
  ins.op = Op::UTCSHIFT;
  ins.mnemonic = "UTCSHIFT";
  if (w.bit(85)) ins.mods.push_back("2CTA");
  ins.mods.push_back(w.bit(80) ? "DOWN" : "(up)");
  ins.f[1] = w.bit(85);
  const unsigned ur = static_cast<unsigned>(w.field(24, 8));
  const int64_t off = static_cast<int64_t>(w.field(40, 16));
  ins.src.push_back(Txt(tmem_text(ur, off, ins.sm)));
  ins.src.back().reg = ur;
  ins.src.back().imm = off;
}

// UTCATOMSWS (tcgen05.alloc's allocator): 0x5e3 .FIND_AND_SET.ALIGN UPd,
// URd, URb (a column allocation); 0x9e3 .AND URd, URb (a release).
void dec_utcatomsws(Instr& ins, const Word& w, bool find) {
  ins.op = Op::UTCATOMSWS;
  ins.mnemonic = "UTCATOMSWS";
  if (w.bit(85)) ins.mods.push_back("2CTA");
  if (find) {
    ins.mods.push_back("FIND_AND_SET");
    ins.mods.push_back("ALIGN");
    ins.dst.push_back(UP(static_cast<unsigned>(w.field(81, 3))));
  } else {
    ins.mods.push_back("AND");
  }
  ins.f[0] = find;
  ins.dst.push_back(UR(static_cast<unsigned>(w.field(16, 8)), ins.sm));
  ins.src.push_back(UR(static_cast<unsigned>(w.field(32, 8)), ins.sm));
}

// STAS [Ra.64], Rb (st.async: a store to another CTA's shared memory that
// completes on its mbarrier).
void dec_stas(Instr& ins, const Word& w) {
  ins.op = Op::STAS;
  ins.mnemonic = "STAS";
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), true, "", -1, w.sfield(40, 24), ins.sm));
  ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8))));
}

// UVIRTCOUNT.DEALLOC.SMPOOL n and UGETNEXTWORKID.BROADCAST [URa], [URa+1]
// (cluster launch control's try_cancel).
void dec_uvirtcount(Instr& ins, const Word& w) {
  ins.op = Op::NOP;
  ins.mnemonic = "UVIRTCOUNT";
  ins.mods = {"DEALLOC", "SMPOOL"};
  ins.src.push_back(Imm(w.field(32, 8)));
}

void dec_ugetnextworkid(Instr& ins, const Word& w) {
  ins.op = Op::UGETNEXTWORKID;
  ins.mnemonic = "UGETNEXTWORKID";
  if (w.bit(72)) ins.mods.push_back("BROADCAST");
  const int ur = static_cast<int>(w.field(24, 8));
  ins.src.push_back(mem_addr(kRZ, false, "", ur, 0, ins.sm));
  ins.src.push_back(mem_addr(kRZ, false, "", ur + 1, 0, ins.sm));
}

void dec_vabsdiff4(Instr& ins, const Word& w) {
  ins.op = Op::VABSDIFF4;
  ins.mnemonic = "VABSDIFF4";
  if (!w.bit(73)) ins.mods.push_back("U8");
  if (w.bit(75)) ins.mods.push_back("ACC");
  ins.f[0] = w.bit(75);   // ACC: the sum of the four differences plus c
  ins.f[1] = w.bit(73);   // signed bytes
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu3(ins, w, kUnsigned);
}

// IDP.4A: four byte products; IDP.2A: two halfword-by-byte products, from
// b's low (LO) or high (HI) two bytes. Bits 73/74: a/b signed.
void dec_idp(Instr& ins, const Word& w) {
  ins.op = Op::IDP;
  ins.mnemonic = "IDP";
  const bool two = w.bit(76), sa = w.bit(73), sb = w.bit(74);
  ins.mods.push_back(two ? "2A" : "4A");
  if (two) ins.mods.push_back(w.bit(77) ? "HI" : "LO");
  ins.mods.push_back(two ? (sa ? "S16" : "U16") : (sa ? "S8" : "U8"));
  ins.mods.push_back(sb ? "S8" : "U8");
  ins.f[0] = two;
  ins.f[1] = sa;
  ins.f[2] = sb;
  ins.f[3] = w.bit(77);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu3(ins, w, kUnsigned);
}

void dec_bmsk(Instr& ins, const Word& w) {
  ins.op = Op::BMSK;
  ins.mnemonic = "BMSK";
  if (w.bit(75)) ins.mods.push_back("W");
  ins.f[0] = w.bit(75);
  ins.dst.push_back(dst_reg(w, false, ins.sm));
  alu2(ins, w, kUnsigned);
}

void dec_lepc(Instr& ins, const Word& w) {
  ins.op = Op::MOV;   // the program counter into a register pair
  ins.mnemonic = "LEPC";
  ins.f[7] = 0x4e;
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), 2));
}

void dec_cctl(Instr& ins, const Word& w) {
  ins.op = Op::CCTL;
  ins.mnemonic = "CCTL";
  if (w.bit(72)) ins.mods.push_back("E");
  static const char* const ops[] = {"PF1", "PF2", "WB", "IV", "IVALL", "RS", "IVALLP", "WBALL"};
  ins.mods.push_back(ops[w.field(87, 3)]);
  ins.f[0] = static_cast<uint32_t>(w.field(87, 3));
  if (w.field(87, 3) != 4 && w.field(87, 3) != 6 && w.field(87, 3) != 7)
    ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), w.bit(90), "", -1, w.sfield(40, 24), ins.sm));
}

void dec_b2r(Instr& ins, const Word& w) {
  ins.op = Op::BAR;   // reads a barrier's result: executed by the BAR rule
  ins.mnemonic = "B2R";
  ins.mods.push_back("RESULT");
  ins.f[7] = 0x1c;
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
}

void dec_qspc(Instr& ins, const Word& w) {
  ins.op = Op::CCTL;
  ins.mnemonic = "QSPC";
  ins.f[7] = 0xaa;
  if (w.bit(72)) ins.mods.push_back("E");
  static const char* const spaces[] = {"G", "L", "S", "D"};   // D: a cluster's shared memory (sm_90)
  ins.mods.push_back(spaces[w.field(73, 2)]);
  ins.f[0] = static_cast<uint32_t>(w.field(73, 2));
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  const int ur = w.bit(11) ? static_cast<int>(w.field(32, ureg_bits(ins.sm))) : -1;   // the 0x9aa form
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), false, "", ur, w.sfield(40, 24), ins.sm));
}

void dec_errbar(Instr& ins, const Word&) {
  ins.op = Op::ERRBAR;
  ins.mnemonic = "ERRBAR";
}

void dec_break(Instr& ins, const Word& w) {
  ins.op = Op::BREAK;
  ins.mnemonic = "BREAK";
  if (ins.sm >= 100 && w.bit(72)) ins.mods.push_back("RELIABLE");
  const unsigned cond = static_cast<unsigned>(w.field(87, 3));
  if (cond != kPT || w.bit(90)) ins.src.push_back(pred_src(w, 87, 90));
  Operand b;
  b.kind = Kind::Bar;
  b.reg = static_cast<unsigned>(w.field(16, 4));
  ins.src.push_back(b);
}

void dec_bpt(Instr& ins, const Word& w) {
  ins.op = Op::BPT;
  ins.mnemonic = "BPT";
  static const char* const ops[] = {"(0)", "DRAIN", "CAL", "TRAP", "PAUSE", "INT", "(6)", "(7)"};
  ins.mods.push_back(ops[w.field(84, 3)]);
  ins.src.push_back(Imm(w.field(34, 20)));
  ins.f[0] = static_cast<uint32_t>(w.field(84, 3));
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
      {0x25, dec_imad},  {0x26, dec_imad},  {0x27, dec_imad},  {0x1a, dec_sgxt},  {0x1b, dec_ubmsk},
  };
  return t;
}

const std::unordered_map<unsigned, Dec>& float_table() {
  static const std::unordered_map<unsigned, Dec> t = {
      {0x20, dec_fmul},
      {0x21, dec_fadd},
      {0x23, dec_ffma},
      {0x09, dec_fmnmx},
      {0x13, dec_iabs},
      {0x45, dec_i2fp},
      {0x108, dec_mufu},
      {0x109, [](Instr& i, const Word& w) { dec_popc(i, w, false); }},
      {0x0bf, [](Instr& i, const Word& w) { dec_popc(i, w, true); }},
      {0x100, [](Instr& i, const Word& w) { dec_flo(i, w, false); }},
      {0x0bd, [](Instr& i, const Word& w) { dec_flo(i, w, true); }},
      {0x101, [](Instr& i, const Word& w) { dec_brev(i, w, false); }},
      {0x30, dec_hadd2},
      {0x31, dec_hfma2},
      {0x35, [](Instr& i, const Word& w) {   // HFMA2 on the tensor pipe; from sm_100 IADD
         if (i.sm >= 100) {
           dec_iadd(i, w);
           return;
         }
         dec_hfma2(i, w);
         i.mods.insert(i.mods.begin(), "MMA");
       }},
      {0x32, dec_hmul2},
      {0x33, [](Instr& i, const Word& w) { dec_hset2(i, w, false); }},
      {0x34, [](Instr& i, const Word& w) { dec_hset2(i, w, true); }},
      {0x28, [](Instr& i, const Word& w) { dec_dop(i, w, Op::DMUL, "DMUL", 2); }},
      {0x29, [](Instr& i, const Word& w) { dec_dop(i, w, Op::DADD, "DADD", 2); }},
      {0x2b, [](Instr& i, const Word& w) { dec_dop(i, w, Op::DFMA, "DFMA", 3); }},
      {0x2a, dec_dsetp},
      {0x0b, dec_fsetp},
      {0x0a, dec_fset},
      {0x08, dec_fsel},
      {0x107, dec_frnd}, {0x113, dec_frnd},
      {0x102, dec_fchk},
      {0x106, dec_i2f}, {0x112, dec_i2f},
      {0x105, dec_f2i}, {0x111, dec_f2i},
      {0x104, dec_f2f}, {0x110, dec_f2f},
      {0x38, dec_i2i},
      {0x3e, dec_f2fp},
      {0x15, dec_vabsdiff4},
      {0x26, dec_idp},
      {0x1b, dec_bmsk},
      {0x39, dec_i2ip},
      // sm_120's uniform float path (their vector twins' encodings).
      {0x55, [](Instr& i, const Word& w) { dec_uniform_float<dec_ffma>(i, w, Op::FFMA, "UFFMA"); }},
      {0x54, [](Instr& i, const Word& w) {
         dec_uniform_float<dec_fadd>(i, w, Op::FADD, "UFADD");
         if (w.field(9, 3) == 1) {   // UFADD's register b is in the third slot
           Operand b = UR(static_cast<unsigned>(w.field(64, 8)), i.sm);
           b.slot = static_cast<uint8_t>(Slot::R64);
           src_neg_abs(b, w, true);
           b.reuse = i.src[1].reuse;
           i.src[1] = b;
         }
       }},
      {0x56, [](Instr& i, const Word& w) { dec_uniform_float<dec_fmul>(i, w, Op::FMUL, "UFMUL"); }},
      {0x53, [](Instr& i, const Word& w) { dec_uniform_float<dec_fsetp>(i, w, Op::FSETP, "UFSETP"); }},
      {0x51, [](Instr& i, const Word& w) { dec_uniform_float<dec_fsel>(i, w, Op::FSEL, "UFSEL"); }},
      {0x5a, [](Instr& i, const Word& w) { dec_uniform_float<dec_i2f>(i, w, Op::I2F, "UI2F"); }},
      {0x5b, [](Instr& i, const Word& w) { dec_uniform_float<dec_f2f>(i, w, Op::F2F, "UF2F"); }},
      {0x5c, [](Instr& i, const Word& w) { dec_uniform_float<dec_f2i>(i, w, Op::F2I, "UF2I"); }},
      {0x5d, [](Instr& i, const Word& w) { dec_uniform_float<dec_frnd>(i, w, Op::FRND, "UFRND"); }},
      {0x5e, [](Instr& i, const Word& w) { dec_uniform_float<dec_i2fp>(i, w, Op::I2FP, "UI2FP"); }},
      {0x4a, dec_uvimnmx},
      {0x76, dec_fmnmx3},
      {0x36, dec_viadd},
      {0x48, [](Instr& i, const Word& w) { dec_vimnmx(i, w, 2, false); }},
      {0x46, [](Instr& i, const Word& w) { dec_vimnmx(i, w, 3, true); }},
      {0x0f, [](Instr& i, const Word& w) { dec_vimnmx(i, w, 3, false); }},
      {0x0be, [](Instr& i, const Word& w) { dec_brev(i, w, true); }},
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
      {0x389, dec_shfl}, {0x589, dec_shfl}, {0x989, dec_shfl}, {0xf89, dec_shfl},
      {0x806, [](Instr& i, const Word& w) { dec_vote(i, w, false); }},
      {0x886, [](Instr& i, const Word& w) { dec_vote(i, w, true); }},
      {0x3a1, dec_match}, {0x3c4, dec_redux},
      {0x81c, [](Instr& i, const Word& w) { dec_plop3(i, w, false); }},
      {0x89c, [](Instr& i, const Word& w) { dec_plop3(i, w, true); }},
      {0x803, dec_p2r}, {0x804, dec_r2p},
      {0x948, dec_warpsync}, {0x95d, dec_nanosleep}, {0xb1d, dec_bar},
      {0x343, [](Instr& i, const Word& w) { dec_call(i, w, true); }},
      {0x944, [](Instr& i, const Word& w) { dec_call(i, w, false); }},
      {0x344, [](Instr& i, const Word& w) { dec_call(i, w, false, true); }},
      {0x348, dec_warpsync},
      {0x950, dec_ret}, {0x949, dec_brx}, {0x946, dec_yield}, {0x992, dec_membar}, {0x91a, dec_depbar},
      {0x805, dec_cs2r}, {0x3c2, dec_r2ur},
      {0x9a8, [](Instr& i, const Word& w) { dec_atom_g(i, w, Op::ATOMG, "ATOMG"); }},
      {0x3a8, [](Instr& i, const Word& w) { dec_atom_g(i, w, Op::ATOMG, "ATOMG"); }},
      {0x98a, [](Instr& i, const Word& w) { dec_atom_g(i, w, Op::ATOM, "ATOM"); }},
      {0x38a, [](Instr& i, const Word& w) { dec_atom_g(i, w, Op::ATOM, "ATOM"); }},
      {0x98e, dec_red}, {0x38e, dec_red},
      {0x38c, [](Instr& i, const Word& w) { dec_atoms(i, w, false); }},
      {0x98c, [](Instr& i, const Word& w) { dec_atoms(i, w, false); }},
      {0x38d, [](Instr& i, const Word& w) { dec_atoms(i, w, true); }},
      {0xf8c, dec_atoms_popc},
      {0xb60, dec_tex}, {0xb66, dec_tld}, {0xb63, dec_tld4}, {0xb99, dec_suld}, {0xb9d, dec_sust},
      {0xf60, dec_tex}, {0xf66, dec_tld}, {0xf63, dec_tld4}, {0xf99, dec_suld}, {0xf9d, dec_sust},
      {0x34e, dec_lepc}, {0x98f, dec_cctl}, {0x31c, dec_b2r}, {0x3aa, dec_qspc}, {0x9aa, dec_qspc}, {0x9ab, dec_errbar},
      {0x942, dec_break}, {0x95c, dec_bpt},
      {0x3a9, [](Instr& i, const Word& w) { dec_atom_cas(i, w, Op::ATOMG, "ATOMG"); }},
      {0x9a9, [](Instr& i, const Word& w) { dec_atom_cas(i, w, Op::ATOMG, "ATOMG"); }},
      {0x38b, [](Instr& i, const Word& w) { dec_atom_cas(i, w, Op::ATOM, "ATOM"); }},
      {0x98b, [](Instr& i, const Word& w) { dec_atom_cas(i, w, Op::ATOM, "ATOM"); }},
      {0x9a2, [](Instr& i, const Word& w) { dec_atom_f(i, w, Op::ATOM, "ATOM"); }},
      {0x9a3, [](Instr& i, const Word& w) { dec_atom_f(i, w, Op::ATOMG, "ATOMG"); }},
      {0x9a6, [](Instr& i, const Word& w) { dec_atom_f(i, w, Op::RED, "REDG"); }},
      {0x21f, dec_plop3_sign}, {0x2ca, dec_r2ur},
      {0x5b2, dec_syncs_exch}, {0x9a7, dec_syncs_arrive}, {0x5a7, dec_syncs_phasechk}, {0x9b1, dec_syncs_cctl}, {0x9b0, dec_arrives},
      {0x844, dec_stsm}, {0x94e, dec_lepc_target}, {0x82f, dec_elect}, {0x3c6, dec_fence}, {0x9b9, dec_utmacctl},
      {0x9c8, dec_usetmaxreg}, {0x9c5, dec_warpgroup}, {0x9a5, dec_ldgmc},
      {0x5ab, [](Instr& i, const Word&) { dec_plain(i, Op::ERRBAR, "CGAERRBAR"); }},
      {0x91b, [](Instr& i, const Word&) { dec_plain(i, Op::ENDCOLLECTIVE, "ENDCOLLECTIVE"); }},
      {0x82e, [](Instr& i, const Word&) { dec_plain(i, Op::NOP, "ACQBULK"); }},
      {0x82d, [](Instr& i, const Word&) { dec_plain(i, Op::NOP, "PREEXIT"); }},
      {0x9b7, [](Instr& i, const Word&) { dec_plain(i, Op::NOP, "UTMACMDFLUSH"); }},
      {0x9f0, [](Instr& i, const Word& w) { dec_gmma(i, w, kGmmaH); }},
      {0xdf0, [](Instr& i, const Word& w) { dec_gmma(i, w, kGmmaH); }},
      {0x9f1, [](Instr& i, const Word& w) { dec_gmma(i, w, kGmmaI); }},
      {0xdf1, [](Instr& i, const Word& w) { dec_gmma(i, w, kGmmaI); }},
      {0x9f2, [](Instr& i, const Word& w) { dec_gmma(i, w, kGmmaB); }},
      {0x9f3, [](Instr& i, const Word& w) { dec_gmma(i, w, kGmmaQ); }},
      {0xdf3, [](Instr& i, const Word& w) { dec_gmma(i, w, kGmmaQ); }},
      {0x5b4, [](Instr& i, const Word& w) { dec_tma(i, w, Op::UTMALDG, "UTMALDG"); }},
      {0x3b4, [](Instr& i, const Word& w) { dec_tma(i, w, Op::UTMALDG, "UTMALDG"); }},
      {0x3b5, [](Instr& i, const Word& w) { dec_tma(i, w, Op::UTMASTG, "UTMASTG"); }},
      {0x3b6, [](Instr& i, const Word& w) { dec_tma(i, w, Op::UTMAREDG, "UTMAREDG"); }},
      {0x3ba, [](Instr& i, const Word& w) { dec_ublk(i, w, false); }},
      {0x3bb, [](Instr& i, const Word& w) { dec_ublk(i, w, true); }},
      {0x3bc, dec_ublkpf}, {0x5b8, dec_utmapf},
      {0x9c7, [](Instr& i, const Word&) { dec_plain(i, Op::UCGABAR, "UCGABAR_ARV"); }},   // barrier.cluster.arrive
      {0xdc7, [](Instr& i, const Word&) {                                                // barrier.cluster.wait
         dec_plain(i, Op::UCGABAR, "UCGABAR_WAIT");
         i.f[0] = 1;
       }},
      {0x7ac, dec_ldcu}, {0x2cc, dec_credux}, {0x883, dec_up2ur}, {0x8cb, dec_cs2ur},
      {0x352, [](Instr& i, const Word& w) { dec_rpcmov(i, w, false); }},
      {0x353, [](Instr& i, const Word& w) { dec_rpcmov(i, w, true); }},
      {0x97e, [](Instr& i, const Word& w) { dec_gmem256(i, w, false); }},
      {0x97f, [](Instr& i, const Word& w) { dec_gmem256(i, w, true); }},
      {0x540, dec_cctl_ldcu}, {0x35d, dec_nanosleep}, {0x58d, dec_atoms_cast_p}, {0x547, dec_bra_up},
      {0x9ee, [](Instr& i, const Word& w) { dec_tmem_ldst(i, w, false); }},
      {0x9ed, [](Instr& i, const Word& w) { dec_tmem_ldst(i, w, true); }},
      {0x5ea, dec_utcmma}, {0x9ea, dec_utcmma}, {0xdea, dec_utcmma}, {0x9e6, dec_utcshift}, {0x3e9, dec_utcbar}, {0x9e7, dec_utccp},
      {0x5e3, [](Instr& i, const Word& w) { dec_utcatomsws(i, w, true); }},
      {0x9e3, [](Instr& i, const Word& w) { dec_utcatomsws(i, w, false); }},
      {0xdbd, dec_stas}, {0x84c, dec_uvirtcount}, {0x3ca, dec_ugetnextworkid},
      {0x356, dec_bmov}, {0x355, dec_bmov_r}, {0x958, dec_brxu}, {0xfae, dec_ldgsts}, {0x83b, dec_ldsm}, {0x9af, dec_ldgdepbar},
      {0xabb, dec_uldc_idx},
      {0x23c, dec_hmma}, {0x27a, dec_qmma}, {0x237, dec_imma}, {0x23d, dec_bmma}, {0x23f, dec_dmma}, {0x23a, dec_movm},
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
    // The uniform datapath's own ops take a UP guard.
    if (opc == 0x9c3 || opc == 0xab9 || opc == 0xabb || opc == 0x89c || opc == 0x5b2 || opc == 0x5b4 || opc == 0x3b4 ||
        opc == 0x3b5 || opc == 0x3b6 || opc == 0x3ba || opc == 0x3bb || opc == 0x3bc || opc == 0x5b8 ||
        opc == 0x5ea || opc == 0x9ea || opc == 0xdea || opc == 0x9e7 || opc == 0x9e6 || opc == 0x3e9 ||
        opc == 0x5e3 || opc == 0x9e3 || opc == 0x7ac || opc == 0x883 || opc == 0x8cb ||
        opc == 0x9b9)
      ins.guard_uniform = true;
    return ins;
  }
  const unsigned low9 = opc & 0x1ff;
  // Ops with their own decoders that take the ALU operand forms (bits 9-11),
  // whatever bit 8 holds.
  if (w.field(9, 3) != 0) {
    if (const auto it = float_table().find(low9); it != float_table().end()) {
      it->second(ins, w);
      if ((low9 & 0x180) == 0x80) ins.guard_uniform = true;
      return ins;
    }
  }
  // ALU ops keep bit 8 clear; the form in bits 9-11 is never zero for them.
  if (w.field(9, 3) != 0 && (low9 & 0x100) == 0) {
    const bool uniform = (low9 & 0x80) != 0;
    // sm_120 gives the uniform datapath 64-bit ops of its own: UIADD3.64
    // (0x97, the vector IMNMX's number) and UIMNMX.S64/U64 (0x85).
    if (sm >= 120 && uniform && ((low9 & 0x7f) == 0x17 || (low9 & 0x7f) == 0x05)) {
      if ((low9 & 0x7f) == 0x17) {
        dec_iadd3(ins, w, true);
        ins.mods.insert(ins.mods.begin(), "64");
        ins.f[1] = 1;   // 64-bit
        for (Operand& o : ins.src)
          if (o.kind == Kind::UReg) o.width = 2;
        ins.dst[0].width = 2;
      } else {
        dec_imnmx64(ins, w, true);
      }
      ins.guard_uniform = true;
      return ins;
    }
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
      break;
    case Kind::Pred:
    case Kind::UPred:
      return (o.neg ? "!" : "") + reg_text(o, sm) + o.suffix;
    case Kind::Imm:
      if (o.width == 2) {   // a 64-bit immediate prints as its bits
        char b[24];
        std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(o.imm));
        s = b;
      } else {
        s = imm_text(o.imm);
      }
      break;
    case Kind::FImm: s = o.text; break;
    case Kind::CBank:
      // With a swizzle after it nvdisasm puts a space inside: c[0x0] [0x390].H0_H0
      s = "c[" + imm_text(o.reg) + "]" + (o.suffix.empty() ? "" : " ") + "[" + imm_text(o.imm) + "]";
      break;
    case Kind::UCBank: s = "cx[UR" + std::to_string(o.reg) + "][" + imm_text(o.imm) + "]"; break;
    case Kind::Label: s = imm_text(o.imm); break;
    case Kind::Bar: s = "B" + std::to_string(o.reg); break;
    case Kind::Mem:
    case Kind::SReg:
    case Kind::Text: s = o.text; break;
  }
  if (o.abs) s = "|" + s + "|";
  if (o.reuse) s += ".reuse";   // after the bars: |R4|.reuse
  s += o.suffix;                // then the swizzle: R4.reuse.H0_H0
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
      std::string t = operand_text(o, ins.sm);
      // nvdisasm writes an infinity with a space after it ("+INF , PT").
      if (o.kind == Kind::FImm && t.size() >= 3 &&
          (t.compare(t.size() - 3, 3, "INF") == 0 || t.compare(t.size() - 3, 3, "NAN") == 0 || t == "-0.0"))
        t += ' ';
      s += t;
    }
  while (!s.empty() && s.back() == ' ') s.pop_back();
  return s;
}

}  // namespace vgpu::sass
