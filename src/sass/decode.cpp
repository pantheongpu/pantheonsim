// SASS decoder for the sm_70-and-later 128-bit format. Field positions come
// from Mesa NAK's encoder (src/nouveau/compiler/nak/sm70_encode.rs, MIT) and,
// where NAK is silent, from NVIDIA's own disassembler: every decoder here is
// checked against nvdisasm's text for a corpus of real instructions
// (nvidia/tests/data/sass/, test_sass_decode).
#include "vgpu/sass/sass.hpp"

#include <cmath>
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
// A float or double as nvdisasm prints it: twenty significant digits, and
// from 2^24 up in exponent form with twenty after the point.
std::string float_text(double d) {
  char b[48];
  if (d == __builtin_inf() || d == -__builtin_inf()) return d < 0 ? "-INF" : "+INF";
  if (d == 0 && std::signbit(d)) return "-0.0";
  if (std::fabs(d) >= 16777216.0) std::snprintf(b, sizeof b, "%.20e", d);
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
    } else if (!uniform && b.kind == Kind::Imm && b.imm == 1 && !x) {
      ins.mods.push_back("IADD");
      if (!is_signed) ins.mods.push_back("U32");
    } else if (!uniform && b.kind == Kind::Imm && c_rz && !is_signed && !x && b.imm > 0 &&
               b.imm < 0x10000 && (b.imm & (b.imm - 1)) == 0) {
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
  if (p0 != kPT || p1 != kPT) ins.dst.push_back(uniform ? UP(p0) : P(p0));
  if (p1 != kPT) ins.dst.push_back(uniform ? UP(p1) : P(p1));
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
  if (ins.src[0].kind == Kind::Reg) ins.src[0].reuse = reuse(w, 1);   // it is operand b
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
  if (mode == 2) ins.mods.push_back("DIV");
  if (mode == 3) ins.mods.push_back("CONV");
  const unsigned cond = static_cast<unsigned>(w.field(87, 3));
  if (cond != kPT || w.bit(90)) ins.src.push_back(pred_src(w, 87, 90));
  if (mode >= 2) {
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
  Operand b;
  b.kind = Kind::Bar;
  b.reg = static_cast<unsigned>(w.field(16, 4));
  ins.dst.push_back(b);
  ins.src.push_back(label(branch_target(w, ins.pc, 0)));
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
    if (t.empty()) t = off == 0 && ra == kRZ ? "RZ" : signed_hex(off);
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
  if (!shared) evict_mods(ins, w);
  if (*kMemSize[size]) ins.mods.push_back(kMemSize[size]);
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
  const unsigned dsize = static_cast<unsigned>(w.field(75, 2)), ssize = static_cast<unsigned>(w.field(84, 2));
  const bool sgn = w.bit(74);
  if (dsize != 2) ins.mods.push_back(float_type(dsize));
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
  static const char* const dts[] = {"F16", "BF16", "(2)", "(3)", "(4)", "(5)", "E5M2", "E4M3"};
  static const char* const sts[] = {"F32", "(1)", "E5M2", "E4M3"};
  const unsigned dt = static_cast<unsigned>(w.field(76, 3)), st = static_cast<unsigned>(w.field(73, 2));
  const unsigned mode = static_cast<unsigned>(w.field(87, 3));
  if (w.bit(75)) ins.mods.push_back("RELU");   // negatives become zero
  if (w.bit(90)) ins.mods.push_back("SATFINITE");
  if (ins.sm >= 89) {
    ins.mods.push_back(dts[dt]);
    ins.mods.push_back(sts[st]);
  } else if (dt != 0) {
    ins.mods.push_back(dts[dt]);
  }
  ins.mods.push_back(mode == 4 ? "UNPACK_B" : mode == 1 ? "PACK_AB_MERGE_C" : "PACK_AB");
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
  mma_regs(ins, w, ".ROW", ".COL");
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
  ins.mods.push_back("884");
  const unsigned rnd = static_cast<unsigned>(w.field(78, 2));
  if (rnd) ins.mods.push_back(kRnd[rnd]);
  ins.f[0] = rnd;
  mma_regs(ins, w);
}

void dec_movm(Instr& ins, const Word& w) {
  ins.op = Op::MOVM;
  ins.mnemonic = "MOVM";
  ins.mods.push_back("16");
  ins.mods.push_back("MT88");
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
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
  ins.mods.push_back(ops[op]);
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
  if (w.field(72, 2)) ins.mods.push_back("B" + std::to_string(w.field(72, 2)));
  ins.f[0] = static_cast<uint32_t>(w.field(72, 2));
  ins.dst.push_back(Txt("PR"));
  alu2(ins, w, kUnsigned);
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
  one_src(ins, w, kUnsigned, false);
}

void dec_nanosleep(Instr& ins, const Word& w) {
  ins.op = Op::NANOSLEEP;
  ins.mnemonic = "NANOSLEEP";
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
  if (pd != kPT) ins.dst.push_back(P(pd));
  ins.dst.push_back(UR(static_cast<unsigned>(w.field(16, ureg_bits(ins.sm))), ins.sm));
  ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8))));
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
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), w.bit(90) || w.bit(72), "", ur, w.sfield(40, 24), ins.sm));
  ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8))));
}

// RED: an atomic with no result.
void dec_red(Instr& ins, const Word& w) {
  ins.op = Op::RED;
  ins.mnemonic = "RED";
  if (w.bit(72)) ins.mods.push_back("E");
  const unsigned aop = static_cast<unsigned>(w.field(87, 3));
  ins.mods.push_back(kAtomOp[aop]);
  atom_type_mods(ins, w);
  mem_order_mods(ins, w);
  evict_mods(ins, w);
  ins.f[0] = aop;
  const int ur = w.bit(91) && !w.bit(72) ? static_cast<int>(w.field(64, ureg_bits(ins.sm))) : -1;
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), w.bit(90), "", ur, w.sfield(40, 24), ins.sm));
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
void dec_atoms_popc(Instr& ins, const Word& w) {
  ins.op = Op::ATOMS;
  ins.mnemonic = "ATOMS";
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

void tex_common(Instr& ins, const Word& w, bool with_mask) {
  ins.dst.push_back(R(static_cast<unsigned>(w.field(64, 8))));
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  ins.src.push_back(R(static_cast<unsigned>(w.field(24, 8))));
  // A second coordinate register of RZ (a 1D lookup) is not printed.
  if (w.field(32, 8) != kRZ) ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8))));
  ins.src.push_back(Imm(w.field(54, 5)));
  ins.src.push_back(Imm(w.field(40, 14)));   // the header's word index in the bank
  ins.src.push_back(Txt(kTexDim[w.field(61, 3)]));
  if (with_mask && w.field(72, 4) != 0xf) ins.src.push_back(Imm(w.field(72, 4)));   // all four: the default
  ins.f[0] = static_cast<uint32_t>(w.field(61, 3));
  ins.f[1] = static_cast<uint32_t>(w.field(72, 4));
}

// The LOD mode, 87-89: nothing (auto), .LZ, .LB, .LL, .LC? (checked against
// nvdisasm as they appear).
const char* const kLod[] = {"", "LZ", "LB", "LL", "LBA", "LLA", "(6)", "(7)"};

void dec_tex(Instr& ins, const Word& w) {
  ins.op = Op::TEX;
  ins.mnemonic = "TEX";
  if (w.bit(60)) ins.mods.push_back("SCR");
  const unsigned lod = static_cast<unsigned>(w.field(87, 3));
  if (lod) ins.mods.push_back(kLod[lod]);
  ins.f[2] = lod;
  tex_common(ins, w, true);
}

void dec_tld(Instr& ins, const Word& w) {
  ins.op = Op::TLD;
  ins.mnemonic = "TLD";
  if (w.bit(60)) ins.mods.push_back("SCR");
  const unsigned lod = static_cast<unsigned>(w.field(87, 3));
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
  ins.src.push_back(Imm(w.field(54, 5)));
  ins.src.push_back(Imm(w.field(40, 14)));
}

void dec_sust(Instr& ins, const Word& w) {
  ins.op = Op::SUST;
  ins.mnemonic = "SUST";
  surf_mods(ins, w);
  const unsigned size = static_cast<unsigned>(w.field(73, 3));
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), false, "", -1, 0, ins.sm));
  ins.src.push_back(R(static_cast<unsigned>(w.field(32, 8)), mem_regs(size)));
  ins.src.push_back(Imm(w.field(54, 5)));
  ins.src.push_back(Imm(w.field(40, 14)));
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
  ins.mods.push_back("CAS");
  atom_type_mods(ins, w);
  mem_order_mods(ins, w);
  evict_mods(ins, w);
  ins.f[5] = 1;   // CAS
  const unsigned t = static_cast<unsigned>(w.field(73, 3));
  const unsigned width = t == 2 || t == 5 || t == 6 ? 2 : 1;
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8)), width));
  const int ur = w.bit(91) && !w.bit(72) ? static_cast<int>(w.field(64, ureg_bits(ins.sm))) : -1;
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), w.bit(90), "", ur, w.sfield(40, 24), ins.sm));
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
  if (w.bit(72)) ins.mods.push_back("LTC128B");
  const unsigned size = static_cast<unsigned>(w.field(73, 3));
  if (*kMemSize[size]) ins.mods.push_back(kMemSize[size]);
  if (w.bit(82)) ins.mods.push_back("ZFILL");
  ins.f[0] = size;
  ins.f[1] = w.bit(82);
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(16, 8)), false, "", -1,
                             static_cast<int64_t>(w.field(44, 20)), ins.sm));
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), true, "", -1, w.sfield(32, 12), ins.sm));
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
  static const char* const spaces[] = {"G", "L", "S", "(3)"};
  ins.mods.push_back(spaces[w.field(73, 2)]);
  ins.dst.push_back(P(static_cast<unsigned>(w.field(81, 3))));
  ins.dst.push_back(R(static_cast<unsigned>(w.field(16, 8))));
  ins.src.push_back(mem_addr(static_cast<unsigned>(w.field(24, 8)), false, "", -1, w.sfield(40, 24), ins.sm));
}

void dec_errbar(Instr& ins, const Word&) {
  ins.op = Op::ERRBAR;
  ins.mnemonic = "ERRBAR";
}

void dec_break(Instr& ins, const Word& w) {
  ins.op = Op::BREAK;
  ins.mnemonic = "BREAK";
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
      {0x25, dec_imad},  {0x27, dec_imad},  {0x1a, dec_sgxt},  {0x1b, dec_ubmsk},
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
      {0x35, [](Instr& i, const Word& w) {   // HFMA2 on the tensor pipe
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
      {0x34e, dec_lepc}, {0x98f, dec_cctl}, {0x31c, dec_b2r}, {0x3aa, dec_qspc}, {0x9ab, dec_errbar},
      {0x942, dec_break}, {0x95c, dec_bpt},
      {0x3a9, [](Instr& i, const Word& w) { dec_atom_cas(i, w, Op::ATOMG, "ATOMG"); }},
      {0x9a9, [](Instr& i, const Word& w) { dec_atom_cas(i, w, Op::ATOMG, "ATOMG"); }},
      {0x38b, [](Instr& i, const Word& w) { dec_atom_cas(i, w, Op::ATOM, "ATOM"); }},
      {0x98b, [](Instr& i, const Word& w) { dec_atom_cas(i, w, Op::ATOM, "ATOM"); }},
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
    if (opc == 0x9c3 || opc == 0xab9 || opc == 0xabb || opc == 0x89c) ins.guard_uniform = true;
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
    case Kind::Imm: s = imm_text(o.imm); break;
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
