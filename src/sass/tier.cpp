// The tier's analysis: a decoded Instr into a MOp (src/sass/tier.hpp). No execution happens here.
//
// Each case below states the same conditions Runner::fast_alu (exec_ops.inc) checks on every execution -- they depend
// on the Instr alone -- and resolves the operands once. A form fast_alu would decline is Form::None, and the
// interpreter runs it. The handlers (tier_ops.inc) compute what fast_alu computes.
#include "tier.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>

namespace vgpu::sass::tier {

namespace {

// How a form's modifiers act on a source's value: integers have -x (0 - x) and ~x, floats |x| and -x on the bits.
enum class Dom { Plain, Int, IntCompl, Flt };

uint32_t apply(Dom d, uint8_t mod, uint32_t v) {
  switch (d) {
    case Dom::Plain: return v;
    case Dom::Int:
      if (mod & kNeg) v = 0u - v;
      if (mod & kNot) v = ~v;
      return v;
    case Dom::IntCompl:   // IADD3: a negated or complemented source is added as its complement (the +1 is counted apart)
      return (mod & (kNeg | kNot)) ? ~v : v;
    case Dom::Flt:
      if (mod & kAbs) v &= 0x7fffffffu;
      if (mod & kNeg) v ^= 0x80000000u;
      return v;
  }
  return v;
}

// The modifier bits of an operand that matter in a domain (the others are what the interpreter ignores for it).
uint8_t mods_of(const Operand& o, Dom d) {
  switch (d) {
    case Dom::Plain: return 0;
    case Dom::Int: return static_cast<uint8_t>((o.neg ? kNeg : 0) | (o.bnot ? kNot : 0));
    case Dom::IntCompl: return (o.neg || o.bnot) ? kNot : 0;
    case Dom::Flt: return static_cast<uint8_t>((o.abs ? kAbs : 0) | (o.neg ? kNeg : 0));
  }
  return 0;
}

struct Builder {
  Program& prog;
  int sm;

  uint32_t const_row(uint32_t v) {
    const auto it = prog.const_index.find(v);
    if (it != prog.const_index.end()) return it->second;
    Row r;
    for (unsigned l = 0; l < 32; ++l) r.v[l] = v;
    prog.pool.push_back(r);
    return prog.const_index[v] = static_cast<uint32_t>(prog.pool.size() - 1);
  }
  Src constant(uint32_t v) {
    Src s;
    s.kind = SrcKind::Pool;
    s.off = const_row(v);
    return s;
  }
  // A 64-bit constant as two rows, the low words then the high words.
  Src constant64(uint64_t v) {
    Src s;
    s.kind = SrcKind::Pool;
    s.off = const_row(static_cast<uint32_t>(v));
    const uint32_t hi = const_row(static_cast<uint32_t>(v >> 32));
    if (hi != s.off + 1) {   // the pair must be adjacent: append a fresh copy of each
      Row lo_row, hi_row;
      for (unsigned l = 0; l < 32; ++l) {
        lo_row.v[l] = static_cast<uint32_t>(v);
        hi_row.v[l] = static_cast<uint32_t>(v >> 32);
      }
      prog.pool.push_back(lo_row);
      prog.pool.push_back(hi_row);
      s.off = static_cast<uint32_t>(prog.pool.size() - 2);
    }
    return s;
  }
  unsigned urz() const { return sm >= 100 ? 255u : 63u; }
};

// One 32-bit source of an ALU op: a register, RZ, an immediate, a constant-bank word or a uniform register -- what
// fast_alu's plain() accepts -- with `d`'s modifiers. False for anything else, and for a constant bank but 0 (those
// are read through the memory manager).
bool src32(Builder& b, const Operand& o, Dom d, Src& out, uint16_t& maxreg) {
  const uint8_t mod = mods_of(o, d);
  switch (o.kind) {
    case Kind::Reg:
      if (o.reg == kRZ) {
        out = b.constant(apply(d, mod, 0));
        return true;
      }
      out = Src{};
      if (mod) {
        out.kind = SrcKind::Dyn;
        out.dyn = Dyn::Reg;
        out.mod = mod;
        out.off = o.reg;
      } else {
        out.kind = SrcKind::Row;
        out.off = o.reg * 32u;
      }
      maxreg = std::max<uint16_t>(maxreg, static_cast<uint16_t>(o.reg + 1));
      return true;
    case Kind::Imm: out = b.constant(apply(d, mod, static_cast<uint32_t>(o.imm))); return true;
    case Kind::FImm: out = b.constant(apply(d, mod, o.fbits)); return true;
    case Kind::CBank: {
      if (o.reg != 0) return false;
      const uint64_t off = static_cast<uint64_t>(o.imm);
      if (off > 0xffffffffull - 4) {   // cbank32 reads zero past the bank
        out = b.constant(apply(d, mod, 0));
        return true;
      }
      out = Src{};
      out.kind = SrcKind::Dyn;
      out.dyn = Dyn::Bank0;
      out.mod = mod;
      out.off = static_cast<uint32_t>(off);
      return true;
    }
    case Kind::UReg:
      if (o.reg >= b.urz()) {
        out = b.constant(apply(d, mod, 0));
        return true;
      }
      out = Src{};
      out.kind = SrcKind::Dyn;
      out.dyn = Dyn::UReg;
      out.mod = mod;
      out.off = o.reg;
      return true;
    default: return false;
  }
}

bool plain_kind(const Operand& o) {
  return o.kind == Kind::Reg || o.kind == Kind::Imm || o.kind == Kind::FImm || o.kind == Kind::CBank ||
         o.kind == Kind::UReg;
}

// A predicate the executing lanes' result is written to: a plain P<n> (the SIMD forms take no uniform one).
bool pred_dst(const Operand& o) { return o.kind == Kind::Pred; }

// A register destination. RZ is kNoReg.
bool reg_dst(const Operand& o, uint16_t& dst, uint16_t& maxreg) {
  if (o.kind != Kind::Reg) return false;
  if (o.reg == kRZ) {
    dst = kNoReg;
    return true;
  }
  dst = static_cast<uint16_t>(o.reg);
  maxreg = std::max<uint16_t>(maxreg, static_cast<uint16_t>(o.reg + 1));
  return true;
}

// A predicate source as the lane mask is read (fast_alu's pmask): P<n>/PT, UP<n>/UPT, negated or not.
// off = the number, mod kNeg = negated, dyn = UReg for a uniform one (all lanes or none).
Src pred_src(const Operand& o) {
  Src s;
  s.kind = SrcKind::Dyn;
  s.dyn = o.kind == Kind::UPred ? Dyn::UReg : Dyn::Reg;
  s.mod = o.neg ? kNeg : 0;
  s.off = o.reg;
  return s;
}

// One 64-bit source of a double op, as fast_alu's DADD/DMUL/DFMA case reads it: a register pair (its two rows), or a
// value that is the same for every lane.
bool src64(Builder& b, const Operand& o, unsigned i, Src& out, uint32_t& mods, uint16_t& maxreg) {
  const Kind k = o.kind;
  if (!(k == Kind::Reg || k == Kind::UReg || k == Kind::CBank || k == Kind::Imm || k == Kind::FImm)) return false;
  const uint8_t m = static_cast<uint8_t>((o.abs ? kAbs : 0) | (o.neg ? kNeg : 0));
  const auto fold = [&](uint64_t bits) {
    if (o.abs) bits &= 0x7fffffffffffffffull;
    if (o.neg) bits ^= 0x8000000000000000ull;
    return b.constant64(bits);
  };
  if (k == Kind::Reg && o.reg != kRZ) {
    out = Src{};
    out.kind = SrcKind::Row;
    out.off = o.reg * 32u;
    mods |= (o.abs ? 1u : 0u) << (2 * i) | (o.neg ? 1u : 0u) << (2 * i + 1);
    maxreg = std::max<uint16_t>(maxreg, static_cast<uint16_t>(o.reg + 2));
    return true;
  }
  if (k == Kind::Reg) { out = fold(0); return true; }   // RZ
  if (k == Kind::Imm) { out = fold(static_cast<uint64_t>(o.imm)); return true; }
  if (k == Kind::FImm) { out = fold(o.fwidth == 64 ? static_cast<uint64_t>(o.fbits) << 32 : o.fbits); return true; }
  if (k == Kind::UReg) {
    if (o.reg >= b.urz()) { out = fold(0); return true; }
    out = Src{};
    out.kind = SrcKind::Dyn;
    out.dyn = Dyn::UReg;
    out.mod = m;
    out.off = o.reg;
    return true;
  }
  // CBank
  if (o.reg != 0) return false;
  const uint64_t off = static_cast<uint64_t>(o.imm);
  if (off > 0xffffffffull - 16) { out = fold(0); return true; }
  out = Src{};
  out.kind = SrcKind::Dyn;
  out.dyn = Dyn::Bank0;
  out.mod = m;
  out.off = static_cast<uint32_t>(off);
  return true;
}

// A 64-bit integer source (IMAD.WIDE's addend): a register pair, or one value for every lane. -x and ~x on the 64 bits
// are folded into a constant, and left in `mod` for a pair or a value read when the instruction runs.
bool src64i(Builder& b, const Operand& o, uint8_t mod, Src& out, uint16_t& maxreg) {
  const auto fold = [&](uint64_t v) {
    if (mod & kNeg) v = 0 - v;
    if (mod & kNot) v = ~v;
    return b.constant64(v);
  };
  switch (o.kind) {
    case Kind::Reg:
      if (o.reg == kRZ) { out = fold(0); return true; }
      out = Src{};
      out.kind = SrcKind::Row;
      out.mod = mod;
      out.off = o.reg * 32u;
      maxreg = std::max<uint16_t>(maxreg, static_cast<uint16_t>(o.reg + 2));
      return true;
    case Kind::Imm: out = fold(static_cast<uint64_t>(o.imm)); return true;
    case Kind::FImm: out = fold(o.fbits); return true;
    case Kind::UReg:
      if (o.reg >= b.urz()) { out = fold(0); return true; }
      out = Src{};
      out.kind = SrcKind::Dyn;
      out.dyn = Dyn::UReg;
      out.mod = mod;
      out.off = o.reg;
      return true;
    case Kind::CBank: {
      if (o.reg != 0) return false;
      const uint64_t off = static_cast<uint64_t>(o.imm);
      if (off > 0xffffffffull - 16) { out = fold(0); return true; }
      out = Src{};
      out.kind = SrcKind::Dyn;
      out.dyn = Dyn::Bank0;
      out.mod = mod;
      out.off = static_cast<uint32_t>(off);
      return true;
    }
    default: return false;
  }
}

// How many bytes a memory instruction moves (exec_mem's table), and how a sub-word load widens (extend).
unsigned mem_bytes(unsigned size) { return size <= 1 ? 1 : size <= 3 ? 2 : size == 4 ? 4 : size == 5 ? 8 : size == 8 ? 32 : 16; }

}  // namespace

const char* form_name(Form f) {
  static const char* const names[] = {"none",  "mov",   "iadd3", "iadd3x", "imad",  "imad32", "imadwide", "imadhi",
                                      "isetpex", "popc", "i2fp",  "ldg",    "stg",   "lds",    "sts",      "ldl",
                                      "stl",   "lop3",  "lea",    "shf",    "isetp", "fsetp",  "fsel",     "ffma",
                                      "fadd",  "fmul",  "dop"};
  const unsigned i = static_cast<unsigned>(f);
  return i < sizeof names / sizeof names[0] ? names[i] : "?";
}

void analyze(const Instr& ins, int sm, Program& prog, MOp& op) {
  op = MOp{};
  op.ins = &ins;
  op.cls = ins.cls;
  op.straight = ins.straight;
  op.guard = ins.guard;
  op.guard_not = ins.guard_not;
  op.guard_uniform = ins.guard_uniform;
  op.unguarded = ins.guard == kPT && !ins.guard_not;
  op.never = ins.guard == kPT && ins.guard_not;   // Runner::guard_mask: ~(all lanes) is none
  Builder b{prog, sm};
  if (prog.pool.empty()) b.const_row(0);   // pool[0] is the zero row
  uint16_t maxreg = 0;
  MOp m;   // built apart so that a declined instruction leaves `op` as the interpreter's
  m = op;
  const auto accept = [&](Form f) {
    m.form = f;
    m.maxreg = maxreg;
    op = m;
  };

  // Memory instructions: stores have no destination, so these come before the check below.
  switch (ins.op) {
    case Op::LDG: case Op::STG: {
      // Global loads and stores of 1, 2, 4 or 8 bytes on Ampere and later (Turing's address form is the interpreter's).
      const bool store = ins.op == Op::STG;
      const unsigned bytes = mem_bytes(ins.f[0]);
      if (sm < 80 || !(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8) || ins.src.size() < (store ? 2u : 1u)) return;
      const Operand& d = store ? ins.src[1] : ins.dst.empty() ? ins.src[0] : ins.dst[0];
      if (!store && ins.dst.empty()) return;
      if (d.kind != Kind::Reg) return;
      const bool e = ins.w.bit(72), wide = e || ins.w.bit(90), ur = ins.w.bit(91) && !e;
      const unsigned ra = static_cast<unsigned>(ins.w.field(24, 8));
      unsigned u = ~0u;
      if (ur) {
        u = static_cast<unsigned>(ins.w.field(store ? 64 : 32, sm >= 100 ? 8 : 6));
        if (u == (sm >= 100 ? 255u : 63u)) u = ~0u;
      }
      if (ra != kRZ) maxreg = static_cast<uint16_t>(ra + (wide ? 2 : 1));
      if (d.reg != kRZ) maxreg = std::max<uint16_t>(maxreg, static_cast<uint16_t>(d.reg + (bytes == 8 ? 2 : 1)));
      m.a = bytes;
      m.b = ins.f[0];
      m.s[0].off = ra;
      m.s[1].off = u;
      m.imm = ins.src[0].imm;
      m.aux = wide ? 1 : 0;
      m.dst = d.reg == kRZ ? kNoReg : static_cast<uint16_t>(d.reg);
      accept(store ? Form::Stg : Form::Ldg);
      return;
    }
    case Op::LDS: case Op::STS: case Op::LDL: case Op::STL: {
      const bool store = ins.op == Op::STS || ins.op == Op::STL, shared = ins.op == Op::LDS || ins.op == Op::STS;
      const unsigned bytes = mem_bytes(ins.f[0]);
      if (!(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16) || ins.src.size() < (store ? 2u : 1u)) return;
      if (shared && ins.f[3] > 3) return;
      const Operand* d = store ? &ins.src[1] : ins.dst.empty() ? nullptr : &ins.dst[0];
      if (!d || d->kind != Kind::Reg) return;
      const bool ur = ins.w.bit(91);
      const unsigned ra = static_cast<unsigned>(ins.w.field(24, 8));
      unsigned u = ~0u;
      if (ur) {
        u = static_cast<unsigned>(ins.w.field(store ? 64 : 32, sm >= 100 ? 8 : 6));
        if (u == (sm >= 100 ? 255u : 63u)) u = ~0u;
      }
      static const unsigned kStrides[] = {0, 4, 8, 16};
      if (ra != kRZ) maxreg = static_cast<uint16_t>(ra + 1);
      if (d->reg != kRZ) maxreg = std::max<uint16_t>(maxreg, static_cast<uint16_t>(d->reg + std::max(1u, bytes / 4)));
      m.a = bytes;
      m.b = ins.f[0];
      m.c = shared ? kStrides[ins.f[3]] : 0;
      m.s[0].off = ra;
      m.s[1].off = u;
      m.imm = ins.src[0].imm;
      m.dst = d->reg == kRZ ? kNoReg : static_cast<uint16_t>(d->reg);
      accept(shared ? (store ? Form::Sts : Form::Lds) : (store ? Form::Stl : Form::Ldl));
      return;
    }
    default: break;
  }
  if (ins.dst.empty()) return;

  switch (ins.op) {
    case Op::ISETP: {
      const unsigned cmp = ins.f[0];
      if (ins.f[4] || ins.dst.size() < 2 || !pred_dst(ins.dst[0]) || !pred_dst(ins.dst[1]) || ins.src.size() < 3)
        return;
      if (cmp < 1 || cmp > 6) return;   // false and true: the interpreter's loop
      // .EX: the high words of a 64-bit compare, the low words' answer coming in as a predicate (exec_int).
      const bool ex_form = ins.f[3] != 0;
      if (ex_form && (ins.guard_uniform || ins.src.size() < 4)) return;
      for (int i = 0; i < 2; ++i) {
        const Operand& o = ins.src[i];
        if (!(o.kind == Kind::Reg || o.kind == Kind::Imm || o.kind == Kind::CBank || o.kind == Kind::UReg)) return;
        if (!src32(b, o, Dom::Plain, m.s[i], maxreg)) return;
      }
      m.p = pred_src(ins.src[2]);
      m.a = cmp;                         // SASS's LT EQ LE GT NE GE
      m.b = ins.f[2];                    // the combining op: AND, OR, XOR
      m.c = ins.f[1] ? 1 : 0;            // signed
      m.dst = static_cast<uint16_t>(ins.dst[0].reg);
      m.dst2 = static_cast<uint16_t>(ins.dst[1].reg);
      if (ex_form) m.p2 = pred_src(ins.src[3]);
      accept(ex_form ? Form::IsetpEx : Form::Isetp);
      return;
    }
    case Op::FSETP: {
      if (ins.dst.size() < 2 || !pred_dst(ins.dst[0]) || !pred_dst(ins.dst[1]) || ins.src.size() < 3) return;
      for (int i = 0; i < 2; ++i) {
        const Operand& o = ins.src[i];
        if (!(o.kind == Kind::Reg || o.kind == Kind::Imm || o.kind == Kind::FImm || o.kind == Kind::CBank ||
              o.kind == Kind::UReg) ||
            o.bnot)
          return;
        // The flush to zero comes before |x| and -x (fv in fast_alu), so the source keeps its raw bits and the
        // modifiers go to `aux`.
        if (!src32(b, o, Dom::Plain, m.s[i], maxreg)) return;
      }
      m.p = pred_src(ins.src[2]);
      m.a = ins.f[0];
      m.b = ins.f[1];
      m.c = ins.f[2] ? 1 : 0;   // flush to zero
      m.aux = static_cast<uint8_t>((ins.src[0].abs ? 1 : 0) | (ins.src[0].neg ? 2 : 0) | (ins.src[1].abs ? 4 : 0) |
                                   (ins.src[1].neg ? 8 : 0));
      m.dst = static_cast<uint16_t>(ins.dst[0].reg);
      m.dst2 = static_cast<uint16_t>(ins.dst[1].reg);
      accept(Form::Fsetp);
      return;
    }
    case Op::DADD: case Op::DMUL: case Op::DFMA: {
      const unsigned nsrc = ins.op == Op::DFMA ? 3 : 2;
      if (ins.f[0] || ins.guard_uniform || ins.dst[0].kind != Kind::Reg || ins.src.size() < nsrc) return;
      uint32_t mods = 0;
      for (unsigned i = 0; i < nsrc; ++i)
        if (!src64(b, ins.src[i], i, m.s[i], mods, maxreg)) return;
      for (unsigned i = nsrc; i < 3; ++i) m.s[i] = b.constant64(0);   // dop_m ignores c for DADD/DMUL
      const Operand& d = ins.dst[0];
      m.a = ins.op == Op::DADD ? 0 : ins.op == Op::DMUL ? 1 : 2;
      m.b = mods;
      if (d.reg == kRZ) {
        m.dst = kNoReg;
      } else {
        m.dst = static_cast<uint16_t>(d.reg);
        maxreg = std::max<uint16_t>(maxreg, static_cast<uint16_t>(d.reg + 1 == kRZ ? d.reg + 1 : d.reg + 2));
      }
      accept(Form::Dop);
      return;
    }
    default: break;
  }

  // The rest write a register (LOP3 with a predicate: dst[0] the predicate, dst[1] the register).
  if (ins.dst[0].kind != Kind::Reg && !(ins.op == Op::LOP3 && ins.dst.size() == 2)) return;

  switch (ins.op) {
    case Op::MOV: {
      if (ins.mnemonic != "MOV" || ins.f[1] || ins.f[0] != 0xf || ins.src.empty() || !plain_kind(ins.src[0])) return;
      if (!src32(b, ins.src[0], Dom::Plain, m.s[0], maxreg) || !reg_dst(ins.dst[0], m.dst, maxreg)) return;
      accept(Form::Mov);
      return;
    }
    case Op::IADD3: {
      if (ins.f[1] || ins.src.size() < 3) return;
      const bool x = ins.f[0];
      bool simple = !x;
      for (size_t i = 1; i < ins.dst.size(); ++i)   // carry-outs other than PT
        if (!(ins.dst[i].kind == Kind::Pred && ins.dst[i].reg == kPT)) simple = false;
      if (simple) {
        for (unsigned i = 0; i < 3; ++i) {
          if (!plain_kind(ins.src[i])) return;
          if (!src32(b, ins.src[i], Dom::IntCompl, m.s[i], maxreg)) return;
        }
        if (!reg_dst(ins.dst[0], m.dst, maxreg)) return;
        m.a = (ins.src[0].neg ? 1u : 0u) + (ins.src[1].neg ? 1u : 0u) + (ins.src[2].neg ? 1u : 0u);
        accept(Form::Iadd3);
        return;
      }
      // The general form (exec_int): carries out to dst[1] and dst[2], carries in from src[3] and src[4].
      if (ins.guard_uniform || ins.dst.size() > 3 || (x && ins.src.size() < 5)) return;
      for (size_t i = 1; i < ins.dst.size(); ++i)
        if (!pred_dst(ins.dst[i])) return;
      for (unsigned i = 0; i < 3; ++i) {
        if (!plain_kind(ins.src[i])) return;
        if (!src32(b, ins.src[i], Dom::IntCompl, m.s[i], maxreg)) return;
      }
      if (!reg_dst(ins.dst[0], m.dst, maxreg)) return;
      m.a = (ins.src[0].neg ? 1u : 0u) + (ins.src[1].neg ? 1u : 0u) + (ins.src[2].neg ? 1u : 0u);
      if (x) {
        m.p = pred_src(ins.src[3]);
        m.p2 = pred_src(ins.src[4]);
      }
      m.aux = x ? 1 : 0;
      m.dst2 = ins.dst.size() > 1 ? static_cast<uint16_t>(ins.dst[1].reg) : kNoReg;
      m.c = ins.dst.size() > 2 ? ins.dst[2].reg : kNoReg;
      accept(Form::Iadd3x);
      return;
    }
    case Op::IMAD: {
      const unsigned kind = ins.f[0];
      const bool simple = !(kind == 0x25 || kind == 0x27 || ins.f[2] || ins.dst.size() > 1 || ins.src.size() < 3);
      if (simple) {
        const auto zero = [](const Operand& o) {
          return !o.bnot && ((o.kind == Kind::Reg && o.reg == kRZ) || (o.kind == Kind::Imm && o.imm == 0));
        };
        if (!reg_dst(ins.dst[0], m.dst, maxreg)) return;
        if ((zero(ins.src[0]) || zero(ins.src[1])) && plain_kind(ins.src[2])) {
          // IMAD.MOV (RZ * RZ + c) and the like: the addend with its modifiers, through s[0].
          if (!src32(b, ins.src[2], Dom::Int, m.s[0], maxreg)) return;
          accept(Form::Mov);
          return;
        }
        for (unsigned i = 0; i < 3; ++i) {
          if (!plain_kind(ins.src[i])) return;
          if (!src32(b, ins.src[i], Dom::Int, m.s[i], maxreg)) return;
        }
        accept(Form::Imad);
        return;
      }
      // The general forms (exec_int): a carry in (.X) and out, the 64-bit product (.WIDE) and its high half (.HI).
      const bool x = ins.f[2], sgn = ins.f[1];
      if (ins.guard_uniform || ins.dst[0].kind != Kind::Reg || ins.dst.size() > 2 || ins.src.size() < (x ? 4u : 3u)) return;
      if (ins.dst.size() == 2 && !pred_dst(ins.dst[1])) return;
      if (x && ins.src[3].kind != Kind::Pred && ins.src[3].kind != Kind::UPred) return;
      for (unsigned i = 0; i < 2; ++i) {
        if (!plain_kind(ins.src[i])) return;
        if (!src32(b, ins.src[i], Dom::Int, m.s[i], maxreg)) return;
      }
      if (!reg_dst(ins.dst[0], m.dst, maxreg)) return;
      if (x) m.p = pred_src(ins.src[3]);
      m.dst2 = ins.dst.size() == 2 ? static_cast<uint16_t>(ins.dst[1].reg) : kNoReg;
      m.aux = static_cast<uint8_t>((x ? 1 : 0) | (sgn ? 2 : 0));
      const Operand& c2 = ins.src[2];
      if (kind == 0x25) {   // WIDE: a 64-bit product plus a 64-bit addend, into a register pair
        if (ins.dst[0].reg != kRZ) maxreg = std::max<uint16_t>(maxreg, static_cast<uint16_t>(ins.dst[0].reg + (ins.dst[0].reg + 1 == kRZ ? 1 : 2)));
        const uint8_t mod = static_cast<uint8_t>((c2.neg ? kNeg : 0) | (c2.bnot ? kNot : 0));
        if (!src64i(b, c2, mod, m.s[2], maxreg)) return;
        accept(Form::ImadWide);
        return;
      }
      if (kind == 0x27) {   // HI: the high word of the product plus a 64-bit addend
        if (c2.kind == Kind::Reg || c2.kind == Kind::UReg) {
          if (!src64i(b, c2, static_cast<uint8_t>(c2.neg ? kNeg : 0), m.s[2], maxreg)) return;
          m.aux |= 4;   // the addend is a pair
        } else {
          if (!plain_kind(c2)) return;
          if (!src32(b, c2, Dom::Int, m.s[2], maxreg)) return;
        }
        accept(Form::ImadHi);
        return;
      }
      if (!plain_kind(c2)) return;
      if (!src32(b, c2, Dom::Int, m.s[2], maxreg)) return;
      accept(Form::Imad32);
      return;
    }
    case Op::LOP3: {
      const bool has_p = ins.dst.size() > 1;
      if (has_p && (ins.dst.size() != 2 || !pred_dst(ins.dst[0]) || ins.dst[1].kind != Kind::Reg)) return;
      if (ins.src.size() < 3) return;
      for (unsigned i = 0; i < 3; ++i) {
        if (!plain_kind(ins.src[i])) return;
        if (!src32(b, ins.src[i], Dom::Plain, m.s[i], maxreg)) return;
      }
      if (has_p && ins.src.size() < 5) return;
      if (!reg_dst(has_p ? ins.dst[1] : ins.dst[0], m.dst, maxreg)) return;
      m.a = ins.f[0];   // the truth table
      if (has_p) {
        m.aux = static_cast<uint8_t>(1 | (ins.f[1] ? 2 : 0));   // has a predicate; .PAND: all ones rather than any
        m.dst2 = static_cast<uint16_t>(ins.dst[0].reg);
        m.p = pred_src(ins.src[4]);   // the predicate the answer is ORed with (not read with .PAND)
      }
      accept(Form::Lop3);
      return;
    }
    case Op::LEA: {   // (a << s) + b, with .HI, .SX32, .X and the carry out
      const bool hi = ins.f[0], x = ins.f[1], sx32 = ins.f[2];
      const unsigned sh = ins.f[3];
      const bool three = hi && !sx32;   // the third source is the high word
      const size_t pidx = three ? 4 : 3;
      if (ins.src.size() < (x ? pidx + 1 : pidx) || ins.dst.size() > 2) return;
      if (ins.dst.size() == 2 && !pred_dst(ins.dst[1])) return;
      if (!plain_kind(ins.src[0]) || !plain_kind(ins.src[1]) || ins.src[0].neg || ins.src[0].bnot) return;
      if (three && (!plain_kind(ins.src[2]) || ins.src[2].neg || ins.src[2].bnot)) return;
      if (!src32(b, ins.src[0], Dom::Plain, m.s[0], maxreg)) return;
      if (!src32(b, ins.src[1], Dom::Int, m.s[1], maxreg)) return;
      if (three && !src32(b, ins.src[2], Dom::Plain, m.s[2], maxreg)) return;
      if (x) m.p = pred_src(ins.src[pidx]);
      if (!reg_dst(ins.dst[0], m.dst, maxreg)) return;
      m.dst2 = ins.dst.size() == 2 ? static_cast<uint16_t>(ins.dst[1].reg) : kNoReg;
      m.a = sh;
      m.aux = static_cast<uint8_t>((hi ? 1 : 0) | (x ? 2 : 0) | (three ? 4 : 0) | (ins.src[1].neg ? 8 : 0));
      accept(Form::Lea);
      return;
    }
    case Op::SHF: {   // a funnel shift of (src2:src0) by src1
      if (ins.dst.size() != 1 || ins.src.size() < 3) return;
      for (int i = 0; i < 3; ++i)
        if (!plain_kind(ins.src[i]) || ins.src[i].neg || ins.src[i].bnot) return;
      for (int i = 0; i < 3; ++i)
        if (!src32(b, ins.src[i], Dom::Plain, m.s[i], maxreg)) return;
      if (!reg_dst(ins.dst[0], m.dst, maxreg)) return;
      m.a = ins.f[0];   // 0 S64, 1 U64, 2 S32, 3 U32
      m.aux = static_cast<uint8_t>((ins.f[1] ? 1 : 0) | (ins.f[2] ? 2 : 0) | (ins.f[3] ? 4 : 0));   // wrap, right, hi
      accept(Form::Shf);
      return;
    }
    case Op::FSEL: {
      if (ins.src.size() < 3 || ins.dst.size() != 1) return;
      if (!plain_kind(ins.src[0]) || !plain_kind(ins.src[1]) || ins.src[0].bnot || ins.src[1].bnot) return;
      if (!src32(b, ins.src[0], Dom::Flt, m.s[0], maxreg) || !src32(b, ins.src[1], Dom::Flt, m.s[1], maxreg)) return;
      m.p = pred_src(ins.src[2]);
      if (!reg_dst(ins.dst[0], m.dst, maxreg)) return;
      accept(Form::Fsel);
      return;
    }
    case Op::POPC: {
      if (ins.guard_uniform || ins.src.empty() || !plain_kind(ins.src[0])) return;
      if (!src32(b, ins.src[0], Dom::Int, m.s[0], maxreg) || !reg_dst(ins.dst[0], m.dst, maxreg)) return;
      accept(Form::Popc);
      return;
    }
    case Op::I2FP: {   // I2FP.F32.{U32,S32}, to nearest (exec_float: the other roundings set the host mode)
      if (ins.guard_uniform || ins.f[1] != 0 || ins.src.empty() || !plain_kind(ins.src[0])) return;
      if (!src32(b, ins.src[0], Dom::Plain, m.s[0], maxreg) || !reg_dst(ins.dst[0], m.dst, maxreg)) return;
      m.c = ins.f[0] != 0 ? 1 : 0;   // signed
      accept(Form::I2fp);
      return;
    }
    case Op::FFMA: case Op::FADD: case Op::FMUL: {
      const unsigned nsrc = ins.op == Op::FFMA ? 3 : 2;
      if (ins.f[0] || ins.f[1] || ins.f[2] || (ins.op == Op::FMUL && ins.f[4] != 4) || ins.src.size() < nsrc) return;
      for (unsigned i = 0; i < nsrc; ++i) {
        const Operand& o = ins.src[i];
        if (!plain_kind(o) || o.bnot) return;
        if (!src32(b, o, Dom::Flt, m.s[i], maxreg)) return;
      }
      if (!reg_dst(ins.dst[0], m.dst, maxreg)) return;
      accept(ins.op == Op::FFMA ? Form::Ffma : ins.op == Op::FADD ? Form::Fadd : Form::Fmul);
      return;
    }
    default: break;
  }
}

namespace {
bool read_request() {
  const char* v = std::getenv("VGPU_TIER");
  return v && v[0] == '1' && v[1] == 0;
}
}  // namespace

bool requested() {
  static const bool on = read_request();
  return on;
}

}  // namespace vgpu::sass::tier
