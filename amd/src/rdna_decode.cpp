// Decoding RDNA machine code (gfx11, RDNA3): the same kind of instructions
// as CDNA's -- scalar, vector, memory -- in encodings and numberings of its
// own, and printed as LLVM's disassembler prints gfx11 code.
//
// What each encoding's bits are, and every instruction's number and operands,
// come from AMD's machine-readable ISA specification (MIT-licensed), turned
// into rdna_ops_rdna3.inc by amd/tools/rdna-ops.py. The decoder reads an
// instruction's fields by its encoding and its operands by that table; what
// is left is how the assembler writes each kind of operand and modifier.
//
// Code for these GPUs runs 32 lanes to a wave (wave32), which is what HIP
// builds for them and what the disassembler assumes: a lane mask -- VCC, a
// comparison's result, a carry -- is one scalar register.
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "rdna_decode.hpp"
#include "vgpu/error.hpp"

namespace vgpu::amd::gcn {
std::string swizzle_text(uint32_t imm);          // gcn_decode.cpp: the same patterns as gfx9's
bool known_gfx9_name(const std::string& name);   // gcn_decode.cpp
int gfx9_opcode(const std::string& name);        // gcn_decode.cpp
}

namespace vgpu::amd::gcn::rdna {
namespace {

enum class K { Vgpr, Src, Ssrc, Sreg, Sdst, Simm16, Simm32, Label, Hwreg, Sendmsg, Waitcnt, Depctr, Delay, Vcc, Exec };
struct Opnd {
  const char* field;
  K kind;
  uint16_t bits;
  bool out;
};
struct Row {
  Enc enc;
  uint8_t segment;
  uint16_t opcode;
  bool sdst;
  const char* name;
  std::vector<Opnd> ops;
};

const std::vector<Row>& rows_rdna3() {
  static const std::vector<Row> rows = {
#include "rdna_ops_rdna3.inc"
  };
  return rows;
}

struct Table {
  std::map<std::tuple<Enc, uint8_t, uint32_t>, const Row*> by_opcode;
  std::map<std::string, const Row*> short_forms;   // VOP1, VOP2 and VOPC names
  std::map<std::string, const Row*> long_forms;    // VOP3 names
};
const Table& table() {
  static const Table t = [] {
    Table t;
    for (const Row& r : rows_rdna3()) {
      t.by_opcode.emplace(std::make_tuple(r.enc, r.segment, static_cast<uint32_t>(r.opcode)), &r);
      if (r.enc == Enc::Vop1 || r.enc == Enc::Vop2 || r.enc == Enc::Vopc) t.short_forms.emplace(r.name, &r);
      if (r.enc == Enc::Vop3) t.long_forms.emplace(r.name, &r);
    }
    return t;
  }();
  return t;
}
const Row& row(Enc e, uint8_t segment, uint32_t opcode) {
  const auto& m = table().by_opcode;
  const auto it = m.find({e, segment, opcode});
  if (it == m.end())
    throw Error::make(Err::Unsupported, enc_name(e), " opcode ", opcode, " (gfx11) is not decoded yet");
  return *it->second;
}

// What the executor knows an instruction by: the gfx9 instruction that does
// the same, spelled as gfx9's decoder spells it -- its old name where the
// specification records one (global_load_b32 was global_load_dword) or it is
// one gfx11 renamed without saying (v_add_nc_u32 was v_add_u32), with the
// suffix gfx9 gives its short (_e32) and long (_e64) forms. An instruction gfx9
// does not have keeps its own name, and the executor knows it by that.
std::string exec_name(const std::string& name, Enc enc, bool dpp) {
  static const std::map<std::string, std::vector<std::string>> kAliases = {
#include "rdna_ops_rdna3_aliases.inc"
  };
  static const std::map<std::string, std::string> kRenamed = {
      {"v_add_nc_u32", "v_add_u32"},         {"v_sub_nc_u32", "v_sub_u32"},
      {"v_subrev_nc_u32", "v_subrev_u32"},   {"v_add_co_ci_u32", "v_addc_co_u32"},
      {"v_sub_co_ci_u32", "v_subb_co_u32"},  {"v_subrev_co_ci_u32", "v_subbrev_co_u32"},
      {"v_add_nc_u16", "v_add_u16"},         {"v_sub_nc_u16", "v_sub_u16"},
      {"v_add_nc_i32", "v_add_i32"},         {"v_sub_nc_i32", "v_sub_i32"},
      {"v_add_nc_i16", "v_add_i16"},         {"v_sub_nc_i16", "v_sub_i16"},
      {"v_dot2acc_f32_f16", "v_dot2c_f32_f16"},
  };
  std::vector<std::string> candidates;
  if (const auto it = kRenamed.find(name); it != kRenamed.end()) candidates.push_back(it->second);
  if (const auto it = kAliases.find(name); it != kAliases.end())
    candidates.insert(candidates.end(), it->second.begin(), it->second.end());
  candidates.push_back(name);
  const bool short_form = enc == Enc::Vop1 || enc == Enc::Vop2 || enc == Enc::Vopc || enc == Enc::Vopd;
  // gfx9's table keeps a flat, global or scratch access by the name after
  // its segment's prefix.
  if (enc == Enc::Flat) {
    const size_t us = name.find('_');
    for (const std::string& c : candidates) {
      const size_t cu = c.find('_');
      if (known_gfx9_name(c.substr(cu + 1))) return name.substr(0, us) + c.substr(cu);
    }
    return name;
  }
  for (const std::string& c : candidates) {
    if (known_gfx9_name(c + "_e32")) return c + (dpp ? "_dpp" : short_form ? "_e32" : "_e64");
    if (known_gfx9_name(c)) return c;
  }
  // RDNA's own, suffixed as gfx9 suffixes an instruction with a short and a
  // long form, so the executor reads both as one.
  if (table().short_forms.count(name)) return name + (dpp ? "_dpp" : short_form ? "_e32" : "_e64");
  return name;
}

uint32_t word(const std::vector<uint8_t>& code, uint64_t at) {
  if (at + 4 > code.size()) throw Error::make(Err::Unsupported, "code ends inside an instruction");
  return code[at] | code[at + 1] << 8 | code[at + 2] << 16 | static_cast<uint32_t>(code[at + 3]) << 24;
}
uint32_t bits(uint64_t v, uint32_t hi, uint32_t lo) { return static_cast<uint32_t>((v >> lo) & ((uint64_t{1} << (hi - lo + 1)) - 1)); }

// gfx11's operand numbering: scalar registers, the special ones, the inline
// constants and (256 up) the vector registers. It differs from gfx9's in the
// middle: 124 is null (reads zero, writes vanish) and 125 is M0.
Operand operand(uint32_t code, uint32_t width) {
  Operand o;
  o.width = width;
  if (code <= 105) {
    o.kind = OperandKind::Sgpr;
    o.index = code;
  } else if (code == 106) {
    o.kind = OperandKind::Vcc;
    o.width = width >= 2 ? 2 : 1;
  } else if (code == 107) {
    o.kind = OperandKind::VccHi;
    o.width = 1;
  } else if (code >= 108 && code <= 123) {
    o.kind = OperandKind::Ttmp;
    o.index = code - 108;
  } else if (code == 124) {
    o.kind = OperandKind::Null;
  } else if (code == 125) {
    o.kind = OperandKind::M0;
    o.width = 1;
  } else if (code == 126) {
    o.kind = width >= 2 ? OperandKind::Exec : OperandKind::ExecLo;
    o.width = width >= 2 ? 2 : 1;
  } else if (code == 127) {
    o.kind = OperandKind::ExecHi;
    o.width = 1;
  } else if (code >= 128 && code <= 192) {
    o.kind = OperandKind::Inline;
    o.value = code - 128;
  } else if (code >= 193 && code <= 208) {
    o.kind = OperandKind::Inline;
    o.value = -static_cast<int64_t>(code - 192);
  } else if (code == 235 || code == 237) {
    o.kind = code == 235 ? OperandKind::SharedBase : OperandKind::PrivateBase;
  } else if (code == 236 || code == 238) {
    o.kind = code == 236 ? OperandKind::SharedLimit : OperandKind::PrivateLimit;
  } else if (code >= 240 && code <= 248) {
    static const double kFloats[] = {0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 0.15915494309189532};
    o.kind = OperandKind::InlineFloat;
    o.fvalue = kFloats[code - 240];
  } else if (code == 253) {
    o.kind = OperandKind::Scc;
    o.width = 1;
  } else if (code == 255) {
    o.kind = OperandKind::Literal;
  } else if (code >= 256) {
    o.kind = OperandKind::Vgpr;
    o.index = code - 256;
  } else {
    throw Error::make(Err::Unsupported, "operand ", code, " (gfx11) is one this does not decode yet");
  }
  return o;
}
Operand vgpr(uint32_t index, uint32_t width) {
  Operand o;
  o.kind = OperandKind::Vgpr;
  o.index = index;
  o.width = width;
  return o;
}

// The width, in registers, of an operand of `bits` bits: a lane mask is one
// register in wave32, whatever the specification's 64 says.
uint32_t width_of(const Opnd& op, Enc enc) {
  const bool valu = enc == Enc::Vop1 || enc == Enc::Vop2 || enc == Enc::Vopc || enc == Enc::Vop3 ||
                    enc == Enc::Vop3p || enc == Enc::Vopd;
  const bool scalar = op.kind == K::Sreg || op.kind == K::Sdst || op.kind == K::Vcc || op.kind == K::Exec;
  if (valu && scalar && op.bits == 64) return 1;
  return op.bits <= 32 ? 1 : op.bits / 32;
}

bool is_half(const Opnd& op) { return op.bits == 16; }

}  // namespace

Inst decode(const std::vector<uint8_t>& code, uint64_t at, uint64_t pc, Target target) {
  const uint32_t w0 = word(code, at);
  Inst in;
  in.pc = pc;
  in.arch = target;
  in.size = 4;
  bool literal = false;
  uint64_t w = w0;
  const auto second = [&] {
    w = w0 | static_cast<uint64_t>(word(code, at + 4)) << 32;
    in.size = 8;
  };
  const auto take = [&](uint32_t field, uint32_t width) {
    Operand o = operand(field, width);
    if (o.kind == OperandKind::Literal) literal = true;
    return o;
  };

  // Which encoding: the identifying bits at the top of the first word. A word
  // of zeroes is v_illegal, which traps.
  if (w0 == 0) {
    in.enc = Enc::Vop1;
    in.name = "v_illegal";
    return in;
  }
  if ((w0 >> 23) == 0x17d) {
    in.enc = Enc::Sop1;
    in.opcode = bits(w0, 15, 8);
  } else if ((w0 >> 23) == 0x17e) {
    in.enc = Enc::Sopc;
    in.opcode = bits(w0, 22, 16);
  } else if ((w0 >> 23) == 0x17f) {
    in.enc = Enc::Sopp;
    in.opcode = bits(w0, 22, 16);
  } else if ((w0 >> 28) == 0xb) {
    in.enc = Enc::Sopk;
    in.opcode = bits(w0, 27, 23);
  } else if ((w0 >> 30) == 0x2) {
    in.enc = Enc::Sop2;
    in.opcode = bits(w0, 29, 23);
  } else if ((w0 >> 26) == 0x3d) {
    in.enc = Enc::Smem;
    in.opcode = bits(w0, 25, 18);
    second();
  } else if ((w0 >> 25) == 0x3f) {
    in.enc = Enc::Vop1;
    in.opcode = bits(w0, 16, 9);
  } else if ((w0 >> 25) == 0x3e) {
    in.enc = Enc::Vopc;
    in.opcode = bits(w0, 24, 17);
  } else if ((w0 >> 31) == 0) {
    in.enc = Enc::Vop2;
    in.opcode = bits(w0, 30, 25);
  } else if ((w0 >> 26) == 0x35) {
    in.enc = Enc::Vop3;
    in.opcode = bits(w0, 25, 16);
    second();
  } else if ((w0 >> 24) == 0xcc) {
    in.enc = Enc::Vop3p;
    in.opcode = bits(w0, 22, 16);
    second();
  } else if ((w0 >> 26) == 0x32) {
    in.enc = Enc::Vopd;
    second();
  } else if ((w0 >> 26) == 0x36) {
    in.enc = Enc::Ds;
    in.opcode = bits(w0, 25, 18);
    second();
  } else if ((w0 >> 26) == 0x37) {
    in.enc = Enc::Flat;
    in.opcode = bits(w0, 24, 18);
    second();
  } else if ((w0 >> 26) == 0x38) {
    in.enc = Enc::Mubuf;
    in.opcode = bits(w0, 25, 18);
    second();
  } else if ((w0 >> 26) == 0x3a) {
    in.enc = Enc::Mtbuf;
    in.opcode = bits(w0, 18, 15);
    second();
  } else {
    throw Error::make(Err::Unsupported, "gfx11 instruction word ", w0, " is in an encoding this does not decode yet");
  }

  // VOPD: two instructions in one, X and Y, each its own opcode.
  if (in.enc == Enc::Vopd) {
    const uint32_t opx = bits(w, 25, 22), opy = bits(w, 21, 17);
    const Row& rx = row(Enc::Vopd, 0, opx);
    const Row& ry = row(Enc::Vopd, 0, opy);
    const auto half = [&](const Row& r, uint32_t vdst, uint32_t src0, uint32_t vsrc1, bool y) {
      Inst h;
      h.enc = Enc::Vopd;
      h.arch = target;
      h.opcode = y ? opy : opx;
      h.name = r.name;
      h.pc = pc;
      for (const Opnd& op : r.ops) {
        const std::string f = op.field;
        // v_dual_cndmask_b32's lane mask is VCC's low half, which the
        // assembler leaves unwritten.
        if (op.kind == K::Vcc || op.kind == K::Sreg) continue;
        if (op.kind == K::Simm32) {
          Operand l;
          l.kind = OperandKind::Literal;
          h.src.push_back(l);
          literal = true;
          continue;
        }
        if (op.out) h.dst.push_back(vgpr(vdst, 1));
        else if (f.find("SRC") != std::string::npos && f.find("VSRC") == std::string::npos)
          h.src.push_back(take(src0, 1));
        else h.src.push_back(vgpr(vsrc1, 1));
      }
      // v_dual_cndmask_b32 picks by VCC's low half, which the assembler
      // leaves unwritten but the instruction reads.
      if (h.name == "v_dual_cndmask_b32") {
        Operand v;
        v.kind = OperandKind::Vcc;
        v.hidden = true;
        h.src.push_back(v);
      }
      h.asm_name = h.name;
      h.name = exec_name(h.name.substr(std::string("v_dual_").size()).insert(0, "v_"), Enc::Vopd, false);
      if (const int op9 = gfx9_opcode(h.name); op9 >= 0) h.opcode = static_cast<uint32_t>(op9);
      return h;
    };
    const uint32_t vdstx = bits(w, 63, 56), vdsty = (bits(w, 55, 49) << 1) | ((vdstx & 1) ^ 1);
    in.name = rx.name;
    Inst x = half(rx, vdstx, bits(w, 8, 0), bits(w, 16, 9), false);
    Inst y = half(ry, vdsty, bits(w, 40, 32), bits(w, 48, 41), true);
    in.dual = {x, y};
    in.asm_name = in.name;
    if (literal) {
      const uint32_t value = word(code, at + in.size);
      for (Inst* h : {&in.dual[0], &in.dual[1]})
        for (Operand& o : h->src)
          if (o.kind == OperandKind::Literal) o.value = value;
      in.size += 4;
    }
    return in;
  }

  // The rest are one instruction, the VOP3 form of a VOP1/VOP2/VOPC one
  // included: VOP3 numbers every vector instruction.
  // FLAT's SEG field: 0 flat, 1 scratch, 2 global -- the table keys them
  // 0 flat, 1 global, 2 scratch.
  const uint32_t seg = in.enc == Enc::Flat ? bits(w, 17, 16) : 0;
  const uint8_t segment = static_cast<uint8_t>(seg == 1 ? 2 : seg == 2 ? 1 : 0);
  const Row& r = row(in.enc, segment, in.opcode);
  in.name = r.name;
  if (in.enc == Enc::Flat) in.segment = segment == 1 ? Inst::Segment::Global : segment == 2 ? Inst::Segment::Scratch
                                                                                           : Inst::Segment::Flat;
  in.promoted = in.enc == Enc::Vop3 && table().short_forms.count(r.name);

  // DPP: a VOP1/VOP2/VOPC instruction whose first source is 0xFA (DPP16) or
  // 0xE9/0xEA (DPP8) reads it from another lane, a second word saying which.
  uint32_t src0_field = in.enc == Enc::Vop1 || in.enc == Enc::Vop2 || in.enc == Enc::Vopc ? bits(w0, 8, 0) : 0;
  if (in.enc == Enc::Vop3 || in.enc == Enc::Vop3p) src0_field = bits(w, 40, 32);
  const bool short_form = in.enc == Enc::Vop1 || in.enc == Enc::Vop2 || in.enc == Enc::Vopc;
  const bool vop3ish = in.enc == Enc::Vop3 || in.enc == Enc::Vop3p;
  uint64_t dpp_word = 0;
  if ((short_form || vop3ish) && (src0_field == 0xFA || src0_field == 0xE9 || src0_field == 0xEA)) {
    dpp_word = word(code, at + in.size);
    in.size += 4;
    in.dpp = src0_field == 0xFA;
    in.dpp8 = src0_field != 0xFA;
    in.fi = src0_field == 0xEA;
    if (in.dpp) {
      in.dpp_ctrl = bits(dpp_word, 16, 8);
      in.fi = bits(dpp_word, 18, 18);
      in.bound_ctrl = bits(dpp_word, 19, 19);
      in.row_mask = static_cast<uint8_t>(bits(dpp_word, 31, 28));
      in.bank_mask = static_cast<uint8_t>(bits(dpp_word, 27, 24));
    } else {
      in.dpp_ctrl = bits(dpp_word, 31, 8);   // eight lane selects, three bits each
    }
  }

  // Each operand, from its field.
  const auto field = [&](const std::string& f) -> uint32_t {
    switch (in.enc) {
      case Enc::Sop1:
        return f == "SDST" ? bits(w, 22, 16) : bits(w, 7, 0);
      case Enc::Sop2:
      case Enc::Sopc:
        return f == "SDST" ? bits(w, 22, 16) : f == "SSRC1" ? bits(w, 15, 8) : bits(w, 7, 0);
      case Enc::Sopk:
        return f == "SDST" ? bits(w, 22, 16) : bits(w, 15, 0);
      case Enc::Sopp:
        return bits(w, 15, 0);
      case Enc::Smem:
        return f == "SDATA" ? bits(w, 12, 6) : f == "SBASE" ? bits(w, 5, 0) << 1 : bits(w, 63, 57);
      case Enc::Vop1:
        return f == "VDST" ? bits(w, 24, 17) : bits(w, 8, 0);
      case Enc::Vop2:
        return f == "VDST" ? bits(w, 24, 17) : f == "VSRC1" ? bits(w, 16, 9) : bits(w, 8, 0);
      case Enc::Vopc:
        return f == "VSRC1" ? bits(w, 16, 9) : bits(w, 8, 0);
      case Enc::Vop3:
      case Enc::Vop3p:
        return f == "VDST" ? bits(w, 7, 0) : f == "SDST" ? bits(w, 14, 8) : f == "SRC0" ? bits(w, 40, 32)
               : f == "SRC1" ? bits(w, 49, 41) : bits(w, 58, 50);
      case Enc::Ds:
        return f == "VDST" ? bits(w, 63, 56) : f == "ADDR" ? bits(w, 39, 32) : f == "DATA0" ? bits(w, 47, 40)
                                                                                           : bits(w, 55, 48);
      case Enc::Flat:
        return f == "VDST" ? bits(w, 63, 56) : f == "ADDR" ? bits(w, 39, 32) : f == "DATA" ? bits(w, 47, 40)
                                                                                           : bits(w, 54, 48);
      case Enc::Mubuf:
      case Enc::Mtbuf:
        return f == "VDATA" ? bits(w, 47, 40) : f == "VADDR" ? bits(w, 39, 32) : f == "SRSRC" ? bits(w, 52, 48) << 2
                                                                                              : bits(w, 63, 56);
      default:
        return 0;
    }
  };
  const bool vop3_16 = vop3ish;   // VOP3's 16-bit operands pick their half with OP_SEL
  for (const Opnd& op : r.ops) {
    const std::string f = op.field;
    uint32_t width = width_of(op, in.enc);
    // A buffer address is an offset or an index, one register, or both, two.
    if ((in.enc == Enc::Mubuf || in.enc == Enc::Mtbuf) && f == "VADDR") width = bits(w, 55, 55) && bits(w, 54, 54) ? 2 : 1;
    // A global access with a scalar base takes a 32-bit offset from its
    // address register; without one, a whole 64-bit address. Scratch's is
    // always an offset.
    if (in.enc == Enc::Flat && f == "ADDR" && segment != 0) width = segment == 1 && bits(w, 54, 48) == 124 ? 2 : 1;
    // An atomic returns the old value only with GLC set.
    if (op.out && (in.enc == Enc::Flat || in.enc == Enc::Mubuf) && r.name[0] != 'd' &&
        std::string(r.name).find("_atomic_") != std::string::npos && !bits(w, 14, 14))
      continue;
    Operand o;
    switch (op.kind) {
      case K::Exec:
        continue;   // a v_cmpx's destination, which the assembler does not write
      case K::Vcc:
        o.kind = OperandKind::Vcc;
        o.width = width;
        break;
      case K::Simm16:
        // v_fmaak_f16's constant: a literal word, of which it takes the half.
        if (f == "LITERAL") {
          Operand l;
          l.kind = OperandKind::Literal;
          l.bits16 = true;
          in.src.push_back(l);
          literal = true;
          continue;
        }
        [[fallthrough]];
      case K::Label:
      case K::Hwreg:
      case K::Sendmsg:
      case K::Waitcnt:
      case K::Depctr:
      case K::Delay:
        in.simm = static_cast<int32_t>(field(f));
        if (op.kind == K::Label) in.target = pc + 4 + 4 * static_cast<int64_t>(static_cast<int16_t>(in.simm));
        continue;
      case K::Simm32: {
        Operand l;
        l.kind = OperandKind::Literal;
        in.src.push_back(l);
        literal = true;
        continue;
      }
      case K::Vgpr: {
        uint32_t v = field(f);
        // A 9-bit source field names a vector register from 256.
        if (f.rfind("SRC", 0) == 0 && (in.enc == Enc::Vop3 || in.enc == Enc::Vop3p)) v -= 256;
        if (f == "SRC0" && short_form) v -= 256;
        o = vgpr(v, width);
        // A 16-bit operand of a short-form instruction is half a register,
        // the field's top bit saying which.
        if (is_half(op) && short_form) {
          o.index = v & 0x7F;
          o.hi = v & 0x80;
          o.half = true;
        }
        if (is_half(op)) o.bits16 = true;
        break;
      }
      case K::Src:
      case K::Ssrc:
      case K::Sreg:
      case K::Sdst: {
        uint32_t v = field(f);
        if (in.enc == Enc::Smem && f == "SOFFSET") {
          o = take(v, 1);
          break;
        }
        // DPP's first source is the register the second word names.
        if (f == "SRC0" && (in.dpp || in.dpp8)) v = 256 + static_cast<uint32_t>(bits(dpp_word, 7, 0));
        o = take(v, width);
        if (is_half(op) && short_form && o.kind == OperandKind::Vgpr) {
          o.index &= 0x7F;
          o.hi = (v - 256) & 0x80;
          o.half = true;
        }
        if (is_half(op)) o.bits16 = true;
        break;
      }
    }
    (void)vop3_16;
    (op.out ? in.dst : in.src).push_back(o);
  }

  // The modifiers each encoding carries.
  switch (in.enc) {
    case Enc::Smem:
      in.offset = static_cast<int32_t>(bits(w, 52, 32) << 11) >> 11;   // 21 bits, signed
      in.cache = bits(w, 14, 14) | bits(w, 13, 13) << 2;                // glc, dlc
      break;
    case Enc::Vop1:
    case Enc::Vop2:
    case Enc::Vopc:
      if (in.dpp) {
        const uint32_t neg = bits(dpp_word, 20, 20) | bits(dpp_word, 22, 22) << 1;
        const uint32_t abs = bits(dpp_word, 21, 21) | bits(dpp_word, 23, 23) << 1;
        for (uint32_t k = 0; k < in.src.size() && k < 2; ++k) {
          in.src[k].neg = (neg >> k) & 1;
          in.src[k].abs = (abs >> k) & 1;
        }
      }
      break;
    case Enc::Vop3: {
      const uint32_t abs = bits(w, 10, 8), neg = bits(w, 63, 61);
      if (!r.sdst) {
        for (uint32_t k = 0; k < in.src.size() && k < 3; ++k) {
          in.src[k].abs = (abs >> k) & 1;
          in.src[k].neg = (neg >> k) & 1;
        }
        in.op_sel = static_cast<uint8_t>(bits(w, 14, 11));
      } else {
        for (uint32_t k = 0; k < in.src.size() && k < 3; ++k) in.src[k].neg = (neg >> k) & 1;
      }
      in.clamp = bits(w, 15, 15);
      in.omod = static_cast<uint8_t>(bits(w, 60, 59));
      // A 16-bit vector register operand is half a register: OP_SEL's bit for
      // it (bit 3 the destination's) says which half, and the assembler
      // writes that as .l or .h rather than as op_sel.
      if (!r.sdst) {
        for (uint32_t k = 0; k < in.src.size() && k < 3; ++k)
          if (in.src[k].bits16 && in.src[k].kind == OperandKind::Vgpr) {
            in.src[k].half = true;
            in.src[k].hi = (in.op_sel >> k) & 1;
            in.op_sel &= static_cast<uint8_t>(~(1u << k));
          }
        if (!in.dst.empty() && in.dst[0].bits16 && in.dst[0].kind == OperandKind::Vgpr) {
          in.dst[0].half = true;
          in.dst[0].hi = (in.op_sel >> 3) & 1;
          in.op_sel &= static_cast<uint8_t>(~8u);
        }
      }
      if (in.dpp) {
        const uint32_t dneg = bits(dpp_word, 20, 20) | bits(dpp_word, 22, 22) << 1;
        const uint32_t dabs = bits(dpp_word, 21, 21) | bits(dpp_word, 23, 23) << 1;
        (void)dneg;
        (void)dabs;
      }
      break;
    }
    case Enc::Vop3p:
      in.op_sel = static_cast<uint8_t>(bits(w, 13, 11));
      in.op_sel_hi = static_cast<uint8_t>(bits(w, 60, 59) | bits(w, 14, 14) << 2);
      in.neg_lo = static_cast<uint8_t>(bits(w, 63, 61));
      in.neg_hi = static_cast<uint8_t>(bits(w, 10, 8));
      in.clamp = bits(w, 15, 15);
      // The mixed-precision forms use the negation bits as each source's
      // negation (NEG) and absolute value (NEG_HI), on the operands.
      if (in.name.rfind("v_fma_mix", 0) == 0) {
        for (uint32_t k = 0; k < in.src.size() && k < 3; ++k) {
          in.src[k].neg = (in.neg_lo >> k) & 1;
          in.src[k].abs = (in.neg_hi >> k) & 1;
        }
        in.neg_lo = in.neg_hi = 0;
      }
      break;
    case Enc::Ds:
      // Two 8-bit offsets for the two-address forms; one 16-bit one for the
      // rest (ds_swizzle's pattern among them), as gfx9's decoder keeps it.
      if (std::string(r.name).find("2addr") != std::string::npos) {
        in.offset = static_cast<int32_t>(bits(w, 7, 0));
        in.offset1 = static_cast<int32_t>(bits(w, 15, 8));
      } else {
        in.offset = static_cast<int32_t>(bits(w, 15, 0));
      }
      in.gds = bits(w, 17, 17);
      break;
    case Enc::Flat: {
      in.offset = static_cast<int32_t>(bits(w, 12, 0) << 19) >> 19;   // 13 bits, signed
      if (in.segment == Inst::Segment::Flat) in.offset = static_cast<int32_t>(bits(w, 11, 0));
      in.cache = bits(w, 14, 14) | bits(w, 15, 15) << 1 | bits(w, 13, 13) << 2;   // glc, slc, dlc
      const uint32_t saddr = bits(w, 54, 48);
      in.has_saddr = in.segment != Inst::Segment::Flat && saddr != 124;
      in.saddr = saddr;
      // Scratch: SVE says whether the address register is there at all.
      in.has_vaddr = in.segment != Inst::Segment::Scratch || bits(w, 55, 55);
      break;
    }
    case Enc::Mubuf:
    case Enc::Mtbuf:
      in.offset = static_cast<int32_t>(bits(w, 11, 0));
      in.offen = bits(w, 54, 54);
      in.idxen = bits(w, 55, 55);
      in.cache = bits(w, 14, 14) | bits(w, 12, 12) << 1 | bits(w, 13, 13) << 2;   // glc, slc, dlc
      if (in.enc == Enc::Mtbuf) in.format = bits(w, 25, 19);
      break;
    default:
      break;
  }

  if (literal) {
    const uint32_t value = word(code, at + in.size);
    // A 16-bit operand's literal is the low half of the word.
    for (Operand& o : in.src)
      if (o.kind == OperandKind::Literal) o.value = o.bits16 ? (value & 0xFFFF) : value;
    in.size += 4;
  }
  // A half-register source reads its half as a sub-dword instruction reads
  // a word of a register, which is how the executor runs it (the
  // destination's half is narrowed as it runs).
  for (Operand& o : in.src)
    if (o.half && o.kind == OperandKind::Vgpr) o.sel = o.hi ? 5 : 4;
  in.asm_name = in.name;
  in.name = exec_name(in.name, in.enc, in.dpp || in.dpp8);
  // The gfx9 instruction's number, which the executor reads a comparison's
  // condition and type from.
  if (const int op9 = gfx9_opcode(in.name); op9 >= 0) in.opcode = static_cast<uint32_t>(op9);
  else if (const int e32 = gfx9_opcode(in.name.substr(0, in.name.rfind('_')) + "_e32"); e32 >= 0 && in.dpp)
    in.opcode = static_cast<uint32_t>(e32);
  return in;
}

namespace {

std::string reg_text(const Operand& o) {
  char b[64];
  const auto range = [&](const char* kind) {
    if (o.width <= 1) std::snprintf(b, sizeof b, "%s%u", kind, o.index);
    else std::snprintf(b, sizeof b, "%s[%u:%u]", kind, o.index, o.index + o.width - 1);
    return std::string(b);
  };
  switch (o.kind) {
    case OperandKind::Sgpr: return range("s");   // s102 and s103 are ordinary registers on gfx11
    case OperandKind::Vgpr: return range("v") + (o.half ? (o.hi ? ".h" : ".l") : "");
    case OperandKind::Ttmp: return range("ttmp");
    case OperandKind::Null: return "null";
    case OperandKind::Scc: return "src_scc";
    case OperandKind::SharedLimit: return "src_shared_limit";
    case OperandKind::PrivateLimit: return "src_private_limit";
    default: return operand_text(o);
  }
}
// An operand with its modifiers: -v1, |s2|, -|v[3:4]|.
std::string op_text(const Operand& o) {
  if (o.kind == OperandKind::Inline || o.kind == OperandKind::InlineFloat || o.kind == OperandKind::Literal) {
    Operand plain = o;
    plain.abs = false;
    const std::string t = operand_text(plain);
    return o.abs ? "|" + t + "|" : t;
  }
  Operand plain = o;
  plain.neg = plain.abs = false;
  const std::string neg = o.neg ? "-" : "";
  return o.abs ? neg + "|" + reg_text(plain) + "|" : neg + reg_text(plain);
}

std::string hex(uint32_t v) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%x", v);
  return b;
}

std::string hwreg(uint32_t simm) {
  static const std::map<uint32_t, const char*> kNames = {
      {1, "HW_REG_MODE"},        {2, "HW_REG_STATUS"},      {3, "HW_REG_TRAPSTS"},      {5, "HW_REG_GPR_ALLOC"},
      {6, "HW_REG_LDS_ALLOC"},   {7, "HW_REG_IB_STS"},      {15, "HW_REG_SH_MEM_BASES"}, {20, "HW_REG_FLAT_SCR_LO"},
      {21, "HW_REG_FLAT_SCR_HI"}, {23, "HW_REG_HW_ID1"},    {24, "HW_REG_HW_ID2"},       {25, "HW_REG_POPS_PACKER"},
      {29, "HW_REG_SHADER_CYCLES"}};
  const uint32_t id = simm & 0x3F, offset = (simm >> 6) & 0x1F, size = ((simm >> 11) & 0x1F) + 1;
  const auto it = kNames.find(id);
  std::string out = "hwreg(" + (it != kNames.end() ? std::string(it->second) : std::to_string(id));
  if (offset || size != 32) out += ", " + std::to_string(offset) + ", " + std::to_string(size);
  return out + ")";
}

std::string waitcnt(uint32_t simm) {
  // gfx11: expcnt bits 2:0, lgkmcnt 9:4, vmcnt 15:10.
  const uint32_t exp = simm & 7, lgkm = (simm >> 4) & 0x3F, vm = (simm >> 10) & 0x3F;
  std::string s;
  const auto add = [&](const char* n, uint32_t v) { s += (s.empty() ? "" : " ") + std::string(n) + "(" + std::to_string(v) + ")"; };
  if (vm != 0x3F) add("vmcnt", vm);
  if (exp != 7) add("expcnt", exp);
  if (lgkm != 0x3F) add("lgkmcnt", lgkm);
  return s.empty() ? hex(simm) : s;
}

std::string delay_alu(uint32_t simm) {
  static const char* kId[] = {"NO_DEP",        "VALU_DEP_1",   "VALU_DEP_2",   "VALU_DEP_3",
                              "VALU_DEP_4",    "TRANS32_DEP_1", "TRANS32_DEP_2", "TRANS32_DEP_3",
                              "FMA_ACCUM_CYCLE_1", "SALU_CYCLE_1", "SALU_CYCLE_2", "SALU_CYCLE_3"};
  static const char* kSkip[] = {"SAME", "NEXT", "SKIP_1", "SKIP_2", "SKIP_3", "SKIP_4"};
  const uint32_t id0 = simm & 0xF, skip = (simm >> 4) & 7, id1 = (simm >> 7) & 0xF;
  std::string s;
  if (id0 < 12 && id0) s += std::string("instid0(") + kId[id0] + ")";
  if (skip && skip < 6) s += (s.empty() ? "" : " | ") + std::string("instskip(") + kSkip[skip] + ")";
  if (id1 && id1 < 12) s += (s.empty() ? "" : " | ") + std::string("instid1(") + kId[id1] + ")";
  return s.empty() ? "0" : s;
}

std::string dpp_text(const Inst& i) {
  const uint32_t c = i.dpp_ctrl;
  char b[64];
  if (c <= 0xFF) {
    std::snprintf(b, sizeof b, "quad_perm:[%u,%u,%u,%u]", c & 3, (c >> 2) & 3, (c >> 4) & 3, (c >> 6) & 3);
  } else if (c >= 0x101 && c <= 0x10F) {
    std::snprintf(b, sizeof b, "row_shl:%u", c - 0x100);
  } else if (c >= 0x111 && c <= 0x11F) {
    std::snprintf(b, sizeof b, "row_shr:%u", c - 0x110);
  } else if (c >= 0x121 && c <= 0x12F) {
    std::snprintf(b, sizeof b, "row_ror:%u", c - 0x120);
  } else if (c == 0x140) {
    std::snprintf(b, sizeof b, "row_mirror");
  } else if (c == 0x141) {
    std::snprintf(b, sizeof b, "row_half_mirror");
  } else if (c >= 0x150 && c <= 0x15F) {
    std::snprintf(b, sizeof b, "row_share:%u", c - 0x150);
  } else if (c >= 0x160 && c <= 0x16F) {
    std::snprintf(b, sizeof b, "row_xmask:%u", c - 0x160);
  } else {
    std::snprintf(b, sizeof b, "dpp_ctrl:0x%x", c);
  }
  return b;
}

std::string one(const Inst& i) {
  const std::string& name = i.asm_name.empty() ? i.name : i.asm_name;
  std::string s = name;
  const bool short_form = i.enc == Enc::Vop1 || i.enc == Enc::Vop2 || i.enc == Enc::Vopc;
  // A short form with no long one to tell it from (v_fmaak_f32,
  // v_readfirstlane_b32) has no suffix.
  if (short_form && (i.dpp || i.dpp8)) s += "_dpp";
  else if (short_form && table().long_forms.count(name) && name != "v_readfirstlane_b32" && name != "v_nop")
    s += "_e32";
  if (i.enc == Enc::Vop3 && i.promoted) s += i.dpp || i.dpp8 ? "_e64_dpp" : "_e64";
  if (i.enc == Enc::Vop3 && !i.promoted && (i.dpp || i.dpp8)) s += "_e64_dpp";

  if (i.enc == Enc::Sopp) {
    if (name == "s_waitcnt") return s + " " + waitcnt(static_cast<uint32_t>(i.simm));
    if (name == "s_delay_alu") return s + " " + delay_alu(static_cast<uint32_t>(i.simm));
    if (name == "s_endpgm" || name == "s_barrier" || name == "s_icache_inv" || name == "s_wakeup" ||
        name == "s_code_end" || name == "s_endpgm_saved" || name == "s_ttracedata")
      return s;
    if (name.rfind("s_cbranch", 0) == 0 || name == "s_branch") return s + " " + std::to_string(i.simm);
    if (name == "s_sendmsg") {
      // gfx11's messages a compute kernel sends: 3 gives its vector registers
      // back before it ends.
      if (i.simm == 3) return s + " sendmsg(MSG_DEALLOC_VGPRS)";
      if (i.simm == 1) return s + " sendmsg(MSG_INTERRUPT)";
      return s + " " + hex(static_cast<uint32_t>(i.simm));
    }
    if (name == "s_clause") return s + " " + hex(static_cast<uint32_t>(i.simm));
    if (name == "s_sleep" || name == "s_nop" || name == "s_trap" || name == "s_sethalt" ||
        name == "s_setprio")
      return s + " " + std::to_string(i.simm);
    return s + " " + hex(static_cast<uint32_t>(i.simm));
  }
  if (i.enc == Enc::Sopk) {
    if (name == "s_getreg_b32") return s + " " + reg_text(i.dst[0]) + ", " + hwreg(static_cast<uint32_t>(i.simm));
    if (name == "s_setreg_b32") return s + " " + hwreg(static_cast<uint32_t>(i.simm)) + ", " + reg_text(i.src[0]);
    if (name == "s_setreg_imm32_b32")
      return s + " " + hwreg(static_cast<uint32_t>(i.simm)) + ", " + hex(static_cast<uint32_t>(i.src[0].value));
    std::string t = s;
    std::string sep = " ";
    for (const Operand& o : i.dst) {
      t += sep + reg_text(o);
      sep = ", ";
    }
    for (const Operand& o : i.src) {
      t += sep + reg_text(o);
      sep = ", ";
    }
    return t + sep + hex(static_cast<uint32_t>(i.simm) & 0xFFFF);
  }

  if (name == "s_sendmsg_rtn_b32" || name == "s_sendmsg_rtn_b64") {
    static const std::map<int64_t, const char*> kMsgs = {
        {128, "MSG_RTN_GET_DOORBELL"}, {129, "MSG_RTN_GET_DDID"},  {130, "MSG_RTN_GET_TMA"},
        {131, "MSG_RTN_GET_REALTIME"}, {132, "MSG_RTN_SAVE_WAVE"}, {133, "MSG_RTN_GET_TBA"}};
    const int64_t id = i.simm;
    const auto it = kMsgs.find(id);
    return s + " " + reg_text(i.dst[0]) + ", sendmsg(" + (it != kMsgs.end() ? it->second : std::to_string(id)) + ")";
  }
  std::string sep = " ";
  const auto put = [&](const std::string& t) {
    s += sep + t;
    sep = ", ";
  };
  for (const Operand& o : i.dst) put(op_text(o));
  if (i.enc == Enc::Smem) {
    // s_load_b32 s0, s[0:1], 0x10; with an offset register as well,
    // s_load_b32 s0, s[0:1], s2 offset:0x10; with neither, null.
    const uint32_t off = static_cast<uint32_t>(i.offset) & 0x1FFFFF;
    const bool reg = i.src.size() > 1 && i.src[1].kind != OperandKind::Null;
    put(op_text(i.src[0]));
    if (reg) put(op_text(i.src[1]) + (off ? " offset:" + hex(off) : ""));
    else put(off ? hex(off) : "null");
    if (i.cache & 1) s += " glc";
    if (i.cache & 4) s += " dlc";
    return s;
  }
  const bool scratch_no_addr = i.enc == Enc::Flat && i.segment == Inst::Segment::Scratch && !i.has_vaddr;
  for (size_t k = 0; k < i.src.size(); ++k) {
    const Operand& o = i.src[k];
    if (o.hidden) continue;
    if (i.enc == Enc::Flat && k == 0 && scratch_no_addr && (o.kind == OperandKind::Vgpr)) {
      put("off");
      continue;
    }
    if (i.enc == Enc::Mubuf || i.enc == Enc::Mtbuf) {
      const bool is_vaddr = k == (i.dst.empty() ? 1u : 0u) && o.kind == OperandKind::Vgpr && o.width <= 2 &&
                            (&o == &i.src[i.dst.empty() ? 1 : 0]);
      if (is_vaddr && !i.offen && !i.idxen) {
        put("off");
        continue;
      }
    }
    put(op_text(o));
  }
  // A global or scratch access with no scalar base: "off" in its place.
  if (i.enc == Enc::Flat && i.segment != Inst::Segment::Flat) {
    if (!i.src.empty() && i.src.back().kind != OperandKind::Vgpr && !i.has_saddr) s.replace(s.rfind(op_text(i.src.back())), op_text(i.src.back()).size(), "off");
  }
  switch (i.enc) {
    case Enc::Smem:
      s += ", " + hex(static_cast<uint32_t>(i.offset) & 0x1FFFFF);
      if (i.cache & 1) s += " glc";
      if (i.cache & 4) s += " dlc";
      break;
    case Enc::Ds:
      if (name.find("2addr") != std::string::npos) {
        if (i.offset) s += " offset0:" + std::to_string(i.offset);
        if (i.offset1) s += " offset1:" + std::to_string(i.offset1);
      } else {
        const uint32_t off = static_cast<uint32_t>(i.offset) | static_cast<uint32_t>(i.offset1) << 8;
        if (name == "ds_swizzle_b32") s += " offset:" + swizzle_text(off);
        else if (off) s += " offset:" + std::to_string(off);
      }
      if (i.gds) s += " gds";
      break;
    case Enc::Flat:
      if (i.offset) s += " offset:" + std::to_string(i.offset);
      if (i.cache & 1) s += " glc";
      if (i.cache & 2) s += " slc";
      if (i.cache & 4) s += " dlc";
      break;
    case Enc::Mubuf:
    case Enc::Mtbuf:
      if (i.idxen) s += " idxen";
      if (i.offen) s += " offen";
      if (i.offset) s += " offset:" + std::to_string(i.offset);
      if (i.cache & 1) s += " glc";
      if (i.cache & 2) s += " slc";
      if (i.cache & 4) s += " dlc";
      break;
    case Enc::Vop3:
      if (i.op_sel) {
        const uint32_t n = static_cast<uint32_t>(i.src.size());
        std::string t = " op_sel:[";
        for (uint32_t k = 0; k < n; ++k) t += std::to_string((i.op_sel >> k) & 1) + ",";
        t += std::to_string((i.op_sel >> 3) & 1) + "]";
        s += t;
      }
      if (i.clamp) s += " clamp";
      if (i.omod == 1) s += " mul:2";
      if (i.omod == 2) s += " mul:4";
      if (i.omod == 3) s += " div:2";
      break;
    case Enc::Vop3p: {
      const uint32_t n = static_cast<uint32_t>(i.src.size());
      const auto list = [&](const char* name, uint32_t v, uint32_t dflt) {
        if (v == dflt) return;
        std::string t = std::string(" ") + name + ":[";
        for (uint32_t k = 0; k < n; ++k) t += std::to_string((v >> k) & 1) + (k + 1 < n ? "," : "]");
        s += t;
      };
      const uint32_t all = (1u << n) - 1;
      // A mixed-precision instruction's sources are floats unless OP_SEL_HI
      // says half, so it writes the list only where some are halves.
      const bool mix = name.rfind("v_fma_mix", 0) == 0;
      list("op_sel", i.op_sel & all, 0);
      list("op_sel_hi", i.op_sel_hi & all, mix ? 0 : all);
      list("neg_lo", i.neg_lo & all, 0);
      list("neg_hi", i.neg_hi & all, 0);
      if (i.clamp) s += " clamp";
      break;
    }
    default:
      break;
  }
  if (i.dpp) {
    char m[64];
    s += " " + dpp_text(i);
    std::snprintf(m, sizeof m, " row_mask:0x%x bank_mask:0x%x", i.row_mask, i.bank_mask);
    s += m;
    if (i.bound_ctrl) s += " bound_ctrl:1";
    if (i.fi) s += " fi:1";
  }
  if (i.dpp8) {
    std::string t = " dpp8:[";
    for (int k = 0; k < 8; ++k) t += std::to_string((i.dpp_ctrl >> (3 * k)) & 7) + (k < 7 ? "," : "]");
    s += t;
    if (i.fi) s += " fi:1";
  }
  return s;
}

}  // namespace

std::string to_text(const Inst& i) {
  if (i.enc == Enc::Vopd && i.dual.size() == 2) return one(i.dual[0]) + " :: " + one(i.dual[1]);
  return one(i);
}

}  // namespace vgpu::amd::gcn::rdna
