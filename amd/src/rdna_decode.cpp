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
#include <set>
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

enum class K { Vgpr, Src, Ssrc, Sreg, Sdst, Simm16, Simm32, Simm64, Label, Hwreg, Sendmsg, Waitcnt, Depctr, Delay, Vcc, Exec };
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
  // The ordinary instructions, then the image ones (rdna_images_rdna3.inc).
  static const std::vector<Row> rows = [] {
    std::vector<Row> v = {
#include "rdna_ops_rdna3.inc"
    };
    const std::vector<Row> images = {
#include "rdna_images_rdna3.inc"
    };
    v.insert(v.end(), images.begin(), images.end());
    return v;
  }();
  return rows;
}
const std::vector<Row>& rows_rdna2() {
  // The ordinary instructions, then the image ones (rdna_images_rdna2.inc).
  static const std::vector<Row> rows = [] {
    std::vector<Row> v = {
#include "rdna_ops_rdna2.inc"
    };
    const std::vector<Row> images = {
#include "rdna_images_rdna2.inc"
    };
    v.insert(v.end(), images.begin(), images.end());
    return v;
  }();
  return rows;
}
const std::vector<Row>& rows_cdna5() {
  // CDNA 5 (gfx1250): gfx12's encodings and what its specification adds, no graphics and no image sampling.
  static const std::vector<Row> rows = {
#include "rdna_ops_cdna5.inc"
  };
  return rows;
}
const std::vector<Row>& rows_rdna4() {
  // The ordinary instructions, then the image ones (rdna_images_rdna4.inc).
  static const std::vector<Row> rows = [] {
    std::vector<Row> v = {
#include "rdna_ops_rdna4.inc"
    };
    const std::vector<Row> images = {
#include "rdna_images_rdna4.inc"
    };
    v.insert(v.end(), images.begin(), images.end());
    return v;
  }();
  return rows;
}

// Which generation the instruction being decoded or printed is: RDNA2
// (gfx10.3), RDNA3 (gfx11) or RDNA4 (gfx12), whose numbering and memory
// encodings differ. Set for the length of a decode or a to_text.
thread_local bool g_rdna4 = false;   // gfx12's encodings: RDNA4, and CDNA 5 which is built on them
thread_local bool g_rdna2 = false;
thread_local bool g_cdna5 = false;   // gfx1250, whose table and additions are its own
thread_local uint8_t g_vop3px = 0;   // inside a scaled matrix instruction's second half: 2 (X2) or 3 (X3), its table segment
struct Generation {
  bool saved4, saved2, saved5;
  explicit Generation(Target t) : saved4(g_rdna4), saved2(g_rdna2), saved5(g_cdna5) {
    g_rdna4 = is_gfx12(t);
    g_rdna2 = t == Target::Gfx1030;
    g_cdna5 = t == Target::Gfx1250;
  }
  ~Generation() {
    g_rdna4 = saved4;
    g_rdna2 = saved2;
    g_cdna5 = saved5;
  }
};
const char* gen_name() { return g_cdna5 ? " (gfx1250)" : g_rdna4 ? " (gfx12)" : g_rdna2 ? " (gfx10)" : " (gfx11)"; }

struct Table {
  std::map<std::tuple<Enc, uint8_t, uint32_t>, const Row*> by_opcode;
  std::map<std::string, const Row*> short_forms;   // VOP1, VOP2 and VOPC names
  std::map<std::string, const Row*> long_forms;    // VOP3 names
};
Table make_table(const std::vector<Row>& rows) {
    Table t;
    for (const Row& r : rows) {
      t.by_opcode.emplace(std::make_tuple(r.enc, r.segment, static_cast<uint32_t>(r.opcode)), &r);
      if (r.enc == Enc::Vop1 || r.enc == Enc::Vop2 || r.enc == Enc::Vopc) t.short_forms.emplace(r.name, &r);
      if (r.enc == Enc::Vop3) t.long_forms.emplace(r.name, &r);
    }
    return t;
}
// gfx12 keeps gfx11's s_waitcnt (SOPP 9), which its specification leaves
// out; LLVM decodes it, and a library's code has it.
const std::vector<Row>& rows_rdna4_extra() {
  static const std::vector<Row> rows = {{Enc::Sopp, 0, 9, false, "s_waitcnt", {{"SIMM16", K::Waitcnt, 16, false}}}};
  return rows;
}
Table make_table4() {
  std::vector<Row> rows = rows_rdna4();
  // (make_table keeps pointers into the vector, so the rows must outlive it.)
  static std::vector<Row> all = [&] {
    std::vector<Row> v = rows_rdna4();
    for (const Row& r : rows_rdna4_extra()) v.push_back(r);
    return v;
  }();
  (void)rows;
  return make_table(all);
}
const Table& table() {
  static const Table t2 = make_table(rows_rdna2()), t3 = make_table(rows_rdna3()), t4 = make_table4(),
                     t5 = make_table(rows_cdna5());
  return g_cdna5 ? t5 : g_rdna4 ? t4 : g_rdna2 ? t2 : t3;
}
const Row& row(Enc e, uint8_t segment, uint32_t opcode) {
  const auto& m = table().by_opcode;
  const auto it = m.find({e, segment, opcode});
  if (it == m.end())
    throw Error::make(Err::Unsupported, enc_name(e), " opcode ", opcode, gen_name(),
                      " is not decoded yet");
  return *it->second;
}

// What the executor knows an instruction by: the gfx9 instruction that does
// the same, spelled as gfx9's decoder spells it -- its old name where the
// specification records one (global_load_b32 was global_load_dword) or it is
// one gfx11 renamed without saying (v_add_nc_u32 was v_add_u32), with the
// suffix gfx9 gives its short (_e32) and long (_e64) forms. An instruction gfx9
// does not have keeps its own name, and the executor knows it by that.
std::string exec_name(const std::string& name, Enc enc, bool dpp) {
  static const std::map<std::string, std::vector<std::string>> kAliases3 = {
#include "rdna_ops_rdna3_aliases.inc"
  };
  static const std::map<std::string, std::vector<std::string>> kAliases4 = {
#include "rdna_ops_rdna4_aliases.inc"
  };
  static const std::map<std::string, std::vector<std::string>> kAliases2 = {
#include "rdna_ops_rdna2_aliases.inc"
  };
  static const std::map<std::string, std::vector<std::string>> kAliases5 = {
#include "rdna_ops_cdna5_aliases.inc"
  };
  const auto& kAliases = g_cdna5 ? kAliases5 : g_rdna4 ? kAliases4 : g_rdna2 ? kAliases2 : kAliases3;
  // gfx10's own names for what gfx9 has only in 64 bits (s_andn2_saveexec_b32)
  // the executor knows by gfx11's (s_and_not1_saveexec_b32), whose earlier
  // names they are.
  if (g_rdna2 && !known_gfx9_name(name) && !known_gfx9_name(name + "_e32")) {
    static const std::map<std::string, std::string> kLater = [] {
      std::map<std::string, std::string> m;
      for (const auto& [later, earlier] : kAliases3)
        for (const std::string& e : earlier) m.emplace(e, later);
      return m;
    }();
    if (const auto it = kLater.find(name); it != kLater.end() && it->second != name) {
      const Generation gfx11(Target::Gfx1100);
      return exec_name(it->second, enc, dpp);
    }
  }
  static const std::map<std::string, std::string> kRenamed = {
      {"v_add_nc_u32", "v_add_u32"},         {"v_sub_nc_u32", "v_sub_u32"},
      {"v_subrev_nc_u32", "v_subrev_u32"},   {"v_add_co_ci_u32", "v_addc_co_u32"},
      {"v_sub_co_ci_u32", "v_subb_co_u32"},  {"v_subrev_co_ci_u32", "v_subbrev_co_u32"},
      {"v_add_nc_u16", "v_add_u16"},         {"v_sub_nc_u16", "v_sub_u16"},
      {"v_add_nc_i32", "v_add_i32"},         {"v_sub_nc_i32", "v_sub_i32"},
      {"v_add_nc_i16", "v_add_i16"},         {"v_sub_nc_i16", "v_sub_i16"},
      {"v_dot2acc_f32_f16", "v_dot2c_f32_f16"},
  };
  // gfx1250's three-operand "_num" minimum and maximum forms take a number over a NaN and order the zeros, as the
  // older instructions they replaced did not: they keep their own names, and the executor has them.
  static const std::set<std::string> kOwnNames = {
      "v_min3_num_f32", "v_max3_num_f32", "v_minmax_num_f32", "v_maxmin_num_f32", "v_med3_num_f32",
      "v_min3_num_f16", "v_max3_num_f16", "v_minmax_num_f16", "v_maxmin_num_f16", "v_med3_num_f16"};
  const bool own = g_cdna5 && kOwnNames.count(name);
  std::vector<std::string> candidates;
  if (const auto it = kRenamed.find(name); it != kRenamed.end()) candidates.push_back(it->second);
  if (const auto it = kAliases.find(name); it != kAliases.end() && !own)
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
// middle: 124 is null (reads zero, writes vanish) and 125 is M0. gfx10's
// has them the other way round, as gfx9 has M0.
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
  } else if (code == (g_rdna2 ? 125u : 124u)) {
    o.kind = OperandKind::Null;
  } else if (code == (g_rdna2 ? 124u : 125u)) {
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
  } else if (code == 254 && g_cdna5) {
    o.kind = OperandKind::Literal;
    o.lit64 = true;
  } else if (code == 255) {
    o.kind = OperandKind::Literal;
  } else if (code >= 256) {
    o.kind = OperandKind::Vgpr;
    o.index = code - 256;
  } else {
    throw Error::make(Err::Unsupported, "operand ", code, gen_name(), " is one this does not decode yet");
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
// register in wave32, whatever the specification's 64 says, and a pair in
// wave64.
uint32_t width_of(const Opnd& op, Enc enc, bool wave64) {
  const bool valu = enc == Enc::Vop1 || enc == Enc::Vop2 || enc == Enc::Vopc || enc == Enc::Vop3 ||
                    enc == Enc::Vop3p || enc == Enc::Vopd;
  const bool scalar = op.kind == K::Sreg || op.kind == K::Sdst || op.kind == K::Vcc || op.kind == K::Exec;
  if (valu && scalar && op.bits == 64) return wave64 ? 2 : 1;
  return op.bits <= 32 ? 1 : op.bits / 32;
}

bool is_half(const Opnd& op) { return op.bits == 16; }

// A DS instruction with two addresses, each its own 8-bit offset: gfx11's
// ds_load_2addr_b32, gfx10's ds_read2_b32 (and st64, and the exchanges).
bool two_addr(const std::string& name) {
  for (const char* k : {"2addr", "_read2", "_write2", "xchg2"})
    if (name.find(k) != std::string::npos) return true;
  return false;
}

}  // namespace

namespace {

// How many 32-bit address registers an image instruction takes: its
// coordinates by DIM, what its name adds (a mip level, a LOD, a bias, a
// depth to compare with, an offset, derivatives, a LOD clamp), and with a16
// the coordinates, LOD and clamp packed two to a register (the derivatives
// too, a dimension's pair to a register). Throws for the ray-tracing ones.
uint32_t image_address_words(const std::string& name, uint32_t dim, bool a16) {
  static const uint32_t kCoords[8] = {1, 2, 3, 3, 2, 3, 3, 4};   // 1D 2D 3D CUBE 1D_ARR 2D_ARR MSAA MSAA_ARR
  static const uint32_t kGradDims[8] = {1, 2, 3, 2, 1, 2, 2, 2};
  const uint32_t coords = kCoords[dim & 7];
  const auto has = [&](const char* part) {   // "_l" matches "_l" and "_l_o", not "_lz"
    const std::string p(part);
    for (size_t at = name.find(p); at != std::string::npos; at = name.find(p, at + 1)) {
      const size_t end = at + p.size();
      if (end == name.size() || name[end] == '_') return true;
    }
    return false;
  };
  const auto packed = [&](uint32_t n) { return a16 ? (n + 1) / 2 : n; };
  if (name.find("bvh") != std::string::npos)
    throw Error::make(Err::Unsupported, name, ": ray-tracing image instructions are not decoded yet");
  if (name == "image_get_resinfo") return 1;
  if (name.rfind("image_load", 0) == 0 || name.rfind("image_store", 0) == 0 || name.rfind("image_atomic", 0) == 0 ||
      name == "image_msaa_load")
    return packed(coords + (name.find("_mip") != std::string::npos ? 1 : 0));
  // image_sample*, image_gather4*, image_get_lod.
  uint32_t n = 0;
  if (has("_o")) n += 1;   // the offsets, one register
  if (has("_b")) n += 1;   // the bias
  if (has("_c")) n += 1;   // the depth to compare with
  if (has("_d")) {
    const uint32_t g = kGradDims[dim & 7];
    const bool g16 = a16 || has("_g16");
    n += g16 ? 2 * ((g + 1) / 2) : 2 * g;
  }
  return n + packed(coords + (has("_l") ? 1 : 0) + (has("_cl") ? 1 : 0));
}

// The image instructions: gfx10's and gfx11's MIMG, 64 bits and, with NSA,
// the extra address registers after it; RDNA4's VIMAGE and VSAMPLE, 96 bits.
// The name comes from the table; everything else from the fields.
Inst decode_image(const std::vector<uint8_t>& code, uint64_t at, Inst in) {
  const bool r4 = g_rdna4, r2 = g_rdna2;
  const uint32_t w0 = word(code, at), w1 = word(code, at + 4);
  in.enc = Enc::Mimg;
  in.size = 8;
  uint8_t segment = 0;
  uint32_t vdata = 0, rsrc = 0, samp = 0;
  std::vector<uint32_t> addr;   // the address fields, first to last
  uint32_t fields = 0;          // how many there are to name registers in
  if (r4) {
    const uint32_t w2 = word(code, at + 8);
    in.size = 12;
    segment = (w0 >> 26) == 0x39 ? 1 : 0;   // VSAMPLE : VIMAGE
    in.opcode = bits(w0, 21, 14);
    in.dim = static_cast<uint8_t>(bits(w0, 2, 0));
    in.r128 = bits(w0, 4, 4);
    in.d16 = bits(w0, 5, 5);
    in.a16 = bits(w0, 6, 6);
    in.dmask = static_cast<uint8_t>(bits(w0, 25, 22));
    vdata = bits(w1, 7, 0);
    rsrc = bits(w1, 17, 9);
    in.cache = bits(w1, 22, 20) | bits(w1, 19, 18) << 3;   // TH, SCOPE
    in.gfx12_cache = true;
    for (uint32_t k = 0; k < 4; ++k) addr.push_back((w2 >> (8 * k)) & 0xFF);
    if (segment == 1) {
      in.tfe = bits(w0, 3, 3);
      in.unorm = bits(w0, 13, 13);
      in.lwe = bits(w1, 8, 8);
      samp = bits(w1, 31, 23);
      fields = 4;
    } else {
      in.tfe = bits(w1, 23, 23);
      addr.push_back(bits(w1, 31, 24));
      fields = 5;
    }
    in.nsa = true;
  } else {
    const uint64_t w = w0 | static_cast<uint64_t>(w1) << 32;
    uint32_t nsa_words = 0;
    if (r2) {
      nsa_words = bits(w, 2, 1);
      in.dim = static_cast<uint8_t>(bits(w, 5, 3));
      in.cache = bits(w, 13, 13) | bits(w, 25, 25) << 1 | bits(w, 7, 7) << 2;   // glc, slc, dlc
      in.unorm = bits(w, 12, 12);
      in.tfe = bits(w, 16, 16);
      in.lwe = bits(w, 17, 17);
      in.opcode = bits(w, 24, 18) | bits(w, 0, 0) << 7;   // the opcode's top bit is bit 0
      in.a16 = bits(w, 62, 62);
      in.d16 = bits(w, 63, 63);
      samp = bits(w, 57, 53) << 2;
    } else {
      nsa_words = bits(w, 0, 0);
      in.dim = static_cast<uint8_t>(bits(w, 4, 2));
      in.unorm = bits(w, 7, 7);
      in.cache = bits(w, 14, 14) | bits(w, 12, 12) << 1 | bits(w, 13, 13) << 2;   // glc, slc, dlc
      in.a16 = bits(w, 16, 16);
      in.d16 = bits(w, 17, 17);
      in.opcode = bits(w, 25, 18);
      in.tfe = bits(w, 53, 53);
      in.lwe = bits(w, 54, 54);
      samp = bits(w, 62, 58) << 2;
    }
    in.dmask = static_cast<uint8_t>(bits(w, 11, 8));
    in.r128 = bits(w, 15, 15);
    vdata = bits(w, 47, 40);
    rsrc = bits(w, 52, 48) << 2;
    addr.push_back(bits(w, 39, 32));
    in.nsa = nsa_words != 0;
    for (uint32_t k = 0; k < nsa_words; ++k) {
      const uint32_t extra = word(code, at + 8 + 4 * k);
      for (uint32_t b = 0; b < 4; ++b) addr.push_back((extra >> (8 * b)) & 0xFF);
    }
    in.size = 8 + 4 * nsa_words;
    fields = in.nsa ? static_cast<uint32_t>(addr.size()) : 1;
  }
  const Row& r = row(Enc::Mimg, segment, in.opcode);
  in.name = r.name;
  const std::string& name = in.name;
  const bool sample = name.find("sample") != std::string::npos || name.find("gather4") != std::string::npos ||
                      name == "image_get_lod";
  const bool store = name.rfind("image_store", 0) == 0;
  const bool atomic = name.rfind("image_atomic", 0) == 0;
  const bool returns = !store && (!atomic || (r4 ? (in.cache & 1) : (in.cache & 1)));
  // The data: a register per channel DMASK names (a gather4 always four),
  // two 16-bit channels to one with d16, and a word more for tfe or lwe.
  uint32_t channels = name.find("gather4") != std::string::npos ? 4 : std::max(1, __builtin_popcount(in.dmask));
  if (in.d16) channels = (channels + 1) / 2;
  if (in.tfe || in.lwe) channels += 1;
  in.src.push_back(operand(256 + vdata, channels));
  if (returns) in.dst.push_back(operand(256 + vdata, channels));
  // The addresses: a range from the one field, or a register a field (the
  // last taking what the others leave, as a range).
  const uint32_t words = image_address_words(name, in.dim, in.a16);
  if (!in.nsa) {
    in.src.push_back(operand(256 + addr[0], words));
  } else {
    const uint32_t named = std::min(words, fields);
    for (uint32_t k = 0; k < named; ++k) {
      const uint32_t width = k + 1 == named ? words - (named - 1) : 1;
      in.src.push_back(operand(256 + addr[k], width));
    }
  }
  in.src.push_back(operand(rsrc, in.r128 ? 4 : 8));
  if (sample) in.src.push_back(operand(samp, 4));
  return in;
}

}  // namespace

Inst decode(const std::vector<uint8_t>& code, uint64_t at, uint64_t pc, Target target, bool wave64) {
  const Generation generation(target);
  const bool r4 = g_rdna4, r2 = g_rdna2;
  const uint32_t null_code = r2 ? 125 : 124;
  const uint32_t w0 = word(code, at);
  // gfx1250's scaled matrix instructions are two VOP3P words: a prefix that names the scale sources (its opcode
  // says whether a scale is 32 bits, X2, or 64, X3), then the matrix instruction itself. The specification's
  // opcodes for the prefix and the ones LLVM 22 writes differ: the specification's identifiers (0xCC37, 0xCCBD for X2;
  // 0xCC3B, 0xCCBA for X3) are the opcodes of ordinary instructions in the same specification (0x37 is
  // v_pk_maximum3_f16), so only LLVM's (0x35 for X2, 0x3A for X3) are taken.
  if (g_cdna5 && !g_vop3px && (w0 >> 24) == 0xCC) {
    const uint32_t pre = bits(w0, 23, 16);
    const bool x2 = pre == 0x35, x3 = pre == 0x3A;
    if (x2 || x3) {
      const uint64_t prefix = w0 | static_cast<uint64_t>(word(code, at + 4)) << 32;
      g_vop3px = x2 ? 2 : 3;
      struct Reset { ~Reset() { g_vop3px = 0; } } reset;
      Inst main = rdna::decode(code, at + 8, pc + 8, target, wave64);
      // The scale sources: the prefix's SRC0 and SRC1 fields (the matrix word's own are the matrices').
      main.src.push_back(operand(bits(prefix, 40, 32), x2 ? 1 : 2));
      main.src.push_back(operand(bits(prefix, 49, 41), x2 ? 1 : 2));
      main.src[main.src.size() - 2].scale_src = main.src.back().scale_src = true;
      // The prefix's SCL_NEG (A's scale format, bits 62:61), SCL_NEG_HI (B's, 9:8) and the lane-half selects
      // (SCL_OPSEL bit 11 for A, SCL_OPSEL_HI bit 59 for B).
      main.scale_fmt_a = static_cast<uint8_t>(bits(prefix, 62, 61));
      main.scale_fmt_b = static_cast<uint8_t>(bits(prefix, 9, 8));
      main.scale_hi_a = bits(prefix, 11, 11);
      main.scale_hi_b = bits(prefix, 59, 59);
      main.pc = pc;
      main.size = 16;
      return main;
    }
  }
  uint32_t w2 = 0;   // RDNA4's 96-bit memory encodings' third word
  bool vopd3 = false;  // gfx1250's 96-bit dual-issue encoding
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
  // An image instruction: MIMG on gfx10 and gfx11; VIMAGE and VSAMPLE on
  // RDNA4.
  // gfx1250's tensor data mover: VIMAGE with an opcode of its own, whose operands are groups of scalar registers.
  if (g_cdna5 && (w0 >> 26) == 0x34 && (bits(w0, 21, 14) == 196 || bits(w0, 21, 14) == 197)) {
    const uint32_t w1 = word(code, at + 4), g = word(code, at + 8);
    in.enc = Enc::Mimg;
    in.size = 12;
    in.opcode = bits(w0, 21, 14);
    const Row& tr = row(Enc::Mimg, 3, in.opcode);
    in.name = tr.name;
    in.asm_name = in.name;
    in.cache = bits(w1, 22, 20) | bits(w1, 19, 18) << 3;   // TH, SCOPE
    in.gfx12_cache = true;
    in.nv = bits(w0, 7, 7);
    const auto group = [&](uint32_t reg, uint32_t width) {
      Operand o;
      o.kind = reg == 124 ? OperandKind::Null : OperandKind::Sgpr;
      o.index = reg;
      o.width = width;
      return o;
    };
    in.src = {group(g & 0xFF, 4), group(g >> 8 & 0xFF, 8)};
    // The third and fourth groups are named only when the descriptor is longer than the first two give.
    if ((g >> 16 & 0xFF) != 124 || (g >> 24 & 0xFF) != 124) {
      in.src.push_back(group(g >> 16 & 0xFF, 4));
      in.src.push_back(group(g >> 24 & 0xFF, 4));
    }
    return in;
  }
  if ((!r4 && (w0 >> 26) == 0x3c) || (r4 && ((w0 >> 26) == 0x34 || (w0 >> 26) == 0x39)))
    return decode_image(code, at, in);
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
    in.opcode = r4 ? bits(w0, 18, 13) : bits(w0, 25, 18);
    second();
  } else if (r4 && ((w0 >> 26) == 0x31 || (w0 >> 24) == 0xec || (w0 >> 24) == 0xed || (w0 >> 24) == 0xee)) {
    // RDNA4's buffer (VBUFFER) and flat, scratch and global (VFLAT, VSCRATCH,
    // VGLOBAL) accesses: three words.
    in.enc = (w0 >> 26) == 0x31 ? Enc::Mubuf : Enc::Flat;
    in.opcode = bits(w0, 21, 14);
    second();
    w2 = word(code, at + 8);
    in.size = 12;
  } else if ((w0 >> 25) == 0x3f) {
    in.enc = Enc::Vop1;
    in.opcode = r4 ? bits(w0, 15, 9) : bits(w0, 16, 9);
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
    in.opcode = g_cdna5 ? bits(w0, 23, 16) : bits(w0, 22, 16);   // gfx1250's has eight opcode bits
    second();
  } else if ((w0 >> 26) == 0x32) {
    in.enc = Enc::Vopd;
    second();
  } else if (g_cdna5 && (w0 >> 24) == 0xCF) {
    // gfx1250's VOPD3: three words, a six-bit opcode each, three sources and a negation for each half.
    in.enc = Enc::Vopd;
    second();
    w2 = word(code, at + 8);
    in.size = 12;
    vopd3 = true;
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
    if (r2) in.opcode = bits(w, 18, 16) | bits(w, 53, 53) << 3;
  } else {
    throw Error::make(Err::Unsupported, "instruction word ", w0, gen_name(), " is in an encoding this does not decode yet");
  }

  // VOPD: two instructions in one, X and Y, each its own opcode.
  if (in.enc == Enc::Vopd) {
    const uint32_t opx = vopd3 ? bits(w, 23, 18) : bits(w, 25, 22), opy = vopd3 ? bits(w, 17, 12) : bits(w, 21, 17);
    const uint8_t seg = vopd3 ? 1 : 0;
    const Row& rx = row(Enc::Vopd, seg, opx);
    const Row& ry = row(Enc::Vopd, seg, opy);
    const auto half = [&](const Row& r, uint32_t vdst, uint32_t src0, uint32_t vsrc1, bool y, uint32_t vsrc2 = 0,
                          uint32_t neg = 0) {
      Inst h;
      h.enc = Enc::Vopd;
      h.arch = target;
      h.opcode = y ? opy : opx;
      h.name = r.name;
      h.pc = pc;
      const bool bitop2 = r.name == std::string("v_dual_bitop2_b32");
      for (const Opnd& op : r.ops) {
        const std::string f = op.field;
        const uint32_t regs = std::max(1u, op.bits / 32u);
        if (vopd3 && op.kind == K::Sreg) {
          // v_dual_cndmask_b32's mask names its register (VCC, or any pair), where VOPD's is VCC unwritten.
          h.src.push_back(take(vsrc2, wave64 ? 2 : 1));
          continue;
        }
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
        if (op.out) h.dst.push_back(vgpr(vdst, regs));
        else if (f.find("SRC") != std::string::npos && f.find("VSRC") == std::string::npos)
          h.src.push_back(take(src0, regs));
        else if (!f.empty() && f.back() == '2') {
          h.src.push_back(vgpr(vsrc2, regs));
        } else h.src.push_back(vgpr(vsrc1, regs));
      }
      if (vopd3 && bitop2) h.bitop3 = static_cast<uint8_t>(vsrc2);   // the truth table, in the field a third source would use
      if (vopd3) {
        for (uint32_t k = 0; k < h.src.size() && k < 3; ++k)
          if ((neg >> k) & 1) h.src[k].neg = true;
        if (bitop2) {   // v_bitop3 reads three sources; the dual form has two
          Operand z;
          z.kind = OperandKind::Inline;
          z.value = 0;
          z.hidden = true;
          h.src.push_back(z);
        }
      }
      // v_dual_cndmask_b32 picks by VCC's low half, which the assembler
      // leaves unwritten but the instruction reads.
      if (h.name == "v_dual_cndmask_b32" && !vopd3) {
        Operand v;
        v.kind = OperandKind::Vcc;
        v.hidden = true;
        h.src.push_back(v);
      }
      h.asm_name = h.name;
      h.name = exec_name(h.name.substr(std::string("v_dual_").size()).insert(0, "v_"), Enc::Vopd, false);
      if (h.name == "v_bitop2_b32") h.name = "v_bitop3_b32";   // the same operation with two sources
      if (const int op9 = gfx9_opcode(h.name); op9 >= 0) h.opcode = static_cast<uint32_t>(op9);
      return h;
    };
    in.name = rx.name;
    Inst x, y;
    if (vopd3) {
      // Words: 0 SRCX0 | OPY << 12 | OPX << 18; 1 SRCY0, NEGX (11:9), NEGY (14:12), VSRCX1 (23:16), VSRCX2 (31:24);
      // 2 VDSTX, VSRCY1 (15:8), VSRCY2 (23:16), VDSTY (31:24).
      const uint32_t w1 = static_cast<uint32_t>(w >> 32);
      x = half(rx, w2 & 0xFF, w0 & 0x1FF, bits(w1, 23, 16), false, bits(w1, 31, 24), bits(w1, 11, 9));
      y = half(ry, bits(w2, 31, 24), w1 & 0x1FF, bits(w2, 15, 8), true, bits(w2, 23, 16), bits(w1, 14, 12));
    } else {
      const uint32_t vdstx = bits(w, 63, 56), vdsty = (bits(w, 55, 49) << 1) | ((vdstx & 1) ^ 1);
      x = half(rx, vdstx, bits(w, 8, 0), bits(w, 16, 9), false);
      y = half(ry, vdsty, bits(w, 40, 32), bits(w, 48, 41), true);
    }
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
  // (RDNA4 says which by the encoding's top byte: 0xEC flat, 0xED scratch,
  // 0xEE global.)
  const uint32_t seg = in.enc != Enc::Flat ? 0
                     : r4 ? ((w0 >> 24) == 0xed ? 1 : (w0 >> 24) == 0xee ? 2 : 0)
                          : r2 ? bits(w, 15, 14) : bits(w, 17, 16);
  const uint8_t segment = in.enc == Enc::Vop3p && g_vop3px ? static_cast<uint8_t>(g_vop3px - 1)
                                                          : static_cast<uint8_t>(seg == 1 ? 2 : seg == 2 ? 1 : 0);
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
  // gfx10's SDWA: a VOP1/VOP2/VOPC instruction whose first source is 0xF9
  // reads (and writes) part of a register, a second word saying which.
  uint32_t sdwa_word = 0;
  if (r2 && short_form && src0_field == 0xF9) {
    sdwa_word = word(code, at + in.size);
    in.size += 4;
    in.sdwa = true;
  }
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
    if (r4) switch (in.enc) {
        case Enc::Smem:
          return f == "SDATA" ? bits(w, 12, 6) : f == "SBASE" ? bits(w, 5, 0) << 1 : bits(w, 63, 57);
        case Enc::Mubuf:
          return f == "VDATA" ? bits(w, 39, 32) : f == "VADDR" ? (w2 & 0xFF) : f == "RSRC" ? bits(w, 49, 41)
                                                                                         : bits(w, 6, 0);
        case Enc::Flat:
          return f == "VDST" ? bits(w, 39, 32) : f == "VADDR" ? (w2 & 0xFF) : f == "VSRC" ? bits(w, 62, 55)
                                                                                           : bits(w, 6, 0);
        default:
          break;
      }
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
    if (f.rfind("SCALE_", 0) == 0) continue;   // a scaled matrix instruction's, taken from its prefix above
    bool buffer_atomic_data = false;
    uint32_t width = width_of(op, in.enc, wave64);
    // A buffer address is an offset or an index, one register, or both, two.
    const bool idxen = r4 ? bits(w, 63, 63) : r2 ? bits(w, 13, 13) : bits(w, 55, 55);
    const bool offen = r4 ? bits(w, 62, 62) : r2 ? bits(w, 12, 12) : bits(w, 54, 54);
    if ((in.enc == Enc::Mubuf || in.enc == Enc::Mtbuf) && f == "VADDR") width = idxen && offen ? 2 : 1;
    // A global access with a scalar base takes a 32-bit offset from its
    // address register; without one, a whole 64-bit address. Scratch's is
    // always an offset.
    const uint32_t saddr_field = r4 ? bits(w, 6, 0) : bits(w, 54, 48);
    if (in.enc == Enc::Flat && (f == "ADDR" || f == "VADDR") && g_cdna5)
      width = segment != 2 && saddr_field == null_code ? 2 : 1;   // flat and global: 64 bits unless a base is named
    else if (in.enc == Enc::Flat && (f == "ADDR" || f == "VADDR") && segment != 0)
      width = segment == 1 && saddr_field == null_code ? 2 : 1;
    // An atomic returns the old value only with GLC set (RDNA4: TH's
    // return bit).
    const bool returns = r4 ? bits(w, 52, 52) : r2 && in.enc == Enc::Flat ? bits(w, 16, 16) : bits(w, 14, 14);
    if (op.out && (in.enc == Enc::Flat || in.enc == Enc::Mubuf) && r.name[0] != 'd' &&
        std::string(r.name).find("_atomic_") != std::string::npos && !returns) {
      // A flat atomic's data is a separate source (VSRC); a buffer atomic's is the same field, an input here.
      if (in.enc != Enc::Mubuf) continue;
      buffer_atomic_data = true;
    }
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
        // SOPK's constant is signed (s_movk_i32 s4, 0xffec is -20), as gfx9's
        // decoder keeps it; the unsigned forms take its low 16 bits.
        if (in.enc == Enc::Sopk) in.simm = static_cast<int16_t>(in.simm);
        if (op.kind == K::Label) in.target = pc + 4 + 4 * static_cast<int64_t>(static_cast<int16_t>(in.simm));
        continue;
      case K::Simm32: {
        Operand l;
        l.kind = OperandKind::Literal;
        l.constant_k = true;
        in.src.push_back(l);
        literal = true;
        continue;
      }
      case K::Vgpr: {
        uint32_t v = field(f);
        // A 9-bit source field names a vector register from 256.
        if (f.rfind("SRC", 0) == 0 && (in.enc == Enc::Vop3 || in.enc == Enc::Vop3p)) v -= 256;
        if (f == "SRC0" && short_form) v -= 256;
        if (f == "SRC0" && in.sdwa) v = bits(sdwa_word, 7, 0);
        o = vgpr(v, width);
        if (in.sdwa && f == "SRC0" && bits(sdwa_word, 23, 23)) o = take(v, width);
        if (in.sdwa && f == "VSRC1" && bits(sdwa_word, 31, 31)) o = take(v, width);
        // A 16-bit operand of a short-form instruction is half a register,
        // the field's top bit saying which.
        if (is_half(op) && short_form && !r2) {
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
        // SDWA's, the second word's, a scalar register where its S0 bit says.
        if (f == "SRC0" && in.sdwa) v = bits(sdwa_word, 7, 0) + (bits(sdwa_word, 23, 23) ? 0 : 256);
        o = take(v, width);
        if (is_half(op) && short_form && !r2 && o.kind == OperandKind::Vgpr) {
          o.index &= 0x7F;
          o.hi = (v - 256) & 0x80;
          o.half = true;
        }
        if (is_half(op)) o.bits16 = true;
        break;
      }
    }
    (void)vop3_16;
    ((op.out && !buffer_atomic_data) ? in.dst : in.src).push_back(o);
  }

  // The modifiers each encoding carries -- RDNA4's memory ones first: a
  // 24-bit offset, and the temporal hint (TH) and scope in place of
  // glc/slc/dlc, kept in `cache` as TH | SCOPE << 3.
  if (r4 && (in.enc == Enc::Smem || in.enc == Enc::Mubuf || in.enc == Enc::Flat)) {
    in.gfx12_cache = true;
    if (in.enc == Enc::Smem) {
      in.offset = static_cast<int32_t>(bits(w, 55, 32) << 8) >> 8;
      in.cache = bits(w, 24, 23) | bits(w, 22, 21) << 3;
      in.nv = bits(w, 20, 20);
      in.scale_offset = g_cdna5 && bits(w, 56, 56);
    } else {
      in.offset = static_cast<int32_t>((w2 >> 8) << 8) >> 8;   // IOFFSET, 24 bits, signed
      in.cache = bits(w, 54, 52) | bits(w, 51, 50) << 3;
      in.nv = bits(w, 7, 7);
      if (in.enc == Enc::Mubuf) {
        in.offen = bits(w, 62, 62);
        in.idxen = bits(w, 63, 63);
        in.format = bits(w, 61, 55);
      } else {
        in.scale_offset = g_cdna5 && bits(w, 48, 48);
        const uint32_t saddr = bits(w, 6, 0);
        // gfx1250's flat accesses take a scalar base too (and then a 32-bit offset in the vector register).
        in.has_saddr = (g_cdna5 || in.segment != Inst::Segment::Flat) && saddr != 124;
        in.saddr = saddr;
        in.has_vaddr = in.segment != Inst::Segment::Scratch || bits(w, 49, 49);
      }
    }
  } else
  switch (in.enc) {
    case Enc::Smem:
      in.offset = static_cast<int32_t>(bits(w, 52, 32) << 11) >> 11;   // 21 bits, signed
      in.cache = r2 ? bits(w, 16, 16) | bits(w, 14, 14) << 2 : bits(w, 14, 14) | bits(w, 13, 13) << 2;   // glc, dlc
      break;
    case Enc::Vop1:
    case Enc::Vop2:
    case Enc::Vopc:
      if (in.sdwa) {
        // Each source's part, sign and modifiers; the destination's part
        // and what becomes of the rest, or a comparison's scalar result.
        for (uint32_t k = 0; k < in.src.size() && k < 2; ++k) {
          const uint32_t b = k == 0 ? bits(sdwa_word, 23, 16) : bits(sdwa_word, 31, 24);
          if (in.src[k].kind == OperandKind::Vcc) continue;
          in.src[k].sel = static_cast<uint8_t>(b & 7);
          in.src[k].sext = (b >> 3) & 1;
          in.src[k].neg = (b >> 4) & 1;
          in.src[k].abs = (b >> 5) & 1;
        }
        if (in.enc == Enc::Vopc) {
          if (bits(sdwa_word, 15, 15) && !in.dst.empty()) in.dst[0] = take(bits(sdwa_word, 14, 8), in.dst[0].width);
        } else {
          in.dst_sel = static_cast<uint8_t>(bits(sdwa_word, 10, 8));
          in.dst_unused = static_cast<uint8_t>(bits(sdwa_word, 12, 11));
          in.clamp = bits(sdwa_word, 13, 13);
          in.omod = static_cast<uint8_t>(bits(sdwa_word, 15, 14));
        }
      }
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
      // v_bitop3's truth table is in the modifier bits, as on gfx950: NEG, ABS and OMOD, in that order from the
      // low end.
      if (g_cdna5 && in.name.rfind("v_bitop3_", 0) == 0) {
        in.bitop3 = static_cast<uint8_t>(neg | abs << 3 | in.omod << 6);
        in.omod = 0;
        for (Operand& o : in.src) o.neg = o.abs = false;
      }
      // A 16-bit vector register operand is half a register: OP_SEL's bit for
      // it (bit 3 the destination's) says which half, and the assembler
      // writes that as .l or .h rather than as op_sel.
      if (!r.sdst && !r2) {
        // (gfx12's disassembler writes op_sel out rather than .l/.h, so it is
        // kept to print there; the executor reads the halves either way.)
        in.printed_op_sel = in.op_sel;
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
        if (!r4) in.printed_op_sel = in.op_sel;
      } else if (!r.sdst) {
        // gfx10 has no true16: op_sel is gfx9's, which the executor reads.
        in.printed_op_sel = in.op_sel;
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
      if (two_addr(r.name)) {
        in.offset = static_cast<int32_t>(bits(w, 7, 0));
        in.offset1 = static_cast<int32_t>(bits(w, 15, 8));
      } else {
        in.offset = static_cast<int32_t>(bits(w, 15, 0));
      }
      in.gds = bits(w, 17, 17);
      // gfx11 renamed ds_cmpst to ds_cmpstore and swapped its data: DATA0 is
      // the value stored and DATA1 the one compared, as a buffer or flat
      // compare-and-swap has them. gfx9's (which the executor runs) compares
      // with DATA0.
      if (std::string_view(r.name).rfind("ds_cmpstore", 0) == 0 && in.src.size() == 3) std::swap(in.src[1], in.src[2]);
      break;
    case Enc::Flat: {
      in.offset = static_cast<int32_t>(bits(w, 12, 0) << 19) >> 19;   // 13 bits, signed
      if (r2) in.offset = static_cast<int32_t>(bits(w, 11, 0) << 20) >> 20;   // 12
      if (in.segment == Inst::Segment::Flat) in.offset = static_cast<int32_t>(bits(w, 11, 0));
      in.cache = r2 ? bits(w, 16, 16) | bits(w, 17, 17) << 1 | bits(w, 12, 12) << 2
                    : bits(w, 14, 14) | bits(w, 15, 15) << 1 | bits(w, 13, 13) << 2;   // glc, slc, dlc
      const uint32_t saddr = bits(w, 54, 48);
      in.has_saddr = in.segment != Inst::Segment::Flat && saddr != null_code;
      in.saddr = saddr;
      // Scratch: SVE says whether the address register is there at all
      // (gfx10: it is where there is no scalar one).
      in.has_vaddr = in.segment != Inst::Segment::Scratch || (r2 ? !in.has_saddr : bits(w, 55, 55));
      break;
    }
    case Enc::Mubuf:
    case Enc::Mtbuf:
      in.offset = static_cast<int32_t>(bits(w, 11, 0));
      if (r2) {
        in.offen = bits(w, 12, 12);
        in.idxen = bits(w, 13, 13);
        in.cache = bits(w, 14, 14) | bits(w, 54, 54) << 1 | bits(w, 15, 15) << 2;   // glc, slc, dlc
      } else {
        in.offen = bits(w, 54, 54);
        in.idxen = bits(w, 55, 55);
        in.cache = bits(w, 14, 14) | bits(w, 12, 12) << 1 | bits(w, 13, 13) << 2;   // glc, slc, dlc
      }
      if (in.enc == Enc::Mtbuf) in.format = bits(w, 25, 19);
      break;
    default:
      break;
  }

  if (literal) {
    const uint32_t value = word(code, at + in.size);
    // A 16-bit operand's literal is the low half of the word; a double's is
    // its high half.
    const bool f64 = std::string(r.name).find("f64") != std::string::npos;
    bool wide = false;
    for (const Operand& o : in.src) wide |= o.kind == OperandKind::Literal && o.lit64;
    const uint64_t value64 = wide ? (value | static_cast<uint64_t>(word(code, at + in.size + 4)) << 32) : value;
    for (Operand& o : in.src)
      if (o.kind == OperandKind::Literal) {
        if (o.lit64) {
          o.value = static_cast<int64_t>(value64);
          continue;
        }
        o.value = o.bits16 ? (value & 0xFFFF) : value;
        o.literal_high = f64 && o.width == 2;
      }
    in.size += wide ? 8 : 4;
  }
  // A half-register source reads its half as a sub-dword instruction reads
  // a word of a register, which is how the executor runs it (the
  // destination's half is narrowed as it runs).
  for (Operand& o : in.src)
    if (o.half && o.kind == OperandKind::Vgpr) o.sel = o.hi ? 5 : 4;
  in.asm_name = in.name;
  in.name = exec_name(in.name, in.enc, in.dpp || in.dpp8);
  // gfx9 runs a sub-dword instruction by its _sdwa name.
  if (in.sdwa && in.name.size() > 4 && in.name.compare(in.name.size() - 4, 4, "_e32") == 0)
    in.name.replace(in.name.size() - 4, 4, "_sdwa");
  // Promoted only where the executor has the short form too (gfx12's
  // v_add_f64 has one; gfx9's, which it runs as, does not).
  if (in.promoted && (in.name.size() < 4 || in.name.compare(in.name.size() - 4, 4, "_e64") != 0)) in.promoted = false;
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
    // gfx11's disassembler names a 16-bit operand's half (v1.l, v1.h); gfx12's
    // names the register, the half being in the register number's top bit.
    case OperandKind::Vgpr:
      if (o.half && g_rdna4) {
        char n[16];
        std::snprintf(n, sizeof n, "v%u", o.index + (o.hi ? 128 : 0));
        return n;
      }
      return range("v") + (o.half ? (o.hi ? ".h" : ".l") : "");
    case OperandKind::Ttmp: return range("ttmp");
    case OperandKind::Null: return "null";
    case OperandKind::Scc: return "src_scc";
    case OperandKind::SharedLimit: return "src_shared_limit";
    case OperandKind::PrivateLimit: return "src_private_limit";
    default: return operand_text(o);
  }
}
std::string hex(uint32_t v) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%x", v);
  return b;
}

// An operand with its modifiers: -v1, |s2|, -|v[3:4]|.
std::string op_text(const Operand& o) {
  if (o.constant_k) return hex(static_cast<uint32_t>(o.value));
  if (o.kind == OperandKind::Literal && o.lit64) {
    char b[48];
    std::snprintf(b, sizeof b, "lit64(0x%llx)", static_cast<unsigned long long>(o.value));
    return o.neg ? std::string("-") + b : std::string(b);
  }
  if (o.kind == OperandKind::Inline || o.kind == OperandKind::InlineFloat || o.kind == OperandKind::Literal) {
    Operand plain = o;
    plain.abs = false;
    std::string t = operand_text(plain);
    // 1/(2*pi) in a 64-bit operand is the double's digits, not the float's.
    if (o.kind == OperandKind::InlineFloat && o.width == 2 && t == "0.15915494") t = "0.15915494309189532";
    return o.abs ? "|" + t + "|" : t;
  }
  Operand plain = o;
  plain.neg = plain.abs = false;
  const std::string neg = o.neg ? "-" : "";
  const std::string r = o.sext ? "sext(" + reg_text(plain) + ")" : reg_text(plain);
  return o.abs ? neg + "|" + r + "|" : neg + r;
}


std::string hwreg(uint32_t simm) {
  static const std::map<uint32_t, const char*> kNames = {
      {1, "HW_REG_MODE"},        {2, "HW_REG_STATUS"},      {3, "HW_REG_TRAPSTS"},      {5, "HW_REG_GPR_ALLOC"},
      {6, "HW_REG_LDS_ALLOC"},   {7, "HW_REG_IB_STS"},      {15, "HW_REG_SH_MEM_BASES"}, {20, "HW_REG_FLAT_SCR_LO"},
      {21, "HW_REG_FLAT_SCR_HI"}, {23, "HW_REG_HW_ID1"},    {24, "HW_REG_HW_ID2"},       {25, "HW_REG_POPS_PACKER"},
      {29, "HW_REG_SHADER_CYCLES"}, {30, "HW_REG_SHADER_CYCLES_HI"}};
  const uint32_t id = simm & 0x3F, offset = (simm >> 6) & 0x1F, size = ((simm >> 11) & 0x1F) + 1;
  const auto it = kNames.find(id);
  // gfx12 names the cycle counter's two halves.
  const std::string known = g_rdna4 && id == 29 ? "HW_REG_SHADER_CYCLES_LO" : it != kNames.end() ? it->second : "";
  std::string out = "hwreg(" + (!known.empty() ? known : std::to_string(id));
  if (offset || size != 32) out += ", " + std::to_string(offset) + ", " + std::to_string(size);
  return out + ")";
}

std::string waitcnt(uint32_t simm) {
  // gfx11: expcnt bits 2:0, lgkmcnt 9:4, vmcnt 15:10. gfx10: vmcnt 3:0 and
  // 15:14, expcnt 6:4, lgkmcnt 13:8.
  uint32_t exp = simm & 7, lgkm = (simm >> 4) & 0x3F, vm = (simm >> 10) & 0x3F;
  if (g_rdna2) {
    vm = (simm & 0xF) | ((simm >> 14) & 3) << 4;
    exp = (simm >> 4) & 7;
    lgkm = (simm >> 8) & 0x3F;
  }
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

// RDNA4's cache policy as the assembler writes it: the temporal hint by what
// the access is (TH_LOAD_NT, TH_STORE_HT, TH_ATOMIC_RETURN, ...), then the
// scope; the defaults (regular, the compute unit) not at all.
std::string gfx12_cache(const std::string& name, uint32_t cache) {
  const uint32_t th = cache & 7, scope = (cache >> 3) & 3;
  std::string s;
  const bool atomic = name.find("_atomic") != std::string::npos;
  const bool store = !atomic && (name.find("_store") != std::string::npos);
  if (th && g_cdna5) {
    // gfx1250's names (LLVM's printer for it): a store's write-back is plain WB, an atomic's cascade shows only at
    // device scope or wider and its other TH values print as numbers, and a load's reserved value is a number too.
    char num[24];
    if (atomic) {
      const bool ret = th & 1, nt = th & 2;
      if (th & 4) {
        if (scope >= 2) s += std::string(" th:TH_ATOMIC_CASCADE_") + (nt ? "NT" : "RT");
        else { std::snprintf(num, sizeof num, " th:TH_ATOMIC_0x%x", th); s += num; }
      } else {
        s += std::string(" th:TH_ATOMIC") + (nt ? "_NT" : "") + (ret ? "_RETURN" : "");
      }
    } else if (th == 7 && !store) {
      s += " th:0x7";
    } else {
      static const char* kLoad[] = {"", "NT", "HT", "LU", "NT_RT", "RT_NT", "NT_HT", ""};
      static const char* kStore[] = {"", "NT", "HT", "WB", "NT_RT", "RT_NT", "NT_HT", "NT_WB"};
      const std::string what = th == 3 && scope == 3 ? "BYPASS" : store ? kStore[th] : kLoad[th];
      s += std::string(" th:TH_") + (store ? "STORE_" : "LOAD_") + what;
    }
  } else if (th) {
    if (atomic) {
      std::string t;
      if (th & 2) t += "_NT";
      if (th & 4) t += "_CASCADE";
      if (th & 1) t += "_RETURN";
      s += " th:TH_ATOMIC" + t;
    } else {
      static const char* kLoad[] = {"RT", "NT", "HT", "LU", "NT_RT", "RT_NT", "NT_HT", "RESERVED"};
      static const char* kStore[] = {"RT", "NT", "HT", "RT_WB", "NT_RT", "RT_NT", "NT_HT", "NT_WB"};
      const std::string what = th == 3 && scope == 3 ? "BYPASS" : store ? kStore[th] : kLoad[th];
      s += std::string(" th:TH_") + (store ? "STORE_" : "LOAD_") + what;
    }
  }
  static const char* kScope[] = {"", " scope:SCOPE_SE", " scope:SCOPE_DEV", " scope:SCOPE_SYS"};
  return s + kScope[scope];
}

// An image instruction as llvm-objdump writes it: the data, the addresses
// (a range, or a list where the encoding names them one by one; RDNA4 lists
// more than one always), the resource, the sampler, then DMASK, DIM and the
// modifiers.
std::string image_text(const Inst& i) {
  static const char* kDim[8] = {"1D", "2D", "3D", "CUBE", "1D_ARRAY", "2D_ARRAY", "2D_MSAA", "2D_MSAA_ARRAY"};
  const bool sample = i.name.find("sample") != std::string::npos || i.name.find("gather4") != std::string::npos ||
                      i.name == "image_get_lod";
  const size_t addrs = i.src.size() - (sample ? 3 : 2);
  std::string s = i.name + " " + reg_text(i.src[0]) + ", ";
  if (addrs == 1 && (!i.nsa || g_rdna4)) {
    s += reg_text(i.src[1]);
  } else {
    s += "[";
    for (size_t k = 1; k <= addrs; ++k) s += (k > 1 ? ", " : "") + reg_text(i.src[k]);
    s += "]";
  }
  s += ", " + reg_text(i.src[1 + addrs]);
  if (sample) s += ", " + reg_text(i.src[2 + addrs]);
  char b[32];
  std::snprintf(b, sizeof b, " dmask:0x%x", i.dmask);
  s += b;
  s += std::string(" dim:SQ_RSRC_IMG_") + kDim[i.dim & 7];
  if (i.unorm) s += " unorm";
  if (i.gfx12_cache) {
    s += gfx12_cache(i.name, i.cache);
  } else {
    if (i.cache & 1) s += " glc";
    if (i.cache & 2) s += " slc";
    if (i.cache & 4) s += " dlc";
  }
  if (i.r128) s += " r128";
  if (i.a16) s += " a16";
  if (i.tfe) s += " tfe";
  if (i.lwe) s += " lwe";
  if (i.d16) s += " d16";
  return s;
}

std::string one(const Inst& i) {
  const std::string& name = i.asm_name.empty() ? i.name : i.asm_name;
  std::string s = name;
  const bool short_form = i.enc == Enc::Vop1 || i.enc == Enc::Vop2 || i.enc == Enc::Vopc;
  // A short form with no long one to tell it from (v_fmaak_f32,
  // v_readfirstlane_b32) has no suffix.
  // (gfx1250's compare in DPP form is written with no suffix at all.)
  if (short_form && (i.dpp || i.dpp8)) s += g_cdna5 && i.enc == Enc::Vopc ? "" : "_dpp";
  else if (short_form && i.sdwa) s += "_e32";   // (made _sdwa below)
  else if (short_form && table().long_forms.count(name) && name != "v_readfirstlane_b32" && name != "v_nop")
    s += "_e32";
  // The long form of an instruction with a short one says so (_e64); the
  // table says which, whatever the executor runs it as.
  if (i.sdwa) s.resize(s.size() - 4), s += "_sdwa";   // for the _e32
  const bool promoted = i.enc == Enc::Vop3 && table().short_forms.count(name);
  if (i.enc == Enc::Vop3 && promoted) s += i.dpp || i.dpp8 ? "_e64_dpp" : "_e64";
  if (i.enc == Enc::Vop3 && !promoted && (i.dpp || i.dpp8)) s += "_e64_dpp";
  if (g_cdna5 && i.enc == Enc::Vop3p && (i.dpp || i.dpp8)) s += "_e64_dpp";

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
    if (i.src.empty()) return s;   // s_gl1_inv, s_dcache_inv, s_memtime's result alone
    put(op_text(i.src[0]));
    // gfx12's offset is 24 bits, signed; LLVM prints a negative one as -0x...
    const auto offset_text = [&] {
      return i.offset < 0 ? "-" + hex(static_cast<uint32_t>(-static_cast<int64_t>(i.offset))) : hex(static_cast<uint32_t>(i.offset));
    };
    if (i.gfx12_cache) {
      std::string scaled = i.scale_offset ? " scale_offset" : "";
      if (reg) put(op_text(i.src[1]) + " offset:" + offset_text() + scaled);
      else put(offset_text() + scaled);
      return s + gfx12_cache(name, i.cache) + (i.nv ? " nv" : "");
    }
    if (reg) put(op_text(i.src[1]) + (off ? " offset:" + hex(off) : ""));
    else put(off ? hex(off) : "null");
    if (i.cache & 1) s += " glc";
    if (i.cache & 4) s += " dlc";
    return s;
  }
  const bool scratch_no_addr = i.enc == Enc::Flat && i.segment == Inst::Segment::Scratch && !i.has_vaddr;
  // ds_cmpstore's data were put in gfx9's order for the executor; printed
  // in the order the instruction has them.
  const bool cmpstore = i.enc == Enc::Ds && name.rfind("ds_cmpstore", 0) == 0 && i.src.size() == 3;
  for (size_t k = 0; k < i.src.size(); ++k) {
    const Operand& o = cmpstore && k ? i.src[3 - k] : i.src[k];
    if (o.hidden) continue;
    // gfx1250's flat access with no scalar base names none (global and scratch write "off").
    if (g_cdna5 && i.enc == Enc::Flat && i.segment == Inst::Segment::Flat && !i.has_saddr && k + 1 == i.src.size() &&
        o.kind == OperandKind::Null)
      continue;
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
      if (two_addr(name)) {
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
      if (i.scale_offset) s += " scale_offset";
      if (i.gfx12_cache) {
        s += gfx12_cache(name, i.cache);
        if (i.nv) s += " nv";
        break;
      }
      if (i.cache & 1) s += " glc";
      if (i.cache & 2) s += " slc";
      if (i.cache & 4) s += " dlc";
      break;
    case Enc::Mubuf:
    case Enc::Mtbuf:
      if (i.idxen) s += " idxen";
      if (i.offen) s += " offen";
      if (i.offset) s += " offset:" + std::to_string(i.offset);
      if (i.gfx12_cache) {
        s += gfx12_cache(name, i.cache);
        if (i.nv) s += " nv";
        break;
      }
      if (i.cache & 1) s += " glc";
      if (i.cache & 2) s += " slc";
      if (i.cache & 4) s += " dlc";
      break;
    case Enc::Vop3:
      if (i.printed_op_sel) {
        const uint32_t n = static_cast<uint32_t>(i.src.size());
        std::string t = " op_sel:[";
        for (uint32_t k = 0; k < n; ++k) t += std::to_string((i.printed_op_sel >> k) & 1) + ",";
        t += std::to_string((i.printed_op_sel >> 3) & 1) + "]";
        s += t;
      }
      if (i.bitop3) {
        char t[24];
        std::snprintf(t, sizeof t, " bitop3:0x%x", i.bitop3);
        s += t;
      }
      if (i.clamp) s += " clamp";
      if (i.omod == 1) s += " mul:2";
      if (i.omod == 2) s += " mul:4";
      if (i.omod == 3) s += " div:2";
      break;
    case Enc::Vopd:
      if (i.bitop3) {
        char t[24];
        std::snprintf(t, sizeof t, " bitop3:0x%x", i.bitop3);
        s += t;
      }
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
      // gfx1250's matrix instructions use OP_SEL_HI for their formats, which are not written out here.
      const bool matrix = g_cdna5 && (name.rfind("v_wmma_", 0) == 0 || name.rfind("v_swmmac_", 0) == 0);
      list("op_sel", i.op_sel & all, 0);
      if (!matrix) list("op_sel_hi", i.op_sel_hi & all, mix ? 0 : all);
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
  if (i.sdwa) {
    static const char* kParts[8] = {"BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD", "?"};
    static const char* kUnused[4] = {"UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE", "?"};
    if (i.clamp) s += " clamp";
    if (i.omod) s += i.omod == 1 ? " mul:2" : i.omod == 2 ? " mul:4" : " div:2";
    if (i.enc != Enc::Vopc) s += std::string(" dst_sel:") + kParts[i.dst_sel & 7] + " dst_unused:" + kUnused[i.dst_unused & 3];
    for (size_t k = 0; k < i.src.size() && k < 2; ++k)
      if (i.src[k].kind != OperandKind::Vcc) s += " src" + std::to_string(k) + "_sel:" + kParts[i.src[k].sel & 7];
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
  const Generation generation(i.arch);
  if (i.enc == Enc::Vopd && i.dual.size() == 2) return one(i.dual[0]) + " :: " + one(i.dual[1]);
  if (i.enc == Enc::Mimg && i.name.rfind("tensor_", 0) == 0) {
    std::string s = i.name;
    std::string sep = " ";
    for (const Operand& o : i.src) {
      s += sep + op_text(o);
      sep = ", ";
    }
    return s + gfx12_cache(i.name, i.cache) + (i.nv ? " nv" : "");
  }
  if (i.enc == Enc::Mimg) return image_text(i);
  return one(i);
}

}  // namespace vgpu::amd::gcn::rdna
