// What the executor used to read from a memory instruction's name on every execution (narrow, half_access,
// parse_atomic, transpose_kind, global_access's chain of prefix tests, buffer_access's atomic names, scalar_load's
// prefix tests) is now worked out once, when the instruction is decoded (Inst::mem, mem_facts_of). This holds
// the decoded facts against the old text tests, written out here as they were, for every instruction name of the
// ISA corpora and for the names a few thousand combinations of the segments, forms and suffixes make up (most of
// them no instruction), and checks the executor's fallback for an instruction nobody finalized.
#include <cstdint>
#include <string_view>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/amd_gcn.hpp"
#include "vtest.hpp"

using namespace vgpu::amd::gcn;
using vgpu::amd::gcn::Rmw;

namespace {

// ---- the old tests, verbatim (gcn_exec.cpp before this change) --------------------------------------------------
struct OldNarrow {
  uint32_t bytes = 0;
  bool sign = false;
};
bool old_narrow(const std::string& op, OldNarrow* n) {
  const size_t at = op.rfind('_');
  if (at == std::string::npos) return false;
  const std::string what = op.substr(at + 1);
  if (what == "ubyte" || what == "byte") *n = {1, false};
  else if (what == "sbyte") *n = {1, true};
  else if (what == "ushort" || what == "short") *n = {2, false};
  else if (what == "sshort") *n = {2, true};
  else return false;
  return true;
}
struct OldHalf {
  uint32_t bytes = 2;
  bool sign = false, hi = false, store = false;
};
bool old_half(std::string_view body, OldHalf* h) {
  if (body.find("_d16") == std::string_view::npos) return false;
  h->hi = body.size() >= 3 && body.substr(body.size() - 3) == "_hi";
  h->store = body.rfind("store", 0) == 0 || body.rfind("write", 0) == 0;
  h->bytes = body.find("short") != std::string_view::npos || body.find("16_d16") != std::string_view::npos ? 2 : 1;
  h->sign = body.find("sbyte") != std::string_view::npos || body.find("i8") != std::string_view::npos;
  return true;
}
struct OldAtomic {
  Rmw rmw = Rmw::Add;
  uint32_t bytes = 4;
};
bool old_atomic(std::string_view body, OldAtomic* a) {
  if (body.rfind("atomic_", 0) != 0) return false;
  std::string_view what = body.substr(7);
  a->bytes = 4;
  if (what.size() > 3 && what.substr(what.size() - 3) == "_x2") {
    a->bytes = 8;
    what.remove_suffix(3);
  }
  static const std::pair<std::string_view, Rmw> kNames[] = {
      {"add", Rmw::Add},       {"sub", Rmw::Sub},         {"and", Rmw::And},           {"or", Rmw::Or},
      {"xor", Rmw::Xor},       {"swap", Rmw::Swap},       {"smin", Rmw::SMin},         {"umin", Rmw::UMin},
      {"smax", Rmw::SMax},     {"umax", Rmw::UMax},       {"inc", Rmw::Inc},           {"dec", Rmw::Dec},
      {"cmpswap", Rmw::CmpSwap}, {"add_f32", Rmw::AddF32}, {"pk_add_f16", Rmw::PkAddF16},
      {"pk_add_bf16", Rmw::PkAddBf16}, {"add_f64", Rmw::AddF64}, {"min_f64", Rmw::MinF64},
      {"max_f64", Rmw::MaxF64}, {"min_num_f64", Rmw::MinF64}, {"max_num_f64", Rmw::MaxF64},
      {"fmin", Rmw::MinF32}, {"fmax", Rmw::MaxF32}, {"min_f32", Rmw::MinF32}, {"max_f32", Rmw::MaxF32},
      {"min_num_f32", Rmw::MinF32}, {"max_num_f32", Rmw::MaxF32}};
  for (auto [name, rmw] : kNames)
    if (what == name) {
      if (a->bytes == 8 && rmw == Rmw::MinF32) rmw = Rmw::MinF64;
      if (a->bytes == 8 && rmw == Rmw::MaxF32) rmw = Rmw::MaxF64;
      a->rmw = rmw;
      if (rmw == Rmw::AddF64 || rmw == Rmw::MinF64 || rmw == Rmw::MaxF64) a->bytes = 8;
      return !(a->bytes == 8 && (rmw == Rmw::AddF32 || rmw == Rmw::PkAddF16 || rmw == Rmw::PkAddBf16));
    }
  return false;
}
bool old_transpose(std::string_view body, uint32_t* elem_bits) {
  if (body == "load_tr16_b128") return *elem_bits = 16, true;
  if (body == "load_tr8_b64") return *elem_bits = 8, true;
  if (body == "load_tr6_b96") return *elem_bits = 6, true;
  if (body == "load_tr4_b64") return *elem_bits = 4, true;
  return false;
}
// global_access's order of tests.
MemKind old_kind(const std::string& op) {
  const std::string_view body = std::string_view(op).substr(op.find('_') + 1);
  OldHalf half;
  OldAtomic atomic;
  OldNarrow n;
  uint32_t bits;
  if (body == "load_block" || body == "store_block") return MemKind::Block;
  if (old_transpose(body, &bits)) return MemKind::Transpose;
  if (old_narrow(op, &n)) return MemKind::Narrow;
  if (old_half(body, &half)) return MemKind::HalfReg;
  if (body.rfind("prefetch_", 0) == 0) return MemKind::Prefetch;
  if (body.rfind("load_dword", 0) == 0 || body.rfind("load_monitor_b", 0) == 0) return MemKind::Load;
  if (body.rfind("store_dword", 0) == 0) return MemKind::Store;
  if (old_atomic(body, &atomic)) return MemKind::Atomic;
  return MemKind::Unknown;
}
// buffer_access's chain of names for an atomic (the operation each one is, Invalid for the final throw).
Rmw old_buffer_rmw(std::string_view body) {
  if (body == "atomic_cmpswap_x2") return Rmw::CmpSwap;
  if (body == "atomic_cmpswap") return Rmw::CmpSwap;
  if (body == "atomic_add") return Rmw::Add;
  if (body == "atomic_swap") return Rmw::Swap;
  if (body == "atomic_add_f32") return Rmw::AddF32;
  if (body == "atomic_pk_add_f16") return Rmw::PkAddF16;
  if (body == "atomic_sub") return Rmw::Sub;
  if (body == "atomic_smin") return Rmw::SMin;
  if (body == "atomic_smax") return Rmw::SMax;
  if (body == "atomic_umin") return Rmw::UMin;
  if (body == "atomic_umax") return Rmw::UMax;
  if (body == "atomic_and") return Rmw::And;
  if (body == "atomic_or") return Rmw::Or;
  if (body == "atomic_xor") return Rmw::Xor;
  if (body == "atomic_inc") return Rmw::Inc;
  if (body == "atomic_dec") return Rmw::Dec;
  return Rmw::Invalid;
}
// scalar_load's chain of names for a scalar atomic.
Rmw old_scalar_rmw(const std::string& what) {
  if (what == "swap") return Rmw::Swap;
  if (what == "cmpswap") return Rmw::CmpSwap;
  if (what == "add") return Rmw::Add;
  if (what == "sub") return Rmw::Sub;
  if (what == "smin") return Rmw::SMin;
  if (what == "umin") return Rmw::UMin;
  if (what == "smax") return Rmw::SMax;
  if (what == "umax") return Rmw::UMax;
  if (what == "and") return Rmw::And;
  if (what == "or") return Rmw::Or;
  if (what == "xor") return Rmw::Xor;
  if (what == "inc") return Rmw::Inc;
  if (what == "dec") return Rmw::Dec;
  return Rmw::Invalid;
}
SmemKind old_smem(const std::string& n) {
  if (n.find("prefetch") != std::string::npos) return SmemKind::Prefetch;
  if (n == "s_memtime" || n == "s_memrealtime") return SmemKind::Time;
  if (n == "s_dcache_wb" || n == "s_dcache_inv") return SmemKind::Nop;
  if (n.rfind("s_store_dword", 0) == 0) return SmemKind::Store;
  if (n.rfind("s_atomic_", 0) == 0) return SmemKind::Atomic;
  if (n == "s_load_u8" || n == "s_load_i8" || n == "s_load_u16" || n == "s_load_i16") return SmemKind::SubDword;
  if (n.rfind("s_buffer_load_", 0) == 0) return SmemKind::BufferLoad;
  return SmemKind::Load;
}

// Every field of the facts against the old tests, for one name; the first difference as text, "" if none.
std::string compare(const std::string& name) {
  Inst in;
  in.name = name;
  in.enc = Enc::Flat;
  const MemFacts f = mem_facts_of(in);
  const std::string_view body = std::string_view(name).substr(name.find('_') + 1);
  const auto bad = [&](const char* what) { return name + ": " + what; };
  OldNarrow n;
  const bool narrow = old_narrow(name, &n);
  if ((f.narrow_bytes != 0) != narrow || (narrow && (f.narrow_bytes != n.bytes || f.narrow_sign != n.sign))) return bad("narrow");
  OldHalf h;
  const bool half = old_half(body, &h);
  if ((f.half_bytes != 0) != half || (half && (f.half_bytes != h.bytes || f.half_sign != h.sign || f.half_hi != h.hi || f.half_store != h.store)))
    return bad("half");
  OldAtomic a;
  const bool atomic = old_atomic(body, &a);
  if ((f.atomic_bytes != 0) != atomic || (atomic && (f.atomic_bytes != a.bytes || f.rmw != a.rmw))) return bad("atomic");
  uint32_t bits = 0;
  const bool tr = old_transpose(body, &bits);
  if (f.transpose_bits != (tr ? bits : 0)) return bad("transpose");
  if (f.ds_name != (name.rfind("ds_", 0) == 0)) return bad("ds_name");
  if (f.kind != old_kind(name)) return bad("kind");
  if (f.body_store != (body.rfind("store", 0) == 0) || f.body_atomic != (body.rfind("atomic_", 0) == 0) ||
      f.load_dword != (body.rfind("load_dword", 0) == 0) || f.store_dword != (body.rfind("store_dword", 0) == 0) ||
      f.load_monitor != (body.rfind("load_monitor_b", 0) == 0) ||
      f.formatted != (body.rfind("load_format_", 0) == 0 || body.rfind("store_format_", 0) == 0) ||
      f.scratch_store != (name.rfind("scratch_store", 0) == 0) || f.scratch_load != (name.rfind("scratch_load", 0) == 0))
    return bad("prefix");
  if (f.buf_rmw != old_buffer_rmw(body) || f.buf_wide != (body == "atomic_cmpswap_x2")) return bad("buffer atomic");
  if (f.smem != old_smem(name)) return bad("smem kind");
  if (f.smem == SmemKind::Time && f.smem_realtime != (name == "s_memrealtime")) return bad("realtime");
  if (f.smem == SmemKind::SubDword && (f.smem_bytes != (name.back() == '8' ? 1 : 2) || f.smem_sign != (name[7] == 'i'))) return bad("sub-dword");
  if (f.smem == SmemKind::Atomic) {
    const bool wide = name.size() > 3 && name.compare(name.size() - 3, 3, "_x2") == 0;
    if (f.s_wide != wide || f.s_rmw != old_scalar_rmw(name.substr(9, name.size() - 9 - (wide ? 3 : 0)))) return bad("scalar atomic");
  }
  return "";
}

std::vector<std::string> corpus_names() {
  static const char* kFiles[] = {"isa_corpus.txt",        "isa_corpus_sopp_gfx942.txt", "isa_corpus_gfx950.txt", "isa_corpus_gfx90a.txt",
                                 "isa_corpus_gfx1100.txt", "isa_corpus_gfx1201.txt",     "isa_corpus_gfx1250.txt", "isa_corpus_gfx1030.txt"};
  static const Target kTargets[] = {Target::Gfx942,  Target::Gfx942,  Target::Gfx950,  Target::Gfx90a,
                                    Target::Gfx1100, Target::Gfx1200, Target::Gfx1250, Target::Gfx1030};
  std::set<std::string> names;
  for (size_t k = 0; k < std::size(kFiles); ++k) {
    std::ifstream f(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + kFiles[k], std::ios::binary);
    if (!f) throw vtest::Failure(std::string("no fixture ") + kFiles[k]);
    for (std::string line; std::getline(f, line);) {
      if (line.empty() || line[0] == '#') continue;
      const size_t tab = line.find('\t');
      std::vector<uint8_t> code;
      std::istringstream words(line.substr(0, tab));
      for (std::string w; words >> w;) {
        const uint32_t v = static_cast<uint32_t>(std::stoul(w, nullptr, 16));
        for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
      }
      try {
        names.insert(decode(code, 0, 0, kTargets[k]).name);
      } catch (const std::exception&) {
      }
    }
  }
  return {names.begin(), names.end()};
}

}  // namespace

VTEST(the_facts_decoded_with_an_instruction_are_what_the_old_name_tests_said) {
  std::vector<std::string> names = corpus_names();
  VCHECK(names.size() > 1500);
  // Segments, forms and suffixes in every combination (most are no instruction; the tests must agree about those too).
  const char* segs[] = {"global_", "flat_", "scratch_", "buffer_", "tbuffer_", "ds_", "s_", "s_buffer_", "x_"};
  const char* forms[] = {"load_ubyte", "load_sbyte", "load_byte", "load_ushort", "load_sshort", "load_short", "store_byte", "store_short", "load_dword",
                         "load_dwordx2", "load_dwordx4", "store_dword", "store_dwordx3", "load_monitor_b64", "load_monitor_b128", "load_block",
                         "store_block", "prefetch_b8", "load_tr16_b128", "load_tr8_b64", "load_tr6_b96", "load_tr4_b64", "load_format_x",
                         "store_format_xyzw", "atomic_add", "atomic_sub", "atomic_swap", "atomic_cmpswap", "atomic_cmpswap_x2", "atomic_add_x2",
                         "atomic_smin", "atomic_umax", "atomic_inc", "atomic_dec", "atomic_and", "atomic_or", "atomic_xor", "atomic_add_f32",
                         "atomic_pk_add_f16", "atomic_pk_add_bf16", "atomic_add_f64", "atomic_min_f64", "atomic_max_f64", "atomic_fmin",
                         "atomic_fmax_x2", "atomic_min_f32", "atomic_max_num_f32", "atomic_min_num_f64", "atomic_bogus", "atomic_",
                         "load_ubyte_d16", "load_ubyte_d16_hi", "load_sbyte_d16", "load_sbyte_d16_hi", "load_short_d16", "load_short_d16_hi",
                         "store_byte_d16_hi", "store_short_d16_hi", "read_u8_d16", "read_u8_d16_hi", "read_i8_d16", "read_u16_d16",
                         "write_b8_d16_hi", "write_b16_d16_hi", "read_b64", "read2_b32", "write_b128", "memtime", "memrealtime", "dcache_wb",
                         "dcache_inv", "load_u8", "load_i8", "load_u16", "load_i16", "load_dword", "load_b32", "store_dword", "atomic_swap_x2",
                         "atomic_cmpswap_x2", "atomic_smax_x2", "atomic_xor_x2", "atomic_nope", "prefetch_inst", "prefetch_data_pc_rel",
                         "load_lds_dword", "load_lds_dwordx4", "load_async_to_lds_b128", ""};
  const char* tails[] = {"", "_d16", "_hi", "_x2", "_saddr"};
  for (const char* s : segs)
    for (const char* fo : forms)
      for (const char* t : tails) names.push_back(std::string(s) + fo + t);
  size_t wrong = 0;
  std::string report;
  for (const std::string& n : names)
    if (const std::string d = compare(n); !d.empty() && ++wrong <= 10) report += "\n  " + d;
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(names.size()) + " names differ:" + report);
}

VTEST(an_instruction_nobody_finalized_gets_the_same_facts_when_the_executor_asks) {
  for (const char* n : {"global_load_ubyte", "buffer_atomic_cmpswap_x2", "ds_read_u8_d16_hi", "s_atomic_xor_x2", "flat_atomic_add_f32",
                        "s_load_i16", "global_load_tr16_b128", "scratch_store_dword"}) {
    Inst raw;
    raw.name = n;
    raw.enc = Enc::Flat;
    VCHECK(!(raw.flags & kInstFinal));
    MemFacts scratch;
    const MemFacts& a = mem_facts(raw, scratch);
    Inst done = raw;
    finalize(done);
    MemFacts unused;
    const MemFacts& b = mem_facts(done, unused);
    VCHECK(&b == &done.mem);
    VCHECK(a.kind == b.kind && a.smem == b.smem && a.rmw == b.rmw && a.buf_rmw == b.buf_rmw && a.s_rmw == b.s_rmw &&
           a.narrow_bytes == b.narrow_bytes && a.half_bytes == b.half_bytes && a.atomic_bytes == b.atomic_bytes &&
           a.transpose_bits == b.transpose_bits && a.smem_bytes == b.smem_bytes && a.narrow_sign == b.narrow_sign &&
           a.half_sign == b.half_sign && a.half_hi == b.half_hi && a.half_store == b.half_store && a.body_store == b.body_store &&
           a.body_atomic == b.body_atomic && a.load_dword == b.load_dword && a.store_dword == b.store_dword &&
           a.load_monitor == b.load_monitor && a.formatted == b.formatted && a.buf_wide == b.buf_wide && a.s_wide == b.s_wide &&
           a.smem_sign == b.smem_sign && a.smem_realtime == b.smem_realtime && a.scratch_store == b.scratch_store &&
           a.scratch_load == b.scratch_load && a.ds_name == b.ds_name);
  }
}

VTEST_MAIN
