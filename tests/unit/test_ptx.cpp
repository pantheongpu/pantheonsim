// Unit tests for the PTX lexer/parser subset.
#include "vgpu/ptx/parser.hpp"

#include <fstream>
#include <sstream>

#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu::ptx;
using vgpu::Err;
using vgpu::Error;

namespace {

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  VCHECK(f.good());
  std::ostringstream os;
  os << f.rdbuf();
  return os.str();
}

std::string wrap_kernel(const std::string& body, const std::string& params = "") {
  return ".version 8.3\n.target sm_90\n.address_size 64\n.visible .entry k(" + params + ")\n{\n" + body +
         "\n}\n";
}

}  // namespace

VTEST(parses_reference_vector_add) {
  Module m = parse(read_file(VGPU_KERNEL_DIR "/vector_add.ptx"));
  VCHECK_EQ(m.version, "8.3");
  VCHECK_EQ(m.target, "sm_90");
  VCHECK_EQ(m.address_size, 64u);
  VCHECK_EQ(m.entries.size(), size_t{1});

  const EntryFn& fn = m.entries[0];
  VCHECK_EQ(fn.name, "vecAdd");
  VCHECK_EQ(fn.params.size(), size_t{4});
  VCHECK_EQ(fn.params[0].name, "vecAdd_param_0");
  VCHECK_EQ(fn.params[0].ty.bytes(), 8u);
  VCHECK_EQ(fn.params[3].ty.bytes(), 4u);
  VCHECK_EQ(fn.body.size(), size_t{22});

  // Spot-check a few decoded instructions.
  VCHECK(std::holds_alternative<OpLd>(fn.body[0].op));
  VCHECK(std::get<OpLd>(fn.body[0].op).space == Space::Param);
  VCHECK(std::holds_alternative<OpMadLo>(fn.body[7].op));
  VCHECK(std::holds_alternative<OpSetp>(fn.body[8].op));
  const Instr& bra = fn.body[9];
  VCHECK(bra.has_pred);
  VCHECK(!bra.pred_negated);
  VCHECK_EQ(bra.pred, "%p1");
  VCHECK_EQ(std::get<OpBra>(bra.op).target, size_t{21});  // label points at ret
  VCHECK(std::holds_alternative<OpRet>(fn.body[21].op));
  // %p0/%p1 declared via .reg .pred %p<2>
  VCHECK(fn.reg_decls.count("%p1") == 1);
}

VTEST(special_registers_decoded) {
  Module m = parse(wrap_kernel("mov.u32 %r1, %ctaid.x;\nmov.u32 %r2, %laneid;\nret;"));
  auto& body = m.entries[0].body;
  VCHECK(std::get<SregOperand>(std::get<OpMov>(body[0].op).src).reg == Sreg::CtaidX);
  VCHECK(std::get<SregOperand>(std::get<OpMov>(body[1].op).src).reg == Sreg::LaneId);
}

VTEST(float_hex_immediates) {
  Module m = parse(wrap_kernel("mov.f32 %f1, 0f3F800000;\nret;"));
  auto imm = std::get<ImmFloatBits>(std::get<OpMov>(m.entries[0].body[0].op).src);
  VCHECK_EQ(imm.bits, 0x3F800000ull);
  VCHECK_EQ(imm.width, 32u);
}

VTEST(negative_offsets_and_immediates) {
  Module m = parse(wrap_kernel("ld.global.f32 %f1, [%rd1+-8];\nmov.s32 %r1, -5;\nret;"));
  VCHECK_EQ(std::get<OpLd>(m.entries[0].body[0].op).addr.offset, int64_t{-8});
  VCHECK_EQ(std::get<ImmInt>(std::get<OpMov>(m.entries[0].body[1].op).src).value, int64_t{-5});
}

VTEST(unsupported_instruction_names_kernel_and_line) {
  auto err = VCAPTURE(
      Error, parse(wrap_kernel("wmma.load.a.sync.aligned.m8n8k4.row.f16 {%f1}, [%rd1];\nret;")));
  VCHECK(err.code() == Err::UnsupportedPtx);
  VCHECK_CONTAINS(err.what(), "wmma.load");
  VCHECK_CONTAINS(err.what(), "kernel 'k'");
  VCHECK_CONTAINS(err.what(), "line 6");
}

VTEST(shared_memory_declarations_parse) {
  Module m = parse(wrap_kernel(".shared .align 4 .b8 buf[512];\n"
                               "st.shared.u32 [%rd1], %r1;\nret;"));
  const EntryFn& fn = m.entries[0];
  VCHECK_EQ(fn.shared.size(), size_t{1});
  VCHECK_EQ(fn.shared.at("buf").size, 512u);
  VCHECK_EQ(fn.static_shared_size, 512u);
  VCHECK(std::get<OpSt>(fn.body[0].op).space == Space::Shared);
}

VTEST(dynamic_shared_memory_sits_above_static) {
  Module m = parse(".version 8.3\n.target sm_90\n.address_size 64\n"
                   ".extern .shared .align 8 .b8 dyn[];\n"
                   ".visible .entry k()\n{\n.shared .align 4 .b8 st[256];\nret;\n}\n");
  const EntryFn& fn = m.entries[0];
  VCHECK_EQ(fn.static_shared_size, 256u);
  VCHECK(fn.uses_dynamic_shared);
  VCHECK_EQ(fn.shared.at("dyn").offset, 256u);
}

VTEST(undefined_label_is_parse_error) {
  auto err = VCAPTURE(Error, parse(wrap_kernel("bra NOWHERE;\nret;")));
  VCHECK(err.code() == Err::PtxParse);
  VCHECK_CONTAINS(err.what(), "NOWHERE");
}

VTEST(bad_syntax_reports_line) {
  auto err = VCAPTURE(Error, parse(".version 8.3\n.target sm_90\n.address_size 64\n.entry k {\nadd.f32 %f1 %f2;\n}\n"));
  VCHECK(err.code() == Err::PtxParse);
  VCHECK_CONTAINS(err.what(), "line 5");
}

VTEST(predicate_negation) {
  Module m = parse(wrap_kernel("setp.eq.s32 %p1, %r1, 0;\n@!%p1 bra DONE;\nmov.u32 %r2, 1;\nDONE:\nret;"));
  const Instr& bra = m.entries[0].body[1];
  VCHECK(bra.has_pred);
  VCHECK(bra.pred_negated);
  VCHECK_EQ(std::get<OpBra>(bra.op).target, size_t{3});
}

VTEST(comments_are_skipped) {
  Module m = parse(wrap_kernel("// line comment\n/* block\ncomment */ ret;"));
  VCHECK_EQ(m.entries[0].body.size(), size_t{1});
}

VTEST(only_64bit_address_size_supported) {
  auto err = VCAPTURE(Error, parse(".version 8.3\n.target sm_90\n.address_size 32\n"));
  VCHECK(err.code() == Err::UnsupportedPtx);
}

// nvcc emits a prototype ahead of the definition whenever something refers to
// a kernel before it appears -- taking its address, for instance, which is
// exactly what NVRTC's name expressions compile to.
VTEST(entry_prototype_is_a_declaration_not_a_definition) {
  Module m = parse(
      ".version 8.3\n.target sm_90\n.address_size 64\n"
      ".visible .entry k(\n.param .u64 k_param_0\n)\n;\n"
      ".visible .entry k(\n.param .u64 k_param_0\n)\n{\nret;\n}\n");
  VCHECK_EQ(m.entries.size(), size_t{1});
  VCHECK_EQ(m.entries[0].name, std::string("k"));
  VCHECK_EQ(m.entries[0].body.size(), size_t{1});
}

// The lexer gives numbers and identifiers the same token kind, so a scalar
// initialiser is one character away from being mistaken for a symbol -- which
// silently zeroed every initialised global the first time this was written.
VTEST(global_scalar_initialiser_is_not_a_symbol) {
  Module m = parse(
      ".version 8.3\n.target sm_90\n.address_size 64\n"
      ".global .align 4 .u32 answer = 42;\n"
      ".global .align 4 .f32 one = 0f3F800000;\n"
      ".visible .entry k()\n{\nret;\n}\n");
  VCHECK_EQ(m.globals.size(), size_t{2});
  VCHECK(m.globals[0].init_symbol.empty());
  VCHECK_EQ(m.globals[0].init.size(), size_t{4});
  VCHECK_EQ(int(m.globals[0].init[0]), 42);
  VCHECK(m.globals[1].init_symbol.empty());
  VCHECK_EQ(m.globals[1].init.size(), size_t{4});
}

VTEST(global_initialised_with_a_symbol) {
  Module m = parse(
      ".version 8.3\n.target sm_90\n.address_size 64\n"
      ".global .align 4 .u32 target[4];\n"
      ".global .align 8 .u64 pointer = target;\n"
      ".visible .entry k()\n{\nret;\n}\n");
  VCHECK_EQ(m.globals.size(), size_t{2});
  VCHECK_EQ(m.globals[1].name, std::string("pointer"));
  VCHECK_EQ(m.globals[1].init_symbol, std::string("target"));
  VCHECK(m.globals[1].init.empty());  // the address is only known at load time
}

// Cache hints tell the hardware how far to prefetch; they never change what a
// load returns. The lexer has to keep the doubled colon inside the opcode --
// splitting on it tore "ld.global.nc.L2::128B.u32" in half and made every
// flash-attention kernel unparseable rather than merely unsupported.
VTEST(cache_hints_are_accepted_and_ignored) {
  Module m = parse(
      ".version 8.3\n.target sm_90\n.address_size 64\n"
      ".visible .entry k(.param .u64 p)\n{\n"
      ".reg .b64 %rd<3>;\n.reg .b32 %r<3>;\n"
      "ld.param.u64 %rd1, [p];\n"
      "cvta.to.global.u64 %rd2, %rd1;\n"
      "ld.global.nc.L2::128B.u32 %r1, [%rd2];\n"
      "ld.global.L1::no_allocate.u32 %r2, [%rd2];\n"
      "st.global.L2::256B.u32 [%rd2], %r1;\n"
      "ret;\n}\n");
  VCHECK_EQ(m.entries.size(), size_t{1});
  // The hinted loads parse to exactly the instructions the unhinted ones would.
  const auto& body = m.entries[0].body;
  size_t loads = 0, stores = 0;
  for (const auto& i : body) {
    if (std::holds_alternative<OpLd>(i.op)) ++loads;
    if (std::holds_alternative<OpSt>(i.op)) ++stores;
  }
  VCHECK_EQ(loads, size_t{3});   // ld.param plus the two hinted global loads
  VCHECK_EQ(stores, size_t{1});
}

VTEST_MAIN
