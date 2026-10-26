// The AST builder (vgpu/ptx/builder.hpp) and module loading without PTX (Device::load_built_module).
//
// A frontend that does not read PTX text -- a SPIR-V lowering -- makes its kernels with ptx::KernelBuilder, assembles
// them with ptx::ModuleBuilder, hands the module to a device and launches it through the runtime facade. This builds
// a vector add with no PTX text at all and launches it at several warp widths; checks that what the builder lays
// out (registers, parameters, shared and local memory) is what the PTX parser lays out for the same declarations;
// that it leaves the mul/add pair alone where the PTX parser contracts it into an fma; that a module's constant
// data and module-scope .shared reach the kernels; and that misuse is refused by name.
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/ptx/builder.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "ast_ops.hpp"
#include "vtest.hpp"

using namespace vgpu;
using namespace astops;

namespace {

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}
std::vector<uint8_t> arg_u32(uint32_t v) {
  std::vector<uint8_t> b(4);
  std::memcpy(b.data(), &v, 4);
  return b;
}

// c[i] = a[i] + b[i] for i < n, one element a thread: the textbook kernel, built without a line of PTX.
ptx::EntryFn build_vadd(const std::string& name = "vadd") {
  ptx::KernelBuilder kb(name);
  const size_t pa = kb.add_param("a", u64()), pb = kb.add_param("b", u64()), pc = kb.add_param("c", u64()),
               pn = kb.add_param("n", u32());
  const Reg ra = kb.reg("%ra", u64()), rb = kb.reg("%rb", u64()), rc = kb.reg("%rc", u64()), n = kb.reg("%n", u32());
  const Reg tid = kb.reg("%tid", u32()), ntid = kb.reg("%ntid", u32()), cta = kb.reg("%cta", u32()),
            idx = kb.reg("%idx", u32());
  const Reg p = kb.pred();
  const Reg off = kb.reg("%off", u64()), aa = kb.reg("%aa", u64()), ab = kb.reg("%ab", u64()), ac = kb.reg("%ac", u64());
  const Reg x = kb.reg("%x", f32()), y = kb.reg("%y", f32()), z = kb.reg("%z", f32());
  const ptx::Label done = kb.make_label();

  kb.emit(ld(Space::Param, u64(), ra, kb.param_addr(pa)), "ld");
  kb.emit(ld(Space::Param, u64(), rb, kb.param_addr(pb)), "ld");
  kb.emit(ld(Space::Param, u64(), rc, kb.param_addr(pc)), "ld");
  kb.emit(ld(Space::Param, u32(), n, kb.param_addr(pn)), "ld");
  kb.emit(mov(u32(), tid, SR(Sreg::TidX)), "mov");
  kb.emit(mov(u32(), ntid, SR(Sreg::NtidX)), "mov");
  kb.emit(mov(u32(), cta, SR(Sreg::CtaidX)), "mov");
  kb.emit(madlo(u32(), idx, RO(cta), RO(ntid), RO(tid)), "mad");
  kb.emit(setp(CmpOp::Ge, u32(), p, RO(idx), RO(n)), "setp");
  kb.bra(done, p);
  kb.emit(mulwide(false, off, RO(idx), IM(4)), "mul");
  kb.emit(ibin(IntBinOp::Add, u64(), aa, RO(ra), RO(off)), "add");
  kb.emit(ibin(IntBinOp::Add, u64(), ab, RO(rb), RO(off)), "add");
  kb.emit(ibin(IntBinOp::Add, u64(), ac, RO(rc), RO(off)), "add");
  kb.emit(ld(Space::Global, f32(), x, ptx::KernelBuilder::reg_addr(aa)), "ld");
  kb.emit(ld(Space::Global, f32(), y, ptx::KernelBuilder::reg_addr(ab)), "ld");
  kb.emit(fbin(FloatBinOp::Add, z, RO(x), RO(y)), "add");
  kb.emit(st(Space::Global, f32(), ptx::KernelBuilder::reg_addr(ac), RO(z)), "st");
  kb.bind(done);
  kb.emit(OpRet{}, "ret");
  return kb.finish();
}

}  // namespace

// The test the feature exists for: a kernel built in memory, loaded through the runtime facade, launched at 8, 16, 32
// and 64 lanes, giving the host's sums.
VTEST(a_vector_add_built_without_ptx_runs_at_every_warp_width) {
  runtime::Runtime rt(load_gpu("nvidia/h100"));
  auto& dev = rt.device(0);
  ptx::ModuleBuilder mb;
  mb.add_entry(build_vadd());
  const uint64_t mod = dev.load_built_module(mb.finish());
  const ptx::EntryFn* fn = dev.get_function(mod, "vadd");
  VCHECK(fn != nullptr);
  VCHECK_EQ(fn->params.size(), size_t{4});

  const uint32_t n = 1000;
  std::vector<float> a(n), b(n), expect(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = 0.5f * i;
    b[i] = 100.0f - 3.0f * i;
    expect[i] = a[i] + b[i];
  }
  for (uint32_t W : {8u, 16u, 32u, 64u}) {
    auto& m = dev.memory();
    const uint64_t da = m.alloc(n * 4), db = m.alloc(n * 4), dc = m.alloc(n * 4);
    m.write(da, a.data(), n * 4);
    m.write(db, b.data(), n * 4);
    std::vector<float> zeros(n, -1.0f);
    m.write(dc, zeros.data(), n * 4);
    exec::LaunchConfig cfg;
    cfg.block = {96, 1, 1};                   // 96 is a whole number of warps at 8, 16 and 32 lanes, and 1.5 at 64
    cfg.grid = {(n + 95) / 96, 1, 1};
    cfg.warp_size = W;
    dev.launch(*fn, cfg, {arg_u64(da), arg_u64(db), arg_u64(dc), arg_u32(n)}, dev.symbols(mod));
    std::vector<float> c(n);
    m.read(dc, c.data(), n * 4);
    for (uint32_t i = 0; i < n; ++i) VCHECK_EQ(c[i], expect[i]);
    m.free(da);
    m.free(db);
    m.free(dc);
  }
  // The runtime's own entry for a built module is the device's.
  ptx::ModuleBuilder mb2;
  mb2.add_entry(build_vadd("vadd2"));
  const uint64_t mod2 = rt.load_built_module(0, mb2.finish());
  VCHECK(dev.get_function(mod2, "vadd2") != nullptr);
  dev.unload_module(mod2);
  dev.unload_module(mod);
}

// What the builder lays out is what the parser lays out for the same declarations.
VTEST(the_builder_lays_out_a_kernel_as_the_parser_does) {
  const ptx::Module parsed = ptx::parse(R"(
.version 8.3
.target sm_90
.address_size 64
.visible .entry lay(.param .u64 a, .param .u32 n, .param .align 16 .b8 blob[24])
{
  .reg .b32 %r<3>;
  .reg .b64 %rd<2>;
  .reg .pred %p<2>;
  .shared .align 4 .b8 sa[100];
  .shared .align 16 .b8 sb[8];
  .shared .align 32 .b8 dyn[];
  .local .align 8 .b8 lc[40];
  .local .align 16 .b8 l2[8];
  ret;
}
)");
  const ptx::EntryFn& want = parsed.entries.at(0);

  ptx::KernelBuilder kb("lay");
  kb.add_param("a", Type{Type::Kind::U, 64});
  kb.add_param("n", Type{Type::Kind::U, 32});
  kb.add_param("blob", Type{Type::Kind::B, 8}, 24, 16);
  const uint32_t sa = kb.declare_shared("sa", 100, 4), sb = kb.declare_shared("sb", 8, 16);
  kb.declare_dynamic_shared("dyn", 32);
  const uint32_t lc = kb.declare_local("lc", 40, 8), l2 = kb.declare_local("l2", 8, 16);
  kb.emit(OpRet{}, "ret");
  const ptx::EntryFn got = kb.finish();

  VCHECK_EQ(got.params.size(), want.params.size());
  for (size_t i = 0; i < want.params.size(); ++i) {
    VCHECK_EQ(got.params[i].name, want.params[i].name);
    VCHECK_EQ(got.params[i].size, want.params[i].size);
    VCHECK_EQ(got.params[i].align, want.params[i].align);
  }
  VCHECK_EQ(sa, want.shared.at("sa").offset);
  VCHECK_EQ(sb, want.shared.at("sb").offset);
  VCHECK_EQ(got.static_shared_size, want.static_shared_size);
  VCHECK_EQ(got.dynamic_shared_offset, want.dynamic_shared_offset);
  VCHECK_EQ(got.shared.at("dyn").offset, want.shared.at("dyn").offset);
  VCHECK_EQ(got.uses_dynamic_shared, want.uses_dynamic_shared);
  VCHECK_EQ(lc, want.locals.at("lc").offset);
  VCHECK_EQ(l2, want.locals.at("l2").offset);
  VCHECK_EQ(got.local_frame_size, want.local_frame_size);
  // The static data ends at 120 (100 -> 112 for sb's 16-byte alignment, + 8), the dynamic part starts at the next 32.
  VCHECK_EQ(got.static_shared_size, 120u);
  VCHECK_EQ(got.dynamic_shared_offset, 128u);
}

// Registers: dense per file, a predicate among the narrow ones, wide ones in the other file; and the same bits come out
// of a kernel the parser read.
VTEST(registers_are_interned_densely_in_two_files) {
  ptx::KernelBuilder kb("regs");
  const Reg a = kb.reg("%a", u32()), b = kb.reg("%b", u64()), c = kb.reg("%c", u32()), p = kb.pred();
  const Reg a2 = kb.reg("%a", u32());
  VCHECK_EQ(a.id, 0u);
  VCHECK(!a.wide);
  VCHECK_EQ(c.id, 1u);
  VCHECK_EQ(p.id, 2u);            // a predicate is a narrow register
  VCHECK_EQ(b.id, 0u);
  VCHECK(b.wide);
  VCHECK_EQ(a2.id, a.id);         // the same name is the same register
  kb.emit(OpRet{}, "ret");
  const ptx::EntryFn fn = kb.finish();
  VCHECK_EQ(fn.num_regs32, 3u);
  VCHECK_EQ(fn.num_regs64, 1u);
  VCHECK_EQ(fn.num_regs, 4u);
  // A name declared at the other width is refused (the parser does the same).
  ptx::KernelBuilder again("w");
  again.reg("%x", u32());
  auto err = VCAPTURE(Error, again.reg("%x", u64()));
  VCHECK(err.code() == Err::InvalidValue);
}

// The one PTX rewrite the builder does not do. mul.f32 then add.f32 with no rounding modifier is ptxas's to fuse (the
// PTX parser does); a lowering that means an fma writes one. The values are chosen so the two differ: a = 1 + 2^-12,
// a*a = 1 + 2^-11 + 2^-24 (half an ulp over a representable number, which rounds to even), minus 1 + 2^-11:
// unfused 0, fused 2^-24.
VTEST(the_builder_does_not_contract_a_multiply_and_an_add) {
  ::unsetenv("VGPU_PTX_CONTRACT");
  runtime::Runtime rt(load_gpu("nvidia/h100"));
  auto& dev = rt.device(0);

  // The PTX text.
  const uint64_t pmod = dev.load_module(R"(
.version 8.3
.target sm_90
.address_size 64
.visible .entry k(.param .u64 out)
{
  .reg .f32 %f<6>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [out];
  mov.f32 %f1, 0f3F800800;
  mov.f32 %f2, 0f3F800800;
  mov.f32 %f3, 0fBF801000;
  mul.f32 %f4, %f1, %f2;
  add.f32 %f5, %f4, %f3;
  st.global.f32 [%rd1], %f5;
  ret;
}
)");
  // The same program, built.
  ptx::KernelBuilder kb("k");
  const size_t po = kb.add_param("out", u64());
  const Reg rd = kb.reg("%rd1", u64()), f1 = kb.reg("%f1", f32()), f2 = kb.reg("%f2", f32()),
            f3 = kb.reg("%f3", f32()), f4 = kb.reg("%f4", f32()), f5 = kb.reg("%f5", f32());
  kb.emit(ld(Space::Param, u64(), rd, kb.param_addr(po)), "ld");
  const float a = std::bit_cast<float>(0x3F800800u), c = std::bit_cast<float>(0xBF801000u);
  kb.emit(mov(f32(), f1, F32(a)), "mov");
  kb.emit(mov(f32(), f2, F32(a)), "mov");
  kb.emit(mov(f32(), f3, F32(c)), "mov");
  kb.emit(fbin(FloatBinOp::Mul, f4, RO(f1), RO(f2)), "mul");
  kb.emit(fbin(FloatBinOp::Add, f5, RO(f4), RO(f3)), "add");
  kb.emit(st(Space::Global, f32(), ptx::KernelBuilder::reg_addr(rd), RO(f5)), "st");
  kb.emit(OpRet{}, "ret");
  ptx::ModuleBuilder mb;
  mb.add_entry(kb.finish());
  const uint64_t bmod = dev.load_built_module(mb.finish());

  auto count_fma = [](const ptx::EntryFn& f) {
    size_t n = 0;
    for (const ptx::Instr& i : f.body) n += std::holds_alternative<ptx::OpFma>(i.op);
    return n;
  };
  VCHECK_EQ(count_fma(*dev.get_function(pmod, "k")), size_t{1});   // the parser fused them
  VCHECK_EQ(count_fma(*dev.get_function(bmod, "k")), size_t{0});   // the builder left them

  auto run = [&](uint64_t mod) {
    const uint64_t out = dev.memory().alloc(4);
    dev.launch(*dev.get_function(mod, "k"), exec::LaunchConfig{}, {arg_u64(out)}, dev.symbols(mod));
    uint32_t bits = 0;
    dev.memory().read(out, &bits, 4);
    dev.memory().free(out);
    return bits;
  };
  VCHECK_EQ(run(pmod), 0x33800000u);   // 2^-24: one rounding
  VCHECK_EQ(run(bmod), 0u);            // two roundings
}

// A module's constant data reaches the kernels through the symbol table, and a module-scope .shared variable is charged
// to the kernels that mention it and to no others.
VTEST(module_globals_and_module_shared_reach_the_kernels_that_name_them) {
  runtime::Runtime rt(load_gpu("nvidia/h100"));
  auto& dev = rt.device(0);
  ptx::ModuleBuilder mb;

  ptx::GlobalVar g;
  g.name = "tbl";
  g.size = 32;
  g.align = 4;
  g.is_const = true;
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t v = 10 * (i + 1);
    for (int b = 0; b < 4; ++b) g.init.push_back(static_cast<uint8_t>(v >> (8 * b)));
  }
  mb.add_global(g);
  ptx::SharedDecl ms;
  ms.name = "msh";
  ms.size = 64;
  ms.align = 16;
  mb.add_module_shared(ms);

  {  // out[tid] = tbl[tid], and it uses msh
    ptx::KernelBuilder kb("reads_table");
    const size_t po = kb.add_param("out", u64());
    const Reg out = kb.reg("%out", u64()), tid = kb.reg("%tid", u32()), base = kb.reg("%base", u64()),
              off = kb.reg("%off", u64()), src = kb.reg("%src", u64()), dst = kb.reg("%dst", u64()),
              v = kb.reg("%v", u32());
    kb.emit(ld(Space::Param, u64(), out, kb.param_addr(po)), "ld");
    kb.emit(mov(u32(), tid, SR(Sreg::TidX)), "mov");
    kb.emit(mov(u64(), base, SYM("tbl")), "mov");
    kb.emit(mulwide(false, off, RO(tid), IM(4)), "mul");
    kb.emit(ibin(IntBinOp::Add, u64(), src, RO(base), RO(off)), "add");
    kb.emit(ibin(IntBinOp::Add, u64(), dst, RO(out), RO(off)), "add");
    kb.emit(ld(Space::Global, u32(), v, ptx::KernelBuilder::reg_addr(src)), "ld");
    kb.emit(st(Space::Global, u32(), ptx::KernelBuilder::reg_addr(dst), RO(v)), "st");
    kb.emit(st(Space::Shared, u32(), ptx::KernelBuilder::symbol_addr("msh"), RO(v)), "st");   // mentions msh
    kb.emit(OpRet{}, "ret");
    mb.add_entry(kb.finish());
  }
  {  // a kernel that never names msh
    ptx::KernelBuilder kb("plain");
    kb.emit(OpRet{}, "ret");
    mb.add_entry(kb.finish());
  }
  const uint64_t mod = dev.load_built_module(mb.finish());
  const ptx::EntryFn* reads = dev.get_function(mod, "reads_table");
  const ptx::EntryFn* plain = dev.get_function(mod, "plain");
  VCHECK_EQ(reads->static_shared_size, 64u);
  VCHECK(reads->shared.count("msh") == 1);
  VCHECK_EQ(plain->static_shared_size, 0u);
  VCHECK(plain->shared.empty());

  const uint64_t out = dev.memory().alloc(32);
  exec::LaunchConfig cfg;
  cfg.block = {8, 1, 1};
  cfg.warp_size = 16;
  dev.launch(*reads, cfg, {arg_u64(out)}, dev.symbols(mod));
  for (uint32_t i = 0; i < 8; ++i) VCHECK_EQ(dev.memory().load_scalar(out + 4 * i, 4), uint64_t{10 * (i + 1)});
  dev.memory().free(out);
}

// Misuse is refused with the function named.
VTEST(builder_misuse_is_refused_by_name) {
  {
    ptx::KernelBuilder kb("unbound");
    kb.bra(kb.make_label());
    auto err = VCAPTURE(Error, kb.finish());
    VCHECK(err.code() == Err::InvalidValue);
    VCHECK_CONTAINS(err.what(), "unbound");
    VCHECK_CONTAINS(err.what(), "never bound");
  }
  {
    ptx::KernelBuilder kb("twice");
    const ptx::Label l = kb.make_label();
    kb.bind(l);
    auto err = VCAPTURE(Error, kb.bind(l));
    VCHECK_CONTAINS(err.what(), "bound twice");
  }
  {
    ptx::KernelBuilder kb("width");
    kb.reg("%x", u32());
    auto err = VCAPTURE(Error, kb.reg("%x", u64()));
    VCHECK(err.code() == Err::InvalidValue);
    VCHECK_CONTAINS(err.what(), "'%x'");
  }
  {
    ptx::KernelBuilder kb("dup");
    kb.declare_shared("s", 16);
    auto err = VCAPTURE(Error, kb.declare_shared("s", 16));
    VCHECK_CONTAINS(err.what(), "duplicate .shared");
  }
  {
    ptx::KernelBuilder kb("badlocal");
    auto err = VCAPTURE(Error, kb.add_param("p", u32(), 3, 0, /*local_mem=*/true));
    VCHECK_CONTAINS(err.what(), "4 or 8 bytes");
  }
  {
    ptx::KernelBuilder kb("spent");
    kb.emit(OpRet{}, "ret");
    (void)kb.finish();
    auto err = VCAPTURE(Error, kb.emit(OpRet{}, "ret"));
    VCHECK_CONTAINS(err.what(), "spent");
  }
  {
    ptx::ModuleBuilder mb;
    ptx::KernelBuilder f("helper", /*device_func=*/true);
    f.emit(OpRet{}, "ret");
    auto err = VCAPTURE(Error, mb.add_entry(f.finish()));
    VCHECK_CONTAINS(err.what(), "add_func");
  }
}

VTEST_MAIN
