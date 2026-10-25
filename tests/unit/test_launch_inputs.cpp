// The launch inputs a non-PTX frontend needs, and the special registers that read them.
//
//   E4  %warpsize (Sreg::WarpSize), the NDRange origin (Sreg::GlobalOffsetX/Y/Z) and the dimension count
//       (Sreg::WorkDim). PTX has no names for the last four, so these kernels are built with ptx::KernelBuilder.
//   E5  LaunchConfig::global_offset and work_dim, which those registers read; ParamDecl::local_mem, a work-group
//       memory pointer argument whose size the launch chooses and whose value the engine makes a shared offset; and
//       EntryFn::req_subgroups, SPIR-V's SubgroupsPerWorkgroup, checked against the launch.
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/builder.hpp"
#include "vgpu/registry.hpp"
#include "ast_ops.hpp"
#include "vtest.hpp"

using namespace vgpu;
using namespace astops;
using vgpu::exec::LaunchConfig;

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

struct Env {
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/h100");
  uint32_t word(uint64_t at) { return static_cast<uint32_t>(mem.load_scalar(at, 4)); }
};

// out[0..4] = warp size, global offset x, y, z, work dim; written by the one thread there is.
ptx::Module build_info() {
  ptx::KernelBuilder kb("info");
  const size_t po = kb.add_param("out", u64());
  const Reg out = kb.reg("%out", u64()), v = kb.reg("%v", u32());
  kb.emit(ld(Space::Param, u64(), out, kb.param_addr(po)), "ld");
  const Sreg regs[] = {Sreg::WarpSize, Sreg::GlobalOffsetX, Sreg::GlobalOffsetY, Sreg::GlobalOffsetZ, Sreg::WorkDim};
  for (int i = 0; i < 5; ++i) {
    kb.emit(mov(u32(), v, SR(regs[i])), "mov");
    kb.emit(st(Space::Global, u32(), ptx::KernelBuilder::reg_addr(out, 4 * i), RO(v)), "st");
  }
  kb.emit(OpRet{}, "ret");
  ptx::ModuleBuilder mb;
  mb.add_entry(kb.finish());
  return mb.finish();
}

}  // namespace

VTEST(the_new_special_registers_read_the_launch) {
  Env e;
  const ptx::Module m = build_info();
  const uint64_t out = e.mem.alloc(20);
  // A plain launch: the profile's width, no offset, three dimensions.
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.word(out + 0), 32u);
  VCHECK_EQ(e.word(out + 4), 0u);
  VCHECK_EQ(e.word(out + 8), 0u);
  VCHECK_EQ(e.word(out + 12), 0u);
  VCHECK_EQ(e.word(out + 16), 3u);
  // A launch with its own width, origin and dimension count.
  for (uint32_t W : {8u, 16u, 64u}) {
    LaunchConfig cfg;
    cfg.warp_size = W;
    cfg.global_offset = {7, 80, 900};
    cfg.work_dim = 2;
    exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
    VCHECK_EQ(e.word(out + 0), W);
    VCHECK_EQ(e.word(out + 4), 7u);
    VCHECK_EQ(e.word(out + 8), 80u);
    VCHECK_EQ(e.word(out + 12), 900u);
    VCHECK_EQ(e.word(out + 16), 2u);
  }
}

namespace {

// Two work-group memory arguments l0 and l1 (4-byte offsets), a static shared variable `sa` and the dynamic region
// `dyn`. Thread t < 16 writes t + 1 into l0[t] and 1000 + t into l1[t]; thread 0 writes 0xAAAA into sa and thread 1
// 0xBBBB into dyn; after a barrier every thread reads its four words back into out[4t .. 4t + 3].
ptx::Module build_regions() {
  ptx::KernelBuilder kb("regions");
  const size_t po = kb.add_param("out", u64());
  const size_t pl0 = kb.add_param("l0", u32(), 4, 4, /*local_mem=*/true);
  const size_t pl1 = kb.add_param("l1", u32(), 4, 4, /*local_mem=*/true);
  kb.declare_shared("sa", 100, 4);
  kb.declare_dynamic_shared("dyn", 16);
  const Reg out = kb.reg("%out", u64()), tid = kb.reg("%tid", u32()), b0 = kb.reg("%b0v", u32()),
            b1 = kb.reg("%b1v", u32()), a0 = kb.reg("%a0", u32()), a1 = kb.reg("%a1", u32()),
            v0 = kb.reg("%v0", u32()), v1 = kb.reg("%v1", u32()), w0 = kb.reg("%w0", u32()),
            w1 = kb.reg("%w1", u32()), w2 = kb.reg("%w2", u32()), w3 = kb.reg("%w3", u32()),
            off = kb.reg("%off", u64()), at = kb.reg("%at", u64());
  const Reg p0 = kb.pred(), p1 = kb.pred();
  const Addr sa = ptx::KernelBuilder::symbol_addr("sa"), dyn = ptx::KernelBuilder::symbol_addr("dyn");

  kb.emit(ld(Space::Param, u64(), out, kb.param_addr(po)), "ld");
  kb.emit(ld(Space::Param, u32(), b0, kb.param_addr(pl0)), "ld");
  kb.emit(ld(Space::Param, u32(), b1, kb.param_addr(pl1)), "ld");
  kb.emit(mov(u32(), tid, SR(Sreg::TidX)), "mov");
  kb.emit(madlo(u32(), a0, RO(tid), IM(4), RO(b0)), "mad");
  kb.emit(madlo(u32(), a1, RO(tid), IM(4), RO(b1)), "mad");
  kb.emit(ibin(IntBinOp::Add, u32(), v0, RO(tid), IM(1)), "add");
  kb.emit(ibin(IntBinOp::Add, u32(), v1, RO(tid), IM(1000)), "add");
  kb.emit(st(Space::Shared, u32(), ptx::KernelBuilder::reg_addr(a0), RO(v0)), "st");
  kb.emit(st(Space::Shared, u32(), ptx::KernelBuilder::reg_addr(a1), RO(v1)), "st");
  kb.emit(setp(CmpOp::Eq, u32(), p0, RO(tid), IM(0)), "setp");
  kb.emit_if(p0, false, st(Space::Shared, u32(), sa, IM(0xAAAA)), "st");
  kb.emit(setp(CmpOp::Eq, u32(), p1, RO(tid), IM(1)), "setp");
  kb.emit_if(p1, false, st(Space::Shared, u32(), dyn, IM(0xBBBB)), "st");
  kb.emit(barsync(0), "bar");
  kb.emit(ld(Space::Shared, u32(), w0, ptx::KernelBuilder::reg_addr(a0)), "ld");
  kb.emit(ld(Space::Shared, u32(), w1, ptx::KernelBuilder::reg_addr(a1)), "ld");
  kb.emit(ld(Space::Shared, u32(), w2, sa), "ld");
  kb.emit(ld(Space::Shared, u32(), w3, dyn), "ld");
  kb.emit(mulwide(false, off, RO(tid), IM(16)), "mul");
  kb.emit(ibin(IntBinOp::Add, u64(), at, RO(out), RO(off)), "add");
  kb.emit(st(Space::Global, u32(), ptx::KernelBuilder::reg_addr(at, 0), RO(w0)), "st");
  kb.emit(st(Space::Global, u32(), ptx::KernelBuilder::reg_addr(at, 4), RO(w1)), "st");
  kb.emit(st(Space::Global, u32(), ptx::KernelBuilder::reg_addr(at, 8), RO(w2)), "st");
  kb.emit(st(Space::Global, u32(), ptx::KernelBuilder::reg_addr(at, 12), RO(w3)), "st");
  kb.emit(OpRet{}, "ret");
  ptx::ModuleBuilder mb;
  mb.add_entry(kb.finish());
  return mb.finish();
}

// out[0] = the value of the 4-byte local-memory argument, out[1] = the second (or, for the 8-byte kernel, out as a u64).
ptx::Module build_echo(bool wide) {
  ptx::KernelBuilder kb(wide ? "echo8" : "echo");
  const size_t po = kb.add_param("out", u64());
  const size_t pl0 = kb.add_param("l0", wide ? u64() : u32(), wide ? 8 : 4, wide ? 8 : 4, /*local_mem=*/true);
  const Reg out = kb.reg("%out", u64());
  kb.emit(ld(Space::Param, u64(), out, kb.param_addr(po)), "ld");
  if (wide) {
    const Reg v = kb.reg("%v", u64());
    kb.emit(ld(Space::Param, u64(), v, kb.param_addr(pl0)), "ld");
    kb.emit(st(Space::Global, u64(), ptx::KernelBuilder::reg_addr(out), RO(v)), "st");
  } else {
    const size_t pl1 = kb.add_param("l1", u32(), 4, 4, /*local_mem=*/true);
    const Reg v0 = kb.reg("%v0", u32()), v1 = kb.reg("%v1", u32());
    kb.emit(ld(Space::Param, u32(), v0, kb.param_addr(pl0)), "ld");
    kb.emit(ld(Space::Param, u32(), v1, kb.param_addr(pl1)), "ld");
    kb.emit(st(Space::Global, u32(), ptx::KernelBuilder::reg_addr(out, 0), RO(v0)), "st");
    kb.emit(st(Space::Global, u32(), ptx::KernelBuilder::reg_addr(out, 4), RO(v1)), "st");
  }
  kb.emit(OpRet{}, "ret");
  ptx::ModuleBuilder mb;
  mb.add_entry(kb.finish());
  return mb.finish();
}

}  // namespace

// Work-group memory arguments: laid out after the static and dynamic bytes, 16-byte aligned, one after another, and
// the arguments are the offsets. A kernel using all of it at once finds every word where it put it.
VTEST(work_group_memory_arguments_become_shared_offsets) {
  Env e;
  // Just the offsets: nothing else in the shared window, 48 dynamic bytes, then 64 and 128 more.
  {
    const ptx::Module m = build_echo(false);
    const uint64_t out = e.mem.alloc(8);
    LaunchConfig cfg;
    cfg.shared_bytes = 48;
    exec::launch(m.entries[0], cfg, {arg_u64(out), arg_u32(64), arg_u32(128)}, e.mem, e.prof);
    VCHECK_EQ(e.word(out), 48u);          // right after the 48 dynamic bytes
    VCHECK_EQ(e.word(out + 4), 112u);     // 48 + 64
    // The same at another warp width: the layout is the engine's, not the warp's.
    cfg.warp_size = 8;
    exec::launch(m.entries[0], cfg, {arg_u64(out), arg_u32(0), arg_u32(16)}, e.mem, e.prof);
    VCHECK_EQ(e.word(out), 48u);          // a zero-byte argument still gets an (aligned) offset
    VCHECK_EQ(e.word(out + 4), 48u);
  }
  // An 8-byte argument takes the offset as a 64-bit value.
  {
    const ptx::Module m = build_echo(true);
    const uint64_t out = e.mem.alloc(8);
    LaunchConfig cfg;
    cfg.shared_bytes = 40;
    exec::launch(m.entries[0], cfg, {arg_u64(out), arg_u64(256)}, e.mem, e.prof);
    VCHECK_EQ(static_cast<uint64_t>(e.mem.load_scalar(out, 8)), uint64_t{48});   // 40 rounded up to 16
  }
  // The whole kernel, at every width: static `sa` (0..100), dynamic from 112, local arguments from 160 and 224.
  {
    const ptx::Module m = build_regions();
    const ptx::EntryFn& fn = m.entries[0];
    VCHECK_EQ(fn.static_shared_size, 100u);
    VCHECK_EQ(fn.dynamic_shared_offset, 112u);
    for (uint32_t W : {8u, 16u, 32u, 64u}) {
      const uint64_t out = e.mem.alloc(16 * 16);
      LaunchConfig cfg;
      cfg.block = {16, 1, 1};
      cfg.shared_bytes = 48;
      cfg.warp_size = W;
      exec::launch(fn, cfg, {arg_u64(out), arg_u32(64), arg_u32(128)}, e.mem, e.prof);
      for (uint32_t t = 0; t < 16; ++t) {
        VCHECK_EQ(e.word(out + 16 * t + 0), t + 1);
        VCHECK_EQ(e.word(out + 16 * t + 4), 1000 + t);
        VCHECK_EQ(e.word(out + 16 * t + 8), 0xAAAAu);
        VCHECK_EQ(e.word(out + 16 * t + 12), 0xBBBBu);
      }
      e.mem.free(out);
    }
  }
}

VTEST(work_group_memory_arguments_are_checked) {
  Env e;
  const ptx::Module m = build_echo(false);
  const uint64_t out = e.mem.alloc(8);
  // The argument is the size, as wide as the parameter.
  auto wrong = VCAPTURE(Error, exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out), arg_u64(64), arg_u32(8)}, e.mem, e.prof));
  VCHECK(wrong.code() == Err::LaunchConfig);
  VCHECK_CONTAINS(wrong.what(), "parameter 'l0' is 4 bytes, launch supplied 8 bytes");
  // More than a block can have.
  auto huge = VCAPTURE(Error, exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out), arg_u32(0x7FFFFFFFu), arg_u32(8)}, e.mem, e.prof));
  VCHECK(huge.code() == Err::LaunchConfig);
  auto too_much = VCAPTURE(Error, exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out), arg_u32(1u << 20), arg_u32(8)}, e.mem, e.prof));
  VCHECK(too_much.code() == Err::LaunchConfig);
  VCHECK_CONTAINS(too_much.what(), "bytes of shared memory");
}

// SubgroupsPerWorkgroup: the block must be exactly that many warps at the launch's width.
VTEST(a_required_sub_group_count_is_checked_against_the_launch) {
  Env e;
  ptx::KernelBuilder kb("sg");
  kb.set_req_subgroups(4);
  kb.emit(OpRet{}, "ret");
  ptx::ModuleBuilder mb;
  mb.add_entry(kb.finish());
  const ptx::Module m = mb.finish();
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};
  cfg.warp_size = 16;                                  // 64 / 16 = 4
  exec::launch(m.entries[0], cfg, {}, e.mem, e.prof);
  cfg.block = {50, 1, 1};                              // 50 threads of 16 lanes are 4 warps too
  exec::launch(m.entries[0], cfg, {}, e.mem, e.prof);
  cfg.block = {64, 1, 1};
  cfg.warp_size = 32;                                  // 2
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {}, e.mem, e.prof));
  VCHECK(err.code() == Err::LaunchConfig);
  VCHECK_CONTAINS(err.what(), "requires 4 sub-groups");
  VCHECK_CONTAINS(err.what(), "make 2");
}

VTEST_MAIN
