// A call through a register that the scan can bound (docs/perf-design/12-host-gpu-turbo.md section 20), on decoded SASS. Kept apart from host_gpu.cpp so that the
// census tool (tools/turbo-census) runs this very function. The pure part is register_call_slot (vgpu/host_gpu.hpp).
#pragma once

#include <set>
#include <vector>

#include "vgpu/exec/devrt.hpp"
#include "vgpu/host_gpu.hpp"
#include "vgpu/sass/exec.hpp"

namespace vgpu::hostgpu {

// A call through a register (`CALL.ABS.NOINC Rt`) whose target register is the 64-bit load of a slot of bank 4 that the module's relocations fill with an external
// function turbo mode admits (vgpu/host_gpu.hpp register_call_slot: how the slot is found, and what is refused). Sets *assert_seen when that function is __assertfail.
inline bool register_call_to_admitted_extern(const sass::Module& m, const sass::Code& sec, size_t call, const TurboSwitches& sw, bool* assert_seen) {
  const sass::Instr& ci = sec.instr(call);
  if (ci.op != sass::Op::CALL || !mods_have(ci.mods, "ABS") || ci.src.empty() || ci.src[0].kind != sass::Kind::Reg) return false;
  std::vector<RcInsn> view(sec.count);
  std::set<uint64_t> targets;
  for (size_t i = 0; i < sec.count; ++i) {
    const sass::Instr& x = sec.instr(i);
    if (x.op == sass::Op::BRX || x.op == sass::Op::JMP) return false;   // (a jump table: any instruction can be entered)
    for (const auto& o : x.src)
      if (o.kind == sass::Kind::Label) targets.insert(static_cast<uint64_t>(o.imm));
  }
  for (size_t i = 0; i < sec.count; ++i) {
    const sass::Instr& x = sec.instr(i);
    RcInsn& r = view[i];
    r.guarded = x.guard != sass::kPT || x.guard_uniform;
    r.entry = targets.count(x.pc) != 0 || (i > 0 && sec.instr(i - 1).op == sass::Op::CALL);   // (the return of a call lands after it)
    unsigned regs_written = 0;
    for (const auto& d : x.dst)
      if (d.kind == sass::Kind::Reg && d.reg != sass::kRZ) {
        if (!regs_written++) r.dst_reg = d.reg, r.dst_n = d.width;
        else r.dst_reg = 0, r.dst_n = 256;   // (more than one register destination: say it writes every register)
      }
    switch (x.op) {
      case sass::Op::BRA: case sass::Op::BRX: case sass::Op::JMP: case sass::Op::EXIT: case sass::Op::RET: case sass::Op::BSSY: case sass::Op::BSYNC: case sass::Op::BREAK:
        r.kind = RcInsn::Control;
        break;
      case sass::Op::CALL:
        r.kind = (mods_have(x.mods, "ABS") && !x.src.empty() && x.src[0].kind == sass::Kind::Reg) ? RcInsn::CallReg : RcInsn::Control;
        if (r.kind == RcInsn::CallReg) r.reg = x.src[0].reg;
        break;
      case sass::Op::MOV:
        if (x.src.size() == 1 && x.src[0].kind == sass::Kind::Imm && !x.dst.empty() && x.dst[0].kind == sass::Kind::Reg) {
          r.kind = RcInsn::MovImm;
          r.reg = x.dst[0].reg;
          r.imm = x.src[0].imm;
        }
        break;
      case sass::Op::LDC:
        if (x.src.size() == 1 && x.src[0].kind == sass::Kind::Mem && !x.dst.empty()) {
          r.kind = RcInsn::LdcBank;
          r.reg = x.src[0].reg;
          r.imm = x.src[0].imm;
          r.bank = x.f[4];
          r.wide = mods_have(x.mods, "64");
        }
        break;
      default:
        break;
    }
  }
  const int64_t slot = register_call_slot(view, call);
  if (slot < 0) return false;
  const sass::CubinSection* bank4 = m.cubin.section(".nv.constant4");
  if (!bank4) return false;
  const sass::CubinReloc* hit = nullptr;
  for (const sass::CubinReloc& r : bank4->relocs)
    if (r.offset == static_cast<uint64_t>(slot)) {
      if (hit) return false;   // (two relocations at one slot: not understood)
      hit = &r;
    }
  if (!hit || hit->addend != 0 || hit->symbol_index >= m.cubin.symbols.size()) return false;
  const sass::CubinSymbol& sym = m.cubin.symbols[hit->symbol_index];
  if (!sym.function || !sym.section.empty()) return false;   // (a function the module defines is code the scan has to follow: not this form)
  const bool devrt = exec::devrt::lookup(sym.name) != exec::devrt::Fn::None;
  const ExternKind k = extern_kind(sym.name, devrt);
  if (!extern_admitted(k, Mode::Turbo, sw.printf_off, sw.assert_off)) return false;
  if (k == ExternKind::Assert && assert_seen) *assert_seen = true;
  return true;
}

}  // namespace vgpu::hostgpu
