// Turbo refusal census: for one module (the images of one fatbin) and one host card, which image turbo mode would give the card for each kernel, and if the
// kernel stays in the simulator, why. CPU only: no driver, no device.
//
// What is the runtime's own code here (linked, not copied): the image choice (hostgpu::pick_card_image, pick_retarget_image, retarget_ptx_text,
// describe_candidates), the PTX text scan (hostgpu::scan_ptx_for_card), the instruction classes and their refusal (turbo_class, call_needs_scan, call_stays_inside,
// call_to_entry, turbo_refusal, extern_kind, extern_admitted), the reading of a cubin (sass::analyze), the call graph (sass::reachable), the variable layout rule
// (variable_layout_mismatch) and the relocation patch (card_image_at_sim_addresses).
//
// What is NOT shared and was written again here: the walk over a kernel's code and its callees, and the order in which the refusals are tried. In the runtime they are
// inside Impl::module_card / Impl::kernel_eligible (src/runtime/host_gpu.cpp), which need a live Impl (driver state, memory manager, launch arguments) and are not
// callable without one; `scan_kernel` and `classify_module` below follow them statement by statement (the line comments name the function each part follows), and
// test_turbo_census.cpp holds them to turbo_class / turbo_refusal on hand-made modules. If those two functions change, this file must be changed with them.
// Differences from a real run, all in the direction of "this census sees less": what is decided at launch (argument pointers into host memory, VMM and managed
// allocations in use, the fit check, cooperative launches, the card's free memory, the driver refusing a module) is not seen; the card's refusal of PTX text the
// scan admits is not seen; a PTX module reports only its first blocker (scan_ptx_for_card stops at one).
#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <optional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/devrt.hpp"
#include "vgpu/host_gpu.hpp"
#include "vgpu/host_gpu_image.hpp"
#include "vgpu/host_gpu_source.hpp"
#include "vgpu/sass/exec.hpp"

namespace vgpu::census {

using hostgpu::CardCandidate;

// Which image the card would run.
enum class Image : uint8_t { Exact, LowerMinor, PtxJit, Retarget, None };
inline const char* image_name(Image i) {
  static const char* const n[] = {"exact", "lower-minor", "ptx-jit", "retarget", "none"};
  return n[static_cast<unsigned>(i)];
}

struct KernelCell {
  Image image = Image::None;
  bool admitted = false;
  bool unlisted = false;               // admitted only because turbo admits class unlisted (exact would refuse)
  hostgpu::ClassMask classes = 0;      // the classes of the code (SASS)
  std::vector<std::string> blockers;   // why it stays in the simulator, in the order the runtime tries them (the first is the runtime's reason)
};

struct Options {
  bool printf_off = false;   // VGPU_HOST_GPU_TURBO_OFF=printf
  hostgpu::TurboSwitches sw;
};

// Reads of cubins, shared between the cells of one module.
struct Analysis {
  std::map<size_t, std::shared_ptr<sass::Module>> by_candidate;
  std::map<size_t, std::string> failed;
  const sass::Module* get(const std::vector<CardCandidate>& c, size_t i) {
    if (const auto it = by_candidate.find(i); it != by_candidate.end()) return it->second.get();
    if (failed.count(i)) return nullptr;
    try {
      auto m = sass::analyze(reinterpret_cast<const uint8_t*>(c[i].bytes.data()), c[i].bytes.size());
      by_candidate[i] = m;
      return m.get();
    } catch (const Error& e) {
      failed[i] = e.message();
      return nullptr;
    }
  }
};

// The reason group of an instruction turbo_class made class Never (hostgpu::unmeasured_private_mnemonic and the rest of it).
inline std::string never_reason(const std::string& m) {
  if (hostgpu::texture_mnemonic(m)) return "never:texture-surface";
  if (m == "BPT") return "never:bpt";
  if (m == "UTMALDG" || m == "UTMASTG" || m == "UTMAREDG" || m == "UTMAPF" || m == "UTMACMDFLUSH" || m == "UBLKRED" || m == "UBLKCP") return "never:tma";
  if (m == "HGMMA" || m == "IGMMA" || m == "BGMMA" || m == "QGMMA" || m == "WARPGROUP" || m == "USETMAXREG") return "never:warpgroup";
  if (m == "UCGABAR" || m == "ENDCOLLECTIVE" || m == "STAS" || m == "REDAS" || m == "ARRIVES" || m == "LDGMC") return "never:cluster-dsmem";
  if (m == "SYNCS") return "never:syncs-mbarrier";
  if (m == "ELECT") return "never:elect";
  if (m == "STSM") return "never:stsm";
  if (m == "LDTM" || m == "STTM" || m == "UTCSHIFT" || m == "UTCMMA" || m == "UTCBAR" || m == "UTCCP" || m == "UTCATOMSWS") return "never:tensor-memory";
  if (m == "UGETNEXTWORKID" || m == "RPCMOV") return "never:workid-rpcmov";
  if (m == "JMP") return "never:jmp";
  return "never:other";
}

// The reason group of the text a PTX scan refused with (hostgpu::scan_ptx_for_card's messages).
inline std::string ptx_reason(const hostgpu::PtxScan& s) {
  const std::string& r = s.refuse;
  const auto has = [&](const char* t) { return r.find(t) != std::string::npos; };
  if (has("texture or surface")) return "ptx:texture-surface";
  if (has("module variable")) return "ptx:module-variable";
  if (has("trap") || has("brkpt")) return "ptx:trap";
  if (has("through a pointer")) return "ptx:indirect-call";
  if (has("cluster directive")) return "ptx:cluster-directive";
  if (has("which the card's architecture lacks")) return "ptx:retarget-form(" + r.substr(13, r.find(", which") - 13) + ")";
  if (has("sm_90 or later form")) {
    const size_t at = r.find('(');
    const std::string op = at == std::string::npos ? "" : r.substr(at + 1, r.find(')', at) - at - 1);
    if (op.rfind("cp.async.bulk", 0) == 0 || op.rfind("cp.reduce.async.bulk", 0) == 0 || op.rfind("prefetch.tensormap", 0) == 0 || op.rfind("tensormap.", 0) == 0) return "ptx:tma";
    if (op.rfind("wgmma", 0) == 0 || op.rfind("setmaxnreg", 0) == 0) return "ptx:warpgroup";
    if (op.rfind("elect", 0) == 0) return "ptx:elect";
    return "ptx:cluster-other";
  }
  if (has("PTX calls")) return "ptx:extern-call";
  return "ptx:other";
}

// `.entry` names of a PTX text.
inline std::vector<std::string> ptx_entries(const std::string& text) {
  std::vector<std::string> out;
  for (size_t at = text.find(".entry"); at != std::string::npos; at = text.find(".entry", at + 6)) {
    size_t i = at + 6;
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    size_t e = i;
    while (e < text.size() && (std::isalnum(static_cast<unsigned char>(text[e])) || text[e] == '_' || text[e] == '$' || text[e] == '.')) ++e;
    if (e > i) out.push_back(text.substr(i, e - i));
  }
  return out;
}

// The kernel's code and the code of the device functions it calls out of line: the walk of Impl::kernel_eligible. Fills `cell`.
inline void scan_kernel(const sass::Module& m, const std::string& kernel, bool printf_admitted, const Options& opt, KernelCell* cell) {
  using namespace hostgpu;
  std::vector<std::string> blockers;
  const auto add = [&](const std::string& b) {
    if (std::find(blockers.begin(), blockers.end(), b) == blockers.end()) blockers.push_back(b);
  };
  bool printf = false;
  // the externs the kernel reaches (kernel_eligible: the loop over sass::reachable)
  for (const std::string& f : sass::reachable(m, kernel)) {
    const bool devrt = exec::devrt::lookup(f) != exec::devrt::Fn::None;
    bool builtin = false;
    for (const auto& [addr, name] : m.builtins) builtin = builtin || name == f;
    if (!devrt && !builtin && std::find(m.cubin.externs.begin(), m.cubin.externs.end(), f) == m.cubin.externs.end()) continue;
    const ExternKind k = extern_kind(f, devrt);
    if (extern_admitted(k, Mode::Turbo, opt.printf_off) && printf_admitted) {
      printf = true;
      continue;
    }
    add(k == ExternKind::Assert ? "extern:assert" : k == ExternKind::Heap ? "extern:malloc-free" : k == ExternKind::DeviceLaunch ? "extern:device-launch" : k == ExternKind::Printf ? "extern:printf-off" : "extern:other");
  }
  const auto ci = m.code_index.find(".text." + kernel);
  if (ci == m.code_index.end()) {
    add("code:no-section");
    cell->blockers = blockers;
    return;
  }
  const sass::Code& code = m.code[ci->second];
  ClassMask classes = 0;
  std::vector<const sass::Code*> todo{&code};
  std::set<const sass::Code*> seen{&code};
  while (!todo.empty()) {
    const sass::Code& sec = *todo.back();
    todo.pop_back();
    const uint64_t code_lo = sec.count ? sec.instr(0).pc : 0, code_hi = sec.count ? sec.instr(sec.count - 1).pc : 0;
    for (size_t i = 0; i < sec.count; ++i) {
      const sass::Instr& ins = sec.instr(i);
      InsnClass c = turbo_class(ins.mnemonic, ins.mods);
      std::string why;
      if (c == InsnClass::Never) why = never_reason(ins.mnemonic);
      if (call_needs_scan(ins.op == sass::Op::CALL, c)) {
        const bool label = !ins.src.empty() && ins.src[0].kind == sass::Kind::Label;
        const bool rel = mods_have(ins.mods, "REL");
        const int64_t target = label ? ins.src[0].imm : -1;
        const auto b = label ? m.builtins.find(static_cast<uint64_t>(target)) : m.builtins.end();
        const bool to_printf = printf && !rel && b != m.builtins.end() && b->second == "vprintf";
        if (!to_printf && !call_stays_inside(rel, label, target, code_lo, code_hi)) {
          const sass::Code* callee = label && !rel && target >= 0 ? m.code_at(static_cast<uint64_t>(target)) : nullptr;
          if (callee && call_to_entry(rel, label, target, callee->base, callee->count)) {
            if (seen.insert(callee).second) todo.push_back(callee);
          } else {
            c = InsnClass::Never;
            why = label ? "never:call-unfollowed" : "never:call-through-register";
          }
        }
      }
      classes |= class_bit(c);
      if (c == InsnClass::Never) add(why);
    }
  }
  cell->classes = classes;
  const TurboRefusal rr = turbo_refusal(classes, opt.sw);
  if (rr == TurboRefusal::Off) add("switch:turbo-off");
  if (rr == TurboRefusal::Atomics) add("switch:atomics-off");
  cell->blockers = blockers;
  (void)printf;
}

// One module on one host card. `sim`: the index in `c` of the cubin the simulator runs for the profile (src/nvidia fatbin.cpp pick_cubin), or -1 when it runs the
// module from PTX. `kernels` are the module's kernel names (cubin kernels of the first cubin, else PTX entries; see kernel_names).
inline std::vector<KernelCell> classify_module(const std::vector<CardCandidate>& c, int sim, int host_sm, const std::vector<std::string>& kernels, Analysis& an, const Options& opt = {}) {
  using namespace hostgpu;
  std::vector<KernelCell> out(kernels.size());
  const auto all = [&](Image im, const std::string& why) {
    for (KernelCell& k : out) {
      k.image = im;
      k.blockers = {why};
    }
    return out;
  };
  int pick = pick_card_image(c, host_sm);
  bool retarget = false;
  if (pick < 0 && (pick = pick_retarget_image(c, host_sm)) >= 0) retarget = true;   // module_card
  if (pick < 0) return all(Image::None, "image:none");
  const CardCandidate& pc = c[static_cast<size_t>(pick)];
  if (pc.kind == CardCandidate::Ptx) {
    PtxScan ps = scan_ptx_for_card(pc.bytes, opt.printf_off, retarget);
    const Image im = retarget ? Image::Retarget : Image::PtxJit;
    if (retarget && retarget_ptx_text(pc.bytes, host_sm).empty()) return all(im, "image:ptx-not-plain-sm90");
    for (KernelCell& k : out) {
      k.image = im;
      k.classes = ps.classes;
      if (!ps.refuse.empty() && !ps.never) k.blockers.push_back(ptx_reason(ps));   // (kernel_eligible: only a refusal that is not class never is a Safety refusal ...)
      if (ps.never) k.blockers.push_back(ptx_reason(ps));                          // ... and a Never one is the Code refusal; both keep the kernel in the simulator
      if (const ClassMask gated = opt.sw.off | (opt.sw.atomics ? 0 : class_bit(InsnClass::AtomFloat) | class_bit(InsnClass::AtomSwap)); (ps.classes & gated) && k.blockers.empty())
        k.blockers.push_back("switch:turbo-off-or-atomics");
      k.admitted = k.blockers.empty();
    }
    return out;
  }
  // a cubin
  const Image im = pc.arch == host_sm ? Image::Exact : Image::LowerMinor;
  const sass::Module* view = an.get(c, static_cast<size_t>(pick));
  if (!view) return all(im, "image:unreadable");
  const sass::Module* simm = sim >= 0 ? an.get(c, static_cast<size_t>(sim)) : nullptr;
  std::string refuse;   // module_variables
  const bool same = pick == sim;
  if (!view->managed.empty() || (simm && !simm->managed.empty())) {
    refuse = "module:managed";
  } else {
    std::set<std::string> sections;
    bool any_variable = false;
    for (const sass::CubinSymbol& sym : view->cubin.symbols) {
      if (sym.function || sym.name.empty() || sym.name[0] == '.') continue;
      any_variable = true;
      if (simm && simm->symbol_va.count(sym.name) && simm->section_va.count(sym.section)) sections.insert(sym.section);
    }
    if (!same) {
      if (any_variable && !simm) refuse = "module:variables-sim-runs-ptx";
      else if (simm && !variable_layout_mismatch(view->cubin, simm->cubin).empty()) refuse = "module:variable-layout";
    }
    if (refuse.empty() && !sections.empty()) {
      const CardImage ci = card_image_at_sim_addresses(reinterpret_cast<const uint8_t*>(pc.bytes.data()), pc.bytes.size(), [&](const std::string& sec) -> std::optional<uint64_t> {
        if (!sections.count(sec)) return std::nullopt;
        return simm->section_va.at(sec);
      });
      if (!ci.ok) refuse = "module:variable-patch";
    }
  }
  for (size_t i = 0; i < kernels.size(); ++i) {
    KernelCell& k = out[i];
    k.image = im;
    if (!refuse.empty()) {
      k.blockers = {refuse};
      continue;
    }
    scan_kernel(*view, kernels[i], true, opt, &k);   // (turbo, not under VGPU_HOST_GPU_VERIFY: vprintf admitted)
    // cluster launches are refused per launch (kernel_eligible: fn.req_cluster); the cubin says which kernels need one
    const auto kit = view->kernel_index.find(kernels[i]);
    if (kit != view->kernel_index.end() && kit->second < view->cubin.kernels.size()) {
      const sass::CubinKernel& ck = view->cubin.kernels[kit->second];
      if (ck.explicit_cluster || ck.cluster != std::array<uint32_t, 3>{0, 0, 0}) k.blockers.push_back("launch:cluster-kernel");
    }
    k.admitted = k.blockers.empty();
    k.unlisted = k.admitted && (k.classes & class_bit(InsnClass::Unlisted));
  }
  return out;
}

// The kernel names of a module: the cubin kernels of the first cubin candidate, else the `.entry` names of the first PTX.
inline std::vector<std::string> kernel_names(const std::vector<CardCandidate>& c, Analysis& an) {
  std::set<std::string> names;
  for (size_t i = 0; i < c.size(); ++i)
    if (c[i].kind == CardCandidate::Cubin)
      if (const sass::Module* m = an.get(c, i)) {
        for (const sass::CubinKernel& k : m->cubin.kernels) names.insert(k.name);
        if (!names.empty()) return {names.begin(), names.end()};
      }
  for (const CardCandidate& p : c)
    if (p.kind == CardCandidate::Ptx) {
      for (const std::string& n : ptx_entries(p.bytes)) names.insert(n);
      if (!names.empty()) break;
    }
  return {names.begin(), names.end()};
}

}  // namespace vgpu::census
