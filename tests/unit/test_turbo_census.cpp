// The refusal census (tools/turbo-census/census.hpp) must agree with turbo's own classifier. The census re-walks a kernel's code (the runtime's walk is inside
// Impl::kernel_eligible and cannot be called alone), so this test holds the walk to hostgpu::turbo_class / turbo_refusal / pick_card_image /
// scan_ptx_for_card on hand-made modules: cubins built word by word (SASS for sm_86 and sm_90), and PTX texts.
// No driver, no device.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "census.hpp"
#include "vgpu/host_gpu.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;
using hostgpu::CardCandidate;

namespace {

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

struct W {
  uint64_t hi, lo;
};
// opcode word with the guard PT (7) in bits 12-14; hi as nvcc writes it for a control-flow instruction
W op(uint64_t opcode) { return {0x000fea0003800000ull, opcode | uint64_t{7} << 12}; }
W exit_() { return {0x000fea0003800000ull, 0x000000000000794dull}; }
W iadd3() { return {0x000fc80007ffe000ull, 0x7210ull | uint64_t{1} << 16 | uint64_t{2} << 24 | uint64_t{3} << 32}; }

// A linked cubin for sm_<sm> with one kernel "k" (one 8-byte parameter) whose code is `code` (tests/unit/test_sass_dispatch.cpp make_cubin).
std::string make_cubin(unsigned sm, const std::vector<W>& code, bool with_symbol = false) {
  (void)with_symbol;
  const std::string names("\0.shstrtab\0.text.k\0.nv.info.k\0", 31);
  std::string info;
  put<uint8_t>(info, 4), put<uint8_t>(info, 0x17), put<uint16_t>(info, 12);
  put<uint32_t>(info, 0), put<uint32_t>(info, 0), put<uint32_t>(info, 8u << 18);
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  while (out.size() % 16) out.push_back('\0');
  const uint64_t text_off = out.size();
  for (const W& i : code) put(out, i.lo), put(out, i.hi);
  const uint64_t info_off = out.size();
  out += info;
  while (out.size() % 8) out.push_back('\0');
  const uint64_t shoff = out.size();
  const auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size) {
    put<uint32_t>(out, name), put<uint32_t>(out, type);
    put<uint64_t>(out, 0), put<uint64_t>(out, 0), put<uint64_t>(out, off), put<uint64_t>(out, size);
    put<uint32_t>(out, 0), put<uint32_t>(out, 0), put<uint64_t>(out, 16), put<uint64_t>(out, 0);
  };
  shdr(0, 0, 0, 0);
  shdr(1, 3, names_off, names.size());
  shdr(11, 1, text_off, 16 * code.size());
  shdr(19, 0x70000000, info_off, info.size());
  out[0] = 0x7f, out[1] = 'E', out[2] = 'L', out[3] = 'F', out[4] = 2, out[5] = 1, out[6] = 1, out[7] = 0x41, out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = 4, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (sm << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

CardCandidate cubin(unsigned sm, const std::vector<W>& code) {
  CardCandidate c;
  c.kind = CardCandidate::Cubin;
  c.arch = static_cast<int>(sm);
  c.bytes = make_cubin(sm, code);
  return c;
}
CardCandidate ptx(int target, const std::string& body) {
  CardCandidate c;
  c.kind = CardCandidate::Ptx;
  c.arch = target;
  c.bytes = ".version 8.0\n.target sm_" + std::to_string(target) + "\n.address_size 64\n" + body;
  return c;
}
const std::string kPtxOk = ".visible .entry k(.param .u64 p)\n{\n.reg .b32 r<2>;\nmov.u32 r0, %tid.x;\nret;\n}\n";

// The cell of the module's one kernel on a host card; the simulator runs candidate `sim`.
census::KernelCell cell(const std::vector<CardCandidate>& c, int sim, int host) {
  census::Analysis an;
  const std::vector<std::string> names = census::kernel_names(c, an);
  VCHECK_EQ(names.size(), size_t{1});
  return census::classify_module(c, sim, host, names, an).at(0);
}
std::string joined(const census::KernelCell& k) {
  std::string s;
  for (const std::string& b : k.blockers) s += (s.empty() ? "" : ",") + b;
  return s;
}
// What turbo's classifier says of the words, one by one: the union of the class bits, and whether any is Never.
hostgpu::ClassMask classes_of(unsigned sm, const std::vector<W>& code) {
  hostgpu::ClassMask m = 0;
  for (size_t i = 0; i < code.size(); ++i) {
    sass::Word w;
    w.hi = code[i].hi, w.lo = code[i].lo;
    const sass::Instr ins = sass::decode(w, 16 * i, static_cast<int>(sm));
    m |= hostgpu::class_bit(hostgpu::turbo_class(ins.mnemonic, ins.mods));
  }
  return m;
}
std::string text_of(unsigned sm, const W& x) {
  sass::Word w;
  w.hi = x.hi, w.lo = x.lo;
  return sass::decode(w, 0, static_cast<int>(sm)).mnemonic;
}

}  // namespace

VTEST(census_plain_kernel_runs_where_its_image_does) {
  const std::vector<W> code{iadd3(), exit_()};
  const std::vector<CardCandidate> c{cubin(86, code)};
  VCHECK_EQ(text_of(86, code[0]), std::string("IADD3"));
  census::KernelCell k = cell(c, 0, 86);
  VCHECK(k.admitted);
  VCHECK_EQ(std::string(census::image_name(k.image)), std::string("exact"));
  k = cell(c, 0, 89);   // the same major, a newer minor: the lower-minor cubin
  VCHECK(k.admitted);
  VCHECK_EQ(std::string(census::image_name(k.image)), std::string("lower-minor"));
  k = cell(c, 0, 75);   // an sm_86 cubin does not run on sm_75, and there is no PTX
  VCHECK(!k.admitted);
  VCHECK_EQ(joined(k), std::string("image:none"));
  k = cell(c, 0, 90);   // nor on another major
  VCHECK_EQ(joined(k), std::string("image:none"));
}

VTEST(census_picks_the_image_as_the_runtime_does) {
  const std::vector<W> code{iadd3(), exit_()};
  const std::vector<CardCandidate> c{cubin(75, code), cubin(86, code), ptx(86, kPtxOk)};
  for (int host : {75, 80, 86, 87, 89, 90, 120}) {
    const int want = hostgpu::pick_card_image(c, host);
    const census::KernelCell k = cell(c, 1, host);
    const char* expect = want < 0 ? "none" : c[static_cast<size_t>(want)].kind == CardCandidate::Ptx ? "ptx-jit" : c[static_cast<size_t>(want)].arch == host ? "exact" : "lower-minor";
    VCHECK_EQ(std::string(census::image_name(k.image)), std::string(expect));
  }
}

VTEST(census_never_instructions_follow_turbo_class) {
  struct Case {
    const char* what;
    unsigned sm;
    W word;
    const char* mnemonic;
    const char* blocker;
  };
  const Case cases[] = {
      {"breakpoint", 86, {0x000fea0003800000ull, 0x95cull | uint64_t{7} << 12}, "BPT", "never:bpt"},
      {"texture", 86, {0x000fea0003800000ull, 0xb60ull | uint64_t{7} << 12}, "TEX", "never:texture-surface"},
      {"call through a register", 86, {0x000fea0003800000ull, 0x343ull | uint64_t{7} << 12 | uint64_t{4} << 24}, "CALL", "never:call-through-register"},
      {"tensor-map load", 90, {0x000fea0003800000ull, 0x5b4ull | uint64_t{7} << 12}, "UTMALDG", "never:tma"},
  };
  for (const Case& cs : cases) {
    const std::vector<W> code{cs.word, exit_()};
    VCHECK_EQ(text_of(cs.sm, cs.word), std::string(cs.mnemonic));   // (the word is the instruction the case means)
    const hostgpu::ClassMask want = classes_of(cs.sm, code);
    VCHECK(want & hostgpu::class_bit(hostgpu::InsnClass::Never));
    VCHECK(hostgpu::turbo_refusal(want, {}) == hostgpu::TurboRefusal::Never);
    const census::KernelCell k = cell({cubin(cs.sm, code)}, 0, static_cast<int>(cs.sm));
    VCHECK(!k.admitted);
    VCHECK_EQ(k.classes, want);
    VCHECK_EQ(joined(k), std::string(cs.blocker));
  }
}

VTEST(census_admits_what_turbo_admits) {
  // RED / ATOM / MUFU forms and IADD3: classes are turbo_class's, none is Never, so the kernel is admitted (the atomics switch is on by default)
  const std::vector<W> code{iadd3(), {0x000fea0003800000ull, 0x108ull | uint64_t{7} << 12}, exit_()};
  const hostgpu::ClassMask want = classes_of(86, code);
  VCHECK(hostgpu::turbo_refusal(want, {}) == hostgpu::TurboRefusal::None);
  const census::KernelCell k = cell({cubin(86, code)}, 0, 86);
  VCHECK_EQ(k.classes, want);
  VCHECK(k.admitted);
  // the switches turbo has: a class switched off refuses it (hostgpu::turbo_refusal), and the census says so
  census::Options off;
  off.sw.off = want & ~hostgpu::class_bit(hostgpu::InsnClass::Exact);
  if (off.sw.off) {
    census::Analysis an;
    const std::vector<CardCandidate> c{cubin(86, code)};
    const census::KernelCell o = census::classify_module(c, 0, 86, census::kernel_names(c, an), an, off).at(0);
    VCHECK(!o.admitted);
    VCHECK_EQ(joined(o), std::string("switch:turbo-off"));
  }
}

VTEST(census_ptx_follows_the_text_scan) {
  struct Case {
    const char* body;
    const char* blocker;   // "" = admitted
  };
  const std::string head = ".visible .entry k(.param .u64 p)\n{\n";
  const Case cases[] = {
      {".visible .entry k(.param .u64 p)\n{\nret;\n}\n", ""},
      {".global .align 4 .u32 v;\n.visible .entry k(.param .u64 p)\n{\nret;\n}\n", "ptx:module-variable"},
      {".visible .entry k(.param .u64 p)\n{\n.reg .f32 f<4>;\ntex.2d.v4.f32.f32 {f0,f1,f2,f3}, [t, {f0,f1}];\nret;\n}\n", "ptx:texture-surface"},
      {".visible .entry k(.param .u64 p)\n{\ntrap;\nret;\n}\n", "ptx:trap"},
      {".visible .entry k(.param .u64 p)\n{\ncp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [a], [b, {c, d}], [e];\nret;\n}\n", "ptx:tma"},
      {".visible .entry k(.param .u64 p)\n{\nwgmma.fence.sync.aligned;\nret;\n}\n", "ptx:warpgroup"},
      {".visible .entry k(.param .u64 p)\n{\nelect.sync %p0|%p1, 0xffffffff;\nret;\n}\n", "ptx:elect"},
  };
  for (const Case& cs : cases) {
    const std::vector<CardCandidate> c{ptx(90, cs.body)};
    // the runtime's scan, called directly, with the retarget flag a plain sm_90 module on an sm_86 card gets
    const hostgpu::PtxScan scan = hostgpu::scan_ptx_for_card(c[0].bytes, false, true);
    const census::KernelCell k = cell(c, -1, 86);
    VCHECK_EQ(std::string(census::image_name(k.image)), std::string("retarget"));
    VCHECK_EQ(k.admitted, scan.refuse.empty());
    VCHECK_EQ(joined(k), std::string(cs.blocker));
    // on an sm_90 card the PTX is not retargeted: the scan has no retarget rules and the image is the PTX JIT
    const hostgpu::PtxScan scan90 = hostgpu::scan_ptx_for_card(c[0].bytes, false, false);
    const census::KernelCell k90 = cell(c, -1, 90);
    VCHECK_EQ(std::string(census::image_name(k90.image)), std::string("ptx-jit"));
    VCHECK_EQ(k90.admitted, scan90.refuse.empty());
  }
  (void)head;
}

VTEST(census_retarget_forms_and_sm90a) {
  // a form ptxas rejects for sm_86 blocks the retarget but not an sm_90 card
  const std::vector<CardCandidate> c{ptx(90, ".visible .entry k(.param .u64 p)\n{\nfence.proxy.async.shared::cta;\nret;\n}\n")};
  VCHECK_EQ(joined(cell(c, -1, 86)), std::string("ptx:retarget-form(fence.proxy.async)"));
  VCHECK(cell(c, -1, 90).admitted);
  // sm_90a PTX is not retargeted at all: no image on sm_86
  CardCandidate a = ptx(90, kPtxOk);
  a.suffix = 'a';
  VCHECK_EQ(joined(cell({a}, -1, 86)), std::string("image:none"));
  // only sm_80..sm_87 hosts retarget (hostgpu::retarget_host)
  VCHECK_EQ(joined(cell({ptx(90, kPtxOk)}, -1, 75)), std::string("image:none"));
}

VTEST(census_cubin_module_variable_rules) {
  // the simulator runs PTX (index -1) and the card is given a cubin with no variables: admitted; the layout rules need variables, which a hand-made cubin lacks
  const std::vector<W> code{iadd3(), exit_()};
  const std::vector<CardCandidate> c{cubin(86, code)};
  VCHECK(cell(c, -1, 86).admitted);
}

VTEST_MAIN
