// ELECT in the two spellings of ptxas (round 12, ci124): `elect.sync _|%p, 0xffffffff` assembles for sm_90 to `ELECT P0, URZ, PT` (the predicate form, nvcc 12.8
// and 13.0: measured) and to `ELECT P0, URZ, ~URZ` (nvcc 12.4.131: measured, a uniform-register operand where the predicate was; bit 72 of the word is set). The
// decoder once read the second as a predicate source naming P0, which is false, so no thread was elected and a kernel that took `elect.sync` for its leader
// (host_gpu_fit.cu `elect`) printed sum=0 in the plain simulator on the CUDA 12.4 toolchain only.
//
// The kernels are the SASS of that kernel written out word by word, for each form, and run on a simulated H100 (no nvcc is needed, so the test runs the same on every
// toolchain). Exactly one thread of each warp is elected, the lowest. The text of each word is checked against nvdisasm's.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

struct Insn {
  sass::Word w;
  std::string text;
};

Insn make(uint64_t hi, uint64_t lo, const char* text) {
  Insn i;
  i.w.hi = hi, i.w.lo = lo;
  i.text = text;
  return i;
}

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm_90 with one kernel taking one 8-byte pointer (the layout of test_sass_store_gate, with the architecture 90).
std::string make_cubin(const std::vector<Insn>& code) {
  const std::string names("\0.shstrtab\0.text.k\0.nv.info.k\0", 31);
  std::string info;   // KPARAM_INFO: index, ordinal | offset << 16, size << 18
  put<uint8_t>(info, 4), put<uint8_t>(info, 0x17), put<uint16_t>(info, 12);
  put<uint32_t>(info, 0), put<uint32_t>(info, 0), put<uint32_t>(info, 8u << 18);
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  while (out.size() % 16) out.push_back('\0');
  const uint64_t text_off = out.size();
  for (const Insn& i : code) put(out, i.w.lo), put(out, i.w.hi);
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
  const uint32_t flags = 0x06000002u | (90u << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

// host_gpu_fit.cu `elect` for sm_90 (nvcc 13.0 and 12.4.131; the control bits of each word are the compiler's), with the ELECT word and its text as given: the thread
// stores 1 if it was elected and 0 if not.
std::vector<Insn> kernel(const Insn& elect) {
  return {
      make(0x000fe20000000800ull, 0x00000a00ff017b82ull, "LDC R1, c[0x0][0x28]"),
      make(0x000e2e0000002100ull, 0x0000000000057919ull, "S2R R5, SR_TID.X"),
      make(0x000e220000000a00ull, 0x00008400ff027b82ull, "LDC.64 R2, c[0x0][0x210]"),
      elect,
      make(0x000fe20000000a00ull, 0x0000820000047ab9ull, "ULDC.64 UR4, c[0x0][0x208]"),
      make(0x001fc400078e0002ull, 0x0000000405027825ull, "IMAD.WIDE.U32 R2, R5, 0x4, R2"),
      make(0x000fca0004000000ull, 0x00000001ff057807ull, "SEL R5, RZ, 0x1, !P0"),
      make(0x000fe2000c101904ull, 0x0000000502007986ull, "STG.E desc[UR4][R2.64], R5"),
      make(0x000fea0003800000ull, 0x000000000000794dull, "EXIT"),
  };
}

const Insn kPredForm = make(0x000fe20003800000ull, 0x00000000003f782full, "ELECT P0, URZ, PT");      // nvcc 12.8, 13.0 (measured)
const Insn kMaskForm = make(0x000fe20008000100ull, 0x000000003f3f782full, "ELECT P0, URZ, ~URZ");    // nvcc 12.4.131 (measured)
const Insn kMaskFormUr = make(0x000fe20008000100ull, 0x000000003f04782full, "ELECT P0, UR4, ~URZ");  // nvcc 12.4.131 with the lane wanted (measured)

// What `threads` threads store: element t is 1 for the elected thread of its warp.
std::vector<uint32_t> run(const std::vector<Insn>& code, uint32_t threads) {
  ::setenv("VGPU_THREADS", "1", 1);
  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu("nvidia/h100");   // sm_90
  for (size_t i = 0; i < code.size(); ++i) {
    const sass::Instr ins = sass::decode(code[i].w, 16 * i, 90);
    if (sass::to_text(ins) != code[i].text)
      throw vtest::Failure("word " + std::to_string(i) + " is \"" + sass::to_text(ins) + "\", not \"" + code[i].text + "\"");
  }
  const std::string c = make_cubin(code);
  const auto m = sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), mem, prof);
  const uint64_t out = mem.alloc(4 * threads);
  std::vector<uint32_t> init(threads, 7);
  mem.write(out, init.data(), 4 * threads);
  exec::LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {threads, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &out, 8);
  sass::launch(*m, "k", cfg, {arg}, mem, prof);
  std::vector<uint32_t> v(threads);
  mem.read(out, v.data(), 4 * threads);
  mem.free(out);
  ::unsetenv("VGPU_THREADS");
  return v;
}

void expect_lowest_lane_of_each_warp(const std::vector<Insn>& code, const char* what) {
  for (uint32_t threads : {32u, 70u}) {
    const std::vector<uint32_t> v = run(code, threads);
    for (uint32_t t = 0; t < threads; ++t)
      if (v[t] != (t % 32 == 0 ? 1u : 0u))
        throw vtest::Failure(std::string(what) + ", " + std::to_string(threads) + " threads: thread " + std::to_string(t) + " stored " + std::to_string(v[t]) +
                             " (the lowest thread of each warp is the one elected)");
  }
}

}  // namespace

VTEST(the_predicate_form_elects_the_lowest_thread_of_each_warp) { expect_lowest_lane_of_each_warp(kernel(kPredForm), "ELECT P0, URZ, PT"); }

VTEST(the_register_mask_form_of_ptxas_12_4_elects_the_lowest_thread_of_each_warp) {
  expect_lowest_lane_of_each_warp(kernel(kMaskForm), "ELECT P0, URZ, ~URZ");
  expect_lowest_lane_of_each_warp(kernel(kMaskFormUr), "ELECT P0, UR4, ~URZ");
}

VTEST(the_two_forms_decode_to_what_nvdisasm_prints) {
  // (a decoder that took the register form for a predicate source printed `ELECT P0, URZ, P0`)
  const sass::Instr a = sass::decode(kPredForm.w, 0, 90), b = sass::decode(kMaskForm.w, 0, 90);
  VCHECK(sass::to_text(a) == "ELECT P0, URZ, PT");
  VCHECK(sass::to_text(b) == "ELECT P0, URZ, ~URZ");
  VCHECK(b.src[0].kind == sass::Kind::UReg && b.src[0].bnot);
  VCHECK(a.src[0].kind == sass::Kind::Pred);
}

VTEST_MAIN
