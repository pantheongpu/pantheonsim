// Fixed-size copies in the simulator's memory paths are not calls into libc's fortified __memcpy_chk.
//
// With glibc's fortification (Ubuntu and Debian compile with _FORTIFY_SOURCE=3 by default) a std::memcpy whose destination is a local object
// at an offset the compiler cannot bound is a call to __memcpy_chk, even at a fixed 4 or 8 bytes. That costs about 10 cycles a call (the
// call, the size check, and a memmove that cannot be inlined); a store of a warp made 32 of them, a parameter read one a launch. The copies on the hot memory paths are
// __builtin_memcpy, which the check does not touch, and the copies whose size is a run-time unit of 1, 2, 4 or 8 are switched to constants.
//
// This test defines __memcpy_chk itself (the static libvgpu.a resolves its calls to it) and counts the calls made while it runs
//   - MemoryManager::store_run / load_run, whose per-word copies were calls (run_in's size is the access width),
//   - a SASS kernel storing 1, 2, 4 and 8 bytes from a whole warp of consecutive lanes (h_global's run buffer, tier_put),
//   - the AMD tier's v_cndmask_b32 (amd/src/gcn_tier.cpp, h_cndmask_body's lane rows), run both ways with the tier's digests compared,
//   - a PTX kernel doing the same through the interpreter (exec_st's run buffer, pack_run; ld.param's micro-op, load_bytes),
// and requires none. The bytes are checked against the values computed here, so a switch with a wrong width fails too. The count is only
// asserted where the compiler fortifies (an optimised build with _FORTIFY_SOURCE set): elsewhere there is nothing to call, and the bytes
// are still checked.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_gcn.hpp"
#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

#if defined(__OPTIMIZE__) && defined(_FORTIFY_SOURCE) && _FORTIFY_SOURCE > 0 && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
#define VGPU_CHECK_CHK_CALLS 1
#else
#define VGPU_CHECK_CHK_CALLS 0
#endif

namespace {
std::atomic<unsigned long> g_chk_calls{0};
}

// What the fortified headers call: counted, then the copy it stands for (the check itself is kept). Only a copy of up to 256 bytes counts:
// the call is what costs in a copy of a few words; a bulk copy (a register file's 64 KB when a wave is set up) dwarfs it.
extern "C" void* __memcpy_chk(void* d, const void* s, size_t n, size_t dl) noexcept {
  if (n <= 256) g_chk_calls.fetch_add(1, std::memory_order_relaxed);
  if (n > dl) std::abort();
  return std::memcpy(d, s, n);
}

namespace {

struct Calls {
  unsigned long start = g_chk_calls.load();
  unsigned long made() const { return g_chk_calls.load() - start; }
};

void expect_no_chk_calls(const Calls& c, const std::string& what) {
#if VGPU_CHECK_CHK_CALLS
  if (c.made() != 0) throw vtest::Failure(what + ": " + std::to_string(c.made()) + " calls to __memcpy_chk");
#else
  (void)c;
  (void)what;
#endif
}

struct Rng {
  uint64_t s;
  uint64_t next() {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return s;
  }
  uint64_t below(uint64_t n) { return next() % n; }
};

// ---- the memory manager's runs -------------------------------------------------------------------------------------------------------

void runs_with_unit(uint32_t unit) {
  MemoryManager mem{1 << 24};
  constexpr uint64_t kSize = 8192;
  const uint64_t base = mem.alloc(kSize);
  Rng rng{0x1234 + unit};
  std::vector<uint8_t> model(kSize);
  for (auto& b : model) b = static_cast<uint8_t>(rng.next());
  mem.write(base, model.data(), kSize);
  const Calls calls;
  for (int round = 0; round < 400; ++round) {
    // (addresses and lengths are multiples of the unit; the start sweeps every phase against the 8-byte boundary)
    const uint64_t off = rng.below(kSize / 2) / unit * unit;
    const uint64_t len = (1 + rng.below(round % 4 == 0 ? 600 : 64)) * unit;
    std::vector<uint8_t> src(len);
    for (auto& b : src) b = static_cast<uint8_t>(rng.next());
    if (!mem.store_run(base + off, src.data(), len, unit)) throw vtest::Failure("store_run declined a plain run");
    std::memcpy(&model[off], src.data(), len);
    std::vector<uint8_t> got(len, 0xEE);
    if (!mem.load_run(base + off, got.data(), len, unit)) throw vtest::Failure("load_run declined a plain run");
    if (std::memcmp(got.data(), &model[off], len) != 0)
      throw vtest::Failure("load_run differs from the model: unit " + std::to_string(unit) + ", offset " + std::to_string(off) +
                           ", length " + std::to_string(len));
  }
  std::vector<uint8_t> all(kSize);
  mem.read(base, all.data(), kSize);
  if (all != model) throw vtest::Failure("memory differs from the model after the runs: unit " + std::to_string(unit));
  expect_no_chk_calls(calls, "store_run / load_run, unit " + std::to_string(unit));
}

// ---- a SASS kernel: STG of 1, 2, 4 and 8 bytes by 32 consecutive lanes -----------------------------------------------------------------

struct Insn {
  sass::Word w;
  std::string text;
};
Insn make(uint64_t hi, uint64_t lo, std::string text = "") {
  Insn i;
  i.w.hi = hi, i.w.lo = lo;
  i.text = std::move(text);
  return i;
}
// lo: opcode | rd << 16 | ra << 24 | rb or immediate << 32.
uint64_t lo_of(uint64_t op, unsigned rd, unsigned ra, uint64_t b) { return op | uint64_t{rd} << 16 | uint64_t{ra} << 24 | b << 32; }
constexpr unsigned kRZ = 255;
Insn imadi(unsigned d, unsigned a, uint32_t imm, unsigned c) { return make(0x000fe400078e0200ull | c, lo_of(0x7824, d, a, imm)); }
Insn imad_wide_i(unsigned d, unsigned a, uint32_t imm, unsigned c) { return make(0x000fe200078e0000ull | c, lo_of(0x7825, d, a, imm)); }
Insn s2r_tid(unsigned d) { return make(0x000e220000002100ull, lo_of(0x7919, d, 0, 0)); }
Insn exit_() { return make(0x000fea0003800000ull, 0x000000000000794dull); }
// STG.E.<width> [Ra.64], Rk: the width is bits 73-75 (4 is 32 bits; 0 is U8, 2 is U16, 5 is 64).
Insn stg(unsigned a, unsigned k, unsigned width_code) {
  return make(0x000fe2000c101904ull + (static_cast<uint64_t>(width_code) - 4) * 0x200, lo_of(0x7986, 0, a, k));
}

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm_86 with one kernel taking one 8-byte pointer (as test_sass_dispatch.cpp builds them).
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
  const uint32_t flags = 0x06000002u | (86u << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

constexpr uint32_t kMulLo = 0x9e3779b1u, kMulHi = 0x85ebca6bu;
constexpr uint32_t kThreads = 32;

std::vector<uint8_t> arg64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

// Thread t stores (R2 = t * kMulLo, R3 = t * kMulHi), `bytes` of them, at out + t * bytes.
std::vector<uint8_t> expected_store(unsigned bytes) {
  std::vector<uint8_t> e(kThreads * bytes);
  for (uint32_t t = 0; t < kThreads; ++t) {
    const uint64_t v = uint64_t{t * kMulLo} | uint64_t{t * kMulHi} << 32;
    std::memcpy(&e[t * bytes], &v, bytes);
  }
  return e;
}

void sass_store(unsigned bytes, unsigned width_code) {
  std::vector<Insn> code;
  code.push_back(s2r_tid(0));
  code.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16));   // MOV R12, c[0x0][0x160]
  code.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16));   // MOV R13, c[0x0][0x164]
  code.push_back(imad_wide_i(14, 0, bytes, 12));   // R14:R15 = out + tid * bytes
  code.push_back(imadi(2, 0, kMulLo, kRZ));
  code.push_back(imadi(3, 0, kMulHi, kRZ));
  code.push_back(stg(14, 2, width_code));
  code.push_back(exit_());
  for (size_t i = 0; i < code.size(); ++i) {
    const sass::Instr ins = sass::decode(code[i].w, 16 * i, 86);
    if (ins.op == sass::Op::Unknown) throw vtest::Failure("word " + std::to_string(i) + " does not decode");
    if (i == 6 && sass::to_text(ins).find("STG") != 0)
      throw vtest::Failure("the store reads as \"" + sass::to_text(ins) + "\", not an STG");
  }
  const std::string cubin = make_cubin(code);
  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu("nvidia/a10");   // sm_86
  auto m = sass::load(reinterpret_cast<const uint8_t*>(cubin.data()), cubin.size(), mem, prof);
  const uint64_t out = mem.alloc(1024);
  const std::vector<uint8_t> poison(1024, 0x5A);
  mem.write(out, poison.data(), poison.size());
  exec::LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {kThreads, 1, 1};
  const Calls calls;
  sass::launch(*m, "k", cfg, {arg64(out)}, mem, prof);
  std::vector<uint8_t> got(1024);
  mem.read(out, got.data(), got.size());
  std::vector<uint8_t> want = poison;
  const std::vector<uint8_t> e = expected_store(bytes);
  std::memcpy(want.data(), e.data(), e.size());
  if (got != want) throw vtest::Failure("the SASS store of " + std::to_string(bytes) + " bytes by 32 lanes wrote other bytes than the lanes' values");
  expect_no_chk_calls(calls, "SASS STG of " + std::to_string(bytes) + " bytes");
}

// ---- a PTX kernel: the same stores through the interpreter ---------------------------------------------------------------------------

void ptx_store(const std::string& ty, unsigned bytes) {
  // (a byte and a half word are stored from a 16-bit register holding the low bits, as nvcc writes them)
  const std::string reg = bytes == 8 ? "%rd7" : bytes == 4 ? "%r7" : "%h1";
  std::string ptx = ".version 8.3\n.target sm_90\n.address_size 64\n.visible .entry k(.param .u64 pout)\n{\n"
                    " .reg .b16 %h<4>; .reg .b32 %r<16>; .reg .b64 %rd<16>;\n"
                    " ld.param.u64 %rd1,[pout]; mov.u32 %r1,%tid.x;\n"
                    " mul.wide.u32 %rd2,%r1," + std::to_string(bytes) + "; add.u64 %rd3,%rd1,%rd2;\n"
                    " mul.lo.u32 %r7,%r1," + std::to_string(kMulLo) + "; mul.lo.u32 %r8,%r1," + std::to_string(kMulHi) + ";\n"
                    " mov.b64 %rd7,{%r7,%r8}; cvt.u16.u32 %h1,%r7;\n"
                    " st.global." + ty + " [%rd3], " + reg + ";\n ret;\n}\n";
  auto m = ptx::parse(ptx);
  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu("nvidia/a10");
  const uint64_t out = mem.alloc(1024);
  const std::vector<uint8_t> poison(1024, 0x5A);
  mem.write(out, poison.data(), poison.size());
  exec::LaunchConfig cfg;
  cfg.block = {kThreads, 1, 1};
  const Calls calls;
  exec::launch(m.entries[0], cfg, {arg64(out)}, mem, prof);
  std::vector<uint8_t> got(1024);
  mem.read(out, got.data(), got.size());
  std::vector<uint8_t> want = poison;
  const std::vector<uint8_t> e = expected_store(bytes);
  std::memcpy(want.data(), e.data(), e.size());
  if (got != want) throw vtest::Failure("the PTX store " + ty + " by 32 lanes wrote other bytes than the lanes' values");
  expect_no_chk_calls(calls, "PTX st.global." + ty);
}

// ---- the AMD tier's v_cndmask_b32 ----------------------------------------------------------------------------------------------------

// Every v_cndmask_b32 of the tier corpus for gfx942, on pseudo-random state with the tier on and off: the same digest, and no fortified call.
void amd_cndmask() {
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/isa_corpus_tier_gfx942.txt", std::ios::binary);
  if (!in) throw vtest::Failure("no tier corpus");
  const bool was_on = amd::gcn::tier::enabled();
  unsigned shapes = 0;
  unsigned long calls = 0;
  for (std::string line; std::getline(in, line);) {
    const size_t tab = line.find('\t');
    if (line.empty() || line[0] == '#' || tab == std::string::npos || line.find("v_cndmask_b32", tab) == std::string::npos) continue;
    std::vector<uint8_t> code;
    std::istringstream words(line.substr(0, tab));
    for (std::string w; words >> w;) {
      const uint32_t v = static_cast<uint32_t>(std::stoul(w, nullptr, 16));
      for (int b = 0; b < 4; ++b) code.push_back(static_cast<uint8_t>(v >> (8 * b)));
    }
    amd::gcn::Inst ins;
    try {
      ins = amd::gcn::decode(code, 0, 0, amd::gcn::Target::Gfx942);
    } catch (const std::exception&) {
      continue;
    }
    if (!ins.fast.run) continue;
    ++shapes;
    for (uint32_t flavour : {0u, 1u, 3u, 11u}) {
      amd::gcn::tier::set_enabled(false);
      const uint64_t off = amd::digest_instruction(ins, amd::gcn::Target::Gfx942, 77 + shapes, flavour);
      amd::gcn::tier::set_enabled(true);
      const Calls c;
      const uint64_t on = amd::digest_instruction(ins, amd::gcn::Target::Gfx942, 77 + shapes, flavour);
      calls += c.made();
      if (off != on) {
        amd::gcn::tier::set_enabled(was_on);
        throw vtest::Failure("v_cndmask_b32 differs with the tier on: " + line.substr(tab + 1));
      }
    }
  }
  amd::gcn::tier::set_enabled(was_on);
  if (shapes < 3) throw vtest::Failure("the corpus has too few v_cndmask_b32 shapes with a tier handler");
#if VGPU_CHECK_CHK_CALLS
  if (calls != 0) throw vtest::Failure("v_cndmask_b32 in the tier: " + std::to_string(calls) + " calls to __memcpy_chk");
#else
  (void)calls;
#endif
}

struct Scoped {
  Scoped() { ::setenv("VGPU_THREADS", "1", 1); }
  ~Scoped() { ::unsetenv("VGPU_THREADS"); }
};

}  // namespace

VTEST(runs_of_the_memory_manager_are_exact_and_make_no_fortified_copy_calls) {
  const Scoped env;
  for (uint32_t unit : {1u, 2u, 4u, 8u}) runs_with_unit(unit);
}

VTEST(a_sass_store_by_consecutive_lanes_writes_the_lanes_values_without_fortified_copy_calls) {
  const Scoped env;
  sass_store(1, 0);   // STG.E.U8
  sass_store(2, 2);   // STG.E.U16
  sass_store(4, 4);   // STG.E
  sass_store(8, 5);   // STG.E.64
}

VTEST(a_ptx_store_by_consecutive_lanes_writes_the_lanes_values_without_fortified_copy_calls) {
  const Scoped env;
  ptx_store("u8", 1);
  ptx_store("u16", 2);
  ptx_store("u32", 4);
  ptx_store("u64", 8);
}

VTEST(the_amd_tiers_v_cndmask_gives_the_interpreters_digest_without_fortified_copy_calls) {
  const Scoped env;
  amd_cndmask();
}

VTEST_MAIN
