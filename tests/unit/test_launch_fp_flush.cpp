// A host program that turned on the x86 flush-to-zero controls for itself (MXCSR FTZ and DAZ: nvc -fast does it at program start,
// gcc and clang do with -ffast-math) launches a kernel that does double arithmetic with subnormals. A GPU does that arithmetic
// with the subnormals as numbers, and the simulator did it with the host's instructions under the host's controls, so a subnormal
// operand read as zero and a subnormal result was written as zero. NVHPC 26.9's fmod reads the exponent of a subnormal remainder
// through a multiply by 2^54, got a zero, and returned garbage on the simulated H100 (pantheonsim #672: fmod(1.5, 1) was 1.4e-17).
//
// Each launch here runs with FTZ and DAZ on in the host: the PTX engine on one thread and on several, and the SASS executor. The
// kernel's values are the IEEE ones, and afterwards the host has its controls on, as before. A kernel that asks for .ftz still
// flushes (that is the engine's own rule, not the host's).
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

using namespace vgpu;

namespace {

constexpr uint64_t kSub = 0x0008000000000000ull;        // 2^-1023, a subnormal
constexpr uint64_t kTwo54 = 0x4350000000000000ull;      // 2^54
constexpr uint64_t kMinNormal = 0x0010000000000000ull;  // 2^-1022
constexpr uint64_t kHalf = 0x3fe0000000000000ull;       // 0.5
constexpr uint64_t kSub54 = 0x0360000000000000ull;      // 2^-969: kSub * 2^54

std::vector<uint8_t> bytes_of(const void* p, size_t n) {
  std::vector<uint8_t> b(n);
  std::memcpy(b.data(), p, n);
  return b;
}

#if defined(__x86_64__)
constexpr unsigned kFlush = 0x8000u | 0x0040u;   // FTZ | DAZ

// Runs `body` with FTZ and DAZ on in the host, and checks that they are still on afterwards (and puts the old state back).
template <class F>
void with_flush(F&& body) {
  const unsigned saved = _mm_getcsr();
  _mm_setcsr(saved | kFlush);
  std::string err;
  try {
    body();
  } catch (const std::exception& e) {
    err = e.what();
  }
  const unsigned after = _mm_getcsr();
  _mm_setcsr(saved);
  if (!err.empty()) throw vtest::Failure(err);
  VCHECK_EQ(after & kFlush, kFlush);
}
#else
template <class F>
void with_flush(F&& body) {
  body();   // (no such control to turn on here: the values are the same)
}
#endif

// out[0] = 2^-1023 * 2^54 (a subnormal operand), out[1] = 2^-1023 + 2^-1023 (two), out[2] = 2^-1022 * 0.5 (a subnormal result),
// out[3] = fma(2^-1023, 2^54, 0), and out[4] = the same multiply with mul.ftz.f32 on a float subnormal, which does flush.
const char* const kPtx = R"(.version 8.0
.target sm_86
.address_size 64
.visible .entry k(.param .u64 po)
{
    .reg .b64 %rd<4>;
    .reg .f64 %fd<8>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [po];
    mov.f64 %fd1, 0d0008000000000000;
    mov.f64 %fd2, 0d4350000000000000;
    mov.f64 %fd3, 0d0010000000000000;
    mov.f64 %fd4, 0d3fe0000000000000;
    mul.rn.f64 %fd5, %fd1, %fd2;
    add.rn.f64 %fd6, %fd1, %fd1;
    mul.rn.f64 %fd7, %fd3, %fd4;
    st.global.f64 [%rd1], %fd5;
    st.global.f64 [%rd1+8], %fd6;
    st.global.f64 [%rd1+16], %fd7;
    fma.rn.f64 %fd5, %fd1, %fd2, 0d0000000000000000;
    st.global.f64 [%rd1+24], %fd5;
    mov.f32 %f1, 0f00400000;
    mul.ftz.f32 %f2, %f1, 0f3f800000;
    st.global.f32 [%rd1+32], %f2;
    ret;
}
)";

void run_ptx(const char* threads) {
  setenv("VGPU_THREADS", threads, 1);
  MemoryManager mem{1 << 24};
  const DeviceProfile prof = load_gpu("nvidia/a10");
  auto m = ptx::parse(kPtx);
  exec::LaunchConfig cfg;
  cfg.grid = {8, 1, 1};
  cfg.block = {1, 1, 1};
  const uint64_t out = mem.alloc(64);
  uint64_t zero[8] = {};
  mem.write(out, zero, sizeof zero);
  with_flush([&] { exec::launch(m.entries[0], cfg, {bytes_of(&out, 8)}, mem, prof); });
  unsetenv("VGPU_THREADS");
  uint64_t o[8];
  mem.read(out, o, sizeof o);
  VCHECK_EQ(o[0], kSub54);
  VCHECK_EQ(o[1], kMinNormal);
  VCHECK_EQ(o[2], kSub);
  VCHECK_EQ(o[3], kSub54);
  VCHECK_EQ(static_cast<uint32_t>(o[4]), 0u);   // the .ftz multiply flushes the float subnormal, as asked
}

// ---- SASS: k(double* o, double a, double b) { *o = a * b; }, the words nvcc 13.0 -arch=sm_86 emits for it ----

struct Word {
  uint64_t lo, hi;
};
const Word kSass[] = {
    {0x00000a0000017a02ull, 0x000fe40000000f00ull},   // MOV R1, c[0x0][0x28]
    {0x00005a0000027a02ull, 0x000fe20000000f00ull},   // MOV R2, c[0x0][0x168]
    {0x00005b00ff037624ull, 0x000fe200078e00ffull},   // IMAD.MOV.U32 R3, RZ, RZ, c[0x0][0x16c]
    {0x0000580000047a02ull, 0x000fe20000000f00ull},   // MOV R4, c[0x0][0x160]
    {0x00005900ff057624ull, 0x000fe200078e00ffull},   // IMAD.MOV.U32 R5, RZ, RZ, c[0x0][0x164]
    {0x0000460000047ab9ull, 0x000fc60000000a00ull},   // ULDC.64 UR4, c[0x0][0x118]
    {0x00005c0002027a28ull, 0x000e0e0000000000ull},   // DMUL R2, R2, c[0x0][0x170]
    {0x0000000204007986ull, 0x001fe2000c101b04ull},   // STG.E.64 [R4.64], R2
    {0x000000000000794dull, 0x000fea0003800000ull},   // EXIT
    {0xfffffff000007947ull, 0x000fc0000383ffffull},   // BRA 0x90
};

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm_86 whose kernel k takes (8-byte pointer, double, double).
std::string make_cubin() {
  const std::string names("\0.shstrtab\0.text.k\0.nv.info.k\0", 30);
  std::string info;   // KPARAM_INFO for each parameter: index, ordinal | offset << 16, size << 18
  const uint32_t params[3][3] = {{0, 0, 8}, {1, 8, 8}, {2, 16, 8}};
  for (const auto& p : params) {
    put<uint8_t>(info, 4), put<uint8_t>(info, 0x17), put<uint16_t>(info, 12);
    put<uint32_t>(info, 0), put<uint32_t>(info, p[0] | p[1] << 16), put<uint32_t>(info, p[2] << 18);
  }
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  while (out.size() % 16) out.push_back('\0');
  const uint64_t text_off = out.size();
  for (const Word& w : kSass) put(out, w.lo), put(out, w.hi);
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
  shdr(11, 1, text_off, 16 * std::size(kSass));
  shdr(19, 0x70000000, info_off, info.size());
  out[0] = 0x7f, out[1] = 'E', out[2] = 'L', out[3] = 'F', out[4] = 2, out[5] = 1, out[6] = 1, out[7] = 0x41, out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = 4, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (86u << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

uint64_t run_sass(uint64_t a, uint64_t b) {
  MemoryManager mem{1 << 24};
  const DeviceProfile prof = load_gpu("nvidia/a10");   // sm_86
  const std::string c = make_cubin();
  std::shared_ptr<sass::Module> m = sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), mem, prof);
  const uint64_t out = mem.alloc(8);
  exec::LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {32, 1, 1};
  with_flush([&] { sass::launch(*m, "k", cfg, {bytes_of(&out, 8), bytes_of(&a, 8), bytes_of(&b, 8)}, mem, prof); });
  uint64_t r = 0;
  mem.read(out, &r, 8);
  return r;
}

}  // namespace

VTEST(ptx_double_subnormals_under_host_flush_one_thread) { run_ptx("1"); }

VTEST(ptx_double_subnormals_under_host_flush_worker_threads) { run_ptx("4"); }

VTEST(sass_dmul_subnormals_under_host_flush) {
  VCHECK_EQ(run_sass(kSub, kTwo54), kSub54);        // a subnormal operand
  VCHECK_EQ(run_sass(kMinNormal, kHalf), kSub);     // a subnormal result
  VCHECK_EQ(run_sass(kHalf, kTwo54), 0x4340000000000000ull);   // and an ordinary product is still one
}

VTEST_MAIN
