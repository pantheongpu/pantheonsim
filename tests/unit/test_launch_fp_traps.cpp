// A host program that enabled floating-point traps (feenableexcept; gfortran -ffpe-trap=invalid,zero,overflow does it at
// program start) launches kernels that make infs and NaNs. On a GPU that is just arithmetic: an RTX 3060 runs pantheonsim
// #605's repro and prints the inf and the NaN with the traps on or off. The engines did the arithmetic with the host's
// instructions under the host's traps, on the launching thread and on worker threads that copy its environment, and the
// process died of SIGFPE. Each launch here runs with the three traps on: the PTX engine on one thread and on several, and
// the SASS executor. The kernel's values are the IEEE ones, and afterwards the host has its traps, its rounding mode and
// its sticky flags exactly as before (none the kernel raised).
//
// Without the fix this program is killed by SIGFPE, which CTest reports as a failure.
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
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

using namespace vgpu;

namespace {

constexpr int kTraps = FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW;

std::vector<uint8_t> bytes_of(const void* p, size_t n) {
  std::vector<uint8_t> b(n);
  std::memcpy(b.data(), p, n);
  return b;
}

// Runs `body` with the traps on and FE_INEXACT raised, and checks that the host's environment is the same afterwards.
template <class F>
void with_traps(F&& body) {
  std::feclearexcept(FE_ALL_EXCEPT);
  std::feraiseexcept(FE_INEXACT);   // (not trapped) a flag the host had before the launch, which it must still have after
  const int round = std::fegetround();
  feenableexcept(kTraps);
  std::string err;
  try {
    body();
  } catch (const std::exception& e) {
    err = e.what();
  }
  const int traps_after = fegetexcept();
  const int flags_after = std::fetestexcept(FE_ALL_EXCEPT);
  const int round_after = std::fegetround();
  fedisableexcept(FE_ALL_EXCEPT);
  std::feclearexcept(FE_ALL_EXCEPT);
  if (!err.empty()) throw vtest::Failure(err);
  VCHECK_EQ(traps_after, kTraps);
  VCHECK_EQ(flags_after, FE_INEXACT);
  VCHECK_EQ(round_after, round);
}

// out[i] for i = 4 * thread + j: 1/x, sqrt(x - 1), x * 1e38 and x - x * inf, with x = 0 for odd threads and 4 for even.
const char* const kPtx = R"(.version 8.0
.target sm_86
.address_size 64
.visible .entry k(.param .u64 po)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<12>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [po];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r4, %r2, %r3, %r1;
    and.b32 %r5, %r4, 1;
    setp.eq.u32 %p1, %r5, 0;
    selp.f32 %f1, 0f40800000, 0f00000000, %p1;
    rcp.rn.f32 %f2, %f1;
    div.rn.f32 %f9, 0f3F800000, %f1;
    add.f32 %f2, %f2, %f9;
    sub.f32 %f3, %f1, 0f3F800000;
    sqrt.rn.f32 %f4, %f3;
    mul.f32 %f5, %f1, 0f7E967699;
    mul.f32 %f5, %f5, 0f7E967699;
    mul.f32 %f6, %f1, 0f7F800000;
    sub.f32 %f7, %f6, %f6;
    mul.wide.u32 %rd2, %r4, 16;
    add.u64 %rd3, %rd1, %rd2;
    st.global.v4.f32 [%rd3], {%f2, %f4, %f5, %f7};
    ret;
}
)";

void check_ptx_values(const std::vector<float>& o, unsigned threads) {
  const float inf = std::numeric_limits<float>::infinity();
  for (unsigned t = 0; t < threads; ++t) {
    const float* v = &o[4 * t];
    if (t % 2 == 0) {   // x = 4
      VCHECK_EQ(v[0], 0.5f);
      VCHECK_EQ(v[1], std::sqrt(3.0f));
      VCHECK_EQ(v[2], inf);
      VCHECK(std::isnan(v[3]));   // inf - inf
    } else {            // x = 0
      VCHECK_EQ(v[0], inf);
      VCHECK(std::isnan(v[1]));   // sqrt(-1)
      VCHECK_EQ(v[2], 0.0f);
      VCHECK(std::isnan(v[3]));   // 0 * inf
    }
  }
}

void run_ptx(const char* threads) {
  setenv("VGPU_THREADS", threads, 1);
  MemoryManager mem{1 << 24};
  const DeviceProfile prof = load_gpu("nvidia/a10");
  auto m = ptx::parse(kPtx);
  constexpr unsigned kBlocks = 8, kBlock = 64, kThreads = kBlocks * kBlock;
  const uint64_t out = mem.alloc(kThreads * 16);
  exec::LaunchConfig cfg;
  cfg.grid = {kBlocks, 1, 1};
  cfg.block = {kBlock, 1, 1};
  with_traps([&] { exec::launch(m.entries[0], cfg, {bytes_of(&out, 8)}, mem, prof); });
  std::vector<float> o(kThreads * 4);
  mem.read(out, o.data(), o.size() * 4);
  unsetenv("VGPU_THREADS");
  check_ptx_values(o, kThreads);
}

// ---- SASS: k(float* o, float a, float b) { *o = a - b; }, the words nvcc 13.0 -arch=sm_86 emits for it ----

struct Word {
  uint64_t lo, hi;
};
const Word kSass[] = {
    {0x00000a0000017a02ull, 0x000fe40000000f00ull},   // MOV R1, c[0x0][0x28]
    {0x00005a0000057a02ull, 0x000fe20000000f00ull},   // MOV R5, c[0x0][0x168]
    {0x0000460000047ab9ull, 0x000fe20000000a00ull},   // ULDC.64 UR4, c[0x0][0x118]
    {0x0000580000027a02ull, 0x000fe40000000f00ull},   // MOV R2, c[0x0][0x160]
    {0x0000590000037a02ull, 0x000fe20000000f00ull},   // MOV R3, c[0x0][0x164]
    {0x80005b0005057621ull, 0x000fca0000000000ull},   // FADD R5, R5, -c[0x0][0x16c]
    {0x0000000502007986ull, 0x000fe2000c101904ull},   // STG.E [R2.64], R5
    {0x000000000000794dull, 0x000fea0003800000ull},   // EXIT
    {0xfffffff000007947ull, 0x000fc0000383ffffull},   // BRA 0x80
};

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm_86 whose kernel k takes (8-byte pointer, float, float).
std::string make_cubin() {
  const std::string names("\0.shstrtab\0.text.k\0.nv.info.k\0", 30);
  std::string info;   // KPARAM_INFO for each parameter: index, ordinal | offset << 16, size << 18
  const uint32_t params[3][3] = {{0, 0, 8}, {1, 8, 4}, {2, 12, 4}};
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

float run_sass(float a, float b) {
  MemoryManager mem{1 << 24};
  const DeviceProfile prof = load_gpu("nvidia/a10");   // sm_86
  const std::string c = make_cubin();
  std::shared_ptr<sass::Module> m = sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), mem, prof);
  const uint64_t out = mem.alloc(4);
  exec::LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {32, 1, 1};
  with_traps([&] { sass::launch(*m, "k", cfg, {bytes_of(&out, 8), bytes_of(&a, 4), bytes_of(&b, 4)}, mem, prof); });
  float r = 0;
  mem.read(out, &r, 4);
  return r;
}

}  // namespace

VTEST(ptx_kernel_under_host_fp_traps_one_thread) { run_ptx("1"); }

VTEST(ptx_kernel_under_host_fp_traps_worker_threads) { run_ptx("4"); }

VTEST(sass_kernel_under_host_fp_traps) {
  const float inf = std::numeric_limits<float>::infinity();
  VCHECK(std::isnan(run_sass(inf, inf)));             // invalid
  VCHECK_EQ(run_sass(3e38f, -3e38f), inf);            // overflow
  VCHECK_EQ(run_sass(1.5f, 0.25f), 1.25f);            // and an ordinary difference is still one
}

// A launch that fails still gives the host its environment back.
VTEST(failed_launch_restores_host_fp_environment) {
  MemoryManager mem{1 << 24};
  const DeviceProfile prof = load_gpu("nvidia/a10");
  auto m = ptx::parse(kPtx);
  exec::LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {64, 1, 1};
  const uint64_t bad = 0x10;   // no memory there: the store faults
  std::feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(kTraps);
  bool threw = false;
  try {
    exec::launch(m.entries[0], cfg, {bytes_of(&bad, 8)}, mem, prof);
  } catch (const std::exception&) {
    threw = true;
  }
  const int traps_after = fegetexcept();
  fedisableexcept(FE_ALL_EXCEPT);
  std::feclearexcept(FE_ALL_EXCEPT);
  VCHECK(threw);
  VCHECK_EQ(traps_after, kTraps);
}

VTEST_MAIN
