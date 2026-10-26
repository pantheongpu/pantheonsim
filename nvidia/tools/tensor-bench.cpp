// What one tensor-core instruction costs the simulator: warp instructions of the PTX engine (mma.sync, wmma.mma, wgmma.mma_async,
// tcgen05.mma) and of the SASS executor (HMMA, IMMA, QMMA, ... words of the disassembler corpora in a hand-built cubin), each timed in
// a loop of R and of 2R instructions so the difference is the cost of R (launch, parse and set-up drop out), and reported as ns and
// cycles (at the CPU's measured clock) per warp instruction.
//
//   nvidia_tensor_bench [filter] [repeats]      filter: a substring of the row's name (default: every row)
//
// Operands are random normal numbers (exponents near 1), so the fast routes' common case is what is timed, not their fall-backs.
// Run it pinned to one CPU of a core (taskset -c N), the other thread of the core idle.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

using Clock = std::chrono::steady_clock;

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

// CPU cycles a nanosecond, from a calibration loop with a known dependent-add latency (1 cycle each).
double ghz() {
  static const double g = [] {
    uint64_t x = 1;
    const auto t0 = Clock::now();
    for (uint64_t i = 0; i < 400000000ull; ++i) asm volatile("add $1, %0" : "+r"(x));
    const double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    return 400000000.0 / ns;
  }();
  return g;
}

std::mt19937 rng(0x7e5);
uint32_t f32_normal() {
  const uint32_t h = rng();
  return (h & 0x80000000u) | ((120 + (h >> 8) % 12) << 23) | (rng() & 0x7fffffu);
}
uint16_t f16_normal() {
  const uint32_t h = rng();
  return static_cast<uint16_t>((h & 0x8000u) | ((11 + (h >> 8) % 8) << 10) | (rng() & 0x3ffu));
}
uint32_t word_of(const std::string& kind) {
  if (kind == "f16") return f16_normal() | uint32_t{f16_normal()} << 16;
  if (kind == "bf16") return (f32_normal() >> 16) | (f32_normal() & 0xffff0000u);
  if (kind == "f32") return f32_normal();
  return rng();   // integers, fp8 (any byte is a number except a few NaN codes; fine for timing)
}

// ---- PTX: mma.sync and wmma.mma ----------------------------------------------------------------------------------------------

struct Reg {
  const char* name;
  const char* gpu;
  const char* target;
  std::string instr;   // the mnemonic through the types, then the operands are appended
  int a, b, c, d;      // registers a lane holds of A, B, C and D
  const char* data;    // "f16", "bf16", "f32" (tf32), "int" or "fp8": how A and B are drawn
  const char* cdata;   // how C is drawn: "f32", "f16" or "int"
  bool sparse = false;
  bool wmma = false;   // register lists need no fixed 4/2/4 shape: as given
  bool wide = false;   // 64-bit registers
  const char* suffix = "";
};

const Reg kRegs[] = {
    // HMMA family on an RTX A10 (sm_86: the card-measured chunked sum) and an H100 (the plain f32 tile sum)
    {"ptx mma f16 m16n8k8 f32 (a10)", "nvidia/a10", "sm_86", "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32", 2, 1, 4, 4, "f16", "f32"},
    {"ptx mma f16 m16n8k16 f32 (a10)", "nvidia/a10", "sm_86", "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", 4, 2, 4, 4, "f16", "f32"},
    {"ptx mma bf16 m16n8k8 f32 (a10)", "nvidia/a10", "sm_86", "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32", 2, 1, 4, 4, "bf16", "f32"},
    {"ptx mma bf16 m16n8k16 f32 (a10)", "nvidia/a10", "sm_86", "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", 4, 2, 4, 4, "bf16", "f32"},
    {"ptx mma tf32 m16n8k8 f32 (a10)", "nvidia/a10", "sm_86", "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", 4, 2, 4, 4, "f32", "f32"},
    {"ptx mma tf32 m16n8k4 f32 (a10)", "nvidia/a10", "sm_86", "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32", 2, 1, 4, 4, "f32", "f32"},
    {"ptx mma f16 m16n8k16 f16 (a10)", "nvidia/a10", "sm_86", "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16", 4, 2, 2, 2, "f16", "f16"},
    {"ptx mma f16 m16n8k16 f32 (h100)", "nvidia/h100", "sm_90", "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", 4, 2, 4, 4, "f16", "f32"},
    {"ptx mma bf16 m16n8k16 f32 (h100)", "nvidia/h100", "sm_90", "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", 4, 2, 4, 4, "bf16", "f32"},
    {"ptx mma tf32 m16n8k8 f32 (h100)", "nvidia/h100", "sm_90", "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", 4, 2, 4, 4, "f32", "f32"},
    {"ptx mma s8 m16n8k16 (a100)", "nvidia/a100", "sm_80", "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32", 2, 1, 4, 4, "int", "int"},
    {"ptx mma s8 m16n8k32 (a100)", "nvidia/a100", "sm_80", "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", 4, 2, 4, 4, "int", "int"},
    {"ptx mma s8 m16n8k32 sat (a100)", "nvidia/a100", "sm_80", "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32", 4, 2, 4, 4, "int", "int"},
    {"ptx mma s4 m16n8k64 (a100)", "nvidia/a100", "sm_80", "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32", 4, 2, 4, 4, "int", "int"},
    {"ptx mma e4m3 m16n8k32 f32 (4090)", "nvidia/rtx4090", "sm_89", "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", 4, 2, 4, 4, "fp8", "f32"},
    {"ptx mma f16 sparse m16n8k32 (a100)", "nvidia/a100", "sm_80", "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32", 4, 4, 4, 4, "f16", "f32", true},
    {"ptx mma s8 sparse m16n8k64 (a100)", "nvidia/a100", "sm_80", "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32", 4, 4, 4, 4, "int", "int", true},
    {"ptx mma f64 m8n8k4 (a100)", "nvidia/a100", "sm_80", "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64", 1, 1, 2, 2, "f32", "f32", false, false, true, ".rn"},
    {"ptx mma f16 m8n8k4 f32 (a10)", "nvidia/a10", "sm_86", "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32", 2, 2, 8, 8, "f16", "f32"},
    {"ptx wmma f16 m16n16k16 f32 (a10)", "nvidia/a10", "sm_86", "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32", 8, 8, 8, 8, "f16", "f32", false, true},
    {"ptx wmma f16 m32n8k16 f32 (a10)", "nvidia/a10", "sm_86", "wmma.mma.sync.aligned.row.col.m32n8k16.f32.f32", 8, 8, 8, 8, "f16", "f32", false, true},
    {"ptx wmma f16 m8n32k16 f32 (a10)", "nvidia/a10", "sm_86", "wmma.mma.sync.aligned.row.col.m8n32k16.f32.f32", 8, 8, 8, 8, "f16", "f32", false, true},
    {"ptx wmma f16 m16n16k16 f32 (h100)", "nvidia/h100", "sm_90", "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32", 8, 8, 8, 8, "f16", "f32", false, true},
    {"ptx wmma bf16 m16n16k16 f32 (a10)", "nvidia/a10", "sm_86", "wmma.mma.sync.aligned.row.col.m16n16k16.f32.bf16.bf16.f32", 4, 4, 8, 8, "bf16", "f32", false, true},
    {"ptx wmma tf32 m16n16k8 f32 (a10)", "nvidia/a10", "sm_86", "wmma.mma.sync.aligned.row.col.m16n16k8.f32.tf32.tf32.f32", 4, 4, 8, 8, "f32", "f32", false, true},
    {"ptx wmma s8 m16n16k16 (a100)", "nvidia/a100", "sm_80", "wmma.mma.sync.aligned.row.col.m16n16k16.s32.s8.s8.s32", 2, 2, 8, 8, "int", "int", false, true},
    {"ptx wmma f64 m8n8k4 (a100)", "nvidia/a100", "sm_80", "wmma.mma.sync.aligned.row.col.m8n8k4.f64.f64.f64.f64", 1, 1, 2, 2, "f32", "f32", false, true, true, ".rn"},
};

std::string reglist(const char* p, int n) {
  std::string s = "{";
  for (int i = 0; i < n; ++i) s += std::string(i ? "," : "") + "%" + p + std::to_string(i);
  return s + "}";
}

// Seconds of one launch of the form with `reps` instructions in a loop.
double run_reg(const Reg& f, int reps) {
  const char* rt = f.wide ? "b64" : "b32";
  const int w = f.wide ? 8 : 4;
  std::string ptx = std::string(".version 8.3\n.target ") + f.target + R"(
.address_size 64
.visible .entry k(.param .u64 pin)
{
    .reg .pred %again;
    .reg .b32 %r<8>;
    .reg .b32 %meta;
    .reg .)" + rt + R"( %a<16>, %b<16>, %c<16>, %d<16>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [pin];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 256;
    add.u64 %rd4, %rd1, %rd3;
)";
  int off = 0;
  for (int i = 0; i < f.a; ++i, off += w) ptx += std::string("    ld.global.") + rt + " %a" + std::to_string(i) + ", [%rd4+" + std::to_string(off) + "];\n";
  for (int i = 0; i < f.b; ++i, off += w) ptx += std::string("    ld.global.") + rt + " %b" + std::to_string(i) + ", [%rd4+" + std::to_string(off) + "];\n";
  for (int i = 0; i < f.c; ++i, off += w) ptx += std::string("    ld.global.") + rt + " %c" + std::to_string(i) + ", [%rd4+" + std::to_string(off) + "];\n";
  ptx += "    ld.global.b32 %meta, [%rd4+" + std::to_string(off) + "];\n    mov.u32 %r3, 0;\nLOOP:\n";
  ptx += "    " + f.instr + f.suffix + " " + reglist("d", f.d) + ", " + reglist("a", f.a) + ", " + reglist("b", f.b) + ", " + reglist("c", f.c) +
         (f.sparse ? ", %meta, 0" : "") + ";\n";
  ptx += "    add.u32 %r3, %r3, 1;\n    setp.lt.u32 %again, %r3, " + std::to_string(reps) + ";\n    @%again bra LOOP;\n";
  // keep D live
  ptx += std::string("    st.global.") + rt + " [%rd4], %d0;\n    ret;\n}\n";
  MemoryManager mem{1 << 22};
  DeviceProfile prof = load_gpu(f.gpu);
  auto m = ptx::parse(ptx);
  std::vector<uint32_t> in(32 * 64);
  for (int l = 0; l < 32; ++l) {
    uint32_t* wd = &in[l * 64];
    int at = 0;
    const auto fill = [&](int n, const std::string& kind) {
      for (int i = 0; i < n; ++i) {
        if (f.wide) {
          const double v = (static_cast<double>(rng() % 2000) - 1000) / 77.0;
          std::memcpy(&wd[at], &v, 8);
          at += 2;
        } else {
          wd[at++] = word_of(kind);
        }
      }
    };
    fill(f.a, f.data);
    fill(f.b, f.data);
    fill(f.c, f.cdata);
    wd[at] = rng();
  }
  const uint64_t pin = mem.alloc(in.size() * 4);
  mem.write(pin, in.data(), in.size() * 4);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  const auto t0 = Clock::now();
  exec::launch(m.entries[0], cfg, {arg_u64(pin)}, mem, prof);
  return std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
}

// ---- PTX: wgmma ---------------------------------------------------------------------------------------------------------------

struct Wg {
  const char* name;
  const char* shape;   // after wgmma.mma_async.sync.aligned.
  uint32_t N, dregs;
  const char* tail;    // scale and transpose operands
  int eb;              // bytes an element
  bool int_acc;
};
const Wg kWgs[] = {
    {"ptx wgmma f16 m64n128k16 f32", "m64n128k16.f32.f16.f16", 128, 64, ", 1, 1, 0, 0", 2, false},
    {"ptx wgmma f16 m64n256k16 f32", "m64n256k16.f32.f16.f16", 256, 128, ", 1, 1, 0, 0", 2, false},
    {"ptx wgmma f16 m64n128k16 f16", "m64n128k16.f16.f16.f16", 128, 32, ", 1, 1, 0, 0", 2, false},
    {"ptx wgmma bf16 m64n128k16 f32", "m64n128k16.f32.bf16.bf16", 128, 64, ", 1, 1, 0, 0", 2, false},
    {"ptx wgmma tf32 m64n128k8 f32", "m64n128k8.f32.tf32.tf32", 128, 64, ", 1, 1", 4, false},
    {"ptx wgmma e4m3 m64n128k32 f32", "m64n128k32.f32.e4m3.e4m3", 128, 64, ", 1, 1", 1, false},
    {"ptx wgmma s8 m64n128k32 s32", "m64n128k32.s32.s8.s8", 128, 64, "", 1, true},
};

double run_wg(const Wg& f, int reps) {
  std::string dl;
  for (uint32_t i = 0; i < f.dregs; ++i) dl += (i ? ", " : "") + std::string("%d") + std::to_string(i);
  std::string init;
  for (uint32_t i = 0; i < f.dregs; ++i) init += "    mov.u32 %d" + std::to_string(i) + ", 0;\n";
  const std::string ptx = std::string(".version 8.3\n.target sm_90a\n.address_size 64\n") + R"(
.visible .entry k(.param .u64 pimg, .param .u64 da, .param .u64 db)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b32 %d<130>;
    .reg .b64 %rd<12>;
    .shared .align 1024 .b8 smem[40960];
    ld.param.u64 %rd1, [pimg];
    ld.param.u64 %rd5, [da];
    ld.param.u64 %rd6, [db];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, smem;
    shl.b32 %r3, %r1, 2;
COPY:
    setp.ge.u32 %p1, %r3, 40960;
    @%p1 bra COPIED;
    cvt.u64.u32 %rd7, %r3;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r4, [%rd8];
    add.u32 %r5, %r2, %r3;
    st.shared.u32 [%r5], %r4;
    add.u32 %r3, %r3, 512;
    bra COPY;
COPIED:
    bar.sync 0;
    shr.u32 %r6, %r2, 4;
    cvt.u64.u32 %rd11, %r6;
    add.u64 %rd5, %rd5, %rd11;
    add.u64 %rd6, %rd6, %rd11;
    setp.ne.u32 %p2, %r1, 99999;
    mov.u32 %r9, 0;
)" + init + R"(LOOP:
    wgmma.fence.sync.aligned;
    wgmma.mma_async.sync.aligned.)" + f.shape + " {" + dl + "}, %rd5, %rd6, " + "%p2" + f.tail + R"(;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;
    add.u32 %r9, %r9, 1;
    setp.lt.u32 %p3, %r9, )" + std::to_string(reps) + R"(;
    @%p3 bra LOOP;
    ret;
}
)";
  MemoryManager mem{1 << 22};
  DeviceProfile prof = load_gpu("nvidia/h100");
  auto m = ptx::parse(ptx);
  std::vector<uint8_t> img(32768);
  for (size_t at = 0; at + 2 <= img.size(); at += 2) {
    uint16_t v = f.int_acc || f.eb == 1 ? static_cast<uint16_t>(rng()) : f16_normal();
    if (f.eb == 4 && at % 4 == 0) {
      const uint32_t t = f32_normal();
      std::memcpy(&img[at], &t, 4);
      at += 2;
      continue;
    }
    std::memcpy(&img[at], &v, 2);
  }
  const uint64_t pimg = mem.alloc(img.size());
  mem.write(pimg, img.data(), img.size());
  // 128-byte swizzle, K-major: leading offset 16, stride 1024 (eight rows of 128 bytes); B after A's 8 KiB.
  const auto desc = [](uint32_t start) { return uint64_t{start >> 4} | (uint64_t{1} << 16) | (uint64_t{64} << 32) | (uint64_t{1} << 62); };
  LaunchConfig cfg;
  cfg.block = {128, 1, 1};
  const auto t0 = Clock::now();
  exec::launch(m.entries[0], cfg, {arg_u64(pimg), arg_u64(desc(0)), arg_u64(desc(8192))}, mem, prof);
  return std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
}

// ---- PTX: tcgen05.mma ---------------------------------------------------------------------------------------------------------

struct Tc {
  const char* name;
  const char* kind;     // kind::
  const char* kind_mods;
  uint32_t atype, btype, dtype, M, N;
  bool ws;
  bool sparse;
  bool mx;
  int eb;
};
const Tc kTcs[] = {
    {"ptx tcgen05 f16 m128n256k16 f32", "f16", "", 0, 0, 1, 128, 256, false, false, false, 2},
    {"ptx tcgen05 f16 m128n256k16 f16", "f16", "", 0, 0, 0, 128, 256, false, false, false, 2},
    {"ptx tcgen05 f16 m64n256k16 f32 (layout F)", "f16", "", 0, 0, 1, 64, 256, false, false, false, 2},
    {"ptx tcgen05 f16 ws m64n128k16 f32 (layout B)", "f16", "", 0, 0, 1, 64, 128, true, false, false, 2},
    {"ptx tcgen05 f16 sparse m128n128k32 f32", "f16", "", 0, 0, 1, 128, 128, false, true, false, 2},
    {"ptx tcgen05 tf32 m128n256k8 f32", "tf32", "", 2, 2, 1, 128, 256, false, false, false, 4},
    {"ptx tcgen05 e4m3 m128n256k32 f32", "f8f6f4", "", 0, 0, 1, 128, 256, false, false, false, 1},
    {"ptx tcgen05 s8 m128n256k32 s32", "i8", "", 1, 1, 2, 128, 256, false, false, false, 1},
    {"ptx tcgen05 mxf8f6f4 m128n128k32 block", "mxf8f6f4", ".block_scale.scale_vec::1X", 0, 0, 1, 128, 128, false, false, true, 1},
};

double run_tc(const Tc& c, int reps) {
  const uint32_t K = c.eb == 4 ? 8 : c.eb == 2 ? 16 : 32;
  const uint32_t Ke = c.sparse ? 2 * K : K, Ka = K;
  uint32_t id;
  if (c.mx) id = (1u << 23) | (c.atype << 7) | (c.btype << 10) | (c.N >> 3) << 17 | (c.M >> 7) << 27;
  else id = c.dtype << 4 | c.atype << 7 | c.btype << 10 | (c.N >> 3) << 17 | (c.M >> 4) << 24 | (c.sparse ? 4u : 0u);
  uint32_t alloc = 32;
  while (alloc < c.N + 64) alloc *= 2;
  const std::string mma_args = std::string("[%r61], %rd5, %rd6, ") + (c.sparse ? "[%r65], " : "") + "%r50, " + (c.mx ? "[%r63], [%r64], " : (c.ws ? "" : "{0, 0, 0, 0}, ")) +
                               "%p7";
  const std::string g = "1";
  const std::string ptx = std::string(".version 8.7\n.target sm_100a\n.address_size 64\n") + R"(
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 da, .param .u64 db, .param .u32 id)
{
    .reg .pred %p<8>;
    .reg .b32 %r<80>;
    .reg .b32 %v<40>;
    .reg .b64 %rd<24>;
    .shared .align 1024 .b8 smem[16384];
    .shared .align 8 .b64 bar;
    .shared .align 4 .b32 slot;
    ld.param.u64 %rd1, [pa];
    ld.param.u64 %rd2, [pb];
    ld.param.u64 %rd5, [da];
    ld.param.u64 %rd6, [db];
    ld.param.u32 %r50, [id];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, smem;
    shl.b32 %r3, %r1, 2;
COPY:
    setp.ge.u32 %p1, %r3, 8192;
    @%p1 bra COPIED;
    cvt.u64.u32 %rd8, %r3;
    add.u64 %rd9, %rd1, %rd8;
    ld.global.u32 %r4, [%rd9];
    add.u32 %r5, %r2, %r3;
    st.shared.u32 [%r5], %r4;
    add.u64 %rd9, %rd2, %rd8;
    ld.global.u32 %r4, [%rd9];
    add.u32 %r5, %r5, 8192;
    st.shared.u32 [%r5], %r4;
    add.u32 %r3, %r3, 512;
    bra COPY;
COPIED:
    shr.u32 %r6, %r1, 5;
    setp.eq.u32 %p2, %r6, 0;
    @%p2 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [slot], )" + std::to_string(alloc) + R"(;
    setp.eq.u32 %p3, %r1, 0;
    mov.u32 %r7, bar;
    @%p3 mbarrier.init.shared::cta.b64 [%r7], 1;
    fence.proxy.async.shared::cta;
    bar.sync 0;
    ld.shared.u32 %r40, [slot];
    tcgen05.fence::before_thread_sync;
    bar.sync 0;
    tcgen05.fence::after_thread_sync;
    shr.u32 %r8, %r2, 4;
    cvt.u64.u32 %rd12, %r8;
    add.u64 %rd5, %rd5, %rd12;
    add.u64 %rd6, %rd6, %rd12;
    add.u64 %rd6, %rd6, 512;
    add.u32 %r61, %r40, 0;
    add.u32 %r63, %r40, )" + std::to_string(c.N) + R"(;
    add.u32 %r64, %r40, )" + std::to_string(c.N + 4) + R"(;
    add.u32 %r65, %r40, )" + std::to_string(c.N + 8) + R"(;
    setp.eq.u32 %p7, %r1, %r1;
    mov.u32 %r9, 0;
LOOP:
    @%p3 tcgen05.mma)" + (c.ws ? ".ws" : "") + (c.sparse ? ".sp" : "") + ".cta_group::" + g + ".kind::" + c.kind + c.kind_mods + " " + mma_args + R"(;
    add.u32 %r9, %r9, 1;
    setp.lt.u32 %p4, %r9, )" + std::to_string(reps) + R"(;
    @%p4 bra LOOP;
    @%p3 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%r7], 1;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r7], 0;
    @!%p1 bra WAIT;
    tcgen05.fence::after_thread_sync;
    bar.sync 0;
    @%p2 tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r40, )" + std::to_string(alloc) + R"(;
    @%p2 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
    ret;
}
)";
  MemoryManager mem{1 << 22};
  DeviceProfile prof = load_gpu("nvidia/b200");
  auto m = ptx::parse(ptx);
  std::vector<uint8_t> sa(8192), sb(8192);
  for (size_t at = 0; at + 2 <= sa.size(); at += 2) {
    uint16_t v = c.eb == 2 ? f16_normal() : static_cast<uint16_t>(rng() & 0x3f3f);   // (bytes below 0x40: small e4m3 numbers, no NaN codes)
    if (c.eb == 4) v = static_cast<uint16_t>(rng());
    std::memcpy(&sa[at], &v, 2);
    std::memcpy(&sb[at], &v, 2);
  }
  if (c.eb == 4)
    for (size_t at = 0; at + 4 <= sa.size(); at += 4) {
      const uint32_t t = f32_normal();
      std::memcpy(&sa[at], &t, 4);
      std::memcpy(&sb[at], &t, 4);
    }
  const uint64_t pa = mem.alloc(8192), pb = mem.alloc(8192);
  mem.write(pa, sa.data(), 8192);
  mem.write(pb, sb.data(), 8192);
  // K-major, the swizzle atom as wide as a row of K elements (32, 64 or 128 bytes), eight rows an atom
  const auto desc = [](uint32_t start, uint32_t row_bytes) {
    const uint64_t code = row_bytes == 32 ? 6 : row_bytes == 64 ? 4 : 2;
    return uint64_t{start >> 4} | (uint64_t{1} << 16) | (uint64_t{8 * row_bytes >> 4} << 32) | (uint64_t{1} << 46) | (code << 61);
  };
  LaunchConfig cfg;
  cfg.block = {128, 1, 1};
  cfg.grid = {1, 1, 1};
  cfg.cluster = {1, 1, 1};
  const uint32_t a_row = Ka * c.eb, b_row = Ke * c.eb;
  const auto t0 = Clock::now();
  exec::launch(m.entries[0], cfg, {arg_u64(pa), arg_u64(pb), arg_u64(desc(0, a_row)), arg_u64(desc(0, b_row)), arg_u32(id)}, mem, prof);
  return std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
}

// ---- SASS: HMMA, IMMA, QMMA, ... words of the corpora -------------------------------------------------------------------------

using sass::Word;

Word with(Word w, unsigned pos, unsigned len, uint64_t v) {
  for (unsigned i = 0; i < len; ++i) {
    const unsigned p = pos + i;
    uint64_t& half = p >= 64 ? w.hi : w.lo;
    const uint64_t bit = uint64_t{1} << (p % 64);
    half = (v >> i & 1) ? (half | bit) : (half & ~bit);
  }
  return w;
}
struct Line {
  Word w;
  std::string text;
};
std::vector<Line> read_corpus(const std::string& dir, const std::string& name) {
  std::ifstream in(dir + "/nvidia/tests/data/sass/sm_" + name + ".txt");
  std::vector<Line> out;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    const size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1);
    if (t1 == std::string::npos || t2 == std::string::npos) continue;
    Line l;
    l.w.hi = std::stoull(line.substr(0, 16), nullptr, 16);
    l.w.lo = std::stoull(line.substr(16, 16), nullptr, 16);
    l.text = line.substr(t2 + 1);
    out.push_back(std::move(l));
  }
  return out;
}
Word find_word(const std::vector<Line>& c, const char* pattern) {
  const std::regex re(pattern);
  for (const Line& l : c)
    if (std::regex_match(l.text, re)) return l.w;
  std::fprintf(stderr, "the corpus has no instruction like %s\n", pattern);
  std::exit(2);
}
template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}
std::string make_cubin(int sm, const std::vector<Word>& code) {
  const std::string strtab("\0k\0", 3);
  std::string shstr(1, '\0');
  for (const char* n : {".shstrtab", ".strtab", ".symtab", ".nv.info", ".text.k", ".nv.info.k"}) {
    shstr += n;
    shstr.push_back('\0');
  }
  const auto name_of = [&](const char* n) { return static_cast<uint32_t>(shstr.find(n, 1)); };
  std::string info_mod;
  put<uint8_t>(info_mod, 4), put<uint8_t>(info_mod, 0x2f), put<uint16_t>(info_mod, 8), put<uint32_t>(info_mod, 1), put<uint32_t>(info_mod, 250);
  std::string info_k;
  put<uint8_t>(info_k, 4), put<uint8_t>(info_k, 0x17), put<uint16_t>(info_k, 12);
  put<uint32_t>(info_k, 0), put<uint32_t>(info_k, 0), put<uint32_t>(info_k, 8u << 18);
  std::string symtab(24, '\0');
  put<uint32_t>(symtab, 1), put<uint8_t>(symtab, 0x12), put<uint8_t>(symtab, 0), put<uint16_t>(symtab, 5);
  put<uint64_t>(symtab, 0), put<uint64_t>(symtab, 16 * code.size());
  std::string out(64, '\0');
  const auto place = [&](const std::string& bytes) {
    while (out.size() % 16) out.push_back('\0');
    const uint64_t at = out.size();
    out += bytes;
    return at;
  };
  const uint64_t shstr_off = place(shstr), str_off = place(strtab), sym_off = place(symtab), imod_off = place(info_mod);
  std::string text;
  for (const Word& w : code) put(text, w.lo), put(text, w.hi);
  const uint64_t text_off = place(text), ik_off = place(info_k);
  while (out.size() % 8) out.push_back('\0');
  const uint64_t shoff = out.size();
  const auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size, uint32_t link, uint64_t entsize) {
    put<uint32_t>(out, name), put<uint32_t>(out, type);
    put<uint64_t>(out, 0), put<uint64_t>(out, 0), put<uint64_t>(out, off), put<uint64_t>(out, size);
    put<uint32_t>(out, link), put<uint32_t>(out, 0), put<uint64_t>(out, 16), put<uint64_t>(out, entsize);
  };
  shdr(0, 0, 0, 0, 0, 0);
  shdr(name_of(".shstrtab"), 3, shstr_off, shstr.size(), 0, 0);
  shdr(name_of(".strtab"), 3, str_off, strtab.size(), 0, 0);
  shdr(name_of(".symtab"), 2, sym_off, symtab.size(), 2, 24);
  shdr(name_of(".nv.info"), 0x70000000, imod_off, info_mod.size(), 0, 0);
  shdr(name_of(".text.k"), 1, text_off, text.size(), 0, 0);
  shdr(name_of(".nv.info.k"), 0x70000000, ik_off, info_k.size(), 0, 0);
  out[0] = 0x7f, out[1] = 'E', out[2] = 'L', out[3] = 'F', out[4] = 2, out[5] = 1, out[6] = 1, out[7] = 0x41, out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = 7, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (static_cast<uint32_t>(sm) << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

struct SassRow {
  const char* name;
  int sm;
  const char* pattern;   // regex over the corpus text of the MMA word
  const char* data;      // "f16", "bf16", "f32", "int", "fp8"
  const char* cdata;
};
// (the first corpus word of each form whose text matches)
const SassRow kSass[] = {
    {"sass HMMA.16816.F32 (sm_86)", 86, "HMMA\\.16816\\.F32 R[0-9]+, R[0-9]+, R[0-9]+, R[0-9]+", "f16", "f32"},
    {"sass HMMA.16816.F32.BF16 (sm_86)", 86, "HMMA\\.16816\\.F32\\.BF16 R[0-9]+, R[0-9]+, R[0-9]+, R[0-9]+", "bf16", "f32"},
    {"sass HMMA.1688.F32.TF32 (sm_86)", 86, "HMMA\\.1688\\.F32\\.TF32 R[0-9]+, R[0-9]+, R[0-9]+, R[0-9]+", "f32", "f32"},
    {"sass HMMA.1688.F32 (sm_86)", 86, "HMMA\\.1688\\.F32 R[0-9]+, R[0-9]+, R[0-9]+, R[0-9]+", "f16", "f32"},
    {"sass HMMA.16816.F32 (sm_80)", 80, "HMMA\\.16816\\.F32 R[0-9]+, R[0-9]+, R[0-9]+, R[0-9]+", "f16", "f32"},
    {"sass HMMA.SP.16832.F32 (sm_80)", 80, "HMMA\\.SP\\.16832\\.F32 R[0-9]+, R[0-9]+, R[0-9]+, R[0-9]+, R[0-9]+, 0x[0-9a-f]+", "f16", "f32"},
    {"sass IMMA.16816.S8.S8 (sm_80)", 80, "IMMA\\.16816\\.S8\\.S8 R[0-9]+, R[0-9]+\\.ROW, R[0-9]+\\.COL, R[0-9]+", "int", "int"},
    {"sass IMMA.16832.S8.S8 (sm_80)", 80, "IMMA\\.16832\\.S8\\.S8 R[0-9]+, R[0-9]+\\.ROW, R[0-9]+\\.COL, R[0-9]+", "int", "int"},
    {"sass IMMA.16864.S4.S4 (sm_80)", 80, "IMMA\\.16864\\.S4\\.S4 R[0-9]+, R[0-9]+\\.ROW, R[0-9]+\\.COL, R[0-9]+", "int", "int"},
    {"sass QMMA.16832.F32.E4M3 (sm_89)", 89, "QMMA\\.16832\\.F32\\.E4M3\\.E4M3 R[0-9]+, R[0-9]+, R[0-9]+, R[0-9]+", "fp8", "f32"},
};

double run_sass(const SassRow& r, const std::string& src_dir, int reps) {
  static std::map<int, std::vector<Line>> corpora;
  if (!corpora.count(r.sm)) corpora[r.sm] = read_corpus(src_dir, std::to_string(r.sm));
  const std::vector<Line>& corpus = corpora[r.sm];
  const Word mma = find_word(corpus, r.pattern);
  const sass::Instr ins = sass::decode(mma, 0, r.sm);
  // Registers in the corpus word: A, B, C (and the metadata) are where it names them; D likewise. Loads fill R0..R(n) and the word reads them, so
  // the operands are loaded from the registers the word names (up to 240).
  const unsigned rd = ins.dst[0].reg, ra = ins.src[0].reg, rb = ins.src[1].reg, rc = ins.src[2].reg;
  unsigned a = 4, b = 2, c = 4;
  const std::string t = r.name;
  if (t.find("16832.F32 ") != std::string::npos && r.sm == 80 && t.find("SP") == std::string::npos) a = 4, b = 2;
  if (t.find("1688.F32.TF32") != std::string::npos) a = 4, b = 2;
  if (t.find("1688.F32 ") != std::string::npos) a = 2, b = 1;
  if (t.find("16816.F32") != std::string::npos) a = 4, b = 2;
  if (t.find("HMMA.SP") != std::string::npos) a = 4, b = 4;
  if (t.find("IMMA.16816") != std::string::npos) a = 2, b = 1;
  if (t.find("IMMA.16832") != std::string::npos) a = 4, b = 2;
  if (t.find("IMMA.16864") != std::string::npos) a = 4, b = 2;
  if (t.find("QMMA") != std::string::npos) a = 4, b = 2;
  const bool sp = t.find("SP") != std::string::npos;
  std::vector<Word> code;
  const Word mov = find_word(corpus, "MOV R[0-9]+, c\\[0x0\\]\\[0x[0-9a-f]+\\]");
  const Word imad = find_word(corpus, "IMAD\\.WIDE\\.U32 R[0-9]+, R[0-9]+, 0x[0-9a-f]+, R[0-9]+");
  const Word ldg = find_word(corpus, "LDG\\.E R[0-9]+, \\[R[0-9]+\\.64(\\+0x[0-9a-f]+)?\\]");
  const Word stg = find_word(corpus, "STG\\.E \\[R[0-9]+\\.64(\\+0x[0-9a-f]+)?\\], R[0-9]+");
  code.push_back(with(find_word(corpus, "S2R R[0-9]+, SR_TID\\.X"), 16, 8, 250));
  code.push_back(with(with(mov, 16, 8, 248), 40, 16, 0x58));
  code.push_back(with(with(mov, 16, 8, 249), 40, 16, 0x59));
  // R246:247 = base + tid * 256
  code.push_back(with(with(with(with(imad, 16, 8, 246), 24, 8, 250), 32, 32, 256), 64, 8, 248));
  unsigned word = 0;
  const auto load = [&](unsigned base, unsigned n) {
    for (unsigned i = 0; i < n; ++i, ++word) code.push_back(with(with(with(ldg, 16, 8, base + i), 24, 8, 246), 40, 24, word * 4));
  };
  load(ra, a);
  load(rb, b);
  load(rc, c);
  if (sp) {
    const unsigned rm = ins.src[3].reg;
    load(rm, 1);
  }
  for (int i = 0; i < reps; ++i) code.push_back(mma);
  code.push_back(with(with(with(stg, 24, 8, 246), 32, 8, rd), 40, 24, 0));
  code.push_back(find_word(corpus, "EXIT"));
  const std::string cubin = make_cubin(r.sm, code);
  MemoryManager mem{1 << 22};
  DeviceProfile prof = load_gpu(r.sm == 80 ? "nvidia/a100" : r.sm == 86 ? "nvidia/a10" : "nvidia/rtx4090");
  auto m = sass::load(reinterpret_cast<const uint8_t*>(cubin.data()), cubin.size(), mem, prof);
  std::vector<uint32_t> in(32 * 64);
  for (int l = 0; l < 32; ++l) {
    unsigned at = 0;
    for (unsigned i = 0; i < a + b; ++i) in[l * 64 + at++] = word_of(r.data);
    for (unsigned i = 0; i < c; ++i) in[l * 64 + at++] = word_of(r.cdata);
    in[l * 64 + at++] = rng();
  }
  const uint64_t buf = mem.alloc(in.size() * 4 + 4096);
  mem.write(buf, in.data(), in.size() * 4);
  exec::LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {32, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &buf, 8);
  const auto t0 = Clock::now();
  sass::launch(*m, "k", cfg, {arg}, mem, prof);
  return std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
}

double best_of(int n, const std::function<double()>& f) {
  double best = 1e30;
  for (int i = 0; i < n; ++i) best = std::min(best, f());
  return best;
}

void report(const char* name, int r, const std::function<double(int)>& run) {
  try {
    run(2);   // warm-up (tables, first-use checks)
    const double t1 = best_of(5, [&] { return run(r); }), t2 = best_of(5, [&] { return run(2 * r); });
    const double ns = (t2 - t1) / r;
    std::printf("%-46s %12.1f ns  %12.0f cycles\n", name, ns, ns * ghz());
  } catch (const std::exception& e) {
    std::printf("%-46s skipped: %s\n", name, e.what());
  }
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string filter = argc > 1 ? argv[1] : "";
  const int reps = argc > 2 ? std::atoi(argv[2]) : 400;
  const char* src = std::getenv("VGPU_SOURCE_DIR");
  const auto want = [&](const char* name) { return filter.empty() || std::string(name).find(filter) != std::string::npos; };
  for (const Reg& f : kRegs)
    if (want(f.name)) report(f.name, reps, [&](int r) { return run_reg(f, r); });
  for (const Wg& f : kWgs)
    if (want(f.name)) report(f.name, std::max(10, reps / 8), [&](int r) { return run_wg(f, r); });
  for (const Tc& f : kTcs)
    if (want(f.name)) report(f.name, std::max(10, reps / 8), [&](int r) { return run_tc(f, r); });
  if (src)
    for (const SassRow& f : kSass)
      if (want(f.name)) report(f.name, reps, [&](int r) { return run_sass(f, src, r); });
  return 0;
}
