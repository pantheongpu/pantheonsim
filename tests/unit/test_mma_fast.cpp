// The PTX engine's SIMD route for mma.sync (Interpreter::exec_mma_simd: dense f16, bf16 and tf32, m16n8k4 / k8 / k16, f32 or f16 accumulator)
// against its general path (VGPU_FASTPATH=0), bit for bit, on an sm_86 device (the card-measured chunked sum), an sm_80 device and an sm_90
// one (the sequential f32 sum), with half the lanes of one warp predicated off, and operands that break arithmetic: zeros of both signs,
// subnormals, infinities, NaNs of both kinds, the largest finite values, wide and narrow exponents, tf32 words with their low 13 bits set.
// Forms the route declines (mixed C and D types, int, f64) are run too and must agree.
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

constexpr int kThreads = 64;   // warp 0 half predicated off, warp 1 all on
constexpr int kWords = 12;     // per thread and trial: a (4), b (2), c (4), spare
constexpr int kTrials = 160;

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

struct Form {
  const char* name;
  const char* shape;    // m16n8k16 ...
  const char* types;    // f32.f16.f16.f32 ...
  int fmt;              // 0 f16, 1 bf16, 2 tf32
  int a_regs, b_regs;
  bool c16, d16;
};

const Form kForms[] = {
    {"f16 k16 f32", "m16n8k16", "f32.f16.f16.f32", 0, 4, 2, false, false},
    {"f16 k8 f32", "m16n8k8", "f32.f16.f16.f32", 0, 2, 1, false, false},
    {"f16 k16 f16", "m16n8k16", "f16.f16.f16.f16", 0, 4, 2, true, true},
    {"f16 k8 f16", "m16n8k8", "f16.f16.f16.f16", 0, 2, 1, true, true},
    {"bf16 k16 f32", "m16n8k16", "f32.bf16.bf16.f32", 1, 4, 2, false, false},
    {"bf16 k8 f32", "m16n8k8", "f32.bf16.bf16.f32", 1, 2, 1, false, false},
    {"tf32 k8 f32", "m16n8k8", "f32.tf32.tf32.f32", 2, 4, 2, false, false},
    {"tf32 k4 f32", "m16n8k4", "f32.tf32.tf32.f32", 2, 2, 1, false, false},
};

uint32_t f32_class(std::mt19937& g, bool wide) {
  const uint32_t h = g();
  switch (h % 16) {
    case 0: return 0;
    case 1: return 0x80000000u;
    case 2: return h & 0x807fffffu;                     // a float subnormal
    case 3: return 0x7f800000u | (h & 0x80000000u);     // an infinity
    case 4: return 0x7fc00000u | (h & 0x803fffffu);     // a NaN (quiet)
    case 5: return 0x7f800001u | (h & 0x80000000u);     // a NaN (signalling)
    case 6: return (h & 0x80000000u) | 0x7f7fffffu;
    default: {
      const uint32_t e = wide ? 1 + (h >> 8) % 254 : 118 + (h >> 8) % 20;
      return (h & 0x80000000u) | (e << 23) | (g() & 0x7fffffu);
    }
  }
}

uint16_t half_class(std::mt19937& g, bool wide) {
  const uint32_t h = g();
  switch (h % 14) {
    case 0: return 0;
    case 1: return 0x8000;
    case 2: return static_cast<uint16_t>((h & 0x8000u) | ((h >> 8) & 0x3ffu));            // subnormal
    case 3: return static_cast<uint16_t>(0x7c00u | (h & 0x8000u));                       // infinity
    case 4: return static_cast<uint16_t>(0x7e00u | (h & 0x83ffu));                       // NaN
    case 5: return static_cast<uint16_t>((h & 0x8000u) | 0x7bffu);                       // largest
    default: {
      const uint32_t e = wide ? 1 + (h >> 8) % 30 : 10 + (h >> 8) % 10;
      return static_cast<uint16_t>((h & 0x8000u) | (e << 10) | (g() & 0x3ffu));
    }
  }
}

uint16_t bf16_class(std::mt19937& g, bool wide) {
  return static_cast<uint16_t>(f32_class(g, wide) >> 16);
}

std::vector<uint32_t> make_input(const Form& f, std::mt19937& g) {
  std::vector<uint32_t> in(size_t{kTrials} * kThreads * kWords);
  for (int t = 0; t < kTrials; ++t) {
    const bool wide = (t / 2) % 2 == 1;
    for (int th = 0; th < kThreads; ++th) {
      uint32_t* w = &in[(static_cast<size_t>(t) * kThreads + th) * kWords];
      for (int r = 0; r < 6; ++r) {
        if (f.fmt == 0) w[r] = half_class(g, wide) | uint32_t{half_class(g, wide)} << 16;
        else if (f.fmt == 1) w[r] = bf16_class(g, wide) | uint32_t{bf16_class(g, wide)} << 16;
        else w[r] = f32_class(g, wide);   // tf32: the low 13 bits are whatever they are
      }
      for (int r = 0; r < 4; ++r)
        w[6 + r] = f.c16 ? (half_class(g, wide) | uint32_t{half_class(g, wide)} << 16) : f32_class(g, wide);
    }
  }
  return in;
}

std::string reglist(const char* p, int n) {
  std::string s = "{";
  for (int i = 0; i < n; ++i) s += std::string(i ? "," : "") + "%" + p + std::to_string(i);
  return s + "}";
}

std::vector<uint32_t> run(const Form& f, const char* gpu, const std::vector<uint32_t>& in) {
  const int c_regs = f.c16 ? 2 : 4, d_regs = f.d16 ? 2 : 4;
  std::string ptx = R"(.version 8.3
.target sm_86
.address_size 64
.visible .entry k(.param .u64 pin, .param .u64 pout)
{
    .reg .pred %p<4>;
    .reg .pred %q, %again;
    .reg .b32 %r<16>;
    .reg .b32 %a<4>, %b<2>, %c<4>, %d<4>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [pin];
    ld.param.u64 %rd2, [pout];
    mov.u32 %r1, %tid.x;
    and.b32 %r2, %r1, 1;
    setp.eq.u32 %p1, %r2, 0;
    setp.ge.u32 %p2, %r1, 32;
    or.pred %q, %p1, %p2;
    mov.u32 %r3, 0;
LOOP:
    mad.lo.u32 %r4, %r3, 64, %r1;
    mul.wide.u32 %rd3, %r4, 48;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
)";
  for (int i = 0; i < f.a_regs; ++i) ptx += "    ld.global.u32 %a" + std::to_string(i) + ", [%rd4+" + std::to_string(4 * i) + "];\n";
  for (int i = 0; i < f.b_regs; ++i) ptx += "    ld.global.u32 %b" + std::to_string(i) + ", [%rd4+" + std::to_string(16 + 4 * i) + "];\n";
  for (int i = 0; i < c_regs; ++i) ptx += "    ld.global.u32 %c" + std::to_string(i) + ", [%rd4+" + std::to_string(24 + 4 * i) + "];\n";
  for (int i = 0; i < 4; ++i) ptx += "    mov.u32 %d" + std::to_string(i) + ", 0xA5A5A5A5;\n";
  ptx += std::string("    @%q mma.sync.aligned.") + f.shape + ".row.col." + f.types + " " + reglist("d", d_regs) + ", " + reglist("a", f.a_regs) + ", " +
         reglist("b", f.b_regs) + ", " + reglist("c", c_regs) + ";\n";
  for (int i = 0; i < d_regs; ++i) ptx += "    st.global.u32 [%rd5+" + std::to_string(4 * i) + "], %d" + std::to_string(i) + ";\n";
  ptx += R"(    add.u32 %r3, %r3, 1;
    setp.lt.u32 %again, %r3, )" + std::to_string(kTrials) + R"(;
    @%again bra LOOP;
    ret;
}
)";
  MemoryManager mem{1 << 22};
  DeviceProfile prof = load_gpu(gpu);
  auto m = ptx::parse(ptx);
  const uint64_t pin = mem.alloc(in.size() * 4);
  mem.write(pin, in.data(), in.size() * 4);
  const uint64_t pout = mem.alloc(in.size() * 4);
  std::vector<uint32_t> init(in.size(), 0x5A5A5A5Au);
  mem.write(pout, init.data(), init.size() * 4);
  LaunchConfig cfg;
  cfg.block = {kThreads, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(pin), arg_u64(pout)}, mem, prof);
  std::vector<uint32_t> out(in.size());
  mem.read(pout, out.data(), out.size() * 4);
  return out;
}

}  // namespace

VTEST(mma_simd_route_matches_the_general_path) {
  std::mt19937 g(0x3a8f);
  long compared = 0;
  for (const char* gpu : {"nvidia/rtx3060", "nvidia/a100", "nvidia/h100"})
    for (const Form& f : kForms) {
      const auto in = make_input(f, g);
      setenv("VGPU_FASTPATH", "1", 1);
      const auto fast = run(f, gpu, in);
      setenv("VGPU_FASTPATH", "0", 1);
      const auto slow = run(f, gpu, in);
      unsetenv("VGPU_FASTPATH");
      for (size_t i = 0; i < fast.size(); ++i) {
        ++compared;
        if (fast[i] != slow[i]) {
          char buf[300];
          std::snprintf(buf, sizeof buf, "%s on %s: word %zu (trial %zu, thread %zu): route 0x%08x, general 0x%08x", f.name, gpu, i,
                        i / (kThreads * kWords), (i / kWords) % kThreads, fast[i], slow[i]);
          throw vtest::Failure(buf);
        }
      }
    }
  VCHECK(compared > 1000000);
}

VTEST_MAIN
