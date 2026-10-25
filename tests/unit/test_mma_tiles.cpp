// The PTX engine's tile route for the mma.sync forms that exec_mma_simd and the sm_86 sum do not take (Interpreter::exec_mma_tiles): f16, bf16,
// tf32, e4m3 and e5m2 operands with f32 and f16 sums (mixed C and D types, sparse A), the s8, u8, s4 and u4 products with and without .satfinite
// (dense and sparse, m8n8 and m16n8), the f64 form in all four rounding modes, and the single-bit forms, against the general form
// (VGPU_FASTPATH=0), bit for bit, with half the lanes of one warp predicated off and operands that break arithmetic (zeros of both signs,
// subnormals, infinities, NaNs of both kinds, the largest values, wide and narrow exponents, tf32 words with their low 13 bits set, integers at
// their limits, accumulators that saturate).
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
constexpr int kTrials = 48;
constexpr int kStride = 64;    // 32-bit words a thread and trial: a, b, c, the metadata (f64 operands take two words a register: m16n8k16 needs 33)

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(&b[0], &v, 8);
  return b;
}

enum class D { F16, BF16, TF32, Int, F64, Fp8 };   // how the operand words are drawn

struct Form {
  const char* name;
  const char* gpu;
  const char* target;     // ".target" of the module
  const char* mnemonic;   // up to and including the types (and .satfinite or the popc suffix as it spells them)
  int a, b, c, dregs;     // registers of each fragment
  D data;
  bool sparse = false;
  bool wide = false;      // 64-bit registers
  const char* suffix = "";   // after the operands' types (a rounding mode, .xor.popc)
  bool c_int_limits = false;   // draw C near the s32 limits
};

const Form kForms[] = {
    // floating point, other C and D types than exec_mma_simd takes
    {"bf16 k16 f32<-f32 on h100", "nvidia/h100", "sm_90", "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", 4, 2, 4, 4, D::BF16},
    {"f16 sparse k32", "nvidia/a100", "sm_80", "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32", 4, 4, 4, 4, D::F16, true},
    {"f16 sparse k16 f16", "nvidia/a100", "sm_80", "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16", 2, 2, 2, 2, D::F16, true},
    {"bf16 sparse k32", "nvidia/a100", "sm_80", "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32", 4, 4, 4, 4, D::BF16, true},
    {"tf32 sparse k16", "nvidia/a100", "sm_80", "mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32", 4, 4, 4, 4, D::TF32, true},
    {"tf32 sparse k8", "nvidia/h100", "sm_90", "mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", 2, 2, 4, 4, D::TF32, true},
    // fp8 (sm_89)
    {"e4m3 k32 f32", "nvidia/rtx4090", "sm_89", "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", 4, 2, 4, 4, D::Fp8},
    {"e5m2 k32 f32", "nvidia/rtx4090", "sm_89", "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32", 4, 2, 4, 4, D::Fp8},
    {"e4m3.e5m2 k32 f16", "nvidia/rtx4090", "sm_89", "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e5m2.f16", 4, 2, 2, 2, D::Fp8},
    {"e4m3 k16 f32", "nvidia/rtx4090", "sm_89", "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32", 2, 1, 4, 4, D::Fp8},
    {"e5m2 k16 f16", "nvidia/rtx4090", "sm_89", "mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e5m2.f16", 2, 1, 2, 2, D::Fp8},
    {"e4m3 sparse k64", "nvidia/h100", "sm_90", "mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32", 4, 4, 4, 4, D::Fp8, true},
    // integers
    {"s8 k16", "nvidia/a100", "sm_80", "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32", 2, 1, 4, 4, D::Int, false, false, "", true},
    {"s8 k32", "nvidia/a100", "sm_80", "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", 4, 2, 4, 4, D::Int, false, false, "", true},
    {"u8.s8 k32 sat", "nvidia/a100", "sm_80", "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32", 4, 2, 4, 4, D::Int, false, false, "", true},
    {"s8.u8 k16 sat", "nvidia/a100", "sm_80", "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.u8.s32", 2, 1, 4, 4, D::Int, false, false, "", true},
    {"s8 m8n8k16", "nvidia/a100", "sm_80", "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32", 1, 1, 2, 2, D::Int, false, false, "", true},
    {"u8 m8n8k16 sat", "nvidia/a100", "sm_80", "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32", 1, 1, 2, 2, D::Int, false, false, "", true},
    {"s4 k32", "nvidia/a100", "sm_80", "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32", 2, 1, 4, 4, D::Int, false, false, "", true},
    {"u4.s4 k64 sat", "nvidia/a100", "sm_80", "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.s4.s32", 4, 2, 4, 4, D::Int, false, false, "", true},
    {"s4 m8n8k32", "nvidia/a100", "sm_80", "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32", 1, 1, 2, 2, D::Int, false, false, "", true},
    {"s8 sparse k32", "nvidia/a100", "sm_80", "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", 2, 2, 4, 4, D::Int, true, false, "", true},
    {"u8 sparse k64 sat", "nvidia/a100", "sm_80", "mma.sp.sync.aligned.m16n8k64.row.col.satfinite.s32.u8.u8.s32", 4, 4, 4, 4, D::Int, true, false, "", true},
    {"s4 sparse k64", "nvidia/a100", "sm_80", "mma.sp.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32", 2, 2, 4, 4, D::Int, true, false, "", true},
    {"s4 sparse k128 sat", "nvidia/a100", "sm_80", "mma.sp.sync.aligned.m16n8k128.row.col.satfinite.s32.s4.s4.s32", 4, 4, 4, 4, D::Int, true, false, "", true},
    // f64, in every rounding mode
    {"f64 m8n8k4 rn", "nvidia/a100", "sm_80", "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64", 1, 1, 2, 2, D::F64, false, true, ".rn"},
    {"f64 m8n8k4 rz", "nvidia/a100", "sm_80", "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64", 1, 1, 2, 2, D::F64, false, true, ".rz"},
    {"f64 m8n8k4 rm", "nvidia/a100", "sm_80", "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64", 1, 1, 2, 2, D::F64, false, true, ".rm"},
    {"f64 m8n8k4 rp", "nvidia/a100", "sm_80", "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64", 1, 1, 2, 2, D::F64, false, true, ".rp"},
    {"f64 m16n8k4", "nvidia/h100", "sm_90", "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64", 2, 1, 4, 4, D::F64, false, true, ".rz"},
    {"f64 m16n8k8", "nvidia/h100", "sm_90", "mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64", 4, 2, 4, 4, D::F64, false, true, ".rn"},
    {"f64 m16n8k16", "nvidia/h100", "sm_90", "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64", 8, 4, 4, 4, D::F64, false, true, ".rp"},
    // single bit
    {"b1 xor m8n8k128", "nvidia/a100", "sm_80", "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32", 1, 1, 2, 2, D::Int, false, false, ".xor.popc", true},
    {"b1 and m16n8k128", "nvidia/a100", "sm_80", "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32", 2, 1, 4, 4, D::Int, false, false, ".and.popc", true},
    {"b1 xor m16n8k256", "nvidia/a100", "sm_80", "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32", 4, 2, 4, 4, D::Int, false, false, ".xor.popc", true},
};

uint32_t f32_class(std::mt19937& g, bool wide) {
  const uint32_t h = g();
  switch (h % 16) {
    case 0: return 0;
    case 1: return 0x80000000u;
    case 2: return h & 0x807fffffu;
    case 3: return 0x7f800000u | (h & 0x80000000u);
    case 4: return 0x7fc00000u | (h & 0x803fffffu);
    case 5: return 0x7f800001u | (h & 0x80000000u);
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
    case 2: return static_cast<uint16_t>((h & 0x8000u) | ((h >> 8) & 0x3ffu));
    case 3: return static_cast<uint16_t>(0x7c00u | (h & 0x8000u));
    case 4: return static_cast<uint16_t>(0x7e00u | (h & 0x83ffu));
    case 5: return static_cast<uint16_t>((h & 0x8000u) | 0x7bffu);
    default: {
      const uint32_t e = wide ? 1 + (h >> 8) % 30 : 10 + (h >> 8) % 10;
      return static_cast<uint16_t>((h & 0x8000u) | (e << 10) | (g() & 0x3ffu));
    }
  }
}
uint64_t f64_class(std::mt19937& g, bool wide) {
  const uint64_t h = (uint64_t{g()} << 32) | g();
  switch (h % 12) {
    case 0: return 0;
    case 1: return 0x8000000000000000ull;
    case 2: return h & 0x800fffffffffffffull;
    case 3: return 0x7ff0000000000000ull | (h & 0x8000000000000000ull);
    case 4: return 0x7ff8000000000000ull | (h >> 12);
    case 5: return (h & 0x8000000000000000ull) | 0x7fefffffffffffffull;
    default: {
      const uint64_t e = wide ? 1 + (h >> 20) % 2046 : 1013 + (h >> 20) % 20;
      return (h & 0x8000000000000000ull) | (e << 52) | (h & 0xfffffffffffffull);
    }
  }
}

uint32_t draw(const Form& f, std::mt19937& g, bool wide, bool is_c) {
  switch (f.data) {
    case D::F16: return half_class(g, wide) | uint32_t{half_class(g, wide)} << 16;
    case D::BF16: return (f32_class(g, wide) >> 16) | (f32_class(g, wide) & 0xffff0000u);
    case D::TF32: return f32_class(g, wide);
    case D::Fp8: {
      if (is_c) return half_class(g, wide) | uint32_t{half_class(g, wide)} << 16;   // (f32 C is drawn by the caller for the float forms)
      return g();
    }
    default: return g();
  }
}

std::string reglist(const char* p, int n) {
  std::string s = "{";
  for (int i = 0; i < n; ++i) s += std::string(i ? "," : "") + "%" + p + std::to_string(i);
  return s + "}";
}

std::vector<uint32_t> run(const Form& f, const std::vector<uint32_t>& in) {
  const char* rt = f.wide ? "b64" : "b32";
  std::string ptx = std::string(".version 8.3\n.target ") + f.target + R"(
.address_size 64
.visible .entry k(.param .u64 pin, .param .u64 pout)
{
    .reg .pred %p<4>;
    .reg .pred %q, %again;
    .reg .b32 %r<16>;
    .reg .b32 %meta;
    .reg .)" + rt + R"( %a<8>, %b<8>, %c<8>, %d<8>;
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
    mul.wide.u32 %rd3, %r4, 256;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
)";
  const int w = f.wide ? 8 : 4;
  int off = 0;
  for (int i = 0; i < f.a; ++i, off += w) ptx += std::string("    ld.global.") + rt + " %a" + std::to_string(i) + ", [%rd4+" + std::to_string(off) + "];\n";
  for (int i = 0; i < f.b; ++i, off += w) ptx += std::string("    ld.global.") + rt + " %b" + std::to_string(i) + ", [%rd4+" + std::to_string(off) + "];\n";
  for (int i = 0; i < f.c; ++i, off += w) ptx += std::string("    ld.global.") + rt + " %c" + std::to_string(i) + ", [%rd4+" + std::to_string(off) + "];\n";
  ptx += "    ld.global.b32 %meta, [%rd4+" + std::to_string(off) + "];\n";
  for (int i = 0; i < f.dregs; ++i) ptx += std::string("    mov.") + rt + " %d" + std::to_string(i) + ", " + (f.wide ? "0xA5A5A5A5A5A5A5A5" : "0xA5A5A5A5") + ";\n";
  ptx += std::string("    @%q ") + f.mnemonic + f.suffix + " " + reglist("d", f.dregs) + ", " + reglist("a", f.a) + ", " + reglist("b", f.b) + ", " +
         reglist("c", f.c) + (f.sparse ? ", %meta, 0x0" : "") + ";\n";
  for (int i = 0; i < f.dregs; ++i)
    ptx += std::string("    st.global.") + rt + " [%rd5+" + std::to_string(i * w) + "], %d" + std::to_string(i) + ";\n";
  ptx += R"(    add.u32 %r3, %r3, 1;
    setp.lt.u32 %again, %r3, )" + std::to_string(kTrials) + R"(;
    @%again bra LOOP;
    ret;
}
)";
  MemoryManager mem{1 << 22};
  DeviceProfile prof = load_gpu(f.gpu);
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

VTEST(mma_tile_route_matches_the_general_form) {
  std::mt19937 g(0x44a1);
  long compared = 0;
  for (const Form& f : kForms) {
    std::vector<uint32_t> in(size_t{kTrials} * kThreads * kStride);
    for (int t = 0; t < kTrials; ++t) {
      const bool wide = (t / 2) % 2 == 1;
      for (int th = 0; th < kThreads; ++th) {
        uint32_t* word = &in[(static_cast<size_t>(t) * kThreads + th) * kStride];
        // (the layout of run(): a, b, c, meta; 64-bit registers take two words)
        const int w = f.wide ? 2 : 1;
        int at = 0;
        const auto fill = [&](int n, bool is_c) {
          for (int i = 0; i < n * w; i += w, at += w) {
            if (f.wide) {
              const uint64_t v = f64_class(g, wide);
              std::memcpy(&word[at], &v, 8);
            } else if (is_c && f.data == D::Int && f.c_int_limits) {
              const uint32_t h = g();
              word[at] = h % 4 == 0 ? 0x7fffffffu - (h >> 8) % 100000 : h % 4 == 1 ? 0x80000000u + (h >> 8) % 100000 : g();
            } else if (is_c && (f.data == D::TF32 || f.data == D::BF16 || f.data == D::F16 || f.data == D::Fp8)) {
              // (the C registers hold f32 or f16 values, by the form's last type)
              const std::string mn = f.mnemonic;
              word[at] = mn.size() >= 4 && mn.compare(mn.size() - 4, 4, ".f16") == 0 ? half_class(g, wide) | uint32_t{half_class(g, wide)} << 16
                                                                                   : f32_class(g, wide);
            } else {
              word[at] = draw(f, g, wide, false);
            }
          }
        };
        fill(f.a, false);
        fill(f.b, false);
        fill(f.c, true);
        word[at] = g();   // the metadata
      }
    }
    setenv("VGPU_FASTPATH", "1", 1);
    std::vector<uint32_t> fast, slow;
    try {
      fast = run(f, in);
      setenv("VGPU_FASTPATH", "0", 1);
      slow = run(f, in);
    } catch (const Error& e) {
      unsetenv("VGPU_FASTPATH");
      throw vtest::Failure(std::string(f.name) + ": " + e.what());
    }
    unsetenv("VGPU_FASTPATH");
    for (size_t i = 0; i < fast.size(); ++i) {
      ++compared;
      if (fast[i] != slow[i]) {
        char buf[300];
        std::snprintf(buf, sizeof buf, "%s: word %zu (trial %zu, thread %zu): route 0x%08x, general 0x%08x", f.name, i, i / (kThreads * kStride), (i / kStride) % kThreads,
                      fast[i], slow[i]);
        throw vtest::Failure(buf);
      }
    }
  }
  VCHECK(compared > 500000);
}

VTEST_MAIN
