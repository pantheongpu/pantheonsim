// The PTX engine's SIMD wgmma route (Interpreter::exec_wgmma_fast) against its general form (VGPU_FASTPATH=0), bit for bit, for every
// operand type the instruction takes -- f16, bf16, tf32, e4m3, e5m2, s8, u8 -- with f32, f16 and s32 accumulators, A from registers or from
// shared memory, the four swizzles and random leading/stride byte offsets, K-major and (16-bit types) MN-major operands, the -1 input
// scales, sparse A, .satfinite, a scale-d that differs between lanes, and operands that break arithmetic (zeros of both signs, subnormals,
// infinities, NaNs of both kinds, the largest values, wide and narrow exponents, tf32 words with their low 13 bits set).
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/wgmma.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

constexpr uint32_t kSmem = 32768;   // the shared-memory image: A in the first half's range, B anywhere

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

enum class T { F16, BF16, TF32, E4M3, E5M2, S8, U8 };
enum class Acc { F32, F16, S32 };

struct Form {
  const char* name;
  T a, b;
  Acc acc;
  uint32_t K;   // dense K (sparse doubles it)
  bool sat;
};

const char* ptx_type(T t) {
  switch (t) {
    case T::F16: return "f16";
    case T::BF16: return "bf16";
    case T::TF32: return "tf32";
    case T::E4M3: return "e4m3";
    case T::E5M2: return "e5m2";
    case T::S8: return "s8";
    default: return "u8";
  }
}
uint32_t ebytes(T t) { return t == T::F16 || t == T::BF16 ? 2 : t == T::TF32 ? 4 : 1; }
bool is_int(T t) { return t == T::S8 || t == T::U8; }

const Form kForms[] = {
    {"f16 f32", T::F16, T::F16, Acc::F32, 16, false},   {"f16 f16", T::F16, T::F16, Acc::F16, 16, false},
    {"bf16 f32", T::BF16, T::BF16, Acc::F32, 16, false}, {"tf32 f32", T::TF32, T::TF32, Acc::F32, 8, false},
    {"e4m3 f32", T::E4M3, T::E4M3, Acc::F32, 32, false}, {"e5m2 f16", T::E5M2, T::E5M2, Acc::F16, 32, false},
    {"e4m3.e5m2 f32", T::E4M3, T::E5M2, Acc::F32, 32, false}, {"s8 s32", T::S8, T::S8, Acc::S32, 32, false},
    {"u8.s8 s32 sat", T::U8, T::S8, Acc::S32, 32, true}, {"s8.u8 s32", T::S8, T::U8, Acc::S32, 32, false},
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

// A byte image whose every element (of the type) is a random member of the interesting classes.
void fill_elements(std::vector<uint8_t>& img, T t, std::mt19937& g, bool wide) {
  const uint32_t eb = ebytes(t);
  for (size_t at = 0; at + eb <= img.size(); at += eb) {
    uint32_t v;
    switch (t) {
      case T::F16: v = half_class(g, wide); break;
      case T::BF16: v = f32_class(g, wide) >> 16; break;
      case T::TF32: v = f32_class(g, wide); break;
      default: v = g() & 0xff;
    }
    std::memcpy(&img[at], &v, eb);
  }
}

uint64_t desc_bits(uint32_t start, uint32_t lbo, uint32_t sbo, uint32_t swizzle_code) {
  return uint64_t{(start & 0x3FFFF) >> 4} | (uint64_t{(lbo & 0x3FFFF) >> 4} << 16) | (uint64_t{(sbo & 0x3FFFF) >> 4} << 32) |
         (uint64_t{swizzle_code} << 62);
}

// A descriptor for a rows x K operand whose elements all land inside the image.
uint64_t pick_desc(std::mt19937& g, bool k_major, uint32_t eb, uint32_t rows, uint32_t K) {
  for (;;) {
    const uint32_t code = g() % 4;
    exec::WgmmaDesc d;
    d.swizzle = (uint32_t[]){0, 128, 64, 32}[code];
    d.start = 16 * (g() % 128);
    d.lbo = 16 * (g() % 96);
    d.sbo = 16 * (g() % 96);
    uint64_t hi = 0;
    for (uint32_t mn = 0; mn < rows; ++mn)
      for (uint32_t k = 0; k < K; ++k) hi = std::max(hi, exec::wgmma_smem_offset(d, k_major, eb, mn, k));
    if (hi + eb <= kSmem) return desc_bits(d.start, d.lbo, d.sbo, code);
  }
}

struct Case {
  const Form* f;
  uint32_t N;
  bool a_regs, sparse, mixed_scale_d, neg_a, neg_b, tr_a, tr_b;
  bool sat() const { return f->sat; }
};

std::vector<uint32_t> run(const Case& c, const std::vector<uint8_t>& img, const std::vector<uint32_t>& aregs, const std::vector<uint32_t>& din,
                          uint32_t meta, uint32_t sel, uint64_t da, uint64_t db) {
  const Form& f = *c.f;
  const uint32_t K = c.sparse ? 2 * f.K : f.K;
  const uint32_t dregs = f.acc == Acc::F16 ? c.N / 4 : c.N / 2;
  std::string dl, loads, stores;
  for (uint32_t i = 0; i < dregs; ++i) {
    dl += (i ? ", " : "") + std::string("%d") + std::to_string(i);
    loads += "    ld.global.u32 %d" + std::to_string(i) + ", [%rd12+" + std::to_string(4 * i) + "];\n";
    stores += "    st.global.u32 [%rd12+" + std::to_string(4 * i) + "], %d" + std::to_string(i) + ";\n";
  }
  const bool sixteen = f.a == T::F16 || f.a == T::BF16;
  std::string shape = "m64n" + std::to_string(c.N) + "k" + std::to_string(K) + (f.acc == Acc::F32 ? ".f32" : f.acc == Acc::F16 ? ".f16" : ".s32") + "." +
                      ptx_type(f.a) + "." + ptx_type(f.b) + (f.sat ? ".satfinite" : "");
  std::string tail;
  if (f.acc != Acc::S32) tail = std::string(", ") + (c.neg_a ? "-1" : "1") + ", " + (c.neg_b ? "-1" : "1") + (sixteen ? (c.a_regs ? std::string("") : std::string(", ") + (c.tr_a ? "1" : "0")) + ", " + (c.tr_b ? "1" : "0") : "");
  const std::string a = c.a_regs ? "{%r10, %r11, %r12, %r13}" : "%rd5";
  std::string ptx = ".version 8.3\n.target sm_90a\n.address_size 64\n" + std::string(R"(
.visible .entry k(.param .u64 pimg, .param .u64 pw, .param .u64 pd, .param .u64 da, .param .u64 db, .param .u32 pmeta)
{
    .reg .pred %p<4>;
    .reg .b32 %r<40>;
    .reg .b32 %d<130>;
    .reg .b64 %rd<20>;
    .shared .align 1024 .b8 smem[32768];
    ld.param.u64 %rd1, [pimg];
    ld.param.u64 %rd3, [pw];
    ld.param.u64 %rd4, [pd];
    ld.param.u64 %rd5, [da];
    ld.param.u64 %rd6, [db];
    ld.param.u32 %r14, [pmeta];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, smem;
    shl.b32 %r3, %r1, 2;
COPY:
    setp.ge.u32 %p1, %r3, 32768;
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
    mul.wide.u32 %rd9, %r1, 16;
    add.u64 %rd10, %rd3, %rd9;
    ld.global.v4.u32 {%r10, %r11, %r12, %r13}, [%rd10];
    shr.u32 %r6, %r2, 4;
    cvt.u64.u32 %rd11, %r6;
    add.u64 %rd5, %rd5, %rd11;
    add.u64 %rd6, %rd6, %rd11;
    mul.wide.u32 %rd9, %r1, )") + std::to_string(4 * dregs) + R"(;
    add.u64 %rd12, %rd4, %rd9;
)" + loads + (c.mixed_scale_d ? R"(
    mul.lo.u32 %r20, %r1, 7;
    add.u32 %r20, %r20, 3;
    rem.u32 %r21, %r20, 5;
    setp.ne.u32 %p2, %r21, 0;
)" : R"(
    setp.ne.u32 %p2, %r1, 99999;
)") + R"(
    wgmma.fence.sync.aligned;
    wgmma.mma_async)" + (c.sparse ? ".sp" : "") + ".sync.aligned." + shape + " {" + dl + "}, " + a + ", %rd6, " + (c.sparse ? "%r14, " + std::to_string(sel) + ", " : "") + "%p2" +
                    tail + R"(;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;
)" + stores + R"(
    ret;
}
)";
  MemoryManager mem{1 << 22};
  DeviceProfile prof = load_gpu("nvidia/h100");
  auto m = ptx::parse(ptx);
  auto put = [&](const void* p, size_t n) {
    const uint64_t va = mem.alloc(n ? n : 16);
    if (n) mem.write(va, p, n);
    return va;
  };
  const uint64_t pimg = put(img.data(), img.size()), pw = put(aregs.data(), aregs.size() * 4), pd = put(din.data(), din.size() * 4);
  LaunchConfig cfg;
  cfg.block = {128, 1, 1};
  std::vector<uint8_t> pm(4);
  std::memcpy(pm.data(), &meta, 4);
  exec::launch(m.entries[0], cfg, {arg_u64(pimg), arg_u64(pw), arg_u64(pd), arg_u64(da), arg_u64(db), pm}, mem, prof);
  std::vector<uint32_t> out(din.size());
  mem.read(pd, out.data(), out.size() * 4);
  return out;
}

}  // namespace

VTEST(wgmma_simd_route_matches_the_general_form) {
  std::mt19937 g(0x6e11);
  long compared = 0;
  int cases = 0;
  for (const Form& f : kForms) {
    for (uint32_t N : {8u, 24u, 64u, 256u}) {
      if (is_int(f.a) && (N == 40 || N == 56)) continue;
      for (int trial = 0; trial < 6; ++trial) {
        Case c{&f, N, trial % 2 == 1, trial >= 4 && !(f.a == T::TF32 && false), trial % 3 == 2, (trial % 2 == 0 || trial >= 3) && f.acc != Acc::S32, trial % 3 == 1 && f.acc != Acc::S32,   // (A in registers with a negated A too: trials 3 and 5)
               trial % 4 == 1, trial % 4 == 3};
        const bool sixteen = f.a == T::F16 || f.a == T::BF16;
        if (!sixteen || c.a_regs) c.tr_a = false;
        if (!sixteen) c.tr_b = false;
        if (c.sparse && f.a != T::F16 && f.a != T::TF32 && f.a != T::E4M3 && f.a != T::S8) c.sparse = false;
        // (sparse + A from registers + 8-bit is fine too)
        const uint32_t K = c.sparse ? 2 * f.K : f.K;
        const uint32_t Kp = c.sparse ? f.K : K;
        const uint32_t ea = ebytes(f.a), eb = ebytes(f.b);
        std::vector<uint8_t> img(kSmem);
        const bool wide = trial % 2 == 0;
        // A's bytes and B's bytes are not separate regions (the descriptors may overlap them); fill by the A type, then the B type over the odd 64-byte rows
        fill_elements(img, f.a, g, wide);
        if (ebytes(f.b) == ea && f.b != f.a) {
          std::vector<uint8_t> second(kSmem);
          fill_elements(second, f.b, g, wide);
          for (size_t i = 0; i < img.size(); i += 128) std::memcpy(&img[i + 64], &second[i + 64], 64);
        }
        const uint64_t da = c.a_regs ? 0 : pick_desc(g, !c.tr_a, ea, 64, Kp);
        const uint64_t db = pick_desc(g, !c.tr_b, eb, c.N, K);
        std::vector<uint32_t> aregs(128 * 4);
        for (uint32_t& w : aregs) {
          switch (f.a) {
            case T::F16: w = half_class(g, wide) | uint32_t{half_class(g, wide)} << 16; break;
            case T::BF16: w = (f32_class(g, wide) >> 16) | (f32_class(g, wide) & 0xffff0000u); break;
            case T::TF32: w = f32_class(g, wide); break;
            default: w = g();
          }
        }
        const uint32_t dregs = f.acc == Acc::F16 ? N / 4 : N / 2;
        std::vector<uint32_t> din(size_t{128} * dregs);
        for (uint32_t& w : din) {
          const uint32_t h = g();
          switch (f.acc) {
            case Acc::F32: w = f32_class(g, wide); break;
            case Acc::F16: w = half_class(g, wide) | uint32_t{half_class(g, wide)} << 16; break;
            default: w = h % 4 == 0 ? 0x7fffffffu - (h >> 8) % 100000 : h % 4 == 1 ? 0x80000000u + (h >> 8) % 100000 : g();
          }
        }
        // Sparse metadata: random nibbles; the engine's placement rule decides what they mean (the same for both routes).
        const uint32_t meta = c.sparse ? g() : 0;
        const uint32_t sel = c.sparse && (f.a == T::F16 || f.a == T::TF32) ? g() % 2 : 0;
        setenv("VGPU_FASTPATH", "1", 1);
        std::vector<uint32_t> fast, slow;
        try {
          fast = run(c, img, aregs, din, meta, sel, da, db);
          setenv("VGPU_FASTPATH", "0", 1);
          slow = run(c, img, aregs, din, meta, sel, da, db);
        } catch (const Error& e) {
          unsetenv("VGPU_FASTPATH");
          throw vtest::Failure(std::string(f.name) + " N " + std::to_string(N) + " trial " + std::to_string(trial) + ": " + e.what());
        }
        unsetenv("VGPU_FASTPATH");
        ++cases;
        for (size_t i = 0; i < fast.size(); ++i) {
          ++compared;
          if (fast[i] != slow[i]) {
            char buf[300];
            std::snprintf(buf, sizeof buf, "%s N %u trial %d%s%s: word %zu (thread %zu): route 0x%08x, general 0x%08x", f.name, N, trial,
                          c.sparse ? " sparse" : "", c.a_regs ? " A in registers" : "", i, i / dregs, fast[i], slow[i]);
            throw vtest::Failure(buf);
          }
        }
      }
    }
  }
  VCHECK(cases > 150);
  VCHECK(compared > 500000);
}

VTEST_MAIN
