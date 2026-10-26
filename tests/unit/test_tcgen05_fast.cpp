// tcgen05.mma reading its A and B operands whole from shared memory (exec::tcgen05_mma through Tcgen05Host::smem_direct) against the
// element-by-element read it replaces (VGPU_FASTPATH=0 makes the PTX engine's host decline a direct read), bit for bit, for every kind
// with 8-, 16- and 32-bit elements -- f16, bf16, tf32, e4m3, e5m2, u8, s8 -- at M = 64 and 128, K-major and MN-major operands in every
// swizzle with random leading/stride offsets, negated inputs, saturation, accumulation on and off, and operands that break arithmetic
// (zeros of both signs, subnormals, infinities, NaNs, the largest values, tf32 words with their low 13 bits set).
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

// atype/btype codes of Table 51 by kind: f16 (0 f16, 1 bf16), tf32 (2), f8f6f4 (0 e4m3, 1 e5m2), i8 (0 u8, 1 s8)
struct Kind {
  const char* name;       // as tcgen05.mma spells it
  uint32_t eb;            // bytes an element
  uint32_t K;
  uint32_t atype_n;       // how many type codes
  uint32_t tf32_or_half;  // 0 half-like, 1 tf32, 2 fp8, 3 int8
};
const Kind kKinds[] = {{"f16", 2, 16, 2, 0}, {"tf32", 4, 8, 1, 1}, {"f8f6f4", 1, 32, 2, 2}, {"i8", 1, 32, 2, 3}};

void fill(std::vector<uint8_t>& img, const Kind& k, uint32_t type, std::mt19937& g, bool wide) {
  for (size_t at = 0; at + k.eb <= img.size(); at += k.eb) {
    uint32_t v;
    switch (k.tf32_or_half) {
      case 0: v = type == 1 ? f32_class(g, wide) >> 16 : half_class(g, wide); break;
      case 1: v = f32_class(g, wide); break;
      default: v = g() & 0xff;
    }
    std::memcpy(&img[at], &v, k.eb);
  }
}

// Table 49: start, leading and stride byte offsets, the version bit, the swizzle code (0 none, 2 128B, 4 64B, 6 32B).
uint64_t desc_bits(uint32_t start, uint32_t lbo, uint32_t sbo, uint32_t code) {
  return uint64_t{(start & 0x3FFFF) >> 4} | (uint64_t{(lbo & 0x3FFFF) >> 4} << 16) | (uint64_t{(sbo & 0x3FFFF) >> 4} << 32) | (uint64_t{1} << 46) |
         (uint64_t{code} << 61);
}

// A descriptor whose rows x K operand lies inside the 8 KiB image.
uint64_t pick_desc(std::mt19937& g, bool k_major, uint32_t eb, uint32_t rows, uint32_t K) {
  for (;;) {
    const uint32_t which = g() % 4;
    const uint32_t code = (uint32_t[]){0, 2, 4, 6}[which];
    exec::WgmmaDesc d;
    d.swizzle = (uint32_t[]){0, 128, 64, 32}[which];
    d.start = 16 * (g() % 64);
    d.lbo = 16 * (g() % 160);
    d.sbo = 16 * (g() % 160);
    uint64_t hi = 0;
    for (uint32_t mn = 0; mn < rows; ++mn)
      for (uint32_t k = 0; k < K; ++k) hi = std::max(hi, exec::wgmma_smem_offset(d, k_major, eb, mn, k));
    if (hi + eb <= 8192) return desc_bits(d.start, d.lbo, d.sbo, code);
  }
}

struct Case {
  const Kind* kind;
  uint32_t atype, btype, dtype, M, N;
  bool neg_a, neg_b, trans_a, trans_b, sat, enable_d;
};

std::vector<uint32_t> run(const Case& c, const std::vector<uint8_t>& sa, const std::vector<uint8_t>& sb, const std::vector<uint32_t>& din, uint64_t da,
                          uint64_t db) {
  const uint32_t cols = c.N;
  const uint32_t id = (c.sat ? 1u << 3 : 0) | c.dtype << 4 | c.atype << 7 | c.btype << 10 | uint32_t(c.neg_a) << 13 | uint32_t(c.neg_b) << 14 |
                      uint32_t(c.trans_a) << 15 | uint32_t(c.trans_b) << 16 | uint32_t(c.N >> 3) << 17 | uint32_t(c.M >> 4) << 24;
  uint32_t alloc = 32;
  while (alloc < cols) alloc *= 2;   // (a power of two from 32)
  std::string st, readback;
  for (uint32_t col = 0, n = 0; col < cols; col += n) {
    n = 32;
    while (n > cols - col) n /= 2;   // (a power of two: the shapes of tcgen05.ld/st are x1 .. x128)
    std::string regs;
    for (uint32_t i = 0; i < n; ++i) regs += (i ? ", " : "") + std::string("%v") + std::to_string(i);
    for (uint32_t i = 0; i < n; ++i)
      st += "    ld.global.u32 %v" + std::to_string(i) + ", [%rd21+" + std::to_string(4 * (col + i)) + "];\n";
    st += "    add.u32 %r60, %r40, " + std::to_string(col) + ";\n    add.u32 %r60, %r60, %r41;\n";
    st += "    tcgen05.st.sync.aligned.32x32b.x" + std::to_string(n) + ".b32 [%r60], {" + regs + "};\n";
    readback += "    add.u32 %r60, %r40, " + std::to_string(col) + ";\n    add.u32 %r60, %r60, %r41;\n";
    readback += "    tcgen05.ld.sync.aligned.32x32b.x" + std::to_string(n) + ".b32 {" + regs + "}, [%r60];\n    tcgen05.wait::ld.sync.aligned;\n";
    for (uint32_t i = 0; i < n; ++i) readback += "    st.global.u32 [%rd20+" + std::to_string(4 * (col + i)) + "], %v" + std::to_string(i) + ";\n";
  }
  const std::string ptx = std::string(".version 8.7\n.target sm_100a\n.address_size 64\n") + R"(
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pd, .param .u64 pdin, .param .u64 da, .param .u64 db, .param .u32 id)
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
    ld.param.u64 %rd3, [pd];
    ld.param.u64 %rd4, [pdin];
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
    shl.b32 %r41, %r6, 21;
    mul.wide.u32 %rd10, %r1, )" + std::to_string(4 * cols) + R"(;
    add.u64 %rd20, %rd3, %rd10;
    add.u64 %rd21, %rd4, %rd10;
)" + st + R"(
    tcgen05.wait::st.sync.aligned;
    tcgen05.fence::before_thread_sync;
    bar.sync 0;
    tcgen05.fence::after_thread_sync;
    shr.u32 %r8, %r2, 4;
    cvt.u64.u32 %rd12, %r8;
    add.u64 %rd5, %rd5, %rd12;
    add.u64 %rd6, %rd6, %rd12;
    add.u64 %rd6, %rd6, 512;
    mov.b32 %r61, %r40;
    setp.)" + (c.enable_d ? "eq" : "ne") + R"(.u32 %p7, %r1, %r1;
    @%p3 tcgen05.mma.cta_group::1.kind::)" + c.kind->name + R"( [%r61], %rd5, %rd6, %r50, {0, 0, 0, 0}, %p7;
    @%p3 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%r7], 1;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r7], 0;
    @!%p1 bra WAIT;
    tcgen05.fence::after_thread_sync;
)" + readback + R"(
    tcgen05.fence::before_thread_sync;
    bar.sync 0;
    tcgen05.fence::after_thread_sync;
    @%p2 tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r40, )" + std::to_string(alloc) + R"(;
    @%p2 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
    ret;
}
)";
  MemoryManager mem{1 << 22};
  DeviceProfile prof = load_gpu("nvidia/b200");
  auto m = ptx::parse(ptx);
  auto upload = [&](const void* p, size_t n) {
    const uint64_t va = mem.alloc(n ? n : 16);
    if (n) mem.write(va, p, n);
    return va;
  };
  const uint64_t pa = upload(sa.data(), sa.size()), pb = upload(sb.data(), sb.size());
  std::vector<uint32_t> zero(size_t{128} * cols, 0);
  const uint64_t pd = upload(zero.data(), zero.size() * 4), pdin = upload(din.data(), din.size() * 4);
  LaunchConfig cfg;
  cfg.block = {128, 1, 1};
  cfg.grid = {1, 1, 1};
  cfg.cluster = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(pa), arg_u64(pb), arg_u64(pd), arg_u64(pdin), arg_u64(da), arg_u64(db), arg_u32(id)}, mem, prof);
  std::vector<uint32_t> out(size_t{128} * cols);
  mem.read(pd, out.data(), out.size() * 4);
  return out;
}

}  // namespace

VTEST(tcgen05_mma_whole_operand_reads_match_the_element_reads) {
  std::mt19937 g(0x7cc5);
  long compared = 0;
  int cases = 0;
  for (const Kind& kind : kKinds)
    for (int trial = 0; trial < 14; ++trial) {
      Case c{};
      c.kind = &kind;
      c.atype = kind.tf32_or_half == 1 ? 2 : g() % kind.atype_n;
      c.btype = kind.tf32_or_half == 0 ? c.atype : kind.tf32_or_half == 1 ? 2 : g() % kind.atype_n;
      const bool is_int = kind.tf32_or_half == 3;
      c.dtype = is_int ? 2 : ((kind.tf32_or_half == 1 || (kind.tf32_or_half == 0 && trial % 2 == 0)) ? 1 : g() % 2);
      if (kind.tf32_or_half == 0 && c.dtype == 0 && c.atype != 0) c.dtype = 1;   // an f16 accumulator takes f16 operands only
      c.M = trial % 3 == 0 ? 64 : 128;
      const uint32_t ns[] = {8, 16, 24, 64, 136, 256};
      c.N = ns[g() % 6];
      if (is_int && c.M == 128 && c.N % 16) c.N += 8;
      c.neg_a = !is_int && trial % 2 == 1;
      c.neg_b = !is_int && trial % 3 == 1;
      const bool sixteen = kind.eb == 2;
      c.trans_a = sixteen && trial % 4 >= 2;
      c.trans_b = sixteen && trial % 5 >= 2;
      c.sat = is_int && trial % 2 == 0;
      c.enable_d = trial % 4 != 3;
      const bool wide = trial % 2 == 0;
      std::vector<uint8_t> sa(8192), sb(8192);
      fill(sa, kind, c.atype, g, wide);
      fill(sb, kind, c.btype, g, wide);
      const uint64_t da = pick_desc(g, !c.trans_a, kind.eb, c.M, kind.K), db = pick_desc(g, !c.trans_b, kind.eb, c.N, kind.K);
      std::vector<uint32_t> din(size_t{128} * c.N);
      for (uint32_t& w : din) {
        const uint32_t h = g();
        w = c.dtype == 2 ? (h % 4 == 0 ? 0x7fffffffu - (h >> 8) % 100000 : h % 4 == 1 ? 0x80000000u + (h >> 8) % 100000 : g())
                         : c.dtype == 0 ? half_class(g, wide) : f32_class(g, wide);
      }
      setenv("VGPU_FASTPATH", "1", 1);
      std::vector<uint32_t> fast, slow;
      try {
        fast = run(c, sa, sb, din, da, db);
        setenv("VGPU_FASTPATH", "0", 1);
        slow = run(c, sa, sb, din, da, db);
      } catch (const Error& e) {
        unsetenv("VGPU_FASTPATH");
        throw vtest::Failure(std::string(kind.name) + " trial " + std::to_string(trial) + ": " + e.what());
      }
      unsetenv("VGPU_FASTPATH");
      ++cases;
      for (size_t i = 0; i < fast.size(); ++i) {
        ++compared;
        if (fast[i] != slow[i]) {
          char buf[300];
          std::snprintf(buf, sizeof buf, "%s trial %d M %u N %u: word %zu (lane %zu): whole read 0x%08x, element read 0x%08x", kind.name, trial, c.M, c.N, i,
                        i / c.N, fast[i], slow[i]);
          throw vtest::Failure(buf);
        }
      }
    }
  VCHECK(cases >= 56);
  VCHECK(compared > 400000);
}

VTEST_MAIN
