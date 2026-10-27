// tcgen05.mma through the tile kernels for the forms the whole-operand route (test_tcgen05_fast) leaves to the row loop: an f16 D, the data-path
// layouts that spread a row of D over several lanes (M = 64 from lane 0 or 16; .ws at M = 64 and 32), A from Tensor Memory, sparse A, and the
// block-scaled kinds (mxf8f6f4, mxf4, mxf4nvf4, UE8M0 and UE4M3 factors), against the row loop (VGPU_FASTPATH=0), bit for bit over random operands
// that include zeros of both signs, subnormals, infinities, NaNs and the extremes, scale factors from near 1 to 0xFF (the UE8M0 NaN), accumulation
// on and off, a scale-d, saturation, negated and transposed operands, and D, A, metadata and scale-factor cells of random bits.
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

enum class Data { Half, Bf16, Tf32, Byte };   // how the operand images are drawn

struct Kind {
  const char* name;    // tcgen05.mma's spelling after kind::
  const char* mods;    // after it (".block_scale.scale_vec::4X")
  bool mx;
  uint32_t K;          // dense
  uint32_t eb;         // bytes an element takes in shared memory (4-bit packed: 1 for the byte bounds)
  Data data;
  uint32_t cbits;      // an element's container in Tensor Memory
  uint32_t sv;         // scale factors a row (block scaling)
  uint32_t scale_type; // the instruction descriptor's scale type field
  uint32_t atypes[3];  // the codes drawn from (0 terminated by ~0u)
  bool int_kind;
};
const Kind kKinds[] = {
    {"f16", "", false, 16, 2, Data::Half, 16, 0, 0, {0, 1, ~0u}, false},
    {"tf32", "", false, 8, 4, Data::Tf32, 32, 0, 0, {2, ~0u, ~0u}, false},
    {"f8f6f4", "", false, 32, 1, Data::Byte, 8, 0, 0, {0, 1, 5}, false},
    {"i8", "", false, 32, 1, Data::Byte, 8, 0, 0, {0, 1, ~0u}, true},
    {"mxf8f6f4", ".block_scale.scale_vec::1X", true, 32, 1, Data::Byte, 8, 1, 1, {0, 1, 3}, false},
    {"mxf4", ".block_scale.scale_vec::2X", true, 64, 1, Data::Byte, 4, 2, 1, {1, ~0u, ~0u}, false},
    {"mxf4nvf4", ".block_scale.scale_vec::4X", true, 64, 1, Data::Byte, 4, 4, 0, {1, ~0u, ~0u}, false},
    {"mxf4nvf4", ".block_scale.scale_vec::2X", true, 64, 1, Data::Byte, 4, 2, 1, {1, ~0u, ~0u}, false},
};

struct Layout {
  const char* name;
  uint32_t M;
  bool ws;
  uint32_t d_lane0;   // 0 or 16 (M = 64 without .ws)
  uint32_t parts;     // D columns / N
};
const Layout kLayouts[] = {{"M128", 128, false, 0, 1}, {"M64 lane0", 64, false, 0, 1}, {"M64 lane16", 64, false, 16, 1}, {"ws M64", 64, true, 0, 2}, {"ws M32", 32, true, 0, 4}};

uint64_t desc_bits(uint32_t start, uint32_t lbo, uint32_t sbo, uint32_t code) {
  return uint64_t{(start & 0x3FFFF) >> 4} | (uint64_t{(lbo & 0x3FFFF) >> 4} << 16) | (uint64_t{(sbo & 0x3FFFF) >> 4} << 32) | (uint64_t{1} << 46) | (uint64_t{code} << 61);
}
// A descriptor whose rows x row_bytes operand (K-major, byte addresses) lies inside the 8 KiB image.
uint64_t pick_desc(std::mt19937& g, bool k_major, uint32_t eb, uint32_t rows, uint32_t cols) {
  for (;;) {
    const uint32_t which = g() % 4;
    exec::WgmmaDesc d;
    d.swizzle = (uint32_t[]){0, 128, 64, 32}[which];
    d.start = 16 * (g() % 64);
    d.lbo = 16 * (g() % 160);
    d.sbo = 16 * (g() % 160);
    uint64_t hi = 0;
    for (uint32_t mn = 0; mn < rows; ++mn)
      for (uint32_t k = 0; k < cols; ++k) hi = std::max(hi, exec::wgmma_smem_offset(d, k_major, eb, mn, k));
    if (hi + eb <= 8192) return desc_bits(d.start, d.lbo, d.sbo, (uint32_t[]){0, 2, 4, 6}[which]);
  }
}

struct Case {
  const Kind* kind;
  const Layout* lay;
  uint32_t atype, btype, dtype, N;
  bool sparse, sat, accumulate, a_tmem, neg_a, neg_b, trans_a, trans_b, scale_d;
  uint32_t sel, sfa_id, sfb_id;
};

constexpr uint32_t kDCols = 256, kAuxCols = 256;   // the cells of D, and of A / metadata / scale factors, in the 512-column allocation
constexpr uint32_t kACol = kDCols, kSfaCol = kDCols + 128, kSfbCol = kDCols + 136, kMetaCol = kDCols + 144;

std::string st_rows(const char* src_reg, uint32_t ncols, uint32_t col_base) {
  std::string s;
  for (uint32_t c = 0; c < ncols; c += 32) {
    const uint32_t n = std::min(32u, ncols - c);
    std::string regs;
    for (uint32_t i = 0; i < n; ++i) regs += (i ? ", " : "") + std::string("%v") + std::to_string(i);
    for (uint32_t i = 0; i < n; ++i) s += "    ld.global.u32 %v" + std::to_string(i) + ", [" + src_reg + "+" + std::to_string(4 * (c + i)) + "];\n";
    s += "    add.u32 %r60, %r40, " + std::to_string(col_base + c) + ";\n    add.u32 %r60, %r60, %r41;\n";
    s += "    tcgen05.st.sync.aligned.32x32b.x" + std::to_string(n) + ".b32 [%r60], {" + regs + "};\n";
  }
  return s;
}

// Runs one case; D's cells (128 lanes x kDCols words) afterwards.
std::vector<uint32_t> run(const Case& c, const std::vector<uint8_t>& sa, const std::vector<uint8_t>& sb, const std::vector<uint32_t>& dcells,
                          const std::vector<uint32_t>& aux, uint64_t da, uint64_t db, uint32_t id, uint32_t dlane) {
  std::string readback;
  for (uint32_t col = 0; col < kDCols; col += 32) {
    std::string regs;
    for (uint32_t i = 0; i < 32; ++i) regs += (i ? ", " : "") + std::string("%v") + std::to_string(i);
    readback += "    add.u32 %r60, %r40, " + std::to_string(col) + ";\n    add.u32 %r60, %r60, %r41;\n";
    readback += "    tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + regs + "}, [%r60];\n    tcgen05.wait::ld.sync.aligned;\n";
    for (uint32_t i = 0; i < 32; ++i) readback += "    st.global.u32 [%rd20+" + std::to_string(4 * (col + i)) + "], %v" + std::to_string(i) + ";\n";
  }
  const Kind& k = *c.kind;
  const std::string a_operand = c.a_tmem ? "[%r62]" : "%rd5";
  const std::string mma = std::string("tcgen05.mma") + (c.lay->ws ? ".ws" : "") + (c.sparse ? ".sp" : "") + ".cta_group::1.kind::" + k.name + k.mods + " [%r61], " + a_operand +
                          ", %rd6, " + (c.sparse ? "[%r65], " : "") + "%r50, " + (k.mx ? "[%r63], [%r64], " : c.lay->ws ? "" : "{0, 0, 0, 0}, ") + "%p7" + (c.scale_d ? ", 3" : "");
  const std::string ptx = std::string(".version 8.7\n.target sm_100a\n.address_size 64\n") + R"(
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pd, .param .u64 pdin, .param .u64 paux, .param .u64 da, .param .u64 db, .param .u32 id)
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
    ld.param.u64 %rd15, [paux];
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
    @%p2 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [slot], 512;
    setp.eq.u32 %p3, %r1, 0;
    mov.u32 %r7, bar;
    @%p3 mbarrier.init.shared::cta.b64 [%r7], 1;
    fence.proxy.async.shared::cta;
    bar.sync 0;
    ld.shared.u32 %r40, [slot];
    shl.b32 %r41, %r6, 21;
    mul.wide.u32 %rd10, %r1, )" + std::to_string(4 * kDCols) + R"(;
    add.u64 %rd20, %rd3, %rd10;
    add.u64 %rd21, %rd4, %rd10;
    mul.wide.u32 %rd11, %r1, )" + std::to_string(4 * kAuxCols) + R"(;
    add.u64 %rd22, %rd15, %rd11;
)" + st_rows("%rd21", kDCols, 0) + st_rows("%rd22", kAuxCols, kACol) + R"(
    tcgen05.wait::st.sync.aligned;
    tcgen05.fence::before_thread_sync;
    bar.sync 0;
    tcgen05.fence::after_thread_sync;
    shr.u32 %r8, %r2, 4;
    cvt.u64.u32 %rd12, %r8;
    add.u64 %rd5, %rd5, %rd12;
    add.u64 %rd6, %rd6, %rd12;
    add.u64 %rd6, %rd6, 512;
    add.u32 %r61, %r40, )" + std::to_string(dlane << 16) + R"(;
    add.u32 %r62, %r40, )" + std::to_string(kACol | (dlane << 16)) + R"(;
    add.u32 %r63, %r40, )" + std::to_string(kSfaCol) + R"(;
    add.u32 %r64, %r40, )" + std::to_string(kSfbCol) + R"(;
    add.u32 %r65, %r40, )" + std::to_string(kMetaCol | (dlane << 16)) + R"(;
    setp.)" + (c.accumulate ? "eq" : "ne") + R"(.u32 %p7, %r1, %r1;
    @%p3 )" + mma + R"(;
    @%p3 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%r7], 1;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r7], 0;
    @!%p1 bra WAIT;
    tcgen05.fence::after_thread_sync;
)" + readback + R"(
    tcgen05.fence::before_thread_sync;
    bar.sync 0;
    tcgen05.fence::after_thread_sync;
    @%p2 tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r40, 512;
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
  std::vector<uint32_t> zero(size_t{128} * kDCols, 0);
  const uint64_t pd = upload(zero.data(), zero.size() * 4), pdin = upload(dcells.data(), dcells.size() * 4), paux = upload(aux.data(), aux.size() * 4);
  LaunchConfig cfg;
  cfg.block = {128, 1, 1};
  cfg.grid = {1, 1, 1};
  cfg.cluster = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(pa), arg_u64(pb), arg_u64(pd), arg_u64(pdin), arg_u64(paux), arg_u64(da), arg_u64(db), arg_u32(id)}, mem, prof);
  std::vector<uint32_t> out(size_t{128} * kDCols);
  mem.read(pd, out.data(), out.size() * 4);
  return out;
}

void fill_image(std::vector<uint8_t>& img, const Kind& k, std::mt19937& g, bool wide) {
  for (size_t at = 0; at + k.eb <= img.size(); at += k.eb) {
    uint32_t v;
    switch (k.data) {
      case Data::Half: v = half_class(g, wide); break;
      case Data::Bf16: v = f32_class(g, wide) >> 16; break;
      case Data::Tf32: v = f32_class(g, wide); break;
      default: v = g() & 0xff;
    }
    std::memcpy(&img[at], &v, k.eb);
  }
}

}  // namespace

VTEST(tcgen05_tile_route_matches_the_row_loop) {
  std::mt19937 g(0x7c05);
  long compared = 0;
  int cases = 0;
  for (const Kind& kind : kKinds)
    for (const Layout& lay : kLayouts)
      for (int trial = 0; trial < 6; ++trial) {
        Case c{};
        c.kind = &kind;
        c.lay = &lay;
        const uint32_t ntypes = kind.atypes[1] == ~0u ? 1 : kind.atypes[2] == ~0u ? 2 : 3;
        c.atype = kind.atypes[g() % ntypes];
        c.btype = kind.mx || kind.atypes[1] == ~0u ? (kind.name == std::string("f16") ? c.atype : kind.atypes[g() % ntypes]) : kind.atypes[g() % ntypes];
        if (std::string(kind.name) == "f16") {
          if (trial == 2 || trial == 4) c.atype = 0;   // (an f16 D takes f16 operands: trials 2 and 4 make one, with a scale-d at 2 and sparse at 4)
          c.btype = c.atype;
        }
        if (std::string(kind.name) == "tf32") c.btype = 2;
        if (kind.mx && kind.sv == 1) c.btype = kind.atypes[g() % ntypes];
        // (mxf4 kinds: fp4 only)
        if (kind.cbits == 4) c.atype = c.btype = 1;
        c.dtype = kind.int_kind ? 2 : kind.mx ? 1 : std::string(kind.name) == "tf32" ? 1 : (std::string(kind.name) == "f16" && c.atype != 0) ? 1 : trial % 2;
        c.N = std::vector<uint32_t>{8, 16, 24, 64, 128, 136, 256}[g() % 7];
        // Shapes of Table 48: .ws takes N of 64, 128 or 256 (not 256 sparse); a block-scaled MMA is one CTA at M = 128.
        if (lay.ws) c.N = trial % 2 ? 64 : 128;
        if (kind.mx && (lay.M != 128 || lay.ws)) continue;
        if (kind.int_kind && lay.M == 128 && c.N % 16) c.N += 8;
        c.sparse = trial >= 3 && !(kind.cbits == 4) && !(kind.mx && kind.sv != 1) && !(lay.ws && lay.M < 128);
        c.sat = kind.int_kind && trial % 2 == 0;
        c.accumulate = trial % 4 != 3;
        c.a_tmem = !lay.ws && trial % 3 == 1 && !kind.mx;
        const bool sixteen = kind.eb == 2;
        c.neg_a = !kind.int_kind && !kind.mx && trial % 2 == 1;
        c.neg_b = !kind.int_kind && !kind.mx && trial % 3 == 1;
        c.trans_a = sixteen && !c.a_tmem && trial % 4 >= 2;
        c.trans_b = sixteen && trial % 5 >= 2;
        c.scale_d = trial % 3 == 2 && (std::string(kind.name) == "f16" || std::string(kind.name) == "tf32");
        c.sel = c.sparse && (std::string(kind.name) == "f16" || std::string(kind.name) == "tf32") ? trial % 2 : 0;
        // Scale factors: one block's factors (sv of them) in byte sf_id + j of a cell; mxf4's need an even id, K = 64 nvf4 with 4 factors id 0.
        c.sfa_id = kind.sv == 4 ? 0 : kind.sv == 2 ? 2 * (trial % 2) : trial % 4;
        c.sfb_id = kind.sv == 4 ? 0 : kind.sv == 2 ? 2 * (trial % 2) : trial % 4;
        if (c.sparse) c.N = std::min(c.N, 128u);   // (.ws.sp takes N up to 128)
        if (lay.ws && c.sparse && c.N > 128) c.N = 128;
        const uint32_t Ke = c.sparse ? 2 * kind.K : kind.K, Ka = kind.K;
        uint32_t id;
        if (kind.mx) id = c.sfb_id << 4 | c.atype << 7 | c.btype << 10 | (c.N >> 3) << 17 | kind.scale_type << 23 | (lay.M >> 7) << 27 | c.sfa_id << 29 | (c.sparse ? 4u : 0u);
        else
          id = (c.sat ? 1u << 3 : 0) | c.dtype << 4 | c.atype << 7 | c.btype << 10 | uint32_t(c.neg_a) << 13 | uint32_t(c.neg_b) << 14 | uint32_t(c.trans_a) << 15 |
               uint32_t(c.trans_b) << 16 | (c.N >> 3) << 17 | (lay.M >> 4) << 24 | (c.sparse ? 4u | c.sel : 0u);
        // A (rows M) and B (columns N), K-major (or MN-major for the 16-bit kinds), K-byte rows.
        const uint32_t row_a = Ka * kind.eb * (kind.cbits == 4 ? 1 : 1) / (kind.cbits == 4 ? 2 : 1), row_b = Ke * kind.eb / (kind.cbits == 4 ? 2 : 1);
        std::vector<uint8_t> sa(8192), sb(8192);
        const bool wide = trial % 2 == 0;
        fill_image(sa, kind, g, wide);
        fill_image(sb, kind, g, wide);
        // (K-major: byte columns; MN-major, only the 16-bit kinds: element columns, two bytes an element along MN)
        const uint64_t da = c.trans_a ? pick_desc(g, false, kind.eb, lay.M, Ka) : pick_desc(g, true, 1, lay.M, row_a),
                       db = c.trans_b ? pick_desc(g, false, kind.eb, c.N, Ke) : pick_desc(g, true, 1, c.N, row_b);
        // The cells: D (the result's size per lane: N / parts columns), then A in Tensor Memory, the scale factors and the metadata.
        std::vector<uint32_t> dcells(size_t{128} * kDCols), aux(size_t{128} * kAuxCols);
        for (uint32_t& w : dcells) {
          const uint32_t h = g();
          w = c.dtype == 2 ? (h % 4 == 0 ? 0x7fffffffu - (h >> 8) % 100000 : h % 4 == 1 ? 0x80000000u + (h >> 8) % 100000 : g())
                           : c.dtype == 0 ? (half_class(g, wide) | (g() % 3 == 0 ? uint32_t{half_class(g, wide)} << 16 : 0u)) : f32_class(g, wide);
        }
        const bool extreme_scales = trial % 3 == 0;
        for (uint32_t& w : aux) w = g();
        if (kind.mx)
          for (uint32_t lane = 0; lane < 128; ++lane)
            for (uint32_t col = 0; col < 16; ++col) {
              uint32_t& w = aux[size_t{lane} * kAuxCols + (kSfaCol - kACol) + col];
              for (uint32_t b = 0; b < 4; ++b) {
                const uint32_t h = g();
                const uint32_t byte = extreme_scales && h % 17 == 0 ? (h >> 8) & 0xff : kind.scale_type == 0 ? 0x30 + (h >> 8) % 24 : 124 + (h >> 8) % 8;
                w = (w & ~(0xffu << (8 * b))) | byte << (8 * b);
              }
              // (the four lane partitions mostly hold the same factors, as CUTLASS's copy leaves them; every third trial they differ)
              if (trial % 3 != 2 && lane >= 32) w = aux[size_t{lane % 32} * kAuxCols + (kSfaCol - kACol) + col];
            }
        if (!kind.mx && kind.eb == 2 && c.a_tmem)
          for (uint32_t lane = 0; lane < 128; ++lane)
            for (uint32_t col = 0; col < 32; ++col) aux[size_t{lane} * kAuxCols + col] = half_class(g, wide) | uint32_t{half_class(g, wide)} << 16;
        std::vector<uint32_t> fast, slow;
        try {
          setenv("VGPU_FASTPATH", "1", 1);
          fast = run(c, sa, sb, dcells, aux, da, db, id, lay.d_lane0);
          setenv("VGPU_FASTPATH", "0", 1);
          slow = run(c, sa, sb, dcells, aux, da, db, id, lay.d_lane0);
        } catch (const Error& e) {
          unsetenv("VGPU_FASTPATH");
          throw vtest::Failure(std::string(kind.name) + kind.mods + " " + lay.name + " trial " + std::to_string(trial) + ": " + e.what());
        }
        unsetenv("VGPU_FASTPATH");
        ++cases;
        for (size_t i = 0; i < fast.size(); ++i) {
          ++compared;
          if (fast[i] != slow[i]) {
            char buf[300];
            std::snprintf(buf, sizeof buf, "%s%s %s trial %d M %u N %u sparse %d a_tmem %d dtype %u: word %zu (lane %zu col %zu): tile route 0x%08x, row loop 0x%08x", kind.name, kind.mods,
                          lay.name, trial, lay.M, c.N, int(c.sparse), int(c.a_tmem), c.dtype, i, i / kDCols, i % kDCols, fast[i], slow[i]);
            throw vtest::Failure(buf);
          }
        }
      }
  VCHECK(cases >= 100);
  VCHECK(compared > 3000000);
}

VTEST_MAIN
