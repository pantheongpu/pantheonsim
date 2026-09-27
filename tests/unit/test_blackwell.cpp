// Blackwell (sm_100a): Tensor Memory and the fifth-generation tensor core.
//
// The expected layouts here are written from the PTX ISA's own statements of
// them, in a different form from the interpreter's formulas: the tcgen05.ld/st
// fragments as the tables of figures 186-190 (thread and register for each
// lane and column), the shared-memory operands as the CuTe canonical layouts of
// section 9.7.18.3.3, and D's placement from the data-path layout figures of
// 9.7.18.10.5 (for M = 128 with a CTA pair, the upper half of N in lanes
// 64-127). The products are checked against a plain host GEMM.
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/tensormap.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

const char* kHeader100a = ".version 8.7\n.target sm_100a\n.address_size 64\n";

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
uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
float bits_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
// Exact for the small multiples of 1/2 used here.
uint16_t f16_bits(float f) {
  if (f == 0.0f) return 0;
  int e = 0;
  const float m = std::frexp(std::fabs(f), &e);
  const uint16_t mant = static_cast<uint16_t>((m * 2.0f - 1.0f) * 1024.0f);
  return static_cast<uint16_t>((f < 0 ? 0x8000 : 0) | ((e - 1 + 15) << 10) | mant);
}
float f16_value(uint16_t h) {
  const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
  const float v = e ? std::ldexp(1.0f + m / 1024.0f, e - 15) : std::ldexp(m / 1024.0f, -14);
  return (h & 0x8000) ? -v : v;
}
uint16_t bf16_bits(float f) { return static_cast<uint16_t>(f32_bits(f) >> 16); }
// E4M3 (bias 7, three mantissa bits), for 0 and +-{0.5, 1, 1.5, 2}.
uint8_t e4m3_bits(float f) {
  if (f == 0.0f) return 0;
  int e = 0;
  const float m = std::frexp(std::fabs(f), &e);
  const uint8_t mant = static_cast<uint8_t>((m * 2.0f - 1.0f) * 8.0f);
  return static_cast<uint8_t>((f < 0 ? 0x80 : 0) | ((e - 1 + 7) << 3) | mant);
}

// A small multiple of 1/2 in [-2, 2], different for each (i, j).
float val(int i, int j, int salt) { return static_cast<float>(((i * 7 + j * 13 + salt) % 9) - 4) * 0.5f; }

// ---- operand images ----------------------------------------------------------

// Byte offset of element (mn, k) in the canonical layouts of 9.7.18.3.3,
// written as the ISA gives them in CuTe terms, in elements (T per 16 bytes),
// then Swizzle<B,4,3> on the bytes. sbo/lbo are in bytes.
enum class Major { K, MN };
// atom = 32 is the 128-byte swizzle in 32-byte atoms: Swizzle<2,5,2>.
uint32_t canonical(Major major, int swizzle_bytes, int eb, uint32_t lbo, uint32_t sbo, int mn, int k,
                   int atom = 16) {
  const int T = 16 / eb;
  const uint32_t SBO = sbo / eb, LBO = lbo / eb;
  uint32_t e;
  if (major == Major::K) {
    if (swizzle_bytes == 0)   // ((8,m),(T,2k)):((1T,SBO),(1,LBO))
      e = (mn % 8) * T + (mn / 8) * SBO + (k % T) + (k / T) * LBO;
    else {                    // ((8,m),(T,2k)):((sT,SBO),(1,T)), s = swizzle/16
      const int s = swizzle_bytes / 16;
      e = (mn % 8) * s * T + (mn / 8) * SBO + (k % T) + (k / T) * T;
    }
  } else {
    if (swizzle_bytes == 0)   // ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
      e = (mn % T) + (mn / T) * SBO + (k % 8) * T + (k / 8) * LBO;
    else {                    // ((T,s,m),(8,k)):((1,T,LBO),(sT,SBO))
      const int s = swizzle_bytes / 16;
      e = (mn % T) + ((mn / T) % s) * T + (mn / (T * s)) * LBO + (k % 8) * s * T + (k / 8) * SBO;
    }
  }
  uint32_t byte = e * eb;
  if (atom == 32) return byte ^ (((byte >> 7) & 3u) << 5);   // Swizzle<2,5,2>
  const int b = swizzle_bytes == 128 ? 3 : swizzle_bytes == 64 ? 2 : swizzle_bytes == 32 ? 1 : 0;
  return byte ^ (((byte >> 7) & ((1u << b) - 1)) << 4);
}

// A shared-memory descriptor (Table 49), start relative to the operand buffer.
uint64_t desc(uint32_t start, uint32_t lbo, uint32_t sbo, int swizzle_bytes, int atom = 16) {
  const uint64_t code = atom == 32 ? 1 : swizzle_bytes == 128 ? 2 : swizzle_bytes == 64 ? 4 : swizzle_bytes == 32 ? 6 : 0;
  return uint64_t{(start & 0x3FFFF) >> 4} | (uint64_t{(lbo & 0x3FFFF) >> 4} << 16) |
         (uint64_t{(sbo & 0x3FFFF) >> 4} << 32) | (uint64_t{1} << 46) | (code << 61);
}

// An instruction descriptor (Table 51).
uint32_t idesc(uint32_t dtype, uint32_t atype, uint32_t btype, int M, int N, bool neg_a = false,
               bool neg_b = false, bool trans_a = false, bool trans_b = false, bool sat = false) {
  return (sat ? 1u << 3 : 0) | dtype << 4 | atype << 7 | btype << 10 | uint32_t(neg_a) << 13 |
         uint32_t(neg_b) << 14 | uint32_t(trans_a) << 15 | uint32_t(trans_b) << 16 |
         uint32_t(N >> 3) << 17 | uint32_t(M >> 4) << 24;
}

void put(std::vector<uint8_t>& img, uint32_t at, uint64_t bits, int eb) {
  if (img.size() < at + eb) img.resize(at + eb);
  std::memcpy(img.data() + at, &bits, eb);
}

// ---- a kernel that runs one tcgen05.mma ----------------------------------------

// One tcgen05.mma per CTA (pair), with its operands copied into shared memory
// (A at offset 0, B at 8192, each CTA of a pair its own images), D read back
// by every warp with tcgen05.ld.32x32b from its quarter of Tensor Memory:
// out[cta][lane][col] for the allocation's first `cols` columns.
struct Mma {
  std::string kind = "f16";
  int group = 1;
  uint32_t id = 0;
  uint64_t desc_a = 0, desc_b = 0;
  std::vector<std::vector<uint8_t>> smem_a, smem_b;   // per CTA
  int cols = 64;                                       // allocated and read back
  uint32_t d_col = 0;                                  // D's column in the allocation
  uint32_t d_lane = 0;
  std::string enable_d = "0";
  std::string scale_tail;                              // ", 3" for scale-input-d
  std::string mask = "{0, 0, 0, 0}";                   // disable-output-lane, or ""
  // D's cells before the MMA, per CTA, lane-major (cols per lane); empty: 0.
  std::vector<std::vector<uint32_t>> d_in;
  // A from Tensor Memory instead: per CTA, lane-major, a_cols 32-bit cells per
  // lane stored at column `a_col` of the allocation.
  std::vector<std::vector<uint32_t>> a_tmem;
  int a_cols = 0;
  uint32_t a_col = 0;
  // Whether a_tmem is A, or only other Tensor Memory contents (scale factors).
  bool a_is_tmem = true;
  // .block_scale: the modifiers after the kind (".block_scale.scale_vec::4X"),
  // and the scale matrices' columns in the allocation.
  std::string kind_mods;
  bool block_scale = false;
  uint32_t sfa_col = 0, sfb_col = 0;
  // tcgen05.mma.sp: the metadata's address in the allocation.
  bool sparse = false;
  uint32_t meta_col = 0, meta_lane = 0;
  std::string target = kHeader100a;

  std::vector<std::vector<uint32_t>> run() const {
    const int ctas = group;
    // Stores of the initial D and A cells, then the MMA, then the read back.
    auto st_rows = [&](const char* src_param_reg, int ncols, uint32_t col_base) {
      std::string s;
      for (int c = 0; c < ncols; c += 32) {
        const int n = std::min(32, ncols - c);
        std::string regs;
        for (int i = 0; i < n; ++i) regs += (i ? ", " : "") + std::string("%v") + std::to_string(i);
        for (int i = 0; i < n; ++i)
          s += "    ld.global.u32 %v" + std::to_string(i) + ", [" + src_param_reg + "+" +
               std::to_string(4 * (c + i)) + "];\n";
        s += "    add.u32 %r60, %r40, " + std::to_string(col_base + c) + ";\n";
        s += "    add.u32 %r60, %r60, %r41;\n";
        s += "    tcgen05.st.sync.aligned.32x32b.x" + std::to_string(n) + ".b32 [%r60], {" + regs + "};\n";
      }
      return s;
    };
    std::string readback;
    for (int c = 0; c < cols; c += 32) {
      const int n = std::min(32, cols - c);
      std::string regs;
      for (int i = 0; i < n; ++i) regs += (i ? ", " : "") + std::string("%v") + std::to_string(i);
      readback += "    add.u32 %r60, %r40, " + std::to_string(c) + ";\n";
      readback += "    add.u32 %r60, %r60, %r41;\n";
      readback += "    tcgen05.ld.sync.aligned.32x32b.x" + std::to_string(n) + ".b32 {" + regs + "}, [%r60];\n";
      readback += "    tcgen05.wait::ld.sync.aligned;\n";
      for (int i = 0; i < n; ++i)
        readback += "    st.global.u32 [%rd20+" + std::to_string(4 * (c + i)) + "], %v" + std::to_string(i) + ";\n";
    }
    const std::string g = std::to_string(group);
    const std::string a_operand = a_tmem.empty() || !a_is_tmem ? "%rd5" : "[%r62]";
    const std::string lanes_off = group == 2 && mask == "{0, 0, 0, 0}" ? "{0, 0, 0, 0, 0, 0, 0, 0}" : mask;
    const std::string ptx = target + R"(
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pd, .param .u64 pdin, .param .u64 painit,
                  .param .u64 da, .param .u64 db, .param .u32 id)
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
    ld.param.u64 %rd15, [painit];
    ld.param.u64 %rd5, [da];
    ld.param.u64 %rd6, [db];
    ld.param.u32 %r50, [id];
    mov.u32 %r1, %tid.x;
    mov.u32 %r30, %cluster_ctarank;
    mov.u32 %r2, smem;
    // This CTA's operand images: pa/pb + rank * 8192.
    mul.wide.u32 %rd7, %r30, 8192;
    add.u64 %rd1, %rd1, %rd7;
    add.u64 %rd2, %rd2, %rd7;
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
    shr.u32 %r6, %r1, 5;                       // warp
    setp.eq.u32 %p2, %r6, 0;
    @%p2 tcgen05.alloc.cta_group::)" + g + R"(.sync.aligned.shared::cta.b32 [slot], )" + std::to_string(cols < 32 ? 32 : cols) + R"(;
    setp.eq.u32 %p3, %r1, 0;
    mov.u32 %r7, bar;
    @%p3 mbarrier.init.shared::cta.b64 [%r7], 1;
    fence.proxy.async.shared::cta;
    bar.sync 0;
    barrier.cluster.arrive;
    barrier.cluster.wait;
    ld.shared.u32 %r40, [slot];
    // This warp's quarter: lane 32 * warp.
    shl.b32 %r41, %r6, 21;
    // This thread's row of the per-CTA arrays: (rank * 128 + tid) * 4 bytes * cols.
    mad.lo.u32 %r42, %r30, 128, %r1;
    mul.wide.u32 %rd10, %r42, )" + std::to_string(4 * cols) + R"(;
    add.u64 %rd20, %rd3, %rd10;
    add.u64 %rd21, %rd4, %rd10;
    mul.wide.u32 %rd11, %r42, )" + std::to_string(4 * std::max(a_cols, 1)) + R"(;
    add.u64 %rd22, %rd15, %rd11;
)" + (d_in.empty() ? std::string() : st_rows("%rd21", cols, 0)) +
                            (a_tmem.empty() ? std::string() : st_rows("%rd22", a_cols, a_col)) + R"(
    tcgen05.wait::st.sync.aligned;
    tcgen05.fence::before_thread_sync;
    bar.sync 0;
    barrier.cluster.arrive;
    barrier.cluster.wait;
    tcgen05.fence::after_thread_sync;
    // The descriptors' start fields are relative to the operand buffers.
    shr.u32 %r8, %r2, 4;
    cvt.u64.u32 %rd12, %r8;
    add.u64 %rd5, %rd5, %rd12;
    add.u64 %rd6, %rd6, %rd12;
    add.u64 %rd6, %rd6, 512;
    add.u32 %r61, %r40, )" + std::to_string(d_col | (d_lane << 16)) + R"(;
    add.u32 %r62, %r40, )" + std::to_string(a_col) + R"(;
    add.u32 %r63, %r40, )" + std::to_string(sfa_col) + R"(;
    add.u32 %r64, %r40, )" + std::to_string(sfb_col) + R"(;
    add.u32 %r65, %r40, )" + std::to_string(meta_col | (meta_lane << 16)) + R"(;
    setp.ne.u32 %p5, %r50, 0xFFFFFFFF;          // a predicate that is true
    setp.eq.u32 %p6, %r30, 0;
    and.pred %p4, %p3, %p6;                     // thread 0 of the even CTA issues
    setp.)" + (enable_d == "0" ? "ne" : "eq") + R"(.u32 %p7, %r1, %r1;
    @%p4 tcgen05.mma)" + (sparse ? ".sp" : "") + ".cta_group::" + g + ".kind::" + kind + kind_mods + " [%r61], " +
                            a_operand + ", %rd6, " + (sparse ? "[%r65], " : "") + "%r50, " +
                            (block_scale ? std::string("[%r63], [%r64], ") : lanes_off.empty() ? "" : lanes_off + ", ") +
                            "%p7" + scale_tail + R"(;
    @%p4 tcgen05.commit.cta_group::)" + g + R"(.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%r7], )" +
                            std::to_string((1 << ctas) - 1) + R"(;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r7], 0;
    @!%p1 bra WAIT;
    tcgen05.fence::after_thread_sync;
)" + readback + R"(
    tcgen05.fence::before_thread_sync;
    bar.sync 0;
    barrier.cluster.arrive;
    barrier.cluster.wait;
    tcgen05.fence::after_thread_sync;
    @%p2 tcgen05.dealloc.cta_group::)" + g + R"(.sync.aligned.b32 %r40, )" + std::to_string(cols < 32 ? 32 : cols) + R"(;
    @%p2 tcgen05.relinquish_alloc_permit.cta_group::)" + g + R"(.sync.aligned;
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
    std::vector<uint8_t> sa(size_t(8192) * ctas), sb(size_t(8192) * ctas);
    for (int c = 0; c < ctas; ++c) {
      if (c < int(smem_a.size())) std::memcpy(sa.data() + 8192 * c, smem_a[c].data(), std::min<size_t>(8192, smem_a[c].size()));
      if (c < int(smem_b.size())) std::memcpy(sb.data() + 8192 * c, smem_b[c].data(), std::min<size_t>(8192, smem_b[c].size()));
    }
    std::vector<uint32_t> din(size_t(ctas) * 128 * cols, 0), ain(size_t(ctas) * 128 * std::max(a_cols, 1), 0);
    for (int c = 0; c < int(d_in.size()); ++c)
      std::copy(d_in[c].begin(), d_in[c].end(), din.begin() + size_t(c) * 128 * cols);
    for (int c = 0; c < int(a_tmem.size()); ++c)
      std::copy(a_tmem[c].begin(), a_tmem[c].end(), ain.begin() + size_t(c) * 128 * a_cols);
    const uint64_t pa = upload(sa.data(), sa.size()), pb = upload(sb.data(), sb.size());
    const uint64_t pd = upload(din.data(), din.size() * 4);
    const uint64_t pdin = upload(din.data(), din.size() * 4), painit = upload(ain.data(), ain.size() * 4);
    LaunchConfig cfg;
    cfg.block = {128, 1, 1};
    cfg.grid = {static_cast<uint32_t>(ctas), 1, 1};
    cfg.cluster = {static_cast<uint32_t>(ctas), 1, 1};
    exec::launch(m.entries[0], cfg,
                 {arg_u64(pa), arg_u64(pb), arg_u64(pd), arg_u64(pdin), arg_u64(painit), arg_u64(desc_a),
                  arg_u64(desc_b), arg_u32(id)},
                 mem, prof);
    std::vector<uint32_t> all(size_t(ctas) * 128 * cols);
    mem.read(pd, all.data(), all.size() * 4);
    std::vector<std::vector<uint32_t>> out(ctas);
    for (int c = 0; c < ctas; ++c)
      out[c].assign(all.begin() + size_t(c) * 128 * cols, all.begin() + size_t(c + 1) * 128 * cols);
    return out;
  }
};

// Where D(m, n) of an M x N product lands, from the data-path layout figures:
// (CTA, lane, column relative to D's address).
struct Place { int cta, lane, col; };
Place d_place(int group, int M, int N, int m, int n, int lane0 = 0) {
  if (group == 1 && M == 128) return {0, m, n};                                // figure 217
  if (group == 1 && M == 64) return {0, (m / 16) * 32 + m % 16 + lane0, n};    // figures 221-222
  if (group == 2 && M == 256) return {m / 128, m % 128, n};                    // figures 211-212
  // M = 128 over a pair (figures 213-214): rows 0-63 in the even CTA, 64-127
  // in the odd; columns N/2.. in lanes 64-127 at the same column addresses.
  const int local = m % 64;
  return {m / 64, local + (n >= N / 2 ? 64 : 0), n % (N / 2)};
}

// Builds A (M x K) and B (N x K, i.e. B^T) images for a CTA group: CTA v holds
// rows [v*M/G, ...) of A and columns [v*N/G, ...) of B at the same offsets.
struct Operands {
  std::vector<std::vector<uint8_t>> a, b;
};
Operands images(int group, int M, int N, int K, int eb, Major amaj, Major bmaj, int swz_a, int swz_b,
                uint32_t lbo_a, uint32_t sbo_a, uint32_t lbo_b, uint32_t sbo_b,
                const std::function<uint64_t(int, int)>& Abits,
                const std::function<uint64_t(int, int)>& Bbits, int atom = 16) {
  Operands o;
  o.a.resize(group);
  o.b.resize(group);
  for (int v = 0; v < group; ++v) {
    for (int m = 0; m < M / group; ++m)
      for (int k = 0; k < K; ++k)
        put(o.a[v], canonical(amaj, swz_a, eb, lbo_a, sbo_a, m, k, atom), Abits(v * (M / group) + m, k), eb);
    for (int n = 0; n < N / group; ++n)
      for (int k = 0; k < K; ++k)
        put(o.b[v], canonical(bmaj, swz_b, eb, lbo_b, sbo_b, n, k, atom), Bbits(v * (N / group) + n, k), eb);
  }
  return o;
}

void check_d_f32(const std::vector<std::vector<uint32_t>>& out, int group, int M, int N, int K,
                 const std::function<float(int, int)>& A, const std::function<float(int, int)>& B,
                 const std::function<float(int, int)>& D0 = nullptr, int lane0 = 0, int cols = 64) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float want = D0 ? D0(m, n) : 0.0f;
      for (int k = 0; k < K; ++k) want += A(m, k) * B(k, n);
      const Place p = d_place(group, M, N, m, n, lane0);
      const float got = bits_f32(out[p.cta][size_t(p.lane) * cols + p.col]);
      if (got != want)
        VCHECK_EQ(std::to_string(m) + "," + std::to_string(n) + ": " + std::to_string(got),
                  std::to_string(m) + "," + std::to_string(n) + ": " + std::to_string(want));
    }
}

// The OCP MX small floats (e2m3, e3m2, e2m1: no infinities or NaNs), for the
// multiples of 1/2 in [-4, 4] used here; subnormal below 2^(1 - bias).
uint8_t mx_bits(float f, int ebits, int mbits, int bias) {
  if (f == 0.0f) return 0;
  const uint8_t sign = static_cast<uint8_t>(f < 0 ? 1u << (ebits + mbits) : 0);
  const float a = std::fabs(f);
  int e = 0;
  const float m = std::frexp(a, &e);   // a = m * 2^e, m in [0.5, 1)
  const int biased = e - 1 + bias;
  if (biased <= 0)
    return sign | static_cast<uint8_t>(std::ldexp(a, mbits - (1 - bias)));
  return sign | static_cast<uint8_t>(biased << mbits) |
         static_cast<uint8_t>((m * 2.0f - 1.0f) * float(1 << mbits));
}
uint8_t e2m1_bits(float f) { return mx_bits(f, 2, 1, 1); }
uint8_t e3m2_bits(float f) { return mx_bits(f, 3, 2, 3); }
uint8_t e2m3_bits(float f) { return mx_bits(f, 2, 3, 1); }

// A K-major operand of sub-byte elements without swizzling (Tables 58-60 and
// the cp formats .b6x16_p32 / .b4x16_p64): each row is K / per16 groups of 16
// bytes, element k at bit (k % per16) * bits of group k / per16, little-endian;
// the rows' bytes then go through the 1-byte canonical layout. per16 = 16
// leaves a 4-bit group's upper 8 bytes and a 6-bit group's upper 4 as padding;
// per16 = 32 is the packed fp4 of .kind::mxf4*.
std::vector<uint8_t> packed_k_major(int rows, int K, int bits, int per16, uint32_t lbo, uint32_t sbo,
                                    const std::function<uint64_t(int, int)>& elem) {
  std::vector<uint8_t> img;
  for (int r = 0; r < rows; ++r) {
    std::vector<uint8_t> row(size_t(16) * (K / per16), 0);
    for (int k = 0; k < K; ++k) {
      const int bit = 128 * (k / per16) + (k % per16) * bits;
      const uint64_t v = elem(r, k) & ((1u << bits) - 1);
      for (int b = 0; b < bits; ++b)
        if (v >> b & 1) row[(bit + b) / 8] |= static_cast<uint8_t>(1u << ((bit + b) % 8));
    }
    for (size_t i = 0; i < row.size(); ++i) put(img, canonical(Major::K, 0, 1, lbo, sbo, r, int(i)), row[i], 1);
  }
  return img;
}

// A block-scaled kind's instruction descriptor (Tables 52-53): SFB_ID in bits
// 4-5, the scale type from bit 23, M / 128 in bits 27-28, SFA_ID in bits 29-30.
uint32_t idesc_mx(uint32_t atype, uint32_t btype, int M, int N, uint32_t sfa_id, uint32_t sfb_id,
                  uint32_t scale_type) {
  return sfb_id << 4 | atype << 7 | btype << 10 | uint32_t(N >> 3) << 17 | scale_type << 23 |
         uint32_t(M >> 7) << 27 | sfa_id << 29;
}

// Scale factors in Tensor Memory (9.7.18.10.7.2-3), in the Mma kernel's
// auxiliary cells (a_tmem with a_is_tmem = false): row m's j-th factor in
// byte sf_id + j of lane m % 32, column col + m / 32, copied into each of the
// four 32-lane partitions. Other bytes get junk.
void put_scales(std::vector<uint32_t>& cells, int a_cols, int col, int rows, int per_row, int sf_id,
                const std::function<uint8_t(int, int)>& sf) {
  for (int m = 0; m < rows; ++m)
    for (int part = 0; part < 4; ++part) {
      uint32_t& c = cells[size_t(m % 32 + 32 * part) * a_cols + col + m / 32];
      for (int byte = 0; byte < 4; ++byte)
        if (byte < sf_id || byte >= sf_id + per_row) c = (c & ~(0xFFu << 8 * byte)) | (0xA5u << 8 * byte);
      for (int j = 0; j < per_row; ++j)
        c = (c & ~(0xFFu << 8 * (sf_id + j))) | uint32_t(sf(m, j)) << 8 * (sf_id + j);
    }
}

// One block-scaled product checked against the host: D(m, n) = sum over k of
// A(m, k) * SA(m, k / blk) * B(k, n) * SB(n, k / blk).
void check_scaled(const std::vector<std::vector<uint32_t>>& out, int M, int N, int K, int blk, int cols,
                  const std::function<float(int, int)>& A, const std::function<float(int, int)>& B,
                  const std::function<float(int, int)>& SA, const std::function<float(int, int)>& SB) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double want = 0;
      for (int k = 0; k < K; ++k) want += double(A(m, k)) * SA(m, k / blk) * B(k, n) * SB(n, k / blk);
      const float got = bits_f32(out[0][size_t(m) * cols + n]);
      if (got != float(want))
        VCHECK_EQ(std::to_string(m) + "," + std::to_string(n) + ": " + std::to_string(got),
                  std::to_string(m) + "," + std::to_string(n) + ": " + std::to_string(float(want)));
    }
}

// ---- sparse A ------------------------------------------------------------------

// Where the metadata field of (row, chunk) sits in one 32-lane partition, as
// figures 287-292 draw it: (lane, column, bit). For .kind::f16 and tf32
// (eight fields a row), lanes 0-7 hold rows 0-7's chunks 0-3 then rows 8-15's;
// lanes 8-15 the same rows' chunks 4-7; lanes 16-31 repeat that for rows
// 16-31; the selector picks the column. The 8-bit kinds (sixteen fields a
// row) put row r in lane r, chunks 0-7 in the first column and 8-15 in the
// second.
struct MetaCell { int lane, col, bit; };
MetaCell meta_cell(bool eight_bit, int row, int chunk, int sel) {
  if (eight_bit) return {row, chunk / 8, 4 * (chunk % 8)};
  const int base = row >= 16 ? 16 : 0, r = row - base;
  const bool upper_rows = r >= 8, upper_k = chunk >= 4;
  const int lane = base + r % 8 + (upper_k ? 8 : 0);
  const int slot = (upper_rows ? 4 : 0) + chunk % 4;   // the figure's cell, left to right
  return {lane, sel, 4 * slot};
}

// A structured-sparse row: chunk c of width w keeps the positions `keep`
// returns, and the field that says so (2:4: two 2-bit positions, low first;
// 1:2: 0b0100 or 0b1110; 4:8 in pairs: two 2-bit pair indices).
struct Sparse {
  int w;   // chunk width: 4, 2 (tf32) or 8 (mxf4)
  std::vector<int> keep(int m, int c) const {
    const int h = (m * 5 + c * 3) % 6;
    static const int pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    if (w == 2) return {(m + c) % 2};
    if (w == 4) return {pairs[h][0], pairs[h][1]};
    return {2 * pairs[h][0], 2 * pairs[h][0] + 1, 2 * pairs[h][1], 2 * pairs[h][1] + 1};
  }
  uint32_t field(int m, int c) const {
    const std::vector<int> k = keep(m, c);
    if (w == 2) return k[0] ? 0b1110 : 0b0100;
    if (w == 4) return uint32_t(k[0]) | uint32_t(k[1]) << 2;
    return uint32_t(k[0] / 2) | uint32_t(k[2] / 2) << 2;
  }
  // Dense A(m, k) from the stored values: zero where the chunk keeps nothing.
  float dense(int m, int k, const std::function<float(int, int)>& stored) const {
    const int c = k / w, per = w / 2;
    const std::vector<int> kp = keep(m, c);
    for (int s = 0; s < per; ++s)
      if (c * w + kp[s] == k) return stored(m, c * per + s);
    return 0.0f;
  }
};

// Writes the metadata of rows 0..rows-1 into the Mma kernel's auxiliary
// cells at column `col`, each partition p holding rows 32p.. (16p.. for M =
// 64, from lane `lane0`).
void put_meta(std::vector<uint32_t>& cells, int a_cols, int col, int rows, int per_part, int lane0,
              bool eight_bit, int sel, int K, const Sparse& sp) {
  for (int m = 0; m < rows; ++m)
    for (int c = 0; c < K / sp.w; ++c) {
      const MetaCell mc = meta_cell(eight_bit, m % per_part, c, sel);
      const int lane = 32 * (m / per_part) + lane0 + mc.lane;
      cells[size_t(lane) * a_cols + col + mc.col] |= sp.field(m, c) << mc.bit;
    }
}

}  // namespace

// ---- parsing -------------------------------------------------------------------

VTEST(tcgen05_parses_the_isa_examples) {
  const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k()
{
    .reg .pred p;
    .reg .b32 r0, r1, r2, r3, r4, r5, r6, r7, taddr0, taddr1, taddr2, taddr3, idesc, sMemAddr1, m0, m1, m2, m3;
    .reg .b64 adesc, bdesc, mbarObj0;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [sMemAddr1], 32;
    tcgen05.dealloc.cta_group::1.sync.aligned.b32  taddr0, 32;
    tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
    tcgen05.ld.sync.aligned.32x32b.x2.b32     {r0, r1}, [taddr1];
    tcgen05.ld.sync.aligned.16x128b.x4.b32    {r0, r1, r2, r3, r4, r5, r6, r7}, [taddr2];
    tcgen05.st.sync.aligned.16x64b.x4.b32               [taddr0], {r0,  r1,  r2,  r3};
    tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32  [taddr1], {r0,  r1};
    tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 {r0, r1}, [taddr3], 16;
    tcgen05.wait::ld.sync.aligned;
    tcgen05.wait::st.sync.aligned;
    tcgen05.fence::before_thread_sync;
    tcgen05.fence::after_thread_sync;
    tcgen05.mma.cta_group::1.kind::tf32      [taddr0],  adesc,  bdesc, idesc, {m0, m1, m2, m3}, p;
    tcgen05.mma.cta_group::1.kind::f16       [taddr0],  [taddr1],  bdesc, idesc, p, 3;
    tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbarObj0];
    ret;
}
)";
  auto m = ptx::parse(ptx);
  VCHECK_EQ(m.entries.size(), size_t{1});
}

VTEST(tcgen05_needs_a_blackwell_arch_specific_target) {
  const std::string body = R"(
.visible .entry k()
{
    tcgen05.fence::before_thread_sync;
    ret;
}
)";
  for (const char* t : {".target sm_90a", ".target sm_100", ".target sm_120a"}) {
    auto err = VCAPTURE(Error, ptx::parse(std::string(".version 8.7\n") + t + "\n.address_size 64\n" + body));
    VCHECK_CONTAINS(err.message(), "tcgen05 requires");
  }
  ptx::parse(std::string(".version 8.8\n.target sm_100f\n.address_size 64\n") + body);
  ptx::parse(std::string(".version 8.8\n.target sm_103a\n.address_size 64\n") + body);
}

VTEST(tcgen05_refuses_what_is_not_implemented_by_name) {
  auto parse = [](const std::string& ins) {
    return VCAPTURE(Error, ptx::parse(std::string(kHeader100a) +
                                      ".visible .entry k()\n{\n .reg .b32 a, b;\n .reg .b64 d;\n .reg .pred p;\n " +
                                      ins + "\n ret;\n}\n"));
  };
  VCHECK_CONTAINS(parse("tcgen05.shift.cta_group::1.down [a];").message(), "tcgen05.shift");
  VCHECK_CONTAINS(parse("tcgen05.cp.cta_group::1.128x128b.b8x16.b6x16_p32 [a], d;").message(), "decompression");
  VCHECK_CONTAINS(parse("tcgen05.cp.cta_group::1.32x128b [a], d;").message(), ".warpx4");
  VCHECK_CONTAINS(parse("tcgen05.mma.ws.cta_group::1.kind::f16 [a], d, d, a, p;").message(), "weight-stationary");
  VCHECK_CONTAINS(parse("tcgen05.mma.cta_group::1.kind::f16.block_scale [a], d, d, a, [b], [b], p;").message(),
                  ".block_scale");
  VCHECK_CONTAINS(parse("tcgen05.ld.red.sync.aligned.32x32b.x2.max.f32 {a, b}, a, [a];").message(), "tcgen05.ld.red");
}

// ---- tcgen05.ld / tcgen05.st ------------------------------------------------------

// Each shape's register placement, from figures 186-190 as tables: for lane
// row r (0-15, or 0-31) and 32-bit column c of one .x1 repeat, the thread and
// register there.
struct Cell { int t, r; };
Cell figure_cell(const std::string& shape, int row, int col) {
  if (shape == "32x32b") return {row, col};                              // figure 186
  if (shape == "16x64b") {                                               // figure 187
    // row 0: T0 T2 | row 1: T4 T6 ... row 8: T1 T3 | row 9: T5 T7 ...; r = 64-bit chunk
    return {4 * (row % 8) + row / 8 + 2 * (col % 2), col / 2};
  }
  if (shape == "16x128b")                                                // figure 188
    // rows 0-7: T(4r)..T(4r+3) : r0; rows 8-15: the same threads : r1
    return {4 * (row % 8) + col % 4, row / 8 + 2 * (col / 4)};
  if (shape == "16x256b")                                                // figure 189
    // rows 0-7: T4r:r0 T4r:r1 T(4r+1):r0 ...; rows 8-15: r2/r3
    return {4 * (row % 8) + (col % 8) / 2, (col % 2) + 2 * (row / 8) + 4 * (col / 8)};
  // 16x32bx2 (figure 190): T0-15 at the base address, T16-31 at the split.
  return {row + (col >= 64 ? 16 : 0), col % 64};
}

// Stores with one shape and reads back with .32x32b, which maps thread t to
// lane t exactly; every cell must hold the value the figure puts there.
VTEST(tcgen05_st_places_registers_as_the_figures_show) {
  struct Shape { std::string name; int lanes, cols, regs; };
  // .x2 of each: columns covered, registers per thread.
  const std::vector<Shape> shapes = {{"32x32b", 32, 2, 2}, {"16x64b", 16, 4, 2},
                                     {"16x128b", 16, 8, 4}, {"16x256b", 16, 16, 8},
                                     {"16x32bx2", 16, 66, 2}};
  for (const Shape& s : shapes) {
    std::string regs;
    for (int i = 0; i < s.regs; ++i) regs += (i ? ", " : "") + std::string("%v") + std::to_string(i);
    std::string sets;
    for (int i = 0; i < s.regs; ++i)   // value = thread * 256 + register + 1
      sets += "    mad.lo.u32 %v" + std::to_string(i) + ", %r1, 256, " + std::to_string(i + 1) + ";\n";
    std::string reads;
    for (int c = 0; c < 128; c += 32) {
      std::string rr;
      for (int i = 0; i < 32; ++i) rr += (i ? ", " : "") + std::string("%w") + std::to_string(i);
      reads += "    add.u32 %r5, %r4, " + std::to_string(c) + ";\n";
      reads += "    tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + rr + "}, [%r5];\n    tcgen05.wait::ld.sync.aligned;\n";
      for (int i = 0; i < 32; ++i)
        reads += "    st.global.u32 [%rd2+" + std::to_string(4 * (c + i)) + "], %w" + std::to_string(i) + ";\n";
    }
    const std::string split = s.name == "16x32bx2" ? ", 64" : "";
    const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<10>;
    .reg .b32 %v<10>;
    .reg .b32 %w<32>;
    .reg .b64 %rd<4>;
    .shared .align 4 .b32 slot;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %tid.x;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [slot], 128;
    ld.shared.u32 %r4, [slot];
)" + sets + "    tcgen05.st.sync.aligned." + s.name + ".x2.b32 [%r4]" + split + ", {" + regs + R"(};
    tcgen05.wait::st.sync.aligned;
    mul.wide.u32 %rd3, %r1, 512;
    add.u64 %rd2, %rd1, %rd3;
)" + reads + R"(
    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r4, 128;
    ret;
}
)";
    MemoryManager mem{1 << 20};
    const uint64_t out = mem.alloc(32 * 128 * 4);
    std::vector<uint32_t> zero(32 * 128, 0);
    mem.write(out, zero.data(), zero.size() * 4);
    LaunchConfig cfg;
    cfg.block = {32, 1, 1};
    exec::launch(ptx::parse(ptx).entries[0], cfg, {arg_u64(out)}, mem, load_gpu("nvidia/b200"));
    std::vector<uint32_t> got(32 * 128);
    mem.read(out, got.data(), got.size() * 4);
    for (int lane = 0; lane < 32; ++lane)
      for (int col = 0; col < 128; ++col) {
        uint32_t want = 0;
        const bool covered = lane < s.lanes &&
                             (s.name == "16x32bx2" ? (col < 2 || (col >= 64 && col < 66)) : col < s.cols);
        if (covered) {
          // One .x1 repeat of figure_cell covers s.cols/2 columns; .x2 repeats it
          // to the right, register numbers following on.
          const int per = s.name == "16x32bx2" ? 1 : s.cols / 2;
          const int rep = s.name == "16x32bx2" ? col % 64 : col / per;
          const Cell c = figure_cell(s.name, lane, s.name == "16x32bx2" ? (col >= 64 ? 64 : 0) : col % per);
          const int regs_per_x1 = s.regs / 2;
          want = uint32_t(c.t) * 256 + uint32_t(c.r + rep * regs_per_x1) + 1;
        }
        if (got[size_t(lane) * 128 + col] != want)
          VCHECK_EQ(s.name + " lane " + std::to_string(lane) + " col " + std::to_string(col) + ": " +
                        std::to_string(got[size_t(lane) * 128 + col]),
                    s.name + " lane " + std::to_string(lane) + " col " + std::to_string(col) + ": " +
                        std::to_string(want));
      }
  }
}

// .unpack::16b splits each register over two columns, low half first, and
// .pack::16b puts them back together (figure 196).
VTEST(tcgen05_pack_and_unpack_16b) {
  const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<4>;
    .shared .align 4 .b32 slot;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %tid.x;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [slot], 32;
    ld.shared.u32 %r4, [slot];
    mad.lo.u32 %r5, %r1, 0x10001, 0x00020001;   // (2t+2) << 16 | (t+1), roughly
    add.u32 %r6, %r5, 0x01000100;
    tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32 [%r4], {%r5, %r6};
    tcgen05.wait::st.sync.aligned;
    tcgen05.ld.sync.aligned.32x32b.x4.b32 {%r7, %r8, %r9, %r10}, [%r4];
    tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 {%r11, %r12}, [%r4];
    tcgen05.wait::ld.sync.aligned;
    mul.wide.u32 %rd3, %r1, 32;
    add.u64 %rd2, %rd1, %rd3;
    st.global.v4.u32 [%rd2], {%r7, %r8, %r9, %r10};
    st.global.v2.u32 [%rd2+16], {%r11, %r12};
    st.global.v2.u32 [%rd2+24], {%r5, %r6};
    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r4, 32;
    ret;
}
)";
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(32 * 32);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(ptx::parse(ptx).entries[0], cfg, {arg_u64(out)}, mem, load_gpu("nvidia/b200"));
  std::vector<uint32_t> got(32 * 8);
  mem.read(out, got.data(), got.size() * 4);
  for (int t = 0; t < 32; ++t) {
    const uint32_t* g = &got[size_t(t) * 8];
    const uint32_t r5 = g[6], r6 = g[7];
    VCHECK_EQ(g[0], r5 & 0xFFFF);
    VCHECK_EQ(g[1], r5 >> 16);
    VCHECK_EQ(g[2], r6 & 0xFFFF);
    VCHECK_EQ(g[3], r6 >> 16);
    VCHECK_EQ(g[4], r5);
    VCHECK_EQ(g[5], r6);
  }
}

// A warp reaches only its quarter of the lanes (9.7.18.8.1), and only
// columns that are allocated.
VTEST(tcgen05_ld_st_stay_in_the_warps_lanes_and_allocated_columns) {
  auto run = [](const std::string& addr_setup) {
    const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k()
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .shared .align 4 .b32 slot;
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;
    setp.eq.u32 %p1, %r2, 0;
    @%p1 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [slot], 32;
    bar.sync 0;
    ld.shared.u32 %r4, [slot];
)" + addr_setup + R"(
    tcgen05.st.sync.aligned.32x32b.x1.b32 [%r4], {%r1};
    tcgen05.wait::st.sync.aligned;
    bar.sync 0;
    @%p1 tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r4, 32;
    ret;
}
)";
    MemoryManager mem{1 << 20};
    LaunchConfig cfg;
    cfg.block = {64, 1, 1};
    exec::launch(ptx::parse(ptx).entries[0], cfg, {}, mem, load_gpu("nvidia/b200"));
  };
  // Each warp in its own quarter: fine.
  run("    shl.b32 %r5, %r2, 21;\n    add.u32 %r4, %r4, %r5;\n    and.b32 %r4, %r4, 0xFFFFFFFF;\n");
  // Warp 1 at lane 0: outside its lanes 32-63.
  auto err = VCAPTURE(Error, run("    mov.u32 %r4, %r4;\n"));
  VCHECK_CONTAINS(err.message(), "may only reach lanes 32-63");
  // Column 40 of a 32-column allocation.
  err = VCAPTURE(Error, run("    shl.b32 %r5, %r2, 21;\n    add.u32 %r4, %r4, %r5;\n    add.u32 %r4, %r4, 40;\n"));
  VCHECK_CONTAINS(err.message(), "column 40");
}

VTEST(tcgen05_memory_must_be_deallocated_before_exit) {
  const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k()
{
    .shared .align 4 .b32 slot;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [slot], 64;
    ret;
}
)";
  MemoryManager mem{1 << 20};
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  auto err = VCAPTURE(Error, exec::launch(ptx::parse(ptx).entries[0], cfg, {}, mem, load_gpu("nvidia/b200")));
  VCHECK_CONTAINS(err.message(), "64 Tensor Memory columns still allocated");
}

// Allocations are power-of-two runs; two live ones do not overlap, and a
// freed run is handed out again.
VTEST(tcgen05_alloc_hands_out_disjoint_runs) {
  const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<2>;
    .shared .align 4 .b32 s0;
    .shared .align 4 .b32 s1;
    .shared .align 4 .b32 s2;
    .shared .align 4 .b32 s3;
    ld.param.u64 %rd1, [out];
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [s0], 32;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [s1], 128;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [s2], 64;
    ld.shared.u32 %r1, [s0];
    ld.shared.u32 %r2, [s1];
    ld.shared.u32 %r3, [s2];
    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r1, 32;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [s3], 32;
    ld.shared.u32 %r4, [s3];
    st.global.v4.u32 [%rd1], {%r1, %r2, %r3, %r4};
    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r2, 128;
    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r3, 64;
    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r4, 32;
    ret;
}
)";
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(16);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(ptx::parse(ptx).entries[0], cfg, {arg_u64(out)}, mem, load_gpu("nvidia/b200"));
  uint32_t got[4];
  mem.read(out, got, 16);
  auto overlap = [](uint32_t a, uint32_t na, uint32_t b, uint32_t nb) { return a < b + nb && b < a + na; };
  VCHECK(!overlap(got[0], 32, got[1], 128));
  VCHECK(!overlap(got[0], 32, got[2], 64));
  VCHECK(!overlap(got[1], 128, got[2], 64));
  VCHECK_EQ(got[3], got[0]);
  for (uint32_t g : got) VCHECK_EQ(g >> 16, 0u);   // lane 0
  VCHECK_EQ(got[1] % 128, 0u);
}

// ---- tcgen05.mma, one CTA ---------------------------------------------------------

// f16 x f16 -> f32, M = 128, N = 64, both K-major without swizzling: the
// canonical layout ((8,m),(T,2k)):((1T,SBO),(1,LBO)) with 128-byte core
// matrices, SBO between row groups and LBO between the two K halves.
VTEST(tcgen05_mma_f16_m128_k_major) {
  const int M = 128, N = 64, K = 16;
  auto A = [](int m, int k) { return val(m, k, 1); };
  auto B = [](int k, int n) { return val(k, n, 5); };
  const Operands o = images(1, M, N, K, 2, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int k) { return f16_bits(A(m, k)); },
                            [&](int n, int k) { return f16_bits(B(k, n)); });
  Mma x;
  x.id = idesc(1, 0, 0, M, N);
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_a = o.a;
  x.smem_b = o.b;
  check_d_f32(x.run(), 1, M, N, K, A, B);
}

// bf16, both MN-major with the 128-byte swizzle, M = 64 (layout F) with D at
// lane 16: rows go to lanes 16-31 of each warp's quarter.
VTEST(tcgen05_mma_bf16_m64_mn_major_128b_swizzle_layout_f) {
  const int M = 64, N = 64, K = 16;
  auto A = [](int m, int k) { return val(m, k, 2); };
  auto B = [](int k, int n) { return val(k, n, 3); };
  // MN-major 128B: T = 8 elements, 64 MN per swizzled row; LBO steps 64 MN,
  // SBO steps 8 K rows (1024 bytes).
  const Operands o = images(1, M, N, K, 2, Major::MN, Major::MN, 128, 128, 8192, 1024, 8192, 1024,
                            [&](int m, int k) { return bf16_bits(A(m, k)); },
                            [&](int n, int k) { return bf16_bits(B(k, n)); });
  Mma x;
  x.id = idesc(1, 1, 1, M, N, false, false, true, true);
  x.desc_a = desc(0, 8192, 1024, 128);
  x.desc_b = desc(0, 8192, 1024, 128);
  x.smem_a = o.a;
  x.smem_b = o.b;
  x.d_lane = 16;
  check_d_f32(x.run(), 1, M, N, K, A, B, nullptr, 16);
}

// K-major with the 128-byte swizzle, and the descriptors advanced 32 bytes
// along K inside the swizzle atom -- how CUTLASS walks a 64-wide K tile in
// four MMAs. The swizzle is a function of the address, so the second K block
// must come out of the same image.
VTEST(tcgen05_mma_128b_swizzle_k_block_inside_the_atom) {
  // M = 64: 64 rows of 128 swizzled bytes fill the 8 KiB operand buffer.
  const int M = 64, N = 32, K = 16, KT = 64;
  auto A = [](int m, int k) { return val(m, k, 4); };
  auto B = [](int k, int n) { return val(k, n, 6); };
  const Operands o = images(1, M, N, KT, 2, Major::K, Major::K, 128, 128, 16, 1024, 16, 1024,
                            [&](int m, int k) { return f16_bits(A(m, k)); },
                            [&](int n, int k) { return f16_bits(B(k, n)); });
  Mma x;
  x.id = idesc(1, 0, 0, M, N);
  x.desc_a = desc(32, 16, 1024, 128);   // K block 1: elements 16-31
  x.desc_b = desc(32, 16, 1024, 128);
  x.smem_a = o.a;
  x.smem_b = o.b;
  x.cols = 32;
  check_d_f32(x.run(), 1, M, N, K, [&](int m, int k) { return A(m, k + 16); },
              [&](int k, int n) { return B(k + 16, n); }, nullptr, 0, 32);
}

// tf32, both MN-major in the 128-byte swizzle with 32-byte atoms (descriptor
// mode 1), the only swizzle a 32-bit transpose may use (Table 65).
VTEST(tcgen05_mma_tf32_mn_major_128b_swizzle_32b_atoms) {
  const int M = 64, N = 32, K = 8;
  auto A = [](int m, int k) { return val(m, k, 3); };
  auto B = [](int k, int n) { return val(k, n, 4); };
  // T = 4 tf32 per 16 bytes, 32 MN per 128-byte row; SBO steps 8 K rows.
  const Operands o = images(1, M, N, K, 4, Major::MN, Major::MN, 128, 128, 1024, 1024, 1024, 1024,
                            [&](int m, int k) { return f32_bits(A(m, k)); },
                            [&](int n, int k) { return f32_bits(B(k, n)); }, 32);
  Mma x;
  x.kind = "tf32";
  x.id = idesc(1, 2, 2, M, N, false, false, true, true);
  x.desc_a = desc(0, 1024, 1024, 128, 32);
  x.desc_b = desc(0, 1024, 1024, 128, 32);
  x.smem_a = o.a;
  x.smem_b = o.b;
  x.cols = 32;
  check_d_f32(x.run(), 1, M, N, K, A, B, nullptr, 0, 32);
}

// enable-input-d accumulates into what is there, scale-input-d scales it by
// 2^-s first, and the negate bits flip A and B.
VTEST(tcgen05_mma_accumulate_scale_and_negate) {
  const int M = 128, N = 32, K = 8;
  auto A = [](int m, int k) { return val(m, k, 7); };
  auto B = [](int k, int n) { return val(k, n, 8); };
  const Operands o = images(1, M, N, K, 4, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int k) { return f32_bits(A(m, k)); },
                            [&](int n, int k) { return f32_bits(B(k, n)); });
  Mma x;
  x.kind = "tf32";
  x.id = idesc(1, 2, 2, M, N, true, false);
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_a = o.a;
  x.smem_b = o.b;
  x.cols = 32;
  x.enable_d = "1";
  x.scale_tail = ", 2";
  auto D0 = [](int m, int n) { return static_cast<float>((m + 3 * n) % 11) - 5.0f; };
  x.d_in.assign(1, std::vector<uint32_t>(128 * 32));
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) x.d_in[0][size_t(m) * 32 + n] = f32_bits(D0(m, n));
  check_d_f32(x.run(), 1, M, N, K, [&](int m, int k) { return -A(m, k); }, B,
              [&](int m, int n) { return D0(m, n) / 4.0f; }, 0, 32);
}

// disable-output-lane leaves the lanes it names as they were.
VTEST(tcgen05_mma_disable_output_lane) {
  const int M = 128, N = 32, K = 16;
  auto A = [](int m, int k) { return val(m, k, 1); };
  auto B = [](int k, int n) { return val(k, n, 2); };
  const Operands o = images(1, M, N, K, 2, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int k) { return f16_bits(A(m, k)); },
                            [&](int n, int k) { return f16_bits(B(k, n)); });
  Mma x;
  x.id = idesc(1, 0, 0, M, N);
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_a = o.a;
  x.smem_b = o.b;
  x.cols = 32;
  x.mask = "{0x0000000F, 0, 0x80000000, 0}";   // lanes 0-3 and 95
  x.d_in.assign(1, std::vector<uint32_t>(128 * 32, f32_bits(99.0f)));
  const auto out = x.run();
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float want = 0;
      for (int k = 0; k < K; ++k) want += A(m, k) * B(k, n);
      if (m < 4 || m == 95) want = 99.0f;
      VCHECK_EQ(bits_f32(out[0][size_t(m) * 32 + n]), want);
    }
}

// An f16 D is one half in the low 16 bits of its cell.
VTEST(tcgen05_mma_f16_accumulator_in_the_low_half) {
  const int M = 128, N = 16, K = 16;
  auto A = [](int m, int k) { return val(m, k, 3); };
  auto B = [](int k, int n) { return val(k, n, 1); };
  const Operands o = images(1, M, N, K, 2, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int k) { return f16_bits(A(m, k)); },
                            [&](int n, int k) { return f16_bits(B(k, n)); });
  Mma x;
  x.id = idesc(0, 0, 0, M, N);
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_a = o.a;
  x.smem_b = o.b;
  x.cols = 32;
  const auto out = x.run();
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float want = 0;
      for (int k = 0; k < K; ++k) want += A(m, k) * B(k, n);
      const uint32_t cell = out[0][size_t(m) * 32 + n];
      VCHECK_EQ(cell >> 16, 0u);
      VCHECK_EQ(f16_value(static_cast<uint16_t>(cell)), want);
    }
}

// e4m3 x e5m2 -> f32 (.kind::f8f6f4), K = 32.
VTEST(tcgen05_mma_fp8) {
  const int M = 128, N = 32, K = 32;
  auto A = [](int m, int k) { return val(m, k, 5); };
  auto B = [](int k, int n) { return val(k, n, 2); };
  // e5m2 is e4m3's neighbour with bias 15 and two mantissa bits.
  auto e5m2 = [](float f) -> uint8_t {
    if (f == 0.0f) return 0;
    int e = 0;
    const float m = std::frexp(std::fabs(f), &e);
    const uint8_t mant = static_cast<uint8_t>((m * 2.0f - 1.0f) * 4.0f);
    return static_cast<uint8_t>((f < 0 ? 0x80 : 0) | ((e - 1 + 15) << 2) | mant);
  };
  const Operands o = images(1, M, N, K, 1, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int k) { return e4m3_bits(A(m, k)); },
                            [&](int n, int k) { return e5m2(B(k, n)); });
  Mma x;
  x.kind = "f8f6f4";
  x.id = idesc(1, 0, 1, M, N);
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_a = o.a;
  x.smem_b = o.b;
  x.cols = 32;
  check_d_f32(x.run(), 1, M, N, K, A, B, nullptr, 0, 32);
}

// s8 x u8 -> s32 with .satfinite (instruction descriptor bit 3) clamping.
VTEST(tcgen05_mma_i8_saturates) {
  const int M = 128, N = 32, K = 32;
  auto A = [](int m, int k) { return m == 5 ? 127 : ((m * 3 + k * 5) % 21) - 10; };
  auto B = [](int k, int n) { return n == 7 ? 255 : (k * 7 + n) % 13; };
  const Operands o = images(1, M, N, K, 1, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int k) { return uint64_t(uint8_t(int8_t(A(m, k)))); },
                            [&](int n, int k) { return uint64_t(uint8_t(B(k, n))); });
  Mma x;
  x.kind = "i8";
  x.id = idesc(2, 1, 0, M, N, false, false, false, false, true);
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_a = o.a;
  x.smem_b = o.b;
  x.cols = 32;
  x.enable_d = "1";
  x.d_in.assign(1, std::vector<uint32_t>(128 * 32, 0x7FFFFF00u));
  const auto out = x.run();
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int64_t want = 0x7FFFFF00;
      for (int k = 0; k < K; ++k) want += int64_t(A(m, k)) * B(k, n);
      want = std::min<int64_t>(want, 0x7FFFFFFF);
      want = std::max<int64_t>(want, INT32_MIN);
      VCHECK_EQ(int32_t(out[0][size_t(m) * 32 + n]), int32_t(want));
    }
}

// A from Tensor Memory: row m in lane m, K packed 32 bits to a column
// (two f16s, low half first).
VTEST(tcgen05_mma_a_from_tensor_memory) {
  const int M = 128, N = 32, K = 16;
  auto A = [](int m, int k) { return val(m, k, 6); };
  auto B = [](int k, int n) { return val(k, n, 7); };
  const Operands o = images(1, M, N, K, 2, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int k) { return f16_bits(A(m, k)); },
                            [&](int n, int k) { return f16_bits(B(k, n)); });
  Mma x;
  x.id = idesc(1, 0, 0, M, N);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_b = o.b;
  x.cols = 64;
  x.a_cols = 8;
  x.a_col = 32;
  x.a_tmem.assign(1, std::vector<uint32_t>(128 * 8));
  for (int m = 0; m < M; ++m)
    for (int c = 0; c < 8; ++c)
      x.a_tmem[0][size_t(m) * 8 + c] = f16_bits(A(m, 2 * c)) | uint32_t(f16_bits(A(m, 2 * c + 1))) << 16;
  const auto out = x.run();
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float want = 0;
      for (int k = 0; k < K; ++k) want += A(m, k) * B(k, n);
      VCHECK_EQ(bits_f32(out[0][size_t(m) * 64 + n]), want);
    }
}

// e2m1 x e3m2 -> f32 under .kind::f8f6f4: 16 elements to a 16-byte group of
// shared memory, the rest of the group padding.
VTEST(tcgen05_mma_fp4_times_fp6_from_padded_shared_memory) {
  const int M = 128, N = 32, K = 32;
  auto A = [](int m, int k) { return val(m, k, 3); };
  auto B = [](int k, int n) { return val(k, n, 4); };
  Mma x;
  x.kind = "f8f6f4";
  x.id = idesc(1, 5, 4, M, N);
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_a = {packed_k_major(M, K, 4, 16, 128, 256, [&](int m, int k) { return e2m1_bits(A(m, k)); })};
  x.smem_b = {packed_k_major(N, K, 6, 16, 128, 256, [&](int n, int k) { return e3m2_bits(B(k, n)); })};
  x.cols = 32;
  check_d_f32(x.run(), 1, M, N, K, A, B, nullptr, 0, 32);
}

// A in Tensor Memory in 8-bit containers: e2m1 in bits 2-5 of each byte
// (figure 202), four to a column; B e2m3.
VTEST(tcgen05_mma_fp4_a_from_tensor_memory_containers) {
  const int M = 128, N = 32, K = 32;
  auto A = [](int m, int k) { return val(m, k, 8); };
  auto B = [](int k, int n) { return val(k, n, 1); };
  Mma x;
  x.kind = "f8f6f4";
  x.id = idesc(1, 5, 3, M, N);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_b = {packed_k_major(N, K, 6, 16, 128, 256, [&](int n, int k) { return e2m3_bits(B(k, n)); })};
  x.cols = 64;
  x.a_cols = 8;
  x.a_col = 32;
  x.a_tmem.assign(1, std::vector<uint32_t>(128 * 8));
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      x.a_tmem[0][size_t(m) * 8 + k / 4] |= uint32_t(e2m1_bits(A(m, k))) << (2 + 8 * (k % 4));
  check_d_f32(x.run(), 1, M, N, K, A, B, nullptr, 0, 64);
}

// .kind::mxf8f6f4.block_scale: e4m3 x e2m3 with one UE8M0 factor per row of A
// and column of B (block 32), from bytes 1 and 2 of their cells.
VTEST(tcgen05_mma_mxf8f6f4_ue8m0_scales) {
  const int M = 128, N = 32, K = 32;
  auto A = [](int m, int k) { return val(m, k, 2); };
  auto B = [](int k, int n) { return val(k, n, 6); };
  auto sa = [](int m, int) { return uint8_t(124 + m % 7); };
  auto sb = [](int n, int) { return uint8_t(125 + n % 5); };
  Mma x;
  x.kind = "mxf8f6f4";
  x.kind_mods = ".block_scale.scale_vec::1X";
  x.block_scale = true;
  x.id = idesc_mx(0, 3, M, N, 1, 2, 1);
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  const Operands o = images(1, M, N, K, 1, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int k) { return e4m3_bits(A(m, k)); }, [](int, int) { return 0; });
  x.smem_a = o.a;
  x.smem_b = {packed_k_major(N, K, 6, 16, 128, 256, [&](int n, int k) { return e2m3_bits(B(k, n)); })};
  x.cols = 64;
  x.a_is_tmem = false;
  x.a_cols = 8;
  x.a_col = 32;
  x.sfa_col = 32;
  x.sfb_col = 36;
  x.a_tmem.assign(1, std::vector<uint32_t>(128 * 8));
  put_scales(x.a_tmem[0], 8, 0, M, 1, 1, sa);
  put_scales(x.a_tmem[0], 8, 4, N, 1, 2, sb);
  auto ue8m0 = [](uint8_t v) { return std::ldexp(1.0f, int(v) - 127); };
  check_scaled(x.run(), M, N, K, 32, 64, A, B, [&](int m, int j) { return ue8m0(sa(m, j)); },
               [&](int n, int j) { return ue8m0(sb(n, j)); });
}

// .kind::mxf4.block_scale.scale_vec::2X: fp4 packed two to a byte, K = 64 in
// two blocks of 32, the factors from bytes 2-3 (A) and 0-1 (B).
VTEST(tcgen05_mma_mxf4_two_blocks) {
  const int M = 128, N = 64, K = 64;
  auto A = [](int m, int k) { return val(m, k, 4) * (k >= 32 ? 2.0f : 1.0f); };
  auto B = [](int k, int n) { return val(k, n, 9); };
  auto sa = [](int m, int j) { return uint8_t(126 + (m + 3 * j) % 4); };
  auto sb = [](int n, int j) { return uint8_t(127 - (n + j) % 3); };
  Mma x;
  x.kind = "mxf4";
  x.kind_mods = ".block_scale.scale_vec::2X";
  x.block_scale = true;
  x.id = idesc_mx(1, 1, M, N, 2, 0, 1);
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_a = {packed_k_major(M, K, 4, 32, 128, 256, [&](int m, int k) { return e2m1_bits(A(m, k)); })};
  x.smem_b = {packed_k_major(N, K, 4, 32, 128, 256, [&](int n, int k) { return e2m1_bits(B(k, n)); })};
  x.cols = 128;
  x.a_is_tmem = false;
  x.a_cols = 8;
  x.a_col = 64;
  x.sfa_col = 64;
  x.sfb_col = 68;
  x.a_tmem.assign(1, std::vector<uint32_t>(128 * 8));
  put_scales(x.a_tmem[0], 8, 0, M, 2, 2, sa);
  put_scales(x.a_tmem[0], 8, 4, N, 2, 0, sb);
  auto ue8m0 = [](uint8_t v) { return std::ldexp(1.0f, int(v) - 127); };
  check_scaled(x.run(), M, N, K, 32, 128, A, B, [&](int m, int j) { return ue8m0(sa(m, j)); },
               [&](int n, int j) { return ue8m0(sb(n, j)); });
}

// .kind::mxf4nvf4.block_scale.scale_vec::4X with UE4M3 factors (scale type
// 0): four blocks of 16.
VTEST(tcgen05_mma_mxf4nvf4_ue4m3_block16) {
  const int M = 128, N = 32, K = 64;
  auto A = [](int m, int k) { return val(m, k, 7); };
  auto B = [](int k, int n) { return val(k, n, 2); };
  const float steps[] = {0.5f, 1.0f, 1.5f, 2.0f, 0.75f};
  auto sa = [&](int m, int j) { return steps[(m + j) % 5]; };
  auto sb = [&](int n, int j) { return steps[(2 * n + 3 * j) % 5]; };
  Mma x;
  x.kind = "mxf4nvf4";
  x.kind_mods = ".block_scale.scale_vec::4X";
  x.block_scale = true;
  x.id = idesc_mx(1, 1, M, N, 0, 0, 0);
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_a = {packed_k_major(M, K, 4, 32, 128, 256, [&](int m, int k) { return e2m1_bits(A(m, k)); })};
  x.smem_b = {packed_k_major(N, K, 4, 32, 128, 256, [&](int n, int k) { return e2m1_bits(B(k, n)); })};
  x.cols = 64;
  x.a_is_tmem = false;
  x.a_cols = 8;
  x.a_col = 32;
  x.sfa_col = 32;
  x.sfb_col = 36;
  x.a_tmem.assign(1, std::vector<uint32_t>(128 * 8));
  put_scales(x.a_tmem[0], 8, 0, M, 4, 0, [&](int m, int j) { return e4m3_bits(sa(m, j)); });
  put_scales(x.a_tmem[0], 8, 4, N, 4, 0, [&](int n, int j) { return e4m3_bits(sb(n, j)); });
  check_scaled(x.run(), M, N, K, 16, 64, A, B, sa, sb);
}

VTEST(tcgen05_mma_block_scale_refuses_what_the_isa_rules_out) {
  Mma x;
  x.kind = "mxf4nvf4";
  x.kind_mods = ".block_scale.scale_vec::2X";
  x.block_scale = true;
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  x.a_is_tmem = false;
  x.a_cols = 8;
  x.a_col = 32;
  x.sfa_col = 32;
  x.sfb_col = 36;
  x.a_tmem.assign(1, std::vector<uint32_t>(128 * 8));
  x.id = idesc_mx(1, 1, 128, 32, 0, 0, 0);   // UE4M3 with 2X
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "sm_107f");
  x.kind_mods = ".block_scale.scale_vec::4X";
  x.id = idesc_mx(1, 1, 128, 32, 2, 0, 0);   // 4X needs SFA_ID 0
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "Tables 51-53");
  x.kind = "mxf8f6f4";
  x.kind_mods = ".block_scale";
  x.id = idesc_mx(0, 0, 128, 32, 0, 0, 0);   // mxf8f6f4 scales are UE8M0 only
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "Tables 51-53");
  x.id = idesc_mx(0, 0, 128, 32, 0, 0, 1) | (1u << 26);
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "sm_107f");
  x.id = idesc_mx(0, 0, 128, 32, 0, 0, 1);
  x.sfb_col = 100;   // past the 64 allocated columns
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "scale factor");
}

// f16 2:4, M = 128, K = 32 (A stored 128 x 16), metadata through selector 1.
VTEST(tcgen05_mma_sp_f16_selector_1) {
  const int M = 128, N = 32, K = 32;
  const Sparse sp{4};
  auto S = [](int m, int j) { return val(m, j, 3); };
  auto A = [&](int m, int k) { return sp.dense(m, k, S); };
  auto B = [](int k, int n) { return val(k, n, 8); };
  const Operands o = images(1, M, N, K / 2, 2, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int j) { return f16_bits(S(m, j)); }, [](int, int) { return 0; });
  Mma x;
  x.sparse = true;
  x.id = idesc(1, 0, 0, M, N) | 4 | 1;
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 512, 0);
  x.smem_a = o.a;
  x.smem_b = {std::vector<uint8_t>()};
  // B for K = 32: four core matrices along K (LBO 128), SBO 512.
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      put(x.smem_b[0], canonical(Major::K, 0, 2, 128, 512, n, k), f16_bits(B(k, n)), 2);
  x.cols = 64;
  x.a_is_tmem = false;
  x.a_cols = 4;
  x.a_col = 32;
  x.meta_col = 32;
  x.a_tmem.assign(1, std::vector<uint32_t>(128 * 4));
  put_meta(x.a_tmem[0], 4, 0, M, 32, 0, false, 1, K, sp);
  check_d_f32(x.run(), 1, M, N, K, A, B, nullptr, 0, 64);
}

// tf32 1:2, K = 16 (A stored 128 x 8).
VTEST(tcgen05_mma_sp_tf32) {
  const int M = 128, N = 16, K = 16;
  const Sparse sp{2};
  auto S = [](int m, int j) { return val(m, j, 6); };
  auto A = [&](int m, int k) { return sp.dense(m, k, S); };
  auto B = [](int k, int n) { return val(k, n, 2); };
  const Operands o = images(1, M, N, K / 2, 4, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int j) { return f32_bits(S(m, j)); }, [](int, int) { return 0; });
  Mma x;
  x.kind = "tf32";
  x.sparse = true;
  x.id = idesc(1, 2, 2, M, N) | 4;
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 512, 0);
  x.smem_a = o.a;
  x.smem_b = {std::vector<uint8_t>()};
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      put(x.smem_b[0], canonical(Major::K, 0, 4, 128, 512, n, k), f32_bits(B(k, n)), 4);
  x.cols = 32;
  x.a_is_tmem = false;
  x.a_cols = 2;
  x.a_col = 16;
  x.meta_col = 16;
  x.a_tmem.assign(1, std::vector<uint32_t>(128 * 2));
  put_meta(x.a_tmem[0], 2, 0, M, 32, 0, false, 0, K, sp);
  check_d_f32(x.run(), 1, M, N, K, A, B, nullptr, 0, 32);
}

// e4m3 x e4m3 2:4, K = 64: one row a lane, sixteen fields over two columns.
VTEST(tcgen05_mma_sp_fp8_row_per_lane) {
  const int M = 128, N = 16, K = 64;
  const Sparse sp{4};
  auto S = [](int m, int j) { return val(m, j, 1); };
  auto A = [&](int m, int k) { return sp.dense(m, k, S); };
  auto B = [](int k, int n) { return val(k, n, 5); };
  Mma x;
  x.kind = "f8f6f4";
  x.sparse = true;
  x.id = idesc(1, 0, 0, M, N) | 4;
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 512, 0);
  x.smem_a = {std::vector<uint8_t>()};
  x.smem_b = {std::vector<uint8_t>()};
  for (int m = 0; m < M; ++m)
    for (int j = 0; j < K / 2; ++j) put(x.smem_a[0], canonical(Major::K, 0, 1, 128, 256, m, j), e4m3_bits(S(m, j)), 1);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) put(x.smem_b[0], canonical(Major::K, 0, 1, 128, 512, n, k), e4m3_bits(B(k, n)), 1);
  x.cols = 32;
  x.a_is_tmem = false;
  x.a_cols = 2;
  x.a_col = 16;
  x.meta_col = 16;
  x.a_tmem.assign(1, std::vector<uint32_t>(128 * 2));
  put_meta(x.a_tmem[0], 2, 0, M, 32, 0, true, 0, K, sp);
  check_d_f32(x.run(), 1, M, N, K, A, B, nullptr, 0, 32);
}

// M = 64 (layout F) at lane 16: D rows, and the metadata, in lanes 16-31 of
// each quarter.
VTEST(tcgen05_mma_sp_f16_m64_lane16) {
  const int M = 64, N = 16, K = 32;
  const Sparse sp{4};
  auto S = [](int m, int j) { return val(m, j, 7); };
  auto A = [&](int m, int k) { return sp.dense(m, k, S); };
  auto B = [](int k, int n) { return val(k, n, 4); };
  const Operands o = images(1, M, N, K / 2, 2, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int j) { return f16_bits(S(m, j)); }, [](int, int) { return 0; });
  Mma x;
  x.sparse = true;
  x.id = idesc(1, 0, 0, M, N) | 4;
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 512, 0);
  x.smem_a = o.a;
  x.smem_b = {std::vector<uint8_t>()};
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) put(x.smem_b[0], canonical(Major::K, 0, 2, 128, 512, n, k), f16_bits(B(k, n)), 2);
  x.cols = 32;
  x.d_lane = 16;
  x.a_is_tmem = false;
  x.a_cols = 2;
  x.a_col = 16;
  x.meta_col = 16;
  x.meta_lane = 16;
  x.a_tmem.assign(1, std::vector<uint32_t>(128 * 2));
  put_meta(x.a_tmem[0], 2, 0, M, 16, 16, false, 0, K, sp);
  check_d_f32(x.run(), 1, M, N, K, A, B, nullptr, 16, 32);
}

// .kind::mxf4 4:8 in pairs, K = 128: its default .block32 gives four
// factors a row.
VTEST(tcgen05_mma_sp_mxf4_pairs_four_blocks) {
  const int M = 128, N = 16, K = 128;
  const Sparse sp{8};
  auto S = [](int m, int j) { return val(m, j, 2); };
  auto A = [&](int m, int k) { return sp.dense(m, k, S); };
  auto B = [](int k, int n) { return val(k, n, 3); };
  auto sa = [](int m, int j) { return uint8_t(125 + (m + j) % 5); };
  auto sb = [](int n, int j) { return uint8_t(126 + (2 * n + j) % 3); };
  Mma x;
  x.kind = "mxf4";
  x.kind_mods = ".block_scale";
  x.block_scale = true;
  x.sparse = true;
  x.id = idesc_mx(1, 1, M, N, 0, 0, 1) | 4;
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 512, 0);
  x.smem_a = {packed_k_major(M, K / 2, 4, 32, 128, 256, [&](int m, int j) { return e2m1_bits(S(m, j)); })};
  x.smem_b = {packed_k_major(N, K, 4, 32, 128, 512, [&](int n, int k) { return e2m1_bits(B(k, n)); })};
  x.cols = 32;
  x.a_is_tmem = false;
  x.a_cols = 8;
  x.a_col = 16;
  x.sfa_col = 16;
  x.sfb_col = 20;
  x.meta_col = 22;
  x.a_tmem.assign(1, std::vector<uint32_t>(128 * 8));
  put_scales(x.a_tmem[0], 8, 0, M, 4, 0, sa);
  put_scales(x.a_tmem[0], 8, 4, N, 4, 0, sb);
  put_meta(x.a_tmem[0], 8, 6, M, 32, 0, true, 0, K, sp);
  auto ue8m0 = [](uint8_t v) { return std::ldexp(1.0f, int(v) - 127); };
  check_scaled(x.run(), M, N, K, 32, 32, A, B, [&](int m, int j) { return ue8m0(sa(m, j)); },
               [&](int n, int j) { return ue8m0(sb(n, j)); });
}

VTEST(tcgen05_mma_sp_refuses_what_the_isa_rules_out) {
  Mma x;
  x.sparse = true;
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 512, 0);
  x.a_is_tmem = false;
  x.a_cols = 2;
  x.a_col = 32;
  x.meta_col = 32;
  x.a_tmem.assign(1, std::vector<uint32_t>(128 * 2));
  x.id = idesc(1, 0, 0, 128, 16);   // no sparsity bit
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "sparsity bit (2)");
  x.id = idesc(1, 0, 0, 128, 16) | 4 | 2;   // selector 2
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "selectors 0 and 1");
  x.kind = "f8f6f4";
  x.id = idesc(1, 0, 0, 128, 16) | 4 | 1;
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "must be 0");
  x.id = idesc(1, 0, 0, 128, 16) | 4;
  x.meta_lane = 16;   // M = 128 starts at lane 0
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "9.7.18.10.9.5");
  x.meta_lane = 0;
  x.meta_col = 63;    // the second column is past the allocation
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "metadata");
}

VTEST(tcgen05_mma_refuses_descriptors_and_shapes_the_isa_rules_out) {
  Mma x;
  x.id = idesc(1, 0, 0, 192, 16);   // M = 192
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "Table 48");
  x.id = idesc(1, 0, 0, 128, 16) | 4;   // sparse
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "sparse A");
  x.id = idesc(1, 0, 0, 128, 16);
  x.desc_b = desc(0, 128, 256, 0) & ~(uint64_t{1} << 46);
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "0b001 in bits 46-48");
  x.desc_b = desc(0, 128, 256, 0) | (uint64_t{2} << 49);
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "nonzero base offset");
  x.desc_b = desc(0, 128, 256, 0) | (uint64_t{3} << 61);
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "swizzle mode 3");
  x.desc_b = desc(0, 128, 256, 0);
  x.id = idesc(1, 3, 3, 128, 16);   // e2m3 is not a .kind::f16 type
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "Tables 51-53");
}

// ---- tcgen05.cp -----------------------------------------------------------------------

// Copies an image of shared memory into Tensor Memory at column 8 with one
// tcgen05.cp, then reads the first 16 columns of every lane back with
// .32x32b: out[lane][col].
std::vector<uint32_t> run_cp(const std::string& shape, uint64_t sdesc, const std::vector<uint8_t>& image) {
  const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k(.param .u64 pimg, .param .u64 pout, .param .u64 sdesc)
{
    .reg .pred %p<4>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<12>;
    .shared .align 1024 .b8 smem[4096];
    .shared .align 8 .b64 bar;
    .shared .align 4 .b32 slot;
    ld.param.u64 %rd1, [pimg];
    ld.param.u64 %rd2, [pout];
    ld.param.u64 %rd3, [sdesc];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, smem;
    shl.b32 %r3, %r1, 2;
COPY:
    setp.ge.u32 %p1, %r3, 4096;
    @%p1 bra COPIED;
    cvt.u64.u32 %rd4, %r3;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.u32 %r4, [%rd5];
    add.u32 %r5, %r2, %r3;
    st.shared.u32 [%r5], %r4;
    add.u32 %r3, %r3, 512;
    bra COPY;
COPIED:
    shr.u32 %r6, %r1, 5;
    setp.eq.u32 %p2, %r6, 0;
    @%p2 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [slot], 32;
    setp.eq.u32 %p3, %r1, 0;
    mov.u32 %r7, bar;
    @%p3 mbarrier.init.shared::cta.b64 [%r7], 1;
    fence.proxy.async.shared::cta;
    bar.sync 0;
    ld.shared.u32 %r8, [slot];
    shr.u32 %r9, %r2, 4;
    cvt.u64.u32 %rd6, %r9;
    add.u64 %rd3, %rd3, %rd6;
    tcgen05.fence::after_thread_sync;
    add.u32 %r10, %r8, 8;
    @%p3 tcgen05.cp.cta_group::1.)" + shape + R"( [%r10], %rd3;
    @%p3 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r7];
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r7], 0;
    @!%p1 bra WAIT;
    tcgen05.fence::after_thread_sync;
    shl.b32 %r11, %r6, 21;
    add.u32 %r11, %r11, %r8;
    tcgen05.ld.sync.aligned.32x32b.x16.b32 {%r20, %r21, %r22, %r23, %r24, %r25, %r26, %r27, %r28, %r29, %r30, %r31, %r32, %r33, %r34, %r35}, [%r11];
    tcgen05.wait::ld.sync.aligned;
    mul.wide.u32 %rd7, %r1, 64;
    add.u64 %rd8, %rd2, %rd7;
    st.global.v4.u32 [%rd8], {%r20, %r21, %r22, %r23};
    st.global.v4.u32 [%rd8+16], {%r24, %r25, %r26, %r27};
    st.global.v4.u32 [%rd8+32], {%r28, %r29, %r30, %r31};
    st.global.v4.u32 [%rd8+48], {%r32, %r33, %r34, %r35};
    tcgen05.fence::before_thread_sync;
    bar.sync 0;
    @%p2 tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r8, 32;
    @%p2 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
    ret;
}
)";
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/b200");
  auto m = ptx::parse(ptx);
  std::vector<uint8_t> img(4096, 0);
  std::memcpy(img.data(), image.data(), std::min<size_t>(4096, image.size()));
  const uint64_t pimg = mem.alloc(img.size()), pout = mem.alloc(128 * 64);
  mem.write(pimg, img.data(), img.size());
  std::vector<uint32_t> zero(128 * 16, 0);
  mem.write(pout, zero.data(), zero.size() * 4);
  LaunchConfig cfg;
  cfg.block = {128, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(pimg), arg_u64(pout), arg_u64(sdesc)}, mem, prof);
  std::vector<uint32_t> out(128 * 16);
  mem.read(pout, out.data(), out.size() * 4);
  return out;
}

// A byte of row r, byte b of a copy's source matrix.
uint8_t cp_byte(int r, int b) { return static_cast<uint8_t>(r * 31 + b * 7 + 1); }
uint32_t cp_word(int r, int w) {
  return uint32_t(cp_byte(r, 4 * w)) | uint32_t(cp_byte(r, 4 * w + 1)) << 8 |
         uint32_t(cp_byte(r, 4 * w + 2)) << 16 | uint32_t(cp_byte(r, 4 * w + 3)) << 24;
}

// .128x128b: 128 rows of 16 bytes, K-major in 8-row core matrices (SBO 128),
// row r to lane r, its 16 bytes to four columns.
VTEST(tcgen05_cp_128x128b_row_per_lane) {
  std::vector<uint8_t> img;
  for (int r = 0; r < 128; ++r)
    for (int b = 0; b < 16; ++b) put(img, canonical(Major::K, 0, 1, 2048, 128, r, b), cp_byte(r, b), 1);
  const auto out = run_cp("128x128b", desc(0, 2048, 128, 0), img);
  for (int l = 0; l < 128; ++l)
    for (int c = 0; c < 16; ++c)
      VCHECK_EQ(out[size_t(l) * 16 + c], c >= 8 && c < 12 ? cp_word(l, c - 8) : 0u);
}

// .4x256b: four rows of 32 bytes, two core matrices along the row (LBO).
VTEST(tcgen05_cp_4x256b) {
  std::vector<uint8_t> img;
  for (int r = 0; r < 4; ++r)
    for (int b = 0; b < 32; ++b) put(img, canonical(Major::K, 0, 1, 128, 256, r, b), cp_byte(r, b), 1);
  const auto out = run_cp("4x256b", desc(0, 128, 256, 0), img);
  for (int l = 0; l < 128; ++l)
    for (int c = 0; c < 16; ++c)
      VCHECK_EQ(out[size_t(l) * 16 + c], l < 4 && c >= 8 ? cp_word(l, c - 8) : 0u);
}

// .32x128b.warpx4: 32 rows, each copied into all four warps' lane quarters --
// how scale factors reach every partition.
VTEST(tcgen05_cp_32x128b_warpx4_fills_every_quarter) {
  std::vector<uint8_t> img;
  for (int r = 0; r < 32; ++r)
    for (int b = 0; b < 16; ++b) put(img, canonical(Major::K, 0, 1, 512, 128, r, b), cp_byte(r, b), 1);
  const auto out = run_cp("32x128b.warpx4", desc(0, 512, 128, 0), img);
  for (int l = 0; l < 128; ++l)
    for (int c = 0; c < 16; ++c)
      VCHECK_EQ(out[size_t(l) * 16 + c], c >= 8 && c < 12 ? cp_word(l % 32, c - 8) : 0u);
}

VTEST(tcgen05_cp_refuses_warpx2) {
  VCHECK_CONTAINS(VCAPTURE(Error, run_cp("64x128b.warpx2::02_13", desc(0, 1024, 128, 0), {})).message(),
                  "64x128b.warpx2");
}

// ---- tcgen05.mma over a CTA pair ---------------------------------------------------

// M = 256 (layout A): each CTA supplies half of A's rows and half of B's
// columns from its own shared memory, and holds its 128 rows of D.
VTEST(tcgen05_mma_cta_pair_m256) {
  const int M = 256, N = 64, K = 16;
  auto A = [](int m, int k) { return val(m, k, 2); };
  auto B = [](int k, int n) { return val(k, n, 9); };
  const Operands o = images(2, M, N, K, 2, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int k) { return f16_bits(A(m, k)); },
                            [&](int n, int k) { return f16_bits(B(k, n)); });
  Mma x;
  x.group = 2;
  x.id = idesc(1, 0, 0, M, N);
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_a = o.a;
  x.smem_b = o.b;
  check_d_f32(x.run(), 2, M, N, K, A, B);
}

// M = 128 over a pair (layout B): 64 rows per CTA, the upper half of N in
// lanes 64-127.
VTEST(tcgen05_mma_cta_pair_m128) {
  const int M = 128, N = 64, K = 16;
  auto A = [](int m, int k) { return val(m, k, 4); };
  auto B = [](int k, int n) { return val(k, n, 1); };
  const Operands o = images(2, M, N, K, 2, Major::K, Major::K, 0, 0, 128, 256, 128, 256,
                            [&](int m, int k) { return f16_bits(A(m, k)); },
                            [&](int n, int k) { return f16_bits(B(k, n)); });
  Mma x;
  x.group = 2;
  x.id = idesc(1, 0, 0, M, N);
  x.desc_a = desc(0, 128, 256, 0);
  x.desc_b = desc(0, 128, 256, 0);
  x.smem_a = o.a;
  x.smem_b = o.b;
  check_d_f32(x.run(), 2, M, N, K, A, B);
}

VTEST(tcgen05_cta_group_2_needs_a_cta_pair) {
  // A pair of CTAs, but launched as clusters of one.
  const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k()
{
    .shared .align 4 .b32 slot;
    tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [slot], 32;
    ret;
}
)";
  MemoryManager mem{1 << 20};
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  cfg.grid = {2, 1, 1};
  auto err = VCAPTURE(Error, exec::launch(ptx::parse(ptx).entries[0], cfg, {}, mem, load_gpu("nvidia/b200")));
  VCHECK_CONTAINS(err.message(), "has no peer");
}

// ---- around the tensor core ------------------------------------------------------

// clusterlaunchcontrol: a block that cancels the launch of blocks not yet
// started does their work instead, and they never run. On one host thread the
// first block takes every other one.
VTEST(clusterlaunchcontrol_takes_over_blocks_that_have_not_started) {
  setenv("VGPU_THREADS", "1", 1);
  const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .reg .b128 %q;
    .shared .align 16 .b8 resp[16];
    .shared .align 8 .b64 bar;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, bar;
    mov.u32 %r3, resp;
    mov.u32 %r10, 0;
    mbarrier.init.shared::cta.b64 [%r2], 1;
LOOP:
    mov.u32 %r4, %ctaid.x;
    add.u32 %r4, %r4, 1;
    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r4;
    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r2], 16;
    clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%r3], [%r2];
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r2], %r10;
    @!%p1 bra WAIT;
    xor.b32 %r10, %r10, 1;
    ld.shared.b128 %q, [%r3];
    clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %p2, %q;
    @!%p2 bra DONE;
    clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%r1, %r5, %r6, _}, %q;
    bra LOOP;
DONE:
    ret;
}
)";
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(6 * 4);
  std::vector<uint32_t> zero(6, 0);
  mem.write(out, zero.data(), 24);
  LaunchConfig cfg;
  cfg.grid = {6, 1, 1};
  exec::launch(ptx::parse(ptx).entries[0], cfg, {arg_u64(out)}, mem, load_gpu("nvidia/b200"));
  unsetenv("VGPU_THREADS");
  std::vector<uint32_t> got(6);
  mem.read(out, got.data(), 24);
  for (int i = 0; i < 6; ++i) VCHECK_EQ(got[i], 1u);   // every block's work, done by block 0
}

// .b128 registers: loaded, stored and moved as their two halves.
VTEST(b128_registers_round_trip_through_memory) {
  const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k(.param .u64 buf)
{
    .reg .b64 %rd<4>;
    .reg .b128 %q<2>;
    ld.param.u64 %rd1, [buf];
    ld.global.b128 %q0, [%rd1];
    st.global.b128 [%rd1+16], %q0;
    ret;
}
)";
  MemoryManager mem{1 << 20};
  const uint64_t buf = mem.alloc(32);
  const uint64_t in[4] = {0x0123456789abcdefull, 0xfedcba9876543210ull, 0, 0};
  mem.write(buf, in, 32);
  exec::launch(ptx::parse(ptx).entries[0], LaunchConfig{}, {arg_u64(buf)}, mem, load_gpu("nvidia/b200"));
  uint64_t got[4];
  mem.read(buf, got, 32);
  VCHECK_EQ(got[2], in[0]);
  VCHECK_EQ(got[3], in[1]);
}

// TMA's .cta_group::2: each CTA of a pair loads its own half of a tile into
// its own shared memory, and both complete on the even CTA's barrier, reached
// the way CUTLASS reaches it -- by clearing bit 24 of the CTA's own barrier
// address, which holds its rank.
VTEST(tma_cta_group_2_completes_on_the_even_ctas_barrier) {
  MemoryManager mem{1 << 20};
  const uint64_t g = mem.alloc(16 * 8 * 4), out = mem.alloc(2 * 65 * 4);
  for (int y = 0; y < 16; ++y)
    for (int x = 0; x < 8; ++x) mem.store_scalar(g + uint64_t(y * 8 + x) * 4, 4, f32_bits(float(y * 100 + x)));
  exec::TensorMap t;
  t.address = g;
  t.rank = 2;
  t.type = exec::TmapType::F32;
  t.dim = {8, 16, 1, 1, 1};
  t.stride = {4, 32, 0, 0, 0};
  t.box = {8, 8, 1, 1, 1};
  t.elem_stride = {1, 1, 1, 1, 1};
  std::vector<uint8_t> map(128);
  t.encode(map.data());
  const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k(.param .align 64 .b8 tmap[128], .param .u64 out)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .shared .align 128 .b8 tile[256];
    .shared .align 8 .b64 bar;
    ld.param.u64 %rd1, [out];
    mov.b64 %rd2, tmap;
    cvta.param.u64 %rd3, %rd2;
    mov.u32 %r1, tile;
    mov.u32 %r2, bar;
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %cluster_ctarank;
    setp.eq.u32 %p1, %r3, 0;
    setp.eq.u32 %p2, %r4, 0;
    and.pred %p3, %p1, %p2;
    @%p3 mbarrier.init.shared::cta.b64 [%r2], 1;
    @%p3 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r2], 512;
    barrier.cluster.arrive;
    barrier.cluster.wait;
    and.b32 %r5, %r2, 0xFEFFFFFF;
    mov.u32 %r6, 0;
    shl.b32 %r7, %r4, 3;
    @%p1 cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%rd3, {%r6, %r7}], [%r5];
    @!%p2 bra WAITED;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r2], 0;
    @!%p1 bra WAIT;
WAITED:
    barrier.cluster.arrive;
    barrier.cluster.wait;
    // out[rank*65 + t] = tile word t; out[rank*65 + 64] = tile's address >> 24.
    shl.b32 %r8, %r3, 2;
    add.u32 %r9, %r1, %r8;
    ld.shared.u32 %r10, [%r9];
    mad.lo.u32 %r11, %r4, 65, %r3;
    mul.wide.u32 %rd4, %r11, 4;
    add.u64 %rd5, %rd1, %rd4;
    st.global.u32 [%rd5], %r10;
    shr.u32 %r12, %r1, 24;
    mad.lo.u32 %r13, %r4, 65, 64;
    mul.wide.u32 %rd6, %r13, 4;
    add.u64 %rd7, %rd1, %rd6;
    st.global.u32 [%rd7], %r12;
    barrier.cluster.arrive;
    barrier.cluster.wait;
    ret;
}
)";
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};
  cfg.grid = {2, 1, 1};
  cfg.cluster = {2, 1, 1};
  exec::launch(ptx::parse(ptx).entries[0], cfg, {map, arg_u64(out)}, mem, load_gpu("nvidia/b200"));
  std::vector<uint32_t> got(2 * 65);
  mem.read(out, got.data(), got.size() * 4);
  for (int r = 0; r < 2; ++r) {
    for (int i = 0; i < 64; ++i) {
      const float want = float((r * 8 + i / 8) * 100 + i % 8);
      VCHECK_EQ(bits_f32(got[r * 65 + i]), want);
    }
    VCHECK_EQ(got[r * 65 + 64], uint32_t(r));   // the rank, in bits 24 and up
  }
}

// .tile::gather4 / .tile::scatter4 (5.5.3.4): four rows of a 2D tensor at one
// x, packed in shared memory one after another; the store puts them back as
// four rows elsewhere.
VTEST(tma_gather4_and_scatter4_move_four_rows) {
  MemoryManager mem{1 << 20};
  constexpr int W = 16, H = 10;
  const uint64_t src = mem.alloc(W * H * 4), dst = mem.alloc(W * H * 4), out = mem.alloc(32 * 4);
  std::vector<uint32_t> zero(W * H, 0);
  mem.write(dst, zero.data(), zero.size() * 4);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) mem.store_scalar(src + uint64_t(y * W + x) * 4, 4, uint64_t(y * 100 + x));
  auto map_of = [](uint64_t a) {
    exec::TensorMap t;
    t.address = a;
    t.rank = 2;
    t.type = exec::TmapType::U32;
    t.dim = {W, H, 1, 1, 1};
    t.stride = {4, W * 4, 0, 0, 0};
    t.box = {8, 1, 1, 1, 1};
    t.elem_stride = {1, 1, 1, 1, 1};
    std::vector<uint8_t> m(128);
    t.encode(m.data());
    return m;
  };
  const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k(.param .align 64 .b8 ms[128], .param .align 64 .b8 md[128], .param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .shared .align 128 .b8 tile[128];
    .shared .align 8 .b64 bar;
    ld.param.u64 %rd1, [out];
    mov.b64 %rd2, ms;
    cvta.param.u64 %rd3, %rd2;
    mov.b64 %rd4, md;
    cvta.param.u64 %rd5, %rd4;
    mov.u32 %r1, tile;
    mov.u32 %r2, bar;
    mov.u32 %r3, %tid.x;
    setp.eq.u32 %p1, %r3, 0;
    @%p1 mbarrier.init.shared::cta.b64 [%r2], 1;
    @%p1 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r2], 128;
    mov.u32 %r4, 2;
    mov.u32 %r5, 3;
    mov.u32 %r6, 0;
    mov.u32 %r7, 7;
    mov.u32 %r8, 5;
    @%p1 cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes [%r1], [%rd3, {%r4, %r5, %r6, %r7, %r8}], [%r2];
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r2], 0;
    @!%p1 bra WAIT;
    shl.b32 %r9, %r3, 2;
    add.u32 %r10, %r1, %r9;
    ld.shared.u32 %r11, [%r10];
    mul.wide.u32 %rd6, %r3, 4;
    add.u64 %rd7, %rd1, %rd6;
    st.global.u32 [%rd7], %r11;
    bar.sync 0;
    setp.eq.u32 %p1, %r3, 0;
    mov.u32 %r4, 4;
    mov.u32 %r5, 9;
    mov.u32 %r6, 1;
    mov.u32 %r7, 6;
    mov.u32 %r8, 2;
    @%p1 cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group [%rd5, {%r4, %r5, %r6, %r7, %r8}], [%r1];
    @%p1 cp.async.bulk.commit_group;
    @%p1 cp.async.bulk.wait_group 0;
    ret;
}
)";
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(ptx::parse(ptx).entries[0], cfg, {map_of(src), map_of(dst), arg_u64(out)}, mem,
               load_gpu("nvidia/b200"));
  const int rows[4] = {3, 0, 7, 5}, to[4] = {9, 1, 6, 2};
  std::vector<uint32_t> got(32), d(W * H);
  mem.read(out, got.data(), got.size() * 4);
  mem.read(dst, d.data(), d.size() * 4);
  for (int i = 0; i < 32; ++i) VCHECK_EQ(got[i], uint32_t(rows[i / 8] * 100 + 2 + i % 8));
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      uint32_t want = 0;
      for (int r = 0; r < 4; ++r)
        if (y == to[r] && x >= 4 && x < 12) want = uint32_t(rows[r] * 100 + 2 + (x - 4));
      VCHECK_EQ(d[size_t(y) * W + x], want);
    }
}

// ---- TMA's packed sub-byte types --------------------------------------------------

namespace {
// Values packed contiguously, value i at bit bits * i (5.5.1.1: "packed
// contiguously in the global memory").
std::vector<uint8_t> pack_bits(const std::vector<uint8_t>& v, int bits) {
  std::vector<uint8_t> out((v.size() * bits + 7) / 8, 0);
  for (size_t i = 0; i < v.size(); ++i)
    for (int b = 0; b < bits; ++b)
      if (v[i] >> b & 1) out[(i * bits + b) / 8] |= static_cast<uint8_t>(1u << ((i * bits + b) % 8));
  return out;
}

// One 2D tensor load or store through `map` at {x, y}: shared memory (1024
// bytes, 128-byte swizzle aligned) starts as `smem`, the load expects `tx`
// bytes, and the tile afterwards is returned.
std::vector<uint8_t> tma_packed(MemoryManager& mem, const std::vector<uint8_t>& map, bool load, uint32_t x,
                                uint32_t y, uint32_t tx, const std::vector<uint8_t>& smem) {
  const uint64_t in = mem.alloc(1024), out = mem.alloc(1024);
  std::vector<uint8_t> init(1024, 0xEE);
  std::copy(smem.begin(), smem.end(), init.begin());
  mem.write(in, init.data(), init.size());
  const std::string copy =
      load ? "@%p1 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%rd3, {%r4, %r5}], [%r2];\n"
             "WAIT:\n    mbarrier.try_wait.parity.shared::cta.b64 %p2, [%r2], 0;\n    @!%p2 bra WAIT;\n"
           : "fence.proxy.async.shared::cta;\n    bar.sync 0;\n"
             "    @%p1 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd3, {%r4, %r5}], [%r1];\n"
             "    @%p1 cp.async.bulk.commit_group;\n    @%p1 cp.async.bulk.wait_group 0;\n";
  const std::string ptx = std::string(kHeader100a) + R"(
.visible .entry k(.param .align 64 .b8 m[128], .param .u64 pin, .param .u64 pout, .param .u32 px, .param .u32 py, .param .u32 ptx)
{
    .reg .pred %p<3>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<12>;
    .shared .align 1024 .b8 tile[1024];
    .shared .align 8 .b64 bar;
    ld.param.u64 %rd1, [pin];
    ld.param.u64 %rd8, [pout];
    ld.param.u32 %r4, [px];
    ld.param.u32 %r5, [py];
    ld.param.u32 %r6, [ptx];
    mov.b64 %rd2, m;
    cvta.param.u64 %rd3, %rd2;
    mov.u32 %r1, tile;
    mov.u32 %r2, bar;
    mov.u32 %r3, %tid.x;
    shl.b32 %r7, %r3, 3;
    add.u32 %r8, %r1, %r7;
    cvt.u64.u32 %rd4, %r7;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.u64 %rd6, [%rd5];
    st.shared.u64 [%r8], %rd6;
    setp.eq.u32 %p1, %r3, 0;
    @%p1 mbarrier.init.shared::cta.b64 [%r2], 1;
    bar.sync 0;
    @%p1 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r2], %r6;
    )" + copy + R"(
    bar.sync 0;
    ld.shared.u64 %rd6, [%r8];
    add.u64 %rd9, %rd8, %rd4;
    st.global.u64 [%rd9], %rd6;
    ret;
}
)";
  LaunchConfig cfg;
  cfg.block = {128, 1, 1};
  exec::launch(ptx::parse(ptx).entries[0], cfg,
               {map, arg_u64(in), arg_u64(out), arg_u32(x), arg_u32(y), arg_u32(load ? tx : 0)}, mem,
               load_gpu("nvidia/b200"));
  std::vector<uint8_t> tile(1024);
  mem.read(out, tile.data(), tile.size());
  return tile;
}

uint8_t six(int x, int y) { return static_cast<uint8_t>((x * 7 + y * 13 + 5) & 63); }
}  // namespace

// cuTensorMapEncodeTiled's rules for the packed types, from cuda.h.
VTEST(tma_packed_types_encoder_rules) {
  alignas(64) uint8_t buf[128];
  std::string why;
  const unsigned long long dims[2] = {256, 4}, strides[1] = {192};
  const unsigned box[2] = {128, 2}, es[2] = {1, 1};
  auto enc = [&](unsigned type, uint64_t addr, const unsigned long long* d, const unsigned long long* st,
                 const unsigned* b, unsigned swizzle) {
    return exec::encode_tiled(buf, type, 2, reinterpret_cast<void*>(addr), d, st, b, es, 0, swizzle, 0, 0, &why);
  };
  VCHECK(enc(15, 0x10000, dims, strides, box, 3) == exec::TmapResult::Ok);
  exec::TensorMap m;
  VCHECK(m.decode(buf));
  VCHECK(m.type == exec::TmapType::U6x16Align16);
  VCHECK(enc(15, 0x10010, dims, strides, box, 3) == exec::TmapResult::Invalid);
  VCHECK_CONTAINS(why, "32-byte aligned");
  const unsigned long long odd_dims[2] = {192, 4};
  VCHECK(enc(15, 0x10000, odd_dims, strides, box, 3) == exec::TmapResult::Invalid);
  VCHECK_CONTAINS(why, "multiple of 128");
  const unsigned small_box[2] = {64, 2};
  VCHECK(enc(14, 0x10000, dims, strides, small_box, 0) == exec::TmapResult::Invalid);
  VCHECK_CONTAINS(why, "boxDim[0] must be 128");
  const unsigned long long stride48[1] = {208};
  VCHECK(enc(15, 0x10000, dims, stride48, box, 0) == exec::TmapResult::Invalid);
  VCHECK_CONTAINS(why, "multiples of 32");
  VCHECK(enc(15, 0x10000, dims, strides, box, 1) == exec::TmapResult::Invalid);   // 32B swizzle
  VCHECK_CONTAINS(why, "swizzle mode");
  VCHECK(enc(14, 0x10000, dims, strides, box, 6) == exec::TmapResult::Invalid);   // 16U4_ALIGN16B, ATOM_64B
  VCHECK(enc(15, 0x10000, dims, strides, box, 6) == exec::TmapResult::Ok);        // 16U6 stores may use it
  const unsigned long long d13[2] = {63, 4}, s13[1] = {32};
  const unsigned b13[2] = {32, 1};
  VCHECK(enc(13, 0x10000, d13, s13, b13, 0) == exec::TmapResult::Invalid);
  VCHECK_CONTAINS(why, "multiple of 2");
  const unsigned b13_16[2] = {16, 1};   // eight bytes: not a multiple of 16
  const unsigned long long d64[2] = {64, 4};
  VCHECK(enc(13, 0x10000, d64, s13, b13_16, 0) == exec::TmapResult::Invalid);
  VCHECK_CONTAINS(why, "multiple of 16 bytes");
  VCHECK(exec::encode_tiled(buf, 15, 2, reinterpret_cast<void*>(0x10000), dims, strides, box, es, 0, 3, 0, 1, &why) ==
         exec::TmapResult::Invalid);
  VCHECK_CONTAINS(why, "NaN out-of-bounds fill");
}

// .b6x16_p32 load with the 128-byte swizzle: each 16 values' 12 packed bytes
// open a 16-byte slot, the other 4 bytes left as they were.
VTEST(tma_load_b6x16_p32_pads_each_group) {
  MemoryManager mem{1 << 20};
  const int W = 256, H = 4;
  std::vector<uint8_t> vals(W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) vals[size_t(y) * W + x] = six(x, y);
  const std::vector<uint8_t> packed = pack_bits(vals, 6);   // 192 bytes a row
  const uint64_t raw = mem.alloc(packed.size() + 64), g = (raw + 31) / 32 * 32;
  mem.write(g, packed.data(), packed.size());
  alignas(64) uint8_t buf[128];
  std::string why;
  const unsigned long long dims[2] = {W, H}, strides[1] = {W * 6 / 8};
  const unsigned box[2] = {128, 2}, es[2] = {1, 1};
  VCHECK(exec::encode_tiled(buf, 15, 2, reinterpret_cast<void*>(g), dims, strides, box, es, 0, 3, 0, 0, &why) ==
         exec::TmapResult::Ok);
  const auto tile = tma_packed(mem, std::vector<uint8_t>(buf, buf + 128), true, 128, 1, 16 * 16, {});
  for (int grp = 0; grp < 16; ++grp) {
    const int y = 1 + grp / 8, x0 = 128 + 16 * (grp % 8);
    std::vector<uint8_t> v(vals.begin() + y * W + x0, vals.begin() + y * W + x0 + 16);
    const std::vector<uint8_t> want = pack_bits(v, 6);
    const uint32_t off = grp * 16, at = off ^ (((off >> 7) & 7) << 4);
    for (int b = 0; b < 16; ++b) VCHECK_EQ(int(tile[at + b]), b < 12 ? int(want[b]) : 0xEE);
  }
  for (size_t b = 256; b < tile.size(); ++b) VCHECK_EQ(int(tile[b]), 0xEE);
}

// .b4x16: sixteen values in eight bytes, copied as they are; a group past the
// tensor's edge reads as zeros.
VTEST(tma_load_b4x16_copies_packed_groups) {
  MemoryManager mem{1 << 20};
  const int W = 64, H = 2;
  std::vector<uint8_t> vals(W * H);
  for (int i = 0; i < W * H; ++i) vals[i] = static_cast<uint8_t>((i * 5 + 3) & 15);
  const std::vector<uint8_t> packed = pack_bits(vals, 4);   // 32 bytes a row
  const uint64_t g = mem.alloc(packed.size());
  mem.write(g, packed.data(), packed.size());
  alignas(64) uint8_t buf[128];
  std::string why;
  const unsigned long long dims[2] = {W, H}, strides[1] = {32};
  const unsigned box[2] = {32, 2}, es[2] = {1, 1};
  VCHECK(exec::encode_tiled(buf, 13, 2, reinterpret_cast<void*>(g), dims, strides, box, es, 0, 0, 0, 0, &why) ==
         exec::TmapResult::Ok);
  const auto tile = tma_packed(mem, std::vector<uint8_t>(buf, buf + 128), true, 48, 0, 4 * 8, {});
  for (int grp = 0; grp < 4; ++grp) {
    const int y = grp / 2, x0 = 48 + 16 * (grp % 2);
    for (int b = 0; b < 8; ++b) {
      const int want = x0 >= W ? 0 : packed[size_t(y) * 32 + x0 / 2 + b];
      VCHECK_EQ(int(tile[grp * 8 + b]), want);
    }
  }
  VCHECK_EQ(int(tile[32]), 0xEE);
}

// A store of type 15 is .b6p2x16: sixteen bytes, each value in bits 0-5 and
// bits 6-7 dropped, packed into twelve.
VTEST(tma_store_b6p2x16_packs_the_low_six_bits) {
  MemoryManager mem{1 << 20};
  const int W = 128;
  const uint64_t raw = mem.alloc(256), g = (raw + 31) / 32 * 32;
  std::vector<uint8_t> zero(96, 0);
  mem.write(g, zero.data(), zero.size());
  alignas(64) uint8_t buf[128];
  std::string why;
  const unsigned long long dims[2] = {W, 1}, strides[1] = {96};
  const unsigned box[2] = {128, 1}, es[2] = {1, 1};
  VCHECK(exec::encode_tiled(buf, 15, 2, reinterpret_cast<void*>(g), dims, strides, box, es, 0, 0, 0, 0, &why) ==
         exec::TmapResult::Ok);
  std::vector<uint8_t> smem(128), vals(128);
  for (int i = 0; i < 128; ++i) {
    vals[i] = six(i, 3);
    smem[i] = static_cast<uint8_t>(vals[i] | ((i % 4) << 6));
  }
  tma_packed(mem, std::vector<uint8_t>(buf, buf + 128), false, 0, 0, 0, smem);
  std::vector<uint8_t> got(96);
  mem.read(g, got.data(), got.size());
  VCHECK(got == pack_bits(vals, 6));
}

VTEST(tma_packed_types_refuse_what_the_isa_rules_out) {
  MemoryManager mem{1 << 20};
  const uint64_t raw = mem.alloc(1024), g = (raw + 31) / 32 * 32;
  alignas(64) uint8_t buf[128];
  std::string why;
  const unsigned long long dims[2] = {256, 2}, strides[1] = {128};
  const unsigned box[2] = {128, 1}, es[2] = {1, 1};
  VCHECK(exec::encode_tiled(buf, 14, 2, reinterpret_cast<void*>(g), dims, strides, box, es, 0, 0, 0, 0, &why) ==
         exec::TmapResult::Ok);
  const std::vector<uint8_t> map(buf, buf + 128);
  VCHECK_CONTAINS(VCAPTURE(Error, tma_packed(mem, map, true, 64, 0, 128, {})).message(), "multiple of 128");
  VCHECK_CONTAINS(VCAPTURE(Error, tma_packed(mem, map, false, 0, 0, 0, {})).message(), ".b4x16_p64");
}

VTEST_MAIN
