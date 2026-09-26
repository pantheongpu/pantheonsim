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
#include <cstring>
#include <functional>
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
    const std::string a_operand = a_tmem.empty() ? "%rd5" : "[%r62]";
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
    setp.ne.u32 %p5, %r50, 0xFFFFFFFF;          // a predicate that is true
    setp.eq.u32 %p6, %r30, 0;
    and.pred %p4, %p3, %p6;                     // thread 0 of the even CTA issues
    setp.)" + (enable_d == "0" ? "ne" : "eq") + R"(.u32 %p7, %r1, %r1;
    @%p4 tcgen05.mma.cta_group::)" + g + ".kind::" + kind + " [%r61], " + a_operand + ", %rd6, %r50, " +
                            (lanes_off.empty() ? "" : lanes_off + ", ") + "%p7" + scale_tail + R"(;
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
  VCHECK_CONTAINS(parse("tcgen05.cp.cta_group::1.128x256b [a], d;").message(), "tcgen05.cp");
  VCHECK_CONTAINS(parse("tcgen05.mma.sp.cta_group::1.kind::f16 [a], d, d, [b], a, p;").message(), "sparse");
  VCHECK_CONTAINS(parse("tcgen05.mma.ws.cta_group::1.kind::f16 [a], d, d, a, p;").message(), "weight-stationary");
  VCHECK_CONTAINS(parse("tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [a], d, d, a, [b], [b], p;").message(),
                  "block-scaled");
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
  VCHECK_CONTAINS(VCAPTURE(Error, x.run()).message(), "Table 51");
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

VTEST_MAIN
