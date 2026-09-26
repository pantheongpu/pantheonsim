// Hopper (sm_90a): the warpgroup MMA, and the TMA data path around it --
// mbarrier transaction counts, bulk and tensor copies, thread-block clusters.
//
// The shared-memory tiles here are laid out from the worked examples in the
// PTX ISA itself (section 9.7.17.5.1.2.1.3, figures 169-173), each written
// down as the exact CuTe layout the ISA gives for it, and the register
// fragments from figures 151-158. Those are a different statement of the rules
// than the canonical-layout formulas the interpreter implements, so a mistake
// in one shows up as a disagreement. The e2e test wgmma_cute.cu checks the
// same instruction against CuTe's own layouts, including the 128-byte swizzle
// these examples do not cover.
#include <cmath>
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

const char* kHeader90a = ".version 8.3\n.target sm_90a\n.address_size 64\n";

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
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
uint16_t bf16_bits(float f) { return static_cast<uint16_t>(f32_bits(f) >> 16); }
uint16_t f16_bits(float f) {
  if (f == 0.0f) return std::signbit(f) ? 0x8000 : 0;
  int e = 0;
  const float m = std::frexp(std::fabs(f), &e);   // f = m * 2^e, m in [0.5, 1)
  const uint16_t mant = static_cast<uint16_t>((m * 2.0f - 1.0f) * 1024.0f);
  return static_cast<uint16_t>((f < 0 ? 0x8000 : 0) | ((e - 1 + 15) << 10) | mant);
}
float f16_value(uint16_t h) {
  const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
  const float v = e ? std::ldexp(1.0f + m / 1024.0f, e - 15) : std::ldexp(m / 1024.0f, -14);
  return (h & 0x8000) ? -v : v;
}

// A small multiple of 1/2 in [-2, 2], different for each (i, j).
float val(int i, int j, int salt) { return static_cast<float>(((i * 7 + j * 13 + salt) % 9) - 4) * 0.5f; }

// The fragment layouts of figures 151-158, for thread t of the warpgroup.
// A: register r, element e within it (per_reg elements of `bits` each).
void a_pos(int t, int r, int e, int per_reg, int* row, int* col) {
  const int w = t / 32, g = (t % 32) / 4, q = t % 4;
  *row = 16 * w + g + 8 * (r % 2);
  *col = q * per_reg + e + (r / 2) * per_reg * 4;
}
// D: accumulator element e of thread t.
void d_pos(int t, int e, int* row, int* col) {
  const int w = t / 32, g = (t % 32) / 4, q = t % 4, j = e / 4, s = e % 4;
  *row = 16 * w + g + 8 * (s / 2);
  *col = 8 * j + 2 * q + (s % 2);
}

// Runs one wgmma over a 128-thread block. `smem_a` and `smem_b` are copied to
// shared offsets 0 and 8192 (each at most 8 KiB); the descriptors' start
// fields are relative to them and the kernel adds the real base. `a_words`
// holds each thread's four A registers when A comes from registers, and
// `d_in` each thread's accumulator registers before the instruction.
struct Wgmma {
  std::string form;                  // e.g. "m64n16k8.f32.tf32.tf32"
  bool a_regs = true;
  std::string tail;                  // operands after scale-d, e.g. ", 1, 1"
  std::string scale_d = "1";
  int d_regs = 0;
  std::vector<uint8_t> smem_a, smem_b;
  uint64_t desc_a = 0, desc_b = 0;
  std::vector<uint32_t> a_words;     // 128 x 4
  std::vector<uint32_t> d_in;        // 128 x d_regs
  uint32_t threads = 128;
  std::string target = kHeader90a;
  std::string device = "nvidia/h100";

  std::vector<uint32_t> run() const {
    std::string dl, loads, stores;
    for (int i = 0; i < d_regs; ++i) {
      dl += (i ? ", " : "") + std::string("%d") + std::to_string(i);
      loads += "    ld.global.u32 %d" + std::to_string(i) + ", [%rd12+" + std::to_string(4 * i) + "];\n";
      stores += "    st.global.u32 [%rd12+" + std::to_string(4 * i) + "], %d" + std::to_string(i) + ";\n";
    }
    const std::string a = a_regs ? "{%r10, %r11, %r12, %r13}" : "%rd5";
    const std::string ptx = target + R"(
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pw, .param .u64 pd,
                  .param .u64 da, .param .u64 db)
{
    .reg .pred %p<4>;
    .reg .b32 %r<40>;
    .reg .b32 %d<130>;
    .reg .b64 %rd<20>;
    .shared .align 1024 .b8 smem[16384];
    ld.param.u64 %rd1, [pa];
    ld.param.u64 %rd2, [pb];
    ld.param.u64 %rd3, [pw];
    ld.param.u64 %rd4, [pd];
    ld.param.u64 %rd5, [da];
    ld.param.u64 %rd6, [db];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, smem;
    shl.b32 %r3, %r1, 2;
COPY:
    setp.ge.u32 %p1, %r3, 8192;
    @%p1 bra COPIED;
    cvt.u64.u32 %rd7, %r3;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r4, [%rd8];
    add.u32 %r5, %r2, %r3;
    st.shared.u32 [%r5], %r4;
    add.u64 %rd8, %rd2, %rd7;
    ld.global.u32 %r4, [%rd8];
    add.u32 %r5, %r5, 8192;
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
    add.u64 %rd6, %rd6, 512;
    mul.wide.u32 %rd9, %r1, )" + std::to_string(4 * d_regs) + R"(;
    add.u64 %rd12, %rd4, %rd9;
)" + loads + R"(
    setp.ne.u32 %p2, %r1, 999;
    wgmma.fence.sync.aligned;
    wgmma.mma_async.sync.aligned.)" + form + " {" + dl + "}, " + a + ", %rd6, " + scale_d + tail + R"(;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;
)" + stores + R"(
    ret;
}
)";
    MemoryManager mem{1 << 22};
    DeviceProfile prof = load_gpu(device);
    auto m = ptx::parse(ptx);
    auto put = [&](const void* p, size_t n) {
      const uint64_t va = mem.alloc(n ? n : 16);
      if (n) mem.write(va, p, n);
      return va;
    };
    std::vector<uint8_t> sa = smem_a, sb = smem_b;
    sa.resize(8192);
    sb.resize(8192);
    std::vector<uint32_t> aw = a_words;
    aw.resize(4 * threads);
    std::vector<uint32_t> din = d_in;
    din.resize(size_t(d_regs) * threads);
    const uint64_t pa = put(sa.data(), sa.size()), pb = put(sb.data(), sb.size());
    const uint64_t pw = put(aw.data(), aw.size() * 4), pd = put(din.data(), din.size() * 4);
    LaunchConfig cfg;
    cfg.block = {threads, 1, 1};
    exec::launch(m.entries[0], cfg,
                 {arg_u64(pa), arg_u64(pb), arg_u64(pw), arg_u64(pd), arg_u64(desc_a), arg_u64(desc_b)},
                 mem, prof);
    std::vector<uint32_t> out(din.size());
    mem.read(pd, out.data(), out.size() * 4);
    return out;
  }
};

// A descriptor, from byte quantities (9.7.17.5.1.2.2). `start` is relative to
// the operand's shared buffer.
uint64_t desc(uint32_t start, uint32_t lbo, uint32_t sbo, uint32_t swizzle_code) {
  return uint64_t{(start & 0x3FFFF) >> 4} | (uint64_t{(lbo & 0x3FFFF) >> 4} << 16) |
         (uint64_t{(sbo & 0x3FFFF) >> 4} << 32) | (uint64_t{swizzle_code} << 62);
}

// Swizzle<B,4,3> on a byte offset: bits [4, 4+B) ^= bits [7, 7+B).
uint32_t swizzle(uint32_t off, int b) { return off ^ (((off >> 7) & ((1u << b) - 1)) << 4); }

// Writes an element into a byte image.
template <class T>
void put_elem(std::vector<uint8_t>& img, size_t byte, T v) {
  if (img.size() < byte + sizeof(T)) img.resize(byte + sizeof(T));
  std::memcpy(img.data() + byte, &v, sizeof(T));
}

// A as tf32 registers (figure 153): four registers of one element each.
std::vector<uint32_t> tf32_a_regs(const std::function<float(int, int)>& A) {
  std::vector<uint32_t> w(128 * 4);
  for (int t = 0; t < 128; ++t)
    for (int r = 0; r < 4; ++r) {
      int row, col;
      a_pos(t, r, 0, 1, &row, &col);
      w[t * 4 + r] = f32_bits(A(row, col));
    }
  return w;
}
// A as 16-bit registers (figure 151): two elements per register, low first.
std::vector<uint32_t> half_a_regs(const std::function<float(int, int)>& A, bool bf) {
  std::vector<uint32_t> w(128 * 4);
  for (int t = 0; t < 128; ++t)
    for (int r = 0; r < 4; ++r) {
      uint32_t word = 0;
      for (int e = 0; e < 2; ++e) {
        int row, col;
        a_pos(t, r, e, 2, &row, &col);
        const float v = A(row, col);
        word |= uint32_t{bf ? bf16_bits(v) : f16_bits(v)} << (16 * e);
      }
      w[t * 4 + r] = word;
    }
  return w;
}

// Checks an f32 accumulator against sum_k A(m,k) B(k,n) (+ C).
void check_f32(const std::vector<uint32_t>& out, int N, int K, const std::function<float(int, int)>& A,
               const std::function<float(int, int)>& B, const std::function<float(int, int)>& C) {
  for (int t = 0; t < 128; ++t)
    for (int e = 0; e < N / 2; ++e) {
      int row, col;
      d_pos(t, e, &row, &col);
      float want = C(row, col);
      for (int k = 0; k < K; ++k) want += A(row, k) * B(k, col);
      const float got = bits_f32(out[size_t(t) * (N / 2) + e]);
      if (got != want)
        throw vtest::Failure("D[" + std::to_string(row) + "][" + std::to_string(col) + "] = " +
                             std::to_string(got) + ", expected " + std::to_string(want));
    }
}

auto zero = [](int, int) { return 0.0f; };

}  // namespace

// ---- parsing ---------------------------------------------------------------

VTEST(wgmma_parses_the_isa_examples) {
  // The examples from 9.7.17.5.2 and 9.7.17.7, verbatim but for register names.
  const std::string ptx = std::string(kHeader90a) + R"(
.visible .entry k()
{
    .reg .b32 %r<130>;
    .reg .b64 %rd<4>;
    .reg .pred %p<2>;
    wgmma.fence.sync.aligned;
    wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%r0, %r1, %r2, %r3}, {%r4, %r5, %r6, %r7}, %rd1, 1, -1, -1, 1;
    wgmma.mma_async.sync.aligned.m64n16k8.f32.tf32.tf32 {%r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7}, %rd0, %rd1, 0, -1, -1;
    wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e5m2 {%r0, %r1}, %rd0, %rd1, %p1, -1, 1;
    wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite {%r0, %r1, %r2, %r3}, {%r4, %r5, %r6, %r7}, %rd1, 1;
    wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8 {%r0, %r1, %r2, %r3}, %rd0, %rd1, %p1;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;
    ret;
}
)";
  auto m = ptx::parse(ptx);
  VCHECK_EQ(m.entries[0].body.size(), size_t{9});
}

VTEST(wgmma_refuses_forms_the_isa_does_not_define) {
  auto parse_one = [](const std::string& target, const std::string& ins) {
    const std::string ptx = target + R"(
.visible .entry k()
{
    .reg .b32 %r<130>;
    .reg .b64 %rd<4>;
    )" + ins + R"(
    ret;
}
)";
    return VCAPTURE(Error, ptx::parse(ptx));
  };
  // Not sm_90a: ptxas refuses it, and so does this.
  VCHECK_CONTAINS(parse_one(".version 8.3\n.target sm_90\n.address_size 64\n",
                            "wgmma.fence.sync.aligned;").message(),
                  "requires .target sm_90a");
  VCHECK_CONTAINS(parse_one(".version 8.7\n.target sm_100a\n.address_size 64\n",
                            "wgmma.fence.sync.aligned;").message(),
                  "requires .target sm_90a");
  // bf16 has no f16 accumulator; tf32 is k8 only; the integer shapes skip n40.
  VCHECK_CONTAINS(parse_one(kHeader90a, "wgmma.mma_async.sync.aligned.m64n8k16.f16.bf16.bf16 "
                                        "{%r0, %r1}, %rd0, %rd1, 1, 1, 1, 0, 0;").message(),
                  "not a form the ISA defines");
  VCHECK_CONTAINS(parse_one(kHeader90a, "wgmma.mma_async.sync.aligned.m64n8k16.f32.tf32.tf32 "
                                        "{%r0, %r1, %r2, %r3}, %rd0, %rd1, 1, 1, 1;").message(),
                  "not a form the ISA defines");
  VCHECK_CONTAINS(parse_one(kHeader90a, "wgmma.mma_async.sync.aligned.m64n40k32.s32.s8.s8 "
                                        "{%r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9, %r10, %r11, %r12, "
                                        "%r13, %r14, %r15, %r16, %r17, %r18, %r19}, %rd0, %rd1, 1;").message(),
                  "not a form the ISA defines");
  // The wrong number of accumulator registers, and the wrong immediates.
  VCHECK_CONTAINS(parse_one(kHeader90a, "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
                                        "{%r0, %r1, %r2, %r3}, %rd0, %rd1, 1, 1, 1, 0, 0;").message(),
                  "accumulator arity");
  VCHECK_CONTAINS(parse_one(kHeader90a, "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                                        "{%r0, %r1, %r2, %r3}, %rd0, %rd1, 1, 1, 1;").message(),
                  "4 immediate arguments");
  VCHECK_CONTAINS(parse_one(kHeader90a, "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                                        "{%r0, %r1, %r2, %r3}, %rd0, %rd1, 1, 2, 1, 0, 0;").message(),
                  "must be 1 or -1");
  VCHECK_CONTAINS(parse_one(kHeader90a, "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                                        "{%r0, %r1, %r2, %r3}, %rd0, %rd1, 1, 1, 1, 0, 2;").message(),
                  "must be 0 or 1");
  // Sparse and single-bit forms are named, not mistaken for something else.
  VCHECK_CONTAINS(parse_one(kHeader90a, "wgmma.mma_async.sp.sync.aligned.m64n8k32.f32.f16.f16 "
                                        "{%r0, %r1, %r2, %r3}, %rd0, %rd1, %r5, 0, 1, 1, 1, 0, 0;").message(),
                  "sparse");
  VCHECK_CONTAINS(parse_one(kHeader90a, "wgmma.mma_async.sync.aligned.m64n8k256.s32.b1.b1.and.popc "
                                        "{%r0, %r1, %r2, %r3}, %rd0, %rd1, 1;").message(),
                  "single-bit");
}

// ---- shared-memory layouts, from the ISA's own examples ----------------------

// Figure 169: K-major, no swizzle, tf32.
//   Swizzle<0,4,3> o ((8,2),(4,4)):((4,32),(1,64)), LBO = 64*4, SBO = 32*4 bytes.
// As B (N = 16 rows, K = 8 of its 16 columns), with A in registers.
VTEST(wgmma_tf32_k_major_no_swizzle_figure_169) {
  auto A = [](int m, int k) { return val(m, k, 1); };
  auto B = [](int k, int n) { return val(n, k, 2); };
  Wgmma w;
  w.form = "m64n16k8.f32.tf32.tf32";
  w.tail = ", 1, 1";
  w.d_regs = 8;
  for (int n = 0; n < 16; ++n)
    for (int k = 0; k < 16; ++k) {
      const int off = 4 * (n % 8) + 32 * (n / 8) + (k % 4) + 64 * (k / 4);
      put_elem(w.smem_b, size_t(off) * 4, f32_bits(B(k, n)));
    }
  w.desc_b = desc(0, 64 * 4, 32 * 4, 0);
  w.a_words = tf32_a_regs(A);
  check_f32(w.run(), 16, 8, A, B, zero);
}

// Figure 170: K-major, 32-byte swizzle, tf32.
//   Swizzle<1,4,3> o ((8,2),(4,2)):((8,64),(1,4)), LBO unused, SBO = 64*4 bytes.
VTEST(wgmma_tf32_k_major_32b_swizzle_figure_170) {
  auto A = [](int m, int k) { return val(m, k, 3); };
  auto B = [](int k, int n) { return val(n, k, 4); };
  Wgmma w;
  w.form = "m64n16k8.f32.tf32.tf32";
  w.tail = ", 1, 1";
  w.d_regs = 8;
  for (int n = 0; n < 16; ++n)
    for (int k = 0; k < 8; ++k) {
      const int off = 8 * (n % 8) + 64 * (n / 8) + (k % 4) + 4 * (k / 4);
      put_elem(w.smem_b, swizzle(uint32_t(off) * 4, 1), f32_bits(B(k, n)));
    }
  w.desc_b = desc(0, 16, 64 * 4, 3);   // LBO "assumed 1"
  w.a_words = tf32_a_regs(A);
  check_f32(w.run(), 16, 8, A, B, zero);
}

// Figure 171: MN-major, no swizzle, bf16.
//   Swizzle<0,4,3> o ((8,1,2),(8,2)):((1,8,64),(8,128)), LBO = 128*2, SBO = 64*2.
// As B transposed (imm-trans-b = 1): N = 16, K = 16.
VTEST(wgmma_bf16_mn_major_no_swizzle_figure_171) {
  auto A = [](int m, int k) { return val(m, k, 5); };
  auto B = [](int k, int n) { return val(n, k, 6); };
  Wgmma w;
  w.form = "m64n16k16.f32.bf16.bf16";
  w.tail = ", 1, 1, 1";   // scale-a, scale-b, trans-b
  w.d_regs = 8;
  for (int n = 0; n < 16; ++n)
    for (int k = 0; k < 16; ++k) {
      const int off = (n % 8) + 64 * (n / 8) + 8 * (k % 8) + 128 * (k / 8);
      put_elem(w.smem_b, size_t(off) * 2, bf16_bits(B(k, n)));
    }
  w.desc_b = desc(0, 128 * 2, 64 * 2, 0);
  w.a_words = half_a_regs(A, true);
  check_f32(w.run(), 16, 16, A, B, zero);
}

// Figure 172: MN-major, 32-byte swizzle, bf16.
//   Swizzle<1,4,3> o ((8,2,2),(8,2)):((1,8,128),(16,256)), LBO = 128*2, SBO = 256*2.
VTEST(wgmma_bf16_mn_major_32b_swizzle_figure_172) {
  auto A = [](int m, int k) { return val(m, k, 7); };
  auto B = [](int k, int n) { return val(n, k, 8); };
  Wgmma w;
  w.form = "m64n32k16.f32.bf16.bf16";
  w.tail = ", 1, 1, 1";
  w.d_regs = 16;
  for (int n = 0; n < 32; ++n)
    for (int k = 0; k < 16; ++k) {
      const int off = (n % 8) + 8 * ((n / 8) % 2) + 128 * (n / 16) + 16 * (k % 8) + 256 * (k / 8);
      put_elem(w.smem_b, swizzle(uint32_t(off) * 2, 1), bf16_bits(B(k, n)));
    }
  w.desc_b = desc(0, 128 * 2, 256 * 2, 3);
  w.a_words = half_a_regs(A, true);
  check_f32(w.run(), 32, 16, A, B, zero);
}

// Figure 173: MN-major, 64-byte swizzle, bf16.
//   Swizzle<2,4,3> o ((8,4,2),(8,2)):((1,8,256),(32,512)), LBO = 256*2, SBO = 512*2.
VTEST(wgmma_bf16_mn_major_64b_swizzle_figure_173) {
  auto A = [](int m, int k) { return val(m, k, 9); };
  auto B = [](int k, int n) { return val(n, k, 10); };
  Wgmma w;
  w.form = "m64n64k16.f32.bf16.bf16";
  w.tail = ", 1, 1, 1";
  w.d_regs = 32;
  for (int n = 0; n < 64; ++n)
    for (int k = 0; k < 16; ++k) {
      const int off = (n % 8) + 8 * ((n / 8) % 4) + 256 * (n / 32) + 32 * (k % 8) + 512 * (k / 8);
      put_elem(w.smem_b, swizzle(uint32_t(off) * 2, 2), bf16_bits(B(k, n)));
    }
  w.desc_b = desc(0, 256 * 2, 512 * 2, 2);
  w.a_words = half_a_regs(A, true);
  check_f32(w.run(), 64, 16, A, B, zero);
}

// A from shared memory too, K-major without swizzle -- the figure 169 layout
// grown to 64 rows of f16 -- with an f16 accumulator, a starting D, the
// negate flag on B and scale-d as a predicate.
VTEST(wgmma_f16_both_from_shared_with_accumulator_and_negate) {
  auto A = [](int m, int k) { return val(m, k, 11); };
  auto Bneg = [](int k, int n) { return -val(n, k, 12); };
  auto C = [](int m, int n) { return val(m, n, 13) * 2.0f; };
  Wgmma w;
  w.form = "m64n8k16.f16.f16.f16";
  w.a_regs = false;
  w.scale_d = "%p2";   // true in every thread
  w.tail = ", 1, -1, 0, 0";
  w.d_regs = 2;
  // ((8,8),(8,2)):((8,64),(1,512)) in f16 elements: SBO = 128 B, LBO = 1024 B.
  for (int m = 0; m < 64; ++m)
    for (int k = 0; k < 16; ++k)
      put_elem(w.smem_a, size_t((m % 8) * 8 + (m / 8) * 64 + (k % 8) + (k / 8) * 512) * 2,
               f16_bits(A(m, k)));
  w.desc_a = desc(0, 1024, 128, 0);
  for (int n = 0; n < 8; ++n)
    for (int k = 0; k < 16; ++k)
      put_elem(w.smem_b, size_t(n * 8 + (k % 8) + (k / 8) * 64) * 2, f16_bits(-Bneg(k, n)));
  w.desc_b = desc(0, 128, 1024, 0);
  w.d_in.resize(128 * 2);
  for (int t = 0; t < 128; ++t)
    for (int e = 0; e < 4; ++e) {
      int row, col;
      d_pos(t, e, &row, &col);
      w.d_in[t * 2 + e / 2] |= uint32_t{f16_bits(C(row, col))} << (16 * (e % 2));
    }
  const auto out = w.run();
  for (int t = 0; t < 128; ++t)
    for (int e = 0; e < 4; ++e) {
      int row, col;
      d_pos(t, e, &row, &col);
      float want = C(row, col);
      for (int k = 0; k < 16; ++k) want += A(row, k) * Bneg(k, col);
      const float got = f16_value(static_cast<uint16_t>(out[t * 2 + e / 2] >> (16 * (e % 2))));
      VCHECK_EQ(got, want);
    }
}

// Integers: s8 x u8 with scale-d false (the garbage accumulator is ignored),
// and .satfinite clamping where the plain form would wrap.
VTEST(wgmma_integer_scale_d_false_and_satfinite) {
  // K-major, no swizzle: 8 rows of 16 bytes per core matrix, two across K.
  auto run = [](bool sat, int a_scale) {
    Wgmma w;
    w.form = std::string("m64n8k32.s32.s8.u8") + (sat ? ".satfinite" : "");
    w.tail = "";
    w.scale_d = sat ? "1" : "0";
    w.d_regs = 4;
    for (int n = 0; n < 8; ++n)
      for (int k = 0; k < 32; ++k)
        put_elem(w.smem_b, size_t(n * 16 + (k % 16) + (k / 16) * 128), uint8_t(200 + (n + k) % 50));
    w.desc_b = desc(0, 128, 1024, 0);
    w.a_words.resize(128 * 4);
    for (int t = 0; t < 128; ++t)
      for (int r = 0; r < 4; ++r) {
        uint32_t word = 0;
        for (int e = 0; e < 4; ++e) {
          int row, col;
          a_pos(t, r, e, 4, &row, &col);
          word |= uint32_t(uint8_t(int8_t(a_scale * (1 + (row + col) % 3)))) << (8 * e);
        }
        w.a_words[t * 4 + r] = word;
      }
    w.d_in.assign(128 * 4, sat ? 0x7FFFFF00u : 12345u);
    return w.run();
  };
  // scale-d false: D = A*B exactly, whatever D held.
  const auto plain = run(false, 1);
  for (int t = 0; t < 128; ++t)
    for (int e = 0; e < 4; ++e) {
      int row, col;
      d_pos(t, e, &row, &col);
      int64_t want = 0;
      for (int k = 0; k < 32; ++k) want += int64_t(1 + (row + k) % 3) * (200 + (col + k) % 50);
      VCHECK_EQ(int64_t(int32_t(plain[t * 4 + e])), want);
    }
  // Starting near INT32_MAX and adding a positive product: clamped, not wrapped.
  const auto sat = run(true, 1);
  for (uint32_t v : sat) VCHECK_EQ(int32_t(v), INT32_MAX);
}

// ---- what the instruction demands of its caller ------------------------------

VTEST(wgmma_needs_a_whole_warpgroup) {
  Wgmma w;
  w.form = "m64n8k8.f32.tf32.tf32";
  w.tail = ", 1, 1";
  w.d_regs = 4;
  w.threads = 96;   // warps 0-2 of a warpgroup, and no warp 3
  auto err = VCAPTURE(Error, w.run());
  VCHECK_CONTAINS(err.message(), "whole warpgroup");
}

VTEST(wgmma_refuses_a_nonzero_descriptor_base_offset) {
  Wgmma w;
  w.form = "m64n8k8.f32.tf32.tf32";
  w.tail = ", 1, 1";
  w.d_regs = 4;
  w.desc_b = desc(0, 128, 256, 1) | (uint64_t{3} << 49);
  auto err = VCAPTURE(Error, w.run());
  VCHECK_CONTAINS(err.message(), "nonzero base offset");
}

// ---- brkpt -------------------------------------------------------------------

VTEST(brkpt_is_accepted_and_faults_only_when_reached) {
  const std::string ptx = std::string(kHeader90a) + R"(
.visible .entry k(.param .u32 go)
{
    .reg .pred %p<2>;
    .reg .b32 %r<2>;
    ld.param.u32 %r1, [go];
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra SKIP;
    brkpt;
SKIP:
    ret;
}
)";
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/h100");
  auto m = ptx::parse(ptx);
  auto arg = [](uint32_t v) {
    std::vector<uint8_t> b(4);
    std::memcpy(b.data(), &v, 4);
    return b;
  };
  exec::launch(m.entries[0], LaunchConfig{}, {arg(0)}, mem, prof);   // never reached
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], LaunchConfig{}, {arg(1)}, mem, prof));
  VCHECK(err.code() == Err::Trap);
  VCHECK_CONTAINS(err.message(), "brkpt");
}

// ---- the TMA data path ---------------------------------------------------------

namespace {

// Runs a one-kernel module; returns nothing, the kernel writes `out`.
void run(const std::string& body_ptx, const LaunchConfig& cfg,
         std::vector<std::vector<uint8_t>> args, MemoryManager& mem) {
  DeviceProfile prof = load_gpu("nvidia/h100");
  auto m = ptx::parse(std::string(kHeader90a) + body_ptx);
  exec::launch(m.entries[0], cfg, args, mem, prof);
}

std::vector<uint8_t> tmap_arg(const exec::TensorMap& t) {
  std::vector<uint8_t> b(128);
  t.encode(b.data());
  return b;
}

}  // namespace

// Transactions hold a phase open: arrivals alone do not complete it while
// expect-tx bytes are outstanding, and complete-tx releases it.
VTEST(mbarrier_phase_waits_for_its_transaction_bytes) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(16);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .shared .align 8 .b64 bar;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, bar;
    mbarrier.init.shared::cta.b64 [%r1], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r1], 64;
    mbarrier.test_wait.parity.shared::cta.b64 %p1, [%r1], 0;
    mbarrier.complete_tx.shared::cta.b64 [%r1], 32;
    mbarrier.test_wait.parity.shared::cta.b64 %p2, [%r1], 0;
    mbarrier.complete_tx.shared::cta.b64 [%r1], 32;
    mbarrier.try_wait.parity.shared::cta.b64 %p3, [%r1], 0, 1000;
    selp.u32 %r2, 1, 0, %p1;
    selp.u32 %r3, 1, 0, %p2;
    selp.u32 %r4, 1, 0, %p3;
    st.global.u32 [%rd1], %r2;
    st.global.u32 [%rd1+4], %r3;
    st.global.u32 [%rd1+8], %r4;
    ret;
}
)", LaunchConfig{}, {arg_u64(out)}, mem);
  VCHECK_EQ(mem.load_scalar(out, 4), uint64_t{0});   // arrived, 64 bytes outstanding
  VCHECK_EQ(mem.load_scalar(out + 4, 4), uint64_t{0});   // 32 still outstanding
  VCHECK_EQ(mem.load_scalar(out + 8, 4), uint64_t{1});   // complete
}

// A plain bulk copy in and out: global -> shared on an mbarrier, shared ->
// global in a bulk group. The loaded bytes are not in shared memory until the
// barrier is waited on, as on hardware they need not be.
VTEST(bulk_copy_round_trip_lands_when_the_barrier_is_observed) {
  MemoryManager mem{1 << 20};
  std::vector<uint8_t> src(256);
  for (int i = 0; i < 256; ++i) src[i] = static_cast<uint8_t>(i * 7 + 1);
  const uint64_t in = mem.alloc(256), dst = mem.alloc(256), early = mem.alloc(4);
  mem.write(in, src.data(), 256);
  run(R"(
.visible .entry k(.param .u64 in, .param .u64 dst, .param .u64 early)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .shared .align 128 .b8 buf[256];
    .shared .align 8 .b64 bar;
    ld.param.u64 %rd1, [in];
    ld.param.u64 %rd2, [dst];
    ld.param.u64 %rd3, [early];
    mov.u32 %r1, buf;
    mov.u32 %r2, bar;
    mbarrier.init.shared::cta.b64 [%r2], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r2], 256;
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%rd1], 256, [%r2];
    ld.shared.u32 %r3, [%r1+4];
    st.global.u32 [%rd3], %r3;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r2], 0;
    @!%p1 bra WAIT;
    cp.async.bulk.global.shared::cta.bulk_group [%rd2], [%r1], 256;
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group.read 0;
    ret;
}
)", LaunchConfig{}, {arg_u64(in), arg_u64(dst), arg_u64(early)}, mem);
  VCHECK_EQ(mem.load_scalar(early, 4), uint64_t{0});   // not there before the wait
  std::vector<uint8_t> got(256);
  mem.read(dst, got.data(), 256);
  VCHECK(got == src);
}

// A 2D tensor load with the 128-byte swizzle and a box that runs off two
// edges of the tensor. The expected shared-memory image is built from the
// swizzle table in PTX ISA 5.5.7 -- each row lists which source 16-byte chunk
// lands in each position -- not from the formula the interpreter uses.
VTEST(tensor_load_swizzles_128b_and_zero_fills_past_the_edge) {
  static const int kTable128[8][8] = {
      {0, 1, 2, 3, 4, 5, 6, 7}, {1, 0, 3, 2, 5, 4, 7, 6}, {2, 3, 0, 1, 6, 7, 4, 5},
      {3, 2, 1, 0, 7, 6, 5, 4}, {4, 5, 6, 7, 0, 1, 2, 3}, {5, 4, 7, 6, 1, 0, 3, 2},
      {6, 7, 4, 5, 2, 3, 0, 1}, {7, 6, 5, 4, 3, 2, 1, 0}};
  constexpr int W = 40, H = 20, kStride = 96;   // u16, rows padded to 96 bytes
  constexpr int X0 = 8, Y0 = 16;                // box origin; 64 x 8 box
  MemoryManager mem{1 << 20};
  const uint64_t g = mem.alloc(size_t(H) * kStride), out = mem.alloc(1024);
  auto gval = [](int x, int y) { return static_cast<uint16_t>(1000 + y * 64 + x); };
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) mem.store_scalar(g + uint64_t(y) * kStride + x * 2, 2, gval(x, y));
  exec::TensorMap t;
  t.address = g;
  t.rank = 2;
  t.type = exec::TmapType::U16;
  t.swizzle = exec::TmapSwizzle::B128;
  t.dim = {W, H, 1, 1, 1};
  t.stride = {2, kStride, 0, 0, 0};
  t.box = {64, 8, 1, 1, 1};
  t.elem_stride = {1, 1, 1, 1, 1};
  run(R"(
.visible .entry k(.param .align 64 .b8 tmap[128], .param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .shared .align 1024 .b8 tile[1024];
    .shared .align 8 .b64 bar;
    ld.param.u64 %rd1, [out];
    mov.b64 %rd2, tmap;
    cvta.param.u64 %rd3, %rd2;
    mov.u32 %r1, tile;
    mov.u32 %r2, bar;
    mov.u32 %r3, %tid.x;
    setp.ne.u32 %p1, %r3, 0;
    @%p1 bra ISSUED;
    mbarrier.init.shared::cta.b64 [%r2], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r2], 1024;
    mov.u32 %r4, 8;
    mov.u32 %r5, 16;
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%r1], [%rd3, {%r4, %r5}], [%r2], %rd2;
ISSUED:
    bar.sync 0;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r2], 0;
    @!%p1 bra WAIT;
    shl.b32 %r6, %r3, 3;
    add.u32 %r7, %r1, %r6;
    ld.shared.u64 %rd4, [%r7];
    cvt.u64.u32 %rd5, %r6;
    add.u64 %rd6, %rd1, %rd5;
    st.global.u64 [%rd6], %rd4;
    ret;
}
)", [] { LaunchConfig c; c.block = {128, 1, 1}; return c; }(), {tmap_arg(t), arg_u64(out)}, mem);
  for (int r = 0; r < 8; ++r)
    for (int c = 0; c < 64; ++c) {
      const int dense = (r * 64 + c) * 2;
      const int line = dense / 128, chunk = (dense % 128) / 16, within = dense % 16;
      int pos = 0;
      while (kTable128[line % 8][pos] != chunk) ++pos;
      const uint64_t at = uint64_t(line) * 128 + pos * 16 + within;
      const int x = X0 + c, y = Y0 + r;
      const uint16_t want = (x < W && y < H) ? gval(x, y) : 0;
      VCHECK_EQ(mem.load_scalar(out + at, 2), uint64_t{want});
    }
}

// A tensor store writes the part of the box inside the tensor and nothing
// past its edge.
VTEST(tensor_store_writes_only_inside_the_tensor) {
  constexpr int W = 24, H = 6;   // u32; the 16 x 4 box at (16, 4) half hangs off
  MemoryManager mem{1 << 20};
  const uint64_t g = mem.alloc(size_t(W) * H * 4 + 256);
  for (int i = 0; i < W * H + 64; ++i) mem.store_scalar(g + uint64_t(i) * 4, 4, 0xEEEEEEEEu);
  exec::TensorMap t;
  t.address = g;
  t.rank = 2;
  t.type = exec::TmapType::U32;
  t.dim = {W, H, 1, 1, 1};
  t.stride = {4, W * 4, 0, 0, 0};
  t.box = {16, 4, 1, 1, 1};
  t.elem_stride = {1, 1, 1, 1, 1};
  run(R"(
.visible .entry k(.param .align 64 .b8 tmap[128])
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .shared .align 128 .b8 tile[256];
    mov.b64 %rd1, tmap;
    cvta.param.u64 %rd2, %rd1;
    mov.u32 %r1, tile;
    mov.u32 %r2, %tid.x;
    shl.b32 %r3, %r2, 2;
    add.u32 %r4, %r1, %r3;
    add.u32 %r5, %r2, 100;
    st.shared.u32 [%r4], %r5;
    bar.sync 0;
    setp.ne.u32 %p1, %r2, 0;
    @%p1 bra DONE;
    mov.u32 %r6, 16;
    mov.u32 %r7, 4;
    fence.proxy.async.shared::cta;
    cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd2, {%r6, %r7}], [%r1];
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group 0;
DONE:
    ret;
}
)", [] { LaunchConfig c; c.block = {64, 1, 1}; return c; }(),
      {tmap_arg(t)}, mem);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const bool in_box = x >= 16 && y >= 4;
      const uint64_t want = in_box ? uint64_t((y - 4) * 16 + (x - 16) + 100) : 0xEEEEEEEEu;
      VCHECK_EQ(mem.load_scalar(g + (uint64_t(y) * W + x) * 4, 4), want);
    }
  // Past the tensor's end, untouched.
  VCHECK_EQ(mem.load_scalar(g + uint64_t(W) * H * 4, 4), uint64_t{0xEEEEEEEEu});
}

// cp.reduce.async.bulk.tensor, as CUTLASS's SM90_TMA_REDUCE_ADD emits it: two
// blocks add their f32 tiles into the same box, and the elements past the
// tensor's edge are left alone. The adds are exact, so the order the blocks
// run in cannot change the answer, and none of either block's is lost.
VTEST(tensor_reduce_adds_into_the_box_and_only_inside_the_tensor) {
  constexpr int W = 24, H = 6;
  MemoryManager mem{1 << 20};
  const uint64_t g = mem.alloc(size_t(W) * H * 4 + 256);
  for (int i = 0; i < W * H + 64; ++i) mem.store_scalar(g + uint64_t(i) * 4, 4, f32_bits(0.5f));
  exec::TensorMap t;
  t.address = g;
  t.rank = 2;
  t.type = exec::TmapType::F32;
  t.dim = {W, H, 1, 1, 1};
  t.stride = {4, W * 4, 0, 0, 0};
  t.box = {16, 4, 1, 1, 1};
  t.elem_stride = {1, 1, 1, 1, 1};
  run(R"(
.visible .entry k(.param .align 64 .b8 tmap[128])
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<4>;
    .shared .align 128 .b8 tile[256];
    mov.b64 %rd1, tmap;
    cvta.param.u64 %rd2, %rd1;
    mov.u32 %r1, tile;
    mov.u32 %r2, %tid.x;
    shl.b32 %r3, %r2, 2;
    add.u32 %r4, %r1, %r3;
    cvt.rn.f32.u32 %f1, %r2;
    add.f32 %f1, %f1, 0f3E800000;
    st.shared.f32 [%r4], %f1;
    bar.sync 0;
    setp.ne.u32 %p1, %r2, 0;
    @%p1 bra DONE;
    mov.u32 %r6, 16;
    mov.u32 %r7, 4;
    fence.proxy.async.shared::cta;
    cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%rd2, {%r6, %r7}], [%r1];
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group.read 0;
DONE:
    ret;
}
)", [] { LaunchConfig c; c.grid = {2, 1, 1}; c.block = {64, 1, 1}; return c; }(),
      {tmap_arg(t)}, mem);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const bool in_box = x >= 16 && y >= 4;
      const float want = in_box ? 0.5f + 2.0f * (float((y - 4) * 16 + (x - 16)) + 0.25f) : 0.5f;
      VCHECK_EQ(mem.load_scalar(g + (uint64_t(y) * W + x) * 4, 4), uint64_t{f32_bits(want)});
    }
  VCHECK_EQ(mem.load_scalar(g + uint64_t(W) * H * 4, 4), uint64_t{f32_bits(0.5f)});
}

// The plain form into global memory, one operation and type at a time, over
// the values that separate a right reduction from a plausible one: an
// unsigned add that wraps, a signed min against INT_MIN, a half-precision
// add that keeps subnormals (.noftz) and rounds ties to even, inc's wrap to
// zero, and a 64-bit xor.
VTEST(bulk_reduce_into_global_memory_per_operation) {
  MemoryManager mem{1 << 20};
  const uint64_t g = mem.alloc(5 * 16);
  const uint32_t add_g[4] = {1, 2, 3, 0xFFFFFFFFu}, add_s[4] = {10, 20, 30, 1};
  const uint32_t min_g[4] = {5, uint32_t(-5), 7, 0}, min_s[4] = {uint32_t(-1), 3, 7, 0x80000000u};
  const uint16_t h_g[8] = {0x0001, 0x3C00, 0x3C00, 0x3C00, 0x3C01, 0, 0x8001, 0x7C00};
  const uint16_t h_s[8] = {0x0001, 0x3800, 0x1400, 0x1000, 0x1000, 0, 0x8001, 0x3C00};
  const uint32_t inc_g[4] = {0, 5, 9, 3}, inc_s[4] = {10, 5, 9, 2};
  const uint64_t x_g[2] = {0xF0F0F0F0F0F0F0F0ull, 1}, x_s[2] = {0xFFFFFFFF00000000ull, 3};
  std::vector<uint8_t> src(5 * 16);
  std::memcpy(src.data(), add_s, 16);
  std::memcpy(src.data() + 16, min_s, 16);
  std::memcpy(src.data() + 32, h_s, 16);
  std::memcpy(src.data() + 48, inc_s, 16);
  std::memcpy(src.data() + 64, x_s, 16);
  mem.write(g, add_g, 16);
  mem.write(g + 16, min_g, 16);
  mem.write(g + 32, h_g, 16);
  mem.write(g + 48, inc_g, 16);
  mem.write(g + 64, x_g, 16);
  const uint64_t s = mem.alloc(5 * 16);
  mem.write(s, src.data(), src.size());
  run(R"(
.visible .entry k(.param .u64 g, .param .u64 s)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .shared .align 16 .b8 buf[80];
    ld.param.u64 %rd1, [g];
    ld.param.u64 %rd2, [s];
    mov.u32 %r1, buf;
    mov.u32 %r2, %tid.x;
    shl.b32 %r3, %r2, 2;
    cvt.u64.u32 %rd3, %r3;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.u32 %r4, [%rd4];
    add.u32 %r5, %r1, %r3;
    st.shared.u32 [%r5], %r4;
    bar.sync 0;
    setp.ne.u32 %p1, %r2, 0;
    @%p1 bra DONE;
    fence.proxy.async.shared::cta;
    cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%rd1], [%r1], 16;
    cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s32 [%rd1+16], [%r1+16], 16;
    cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [%rd1+32], [%r1+32], 16;
    cp.reduce.async.bulk.global.shared::cta.bulk_group.inc.u32 [%rd1+48], [%r1+48], 16;
    cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.xor.b64 [%rd1+64], [%r1+64], 16, %rd2;
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group 0;
DONE:
    ret;
}
)", [] { LaunchConfig c; c.block = {20, 1, 1}; return c; }(), {arg_u64(g), arg_u64(s)}, mem);
  const uint32_t add_w[4] = {11, 22, 33, 0};
  const uint32_t min_w[4] = {uint32_t(-1), uint32_t(-5), 7, 0x80000000u};
  // 2^-24 + 2^-24; 1 + 0.5; 1 + 2^-10 exactly; 1 + 2^-11 ties to even (1);
  // (1 + 2^-10) + 2^-11 ties to even (1 + 2^-9); 0 + 0; two negative
  // subnormals; infinity + 1.
  const uint16_t h_w[8] = {0x0002, 0x3E00, 0x3C01, 0x3C00, 0x3C02, 0, 0x8002, 0x7C00};
  const uint32_t inc_w[4] = {1, 0, 0, 0};
  const uint64_t x_w[2] = {0x0F0F0F0FF0F0F0F0ull, 2};
  for (int i = 0; i < 4; ++i) {
    VCHECK_EQ(mem.load_scalar(g + i * 4, 4), uint64_t{add_w[i]});
    VCHECK_EQ(mem.load_scalar(g + 16 + i * 4, 4), uint64_t{min_w[i]});
    VCHECK_EQ(mem.load_scalar(g + 48 + i * 4, 4), uint64_t{inc_w[i]});
  }
  for (int i = 0; i < 8; ++i) VCHECK_EQ(mem.load_scalar(g + 32 + i * 2, 2), uint64_t{h_w[i]});
  for (int i = 0; i < 2; ++i) VCHECK_EQ(mem.load_scalar(g + 64 + i * 8, 8), x_w[i]);
}

VTEST(bulk_reduce_forms_the_isa_does_not_define_are_refused) {
  auto refusal = [](const std::string& line) -> std::string {
    try {
      ptx::parse(std::string(kHeader90a) + ".visible .entry k()\n{\n    .reg .b32 %r<4>;\n"
                 "    .reg .b64 %rd<4>;\n    " + line + "\n    ret;\n}\n");
    } catch (const Error& e) {
      return e.message();
    }
    return "";
  };
  VCHECK_CONTAINS(refusal("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f16 [%rd1], [%r1], 16;"),
                  "requires .noftz");
  VCHECK_CONTAINS(refusal("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.s64 [%rd1], [%r1], 16;"),
                  "no .s64 form");
  VCHECK_CONTAINS(refusal("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                          ".min.u64 [%r1], [%r2], 16, [%r3];"),
                  "no .u64 form");
}

// tensormap.replace the way CUTLASS's grouped GEMMs use it: copy the map
// into shared memory, point it at another tensor with other extents and a
// wider row, fence it back out to global memory, and load through it. The
// new dimension cuts the box, so the load's zero fill shows the dimension
// took; the new stride shows the row pitch did. global_stride changed unit
// at PTX ISA 8.5 (see the parser), so the same kernel runs under both.
namespace {
void retarget_and_load(const char* version, uint64_t stride_operand) {
  constexpr int W1 = 16, W2 = 32, H = 4;
  MemoryManager mem{1 << 20};
  const uint64_t g1 = mem.alloc(W1 * H * 4), g2 = mem.alloc(W2 * H * 4);
  const uint64_t gmap = mem.alloc(128 + 128), out = mem.alloc(16 * H * 4);
  const uint64_t gmap_al = (gmap + 127) / 128 * 128;
  for (int i = 0; i < W1 * H; ++i) mem.store_scalar(g1 + i * 4, 4, 0xDEAD0000u + i);
  for (int i = 0; i < W2 * H; ++i) mem.store_scalar(g2 + i * 4, 4, 0x20000u + i);
  exec::TensorMap t;
  t.address = g1;
  t.rank = 2;
  t.type = exec::TmapType::U32;
  t.dim = {W1, H, 1, 1, 1};
  t.stride = {4, W1 * 4, 0, 0, 0};
  t.box = {16, H, 1, 1, 1};
  t.elem_stride = {1, 1, 1, 1, 1};
  const std::string ptx = std::string(".version ") + version + R"(
.target sm_90a
.address_size 64
.visible .entry k(.param .align 64 .b8 tmap[128], .param .u64 g2, .param .u64 gmap, .param .u64 stride, .param .u64 out)
{
    .reg .pred %p<4>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<16>;
    .shared .align 128 .b8 smap[128];
    .shared .align 128 .b8 tile[256];
    .shared .align 8 .b64 bar;
    mov.u32 %r1, %tid.x;
    setp.ne.u32 %p1, %r1, 0;
    @%p1 bra WAIT;
    mov.b64 %rd1, tmap;
    cvta.param.u64 %rd2, %rd1;
    mov.u32 %r2, smap;
    mov.u32 %r3, 0;
COPY:
    cvt.u64.u32 %rd3, %r3;
    add.u64 %rd4, %rd2, %rd3;
    ld.u64 %rd5, [%rd4];
    add.u32 %r4, %r2, %r3;
    st.shared.u64 [%r4], %rd5;
    add.u32 %r3, %r3, 8;
    setp.lt.u32 %p2, %r3, 128;
    @%p2 bra COPY;
    ld.param.u64 %rd6, [g2];
    ld.param.u64 %rd7, [gmap];
    ld.param.u64 %rd8, [stride];
    tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%r2], %rd6;
    tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%r2], 0, 8;
    tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%r2], 0, %rd8;
    tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%rd7], [%r2], 128;
    fence.proxy.tensormap::generic.acquire.gpu [%rd7], 128;
    mov.u32 %r5, bar;
    mbarrier.init.shared::cta.b64 [%r5], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r5], 256;
    mov.u32 %r6, tile;
    mov.u32 %r7, 0;
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r6], [%rd7, {%r7, %r7}], [%r5];
WAIT:
    bar.sync 0;
    @%p1 bra DONE;
    mov.u32 %r5, bar;
SPIN:
    mbarrier.try_wait.parity.shared::cta.b64 %p3, [%r5], 0;
    @!%p3 bra SPIN;
    ld.param.u64 %rd9, [out];
    mov.u32 %r6, tile;
    mov.u32 %r3, 0;
OUT:
    add.u32 %r4, %r6, %r3;
    ld.shared.u32 %r8, [%r4];
    cvt.u64.u32 %rd10, %r3;
    add.u64 %rd11, %rd9, %rd10;
    st.global.u32 [%rd11], %r8;
    add.u32 %r3, %r3, 4;
    setp.lt.u32 %p2, %r3, 256;
    @%p2 bra OUT;
DONE:
    ret;
}
)";
  DeviceProfile prof = load_gpu("nvidia/h100");
  auto m = ptx::parse(ptx);
  LaunchConfig c;
  c.block = {32, 1, 1};
  exec::launch(m.entries[0], c, {tmap_arg(t), arg_u64(g2), arg_u64(gmap_al), arg_u64(stride_operand), arg_u64(out)},
               mem, prof);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < 16; ++x) {
      const uint64_t want = x < 8 ? 0x20000u + uint64_t(y * W2 + x) : 0;
      VCHECK_EQ(mem.load_scalar(out + (y * 16 + x) * 4, 4), want);
    }
}
}  // namespace

VTEST(tensormap_replace_retargets_a_map_that_a_load_then_uses) {
  retarget_and_load("8.5", 32 * 4);        // bytes, from PTX ISA 8.5
  retarget_and_load("8.3", (32 * 4) >> 4);  // 16-byte units before it
}

// Every field, written into a map in global memory and read back by the
// host's decoder: the value lands in the field it names, element types take
// the ISA's numbering (not the driver's), and the rank is zero-based.
VTEST(tensormap_replace_writes_each_field) {
  MemoryManager mem{1 << 20};
  const uint64_t raw = mem.alloc(256), gmap = (raw + 127) / 128 * 128;
  exec::TensorMap t;
  t.address = 0x1000;
  t.rank = 2;
  t.type = exec::TmapType::U32;
  t.dim = {16, 4, 1, 1, 1};
  t.stride = {4, 64, 0, 0, 0};
  t.box = {16, 4, 1, 1, 1};
  t.elem_stride = {1, 1, 1, 1, 1};
  std::vector<uint8_t> bytes(128);
  t.encode(bytes.data());
  mem.write(gmap, bytes.data(), 128);
  run(R"(
.visible .entry k(.param .u64 m)
{
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [m];
    mov.u64 %rd2, 0x7000;
    tensormap.replace.tile.global_address.global.b1024.b64 [%rd1], %rd2;
    tensormap.replace.tile.rank.global.b1024.b32 [%rd1], 2;
    tensormap.replace.tile.box_dim.global.b1024.b32 [%rd1], 2, 5;
    tensormap.replace.tile.global_dim.global.b1024.b32 [%rd1], 2, 9;
    mov.u64 %rd3, 256;   // 4096 bytes: this header is PTX 8.3, 16-byte units
    tensormap.replace.tile.global_stride.global.b1024.b64 [%rd1], 1, %rd3;
    tensormap.replace.tile.element_stride.global.b1024.b32 [%rd1], 1, 2;
    tensormap.replace.tile.elemtype.global.b1024.b32 [%rd1], 6;
    tensormap.replace.tile.swizzle_mode.global.b1024.b32 [%rd1], 3;
    tensormap.replace.tile.fill_mode.global.b1024.b32 [%rd1], 1;
    ret;
}
)", LaunchConfig{}, {arg_u64(gmap)}, mem);
  mem.read(gmap, bytes.data(), 128);
  exec::TensorMap r;
  VCHECK(r.decode(bytes.data()));
  VCHECK_EQ(r.address, uint64_t{0x7000});
  VCHECK_EQ(r.rank, 3u);
  VCHECK_EQ(r.box[2], 5u);
  VCHECK_EQ(r.dim[2], uint64_t{9});
  VCHECK_EQ(r.stride[2], uint64_t{4096});
  VCHECK_EQ(r.elem_stride[1], 2u);
  VCHECK(r.type == exec::TmapType::F16);   // ISA value 6; the driver's 6 is also F16,
  VCHECK_EQ(r.stride[0], uint64_t{2});     // so the element size is the check that matters
  VCHECK(r.swizzle == exec::TmapSwizzle::B128);
  VCHECK_EQ(unsigned(r.oob_nan), 1u);
  // Untouched fields stay as they were.
  VCHECK_EQ(r.box[0], 16u);
  VCHECK_EQ(r.dim[1], uint64_t{4});
  VCHECK_EQ(r.stride[1], uint64_t{64});
}

// Where the two numberings differ: the ISA's 9 is f64, the driver's 9 is bf16.
VTEST(tensormap_replace_elemtype_uses_the_isas_numbering) {
  MemoryManager mem{1 << 20};
  const uint64_t raw = mem.alloc(256), gmap = (raw + 127) / 128 * 128;
  exec::TensorMap t;
  t.address = 0x1000;
  t.rank = 1;
  t.type = exec::TmapType::U32;
  t.dim = {16, 1, 1, 1, 1};
  t.box = {16, 1, 1, 1, 1};
  t.elem_stride = {1, 1, 1, 1, 1};
  std::vector<uint8_t> bytes(128);
  t.encode(bytes.data());
  mem.write(gmap, bytes.data(), 128);
  run(R"(
.visible .entry k(.param .u64 m)
{
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [m];
    tensormap.replace.tile.elemtype.global.b1024.b32 [%rd1], 9;
    ret;
}
)", LaunchConfig{}, {arg_u64(gmap)}, mem);
  mem.read(gmap, bytes.data(), 128);
  exec::TensorMap r;
  VCHECK(r.decode(bytes.data()));
  VCHECK(r.type == exec::TmapType::F64);
}

VTEST(tensormap_replace_refuses_what_no_map_could_hold) {
  MemoryManager mem{1 << 20};
  const uint64_t raw = mem.alloc(256), gmap = (raw + 127) / 128 * 128;
  exec::TensorMap t;
  t.address = 0x1000;
  t.rank = 1;
  t.type = exec::TmapType::U32;
  t.dim = {16, 1, 1, 1, 1};
  t.box = {16, 1, 1, 1, 1};
  t.elem_stride = {1, 1, 1, 1, 1};
  std::vector<uint8_t> bytes(128);
  t.encode(bytes.data());
  mem.write(gmap, bytes.data(), 128);
  auto one = [&](const std::string& line) {
    return VCAPTURE(Error, run(".visible .entry k(.param .u64 m)\n{\n    .reg .b64 %rd<2>;\n"
                               "    ld.param.u64 %rd1, [m];\n    " + line + "\n    ret;\n}\n",
                               LaunchConfig{}, {arg_u64(gmap)}, mem));
  };
  VCHECK_CONTAINS(one("tensormap.replace.tile.element_stride.global.b1024.b32 [%rd1], 0, 0;").message(),
                  "element stride is 1 to 8");
  VCHECK_CONTAINS(one("tensormap.replace.tile.box_dim.global.b1024.b32 [%rd1], 0, 300;").message(),
                  "box dimension is 1 to 256");
  VCHECK_CONTAINS(one("tensormap.replace.tile.swizzle_mode.global.b1024.b32 [%rd1], 4;").message(),
                  "96-byte swizzle");
  auto parse_err = [](const std::string& target, const std::string& line) {
    return VCAPTURE(Error, ptx::parse(".version 8.3\n.target " + target + "\n.address_size 64\n"
                                      ".visible .entry k()\n{\n    .reg .b32 %r<2>;\n    .reg .b64 %rd<2>;\n    " +
                                      line + "\n    ret;\n}\n")).message();
  };
  VCHECK_CONTAINS(parse_err("sm_90", "tensormap.replace.tile.rank.global.b1024.b32 [%rd1], 1;"),
                  "arch-specific target");
  VCHECK_CONTAINS(parse_err("sm_90a", "tensormap.replace.tile.global_address.global.b1024.b32 [%rd1], %r1;"),
                  ".b64 for global_address");
  VCHECK_CONTAINS(parse_err("sm_90a", "tensormap.replace.tile.swizzle_mode.global.b1024.b32 [%rd1], %r1;"),
                  "value is an immediate");
}

// im2col maps: the corners survive encoding at the width each rank keeps
// (16, 8 and 5 bits, negative values included), and the encoder refuses what
// cuda.h says it must.
VTEST(im2col_map_encoding_and_its_limits) {
  alignas(64) uint8_t buf[128];
  const unsigned long long dims[5] = {16, 9, 7, 5, 2}, strides[4] = {64, 576, 4032, 20160};
  const unsigned es[5] = {1, 2, 1, 1, 1};
  std::string why;
  for (unsigned rank = 3; rank <= 5; ++rank) {
    const int lim = rank == 3 ? 32768 : rank == 4 ? 128 : 16;
    const int lower[3] = {-lim, -1, -lim}, upper[3] = {lim - 1, -2, 0};
    VCHECK(exec::encode_im2col(buf, 7, rank, reinterpret_cast<void*>(0x10000), dims, strides, lower, upper,
                               16, 32, es, 0, 0, 0, 0, &why) == exec::TmapResult::Ok);
    exec::TensorMap m;
    VCHECK(m.decode(buf));
    VCHECK(m.im2col);
    VCHECK_EQ(m.channels, 16u);
    VCHECK_EQ(m.pixels, 32u);
    for (unsigned i = 0; i + 2 < rank; ++i) {
      VCHECK_EQ(m.lower[i], lower[i]);
      VCHECK_EQ(m.upper[i], upper[i]);
    }
    VCHECK_EQ(m.elem_stride[1], 2u);
  }
  const int lo[3] = {0, 0, 0}, far[3] = {200, 0, 0}, empty[3] = {-20, 0, 0};
  auto refused = [&](unsigned rank, const int* lower, const int* upper, unsigned c, unsigned px) {
    return exec::encode_im2col(buf, 7, rank, reinterpret_cast<void*>(0x10000), dims, strides, lower, upper, c,
                               px, es, 0, 0, 0, 0, &why) != exec::TmapResult::Ok;
  };
  VCHECK(refused(2, lo, lo, 16, 32));
  VCHECK_CONTAINS(why, "3, 4 or 5");
  VCHECK(refused(4, far, lo, 16, 32));
  VCHECK_CONTAINS(why, "[-128, 127]");
  VCHECK(refused(3, lo, empty, 16, 32));   // W = 9, upper -20: no pixel left
  VCHECK_CONTAINS(why, "non-zero area");
  VCHECK(refused(3, lo, lo, 257, 32));
  VCHECK_CONTAINS(why, "channelsPerPixel");
  VCHECK(refused(3, lo, lo, 16, 1025));
  VCHECK_CONTAINS(why, "pixelsPerColumn");
}

namespace {
// A 3D (NWC) f32 activation, W = 5, N = 2, C = 4, element (n, w, c) = 100n + 10w + c + 1.
exec::TensorMap nwc_map(uint64_t addr) {
  exec::TensorMap t;
  t.address = addr;
  t.rank = 3;
  t.type = exec::TmapType::F32;
  t.dim = {4, 5, 2, 1, 1};
  t.stride = {4, 16, 80, 0, 0};
  t.elem_stride = {1, 1, 1, 1, 1};
  t.im2col = true;
  t.channels = 4;
  t.pixels = 8;
  t.lower = {-1, 0, 0};    // padding 1
  t.upper = {-1, 0, 0};    // 3-tap filter, padding 1: bases -1 .. 3
  return t;
}
}  // namespace

// An im2col load walks its pixels through the box -- W from lower to W + upper
// - 1, then the next image -- reading each at base + offset, zero outside the
// image, and zero once the batch has run out.
VTEST(im2col_load_walks_the_box_and_the_batch) {
  MemoryManager mem{1 << 20};
  const uint64_t g = mem.alloc(2 * 5 * 4 * 4), out = mem.alloc(8 * 4 * 4);
  for (int n = 0; n < 2; ++n)
    for (int w = 0; w < 5; ++w)
      for (int c = 0; c < 4; ++c)
        mem.store_scalar(g + ((n * 5 + w) * 4 + c) * 4, 4, f32_bits(float(100 * n + 10 * w + c + 1)));
  run(R"(
.visible .entry k(.param .align 64 .b8 tmap[128], .param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    .shared .align 128 .b8 tile[128];
    .shared .align 8 .b64 bar;
    mov.b64 %rd1, tmap;
    cvta.param.u64 %rd2, %rd1;
    mov.u32 %r1, tile;
    mov.u32 %r2, bar;
    mbarrier.init.shared::cta.b64 [%r2], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r2], 128;
    mov.u32 %r3, 0;     // c
    mov.u32 %r4, 1;     // w base: output 2 of a padding-1 conv
    mov.u32 %r5, 1;     // n
    mov.b16 %rs1, 2;    // filter tap 2
    cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%r1], [%rd2, {%r3, %r4, %r5}], [%r2], {%rs1};
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r2], 0;
    @!%p1 bra WAIT;
    ld.param.u64 %rd3, [out];
    mov.u32 %r6, 0;
COPY:
    add.u32 %r7, %r1, %r6;
    ld.shared.u32 %r8, [%r7];
    cvt.u64.u32 %rd4, %r6;
    add.u64 %rd5, %rd3, %rd4;
    st.global.u32 [%rd5], %r8;
    add.u32 %r6, %r6, 4;
    setp.lt.u32 %p1, %r6, 128;
    @%p1 bra COPY;
    ret;
}
)", LaunchConfig{}, {tmap_arg(nwc_map(g)), arg_u64(out)}, mem);
  // Bases walk 1, 2, 3 in image 1, then image 2 does not exist. Reading at
  // base + 2: w = 3, 4, then 5 (past the edge), then nothing.
  const int want_w[8] = {3, 4, -1, -1, -1, -1, -1, -1};
  for (int px = 0; px < 8; ++px)
    for (int c = 0; c < 4; ++c) {
      const float want = want_w[px] < 0 ? 0.0f : float(100 + 10 * want_w[px] + c + 1);
      VCHECK_EQ(mem.load_scalar(out + (px * 4 + c) * 4, 4), uint64_t{f32_bits(want)});
    }
}

// The store side, .im2col_no_offs: pixels written back along the same walk,
// with nothing written outside the tensor.
VTEST(im2col_store_writes_the_pixels_back) {
  MemoryManager mem{1 << 20};
  const uint64_t g = mem.alloc(2 * 5 * 4 * 4);
  for (int i = 0; i < 40; ++i) mem.store_scalar(g + i * 4, 4, 0xEEEEEEEEu);
  run(R"(
.visible .entry k(.param .align 64 .b8 tmap[128])
{
    .reg .pred %p<2>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<4>;
    .shared .align 128 .b8 tile[128];
    mov.b64 %rd1, tmap;
    cvta.param.u64 %rd2, %rd1;
    mov.u32 %r1, tile;
    mov.u32 %r2, %tid.x;
    shl.b32 %r3, %r2, 2;
    add.u32 %r4, %r1, %r3;
    st.shared.u32 [%r4], %r2;
    bar.sync 0;
    setp.ne.u32 %p1, %r2, 0;
    @%p1 bra DONE;
    mov.u32 %r5, 0;
    mov.u32 %r6, -1;    // w base -1: the first pixel is in the padding
    fence.proxy.async.shared::cta;
    cp.async.bulk.tensor.3d.global.shared::cta.im2col_no_offs.bulk_group [%rd2, {%r5, %r6, %r5}], [%r1];
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group 0;
DONE:
    ret;
}
)", [] { LaunchConfig c; c.block = {32, 1, 1}; return c; }(), {tmap_arg(nwc_map(g))}, mem);
  // Pixels -1, 0, 1, 2, 3 of image 0 (the box ends at W + upper - 1 = 3, so
  // column 4 is never a base), then -1, 0, 1 of image 1; -1 is in the
  // padding and not written. Pixel p's channels hold 4p .. 4p + 3.
  for (int n = 0; n < 2; ++n)
    for (int w = 0; w < 5; ++w)
      for (int c = 0; c < 4; ++c) {
        const int px = n == 0 ? (w <= 3 ? w + 1 : -1) : (w <= 1 ? 6 + w : -1);
        const uint64_t want = px < 0 ? 0xEEEEEEEEu : uint64_t(px * 4 + c);
        VCHECK_EQ(mem.load_scalar(g + ((n * 5 + w) * 4 + c) * 4, 4), want);
      }
}

VTEST(tensor_copies_refuse_a_map_of_the_other_mode) {
  MemoryManager mem{1 << 20};
  const uint64_t g = mem.alloc(256);
  const std::string tile_load = R"(
.visible .entry k(.param .align 64 .b8 tmap[128])
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;
    .shared .align 128 .b8 tile[128];
    .shared .align 8 .b64 bar;
    mov.b64 %rd1, tmap;
    cvta.param.u64 %rd2, %rd1;
    mov.u32 %r1, tile;
    mov.u32 %r2, bar;
    mov.u32 %r3, 0;
    mbarrier.init.shared::cta.b64 [%r2], 1;
    cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%rd2, {%r3, %r3, %r3}], [%r2];
    ret;
}
)";
  auto err = VCAPTURE(Error, run(tile_load, LaunchConfig{}, {tmap_arg(nwc_map(g))}, mem));
  VCHECK_CONTAINS(err.message(), "made by cuTensorMapEncodeIm2col");
}

VTEST(tensor_copy_refuses_a_map_nothing_encoded) {
  MemoryManager mem{1 << 20};
  std::vector<uint8_t> junk(128, 0x5A);
  auto err = VCAPTURE(Error, run(R"(
.visible .entry k(.param .align 64 .b8 tmap[128])
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;
    .shared .align 128 .b8 tile[256];
    .shared .align 8 .b64 bar;
    mov.b64 %rd1, tmap;
    cvta.param.u64 %rd2, %rd1;
    mov.u32 %r1, tile;
    mov.u32 %r2, bar;
    mov.u32 %r3, 0;
    mbarrier.init.shared::cta.b64 [%r2], 1;
    cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%rd2, {%r3}], [%r2];
    ret;
}
)", LaunchConfig{}, {junk}, mem));
  VCHECK_CONTAINS(err.message(), "not made by cuTensorMapEncodeTiled");
}

// barrier.cluster across the two blocks of a cluster: rank 1 publishes, rank 0
// reads after the wait. Both blocks have four warps, which needs each block
// to be scheduled on its own -- one scheduler shared by both handed block 0
// only warps 0 and 2 -- and some of rank 1's threads exit early, which must
// not hold the barrier.
VTEST(barrier_cluster_orders_the_blocks_of_a_cluster) {
  MemoryManager mem{1 << 20};
  const uint64_t flag = mem.alloc(4), out = mem.alloc(128 * 4);
  mem.store_scalar(flag, 4, 0);
  run(R"(
.visible .entry k(.param .u64 flag, .param .u64 out)
{
    .reg .pred %p<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [flag];
    ld.param.u64 %rd2, [out];
    mov.u32 %r1, %cluster_ctarank;
    mov.u32 %r2, %tid.x;
    setp.eq.u32 %p1, %r1, 1;
    setp.ge.u32 %p2, %r2, 120;
    and.pred %p3, %p1, %p2;
    @%p3 bra GONE;
    setp.ne.u32 %p2, %r2, 0;
    not.pred %p3, %p1;
    or.pred %p2, %p2, %p3;
    @%p2 bra ARRIVE;
    mov.u32 %r3, 77;
    st.global.u32 [%rd1], %r3;
ARRIVE:
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    @%p1 bra GONE;
    ld.global.u32 %r4, [%rd1];
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r4;
GONE:
    ret;
}
)", [] {
    LaunchConfig c;
    c.grid = {2, 1, 1};
    c.block = {128, 1, 1};
    c.cluster = {2, 1, 1};
    return c;
  }(), {arg_u64(flag), arg_u64(out)}, mem);
  for (int i = 0; i < 128; ++i) VCHECK_EQ(mem.load_scalar(out + i * 4, 4), uint64_t{77});
}

// Labels belong to the { } block that defines them: inline asm repeats the
// same names in every block it expands to.
VTEST(labels_are_scoped_to_their_block) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(8);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, 0;
    {
    .reg .pred P1;
    LOOP:
    add.u32 %r1, %r1, 1;
    setp.lt.u32 P1, %r1, 3;
    @P1 bra LOOP;
    }
    {
    .reg .pred P1;
    LOOP:
    add.u32 %r1, %r1, 10;
    setp.lt.u32 P1, %r1, 33;
    @P1 bra LOOP;
    bra DONE;
    DONE:
    }
    st.global.u32 [%rd1], %r1;
    ret;
}
)", LaunchConfig{}, {arg_u64(out)}, mem);
  VCHECK_EQ(mem.load_scalar(out, 4), uint64_t{33});
}

// A lane can reach bar.sync from a higher pc: nvcc puts a rarely taken block
// after the kernel's ret and branches back from it. The barrier waits for it.
VTEST(bar_sync_waits_for_lanes_in_code_placed_after_ret) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(64 * 4);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<4>;
    .shared .align 4 .b32 x;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %tid.x;
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra OUTLINE;
BACK:
    bar.sync 0;
    ld.shared.u32 %r2, [x];
    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r2;
    ret;
OUTLINE:
    mov.u32 %r3, 42;
    st.shared.u32 [x], %r3;
    bra BACK;
}
)", [] { LaunchConfig c; c.block = {64, 1, 1}; return c; }(), {arg_u64(out)}, mem);
  for (int i = 0; i < 64; ++i) VCHECK_EQ(mem.load_scalar(out + i * 4, 4), uint64_t{42});
}

// Dynamic shared memory starts at its declared alignment, not wherever the
// static data happens to end.
VTEST(dynamic_shared_memory_starts_at_its_declared_alignment) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(8);
  const std::string ptx = R"(
.extern .shared .align 128 .b8 dyn[];
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<2>;
    .shared .align 4 .b8 small[12];
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, dyn;
    mov.u32 %r2, small;
    st.shared.u32 [%r2], %r1;
    st.global.u32 [%rd1], %r1;
    st.shared.u32 [%r1], %r1;
    ret;
}
)";
  LaunchConfig c;
  c.shared_bytes = 64;
  run(ptx, c, {arg_u64(out)}, mem);
  const uint64_t off = mem.load_scalar(out, 4);
  VCHECK(off >= 12 && off % 128 == 0);
}

// A kernel parameter reached through a generic pointer -- how a
// __grid_constant__ struct is read -- and cvta.param, which is the identity
// on a parameter's address.
VTEST(generic_pointer_to_a_kernel_parameter) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(8);
  std::vector<uint8_t> blob(16);
  const uint64_t v = 0x1122334455667788ull;
  std::memcpy(blob.data() + 8, &v, 8);
  run(R"(
.visible .entry k(.param .align 8 .b8 blob[16], .param .u64 out)
{
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    mov.b64 %rd2, blob;
    cvta.param.u64 %rd3, %rd2;
    ld.u64 %rd4, [%rd3+8];
    st.global.u64 [%rd1], %rd4;
    ret;
}
)", LaunchConfig{}, {blob, arg_u64(out)}, mem);
  VCHECK_EQ(mem.load_scalar(out, 8), v);
}

VTEST_MAIN
