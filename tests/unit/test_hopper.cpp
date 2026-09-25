// Hopper (sm_90a): the warpgroup MMA.
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

VTEST_MAIN
