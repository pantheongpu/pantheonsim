// The sums of mma.m8n8k4, wgmma and tcgen05.mma on f16, bf16 and tf32 operands (include/vgpu/exec/tc_forms.hpp, vgpu::FormSum, issue #664):
//   * volta_rule_reproduces_the_v100_vectors: 24 f32-accumulator and 24 f16-accumulator vectors recorded on a V100 by Khattak and Mikaitis
//     (nvidia/tests/data/tcpapers/V100.*.txt, BSD-2, NOTICE): the Volta rule (chunks of 4, cut at 2^(top - 24)) gives every one, and the rules next to
//     it (Ampere's grid 25, grid 23, chunks of 2) do not.
//   * m8n8k4_instruction: the same vectors through a real mma.sync.aligned.m8n8k4 in the PTX engine on the t4 profile; VGPU_TC_EXACT=0 and a profile
//     with no rule for the form (h100) give the sequential sum, which differs.
//   * wgmma_instruction: 24 recorded H100 vectors of each f16, bf16, tf32 (f32 accumulator) and f16 (f16 accumulator), with the accumulator in the
//     registers, through a real wgmma.mma_async in the PTX engine on the h100 profile (the SIMD route and the general form). These are mma.sync
//     records: that wgmma sums as the same device's mma.sync is MMA-Sim's claim (Xie et al., arXiv 2511.10909), which is what the profile says
//     (`inferred`). The sequential sum (VGPU_TC_EXACT=0) must not reproduce them.
//   * profiles_name_their_form_sums: every NVIDIA profile has the sums its compute capability's evidence gives it, with a class and evidence.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/tc_forms.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using namespace vgpu::exec::tc86;

namespace {

float fl(uint32_t u) { return std::bit_cast<float>(u); }
uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }

// A float rounded to the nearest binary16 (ties to even), as a float.
float to_f16(float x) {
  if (!std::isfinite(x) || x == 0) return x;
  int e;
  std::frexp(static_cast<double>(x), &e);
  const double q = std::ldexp(1.0, std::max(e - 11, -24));
  const double r = std::nearbyint(static_cast<double>(x) / q) * q;
  return std::fabs(r) > 65504.0 ? std::copysign(INFINITY, x) : static_cast<float>(r);
}
uint16_t f16_bits(float x) {   // x is a binary16 value
  if (x != x) return 0x7e00;
  const uint16_t s = std::signbit(x) ? 0x8000 : 0;
  const double a = std::fabs(static_cast<double>(x));
  if (a == 0) return s;
  if (std::isinf(a)) return s | 0x7c00;
  int e;
  const double m = std::frexp(a, &e);   // a = m * 2^e, m in [0.5, 1)
  if (e - 1 < -14) return s | static_cast<uint16_t>(std::ldexp(a, 24));
  return s | static_cast<uint16_t>(((e - 1 + 15) << 10) | (static_cast<int>(std::ldexp(m, 11)) & 0x3ff));
}
float f16_from_bits(uint16_t h) {
  const int e = (h >> 10) & 31, m = h & 0x3ff;
  const double v = e == 0 ? std::ldexp(m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(1024 + m, e - 25);
  return static_cast<float>(h & 0x8000 ? -v : v);
}

struct Vec {
  float a[16], b[16], c, d;   // d: the recorded result (a float holding it)
};
std::vector<Vec> read_vectors(const std::string& name, unsigned K) {
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/nvidia/tests/data/tcpapers/" + name);
  if (!in) throw vtest::Failure("cannot open " + name);
  std::vector<Vec> out;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::vector<uint32_t> t;
    std::string tok;
    while (ss >> tok) t.push_back(static_cast<uint32_t>(std::stoul(tok, nullptr, 16)));
    if (t.size() != 2 * K + 2) throw vtest::Failure(name + ": a line has " + std::to_string(t.size()) + " tokens, not " + std::to_string(2 * K + 2));
    Vec v{};
    for (unsigned k = 0; k < K; ++k) v.a[k] = fl(t[k]), v.b[k] = fl(t[K + k]);
    v.c = fl(t[2 * K]);
    v.d = fl(t[2 * K + 1]);
    out.push_back(v);
  }
  return out;
}

// the rule on one vector (an f16 accumulator takes c rounded to binary16)
float rule_dot(const SumRule& r, bool tf32, int emin, bool acc16, const Vec& v, unsigned K) {
  double a[16], b[16];
  for (unsigned k = 0; k < K; ++k) a[k] = v.a[k], b[k] = v.b[k];
  return form_dot(r, tf32, emin, acc16, a, 1, b, 1, K, acc16 ? to_f16(v.c) : v.c);
}
float sequential_dot(bool acc16, const Vec& v, unsigned K) {
  float acc = acc16 ? to_f16(v.c) : v.c;
  for (unsigned k = 0; k < K; ++k) acc += v.a[k] * v.b[k];
  return acc16 ? to_f16(acc) : acc;
}

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
uint64_t put(MemoryManager& mem, const void* p, size_t n) {
  const uint64_t va = mem.alloc(n);
  mem.write(va, p, n);
  return va;
}

uint32_t pack_h(float lo, float hi) { return f16_bits(to_f16(lo)) | (uint32_t{f16_bits(to_f16(hi))} << 16); }

// ---- mma.m8n8k4: lane 0 holds row 0 of A, column 0 of B and C(0,0); D(0,0) is lane 0's first register ----
float run_m8n8k4(const Vec& v, bool acc16, const DeviceProfile& prof) {
  std::vector<uint32_t> in(32 * 5, 0);
  in[0] = pack_h(v.a[0], v.a[1]);
  in[1] = pack_h(v.a[2], v.a[3]);
  in[2] = pack_h(v.b[0], v.b[1]);
  in[3] = pack_h(v.b[2], v.b[3]);
  in[4] = acc16 ? pack_h(v.c, 0) : bits(v.c);
  const std::string ptx = std::string(".version 7.0\n.target sm_75\n.address_size 64\n") +
      (acc16 ? R"(
.visible .entry k(.param .u64 pin, .param .u64 pout)
{
    .reg .b32 %r<24>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [pin];
    ld.param.u64 %rd2, [pout];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 20;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.b32 %r2, [%rd4];
    ld.global.b32 %r3, [%rd4+4];
    ld.global.b32 %r4, [%rd4+8];
    ld.global.b32 %r5, [%rd4+12];
    ld.global.b32 %r6, [%rd4+16];
    mov.b32 %r7, 0;
    mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 {%r10, %r11, %r12, %r13}, {%r2, %r3}, {%r4, %r5}, {%r6, %r7, %r7, %r7};
    mul.wide.u32 %rd5, %r1, 4;
    add.u64 %rd6, %rd2, %rd5;
    st.global.b32 [%rd6], %r10;
    ret;
}
)"
                 : R"(
.visible .entry k(.param .u64 pin, .param .u64 pout)
{
    .reg .b32 %r<8>;
    .reg .f32 %f<24>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [pin];
    ld.param.u64 %rd2, [pout];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 20;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.b32 %r2, [%rd4];
    ld.global.b32 %r3, [%rd4+4];
    ld.global.b32 %r4, [%rd4+8];
    ld.global.b32 %r5, [%rd4+12];
    ld.global.f32 %f1, [%rd4+16];
    mov.f32 %f2, 0f00000000;
    mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%f10, %f11, %f12, %f13, %f14, %f15, %f16, %f17}, {%r2, %r3}, {%r4, %r5}, {%f1, %f2, %f2, %f2, %f2, %f2, %f2, %f2};
    mul.wide.u32 %rd5, %r1, 4;
    add.u64 %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f10;
    ret;
}
)");
  MemoryManager mem{1 << 20};
  auto m = ptx::parse(ptx);
  const uint64_t pin = put(mem, in.data(), in.size() * 4);
  std::vector<uint32_t> zeros(32, 0);
  const uint64_t pout = put(mem, zeros.data(), 32 * 4);
  exec::LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(pin), arg_u64(pout)}, mem, prof);
  uint32_t out = 0;
  mem.read(pout, &out, 4);
  return acc16 ? f16_from_bits(static_cast<uint16_t>(out & 0xffff)) : fl(out);
}

// ---- wgmma m64nNk{16,8}: A in row 0, B in column 0 (K-major, no swizzle: 8-row core matrices of 16 bytes), the accumulator in thread 0's register 0 ----
constexpr uint32_t kLbo = 128, kSbo = 256, kBBase = 4096, kSmem = 8192;
uint32_t smem_at(uint32_t mn, uint32_t kb) { return (mn % 8) * 16 + (mn / 8) * kSbo + kb % 16 + (kb / 16) * kLbo; }
uint64_t desc(uint32_t start) { return uint64_t{start >> 4} | (uint64_t{kLbo >> 4} << 16) | (uint64_t{kSbo >> 4} << 32); }

enum class WFmt { F16, BF16, TF32 };
uint16_t bf16_bits(float x) { return static_cast<uint16_t>(bits(x) >> 16); }   // a bf16 value: the top half of the float

float run_wgmma(const Vec& v, WFmt fmt, unsigned K, bool acc16, const DeviceProfile& prof) {
  const uint32_t eb = fmt == WFmt::TF32 ? 4 : 2;
  std::vector<uint8_t> img(kSmem, 0);
  for (uint32_t k = 0; k < K; ++k) {
    uint8_t* pa = &img[smem_at(0, k * eb)];
    uint8_t* pb = &img[kBBase + smem_at(0, k * eb)];
    if (fmt == WFmt::TF32) {
      const uint32_t ua = bits(v.a[k]), ub = bits(v.b[k]);
      std::memcpy(pa, &ua, 4);
      std::memcpy(pb, &ub, 4);
    } else {
      const uint16_t ua = fmt == WFmt::F16 ? f16_bits(v.a[k]) : bf16_bits(v.a[k]), ub = fmt == WFmt::F16 ? f16_bits(v.b[k]) : bf16_bits(v.b[k]);
      std::memcpy(pa, &ua, 2);
      std::memcpy(pb, &ub, 2);
    }
  }
  const std::string shape = fmt == WFmt::TF32 ? "m64n8k8" : "m64n8k16";
  const std::string ty = fmt == WFmt::F16 ? "f16" : fmt == WFmt::BF16 ? "bf16" : "tf32";
  const std::string dty = acc16 ? "f16" : "f32";
  const std::string regs = acc16 ? "{%d0, %d1}" : "{%d0, %d1, %d2, %d3}";
  const std::string trans = fmt == WFmt::TF32 ? "" : ", 0, 0";
  const std::string ptx = std::string(".version 8.3\n.target sm_90a\n.address_size 64\n") + R"(
.visible .entry k(.param .u64 pimg, .param .u64 pd, .param .u64 da, .param .u64 db, .param .u32 pc)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b32 %d<4>;
    .reg .b64 %rd<16>;
    .shared .align 1024 .b8 smem[8192];
    ld.param.u64 %rd1, [pimg];
    ld.param.u64 %rd4, [pd];
    ld.param.u64 %rd5, [da];
    ld.param.u64 %rd6, [db];
    ld.param.u32 %r9, [pc];
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
    add.u32 %r3, %r3, 512;
    bra COPY;
COPIED:
    bar.sync 0;
    shr.u32 %r6, %r2, 4;
    cvt.u64.u32 %rd11, %r6;
    add.u64 %rd5, %rd5, %rd11;
    add.u64 %rd6, %rd6, %rd11;
    setp.eq.u32 %p3, %r1, 0;
    selp.b32 %d0, %r9, 0, %p3;
    mov.b32 %d1, 0;
    mov.b32 %d2, 0;
    mov.b32 %d3, 0;
    setp.ne.u32 %p2, %r1, 99999;
    wgmma.fence.sync.aligned;
    wgmma.mma_async.sync.aligned.)" + shape + "." + dty + "." + ty + "." + ty + " " + regs + R"(, %rd5, %rd6, %p2, 1, 1)" + trans + R"(;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;
    mul.wide.u32 %rd9, %r1, 16;
    add.u64 %rd12, %rd4, %rd9;
    st.global.u32 [%rd12+0], %d0;
    st.global.u32 [%rd12+4], %d1;
    st.global.u32 [%rd12+8], %d2;
    st.global.u32 [%rd12+12], %d3;
    ret;
}
)";
  MemoryManager mem{1 << 22};
  auto m = ptx::parse(ptx);
  const uint64_t pimg = put(mem, img.data(), img.size());
  std::vector<uint32_t> zeros(128 * 4, 0);
  const uint64_t pd = put(mem, zeros.data(), zeros.size() * 4);
  exec::LaunchConfig cfg;
  cfg.block = {128, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(pimg), arg_u64(pd), arg_u64(desc(0)), arg_u64(desc(kBBase)), arg_u32(acc16 ? pack_h(v.c, 0) : bits(v.c))}, mem, prof);
  uint32_t out = 0;
  mem.read(pd, &out, 4);
  return acc16 ? f16_from_bits(static_cast<uint16_t>(out & 0xffff)) : fl(out);
}

}  // namespace

VTEST(volta_rule_reproduces_the_v100_vectors) {
  for (const bool acc16 : {false, true}) {
    const std::vector<Vec> vs = read_vectors(acc16 ? "V100.fp16.fp16.txt" : "V100.fp16.fp32.txt", 4);
    VCHECK_EQ(vs.size(), size_t{24});
    int seq_differs = 0;
    for (const Vec& v : vs) {
      const float got = rule_dot(kVoltaSumRule, false, kEminF16, acc16, v, 4);
      if (bits(got) != bits(v.d)) throw vtest::Failure("V100 vector " + std::to_string(&v - vs.data()) + (acc16 ? " (f16)" : " (f32)") + ": rule " + std::to_string(got) + ", recorded " + std::to_string(v.d));
      if (bits(sequential_dot(acc16, v, 4)) != bits(v.d)) ++seq_differs;
    }
    if (!acc16) {
      VCHECK(seq_differs >= 1);
      // the rules next to it: Ampere's grid 25 (the sample holds 16 vectors it gets wrong), grid 23, chunks of 2
      const TensorSumRule other[3] = {{"grid 25", 4, 4, 25}, {"grid 23", 4, 4, 23}, {"chunks of 2", 2, 2, 24}};
      for (const TensorSumRule& r : other) {
        int rejected = 0;
        for (const Vec& v : vs)
          if (bits(rule_dot(r, false, kEminF16, false, v, 4)) != bits(v.d)) ++rejected;
        if (rejected < 1) throw vtest::Failure(std::string(r.name) + " reproduces all of the V100 sample: it does not tell the rules apart");
      }
    }
  }
}

VTEST(m8n8k4_instruction) {
  const DeviceProfile t4 = load_gpu("nvidia/t4");
  VCHECK(t4.numerics.tensor_sum_m8n8k4 == FormSum::Volta);
  for (const bool acc16 : {false, true}) {
    const std::vector<Vec> vs = read_vectors(acc16 ? "V100.fp16.fp16.txt" : "V100.fp16.fp32.txt", 4);
    for (const char* fast : {"1", "0"}) {
      setenv("VGPU_FASTPATH", fast, 1);
      for (const Vec& v : vs) {
        const float got = run_m8n8k4(v, acc16, t4);
        if (bits(got) != bits(v.d)) {
          unsetenv("VGPU_FASTPATH");
          throw vtest::Failure("vector " + std::to_string(&v - vs.data()) + (acc16 ? " (f16)" : " (f32)") + ", fast path " + fast + ": mma.m8n8k4 gave " + std::to_string(got) + ", recorded " + std::to_string(v.d));
        }
      }
    }
    unsetenv("VGPU_FASTPATH");
  }
  // without the rule: the sequential sum, which is not the card's on this sample
  const std::vector<Vec> vs = read_vectors("V100.fp16.fp32.txt", 4);
  int differ_off = 0, differ_other = 0;
  setenv("VGPU_TC_EXACT", "0", 1);
  refresh_exact("0");
  for (const Vec& v : vs)
    if (bits(run_m8n8k4(v, false, t4)) != bits(v.d)) ++differ_off;
  refresh_exact(nullptr);
  unsetenv("VGPU_TC_EXACT");
  const DeviceProfile r = load_gpu("nvidia/h100");   // (no card of ours has run the form on a Hopper device: the sequential sum)
  VCHECK(r.numerics.tensor_sum_m8n8k4 == FormSum::Sequential);
  for (const Vec& v : vs)
    if (bits(run_m8n8k4(v, false, r)) != bits(v.d)) ++differ_other;
  VCHECK(differ_off >= 1);
  VCHECK_EQ(differ_other, differ_off);
}

VTEST(m8n8k4_emulation_on_sm_80_and_later) {
  // an A10G's results (nvidia/tests/data/tcpapers/m8n8k4.a10g.fp32.txt: 800 inputs of ours: subnormals, cancellation, large and small terms): ptxas emulates
  // the form, the four products added in order and C last, each add rounded to f32; the simulator's PTX engine gives them on every profile of 8.x
  const std::vector<Vec> vs = read_vectors("m8n8k4.a10g.fp32.txt", 4);
  VCHECK_EQ(vs.size(), size_t{800});
  for (const char* id : {"nvidia/a10g", "nvidia/rtx3060", "nvidia/a100", "nvidia/l40s"}) {
    const DeviceProfile p = load_gpu(id);
    VCHECK(p.numerics.tensor_sum_m8n8k4 == FormSum::ChainCLast);
    for (const char* fast : {"1", "0"}) {
      setenv("VGPU_FASTPATH", fast, 1);
      for (const Vec& v : vs) {
        const float got = run_m8n8k4(v, false, p);
        if (bits(got) != bits(v.d) && !(got != got && v.d != v.d)) {
          unsetenv("VGPU_FASTPATH");
          throw vtest::Failure(std::string(id) + " vector " + std::to_string(&v - vs.data()) + ", fast path " + fast + ": gave " + std::to_string(got) + ", the A10G " + std::to_string(v.d));
        }
      }
    }
    unsetenv("VGPU_FASTPATH");
  }
  // not the T4's rule: an A10G's results are not a Volta step's, and a T4 profile does not give them
  int volta_differs = 0;
  for (const Vec& v : vs) {
    Vec u = v;
    if (bits(rule_dot(kVoltaSumRule, false, kEminF16, false, u, 4)) != bits(v.d)) ++volta_differs;
  }
  VCHECK(volta_differs >= 50);
  setenv("VGPU_TC_EXACT", "0", 1);
  refresh_exact("0");
  int seq_differs = 0;
  const DeviceProfile a = load_gpu("nvidia/a10g");
  for (const Vec& v : vs)
    if (bits(run_m8n8k4(v, false, a)) != bits(v.d)) ++seq_differs;
  refresh_exact(nullptr);
  unsetenv("VGPU_TC_EXACT");
  VCHECK(seq_differs >= 50);
  // the T4's 800 results, both accumulators, through the t4 profile
  for (const bool acc16 : {false, true}) {
    const std::vector<Vec> ts = read_vectors(acc16 ? "m8n8k4.t4.fp16.txt" : "m8n8k4.t4.fp32.txt", 4);
    const DeviceProfile t4 = load_gpu("nvidia/t4");
    for (const Vec& v : ts) {
      const float got = run_m8n8k4(v, acc16, t4);
      if (bits(got) != bits(v.d) && !(got != got && v.d != v.d))
        throw vtest::Failure(std::string(acc16 ? "T4 f16" : "T4 f32") + " vector " + std::to_string(&v - ts.data()) + ": gave " + std::to_string(got) + ", the T4 " + std::to_string(v.d));
    }
  }
}

VTEST(wgmma_instruction) {
  const DeviceProfile h = load_gpu("nvidia/h100");
  VCHECK(h.numerics.tensor_sum_wgmma == FormSum::Hopper);
  struct File { const char* name; WFmt fmt; unsigned K, Kwgmma; bool acc16; };
  const File files[] = {{"H100.fp16.fp32.txt", WFmt::F16, 16, 16, false},
                        {"H100.bf16.fp32.txt", WFmt::BF16, 16, 16, false},
                        {"H100.tf32.fp32.txt", WFmt::TF32, 4, 8, false},    // (a tf32 vector is 4 products: the other 4 of K = 8 are zeros)
                        {"H100.fp16.fp16.txt", WFmt::F16, 16, 16, true}};
  for (const File& f : files) {
    const std::vector<Vec> vs = read_vectors(f.name, f.K);
    VCHECK_EQ(vs.size(), size_t{24});
    for (const char* fast : {"1", "0"}) {
      setenv("VGPU_FASTPATH", fast, 1);
      for (const Vec& v : vs) {
        float got;
        try {
          got = run_wgmma(v, f.fmt, f.Kwgmma, f.acc16, h);
        } catch (const Error& e) {
          unsetenv("VGPU_FASTPATH");
          throw vtest::Failure(std::string(f.name) + " vector " + std::to_string(&v - vs.data()) + ": " + e.what());
        }
        if (bits(got) != bits(v.d)) {
          unsetenv("VGPU_FASTPATH");
          throw vtest::Failure(std::string(f.name) + " vector " + std::to_string(&v - vs.data()) + ", fast path " + fast + ": wgmma gave " + std::to_string(got) + ", recorded " + std::to_string(v.d));
        }
      }
    }
    unsetenv("VGPU_FASTPATH");
    if (f.acc16) continue;   // (the sequential sum is the card's on nearly every f16-accumulator vector, and the sample does not tell them apart)
    setenv("VGPU_TC_EXACT", "0", 1);
    refresh_exact("0");
    int differ = 0;
    for (const Vec& v : vs)
      if (bits(run_wgmma(v, f.fmt, f.Kwgmma, f.acc16, h)) != bits(v.d)) ++differ;
    refresh_exact(nullptr);
    unsetenv("VGPU_TC_EXACT");
    if (differ < 1) throw vtest::Failure(std::string(f.name) + ": VGPU_TC_EXACT=0 reproduces the whole sample, so it does not tell the rule from the sequential sum");
  }
  // a device with no rule for the form (an L40S has no wgmma, but the profile key is the only thing the engine reads) keeps the sequential sum
  DeviceProfile none = h;
  none.numerics.tensor_sum_wgmma = FormSum::Sequential;
  const std::vector<Vec> vs = read_vectors("H100.fp16.fp32.txt", 16);
  int differ = 0;
  for (const Vec& v : vs)
    if (bits(run_wgmma(v, WFmt::F16, 16, false, none)) != bits(v.d)) ++differ;
  VCHECK(differ >= 1);
}

VTEST(profiles_name_their_form_sums) {
  int wg = 0, tc = 0, m8 = 0, sp = 0;
  for (const std::string& id : available_gpus()) {
    const DeviceProfile p = load_gpu(id);
    const bool nv = p.vendor == "nvidia";
    const FormSum want_wg = nv && p.cc_major == 9 && p.cc_minor == 0 ? FormSum::Hopper : FormSum::Sequential;
    const FormSum want_tc = nv && p.cc_major == 10 && p.cc_minor == 0 ? FormSum::Hopper : FormSum::Sequential;
    const bool cc8 = nv && p.cc_major == 8 && (p.cc_minor == 0 || p.cc_minor == 6 || p.cc_minor == 9);
    const FormSum want_m8 = nv && p.cc_major == 7 && p.cc_minor == 5 ? FormSum::Volta : cc8 ? FormSum::ChainCLast : FormSum::Sequential;
    const FormSum want_sp = cc8 ? FormSum::Ampere : FormSum::Sequential;
    if (p.numerics.tensor_sum_wgmma != want_wg) throw vtest::Failure(id + ": numerics.tensor_sum_wgmma is not what its compute capability's evidence gives");
    if (p.numerics.tensor_sum_tcgen05 != want_tc) throw vtest::Failure(id + ": numerics.tensor_sum_tcgen05 is not what its compute capability's evidence gives");
    if (p.numerics.tensor_sum_m8n8k4 != want_m8) throw vtest::Failure(id + ": numerics.tensor_sum_m8n8k4 is not what its compute capability's evidence gives");
    const auto named = [&](FormSum s, const char* key, int& count) {
      if (s == FormSum::Sequential) {
        VCHECK(!p.provenance_of(key));   // not stated, so no class is claimed
        return;
      }
      ++count;
      const auto c = p.provenance_of(key);
      if (!c) throw vtest::Failure(id + ": " + key + " has no class");
      VCHECK(*c == Provenance::Inferred || *c == Provenance::Measured);
      VCHECK(!p.evidence_of(key).empty());
    };
    named(p.numerics.tensor_sum_wgmma, "numerics.tensor_sum_wgmma", wg);
    named(p.numerics.tensor_sum_tcgen05, "numerics.tensor_sum_tcgen05", tc);
    named(p.numerics.tensor_sum_m8n8k4, "numerics.tensor_sum_m8n8k4", m8);
    if (p.numerics.tensor_sum_sparse != want_sp) throw vtest::Failure(id + ": numerics.tensor_sum_sparse is not what its compute capability's evidence gives");
    named(p.numerics.tensor_sum_sparse, "numerics.tensor_sum_sparse", sp);
    // measured is for the cards that ran: the T4's m8n8k4 and the A10G's m8n8k4 and sparse forms; every other profile of those compute capabilities is inferred
    const auto cls = [&](const char* key) { return *p.provenance_of(key); };
    if (want_m8 != FormSum::Sequential) VCHECK((cls("numerics.tensor_sum_m8n8k4") == Provenance::Measured) == (id == "nvidia/t4" || id == "nvidia/a10g"));
    if (want_sp != FormSum::Sequential) VCHECK((cls("numerics.tensor_sum_sparse") == Provenance::Measured) == (id == "nvidia/a10g"));
    // no card of ours ran a wgmma or a tcgen05: those are inferred, never measured
    if (want_wg != FormSum::Sequential) VCHECK(*p.provenance_of("numerics.tensor_sum_wgmma") == Provenance::Inferred);
    if (want_tc != FormSum::Sequential) VCHECK(*p.provenance_of("numerics.tensor_sum_tcgen05") == Provenance::Inferred);
  }
  VCHECK_EQ(wg, 6);   // h100 h100-pcie h100-nvl h200 h200-nvl gh200-480gb
  VCHECK_EQ(tc, 2);   // b200 gb200
  VCHECK_EQ(m8, 21);   // t4 rtx2080ti, and the 19 profiles of compute capability 8.0, 8.6 and 8.9
  VCHECK_EQ(sp, 19);
  const DeviceProfile h = load_gpu("nvidia/h100"), l = load_gpu("nvidia/l40s");
  VCHECK(wgmma_rule(h) != nullptr && wgmma_rule(l) == nullptr);
  refresh_exact("0");
  VCHECK(wgmma_rule(h) == nullptr);
  refresh_exact(nullptr);
}

VTEST_MAIN
