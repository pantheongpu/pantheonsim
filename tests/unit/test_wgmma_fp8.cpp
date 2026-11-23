// The e4m3 / e5m2 sum of wgmma on a Hopper device (include/vgpu/exec/tc_fp8.hpp, vgpu::kHopperWgmmaFp8Rule, docs/numerics/fp8-vectors.md):
//   * recorded_vectors: 48 vectors recorded on an H100 and an H200 through wgmma by Khattak and Mikaitis (tc_fp8_wgmma_vectors.inc), each one a
//     result the sequential f32 sum does not give; the rule gives every one of them, bit for bit.
//   * wgmma_instruction: the same vectors through a real wgmma.mma_async in the PTX engine (the SIMD route and the general form), on the H100 profile.
//   * the sequential sum comes back with VGPU_TC_EXACT=0, for the sparse forms, and on a device whose profile names no rule.
//   * profiles_name_their_wgmma_fp8_sum: every NVIDIA profile has the sum its compute capability's evidence gives it.
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/tc_fp8.hpp"
#include "vgpu/exec/wgmma.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using namespace vgpu::exec::tc86;

namespace {

float f(uint32_t u) { return std::bit_cast<float>(u); }
uint32_t bits(double x) { return std::bit_cast<uint32_t>(static_cast<float>(x)); }

struct Vec {
  int fmt;
  uint32_t a[32], b[32], d;
};
const Vec kVectors[] = {
#include "tc_fp8_wgmma_vectors.inc"
};

int emin_of(int fmt) { return fmt ? kEminE5M2 : kEminE4M3; }

// float values (exact e4m3 / e5m2 ones) to the format's byte
uint8_t to_byte(float x, int fmt) {
  if (x == 0) return std::signbit(x) ? 0x80 : 0;
  for (unsigned byte = 0; byte < 256; ++byte) {
    const int eb = fmt ? 5 : 4, mb = fmt ? 2 : 3, bias = fmt ? 15 : 7;
    const int e = (byte >> mb) & ((1 << eb) - 1), m = byte & ((1 << mb) - 1);
    if (fmt ? e == 31 : (e == 15 && m == 7)) continue;
    const double v = (byte >> 7 ? -1 : 1) * (e ? std::ldexp(1.0 + m / double(1 << mb), e - bias) : std::ldexp(m / double(1 << mb), 1 - bias));
    if (v == x) return static_cast<uint8_t>(byte);
  }
  throw vtest::Failure("a recorded value is not an FP8 value: " + std::to_string(x));
}

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

// A K-major, unswizzled operand of 8-row core matrices (16 bytes a row): row mn, byte k at (mn % 8) * 16 + (mn / 8) * sbo + k % 16 + (k / 16) * lbo.
constexpr uint32_t kLbo = 128, kSbo = 256, kBBase = 4096, kSmem = 8192;
uint32_t smem_at(uint32_t mn, uint32_t k) { return (mn % 8) * 16 + (mn / 8) * kSbo + k % 16 + (k / 16) * kLbo; }
uint64_t desc(uint32_t start) { return uint64_t{start >> 4} | (uint64_t{kLbo >> 4} << 16) | (uint64_t{kSbo >> 4} << 32); }

// D[0][0] of a m64n8k32 wgmma on the vector (row 0 of A, column 0 of B, the rest zero), from a zero accumulator; the word thread 0 holds.
uint32_t run_wgmma(const Vec& v, const DeviceProfile& prof) {
  const char* fmt = v.fmt ? "e5m2" : "e4m3";
  std::vector<uint8_t> img(kSmem, 0);
  for (uint32_t k = 0; k < 32; ++k) {
    img[smem_at(0, k)] = to_byte(f(v.a[k]), v.fmt);
    img[kBBase + smem_at(0, k)] = to_byte(f(v.b[k]), v.fmt);
  }
  const std::string ptx = std::string(".version 8.3\n.target sm_90a\n.address_size 64\n") + R"(
.visible .entry k(.param .u64 pimg, .param .u64 pd, .param .u64 da, .param .u64 db)
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
    mov.b32 %d0, 0;
    mov.b32 %d1, 0;
    mov.b32 %d2, 0;
    mov.b32 %d3, 0;
    setp.ne.u32 %p2, %r1, 99999;
    wgmma.fence.sync.aligned;
    wgmma.mma_async.sync.aligned.m64n8k32.f32.)" + fmt + "." + fmt + R"( {%d0, %d1, %d2, %d3}, %rd5, %rd6, %p2, 1, 1;
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
  auto put = [&](const void* p, size_t n) {
    const uint64_t va = mem.alloc(n);
    mem.write(va, p, n);
    return va;
  };
  const uint64_t pimg = put(img.data(), img.size());
  std::vector<uint32_t> zeros(128 * 4, 0);
  const uint64_t pd = put(zeros.data(), zeros.size() * 4);
  exec::LaunchConfig cfg;
  cfg.block = {128, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(pimg), arg_u64(pd), arg_u64(desc(0)), arg_u64(desc(kBBase))}, mem, prof);
  uint32_t out = 0;
  mem.read(pd, &out, 4);
  return out;
}

}  // namespace

VTEST(recorded_vectors) {
  int seq_differs = 0;
  for (const Vec& v : kVectors) {
    double a[32], b[32];
    for (int k = 0; k < 32; ++k) a[k] = f(v.a[k]), b[k] = f(v.b[k]);
    const double got = fp8_dot(kHopperWgmmaFp8Rule, false, emin_of(v.fmt), emin_of(v.fmt), a, 1, b, 1, 32, 0.0);
    if (bits(got) != v.d) throw vtest::Failure("vector " + std::to_string(&v - kVectors) + ": rule " + std::to_string(got) + ", recorded " + std::to_string(f(v.d)));
    float acc = 0.0f;
    for (int k = 0; k < 32; ++k) acc += static_cast<float>(a[k] * b[k]);
    if (bits(acc) != v.d) ++seq_differs;
  }
  VCHECK_EQ(seq_differs, 48);   // the generator kept only vectors the sequential sum gets wrong
  // the rule is one the vectors tell from its neighbours: a grid bit more, chunks of 16, a full 23-bit result
  int rejected[3] = {};
  for (const Vec& v : kVectors) {
    double a[32], b[32];
    for (int k = 0; k < 32; ++k) a[k] = f(v.a[k]), b[k] = f(v.b[k]);
    const Fp8SumRule other[3] = {{"grid 15", 32, 15, 13}, {"chunks of 16", 16, 14, 13}, {"23-bit result", 32, 14, 23}};
    for (int i = 0; i < 3; ++i)
      if (bits(fp8_dot(other[i], false, emin_of(v.fmt), emin_of(v.fmt), a, 1, b, 1, 32, 0.0)) != v.d) ++rejected[i];
  }
  for (int i = 0; i < 3; ++i) VCHECK(rejected[i] >= 1);
}

VTEST(wgmma_instruction) {
  const DeviceProfile h = load_gpu("nvidia/h100");
  VCHECK(h.numerics.tensor_sum_wgmma_fp8 == WgmmaFp8Sum::Hopper);
  for (const char* fast : {"1", "0"}) {
    setenv("VGPU_FASTPATH", fast, 1);
    for (const Vec& v : kVectors) {
      uint32_t got;
      try {
        got = run_wgmma(v, h);
      } catch (const Error& e) {
        unsetenv("VGPU_FASTPATH");
        throw vtest::Failure(std::string("vector ") + std::to_string(&v - kVectors) + ": " + e.what());
      }
      if (got != v.d) {
        unsetenv("VGPU_FASTPATH");
        char buf[160];
        std::snprintf(buf, sizeof buf, "vector %zu, fast path %s: wgmma gave 0x%08x, recorded 0x%08x", static_cast<size_t>(&v - kVectors), fast, got, v.d);
        throw vtest::Failure(buf);
      }
    }
  }
  unsetenv("VGPU_FASTPATH");
  // VGPU_TC_EXACT=0 gives the sequential sum back: some of these vectors then come out different
  setenv("VGPU_TC_EXACT", "0", 1);
  refresh_exact("0");
  int differ = 0;
  for (const Vec& v : kVectors)
    if (run_wgmma(v, h) != v.d) ++differ;
  refresh_exact(nullptr);
  unsetenv("VGPU_TC_EXACT");
  VCHECK_EQ(differ, 48);
}

VTEST(profiles_name_their_wgmma_fp8_sum) {
  int hopper = 0;
  for (const std::string& id : available_gpus()) {
    const DeviceProfile p = load_gpu(id);
    const bool sm90 = p.vendor == "nvidia" && p.cc_major == 9 && p.cc_minor == 0;
    const WgmmaFp8Sum want = sm90 ? WgmmaFp8Sum::Hopper : WgmmaFp8Sum::Sequential;
    if (p.numerics.tensor_sum_wgmma_fp8 != want) throw vtest::Failure(id + ": numerics.tensor_sum_wgmma_fp8 is not what its compute capability's evidence gives");
    if (!sm90) {
      VCHECK(!p.provenance_of("numerics.tensor_sum_wgmma_fp8"));   // not stated, so no class is claimed
      continue;
    }
    ++hopper;
    for (const char* key : {"numerics.tensor_sum_wgmma_fp8", "numerics.tensor_sum_wgmma_fp8_chunk", "numerics.tensor_sum_wgmma_fp8_grid", "numerics.tensor_sum_wgmma_fp8_out_bits"}) {
      const auto c = p.provenance_of(key);
      if (!c) throw vtest::Failure(id + ": " + key + " has no class");
      VCHECK(*c == Provenance::Inferred);   // no card of ours ran a wgmma
      VCHECK(!p.evidence_of(key).empty());
    }
    const Fp8SumRule* r = wgmma_fp8_sum_rule(want);
    VCHECK(p.numerics.tensor_sum_wgmma_fp8_chunk == r->chunk && p.numerics.tensor_sum_wgmma_fp8_grid == r->grid &&
           p.numerics.tensor_sum_wgmma_fp8_out_bits == r->out_bits);
  }
  VCHECK_EQ(hopper, 6);   // h100 h100-pcie h100-nvl h200 h200-nvl gh200-480gb
  const DeviceProfile h = load_gpu("nvidia/h100"), l = load_gpu("nvidia/l40s");
  VCHECK(wgmma_fp8_rule(h) != nullptr && wgmma_fp8_rule(l) == nullptr);
  refresh_exact("0");
  VCHECK(wgmma_fp8_rule(h) == nullptr);
  refresh_exact(nullptr);
}

VTEST_MAIN
