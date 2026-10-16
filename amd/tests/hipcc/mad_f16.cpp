// The half-precision multiply-adds of the CDNA chips: v_mad_f16, v_mad_legacy_f16 and v_mac_f16 are the MAD forms, whose product is
// rounded to a half before the add ("1ULP accuracy"), where v_fma_f16 and v_fma_legacy_f16 are fused ("Fused half precision multiply
// add. 0.5ULP accuracy": CDNA2, CDNA3 and CDNA4 guides, V_MAD_F16 and V_FMA_F16; LLVM's definition of the node v_mad_f16 is selected for,
// ISD::FMAD in llvm/include/llvm/CodeGen/ISDOpcodes.h: "the same result as the separately rounded operations").
//
// The operands are chosen so that the two differ: a product that needs more than a half's eleven bits, and an addend that cancels its
// rounded value, so the fused result is the rounding error itself and the MAD result is zero. The expected values are computed here, on
// the host, by the rule in the guide (round the product to a half, then add and round), never from the simulator.
//
// Built for gfx90a, gfx942 and gfx950 (mad_f16.<arch>), wave64.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#pragma clang diagnostic ignored "-Winline-asm"

#define V3(NAME, INSTR) \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* o) { \
    const int i = threadIdx.x; uint32_t r = 0; asm volatile(INSTR : "+v"(r) : "v"(a[i]), "v"(b[i]), "v"(c[i])); o[i] = r; }
// A destination that is also the addend: it starts as c.
#define VACC(NAME, INSTR) \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* o) { \
    const int i = threadIdx.x; uint32_t r = c[i]; asm volatile(INSTR : "+v"(r) : "v"(a[i]), "v"(b[i])); o[i] = r; }

V3(k_mad_f16, "v_mad_f16 %0, %1, %2, %3")
V3(k_mad_legacy_f16, "v_mad_legacy_f16 %0, %1, %2, %3")
V3(k_fma_f16, "v_fma_f16 %0, %1, %2, %3")
V3(k_fma_legacy_f16, "v_fma_legacy_f16 %0, %1, %2, %3")
VACC(k_mac_f16, "v_mac_f16 %0, %1, %2")

static uint16_t h(float f) { _Float16 x = static_cast<_Float16>(f); uint16_t b; std::memcpy(&b, &x, 2); return b; }
static float hf(uint16_t b) { _Float16 x; std::memcpy(&x, &b, 2); return static_cast<float>(x); }

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return out;                                                                       \
    }                                                                                   \
  } while (0)

template <typename K>
static std::vector<uint32_t> run(K kernel, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const std::vector<uint32_t>& c) {
  uint32_t *da, *db, *dc, *dout;
  std::vector<uint32_t> out(64, 0xDEADBEEFu);
  CHECK(hipMalloc(&da, 256));
  CHECK(hipMalloc(&db, 256));
  CHECK(hipMalloc(&dc, 256));
  CHECK(hipMalloc(&dout, 256));
  CHECK(hipMemcpy(da, a.data(), 256, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(db, b.data(), 256, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dc, c.data(), 256, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(kernel, dim3(1), dim3(64), 0, 0, da, db, dc, dout);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(out.data(), dout, 256, hipMemcpyDeviceToHost));
  (void)hipFree(da), (void)hipFree(db), (void)hipFree(dc), (void)hipFree(dout);
  return out;
}

static int g_failed = 0;
static void check(const char* what, const std::vector<uint32_t>& out, const std::vector<uint32_t>& want) {
  int wrong = 0;
  for (int l = 0; l < 64; ++l)
    if ((out[l] & 0xFFFF) != want[l]) {
      if (wrong < 2) std::printf("  %s lane %d: got %04x want %04x\n", what, l, out[l] & 0xFFFF, want[l]);
      ++wrong;
    }
  std::printf("%s: %d of 64 wrong\n", what, wrong);
  g_failed += wrong != 0;
}

int main() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return 2;
  std::printf("device %s\n", prop.gcnArchName);
  std::vector<uint32_t> a(64), b(64), c(64), mad(64), fused(64);
  for (int l = 0; l < 64; ++l) {
    // 1 + i * 2^-10 times 1 + j * 2^-10: the exact product has up to 22 bits, a half keeps 11.
    const float x = 1.0f + static_cast<float>(1 + l % 7) * 0.0009765625f, y = 1.0f + static_cast<float>(1 + l % 5) * 0.0009765625f;
    a[l] = h(x), b[l] = h(y);
    const float exact = hf(a[l]) * hf(b[l]);   // (exact in a float)
    const float rounded = hf(h(exact));         // the product rounded to a half
    // Three lanes in four cancel the rounded product; the rest add an unrelated value.
    const float addend = l % 4 != 3 ? -rounded : 0.375f * static_cast<float>(l % 9 - 4);
    c[l] = h(addend);
    mad[l] = h(rounded + hf(c[l]));                // the product rounded, then the sum rounded
    fused[l] = h(static_cast<float>(static_cast<double>(hf(a[l])) * hf(b[l]) + hf(c[l])));   // one rounding (exact in a double)
  }
  int differ = 0;
  for (int l = 0; l < 64; ++l) differ += mad[l] != fused[l];
  std::printf("lanes where the two roundings differ: %d of 64\n", differ);   // the operands have to tell them apart
  check("v_mad_f16 rounds the product first", run(k_mad_f16, a, b, c), mad);
  check("v_mad_legacy_f16 rounds the product first", run(k_mad_legacy_f16, a, b, c), mad);
  check("v_mac_f16 rounds the product first", run(k_mac_f16, a, b, c), mad);
  check("v_fma_f16 is fused", run(k_fma_f16, a, b, c), fused);
  check("v_fma_legacy_f16 is fused", run(k_fma_legacy_f16, a, b, c), fused);
  std::printf("mad_f16 %s\n", g_failed ? "FAILED" : "ok");
  return g_failed;
}
