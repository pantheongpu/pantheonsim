// The instructions and rules of the CDNA3 and CDNA4 ISA guides that gfx942 and gfx950 had left decoded-but-refused or undecoded (gfx9_decode.gfx942 and
// .gfx950, wave64), each in inline assembly with a case worked out by hand from the guide: v_cvt_pkaccum_u8_f32, v_madmk_f16 and v_madak_f16, the
// relative scalar moves s_movrels / s_movreld, gfx950's v_pk_minimum3_f16 and v_pk_maximum3_f16, the float dot products' flushing of denormals
// (guides, 6.4), and the destination half of the 16-bit VOP3 forms under OPSEL[3] (V_MAD_F16 and its kin keep the other half; the "legacy" forms zero it).
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#pragma clang diagnostic ignored "-Winline-asm"

static int failed = 0;
static void report(const char* what, int wrong, int of) {
  std::printf("%s: %d of %d wrong\n", what, wrong, of);
  failed += wrong != 0;
}

// r starts as `init`; the instruction reads a, b, c per lane and writes r.
#define KERNEL(NAME, INIT, INSTR, ...) \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* o) { \
    const int i = threadIdx.x; uint32_t r = INIT; asm volatile(INSTR : "+v"(r) : __VA_ARGS__); o[i] = r; }
#define A1 "v"(a[i])
#define A2 "v"(a[i]), "v"(b[i])
#define A3 "v"(a[i]), "v"(b[i]), "v"(c[i])

KERNEL(k_pkaccum, 0x11223344u, "v_cvt_pkaccum_u8_f32 %0, %1, %2", A2)
KERNEL(k_madmk, 0xFFFF0000u, "v_madmk_f16 %0, %1, 0x4200, %2", A2)   // a * 3.0 + b
KERNEL(k_madak, 0xFFFF0000u, "v_madak_f16 %0, %1, %2, 0x4200", A2)   // a * b + 3.0
#ifdef VGPU_GFX950
KERNEL(k_pk_min3, 0, "v_pk_minimum3_f16 %0, %1, %2, %3", A3)
KERNEL(k_pk_max3, 0, "v_pk_maximum3_f16 %0, %1, %2, %3", A3)
KERNEL(k_dot2_bf16, 0x7777u, "v_dot2_f32_bf16 %0, %1, %2, %3", A3)
KERNEL(k_dot2c_bf16, 0, "v_dot2c_f32_bf16 %0, %1, %2", A2)
#endif
KERNEL(k_dot2_f16, 0x7777u, "v_dot2_f32_f16 %0, %1, %2, %3", A3)
KERNEL(k_dot2c_f16, 0, "v_dot2c_f32_f16 %0, %1, %2", A2)

// 16-bit VOP3 forms with OPSEL[3] clear and set. The destination starts as 0xA5A5A5A5.
#define OPSEL_PAIR(NAME, OP) \
  KERNEL(NAME##_lo, 0xA5A5A5A5u, OP " %0, %1, %2, %3", A3) \
  KERNEL(NAME##_hi, 0xA5A5A5A5u, OP " %0, %1, %2, %3 op_sel:[0,0,0,1]", A3)
OPSEL_PAIR(k_fma_f16, "v_fma_f16")
OPSEL_PAIR(k_mad_f16, "v_mad_f16")
OPSEL_PAIR(k_mad_u16, "v_mad_u16")
OPSEL_PAIR(k_mad_i16, "v_mad_i16")
OPSEL_PAIR(k_div_fixup_f16, "v_div_fixup_f16")
KERNEL(k_mad_legacy_f16, 0xA5A5A5A5u, "v_mad_legacy_f16 %0, %1, %2, %3", A3)
KERNEL(k_fma_legacy_f16, 0xA5A5A5A5u, "v_fma_legacy_f16 %0, %1, %2, %3", A3)
KERNEL(k_mad_legacy_u16, 0xA5A5A5A5u, "v_mad_legacy_u16 %0, %1, %2, %3", A3)
KERNEL(k_mad_legacy_i16, 0xA5A5A5A5u, "v_mad_legacy_i16 %0, %1, %2, %3", A3)
KERNEL(k_div_fixup_legacy_f16, 0xA5A5A5A5u, "v_div_fixup_legacy_f16 %0, %1, %2, %3", A3)


#ifdef VGPU_GFX950
// gfx950's scaled conversions of small floats to wider ones: each value widened exactly and scaled by 2^(e - 127), e the exponent field of the second
// source. The 2-element forms write a register pair (f32) or one register (f16, bf16); pk32 reads 32 six-bit values from six registers.
typedef uint32_t v6u __attribute__((ext_vector_type(6)));
typedef uint32_t v16u __attribute__((ext_vector_type(16)));
typedef uint32_t v32u __attribute__((ext_vector_type(32)));
#define WIDEN2(NAME, INSTR) \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, uint32_t* o) { \
    const int i = threadIdx.x; uint64_t r; asm volatile(INSTR : "=v"(r) : "v"(a[i]), "v"(b[i])); o[2 * i] = static_cast<uint32_t>(r); o[2 * i + 1] = static_cast<uint32_t>(r >> 32); }
#define WIDEN1(NAME, INSTR) \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, uint32_t* o) { \
    const int i = threadIdx.x; uint32_t r; asm volatile(INSTR : "=v"(r) : "v"(a[i]), "v"(b[i])); o[i] = r; }
WIDEN2(k_sc_pk_f32_fp8_0, "v_cvt_scalef32_pk_f32_fp8 %0, %1, %2")
WIDEN2(k_sc_pk_f32_fp8_1, "v_cvt_scalef32_pk_f32_fp8 %0, %1, %2 op_sel:[1,0,0]")
WIDEN2(k_sc_pk_f32_bf8_0, "v_cvt_scalef32_pk_f32_bf8 %0, %1, %2")
WIDEN2(k_sc_pk_f32_bf8_1, "v_cvt_scalef32_pk_f32_bf8 %0, %1, %2 op_sel:[1,0,0]")
WIDEN1(k_sc_f32_fp8_0, "v_cvt_scalef32_f32_fp8 %0, %1, %2")
WIDEN1(k_sc_f32_fp8_1, "v_cvt_scalef32_f32_fp8 %0, %1, %2 op_sel:[1,0,0]")
WIDEN1(k_sc_f32_fp8_2, "v_cvt_scalef32_f32_fp8 %0, %1, %2 op_sel:[0,1,0]")
WIDEN1(k_sc_f32_fp8_3, "v_cvt_scalef32_f32_fp8 %0, %1, %2 op_sel:[1,1,0]")
WIDEN1(k_sc_f32_bf8_2, "v_cvt_scalef32_f32_bf8 %0, %1, %2 op_sel:[0,1,0]")
WIDEN1(k_sc_pk_f16_fp8_1, "v_cvt_scalef32_pk_f16_fp8 %0, %1, %2 op_sel:[1,0,0]")
WIDEN1(k_sc_pk_f16_bf8_0, "v_cvt_scalef32_pk_f16_bf8 %0, %1, %2")
WIDEN1(k_sc_pk_bf16_fp8_0, "v_cvt_scalef32_pk_bf16_fp8 %0, %1, %2")
WIDEN1(k_sc_pk_bf16_bf8_1, "v_cvt_scalef32_pk_bf16_bf8 %0, %1, %2 op_sel:[1,0,0]")
WIDEN2(k_sc_pk_f32_fp4_0, "v_cvt_scalef32_pk_f32_fp4 %0, %1, %2")
WIDEN2(k_sc_pk_f32_fp4_2, "v_cvt_scalef32_pk_f32_fp4 %0, %1, %2 op_sel:[0,1,0]")
WIDEN1(k_sc_pk_f16_fp4_3, "v_cvt_scalef32_pk_f16_fp4 %0, %1, %2 op_sel:[1,1,0]")
WIDEN1(k_sc_pk_bf16_fp4_1, "v_cvt_scalef32_pk_bf16_fp4 %0, %1, %2 op_sel:[1,0,0]")
// An inline constant as the scale: 2.0's exponent field is 128, so the values are doubled.
WIDEN2(k_sc_pk_f32_fp8_two, "v_cvt_scalef32_pk_f32_fp8 %0, %1, 2.0")
#define WIDEN32(NAME, INSTR, OUT, N) \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, uint32_t* o) { \
    const int i = threadIdx.x; v6u in; for (int k = 0; k < 6; ++k) in[k] = a[6 * i + k]; \
    OUT r; asm volatile(INSTR : "=v"(r) : "v"(in), "v"(b[i])); for (int k = 0; k < N; ++k) o[N * i + k] = r[k]; }
WIDEN32(k_sc_pk32_f32_fp6, "v_cvt_scalef32_pk32_f32_fp6 %0, %1, %2", v32u, 32)
WIDEN32(k_sc_pk32_f32_bf6, "v_cvt_scalef32_pk32_f32_bf6 %0, %1, %2", v32u, 32)
WIDEN32(k_sc_pk32_f16_fp6, "v_cvt_scalef32_pk32_f16_fp6 %0, %1, %2", v16u, 16)
WIDEN32(k_sc_pk32_bf16_fp6, "v_cvt_scalef32_pk32_bf16_fp6 %0, %1, %2", v16u, 16)
WIDEN32(k_sc_pk32_f16_bf6, "v_cvt_scalef32_pk32_f16_bf6 %0, %1, %2", v16u, 16)
WIDEN32(k_sc_pk32_bf16_bf6, "v_cvt_scalef32_pk32_bf16_bf6 %0, %1, %2", v16u, 16)
#endif

// v_mfma_f64_4x4x4_4b_f64: four 4x4x4 products at once, in the guide's layout (7.1.4): A[b,i,k] in lane i + 4 * (b + 4 * k), B[b,k,j] in lane
// j + 4 * (b + 4 * k), and C and D[b,i,j] in lane j + 4 * (i + 4 * b).
__global__ void k_mfma_f64_4b(const double* a, const double* b, const double* c, double* d) {
  const int l = threadIdx.x;
  d[l] = __builtin_amdgcn_mfma_f64_4x4x4f64(a[l], b[l], c[l], 0, 0, 0);
}


// MODE.FP16_OVFL (bit 23) set around an instruction: "an overflowed FP16 result is clamped to +/- MAX_FP16, regardless of round mode, while still
// preserving true INF values", and the F32 to FP8/BF8 conversions' overflow rule (guides, 7.3). The destination starts as zero.
#define OVFL(NAME, SET, INSTR) \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, const uint32_t*, uint32_t* o) { \
    const int i = threadIdx.x; uint32_t r = 0; \
    asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), " #SET "\n " INSTR "\n s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0" : "+v"(r) : "v"(a[i]), "v"(b[i])); \
    o[i] = r; }
OVFL(k_add_f16_ovfl1, 1, "v_add_f16 %0, %1, %2")
OVFL(k_add_f16_ovfl0, 0, "v_add_f16 %0, %1, %2")
OVFL(k_pk_fp8_ovfl1, 1, "v_cvt_pk_fp8_f32 %0, %1, %2")
OVFL(k_pk_fp8_ovfl0, 0, "v_cvt_pk_fp8_f32 %0, %1, %2")
OVFL(k_pk_bf8_ovfl1, 1, "v_cvt_pk_bf8_f32 %0, %1, %2")
OVFL(k_pk_bf8_ovfl0, 0, "v_cvt_pk_bf8_f32 %0, %1, %2")
OVFL(k_sr_fp8_ovfl1, 1, "v_cvt_sr_fp8_f32 %0, %1, %2")
OVFL(k_sr_fp8_ovfl0, 0, "v_cvt_sr_fp8_f32 %0, %1, %2")

// The relative moves: the register number in the instruction plus M0. Lane 0 only; o[0..7] hold the results.
__global__ void k_movrel(uint32_t* o) {
  if (threadIdx.x != 0) return;
  uint32_t r1, r2;
  uint64_t r3, r4;
  // s_movrels_b32 s_dst, s7 with M0 = 10 reads s17 (the guide's example: "s5 <= s17").
  asm volatile("s_mov_b32 s17, %1\n s_mov_b32 m0, 10\n s_movrels_b32 %0, s7" : "=&s"(r1) : "s"(0x1234u) : "s17", "m0");
  // s_movreld_b32 s5, S0 with M0 = 10 writes s15 (the guide's example: "s15 <= s7").
  asm volatile("s_mov_b32 m0, 10\n s_movreld_b32 s5, %1\n s_mov_b32 %0, s15" : "=&s"(r2) : "s"(0xCAFE5678u) : "s15", "m0");
  // The 64-bit forms: s[8:9] with M0 = 4 reads s[12:13]; s[6:7] with M0 = 2 writes s[8:9].
  asm volatile("s_mov_b64 s[12:13], %1\n s_mov_b32 m0, 4\n s_movrels_b64 %0, s[8:9]" : "=&s"(r3) : "s"(uint64_t{0x1122334455667788ull}) : "s12", "s13", "m0");
  asm volatile("s_mov_b32 m0, 2\n s_movreld_b64 s[6:7], %1\n s_mov_b64 %0, s[8:9]" : "=&s"(r4) : "s"(uint64_t{0xAABBCCDD11223344ull}) : "s8", "s9", "m0");
  o[0] = r1, o[1] = r2;
  o[2] = static_cast<uint32_t>(r3), o[3] = static_cast<uint32_t>(r3 >> 32);
  o[4] = static_cast<uint32_t>(r4), o[5] = static_cast<uint32_t>(r4 >> 32);
}

static uint32_t fb(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint16_t h16(float f) { _Float16 x = static_cast<_Float16>(f); uint16_t b; std::memcpy(&b, &x, 2); return b; }
static uint32_t h(float f) { return h16(f); }

template <typename K>
static std::vector<uint32_t> run(K kernel, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const std::vector<uint32_t>& c) {
  uint32_t *da, *db, *dc, *dout;
  std::vector<uint32_t> out(64, 0xDEADBEEFu);
  hipMalloc(&da, 256), hipMalloc(&db, 256), hipMalloc(&dc, 256), hipMalloc(&dout, 256);
  hipMemcpy(da, a.data(), 256, hipMemcpyHostToDevice), hipMemcpy(db, b.data(), 256, hipMemcpyHostToDevice), hipMemcpy(dc, c.data(), 256, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(kernel, dim3(1), dim3(64), 0, 0, da, db, dc, dout);
  if (hipDeviceSynchronize() != hipSuccess) return out;
  hipMemcpy(out.data(), dout, 256, hipMemcpyDeviceToHost);
  hipFree(da), hipFree(db), hipFree(dc), hipFree(dout);
  return out;
}
// Cases (a, b, c) cycle over the 64 lanes; the wanted result of each case, compared under `mask`.
struct Case { uint32_t a, b, c, want; };
template <typename K>
static void cases(const char* what, K kernel, const std::vector<Case>& cs, uint32_t mask = 0xFFFFFFFFu) {
  std::vector<uint32_t> a(64), b(64), c(64);
  for (int l = 0; l < 64; ++l) a[l] = cs[l % cs.size()].a, b[l] = cs[l % cs.size()].b, c[l] = cs[l % cs.size()].c;
  const auto out = run(kernel, a, b, c);
  int wrong = 0;
  for (int l = 0; l < 64; ++l) {
    const uint32_t want = cs[l % cs.size()].want & mask;
    if ((out[l] & mask) != want) {
      if (wrong < 2) std::printf("  %s lane %d (case %d): got %08x want %08x\n", what, l, l % static_cast<int>(cs.size()), out[l] & mask, want);
      ++wrong;
    }
  }
  report(what, wrong, 64);
}
static bool is_nan_half(uint32_t v) { return (v & 0x7C00u) == 0x7C00u && (v & 0x3FFu) != 0; }


#ifdef VGPU_GFX950
// The small floats of the CDNA4 guide's Table 30, decoded on the host: sign, exponent (bias), mantissa; the least exponent is the denormal one.
static double small_value(uint32_t code, int ebits, int mbits, int bias, int kind) {   // kind: 0 plain (no NaN, no infinity), 1 E4M3, 2 E5M2
  const int sign = (code >> (ebits + mbits)) & 1, e = (code >> mbits) & ((1 << ebits) - 1), m = code & ((1 << mbits) - 1);
  if (kind == 1 && e == 15 && m == 7) return std::nan("");
  if (kind == 2 && e == 31) return m ? std::nan("") : (sign ? -INFINITY : INFINITY);
  const double mag = e == 0 ? std::ldexp(static_cast<double>(m), 1 - bias - mbits) : std::ldexp(static_cast<double>((1 << mbits) | m), e - bias - mbits);
  return sign ? -mag : mag;
}
static double fp8_value(uint32_t c) { return small_value(c & 0xFF, 4, 3, 7, 1); }
static double bf8_value(uint32_t c) { return small_value(c & 0xFF, 5, 2, 15, 2); }
static double fp6_value(uint32_t c) { return small_value(c & 0x3F, 2, 3, 1, 0); }
static double bf6_value(uint32_t c) { return small_value(c & 0x3F, 3, 2, 3, 0); }
static double fp4_value(uint32_t c) { return small_value(c & 0xF, 2, 1, 1, 0); }
static uint32_t f32_bits(double scaled) { const float f = static_cast<float>(scaled); uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint32_t f16_bits(double scaled) { return h16(static_cast<float>(scaled)); }
static uint32_t bf16_bits(double scaled) {   // round to nearest even from the float
  const uint32_t u = f32_bits(scaled);
  if (std::isnan(scaled)) return 0x7FC0;
  return (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
}
static bool nan_bits(uint32_t bits, int width) { return width == 32 ? (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x7FFFFFu) : width == 16 ? is_nan_half(bits) : (bits & 0x7F80u) == 0x7F80u && (bits & 0x7Fu); }
static uint32_t next_random() { static uint32_t x = 0x2545F491u; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }
// One scale per lane (the float whose exponent field it is): 1.0, 8, 2^-7, 2^-100 and a field of 255, repeating.
static uint32_t scale_for(int lane) { static const uint32_t e[5] = {127, 130, 120, 27, 255}; return e[lane % 5] << 23 | (lane % 3 ? 0x155555u : 0); }
static int scale_exp(int lane) { return static_cast<int>(scale_for(lane) >> 23) - 127; }

template <typename K>
static std::vector<uint32_t> run_words(K kernel, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, size_t out_words) {
  uint32_t *da, *db, *dout;
  std::vector<uint32_t> out(out_words, 0xDEADBEEFu);
  hipMalloc(&da, a.size() * 4), hipMalloc(&db, b.size() * 4), hipMalloc(&dout, out_words * 4);
  hipMemcpy(da, a.data(), a.size() * 4, hipMemcpyHostToDevice), hipMemcpy(db, b.data(), b.size() * 4, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(kernel, dim3(1), dim3(64), 0, 0, da, db, dout);
  if (hipDeviceSynchronize() != hipSuccess) return out;
  hipMemcpy(out.data(), dout, out_words * 4, hipMemcpyDeviceToHost);
  hipFree(da), hipFree(db), hipFree(dout);
  return out;
}
// n values a lane, each from `code(lane, i, words)` through `value`, scaled and encoded by `encode` (width bits); the kernel's output is compared lane by lane.
template <typename K, typename Code, typename Value, typename Encode>
static void widen_check(const char* what, K kernel, int src_words, int n, int width, Code code, Value value, Encode encode) {
  std::vector<uint32_t> a(64 * src_words), b(64);
  for (auto& x : a) x = next_random();
  for (int l = 0; l < 64; ++l) b[l] = scale_for(l);
  const int out_words = n * width / 32;
  const auto out = run_words(kernel, a, b, 64 * out_words);
  int wrong = 0;
  for (int l = 0; l < 64; ++l)
    for (int i = 0; i < n; ++i) {
      const double v = value(code(&a[l * src_words], i));
      const bool nan = std::isnan(v) || scale_exp(l) == 128;   // (the field 255: E8M0's NaN)
      const double scaled = nan ? std::nan("") : std::ldexp(v, scale_exp(l));
      const uint32_t want = encode(scaled);
      const size_t at = static_cast<size_t>(l) * n + i;
      const uint32_t got = width == 32 ? out[at] : (out[at / 2] >> (16 * (at % 2))) & 0xFFFFu;
      if (nan ? !nan_bits(got, width) : got != want) {
        if (wrong < 2) std::printf("  %s lane %d value %d: got %08x want %08x\n", what, l, i, got, want);
        ++wrong;
      }
    }
  report(what, wrong, 64 * n);
}
#endif

int main() {
  // v_cvt_pkaccum_u8_f32: the byte S1[1:0] names takes the float (whole numbers here, so the conversion's rounding does not matter); the
  // other bytes of the destination (0x11223344 to begin with) are kept, and S1's other bits are ignored.
  cases("v_cvt_pkaccum_u8_f32 (byte 0)", k_pkaccum, {{fb(3.0f), 0, 0, 0x11223303u}});
  cases("v_cvt_pkaccum_u8_f32 (byte 1)", k_pkaccum, {{fb(200.0f), 1, 0, 0x1122C844u}});
  cases("v_cvt_pkaccum_u8_f32 (byte 2)", k_pkaccum, {{fb(0.0f), 2, 0, 0x11003344u}});
  cases("v_cvt_pkaccum_u8_f32 (byte 3; S1's upper bits are ignored)", k_pkaccum, {{fb(7.0f), 0xFFFFFFFFu, 0, 0x07223344u}});
  // v_madmk_f16 = S0 * K + S1 and v_madak_f16 = S0 * S1 + K with K = 3.0, "D0 = { 16'0, tmp.f16 }".
  cases("v_madmk_f16 (a * 3 + b; the high half zeroed)", k_madmk, {{h(1.5f), h(2.0f), 0, h(6.5f)}, {h(-2.0f), h(1.0f), 0, h(-5.0f)}, {h(0.0f), h(4.0f), 0, h(4.0f)}});
  cases("v_madak_f16 (a * b + 3; the high half zeroed)", k_madak, {{h(1.5f), h(2.0f), 0, h(6.0f)}, {h(-2.0f), h(1.0f), 0, h(1.0f)}, {h(0.5f), h(0.5f), 0, h(3.25f)}});

  // The four-block double-precision matrix product, worked out on the host from the guide's layout. Every value is a small dyadic number, so any
  // order of summation is exact.
  {
    double A[4][4][4], B[4][4][4], C[4][4][4];
    std::vector<double> la(64), lb(64), lc(64), out(64, -1.0), want(64);
    for (int bk = 0; bk < 4; ++bk)
      for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y) {
          A[bk][x][y] = (bk + 1) * (x + 1) + 0.5 * y - 1.0;       // A[b][i][k]
          B[bk][x][y] = (x + 2) - 0.25 * y + bk;                  // B[b][k][j]
          C[bk][x][y] = 100.0 * bk + 10.0 * x + y;                // C[b][i][j]
        }
    for (int bk = 0; bk < 4; ++bk)
      for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y) {
          la[x + 4 * (bk + 4 * y)] = A[bk][x][y];   // lane i + 4 * (b + 4 * k)
          lb[y + 4 * (bk + 4 * x)] = B[bk][x][y];   // lane j + 4 * (b + 4 * k), x the row k and y the column j
          lc[y + 4 * (x + 4 * bk)] = C[bk][x][y];
          double sum = C[bk][x][y];
          for (int k = 0; k < 4; ++k) sum += A[bk][x][k] * B[bk][k][y];
          want[y + 4 * (x + 4 * bk)] = sum;
        }
    double *da, *db, *dc, *dd;
    hipMalloc(&da, 512), hipMalloc(&db, 512), hipMalloc(&dc, 512), hipMalloc(&dd, 512);
    hipMemcpy(da, la.data(), 512, hipMemcpyHostToDevice), hipMemcpy(db, lb.data(), 512, hipMemcpyHostToDevice), hipMemcpy(dc, lc.data(), 512, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_mfma_f64_4b, dim3(1), dim3(64), 0, 0, da, db, dc, dd);
    if (hipDeviceSynchronize() != hipSuccess) { std::printf("FAIL k_mfma_f64_4b\n"); return 1; }
    hipMemcpy(out.data(), dd, 512, hipMemcpyDeviceToHost);
    int wrong = 0;
    for (int l = 0; l < 64; ++l) wrong += out[l] != want[l];
    report("v_mfma_f64_4x4x4_4b_f64 (four 4x4x4 products, the guide's layout)", wrong, 64);
  }


#ifdef VGPU_GFX950
  {
    // 8-bit sources: the two bytes of the half OPSEL[0] names, or the byte OPSEL[1:0] names.
    const auto byte_of = [](int which) { return [which](const uint32_t* w, int) { return (w[0] >> (8 * which)) & 0xFF; }; };
    const auto pair_of = [](int half) { return [half](const uint32_t* w, int i) { return (w[0] >> (16 * half + 8 * i)) & 0xFF; }; };
    widen_check("v_cvt_scalef32_pk_f32_fp8 (low half)", k_sc_pk_f32_fp8_0, 1, 2, 32, pair_of(0), fp8_value, f32_bits);
    widen_check("v_cvt_scalef32_pk_f32_fp8 (high half, op_sel[0])", k_sc_pk_f32_fp8_1, 1, 2, 32, pair_of(1), fp8_value, f32_bits);
    widen_check("v_cvt_scalef32_pk_f32_bf8 (low half)", k_sc_pk_f32_bf8_0, 1, 2, 32, pair_of(0), bf8_value, f32_bits);
    widen_check("v_cvt_scalef32_pk_f32_bf8 (high half)", k_sc_pk_f32_bf8_1, 1, 2, 32, pair_of(1), bf8_value, f32_bits);
    widen_check("v_cvt_scalef32_f32_fp8 (byte 0)", k_sc_f32_fp8_0, 1, 1, 32, byte_of(0), fp8_value, f32_bits);
    widen_check("v_cvt_scalef32_f32_fp8 (byte 1)", k_sc_f32_fp8_1, 1, 1, 32, byte_of(1), fp8_value, f32_bits);
    widen_check("v_cvt_scalef32_f32_fp8 (byte 2)", k_sc_f32_fp8_2, 1, 1, 32, byte_of(2), fp8_value, f32_bits);
    widen_check("v_cvt_scalef32_f32_fp8 (byte 3)", k_sc_f32_fp8_3, 1, 1, 32, byte_of(3), fp8_value, f32_bits);
    widen_check("v_cvt_scalef32_f32_bf8 (byte 2)", k_sc_f32_bf8_2, 1, 1, 32, byte_of(2), bf8_value, f32_bits);
    widen_check("v_cvt_scalef32_pk_f16_fp8 (high half)", k_sc_pk_f16_fp8_1, 1, 2, 16, pair_of(1), fp8_value, f16_bits);
    widen_check("v_cvt_scalef32_pk_f16_bf8 (low half)", k_sc_pk_f16_bf8_0, 1, 2, 16, pair_of(0), bf8_value, f16_bits);
    widen_check("v_cvt_scalef32_pk_bf16_fp8 (low half)", k_sc_pk_bf16_fp8_0, 1, 2, 16, pair_of(0), fp8_value, bf16_bits);
    widen_check("v_cvt_scalef32_pk_bf16_bf8 (high half)", k_sc_pk_bf16_bf8_1, 1, 2, 16, pair_of(1), bf8_value, bf16_bits);
    // 4-bit sources: the byte OPSEL[1:0] names, low nibble first.
    const auto nib_of = [](int byte) { return [byte](const uint32_t* w, int i) { return (w[0] >> (8 * byte + 4 * i)) & 0xF; }; };
    widen_check("v_cvt_scalef32_pk_f32_fp4 (byte 0)", k_sc_pk_f32_fp4_0, 1, 2, 32, nib_of(0), fp4_value, f32_bits);
    widen_check("v_cvt_scalef32_pk_f32_fp4 (byte 2)", k_sc_pk_f32_fp4_2, 1, 2, 32, nib_of(2), fp4_value, f32_bits);
    widen_check("v_cvt_scalef32_pk_f16_fp4 (byte 3)", k_sc_pk_f16_fp4_3, 1, 2, 16, nib_of(3), fp4_value, f16_bits);
    widen_check("v_cvt_scalef32_pk_bf16_fp4 (byte 1)", k_sc_pk_bf16_fp4_1, 1, 2, 16, nib_of(1), fp4_value, bf16_bits);
    // 6-bit sources: 32 values of six bits, the first in the lowest bits of the first register.
    const auto six_of = [](const uint32_t* w, int i) {
      const int at = 6 * i;
      uint64_t v = w[at / 32];
      if (at % 32 + 6 > 32) v |= static_cast<uint64_t>(w[at / 32 + 1]) << 32;
      return static_cast<uint32_t>(v >> (at % 32)) & 0x3F;
    };
    widen_check("v_cvt_scalef32_pk32_f32_fp6", k_sc_pk32_f32_fp6, 6, 32, 32, six_of, fp6_value, f32_bits);
    widen_check("v_cvt_scalef32_pk32_f32_bf6", k_sc_pk32_f32_bf6, 6, 32, 32, six_of, bf6_value, f32_bits);
    widen_check("v_cvt_scalef32_pk32_f16_fp6", k_sc_pk32_f16_fp6, 6, 32, 16, six_of, fp6_value, f16_bits);
    widen_check("v_cvt_scalef32_pk32_bf16_fp6", k_sc_pk32_bf16_fp6, 6, 32, 16, six_of, fp6_value, bf16_bits);
    widen_check("v_cvt_scalef32_pk32_f16_bf6", k_sc_pk32_f16_bf6, 6, 32, 16, six_of, bf6_value, f16_bits);
    widen_check("v_cvt_scalef32_pk32_bf16_bf6", k_sc_pk32_bf16_bf6, 6, 32, 16, six_of, bf6_value, bf16_bits);
    // The scale as an inline constant (2.0: exponent field 128, scale 2^1); lane l's scale operand is ignored.
    {
      std::vector<uint32_t> a(64), b(64, 0);
      for (auto& x : a) x = next_random();
      const auto out = run_words(k_sc_pk_f32_fp8_two, a, b, 128);
      int wrong = 0;
      for (int l = 0; l < 64; ++l)
        for (int i = 0; i < 2; ++i) {
          const double v = fp8_value((a[l] >> (8 * i)) & 0xFF);
          if (std::isnan(v) ? !nan_bits(out[2 * l + i], 32) : out[2 * l + i] != f32_bits(2.0 * v)) ++wrong;
        }
      report("v_cvt_scalef32_pk_f32_fp8 with the inline constant 2.0 as the scale", wrong, 128);
    }
  }
#endif


  {
    // Two finite halves that overflow, and a true infinity.
    const uint32_t inf16 = 0x7C00u;
    cases("v_add_f16 overflowing, MODE.FP16_OVFL set: clamped to the largest half", k_add_f16_ovfl1,
          {{h(60000.0f), h(60000.0f), 0, 0x7BFFu}, {h(-60000.0f), h(-60000.0f), 0, 0xFBFFu}, {inf16, h(1.0f), 0, inf16}, {h(1.0f), h(2.0f), 0, h(3.0f)}}, 0xFFFF);
    cases("v_add_f16 overflowing, MODE.FP16_OVFL clear: an infinity", k_add_f16_ovfl0, {{h(60000.0f), h(60000.0f), 0, inf16}, {h(-60000.0f), h(-60000.0f), 0, 0xFC00u}}, 0xFFFF);
    // F32 to FP8/BF8 (the low byte of the pair; S1 is the same value). Table 7.3: Inf and a value past the format's largest are NaN (fp8) or Inf (bf8)
    // with FP16_OVFL clear, and +-max with it set; a NaN stays a NaN. gfx942's 8-bit floats are the FNUZ ones: largest 240
    // (0x7F) and 57344 (0x7F), their NaN 0x80, and no infinity.
    const uint32_t inf = fb(INFINITY), ninf = fb(-INFINITY), big = fb(1.0e9f), nbig = fb(-1.0e9f), nan = 0x7FC00000u;
#ifdef VGPU_GFX950
    cases("v_cvt_pk_fp8_f32 with FP16_OVFL set: +-max (448)", k_pk_fp8_ovfl1, {{inf, inf, 0, 0x7E}, {ninf, ninf, 0, 0xFE}, {big, big, 0, 0x7E}, {nbig, nbig, 0, 0xFE}, {fb(448.0f), fb(448.0f), 0, 0x7E}}, 0xFF);
    cases("v_cvt_pk_fp8_f32 with FP16_OVFL clear: NaN", k_pk_fp8_ovfl0, {{inf, inf, 0, 0x7F}, {ninf, ninf, 0, 0xFF}, {big, big, 0, 0x7F}, {nbig, nbig, 0, 0xFF}, {fb(448.0f), fb(448.0f), 0, 0x7E}}, 0xFF);
    cases("v_cvt_pk_bf8_f32 with FP16_OVFL set: +-max (57344)", k_pk_bf8_ovfl1, {{inf, inf, 0, 0x7B}, {ninf, ninf, 0, 0xFB}, {big, big, 0, 0x7B}, {fb(57344.0f), fb(57344.0f), 0, 0x7B}}, 0xFF);
    cases("v_cvt_pk_bf8_f32 with FP16_OVFL clear: +-Inf", k_pk_bf8_ovfl0, {{inf, inf, 0, 0x7C}, {ninf, ninf, 0, 0xFC}, {big, big, 0, 0x7C}, {fb(57344.0f), fb(57344.0f), 0, 0x7B}}, 0xFF);
    cases("v_cvt_sr_fp8_f32 with FP16_OVFL set: +-max", k_sr_fp8_ovfl1, {{inf, 0x12345678u, 0, 0x7E}, {nbig, 0x9abcdef0u, 0, 0xFE}}, 0xFF);
    cases("v_cvt_sr_fp8_f32 with FP16_OVFL clear: NaN", k_sr_fp8_ovfl0, {{inf, 0x12345678u, 0, 0x7F}, {nbig, 0x9abcdef0u, 0, 0xFF}}, 0xFF);
#else
    cases("v_cvt_pk_fp8_f32 with FP16_OVFL set: +-max (240)", k_pk_fp8_ovfl1, {{inf, inf, 0, 0x7F}, {ninf, ninf, 0, 0xFF}, {big, big, 0, 0x7F}, {nbig, nbig, 0, 0xFF}, {fb(240.0f), fb(240.0f), 0, 0x7F}}, 0xFF);
    cases("v_cvt_pk_fp8_f32 with FP16_OVFL clear: NaN", k_pk_fp8_ovfl0, {{inf, inf, 0, 0x80}, {ninf, ninf, 0, 0x80}, {big, big, 0, 0x80}, {nbig, nbig, 0, 0x80}, {fb(240.0f), fb(240.0f), 0, 0x7F}}, 0xFF);
    cases("v_cvt_pk_bf8_f32 with FP16_OVFL set: +-max (57344)", k_pk_bf8_ovfl1, {{inf, inf, 0, 0x7F}, {ninf, ninf, 0, 0xFF}, {big, big, 0, 0x7F}, {fb(57344.0f), fb(57344.0f), 0, 0x7F}}, 0xFF);
    cases("v_cvt_sr_fp8_f32 with FP16_OVFL set: +-max", k_sr_fp8_ovfl1, {{inf, 0x12345678u, 0, 0x7F}, {nbig, 0x9abcdef0u, 0, 0xFF}}, 0xFF);
    cases("v_cvt_sr_fp8_f32 with FP16_OVFL clear: NaN", k_sr_fp8_ovfl0, {{inf, 0x12345678u, 0, 0x80}, {nbig, 0x9abcdef0u, 0, 0x80}}, 0xFF);
#endif
#ifdef VGPU_GFX950
    cases("v_cvt_pk_fp8_f32 of a NaN is a NaN", k_pk_fp8_ovfl1, {{nan, nan, 0, 0x7F}}, 0x7F);
#else
    cases("v_cvt_pk_fp8_f32 of a NaN is a NaN", k_pk_fp8_ovfl1, {{nan, nan, 0, 0x80}}, 0xFF);
#endif
  }

  // Relative scalar moves.
  {
    uint32_t* d;
    std::vector<uint32_t> o(8, 0xDEADBEEFu);
    hipMalloc(&d, 32);
    hipMemcpy(d, o.data(), 32, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_movrel, dim3(1), dim3(64), 0, 0, d);
    if (hipDeviceSynchronize() != hipSuccess) { std::printf("FAIL k_movrel\n"); return 1; }
    hipMemcpy(o.data(), d, 32, hipMemcpyDeviceToHost);
    report("s_movrels_b32 (s7 + M0 = s17)", o[0] != 0x1234u, 1);
    report("s_movreld_b32 (s5 + M0 = s15)", o[1] != 0xCAFE5678u, 1);
    report("s_movrels_b64 (s[8:9] + 4 = s[12:13])", (o[2] != 0x55667788u) + (o[3] != 0x11223344u), 2);
    report("s_movreld_b64 (s[6:7] + 2 = s[8:9])", (o[4] != 0x11223344u) + (o[5] != 0xAABBCCDDu), 2);
  }

  // The float dot products flush denormal inputs and outputs whatever MODE says (guides, 6.4).
  // v_dot2c_f32_f16 adds into its destination (r starts at 0): a denormal half (0x0001 = 2^-24) times 1.0 contributes nothing; a normal pair does.
  cases("v_dot2c_f32_f16 flushes a denormal half input", k_dot2c_f16,
        {{0x0001u, h(1.0f), 0, 0}, {0x00010000u | h(2.0f), h(1.0f) | h(3.0f) << 16, 0, fb(2.0f)}});
  // The VOP3P form adds S2 (r starts as junk and is overwritten): a denormal addend is flushed (the result is 0, not 2^-127), and the half too.
  cases("v_dot2_f32_f16 flushes a denormal float addend and a denormal half", k_dot2_f16,
        {{0, 0, 0x00400000u, 0}, {0x0001u, h(1.0f), 0, 0}, {h(1.5f), h(2.0f), fb(1.0f), fb(4.0f)}});
#ifdef VGPU_GFX950
  // bfloat16: 0x0001 is 2^-133, a denormal; 0x1C80 is 2^-70, whose square, 2^-140, is a denormal float (the output is flushed).
  cases("v_dot2_f32_bf16 flushes a denormal bfloat16 input, a denormal addend and a denormal result", k_dot2_bf16,
        {{0x0001u, 0x7180u, 0, 0},                 // 2^-133 * 2^100
         {0x1C80u, 0x1C80u, 0, 0},                 // 2^-70 * 2^-70 = 2^-140
         {0, 0, 0x00400000u, 0},                   // 2^-127 alone
         {0x3F80u, 0x4000u, fb(1.0f), fb(3.0f)}});  // 1 * 2 + 1
  cases("v_dot2c_f32_bf16 flushes a denormal bfloat16 input and a denormal result", k_dot2c_bf16,
        {{0x0001u, 0x7180u, 0, 0}, {0x1C80u, 0x1C80u, 0, 0}, {0x3F80u, 0x4000u, 0, fb(2.0f)}});
#endif

  // OPSEL[3] on the 16-bit VOP3 forms. Clear: the result goes to the low half and the high half of the destination (0xA5A5 here) is kept;
  // set: the high half is written and the low one kept. The "legacy" forms zero the high half.
  const auto lo = [](uint32_t v) { return 0xA5A50000u | v; };
  const auto hi = [](uint32_t v) { return v << 16 | 0xA5A5u; };
  const std::vector<Case> fma_cases = {{h(2.0f), h(3.0f), h(1.0f), h(7.0f)}, {h(-1.5f), h(2.0f), h(0.5f), h(-2.5f)}};
  auto with = [](std::vector<Case> cs, auto f) { for (auto& x : cs) x.want = f(x.want); return cs; };
  cases("v_fma_f16 with OPSEL[3] clear keeps the high half", k_fma_f16_lo, with(fma_cases, lo));
  cases("v_fma_f16 with OPSEL[3] set writes the high half", k_fma_f16_hi, with(fma_cases, hi));
  cases("v_mad_f16 with OPSEL[3] clear keeps the high half", k_mad_f16_lo, with(fma_cases, lo));
  cases("v_mad_f16 with OPSEL[3] set writes the high half", k_mad_f16_hi, with(fma_cases, hi));
  const std::vector<Case> mad_cases = {{3, 4, 5, 17}, {1000, 7, 2, 7002}};
  cases("v_mad_u16 with OPSEL[3] clear keeps the high half", k_mad_u16_lo, with(mad_cases, lo));
  cases("v_mad_u16 with OPSEL[3] set writes the high half", k_mad_u16_hi, with(mad_cases, hi));
  const std::vector<Case> madi_cases = {{static_cast<uint32_t>(-3) & 0xFFFF, 4, 5, 0xFFF9u}, {10, 11, static_cast<uint32_t>(-110) & 0xFFFF, 0}};
  cases("v_mad_i16 with OPSEL[3] clear keeps the high half", k_mad_i16_lo, with(madi_cases, lo));
  cases("v_mad_i16 with OPSEL[3] set writes the high half", k_mad_i16_hi, with(madi_cases, hi));
  // v_div_fixup_f16 (S0 the quotient, S1 the denominator, S2 the numerator): the quotient's magnitude with the operands' sign.
  const std::vector<Case> fix_cases = {{h(2.0f), h(-2.0f), h(4.0f), h(-2.0f)}, {h(-3.0f), h(2.0f), h(6.0f), h(3.0f)}};
  cases("v_div_fixup_f16 with OPSEL[3] clear keeps the high half", k_div_fixup_f16_lo, with(fix_cases, lo));
  cases("v_div_fixup_f16 with OPSEL[3] set writes the high half", k_div_fixup_f16_hi, with(fix_cases, hi));
  const auto zero_hi = [](uint32_t v) { return v; };
  cases("v_mad_legacy_f16 zeroes the high half", k_mad_legacy_f16, with(fma_cases, zero_hi));
  cases("v_fma_legacy_f16 zeroes the high half", k_fma_legacy_f16, with(fma_cases, zero_hi));
  cases("v_mad_legacy_u16 zeroes the high half", k_mad_legacy_u16, with(mad_cases, zero_hi));
  cases("v_mad_legacy_i16 zeroes the high half", k_mad_legacy_i16, with(madi_cases, zero_hi));
  cases("v_div_fixup_legacy_f16 zeroes the high half", k_div_fixup_legacy_f16, with(fix_cases, zero_hi));

#ifdef VGPU_GFX950
  // v_pk_minimum3_f16 / v_pk_maximum3_f16: per half, the IEEE minimum() or maximum() of three (-0 below +0; a NaN is the result).
  {
    struct Half3 { uint16_t x, y, z; };
    const Half3 lo3[4] = {{h16(1.0f), h16(2.0f), h16(3.0f)}, {h16(0.0f), 0x8000, h16(0.0f)}, {h16(5.0f), 0x7E00, h16(2.0f)}, {h16(-4.0f), h16(0.5f), h16(9.0f)}};
    const Half3 hi3[4] = {{h16(-4.0f), h16(0.5f), h16(9.0f)}, {0x8000, h16(0.0f), 0x8000}, {h16(3.0f), h16(3.0f), h16(3.0f)}, {h16(1.0f), 0x7E00, h16(2.0f)}};
    // wanted (lo, hi) of the minimum and of the maximum; 0x7E00 stands for "a NaN".
    const uint16_t min_lo[4] = {h16(1.0f), 0x8000, 0x7E00, h16(-4.0f)}, min_hi[4] = {h16(-4.0f), 0x8000, h16(3.0f), 0x7E00};
    const uint16_t max_lo[4] = {h16(3.0f), h16(0.0f), 0x7E00, h16(9.0f)}, max_hi[4] = {h16(9.0f), h16(0.0f), h16(3.0f), 0x7E00};
    std::vector<uint32_t> a(64), b(64), c(64);
    for (int l = 0; l < 64; ++l) {
      const int k = l % 4;
      a[l] = lo3[k].x | hi3[k].x << 16, b[l] = lo3[k].y | hi3[k].y << 16, c[l] = lo3[k].z | hi3[k].z << 16;
    }
    for (int which = 0; which < 2; ++which) {
      const auto out = which == 0 ? run(k_pk_min3, a, b, c) : run(k_pk_max3, a, b, c);
      int wrong = 0;
      for (int l = 0; l < 64; ++l) {
        const int k = l % 4;
        const uint16_t wl = which == 0 ? min_lo[k] : max_lo[k], wh = which == 0 ? min_hi[k] : max_hi[k];
        const uint32_t gl = out[l] & 0xFFFF, gh = out[l] >> 16;
        wrong += (wl == 0x7E00 ? !is_nan_half(gl) : gl != wl) + (wh == 0x7E00 ? !is_nan_half(gh) : gh != wh);
      }
      report(which == 0 ? "v_pk_minimum3_f16" : "v_pk_maximum3_f16", wrong, 128);
    }
  }
#endif
  std::printf("gfx9_decode: %d failed\n", failed);
  return failed != 0;
}
