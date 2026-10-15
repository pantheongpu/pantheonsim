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

// v_mfma_f64_4x4x4_4b_f64: four 4x4x4 products at once, in the guide's layout (7.1.4): A[b,i,k] in lane i + 4 * (b + 4 * k), B[b,k,j] in lane
// j + 4 * (b + 4 * k), and C and D[b,i,j] in lane j + 4 * (i + 4 * b).
__global__ void k_mfma_f64_4b(const double* a, const double* b, const double* c, double* d) {
  const int l = threadIdx.x;
  d[l] = __builtin_amdgcn_mfma_f64_4x4x4f64(a[l], b[l], c[l], 0, 0, 0);
}

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
