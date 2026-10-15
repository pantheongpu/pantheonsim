// Instructions the CDNA3 and CDNA4 ISA guides list for gfx942 and gfx950 that the decoder did not know (gfx9_added.gfx942 and .gfx950,
// wave64), each written in inline assembly with a case worked out by hand from the guide: the rest of the EXEC-saving family, the 64-bit bit
// instructions, s_cmovk_i32 and s_flbit_i32_i64, v_sqrt_f64, the half-precision sine, cosine, frexp, ldexp and multiply-adds
// (gfx9's "legacy" and mac forms), the normalised and saturating conversions, v_cvt_rpi_i32_f32, the cube-map helpers, and gfx950's
// v_ashr_pk and the IEEE minimum and maximum of three.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#pragma clang diagnostic ignored "-Winline-asm"

typedef float v2f __attribute__((ext_vector_type(2)));
static int failed = 0;
static void report(const char* what, int wrong, int of) {
  std::printf("%s: %d of %d wrong\n", what, wrong, of);
  failed += wrong != 0;
}
#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

// One value in per lane, one out.
#define V1(NAME, INSTR) \
  __global__ void NAME(const uint32_t* a, const uint32_t*, const uint32_t*, uint32_t* o) { \
    const int i = threadIdx.x; uint32_t r = 0; asm volatile(INSTR : "+v"(r) : "v"(a[i])); o[i] = r; }
#define V2(NAME, INSTR) \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, const uint32_t*, uint32_t* o) { \
    const int i = threadIdx.x; uint32_t r = 0; asm volatile(INSTR : "+v"(r) : "v"(a[i]), "v"(b[i])); o[i] = r; }
#define V3(NAME, INSTR) \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* o) { \
    const int i = threadIdx.x; uint32_t r = 0; asm volatile(INSTR : "+v"(r) : "v"(a[i]), "v"(b[i]), "v"(c[i])); o[i] = r; }
// A destination that is also the accumulator: it starts as c.
#define VACC(NAME, INSTR) \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* o) { \
    const int i = threadIdx.x; uint32_t r = c[i]; asm volatile(INSTR : "+v"(r) : "v"(a[i]), "v"(b[i])); o[i] = r; }

V1(k_sin_f16, "v_sin_f16 %0, %1")
V1(k_cos_f16, "v_cos_f16 %0, %1")
V1(k_frexp_mant_f16, "v_frexp_mant_f16 %0, %1")
V1(k_frexp_exp_f16, "v_frexp_exp_i16_f16 %0, %1")
V1(k_norm_i16, "v_cvt_norm_i16_f16 %0, %1")
V1(k_norm_u16, "v_cvt_norm_u16_f16 %0, %1")
V1(k_sat_pk_u8, "v_sat_pk_u8_i16 %0, %1")
V1(k_rpi, "v_cvt_rpi_i32_f32 %0, %1")
V2(k_ldexp_f16, "v_ldexp_f16 %0, %1, %2")
VACC(k_mac_f16, "v_mac_f16 %0, %1, %2")
VACC(k_pk_fmac_f16, "v_pk_fmac_f16 %0, %1, %2")
V3(k_mad_legacy_f16, "v_mad_legacy_f16 %0, %1, %2, %3")
V3(k_fma_legacy_f16, "v_fma_legacy_f16 %0, %1, %2, %3")
V3(k_mad_legacy_i16, "v_mad_legacy_i16 %0, %1, %2, %3")
V3(k_cubeid, "v_cubeid_f32 %0, %1, %2, %3")
V3(k_cubema, "v_cubema_f32 %0, %1, %2, %3")
#ifdef VGPU_GFX950
V3(k_ashr_pk_i8, "v_ashr_pk_i8_i32 %0, %1, %2, %3")
V3(k_ashr_pk_u8, "v_ashr_pk_u8_i32 %0, %1, %2, %3")
V3(k_minimum3, "v_minimum3_f32 %0, %1, %2, %3")
// v_cvt_pk_fp8_f32 with OPSEL[3]: the two bytes go to the destination's high half, the low half kept.
__global__ void k_cvt_pk_fp8_hi(const uint32_t* a, const uint32_t* b, const uint32_t*, uint32_t* o) {
  const int i = threadIdx.x; uint32_t r = 0xA5A5A5A5u;
  asm volatile("v_cvt_pk_fp8_f32 %0, %1, %2 op_sel:[0,0,1]" : "+v"(r) : "v"(a[i]), "v"(b[i]));
  o[i] = r;
}
__global__ void k_cvt_pk_fp8_lo(const uint32_t* a, const uint32_t* b, const uint32_t*, uint32_t* o) {
  const int i = threadIdx.x; uint32_t r = 0xA5A5A5A5u;
  asm volatile("v_cvt_pk_fp8_f32 %0, %1, %2" : "+v"(r) : "v"(a[i]), "v"(b[i]));
  o[i] = r;
}
V3(k_maximum3, "v_maximum3_f32 %0, %1, %2, %3")
#endif
__global__ void k_sqrt_f64(const double* a, double* o) {
  const int i = threadIdx.x; double r = 0;
  asm volatile("v_sqrt_f64 %0, %1" : "+v"(r) : "v"(a[i]));
  o[i] = r;
}

// The scalar ones, from lane 0's values: o[k * 8 ...] holds up to eight words of case k.
__global__ void k_scalar(uint32_t* o) {
  if (threadIdx.x != 0) return;
  uint64_t x = 0xF0F0F0F0A5A5A5A5ull, y = 0x0123456789ABCDEFull, exec_in = 0xFFFFFFFFFFFFFFFFull;
  uint64_t r;
  uint32_t scc;
  uint64_t after;
  // Each EXEC-saving instruction: EXEC is made `y`, the instruction takes `x`; the new EXEC, the saved one and SCC come out.
#define SAVEEXEC(K, NAME)                                                                                                        \
  asm volatile("s_mov_b64 %3, exec\n s_mov_b64 exec, %4\n " NAME " %0, %5\n s_cselect_b32 %2, 1, 0\n s_mov_b64 %1, exec\n s_mov_b64 exec, %3" \
               : "=&s"(r), "=&s"(after), "=&s"(scc), "=&s"(exec_in) : "s"(y), "s"(x) : "scc");                                  \
  o[(K) * 8 + 0] = static_cast<uint32_t>(r), o[(K) * 8 + 1] = static_cast<uint32_t>(r >> 32);                                    \
  o[(K) * 8 + 2] = static_cast<uint32_t>(after), o[(K) * 8 + 3] = static_cast<uint32_t>(after >> 32), o[(K) * 8 + 4] = scc;
  SAVEEXEC(0, "s_nand_saveexec_b64")
  SAVEEXEC(1, "s_nor_saveexec_b64")
  SAVEEXEC(2, "s_xnor_saveexec_b64")
  SAVEEXEC(3, "s_orn2_saveexec_b64")
  SAVEEXEC(4, "s_andn1_saveexec_b64")
  SAVEEXEC(5, "s_orn1_saveexec_b64")
  SAVEEXEC(6, "s_andn1_wrexec_b64")
  SAVEEXEC(7, "s_andn2_wrexec_b64")
  // s_bitset0/1_b64 (the destination's bit, the rest kept), s_bitcmp0/1_b64, s_flbit_i32_i64, s_quadmask, s_bitreplicate.
  uint64_t d = x;
  asm volatile("s_bitset0_b64 %0, %1" : "+s"(d) : "s"(uint32_t{34}));
  o[8 * 8 + 0] = static_cast<uint32_t>(d), o[8 * 8 + 1] = static_cast<uint32_t>(d >> 32);
  d = x;
  asm volatile("s_bitset1_b64 %0, %1" : "+s"(d) : "s"(uint32_t{33}));
  o[9 * 8 + 0] = static_cast<uint32_t>(d), o[9 * 8 + 1] = static_cast<uint32_t>(d >> 32);
  uint32_t c0, c1;
  asm volatile("s_bitcmp0_b64 %2, %3\n s_cselect_b32 %0, 1, 0\n s_bitcmp1_b64 %2, %3\n s_cselect_b32 %1, 1, 0" : "=&s"(c0), "=&s"(c1) : "s"(y), "s"(uint32_t{40}) : "scc");
  o[10 * 8 + 0] = c0, o[10 * 8 + 1] = c1;
  uint32_t fl;
  asm volatile("s_flbit_i32_i64 %0, %1" : "=s"(fl) : "s"(uint64_t{0xFFFF000000000000ull}));
  o[11 * 8 + 0] = fl;
  asm volatile("s_flbit_i32_i64 %0, %1" : "=s"(fl) : "s"(uint64_t{0}));
  o[11 * 8 + 1] = fl;
  uint32_t q32;
  asm volatile("s_quadmask_b32 %0, %1" : "=s"(q32) : "s"(uint32_t{0x00F0010Fu}));
  o[12 * 8 + 0] = q32;
  uint64_t q64;
  asm volatile("s_quadmask_b64 %0, %1" : "=s"(q64) : "s"(uint64_t{0x8000000000000001ull}));
  o[12 * 8 + 1] = static_cast<uint32_t>(q64), o[12 * 8 + 2] = static_cast<uint32_t>(q64 >> 32);
  uint64_t br;
  asm volatile("s_bitreplicate_b64_b32 %0, %1" : "=s"(br) : "s"(uint32_t{0x80000005u}));
  o[13 * 8 + 0] = static_cast<uint32_t>(br), o[13 * 8 + 1] = static_cast<uint32_t>(br >> 32);
  // s_cmovk_i32: moves the sign-extended 16-bit constant where SCC is set.
  uint32_t m1 = 0x1111, m2 = 0x2222;
  asm volatile("s_cmp_eq_u32 %2, %2\n s_cmovk_i32 %0, 0x8001\n s_cmp_lg_u32 %2, %2\n s_cmovk_i32 %1, 0x8001" : "+s"(m1), "+s"(m2) : "s"(uint32_t{7}) : "scc");
  o[14 * 8 + 0] = m1, o[14 * 8 + 1] = m2;
}

static uint16_t h(float f) { _Float16 x = static_cast<_Float16>(f); uint16_t b; std::memcpy(&b, &x, 2); return b; }
static float hf(uint16_t b) { _Float16 x; std::memcpy(&x, &b, 2); return static_cast<float>(x); }
static uint32_t fb(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

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
// Lane l of a table of cases: wants[l % n]; the low `bits` of the result are compared.
static void check(const char* what, const std::vector<uint32_t>& out, const std::vector<uint32_t>& want, uint32_t mask = 0xFFFFFFFFu) {
  int wrong = 0;
  for (int l = 0; l < 64; ++l) {
    const uint32_t w = want[l % want.size()] & mask;
    if ((out[l] & mask) != w) {
      if (wrong < 2) std::printf("  %s lane %d: got %08x want %08x\n", what, l, out[l] & mask, w);
      ++wrong;
    }
  }
  report(what, wrong, 64);
}
template <typename K>
static void unary(const char* what, K kernel, const std::vector<uint32_t>& in, const std::vector<uint32_t>& want, uint32_t mask) {
  std::vector<uint32_t> a(64), z(64, 0);
  for (int l = 0; l < 64; ++l) a[l] = in[l % in.size()];
  check(what, run(kernel, a, z, z), want, mask);
}

int main() {
  // Half-precision transcendentals and frexp: x = 0.25 turns is the top of the sine, 0.5 the bottom of the cosine.
  unary("v_sin_f16", k_sin_f16, {h(0.25f), h(0.0f), h(0.5f), h(-0.25f)}, {h(1.0f), h(0.0f), h(0.0f), h(-1.0f)}, 0xFFFF);
  unary("v_cos_f16", k_cos_f16, {h(0.5f), h(0.0f), h(0.25f)}, {h(-1.0f), h(1.0f), h(0.0f)}, 0xFFFF);
  unary("v_frexp_mant_f16", k_frexp_mant_f16, {h(8.0f), h(-3.0f), h(0.0f)}, {h(0.5f), h(-0.75f), h(0.0f)}, 0xFFFF);
  unary("v_frexp_exp_i16_f16", k_frexp_exp_f16, {h(8.0f), h(-3.0f), h(0.0f)}, {4, 2, 0}, 0xFFFF);
  unary("v_cvt_norm_i16_f16", k_norm_i16, {h(1.0f), h(-1.0f), h(0.0f), h(2.0f)}, {0x7FFF, 0x8001, 0, 0x7FFF}, 0xFFFF);
  unary("v_cvt_norm_u16_f16", k_norm_u16, {h(1.0f), h(0.0f), h(-1.0f), h(0.5f)}, {0xFFFF, 0, 0, 0x8000}, 0xFFFF);
  unary("v_sat_pk_u8_i16", k_sat_pk_u8, {0xFFFB012Cu, 0x00640064u}, {0x00FFu, 0x6464u}, 0xFFFF);
  unary("v_cvt_rpi_i32_f32", k_rpi, {fb(2.5f), fb(-2.5f), fb(1e10f), fb(std::nanf("")), fb(0.49f)}, {3u, 0xFFFFFFFEu, 0x7FFFFFFFu, 0u, 0u}, 0xFFFFFFFFu);
  {
    std::vector<uint32_t> a(64), b(64), z(64, 0);
    for (int l = 0; l < 64; ++l) a[l] = h(1.5f * (1 + l % 4)), b[l] = l % 5;   // ldexp: a * 2^b (b an integer in the half's place)
    std::vector<uint32_t> want(64);
    for (int l = 0; l < 64; ++l) want[l] = h(std::ldexp(hf(a[l]), static_cast<int>(b[l])));
    check("v_ldexp_f16", run(k_ldexp_f16, a, b, z), want, 0xFFFF);
    std::vector<uint32_t> c(64);
    for (int l = 0; l < 64; ++l) a[l] = h(0.5f + l % 3), b[l] = h(2.0f + l % 5), c[l] = h(0.25f * (l % 7));
    for (int l = 0; l < 64; ++l) want[l] = h(std::fma(hf(a[l]), hf(b[l]), hf(c[l])));
    check("v_mac_f16 (the destination is the addend)", run(k_mac_f16, a, b, c), want, 0xFFFF);
    check("v_mad_legacy_f16", run(k_mad_legacy_f16, a, b, c), want, 0xFFFF);
    check("v_fma_legacy_f16", run(k_fma_legacy_f16, a, b, c), want, 0xFFFF);
    for (int l = 0; l < 64; ++l) a[l] = h(1.5f + l % 3) | h(0.5f + l % 2) << 16, b[l] = h(2.0f) | h(3.0f) << 16, c[l] = h(1.0f) | h(0.25f * (l % 4)) << 16;
    for (int l = 0; l < 64; ++l)
      want[l] = h(std::fma(hf(a[l] & 0xFFFF), hf(b[l] & 0xFFFF), hf(c[l] & 0xFFFF))) | static_cast<uint32_t>(h(std::fma(hf(a[l] >> 16), hf(b[l] >> 16), hf(c[l] >> 16)))) << 16;
    check("v_pk_fmac_f16", run(k_pk_fmac_f16, a, b, c), want);
    for (int l = 0; l < 64; ++l) a[l] = (l % 7) * 11 - 20, b[l] = (l % 5) * 13 - 25, c[l] = (l % 3) * 1000 - 1000;
    for (int l = 0; l < 64; ++l) want[l] = static_cast<uint16_t>(static_cast<int16_t>(a[l]) * static_cast<int16_t>(b[l]) + static_cast<int16_t>(c[l]));
    check("v_mad_legacy_i16", run(k_mad_legacy_i16, a, b, c), want, 0xFFFF);
  }
  {
    // The cube map face: the major axis and its sign (+x 0, -x 1, +y 2, -y 3, +z 4, -z 5); and 2 * the major axis (cubema).
    std::vector<uint32_t> a, b, c, want_id, want_ma;
    const float v[6][3] = {{3, 1, 2}, {-3, 1, 2}, {1, 3, 2}, {1, -3, 2}, {1, 2, 3}, {1, 2, -3}};
    for (int l = 0; l < 6; ++l) {
      a.push_back(fb(v[l][0])), b.push_back(fb(v[l][1])), c.push_back(fb(v[l][2]));
      want_id.push_back(fb(static_cast<float>(l)));
      want_ma.push_back(fb(2.0f * v[l][l / 2]));   // twice the major axis, signed
    }
    a.resize(64, fb(3)), b.resize(64, fb(1)), c.resize(64, fb(2));
    std::vector<uint32_t> out = run(k_cubeid, a, b, c);
    int wrong = 0;
    for (int l = 0; l < 6; ++l) wrong += out[l] != want_id[l];
    report("v_cubeid_f32", wrong, 6);
    out = run(k_cubema, a, b, c);
    wrong = 0;
    for (int l = 0; l < 6; ++l) wrong += out[l] != want_ma[l];
    report("v_cubema_f32", wrong, 6);
  }
  {
    double *da, *dout;
    std::vector<double> in(64), out(64, -1);
    for (int l = 0; l < 64; ++l) in[l] = l * l * 0.25;
    hipMalloc(&da, 512), hipMalloc(&dout, 512);
    hipMemcpy(da, in.data(), 512, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_sqrt_f64, dim3(1), dim3(64), 0, 0, da, dout);
    if (hipDeviceSynchronize() != hipSuccess) return 1;
    hipMemcpy(out.data(), dout, 512, hipMemcpyDeviceToHost);
    int wrong = 0;
    for (int l = 0; l < 64; ++l) wrong += out[l] != std::sqrt(in[l]);
    report("v_sqrt_f64", wrong, 64);
  }
#ifdef VGPU_GFX950
  {
    std::vector<uint32_t> a(64), b(64), c(64);
    for (int l = 0; l < 64; ++l) a[l] = (l * 977) ^ 0x12345, b[l] = (l * 1237) ^ 0xFFFF0000u, c[l] = l % 20;
    const auto sat8 = [](int64_t n, bool is_signed) { return is_signed ? static_cast<uint8_t>(std::max<int64_t>(-128, std::min<int64_t>(127, n))) : static_cast<uint8_t>(std::max<int64_t>(0, std::min<int64_t>(255, n))); };
    std::vector<uint32_t> w_i(64), w_u(64);
    for (int l = 0; l < 64; ++l) {
      const int n = c[l] & 31;
      w_i[l] = sat8(static_cast<int32_t>(a[l]) >> n, true) | sat8(static_cast<int32_t>(b[l]) >> n, true) << 8;
      w_u[l] = sat8(static_cast<int32_t>(a[l]) >> n, false) | sat8(static_cast<int32_t>(b[l]) >> n, false) << 8;
    }
    check("v_ashr_pk_i8_i32", run(k_ashr_pk_i8, a, b, c), w_i, 0xFFFF);
    check("v_ashr_pk_u8_i32", run(k_ashr_pk_u8, a, b, c), w_u, 0xFFFF);
    // The 8-bit conversions' destination half (CDNA4 guide: "store the result into 16 bits of a vector register using OPSEL").
    for (int l = 0; l < 64; ++l) a[l] = fb(-5.25f), b[l] = fb(-0.0f);
    {
      const auto hi = run(k_cvt_pk_fp8_hi, a, b, c), lo = run(k_cvt_pk_fp8_lo, a, b, c);
      int wrong = 0;
      for (int l = 0; l < 64; ++l) wrong += (hi[l] & 0xFFFFu) != 0xA5A5u || (hi[l] >> 16) != (lo[l] & 0xFFFFu) || (lo[l] >> 16) != 0xA5A5u;
      report("v_cvt_pk_fp8_f32 op_sel:[0,0,1] (the high half) against the low half", wrong, 64);
    }
    // IEEE minimum and maximum of three: -0 below +0, and a NaN wins.
    const float t[4][3] = {{1, 2, 3}, {0.0f, -0.0f, 1}, {-0.0f, 0.0f, -1.0f}, {5, std::nanf(""), 2}};
    for (int l = 0; l < 64; ++l) a[l] = fb(t[l % 4][0]), b[l] = fb(t[l % 4][1]), c[l] = fb(t[l % 4][2]);
    const auto out_min = run(k_minimum3, a, b, c), out_max = run(k_maximum3, a, b, c);
    const uint32_t want_min[4] = {fb(1), fb(-0.0f), fb(-1), 0x7FC00000u}, want_max[4] = {fb(3), fb(1), fb(0.0f), 0x7FC00000u};
    int w1 = 0, w2 = 0;
    for (int l = 0; l < 64; ++l) {
      if (l % 4 == 3) { w1 += !std::isnan(*reinterpret_cast<const float*>(&out_min[l])); w2 += !std::isnan(*reinterpret_cast<const float*>(&out_max[l])); }
      else { w1 += out_min[l] != want_min[l % 4]; w2 += out_max[l] != want_max[l % 4]; }
    }
    report("v_minimum3_f32", w1, 64);
    report("v_maximum3_f32", w2, 64);
  }
#endif
  // The scalar instructions.
  {
    uint32_t* d;
    std::vector<uint32_t> o(16 * 8, 0xDEADBEEFu);
    hipMalloc(&d, 512);
    hipMemcpy(d, o.data(), 512, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_scalar, dim3(1), dim3(64), 0, 0, d);
    if (hipDeviceSynchronize() != hipSuccess) { std::printf("FAIL scalar kernel\n"); return 1; }
    hipMemcpy(o.data(), d, 512, hipMemcpyDeviceToHost);
    const uint64_t x = 0xF0F0F0F0A5A5A5A5ull, y = 0x0123456789ABCDEFull;
    const uint64_t news[8] = {~(x & y), ~(x | y), ~(x ^ y), x | ~y, ~x & y, ~x | y, ~x & y, x & ~y};
    static const char* names[8] = {"s_nand_saveexec_b64", "s_nor_saveexec_b64", "s_xnor_saveexec_b64", "s_orn2_saveexec_b64",
                                   "s_andn1_saveexec_b64", "s_orn1_saveexec_b64", "s_andn1_wrexec_b64", "s_andn2_wrexec_b64"};
    for (int k = 0; k < 8; ++k) {
      const uint64_t r = o[k * 8] | static_cast<uint64_t>(o[k * 8 + 1]) << 32, after = o[k * 8 + 2] | static_cast<uint64_t>(o[k * 8 + 3]) << 32;
      const bool wr = k >= 6;
      // saveexec: the destination is the old EXEC (y), EXEC the new value; wrexec: both are the new value.
      const int wrong = (r != (wr ? news[k] : y)) + (after != news[k]) + (o[k * 8 + 4] != (news[k] != 0));
      report(names[k], wrong, 3);
    }
    report("s_bitset0_b64 (bit 34)", (o[64] != static_cast<uint32_t>(x)) + (o[65] != static_cast<uint32_t>((x & ~(1ull << 34)) >> 32)), 2);
    report("s_bitset1_b64 (bit 33)", (o[72] != static_cast<uint32_t>(x)) + (o[73] != static_cast<uint32_t>((x | (1ull << 33)) >> 32)), 2);
    report("s_bitcmp0_b64 / s_bitcmp1_b64 (bit 40)", (o[80] != !((y >> 40) & 1)) + (o[81] != ((y >> 40) & 1)), 2);
    report("s_flbit_i32_i64 (16 leading ones, then all zeroes)", (o[88] != 16u) + (o[89] != 0xFFFFFFFFu), 2);
    report("s_quadmask_b32 / _b64", (o[96] != 0x25u) + (o[97] != 0x8001u) + (o[98] != 0u), 3);
    uint64_t br = 0;
    for (int i = 0; i < 32; ++i) br |= (uint64_t{3} * ((0x80000005u >> i) & 1)) << (2 * i);
    report("s_bitreplicate_b64_b32", (o[104] != static_cast<uint32_t>(br)) + (o[105] != static_cast<uint32_t>(br >> 32)), 2);
    report("s_cmovk_i32 (moves with SCC set, not without)", (o[112] != 0xFFFF8001u) + (o[113] != 0x2222u), 2);
  }
  std::printf("gfx9_added: %d failed\n", failed);
  return failed != 0;
}
