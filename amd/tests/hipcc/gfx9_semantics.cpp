// What the CDNA3 and CDNA4 ISA guides say gfx942's and gfx950's vector and scalar ALU do, where the simulator did otherwise (gfx9_semantics.gfx942
// and .gfx950, wave64), each in inline assembly with the case worked out from the guide's pseudocode: the float minimum and maximum
// (a signed zero's order, a signaling NaN under MODE.IEEE), the output modifier (ignored when IEEE is on or output denormals are
// allowed), half-precision absolute values, v_div_fixup's sign, scalar maximum's SCC and the 32-bit absolute difference,
// v_lshl_add_u64's shift count, the sine and cosine of a huge argument, 32-bit literals in 64-bit signed operands, integer CLAMP
// (saturation), CLAMP on the float conversions, the second source's modifiers under DPP, and ttmp, SCC and VCCZ as sources.
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
#define V1(NAME, INSTR) \
  __global__ void NAME(const uint32_t* a, const uint32_t*, const uint32_t*, uint32_t* o) { \
    const int i = threadIdx.x; uint32_t r = 0; asm volatile(INSTR : "+v"(r) : "v"(a[i])); o[i] = r; }
#define V2(NAME, INSTR) \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, const uint32_t*, uint32_t* o) { \
    const int i = threadIdx.x; uint32_t r = 0; asm volatile(INSTR : "+v"(r) : "v"(a[i]), "v"(b[i])); o[i] = r; }
#define V3(NAME, INSTR) \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* o) { \
    const int i = threadIdx.x; uint32_t r = 0; asm volatile(INSTR : "+v"(r) : "v"(a[i]), "v"(b[i]), "v"(c[i])); o[i] = r; }

V2(k_min_f32, "v_min_f32 %0, %1, %2")
V2(k_max_f32, "v_max_f32 %0, %1, %2")
V2(k_min_f16, "v_min_f16 %0, %1, %2")
V2(k_max_f16, "v_max_f16 %0, %1, %2")
// IEEE mode off: a signaling NaN is no longer quieted, the number wins.
__global__ void k_min_f32_ieee0(const uint32_t* a, const uint32_t* b, const uint32_t*, uint32_t* o) {
  const int i = threadIdx.x; uint32_t r = 0;
  asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 9, 1), 0\n v_min_f32 %0, %1, %2\n s_setreg_imm32_b32 hwreg(HW_REG_MODE, 9, 1), 1" : "+v"(r) : "v"(a[i]), "v"(b[i]));
  o[i] = r;
}
// The output modifier: ignored by default (IEEE on); applied with IEEE off and output denormals off.
V2(k_add_mul4, "v_add_f32 %0, %1, %2 mul:4")
__global__ void k_add_mul4_flush(const uint32_t* a, const uint32_t* b, const uint32_t*, uint32_t* o) {
  const int i = threadIdx.x; uint32_t r = 0;
  asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 9, 1), 0\n s_setreg_imm32_b32 hwreg(HW_REG_MODE, 5, 1), 0\n v_add_f32 %0, %1, %2 mul:4\n"
               "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 9, 1), 1\n s_setreg_imm32_b32 hwreg(HW_REG_MODE, 5, 1), 1" : "+v"(r) : "v"(a[i]), "v"(b[i]));
  o[i] = r;
}
V2(k_add_f16_negabs, "v_add_f16 %0, -%1, |%2|")
V3(k_div_fixup, "v_div_fixup_f32 %0, %1, %2, %3")
V1(k_sin, "v_sin_f32 %0, %1")
V1(k_cos, "v_cos_f32 %0, %1")
V3(k_mad_u16_clamp, "v_mad_u16 %0, %1, %2, %3 clamp")
V3(k_mad_i16_clamp, "v_mad_i16 %0, %1, %2, %3 clamp")
V2(k_subrev_clamp, "v_subrev_u32 %0, %1, %2 clamp")
V1(k_cvt_f32_i32_clamp, "v_cvt_f32_i32 %0, %1 clamp")
V1(k_cvt_f32_u32_mul2, "v_cvt_f32_u32 %0, %1 mul:2")
V2(k_dpp_negsrc1, "v_add_f32_dpp %0, %1, -%2 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf")
V2(k_dpp_abssrc1, "v_mul_f32_dpp %0, %1, |%2| quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf")
// ttmp, SCC and VCCZ as sources (the trap registers are zero here; SCC is 1 after an equal compare; VCCZ is 1 with VCC zero).
__global__ void k_sources(const uint32_t* a, const uint32_t*, const uint32_t*, uint32_t* o) {
  const int i = threadIdx.x; uint32_t t = 7, s = 7, z = 7;
  asm volatile("v_mov_b32 %0, ttmp0\n v_add_u32 %0, ttmp1, %0\n s_cmp_eq_u32 %3, %3\n v_mov_b32 %1, src_scc\n s_mov_b64 vcc, 0\n v_mov_b32 %2, src_vccz"
               : "=&v"(t), "=&v"(s), "=&v"(z) : "s"(__builtin_amdgcn_readfirstlane(a[i])) : "scc", "vcc");
  o[i] = t | s << 8 | z << 16;
}
// v_cmp_gt_i64 with a 32-bit literal: sign-extended, so 0x80000001 is negative and not greater than 46.
__global__ void k_lit64(const uint32_t* a, const uint32_t*, const uint32_t*, uint32_t* o) {
  const int i = threadIdx.x; uint64_t m; uint64_t v = a[i];
  asm volatile("v_cmp_gt_i64_e32 vcc, 0x80000001, %1\n s_mov_b64 %0, vcc" : "=s"(m) : "v"(v) : "vcc");
  o[i] = static_cast<uint32_t>(m & 1);
}
__global__ void k_lshl_add_u64(const uint32_t* a, const uint32_t* b, const uint32_t*, uint32_t* o) {
  const int i = threadIdx.x; uint64_t r = 0, x = 0x100000001ull;
  asm volatile("v_lshl_add_u64 %0, %1, %2, %3" : "+v"(r) : "v"(x), "v"(b[i]), "v"(uint64_t{0x100}));
  o[i] = static_cast<uint32_t>(r);
}
__global__ void k_scalar(uint32_t* o) {
  if (threadIdx.x != 0) return;
  uint32_t c, r, d;
  asm volatile("s_max_i32 %1, %2, %2\n s_cselect_b32 %0, 1, 0" : "=&s"(c), "=&s"(r) : "s"(uint32_t{5}) : "scc");
  o[0] = c;
  asm volatile("s_max_u32 %1, %2, %2\n s_cselect_b32 %0, 1, 0" : "=&s"(c), "=&s"(r) : "s"(uint32_t{0}) : "scc");
  o[1] = c;
  asm volatile("s_absdiff_i32 %0, %1, %2" : "=s"(d) : "s"(uint32_t{1}), "s"(uint32_t{0x80000000u}));
  o[2] = d;
}
__global__ void k_min_f64(const double* a, const double* b, double* o) {
  const int i = threadIdx.x; double r = 0;
  asm volatile("v_min_f64 %0, %1, %2" : "+v"(r) : "v"(a[i]), "v"(b[i]));
  o[i] = r;
}

static uint32_t fb(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint16_t h(float f) { _Float16 x = static_cast<_Float16>(f); uint16_t b; std::memcpy(&b, &x, 2); return b; }

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

int main() {
  const uint32_t pz = 0, nz = 0x80000000u, snan = 0x7f800001u, qsnan = 0x7fc00001u;
  cases("v_min_f32 (-0 below +0; a signaling NaN quieted under IEEE)", k_min_f32,
        {{nz, pz, 0, nz}, {pz, nz, 0, nz}, {fb(1), fb(2), 0, fb(1)}, {pz, snan, 0, qsnan}, {snan, fb(1), 0, qsnan}, {0x7fc00000u, fb(3), 0, fb(3)}});
  cases("v_max_f32 (+0 above -0)", k_max_f32, {{pz, nz, 0, pz}, {nz, pz, 0, pz}, {fb(1), fb(2), 0, fb(2)}, {fb(1), snan, 0, qsnan}});
  cases("v_min_f32 with MODE.IEEE clear (the number wins over a signaling NaN)", k_min_f32_ieee0, {{pz, snan, 0, pz}, {snan, fb(1), 0, fb(1)}});
  cases("v_min_f16 (-0 below +0; a signaling NaN quieted)", k_min_f16, {{h(0.0f), 0x8000u, 0, 0x8000u}, {h(1.0f), 0x7c01u, 0, 0x7e01u}, {h(1.0f), h(2.0f), 0, h(1.0f)}}, 0xFFFF);
  cases("v_max_f16 (+0 above -0; a signaling NaN quieted)", k_max_f16, {{0x0000u, 0x8000u, 0, 0x0000u}, {h(1.0f), 0x7c01u, 0, 0x7e01u}}, 0xFFFF);
  {
    double *da, *db, *dout;
    std::vector<double> a(64), b(64), o(64);
    for (int l = 0; l < 64; ++l) a[l] = l % 2 ? -0.0 : 0.0, b[l] = l % 2 ? 0.0 : -0.0;
    hipMalloc(&da, 512), hipMalloc(&db, 512), hipMalloc(&dout, 512);
    hipMemcpy(da, a.data(), 512, hipMemcpyHostToDevice), hipMemcpy(db, b.data(), 512, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_min_f64, dim3(1), dim3(64), 0, 0, da, db, dout);
    if (hipDeviceSynchronize() != hipSuccess) return 1;
    hipMemcpy(o.data(), dout, 512, hipMemcpyDeviceToHost);
    int wrong = 0;
    for (int l = 0; l < 64; ++l) wrong += !std::signbit(o[l]) || o[l] != 0;
    report("v_min_f64 of +0 and -0 (-0)", wrong, 64);
  }
  cases("v_add_f32 mul:4 with IEEE on is not scaled", k_add_mul4, {{fb(1), fb(1), 0, fb(2)}});
  cases("v_add_f32 mul:4 with IEEE off and output denormals off is scaled", k_add_mul4_flush, {{fb(1), fb(1), 0, fb(8)}});
  cases("v_add_f16 -a,|b| of +0 and -0 (a +0: the absolute value clears -0)", k_add_f16_negabs, {{0x0000u, 0x8000u, 0, 0x0000u}}, 0xFFFF);
  cases("v_div_fixup_f32 (the quotient takes the operands' sign)", k_div_fixup, {{fb(-1), fb(1), fb(2), fb(1)}, {fb(1), fb(-2), fb(4), fb(-1)}});
  cases("v_sin_f32 of a huge float (a whole number of turns)", k_sin, {{0xff7fffffu, 0, 0, 0}, {0x4b000001u, 0, 0, 0}, {fb(0.25f), 0, 0, fb(1)}});
  cases("v_cos_f32 of a huge float", k_cos, {{0xff7fffffu, 0, 0, fb(1)}, {fb(0.5f), 0, 0, fb(-1)}});
  cases("v_mad_u16 clamp", k_mad_u16_clamp, {{0xff, 0xffff, 0xb42d, 0xffff}, {3, 4, 5, 17}}, 0xFFFF);
  cases("v_mad_i16 clamp", k_mad_i16_clamp, {{0x3788, 0x7f3c, 0x8830, 0x7fff}, {static_cast<uint32_t>(-3) & 0xFFFF, 4, 5, static_cast<uint32_t>(-7) & 0xFFFF}}, 0xFFFF);
  cases("v_subrev_u32 clamp", k_subrev_clamp, {{1, 0, 0, 0}, {1, 5, 0, 4}});
  cases("v_cvt_f32_i32 clamp", k_cvt_f32_i32_clamp, {{5, 0, 0, fb(1)}, {static_cast<uint32_t>(-5), 0, 0, 0}});
  cases("v_cvt_f32_u32 mul:2 with IEEE on is not scaled", k_cvt_f32_u32_mul2, {{5, 0, 0, fb(5)}});
  cases("v_add_f32 with DPP and -src1", k_dpp_negsrc1, {{fb(5), fb(2), 0, fb(3)}});
  cases("v_mul_f32 with DPP and |src1|", k_dpp_abssrc1, {{fb(5), fb(-2), 0, fb(10)}});
  cases("ttmp, src_scc and src_vccz as sources", k_sources, {{3, 0, 0, 0u | 1u << 8 | 1u << 16}}, 0xFFFFFF);
  cases("v_cmp_gt_i64 with the 32-bit literal 0x80000001 (sign-extended)", k_lit64, {{46, 0, 0, 0}, {static_cast<uint32_t>(-1), 0, 0, 0}});
  cases("v_lshl_add_u64 (a count over 4 is zero)", k_lshl_add_u64, {{0, 5, 0, 0x101}, {0, 3, 0, 0x108}, {0, 8, 0, 0x101}});
  {
    uint32_t* d;
    std::vector<uint32_t> o(8, 0xDEADBEEFu);
    hipMalloc(&d, 32);
    hipLaunchKernelGGL(k_scalar, dim3(1), dim3(64), 0, 0, d);
    if (hipDeviceSynchronize() != hipSuccess) return 1;
    hipMemcpy(o.data(), d, 32, hipMemcpyDeviceToHost);
    report("s_max_i32 / s_max_u32 set SCC on a tie", (o[0] != 1) + (o[1] != 1), 2);
    report("s_absdiff_i32 wraps in 32 bits (1 against 0x80000000)", o[2] != 0x7fffffffu, 1);
  }
  std::printf("gfx9_semantics: %d failed\n", failed);
  return failed != 0;
}
