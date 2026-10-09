// gfx1250 (CDNA 5, MI455X) arithmetic that the compiler seldom emits, each instruction written in inline assembly and
// checked against the host (alu1250.gfx1250, wave32). The oracle is AMD's CDNA5 ISA pseudocode (sections 15.3, 15.7, 15.10,
// 15.14), written out again here for the host: the IEEE minimum() and maximum() families, the "_num" three-operand forms,
// 16-bit integer and half-precision operations, packed bfloat16, half and 16-bit integer operations, the DX9 multiplies,
// the scalar float and bit operations, and the EXEC-writing scalar operations.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#pragma clang diagnostic ignored "-Winline-asm"

// The scalar instructions take their operands from scalar registers: a load through a pointer the compiler cannot prove
// constant arrives in a vector register, so the value is read first from the wave's first lane.
#define U(x) static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(x)))
#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

// --- kernels: a, b, c are per-lane inputs; o the output. -----------------------------------------------------------
#define V1(NAME, INSTR)                                                                  \
  __global__ void NAME(const uint32_t* a, const uint32_t*, const uint32_t*, uint32_t* o) { \
    const int i = blockIdx.x * 32 + threadIdx.x;                                         \
    uint32_t r = 0;                                                                      \
    asm volatile(INSTR : "+v"(r) : "v"(a[i]));                                           \
    o[i] = r;                                                                            \
  }
#define V2(NAME, INSTR)                                                                  \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, const uint32_t*, uint32_t* o) { \
    const int i = blockIdx.x * 32 + threadIdx.x;                                         \
    uint32_t r = 0;                                                                      \
    asm volatile(INSTR : "+v"(r) : "v"(a[i]), "v"(b[i]));                                \
    o[i] = r;                                                                            \
  }
#define V3(NAME, INSTR)                                                                  \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* o) { \
    const int i = blockIdx.x * 32 + threadIdx.x;                                         \
    uint32_t r = 0;                                                                      \
    asm volatile(INSTR : "+v"(r) : "v"(a[i]), "v"(b[i]), "v"(c[i]));                     \
    o[i] = r;                                                                            \
  }
// A destination that is also the accumulator (v_pk_fmac_f16): it starts as c.
#define VACC(NAME, INSTR)                                                                \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* o) { \
    const int i = blockIdx.x * 32 + threadIdx.x;                                         \
    uint32_t r = c[i];                                                                   \
    asm volatile(INSTR : "+v"(r) : "v"(a[i]), "v"(b[i]));                                \
    o[i] = r;                                                                            \
  }

V2(k_minimum_f32, "v_minimum_f32 %0, %1, %2")
V2(k_maximum_f32, "v_maximum_f32 %0, %1, %2")
V3(k_minimum3_f32, "v_minimum3_f32 %0, %1, %2, %3")
V3(k_maximum3_f32, "v_maximum3_f32 %0, %1, %2, %3")
V3(k_minimummaximum_f32, "v_minimummaximum_f32 %0, %1, %2, %3")
V3(k_maximumminimum_f32, "v_maximumminimum_f32 %0, %1, %2, %3")
V3(k_min3_num_f32, "v_min3_num_f32 %0, %1, %2, %3")
V3(k_max3_num_f32, "v_max3_num_f32 %0, %1, %2, %3")
V3(k_minmax_num_f32, "v_minmax_num_f32 %0, %1, %2, %3")
V3(k_maxmin_num_f32, "v_maxmin_num_f32 %0, %1, %2, %3")
V3(k_med3_num_f32, "v_med3_num_f32 %0, %1, %2, %3")
V2(k_minimum_f16, "v_minimum_f16 %0, %1, %2")
V2(k_maximum_f16, "v_maximum_f16 %0, %1, %2")
V3(k_minimum3_f16, "v_minimum3_f16 %0, %1, %2, %3")
V3(k_maximum3_f16, "v_maximum3_f16 %0, %1, %2, %3")
V3(k_minimummaximum_f16, "v_minimummaximum_f16 %0, %1, %2, %3")
V3(k_maximumminimum_f16, "v_maximumminimum_f16 %0, %1, %2, %3")
V3(k_min3_num_f16, "v_min3_num_f16 %0, %1, %2, %3")
V3(k_max3_num_f16, "v_max3_num_f16 %0, %1, %2, %3")
V3(k_minmax_num_f16, "v_minmax_num_f16 %0, %1, %2, %3")
V3(k_maxmin_num_f16, "v_maxmin_num_f16 %0, %1, %2, %3")
V3(k_med3_num_f16, "v_med3_num_f16 %0, %1, %2, %3")
V2(k_add_nc_i16, "v_add_nc_i16 %0, %1, %2")
V2(k_sub_nc_i16, "v_sub_nc_i16 %0, %1, %2")
V2(k_max_i16, "v_max_i16 %0, %1, %2")
V2(k_min_i16, "v_min_i16 %0, %1, %2")
V2(k_ashrrev_i16, "v_ashrrev_i16 %0, %1, %2")
V1(k_cvt_i32_i16, "v_cvt_i32_i16 %0, %1")
V1(k_cvt_u32_u16, "v_cvt_u32_u16 %0, %1")
V1(k_sat_pk_u8_i16, "v_sat_pk_u8_i16 %0, %1")
V1(k_sin_f16, "v_sin_f16 %0, %1")
V1(k_cos_f16, "v_cos_f16 %0, %1")
V1(k_frexp_mant_f16, "v_frexp_mant_f16 %0, %1")
V1(k_frexp_exp_i16_f16, "v_frexp_exp_i16_f16 %0, %1")
V2(k_ldexp_f16, "v_ldexp_f16 %0, %1, %2")
V1(k_cvt_nearest_i32_f32, "v_cvt_nearest_i32_f32 %0, %1")
V1(k_cvt_off_f32_i4, "v_cvt_off_f32_i4 %0, %1")
V1(k_cvt_norm_i16_f16, "v_cvt_norm_i16_f16 %0, %1")
V1(k_cvt_norm_u16_f16, "v_cvt_norm_u16_f16 %0, %1")
V2(k_mul_dx9_zero_f32, "v_mul_dx9_zero_f32 %0, %1, %2")
V3(k_fma_dx9_zero_f32, "v_fma_dx9_zero_f32 %0, %1, %2, %3")
V3(k_mullit_f32, "v_mullit_f32 %0, %1, %2, %3")
V2(k_pk_add_bf16, "v_pk_add_bf16 %0, %1, %2")
V2(k_pk_mul_bf16, "v_pk_mul_bf16 %0, %1, %2")
V3(k_pk_fma_bf16, "v_pk_fma_bf16 %0, %1, %2, %3")
V2(k_pk_min_num_bf16, "v_pk_min_num_bf16 %0, %1, %2")
V2(k_pk_max_num_bf16, "v_pk_max_num_bf16 %0, %1, %2")
VACC(k_pk_fmac_f16, "v_pk_fmac_f16 %0, %1, %2")
V2(k_pk_minimum_f16, "v_pk_minimum_f16 %0, %1, %2")
V2(k_pk_maximum_f16, "v_pk_maximum_f16 %0, %1, %2")
V3(k_pk_minimum3_f16, "v_pk_minimum3_f16 %0, %1, %2, %3")
V3(k_pk_maximum3_f16, "v_pk_maximum3_f16 %0, %1, %2, %3")
V3(k_pk_min3_num_f16, "v_pk_min3_num_f16 %0, %1, %2, %3")
V3(k_pk_max3_num_f16, "v_pk_max3_num_f16 %0, %1, %2, %3")
V3(k_pk_add_max_i16, "v_pk_add_max_i16 %0, %1, %2, %3")
V3(k_pk_add_max_u16, "v_pk_add_max_u16 %0, %1, %2, %3")
V3(k_pk_add_min_i16, "v_pk_add_min_i16 %0, %1, %2, %3")
V3(k_pk_add_min_u16, "v_pk_add_min_u16 %0, %1, %2, %3")
V3(k_pk_max3_i16, "v_pk_max3_i16 %0, %1, %2, %3")
V3(k_pk_max3_u16, "v_pk_max3_u16 %0, %1, %2, %3")
V3(k_pk_min3_i16, "v_pk_min3_i16 %0, %1, %2, %3")
V3(k_pk_min3_u16, "v_pk_min3_u16 %0, %1, %2, %3")

V3(k_fma_mix_f32_bf16, "v_fma_mix_f32_bf16 %0, %1, %2, %3 op_sel:[1,0,0] op_sel_hi:[1,0,1]")
V3(k_fma_mix_f32_bf16_neg, "v_fma_mix_f32_bf16 %0, %1, %2, -%3 op_sel:[1,0,0] op_sel_hi:[1,0,0]")
V3(k_fma_mix_f32_bf16_abs, "v_fma_mix_f32_bf16 %0, |%1|, %2, |%3| op_sel:[1,0,0] op_sel_hi:[1,0,1]")
V3(k_fma_mixlo_bf16, "v_fma_mixlo_bf16 %0, %1, %2, %3 op_sel:[1,0,0] op_sel_hi:[1,0,1]")
V3(k_fma_mixhi_bf16, "v_fma_mixhi_bf16 %0, %1, %2, %3 op_sel:[1,0,0] op_sel_hi:[1,0,1]")
V1(k_cvt_pk_fp8_f16, "v_cvt_pk_fp8_f16 %0, %1")
V1(k_cvt_pk_bf8_f16, "v_cvt_pk_bf8_f16 %0, %1")
V1(k_cvt_pk_fp8_f16_hi, "v_cvt_pk_fp8_f16 %0, %1 op_sel:[0,1]")
V2(k_cvt_sr_fp8_f16, "v_cvt_sr_fp8_f16 %0, %1, %2")
V2(k_cvt_sr_bf8_f16, "v_cvt_sr_bf8_f16 %0, %1, %2")
V3(k_cvt_sr_pk_bf16_f32, "v_cvt_sr_pk_bf16_f32 %0, %1, %2, %3")
V3(k_cvt_sr_pk_f16_f32, "v_cvt_sr_pk_f16_f32 %0, %1, %2, %3")

V3(k_cubeid_f32, "v_cubeid_f32 %0, %1, %2, %3")
V3(k_cubesc_f32, "v_cubesc_f32 %0, %1, %2, %3")
V3(k_cubetc_f32, "v_cubetc_f32 %0, %1, %2, %3")
V3(k_cubema_f32, "v_cubema_f32 %0, %1, %2, %3")

typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
// The lookup-table permutes: a table in the first two sources, sixteen 4-bit indices in the third (a pair of registers).
__global__ void k_perm_pk16(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* o) {
  const int i = blockIdx.x * 32 + threadIdx.x;
  const uint32_t lo = a[i], hi = b[i];
  const uint64_t pair0 = uint64_t{a[i]} | uint64_t{b[i]} << 32, pair1 = uint64_t{c[i]} | uint64_t{a[(i + 5) % 512]} << 32,
                 idx = uint64_t{c[(i + 3) % 512]} | uint64_t{b[(i + 9) % 512]} << 32;
  uint64_t r4;
  u32x3 r6;
  u32x4 r8;
  asm volatile("v_perm_pk16_b4_u4 %0, %1, %2, %3" : "=v"(r4) : "v"(lo), "v"(hi), "v"(idx));
  asm volatile("v_perm_pk16_b6_u4 %0, %1, %2, %3" : "=v"(r6) : "v"(lo), "v"(pair1), "v"(idx));
  asm volatile("v_perm_pk16_b8_u4 %0, %1, %2, %3" : "=v"(r8) : "v"(pair0), "v"(pair1), "v"(idx));
  o[i] = static_cast<uint32_t>(r4);
  o[i + 512] = static_cast<uint32_t>(r4 >> 32);
  for (int k = 0; k < 3; ++k) o[i + 1024 + 512 * k] = r6[k];
  for (int k = 0; k < 4; ++k) o[i + 2560 + 512 * k] = r8[k];
}

__global__ void k_swap_b32(const uint32_t* a, const uint32_t* b, const uint32_t*, uint32_t* o) {
  const int i = blockIdx.x * 32 + threadIdx.x;
  uint32_t x = a[i], y = b[i];
  asm volatile("v_swap_b32 %0, %1" : "+v"(x), "+v"(y));
  o[i] = x;
  o[i + 512] = y;
}

// v_fmamk_f64 and v_fmaak_f64: a 64-bit constant in the instruction stream. The first takes 2.0 (the high half of the double
// alone, as the assembler writes it) and the second a constant with a low half too.
__global__ void k_fma_k64(const uint32_t* xlo, const uint32_t* xhi, const uint32_t* ylo, uint32_t* o) {
  const int i = blockIdx.x * 32 + threadIdx.x;
  const uint64_t x = uint64_t{xlo[i]} | uint64_t{xhi[i]} << 32, y = uint64_t{ylo[i]} | uint64_t{ylo[i + 512]} << 32;
  uint64_t mk, ak, mk2, ak2;
  asm volatile("v_fmamk_f64 %0, %1, 0x40000000, %2" : "=v"(mk) : "v"(x), "v"(y));
  asm volatile("v_fmaak_f64 %0, %1, %2, 0x40080000" : "=v"(ak) : "v"(x), "v"(y));
  asm volatile("v_fmamk_f64 %0, %1, lit64(0x3ff0000000000001), %2" : "=v"(mk2) : "v"(x), "v"(y));
  asm volatile("v_fmaak_f64 %0, %1, %2, lit64(0x4008000000000001)" : "=v"(ak2) : "v"(x), "v"(y));
  o[i] = static_cast<uint32_t>(mk); o[i + 512] = static_cast<uint32_t>(mk >> 32);
  o[i + 1024] = static_cast<uint32_t>(ak); o[i + 1536] = static_cast<uint32_t>(ak >> 32);
  o[i + 2048] = static_cast<uint32_t>(mk2); o[i + 2560] = static_cast<uint32_t>(mk2 >> 32);
  o[i + 3072] = static_cast<uint32_t>(ak2); o[i + 3584] = static_cast<uint32_t>(ak2 >> 32);
}

// Doubles, in pairs of the 32-bit arrays (a: low words, b: high words of x; c, d of y -- here passed as a, b and c, o2).
__global__ void k_f64(const uint32_t* xlo, const uint32_t* xhi, const uint32_t* ylo, uint32_t* o) {
  const int i = blockIdx.x * 32 + threadIdx.x;
  // The second operand's high word follows the first's in the same arrays, one block on.
  const uint64_t x = uint64_t{xlo[i]} | uint64_t{xhi[i]} << 32, y = uint64_t{ylo[i]} | uint64_t{ylo[i + 512]} << 32;
  uint64_t mn, mx, sq;
  asm volatile("v_minimum_f64 %0, %1, %2" : "=v"(mn) : "v"(x), "v"(y));
  asm volatile("v_maximum_f64 %0, %1, %2" : "=v"(mx) : "v"(x), "v"(y));
  asm volatile("v_sqrt_f64 %0, %1" : "=v"(sq) : "v"(x));
  o[i] = static_cast<uint32_t>(mn);
  o[i + 512] = static_cast<uint32_t>(mn >> 32);
  o[i + 1024] = static_cast<uint32_t>(mx);
  o[i + 1536] = static_cast<uint32_t>(mx >> 32);
  o[i + 2048] = static_cast<uint32_t>(sq);
  o[i + 2560] = static_cast<uint32_t>(sq >> 32);
}

// Scalar instructions: one wave, one element after another (the operands are uniform loads).
#define S2(NAME, INSTR)                                                                  \
  __global__ void NAME(const uint32_t* a, const uint32_t* b, uint32_t* o) {              \
    for (int i = 0; i < 512; ++i) {                                                      \
      uint32_t r = 0;                                                                    \
      asm volatile(INSTR : "+s"(r) : "s"(U(a[i])), "s"(U(b[i])));                              \
      if (threadIdx.x == 0) o[i] = r;                                                    \
    }                                                                                    \
  }
#define S1(NAME, INSTR)                                                                  \
  __global__ void NAME(const uint32_t* a, const uint32_t*, uint32_t* o) {                \
    for (int i = 0; i < 512; ++i) {                                                      \
      uint32_t r = 0;                                                                    \
      asm volatile(INSTR : "+s"(r) : "s"(U(a[i])));                                         \
      if (threadIdx.x == 0) o[i] = r;                                                    \
    }                                                                                    \
  }
S2(k_s_minimum_f32, "s_minimum_f32 %0, %1, %2")
S2(k_s_maximum_f32, "s_maximum_f32 %0, %1, %2")
S2(k_s_minimum_f16, "s_minimum_f16 %0, %1, %2")
S2(k_s_maximum_f16, "s_maximum_f16 %0, %1, %2")
S1(k_s_ceil_f16, "s_ceil_f16 %0, %1")
S1(k_s_floor_f16, "s_floor_f16 %0, %1")
S1(k_s_trunc_f16, "s_trunc_f16 %0, %1")
S1(k_s_rndne_f16, "s_rndne_f16 %0, %1")
S2(k_s_cvt_pk_rtz, "s_cvt_pk_rtz_f16_f32 %0, %1, %2")
S1(k_s_quadmask_b32, "s_quadmask_b32 %0, %1")
__global__ void k_s_bitreplicate(const uint32_t* a, const uint32_t*, uint32_t* o) {
  for (int i = 0; i < 512; ++i) {
    uint64_t r;
    asm volatile("s_bitreplicate_b64_b32 %0, %1" : "=s"(r) : "s"(U(a[i])));
    if (threadIdx.x == 0) o[i] = static_cast<uint32_t>(r), o[i + 512] = static_cast<uint32_t>(r >> 32);
  }
}
// The 64-bit scalar forms act on a pair: s_quadmask_b64 and s_cls_i32_i64 read a pair (a[i] and a[i+1]); s_bitset0/1_b64 change
// a pair whose bit a[i] names.
__global__ void k_s_pairs(const uint32_t* a, const uint32_t*, uint32_t* o) {
  for (int i = 0; i < 256; ++i) {
    const uint32_t xl = U(a[2 * i]), xh = U(a[2 * i + 1]);
    const uint64_t x = uint64_t{xl} | uint64_t{xh} << 32;
    uint64_t q;
    uint32_t cls;
    uint64_t set0 = x, set1 = x;
    asm volatile("s_quadmask_b64 %0, %1" : "=s"(q) : "s"(x));
    asm volatile("s_cls_i32_i64 %0, %1" : "=s"(cls) : "s"(x));
    asm volatile("s_bitset0_b64 %0, %1" : "+s"(set0) : "s"(xl));
    asm volatile("s_bitset1_b64 %0, %1" : "+s"(set1) : "s"(xh));
    if (threadIdx.x == 0) {
      o[4 * i] = static_cast<uint32_t>(q);
      o[4 * i + 1] = cls;
      o[4 * i + 2] = static_cast<uint32_t>(set0) ^ static_cast<uint32_t>(set0 >> 32);
      o[4 * i + 3] = static_cast<uint32_t>(set1) ^ static_cast<uint32_t>(set1 >> 32);
    }
  }
}

// The EXEC-writing ones: EXEC is set to e, the instruction run on x, and what it left in EXEC, in its destination and in
// SCC read back, before EXEC goes back to all ones. Only the low 32 bits of EXEC are compared (the wave is 32 wide).
#define EXECOP(NAME, INSTR)                                                              \
  __global__ void NAME(const uint32_t* x, const uint32_t* e, uint32_t* o) {              \
    for (int i = 0; i < 128; ++i) {                                                      \
      uint32_t dst = 0, now = 0, scc = 0;                                                \
      asm volatile("s_mov_b32 exec_lo, %3\n\t" INSTR "\n\ts_cselect_b32 %2, 1, 0\n\ts_mov_b32 %1, exec_lo\n\ts_mov_b32 exec_lo, -1" \
                   : "=&s"(dst), "=&s"(now), "=&s"(scc) : "s"(U(e[i])), "s"(U(x[i])) : "scc");  \
      if (threadIdx.x == 0) o[3 * i] = dst, o[3 * i + 1] = now, o[3 * i + 2] = scc;      \
    }                                                                                    \
  }
EXECOP(k_nand32, "s_nand_saveexec_b32 %0, %4")
EXECOP(k_nor32, "s_nor_saveexec_b32 %0, %4")
EXECOP(k_xnor32, "s_xnor_saveexec_b32 %0, %4")
EXECOP(k_wr0_32, "s_and_not0_wrexec_b32 %0, %4")
EXECOP(k_wr1_32, "s_and_not1_wrexec_b32 %0, %4")

// DPP8: each lane of a group of eight takes the lane its three-bit selector names. Lanes that are off (every third)
// supply a zero, or, with fi, what they hold. A comparison takes its first operand the same way.
__global__ void k_dpp8(const uint32_t* a, const uint32_t* b, uint32_t* o) {
  const int l = threadIdx.x;
  uint32_t x = a[l], y = b[l];
  asm volatile("" : "+v"(x), "+v"(y));   // keep the loads before the branch: a lane that is off must still hold its value
  uint32_t r = 0, rf = 0;
  if (l % 3 != 0) {
    asm volatile("v_add_nc_u32_dpp %0, %1, %2 dpp8:[3,0,5,2,7,4,1,6]" : "=v"(r) : "v"(x), "v"(y));
    asm volatile("v_add_nc_u32_dpp %0, %1, %2 dpp8:[3,0,5,2,7,4,1,6] fi:1" : "=v"(rf) : "v"(x), "v"(y));
  }
  o[l] = r;
  o[32 + l] = rf;
}
__global__ void k_dpp_cmp(const uint32_t* a, const uint32_t* b, uint32_t* o) {
  const uint32_t x = a[threadIdx.x], y = b[threadIdx.x];
  uint32_t m8, m16;
  asm volatile("v_cmp_lt_u32_dpp vcc_lo, %1, %2 dpp8:[7,6,5,4,3,2,1,0]\n\ts_mov_b32 %0, vcc_lo" : "=s"(m8) : "v"(x), "v"(y) : "vcc");
  asm volatile("v_cmp_lt_u32_dpp vcc_lo, %1, %2 row_xmask:1 row_mask:0xf bank_mask:0xf\n\ts_mov_b32 %0, vcc_lo" : "=s"(m16) : "v"(x), "v"(y) : "vcc");
  if (threadIdx.x == 0) o[0] = m8, o[1] = m16;
}

// --- the host's side ------------------------------------------------------------------------------------------------
static uint32_t seed = 20260101;
static uint32_t rnd32() {
  seed = seed * 1664525u + 1013904223u;
  return seed;
}
static float bf(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
static uint32_t fb(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
static double bd(uint64_t b) { double f; std::memcpy(&f, &b, 8); return f; }
static uint64_t db(double f) { uint64_t b; std::memcpy(&b, &f, 8); return b; }
static float h2f(uint16_t h) {
  const int sign = h >> 15 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e == 0) return sign * std::ldexp(static_cast<float>(m), -24);
  if (e == 31) return m ? NAN : sign * INFINITY;
  return sign * std::ldexp(static_cast<float>(m + 1024), e - 25);
}
static uint16_t f2h(float f) {
  _Float16 h = static_cast<_Float16>(f);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
static uint16_t bf16_bits(float f) {   // round to nearest even
  const uint32_t b = fb(f);
  if (std::isnan(f)) return static_cast<uint16_t>(b >> 16 | 0x40);
  return static_cast<uint16_t>((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}
static float bf16f(uint16_t b) { return bf(uint32_t{b} << 16); }

// IEEE minimum()/maximum() from the ISA's pseudocode, on a format of `eb` exponent and `mb` mantissa bits.
template <int EB, int MB>
static uint64_t ref_minmax(uint64_t a, uint64_t b, bool mx) {
  const uint64_t emask = ((1ull << EB) - 1) << MB, mmask = (1ull << MB) - 1, qbit = 1ull << (MB - 1), sbit = 1ull << (EB + MB);
  const auto nan = [&](uint64_t v) { return (v & emask) == emask && (v & mmask); };
  const auto snan = [&](uint64_t v) { return nan(v) && !(v & qbit); };
  if (snan(a)) return a | qbit;
  if (snan(b)) return b | qbit;
  if (nan(a)) return a;
  if (nan(b)) return b;
  const auto val = [&](uint64_t v) -> double {
    if constexpr (EB == 5) return h2f(static_cast<uint16_t>(v));
    else if constexpr (EB == 8) return bf(static_cast<uint32_t>(v));
    else return bd(v);
  };
  const double x = val(a), y = val(b);
  const bool zeros = x == 0 && y == 0;
  if (!mx) return (x < y || (zeros && (a & sbit) && !(b & sbit))) ? a : b;
  return (x > y || (zeros && !(a & sbit) && (b & sbit))) ? a : b;
}
static bool is_nan32(uint32_t b) { return std::isnan(bf(b)); }
static bool is_nan16(uint16_t b) { return (b & 0x7C00) == 0x7C00 && (b & 0x3FF); }

struct Run {
  int failed = 0, total = 0;
};
static void report(const char* what, int wrong, int of, int* failed) {
  std::printf("%s: %d of %d wrong\n", what, wrong, of);
  *failed += wrong != 0;
}

typedef void (*Kernel)(const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*);
constexpr int N = 512;
static uint32_t *da, *db_, *dc, *dout;

template <class T>
static T* up(const std::vector<T>& v) {
  T* p = nullptr;
  if (hipMalloc(&p, v.size() * sizeof(T)) != hipSuccess) return nullptr;
  if (hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess) return nullptr;
  return p;
}

#define LAUNCH(K, A, B, C) \
  (K<<<N / 32, 32>>>(A, B, C, dout), hipDeviceSynchronize() == hipSuccess)

static std::vector<uint32_t> fetch(size_t n) {
  std::vector<uint32_t> out(n);
  if (hipMemcpy(out.data(), dout, n * 4, hipMemcpyDeviceToHost) != hipSuccess) out.clear();
  return out;
}

// Pools of operands: the awkward values first, then random ones.
static std::vector<uint32_t> pool_f32() {
  const uint32_t special[] = {0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x40000000, 0xC0400000, 0x7F800000, 0xFF800000,
                              0x7FC00000, 0xFFC00001, 0x7F800001, 0xFF800123, 0x7F7FFFFF, 0xFF7FFFFF, 0x00000001, 0x80400000,
                              0x3F000000, 0x40490FDB};
  std::vector<uint32_t> v;
  for (int i = 0; i < N; ++i) v.push_back(i < 18 * 18 ? special[(i / 18 + i % 18 * 5) % 18] : (rnd32() & 1 ? rnd32() : fb((static_cast<int>(rnd32() % 4001) - 2000) / 64.0f)));
  return v;
}
static std::vector<uint32_t> pool_f16() {
  const uint16_t special[] = {0x0000, 0x8000, 0x3C00, 0xBC00, 0x4000, 0xC200, 0x7C00, 0xFC00, 0x7E00, 0xFE01, 0x7C01, 0xFD23,
                              0x7BFF, 0xFBFF, 0x0001, 0x8200, 0x3800, 0x4248};
  std::vector<uint32_t> v;
  for (int i = 0; i < N; ++i) v.push_back(i < 18 * 18 ? special[(i / 18 + i % 18 * 5) % 18] : (rnd32() & 1 ? (rnd32() & 0xFFFF) : f2h((static_cast<int>(rnd32() % 401) - 200) / 16.0f)));
  return v;
}
static std::vector<uint32_t> pool_bits() {
  std::vector<uint32_t> v;
  for (int i = 0; i < N; ++i) v.push_back(rnd32() & 3 ? rnd32() : (rnd32() & 1 ? 0x8000u | (rnd32() & 0xFF) : 0x7FFFu - (rnd32() & 0xFF)) * 0x10001u);
  return v;
}
// Packed pools: two values of a kind in a register.
static std::vector<uint32_t> pool_pk(const std::vector<uint32_t>& one) {
  std::vector<uint32_t> v(N);
  for (int i = 0; i < N; ++i) v[i] = (one[i] & 0xFFFF) | (one[(i * 7 + 3) % N] & 0xFFFF) << 16;
  return v;
}

struct Ctx {
  int* failed;
};

// Compares a kernel's output with a reference over the inputs; `nan_ok` accepts any NaN where the reference is one.
template <class Ref>
static int run3(const char* what, Kernel k, const std::vector<uint32_t>& A, const std::vector<uint32_t>& B, const std::vector<uint32_t>& C,
                Ref ref, int mask_bits, bool any_nan, bool f16nan, int* failed) {
  uint32_t* a = up(A);
  uint32_t* b = up(B);
  uint32_t* c = up(C);
  if (!LAUNCH(k, a, b, c)) { std::printf("FAIL %s: launch failed\n", what); return 1; }
  const auto out = fetch(N);
  int wrong = 0;
  for (int i = 0; i < N; ++i) {
    const uint32_t want = ref(A[i], B[i], C[i]);
    const uint32_t m = mask_bits == 16 ? 0xFFFFu : 0xFFFFFFFFu;
    const uint32_t got = out[i] & m;
    const bool both_nan = any_nan && (f16nan ? is_nan16(static_cast<uint16_t>(want)) && is_nan16(static_cast<uint16_t>(got)) : is_nan32(want) && is_nan32(got));
    if (got != (want & m) && !both_nan) {
      if (wrong < 3) std::printf("  %s: a=%08x b=%08x c=%08x want %08x got %08x\n", what, A[i], B[i], C[i], want & m, got);
      ++wrong;
    }
  }
  hipFree(a); hipFree(b); hipFree(c);
  report(what, wrong, N, failed);
  return 0;
}

int main() {
  int failed = 0;
  CHECK(hipMalloc(&dout, 8192 * 4));
  const auto F = pool_f32(), F2 = pool_f32(), F3 = pool_f32();
  std::vector<uint32_t> G(N), G3(N);
  for (int i = 0; i < N; ++i) G[i] = F2[(i * 13 + 5) % N], G3[i] = F3[(i * 29 + 11) % N];
  const auto H = pool_f16(), H2 = pool_f16(), H3 = pool_f16();
  std::vector<uint32_t> J(N), J3(N);
  for (int i = 0; i < N; ++i) J[i] = H2[(i * 13 + 5) % N], J3[i] = H3[(i * 29 + 11) % N];

  // --- the IEEE minimum and maximum, f32 and f16 ---
  const auto mm32 = [](uint32_t a, uint32_t b, bool mx) { return static_cast<uint32_t>(ref_minmax<8, 23>(a, b, mx)); };
  const auto mm16 = [](uint32_t a, uint32_t b, bool mx) { return static_cast<uint32_t>(ref_minmax<5, 10>(a & 0xFFFF, b & 0xFFFF, mx)); };
  run3("v_minimum_f32", k_minimum_f32, F, G, G3, [&](uint32_t a, uint32_t b, uint32_t) { return mm32(a, b, false); }, 32, false, false, &failed);
  run3("v_maximum_f32", k_maximum_f32, F, G, G3, [&](uint32_t a, uint32_t b, uint32_t) { return mm32(a, b, true); }, 32, false, false, &failed);
  run3("v_minimum3_f32", k_minimum3_f32, F, G, G3, [&](uint32_t a, uint32_t b, uint32_t c) { return mm32(mm32(a, b, false), c, false); }, 32, false, false, &failed);
  run3("v_maximum3_f32", k_maximum3_f32, F, G, G3, [&](uint32_t a, uint32_t b, uint32_t c) { return mm32(mm32(a, b, true), c, true); }, 32, false, false, &failed);
  run3("v_minimummaximum_f32", k_minimummaximum_f32, F, G, G3, [&](uint32_t a, uint32_t b, uint32_t c) { return mm32(mm32(a, b, false), c, true); }, 32, false, false, &failed);
  run3("v_maximumminimum_f32", k_maximumminimum_f32, F, G, G3, [&](uint32_t a, uint32_t b, uint32_t c) { return mm32(mm32(a, b, true), c, false); }, 32, false, false, &failed);
  run3("v_minimum_f16", k_minimum_f16, H, J, J3, [&](uint32_t a, uint32_t b, uint32_t) { return mm16(a, b, false); }, 16, false, false, &failed);
  run3("v_maximum_f16", k_maximum_f16, H, J, J3, [&](uint32_t a, uint32_t b, uint32_t) { return mm16(a, b, true); }, 16, false, false, &failed);
  run3("v_minimum3_f16", k_minimum3_f16, H, J, J3, [&](uint32_t a, uint32_t b, uint32_t c) { return mm16(mm16(a, b, false), c, false); }, 16, false, false, &failed);
  run3("v_maximum3_f16", k_maximum3_f16, H, J, J3, [&](uint32_t a, uint32_t b, uint32_t c) { return mm16(mm16(a, b, true), c, true); }, 16, false, false, &failed);
  run3("v_minimummaximum_f16", k_minimummaximum_f16, H, J, J3, [&](uint32_t a, uint32_t b, uint32_t c) { return mm16(mm16(a, b, false), c, true); }, 16, false, false, &failed);
  run3("v_maximumminimum_f16", k_maximumminimum_f16, H, J, J3, [&](uint32_t a, uint32_t b, uint32_t c) { return mm16(mm16(a, b, true), c, false); }, 16, false, false, &failed);

  // --- the "_num" forms: a number beats a NaN; the result is a NaN only where all the inputs are. ---
  const auto num3_32 = [](const char* kind, uint32_t a, uint32_t b, uint32_t c) -> uint32_t {
    const float x = bf(a), y = bf(b), z = bf(c);
    const std::string k = kind;
    const auto mn = [](float p, float q) { return std::isnan(p) ? q : std::isnan(q) ? p : (p < q || (p == 0 && q == 0 && std::signbit(p)) ? p : q); };
    const auto mxf = [](float p, float q) { return std::isnan(p) ? q : std::isnan(q) ? p : (p > q || (p == 0 && q == 0 && !std::signbit(p)) ? p : q); };
    float r;
    if (k == "min3") r = mn(mn(x, y), z);
    else if (k == "max3") r = mxf(mxf(x, y), z);
    else if (k == "minmax") r = mxf(mn(x, y), z);
    else if (k == "maxmin") r = mn(mxf(x, y), z);
    else if (std::isnan(x) || std::isnan(y) || std::isnan(z)) r = mn(mn(x, y), z);
    else {
      const float m = mxf(mxf(x, y), z);
      r = m == x ? mxf(y, z) : m == y ? mxf(x, z) : mxf(x, y);
    }
    return fb(r);
  };
  struct { const char* name; Kernel k; const char* kind; } nums32[] = {
      {"v_min3_num_f32", k_min3_num_f32, "min3"}, {"v_max3_num_f32", k_max3_num_f32, "max3"}, {"v_minmax_num_f32", k_minmax_num_f32, "minmax"},
      {"v_maxmin_num_f32", k_maxmin_num_f32, "maxmin"}, {"v_med3_num_f32", k_med3_num_f32, "med3"}};
  for (const auto& n : nums32)
    run3(n.name, n.k, F, G, G3, [&](uint32_t a, uint32_t b, uint32_t c) { return num3_32(n.kind, a, b, c); }, 32, true, false, &failed);
  struct { const char* name; Kernel k; const char* kind; } nums16[] = {
      {"v_min3_num_f16", k_min3_num_f16, "min3"}, {"v_max3_num_f16", k_max3_num_f16, "max3"}, {"v_minmax_num_f16", k_minmax_num_f16, "minmax"},
      {"v_maxmin_num_f16", k_maxmin_num_f16, "maxmin"}, {"v_med3_num_f16", k_med3_num_f16, "med3"}};
  for (const auto& n : nums16)
    run3(n.name, n.k, H, J, J3, [&](uint32_t a, uint32_t b, uint32_t c) {
      return uint32_t{f2h(bf(num3_32(n.kind, fb(h2f(a & 0xFFFF)), fb(h2f(b & 0xFFFF)), fb(h2f(c & 0xFFFF)))))};
    }, 16, true, true, &failed);

  // --- 16-bit integers ---
  const auto I = pool_bits(), I2 = pool_bits(), I3 = pool_bits();
  run3("v_add_nc_i16", k_add_nc_i16, I, I2, I3, [](uint32_t a, uint32_t b, uint32_t) { return (a + b) & 0xFFFF; }, 16, false, false, &failed);
  run3("v_sub_nc_i16", k_sub_nc_i16, I, I2, I3, [](uint32_t a, uint32_t b, uint32_t) { return (a - b) & 0xFFFF; }, 16, false, false, &failed);
  run3("v_max_i16", k_max_i16, I, I2, I3, [](uint32_t a, uint32_t b, uint32_t) { return static_cast<uint32_t>(std::max<int16_t>(a, b)) & 0xFFFF; }, 16, false, false, &failed);
  run3("v_min_i16", k_min_i16, I, I2, I3, [](uint32_t a, uint32_t b, uint32_t) { return static_cast<uint32_t>(std::min<int16_t>(a, b)) & 0xFFFF; }, 16, false, false, &failed);
  run3("v_ashrrev_i16", k_ashrrev_i16, I, I2, I3, [](uint32_t a, uint32_t b, uint32_t) { return static_cast<uint32_t>(static_cast<int16_t>(b) >> (a & 15)) & 0xFFFF; }, 16, false, false, &failed);
  run3("v_cvt_i32_i16", k_cvt_i32_i16, I, I2, I3, [](uint32_t a, uint32_t, uint32_t) { return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(a))); }, 32, false, false, &failed);
  run3("v_cvt_u32_u16", k_cvt_u32_u16, I, I2, I3, [](uint32_t a, uint32_t, uint32_t) { return a & 0xFFFF; }, 32, false, false, &failed);
  run3("v_sat_pk_u8_i16", k_sat_pk_u8_i16, I, I2, I3, [](uint32_t a, uint32_t, uint32_t) {
    const auto sat = [](int16_t v) { return v <= 0 ? 0u : v >= 255 ? 255u : static_cast<uint32_t>(v); };
    return sat(static_cast<int16_t>(a)) | sat(static_cast<int16_t>(a >> 16)) << 8;
  }, 16, false, false, &failed);

  // --- half-precision transcendentals and exponents ---
  run3("v_frexp_mant_f16", k_frexp_mant_f16, H, J, J3, [](uint32_t a, uint32_t, uint32_t) {
    const float x = h2f(a & 0xFFFF);
    int e;
    return uint32_t{!std::isfinite(x) ? static_cast<uint16_t>(a) : f2h(std::frexp(x, &e))};
  }, 16, true, true, &failed);
  run3("v_frexp_exp_i16_f16", k_frexp_exp_i16_f16, H, J, J3, [](uint32_t a, uint32_t, uint32_t) {
    const float x = h2f(a & 0xFFFF);
    int e = 0;
    if (std::isfinite(x)) std::frexp(x, &e);
    return static_cast<uint32_t>(e) & 0xFFFF;
  }, 16, false, false, &failed);
  {
    // v_ldexp_f16: the exponent is a signed 16-bit integer; keep it small.
    std::vector<uint32_t> E(N);
    for (int i = 0; i < N; ++i) E[i] = static_cast<uint32_t>(static_cast<int>(rnd32() % 41) - 20) & 0xFFFF;
    run3("v_ldexp_f16", k_ldexp_f16, H, E, J3, [](uint32_t a, uint32_t b, uint32_t) {
      return uint32_t{f2h(std::ldexp(h2f(a & 0xFFFF), static_cast<int16_t>(b)))};
    }, 16, true, true, &failed);
  }
  {
    // sin and cos of a half in turns: within one unit of the last place of the exact answer.
    for (int cosine = 0; cosine < 2; ++cosine) {
      uint32_t* a = up(H);
      uint32_t* b = up(J);
      uint32_t* c = up(J3);
      if (cosine) { k_cos_f16<<<N / 32, 32>>>(a, b, c, dout); } else { k_sin_f16<<<N / 32, 32>>>(a, b, c, dout); }
      CHECK(hipDeviceSynchronize());
      const auto out = fetch(N);
      int wrong = 0;
      for (int i = 0; i < N; ++i) {
        const float x = h2f(H[i] & 0xFFFF);
        const uint16_t got = out[i] & 0xFFFF;
        if (!std::isfinite(x)) { wrong += !is_nan16(got); continue; }
        const double frac = x - std::floor(static_cast<double>(x));
        const double exact = cosine ? std::cos(frac * 2 * M_PI) : std::sin(frac * 2 * M_PI);
        const double g = h2f(got);
        // a unit in the last place of the exact answer's half, or a half-precision quantum near zero
        const double tol = std::max(std::fabs(exact) * 0.001, 1e-6);
        wrong += !(std::fabs(g - exact) <= tol);
      }
      hipFree(a); hipFree(b); hipFree(c);
      report(cosine ? "v_cos_f16" : "v_sin_f16", wrong, N, &failed);
    }
  }

  // --- conversions ---
  run3("v_cvt_nearest_i32_f32", k_cvt_nearest_i32_f32, F, G, G3, [](uint32_t a, uint32_t, uint32_t) {
    const float x = bf(a);
    if (std::isnan(x)) return 0u;
    const double r = std::floor(static_cast<double>(x) + 0.5);
    return static_cast<uint32_t>(r <= -2147483648.0 ? INT32_MIN : r >= 2147483647.0 ? INT32_MAX : static_cast<int32_t>(r));
  }, 32, false, false, &failed);
  {
    // The offset table of the ISA: 1000 is -0.5, up to 1111 at -0.0625, then 0 to 0.4375.
    const float table[16] = {0.0f, 0.0625f, 0.125f, 0.1875f, 0.25f, 0.3125f, 0.375f, 0.4375f,
                             -0.5f, -0.4375f, -0.375f, -0.3125f, -0.25f, -0.1875f, -0.125f, -0.0625f};
    run3("v_cvt_off_f32_i4", k_cvt_off_f32_i4, I, I2, I3, [&](uint32_t a, uint32_t, uint32_t) { return fb(table[a & 15]); }, 32, false, false, &failed);
  }
  run3("v_cvt_norm_i16_f16", k_cvt_norm_i16_f16, H, J, J3, [](uint32_t a, uint32_t, uint32_t) {
    const float x = h2f(a & 0xFFFF);
    if (std::isnan(x)) return 0u;
    return static_cast<uint32_t>(static_cast<int32_t>(std::nearbyint(std::fmin(1.0f, std::fmax(-1.0f, x)) * 32767.0f))) & 0xFFFF;
  }, 16, false, false, &failed);
  run3("v_cvt_norm_u16_f16", k_cvt_norm_u16_f16, H, J, J3, [](uint32_t a, uint32_t, uint32_t) {
    const float x = h2f(a & 0xFFFF);
    if (std::isnan(x)) return 0u;
    return static_cast<uint32_t>(std::nearbyint(std::fmin(1.0f, std::fmax(0.0f, x)) * 65535.0f)) & 0xFFFF;
  }, 16, false, false, &failed);

  // --- DX9 and lighting multiplies ---
  run3("v_mul_dx9_zero_f32", k_mul_dx9_zero_f32, F, G, G3, [](uint32_t a, uint32_t b, uint32_t) {
    return bf(a) == 0 || bf(b) == 0 ? 0u : fb(bf(a) * bf(b));
  }, 32, true, false, &failed);
  run3("v_fma_dx9_zero_f32", k_fma_dx9_zero_f32, F, G, G3, [](uint32_t a, uint32_t b, uint32_t c) {
    return bf(a) == 0 || bf(b) == 0 ? c : fb(std::fma(bf(a), bf(b), bf(c)));
  }, 32, true, false, &failed);
  run3("v_mullit_f32", k_mullit_f32, F, G, G3, [](uint32_t a, uint32_t b, uint32_t c) {
    const float lowest = -3.4028234663852886e38f;
    return (bf(b) == lowest || (std::isinf(bf(b)) && bf(b) < 0) || std::isnan(bf(b)) || bf(c) <= 0.0f || std::isnan(bf(c))) ? fb(lowest) : fb(bf(a) * bf(b));
  }, 32, true, false, &failed);

  // --- packed bfloat16: values small enough that the arithmetic is exact in float ---
  {
    std::vector<uint32_t> P(N), P2(N), P3(N);
    for (int i = 0; i < N; ++i) {
      const auto one = [&]() { return uint32_t{bf16_bits((static_cast<int>(rnd32() % 65) - 32) / 4.0f)}; };
      P[i] = one() | one() << 16;
      P2[i] = one() | one() << 16;
      P3[i] = one() | one() << 16;
    }
    const auto lanes2 = [](uint32_t a, uint32_t b, uint32_t c, auto f) {
      uint32_t r = 0;
      for (int h = 0; h < 2; ++h) r |= uint32_t{bf16_bits(f(bf16f(a >> (16 * h)), bf16f(b >> (16 * h)), bf16f(c >> (16 * h))))} << (16 * h);
      return r;
    };
    run3("v_pk_add_bf16", k_pk_add_bf16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return lanes2(a, b, c, [](float x, float y, float) { return x + y; }); }, 32, false, false, &failed);
    run3("v_pk_mul_bf16", k_pk_mul_bf16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return lanes2(a, b, c, [](float x, float y, float) { return x * y; }); }, 32, false, false, &failed);
    run3("v_pk_fma_bf16", k_pk_fma_bf16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return lanes2(a, b, c, [](float x, float y, float z) { return std::fma(x, y, z); }); }, 32, false, false, &failed);
    run3("v_pk_min_num_bf16", k_pk_min_num_bf16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return lanes2(a, b, c, [](float x, float y, float) { return std::fmin(x, y); }); }, 32, false, false, &failed);
    run3("v_pk_max_num_bf16", k_pk_max_num_bf16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return lanes2(a, b, c, [](float x, float y, float) { return std::fmax(x, y); }); }, 32, false, false, &failed);
  }

  // --- packed half ---
  {
    const auto P = pool_pk(H), P2 = pool_pk(J), P3 = pool_pk(J3);
    const auto per_half = [](uint32_t a, uint32_t b, uint32_t c, auto f) {
      uint32_t r = 0;
      for (int h = 0; h < 2; ++h) r |= f((a >> (16 * h)) & 0xFFFF, (b >> (16 * h)) & 0xFFFF, (c >> (16 * h)) & 0xFFFF) << (16 * h);
      return r;
    };
    run3("v_pk_minimum_f16", k_pk_minimum_f16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return per_half(a, b, c, [&](uint32_t x, uint32_t y, uint32_t) { return mm16(x, y, false); }); }, 32, false, false, &failed);
    run3("v_pk_maximum_f16", k_pk_maximum_f16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return per_half(a, b, c, [&](uint32_t x, uint32_t y, uint32_t) { return mm16(x, y, true); }); }, 32, false, false, &failed);
    run3("v_pk_minimum3_f16", k_pk_minimum3_f16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return per_half(a, b, c, [&](uint32_t x, uint32_t y, uint32_t z) { return mm16(mm16(x, y, false), z, false); }); }, 32, false, false, &failed);
    run3("v_pk_maximum3_f16", k_pk_maximum3_f16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return per_half(a, b, c, [&](uint32_t x, uint32_t y, uint32_t z) { return mm16(mm16(x, y, true), z, true); }); }, 32, false, false, &failed);
    // The "_num" ones: any NaN result is a NaN, so compare per half by value.
    for (int is_max = 0; is_max < 2; ++is_max) {
      uint32_t* a = up(P);
      uint32_t* b = up(P2);
      uint32_t* c = up(P3);
      if (is_max) k_pk_max3_num_f16<<<N / 32, 32>>>(a, b, c, dout); else k_pk_min3_num_f16<<<N / 32, 32>>>(a, b, c, dout);
      CHECK(hipDeviceSynchronize());
      const auto out = fetch(N);
      int wrong = 0;
      for (int i = 0; i < N; ++i)
        for (int h = 0; h < 2; ++h) {
          const uint32_t want = num3_32(is_max ? "max3" : "min3", fb(h2f((P[i] >> (16 * h)) & 0xFFFF)), fb(h2f((P2[i] >> (16 * h)) & 0xFFFF)), fb(h2f((P3[i] >> (16 * h)) & 0xFFFF)));
          const uint16_t w = f2h(bf(want)), g = (out[i] >> (16 * h)) & 0xFFFF;
          wrong += !(w == g || (is_nan16(w) && is_nan16(g)));
        }
      hipFree(a); hipFree(b); hipFree(c);
      report(is_max ? "v_pk_max3_num_f16" : "v_pk_min3_num_f16", wrong, N, &failed);
    }
    // v_pk_fmac_f16: the destination is the addend. Small values, so the fused result needs no care.
    std::vector<uint32_t> Q(N), Q2(N), Q3(N);
    for (int i = 0; i < N; ++i) {
      const auto one = [&]() { return uint32_t{f2h((static_cast<int>(rnd32() % 33) - 16) / 8.0f)}; };
      Q[i] = one() | one() << 16; Q2[i] = one() | one() << 16; Q3[i] = one() | one() << 16;
    }
    run3("v_pk_fmac_f16", k_pk_fmac_f16, Q, Q2, Q3, [&](uint32_t a, uint32_t b, uint32_t c) {
      return per_half(a, b, c, [](uint32_t x, uint32_t y, uint32_t z) { return uint32_t{f2h(std::fma(h2f(x), h2f(y), h2f(z)))}; });
    }, 32, false, false, &failed);
  }

  // --- packed 16-bit integers ---
  {
    const auto P = pool_bits(), P2 = pool_bits(), P3 = pool_bits();
    const auto per_half = [](uint32_t a, uint32_t b, uint32_t c, auto f) {
      uint32_t r = 0;
      for (int h = 0; h < 2; ++h) r |= (f((a >> (16 * h)) & 0xFFFF, (b >> (16 * h)) & 0xFFFF, (c >> (16 * h)) & 0xFFFF) & 0xFFFF) << (16 * h);
      return r;
    };
    const auto sat_s = [](int x, int y) { return std::max(-32768, std::min(32767, x + y)); };
    const auto sat_u = [](int x, int y) { return std::max(0, std::min(65535, x + y)); };
    const auto s16 = [](uint32_t v) { return static_cast<int>(static_cast<int16_t>(v)); };
    run3("v_pk_add_max_i16", k_pk_add_max_i16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return per_half(a, b, c, [&](uint32_t x, uint32_t y, uint32_t z) { return static_cast<uint32_t>(std::max(sat_s(s16(x), s16(y)), s16(z))); }); }, 32, false, false, &failed);
    run3("v_pk_add_min_i16", k_pk_add_min_i16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return per_half(a, b, c, [&](uint32_t x, uint32_t y, uint32_t z) { return static_cast<uint32_t>(std::min(sat_s(s16(x), s16(y)), s16(z))); }); }, 32, false, false, &failed);
    run3("v_pk_add_max_u16", k_pk_add_max_u16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return per_half(a, b, c, [&](uint32_t x, uint32_t y, uint32_t z) { return static_cast<uint32_t>(std::max(sat_u(x, y), static_cast<int>(z))); }); }, 32, false, false, &failed);
    run3("v_pk_add_min_u16", k_pk_add_min_u16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return per_half(a, b, c, [&](uint32_t x, uint32_t y, uint32_t z) { return static_cast<uint32_t>(std::min(sat_u(x, y), static_cast<int>(z))); }); }, 32, false, false, &failed);
    run3("v_pk_max3_i16", k_pk_max3_i16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return per_half(a, b, c, [&](uint32_t x, uint32_t y, uint32_t z) { return static_cast<uint32_t>(std::max({s16(x), s16(y), s16(z)})); }); }, 32, false, false, &failed);
    run3("v_pk_min3_i16", k_pk_min3_i16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return per_half(a, b, c, [&](uint32_t x, uint32_t y, uint32_t z) { return static_cast<uint32_t>(std::min({s16(x), s16(y), s16(z)})); }); }, 32, false, false, &failed);
    run3("v_pk_max3_u16", k_pk_max3_u16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return per_half(a, b, c, [&](uint32_t x, uint32_t y, uint32_t z) { return std::max({x, y, z}); }); }, 32, false, false, &failed);
    run3("v_pk_min3_u16", k_pk_min3_u16, P, P2, P3, [&](uint32_t a, uint32_t b, uint32_t c) { return per_half(a, b, c, [&](uint32_t x, uint32_t y, uint32_t z) { return std::min({x, y, z}); }); }, 32, false, false, &failed);
  }

  // --- mixed bfloat16 multiply-adds: source 0 is a bfloat16 (the high half), 1 a float, 2 a bfloat16 (the low half) ---
  {
    std::vector<uint32_t> M0(N), M2(N);
    for (int i = 0; i < N; ++i) {
      M0[i] = uint32_t{bf16_bits((static_cast<int>(rnd32() % 65) - 32) / 4.0f)} << 16 | (rnd32() & 0xFFFF);
      M2[i] = (rnd32() & 0xFFFF0000u) | uint32_t{bf16_bits((static_cast<int>(rnd32() % 65) - 32) / 8.0f)};
    }
    std::vector<uint32_t> M1(N);
    for (int i = 0; i < N; ++i) M1[i] = fb((static_cast<int>(rnd32() % 65) - 32) / 16.0f);
    const auto mix = [](uint32_t a, uint32_t b, uint32_t c) { return std::fma(bf16f(static_cast<uint16_t>(a >> 16)), bf(b), bf16f(static_cast<uint16_t>(c))); };
    run3("v_fma_mix_f32_bf16", k_fma_mix_f32_bf16, M0, M1, M2, [&](uint32_t a, uint32_t b, uint32_t c) { return fb(mix(a, b, c)); }, 32, false, false, &failed);
    // A negated float source (what the compiler's exp of a bfloat16 relies on), and absolute values (neg_hi on a mix form).
    run3("v_fma_mix_f32_bf16 with -src2", k_fma_mix_f32_bf16_neg, M0, M1, M2, [&](uint32_t a, uint32_t b, uint32_t c) { return fb(std::fma(bf16f(static_cast<uint16_t>(a >> 16)), bf(b), -bf(c))); }, 32, false, false, &failed);
    run3("v_fma_mix_f32_bf16 with abs on sources 0 and 2", k_fma_mix_f32_bf16_abs, M0, M1, M2, [&](uint32_t a, uint32_t b, uint32_t c) { return fb(std::fma(std::fabs(bf16f(static_cast<uint16_t>(a >> 16))), bf(b), std::fabs(bf16f(static_cast<uint16_t>(c))))); }, 32, false, false, &failed);
    run3("v_fma_mixlo_bf16", k_fma_mixlo_bf16, M0, M1, M2, [&](uint32_t a, uint32_t b, uint32_t c) { return uint32_t{bf16_bits(mix(a, b, c))}; }, 32, false, false, &failed);
    run3("v_fma_mixhi_bf16", k_fma_mixhi_bf16, M0, M1, M2, [&](uint32_t a, uint32_t b, uint32_t c) { return uint32_t{bf16_bits(mix(a, b, c))} << 16; }, 32, false, false, &failed);
  }

  // --- 8-bit floats from halves. Values that are exact in the 8-bit format go through unchanged, and a stochastic
  // rounding of one with a seed under half an ulp (in the half's ten mantissa bits) leaves it, one of 64 or more takes the
  // next code up. ---
  {
    const auto e4m3 = [](uint32_t b, bool* ok) {
      const int e = (b >> 3) & 15, m = b & 7;
      *ok = !(e == 15 && m == 7);
      const float v = e == 0 ? std::ldexp(m / 8.0f, -6) : std::ldexp(1.0f + m / 8.0f, e - 7);
      return (b & 0x80) ? -v : v;
    };
    const auto e5m2 = [](uint32_t b, bool* ok) {
      const int e = (b >> 2) & 31, m = b & 3;
      *ok = e != 31;
      const float v = e == 0 ? std::ldexp(m / 4.0f, -14) : std::ldexp(1.0f + m / 4.0f, e - 15);
      return (b & 0x80) ? -v : v;
    };
    for (int bf8 = 0; bf8 < 2; ++bf8) {
      std::vector<uint32_t> bytes;
      for (uint32_t b = 0; b < 256; ++b) {
        bool ok;
        (bf8 ? e5m2 : e4m3)(b, &ok);
        if (ok) bytes.push_back(b);
      }
      std::vector<uint32_t> pk(N), want(N), single(N), seed_lo(N), seed_hi(N), z(N, 0);
      for (int i = 0; i < N; ++i) {
        bool ok;
        const uint32_t b0 = bytes[rnd32() % bytes.size()], b1 = bytes[rnd32() % bytes.size()];
        pk[i] = uint32_t{f2h((bf8 ? e5m2 : e4m3)(b0, &ok))} | uint32_t{f2h((bf8 ? e5m2 : e4m3)(b1, &ok))} << 16;
        want[i] = b0 | b1 << 8;
        const uint32_t sb = bytes[rnd32() % (bytes.size() - 4)];
        const uint32_t sbyte = (sb & 0x7F) >= (bf8 ? 0x7B : 0x7E) ? sb & 0x70 : sb;   // not the largest magnitudes: past them a seed overflows
        single[i] = uint32_t{f2h((bf8 ? e5m2 : e4m3)(sbyte, &ok))};
        seed_lo[i] = (rnd32() % (bf8 ? 128 : 64)) << (bf8 ? 24 : 25);   // under half an ulp of the 8-bit format
      }
      run3(bf8 ? "v_cvt_pk_bf8_f16" : "v_cvt_pk_fp8_f16", bf8 ? k_cvt_pk_bf8_f16 : k_cvt_pk_fp8_f16, pk, pk, pk,
           [&](uint32_t a, uint32_t, uint32_t) {
             const uint32_t lo = a & 0xFFFF, hi = a >> 16;
             uint32_t r = 0;
             for (int h = 0; h < 2; ++h) {
               const uint32_t half = h ? hi : lo;
               for (uint32_t b : bytes) {
                 bool ok;
                 if (f2h((bf8 ? e5m2 : e4m3)(b, &ok)) == half) { r |= b << (8 * h); break; }
               }
             }
             return r;
           }, 32, false, false, &failed);
      if (!bf8)
        run3("v_cvt_pk_fp8_f16 into the high half", k_cvt_pk_fp8_f16_hi, pk, pk, pk, [&](uint32_t a, uint32_t, uint32_t) {
          uint32_t r = 0;
          for (int h = 0; h < 2; ++h)
            for (uint32_t b : bytes) {
              bool ok;
              if (f2h(e4m3(b, &ok)) == ((a >> (16 * h)) & 0xFFFF)) { r |= b << (8 * h); break; }
            }
          return r << 16;
        }, 32, false, false, &failed);
      run3(bf8 ? "v_cvt_sr_bf8_f16 (a small seed)" : "v_cvt_sr_fp8_f16 (a small seed)", bf8 ? k_cvt_sr_bf8_f16 : k_cvt_sr_fp8_f16, single, seed_lo, z,
           [&](uint32_t a, uint32_t, uint32_t) {
             for (uint32_t b : bytes) {
               bool ok;
               if (f2h((bf8 ? e5m2 : e4m3)(b, &ok)) == (a & 0xFFFF)) return b;
             }
             return 0xFFFFu;
           }, 32, false, false, &failed);
    }
  }

  // --- stochastic packs: the seed's halves are added to the floats' bits ---
  {
    std::vector<uint32_t> A2(N), B2(N), Z(N);
    for (int i = 0; i < N; ++i) A2[i] = fb((static_cast<int>(rnd32() % 4001) - 2000) / 64.0f), B2[i] = fb((static_cast<int>(rnd32() % 4001) - 2000) / 64.0f), Z[i] = rnd32();
    run3("v_cvt_sr_pk_bf16_f32", k_cvt_sr_pk_bf16_f32, A2, B2, Z, [](uint32_t a, uint32_t b, uint32_t seed) {
      return ((a + (seed & 0xFFFF)) >> 16) | ((b + (seed >> 16)) & 0xFFFF0000u);
    }, 32, false, false, &failed);
    run3("v_cvt_sr_pk_f16_f32", k_cvt_sr_pk_f16_f32, A2, B2, Z, [](uint32_t a, uint32_t b, uint32_t seed) {
      return uint32_t{f2h(bf(a + (seed & 0xFFFF)))} | uint32_t{f2h(bf(b + (seed >> 16)))} << 16;
    }, 32, false, false, &failed);
  }

  // --- cube faces ---
  {
    const auto cube = [](const char* what, uint32_t a, uint32_t b, uint32_t c) -> uint32_t {
      const float x = bf(a), y = bf(b), z = bf(c);
      const bool zm = std::fabs(z) >= std::fabs(x) && std::fabs(z) >= std::fabs(y), ym = std::fabs(y) >= std::fabs(x);
      const std::string w = what;
      if (w == "id") return fb(zm ? (z < 0 ? 5.f : 4.f) : ym ? (y < 0 ? 3.f : 2.f) : (x < 0 ? 1.f : 0.f));
      if (w == "sc") return fb(zm ? (z < 0 ? -x : x) : ym ? x : (x < 0 ? z : -z));
      if (w == "tc") return fb(zm ? -y : ym ? (y < 0 ? -z : z) : -y);
      return fb((zm ? z : ym ? y : x) * 2.0f);
    };
    std::vector<uint32_t> X(N), Y(N), Z(N);
    for (int i = 0; i < N; ++i) X[i] = fb((static_cast<int>(rnd32() % 41) - 20) / 4.0f), Y[i] = fb((static_cast<int>(rnd32() % 41) - 20) / 4.0f), Z[i] = fb((static_cast<int>(rnd32() % 41) - 20) / 4.0f);
    run3("v_cubeid_f32", k_cubeid_f32, X, Y, Z, [&](uint32_t a, uint32_t b, uint32_t c) { return cube("id", a, b, c); }, 32, false, false, &failed);
    run3("v_cubesc_f32", k_cubesc_f32, X, Y, Z, [&](uint32_t a, uint32_t b, uint32_t c) { return cube("sc", a, b, c); }, 32, false, false, &failed);
    run3("v_cubetc_f32", k_cubetc_f32, X, Y, Z, [&](uint32_t a, uint32_t b, uint32_t c) { return cube("tc", a, b, c); }, 32, false, false, &failed);
    run3("v_cubema_f32", k_cubema_f32, X, Y, Z, [&](uint32_t a, uint32_t b, uint32_t c) { return cube("ma", a, b, c); }, 32, false, false, &failed);
  }

  // --- the lookup-table permutes ---
  {
    std::vector<uint32_t> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) A[i] = rnd32(), B[i] = rnd32(), C[i] = rnd32();
    uint32_t* a = up(A);
    uint32_t* b = up(B);
    uint32_t* c = up(C);
    uint32_t* o = nullptr;
    CHECK(hipMalloc(&o, 4608 * 4));
    k_perm_pk16<<<N / 32, 32>>>(a, b, c, o);
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out(4608);
    CHECK(hipMemcpy(out.data(), o, 4608 * 4, hipMemcpyDeviceToHost));
    int w4 = 0, w6 = 0, w8 = 0;
    for (int i = 0; i < N; ++i) {
      const uint32_t lo = A[i], hi = B[i];
      const uint64_t pair0 = uint64_t{A[i]} | uint64_t{B[i]} << 32, pair1 = uint64_t{C[i]} | uint64_t{A[(i + 5) % 512]} << 32,
                     idx = uint64_t{C[(i + 3) % 512]} | uint64_t{B[(i + 9) % 512]} << 32;
      // table as one wide number, the first source on top
      const unsigned __int128 t4 = (static_cast<unsigned __int128>(lo) << 32) | hi;
      const unsigned __int128 t6 = (static_cast<unsigned __int128>(lo) << 64) | pair1;
      const unsigned __int128 t8 = (static_cast<unsigned __int128>(pair0) << 64) | pair1;
      unsigned __int128 r4 = 0, r6 = 0, r8 = 0;
      for (int k = 0; k < 16; ++k) {
        const int n = static_cast<int>((idx >> (4 * k)) & 15);
        r4 |= ((t4 >> (4 * n)) & 0xF) << (4 * k);
        r6 |= ((t6 >> (6 * n)) & 0x3F) << (6 * k);
        r8 |= ((t8 >> (8 * n)) & 0xFF) << (8 * k);
      }
      w4 += out[i] != static_cast<uint32_t>(r4) || out[i + 512] != static_cast<uint32_t>(r4 >> 32);
      for (int k = 0; k < 3; ++k) w6 += out[i + 1024 + 512 * k] != static_cast<uint32_t>(r6 >> (32 * k));
      for (int k = 0; k < 4; ++k) w8 += out[i + 2560 + 512 * k] != static_cast<uint32_t>(r8 >> (32 * k));
    }
    report("v_perm_pk16_b4_u4", w4, N, &failed);
    report("v_perm_pk16_b6_u4", w6, N, &failed);
    report("v_perm_pk16_b8_u4", w8, N, &failed);
  }

  // --- v_swap_b32: both registers are written ---
  {
    std::vector<uint32_t> X(N), Y(N);
    for (int i = 0; i < N; ++i) X[i] = rnd32(), Y[i] = rnd32();
    uint32_t* x = up(X);
    uint32_t* y = up(Y);
    k_swap_b32<<<N / 32, 32>>>(x, y, y, dout);
    CHECK(hipDeviceSynchronize());
    const auto out = fetch(1024);
    int wrong = 0;
    for (int i = 0; i < N; ++i) wrong += out[i] != Y[i] || out[i + 512] != X[i];
    report("v_swap_b32", wrong, N, &failed);
  }

  // --- fused multiply-adds with a 64-bit constant ---
  {
    std::vector<uint32_t> xlo(N), xhi(N), ylo(2 * N);
    std::vector<double> xs(N), ys(N);
    for (int i = 0; i < N; ++i) {
      xs[i] = (static_cast<int>(rnd32() % 2001) - 1000) / 16.0, ys[i] = (static_cast<int>(rnd32() % 2001) - 1000) / 8.0;
      uint64_t xb, yb;
      std::memcpy(&xb, &xs[i], 8), std::memcpy(&yb, &ys[i], 8);
      xlo[i] = static_cast<uint32_t>(xb), xhi[i] = static_cast<uint32_t>(xb >> 32);
      ylo[i] = static_cast<uint32_t>(yb), ylo[i + N] = static_cast<uint32_t>(yb >> 32);
    }
    uint32_t* a = up(xlo);
    uint32_t* b = up(xhi);
    uint32_t* c = up(ylo);
    uint32_t* o = nullptr;
    CHECK(hipMalloc(&o, 4096 * 4));
    k_fma_k64<<<N / 32, 32>>>(a, b, c, o);
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out(4096);
    CHECK(hipMemcpy(out.data(), o, 4096 * 4, hipMemcpyDeviceToHost));
    const auto get = [&](int part, int i) { return uint64_t{out[i + 1024 * part]} | uint64_t{out[i + 1024 * part + 512]} << 32; };
    int w1 = 0, w2 = 0, w3 = 0, w4 = 0;
    const double k_mk2 = bd(0x3ff0000000000001ull), k_ak2 = bd(0x4008000000000001ull);
    for (int i = 0; i < N; ++i) {
      w1 += get(0, i) != db(std::fma(xs[i], 2.0, ys[i]));
      w2 += get(1, i) != db(std::fma(xs[i], ys[i], 3.0));
      w3 += get(2, i) != db(std::fma(xs[i], k_mk2, ys[i]));
      w4 += get(3, i) != db(std::fma(xs[i], ys[i], k_ak2));
    }
    report("v_fmamk_f64 with a constant", w1, N, &failed);
    report("v_fmaak_f64 with a constant", w2, N, &failed);
    report("v_fmamk_f64 with a lit64 constant", w3, N, &failed);
    report("v_fmaak_f64 with a lit64 constant", w4, N, &failed);
  }

  // --- doubles ---
  {
    std::vector<uint32_t> xlo(N), xhi(N), ylo(2 * N);
    std::vector<uint64_t> xs(N), ys(N);
    const double special[] = {0.0, -0.0, 1.0, -1.0, 2.5, -3.0, INFINITY, -INFINITY, NAN, 4.0, 1e300, -1e-300, 9.0, 0.25};
    for (int i = 0; i < N; ++i) {
      xs[i] = i < 14 * 14 ? db(special[i / 14]) : (rnd32() & 1 ? db(static_cast<double>(rnd32() % 100000) / 64.0) : (uint64_t{rnd32()} << 32 | rnd32()));
      ys[i] = i < 14 * 14 ? db(special[i % 14]) : db(static_cast<double>(static_cast<int>(rnd32() % 2001) - 1000) / 8.0);
      if (i == 20) xs[i] = 0x7FF0000000000001ull;   // a signaling NaN
      if (i == 30) ys[i] = 0xFFF0000000000123ull;
      xlo[i] = static_cast<uint32_t>(xs[i]); xhi[i] = static_cast<uint32_t>(xs[i] >> 32);
      ylo[i] = static_cast<uint32_t>(ys[i]); ylo[i + N] = static_cast<uint32_t>(ys[i] >> 32);
    }
    uint32_t* a = up(xlo);
    uint32_t* b = up(xhi);
    uint32_t* c = up(ylo);
    uint32_t* o = nullptr;
    CHECK(hipMalloc(&o, 3072 * 4 * 2));
    k_f64<<<N / 32, 32>>>(a, b, c, o);
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out(3072);
    CHECK(hipMemcpy(out.data(), o, 3072 * 4, hipMemcpyDeviceToHost));
    int wrong_min = 0, wrong_max = 0, wrong_sqrt = 0;
    for (int i = 0; i < N; ++i) {
      const uint64_t mn = uint64_t{out[i]} | uint64_t{out[i + 512]} << 32, mx = uint64_t{out[i + 1024]} | uint64_t{out[i + 1536]} << 32;
      const uint64_t sq = uint64_t{out[i + 2048]} | uint64_t{out[i + 2560]} << 32;
      wrong_min += mn != ref_minmax<11, 52>(xs[i], ys[i], false);
      wrong_max += mx != ref_minmax<11, 52>(xs[i], ys[i], true);
      const double s = std::sqrt(bd(xs[i]));
      wrong_sqrt += !(sq == db(s) || (std::isnan(s) && std::isnan(bd(sq))));
    }
    report("v_minimum_f64", wrong_min, N, &failed);
    report("v_maximum_f64", wrong_max, N, &failed);
    report("v_sqrt_f64", wrong_sqrt, N, &failed);
  }

  // --- scalar ---
  {
    const auto run_s = [&](const char* what, void (*k)(const uint32_t*, const uint32_t*, uint32_t*), const std::vector<uint32_t>& A, const std::vector<uint32_t>& B, auto ref, bool any_nan, int mask_bits) {
      uint32_t* a = up(A);
      uint32_t* b = up(B);
      k<<<1, 32>>>(a, b, dout);
      if (hipDeviceSynchronize() != hipSuccess) { std::printf("FAIL %s: launch failed\n", what); ++failed; return; }
      const auto out = fetch(N);
      int wrong = 0;
      for (int i = 0; i < N; ++i) {
        const uint32_t m = mask_bits == 16 ? 0xFFFFu : 0xFFFFFFFFu, want = ref(A[i], B[i]) & m, got = out[i] & m;
        wrong += !(want == got || (any_nan && (mask_bits == 16 ? is_nan16(want) && is_nan16(got) : is_nan32(want) && is_nan32(got))));
      }
      hipFree(a); hipFree(b);
      report(what, wrong, N, &failed);
    };
    run_s("s_minimum_f32", k_s_minimum_f32, F, G, [&](uint32_t a, uint32_t b) { return mm32(a, b, false); }, false, 32);
    run_s("s_maximum_f32", k_s_maximum_f32, F, G, [&](uint32_t a, uint32_t b) { return mm32(a, b, true); }, false, 32);
    run_s("s_minimum_f16", k_s_minimum_f16, H, J, [&](uint32_t a, uint32_t b) { return mm16(a, b, false); }, false, 16);
    run_s("s_maximum_f16", k_s_maximum_f16, H, J, [&](uint32_t a, uint32_t b) { return mm16(a, b, true); }, false, 16);
    run_s("s_ceil_f16", k_s_ceil_f16, H, J, [](uint32_t a, uint32_t) { return uint32_t{f2h(std::ceil(h2f(a & 0xFFFF)))}; }, true, 16);
    run_s("s_floor_f16", k_s_floor_f16, H, J, [](uint32_t a, uint32_t) { return uint32_t{f2h(std::floor(h2f(a & 0xFFFF)))}; }, true, 16);
    run_s("s_trunc_f16", k_s_trunc_f16, H, J, [](uint32_t a, uint32_t) { return uint32_t{f2h(std::trunc(h2f(a & 0xFFFF)))}; }, true, 16);
    run_s("s_rndne_f16", k_s_rndne_f16, H, J, [](uint32_t a, uint32_t) { return uint32_t{f2h(std::nearbyint(h2f(a & 0xFFFF)))}; }, true, 16);
    run_s("s_cvt_pk_rtz_f16_f32", k_s_cvt_pk_rtz, F, G, [](uint32_t a, uint32_t b) {
      // Round toward zero: the nearest half, stepped back where it overshot.
      const auto rtz = [](float x) {
        uint16_t h = f2h(x);
        if (std::isnan(x)) return h;
        if (std::fabs(h2f(h)) > std::fabs(x)) --h;
        return h;
      };
      return uint32_t{rtz(bf(a))} | uint32_t{rtz(bf(b))} << 16;
    }, false, 32);
    run_s("s_quadmask_b32", k_s_quadmask_b32, I, I2, [](uint32_t a, uint32_t) {
      uint32_t r = 0;
      for (int i = 0; i < 8; ++i) r |= uint32_t{((a >> (4 * i)) & 0xF) != 0} << i;
      return r;
    }, false, 32);
    {
      uint32_t* a = up(I);
      k_s_bitreplicate<<<1, 32>>>(a, a, dout);
      CHECK(hipDeviceSynchronize());
      const auto out = fetch(1024);
      int wrong = 0;
      for (int i = 0; i < N; ++i) {
        uint64_t r = 0;
        for (int b = 0; b < 32; ++b) r |= (uint64_t{3} * ((I[i] >> b) & 1)) << (2 * b);
        wrong += out[i] != static_cast<uint32_t>(r) || out[i + 512] != static_cast<uint32_t>(r >> 32);
      }
      report("s_bitreplicate_b64_b32", wrong, N, &failed);
    }
  }
  {
    std::vector<uint32_t> A(512);
    for (auto& x : A) x = rnd32() & 7 ? rnd32() : (rnd32() & 1 ? 0u : 0xFFFFFFFFu);
    A[2] = 0; A[3] = 0;   // all zero
    A[4] = 0xFFFFFFFFu; A[5] = 0xFFFFFFFFu;   // all one
    uint32_t* a = up(A);
    k_s_pairs<<<1, 32>>>(a, a, dout);
    CHECK(hipDeviceSynchronize());
    const auto out = fetch(1024);
    int wq = 0, wc = 0, w0 = 0, w1 = 0;
    for (int i = 0; i < 256; ++i) {
      const uint64_t x = uint64_t{A[2 * i]} | uint64_t{A[2 * i + 1]} << 32;
      uint32_t q = 0;
      for (int g = 0; g < 16; ++g) q |= uint32_t{((x >> (4 * g)) & 0xF) != 0} << g;
      int cls = -1;
      for (int b = 1; b < 64; ++b)
        if (((x >> (63 - b)) & 1) != (x >> 63)) { cls = b; break; }
      const uint64_t s0 = x & ~(uint64_t{1} << (A[2 * i] & 63)), s1 = x | (uint64_t{1} << (A[2 * i + 1] & 63));
      wq += out[4 * i] != q;
      wc += static_cast<int>(out[4 * i + 1]) != cls;
      w0 += out[4 * i + 2] != (static_cast<uint32_t>(s0) ^ static_cast<uint32_t>(s0 >> 32));
      w1 += out[4 * i + 3] != (static_cast<uint32_t>(s1) ^ static_cast<uint32_t>(s1 >> 32));
    }
    report("s_quadmask_b64", wq, 256, &failed);
    report("s_cls_i32_i64", wc, 256, &failed);
    report("s_bitset0_b64", w0, 256, &failed);
    report("s_bitset1_b64", w1, 256, &failed);
  }

  // --- the EXEC-writing scalar operations ---
  {
    std::vector<uint32_t> X(128), E(128);
    for (int i = 0; i < 128; ++i) X[i] = rnd32(), E[i] = i % 5 == 0 ? 0xFFFFFFFFu : rnd32() | 1;
    uint32_t* x = up(X);
    uint32_t* e = up(E);
    struct { const char* name; void (*k)(const uint32_t*, const uint32_t*, uint32_t*); int kind; } ops[] = {
        {"s_nand_saveexec_b32", k_nand32, 0}, {"s_nor_saveexec_b32", k_nor32, 1}, {"s_xnor_saveexec_b32", k_xnor32, 2},
        {"s_and_not0_wrexec_b32", k_wr0_32, 3}, {"s_and_not1_wrexec_b32", k_wr1_32, 4}};
    for (const auto& op : ops) {
      op.k<<<1, 32>>>(x, e, dout);
      CHECK(hipDeviceSynchronize());
      const auto out = fetch(384);
      int wrong = 0;
      for (int i = 0; i < 128; ++i) {
        const uint32_t s = X[i], ex = E[i];
        const uint32_t res = op.kind == 0 ? ~(s & ex) : op.kind == 1 ? ~(s | ex) : op.kind == 2 ? ~(s ^ ex) : op.kind == 3 ? (~s & ex) : (s & ~ex);
        const uint32_t dst = op.kind >= 3 ? res : ex;
        wrong += out[3 * i] != dst || out[3 * i + 1] != res || out[3 * i + 2] != (res != 0);
      }
      report(op.name, wrong, 128, &failed);
    }
  }

  // --- DPP8, and a comparison with a shuffled first operand ---
  {
    std::vector<uint32_t> X(32), Y(32);
    for (int i = 0; i < 32; ++i) X[i] = 1000 + 7 * i, Y[i] = 50000 + i;
    uint32_t* x = up(X);
    uint32_t* y = up(Y);
    k_dpp8<<<1, 32>>>(x, y, dout);
    CHECK(hipDeviceSynchronize());
    auto out = fetch(64);
    const int sel[8] = {3, 0, 5, 2, 7, 4, 1, 6};
    int wrong = 0, wrong_fi = 0;
    for (int l = 0; l < 32; ++l) {
      const bool on = l % 3 != 0;
      const int from = (l & ~7) + sel[l & 7];
      const bool from_on = from % 3 != 0;
      wrong += out[l] != (on ? (from_on ? X[from] : 0u) + Y[l] : 0u);
      wrong_fi += out[32 + l] != (on ? X[from] + Y[l] : 0u);
    }
    report("v_add_nc_u32 with dpp8", wrong, 32, &failed);
    report("v_add_nc_u32 with dpp8 and fi", wrong_fi, 32, &failed);
    std::vector<uint32_t> X2(32), Y2(32);
    for (int i = 0; i < 32; ++i) X2[i] = rnd32() % 100, Y2[i] = rnd32() % 100;
    x = up(X2);
    y = up(Y2);
    k_dpp_cmp<<<1, 32>>>(x, y, dout);
    CHECK(hipDeviceSynchronize());
    out = fetch(2);
    uint32_t want8 = 0, want16 = 0;
    for (int l = 0; l < 32; ++l) {
      want8 |= uint32_t{X2[(l & ~7) + (7 - (l & 7))] < Y2[l]} << l;
      want16 |= uint32_t{X2[l ^ 1] < Y2[l]} << l;
    }
    if (out[0] != want8 || out[1] != want16) std::printf("  dpp compares: dpp8 %08x (want %08x), row_xmask %08x (want %08x)\n", out[0], want8, out[1], want16);
    report("v_cmp_lt_u32 with dpp8", out[0] != want8, 1, &failed);
    report("v_cmp_lt_u32 with row_xmask", out[1] != want16, 1, &failed);
  }

  std::printf("alu1250: %d failed\n", failed);
  return failed != 0;
}
