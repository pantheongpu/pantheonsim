// One lane-wise operation per kernel, for the cross-architecture census (test_amd_gcn_cross_arch): the same kernel is built for every AMD
// target and run on the same edge-case operands (NaNs with payloads, denormals, infinities, signed zeros), and where the targets'
// results differ the test says which side is the hardware's. Built for gfx90a, gfx942, gfx950, gfx1030, gfx1100, gfx1201 and gfx1250
// by build.sh. Every kernel is out[i] = f(a[i], b[i], c[i]) with i the global work-item; f is one builtin, which is one instruction
// (or a short fixed sequence) on every target.
typedef _Float16 half;
typedef unsigned u32;
typedef unsigned long long u64;

static inline int gid(void) {
  return (int)(__builtin_amdgcn_workgroup_id_x() * 64u + __builtin_amdgcn_workitem_id_x());
}
static inline float F(u32 x) { return __builtin_bit_cast(float, x); }
static inline u32 U(float x) { return __builtin_bit_cast(u32, x); }
static inline half H(unsigned short x) { return __builtin_bit_cast(half, x); }
static inline unsigned short UH(half x) { return __builtin_bit_cast(unsigned short, x); }

// f32 in, f32 out (a, b, c are the operands' bits; the output is the result's bits)
#define K3(name, expr) \
  __attribute__((amdgpu_kernel)) void name(const u32* pa, const u32* pb, const u32* pc, u32* out, int n) { \
    int i = gid(); \
    if (i >= n) return; \
    float a = F(pa[i]), b = F(pb[i]), c = F(pc[i]); \
    (void)a; (void)b; (void)c; \
    out[i] = U(expr); \
  }
// f32 in, 32-bit integer out
#define KI(name, expr) \
  __attribute__((amdgpu_kernel)) void name(const u32* pa, const u32* pb, const u32* pc, u32* out, int n) { \
    int i = gid(); \
    if (i >= n) return; \
    float a = F(pa[i]), b = F(pb[i]), c = F(pc[i]); \
    u32 ua = pa[i], ub = pb[i], uc = pc[i]; \
    (void)a; (void)b; (void)c; (void)ua; (void)ub; (void)uc; \
    out[i] = (u32)(expr); \
  }
// f16 in (the low halves of a, b, c), f16 out
#define KH(name, expr) \
  __attribute__((amdgpu_kernel)) void name(const u32* pa, const u32* pb, const u32* pc, u32* out, int n) { \
    int i = gid(); \
    if (i >= n) return; \
    half a = H((unsigned short)pa[i]), b = H((unsigned short)pb[i]), c = H((unsigned short)pc[i]); \
    (void)a; (void)b; (void)c; \
    out[i] = UH(expr); \
  }
// f64 in (a's bits and b's bits as the two halves), f64 out in two words
#define KD(name, expr) \
  __attribute__((amdgpu_kernel)) void name(const u32* pa, const u32* pb, const u32* pc, u32* out, int n) { \
    int i = gid(); \
    if (i >= n) return; \
    double a = __builtin_bit_cast(double, ((u64)pb[i] << 32) | pa[i]), b = __builtin_bit_cast(double, ((u64)pc[i] << 32) | pb[i]); \
    double c = __builtin_bit_cast(double, ((u64)pa[i] << 32) | pc[i]); \
    (void)a; (void)b; (void)c; \
    u64 r = __builtin_bit_cast(u64, (double)(expr)); \
    out[2 * i] = (u32)r; \
    out[2 * i + 1] = (u32)(r >> 32); \
  }

K3(k_add, a + b)
K3(k_mul, a * b)
K3(k_fma, __builtin_fmaf(a, b, c))
K3(k_min, __builtin_fminf(a, b))
K3(k_max, __builtin_fmaxf(a, b))
K3(k_med3, __builtin_amdgcn_fmed3f(a, b, c))
K3(k_rcp, __builtin_amdgcn_rcpf(a))
K3(k_rsq, __builtin_amdgcn_rsqf(a))
K3(k_sqrt, __builtin_amdgcn_sqrtf(a))
K3(k_log, __builtin_amdgcn_logf(a))
K3(k_exp, __builtin_amdgcn_exp2f(a))
K3(k_sin, __builtin_amdgcn_sinf(a))
K3(k_cos, __builtin_amdgcn_cosf(a))
K3(k_fract, __builtin_amdgcn_fractf(a))
K3(k_frexp_mant, __builtin_amdgcn_frexp_mantf(a))
K3(k_canon, __builtin_canonicalizef(a))
K3(k_trunc, __builtin_truncf(a))
K3(k_floor, __builtin_floorf(a))
K3(k_ceil, __builtin_ceilf(a))
K3(k_rint, __builtin_rintf(a))
K3(k_ldexp, __builtin_amdgcn_ldexpf(a, (int)U(b) % 300 - 150))
K3(k_cvt_f16_rt, (float)(half)a)
K3(k_cvt_bf16_rt, (float)(__bf16)a)
K3(k_cubeid, __builtin_amdgcn_cubeid(a, b, c))
K3(k_cubesc, __builtin_amdgcn_cubesc(a, b, c))
K3(k_cubetc, __builtin_amdgcn_cubetc(a, b, c))
K3(k_cubema, __builtin_amdgcn_cubema(a, b, c))
K3(k_fdiv, a / b)
K3(k_sqrt_ieee, __builtin_sqrtf(a))
KI(k_f2i, (int)a)
KI(k_f2u, (unsigned)a)
KI(k_class, __builtin_amdgcn_class(a, (int)(ub & 0x3ff)))
KI(k_frexp_exp, __builtin_amdgcn_frexp_expf(a))
KI(k_cmp_lt, a < b)
KI(k_cmp_nlt, !(a < b))
KI(k_cmp_eq, a == b)
KI(k_cmp_ord, a == a && b == b)
KI(k_cvt_pkrtz_bits, __builtin_bit_cast(u32, __builtin_amdgcn_cvt_pkrtz(a, b)))
KH(k_h_add, a + b)
KH(k_h_mul, a * b)
KH(k_h_fma, __builtin_fmaf16(a, b, c))
KH(k_h_min, __builtin_fminf16(a, b))
KH(k_h_max, __builtin_fmaxf16(a, b))
KH(k_h_rcp, __builtin_amdgcn_rcph(a))
KH(k_h_rsq, __builtin_amdgcn_rsqh(a))
KH(k_h_sqrt, __builtin_amdgcn_sqrth(a))
KH(k_h_exp, __builtin_elementwise_exp2(a))
KH(k_h_log, __builtin_elementwise_log2(a))
KH(k_h_sin, __builtin_amdgcn_sinh(a))
KH(k_h_cos, __builtin_amdgcn_cosh(a))
KD(k_d_add, a + b)
KD(k_d_mul, a * b)
KD(k_d_fma, __builtin_fma(a, b, c))
KD(k_d_div, a / b)
KD(k_d_sqrt, __builtin_sqrt(a))
KD(k_d_rcp, __builtin_amdgcn_rcp(a))
KD(k_d_rsq, __builtin_amdgcn_rsq(a))
KD(k_d_min, __builtin_fmin(a, b))
KD(k_d_max, __builtin_fmax(a, b))
KD(k_d_frexp_mant, __builtin_amdgcn_frexp_mant(a))
KD(k_d_fract, __builtin_amdgcn_fract(a))
KD(k_d_rint, __builtin_rint(a))
KD(k_d_trunc, __builtin_trunc(a))
KD(k_d_f2f, (double)(float)a)
KD(k_d_trig_preop, __builtin_amdgcn_trig_preop(a, (int)__builtin_bit_cast(u64, b) & 3))

// ---- integer, byte and dot-product forms (operands as raw 32-bit values)
KI(k_i_perm, __builtin_amdgcn_perm(ua, ub, uc))
KI(k_i_alignbit, __builtin_amdgcn_alignbit(ua, ub, uc))
KI(k_i_alignbyte, __builtin_amdgcn_alignbyte(ua, ub, uc))
KI(k_i_sad, __builtin_amdgcn_sad_u8(ua, ub, uc))
KI(k_i_msad, __builtin_amdgcn_msad_u8(ua, ub, uc))
KI(k_i_lerp, __builtin_amdgcn_lerp(ua, ub, uc))
KI(k_i_mulhi_u, (u32)(((u64)ua * ub) >> 32))
KI(k_i_cvt_pk_u8, __builtin_amdgcn_cvt_pk_u8_f32(a, (int)(ub & 3), uc))
KI(k_i_bcnt, __builtin_popcount(ua) + __builtin_ctz(ub | 0x80000000u) + (ua ? __builtin_clz(ua) : 32))
#if __has_builtin(__builtin_amdgcn_sdot4)
KI(k_i_sdot4, __builtin_amdgcn_sdot4((int)ua, (int)ub, (int)uc, 0))
KI(k_i_sdot4_clamp, __builtin_amdgcn_sdot4((int)ua, (int)ub, (int)uc, 1))
KI(k_i_udot4, __builtin_amdgcn_udot4(ua, ub, uc, 0))
#endif
#if __has_builtin(__builtin_amdgcn_fdot2)
typedef _Float16 half2v __attribute__((ext_vector_type(2)));
KI(k_dot2_f32_f16, U(__builtin_amdgcn_fdot2(__builtin_bit_cast(half2v, ua), __builtin_bit_cast(half2v, ub), c, 0)))
KI(k_dot2_f32_f16_clamp, U(__builtin_amdgcn_fdot2(__builtin_bit_cast(half2v, ua), __builtin_bit_cast(half2v, ub), c, 1)))
#endif

// ---- 8-bit floats: gfx942's are the FNUZ forms, gfx950's, gfx12's and gfx1250's the OCP ones
#if __has_builtin(__builtin_amdgcn_cvt_f32_fp8)
KI(k_fp8_to_f32, U(__builtin_amdgcn_cvt_f32_fp8((int)(ua & 0xff), 0)))
KI(k_bf8_to_f32, U(__builtin_amdgcn_cvt_f32_bf8((int)(ua & 0xff), 0)))
KI(k_f32_to_fp8, __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, 0) & 0xffff)
KI(k_f32_to_bf8, __builtin_amdgcn_cvt_pk_bf8_f32(a, b, 0, 0) & 0xffff)
KI(k_f32_to_fp8_sr, __builtin_amdgcn_cvt_sr_fp8_f32(a, (int)ub, 0, 0) & 0xff)
#endif

// ---- wave-wide forms: the answer depends on the wave's width (64 on CDNA, 32 on RDNA and gfx1250)
#define KW(name, expr) \
  __attribute__((amdgpu_kernel)) void name(const u32* pa, const u32* pb, const u32* pc, u32* out, int n) { \
    int i = gid(); \
    u32 ua = pa[i], ub = pb[i], uc = pc[i]; \
    float a = F(ua); (void)a; (void)ub; (void)uc; \
    u64 r = (u64)(expr); \
    out[2 * i] = (u32)r; \
    out[2 * i + 1] = (u32)(r >> 32); \
  }
#if __AMDGCN_WAVEFRONT_SIZE__ == 64
#define BALLOT(p) __builtin_amdgcn_ballot_w64(p)
#else
#define BALLOT(p) ((u64)__builtin_amdgcn_ballot_w32(p))
#endif
KW(k_w_size, __builtin_amdgcn_wavefrontsize())
KW(k_w_ballot, BALLOT((ua & 3) != 0))
KW(k_w_ballot_true, BALLOT(1))
KW(k_w_readlane, __builtin_amdgcn_readlane((int)ua, (int)(ub & 63)))
KW(k_w_readlane31, __builtin_amdgcn_readlane((int)ua, 31))
KW(k_w_readfirstlane, __builtin_amdgcn_readfirstlane(ua))
KW(k_w_mbcnt, __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0)))
KW(k_w_mbcnt_lo_only, __builtin_amdgcn_mbcnt_lo(~0u, 0))
KW(k_w_mbcnt_hi_only, __builtin_amdgcn_mbcnt_hi(~0u, 0))
KW(k_w_bpermute, __builtin_amdgcn_ds_bpermute((int)(ub << 2), (int)ua))
KW(k_w_permute, __builtin_amdgcn_ds_permute((int)(ub << 2), (int)ua))
KW(k_w_swizzle_rev, __builtin_amdgcn_ds_swizzle((int)ua, 0x041f))
KW(k_w_swizzle_rot, __builtin_amdgcn_ds_swizzle((int)ua, 0x0c1f))
KW(k_w_dpp_quad_rev, __builtin_amdgcn_update_dpp(0x55555555, (int)ua, 0x1b, 0xf, 0xf, 0))
KW(k_w_dpp_row_shr1, __builtin_amdgcn_update_dpp(0x55555555, (int)ua, 0x111, 0xf, 0xf, 0))
KW(k_w_dpp_row_shr1_bc, __builtin_amdgcn_update_dpp(0x55555555, (int)ua, 0x111, 0xf, 0xf, 1))
KW(k_w_dpp_row_mirror, __builtin_amdgcn_update_dpp(0x55555555, (int)ua, 0x140, 0xf, 0xf, 0))
KW(k_w_dpp_row_half_mirror, __builtin_amdgcn_update_dpp(0x55555555, (int)ua, 0x141, 0xf, 0xf, 0))
KW(k_w_dpp_row_ror3, __builtin_amdgcn_update_dpp(0x55555555, (int)ua, 0x123, 0xf, 0xf, 0))
KW(k_w_dpp_row_masked, __builtin_amdgcn_update_dpp(0x55555555, (int)ua, 0x101, 0x5, 0x3, 0))
#if defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)
KW(k_w_dpp_wave_shr1, __builtin_amdgcn_update_dpp(0x55555555, (int)ua, 0x138, 0xf, 0xf, 0))
KW(k_w_dpp_row_bcast15, __builtin_amdgcn_update_dpp(0x55555555, (int)ua, 0x142, 0xa, 0xf, 0))
KW(k_w_dpp_row_bcast31, __builtin_amdgcn_update_dpp(0x55555555, (int)ua, 0x143, 0xc, 0xf, 0))
#endif
#if __has_builtin(__builtin_amdgcn_permlanex16)
KW(k_w_permlanex16, __builtin_amdgcn_permlanex16(0x55555555, (int)ua, 0x76543210, 0xfedcba98, 0, 0))
KW(k_w_permlane16, __builtin_amdgcn_permlane16(0x55555555, (int)ua, 0x76543210, 0xfedcba98, 0, 0))
#endif

// ---- IEEE minimum() and maximum(), minimumNumber() and maximumNumber() written out (RDNA4 and gfx1250 have them; asm so no builtin is needed)
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1250__)
static inline float asm_maximum(float x, float y) { float r; __asm__ volatile("v_maximum_f32 %0, %1, %2" : "=v"(r) : "v"(x), "v"(y)); return r; }
static inline float asm_minimum(float x, float y) { float r; __asm__ volatile("v_minimum_f32 %0, %1, %2" : "=v"(r) : "v"(x), "v"(y)); return r; }
static inline float asm_max_num(float x, float y) { float r; __asm__ volatile("v_max_num_f32 %0, %1, %2" : "=v"(r) : "v"(x), "v"(y)); return r; }
static inline float asm_min_num(float x, float y) { float r; __asm__ volatile("v_min_num_f32 %0, %1, %2" : "=v"(r) : "v"(x), "v"(y)); return r; }
static inline float asm_max3_num(float x, float y, float z) { float r; __asm__ volatile("v_max3_num_f32 %0, %1, %2, %3" : "=v"(r) : "v"(x), "v"(y), "v"(z)); return r; }
static inline float asm_min3_num(float x, float y, float z) { float r; __asm__ volatile("v_min3_num_f32 %0, %1, %2, %3" : "=v"(r) : "v"(x), "v"(y), "v"(z)); return r; }
static inline float asm_med3_num(float x, float y, float z) { float r; __asm__ volatile("v_med3_num_f32 %0, %1, %2, %3" : "=v"(r) : "v"(x), "v"(y), "v"(z)); return r; }
K3(k_maximum, asm_maximum(a, b))
K3(k_minimum, asm_minimum(a, b))
K3(k_max_num, asm_max_num(a, b))
K3(k_min_num, asm_min_num(a, b))
K3(k_max3_num, asm_max3_num(a, b, c))
K3(k_min3_num, asm_min3_num(a, b, c))
K3(k_med3_num, asm_med3_num(a, b, c))
#endif

// ---- the matrix core: one 16x16x16 f16 instruction on pseudo-random finite halves (gfx90a, gfx942 and gfx950 have it under one name)
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16f16)
typedef _Float16 v4h __attribute__((ext_vector_type(4)));
typedef float v4f __attribute__((ext_vector_type(4)));
static inline u32 mix(u32 x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
static inline u32 rhalf(u32 s) { return ((s >> 15 & 1u) << 15) | ((10u + (s >> 10) % 11u) << 10) | (s & 0x3ffu); }
static inline u32 rfloat(u32 s) { return ((s >> 31) << 31) | ((120u + (s >> 8) % 16u) << 23) | (s & 0x7fffffu); }
__attribute__((amdgpu_kernel)) void k_mfma_f16(const u32* pa, const u32* pb, const u32* pc, u32* out, int n) {
  int i = gid();
  if (i >= n) return;
  u32 s = mix((u32)i * 8u + 1u);
  u64 av = (u64)(rhalf(mix(s + 1)) | rhalf(mix(s + 2)) << 16) | ((u64)(rhalf(mix(s + 3)) | rhalf(mix(s + 4)) << 16) << 32);
  u64 bv = (u64)(rhalf(mix(s + 5)) | rhalf(mix(s + 6)) << 16) | ((u64)(rhalf(mix(s + 7)) | rhalf(mix(s + 8)) << 16) << 32);
  v4h a = __builtin_bit_cast(v4h, av), b = __builtin_bit_cast(v4h, bv);
  v4f c = {F(rfloat(mix(s + 9))), F(rfloat(mix(s + 10))), F(rfloat(mix(s + 11))), F(rfloat(mix(s + 12)))};
  v4f r = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
  out[2 * i] = U(r[0]) ^ U(r[2]);
  out[2 * i + 1] = U(r[1]) ^ U(r[3]);
}
#endif
