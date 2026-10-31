// The lane kernels of the AMD tier, written once and compiled once per level: gcn_tier.cpp includes this file inside a
// namespace of the level's name, the second time under #pragma GCC target("avx2,fma"). Plain C++ loops over the lanes
// of a register row (n a multiple of 8, 32 or 64); every level gives the same bits, which the tier-vs-interpreter test
// holds them to. TIER_AVX2 (1 in the AVX2 level) says which intrinsics may be used.
//
// A kernel reads a, b and c at index i and writes d at index i only, so d may be one of them (`ivdep`: no lane depends on
// another). Floating-point kernels are one IEEE operation each (fp-contract is off for the whole file); where an operand is a
// NaN the handler does not use the result (it hands the instruction to the interpreter), so a NaN's payload never decides a bit.

namespace TIER_NS {

typedef float __attribute__((may_alias)) F32;
typedef double __attribute__((may_alias)) F64;
typedef uint32_t V8u __attribute__((vector_size(32)));
typedef int32_t V8i __attribute__((vector_size(32)));
typedef float V8f __attribute__((vector_size(32)));
typedef uint32_t V4u __attribute__((vector_size(16)));
typedef int32_t V4i __attribute__((vector_size(16)));
typedef float V4f __attribute__((vector_size(16)));

using K3 = void (*)(uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c, unsigned n);

#define TIER_K(name, ...)                                                                                    \
  static void name(uint32_t* d, [[maybe_unused]] const uint32_t* a, [[maybe_unused]] const uint32_t* b,      \
                   [[maybe_unused]] const uint32_t* c, unsigned n) {                                          \
    _Pragma("GCC ivdep") for (unsigned i = 0; i < n; ++i) { __VA_ARGS__; }                                          \
  }

// ---- integer
TIER_K(k_mov, d[i] = a[i])
TIER_K(k_not, d[i] = ~a[i])
TIER_K(k_add, d[i] = a[i] + b[i])
TIER_K(k_sub, d[i] = a[i] - b[i])
TIER_K(k_subrev, d[i] = b[i] - a[i])
TIER_K(k_mul_lo, d[i] = a[i] * b[i])
TIER_K(k_mul_hi_u, d[i] = static_cast<uint32_t>((static_cast<uint64_t>(a[i]) * b[i]) >> 32))
TIER_K(k_mul_hi_i,
       d[i] = static_cast<uint32_t>(static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(a[i])) *
                                                          static_cast<int32_t>(b[i])) >> 32))
TIER_K(k_mul_u24, d[i] = (a[i] & 0xFFFFFFu) * (b[i] & 0xFFFFFFu))
TIER_K(k_mul_i24, d[i] = static_cast<uint32_t>(static_cast<int32_t>(a[i] << 8) >> 8) *
                         static_cast<uint32_t>(static_cast<int32_t>(b[i] << 8) >> 8))
TIER_K(k_mad_u24, d[i] = (a[i] & 0xFFFFFFu) * (b[i] & 0xFFFFFFu) + c[i])
TIER_K(k_mad_i24, d[i] = static_cast<uint32_t>(static_cast<int32_t>(a[i] << 8) >> 8) *
                             static_cast<uint32_t>(static_cast<int32_t>(b[i] << 8) >> 8) + c[i])
TIER_K(k_and, d[i] = a[i] & b[i])
TIER_K(k_or, d[i] = a[i] | b[i])
TIER_K(k_xor, d[i] = a[i] ^ b[i])
TIER_K(k_lshlrev, d[i] = b[i] << (a[i] & 31))
TIER_K(k_lshrrev, d[i] = b[i] >> (a[i] & 31))
TIER_K(k_ashrrev, d[i] = static_cast<uint32_t>(static_cast<int32_t>(b[i]) >> (a[i] & 31)))
TIER_K(k_min_u, d[i] = a[i] < b[i] ? a[i] : b[i])
TIER_K(k_max_u, d[i] = a[i] > b[i] ? a[i] : b[i])
TIER_K(k_min_i, d[i] = static_cast<int32_t>(a[i]) < static_cast<int32_t>(b[i]) ? a[i] : b[i])
TIER_K(k_max_i, d[i] = static_cast<int32_t>(a[i]) > static_cast<int32_t>(b[i]) ? a[i] : b[i])
TIER_K(k_bfi, d[i] = (a[i] & b[i]) | (~a[i] & c[i]))
TIER_K(k_alignbit, d[i] = static_cast<uint32_t>((static_cast<uint64_t>(a[i]) << 32 | b[i]) >> (c[i] & 31)))
TIER_K(k_alignbyte, d[i] = static_cast<uint32_t>((static_cast<uint64_t>(a[i]) << 32 | b[i]) >> (8 * (c[i] & 3))))
TIER_K(k_add3, d[i] = a[i] + b[i] + c[i])
TIER_K(k_or3, d[i] = a[i] | b[i] | c[i])
TIER_K(k_xor3, d[i] = a[i] ^ b[i] ^ c[i])
TIER_K(k_and_or, d[i] = (a[i] & b[i]) | c[i])
TIER_K(k_xad, d[i] = (a[i] ^ b[i]) + c[i])
TIER_K(k_lshl_or, d[i] = (a[i] << (b[i] & 31)) | c[i])
TIER_K(k_lshl_add, d[i] = (a[i] << (b[i] & 31)) + c[i])
TIER_K(k_add_lshl, d[i] = (a[i] + b[i]) << (c[i] & 31))
TIER_K(k_bcnt, d[i] = b[i] + static_cast<uint32_t>(__builtin_popcount(a[i])))
TIER_K(k_bfe_u, {
  const uint32_t start = b[i] & 31, width = c[i] & 31;
  d[i] = width == 0 ? 0u : (a[i] >> start) & ((1u << width) - 1);
})
TIER_K(k_med3_u, {
  const uint32_t x = a[i], y = b[i], z = c[i];
  const uint32_t lo = x < y ? x : y, hi = x < y ? y : x, m = hi < z ? hi : z;
  d[i] = lo > m ? lo : m;
})
TIER_K(k_med3_i, {
  const int32_t x = static_cast<int32_t>(a[i]), y = static_cast<int32_t>(b[i]), z = static_cast<int32_t>(c[i]);
  const int32_t lo = x < y ? x : y, hi = x < y ? y : x, m = hi < z ? hi : z;
  d[i] = static_cast<uint32_t>(lo > m ? lo : m);
})
TIER_K(k_min3_u, {
  const uint32_t m = a[i] < b[i] ? a[i] : b[i];
  d[i] = m < c[i] ? m : c[i];
})
TIER_K(k_max3_u, {
  const uint32_t m = a[i] > b[i] ? a[i] : b[i];
  d[i] = m > c[i] ? m : c[i];
})
TIER_K(k_mbcnt_lo, {
  const uint32_t below = i >= 32 ? 0xFFFFFFFFu : (i ? (1u << i) - 1 : 0u);
  d[i] = b[i] + static_cast<uint32_t>(__builtin_popcount(a[i] & below));
})
TIER_K(k_mbcnt_hi, {
  const uint32_t below = i < 32 ? 0u : (1u << (i - 32)) - 1;
  d[i] = b[i] + static_cast<uint32_t>(__builtin_popcount(a[i] & below));
})
// (A float source, an integer result: v_cvt_i32_f32 and v_cvt_u32_f32, a NaN being zero and the range clamped.)
TIER_K(k_cvt_i32_f32, {
  const float f = reinterpret_cast<const F32*>(a)[i];
  d[i] = static_cast<uint32_t>(f != f ? 0 : f <= -2147483648.0f ? INT32_MIN : f >= 2147483648.0f ? INT32_MAX : static_cast<int32_t>(f));
})
TIER_K(k_cvt_u32_f32, {
  const float f = reinterpret_cast<const F32*>(a)[i];
  d[i] = f != f || f <= 0 ? 0u : f >= 4294967296.0f ? 0xFFFFFFFFu : static_cast<uint32_t>(f);
})
TIER_K(k_cvt_f32_u, reinterpret_cast<F32*>(d)[i] = static_cast<float>(a[i]))
TIER_K(k_cvt_f32_i, reinterpret_cast<F32*>(d)[i] = static_cast<float>(static_cast<int32_t>(a[i])))
TIER_K(k_cvt_f32_ubyte0, reinterpret_cast<F32*>(d)[i] = static_cast<float>(a[i] & 0xFF))
// ---- 16-bit integer operations of the CDNA parts (a 16-bit result fills the low half of the register and zeroes the high one)
TIER_K(k_add_u16, d[i] = static_cast<uint16_t>(static_cast<uint16_t>(a[i]) + static_cast<uint16_t>(b[i])))
TIER_K(k_sub_u16, d[i] = static_cast<uint16_t>(static_cast<uint16_t>(a[i]) - static_cast<uint16_t>(b[i])))
TIER_K(k_subrev_u16, d[i] = static_cast<uint16_t>(static_cast<uint16_t>(b[i]) - static_cast<uint16_t>(a[i])))
TIER_K(k_lshlrev_b16, d[i] = static_cast<uint16_t>(b[i] << (a[i] & 15)))
TIER_K(k_lshrrev_b16, d[i] = static_cast<uint16_t>(b[i]) >> (a[i] & 15))
TIER_K(k_ashrrev_i16, d[i] = static_cast<uint16_t>(static_cast<int16_t>(b[i]) >> (a[i] & 15)))
TIER_K(k_mul_lo_u16, d[i] = static_cast<uint16_t>(a[i] * b[i]))
// Leading and trailing zero counts (-1 for no set bit): v_ffbh_u32, v_ffbl_b32, v_ffbh_i32.
TIER_K(k_ffbh_u, d[i] = a[i] ? static_cast<uint32_t>(__builtin_clz(a[i])) : 0xFFFFFFFFu)
TIER_K(k_ffbl, d[i] = a[i] ? static_cast<uint32_t>(__builtin_ctz(a[i])) : 0xFFFFFFFFu)
TIER_K(k_ffbh_i, d[i] = (a[i] >> 31 ? ~a[i] : a[i]) ? static_cast<uint32_t>(__builtin_clz(a[i] >> 31 ? ~a[i] : a[i])) : 0xFFFFFFFFu)
// Packed 16-bit integers (v_pk_*_u16, _i16, _b16): a, b and c are one half of each lane's sources (zero-extended), d the half's result in the low
// 16 bits. The clamped forms hold an add, a subtract or a multiply-add to what 16 bits hold.
#define PK_S(x) static_cast<int32_t>(static_cast<int16_t>(x))
TIER_K(k_pk_mullo, d[i] = (a[i] * b[i]) & 0xFFFFu)
TIER_K(k_pk_addu, d[i] = (a[i] + b[i]) & 0xFFFFu)
TIER_K(k_pk_addi, d[i] = static_cast<uint32_t>(PK_S(a[i]) + PK_S(b[i])) & 0xFFFFu)
TIER_K(k_pk_subu, d[i] = (a[i] - b[i]) & 0xFFFFu)
TIER_K(k_pk_subi, d[i] = static_cast<uint32_t>(PK_S(a[i]) - PK_S(b[i])) & 0xFFFFu)
TIER_K(k_pk_shl, d[i] = (b[i] << (a[i] & 15)) & 0xFFFFu)
TIER_K(k_pk_shr, d[i] = b[i] >> (a[i] & 15))
TIER_K(k_pk_ashr, d[i] = static_cast<uint32_t>(PK_S(b[i]) >> (a[i] & 15)) & 0xFFFFu)
TIER_K(k_pk_maxi, d[i] = static_cast<uint32_t>(PK_S(a[i]) > PK_S(b[i]) ? PK_S(a[i]) : PK_S(b[i])) & 0xFFFFu)
TIER_K(k_pk_mini, d[i] = static_cast<uint32_t>(PK_S(a[i]) < PK_S(b[i]) ? PK_S(a[i]) : PK_S(b[i])) & 0xFFFFu)
TIER_K(k_pk_maxu, d[i] = a[i] > b[i] ? a[i] : b[i])
TIER_K(k_pk_minu, d[i] = a[i] < b[i] ? a[i] : b[i])
TIER_K(k_pk_madu, d[i] = (a[i] * b[i] + c[i]) & 0xFFFFu)
TIER_K(k_pk_madi, d[i] = static_cast<uint32_t>(PK_S(a[i]) * PK_S(b[i]) + PK_S(c[i])) & 0xFFFFu)
TIER_K(k_pk_addu_c, d[i] = a[i] + b[i] > 0xFFFFu ? 0xFFFFu : a[i] + b[i])
TIER_K(k_pk_addi_c, d[i] = static_cast<uint32_t>(PK_S(a[i]) + PK_S(b[i]) < -32768 ? -32768 : PK_S(a[i]) + PK_S(b[i]) > 32767 ? 32767 : PK_S(a[i]) + PK_S(b[i])) & 0xFFFFu)
TIER_K(k_pk_subu_c, d[i] = a[i] < b[i] ? 0u : a[i] - b[i])
TIER_K(k_pk_subi_c, d[i] = static_cast<uint32_t>(PK_S(a[i]) - PK_S(b[i]) < -32768 ? -32768 : PK_S(a[i]) - PK_S(b[i]) > 32767 ? 32767 : PK_S(a[i]) - PK_S(b[i])) & 0xFFFFu)
TIER_K(k_pk_madu_c, d[i] = a[i] * b[i] + c[i] > 0xFFFFu ? 0xFFFFu : a[i] * b[i] + c[i])
TIER_K(k_pk_madi_c, d[i] = static_cast<uint32_t>(PK_S(a[i]) * PK_S(b[i]) + PK_S(c[i]) < -32768 ? -32768 : PK_S(a[i]) * PK_S(b[i]) + PK_S(c[i]) > 32767 ? 32767 : PK_S(a[i]) * PK_S(b[i]) + PK_S(c[i])) & 0xFFFFu)
#undef PK_S
TIER_K(k_min_u16, d[i] = std::min(static_cast<uint16_t>(a[i]), static_cast<uint16_t>(b[i])))
TIER_K(k_max_u16, d[i] = std::max(static_cast<uint16_t>(a[i]), static_cast<uint16_t>(b[i])))
TIER_K(k_min_i16, d[i] = static_cast<uint16_t>(std::min(static_cast<int16_t>(a[i]), static_cast<int16_t>(b[i]))))
TIER_K(k_max_i16, d[i] = static_cast<uint16_t>(std::max(static_cast<int16_t>(a[i]), static_cast<int16_t>(b[i]))))
TIER_K(k_pack_b32_f16, d[i] = (a[i] & 0xFFFFu) | (b[i] & 0xFFFFu) << 16)
// v_bfe_i32: the field, with its top bit carried into the rest of the register.
TIER_K(k_bfe_i, {
  const uint32_t start = b[i] & 31, width = c[i] & 31;
  uint32_t out = 0;
  if (width != 0) {
    const uint32_t shifted = static_cast<uint32_t>(static_cast<int32_t>(a[i]) >> start);
    out = shifted & ((1u << width) - 1);
    if (out >> (width - 1) & 1) out |= ~((1u << width) - 1);
  }
  d[i] = out;
})
// v_perm_b32: four bytes chosen out of the eight the two sources make (the first holding the top four); selectors 8 to 11 ask for the
// sign of a byte spread over a byte, 12 for a zero byte, and more for a byte of ones.
TIER_K(k_perm, {
  const uint64_t bytes = static_cast<uint64_t>(a[i]) << 32 | b[i];
  const uint32_t sel = c[i];
  uint32_t out = 0;
  for (uint32_t k = 0; k < 4; ++k) {
    const uint32_t which = (sel >> (8 * k)) & 0xFF;
    const uint32_t byte = which < 8 ? (bytes >> (8 * which)) & 0xFF
                          : which < 12 ? ((bytes >> (8 * (2 * (which - 8) + 1) + 7)) & 1) * 0xFFu
                          : which == 12 ? 0u
                                        : 0xFFu;
    out |= byte << (8 * k);
  }
  d[i] = out;
})
// v_cvt_f16_f32: the float (source modifiers applied to it) rounded to a half, in the low half of the register.
TIER_K(k_cvt_f16_f32, d[i] = vgpu::amd::packed::float_to_half(reinterpret_cast<const F32*>(a)[i]))

// ---- binary32 (one IEEE operation each; the handler throws away a NaN result and lets the interpreter decide the bits)
#define TIER_F(name, ...)                                                                                    \
  static void name(uint32_t* d, [[maybe_unused]] const uint32_t* ua, [[maybe_unused]] const uint32_t* ub,    \
                   [[maybe_unused]] const uint32_t* uc, unsigned n) {                                         \
    [[maybe_unused]] const F32* a = reinterpret_cast<const F32*>(ua);                                                          \
    [[maybe_unused]] const F32* b = reinterpret_cast<const F32*>(ub);                                                          \
    [[maybe_unused]] const F32* c = reinterpret_cast<const F32*>(uc);                                                          \
    F32* r = reinterpret_cast<F32*>(d);                                                                       \
    _Pragma("GCC ivdep") for (unsigned i = 0; i < n; ++i) { __VA_ARGS__; }                                          \
  }
TIER_F(k_fadd, r[i] = a[i] + b[i])
TIER_F(k_fsub, r[i] = a[i] - b[i])
TIER_F(k_fsubrev, r[i] = b[i] - a[i])
TIER_F(k_fmul, r[i] = a[i] * b[i])
// a * b + c, fused: v_fma_f32, v_fmac_f32 (c is the destination), v_fmaak_f32 (c a constant); fmamk is a * K + c (K in b).
TIER_F(k_ffma, r[i] = __builtin_fmaf(a[i], b[i], c[i]))
// (v_min_f32, v_max_f32: libm's fmin and fmax, as the interpreter calls them; a NaN in one operand gives the other, both give a NaN.)
TIER_F(k_fmin, r[i] = __builtin_fminf(a[i], b[i]))
TIER_F(k_fmax, r[i] = __builtin_fmaxf(a[i], b[i]))
TIER_F(k_frcp, r[i] = 1.0f / a[i])
// (the one-input functions of the vector unit: the host's own, as the interpreter calls them; a NaN result goes back to the interpreter)
TIER_F(k_fsqrt, r[i] = __builtin_sqrtf(a[i]))
TIER_F(k_frsq, r[i] = 1.0f / __builtin_sqrtf(a[i]))
TIER_F(k_fexp, r[i] = __builtin_exp2f(a[i]))
TIER_F(k_flog, r[i] = __builtin_log2f(a[i]))
// (floor, ceil and trunc of a value of 2^23 or more in magnitude are the value, as the interpreter's floor_inline and the rest have it)
TIER_F(k_ffloor, r[i] = __builtin_fabsf(a[i]) < 8388608.0f ? __builtin_floorf(a[i]) : a[i])
TIER_F(k_fceil, r[i] = __builtin_fabsf(a[i]) < 8388608.0f ? __builtin_ceilf(a[i]) : a[i])
TIER_F(k_ftrunc, r[i] = __builtin_fabsf(a[i]) < 8388608.0f ? __builtin_truncf(a[i]) : a[i])
TIER_F(k_frndne, r[i] = __builtin_nearbyintf(a[i]))
// (x - floor(x) held below 1; an infinity gives a NaN, which goes back)
TIER_F(k_ffract, r[i] = a[i] != a[i] ? a[i] : __builtin_isinf(a[i]) ? __builtin_nanf("") : __builtin_fminf(a[i] - __builtin_floorf(a[i]), 0x1.fffffep-1f))
// v_sin_f32 and v_cos_f32 count their angle in turns. The host's sin and cos of the angle in radians (the architectures the guides
// cover keep only the fraction of a turn, which is exact in a double, so the argument stays small).
TIER_F(k_fsin_r, r[i] = static_cast<float>(__builtin_sin(static_cast<double>(a[i]) * 6.283185307179586476925286766559)))
TIER_F(k_fsin_t, {
  const double turns = static_cast<double>(a[i]);
  const double frac = turns == 0 ? turns : turns - __builtin_trunc(turns);
  r[i] = static_cast<float>(__builtin_sin(frac * 6.283185307179586476925286766559));
})
TIER_F(k_fcos_r, r[i] = static_cast<float>(__builtin_cos(static_cast<double>(a[i]) * 6.283185307179586476925286766559)))
TIER_F(k_fcos_t, {
  const double turns = static_cast<double>(a[i]);
  const double frac = turns == 0 ? turns : turns - __builtin_trunc(turns);
  r[i] = static_cast<float>(__builtin_cos(frac * 6.283185307179586476925286766559));
})
// v_cvt_f32_f16: the half in the low 16 bits of the register, widened (a NaN stays one, quieted, its payload shifted up).
TIER_F(k_fcvt_f32_f16, r[i] = vgpu::amd::packed::half_to_float(static_cast<uint16_t>(ua[i])))
// (the exponent is the second source's bits as an integer)
TIER_F(k_fldexp, r[i] = __builtin_ldexpf(a[i], static_cast<int32_t>(ub[i])))

// ---- binary64 on 64-bit lanes (the handler gathers the register pairs)
#define TIER_D(name, ...)                                                                                    \
  static void name(uint64_t* d, [[maybe_unused]] const uint64_t* ua, [[maybe_unused]] const uint64_t* ub,    \
                   [[maybe_unused]] const uint64_t* uc, unsigned n) {                                         \
    [[maybe_unused]] const F64* a = reinterpret_cast<const F64*>(ua);                                                          \
    [[maybe_unused]] const F64* b = reinterpret_cast<const F64*>(ub);                                                          \
    [[maybe_unused]] const F64* c = reinterpret_cast<const F64*>(uc);                                                          \
    F64* r = reinterpret_cast<F64*>(d);                                                                       \
    _Pragma("GCC ivdep") for (unsigned i = 0; i < n; ++i) { __VA_ARGS__; }                                          \
  }
TIER_D(k_dadd, r[i] = a[i] + b[i])
TIER_D(k_dmul, r[i] = a[i] * b[i])
TIER_D(k_dfma, r[i] = __builtin_fma(a[i], b[i], c[i]))
// (the one-input functions: the host's exact ones; floor, ceil and trunc of a value of 2^52 or more in magnitude are the value, as the interpreter's)
TIER_D(k_drcp, r[i] = 1.0 / a[i])
TIER_D(k_dsqrt, r[i] = __builtin_sqrt(a[i]))
TIER_D(k_dfloor, r[i] = __builtin_fabs(a[i]) < 4503599627370496.0 ? __builtin_floor(a[i]) : a[i])
TIER_D(k_dceil, r[i] = __builtin_fabs(a[i]) < 4503599627370496.0 ? __builtin_ceil(a[i]) : a[i])
TIER_D(k_dtrunc, r[i] = __builtin_fabs(a[i]) < 4503599627370496.0 ? __builtin_trunc(a[i]) : a[i])
TIER_D(k_drndne, r[i] = __builtin_nearbyint(a[i]))
// (minimum and maximum: a NaN result, and the two zeros of opposite sign, go to the interpreter as the float forms'.)
TIER_D(k_dmin, r[i] = __builtin_fmin(a[i], b[i]))
TIER_D(k_dmax, r[i] = __builtin_fmax(a[i], b[i]))

// The same three operations (add, multiply, fused multiply-add) straight on register pairs: the low and high words of each operand in their rows,
// the result's the same, and the lanes whose result is a NaN as the bits of the return value. Lane by lane the arithmetic is the one above.
// UM: the operands that are the same in every lane (bit k: operand k, whose value is u[k] and whose rows are not read).
using F64RowsFn = uint64_t (*)(uint32_t* rlo, uint32_t* rhi, const uint32_t* alo, const uint32_t* ahi, const uint32_t* blo, const uint32_t* bhi,
                               const uint32_t* clo, const uint32_t* chi, const uint64_t* u, unsigned n);
#if TIER_AVX2
// Eight doubles from their low and high words, as two vectors of four (lanes 0 to 3 and 4 to 7).
static inline __attribute__((always_inline)) void load_pairs(const uint32_t* lo, const uint32_t* hi, __m256d& d0, __m256d& d1) {
  const __m256i l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lo)), h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hi));
  const __m256i t0 = _mm256_unpacklo_epi32(l, h), t1 = _mm256_unpackhi_epi32(l, h);
  d0 = _mm256_castsi256_pd(_mm256_permute2x128_si256(t0, t1, 0x20));
  d1 = _mm256_castsi256_pd(_mm256_permute2x128_si256(t0, t1, 0x31));
}
// ... and back: the low and the high words of eight results.
static inline __attribute__((always_inline)) void store_pairs(uint32_t* lo, uint32_t* hi, __m256d r0, __m256d r1) {
  const __m256 a = _mm256_castpd_ps(r0), b = _mm256_castpd_ps(r1);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(lo), _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(a, b, 0x88)), 0xD8));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(hi), _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(a, b, 0xDD)), 0xD8));
}
#endif
template <int OP, int UM>
static uint64_t k_f64r(uint32_t* rlo, uint32_t* rhi, const uint32_t* alo, const uint32_t* ahi, const uint32_t* blo, const uint32_t* bhi,
                       [[maybe_unused]] const uint32_t* clo, [[maybe_unused]] const uint32_t* chi, const uint64_t* u, unsigned n) {
  uint64_t nan = 0;
#if TIER_AVX2
  const __m256d ua = _mm256_set1_pd(__builtin_bit_cast(double, u[0])), ub = _mm256_set1_pd(__builtin_bit_cast(double, u[1])),
                uc = _mm256_set1_pd(__builtin_bit_cast(double, u[2]));
  for (unsigned i = 0; i < n; i += 8) {
    __m256d a0 = ua, a1 = ua, b0 = ub, b1 = ub, c0 = uc, c1 = uc;
    if constexpr (!(UM & 1)) load_pairs(alo + i, ahi + i, a0, a1);
    if constexpr (!(UM & 2)) load_pairs(blo + i, bhi + i, b0, b1);
    __m256d r0, r1;
    if constexpr (OP == 0) r0 = _mm256_add_pd(a0, b0), r1 = _mm256_add_pd(a1, b1);
    else if constexpr (OP == 1) r0 = _mm256_mul_pd(a0, b0), r1 = _mm256_mul_pd(a1, b1);
    else {
      if constexpr (!(UM & 4)) load_pairs(clo + i, chi + i, c0, c1);
      r0 = _mm256_fmadd_pd(a0, b0, c0), r1 = _mm256_fmadd_pd(a1, b1, c1);
    }
    nan |= static_cast<uint64_t>(_mm256_movemask_pd(_mm256_cmp_pd(r0, r0, _CMP_UNORD_Q)) | _mm256_movemask_pd(_mm256_cmp_pd(r1, r1, _CMP_UNORD_Q)) << 4) << i;
    store_pairs(rlo + i, rhi + i, r0, r1);
  }
#else
  for (unsigned i = 0; i < n; ++i) {
    const uint64_t ua = UM & 1 ? u[0] : static_cast<uint64_t>(ahi[i]) << 32 | alo[i], ub = UM & 2 ? u[1] : static_cast<uint64_t>(bhi[i]) << 32 | blo[i];
    double a, b, c = 0, r;
    __builtin_memcpy(&a, &ua, 8);
    __builtin_memcpy(&b, &ub, 8);
    if constexpr (OP == 2) {
      const uint64_t uc = UM & 4 ? u[2] : static_cast<uint64_t>(chi[i]) << 32 | clo[i];
      __builtin_memcpy(&c, &uc, 8);
    }
    if constexpr (OP == 0) r = a + b;
    else if constexpr (OP == 1) r = a * b;
    else r = __builtin_fma(a, b, c);
    uint64_t ur;
    __builtin_memcpy(&ur, &r, 8);
    nan |= static_cast<uint64_t>(r != r) << i;
    rlo[i] = static_cast<uint32_t>(ur), rhi[i] = static_cast<uint32_t>(ur >> 32);
  }
#endif
  return nan;
}
#define TIER_F64R_ROW(op) k_f64r<op, 0>, k_f64r<op, 1>, k_f64r<op, 2>, k_f64r<op, 3>, k_f64r<op, 4>, k_f64r<op, 5>, k_f64r<op, 6>, k_f64r<op, 7>
static const F64RowsFn kF64Rows[24] = {TIER_F64R_ROW(0), TIER_F64R_ROW(1), TIER_F64R_ROW(2)};   // (operation * 8 + the uniform operands)

// ---- 64-bit integer lanes (gathered register pairs)
using K64 = void (*)(uint64_t* d, const uint64_t* a, const uint64_t* b, const uint64_t* c, unsigned n);
#define TIER_L(name, ...)                                                                                    \
  static void name(uint64_t* d, [[maybe_unused]] const uint64_t* a, [[maybe_unused]] const uint64_t* b,     \
                   [[maybe_unused]] const uint64_t* c, unsigned n) {                                          \
    _Pragma("GCC ivdep") for (unsigned i = 0; i < n; ++i) { __VA_ARGS__; }                                          \
  }
TIER_L(l_mov, d[i] = a[i])
TIER_L(l_lshlrev, d[i] = b[i] << (a[i] & 63))
TIER_L(l_lshrrev, d[i] = b[i] >> (a[i] & 63))
TIER_L(l_ashrrev, d[i] = static_cast<uint64_t>(static_cast<int64_t>(b[i]) >> (a[i] & 63)))
TIER_L(l_lshl_add, d[i] = (a[i] << (b[i] & 63)) + c[i])
// (gfx942 and gfx950, CDNA3 and CDNA4 guides: the count is bits 2:0 and a count over 4 is a shift of zero.)
TIER_L(l_lshl_add_g, d[i] = (a[i] << ((b[i] & 7) > 4 ? 0 : (b[i] & 7))) + c[i])
TIER_L(l_cvt_f64_u, reinterpret_cast<F64*>(d)[i] = static_cast<double>(static_cast<uint32_t>(a[i])))
TIER_L(l_cvt_f64_i, reinterpret_cast<F64*>(d)[i] = static_cast<double>(static_cast<int32_t>(a[i])))
// The double-precision functions of one source and the conversions between the widths (the interpreter's own expressions, libm's ldexp and frexp
// included, so a NaN or a denormal is what it makes of it). The 64 -> 32 conversions give the 32-bit result in the low word.
TIER_L(l_ldexp, { double x, y; std::memcpy(&x, &a[i], 8); y = std::ldexp(x, static_cast<int32_t>(static_cast<uint32_t>(b[i]))); std::memcpy(&d[i], &y, 8); })
TIER_L(l_frexp_mant, { double x, m; std::memcpy(&x, &a[i], 8); int e = 0; m = std::isfinite(x) ? std::frexp(x, &e) : x; std::memcpy(&d[i], &m, 8); })
TIER_L(l_frexp_exp, { double x; std::memcpy(&x, &a[i], 8); int e = 0; if (std::isfinite(x)) std::frexp(x, &e); d[i] = static_cast<uint32_t>(e); })
TIER_L(l_cvt_f64_f32, { float x; const uint32_t lo = static_cast<uint32_t>(a[i]); std::memcpy(&x, &lo, 4); const double y = static_cast<double>(x); std::memcpy(&d[i], &y, 8); })
TIER_L(l_cvt_f32_f64, { double x; std::memcpy(&x, &a[i], 8); const float y = static_cast<float>(x); uint32_t b32; std::memcpy(&b32, &y, 4); d[i] = b32; })
TIER_L(l_cvt_i32_f64, { double x; std::memcpy(&x, &a[i], 8); d[i] = std::isnan(x) ? 0u : x >= 2147483647.0 ? 0x7FFFFFFFu : x <= -2147483648.0 ? 0x80000000u : static_cast<uint32_t>(static_cast<int32_t>(x)); })
TIER_L(l_cvt_u32_f64, { double x; std::memcpy(&x, &a[i], 8); d[i] = !(x > 0) ? 0u : x >= 4294967295.0 ? 0xFFFFFFFFu : static_cast<uint32_t>(x); })

// ---- comparisons: bit l of the result is `a[l] test b[l]`, for the first n lanes
// The integer tests, in the ISA's order after F (0): LT, EQ, LE, GT, NE, GE, T (7).
// The float tests: F LT EQ LE GT LG GE O U NGE NLG NGT NLE NEQ NLT TRU.
#if TIER_AVX2
static inline uint32_t movmask8(V8i m) { return static_cast<uint32_t>(_mm256_movemask_ps(reinterpret_cast<__m256>(m))); }
static inline uint32_t movmask8f(V8f v) { return static_cast<uint32_t>(_mm256_movemask_ps(reinterpret_cast<__m256>(v))); }
#else
static inline uint32_t movmask4(V4i m) { return static_cast<uint32_t>(_mm_movemask_ps(reinterpret_cast<__m128>(m))); }
#endif

template <int T, bool Signed>
static uint64_t cmp_i32(const uint32_t* a, const uint32_t* b, unsigned n) {
  uint64_t out = 0;
  if constexpr (T == 0) return 0;
  if constexpr (T == 7) return n == 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1;
#if TIER_AVX2
  for (unsigned i = 0; i < n; i += 8) {
    V8u x, y;
    __builtin_memcpy(&x, a + i, 32);
    __builtin_memcpy(&y, b + i, 32);
    V8i m;
    if constexpr (Signed) {
      const V8i sx = reinterpret_cast<V8i>(x), sy = reinterpret_cast<V8i>(y);
      if constexpr (T == 1) m = sx < sy;
      else if constexpr (T == 2) m = sx == sy;
      else if constexpr (T == 3) m = sx <= sy;
      else if constexpr (T == 4) m = sx > sy;
      else if constexpr (T == 5) m = sx != sy;
      else m = sx >= sy;
    } else {
      if constexpr (T == 1) m = reinterpret_cast<V8i>(x < y);
      else if constexpr (T == 2) m = reinterpret_cast<V8i>(x == y);
      else if constexpr (T == 3) m = reinterpret_cast<V8i>(x <= y);
      else if constexpr (T == 4) m = reinterpret_cast<V8i>(x > y);
      else if constexpr (T == 5) m = reinterpret_cast<V8i>(x != y);
      else m = reinterpret_cast<V8i>(x >= y);
    }
    out |= static_cast<uint64_t>(movmask8(m)) << i;
  }
#else
  for (unsigned i = 0; i < n; i += 4) {
    V4u x, y;
    __builtin_memcpy(&x, a + i, 16);
    __builtin_memcpy(&y, b + i, 16);
    V4i m;
    if constexpr (Signed) {
      const V4i sx = reinterpret_cast<V4i>(x), sy = reinterpret_cast<V4i>(y);
      if constexpr (T == 1) m = sx < sy;
      else if constexpr (T == 2) m = sx == sy;
      else if constexpr (T == 3) m = sx <= sy;
      else if constexpr (T == 4) m = sx > sy;
      else if constexpr (T == 5) m = sx != sy;
      else m = sx >= sy;
    } else {
      if constexpr (T == 1) m = reinterpret_cast<V4i>(x < y);
      else if constexpr (T == 2) m = reinterpret_cast<V4i>(x == y);
      else if constexpr (T == 3) m = reinterpret_cast<V4i>(x <= y);
      else if constexpr (T == 4) m = reinterpret_cast<V4i>(x > y);
      else if constexpr (T == 5) m = reinterpret_cast<V4i>(x != y);
      else m = reinterpret_cast<V4i>(x >= y);
    }
    out |= static_cast<uint64_t>(movmask4(m)) << i;
  }
#endif
  return out;
}

template <int T>
static uint64_t cmp_f32(const uint32_t* ua, const uint32_t* ub, unsigned n) {
  if constexpr (T == 0) return 0;
  if constexpr (T == 15) return n == 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1;
  uint64_t out = 0;
  for (unsigned i = 0; i < n; i += 4) {
    V4f x, y;
    __builtin_memcpy(&x, ua + i, 16);
    __builtin_memcpy(&y, ub + i, 16);
    V4i m;
    if constexpr (T == 1) m = x < y;
    else if constexpr (T == 2) m = x == y;
    else if constexpr (T == 3) m = x <= y;
    else if constexpr (T == 4) m = x > y;
    else if constexpr (T == 5) m = (x < y) | (x > y);
    else if constexpr (T == 6) m = x >= y;
    else if constexpr (T == 7) m = (x == x) & (y == y);
    else if constexpr (T == 8) m = (x != x) | (y != y);
    else if constexpr (T == 9) m = ~(x >= y);
    else if constexpr (T == 10) m = ~((x < y) | (x > y));
    else if constexpr (T == 11) m = ~(x > y);
    else if constexpr (T == 12) m = ~(x <= y);
    else if constexpr (T == 13) m = ~(x == y);
    else m = ~(x < y);
    out |= static_cast<uint64_t>(_mm_movemask_ps(reinterpret_cast<__m128>(m))) << i;
  }
  return out;
}

// 64-bit lanes: integers (signed or not) and binary64; the tests as above.
template <int T, bool Signed>
static uint64_t cmp_i64(const uint64_t* a, const uint64_t* b, unsigned n) {
  uint64_t out = 0;
#if TIER_AVX2
  if constexpr (T == 0) return 0;
  else if constexpr (T == 7) return n == 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1;
  else {
    typedef int64_t V4l __attribute__((vector_size(32)));
    typedef uint64_t V4ul __attribute__((vector_size(32)));
    for (unsigned i = 0; i < n; i += 4) {
      V4ul x, y;
      __builtin_memcpy(&x, a + i, 32);
      __builtin_memcpy(&y, b + i, 32);
      V4l m;
      if constexpr (Signed) {
        const V4l sx = reinterpret_cast<V4l>(x), sy = reinterpret_cast<V4l>(y);
        if constexpr (T == 1) m = sx < sy;
        else if constexpr (T == 2) m = sx == sy;
        else if constexpr (T == 3) m = sx <= sy;
        else if constexpr (T == 4) m = sx > sy;
        else if constexpr (T == 5) m = sx != sy;
        else m = sx >= sy;
      } else {
        if constexpr (T == 1) m = reinterpret_cast<V4l>(x < y);
        else if constexpr (T == 2) m = reinterpret_cast<V4l>(x == y);
        else if constexpr (T == 3) m = reinterpret_cast<V4l>(x <= y);
        else if constexpr (T == 4) m = reinterpret_cast<V4l>(x > y);
        else if constexpr (T == 5) m = reinterpret_cast<V4l>(x != y);
        else m = reinterpret_cast<V4l>(x >= y);
      }
      out |= static_cast<uint64_t>(_mm256_movemask_pd(reinterpret_cast<__m256d>(m))) << i;
    }
    return out;
  }
#endif
  for (unsigned i = 0; i < n; ++i) {
    bool r;
    if constexpr (Signed) {
      const int64_t x = static_cast<int64_t>(a[i]), y = static_cast<int64_t>(b[i]);
      r = T == 0 ? false : T == 1 ? x < y : T == 2 ? x == y : T == 3 ? x <= y : T == 4 ? x > y : T == 5 ? x != y : T == 6 ? x >= y : true;
    } else {
      const uint64_t x = a[i], y = b[i];
      r = T == 0 ? false : T == 1 ? x < y : T == 2 ? x == y : T == 3 ? x <= y : T == 4 ? x > y : T == 5 ? x != y : T == 6 ? x >= y : true;
    }
    out |= static_cast<uint64_t>(r) << i;
  }
  return out;
}
template <int T>
static uint64_t cmp_f64(const uint64_t* ua, const uint64_t* ub, unsigned n) {
  uint64_t out = 0;
  for (unsigned i = 0; i < n; ++i) {
    const double x = reinterpret_cast<const F64*>(ua)[i], y = reinterpret_cast<const F64*>(ub)[i];
    bool r;
    switch (T) {
      case 0: r = false; break;
      case 1: r = x < y; break;
      case 2: r = x == y; break;
      case 3: r = x <= y; break;
      case 4: r = x > y; break;
      case 5: r = x < y || x > y; break;
      case 6: r = x >= y; break;
      case 7: r = x == x && y == y; break;
      case 8: r = x != x || y != y; break;
      case 9: r = !(x >= y); break;
      case 10: r = !(x < y || x > y); break;
      case 11: r = !(x > y); break;
      case 12: r = !(x <= y); break;
      case 13: r = !(x == y); break;
      case 14: r = !(x < y); break;
      default: r = true; break;
    }
    out |= static_cast<uint64_t>(r) << i;
  }
  return out;
}
using Cmp64Fn = uint64_t (*)(const uint64_t*, const uint64_t*, unsigned);
static const Cmp64Fn kCmpU64[8] = {cmp_i64<0, false>, cmp_i64<1, false>, cmp_i64<2, false>, cmp_i64<3, false>,
                                   cmp_i64<4, false>, cmp_i64<5, false>, cmp_i64<6, false>, cmp_i64<7, false>};
static const Cmp64Fn kCmpI64[8] = {cmp_i64<0, true>, cmp_i64<1, true>, cmp_i64<2, true>, cmp_i64<3, true>,
                                   cmp_i64<4, true>, cmp_i64<5, true>, cmp_i64<6, true>, cmp_i64<7, true>};
static const Cmp64Fn kCmpF64[16] = {cmp_f64<0>, cmp_f64<1>, cmp_f64<2>,  cmp_f64<3>,  cmp_f64<4>,  cmp_f64<5>,  cmp_f64<6>,  cmp_f64<7>,
                                    cmp_f64<8>, cmp_f64<9>, cmp_f64<10>, cmp_f64<11>, cmp_f64<12>, cmp_f64<13>, cmp_f64<14>, cmp_f64<15>};

// 16-bit integers (the low half of each register), and halves (widened to float, so that every test is the float's), and the class tests.
template <int T, bool Signed>
static uint64_t cmp_i16(const uint32_t* a, const uint32_t* b, unsigned n) {
  uint64_t out = 0;
  for (unsigned i = 0; i < n; ++i) {
    bool r;
    if constexpr (Signed) {
      const int16_t x = static_cast<int16_t>(a[i]), y = static_cast<int16_t>(b[i]);
      r = T == 0 ? false : T == 1 ? x < y : T == 2 ? x == y : T == 3 ? x <= y : T == 4 ? x > y : T == 5 ? x != y : T == 6 ? x >= y : true;
    } else {
      const uint16_t x = static_cast<uint16_t>(a[i]), y = static_cast<uint16_t>(b[i]);
      r = T == 0 ? false : T == 1 ? x < y : T == 2 ? x == y : T == 3 ? x <= y : T == 4 ? x > y : T == 5 ? x != y : T == 6 ? x >= y : true;
    }
    out |= static_cast<uint64_t>(r) << i;
  }
  return out;
}
template <int T>
static uint64_t cmp_f16(const uint32_t* a, const uint32_t* b, unsigned n) {
  uint64_t out = 0;
  for (unsigned i = 0; i < n; ++i) {
    const float x = vgpu::amd::packed::half_to_float(static_cast<uint16_t>(a[i])), y = vgpu::amd::packed::half_to_float(static_cast<uint16_t>(b[i]));
    bool r;
    switch (T) {
      case 0: r = false; break;
      case 1: r = x < y; break;
      case 2: r = x == y; break;
      case 3: r = x <= y; break;
      case 4: r = x > y; break;
      case 5: r = x < y || x > y; break;
      case 6: r = x >= y; break;
      case 7: r = x == x && y == y; break;
      case 8: r = x != x || y != y; break;
      case 9: r = !(x >= y); break;
      case 10: r = !(x < y || x > y); break;
      case 11: r = !(x > y); break;
      case 12: r = !(x <= y); break;
      case 13: r = !(x == y); break;
      case 14: r = !(x < y); break;
      default: r = true; break;
    }
    out |= static_cast<uint64_t>(r) << i;
  }
  return out;
}
// v_cmp_class_f32: which of ten kinds of number the value is (signaling NaN, quiet NaN, -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf: one bit
// each, in that order) against the mask in the second operand.
static uint64_t cmp_class_f32(const uint32_t* a, const uint32_t* m, unsigned n) {
  uint64_t out = 0;
  for (unsigned i = 0; i < n; ++i) {
    const uint32_t b = a[i], ab = b & 0x7FFFFFFFu;
    const bool negative = b >> 31;
    uint32_t bit;
    if (ab > 0x7F800000u) bit = (b & 0x400000u) ? 1u << 1 : 1u << 0;
    else if (ab == 0x7F800000u) bit = negative ? 1u << 2 : 1u << 9;
    else if (ab == 0) bit = negative ? 1u << 5 : 1u << 6;
    else if (ab < 0x00800000u) bit = negative ? 1u << 4 : 1u << 7;
    else bit = negative ? 1u << 3 : 1u << 8;
    out |= static_cast<uint64_t>((m[i] & bit) != 0) << i;
  }
  return out;
}
// (`Negated`: the half was negated by the instruction, and the negation of a half quiets a signaling NaN, as the interpreter's `-h` does.)
template <bool Negated>
static uint64_t cmp_class_f16(const uint32_t* a, const uint32_t* m, unsigned n) {
  uint64_t out = 0;
  for (unsigned i = 0; i < n; ++i) {
    uint32_t b = a[i] & 0xFFFFu;
    if (Negated && (b & 0x7FFFu) > 0x7C00u) b |= 0x200u;
    const uint32_t ab = b & 0x7FFFu;
    const bool negative = b >> 15;
    uint32_t bit;
    if (ab > 0x7C00u) bit = (b & 0x200u) ? 1u << 1 : 1u << 0;
    else if (ab == 0x7C00u) bit = negative ? 1u << 2 : 1u << 9;
    else if (ab == 0) bit = negative ? 1u << 5 : 1u << 6;
    else if (ab < 0x0400u) bit = negative ? 1u << 4 : 1u << 7;
    else bit = negative ? 1u << 3 : 1u << 8;
    out |= static_cast<uint64_t>((m[i] & bit) != 0) << i;
  }
  return out;
}
using CmpFn = uint64_t (*)(const uint32_t*, const uint32_t*, unsigned);
static const CmpFn kCmpI16[8] = {cmp_i16<0, true>, cmp_i16<1, true>, cmp_i16<2, true>, cmp_i16<3, true>,
                                 cmp_i16<4, true>, cmp_i16<5, true>, cmp_i16<6, true>, cmp_i16<7, true>};
static const CmpFn kCmpU16[8] = {cmp_i16<0, false>, cmp_i16<1, false>, cmp_i16<2, false>, cmp_i16<3, false>,
                                 cmp_i16<4, false>, cmp_i16<5, false>, cmp_i16<6, false>, cmp_i16<7, false>};
static const CmpFn kCmpF16[16] = {cmp_f16<0>, cmp_f16<1>, cmp_f16<2>,  cmp_f16<3>,  cmp_f16<4>,  cmp_f16<5>,  cmp_f16<6>,  cmp_f16<7>,
                                  cmp_f16<8>, cmp_f16<9>, cmp_f16<10>, cmp_f16<11>, cmp_f16<12>, cmp_f16<13>, cmp_f16<14>, cmp_f16<15>};
static const CmpFn kCmpU32[8] = {cmp_i32<0, false>, cmp_i32<1, false>, cmp_i32<2, false>, cmp_i32<3, false>,
                                 cmp_i32<4, false>, cmp_i32<5, false>, cmp_i32<6, false>, cmp_i32<7, false>};
static const CmpFn kCmpI32[8] = {cmp_i32<0, true>, cmp_i32<1, true>, cmp_i32<2, true>, cmp_i32<3, true>,
                                 cmp_i32<4, true>, cmp_i32<5, true>, cmp_i32<6, true>, cmp_i32<7, true>};
static const CmpFn kCmpF32[16] = {cmp_f32<0>, cmp_f32<1>, cmp_f32<2>,  cmp_f32<3>,  cmp_f32<4>,  cmp_f32<5>,
                                  cmp_f32<6>, cmp_f32<7>, cmp_f32<8>,  cmp_f32<9>,  cmp_f32<10>, cmp_f32<11>,
                                  cmp_f32<12>, cmp_f32<13>, cmp_f32<14>, cmp_f32<15>};

// v_mad_u64_u32 over n lanes: lo/hi receive the 64-bit sum a*b + c (c as two rows, its low and high words), and bit l of the result says
// that lane l's sum carried out of 64 bits (the product of two 32-bit values fits in 64 bits, so that is exactly a wrapped sum: sum < c).
// A lane reads its inputs before it writes its outputs, so lo and hi may be rows an input also is.
static uint64_t k_mad_u64(uint32_t* lo, uint32_t* hi, const uint32_t* a, const uint32_t* b, const uint32_t* cl, const uint32_t* ch, unsigned n) {
  uint64_t carry = 0;
#if TIER_AVX2
  const __m256i bias = _mm256_set1_epi64x(INT64_MIN);
  const __m256i evens = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6), odds = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);
  for (unsigned i = 0; i < n; i += 4) {
    const __m256i x = _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
    const __m256i y = _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
    const __m256i c = _mm256_or_si256(_mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cl + i))),
                                      _mm256_slli_epi64(_mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ch + i))), 32));
    const __m256i sum = _mm256_add_epi64(_mm256_mul_epu32(x, y), c);
    const __m256i wrapped = _mm256_cmpgt_epi64(_mm256_xor_si256(c, bias), _mm256_xor_si256(sum, bias));   // c > sum, unsigned
    carry |= static_cast<uint64_t>(_mm256_movemask_pd(_mm256_castsi256_pd(wrapped))) << i;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(lo + i), _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(sum, evens)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(hi + i), _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(sum, odds)));
  }
#else
  for (unsigned i = 0; i < n; ++i) {
    const uint64_t c = static_cast<uint64_t>(ch[i]) << 32 | cl[i];
    const uint64_t sum = static_cast<uint64_t>(a[i]) * b[i] + c;
    carry |= static_cast<uint64_t>(sum < c) << i;
    lo[i] = static_cast<uint32_t>(sum), hi[i] = static_cast<uint32_t>(sum >> 32);
  }
#endif
  return carry;
}

static const K3 kIntTab[] = {TIER_INT_OPS(TIER_PICK)};
static const K3 kF32Tab[] = {TIER_F32_OPS(TIER_PICK)};
static const K64 kF64Tab[] = {TIER_F64_OPS(TIER_PICK)};
static const K64 kL64Tab[] = {TIER_L64_OPS(TIER_PICK)};

// v_add_co_u32, v_addc_co_u32, v_sub_co_u32, v_subb_co_u32 and the reversed ones over n lanes: d receives the sum or difference, and bit l of the
// result is lane l's carry or borrow. SUB: bit 0 add, bit 1 takes a carry in (bit l of `cin`), bit 2 reversed. Every lane is computed (the caller
// masks the result and merges d under EXEC); a lane reads its inputs before it writes d, so d may be a or b.
using CarryFn = uint64_t (*)(uint32_t* d, const uint32_t* a, const uint32_t* b, uint64_t cin, unsigned n);
template <unsigned SUB>
static uint64_t k_carry(uint32_t* d, const uint32_t* a, const uint32_t* b, [[maybe_unused]] uint64_t cin, unsigned n) {
  constexpr bool add = SUB & 1, takes = SUB & 2, rev = SUB & 4;
  uint64_t out = 0;
#if TIER_AVX2
  const __m256i bias = _mm256_set1_epi32(INT32_MIN), lanebit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  for (unsigned i = 0; i < n; i += 8) {
    __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    if (rev) std::swap(x, y);
    __m256i c1 = _mm256_setzero_si256();   // the carry in, 0 or 1 a lane
    if (takes) {
      const __m256i bits = _mm256_set1_epi32(static_cast<int>((cin >> i) & 0xFF));
      c1 = _mm256_srli_epi32(_mm256_cmpeq_epi32(_mm256_and_si256(bits, lanebit), lanebit), 31);
    }
    __m256i r, k;   // k: all ones in a lane that carried
    if (add) {
      const __m256i s1 = _mm256_add_epi32(x, y);
      r = takes ? _mm256_add_epi32(s1, c1) : s1;
      k = _mm256_cmpgt_epi32(_mm256_xor_si256(x, bias), _mm256_xor_si256(s1, bias));   // the sum wrapped: s1 < x
      if (takes) k = _mm256_or_si256(k, _mm256_cmpgt_epi32(_mm256_xor_si256(s1, bias), _mm256_xor_si256(r, bias)));
    } else {
      r = _mm256_sub_epi32(_mm256_sub_epi32(x, y), c1);
      k = _mm256_cmpgt_epi32(_mm256_xor_si256(y, bias), _mm256_xor_si256(x, bias));   // x < y
      if (takes) k = _mm256_or_si256(k, _mm256_and_si256(_mm256_cmpeq_epi32(x, y), _mm256_cmpeq_epi32(c1, _mm256_set1_epi32(1))));
    }
    out |= static_cast<uint64_t>(static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(k)))) << i;
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + i), r);
  }
#else
  for (unsigned l = 0; l < n; ++l) {
    uint64_t x = a[l], y = b[l];
    if (rev) std::swap(x, y);
    const uint64_t c = takes ? (cin >> l) & 1 : 0;
    const uint64_t v = add ? x + y + c : x - y - c;
    out |= static_cast<uint64_t>(add ? (v >> 32) & 1 : x < y + c) << l;
    d[l] = static_cast<uint32_t>(v);
  }
#endif
  return out;
}
static const CarryFn kCarry[8] = {k_carry<0>, k_carry<1>, k_carry<2>, k_carry<3>, k_carry<4>, k_carry<5>, k_carry<6>, k_carry<7>};

}  // namespace TIER_NS
