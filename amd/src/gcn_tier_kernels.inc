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

using CmpFn = uint64_t (*)(const uint32_t*, const uint32_t*, unsigned);
static const CmpFn kCmpU32[8] = {cmp_i32<0, false>, cmp_i32<1, false>, cmp_i32<2, false>, cmp_i32<3, false>,
                                 cmp_i32<4, false>, cmp_i32<5, false>, cmp_i32<6, false>, cmp_i32<7, false>};
static const CmpFn kCmpI32[8] = {cmp_i32<0, true>, cmp_i32<1, true>, cmp_i32<2, true>, cmp_i32<3, true>,
                                 cmp_i32<4, true>, cmp_i32<5, true>, cmp_i32<6, true>, cmp_i32<7, true>};
static const CmpFn kCmpF32[16] = {cmp_f32<0>, cmp_f32<1>, cmp_f32<2>,  cmp_f32<3>,  cmp_f32<4>,  cmp_f32<5>,
                                  cmp_f32<6>, cmp_f32<7>, cmp_f32<8>,  cmp_f32<9>,  cmp_f32<10>, cmp_f32<11>,
                                  cmp_f32<12>, cmp_f32<13>, cmp_f32<14>, cmp_f32<15>};

static const K3 kIntTab[] = {TIER_INT_OPS(TIER_PICK)};
static const K3 kF32Tab[] = {TIER_F32_OPS(TIER_PICK)};
static const K64 kF64Tab[] = {TIER_F64_OPS(TIER_PICK)};
static const K64 kL64Tab[] = {TIER_L64_OPS(TIER_PICK)};

}  // namespace TIER_NS
