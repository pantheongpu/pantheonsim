// The kernels, written once and compiled once per level: simd.cpp includes
// this file inside a namespace of the level's name, the second time under
// #pragma GCC target. Plain C++ and GCC vector types, so every level is the
// same source and the same bits; the intrinsics below are only for what the
// baseline cannot say (F16C, FMA), behind VGPU_SIMD_AVX2.
//
// VGPU_SIMD_AVX2 (1 in the AVX2 level, 0 in the baseline) says which intrinsics may be used: GCC does not
// define __AVX2__ and the like under #pragma GCC target.
//
// Contraction is off for the whole file: a product that the card rounds
// before the sum must never be fused into an FMA by a build that has one.

namespace VGPU_SIMD_NS {

typedef float V8f __attribute__((vector_size(32)));

// n a multiple of 8.
inline void f16_to_f32_blocks(const uint16_t* in, float* out, size_t n) {
#if VGPU_SIMD_AVX2
  for (size_t i = 0; i < n; i += 8)
    _mm256_storeu_ps(out + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i))));
#else
  for (size_t i = 0; i < n; ++i) out[i] = f16_to_f32(in[i]);
#endif
}
inline void f32_to_f16_blocks(const float* in, uint16_t* out, size_t n) {
#if VGPU_SIMD_AVX2
  for (size_t i = 0; i < n; i += 8)
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i),
                     _mm256_cvtps_ph(_mm256_loadu_ps(in + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#else
  for (size_t i = 0; i < n; ++i) out[i] = f32_to_f16_bits(in[i]);
#endif
}

void f16_to_f32_k(const uint16_t* in, float* out, size_t n) {
  const size_t whole = n & ~size_t{7};
  f16_to_f32_blocks(in, out, whole);
  for (size_t i = whole; i < n; ++i) out[i] = f16_to_f32(in[i]);
}

void f32_to_f16_k(const float* in, uint16_t* out, size_t n) {
  const size_t whole = n & ~size_t{7};
  f32_to_f16_blocks(in, out, whole);
  for (size_t i = whole; i < n; ++i) out[i] = f32_to_f16_bits(in[i]);
}

typedef float V4f __attribute__((vector_size(16)));
typedef int32_t V4i __attribute__((vector_size(16)));
typedef uint32_t V4u __attribute__((vector_size(16)));
typedef uint32_t V8u __attribute__((vector_size(32)));
typedef uint16_t V8h __attribute__((vector_size(16)));

static_assert(std::endian::native == std::endian::little, "the fragment kernels read a register's halves in memory order");

// n 32-bit words of 16-bit (kind 0, 1) or tf32 (kind 2) operands, as floats, in memory order: word w's low half is
// out[2w], its high half out[2w + 1]. n_words is a multiple of 4.
inline void operands_to_f32(unsigned kind, const uint32_t* w, float* out, size_t n_words) {
  if (kind == 0) {
    f16_to_f32_blocks(reinterpret_cast<const uint16_t*>(w), out, 2 * n_words);
  } else if (kind == 1) {
    for (size_t i = 0; i < 2 * n_words; i += 8) {
      V8h h;
      std::memcpy(&h, reinterpret_cast<const char*>(w) + 2 * i, sizeof h);
      const V8u u = __builtin_convertvector(h, V8u) << 16;
      std::memcpy(out + i, &u, sizeof u);
    }
  } else {
    // tf32: the top 19 bits of an f32; the rest are ignored.
    for (size_t i = 0; i < n_words; i += 4) {
      V4u u;
      std::memcpy(&u, w + i, sizeof u);
      u &= 0xffffe000u;
      std::memcpy(out + i, &u, sizeof u);
    }
  }
}

// dst[j * 8 + g] = src[g * 8 + j], for g, j < 8.
inline void transpose8x8(float* dst, const float* src) {
#if VGPU_SIMD_AVX2
  __m256 r[8];
  for (int g = 0; g < 8; ++g) r[g] = _mm256_loadu_ps(src + g * 8);
  const __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]);
  const __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
  const __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]);
  const __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
  const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
  const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
  const __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44), u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
  const __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44), u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
  _mm256_storeu_ps(dst + 0 * 8, _mm256_permute2f128_ps(u0, u4, 0x20));
  _mm256_storeu_ps(dst + 1 * 8, _mm256_permute2f128_ps(u1, u5, 0x20));
  _mm256_storeu_ps(dst + 2 * 8, _mm256_permute2f128_ps(u2, u6, 0x20));
  _mm256_storeu_ps(dst + 3 * 8, _mm256_permute2f128_ps(u3, u7, 0x20));
  _mm256_storeu_ps(dst + 4 * 8, _mm256_permute2f128_ps(u0, u4, 0x31));
  _mm256_storeu_ps(dst + 5 * 8, _mm256_permute2f128_ps(u1, u5, 0x31));
  _mm256_storeu_ps(dst + 6 * 8, _mm256_permute2f128_ps(u2, u6, 0x31));
  _mm256_storeu_ps(dst + 7 * 8, _mm256_permute2f128_ps(u3, u7, 0x31));
#else
  for (int g = 0; g < 8; ++g)
    for (int j = 0; j < 8; ++j) dst[j * 8 + g] = src[g * 8 + j];
#endif
}

void hmma_k(unsigned kind, unsigned K, bool acc_f16, const uint32_t* a, const uint32_t* b, const uint32_t* c,
            uint32_t* out) {
  const unsigned per_reg = kind == 2 ? 1 : 2, L = 4 * per_reg;
  const unsigned a_regs = K / (2 * per_reg), b_regs = K / (4 * per_reg);
  alignas(32) float FA[256], FB[128];
  operands_to_f32(kind, a, FA, size_t{a_regs} * 32);
  operands_to_f32(kind, b, FB, size_t{b_regs} * 32);
  // The fragments are laid out so that, for a lane group g (lanes 4g..4g+3), the L consecutive elements
  // FA[r * 32 * per_reg + g * L ...] are one row segment of A: row g + 8 (r % 2), columns (r / 2) * L ...; and for
  // B they are one column segment: column g, rows r * L ... (PTX ISA 9.7.16).
  alignas(32) float A[16 * 16], Bm[16 * 8], Cm[16 * 8], Dm[16 * 8];
  if (L == 8) {
    for (unsigned r = 0; r < a_regs; ++r)
      for (unsigned g = 0; g < 8; ++g)
        std::memcpy(&A[(g + 8 * (r & 1)) * K + (r >> 1) * 8], &FA[r * 64 + g * 8], 8 * sizeof(float));
    for (unsigned r = 0; r < b_regs; ++r) transpose8x8(&Bm[r * 64], &FB[r * 64]);
  } else {
    for (unsigned r = 0; r < a_regs; ++r)
      for (unsigned g = 0; g < 8; ++g)
        std::memcpy(&A[(g + 8 * (r & 1)) * K + (r >> 1) * L], &FA[r * 32 + g * L], L * sizeof(float));
    for (unsigned r = 0; r < b_regs; ++r)
      for (unsigned g = 0; g < 8; ++g)
        for (unsigned j = 0; j < L; ++j) Bm[(r * L + j) * 8 + g] = FB[r * 32 + g * L + j];
  }
  // C: element s = 2h + e of lane 4g+t is row g + 8h, column 2t + e. With a half accumulator that is the
  // converted halves in memory order, unchanged; with an f32 one, registers 2h and 2h+1 interleave.
  if (!c) {
    std::memset(Cm, 0, sizeof Cm);
  } else if (acc_f16) {
    f16_to_f32_blocks(reinterpret_cast<const uint16_t*>(c), Cm, 128);
  } else {
    for (unsigned h = 0; h < 2; ++h)
      for (unsigned g = 0; g < 8; ++g) {
        V4u x, y;
        std::memcpy(&x, c + (2 * h) * 32 + 4 * g, sizeof x);
        std::memcpy(&y, c + (2 * h + 1) * 32 + 4 * g, sizeof y);
        const V4u lo = __builtin_shuffle(x, y, (V4i){0, 4, 1, 5}), hi = __builtin_shuffle(x, y, (V4i){2, 6, 3, 7});
        std::memcpy(&Cm[(g + 8 * h) * 8], &lo, sizeof lo);
        std::memcpy(&Cm[(g + 8 * h) * 8 + 4], &hi, sizeof hi);
      }
  }
  for (unsigned i0 = 0; i0 < 16; i0 += 4) {
    V8f acc0, acc1, acc2, acc3;
    std::memcpy(&acc0, &Cm[(i0 + 0) * 8], sizeof acc0);
    std::memcpy(&acc1, &Cm[(i0 + 1) * 8], sizeof acc1);
    std::memcpy(&acc2, &Cm[(i0 + 2) * 8], sizeof acc2);
    std::memcpy(&acc3, &Cm[(i0 + 3) * 8], sizeof acc3);
    for (unsigned k = 0; k < K; ++k) {
      V8f bv;
      std::memcpy(&bv, &Bm[k * 8], sizeof bv);
      const V8f p0 = A[(i0 + 0) * K + k] * bv, p1 = A[(i0 + 1) * K + k] * bv, p2 = A[(i0 + 2) * K + k] * bv,
                p3 = A[(i0 + 3) * K + k] * bv;
      acc0 = acc0 + p0;
      acc1 = acc1 + p1;
      acc2 = acc2 + p2;
      acc3 = acc3 + p3;
    }
    std::memcpy(&Dm[(i0 + 0) * 8], &acc0, sizeof acc0);
    std::memcpy(&Dm[(i0 + 1) * 8], &acc1, sizeof acc1);
    std::memcpy(&Dm[(i0 + 2) * 8], &acc2, sizeof acc2);
    std::memcpy(&Dm[(i0 + 3) * 8], &acc3, sizeof acc3);
  }
  if (!acc_f16) {
    for (unsigned h = 0; h < 2; ++h)
      for (unsigned g = 0; g < 8; ++g) {
        V4f lo, hi;
        std::memcpy(&lo, &Dm[(g + 8 * h) * 8], sizeof lo);
        std::memcpy(&hi, &Dm[(g + 8 * h) * 8 + 4], sizeof hi);
        const V4f e = __builtin_shuffle(lo, hi, (V4i){0, 2, 4, 6}), o = __builtin_shuffle(lo, hi, (V4i){1, 3, 5, 7});
        // A NaN result is the canonical 0x7fffffff.
        const V4u me = e != e, mo = o != o;
        V4u ue = reinterpret_cast<V4u>(e), uo = reinterpret_cast<V4u>(o);
        ue = (ue & ~me) | (0x7fffffffu & me);
        uo = (uo & ~mo) | (0x7fffffffu & mo);
        std::memcpy(out + (2 * h) * 32 + 4 * g, &ue, sizeof ue);
        std::memcpy(out + (2 * h + 1) * 32 + 4 * g, &uo, sizeof uo);
      }
  } else {
    uint16_t hh[128];   // in memory order, as C was
    f32_to_f16_blocks(Dm, hh, 128);
    for (unsigned i = 0; i < 128; ++i)
      if (Dm[i] != Dm[i]) hh[i] = 0x7fff;   // the canonical NaN, whatever the payload
    std::memcpy(out, hh, sizeof hh);
  }
}

// ---- 32-lane ALU kernels (the SASS executor's fast_alu) ----------------------------------------------------------

// A float result as the register holds it: a NaN is the canonical 0x7fffffff, whatever its payload or sign.
// (A macro, not a function: a 32-byte vector parameter changes the ABI on a baseline x86-64 build.)
#define VGPU_CANON_NAN(v) \
  ({ const V8f f_ = (v); const V8u m_ = reinterpret_cast<V8u>(f_ != f_); (reinterpret_cast<V8u>(f_) & ~m_) | (0x7fffffffu & m_); })

void iadd3_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t add) {
  for (unsigned i = 0; i < 32; i += 8) {
    V8u x, y, z;
    std::memcpy(&x, a + i, sizeof x);
    std::memcpy(&y, b + i, sizeof y);
    std::memcpy(&z, c + i, sizeof z);
    const V8u v = x + y + z + add;
    std::memcpy(r + i, &v, sizeof v);
  }
}

void imad_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c) {
  for (unsigned i = 0; i < 32; i += 8) {
    V8u x, y, z;
    std::memcpy(&x, a + i, sizeof x);
    std::memcpy(&y, b + i, sizeof y);
    std::memcpy(&z, c + i, sizeof z);
    const V8u v = x * y + z;
    std::memcpy(r + i, &v, sizeof v);
  }
}

void lop3_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t lut) {
  // Each set bit t of the table is one minterm of (a, b, c): the lanes where a, b, c equal t's bits.
  uint32_t ma[8], mb[8], mc[8];
  unsigned n = 0;
  for (unsigned t = 0; t < 8; ++t)
    if ((lut >> t) & 1) {
      ma[n] = (t & 4) ? 0u : ~0u;
      mb[n] = (t & 2) ? 0u : ~0u;
      mc[n] = (t & 1) ? 0u : ~0u;
      ++n;
    }
  for (unsigned i = 0; i < 32; i += 8) {
    V8u x, y, z, v = {};
    std::memcpy(&x, a + i, sizeof x);
    std::memcpy(&y, b + i, sizeof y);
    std::memcpy(&z, c + i, sizeof z);
    for (unsigned m = 0; m < n; ++m) v |= (x ^ ma[m]) & (y ^ mb[m]) & (z ^ mc[m]);
    std::memcpy(r + i, &v, sizeof v);
  }
}

void ffma_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c) {
  for (unsigned i = 0; i < 32; i += 8) {
    V8f x, y, z;
    std::memcpy(&x, a + i, sizeof x);
    std::memcpy(&y, b + i, sizeof y);
    std::memcpy(&z, c + i, sizeof z);
#if VGPU_SIMD_AVX2
    const V8f v = reinterpret_cast<V8f>(_mm256_fmadd_ps(reinterpret_cast<__m256>(x), reinterpret_cast<__m256>(y),
                                                         reinterpret_cast<__m256>(z)));
    const V8u u = VGPU_CANON_NAN(v);
    std::memcpy(r + i, &u, sizeof u);
#else
    // No FMA unit to ask for: the correctly rounded fused multiply-add, one libm call a lane.
    for (unsigned l = 0; l < 8; ++l) {
      const float f = __builtin_fmaf(x[l], y[l], z[l]);
      r[i + l] = f != f ? 0x7fffffffu : std::bit_cast<uint32_t>(f);
    }
#endif
  }
}

void fadd_k(uint32_t* r, const uint32_t* a, const uint32_t* b) {
  for (unsigned i = 0; i < 32; i += 8) {
    V8f x, y;
    std::memcpy(&x, a + i, sizeof x);
    std::memcpy(&y, b + i, sizeof y);
    const V8u u = VGPU_CANON_NAN(x + y);
    std::memcpy(r + i, &u, sizeof u);
  }
}

void fmul_k(uint32_t* r, const uint32_t* a, const uint32_t* b) {
  for (unsigned i = 0; i < 32; i += 8) {
    V8f x, y;
    std::memcpy(&x, a + i, sizeof x);
    std::memcpy(&y, b + i, sizeof y);
    const V8u u = VGPU_CANON_NAN(x * y);
    std::memcpy(r + i, &u, sizeof u);
  }
}

// ---- masked lane kernels (the PTX interpreter's lane loops) --------------------------------------------------------

// d[l] = r[l] for the lanes of `mask` (bit l), l < n (n a multiple of 8, at most 64); the other lanes of d keep their
// value. r is computed in full first, so d may be one of the sources.
inline void store_masked(uint32_t* d, const uint32_t* r, uint64_t mask, unsigned n) {
  const uint64_t all = n >= 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1;
  if ((mask & all) == all) {
    for (unsigned i = 0; i < n; i += 8) {   // (not memcpy(d, r, n * 4): a call into libc for a size not known here)
      V8u v;
      std::memcpy(&v, r + i, sizeof v);
      std::memcpy(d + i, &v, sizeof v);
    }
    return;
  }
  for (unsigned i = 0; i < n; i += 8) {
    const uint32_t bits = static_cast<uint32_t>(mask >> i) & 0xffu;
    if (bits == 0) continue;
    const V8u pick = {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u};
    const V8u sel = reinterpret_cast<V8u>((pick & bits) != 0);
    V8u old, v;
    std::memcpy(&old, d + i, sizeof old);
    std::memcpy(&v, r + i, sizeof v);
    old = (old & ~sel) | (v & sel);
    std::memcpy(d + i, &old, sizeof old);
  }
}

inline bool full_mask(uint64_t mask, unsigned n) {
  const uint64_t all = n >= 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1;
  return (mask & all) == all;
}

void ffma_m_k(uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t mask, unsigned n) {
  if (full_mask(mask, n)) {   // (the kernel reads a chunk of every source before it writes the chunk: d may be one)
    for (unsigned i = 0; i < n; i += 32) ffma_k(d + i, a + i, b + i, c + i);
    return;
  }
  uint32_t r[64];
  for (unsigned i = 0; i < n; i += 32) ffma_k(r + i, a + i, b + i, c + i);
  store_masked(d, r, mask, n);
}

// ---- integer lane kernels ---------------------------------------------------------------------------------------------

typedef int32_t V8i __attribute__((vector_size(32)));

// The bits of a vector of lane masks (each lane all ones or zero), lane i to bit i.
#if defined(__SSE2__)
#define VGPU_MASK_BITS(m)                                                                                  \
  ({ __m128i lo_, hi_;                                                                                       \
     std::memcpy(&lo_, reinterpret_cast<const char*>(&(m)), 16);                                             \
     std::memcpy(&hi_, reinterpret_cast<const char*>(&(m)) + 16, 16);                                        \
     static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(lo_)) | (_mm_movemask_ps(_mm_castsi128_ps(hi_)) << 4)); })
#else
#define VGPU_MASK_BITS(m)                                                           \
  ({ uint32_t bits_ = 0; for (unsigned k_ = 0; k_ < 8; ++k_) bits_ |= (m[k_] & 1u) << k_; bits_; })
#endif

template <unsigned Cmp, bool Sgn>
inline uint64_t icmp_t(const uint32_t* a, const uint32_t* b, unsigned n) {
  uint64_t out = 0;
#if !VGPU_SIMD_AVX2
  // The baseline has no 256-bit compare: the compiler would split these vectors into scalar code anyway.
  for (unsigned i = 0; i < n; ++i) {
    const uint32_t x = a[i], y = b[i];
    const int32_t xs = static_cast<int32_t>(x), ys = static_cast<int32_t>(y);
    bool r;
    if constexpr (Sgn) r = Cmp == 0 ? xs == ys : Cmp == 1 ? xs != ys : Cmp == 2 ? xs < ys : Cmp == 3 ? xs <= ys : Cmp == 4 ? xs > ys : xs >= ys;
    else r = Cmp == 0 ? x == y : Cmp == 1 ? x != y : Cmp == 2 ? x < y : Cmp == 3 ? x <= y : Cmp == 4 ? x > y : x >= y;
    out |= uint64_t{r} << i;
  }
  return out;
#endif
  for (unsigned i = 0; i < n; i += 8) {
    V8u x, y;
    std::memcpy(&x, a + i, sizeof x);
    std::memcpy(&y, b + i, sizeof y);
    V8u m;
    if constexpr (Sgn) {
      const V8i xs = reinterpret_cast<V8i>(x), ys = reinterpret_cast<V8i>(y);
      m = Cmp == 0 ? reinterpret_cast<V8u>(xs == ys) : Cmp == 1 ? reinterpret_cast<V8u>(xs != ys)
        : Cmp == 2 ? reinterpret_cast<V8u>(xs < ys) : Cmp == 3 ? reinterpret_cast<V8u>(xs <= ys)
        : Cmp == 4 ? reinterpret_cast<V8u>(xs > ys) : reinterpret_cast<V8u>(xs >= ys);
    } else {
      m = Cmp == 0 ? reinterpret_cast<V8u>(x == y) : Cmp == 1 ? reinterpret_cast<V8u>(x != y)
        : Cmp == 2 ? reinterpret_cast<V8u>(x < y) : Cmp == 3 ? reinterpret_cast<V8u>(x <= y)
        : Cmp == 4 ? reinterpret_cast<V8u>(x > y) : reinterpret_cast<V8u>(x >= y);
    }
#if VGPU_SIMD_AVX2
    out |= uint64_t{static_cast<uint32_t>(_mm256_movemask_ps(reinterpret_cast<__m256>(m)))} << i;
#else
    out |= uint64_t{VGPU_MASK_BITS(m)} << i;
#endif
  }
  return out;
}

uint64_t icmp_mask_k(unsigned cmp, bool sgn, const uint32_t* a, const uint32_t* b, unsigned n) {
  switch (cmp * 2 + (sgn ? 1 : 0)) {
    case 0: return icmp_t<0, false>(a, b, n);
    case 1: return icmp_t<0, true>(a, b, n);
    case 2: return icmp_t<1, false>(a, b, n);
    case 3: return icmp_t<1, true>(a, b, n);
    case 4: return icmp_t<2, false>(a, b, n);
    case 5: return icmp_t<2, true>(a, b, n);
    case 6: return icmp_t<3, false>(a, b, n);
    case 7: return icmp_t<3, true>(a, b, n);
    case 8: return icmp_t<4, false>(a, b, n);
    case 9: return icmp_t<4, true>(a, b, n);
    case 10: return icmp_t<5, false>(a, b, n);
    default: return icmp_t<5, true>(a, b, n);
  }
}

template <unsigned Op, bool Sgn>
inline void ibin_t(uint32_t* r, const uint32_t* a, const uint32_t* b, unsigned n) {
  for (unsigned i = 0; i < n; i += 8) {
    V8u x, y;
    std::memcpy(&x, a + i, sizeof x);
    std::memcpy(&y, b + i, sizeof y);
    V8u v;
    const V8i xs = reinterpret_cast<V8i>(x), ys = reinterpret_cast<V8i>(y);
    if constexpr (Op == 0) v = x + y;
    else if constexpr (Op == 1) v = x - y;
    else if constexpr (Op == 2) v = x * y;
    else if constexpr (Op == 3) v = x & y;
    else if constexpr (Op == 4) v = x | y;
    else if constexpr (Op == 5) v = x ^ y;
    else if constexpr (Op == 6) {   // min
      if constexpr (Sgn) v = reinterpret_cast<V8u>(xs < ys ? xs : ys);
      else v = x < y ? x : y;
    } else if constexpr (Op == 7) {   // max
      if constexpr (Sgn) v = reinterpret_cast<V8u>(xs > ys ? xs : ys);
      else v = x > y ? x : y;
    } else if constexpr (Op == 8) {   // shl: a count past the width shifts everything out
      v = (x << (y & 31u)) & reinterpret_cast<V8u>(y < 32u);
    } else {   // shr: past the width, zero (unsigned) or the sign (signed)
      if constexpr (Sgn) {
        const V8u big = reinterpret_cast<V8u>(y > 31u);
        const V8u count = (y & ~big) | (31u & big);
        v = reinterpret_cast<V8u>(xs >> reinterpret_cast<V8i>(count));
      } else {
        v = (x >> (y & 31u)) & reinterpret_cast<V8u>(y < 32u);
      }
    }
    std::memcpy(r + i, &v, sizeof v);
  }
}

void ibin_m_k(unsigned op, bool sgn, uint32_t* d, const uint32_t* a, const uint32_t* b, uint64_t mask, unsigned n) {
  uint32_t tmp[64];
  uint32_t* const r = full_mask(mask, n) ? d : tmp;   // (chunk by chunk, every source is read before its chunk is written)
  switch (op * 2 + (sgn ? 1 : 0)) {
    case 0: case 1: ibin_t<0, false>(r, a, b, n); break;
    case 2: case 3: ibin_t<1, false>(r, a, b, n); break;
    case 4: case 5: ibin_t<2, false>(r, a, b, n); break;
    case 6: case 7: ibin_t<3, false>(r, a, b, n); break;
    case 8: case 9: ibin_t<4, false>(r, a, b, n); break;
    case 10: case 11: ibin_t<5, false>(r, a, b, n); break;
    case 12: ibin_t<6, false>(r, a, b, n); break;
    case 13: ibin_t<6, true>(r, a, b, n); break;
    case 14: ibin_t<7, false>(r, a, b, n); break;
    case 15: ibin_t<7, true>(r, a, b, n); break;
    case 16: case 17: ibin_t<8, false>(r, a, b, n); break;
    case 18: ibin_t<9, false>(r, a, b, n); break;
    default: ibin_t<9, true>(r, a, b, n); break;
  }
  if (r != d) store_masked(d, r, mask, n);
}

void imad_m_k(uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t mask, unsigned n) {
  uint32_t tmp[64];
  uint32_t* const r = full_mask(mask, n) ? d : tmp;
  for (unsigned i = 0; i < n; i += 32) imad_k(r + i, a + i, b + i, c + i);
  if (r != d) store_masked(d, r, mask, n);
}

// ---- binary64 lanes held as register pairs (the SASS executor: the low words of a register, the high words of the next)

// kind 0: a + b, 1: a * b, 2: fma(a, b, c); `mods` has bit 2i for |source i| and bit 2i + 1 for -source i (the
// absolute value first). d, a, b, c are (low, high) pairs of 32-lane arrays; only the lanes of `mask` are written.
void dop_m_k(unsigned kind, unsigned mods, uint32_t* dlo, uint32_t* dhi, const uint32_t* alo, const uint32_t* ahi,
             const uint32_t* blo, const uint32_t* bhi, const uint32_t* clo, const uint32_t* chi, uint32_t mask) {
  uint32_t rlo[32], rhi[32];
  const uint32_t absm[3] = {(mods & 1) ? 0x7fffffffu : ~0u, (mods & 4) ? 0x7fffffffu : ~0u, (mods & 16) ? 0x7fffffffu : ~0u};
  const uint32_t negm[3] = {(mods & 2) ? 0x80000000u : 0u, (mods & 8) ? 0x80000000u : 0u, (mods & 32) ? 0x80000000u : 0u};
#if VGPU_SIMD_AVX2
  for (unsigned i = 0; i < 32; i += 4) {
    const auto load = [&](const uint32_t* lo, const uint32_t* hi, unsigned k) {
      __m128i l = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lo + i));
      __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hi + i));
      h = _mm_xor_si128(_mm_and_si128(h, _mm_set1_epi32(static_cast<int>(absm[k]))), _mm_set1_epi32(static_cast<int>(negm[k])));
      return _mm256_castsi256_pd(_mm256_or_si256(_mm256_cvtepu32_epi64(l), _mm256_slli_epi64(_mm256_cvtepu32_epi64(h), 32)));
    };
    const __m256d x = load(alo, ahi, 0), y = load(blo, bhi, 1);
    __m256d v;
    if (kind == 0) v = _mm256_add_pd(x, y);
    else if (kind == 1) v = _mm256_mul_pd(x, y);
    else v = _mm256_fmadd_pd(x, y, load(clo, chi, 2));
    const __m256i u = _mm256_castpd_si256(v);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(rlo + i),
                     _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(u, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0))));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(rhi + i),
                     _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(u, _mm256_setr_epi32(1, 3, 5, 7, 0, 0, 0, 0))));
  }
#else
  for (unsigned i = 0; i < 32; ++i) {
    const auto get = [&](const uint32_t* lo, const uint32_t* hi, unsigned k) {
      return std::bit_cast<double>(uint64_t{lo[i]} | (uint64_t{(hi[i] & absm[k]) ^ negm[k]} << 32));
    };
    const double x = get(alo, ahi, 0), y = get(blo, bhi, 1);
    const double v = kind == 0 ? x + y : kind == 1 ? x * y : __builtin_fma(x, y, get(clo, chi, 2));
    const uint64_t u = std::bit_cast<uint64_t>(v);
    rlo[i] = static_cast<uint32_t>(u);
    rhi[i] = static_cast<uint32_t>(u >> 32);
  }
#endif
  store_masked(dlo, rlo, mask, 32);
  store_masked(dhi, rhi, mask, 32);
}

#if VGPU_SIMD_AVX2
// Four doubles to four binary16 (in the low 64 bits): to nearest even in one rounding, a NaN the canonical 0x7fff.
// The integer form of f64_to_f16_bits (simd/fp16.hpp), whose comments say why each constant is what it is.
inline __m128i f64x4_to_f16x4(__m256d v) {
  const __m256i one = _mm256_set1_epi64x(1);
  const __m256i u = _mm256_castpd_si256(v);
  const __m256i mag = _mm256_and_si256(u, _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFll));
  const __m256i sign16 = _mm256_and_si256(_mm256_srli_epi64(u, 48), _mm256_set1_epi64x(0x8000));
  const __m256i is_nan = _mm256_cmpgt_epi64(mag, _mm256_set1_epi64x(0x7FF0000000000000ll));
  const __m256i is_ovf = _mm256_cmpgt_epi64(mag, _mm256_set1_epi64x(0x40EFFDFFFFFFFFFFll));    // >= 65520
  const __m256i is_norm = _mm256_cmpgt_epi64(mag, _mm256_set1_epi64x(0x3F0FFFFFFFFFFFFFll));   // >= 2^-14
  const __m256i is_sub = _mm256_cmpgt_epi64(mag, _mm256_set1_epi64x(0x3E5FFFFFFFFFFFFFll));    // >= 2^-25
  // A normal half: re-bias the exponent and round the 42 dropped bits to even.
  const __m256i r = _mm256_sub_epi64(mag, _mm256_set1_epi64x(1008ll << 52));
  const __m256i lsb = _mm256_and_si256(_mm256_srli_epi64(r, 42), one);
  const __m256i vn = _mm256_srli_epi64(
      _mm256_add_epi64(_mm256_add_epi64(r, _mm256_set1_epi64x((1ll << 41) - 1)), lsb), 42);
  // A subnormal half: the 53-bit significand shifted to the half's grid, 28 - e places with e the unbiased exponent
  // (a count past 63 gives zero, which is what a value below 2^-25 rounds to).
  const __m256i ef = _mm256_srli_epi64(mag, 52);
  const __m256i shift = _mm256_sub_epi64(_mm256_set1_epi64x(1051), ef);
  const __m256i sig =
      _mm256_or_si256(_mm256_and_si256(mag, _mm256_set1_epi64x((1ll << 52) - 1)), _mm256_set1_epi64x(1ll << 52));
  const __m256i half = _mm256_sllv_epi64(one, _mm256_sub_epi64(shift, one));
  const __m256i lsbq = _mm256_and_si256(_mm256_srlv_epi64(sig, shift), one);
  const __m256i q =
      _mm256_srlv_epi64(_mm256_add_epi64(_mm256_add_epi64(sig, _mm256_sub_epi64(half, one)), lsbq), shift);
  __m256i res = _mm256_and_si256(is_sub, q);
  res = _mm256_blendv_epi8(res, vn, is_norm);
  res = _mm256_blendv_epi8(res, _mm256_set1_epi64x(0x7C00), is_ovf);
  res = _mm256_or_si256(res, sign16);
  res = _mm256_blendv_epi8(res, _mm256_set1_epi64x(0x7FFF), is_nan);
  const __m256i lo = _mm256_permutevar8x32_epi32(res, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0));
  return _mm_packus_epi32(_mm256_castsi256_si128(lo), _mm256_castsi256_si128(lo));
}
#endif

void f64_to_f16_canon_k(const double* in, uint16_t* out, size_t n) {
  size_t i = 0;
#if VGPU_SIMD_AVX2
  for (; i + 4 <= n; i += 4) {
    const __m128i h = f64x4_to_f16x4(_mm256_loadu_pd(in + i));
    const uint64_t w = static_cast<uint64_t>(_mm_cvtsi128_si64(h));
    std::memcpy(out + i, &w, sizeof w);
  }
#endif
  for (; i < n; ++i) out[i] = in[i] != in[i] ? uint16_t{0x7fff} : f64_to_f16_bits(in[i]);
}

// d[l] = the fused multiply-add of the lanes' f16 halves, `halves` of them (1: the low half, the high half of the
// result zero), as double-precision fused multiply-adds each rounded once to binary16 -- NaN the canonical 0x7fff.
void hfma2_m_k(uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t mask, unsigned n,
               unsigned halves) {
  uint32_t r[64];
#if VGPU_SIMD_AVX2
  for (unsigned i = 0; i < n; i += 4) {
    // Four words are eight halves, in memory order: word w's low half then its high half.
    const __m256 fa = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
    const __m256 fb = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
    const __m256 fc = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c + i)));
    const __m256d v0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(fa)),
                                       _mm256_cvtps_pd(_mm256_castps256_ps128(fb)),
                                       _mm256_cvtps_pd(_mm256_castps256_ps128(fc)));
    const __m256d v1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(fa, 1)),
                                       _mm256_cvtps_pd(_mm256_extractf128_ps(fb, 1)),
                                       _mm256_cvtps_pd(_mm256_extractf128_ps(fc, 1)));
    const __m128i h0 = f64x4_to_f16x4(v0), h1 = f64x4_to_f16x4(v1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(r + i), _mm_unpacklo_epi64(h0, h1));
  }
#else
  for (unsigned l = 0; l < n; ++l) {
    uint32_t out = 0;
    for (unsigned h = 0; h < 2; ++h) {
      const double v = __builtin_fma(static_cast<double>(f16_to_f32(static_cast<uint16_t>(a[l] >> (16 * h)))),
                                     static_cast<double>(f16_to_f32(static_cast<uint16_t>(b[l] >> (16 * h)))),
                                     static_cast<double>(f16_to_f32(static_cast<uint16_t>(c[l] >> (16 * h)))));
      out |= static_cast<uint32_t>(v != v ? uint16_t{0x7fff} : f64_to_f16_bits(v)) << (16 * h);
    }
    r[l] = out;
  }
#endif
  if (halves == 1)
    for (unsigned l = 0; l < n; ++l) r[l] &= 0xffffu;
  store_masked(d, r, mask, n);
}

}  // namespace VGPU_SIMD_NS
