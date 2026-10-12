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

}  // namespace VGPU_SIMD_NS
