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
// VGPU_LANES_ONLY (1 in the JIT's generated code, src/sass/jit.cpp, which includes this same text): leaves out the kernels the
// SASS ALU forms do not use (the half conversions, HMMA, binary64, the half-precision fused forms), which need the simulator's headers.

namespace VGPU_SIMD_NS {

typedef float V8f __attribute__((vector_size(32)));
// (A half array read through the 32-bit words that hold it.)
typedef uint16_t __attribute__((may_alias)) U16A;

#if !VGPU_LANES_ONLY
// n a multiple of 8.
inline void f16_to_f32_blocks(const U16A* in, float* out, size_t n) {
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

#endif  // !VGPU_LANES_ONLY
typedef float V4f __attribute__((vector_size(16)));
typedef int32_t V4i __attribute__((vector_size(16)));
typedef uint32_t V4u __attribute__((vector_size(16)));
typedef uint32_t V8u __attribute__((vector_size(32)));
typedef uint16_t V8h __attribute__((vector_size(16)));

static_assert(std::endian::native == std::endian::little, "the fragment kernels read a register's halves in memory order");

#if !VGPU_LANES_ONLY
// n 32-bit words of 16-bit (kind 0, 1) or tf32 (kind 2) operands, as floats, in memory order: word w's low half is
// out[2w], its high half out[2w + 1]. n_words is a multiple of 4.
inline void operands_to_f32(unsigned kind, const uint32_t* w, float* out, size_t n_words) {
  if (kind == 0) {
    f16_to_f32_blocks(reinterpret_cast<const U16A*>(w), out, 2 * n_words);
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
  // Bit 4 of `kind` asks for an RTX 3060's tensor-core sum (vgpu/exec/tc_ampere.hpp) in place of the sequential f32 one.
  const bool ampere = (kind & 16) != 0;
  kind &= 3;
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
    f16_to_f32_blocks(reinterpret_cast<const U16A*>(c), Cm, 128);
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
  if (ampere) {
    vgpu::exec::tc86::mma(16, 8, K, kind == 2 ? 4 : 8, kind == 0 ? vgpu::exec::tc86::kEminF16 : vgpu::exec::tc86::kEminWide, acc_f16, A, K, Bm, 8, Cm, 8,
                          Dm, 8);
  } else
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

#endif  // !VGPU_LANES_ONLY
// ---- 32-lane ALU kernels (the SASS executor's fast_alu) ----------------------------------------------------------

// A float result as the register holds it: a NaN is the canonical 0x7fffffff, whatever its payload or sign.
// (A macro, not a function: a 32-byte vector parameter changes the ABI on a baseline x86-64 build.)
#define VGPU_CANON_NAN(v) \
  ({ const V8f f_ = (v); const V8u m_ = reinterpret_cast<V8u>(f_ != f_); (reinterpret_cast<V8u>(f_) & ~m_) | (0x7fffffffu & m_); })

VGPU_KN void iadd3_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t add) {
  for (unsigned i = 0; i < 32; i += 8) {
    V8u x, y, z;
    std::memcpy(&x, a + i, sizeof x);
    std::memcpy(&y, b + i, sizeof y);
    std::memcpy(&z, c + i, sizeof z);
    const V8u v = x + y + z + add;
    std::memcpy(r + i, &v, sizeof v);
  }
}

VGPU_KN void imad_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c) {
  for (unsigned i = 0; i < 32; i += 8) {
    V8u x, y, z;
    std::memcpy(&x, a + i, sizeof x);
    std::memcpy(&y, b + i, sizeof y);
    std::memcpy(&z, c + i, sizeof z);
    const V8u v = x * y + z;
    std::memcpy(r + i, &v, sizeof v);
  }
}

// (A macro, not a function taking the vector type: a 32-byte vector parameter changes the ABI on a baseline x86-64 build.)
#define VGPU_LOP3_LOOP(EXPR)                          \
  do {                                                \
    for (unsigned i = 0; i < 32; i += 8) {            \
      V8u x, y, z;                                    \
      std::memcpy(&x, a + i, sizeof x);               \
      std::memcpy(&y, b + i, sizeof y);               \
      std::memcpy(&z, c + i, sizeof z);               \
      const V8u v = (EXPR);                           \
      std::memcpy(r + i, &v, sizeof v);               \
    }                                                 \
    return;                                           \
  } while (0)

VGPU_KN void lop3_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t lut) {
  // The tables compilers use most -- and, xor, or of two or three, a pass-through, a complement, and the few two-level ones --
  // by the expression itself (bit 4a + 2b + c of the table is the answer for those a, b, c); tests/unit/test_simd.cpp checks
  // every table.
  switch (lut) {
    case 0xc0: VGPU_LOP3_LOOP(x & y);
    case 0x3c: VGPU_LOP3_LOOP(x ^ y);
    case 0xfc: VGPU_LOP3_LOOP(x | y);
    case 0xf0: VGPU_LOP3_LOOP(x);
    case 0xcc: VGPU_LOP3_LOOP(y);
    case 0xaa: VGPU_LOP3_LOOP(z);
    case 0x0f: VGPU_LOP3_LOOP(~x);
    case 0x33: VGPU_LOP3_LOOP(~y);
    case 0x55: VGPU_LOP3_LOOP(~z);
    case 0x96: VGPU_LOP3_LOOP(x ^ y ^ z);
    case 0x80: VGPU_LOP3_LOOP(x & y & z);
    case 0xfe: VGPU_LOP3_LOOP(x | y | z);
    case 0xf8: VGPU_LOP3_LOOP(x | (y & z));
    case 0xe0: VGPU_LOP3_LOOP(x & (y | z));
    case 0xea: VGPU_LOP3_LOOP((x & y) | z);
    case 0xe8: VGPU_LOP3_LOOP((x & y) | (z & (x | y)));
    case 0x30: VGPU_LOP3_LOOP(x & ~y);
    case 0x0c: VGPU_LOP3_LOOP(~x & y);
    case 0xc3: VGPU_LOP3_LOOP(~(x ^ y));
    default: break;
  }
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

VGPU_KN void ffma_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c) {
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

VGPU_KN void fadd_k(uint32_t* r, const uint32_t* a, const uint32_t* b) {
  for (unsigned i = 0; i < 32; i += 8) {
    V8f x, y;
    std::memcpy(&x, a + i, sizeof x);
    std::memcpy(&y, b + i, sizeof y);
    const V8u u = VGPU_CANON_NAN(x + y);
    std::memcpy(r + i, &u, sizeof u);
  }
}

VGPU_KN void fmul_k(uint32_t* r, const uint32_t* a, const uint32_t* b) {
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
VGPU_KA inline void store_masked(uint32_t* d, const uint32_t* r, uint64_t mask, unsigned n) {
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
VGPU_KA inline uint64_t icmp_t(const uint32_t* a, const uint32_t* b, unsigned n) {
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

VGPU_KN uint64_t icmp_mask_k(unsigned cmp, bool sgn, const uint32_t* a, const uint32_t* b, unsigned n) {
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

#if !VGPU_LANES_ONLY
// ---- binary64 lanes held as register pairs (the SASS executor: the low words of a register, the high words of the next)

// kind 0: a + b, 1: a * b, 2: fma(a, b, c); `mods` has bit 2i for |source i| and bit 2i + 1 for -source i (the
// absolute value first). d, a, b, c are (low, high) pairs of 32-lane arrays; only the lanes of `mask` are written.
//
// When an operand is a NaN, which one the result is made from is decided by simd::dop_nan (simd.hpp), from the operands as the
// registers hold them: the |x| and -x modifiers apply to the arithmetic, not to a NaN that is the answer.
#if VGPU_SIMD_AVX2
// Eight lanes at a time. A register pair's two halves come as the low words and the high words of 8 lanes; unpacking
// them against each other gives the doubles of lanes 0 1 4 5 (the low unpack) and 2 3 6 7 (the high unpack), and the
// two shuffles at the end put the result's words back in lane order. A chunk of 8 lanes is read in full before it is
// written, and a lane's result depends on that lane alone, so d may be a source (a register of the pair as well).
//
// A NaN operand gives a NaN result, whatever the operation, so the result is what is looked at: a lane whose result is
// not a NaN had no NaN operand, and the rest are given the rule of dop_nan (a result that is a NaN with no NaN operand,
// inf - inf, keeps the host's).
template <unsigned Kind, bool Mods>
inline void dop_avx2_t(const uint32_t* absm, const uint32_t* negm, uint32_t* dlo, uint32_t* dhi, const uint32_t* alo,
                       const uint32_t* ahi, const uint32_t* blo, const uint32_t* bhi, const uint32_t* clo,
                       const uint32_t* chi, uint32_t mask) {
  __m256i absv[3], negv[3];
  for (int k = 0; k < 3; ++k) {
    absv[k] = _mm256_set1_epi32(static_cast<int>(absm[k]));
    negv[k] = _mm256_set1_epi32(static_cast<int>(negm[k]));
  }
  const auto load = [&](const uint32_t* lo, const uint32_t* hi, unsigned k, unsigned i, __m256d& p, __m256d& q) {
    const __m256i l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lo + i));
    __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hi + i));
    if constexpr (Mods) h = _mm256_xor_si256(_mm256_and_si256(h, absv[k]), negv[k]);
    p = _mm256_castsi256_pd(_mm256_unpacklo_epi32(l, h));
    q = _mm256_castsi256_pd(_mm256_unpackhi_epi32(l, h));
  };
  const auto load_raw = [&](const uint32_t* lo, const uint32_t* hi, unsigned i, __m256d& p, __m256d& q) {
    const __m256i l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lo + i));
    const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hi + i));
    p = _mm256_castsi256_pd(_mm256_unpacklo_epi32(l, h));
    q = _mm256_castsi256_pd(_mm256_unpackhi_epi32(l, h));
  };
  const auto is_nan = [](uint64_t u) { return (u & 0x7fffffffffffffffull) > 0x7ff0000000000000ull; };
  const __m256i pick = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  for (unsigned i = 0; i < 32; i += 8) {
    const uint32_t bits = (mask >> i) & 0xffu;
    __m256d xa, xb, ya, yb, za = _mm256_setzero_pd(), zb = _mm256_setzero_pd();
    load(alo, ahi, 0, i, xa, xb);
    load(blo, bhi, 1, i, ya, yb);
    __m256d va, vb;
    if constexpr (Kind == 0) {
      va = _mm256_add_pd(xa, ya);
      vb = _mm256_add_pd(xb, yb);
    } else if constexpr (Kind == 1) {
      va = _mm256_mul_pd(xa, ya);
      vb = _mm256_mul_pd(xb, yb);
    } else {
      load(clo, chi, 2, i, za, zb);
      va = _mm256_fmadd_pd(xa, ya, za);
      vb = _mm256_fmadd_pd(xb, yb, zb);
    }
    if (_mm256_movemask_pd(_mm256_or_pd(_mm256_cmp_pd(va, va, _CMP_UNORD_Q), _mm256_cmp_pd(vb, vb, _CMP_UNORD_Q)))) {
      // Rare: a NaN result. The operands and results of the 8 lanes, in lane order, and the rule applied to each.
      alignas(32) double X[8], Y[8], Z[8], R[8];
      // (The operands as the registers hold them, for the NaN rule: reloaded without the modifiers.)
      if constexpr (Mods) {
        load_raw(alo, ahi, i, xa, xb);
        load_raw(blo, bhi, i, ya, yb);
        if constexpr (Kind == 2) load_raw(clo, chi, i, za, zb);
      }
      const auto spill = [](double* o, __m256d p, __m256d q) {
        _mm_storeu_pd(o, _mm256_castpd256_pd128(p));
        _mm_storeu_pd(o + 4, _mm256_extractf128_pd(p, 1));
        _mm_storeu_pd(o + 2, _mm256_castpd256_pd128(q));
        _mm_storeu_pd(o + 6, _mm256_extractf128_pd(q, 1));
      };
      spill(X, xa, xb);
      spill(Y, ya, yb);
      spill(Z, za, zb);
      spill(R, va, vb);
      for (unsigned k = 0; k < 8; ++k) {
        const uint64_t ua = std::bit_cast<uint64_t>(X[k]), ub = std::bit_cast<uint64_t>(Y[k]),
                       uc = Kind == 2 ? std::bit_cast<uint64_t>(Z[k]) : 0;
        if (is_nan(ua) || is_nan(ub) || (Kind == 2 && is_nan(uc))) R[k] = std::bit_cast<double>(dop_nan(Kind, ua, ub, uc));
      }
      va = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(R)), _mm_loadu_pd(R + 4), 1);
      vb = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(R + 2)), _mm_loadu_pd(R + 6), 1);
    }
    const __m256i rlo = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castpd_ps(va), _mm256_castpd_ps(vb), 0x88));
    const __m256i rhi = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castpd_ps(va), _mm256_castpd_ps(vb), 0xDD));
    if (bits == 0xffu) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dlo + i), rlo);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dhi + i), rhi);
    } else if (bits) {
      const __m256i m = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(static_cast<int>(bits)), pick), pick);
      _mm256_maskstore_epi32(reinterpret_cast<int*>(dlo + i), m, rlo);
      _mm256_maskstore_epi32(reinterpret_cast<int*>(dhi + i), m, rhi);
    }
  }
}
#endif

void dop_m_k(unsigned kind, unsigned mods, uint32_t* dlo, uint32_t* dhi, const uint32_t* alo, const uint32_t* ahi,
             const uint32_t* blo, const uint32_t* bhi, const uint32_t* clo, const uint32_t* chi, uint32_t mask) {
  const uint32_t absm[3] = {(mods & 1) ? 0x7fffffffu : ~0u, (mods & 4) ? 0x7fffffffu : ~0u, (mods & 16) ? 0x7fffffffu : ~0u};
  const uint32_t negm[3] = {(mods & 2) ? 0x80000000u : 0u, (mods & 8) ? 0x80000000u : 0u, (mods & 32) ? 0x80000000u : 0u};
#if VGPU_SIMD_AVX2
  if ((mods & 63u) == 0) {
    if (kind == 0) dop_avx2_t<0, false>(absm, negm, dlo, dhi, alo, ahi, blo, bhi, clo, chi, mask);
    else if (kind == 1) dop_avx2_t<1, false>(absm, negm, dlo, dhi, alo, ahi, blo, bhi, clo, chi, mask);
    else dop_avx2_t<2, false>(absm, negm, dlo, dhi, alo, ahi, blo, bhi, clo, chi, mask);
  } else {
    if (kind == 0) dop_avx2_t<0, true>(absm, negm, dlo, dhi, alo, ahi, blo, bhi, clo, chi, mask);
    else if (kind == 1) dop_avx2_t<1, true>(absm, negm, dlo, dhi, alo, ahi, blo, bhi, clo, chi, mask);
    else dop_avx2_t<2, true>(absm, negm, dlo, dhi, alo, ahi, blo, bhi, clo, chi, mask);
  }
#else
  uint32_t rlo[32], rhi[32];
  const auto bits = [&](const uint32_t* lo, const uint32_t* hi, unsigned i, unsigned k) {
    return uint64_t{lo[i]} | (uint64_t{(hi[i] & absm[k]) ^ negm[k]} << 32);
  };
  const auto raw = [&](const uint32_t* lo, const uint32_t* hi, unsigned i) { return uint64_t{lo[i]} | (uint64_t{hi[i]} << 32); };
  const auto is_nan = [](uint64_t u) { return (u & 0x7fffffffffffffffull) > 0x7ff0000000000000ull; };
  for (unsigned i = 0; i < 32; ++i) {
    const uint64_t ua = bits(alo, ahi, i, 0), ub = bits(blo, bhi, i, 1), uc = kind == 2 ? bits(clo, chi, i, 2) : 0;
    uint64_t u;
    if (is_nan(ua) || is_nan(ub) || (kind == 2 && is_nan(uc))) {
      u = dop_nan(kind, raw(alo, ahi, i), raw(blo, bhi, i), kind == 2 ? raw(clo, chi, i) : 0);
    } else {
      const double x = std::bit_cast<double>(ua), y = std::bit_cast<double>(ub);
      const double v = kind == 0 ? x + y : kind == 1 ? x * y : __builtin_fma(x, y, std::bit_cast<double>(uc));
      u = std::bit_cast<uint64_t>(v);
    }
    rlo[i] = static_cast<uint32_t>(u);
    rhi[i] = static_cast<uint32_t>(u >> 32);
  }
  store_masked(dlo, rlo, mask, 32);
  store_masked(dhi, rhi, mask, 32);
#endif
}

#endif  // !VGPU_LANES_ONLY
// ---- wider integer forms of the SASS executor's fast_alu (32 lanes) ---------------------------------------------------

VGPU_KN void blend_m_k(uint32_t* d, const uint32_t* r, uint32_t mask) { store_masked(d, r, mask, 32); }

#if VGPU_SIMD_AVX2
// Eight lanes of a 32-bit vector widened to 64-bit lanes, in the order the AVX2 unpacks give: `p` holds lanes 0 1 4 5
// and `q` lanes 2 3 6 7 (the high dword of each is `hi`, or zero). pair_pack_lo/hi put 64-bit results back in lane order.
VGPU_KA inline __m256i pair_pack_lo(__m256i p, __m256i q) {
  return _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(p), _mm256_castsi256_ps(q), 0x88));
}
VGPU_KA inline __m256i pair_pack_hi(__m256i p, __m256i q) {
  return _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(p), _mm256_castsi256_ps(q), 0xDD));
}
// Bit k of the low 8 bits of `bits` as an all-ones dword in lane k.
VGPU_KA inline __m256i lane_mask_dwords(uint32_t bits) {
  const __m256i pick = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  return _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(static_cast<int>(bits & 0xffu)), pick), pick);
}
#endif

VGPU_KN void iadd3x_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t add, uint32_t cin_a,
              uint32_t cin_b, uint32_t* ge1, uint32_t* ge2) {
  uint32_t o1 = 0, o2 = 0;
#if VGPU_SIMD_AVX2
  const __m256i zero = _mm256_setzero_si256();
  for (unsigned i = 0; i < 32; i += 8) {
    // The addend and the two carries in (each 0 or 1 a lane), one number a lane.
    const __m256i k = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_set1_epi32(static_cast<int>(add)), lane_mask_dwords(cin_a >> i)),
                                       lane_mask_dwords(cin_b >> i));
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    const __m256i z = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + i));
    const __m256i p = _mm256_add_epi64(_mm256_add_epi64(_mm256_unpacklo_epi32(x, zero), _mm256_unpacklo_epi32(y, zero)),
                                       _mm256_add_epi64(_mm256_unpacklo_epi32(z, zero), _mm256_unpacklo_epi32(k, zero)));
    const __m256i q = _mm256_add_epi64(_mm256_add_epi64(_mm256_unpackhi_epi32(x, zero), _mm256_unpackhi_epi32(y, zero)),
                                       _mm256_add_epi64(_mm256_unpackhi_epi32(z, zero), _mm256_unpackhi_epi32(k, zero)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i), pair_pack_lo(p, q));
    const __m256i over = pair_pack_hi(p, q);   // the carries out of 32 bits: 0 to 3
    o1 |= static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(over, zero)))) << i;
    o2 |= static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(over, _mm256_set1_epi32(1))))) << i;
  }
#else
  for (unsigned i = 0; i < 32; ++i) {
    const uint64_t sum = uint64_t{a[i]} + b[i] + c[i] + add + ((cin_a >> i) & 1u) + ((cin_b >> i) & 1u);
    r[i] = static_cast<uint32_t>(sum);
    o1 |= static_cast<uint32_t>((sum >> 32) >= 1) << i;
    o2 |= static_cast<uint32_t>((sum >> 32) >= 2) << i;
  }
#endif
  *ge1 = o1;
  *ge2 = o2;
}

#if VGPU_SIMD_AVX2
template <bool Sgn>
VGPU_KA inline void imad_wide_avx2(uint32_t* rlo, uint32_t* rhi, const uint32_t* a, const uint32_t* b, const uint32_t* clo,
                           const uint32_t* chi, uint32_t cin, uint32_t* carry_out) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i sign = _mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ull));
  uint32_t out = 0;
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    const __m256i cl = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(clo + i));
    const __m256i ch = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(chi + i));
    const __m256i cm = lane_mask_dwords(cin >> i);
    const auto half = [&](bool high, __m256i& res, __m256i& carry) {
      const __m256i xa = high ? _mm256_unpackhi_epi32(x, zero) : _mm256_unpacklo_epi32(x, zero);
      const __m256i ya = high ? _mm256_unpackhi_epi32(y, zero) : _mm256_unpacklo_epi32(y, zero);
      const __m256i prod = Sgn ? _mm256_mul_epi32(xa, ya) : _mm256_mul_epu32(xa, ya);
      const __m256i cc = high ? _mm256_unpackhi_epi32(cl, ch) : _mm256_unpacklo_epi32(cl, ch);
      const __m256i ci = _mm256_srli_epi64(high ? _mm256_unpackhi_epi32(cm, cm) : _mm256_unpacklo_epi32(cm, cm), 63);
      const __m256i s1 = _mm256_add_epi64(prod, cc);
      const __m256i s2 = _mm256_add_epi64(s1, ci);
      // Unsigned: a sum below an addend wrapped. (cmpgt is signed; the sign bit flipped on both makes it unsigned.)
      const __m256i c1 = _mm256_cmpgt_epi64(_mm256_xor_si256(prod, sign), _mm256_xor_si256(s1, sign));
      const __m256i c2 = _mm256_cmpgt_epi64(_mm256_xor_si256(s1, sign), _mm256_xor_si256(s2, sign));
      res = s2;
      carry = _mm256_or_si256(c1, c2);
    };
    __m256i p, q, cp, cq;
    half(false, p, cp);
    half(true, q, cq);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(rlo + i), pair_pack_lo(p, q));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(rhi + i), pair_pack_hi(p, q));
    out |= static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(pair_pack_lo(cp, cq)))) << i;
  }
  *carry_out = out;
}
#endif

VGPU_KN void imad_wide_k(uint32_t* rlo, uint32_t* rhi, const uint32_t* a, const uint32_t* b, const uint32_t* clo,
                 const uint32_t* chi, bool sgn, uint32_t cin, uint32_t* carry_out) {
#if VGPU_SIMD_AVX2
  if (sgn) imad_wide_avx2<true>(rlo, rhi, a, b, clo, chi, cin, carry_out);
  else imad_wide_avx2<false>(rlo, rhi, a, b, clo, chi, cin, carry_out);
#else
  uint32_t out = 0;
  for (unsigned i = 0; i < 32; ++i) {
    const uint64_t prod = sgn ? static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(a[i])) * static_cast<int32_t>(b[i]))
                              : static_cast<uint64_t>(a[i]) * b[i];
    const uint64_t c = uint64_t{clo[i]} | (uint64_t{chi[i]} << 32);
    const unsigned __int128 full = static_cast<unsigned __int128>(prod) + c + ((cin >> i) & 1u);
    rlo[i] = static_cast<uint32_t>(full);
    rhi[i] = static_cast<uint32_t>(static_cast<uint64_t>(full) >> 32);
    out |= static_cast<uint32_t>((full >> 64) != 0) << i;
  }
  *carry_out = out;
#endif
}

#if VGPU_SIMD_AVX2
template <bool Right, bool Sgn, bool Hi>
VGPU_KA inline void shf_avx2(uint32_t* r, const uint32_t* lo, const uint32_t* n, const uint32_t* top, bool wrap, unsigned width) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i wmask = _mm256_set1_epi32(static_cast<int>(width - 1)), wmax = _mm256_set1_epi32(static_cast<int>(width));
  const __m256i k63 = _mm256_set1_epi64x(63);
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256i l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lo + i));
    const __m256i t = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(top + i));
    __m256i k = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(n + i));
    k = wrap ? _mm256_and_si256(k, wmask) : _mm256_min_epu32(k, wmax);   // (a power of two: n % width is n & (width - 1))
    const auto half = [&](bool high) {
      const __m256i v = high ? _mm256_unpackhi_epi32(l, t) : _mm256_unpacklo_epi32(l, t);
      const __m256i kk = high ? _mm256_unpackhi_epi32(k, zero) : _mm256_unpacklo_epi32(k, zero);
      if constexpr (!Right) {
        return _mm256_sllv_epi64(v, kk);   // (a count of 64 or more gives zero)
      } else if constexpr (Sgn) {
        const __m256i s = _mm256_cmpgt_epi64(zero, v);
        return _mm256_xor_si256(_mm256_srlv_epi64(_mm256_xor_si256(v, s), _mm256_min_epu32(kk, k63)), s);
      } else {
        return _mm256_srlv_epi64(v, kk);
      }
    };
    const __m256i p = half(false), q = half(true);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i), Hi ? pair_pack_hi(p, q) : pair_pack_lo(p, q));
  }
}
#endif

VGPU_KN void shf_k(uint32_t* r, const uint32_t* lo, const uint32_t* n, const uint32_t* top, unsigned type, bool wrap,
           bool right, bool hi) {
  const bool is64 = type < 2, sgn = type == 0 || type == 2;
  const unsigned width = is64 ? 64 : 32;
#if VGPU_SIMD_AVX2
  if (!right) {
    if (hi) shf_avx2<false, false, true>(r, lo, n, top, wrap, width);
    else shf_avx2<false, false, false>(r, lo, n, top, wrap, width);
  } else if (sgn) {
    if (hi) shf_avx2<true, true, true>(r, lo, n, top, wrap, width);
    else shf_avx2<true, true, false>(r, lo, n, top, wrap, width);
  } else {
    if (hi) shf_avx2<true, false, true>(r, lo, n, top, wrap, width);
    else shf_avx2<true, false, false>(r, lo, n, top, wrap, width);
  }
#else
  for (unsigned l = 0; l < 32; ++l) {
    const uint64_t v = lo[l] | (static_cast<uint64_t>(top[l]) << 32);
    const uint32_t k = wrap ? n[l] % width : std::min(n[l], width);
    uint64_t x;
    if (!right) x = k >= 64 ? 0 : v << k;
    else if (sgn) x = static_cast<uint64_t>(static_cast<int64_t>(v) >> std::min(k, 63u));
    else x = k >= 64 ? 0 : v >> k;
    r[l] = static_cast<uint32_t>(hi ? x >> 32 : x);
  }
#endif
}

// One-operand lane ops of the SASS executor, 32 lanes: kind 0 POPC, 1 BREV (the bits reversed), 2 + (sgn ? 1 : 0) + (sh ? 2 : 0)
// FLO (the index of the highest set bit, of the complement of a negative value if `sgn`; all ones for none; counted
// from the other end, 31 - index, if `sh`).
void lane_unary_k(unsigned kind, uint32_t* r, const uint32_t* a) {
  if (kind == 0) {
    for (unsigned i = 0; i < 32; i += 8) {
      V8u x;
      std::memcpy(&x, a + i, sizeof x);
      x = x - ((x >> 1) & 0x55555555u);
      x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
      x = (x + (x >> 4)) & 0x0f0f0f0fu;
      x = (x * 0x01010101u) >> 24;
      std::memcpy(r + i, &x, sizeof x);
    }
  } else if (kind == 1) {
    for (unsigned i = 0; i < 32; i += 8) {
      V8u x;
      std::memcpy(&x, a + i, sizeof x);
      x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
      x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
      x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
      x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
      x = (x >> 16) | (x << 16);
      std::memcpy(r + i, &x, sizeof x);
    }
  } else {
    const bool sgn = ((kind - 2) & 1) != 0, sh = ((kind - 2) & 2) != 0;
    for (unsigned l = 0; l < 32; ++l) {
      uint32_t v = a[l];
      if (sgn && (v >> 31)) v = ~v;
      uint32_t res = v ? 31u - static_cast<uint32_t>(__builtin_clz(v)) : ~0u;
      if (sh && res != ~0u) res = 31u - res;
      r[l] = res;
    }
  }
}

// ---- float forms of the SASS executor's fast path: flush and saturation, compares, MUFU, conversions, min and max ---------

// (Scalar helpers: the base level's loops, and the lanes of the AVX2 ones that the instructions cannot do.)
inline uint32_t sc_ftz(uint32_t x) { return (x & 0x7f800000u) == 0 ? (x & 0x80000000u) : x; }
inline float sc_sat(float f) { return f != f ? 0.0f : f <= 0.0f ? 0.0f : f > 1.0f ? 1.0f : f; }   // (-0 is +0: see sat() in exec_ops.inc)
// A float result as the register holds it: flushed if flags bit 0, saturated if bit 1, a NaN the canonical one.
inline uint32_t sc_finish(float f, unsigned flags) {
  uint32_t u = std::bit_cast<uint32_t>(f);
  if (flags & 1) u = sc_ftz(u);
  float g = std::bit_cast<float>(u);
  if (flags & 2) g = sc_sat(g);
  return g != g ? 0x7fffffffu : std::bit_cast<uint32_t>(g);
}
inline float sc_in(uint32_t x, unsigned flags) { return std::bit_cast<float>((flags & 1) ? sc_ftz(x) : x); }

#if VGPU_SIMD_AVX2
VGPU_KA inline __m256i fx_flush(__m256i x) {
  const __m256i m = _mm256_cmpeq_epi32(_mm256_and_si256(x, _mm256_set1_epi32(0x7f800000)), _mm256_setzero_si256());
  return _mm256_blendv_epi8(x, _mm256_and_si256(x, _mm256_set1_epi32(static_cast<int>(0x80000000u))), m);
}
VGPU_KA inline __m256i fx_finish(__m256 v, unsigned flags) {
  __m256i u = _mm256_castps_si256(v);
  if (flags & 1) u = fx_flush(u);
  if (flags & 2) {
    const __m256 f = _mm256_castsi256_ps(u), zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    const __m256 low = _mm256_or_ps(_mm256_cmp_ps(f, zero, _CMP_LE_OQ), _mm256_cmp_ps(f, f, _CMP_UNORD_Q));
    const __m256 g = _mm256_blendv_ps(_mm256_blendv_ps(f, zero, low), one, _mm256_cmp_ps(f, one, _CMP_GT_OQ));
    u = _mm256_castps_si256(g);
  }
  const __m256 f = _mm256_castsi256_ps(u);
  return _mm256_blendv_epi8(u, _mm256_set1_epi32(0x7fffffff), _mm256_castps_si256(_mm256_cmp_ps(f, f, _CMP_UNORD_Q)));
}
VGPU_KA inline __m256i fx_load(const uint32_t* p, unsigned flags) {
  const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return (flags & 1) ? fx_flush(x) : x;
}
#endif

VGPU_KN void ffma_x_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* c, unsigned flags) {
#if VGPU_SIMD_AVX2
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256 v = _mm256_fmadd_ps(_mm256_castsi256_ps(fx_load(a + i, flags)), _mm256_castsi256_ps(fx_load(b + i, flags)),
                                     _mm256_castsi256_ps(fx_load(c + i, flags)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i), fx_finish(v, flags));
  }
#else
  for (unsigned l = 0; l < 32; ++l) r[l] = sc_finish(__builtin_fmaf(sc_in(a[l], flags), sc_in(b[l], flags), sc_in(c[l], flags)), flags);
#endif
}

VGPU_KN void fadd_x_k(uint32_t* r, const uint32_t* a, const uint32_t* b, unsigned flags) {
#if VGPU_SIMD_AVX2
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256 v = _mm256_add_ps(_mm256_castsi256_ps(fx_load(a + i, flags)), _mm256_castsi256_ps(fx_load(b + i, flags)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i), fx_finish(v, flags));
  }
#else
  for (unsigned l = 0; l < 32; ++l) r[l] = sc_finish(sc_in(a[l], flags) + sc_in(b[l], flags), flags);
#endif
}

VGPU_KN void fmul_x_k(uint32_t* r, const uint32_t* a, const uint32_t* b, unsigned flags) {
#if VGPU_SIMD_AVX2
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256 v = _mm256_mul_ps(_mm256_castsi256_ps(fx_load(a + i, flags)), _mm256_castsi256_ps(fx_load(b + i, flags)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i), fx_finish(v, flags));
  }
#else
  for (unsigned l = 0; l < 32; ++l) r[l] = sc_finish(sc_in(a[l], flags) * sc_in(b[l], flags), flags);
#endif
}

// SASS's 4-bit float comparison code, on two floats (as the executor's float_cmp does it on their doubles).
inline bool sc_fcmp(unsigned cmp, float a, float b) {
  const bool un = a != a || b != b;
  switch (cmp) {
    case 0: return false;
    case 1: return !un && a < b;
    case 2: return !un && a == b;
    case 3: return !un && a <= b;
    case 4: return !un && a > b;
    case 5: return !un && a != b;
    case 6: return !un && a >= b;
    case 7: return !un;
    case 8: return un;
    case 9: return un || a < b;
    case 10: return un || a == b;
    case 11: return un || a <= b;
    case 12: return un || a > b;
    case 13: return un || a != b;
    case 14: return un || a >= b;
    default: return true;
  }
}

#if VGPU_SIMD_AVX2
template <int Pred>
VGPU_KA inline uint32_t fcmp_avx2(const uint32_t* a, const uint32_t* b, unsigned flags) {
  uint32_t out = 0;
  for (unsigned i = 0; i < 32; i += 8)
    out |= static_cast<uint32_t>(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_castsi256_ps(fx_load(a + i, flags)),
                                                                  _mm256_castsi256_ps(fx_load(b + i, flags)), Pred)))
           << i;
  return out;
}
#endif

VGPU_KN uint32_t fcmp_mask_k(unsigned cmp, bool ftz, const uint32_t* a, const uint32_t* b) {
#if VGPU_SIMD_AVX2
  const unsigned fl = ftz ? 1u : 0u;
  switch (cmp) {
    case 0: return 0;
    case 1: return fcmp_avx2<_CMP_LT_OQ>(a, b, fl);
    case 2: return fcmp_avx2<_CMP_EQ_OQ>(a, b, fl);
    case 3: return fcmp_avx2<_CMP_LE_OQ>(a, b, fl);
    case 4: return fcmp_avx2<_CMP_GT_OQ>(a, b, fl);
    case 5: return fcmp_avx2<_CMP_NEQ_OQ>(a, b, fl);
    case 6: return fcmp_avx2<_CMP_GE_OQ>(a, b, fl);
    case 7: return fcmp_avx2<_CMP_ORD_Q>(a, b, fl);
    case 8: return fcmp_avx2<_CMP_UNORD_Q>(a, b, fl);
    case 9: return fcmp_avx2<_CMP_NGE_UQ>(a, b, fl);
    case 10: return fcmp_avx2<_CMP_EQ_UQ>(a, b, fl);
    case 11: return fcmp_avx2<_CMP_NGT_UQ>(a, b, fl);
    case 12: return fcmp_avx2<_CMP_NLE_UQ>(a, b, fl);
    case 13: return fcmp_avx2<_CMP_NEQ_UQ>(a, b, fl);
    case 14: return fcmp_avx2<_CMP_NLT_UQ>(a, b, fl);
    default: return ~0u;
  }
#else
  uint32_t out = 0;
  for (unsigned l = 0; l < 32; ++l) out |= static_cast<uint32_t>(sc_fcmp(cmp, sc_in(a[l], ftz), sc_in(b[l], ftz))) << l;
  return out;
#endif
}

// MUFU.RCP (kind 4), .RSQ (5), .SQRT (8): in double, rounded once to float; the input flushed first, the result after.
void mufu_k(unsigned kind, uint32_t* r, const uint32_t* a) {
#if VGPU_SIMD_AVX2
  const __m256d one = _mm256_set1_pd(1.0);
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256 x = _mm256_castsi256_ps(fx_load(a + i, 1));
    const auto op = [&](__m128 h) {
      const __m256d d = _mm256_cvtps_pd(h);
      const __m256d v = kind == 4 ? _mm256_div_pd(one, d) : kind == 5 ? _mm256_div_pd(one, _mm256_sqrt_pd(d)) : _mm256_sqrt_pd(d);
      return _mm256_cvtpd_ps(v);
    };
    const __m256 v = _mm256_set_m128(op(_mm256_extractf128_ps(x, 1)), op(_mm256_castps256_ps128(x)));
    // (A NaN result -- the root of a negative number, an operand that was a NaN -- is the canonical one, as on an RTX 3060.)
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i),
                        _mm256_blendv_epi8(fx_flush(_mm256_castps_si256(v)), _mm256_set1_epi32(0x7fffffff),
                                           _mm256_castps_si256(_mm256_cmp_ps(v, v, _CMP_UNORD_Q))));
  }
#else
  for (unsigned l = 0; l < 32; ++l) {
    const double x = sc_in(a[l], 1);
    const float f = static_cast<float>(kind == 4 ? 1.0 / x : kind == 5 ? 1.0 / __builtin_sqrt(x) : __builtin_sqrt(x));
    r[l] = f != f ? 0x7fffffffu : sc_ftz(std::bit_cast<uint32_t>(f));
  }
#endif
}

// I2F: an s32 or u32 to f32, to nearest even.
VGPU_KN void i2f_k(bool sgn, uint32_t* r, const uint32_t* a) {
#if VGPU_SIMD_AVX2
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256 v;
    if (sgn) {
      v = _mm256_cvtepi32_ps(x);
    } else {
      // As doubles, which hold a u32 exactly: the low 31 bits, plus 2^31 where the top bit is set. (Both parts are never
      // negative, so a zero stays +0 in every rounding mode -- the sum of 2^31 and -2^31, which flipping the sign bit and adding
      // 2^31 back would make, is -0 when the host rounds down.) The narrowing to float uses the host's rounding mode.
      const __m256i low = _mm256_and_si256(x, _mm256_set1_epi32(0x7fffffff));
      const __m256i top = _mm256_srli_epi32(x, 31);
      const __m256d two31 = _mm256_set1_pd(2147483648.0);
      const auto wide = [&](__m128i l, __m128i t) { return _mm256_add_pd(_mm256_cvtepi32_pd(l), _mm256_mul_pd(_mm256_cvtepi32_pd(t), two31)); };
      const __m256d lo = wide(_mm256_castsi256_si128(low), _mm256_castsi256_si128(top));
      const __m256d hi = wide(_mm256_extracti128_si256(low, 1), _mm256_extracti128_si256(top, 1));
      v = _mm256_set_m128(_mm256_cvtpd_ps(hi), _mm256_cvtpd_ps(lo));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i), _mm256_castps_si256(v));
  }
#else
  for (unsigned l = 0; l < 32; ++l)
    r[l] = std::bit_cast<uint32_t>(sgn ? static_cast<float>(static_cast<int64_t>(static_cast<int32_t>(a[l])))
                                       : static_cast<float>(static_cast<int64_t>(a[l])));
#endif
}

// F2I to 32 bits: f32 (flushed first if ftz) rounded by rnd (0 to nearest even, 1 down, 2 up, 3 towards zero), a NaN giving 0, an
// out-of-range value the nearest end of the signed or unsigned range.
inline uint32_t sc_f2i(bool sgn, unsigned rnd, float f) {
  double x = f;
  switch (rnd) {
    case 0: x = __builtin_nearbyint(x); break;
    case 1: x = __builtin_floor(x); break;
    case 2: x = __builtin_ceil(x); break;
    default: x = __builtin_trunc(x); break;
  }
  uint64_t v;
  if (x != x) v = 0;
  else if (sgn) v = static_cast<uint64_t>(static_cast<int64_t>(x < -2147483648.0 ? -2147483648.0 : x > 2147483647.0 ? 2147483647.0 : x));
  else v = static_cast<uint64_t>(x < 0 ? 0.0 : x > 4294967295.0 ? 4294967295.0 : x);
  return static_cast<uint32_t>(v);
}

#if VGPU_SIMD_AVX2
// The four 64-bit lane masks of a __m256d compare as four 32-bit lane masks.
inline __m128i narrow_mask_pd(__m256d m) {
  return _mm_castps_si128(_mm_shuffle_ps(_mm256_castps256_ps128(_mm256_castpd_ps(m)), _mm256_extractf128_ps(_mm256_castpd_ps(m), 1), 0x88));
}

template <int Mode>
inline void f2i_avx2(bool sgn, bool ftz, uint32_t* r, const uint32_t* a) {
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256 x = _mm256_castsi256_ps(fx_load(a + i, ftz ? 1u : 0u));
    const auto half = [&](__m128 h) -> __m128i {
      const __m256d d = _mm256_round_pd(_mm256_cvtps_pd(h), Mode | _MM_FROUND_NO_EXC);
      __m128i v;
      if (sgn) {
        const __m256d c = _mm256_min_pd(_mm256_max_pd(d, _mm256_set1_pd(-2147483648.0)), _mm256_set1_pd(2147483647.0));
        v = _mm256_cvttpd_epi32(c);
      } else {
        // Below 2^31 as it is; from there, 2^31 less (which the signed conversion takes) and the top bit put back.
        const __m256d c = _mm256_min_pd(_mm256_max_pd(d, _mm256_setzero_pd()), _mm256_set1_pd(4294967295.0));
        const __m256d big = _mm256_cmp_pd(c, _mm256_set1_pd(2147483648.0), _CMP_GE_OQ);
        const __m256d adj = _mm256_sub_pd(c, _mm256_and_pd(big, _mm256_set1_pd(2147483648.0)));
        v = _mm_or_si128(_mm256_cvttpd_epi32(adj), _mm_and_si128(narrow_mask_pd(big), _mm_set1_epi32(static_cast<int>(0x80000000u))));
      }
      // (max and min give their second operand for a NaN: the NaN lanes are cleared here)
      return _mm_andnot_si128(narrow_mask_pd(_mm256_cmp_pd(d, d, _CMP_UNORD_Q)), v);
    };
    const __m128i lo = half(_mm256_castps256_ps128(x)), hi = half(_mm256_extractf128_ps(x, 1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i), _mm256_set_m128i(hi, lo));
  }
}
#endif

void f2i_k(bool sgn, unsigned rnd, bool ftz, uint32_t* r, const uint32_t* a) {
#if VGPU_SIMD_AVX2
  switch (rnd) {
    case 0: f2i_avx2<_MM_FROUND_TO_NEAREST_INT>(sgn, ftz, r, a); break;
    case 1: f2i_avx2<_MM_FROUND_TO_NEG_INF>(sgn, ftz, r, a); break;
    case 2: f2i_avx2<_MM_FROUND_TO_POS_INF>(sgn, ftz, r, a); break;
    default: f2i_avx2<_MM_FROUND_TO_ZERO>(sgn, ftz, r, a); break;
  }
#else
  for (unsigned l = 0; l < 32; ++l) r[l] = sc_f2i(sgn, rnd, sc_in(a[l], ftz ? 1u : 0u));
#endif
}

// FMNMX: lane l is the minimum of a and b if bit l of `min_mask` is set, else the maximum; a NaN loses (the other operand
// is the answer; two NaNs give the canonical NaN, as on an RTX 3060), of two zeros the minimum is the negative one, the maximum the positive one; with `nan` a
// NaN operand gives the canonical NaN. The operands are flushed first if ftz.
inline uint32_t sc_fmnmx(float a, float b, bool mn, bool nan) {
  if ((nan && (a != a || b != b)) || (a != a && b != b)) return 0x7fffffffu;
  float r;
  if (a != a) r = b;
  else if (b != b) r = a;
  else if (a == 0 && b == 0) r = mn ? (__builtin_signbit(a) ? a : b) : (__builtin_signbit(a) ? b : a);
  else r = mn ? (b < a ? b : a) : (a < b ? b : a);
  return std::bit_cast<uint32_t>(r);
}

void fmnmx_k(uint32_t* r, const uint32_t* a, const uint32_t* b, uint32_t min_mask, bool ftz, bool nan) {
#if VGPU_SIMD_AVX2
  const unsigned fl = ftz ? 1u : 0u;
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256 x = _mm256_castsi256_ps(fx_load(a + i, fl)), y = _mm256_castsi256_ps(fx_load(b + i, fl));
    const __m256 mn = _mm256_castsi256_ps(lane_mask_dwords(min_mask >> i));
    const __m256 xnan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q), ynan = _mm256_cmp_ps(y, y, _CMP_UNORD_Q);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 both_zero = _mm256_and_ps(_mm256_cmp_ps(x, zero, _CMP_EQ_OQ), _mm256_cmp_ps(y, zero, _CMP_EQ_OQ));
    const __m256 xneg = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(x), 31));   // all ones where a's sign bit is set
    // min: b < a ? b : a; max: a < b ? b : a (no NaN, no two zeros here)
    const __m256 vmin = _mm256_min_ps(y, x), vmax = _mm256_max_ps(y, x);
    __m256 v = _mm256_blendv_ps(vmax, vmin, mn);
    const __m256 zmin = _mm256_blendv_ps(y, x, xneg), zmax = _mm256_blendv_ps(x, y, xneg);
    v = _mm256_blendv_ps(v, _mm256_blendv_ps(zmax, zmin, mn), both_zero);
    v = _mm256_blendv_ps(v, x, ynan);
    v = _mm256_blendv_ps(v, y, xnan);
    v = _mm256_blendv_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)), nan ? _mm256_or_ps(xnan, ynan) : _mm256_and_ps(xnan, ynan));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i), _mm256_castps_si256(v));
  }
#else
  for (unsigned l = 0; l < 32; ++l) r[l] = sc_fmnmx(sc_in(a[l], ftz ? 1u : 0u), sc_in(b[l], ftz ? 1u : 0u), (min_mask >> l) & 1, nan);
#endif
}

// PRMT in its default mode: lane l's result byte i is byte (n & 7) of the 8 bytes (b : a) -- a's four, then b's four -- for
// the i-th nibble n of sel; if bit 3 of the nibble is set, that byte's sign bit replicated through the byte instead.
inline uint32_t sc_prmt0(uint32_t a, uint32_t b, uint32_t sel) {
  const uint64_t x = (static_cast<uint64_t>(b) << 32) | a;
  uint32_t r = 0;
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned s = (sel >> (4 * i)) & 0xf;
    uint32_t v = static_cast<uint32_t>((x >> (8 * (s & 7))) & 0xff);
    if (s & 8) v = (v & 0x80) ? 0xff : 0;
    r |= v << (8 * i);
  }
  return r;
}

VGPU_KN void prmt0_k(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* sel) {
#if VGPU_SIMD_AVX2
  // Per 128-bit half, four lanes' words. The nibbles of each selector spread to four bytes; each byte then picks from its own lane's
  // word of a or of b (pshufb stays within a 128-bit half, and a lane's words are the same dword in both), and the sign
  // replication is a signed compare of the picked byte.
  const __m256i spread = _mm256_setr_epi8(0, 0, 1, 1, 4, 4, 5, 5, 8, 8, 9, 9, 12, 12, 13, 13, 0, 0, 1, 1, 4, 4, 5, 5, 8, 8, 9, 9, 12, 12, 13, 13);
  const __m256i base = _mm256_setr_epi8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12);
  const __m256i zero = _mm256_setzero_si256();
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    const __m256i t = _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sel + i)), spread);
    const __m256i nib = _mm256_or_si256(_mm256_and_si256(t, _mm256_set1_epi32(0x000f000f)),
                                        _mm256_and_si256(_mm256_srli_epi16(t, 4), _mm256_set1_epi32(0x0f000f00)));
    const __m256i idx = _mm256_add_epi8(_mm256_and_si256(nib, _mm256_set1_epi8(3)), base);
    const __m256i from_b = _mm256_cmpeq_epi8(_mm256_and_si256(nib, _mm256_set1_epi8(4)), _mm256_set1_epi8(4));
    __m256i v = _mm256_blendv_epi8(_mm256_shuffle_epi8(va, idx), _mm256_shuffle_epi8(vb, idx), from_b);
    const __m256i sign = _mm256_cmpeq_epi8(_mm256_and_si256(nib, _mm256_set1_epi8(8)), _mm256_set1_epi8(8));
    v = _mm256_blendv_epi8(v, _mm256_cmpgt_epi8(zero, v), sign);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i), v);
  }
#else
  for (unsigned l = 0; l < 32; ++l) r[l] = sc_prmt0(a[l], b[l], sel[l]);
#endif
}

#if !VGPU_LANES_ONLY
// DSETP's comparison of two doubles held as register pairs (the low words of 32 lanes and the high words), after |x| and -x as
// dop_m's `mods` say (bit 2i |source i|, bit 2i + 1 -source i; sources a, b are 0, 1). For the codes of float compares (1 to 14 --
// as fcmp_mask) bit l of *r is the comparison. Code 0 is MIN and 15 MAX: bit l of *r is whether a is the answer (a NaN loses; of two
// equal values MIN takes a if its sign is set and MAX if not), and bit l of *both_nan whether both are NaN. *both_nan is
// set to 0 for the other codes.
void dcmp_k(unsigned cmp, unsigned mods, const uint32_t* alo, const uint32_t* ahi, const uint32_t* blo, const uint32_t* bhi,
            uint32_t* r, uint32_t* both_nan) {
  const uint32_t absm[2] = {(mods & 1) ? 0x7fffffffu : ~0u, (mods & 4) ? 0x7fffffffu : ~0u};
  const uint32_t negm[2] = {(mods & 2) ? 0x80000000u : 0u, (mods & 8) ? 0x80000000u : 0u};
  uint32_t out = 0, bn = 0;
#if VGPU_SIMD_AVX2
  const __m256i absv0 = _mm256_set1_epi32(static_cast<int>(absm[0])), absv1 = _mm256_set1_epi32(static_cast<int>(absm[1]));
  const __m256i negv0 = _mm256_set1_epi32(static_cast<int>(negm[0])), negv1 = _mm256_set1_epi32(static_cast<int>(negm[1]));
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256i al = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(alo + i));
    const __m256i ah = _mm256_xor_si256(_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ahi + i)), absv0), negv0);
    const __m256i bl = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blo + i));
    const __m256i bh = _mm256_xor_si256(_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(bhi + i)), absv1), negv1);
    // (lanes 0 1 4 5, then 2 3 6 7)
    const __m256d xa = _mm256_castsi256_pd(_mm256_unpacklo_epi32(al, ah)), xb = _mm256_castsi256_pd(_mm256_unpackhi_epi32(al, ah));
    const __m256d ya = _mm256_castsi256_pd(_mm256_unpacklo_epi32(bl, bh)), yb = _mm256_castsi256_pd(_mm256_unpackhi_epi32(bl, bh));
    const auto bits_of = [](__m256d p, __m256d q) {   // 64-bit lane masks -> 8 bits in lane order
      return static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(pair_pack_lo(_mm256_castpd_si256(p), _mm256_castpd_si256(q)))));
    };
    uint32_t res;
    if (cmp == 0 || cmp == 15) {
      const __m256d an_a = _mm256_cmp_pd(xa, xa, _CMP_UNORD_Q), an_b = _mm256_cmp_pd(xb, xb, _CMP_UNORD_Q);
      const __m256d bn_a = _mm256_cmp_pd(ya, ya, _CMP_UNORD_Q), bn_b = _mm256_cmp_pd(yb, yb, _CMP_UNORD_Q);
      const auto pick = [&](__m256d x, __m256d y, __m256d anan, __m256d bnan) {
        const __m256d eq = _mm256_cmp_pd(x, y, _CMP_EQ_OQ);
        const __m256d cmpv = cmp == 0 ? _mm256_cmp_pd(x, y, _CMP_LT_OQ) : _mm256_cmp_pd(x, y, _CMP_GT_OQ);
        const __m256d neg = _mm256_castsi256_pd(_mm256_srai_epi32(_mm256_shuffle_epi32(_mm256_castpd_si256(x), 0xF5), 31));   // a's sign bit, through the lane
        const __m256d same = cmp == 0 ? neg : _mm256_xor_pd(neg, _mm256_castsi256_pd(_mm256_set1_epi64x(-1)));
        __m256d v = _mm256_blendv_pd(cmpv, same, eq);                                              // equal: by sign; else by order
        v = _mm256_blendv_pd(v, _mm256_xor_pd(anan, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))), _mm256_or_pd(anan, bnan));   // a NaN loses
        return v;
      };
      res = bits_of(pick(xa, ya, an_a, bn_a), pick(xb, yb, an_b, bn_b));
      bn |= bits_of(_mm256_and_pd(an_a, bn_a), _mm256_and_pd(an_b, bn_b)) << i;
    } else {
      // (the same predicates as the float compares)
      __m256d ra, rb;
      switch (cmp) {
#define VGPU_DCMP_CASE(N, P) case N: ra = _mm256_cmp_pd(xa, ya, P); rb = _mm256_cmp_pd(xb, yb, P); break;
        VGPU_DCMP_CASE(1, _CMP_LT_OQ) VGPU_DCMP_CASE(2, _CMP_EQ_OQ) VGPU_DCMP_CASE(3, _CMP_LE_OQ) VGPU_DCMP_CASE(4, _CMP_GT_OQ)
        VGPU_DCMP_CASE(5, _CMP_NEQ_OQ) VGPU_DCMP_CASE(6, _CMP_GE_OQ) VGPU_DCMP_CASE(7, _CMP_ORD_Q) VGPU_DCMP_CASE(8, _CMP_UNORD_Q)
        VGPU_DCMP_CASE(9, _CMP_NGE_UQ) VGPU_DCMP_CASE(10, _CMP_EQ_UQ) VGPU_DCMP_CASE(11, _CMP_NGT_UQ) VGPU_DCMP_CASE(12, _CMP_NLE_UQ)
        VGPU_DCMP_CASE(13, _CMP_NEQ_UQ) VGPU_DCMP_CASE(14, _CMP_NLT_UQ)
#undef VGPU_DCMP_CASE
        default: ra = rb = _mm256_castsi256_pd(_mm256_set1_epi64x(-1)); break;
      }
      res = bits_of(ra, rb);
    }
    out |= res << i;
  }
#else
  for (unsigned l = 0; l < 32; ++l) {
    const double a = std::bit_cast<double>(uint64_t{alo[l]} | (uint64_t{(ahi[l] & absm[0]) ^ negm[0]} << 32));
    const double b = std::bit_cast<double>(uint64_t{blo[l]} | (uint64_t{(bhi[l] & absm[1]) ^ negm[1]} << 32));
    const bool an = a != a, bnn = b != b;
    bool v;
    if (cmp == 0 || cmp == 15) {
      if (an || bnn) v = !an;
      else if (a == b) v = cmp == 0 ? __builtin_signbit(a) : !__builtin_signbit(a);
      else v = cmp == 0 ? a < b : a > b;
      bn |= static_cast<uint32_t>(an && bnn) << l;
    } else {
      const bool un = an || bnn;
      switch (cmp) {
        case 1: v = !un && a < b; break;
        case 2: v = !un && a == b; break;
        case 3: v = !un && a <= b; break;
        case 4: v = !un && a > b; break;
        case 5: v = !un && a != b; break;
        case 6: v = !un && a >= b; break;
        case 7: v = !un; break;
        case 8: v = un; break;
        case 9: v = un || a < b; break;
        case 10: v = un || a == b; break;
        case 11: v = un || a <= b; break;
        case 12: v = un || a > b; break;
        case 13: v = un || a != b; break;
        case 14: v = un || a >= b; break;
        default: v = true; break;
      }
    }
    out |= static_cast<uint32_t>(v) << l;
  }
#endif
  *r = out;
  *both_nan = bn;
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

// The bfloat16 twin of hfma2_m_k: each half of d[l] is the fused multiply-add of the lanes' bf16 halves, rounded to bfloat16 as
// the SASS executor's general form rounds it -- the exact result in binary64, then to binary32 (to nearest even), then to
// bfloat16 (to nearest even) -- a NaN the canonical 0x7fff. A bf16 product has 16 significant bits, so it is exact in binary64
// and the multiply-add is the product plus the addend rounded once, which is what a fused one gives (no FMA unit needed).
// n a multiple of 4.
typedef double V4d __attribute__((vector_size(32)));
void hfma2_bf_m_k(uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint64_t mask, unsigned n) {
  uint32_t r[64];
  for (unsigned i = 0; i < n; i += 4) {
    // Four words are eight halves, in memory order: word w's low half then its high half.
    alignas(32) double f[8];
    for (int part = 0; part < 2; ++part) {
      // Written through a pointer: returning a 32-byte vector by value from a function compiled without AVX
      // changes the ABI (GCC's -Wpsabi, an error under -Werror).
      const auto wide = [&](const uint32_t* p, double* out) {
        V8h h;
        std::memcpy(&h, p + i, sizeof h);
        const V8u u = __builtin_convertvector(h, V8u) << 16;
        V4u q;
        std::memcpy(&q, reinterpret_cast<const char*>(&u) + 16 * part, sizeof q);
        const V4d v = __builtin_convertvector(reinterpret_cast<V4f>(q), V4d);
        std::memcpy(out, &v, sizeof v);
      };
      alignas(32) double x[4], y[4], z[4];
      wide(a, x);
      wide(b, y);
      wide(c, z);
      for (int k = 0; k < 4; ++k) f[4 * part + k] = fma_round_odd(x[k], y[k], z[k]);
    }
    // Rounded once, from the double (a float in between rounded twice; an RTX 3060 rounds once).
    U16A* const out = reinterpret_cast<U16A*>(r + i);
    for (unsigned k = 0; k < 8; ++k) out[k] = f[k] != f[k] ? uint16_t{0x7fff} : static_cast<uint16_t>(narrow_from_double(f[k], 4, 0));
  }
  store_masked(d, r, mask, n);
}

// ---- tile products: the tensor-core instructions' sums over whole tiles ------------------------------------------------------

typedef int32_t V8i __attribute__((vector_size(32)));

// R rows by V vectors of 8 columns of D = C + A x B in f32, k in order, each product rounded and then added (never fused). A is M x K
// (row stride K), B is K x N (row stride N), C and D are M x N with row strides ldc and ldd (C null: zeros). N is a multiple of 8.
template <unsigned R, unsigned V, bool Canon>
inline void tile_f32_block(unsigned i0, unsigned j0, unsigned N, unsigned K, const float* A, const float* B, const float* C, size_t ldc, float* D,
                           size_t ldd) {
  V8f acc[R][V];
  for (unsigned r = 0; r < R; ++r)
    for (unsigned v = 0; v < V; ++v) {
      if (C) std::memcpy(&acc[r][v], C + size_t{i0 + r} * ldc + j0 + 8 * v, sizeof(V8f));
      else acc[r][v] = V8f{};
    }
  for (unsigned k = 0; k < K; ++k) {
    V8f b[V];
    for (unsigned v = 0; v < V; ++v) std::memcpy(&b[v], B + size_t{k} * N + j0 + 8 * v, sizeof(V8f));
    for (unsigned r = 0; r < R; ++r) {
      const float a = A[size_t{i0 + r} * K + k];
      for (unsigned v = 0; v < V; ++v) {
        const V8f p = a * b[v];
        acc[r][v] = acc[r][v] + p;
      }
    }
  }
  for (unsigned r = 0; r < R; ++r)
    for (unsigned v = 0; v < V; ++v) {
      if (Canon) {
        // every NaN the host's additions kept becomes the canonical 0x7fffffff (the float compare is false for a NaN only)
        const V8u nan = reinterpret_cast<V8u>(acc[r][v] != acc[r][v]);
        acc[r][v] = reinterpret_cast<V8f>((reinterpret_cast<V8u>(acc[r][v]) & ~nan) | (0x7fffffffu & nan));
      }
      std::memcpy(D + size_t{i0 + r} * ldd + j0 + 8 * v, &acc[r][v], sizeof(V8f));
    }
}

// `Canon`: a NaN in the result is the canonical 0x7fffffff (tile_f32c); else whichever NaN the host's additions kept.
template <bool Canon>
inline void tile_f32_impl(unsigned M, unsigned N, unsigned K, const float* A, const float* B, const float* C, size_t ldc, float* D, size_t ldd) {
  unsigned i0 = 0;
  for (; i0 + 4 <= M; i0 += 4) {
    unsigned j0 = 0;
    for (; j0 + 16 <= N; j0 += 16) tile_f32_block<4, 2, Canon>(i0, j0, N, K, A, B, C, ldc, D, ldd);
    for (; j0 + 8 <= N; j0 += 8) tile_f32_block<4, 1, Canon>(i0, j0, N, K, A, B, C, ldc, D, ldd);
  }
  for (; i0 < M; ++i0) {
    unsigned j0 = 0;
    for (; j0 + 16 <= N; j0 += 16) tile_f32_block<1, 2, Canon>(i0, j0, N, K, A, B, C, ldc, D, ldd);
    for (; j0 + 8 <= N; j0 += 8) tile_f32_block<1, 1, Canon>(i0, j0, N, K, A, B, C, ldc, D, ldd);
  }
}

void tile_f32_k(unsigned M, unsigned N, unsigned K, const float* A, const float* B, const float* C, size_t ldc, float* D, size_t ldd) {
  tile_f32_impl<false>(M, N, K, A, B, C, ldc, D, ldd);
}
void tile_f32c_k(unsigned M, unsigned N, unsigned K, const float* A, const float* B, const float* C, size_t ldc, float* D, size_t ldd) {
  tile_f32_impl<true>(M, N, K, A, B, C, ldc, D, ldd);
}

// The same for 32-bit integers (the operands are small: 8-bit elements or less, so a block's sum of products fits in 32 bits). Without `sat`
// one wrapping sum of C and every product; with it the products are summed `block` at a time and C, or the running value, plus a block's
// sum is clamped to the s32 range after each block (the sum of an int64 accumulator clamped, as the engines' scalar form has it).
template <unsigned R>
inline void tile_i32_block(unsigned i0, unsigned j0, unsigned N, unsigned K, const int32_t* A, const int32_t* B, const int32_t* C, size_t ldc,
                           unsigned block, bool sat, int32_t* D, size_t ldd) {
  V8u acc[R];
  for (unsigned r = 0; r < R; ++r) {
    if (C) std::memcpy(&acc[r], C + size_t{i0 + r} * ldc + j0, sizeof(V8u));
    else acc[r] = V8u{};
  }
  if (!sat) {
    for (unsigned k = 0; k < K; ++k) {
      V8u b;
      std::memcpy(&b, B + size_t{k} * N + j0, sizeof b);
      for (unsigned r = 0; r < R; ++r) acc[r] = acc[r] + static_cast<uint32_t>(A[size_t{i0 + r} * K + k]) * b;
    }
  } else {
    for (unsigned k0 = 0; k0 < K; k0 += block) {
      V8u s[R] = {};
      const unsigned k1 = k0 + block < K ? k0 + block : K;
      for (unsigned k = k0; k < k1; ++k) {
        V8u b;
        std::memcpy(&b, B + size_t{k} * N + j0, sizeof b);
        for (unsigned r = 0; r < R; ++r) s[r] = s[r] + static_cast<uint32_t>(A[size_t{i0 + r} * K + k]) * b;
      }
      for (unsigned r = 0; r < R; ++r) {
        const V8u sum = acc[r] + s[r];
        // Signed overflow of acc + s: both operands of one sign and the sum of the other.
        const V8i ov = reinterpret_cast<V8i>((acc[r] ^ sum) & (s[r] ^ sum)) < 0;
        // acc negative: INT32_MIN (0x80000000), else INT32_MAX (0x7fffffff)
        const V8u neg = reinterpret_cast<V8u>(reinterpret_cast<V8i>(acc[r]) >> 31);
        const V8u lim = (neg & 0x80000000u) | (~neg & 0x7fffffffu);
        const V8u om = reinterpret_cast<V8u>(ov);
        acc[r] = (om & lim) | (~om & sum);
      }
    }
  }
  for (unsigned r = 0; r < R; ++r) std::memcpy(D + size_t{i0 + r} * ldd + j0, &acc[r], sizeof(V8u));
}

void tile_i32_k(unsigned M, unsigned N, unsigned K, const int32_t* A, const int32_t* B, const int32_t* C, size_t ldc, unsigned block, bool sat,
                int32_t* D, size_t ldd) {
  unsigned i0 = 0;
  for (; i0 + 4 <= M; i0 += 4)
    for (unsigned j0 = 0; j0 + 8 <= N; j0 += 8) tile_i32_block<4>(i0, j0, N, K, A, B, C, ldc, block, sat, D, ldd);
  for (; i0 < M; ++i0)
    for (unsigned j0 = 0; j0 + 8 <= N; j0 += 8) tile_i32_block<1>(i0, j0, N, K, A, B, C, ldc, block, sat, D, ldd);
}

// D = C + A x B in binary64, k in order, each step one fused multiply-add in the host's current rounding mode (the caller sets it).
void tile_f64_k(unsigned M, unsigned N, unsigned K, const double* A, const double* B, const double* C, double* D) {
  for (unsigned i = 0; i < M; ++i) {
    unsigned j = 0;
#if VGPU_SIMD_AVX2
    for (; j + 4 <= N; j += 4) {
      __m256d acc = C ? _mm256_loadu_pd(C + size_t{i} * N + j) : _mm256_setzero_pd();
      for (unsigned k = 0; k < K; ++k)
        acc = _mm256_fmadd_pd(_mm256_set1_pd(A[size_t{i} * K + k]), _mm256_loadu_pd(B + size_t{k} * N + j), acc);
      _mm256_storeu_pd(D + size_t{i} * N + j, acc);
    }
#endif
    for (; j < N; ++j) {
      double acc = C ? C[size_t{i} * N + j] : 0.0;
      for (unsigned k = 0; k < K; ++k) acc = __builtin_fma(A[size_t{i} * K + k], B[size_t{k} * N + j], acc);
      D[size_t{i} * N + j] = acc;
    }
  }
}

// ---- f32 <-> f64 and 32-bit integer -> f64 (SASS F2F.F64.F32, F2F.F32.F64, I2F.F64.U32/.S32), 32 lanes: see Kernels::f2f_widen ----------------------
#if VGPU_SIMD_AVX2
// Eight binary64 lanes held as two vectors (lanes 0-3 and 4-7) to the low words and the high words of the eight lanes, in lane order.
inline void split_pd8(__m256d d0, __m256d d1, uint32_t* lo, uint32_t* hi) {
  const __m256 p0 = _mm256_castpd_ps(d0), p1 = _mm256_castpd_ps(d1);
  // (per 128 bits: lanes 0 1 4 5 | 2 3 6 7, as 64-bit pieces 0 2 1 3 of the result)
  const __m256i l = _mm256_castps_si256(_mm256_shuffle_ps(p0, p1, 0x88)), h = _mm256_castps_si256(_mm256_shuffle_ps(p0, p1, 0xDD));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(lo), _mm256_permute4x64_epi64(l, 0xD8));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(hi), _mm256_permute4x64_epi64(h, 0xD8));
}
#endif

void f2f_widen_k(const uint32_t* a, unsigned mods, uint32_t* lo, uint32_t* hi) {
  const uint32_t absm = (mods & 1) ? 0x7fffffffu : ~0u, negm = (mods & 2) ? 0x80000000u : 0u;
#if VGPU_SIMD_AVX2
  const __m256i va = _mm256_set1_epi32(static_cast<int>(absm)), vn = _mm256_set1_epi32(static_cast<int>(negm));
  const __m256i mag = _mm256_set1_epi32(0x7fffffff), inf = _mm256_set1_epi32(0x7f800000);
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    // A NaN goes through as the register holds it (its |x| and -x are not applied): the conversion shifts its payload up, quiets it and
    // keeps its sign, which is the value the card writes.
    const __m256i nan = _mm256_cmpgt_epi32(_mm256_and_si256(x, mag), inf);
    const __m256i y = _mm256_blendv_epi8(_mm256_xor_si256(_mm256_and_si256(x, va), vn), x, nan);
    split_pd8(_mm256_cvtps_pd(_mm_castsi128_ps(_mm256_castsi256_si128(y))), _mm256_cvtps_pd(_mm_castsi128_ps(_mm256_extracti128_si256(y, 1))), lo + i, hi + i);
  }
#else
  for (unsigned l = 0; l < 32; ++l) {
    const uint32_t x = a[l];
    const bool nan = (x & 0x7fffffffu) > 0x7f800000u;
    const uint32_t y = nan ? x : ((x & absm) ^ negm);
    double d = static_cast<double>(std::bit_cast<float>(y));
#if defined(__x86_64__)
    asm("" : "+x"(d));   // (the conversion of a NaN lane stays: the scalar form raises "invalid" for a signalling NaN too)
#endif
    const uint64_t conv = std::bit_cast<uint64_t>(d);
    const uint64_t b = nan ? ((uint64_t{x >> 31} << 63) | 0x7ff8000000000000ull | (uint64_t{x & 0x7fffffu} << 29)) : conv;
    lo[l] = static_cast<uint32_t>(b);
    hi[l] = static_cast<uint32_t>(b >> 32);
  }
#endif
}

void f2f_narrow_k(const uint32_t* alo, const uint32_t* ahi, unsigned mods, uint32_t* r) {
  const uint64_t absm = (mods & 1) ? 0x7fffffffffffffffull : ~uint64_t{0}, negm = (mods & 2) ? 0x8000000000000000ull : 0;
#if VGPU_SIMD_AVX2
  const __m256i va = _mm256_set1_epi64x(static_cast<long long>(absm)), vn = _mm256_set1_epi64x(static_cast<long long>(negm));
  const __m256i mag = _mm256_set1_epi64x(0x7fffffffffffffffll), inf = _mm256_set1_epi64x(0x7ff0000000000000ll);
  // Four doubles to four floats, in the host's rounding mode; a NaN is the register's own (not |x| or -x of it), converted as it is.
  const auto narrow4 = [&](__m256i x) {
    const __m256i nan = _mm256_cmpgt_epi64(_mm256_and_si256(x, mag), inf);
    const __m256i y = _mm256_blendv_epi8(_mm256_xor_si256(_mm256_and_si256(x, va), vn), x, nan);
    return _mm256_cvtpd_ps(_mm256_castsi256_pd(y));
  };
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256i l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(alo + i));
    const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ahi + i));
    const __m256i u = _mm256_unpacklo_epi32(l, h), v = _mm256_unpackhi_epi32(l, h);   // doubles 0 1 4 5 and 2 3 6 7
    const __m128 f0 = narrow4(_mm256_permute2x128_si256(u, v, 0x20)), f1 = narrow4(_mm256_permute2x128_si256(u, v, 0x31));
    _mm256_storeu_ps(reinterpret_cast<float*>(r + i), _mm256_set_m128(f1, f0));
  }
#else
  for (unsigned l = 0; l < 32; ++l) {
    const uint64_t x = alo[l] | (uint64_t{ahi[l]} << 32);
    const bool nan = (x & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
    const uint64_t y = nan ? x : ((x & absm) ^ negm);
    float f = static_cast<float>(std::bit_cast<double>(y));
#if defined(__x86_64__)
    asm("" : "+x"(f));
#endif
    r[l] = std::bit_cast<uint32_t>(f);
  }
#endif
}

void i2d_k(bool sgn, const uint32_t* a, uint32_t* lo, uint32_t* hi) {
#if VGPU_SIMD_AVX2
  const __m256i low31 = _mm256_set1_epi32(0x7fffffff);
  const __m256d two31 = _mm256_set1_pd(2147483648.0);
  for (unsigned i = 0; i < 32; i += 8) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256d d0, d1;
    if (sgn) {
      d0 = _mm256_cvtepi32_pd(_mm256_castsi256_si128(x));
      d1 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(x, 1));
    } else {
      // The low 31 bits plus 2^31 where the top bit is set: both parts are never negative, so the sum is exact and a zero is +0 in every
      // rounding mode (flipping the sign bit and adding 2^31 back would give -0 when the host rounds down).
      const __m256i m = _mm256_and_si256(x, low31), t = _mm256_srli_epi32(x, 31);
      d0 = _mm256_add_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(m)), _mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(t)), two31));
      d1 = _mm256_add_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(m, 1)), _mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(t, 1)), two31));
    }
    split_pd8(d0, d1, lo + i, hi + i);
  }
#else
  for (unsigned l = 0; l < 32; ++l) {
    const uint64_t b = std::bit_cast<uint64_t>(sgn ? static_cast<double>(static_cast<int32_t>(a[l])) : static_cast<double>(a[l]));
    lo[l] = static_cast<uint32_t>(b);
    hi[l] = static_cast<uint32_t>(b >> 32);
  }
#endif
}

#endif  // !VGPU_LANES_ONLY
}  // namespace VGPU_SIMD_NS
