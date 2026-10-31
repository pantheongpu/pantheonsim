// The AVX2 / FMA tile kernels of the AMD matrix instructions (mfma_math.hpp declares them), and the wide operand decoders of the matrix path
// (gcn_matrix_kernels.hpp), in a translation unit of their own that is compiled without link-time optimization (CMakeLists.txt) and with
// their functions on 64-byte and their loops on 32-byte boundaries. Their loops are a few instructions long and their speed depended on where in a 64-byte line
// they landed (4 to 11 % on a Zen 5 host, equal instruction counts), which changed with every unrelated edit of the library; in a unit
// of their own the layout depends only on this file. Same arithmetic, same bits as when they were inline.
#include "gcn_matrix_kernels.hpp"

#include "mfma_math.hpp"

namespace vgpu::amd {


#if defined(__x86_64__) && defined(__GNUC__)

__attribute__((target("avx2"))) bool mfma_tile_f_avx2(uint32_t m, uint32_t n, uint32_t k, const double* a, const double* bk,
                                                            const double* c, double* out) {
#define MFMA_LD4(p) (*reinterpret_cast<const mfma_v4d*>(p))
#define MFMA_ST4(p, v) (*reinterpret_cast<mfma_v4d*>(p) = (v))
  __m256d nan = _mm256_setzero_pd();
  for (uint32_t i = 0; i < m; i += 4) {
    const double* const a0 = a + size_t{i} * k;
    const double* const a1 = a0 + k;
    const double* const a2 = a1 + k;
    const double* const a3 = a2 + k;
    for (uint32_t j = 0; j < n; j += 8) {
      const double* const cr = c + size_t{i} * n + j;
      mfma_v4d s00 = MFMA_LD4(cr), s01 = MFMA_LD4(cr + 4), s10 = MFMA_LD4(cr + n), s11 = MFMA_LD4(cr + n + 4),
               s20 = MFMA_LD4(cr + 2 * n), s21 = MFMA_LD4(cr + 2 * n + 4), s30 = MFMA_LD4(cr + 3 * n), s31 = MFMA_LD4(cr + 3 * n + 4);
      const double* b = bk + j;
      for (uint32_t t = 0; t < k; ++t, b += n) {
        const mfma_v4d b0 = MFMA_LD4(b), b1 = MFMA_LD4(b + 4);
        const mfma_v4d x0 = {a0[t], a0[t], a0[t], a0[t]}, x1 = {a1[t], a1[t], a1[t], a1[t]}, x2 = {a2[t], a2[t], a2[t], a2[t]},
                       x3 = {a3[t], a3[t], a3[t], a3[t]};
        s00 += x0 * b0, s01 += x0 * b1;
        s10 += x1 * b0, s11 += x1 * b1;
        s20 += x2 * b0, s21 += x2 * b1;
        s30 += x3 * b0, s31 += x3 * b1;
      }
      double* const o = out + size_t{i} * n + j;
      MFMA_ST4(o, s00), MFMA_ST4(o + 4, s01), MFMA_ST4(o + n, s10), MFMA_ST4(o + n + 4, s11);
      MFMA_ST4(o + 2 * n, s20), MFMA_ST4(o + 2 * n + 4, s21), MFMA_ST4(o + 3 * n, s30), MFMA_ST4(o + 3 * n + 4, s31);
      // (a sum is a NaN if it differs from itself: the unordered compare of each with itself, or'd together)
#define MFMA_NAN4(v) nan = _mm256_or_pd(nan, _mm256_cmp_pd(v, v, _CMP_UNORD_Q))
      MFMA_NAN4(s00), MFMA_NAN4(s01), MFMA_NAN4(s10), MFMA_NAN4(s11), MFMA_NAN4(s20), MFMA_NAN4(s21), MFMA_NAN4(s30), MFMA_NAN4(s31);
#undef MFMA_NAN4
    }
  }
#undef MFMA_LD4
#undef MFMA_ST4
  return _mm256_movemask_pd(nan) != 0;
}

__attribute__((target("fma"))) void mfma_tile_fma_hw(uint32_t m, uint32_t n, uint32_t k, const double* a, const double* bk,
                                                           const double* c, double* out) {
  if (k == 4) {
    // The double-precision matrix instructions' K: four terms, the fused multiply-adds one after another on one sum, written out so that
    // the loop has no trip count to run: its few instructions were three times as slow in some positions of the code as in others.
    for (uint32_t i = 0; i < m; ++i) {
      const double* arow = a + size_t{i} * 4;
      for (uint32_t j = 0; j < n; ++j) {
        double sum = c[size_t{i} * n + j];
        sum = __builtin_fma(bk[j], arow[0], sum);
        sum = __builtin_fma(bk[size_t{n} + j], arow[1], sum);
        sum = __builtin_fma(bk[size_t{2} * n + j], arow[2], sum);
        sum = __builtin_fma(bk[size_t{3} * n + j], arow[3], sum);
        out[size_t{i} * n + j] = sum;
      }
    }
    return;
  }
  for (uint32_t i = 0; i < m; ++i)
    for (uint32_t j = 0; j < n; ++j) {
      double sum = c[size_t{i} * n + j];
      const double* arow = a + size_t{i} * k;
      for (uint32_t t = 0; t < k; ++t) sum = __builtin_fma(bk[size_t{t} * n + j], arow[t], sum);
      out[size_t{i} * n + j] = sum;
    }
}
// The same, four columns to an instruction (n a multiple of 4): a fused multiply-add is exact, so the width cannot show in the bits.
__attribute__((target("avx2,fma"))) void mfma_tile_fma_avx2(uint32_t m, uint32_t n, uint32_t k, const double* a, const double* bk,
                                                                  const double* c, double* out) {
  // (K = 4, the double-precision matrix instructions', is its own pair of loops: with the test inside the loop over columns, as inlined into a
  // caller the compiler knew K, but out of line it kept the test and the generic loop in the hot loop, 5 % slower.)
  if (k == 4) {
    for (uint32_t i = 0; i < m; ++i) {
      const double* const arow = a + size_t{i} * 4;
      for (uint32_t j = 0; j < n; j += 4) {
        const double* const b = bk + j;
        __m256d sum = _mm256_loadu_pd(c + size_t{i} * n + j);
        sum = _mm256_fmadd_pd(_mm256_loadu_pd(b), _mm256_broadcast_sd(arow), sum);
        sum = _mm256_fmadd_pd(_mm256_loadu_pd(b + n), _mm256_broadcast_sd(arow + 1), sum);
        sum = _mm256_fmadd_pd(_mm256_loadu_pd(b + 2 * size_t{n}), _mm256_broadcast_sd(arow + 2), sum);
        sum = _mm256_fmadd_pd(_mm256_loadu_pd(b + 3 * size_t{n}), _mm256_broadcast_sd(arow + 3), sum);
        _mm256_storeu_pd(out + size_t{i} * n + j, sum);
      }
    }
    return;
  }
  for (uint32_t i = 0; i < m; ++i) {
    const double* const arow = a + size_t{i} * k;
    for (uint32_t j = 0; j < n; j += 4) {
      __m256d sum = _mm256_loadu_pd(c + size_t{i} * n + j);
      const double* b = bk + j;
      for (uint32_t t = 0; t < k; ++t, b += n) sum = _mm256_fmadd_pd(_mm256_loadu_pd(b), _mm256_broadcast_sd(arow + t), sum);
      _mm256_storeu_pd(out + size_t{i} * n + j, sum);
    }
  }
}

// (AVX2: eight columns to an instruction; n is a multiple of 8)
__attribute__((target("avx2"))) void mfma_tile_int_avx2(uint32_t m, uint32_t n, uint32_t k, const int16_t* ia, const int16_t* bi,
                                                              int32_t* out) {
  for (uint32_t i = 0; i < m; i += 4) {
    const int16_t* const a0 = ia + size_t{i} * k;
    for (uint32_t j = 0; j < n; j += 8) {
      __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
      const int16_t* b = bi + size_t{j} * 2;
      for (uint32_t t = 0; t < k; t += 2, b += size_t{n} * 2) {
        const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
        int32_t p0, p1, p2, p3;
        std::memcpy(&p0, a0 + t, 4), std::memcpy(&p1, a0 + k + t, 4), std::memcpy(&p2, a0 + 2 * size_t{k} + t, 4), std::memcpy(&p3, a0 + 3 * size_t{k} + t, 4);
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_set1_epi32(p0), bv));
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_set1_epi32(p1), bv));
        s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(_mm256_set1_epi32(p2), bv));
        s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(_mm256_set1_epi32(p3), bv));
      }
      int32_t* const o = out + size_t{i} * n + j;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(o), s0);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + n), s1);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 2 * size_t{n}), s2);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 3 * size_t{n}), s3);
    }
  }
}

// matrix_decode_wide (gcn_exec.cpp), eight lanes to an instruction: the binary16 values by F16C (exact, a signalling NaN made quiet as
// half_bits_to_double makes it), the others a shift or a mask and the float to double conversion. The same doubles, to the same places.
// `rows` are the source operand's register rows (MatrixSrc::rows).
// (A pragma and not an attribute: on a function template's definition the target did not reach the explicit instantiations or the lambdas in them.)
#pragma GCC push_options
#pragma GCC target("avx2,f16c")
template <char T, bool Run>
void matrix_decode_wide_avx2(const uint32_t* const* rows, const uint32_t* dest, uint32_t step, uint32_t per_lane, double* dst, uint32_t lanes) {
  // Element e of lanes l to l + 7, as two vectors of four doubles.
  const auto convert = [&](uint32_t e, uint32_t l, __m256d* lo, __m256d* hi) __attribute__((target("avx2,f16c"), always_inline)) {
    constexpr bool packed = T == 'h' || T == 'b';   // two to a register
    const int shift = packed ? 16 * static_cast<int>(e % 2) : 0;
    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rows[packed ? e / 2 : e] + l));
    __m256 f;
    if constexpr (T == 'h') {
      half8_to_double(shift ? _mm256_srli_epi32(raw, 16) : _mm256_and_si256(raw, _mm256_set1_epi32(0xFFFF)), lo, hi);
      return;
    } else if constexpr (T == 'b') {
      f = _mm256_castsi256_ps(shift ? _mm256_and_si256(raw, _mm256_set1_epi32(static_cast<int>(0xFFFF0000u))) : _mm256_slli_epi32(raw, 16));
    } else if constexpr (T == 'F') {   // xf32: the mantissa cut to ten bits, except an infinity or a NaN
      const __m256i special = _mm256_cmpeq_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(0x7F800000)), _mm256_set1_epi32(0x7F800000));
      f = _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_and_si256(raw, _mm256_set1_epi32(~0x1FFF)), raw, special));
    } else {
      f = _mm256_castsi256_ps(raw);
    }
    *lo = _mm256_cvtps_pd(_mm256_castps256_ps128(f)), *hi = _mm256_cvtps_pd(_mm256_extractf128_ps(f, 1));
  };
  if constexpr (!Run) {
    if (step == 1 && per_lane % 4 == 0) {
      // A: a lane's run of K is consecutive in its row of the tile, so four elements of four lanes are a 4x4 transpose and a 32-byte
      // store a lane.
      for (uint32_t l = 0; l < lanes; l += 8)
        for (uint32_t e = 0; e < per_lane; e += 4) {
          __m256d lo[4], hi[4];
          for (uint32_t q = 0; q < 4; ++q) convert(e + q, l, &lo[q], &hi[q]);
          const auto transpose = [&](const __m256d* r, uint32_t first) __attribute__((target("avx2,f16c"), always_inline)) {
            const __m256d t0 = _mm256_unpacklo_pd(r[0], r[1]), t1 = _mm256_unpackhi_pd(r[0], r[1]);
            const __m256d t2 = _mm256_unpacklo_pd(r[2], r[3]), t3 = _mm256_unpackhi_pd(r[2], r[3]);
            _mm256_storeu_pd(dst + dest[first] + e, _mm256_permute2f128_pd(t0, t2, 0x20));
            _mm256_storeu_pd(dst + dest[first + 1] + e, _mm256_permute2f128_pd(t1, t3, 0x20));
            _mm256_storeu_pd(dst + dest[first + 2] + e, _mm256_permute2f128_pd(t0, t2, 0x31));
            _mm256_storeu_pd(dst + dest[first + 3] + e, _mm256_permute2f128_pd(t1, t3, 0x31));
          };
          transpose(lo, l);
          transpose(hi, l + 4);
        }
      return;
    }
  }
  for (uint32_t e = 0; e < per_lane; ++e) {
    const size_t off = size_t{e} * step;
    for (uint32_t l = 0; l < lanes; l += 8) {
      __m256d lo, hi;
      convert(e, l, &lo, &hi);
      if constexpr (Run) {
        _mm256_storeu_pd(dst + dest[l] + off, lo);
        _mm256_storeu_pd(dst + dest[l + 4] + off, hi);   // (four lanes' columns are consecutive, not always the next four's)
      } else {
        const __m128d d0 = _mm256_castpd256_pd128(lo), d1 = _mm256_extractf128_pd(lo, 1), d2 = _mm256_castpd256_pd128(hi), d3 = _mm256_extractf128_pd(hi, 1);
        _mm_storel_pd(dst + dest[l] + off, d0);
        _mm_storeh_pd(dst + dest[l + 1] + off, d0);
        _mm_storel_pd(dst + dest[l + 2] + off, d1);
        _mm_storeh_pd(dst + dest[l + 3] + off, d1);
        _mm_storel_pd(dst + dest[l + 4] + off, d2);
        _mm_storeh_pd(dst + dest[l + 5] + off, d2);
        _mm_storel_pd(dst + dest[l + 6] + off, d3);
        _mm_storeh_pd(dst + dest[l + 7] + off, d3);
      }
    }
  }
}
template void matrix_decode_wide_avx2<'h', false>(const uint32_t* const*, const uint32_t*, uint32_t, uint32_t, double*, uint32_t);
template void matrix_decode_wide_avx2<'h', true>(const uint32_t* const*, const uint32_t*, uint32_t, uint32_t, double*, uint32_t);
template void matrix_decode_wide_avx2<'b', false>(const uint32_t* const*, const uint32_t*, uint32_t, uint32_t, double*, uint32_t);
template void matrix_decode_wide_avx2<'b', true>(const uint32_t* const*, const uint32_t*, uint32_t, uint32_t, double*, uint32_t);
template void matrix_decode_wide_avx2<'f', false>(const uint32_t* const*, const uint32_t*, uint32_t, uint32_t, double*, uint32_t);
template void matrix_decode_wide_avx2<'f', true>(const uint32_t* const*, const uint32_t*, uint32_t, uint32_t, double*, uint32_t);
template void matrix_decode_wide_avx2<'F', false>(const uint32_t* const*, const uint32_t*, uint32_t, uint32_t, double*, uint32_t);
template void matrix_decode_wide_avx2<'F', true>(const uint32_t* const*, const uint32_t*, uint32_t, uint32_t, double*, uint32_t);

#pragma GCC pop_options

#endif

}  // namespace vgpu::amd
