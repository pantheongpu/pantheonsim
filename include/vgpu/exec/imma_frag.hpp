// IMMA.16816 and IMMA.16832 with 8-bit operands (s8 or u8 each, a 32-bit wrapping sum, no .SAT), on the registers of the instruction at the AVX-512 level.
// The SASS tier's h_imma used to take the fragments apart into int32 tiles (A 16 x K, B K x 8, C 16 x 8), run simd::Kernels::tile_i32 and put the answer
// together again; here the 16-byte runs the registers hold are widened to 16-bit pairs straight from the registers and summed by vpmaddwd. Each product is
// at most 2^16 and the sum is taken modulo 2^32 whatever the order, so the answer is the tile kernel's in every bit (tests/unit/test_imma_frag.cpp
// holds it to an element-by-element definition).
//
// Register layout (vgpu/exec/tile_mma.hpp, PTX ISA 9.7.14): A register r holds, in the four lanes 4g .. 4g + 3, the 16 bytes of row g + 8 (r % 2) at byte
// 16 (r / 2) of the row; B register r holds those of column g at byte 16 r of the column; C and D register 2h + e, lane 4g + t, is row g + 8h, column
// 2t + e. A sixteen-lane unit is two rows (2p, 2p + 1) by 8 columns, lane 2j + r (as the float kernel of tc_ampere.hpp has it): the A side of a pair of
// k (a 32-bit pair of 16-bit elements) is a broadcast of the 64-bit element (row 2p's pair, row 2p + 1's pair), the B side the pairs of the 8 columns, each
// doubled.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define VGPU_IMMA_FRAG 1
#define VGPU_IMMA_T512 target("avx2,avx512f,avx512bw,avx512dq,avx512vl")

namespace vgpu::exec {

template <bool S>
__attribute__((VGPU_IMMA_T512, always_inline)) inline __m512i imma_widen_a(__m256i v) {
  return S ? _mm512_cvtepi8_epi16(v) : _mm512_cvtepu8_epi16(v);
}
template <bool S>
__attribute__((VGPU_IMMA_T512, always_inline)) inline __m256i imma_widen_b(__m128i v) {
  return S ? _mm256_cvtepi8_epi16(v) : _mm256_cvtepu8_epi16(v);
}

// K is 16 or 32. `a`: K / 8 registers, `b`: K / 16, `c`: 4 (null: a zero accumulator), `out`: 4. Reads nothing past a register.
template <bool SA, bool SB>
__attribute__((VGPU_IMMA_T512, aligned(64), noinline)) inline void imma_frag_m16n8(unsigned K, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out) {
  constexpr unsigned kMaxQ = 16;   // pairs of k: K / 2
  alignas(64) int64_t apk[8 * kMaxQ];
  alignas(64) int32_t bq[kMaxQ * 16];
  const unsigned nq = K / 2, nch = K / 16;
  {
    const __m512i in8 = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    for (unsigned p = 0; p < 8; ++p)
      for (unsigned ch = 0; ch < nch; ++ch) {   // rows 2p, 2p + 1: 32 contiguous bytes at the groups g, g + 1 (g = 2p mod 8) of register 2 ch + (p >= 4)
        const uint32_t* const src = a + size_t{2 * ch + (p >> 2)} * 32 + 4 * ((2 * p) & 7);
        const __m512i w = imma_widen_a<SA>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src)));   // [row 2p: 8 pairs][row 2p + 1: 8 pairs]
        _mm512_store_si512(&apk[p * nq + ch * 8], _mm512_permutexvar_epi32(in8, w));
      }
  }
  {
    const __m512i dup = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    for (unsigned ch = 0; ch < nch; ++ch) {   // column g: 16 bytes = 8 pairs of k; transposed to a vector of the 8 columns for each pair
      const uint32_t* const src = b + size_t{ch} * 32;
      __m256i r0 = imma_widen_b<SB>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src))), r1 = imma_widen_b<SB>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4))),
              r2 = imma_widen_b<SB>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 8))), r3 = imma_widen_b<SB>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 12))),
              r4 = imma_widen_b<SB>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 16))), r5 = imma_widen_b<SB>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 20))),
              r6 = imma_widen_b<SB>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 24))), r7 = imma_widen_b<SB>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 28)));
      // 8 x 8 transpose of 32-bit elements (rows = columns of B, elements = pairs of k)
      const __m256i t0 = _mm256_unpacklo_epi32(r0, r1), t1 = _mm256_unpackhi_epi32(r0, r1), t2 = _mm256_unpacklo_epi32(r2, r3), t3 = _mm256_unpackhi_epi32(r2, r3),
                    t4 = _mm256_unpacklo_epi32(r4, r5), t5 = _mm256_unpackhi_epi32(r4, r5), t6 = _mm256_unpacklo_epi32(r6, r7), t7 = _mm256_unpackhi_epi32(r6, r7);
      const __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2), u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3),
                    u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6), u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
      const __m256i v[8] = {_mm256_permute2x128_si256(u0, u4, 0x20), _mm256_permute2x128_si256(u1, u5, 0x20), _mm256_permute2x128_si256(u2, u6, 0x20),
                            _mm256_permute2x128_si256(u3, u7, 0x20), _mm256_permute2x128_si256(u0, u4, 0x31), _mm256_permute2x128_si256(u1, u5, 0x31),
                            _mm256_permute2x128_si256(u2, u6, 0x31), _mm256_permute2x128_si256(u3, u7, 0x31)};   // v[q] = pair q of the 8 columns
      for (unsigned q = 0; q < 8; ++q) _mm512_store_si512(&bq[(ch * 8 + q) * 16], _mm512_permutexvar_epi32(dup, _mm512_castsi256_si512(v[q])));
    }
  }
  const __m512i idx_in = _mm512_setr_epi32(0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7, 19, 23),   // lane 2j + r <- e * 16 + r * 4 + t (j = 2t + e)
                idx_out = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);   // [even columns of the two rows | odd columns]
  for (unsigned p = 0; p < 8; ++p) {
    const size_t at = size_t{2 * (p >> 2)} * 32 + 4 * ((2 * p) & 7);
    __m512i acc = _mm512_setzero_si512();
    if (c) {
      const __m512i x = _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + at))),
                    y = _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + at + 32)));
      acc = _mm512_permutex2var_epi32(x, idx_in, y);
    }
    __m512i s0 = _mm512_setzero_si512(), s1 = _mm512_setzero_si512();
    for (unsigned q = 0; q < nq; q += 2) {
      s0 = _mm512_add_epi32(s0, _mm512_madd_epi16(_mm512_set1_epi64(apk[p * nq + q]), _mm512_load_si512(&bq[q * 16])));
      s1 = _mm512_add_epi32(s1, _mm512_madd_epi16(_mm512_set1_epi64(apk[p * nq + q + 1]), _mm512_load_si512(&bq[(q + 1) * 16])));
    }
    const __m512i v = _mm512_permutexvar_epi32(idx_out, _mm512_add_epi32(acc, _mm512_add_epi32(s0, s1)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + at), _mm512_castsi512_si256(v));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + at + 32), _mm512_extracti64x4_epi64(v, 1));
  }
}

}  // namespace vgpu::exec
#endif
