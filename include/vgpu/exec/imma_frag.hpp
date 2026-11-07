// IMMA.16816 and IMMA.16832 with 8-bit operands (s8 or u8 each, a 32-bit wrapping sum, no .SAT), on the registers of the instruction, at the AVX-512 level
// (vpdpbusd, imma_frag_m16n8) and the AVX2 level (widened to 16-bit pairs and summed by vpmaddwd, imma_frag_m16n8_avx2).
// The SASS tier's h_imma used to take the fragments apart into int32 tiles (A 16 x K, B K x 8, C 16 x 8), run simd::Kernels::tile_i32 and put the answer
// together again. Each product is at most 2^16 and the sum is taken modulo 2^32 whatever the order, so the answer is the tile kernel's in every bit
// (tests/unit/test_imma_frag.cpp holds both forms to an element-by-element definition).
//
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define VGPU_IMMA_FRAG 1
#define VGPU_IMMA_T512 target("avx2,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")
#define VGPU_IMMA_T256 target("avx2")

namespace vgpu::exec {

// K is 16 or 32. `a`: K / 8 registers, `b`: K / 16, `c`: 4 (null: a zero accumulator), `out`: 4. Reads nothing past a register.
//
// The sums are made by vpdpbusd (AVX512-VNNI: four unsigned-byte by signed-byte products added to a 32-bit lane, wrapping), which the Avx512 level has. It takes
// one operand unsigned and the other signed, so the four sign combinations are made to fit, exactly (the sums are modulo 2^32, so no step can lose a bit):
//   A unsigned, B signed:   dpbusd(A, B).
//   A signed, B unsigned:   dpbusd(B, A), the product being commutative.
//   both signed:            A + 128 is unsigned (the byte with its top bit flipped), so dpbusd(A ^ 0x80, B) = sum A.B + 128 sum B: subtract 128 x (the column sum of B).
//   both unsigned:          A - 128 is signed (the same flip, read as signed), so dpbusd(B, A ^ 0x80) = sum A.B - 128 sum B: add 128 x (the column sum of B).
// The column sum of B is a dot product of the same B bytes with ones, one extra vpdpbusd for each four rows of B.
// Layout (vgpu/exec/tile_mma.hpp, PTX ISA 9.7.14): A register r holds, in the four lanes 4g .. 4g + 3, the 16 bytes of row g + 8 (r % 2) at byte
// 16 (r / 2) of the row, so word t of that group is the dword of four k; B register r holds those of column g at byte 16 r of the column; C and D register 2h + e,
// lane 4g + t, is row g + 8h, column 2t + e. A sixteen-lane unit is two rows (2p, 2p + 1) by 8 columns, lane 2j + r (as the float kernel of tc_ampere.hpp has it):
// the A side of a dword of k is a broadcast of the 64-bit element (row 2p's dword, row 2p + 1's dword), the B side the dwords of the 8 columns, each doubled.
template <bool SA, bool SB, unsigned K>
__attribute__((VGPU_IMMA_T512, aligned(64), noinline)) inline void imma_frag_k(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out) {
  static_assert(K == 16 || K == 32);
  alignas(64) int64_t apk[8 * (K / 4)];
  alignas(64) int32_t bq[(K / 4) * 16];
  constexpr unsigned nq = K / 4, nch = K / 16;   // dwords of k, 16-byte runs
  constexpr bool kFlip = SA == SB;   // equal signs: the A side is flipped to the other signedness
  {
    const __m256i flip2 = _mm256_set1_epi8(static_cast<char>(0x80));
    const __m256i in8 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);   // [row 2p: dwords 0..3 | row 2p + 1: dwords 0..3] -> [d0 of both rows, d1 of both, ...]
#pragma GCC unroll 16
    for (unsigned p = 0; p < 8; ++p)
#pragma GCC unroll 2
      for (unsigned ch = 0; ch < nch; ++ch) {   // rows 2p, 2p + 1: 32 contiguous bytes at the groups g, g + 1 (g = 2p mod 8) of register 2 ch + (p >= 4)
        const uint32_t* const src = a + size_t{2 * ch + (p >> 2)} * 32 + 4 * ((2 * p) & 7);
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
        if (kFlip) v = _mm256_xor_si256(v, flip2);
        _mm256_store_si256(reinterpret_cast<__m256i*>(&apk[p * nq + ch * 4]), _mm256_permutevar8x32_epi32(v, in8));
      }
  }
  // B: the 128 bytes of a chunk are [column g: dwords 0..3] for g = 0 .. 7; the vector of dword q is lane 2j, 2j + 1 = column j's dword q, a two-source permute.
  const __m512i b_idx[4] = {
      _mm512_setr_epi32(0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28), _mm512_setr_epi32(1, 1, 5, 5, 9, 9, 13, 13, 17, 17, 21, 21, 25, 25, 29, 29),
      _mm512_setr_epi32(2, 2, 6, 6, 10, 10, 14, 14, 18, 18, 22, 22, 26, 26, 30, 30), _mm512_setr_epi32(3, 3, 7, 7, 11, 11, 15, 15, 19, 19, 23, 23, 27, 27, 31, 31)};
#pragma GCC unroll 2
  for (unsigned ch = 0; ch < nch; ++ch) {
    const __m512i lo = _mm512_loadu_si512(b + size_t{ch} * 32), hi = _mm512_loadu_si512(b + size_t{ch} * 32 + 16);
    for (unsigned q = 0; q < 4; ++q) _mm512_store_si512(&bq[(ch * 4 + q) * 16], _mm512_permutex2var_epi32(lo, b_idx[q], hi));
  }
  const __m512i idx_in = _mm512_setr_epi32(0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7, 19, 23),   // lane 2j + r <- e * 16 + r * 4 + t (j = 2t + e)
                idx_out = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);   // [even columns of the two rows | odd columns]
  // Both signed or both unsigned: the correction, 128 x the column sums of B (a lane of the B side is a column), added (unsigned) or subtracted (signed).
  __m512i corr = _mm512_setzero_si512();
  if (kFlip) {
    const __m512i ones = _mm512_set1_epi8(1);
    __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();   // two chains
#pragma GCC unroll 8
    for (unsigned q = 0; q < nq; ++q) {
      const __m512i bv = _mm512_load_si512(&bq[q * 16]);
      __m512i& cc = (q & 1) ? c1 : c0;
      cc = SB ? _mm512_dpbusd_epi32(cc, ones, bv) : _mm512_dpbusd_epi32(cc, bv, ones);
    }
    corr = _mm512_slli_epi32(_mm512_add_epi32(c0, c1), 7);
  }
  __m512i acc[8];
#pragma GCC unroll 8
  for (unsigned p = 0; p < 8; ++p) {
    __m512i x = _mm512_setzero_si512();
    if (c) {
      const size_t at = size_t{2 * (p >> 2)} * 32 + 4 * ((2 * p) & 7);
      const __m512i lo = _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + at))),
                    hi = _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + at + 32)));
      x = _mm512_permutex2var_epi32(lo, idx_in, hi);
    }
    acc[p] = x;
  }
#pragma GCC unroll 8
  for (unsigned q = 0; q < nq; ++q) {
    const __m512i bv = _mm512_load_si512(&bq[q * 16]);
#pragma GCC unroll 8
    for (unsigned p = 0; p < 8; ++p) {
      const __m512i av = _mm512_set1_epi64(apk[p * nq + q]);
      // The unsigned operand first: B when A is signed and B not, the flipped A when the signs are equal, A when only B is signed.
      acc[p] = SB ? _mm512_dpbusd_epi32(acc[p], av, bv) : _mm512_dpbusd_epi32(acc[p], bv, av);
    }
  }
#pragma GCC unroll 8
  for (unsigned p = 0; p < 8; ++p) {
    const size_t at = size_t{2 * (p >> 2)} * 32 + 4 * ((2 * p) & 7);
    __m512i r = acc[p];
    if (kFlip) r = SA ? _mm512_sub_epi32(r, corr) : _mm512_add_epi32(r, corr);
    const __m512i v = _mm512_permutexvar_epi32(idx_out, r);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + at), _mm512_castsi512_si256(v));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + at + 32), _mm512_extracti64x4_epi64(v, 1));
  }
}

template <bool SA, bool SB>
inline void imma_frag_m16n8(unsigned K, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out) {
  if (K == 16) imma_frag_k<SA, SB, 16>(a, b, c, out);
  else imma_frag_k<SA, SB, 32>(a, b, c, out);
}

// The same at the AVX2 level (256-bit registers): a unit is one row by the 8 columns (lane j is column j), the A side of a pair of k a broadcast of that row's 32-bit
// pair, the B side the pairs of the 8 columns (not doubled). Row g + 8h is the 16 bytes at word 4g of register 2 ch + h; the result row, even columns and then odd
// columns, is words 4g .. 4g + 3 of registers 2h and 2h + 1 (column 2t + e of row g + 8h is register 2h + e, word 4g + t). Every sum is a wrapping 32-bit sum, so
// the answer is the same in every order, and the 512-bit form's.
template <bool SA, bool SB, unsigned K>
__attribute__((VGPU_IMMA_T256, aligned(64), noinline)) inline void imma_frag_avx2_k(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out) {
  static_assert(K == 16 || K == 32);
  alignas(32) int32_t apk[16 * (K / 2)];   // row i, pair q of k: the two elements as 16-bit halves
  alignas(32) int32_t bq[(K / 2) * 8];     // pair q, column j
  constexpr unsigned nq = K / 2, nch = K / 16;
#pragma GCC unroll 16
  for (unsigned row = 0; row < 16; ++row)
#pragma GCC unroll 2
    for (unsigned ch = 0; ch < nch; ++ch) {
      const uint32_t* const src = a + size_t{2 * ch + (row >> 3)} * 32 + 4 * (row & 7);
      const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
      _mm256_store_si256(reinterpret_cast<__m256i*>(&apk[row * nq + ch * 8]), SA ? _mm256_cvtepi8_epi16(v) : _mm256_cvtepu8_epi16(v));
    }
#pragma GCC unroll 2
  for (unsigned ch = 0; ch < nch; ++ch) {   // column g: 16 bytes = 8 pairs of k; transposed to a vector of the 8 columns for each pair
    const uint32_t* const src = b + size_t{ch} * 32;
    const auto wide = [&](unsigned g) __attribute__((VGPU_IMMA_T256)) {
      const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4 * g));
      return SB ? _mm256_cvtepi8_epi16(v) : _mm256_cvtepu8_epi16(v);
    };
    const __m256i r0 = wide(0), r1 = wide(1), r2 = wide(2), r3 = wide(3), r4 = wide(4), r5 = wide(5), r6 = wide(6), r7 = wide(7);
    const __m256i t0 = _mm256_unpacklo_epi32(r0, r1), t1 = _mm256_unpackhi_epi32(r0, r1), t2 = _mm256_unpacklo_epi32(r2, r3), t3 = _mm256_unpackhi_epi32(r2, r3),
                  t4 = _mm256_unpacklo_epi32(r4, r5), t5 = _mm256_unpackhi_epi32(r4, r5), t6 = _mm256_unpacklo_epi32(r6, r7), t7 = _mm256_unpackhi_epi32(r6, r7);
    const __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2), u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3),
                  u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6), u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
    const __m256i v[8] = {_mm256_permute2x128_si256(u0, u4, 0x20), _mm256_permute2x128_si256(u1, u5, 0x20), _mm256_permute2x128_si256(u2, u6, 0x20),
                          _mm256_permute2x128_si256(u3, u7, 0x20), _mm256_permute2x128_si256(u0, u4, 0x31), _mm256_permute2x128_si256(u1, u5, 0x31),
                          _mm256_permute2x128_si256(u2, u6, 0x31), _mm256_permute2x128_si256(u3, u7, 0x31)};   // v[q] = pair q of the 8 columns
    for (unsigned q = 0; q < 8; ++q) _mm256_store_si256(reinterpret_cast<__m256i*>(&bq[(ch * 8 + q) * 8]), v[q]);
  }
  const __m256i to_cols = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7),     // [even | odd] -> column order
                to_halves = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);   // column order -> [even | odd]
#pragma GCC unroll 8
  for (unsigned g = 0; g < 8; ++g)
#pragma GCC unroll 2
    for (unsigned h = 0; h < 2; ++h) {
      const unsigned row = g + 8 * h;
      __m256i acc = _mm256_setzero_si256();
      if (c) {
        const __m256i eo = _mm256_set_m128i(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c + size_t{2 * h + 1} * 32 + 4 * g)),
                                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(c + size_t{2 * h} * 32 + 4 * g)));
        acc = _mm256_permutevar8x32_epi32(eo, to_cols);
      }
      __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
#pragma GCC unroll 8
      for (unsigned q = 0; q < nq; q += 2) {
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_set1_epi32(apk[row * nq + q]), _mm256_load_si256(reinterpret_cast<const __m256i*>(&bq[q * 8]))));
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_set1_epi32(apk[row * nq + q + 1]), _mm256_load_si256(reinterpret_cast<const __m256i*>(&bq[(q + 1) * 8]))));
      }
      const __m256i r = _mm256_permutevar8x32_epi32(_mm256_add_epi32(acc, _mm256_add_epi32(s0, s1)), to_halves);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + size_t{2 * h} * 32 + 4 * g), _mm256_castsi256_si128(r));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + size_t{2 * h + 1} * 32 + 4 * g), _mm256_extracti128_si256(r, 1));
    }
}

template <bool SA, bool SB>
inline void imma_frag_m16n8_avx2(unsigned K, const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out) {
  if (K == 16) imma_frag_avx2_k<SA, SB, 16>(a, b, c, out);
  else imma_frag_avx2_k<SA, SB, 32>(a, b, c, out);
}

}  // namespace vgpu::exec
#endif
