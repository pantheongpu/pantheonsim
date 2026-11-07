// The data movement of LDSM (ldmatrix of 16-bit elements) once every row address is known to lie inside the block's shared array: matrix i's row q is the
// 16 bytes at the address of lane 8i + q, and lane t of the destination register i gets that matrix's row t / 4, 32-bit word t % 4 -- or, with `trans`
// (.MT88), the two halves of column t / 4 taken from rows 2 (t % 4) and 2 (t % 4) + 1. In words: a plain matrix is its eight rows laid one after the
// other (a 16-byte copy a row); a transposed one is the 8 x 8 matrix of halves transposed, and column c of it is then the 16 bytes
// out[4c .. 4c + 3]. (The element-at-a-time form this replaces is held in tests/unit/test_ldsm_move.cpp.)
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace vgpu::exec {

// `rows`: the 32 lane addresses (only the low 24 bits, the offset in the array, are used); `out`: nmat registers of 32 words, one after the other.
// Reads 16 bytes at each of the 8 * nmat row offsets, so every offset must be at most size - 16 (the caller's window check).
inline void ldsm_move(const uint8_t* shared, const uint32_t* rows, unsigned nmat, bool trans, uint32_t* out) {
  for (unsigned i = 0; i < nmat; ++i) {
    const uint32_t* const rw = rows + 8 * i;
    uint32_t* const o = out + 32 * i;
#if defined(__SSE2__)
    __m128i r[8];
    for (unsigned q = 0; q < 8; ++q) r[q] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(shared + (rw[q] & 0xffffffu)));
    if (!trans) {
      for (unsigned q = 0; q < 8; ++q) _mm_storeu_si128(reinterpret_cast<__m128i*>(o) + q, r[q]);
    } else {
      const __m128i a0 = _mm_unpacklo_epi16(r[0], r[1]), a1 = _mm_unpackhi_epi16(r[0], r[1]), a2 = _mm_unpacklo_epi16(r[2], r[3]),
                    a3 = _mm_unpackhi_epi16(r[2], r[3]), a4 = _mm_unpacklo_epi16(r[4], r[5]), a5 = _mm_unpackhi_epi16(r[4], r[5]),
                    a6 = _mm_unpacklo_epi16(r[6], r[7]), a7 = _mm_unpackhi_epi16(r[6], r[7]);
      const __m128i b0 = _mm_unpacklo_epi32(a0, a2), b1 = _mm_unpackhi_epi32(a0, a2), b2 = _mm_unpacklo_epi32(a1, a3), b3 = _mm_unpackhi_epi32(a1, a3),
                    b4 = _mm_unpacklo_epi32(a4, a6), b5 = _mm_unpackhi_epi32(a4, a6), b6 = _mm_unpacklo_epi32(a5, a7), b7 = _mm_unpackhi_epi32(a5, a7);
      __m128i* const ov = reinterpret_cast<__m128i*>(o);
      _mm_storeu_si128(ov + 0, _mm_unpacklo_epi64(b0, b4));
      _mm_storeu_si128(ov + 1, _mm_unpackhi_epi64(b0, b4));
      _mm_storeu_si128(ov + 2, _mm_unpacklo_epi64(b1, b5));
      _mm_storeu_si128(ov + 3, _mm_unpackhi_epi64(b1, b5));
      _mm_storeu_si128(ov + 4, _mm_unpacklo_epi64(b2, b6));
      _mm_storeu_si128(ov + 5, _mm_unpackhi_epi64(b2, b6));
      _mm_storeu_si128(ov + 6, _mm_unpacklo_epi64(b3, b7));
      _mm_storeu_si128(ov + 7, _mm_unpackhi_epi64(b3, b7));
    }
#else
    for (unsigned t = 0; t < 32; ++t) {
      if (!trans) {
        std::memcpy(&o[t], shared + (rw[t / 4] & 0xffffffu) + 4 * (t % 4), 4);
      } else {
        uint16_t lo, hi;
        std::memcpy(&lo, shared + (rw[2 * (t % 4)] & 0xffffffu) + 2 * (t / 4), 2);
        std::memcpy(&hi, shared + (rw[2 * (t % 4) + 1] & 0xffffffu) + 2 * (t / 4), 2);
        o[t] = lo | (static_cast<uint32_t>(hi) << 16);
      }
    }
#endif
  }
}

}  // namespace vgpu::exec
