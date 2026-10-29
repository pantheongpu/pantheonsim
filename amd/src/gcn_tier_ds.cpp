// See gcn_tier_ds.hpp.
#include "gcn_tier_ds.hpp"

#include <immintrin.h>

#include <cstring>

namespace vgpu::amd::gcn::tier_ds {

// The lanes' words of a wide LDS access (W = 2 or 4 words a lane, 4 lanes at a time) between the LDS's lane-major layout and the register rows:
// `quads` loads W words from each of four byte addresses into rows 0..W-1 of four lanes (`rows[k][l..l+3]` = word k of lanes l..l+3), `unquads` the
// opposite. Pure moves of bits.
[[gnu::always_inline]] inline void transpose4(__m128i& a, __m128i& b, __m128i& c, __m128i& d) {
  const __m128i t0 = _mm_unpacklo_epi32(a, b), t1 = _mm_unpacklo_epi32(c, d), t2 = _mm_unpackhi_epi32(a, b), t3 = _mm_unpackhi_epi32(c, d);
  a = _mm_unpacklo_epi64(t0, t1), b = _mm_unpackhi_epi64(t0, t1), c = _mm_unpacklo_epi64(t2, t3), d = _mm_unpackhi_epi64(t2, t3);
}
template <unsigned W>
[[gnu::always_inline]] inline void lds_to_rows(uint32_t* const* rows, const uint8_t* const* from, unsigned lanes_at) {
  if constexpr (W == 4) {
    __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(from[0])), q1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(from[1]));
    __m128i q2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(from[2])), q3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(from[3]));
    transpose4(q0, q1, q2, q3);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(rows[0] + lanes_at), q0), _mm_storeu_si128(reinterpret_cast<__m128i*>(rows[1] + lanes_at), q1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(rows[2] + lanes_at), q2), _mm_storeu_si128(reinterpret_cast<__m128i*>(rows[3] + lanes_at), q3);
  } else {
    const __m128i p0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(from[0])), p1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(from[1]));
    const __m128i p2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(from[2])), p3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(from[3]));
    const __m128i x = _mm_unpacklo_epi64(p0, p1), y = _mm_unpacklo_epi64(p2, p3);   // [a0 b0 a1 b1], [a2 b2 a3 b3]
    _mm_storeu_si128(reinterpret_cast<__m128i*>(rows[0] + lanes_at), _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(x), _mm_castsi128_ps(y), _MM_SHUFFLE(2, 0, 2, 0))));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(rows[1] + lanes_at), _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(x), _mm_castsi128_ps(y), _MM_SHUFFLE(3, 1, 3, 1))));
  }
}
// The same the other way, stored to the four addresses in lane order (so that lanes whose words overlap end as they do one at a time).
template <unsigned W>
[[gnu::always_inline]] inline void rows_to_lds(uint32_t* const* rows, uint8_t* const* to, unsigned lanes_at) {
  if constexpr (W == 4) {
    __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rows[0] + lanes_at)), r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rows[1] + lanes_at));
    __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rows[2] + lanes_at)), r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rows[3] + lanes_at));
    transpose4(r0, r1, r2, r3);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(to[0]), r0), _mm_storeu_si128(reinterpret_cast<__m128i*>(to[1]), r1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(to[2]), r2), _mm_storeu_si128(reinterpret_cast<__m128i*>(to[3]), r3);
  } else {
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rows[0] + lanes_at)), b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rows[1] + lanes_at));
    const __m128i x = _mm_unpacklo_epi32(a, b), y = _mm_unpackhi_epi32(a, b);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(to[0]), x), _mm_storel_epi64(reinterpret_cast<__m128i*>(to[1]), _mm_unpackhi_epi64(x, x));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(to[2]), y), _mm_storel_epi64(reinterpret_cast<__m128i*>(to[3]), _mm_unpackhi_epi64(y, y));
  }
}

// The wide LDS accesses of the lanes of `ex`, one by one, for a width known here (so the words of a lane are straight-line code on rows held in
// registers, not a loop re-reading the operand's register number): lane l's W words from the LDS address addr[l] + x0 into the rows, or back.
template <unsigned W>
static void ds_read_lanes_w(uint32_t* const* rows, const uint8_t* lds, const uint32_t* addr, uint32_t x0, uint64_t ex) {
  uint32_t* r[W];
  for (unsigned k = 0; k < W; ++k) r[k] = rows[k];
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
    const uint8_t* p = lds + static_cast<uint32_t>(addr[l] + x0);
    for (unsigned k = 0; k < W; ++k) std::memcpy(&r[k][l], p + 4 * k, 4);
  }
}
template <unsigned W>
static void ds_write_lanes_w(uint32_t* const* rows, uint8_t* lds, const uint32_t* addr, uint32_t x0, uint64_t ex) {
  uint32_t* r[W];
  for (unsigned k = 0; k < W; ++k) r[k] = rows[k];
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));
    uint8_t* p = lds + static_cast<uint32_t>(addr[l] + x0);
    for (unsigned k = 0; k < W; ++k) std::memcpy(p + 4 * k, &r[k][l], 4);
  }
}

// The wide accesses of a run of cnt consecutive lanes (rows already offset to the first lane), and of every lane at any addresses, as functions
// of their own: out of the handler's body, which stays the size it had.
void ds_run_read(uint32_t* const* rows, const uint8_t* at, unsigned cnt, unsigned w) {
  unsigned l = 0;
  for (; l + 4 <= cnt; l += 4) {
    const uint8_t* from[4] = {at + 4u * w * l, at + 4u * w * (l + 1), at + 4u * w * (l + 2), at + 4u * w * (l + 3)};
    if (w == 4) lds_to_rows<4>(rows, from, l);
    else lds_to_rows<2>(rows, from, l);
  }
  for (; l < cnt; ++l)
    for (unsigned k = 0; k < w; ++k) std::memcpy(&rows[k][l], at + 4u * (w * l + k), 4);
}
void ds_run_write(uint32_t* const* rows, uint8_t* at, unsigned cnt, unsigned w) {
  unsigned l = 0;
  for (; l + 4 <= cnt; l += 4) {
    uint8_t* to[4] = {at + 4u * w * l, at + 4u * w * (l + 1), at + 4u * w * (l + 2), at + 4u * w * (l + 3)};
    if (w == 4) rows_to_lds<4>(rows, to, l);
    else rows_to_lds<2>(rows, to, l);
  }
  for (; l < cnt; ++l)
    for (unsigned k = 0; k < w; ++k) std::memcpy(at + 4u * (w * l + k), &rows[k][l], 4);
}
void ds_all_read(uint32_t* const* rows, const uint8_t* lds, const uint32_t* addr, uint32_t x0, unsigned w, unsigned n) {
  for (unsigned l = 0; l < n; l += 4) {
    const uint8_t* from[4] = {lds + static_cast<uint32_t>(addr[l] + x0), lds + static_cast<uint32_t>(addr[l + 1] + x0), lds + static_cast<uint32_t>(addr[l + 2] + x0),
                              lds + static_cast<uint32_t>(addr[l + 3] + x0)};
    if (w == 4) lds_to_rows<4>(rows, from, l);
    else lds_to_rows<2>(rows, from, l);
  }
}
void ds_all_write(uint32_t* const* rows, uint8_t* lds, const uint32_t* addr, uint32_t x0, unsigned w, unsigned n) {
  for (unsigned l = 0; l < n; l += 4) {
    uint8_t* to[4] = {lds + static_cast<uint32_t>(addr[l] + x0), lds + static_cast<uint32_t>(addr[l + 1] + x0), lds + static_cast<uint32_t>(addr[l + 2] + x0),
                      lds + static_cast<uint32_t>(addr[l + 3] + x0)};
    if (w == 4) rows_to_lds<4>(rows, to, l);
    else rows_to_lds<2>(rows, to, l);
  }
}

void ds_lanes_read(uint32_t* const* rows, const uint8_t* lds, const uint32_t* addr, uint32_t x0, uint64_t ex, unsigned w) {
  if (w == 2) ds_read_lanes_w<2>(rows, lds, addr, x0, ex);
  else if (w == 3) ds_read_lanes_w<3>(rows, lds, addr, x0, ex);
  else ds_read_lanes_w<4>(rows, lds, addr, x0, ex);
}
void ds_lanes_write(uint32_t* const* rows, uint8_t* lds, const uint32_t* addr, uint32_t x0, uint64_t ex, unsigned w) {
  if (w == 2) ds_write_lanes_w<2>(rows, lds, addr, x0, ex);
  else if (w == 3) ds_write_lanes_w<3>(rows, lds, addr, x0, ex);
  else ds_write_lanes_w<4>(rows, lds, addr, x0, ex);
}

}  // namespace vgpu::amd::gcn::tier_ds
