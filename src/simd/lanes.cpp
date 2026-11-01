asm(".pushsection .text,\"ax\",@progbits\n.p2align 12\n.popsection");
// (The line above puts this unit's code on a 4096-byte boundary: where it starts modulo a page no longer depends on the units linked ahead of it. docs/performance.md, Round sp14, combined.)
// Which lanes the SASS scheduler runs next (include/vgpu/exec/sass_lanes.hpp, pick_group): the lowest address among the runnable lanes' and the lanes there,
// in one pass of 256-bit compares where the host has AVX2 and the plain functions of the header where it has not.
#include "vgpu/exec/sass_lanes.hpp"

#include <bit>

#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#include <immintrin.h>
#define VGPU_LANES_AVX2 1
#endif

namespace vgpu::exec::sasslanes {

namespace {

Mask pick_group_plain(const uint64_t* pc, Mask run, uint64_t& lo) {
  lo = lowest_pc(pc, run, pc[std::countr_zero(run)]);
  return lanes_at(pc, lo) & run;
}

#ifdef VGPU_LANES_AVX2
// The addresses are compared as unsigned numbers: the sign bit is flipped, and the signed compare of the AVX2 then orders them so. A lane that
// does not run counts as the largest address (the biggest signed number after the flip), and is left out of the group at the end.
[[gnu::target("avx2")]] Mask pick_group_avx2(const uint64_t* pc, Mask run, uint64_t& lo) {
  const __m256i bit = _mm256_set_epi64x(8, 4, 2, 1);
  const __m256i flip = _mm256_set1_epi64x(INT64_MIN);
  const __m256i big = _mm256_set1_epi64x(INT64_MAX);
  __m256i v[8];
  __m256i best = big;
  for (unsigned i = 0; i < 8; ++i) {
    // (Loaded as two halves: the addresses were just stored 16 bytes or less at a time, and a load wider than the store it follows cannot be
    // served from it; the halves can.)
    const __m256i raw = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pc + 4 * i))),
                                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(pc + 4 * i + 2)), 1);
    const __m256i x = _mm256_xor_si256(raw, flip);
    const __m256i mine = _mm256_set1_epi64x(static_cast<long long>((run >> (4 * i)) & 15u));
    const __m256i in = _mm256_cmpeq_epi64(_mm256_and_si256(mine, bit), bit);
    v[i] = _mm256_blendv_epi8(big, x, in);
    best = _mm256_blendv_epi8(best, v[i], _mm256_cmpgt_epi64(best, v[i]));
  }
  alignas(32) int64_t lanes[4];
  _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), best);
  const int64_t low = std::min(std::min(lanes[0], lanes[1]), std::min(lanes[2], lanes[3]));
  const __m256i want = _mm256_set1_epi64x(low);
  Mask group = 0;
  for (unsigned i = 0; i < 8; ++i)
    group |= static_cast<Mask>(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v[i], want)))) << (4 * i);
  lo = static_cast<uint64_t>(low) ^ (uint64_t{1} << 63);
  return group & run;
}

// The four lanes of a nibble of a lane mask, as a 64-bit-lane select mask.
struct NibbleMasks {
  alignas(32) int64_t v[16][4];
  constexpr NibbleMasks() : v{} {
    for (unsigned n = 0; n < 16; ++n)
      for (unsigned j = 0; j < 4; ++j) v[n][j] = ((n >> j) & 1) ? -1 : 0;
  }
};
constexpr NibbleMasks kNibble;

[[gnu::target("avx2")]] void set_pcs_avx2(uint64_t* pc, Mask group, uint64_t v) {
  const __m256i vv = _mm256_set1_epi64x(static_cast<long long>(v));
  for (unsigned i = 0; i < 8; ++i)
  {
    __m256i* const at = reinterpret_cast<__m256i*>(pc + 4 * i);
    _mm256_storeu_si256(at, _mm256_blendv_epi8(_mm256_loadu_si256(at), vv, _mm256_load_si256(reinterpret_cast<const __m256i*>(kNibble.v[(group >> (4 * i)) & 15]))));
  }
}

[[gnu::target("avx2")]] void select_pcs_avx2(uint64_t* pc, Mask group, Mask take, uint64_t target, uint64_t next) {
  const __m256i t = _mm256_set1_epi64x(static_cast<long long>(target)), n = _mm256_set1_epi64x(static_cast<long long>(next));
  for (unsigned i = 0; i < 8; ++i) {
    const __m256i mt = _mm256_load_si256(reinterpret_cast<const __m256i*>(kNibble.v[(take >> (4 * i)) & 15]));
    const __m256i mg = _mm256_load_si256(reinterpret_cast<const __m256i*>(kNibble.v[(group >> (4 * i)) & 15]));
    __m256i* const at = reinterpret_cast<__m256i*>(pc + 4 * i);
    _mm256_storeu_si256(at, _mm256_blendv_epi8(_mm256_loadu_si256(at), _mm256_blendv_epi8(n, t, mt), mg));
  }
}
#endif

}  // namespace

void set_pcs_partial(uint64_t* pc, Mask group, uint64_t v) {
#ifdef VGPU_LANES_AVX2
  static const bool avx2 = simd::level() == simd::Level::Avx2;
  if (avx2) return set_pcs_avx2(pc, group, v);
#endif
  for (Mask m = group; m; m &= m - 1) pc[std::countr_zero(m)] = v;
}

void select_pcs(uint64_t* pc, Mask group, Mask take, uint64_t target, uint64_t next) {
#ifdef VGPU_LANES_AVX2
  static const bool avx2 = simd::level() == simd::Level::Avx2;
  if (avx2) return select_pcs_avx2(pc, group, take, target, next);
#endif
  for (Mask m = group; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    pc[l] = ((take >> l) & 1) ? target : next;
  }
}

Mask pick_group_diverged(const uint64_t* pc, Mask run, uint64_t& lo) {
#ifdef VGPU_LANES_AVX2
  static const bool avx2 = simd::level() == simd::Level::Avx2;
  if (avx2) return pick_group_avx2(pc, run, lo);
#endif
  return pick_group_plain(pc, run, lo);
}

Mask pick_group_reference(const uint64_t* pc, Mask run, uint64_t& lo) {
  const uint64_t first = pc[std::countr_zero(run)];
  const Mask at = lanes_at(pc, first);
  if ((run & ~at) == 0) {
    lo = first;
    return run;
  }
  return pick_group_plain(pc, run, lo);
}

}  // namespace vgpu::exec::sasslanes
