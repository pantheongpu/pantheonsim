// The question the SASS tier's global memory handlers ask of a warp's address rows (vgpu/exec/sass_mem_rows.hpp), against the lane-by-lane code
// it replaced:
//  * global_run: whether the lanes of a contiguous mask are on consecutive elements, and the first lane's address, for a base that is a pair of
//    rows, a single row or none, any added offset (carries from the low half into the high one among them), element sizes 1 to 16, and rows
//    that are a run, a run with one lane off by a single bit, or random;
#include <algorithm>
#include <cstdint>
#include <random>

#include "vgpu/exec/mem_counters.hpp"
#include "vgpu/exec/sass_mem_rows.hpp"
#include "vgpu/simd/simd.hpp"
#include "vtest.hpp"

namespace sm = vgpu::exec::sassmem;
namespace simd = vgpu::simd;

namespace {

uint64_t lane_address(const uint32_t* lo, const uint32_t* hi, uint64_t add, unsigned l) {
  if (!lo) return add;
  return (hi ? (static_cast<uint64_t>(hi[l]) << 32) | lo[l] : uint64_t{lo[l]}) + add;
}

}  // namespace

VTEST(global_run_agrees_with_the_lane_loop) {
  std::mt19937_64 rng(5);
  unsigned runs = 0, nonruns = 0;
  for (int round = 0; round < 200000; ++round) {
    uint32_t lo[32], hi[32];
    const unsigned bytes = 1u << (rng() % 5);
    const unsigned kind = rng() % 6;   // 0 pair, 1 pair with a run, 2 row, 3 row with a run, 4 none, 5 run through a carry
    const bool pair = kind == 0 || kind == 1 || kind >= 5;
    const bool run = kind == 1 || kind == 3 || kind >= 5;
    const uint64_t add = pair && rng() % 3 == 0 ? rng() : rng() % 4096;
    uint64_t start = !pair ? add + rng() % 1000000 : rng() % 3 == 0 ? rng() : 0x7f0000000000ull + (rng() & 0xffffff);
    if (kind == 5) start = 0xfffffff0ull - (rng() % 8) * bytes + (rng() % 2 ? 0 : uint64_t{1} << 32);
    for (unsigned l = 0; l < 32; ++l) {
      uint64_t a = run ? start + uint64_t{l} * bytes - add : rng();
      if (!pair) a = static_cast<uint32_t>(a);
      lo[l] = static_cast<uint32_t>(a);
      hi[l] = static_cast<uint32_t>(a >> 32);
    }
    // A mask of consecutive lanes (a full warp half the time), and sometimes one lane's address a bit off.
    unsigned first = 0, count = 32;
    if (rng() % 2) {
      first = rng() % 32;
      count = 1 + rng() % (32 - first);
    }
    if (run && rng() % 3 == 0) {
      const unsigned l = first + rng() % count;
      (pair && rng() % 2 ? hi : lo)[l] ^= 1u << (rng() % 32);
    }
    const uint32_t* plo = kind == 4 ? nullptr : lo;
    const uint32_t* phi = pair ? hi : nullptr;
    uint64_t want_g0 = lane_address(plo, phi, add, first), differ = 0;
    for (unsigned i = 0; i < count; ++i) differ |= lane_address(plo, phi, add, first + i) ^ (want_g0 + uint64_t{i} * bytes);
    uint64_t g0 = 0;
    VCHECK_EQ(sm::global_run(plo, phi, add, first, count, bytes, g0), differ == 0);
    VCHECK_EQ(g0, want_g0);
    (differ == 0 ? runs : nonruns) += 1;
  }
  VCHECK(runs > 20000 && nonruns > 20000);
}

VTEST(full_warp_addresses_are_the_lane_loops_and_find_a_lane_in_the_window) {
  std::mt19937_64 rng(11);
  constexpr uint64_t kWindow = 0x6ff9'0000'0000ull, kSpan = uint64_t{1} << 32;
  unsigned inside = 0, outside = 0, edges = 0;
  for (int round = 0; round < 100000; ++round) {
    uint32_t lo[32], hi[32];
    const unsigned kind = rng() % 3;   // a pair of rows, a row, none
    const uint64_t add = rng() % 2 ? rng() % 4096 : rng();
    // Most warps far from the window; some with a lane (or all) in it, at its first byte, its last, and one before and after.
    const unsigned where = rng() % 6;   // 0..2 far, 3 in, 4 on the edges, 5 near the edges
    for (unsigned l = 0; l < 32; ++l) {
      uint64_t a = rng();
      if (kind == 1) a = static_cast<uint32_t>(a);
      if (where == 3 && (l == 0 || rng() % 4 == 0)) a = kWindow + rng() % kSpan - add;
      if (where >= 4 && l == rng() % 32) {
        static const int64_t d[] = {0, -1, 1, static_cast<int64_t>(kSpan) - 1, static_cast<int64_t>(kSpan), static_cast<int64_t>(kSpan) + 1};
        a = kWindow + static_cast<uint64_t>(d[rng() % 6]) - add;
      }
      if (kind == 1 && where == 3) a = static_cast<uint32_t>(a);
      lo[l] = static_cast<uint32_t>(a);
      hi[l] = static_cast<uint32_t>(a >> 32);
    }
    const uint32_t* plo = kind == 2 ? nullptr : lo;
    const uint32_t* phi = kind == 0 ? hi : nullptr;
    uint64_t got[32];
    const uint64_t nearest = sm::full_warp_addresses(plo, phi, add, kWindow, got);
    bool any_in = false;
    for (unsigned l = 0; l < 32; ++l) {
      const uint64_t want = lane_address(plo, phi, add, l);
      VCHECK_EQ(got[l], want);
      any_in |= want >= kWindow && want < kWindow + kSpan;
      edges += want == kWindow || want == kWindow + kSpan - 1 || want == kWindow - 1 || want == kWindow + kSpan;
    }
    VCHECK_EQ(nearest < kSpan, any_in);
    (any_in ? inside : outside) += 1;
  }
  VCHECK(inside > 5000 && outside > 10000 && edges > 1000);
}

// The AVX2 kernel behind the tier's full-warp global access (simd::warp_rows_kernel; null on a host or at a VGPU_SIMD level without AVX2, where the scalar
// full_warp_addresses is all there is and the VGPU_SIMD=base twin of this test runs only the checks that do not need it): the addresses, the first
// lane's, the stride of the first two, whether a lane is in the window and whether all lanes are on the stride, against the lane definitions, over the
// shapes a warp's addresses come in -- every lane the same, a stride, a carry out of the low half, a wrap of the 64-bit sum -- and a lane moved off the
// stride by one bit or into the window alone. And the sectors of such a warp from (a0, stride) against the lane-walking count of its addresses.
VTEST(warp_rows_kernel_shapes) {
  constexpr uint64_t kWindow = 0x6ff9'0000'0000ull, kSpan = uint64_t{1} << 32;
  const uint64_t adds[] = {0, 1, 0xffff'ffffull, 0x1'0000'0000ull, ~uint64_t{0}, ~uint64_t{0} - 7, 0x8000'0000'0000'0000ull, kWindow};
  const uint64_t bases[] = {0, 0xffff'ffe0ull, 0xffff'ffffull, 0x7f00'0000'0000ull, kWindow - 4, kWindow - 1, kWindow, kWindow + kSpan - 1,
                            kWindow + kSpan, ~uint64_t{0} - 100, 0xffff'ffff'ffff'ff00ull, 0x7fff'ffff'fff0ull};
  const int64_t strides[] = {0, 1, 4, 8, 16, 32, 33, 128, 1000, 0x1000'0000, -4, -32, -1000, int64_t{1} << 32, -(int64_t{1} << 32), (int64_t{1} << 32) + 32};
  const simd::WarpRowsFn kernel = simd::warp_rows_kernel();
  unsigned prog = 0, off = 0, inw = 0, counted = 0, declined = 0;
  for (uint64_t add : adds)
    for (uint64_t base : bases)
      for (int64_t stride : strides)
        for (unsigned kind = 0; kind < 3; ++kind)
          for (unsigned lone = 0; lone < 34; ++lone) {   // lone: 32 none, 33 a bit flipped in a lane, else a lane moved into the window alone
            uint32_t lo[32], hi[32];
            for (unsigned l = 0; l < 32; ++l) {
              uint64_t a = base + uint64_t{l} * static_cast<uint64_t>(stride) - add;
              if (lone < 32 && l == lone) a = kWindow + 5 - add;
              if (lone == 33 && l == 19) a ^= uint64_t{1} << (l % 40);
              if (kind == 1) a = static_cast<uint32_t>(a);
              lo[l] = static_cast<uint32_t>(a);
              hi[l] = static_cast<uint32_t>(a >> 32);
            }
            const uint32_t* plo = kind == 2 ? nullptr : lo;
            const uint32_t* phi = kind == 0 ? hi : nullptr;
            uint64_t ref[32];
            bool want_in = false, want_prog = true;
            for (unsigned l = 0; l < 32; ++l) {
              ref[l] = lane_address(plo, phi, add, l);
              want_in |= ref[l] - kWindow < kSpan;
            }
            const uint64_t want_stride = ref[1] - ref[0];
            for (unsigned l = 0; l < 32; ++l) want_prog &= ref[l] == ref[0] + uint64_t{l} * want_stride;
            if (kernel) {
              uint64_t got[32];
              simd::WarpRowsInfo ri{};
              kernel(plo, phi, add, kWindow, got, ri);
              for (unsigned l = 0; l < 32; ++l) VCHECK_EQ(got[l], ref[l]);
              VCHECK_EQ(ri.a0, ref[0]);
              VCHECK_EQ(ri.stride, want_stride);
              VCHECK_EQ(ri.in_window, want_in);
              VCHECK_EQ(ri.progression, want_prog);
              (want_prog ? prog : off) += 1;
              inw += want_in;
            }
            if (want_prog) {   // the closed form of the progression against the count of its addresses, whenever it answers
              for (unsigned bytes : {1u, 2u, 4u, 8u}) {
                uint64_t d;
                if (vgpu::exec::memcount::detail::full_warp_progression_sectors(ref[0], want_stride, bytes, d)) {
                  VCHECK_EQ(d, vgpu::exec::memcount::distinct_sectors_uncapped(ref, 0xffffffffu, 32, bytes));
                  ++counted;
                } else {
                  ++declined;
                }
              }
            }
          }
  if (simd::level() == simd::Level::Avx2) VCHECK(kernel != nullptr);   // (the kernel is there wherever the host and VGPU_SIMD have AVX2)
  if (kernel) VCHECK(prog > 1000 && off > 1000 && inw > 500);
  VCHECK(counted > 1000 && declined > 100);
}

VTEST_MAIN
