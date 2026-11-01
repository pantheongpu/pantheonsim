// The question the SASS tier's global memory handlers ask of a warp's address rows (vgpu/exec/sass_mem_rows.hpp), against the lane-by-lane code
// it replaced:
//  * global_run: whether the lanes of a contiguous mask are on consecutive elements, and the first lane's address, for a base that is a pair of
//    rows, a single row or none, any added offset (carries from the low half into the high one among them), element sizes 1 to 16, and rows
//    that are a run, a run with one lane off by a single bit, or random;
#include <algorithm>
#include <cstdint>
#include <random>

#include "vgpu/exec/sass_mem_rows.hpp"
#include "vtest.hpp"

namespace sm = vgpu::exec::sassmem;

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

VTEST(a_window_that_is_not_on_a_4_gib_boundary_takes_the_scalar_distance) {
  // The vector pass compares high words and so needs a window on a multiple of 4 GiB; any other window is the scalar loop, whose answer is
  // the smallest distance above the window (below 2^32 when a lane is in it).
  std::mt19937_64 rng(23);
  unsigned inside = 0, outside = 0;
  for (int round = 0; round < 50000; ++round) {
    const uint64_t window = (rng() % 8 + 1) * (uint64_t{1} << 32) + (rng() % 2 ? rng() % (uint64_t{1} << 32) : 0x1000);
    uint32_t lo[32], hi[32];
    const unsigned near_lane = rng() % 2 ? rng() % 32 : 32;   // half the warps have one lane within 2 GiB of the window's ends
    for (unsigned l = 0; l < 32; ++l) {
      uint64_t a = l != near_lane ? rng() : window + rng() % (uint64_t{1} << 33) - (uint64_t{1} << 31);
      lo[l] = static_cast<uint32_t>(a);
      hi[l] = static_cast<uint32_t>(a >> 32);
    }
    uint64_t got[32];
    const uint64_t nearest = sm::full_warp_addresses(lo, hi, 0, window, got);
    bool any_in = false;
    for (unsigned l = 0; l < 32; ++l) {
      VCHECK_EQ(got[l], lane_address(lo, hi, 0, l));
      any_in |= got[l] - window < (uint64_t{1} << 32);
    }
    VCHECK_EQ(nearest < (uint64_t{1} << 32), any_in);
    (any_in ? inside : outside) += 1;
  }
  VCHECK(inside > 5000 && outside > 5000);
}

VTEST(a_lane_in_the_window_is_found_in_every_position_and_by_its_high_word_alone) {
  // One lane in the constant window, at each lane number, with the low word anything (the vector compare looks at the high word only), and
  // every other lane a word away from it in the high half; and none.
  constexpr uint64_t kWindow = 0x6ff9'0000'0000ull;
  std::mt19937_64 rng(31);
  for (unsigned at = 0; at < 32; ++at)
    for (int round = 0; round < 200; ++round) {
      uint32_t lo[32], hi[32];
      for (unsigned l = 0; l < 32; ++l) {
        lo[l] = static_cast<uint32_t>(rng());
        hi[l] = static_cast<uint32_t>(kWindow >> 32) + 1 + static_cast<uint32_t>(rng() % 3);   // just above the window
        if (rng() % 3 == 0) hi[l] = static_cast<uint32_t>(kWindow >> 32) - 1 - static_cast<uint32_t>(rng() % 3);   // or just below
      }
      uint64_t got[32];
      VCHECK(sm::full_warp_addresses(lo, hi, 0, kWindow, got) >= (uint64_t{1} << 32));
      hi[at] = static_cast<uint32_t>(kWindow >> 32);
      VCHECK(sm::full_warp_addresses(lo, hi, 0, kWindow, got) < (uint64_t{1} << 32));
      // the same lane found through the added offset's carry: the pair is just below the window and `add` lifts it in
      for (unsigned l = 0; l < 32; ++l) lo[l] = 0;   // (no other lane carries into the window)
      hi[at] = static_cast<uint32_t>(kWindow >> 32) - 1;
      lo[at] = 0xfffffff0u;
      VCHECK(sm::full_warp_addresses(lo, hi, 0x20, kWindow, got) < (uint64_t{1} << 32));
      VCHECK(sm::full_warp_addresses(lo, hi, 0x10, kWindow, got) < (uint64_t{1} << 32));   // (to the window's first byte exactly)
      VCHECK(sm::full_warp_addresses(lo, hi, 0xf, kWindow, got) >= (uint64_t{1} << 32));   // (one byte short)
    }
}

VTEST(full_warp_store_values_are_the_lane_loop) {
  std::mt19937_64 rng(41);
  const unsigned sizes[] = {1, 2, 4, 8};
  for (int round = 0; round < 20000; ++round) {
    uint32_t rows[64];
    for (uint32_t& w : rows) w = static_cast<uint32_t>(rng());
    for (unsigned bytes : sizes) {
      uint64_t got[32], want[32];
      std::fill(got, got + 32, 0x5a5a5a5a5a5a5a5aull);
      sm::full_warp_store_values(rows, bytes, got);
      for (unsigned l = 0; l < 32; ++l) {
        const uint32_t v0 = rows[l], v1 = bytes == 8 ? rows[32 + l] : 0u;
        want[l] = bytes == 8 ? (static_cast<uint64_t>(v1) << 32 | v0) : bytes == 4 ? v0 : (v0 & ((1u << (8 * bytes)) - 1));
        VCHECK_EQ(got[l], want[l]);
      }
    }
  }
  uint64_t z[32];
  std::fill(z, z + 32, 7);
  sm::full_warp_store_values(nullptr, 4, z);   // RZ stores zeros
  for (unsigned l = 0; l < 32; ++l) VCHECK_EQ(z[l], 0u);
}

VTEST_MAIN
