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

VTEST_MAIN
