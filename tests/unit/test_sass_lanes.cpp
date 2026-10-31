// The SASS executor's lane-state helpers (vgpu/exec/sass_lanes.hpp) and a full warp's shuffle shapes (src/sass/tier.hpp), against the
// lane-by-lane definitions they replace:
//  * lanes_at (the lanes standing at an address) and lowest_pc (the lowest address among some lanes), with addresses that differ only in the
//    low half, only in the high half, in both, and masks of every kind;
//  * pick_group (the lowest address among the runnable lanes' and the lanes there: the AVX2 pass of src/simd/lanes.cpp where the host has it) against
//    the definition, with addresses of every kind -- equal, differing in one half, above 2^63 (compared as unsigned numbers) -- and every mask;
//  * set_pcs_partial and select_pcs (the addresses a branch gives the lanes of a group), against the lane loop, the lanes outside the group untouched;
//  * the wait state of lanes at a barrier: wait, waiting_on and wake, for whole warps and partial masks, other kinds and other arguments
//    among the lanes (they must stay);
//  * row_uniform (whether all 32 words of a register row are one value);
//  * shfl_move: the words a shuffle shape moves, for every shape the analysis gives, against exec_warp's per-lane rule.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>

#include "sass/tier.hpp"
#include "vgpu/exec/sass_lanes.hpp"
#include "vtest.hpp"

namespace {

using vgpu::exec::sasslanes::Mask;
namespace sl = vgpu::exec::sasslanes;

struct MockWarp {
  uint64_t pc[32] = {};
  Mask waiting = 0;
  uint8_t wait_kind[32] = {};
  uint32_t wait_arg[32] = {};
};

Mask random_mask(std::mt19937_64& rng) {
  switch (rng() % 6) {
    case 0: return 0xffffffffu;
    case 1: return static_cast<Mask>(rng());
    case 2: return static_cast<Mask>(rng()) & static_cast<Mask>(rng());
    case 3: return static_cast<Mask>(rng()) | static_cast<Mask>(rng());
    case 4: return Mask{1} << (rng() % 32);
    default: return (Mask{1} << (rng() % 33 == 32 ? 31 : rng() % 32)) - 1;
  }
}

}  // namespace

VTEST(lanes_at_and_lowest_pc_agree_with_the_lane_loop) {
  std::mt19937_64 rng(7);
  for (int round = 0; round < 20000; ++round) {
    MockWarp w;
    // A few distinct addresses, some differing only in one half of the word.
    uint64_t pool[4];
    const uint64_t base = rng();
    pool[0] = base;
    pool[1] = base ^ (uint64_t{1} << (rng() % 32));              // low half only
    pool[2] = base ^ (uint64_t{1} << (32 + rng() % 32));         // high half only
    pool[3] = rng() % 3 ? rng() : (base & 0xffffffff) | (rng() << 32);
    for (unsigned l = 0; l < 32; ++l) w.pc[l] = pool[rng() % 4];
    for (const uint64_t p : pool) {
      Mask expect = 0;
      for (unsigned l = 0; l < 32; ++l) expect |= static_cast<Mask>(w.pc[l] == p) << l;
      VCHECK_EQ(sl::lanes_at(w.pc, p), expect);
    }
    Mask run = random_mask(rng);
    if (!run) run = 1;
    uint64_t lo = ~uint64_t{0};
    for (unsigned l = 0; l < 32; ++l)
      if ((run >> l) & 1) lo = std::min(lo, w.pc[l]);
    VCHECK_EQ(sl::lowest_pc(w.pc, run, w.pc[__builtin_ctz(run)]), lo);
  }
}

VTEST(pick_group_finds_the_lowest_address_and_its_runnable_lanes) {
  std::mt19937_64 rng(11);
  for (int round = 0; round < 40000; ++round) {
    MockWarp w;
    uint64_t pool[6];
    const uint64_t base = rng() % 4 == 0 ? (uint64_t{1} << 63) + (rng() & 0xffff) * 16 : 0x7f0000000000ull + (rng() & 0xffff) * 16;   // (some above 2^63)
    pool[0] = base;
    pool[1] = base + 16 * (1 + rng() % 4);
    pool[2] = base ^ (uint64_t{1} << (32 + rng() % 16));   // high half only
    pool[3] = base - 16 * (1 + rng() % 4);
    pool[4] = rng() % 8 == 0 ? ~uint64_t{0} : rng();       // (the largest number, which the pass uses for a lane that does not run)
    pool[5] = rng() % 8 == 0 ? 0 : base + 16;
    const unsigned kinds = 1 + rng() % 6;                  // how many of them the warp spreads over
    for (unsigned l = 0; l < 32; ++l) w.pc[l] = pool[rng() % kinds];
    Mask run = random_mask(rng);
    if (!run) run = Mask{1} << (rng() % 32);
    uint64_t want_lo = ~uint64_t{0};
    for (unsigned l = 0; l < 32; ++l)
      if ((run >> l) & 1) want_lo = std::min(want_lo, w.pc[l]);
    Mask want = 0;
    for (unsigned l = 0; l < 32; ++l) want |= static_cast<Mask>(((run >> l) & 1) && w.pc[l] == want_lo) << l;
    uint64_t lo = 0;
    VCHECK_EQ(sl::pick_group(w.pc, run, lo), want);
    VCHECK_EQ(lo, want_lo);
    lo = 0;
    VCHECK_EQ(sl::pick_group_reference(w.pc, run, lo), want);
    VCHECK_EQ(lo, want_lo);
    lo = 0;
    VCHECK_EQ(sl::pick_group_diverged(w.pc, run, lo), want);   // (the pass itself, converged warps included)
    VCHECK_EQ(lo, want_lo);
  }
}

VTEST(set_pcs_and_select_pcs_agree_with_the_lane_loop) {
  std::mt19937_64 rng(13);
  for (int round = 0; round < 40000; ++round) {
    MockWarp w;
    for (unsigned l = 0; l < 32; ++l) w.pc[l] = 0x1000 + 16 * (rng() % 9);
    const Mask group = random_mask(rng), take = random_mask(rng) & group;
    const uint64_t target = rng() % 5 == 0 ? rng() : 0x2000 + 16 * (rng() % 9), next = rng() % 5 == 0 ? (uint64_t{1} << 63) + rng() % 99 : 0x3000 + 16 * (rng() % 9);
    MockWarp a = w, ref = w;
    sl::set_pcs_partial(a.pc, group, target);
    for (unsigned l = 0; l < 32; ++l)
      if ((group >> l) & 1) ref.pc[l] = target;
    VCHECK(std::memcmp(&a, &ref, sizeof a) == 0);
    MockWarp b = w, ref2 = w;
    sl::select_pcs(b.pc, group, take, target, next);
    for (unsigned l = 0; l < 32; ++l)
      if ((group >> l) & 1) ref2.pc[l] = ((take >> l) & 1) ? target : next;
    VCHECK(std::memcmp(&b, &ref2, sizeof b) == 0);
  }
}

VTEST(wait_state_helpers_agree_with_the_lane_loop) {
  std::mt19937_64 rng(11);
  for (int round = 0; round < 20000; ++round) {
    MockWarp a;
    for (unsigned l = 0; l < 32; ++l) a.pc[l] = rng() % 7 * 16 + 0x1000;
    // Some lanes waiting on something (kinds 1 to 4, arguments 0 to 3).
    a.waiting = random_mask(rng);
    for (unsigned l = 0; l < 32; ++l) {
      a.wait_kind[l] = ((a.waiting >> l) & 1) ? static_cast<uint8_t>(1 + rng() % 4) : 0;
      a.wait_arg[l] = static_cast<uint32_t>(rng() % 4);
    }
    const uint8_t kind = static_cast<uint8_t>(1 + rng() % 4);
    const uint32_t arg = static_cast<uint32_t>(rng() % 4);
    // waiting_on: waiting lanes of this kind on this argument.
    Mask expect = 0;
    for (unsigned l = 0; l < 32; ++l)
      if (((a.waiting >> l) & 1) && a.wait_kind[l] == kind && a.wait_arg[l] == arg) expect |= Mask{1} << l;
    VCHECK_EQ(sl::waiting_on(a, kind, arg), expect);
    // wake: those lanes stop waiting and go on to the next instruction; nothing else changes.
    MockWarp b = a;
    sl::wake(b, expect);
    MockWarp ref = a;
    for (unsigned l = 0; l < 32; ++l)
      if ((expect >> l) & 1) {
        ref.waiting &= ~(Mask{1} << l);
        ref.wait_kind[l] = 0;
        ref.pc[l] += 16;
      }
    VCHECK(std::memcmp(&b, &ref, sizeof b) == 0);
    // wait: lanes of a mask start to wait (a whole warp, or some of it).
    const Mask ex = random_mask(rng);
    MockWarp c = a;
    sl::wait(c, ex, kind, arg);
    MockWarp ref2 = a;
    for (unsigned l = 0; l < 32; ++l)
      if ((ex >> l) & 1) {
        ref2.wait_kind[l] = kind;
        ref2.wait_arg[l] = arg;
      }
    ref2.waiting |= ex;
    VCHECK(std::memcmp(&c, &ref2, sizeof c) == 0);
  }
  // A whole warp that waits, and is let go: every lane's address moves on by 16.
  MockWarp w;
  for (unsigned l = 0; l < 32; ++l) w.pc[l] = 0x2000 + 16 * l;
  sl::wait(w, 0xffffffffu, uint8_t{3}, 5);
  VCHECK_EQ(sl::waiting_on(w, uint8_t{3}, 5), 0xffffffffu);
  VCHECK_EQ(sl::waiting_on(w, uint8_t{3}, 4), 0u);
  VCHECK_EQ(sl::waiting_on(w, uint8_t{2}, 5), 0u);
  sl::wake(w, 0xffffffffu);
  VCHECK_EQ(w.waiting, 0u);
  for (unsigned l = 0; l < 32; ++l) {
    VCHECK_EQ(w.pc[l], 0x2000 + 16 * l + 16);
    VCHECK_EQ(w.wait_kind[l], 0);
  }
}

VTEST(row_uniform_finds_a_row_of_one_value) {
  std::mt19937_64 rng(3);
  for (int round = 0; round < 5000; ++round) {
    uint32_t r[32];
    const uint32_t v = static_cast<uint32_t>(rng());
    for (uint32_t& x : r) x = v;
    uint32_t got = 0;
    VCHECK(sl::row_uniform(r, got));
    VCHECK_EQ(got, v);
    r[rng() % 32] ^= 1u << (rng() % 32);   // one lane differs, in one bit
    uint32_t again = 0;
    VCHECK(!sl::row_uniform(r, again));
  }
}

VTEST(shuffle_shapes_move_the_words_the_per_lane_rule_moves) {
  using vgpu::sass::tier::ShflShape;
  std::mt19937_64 rng(5);
  size_t moved = 0;
  for (unsigned mode = 0; mode < 4; ++mode)
    for (uint32_t b = 0; b < 33; ++b)
      for (uint32_t clamp = 0; clamp < 32; ++clamp) {
        const uint32_t cc = clamp;   // a segment mask of 0 (the shapes' case)
        ShflShape sh;
        if (!vgpu::sass::tier::shfl_shape(mode, b, cc, sh)) continue;
        uint32_t in[32], out[32], expect[32];
        for (uint32_t& x : in) x = static_cast<uint32_t>(rng());
        for (uint32_t l = 0; l < 32; ++l) {
          uint32_t j;
          bool ok;
          const uint32_t maxl = clamp, minl = 0;
          switch (mode) {
            case 0: j = minl | (b & 0x1f); ok = j <= maxl; break;
            case 1: j = l - b; ok = static_cast<int32_t>(j) >= static_cast<int32_t>(maxl); break;
            case 2: j = l + b; ok = j <= maxl; break;
            default: j = l ^ b; ok = j <= maxl; break;
          }
          expect[l] = in[ok ? (j & 31) : l];
        }
        std::memset(out, 0xa5, sizeof out);
        vgpu::sass::tier::shfl_move(sh, in, out);
        VCHECK(std::memcmp(out, expect, sizeof out) == 0);
        ++moved;
      }
  VCHECK(moved > 1000);
}

VTEST_MAIN
