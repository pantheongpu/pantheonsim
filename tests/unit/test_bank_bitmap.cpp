// The bitmap counting of bank passes for 1 to 4 byte accesses (vgpu/exec/mem_counters.hpp, detail::bank_passes_bitmap, which
// sass_bank_passes_table takes before its table of words), against a definition written from scratch: the distinct words the
// bytes of every active lane cover, the most in any one bank.
//
// The counting puts each lane's word into a bitmap of 32-bit rows (a row is 32 consecutive words, a bit is a bank) and counts a bank
// up when its bit was not set. Everything it can get wrong is a shape here: a spread over rows, a lowest word that is not a
// multiple of 32, words that two lanes share, lanes whose bytes run into the next word at every byte alignment, a lane at the last
// bit of a row, one active lane, sparse masks, the widest spread the bitmap holds and one word more (counted by the table
// instead), and addresses at the top of the space (counted by the reference). Every answer is also asked through sass_bank_passes_table
// directly and through sass_bank_passes (which finds the strides first).
//
// The last two tests are the warp kept while the offsets table is left alone (BankMemo::last_offs, sass_bank_passes_offsets_kept): the same
// warp again is answered without counting, and a warp that differs in any one offset (each group of four lanes), in the mask or in the width is not.
#include <algorithm>
#include <cstdint>
#include <vector>

#include "vgpu/exec/mem_counters.hpp"
#include "vtest.hpp"

namespace mc = vgpu::exec::memcount;

namespace {

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  uint64_t below(uint64_t n) { return next() % n; }
};

uint32_t passes_def(const uint64_t* a, uint32_t mask, uint32_t bytes) {
  std::vector<uint64_t> words[32];
  for (unsigned l = 0; l < 32; ++l) {
    if (!(mask >> l & 1)) continue;
    for (uint64_t w = a[l] / 4; w <= (a[l] + bytes - 1) / 4; ++w) words[w % 32].push_back(w);
  }
  size_t worst = 1;
  for (auto& v : words) {
    std::sort(v.begin(), v.end());
    worst = std::max<size_t>(worst, static_cast<size_t>(std::unique(v.begin(), v.end()) - v.begin()));
  }
  return static_cast<uint32_t>(worst);
}

void check(const uint64_t* a, uint32_t mask, uint32_t bytes) {
  const uint32_t want = passes_def(a, mask, bytes);
  VCHECK_EQ(mc::sass_bank_passes_table(a, mask, 32, bytes), want);
  VCHECK_EQ(mc::sass_bank_passes(a, mask, 32, bytes), want);
}

uint32_t random_mask(Rng& r) {
  switch (r.below(5)) {
    case 0: return 0xFFFFFFFFu;
    case 1: return 1u << r.below(32);
    case 2: return static_cast<uint32_t>(r.next()) | 1u;
    case 3: return (1u << (1 + r.below(31))) - 1;
    default: return static_cast<uint32_t>(r.next() & r.next()) | (1u << r.below(32));
  }
}

}  // namespace

VTEST(random_words_in_a_histogram_sized_array) {
  Rng r(1);
  for (int i = 0; i < 20000; ++i) {
    uint64_t a[32];
    const uint64_t base = r.below(4) == 0 ? 0 : r.below(1 << 20);
    const uint64_t words = 1 + r.below(r.below(2) ? 64 : 1536);
    const uint32_t bytes = static_cast<uint32_t>(1 + r.below(4));
    for (auto& x : a) x = base + 4 * r.below(words) + (bytes == 4 ? 0 : r.below(4));
    check(a, random_mask(r), bytes);
  }
}

VTEST(lanes_whose_bytes_run_into_the_next_word_at_every_alignment) {
  Rng r(2);
  for (int i = 0; i < 20000; ++i) {
    uint64_t a[32];
    const uint32_t bytes = static_cast<uint32_t>(1 + r.below(4));
    const uint64_t words = 1 + r.below(70);
    for (auto& x : a) x = 4 * (31 + r.below(words)) + r.below(4);   // (rows start at the lowest word's row: lanes at the last bit of one)
    check(a, random_mask(r), bytes);
  }
  // Two lanes whose second words are the same word, and a lane whose second word is another lane's first: counted once.
  uint64_t a[32] = {};
  for (unsigned l = 0; l < 32; ++l) a[l] = 4 * (l * 32) + 3;
  check(a, 0xFFFFFFFFu, 2);   // (each lane covers its word and the next, bank 0 and 1: 32 words in bank 0, 32 in bank 1)
  for (unsigned l = 0; l < 32; ++l) a[l] = 4 * (l % 2) + 3;
  check(a, 0xFFFFFFFFu, 2);
  check(a, 0x55555555u, 2);
}

VTEST(the_lowest_word_is_any_bank_and_rows_are_crossed) {
  Rng r(3);
  for (unsigned lowest = 0; lowest < 70; ++lowest) {
    for (int i = 0; i < 200; ++i) {
      uint64_t a[32];
      for (auto& x : a) x = 4 * (uint64_t{lowest} + r.below(100)) + 40000;
      a[r.below(32)] = 4 * uint64_t{lowest} + 40000;
      check(a, 0xFFFFFFFFu, static_cast<uint32_t>(1 + r.below(4)));
    }
  }
}

VTEST(the_widest_spread_the_bitmap_holds_and_one_word_more) {
  Rng r(4);
  for (uint64_t lowest : {0ull, 1ull, 31ull, 32ull, 33ull, 1000ull, 100000ull}) {
    for (uint64_t span : {2046ull, 2047ull, 2048ull, 2049ull, 2050ull, 2100ull, 4096ull, 1ull << 20}) {
      for (uint32_t bytes : {1u, 3u, 4u}) {
        for (int i = 0; i < 30; ++i) {
          uint64_t a[32];
          for (auto& x : a) x = 4 * (lowest + r.below(span));
          a[0] = 4 * lowest;
          a[1] = 4 * (lowest + span - 1) + (bytes == 4 ? 0 : 3);
          check(a, 0xFFFFFFFFu, bytes);
        }
      }
    }
  }
}

VTEST(the_bitmap_takes_2048_words_from_the_lowest_and_no_more) {
  // (Directly: past its end the bitmap would be written outside the array; the table counts such a warp instead.)
  uint64_t a[32];
  uint32_t out = 0;
  for (uint64_t lowest : {0ull, 5ull, 31ull, 1000ull}) {
    for (unsigned l = 0; l < 32; ++l) a[l] = 4 * (lowest + l);
    a[31] = 4 * (lowest + 2047);   // (the last word the bitmap holds)
    VCHECK(mc::detail::bank_passes_bitmap(a, 0xFFFFFFFFu, 4, out));
    VCHECK_EQ(out, passes_def(a, 0xFFFFFFFFu, 4));
    a[31] = 4 * (lowest + 2048);
    VCHECK(!mc::detail::bank_passes_bitmap(a, 0xFFFFFFFFu, 4, out));
    a[31] = 4 * (lowest + 2047) + 3;   // (a lane whose bytes run into word 2048)
    VCHECK(!mc::detail::bank_passes_bitmap(a, 0xFFFFFFFFu, 2, out));
    VCHECK(mc::detail::bank_passes_bitmap(a, 0xFFFFFFFFu, 1, out));
  }
}

VTEST(a_lane_on_the_last_word_beside_a_lane_that_runs_over_stays_inside_the_array) {
  // (Some lane's bytes run into a second word, so the general loop runs; a lane on word 2047 whose bytes do not run over must not touch
  // row 64 of 64: caught only under a sanitizer, the answer is right either way.)
  uint64_t a[32];
  uint32_t out = 0;
  for (uint64_t lowest : {0ull, 5ull, 1000ull}) {
    for (unsigned l = 0; l < 32; ++l) a[l] = 4 * (lowest + l);
    a[30] = 4 * (lowest + 2047);        // (the last word, 2 bytes from its start: no run over)
    a[31] = 4 * (lowest + 5) + 3;       // (3 + 2 > 4: runs into the next word)
    VCHECK(mc::detail::bank_passes_bitmap(a, 0xFFFFFFFFu, 2, out));
    VCHECK_EQ(out, passes_def(a, 0xFFFFFFFFu, 2));
    a[30] = 4 * (lowest + 2047) + 3;    // (this one runs over into word 2048: the bitmap declines)
    VCHECK(!mc::detail::bank_passes_bitmap(a, 0xFFFFFFFFu, 2, out));
  }
}

VTEST(lanes_on_one_word_a_bank_each_and_all_in_one_bank) {
  uint64_t a[32];
  for (unsigned l = 0; l < 32; ++l) a[l] = 4096;
  check(a, 0xFFFFFFFFu, 4);
  for (unsigned l = 0; l < 32; ++l) a[l] = 4096 + 4 * l;
  check(a, 0xFFFFFFFFu, 4);
  VCHECK_EQ(mc::sass_bank_passes_table(a, 0xFFFFFFFFu, 32, 4), 1u);
  for (unsigned l = 0; l < 32; ++l) a[l] = 4096 + 128 * l;   // (a word of bank 0 each: 32 passes)
  check(a, 0xFFFFFFFFu, 4);
  VCHECK_EQ(mc::sass_bank_passes_table(a, 0xFFFFFFFFu, 32, 4), 32u);
  for (unsigned l = 0; l < 32; ++l) a[l] = 4096 + 128 * (l % 5);   // (five words of bank 0)
  check(a, 0xFFFFFFFFu, 4);
  VCHECK_EQ(mc::sass_bank_passes_table(a, 0xFFFFFFFFu, 32, 4), 5u);
}

VTEST(addresses_at_the_top_of_the_space_and_far_apart) {
  Rng r(5);
  for (int i = 0; i < 2000; ++i) {
    uint64_t a[32];
    const uint32_t bytes = static_cast<uint32_t>(1 + r.below(4));
    const uint64_t top = r.below(3) == 0 ? UINT64_MAX - 1000 : (r.below(2) ? uint64_t{1} << 40 : (uint64_t{1} << 63) - 100);
    for (auto& x : a) x = top + 4 * r.below(200) + r.below(4);
    if (r.below(2)) a[r.below(32)] = r.below(100);
    check(a, random_mask(r), bytes);
  }
}

// ---- the warp asked about last (sass_bank_passes_offsets answers a warp that comes twice running without the table)

namespace {
uint32_t offsets_def(const uint32_t* o, uint32_t mask, uint32_t bytes) {
  uint64_t a[32];
  for (unsigned l = 0; l < 32; ++l) a[l] = (mask >> l & 1) ? o[l] : 0;
  return passes_def(a, mask, bytes);
}
}  // namespace

VTEST(a_warp_asked_twice_running_is_answered_and_one_that_differs_in_anything_is_counted) {
  // A sequence of calls on ONE table: the warp again (a load and a store of the same elements), or the warp with one lane's offset, one
  // lane's active bit, the width, or all of it changed. Each answer is the definition's. The offsets come from 40 words so that a change of
  // one lane often changes the passes.
  Rng r(11);
  mc::BankMemo memo;
  uint32_t o[32];
  uint32_t mask = 0xFFFFFFFFu, bytes = 4;
  const auto fresh = [&] {
    for (auto& x : o) x = 4096 + 4 * static_cast<uint32_t>(r.below(40)) + (r.below(3) == 0 ? static_cast<uint32_t>(r.below(4)) : 0);
    mask = r.below(3) == 0 ? static_cast<uint32_t>(r.next()) | 1u : 0xFFFFFFFFu;
    bytes = r.below(2) ? 4 : static_cast<uint32_t>(1 + r.below(16));
  };
  fresh();
  for (int i = 0; i < 60000; ++i) {
    if (i % 700 == 0) memo.skip32 = 600;   // (the warp asked last is kept while the table is left alone: most of these calls)
    switch (r.below(8)) {
      case 0: case 1: case 2: break;   // the same warp
      case 3: o[r.below(32)] += 4 * static_cast<uint32_t>(1 + r.below(40)); break;
      case 4: o[r.below(32)] = 4096 + 4 * static_cast<uint32_t>(r.below(40)); break;
      case 5: mask ^= 1u << r.below(32); if (!mask) mask = 1; break;
      case 6: bytes = bytes == 4 ? 8 : 4; break;
      default: fresh(); break;
    }
    VCHECK_EQ(mc::sass_bank_passes_offsets(memo, o, mask, bytes), offsets_def(o, mask, bytes));
  }
  // A lane that does not run holds whatever its register does: the same warp with other values there is still right.
  for (int i = 0; i < 2000; ++i) {
    memo.skip32 = 5;
    fresh();
    mask = static_cast<uint32_t>(r.next()) | 1u;
    const uint32_t want = offsets_def(o, mask, bytes);
    VCHECK_EQ(mc::sass_bank_passes_offsets(memo, o, mask, bytes), want);
    for (unsigned l = 0; l < 32; ++l) if (!(mask >> l & 1)) o[l] = static_cast<uint32_t>(r.next());
    VCHECK_EQ(mc::sass_bank_passes_offsets(memo, o, mask, bytes), want);
  }
}

VTEST(a_warp_that_changed_in_any_group_of_four_lanes_and_changed_back_is_not_taken_for_the_one_between) {
  // A, then B (A with the lanes of one group of four moved onto other words of bank 0, so that the passes change), then C (B with that group
  // as A had it): C is not B, and a table that kept only part of a warp's offsets would take it for B.
  for (unsigned g = 0; g < 8; ++g) {
    mc::BankMemo memo;
    memo.skip32 = 100;
    uint32_t a[32], b[32], c[32];
    for (unsigned l = 0; l < 32; ++l) a[l] = 4096 + 4 * l;
    for (unsigned l = 0; l < 4; ++l) a[4 * g + l] = 0;   // (zero: what a table that never kept this group would still hold)
    for (unsigned l = 0; l < 32; ++l) b[l] = a[l];
    for (unsigned l = 0; l < 4; ++l) b[4 * g + l] = 128 * (l + 1);
    for (unsigned l = 0; l < 32; ++l) c[l] = b[l];
    for (unsigned l = 0; l < 4; ++l) c[4 * g + l] = a[4 * g + l];
    VCHECK_EQ(mc::sass_bank_passes_offsets(memo, a, 0xFFFFFFFFu, 4), offsets_def(a, 0xFFFFFFFFu, 4));
    VCHECK_EQ(mc::sass_bank_passes_offsets(memo, b, 0xFFFFFFFFu, 4), offsets_def(b, 0xFFFFFFFFu, 4));
    VCHECK(offsets_def(b, 0xFFFFFFFFu, 4) != offsets_def(c, 0xFFFFFFFFu, 4));
    VCHECK_EQ(mc::sass_bank_passes_offsets(memo, c, 0xFFFFFFFFu, 4), offsets_def(c, 0xFFFFFFFFu, 4));
    VCHECK_EQ(mc::sass_bank_passes_offsets(memo, c, 0xFFFFFFFFu, 4), offsets_def(c, 0xFFFFFFFFu, 4));
  }
}

VTEST_MAIN
