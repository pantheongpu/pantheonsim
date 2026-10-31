// The table of bank passes by warp shape (vgpu/exec/mem_counters.hpp, BankMemo / sass_bank_passes_memo) against the
// function it sits in front of.
//
// A warp's passes are remembered by its shape (active lanes, width, every lane's address less the first active lane's
// word), so a warp that comes back in the same shape anywhere else in the array is answered without counting. That is
// only right if the key holds everything the passes depend on, and the counters it feeds are reported to users, so
// every answer here is compared with a definition written from scratch (sorted vectors of the words of each bank),
// with ONE table kept across all the calls so that hits, evictions (a table of 64 entries is given far more shapes
// than it holds, and shapes that share a slot), the back-off after a run of misses, and entries that were stored for
// a different warp all happen. The shapes are the ones kernels make and that no closed form takes: two rows of a tile,
// a pair of words every four, groups of lanes on one word, a data-dependent gather, a column of a padded tile; each
// at every byte alignment, every width 1..64 bytes, full, partial and sparse lane masks, shifted by whole words, by
// bytes, by a cluster rank in the high bits and to the top of what the table takes (addresses of 2^31 and over go to
// the counting). A shape that is the same except for one lane, one byte or one active lane must not hit.
#include <algorithm>
#include <cstdint>
#include <cstring>
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

// The SASS executor's definition: the distinct words the bytes of every active lane cover, the most in any one bank.
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

// A shape: a warp's offsets from its first lane, in bytes, and its mask.
struct Shape {
  uint64_t off[32];
};

Shape make_shape(Rng& r, unsigned kind) {
  Shape s{};
  switch (kind) {
    case 0: {   // two (or four) rows of a tile: rows of `seg` lanes, `row` words apart
      const unsigned seg = 1u << (1 + r.below(4)), row = static_cast<unsigned>(1 + r.below(200));
      const unsigned step = static_cast<unsigned>(1 + r.below(3));
      for (unsigned l = 0; l < 32; ++l) s.off[l] = 4 * (uint64_t{l / seg} * row + uint64_t{l % seg} * step);
      break;
    }
    case 1: {   // a group of lanes on each word, the groups some words apart
      const unsigned g = 1u << (1 + r.below(4)), gap = static_cast<unsigned>(1 + r.below(40));
      for (unsigned l = 0; l < 32; ++l) s.off[l] = 4 * (uint64_t{l / g} * gap);
      break;
    }
    case 2: {   // a pair of words every `period`
      const unsigned period = static_cast<unsigned>(2 + r.below(14));
      for (unsigned l = 0; l < 32; ++l) s.off[l] = 4 * (uint64_t{l / 2} * period + l % 2);
      break;
    }
    case 3: {   // a gather over a small window (many lanes share words and banks)
      const unsigned win = static_cast<unsigned>(1 + r.below(300));
      for (unsigned l = 0; l < 32; ++l) s.off[l] = 4 * r.below(win);
      break;
    }
    case 4: {   // a gather over bytes
      const unsigned win = static_cast<unsigned>(1 + r.below(2000));
      for (unsigned l = 0; l < 32; ++l) s.off[l] = r.below(win);
      break;
    }
    case 5: {   // a permutation of one row of banks, then lanes moved to another row
      unsigned p[32];
      for (unsigned i = 0; i < 32; ++i) p[i] = i;
      for (unsigned i = 31; i > 0; --i) std::swap(p[i], p[r.below(i + 1)]);
      for (unsigned l = 0; l < 32; ++l) s.off[l] = 4 * (p[l] + 32 * r.below(1 + r.below(3)));
      break;
    }
    default: {   // a bitonic-network style pair index: 2 * i - (i & (stride - 1))
      const unsigned stride = 1u << r.below(6);
      for (unsigned l = 0; l < 32; ++l) s.off[l] = 4 * (2 * l - (l & (stride - 1)));
      break;
    }
  }
  return s;
}

}  // namespace

VTEST(shapes_are_counted_as_the_definition_counts_them) {
  Rng r(20261029);
  mc::BankMemo memo;
  const uint32_t widths[] = {1, 2, 3, 4, 4, 4, 4, 6, 8, 8, 12, 16, 16, 24, 32, 64};
  std::vector<Shape> pool;
  for (int i = 0; i < 400; ++i) pool.push_back(make_shape(r, static_cast<unsigned>(r.below(7))));
  uint64_t checked = 0;
  for (int i = 0; i < 400000; ++i) {
    const Shape& sh = pool[r.below(r.below(8) == 0 ? pool.size() : 40)];   // a hot few, now and then any
    const uint32_t bytes = widths[r.below(16)];
    uint32_t mask = 0xFFFFFFFFu;
    switch (r.below(6)) {
      case 0: mask = static_cast<uint32_t>(r.next() | r.next()); break;
      case 1: mask = static_cast<uint32_t>(r.next()); break;
      case 2: mask = 0xFFFFu << r.below(16); break;
      case 3: mask = 1u << r.below(32); break;
      default: break;
    }
    if (!mask) mask = 1;
    uint64_t base;
    switch (r.below(8)) {
      case 0: base = r.below(1 << 14); break;                                  // any byte
      case 1: base = (r.below(1 << 12)) * 4 + (r.below(4) == 0 ? r.below(4) : 0); break;
      case 2: base = (uint64_t{r.below(16)} << 24) + r.below(1 << 20) * 4; break;   // a cluster rank above the offset
      case 3: base = (uint64_t{1} << 31) - 600 + r.below(1200); break;           // the edge of what the table takes
      case 4: base = (uint64_t{1} << 32) + r.below(4096) * 4; break;
      default: base = r.below(1 << 16) * 4; break;
    }
    uint64_t a[32];
    for (unsigned l = 0; l < 32; ++l) a[l] = (mask >> l & 1) ? base + sh.off[l] : 0;   // (lanes that do not run hold zero)
    if (r.below(5) == 0) {                                                      // ... or whatever the caller left there
      for (unsigned l = 0; l < 32; ++l)
        if (!(mask >> l & 1)) a[l] = r.next();
    }
    if (r.below(7) == 0) a[r.below(32)] += 4 * (1 + r.below(70)) + (r.below(3) == 0 ? 1 : 0);   // one lane moved
    const uint32_t want = passes_def(a, mask, bytes);
    VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, mask, 32, bytes), want);
    ++checked;
  }
  VCHECK(checked == 400000);
}

VTEST(a_warp_that_returns_in_another_place_gets_the_answer_of_its_own_banks) {
  // The same shape at every alignment, moved word by word through the banks and byte by byte through the words: the
  // passes follow the definition each time, with one table throughout.
  Rng r(77);
  mc::BankMemo memo;
  for (unsigned kind = 0; kind < 7; ++kind) {
    for (int rep = 0; rep < 6; ++rep) {
      const Shape sh = make_shape(r, kind);
      for (uint32_t bytes : {1u, 2u, 4u, 8u, 16u}) {
        for (uint64_t base = 4096; base < 4096 + 4 * 70; base += (base & 3) ? 1 : 3) {
          uint64_t a[32];
          for (unsigned l = 0; l < 32; ++l) a[l] = base + sh.off[l];
          VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, bytes), passes_def(a, 0xFFFFFFFFu, bytes));
        }
      }
    }
  }
}

VTEST(shapes_that_differ_in_one_lane_one_byte_or_the_mask_are_counted_for_what_they_are) {
  mc::BankMemo memo;
  uint64_t a[32];
  // Two rows of 16 words, 160 words apart: lanes i and i + 16 share a bank on different words (2 passes).
  for (unsigned l = 0; l < 32; ++l) a[l] = 4096 + 4 * (l % 16) + 4 * 160 * (l / 16);
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);   // (a hit)
  // The second row moved onto the other half of the banks: no conflict now, not the remembered 2.
  for (unsigned l = 16; l < 32; ++l) a[l] += 4 * 16;
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), passes_def(a, 0xFFFFFFFFu, 4));
  // One lane of the second row on the first row's word: still 2 (the others) -- and with that lane alone it is 1.
  for (unsigned l = 0; l < 32; ++l) a[l] = 4096 + 4 * (l % 16) + 4 * 160 * (l / 16);
  a[31] = a[15];
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);
  const uint32_t pair = 0x80008000u;   // lanes 15 and 31: the same word, one pass
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, pair, 32, 4), 1u);
  // The same shape with the mask of lanes 15 and 16 (different words, the same bank).
  for (unsigned l = 0; l < 32; ++l) a[l] = 4096 + 4 * (l % 16) + 4 * 160 * (l / 16);
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0x00018000u, 32, 4), 1u);   // (lane 15 and 16: banks 15 and 0)
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0x00010001u, 32, 4), 2u);   // (lanes 0 and 16: the same bank)
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);
  // A byte offset: the lanes' words are the same, but a 4-byte access that is not aligned covers two words.
  for (unsigned l = 0; l < 32; ++l) a[l] = 4096 + 1 + 4 * (l % 16) + 4 * 160 * (l / 16);
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), passes_def(a, 0xFFFFFFFFu, 4));
  // The addresses at and past 2^31 are counted, not remembered.
  for (unsigned l = 0; l < 32; ++l) a[l] = (uint64_t{1} << 31) + 4 * (l % 16) + 4 * 160 * (l / 16);
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);
  for (unsigned l = 0; l < 32; ++l) a[l] = (uint64_t{1} << 31) - 4 * 160 * 2 + 4 * (l % 16) + 4 * 160 * (l / 16);
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);
}

VTEST(shapes_on_a_stride_are_counted_by_the_closed_forms_and_not_remembered) {
  mc::BankMemo memo;
  uint64_t a[32];
  for (uint64_t stride : {4, 8, 16, 128, 132, 36, 1}) {
    for (unsigned l = 0; l < 32; ++l) a[l] = 4096 + stride * l;
    VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), passes_def(a, 0xFFFFFFFFu, 4));
    VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0x0FF0FF00u, 32, 4), passes_def(a, 0x0FF0FF00u, 4));   // (a gap: not a stride)
  }
  // the six full warps above never reached the table; the gapped ones did (twice for stride 4 and 8 and so on)
  VCHECK(memo.lookups <= 7);
  mc::BankMemo clean;
  for (unsigned l = 0; l < 32; ++l) a[l] = 4096 + 128 * l;
  for (int i = 0; i < 50; ++i) VCHECK_EQ(mc::sass_bank_passes_memo(clean, a, 0xFFFFFFFFu, 32, 4), 32u);
  VCHECK_EQ(clean.lookups, 0u);
}

VTEST(a_run_of_misses_backs_off_and_the_answers_stay_right) {
  // A gather that never repeats: after kMissRun misses the table is left alone for kSkip calls and then tried again.
  Rng r(5);
  mc::BankMemo memo;
  bool skipped = false;
  for (int i = 0; i < 6000; ++i) {
    uint64_t a[32];
    for (unsigned l = 0; l < 32; ++l) a[l] = 4 * r.below(1 << 12);
    VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), passes_def(a, 0xFFFFFFFFu, 4));
    skipped = skipped || memo.skip > 0;
  }
  VCHECK(skipped);
  // ... and shapes that repeat are found again after a back-off.
  uint64_t a[32];
  for (unsigned l = 0; l < 32; ++l) a[l] = 4096 + 4 * (l % 16) + 4 * 160 * (l / 16);
  for (int i = 0; i < 2000; ++i) VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);
  VCHECK(memo.skip == 0);
}

VTEST(the_counting_behind_the_table_is_the_earlier_counting) {
  // sass_bank_passes_table (a counted walk of the lanes, one word each for 1 to 4 bytes, a table with byte counts)
  // against sass_bank_passes_reference, the vectors-of-words counting it replaced: every width from 0 to 64 bytes (a
  // 0-byte access covers a word only at an unaligned address), every byte in the word, full, low-run, partial and sparse
  // masks, lanes on the same word, in the same bank, spread.
  Rng r(913);
  uint64_t checked = 0;
  for (int i = 0; i < 300000; ++i) {
    uint64_t a[32];
    const unsigned kind = static_cast<unsigned>(r.below(4));
    const uint64_t base = r.below(1 << 20);
    for (unsigned l = 0; l < 32; ++l) {
      switch (kind) {
        case 0: a[l] = base + r.below(64); break;                      // a few words, every byte
        case 1: a[l] = base + 128 * r.below(6) + 4 * r.below(3); break;   // banks of few words, many rows
        case 2: a[l] = base + r.below(1 << 14); break;
        default: a[l] = (base & ~3ull) + 4 * (l / (1 + r.below(3))) + (r.below(4) == 0 ? r.below(4) : 0); break;
      }
    }
    uint32_t mask = 0xFFFFFFFFu;
    switch (r.below(5)) {
      case 0: mask = static_cast<uint32_t>(r.next()); break;
      case 1: mask = 0xFFFFu << r.below(16); break;
      case 2: mask = static_cast<uint32_t>(r.next() & r.next() & r.next()); break;
      default: break;
    }
    const uint32_t bytes = r.below(3) == 0 ? static_cast<uint32_t>(r.below(65)) : static_cast<uint32_t>(r.below(5));
    VCHECK_EQ(mc::sass_bank_passes_table(a, mask, 32, bytes), mc::sass_bank_passes_reference(a, mask, 32, bytes));
    VCHECK_EQ(mc::sass_bank_passes(a, mask, 32, bytes), mc::sass_bank_passes_reference(a, mask, 32, bytes));
    ++checked;
  }
  VCHECK(checked == 300000);
}

VTEST(a_shape_that_returns_is_found_wherever_it_sits_whatever_the_idle_lanes_hold) {
  // The table is only worth having if the shapes are found again: moved by whole words, moved far, and with the lanes
  // that do not run holding anything at all. (hits counts the lookups of this window that were found.)
  uint64_t a[32];
  const auto rows = [&](uint64_t base) {
    for (unsigned l = 0; l < 32; ++l) a[l] = base + 4 * (l % 16) + 4 * 160 * (l / 16);
  };
  mc::BankMemo memo;
  rows(4096);
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);
  VCHECK_EQ(memo.hits, 0u);
  rows(4096 + 4 * 37);
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);
  rows((uint64_t{5} << 24) + 4 * 1000);
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);
  VCHECK_EQ(memo.hits, 2u);
  // Lanes 3 and 20 idle, holding zero, and then holding anything: the same shape each time.
  const uint32_t mask = ~((1u << 3) | (1u << 20));
  mc::BankMemo memo2;
  rows(8192);
  a[3] = 0; a[20] = 0;
  VCHECK_EQ(mc::sass_bank_passes_memo(memo2, a, mask, 32, 4), 2u);
  rows(8192 + 4 * 64);
  a[3] = 0x123456789ull; a[20] = ~uint64_t{0};
  VCHECK_EQ(mc::sass_bank_passes_memo(memo2, a, mask, 32, 4), 2u);
  rows(4 * 5);
  a[3] = 77; a[20] = 1;
  VCHECK_EQ(mc::sass_bank_passes_memo(memo2, a, mask, 32, 4), 2u);
  VCHECK_EQ(memo2.hits, 2u);
  // ... a lane that runs is part of the shape, though.
  rows(8192);
  VCHECK_EQ(mc::sass_bank_passes_memo(memo2, a, 0xFFFFFFFFu, 32, 4), 2u);
  VCHECK_EQ(memo2.hits, 2u);
}

VTEST(a_shape_remembered_low_is_not_answered_for_addresses_at_the_top_of_the_space) {
  // One lane in a bank of its own, one on a word of lane 0's bank: 2 passes. At the top of the address space the last
  // lane's bytes wrap, and the counting (the reference, which reads no word for a range that wraps) finds 1.
  uint64_t a[32];
  mc::BankMemo memo;
  for (unsigned l = 0; l < 31; ++l) a[l] = 4096 + 4 * l;
  a[31] = 4096 + 4 * 32 * 3;
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);
  const uint64_t top = ~uint64_t{0} - 3;   // lane 31's word is the last one: its 4 bytes end past the top
  for (unsigned l = 0; l < 31; ++l) a[l] = top - 4 * 32 * 3 + 4 * l;
  a[31] = top;
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), mc::sass_bank_passes_reference(a, 0xFFFFFFFFu, 32, 4));
  // and shifted so that the lanes below the first one's word wrap round the other way
  for (unsigned l = 0; l < 32; ++l) a[l] = 4096 + 4 * (l % 16) + 4 * 160 * (l / 16);
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), 2u);
  for (unsigned l = 0; l < 32; ++l) a[l] = 64 + 4 * (l % 16) + 4 * 160 * (l / 16);
  std::swap(a[0], a[16]);   // the first lane is now the high row: the others lie below it
  VCHECK_EQ(mc::sass_bank_passes_memo(memo, a, 0xFFFFFFFFu, 32, 4), passes_def(a, 0xFFFFFFFFu, 4));
}

VTEST_MAIN
