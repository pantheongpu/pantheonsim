// The closed forms in vgpu/exec/mem_counters.hpp against the definition of what they count.
//
// Both engines count, for every global, local and shared access of a warp, the 32-byte sectors its lanes touch and
// the passes shared memory's 32 banks need. Those counters are reported to users (vgpu counters, the CUPTI
// metrics), so the faster ways of getting them -- the ascending / descending / window sector counts, the
// arithmetic-progression bank count, the run count the executors use for a coalesced access -- must give exactly
// the numbers of the counting they replace. The oracles below are written from the definition (sorted vectors of
// the sectors and the words) and share no code with the header (test_memfast.cpp holds the earlier code as it stood and compares
// the header's reference functions with it; these compare the fast entry points with the definition).
//
// What is covered: every stride from -70 to 130 bytes and a spread of large ones (multiples of 4 and not,
// negative, and so large the lanes' addresses leave any window), every access width the engines make, base
// addresses at every alignment that matters (the byte in the word, the word in the bank row, the byte in the
// sector), masks that are full, a low run, a run starting above lane 0, a run with a gap, sparse and random, warps of
// 32 and 64 lanes, and warps of the same shapes with one lane moved by a word, a byte or a sector (a closed form
// that stops checking every lane fails on those). Addresses at the top of the space, where adding a width wraps,
// are the general code's, and are compared too.
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "vgpu/exec/mem_counters.hpp"
#include "vgpu/exec/window_check.hpp"
#include "vgpu/simd/simd.hpp"
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

// ---- the definitions ----------------------------------------------------------------

uint64_t distinct_count(std::vector<uint64_t>& v) {
  std::sort(v.begin(), v.end());
  return static_cast<uint64_t>(std::unique(v.begin(), v.end()) - v.begin());
}

// The distinct 32-byte sectors the active lanes' accesses of `span` bytes touch. The interpreter keeps the first
// `cap` it meets and counts no more, so its answer saturates at cap; the SASS executor counts all of them (cap =
// SIZE_MAX).
uint64_t sectors_def(const std::vector<uint64_t>& a, uint64_t mask, unsigned lanes, uint64_t span, size_t cap) {
  std::vector<uint64_t> seen;
  for (unsigned l = 0; l < lanes; ++l) {
    if (!(mask >> l & 1)) continue;
    const uint64_t first = a[l] / 32, last = (a[l] + (span ? span - 1 : 0)) / 32;
    if (last < first) continue;   // (the address plus the width wrapped)
    for (uint64_t s = first; s <= last; ++s) seen.push_back(s);
  }
  return std::min<uint64_t>(distinct_count(seen), cap);
}

// The most distinct words any one bank is asked for, over the words each active lane covers (`words`: the first
// word and how many).
template <class Words>
uint64_t passes_def(const std::vector<uint64_t>& a, uint64_t mask, unsigned lanes, Words words) {
  std::vector<uint64_t> banks[32];
  for (unsigned l = 0; l < lanes; ++l) {
    if (!(mask >> l & 1)) continue;
    words(a[l], [&](uint64_t w) { banks[w % 32].push_back(w); });
  }
  uint64_t worst = 1;
  for (auto& b : banks) worst = std::max<uint64_t>(worst, distinct_count(b));
  return worst;
}

// The SASS executor's: the words the bytes cover.
uint64_t sass_passes_def(const std::vector<uint64_t>& a, uint64_t mask, unsigned lanes, uint32_t bytes) {
  return passes_def(a, mask, lanes, [&](uint64_t addr, auto&& add) {
    for (uint64_t w = addr >> 2; w <= (addr + bytes - 1) >> 2; ++w) add(w);
  });
}

// The interpreter's: the words (address + 4k) / 4 for 4k < span.
uint64_t ptx_passes_def(const std::vector<uint64_t>& a, uint64_t mask, unsigned lanes, uint64_t span) {
  return passes_def(a, mask, lanes, [&](uint64_t addr, auto&& add) {
    for (uint64_t off = 0; off < span; off += 4) add((addr + off) / 4);
  });
}

// ---- one comparison -----------------------------------------------------------------

uint64_t g_checked = 0;

std::string describe(const std::vector<uint64_t>& a, uint64_t mask, unsigned lanes, uint64_t span) {
  std::string s = "lanes " + std::to_string(lanes) + " span " + std::to_string(span) + " mask " + std::to_string(mask) + " addrs";
  unsigned shown = 0;
  for (unsigned l = 0; l < lanes && shown < 10; ++l)
    if (mask >> l & 1) {
      s += " " + std::to_string(a[l]);
      ++shown;
    }
  return s;
}

// Every entry point the engines use, against the definitions. `top` is whether an address is within 256 bytes of
// the top of the space, where the SASS executor's bank count is not defined (it never ends) and is not asked.
void check(const std::vector<uint64_t>& a, uint64_t mask, unsigned lanes, uint64_t span) {
  ++g_checked;
  const std::string where = describe(a, mask, lanes, span);
  // The interpreter's: any warp up to 64 lanes, saturating at 512 sectors.
  const uint64_t ps = sectors_def(a, mask, lanes, span, 512);
  if (mc::distinct_sectors(a.data(), mask, lanes, span, 512) != ps)
    throw vtest::Failure("ptx sectors " + std::to_string(mc::distinct_sectors(a.data(), mask, lanes, span, 512)) + " vs " +
                         std::to_string(ps) + ": " + where);
  // (Past 128 bytes an access is counted by the earlier code, which remembers at most 256 words a bank; the definition
  // has no such limit, so it is compared up to there.)
  if (span <= 128) {
    const uint64_t pb = ptx_passes_def(a, mask, lanes, span);
    if (mc::ptx_bank_passes(a.data(), mask, lanes, span) != pb)
      throw vtest::Failure("ptx banks " + std::to_string(mc::ptx_bank_passes(a.data(), mask, lanes, span)) + " vs " +
                           std::to_string(pb) + ": " + where);
  }
  // The SASS executor's: 32 lanes, 1 to 64 bytes (counted as exec_mem hands them over).
  if (lanes == 32 && span >= 1 && span <= 64) {
    const uint32_t bytes = static_cast<uint32_t>(span), m32 = static_cast<uint32_t>(mask);
    const uint64_t ss = sectors_def(a, m32, 32, span, SIZE_MAX);
    if (mc::distinct_sectors(a.data(), m32, 32, bytes, SIZE_MAX) != ss)
      throw vtest::Failure("sass sectors " + std::to_string(mc::distinct_sectors(a.data(), m32, 32, bytes, SIZE_MAX)) + " vs " +
                           std::to_string(ss) + ": " + where);
    if (mc::distinct_sectors_uncapped(a.data(), m32, 32, bytes) != ss)   // (the SASS executor's own entry: the closed forms, then the list)
      throw vtest::Failure("sass sectors (uncapped) " + std::to_string(mc::distinct_sectors_uncapped(a.data(), m32, 32, bytes)) + " vs " +
                           std::to_string(ss) + ": " + where);
    bool sane = true;
    for (unsigned l = 0; l < 32; ++l) sane = sane && (!(m32 >> l & 1) || a[l] <= ~uint64_t{0} - 256);
    if (sane && m32) {
      const uint64_t sb = sass_passes_def(a, m32, 32, bytes);
      if (mc::sass_bank_passes(a.data(), m32, 32, bytes) != sb)
        throw vtest::Failure("sass banks " + std::to_string(mc::sass_bank_passes(a.data(), m32, 32, bytes)) + " vs " +
                             std::to_string(sb) + ": " + where);
    }
  }
}

std::vector<uint64_t> progression(uint64_t base, int64_t stride, unsigned lanes = 64) {
  std::vector<uint64_t> a(64, 0);
  for (unsigned l = 0; l < lanes; ++l) a[l] = base + static_cast<uint64_t>(stride) * l;
  return a;
}

std::vector<uint64_t> masks_for(unsigned lanes, Rng& rng) {
  const uint64_t all = lanes >= 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1;
  std::vector<uint64_t> m = {all, 1, all >> 1, all >> 5, 0xFF, 0xFF00 & all, (all >> 3) << 3, 0x5555555555555555ull & all,
                             0xAAAAAAAAAAAAAAAAull & all, (uint64_t{1} << (lanes - 1)), 0x1Full << 7, 0x7 | (0x7 << 9)};
  for (int i = 0; i < 3; ++i) m.push_back(rng.next() & all);
  return m;
}

}  // namespace

// Every small stride, width and alignment, against the definition.
VTEST(closed_forms_small_patterns) {
  Rng rng(1);
  const uint64_t bases[] = {0, 1, 2, 3, 4, 30, 31, 124, 127, 0x7f0012345678ull};
  const uint64_t spans[] = {0, 1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 33, 64, 100, 128};
  for (unsigned lanes : {32u, 64u})
    for (int64_t stride = -70; stride <= 130; ++stride)
      for (uint64_t base : bases) {
        const std::vector<uint64_t> a = progression(base + 4096, stride, lanes);
        for (uint64_t span : spans)
          for (uint64_t mask : masks_for(lanes, rng)) check(a, mask, lanes, span);
      }
  VCHECK(g_checked > 100000);
}

// Strides too large for a window, multiples of 4 and not, either direction.
VTEST(closed_forms_large_strides) {
  Rng rng(2);
  const int64_t strides[] = {255,     256,        257,       512,       1000,     1024,      4096,       4100,       -4096,
                             65536,   1 << 20,    -(1 << 20), (1 << 20) + 4, 1ll << 31, (1ll << 32) - 4, 1ll << 32, (1ll << 32) + 4, -(1ll << 32),
                             1ll << 33, 1ll << 40, -(1ll << 40), 1ll << 62, int64_t(1) << 63};
  const uint64_t spans[] = {1, 4, 8, 16, 32, 64, 128};
  for (unsigned lanes : {32u, 64u})
    for (int64_t stride : strides)
      for (uint64_t base : {uint64_t{0x1000}, uint64_t{0x7f0000000003ull}, uint64_t{1} << 62}) {
        const std::vector<uint64_t> a = progression(base, stride, lanes);
        for (uint64_t span : spans)
          for (uint64_t mask : masks_for(lanes, rng)) check(a, mask, lanes, span);
      }
}

// Random shapes: progressions with a lane moved, two segments (a tile whose rows are 8 or 16 lanes wide),
// broadcasts, thread-private windows, and noise.
VTEST(closed_forms_random_warps) {
  Rng rng(3);
  const uint64_t spans[] = {1, 2, 4, 4, 4, 8, 8, 16, 16, 12, 32, 64, 128};
  for (int iter = 0; iter < 120000; ++iter) {
    const unsigned lanes = rng.below(6) == 0 ? 64 : 32;
    const uint64_t all = lanes == 64 ? ~uint64_t{0} : 0xFFFFFFFFull;
    std::vector<uint64_t> a(64, 0);
    const uint64_t base = rng.below(8) == 0 ? ~uint64_t{0} - rng.below(2048) : (rng.below(4) ? rng.below(1ull << 20) : rng.next() >> 16);
    const uint64_t strides[] = {0, 1, 2, 4, 4, 8, 8, 12, 16, 16, 32, 64, 128, 132, 256, 512, 1024, 4, 4};
    uint64_t stride = strides[rng.below(sizeof strides / sizeof strides[0])];
    if (rng.below(4) == 0) stride = 0 - stride;
    if (rng.below(8) == 0) stride = rng.below(300);
    const uint64_t span = spans[rng.below(sizeof spans / sizeof spans[0])];
    switch (rng.below(8)) {
      case 0: case 1: case 2:   // a progression
        for (unsigned l = 0; l < 64; ++l) a[l] = base + stride * l;
        break;
      case 3:   // a progression with one lane moved a little
        for (unsigned l = 0; l < 64; ++l) a[l] = base + stride * l;
        {
          const uint64_t deltas[] = {1, 2, 3, 4, 8, 32, 128, 0 - 1ull, 0 - 4ull, 0 - 32ull, 0 - 128ull};
          a[rng.below(lanes)] += deltas[rng.below(sizeof deltas / sizeof deltas[0])];
        }
        break;
      case 4: {   // two or four segments: rows of a tile
        const unsigned width = 1u << (1 + rng.below(4));
        const uint64_t row = 4 * (width + rng.below(40));
        for (unsigned l = 0; l < 64; ++l) a[l] = base + (l / width) * row + (l % width) * (stride ? stride : 4);
        break;
      }
      case 5:   // a broadcast, or a few distinct addresses
        for (unsigned l = 0; l < 64; ++l) a[l] = base + stride * (l % (1 + rng.below(4)));
        break;
      case 6: {   // a thread-private array: words in a small window, no order
        const uint64_t win = 1 + rng.below(rng.below(2) ? 64 : 4000);
        for (unsigned l = 0; l < 64; ++l) a[l] = (base & ~3ull) + 4 * rng.below(win);
        break;
      }
      default:   // noise
        for (unsigned l = 0; l < 64; ++l) a[l] = rng.below(2) ? rng.next() >> rng.below(40) : base + rng.below(1 << 14);
        break;
    }
    uint64_t mask = all;
    switch (rng.below(6)) {
      case 0: mask = all; break;
      case 1: mask = all >> rng.below(lanes); break;                       // a low run
      case 2: mask = (all >> rng.below(lanes)) << rng.below(lanes / 2); mask &= all; break;   // a run above lane 0 (or two)
      case 3: mask = rng.next() & all; break;
      case 4: mask = rng.next() & rng.next() & all; break;
      default: mask = 1ull << rng.below(lanes); break;
    }
    if (mask == 0) mask = 1;
    check(a, mask, lanes, span);
  }
}

// Near the top of the address space, where adding a width wraps or the words of an access cross zero: the closed
// forms hand all of it to the general counting, and the answers still agree with the definition.
VTEST(closed_forms_top_of_address_space) {
  Rng rng(4);
  std::vector<uint64_t> backs = {1, 3, 4, 16, 31, 32, 40, 255, 256, 257, 300, 1000};
  for (uint64_t b = 5; b < 140; b += 7) backs.push_back(b);
  for (uint64_t back : backs)
    for (int64_t stride : {-64, -32, -16, -8, -4, 0, 1, 4, 8, 16, 100, 128})
      for (uint64_t span : {1ull, 4ull, 8ull, 12ull, 16ull, 32ull, 64ull, 128ull}) {
        const std::vector<uint64_t> a = progression(~uint64_t{0} - back, stride, 32);
        for (uint64_t mask : masks_for(32, rng)) {
          // (the interpreter's sector count and the SASS executor's agree with the definition even when it wraps;
          // the bank counts of such addresses are not defined for the SASS executor and check() skips them)
          check(a, mask, 32, span);
        }
      }
}

// A few lanes, in no order, near each other and straddling sectors: the thread-private and gather shapes the window and
// hash counts take, where one lane's second sector may be the only lane to touch it.
VTEST(closed_forms_few_lanes_scattered) {
  Rng rng(6);
  for (int iter = 0; iter < 150000; ++iter) {
    const unsigned lanes = rng.below(8) == 0 ? 64 : 32;
    std::vector<uint64_t> a(64, 0);
    const uint64_t base = rng.below(2) ? 0x7f0000001000ull + rng.below(32) : rng.below(1ull << 12);
    const uint64_t reach = uint64_t{1} << (4 + rng.below(12));   // 16 bytes to 32 KB around the base
    for (unsigned l = 0; l < 64; ++l) a[l] = base + rng.below(reach) - (rng.below(2) ? reach / 4 : 0);
    if (rng.below(2)) for (unsigned l = 0; l < 64; ++l) a[l] &= ~uint64_t{3};
    uint64_t mask = 0;
    const unsigned active = 1 + static_cast<unsigned>(rng.below(rng.below(2) ? 6 : lanes));
    for (unsigned k = 0; k < active; ++k) mask |= uint64_t{1} << rng.below(lanes);
    const uint64_t spans[] = {1, 2, 4, 8, 12, 16, 24, 31, 32, 33, 64};
    check(a, mask, lanes, spans[rng.below(sizeof spans / sizeof spans[0])]);
  }
}

// The thread-private array a kernel indexes by a value of its own -- words (or wider accesses) in a few sectors in no
// order, partial masks, a window that starts below the first lane's, and one that runs out of it -- counted with the AVX2
// window kernel and without it (a host without AVX2, VGPU_SIMD=base: the kernel is then absent and both runs are the
// scalar forms), each against the definition.
VTEST(closed_forms_private_arrays_with_and_without_the_simd_window) {
  for (int pass = 0; pass < 2; ++pass) {
    vgpu::simd::testing::use_sector_window_kernel(pass == 0);
    Rng rng(11);
    for (int iter = 0; iter < 80000; ++iter) {
      const unsigned lanes = rng.below(5) == 0 ? 64 : 32;
      const uint64_t all = lanes == 64 ? ~uint64_t{0} : 0xFFFFFFFFull;
      std::vector<uint64_t> a(64, 0);
      const uint64_t bases[] = {0, 4, 28, 31, 60, 1000, 0x7f0000001000ull + 28, (uint64_t{1} << 62) - 300, (uint64_t{1} << 62) - 40,
                                (uint64_t{1} << 62), ~uint64_t{0} - 200, rng.below(1ull << 20)};
      const uint64_t base = bases[rng.below(sizeof bases / sizeof bases[0])];
      const unsigned words = 1 + static_cast<unsigned>(rng.below(rng.below(2) ? 64 : 520));   // the window, in words
      const uint64_t first_back = rng.below(3) ? 0 : 4 * rng.below(words);                    // the first lane sits above the window's start
      const unsigned wrap = 1 + static_cast<unsigned>(rng.below(64));
      const unsigned how = static_cast<unsigned>(rng.below(3));
      for (unsigned l = 0; l < 64; ++l) {
        const uint64_t w = how == 0 ? rng.below(words) : how == 1 ? (l * 5 + 3) % wrap : (rng.below(wrap) + l) % words;
        a[l] = base + 4 * w;
      }
      a[0] = a[0] + first_back;
      if (rng.below(3) == 0) for (unsigned l = 0; l < 64; ++l) a[l] += rng.below(4);   // unaligned
      uint64_t mask = all;
      switch (rng.below(5)) {
        case 0: break;
        case 1: mask = rng.next() & all; break;
        case 2: mask = (all >> rng.below(lanes)) << rng.below(lanes / 2); mask &= all; break;
        case 3: mask = all >> rng.below(lanes); break;
        default: mask = (uint64_t{1} << rng.below(lanes)) | (uint64_t{1} << rng.below(lanes)); break;
      }
      if (mask == 0) mask = 1;
      const uint64_t spans[] = {1, 2, 4, 4, 4, 8, 16, 12, 24, 31, 32};
      check(a, mask, lanes, spans[rng.below(sizeof spans / sizeof spans[0])]);
    }
  }
  vgpu::simd::testing::use_sector_window_kernel(true);   // (as the library left it)
}

// 64 lanes of 8 bytes, each straddling two sectors of its own, in no order: 128 sectors, which a 128-slot table could not
// hold (the hash count is only for up to 32 lanes), and wide accesses that the closed forms are not made for (their
// answers are the general counting's, and agree with the definition).
VTEST(closed_forms_many_sectors_and_wide_accesses) {
  Rng rng(7);
  std::vector<uint64_t> a(64);
  for (unsigned l = 0; l < 64; ++l) a[l] = 0x7f0000000000ull + ((l * 37) % 64) * 1024 + 28;
  check(a, ~uint64_t{0}, 64, 8);
  check(a, 0xFFFFFFFFull, 32, 8);
  for (unsigned l = 0; l < 64; ++l) a[l] = 0x7f0000000000ull + (63 - l) * 1024 + 28;
  check(a, ~uint64_t{0}, 64, 8);
  for (uint64_t span : {129ull, 200ull, 4096ull, 4097ull, 5000ull})
    for (int iter = 0; iter < 60; ++iter) {
      const unsigned lanes = rng.below(2) ? 64 : 32;
      const uint64_t base = 0x7f0000000000ull + rng.below(4096);
      const uint64_t stride = (iter % 4 == 0) ? span : (iter % 4 == 1) ? 0 : (iter % 4 == 2) ? 4 * rng.below(2000) : rng.below(1ull << 20);
      for (unsigned l = 0; l < 64; ++l) a[l] = base + stride * l;
      uint64_t mask = lanes == 64 ? ~uint64_t{0} : 0xFFFFFFFFull;
      if (iter % 3 == 0) mask &= rng.next();
      if (mask == 0) mask = 1;
      check(a, mask, lanes, span);
    }
}

// The run the executors count in one step: lanes on consecutive elements are the sectors from the first byte's to
// the last's, whatever the width and alignment.
VTEST(run_sectors_equal_the_lanes_addresses) {
  Rng rng(5);
  uint64_t n = 0;
  for (uint32_t bytes : {1u, 2u, 4u, 8u}) {
    for (uint64_t g0 : {uint64_t{0}, uint64_t{1}, uint64_t{3}, uint64_t{4}, uint64_t{31}, uint64_t{32}, uint64_t{33}, uint64_t{4093},
                        uint64_t{0x7f0012345678ull}, uint64_t{0x7f0012345600ull}, ~uint64_t{0} - 5000}) {
      for (unsigned first = 0; first < 32; first += 1 + rng.below(5)) {
        for (unsigned count = 1; first + count <= 32; count += 1 + rng.below(6)) {
          // (only runs an element-aligned address starts, as the executors' coalesced path makes them)
          const uint64_t start = g0 & ~uint64_t{bytes - 1};
          std::vector<uint64_t> a(32, 0);
          uint64_t mask = 0;
          for (unsigned i = 0; i < count; ++i) {
            a[first + i] = start + uint64_t{i} * bytes;
            mask |= uint64_t{1} << (first + i);
          }
          uint64_t got;
          const bool ok = mc::run_sectors(start, uint64_t{count} * bytes, got);
          const uint64_t want = sectors_def(a, mask, 32, bytes, SIZE_MAX);
          if (!ok) throw vtest::Failure("run_sectors declined a run that does not wrap: " + describe(a, mask, 32, bytes));
          if (got != want)
            throw vtest::Failure("run_sectors " + std::to_string(got) + " vs " + std::to_string(want) + ": " + describe(a, mask, 32, bytes));
          ++n;
        }
      }
    }
  }
  VCHECK(n > 1000);
  // Not a run: empty, or one that runs off the end of the address space.
  uint64_t out;
  VCHECK(!mc::run_sectors(0x1000, 0, out));
  VCHECK(!mc::run_sectors(~uint64_t{0} - 3, 8, out));
  VCHECK(mc::run_sectors(~uint64_t{0} - 3, 4, out));
  VCHECK_EQ(out, uint64_t{1});
}

// The SASS executor's shared- and local-memory fast path takes the window checks of vgpu/exec/window_check.hpp: they must
// answer exactly what testing each lane in turn answers (the earlier code, copied here), for lanes in this block's window and
// in others', inside the array and past its end, with one lane off or every lane, and with no lane at all.
VTEST(window_checks_equal_the_per_lane_tests) {
  Rng rng(8);
  uint64_t inside_count = 0, outside_count = 0;
  for (int iter = 0; iter < 400000; ++iter) {
    const uint32_t mask = rng.below(8) == 0 ? 0u : (rng.below(2) ? 0xFFFFFFFFu : static_cast<uint32_t>(rng.next()));
    const uint64_t rank = rng.below(4), size = 1 + rng.below(1 << (4 + rng.below(18))), frame = 1 + rng.below(1 << (4 + rng.below(12)));
    const uint32_t bytes = 1u << rng.below(6);
    uint64_t offs[32];
    for (unsigned l = 0; l < 32; ++l) {
      const uint64_t in_array = rng.below(size), near_end = size - std::min<uint64_t>(size, rng.below(40));
      uint64_t off = rng.below(3) == 0 ? near_end : in_array;
      if (rng.below(25) == 0) off |= (rank ^ (1 + rng.below(3))) << 24;   // another block's window
      else off |= rank << 24;
      offs[l] = off & 0xFFFFFFFFull;
    }
    // Shared: the earlier loop.
    bool want = true;
    for (unsigned l = 0; l < 32; ++l)
      if (mask >> l & 1) want = want && (offs[l] >> 24) == rank && (offs[l] & 0xffffff) + bytes <= size;
    vgpu::exec::SharedWindowCheck sc;
    for (unsigned l = 0; l < 32; ++l)
      if (mask >> l & 1) sc.add(offs[l], rank);
    if (sc.inside(size, bytes, mask != 0) != want) throw vtest::Failure("shared window check differs from the per-lane test at iteration " + std::to_string(iter));
    if (mask == 0xFFFFFFFFu) {   // a whole warp: the vectored form of the same test
      uint32_t w32[32];
      for (unsigned l = 0; l < 32; ++l) w32[l] = static_cast<uint32_t>(offs[l]);
      vgpu::exec::SharedWindowCheck wc;
      wc.add_warp(w32, static_cast<uint32_t>(rank));
      if (wc.inside(size, bytes, true) != want) throw vtest::Failure("whole-warp shared window check differs from the per-lane test at iteration " + std::to_string(iter));
    }
    // Local: the earlier loop, over frame offsets.
    bool want_local = true;
    uint64_t loc[32];
    for (unsigned l = 0; l < 32; ++l) {
      loc[l] = rng.below(3) == 0 ? frame - std::min<uint64_t>(frame, rng.below(10)) : rng.below(frame);
      if (mask >> l & 1) want_local = want_local && loc[l] + bytes <= frame;
    }
    vgpu::exec::LocalWindowCheck lc;
    for (unsigned l = 0; l < 32; ++l)
      if (mask >> l & 1) lc.add(loc[l]);
    if (lc.inside(frame, bytes, mask != 0) != want_local) throw vtest::Failure("local window check differs from the per-lane test at iteration " + std::to_string(iter));
    (want ? inside_count : outside_count) += 1;
  }
  VCHECK(inside_count > 20000 && outside_count > 20000);   // both answers were exercised
}

VTEST_MAIN
