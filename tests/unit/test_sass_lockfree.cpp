// A warp's SASS atomics on plain device memory without a lock (sassatom::global_lockfree, over MemoryManager::rmw_batch with lock_free):
// each distinct word gets a compare-and-swap loop whose step runs the word's lanes in lane order.
//  - against the lane-by-lane rule (sassatom::apply, itself held against the frozen executor lambda in test_sass_atomics): every
//    operation and type, compare-and-swap, partial masks, lanes that share a word, words in chunks never written, uniform and real:
//    the old value each lane gets, the words left, whether anything changed;
//  - declined, with nothing touched, for a misaligned or out-of-allocation word;
//  - from several threads, where no update may be lost: a fetch-add hands out each old value once, and the commutative operations
//    (add, and, or, xor, min, max) leave the serial result at any thread count.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "vgpu/exec/host_atomic.hpp"
#include "vgpu/exec/sass_atomics.hpp"
#include "vgpu/memory.hpp"
#include "vtest.hpp"

namespace {

namespace sa = vgpu::exec::sassatom;
using namespace vgpu;

uint64_t operand(std::mt19937_64& rng, unsigned type) {
  static const uint32_t f32s[] = {0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0x7fc00000, 0x00400000, 0x00c00000, 0x40490fdb};
  static const uint64_t e[] = {0, 1, ~uint64_t{0}, 0x8000000000000000ull, 0x7fffffffffffffffull, 0xffffffffull, 0x100000000ull};
  const unsigned pick = static_cast<unsigned>(rng() % 8);
  if (type == 3) return pick < 5 ? f32s[rng() % 9] : (rng() & 0xffffffffu);
  if (type == 2 || type == 5 || type == 6) return pick < 4 ? e[rng() % 7] : rng();
  static const uint32_t w[] = {0, 1, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0xfffffffeu, 0x7fff, 0x8000, 0xffff, 0x10000};
  return pick < 5 ? w[rng() % 10] : (rng() & 0xffffffffu);
}

struct Warp {
  uint64_t uaddr[32];
  uint32_t lane_mask[32];
  unsigned slot[32];
  unsigned nu = 0;
};

// What exec_ops.inc does to group a warp's addresses: the distinct words in the order the lanes first name them, and each word's lanes.
Warp group(const uint64_t* lane_addr, uint32_t ex) {
  Warp w;
  std::memset(w.lane_mask, 0, sizeof w.lane_mask);
  for (uint32_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctz(m));
    unsigned k = 0;
    while (k < w.nu && w.uaddr[k] != lane_addr[l]) ++k;
    if (k == w.nu) w.uaddr[w.nu++] = lane_addr[l];
    w.slot[l] = k;
    w.lane_mask[k] |= uint32_t{1} << l;
  }
  return w;
}

}  // namespace

VTEST(a_warps_lock_free_atomics_are_the_lane_by_lane_ones) {
  std::mt19937_64 rng(11);
  size_t done = 0;
  for (unsigned type = 0; type < 8; ++type)
    for (unsigned op = 0; op < 9; ++op)
      for (int cas = 0; cas < 2; ++cas)
        for (int it = 0; it < 60; ++it) {
          const bool wide = type == 2 || type == 5 || type == 6;
          const uint32_t n = wide ? 8 : 4;
          const uint64_t keep = wide ? ~uint64_t{0} : 0xffffffffull;
          MemoryManager mm(1ull << 28);
          const uint64_t p = mm.alloc(3 * kChunkSize);
          const uint8_t pat[1] = {0x5a};
          mm.fill(p + kChunkSize, pat, 1, kChunkSize);   // a uniform chunk (0x5a bytes)
          const uint8_t junk[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
          mm.write(p + 2 * kChunkSize, junk, 16);        // a real one; the first chunk is never written
          // The words a warp picks from: a few in each chunk, so lanes meet on a word often.
          std::vector<uint64_t> pool;
          const unsigned npool = 1 + static_cast<unsigned>(rng() % 12);
          for (unsigned i = 0; i < npool; ++i) {
            const uint64_t a = p + (rng() % 3) * kChunkSize + n * (rng() % 6);
            if (std::find(pool.begin(), pool.end(), a) == pool.end()) pool.push_back(a);
          }
          uint64_t lane_addr[32], v[32], swp[32], old_of[32] = {}, want_old[32];
          for (unsigned l = 0; l < 32; ++l) {
            lane_addr[l] = pool[rng() % pool.size()];
            v[l] = operand(rng, type);
            swp[l] = operand(rng, type);
            if (cas && rng() % 3 == 0) v[l] = mm.load_scalar(lane_addr[l], n);   // (a compare that can win)
          }
          const uint32_t ex = it % 3 == 0 ? 0xffffffffu : (static_cast<uint32_t>(rng()) | 1u);
          // The model: the words as they are now, the lanes one after another.
          std::vector<std::pair<uint64_t, uint64_t>> model;   // address, value
          for (uint64_t a : pool) model.push_back({a, mm.load_scalar(a, n)});
          bool want_changed = false;
          for (uint32_t m = ex; m; m &= m - 1) {
            const unsigned l = static_cast<unsigned>(__builtin_ctz(m));
            for (auto& e : model)
              if (e.first == lane_addr[l]) {
                uint64_t nw = 0;
                VCHECK(sa::apply(op, type, cas, false, e.second, v[l], swp[l], nw));
                want_old[l] = e.second;
                want_changed |= nw != e.second;
                e.second = nw;
              }
          }
          const Warp w = group(lane_addr, ex);
          const int got = sa::global_lockfree(mm, op, type, cas, w.nu, w.uaddr, w.lane_mask, v, cas ? swp : nullptr, old_of);
          VCHECK_EQ(got, want_changed ? 1 : 0);
          for (uint32_t m = ex; m; m &= m - 1) {
            const unsigned l = static_cast<unsigned>(__builtin_ctz(m));
            VCHECK_EQ(old_of[l], want_old[l]);
          }
          for (auto& e : model) VCHECK_EQ(mm.load_scalar(e.first, n), e.second & keep);
          ++done;
        }
  VCHECK(done > 4000);
}

VTEST(a_warp_with_a_word_the_memory_declines_is_declined_with_nothing_done) {
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(4096);
  uint64_t v[32], old_of[32];
  for (unsigned l = 0; l < 32; ++l) v[l] = 1, old_of[l] = 99;
  const uint64_t good = p + 16;
  const uint64_t bad[] = {p + 2, p + 4094 + 4, p - 8};   // misaligned, runs past the end, before the start
  for (uint64_t b : bad) {
    const uint64_t uaddr[2] = {good, b};
    const uint32_t lm[2] = {1u, 2u};
    VCHECK_EQ(sa::global_lockfree(mm, 0, 0, false, 2, uaddr, lm, v, nullptr, old_of), -1);
    VCHECK_EQ(mm.load_scalar(good, 4), 0u);   // (the good word is untouched too)
    VCHECK_EQ(old_of[0], 99u);
  }
  mm.free(p);
  const uint64_t uaddr[1] = {good};
  const uint32_t lm[1] = {1u};
  VCHECK_EQ(sa::global_lockfree(mm, 0, 0, false, 1, uaddr, lm, v, nullptr, old_of), -1);   // freed
}

namespace {
// Several threads, each running full warps (32 lanes, one word or a word a lane) of one operation.
template <class Check>
void run_threads(unsigned nthreads, Check&& body) {
  std::atomic<int> go{0};
  std::vector<std::thread> ts;
  for (unsigned t = 0; t < nthreads; ++t)
    ts.emplace_back([&, t] {
      while (!go.load()) {
      }
      body(t);
    });
  go.store(1);
  for (auto& t : ts) t.join();
}
}  // namespace

VTEST(a_fetch_add_from_many_threads_hands_out_each_old_value_once) {
  // Every lane of every warp of every thread adds 1 to one word: the old values over all of them are exactly 0 .. N-1, whatever the
  // order, for u32, u64 and f32 (exact below 2^24); and the word ends at N.
  for (unsigned type : {0u, 2u, 3u}) {
    MemoryManager mm(1ull << 28);
    const uint64_t p = mm.alloc(4096);
    const uint32_t n = type == 2 ? 8 : 4;
    const unsigned kThreads = 6, kWarps = 400;
    std::vector<std::vector<uint64_t>> seen(kThreads);
    run_threads(kThreads, [&](unsigned t) {
      uint64_t v[32], old_of[32];
      const uint64_t one = type == 3 ? 0x3f800000u : 1u;
      for (unsigned l = 0; l < 32; ++l) v[l] = one;
      const uint64_t uaddr[1] = {p};
      const uint32_t lm[1] = {0xffffffffu};
      for (unsigned i = 0; i < kWarps; ++i) {
        if (sa::global_lockfree(mm, 0, type, false, 1, uaddr, lm, v, nullptr, old_of) < 0) throw std::runtime_error("declined");
        for (unsigned l = 0; l < 32; ++l) seen[t].push_back(old_of[l]);
      }
    });
    const uint64_t total = uint64_t{kThreads} * kWarps * 32;
    std::vector<uint8_t> hit(total, 0);
    for (auto& s : seen)
      for (uint64_t o : s) {
        const uint64_t k = type == 3 ? static_cast<uint64_t>(sa::as_f32(static_cast<uint32_t>(o))) : o;
        VCHECK(k < total);
        if (k < total) {
          VCHECK_EQ(hit[k], 0u);   // (handed out once)
          hit[k] = 1;
        }
      }
    const uint64_t last = mm.load_scalar(p, n);
    VCHECK_EQ(type == 3 ? static_cast<uint64_t>(sa::as_f32(static_cast<uint32_t>(last))) : last, total);
  }
}

VTEST(the_commutative_operations_from_many_threads_leave_the_serial_result) {
  // add / and / or / xor / min / max on words of both widths, spread (a word a lane, 64 words, so warps overlap) and contended (one
  // word). Each thread's operands are known, so the serial result is computed here.
  const unsigned kThreads = 6, kWarps = 300;
  for (unsigned wide = 0; wide < 2; ++wide)
    for (unsigned op : {0u, 1u, 2u, 5u, 6u, 7u})
      for (unsigned spread = 0; spread < 2; ++spread) {
        const unsigned type = wide ? 2 : 0;   // u32 / u64
        const uint32_t n = wide ? 8 : 4;
        const uint64_t keep = wide ? ~uint64_t{0} : 0xffffffffull;
        MemoryManager mm(1ull << 28);
        const uint64_t p = mm.alloc(1 << 16);
        const unsigned nwords = spread ? 64 : 1;
        // Initial words: all ones for 'and' and min, else zero.
        const uint64_t init = (op == 5 || op == 1) ? keep : 0;
        for (unsigned k = 0; k < nwords; ++k) {
          const uint64_t x = init;
          mm.write(p + n * k, &x, n);
        }
        const auto operand_of = [&](unsigned t, unsigned i, unsigned l) -> uint64_t {
          std::mt19937_64 r((uint64_t{t} << 40) ^ (uint64_t{i} << 8) ^ l ^ 0x9e3779b9u);
          uint64_t x = r() & keep;
          if (op == 5) x |= r() & r();   // (and: mostly ones, so bits survive a while)
          return x;
        };
        run_threads(kThreads, [&](unsigned t) {
          uint64_t v[32], old_of[32];
          uint64_t uaddr[32];
          uint32_t lm[32];
          for (unsigned i = 0; i < kWarps; ++i) {
            for (unsigned l = 0; l < 32; ++l) v[l] = operand_of(t, i, l);
            unsigned nu;
            if (spread) {
              nu = 32;
              for (unsigned l = 0; l < 32; ++l) uaddr[l] = p + n * ((l * 2 + i) % nwords), lm[l] = uint32_t{1} << l;
              // (lanes l and l+... can name one word: make them distinct words by construction, 32 of 64)
            } else {
              nu = 1;
              uaddr[0] = p;
              lm[0] = 0xffffffffu;
            }
            if (sa::global_lockfree(mm, op, type, false, nu, uaddr, lm, v, nullptr, old_of) < 0) throw std::runtime_error("declined");
          }
        });
        std::vector<uint64_t> want(nwords, init);
        for (unsigned t = 0; t < kThreads; ++t)
          for (unsigned i = 0; i < kWarps; ++i)
            for (unsigned l = 0; l < 32; ++l) {
              uint64_t& w = want[spread ? (l * 2 + i) % nwords : 0];
              const uint64_t x = operand_of(t, i, l);
              switch (op) {
                case 0: w = (w + x) & keep; break;
                case 1: w = std::min(w, x); break;
                case 2: w = std::max(w, x); break;
                case 5: w &= x; break;
                case 6: w |= x; break;
                default: w ^= x; break;
              }
            }
        for (unsigned k = 0; k < nwords; ++k) VCHECK_EQ(mm.load_scalar(p + n * k, n), want[k]);
      }
}

VTEST(two_byte_atomics_are_atomic_against_four_byte_ones_on_the_same_word) {
  // Threads add 1 to the low half of each word with atomic_rmw16 (the 2-byte road) and others add 0x10000 to the whole word with a
  // 4-byte swap: both counts must come out exact, which a plain 16-bit store beside a 4-byte swap cannot do. Also the high half by
  // atomic_rmw16, and a word in an untouched chunk.
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(4096);
  const unsigned kWords = 8, kIters = 20000, kLo = 3, kHi = 3, kWide = 3;
  run_threads(kLo + kHi + kWide, [&](unsigned t) {
    for (unsigned i = 0; i < kIters; ++i)
      for (unsigned k = 0; k < kWords; ++k) {
        uint64_t old = 0;
        bool ok;
        if (t < kLo) ok = vgpu::exec::atomic_rmw16(mm, p + 4 * k, [](uint64_t v) { return (v + 1) & 0xffff; }, &old);
        else if (t < kLo + kHi) ok = vgpu::exec::atomic_rmw16(mm, p + 4 * k + 2, [](uint64_t v) { return (v + 1) & 0xffff; }, &old);
        else ok = mm.atomic_rmw(p + 4 * k, 4, [](uint64_t v) { return (v + 0x10000) & 0xffffffffull; }, &old, true);
        if (!ok) throw std::runtime_error("declined");
      }
  });
  const uint64_t lo = uint64_t{kLo} * kIters & 0xffff, hi = (uint64_t{kHi} * kIters + uint64_t{kWide} * kIters) & 0xffff;
  for (unsigned k = 0; k < kWords; ++k) VCHECK_EQ(mm.load_scalar(p + 4 * k, 4), (hi << 16) | lo);
  uint64_t old = 7;
  VCHECK(!vgpu::exec::atomic_rmw16(mm, p + 1, [](uint64_t v) { return v; }, &old));   // misaligned: declined
  VCHECK_EQ(old, 7u);
}

VTEST_MAIN
