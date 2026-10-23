// The allocation table's bookkeeping: what alloc and free hand out and report, the quarantine of freed
// ranges, the chunks that go back to the backing and come out again, and the lock and epoch under the
// table that let a thread keep the allocations it found across other allocations and frees.
//
// These behaviours are observable (addresses, diagnostics, resident memory) and the implementation
// underneath them is built for speed (a quiet lock for allocations and frees, a targeted invalidation
// of what threads remember, recycled table nodes and chunks), so the checks here are of three kinds:
//
//  * a model: a few lines that say what a bump allocator with a bounded quarantine must answer, run
//    against the manager over random operations, error code for error code;
//  * a digest: the same random operations hashed with every value, address and error message, which
//    must equal the digest the implementation before the speed work produced (recorded from it);
//  * the mechanisms: a freed allocation is never found again through a memory of it, however many other
//    frees came between (the count wraps), chunks come back from the pool zeroed, the lock excludes
//    readers when quiet, and a quiet hold leaves the epoch alone.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/brlock.hpp"
#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/memory_backing.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

struct Rng {
  uint64_t s;
  uint64_t next() {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return s;
  }
  uint64_t below(uint64_t n) { return next() % n; }
};

struct Hash {
  uint64_t h = 1469598103934665603ull;
  void add(uint64_t v) {
    for (int i = 0; i < 8; ++i) h = (h ^ ((v >> (8 * i)) & 0xff)) * 1099511628211ull;
  }
  void add(const std::string& t) {
    for (unsigned char c : t) h = (h ^ c) * 1099511628211ull;
    add(t.size());
  }
};

// One operation's outcome: its error (code, or -1) and text, hashed by the caller.
struct Outcome {
  int code = -1;
  std::string text;
  uint64_t value = 0;
};
template <typename F>
Outcome attempt(F&& f) {
  Outcome o;
  try {
    o.value = f();
  } catch (const Error& e) {
    o.code = static_cast<int>(e.code());
    o.text = e.what();
  }
  return o;
}

const uint64_t kSizes[] = {1, 4, 8, 63, 64, 100, 192, 255, 256, 257, 1000, 4095, 4096, 4097,
                           65535, 65536, 65537, 70000, 131072, 200000};
constexpr size_t kNumSizes = sizeof kSizes / sizeof kSizes[0];

// ---- the model -----------------------------------------------------------------

// What a bump allocator with a quarantine of the last kQuarantineEntries frees must say.
struct Model {
  uint64_t va_base, next, high;
  uint64_t page = 0;
  uint64_t capacity, used = 0;
  std::map<uint64_t, uint64_t> live;
  std::map<uint64_t, uint64_t> freed;   // base -> size, the quarantine
  std::deque<uint64_t> order;
  Model(uint64_t cap) : va_base(kDeviceVaBase), next(kDeviceVaBase), high(kDeviceVaBase), capacity(cap) {}

  // The address alloc must return, or 0 when it must run out of memory.
  uint64_t alloc(uint64_t size, uint64_t request) {
    if (used + size > capacity) return 0;
    const uint64_t align = std::max<uint64_t>({256, page, request});
    const uint64_t base = (next + align - 1) / align * align;
    next = base + (size + align - 1) / align * align;
    high = next;
    used += size;
    live[base] = size;
    return base;
  }
  // What free(ptr) must raise: -1 for nothing.
  int free(uint64_t ptr) {
    if (auto it = live.find(ptr); it != live.end()) {
      used -= it->second;
      freed[ptr] = it->second;
      order.push_back(ptr);
      while (order.size() > kQuarantineEntries) {
        // The oldest entry goes; a pointer freed twice is not in `live` to be freed twice, so each
        // appears in the order once.
        freed.erase(order.front());
        order.pop_front();
      }
      live.erase(it);
      return -1;
    }
    if (freed.count(ptr)) return static_cast<int>(Err::DoubleFree);
    auto up = live.upper_bound(ptr);
    if (up != live.begin()) {
      auto prev = std::prev(up);
      if (ptr < prev->first + prev->second) return static_cast<int>(Err::InvalidFree);
    }
    return static_cast<int>(Err::InvalidPointer);
  }
  // What a one-byte read at `addr` must raise: -1 for nothing.
  int read(uint64_t addr) const {
    auto up = live.upper_bound(addr);
    if (up != live.begin()) {
      auto prev = std::prev(up);
      const uint64_t align = std::max<uint64_t>(256, page);
      if (addr < prev->first + prev->second) return -1;
      if (addr < prev->first + (prev->second + align - 1) / align * align) return static_cast<int>(Err::OutOfBounds);
    }
    auto fup = freed.upper_bound(addr);
    if (fup != freed.begin()) {
      auto prev = std::prev(fup);
      if (addr < prev->first + prev->second) return static_cast<int>(Err::UseAfterFree);
    }
    if (addr >= va_base && addr < high) return static_cast<int>(Err::UseAfterFree);
    return static_cast<int>(Err::InvalidPointer);
  }
};

// Random alloc, free, stale free, interior free and read, against the model, across the quarantine's
// bound (more than kQuarantineEntries frees), checking the address handed out, the error code of every
// refusal, and the manager's own accounting.
void run_model(uint64_t seed, uint64_t page, int rounds, uint64_t capacity) {
  MemoryManager mm(capacity);
  Model model(capacity);
  if (page) mm.set_page_size(page), model.page = page;
  Rng rng{seed};
  std::vector<uint64_t> bases, stale;
  auto fail = [&](const char* what, int round, long expected, long got) {
    char buf[300];
    std::snprintf(buf, sizeof buf, "seed %llu round %d (%s): model says %ld, manager says %ld",
                  (unsigned long long)seed, round, what, expected, got);
    throw vtest::Failure(buf);
  };
  for (int r = 0; r < rounds; ++r) {
    const uint64_t kind = rng.below(100);
    if (kind < 45) {
      const uint64_t size = kSizes[rng.below(kNumSizes)];
      const uint64_t request = rng.below(8) == 0 ? (uint64_t{1} << (8 + rng.below(8))) : 0;
      const uint64_t want = model.alloc(size, request);
      Outcome o = attempt([&] { return mm.alloc(size, request); });
      if (!want) {
        if (o.code != static_cast<int>(Err::OutOfMemory)) fail("alloc must run out of memory", r, 3, o.code);
      } else {
        if (o.code != -1) fail("alloc must succeed", r, -1, o.code);
        if (o.value != want) fail("alloc address", r, static_cast<long>(want), static_cast<long>(o.value));
        bases.push_back(want);
      }
    } else if (kind < 80 && !bases.empty()) {
      const size_t k = rng.below(bases.size());
      const uint64_t ptr = bases[k];
      bases.erase(bases.begin() + k);
      stale.push_back(ptr);
      const int want = model.free(ptr);
      Outcome o = attempt([&] { mm.free(ptr); return 0; });
      if (o.code != want) fail("free", r, want, o.code);
    } else if (kind < 88 && !stale.empty()) {
      const uint64_t ptr = stale[rng.below(stale.size())];
      const int want = model.free(ptr);
      Outcome o = attempt([&] { mm.free(ptr); return 0; });
      if (o.code != want) fail("stale free", r, want, o.code);
    } else if (kind < 92 && !bases.empty()) {
      const uint64_t ptr = bases[rng.below(bases.size())] + 1 + rng.below(3);
      const int want = model.free(ptr);
      Outcome o = attempt([&] { mm.free(ptr); return 0; });
      if (o.code != want) fail("interior free", r, want, o.code);
    } else {
      // A read: inside a live allocation, in a freed one, in the padding, in the retired range, or nowhere.
      uint64_t addr;
      switch (rng.below(4)) {
        case 0: addr = bases.empty() ? kDeviceVaBase : bases[rng.below(bases.size())] + rng.below(300); break;
        case 1: addr = stale.empty() ? kDeviceVaBase : stale[rng.below(stale.size())] + rng.below(300); break;
        case 2: addr = model.va_base + rng.below(model.high - model.va_base + 4096); break;
        default: addr = 0x1000 + rng.below(1 << 20); break;
      }
      const int want = model.read(addr);
      uint8_t byte = 0;
      Outcome o = attempt([&] { mm.read(addr, &byte, 1); return 0; });
      if (o.code != want) fail("read", r, want, o.code);
      if (want == -1 && byte != 0) fail("read of fresh memory", r, 0, byte);
    }
    if (mm.used() != model.used) fail("used()", r, static_cast<long>(model.used), static_cast<long>(mm.used()));
    if (mm.live_allocations() != model.live.size())
      fail("live_allocations()", r, static_cast<long>(model.live.size()), static_cast<long>(mm.live_allocations()));
  }
  mm.free_all();
  if (mm.used() != 0 || mm.live_allocations() != 0) throw vtest::Failure("free_all left allocations");
}

// ---- the digest ----------------------------------------------------------------

// The same kind of random operations with values written and read, hashing everything observable:
// addresses, values, error codes and texts, used(), resident_bytes().
uint64_t run_digest(uint64_t seed, uint64_t page, int rounds, uint64_t capacity) {
  MemoryManager mm(capacity);
  if (page) mm.set_page_size(page);
  Rng rng{seed};
  Hash h;
  std::vector<uint64_t> bases, sizes, stale;
  auto note = [&](const Outcome& o) {
    h.add(static_cast<uint64_t>(o.code + 1));
    h.add(o.text);
    h.add(o.value);
  };
  for (int r = 0; r < rounds; ++r) {
    const uint64_t kind = rng.below(100);
    if (kind < 30) {
      const uint64_t size = kSizes[rng.below(kNumSizes)];
      Outcome o = attempt([&] { return mm.alloc(size); });
      note(o);
      if (o.code < 0) bases.push_back(o.value), sizes.push_back(size);
    } else if (kind < 52 && !bases.empty()) {
      const size_t k = rng.below(bases.size());
      const uint64_t ptr = bases[k];
      bases.erase(bases.begin() + k), sizes.erase(sizes.begin() + k);
      stale.push_back(ptr);
      note(attempt([&] { mm.free(ptr); return 0; }));
    } else if (kind < 58 && !stale.empty()) {
      const uint64_t ptr = stale[rng.below(stale.size())];
      note(attempt([&] { mm.free(ptr); return 0; }));
    } else if (kind < 60 && !bases.empty()) {
      const uint64_t ptr = bases[rng.below(bases.size())] + 1 + rng.below(3);
      note(attempt([&] { mm.free(ptr); return 0; }));
    } else if (kind < 76 && !bases.empty()) {
      const size_t k = rng.below(bases.size());
      const uint64_t off = rng.below(sizes[k] + (rng.below(10) == 0 ? 600 : 0)) & ~uint64_t{3};
      const uint64_t v = rng.next();
      note(attempt([&] { mm.store_scalar(bases[k] + off, 4, v); return 0; }));
    } else if (kind < 88 && !bases.empty()) {
      const size_t k = rng.below(bases.size());
      const uint64_t off = rng.below(sizes[k] + (rng.below(10) == 0 ? 600 : 0)) & ~uint64_t{3};
      note(attempt([&] { return mm.load_scalar(bases[k] + off, 4); }));
    } else if (kind < 92 && !bases.empty()) {
      const size_t k = rng.below(bases.size());
      const uint64_t len = 1 + rng.below(std::min<uint64_t>(sizes[k], 5000));
      const uint64_t off = rng.below(sizes[k]);
      std::vector<uint8_t> buf(len, static_cast<uint8_t>(r));
      note(attempt([&] { mm.write(bases[k] + off, buf.data(), len); return 0; }));
    } else if (kind < 94 && !bases.empty()) {
      const size_t k = rng.below(bases.size());
      const uint8_t pattern = static_cast<uint8_t>(rng.below(3) == 0 ? 0 : rng.next());
      note(attempt([&] { mm.fill(bases[k], &pattern, 1, sizes[k]); return 0; }));
    } else if (kind < 98) {
      // A read from anywhere, freed memory included.
      uint64_t addr;
      switch (rng.below(3)) {
        case 0: addr = stale.empty() ? kDeviceVaBase : stale[rng.below(stale.size())] + rng.below(64); break;
        case 1: addr = kDeviceVaBase + rng.below(1 << 24); break;
        default: addr = bases.empty() ? 0x4000 : bases[rng.below(bases.size())] + rng.below(70000); break;
      }
      addr &= ~uint64_t{3};
      note(attempt([&] { return mm.load_scalar(addr, 4); }));
    } else if (!bases.empty()) {
      const size_t k = rng.below(bases.size());
      std::vector<uint8_t> all(sizes[k]);
      Outcome o = attempt([&] { mm.read(bases[k], all.data(), all.size()); return 0; });
      Hash c;
      for (uint8_t b : all) c.add(b);
      o.value = c.h;
      note(o);
    }
    h.add(mm.used());
    h.add(mm.live_allocations());
    if ((r & 63) == 0) h.add(mm.resident_bytes());
  }
  h.add(mm.resident_bytes());
  mm.free_all();
  h.add(mm.used());
  return h.h;
}

// The digests the implementation before the speed work produced for these runs (recorded from it with
// this file: `test_memalloc --digests`). A difference is a change in what the manager answers.
struct DigestCase {
  uint64_t seed, page;
  int rounds;
  uint64_t capacity;
  uint64_t expected;
};
const DigestCase kDigests[] = {
    {101, 0, 12000, 1ull << 34, 0xd36551903dafe3d2ull},
    {202, 4096, 12000, 1ull << 34, 0x0af85cd27e5c6449ull},
    {303, 0, 12000, 3ull << 20, 0x72aa3866e7982dbdull},        // runs out of memory often
    {404, 0, 40000, 1ull << 34, 0x8bf6312a90beaebeull},        // frees past the quarantine bound
};

}  // namespace

VTEST(the_manager_answers_as_the_model_does) {
  run_model(1, 0, 9000, 1ull << 32);
  run_model(2, 4096, 9000, 1ull << 32);
  run_model(3, 0, 9000, 1ull << 20);     // out of memory is common
  run_model(4, 65536, 6000, 1ull << 32);
}

VTEST(the_quarantine_holds_exactly_its_bound) {
  MemoryManager mm(1ull << 32);
  std::vector<uint64_t> p;
  for (size_t i = 0; i < kQuarantineEntries + 3; ++i) p.push_back(mm.alloc(64));
  for (uint64_t x : p) mm.free(x);
  // The three oldest have left the quarantine: a second free is an invalid pointer in the retired
  // range, not a double free. The rest are double frees.
  for (size_t i = 0; i < 3; ++i) VCHECK(VCAPTURE(Error, mm.free(p[i])).code() == Err::InvalidPointer);
  for (size_t i = 3; i < p.size(); ++i) VCHECK(VCAPTURE(Error, mm.free(p[i])).code() == Err::DoubleFree);
  uint8_t b = 0;
  VCHECK(VCAPTURE(Error, mm.read(p[0], &b, 1)).code() == Err::UseAfterFree);
  VCHECK_CONTAINS(VCAPTURE(Error, mm.read(p[0], &b, 1)).what(), "quarantine");
  VCHECK_CONTAINS(VCAPTURE(Error, mm.read(p[5], &b, 1)).what(), "touches freed allocation");
}

VTEST(random_operations_digest_as_before) {
  for (const DigestCase& c : kDigests) {
    const uint64_t got = run_digest(c.seed, c.page, c.rounds, c.capacity);
    if (std::getenv("VGPU_PRINT_DIGESTS"))
      std::printf("    {%llu, %llu, %d, %lluull, 0x%016llxull},\n", (unsigned long long)c.seed,
                  (unsigned long long)c.page, c.rounds, (unsigned long long)c.capacity, (unsigned long long)got);
    else
      VCHECK_EQ(got, c.expected);
  }
}

// Reading an allocation puts it in the thread's memory; freeing it, after any number of other frees
// (the count that tells them apart wraps at 65536), must make the next read an error, and the frees of
// others must not.
VTEST(a_freed_allocation_is_never_found_through_a_memory_of_it) {
  const int others[] = {0, 1, 2, 63, 64, 65, 66, 130, 4097, 65534, 65535, 65536, 65537, 140000};
  for (int n : others) {
    MemoryManager mm(1ull << 34);
    const uint64_t a = mm.alloc(4096), b = mm.alloc(4096);
    mm.store_scalar(a, 4, 11), mm.store_scalar(b, 4, 22);
    VCHECK_EQ(mm.load_scalar(a, 4), uint64_t{11});
    VCHECK_EQ(mm.load_scalar(b, 4), uint64_t{22});
    for (int i = 0; i < n; ++i) {
      const uint64_t t = mm.alloc(192);
      mm.store_scalar(t, 8, static_cast<uint64_t>(i));
      mm.free(t);
      // `b` stays right across the others' frees, and keeps being found.
      if (i % 997 == 0) VCHECK_EQ(mm.load_scalar(b, 4), uint64_t{22});
    }
    VCHECK_EQ(mm.load_scalar(a, 4), uint64_t{11});   // still live: not an error, not a stale answer
    mm.free(a);
    auto err = VCAPTURE(Error, mm.load_scalar(a, 4));
    VCHECK(err.code() == Err::UseAfterFree);
    VCHECK_EQ(mm.load_scalar(b, 4), uint64_t{22});
    auto werr = VCAPTURE(Error, mm.store_scalar(a, 4, 5));
    VCHECK(werr.code() == Err::UseAfterFree);
    uint64_t run[4];
    VCHECK(!mm.load_run(a, run, sizeof run, 4));   // the run path does not find it either
    VCHECK(mm.load_run(b, run, sizeof run, 4));
  }
}

// Several allocations in the thread's memory at once, one freed at a time, in every order.
VTEST(frees_leave_the_other_remembered_allocations_right) {
  MemoryManager mm(1ull << 34);
  for (int round = 0; round < 40; ++round) {
    std::vector<uint64_t> p;
    for (int i = 0; i < 6; ++i) {
      p.push_back(mm.alloc(1024));
      mm.store_scalar(p.back(), 4, 100 + i);
    }
    for (uint64_t x : p) (void)mm.load_scalar(x, 4);
    Rng rng{static_cast<uint64_t>(round) * 2654435761u + 1};
    std::vector<bool> alive(6, true);
    for (int step = 0; step < 6; ++step) {
      size_t k;
      do k = rng.below(6); while (!alive[k]);
      mm.free(p[k]);
      alive[k] = false;
      for (size_t i = 0; i < 6; ++i) {
        if (alive[i]) VCHECK_EQ(mm.load_scalar(p[i], 4), uint64_t{100 + i});
        else VCHECK(VCAPTURE(Error, mm.load_scalar(p[i], 4)).code() == Err::UseAfterFree);
      }
    }
  }
}

// Threads read long-lived allocations while another allocates and frees, and one more frees and
// allocates others: every value read is right.
VTEST(readers_beside_allocations_and_frees_read_the_right_values) {
  MemoryManager mm(1ull << 36);
  constexpr int kLive = 5;
  uint64_t live[kLive];
  for (int i = 0; i < kLive; ++i) {
    live[i] = mm.alloc(1 << 16);
    for (int j = 0; j < 64; ++j) mm.store_scalar(live[i] + 8 * j, 8, (static_cast<uint64_t>(i) << 32) | static_cast<uint64_t>(j));
  }
  std::atomic<bool> stop{false};
  std::atomic<long> wrong{0}, reads{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t)
    threads.emplace_back([&, t] {
      Rng rng{static_cast<uint64_t>(t) + 7};
      while (!stop.load(std::memory_order_relaxed)) {
        for (int n = 0; n < 200; ++n) {
          const int i = static_cast<int>(rng.below(kLive)), j = static_cast<int>(rng.below(64));
          if (mm.load_scalar(live[i] + 8 * j, 8) != ((uint64_t{static_cast<uint64_t>(i)} << 32) | static_cast<uint64_t>(j)))
            wrong.fetch_add(1);
        }
        reads.fetch_add(200);
      }
    });
  std::thread churn([&] {
    std::vector<uint8_t> seg(192, 3);
    while (!stop.load(std::memory_order_relaxed)) {
      const uint64_t k = mm.alloc(192);
      mm.write(k, seg.data(), seg.size());
      if (mm.load_scalar(k, 1) != 3) wrong.fetch_add(1);
      mm.free(k);
    }
  });
  std::thread churn2([&] {
    Rng rng{99};
    std::vector<uint64_t> mine;
    while (!stop.load(std::memory_order_relaxed)) {
      if (mine.size() < 20 || rng.below(2)) {
        mine.push_back(mm.alloc(1 + rng.below(100000)));
      } else {
        const size_t k = rng.below(mine.size());
        mm.free(mine[k]);
        mine.erase(mine.begin() + k);
      }
    }
    for (uint64_t x : mine) mm.free(x);
  });
  while (reads.load() < 3000000) std::this_thread::yield();
  stop = true;
  for (auto& t : threads) t.join();
  churn.join();
  churn2.join();
  VCHECK_EQ(wrong.load(), 0);
  for (int i = 0; i < kLive; ++i) VCHECK_EQ(mm.load_scalar(live[i], 8), (uint64_t{static_cast<uint64_t>(i)} << 32));
}

// A thread's lookup memory is found by its thread pointer in a table of 1024 slots: threads that come
// and go (a stream makes one) give their slot back, and more threads at once than the table has slots
// still read the right values (the ones without a slot find nothing remembered and take the table).
VTEST(threads_that_come_and_go_and_more_than_the_table_holds_read_right) {
  MemoryManager mm(1ull << 34);
  const uint64_t a = mm.alloc(1 << 16);
  for (int i = 0; i < 64; ++i) mm.store_scalar(a + 8 * i, 8, 1000 + i);
  std::atomic<long> wrong{0};
  auto reads = [&] {
    for (int k = 0; k < 200; ++k)
      for (int i = 0; i < 64; i += 7)
        if (mm.load_scalar(a + 8 * i, 8) != uint64_t(1000 + i)) wrong.fetch_add(1);
  };
  for (int round = 0; round < 3000; ++round) std::thread(reads).join();   // slots are given back
  std::atomic<bool> go{false};
  std::vector<std::thread> many;
  for (int t = 0; t < 1400; ++t)
    many.emplace_back([&] {
      while (!go.load()) std::this_thread::yield();
      reads();
    });
  go = true;
  for (auto& t : many) t.join();
  VCHECK_EQ(wrong.load(), 0);
}

// ---- chunks that go back and come out again ------------------------------------

namespace {
bool all_zero(MemoryManager& mm, uint64_t base, uint64_t size) {
  std::vector<uint8_t> buf(size);
  mm.read(base, buf.data(), size);
  for (uint8_t b : buf)
    if (b) return false;
  return true;
}
}  // namespace

VTEST(a_new_allocation_reads_zero_whatever_the_last_one_held) {
  for (uint64_t page : {uint64_t{0}, uint64_t{4096}, uint64_t{65536}}) {
    MemoryManager mm(1ull << 34);
    if (page) mm.set_page_size(page);
    Rng rng{page + 17};
    for (int round = 0; round < 600; ++round) {
      const uint64_t size = kSizes[rng.below(kNumSizes)];
      const uint64_t base = mm.alloc(size);
      // Make every chunk real (a store of zero), so that what the recycled ones held would show.
      for (uint64_t at = 0; at < size; at += kChunkSize) mm.store_scalar(base + at, 1, 0);
      mm.store_scalar(base + size - 1, 1, 0);
      VCHECK(all_zero(mm, base, size));
      // Reads in the page slack past the end are zero as well (a kernel's read may run on).
      if (page) {
        const uint64_t slack_end = (size + page - 1) / page * page;
        for (uint64_t at = size & ~uint64_t{3}; at + 4 <= slack_end && at < size + 64; at += 4)
          if (at >= size) VCHECK_EQ(mm.load_scalar(base + at, 4), uint64_t{0});
      }
      // Dirty it: scalar writes, a copy, a fill (the whole of it with a byte makes chunks uniform),
      // and a write over a uniform chunk, which makes it real with the byte written all through it.
      const int dirt = static_cast<int>(rng.below(5));
      if (dirt == 0 || dirt == 4) {
        for (int i = 0; i < 20; ++i) mm.store_scalar(base + (rng.below(size) & ~uint64_t{3}) , 1, 0xff);
      }
      if (dirt == 1) {
        std::vector<uint8_t> junk(size, 0xA5);
        mm.write(base, junk.data(), junk.size());
      }
      if (dirt == 2) {
        const uint8_t pat = 0x5A;
        mm.fill(base, &pat, 1, size);
        mm.store_scalar(base + (rng.below(size) & ~uint64_t{3}), 1, 7);   // uniform chunk becomes real, written whole
      }
      if (dirt == 3) {
        const uint8_t pat = 0x33;
        mm.fill(base, &pat, 1, size);
      }
      if (dirt == 4) {
        const uint8_t pat[3] = {1, 2, 3};
        mm.fill(base, pat, 3, size);
      }
      mm.free(base);
    }
  }
}

// The tests below use the lock and the backing as they are now; `-DVGPU_MEMALLOC_DIGESTS_ONLY` leaves them out
// so that this file builds against the code it was recorded from.
#ifndef VGPU_MEMALLOC_DIGESTS_ONLY
VTEST(chunks_returned_nearly_untouched_come_back_zeroed_and_counted) {
  const uint64_t ram = backing::ram_bytes();
  uint8_t* c = backing::acquire();
  VCHECK_EQ(backing::ram_bytes(), ram + kChunkSize);
  for (uint64_t i = 0; i < kChunkSize; ++i) VCHECK_EQ(c[i], uint8_t{0});
  std::memset(c, 0xEE, 192);
  backing::release(c, 192);
  VCHECK_EQ(backing::ram_bytes(), ram);   // pooled chunks are not in use
  // Whatever comes next -- the pooled chunk or a new one -- is zero, all of it.
  std::vector<uint8_t*> got;
  for (int i = 0; i < 100; ++i) got.push_back(backing::acquire());
  VCHECK_EQ(backing::ram_bytes(), ram + 100 * kChunkSize);
  for (uint8_t* g : got)
    for (uint64_t i = 0; i < kChunkSize; ++i)
      if (g[i]) throw vtest::Failure("a chunk from the backing was not zero");
  // A chunk given back with no promise goes to the heap, and the count follows.
  for (uint8_t* g : got) backing::release(g);
  VCHECK_EQ(backing::ram_bytes(), ram);
}

VTEST(resident_memory_follows_allocations_through_recycling) {
  MemoryManager mm(1ull << 34);
  VCHECK_EQ(mm.resident_bytes(), uint64_t{0});
  const uint64_t ram = backing::ram_bytes();
  for (int i = 0; i < 300; ++i) {
    const uint64_t a = mm.alloc(192);
    VCHECK_EQ(mm.resident_bytes(), uint64_t{0});
    mm.store_scalar(a + 64, 4, 1);
    VCHECK_EQ(mm.resident_bytes(), kChunkSize);
    VCHECK_EQ(backing::ram_bytes(), ram + kChunkSize);
    mm.free(a);
    VCHECK_EQ(mm.resident_bytes(), uint64_t{0});
    VCHECK_EQ(backing::ram_bytes(), ram);
  }
}

// ---- the lock under the table --------------------------------------------------

VTEST(a_quiet_hold_leaves_the_epoch_and_a_full_one_changes_it) {
  BigReaderLock lock;
  const uint64_t e0 = lock.epoch();
  lock.lock_quiet();
  VCHECK(lock.written_by_me());
  VCHECK_EQ(lock.epoch(), e0);
  lock.unlock();
  VCHECK_EQ(lock.epoch(), e0);
  lock.lock();
  const uint64_t during = lock.epoch();
  VCHECK(during != e0);
  lock.unlock();
  VCHECK(lock.epoch() != during);
  // A full hold inside a quiet one voids at once and again at the end.
  const uint64_t e1 = lock.epoch();
  lock.lock_quiet();
  lock.lock();
  VCHECK(lock.epoch() != e1);
  const uint64_t inner = lock.epoch();
  lock.unlock();
  VCHECK_EQ(lock.epoch(), inner);
  lock.unlock();
  VCHECK(lock.epoch() != inner);
  // A quiet hold inside a full one does nothing more.
  lock.lock();
  const uint64_t full = lock.epoch();
  lock.lock_quiet();
  lock.unlock();
  VCHECK_EQ(lock.epoch(), full);
  lock.unlock();
}

VTEST(a_remembered_answer_survives_the_frees_of_others_and_not_its_own) {
  BigReaderLock lock;
  lock.lock_quiet();
  const uint64_t seen = lock.epoch();
  lock.unlock();
  uint64_t now = 0;
  VCHECK(lock.still_valid(seen, 0x1000, &now));
  VCHECK_EQ(now, seen);
  for (uint64_t i = 0; i < 10; ++i) {
    lock.lock_quiet();
    lock.note_free(0x2000 + i);
    lock.unlock();
  }
  VCHECK(lock.epoch() != seen);
  VCHECK(lock.still_valid(seen, 0x1000, &now));
  VCHECK_EQ(now, lock.epoch());
  VCHECK(!lock.still_valid(seen, 0x2005, &now));   // one of the ten
  VCHECK(lock.still_valid(now, 0x2005, &now));     // remembered after them
  // More than the log holds: not provable, so not valid.
  const uint64_t old = lock.epoch();
  for (uint64_t i = 0; i < 70; ++i) {
    lock.lock_quiet();
    lock.note_free(0x9000 + i);
    lock.unlock();
  }
  VCHECK(!lock.still_valid(old, 0x1000, &now));
  // A full hold voids every answer.
  uint64_t fresh = lock.epoch();
  lock.lock();
  lock.unlock();
  VCHECK(!lock.still_valid(fresh, 0x1000, &now));
  // The count wrapping draws a new random part; an old number is no longer equal to any new one.
  fresh = lock.epoch();
  for (uint64_t i = 0; i < 70000; ++i) {
    lock.lock_quiet();
    lock.note_free(0x40000 + i);
    lock.unlock();
    if (i % 5000 == 0) VCHECK(lock.epoch() != fresh);
  }
  VCHECK(!lock.still_valid(fresh, 0x1000, &now));
}

namespace {
// Readers never see a quiet writer's update half done, and a quiet writer waits for readers, with more
// reader threads than the lock has slots (some share a slot).
void exclusion(bool quiet) {
  BigReaderLock lock;
  int value = 0;
  std::atomic<int> torn{0};
  std::atomic<bool> stop{false};
  std::vector<std::thread> readers;
  for (int t = 0; t < 80; ++t)
    readers.emplace_back([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        SharedGuard g(&lock);
        const int a = value, b = value;
        if (a != b || a % 2) torn.fetch_add(1);
        std::this_thread::yield();
      }
    });
  for (int i = 0; i < 1500; ++i) {
    ExclusiveGuard g(&lock, quiet);
    ++value;
    std::this_thread::yield();
    {
      ExclusiveGuard again(&lock, quiet);
      SharedGuard read(&lock);
      ++value;
    }
  }
  stop = true;
  for (auto& r : readers) r.join();
  VCHECK_EQ(torn.load(), 0);
  VCHECK_EQ(value, 3000);
}
}  // namespace

VTEST(a_quiet_writer_excludes_readers) { exclusion(true); }
VTEST(a_full_writer_excludes_readers) { exclusion(false); }

#endif  // VGPU_MEMALLOC_DIGESTS_ONLY

VTEST_MAIN
