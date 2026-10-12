// The memory manager's lookup cache against its general path.
//
// A kernel's every scalar load and store looks its address up in the device's
// allocation table (MemoryManager::resolve_scalar). Each thread keeps the last
// few allocations it found, with the table lock's epoch, so that an access
// inside one needs neither the lock nor the table's walk. That is a second way
// to answer a question the table answers, and a second place to be wrong, so
// this runs every operation twice -- on a manager with the cache and on one
// without (set_lookup_cache(false), which VGPU_LOOKUP_CACHE=0 does for a whole
// run) -- and requires the same value, the same error code and the same error
// text, for accesses inside an allocation, past its end, before its start, in
// its padding, in freed memory, outside every device window, misaligned, and
// armed with a fault. Concurrent accesses beside allocations and frees must add
// up to the checksum computed without any of it.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"
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

// What an operation did: its value, or the error it raised.
struct Outcome {
  bool failed = false;
  int code = -1;
  std::string text;
  uint64_t v0 = 0, v1 = 0;
  bool operator==(const Outcome& o) const {
    return failed == o.failed && code == o.code && text == o.text && v0 == o.v0 && v1 == o.v1;
  }
};

template <typename F>
Outcome attempt(F&& f) {
  Outcome o;
  try {
    f(o);
  } catch (const Error& e) {
    o.failed = true;
    o.code = static_cast<int>(e.code());
    o.text = e.what();
  }
  return o;
}

// A fault armed on loads: every scalar load comes back with its low byte inverted.
struct FlipLoads : MemoryManager::AccessFault {
  uint64_t armed = 1;
  FlipLoads() { load_pending = &armed; }
  uint64_t on_load(uint64_t, uint32_t, uint64_t value) override { return value ^ 0xFF; }
  uint64_t on_store(uint64_t, uint32_t, uint64_t value) override { return value; }
  uint64_t on_shared_load(uint64_t, uint32_t, uint64_t value) override { return value; }
  void on_read(uint64_t, uint8_t*, uint64_t) override {}
  uint64_t on_alu(uint64_t value, uint32_t) override { return value; }
  void on_copy(uint64_t, uint8_t*, uint64_t) override {}
};

// The same random operations against two managers; returns how many failed (so
// a test can see it exercised the error paths) and throws on the first difference.
struct Pair {
  MemoryManager a{1ull << 30}, b{1ull << 30};
  Pair(uint64_t page) {
    a.set_lookup_cache(true);
    b.set_lookup_cache(false);
    if (page) a.set_page_size(page), b.set_page_size(page);
  }
};

struct Stats {
  uint64_t ops = 0, failed = 0;
};

Stats run_random(uint64_t seed, uint64_t page, int rounds, bool fault) {
  Pair p(page);
  FlipLoads flip_a, flip_b;
  if (fault) p.a.set_access_fault(&flip_a), p.b.set_access_fault(&flip_b);
  Rng rng{seed};
  std::vector<uint64_t> live, freed;
  std::vector<uint64_t> sizes;
  Stats st;
  uint8_t stack_byte = 0;
  const uint64_t host_ptr = reinterpret_cast<uint64_t>(&stack_byte);
  auto same = [&](const char* what, uint64_t i, const Outcome& x, const Outcome& y) {
    if (!(x == y)) {
      char buf[600];
      std::snprintf(buf, sizeof buf, "seed %llu op %llu (%s): cached {failed=%d code=%d v=%llx,%llx '%.120s'} general {failed=%d code=%d v=%llx,%llx '%.120s'}",
                    (unsigned long long)seed, (unsigned long long)i, what, x.failed, x.code,
                    (unsigned long long)x.v0, (unsigned long long)x.v1, x.text.c_str(), y.failed, y.code,
                    (unsigned long long)y.v0, (unsigned long long)y.v1, y.text.c_str());
      throw vtest::Failure(buf);
    }
    ++st.ops;
    st.failed += x.failed;
  };
  const uint64_t kSizes[] = {1, 3, 4, 7, 8, 16, 100, 255, 256, 257, 4096, 65535, 65536, 65537, 200000};
  for (int i = 0; i < 6; ++i) {
    const uint64_t sz = kSizes[rng.below(sizeof kSizes / sizeof kSizes[0])];
    const uint64_t pa = p.a.alloc(sz), pb = p.b.alloc(sz);
    if (pa != pb) throw vtest::Failure("managers handed out different addresses");
    live.push_back(pa), sizes.push_back(sz);
  }
  for (int r = 0; r < rounds; ++r) {
    const uint64_t kind = rng.below(100);
    if (kind < 3) {   // allocate
      const uint64_t sz = kSizes[rng.below(sizeof kSizes / sizeof kSizes[0])];
      uint64_t pa = p.a.alloc(sz);
      if (pa != p.b.alloc(sz)) throw vtest::Failure("alloc addresses differ");
      live.push_back(pa), sizes.push_back(sz);
      continue;
    }
    if (kind < 6 && live.size() > 2) {   // free one, sometimes freeing a stale or interior pointer
      const size_t k = rng.below(live.size());
      const uint64_t target = live[k];
      same("free", r, attempt([&](Outcome&) { p.a.free(target); }), attempt([&](Outcome&) { p.b.free(target); }));
      freed.push_back(target);
      live.erase(live.begin() + k), sizes.erase(sizes.begin() + k);
      continue;
    }
    // Pick an address.
    uint64_t addr;
    const uint64_t where = rng.below(100);
    if (where < 55) {   // inside a live allocation
      const size_t k = rng.below(live.size());
      addr = live[k] + rng.below(sizes[k]);
    } else if (where < 70) {   // near an end of a live one: over, at, and under the line
      const size_t k = rng.below(live.size());
      addr = live[k] + sizes[k] - 12 + rng.below(40);
      if (rng.below(4) == 0) addr = live[k] - 1 - rng.below(8);
    } else if (where < 80 && !freed.empty()) {   // freed memory
      addr = freed[rng.below(freed.size())] + rng.below(64);
    } else if (where < 85) {   // outside every window, and a host pointer
      addr = rng.below(2) ? host_ptr : rng.next();
    } else if (where < 90) {   // another device's window
      addr = kDeviceVaBase + kDeviceVaStride * (1 + rng.below(3)) + rng.below(1 << 20);
    } else {   // anywhere in this device's window
      addr = kDeviceVaBase + rng.below(1 << 24);
    }
    const uint32_t sizes_[] = {1, 2, 4, 8};
    const uint32_t size = sizes_[rng.below(4)];
    if (rng.below(8) != 0) addr &= ~uint64_t{size - 1};   // mostly aligned; the rest may not be
    const uint64_t value = rng.next();
    switch (rng.below(5)) {
      case 0:
      case 1:
        same("load", r, attempt([&](Outcome& o) { o.v0 = p.a.load_scalar(addr, size); }),
             attempt([&](Outcome& o) { o.v0 = p.b.load_scalar(addr, size); }));
        break;
      case 2:
        same("store", r, attempt([&](Outcome&) { p.a.store_scalar(addr, size, value); }),
             attempt([&](Outcome&) { p.b.store_scalar(addr, size, value); }));
        break;
      case 3: {
        const uint64_t q[2] = {value, ~value};
        same("quad store", r, attempt([&](Outcome&) { p.a.store_quad(addr & ~15ull, q); }),
             attempt([&](Outcome&) { p.b.store_quad(addr & ~15ull, q); }));
        break;
      }
      default:
        same("quad load", r,
             attempt([&](Outcome& o) { uint64_t q[2]; p.a.load_quad(addr & ~15ull, q); o.v0 = q[0], o.v1 = q[1]; }),
             attempt([&](Outcome& o) { uint64_t q[2]; p.b.load_quad(addr & ~15ull, q); o.v0 = q[0], o.v1 = q[1]; }));
    }
  }
  // What each live allocation holds at the end must be the same too.
  for (size_t k = 0; k < live.size(); ++k) {
    std::vector<uint8_t> x(sizes[k]), y(sizes[k]);
    p.a.set_access_fault(nullptr), p.b.set_access_fault(nullptr);
    p.a.read(live[k], x.data(), x.size());
    p.b.read(live[k], y.data(), y.size());
    if (x != y) throw vtest::Failure("final contents differ in allocation " + std::to_string(k));
  }
  VCHECK_EQ(p.a.used(), p.b.used());
  return st;
}

}  // namespace

VTEST(cache_gives_the_general_paths_values_and_errors) {
  Stats total;
  for (uint64_t seed : {1ull, 2ull, 3ull, 0x9E3779B97F4A7C15ull}) {
    const Stats s = run_random(seed, 0, 40000, false);
    total.ops += s.ops, total.failed += s.failed;
  }
  // The comparison must have covered both the hits and the errors.
  VCHECK(total.ops > 100000);
  VCHECK(total.failed > 5000);
  VCHECK(total.ops - total.failed > 50000);
}

VTEST(cache_matches_with_a_page_size_set) {
  // AMD's pages let a read run on past an allocation's end; the cache does not
  // answer those, so they must come out as they do without it.
  const Stats s = run_random(7, 4096, 40000, false);
  VCHECK(s.failed > 1000);
}

VTEST(cache_matches_with_a_fault_armed_on_loads) {
  const Stats s = run_random(11, 0, 20000, true);
  VCHECK(s.ops > 15000);
}

VTEST(cache_never_serves_a_freed_or_replaced_allocation) {
  MemoryManager mm(1 << 26);
  mm.set_lookup_cache(true);
  const uint64_t p = mm.alloc(4096);
  mm.store_scalar(p + 64, 4, 0x11223344);
  VCHECK_EQ(mm.load_scalar(p + 64, 4), 0x11223344ull);   // now remembered
  mm.free(p);
  bool threw = false;
  try {
    mm.load_scalar(p + 64, 4);
  } catch (const Error& e) {
    threw = e.code() == Err::UseAfterFree;
  }
  VCHECK(threw);   // not served from memory
  // A different allocation never reuses the address (the allocator does not), but a
  // second manager does start at the same one: its allocation must not be answered
  // with the first's.
  MemoryManager other(1 << 26);
  const uint64_t q = other.alloc(4096);
  VCHECK_EQ(q, p);
  other.store_scalar(q + 64, 4, 0x55667788);
  MemoryManager third(1 << 26);
  const uint64_t r = third.alloc(4096);
  third.store_scalar(r + 64, 4, 0xAABBCCDD);
  for (int i = 0; i < 4; ++i) {
    VCHECK_EQ(other.load_scalar(q + 64, 4), 0x55667788ull);
    VCHECK_EQ(third.load_scalar(r + 64, 4), 0xAABBCCDDull);
  }
  // A manager that is destroyed and another made in the same place.
  for (int i = 0; i < 20; ++i) {
    auto m = std::make_unique<MemoryManager>(1 << 26);
    const uint64_t a = m->alloc(256);
    m->store_scalar(a, 8, 0x0100 + i);
    VCHECK_EQ(m->load_scalar(a, 8), uint64_t(0x0100 + i));
  }
  // free_all and a shrinking table.
  const uint64_t x = other.alloc(256), y = other.alloc(256);
  other.store_scalar(x, 8, 1), other.store_scalar(y, 8, 2);
  VCHECK_EQ(other.load_scalar(x, 8) + other.load_scalar(y, 8), 3ull);
  other.free_all();
  threw = false;
  try {
    other.load_scalar(y, 8);
  } catch (const Error& e) {
    threw = e.code() == Err::UseAfterFree;
  }
  VCHECK(threw);
}

VTEST(more_allocations_than_ways_still_answer_correctly) {
  MemoryManager mm(1ull << 28);
  std::vector<uint64_t> ps;
  for (int i = 0; i < 40; ++i) {
    ps.push_back(mm.alloc(1024));
    mm.store_scalar(ps.back() + 8, 8, 0xA000 + i);
  }
  uint64_t sum = 0;
  for (int round = 0; round < 50; ++round)
    for (int i = 0; i < 40; ++i) sum += mm.load_scalar(ps[(i * 7 + round) % 40] + 8, 8);
  uint64_t expect = 0;
  for (int i = 0; i < 40; ++i) expect += 50 * (0xA000 + uint64_t(i));
  VCHECK_EQ(sum, expect);
}

VTEST(concurrent_access_beside_allocation_churn_adds_up) {
  // Each worker owns an allocation and does a deterministic walk of stores and
  // loads over it while another thread allocates and frees (which changes the
  // table under the workers' cached lookups). The sums are computed by formula.
  MemoryManager mm(1ull << 30);
  mm.set_lookup_cache(true);
  constexpr int kWorkers = 6, kWords = 4096, kRounds = 30;
  std::vector<uint64_t> base(kWorkers);
  for (auto& b : base) b = mm.alloc(kWords * 8);
  std::vector<uint64_t> result(kWorkers, 0);
  std::atomic<bool> stop{false};
  std::atomic<int> failures{0};
  std::thread churn([&] {
    Rng rng{99};
    std::vector<uint64_t> mine;
    while (!stop.load()) {
      mine.push_back(mm.alloc(1 + rng.below(100000)));
      if (mine.size() > 64) {
        for (uint64_t q : mine) mm.free(q);
        mine.clear();
      }
    }
    for (uint64_t q : mine) mm.free(q);
  });
  std::vector<std::thread> workers;
  for (int w = 0; w < kWorkers; ++w)
    workers.emplace_back([&, w] {
      try {
        uint64_t sum = 0;
        for (int r = 0; r < kRounds; ++r) {
          for (int i = 0; i < kWords; ++i) mm.store_scalar(base[w] + 8 * i, 8, uint64_t(i) * 31 + w + r);
          for (int i = 0; i < kWords; ++i) sum += mm.load_scalar(base[w] + 8 * i, 8) ^ uint64_t(i);
        }
        result[w] = sum;
      } catch (...) {
        ++failures;
      }
    });
  for (auto& t : workers) t.join();
  stop = true;
  churn.join();
  VCHECK_EQ(failures.load(), 0);
  for (int w = 0; w < kWorkers; ++w) {
    uint64_t expect = 0;
    for (int r = 0; r < kRounds; ++r)
      for (int i = 0; i < kWords; ++i) expect += (uint64_t(i) * 31 + w + r) ^ uint64_t(i);
    VCHECK_EQ(result[w], expect);
  }
}

VTEST_MAIN
