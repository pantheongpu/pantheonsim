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

#include <algorithm>
#include <array>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/mem_counters.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
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

// ---- the counters of a warp's memory access ----------------------------------
//
// The two functions below are the engines' counting as it was before
// vgpu/exec/mem_counters.hpp, copied as it stood (the interpreter's
// count_addresses and the SASS executor's count_access).

namespace {

constexpr uint32_t kMaxWarpSize = 64;

uint32_t old_ptx_banks(const std::array<uint64_t, 64>& addrs, uint64_t m, uint32_t W_, uint64_t span) {
  uint32_t worst = 1;
  for (uint32_t bank = 0; bank < 32; ++bank) {
    std::array<uint64_t, kMaxWarpSize * 4> words{};
    uint32_t distinct = 0;
    for (uint32_t lane = 0; lane < W_; ++lane) {
      if (!(m & (uint64_t{1} << lane))) continue;
      for (uint64_t off = 0; off < span; off += 4) {
        const uint64_t word = (addrs[lane] + off) / 4;
        if (word % 32 != bank) continue;
        bool seen = false;
        for (uint32_t i = 0; i < distinct; ++i)
          if (words[i] == word) { seen = true; break; }
        if (!seen && distinct < words.size()) words[distinct++] = word;
      }
    }
    if (distinct > worst) worst = distinct;
  }
  return worst;
}

uint32_t old_ptx_sectors(const std::array<uint64_t, 64>& addrs, uint64_t m, uint32_t W_, uint64_t span) {
  std::array<uint64_t, kMaxWarpSize * 8> sectors{};
  uint32_t distinct = 0;
  for (uint32_t lane = 0; lane < W_; ++lane) {
    if (!(m & (uint64_t{1} << lane))) continue;
    const uint64_t first = addrs[lane] / 32;
    const uint64_t last = (addrs[lane] + (span ? span - 1 : 0)) / 32;
    for (uint64_t sec = first; sec <= last; ++sec) {
      bool seen = false;
      for (uint32_t i = 0; i < distinct; ++i)
        if (sectors[i] == sec) { seen = true; break; }
      if (!seen && distinct < sectors.size()) sectors[distinct++] = sec;
    }
  }
  return distinct;
}

uint64_t old_sass_sectors(const uint64_t* addr, uint32_t lanes, uint32_t bytes) {
  std::vector<uint64_t> sectors;
  for (unsigned l = 0; l < 32; ++l)
    if (lanes & (1u << l))
      for (uint64_t s = addr[l] / 32; s <= (addr[l] + bytes - 1) / 32; ++s) sectors.push_back(s);
  std::sort(sectors.begin(), sectors.end());
  return static_cast<uint64_t>(std::unique(sectors.begin(), sectors.end()) - sectors.begin());
}

uint64_t old_sass_passes(const uint64_t* addr, uint32_t lanes, uint32_t bytes) {
  std::vector<uint64_t> words[32];
  for (unsigned l = 0; l < 32; ++l)
    if (lanes & (1u << l))
      for (uint64_t a = addr[l] & ~3ull; a < addr[l] + bytes; a += 4) words[(a / 4) % 32].push_back(a / 4);
  uint64_t passes = 1;
  for (auto& v : words) {
    std::sort(v.begin(), v.end());
    passes = std::max<uint64_t>(passes, static_cast<uint64_t>(std::unique(v.begin(), v.end()) - v.begin()));
  }
  return passes;
}

// Addresses in the shapes kernels make: unit stride, strided, reversed, a
// broadcast, a gather, a transpose column, unaligned, and the extremes.
void fill_addresses(Rng& rng, std::array<uint64_t, 64>& a, uint32_t lanes) {
  const uint64_t base = rng.below(8) == 0 ? ~uint64_t{0} - rng.below(4096) : rng.below(1ull << 40);
  const uint64_t pattern = rng.below(11);
  const uint64_t stride_choices[] = {1, 2, 4, 8, 12, 16, 32, 33 * 4, 64, 128, 132, 256, 1024, 4100};
  const uint64_t stride = stride_choices[rng.below(sizeof stride_choices / sizeof stride_choices[0])];
  for (uint32_t l = 0; l < 64; ++l) {
    switch (pattern) {
      case 0: a[l] = base + l * 4; break;
      case 1: a[l] = base + l * stride; break;
      case 2: a[l] = base + (lanes - 1 - l % lanes) * stride; break;
      case 3: a[l] = base; break;
      case 4: a[l] = rng.next() % (1 << 20); break;
      case 5: a[l] = base + (l % 8) * stride + (l / 8) * 4; break;
      case 6: a[l] = base + l * stride + (rng.below(2) ? 1 : 3); break;   // unaligned
      case 7: a[l] = (base & ~3ull) + (l % 4) * 4096 + (l / 4) * 4; break;
      // A thread-private array indexed by a value of each lane's own: a few sectors, in no order.
      case 8: a[l] = (base & ~3ull) + rng.below(64) * 4; break;
      // Strictly ascending for the first lanes, then back over sectors already seen and on to new ones.
      case 9: a[l] = (base & ~3ull) + (l < 12 ? l * 64 : (l * 7 % 20) * 64); break;
      default: a[l] = rng.next(); break;
    }
  }
}

}  // namespace

VTEST(counters_of_a_warps_access_match_the_earlier_counting) {
  Rng rng{2026};
  uint64_t checked = 0;
  for (int i = 0; i < 300000; ++i) {
    const uint32_t lanes = rng.below(5) == 0 ? 64 : 32;
    std::array<uint64_t, 64> a;
    fill_addresses(rng, a, lanes);
    uint64_t mask = rng.below(4) == 0 ? ~uint64_t{0} : rng.next() & rng.next();
    if (lanes == 32) mask &= 0xFFFFFFFFull;
    const uint64_t bytes_choices[] = {1, 2, 4, 8, 16};
    const uint64_t count_choices[] = {1, 2, 4};
    uint64_t span = bytes_choices[rng.below(5)] * count_choices[rng.below(3)];
    if (rng.below(40) == 0) span = 1 + rng.below(400);   // past what the fast versions are shaped for
    if (rng.below(200) == 0) span = 0;
    // The PTX interpreter's counting (any warp size up to 64).
    VCHECK_EQ(exec::memcount::ptx_bank_passes(a.data(), mask, lanes, span), old_ptx_banks(a, mask, lanes, span));
    VCHECK_EQ(exec::memcount::distinct_sectors(a.data(), mask, lanes, span, kMaxWarpSize * 8),
              old_ptx_sectors(a, mask, lanes, span));
    // The SASS executor's (32 lanes, 1..32 byte accesses, as exec_mem makes them).
    const uint32_t m32 = static_cast<uint32_t>(mask);
    const uint32_t sb = span ? static_cast<uint32_t>(std::min<uint64_t>(span, 200)) : 1;
    bool sane = m32 != 0;   // SASS counting of an address in the top 256 bytes of the space never ends
    for (int l = 0; l < 32; ++l) sane = sane && (!(m32 >> l & 1) || a[l] <= UINT64_MAX - 256);
    if (sane) {
      VCHECK_EQ(exec::memcount::sass_bank_passes(a.data(), m32, 32, sb), old_sass_passes(a.data(), m32, sb));
      VCHECK_EQ(exec::memcount::distinct_sectors(a.data(), m32, 32, sb, SIZE_MAX), old_sass_sectors(a.data(), m32, sb));
    }
    // The references kept in the header (what VGPU_FASTCOUNT=0 runs) are that earlier code too.
    VCHECK_EQ(exec::memcount::ptx_bank_passes_reference(a.data(), mask, lanes, span), old_ptx_banks(a, mask, lanes, span));
    VCHECK_EQ(exec::memcount::sectors_reference(a.data(), mask, lanes, span, kMaxWarpSize * 8),
              old_ptx_sectors(a, mask, lanes, span));
    if (sane) {
      VCHECK_EQ(exec::memcount::sass_bank_passes_reference(a.data(), m32, 32, sb), old_sass_passes(a.data(), m32, sb));
      VCHECK_EQ(exec::memcount::sass_sectors_reference(a.data(), m32, 32, sb), old_sass_sectors(a.data(), m32, sb));
    }
    ++checked;
  }
  VCHECK(checked == 300000);
}

VTEST(known_access_shapes_count_as_the_hardware_documents) {
  // 32 lanes reading consecutive floats: 4 sectors, no bank conflict; a column of a
  // 32-wide tile: 32 sectors and a 32-way conflict, which padding to 33 removes.
  std::array<uint64_t, 64> a{};
  for (int l = 0; l < 32; ++l) a[l] = 0x1000 + 4 * l;
  VCHECK_EQ(exec::memcount::distinct_sectors(a.data(), 0xFFFFFFFFull, 32, 4, 512), 4u);
  VCHECK_EQ(exec::memcount::ptx_bank_passes(a.data(), 0xFFFFFFFFull, 32, 4), 1u);
  for (int l = 0; l < 32; ++l) a[l] = 128 * l;
  VCHECK_EQ(exec::memcount::distinct_sectors(a.data(), 0xFFFFFFFFull, 32, 4, 512), 32u);
  VCHECK_EQ(exec::memcount::ptx_bank_passes(a.data(), 0xFFFFFFFFull, 32, 4), 32u);
  VCHECK_EQ(exec::memcount::sass_bank_passes(a.data(), 0xFFFFFFFFu, 32, 4), 32u);
  for (int l = 0; l < 32; ++l) a[l] = 132 * l;
  VCHECK_EQ(exec::memcount::ptx_bank_passes(a.data(), 0xFFFFFFFFull, 32, 4), 1u);
  for (int l = 0; l < 32; ++l) a[l] = 0x40;   // a broadcast
  VCHECK_EQ(exec::memcount::ptx_bank_passes(a.data(), 0xFFFFFFFFull, 32, 4), 1u);
  VCHECK_EQ(exec::memcount::distinct_sectors(a.data(), 0xFFFFFFFFull, 32, 4, 512), 1u);
  VCHECK_EQ(exec::memcount::distinct_sectors(a.data(), 0, 32, 4, 512), 0u);
}

// ---- loads and stores of a whole warp, through the kernel -----------------------
//
// A kernel's ld.global / st.global go to the memory manager a warp at a time
// (load_batch / store_batch) when every access is ordinary, and lane by lane
// otherwise. Each case below runs the same kernel with the lookup cache on (so
// batches happen) and off (VGPU_LOOKUP_CACHE=0: every access one at a time) and
// requires the same memory afterwards, the same counters, and the same error
// text, for: plain accesses, every width, vectors, lanes predicated off, lanes
// writing the same address, a lane past the end of its buffer, a misaligned lane,
// a freed buffer, a lane at a null pointer, and a store into untouched memory.

namespace {

std::vector<uint8_t> arg64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

struct Ran {
  std::string error;
  std::vector<uint8_t> out;
  std::vector<uint64_t> counters;
};

// `ty` is the PTX type of the access, `vec` the vector width (1, 2 or 4),
// `offsets` each thread's byte offset into both buffers (the table is read
// through the same path). Thread t is predicated off when `off_mask` has bit t.
Ran run_ldst(bool cache, const std::string& ty, int vec, const std::vector<uint64_t>& offsets,
             uint64_t off_mask, bool free_input, uint64_t buf_bytes) {
  setenv("VGPU_LOOKUP_CACHE", cache ? "1" : "0", 1);
  setenv("VGPU_THREADS", "1", 1);
  const bool is64 = ty == "u64";
  std::string vals;
  for (int i = 0; i < vec; ++i) vals += (i ? ", " : "") + std::string(is64 ? "%rd1" : "%r1") + std::to_string(i + 1);
  const std::string vreg = vec == 1 ? vals : "{" + vals + "}";
  const std::string vec_s = vec == 1 ? "" : ".v" + std::to_string(vec);
  std::string ptx = ".version 8.3\n.target sm_90\n.address_size 64\n.visible .entry k(.param .u64 pin, .param .u64 pout, .param .u64 ptbl, .param .u64 pmask)\n{\n"
    " .reg .pred %p<4>; .reg .b32 %r<16>; .reg .b64 %rd<24>;\n"
    " ld.param.u64 %rd1,[pin]; ld.param.u64 %rd2,[pout]; ld.param.u64 %rd3,[ptbl]; ld.param.u64 %rd4,[pmask];\n"
    " mov.u32 %r1,%tid.x; mul.wide.u32 %rd5,%r1,8; add.u64 %rd6,%rd3,%rd5; ld.global.u64 %rd7,[%rd6];\n"
    " shr.u64 %rd8,%rd4,%r1; and.b64 %rd8,%rd8,1; setp.eq.u64 %p1,%rd8,0;\n"
    " ld.global.u64 %rd15,[%rd6+256];\n"
    " add.u64 %rd9,%rd1,%rd7; add.u64 %rd10,%rd2,%rd15;\n";
  // %rd11.. / %r11.. are the data registers named in `vals`.
  const std::string t = ty == "f32" ? "f32" : ty;
  ptx += " @%p1 ld.global." + t + vec_s + " " + vreg + ", [%rd9];\n";
  ptx += " @%p1 st.global." + t + vec_s + " [%rd10], " + vreg + ";\n ret;\n}\n";
  Ran r;
  MemoryManager mem{1 << 26};
  DeviceProfile prof = load_gpu("nvidia/a10");
  auto m = ptx::parse(ptx);
  const uint64_t in = mem.alloc(buf_bytes), out = mem.alloc(buf_bytes), tbl = mem.alloc(512),
                 untouched = mem.alloc(1 << 20);
  (void)untouched;
  std::vector<uint8_t> init(buf_bytes);
  for (size_t i = 0; i < init.size(); ++i) init[i] = static_cast<uint8_t>(i * 29 + 5);
  mem.write(in, init.data(), init.size());
  // The table: each thread's input offset, then (32 entries on) its output offset. Outputs
  // collide where `offsets` repeat, and are scrambled so lane order decides who wins.
  std::vector<uint64_t> table(64, 0);
  for (size_t i = 0; i < offsets.size() && i < 32; ++i) {
    table[i] = offsets[i];
    table[32 + i] = offsets[(i * 7 + i / 4) % offsets.size()];
  }
  mem.write(tbl, table.data(), table.size() * 8);
  uint64_t in_arg = in;
  if (free_input) {
    const uint64_t dead = mem.alloc(buf_bytes);
    mem.write(dead, init.data(), init.size());
    mem.free(dead);
    in_arg = dead;
  }
  exec::LaunchConfig cfg;
  cfg.block = {static_cast<uint32_t>(offsets.size()), 1, 1};
  try {
    const exec::LaunchStats st = exec::launch(m.entries[0], cfg, {arg64(in_arg), arg64(out), arg64(tbl), arg64(off_mask)}, mem, prof);
    r.counters = {st.instructions, st.thread_instructions, st.global_loads, st.global_stores, st.global_bytes_read,
                  st.global_bytes_written, st.global_sectors, st.global_requests, st.global_sectors_ld,
                  st.global_sectors_st};
  } catch (const Error& e) {
    r.error = e.what();
  }
  r.out.resize(buf_bytes);
  mem.read(out, r.out.data(), r.out.size());
  unsetenv("VGPU_LOOKUP_CACHE");
  unsetenv("VGPU_THREADS");
  return r;
}

void same_ldst(const char* what, const std::string& ty, int vec, const std::vector<uint64_t>& offs, uint64_t off_mask,
               bool free_input, uint64_t buf_bytes) {
  const Ran a = run_ldst(true, ty, vec, offs, off_mask, free_input, buf_bytes);
  const Ran b = run_ldst(false, ty, vec, offs, off_mask, free_input, buf_bytes);
  if (a.error != b.error || a.out != b.out || a.counters != b.counters)
    throw vtest::Failure(std::string(what) + " (" + ty + " x" + std::to_string(vec) + "): batched {" + a.error +
                         "} vs lane by lane {" + b.error + "}, memory " + (a.out == b.out ? "same" : "differs") +
                         ", counters " + (a.counters == b.counters ? "same" : "differ"));
}

}  // namespace

VTEST(warp_wide_loads_and_stores_match_lane_by_lane) {
  struct T { const char* ty; int bytes; };
  const T types[] = {{"u8", 1}, {"u16", 2}, {"u32", 4}, {"f32", 4}, {"u64", 8}};
  const uint64_t kBuf = 8192;
  for (const T& t : types)
    for (int vec : {1, 2, 4}) {
      if (std::string(t.ty) == "u8" && vec > 1) continue;
      const uint64_t span = uint64_t(t.bytes) * vec;
      std::vector<uint64_t> unit, strided, same_addr, scattered, oob, misaligned, past, null_lane;
      for (int i = 0; i < 32; ++i) {
        unit.push_back(i * span);
        strided.push_back(((i * 7) % 32) * span * 2);
        same_addr.push_back((i % 4) * span);
        scattered.push_back(((i * 13 + 5) % 64) * span);
        oob.push_back(i == 17 ? kBuf : i * span);
        misaligned.push_back(i == 9 ? i * span + 1 : i * span);
        past.push_back(i == 31 ? kBuf - span + 1 : i * span);
        null_lane.push_back(i == 3 ? uint64_t(0) - 0x2000'0000'0000ull : i * span);   // far outside every allocation
      }
      const uint64_t all = 0xFFFFFFFFull;
      same_ldst("unit stride", t.ty, vec, unit, all, false, kBuf);
      same_ldst("strided", t.ty, vec, strided, all, false, kBuf);
      same_ldst("lanes writing one address", t.ty, vec, same_addr, all, false, kBuf);
      same_ldst("scattered", t.ty, vec, scattered, all, false, kBuf);
      same_ldst("lanes predicated off", t.ty, vec, unit, 0xA5A5A5A5ull, false, kBuf);
      same_ldst("one lane past the end", t.ty, vec, oob, all, false, kBuf);
      same_ldst("one lane misaligned", t.ty, vec, misaligned, all, false, kBuf);
      same_ldst("one lane straddling the end", t.ty, vec, past, all, false, kBuf);
      same_ldst("one lane at a wild address", t.ty, vec, null_lane, all, false, kBuf);
      same_ldst("freed input", t.ty, vec, unit, all, true, kBuf);
    }
}

// ---- the batch calls themselves, against one-by-one ---------------------------

VTEST(batch_calls_do_what_one_by_one_calls_do_or_decline_untouched) {
  MemoryManager a(1ull << 28), b(1ull << 28);
  a.set_lookup_cache(true);
  b.set_lookup_cache(false);
  FlipLoads flip;
  a.set_access_fault(&flip);
  const uint64_t sizes[] = {70000, 4096, 300000};
  std::vector<uint64_t> base;
  for (uint64_t sz : sizes) {
    base.push_back(a.alloc(sz));
    if (b.alloc(sz) != base.back()) throw vtest::Failure("addresses differ");
  }
  // Some memory uniform, some untouched, some written.
  const uint8_t pat[1] = {0xA5};
  a.fill(base[2], pat, 1, 3 * kChunkSize), b.fill(base[2], pat, 1, 3 * kChunkSize);
  std::vector<uint8_t> junk(4096);
  for (size_t i = 0; i < junk.size(); ++i) junk[i] = uint8_t(i * 7 + 1);
  a.write(base[0], junk.data(), junk.size()), b.write(base[0], junk.data(), junk.size());
  Rng rng{424242};
  uint64_t batched_loads = 0, batched_stores = 0, declined = 0;
  auto same_memory = [&](int step) {
    for (int k = 0; k < 3; ++k) {
      std::vector<uint8_t> x(sizes[k]), y(sizes[k]);
      flip.armed = 0;
      a.read(base[k], x.data(), x.size());
      flip.armed = 1;
      b.read(base[k], y.data(), y.size());
      if (x != y) throw vtest::Failure("memory differs after step " + std::to_string(step));
    }
  };
  for (int step = 0; step < 6000; ++step) {
    const uint32_t sz = 1u << rng.below(4);
    const size_t count = 1 + rng.below(40);
    const size_t k = rng.below(3);
    std::vector<uint64_t> addrs(count), vals(count);
    const uint64_t mode = rng.below(11);
    for (size_t i = 0; i < count; ++i) {
      uint64_t off;
      if (mode < 5) off = (i * sz * (1 + mode)) % sizes[k];                     // regular
      else if (mode < 7) off = rng.below(sizes[k]) & ~uint64_t{sz - 1};          // scattered
      else if (mode == 7) off = (i == count / 2) ? sizes[k] - sz + (rng.below(2) ? 1 : 0) : i * sz;   // edge/misaligned
      else if (mode == 8) off = kChunkSize - sz * (count / 2) + i * sz;          // straddling a chunk
      else if (mode == 10) off = i * sz + (i == count / 3 ? 1 : 0);              // one lane misaligned, inside the allocation
      else off = (i % 3) * sz;                                                   // repeats
      addrs[i] = base[k] + off;
      if (mode == 9 && i == 2) addrs[i] = base[(k + 1) % 3] + 8;                 // another allocation
      vals[i] = rng.next() & (sz == 8 ? ~uint64_t{0} : (uint64_t{1} << (8 * sz)) - 1);
      if (rng.below(6) == 0) vals[i] = 0xA5A5A5A5A5A5A5A5ull & (sz == 8 ? ~uint64_t{0} : (uint64_t{1} << (8 * sz)) - 1);
    }
    flip.armed = rng.below(7) == 0 ? 1 : 0;   // armed on loads: the batch must decline loads
    if (rng.below(2)) {
      std::vector<uint64_t> got(count, 0xDEAD);
      const bool done = a.load_batch(addrs.data(), count, sz, got.data());
      if (done) {
        ++batched_loads;
        VCHECK(!flip.armed);
        flip.armed = 0;   // b has no fault object armed: it is the reference
        for (size_t i = 0; i < count; ++i) VCHECK_EQ(got[i], b.load_scalar(addrs[i], sz));
      } else {
        ++declined;
        for (uint64_t v : got) VCHECK_EQ(v, 0xDEADull);   // nothing written
      }
    } else {
      flip.armed = 0;
      size_t failed_at = 0;
      bool done = false;
      try { done = a.store_batch(addrs.data(), count, sz, vals.data(), &failed_at); } catch (const Error&) { throw vtest::Failure("store_batch threw"); }
      if (done) {
        ++batched_stores;
        for (size_t i = 0; i < count; ++i) b.store_scalar(addrs[i], sz, vals[i]);
      } else {
        ++declined;
        // Declined with nothing changed: so a lane-by-lane store does the same on both.
        same_memory(step);
        for (size_t i = 0; i < count; ++i) {
          const Outcome x = attempt([&](Outcome&) { a.store_scalar(addrs[i], sz, vals[i]); });
          const Outcome y = attempt([&](Outcome&) { b.store_scalar(addrs[i], sz, vals[i]); });
          VCHECK(x == y);
          if (x.failed) break;
        }
      }
    }
    if (step % 50 == 0) same_memory(step);
  }
  same_memory(-1);
  VCHECK(batched_loads > 500);
  VCHECK(batched_stores > 500);
  VCHECK(declined > 500);
}

VTEST(a_batch_declines_where_a_host_mapping_covers_one_of_its_addresses) {
  // Mapped host memory over part of an allocation is read and written in place, not in the chunks: a batch with an address in it
  // is the one-by-one calls' to make, and one without is not held up by the map.
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(1 << 16);
  std::vector<uint32_t> host(1024, 7);
  mm.map_host(p + 4096, host.data(), 4096);
  uint64_t out[4] = {9, 9, 9, 9};
  const uint64_t vals[4] = {1, 2, 3, 4};
  const uint64_t over[4] = {p, p + 8, p + 4096 + 16, p + 64};
  VCHECK(!mm.load_batch(over, 4, 4, out));
  VCHECK(!mm.store_batch(over, 4, 4, vals));
  VCHECK_EQ(out[2], 9u);
  VCHECK_EQ(host[4], 7u);
  const uint64_t beside[3] = {p, p + 8, p + 8192};
  VCHECK(mm.load_batch(beside, 3, 4, out));
  VCHECK(mm.store_batch(beside, 3, 4, vals));
  VCHECK_EQ(mm.load_scalar(p + 8192, 4), 3u);
  VCHECK_EQ(mm.load_scalar(p + 4096 + 16, 4), 7u);
  mm.unmap_host(p + 4096);
}

// Both faults at once, each armed on its own switch: a load fault and a store fault.
struct FlipBoth : FlipLoads {
  uint64_t store_armed = 0;
  FlipBoth() { store_pending = &store_armed; }
};

VTEST(a_window_for_read_modify_writes_does_what_the_load_and_the_store_batch_do) {
  // A warp's atomics load the words of their addresses, work out the new ones and store them (exec_mem): batch_window once, then batch_load and
  // batch_store, must be the load_batch then store_batch it replaces -- the same acceptance (a fault armed on loads or on stores, a host mapping in
  // reach, a misaligned address, one outside the allocation each decline it), the same values loaded, the same memory afterwards.
  MemoryManager a(1ull << 28), b(1ull << 28);
  FlipBoth fa, fb;
  a.set_access_fault(&fa), b.set_access_fault(&fb);
  const uint64_t pa = a.alloc(1 << 17), pb = b.alloc(1 << 17);
  VCHECK_EQ(pa, pb);
  std::vector<uint32_t> host(2048, 5);
  a.map_host(pa + 8192, host.data(), 4096), b.map_host(pb + 8192, host.data(), 4096);
  const uint8_t pat[1] = {0x3C};
  a.fill(pa + kChunkSize, pat, 1, kChunkSize), b.fill(pb + kChunkSize, pat, 1, kChunkSize);   // (some memory uniform, some untouched)
  Rng rng{99};
  uint64_t windows = 0, declined = 0;
  for (int step = 0; step < 4000; ++step) {
    const uint32_t sz = 1u << rng.below(4);
    const size_t count = 1 + rng.below(32);
    std::vector<uint64_t> addrs(count);
    const unsigned mode = static_cast<unsigned>(rng.below(8));
    for (size_t i = 0; i < count; ++i) {
      uint64_t off = (rng.below(1 << 17) & ~uint64_t{sz - 1});
      if (mode < 3) off = (i * sz * (1 + mode)) % (1 << 15) + 16384;                      // regular, away from the host mapping
      if (mode == 3 && i == count / 2) off = 8192 + sz * rng.below(100);                    // one in the host-mapped range
      if (mode == 4 && i == count / 3) off += 1;                                            // one misaligned (a no-op for size 1)
      if (mode == 5 && i == count - 1) off = (1 << 17) - sz + (rng.below(2) ? sz : 0);      // the last one at, or past, the end
      addrs[i] = pa + off;
    }
    fa.armed = fb.armed = rng.below(5) == 0;
    fa.store_armed = fb.store_armed = rng.below(7) == 0;
    // The reference: the batches the old code made, then the lane loop's stores when the batch declined (here: nothing, only the acceptance).
    std::vector<uint64_t> want(count, 0xDEAD), got(count, 0xDEAD);
    bool ref = b.load_batch(addrs.data(), count, sz, want.data());
    if (ref) {
      for (uint64_t& v : want) v = v * 3 + 1;
      ref = b.store_batch(addrs.data(), count, sz, want.data());   // (declined: nothing stored; the old code then left everything to the loop)
      if (!ref) for (uint64_t& v : want) v = 0xDEAD;
    }
    uint64_t loaded[32];
    MemoryManager::BatchWindow w;
    const bool ok = a.batch_window(addrs.data(), count, sz, &w);
    VCHECK_EQ(ok, ref);
    if (ok) {
      ++windows;
      a.batch_load(w, addrs.data(), count, sz, loaded);
      for (size_t i = 0; i < count; ++i) got[i] = loaded[i] * 3 + 1;
      a.batch_store(w, addrs.data(), count, sz, got.data());
      VCHECK(got == want);
    } else {
      ++declined;
    }
    if (step % 40 == 0) {
      fa.armed = fb.armed = 0, fa.store_armed = fb.store_armed = 0;
      std::vector<uint8_t> x(1 << 17), y(1 << 17);
      a.read(pa, x.data(), x.size()), b.read(pb, y.data(), y.size());
      VCHECK(x == y);
    }
  }
  VCHECK(windows > 400);
  VCHECK(declined > 400);
  fa.armed = fb.armed = 0, fa.store_armed = fb.store_armed = 0;
  a.unmap_host(pa + 8192), b.unmap_host(pb + 8192);
}

VTEST(fill_with_a_repeating_pattern_is_the_pattern_at_every_offset) {
  // fill() repeats a pattern of any length from the phase the range starts at: compare it with
  // the byte-by-byte definition over lengths, phases and chunk boundaries.
  Rng rng{77};
  MemoryManager mm(1ull << 28);
  const uint64_t total = 3 * kChunkSize + 1000;
  const uint64_t p = mm.alloc(total);
  for (int round = 0; round < 300; ++round) {
    const uint32_t plen = 1 + static_cast<uint32_t>(rng.below(9));
    uint8_t pat[16];
    for (auto& b : pat) b = static_cast<uint8_t>(rng.next() | 1);
    const uint64_t start = rng.below(round % 3 == 0 ? 100 : total - 1);
    uint64_t len = 1 + rng.below(total - start);
    if (round % 5 == 0) len = std::min<uint64_t>(len, 40);
    std::vector<uint8_t> expect(total);
    mm.read(p, expect.data(), total);
    for (uint64_t i = 0; i < len; ++i) expect[start + i] = pat[i % plen];
    mm.fill(p + start, pat, plen, len);
    std::vector<uint8_t> got(total);
    mm.read(p, got.data(), total);
    if (got != expect) throw vtest::Failure("fill differs: pattern length " + std::to_string(plen) + ", start " + std::to_string(start) + ", length " + std::to_string(len));
  }
}

VTEST_MAIN
