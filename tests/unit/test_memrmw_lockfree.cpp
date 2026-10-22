// MemoryManager::atomic_rmw, the lock-free read-modify-write the AMD engine's atomics use across host threads: a
// compare-and-swap on the word where it is an ordinary one, a refusal (false, nothing done) everywhere else. Held
// against the three calls it stands for -- load_scalar, the operation, store_scalar of the result where it
// differs (or always) -- over many addresses (aligned and not, inside, over the edges, freed, in chunks never
// written, uniform or real, host-mapped), with faults armed or cells stuck; and from several threads, where the
// final values must be the serial result and no update may be lost.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
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

struct Outcome {
  bool failed = false;
  int code = -1;
  std::string text;
  uint64_t v0 = 0;
  bool operator==(const Outcome& o) const { return failed == o.failed && code == o.code && text == o.text && v0 == o.v0; }
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
  } catch (const std::runtime_error& e) {
    o.failed = true;
    o.code = -2;
    o.text = e.what();
  }
  return o;
}

struct Faulty : MemoryManager::AccessFault {
  uint64_t on = 1;
  explicit Faulty(int which) {
    if (which == 1) load_pending = &on;
    if (which == 2) store_pending = &on;
    if (which == 3) stuck_pending = &on;
  }
  uint64_t on_load(uint64_t, uint32_t, uint64_t v) override { return v ^ 0xFF; }
  uint64_t on_store(uint64_t, uint32_t, uint64_t v) override { return v ^ 0xFF; }
  uint64_t on_shared_load(uint64_t, uint32_t, uint64_t v) override { return v; }
  void on_read(uint64_t, uint8_t* bytes, uint64_t len) override {
    if (len) bytes[0] |= 0x80;
  }
  uint64_t on_alu(uint64_t v, uint32_t) override { return v; }
  void on_copy(uint64_t, uint8_t*, uint64_t) override {}
};

uint64_t width_mask(uint32_t size) { return size == 8 ? ~uint64_t{0} : (uint64_t{1} << (8 * size)) - 1; }

// The operations: adds, an exchange, an exclusive or, a minimum, a CAS that sometimes matches, one that leaves the
// value as it is, one that writes a uniform chunk's byte, one that throws.
uint64_t op_result(int code, uint32_t size, uint64_t old, uint64_t b) {
  const uint64_t m = width_mask(size);
  switch (code) {
    case 0: return (old + b) & m;
    case 1: return b & m;
    case 2: return (old ^ b) & m;
    case 3: return old < (b & m) ? old : (b & m);
    case 4: return old == (b & m) ? ((b >> 8) & m) : old;
    case 5: return old;
    case 6: return 0xFFFFFFFFFFFFFFFFull & m;
    default: throw std::runtime_error("the operation fails");
  }
}

// `a`: the lock-free call with the fallback exactly as the interpreter makes it. `b`: the three calls.
Outcome rmw_new(MemoryManager& mm, uint64_t addr, uint32_t size, int code, uint64_t b, bool always, bool* took) {
  return attempt([&](Outcome& o) {
    uint64_t old = 0;
    if (mm.atomic_rmw(addr, size, [&](uint64_t v) { return op_result(code, size, v, b); }, &old, always)) {
      *took = true;
    } else {
      old = mm.load_scalar(addr, size);
      const uint64_t now = op_result(code, size, old, b);
      if (always || now != old) mm.store_scalar(addr, size, now);
    }
    o.v0 = old;
  });
}
Outcome rmw_old(MemoryManager& mm, uint64_t addr, uint32_t size, int code, uint64_t b, bool always) {
  return attempt([&](Outcome& o) {
    const uint64_t old = mm.load_scalar(addr, size);
    const uint64_t now = op_result(code, size, old, b);
    if (always || now != old) mm.store_scalar(addr, size, now);
    o.v0 = old;
  });
}

struct Stats {
  uint64_t ops = 0, failed = 0, fast = 0;
};

Stats run_random(uint64_t seed, int rounds, int fault) {
  MemoryManager ma{1ull << 30}, mb{1ull << 30};
  Faulty fa(fault), fb(fault);
  if (fault) ma.set_access_fault(&fa), mb.set_access_fault(&fb);
  Rng rng{seed};
  std::vector<uint64_t> live, freed, sizes;
  Stats st;
  uint8_t stack_byte = 0;
  const uint64_t host_ptr = reinterpret_cast<uint64_t>(&stack_byte);
  const uint64_t kSizes[] = {8, 16, 100, 4096, 65536, 65537, 3 * kChunkSize, 3 * kChunkSize + 24};
  auto add_alloc = [&] {
    const uint64_t sz = kSizes[rng.below(sizeof kSizes / sizeof kSizes[0])];
    const uint64_t pa = ma.alloc(sz);
    if (pa != mb.alloc(sz)) throw vtest::Failure("different addresses");
    live.push_back(pa), sizes.push_back(sz);
    for (uint64_t off = 0; off + kChunkSize <= sz; off += kChunkSize) {
      const uint64_t k = rng.below(3);
      if (k == 0) {
        const uint8_t pat[1] = {static_cast<uint8_t>(rng.below(2) ? 0xFF : 0x5A)};
        ma.fill(pa + off, pat, 1, kChunkSize), mb.fill(pa + off, pat, 1, kChunkSize);
      } else if (k == 1) {
        std::vector<uint8_t> junk(256);
        for (auto& c : junk) c = static_cast<uint8_t>(rng.next());
        ma.write(pa + off, junk.data(), junk.size()), mb.write(pa + off, junk.data(), junk.size());
      }
    }
  };
  for (int i = 0; i < 5; ++i) add_alloc();
  for (int r = 0; r < rounds; ++r) {
    const uint64_t kind = rng.below(100);
    if (kind < 2) {
      add_alloc();
      continue;
    }
    if (kind < 4 && live.size() > 2) {
      const size_t k = rng.below(live.size());
      ma.free(live[k]), mb.free(live[k]);
      freed.push_back(live[k]);
      live.erase(live.begin() + k), sizes.erase(sizes.begin() + k);
      continue;
    }
    uint64_t addr;
    const uint64_t where = rng.below(100);
    if (where < 60) {
      const size_t k = rng.below(live.size());
      addr = live[k] + (rng.below(3) == 0 ? rng.below(300) : rng.below(sizes[k]));
    } else if (where < 78) {
      const size_t k = rng.below(live.size());
      addr = rng.below(2) ? live[k] + sizes[k] - 12 + rng.below(30)
                          : live[k] + kChunkSize * rng.below(1 + sizes[k] / kChunkSize) - 12 + rng.below(24);
    } else if (where < 88 && !freed.empty()) {
      addr = freed[rng.below(freed.size())] + rng.below(64);
    } else if (where < 94) {
      addr = rng.below(2) ? host_ptr : rng.next();
    } else {
      addr = kDeviceVaBase + rng.below(1 << 24);
    }
    const uint32_t sz[] = {1, 2, 4, 8, 4, 8};
    const uint32_t size = sz[rng.below(6)];
    if (rng.below(10) != 0) addr &= ~uint64_t{size - 1};
    const int code = static_cast<int>(rng.below(8));
    const uint64_t b = rng.next();
    const bool always = rng.below(2);
    bool took = false;
    const Outcome x = rmw_new(ma, addr, size, code, b, always, &took), y = rmw_old(mb, addr, size, code, b, always);
    if (!(x == y)) {
      char buf[400];
      std::snprintf(buf, sizeof buf, "seed %llu round %d: new {failed=%d v=%llx '%.80s'} old {failed=%d v=%llx '%.80s'}",
                    (unsigned long long)seed, r, x.failed, (unsigned long long)x.v0, x.text.c_str(), y.failed,
                    (unsigned long long)y.v0, y.text.c_str());
      throw vtest::Failure(buf);
    }
    ++st.ops;
    st.failed += x.failed;
    st.fast += took;
  }
  ma.set_access_fault(nullptr), mb.set_access_fault(nullptr);
  for (size_t k = 0; k < live.size(); ++k) {
    std::vector<uint8_t> x(sizes[k]), y(sizes[k]);
    ma.read(live[k], x.data(), x.size());
    mb.read(live[k], y.data(), y.size());
    if (x != y) throw vtest::Failure("contents differ in allocation " + std::to_string(k) + ", seed " + std::to_string(seed));
  }
  if (ma.resident_bytes() != mb.resident_bytes())
    throw vtest::Failure("chunks made real differ: new " + std::to_string(ma.resident_bytes()) + " old " +
                         std::to_string(mb.resident_bytes()) + ", seed " + std::to_string(seed));
  VCHECK_EQ(ma.used(), mb.used());
  return st;
}

}  // namespace

VTEST(lock_free_read_modify_write_matches_the_calls_it_stands_for) {
  Stats total;
  for (uint64_t seed = 1; seed <= 40; ++seed) {
    const Stats s = run_random(seed * 7919, 3000, 0);
    total.ops += s.ops, total.failed += s.failed, total.fast += s.fast;
  }
  VCHECK(total.ops > 50000);
  VCHECK(total.failed > 3000);
  VCHECK(total.fast > 5000);   // the compare-and-swap road was taken a good many times
}

VTEST(lock_free_read_modify_write_matches_with_faults_armed_or_stuck_cells) {
  for (int fault = 1; fault <= 3; ++fault)
    for (uint64_t seed = 1; seed <= 8; ++seed) run_random(seed * 104729 + fault, 2000, fault);
}

VTEST(lock_free_read_modify_write_declines_without_doing_anything) {
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(4096);
  uint64_t old = 77;
  bool ran = false;
  auto spy = [&](uint64_t v) { ran = true; return v; };
  VCHECK(mm.atomic_rmw(p, 4, spy, &old));
  VCHECK(ran);
  ran = false;
  old = 77;
  VCHECK(!mm.atomic_rmw(p + 2, 4, spy, &old));      // misaligned
  VCHECK(!mm.atomic_rmw(p, 2, spy, &old));          // a width it does not do
  VCHECK(!mm.atomic_rmw(p + 4092, 8, spy, &old));   // runs past the end
  VCHECK(!mm.atomic_rmw(p - 8, 8, spy, &old));      // before the start
  mm.free(p);
  VCHECK(!mm.atomic_rmw(p, 4, spy, &old));          // freed
  VCHECK(!ran);
  VCHECK_EQ(old, 77u);
}

VTEST(lock_free_read_modify_write_from_several_threads_adds_up) {
  // Six threads, no lock: each adds to words of both widths -- in a chunk never written (the first touch is part of the
  // race), a uniform chunk, a real one -- takes the maximum of a word, xors another, and the final values are the sums.
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(4 * kChunkSize);
  const uint8_t pat[1] = {0};
  mm.fill(p + kChunkSize, pat, 1, kChunkSize);   // uniform
  const uint8_t junk[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  mm.write(p + 2 * kChunkSize, junk, 8);         // real
  const uint64_t add4[] = {p + 8, p + kChunkSize + 8, p + 2 * kChunkSize + 8};
  const uint64_t add8[] = {p + 16, p + kChunkSize + 16, p + 2 * kChunkSize + 16, p + 3 * kChunkSize};
  const uint64_t xor4 = p + 3 * kChunkSize + 8, max8 = p + 3 * kChunkSize + 16;
  const int kThreads = 6, kIters = 30000;
  std::vector<std::thread> ts;
  std::atomic<int> go{0};
  for (int t = 0; t < kThreads; ++t)
    ts.emplace_back([&, t] {
      while (!go.load()) {
      }
      for (int i = 0; i < kIters; ++i) {
        uint64_t old = 0;
        for (uint64_t a : add4) {
          if (!mm.atomic_rmw(a, 4, [&](uint64_t v) { return (v + 1 + t) & 0xFFFFFFFFull; }, &old)) throw std::runtime_error("declined");
        }
        for (uint64_t a : add8) {
          if (!mm.atomic_rmw(a, 8, [&](uint64_t v) { return v + 1 + t; }, &old)) throw std::runtime_error("declined");
        }
        if (!mm.atomic_rmw(xor4, 4, [&](uint64_t v) { return v ^ static_cast<uint64_t>(i * 7 + t); }, &old)) throw std::runtime_error("declined");
        const uint64_t cand = static_cast<uint64_t>(i) * 8 + t;
        if (!mm.atomic_rmw(max8, 8, [&](uint64_t v) { return v > cand ? v : cand; }, &old)) throw std::runtime_error("declined");
      }
    });
  go.store(1);
  for (auto& t : ts) t.join();
  uint64_t sum = 0, x = 0;
  for (int t = 0; t < kThreads; ++t) {
    sum += static_cast<uint64_t>(kIters) * (1 + t);
    for (int i = 0; i < kIters; ++i) x ^= static_cast<uint64_t>(i * 7 + t);
  }
  for (uint64_t a : add4) VCHECK_EQ(mm.load_scalar(a, 4), sum & 0xFFFFFFFFull);
  for (uint64_t a : add8) VCHECK_EQ(mm.load_scalar(a, 8), sum);
  VCHECK_EQ(mm.load_scalar(xor4, 4), x & 0xFFFFFFFFull);
  VCHECK_EQ(mm.load_scalar(max8, 8), static_cast<uint64_t>(kIters - 1) * 8 + (kThreads - 1));
}

VTEST(an_update_that_changes_nothing_stores_nothing_but_always_store_makes_the_chunk) {
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(2 * kChunkSize);
  uint64_t old = 5;
  VCHECK(mm.atomic_rmw(p, 4, [](uint64_t v) { return v; }, &old));   // zero stays zero: the chunk is not made
  VCHECK_EQ(old, 0u);
  VCHECK_EQ(mm.resident_bytes(), 0u);
  VCHECK(mm.atomic_rmw(p, 4, [](uint64_t v) { return v; }, &old, /*always_store=*/true));
  VCHECK(mm.resident_bytes() > 0);
  VCHECK(mm.atomic_rmw(p + kChunkSize, 4, [](uint64_t v) { return v + 1; }, &old));
  VCHECK_EQ(mm.load_scalar(p + kChunkSize, 4), 1u);
}

// ---------------------------------------------------------------------------------------------------------------
// load_quad_run / store_quad_run against load_quad / store_quad one value at a time.
// ---------------------------------------------------------------------------------------------------------------
VTEST(quad_runs_do_what_the_quads_one_at_a_time_do) {
  uint64_t taken = 0, declined = 0;
  for (int fault = 0; fault <= 3; ++fault)
    for (uint64_t seed = 1; seed <= 12; ++seed) {
      MemoryManager ma{1ull << 30}, mb{1ull << 30};
      Faulty fa(fault), fb(fault);
      if (fault) ma.set_access_fault(&fa), mb.set_access_fault(&fb);
      Rng rng{seed * 977 + fault};
      const uint64_t sizes[] = {64, 4096, kChunkSize, 3 * kChunkSize + 48, 3 * kChunkSize + 4096};
      std::vector<uint64_t> live, sz;
      for (uint64_t n : sizes) {
        const uint64_t pa = ma.alloc(n);
        if (pa != mb.alloc(n)) throw vtest::Failure("different addresses");
        live.push_back(pa), sz.push_back(n);
        for (uint64_t off = 0; off + kChunkSize <= n; off += kChunkSize) {
          const uint64_t k = rng.below(3);
          if (k == 0) {
            const uint8_t pat[1] = {static_cast<uint8_t>(rng.below(2) ? 0xFF : 0x00)};
            ma.fill(pa + off, pat, 1, kChunkSize), mb.fill(pa + off, pat, 1, kChunkSize);
          } else if (k == 1) {
            std::vector<uint8_t> junk(300);
            for (auto& c : junk) c = static_cast<uint8_t>(rng.next());
            ma.write(pa + off, junk.data(), junk.size()), mb.write(pa + off, junk.data(), junk.size());
          }
        }
      }
      for (int round = 0; round < 1500; ++round) {
        const size_t k = rng.below(live.size());
        const uint64_t count = 1 + rng.below(64);
        uint64_t addr = live[k] + (rng.below(4) == 0 ? sz[k] - 16 * rng.below(70) : rng.below(sz[k]));
        if (rng.below(8) != 0) addr &= ~uint64_t{15};
        if (rng.below(40) == 0) addr = kDeviceVaBase + rng.below(1 << 20);
        const bool store = rng.below(2);
        std::vector<uint64_t> vals(2 * count), got_a(2 * count, 7), got_b(2 * count, 7);
        for (auto& v : vals) v = rng.below(3) == 0 ? 0 : rng.next();
        if (rng.below(6) == 0)
          for (uint64_t i = 0; i < count; ++i) vals[2 * i] = vals[2 * i + 1] = 0x0101010101010101ull * (rng.below(2) ? 0xFF : 0);
        // The run, with the fallback the interpreter makes: the quads one at a time.
        Outcome x = attempt([&](Outcome& o) {
          if (store) {
            if (ma.store_quad_run(addr, count, vals.data())) ++taken;
            else {
              ++declined;
              for (uint64_t i = 0; i < count; ++i) ma.store_quad(addr + 16 * i, vals.data() + 2 * i);
            }
          } else {
            if (ma.load_quad_run(addr, count, got_a.data())) ++taken;
            else {
              ++declined;
              for (uint64_t i = 0; i < count; ++i) ma.load_quad(addr + 16 * i, got_a.data() + 2 * i);
            }
          }
          o.v0 = 0;
        });
        Outcome y = attempt([&](Outcome& o) {
          for (uint64_t i = 0; i < count; ++i) {
            if (store) mb.store_quad(addr + 16 * i, vals.data() + 2 * i);
            else mb.load_quad(addr + 16 * i, got_b.data() + 2 * i);
          }
          o.v0 = 0;
        });
        if (!(x == y) || got_a != got_b) throw vtest::Failure("quad run differs, seed " + std::to_string(seed) + " round " + std::to_string(round) + " fault " + std::to_string(fault) + ": {" + x.text + "} vs {" + y.text + "}");
      }
      ma.set_access_fault(nullptr), mb.set_access_fault(nullptr);
      for (size_t k = 0; k < live.size(); ++k) {
        std::vector<uint8_t> p(sz[k]), q(sz[k]);
        ma.read(live[k], p.data(), p.size());
        mb.read(live[k], q.data(), q.size());
        if (p != q) throw vtest::Failure("contents differ, seed " + std::to_string(seed));
      }
      VCHECK_EQ(ma.resident_bytes(), mb.resident_bytes());
    }
  VCHECK(taken > 1000);
  VCHECK(declined > 300);
}

VTEST_MAIN
