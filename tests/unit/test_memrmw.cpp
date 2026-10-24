// The memory manager's one-lookup read-modify-write and multi-field read, against the calls they replace.
//
// An atomic's memory side used to be load_scalar, the operation, store_scalar: two lookups of one address.
// MemoryManager::rmw_scalar does the three with one, and load_fields reads the channels of a texel with one
// lookup instead of one load_scalar each. Both decline (returning false, having done nothing) whenever the
// calls they stand for could differ, and the caller then makes those calls; these tests check, over many
// addresses, that whatever they do is exactly what the calls do -- the values, the memory left behind, the
// chunks made real, the errors and their text -- and then that the interpreter's atomics and texture fetches,
// which use them, give the same bits as with the lookup cache off (VGPU_LOOKUP_CACHE=0), which sends every
// access down the old calls.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/texture.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

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
  } catch (const std::runtime_error& e) {   // the operation's own
    o.failed = true;
    o.code = -2;
    o.text = e.what();
  }
  return o;
}

// A fault armed on loads (low byte inverted) or on stores (low byte inverted), or cells stuck.
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
    if (len) bytes[0] |= 0x80;   // a stuck-at-one cell
  }
  uint64_t on_alu(uint64_t v, uint32_t) override { return v; }
  void on_copy(uint64_t, uint8_t*, uint64_t) override {}
};

uint64_t width_mask(uint32_t size) { return size == 8 ? ~uint64_t{0} : (uint64_t{1} << (8 * size)) - 1; }

// What an atomic does to the value it finds, by code: the operations the interpreter has (add, exchange, xor,
// unsigned min, a CAS that sometimes matches), one that leaves the value as it is (so a uniform chunk is not made
// real), one that writes a uniform chunk's own byte, and one that throws after the load.
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

struct Pair {
  MemoryManager a{1ull << 30}, b{1ull << 30};
};

// Manager `a`: the new path with its fallback exactly as the interpreter makes it. Manager `b`: the old calls.
Outcome rmw_new(MemoryManager& mm, uint64_t addr, uint32_t size, int code, uint64_t b) {
  return attempt([&](Outcome& o) {
    uint64_t old = 0;
    if (!mm.rmw_scalar(addr, size, [&](uint64_t v) { return op_result(code, size, v, b); }, &old)) {
      old = mm.load_scalar(addr, size);
      mm.store_scalar(addr, size, op_result(code, size, old, b));
    }
    o.v0 = old;
  });
}
Outcome rmw_old(MemoryManager& mm, uint64_t addr, uint32_t size, int code, uint64_t b) {
  return attempt([&](Outcome& o) {
    const uint64_t old = mm.load_scalar(addr, size);
    mm.store_scalar(addr, size, op_result(code, size, old, b));
    o.v0 = old;
  });
}

// The executors' atomics on one thread store only a result that differs from what they loaded (rmw_if_changed, against the
// two calls with the store left out when nothing changed): a chunk nothing has written stays so.
Outcome changed_new(MemoryManager& mm, uint64_t addr, uint32_t size, int code, uint64_t b) {
  return attempt([&](Outcome& o) {
    uint64_t old = 0;
    if (!mm.rmw_if_changed(addr, size, [&](uint64_t v) { return op_result(code, size, v, b); }, &old)) {
      old = mm.load_scalar(addr, size);
      const uint64_t now = op_result(code, size, old, b);
      if (now != old) mm.store_scalar(addr, size, now);
    }
    o.v0 = old;
  });
}
Outcome changed_old(MemoryManager& mm, uint64_t addr, uint32_t size, int code, uint64_t b) {
  return attempt([&](Outcome& o) {
    const uint64_t old = mm.load_scalar(addr, size);
    const uint64_t now = op_result(code, size, old, b);
    if (now != old) mm.store_scalar(addr, size, now);
    o.v0 = old;
  });
}

// Fields: the channels of a texel, new (load_fields, else one load_scalar each) and old (one load_scalar each).
Outcome fields_new(const MemoryManager& mm, uint64_t addr, const uint8_t* sizes, unsigned n) {
  return attempt([&](Outcome& o) {
    uint64_t v[4] = {0, 0, 0, 0};
    if (!mm.load_fields(addr, sizes, n, v)) {
      uint64_t at = addr;
      for (unsigned i = 0; i < n; ++i) {
        v[i] = sizes[i] ? mm.load_scalar(at, sizes[i]) : 0;
        at += sizes[i];
      }
    }
    for (unsigned i = 0; i < n; ++i) o.v0 = o.v0 * 1000003 + v[i], o.v1 ^= v[i] << (i * 8);
  });
}
Outcome fields_old(const MemoryManager& mm, uint64_t addr, const uint8_t* sizes, unsigned n) {
  return attempt([&](Outcome& o) {
    uint64_t v[4] = {0, 0, 0, 0};
    uint64_t at = addr;
    for (unsigned i = 0; i < n; ++i) {
      v[i] = sizes[i] ? mm.load_scalar(at, sizes[i]) : 0;
      at += sizes[i];
    }
    for (unsigned i = 0; i < n; ++i) o.v0 = o.v0 * 1000003 + v[i], o.v1 ^= v[i] << (i * 8);
  });
}

struct Stats {
  uint64_t ops = 0, failed = 0, fast = 0;
};

// Random atomics and field reads against two managers given the same allocations and the same history.
Stats run_random(uint64_t seed, int rounds, int fault) {
  Pair p;
  Faulty fa(fault), fb(fault);
  if (fault) p.a.set_access_fault(&fa), p.b.set_access_fault(&fb);
  Rng rng{seed};
  std::vector<uint64_t> live, freed, sizes;
  Stats st;
  uint8_t stack_byte = 0;
  const uint64_t host_ptr = reinterpret_cast<uint64_t>(&stack_byte);
  const uint64_t kSizes[] = {8, 16, 100, 4096, 65536, 65537, 3 * kChunkSize, 3 * kChunkSize + 24};
  auto same = [&](const char* what, int r, const Outcome& x, const Outcome& y) {
    if (!(x == y)) {
      char buf[600];
      std::snprintf(buf, sizeof buf,
                    "seed %llu round %d (%s): new {failed=%d code=%d v=%llx,%llx '%.100s'} old {failed=%d code=%d v=%llx,%llx '%.100s'}",
                    (unsigned long long)seed, r, what, x.failed, x.code, (unsigned long long)x.v0,
                    (unsigned long long)x.v1, x.text.c_str(), y.failed, y.code, (unsigned long long)y.v0,
                    (unsigned long long)y.v1, y.text.c_str());
      throw vtest::Failure(buf);
    }
    ++st.ops;
    st.failed += x.failed;
  };
  auto add_alloc = [&] {
    const uint64_t sz = kSizes[rng.below(sizeof kSizes / sizeof kSizes[0])];
    const uint64_t pa = p.a.alloc(sz);
    if (pa != p.b.alloc(sz)) throw vtest::Failure("different addresses");
    live.push_back(pa), sizes.push_back(sz);
    // Some chunks uniform, some real, some left untouched.
    for (uint64_t off = 0; off + kChunkSize <= sz; off += kChunkSize) {
      const uint64_t k = rng.below(3);
      if (k == 0) {
        const uint8_t pat[1] = {static_cast<uint8_t>(rng.below(2) ? 0xFF : 0x5A)};
        p.a.fill(pa + off, pat, 1, kChunkSize), p.b.fill(pa + off, pat, 1, kChunkSize);
      } else if (k == 1) {
        std::vector<uint8_t> junk(256);
        for (auto& c : junk) c = static_cast<uint8_t>(rng.next());
        p.a.write(pa + off, junk.data(), junk.size()), p.b.write(pa + off, junk.data(), junk.size());
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
      p.a.free(live[k]), p.b.free(live[k]);
      freed.push_back(live[k]);
      live.erase(live.begin() + k), sizes.erase(sizes.begin() + k);
      continue;
    }
    uint64_t addr;
    const uint64_t where = rng.below(100);
    if (where < 60) {
      const size_t k = rng.below(live.size());
      addr = live[k] + (rng.below(3) == 0 ? rng.below(300) : rng.below(sizes[k]));
    } else if (where < 78) {   // at the ends and over them, and at chunk boundaries
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
    const uint32_t sz[] = {1, 2, 4, 8};
    const uint32_t size = sz[rng.below(4)];
    if (rng.below(10) != 0) addr &= ~uint64_t{size - 1};   // mostly aligned
    if (const uint64_t pick = rng.below(3); pick < 2) {
      const int code = static_cast<int>(rng.below(8));
      const uint64_t b = rng.next();
      if (pick == 0) same("atomic", r, rmw_new(p.a, addr, size, code, b), rmw_old(p.b, addr, size, code, b));
      else same("atomic, stored if changed", r, changed_new(p.a, addr, size, code, b), changed_old(p.b, addr, size, code, b));
    } else {
      // A texel's channels: end to end, sizes from {0, 1, 2, 4, 8} (a 3 sometimes, which has no load).
      uint8_t fs[4];
      const uint8_t choices[] = {0, 1, 1, 2, 2, 4, 4, 4, 8, 3};
      for (auto& f : fs) f = choices[rng.below(rng.below(8) ? 9 : 10)];
      same("fields", r, fields_new(p.a, addr, fs, 4), fields_old(p.b, addr, fs, 4));
    }
    // Counting the new path's own successes: one that went through.
    uint64_t old;
    if (rng.below(16) == 0) {
      const uint64_t probe = addr & ~uint64_t{3};
      if (!fault) {
        if (p.a.rmw_scalar(probe, 4, [&](uint64_t v) { return v; }, &old)) ++st.fast;
        (void)rmw_old(p.b, probe, 4, 5, 0);
      }
    }
  }
  // Memory at the end must be the same too, down to which chunks are real.
  p.a.set_access_fault(nullptr), p.b.set_access_fault(nullptr);
  for (size_t k = 0; k < live.size(); ++k) {
    std::vector<uint8_t> x(sizes[k]), y(sizes[k]);
    p.a.read(live[k], x.data(), x.size());
    p.b.read(live[k], y.data(), y.size());
    if (x != y) throw vtest::Failure("contents differ in allocation " + std::to_string(k) + ", seed " + std::to_string(seed));
  }
  if (p.a.resident_bytes() != p.b.resident_bytes())
    throw vtest::Failure("chunks made real differ: new " + std::to_string(p.a.resident_bytes()) + " old " +
                         std::to_string(p.b.resident_bytes()) + ", seed " + std::to_string(seed));
  VCHECK_EQ(p.a.used(), p.b.used());
  return st;
}

}  // namespace

VTEST(read_modify_write_and_field_reads_match_the_calls_they_replace) {
  Stats total;
  for (uint64_t seed = 1; seed <= 40; ++seed) {
    const Stats s = run_random(seed * 7919, 3000, /*fault=*/0);
    total.ops += s.ops, total.failed += s.failed, total.fast += s.fast;
  }
  VCHECK(total.ops > 50000);
  VCHECK(total.failed > 3000);   // the error paths were met
  VCHECK(total.fast > 500);      // and the one-lookup path was taken
}

VTEST(read_modify_write_and_field_reads_match_with_faults_armed_or_stuck_cells) {
  for (int fault = 1; fault <= 3; ++fault)
    for (uint64_t seed = 1; seed <= 8; ++seed) run_random(seed * 104729 + fault, 2000, fault);
}

VTEST(read_modify_write_declines_without_doing_anything) {
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(4096);
  uint64_t old = 77;
  bool ran = false;
  auto spy = [&](uint64_t v) { ran = true; return v; };
  VCHECK(mm.rmw_scalar(p, 4, spy, &old));        // inside, aligned: taken
  VCHECK(ran);
  ran = false;
  old = 77;
  VCHECK(!mm.rmw_scalar(p + 2, 4, spy, &old));   // misaligned
  VCHECK(!mm.rmw_scalar(p, 3, spy, &old));       // no such width
  VCHECK(!mm.rmw_scalar(p + 4092, 8, spy, &old)); // runs past the end
  VCHECK(!mm.rmw_scalar(p - 8, 8, spy, &old));   // before the start
  mm.free(p);
  VCHECK(!mm.rmw_scalar(p, 4, spy, &old));       // freed
  VCHECK(!ran);
  VCHECK_EQ(old, 77u);
  mm.set_lookup_cache(false);
  const uint64_t q = mm.alloc(64);
  VCHECK(!mm.rmw_scalar(q, 4, spy, &old));       // no lookup memory: always the old calls
  const uint8_t sizes[2] = {4, 4};
  uint64_t v[2];
  VCHECK(!mm.load_fields(q, sizes, 2, v));
  mm.set_lookup_cache(true);
  VCHECK(mm.load_fields(q, sizes, 2, v));
}

VTEST(a_single_thread_atomic_stores_only_a_changed_word_and_only_in_a_real_chunk) {
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(2 * kChunkSize);
  uint64_t old = 77;
  int stores = 0;
  const auto same = [&](uint64_t v) { return v; };
  const auto plus = [&](uint64_t v) { return ++stores, v + 5; };
  VCHECK(!mm.rmw_if_changed(p + 16, 4, plus, &old));   // a chunk nothing has written: the general path's
  VCHECK_EQ(stores, 0);
  VCHECK_EQ(mm.resident_bytes(), 0u);
  const uint32_t seed = 9;
  mm.write(p, &seed, 4);                                // makes the first chunk real
  (void)mm.load_scalar(p + 16, 4);                      // (and a load leaves the allocation in the thread's lookup memory)
  VCHECK(mm.rmw_if_changed(p + 16, 4, plus, &old));
  VCHECK_EQ(old, 0u);
  VCHECK_EQ(mm.load_scalar(p + 16, 4), 5u);
  VCHECK(mm.rmw_if_changed(p + 16, 4, same, &old));
  VCHECK_EQ(old, 5u);
  VCHECK(!mm.rmw_if_changed(p + 18, 4, plus, &old));   // misaligned
  VCHECK(!mm.rmw_if_changed(p + 16, 3, plus, &old));   // no such width
  VCHECK(!mm.rmw_if_changed(p + kChunkSize + 8, 4, plus, &old));   // the second chunk is still untouched
  VCHECK_EQ(mm.resident_bytes(), kChunkSize);
  struct Armed : MemoryManager::AccessFault {
    uint64_t on = 1;
    Armed() { store_pending = &on; }
    uint64_t on_load(uint64_t, uint32_t, uint64_t v) override { return v; }
    uint64_t on_store(uint64_t, uint32_t, uint64_t v) override { return v; }
    uint64_t on_shared_load(uint64_t, uint32_t, uint64_t v) override { return v; }
    void on_read(uint64_t, uint8_t*, uint64_t) override {}
    uint64_t on_alu(uint64_t v, uint32_t) override { return v; }
    void on_copy(uint64_t, uint8_t*, uint64_t) override {}
  } armed;
  mm.set_access_fault(&armed);
  VCHECK(!mm.rmw_if_changed(p + 16, 4, plus, &old));   // a fault armed on stores: the stores take it
  mm.set_access_fault(nullptr);
}

namespace {

// A wave's atomics as one batch (rmw_batch, with the executors' fallback of one at a time when it declines) against the same accesses one
// at a time through load_scalar, the operation and store_scalar where the result differs: the values each loaded, what was left in
// memory, the chunks made real, and the error (and the accesses made before it).
struct BatchResult {
  std::string error;
  std::vector<uint64_t> olds;
  bool operator==(const BatchResult& o) const { return error == o.error && olds == o.olds; }
};

BatchResult batch_new(MemoryManager& mm, const std::vector<uint64_t>& addrs, uint32_t size, int code, const std::vector<uint64_t>& b,
                      bool lock_free, bool* batched) {
  BatchResult r;
  r.olds.assign(addrs.size(), 0);
  size_t done = 0;
  try {
    if (mm.rmw_batch(addrs.data(), addrs.size(), size, [&](size_t i, uint64_t v) { return op_result(code, size, v, b[i]); },
                     r.olds.data(), &done, lock_free)) {
      *batched = true;
      return r;
    }
    for (size_t i = 0; i < addrs.size(); ++i, done = i) {
      r.olds[i] = mm.load_scalar(addrs[i], size);
      const uint64_t now = op_result(code, size, r.olds[i], b[i]);
      if (now != r.olds[i]) mm.store_scalar(addrs[i], size, now);
    }
  } catch (const Error& e) {
    r.error = e.what();
  } catch (const std::runtime_error& e) {
    r.error = e.what();
  }
  r.olds.resize(done);
  return r;
}
BatchResult batch_old(MemoryManager& mm, const std::vector<uint64_t>& addrs, uint32_t size, int code, const std::vector<uint64_t>& b) {
  BatchResult r;
  try {
    for (size_t i = 0; i < addrs.size(); ++i) {
      const uint64_t old = mm.load_scalar(addrs[i], size);
      const uint64_t now = op_result(code, size, old, b[i]);
      if (now != old) mm.store_scalar(addrs[i], size, now);
      r.olds.push_back(old);
    }
  } catch (const Error& e) {
    r.error = e.what();
  } catch (const std::runtime_error& e) {
    r.error = e.what();
  }
  return r;
}

uint64_t run_batches(uint64_t seed, int rounds, int fault) {
  Pair p;
  Faulty fa(fault), fb(fault);
  if (fault) p.a.set_access_fault(&fa), p.b.set_access_fault(&fb);
  Rng rng{seed};
  const uint64_t sizes[] = {64, 4096, 65536, 2 * kChunkSize + 100};
  uint64_t base[4];
  uint8_t stack_byte = 0;
  for (int k = 0; k < 4; ++k) {
    base[k] = p.a.alloc(sizes[k]);
    if (base[k] != p.b.alloc(sizes[k])) throw vtest::Failure("different addresses");
    for (uint64_t off = 0; off + kChunkSize <= sizes[k]; off += kChunkSize) {
      const uint64_t kind = rng.below(3);   // uniform, real or never written
      if (kind == 0) {
        const uint8_t pat[1] = {static_cast<uint8_t>(rng.below(2) ? 0xFF : 0)};
        p.a.fill(base[k] + off, pat, 1, kChunkSize), p.b.fill(base[k] + off, pat, 1, kChunkSize);
      } else if (kind == 1) {
        std::vector<uint8_t> junk(300);
        for (auto& c : junk) c = static_cast<uint8_t>(rng.below(4));   // (small values, so a CAS sometimes matches)
        p.a.write(base[k] + off, junk.data(), junk.size()), p.b.write(base[k] + off, junk.data(), junk.size());
      }
    }
  }
  uint64_t batched = 0;
  for (int r = 0; r < rounds; ++r) {
    const uint32_t kSz[] = {1, 2, 4, 8};
    const uint32_t size = kSz[rng.below(4)];
    const size_t count = 1 + rng.below(rng.below(4) ? 64 : 5);
    const size_t k = rng.below(4);
    const uint64_t shape = rng.below(10);
    std::vector<uint64_t> addrs(count), b(count);
    for (size_t i = 0; i < count; ++i) {
      uint64_t a;
      switch (shape) {
        case 0: a = base[k] + (rng.below(8) * size); break;                                    // a few words, repeated
        case 1: a = base[k] + i * size; break;                                                 // consecutive
        case 2: a = base[k] + rng.below(sizes[k]); break;                                      // any byte
        case 3: a = base[k] + sizes[k] - size * rng.below(6); break;                           // on the end, and over it
        case 4: a = base[k] + kChunkSize - size * rng.below(4); break;                         // at a chunk edge
        case 5: a = rng.below(5) == 0 ? base[rng.below(4)] + rng.below(32) : base[k] + 8 * i; break;   // another allocation now and then
        case 6: a = rng.below(7) == 0 ? reinterpret_cast<uint64_t>(&stack_byte) : base[k] + 4 * i; break;   // host memory
        default: a = base[k] + (rng.below(sizes[k] / size)) * size; break;                    // random, aligned
      }
      if (shape != 2 && shape != 3 && rng.below(40) != 0) a &= ~uint64_t{size - 1};
      addrs[i] = a;
      b[i] = rng.next();
    }
    const int code = static_cast<int>(rng.below(rng.below(6) ? 7 : 8));
    const bool lock_free = rng.below(2) != 0 && (size == 4 || size == 8);
    bool was_batched = false;
    const BatchResult x = batch_new(p.a, addrs, size, code, b, lock_free, &was_batched);
    const BatchResult y = batch_old(p.b, addrs, size, code, b);
    batched += was_batched;
    if (!(x == y))
      throw vtest::Failure("seed " + std::to_string(seed) + " round " + std::to_string(r) + ": batch {" + x.error + "} " +
                           std::to_string(x.olds.size()) + " loaded, one at a time {" + y.error + "} " + std::to_string(y.olds.size()));
  }
  p.a.set_access_fault(nullptr), p.b.set_access_fault(nullptr);
  for (int k = 0; k < 4; ++k) {
    std::vector<uint8_t> x(sizes[k]), y(sizes[k]);
    p.a.read(base[k], x.data(), x.size());
    p.b.read(base[k], y.data(), y.size());
    if (x != y) throw vtest::Failure("contents differ in allocation " + std::to_string(k) + ", seed " + std::to_string(seed));
  }
  if (p.a.resident_bytes() != p.b.resident_bytes())
    throw vtest::Failure("chunks made real differ: batch " + std::to_string(p.a.resident_bytes()) + " one at a time " +
                         std::to_string(p.b.resident_bytes()) + ", seed " + std::to_string(seed));
  return batched;
}

}  // namespace

VTEST(a_wave_of_atomics_as_a_batch_matches_the_same_accesses_one_at_a_time) {
  uint64_t batched = 0;
  for (uint64_t seed = 1; seed <= 60; ++seed) batched += run_batches(seed * 6151, 400, /*fault=*/0);
  VCHECK(batched > 3000);   // the batch path was taken, not only its fallback
}

VTEST(a_wave_of_atomics_as_a_batch_declines_with_faults_armed_or_stuck_cells) {
  for (int fault = 1; fault <= 3; ++fault)
    for (uint64_t seed = 1; seed <= 6; ++seed) VCHECK_EQ(run_batches(seed * 977 + fault, 300, fault), 0u);
}

VTEST(concurrent_atomics_from_several_threads_add_up) {
  // Four threads, each adding to a few shared words under a stripe lock as the interpreter does, a word of each
  // width and one in a uniform chunk: the final values are the sums.
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(3 * kChunkSize);
  const uint8_t pat[1] = {0};
  mm.fill(p + kChunkSize, pat, 1, kChunkSize);   // a uniform chunk
  const uint64_t addrs[] = {p + 8, p + kChunkSize + 8, p + 2 * kChunkSize + 8, p + 2 * kChunkSize + 16};
  const uint32_t sizes[] = {8, 4, 2, 1};
  std::mutex locks[8];
  std::vector<std::thread> ts;
  const int kIters = 20000;
  for (int t = 0; t < 4; ++t)
    ts.emplace_back([&, t] {
      for (int i = 0; i < kIters; ++i)
        for (int k = 0; k < 4; ++k) {
          std::lock_guard<std::mutex> g(locks[(addrs[k] >> 3) & 7]);
          uint64_t old = 0;
          const uint32_t sz = sizes[k];
          if (!mm.rmw_scalar(addrs[k], sz, [&](uint64_t v) { return (v + 1 + t) & width_mask(sz); }, &old)) {
            old = mm.load_scalar(addrs[k], sz);
            mm.store_scalar(addrs[k], sz, (old + 1 + t) & width_mask(sz));
          }
        }
    });
  for (auto& t : ts) t.join();
  uint64_t expect = 0;
  for (int t = 0; t < 4; ++t) expect += static_cast<uint64_t>(kIters) * (1 + t);
  for (int k = 0; k < 4; ++k) VCHECK_EQ(mm.load_scalar(addrs[k], sizes[k]), expect & width_mask(sizes[k]));
}

// ---------------------------------------------------------------------------
// The interpreter's atomics, with the lookup cache on (one lookup) and off (the old calls).
// ---------------------------------------------------------------------------
namespace {

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

struct AtomRun {
  std::vector<uint8_t> olds, mem;
  std::string error;
  exec::LaunchStats stats;
  uint64_t resident = 0;
};

// One kernel: every thread does `op` on a word picked by its id -- four chunks (never written, uniform, real, real),
// four words a chunk, so lanes of a warp meet on the same word -- with a value drawn from its id, and stores what it
// found. `ty` is the PTX type, `bytes` its width.
AtomRun run_atom(const std::string& op, const std::string& ty, int bytes, bool two_operands, int blocks,
                 bool lookup_cache, uint64_t bad_offset = 0, bool bad = false) {
  const bool wide = bytes == 8;
  std::string ptx = ".version 8.3\n.target sm_86\n.address_size 64\n";
  ptx += ".visible .entry k(.param .u64 pbuf, .param .u64 pout)\n{\n";
  ptx += "    .reg .b32 %r<16>;\n    .reg .b64 %rd<16>;\n    .reg .f32 %f<4>;\n    .reg .f64 %fd<4>;\n    .reg .b16 %h<4>;\n";
  ptx += "    ld.param.u64 %rd1, [pbuf];\n    ld.param.u64 %rd2, [pout];\n";
  ptx += "    mov.u32 %r1, %tid.x;\n    mov.u32 %r2, %ctaid.x;\n    mov.u32 %r3, %ntid.x;\n    mad.lo.u32 %r4, %r2, %r3, %r1;\n";
  // word = ((gid / 4) % 4) * 8 within chunk gid % 4
  ptx += "    and.b32 %r5, %r4, 3;\n    mul.lo.u32 %r5, %r5, 65536;\n";
  ptx += "    shr.u32 %r6, %r4, 2;\n    and.b32 %r6, %r6, 3;\n    mul.lo.u32 %r6, %r6, 8;\n    add.u32 %r5, %r5, %r6;\n";
  if (bad) ptx += "    add.u32 %r5, %r5, " + std::to_string(bad_offset) + ";\n";
  ptx += "    cvt.u64.u32 %rd3, %r5;\n    add.u64 %rd3, %rd1, %rd3;\n";
  // operand values from the id
  ptx += "    mul.lo.u32 %r7, %r4, 2654435761;\n    xor.b32 %r7, %r7, 305419896;\n    and.b32 %r8, %r4, 3;\n";
  ptx += "    mul.wide.u32 %rd4, %r7, 40503;\n    cvt.u64.u32 %rd7, %r4;\n    add.u64 %rd4, %rd4, %rd7;\n";
  const std::string dty = ty;
  std::string d, a, c;
  if (ty == "f32") {
    ptx += "    cvt.rn.f32.u32 %f1, %r8;\n    mul.f32 %f1, %f1, 0f3F000000;\n    add.f32 %f1, %f1, 0f3DCCCCCD;\n";
    d = "%f2", a = "%f1";
  } else if (ty == "f64") {
    ptx += "    cvt.rn.f64.u32 %fd1, %r8;\n    mul.f64 %fd1, %fd1, 0d3FE0000000000000;\n    add.f64 %fd1, %fd1, 0d3FB999999999999A;\n";
    d = "%fd2", a = "%fd1";
  } else if (ty == "f16" || ty == "bf16") {
    // a half from a small float
    ptx += std::string("    cvt.rn.f32.u32 %f1, %r8;\n    add.f32 %f1, %f1, 0f3F800000;\n    cvt.rn.") + ty + ".f32 %h1, %f1;\n";
    d = "%h2", a = "%h1";
  } else if (ty == "f16x2" || ty == "bf16x2") {
    ptx += std::string("    cvt.rn.f32.u32 %f1, %r8;\n    add.f32 %f1, %f1, 0f3F800000;\n    cvt.rn.") +
           (ty == "f16x2" ? "f16" : "bf16") + ".f32 %h1, %f1;\n    mov.b32 %r9, {%h1, %h1};\n";
    d = "%r10", a = "%r9";
  } else if (wide) {
    d = "%rd5", a = "%rd4";
  } else {
    ptx += "    cvt.u32.u64 %r9, %rd4;\n";
    d = "%r10", a = "%r9";
  }
  c = a;
  ptx += "    atom.global." + op + "." + ty + " " + d + ", [%rd3], " + a + (two_operands ? ", " + c : "") + ";\n";
  // store the old value at out[gid * 8]
  ptx += "    mul.wide.u32 %rd6, %r4, 8;\n    add.u64 %rd6, %rd2, %rd6;\n";
  if (ty == "f32") ptx += "    mov.b32 %r11, %f2;\n    st.global.u32 [%rd6], %r11;\n";
  else if (ty == "f64") ptx += "    st.global.f64 [%rd6], %fd2;\n";
  else if (ty == "f16" || ty == "bf16") ptx += "    st.global.b16 [%rd6], %h2;\n";
  else if (wide) ptx += "    st.global.u64 [%rd6], %rd5;\n";
  else ptx += "    st.global.u32 [%rd6], %r10;\n";
  ptx += "    ret;\n}\n";

  AtomRun r;
  setenv("VGPU_LOOKUP_CACHE", lookup_cache ? "1" : "0", 1);
  MemoryManager mem{1ull << 26};
  unsetenv("VGPU_LOOKUP_CACHE");
  DeviceProfile prof = load_gpu("nvidia/a10");
  auto m = ptx::parse(ptx);
  const uint64_t buf = mem.alloc(4 * kChunkSize), out = mem.alloc(64 * 8 * 8);
  const uint8_t pat[1] = {0xFF};
  mem.fill(buf + kChunkSize, pat, 1, kChunkSize);   // chunk 1 uniform
  Rng rng{0xC0FFEE};
  for (int chunk = 2; chunk < 4; ++chunk)           // chunks 2 and 3 real, with small floats and small ints in them
    for (int w = 0; w < 4; ++w) {
      uint64_t v = (rng.next() & 0x00FF00FF00FF00FFull);
      if (ty[0] == 'f' || ty[0] == 'b') v = 0x3F8000003F800000ull;
      mem.write(buf + chunk * kChunkSize + w * 8, &v, 8);
    }
  LaunchConfig cfg;
  cfg.block = {32 * 2, 1, 1};
  cfg.grid = {static_cast<uint32_t>(blocks), 1, 1};
  try {
    r.stats = exec::launch(m.entries[0], cfg, {arg_u64(buf), arg_u64(out)}, mem, prof);
  } catch (const std::exception& e) {
    r.error = e.what();
  }
  r.olds.resize(64 * 8 * 8);
  mem.read(out, r.olds.data(), r.olds.size());
  r.mem.resize(4 * kChunkSize);
  mem.read(buf, r.mem.data(), r.mem.size());
  r.resident = mem.resident_bytes();
  return r;
}

bool same_stats(const exec::LaunchStats& a, const exec::LaunchStats& b) {
  return std::memcmp(&a, &b, exec::kCounterWords * sizeof(uint64_t)) == 0 && a.inst_by_opcode == b.inst_by_opcode;
}

void same_atom(const std::string& op, const std::string& ty, int bytes, bool two, int blocks, bool compare_olds) {
  const AtomRun fast = run_atom(op, ty, bytes, two, blocks, true);
  const AtomRun slow = run_atom(op, ty, bytes, two, blocks, false);
  const std::string what = op + "." + ty + " x" + std::to_string(blocks) + " blocks";
  if (fast.error != slow.error) throw vtest::Failure(what + ": errors differ: '" + fast.error + "' vs '" + slow.error + "'");
  if (fast.mem != slow.mem) throw vtest::Failure(what + ": memory differs");
  if (compare_olds && fast.olds != slow.olds) throw vtest::Failure(what + ": the values the atomics found differ");
  if (fast.resident != slow.resident) throw vtest::Failure(what + ": chunks made real differ");
  if (!same_stats(fast.stats, slow.stats)) throw vtest::Failure(what + ": counters differ");
}

}  // namespace

VTEST(ptx_atomics_match_the_old_calls_for_every_operation_width_and_type) {
  struct Form { const char* op; const char* ty; int bytes; bool two; };
  const Form forms[] = {
      {"add", "u32", 4, false}, {"add", "s32", 4, false}, {"add", "u64", 8, false}, {"and", "b32", 4, false},
      {"or", "b32", 4, false},  {"xor", "b32", 4, false}, {"and", "b64", 8, false},  {"or", "b64", 8, false},
      {"xor", "b64", 8, false}, {"exch", "b32", 4, false}, {"exch", "b64", 8, false}, {"min", "u32", 4, false},
      {"max", "u32", 4, false}, {"min", "s32", 4, false}, {"max", "s32", 4, false},  {"min", "u64", 8, false},
      {"max", "u64", 8, false}, {"min", "s64", 8, false}, {"max", "s64", 8, false},  {"inc", "u32", 4, false},
      {"dec", "u32", 4, false}, {"cas", "b32", 4, true},  {"cas", "b64", 8, true},   {"add", "f32", 4, false},
      {"min", "f32", 4, false}, {"max", "f32", 4, false}, {"exch", "b32", 4, false}, {"add", "f64", 8, false},
      {"add.noftz", "f16", 2, false}, {"add.noftz", "bf16", 2, false},
      {"add.noftz", "f16x2", 4, false}, {"add.noftz", "bf16x2", 4, false},
  };
  for (const Form& f : forms) {
    same_atom(f.op, f.ty, f.bytes, f.two, /*blocks=*/1, /*compare_olds=*/true);   // one block: a fixed order
    // Several blocks on several host threads, colliding: the final memory is what is compared for the
    // commutative forms (the values found depend on the order blocks ran in).
    const std::string op = f.op;
    const bool commutes = op == "add" || op == "and" || op == "or" || op == "xor" || op == "min" || op == "max" ||
                          op == "add.noftz";
    if (commutes && std::string(f.ty).rfind("f", 0) != 0 && std::string(f.ty).rfind("bf", 0) != 0)
      same_atom(f.op, f.ty, f.bytes, f.two, /*blocks=*/8, /*compare_olds=*/false);
  }
}

VTEST(ptx_atomics_meet_the_same_errors_as_the_old_calls) {
  // A lane past the end of the buffer, and a lane not aligned: the same error text with and without the lookup memory.
  for (const uint64_t off : {uint64_t{1} << 28, uint64_t{4}, uint64_t{1}, uint64_t{4 * kChunkSize - 4}}) {
    const AtomRun fast = run_atom("add", "u64", 8, false, 1, true, off, true);
    const AtomRun slow = run_atom("add", "u64", 8, false, 1, false, off, true);
    VCHECK(!slow.error.empty());
    if (fast.error != slow.error) throw vtest::Failure("errors differ at +" + std::to_string(off) + ": '" + fast.error + "' vs '" + slow.error + "'");
    if (fast.mem != slow.mem || fast.olds != slow.olds) throw vtest::Failure("state after the error differs at +" + std::to_string(off));
  }
}

// ---------------------------------------------------------------------------
// Texture fetches: texel channels read together against one at a time.
// ---------------------------------------------------------------------------
VTEST(texture_fetches_read_the_same_channels_with_and_without_the_one_lookup) {
  Rng rng{99};
  struct Fmt { uint32_t bits[4]; exec::ChannelKind kind; bool packed; uint32_t texel_bytes; };
  const Fmt fmts[] = {
      {{32, 0, 0, 0}, exec::ChannelKind::Float, false, 4},     {{32, 32, 32, 32}, exec::ChannelKind::Float, false, 16},
      {{16, 16, 16, 16}, exec::ChannelKind::Float, false, 8},  {{8, 8, 8, 8}, exec::ChannelKind::Unsigned, false, 4},
      {{8, 8, 0, 0}, exec::ChannelKind::Signed, false, 2},     {{16, 16, 0, 0}, exec::ChannelKind::Unsigned, false, 4},
      {{10, 10, 10, 2}, exec::ChannelKind::Unsigned, true, 4}, {{8, 0, 0, 0}, exec::ChannelKind::Unsigned, false, 1},
      {{8, 8, 8, 0}, exec::ChannelKind::Unsigned, false, 3},   // three one-byte channels: a 3-byte texel, misaligned rows
  };
  for (const Fmt& f : fmts)
    for (int filter = 0; filter < 2; ++filter)
      for (int part = 0; part < 3; ++part) {   // 0: all real; 1: a uniform chunk under half of it; 2: part untouched
        MemoryManager mem{1ull << 28};
        const uint32_t w = 37, h = 29;
        const uint64_t bytes = uint64_t{w} * h * f.texel_bytes;
        const uint64_t base = mem.alloc(std::max<uint64_t>(bytes, 2 * kChunkSize) + 64);
        std::vector<uint8_t> data(bytes);
        for (auto& b : data) b = static_cast<uint8_t>(rng.next());
        if (part != 2) mem.write(base, data.data(), data.size());
        else mem.write(base, data.data(), data.size() / 2);
        if (part == 1) {
          const uint8_t pat[1] = {0x6B};
          mem.fill(base + kChunkSize, pat, 1, kChunkSize);
        }
        exec::TextureDesc d;
        d.base = base;
        d.width = w;
        d.height = h;
        d.channels = 0;
        for (int i = 0; i < 4; ++i) d.channel_bits[i] = f.bits[i], d.channels += f.bits[i] ? 1 : 0;
        d.texel_bytes = f.texel_bytes;
        d.kind = f.kind;
        d.packed_1010102 = f.packed;
        d.filter = filter ? exec::TexFilter::Linear : exec::TexFilter::Point;
        if (f.kind != exec::ChannelKind::Float) d.read_as_normalized_float = true;
        for (auto& a : d.address) a = exec::TexAddress::Clamp;
        auto run = [&](bool cache) {
          mem.set_lookup_cache(cache);
          std::vector<uint32_t> out;
          std::string errs;
          Rng c{12345};
          for (int i = 0; i < 300; ++i) {
            exec::TexFetch t;
            t.dims = 2;
            const float x = static_cast<float>(c.below(100000)) / 100000.0f * (w + 3) - 1.0f;
            const float y = static_cast<float>(c.below(100000)) / 100000.0f * (h + 3) - 1.0f;
            std::memcpy(&t.coord[0], &x, 4);
            std::memcpy(&t.coord[1], &y, 4);
            t.float_result = c.below(2) != 0 || f.kind == exec::ChannelKind::Float;
            if (i % 5 == 4) t.gather = static_cast<int>(c.below(4));
            uint32_t r[4] = {0, 0, 0, 0};
            try {
              exec::texture_fetch(mem, d, t, r);
            } catch (const Error& e) {
              errs += e.what();
              errs += '\n';
            }
            out.insert(out.end(), r, r + 4);
          }
          return std::make_pair(out, errs);
        };
        const auto fast = run(true), slow = run(false);
        if (fast.second != slow.second) throw vtest::Failure("texture errors differ: '" + fast.second + "' vs '" + slow.second + "'");
        if (fast.first != slow.first) {
          size_t i = 0;
          while (fast.first[i] == slow.first[i]) ++i;
          throw vtest::Failure("texel values differ: format " + std::to_string(f.bits[0]) + "/" + std::to_string(f.bits[1]) +
                               " part " + std::to_string(part) + " fetch " + std::to_string(i / 4) + " channel " + std::to_string(i % 4));
        }
      }
}

VTEST_MAIN
