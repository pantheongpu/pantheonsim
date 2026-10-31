// MemoryManager::lookup_host: the per-thread cache of host-mapped buffers that keeps a kernel's scalar accesses to
// mapped memory (PantheonZE maps every Level Zero allocation this way) from taking the table's mutex and scanning it
// on every access.
//
// The cache must never serve a mapping that is gone or changed: an unmap, a remap of the same address to another
// buffer, another manager, and a manager destroyed and replaced all have to be seen at once, and a mapped buffer's
// read-only flag must survive it. And it must be safe with several threads accessing while another maps and unmaps.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {
struct Buf {
  uint8_t* p;
  size_t n;
  Buf(size_t bytes, uint8_t fill) : p(static_cast<uint8_t*>(std::aligned_alloc(4096, (bytes + 4095) / 4096 * 4096))), n(bytes) {
    std::memset(p, fill, n);
  }
  Buf(const Buf&) = delete;
  Buf(Buf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }
  ~Buf() { std::free(p); }
  uint64_t addr() const { return reinterpret_cast<uint64_t>(p); }
};
}  // namespace

VTEST(a_mapped_buffer_reads_and_writes_through_the_cache) {
  MemoryManager mem{1 << 20};
  Buf b(4096, 0x11);
  mem.map_host(b.addr(), b.p, b.n);
  for (int i = 0; i < 100; ++i) {   // the first access fills the cache, the rest are served from it
    if (i == 0) VCHECK_EQ(mem.load_scalar(b.addr() + 8, 4), uint64_t{0x11111111});
    mem.store_scalar(b.addr() + 8, 4, 0xA0u + i);
    VCHECK_EQ(mem.load_scalar(b.addr() + 8, 4), uint64_t{0xA0u + i});
  }
  uint32_t host_view;
  std::memcpy(&host_view, b.p + 8, 4);
  VCHECK_EQ(host_view, 0xA0u + 99);   // the kernel wrote the host's own bytes
  uint8_t two[2];
  mem.read(b.addr() + 100, two, 2);
  VCHECK_EQ(two[0], uint8_t{0x11});
  const uint8_t w[3] = {7, 8, 9};
  mem.write(b.addr() + 100, w, 3);
  VCHECK_EQ(b.p[102], uint8_t{9});
  VCHECK(mem.is_host_mapped(b.addr() + 4095));
  VCHECK(!mem.is_host_mapped(b.addr() + 4096));
  VCHECK(mem.host_address(b.addr() + 12, 4) == b.p + 12);
  // an access that runs off the end of the mapping is not a hit, whatever the cache holds
  auto off = VCAPTURE(Error, mem.load_scalar(b.addr() + 4096, 4));
  VCHECK(off.code() == Err::InvalidPointer || off.code() == Err::UseAfterFree || off.code() == Err::OutOfBounds);
}

VTEST(an_unmap_is_seen_by_a_thread_that_cached_the_mapping) {
  MemoryManager mem{1 << 20};
  Buf b(4096, 0x22);
  mem.map_host(b.addr(), b.p, b.n);
  VCHECK_EQ(mem.load_scalar(b.addr(), 4), uint64_t{0x22222222});   // cached
  mem.unmap_host(b.addr());
  auto err = VCAPTURE(Error, mem.load_scalar(b.addr(), 4));
  VCHECK(err.code() == Err::UseAfterFree);                          // the retired record, not a stale hit
  auto werr = VCAPTURE(Error, mem.store_scalar(b.addr(), 4, 1));
  VCHECK(werr.code() == Err::UseAfterFree);
  VCHECK(!mem.is_host_mapped(b.addr()));
  VCHECK(mem.host_address(b.addr(), 4) == nullptr);
}

VTEST(a_remap_of_the_same_address_to_another_buffer_is_seen) {
  MemoryManager mem{1 << 20};
  Buf a(4096, 0x33), b(4096, 0x44);
  mem.map_host(a.addr(), a.p, a.n);
  VCHECK_EQ(mem.load_scalar(a.addr(), 4), uint64_t{0x33333333});
  mem.unmap_host(a.addr());
  // the same engine address now backed by b's bytes (what a backend does for a reused allocation)
  mem.map_host(a.addr(), b.p, b.n);
  VCHECK_EQ(mem.load_scalar(a.addr(), 4), uint64_t{0x44444444});
  mem.store_scalar(a.addr(), 4, 0x55555555u);
  VCHECK_EQ(uint32_t{b.p[0]}, 0x55u);
  VCHECK_EQ(uint32_t{a.p[0]}, 0x33u);   // a was not touched
}

VTEST(a_read_only_mapping_stays_read_only_in_the_cache) {
  MemoryManager mem{1 << 20};
  Buf b(4096, 0x66);
  mem.map_host(b.addr(), b.p, b.n, /*read_only=*/true);
  for (int i = 0; i < 3; ++i) {
    VCHECK_EQ(mem.load_scalar(b.addr(), 4), uint64_t{0x66666666});
    auto err = VCAPTURE(Error, mem.store_scalar(b.addr(), 4, 1));
    VCHECK(err.code() != Err::Internal);
    VCHECK_EQ(uint32_t{b.p[0]}, 0x66u);
  }
}

VTEST(another_manager_and_a_replaced_manager_never_see_each_others_entries) {
  Buf a(4096, 0x71), b(4096, 0x72), c(4096, 0x73);
  const uint64_t at = a.addr();
  {
    MemoryManager m1{1 << 20};
    m1.map_host(at, a.p, a.n);
    VCHECK_EQ(m1.load_scalar(at, 4), uint64_t{0x71717171});   // cached for m1
    MemoryManager m2{1 << 20};
    m2.map_host(at, b.p, b.n);
    VCHECK_EQ(m2.load_scalar(at, 4), uint64_t{0x72727272});   // not m1's entry
    VCHECK_EQ(m1.load_scalar(at, 4), uint64_t{0x71717171});
  }
  // m1 and m2 are gone; managers made now may reuse their heap addresses, and must not inherit their cached entries
  for (int i = 0; i < 8; ++i) {
    MemoryManager m3{1 << 20};
    m3.map_host(at, c.p, c.n);
    VCHECK_EQ(m3.load_scalar(at, 4), uint64_t{0x73737373});
  }
}

VTEST(threads_access_their_buffers_while_another_maps_and_unmaps) {
  MemoryManager mem{1 << 20};
  constexpr int kReaders = 4, kIters = 20000;
  std::vector<Buf> bufs;
  for (int i = 0; i < kReaders; ++i) bufs.emplace_back(4096, 0);
  for (auto& b : bufs) mem.map_host(b.addr(), b.p, b.n);
  std::atomic<bool> stop{false};
  std::atomic<int> bad{0};
  std::vector<std::thread> th;
  for (int t = 0; t < kReaders; ++t)
    th.emplace_back([&, t] {
      try {
        for (int i = 0; i < kIters; ++i) {
          const uint64_t at = bufs[t].addr() + 16 * (i % 200);
          mem.store_scalar(at, 4, static_cast<uint64_t>(t * 1000000 + i));
          if (mem.load_scalar(at, 4) != static_cast<uint64_t>(t * 1000000 + i)) ++bad;
        }
      } catch (...) {
        ++bad;
      }
    });
  std::thread churn([&] {
    Buf extra(4096, 0);
    while (!stop.load()) {
      mem.map_host(extra.addr(), extra.p, extra.n);
      mem.unmap_host(extra.addr());
    }
  });
  for (auto& x : th) x.join();
  stop = true;
  churn.join();
  VCHECK_EQ(bad.load(), 0);
}

// host_load_batch / host_store_batch (a warp's accesses to one host-mapped buffer, made together) against load_scalar / store_scalar
// lane by lane: the same values and bytes at every width, scattered and repeated addresses, partial counts; and the cases they decline
// (an address outside the buffer or in none, misaligned, a read-only buffer, two buffers in one warp) with nothing done.
VTEST(a_warps_accesses_to_a_mapped_buffer_are_the_lane_by_lane_ones) {
  MemoryManager mem{1 << 20};
  Buf b(8192, 0), other(4096, 0x77);
  mem.map_host(b.addr(), b.p, b.n);
  mem.map_host(other.addr(), other.p, other.n);
  uint32_t seed = 12345;
  const auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
  const auto word = [&](uint32_t size) {
    return (uint64_t{rnd()} << 32 | rnd()) & (size == 8 ? ~uint64_t{0} : (uint64_t{1} << (8 * size)) - 1);
  };
  for (size_t i = 0; i < b.n; ++i) b.p[i] = static_cast<uint8_t>(rnd());
  for (uint32_t size : {1u, 2u, 4u, 8u})
    for (int iter = 0; iter < 200; ++iter) {
      const size_t count = 1 + rnd() % 32;
      uint64_t addrs[32], got[32];
      for (size_t i = 0; i < count; ++i)
        addrs[i] = iter % 4 == 0 ? b.addr() + 64 + i * size : b.addr() + (rnd() % (b.n / size)) * size;
      VCHECK(mem.host_load_batch(addrs, count, size, got));
      for (size_t i = 0; i < count; ++i) VCHECK_EQ(got[i], mem.load_scalar(addrs[i], size));
    }
  // stores: a buffer written by the batch against its twin written lane by lane (few words, so lanes repeat and the later one wins)
  Buf c(8192, 0), d(8192, 0);
  mem.map_host(c.addr(), c.p, c.n);
  mem.map_host(d.addr(), d.p, d.n);
  for (uint32_t size : {1u, 2u, 4u, 8u})
    for (int iter = 0; iter < 100; ++iter) {
      const size_t count = 1 + rnd() % 32;
      uint64_t ca[32], da[32], vals[32];
      for (size_t i = 0; i < count; ++i) {
        const uint64_t off = (rnd() % 64) * size;
        ca[i] = c.addr() + off;
        da[i] = d.addr() + off;
        vals[i] = word(size);
      }
      VCHECK(mem.host_store_batch(ca, count, size, vals));
      for (size_t i = 0; i < count; ++i) mem.store_scalar(da[i], size, vals[i]);
      VCHECK(std::memcmp(c.p, d.p, c.n) == 0);
    }
  // the declines
  uint64_t a2[2], v2[2] = {1, 2}, o2[2];
  a2[0] = b.addr() + 8, a2[1] = b.addr() + b.n;                       // the second one past the end
  VCHECK(!mem.host_load_batch(a2, 2, 4, o2));
  VCHECK(!mem.host_store_batch(a2, 2, 4, v2));
  a2[1] = b.addr() + 2;                                              // misaligned for 4 bytes
  VCHECK(!mem.host_load_batch(a2, 2, 4, o2));
  a2[1] = other.addr();                                              // another buffer
  VCHECK(!mem.host_load_batch(a2, 2, 4, o2));
  a2[0] = b.addr() - 4, a2[1] = b.addr();                            // the first below the buffer
  VCHECK(!mem.host_load_batch(a2, 2, 4, o2));
  VCHECK(!mem.host_load_batch(a2, 0, 4, o2));
  VCHECK(!mem.host_load_batch(a2 + 1, 1, 3, o2));                    // not a width
  Buf ro(4096, 0x5a);
  mem.map_host(ro.addr(), ro.p, ro.n, /*read_only=*/true);
  a2[0] = ro.addr(), a2[1] = ro.addr() + 4;
  VCHECK(mem.host_load_batch(a2, 2, 4, o2));
  VCHECK_EQ(o2[0], uint64_t{0x5a5a5a5a});
  VCHECK(!mem.host_store_batch(a2, 2, 4, v2));                       // the lane-by-lane store reports the read-only write
  VCHECK_EQ(ro.p[0], uint8_t{0x5a});
}

// The host's own atomics race the batch (a kernel polling or filling zero-copy memory while the program works on the same words): the batch's
// accesses are the relaxed atomic ones load_scalar and store_scalar make, so ThreadSanitizer sees no race and no value is torn.
VTEST(a_warps_batch_on_a_mapped_buffer_races_the_hosts_atomics_cleanly) {
  MemoryManager mem{1 << 20};
  Buf b(4096, 0);
  mem.map_host(b.addr(), b.p, b.n);
  std::atomic<bool> stop{false};
  std::atomic<int> torn{0};
  std::thread host([&] {
    uint32_t* w = reinterpret_cast<uint32_t*>(b.p);
    for (uint32_t i = 1; !stop.load(); ++i)
      for (int k = 0; k < 32; ++k) __atomic_store_n(&w[k], (i & 0xffu) * 0x01010101u, __ATOMIC_RELAXED);
  });
  for (int iter = 0; iter < 2000; ++iter) {
    uint64_t addrs[32], vals[32], got[32];
    for (int k = 0; k < 32; ++k) addrs[k] = b.addr() + 4 * k;
    VCHECK(mem.host_load_batch(addrs, 32, 4, got));
    for (int k = 0; k < 32; ++k) {
      const uint32_t x = static_cast<uint32_t>(got[k]);
      if ((x & 0xff) != ((x >> 8) & 0xff) || (x & 0xff) != ((x >> 24) & 0xff)) ++torn;
    }
    for (int k = 0; k < 32; ++k) vals[k] = 0x02020202u * (iter % 100 + 1);
    VCHECK(mem.host_store_batch(addrs, 32, 4, vals));
  }
  stop = true;
  host.join();
  VCHECK_EQ(torn.load(), 0);
}

VTEST_MAIN
