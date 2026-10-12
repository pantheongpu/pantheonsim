// The SASS executor's memory-access counting (vgpu/exec/mem_counters.hpp:
// distinct_sectors with SIZE_MAX as the cap, and sass_bank_passes) against the
// straightforward version it replaced, which gathered the touched sectors and
// the words of each shared-memory bank into std::vectors on every access. The
// counters these feed (global_sectors, shared_bank_conflicts, ...) are reported
// to users, so the allocation-free version must agree with the vector version
// on every access: aligned and unaligned, every access size, partial lane
// masks, broadcasts, strides that land every lane in one bank, and addresses at
// the top of the address space. (These patterns were first written for the
// SASS executor's own counting header, which the memory counters replaced.)
#include <algorithm>
#include <cstdint>
#include <vector>

#include "vgpu/exec/mem_counters.hpp"
#include "vtest.hpp"

namespace {

using Mask = uint32_t;

// The vector version, as exec_ops.inc counted before this change.
uint64_t ref_sectors(const uint64_t* addr, Mask lanes, uint32_t bytes) {
  std::vector<uint64_t> sectors;
  for (unsigned l = 0; l < 32; ++l) {
    if (!((lanes >> l) & 1)) continue;
    for (uint64_t s = addr[l] / 32; s <= (addr[l] + bytes - 1) / 32; ++s) sectors.push_back(s);
  }
  std::sort(sectors.begin(), sectors.end());
  return static_cast<uint64_t>(std::unique(sectors.begin(), sectors.end()) - sectors.begin());
}

uint64_t ref_passes(const uint64_t* addr, Mask lanes, uint32_t bytes) {
  std::vector<uint64_t> words[32];
  for (unsigned l = 0; l < 32; ++l) {
    if (!((lanes >> l) & 1)) continue;
    for (uint64_t a = addr[l] & ~3ull; a < addr[l] + bytes; a += 4) words[(a / 4) % 32].push_back(a / 4);
  }
  uint64_t passes = 1;
  for (auto& v : words) {
    std::sort(v.begin(), v.end());
    passes = std::max<uint64_t>(passes, static_cast<uint64_t>(std::unique(v.begin(), v.end()) - v.begin()));
  }
  return passes;
}

struct Rng {
  uint64_t s = 0x9E3779B97F4A7C15ull;
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
};

void check(const uint64_t* addr, Mask lanes, uint32_t bytes, const char* what) {
  const uint64_t s_new = vgpu::exec::memcount::distinct_sectors(addr, lanes, 32, bytes, SIZE_MAX),
                 s_ref = ref_sectors(addr, lanes, bytes);
  if (s_new != s_ref)
    throw vtest::Failure(std::string(what) + ": sectors " + std::to_string(s_new) + " vs " + std::to_string(s_ref) +
                         " (bytes " + std::to_string(bytes) + ", mask " + std::to_string(lanes) + ")");
  // The earlier code the fast version hands anything unusual to is checked on every access too.
  if (vgpu::exec::memcount::sass_sectors_reference(addr, lanes, 32, bytes) != s_ref)
    throw vtest::Failure(std::string(what) + ": reference sectors differ (bytes " + std::to_string(bytes) + ")");
  if (lanes) {
    const uint64_t b_new = vgpu::exec::memcount::sass_bank_passes(addr, lanes, 32, bytes),
                   b_ref = ref_passes(addr, lanes, bytes);
    if (b_new != b_ref)
      throw vtest::Failure(std::string(what) + ": bank passes " + std::to_string(b_new) + " vs " + std::to_string(b_ref) +
                           " (bytes " + std::to_string(bytes) + ", mask " + std::to_string(lanes) + ")");
  }
}

const uint32_t kSizes[] = {1, 2, 4, 8, 16, 32};

}  // namespace

VTEST(sass_access_count_patterns) {
  uint64_t addr[32];
  for (uint32_t bytes : kSizes) {
    for (uint64_t base : {0ull, 4ull, 28ull, 100ull, 0x7fffull, 0x10000001ull, 0x7f0012345678ull}) {
      for (uint64_t stride : {0ull, 1ull, 2ull, 4ull, 8ull, 12ull, 16ull, 32ull, 33ull, 64ull, 128ull, 132ull, 4096ull}) {
        for (unsigned l = 0; l < 32; ++l) addr[l] = base + stride * l;
        for (Mask m : {0xffffffffu, 0x1u, 0x80000000u, 0x0000ffffu, 0xaaaaaaaau, 0x55555555u, 0u})
          check(addr, m, bytes, "pattern");
      }
    }
  }
}

VTEST(sass_access_count_random) {
  Rng rng;
  uint64_t addr[32];
  for (int iter = 0; iter < 20000; ++iter) {
    const uint32_t bytes = kSizes[rng.next() % 6];
    const uint64_t span = uint64_t{1} << (rng.next() % 20 + 1);
    const uint64_t base = (rng.next() % 3 == 0) ? rng.next() : rng.next() % (1 << 24);
    for (unsigned l = 0; l < 32; ++l) {
      uint64_t a = base + rng.next() % span;
      if (rng.next() % 2) a &= ~uint64_t{bytes - 1};   // aligned to the access size half the time
      addr[l] = a;
    }
    check(addr, static_cast<Mask>(rng.next()), bytes, "random");
  }
}

VTEST(sass_access_count_top_of_address_space) {
  uint64_t addr[32];
  for (uint32_t bytes : kSizes)
    for (uint64_t back : {1ull, 3ull, 4ull, 16ull, 31ull, 32ull, 40ull}) {
      for (unsigned l = 0; l < 32; ++l) addr[l] = ~uint64_t{0} - back - l * 4;
      // An access that wraps the address space counts nothing in the vector
      // version; the shared-memory case never sees such addresses, so only the
      // sector count is compared.
      const uint64_t a = vgpu::exec::memcount::distinct_sectors(addr, 0xffffffffu, 32, bytes, SIZE_MAX),
                     b = ref_sectors(addr, 0xffffffffu, bytes);
      VCHECK_EQ(a, b);
    }
}

VTEST_MAIN
