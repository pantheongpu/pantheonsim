// MemoryManager::copy_to, a device-to-device copy made chunk to chunk, against the pair of calls it replaces
// (read into a buffer, then write from it).
//
// The destination must come out the same byte for byte AND chunk for chunk: resident_bytes() counts the real chunks, and
// write() leaves a uniform chunk alone when it is given its own byte again, so a shortcut that made such a chunk real would
// be seen there. Every kind of source and destination chunk is mixed (untouched, uniform, real, real but all one byte),
// at offsets and lengths that cross chunk boundaries on either side, within one manager and across two. The cases that
// must decline -- overlapping ranges, a range outside its allocation, a freed one, a host mapping in reach, a fault armed on
// copies -- are checked to do nothing and report false.
#include <cstdint>
#include <cstring>
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

constexpr uint64_t kChunks = 6;

// Fills an allocation of kChunks chunks with a mix of chunk kinds, the same way in both managers of a pair.
void populate(MemoryManager& mm, uint64_t base, uint64_t seed) {
  Rng rng{seed};
  std::vector<uint8_t> bytes(kChunkSize);
  for (uint64_t c = 0; c < kChunks; ++c) {
    switch (rng.below(5)) {
      case 0:   // untouched
        break;
      case 1: {   // uniform
        const uint8_t pat = static_cast<uint8_t>(rng.below(3) ? rng.next() : 0);
        mm.fill(base + c * kChunkSize, &pat, 1, kChunkSize);
        break;
      }
      case 2: {   // real, random bytes
        for (auto& b : bytes) b = static_cast<uint8_t>(rng.next());
        mm.write(base + c * kChunkSize, bytes.data(), kChunkSize);
        break;
      }
      case 3: {   // real, but every byte the same (made real by a partial write, then the rest)
        const uint8_t v = static_cast<uint8_t>(rng.next());
        std::memset(bytes.data(), v, kChunkSize);
        mm.write(base + c * kChunkSize + 100, bytes.data(), 100);
        break;
      }
      default: {   // real, half of it written
        for (auto& b : bytes) b = static_cast<uint8_t>(rng.next());
        mm.write(base + c * kChunkSize + kChunkSize / 2, bytes.data(), kChunkSize / 2);
        break;
      }
    }
  }
}

std::vector<uint8_t> dump(const MemoryManager& mm, uint64_t base) {
  std::vector<uint8_t> v(kChunks * kChunkSize);
  mm.read(base, v.data(), v.size());
  return v;
}

// A fault armed on copies only.
struct CopyFault : MemoryManager::AccessFault {
  uint64_t armed = 1;
  CopyFault() { copy_pending = &armed; }
  uint64_t on_load(uint64_t, uint32_t, uint64_t value) override { return value; }
  uint64_t on_store(uint64_t, uint32_t, uint64_t value) override { return value; }
  uint64_t on_shared_load(uint64_t, uint32_t, uint64_t value) override { return value; }
  void on_read(uint64_t, uint8_t*, uint64_t) override {}
  uint64_t on_alu(uint64_t value, uint32_t) override { return value; }
  void on_copy(uint64_t, uint8_t*, uint64_t) override {}
};
}  // namespace

// `cross`: source and destination live in two managers; otherwise in one (two allocations, or one allocation's disjoint halves).
static void differential(bool cross, bool one_allocation, uint64_t seed0, int rounds) {
  unsigned taken = 0;
  for (int it = 0; it < rounds; ++it) {
    const uint64_t seed = seed0 + static_cast<uint64_t>(it) * 7919;
    Rng rng{seed ^ 0xABCDEF};
    // Two identical worlds: A copies with copy_to, B with the read/write pair.
    MemoryManager a0(1ull << 30, 0), a1(1ull << 30, 1), b0(1ull << 30, 0), b1(1ull << 30, 1);
    MemoryManager& asrc = a0;
    MemoryManager& adst = cross ? a1 : a0;
    MemoryManager& bsrc = b0;
    MemoryManager& bdst = cross ? b1 : b0;
    const uint64_t as = asrc.alloc(kChunks * kChunkSize), bs = bsrc.alloc(kChunks * kChunkSize);
    VCHECK_EQ(as, bs);
    populate(asrc, as, seed);
    populate(bsrc, bs, seed);
    uint64_t ad = as, bd = bs;
    if (!one_allocation) {
      ad = adst.alloc(kChunks * kChunkSize);
      bd = bdst.alloc(kChunks * kChunkSize);
      VCHECK_EQ(ad, bd);
      populate(adst, ad, seed + 1);
      populate(bdst, bd, seed + 1);
    }
    // Offsets near chunk boundaries half the time. Within one allocation the two ranges are kept disjoint.
    uint64_t len = 1 + rng.below(3 * kChunkSize);
    uint64_t soff = it % 2 ? kChunkSize - rng.below(40) : rng.below(kChunks * kChunkSize - len);
    uint64_t doff = it % 3 ? 2 * kChunkSize - rng.below(40) : rng.below(kChunks * kChunkSize - len);
    if (soff + len > kChunks * kChunkSize) len = kChunks * kChunkSize - soff;
    if (doff + len > kChunks * kChunkSize) len = kChunks * kChunkSize - doff;
    if (one_allocation && soff < doff + len && doff < soff + len) {   // make them disjoint: the destination follows the source
      doff = soff + len;
      if (doff + len > kChunks * kChunkSize) {
        len = (kChunks * kChunkSize - soff) / 2;
        doff = soff + len;
      }
      if (len == 0) continue;
    }
    VCHECK(asrc.copy_to(adst, ad + doff, as + soff, len));   // (nothing in the way: it is taken)
    ++taken;
    std::vector<uint8_t> tmp(len);
    bsrc.read(bs + soff, tmp.data(), len);
    bdst.write(bd + doff, tmp.data(), len);
    VCHECK(dump(adst, ad) == dump(bdst, bd));
    VCHECK(dump(asrc, as) == dump(bsrc, bs));
    VCHECK_EQ(adst.resident_bytes(), bdst.resident_bytes());
    VCHECK_EQ(asrc.resident_bytes(), bsrc.resident_bytes());
  }
  VCHECK(taken > static_cast<unsigned>(rounds) / 2);
}

VTEST(copy_to_matches_read_then_write_between_two_allocations_of_one_manager) { differential(false, false, 11, 600); }
VTEST(copy_to_matches_read_then_write_between_two_managers) { differential(true, false, 29, 600); }
VTEST(copy_to_matches_read_then_write_between_disjoint_halves_of_one_allocation) { differential(false, true, 47, 600); }

VTEST(copy_to_copies_untouched_memory_as_zeros_over_written_memory_and_makes_the_chunks_real_as_write_does) {
  MemoryManager mm(1ull << 28);
  const uint64_t a = mm.alloc(4 * kChunkSize), b = mm.alloc(4 * kChunkSize);
  std::vector<uint8_t> junk(4 * kChunkSize, 0x5A);
  mm.write(b, junk.data(), junk.size());
  VCHECK(mm.copy_to(mm, b, a, 4 * kChunkSize));   // a is untouched
  std::vector<uint8_t> out(4 * kChunkSize, 1);
  mm.read(b, out.data(), out.size());
  for (uint8_t v : out) VCHECK_EQ(v, 0);
  // Untouched memory written to untouched memory: write() makes the chunks real, so does the copy.
  MemoryManager m2(1ull << 28), m3(1ull << 28);
  const uint64_t s2 = m2.alloc(3 * kChunkSize), d2 = m2.alloc(3 * kChunkSize);
  const uint64_t s3 = m3.alloc(3 * kChunkSize), d3 = m3.alloc(3 * kChunkSize);
  VCHECK(m2.copy_to(m2, d2, s2, 3 * kChunkSize));
  std::vector<uint8_t> zeros(3 * kChunkSize);
  m3.write(d3, zeros.data(), zeros.size());
  VCHECK_EQ(m2.resident_bytes(), m3.resident_bytes());
  (void)s3;
}

VTEST(copy_to_declines_and_changes_nothing_for_an_overlap_a_bad_range_a_freed_range_a_host_mapping_or_a_copy_fault) {
  MemoryManager mm(1ull << 28), other(1ull << 28, 1);
  const uint64_t a = mm.alloc(4 * kChunkSize), b = mm.alloc(4 * kChunkSize);
  std::vector<uint8_t> in(4 * kChunkSize);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i * 13 + (i >> 9));
  mm.write(a, in.data(), in.size());
  const auto exact = [&](uint64_t p) {
    std::vector<uint8_t> v(4 * kChunkSize);
    mm.read(p, v.data(), v.size());
    return v;
  };
  const auto sa0 = exact(a), sb0 = exact(b);
  // Overlap inside one allocation, both directions.
  VCHECK(!mm.copy_to(mm, a + 1000, a, 3 * kChunkSize));
  VCHECK(!mm.copy_to(mm, a, a + 1000, 3 * kChunkSize));
  VCHECK(!mm.copy_to(mm, a, a, kChunkSize));
  // A range past the end of either allocation; an address that is no allocation's.
  VCHECK(!mm.copy_to(mm, b, a, 4 * kChunkSize + 1));
  VCHECK(!mm.copy_to(mm, b + 1, a, 4 * kChunkSize));
  VCHECK(!mm.copy_to(mm, b + 64 * kChunkSize, a, 100));
  VCHECK(!mm.copy_to(other, a, a, 100));   // the other manager owns no such destination
  // A freed destination.
  const uint64_t gone = mm.alloc(2 * kChunkSize);
  mm.free(gone);
  VCHECK(!mm.copy_to(mm, gone, a, kChunkSize));
  VCHECK(!mm.copy_to(mm, b, gone, kChunkSize));
  // A host buffer mapped where the destination's range starts or ends, and where the source's starts (write() and read() look at the
  // start of a range; the copy asks the end too).
  std::vector<uint32_t> host(1024, 7);
  mm.map_host(b + 2 * kChunkSize, host.data(), 4096);
  VCHECK(!mm.copy_to(mm, b, a, 2 * kChunkSize + 4096));   // ends in the map
  VCHECK(!mm.copy_to(mm, b + 2 * kChunkSize, a, 100));     // starts in it
  VCHECK(!mm.copy_to(mm, a, b + 2 * kChunkSize + 8, 100));
  mm.unmap_host(b + 2 * kChunkSize);
  // A fault armed on copies: read() takes it, so the pair of calls must.
  CopyFault f;
  mm.set_access_fault(&f);
  VCHECK(!mm.copy_to(mm, b, a, 2 * kChunkSize));
  mm.set_access_fault(nullptr);
  VCHECK(exact(a) == sa0);
  VCHECK(exact(b) == sb0);
  // And with all of that out of the way it is taken, and the bytes arrive.
  VCHECK(mm.copy_to(mm, b, a, 4 * kChunkSize));
  VCHECK(exact(b) == in);
}

VTEST(copy_to_of_zero_bytes_is_taken_and_does_nothing) {
  MemoryManager mm(1ull << 28);
  const uint64_t a = mm.alloc(kChunkSize);
  VCHECK(mm.copy_to(mm, a, a, 0));
  VCHECK_EQ(mm.resident_bytes(), uint64_t{0});
}

VTEST_MAIN
