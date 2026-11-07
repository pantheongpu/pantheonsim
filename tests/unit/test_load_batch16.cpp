// A warp's 16-byte loads (MemoryManager::load_batch16, one lookup and one pass for the two 8-byte halves) against the definition: each lane the two
// load_scalar words of its address and address + 8 (the two load_batch calls of 8-byte pieces the SASS tier made before), over every kind of chunk
// (untouched, uniform, real), addresses scattered over the allocation, on a chunk's last 16 bytes, at the end of the allocation. Where any lane is
// misaligned (to 8), not wholly inside the allocation, or its 16 bytes cross a chunk end, the batch declines (its output is then of no use) and the
// caller's two 8-byte calls run; the ones it takes must be right, and the ordinary ones must be taken.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
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
}  // namespace

VTEST(sixteen_byte_gather_gives_each_lane_its_two_words_over_every_kind_of_chunk) {
  Rng rng{0x16b17e5};
  MemoryManager mm(1ull << 28);
  constexpr uint64_t kChunks = 8;
  const uint64_t total = kChunks * kChunkSize - 104;   // (the last chunk is partial; the end is 8-aligned but not 16-aligned)
  const uint64_t p = mm.alloc(total);
  std::vector<uint8_t> bytes(kChunkSize);
  const uint8_t pat[3] = {0x5A, 0x00, 0xC3};
  for (uint64_t c = 0; c < kChunks; ++c) {
    // chunks 0, 3 and 6 real; 1 and 4 uniform (one of them of zero bytes); 2, 5 and 7 untouched
    if (c == 0 || c == 3 || c == 6) {
      for (auto& b : bytes) b = static_cast<uint8_t>(rng.next());
      mm.write(p + c * kChunkSize, bytes.data(), kChunkSize);
    } else if (c == 1 || c == 4) {
      mm.fill(p + c * kChunkSize, &pat[c == 1 ? 0 : 2], 1, kChunkSize);
    }
  }
  unsigned taken = 0, declined_misaligned = 0, declined_end = 0, declined_cross = 0, mixed = 0, at_chunk_last = 0;
  for (int it = 0; it < 30000; ++it) {
    const size_t count = 1 + rng.below(64);
    std::vector<uint64_t> addrs(count), lo(count), hi(count);
    const bool allreal = it % 3 == 0;   // a warp wholly on real chunks, or anywhere
    bool misaligned = false, outside = false, cross = false;
    unsigned kinds = 0;
    const size_t special_lane = rng.below(count);
    const unsigned special = it % 9 == 1 ? 1 + static_cast<unsigned>(rng.below(5)) : 0;
    for (size_t i = 0; i < count; ++i) {
      uint64_t off;
      if (allreal) off = (rng.below(2) ? 3 : 6) * kChunkSize + rng.below(kChunkSize / 16 - 1) * 16;
      else off = rng.below(total / 16) * 16;
      if (special && i == special_lane) {
        switch (special) {
          case 1: off = (1 + rng.below(kChunks - 1)) * kChunkSize - 8; break;     // 16 bytes across a chunk end (8-aligned)
          case 2: off = (1 + rng.below(kChunks - 1)) * kChunkSize - 16; ++at_chunk_last; break;   // the last 16 bytes of a chunk: taken
          case 3: off = total - 8; break;                                          // the second word past the end
          case 4: off = 0 - 16 * (1 + rng.below(3)); break;                       // below the base
          default: off = rng.below(total / 16 - 1) * 16 + 4; break;               // not 8-aligned
        }
      }
      addrs[i] = p + off;
      if (off & 7) misaligned = true;
      else if (off >= total || off + 16 > total) outside = true;
      else if (off % kChunkSize > kChunkSize - 16) cross = true;
      else kinds |= 1u << (off / kChunkSize);
    }
    std::fill(lo.begin(), lo.end(), 0xDEADBEEFDEADBEEFull);
    std::fill(hi.begin(), hi.end(), 0xDEADBEEFDEADBEEFull);
    const bool done = mm.load_batch16(addrs.data(), count, lo.data(), hi.data());
    if (misaligned || outside || cross) {
      VCHECK(!done);
      declined_misaligned += misaligned;
      declined_end += !misaligned && outside;
      declined_cross += !misaligned && !outside && cross;
      continue;
    }
    VCHECK(done);
    ++taken;
    if ((kinds & 0x49u) && (kinds & ~0x49u)) ++mixed;
    for (size_t i = 0; i < count; ++i) {
      const uint64_t w0 = mm.load_scalar(addrs[i], 8), w1 = mm.load_scalar(addrs[i] + 8, 8);
      if (lo[i] != w0 || hi[i] != w1)
        throw vtest::Failure("lane " + std::to_string(i) + " of " + std::to_string(count) + " differs (first word " + (lo[i] == w0 ? "equal" : "different") +
                             ", second " + (hi[i] == w1 ? "equal" : "different") + ")");
      // and the two 8-byte batches the tier ran before give the same words
      uint64_t a = addrs[i], b = addrs[i] + 8, x = 0, y = 0;
      VCHECK(mm.load_batch(&a, 1, 8, &x) && mm.load_batch(&b, 1, 8, &y));
      VCHECK(x == w0 && y == w1);
    }
  }
  VCHECK(taken > 15000);
  VCHECK(declined_misaligned > 500);
  VCHECK(declined_end > 500);
  VCHECK(declined_cross > 500);
  VCHECK(mixed > 3000);
  VCHECK(at_chunk_last > 500);
}

VTEST(sixteen_byte_gather_declines_what_the_ordinary_batches_leave_to_the_lane_loop) {
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(4 * kChunkSize);
  std::vector<uint8_t> bytes(kChunkSize, 0x11);
  mm.write(p, bytes.data(), kChunkSize);
  uint64_t lo[2], hi[2], addrs[2] = {p, p + 16};
  VCHECK(mm.load_batch16(addrs, 2, lo, hi));
  VCHECK(lo[0] == 0x1111111111111111ull && hi[1] == 0x1111111111111111ull);
  VCHECK(!mm.load_batch16(addrs, 0, lo, hi));            // no lanes
  addrs[1] = p + 4 * kChunkSize;                           // the first byte past the allocation
  VCHECK(!mm.load_batch16(addrs, 2, lo, hi));
  addrs[0] = p + 4 * kChunkSize - 8;                       // the first lane's second word is past the end
  VCHECK(!mm.load_batch16(addrs, 1, lo, hi));
  addrs[0] = p + 4;                                        // not 8-aligned
  VCHECK(!mm.load_batch16(addrs, 1, lo, hi));
  addrs[0] = 0x10;                                         // in no allocation
  VCHECK(!mm.load_batch16(addrs, 1, lo, hi));
  uint64_t many[65], l65[65], h65[65];
  for (unsigned i = 0; i < 65; ++i) many[i] = p + 16 * i;
  VCHECK(!mm.load_batch16(many, 65, l65, h65));            // more than a wave: the caller's two-step path
  VCHECK(mm.load_batch16(many, 64, l65, h65));
}

VTEST(sixteen_byte_gather_declines_when_a_host_mapping_lies_in_the_allocation) {
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(4 * kChunkSize);
  std::vector<uint8_t> bytes(kChunkSize, 0x22);
  mm.write(p, bytes.data(), kChunkSize);
  uint64_t lo[2], hi[2], addrs[2] = {p, p + 16};
  VCHECK(mm.load_batch16(addrs, 2, lo, hi));
  void* host = std::aligned_alloc(4096, 4096);
  std::memset(host, 0x33, 4096);
  // a mapped buffer in the allocation's address range: the batch is not the ordinary case (load_batch makes its two-step check; the 16-byte form leaves it)
  mm.map_host(p + 2 * kChunkSize, host, 4096);
  const bool declined = !mm.load_batch16(addrs, 2, lo, hi);
  mm.unmap_host(p + 2 * kChunkSize);
  std::free(host);
  VCHECK(declined);   // (the memory manager keeps the flag once a mapping has been in the device window: the caller's two-step check then decides)
}

VTEST_MAIN
