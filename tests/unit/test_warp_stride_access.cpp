// A warp's load of one address (MemoryManager::load_broadcast: one lookup, one load, the word to every lane) and a warp's store on a stride (store_strided: one
// lookup for the lanes that share a chunk) against the definition: each lane a load_scalar / store_scalar of its width, over every kind of chunk (untouched,
// uniform, real), every width, bases next to chunk edges and to the ends of the allocation; both take exactly the warps whose lanes are all inside the
// allocation and aligned (and decline every other, so the caller's load_batch / store_batch / one-by-one path meets any error), and leave memory and
// chunk state as the one-by-one accesses do.
#include <cstdint>
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

struct Fixture {
  static constexpr uint64_t kChunks = 8;
  static constexpr uint64_t kTotal = kChunks * kChunkSize - 100;   // (the last chunk is partial; the end is not a multiple of 8)
  MemoryManager mm{1ull << 28};
  uint64_t p = 0;
  explicit Fixture(Rng& rng) {
    p = mm.alloc(kTotal);
    std::vector<uint8_t> bytes(kChunkSize);
    const uint8_t pat[2] = {0x5A, 0xC3};
    for (uint64_t c = 0; c < kChunks; ++c) {
      // chunks 0, 3 and 6 real; 1 and 4 uniform; 2, 5 and 7 untouched
      if (c == 0 || c == 3 || c == 6) {
        for (auto& b : bytes) b = static_cast<uint8_t>(rng.next());
        mm.write(p + c * kChunkSize, bytes.data(), c == kChunks - 1 ? kChunkSize - 100 : kChunkSize);
      } else if (c == 1 || c == 4) {
        mm.fill(p + c * kChunkSize, &pat[c == 1 ? 0 : 1], 1, kChunkSize);
      }
    }
  }
};
}  // namespace

VTEST(a_broadcast_load_gives_every_lane_what_load_scalar_gives_and_declines_what_it_cannot_take) {
  Rng rng{0xb40adca5};
  Fixture f(rng);
  MemoryManager& mm = f.mm;
  const uint64_t total = Fixture::kTotal;
  unsigned taken = 0, declined = 0, uniform = 0, untouched = 0, real = 0;
  for (int it = 0; it < 20000; ++it) {
    static const uint32_t widths[] = {1, 2, 4, 8};
    const uint32_t w = widths[rng.below(4)];
    const size_t count = 1 + rng.below(32);
    uint64_t off;
    switch (rng.below(4)) {
      case 0: off = rng.below(total / w) * w; break;
      case 1: off = (rng.below(Fixture::kChunks) * kChunkSize + kChunkSize - w * rng.below(8)) % total / w * w; break;
      case 2: off = (rng.below(Fixture::kChunks) * kChunkSize + w * rng.below(8)) % total / w * w; break;
      default: off = (total - w * (1 + rng.below(64))) / w * w; break;
    }
    if (it % 11 == 0 && w > 1) off += 1 + rng.below(w - 1);   // a misaligned address
    if (it % 17 == 0) off = total - w + w * rng.below(3);       // the last word, or past the end
    if (it % 19 == 0) off = 0 - w * (1 + rng.below(3));         // below the base
    const uint64_t addr = f.p + off;
    const bool want_taken = !(addr & (w - 1)) && off < total && off + w <= total;
    std::vector<uint64_t> out(count, 0xDEADBEEFDEADBEEFull);
    const bool done = mm.load_broadcast(addr, count, w, out.data());
    VCHECK(done == want_taken);
    if (!done) {
      ++declined;
      continue;
    }
    ++taken;
    const uint64_t want = mm.load_scalar(addr, w);
    const uint64_t c = off / kChunkSize;
    if (c == 0 || c == 3 || c == 6) ++real; else if (c == 1 || c == 4) ++uniform; else ++untouched;
    for (size_t i = 0; i < count; ++i)
      if (out[i] != want) throw vtest::Failure("lane " + std::to_string(i) + " of " + std::to_string(count) + " at width " + std::to_string(w) + " differs");
  }
  VCHECK(taken > 10000);
  VCHECK(declined > 1000);
  VCHECK(real > 2000);
  VCHECK(uniform > 1500);
  VCHECK(untouched > 2000);
  // more than a warp, no lanes, a width that is not one: the caller's
  uint64_t out[40];
  VCHECK(!mm.load_broadcast(f.p, 33, 4, out));
  VCHECK(!mm.load_broadcast(f.p, 0, 4, out));
  VCHECK(!mm.load_broadcast(f.p, 4, 3, out));
  VCHECK(!mm.load_broadcast(0x1000, 4, 4, out));   // (no allocation)
}

VTEST(a_strided_store_leaves_memory_and_chunk_state_as_store_scalar_does_and_declines_what_it_cannot_take) {
  Rng seed{0x5707e5};
  unsigned taken = 0, declined = 0, uniform_kept = 0, made_real = 0;
  for (int round = 0; round < 600; ++round) {
    const uint64_t s0 = seed.next();
    Rng r1{s0}, r2{s0};
    Fixture a(r1), b(r2);   // two identical worlds: `a` stores strided, `b` lane by lane
    Rng rng{seed.next()};
    const uint64_t total = Fixture::kTotal;
    for (int it = 0; it < 40; ++it) {   // (a short life per world, so that the chunks are not all real by the time most stores are made)
      static const uint32_t widths[] = {1, 2, 4, 8};
      const uint32_t w = widths[rng.below(4)];
      const size_t count = 2 + rng.below(31);
      int64_t stride = static_cast<int64_t>(w) * static_cast<int64_t>(rng.below(rng.below(2) ? 64 : kChunkSize / 2 / w + 1));
      if (rng.below(8) == 0) stride = static_cast<int64_t>(kChunkSize / 2) + static_cast<int64_t>(w);   // (declined)
      if (rng.below(2)) stride = -stride;
      uint64_t off = rng.below(total / w) * w;
      if (rng.below(3) == 0) off = (rng.below(Fixture::kChunks) * kChunkSize + w * rng.below(16)) % total / w * w;
      if (it % 13 == 0 && w > 1) off += 1;   // misaligned
      std::vector<uint64_t> vals(count);
      // values: the uniform chunks' own byte now and then (a store that leaves them alone), else anything
      const uint64_t pick = rng.below(4);
      for (auto& v : vals) {
        const uint64_t any = rng.next();
        const uint64_t byte = pick == 0 ? 0x5A : pick == 1 ? 0xC3 : pick == 2 ? 0 : any;
        v = pick < 3 ? byte * 0x0101010101010101ull : any;
        if (w < 8) v &= (uint64_t{1} << (8 * w)) - 1;
      }
      const uint64_t a0 = a.p + off;
      bool want_taken = !((a0 & (w - 1)) || (static_cast<uint64_t>(stride) & (w - 1)));
      if (stride > static_cast<int64_t>(kChunkSize / 2) || stride < -static_cast<int64_t>(kChunkSize / 2)) want_taken = false;
      for (size_t i = 0; i < count; ++i) {
        const uint64_t o = off + static_cast<uint64_t>(stride) * i;
        if (o >= total || o + w > total) want_taken = false;
      }
      const uint64_t before = a.mm.resident_bytes();
      const bool done = a.mm.store_strided(a0, stride, count, w, vals.data());
      VCHECK(done == want_taken);
      if (!done) {
        ++declined;
        VCHECK(a.mm.resident_bytes() == before);   // (declined: nothing done)
        continue;
      }
      ++taken;
      for (size_t i = 0; i < count; ++i) b.mm.store_scalar(b.p + off + static_cast<uint64_t>(stride) * i, w, vals[i]);
      VCHECK(a.mm.resident_bytes() == b.mm.resident_bytes());
      if (a.mm.resident_bytes() == before) ++uniform_kept; else ++made_real;
    }
    std::vector<uint8_t> ba(total), bb(total);
    a.mm.read(a.p, ba.data(), total);
    b.mm.read(b.p, bb.data(), total);
    if (ba != bb) throw vtest::Failure("round " + std::to_string(round) + ": the bytes differ");
  }
  VCHECK(taken > 8000);
  VCHECK(declined > 2500);
  VCHECK(uniform_kept > 300);
  VCHECK(made_real > 300);
}

// The edges, deterministically: a lane that lands exactly on the first byte of the next chunk (either way), a last lane that ends exactly at the end of the
// allocation (taken) or one byte past it (declined), a first lane at the base or one byte below it.
VTEST(a_strided_store_at_the_edges_of_a_chunk_and_of_the_allocation) {
  Rng seed{0xed9e5};
  const uint64_t total = Fixture::kTotal;
  for (uint32_t w : {1u, 2u, 4u, 8u})
    for (int64_t unit : {1, 2, 3, 8, 64, 1000}) {
      const int64_t stride = unit * w;
      for (int dir : {1, -1})
        for (unsigned j = 1; j <= 6; ++j) {
          Rng r1{seed.next()}, r2 = r1;
          Fixture a(r1), b(r2);
          // lane j is the first of chunk 3 (ascending) or the last of chunk 2 (descending: lane j at chunk 3's first byte - w would leave it; so end on the start)
          const int64_t s = dir * stride;
          const uint64_t target = 3 * kChunkSize;   // offset of the lane that must land exactly on a chunk's first byte
          const uint64_t a0off = target - static_cast<uint64_t>(s) * j;
          const size_t count = 8;
          std::vector<uint64_t> vals(count);
          for (size_t i = 0; i < count; ++i) vals[i] = (0x1111111111111111ull * (i + 1)) & (w == 8 ? ~0ull : ((1ull << (8 * w)) - 1));
          const bool done = a.mm.store_strided(a.p + a0off, s, count, w, vals.data());
          bool inside = true;
          for (size_t i = 0; i < count; ++i) {
            const uint64_t o = a0off + static_cast<uint64_t>(s) * i;
            inside = inside && o < total && o + w <= total;
          }
          VCHECK(done == inside);
          if (!done) continue;
          for (size_t i = 0; i < count; ++i) b.mm.store_scalar(b.p + a0off + static_cast<uint64_t>(s) * i, w, vals[i]);
          VCHECK(a.mm.resident_bytes() == b.mm.resident_bytes());
          std::vector<uint8_t> ba(total), bb(total);
          a.mm.read(a.p, ba.data(), total);
          b.mm.read(b.p, bb.data(), total);
          if (ba != bb) throw vtest::Failure("width " + std::to_string(w) + " stride " + std::to_string(s) + " lane " + std::to_string(j) + " on the chunk edge: the bytes differ");
        }
    }
  // the last lane ends exactly at the end of the allocation: taken; one byte further: declined (width 1, so that the end can be any byte)
  Rng r(0xe9d);
  Fixture f(r);
  uint64_t v[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  VCHECK(f.mm.store_strided(f.p + total - 1 - 3 * 4, 3, 5, 1, v));    // lanes at total-13, -10, -7, -4, -1: the last byte
  VCHECK(!f.mm.store_strided(f.p + total - 3 * 4, 3, 5, 1, v));       // the last lane at `total`: past the end
  VCHECK(f.mm.store_strided(f.p + 12, -3, 5, 1, v));                  // lanes at 12, 9, 6, 3, 0: the first byte
  VCHECK(!f.mm.store_strided(f.p + 11, -3, 5, 1, v));                 // lanes at 11, 8, 5, 2, -1: below the base
}

VTEST_MAIN
