// A warp's gather load (MemoryManager::load_batch, whose tests and loads share their passes) against the definition: each lane a
// load_scalar of its width, over every kind of chunk (untouched, uniform, real), warps of every size up to a warp and past it,
// every width, addresses scattered over the whole allocation. Where any lane is out of the allocation or misaligned the batch
// declines (its output is then of no use); a load_batch_in-based window load (batch_window / batch_load) gives the same values.
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
}  // namespace

VTEST(a_gather_load_gives_each_lane_what_load_scalar_gives_over_every_kind_of_chunk) {
  Rng rng{0xfeed5eed};
  MemoryManager mm(1ull << 28);
  constexpr uint64_t kChunks = 8;
  const uint64_t total = kChunks * kChunkSize - 100;   // (the last chunk is partial; the end is not a multiple of 8)
  const uint64_t p = mm.alloc(total);
  std::vector<uint8_t> bytes(kChunkSize);
  const uint8_t pat[3] = {0x5A, 0x00, 0xC3};
  for (uint64_t c = 0; c < kChunks; ++c) {
    // chunks 0, 3 and 6 real; 1 and 4 uniform (one of them of zero bytes); 2, 5 and 7 untouched
    if (c == 0 || c == 3 || c == 6) {
      for (auto& b : bytes) b = static_cast<uint8_t>(rng.next());
      mm.write(p + c * kChunkSize, bytes.data(), c == kChunks - 1 ? kChunkSize - 100 : kChunkSize);
    } else if (c == 1 || c == 4) {
      mm.fill(p + c * kChunkSize, &pat[c == 1 ? 0 : 2], 1, kChunkSize);
    }
  }
  unsigned taken = 0, declined = 0, mixed = 0;
  for (int it = 0; it < 30000; ++it) {
    static const uint32_t widths[] = {1, 2, 4, 8};
    const uint32_t w = widths[rng.below(4)];
    size_t count = 1 + rng.below(32);
    if (it % 50 == 0) count = 33 + rng.below(40);   // more than a warp: the two-step path
    std::vector<uint64_t> addrs(count), out(count), want(count);
    const bool allreal = it % 3 == 0;   // a warp wholly on real chunks (the straight loop), or anywhere
    bool bad = false;
    unsigned kinds = 0;
    for (size_t i = 0; i < count; ++i) {
      const uint64_t span = allreal ? kChunkSize : total;
      uint64_t off = rng.below(span / w) * w;
      if (allreal) off += (rng.below(2) ? 3 : 6) * kChunkSize % (total - kChunkSize + 1);
      if (it % 7 == 0 && i == count / 2) {   // one lane that is wrong: past the end, below the base or misaligned
        const unsigned how = static_cast<unsigned>(rng.below(3));
        if (how == 0) off = total - w + w * (1 + rng.below(3));
        else if (how == 1) off = 0 - w * (1 + rng.below(3));
        else if (w > 1) off = (rng.below(total / w - 1)) * w + 1;
      }
      addrs[i] = p + off;
      if (off >= total || off + w > total || (off & (w - 1))) bad = true;
      else kinds |= 1u << (off / kChunkSize);
    }
    std::fill(out.begin(), out.end(), 0xDEADBEEFDEADBEEFull);
    const bool done = mm.load_batch(addrs.data(), count, w, out.data());
    if (bad) {
      VCHECK(!done);
      ++declined;
      continue;
    }
    VCHECK(done);
    ++taken;
    if ((kinds & 0x49u) && (kinds & ~0x49u)) ++mixed;
    for (size_t i = 0; i < count; ++i) {
      want[i] = mm.load_scalar(addrs[i], w);
      if (out[i] != want[i])
        throw vtest::Failure("lane " + std::to_string(i) + " of " + std::to_string(count) + " at width " + std::to_string(w) + " differs");
    }
    // The window form (the atomics' pair of checks and loads) gives the same values.
    MemoryManager::BatchWindow win;
    if (mm.batch_window(addrs.data(), count, w, &win)) {
      std::vector<uint64_t> again(count);
      mm.batch_load(win, addrs.data(), count, w, again.data());
      VCHECK(again == want);
    }
  }
  VCHECK(taken > 15000);
  VCHECK(declined > 1500);
  VCHECK(mixed > 3000);
}

VTEST_MAIN
