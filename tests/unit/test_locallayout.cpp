// The interleaved local-memory layout (include/vgpu/exec/local_layout.hpp) against the layout it replaced.
//
// A warp's local memory used to be one frame per lane, back to back (lane l's byte o at l * size + o). It is now interleaved by
// 4-byte words, so a warp-wide word access is contiguous. The reference here is the old layout, a plain array per lane; random
// stores and loads of every width (1 to 16 bytes, so also across words) at every offset, in a random lane, are made on both and
// every read must give the same bytes, and at the end the two must hold the same bytes (read back byte by byte through the
// layout). local_load / local_store (the sizes 1, 2, 4 and 8 the PTX interpreter uses, with the word-aligned shortcuts) are checked
// against local_read / local_write, and the array grown mid-run (a callee's frame) must keep what was written.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/exec/local_layout.hpp"
#include "vtest.hpp"

namespace {

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
  uint64_t below(uint64_t n) { return next() % n; }
};

using namespace vgpu::exec;

void run(uint32_t lanes, uint64_t seed) {
  Rng rng(seed);
  uint64_t per = 64 + rng.below(100);
  std::vector<std::vector<uint8_t>> ref(lanes, std::vector<uint8_t>(per, 0));
  std::vector<uint8_t> arr(local_array_bytes(lanes, per), 0);
  for (int it = 0; it < 60000; ++it) {
    if (it % 10000 == 9999) {   // a deeper frame: the array grows, every byte stays where it was
      const uint64_t more = per + 1 + rng.below(40);
      for (auto& r : ref) r.resize(more, 0);
      arr.resize(local_array_bytes(lanes, more), 0);
      per = more;
    }
    const uint32_t lane = static_cast<uint32_t>(rng.below(lanes));
    static const uint32_t widths[] = {1, 2, 4, 8, 4, 4, 1, 2, 3, 5, 6, 7, 12, 16};
    const uint32_t n = widths[rng.below(sizeof widths / sizeof widths[0])];
    // Aligned half the time (the shapes the executors take their shortcuts for), any offset otherwise.
    uint64_t off = rng.below(per - n + 1);
    if (rng.below(2) && n >= 2) off &= ~uint64_t{(n >= 4 ? 4 : n) - 1};
    if (off + n > per) off = per - n;
    if (rng.below(2)) {
      uint8_t v[16];
      for (uint32_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(rng.next());
      std::memcpy(&ref[lane][off], v, n);
      if (n <= 8 && rng.below(2)) {
        uint64_t x = 0;
        std::memcpy(&x, v, n);
        local_store(arr.data(), lanes, lane, off, n, x);
      } else {
        local_write(arr.data(), lanes, lane, off, v, n);
      }
    } else {
      uint8_t want[16], got[16] = {};
      std::memcpy(want, &ref[lane][off], n);
      if (n <= 8 && rng.below(2)) {
        const uint64_t x = local_load(arr.data(), lanes, lane, off, n);
        std::memcpy(got, &x, n);
      } else {
        local_read(arr.data(), lanes, lane, off, got, n);
      }
      if (std::memcmp(want, got, n) != 0)
        throw vtest::Failure("read of " + std::to_string(n) + " bytes at offset " + std::to_string(off) + " of lane " +
                             std::to_string(lane) + " differs from the per-lane layout at iteration " + std::to_string(it));
    }
  }
  for (uint32_t l = 0; l < lanes; ++l)
    for (uint64_t o = 0; o < per; ++o) {
      uint8_t b;
      local_read(arr.data(), lanes, l, o, &b, 1);
      if (b != ref[l][o]) throw vtest::Failure("final byte " + std::to_string(o) + " of lane " + std::to_string(l) + " differs");
    }
  // The point of the layout: the same word of every lane is contiguous.
  for (uint64_t o = 0; o + 4 <= per; o += 4)
    for (uint32_t l = 0; l + 1 < lanes; ++l)
      if (local_word(arr.data(), lanes, l + 1, o) != local_word(arr.data(), lanes, l, o) + 4)
        throw vtest::Failure("a warp-wide word is not contiguous");
}

}  // namespace

VTEST(LocalLayout32Lanes) { for (uint64_t seed = 1; seed <= 6; ++seed) run(32, seed); }
VTEST(LocalLayoutOtherWarpWidths) { run(64, 11); run(8, 12); }

VTEST_MAIN
