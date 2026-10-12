// The SASS executor's FFMA lane loop on the CPU's FMA unit (src/sass/lane_math.hpp)
// against the libm version it replaces for speed. Both are the correctly
// rounded fused multiply-add and must agree to the bit on every input: zeros of
// both signs, subnormals, the extremes, infinities, NaNs (quiet and signalling,
// with payloads -- the executor makes any NaN result 0x7fffffff), values whose
// product needs more than 24 bits to be exact, and random bit patterns.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../../src/sass/lane_math.hpp"
#include "vtest.hpp"

namespace {

struct Rng {
  uint64_t s = 0x2545F4914F6CDD1Dull;
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<uint32_t>(s >> 11);
  }
};

const std::vector<uint32_t>& special() {
  static const std::vector<uint32_t> v = {
      0x00000000, 0x80000000,                          // +0, -0
      0x00000001, 0x80000001, 0x007fffff, 0x807fffff,  // subnormals
      0x00800000, 0x80800000,                          // smallest normals
      0x3f800000, 0xbf800000, 0x40000000, 0x3effffff,  // 1, -1, 2, just under 1/2
      0x7f7fffff, 0xff7fffff,                          // max finite
      0x7f800000, 0xff800000,                          // infinities
      0x7fc00000, 0xffc00000, 0x7fc12345, 0x7f800001,  // NaNs: quiet, negative, payload, signalling
      0xffbfffff, 0x7fffffff,
      0x4b000000, 0x4b800001, 0x33800000, 0x3f800001,  // 2^23, 2^24+, 2^-24, 1+ulp
      0x0f800000, 0x7e800000, 0x1e800000, 0x61800000};
  return v;
}

void compare(const uint32_t* a, const uint32_t* b, const uint32_t* c, const char* what) {
  uint32_t want[32], got[32], got_hw[32];
  vgpu::sass::ffma_lanes_libm(want, a, b, c);
  vgpu::sass::ffma_lanes(got, a, b, c);
  if (std::memcmp(want, got, sizeof want) != 0) {
    for (unsigned l = 0; l < 32; ++l)
      if (want[l] != got[l]) {
        char m[200];
        std::snprintf(m, sizeof m, "%s: lane %u fma(%08x, %08x, %08x) = %08x, libm says %08x", what, l, a[l], b[l], c[l], got[l], want[l]);
        throw vtest::Failure(m);
      }
  }
  if (vgpu::sass::ffma_lanes_hw_available()) {
    vgpu::sass::ffma_lanes_hw(got_hw, a, b, c);
    if (std::memcmp(want, got_hw, sizeof want) != 0) throw vtest::Failure(std::string(what) + ": hardware loop differs from libm");
  }
}

}  // namespace

VTEST(sass_ffma_lanes_special_values) {
  const auto& v = special();
  uint32_t a[32], b[32], c[32];
  // Every triple of special values, 32 at a time.
  std::vector<std::array<uint32_t, 3>> triples;
  for (uint32_t x : v)
    for (uint32_t y : v)
      for (uint32_t z : v) triples.push_back({x, y, z});
  for (size_t i = 0; i < triples.size(); i += 32) {
    for (unsigned l = 0; l < 32; ++l) {
      const auto& t = triples[(i + l) % triples.size()];
      a[l] = t[0];
      b[l] = t[1];
      c[l] = t[2];
    }
    compare(a, b, c, "special");
  }
}

VTEST(sass_ffma_lanes_random_bits) {
  Rng rng;
  uint32_t a[32], b[32], c[32];
  for (int iter = 0; iter < 20000; ++iter) {
    for (unsigned l = 0; l < 32; ++l) {
      a[l] = rng.next() ^ (rng.next() << 20);
      b[l] = rng.next() ^ (rng.next() << 20);
      c[l] = rng.next() ^ (rng.next() << 20);
      // Often make the product and addend cancel to exercise the rounding.
      if (iter % 3 == 0) c[l] = (a[l] & 0x80000000u ? 0u : 0x80000000u) ^ (rng.next() & 0x7f800000u) ^ (b[l] & 0x007fffffu);
    }
    compare(a, b, c, "random");
  }
}

namespace {
double dbl(uint64_t b) {
  double d;
  std::memcpy(&d, &b, 8);
  return d;
}
void compare_d(const double* a, const double* b, const double* c) {
  double want[32], got[32], hw[32];
  vgpu::sass::dfma_lanes_libm(want, a, b, c);
  vgpu::sass::dfma_lanes(got, a, b, c);
  vgpu::sass::dfma_lanes_hw(hw, a, b, c);
  for (unsigned l = 0; l < 32; ++l) {
    uint64_t w, g, h;
    std::memcpy(&w, &want[l], 8);
    std::memcpy(&g, &got[l], 8);
    std::memcpy(&h, &hw[l], 8);
    // A NaN's payload may differ between libm and the instruction; any NaN is a NaN, and every other result is exact.
    const bool both_nan = want[l] != want[l] && got[l] != got[l] && hw[l] != hw[l];
    if (!both_nan && (w != g || w != h)) {
      char m[200];
      std::snprintf(m, sizeof m, "lane %u fma(%016llx, %016llx, %016llx): libm %016llx dispatch %016llx hw %016llx", l,
                    (unsigned long long)*(const uint64_t*)&a[l], (unsigned long long)*(const uint64_t*)&b[l],
                    (unsigned long long)*(const uint64_t*)&c[l], (unsigned long long)w, (unsigned long long)g, (unsigned long long)h);
      throw vtest::Failure(m);
    }
  }
}
}  // namespace

VTEST(sass_dfma_lanes_special_and_random) {
  const std::vector<uint64_t> sp = {0x0000000000000000ull, 0x8000000000000000ull, 0x0000000000000001ull,
                                    0x000fffffffffffffull, 0x0010000000000000ull, 0x3ff0000000000000ull,
                                    0xbff0000000000000ull, 0x7fefffffffffffffull, 0x7ff0000000000000ull,
                                    0xfff0000000000000ull, 0x7ff8000000000000ull, 0x7ff0000000000001ull,
                                    0x4340000000000000ull, 0x3ca0000000000000ull, 0x3ff0000000000001ull};
  double a[32], b[32], c[32];
  std::vector<std::array<uint64_t, 3>> triples;
  for (uint64_t x : sp)
    for (uint64_t y : sp)
      for (uint64_t z : sp) triples.push_back({x, y, z});
  for (size_t i = 0; i < triples.size(); i += 32) {
    for (unsigned l = 0; l < 32; ++l) {
      const auto& t = triples[(i + l) % triples.size()];
      a[l] = dbl(t[0]);
      b[l] = dbl(t[1]);
      c[l] = dbl(t[2]);
    }
    compare_d(a, b, c);
  }
  Rng rng;
  for (int iter = 0; iter < 20000; ++iter) {
    for (unsigned l = 0; l < 32; ++l) {
      const uint64_t x = (uint64_t{rng.next()} << 32) ^ rng.next(), y = (uint64_t{rng.next()} << 32) ^ rng.next(),
                     z = (uint64_t{rng.next()} << 32) ^ rng.next();
      a[l] = dbl(x);
      b[l] = dbl(y);
      c[l] = iter % 3 == 0 ? -(a[l] * b[l]) : dbl(z);   // cancellation: the rounding of the product matters
    }
    compare_d(a, b, c);
  }
}

VTEST_MAIN
