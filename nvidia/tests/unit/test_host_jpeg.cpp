// The inverse DCT of nvJPEG's host decoder (src/host_jpeg.hpp), bit for bit against the scalar loops it replaced
// (the same function, kept in the header as idct8x8_scalar and below as the reference's own copy of the original).
// Blocks of dequantised coefficients: random in the range real files use, the extremes of what a 16-bit coefficient
// times a 16-bit quantiser can reach, sparse blocks (DC only, one AC), and all zeros.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../../src/host_jpeg.hpp"
#include "vtest.hpp"

namespace {

// nvjpeg_api.cpp's idct8x8 as it was before host_jpeg.hpp, verbatim apart from the name.
const double (*dct_basis())[8] {
  static double c[8][8];
  static bool once = false;
  if (!once) {
    for (int u = 0; u < 8; ++u)
      for (int x = 0; x < 8; ++x)
        c[u][x] = (u == 0 ? std::sqrt(0.125) : 0.5) * std::cos((2 * x + 1) * u * M_PI / 16.0);
    once = true;
  }
  return c;
}

void original_idct8x8(const int* in, uint8_t* out, int stride) {
  const double(*cd)[8] = dct_basis();
  float c[8][8];
  for (int u = 0; u < 8; ++u) for (int x = 0; x < 8; ++x) c[u][x] = static_cast<float>(cd[u][x]);
  float tmp[64];
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x) {
      float s = 0;
      for (int u = 0; u < 8; ++u) s = std::fmaf(c[u][x], static_cast<float>(in[y * 8 + u]), s);
      tmp[y * 8 + x] = s;
    }
  for (int x = 0; x < 8; ++x)
    for (int y = 0; y < 8; ++y) {
      float s = 0;
      for (int v = 0; v < 8; ++v) s = std::fmaf(c[v][y], tmp[v * 8 + x], s);
      const float r = std::floor(s + 128.5f);
      out[y * stride + x] = static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
    }
}

}  // namespace

VTEST(host_idct_matches_the_scalar_loops) {
  std::mt19937_64 rng(31337);
  long blocks = 0, bad = 0;
  for (int round = 0; round < 60000; ++round) {
    int in[64] = {};
    switch (round % 6) {
      case 0: for (auto& v : in) v = static_cast<int>(rng() % 2001) - 1000; break;
      case 1: for (auto& v : in) v = static_cast<int>(rng() % 65535) - 32767; break;
      case 2: for (auto& v : in) v = (rng() & 1 ? 1 : -1) * static_cast<int>(rng() % 32768) * static_cast<int>(1 + rng() % 65535); break;   // the extremes
      case 3: in[0] = static_cast<int>(rng() % 4001) - 2000; break;                                                                      // DC only
      case 4: in[0] = static_cast<int>(rng() % 2001) - 1000; in[rng() % 64] = static_cast<int>(rng() % 801) - 400; break;               // one AC
      default: break;                                                                                                                   // zeros
    }
    uint8_t want[8 * 12], got[8 * 12];
    std::memset(want, 0x5a, sizeof want);
    std::memset(got, 0x5a, sizeof got);
    original_idct8x8(in, want, 12);
    for (const bool wide : {false, true}) {
      if (wide && !vgpu::hostjpeg::wide_available()) continue;
      vgpu::hostjpeg::idct8x8(in, got, 12, wide);
      ++blocks;
      if (std::memcmp(want, got, sizeof want) != 0) ++bad;   // (the bytes between rows too: nothing else is written)
    }
  }
  std::printf("  %ld transforms, %ld differ\n", blocks, bad);
  VCHECK_EQ(bad, 0L);
}

namespace {
uint8_t orig_to_byte(float v) {
  const float r = std::nearbyint(v);
  return static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
}
}  // namespace

// nvjpeg_api.cpp's ycbcr_to_rgb and the luma of its Y output, per pixel as they were, against the row versions: every
// (y, cb, cr) triple of a coarse grid including all the edges, plus random triples, at lengths that leave a tail.
VTEST(host_colour_rows_match_the_per_pixel_code) {
  std::mt19937_64 rng(4242);
  long px = 0, bad = 0;
  for (int round = 0; round < 3000; ++round) {
    const int n = 1 + static_cast<int>(rng() % 70);
    std::vector<uint8_t> y(n), cb(n), cr(n);
    for (int i = 0; i < n; ++i) {
      const bool edge = rng() % 4 == 0;
      y[i] = edge ? (rng() % 2 ? 0 : 255) : static_cast<uint8_t>(rng());
      cb[i] = edge ? (rng() % 2 ? 0 : 255) : static_cast<uint8_t>(rng());
      cr[i] = edge ? (rng() % 2 ? 0 : 255) : static_cast<uint8_t>(rng());
    }
    for (const bool wide : {false, true}) {
      if (wide && !vgpu::hostjpeg::avx2_available()) continue;
      std::vector<uint8_t> r(n), g(n), b(n), l(n);
      vgpu::hostjpeg::ycc_to_rgb(y.data(), cb.data(), cr.data(), r.data(), g.data(), b.data(), n, wide);
      vgpu::hostjpeg::rgb_to_luma(y.data(), cb.data(), cr.data(), l.data(), n, wide);
      for (int i = 0; i < n; ++i) {
        const float yy = y[i], bb = cb[i], rr = cr[i];
        const uint8_t wr = orig_to_byte(yy + 1.402f * (rr - 128.0f));
        const uint8_t wg = orig_to_byte(yy - 0.344136f * (bb - 128.0f) - 0.714136f * (rr - 128.0f));
        const uint8_t wb = orig_to_byte(yy + 1.772f * (bb - 128.0f));
        const uint8_t wl = orig_to_byte(0.299f * y[i] + 0.587f * cb[i] + 0.114f * cr[i]);
        ++px;
        if (r[i] != wr || g[i] != wg || b[i] != wb || l[i] != wl) ++bad;
      }
    }
  }
  std::printf("  %ld pixels, %ld differ\n", px, bad);
  VCHECK_EQ(bad, 0L);
}

// All 16.7 million (y, cb, cr) triples, one row of cr per y and cb.
VTEST(host_colour_rows_match_for_every_triple) {
  long bad = 0;
  std::vector<uint8_t> y(256), cb(256), cr(256), r(256), g(256), b(256);
  for (int i = 0; i < 256; ++i) cr[i] = static_cast<uint8_t>(i);
  for (const bool wide : {false, true}) {
    if (wide && !vgpu::hostjpeg::avx2_available()) continue;
    for (int yy = 0; yy < 256; ++yy)
      for (int bb = 0; bb < 256; ++bb) {
        for (int i = 0; i < 256; ++i) y[i] = static_cast<uint8_t>(yy), cb[i] = static_cast<uint8_t>(bb);
        vgpu::hostjpeg::ycc_to_rgb(y.data(), cb.data(), cr.data(), r.data(), g.data(), b.data(), 256, wide);
        for (int i = 0; i < 256; ++i) {
          const float fy = y[i], fb = cb[i], fr = cr[i];
          if (r[i] != orig_to_byte(fy + 1.402f * (fr - 128.0f)) || g[i] != orig_to_byte(fy - 0.344136f * (fb - 128.0f) - 0.714136f * (fr - 128.0f)) ||
              b[i] != orig_to_byte(fy + 1.772f * (fb - 128.0f)))
            ++bad;
        }
      }
  }
  VCHECK_EQ(bad, 0L);
}

namespace {
// nvjpeg_api.cpp's fdct8x8 as it was.
void original_fdct8x8(const double* in, double* out) {
  const double(*c)[8] = dct_basis();
  double tmp[64];
  for (int y = 0; y < 8; ++y)
    for (int u = 0; u < 8; ++u) {
      double s = 0;
      for (int x = 0; x < 8; ++x) s += c[u][x] * in[y * 8 + x];
      tmp[y * 8 + u] = s;
    }
  for (int u = 0; u < 8; ++u)
    for (int v = 0; v < 8; ++v) {
      double s = 0;
      for (int y = 0; y < 8; ++y) s += c[v][y] * tmp[y * 8 + u];
      out[v * 8 + u] = s;
    }
}
}  // namespace

VTEST(host_fdct_matches_the_scalar_loops) {
  std::mt19937_64 rng(808);
  std::uniform_real_distribution<double> u(-128.0, 127.0);
  long blocks = 0, bad = 0;
  for (int round = 0; round < 40000; ++round) {
    double in[64];
    for (auto& v : in) v = round % 4 == 0 ? static_cast<double>(static_cast<int>(rng() % 256) - 128) : u(rng);
    if (round % 7 == 0) for (auto& v : in) v = 0.0;
    double want[64], got[64];
    original_fdct8x8(in, want);
    for (const bool wide : {false, true}) {
      if (wide && !vgpu::hostjpeg::avx2_available()) continue;
      vgpu::hostjpeg::fdct8x8(in, got, wide);
      ++blocks;
      if (std::memcmp(want, got, sizeof want) != 0) ++bad;
    }
  }
  std::printf("  %ld forward transforms, %ld differ\n", blocks, bad);
  VCHECK_EQ(bad, 0L);
}

// round_half_away is std::lround: halves away from zero, the doubles just below and above a half, integers, huge values, NaN.
VTEST(host_round_half_away_is_lround) {
  std::mt19937_64 rng(5150);
  long checked = 0, bad = 0;
  auto check = [&](double x) {
    ++checked;
    const long want = std::lround(x), got = vgpu::hostjpeg::round_half_away(x);
    if (want != got && !(std::isnan(x))) ++bad;   // (lround of a NaN is unspecified; round_half_away hands it to lround)
  };
  for (long i = -2000; i <= 2000; ++i) {
    const double base = static_cast<double>(i) / 4.0;   // multiples of a quarter: halves and their neighbours
    check(base);
    check(std::nextafter(base, 1e300));
    check(std::nextafter(base, -1e300));
  }
  for (int i = 0; i < 2000000; ++i) {
    uint64_t bits = rng();
    double x;
    std::memcpy(&x, &bits, 8);
    check(x);   // every exponent, including huge, tiny, infinities
    check((static_cast<double>(rng() % 20000001) - 1e7) / 1000.0);
    check((static_cast<double>(rng() % 2001) - 1000.0) / 7.0);
  }
  for (double x : {0.49999999999999994, -0.49999999999999994, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 4503599627370495.5, 4503599627370496.0, 9007199254740993.0, 3.9e18, 4.1e18, 1e19, -1e19})
    check(x);
  std::printf("  %ld values, %ld differ\n", checked, bad);
  VCHECK_EQ(bad, 0L);
}

VTEST_MAIN
