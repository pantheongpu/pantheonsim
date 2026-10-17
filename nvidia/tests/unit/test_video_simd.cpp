// The AVX2 kernels of the video codecs (src/video_simd.hpp) against the scalar loops they stand in for, which are kept here as the
// reference (they are nvenc_hevc_enc.cpp's fdct1d, idct1d, forward_transform, inverse_transform, mc_luma and mc_chroma as they were).
// Integer arithmetic, so equal means equal bits: random inputs, the extremes (all samples 0 or 255, residuals of +-255, coefficients
// of +-32768), every filter phase and every block shape the encoder uses.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../../src/video_simd.hpp"
#include "vtest.hpp"

namespace dec {
#include "../../src/hevc_tables.inc"   // the decoder's kTransMatrix (and its context tables)
}

namespace {

constexpr int kDctCoef[33] = {64, 90, 90, 90, 89, 88, 87, 85, 83, 82, 80, 78, 75, 73, 70, 67, 64, 61, 57, 54, 50, 46, 43, 38, 36, 31, 25, 22, 18, 13, 9, 4, 0};
struct DctMatrix {
  int8_t t[32][32];
  DctMatrix() {
    for (int k = 0; k < 32; ++k)
      for (int n = 0; n < 32; ++n) {
        if (k == 0) {
          t[k][n] = 64;
          continue;
        }
        int a = (k * (2 * n + 1)) % 128;
        if (a > 64) a = 128 - a;
        int sign = 1;
        if (a > 32) {
          a = 64 - a;
          sign = -1;
        }
        t[k][n] = static_cast<int8_t>(sign * kDctCoef[a]);
      }
  }
};
const DctMatrix& dct_matrix() {
  static const DctMatrix m;
  return m;
}
inline int clip3(int lo, int hi, int v) { return v < lo ? lo : (v > hi ? hi : v); }

void fdct1d(const int32_t* x, int32_t* y, int N) {
  if (N == 4) {
    const int32_t s0 = x[0] + x[3], s1 = x[1] + x[2], d0 = x[0] - x[3], d1 = x[1] - x[2];
    y[0] = 64 * (s0 + s1);
    y[2] = 64 * (s0 - s1);
    y[1] = 83 * d0 + 36 * d1;
    y[3] = 36 * d0 - 83 * d1;
    return;
  }
  const DctMatrix& M = dct_matrix();
  int32_t E[16] = {}, O[16] = {}, YE[16] = {};
  const int hn = N / 2;
  for (int n = 0; n < hn; ++n) {
    E[n] = x[n] + x[N - 1 - n];
    O[n] = x[n] - x[N - 1 - n];
  }
  fdct1d(E, YE, hn);
  for (int k = 0; k < hn; ++k) y[2 * k] = YE[k];
  const int step = 32 / N;
  for (int k = 1; k < N; k += 2) {
    int32_t s = 0;
    for (int n = 0; n < hn; ++n) s += M.t[k * step][n] * O[n];
    y[k] = s;
  }
}
void idct1d(const int32_t* y, int32_t* x, int N) {
  if (N == 4) {
    const int32_t o0 = 83 * y[1] + 36 * y[3], o1 = 36 * y[1] - 83 * y[3], e0 = 64 * (y[0] + y[2]), e1 = 64 * (y[0] - y[2]);
    x[0] = e0 + o0;
    x[1] = e1 + o1;
    x[2] = e1 - o1;
    x[3] = e0 - o0;
    return;
  }
  const DctMatrix& M = dct_matrix();
  const int hn = N / 2;
  int32_t ye[16] = {}, E[16] = {};
  for (int k = 0; k < hn; ++k) ye[k] = y[2 * k];
  idct1d(ye, E, hn);
  const int step = 32 / N;
  for (int n = 0; n < hn; ++n) {
    int32_t o = 0;
    for (int k = 1; k < N; k += 2) o += M.t[k * step][n] * y[k];
    x[n] = E[n] + o;
    x[N - 1 - n] = E[n] - o;
  }
}
void forward_transform(const int32_t* res, int32_t* coef, int log2) {
  const int N = 1 << log2;
  int32_t tmp[32 * 32], col[32], out[32];
  const int shift1 = log2 - 1, shift2 = log2 + 6;
  for (int y = 0; y < N; ++y) {
    fdct1d(res + y * N, out, N);
    for (int k = 0; k < N; ++k) tmp[y * N + k] = (out[k] + (1 << (shift1 - 1))) >> shift1;
  }
  for (int x = 0; x < N; ++x) {
    for (int y = 0; y < N; ++y) col[y] = tmp[y * N + x];
    fdct1d(col, out, N);
    for (int k = 0; k < N; ++k) coef[k * N + x] = (out[k] + (1 << (shift2 - 1))) >> shift2;
  }
}
void inverse_transform(const int32_t* d, int32_t* res, int log2) {
  const int N = 1 << log2;
  int32_t g[32 * 32], col[32], out[32];
  for (int x = 0; x < N; ++x) {
    for (int y = 0; y < N; ++y) col[y] = d[y * N + x];
    idct1d(col, out, N);
    for (int y = 0; y < N; ++y) g[y * N + x] = clip3(-32768, 32767, (out[y] + 64) >> 7);
  }
  for (int y = 0; y < N; ++y) {
    idct1d(g + y * N, out, N);
    for (int x = 0; x < N; ++x) res[y * N + x] = (out[x] + (1 << 11)) >> 12;
  }
}

constexpr int kLumaFilter[4][8] = {{0, 0, 0, 64, 0, 0, 0, 0}, {-1, 4, -10, 58, 17, -5, 1, 0}, {-1, 4, -11, 40, 40, -11, 4, -1}, {0, 1, -5, 17, 58, -10, 4, -1}};
constexpr int kChromaFilter[8][4] = {{0, 64, 0, 0}, {-2, 58, 10, -2}, {-4, 54, 16, -2}, {-6, 46, 28, -4}, {-4, 36, 36, -4}, {-4, 28, 46, -6}, {-2, 16, 54, -4}, {-2, 10, 58, -2}};

// The scalar interpolation: patch is (bw + taps - 1) wide, (bh + taps - 1) high, stride st; base points at the sample under (0, 0).
void ref_mc(const uint8_t* base, int st, int bw, int bh, int taps, int fx, int fy, const int* cx, const int* cy, int16_t* out) {
  const int before = taps / 2 - 1;
  if (!fy) {
    for (int j = 0; j < bh; ++j)
      for (int i = 0; i < bw; ++i) {
        const uint8_t* s = base + j * st + i - before;
        int sum = 0;
        for (int t = 0; t < taps; ++t) sum += cx[t] * s[t];
        out[j * bw + i] = static_cast<int16_t>(sum);
      }
    return;
  }
  if (!fx) {
    for (int j = 0; j < bh; ++j)
      for (int i = 0; i < bw; ++i) {
        const uint8_t* s = base + (j - before) * st + i;
        int sum = 0;
        for (int t = 0; t < taps; ++t) sum += cy[t] * s[t * st];
        out[j * bw + i] = static_cast<int16_t>(sum);
      }
    return;
  }
  int tmp[(32 + 7) * 32];
  for (int j = 0; j < bh + taps - 1; ++j)
    for (int i = 0; i < bw; ++i) {
      const uint8_t* s = base + (j - before) * st + i - before;
      int sum = 0;
      for (int t = 0; t < taps; ++t) sum += cx[t] * s[t];
      tmp[j * bw + i] = sum;
    }
  for (int j = 0; j < bh; ++j)
    for (int i = 0; i < bw; ++i) {
      int sum = 0;
      for (int t = 0; t < taps; ++t) sum += cy[t] * tmp[(j + t) * bw + i];
      out[j * bw + i] = static_cast<int16_t>(sum >> 6);
    }
}

std::mt19937_64 rng(2025);

}  // namespace

VTEST(video_simd_transforms_match_the_scalar_butterflies) {
  if (!vgpu::videosimd::avx2_available()) {
    std::printf("  no AVX2 on this host: nothing to compare\n");
    return;
  }
  const DctMatrix& M = dct_matrix();
  size_t bad = 0;
  int runs = 0;
  for (const int log2 : {3, 4, 5})
    for (int trial = 0; trial < 400; ++trial) {
      const int N = 1 << log2;
      std::vector<int32_t> res(N * N), coef_s(N * N), coef_v(N * N, 12345), back_s(N * N), back_v(N * N, 777), d(N * N);
      const int kind = trial % 4;
      for (auto& v : res) v = kind == 0 ? static_cast<int>(rng() % 511) - 255 : kind == 1 ? (rng() % 2 ? 255 : -255) : kind == 2 ? static_cast<int>(rng() % 31) - 15 : 0;
      for (auto& v : d) v = kind == 0 ? static_cast<int>(rng() % 65536) - 32768 : kind == 1 ? (rng() % 2 ? 32767 : -32768) : kind == 2 ? static_cast<int>(rng() % 201) - 100 : (rng() % 5 == 0 ? static_cast<int>(rng() % 65536) - 32768 : 0);
      forward_transform(res.data(), coef_s.data(), log2);
      vgpu::videosimd::forward_dct(res.data(), coef_v.data(), log2, M.t);
      inverse_transform(d.data(), back_s.data(), log2);
      vgpu::videosimd::inverse_dct(d.data(), back_v.data(), log2, M.t);
      bad += std::memcmp(coef_s.data(), coef_v.data(), coef_s.size() * 4) != 0;
      bad += std::memcmp(back_s.data(), back_v.data(), back_s.size() * 4) != 0;
      ++runs;
    }
  std::printf("  transforms: %d blocks, %zu differ\n", runs, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST(video_simd_interpolation_matches_the_scalar_filters) {
  if (!vgpu::videosimd::avx2_available()) {
    std::printf("  no AVX2 on this host: nothing to compare\n");
    return;
  }
  size_t bad = 0;
  int runs = 0;
  struct Shape { int bw, bh; };
  for (const int taps : {8, 4})
    for (const Shape s : {Shape{4, 4}, Shape{8, 4}, Shape{4, 8}, Shape{8, 8}, Shape{12, 16}, Shape{16, 4}, Shape{16, 16}, Shape{24, 32}, Shape{32, 8}, Shape{32, 32},
                          Shape{16, 12}, Shape{4, 16}}) {
      if (taps == 4 && (s.bw > 16 || s.bh > 16)) continue;
      const int phases = taps == 8 ? 4 : 8;
      for (int fx = 0; fx < phases; ++fx)
        for (int fy = 0; fy < phases; ++fy) {
          if (!fx && !fy) continue;
          for (int kind = 0; kind < 4; ++kind) {
            const int st = s.bw + taps - 1, rows = s.bh + taps - 1, before = taps / 2 - 1;
            std::vector<uint8_t> patch(static_cast<size_t>(st) * rows);
            for (auto& v : patch) v = kind == 0 ? static_cast<uint8_t>(rng()) : kind == 1 ? (rng() % 2 ? 255 : 0) : kind == 2 ? 255 : static_cast<uint8_t>((rng() % 4) ? 0 : 255);
            const uint8_t* base = patch.data() + before * st + before;
            const int* cx = taps == 8 ? kLumaFilter[fx] : kChromaFilter[fx];
            const int* cy = taps == 8 ? kLumaFilter[fy] : kChromaFilter[fy];
            std::vector<int16_t> want(s.bw * s.bh), got(s.bw * s.bh, 4242);
            ref_mc(base, st, s.bw, s.bh, taps, fx, fy, cx, cy, want.data());
            if (!fy) vgpu::videosimd::fir_h(base - before, st, s.bw, s.bh, taps, cx, got.data());
            else if (!fx) vgpu::videosimd::fir_v(base - before * st, st, s.bw, s.bh, taps, cy, got.data());
            else vgpu::videosimd::fir_hv(base - before * st - before, st, s.bw, s.bh, taps, cx, cy, got.data());
            bad += std::memcmp(want.data(), got.data(), want.size() * 2) != 0;
            ++runs;
          }
        }
    }
  std::printf("  interpolation: %d blocks, %zu differ\n", runs, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST(video_simd_sums_of_absolute_differences_match) {
  if (!vgpu::videosimd::avx2_available()) {
    std::printf("  no AVX2 on this host: nothing to compare\n");
    return;
  }
  size_t bad = 0;
  int runs = 0;
  struct Shape { int bw, bh; };
  for (const Shape s : {Shape{4, 4}, Shape{8, 4}, Shape{12, 16}, Shape{16, 8}, Shape{24, 32}, Shape{32, 32}, Shape{16, 16}, Shape{4, 32}, Shape{20, 12}, Shape{36, 4}})
    for (int kind = 0; kind < 6; ++kind) {
      const int ss = s.bw + 5, bs = s.bw + 3;
      std::vector<uint8_t> src(static_cast<size_t>(ss) * s.bh), other(static_cast<size_t>(bs) * s.bh);
      std::vector<int16_t> p(s.bw * s.bh), q(s.bw * s.bh);
      for (auto& v : src) v = kind == 0 ? static_cast<uint8_t>(rng()) : kind == 1 ? (rng() % 2 ? 255 : 0) : kind == 2 ? 255 : 0;
      for (auto& v : other) v = kind == 0 ? static_cast<uint8_t>(rng()) : kind == 1 ? (rng() % 2 ? 0 : 255) : kind == 2 ? 0 : 255;
      // the 14-bit prediction range of the filters, and its extremes (a sample of 255 through a filter of gain 1.75, and the negative lobes)
      for (auto& v : p) v = kind == 0 ? static_cast<int16_t>(static_cast<int>(rng() % 20000) - 2000) : kind == 1 ? (rng() % 2 ? 28560 : -28560) : kind == 2 ? 16320 : static_cast<int16_t>(static_cast<int>(rng() % 65536) - 32768);
      for (auto& v : q) v = kind == 0 ? static_cast<int16_t>(static_cast<int>(rng() % 20000) - 2000) : kind == 1 ? (rng() % 2 ? 28560 : -28560) : kind == 2 ? 16320 : static_cast<int16_t>(static_cast<int>(rng() % 65536) - 32768);
      int want_u8 = 0, want_uni = 0, want_bi = 0;
      for (int j = 0; j < s.bh; ++j)
        for (int i = 0; i < s.bw; ++i) {
          want_u8 += std::abs(src[j * ss + i] - other[j * bs + i]);
          const int vu = clip3(0, 255, (p[j * s.bw + i] + 32) >> 6), vb = clip3(0, 255, (p[j * s.bw + i] + q[j * s.bw + i] + 64) >> 7);
          want_uni += std::abs(src[j * ss + i] - vu);
          want_bi += std::abs(src[j * ss + i] - vb);
        }
      bad += vgpu::videosimd::sad_u8(src.data(), ss, other.data(), bs, s.bw, s.bh) != want_u8;
      bad += vgpu::videosimd::sad_uni(p.data(), s.bw, s.bh, src.data(), ss) != want_uni;
      bad += vgpu::videosimd::sad_bi(p.data(), q.data(), s.bw, s.bh, src.data(), ss) != want_bi;
      ++runs;
    }
  std::printf("  sums of absolute differences: %d blocks, %zu differ\n", runs, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

// The HEVC decoder's inverse transform (hevc_dec_residual.inc) takes the same butterflies as the encoder with its own matrix and the
// shift of its bit depth. The reference is its dense sum over 64-bit integers, as it was.
namespace {
void ref_dec_idct_1d(const int32_t* x, int xstride, int32_t* y, int n, int last_nz) {
  const int step = 32 / n;
  for (int i = 0; i < n; ++i) {
    int64_t sum = 0;
    for (int j = 0; j <= last_nz; ++j) {
      const int32_t v = x[j * xstride];
      if (v) sum += static_cast<int64_t>(dec::kTransMatrix[j * step][i]) * v;
    }
    y[i] = static_cast<int32_t>(sum);
  }
}
void ref_dec_inverse(int32_t* coeff, int log2_tb, int bd, int32_t* res) {
  const int n = 1 << log2_tb;
  int32_t tmp[32 * 32], col[32];
  for (int x = 0; x < n; ++x) {
    ref_dec_idct_1d(coeff + x, n, col, n, n - 1);
    for (int y = 0; y < n; ++y) {
      int32_t g = (col[y] + 64) >> 7;
      g = g < -32768 ? -32768 : (g > 32767 ? 32767 : g);
      tmp[y * n + x] = g;
    }
  }
  const int shift = 20 - bd;
  const int32_t round = shift > 0 ? 1 << (shift - 1) : 0;
  for (int y = 0; y < n; ++y) {
    int32_t row[32];
    ref_dec_idct_1d(tmp + y * n, 1, row, n, n - 1);
    for (int x = 0; x < n; ++x) res[y * n + x] = (row[x] + round) >> shift;
  }
}
}  // namespace

VTEST(video_simd_decoder_inverse_transform_matches_the_dense_sums) {
  if (!vgpu::videosimd::avx2_available()) {
    std::printf("  no AVX2 on this host: nothing to compare\n");
    return;
  }
  VCHECK(std::memcmp(dec::kTransMatrix, dct_matrix().t, sizeof dec::kTransMatrix) == 0);   // the two codecs' matrices are one
  size_t bad = 0;
  int runs = 0;
  for (const int log2 : {3, 4, 5})
    for (const int bd : {8, 10, 12})
      for (int trial = 0; trial < 150; ++trial) {
        const int N = 1 << log2;
        std::vector<int32_t> d(N * N), want(N * N), got(N * N, 99);
        const int kind = trial % 4;
        for (auto& v : d) v = kind == 0 ? static_cast<int>(rng() % 65536) - 32768 : kind == 1 ? (rng() % 2 ? 32767 : -32768) : kind == 2 ? static_cast<int>(rng() % 41) - 20 : (rng() % 6 == 0 ? static_cast<int>(rng() % 65536) - 32768 : 0);
        std::vector<int32_t> copy = d;
        ref_dec_inverse(copy.data(), log2, bd, want.data());
        vgpu::videosimd::inverse_dct(d.data(), got.data(), log2, dec::kTransMatrix, 20 - bd);
        bad += std::memcmp(want.data(), got.data(), want.size() * 4) != 0;
        ++runs;
      }
  std::printf("  decoder inverse transform: %d blocks, %zu differ\n", runs, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

// MPEG-2's inverse DCT (mpeg2_idct.inc's idct as it was: two passes of eight-term sums in double, each term rounded then added, the
// result floor(s + 0.5) saturated to -256 .. 255) against the vector form.
void ref_mpeg2_idct(int16_t* blk) {
  static double c[8][8];
  static bool init = false;
  if (!init) {
    for (int x = 0; x < 8; ++x)
      for (int u = 0; u < 8; ++u) c[x][u] = (u == 0 ? std::sqrt(0.5) : 1.0) * std::cos((2 * x + 1) * u * 3.14159265358979323846 / 16) / 2;
    init = true;
  }
  double tmp[64];
  for (int v = 0; v < 8; ++v)
    for (int x = 0; x < 8; ++x) {
      double s = 0;
      for (int u = 0; u < 8; ++u) s += c[x][u] * blk[v * 8 + u];
      tmp[v * 8 + x] = s;
    }
  for (int x = 0; x < 8; ++x)
    for (int y = 0; y < 8; ++y) {
      double s = 0;
      for (int v = 0; v < 8; ++v) s += c[y][v] * tmp[v * 8 + x];
      const int r = static_cast<int>(std::floor(s + 0.5));
      blk[y * 8 + x] = static_cast<int16_t>(std::min(std::max(r, -256), 255));
    }
}

VTEST(mpeg2_idct_matches_the_scalar_loops) {
  if (!vgpu::videosimd::avx2_available()) return;
  static double c[8][8];
  for (int x = 0; x < 8; ++x)
    for (int u = 0; u < 8; ++u) c[x][u] = (u == 0 ? std::sqrt(0.5) : 1.0) * std::cos((2 * x + 1) * u * 3.14159265358979323846 / 16) / 2;
  std::mt19937_64 rng(77);
  size_t bad = 0;
  int blocks = 0;
  for (int trial = 0; trial < 200000; ++trial) {
    int16_t a[64], b[64];
    const int kind = trial % 6;
    for (int i = 0; i < 64; ++i) {
      switch (kind) {
        case 0: a[i] = static_cast<int16_t>(static_cast<int>(rng() % 4096) - 2048); break;       // what an 8-bit stream codes
        case 1: a[i] = static_cast<int16_t>(rng()); break;                                       // anything 16 bits hold
        case 2: a[i] = rng() % 2 ? 32767 : -32768; break;                                        // the extremes
        case 3: a[i] = rng() % 8 == 0 ? static_cast<int16_t>(static_cast<int>(rng() % 801) - 400) : 0; break;   // sparse
        case 4: a[i] = i == 0 ? static_cast<int16_t>(static_cast<int>(rng() % 4096) - 2048) : 0; break;        // DC only
        default: a[i] = static_cast<int16_t>(static_cast<int>(rng() % 9) - 4); break;             // small: sums land near halves
      }
      b[i] = a[i];
    }
    ref_mpeg2_idct(a);
    vgpu::videosimd::mpeg2_idct(b, c);
    bad += std::memcmp(a, b, sizeof a) != 0;
    ++blocks;
  }
  std::printf("  MPEG-2 inverse DCT: %d blocks, %zu differ\n", blocks, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST_MAIN
