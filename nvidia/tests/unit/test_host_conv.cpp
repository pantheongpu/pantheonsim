// The host convolution behind cuDNN (src/host_conv.hpp), bit for bit against the element-by-element loops it
// replaced, which are kept here as the reference (they are cudnn_common.cpp's convolve before host_conv.hpp, for
// Accum::Exact and Accum::Int). Every output is one chain s = s + x*w in a fixed order of taps; the kernels run
// the chains of a row side by side and must give each chain's exact bits, on both instruction sets and for any
// thread count, over padding, strides, dilation, groups, flipped filters and one to three spatial dimensions, with
// inputs that break arithmetic (signed zeros, subnormals, infinities, NaN, the extremes).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "../../src/host_conv.hpp"
#include "vtest.hpp"

namespace {

using vgpu::hostconv::Algo;
using vgpu::hostconv::Dir;
using vgpu::hostconv::Geom;
using vgpu::hostconv::Isa;

void ref_convolve(const Geom& g, Dir dir, const std::vector<double>& a, const std::vector<double>& b, std::vector<double>* out) {
  const int64_t I0 = g.in[0], I1 = g.in[1], I2 = g.in[2];
  const int64_t O0 = g.out[0], O1 = g.out[1], O2 = g.out[2];
  const int64_t F0 = g.flt[0], F1 = g.flt[1], F2 = g.flt[2];
  const int64_t isz = I0 * I1 * I2, osz = O0 * O1 * O2, fsz = F0 * F1 * F2;
  auto in_pos = [&](int d, int64_t o, int64_t f) {
    const int64_t tap = g.flip ? g.flt[d] - 1 - f : f;
    return o * g.str[d] - g.pad[d] + tap * g.dil[d];
  };
  auto add = [&](double s, double x, double y) { return s + x * y; };
  switch (dir) {
    case Dir::Forward: {
      out->assign(static_cast<size_t>(g.N * g.K * osz), 0.0);
      for (int64_t n = 0; n < g.N; ++n)
        for (int64_t k = 0; k < g.K; ++k) {
          const int64_t c0 = (k / g.Kg) * g.Cg;
          for (int64_t o0 = 0; o0 < O0; ++o0)
            for (int64_t o1 = 0; o1 < O1; ++o1)
              for (int64_t o2 = 0; o2 < O2; ++o2) {
                double s = 0.0;
                for (int64_t ci = 0; ci < g.Cg; ++ci) {
                  const double* xb = a.data() + ((n * g.C + c0 + ci) * isz);
                  const double* wb = b.data() + ((k * g.Cg + ci) * fsz);
                  for (int64_t f0 = 0; f0 < F0; ++f0) {
                    const int64_t p0 = in_pos(0, o0, f0);
                    if (p0 < 0 || p0 >= I0) continue;
                    for (int64_t f1 = 0; f1 < F1; ++f1) {
                      const int64_t p1 = in_pos(1, o1, f1);
                      if (p1 < 0 || p1 >= I1) continue;
                      for (int64_t f2 = 0; f2 < F2; ++f2) {
                        const int64_t p2 = in_pos(2, o2, f2);
                        if (p2 < 0 || p2 >= I2) continue;
                        s = add(s, xb[(p0 * I1 + p1) * I2 + p2], wb[(f0 * F1 + f1) * F2 + f2]);
                      }
                    }
                  }
                }
                (*out)[static_cast<size_t>((n * g.K + k) * osz + (o0 * O1 + o1) * O2 + o2)] = s;
              }
        }
      break;
    }
    case Dir::Data: {
      out->assign(static_cast<size_t>(g.N * g.C * isz), 0.0);
      auto out_pos = [&](int d, int64_t p, int64_t f, int64_t* o) {
        const int64_t tap = g.flip ? g.flt[d] - 1 - f : f;
        const int64_t num = p + g.pad[d] - tap * g.dil[d];
        if (num < 0 || num % g.str[d]) return false;
        *o = num / g.str[d];
        return *o < g.out[d];
      };
      for (int64_t n = 0; n < g.N; ++n)
        for (int64_t c = 0; c < g.C; ++c) {
          const int64_t grp = c / g.Cg, ci = c % g.Cg;
          for (int64_t p0 = 0; p0 < I0; ++p0)
            for (int64_t p1 = 0; p1 < I1; ++p1)
              for (int64_t p2 = 0; p2 < I2; ++p2) {
                double s = 0.0;
                for (int64_t kk = 0; kk < g.Kg; ++kk) {
                  const int64_t k = grp * g.Kg + kk;
                  const double* yb = a.data() + ((n * g.K + k) * osz);
                  const double* wb = b.data() + ((k * g.Cg + ci) * fsz);
                  int64_t o0, o1, o2;
                  for (int64_t f0 = 0; f0 < F0; ++f0) {
                    if (!out_pos(0, p0, f0, &o0)) continue;
                    for (int64_t f1 = 0; f1 < F1; ++f1) {
                      if (!out_pos(1, p1, f1, &o1)) continue;
                      for (int64_t f2 = 0; f2 < F2; ++f2) {
                        if (!out_pos(2, p2, f2, &o2)) continue;
                        s = add(s, yb[(o0 * O1 + o1) * O2 + o2], wb[(f0 * F1 + f1) * F2 + f2]);
                      }
                    }
                  }
                }
                (*out)[static_cast<size_t>((n * g.C + c) * isz + (p0 * I1 + p1) * I2 + p2)] = s;
              }
        }
      break;
    }
    case Dir::Filter: {
      out->assign(static_cast<size_t>(g.K * g.Cg * fsz), 0.0);
      for (int64_t k = 0; k < g.K; ++k) {
        const int64_t c0 = (k / g.Kg) * g.Cg;
        for (int64_t ci = 0; ci < g.Cg; ++ci)
          for (int64_t f0 = 0; f0 < F0; ++f0)
            for (int64_t f1 = 0; f1 < F1; ++f1)
              for (int64_t f2 = 0; f2 < F2; ++f2) {
                double s = 0.0;
                for (int64_t n = 0; n < g.N; ++n) {
                  const double* yb = a.data() + ((n * g.K + k) * osz);
                  const double* xb = b.data() + ((n * g.C + c0 + ci) * isz);
                  for (int64_t o0 = 0; o0 < O0; ++o0) {
                    const int64_t p0 = in_pos(0, o0, f0);
                    if (p0 < 0 || p0 >= I0) continue;
                    for (int64_t o1 = 0; o1 < O1; ++o1) {
                      const int64_t p1 = in_pos(1, o1, f1);
                      if (p1 < 0 || p1 >= I1) continue;
                      for (int64_t o2 = 0; o2 < O2; ++o2) {
                        const int64_t p2 = in_pos(2, o2, f2);
                        if (p2 < 0 || p2 >= I2) continue;
                        s = add(s, yb[(o0 * O1 + o1) * O2 + o2], xb[(p0 * I1 + p1) * I2 + p2]);
                      }
                    }
                  }
                }
                (*out)[static_cast<size_t>((k * g.Cg + ci) * fsz + (f0 * F1 + f1) * F2 + f2)] = s;
              }
      }
      break;
    }
  }
}

enum class Mix { Plain, Special, SpecialSparse, Ints, SpecialInput };   // SpecialInput: specials in the first operand only (weights stay finite)

std::vector<double> fill(size_t n, Mix mix, std::mt19937_64& rng) {
  using L = std::numeric_limits<double>;
  const double sp[] = {0.0, -0.0, 1.0, -1.0, L::denorm_min(), -L::denorm_min(), L::min(), L::max(), -L::max(),
                       L::infinity(), -L::infinity(), L::quiet_NaN(), 3.0, 0.1, 1e-200, 1e200, 9007199254740993.0};
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  std::vector<double> v(n);
  for (auto& x : v) {
    switch (mix) {
      case Mix::Plain: x = u(rng); break;
      case Mix::Special: x = rng() % 2 ? sp[rng() % (sizeof sp / sizeof sp[0])] : u(rng); break;
      case Mix::SpecialSparse: x = rng() % 40 == 0 ? sp[rng() % (sizeof sp / sizeof sp[0])] : u(rng); break;
      case Mix::Ints: x = static_cast<double>(static_cast<int>(rng() % 255) - 127); break;
      case Mix::SpecialInput: x = rng() % 8 == 0 ? sp[rng() % (sizeof sp / sizeof sp[0])] : u(rng); break;
    }
  }
  return v;
}

// Compares bits; two NaNs count as equal (which NaN payload wins when two meet is not defined by the card).
size_t differing(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return a.size() + 1;
  size_t bad = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], 8) == 0) continue;
    if (std::isnan(a[i]) && std::isnan(b[i])) continue;
    ++bad;
  }
  return bad;
}

size_t g_ran = 0;   // geometries that were valid and ran

struct Case {
  int nsp;           // spatial dimensions
  int64_t N, Cg, Kg, G;
  int64_t in[3], flt[3], pad[3], str[3], dil[3];
  bool flip;
};

bool make_geom(const Case& c, Geom* g) {
  *g = Geom();
  g->N = c.N, g->Cg = c.Cg, g->Kg = c.Kg, g->G = c.G, g->C = c.Cg * c.G, g->K = c.Kg * c.G, g->flip = c.flip;
  for (int i = 0; i < 3; ++i) {
    const bool real = i >= 3 - c.nsp;
    g->in[i] = real ? c.in[i] : 1, g->flt[i] = real ? c.flt[i] : 1, g->pad[i] = real ? c.pad[i] : 0;
    g->str[i] = real ? c.str[i] : 1, g->dil[i] = real ? c.dil[i] : 1;
    const int64_t o = 1 + (g->in[i] + 2 * g->pad[i] - ((g->flt[i] - 1) * g->dil[i] + 1)) / g->str[i];
    if (g->in[i] + 2 * g->pad[i] < (g->flt[i] - 1) * g->dil[i] + 1 || o < 1) return false;
    g->out[i] = o;
  }
  return true;
}

size_t sizes(const Geom& g, Dir d, size_t* na, size_t* nb) {
  const size_t isz = g.in[0] * g.in[1] * g.in[2], osz = g.out[0] * g.out[1] * g.out[2], fsz = g.flt[0] * g.flt[1] * g.flt[2];
  const size_t x = g.N * g.C * isz, y = g.N * g.K * osz, w = g.K * g.Cg * fsz;
  switch (d) {
    case Dir::Forward: *na = x, *nb = w; return y;
    case Dir::Data: *na = y, *nb = w; return x;
    default: *na = y, *nb = x; return w;
  }
}

size_t run_case(const Case& c, Mix mix, std::mt19937_64& rng) {
  Geom g;
  if (!make_geom(c, &g)) return 0;
  ++g_ran;
  size_t bad = 0;
  for (const Dir d : {Dir::Forward, Dir::Data, Dir::Filter}) {
    size_t na, nb;
    sizes(g, d, &na, &nb);
    const auto a = fill(na, mix, rng), b = fill(nb, mix == Mix::SpecialInput ? Mix::Plain : mix, rng);
    std::vector<double> ref;
    ref_convolve(g, d, a, b, &ref);
    for (const Isa isa : {Isa::Generic, Isa::Avx2}) {
      if (isa == Isa::Avx2 && !vgpu::hostconv::avx2_available()) continue;
      for (const unsigned threads : {1u, 3u})
        for (const Algo algo : {Algo::Auto, Algo::Rows, Algo::Gemm}) {
          // The result is every element's own; poison the output to prove each one is written.
          std::vector<double> got(ref.size(), 123.456);
          vgpu::hostconv::convolve_exact(g, d, a.data(), b.data(), got.data(), isa, threads, algo);
          bad += differing(ref, got);
        }
    }
  }
  return bad;
}

}  // namespace

VTEST(host_conv_matches_the_loops) {
  std::mt19937_64 rng(2468);
  size_t bad = 0;
  int cases = 0;
  const Mix mixes[] = {Mix::Plain, Mix::Special, Mix::SpecialSparse, Mix::Ints, Mix::SpecialInput};
  // Hand-picked: the common shapes (3x3 same, 1x1, strided, dilated, grouped, depthwise, 1-D and 3-D).
  const Case fixed[] = {
      {2, 2, 3, 4, 1, {0, 9, 11}, {0, 3, 3}, {0, 1, 1}, {0, 1, 1}, {0, 1, 1}, false},
      {2, 1, 5, 9, 1, {0, 8, 8}, {0, 3, 3}, {0, 1, 1}, {0, 1, 1}, {0, 1, 1}, true},
      {2, 2, 4, 4, 2, {0, 10, 10}, {0, 3, 3}, {0, 2, 2}, {0, 2, 2}, {0, 1, 1}, false},
      {2, 1, 1, 1, 6, {0, 9, 9}, {0, 3, 3}, {0, 1, 1}, {0, 1, 1}, {0, 1, 1}, false},   // depthwise
      {2, 2, 8, 8, 1, {0, 5, 5}, {0, 1, 1}, {0, 0, 0}, {0, 1, 1}, {0, 1, 1}, false},   // 1x1
      // depthwise and few-chain filters: the filter row as vector lanes (3, 4, 5, 7 and 8 taps, padding wider than the
      // input, stride, dilation across rows, flipped)
      {2, 2, 1, 1, 6, {0, 9, 12}, {0, 3, 5}, {0, 1, 2}, {0, 1, 1}, {0, 1, 1}, false},
      {2, 2, 1, 1, 3, {0, 10, 12}, {0, 3, 7}, {0, 1, 3}, {0, 2, 2}, {0, 2, 1}, true},
      {2, 2, 2, 2, 4, {0, 6, 4}, {0, 2, 3}, {0, 2, 3}, {0, 1, 1}, {0, 1, 1}, false},
      {2, 1, 1, 3, 3, {0, 9, 15}, {0, 3, 8}, {0, 1, 4}, {0, 1, 3}, {0, 1, 1}, true},
      {2, 3, 1, 1, 5, {0, 14, 20}, {0, 4, 4}, {0, 2, 1}, {0, 1, 1}, {0, 3, 1}, false},
      {3, 2, 1, 1, 3, {4, 7, 9}, {2, 3, 3}, {1, 1, 1}, {1, 1, 2}, {1, 1, 1}, false},
      {2, 1, 3, 2, 1, {0, 12, 13}, {0, 3, 2}, {0, 2, 1}, {0, 1, 3}, {0, 2, 3}, false},
      {2, 1, 2, 2, 1, {0, 7, 20}, {0, 5, 7}, {0, 4, 6}, {0, 2, 3}, {0, 1, 1}, true},
      {1, 2, 3, 5, 1, {0, 0, 33}, {0, 0, 5}, {0, 0, 2}, {0, 0, 2}, {0, 0, 1}, false},
      {3, 1, 2, 3, 1, {5, 6, 7}, {3, 3, 3}, {1, 1, 1}, {1, 1, 2}, {1, 2, 1}, false},
      {3, 2, 3, 2, 2, {4, 5, 9}, {2, 3, 3}, {0, 1, 2}, {1, 2, 1}, {1, 1, 2}, true},
      // big enough for the blocked product kernel (partial and full tiles, several blocks, more than one chunk of taps)
      {2, 2, 8, 8, 1, {0, 12, 12}, {0, 3, 3}, {0, 1, 1}, {0, 1, 1}, {0, 1, 1}, false},
      {2, 2, 12, 20, 2, {0, 15, 14}, {0, 3, 3}, {0, 2, 1}, {0, 2, 1}, {0, 2, 1}, true},
      {2, 1, 40, 24, 1, {0, 10, 10}, {0, 3, 3}, {0, 1, 1}, {0, 1, 1}, {0, 1, 1}, false},   // 360 taps, 24 outputs
  };
  for (const Case& c : fixed)
    for (const Mix mix : mixes) {
      bad += run_case(c, mix, rng);
      ++cases;
    }
  // Random geometries.
  for (int t = 0; t < 120; ++t) {
    Case c;
    c.nsp = 1 + static_cast<int>(rng() % 3);
    c.N = 1 + static_cast<int64_t>(rng() % 3);
    c.G = 1 + static_cast<int64_t>(rng() % 3);
    c.Cg = 1 + static_cast<int64_t>(rng() % 10);
    c.Kg = 1 + static_cast<int64_t>(rng() % 9);
    for (int i = 0; i < 3; ++i) {
      c.flt[i] = 1 + static_cast<int64_t>(rng() % 4);
      c.in[i] = c.flt[i] + static_cast<int64_t>(rng() % (c.nsp == 3 ? 5 : 14));
      c.pad[i] = static_cast<int64_t>(rng() % 3);
      c.str[i] = 1 + static_cast<int64_t>(rng() % 3);
      c.dil[i] = 1 + static_cast<int64_t>(rng() % 3);
    }
    c.flip = rng() % 2;
    bad += run_case(c, mixes[t % 5], rng);
    ++cases;
  }
  std::printf("  %zu of %d geometries ran (each x 3 directions x 2 instruction sets x 2 thread counts x 3 algorithms): %zu elements differ\n",
              g_ran, cases, bad);
  VCHECK(g_ran * 4 > static_cast<size_t>(cases) * 3);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST_MAIN
