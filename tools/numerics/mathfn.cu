// The math functions a GPU program calls, single and double precision, evaluated over three classes of input and printed
// so that two runs -- the simulator's two engines, two profiles, the simulator and a card -- compare line by line
// (docs/numerics/math-functions.md). One source for CUDA (nvcc) and HIP (hipcc).
//
//   worst      the inputs published as giving the largest known error of the CUDA and ROCm math libraries, and the other
//              published single inputs (tools/numerics/published.tsv; read at run time with --published <file>)
//   edges      zeros, subnormals, the overflow and underflow thresholds, huge trigonometric arguments, integers and
//              half-integers, poles, infinities, quiet and signaling NaNs (for a two-argument function: all pairs of a
//              shorter list)
//   rand_bits  --n uniformly random bit patterns (every exponent, NaNs included)
//   rand_mid   --n values of random sign and mantissa with the exponent uniform in [-12, 7]
//
// One line a function, precision and class:
//
//   <fn> <f32|f64> <class> n=<count> hash=<FNV-1a 64 of every result's bits, and of frexp's exponents> | <first inputs>=<results>
//
// The worst class prints every input and result (hexadecimal bit patterns), the others the first four. --dump <dir> also
// writes every result (<fn>.<prec>.<class>.r, raw little-endian words; frexp's exponents in .aux) and the inputs once
// (in.<prec>.<class>.x, .y; the worst class's per function), which is what tools/numerics/compare.py counts ulp over.
//
//   mathfn [--n 131072] [--published published.tsv] [--dump <dir>] [--only <fn>] [--prec f32|f64]
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "gpu_api.h"

// X(name, arity, single-precision expression, double-precision expression); a, b are the arguments, e the integer result.
#define FUNCTIONS(X) \
  X(sin, 1, sinf(a), sin(a)) X(cos, 1, cosf(a), cos(a)) X(tan, 1, tanf(a), tan(a)) \
  X(asin, 1, asinf(a), asin(a)) X(acos, 1, acosf(a), acos(a)) X(atan, 1, atanf(a), atan(a)) \
  X(sinh, 1, sinhf(a), sinh(a)) X(cosh, 1, coshf(a), cosh(a)) X(tanh, 1, tanhf(a), tanh(a)) \
  X(asinh, 1, asinhf(a), asinh(a)) X(acosh, 1, acoshf(a), acosh(a)) X(atanh, 1, atanhf(a), atanh(a)) \
  X(exp, 1, expf(a), exp(a)) X(exp2, 1, exp2f(a), exp2(a)) X(exp10, 1, exp10f(a), exp10(a)) X(expm1, 1, expm1f(a), expm1(a)) \
  X(log, 1, logf(a), log(a)) X(log2, 1, log2f(a), log2(a)) X(log10, 1, log10f(a), log10(a)) X(log1p, 1, log1pf(a), log1p(a)) \
  X(sqrt, 1, sqrtf(a), sqrt(a)) X(rsqrt, 1, rsqrtf(a), rsqrt(a)) X(cbrt, 1, cbrtf(a), cbrt(a)) \
  X(erf, 1, erff(a), erf(a)) X(erfc, 1, erfcf(a), erfc(a)) X(lgamma, 1, lgammaf(a), lgamma(a)) X(tgamma, 1, tgammaf(a), tgamma(a)) \
  X(sinpi, 1, sinpif(a), sinpi(a)) X(cospi, 1, cospif(a), cospi(a)) \
  X(ceil, 1, ceilf(a), ceil(a)) X(floor, 1, floorf(a), floor(a)) X(trunc, 1, truncf(a), trunc(a)) \
  X(rint, 1, rintf(a), rint(a)) X(round, 1, roundf(a), round(a)) \
  X(rcp, 1, 1.0f / a, 1.0 / a) X(rsqrt_div, 1, 1.0f / sqrtf(a), 1.0 / sqrt(a)) \
  X(frexp, 1, frexpf(a, &e), frexp(a, &e)) \
  X(pow, 2, powf(a, b), pow(a, b)) X(atan2, 2, atan2f(a, b), atan2(a, b)) X(hypot, 2, hypotf(a, b), hypot(a, b)) \
  X(fmod, 2, fmodf(a, b), fmod(a, b)) X(remainder, 2, remainderf(a, b), remainder(a, b)) \
  X(div, 2, a / b, a / b) X(ldexp, 2, ldexpf(a, scale_of(b)), ldexp(a, scale_of(b)))

enum Fn {
#define X(name, arity, f, d) F_##name,
  FUNCTIONS(X)
#undef X
  F_COUNT
};
struct FnInfo { const char* name; int arity; };
static const FnInfo kFns[] = {
#define X(name, arity, f, d) {#name, arity},
    FUNCTIONS(X)
#undef X
};

// ldexp's exponent, from the second argument's bits: -2200 .. 2200, past both ends of either format.
__device__ __host__ inline int scale_of(float b) { uint32_t u; memcpy(&u, &b, 4); return static_cast<int>(u % 4401u) - 2200; }
__device__ __host__ inline int scale_of(double b) {
  uint64_t u; memcpy(&u, &b, 8);
  return static_cast<int>(static_cast<uint32_t>(u ^ (u >> 32)) % 4401u) - 2200;
}

__global__ void k32(const float* x, const float* y, float* r, int* aux, int n, int fn) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float a = x[i], b = y[i];
  float v = 0;
  int e = 0;
  switch (fn) {
#define X(name, arity, f, d) case F_##name: v = f; break;
    FUNCTIONS(X)
#undef X
  }
  r[i] = v;
  aux[i] = e;
}

__global__ void k64(const double* x, const double* y, double* r, int* aux, int n, int fn) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double a = x[i], b = y[i];
  double v = 0;
  int e = 0;
  switch (fn) {
#define X(name, arity, f, d) case F_##name: v = d; break;
    FUNCTIONS(X)
#undef X
  }
  r[i] = v;
  aux[i] = e;
}

static uint64_t splitmix(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Values both precisions take (a double is rounded to float for f32; one that overflows there is an infinity twice, harmless).
static const double kEdgeValues[] = {
    0.0, 1.0, 0.5, 1.5, 2.5, 3.5, 0.25, 0.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 100.0, 171.0, 172.0, 34.0, 35.0, 36.0,
    0.1, 0.3, 1.0 / 3.0, 2.0 / 3.0, 0.9, 1.1, 0.999, 1.001, 1.083, 4.75, 9.0, 9.2, 0.55, 3.9, 10.05, 10.06, 26.5, 27.3,
    3.14159265358979323846, 1.57079632679489661923, 0.78539816339744830962, 6.28318530717958647692, 4.71238898038468985769,
    2.71828182845904523536, 0.36787944117144232160, 0.69314718055994530942, 2.30258509299404568402,
    1e-5, 1e-10, 1e-20, 1e-30, 1e-37, 1e-38, 1e-39, 1e-44, 1e-45, 1e-300, 1e-307, 1e-308, 1e-310, 1e-323,
    1e5, 1e10, 1e15, 1e22, 1e30, 1e38, 1e39, 1e300, 1e308, 5992555.0, 105615.0, 252.898206, 16777216.0, 16777217.0, 16777215.0,
    8388607.5, 8388608.0, 8388609.0, 4503599627370495.5, 4503599627370496.0, 4503599627370497.0, 9007199254740992.0,
    9007199254740993.0, 2147483648.0, 4294967296.0, 9223372036854775808.0, 18446744073709551616.0,
    88.72283, 88.72284, 88.7229, 87.33655, 103.27893, 103.97208, 104.0, 127.0, 127.99999, 128.0, 126.0, 149.0, 150.0,
    38.53184, 37.9, 45.0, 709.782712893384, 709.79, 745.13, 745.14, 708.4, 1023.0, 1023.9999999999999, 1024.0, 1074.0, 1075.0,
    308.25, 323.3, 171.62437695630272, 35.04, 0x1p-126, 0x1p-149, 0x1p-1022, 0x1p-1074, 0x1p127, 0x1p1023, 0x1p100, 0x1p120,
    0x1.fffffep127, 0x1.fffffffffffffp1023, 0x1.fffffep-1, 0x1.000002p0, 0x1.fffffffffffffp-1, 0x1.0000000000001p0};
// Bit patterns of one precision: the subnormal range's ends, the smallest normal, the largest, infinities, NaNs.
static const uint32_t kEdgeBits32[] = {0x00000001u, 0x007FFFFFu, 0x00800000u, 0x00800001u, 0x7F7FFFFFu, 0x7F800000u, 0xFF800000u,
                                       0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0x7FBFFFFFu, 0xFFA00000u, 0x7FC12345u, 0xFFFFFFFFu,
                                       0x7FFFFFFFu, 0x80000000u, 0x80000001u, 0x807FFFFFu, 0x80800000u, 0xFF7FFFFFu};
static const uint64_t kEdgeBits64[] = {0x0000000000000001ull, 0x000FFFFFFFFFFFFFull, 0x0010000000000000ull, 0x0010000000000001ull,
                                       0x7FEFFFFFFFFFFFFFull, 0x7FF0000000000000ull, 0xFFF0000000000000ull, 0x7FF8000000000000ull,
                                       0xFFF8000000000000ull, 0x7FF0000000000001ull, 0x7FF7FFFFFFFFFFFFull, 0xFFF4000000000000ull,
                                       0x7FF8000012345678ull, 0xFFFFFFFFFFFFFFFFull, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull,
                                       0x8000000000000001ull, 0x800FFFFFFFFFFFFFull, 0x8010000000000000ull, 0xFFEFFFFFFFFFFFFFull};
// The shorter list a two-argument function takes all pairs of.
static const double kPairValues[] = {0.0, 1.0, 0.5, 2.0, 3.0, 1.5, 2.5, 0.75, 0.1, 0.3, 1.0 / 3.0, 7.0, 100.0, 1e-5, 1e10, 1e-30, 1e30,
                                     3.14159265358979323846, 16777215.0, 16777216.0, 9007199254740991.0, 0x1.fffffep-1,
                                     0x1.000002p0, 0x1p-126, 0x1p-1022, 0x1p127, 0x1p1023};
static const int kPairBitIndices[] = {0, 1, 2, 4, 5, 7, 9, 12};   // of kEdgeBits: subnormals, smallest normal, largest, inf, NaNs

template <typename T, typename U>
struct Precision {
  static T from_bits(U u) { T t; memcpy(&t, &u, sizeof t); return t; }
  static U to_bits(T t) { U u; memcpy(&u, &t, sizeof u); return u; }
};

template <typename T, typename U>
static std::vector<T> edge_values(const U* bits, size_t nbits) {
  std::vector<T> v;
  for (double d : kEdgeValues) { v.push_back(static_cast<T>(d)); v.push_back(static_cast<T>(-d)); }
  for (size_t i = 0; i < nbits; ++i) v.push_back(Precision<T, U>::from_bits(bits[i]));
  return v;
}
template <typename T, typename U>
static std::vector<T> pair_values(const U* bits) {
  std::vector<T> v;
  for (double d : kPairValues) { v.push_back(static_cast<T>(d)); v.push_back(static_cast<T>(-d)); }
  for (int i : kPairBitIndices) v.push_back(Precision<T, U>::from_bits(bits[i]));
  return v;
}

template <typename T, typename U>
static T random_value(uint64_t counter, uint64_t stream, bool mid) {
  const uint64_t h = splitmix(counter * 2 + stream + (mid ? 0x5151515151515151ull : 0));
  const int mant = sizeof(T) == 4 ? 23 : 52, bias = sizeof(T) == 4 ? 127 : 1023;
  U u = static_cast<U>(sizeof(T) == 4 ? (h >> 32) : h);
  if (mid) {
    const U sign = u >> (mant + (sizeof(T) == 4 ? 8 : 11)), frac = u & ((static_cast<U>(1) << mant) - 1);
    const int e = static_cast<int>(splitmix(h) % 20u) - 12;
    u = (sign << (mant + (sizeof(T) == 4 ? 8 : 11))) | (static_cast<U>(e + bias) << mant) | frac;
  }
  return Precision<T, U>::from_bits(u);
}

struct Published { std::string fn, prec; double x, y; bool has_y; };
static std::vector<Published> load_published(const char* path) {
  std::vector<Published> rows;
  FILE* f = fopen(path, "r");
  if (!f) { fprintf(stderr, "mathfn: cannot read %s\n", path); exit(2); }
  char line[1024];
  while (fgets(line, sizeof line, f)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    char fn[64], prec[16], xs[128], ys[128];
    if (sscanf(line, "%63s %15s %127s %127s", fn, prec, xs, ys) != 4) continue;
    Published p{fn, prec, strtod(xs, nullptr), 0, false};
    if (strcmp(ys, "-") != 0) { p.y = strtod(ys, nullptr); p.has_y = true; }
    rows.push_back(p);
  }
  fclose(f);
  return rows;
}

struct Options { int n = 131072; const char* published = nullptr; const char* dump = nullptr; const char* only = nullptr; const char* prec = nullptr; };

static void write_file(const std::string& path, const void* data, size_t bytes) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "mathfn: cannot write %s\n", path.c_str()); exit(2); }
  fwrite(data, 1, bytes, f);
  fclose(f);
}

template <typename T, typename U, typename Kernel>
static void run_precision(const char* prec, const Options& opt, const std::vector<Published>& published, const U* edge_bits,
                          size_t n_edge_bits, Kernel launch) {
  const char* fmt = sizeof(T) == 4 ? "%08llx" : "%016llx";
  struct Class { const char* name; std::vector<T> x1, x2, y2; };   // x1: one-argument inputs; x2, y2: two-argument inputs
  std::vector<Class> classes(3);
  classes[0].name = "edges";
  classes[0].x1 = edge_values<T, U>(edge_bits, n_edge_bits);
  { const std::vector<T> p = pair_values<T, U>(edge_bits);
    for (T a : p) for (T b : p) { classes[0].x2.push_back(a); classes[0].y2.push_back(b); } }
  for (int c = 1; c < 3; ++c) {
    classes[c].name = c == 1 ? "rand_bits" : "rand_mid";
    for (int i = 0; i < opt.n; ++i) {
      classes[c].x1.push_back(random_value<T, U>(i, 0, c == 2));
      classes[c].y2.push_back(random_value<T, U>(i, 1, c == 2));
    }
    classes[c].x2 = classes[c].x1;
  }
  size_t cap = 16;
  for (const Class& c : classes) cap = std::max(cap, std::max(c.x1.size(), c.x2.size()));
  cap = std::max(cap, published.size());
  T *dx = nullptr, *dy = nullptr, *dr = nullptr; int* da = nullptr;
  if (gpuMalloc(&dx, cap * sizeof(T)) != gpuOk || gpuMalloc(&dy, cap * sizeof(T)) != gpuOk || gpuMalloc(&dr, cap * sizeof(T)) != gpuOk ||
      gpuMalloc(&da, cap * sizeof(int)) != gpuOk) { fprintf(stderr, "mathfn: device allocation failed\n"); exit(2); }
  std::vector<T> r(cap); std::vector<int> aux(cap);
  if (opt.dump)
    for (const Class& c : classes) {
      const std::string base = std::string(opt.dump) + "/in." + prec + "." + c.name;
      write_file(base + ".x1", c.x1.data(), c.x1.size() * sizeof(T));
      write_file(base + ".x2", c.x2.data(), c.x2.size() * sizeof(T));
      write_file(base + ".y2", c.y2.data(), c.y2.size() * sizeof(T));
    }
  auto evaluate = [&](int fn, const char* cls, const std::vector<T>& x, const std::vector<T>& y, bool print_all) {
    const int n = static_cast<int>(x.size());
    if (n == 0) return;
    const bool two = kFns[fn].arity == 2;
    gpuMemcpy(dx, x.data(), n * sizeof(T), gpuToDevice);
    gpuMemcpy(dy, two ? y.data() : x.data(), n * sizeof(T), gpuToDevice);
    launch(dx, dy, dr, da, n, fn);
    if (gpuSync() != gpuOk) { printf("%s %s %s n=%d FAILED (the kernel did not run)\n", kFns[fn].name, prec, cls, n); return; }
    gpuMemcpy(r.data(), dr, n * sizeof(T), gpuToHost);
    gpuMemcpy(aux.data(), da, n * sizeof(int), gpuToHost);
    uint64_t h = 0xcbf29ce484222325ull;
    auto eat = [&h](uint64_t v, int bytes) { for (int k = 0; k < bytes; ++k) { h ^= (v >> (8 * k)) & 0xff; h *= 0x100000001b3ull; } };
    for (int i = 0; i < n; ++i) { eat(Precision<T, U>::to_bits(r[i]), sizeof(T)); if (fn == F_frexp) eat(static_cast<uint32_t>(aux[i]), 4); }
    printf("%s %s %s n=%d hash=%016llx |", kFns[fn].name, prec, cls, n, static_cast<unsigned long long>(h));
    for (int i = 0; i < n && (print_all || i < 4); ++i) {
      printf(" "); printf(fmt, static_cast<unsigned long long>(Precision<T, U>::to_bits(x[i])));
      if (two) { printf(","); printf(fmt, static_cast<unsigned long long>(Precision<T, U>::to_bits(y[i]))); }
      printf("="); printf(fmt, static_cast<unsigned long long>(Precision<T, U>::to_bits(r[i])));
      if (fn == F_frexp) printf("/%d", aux[i]);
    }
    printf("\n");
    fflush(stdout);
    if (opt.dump) {
      const std::string base = std::string(opt.dump) + "/" + kFns[fn].name + "." + prec + "." + cls;
      write_file(base + ".r", r.data(), n * sizeof(T));
      if (fn == F_frexp) write_file(base + ".aux", aux.data(), n * sizeof(int));
      if (print_all) {
        write_file(base + ".x", x.data(), n * sizeof(T));
        if (two) write_file(base + ".y", y.data(), n * sizeof(T));
      }
    }
  };
  for (int fn = 0; fn < F_COUNT; ++fn) {
    if (opt.only && strcmp(opt.only, kFns[fn].name) != 0) continue;
    std::vector<T> wx, wy;
    for (const Published& p : published) {
      if (p.fn != kFns[fn].name || p.prec != prec) continue;
      const T x = static_cast<T>(p.x), y = static_cast<T>(p.y);
      bool seen = false;
      for (size_t i = 0; i < wx.size(); ++i)
        seen = seen || (Precision<T, U>::to_bits(wx[i]) == Precision<T, U>::to_bits(x) && Precision<T, U>::to_bits(wy[i]) == Precision<T, U>::to_bits(y));
      if (!seen) { wx.push_back(x); wy.push_back(y); }
    }
    evaluate(fn, "worst", wx, wy, true);
    for (const Class& c : classes) {
      if (kFns[fn].arity == 2) evaluate(fn, c.name, c.x2, c.y2, false);
      else evaluate(fn, c.name, c.x1, c.x1, false);
    }
  }
  gpuFree(dx); gpuFree(dy); gpuFree(dr); gpuFree(da);
}

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--n" && i + 1 < argc) opt.n = atoi(argv[++i]);
    else if (a == "--published" && i + 1 < argc) opt.published = argv[++i];
    else if (a == "--dump" && i + 1 < argc) opt.dump = argv[++i];
    else if (a == "--only" && i + 1 < argc) opt.only = argv[++i];
    else if (a == "--prec" && i + 1 < argc) opt.prec = argv[++i];
    else { fprintf(stderr, "usage: mathfn [--n N] [--published file] [--dump dir] [--only fn] [--prec f32|f64]\n"); return 2; }
  }
  std::vector<Published> published;
  if (opt.published) published = load_published(opt.published);
  printf("# mathfn api=%s n=%d published=%zu\n", GPU_API, opt.n, published.size());
  if (!opt.prec || strcmp(opt.prec, "f32") == 0)
    run_precision<float, uint32_t>("f32", opt, published, kEdgeBits32, sizeof kEdgeBits32 / sizeof kEdgeBits32[0],
                                   [](float* x, float* y, float* r, int* aux, int n, int fn) { k32<<<(n + 255) / 256, 256>>>(x, y, r, aux, n, fn); });
  if (!opt.prec || strcmp(opt.prec, "f64") == 0)
    run_precision<double, uint64_t>("f64", opt, published, kEdgeBits64, sizeof kEdgeBits64 / sizeof kEdgeBits64[0],
                                    [](double* x, double* y, double* r, int* aux, int n, int fn) { k64<<<(n + 255) / 256, 256>>>(x, y, r, aux, n, fn); });
  return 0;
}
