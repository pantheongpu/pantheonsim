// cuRAND's host API as bindings drive it (cudarc's tests call each of these):
// uniform, normal and log-normal numbers in single and double precision, and
// 32-bit integers. The streams are the library's own, so what is checked is
// what holds for any correct generator: the range, the moments to within
// sampling error, a seed and offset reproducing their stream, and a
// log-normal draw being exp of the normal draw the same position gives.
//
// Setting the seed does not rewind: the stream carries on from wherever the
// generator was (an RTX 3060's cuRAND does the same). curandSetGeneratorOffset
// is what restarts it, so each comparison below sets both.
#include <cuda_runtime.h>
#include <curand.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what, double v) {
  std::printf("%-4s %s (%.4g)\n", ok ? "ok" : "FAIL", what, v);
  if (!ok) ++failures;
}

#define CK(x)                                                   \
  do {                                                          \
    const int r_ = (int)(x);                                    \
    if (r_ != 0) {                                              \
      std::printf("FAIL %s returned %d\n", #x, r_);             \
      ++failures;                                               \
      return;                                                   \
    }                                                           \
  } while (0)

static const size_t N = 1 << 16;

static curandStatus_t restart(curandGenerator_t g, unsigned long long seed) {
  curandStatus_t s = curandSetPseudoRandomGeneratorSeed(g, seed);
  return s ? s : curandSetGeneratorOffset(g, 0);
}

template <class T> static std::vector<T> down(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}
template <class T> static void moments(const std::vector<T>& v, double& mean, double& sd) {
  mean = 0;
  for (T x : v) mean += x;
  mean /= v.size();
  double var = 0;
  for (T x : v) var += (x - mean) * (x - mean);
  sd = std::sqrt(var / v.size());
}

static curandStatus_t uniform(curandGenerator_t g, float* d, size_t n) { return curandGenerateUniform(g, d, n); }
static curandStatus_t uniform(curandGenerator_t g, double* d, size_t n) { return curandGenerateUniformDouble(g, d, n); }
static curandStatus_t normal(curandGenerator_t g, float* d, size_t n, double m, double s) { return curandGenerateNormal(g, d, n, (float)m, (float)s); }
static curandStatus_t normal(curandGenerator_t g, double* d, size_t n, double m, double s) { return curandGenerateNormalDouble(g, d, n, m, s); }
static curandStatus_t lognormal(curandGenerator_t g, float* d, size_t n, double m, double s) { return curandGenerateLogNormal(g, d, n, (float)m, (float)s); }
static curandStatus_t lognormal(curandGenerator_t g, double* d, size_t n, double m, double s) { return curandGenerateLogNormalDouble(g, d, n, m, s); }

template <class T> static void real(curandGenerator_t g, const char* ty) {
  T* d = nullptr;
  cudaMalloc(&d, N * sizeof(T));
  char what[96];
  double m, s;

  CK(curandSetPseudoRandomGeneratorSeed(g, 1234));
  CK(uniform(g, d, N));
  auto u = down(d, N);
  bool range = true;
  for (T x : u) range = range && x > 0 && x <= 1;
  moments(u, m, s);
  std::snprintf(what, sizeof what, "%s uniform in (0, 1]", ty);
  check(range, what, 0);
  // Standard error of the mean: 0.289 / sqrt(65536) = 1.1e-3; allow 5 of them.
  std::snprintf(what, sizeof what, "%s uniform mean 0.5", ty);
  check(std::fabs(m - 0.5) < 6e-3, what, m);

  CK(curandSetPseudoRandomGeneratorSeed(g, 99));
  CK(normal(g, d, N, 2.0, 3.0));
  auto nn = down(d, N);
  moments(nn, m, s);
  std::snprintf(what, sizeof what, "%s normal mean 2", ty);
  check(std::fabs(m - 2.0) < 0.06, what, m);
  std::snprintf(what, sizeof what, "%s normal stddev 3", ty);
  check(std::fabs(s - 3.0) < 0.05, what, s);

  // A log-normal draw is exp of the normal draw from the same seed.
  CK(restart(g, 7));
  CK(normal(g, d, N, 0.5, 0.25));
  auto base = down(d, N);
  CK(restart(g, 7));
  CK(lognormal(g, d, N, 0.5, 0.25));
  auto ln = down(d, N);
  double worst = 0;
  bool positive = true;
  for (size_t i = 0; i < N; ++i) {
    positive = positive && ln[i] > 0;
    worst = std::fmax(worst, std::fabs((double)ln[i] - std::exp((double)base[i])) / std::exp((double)base[i]));
  }
  std::snprintf(what, sizeof what, "%s log-normal is exp(normal), same seed", ty);
  check(positive && worst < (sizeof(T) == 4 ? 1e-5 : 1e-12), what, worst);
  moments(ln, m, s);
  // E[X] = exp(mu + sigma^2 / 2) = 1.6997 for mu 0.5, sigma 0.25.
  std::snprintf(what, sizeof what, "%s log-normal mean", ty);
  check(std::fabs(m - std::exp(0.5 + 0.25 * 0.25 / 2)) < 0.01, what, m);

  // The same seed and offset reproduce the same stream; the seed alone
  // carries on from where the stream was; an offset of k skips k numbers.
  CK(restart(g, 4242));
  CK(uniform(g, d, 1024));
  auto r1 = down(d, 1024);
  CK(restart(g, 4242));
  CK(uniform(g, d, 1024));
  auto r2 = down(d, 1024);
  std::snprintf(what, sizeof what, "%s seed and offset reproduce the stream", ty);
  check(r1 == r2, what, 0);
  CK(curandSetPseudoRandomGeneratorSeed(g, 4242));
  CK(uniform(g, d, 1024));
  auto r3 = down(d, 1024);
  std::snprintf(what, sizeof what, "%s seed alone carries on", ty);
  check(r3 != r1, what, 0);
  CK(curandSetGeneratorOffset(g, 512));
  CK(uniform(g, d, 512));
  auto r4 = down(d, 512);
  std::snprintf(what, sizeof what, "%s offset 512 skips 512 numbers", ty);
  check(std::vector<T>(r1.begin() + 512, r1.end()) == r4, what, 0);
  cudaFree(d);
}

static void integers(curandGenerator_t g) {
  unsigned* d = nullptr;
  cudaMalloc(&d, N * sizeof(unsigned));
  CK(curandSetPseudoRandomGeneratorSeed(g, 5));
  CK(curandGenerate(g, d, N));
  auto v = down(d, N);
  // Each bit set about half the time: the top and bottom bits, 65536 draws.
  size_t top = 0, bottom = 0;
  for (unsigned x : v) { top += x >> 31; bottom += x & 1; }
  check(std::fabs(top / (double)N - 0.5) < 0.01, "u32 top bit set half the time", top / (double)N);
  check(std::fabs(bottom / (double)N - 0.5) < 0.01, "u32 low bit set half the time", bottom / (double)N);
  cudaFree(d);
}

int main() {
  curandGenerator_t g;
  if (curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT)) {
    std::printf("FAIL: curandCreateGenerator\n");
    return 1;
  }
  real<float>(g, "f32");
  real<double>(g, "f64");
  integers(g);
  curandDestroyGenerator(g);
  std::printf(failures ? "FAIL: %d cuRAND checks\n" : "PASS: every cuRAND check\n", failures);
  return failures ? 1 : 0;
}
