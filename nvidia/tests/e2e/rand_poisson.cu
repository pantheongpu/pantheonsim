// cuRAND's Poisson functions (pantheonsim #629: RubixML/numpower calls curandGeneratePoisson, which the simulator's
// libcurand did not export): curandGeneratePoisson on device and host generators of every type,
// curandCreatePoissonDistribution / curandDestroyDistribution and curand_discrete reading that distribution in a
// kernel, curandGeneratePoissonMethod, curandGenerateSeeds and curandGetProperty.
//
// What is checked, all of it measured on an RTX 3060 (cuRAND 13.0) and passing there too:
//  * the statuses at the edges (lambda 0, negative, above 400,000; a length that is not a multiple of the
//    dimensions; null handles; the methods the card refuses);
//  * mean and variance within sampling error, for small lambdas, ones where the card's method changes (10, 2000,
//    8000) and large ones, and that a seed and offset reproduce their numbers;
//  * a 32-bit Sobol' generator's Poisson integers are the card's, bit for bit (its points are, and the histogram,
//    the inversion and the rounded normal on top of them are the card's), and so are a 64-bit one's;
//  * curand_discrete in a kernel, with XORWOW states from curand_init, draws the card's integers from the
//    distribution curandCreatePoissonDistribution made (the histogram below lambda 2000, the normal above).
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what, double v) {
  std::printf("%-4s %s (%.6g)\n", ok ? "ok" : "FAIL", what, v);
  if (!ok) ++failures;
}

static uint64_t fnv(const std::vector<unsigned>& v) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned x : v)
    for (int b = 0; b < 4; ++b) h = (h ^ ((x >> (8 * b)) & 0xffu)) * 1099511628211ull;
  return h;
}

static const size_t N = 1 << 16;

static std::vector<unsigned> draw(curandGenerator_t g, bool host, unsigned* d, size_t n, double lambda, curandStatus_t* st) {
  std::vector<unsigned> h(n, 0xdeadbeefu);
  *st = curandGeneratePoisson(g, host ? h.data() : d, n, lambda);
  if (!host && *st == CURAND_STATUS_SUCCESS) cudaMemcpy(h.data(), d, n * sizeof(unsigned), cudaMemcpyDeviceToHost);
  return h;
}

static void statuses(unsigned* d) {
  curandGenerator_t g;
  curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
  check(curandGeneratePoisson(g, d, 16, 0.0) == CURAND_STATUS_OUT_OF_RANGE, "lambda 0 is OUT_OF_RANGE", 0);
  check(curandGeneratePoisson(g, d, 16, -1.0) == CURAND_STATUS_OUT_OF_RANGE, "a negative lambda is OUT_OF_RANGE", -1);
  check(curandGeneratePoisson(g, d, 16, 400000.5) == CURAND_STATUS_OUT_OF_RANGE, "lambda above 400,000 is OUT_OF_RANGE", 400000.5);
  check(curandGeneratePoisson(g, d, 16, 400000.0) == CURAND_STATUS_SUCCESS, "lambda 400,000 is taken", 400000);
  check(curandGeneratePoisson(g, d, 0, 4.0) == CURAND_STATUS_SUCCESS, "no values is SUCCESS", 0);
  check(curandGeneratePoisson(g, d, 1001, 4.0) == CURAND_STATUS_SUCCESS, "an odd count is taken", 1001);
  check(curandGeneratePoisson(nullptr, d, 16, 4.0) == CURAND_STATUS_NOT_INITIALIZED, "no generator", 0);
  check(curandGeneratePoissonMethod(g, d, 16, 4.0, CURAND_3RD) == CURAND_STATUS_SUCCESS, "method CURAND_3RD is taken", 11);
  check(curandGeneratePoissonMethod(g, d, 16, 4.0, CURAND_DEVICE_API) == CURAND_STATUS_SUCCESS, "method CURAND_DEVICE_API is taken", 9);
  check(curandGeneratePoissonMethod(g, d, 16, 4.0, CURAND_M2) == CURAND_STATUS_INTERNAL_ERROR, "method CURAND_M2 is INTERNAL_ERROR", 5);
  check(curandGeneratePoissonMethod(g, d, 16, 4.0, CURAND_CHOOSE_BEST) == CURAND_STATUS_INTERNAL_ERROR, "method CURAND_CHOOSE_BEST is INTERNAL_ERROR", 0);
  check(curandGenerateSeeds(g) == CURAND_STATUS_SUCCESS, "curandGenerateSeeds", 0);
  check(curandGenerateSeeds(nullptr) == CURAND_STATUS_NOT_INITIALIZED, "curandGenerateSeeds without a generator", 0);
  curandDestroyGenerator(g);
  curandCreateGenerator(&g, CURAND_RNG_QUASI_SOBOL32);
  curandSetQuasiRandomGeneratorDimensions(g, 3);
  check(curandGeneratePoisson(g, d, 10, 4.0) == CURAND_STATUS_LENGTH_NOT_MULTIPLE, "10 values of 3 dimensions", 10);
  check(curandGeneratePoisson(g, d, 9, 4.0) == CURAND_STATUS_SUCCESS, "9 values of 3 dimensions", 9);
  curandDestroyGenerator(g);

  curandDiscreteDistribution_t dist = nullptr;
  check(curandCreatePoissonDistribution(4.0, nullptr) == CURAND_STATUS_NOT_INITIALIZED, "a distribution needs somewhere to go", 0);
  check(curandCreatePoissonDistribution(0.0, &dist) == CURAND_STATUS_OUT_OF_RANGE, "a distribution of lambda 0", 0);
  check(curandCreatePoissonDistribution(400001.0, &dist) == CURAND_STATUS_OUT_OF_RANGE, "a distribution of lambda 400,001", 0);
  check(curandDestroyDistribution(nullptr) == CURAND_STATUS_NOT_INITIALIZED, "destroying no distribution", 0);

  int v = -1;
  check(curandGetProperty(MAJOR_VERSION, &v) == CURAND_STATUS_SUCCESS && v == CURAND_VER_MAJOR, "curandGetProperty major", v);
  check(curandGetProperty(MINOR_VERSION, &v) == CURAND_STATUS_SUCCESS && v == CURAND_VER_MINOR, "curandGetProperty minor", v);
  check(curandGetProperty(PATCH_LEVEL, &v) == CURAND_STATUS_SUCCESS && v == CURAND_VER_PATCH, "curandGetProperty patch", v);
  check(curandGetProperty((libraryPropertyType)3, &v) == CURAND_STATUS_OUT_OF_RANGE, "curandGetProperty of no property", 3);
}

static void moments(unsigned* d) {
  const curandRngType_t types[] = {CURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MTGP32,
                                   CURAND_RNG_PSEUDO_MT19937, CURAND_RNG_PSEUDO_PHILOX4_32_10, CURAND_RNG_QUASI_SOBOL32,
                                   CURAND_RNG_QUASI_SCRAMBLED_SOBOL32, CURAND_RNG_QUASI_SOBOL64,
                                   CURAND_RNG_QUASI_SCRAMBLED_SOBOL64};
  const double lambdas[] = {0.5, 4, 10, 100, 1999, 2000, 8000, 9000, 400000};
  for (int host = 0; host < 2; ++host)
    for (curandRngType_t t : types) {
      curandGenerator_t g;
      if ((host ? curandCreateGeneratorHost(&g, t) : curandCreateGenerator(&g, t)) != CURAND_STATUS_SUCCESS) {
        check(false, "create a generator", t);
        continue;
      }
      bool all = true;
      for (double lam : lambdas) {
        if (t < CURAND_RNG_QUASI_DEFAULT) curandSetPseudoRandomGeneratorSeed(g, 4321);
        curandSetGeneratorOffset(g, 0);
        curandStatus_t st;
        std::vector<unsigned> h = draw(g, host, d, N, lam, &st);
        if (st != CURAND_STATUS_SUCCESS) {
          std::printf("FAIL type %d host %d lambda %g: status %d\n", t, host, lam, st);
          all = false;
          continue;
        }
        double m = 0, var = 0;
        for (unsigned x : h) m += x;
        m /= N;
        for (unsigned x : h) var += (x - m) * (x - m);
        var /= N;
        // the mean within 5 standard errors; the variance within 5 of its own (2 lambda^2 / N, plus the fourth
        // cumulant's lambda / N)
        const bool ok = std::fabs(m - lam) < 5 * std::sqrt(lam / N) + 1e-9 &&
                        std::fabs(var - lam) < 5 * std::sqrt((2 * lam * lam + lam) / N);
        if (!ok) {
          std::printf("FAIL type %d host %d lambda %g: mean %.6g variance %.6g\n", t, host, lam, m, var);
          all = false;
        }
        // the same seed and offset, the same integers (not asked of the Mersenne Twisters, which the card's
        // curandSetGeneratorOffset does not rewind)
        if (t == CURAND_RNG_PSEUDO_MTGP32 || t == CURAND_RNG_PSEUDO_MT19937) continue;
        curandSetGeneratorOffset(g, 0);
        if (t < CURAND_RNG_QUASI_DEFAULT) curandSetPseudoRandomGeneratorSeed(g, 4321);
        std::vector<unsigned> again = draw(g, host, d, N, lam, &st);
        if (again != h) {
          std::printf("FAIL type %d host %d lambda %g: not reproduced from the offset\n", t, host, lam);
          all = false;
        }
      }
      char what[96];
      std::snprintf(what, sizeof what, "type %d, %s generator: Poisson moments and reproduction", t, host ? "host" : "device");
      check(all, what, t);
      curandDestroyGenerator(g);
    }
}

// The integers an RTX 3060 gives (FNV-1a of 65,536 values from the start of the sequence).
struct Golden {
  curandRngType_t type;
  double lambda;
  uint64_t hash;
};

static void sobol_exact(unsigned* d) {
  const Golden golden[] = {
      {CURAND_RNG_QUASI_SOBOL32, 0.5, 0x7646ba4984a36143ull},  {CURAND_RNG_QUASI_SOBOL32, 4, 0x774ac9d2333b4a79ull},   {CURAND_RNG_QUASI_SOBOL32, 9.5, 0xe4da5ef3e145ad18ull},
      {CURAND_RNG_QUASI_SOBOL32, 10, 0xf5d7942811a4e023ull},   {CURAND_RNG_QUASI_SOBOL32, 100, 0xb6938775d4999419ull}, {CURAND_RNG_QUASI_SOBOL32, 1999, 0xbb7b65a6794b8835ull},
      {CURAND_RNG_QUASI_SOBOL32, 8000.5, 0x85c92d729f609ccfull}, {CURAND_RNG_QUASI_SOBOL32, 400000, 0x5902a570c1fb748ull},
      {CURAND_RNG_QUASI_SOBOL64, 4, 0x19b083b85f60c14aull},    {CURAND_RNG_QUASI_SOBOL64, 100, 0x292c5c9f8a0f6628ull}, {CURAND_RNG_QUASI_SOBOL64, 2000, 0x89946c6a7c21a529ull},
  };
  for (const Golden& c : golden) {
    curandGenerator_t g;
    curandCreateGenerator(&g, c.type);
    curandStatus_t st;
    const uint64_t h = fnv(draw(g, false, d, N, c.lambda, &st));
    std::printf("     golden {%d, %g, 0x%llxull}\n", c.type, c.lambda, (unsigned long long)h);
    char what[96];
    std::snprintf(what, sizeof what, "Sobol' type %d, lambda %g: the card's integers", c.type, c.lambda);
    check(st == CURAND_STATUS_SUCCESS && h == c.hash, what, c.lambda);
    curandDestroyGenerator(g);
  }
}

__global__ void discrete_kernel(unsigned long long seed, curandDiscreteDistribution_t dist, unsigned* out, int per) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  curandState s;
  curand_init(seed, t, 0, &s);
  for (int i = 0; i < per; ++i) out[t * per + i] = curand_discrete(&s, dist);
}

static void device_discrete(unsigned* d) {
  const struct {
    double lambda;
    uint64_t hash;
  } golden[] = {{0.5, 0x6e5f61e543733187ull}, {4, 0x5bbd20d190c76020ull}, {100, 0x555784d85b6d0235ull}, {1999, 0x4220a14422253369ull}, {2000, 0x23aeb43ca567986ull}, {50000, 0x425ca1e4cc1516deull}};
  const int threads = 256, per = 16;   // 4,096 integers
  for (const auto& c : golden) {
    curandDiscreteDistribution_t dist = nullptr;
    if (curandCreatePoissonDistribution(c.lambda, &dist) != CURAND_STATUS_SUCCESS) {
      check(false, "curandCreatePoissonDistribution", c.lambda);
      continue;
    }
    discrete_kernel<<<threads / 64, 64>>>(99, dist, d, per);
    std::vector<unsigned> h(threads * per);
    cudaError_t e = cudaMemcpy(h.data(), d, h.size() * sizeof(unsigned), cudaMemcpyDeviceToHost);
    const uint64_t hash = fnv(h);
    std::printf("     golden {%g, 0x%llxull}\n", c.lambda, (unsigned long long)hash);
    char what[96];
    std::snprintf(what, sizeof what, "curand_discrete in a kernel, lambda %g: the card's integers", c.lambda);
    check(e == cudaSuccess && hash == c.hash, what, c.lambda);
    check(curandDestroyDistribution(dist) == CURAND_STATUS_SUCCESS, "curandDestroyDistribution", c.lambda);
  }
}

int main() {
  unsigned* d = nullptr;
  cudaMalloc(&d, N * sizeof(unsigned));
  statuses(d);
  moments(d);
  sobol_exact(d);
  device_discrete(d);
  cudaFree(d);
  std::printf(failures ? "FAIL: %d cuRAND Poisson checks\n" : "PASS: every cuRAND Poisson check\n", failures);
  return failures ? 1 : 0;
}
