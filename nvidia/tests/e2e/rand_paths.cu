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
//
// Also the Sobol' direction vectors and scramble constants, the Sobol'
// generators themselves, and which orderings each generator takes.
#include <cuda_runtime.h>
#include <curand.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
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

// FNV-1a over a table's bytes.
static uint64_t fnv(const void* p, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const unsigned char*>(p)[i]) * 0x100000001b3ull;
  return h;
}

// Whether vector i's lowest one is bit (bits - 1 - i): the direction numbers
// m_i are odd and shifted up by that much.
template <class T> static bool lowest_bits(const T* v) {
  const int bits = static_cast<int>(sizeof(T) * 8);
  for (int i = 0; i < bits; ++i) {
    const T one = T{1} << (bits - 1 - i);
    if (!(v[i] & one) || (v[i] & (one - 1))) return false;
  }
  return true;
}
// Whether vector i's highest one is bit (bits - 1 - i): the first dimension's
// vectors are the identity, and scrambling multiplies them by a
// lower-triangular matrix with a unit diagonal, whose columns these are.
template <class T> static bool highest_bits(const T* v) {
  const int bits = static_cast<int>(sizeof(T) * 8);
  for (int i = 0; i < bits; ++i)
    if ((v[i] >> (bits - 1 - i)) != 1) return false;
  return true;
}

// Sobol' direction vectors. The unscrambled sets are Joe and Kuo's, and the
// hashes are of the whole of what an RTX 3060's cuRAND returns, 20,000
// dimensions each. The scrambled sets are cuRAND's own and not reproduced
// bit for bit: what is checked is what any scrambled set must be.
static void directions() {
  curandDirectionVectors32_t* v32 = nullptr;
  curandDirectionVectors32_t* s32 = nullptr;
  curandDirectionVectors64_t* v64 = nullptr;
  curandDirectionVectors64_t* s64 = nullptr;
  CK(curandGetDirectionVectors32(&v32, CURAND_DIRECTION_VECTORS_32_JOEKUO6));
  CK(curandGetDirectionVectors32(&s32, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
  CK(curandGetDirectionVectors64(&v64, CURAND_DIRECTION_VECTORS_64_JOEKUO6));
  CK(curandGetDirectionVectors64(&s64, CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));
  const uint64_t h32 = fnv(v32, 20000 * sizeof(curandDirectionVectors32_t));
  const uint64_t h64 = fnv(v64, 20000 * sizeof(curandDirectionVectors64_t));
  check(h32 == 0x977d0241e3974985ull, "32-bit direction vectors, all 20,000 dimensions, as the card's", (double)(h32 >> 40));
  check(h64 == 0xd1d26f6f25e5d2f2ull, "64-bit direction vectors, all 20,000 dimensions, as the card's", (double)(h64 >> 40));
  check(v32[1][3] == 0xf0000000u && v64[2][4] == 0xe800000000000000ull, "dimensions 1 and 2 by value", 0);
  curandDirectionVectors32_t* again = nullptr;
  CK(curandGetDirectionVectors32(&again, CURAND_DIRECTION_VECTORS_32_JOEKUO6));
  check(again == v32, "the same table each time", 0);
  bool low = true, differ = false;
  for (int d : {0, 1, 2, 999, 19999}) {
    low = low && lowest_bits(v32[d]) && lowest_bits(v64[d]);
    differ = differ || std::memcmp(v32[d], s32[d], sizeof v32[d]) != 0;
  }
  check(low, "each direction number odd, in its place", 0);
  check(highest_bits(s32[0]) && highest_bits(s64[0]), "the first dimension scrambled by a unit lower-triangular matrix", 0);
  check(differ, "the scrambled vectors are scrambled", 0);
  curandDirectionVectors32_t* x = nullptr;
  curandDirectionVectors64_t* y = nullptr;
  check(curandGetDirectionVectors32(&x, CURAND_DIRECTION_VECTORS_64_JOEKUO6) == CURAND_STATUS_OUT_OF_RANGE,
        "a 64-bit set is out of range for 32 bits", 0);
  check(curandGetDirectionVectors64(&y, CURAND_DIRECTION_VECTORS_32_JOEKUO6) == CURAND_STATUS_OUT_OF_RANGE,
        "and a 32-bit one for 64", 0);
  unsigned* c32 = nullptr;
  unsigned long long* c64 = nullptr;
  CK(curandGetScrambleConstants32(&c32));
  CK(curandGetScrambleConstants64(&c64));
  check(c32 && c64, "scramble constants", 0);
}

// The Sobol' generators, whose sequence cuRAND documents in full: these values
// are an RTX 3060's, bit for bit.
static void sobol() {
  curandGenerator_t g;
  CK(curandCreateGenerator(&g, CURAND_RNG_QUASI_SOBOL32));
  CK(curandSetQuasiRandomGeneratorDimensions(g, 3));
  unsigned* d = nullptr;
  cudaMalloc(&d, 64 * sizeof(unsigned));
  CK(curandGenerate(g, d, 24));
  auto v = down(d, 24);
  // Top nibbles of each dimension's first eight points, dimension by dimension.
  const unsigned want[24] = {0x0, 0x8, 0xc, 0x4, 0x6, 0xe, 0xa, 0x2, 0x0, 0x8, 0x4, 0xc,
                             0x6, 0xe, 0x2, 0xa, 0x0, 0x8, 0x4, 0xc, 0xa, 0x2, 0xe, 0x6};
  bool same = true;
  for (int i = 0; i < 24; ++i) same = same && v[i] == want[i] << 28;
  check(same, "Sobol32: three dimensions, eight points each, in Gray-code order", 0);
  CK(curandGenerate(g, d, 6));
  v = down(d, 6);
  check(v[0] == 0x30000000u && v[1] == 0xb0000000u && v[2] == 0x50000000u && v[5] == 0x70000000u,
        "the next call carries on from point 8", 0);
  check(curandGenerate(g, d, 7) == CURAND_STATUS_LENGTH_NOT_MULTIPLE, "a count that is not a multiple of 3", 0);
  CK(curandSetQuasiRandomGeneratorDimensions(g, 3));
  CK(curandGenerate(g, d, 3));
  v = down(d, 3);
  check(v[0] == 0xf0000000u && v[1] == 0x10000000u,   // point 10: the Gray code sets bits 0 to 3
        "setting the dimensions again keeps the position", v[0]);
  CK(curandSetGeneratorOffset(g, 5));
  CK(curandGenerate(g, d, 6));
  v = down(d, 6);
  check(v[0] == 0xe0000000u && v[1] == 0xa0000000u && v[2] == 0xe0000000u && v[3] == 0x20000000u,
        "an offset of 5 starts each dimension at point 5", 0);
  float* f = nullptr;
  double* dd = nullptr;
  cudaMalloc(&f, 64 * sizeof(float));
  cudaMalloc(&dd, 64 * sizeof(double));
  CK(curandSetGeneratorOffset(g, 0));
  CK(curandGenerateUniform(g, f, 6));
  auto u = down(f, 6);
  check(u[0] == 1.16415322e-10f && u[1] == 0.5f, "uniform floats: x / 2^32 + 2^-33", u[0]);
  CK(curandSetGeneratorOffset(g, 0));
  CK(curandGenerateUniformDouble(g, dd, 6));
  auto ud = down(dd, 6);
  check(ud[0] == 2.3283064365386963e-10 && ud[1] == 0.50000000023283064, "uniform doubles: (x + 1) / 2^32", ud[0]);
  CK(curandSetQuasiRandomGeneratorDimensions(g, 1));
  CK(curandSetGeneratorOffset(g, 0));
  CK(curandGenerateNormal(g, f, 8, 0.0f, 1.0f));
  auto nn = down(f, 8);
  check(nn[1] == 0 && std::fabs(nn[2] - 0.674489737f) < 1e-6f && nn[3] == -nn[2] && nn[0] < -6.3f && nn[0] > -6.4f,
        "normals from the inverse distribution function", nn[2]);
  check(curandSetPseudoRandomGeneratorSeed(g, 5) == CURAND_STATUS_TYPE_ERROR, "a quasirandom generator has no seed", 0);
  check(curandSetQuasiRandomGeneratorDimensions(g, 0) == CURAND_STATUS_OUT_OF_RANGE &&
            curandSetQuasiRandomGeneratorDimensions(g, 20001) == CURAND_STATUS_OUT_OF_RANGE &&
            curandSetQuasiRandomGeneratorDimensions(g, 20000) == CURAND_STATUS_SUCCESS,
        "1 to 20,000 dimensions", 0);
  curandDestroyGenerator(g);

  CK(curandCreateGenerator(&g, CURAND_RNG_QUASI_SOBOL64));
  CK(curandSetQuasiRandomGeneratorDimensions(g, 2));
  check(curandGenerate(g, d, 4) == CURAND_STATUS_TYPE_ERROR, "Sobol64 has no 32-bit output", 0);
  unsigned long long* l = nullptr;
  cudaMalloc(&l, 8 * sizeof(unsigned long long));
  CK(curandGenerateLongLong(g, l, 8));
  auto ll = down(l, 8);
  check(ll[1] == 0x8000000000000000ull && ll[2] == 0xc000000000000000ull && ll[6] == 0x4000000000000000ull,
        "Sobol64's 64-bit integers", 0);
  CK(curandSetGeneratorOffset(g, 0));
  CK(curandGenerateUniformDouble(g, dd, 8));
  ud = down(dd, 8);
  check(ud[0] == 5.5511151231257827e-17 && ud[1] == 0.5 && ud[3] == 0.25000000000000006,
        "Sobol64's doubles: (x >> 11) / 2^53 + 2^-54", ud[3]);
  curandDestroyGenerator(g);

  // Scrambled: the same sequence from each dimension's scramble constant,
  // through the scrambled vectors.
  CK(curandCreateGenerator(&g, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
  CK(curandGenerate(g, d, 3));
  v = down(d, 3);
  unsigned* c = nullptr;
  curandDirectionVectors32_t* sv = nullptr;
  CK(curandGetScrambleConstants32(&c));
  CK(curandGetDirectionVectors32(&sv, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
  check(v[0] == c[0] && v[1] == (c[0] ^ sv[0][0]) && v[2] == (c[0] ^ sv[0][0] ^ sv[0][1]),
        "scrambled Sobol32 starts from the scramble constant", 0);
  curandDestroyGenerator(g);
  curandGenerator_t p;
  CK(curandCreateGenerator(&p, CURAND_RNG_PSEUDO_DEFAULT));
  check(curandSetQuasiRandomGeneratorDimensions(p, 2) == CURAND_STATUS_TYPE_ERROR &&
            curandGenerateLongLong(p, l, 2) == CURAND_STATUS_TYPE_ERROR,
        "a pseudorandom generator has no dimensions and no 64-bit output", 0);
  curandDestroyGenerator(p);
  cudaFree(d);
  cudaFree(f);
  cudaFree(dd);
  cudaFree(l);
}

// The orderings each generator takes, as an RTX 3060's cuRAND answers.
static void orderings() {
  const int types[] = {CURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MTGP32,
                       CURAND_RNG_PSEUDO_MT19937, CURAND_RNG_PSEUDO_PHILOX4_32_10, CURAND_RNG_QUASI_SOBOL32,
                       CURAND_RNG_QUASI_SCRAMBLED_SOBOL64};
  // BEST, DEFAULT, SEEDED, LEGACY, DYNAMIC, QUASI_DEFAULT: 1 where it is taken.
  const int orders[] = {100, 101, 102, 103, 104, 201};
  const char* want[] = {"111110", "110110", "110110", "110100", "110110", "000001", "000001"};
  for (int t = 0; t < 7; ++t) {
    curandGenerator_t g;
    if (curandCreateGenerator(&g, (curandRngType_t)types[t]) != CURAND_STATUS_SUCCESS) {
      check(false, "create a generator", types[t]);
      continue;
    }
    std::string got;
    for (int o : orders) {
      const curandStatus_t st = curandSetGeneratorOrdering(g, (curandOrdering_t)o);
      got += st == CURAND_STATUS_SUCCESS ? '1' : st == CURAND_STATUS_OUT_OF_RANGE ? '0' : '?';
    }
    char what[96];
    std::snprintf(what, sizeof what, "generator %d takes orderings %s", types[t], want[t]);
    check(got == want[t], what, 0);
    curandDestroyGenerator(g);
  }
  check(curandSetGeneratorOrdering(nullptr, CURAND_ORDERING_PSEUDO_DEFAULT) == CURAND_STATUS_NOT_INITIALIZED,
        "no generator", 0);
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
  directions();
  sobol();
  orderings();
  std::printf(failures ? "FAIL: %d cuRAND checks\n" : "PASS: every cuRAND check\n", failures);
  return failures ? 1 : 0;
}
