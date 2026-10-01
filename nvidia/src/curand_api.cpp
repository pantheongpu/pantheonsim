// libvgpucurand — VirtualGPU's implementation of the cuRAND host API.
//
// Same boundary as cuBLAS: a random number generator is a library, so it runs
// on the host and writes results into virtual device memory rather than being
// interpreted. The generators reproduce cuRAND's documented algorithms so that
// a given seed produces a usable, deterministic stream.
//
// IMPORTANT AND DOCUMENTED DIVERGENCE: the exact bit sequence is *not*
// guaranteed to match NVIDIA's. cuRAND's per-generator state layout and
// substream skipping are unpublished in the detail needed to reproduce them
// exactly, so anything asserting on specific random values from hardware will
// differ. What is guaranteed: the right distribution, the right shape, and
// determinism for a given seed and offset -- which is what dropout masks,
// weight init and Monte-Carlo tests actually need.
//
// The Sobol' generators are the exception: their sequence is documented in
// full, and the unscrambled ones produce NVIDIA's integers and uniforms bit
// for bit (see "Sobol' sequences" below).
#include <curand.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

#include <cuda_runtime.h>

namespace {

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

struct Generator {
  curandRngType_t type = CURAND_RNG_PSEUDO_DEFAULT;
  unsigned long long seed = 0;
  unsigned long long offset = 0;
  unsigned long long counter = 0;
  curandOrdering_t ordering = CURAND_ORDERING_PSEUDO_DEFAULT;   // kept; see curandSetGeneratorOrdering
  unsigned dims = 1;   // a quasirandom generator's dimensions

  bool quasi() const { return type >= CURAND_RNG_QUASI_DEFAULT; }
  bool quasi64() const { return type == CURAND_RNG_QUASI_SOBOL64 || type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64; }
  bool scrambled() const {
    return type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 || type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
  }
};

std::mutex g_mu;
std::set<Generator*> g_gens;

bool valid(curandGenerator_t g) {
  std::lock_guard<std::mutex> lock(g_mu);
  return g && g_gens.count(reinterpret_cast<Generator*>(g));
}

// Philox-style counter-based bijection. Counter-based generation is what makes
// this reproducible without carrying hidden state: value i depends only on
// (seed, offset + i).
inline uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

inline uint32_t draw32(uint64_t seed, uint64_t index) {
  return static_cast<uint32_t>(mix(seed ^ mix(index)) >> 32);
}

// (0, 1] like cuRAND: never returns 0, may return 1.
inline float uniform_float(uint32_t r) {
  return (static_cast<float>(r) + 1.0f) * 2.3283064e-10f;
}
inline double uniform_double(uint64_t r) {
  return (static_cast<double>(r >> 11) + 1.0) * (1.0 / 9007199254740992.0);
}

template <class T>
void store(void* dev, const std::vector<T>& host) {
  if (!host.empty()) cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
}

// ---- Sobol' direction vectors ----
//
// cuRAND documents its direction vectors as generated from the polynomials
// S. Joe and F. Y. Kuo recommend, for 20,000 dimensions. These are built from
// Joe and Kuo's published direction numbers (nvidia/third_party/joe_kuo) with
// the standard recurrence: v_i = m_i << (bits - 1 - i) for the first s, then
// v_i = v_{i-s} ^ (v_{i-s} >> s) ^ the v_{i-k} whose coefficient a_k is set.
// Every one of the 20,000 dimensions, at 32 and at 64 bits, equals what an
// RTX 3060's cuRAND returns.
//
// The scrambled sets are a different matter: cuRAND does not document its
// scrambling, so these are scrambled here -- each dimension's vectors
// multiplied by a lower-triangular binary matrix with a unit diagonal (a
// Matousek linear scramble, the kind cuRAND's vectors show), from a fixed
// seed. That keeps every property a scrambled Sobol' sequence has, but not
// NVIDIA's bits, like the rest of this file. The scramble constants likewise.
#include "../third_party/joe_kuo/sobol_directions.inc"

constexpr int kSobolDims = 20000;

struct SobolParams {
  unsigned s = 0, a = 0;
  unsigned m[18] = {};   // Joe and Kuo's degrees stop at 18
};

std::vector<SobolParams> sobol_params() {
  std::vector<uint8_t> bytes;
  uint32_t acc = 0;
  int have = 0;
  for (const char* c = kSobolJoeKuo; *c && *c != '='; ++c) {
    const char* digits = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    acc = (acc << 6) | static_cast<uint32_t>(std::strchr(digits, *c) - digits);
    if ((have += 6) >= 8) bytes.push_back(static_cast<uint8_t>(acc >> (have -= 8)));
  }
  size_t bit = 0;
  auto take = [&](unsigned width) {
    unsigned v = 0;
    for (unsigned i = 0; i < width; ++i, ++bit) v = (v << 1) | ((bytes[bit / 8] >> (7 - bit % 8)) & 1u);
    return v;
  };
  std::vector<SobolParams> out(kSobolDims);   // dimension 0 has no parameters
  for (int d = 1; d < kSobolDims; ++d) {
    SobolParams& p = out[d];
    p.s = take(5);
    p.a = take(p.s - 1);
    for (unsigned k = 1; k <= p.s; ++k) p.m[k - 1] = 2 * take(k - 1) + 1;
  }
  return out;
}

template <class T>
void directions(const SobolParams& p, bool first, T* v) {
  constexpr int bits = static_cast<int>(sizeof(T) * 8);
  for (int i = 0; i < bits; ++i) {
    if (first) {
      v[i] = T{1} << (bits - 1 - i);
    } else if (i < static_cast<int>(p.s)) {
      v[i] = static_cast<T>(p.m[i]) << (bits - 1 - i);
    } else {
      T x = v[i - p.s] ^ (v[i - p.s] >> p.s);
      for (unsigned k = 1; k < p.s; ++k)
        if ((p.a >> (p.s - 1 - k)) & 1u) x ^= v[i - k];
      v[i] = x;
    }
  }
}

// The Matousek scramble of one dimension's vectors: output bit j (counting
// from the top) is the parity of the vector's bits at and above j, masked by
// row j of a random lower-triangular matrix whose diagonal is one.
template <class T>
void scramble(T* v, uint64_t seed) {
  constexpr int bits = static_cast<int>(sizeof(T) * 8);
  T rows[bits];
  for (int j = 0; j < bits; ++j) {
    const T above = j == bits - 1 ? ~T{0} : static_cast<T>(~(~T{0} >> (j + 1)));   // columns 0..j
    const T top = T{1} << (bits - 1 - j);
    rows[j] = (static_cast<T>(mix(seed + j)) & above) | top;
  }
  for (int i = 0; i < bits; ++i) {
    T out = 0;
    for (int j = 0; j < bits; ++j)
      if (__builtin_popcountll(static_cast<unsigned long long>(rows[j] & v[i])) & 1) out |= T{1} << (bits - 1 - j);
    v[i] = out;
  }
}

template <class T>
const std::vector<T>& direction_table(bool scrambled) {
  static std::once_flag once[2];
  static std::vector<T> table[2];
  std::call_once(once[scrambled], [&] {
    const std::vector<SobolParams> params = sobol_params();
    constexpr size_t bits = sizeof(T) * 8;
    std::vector<T>& t = table[scrambled];
    t.resize(kSobolDims * bits);
    for (int d = 0; d < kSobolDims; ++d) {
      directions(params[d], d == 0, &t[d * bits]);
      if (scrambled) scramble(&t[d * bits], mix(0x5c7a3b1e00000000ull ^ (uint64_t{bits} << 40) ^ d));
    }
  });
  return table[scrambled];
}

template <class T>
const std::vector<T>& scramble_constants() {
  static std::once_flag once;
  static std::vector<T> c;
  std::call_once(once, [] {
    c.resize(kSobolDims);
    for (int d = 0; d < kSobolDims; ++d) c[d] = static_cast<T>(mix(0x3c6ef372fe94f82bull ^ (uint64_t{sizeof(T)} << 40) ^ d));
  });
  return c;
}

// ---- Sobol' sequences ----
//
// The quasirandom generators, as cuRAND documents them and an RTX 3060's
// cuRAND produces them, bit for bit for the unscrambled ones: point n of a
// dimension is the XOR of the direction vectors picked by the bits of n's Gray
// code, n ^ (n >> 1), starting from 0 -- or from the dimension's scramble
// constant, for the scrambled ones. Results come dimension by dimension
// (CURAND_ORDERING_QUASI_DEFAULT): of n values from d dimensions, the first
// n / d are dimension 0's next points, and so on, and each call carries on
// from where the last stopped. The conversions to floating point are the ones
// curand_uniform.h spells out.
template <class Raw>
Raw sobol_point(const Generator& g, unsigned d, uint64_t n) {
  constexpr unsigned bits = sizeof(Raw) * 8;
  const std::vector<Raw>& v = direction_table<Raw>(g.scrambled());
  Raw x = g.scrambled() ? scramble_constants<Raw>()[d] : 0;
  for (uint64_t gray = n ^ (n >> 1), b = 0; gray; gray >>= 1, ++b)
    if (gray & 1) x ^= v[d * bits + b];
  return x;
}

// n values of a quasirandom generator, each point through `convert`.
template <class T, class F>
curandStatus_t quasi_fill(Generator* g, T* out, size_t n, F&& convert) {
  if (n % g->dims) return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
  const size_t per = n / g->dims;
  std::vector<T> h(n);
  for (unsigned d = 0; d < g->dims; ++d)
    for (size_t k = 0; k < per; ++k) {
      const uint64_t i = g->offset + g->counter + k;
      h[d * per + k] = g->quasi64() ? convert(sobol_point<unsigned long long>(*g, d, i))
                                    : convert(sobol_point<uint32_t>(*g, d, i));
    }
  g->counter += per;
  store(out, h);
  return CURAND_STATUS_SUCCESS;
}

inline float quasi_float(uint32_t x) { return x * 2.3283064e-10f + 2.3283064e-10f / 2.0f; }
inline float quasi_float(unsigned long long x) { return quasi_float(static_cast<uint32_t>(x >> 32)); }
inline double quasi_double(uint32_t x) { return x * 2.3283064365386963e-10 + 2.3283064365386963e-10; }
inline double quasi_double(unsigned long long x) {
  return static_cast<double>(x >> 11) * 1.1102230246251565e-16 + 1.1102230246251565e-16 / 2.0;
}

// The standard normal quantile, for the quasirandom normals: they come from
// the inverse distribution function rather than Box-Muller, which would pair
// up and so scramble the dimensions. Acklam's rational approximation, then a
// Halley step against erfc, to double precision. Not cuRAND's own erfcinv,
// so the last bits may differ from the card's.
double normal_quantile(double p) {
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                             6.680131188771972e+01, -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
  static const double e[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                             3.754408661907416e+00};
  double x;
  if (p < 0.02425) {
    const double q = std::sqrt(-2 * std::log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((e[0] * q + e[1]) * q + e[2]) * q + e[3]) * q + 1);
  } else if (p > 1 - 0.02425) {
    const double q = std::sqrt(-2 * std::log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((e[0] * q + e[1]) * q + e[2]) * q + e[3]) * q + 1);
  } else {
    const double q = p - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  const double err = 0.5 * std::erfc(-x / std::sqrt(2.0)) - p;
  const double u = err * std::sqrt(2 * 3.14159265358979323846) * std::exp(x * x / 2);
  return x - u / (1 + x * u / 2);
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

VGPU_EXPORT curandStatus_t curandCreateGenerator(curandGenerator_t* gen, curandRngType_t type) {
  if (!gen) return CURAND_STATUS_INITIALIZATION_FAILED;
  auto* g = new Generator();
  g->type = type == CURAND_RNG_QUASI_DEFAULT ? CURAND_RNG_QUASI_SOBOL32 : type;
  if (g->quasi()) g->ordering = CURAND_ORDERING_QUASI_DEFAULT;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_gens.insert(g);
  }
  *gen = reinterpret_cast<curandGenerator_t>(g);
  if (!quiet())
    std::fprintf(stderr, "[vgpu] cuRAND generator created (host-computed; bit stream differs "
                         "from NVIDIA's -- see nvidia/docs/libraries.md)\n");
  return CURAND_STATUS_SUCCESS;
}
VGPU_EXPORT curandStatus_t curandCreateGeneratorHost(curandGenerator_t* g, curandRngType_t t) {
  return curandCreateGenerator(g, t);
}

VGPU_EXPORT curandStatus_t curandDestroyGenerator(curandGenerator_t gen) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto* g = reinterpret_cast<Generator*>(gen);
  if (!g || !g_gens.erase(g)) return CURAND_STATUS_NOT_INITIALIZED;
  delete g;
  return CURAND_STATUS_SUCCESS;
}

VGPU_EXPORT curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t gen,
                                                              unsigned long long seed) {
  if (!valid(gen)) return CURAND_STATUS_NOT_INITIALIZED;
  if (reinterpret_cast<Generator*>(gen)->quasi()) return CURAND_STATUS_TYPE_ERROR;   // no seed to set
  // Matching hardware: setting the seed does NOT rewind the stream. Real
  // cuRAND keeps its position, so two runs that only re-seed do not reproduce
  // each other; curandSetGeneratorOffset is what rewinds.
  reinterpret_cast<Generator*>(gen)->seed = seed;
  return CURAND_STATUS_SUCCESS;
}
VGPU_EXPORT curandStatus_t curandSetGeneratorOffset(curandGenerator_t gen,
                                                    unsigned long long offset) {
  if (!valid(gen)) return CURAND_STATUS_NOT_INITIALIZED;
  auto* g = reinterpret_cast<Generator*>(gen);
  g->offset = offset;
  g->counter = 0;
  return CURAND_STATUS_SUCCESS;
}
// A quasirandom generator's dimensions, 1 to 20,000. The position in the
// sequence is kept, as the card keeps it.
VGPU_EXPORT curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t gen, unsigned int dims) {
  if (!valid(gen)) return CURAND_STATUS_NOT_INITIALIZED;
  auto* g = reinterpret_cast<Generator*>(gen);
  if (!g->quasi()) return CURAND_STATUS_TYPE_ERROR;
  if (dims == 0 || dims > static_cast<unsigned>(kSobolDims)) return CURAND_STATUS_OUT_OF_RANGE;
  g->dims = dims;
  return CURAND_STATUS_SUCCESS;
}

// Which orderings each generator takes, as an RTX 3060's cuRAND answers: every
// pseudorandom one takes BEST, DEFAULT and LEGACY; XORWOW alone takes SEEDED;
// all but MT19937 take DYNAMIC; the quasirandom ones take only QUASI_DEFAULT.
// The ordering is kept but changes nothing, since the stream here is not
// NVIDIA's in any ordering (see the top of this file).
VGPU_EXPORT curandStatus_t curandSetGeneratorOrdering(curandGenerator_t gen, curandOrdering_t order) {
  if (!valid(gen)) return CURAND_STATUS_NOT_INITIALIZED;
  auto* g = reinterpret_cast<Generator*>(gen);
  const int t = g->type, o = order;
  bool ok = false;
  if (t >= CURAND_RNG_QUASI_DEFAULT) {
    ok = o == CURAND_ORDERING_QUASI_DEFAULT;
  } else {
    const bool xorwow = t == CURAND_RNG_PSEUDO_DEFAULT || t == CURAND_RNG_PSEUDO_XORWOW;
    switch (o) {
      case CURAND_ORDERING_PSEUDO_BEST:
      case CURAND_ORDERING_PSEUDO_DEFAULT:
      case CURAND_ORDERING_PSEUDO_LEGACY: ok = true; break;
      case CURAND_ORDERING_PSEUDO_SEEDED: ok = xorwow; break;
      case CURAND_ORDERING_PSEUDO_DYNAMIC: ok = t != CURAND_RNG_PSEUDO_MT19937; break;
      default: break;
    }
  }
  if (!ok) return CURAND_STATUS_OUT_OF_RANGE;
  g->ordering = order;
  return CURAND_STATUS_SUCCESS;
}

// Host arrays, one set of 32 (or 64) vectors per dimension, for 20,000
// dimensions; the same pointer each time. A set of the other width is
// OUT_OF_RANGE, as on the card.
VGPU_EXPORT curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t* vectors[],
                                                       curandDirectionVectorSet_t set) {
  if (set != CURAND_DIRECTION_VECTORS_32_JOEKUO6 && set != CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
    return CURAND_STATUS_OUT_OF_RANGE;
  if (!vectors) return CURAND_STATUS_OUT_OF_RANGE;   // the card's library crashes
  const auto& t = direction_table<uint32_t>(set == CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
  *vectors = reinterpret_cast<curandDirectionVectors32_t*>(const_cast<uint32_t*>(t.data()));
  return CURAND_STATUS_SUCCESS;
}
VGPU_EXPORT curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t* vectors[],
                                                       curandDirectionVectorSet_t set) {
  if (set != CURAND_DIRECTION_VECTORS_64_JOEKUO6 && set != CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)
    return CURAND_STATUS_OUT_OF_RANGE;
  if (!vectors) return CURAND_STATUS_OUT_OF_RANGE;
  static_assert(sizeof(unsigned long long) == sizeof(uint64_t));
  const auto& t = direction_table<unsigned long long>(set == CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6);
  *vectors = reinterpret_cast<curandDirectionVectors64_t*>(const_cast<unsigned long long*>(t.data()));
  return CURAND_STATUS_SUCCESS;
}
// One constant per dimension, for the scrambled sequences.
VGPU_EXPORT curandStatus_t curandGetScrambleConstants32(unsigned int** constants) {
  if (!constants) return CURAND_STATUS_OUT_OF_RANGE;
  *constants = const_cast<unsigned int*>(scramble_constants<unsigned int>().data());
  return CURAND_STATUS_SUCCESS;
}
VGPU_EXPORT curandStatus_t curandGetScrambleConstants64(unsigned long long** constants) {
  if (!constants) return CURAND_STATUS_OUT_OF_RANGE;
  *constants = const_cast<unsigned long long*>(scramble_constants<unsigned long long>().data());
  return CURAND_STATUS_SUCCESS;
}

VGPU_EXPORT curandStatus_t curandSetStream(curandGenerator_t gen, cudaStream_t) {
  return valid(gen) ? CURAND_STATUS_SUCCESS : CURAND_STATUS_NOT_INITIALIZED;
}
VGPU_EXPORT curandStatus_t curandGetVersion(int* version) {
  if (version) *version = CURAND_VERSION;
  return CURAND_STATUS_SUCCESS;
}

VGPU_EXPORT curandStatus_t curandGenerate(curandGenerator_t gen, unsigned int* out, size_t n) {
  if (!valid(gen)) return CURAND_STATUS_NOT_INITIALIZED;
  auto* g = reinterpret_cast<Generator*>(gen);
  if (g->quasi64()) return CURAND_STATUS_TYPE_ERROR;   // curandGenerateLongLong's
  if (g->quasi()) return quasi_fill(g, out, n, [](auto x) { return static_cast<unsigned int>(x); });
  std::vector<unsigned int> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = draw32(g->seed, g->offset + g->counter + i);
  g->counter += n;
  store(out, h);
  return CURAND_STATUS_SUCCESS;
}

// 64-bit integers: the 64-bit Sobol' generators' own output, and nothing else's.
VGPU_EXPORT curandStatus_t curandGenerateLongLong(curandGenerator_t gen, unsigned long long* out, size_t n) {
  if (!valid(gen)) return CURAND_STATUS_NOT_INITIALIZED;
  auto* g = reinterpret_cast<Generator*>(gen);
  if (!g->quasi64()) return CURAND_STATUS_TYPE_ERROR;
  return quasi_fill(g, out, n, [](auto x) { return static_cast<unsigned long long>(x); });
}

VGPU_EXPORT curandStatus_t curandGenerateUniform(curandGenerator_t gen, float* out, size_t n) {
  if (!valid(gen)) return CURAND_STATUS_NOT_INITIALIZED;
  auto* g = reinterpret_cast<Generator*>(gen);
  if (g->quasi()) return quasi_fill(g, out, n, [](auto x) { return quasi_float(x); });
  std::vector<float> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = uniform_float(draw32(g->seed, g->offset + g->counter + i));
  g->counter += n;
  store(out, h);
  return CURAND_STATUS_SUCCESS;
}

VGPU_EXPORT curandStatus_t curandGenerateUniformDouble(curandGenerator_t gen, double* out,
                                                       size_t n) {
  if (!valid(gen)) return CURAND_STATUS_NOT_INITIALIZED;
  auto* g = reinterpret_cast<Generator*>(gen);
  if (g->quasi()) return quasi_fill(g, out, n, [](auto x) { return quasi_double(x); });
  std::vector<double> h(n);
  for (size_t i = 0; i < n; ++i)
    h[i] = uniform_double(mix(g->seed ^ mix(g->offset + g->counter + i)));
  g->counter += n;
  store(out, h);
  return CURAND_STATUS_SUCCESS;
}

// Box-Muller, as cuRAND documents for the pseudorandom normal generators: pairs
// of uniforms become pairs of normals, so n must be even for the device API.
// The quasirandom ones take the inverse distribution function instead.
VGPU_EXPORT curandStatus_t curandGenerateNormal(curandGenerator_t gen, float* out, size_t n,
                                                float mean, float stddev) {
  if (!valid(gen)) return CURAND_STATUS_NOT_INITIALIZED;
  if (auto* q = reinterpret_cast<Generator*>(gen); q->quasi())
    return quasi_fill(q, out, n, [&](auto x) {
      return mean + stddev * static_cast<float>(normal_quantile(quasi_float(x)));
    });
  if (n % 2) return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
  auto* g = reinterpret_cast<Generator*>(gen);
  std::vector<float> h(n);
  for (size_t i = 0; i < n; i += 2) {
    float u1 = uniform_float(draw32(g->seed, g->offset + g->counter + i));
    float u2 = uniform_float(draw32(g->seed, g->offset + g->counter + i + 1));
    float r = std::sqrt(-2.0f * std::log(u1)), theta = 6.2831853f * u2;
    h[i] = mean + stddev * r * std::cos(theta);
    h[i + 1] = mean + stddev * r * std::sin(theta);
  }
  g->counter += n;
  store(out, h);
  return CURAND_STATUS_SUCCESS;
}

VGPU_EXPORT curandStatus_t curandGenerateNormalDouble(curandGenerator_t gen, double* out, size_t n,
                                                      double mean, double stddev) {
  if (!valid(gen)) return CURAND_STATUS_NOT_INITIALIZED;
  if (auto* q = reinterpret_cast<Generator*>(gen); q->quasi())
    return quasi_fill(q, out, n, [&](auto x) { return mean + stddev * normal_quantile(quasi_double(x)); });
  if (n % 2) return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
  auto* g = reinterpret_cast<Generator*>(gen);
  std::vector<double> h(n);
  for (size_t i = 0; i < n; i += 2) {
    double u1 = uniform_double(mix(g->seed ^ mix(g->offset + g->counter + i)));
    double u2 = uniform_double(mix(g->seed ^ mix(g->offset + g->counter + i + 1)));
    double r = std::sqrt(-2.0 * std::log(u1)), theta = 6.283185307179586 * u2;
    h[i] = mean + stddev * r * std::cos(theta);
    h[i + 1] = mean + stddev * r * std::sin(theta);
  }
  g->counter += n;
  store(out, h);
  return CURAND_STATUS_SUCCESS;
}

VGPU_EXPORT curandStatus_t curandGenerateLogNormal(curandGenerator_t gen, float* out, size_t n,
                                                   float mean, float stddev) {
  curandStatus_t s = curandGenerateNormal(gen, out, n, mean, stddev);
  if (s != CURAND_STATUS_SUCCESS) return s;
  std::vector<float> h(n);
  cudaMemcpy(h.data(), out, n * sizeof(float), cudaMemcpyDeviceToHost);
  for (auto& v : h) v = std::exp(v);
  store(out, h);
  return CURAND_STATUS_SUCCESS;
}

VGPU_EXPORT curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t gen, double* out,
                                                         size_t n, double mean, double stddev) {
  curandStatus_t s = curandGenerateNormalDouble(gen, out, n, mean, stddev);
  if (s != CURAND_STATUS_SUCCESS) return s;
  std::vector<double> h(n);
  cudaMemcpy(h.data(), out, n * sizeof(double), cudaMemcpyDeviceToHost);
  for (auto& v : h) v = std::exp(v);
  store(out, h);
  return CURAND_STATUS_SUCCESS;
}
