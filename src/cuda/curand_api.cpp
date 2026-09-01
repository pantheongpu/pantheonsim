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
#include <curand.h>

#include <cmath>
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

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

VGPU_EXPORT curandStatus_t curandCreateGenerator(curandGenerator_t* gen, curandRngType_t type) {
  if (!gen) return CURAND_STATUS_INITIALIZATION_FAILED;
  auto* g = new Generator();
  g->type = type;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_gens.insert(g);
  }
  *gen = reinterpret_cast<curandGenerator_t>(g);
  if (!quiet())
    std::fprintf(stderr, "[vgpu] cuRAND generator created (host-computed; bit stream differs "
                         "from NVIDIA's -- see docs/libraries.md)\n");
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
  std::vector<unsigned int> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = draw32(g->seed, g->offset + g->counter + i);
  g->counter += n;
  store(out, h);
  return CURAND_STATUS_SUCCESS;
}

VGPU_EXPORT curandStatus_t curandGenerateUniform(curandGenerator_t gen, float* out, size_t n) {
  if (!valid(gen)) return CURAND_STATUS_NOT_INITIALIZED;
  auto* g = reinterpret_cast<Generator*>(gen);
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
  std::vector<double> h(n);
  for (size_t i = 0; i < n; ++i)
    h[i] = uniform_double(mix(g->seed ^ mix(g->offset + g->counter + i)));
  g->counter += n;
  store(out, h);
  return CURAND_STATUS_SUCCESS;
}

// Box-Muller, as cuRAND documents for the normal generators: pairs of uniforms
// become pairs of normals, so n must be even for the device API.
VGPU_EXPORT curandStatus_t curandGenerateNormal(curandGenerator_t gen, float* out, size_t n,
                                                float mean, float stddev) {
  if (!valid(gen)) return CURAND_STATUS_NOT_INITIALIZED;
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
