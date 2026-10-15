// cuRAND's host generators fill their output in pieces on the host's threads. This pins what that must not change:
//  * a quasirandom (Sobol') generator's n points in one call are the n points it gives one call at a time (a point is the XOR of the
//    direction vectors its Gray code picks; the long call steps from one point to the next by one vector, the short calls work
//    each point out whole), for 32- and 64-bit and scrambled generators, in several dimensions, from an offset, floats and integers;
//  * a pseudorandom generator's long call is the same numbers as the calls that make it up, whatever the split (value i depends
//    only on seed and position);
//  * log-normal numbers are exp() of the normals the same generator gives.
// Not run against the card (the Sobol' integers themselves are checked against it in rand_paths).
#include <cuda_runtime.h>
#include <curand.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0, checks = 0;
static void report(bool ok, const char* what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what);
  }
}
#define CK(x) do { const curandStatus_t s_ = (x); if (s_ != CURAND_STATUS_SUCCESS) { std::printf("FAIL %s -> %d\n", #x, (int)s_); ++failures; } } while (0)

template <class T> static std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

static void sobol_case(curandRngType_t type, unsigned dims, unsigned long long offset, bool as_float) {
  const size_t per = 40000;   // across the host's pieces
  const size_t n = per * dims;
  curandGenerator_t g;
  CK(curandCreateGenerator(&g, type));
  CK(curandSetQuasiRandomGeneratorDimensions(g, dims));
  if (offset) CK(curandSetGeneratorOffset(g, offset));
  void* d = nullptr;
  cudaMalloc(&d, n * 8 + 16);
  std::vector<unsigned char> big, one;
  const bool is64 = type == CURAND_RNG_QUASI_SOBOL64 || type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
  const size_t es = as_float ? 4 : (is64 ? 8 : 4);
  auto draw = [&](size_t count) {
    if (as_float) CK(curandGenerateUniform(g, (float*)d, count));
    else if (is64) CK(curandGenerateLongLong(g, (unsigned long long*)d, count));
    else CK(curandGenerate(g, (unsigned int*)d, count));
  };
  draw(n);
  big.resize(n * es);
  cudaMemcpy(big.data(), d, n * es, cudaMemcpyDeviceToHost);
  CK(curandDestroyGenerator(g));
  // the same points one call at a time: each call is one point of every dimension
  CK(curandCreateGenerator(&g, type));
  CK(curandSetQuasiRandomGeneratorDimensions(g, dims));
  if (offset) CK(curandSetGeneratorOffset(g, offset));
  bool ok = true;
  const size_t step = 1;   // every point
  for (size_t k = 0; k < per && ok; k += step) {
    draw(dims);
    one.resize(dims * es);
    cudaMemcpy(one.data(), d, dims * es, cudaMemcpyDeviceToHost);
    for (unsigned dim = 0; dim < dims; ++dim)
      if (std::memcmp(one.data() + dim * es, big.data() + (dim * per + k) * es, es) != 0) { ok = false; break; }
  }
  CK(curandDestroyGenerator(g));
  char what[160];
  std::snprintf(what, sizeof what, "Sobol type %d dims %u offset %llu %s: %zu points in one call are the points one call at a time", (int)type, dims, offset, as_float ? "float" : "integer", per);
  report(ok, what);
  cudaFree(d);
}

static void pseudo_case(curandRngType_t type) {
  const size_t n = 100000;
  float* d = nullptr;
  cudaMalloc(&d, n * 4 * 2);
  curandGenerator_t g;
  CK(curandCreateGenerator(&g, type));
  CK(curandSetPseudoRandomGeneratorSeed(g, 4242ULL));
  CK(curandGenerateUniform(g, d, n));
  const auto whole = host(d, n);
  CK(curandDestroyGenerator(g));
  CK(curandCreateGenerator(&g, type));
  CK(curandSetPseudoRandomGeneratorSeed(g, 4242ULL));
  std::vector<float> parts;
  for (size_t done = 0, sz = 1; done < n; done += sz, sz = sz * 3 + 1) {
    const size_t c = std::min(sz, n - done);
    CK(curandGenerateUniform(g, d, c));
    const auto h = host(d, c);
    parts.insert(parts.end(), h.begin(), h.end());
  }
  CK(curandDestroyGenerator(g));
  char what[120];
  std::snprintf(what, sizeof what, "type %d: one call of %zu uniforms is the calls that make it up", (int)type, n);
  report(parts.size() == whole.size() && std::memcmp(parts.data(), whole.data(), n * 4) == 0, what);
  // log-normal = exp(normal), float and double
  curandGenerator_t gn, gl;
  CK(curandCreateGenerator(&gn, type));
  CK(curandCreateGenerator(&gl, type));
  CK(curandSetPseudoRandomGeneratorSeed(gn, 99ULL));
  CK(curandSetPseudoRandomGeneratorSeed(gl, 99ULL));
  CK(curandGenerateNormal(gn, d, n, 0.25f, 1.5f));
  const auto nv = host(d, n);
  CK(curandGenerateLogNormal(gl, d, n, 0.25f, 1.5f));
  const auto lv = host(d, n);
  bool ok = true;
  for (size_t i = 0; i < n; ++i) {
    const float want = std::exp(nv[i]);
    if (std::memcmp(&want, &lv[i], 4) != 0 && !(std::isnan(want) && std::isnan(lv[i]))) { ok = false; break; }
  }
  std::snprintf(what, sizeof what, "type %d: log-normal floats are exp of the normals", (int)type);
  report(ok, what);
  double* dd = nullptr;
  cudaMalloc(&dd, n * 8);
  CK(curandGenerateNormalDouble(gn, dd, n, -0.5, 2.0));
  const auto ndv = host(dd, n);
  CK(curandGenerateLogNormalDouble(gl, dd, n, -0.5, 2.0));
  const auto ldv = host(dd, n);
  ok = true;
  for (size_t i = 0; i < n; ++i) {
    const double want = std::exp(ndv[i]);
    if (std::memcmp(&want, &ldv[i], 8) != 0 && !(std::isnan(want) && std::isnan(ldv[i]))) { ok = false; break; }
  }
  std::snprintf(what, sizeof what, "type %d: log-normal doubles are exp of the normals", (int)type);
  report(ok, what);
  CK(curandDestroyGenerator(gn));
  CK(curandDestroyGenerator(gl));
  cudaFree(d);
  cudaFree(dd);
}

int main() {
  for (const curandRngType_t t : {CURAND_RNG_QUASI_SOBOL32, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32, CURAND_RNG_QUASI_SOBOL64, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64})
    for (const unsigned dims : {1u, 3u})
      for (const unsigned long long offset : {0ULL, 12345ULL, 70000ULL}) {
        sobol_case(t, dims, offset, false);
        sobol_case(t, dims, offset, true);
      }
  for (const curandRngType_t t : {CURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_PHILOX4_32_10}) pseudo_case(t);
  std::printf("%d checks, %d failed\n", checks, failures);
  std::printf(failures ? "FAIL\n" : "PASS\n");
  return failures ? 1 : 0;
}
