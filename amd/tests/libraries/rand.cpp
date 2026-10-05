// AMD's hipRAND, unmodified (over rocRAND), on a simulated MI300X. rocRAND
// makes the same numbers on the host as on the device for the same generator
// and seed, so each device generator is checked against a host one:
//
//   PHILOX4_32_10, XORWOW, MRG32K3A and MTGP32: 32-bit integers, exactly;
//     uniform floats, exactly; normal floats and log-normal doubles, to
//     within rounding (the host's exp, log and sqrt are its own)
//   SOBOL32, a quasi-random generator: 32-bit integers in two dimensions
//   Poisson integers from PHILOX4_32_10, exactly
//
// And the device API that kernels call themselves (hiprand_kernel.h): each
// thread's Philox state, from hiprand_init with the thread as its subsequence,
// gives what rocRAND's Philox engine gives on the host from the same seed.
//
// Built ahead of time by build.sh, from hipRAND's documented API.
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr size_t kCount = 4096;
int ok = 0, total = 0;
void check(const char* what, bool good) {
  ++total;
  ok += good;
  if (!good) std::printf("wrong: %s\n", what);
}

struct Pair {
  hiprandGenerator_t device = nullptr, host = nullptr;
  bool made = false;
  Pair(hiprandRngType_t type, unsigned dims = 0) {
    made = hiprandCreateGenerator(&device, type) == HIPRAND_STATUS_SUCCESS &&
           hiprandCreateGeneratorHost(&host, type) == HIPRAND_STATUS_SUCCESS;
    if (made && dims) {
      made = hiprandSetQuasiRandomGeneratorDimensions(device, dims) == HIPRAND_STATUS_SUCCESS &&
             hiprandSetQuasiRandomGeneratorDimensions(host, dims) == HIPRAND_STATUS_SUCCESS;
    } else if (made) {
      made = hiprandSetPseudoRandomGeneratorSeed(device, 1234) == HIPRAND_STATUS_SUCCESS &&
             hiprandSetPseudoRandomGeneratorSeed(host, 1234) == HIPRAND_STATUS_SUCCESS;
    }
  }
  ~Pair() {
    if (device) hiprandDestroyGenerator(device);
    if (host) hiprandDestroyGenerator(host);
  }
  // Fills `count` of T on the device and on the host with `gen`.
  template <class T, class Gen>
  bool both(Gen gen, std::vector<T>& d, std::vector<T>& h) {
    d.assign(kCount, T{});
    h.assign(kCount, T{});
    T* buf;
    if (!made || hipMalloc(&buf, kCount * sizeof(T)) != hipSuccess) return false;
    const bool good = gen(device, buf) == HIPRAND_STATUS_SUCCESS && gen(host, h.data()) == HIPRAND_STATUS_SUCCESS &&
                      hipMemcpy(d.data(), buf, kCount * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess;
    (void)hipFree(buf);
    return good;
  }
};

template <class T>
bool same(const std::vector<T>& a, const std::vector<T>& b) {
  return a == b;
}
template <class T>
bool near(const std::vector<T>& a, const std::vector<T>& b, double rel) {
  for (size_t i = 0; i < a.size(); ++i)
    if (!(std::abs(double(a[i]) - double(b[i])) <= rel * std::max(1.0, std::abs(double(b[i]))))) return false;
  return true;
}

// One generator of `type` on each side, seeded alike, drawn from in the same
// order: each draw continues both sequences in step. (One pair, not one a
// draw: XORWOW seeds a table of states with skip-ahead each time one is made.)
void pseudo(hiprandRngType_t type, const char* name) {
  Pair p(type);
  char what[96];
  {
    std::vector<unsigned> d, h;
    const bool made = p.both<unsigned>([](hiprandGenerator_t g, unsigned* o) { return hiprandGenerate(g, o, kCount); }, d, h);
    std::snprintf(what, sizeof what, "%s: integers", name);
    check(what, made && same(d, h));
  }
  {
    std::vector<float> d, h;
    const bool made =
        p.both<float>([](hiprandGenerator_t g, float* o) { return hiprandGenerateUniform(g, o, kCount); }, d, h);
    std::snprintf(what, sizeof what, "%s: uniform floats", name);
    check(what, made && same(d, h));
  }
  {
    std::vector<float> d, h;
    const bool made = p.both<float>(
        [](hiprandGenerator_t g, float* o) { return hiprandGenerateNormal(g, o, kCount, 1.0f, 2.0f); }, d, h);
    std::snprintf(what, sizeof what, "%s: normal floats", name);
    check(what, made && near(d, h, 1e-5));
  }
  {
    std::vector<double> d, h;
    const bool made = p.both<double>(
        [](hiprandGenerator_t g, double* o) { return hiprandGenerateLogNormalDouble(g, o, kCount, 0.0, 0.5); }, d, h);
    std::snprintf(what, sizeof what, "%s: log-normal doubles", name);
    check(what, made && near(d, h, 1e-12));
  }
}

__global__ void philox_kernel(unsigned* out, float* uniform) {
  const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  hiprandStatePhilox4_32_10_t s;
  hiprand_init(42, t, 0, &s);
  out[2 * t] = hiprand(&s);
  out[2 * t + 1] = hiprand(&s);
  uniform[t] = hiprand_uniform(&s);
}

}  // namespace

int main() {
  pseudo(HIPRAND_RNG_PSEUDO_PHILOX4_32_10, "PHILOX4_32_10");
  pseudo(HIPRAND_RNG_PSEUDO_XORWOW, "XORWOW");
  pseudo(HIPRAND_RNG_PSEUDO_MRG32K3A, "MRG32K3A");
  pseudo(HIPRAND_RNG_PSEUDO_MTGP32, "MTGP32");
  {
    Pair p(HIPRAND_RNG_QUASI_SOBOL32, 2);
    std::vector<unsigned> d, h;
    const bool made = p.both<unsigned>([](hiprandGenerator_t g, unsigned* o) { return hiprandGenerate(g, o, kCount); }, d, h);
    check("SOBOL32: integers in two dimensions", made && same(d, h));
  }
  {
    Pair p(HIPRAND_RNG_PSEUDO_PHILOX4_32_10);
    std::vector<unsigned> d, h;
    const bool made = p.both<unsigned>(
        [](hiprandGenerator_t g, unsigned* o) { return hiprandGeneratePoisson(g, o, kCount, 7.5); }, d, h);
    check("PHILOX4_32_10: Poisson integers", made && same(d, h));
  }
  {
    constexpr unsigned kThreads = 256;
    unsigned* d_out;
    float* d_uni;
    std::vector<unsigned> out(2 * kThreads);
    std::vector<float> uni(kThreads);
    bool good = hipMalloc(&d_out, out.size() * 4) == hipSuccess && hipMalloc(&d_uni, uni.size() * 4) == hipSuccess;
    if (good) {
      philox_kernel<<<kThreads / 64, 64>>>(d_out, d_uni);
      good = hipMemcpy(out.data(), d_out, out.size() * 4, hipMemcpyDeviceToHost) == hipSuccess &&
             hipMemcpy(uni.data(), d_uni, uni.size() * 4, hipMemcpyDeviceToHost) == hipSuccess;
    }
    // On the host, the engine hiprand_init sets up (rocRAND's, which runs
    // on either side), and the conversion hiprand_uniform makes.
    for (unsigned t = 0; t < kThreads && good; ++t) {
      rocrand_device::philox4x32_10_engine e(42, t, 0);
      const unsigned a = e(), b = e();
      const float u = rocrand_device::detail::uniform_distribution(e());
      good = out[2 * t] == a && out[2 * t + 1] == b && uni[t] == u;
    }
    check("the device API's Philox, thread by thread", good);
  }
  std::printf("hipRAND: %d of %d sequences match the host's\n", ok, total);
  return ok == total ? 0 : 1;
}
