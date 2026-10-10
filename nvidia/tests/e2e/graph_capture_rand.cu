// cuRAND's generate calls inside a captured CUDA graph (see graph_capture_common.h): the call is
// recorded and draws numbers each time the graph runs, from where the generator has got to -- so a
// generator that made the numbers eagerly, and one with the same seed that made them in graph launches,
// agree call for call. run_graph_capture.sh rand curand --card runs the same program on NVIDIA's cuRAND.
#include <curand.h>

#include "graph_capture_common.h"

using namespace gc;

#define OK(x) do { curandStatus_t s_ = (x); if (s_ != CURAND_STATUS_SUCCESS) { \
  std::printf("     %s -> %d\n", #x, (int)s_); ok = false; } } while (0)

struct Gen {
  curandGenerator_t g = nullptr;
  Gen(cudaStream_t st, curandRngType_t type, unsigned long long seed) {
    curandCreateGenerator(&g, type);
    curandSetPseudoRandomGeneratorSeed(g, seed);
    curandSetStream(g, st);
  }
  ~Gen() { curandDestroyGenerator(g); }
};

// One kind of generate call, as a captured generator `a` and a reference generator `b` with the same seed.
// What NVIDIA's cuRAND does (measured on an RTX 3060): the capture takes its place in the stream, so an
// eager call after the graph carries on after it; the first launch draws exactly what an eager call
// would have at the capture; later launches of a pseudorandom generator draw other numbers, and those of
// a quasirandom one the same points again.
template <class T, class Call>
static void case_of(Runner& r, const char* name, curandRngType_t type, size_t n, Dt dt, Call call, bool quasi = false) {
  Gen a(r.st, type, 1234), b(r.st, type, 1234);
  T* out = r.alloc<T>(n);
  T* want = r.alloc<T>(n);
  T* later = r.alloc<T>(n);
  int made = 0;   // how many times the reference has drawn
  std::vector<std::vector<unsigned char>> drawn;   // what each draw of the reference was
  auto reference = [&](int index) {
    while (made <= index) {
      cudaMemset(want, 0, n * sizeof(T));
      call(b.g, want, n);
      cudaStreamSynchronize(r.st);
      drawn.push_back(Runner::read({want, n, dt}));
      ++made;
    }
    return drawn[index];
  };
  std::vector<std::vector<unsigned char>> seen;   // the launches' results
  r.run(name, [&] {
    bool ok = true;
    OK(call(a.g, out, n));
    return ok;
  }, {{out, n, dt}}, 0, [&](int, int launch) {
    const auto got = Runner::read({out, n, dt});
    bool good = true;
    if (launch == 1) good = got == reference(1);   // the eager call was draw 0, the capture took draw 1
    else if (quasi) good = got == seen.back();
    else
      for (const auto& earlier : seen) good = good && got != earlier;
    seen.push_back(got);
    if (launch == 2) {
      // An eager call after the launches carries on after the capture, not after the launches.
      cudaMemset(later, 0, n * sizeof(T));
      call(a.g, later, n);
      cudaStreamSynchronize(r.st);
      good = good && Runner::read({later, n, dt}) == reference(2);
    }
    return good;
  });
}

int main() {
  Runner r;
  for (const auto type : {CURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_PHILOX4_32_10, CURAND_RNG_PSEUDO_MRG32K3A}) {
    const char* tn = type == CURAND_RNG_PSEUDO_XORWOW ? "XORWOW" : type == CURAND_RNG_PSEUDO_PHILOX4_32_10 ? "Philox" : "MRG32k3a";
    case_of<unsigned int>(r, (std::string("curandGenerate (") + tn + ")").c_str(), type, 1000, Dt::I32,
                          [](curandGenerator_t g, unsigned int* o, size_t n) { return curandGenerate(g, o, n); });
    case_of<float>(r, (std::string("curandGenerateUniform (") + tn + ")").c_str(), type, 1000, Dt::F32,
                   [](curandGenerator_t g, float* o, size_t n) { return curandGenerateUniform(g, o, n); });
    case_of<double>(r, (std::string("curandGenerateNormalDouble (") + tn + ")").c_str(), type, 1000, Dt::F64,
                    [](curandGenerator_t g, double* o, size_t n) { return curandGenerateNormalDouble(g, o, n, 1.0, 2.0); });
  }
  case_of<float>(r, "curandGenerateLogNormal (XORWOW)", CURAND_RNG_PSEUDO_XORWOW, 1000, Dt::F32,
                 [](curandGenerator_t g, float* o, size_t n) { return curandGenerateLogNormal(g, o, n, 0.0f, 0.5f); });
  case_of<float>(r, "curandGenerateNormal (Sobol')", CURAND_RNG_QUASI_SOBOL32, 1000, Dt::F32,
                 [](curandGenerator_t g, float* o, size_t n) { return curandGenerateNormal(g, o, n, 0.0f, 1.0f); }, true);
  return finish();
}
