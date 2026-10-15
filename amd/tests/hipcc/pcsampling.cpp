// A kernel that runs long enough to be PC sampled at an interval of ten
// thousand instructions (the interval AMD's rocprofiler-sdk PC sampling sample
// asks for), and checks its own answers. run_rocprofiler_pcsampling.sh runs it
// under that sample's tool.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

__global__ void spin(unsigned* out, int rounds) {
  unsigned a = threadIdx.x + 1;
  for (int i = 0; i < rounds; ++i) a = a * 3u + 1u;
  out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

int main() {
  const int groups = 2, threads = 128, rounds = 20000;
  unsigned* d = nullptr;
  if (hipMalloc(&d, groups * threads * sizeof(unsigned)) != hipSuccess) return 1;
  spin<<<groups, threads>>>(d, rounds);
  std::vector<unsigned> out(groups * threads);
  if (hipMemcpy(out.data(), d, out.size() * sizeof(unsigned), hipMemcpyDeviceToHost) != hipSuccess) return 1;
  int wrong = 0;
  for (int g = 0; g < groups; ++g)
    for (int t = 0; t < threads; ++t) {
      unsigned a = t + 1;
      for (int i = 0; i < rounds; ++i) a = a * 3u + 1u;
      wrong += out[g * threads + t] != a;
    }
  std::printf("spin wrong %d of %d\n", wrong, groups * threads);
  (void)hipFree(d);
  return wrong != 0;
}
