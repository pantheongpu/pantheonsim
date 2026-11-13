// Round sp66 (docs/perf-design/12-host-gpu-turbo.md section 2, the call class): built with -rdc=true, together with host_gpu_rdc_fn.cu, so that
// `mix` is a device function of its own section and the kernel calls it with an absolute call (CALL.ABS.NOINC, a relocation the driver links).
// Turbo mode classes the callee's code with the kernel's and sends the kernel to the card; exact mode does not (CALL is outside its set).
// Integer arithmetic only: the same bits on the card and in the simulator. Prints one hash per launch.
#include <cstdint>
#include <cstdio>
#include <vector>

__device__ uint32_t mix(uint32_t x, int rounds);   // host_gpu_rdc_fn.cu

__global__ void calls_out(uint32_t* d, int n, int rounds) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = mix(d[i] ^ (uint32_t(i) * 2654435761u), rounds);
}

int main() {
  const int n = 1 << 18;
  uint32_t* d = nullptr;
  cudaMalloc(&d, n * 4);
  cudaMemset(d, 3, n * 4);
  std::vector<uint32_t> back(n);
  for (int launch = 0; launch < 4; ++launch) {
    calls_out<<<n / 256, 256>>>(d, n, 64);
    cudaMemcpy(back.data(), d, n * 4, cudaMemcpyDeviceToHost);
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint32_t v : back) h = (h ^ v) * 0x100000001b3ull;
    std::printf("rdc %d: %016llx\n", launch, (unsigned long long)h);
  }
  cudaFree(d);
  return 0;
}
