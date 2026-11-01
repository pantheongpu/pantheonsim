// The program run_host_gpu_stub.sh runs with a stand-in driver (host_gpu_stub_driver.c): `big` launches a kernel of the exact set (integer arithmetic) on a grid
// worth a card, `small` the same kernel on a few warps, `mufu` a kernel outside the set (expf). It prints the results on stdout, which must not depend on the
// host-GPU mode, and on stderr how long the first launch took in milliseconds ("first_launch_ms=").
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

__global__ void scramble(uint32_t* d, int n, int rounds) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t x = d[i] ^ (uint32_t(i) * 2654435761u);
  for (int r = 0; r < rounds; ++r) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x += uint32_t(r) * 0x9e3779b9u;
  }
  d[i] = x;
}
__global__ void use_expf(float* d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = expf(d[i] * 0.001f) + float(i);
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "big";
  const bool small = !std::strcmp(mode, "small"), mufu = !std::strcmp(mode, "mufu");
  const int n = small ? 64 : 1 << 18;
  void* d = nullptr;
  cudaMalloc(&d, n * 4);
  cudaMemset(d, 1, n * 4);
  uint64_t h = 0xcbf29ce484222325ull;
  for (int launch = 0; launch < 4; ++launch) {
    const auto t0 = std::chrono::steady_clock::now();
    if (mufu) use_expf<<<(n + 255) / 256, 256>>>(static_cast<float*>(d), n);
    else scramble<<<(n + 255) / 256, 256>>>(static_cast<uint32_t*>(d), n, 64);
    cudaDeviceSynchronize();
    if (launch == 0) std::fprintf(stderr, "first_launch_ms=%.0f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    std::vector<unsigned char> back(n * 4);
    cudaMemcpy(back.data(), d, n * 4, cudaMemcpyDeviceToHost);
    for (unsigned char c : back) h = (h ^ c) * 0x100000001b3ull;
    std::printf("launch %d: %016llx\n", launch, static_cast<unsigned long long>(h));
  }
  cudaFree(d);
  return 0;
}
