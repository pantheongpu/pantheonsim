// Data-parallel kernels on several simulated devices, for the multi-card measurements of docs/perf-design/12-host-gpu-turbo.md section 17.
//   ./multibench <devices> <threads 0|1> <n> <iters> <launches>
// Each device holds n floats, runs `launches` launches of a kernel that does `iters` fused multiply-adds per element (no libm, so a card and the simulator agree
// bit for bit), reads them back and prints a checksum per device. threads=1: one host thread per device (the pattern of data-parallel training loops);
// threads=0: one thread visits the devices in turn.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

__global__ void work(float* x, int n, int iters, float a) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  for (int k = 0; k < iters; ++k) v = fmaf(v, a, 0.5f);
  x[i] = v;
}

static void one(int d, int n, int iters, int launches, double* out) {
  cudaSetDevice(d);
  float* p = nullptr;
  cudaMalloc(&p, n * sizeof(float));
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) h[i] = static_cast<float>((i * 7 + d * 13) % 101) * 0.01f;
  cudaMemcpy(p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  for (int l = 0; l < launches; ++l) work<<<(n + 255) / 256, 256>>>(p, n, iters, 0.999f);
  cudaDeviceSynchronize();
  cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost);
  double s = 0;
  for (int i = 0; i < n; ++i) s += h[i];
  *out = s;
  cudaFree(p);
}

int main(int argc, char** argv) {
  const int devs = argc > 1 ? std::atoi(argv[1]) : 2, th = argc > 2 ? std::atoi(argv[2]) : 0, n = argc > 3 ? std::atoi(argv[3]) : (1 << 18);
  const int iters = argc > 4 ? std::atoi(argv[4]) : 128, launches = argc > 5 ? std::atoi(argv[5]) : 4;
  int have = 0;
  cudaGetDeviceCount(&have);
  const int use = devs < have ? devs : have;
  std::vector<double> sums(use, 0);
  if (th) {
    std::vector<std::thread> t;
    for (int d = 0; d < use; ++d) t.emplace_back(one, d, n, iters, launches, &sums[d]);
    for (auto& x : t) x.join();
  } else {
    for (int d = 0; d < use; ++d) one(d, n, iters, launches, &sums[d]);
  }
  for (int d = 0; d < use; ++d) std::printf("device %d checksum %.9g\n", d, sums[d]);
  return 0;
}
