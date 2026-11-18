// Data-parallel kernels on several simulated devices, for the multi-card measurements of docs/perf-design/12-host-gpu-turbo.md section 17.
//   ./multibench <devices> <pattern 0|1|2> <n> <iters> <launches>
// Each device holds n floats, runs `launches` launches of a kernel that does `iters` fused multiply-adds per element (no libm, so a card and the simulator agree
// bit for bit), reads them back and prints a checksum per device. pattern 0: one thread launches on every device, then reads them all back (the usual loop);
// 1: one thread per device, meeting at a barrier after their launches (a data-parallel step); 2: one thread per device, each reading back as soon as it has launched.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

__global__ void work(float* x, int n, int iters, float a) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  for (int k = 0; k < iters; ++k) {
    const float t = v * 0.0001f;
    v = fmaf(v, a + t, static_cast<float>(k & 3) * 0.125f);   // (not an affine recurrence, no libm: a card and the simulator agree bit for bit)
  }
  x[i] = v;
}

struct Dev {
  float* p = nullptr;
  std::vector<float> h;
};

static void launch_on(int d, int n, int iters, int launches, Dev* v) {
  cudaSetDevice(d);
  cudaMalloc(&v->p, n * sizeof(float));
  v->h.resize(n);
  for (int i = 0; i < n; ++i) v->h[i] = static_cast<float>((i * 7 + d * 13) % 101) * 0.01f;
  cudaMemcpy(v->p, v->h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  for (int l = 0; l < launches; ++l) work<<<(n + 255) / 256, 256>>>(v->p, n, iters, 0.99f);
}

static void read_on(int d, int n, Dev* v, double* out) {
  cudaSetDevice(d);
  cudaDeviceSynchronize();
  cudaMemcpy(v->h.data(), v->p, n * sizeof(float), cudaMemcpyDeviceToHost);
  double s = 0;
  for (int i = 0; i < n; ++i) s += v->h[i];
  *out = s;
  cudaFree(v->p);
}

struct Barrier {   // (C++17 has none)
  std::mutex mu;
  std::condition_variable cv;
  int need = 0, have = 0;
  void wait() {
    std::unique_lock<std::mutex> lk(mu);
    if (++have == need) cv.notify_all();
    else cv.wait(lk, [&] { return have == need; });
  }
};

int main(int argc, char** argv) {
  const int devs = argc > 1 ? std::atoi(argv[1]) : 2, th = argc > 2 ? std::atoi(argv[2]) : 0, n = argc > 3 ? std::atoi(argv[3]) : (1 << 18);
  const int iters = argc > 4 ? std::atoi(argv[4]) : 128, launches = argc > 5 ? std::atoi(argv[5]) : 4;
  int have = 0;
  cudaGetDeviceCount(&have);
  const int use = devs < have ? devs : have;
  std::vector<double> sums(use, 0);
  std::vector<Dev> dv(use);
  if (th == 0) {   // one thread: every device's launches, then every read-back
    for (int d = 0; d < use; ++d) launch_on(d, n, iters, launches, &dv[d]);
    for (int d = 0; d < use; ++d) read_on(d, n, &dv[d], &sums[d]);
  } else {         // one thread per device; th=1: they meet after their launches (a training step's barrier), th=2: each reads back as soon as it has launched
    Barrier bar;
    bar.need = use;
    std::vector<std::thread> t;
    for (int d = 0; d < use; ++d)
      t.emplace_back([&, d] {
        launch_on(d, n, iters, launches, &dv[d]);
        if (th == 1) bar.wait();
        read_on(d, n, &dv[d], &sums[d]);
      });
    for (auto& x : t) x.join();
  }
  for (int d = 0; d < use; ++d) std::printf("device %d checksum %.9g\n", d, sums[d]);
  return 0;
}
