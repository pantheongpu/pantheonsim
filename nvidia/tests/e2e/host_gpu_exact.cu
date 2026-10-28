// The program run_host_gpu.sh runs with VGPU_HOST_GPU=exact (docs/perf-design/09-host-gpu-spike.md): kernels that may run on a real card
// with the simulator's bits, and ones that must stay in the simulator. It prints a checksum of each kernel's results, so a run with the
// mode on and a run with it off have to print the same lines.
#include <cstdint>
#include <cstdio>
#include <vector>

__global__ void scramble(uint32_t* d, int n, int rounds) {   // integer arithmetic only: IMAD, LOP3, SHF, IADD3
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

__global__ void fma_chain(const float* a, float* b, int n, int rounds) {   // IEEE FFMA/FMUL/FADD
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i], y = 1.0f;
  for (int r = 0; r < rounds; ++r) {
    y = fmaf(y, 0.999f, x);
    x = x * 1.0001f + 0.25f;
  }
  b[i] = y + x;
}

__global__ void use_expf(const float* a, float* b, int n) {   // MUFU.EX2: not in the exact set, stays in the simulator
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = expf(a[i] * 0.001f);
}

__global__ void tiny(int* p) { p[threadIdx.x] += 1; }   // one warp, a few instructions: too small to be worth the card

// host_gpu_exact_mod.cu: a kernel whose module has a __device__ variable (its value lives in the simulator's module memory): refused.
void launch_uses_global(int* p);

static uint64_t fnv(const void* data, size_t bytes) {
  uint64_t h = 0xcbf29ce484222325ull;
  const unsigned char* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}

int main() {
  const int n = 1 << 20;
  std::vector<uint32_t> hi(n);
  std::vector<float> hf(n), out(n);
  for (int i = 0; i < n; ++i) {
    hi[i] = uint32_t(i) * 747796405u + 2891336453u;
    hf[i] = float((i * 7919) % 1000) * 0.01f + 1.0f;
  }
  uint32_t* di;
  float *da, *db;
  int* dp;
  cudaMalloc(&di, n * 4);
  cudaMalloc(&da, n * 4);
  cudaMalloc(&db, n * 4);
  cudaMalloc(&dp, 4096);
  cudaMemcpy(di, hi.data(), n * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(da, hf.data(), n * 4, cudaMemcpyHostToDevice);
  cudaMemset(dp, 0, 4096);
  for (int pass = 0; pass < 4; ++pass) {   // the first two passes teach the size rule (the simulator runs them), the later ones can be offloaded
    scramble<<<n / 256, 256>>>(di, n, 64);
    fma_chain<<<n / 256, 256>>>(da, db, n, 64);
    use_expf<<<n / 256, 256>>>(da, db, n);
    tiny<<<1, 32>>>(dp);
    launch_uses_global(dp);
    cudaDeviceSynchronize();
    cudaMemcpy(hi.data(), di, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.data(), db, n * 4, cudaMemcpyDeviceToHost);
    int small[32];
    cudaMemcpy(small, dp, sizeof small, cudaMemcpyDeviceToHost);
    std::printf("pass %d scramble %016llx expf %016llx tiny %016llx\n", pass, (unsigned long long)fnv(hi.data(), n * 4),
                (unsigned long long)fnv(out.data(), n * 4), (unsigned long long)fnv(small, sizeof small));
  }
  // fma_chain's result, read before use_expf overwrote it, is checked through a second buffer.
  fma_chain<<<n / 256, 256>>>(da, db, n, 64);
  cudaMemcpy(out.data(), db, n * 4, cudaMemcpyDeviceToHost);
  std::printf("fma_chain %016llx\n", (unsigned long long)fnv(out.data(), n * 4));
  const cudaError_t e = cudaGetLastError();
  std::printf("%s\n", e == cudaSuccess ? "done" : cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}
