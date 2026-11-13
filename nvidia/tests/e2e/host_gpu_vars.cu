// Round sp66 (docs/perf-design/12-host-gpu-turbo.md section 3): what a kernel's module holds besides code no longer keeps the kernel in the simulator
// by itself. One module (this translation unit) with __device__ and __constant__ variables, string literals, vprintf, malloc/free and assert, and a
// texture object on the device. run_host_gpu_stub.sh runs it against the stand-in driver, run_host_gpu.sh on a card; its stdout must not depend on the
// host-GPU mode (with `printf`, the card's lines are the same text: what is printed is integers).
//   vars     `with_vars` reads __constant__ coefficients (c[3][...]) and a __constant__ int, and reads and writes a __device__ array by name; the host
//            sets the constants with cudaMemcpyToSymbol between launches (the card must see each new value), takes the array's address with
//            cudaGetSymbolAddress and passes it to `through_pointer`, which reads and writes the same array through it, and reads it back with
//            cudaMemcpyFromSymbol. Every value is a small integer held in a float: the same bits on the card and in the simulator.
//   printf   `prints` does the work of `with_vars`'s kind and has one thread print a line per launch, between lines the host prints: the order of the
//            lines must be the mode-off order with stdout a file (fully buffered) or a terminal.
//   tex      a texture object exists while `plain` runs (until Round sp66 that kept every launch off the card), then `fetch` reads through it (a
//            texture instruction: it stays in the simulator).
//   heap     `uses_heap` calls malloc and free: it stays in the simulator in both modes, and nothing reaches the driver.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr int kN = 1 << 18;
__device__ float dev_acc[kN];
__constant__ float coef[16];
__constant__ int cshift;

__global__ void with_vars(float* out, int n, int rounds) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = dev_acc[i];
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) x = float((int(x) * 7 + int(coef[(i + r) & 15]) + cshift) & 1023);
  dev_acc[i] = x;
  out[i] = x + coef[i & 15];
}
__global__ void through_pointer(float* acc, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] += acc[i];
  acc[i] = float((int(acc[i]) * 3 + 1) & 1023);
}
__global__ void prints(float* out, int n, int launch, int rounds) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = out[i];
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) x = float((int(x) * 5 + r) & 1023);
  out[i] = x;
  if (i == 0) printf("kernel line of launch %d: n=%d\n", launch, n);
}
__global__ void plain(uint32_t* d, int n, int rounds) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t x = d[i] ^ (uint32_t(i) * 2654435761u);
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
  }
  d[i] = x;
}
__global__ void fetch(cudaTextureObject_t t, uint32_t* d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] += tex1Dfetch<unsigned>(t, i);
}
__global__ void uses_heap(uint32_t* d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t* p = static_cast<uint32_t*>(malloc(4));
  assert(p != nullptr);
  *p = d[i] * 3u;
  d[i] = *p + 1u;
  free(p);
}

static uint64_t fnv(const void* data, size_t bytes) {
  uint64_t h = 0xcbf29ce484222325ull;
  const unsigned char* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "vars";
  const int n = kN, blocks = n / 256;
  if (!std::strcmp(mode, "vars")) {
    std::vector<float> init(n), back(n), out(n);
    for (int i = 0; i < n; ++i) init[i] = float(i & 1023);
    cudaMemcpyToSymbol(dev_acc, init.data(), n * sizeof(float));
    float* dout = nullptr;
    cudaMalloc(&dout, n * sizeof(float));
    float* acc = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&acc), dev_acc);
    for (int launch = 0; launch < 6; ++launch) {
      float c[16];
      for (int k = 0; k < 16; ++k) c[k] = float((k * 37 + launch * 11) & 63);
      const int shift = launch * 5 + 1;
      cudaMemcpyToSymbol(coef, c, sizeof c);   // new constants before every launch: the card has to see each
      cudaMemcpyToSymbol(cshift, &shift, sizeof shift);
      with_vars<<<blocks, 256>>>(dout, n, 64);
      through_pointer<<<blocks, 256>>>(acc, dout, n);
      cudaMemcpy(out.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpyFromSymbol(back.data(), dev_acc, n * sizeof(float));
      std::printf("vars %d: out %016llx acc %016llx\n", launch, (unsigned long long)fnv(out.data(), n * 4), (unsigned long long)fnv(back.data(), n * 4));
    }
    cudaFree(dout);
  } else if (!std::strcmp(mode, "printf")) {
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));
    for (int launch = 0; launch < 6; ++launch) {
      std::printf("host line before launch %d\n", launch);
      prints<<<blocks, 256>>>(d, n, launch, 64);
      cudaDeviceSynchronize();
      std::printf("host line after launch %d\n", launch);
    }
    std::vector<float> back(n);
    cudaMemcpy(back.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("printf data %016llx\n", (unsigned long long)fnv(back.data(), n * 4));
    cudaFree(d);
  } else if (!std::strcmp(mode, "tex") || !std::strcmp(mode, "heap")) {
    uint32_t* d = nullptr;
    cudaMalloc(&d, n * 4);
    cudaMemset(d, 1, n * 4);
    cudaTextureObject_t t = 0;
    const bool tex = !std::strcmp(mode, "tex");
    if (tex) {
      cudaResourceDesc rd = {};
      rd.resType = cudaResourceTypeLinear;
      rd.res.linear.devPtr = d;
      rd.res.linear.desc = cudaCreateChannelDesc<unsigned>();
      rd.res.linear.sizeInBytes = n * 4;
      cudaTextureDesc td = {};
      td.readMode = cudaReadModeElementType;
      if (cudaCreateTextureObject(&t, &rd, &td, nullptr) != cudaSuccess) { std::printf("no texture object\n"); return 1; }
    }
    std::vector<uint32_t> back(n);
    for (int launch = 0; launch < 4; ++launch) {
      if (tex) plain<<<blocks, 256>>>(d, n, 64);
      else uses_heap<<<blocks, 256>>>(d, n);
      cudaMemcpy(back.data(), d, n * 4, cudaMemcpyDeviceToHost);
      std::printf("%s %d: %016llx\n", mode, launch, (unsigned long long)fnv(back.data(), n * 4));
    }
    if (tex) {
      uint32_t* e = nullptr;
      cudaMalloc(&e, n * 4);
      cudaMemset(e, 0, n * 4);
      fetch<<<blocks, 256>>>(t, e, n);
      cudaMemcpy(back.data(), e, n * 4, cudaMemcpyDeviceToHost);
      std::printf("fetch: %016llx\n", (unsigned long long)fnv(back.data(), n * 4));
      cudaDestroyTextureObject(t);
      cudaFree(e);
    }
    cudaFree(d);
  } else {
    std::printf("unknown mode %s\n", mode);
    return 2;
  }
  return 0;
}
