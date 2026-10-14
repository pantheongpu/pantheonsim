// What the runtime remembers between launches must never show: the same kernel launched thousands of times,
// with modules loaded and unloaded through the driver API in between (one of them with a kernel of the same
// name), across cudaDeviceReset (the kernels' module is loaded again, under another id), and on two devices at
// once; and the errors a bad launch gets, the same before and after the kernel has been launched. Every check is
// against values the kernels must write, so the program also passes against NVIDIA's libraries on a card.
// Run by run_launch_cache.sh, which also runs it with VGPU_LAUNCH_CACHE=0 and requires the same output.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace {
int g_failures = 0;
#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::printf("FAIL: %s (line %d)\n", #cond, __LINE__);                      \
      ++g_failures;                                                              \
    }                                                                            \
  } while (0)

constexpr int kN = 64;
}  // namespace

// Three kernels with different arithmetic, so a launch that ran another's code is seen in the numbers. (Global
// scope: the driver module below names ka by its mangled name, _Z2kaPii.)
__global__ void ka(int* p, int v) {
  const int i = threadIdx.x;
  if (i < kN) p[i] += v;
}
__global__ void kb(int* p, int v) {
  const int i = threadIdx.x;
  if (i < kN) p[i] ^= v;
}
__global__ void kc(int* p, int v) {
  const int i = threadIdx.x;
  if (i < kN) p[i] -= v;
}

namespace {

// A driver-API module with a kernel named like the runtime's `ka`, which writes 1000 + v instead.
// (The mangled name of ka(int*, int) is _Z2kaPii.)
const char* kOtherModule = R"(
.version 7.0
.target sm_80
.address_size 64
.visible .entry _Z2kaPii(.param .u64 p, .param .u32 v)
{
  .reg .pred %q<2>;
  .reg .b32 %r<6>;
  .reg .b64 %rd<5>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r1, [v];
  mov.u32 %r2, %tid.x;
  setp.ge.u32 %q1, %r2, 64;
  @%q1 bra DONE;
  add.u32 %r3, %r1, 1000;
  cvta.to.global.u64 %rd2, %rd1;
  mul.wide.u32 %rd3, %r2, 4;
  add.u64 %rd4, %rd2, %rd3;
  st.global.u32 [%rd4], %r3;
DONE:
  ret;
}
)";

std::vector<int> fetch(const int* d) {
  std::vector<int> h(kN);
  cudaMemcpy(h.data(), d, kN * sizeof(int), cudaMemcpyDeviceToHost);
  return h;
}

bool all_equal(const std::vector<int>& v, int want) {
  for (int x : v)
    if (x != want) return false;
  return true;
}

int* fresh(int value) {
  int* d = nullptr;
  cudaMalloc(&d, kN * sizeof(int));
  std::vector<int> h(kN, value);
  cudaMemcpy(d, h.data(), kN * sizeof(int), cudaMemcpyHostToDevice);
  return d;
}

}  // namespace

int main() {
  // 1. The same kernels, over and over, interleaved: the numbers are the arithmetic of each launch in turn.
  {
    int* d = fresh(0);
    int want = 0;
    for (int round = 0; round < 300; ++round) {
      ka<<<1, kN>>>(d, 3);
      want += 3;
      kb<<<1, kN>>>(d, 5);
      want ^= 5;
      kc<<<1, kN>>>(d, 2);
      want -= 2;
    }
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(all_equal(fetch(d), want));
    cudaFree(d);
  }

  // 2. Modules loaded and unloaded through the driver API between launches of the runtime's kernels, one of them
  //    with a kernel called the same.
  {
    CHECK(cuInit(0) == CUDA_SUCCESS);
    int* d = fresh(0);
    for (int round = 0; round < 20; ++round) {
      ka<<<1, kN>>>(d, 1);
      CUmodule mod;
      CHECK(cuModuleLoadData(&mod, kOtherModule) == CUDA_SUCCESS);
      CUfunction f;
      CHECK(cuModuleGetFunction(&f, mod, "_Z2kaPii") == CUDA_SUCCESS);
      int* e = fresh(0);
      int v = round;
      void* args[] = {&e, &v};
      CHECK(cuLaunchKernel(f, 1, 1, 1, kN, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS);
      CHECK(cudaDeviceSynchronize() == cudaSuccess);
      CHECK(all_equal(fetch(e), 1000 + round));   // the driver module's kernel, not the runtime's
      cudaFree(e);
      ka<<<1, kN>>>(d, 1);
      if (round % 2) CHECK(cuModuleUnload(mod) == CUDA_SUCCESS);   // some stay loaded, some go
      ka<<<1, kN>>>(d, 1);
    }
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(all_equal(fetch(d), 20 * 3));
    cudaFree(d);
  }

  // 3. A device reset unloads everything; the kernels load again on their next launch.
  {
    int* d = fresh(5);
    ka<<<1, kN>>>(d, 1);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(all_equal(fetch(d), 6));
    CHECK(cudaDeviceReset() == cudaSuccess);
    d = fresh(7);
    ka<<<1, kN>>>(d, 1);
    kb<<<1, kN>>>(d, 1);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(all_equal(fetch(d), (7 + 1) ^ 1));
    for (int i = 0; i < 100; ++i) kc<<<1, kN>>>(d, 1);
    CHECK(all_equal(fetch(d), ((7 + 1) ^ 1) - 100));
    cudaFree(d);
  }

  // 4. Two devices: the same kernel launched on each in turn.
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count >= 2) {
    int* d[2];
    for (int dev = 0; dev < 2; ++dev) {
      cudaSetDevice(dev);
      d[dev] = fresh(dev * 100);
    }
    for (int round = 0; round < 50; ++round)
      for (int dev = 0; dev < 2; ++dev) {
        cudaSetDevice(dev);
        ka<<<1, kN>>>(d[dev], dev + 1);
      }
    for (int dev = 0; dev < 2; ++dev) {
      cudaSetDevice(dev);
      CHECK(cudaDeviceSynchronize() == cudaSuccess);
      CHECK(all_equal(fetch(d[dev]), dev * 100 + 50 * (dev + 1)));
    }
    // Reset one device only; the other keeps what it had loaded.
    cudaSetDevice(1);
    cudaDeviceReset();
    cudaSetDevice(0);
    ka<<<1, kN>>>(d[0], 1);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(all_equal(fetch(d[0]), 50 + 1));
    cudaSetDevice(1);
    int* e = fresh(9);
    ka<<<1, kN>>>(e, 1);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(all_equal(fetch(e), 10));
    cudaSetDevice(0);
    cudaFree(d[0]);
  } else {
    std::printf("one device: the two-device case is skipped\n");
  }
  cudaSetDevice(0);

  // 5. The errors of a bad launch, and the attributes of a kernel, are the same once it has been launched many
  //    times as the first time.
  {
    int* d = fresh(0);
    cudaGetLastError();
    ka<<<1, 0>>>(d, 1);   // no threads
    const cudaError_t first_zero = cudaGetLastError();
    ka<<<1, 4096>>>(d, 1);   // more than a block can hold
    const cudaError_t first_big = cudaGetLastError();
    cudaFuncAttributes attr0{};
    CHECK(cudaFuncGetAttributes(&attr0, ka) == cudaSuccess);
    int blocks0 = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks0, ka, 128, 0) == cudaSuccess);
    for (int i = 0; i < 1000; ++i) ka<<<1, kN>>>(d, 1);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(all_equal(fetch(d), 1000));
    ka<<<1, 0>>>(d, 1);
    CHECK(cudaGetLastError() == first_zero);
    ka<<<1, 4096>>>(d, 1);
    CHECK(cudaGetLastError() == first_big);
    CHECK(first_zero == cudaErrorInvalidConfiguration);
    CHECK(first_big == cudaErrorInvalidConfiguration);
    cudaFuncAttributes attr1{};
    CHECK(cudaFuncGetAttributes(&attr1, ka) == cudaSuccess);
    CHECK(attr0.numRegs == attr1.numRegs && attr0.sharedSizeBytes == attr1.sharedSizeBytes &&
          attr0.maxThreadsPerBlock == attr1.maxThreadsPerBlock && attr0.ptxVersion == attr1.ptxVersion);
    int blocks1 = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks1, ka, 128, 0) == cudaSuccess);
    CHECK(blocks0 == blocks1);
    cudaFree(d);
  }

  if (g_failures == 0) std::printf("PASS\n");
  else std::printf("FAIL (%d)\n", g_failures);
  return g_failures != 0;
}
