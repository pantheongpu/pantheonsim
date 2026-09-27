// The runtime API and the driver API in one program, on one machine: memory
// either one allocates is memory the other can fill, copy and hand a kernel,
// and both count the same devices. libcudart and libcuda each carry the
// simulator's core, and each used to make its own machine -- its own device
// memory at the same addresses -- so CUTLASS's cuMemsetD32Async on a
// cudaMalloc'd workspace was refused as "not inside any device allocation".
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

__global__ void count_up(unsigned* c, int reps) {
  for (int i = 0; i < reps; ++i) atomicAdd(c, 1u);
}

__global__ void add_one(unsigned* p, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1;
}

// The same kernel as PTX, for the driver API to load.
static const char* kPtx = R"(
.version 7.0
.target sm_75
.address_size 64
.visible .entry twice(.param .u64 p, .param .u32 n)
{
  .reg .pred %q;
  .reg .b32 %r<5>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r1, [n];
  mov.u32 %r2, %tid.x;
  setp.ge.u32 %q, %r2, %r1;
  @%q bra done;
  mul.wide.u32 %rd2, %r2, 4;
  add.u64 %rd3, %rd1, %rd2;
  ld.global.u32 %r3, [%rd3];
  add.u32 %r4, %r3, %r3;
  st.global.u32 [%rd3], %r4;
done:
  ret;
}
.visible .entry count_up(.param .u64 c, .param .u32 reps)
{
  .reg .pred %q;
  .reg .b32 %r<4>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [c];
  ld.param.u32 %r1, [reps];
  mov.u32 %r2, 0;
loop:
  setp.ge.u32 %q, %r2, %r1;
  @%q bra out;
  atom.global.add.u32 %r3, [%rd1], 1;
  add.u32 %r2, %r2, 1;
  bra loop;
out:
  ret;
}
)";

static int fails = 0;
static void expect(const char* what, bool ok) {
  if (!ok) { std::printf("FAIL %s\n", what); ++fails; }
}

int main() {
  const int n = 64;
  int rt_count = 0, drv_count = 0;
  cudaGetDeviceCount(&rt_count);
  cuInit(0);
  cuDeviceGetCount(&drv_count);
  expect("both APIs count the same devices", rt_count == drv_count && rt_count > 0);

  // Runtime memory, filled by the driver, updated by a runtime kernel, and
  // doubled by a driver-launched kernel.
  unsigned* a = nullptr;
  expect("cudaMalloc", cudaMalloc(&a, n * sizeof(unsigned)) == cudaSuccess);
  expect("cuMemsetD32 on cudaMalloc memory", cuMemsetD32((CUdeviceptr)a, 20, n) == CUDA_SUCCESS);
  add_one<<<1, n>>>(a, n);
  expect("a runtime kernel on it", cudaDeviceSynchronize() == cudaSuccess);
  CUmodule mod = nullptr;
  CUfunction fn = nullptr;
  expect("cuModuleLoadData", cuModuleLoadData(&mod, kPtx) == CUDA_SUCCESS);
  expect("cuModuleGetFunction", cuModuleGetFunction(&fn, mod, "twice") == CUDA_SUCCESS);
  CUdeviceptr pa = (CUdeviceptr)a;
  unsigned un = n;
  void* args[] = {&pa, &un};
  expect("cuLaunchKernel on cudaMalloc memory", cuLaunchKernel(fn, 1, 1, 1, n, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS);
  expect("cuCtxSynchronize", cuCtxSynchronize() == CUDA_SUCCESS);
  unsigned h[n] = {};
  expect("cuMemcpyDtoH from cudaMalloc memory", cuMemcpyDtoH(h, (CUdeviceptr)a, sizeof h) == CUDA_SUCCESS);
  int wrong = 0;
  for (unsigned v : h) wrong += v != 42;
  expect("filled by the driver, +1 by the runtime, doubled by the driver: 42", wrong == 0);

  // Driver memory, used by the runtime.
  CUdeviceptr b = 0;
  expect("cuMemAlloc", cuMemAlloc(&b, n * sizeof(unsigned)) == CUDA_SUCCESS);
  expect("cudaMemset on cuMemAlloc memory", cudaMemset((void*)b, 0, n * sizeof(unsigned)) == cudaSuccess);
  add_one<<<1, n>>>((unsigned*)b, n);
  expect("cudaMemcpy from it", cudaMemcpy(h, (void*)b, sizeof h, cudaMemcpyDeviceToHost) == cudaSuccess);
  wrong = 0;
  for (unsigned v : h) wrong += v != 1;
  expect("cleared by the runtime, +1 by a runtime kernel: 1", wrong == 0);
  // cuModuleLoad: the same module from a file, as frameworks cache compiled
  // kernels on disk.
  char path[] = "/tmp/vgpu_mixed_apis_XXXXXX";
  const int fd = mkstemp(path);
  expect("a temporary file", fd >= 0);
  if (fd >= 0) {
    expect("the module written", write(fd, kPtx, std::strlen(kPtx)) == (ssize_t)std::strlen(kPtx));
    close(fd);
    CUmodule from_file = nullptr;
    CUfunction twice = nullptr;
    expect("cuModuleLoad from a file", cuModuleLoad(&from_file, path) == CUDA_SUCCESS);
    expect("its kernel", cuModuleGetFunction(&twice, from_file, "twice") == CUDA_SUCCESS);
    CUdeviceptr pb = b;
    void* bargs[] = {&pb, &un};
    expect("launched on driver memory", cuLaunchKernel(twice, 1, 1, 1, n, 1, 1, 0, nullptr, bargs, nullptr) == CUDA_SUCCESS);
    expect("cudaMemcpy of the result", cudaMemcpy(h, (void*)b, sizeof h, cudaMemcpyDeviceToHost) == cudaSuccess);
    wrong = 0;
    for (unsigned v : h) wrong += v != 2;
    expect("doubled by the kernel cuModuleLoad loaded: 2", wrong == 0);
    CUmodule missing = nullptr;
    expect("a file that is not there is CUDA_ERROR_FILE_NOT_FOUND", cuModuleLoad(&missing, "/nonexistent/k.ptx") == CUDA_ERROR_FILE_NOT_FOUND);
    unlink(path);
  }
  expect("cudaFree of runtime memory", cudaFree(a) == cudaSuccess);
  expect("cuMemFree of driver memory", cuMemFree(b) == CUDA_SUCCESS);

  // Two threads, one launching through the runtime and one through the
  // driver, loading and counting on the same word of the one machine: every
  // increment arrives. (A correctness check of mixed use from two threads.
  // The libraries also share one API lock -- shared_runtime.cpp -- so their
  // calls into the machine never overlap; this does not show that on its own,
  // since kernels on one device do not run at the same time here anyway.)
  {
    unsigned* counter = nullptr;
    cudaMalloc(&counter, sizeof(unsigned));
    cudaMemset(counter, 0, sizeof(unsigned));
    const int rounds = 10, blocks = 64, threads = 32, reps = 40;
    std::thread by_runtime([&] {
      for (int r = 0; r < rounds; ++r) {
        count_up<<<blocks, threads>>>(counter, reps);
        cudaDeviceSynchronize();
      }
    });
    std::thread by_driver([&] {
      CUcontext ctx = nullptr;
      cuDevicePrimaryCtxRetain(&ctx, 0);
      cuCtxSetCurrent(ctx);
      CUfunction count = nullptr;
      cuModuleGetFunction(&count, mod, "count_up");
      CUdeviceptr pc = (CUdeviceptr)counter;
      int rp = reps;
      void* cargs[] = {&pc, &rp};
      for (int r = 0; r < rounds; ++r) {
        cuLaunchKernel(count, blocks, 1, 1, threads, 1, 1, 0, nullptr, cargs, nullptr);
        cuCtxSynchronize();
      }
      cuDevicePrimaryCtxRelease(0);
    });
    by_runtime.join();
    by_driver.join();
    unsigned total = 0;
    cudaMemcpy(&total, counter, sizeof total, cudaMemcpyDeviceToHost);
    expect("atomics from runtime and driver launches on two threads all arrive",
           total == 2u * rounds * blocks * threads * reps);
    cudaFree(counter);
  }

  std::printf(fails ? "FAIL\n" : "PASS\n");
  return fails != 0;
}
