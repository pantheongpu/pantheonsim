// Atomics as a card makes them: a flat atomic that lands in LDS, a kernel's
// system-scope atomics and a host thread's on the same pinned counter all
// landing, and a float max through its compare-and-swap loop. Each check
// prints "ok <what>" or "FAIL <what>: <why>", and the last line counts them.
// Built by build.sh with hipcc; run by amd/tests/e2e/run_hipcc.sh.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

static int checks = 0, failures = 0;
static void check(bool ok, const char* what, const std::string& why = "") {
  ++checks;
  if (ok) {
    std::printf("ok    %s\n", what);
  } else {
    ++failures;
    std::printf("FAIL  %s%s%s\n", what, why.empty() ? "" : ": ", why.c_str());
  }
}

// The pointer is LDS or global memory as the launch says, so the compiler
// cannot tell which: the add is a flat atomic, and the address says where it
// lands.
__global__ void flat_into_lds(int* out, int* elsewhere, int use_lds) {
  __shared__ int total;
  if (threadIdx.x == 0) total = 0;
  __syncthreads();
  int* p = use_lds ? &total : elsewhere;
  atomicAdd(p, 1);
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = total;
}

__global__ void count_system(int* counter) { atomicAdd_system(counter, 1); }

__global__ void max_of(float* mem, float* old) {
  old[blockIdx.x * blockDim.x + threadIdx.x] = unsafeAtomicMax(mem, 7.5f);
}

int main() {
  // A flat atomic on an LDS address, every thread of each block once.
  {
    const int blocks = 4, threads = 256;
    int* out = nullptr;
    (void)hipMalloc(&out, blocks * sizeof(int));
    hipLaunchKernelGGL(flat_into_lds, dim3(blocks), dim3(threads), 0, nullptr, out, out, 1);
    const hipError_t e = hipDeviceSynchronize();
    std::vector<int> host(blocks, 0);
    (void)hipMemcpy(host.data(), out, blocks * sizeof(int), hipMemcpyDeviceToHost);
    check(e == hipSuccess && host[0] == threads && host[blocks - 1] == threads,
          "a flat atomic lands in LDS", std::string(hipGetErrorName(e)) + ", " + std::to_string(host[0]));
    (void)hipFree(out);
  }

  // A kernel's system-scope atomics and a host thread's, on one pinned
  // counter at the same time: none is lost.
  {
    int* counter = nullptr;
    (void)hipHostMalloc(&counter, sizeof(int));
    *counter = 0;
    const int blocks = 512, threads = 256, host_adds = 200000;
    hipLaunchKernelGGL(count_system, dim3(blocks), dim3(threads), 0, nullptr, counter);
    std::thread host([&] {
      for (int i = 0; i < host_adds; ++i) __atomic_fetch_add(counter, 1, __ATOMIC_RELAXED);
    });
    host.join();
    (void)hipDeviceSynchronize();
    const int want = blocks * threads + host_adds;
    check(*counter == want, "a kernel's and a host thread's atomics on pinned memory all land",
          std::to_string(*counter) + " of " + std::to_string(want));
    (void)hipHostFree(counter);
  }

  // A float max by compare-and-swap: one thread finds the value it started
  // with, and every other the larger one.
  {
    const int blocks = 256, threads = 256, n = blocks * threads;
    float *mem = nullptr, *old = nullptr;
    (void)hipMalloc(&mem, sizeof(float));
    (void)hipMalloc(&old, n * sizeof(float));
    const float init = 5.5f;
    (void)hipMemcpy(mem, &init, sizeof init, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(max_of, dim3(blocks), dim3(threads), 0, nullptr, mem, old);
    std::vector<float> host(n);
    (void)hipMemcpy(host.data(), old, n * sizeof(float), hipMemcpyDeviceToHost);
    int first = 0, larger = 0;
    for (float v : host) {
      first += v == 5.5f;
      larger += v == 7.5f;
    }
    check(first == 1 && larger == n - 1, "unsafeAtomicMax hands the old value to one thread",
          std::to_string(first) + " saw 5.5, " + std::to_string(larger) + " saw 7.5");
    (void)hipFree(mem);
    (void)hipFree(old);
  }

  std::printf("atomics: %d checks, %d failed\n", checks, failures);
  return failures ? 1 : 0;
}
