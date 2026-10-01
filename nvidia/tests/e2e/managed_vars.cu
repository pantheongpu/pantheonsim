// __managed__ variables: one variable the host and every device see at one
// address, starting from its initialiser (HeCBench's hungarian-cuda keeps its
// state in them). nvcc's host code reaches each through a pointer the runtime
// sets when the module loads (__cudaRegisterManagedVar, __cudaInitModule), so
// the checks touch the variables from the host before any kernel runs as well
// as after. Every check passes on an RTX 3060 too.
#include <cuda_runtime.h>

#include <cstdio>

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

struct Pair {
  int a;
  double b;
};

__managed__ int counter = 5;
__managed__ float table[4] = {1, 2, 3, 4};
__managed__ Pair pair = {7, 0.5};
__managed__ int untouched_by_host = 11;
__managed__ int on_device[2];

__global__ void bump(int by) {
  atomicAdd(&counter, by);
  table[threadIdx.x] *= 2;
  if (threadIdx.x == 0) {
    pair.a += pair.b > 0 ? 1 : -1;
    pair.b *= 4;
    untouched_by_host += 1;
  }
}

__global__ void record(int device) { on_device[device] = counter; }

int main() {
  // Initial values, read on the host before any kernel has run.
  check(counter == 5 && table[3] == 4 && pair.a == 7 && pair.b == 0.5, "initial values, seen from the host");
  // Written on the host, then by a kernel, then read on the host again.
  counter = 100;
  bump<<<1, 4>>>(3);
  check(cudaDeviceSynchronize() == cudaSuccess, "the kernel ran");
  check(counter == 112, "a host write the kernel saw, and the kernel's four adds");
  check(table[0] == 2 && table[3] == 8, "an array doubled on the device");
  check(pair.a == 8 && pair.b == 2.0, "a struct updated on the device");
  check(untouched_by_host == 12, "a variable only the device changed, from its initial value");

  // It is managed memory, at one address: the runtime calls it that, and
  // copies and memsets reach it.
  cudaPointerAttributes attr{};
  check(cudaPointerGetAttributes(&attr, &counter) == cudaSuccess && attr.type == cudaMemoryTypeManaged,
        "cudaPointerGetAttributes calls it managed");
  int seven = 7;
  check(cudaMemcpy(&counter, &seven, sizeof seven, cudaMemcpyDefault) == cudaSuccess && counter == 7,
        "cudaMemcpy into it");
  check(cudaMemset(table, 0, sizeof table) == cudaSuccess && table[2] == 0, "cudaMemset over it");

  // Every device sees the same variable.
  int devices = 0;
  cudaGetDeviceCount(&devices);
  for (int d = 0; d < devices && d < 2; ++d) {
    cudaSetDevice(d);
    record<<<1, 1>>>(d);
    cudaDeviceSynchronize();
  }
  cudaSetDevice(0);
  check(on_device[0] == 7 && (devices < 2 || on_device[1] == 7), "each device reads the host's value");
  std::printf(failures ? "FAIL: %d managed-variable checks\n" : "PASS: every managed-variable check\n", failures);
  return failures ? 1 : 0;
}
