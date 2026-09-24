/*
 * The rest of what a HIP program asks its runtime for: how much memory the
 * device has and how much is left, copies and fills issued on a stream, LDS
 * the launch pays for rather than the kernel, and a second device that keeps
 * its own memory.
 *
 * Like vector_add_hip.c it links against VirtualGPU's libamdhip64 and calls
 * nothing this does not implement.
 */
#include <stdio.h>
#include <stdlib.h>

#include "vgpu_hip.h"

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t _e = (call);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "%s failed: %s (%s)\n", #call, hipGetErrorString(_e),     \
              hipGetErrorName(_e));                                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: runtime_hip <memory code object>\n");
    return 2;
  }
  const int n = 64;

  /* What the device has, and what is left of it. */
  size_t total = 0, free_before = 0, free_after = 0;
  hipDevice_t dev = -1;
  char name[64] = {0};
  CHECK(hipDeviceGet(&dev, 0));
  CHECK(hipDeviceGetName(name, sizeof name, dev));
  CHECK(hipDeviceTotalMem(&total, dev));
  CHECK(hipMemGetInfo(&free_before, &total));
  printf("device %d is %s with %zu bytes\n", dev, name, total);

  void* big = NULL;
  const size_t chunk = 16u << 20;
  CHECK(hipMalloc(&big, chunk));
  CHECK(hipMemGetInfo(&free_after, &total));
  printf("an allocation costs what it asked for %d\n", free_before - free_after >= chunk);
  CHECK(hipFree(big));
  CHECK(hipMemGetInfo(&free_after, &total));
  printf("and freeing it gives that back %d\n", free_after == free_before);

  /* A copy and a fill issued on a stream. Nothing here runs behind the
   * program's back, so a stream's work is done when the call returns, and
   * synchronising is allowed to find nothing left to do. */
  hipStream_t stream = NULL;
  CHECK(hipStreamCreate(&stream));
  float host[64], back[64];
  for (int i = 0; i < n; ++i) host[i] = (float)i * 0.5f - 3.0f;
  float *din = NULL, *dout = NULL;
  CHECK(hipMalloc((void**)&din, sizeof host));
  CHECK(hipMalloc((void**)&dout, sizeof host));
  CHECK(hipMemsetAsync(dout, 0, sizeof host, stream));
  CHECK(hipMemcpyAsync(din, host, sizeof host, hipMemcpyHostToDevice, stream));
  CHECK(hipStreamSynchronize(stream));
  CHECK(hipMemcpy(back, din, sizeof back, hipMemcpyDeviceToHost));
  int same = 1;
  for (int i = 0; i < n; ++i) same = same && back[i] == host[i];
  printf("a copy on a stream lands %d\n", same);

  /* LDS the launch sizes: the kernel's shared array has no size of its own,
   * and the launch's shared-memory parameter is what pays for it. */
  hipModule_t module;
  hipFunction_t dyn;
  CHECK(hipModuleLoad(&module, argv[1]));
  CHECK(hipModuleGetFunction(&dyn, module, "dyn_lds"));
  int count = n;
  void* args[] = {&din, &dout, &count};
  CHECK(hipModuleLaunchKernel(dyn, 1, 1, 1, n, 1, 1, n * sizeof(float), stream, args, NULL));
  CHECK(hipStreamSynchronize(stream));
  CHECK(hipMemcpy(back, dout, sizeof back, hipMemcpyDeviceToHost));
  int wrong = 0;
  for (int i = 0; i < n; ++i)
    if (back[i] != host[(i + 1) % n] * 2.0f) ++wrong;
  printf("a kernel whose LDS the launch paid for %d\n", wrong == 0);

  /* And a launch that forgets to pay for it is refused rather than reading
   * memory no one reserved. */
  printf("a launch that forgets it is refused %d\n",
         hipModuleLaunchKernel(dyn, 1, 1, 1, n, 1, 1, 0, stream, args, NULL) != hipSuccess);

  CHECK(hipStreamDestroy(stream));
  CHECK(hipModuleUnload(module));

  /* A second device, when the rack has one: its memory is its own, so what
   * the first device holds is not missing from the second. */
  int devices = 0;
  size_t here_before = 0;
  CHECK(hipMemGetInfo(&here_before, &total));
  CHECK(hipGetDeviceCount(&devices));
  printf("devices %d\n", devices);
  if (devices > 1) {
    size_t other_free = 0, other_total = 0;
    CHECK(hipSetDevice(1));
    CHECK(hipMemGetInfo(&other_free, &other_total));
    printf("a second device keeps its own memory %d\n", other_free == free_before && other_total == total);
    void* there = NULL;
    CHECK(hipMalloc(&there, chunk));
    CHECK(hipSetDevice(0));
    size_t here_free = 0;
    CHECK(hipMemGetInfo(&here_free, &total));
    printf("and what it holds is not missing from the first %d\n", here_free == here_before);
    CHECK(hipSetDevice(1));
    CHECK(hipFree(there));
    CHECK(hipSetDevice(0));
  }
  CHECK(hipFree(din));
  CHECK(hipFree(dout));
  return wrong ? 1 : 0;
}
