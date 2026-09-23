/*
 * A HIP program, written against the documented API, run on a simulated AMD
 * GPU: it asks what device it has, allocates memory on it, loads a code
 * object built for gfx942, launches the kernel in it, and checks the answer.
 *
 * It links against VirtualGPU's libamdhip64 the way it would link against
 * AMD's, and calls nothing this does not implement.
 */
#include <stdio.h>
#include <stdlib.h>

#include "vgpu_hip.h"

#define CHECK(call)                                                                  \
  do {                                                                               \
    hipError_t _e = (call);                                                          \
    if (_e != hipSuccess) {                                                          \
      fprintf(stderr, "%s failed: %s (%s)\n", #call, hipGetErrorString(_e),          \
              hipGetErrorName(_e));                                                  \
      return 1;                                                                      \
    }                                                                                \
  } while (0)

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: vector_add_hip <code object>\n");
    return 2;
  }
  const int n = 1000;

  int devices = 0;
  CHECK(hipGetDeviceCount(&devices));
  printf("devices %d\n", devices);

  hipDeviceProp_t props;
  CHECK(hipGetDeviceProperties(&props, 0));
  printf("device %s %s warp %d cus %d vram %zu\n", props.name, props.gcnArchName, props.warpSize,
         props.multiProcessorCount, props.totalGlobalMem);

  float *a = malloc(n * sizeof(float)), *b = malloc(n * sizeof(float)), *out = malloc(n * sizeof(float));
  for (int i = 0; i < n; ++i) {
    a[i] = (float)i;
    b[i] = (float)(2 * i + 1);
    out[i] = -1.0f;
  }

  void *da = NULL, *db = NULL, *dout = NULL;
  CHECK(hipMalloc(&da, n * sizeof(float)));
  CHECK(hipMalloc(&db, n * sizeof(float)));
  CHECK(hipMalloc(&dout, n * sizeof(float)));
  CHECK(hipMemcpy(da, a, n * sizeof(float), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(db, b, n * sizeof(float), hipMemcpyHostToDevice));
  CHECK(hipMemset(dout, 0, n * sizeof(float)));

  hipModule_t module;
  hipFunction_t kernel;
  CHECK(hipModuleLoad(&module, argv[1]));
  CHECK(hipModuleGetFunction(&kernel, module, "vector_add"));

  int count = n;
  void* args[] = {&da, &db, &dout, &count};
  CHECK(hipModuleLaunchKernel(kernel, (n + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(out, dout, n * sizeof(float), hipMemcpyDeviceToHost));

  int wrong = 0;
  for (int i = 0; i < n; ++i)
    if (out[i] != a[i] + b[i]) {
      if (wrong < 3) fprintf(stderr, "out[%d] = %g, not %g\n", i, out[i], a[i] + b[i]);
      ++wrong;
    }
  printf("wrong %d of %d\n", wrong, n);

  /* A kernel this code object does not have, and a launch with no work-items:
   * both must be refused rather than quietly doing nothing. */
  hipFunction_t missing;
  printf("missing kernel refused %d\n", hipModuleGetFunction(&missing, module, "no_such_kernel") != hipSuccess);
  printf("empty launch refused %d\n",
         hipModuleLaunchKernel(kernel, 1, 1, 1, 0, 1, 1, 0, NULL, args, NULL) != hipSuccess);

  CHECK(hipModuleUnload(module));
  CHECK(hipFree(da));
  CHECK(hipFree(db));
  CHECK(hipFree(dout));
  free(a);
  free(b);
  free(out);
  return wrong ? 1 : 0;
}
