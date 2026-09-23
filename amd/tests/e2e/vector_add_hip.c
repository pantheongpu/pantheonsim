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

  /* A module's own variables, through the documented interface: the second
   * code object declares scale and counters, and a program sets one before
   * the kernel reads it. */
  if (argc > 2) {
    hipModule_t globals;
    hipFunction_t use_global;
    CHECK(hipModuleLoad(&globals, argv[2]));
    CHECK(hipModuleGetFunction(&use_global, globals, "use_global"));
    void* dscale = NULL;
    size_t scale_bytes = 0;
    CHECK(hipModuleGetGlobal(&dscale, &scale_bytes, globals, "scale"));
    printf("scale is %zu bytes\n", scale_bytes);
    float two = 2.0f;
    CHECK(hipMemcpy(dscale, &two, sizeof two, hipMemcpyHostToDevice));

    int* din = NULL;
    int* dsum = NULL;
    int ones[64];
    for (int i = 0; i < 64; ++i) ones[i] = 10;
    CHECK(hipMalloc((void**)&din, sizeof ones));
    CHECK(hipMalloc((void**)&dsum, sizeof ones));
    CHECK(hipMemcpy(din, ones, sizeof ones, hipMemcpyHostToDevice));
    int sixty_four = 64;
    void* gargs[] = {&din, &dsum, &sixty_four};
    CHECK(hipModuleLaunchKernel(use_global, 1, 1, 1, 64, 1, 1, 0, NULL, gargs, NULL));
    int back[64];
    CHECK(hipMemcpy(back, dsum, sizeof back, hipMemcpyDeviceToHost));
    printf("global scale applied %d\n", back[0] == 20 && back[63] == 20);
    /* What the kernel wrote into the module's own array, read from the host. */
    int* dcounters = NULL;
    CHECK(hipModuleGetGlobal((void**)&dcounters, NULL, globals, "counters"));
    CHECK(hipMemcpy(back, dcounters, sizeof back, hipMemcpyDeviceToHost));
    printf("kernel wrote its own array %d\n", back[0] == 10 && back[63] == 10);
    CHECK(hipFree(din));
    CHECK(hipFree(dsum));
    CHECK(hipModuleUnload(globals));
  }

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
