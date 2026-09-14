/*
 * vector_add_driver.c — a plain C11 "external CUDA application" for VirtualGPU.
 *
 * This is the M7 stand-in: it is compiled separately from the emulator with a
 * plain C compiler and talks to the virtual GPU purely through the documented
 * driver API ABI (vgpu_cuda.h + libvgpucuda.so). No VirtualGPU internals.
 *
 * It runs discovery, memory, module, and launch paths, verifies vectorAdd
 * results exactly, and probes several error paths. Prints PASS and exits 0
 * on success.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vgpu_cuda.h"

#define CHECK(call)                                                        \
  do {                                                                     \
    CUresult rc_ = (call);                                                 \
    if (rc_ != CUDA_SUCCESS) {                                             \
      const char* name_ = NULL;                                            \
      cuGetErrorName(rc_, &name_);                                         \
      fprintf(stderr, "FAIL %s:%d: %s -> %s (%d)\n", __FILE__, __LINE__,   \
              #call, name_ ? name_ : "?", (int)rc_);                       \
      return 1;                                                            \
    }                                                                      \
  } while (0)

#define EXPECT(call, expected)                                             \
  do {                                                                     \
    CUresult rc_ = (call);                                                 \
    if (rc_ != (expected)) {                                               \
      fprintf(stderr, "FAIL %s:%d: %s returned %d, expected %d\n",         \
              __FILE__, __LINE__, #call, (int)rc_, (int)(expected));       \
      return 1;                                                            \
    }                                                                      \
  } while (0)

static const char* kVecAddPtx =
    ".version 8.3\n"
    ".target sm_90\n"
    ".address_size 64\n"
    ".visible .entry vecAdd(\n"
    "    .param .u64 vecAdd_param_0,\n"
    "    .param .u64 vecAdd_param_1,\n"
    "    .param .u64 vecAdd_param_2,\n"
    "    .param .u32 vecAdd_param_3\n"
    ")\n"
    "{\n"
    "    .reg .pred %p<2>;\n"
    "    .reg .f32 %f<4>;\n"
    "    .reg .b32 %r<6>;\n"
    "    .reg .b64 %rd<11>;\n"
    "    ld.param.u64 %rd1, [vecAdd_param_0];\n"
    "    ld.param.u64 %rd2, [vecAdd_param_1];\n"
    "    ld.param.u64 %rd3, [vecAdd_param_2];\n"
    "    ld.param.u32 %r2, [vecAdd_param_3];\n"
    "    mov.u32 %r3, %ctaid.x;\n"
    "    mov.u32 %r4, %ntid.x;\n"
    "    mov.u32 %r5, %tid.x;\n"
    "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
    "    setp.ge.s32 %p1, %r1, %r2;\n"
    "    @%p1 bra $L__BB0_2;\n"
    "    cvta.to.global.u64 %rd4, %rd1;\n"
    "    mul.wide.s32 %rd5, %r1, 4;\n"
    "    add.s64 %rd6, %rd4, %rd5;\n"
    "    cvta.to.global.u64 %rd7, %rd2;\n"
    "    add.s64 %rd8, %rd7, %rd5;\n"
    "    ld.global.f32 %f1, [%rd8];\n"
    "    ld.global.f32 %f2, [%rd6];\n"
    "    add.f32 %f3, %f2, %f1;\n"
    "    cvta.to.global.u64 %rd9, %rd3;\n"
    "    add.s64 %rd10, %rd9, %rd5;\n"
    "    st.global.f32 [%rd10], %f3;\n"
    "$L__BB0_2:\n"
    "    ret;\n"
    "}\n";

int main(void) {
  /* ---- Level 1: discovery ---- */
  EXPECT(cuDeviceGetCount(&(int){0}), CUDA_ERROR_NOT_INITIALIZED); /* pre-init check */
  CHECK(cuInit(0));

  int version = 0;
  CHECK(cuDriverGetVersion(&version));

  int count = 0;
  CHECK(cuDeviceGetCount(&count));
  if (count < 1) {
    fprintf(stderr, "FAIL: no devices\n");
    return 1;
  }

  CUdevice dev;
  CHECK(cuDeviceGet(&dev, 0));
  EXPECT(cuDeviceGet(&dev, 42), CUDA_ERROR_INVALID_DEVICE);

  char name[256];
  CHECK(cuDeviceGetName(name, sizeof name, dev));

  size_t total_mem = 0;
  CHECK(cuDeviceTotalMem(&total_mem, dev));

  int cc_major = 0, cc_minor = 0, warp = 0, mp = 0, max_threads = 0;
  CHECK(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CHECK(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  CHECK(cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev));
  CHECK(cuDeviceGetAttribute(&mp, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  CHECK(cuDeviceGetAttribute(&max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev));

  printf("driver %d | device 0: %s | sm_%d%d | warp %d | %d SMs | %.1f GiB\n", version, name,
         cc_major, cc_minor, warp, mp, (double)total_mem / (1024.0 * 1024.0 * 1024.0));
  if (warp != 32 || max_threads != 1024) {
    fprintf(stderr, "FAIL: implausible device properties\n");
    return 1;
  }

  /* ---- Level 2: context + memory ---- */
  CUcontext ctx;
  CHECK(cuCtxCreate(&ctx, 0, dev));

  CUcontext cur = NULL;
  CHECK(cuCtxGetCurrent(&cur));
  if (cur != ctx) {
    fprintf(stderr, "FAIL: cuCtxCreate did not make context current\n");
    return 1;
  }

  size_t free_b = 0, total_b = 0;
  CHECK(cuMemGetInfo(&free_b, &total_b));
  if (total_b != total_mem || free_b != total_b) {
    fprintf(stderr, "FAIL: cuMemGetInfo mismatch\n");
    return 1;
  }

  enum { N = 1000 };
  float a[N], b[N], c[N];
  for (int i = 0; i < N; ++i) {
    a[i] = 0.5f * (float)i;
    b[i] = 0.25f * (float)i;
    c[i] = -1.0f;
  }

  CUdeviceptr da, db, dc;
  CHECK(cuMemAlloc(&da, N * sizeof(float)));
  CHECK(cuMemAlloc(&db, N * sizeof(float)));
  CHECK(cuMemAlloc(&dc, N * sizeof(float)));
  CHECK(cuMemcpyHtoD(da, a, N * sizeof(float)));
  CHECK(cuMemcpyHtoD(db, b, N * sizeof(float)));

  CHECK(cuMemGetInfo(&free_b, &total_b));
  if (total_b - free_b != 3 * N * sizeof(float)) {
    fprintf(stderr, "FAIL: memory accounting\n");
    return 1;
  }

  /* ---- Level 3: module + launch ---- */
  CUmodule mod;
  CHECK(cuModuleLoadData(&mod, kVecAddPtx));

  CUfunction fn;
  EXPECT(cuModuleGetFunction(&fn, mod, "noSuchKernel"), CUDA_ERROR_NOT_FOUND);
  CHECK(cuModuleGetFunction(&fn, mod, "vecAdd"));

  unsigned int n = N;
  void* params[] = {&da, &db, &dc, &n};
  CHECK(cuLaunchKernel(fn, (N + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, params, NULL));
  CHECK(cuCtxSynchronize());
  CHECK(cuMemcpyDtoH(c, dc, N * sizeof(float)));

  for (int i = 0; i < N; ++i) {
    float expect = a[i] + b[i];
    if (c[i] != expect) {
      fprintf(stderr, "FAIL: c[%d] = %g, expected %g\n", i, c[i], expect);
      return 1;
    }
  }

  /* ---- error paths ---- */
  /* Kernel-side out-of-bounds: lie about n so threads index past the buffers. */
  CUdeviceptr small_a, small_b, small_c;
  CHECK(cuMemAlloc(&small_a, 16 * sizeof(float)));
  CHECK(cuMemAlloc(&small_b, 16 * sizeof(float)));
  CHECK(cuMemAlloc(&small_c, 16 * sizeof(float)));
  unsigned int lie = 64;
  void* bad_params[] = {&small_a, &small_b, &small_c, &lie};
  EXPECT(cuLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, NULL, bad_params, NULL),
         CUDA_ERROR_ILLEGAL_ADDRESS);

  /* Block size beyond the profile limit. */
  EXPECT(cuLaunchKernel(fn, 1, 1, 1, 2048, 1, 1, 0, NULL, params, NULL), CUDA_ERROR_INVALID_VALUE);

  /* cubin images are rejected with a clear NOT_SUPPORTED, not garbage. */
  const char elf[8] = {0x7f, 'E', 'L', 'F', 2, 1, 1, 0};
  CUmodule bad_mod;
  EXPECT(cuModuleLoadData(&bad_mod, elf), CUDA_ERROR_NOT_SUPPORTED);

  /* Free/double-free. */
  CHECK(cuMemFree(da));
  EXPECT(cuMemFree(da), CUDA_ERROR_INVALID_VALUE);
  CHECK(cuMemFree(db));
  CHECK(cuMemFree(dc));
  CHECK(cuMemFree(small_a));
  CHECK(cuMemFree(small_b));
  CHECK(cuMemFree(small_c));

  CHECK(cuModuleUnload(mod));
  CHECK(cuCtxDestroy(ctx));

  printf("PASS\n");
  return 0;
}
