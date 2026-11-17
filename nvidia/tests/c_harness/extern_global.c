/*
 * extern_global.c -- an .extern .global variable that nothing in the module defines, from a plain C program.
 *
 * PoCL's CUDA backend (OpenCL C compiled to PTX) declares its work offset as
 *     .extern .global .align 4 .u32 _global_offset_x;
 * and never defines it, then reads it in every kernel. A real driver loading the module as a whole program ignores the
 * qualifier (the JIT log says "Unresolved extern variable '_global_offset_x' in whole program compilation, ignoring extern
 * qualifier") and gives the variable zeroed storage of the declared size: cuModuleGetGlobal returns it and the host can write
 * it. Measured on an RTX 3080 Ti (driver 595.99.02): the kernel reads 0, and 7 after the host writes 7. This checks the same
 * on the simulator, and that an .extern declaration of a variable the module does define is not a second variable.
 */
#include <stdio.h>
#include <string.h>

#include "vgpu_cuda.h"

/* Not in vgpu_cuda.h. */
CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);

#define CHECK(call)                                                      \
  do {                                                                   \
    CUresult rc_ = (call);                                               \
    if (rc_ != CUDA_SUCCESS) {                                           \
      fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #call, (int)rc_); \
      return 1;                                                          \
    }                                                                    \
  } while (0)

/* Reads the variable into out[0]. */
static const char* kUndefined =
    ".version 6.3\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".extern .global .align 4 .u32 _global_offset_x;\n"
    ".visible .entry k(.param .u64 out)\n"
    "{\n"
    "  .reg .b32 %r<2>;\n"
    "  .reg .b64 %rd<2>;\n"
    "  ld.param.u64 %rd1, [out];\n"
    "  ld.global.u32 %r1, [_global_offset_x];\n"
    "  st.global.u32 [%rd1], %r1;\n"
    "  ret;\n"
    "}\n";

/* The same name declared .extern and then defined (with a value) in the module: one variable, the defined one. */
static const char* kDefined =
    ".version 6.3\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".extern .global .align 4 .u32 v;\n"
    ".global .align 4 .u32 v = 42;\n"
    ".visible .entry k(.param .u64 out)\n"
    "{\n"
    "  .reg .b32 %r<2>;\n"
    "  .reg .b64 %rd<2>;\n"
    "  ld.param.u64 %rd1, [out];\n"
    "  ld.global.u32 %r1, [v];\n"
    "  st.global.u32 [%rd1], %r1;\n"
    "  ret;\n"
    "}\n";

static int run(CUfunction fn, CUdeviceptr out, unsigned* value) {
  unsigned poison = 0xdeadbeefu;
  void* args[] = {&out};
  CHECK(cuMemcpyHtoD(out, &poison, sizeof poison));
  CHECK(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL));
  CHECK(cuCtxSynchronize());
  CHECK(cuMemcpyDtoH(value, out, sizeof *value));
  return 0;
}

int main(void) {
  CUdevice dev;
  CUcontext ctx;
  CHECK(cuInit(0));
  CHECK(cuDeviceGet(&dev, 0));
  CHECK(cuCtxCreate(&ctx, 0, dev));
  CUdeviceptr out;
  CHECK(cuMemAlloc(&out, 4));

  /* An undefined extern is zeroed storage the host can see and write. */
  CUmodule mod;
  CUfunction fn;
  CHECK(cuModuleLoadData(&mod, kUndefined));
  CHECK(cuModuleGetFunction(&fn, mod, "k"));
  CUdeviceptr g = 0;
  size_t gsize = 0;
  CHECK(cuModuleGetGlobal_v2(&g, &gsize, mod, "_global_offset_x"));
  if (g == 0 || gsize != 4) {
    fprintf(stderr, "FAIL: cuModuleGetGlobal gave address %llx, size %zu; expected a variable of 4 bytes\n",
            (unsigned long long)g, gsize);
    return 1;
  }
  unsigned v = 1;
  if (run(fn, out, &v)) return 1;
  if (v != 0) {
    fprintf(stderr, "FAIL: the undefined extern read as %u, expected 0\n", v);
    return 1;
  }
  unsigned seven = 7;
  CHECK(cuMemcpyHtoD(g, &seven, sizeof seven));
  if (run(fn, out, &v)) return 1;
  if (v != 7) {
    fprintf(stderr, "FAIL: after the host wrote 7 the kernel read %u\n", v);
    return 1;
  }
  CHECK(cuModuleUnload(mod));

  /* An .extern declaration of a variable the module defines changes nothing. */
  CHECK(cuModuleLoadData(&mod, kDefined));
  CHECK(cuModuleGetFunction(&fn, mod, "k"));
  CHECK(cuModuleGetGlobal_v2(&g, &gsize, mod, "v"));
  if (gsize != 4) {
    fprintf(stderr, "FAIL: the defined variable has size %zu, expected 4\n", gsize);
    return 1;
  }
  if (run(fn, out, &v)) return 1;
  if (v != 42) {
    fprintf(stderr, "FAIL: the defined variable read as %u, expected 42\n", v);
    return 1;
  }
  CHECK(cuModuleUnload(mod));

  CHECK(cuMemFree(out));
  printf("PASS\n");
  return 0;
}
