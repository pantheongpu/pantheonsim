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
 *
 * It also checks what the same card refuses, which is not the same as zeroed storage: an .extern with an initialiser
 * ("No initial value is allowed for external symbol"), an unsized .extern array ("Unresolved extern variable") and an .extern
 * resolved only by a plain (non-.visible) definition (".extern variable cannot be resolved by a .static") all fail the load
 * with CUDA_ERROR_INVALID_PTX; cuLinkComplete on an undefined .extern fails with CUDA_ERROR_INVALID_SOURCE and "Undefined
 * reference to 'x'", yet links when another input defines it; and the load's info log carries the ptxas warning.
 */
#include <stdio.h>
#include <string.h>

#include "vgpu_cuda.h"

/* Not in vgpu_cuda.h. */
CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);
typedef struct CUlinkState_st* CUlinkState;
CUresult cuLinkCreate(unsigned int numOptions, int* options, void** optionValues, CUlinkState* stateOut);
CUresult cuLinkAddData(CUlinkState state, int type, void* data, size_t size, const char* name,
                       unsigned int numOptions, int* options, void** optionValues);
CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut);
CUresult cuLinkDestroy(CUlinkState state);
enum { kInputPtx = 1, kInvalidSource = 300 };

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

/* The same name declared .extern and then defined (with a value, and .visible so the declaration can resolve to it): one variable. */
static const char* kDefined =
    ".version 6.3\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".extern .global .align 4 .u32 v;\n"
    ".visible .global .align 4 .u32 v = 42;\n"
    ".visible .entry k(.param .u64 out)\n"
    "{\n"
    "  .reg .b32 %r<2>;\n"
    "  .reg .b64 %rd<2>;\n"
    "  ld.param.u64 %rd1, [out];\n"
    "  ld.global.u32 %r1, [v];\n"
    "  st.global.u32 [%rd1], %r1;\n"
    "  ret;\n"
    "}\n";

/* A kernel reading a variable, for the modules below: the first line is the declaration under test. */
#define KERNEL_READING(decl, name)                                                  \
  ".version 6.3\n.target sm_75\n.address_size 64\n" decl                            \
  ".visible .entry k(.param .u64 out)\n{\n"                                         \
  "  .reg .b32 %r<2>;\n  .reg .b64 %rd<2>;\n"                                       \
  "  ld.param.u64 %rd1, [out];\n"                                                  \
  "  ld.global.u32 %r1, [" name "];\n"                                             \
  "  st.global.u32 [%rd1], %r1;\n  ret;\n}\n"

static const char* kInitialiser = KERNEL_READING(".extern .global .align 4 .u32 e = 5;\n", "e");
static const char* kUnsized = KERNEL_READING(".extern .global .align 4 .b8 arr[];\n", "arr");
static const char* kStaticDefinition =
    KERNEL_READING(".extern .global .align 4 .u32 s;\n.global .align 4 .u32 s = 42;\n", "s");
/* Two inputs to the linker: one declares .extern, the other defines it (.visible). */
static const char* kLinkUser = KERNEL_READING(".extern .global .align 4 .u32 base;\n", "base");
static const char* kLinkDefiner = ".version 6.3\n.target sm_75\n.address_size 64\n.visible .global .align 4 .u32 base = 77;\n";

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

  /* What the card refuses: the load fails with INVALID_PTX. */
  {
    const char* bad[] = {kInitialiser, kUnsized, kStaticDefinition};
    const char* what[] = {"an .extern with an initialiser", "an unsized .extern array", "an .extern resolved by a plain definition"};
    for (int i = 0; i < 3; ++i) {
      CUmodule m = NULL;
      const CUresult rc = cuModuleLoadData(&m, bad[i]);
      if (rc != CUDA_ERROR_INVALID_PTX) {
        fprintf(stderr, "FAIL: %s loaded with %d, expected CUDA_ERROR_INVALID_PTX (%d)\n", what[i], (int)rc, (int)CUDA_ERROR_INVALID_PTX);
        return 1;
      }
    }
  }

  /* The load's info log carries the ptxas warning for an undefined .extern. */
  {
    char log[512] = {0};
    int opt[] = {3, 4};   /* CU_JIT_INFO_LOG_BUFFER, ..._SIZE_BYTES */
    void* val[] = {log, (void*)(size_t)sizeof log};
    CHECK(cuModuleLoadDataEx(&mod, kUndefined, 2, opt, val));
    if (!strstr(log, "Unresolved extern variable '_global_offset_x'") || !strstr(log, "ignoring extern qualifier")) {
      fprintf(stderr, "FAIL: the info log is '%s'\n", log);
      return 1;
    }
    CHECK(cuModuleUnload(mod));
  }

  /* The linker: an .extern nothing defines is an undefined reference (INVALID_SOURCE), and one another input defines links. */
  {
    char log[512] = {0};
    int opt[] = {5, 6};   /* CU_JIT_ERROR_LOG_BUFFER, ..._SIZE_BYTES */
    void* val[] = {log, (void*)(size_t)sizeof log};
    CUlinkState st;
    void* image = NULL;
    size_t image_size = 0;
    CHECK(cuLinkCreate(2, opt, val, &st));
    CHECK(cuLinkAddData(st, kInputPtx, (void*)kUndefined, strlen(kUndefined) + 1, "undefined.ptx", 0, NULL, NULL));
    const CUresult rc = cuLinkComplete(st, &image, &image_size);
    if (rc != (CUresult)kInvalidSource || !strstr(log, "Undefined reference to '_global_offset_x'")) {
      fprintf(stderr, "FAIL: linking an undefined .extern gave %d with log '%s', expected %d and 'Undefined reference'\n", (int)rc, log,
              kInvalidSource);
      return 1;
    }
    CHECK(cuLinkDestroy(st));

    CHECK(cuLinkCreate(0, NULL, NULL, &st));
    CHECK(cuLinkAddData(st, kInputPtx, (void*)kLinkUser, strlen(kLinkUser) + 1, "user.ptx", 0, NULL, NULL));
    CHECK(cuLinkAddData(st, kInputPtx, (void*)kLinkDefiner, strlen(kLinkDefiner) + 1, "definer.ptx", 0, NULL, NULL));
    CHECK(cuLinkComplete(st, &image, &image_size));
    CHECK(cuModuleLoadData(&mod, image));
    CHECK(cuModuleGetFunction(&fn, mod, "k"));
    if (run(fn, out, &v)) return 1;
    if (v != 77) {
      fprintf(stderr, "FAIL: the linked definition read as %u, expected 77\n", v);
      return 1;
    }
    CHECK(cuModuleUnload(mod));
    CHECK(cuLinkDestroy(st));
  }

  CHECK(cuMemFree(out));
  printf("PASS\n");
  return 0;
}
