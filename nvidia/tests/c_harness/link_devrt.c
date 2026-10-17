/*
 * link_devrt.c -- the driver API's JIT linker taking libcudadevrt.a, from a plain C program.
 *
 * A program whose kernels launch kernels (dynamic parallelism) links the device runtime with
 * cuLinkAddData(CU_JIT_INPUT_LIBRARY, libcudadevrt.a) -- LuisaCompute does it at startup, as a probe
 * of whether indirect dispatch can work. VirtualGPU implements the device runtime itself, so the
 * archive adds nothing to the module and is accepted; any other archive, or bytes that are no archive
 * at all, would be code to link in and is refused rather than dropped. The kernels then run: the
 * parent below launches the child from the device, and the module is the PTX nvcc -rdc=true wrote
 * (CUDA 13.0, sm_75).
 *
 * The archive is built here, with the layout the toolkit's has: a symbol table, a long-name table
 * and one member named cuda_device_runtime.o, whose body is not read.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vgpu_cuda.h"

/* These entry points are not in vgpu_cuda.h. */
CUresult cuMemsetD32(CUdeviceptr dst, unsigned int value, size_t count);
typedef struct CUlinkState_st* CUlinkState;
CUresult cuLinkCreate(unsigned int numOptions, int* options, void** optionValues, CUlinkState* stateOut);
CUresult cuLinkAddData(CUlinkState state, int type, void* data, size_t size, const char* name,
                       unsigned int numOptions, int* options, void** optionValues);
CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut);
CUresult cuLinkDestroy(CUlinkState state);
enum { kInputPtx = 1, kInputLibrary = 4 };

#define CHECK(call)                                                      \
  do {                                                                   \
    CUresult rc_ = (call);                                               \
    if (rc_ != CUDA_SUCCESS) {                                           \
      fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #call, (int)rc_); \
      return 1;                                                          \
    }                                                                    \
  } while (0)

#define EXPECT_FAIL(call)                                                \
  do {                                                                   \
    if ((call) == CUDA_SUCCESS) {                                        \
      fprintf(stderr, "FAIL %s:%d: %s succeeded, expected a refusal\n", __FILE__, __LINE__, #call); \
      return 1;                                                          \
    }                                                                    \
  } while (0)

static const char* kModule =
    ".version 9.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    "  // .globl  child\n"
    ".extern .func  (.param .b64 func_retval0) __cudaCDP2GetParameterBufferV2\n"
    "(\n"
    "  .param .b64 __cudaCDP2GetParameterBufferV2_param_0,\n"
    "  .param .align 4 .b8 __cudaCDP2GetParameterBufferV2_param_1[12],\n"
    "  .param .align 4 .b8 __cudaCDP2GetParameterBufferV2_param_2[12],\n"
    "  .param .b32 __cudaCDP2GetParameterBufferV2_param_3\n"
    ")\n"
    ";\n"
    ".extern .func  (.param .b32 func_retval0) __cudaCDP2LaunchDeviceV2\n"
    "(\n"
    "  .param .b64 __cudaCDP2LaunchDeviceV2_param_0,\n"
    "  .param .b64 __cudaCDP2LaunchDeviceV2_param_1\n"
    ")\n"
    ";\n"
    ".visible .entry child(\n"
    "  .param .u64 child_param_0,\n"
    "  .param .u32 child_param_1\n"
    ")\n"
    "{\n"
    "  .reg .b32   %r<4>;\n"
    "  .reg .b64   %rd<5>;\n"
    "  ld.param.u64   %rd1, [child_param_0];\n"
    "  ld.param.u32   %r1, [child_param_1];\n"
    "  cvta.to.global.u64   %rd2, %rd1;\n"
    "  mov.u32   %r2, %tid.x;\n"
    "  add.s32   %r3, %r2, %r1;\n"
    "  mul.wide.u32   %rd3, %r2, 4;\n"
    "  add.s64   %rd4, %rd2, %rd3;\n"
    "  st.global.u32   [%rd4], %r3;\n"
    "  ret;\n"
    "}\n"
    "  // .globl  parent\n"
    ".visible .entry parent(\n"
    "  .param .u64 parent_param_0,\n"
    "  .param .u32 parent_param_1\n"
    ")\n"
    "{\n"
    "  .reg .pred   %p<3>;\n"
    "  .reg .b32   %r<4>;\n"
    "  .reg .b64   %rd<4>;\n"
    "  ld.param.u64   %rd2, [parent_param_0];\n"
    "  ld.param.u32   %r1, [parent_param_1];\n"
    "  mov.u32   %r2, %tid.x;\n"
    "  setp.ne.s32   %p1, %r2, 0;\n"
    "  @%p1 bra   $L__BB1_3;\n"
    "  mov.u64   %rd3, child;\n"
    "  { // callseq 0, 0\n"
    "  .reg .b32 temp_param_reg;\n"
    "  .param .b64 param0;\n"
    "  st.param.b64   [param0+0], %rd3;\n"
    "  .param .align 4 .b8 param1[12];\n"
    "  st.param.b32   [param1+0], 1;\n"
    "  st.param.b32   [param1+4], 1;\n"
    "  st.param.b32   [param1+8], 1;\n"
    "  .param .align 4 .b8 param2[12];\n"
    "  st.param.b32   [param2+0], 32;\n"
    "  st.param.b32   [param2+4], 1;\n"
    "  st.param.b32   [param2+8], 1;\n"
    "  .param .b32 param3;\n"
    "  st.param.b32   [param3+0], 0;\n"
    "  .param .b64 retval0;\n"
    "  call.uni (retval0), \n"
    "  __cudaCDP2GetParameterBufferV2, \n"
    "  (\n"
    "  param0, \n"
    "  param1, \n"
    "  param2, \n"
    "  param3\n"
    "  );\n"
    "  ld.param.b64   %rd1, [retval0+0];\n"
    "  } // callseq 0\n"
    "  setp.eq.s64   %p2, %rd1, 0;\n"
    "  @%p2 bra   $L__BB1_3;\n"
    "  st.u64   [%rd1], %rd2;\n"
    "  st.u32   [%rd1+8], %r1;\n"
    "  { // callseq 1, 0\n"
    "  .reg .b32 temp_param_reg;\n"
    "  .param .b64 param0;\n"
    "  st.param.b64   [param0+0], %rd1;\n"
    "  .param .b64 param1;\n"
    "  st.param.b64   [param1+0], 0;\n"
    "  .param .b32 retval0;\n"
    "  call.uni (retval0), \n"
    "  __cudaCDP2LaunchDeviceV2, \n"
    "  (\n"
    "  param0, \n"
    "  param1\n"
    "  );\n"
    "  ld.param.b32   %r3, [retval0+0];\n"
    "  } // callseq 1\n"
    "$L__BB1_3:\n"
    "  ret;\n"
    "}\n"
    ;

/* One `ar` member header: name(16) mtime(12) uid(6) gid(6) mode(8) size(10) "`\n". */
static size_t put_header(char* out, const char* name, size_t body) {
  snprintf(out, 61, "%-16.16s%-12s%-6s%-6s%-8s%-10u`\n", name, "0", "0", "0", "100644", (unsigned)body);
  return 60;
}

/* An archive with a symbol table, a long-name table and one member, `member` (longer than 15
 * characters, so named through the table as the toolkit's is). */
static size_t make_archive(char* out, const char* member) {
  size_t at = 0;
  memcpy(out, "!<arch>\n", 8);
  at = 8;
  /* Symbol table: a count of 0, in the big-endian form `ar` writes. */
  at += put_header(out + at, "/", 4);
  memset(out + at, 0, 4);
  at += 4;
  /* Long names: "<member>/\n", padded to an even length. */
  char names[64];
  size_t nlen = (size_t)snprintf(names, sizeof names, "%s/\n", member);
  size_t padded = nlen + (nlen & 1);
  at += put_header(out + at, "//", padded);
  memset(out + at, '\n', padded);
  memcpy(out + at, names, nlen);
  at += padded;
  /* The member, named by its offset in the table. */
  at += put_header(out + at, "/0", 16);
  memset(out + at, 0x5a, 16);
  at += 16;
  return at;
}

int main(void) {
  CHECK(cuInit(0));
  CUdevice dev;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CHECK(cuCtxCreate(&ctx, 0, dev));

  char devrt[512], other[512], junk[64];
  size_t devrt_size = make_archive(devrt, "cuda_device_runtime.o");
  size_t other_size = make_archive(other, "libother_object_file.o");
  memset(junk, 0x11, sizeof junk);

  /* The device runtime, then the module: the link completes and the module runs the launch. */
  CUlinkState st;
  CHECK(cuLinkCreate(0, NULL, NULL, &st));
  CHECK(cuLinkAddData(st, kInputLibrary, devrt, devrt_size, "cudadevrt", 0, NULL, NULL));
  CHECK(cuLinkAddData(st, kInputPtx, (void*)kModule, strlen(kModule) + 1, "parent.ptx", 0, NULL, NULL));
  void* image = NULL;
  size_t image_size = 0;
  CHECK(cuLinkComplete(st, &image, &image_size));

  CUmodule mod;
  CHECK(cuModuleLoadData(&mod, image));
  CUfunction parent;
  CHECK(cuModuleGetFunction(&parent, mod, "parent"));
  CUdeviceptr d;
  CHECK(cuMemAlloc(&d, 32 * sizeof(int)));
  CHECK(cuMemsetD32(d, 0xffffffffu, 32));
  int base = 100;
  void* params[] = {&d, &base};
  CHECK(cuLaunchKernel(parent, 1, 1, 1, 32, 1, 1, 0, NULL, params, NULL));
  CHECK(cuCtxSynchronize());
  int got[32];
  CHECK(cuMemcpyDtoH(got, d, sizeof got));
  for (int i = 0; i < 32; ++i)
    if (got[i] != base + i) {
      fprintf(stderr, "FAIL: the child grid wrote got[%d] = %d, expected %d\n", i, got[i], base + i);
      return 1;
    }
  CHECK(cuMemFree(d));
  CHECK(cuLinkDestroy(st));

  /* The library alone is no module. */
  CHECK(cuLinkCreate(0, NULL, NULL, &st));
  CHECK(cuLinkAddData(st, kInputLibrary, devrt, devrt_size, "cudadevrt", 0, NULL, NULL));
  EXPECT_FAIL(cuLinkComplete(st, &image, &image_size));
  CHECK(cuLinkDestroy(st));

  /* Any other archive, and anything that is not an archive, is refused. */
  CHECK(cuLinkCreate(0, NULL, NULL, &st));
  EXPECT_FAIL(cuLinkAddData(st, kInputLibrary, other, other_size, "other", 0, NULL, NULL));
  EXPECT_FAIL(cuLinkAddData(st, kInputLibrary, junk, sizeof junk, "junk", 0, NULL, NULL));
  EXPECT_FAIL(cuLinkAddData(st, kInputLibrary, devrt, 40, "truncated", 0, NULL, NULL));
  CHECK(cuLinkDestroy(st));

  printf("PASS\n");
  return 0;
}
