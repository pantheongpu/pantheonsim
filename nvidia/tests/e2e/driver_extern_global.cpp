// A PTX module that declares ".extern .global" (or ".extern .const") a variable nothing in the module defines. A whole-program module has nothing
// to resolve it against: NVIDIA's driver ignores the qualifier ("ptxas warning : Unresolved extern variable '_global_offset_x' in whole program
// compilation, ignoring extern qualifier"), makes it an ordinary zero-initialised module variable of that size, returns it from cuModuleGetGlobal
// and lets the host write it. PoCL's CUDA backend declares its work offset that way and launches every kernel with it (pantheonsim issue #681).
// Checked on an RTX 3080 Ti (driver 595.99.02): the load succeeds, the symbol is 4 bytes, a kernel reads 0, and after a host write of 7 it reads 7.
// Also: a name the module defines itself is not the extern one (its initial value wins, before or after the declaration), and an unsized
// ".extern .global x[]", which has no size to give, still loads.
#include <cuda.h>

#include <cstdio>
#include <cstring>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

static const char* kUndefined = R"(
.version 6.3
.target sm_75
.address_size 64
.extern .global .align 4 .u32 _global_offset_x;
.extern .const .align 4 .u32 _const_extern;
.visible .entry k(.param .u64 out)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [out];
  ld.global.u32 %r1, [_global_offset_x];
  ld.const.u32 %r2, [_const_extern];
  st.global.u32 [%rd1], %r1;
  st.global.u32 [%rd1+4], %r2;
  ret;
}
)";

// Defined after its declaration, and defined before another declaration of it.
static const char* kDefined = R"(
.version 6.3
.target sm_75
.address_size 64
.extern .global .align 4 .u32 later;
.global .align 4 .u32 later = 5;
.global .align 4 .u32 earlier = 6;
.extern .global .align 4 .u32 earlier;
.extern .global .b8 unsized[];
.visible .entry k(.param .u64 out)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [out];
  ld.global.u32 %r1, [later];
  ld.global.u32 %r2, [earlier];
  st.global.u32 [%rd1], %r1;
  st.global.u32 [%rd1+4], %r2;
  ret;
}
)";

static bool launch(CUfunction k, CUdeviceptr out, unsigned v[2]) {
  unsigned sentinel[2] = {0xdeadbeef, 0xdeadbeef};
  if (cuMemcpyHtoD(out, sentinel, 8)) return false;
  void* args[] = {&out};
  if (cuLaunchKernel(k, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr)) return false;
  if (cuCtxSynchronize()) return false;
  return cuMemcpyDtoH(v, out, 8) == CUDA_SUCCESS;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) || cuDeviceGet(&dev, 0) || cuDevicePrimaryCtxRetain(&ctx, dev) || cuCtxSetCurrent(ctx)) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  CUdeviceptr out;
  IS(cuMemAlloc(&out, 8), CUDA_SUCCESS);

  CUmodule mod = nullptr;
  IS(cuModuleLoadData(&mod, kUndefined), CUDA_SUCCESS);
  if (mod) {
    CUdeviceptr g = 0, c = 0;
    size_t gs = 0, cs = 0;
    IS(cuModuleGetGlobal(&g, &gs, mod, "_global_offset_x"), CUDA_SUCCESS);
    check(gs == 4, "the undefined .extern .global is 4 bytes");
    IS(cuModuleGetGlobal(&c, &cs, mod, "_const_extern"), CUDA_SUCCESS);
    check(cs == 4, "the undefined .extern .const is 4 bytes");
    CUfunction k = nullptr;
    IS(cuModuleGetFunction(&k, mod, "k"), CUDA_SUCCESS);
    unsigned v[2] = {1, 1};
    check(k && launch(k, out, v), "the kernel that reads them launches");
    check(v[0] == 0 && v[1] == 0, "both read zero");
    unsigned seven = 7;
    check(g && cuMemcpyHtoD(g, &seven, 4) == CUDA_SUCCESS, "the host writes the .global");
    check(k && launch(k, out, v), "the kernel launches again");
    check(v[0] == 7 && v[1] == 0, "it reads what the host wrote (the .const one, not checked on a card, still reads zero)");
    IS(cuModuleUnload(mod), CUDA_SUCCESS);
  }

  mod = nullptr;
  IS(cuModuleLoadData(&mod, kDefined), CUDA_SUCCESS);
  if (mod) {
    CUdeviceptr g = 0;
    size_t gs = 0;
    IS(cuModuleGetGlobal(&g, &gs, mod, "later"), CUDA_SUCCESS);
    check(gs == 4, "a name the module defines is one variable of its own size");
    CUfunction k = nullptr;
    IS(cuModuleGetFunction(&k, mod, "k"), CUDA_SUCCESS);
    unsigned v[2] = {0, 0};
    check(k && launch(k, out, v), "the defining module's kernel launches");
    check(v[0] == 5 && v[1] == 6, "the definitions' initial values win over their declarations");
    IS(cuModuleUnload(mod), CUDA_SUCCESS);
  }

  IS(cuMemFree(out), CUDA_SUCCESS);
  IS(cuDevicePrimaryCtxRelease(dev), CUDA_SUCCESS);
  std::printf(failures ? "FAIL: %d checks\n" : "PASS: every extern-global check\n", failures);
  return failures ? 1 : 0;
}
