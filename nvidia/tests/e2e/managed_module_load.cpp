// __managed__ variables in a module loaded through the driver API, from each
// form nvcc writes one in: a cubin, a fatbin and PTX.
//
//   managed_module_load cubin:<file> fatbin:<file> ptx:<file>
//
// For each: cuModuleGetGlobal names the variable and its size,
// CU_POINTER_ATTRIBUTE_IS_MANAGED says it is managed (and a plain __device__
// variable is not), the host reads its initial value and writes it in place,
// and a kernel's writes reach the host after cuCtxSynchronize. Every check
// passes on an RTX 3060 for all three forms.
//
// A cubin is SASS, which VirtualGPU executes only once SASS support is in the
// tree (PR #255). Until then run_managed_module.sh sets VGPU_CUBIN_PENDING,
// and a cubin need only be refused cleanly; the other two forms carry PTX and
// are checked in full regardless.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool ok, const std::string& what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++failures;
}

static void module_checks(const std::string& kind, const std::vector<char>& image) {
  const std::string in = " (" + kind + ")";
  CUmodule mod;
  const CUresult loaded = cuModuleLoadData(&mod, image.data());
  if (kind == "cubin" && std::getenv("VGPU_CUBIN_PENDING")) {
    check(loaded != CUDA_SUCCESS, "a cubin is refused until SASS support lands" + in);
    if (loaded != CUDA_SUCCESS) std::printf("note: cubin checks wait for SASS support (PR #255)\n");
    if (loaded == CUDA_SUCCESS) cuModuleUnload(mod);
    return;
  }
  check(loaded == CUDA_SUCCESS, "the module loads" + in);
  if (loaded != CUDA_SUCCESS) return;
  CUdeviceptr counter = 0, scaled = 0, plain = 0;
  size_t bytes = 0;
  check(cuModuleGetGlobal(&counter, &bytes, mod, "counter") == CUDA_SUCCESS && bytes == 4,
        "cuModuleGetGlobal: the managed int and its size" + in);
  check(cuModuleGetGlobal(&scaled, &bytes, mod, "scaled") == CUDA_SUCCESS && bytes == 16,
        "cuModuleGetGlobal: the managed array and its size" + in);
  check(cuModuleGetGlobal(&plain, &bytes, mod, "plain") == CUDA_SUCCESS && bytes == 4,
        "cuModuleGetGlobal: the plain __device__ int" + in);
  if (!counter || !scaled || !plain) {
    cuModuleUnload(mod);
    return;
  }
  int managed = -1;
  check(cuPointerGetAttribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, counter) == CUDA_SUCCESS &&
            managed == 1,
        "CU_POINTER_ATTRIBUTE_IS_MANAGED is 1 for a managed global" + in);
  managed = -1;
  check(cuPointerGetAttribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, plain) == CUDA_SUCCESS &&
            managed == 0,
        "and 0 for a plain one" + in);
  int* hc = reinterpret_cast<int*>(counter);
  float* hs = reinterpret_cast<float*>(scaled);
  check(*hc == 5 && hs[0] == 1.0f && hs[3] == 4.0f, "the host reads the initial values" + in);
  *hc = 10;   // the host writes in place; the kernel sees it
  CUfunction fn;
  int by = 4;
  void* args[] = {&by};
  const bool ran = cuModuleGetFunction(&fn, mod, "bump") == CUDA_SUCCESS &&
                   cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS &&
                   cuCtxSynchronize() == CUDA_SUCCESS;
  check(ran, "the kernel runs" + in);
  check(ran && *hc == 10 + 4 + 3, "the kernel's write to the managed int reaches the host" + in);
  check(ran && hs[0] == 2.0f && hs[1] == 4.0f && hs[2] == 6.0f && hs[3] == 8.0f,
        "and to the managed array" + in);
  int v = 0;
  check(cuMemcpyDtoH(&v, counter, 4) == CUDA_SUCCESS && v == 17, "cuMemcpyDtoH reads the same" + in);
  check(cuModuleUnload(mod) == CUDA_SUCCESS, "the module unloads" + in);
}

int main(int argc, char** argv) {
  if (cuInit(0) != CUDA_SUCCESS) {
    std::printf("FAIL: cuInit\n");
    return 1;
  }
  CUdevice dev;
  CUcontext ctx;
  cuDeviceGet(&dev, 0);
  cuDevicePrimaryCtxRetain(&ctx, dev);
  cuCtxSetCurrent(ctx);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t colon = arg.find(':');
    if (colon == std::string::npos) continue;
    std::ifstream f(arg.substr(colon + 1), std::ios::binary);
    std::vector<char> image((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    image.push_back('\0');   // PTX is loaded as a C string
    module_checks(arg.substr(0, colon), image);
  }
  std::printf(failures ? "FAIL: %d managed-module checks\n" : "PASS: every managed-module check\n",
              failures);
  return failures != 0;
}
