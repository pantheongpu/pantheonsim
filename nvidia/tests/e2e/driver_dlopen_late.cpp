// A program that links only libcudart, starts the runtime, and then loads the driver itself with a plain dlopen("libcuda.so.1", RTLD_NOW) -- the
// way cedana-samples' gpu_smr/jit_modules.cu and vmm_pools.cu "reach the driver through dlopen so the binary links only libcudart".
// On a card the runtime has brought the driver up by then, and made the device's primary context current on the thread, so the program's first
// driver call works with no cuInit of its own: cuCtxGetDevice answers 0, device 0. The simulator's driver shim kept its own "initialized" flag
// and answered CUDA_ERROR_NOT_INITIALIZED, because the runtime looks cuInit up in the global scope, which a library opened RTLD_LOCAL, or
// opened after the runtime started, is not in (pantheonsim issue #685; checked on an RTX 3080 Ti, driver 595.99.02).
//
// Two orders: the driver opened after the runtime's first call, and before it. The second run is this program again, with "early".
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

static int run(bool driver_first) {
  void* h = nullptr;
  if (driver_first) h = dlopen("libcuda.so.1", RTLD_NOW);   // (opened before the runtime has done anything, privately)
  check(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0)");
  check(cudaFree(nullptr) == cudaSuccess, "cudaFree(0) starts the runtime");
  if (!driver_first) h = dlopen("libcuda.so.1", RTLD_NOW);   // (opened after it, privately: in no scope the runtime looks in)
  check(h != nullptr, "dlopen(libcuda.so.1)");
  if (!h) return 1;
  auto getDevice = reinterpret_cast<CUresult (*)(CUdevice*)>(dlsym(h, "cuCtxGetDevice"));
  auto getCurrent = reinterpret_cast<CUresult (*)(CUcontext*)>(dlsym(h, "cuCtxGetCurrent"));
  auto count = reinterpret_cast<CUresult (*)(int*)>(dlsym(h, "cuDeviceGetCount"));
  auto init = reinterpret_cast<CUresult (*)(unsigned)>(dlsym(h, "cuInit"));
  check(getDevice && getCurrent && count && init, "the driver entry points are found");
  if (!(getDevice && getCurrent && count && init)) return 1;
  CUdevice d = -1;
  const CUresult r = getDevice(&d);
  std::printf("cuCtxGetDevice -> %d, device %d\n", static_cast<int>(r), static_cast<int>(d));
  check(r == CUDA_SUCCESS && d == 0, "cuCtxGetDevice finds the runtime's context, with no cuInit of the program's own");
  CUcontext c = nullptr;
  check(getCurrent(&c) == CUDA_SUCCESS && c != nullptr, "cuCtxGetCurrent has a current context");
  int n = 0;
  check(count(&n) == CUDA_SUCCESS && n >= 1, "cuDeviceGetCount sees the device");
  check(init(0) == CUDA_SUCCESS, "a cuInit after that is a no-op that succeeds");
  return 0;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc > 1 && std::strcmp(argv[1], "early") == 0) {
    run(true);
    std::printf(failures ? "FAIL (early order): %d checks\n" : "PASS (early order)\n", failures);
    return failures ? 1 : 0;
  }
  run(false);
  std::fflush(stdout);
  // The other order needs a process whose runtime has not started yet.
  const std::string self = std::string("\"") + argv[0] + "\" early";
  const int rc = std::system(self.c_str());
  check(rc == 0, "the driver opened before the runtime starts works too");
  std::printf(failures ? "FAIL: %d checks\n" : "PASS: every late-dlopen check\n", failures);
  return failures ? 1 : 0;
}
