// How PyTorch reaches the driver, under `vgpu run --preload`: the program names no CUDA library (python does not), so the runtime is
// preloaded and the driver is loaded privately (VGPU_PRELOAD_PRIVATE, #597). PyTorch then uses the runtime (its libraries link
// libcudart), opens the already loaded driver with dlopen("libcuda.so.1", RTLD_NOLOAD) and calls it without a cuInit of its own,
// counting on the runtime to have initialised the driver and made its primary context current, as on the card. With the driver
// out of the global scope the runtime did neither, and every driver call answered CUDA_ERROR_NOT_INITIALIZED ("CUDA driver error:
// initialization error" on every PyTorch check). Also: cudaGetDriverEntryPoint, which PyTorch calls too, must not move the
// driver into the global scope, where a later library's own `cuInit` variable would bind to it again (#597).
// Run by run_preload_torch_like.sh; names no CUDA library, so it loads the runtime with dlopen.
#include <dlfcn.h>
#include <pthread.h>
#include <cstdio>
#include <cstring>

namespace {
using CudaSetDevice = int (*)(int);
using CudaMalloc = int (*)(void**, size_t);
using CudaFree = int (*)(void*);
using CudaEntry = int (*)(const char*, void**, unsigned long long, int*);
using CuCtxGetCurrent = int (*)(void**);
using CuMemGetAddressRange = int (*)(unsigned long long*, size_t*, unsigned long long);
using CuDriverGetVersion = int (*)(int*);
using CuDevicePrimaryCtxGetState = int (*)(int, unsigned*, int*);

void* g_driver = nullptr;
CudaSetDevice g_set_device = nullptr;
int g_fails = 0;

void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++g_fails;
}

// A thread the runtime used, as PyTorch's autograd threads are: its first driver call finds the runtime's primary context.
void* worker(void*) {
  g_set_device(0);
  const auto get_ctx = reinterpret_cast<CuCtxGetCurrent>(dlsym(g_driver, "cuCtxGetCurrent"));
  void* ctx = nullptr;
  const int rc = get_ctx ? get_ctx(&ctx) : -1;
  check(rc == 0 && ctx != nullptr, "cuCtxGetCurrent on a second thread the runtime used: success, the primary context");
  return nullptr;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  void* rt = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!rt) {
    std::printf("cannot load %s: %s\n", argv[1], dlerror());
    return 2;
  }
  g_set_device = reinterpret_cast<CudaSetDevice>(dlsym(rt, "cudaSetDevice"));
  const auto cuda_malloc = reinterpret_cast<CudaMalloc>(dlsym(rt, "cudaMalloc"));
  const auto cuda_free = reinterpret_cast<CudaFree>(dlsym(rt, "cudaFree"));
  const auto entry = reinterpret_cast<CudaEntry>(dlsym(rt, "cudaGetDriverEntryPoint"));
  if (!g_set_device || !cuda_malloc || !cuda_free || !entry) return 2;

  void* p = nullptr;
  check(g_set_device(0) == 0 && cuda_malloc(&p, 4096) == 0, "the runtime starts and allocates");

  // PyTorch's c10::cuda::DriverAPI: the driver the runtime already loaded, not a second one.
  g_driver = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
  check(g_driver != nullptr, "libcuda.so.1 is already loaded");
  if (!g_driver) return 1;
  const auto version = reinterpret_cast<CuDriverGetVersion>(dlsym(g_driver, "cuDriverGetVersion"));
  const auto get_ctx = reinterpret_cast<CuCtxGetCurrent>(dlsym(g_driver, "cuCtxGetCurrent"));
  const auto range = reinterpret_cast<CuMemGetAddressRange>(dlsym(g_driver, "cuMemGetAddressRange_v2"));
  const auto state = reinterpret_cast<CuDevicePrimaryCtxGetState>(dlsym(g_driver, "cuDevicePrimaryCtxGetState"));
  if (!version || !get_ctx || !range || !state) return 2;
  int v = 0;
  check(version(&v) == 0 && v > 0, "cuDriverGetVersion");
  void* ctx = nullptr;
  check(get_ctx(&ctx) == 0 && ctx != nullptr, "cuCtxGetCurrent without a cuInit of the program's own: success, the primary context");
  unsigned long long base = 0;
  size_t size = 0;
  check(range(&base, &size, reinterpret_cast<unsigned long long>(p)) == 0 && base == reinterpret_cast<unsigned long long>(p) &&
            size >= 4096,
        "cuMemGetAddressRange of the runtime's allocation");
  unsigned flags = 0;
  int active = 0;
  check(state(0, &flags, &active) == 0 && active == 1, "cuDevicePrimaryCtxGetState: the runtime's primary context is active");

  pthread_t t;
  if (pthread_create(&t, nullptr, worker, nullptr) == 0) pthread_join(t, nullptr);

  // cudaGetDriverEntryPoint gives the driver's own function, and leaves the driver where it was: out of the global scope.
  void* fn = nullptr;
  int status = -1;
  check(entry("cuCtxGetCurrent", &fn, 0, &status) == 0 && fn == reinterpret_cast<void*>(get_ctx) && status == 0,
        "cudaGetDriverEntryPoint(cuCtxGetCurrent) is the driver's function");
  check(dlsym(RTLD_DEFAULT, "cuInit") == nullptr, "the driver stays out of the global scope");

  check(cuda_free(p) == 0, "cudaFree");
  dlclose(g_driver);
  std::printf("%s\n", g_fails == 0 ? "ok" : "failed");
  return g_fails == 0 ? 0 : 1;
}
