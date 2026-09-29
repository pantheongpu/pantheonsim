// libcuda and libcudart loaded privately (RTLD_LOCAL), one after the other:
// memory the driver API allocated is visible to the runtime API.
//
// Rust's libloading loads libraries this way, and cudarc loads libcuda first
// and libcudart (under libcurand) later. Each library carries the simulator's
// core, and each found only its own machine, so cuRAND's runtime-API copy went
// to an address only libcuda had allocated: "not inside any device
// allocation". No CUDA headers: everything is looked up by name, as cudarc does.
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

template <class F> F sym(void* lib, const char* name) {
  void* p = dlsym(lib, name);
  if (!p) std::printf("FAIL: %s not found\n", name);
  return reinterpret_cast<F>(p);
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  void* drv = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!drv) { std::printf("FAIL: libcuda.so.1: %s\n", dlerror()); return 1; }
  auto cuInit = sym<int (*)(unsigned)>(drv, "cuInit");
  auto cuDeviceGet = sym<int (*)(int*, int)>(drv, "cuDeviceGet");
  auto cuRetain = sym<int (*)(void**, int)>(drv, "cuDevicePrimaryCtxRetain");
  auto cuCtxSetCurrent = sym<int (*)(void*)>(drv, "cuCtxSetCurrent");
  auto cuMemAlloc = sym<int (*)(uint64_t*, size_t)>(drv, "cuMemAlloc_v2");
  auto cuMemcpyDtoH = sym<int (*)(void*, uint64_t, size_t)>(drv, "cuMemcpyDtoH_v2");
  if (!cuInit || !cuDeviceGet || !cuRetain || !cuCtxSetCurrent || !cuMemAlloc || !cuMemcpyDtoH) return 1;
  int dev = -1;
  void* ctx = nullptr;
  uint64_t ptr = 0;
  if (cuInit(0) || cuDeviceGet(&dev, 0) || cuRetain(&ctx, dev) || cuCtxSetCurrent(ctx) || cuMemAlloc(&ptr, 16)) {
    std::printf("FAIL: driver setup\n");
    return 1;
  }

  // Loaded after the driver's machine exists, privately, like libcurand's libcudart.
  void* rt = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!rt) { std::printf("FAIL: %s: %s\n", argv[1], dlerror()); return 1; }
  auto cudaMemcpy = sym<int (*)(void*, const void*, size_t, int)>(rt, "cudaMemcpy");
  if (!cudaMemcpy) return 1;
  const int host[4] = {11, 22, 33, 44};
  const int e = cudaMemcpy(reinterpret_cast<void*>(ptr), host, sizeof host, 1 /* HostToDevice */);
  int back[4] = {0, 0, 0, 0};
  const int d = cuMemcpyDtoH(back, ptr, sizeof back);
  if (e || d || std::memcmp(host, back, sizeof host)) {
    std::printf("FAIL: runtime copy into driver memory: cudaMemcpy %d, cuMemcpyDtoH %d, back %d %d %d %d\n", e, d,
                back[0], back[1], back[2], back[3]);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
