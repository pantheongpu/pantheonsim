// E5c: what a driver-API passthrough would pay per operation on the real A10G, and the same calls on the simulator's libcuda.
// (1) cuModuleLoadData of a cubin + cuLaunchKernel of an empty kernel + sync, (2) allocation: cuMemAlloc vs the VMM sequence that
// places memory at a CHOSEN address (cuMemAddressReserve + cuMemCreate + cuMemMap + cuMemSetAccess), the way pointer values of the
// simulator could be kept on the real device, (3) 4 KiB and 1 MiB copies each way.
// Build: nvcc -arch=sm_86 -O2 -cudart shared e5_driver.cu -o e5_driver -lcuda ; kernel cubin: nvcc -arch=sm_86 -cubin empty.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <time.h>
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define DK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s = "?"; cuGetErrorName(r_, &s); printf("FAIL %d %s -> %s\n", __LINE__, #x, s); return 1; } } while (0)
int main(int argc, char** argv) {
  const char* cubin = argc > 1 ? argv[1] : "empty.cubin"; int reps = 300;
  DK(cuInit(0)); CUdevice dev; DK(cuDeviceGet(&dev, 0)); CUcontext ctx; DK(cuDevicePrimaryCtxRetain(&ctx, dev)); DK(cuCtxSetCurrent(ctx));
  FILE* f = fopen(cubin, "rb"); if (!f) { printf("no %s\n", cubin); return 1; } std::vector<char> img(1 << 20); size_t n = fread(img.data(), 1, img.size(), f); fclose(f);
  double t0 = now(); CUmodule mod; DK(cuModuleLoadData(&mod, img.data())); CUfunction fn; DK(cuModuleGetFunction(&fn, mod, "empty_k")); double t1 = now();
  printf("module_load+getfunction_us %.1f\n", (t1 - t0) * 1e6);
  for (int i = 0; i < 20; ++i) DK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, nullptr, nullptr)); DK(cuCtxSynchronize());
  t0 = now(); for (int i = 0; i < reps; ++i) { DK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, nullptr, nullptr)); DK(cuCtxSynchronize()); } t1 = now();
  printf("cuLaunchKernel+sync_us %.2f\n", (t1 - t0) / reps * 1e6);
  t0 = now(); for (int i = 0; i < reps; ++i) DK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, nullptr, nullptr)); DK(cuCtxSynchronize()); t1 = now();
  printf("cuLaunchKernel_pipelined_us %.2f\n", (t1 - t0) / reps * 1e6);
  // allocation
  const int na = 100; CUdeviceptr p[na];
  t0 = now(); for (int i = 0; i < na; ++i) DK(cuMemAlloc(&p[i], 1 << 20)); t1 = now(); printf("cuMemAlloc_1MiB_us %.2f\n", (t1 - t0) / na * 1e6);
  t0 = now(); for (int i = 0; i < na; ++i) DK(cuMemFree(p[i])); t1 = now(); printf("cuMemFree_1MiB_us %.2f\n", (t1 - t0) / na * 1e6);
  CUmemAllocationProp prop = {}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
  CUmemAccessDesc ad = {}; ad.location = prop.location; ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  const size_t sz = 2 << 20; CUmemGenericAllocationHandle h[na]; CUdeviceptr va[na]; int exact = 0, nv = 0; bool vmm_ok = true;
  t0 = now();
  for (int i = 0; i < na && vmm_ok; ++i) {
    CUresult r = cuMemAddressReserve(&va[i], sz, 0, 0x200000000000ull + (unsigned long long)i * sz, 0);
    if (r != CUDA_SUCCESS) { const char* e = "?"; cuGetErrorName(r, &e); printf("cuMemAddressReserve(fixed hint) -> %s\n", e); vmm_ok = false; break; }
    exact += va[i] == 0x200000000000ull + (unsigned long long)i * sz;
    if (cuMemCreate(&h[i], sz, &prop, 0) != CUDA_SUCCESS || cuMemMap(va[i], sz, 0, h[i], 0) != CUDA_SUCCESS || cuMemSetAccess(va[i], sz, &ad, 1) != CUDA_SUCCESS) { printf("vmm create/map failed\n"); vmm_ok = false; break; }
    ++nv;
  }
  t1 = now();
  if (nv) printf("vmm_reserve+create+map+access_2MiB_us %.2f (hint honoured exactly %d of %d)\n", (t1 - t0) / nv * 1e6, exact, nv);
  t0 = now(); for (int i = 0; i < nv; ++i) { cuMemUnmap(va[i], sz); cuMemRelease(h[i]); cuMemAddressFree(va[i], sz); } t1 = now(); if (nv) printf("vmm_unmap+release+free_us %.2f\n", (t1 - t0) / nv * 1e6);
  // copies
  for (size_t bytes : {4096ul, 1ul << 20, 64ul << 20}) {
    std::vector<char> hbuf(bytes, 1); CUdeviceptr d; DK(cuMemAlloc(&d, bytes)); int it = bytes > (1 << 20) ? 10 : reps;
    DK(cuMemcpyHtoD(d, hbuf.data(), bytes));
    t0 = now(); for (int i = 0; i < it; ++i) DK(cuMemcpyHtoD(d, hbuf.data(), bytes)); t1 = now(); double h2d = (t1 - t0) / it;
    t0 = now(); for (int i = 0; i < it; ++i) DK(cuMemcpyDtoH(hbuf.data(), d, bytes)); t1 = now(); double d2h = (t1 - t0) / it;
    printf("copy %zu B: H2D_us %.1f D2H_us %.1f\n", bytes, h2d * 1e6, d2h * 1e6); cuMemFree(d);
  }
  return 0;
}
