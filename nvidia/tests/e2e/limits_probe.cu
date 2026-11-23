// Device limits and the access policy window as a card answers them: one line a call, printed. The expected lines are recorded from a Tesla T4
// and an A10G (nvidia/tests/data/card/limits.t4.txt, limits.a10g.txt: driver 595.91, CUDA 13.5 runtime, AWS g4dn.xlarge and g5.xlarge,
// 2026-11-23); run_limits_card.sh runs this on the t4 and a10g profiles and compares. Build: nvcc -arch=sm_75 -cudart shared limits_probe.cu
#include <cuda_runtime.h>
#include <cstdio>

static void show(const char* what, size_t v, cudaError_t e) { printf("%-44s %s %zu\n", what, cudaGetErrorName(e), v); }

static void persisting_and_window() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("%s l2 %d persistMax %d accessWindowMax %d\n", p.name, p.l2CacheSize, p.persistingL2CacheMaxSize, p.accessPolicyMaxWindowSize);
  size_t v = 999; cudaError_t e = cudaDeviceGetLimit(&v, cudaLimitPersistingL2CacheSize); show("get default", v, e);
  size_t asks[] = {4u<<20, 1u<<20, 1000, (size_t)p.persistingL2CacheMaxSize, (size_t)p.persistingL2CacheMaxSize + 1, (size_t)p.persistingL2CacheMaxSize * 2, (size_t)p.l2CacheSize, 0};
  for (size_t a : asks) {
    char b[64]; snprintf(b, 64, "set %zu", a);
    e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, a); show(b, 0, e);
    v = 999; e = cudaDeviceGetLimit(&v, cudaLimitPersistingL2CacheSize); show("  get", v, e);
  }
  e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 1u<<20);
  e = cudaCtxResetPersistingL2Cache(); show("reset", 0, e);
  v = 999; e = cudaDeviceGetLimit(&v, cudaLimitPersistingL2CacheSize); show("get after reset", v, e);
  e = cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 64); v=999; cudaDeviceGetLimit(&v, cudaLimitMaxL2FetchGranularity); show("set fetch gran 64 -> get", v, e);
  cudaStream_t s; cudaStreamCreate(&s);
  void* d; cudaMalloc(&d, 64u<<20);
  cudaStreamAttrValue a = {};
  a.accessPolicyWindow.base_ptr = d; a.accessPolicyWindow.num_bytes = 1u<<20; a.accessPolicyWindow.hitRatio = 0.6f;
  a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  e = cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &a); show("window 1MiB", 0, e);
  a.accessPolicyWindow.num_bytes = (size_t)p.accessPolicyMaxWindowSize + 1;
  e = cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &a); show("window max+1", 0, e);
}

static void limits() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  struct { const char* n; cudaLimit l; } L[] = {{"stack", cudaLimitStackSize}, {"printf", cudaLimitPrintfFifoSize}, {"heap", cudaLimitMallocHeapSize}, {"syncdepth", cudaLimitDevRuntimeSyncDepth}, {"pending", cudaLimitDevRuntimePendingLaunchCount}, {"fetch", cudaLimitMaxL2FetchGranularity}};
  printf("%s sm_%d%d\n", p.name, p.major, p.minor);
  for (auto& x : L) { size_t v = 777; cudaError_t e = cudaDeviceGetLimit(&v, x.l); printf("%-10s %s %zu\n", x.n, cudaGetErrorName(e), v); }
  size_t v; cudaDeviceSetLimit(cudaLimitStackSize, 1000); cudaDeviceGetLimit(&v, cudaLimitStackSize); printf("stack after 1000: %zu\n", v);
  cudaError_t e = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 100); cudaDeviceGetLimit(&v, cudaLimitDevRuntimeSyncDepth); printf("syncdepth set 100: %s -> %zu\n", cudaGetErrorName(e), v);
  e = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1<<22); cudaDeviceGetLimit(&v, cudaLimitPrintfFifoSize); printf("printf set 4MiB: %s -> %zu\n", cudaGetErrorName(e), v);
  e = cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 4096); cudaDeviceGetLimit(&v, cudaLimitMaxL2FetchGranularity); printf("fetch set 4096: %s -> %zu\n", cudaGetErrorName(e), v);
}

int main() {
  persisting_and_window();
  limits();
  return 0;
}
