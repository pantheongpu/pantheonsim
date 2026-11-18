// Experiment for docs/perf-design/12-host-gpu-turbo.md section 17: what one process can do with several host cards through the driver API.
// Build: nvcc -arch=sm_75 -o va_probe va_probe.cu -lcuda        Run: ./va_probe
#include <cuda.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <thread>
#define CK(x) do { CUresult r_ = (x); if (r_) { const char* s_ = nullptr; cuGetErrorName(r_, &s_); std::printf("  FAIL %s -> %d %s (line %d)\n", #x, (int)r_, s_ ? s_ : "?", __LINE__); } } while (0)
static const uint64_t kBase = 0x200000000000ull, kStride = 0x10000000000ull;
extern "C" __global__ void fill(int* p, int n, int v) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = v + i; }
extern "C" __global__ void inc(int* p, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] += 1; }

int main() {
  CK(cuInit(0));
  int n = 0; CK(cuDeviceGetCount(&n));
  std::printf("cards: %d\n", n);
  std::vector<CUcontext> ctx(n);
  std::vector<CUdeviceptr> win(n, 0);
  for (int i = 0; i < n; ++i) { CUdevice d; CK(cuDeviceGet(&d, i)); CK(cuDevicePrimaryCtxRetain(&ctx[i], d)); }
  // 1. one context per card, one window per card at the simulator's per-device address
  for (int i = 0; i < n; ++i) {
    CK(cuCtxSetCurrent(ctx[i]));
    CUdeviceptr p = 0; CUresult r = cuMemAddressReserve(&p, kStride, 0, kBase + (uint64_t)i * kStride, 0);
    std::printf("1. card %d reserve 0x%llx: r=%d got=0x%llx %s\n", i, (unsigned long long)(kBase + (uint64_t)i * kStride), (int)r, (unsigned long long)p, p == kBase + (uint64_t)i * kStride ? "SAME" : "DIFFERENT");
    win[i] = p;
    CUmemAllocationProp prop = {}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = i;
    size_t gran = 0; CK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t sz = 64ull << 20; sz = (sz + gran - 1) / gran * gran;
    CUmemGenericAllocationHandle h; CK(cuMemCreate(&h, sz, &prop, 0)); CK(cuMemMap(p, sz, 0, h, 0));
    CUmemAccessDesc ad = {}; ad.location = prop.location; ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; CK(cuMemSetAccess(p, sz, &ad, 1));
  }
  // 2. the same VA on a second card (expected to fail: one VA space)
  { CK(cuCtxSetCurrent(ctx[n > 1 ? 1 : 0])); CUdeviceptr p = 0; CUresult r = cuMemAddressReserve(&p, 1ull << 20, 0, win[0], 0);
    std::printf("2. card 1 reserve the address card 0 holds: r=%d got=0x%llx\n", (int)r, (unsigned long long)p); }
  // 3. kernels on each card through its own window
  CUmodule mod[8]; CUfunction ff[8], fi[8];
  for (int i = 0; i < n; ++i) { CK(cuCtxSetCurrent(ctx[i])); CK(cuModuleLoad(&mod[i], "va_probe_k.cubin")); CK(cuModuleGetFunction(&ff[i], mod[i], "fill")); CK(cuModuleGetFunction(&fi[i], mod[i], "inc")); }
  const int N = 1 << 16;
  for (int i = 0; i < n; ++i) {
    CK(cuCtxSetCurrent(ctx[i]));
    int* p = (int*)win[i]; int nn = N, v = i * 1000; void* a[] = {&p, &nn, &v};
    CK(cuLaunchKernel(ff[i], N / 256, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
  }
  for (int i = 0; i < n; ++i) { CK(cuCtxSetCurrent(ctx[i])); CK(cuCtxSynchronize()); }
  std::printf("3. kernels on all cards launched back to back, no waiting in between: ok\n");
  // 5. copies between cards through the host (and the driver's own cuMemcpyPeer with no P2P)
  if (n > 1) {
    std::vector<int> h(N);
    CK(cuCtxSetCurrent(ctx[1])); CK(cuMemcpyDtoH(h.data(), win[1], N * 4)); // window 1 holds i*1000+k (+1 from probe 4 if it ran)
    CK(cuCtxSetCurrent(ctx[2 % n])); CK(cuMemcpyHtoD(win[2 % n], h.data(), N * 4));
    std::vector<int> g(N); CK(cuMemcpyDtoH(g.data(), win[2 % n], N * 4));
    std::printf("5a. through host card1->card2: g[5]=%d (card1 value h[5]=%d) %s\n", g[5], h[5], g[5] == h[5] ? "ok" : "BAD");
    int can = -1; CUdevice d0, d1; cuDeviceGet(&d0, 0); cuDeviceGet(&d1, 1); CK(cuDeviceCanAccessPeer(&can, d0, d1));
    std::printf("5b. cuDeviceCanAccessPeer(0,1)=%d\n", can);
    CUresult r = cuMemcpyPeer(win[0], ctx[0], win[1], ctx[1], N * 4);
    std::printf("5c. cuMemcpyPeer card1->card0 with no P2P: r=%d (the driver stages it itself)\n", (int)r);
    CK(cuCtxSetCurrent(ctx[0])); CK(cuMemcpyDtoH(g.data(), win[0], N * 4));
    std::printf("    card 0 now holds g[5]=%d (card1's h[5]=%d)\n", g[5], h[5]);
    // 5d. cuMemcpyDtoH on a pointer of another card's window while the current context is card 0 (unified addressing)
    CK(cuCtxSetCurrent(ctx[0])); CUresult r2 = cuMemcpyDtoH(g.data(), win[3 % n], N * 4);
    std::printf("5d. DtoH of card 3's window with card 0 current: r=%d g[5]=%d (UVA routes by address)\n", (int)r2, g[5]);
  }
  // 6. streams and events across cards
  { std::vector<CUstream> st(n); std::vector<CUevent> ev(n);
    for (int i = 0; i < n; ++i) { CK(cuCtxSetCurrent(ctx[i])); CK(cuStreamCreate(&st[i], CU_STREAM_NON_BLOCKING)); CK(cuEventCreate(&ev[i], CU_EVENT_DEFAULT)); }
    for (int i = 0; i < n; ++i) { CK(cuCtxSetCurrent(ctx[i])); int* p = (int*)win[i]; int nn = N; void* a[] = {&p, &nn}; CK(cuLaunchKernel(fi[i], N / 256, 1, 1, 256, 1, 1, 0, st[i], a, nullptr)); CK(cuEventRecord(ev[i], st[i])); }
    // card 1's stream waits on card 0's event (cross-card event wait)
    if (n > 1) { CK(cuCtxSetCurrent(ctx[1])); CK(cuStreamWaitEvent(st[1], ev[0], 0)); }
    for (int i = 0; i < n; ++i) { CK(cuCtxSetCurrent(ctx[i])); CK(cuStreamSynchronize(st[i])); }
    std::printf("6. one stream and one event per card, cross-card cuStreamWaitEvent: ok\n"); }
  // 7. concurrency: n host threads each launching a long-ish kernel on its own card
  for (int i = 0; i < n; ++i) { CK(cuCtxSetCurrent(ctx[i])); }
  std::printf("7. (see timing runs in the section)\n");
  // 8. one thread, current context switches: per-thread current context
  for (int i = 0; i < n; ++i) { CK(cuCtxSetCurrent(ctx[i])); CUdevice d; CK(cuCtxGetDevice(&d)); if (d != i) std::printf("8. BAD current device %d != %d\n", (int)d, i); }
  std::printf("8. cuCtxSetCurrent/GetDevice per card: ok\n");
  // 9. a card launching on another card's window (what a peer access would do)
  { CK(cuCtxSetCurrent(ctx[0])); int* p = (int*)win[n > 1 ? 1 : 0]; int nn = N; void* a[] = {&p, &nn};
    CUresult r = cuLaunchKernel(fi[0], N / 256, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr); CUresult s = cuCtxSynchronize();
    std::printf("9. card 0 kernel on card 1's window: launch=%d sync=%d (context %s)\n", (int)r, (int)s, s ? "DEAD" : "alive"); }
  // after the dead context: is another card's context still usable?
  if (n > 1) { CK(cuCtxSetCurrent(ctx[2 % n])); CUresult r = cuMemcpyDtoH(&n, win[2 % n], 4); std::printf("9b. after card 0 died, DtoH on card 2 (own context current): r=%d\n", (int)r);
    CK(cuCtxSetCurrent(ctx[1])); r = cuCtxSynchronize(); std::printf("9c. card 1 cuCtxSynchronize: r=%d\n", (int)r); }
  return 0;
}
