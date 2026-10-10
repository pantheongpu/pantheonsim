// Peer access and managed memory on a pair of devices, answer for answer
// (nvidia/tests/data/p2p_managed.rtx3060.expected is what two RTX 3060s printed under NVIDIA's libraries).
//
// A GeForce pair has no peer path: cudaDeviceCanAccessPeer is 0 each way, every attribute
// cudaDeviceGetP2PAttribute names is 0, enabling peer access is cudaErrorPeerAccessUnsupported (the flags are
// looked at first), and a copy between the two still works, through the host. The same host has no concurrent
// managed access (WSL's driver does not page managed memory on demand): cudaMemPrefetchAsync is
// cudaErrorInvalidDevice, and so is the advice that names a device; the advice that does not is accepted and
// forgotten, and cudaMemRangeGetAttribute reads back the defaults.
// The simulator's rtx3060 profile must print the same; a data-centre profile keeps its peer path (the runner
// checks that separately). Lines marked [cuda13] are printed by CUDA 13 builds only (the older headers have no
// flags argument on the prefetch and no location on the advice).
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

static const char* drv(CUresult r) {
  const char* n = nullptr;
  cuGetErrorName(r, &n);
  return n ? n : "?";
}
static void rt(const char* label, cudaError_t e) { std::printf("%-62s %s\n", label, cudaGetErrorName(e)); }
// A call that stores an int through a pointer: the value is printed only when the call succeeded.
#define RTV(label, var, call)                                                                             \
  do {                                                                                                    \
    (var) = -9;                                                                                           \
    const cudaError_t e_ = (call);                                                                        \
    std::printf("%-62s %s %d\n", label, cudaGetErrorName(e_), e_ == cudaSuccess ? (var) : -99);           \
  } while (0)
static void dr(const char* label, CUresult r) { std::printf("%-62s %s\n", label, drv(r)); }

int main() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n < 2) {
    std::printf("SKIP: needs two devices\n");
    return 0;
  }
  // ---- attributes
  const struct { const char* name; int id; } attrs[] = {
      {"ConcurrentManagedAccess", cudaDevAttrConcurrentManagedAccess},
      {"DirectManagedMemAccessFromHost", cudaDevAttrDirectManagedMemAccessFromHost},
      {"PageableMemoryAccess", cudaDevAttrPageableMemoryAccess},
      {"PageableMemoryAccessUsesHostPageTables", cudaDevAttrPageableMemoryAccessUsesHostPageTables},
      {"ManagedMemory", cudaDevAttrManagedMemory},
      {"UnifiedAddressing", cudaDevAttrUnifiedAddressing},
  };
  for (int d = 0; d < 2; ++d)
    for (const auto& a : attrs) {
      int v = -9;
      char label[96];
      std::snprintf(label, sizeof label, "device %d %s", d, a.name);
      RTV(label, v, cudaDeviceGetAttribute(&v, static_cast<cudaDeviceAttr>(a.id), d));
    }
  // ---- peer access
  int v = -9;
  const int pairs[][2] = {{0, 1}, {1, 0}, {0, 0}, {0, 7}, {-1, 0}};
  for (const auto& p : pairs) {
    char label[96];
    std::snprintf(label, sizeof label, "cudaDeviceCanAccessPeer(%d, %d)", p[0], p[1]);
    RTV(label, v, cudaDeviceCanAccessPeer(&v, p[0], p[1]));
  }
  rt("cudaDeviceCanAccessPeer(null, 0, 1)", cudaDeviceCanAccessPeer(nullptr, 0, 1));
  for (int attr = 0; attr <= 7; ++attr)
    for (int dir = 0; dir < 2; ++dir) {
      char label[96];
      std::snprintf(label, sizeof label, "cudaDeviceGetP2PAttribute(%d, %d -> %d)", attr, dir, 1 - dir);
      RTV(label, v, cudaDeviceGetP2PAttribute(&v, static_cast<cudaDeviceP2PAttr>(attr), dir, 1 - dir));
    }
  RTV("cudaDeviceGetP2PAttribute(2, 0 -> 0)", v, cudaDeviceGetP2PAttribute(&v, cudaDevP2PAttrAccessSupported, 0, 0));
  RTV("cudaDeviceGetP2PAttribute(2, 0 -> 7)", v, cudaDeviceGetP2PAttribute(&v, cudaDevP2PAttrAccessSupported, 0, 7));
  rt("cudaDeviceGetP2PAttribute(2, null)", cudaDeviceGetP2PAttribute(nullptr, cudaDevP2PAttrAccessSupported, 0, 1));
  cudaSetDevice(0);
  rt("cudaDeviceEnablePeerAccess(1, 0)", cudaDeviceEnablePeerAccess(1, 0));
  rt("cudaDeviceEnablePeerAccess(1, 0) again", cudaDeviceEnablePeerAccess(1, 0));
  rt("cudaDeviceEnablePeerAccess(1, 1)", cudaDeviceEnablePeerAccess(1, 1));
  rt("cudaDeviceEnablePeerAccess(0, 0) (itself)", cudaDeviceEnablePeerAccess(0, 0));
  rt("cudaDeviceEnablePeerAccess(5, 0)", cudaDeviceEnablePeerAccess(5, 0));
  rt("cudaDeviceDisablePeerAccess(1)", cudaDeviceDisablePeerAccess(1));
  rt("cudaDeviceDisablePeerAccess(0)", cudaDeviceDisablePeerAccess(0));
  rt("cudaDeviceDisablePeerAccess(5)", cudaDeviceDisablePeerAccess(5));
  cudaGetLastError();

  // the driver API's
  CUdevice d0, d1;
  cuInit(0);
  cuDeviceGet(&d0, 0);
  cuDeviceGet(&d1, 1);
  {
    int c = -9;
    CUresult r = cuDeviceCanAccessPeer(&c, d0, d1);
    std::printf("%-62s %s %d\n", "cuDeviceCanAccessPeer(0, 1)", drv(r), c);
    c = -9;
    r = cuDeviceCanAccessPeer(&c, d0, d0);
    std::printf("%-62s %s %d\n", "cuDeviceCanAccessPeer(0, 0)", drv(r), c);
    r = cuDeviceCanAccessPeer(&c, d0, 7);
    std::printf("%-62s %s\n", "cuDeviceCanAccessPeer(0, 7)", drv(r));
  }
  for (int a = 0; a <= 7; ++a) {
    int c = -9;
    char label[96];
    std::snprintf(label, sizeof label, "cuDeviceGetP2PAttribute(%d, 0 -> 1)", a);
    const CUresult r = cuDeviceGetP2PAttribute(&c, static_cast<CUdevice_P2PAttribute>(a), d0, d1);
    std::printf("%-62s %s %d\n", label, drv(r), r == CUDA_SUCCESS ? c : -99);
  }
  CUcontext c0, c1;
  cuDevicePrimaryCtxRetain(&c0, d0);
  cuDevicePrimaryCtxRetain(&c1, d1);
  cuCtxSetCurrent(c0);
  dr("cuCtxEnablePeerAccess(other, 0)", cuCtxEnablePeerAccess(c1, 0));
  dr("cuCtxEnablePeerAccess(other, 0) again", cuCtxEnablePeerAccess(c1, 0));
  dr("cuCtxEnablePeerAccess(itself, 0)", cuCtxEnablePeerAccess(c0, 0));
  dr("cuCtxEnablePeerAccess(other, 1)", cuCtxEnablePeerAccess(c1, 1));
  dr("cuCtxDisablePeerAccess(other)", cuCtxDisablePeerAccess(c1));

  // a copy between the two works without a peer path
  {
    void *a0 = nullptr, *a1 = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&a0, 1 << 16);
    cudaMemset(a0, 0x5a, 1 << 16);
    cudaSetDevice(1);
    cudaMalloc(&a1, 1 << 16);
    cudaMemset(a1, 0, 1 << 16);
    rt("cudaMemcpyPeer(1 <- 0)", cudaMemcpyPeer(a1, 1, a0, 0, 1 << 16));
    unsigned char h[16] = {};
    cudaMemcpy(h, a1, sizeof h, cudaMemcpyDeviceToHost);
    std::printf("%-62s %02x\n", "the first byte after the peer copy", h[0]);
    cudaMemset(a1, 0, 1 << 16);
    rt("cudaMemcpy(1 <- 0, default kind)", cudaMemcpy(a1, a0, 1 << 16, cudaMemcpyDefault));
    cudaMemcpy(h, a1, sizeof h, cudaMemcpyDeviceToHost);
    std::printf("%-62s %02x\n", "the first byte after the default-kind copy", h[0]);
    cudaFree(a1);
    cudaSetDevice(0);
    cudaFree(a0);
  }

  // ---- managed memory, driver API
  // (each API's managed allocations are its own here: the driver's calls take what cuMemAllocManaged made)
  cuCtxSetCurrent(c0);
  cudaSetDevice(0);
  const size_t sz = 1 << 20;
  CUdeviceptr dp = 0, dd = 0;
  cuMemAllocManaged(&dp, sz, CU_MEM_ATTACH_GLOBAL);
  cuMemAlloc(&dd, sz);
#if CUDA_VERSION >= 13000
  CUmemLocation l0{CU_MEM_LOCATION_TYPE_DEVICE, 0}, l1{CU_MEM_LOCATION_TYPE_DEVICE, 1}, l9{CU_MEM_LOCATION_TYPE_DEVICE, 9},
      lh{CU_MEM_LOCATION_TYPE_HOST, 0};
  dr("cuMemPrefetchAsync(managed, device 0)", cuMemPrefetchAsync(dp, sz, l0, 0, nullptr));
  dr("cuMemPrefetchAsync(managed, device 1)", cuMemPrefetchAsync(dp, sz, l1, 0, nullptr));
  dr("cuMemPrefetchAsync(managed, device 9)", cuMemPrefetchAsync(dp, sz, l9, 0, nullptr));
  dr("cuMemPrefetchAsync(managed, host)", cuMemPrefetchAsync(dp, sz, lh, 0, nullptr));
  dr("cuMemPrefetchAsync(managed, host, flags 7) [cuda13]", cuMemPrefetchAsync(dp, sz, lh, 7, nullptr));
  dr("cuMemPrefetchAsync(device memory, device 0)", cuMemPrefetchAsync(dd, sz, l0, 0, nullptr));
  dr("cuMemPrefetchAsync(null, device 0)", cuMemPrefetchAsync(0, sz, l0, 0, nullptr));
  dr("cuMemPrefetchAsync(managed, size 0)", cuMemPrefetchAsync(dp, 0, l0, 0, nullptr));
  dr("cuMemPrefetchAsync(managed + 1, 100)", cuMemPrefetchAsync(dp + 1, 100, l0, 0, nullptr));
  dr("cuMemAdvise read-mostly (device 0)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_READ_MOSTLY, l0));
  dr("cuMemAdvise unset read-mostly (device 0)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_UNSET_READ_MOSTLY, l0));
  dr("cuMemAdvise preferred location (host)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, lh));
  dr("cuMemAdvise preferred location (device 0)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, l0));
  dr("cuMemAdvise preferred location (device 1)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, l1));
  dr("cuMemAdvise unset preferred location (device 0)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, l0));
  dr("cuMemAdvise accessed by (host)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_ACCESSED_BY, lh));
  dr("cuMemAdvise accessed by (device 1)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_ACCESSED_BY, l1));
  dr("cuMemAdvise unset accessed by (device 0)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_UNSET_ACCESSED_BY, l0));
  dr("cuMemAdvise accessed by (device 9)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_ACCESSED_BY, l9));
  dr("cuMemAdvise read-mostly (device memory)", cuMemAdvise(dd, sz, CU_MEM_ADVISE_SET_READ_MOSTLY, l0));
  dr("cuMemAdvise preferred location (device memory, device 0)", cuMemAdvise(dd, sz, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, l0));
  dr("cuMemAdvise 99", cuMemAdvise(dp, sz, static_cast<CUmem_advise>(99), l0));
#else
  dr("cuMemPrefetchAsync(managed, device 0)", cuMemPrefetchAsync(dp, sz, 0, nullptr));
  dr("cuMemPrefetchAsync(managed, device 1)", cuMemPrefetchAsync(dp, sz, 1, nullptr));
  dr("cuMemPrefetchAsync(managed, device 9)", cuMemPrefetchAsync(dp, sz, 9, nullptr));
  dr("cuMemPrefetchAsync(managed, host)", cuMemPrefetchAsync(dp, sz, CU_DEVICE_CPU, nullptr));
  dr("cuMemPrefetchAsync(device memory, device 0)", cuMemPrefetchAsync(dd, sz, 0, nullptr));
  dr("cuMemPrefetchAsync(null, device 0)", cuMemPrefetchAsync(0, sz, 0, nullptr));
  dr("cuMemPrefetchAsync(managed, size 0)", cuMemPrefetchAsync(dp, 0, 0, nullptr));
  dr("cuMemPrefetchAsync(managed + 1, 100)", cuMemPrefetchAsync(dp + 1, 100, 0, nullptr));
  dr("cuMemAdvise read-mostly (device 0)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_READ_MOSTLY, 0));
  dr("cuMemAdvise unset read-mostly (device 0)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_UNSET_READ_MOSTLY, 0));
  dr("cuMemAdvise preferred location (host)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, CU_DEVICE_CPU));
  dr("cuMemAdvise preferred location (device 0)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, 0));
  dr("cuMemAdvise preferred location (device 1)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, 1));
  dr("cuMemAdvise unset preferred location (device 0)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, 0));
  dr("cuMemAdvise accessed by (host)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_ACCESSED_BY, CU_DEVICE_CPU));
  dr("cuMemAdvise accessed by (device 1)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_ACCESSED_BY, 1));
  dr("cuMemAdvise unset accessed by (device 0)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_UNSET_ACCESSED_BY, 0));
  dr("cuMemAdvise accessed by (device 9)", cuMemAdvise(dp, sz, CU_MEM_ADVISE_SET_ACCESSED_BY, 9));
  dr("cuMemAdvise read-mostly (device memory)", cuMemAdvise(dd, sz, CU_MEM_ADVISE_SET_READ_MOSTLY, 0));
  dr("cuMemAdvise preferred location (device memory, device 0)", cuMemAdvise(dd, sz, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, 0));
  dr("cuMemAdvise 99", cuMemAdvise(dp, sz, static_cast<CUmem_advise>(99), 0));
#endif

  // ---- managed memory, runtime API (CUDA 13's signatures)
#if CUDART_VERSION >= 13000
  void *m = nullptr, *dm = nullptr;
  cudaMallocManaged(&m, sz, cudaMemAttachGlobal);
  cudaMalloc(&dm, sz);
  const cudaMemLocation r0{cudaMemLocationTypeDevice, 0}, r1{cudaMemLocationTypeDevice, 1}, rh{cudaMemLocationTypeHost, 0};
  rt("cudaMemPrefetchAsync(managed, device 0) [cuda13]", cudaMemPrefetchAsync(m, sz, r0, 0, nullptr));
  rt("cudaMemPrefetchAsync(managed, host) [cuda13]", cudaMemPrefetchAsync(m, sz, rh, 0, nullptr));
  rt("cudaMemPrefetchAsync(managed, host, flags 7) [cuda13]", cudaMemPrefetchAsync(m, sz, rh, 7, nullptr));
  rt("cudaMemPrefetchAsync(device memory, device 0) [cuda13]", cudaMemPrefetchAsync(dm, sz, r0, 0, nullptr));
  rt("cudaMemAdvise read-mostly (device 0) [cuda13]", cudaMemAdvise(m, sz, cudaMemAdviseSetReadMostly, r0));
  rt("cudaMemAdvise preferred location (host) [cuda13]", cudaMemAdvise(m, sz, cudaMemAdviseSetPreferredLocation, rh));
  rt("cudaMemAdvise preferred location (device 0) [cuda13]", cudaMemAdvise(m, sz, cudaMemAdviseSetPreferredLocation, r0));
  rt("cudaMemAdvise unset preferred location (device 0) [cuda13]", cudaMemAdvise(m, sz, cudaMemAdviseUnsetPreferredLocation, r0));
  rt("cudaMemAdvise accessed by (device 1) [cuda13]", cudaMemAdvise(m, sz, cudaMemAdviseSetAccessedBy, r1));
  rt("cudaMemAdvise accessed by (host) [cuda13]", cudaMemAdvise(m, sz, cudaMemAdviseSetAccessedBy, rh));
  rt("cudaMemAdvise read-mostly (device memory) [cuda13]", cudaMemAdvise(dm, sz, cudaMemAdviseSetReadMostly, r0));
  const struct { const char* name; int attr; } ranges[] = {{"read-mostly", cudaMemRangeAttributeReadMostly},
                                                           {"preferred location", cudaMemRangeAttributePreferredLocation},
                                                           {"last prefetch location", cudaMemRangeAttributeLastPrefetchLocation}};
  for (const auto& r : ranges) {
    int x = -9;
    char label[96];
    std::snprintf(label, sizeof label, "cudaMemRangeGetAttribute %s [cuda13]", r.name);
    RTV(label, x, cudaMemRangeGetAttribute(&x, sizeof x, static_cast<cudaMemRangeAttribute>(r.attr), m, sz));
  }
  {
    int ab[2] = {-9, -9};
    const cudaError_t e = cudaMemRangeGetAttribute(ab, sizeof ab, cudaMemRangeAttributeAccessedBy, m, sz);
    std::printf("%-62s %s %d %d\n", "cudaMemRangeGetAttribute accessed-by [cuda13]", cudaGetErrorName(e), ab[0], ab[1]);
  }
  cudaFree(m);
  cudaFree(dm);
#endif
  cuMemFree(dd);
  cuMemFree(dp);
  std::printf("PASS\n");
  return 0;
}
