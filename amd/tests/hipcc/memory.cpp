// HIP's copies, fills and allocations of every shape, answered as ROCm's HIP
// answers them on an MI300-class device: what AMD's own tests (hip-tests)
// hold it to. Each check prints "ok <what>" or "FAIL <what>: <why>", and the
// last line counts them. Built by build.sh with hipcc; run on two simulated
// MI300Xs by amd/tests/e2e/run_hipcc.sh.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <string>
#include <vector>

static int checks = 0, failures = 0;
static void check(bool ok, const char* what, const std::string& why = "") {
  ++checks;
  if (ok) {
    std::printf("ok    %s\n", what);
  } else {
    ++failures;
    std::printf("FAIL  %s%s%s\n", what, why.empty() ? "" : ": ", why.c_str());
  }
}
static std::string err(hipError_t e) { return hipGetErrorName(e); }
#define EXPECT(call, want, what) \
  do { \
    const hipError_t got_ = (call); \
    check(got_ == (want), what, "got " + err(got_) + ", want " + err(want)); \
  } while (0)

__global__ void delay(uint64_t ms, uint64_t ticks_per_ms) {
  for (uint64_t m = 0; m < ms; ++m) {
    const uint64_t start = wall_clock64();
    while (wall_clock64() - start < ticks_per_ms) __builtin_amdgcn_s_sleep(10);
  }
}
__global__ void add_double(double* p, double v) { atomicAdd(p, v); }
__global__ void iota3d(uint8_t* p, size_t pitch, size_t height, size_t w, size_t h, size_t d) {
  for (size_t z = 0; z < d; ++z)
    for (size_t y = 0; y < h; ++y)
      for (size_t x = threadIdx.x; x < w; x += blockDim.x) p[z * pitch * height + y * pitch + x] = uint8_t(x + 3 * y + 7 * z);
}

int main() {
  int rate = 0;
  (void)hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0);

  // ---- Pitched memory: rows rounded to 256 bytes, the size exactly theirs
  void* p = nullptr;
  size_t pitch = 0, size = 0;
  (void)hipMallocPitch(&p, &pitch, 260, 2);
  (void)hipMemPtrGetInfo(p, &size);
  check(pitch == 512 && size == 1024, "a pitched row is rounded to 256 bytes, and the allocation is exactly its rows",
        std::to_string(pitch) + " " + std::to_string(size));
  (void)hipFree(p);
  EXPECT(hipMallocPitch(&p, &pitch, 1, SIZE_MAX), hipErrorOutOfMemory, "a pitched allocation too big to count");
  p = reinterpret_cast<void*>(1);
  (void)hipMallocPitch(&p, &pitch, 0, 1);
  check(p == nullptr && pitch == 0, "no bytes is no allocation, and no pitch");
  hipPitchedPtr pp{};
  (void)hipMalloc3D(&pp, make_hipExtent(260, 16, 8));
  check(pp.pitch == 512 && pp.xsize == 260 && pp.ysize == 16, "hipMalloc3D gives the pitch and the extent back");

  // ---- A 3D copy with offsets, through a kernel's pattern and back
  const size_t w = 100, h = 10, d = 4;
  iota3d<<<1, 64>>>(static_cast<uint8_t*>(pp.ptr), pp.pitch, 16, w, h, d);
  std::vector<uint8_t> host(w * h * d, 0);
  hipMemcpy3DParms c{};
  c.srcPtr = pp;
  c.dstPtr = make_hipPitchedPtr(host.data(), w, w, h);
  c.extent = make_hipExtent(w, h, d);
  c.kind = hipMemcpyDeviceToHost;
  (void)hipMemcpy3D(&c);
  bool same = true;
  for (size_t z = 0; z < d; ++z)
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x) same &= host[z * w * h + y * w + x] == uint8_t(x + 3 * y + 7 * z);
  check(same, "a 3D copy reads each slice at its pointer's pitch times its rows");
  std::vector<uint8_t> part(10 * 2 * 2, 0);
  c.srcPos = make_hipPos(5, 3, 1);
  c.dstPtr = make_hipPitchedPtr(part.data(), 10, 10, 2);
  c.extent = make_hipExtent(10, 2, 2);
  (void)hipMemcpy3D(&c);
  check(part[0] == uint8_t(5 + 9 + 7) && part[10 * 2 + 10 + 9] == uint8_t(14 + 12 + 14),
        "and starts where its position says");
  c.srcPtr.pitch = INT32_MAX;
  EXPECT(hipMemcpy3D(&c), hipErrorInvalidValue, "a pitch as wide as the widest there is, is refused");
  EXPECT(hipMemcpy3D(nullptr), hipErrorInvalidValue, "and no parameters at all");

  // ---- The driver API's 2D and 3D copies: sides by memory type
  std::vector<uint8_t> row(512 * 3, 9);   // three rows of it
  hip_Memcpy2D m{};
  m.srcMemoryType = hipMemoryTypeHost;
  m.srcHost = row.data();
  m.srcPitch = 512;
  m.dstMemoryType = hipMemoryTypeDevice;
  m.dstDevice = pp.ptr;
  m.dstPitch = pp.pitch;
  m.WidthInBytes = 200;
  m.Height = 3;
  EXPECT(hipMemcpyParam2D(&m), hipSuccess, "hipMemcpyParam2D, host to device");
  m.dstPitch = 199;
  EXPECT(hipMemcpyParam2D(&m), hipErrorInvalidValue, "a pitch narrower than the row is refused");
  m.dstPitch = pp.pitch;
  m.dstMemoryType = hipMemoryTypeArray;
  m.dstArray = nullptr;
  EXPECT(hipMemcpyParam2D(&m), hipErrorInvalidValue, "and a side named as an array that is null");
  HIP_MEMCPY3D m3{};
  m3.srcMemoryType = hipMemoryTypeDevice;
  m3.srcDevice = pp.ptr;
  m3.srcPitch = pp.pitch;
  m3.dstMemoryType = hipMemoryTypeHost;
  m3.dstHost = host.data();
  m3.dstPitch = w;
  m3.WidthInBytes = 200 < w ? 200 : w;
  m3.Height = 3;
  m3.Depth = 1;
  (void)hipDrvMemcpy3D(&m3);
  check(host[0] == 9 && host[w * 2 + 50] == 9, "hipDrvMemcpy3D reads back what hipMemcpyParam2D wrote");

  // ---- Memsets of every width
  uint32_t* words = nullptr;
  (void)hipMalloc(&words, 1024 * sizeof(uint32_t));
  (void)hipMemsetD32(words, int(0xDEADBEEF), 1024);
  (void)hipMemsetD16(words, 0xBEEF, 2);   // the first word's two halves
  (void)hipMemsetD8(reinterpret_cast<uint8_t*>(words) + 4, 0x11, 1);
  std::vector<uint32_t> back(1024);
  (void)hipMemcpy(back.data(), words, back.size() * 4, hipMemcpyDeviceToHost);
  check(back[0] == 0xBEEFBEEF && back[1] == 0xDEADBE11 && back[1023] == 0xDEADBEEF, "D8, D16 and D32 set what they say");
  size_t dpitch = 0;
  void* plane = nullptr;
  (void)hipMemAllocPitch(&plane, &dpitch, 64 * 4, 8, 4);
  (void)hipMemsetD2D32(plane, dpitch, 0x01020304, 64, 8);
  std::vector<uint32_t> plane_host(dpitch / 4 * 8);
  (void)hipMemcpy(plane_host.data(), plane, dpitch * 8, hipMemcpyDeviceToHost);
  check(plane_host[0] == 0x01020304 && plane_host[dpitch / 4 * 7 + 63] == 0x01020304, "hipMemsetD2D32 fills each row");
  (void)hipMemset3D(pp, 0x5A, make_hipExtent(10, 2, 2));
  (void)hipMemcpy3D(&c);
  check(part[0] == uint8_t(5 + 9 + 7), "hipMemset3D leaves what is outside its box");
  EXPECT(hipMemset2D(words, 1, 0, 184, 2), hipErrorInvalidValue, "a 2D memset whose pitch is narrower than its rows");
  uint8_t* small = nullptr;
  (void)hipMalloc(&small, 184);
  EXPECT(hipMemset(small + 185, 0, 8), hipErrorInvalidValue, "a memset outside every allocation");
  EXPECT(hipMemsetAsync(words, 0, 8192, nullptr), hipErrorInvalidValue, "and one past its end, when it is asked for");
  (void)hipGetLastError();

  // ---- Who waits: a synchronous memset or copy of device memory does not
  hipStream_t slow;
  (void)hipStreamCreate(&slow);
  delay<<<1, 1, 0, slow>>>(500, rate);
  (void)hipMemset(words, 0, 16);
  check(hipStreamQuery(slow) == hipErrorNotReady, "a memset of device memory returns before the device is done");
  (void)hipStreamSynchronize(slow);
  int* pinned = nullptr;
  (void)hipHostMalloc(&pinned, 64);
  delay<<<1, 1, 0, slow>>>(500, rate);
  (void)hipMemset(pinned, 0, 16);
  check(hipStreamQuery(slow) == hipSuccess, "one of host memory waits, for what the null stream waits for too");

  // ---- Driver-style and peer copies, by where the addresses are
  std::vector<uint32_t> src(256, 7);
  (void)hipMemcpyHtoD(words, src.data(), 1024);
  EXPECT(hipMemcpyDtoH(back.data(), words, 8192), hipErrorInvalidValue, "a copy past its allocation is refused");
  (void)hipGetLastError();
  uint32_t* other = nullptr;
  (void)hipSetDevice(1);
  (void)hipMalloc(&other, 1024);
  (void)hipSetDevice(0);
  (void)hipMemcpyPeer(other, 1, words, 0, 1024);
  (void)hipMemcpyDtoH(back.data(), other, 1024);
  check(back[255] == 7, "hipMemcpyPeer copies between devices");
  EXPECT(hipMemcpyPeer(other, 2, words, 0, 1024), hipErrorInvalidDevice, "to a device that is not there, it refuses");

  // ---- Managed memory: advice and prefetches, page by page
  char* managed = nullptr;
  (void)hipMallocManaged(&managed, 3 * 4096);
  (void)hipMemAdvise(managed, 4096, hipMemAdviseSetReadMostly, 0);
  (void)hipMemAdvise(managed, 2 * 4096, hipMemAdviseSetPreferredLocation, 1);
  (void)hipMemPrefetchAsync(managed + 4096, 100, 0, nullptr);
  int read_mostly = -1, preferred = -1, prefetched = -1, whole = -1;
  (void)hipMemRangeGetAttribute(&read_mostly, 4, hipMemRangeAttributeReadMostly, managed, 4096);
  (void)hipMemRangeGetAttribute(&preferred, 4, hipMemRangeAttributePreferredLocation, managed, 2 * 4096);
  (void)hipMemRangeGetAttribute(&prefetched, 4, hipMemRangeAttributeLastPrefetchLocation, managed + 4096, 4096);
  (void)hipMemRangeGetAttribute(&whole, 4, hipMemRangeAttributeReadMostly, managed, 3 * 4096);
  check(read_mostly == 1 && preferred == 1 && prefetched == 0 && whole == 0,
        "advice and prefetches are kept, and read back, page by page",
        std::to_string(read_mostly) + " " + std::to_string(preferred) + " " + std::to_string(prefetched) + " " +
            std::to_string(whole));
  int coarse = -1;
  (void)hipMemRangeGetAttribute(&coarse, 4, hipMemRangeAttributeCoherencyMode, words, 4);
  check(coarse == hipMemRangeCoherencyModeCoarseGrain, "device memory is coarse-grained");
  EXPECT(hipMemPrefetchAsync(managed, 3 * 4096 + 1, 0, nullptr), hipErrorInvalidValue,
         "a prefetch past its allocation is refused");
  EXPECT(hipMemPrefetchAsync(managed, 4096, hipInvalidDeviceId, nullptr), hipErrorInvalidDevice,
         "and one to no device");
  (void)hipGetLastError();

  // ---- Pools: the one a device allocates from
  hipMemPool_t def = nullptr, cur = nullptr, made = nullptr;
  hipMemPoolProps props{};
  props.allocType = hipMemAllocationTypePinned;
  props.location = {hipMemLocationTypeDevice, 0};
  (void)hipDeviceGetDefaultMemPool(&def, 0);
  (void)hipMemPoolCreate(&made, &props);
  (void)hipDeviceSetMemPool(0, made);
  (void)hipDeviceGetMemPool(&cur, 0);
  check(cur == made, "a device's current pool is the one set");
  (void)hipMemPoolDestroy(made);
  (void)hipDeviceGetMemPool(&cur, 0);
  check(cur == def, "and its default again once that is destroyed");
  hipMemAccessFlags access = hipMemAccessFlagsProtNone;
  hipMemLocation own{hipMemLocationTypeDevice, 0}, peer{hipMemLocationTypeDevice, 1};
  (void)hipMemPoolGetAccess(&access, def, &own);
  check(access == hipMemAccessFlagsProtReadWrite, "a pool's own device reads and writes it");
  hipMemAccessDesc grant{peer, hipMemAccessFlagsProtRead};
  (void)hipMemPoolSetAccess(def, &grant, 1);
  (void)hipMemPoolGetAccess(&access, def, &peer);
  check(access == hipMemAccessFlagsProtRead, "another device as it was granted");

  // ---- A pool shared as a file descriptor, and a pointer in it
  hipMemPoolProps shared_props{};
  shared_props.allocType = hipMemAllocationTypePinned;
  shared_props.handleTypes = hipMemHandleTypePosixFileDescriptor;
  shared_props.location = {hipMemLocationTypeDevice, 0};
  hipMemPool_t exported = nullptr, plain = nullptr, imported = nullptr;
  (void)hipMemPoolCreate(&exported, &shared_props);
  shared_props.handleTypes = hipMemHandleTypeNone;
  (void)hipMemPoolCreate(&plain, &shared_props);
  void* fd = nullptr;   // a file descriptor, as hipShareableHdl holds it
  EXPECT(hipMemPoolExportToShareableHandle(&fd, plain, hipMemHandleTypePosixFileDescriptor, 0),
         hipErrorInvalidValue, "a pool not made for export is not exported");
  EXPECT(hipMemPoolExportToShareableHandle(&fd, exported, hipMemHandleTypePosixFileDescriptor, 0), hipSuccess,
         "a pool made for export gives a file descriptor");
  EXPECT(hipMemPoolImportFromShareableHandle(&imported, fd, hipMemHandleTypeNone, 0), hipErrorInvalidValue,
         "a descriptor is imported as a descriptor");
  EXPECT(hipMemPoolImportFromShareableHandle(&imported, fd, hipMemHandleTypePosixFileDescriptor, 0), hipSuccess,
         "and imports as a pool");
  uint32_t* in_pool = nullptr;
  (void)hipMallocFromPoolAsync(reinterpret_cast<void**>(&in_pool), 4096, exported, nullptr);
  std::vector<uint32_t> pattern(1024), readback(1024);
  for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = uint32_t(i * 2654435761u);
  (void)hipMemcpy(in_pool, pattern.data(), 4096, hipMemcpyHostToDevice);
  hipMemPoolPtrExportData exp_data{};
  EXPECT(hipMemPoolExportPointer(&exp_data, in_pool), hipSuccess, "a pointer in an exported pool is exported");
  void* opened = nullptr;
  EXPECT(hipMemPoolImportPointer(&opened, imported, &exp_data), hipSuccess, "and imported through the pool");
  (void)hipMemcpy(readback.data(), opened, 4096, hipMemcpyDeviceToHost);
  check(opened != in_pool && readback == pattern, "the imported pointer reaches the same bytes at an address of its own");
  EXPECT(hipFree(opened), hipSuccess, "an imported pointer is freed with hipFree");
  EXPECT(hipFree(in_pool), hipSuccess, "and the exporter's own allocation after it");
  (void)hipMemPoolDestroy(imported);
  (void)hipMemPoolDestroy(exported);
  (void)hipMemPoolDestroy(plain);

  // ---- Virtual memory at HIP's page granularity
  hipMemAllocationProp vprop{};
  vprop.type = hipMemAllocationTypePinned;
  vprop.location = {hipMemLocationTypeDevice, 0};
  size_t minimum = 0, recommended = 0;
  (void)hipMemGetAllocationGranularity(&minimum, &vprop, hipMemAllocationGranularityMinimum);
  (void)hipMemGetAllocationGranularity(&recommended, &vprop, hipMemAllocationGranularityRecommended);
  check(minimum == 4096 && recommended >= minimum, "the minimum granularity is a 4 KiB page",
        std::to_string(minimum) + " and " + std::to_string(recommended));
  vprop.location.id = 2;
  EXPECT(hipMemGetAllocationGranularity(&minimum, &vprop, hipMemAllocationGranularityMinimum),
         hipErrorInvalidValue, "granularity of a device there is not");
  vprop.location.id = 0;
  void* va = nullptr;
  EXPECT(hipMemAddressReserve(&va, 4095, 0, nullptr, 0), hipErrorInvalidValue,
         "address space is reserved in whole pages");
  EXPECT(hipMemAddressReserve(&va, 4096, recommended, nullptr, 0), hipSuccess, "one page of address space");
  hipMemGenericAllocationHandle_t page{};
  EXPECT(hipMemCreate(&page, 4095, &vprop, 0), hipErrorInvalidValue, "memory is made in whole pages");
  EXPECT(hipMemCreate(&page, 4096, &vprop, 0), hipSuccess, "one page of memory");
  EXPECT(hipMemMap(va, 4096, 0, page, 0), hipSuccess, "mapped into the page of address space");
  hipMemAccessDesc rw{{hipMemLocationTypeDevice, 0}, hipMemAccessFlagsProtReadWrite};
  EXPECT(hipMemSetAccess(va, 4096, &rw, 1), hipSuccess, "and made readable and writable");
  (void)hipMemcpy(va, pattern.data(), 4096, hipMemcpyHostToDevice);
  std::fill(readback.begin(), readback.end(), 0);
  (void)hipMemcpy(readback.data(), va, 4096, hipMemcpyDeviceToHost);
  check(readback == pattern, "a page of virtual memory holds what is copied into it");
  EXPECT(hipMemUnmap(va, 4095), hipErrorInvalidValue, "unmapped in whole pages");
  (void)hipMemUnmap(va, 4096);
  (void)hipMemRelease(page);
  EXPECT(hipMemAddressFree(va, 4096), hipSuccess, "and the address space freed");

  // ---- Virtual memory shared through a file descriptor, and device memory as one
  vprop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  hipMemGenericAllocationHandle_t shared_mem{}, imported_mem{};
  EXPECT(hipMemCreate(&shared_mem, recommended, &vprop, 0), hipSuccess, "memory made to be exported");
  int shared_fd = -1;
  EXPECT(hipMemExportToShareableHandle(&shared_fd, shared_mem, hipMemHandleTypeWin32, 0), hipErrorInvalidValue,
         "exported as a file descriptor, not a Windows handle");
  EXPECT(hipMemExportToShareableHandle(&shared_fd, shared_mem, hipMemHandleTypePosixFileDescriptor, 0), hipSuccess,
         "exports as a file descriptor");
  EXPECT(hipMemImportFromShareableHandle(&imported_mem, reinterpret_cast<void*>(static_cast<uintptr_t>(shared_fd)),
                                         hipMemHandleTypePosixFileDescriptor),
         hipSuccess, "which imports as memory of its own");
  void *va1 = nullptr, *va2 = nullptr;
  (void)hipMemAddressReserve(&va1, recommended, 0, nullptr, 0);
  (void)hipMemAddressReserve(&va2, recommended, 0, nullptr, 0);
  EXPECT(hipMemMap(va1, recommended, 0, shared_mem, 0), hipSuccess, "the exported memory maps");
  EXPECT(hipMemMap(va2, recommended, 0, imported_mem, 0), hipSuccess, "and so does the imported");
  (void)hipMemSetAccess(va1, recommended, &rw, 1);
  (void)hipMemSetAccess(va2, recommended, &rw, 1);
  unsigned long long got_access = 0;
  (void)hipMemGetAccess(&got_access, &rw.location, va2);
  check(got_access == hipMemAccessFlagsProtReadWrite, "with the access it was given");
  (void)hipMemcpy(va1, pattern.data(), 4096, hipMemcpyHostToDevice);
  std::fill(readback.begin(), readback.end(), 0);
  (void)hipMemcpy(readback.data(), va2, 4096, hipMemcpyDeviceToHost);
  check(readback == pattern, "both are the same bytes");
  EXPECT(hipMemUnmap(va1, recommended), hipSuccess, "unmapped");
  (void)hipMemUnmap(va2, recommended);
  (void)hipMemAddressFree(va1, recommended);
  (void)hipMemAddressFree(va2, recommended);
  EXPECT(hipMemRelease(shared_mem), hipSuccess, "and released");
  (void)hipMemRelease(imported_mem);
  close(shared_fd);
  uint32_t* dmabuf_src = nullptr;
  (void)hipMalloc(&dmabuf_src, 4096);
  (void)hipMemcpy(dmabuf_src, pattern.data(), 4096, hipMemcpyHostToDevice);
  int dmabuf = -1;
  EXPECT(hipMemGetHandleForAddressRange(&dmabuf, dmabuf_src, 4096, hipMemRangeHandleTypeDmaBufFd, 0), hipSuccess,
         "device memory as a file descriptor (a dma-buf on a card)");
  (void)hipMemImportFromShareableHandle(&imported_mem, reinterpret_cast<void*>(static_cast<uintptr_t>(dmabuf)),
                                        hipMemHandleTypePosixFileDescriptor);
  (void)hipMemAddressReserve(&va1, recommended, 0, nullptr, 0);
  (void)hipMemMap(va1, recommended, 0, imported_mem, 0);
  std::fill(readback.begin(), readback.end(), 0);
  (void)hipMemcpy(readback.data(), va1, 4096, hipMemcpyDeviceToHost);
  check(readback == pattern, "which imports as memory holding its bytes");
  (void)hipMemUnmap(va1, recommended);
  (void)hipMemAddressFree(va1, recommended);
  (void)hipMemRelease(imported_mem);
  close(dmabuf);
  (void)hipFree(dmabuf_src);

  // ---- Pinned memory reached in whole pages, as a card maps it
  double* small_pinned = nullptr;
  (void)hipHostMalloc(reinterpret_cast<void**>(&small_pinned), sizeof(float), hipHostMallocCoherent);
  *small_pinned = 1.5;
  add_double<<<1, 1>>>(small_pinned, 2.0);
  EXPECT(hipDeviceSynchronize(), hipSuccess, "an 8-byte atomic on a 4-byte pinned allocation stays in its page");
  check(*small_pinned == 3.5, "and adds", std::to_string(*small_pinned));
  (void)hipHostFree(small_pinned);

  // ---- Edge sizes
  void* ext = reinterpret_cast<void*>(1);
  EXPECT(hipExtMallocWithFlags(&ext, 0, hipDeviceMallocDefault), hipSuccess, "zero bytes with hipExtMallocWithFlags");
  check(ext == nullptr, "is no memory");
  EXPECT(hipExtMallocWithFlags(&ext, 16, hipMallocSignalMemory), hipErrorInvalidValue,
         "signal memory is one 8-byte signal");
  EXPECT(hipExtMallocWithFlags(&ext, 8, hipMallocSignalMemory), hipSuccess, "which it makes");
  size_t got_size = 0;
  (void)hipMemPtrGetInfo(ext, &got_size);
  check(got_size == 8, "and whose size is 8 bytes", std::to_string(got_size));
  (void)hipFree(ext);
  got_size = 1;
  EXPECT(hipMemPtrGetInfo(nullptr, &got_size), hipSuccess, "the size of a zero-byte allocation");
  check(got_size == 0, "is zero");
  void* one_src[1] = {pattern.data()};
  void* one_dst[1] = {readback.data()};
  size_t one_size[1] = {4};
  size_t fail_idx = 0;
  EXPECT(hipMemcpyBatchAsync(one_dst, one_src, one_size, 0, nullptr, nullptr, 0, &fail_idx, nullptr),
         hipErrorInvalidValue, "a batch of no copies");
  (void)hipGetLastError();

  // ---- Graph memory's counters
  uint64_t used = 1;
  EXPECT(hipDeviceGetGraphMemAttribute(-1, hipGraphMemAttrUsedMemCurrent, &used), hipErrorInvalidDevice,
         "graph memory of no device");
  (void)hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrUsedMemHigh, &used);
  check(used == 0, "no graph has allocated anything yet");
  uint64_t one = 1;
  EXPECT(hipDeviceSetGraphMemAttribute(0, hipGraphMemAttrUsedMemHigh, &one), hipErrorInvalidValue,
         "a high-water mark is set back to zero, or not at all");
  (void)hipGetLastError();

  // Pinned memory starts zeroed, as ROCm's does (fresh pages from the
  // kernel): hip-tests' atomics read a hipHostMalloc buffer they never
  // wrote. A buffer dirtied and freed, then allocated again, reads zero.
  {
    const size_t n = 3 << 20;
    for (int round = 0; round < 2; ++round) {
      unsigned char* h = nullptr;
      (void)hipHostMalloc(reinterpret_cast<void**>(&h), n);
      size_t nonzero = 0;
      for (size_t i = 0; i < n; ++i) nonzero += h[i] != 0;
      if (round == 1) check(nonzero == 0, "pinned memory starts zeroed", std::to_string(nonzero) + " bytes not zero");
      for (size_t i = 0; i < n; ++i) h[i] = 0xab;
      (void)hipHostFree(h);
    }
  }

  // A fill still queued when the memory is shared: hipMemset of device
  // memory returns before it runs, and sharing moves the bytes into a file,
  // so the handle waits for the fill and the shared memory holds it.
  {
    void* buf = nullptr;
    const size_t n = 1 << 20;
    (void)hipMalloc(&buf, n);
    (void)hipMemset(buf, 0x5a, n);
    hipIpcMemHandle_t h;
    const hipError_t e = hipIpcGetMemHandle(&h, buf);
    std::vector<unsigned char> back(n, 0);
    (void)hipMemcpy(back.data(), buf, n, hipMemcpyDeviceToHost);
    size_t wrong = 0;
    for (unsigned char b : back) wrong += b != 0x5a;
    check(e == hipSuccess && wrong == 0, "a fill queued before sharing lands in the shared memory",
          std::to_string(e) + ", " + std::to_string(wrong) + " bytes wrong");
    (void)hipFree(buf);
  }

  // A discrete GPU's memory is not the host's to touch directly: RCCL reads
  // this attribute to choose between pinned host memory and uncached device
  // memory it then writes from the CPU (an APU's layout).
  int direct = -1;
  (void)hipDeviceGetAttribute(&direct, hipDeviceAttributeDirectManagedMemAccessFromHost, 0);
  check(direct == 0, "a discrete GPU does not give the host direct access to its memory", std::to_string(direct));

  std::printf("memory: %d checks, %d failed\n", checks, failures);
  return failures ? 1 : 0;
}
