// The CUDA runtime API's documented behaviour at its edges, on a simulated
// 2 x Tesla T4. Each check was written from the documentation, and the ones
// marked with a note used to disagree with it.
#include <cuda.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
static int fails = 0;
#define CHECK(name, cond, ...)                                   \
  do {                                                           \
    bool ok_ = (cond);                                           \
    std::printf("%s %s", ok_ ? "PASS" : "FAIL", name);           \
    if (!ok_) { std::printf("  -> "); std::printf(__VA_ARGS__); ++fails; } \
    std::printf("\n");                                           \
  } while (0)

__global__ void fill(int* d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = i;
}
__global__ void oob(int* d) { d[threadIdx.x + (1 << 20)] = 1; }
__device__ int table[4];

int main() {
  int n = -1;
  cudaGetDeviceCount(&n);
  CHECK("device count", n == 2, "got %d", n);
  cudaError_t e = cudaSetDevice(2);
  CHECK("cudaSetDevice out of range", e == cudaErrorInvalidDevice, "got %d", e);
  cudaGetLastError();
  CHECK("cudaFree(nullptr)", cudaFree(nullptr) == cudaSuccess, "unexpected");

  int* big = nullptr;
  e = cudaMalloc(&big, (size_t)1 << 50);
  CHECK("cudaMalloc beyond the device", e == cudaErrorMemoryAllocation, "got %d", e);
  CHECK("peek does not reset", cudaPeekAtLastError() == cudaErrorMemoryAllocation, "unexpected");
  CHECK("get returns it", cudaGetLastError() == cudaErrorMemoryAllocation, "unexpected");
  CHECK("and resets", cudaGetLastError() == cudaSuccess, "unexpected");

  cudaDeviceProp p;
  std::memset(&p, 0, sizeof p);
  cudaGetDeviceProperties(&p, 0);
  CHECK("properties", !std::strcmp(p.name, "Tesla T4") && p.major == 7 && p.minor == 5 &&
                          p.maxThreadsPerBlock == 1024 && p.maxThreadsDim[2] == 64,
        "%s %d.%d %d %d", p.name, p.major, p.minor, p.maxThreadsPerBlock, p.maxThreadsDim[2]);
  size_t free0 = 0, total = 0;
  cudaMemGetInfo(&free0, &total);
  CHECK("memGetInfo total", total == p.totalGlobalMem && free0 <= total, "unexpected");

  // Where a pointer lives. Pinned and managed memory used to report
  // "unregistered".
  int* d = nullptr;
  cudaMalloc(&d, 64 * sizeof(int));
  cudaPointerAttributes a;
  std::memset(&a, 0xff, sizeof a);
  CHECK("attributes: device", cudaPointerGetAttributes(&a, d) == cudaSuccess &&
                                  a.type == cudaMemoryTypeDevice && a.device == 0, "type %d", (int)a.type);
  int* host = (int*)std::malloc(64);
  std::memset(&a, 0xff, sizeof a);
  CHECK("attributes: malloc is unregistered", cudaPointerGetAttributes(&a, host) == cudaSuccess &&
                                                 a.type == cudaMemoryTypeUnregistered, "type %d", (int)a.type);
  CHECK("attributes: malloc belongs to no device", a.device == cudaInvalidDeviceId && a.devicePointer == nullptr &&
                                                      a.hostPointer == host, "device %d", a.device);
  int* pinned = nullptr;
  cudaHostAlloc(&pinned, 64 * sizeof(int), 0);
  std::memset(&a, 0xff, sizeof a);
  cudaPointerGetAttributes(&a, pinned + 3);   // an interior pointer, too
  CHECK("attributes: pinned is host", a.type == cudaMemoryTypeHost, "type %d", (int)a.type);
  int* m = nullptr;
  cudaMallocManaged(&m, 64 * sizeof(int));
  std::memset(&a, 0xff, sizeof a);
  cudaPointerGetAttributes(&a, m);
  CHECK("attributes: managed", a.type == cudaMemoryTypeManaged, "type %d", (int)a.type);

  // Launch configurations the device cannot run. These used to be
  // cudaErrorInvalidValue.
  fill<<<1, 2000>>>(d, 64);
  e = cudaGetLastError();
  CHECK("2000 threads per block", e == cudaErrorInvalidConfiguration, "got %d %s", e, cudaGetErrorName(e));
  fill<<<dim3(1, 1, 1), dim3(1, 1, 65)>>>(d, 64);
  e = cudaGetLastError();
  CHECK("blockDim.z 65", e == cudaErrorInvalidConfiguration, "got %d", e);
  fill<<<0, 16>>>(d, 64);
  e = cudaGetLastError();
  CHECK("an empty grid", e == cudaErrorInvalidConfiguration, "got %d", e);
  fill<<<1, 64>>>(d, 64);
  CHECK("a valid launch still works after them", cudaDeviceSynchronize() == cudaSuccess, "unexpected");

  cudaStream_t s;
  cudaStreamCreate(&s);
  for (int i = 0; i < 64; ++i) pinned[i] = 1000 + i;
  cudaMemcpyAsync(d, pinned, 64 * sizeof(int), cudaMemcpyHostToDevice, s);
  std::memset(pinned, 0, 64 * sizeof(int));
  cudaMemcpyAsync(pinned, d, 64 * sizeof(int), cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  CHECK("async round trip", pinned[63] == 1063, "got %d", pinned[63]);

  // Everything this program allocated on the host, released while the context
  // is still usable -- after the illegal address below, frees fail too. CI runs
  // with LeakSanitizer, which counts these.
  CHECK("frees succeed", cudaFreeHost(pinned) == cudaSuccess && cudaFree(m) == cudaSuccess,
        "unexpected");
  std::free(host);

  // A bad pointer handed to a host call is an argument error. These used to be
  // cudaErrorIllegalAddress, the code for a faulting kernel.
  int stack_int = 0;
  e = cudaFree(&stack_int);
  CHECK("cudaFree(stack pointer) -> invalid value", e == cudaErrorInvalidValue, "got %d %s", e, cudaGetErrorName(e));
  int* d1 = nullptr;
  cudaSetDevice(1);
  cudaMalloc(&d1, 64 * sizeof(int));
  cudaSetDevice(0);
  e = cudaMemcpy(d, &stack_int, sizeof stack_int, cudaMemcpyDeviceToHost);
  CHECK("cudaMemcpy with the wrong kind -> invalid value", e == cudaErrorInvalidValue, "got %d %s", e, cudaGetErrorName(e));
  e = cudaMemcpyPeer(d1, 0, d, 1, 64);
  CHECK("cudaMemcpyPeer with devices swapped -> invalid value", e == cudaErrorInvalidValue, "got %d %s", e, cudaGetErrorName(e));
  int* probe = nullptr;
  CHECK("and none of them poisons the context", cudaMalloc(&probe, 16) == cudaSuccess, "unexpected");
  cudaGetLastError();

  // Nothing to copy succeeds whatever the pointers and the kind say: an RTX
  // 3060 returns cudaSuccess for every one of these, and CUTLASS copies empty
  // tensors (null pointers) as device-to-device when a vector of them grows.
  // The kind check refused them for a while, which failed CUTLASS's SM100
  // ptr-array GEMM tests. A nonzero copy between null pointers still refuses.
  {
    cudaStream_t zs;
    cudaStreamCreate(&zs);
    const cudaError_t z[] = {
        cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyDeviceToDevice),
        cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyHostToDevice),
        cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyDeviceToHost),
        cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyDefault),
        cudaMemcpy(&stack_int, d, 0, cudaMemcpyHostToDevice),  // the wrong kind, but nothing to copy
        cudaMemcpyAsync(nullptr, nullptr, 0, cudaMemcpyDeviceToDevice, zs),
        cudaMemcpy2D(nullptr, 64, nullptr, 64, 16, 0, cudaMemcpyDeviceToDevice),
        cudaMemcpyPeer(nullptr, 1, nullptr, 0, 0),
        cudaMemcpyToSymbol(table, nullptr, 0),
        cudaMemcpyFromSymbol(nullptr, table, 0),
        cudaMemset(nullptr, 0, 0),
    };
    int bad = -1;
    for (int i = 0; i < (int)(sizeof z / sizeof z[0]); ++i)
      if (z[i] != cudaSuccess && bad < 0) bad = i;
    CHECK("zero-byte copies and memsets succeed, null pointers and all", bad < 0,
          "call %d returned %d %s", bad, bad < 0 ? 0 : z[bad], cudaGetErrorName(bad < 0 ? cudaSuccess : z[bad]));
    e = cudaMemcpy(nullptr, nullptr, 4, cudaMemcpyDeviceToDevice);
    CHECK("a nonzero copy between null pointers -> invalid value", e == cudaErrorInvalidValue,
          "got %d %s", e, cudaGetErrorName(e));
    CHECK("and neither poisons the context", cudaStreamSynchronize(zs) == cudaSuccess &&
          cudaDeviceSynchronize() == cudaSuccess, "unexpected");
    cudaStreamDestroy(zs);
    cudaGetLastError();
  }

  // Synchronization reports failed work, not a refused call.
  cudaMalloc(&big, (size_t)1 << 50);
  e = cudaDeviceSynchronize();
  CHECK("synchronize after a failed cudaMalloc", e == cudaSuccess, "got %d %s", e, cudaGetErrorName(e));
  cudaGetLastError();

  // cudaFreeHost frees only what cudaMallocHost/cudaHostAlloc returned. These
  // used to abort or segfault inside the allocator.
  char* plain = (char*)std::malloc(64);
  CHECK("cudaFreeHost(malloc'd) -> invalid value", cudaFreeHost(plain) == cudaErrorInvalidValue, "unexpected");
  std::free(plain);
  CHECK("cudaFreeHost(device pointer) -> invalid value", cudaFreeHost(d1) == cudaErrorInvalidValue, "unexpected");
  void* ph = nullptr;
  cudaMallocHost(&ph, 64);
  CHECK("cudaFreeHost twice -> invalid value", cudaFreeHost(ph) == cudaSuccess && cudaFreeHost(ph) == cudaErrorInvalidValue, "unexpected");
  cudaGetLastError();

  // The current device belongs to the thread, and a new thread starts on 0.
  // It used to be one process-wide variable.
  int seen = -1, now = -1;
  std::thread([&] { cudaGetDevice(&seen); cudaSetDevice(1); }).join();
  cudaGetDevice(&now);
  CHECK("current device is per thread", seen == 0 && now == 0, "thread started on %d, main is on %d", seen, now);

  // Peer access is tracked, with the documented errors. All used to succeed.
  CHECK("enable peer access", cudaDeviceEnablePeerAccess(1, 0) == cudaSuccess, "unexpected");
  e = cudaDeviceEnablePeerAccess(1, 0);
  CHECK("enable twice -> already enabled", e == cudaErrorPeerAccessAlreadyEnabled, "got %d %s", e, cudaGetErrorName(e));
  CHECK("disable", cudaDeviceDisablePeerAccess(1) == cudaSuccess, "unexpected");
  e = cudaDeviceDisablePeerAccess(1);
  CHECK("disable twice -> not enabled", e == cudaErrorPeerAccessNotEnabled, "got %d %s", e, cudaGetErrorName(e));
  CHECK("nonzero flags -> invalid value", cudaDeviceEnablePeerAccess(1, 5) == cudaErrorInvalidValue, "unexpected");
  CHECK("bad peer ordinal -> invalid device", cudaDeviceDisablePeerAccess(7) == cudaErrorInvalidDevice, "unexpected");
  cudaGetLastError();

  // Host registration is recorded and reported.
  char* reg = (char*)std::malloc(4096);
  CHECK("cudaHostRegister", cudaHostRegister(reg, 4096, 0) == cudaSuccess, "unexpected");
  std::memset(&a, 0xff, sizeof a);
  cudaPointerGetAttributes(&a, reg + 10);
  CHECK("attributes: registered memory is host", a.type == cudaMemoryTypeHost, "type %d", (int)a.type);
  e = cudaHostRegister(reg, 4096, 0);
  CHECK("register twice -> already registered", e == cudaErrorHostMemoryAlreadyRegistered, "got %d %s", e, cudaGetErrorName(e));
  CHECK("cudaHostUnregister", cudaHostUnregister(reg) == cudaSuccess, "unexpected");
  e = cudaHostUnregister(reg);
  CHECK("unregister twice -> not registered", e == cudaErrorHostMemoryNotRegistered, "got %d %s", e, cudaGetErrorName(e));
  std::free(reg);
  cudaGetLastError();

  // Mapped pinned memory has a device pointer, and a kernel can use it. It
  // used to be refused here while the attributes call named it anyway, and a
  // kernel given that pointer faulted.
  int* mapped = nullptr;
  cudaHostAlloc(&mapped, 64 * sizeof(int), cudaHostAllocMapped);
  void* mdev = nullptr;
  e = cudaHostGetDevicePointer(&mdev, mapped, 0);
  CHECK("cudaHostGetDevicePointer(mapped)", e == cudaSuccess && mdev == mapped, "got %d %p", e, mdev);
  cudaSetDevice(1);
  fill<<<1, 64>>>((int*)mdev, 64);
  e = cudaDeviceSynchronize();
  cudaSetDevice(0);
  CHECK("a kernel on device 1 writes mapped pinned memory", e == cudaSuccess && mapped[63] == 63, "got %d, [63]=%d", e, mapped[63]);
  cudaFreeHost(mapped);

  // Managed memory from device 0 is addressable on device 1 and frees from
  // there. The launch used to fault and the free to leave it mapped.
  int* mm = nullptr;
  cudaMallocManaged(&mm, 64 * sizeof(int));
  cudaSetDevice(1);
  fill<<<1, 64>>>(mm, 64);
  e = cudaDeviceSynchronize();
  CHECK("managed memory is addressable from another device", e == cudaSuccess && mm[40] == 40, "got %d, [40]=%d", e, mm[40]);
  CHECK("and frees with that device current", cudaFree(mm) == cudaSuccess, "unexpected");
  cudaSetDevice(0);

  // Events time what happens between them.
  cudaEvent_t ev1, ev2, untimed;
  cudaEventCreate(&ev1);
  cudaEventCreate(&ev2);
  cudaEventCreateWithFlags(&untimed, cudaEventDisableTiming);
  float ms = -1;
  e = cudaEventElapsedTime(&ms, ev1, ev2);
  CHECK("elapsed time of unrecorded events -> invalid handle", e == cudaErrorInvalidResourceHandle, "got %d", e);
  cudaEventRecord(ev1, nullptr);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  cudaEventRecord(ev2, nullptr);
  cudaEventRecord(untimed, nullptr);
  e = cudaEventElapsedTime(&ms, ev1, ev2);
  CHECK("elapsed time is measured", e == cudaSuccess && ms >= 15.0f, "got %d, %f ms", e, ms);
  e = cudaEventElapsedTime(&ms, ev1, untimed);
  CHECK("timing-disabled event -> invalid handle", e == cudaErrorInvalidResourceHandle, "got %d", e);
  cudaEventDestroy(ev1);
  cudaEventDestroy(ev2);
  cudaEventDestroy(untimed);

  // Every declared code has its name; an undeclared one is "unrecognized".
  CHECK("error names", !std::strcmp(cudaGetErrorName(cudaErrorNoDevice), "cudaErrorNoDevice") &&
                           !std::strcmp(cudaGetErrorName(cudaErrorNotReady), "cudaErrorNotReady") &&
                           !std::strcmp(cudaGetErrorName(cudaErrorPeerAccessAlreadyEnabled),
                                        "cudaErrorPeerAccessAlreadyEnabled") &&
                           !std::strcmp(cudaGetErrorString((cudaError_t)12345), "unrecognized error code"),
        "%s / %s", cudaGetErrorName(cudaErrorNoDevice), cudaGetErrorString((cudaError_t)12345));

  // A freed device pointer is not a device pointer. It used to report one,
  // with a device ordinal computed from the address.
  int* gone = nullptr;
  cudaMalloc(&gone, 64);
  cudaFree(gone);
  e = cudaPointerGetAttributes(&a, gone);
  CHECK("attributes of a freed device pointer -> invalid value", e == cudaErrorInvalidValue, "got %d type %d", e, (int)a.type);
  cudaGetLastError();

  // Last: a kernel's illegal address corrupts the context, and it stays
  // corrupted until a reset. The next call used to succeed.
  oob<<<1, 1>>>(d);
  e = cudaDeviceSynchronize();
  CHECK("out-of-bounds write -> illegal address", e == cudaErrorIllegalAddress, "got %d", e);
  int* after = nullptr;
  CHECK("sticky: the next malloc fails", cudaMalloc(&after, 16) == cudaErrorIllegalAddress, "unexpected");
  CHECK("sticky: reading it does not clear it", cudaGetLastError() == cudaErrorIllegalAddress &&
                                                   cudaGetLastError() == cudaErrorIllegalAddress, "unexpected");
  CHECK("reset clears it", cudaDeviceReset() == cudaSuccess && cudaMalloc(&after, 16) == cudaSuccess, "unexpected");

  // Copy directions under unified addressing, as an RTX 3060 answers them: a
  // side the kind calls device memory must be device memory, while a side it
  // calls host memory may be a device allocation and is copied as one. Symbol
  // copies follow the same rule. GPUJPEG relies on both.
  {
    int *a = nullptr, *b = nullptr;
    cudaMalloc(&a, 16);
    cudaMalloc(&b, 16);
    int src4[4] = {1, 2, 3, 4}, got[4] = {0}, hdst[4] = {0};
    cudaMemcpy(b, src4, 16, cudaMemcpyHostToDevice);
    cudaMemset(a, 0, 16);
    e = cudaMemcpy(a, b, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(got, a, 16, cudaMemcpyDeviceToHost);
    CHECK("cudaMemcpy HostToDevice from a device pointer copies", e == cudaSuccess && got[3] == 4, "got %d, value %d", e, got[3]);
    cudaMemset(a, 0, 16);
    e = cudaMemcpy(a, b, 16, cudaMemcpyDeviceToHost);
    cudaMemcpy(got, a, 16, cudaMemcpyDeviceToHost);
    CHECK("cudaMemcpy DeviceToHost into a device pointer copies", e == cudaSuccess && got[3] == 4, "got %d, value %d", e, got[3]);
    cudaMemset(a, 0, 16);
    e = cudaMemcpyAsync(a, b, 16, cudaMemcpyHostToDevice, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(got, a, 16, cudaMemcpyDeviceToHost);
    CHECK("cudaMemcpyAsync HostToDevice from a device pointer copies", e == cudaSuccess && got[3] == 4, "got %d, value %d", e, got[3]);
    e = cudaMemcpy(hdst, src4, 16, cudaMemcpyDeviceToHost);
    CHECK("cudaMemcpy DeviceToHost from a host pointer -> invalid value", e == cudaErrorInvalidValue, "got %d %s", e, cudaGetErrorName(e));
    e = cudaMemcpy(hdst, b, 16, cudaMemcpyHostToDevice);
    CHECK("cudaMemcpy HostToDevice into a host pointer -> invalid value", e == cudaErrorInvalidValue, "got %d %s", e, cudaGetErrorName(e));
    e = cudaMemcpy(hdst, src4, 16, cudaMemcpyDeviceToDevice);
    CHECK("cudaMemcpy DeviceToDevice between host pointers -> invalid value", e == cudaErrorInvalidValue, "got %d %s", e, cudaGetErrorName(e));
    cudaGetLastError();

    int nines[4] = {9, 9, 9, 9};
    cudaMemcpyToSymbol(table, nines, 16);
    e = cudaMemcpyToSymbol(table, b, 16, 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyFromSymbol(got, table, 16);
    CHECK("cudaMemcpyToSymbol from device memory", e == cudaSuccess && got[3] == 4, "got %d, value %d", e, got[3]);
    cudaMemcpyToSymbol(table, nines, 16);
    e = cudaMemcpyToSymbolAsync(table, b, 16, 0, cudaMemcpyDeviceToDevice, 0);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(got, table, 16);
    CHECK("cudaMemcpyToSymbolAsync from device memory", e == cudaSuccess && got[3] == 4, "got %d, value %d", e, got[3]);
    cudaMemcpyToSymbol(table, nines, 16);
    e = cudaMemcpyToSymbol(table, b, 16, 0, cudaMemcpyHostToDevice);
    cudaMemcpyFromSymbol(got, table, 16);
    CHECK("cudaMemcpyToSymbol HostToDevice from a device pointer copies", e == cudaSuccess && got[3] == 4, "got %d, value %d", e, got[3]);
    cudaMemcpyToSymbol(table, src4, 16);
    cudaMemset(a, 0, 16);
    e = cudaMemcpyFromSymbol(a, table, 16, 0, cudaMemcpyDeviceToDevice);
    cudaMemcpy(got, a, 16, cudaMemcpyDeviceToHost);
    CHECK("cudaMemcpyFromSymbol into device memory", e == cudaSuccess && got[3] == 4, "got %d, value %d", e, got[3]);
    e = cudaMemcpyToSymbol(table, src4, 16, 0, cudaMemcpyDeviceToDevice);
    CHECK("cudaMemcpyToSymbol DeviceToDevice from a host pointer -> invalid value", e == cudaErrorInvalidValue, "got %d %s", e, cudaGetErrorName(e));
    e = cudaMemcpyFromSymbol(hdst, table, 16, 0, cudaMemcpyDeviceToDevice);
    CHECK("cudaMemcpyFromSymbol DeviceToDevice into a host pointer -> invalid value", e == cudaErrorInvalidValue, "got %d %s", e, cudaGetErrorName(e));
    cudaGetLastError();
    cudaFree(a);
    cudaFree(b);
  }

  // Memory CUDA allocated on the host counts as device memory for a copy:
  // managed and pinned buffers are accepted as the device side in every
  // direction, as an RTX 3060 accepts them; pageable memory, and memory pinned
  // afterwards with cudaHostRegister, are refused there. NanoVDB copies from a
  // managed buffer with cudaMemcpyDeviceToHost.
  {
    const int four[4] = {1, 2, 3, 4};
    int *managed = nullptr, *pinned = nullptr, *dev = nullptr, got[4] = {0};
    cudaMallocManaged(&managed, 16);
    cudaMallocHost(&pinned, 16);
    cudaMalloc(&dev, 16);
    int* regd = static_cast<int*>(std::aligned_alloc(4096, 4096));
    cudaHostRegister(regd, 4096, cudaHostRegisterDefault);
    std::memcpy(managed, four, 16);
    std::memcpy(pinned, four, 16);
    e = cudaMemcpy(got, managed, 16, cudaMemcpyDeviceToHost);
    CHECK("cudaMemcpy DeviceToHost from managed memory", e == cudaSuccess && got[3] == 4, "got %d %s, value %d", e, cudaGetErrorName(e), got[3]);
    got[3] = 0;
    e = cudaMemcpyAsync(got, pinned, 16, cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);
    CHECK("cudaMemcpyAsync DeviceToHost from pinned memory", e == cudaSuccess && got[3] == 4, "got %d %s, value %d", e, cudaGetErrorName(e), got[3]);
    std::memset(managed, 0, 16);
    e = cudaMemcpy(managed, four, 16, cudaMemcpyHostToDevice);
    CHECK("cudaMemcpy HostToDevice into managed memory", e == cudaSuccess && managed[3] == 4, "got %d %s", e, cudaGetErrorName(e));
    cudaMemset(dev, 0, 16);
    e = cudaMemcpy(dev, pinned, 16, cudaMemcpyDeviceToDevice);
    cudaMemcpy(got, dev, 16, cudaMemcpyDeviceToHost);
    CHECK("cudaMemcpy DeviceToDevice from pinned to device memory", e == cudaSuccess && got[3] == 4, "got %d %s, value %d", e, cudaGetErrorName(e), got[3]);
    std::memset(managed, 0, 16);
    e = cudaMemcpy(managed, dev, 16, cudaMemcpyDeviceToDevice);
    CHECK("cudaMemcpy DeviceToDevice from device to managed memory", e == cudaSuccess && managed[3] == 4, "got %d %s", e, cudaGetErrorName(e));
    e = cudaMemcpy(got, regd, 16, cudaMemcpyDeviceToHost);
    CHECK("cudaMemcpy DeviceToHost from cudaHostRegister'd memory -> invalid value", e == cudaErrorInvalidValue, "got %d %s", e, cudaGetErrorName(e));
    cudaGetLastError();
    int nines[4] = {9, 9, 9, 9};
    cudaMemcpyToSymbol(table, nines, 16);
    e = cudaMemcpyToSymbol(table, managed, 16, 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyFromSymbol(got, table, 16);
    CHECK("cudaMemcpyToSymbol DeviceToDevice from managed memory", e == cudaSuccess && got[3] == 4, "got %d %s, value %d", e, cudaGetErrorName(e), got[3]);
    // cudaMemset follows the same rule.
    e = cudaMemset(managed, 7, 16);
    CHECK("cudaMemset on managed memory", e == cudaSuccess && managed[3] == 0x07070707, "got %d %s", e, cudaGetErrorName(e));
    e = cudaMemsetAsync(pinned + 1, 0, 8, 0);
    cudaStreamSynchronize(0);
    CHECK("cudaMemsetAsync on pinned memory", e == cudaSuccess && pinned[0] == 1 && pinned[2] == 0 && pinned[3] == 4,
          "got %d %s", e, cudaGetErrorName(e));
    e = cudaMemset(regd, 0, 16);
    CHECK("cudaMemset on cudaHostRegister'd memory -> invalid value", e == cudaErrorInvalidValue, "got %d %s", e, cudaGetErrorName(e));
    cudaGetLastError();
    cudaHostUnregister(regd);
    std::free(regd);
    cudaFree(managed);
    cudaFreeHost(pinned);
    cudaFree(dev);
  }

  // A driver function outside the runtime, fetched the way NanoVDB fetches
  // cuMemGetAllocationGranularity and then calls without checking for null;
  // and a name the driver does not have, which is not found. Not under
  // AddressSanitizer: fetching it loads the simulator's libcuda beside its
  // libcudart, two copies of the core that ASan refuses as an ODR violation
  // (the reason run_mixed_apis.sh skips sanitizer builds).
#ifndef __SANITIZE_ADDRESS__
  {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
    e = cudaGetDriverEntryPointByVersion("cuMemGetAllocationGranularity", &fn, 12000, cudaEnableDefault, &q);
#else
    e = cudaGetDriverEntryPoint("cuMemGetAllocationGranularity", &fn, cudaEnableDefault, &q);
#endif
    size_t granularity = 0;
    if (e == cudaSuccess && fn) {
      CUmemAllocationProp prop = {};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = 0;
      using Granularity = CUresult (*)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
      reinterpret_cast<Granularity>(fn)(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    }
    CHECK("cuMemGetAllocationGranularity through the driver entry point",
          e == cudaSuccess && q == cudaDriverEntryPointSuccess && granularity > 0,
          "got %d %s, status %d, granularity %zu", e, cudaGetErrorName(e), (int)q, granularity);
    fn = &granularity;
#if CUDART_VERSION >= 12050
    e = cudaGetDriverEntryPointByVersion("cuNoSuchFunction", &fn, 12000, cudaEnableDefault, &q);
#else
    e = cudaGetDriverEntryPoint("cuNoSuchFunction", &fn, cudaEnableDefault, &q);
#endif
    CHECK("a driver function that does not exist is not found",
          e == cudaSuccess && fn == nullptr && q == cudaDriverEntryPointSymbolNotFound,
          "got %d %s, status %d", e, cudaGetErrorName(e), (int)q);
  }
#endif

  // Cache preferences: accepted for a kernel, refused for anything else, and the
  // device-wide one reads back.
  e = cudaFuncSetCacheConfig(fill, cudaFuncCachePreferL1);
  CHECK("cudaFuncSetCacheConfig on a kernel", e == cudaSuccess, "got %d %s", e, cudaGetErrorName(e));
  e = cudaFuncSetCacheConfig((const void*)&stack_int, cudaFuncCachePreferL1);
  CHECK("cudaFuncSetCacheConfig on a non-kernel -> invalid resource handle", e == cudaErrorInvalidResourceHandle, "got %d %s", e, cudaGetErrorName(e));
  cudaGetLastError();
  {
    cudaFuncCache c = cudaFuncCachePreferNone;
    e = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cudaError_t e2 = cudaDeviceGetCacheConfig(&c);
    CHECK("cudaDeviceSetCacheConfig reads back", e == cudaSuccess && e2 == cudaSuccess && c == cudaFuncCachePreferShared,
          "got %d %d, reads %d", e, e2, (int)c);
  }

  // The stream-ordered allocator is implemented, so memory pools are reported,
  // as an RTX 3060 reports them; and what one device can do with another's
  // memory, where a device asked about itself (or one that does not exist) is
  // an invalid device, as the card answers.
  {
    int v = -1;
    e = cudaDeviceGetAttribute(&v, cudaDevAttrMemoryPoolsSupported, 0);
    CHECK("memory pools supported", e == cudaSuccess && v == 1, "got %d %s, value %d", e, cudaGetErrorName(e), v);
    // Concurrent managed access is answered (1 on Linux, 0 on Windows and WSL,
    // so only the answer is checked); NanoVDB filters devices on it.
    v = -1;
    e = cudaDeviceGetAttribute(&v, cudaDevAttrConcurrentManagedAccess, 0);
    CHECK("concurrent managed access is answered", e == cudaSuccess && (v == 0 || v == 1), "got %d %s, value %d", e, cudaGetErrorName(e), v);
    v = -1;
    e = cudaDeviceGetP2PAttribute(&v, cudaDevP2PAttrAccessSupported, 0, 1);
    CHECK("P2P access supported between two devices", e == cudaSuccess && v == 1, "got %d %s, value %d", e, cudaGetErrorName(e), v);
    e = cudaDeviceGetP2PAttribute(&v, cudaDevP2PAttrPerformanceRank, 1, 0);
    CHECK("P2P performance rank", e == cudaSuccess && v == 0, "got %d %s, value %d", e, cudaGetErrorName(e), v);
    e = cudaDeviceGetP2PAttribute(&v, cudaDevP2PAttrAccessSupported, 0, 0);
    CHECK("P2P attribute of a device with itself -> invalid device", e == cudaErrorInvalidDevice, "got %d %s", e, cudaGetErrorName(e));
    e = cudaDeviceGetP2PAttribute(&v, cudaDevP2PAttrAccessSupported, 0, 7);
    CHECK("P2P attribute with no such device -> invalid device", e == cudaErrorInvalidDevice, "got %d %s", e, cudaGetErrorName(e));
    cudaGetLastError();
  }

  // A managed buffer freed with another device current is gone from every
  // device. It used to stay mapped on the device that allocated it, and a
  // kernel there wrote into memory the allocator had taken back.
  int* mf = nullptr;
  cudaMallocManaged(&mf, 64 * sizeof(int));
  cudaSetDevice(1);
  cudaFree(mf);
  cudaSetDevice(0);
  fill<<<1, 64>>>(mf, 64);
  e = cudaDeviceSynchronize();
  CHECK("a kernel writing freed managed memory -> illegal address", e == cudaErrorIllegalAddress, "got %d %s", e, cudaGetErrorName(e));
  cudaDeviceReset();

  // A reset releases what the device held. It used to release nothing.
  int* held = nullptr;
  size_t free_held = 0, free_reset = 0;
  cudaMalloc(&held, (size_t)256 << 20);
  cudaMemGetInfo(&free_held, &total);
  CHECK("cudaDeviceReset", cudaDeviceReset() == cudaSuccess, "unexpected");
  cudaMemGetInfo(&free_reset, &total);
  CHECK("reset releases the device's allocations", free_reset == total && free_reset - free_held >= ((size_t)256 << 20),
        "free %zu before, %zu after, total %zu", free_held, free_reset, total);
  e = cudaFree(held);
  CHECK("a pointer from before the reset no longer frees", e == cudaErrorInvalidValue, "got %d %s", e, cudaGetErrorName(e));
  cudaGetLastError();

  std::printf("%d failed\n", fails);
  return fails != 0;
}
