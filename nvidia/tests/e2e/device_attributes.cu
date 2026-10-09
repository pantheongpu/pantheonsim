// What a program is told about its device, and that the answers agree.
//
// CUDA answers one question three ways -- cudaDeviceGetAttribute,
// cuDeviceGetAttribute and cudaDeviceProp -- and gives the device an identity
// four (its UUID, PCI address, bus id string and the order of both). Each is
// checked against the others here, and the program passes on a real card as it
// does on the simulator: nothing below is a value, only an agreement, a
// refusal or a clamp.
//
//   device_attributes            checks, prints "PASS" or the first failure
//   device_attributes --dump     every attribute, property member and limit,
//                                one per line, for comparison with a card's
//                                (run_device_attributes.sh does)
//   device_attributes --uuids    the UUID of each device, for the
//                                CUDA_VISIBLE_DEVICES runner
//   device_attributes --shown    what the program sees of its machine
//   device_attributes --where    allocates 64 MiB on the program's device 0 and
//                                says, through NVML, which of the machine's
//                                devices has it (a program's device 0 under
//                                CUDA_VISIBLE_DEVICES=1 is the machine's second)
#include <cuda.h>
#include <cuda_runtime.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "device_attributes_names.inc"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::printf("FAIL %s -> %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
#define CU(x) do { CUresult e_ = (x); if (e_ != CUDA_SUCCESS) { \
  std::printf("FAIL %s -> %d (line %d)\n", #x, (int)e_, __LINE__); return 1; } } while (0)
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s (line %d)\n", #c, __LINE__); return 1; } } while (0)

namespace {

int dump(int dev) {
  for (const auto& a : kRtAttrs) {
    int v = -12345;
    const cudaError_t r = cudaDeviceGetAttribute(&v, (cudaDeviceAttr)a.id, dev);
    if (r == cudaSuccess) std::printf("rt %s %d\n", a.name, v);
    else { std::printf("rt %s error(%d)\n", a.name, (int)r); (void)cudaGetLastError(); }
  }
  CUdevice d;
  cuInit(0);
  cuDeviceGet(&d, dev);
  for (const auto& a : kDrvAttrs) {
    int v = -12345;
    const CUresult r = cuDeviceGetAttribute(&v, (CUdevice_attribute)a.id, d);
    if (r == CUDA_SUCCESS) std::printf("drv %s %d\n", a.name, v);
    else std::printf("drv %s error(%d)\n", a.name, (int)r);
  }
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, dev) == cudaSuccess) {
#include "device_attributes_props.inc"
  }
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  std::printf("meminfo total %zu\n", total_b);
  int lo = 0, hi = 0;
  cudaDeviceGetStreamPriorityRange(&lo, &hi);
  std::printf("stream_priority %d %d\n", lo, hi);
  return 0;
}

std::string uuid_hex(const unsigned char* b) {
  char s[40];
  for (int i = 0; i < 16; ++i) std::snprintf(s + 2 * i, 3, "%02x", b[i]);
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "";
  if (!std::strcmp(mode, "--dump")) return dump(0);
  int n = 0;
  if (!std::strcmp(mode, "--uuids") || !std::strcmp(mode, "--shown")) {
    const cudaError_t rc = cudaGetDeviceCount(&n);
    CUresult drc = cuInit(0);
    int dn = -1;
    CUresult drc2 = cuDeviceGetCount(&dn);
    std::printf("runtime %d count %d | driver cuInit %d count-rc %d count %d |", (int)rc, rc == cudaSuccess ? n : -1,
                (int)drc, (int)drc2, dn);
    for (int i = 0; rc == cudaSuccess && i < n; ++i) {
      cudaDeviceProp p;
      CK(cudaGetDeviceProperties(&p, i));
      char bus[32];
      CK(cudaDeviceGetPCIBusId(bus, sizeof bus, i));
      if (!std::strcmp(mode, "--uuids")) std::printf(" GPU-%s", uuid_hex((const unsigned char*)p.uuid.bytes).c_str());
      else std::printf(" [%d %s %s]", i, bus, uuid_hex((const unsigned char*)p.uuid.bytes).substr(0, 8).c_str());
    }
    std::printf("\n");
    return 0;
  }
  if (!std::strcmp(mode, "--where")) {
    void* mem = nullptr;
    CK(cudaSetDevice(0));
    CK(cudaMalloc(&mem, 64u << 20));
    // NVML names the machine's devices, whatever this program calls them.
    void* nvml = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    CHECK(nvml != nullptr);
    typedef int (*Init)();
    typedef int (*Count)(unsigned*);
    typedef int (*Handle)(unsigned, void**);
    typedef struct { unsigned long long total, free, used; } Memory;
    typedef int (*Info)(void*, Memory*);
    Init init = (Init)dlsym(nvml, "nvmlInit_v2");
    Count count = (Count)dlsym(nvml, "nvmlDeviceGetCount_v2");
    Handle handle = (Handle)dlsym(nvml, "nvmlDeviceGetHandleByIndex_v2");
    Info info = (Info)dlsym(nvml, "nvmlDeviceGetMemoryInfo");
    CHECK(init && count && handle && info);
    CHECK(init() == 0);
    unsigned machine = 0;
    CHECK(count(&machine) == 0);
    for (unsigned i = 0; i < machine; ++i) {
      void* h = nullptr;
      Memory m{};
      CHECK(handle(i, &h) == 0 && info(h, &m) == 0);
      std::printf("nvml device %u holds %llu MiB\n", i, m.used >> 20);
    }
    CK(cudaFree(mem));
    return 0;
  }
  CK(cudaGetDeviceCount(&n));
  CHECK(n >= 1);
  CU(cuInit(0));

  for (int dev = 0; dev < n; ++dev) {
    CUdevice d;
    CU(cuDeviceGet(&d, dev));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));

    // ---- the runtime and the driver say the same ----------------------------
    // cudaDeviceAttr's numbers are CUdevice_attribute's. Every attribute either
    // names must answer in both, with one value; a number neither names is
    // refused by both.
    for (const auto& a : kDrvAttrs) {
      int drv = -1, rt = -2;
      const CUresult dr = cuDeviceGetAttribute(&drv, (CUdevice_attribute)a.id, d);
      const cudaError_t rr = cudaDeviceGetAttribute(&rt, (cudaDeviceAttr)a.id, dev);
      if ((dr == CUDA_SUCCESS) != (rr == cudaSuccess) || (dr == CUDA_SUCCESS && drv != rt)) {
        std::printf("FAIL attribute %s (%d): driver %d (%d), runtime %d (%d)\n", a.name, a.id, drv, (int)dr, rt, (int)rr);
        return 1;
      }
      (void)cudaGetLastError();
    }
    for (int bad : {0, -1, 100000}) {
      int v = 5;
      CHECK(cuDeviceGetAttribute(&v, (CUdevice_attribute)bad, d) == CUDA_ERROR_INVALID_VALUE);
      CHECK(cudaDeviceGetAttribute(&v, (cudaDeviceAttr)bad, dev) == cudaErrorInvalidValue);
      (void)cudaGetLastError();
    }

    // ---- cudaDeviceProp says what the attributes say ------------------------
    int v = 0;
#define SAME(field, attr) do { CK(cudaDeviceGetAttribute(&v, (cudaDeviceAttr)(attr), dev)); \
  if ((long long)p.field != (long long)v) { std::printf("FAIL cudaDeviceProp::%s is %lld, attribute %d says %d\n", \
  #field, (long long)p.field, (int)(attr), v); return 1; } } while (0)
    SAME(memoryBusWidth, 37);
    SAME(l2CacheSize, 38);
    SAME(persistingL2CacheMaxSize, 108);
    SAME(accessPolicyMaxWindowSize, 109);
    SAME(pciBusID, 33);
    SAME(pciDeviceID, 34);
    SAME(pciDomainID, 50);
    SAME(asyncEngineCount, 40);
    SAME(ECCEnabled, 32);
    SAME(maxTexture1D, 21);
    SAME(maxTexture2D[0], 22);
    SAME(maxTexture2D[1], 23);
    SAME(maxTexture3D[0], 24);
    SAME(maxTexture3D[2], 26);
    SAME(maxTexture3DAlt[2], 49);
    SAME(maxSurface2D[0], 56);
    SAME(maxSurfaceCubemapLayered[1], 68);
    SAME(maxTexture2DLinear[2], 72);
    SAME(streamPrioritiesSupported, 78);
    SAME(globalL1CacheSupported, 79);
    SAME(localL1CacheSupported, 80);
    SAME(computePreemptionSupported, 90);
    SAME(cooperativeLaunch, 95);
    SAME(memoryPoolsSupported, 115);
    SAME(ipcEventSupported, 125);
    SAME(clusterLaunch, 120);
    SAME(sharedMemPerBlockOptin, 97);
    SAME(maxBlocksPerMultiProcessor, 106);
    SAME(multiProcessorCount, 16);
    SAME(textureAlignment, 14);
    SAME(surfaceAlignment, 30);
    SAME(texturePitchAlignment, 51);
    SAME(memPitch, 11);
#if CUDART_VERSION < 13000
    SAME(clockRate, 13);
    SAME(memoryClockRate, 36);
    SAME(singleToDoublePrecisionPerfRatio, 87);
    SAME(kernelExecTimeoutEnabled, 17);
#else
    SAME(gpuPciDeviceID, 139);
    SAME(gpuPciSubsystemID, 140);
#endif
    // The figures a program divides or sizes by are not zero.
    // (A profile that does not say its L2 reports none: B200, B300, GB200, Rubin, Thor, GB10 and A30, whose L2 NVIDIA has not published.)
    CHECK(p.memoryBusWidth > 0 && p.multiProcessorCount > 0 && p.l2CacheSize >= 0);
    CHECK(p.persistingL2CacheMaxSize <= p.l2CacheSize);

    // ---- one identity ---------------------------------------------------------
    CUuuid u;
    CU(cuDeviceGetUuid(&u, d));
    CHECK(uuid_hex((const unsigned char*)u.bytes) == uuid_hex((const unsigned char*)p.uuid.bytes));
    char rt_bus[32], drv_bus[32];
    CK(cudaDeviceGetPCIBusId(rt_bus, sizeof rt_bus, dev));
    CU(cuDeviceGetPCIBusId(drv_bus, sizeof drv_bus, d));
    CHECK(std::strcmp(rt_bus, drv_bus) == 0);
    char want[32];
    std::snprintf(want, sizeof want, "%04x:%02x:%02x.0", p.pciDomainID, p.pciBusID, p.pciDeviceID);
    CHECK(std::strcmp(rt_bus, want) == 0);
    int back = -1;
    CK(cudaDeviceGetByPCIBusId(&back, rt_bus));
    CHECK(back == dev);
    CUdevice dback = -1;
    CU(cuDeviceGetByPCIBusId(&dback, rt_bus));
    CHECK(dback == d);
    // A short buffer is refused, a well-formed address with no device is an invalid device,
    // and a malformed one an invalid value.
    char tiny[4];
    CHECK(cudaDeviceGetPCIBusId(tiny, sizeof tiny, dev) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    CHECK(cudaDeviceGetByPCIBusId(&back, "0000:ee:00.0") == cudaErrorInvalidDevice);
    CHECK(cudaDeviceGetByPCIBusId(&back, "not a bus id") == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    // Two devices have two identities.
    if (dev > 0) {
      cudaDeviceProp q;
      CK(cudaGetDeviceProperties(&q, dev - 1));
      CHECK(uuid_hex((const unsigned char*)q.uuid.bytes) != uuid_hex((const unsigned char*)p.uuid.bytes));
      CHECK(q.pciBusID != p.pciBusID);
    }
  }

  // ---- stream priorities: a range, kept and clamped, by both APIs ---------------
  CK(cudaFree(0));   // a context, which the driver calls below need
  int least = 99, greatest = 99;
  CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  CHECK(least == 0 && greatest < 0);
  int dl = 99, dg = 99;
  CU(cuCtxGetStreamPriorityRange(&dl, &dg));
  CHECK(dl == least && dg == greatest);
  cudaStream_t s;
  CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, greatest - 3));
  int prio = 99;
  CK(cudaStreamGetPriority(s, &prio));
  CHECK(prio == greatest);
  CK(cudaStreamDestroy(s));
  CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, least + 4));
  CK(cudaStreamGetPriority(s, &prio));
  CHECK(prio == least);
  CK(cudaStreamDestroy(s));
  CK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, greatest));
  CK(cudaStreamGetPriority(s, &prio));
  CHECK(prio == greatest);
  CK(cudaStreamDestroy(s));
  CUstream cs;
  CU(cuStreamCreateWithPriority(&cs, CU_STREAM_NON_BLOCKING, greatest - 1));
  CU(cuStreamGetPriority(cs, &prio));
  CHECK(prio == greatest);
  CU(cuStreamDestroy(cs));
  CU(cuStreamCreateWithPriority(&cs, CU_STREAM_NON_BLOCKING, -1));
  CU(cuStreamGetPriority(cs, &prio));
  CHECK(prio == -1);
  CU(cuStreamDestroy(cs));

  std::printf("PASS\n");
  return 0;
}
