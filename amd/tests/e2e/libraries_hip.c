/*
 * What ROCm's libraries ask of HIP when PyTorch loads them: streams with a
 * priority or a compute-unit mask, host memory a kernel reaches (registered,
 * pinned, managed), memory pools and their counts, virtual memory mapped by
 * hand, one fact about an address, the device's PCI address and limits, a
 * kernel's name, a host function in stream order, a capture's id, and IPC.
 *
 * It links against VirtualGPU's libamdhip64. The calls beyond vgpu_hip.h are
 * declared here, from HIP's documented signatures and layouts.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vgpu_hip.h"

typedef struct { int type; int id; } hipMemLocation;
typedef struct {
  int type;
  int requestedHandleType;
  hipMemLocation location;
  void* win32HandleMetaData;
  unsigned char compressionType, gpuDirectRDMACapable;
  unsigned short usage;
} hipMemAllocationProp;
typedef struct { hipMemLocation location; int flags; } hipMemAccessDesc;
typedef struct {
  int allocType;
  int handleTypes;
  hipMemLocation location;
  void* win32SecurityAttributes;
  size_t maxSize;
  unsigned char reserved[56];
} hipMemPoolProps;
typedef struct { int type; int device; void* devicePointer; void* hostPointer; int isManaged; unsigned flags; }
    hipPointerAttribute_t;
typedef struct { char reserved[64]; } hipIpcMemHandle_t;

hipError_t hipStreamCreateWithPriority(hipStream_t*, unsigned int, int);
hipError_t hipStreamGetPriority(hipStream_t, int*);
hipError_t hipStreamGetFlags(hipStream_t, unsigned int*);
hipError_t hipStreamGetDevice(hipStream_t, hipDevice_t*);
hipError_t hipDeviceGetStreamPriorityRange(int*, int*);
hipError_t hipExtStreamCreateWithCUMask(hipStream_t*, unsigned int, const unsigned int*);
hipError_t hipExtStreamGetCUMask(hipStream_t, unsigned int, unsigned int*);
hipError_t hipHostRegister(void*, size_t, unsigned int);
hipError_t hipHostUnregister(void*);
hipError_t hipHostMalloc(void**, size_t, unsigned int);
hipError_t hipHostFree(void*);
hipError_t hipHostGetDevicePointer(void**, void*, unsigned int);
hipError_t hipMallocManaged(void**, size_t, unsigned int);
hipError_t hipPointerGetAttributes(hipPointerAttribute_t*, const void*);
hipError_t hipPointerGetAttribute(void*, int, void*);
hipError_t hipMemPoolCreate(void**, const hipMemPoolProps*);
hipError_t hipMemPoolDestroy(void*);
hipError_t hipMemPoolGetAttribute(void*, int, void*);
hipError_t hipMemPoolSetAttribute(void*, int, void*);
hipError_t hipMallocFromPoolAsync(void**, size_t, void*, hipStream_t);
hipError_t hipFreeAsync(void*, hipStream_t);
hipError_t hipMemGetAllocationGranularity(size_t*, const hipMemAllocationProp*, int);
hipError_t hipMemAddressReserve(void**, size_t, size_t, void*, unsigned long long);
hipError_t hipMemAddressFree(void*, size_t);
hipError_t hipMemCreate(void**, size_t, const hipMemAllocationProp*, unsigned long long);
hipError_t hipMemRelease(void*);
hipError_t hipMemMap(void*, size_t, size_t, void*, unsigned long long);
hipError_t hipMemUnmap(void*, size_t);
hipError_t hipMemSetAccess(void*, size_t, const hipMemAccessDesc*, size_t);
hipError_t hipMemGetAddressRange(void**, size_t*, void*);
hipError_t hipMemsetD32Async(void*, int, size_t, hipStream_t);
hipError_t hipStreamWriteValue32(hipStream_t, void*, unsigned int, unsigned int);
hipError_t hipDeviceGetPCIBusId(char*, int, int);
hipError_t hipDeviceGetByPCIBusId(int*, const char*);
hipError_t hipDeviceSetLimit(int, size_t);
hipError_t hipDeviceGetLimit(size_t*, int);
const char* hipKernelNameRef(hipFunction_t);
hipError_t hipLaunchHostFunc(hipStream_t, void (*)(void*), void*);
hipError_t hipStreamBeginCapture(hipStream_t, int);
hipError_t hipStreamEndCapture(hipStream_t, void**);
hipError_t hipStreamGetCaptureInfo(hipStream_t, int*, unsigned long long*);
hipError_t hipGraphDestroy(void*);
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t*, void*);
hipError_t hipIpcOpenMemHandle(void**, hipIpcMemHandle_t, unsigned int);

#define CHECK(call)                                                                                \
  do {                                                                                             \
    hipError_t _e = (call);                                                                        \
    if (_e != hipSuccess) {                                                                        \
      fprintf(stderr, "%s failed: %s (%s)\n", #call, hipGetErrorString(_e), hipGetErrorName(_e)); \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

enum { N = 64 };

/* dyn_lds(in, out, n): out[t] = 2 * in[(t + 1) % n], through LDS the launch
 * pays for. Right for every element, or not. */
static int run_dyn_lds(hipFunction_t f, const float* in, float* out) {
  int n = N;
  void* args[] = {&in, &out, &n};
  if (hipModuleLaunchKernel(f, 1, 1, 1, N, 1, 1, N * sizeof(float), NULL, args, NULL) != hipSuccess) return 0;
  return 1;
}
static int right(const float* in, const float* out) {
  for (int t = 0; t < N; ++t)
    if (out[t] != 2.0f * in[(t + 1) % N]) return 0;
  return 1;
}
static void set_flag(void* p) { *(int*)p = 7; }

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: libraries_hip <memory code object>\n");
    return 2;
  }
  hipModule_t module;
  hipFunction_t dyn;
  CHECK(hipModuleLoad(&module, argv[1]));
  CHECK(hipModuleGetFunction(&dyn, module, "dyn_lds"));

  /* Streams. */
  int least = 0, greatest = 0, priority = 0;
  unsigned flags = 9;
  hipDevice_t on = -1;
  hipStream_t high;
  CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
  CHECK(hipStreamCreateWithPriority(&high, hipStreamNonBlocking, greatest));
  CHECK(hipStreamGetPriority(high, &priority));
  CHECK(hipStreamGetFlags(high, &flags));
  CHECK(hipStreamGetDevice(high, &on));
  printf("a stream keeps its priority, flags and device %d\n",
         least == 1 && greatest == -1 && priority == -1 && flags == hipStreamNonBlocking && on == 0);
  unsigned mask[10] = {0}, want = 0xF0F0u;
  hipStream_t masked;
  CHECK(hipExtStreamCreateWithCUMask(&masked, 1, &want));
  CHECK(hipExtStreamGetCUMask(masked, 10, mask));
  int kept = mask[0] == want && mask[1] == 0;
  CHECK(hipExtStreamGetCUMask(NULL, 10, mask));
  int cus = 0;
  for (int w = 0; w < 10; ++w) cus += __builtin_popcount(mask[w]);
  printf("a stream keeps its compute-unit mask, and the null stream has every unit %d\n", kept && cus == 304);
  CHECK(hipStreamDestroy(high));
  printf("a destroyed stream is no longer known %d\n", hipStreamGetPriority(high, &priority) != hipSuccess);

  /* Host memory kernels reach at its own address: registered, pinned and
   * managed. */
  float* in = malloc(N * sizeof(float));
  float* out = malloc(N * sizeof(float));
  for (int t = 0; t < N; ++t) in[t] = (float)t, out[t] = -1.0f;
  CHECK(hipHostRegister(in, N * sizeof(float), 0));
  CHECK(hipHostRegister(out, N * sizeof(float), 0));
  void *din = NULL, *dout = NULL;
  CHECK(hipHostGetDevicePointer(&din, in, 0));
  CHECK(hipHostGetDevicePointer(&dout, out, 0));
  CHECK(run_dyn_lds(dyn, din, dout) ? hipSuccess : hipErrorLaunchFailure);
  printf("a kernel reads and writes registered host memory %d\n", din == in && right(in, out));
  hipPointerAttribute_t attr;
  CHECK(hipPointerGetAttributes(&attr, in + 3));
  printf("registered memory is host memory, at its own address %d\n",
         attr.type == 1 && attr.devicePointer == in + 3 && !attr.isManaged);
  printf("registering it twice is refused %d\n", hipHostRegister(in, 8, 0) == 712);
  CHECK(hipHostUnregister(in));
  CHECK(hipHostUnregister(out));
  printf("and unregistering what is not registered %d\n", hipHostUnregister(in) == 713);

  float *pin = NULL, *pout = NULL;
  CHECK(hipHostMalloc((void**)&pin, N * sizeof(float), 0));
  CHECK(hipHostMalloc((void**)&pout, N * sizeof(float), 0));
  for (int t = 0; t < N; ++t) pin[t] = (float)(3 * t);
  CHECK(run_dyn_lds(dyn, pin, pout) ? hipSuccess : hipErrorLaunchFailure);
  printf("a kernel reads and writes pinned host memory %d\n", right(pin, pout));
  CHECK(hipHostFree(pin));
  CHECK(hipHostFree(pout));

  float *min = NULL, *mout = NULL;
  CHECK(hipMallocManaged((void**)&min, N * sizeof(float), 1));
  CHECK(hipMallocManaged((void**)&mout, N * sizeof(float), 1));
  for (int t = 0; t < N; ++t) min[t] = (float)(t - 20);
  CHECK(run_dyn_lds(dyn, min, mout) ? hipSuccess : hipErrorLaunchFailure);
  CHECK(hipPointerGetAttributes(&attr, mout));
  printf("a kernel reads and writes managed memory, which says it is managed %d\n",
         right(min, mout) && attr.type == 3 && attr.isManaged == 1);
  CHECK(hipFree(min));
  CHECK(hipFree(mout));

  /* A pool counts what is taken from it. */
  hipMemPoolProps props;
  memset(&props, 0, sizeof props);
  props.allocType = 1;
  props.location.type = 1;
  props.location.id = 0;
  void* pool = NULL;
  void* p = NULL;
  unsigned long long used = 1, high_mark = 0, zero = 0;
  CHECK(hipMemPoolCreate(&pool, &props));
  CHECK(hipMallocFromPoolAsync(&p, 1 << 20, pool, NULL));
  CHECK(hipMemPoolGetAttribute(pool, 7, &used));
  int counted = used == (1u << 20);
  CHECK(hipFreeAsync(p, NULL));
  CHECK(hipMemPoolGetAttribute(pool, 7, &used));
  CHECK(hipMemPoolGetAttribute(pool, 8, &high_mark));
  int returned = used == 0 && high_mark == (1u << 20);
  CHECK(hipMemPoolSetAttribute(pool, 8, &zero));
  CHECK(hipMemPoolGetAttribute(pool, 8, &high_mark));
  printf("a pool counts what is in use and its high mark %d\n", counted && returned && high_mark == 0);
  CHECK(hipMemPoolDestroy(pool));

  /* Virtual memory: address space reserved, memory made, one mapped into the
   * other and opened for access, then written and read back. */
  hipMemAllocationProp vprop;
  memset(&vprop, 0, sizeof vprop);
  vprop.type = 1;
  vprop.location.type = 1;
  vprop.location.id = 0;
  size_t granularity = 0;
  CHECK(hipMemGetAllocationGranularity(&granularity, &vprop, 0));
  void *va = NULL, *handle = NULL;
  CHECK(hipMemAddressReserve(&va, 2 * granularity, 0, NULL, 0));
  CHECK(hipMemCreate(&handle, granularity, &vprop, 0));
  CHECK(hipMemMap(va, granularity, 0, handle, 0));
  hipMemAccessDesc access = {{1, 0}, 3};
  CHECK(hipMemSetAccess(va, granularity, &access, 1));
  CHECK(hipMemsetD32Async(va, 0x12345678, granularity / 4, NULL));
  unsigned words[4] = {0};
  CHECK(hipMemcpy(words, (char*)va + granularity - 16, sizeof words, hipMemcpyDeviceToHost));
  printf("virtual memory mapped by hand is written and read back %d\n",
         granularity > 0 && words[0] == 0x12345678u && words[3] == 0x12345678u);
  CHECK(hipMemUnmap(va, granularity));
  CHECK(hipMemRelease(handle));
  CHECK(hipMemAddressFree(va, 2 * granularity));

  /* One allocation, asked about from inside it. */
  char* buf = NULL;
  void* base = NULL;
  size_t size = 0, range = 0;
  int ordinal = -1;
  CHECK(hipMalloc((void**)&buf, 1000));
  CHECK(hipMemGetAddressRange(&base, &size, buf + 100));
  CHECK(hipPointerGetAttribute(&ordinal, 9, buf + 100));
  CHECK(hipPointerGetAttribute(&range, 12, buf + 100));
  printf("an address knows its allocation and its device %d\n",
         base == buf && size == 1000 && ordinal == 0 && range == 1000);
  CHECK(hipStreamWriteValue32(NULL, buf + 8, 0xCAFEu, 0));
  unsigned value = 0;
  CHECK(hipMemcpy(&value, buf + 8, 4, hipMemcpyDeviceToHost));
  printf("a value written in stream order lands %d\n", value == 0xCAFEu);

  /* The device's PCI address, both ways; its limits; a kernel's name. */
  int devices = 0, found_all = 1;
  CHECK(hipGetDeviceCount(&devices));
  for (int d = 0; d < devices; ++d) {
    char bus[32];
    int back = -1;
    CHECK(hipDeviceGetPCIBusId(bus, sizeof bus, d));
    CHECK(hipDeviceGetByPCIBusId(&back, bus));
    found_all &= back == d;
  }
  printf("each device is found by its PCI address %d\n", found_all);
  size_t stack = 0;
  CHECK(hipDeviceSetLimit(0, 4096));
  CHECK(hipDeviceGetLimit(&stack, 0));
  printf("a limit is kept, and one that cannot be set is refused %d\n",
         stack == 4096 && hipDeviceSetLimit(1, 1) == 215);
  printf("a kernel is named %s\n", hipKernelNameRef(dyn));

  /* A host function runs in stream order, and a capture has an id. */
  int flag = 0;
  CHECK(hipLaunchHostFunc(masked, set_flag, &flag));
  printf("a host function runs in stream order %d\n", flag == 7);
  int status = 0;
  unsigned long long id = 0;
  void* graph = NULL;
  CHECK(hipStreamBeginCapture(masked, 0));
  CHECK(hipStreamGetCaptureInfo(masked, &status, &id));
  CHECK(hipStreamEndCapture(masked, &graph));
  int during = status == 1 && id != 0;
  CHECK(hipStreamGetCaptureInfo(masked, &status, &id));
  printf("a capture is active, with an id, until it ends %d\n", during && status == 0);
  CHECK(hipGraphDestroy(graph));

  /* IPC: a handle is made, and the process that made it cannot open it. */
  hipIpcMemHandle_t ipc;
  void* opened = NULL;
  CHECK(hipIpcGetMemHandle(&ipc, buf));
  printf("an IPC handle is made, and its own process may not open it %d\n",
         hipIpcOpenMemHandle(&opened, ipc, 1) != hipSuccess);
  CHECK(hipFree(buf));
  CHECK(hipStreamDestroy(masked));
  CHECK(hipModuleUnload(module));
  free(in);
  free(out);
  return 0;
}
