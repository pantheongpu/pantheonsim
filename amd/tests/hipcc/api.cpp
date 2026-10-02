// HIP's calls beyond the everyday ones, answered as ROCm's HIP answers them on
// an MI300-class device: what AMD's own tests (hip-tests) hold it to. Each
// check prints "ok <what>" or "FAIL <what>: <why>", and the last line counts
// them. Built by build.sh with hipcc; run on two simulated MI300Xs by
// amd/tests/e2e/run_hipcc.sh.
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
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

__global__ void takes_nothing() {}
__global__ void __launch_bounds__(64) bounded(int* p) {
  if (p) p[threadIdx.x] = 1;
}
// A static __constant__ variable is reached through the code object's global
// offset table, which the loader fills in when it places the image.
__device__ static __constant__ float static_const[4];
__global__ void where_static_const(void** out) { *out = static_cast<void*>(static_const); }
__global__ void add_one(int* p, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1;
}
__global__ void add_vectors(const int* a, const int* b, int* c, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
// Spins on the real-time clock for `ms` milliseconds at the wall clock rate
// the runtime reports, as hip-tests' delay kernels do.
__global__ void delay(uint64_t ms, uint64_t ticks_per_ms) {
  for (uint64_t m = 0; m < ms; ++m) {
    const uint64_t start = wall_clock64();
    while (wall_clock64() - start < ticks_per_ms) __builtin_amdgcn_s_sleep(10);
  }
}

// A callback on the thread's own stream is told that stream by its own
// handle, not the reserved hipStreamPerThread.
static int host_seen = 0;
static void callback(hipStream_t stream, hipError_t status, void* data) {
  const bool told = stream != nullptr && stream != hipStreamPerThread;
  host_seen = told && status == hipSuccess ? *static_cast<int*>(data) : -1;
}

int main() {
  int count = 0;
  (void)hipGetDeviceCount(&count);
  check(count == 2, "two devices");
  hipDeviceProp_t props;
  (void)hipGetDeviceProperties(&props, 0);

  // ---- Devices
  EXPECT(hipInit(1), hipErrorInvalidValue, "hipInit refuses a flag");
  unsigned flags = 99;
  (void)hipGetDeviceFlags(&flags);
  check(flags == hipDeviceScheduleSpin, "device flags start at Spin", std::to_string(flags));
  (void)hipSetDeviceFlags(hipDeviceScheduleBlockingSync | hipDeviceMapHost);
  (void)hipGetDeviceFlags(&flags);
  check(flags == hipDeviceScheduleBlockingSync, "hipSetDeviceFlags keeps the schedule bits", std::to_string(flags));
  hipUUID uuid;
  (void)hipDeviceGetUuid(&uuid, 1);
  check(std::string(uuid.bytes, 16) == "5647505500000001", "a device's UUID is its HSA agent's",
        std::string(uuid.bytes, 16));
  check(std::memcmp(props.uuid.bytes, "5647505500000000", 16) == 0, "the properties carry the same UUID");
  int major = 0, minor = 0, chosen = -1;
  (void)hipDeviceComputeCapability(&major, &minor, 0);
  check(major == 9 && minor == 4, "gfx942 is 9.4");
  EXPECT(hipChooseDevice(&chosen, &props), hipSuccess, "hipChooseDevice");
  int rate = 0, wait_value = 0;
  (void)hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0);
  (void)hipDeviceGetAttribute(&wait_value, hipDeviceAttributeCanUseStreamWaitValue, 0);
  check(rate == 100000, "the wall clock runs at 100 MHz", std::to_string(rate));
  check(wait_value == 1, "streams can wait on a value");
  char bus[16];
  std::memset(bus, 1, sizeof bus);
  EXPECT(hipDeviceGetPCIBusId(bus, 6, 0), hipErrorInvalidValue, "a short PCI bus id buffer is refused");
  check(std::strcmp(bus, "0000:") == 0 && bus[6] == 1, "and holds what fit, and nothing past it");
  size_t limit = 0;
  EXPECT(hipDeviceSetLimit(static_cast<hipLimit_t>(0xffff), 1), hipErrorInvalidValue,
         "a limit past hipLimitRange is no limit");
  EXPECT(hipDeviceGetLimit(&limit, static_cast<hipLimit_t>(-1)), hipErrorUnsupportedLimit,
         "an unknown limit is unsupported");
  (void)hipDeviceSetLimit(hipExtLimitScratchCurrent, 16384);
  (void)hipDeviceGetLimit(&limit, hipExtLimitScratchCurrent);
  check(limit == 16384, "the scratch limit reads back as set");

  // ---- Errors
  check(std::string(hipGetErrorName(hipErrorContextIsDestroyed)) == "hipErrorContextIsDestroyed",
        "every error has its name");
  check(std::string(hipGetErrorString(hipErrorNotInitialized)) == "initialization error",
        "and ROCm's text for it");
  const char* name = nullptr;
  EXPECT(hipDrvGetErrorName(static_cast<hipError_t>(-1), &name), hipErrorInvalidValue,
         "the driver form refuses a value that is no error");

  // ---- Contexts: a stack per thread
  hipCtx_t ctx = nullptr, popped = nullptr;
  (void)hipCtxCreate(&ctx, hipDeviceMapHost | hipDeviceScheduleSpin, 1);
  int dev = -1;
  (void)hipGetDevice(&dev);
  check(dev == 0, "a new context leaves the current device as it was");
  (void)hipSetDevice(1);
  (void)hipGetDeviceFlags(&flags);
  (void)hipSetDevice(0);
  check(flags == (hipDeviceMapHost | hipDeviceScheduleSpin), "and gives its device its flags, whole");
  EXPECT(hipCtxSynchronize(), hipErrorNotSupported, "hipCtxSynchronize is not supported");
  (void)hipCtxPopCurrent(&popped);
  check(popped == ctx, "popping gives the context back");
  EXPECT(hipCtxPopCurrent(&popped), hipErrorInvalidContext, "an empty stack has nothing to pop");
  (void)hipCtxPushCurrent(ctx);
  (void)hipGetDevice(&dev);
  check(dev == 1, "pushing a context makes its device current");
  (void)hipCtxPopCurrent(&popped);
  EXPECT(hipDevicePrimaryCtxSetFlags(0, 0), hipErrorContextAlreadyInUse, "the primary context is in use");
  (void)hipSetDevice(0);

  // ---- Streams: the reserved handles, ids, attributes
  unsigned long long id_null = 1, id_legacy = 2, id_thread = 0, id_a = 0, id_b = 0;
  (void)hipStreamGetId(nullptr, &id_null);
  (void)hipStreamGetId(hipStreamLegacy, &id_legacy);
  (void)hipStreamGetId(hipStreamPerThread, &id_thread);
  check(id_null == id_legacy, "the null handle and hipStreamLegacy are one stream");
  check(id_thread != id_null, "the per-thread stream is another");
  unsigned long long id_other_thread = id_thread;
  std::thread([&] { (void)hipStreamGetId(hipStreamPerThread, &id_other_thread); }).join();
  check(id_other_thread != id_thread, "and each thread has its own");
  hipStream_t a, b;
  (void)hipStreamCreate(&a);
  (void)hipStreamGetId(a, &id_a);
  (void)hipStreamDestroy(a);
  (void)hipStreamCreate(&b);
  (void)hipStreamGetId(b, &id_b);
  check(id_a != id_b, "a stream's id is never used again");
  EXPECT(hipStreamDestroy(nullptr), hipErrorInvalidResourceHandle, "the null stream cannot be destroyed");
  unsigned stream_flags = 0;
  EXPECT(hipStreamGetFlags(nullptr, &stream_flags), hipErrorInvalidValue, "the null stream has no flags to give");
  hipStreamAttrValue v{}, got{};
  v.syncPolicy = hipSyncPolicyYield;
  (void)hipStreamSetAttribute(b, hipStreamAttributeSynchronizationPolicy, &v);
  (void)hipStreamGetAttribute(b, hipStreamAttributeSynchronizationPolicy, &got);
  check(got.syncPolicy == hipSyncPolicyYield, "a stream attribute reads back as set");

  // Work on the legacy handle, the per-thread stream, and a callback that
  // sees what the stream's kernel did.
  const int n = 1000;
  int *d = nullptr, *e = nullptr, *f = nullptr;
  (void)hipMalloc(&d, n * sizeof(int));
  (void)hipMalloc(&e, n * sizeof(int));
  (void)hipMalloc(&f, n * sizeof(int));
  std::vector<int> h(n, 5), out(n, 0);
  (void)hipMemcpyAsync(d, h.data(), n * sizeof(int), hipMemcpyHostToDevice, hipStreamLegacy);
  add_one<<<(n + 255) / 256, 256, 0, hipStreamPerThread>>>(d, n);
  int* pinned = nullptr;
  (void)hipHostMalloc(&pinned, sizeof(int));
  (void)hipMemcpyAsync(pinned, d + 7, sizeof(int), hipMemcpyDeviceToHost, hipStreamPerThread);
  (void)hipStreamAddCallback(hipStreamPerThread, callback, pinned, 0);
  (void)hipStreamSynchronize(hipStreamPerThread);
  check(host_seen == 6, "a callback runs after the stream's work, and sees it", std::to_string(host_seen));
  EXPECT(hipStreamAddCallback(b, callback, pinned, 1), hipErrorInvalidValue, "a callback takes no flags");

  // A copy that says host to host into device memory copies into device memory.
  (void)hipMemcpy(e, h.data(), n * sizeof(int), hipMemcpyHostToHost);
  (void)hipMemcpy(out.data(), e, n * sizeof(int), hipMemcpyDeviceToHost);
  check(out[n - 1] == 5, "a copy goes where its addresses are, whatever its kind says");

  // A 2D copy in its stream's order: after the kernel before it.
  add_one<<<(n + 255) / 256, 256, 0, b>>>(d, n);
  (void)hipMemcpy2DAsync(out.data(), 40, d, 40, 40, n / 10, hipMemcpyDeviceToHost, b);
  (void)hipStreamSynchronize(b);
  check(out[0] == 7 && out[n - 1] == 7, "a 2D copy runs in its stream's order", std::to_string(out[0]));

  // ---- Waiting on memory in a stream, released by the host
  unsigned* target = nullptr;
  (void)hipHostMalloc(&target, sizeof(unsigned));
  *target = 0;
  hipEvent_t before, after;
  (void)hipEventCreate(&before);
  (void)hipEventCreate(&after);
  (void)hipStreamWriteValue32(b, pinned, 11, 0);
  (void)hipEventRecord(before, b);
  (void)hipStreamWaitValue32(b, target, 3, hipStreamWaitValueGte, 0xFFFFFFFF);
  (void)hipStreamWriteValue32(b, pinned, 12, 0);
  (void)hipEventRecord(after, b);
  (void)hipEventSynchronize(before);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  check(*pinned == 11 && hipEventQuery(after) == hipErrorNotReady, "a stream waits for a value");
  *target = 4;
  (void)hipStreamSynchronize(b);
  check(*pinned == 12, "and goes on once memory holds it");
  EXPECT(hipStreamWaitValue32(reinterpret_cast<hipStream_t>(0xFFFF), target, 0, 0, 0xFFFFFFFF),
         hipErrorContextIsDestroyed, "a handle that is no stream is a destroyed context");

  // ---- Copies between device memory: the host does not wait, unless asked
  hipStream_t slow;
  (void)hipStreamCreate(&slow);
  delay<<<1, 1, 0, slow>>>(500, rate);
  (void)hipMemcpy(f, e, n * sizeof(int), hipMemcpyDeviceToDevice);
  check(hipStreamQuery(slow) == hipErrorNotReady, "a device-to-device copy returns before the device is done");
  (void)hipStreamSynchronize(slow);
  int sync_memops = 1;
  (void)hipPointerSetAttribute(&sync_memops, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS, reinterpret_cast<hipDeviceptr_t>(f));
  delay<<<1, 1, 0, slow>>>(500, rate);
  (void)hipMemcpy(f, e, n * sizeof(int), hipMemcpyDeviceToDevice);
  check(hipStreamQuery(slow) == hipSuccess, "with SYNC_MEMOPS it waits for the copy, and what it follows");

  // ---- Pinned memory by its other names
  void* p = reinterpret_cast<void*>(1);
  EXPECT(hipHostAlloc(&p, 64, hipHostMallocCoherent), hipErrorInvalidValue, "hipHostAlloc refuses coherence flags");
  (void)hipHostAlloc(&p, 0, 0);
  check(p == nullptr, "no bytes is no allocation");
  (void)hipHostAlloc(&p, 256, hipHostAllocMapped | hipHostAllocWriteCombined);
  unsigned host_flags = 0;
  (void)hipHostGetFlags(&host_flags, static_cast<char*>(p) + 100);
  check(host_flags == (hipHostAllocMapped | hipHostAllocWriteCombined), "hipHostGetFlags gives the flags back");
  (void)hipFreeHost(p);
  (void)hipMallocHost(&p, 128);
  EXPECT(hipFree(p), hipSuccess, "hipFree frees pinned memory too");

  // ---- Launches, every way there is
  int* c = nullptr;
  (void)hipMalloc(&c, n * sizeof(int));
  (void)hipConfigureCall(dim3((n + 127) / 128), dim3(128), 0, nullptr);
  (void)hipSetupArgument(&d, sizeof d, 0);
  (void)hipSetupArgument(&e, sizeof e, 8);
  (void)hipSetupArgument(&c, sizeof c, 16);
  (void)hipSetupArgument(&n, sizeof n, 24);
  EXPECT(hipLaunchByPtr(reinterpret_cast<const void*>(add_vectors)), hipSuccess, "hipLaunchByPtr");
  (void)hipMemcpy(out.data(), c, n * sizeof(int), hipMemcpyDeviceToHost);
  check(out[3] == 12, "with the arguments set up one by one", std::to_string(out[3]));
  hipFunction_t fn = nullptr;
  (void)hipGetFuncBySymbol(&fn, reinterpret_cast<const void*>(add_one));
  check(fn && std::string(hipKernelNameRef(fn)).find("add_one") != std::string::npos,
        "hipGetFuncBySymbol finds the kernel");
  struct {
    int* p;
    int n;
  } packed{c, n};
  size_t size = sizeof packed;
  void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &packed, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  (void)hipModuleLaunchKernel(fn, (n + 127) / 128, 1, 1, 128, 1, 1, 0, nullptr, nullptr, extra);
  hipLaunchAttribute attr{};
  attr.id = hipLaunchAttributeCooperative;
  attr.val.cooperative = 1;
  hipLaunchConfig_t config{dim3((n + 127) / 128), dim3(128), 0, nullptr, &attr, 1};
  void* args[] = {&c, const_cast<int*>(&n)};
  EXPECT(hipLaunchKernelExC(&config, reinterpret_cast<const void*>(add_one), args), hipSuccess,
         "hipLaunchKernelExC, cooperatively");
  (void)hipMemcpy(out.data(), c, n * sizeof(int), hipMemcpyDeviceToHost);
  check(out[3] == 14, "both launches ran", std::to_string(out[3]));
  EXPECT(hipLaunchKernel(reinterpret_cast<const void*>(add_one), dim3(1), dim3(1025), args, 0, nullptr),
         hipErrorInvalidConfiguration, "a block past the device's limit is refused");
  bounded<<<1, 128>>>(nullptr);
  EXPECT(hipGetLastError(), hipErrorLaunchFailure, "a block past the kernel's launch bounds is a launch failure");
  bounded<<<1, 64>>>(nullptr);
  EXPECT(hipGetLastError(), hipSuccess, "and one inside them launches");
  hipStream_t gone;
  (void)hipStreamCreate(&gone);
  (void)hipStreamDestroy(gone);
  EXPECT(hipLaunchKernel(reinterpret_cast<const void*>(add_one), dim3(1), dim3(1), args, 0, gone),
         hipErrorInvalidValue, "a launch on a destroyed stream is refused");

  // One kernel on each device.
  hipStream_t s0, s1;
  int* c1 = nullptr;
  (void)hipSetDevice(1);
  (void)hipStreamCreate(&s1);
  (void)hipMalloc(&c1, n * sizeof(int));
  (void)hipMemset(c1, 0, n * sizeof(int));
  (void)hipSetDevice(0);
  (void)hipStreamCreate(&s0);
  void* args0[] = {&c, const_cast<int*>(&n)};
  void* args1[] = {&c1, const_cast<int*>(&n)};
  hipLaunchParams list[2] = {
      {reinterpret_cast<void*>(add_one), dim3((n + 127) / 128), dim3(128), args0, 0, s0},
      {reinterpret_cast<void*>(add_one), dim3((n + 127) / 128), dim3(128), args1, 0, s1}};
  EXPECT(hipExtLaunchMultiKernelMultiDevice(list, 2, 0), hipSuccess, "one kernel on each device");
  (void)hipStreamSynchronize(s0);
  (void)hipStreamSynchronize(s1);
  (void)hipMemcpy(out.data(), c1, n * sizeof(int), hipMemcpyDeviceToHost);
  check(out[3] == 1, "the second device's ran on the second device");
  list[1].stream = s0;
  EXPECT(hipExtLaunchMultiKernelMultiDevice(list, 2, 0), hipErrorInvalidDevice, "two on one device are refused");

  // hipExtMallocWithFlags: fine-grained and uncached memory is device memory,
  // which RCCL fills and shares between processes; signal memory is host
  // memory the host reads directly. ROCm 7.1's HIP answers the same.
  (void)hipSetDevice(0);
  for (unsigned kind : {1u, 2u, 3u}) {
    void* m = nullptr;
    const size_t bytes = kind == 2 ? 8 : 4096;
    const std::string what = "hipExtMallocWithFlags kind " + std::to_string(kind);
    EXPECT(hipExtMallocWithFlags(&m, bytes, kind), hipSuccess, what.c_str());
    hipPointerAttribute_t attr{};
    (void)hipPointerGetAttributes(&attr, m);
    check(attr.type == (kind == 2 ? hipMemoryTypeHost : hipMemoryTypeDevice), (what + " is where ROCm puts it").c_str(),
          std::to_string(attr.type));
    EXPECT(hipMemset(m, 7, bytes), hipSuccess, (what + " can be filled").c_str());
    unsigned host_flags = 99;
    EXPECT(hipHostGetFlags(&host_flags, m), kind == 2 ? hipSuccess : hipErrorInvalidValue,
           (what + ": hipHostGetFlags").c_str());
    if (kind == 2) check(host_flags == hipHostMallocMapped, "signal memory is mapped host memory");
    if (kind != 2) {
      hipIpcMemHandle_t handle;
      EXPECT(hipIpcGetMemHandle(&handle, m), hipSuccess, (what + " can be shared with another process").c_str());
    }
    (void)hipFree(m);
  }

  EXPECT(hipUnbindTexture(nullptr), hipErrorInvalidValue, "unbinding no texture");

  char api_name[64];
  std::snprintf(api_name, sizeof api_name, "%s", hipApiName(1));
  check(std::string(api_name) == "__hipPopCallConfiguration" && std::string(hipApiName(0)) == "unknown",
        "hipApiName names HIP's functions by their trace id");

  // A kernel that takes nothing of the program's is launched with no
  // arguments at all: what the compiler adds is the runtime's to fill in.
  EXPECT(hipLaunchKernel(reinterpret_cast<const void*>(takes_nothing), dim3(1), dim3(1), nullptr, 0, nullptr),
         hipSuccess, "a kernel without parameters launches with no argument array");
  EXPECT(hipLaunchCooperativeKernel(reinterpret_cast<const void*>(takes_nothing), dim3(1), dim3(1), nullptr, 0,
                                    nullptr),
         hipSuccess, "cooperatively too");
  (void)hipDeviceSynchronize();
  EXPECT(hipFuncSetAttribute(reinterpret_cast<const void*>(takes_nothing), hipFuncAttributeMax, 90), hipSuccess,
         "hipFuncAttributeMax is taken, as ROCm takes it");
  EXPECT(hipFuncSetAttribute(reinterpret_cast<const void*>(takes_nothing), static_cast<hipFuncAttribute>(-1), 90),
         hipErrorInvalidValue, "an attribute that is none is not");
  void** where_d = nullptr;
  (void)hipMalloc(&where_d, sizeof(void*));
  where_static_const<<<1, 1>>>(where_d);
  void* seen = nullptr;
  (void)hipMemcpy(&seen, where_d, sizeof seen, hipMemcpyDeviceToHost);
  void* named = nullptr;
  (void)hipGetSymbolAddress(&named, HIP_SYMBOL(static_const));
  check(seen && seen == named, "a kernel finds a static __constant__ variable where hipGetSymbolAddress says it is");
  (void)hipFree(where_d);
  hipEvent_t gone_event = nullptr;
  (void)hipEventCreate(&gone_event);
  (void)hipEventDestroy(gone_event);
  EXPECT(hipExtLaunchKernel(reinterpret_cast<const void*>(takes_nothing), dim3(1), dim3(1), nullptr, 0, nullptr, gone_event,
                            nullptr, 0),
         hipErrorInvalidValue, "hipExtLaunchKernel with an event that is gone");
  (void)hipGetLastError();
  hipEvent_t waited = nullptr;
  (void)hipEventCreate(&waited);
  EXPECT(hipStreamWaitEvent(nullptr, waited, ~0u), hipErrorInvalidValue, "a wait with flags that are none");
  (void)hipEventDestroy(waited);
  hipStream_t masked_none = nullptr;
  const uint32_t no_cus[4] = {0, 0, 0, 0};
  (void)hipExtStreamCreateWithCUMask(&masked_none, 4, no_cus);
  uint32_t got_mask[4] = {0, 0, 0, 0};
  (void)hipExtStreamGetCUMask(masked_none, 4, got_mask);
  check(got_mask[0] == 0xFFFFFFFFu, "a CU mask with no unit in it gives the stream every unit");
  (void)hipStreamDestroy(masked_none);
  (void)hipGetLastError();

  std::printf("api: %d checks, %d failed\n", checks, failures);
  return failures ? 1 : 0;
}
