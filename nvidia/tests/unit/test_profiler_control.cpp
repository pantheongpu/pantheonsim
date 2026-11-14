// cuProfilerStart / cuProfilerStop and cudaProfilerStart / cudaProfilerStop answer as an RTX 3060 does (CUDA 13.0, driver
// 610.88). They used to succeed always, whatever the state of the driver or the context.
//
// Driver:  CUDA_ERROR_NOT_INITIALIZED (3) before cuInit; CUDA_ERROR_INVALID_CONTEXT (201) with no current context (a primary
//          context retained but not current, a thread the runtime has not used); CUDA_ERROR_CONTEXT_IS_DESTROYED (709) when
//          the current context is dead: one destroyed further down the stack and popped back to, a primary context released
//          to nothing, or reset (until the runtime's next call makes it active again); a dead context's fault; success
//          otherwise, Start twice and Stop with no Start too.
// Runtime: a runtime call like any other: it starts the runtime and makes the primary context current on its thread, answers
//          a dead context's fault and leaves it as the last error, and cudaErrorNoDevice (100) when CUDA_VISIBLE_DEVICES
//          shows no device.
//
// Each group runs in a process of its own (the CTest entries pass a filter): the driver's checks need a driver nothing has
// initialised, the fault kills the context, and the no-device case is an environment.
//
// No toolkit header is needed: the few prototypes are declared here.
#include <cstdint>
#include <thread>

#include "vtest.hpp"

extern "C" {
int cuInit(unsigned int flags);
int cuDeviceGet(int* dev, int ordinal);
int cuDevicePrimaryCtxRetain(void** ctx, int dev);
int cuDevicePrimaryCtxRelease_v2(int dev);
int cuDevicePrimaryCtxReset_v2(int dev);
int cuDevicePrimaryCtxGetState(int dev, unsigned* flags, int* active);
int cuCtxCreate_v2(void** ctx, unsigned flags, int dev);
int cuCtxDestroy_v2(void* ctx);
int cuCtxPopCurrent_v2(void** ctx);
int cuCtxSetCurrent(void* ctx);
int cuCtxGetCurrent(void** ctx);
int cuCtxSynchronize();
int cuModuleLoadData(void** mod, const void* image);
int cuModuleGetFunction(void** fn, void* mod, const char* name);
int cuModuleUnload(void* mod);
int cuLaunchKernel(void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned smem,
                   void* stream, void** params, void** extra);
int cuProfilerStart();
int cuProfilerStop();

int cudaFree(void* p);
int cudaGetLastError();
int cudaProfilerStart();
int cudaProfilerStop();
}

namespace {
constexpr int kSuccess = 0, kNotInitialized = 3, kNoDevice = 100, kInvalidContext = 201, kContextIsDestroyed = 709;

int on_new_thread(int (*fn)()) {
  int rc = -1;
  std::thread([&] { rc = fn(); }).join();
  return rc;
}
}  // namespace

// ---- the driver (filter "driver_"), in the order written ----

VTEST(driver_1_before_cuInit_is_not_initialized) {
  VCHECK_EQ(cuProfilerStart(), kNotInitialized);
  VCHECK_EQ(cuProfilerStop(), kNotInitialized);
}

VTEST(driver_2_without_a_current_context_is_invalid_context) {
  VCHECK_EQ(cuInit(0), kSuccess);
  VCHECK_EQ(cuProfilerStart(), kInvalidContext);
  VCHECK_EQ(cuProfilerStop(), kInvalidContext);
  int dev = -1;
  void* primary = nullptr;
  VCHECK_EQ(cuDeviceGet(&dev, 0), kSuccess);
  VCHECK_EQ(cuDevicePrimaryCtxRetain(&primary, dev), kSuccess);
  VCHECK_EQ(cuProfilerStart(), kInvalidContext);   // retained, not current
  VCHECK_EQ(cuCtxSetCurrent(primary), kSuccess);
  // Any number of times, and a Stop with no Start.
  VCHECK_EQ(cuProfilerStart(), kSuccess);
  VCHECK_EQ(cuProfilerStart(), kSuccess);
  VCHECK_EQ(cuProfilerStop(), kSuccess);
  VCHECK_EQ(cuProfilerStop(), kSuccess);
  VCHECK_EQ(cuProfilerStop(), kSuccess);
  // Another thread has no context.
  VCHECK_EQ(on_new_thread(cuProfilerStart), kInvalidContext);
  VCHECK_EQ(on_new_thread(cuProfilerStop), kInvalidContext);
  VCHECK_EQ(cuCtxSetCurrent(nullptr), kSuccess);
  VCHECK_EQ(cuProfilerStop(), kInvalidContext);
  // Released to nothing, and made current: no longer alive.
  VCHECK_EQ(cuDevicePrimaryCtxRelease_v2(dev), kSuccess);
  VCHECK_EQ(cuCtxSetCurrent(primary), kSuccess);
  VCHECK_EQ(cuProfilerStart(), kContextIsDestroyed);
  VCHECK_EQ(cuProfilerStop(), kContextIsDestroyed);
  VCHECK_EQ(cuCtxSetCurrent(nullptr), kSuccess);
}

VTEST(driver_3_a_destroyed_current_context_is_context_is_destroyed) {
  int dev = -1;
  VCHECK_EQ(cuDeviceGet(&dev, 0), kSuccess);
  void* a = nullptr;
  void* b = nullptr;
  VCHECK_EQ(cuCtxCreate_v2(&a, 0, dev), kSuccess);
  VCHECK_EQ(cuProfilerStart(), kSuccess);
  VCHECK_EQ(cuProfilerStop(), kSuccess);
  VCHECK_EQ(cuCtxCreate_v2(&b, 0, dev), kSuccess);   // pushed over a
  VCHECK_EQ(cuCtxDestroy_v2(a), kSuccess);          // further down: b stays current
  VCHECK_EQ(cuProfilerStart(), kSuccess);
  void* popped = nullptr;
  VCHECK_EQ(cuCtxPopCurrent_v2(&popped), kSuccess);
  VCHECK(popped == b);
  void* cur = nullptr;
  VCHECK_EQ(cuCtxGetCurrent(&cur), kSuccess);
  VCHECK(cur == a);                                 // the destroyed one, handed back
  VCHECK_EQ(cuProfilerStart(), kContextIsDestroyed);
  VCHECK_EQ(cuProfilerStop(), kContextIsDestroyed);
  VCHECK_EQ(cuCtxSetCurrent(nullptr), kSuccess);
  VCHECK_EQ(cuCtxDestroy_v2(b), kSuccess);
}

// A reset primary context is dead until the runtime uses the device again, which makes it active (cudaFree(0) on the card).
VTEST(driver_4_a_reset_primary_context_lives_again_once_used) {
  int dev = -1;
  void* primary = nullptr;
  VCHECK_EQ(cuDeviceGet(&dev, 0), kSuccess);
  VCHECK_EQ(cudaFree(nullptr), kSuccess);           // the runtime's primary context, current on this thread
  void* cur = nullptr;
  VCHECK_EQ(cuCtxGetCurrent(&cur), kSuccess);
  VCHECK(cur != nullptr);
  VCHECK_EQ(cuProfilerStart(), kSuccess);
  VCHECK_EQ(cuDevicePrimaryCtxRetain(&primary, dev), kSuccess);
  VCHECK(primary == cur);
  VCHECK_EQ(cuDevicePrimaryCtxReset_v2(dev), kSuccess);
  unsigned flags = 0;
  int active = -1;
  VCHECK_EQ(cuDevicePrimaryCtxGetState(dev, &flags, &active), kSuccess);
  VCHECK_EQ(active, 0);
  VCHECK_EQ(cuProfilerStart(), kContextIsDestroyed);
  VCHECK_EQ(cudaFree(nullptr), kSuccess);
  VCHECK_EQ(cuDevicePrimaryCtxGetState(dev, &flags, &active), kSuccess);
  VCHECK_EQ(active, 1);
  VCHECK_EQ(cuProfilerStart(), kSuccess);
  VCHECK_EQ(cuProfilerStop(), kSuccess);
  VCHECK_EQ(cuDevicePrimaryCtxRelease_v2(dev), kSuccess);
}

// ---- the runtime (filter "runtime_") ----

VTEST(runtime_start_is_a_runtime_call) {
  // The process's first CUDA call: it starts the runtime, which initialises the driver and makes the primary context
  // current on this thread.
  VCHECK_EQ(cudaProfilerStart(), kSuccess);
  void* cur = nullptr;
  VCHECK_EQ(cuCtxGetCurrent(&cur), kSuccess);
  VCHECK(cur != nullptr);
  VCHECK_EQ(cudaProfilerStop(), kSuccess);
  VCHECK_EQ(cudaProfilerStop(), kSuccess);
  VCHECK_EQ(cudaProfilerStart(), kSuccess);
  VCHECK_EQ(cuProfilerStart(), kSuccess);
  // A thread the runtime has not used: the same.
  void* other = nullptr;
  int rc = -1, get = -1;
  std::thread([&] {
    rc = cudaProfilerStart();
    get = cuCtxGetCurrent(&other);
  }).join();
  VCHECK_EQ(rc, kSuccess);
  VCHECK_EQ(get, kSuccess);
  VCHECK(other == cur);
  VCHECK_EQ(cudaGetLastError(), kSuccess);
}

// ---- a dead context (filter "sticky_") ----

VTEST(sticky_both_answer_the_fault) {
  const char* ptx = R"(
.version 8.0
.target sm_75
.address_size 64
.visible .entry die()
{
  trap;
}
)";
  VCHECK_EQ(cudaFree(nullptr), kSuccess);
  void* mod = nullptr;
  void* die = nullptr;
  VCHECK_EQ(cuModuleLoadData(&mod, ptx), kSuccess);
  VCHECK_EQ(cuModuleGetFunction(&die, mod, "die"), kSuccess);
  VCHECK_EQ(cuLaunchKernel(die, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr), kSuccess);
  const int fault = cuCtxSynchronize();
  VCHECK(fault != kSuccess);
  VCHECK_EQ(cudaProfilerStart(), fault);
  VCHECK_EQ(cudaProfilerStop(), fault);
  VCHECK_EQ(cuProfilerStart(), fault);
  VCHECK_EQ(cuProfilerStop(), fault);
  VCHECK_EQ(cudaGetLastError(), fault);
  cuModuleUnload(mod);
}

// ---- no device (filter "no_device_", CUDA_VISIBLE_DEVICES empty) ----

VTEST(no_device_is_no_device) {
  VCHECK_EQ(cudaProfilerStart(), kNoDevice);
  VCHECK_EQ(cudaGetLastError(), kNoDevice);
  VCHECK_EQ(cudaProfilerStop(), kNoDevice);
  VCHECK_EQ(cuProfilerStart(), kNotInitialized);
}

VTEST_MAIN
