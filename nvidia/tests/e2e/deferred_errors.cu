// A kernel's fault is reported after its launch, not by it. On the card a
// launch returns before the kernel runs, so a kernel that fails a device
// assert, traps or reads an illegal address launches successfully; the fault
// kills the context, and the calls after it report it. Every value here is an
// RTX 3060's:
//
//   - the launch returns success, and the last error stays clear -- both
//     cudaGetLastError and cudaPeekAtLastError report success;
//   - the next call that needs the context fails with the fault, and makes it
//     the last error: cudaDeviceSynchronize, a copy, an allocation, a stream or
//     event call, the next launch;
//   - reading the last error clears it, and the call after fails again;
//   - which device is current, and what a device is, still answer;
//   - a failed assert is cudaErrorAssert (710), a trap cudaErrorLaunchFailure
//     (719), an illegal address cudaErrorIllegalAddress (700);
//   - runtime and driver share the context: a kernel the driver launched kills
//     it for the runtime too.
//
// The simulator used to fail the launch itself, which is how CUDA Samples'
// simpleAssert_nvrtc failed: its cuLaunchKernel is wrapped in a check that
// exits on any error.
//
// A dead context stays dead until a reset, so each case runs in a process of
// its own: run_deferred_errors.sh runs this once per case.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstring>

static int fails = 0;
static int* g_dev = nullptr;   // a live allocation from before the fault
#define WANT(x, want) do { const int got_ = static_cast<int>(x); if (got_ != static_cast<int>(want)) { \
  std::printf("FAIL %s -> %d, expected %s (%d)\n", #x, got_, #want, static_cast<int>(want)); ++fails; } } while (0)

__global__ void fail_assert(const int* p) { assert(p == nullptr); }
__global__ void fail_trap() { __trap(); }
__global__ void fail_address(int* p) { *p = 1; }
__global__ void store(int* p, int v) { *p = v; }

#ifndef DEFERRED_ERRORS_RUNTIME_ONLY
// The driver's kernels, as PTX: a trap, and a harmless one. A sanitizer build
// compiles the runtime cases alone (run_deferred_errors.sh), without libcuda:
// the two libraries' copies of the core are an ODR violation to ASan.
static const char* kPtx = R"(
.version 7.0
.target sm_75
.address_size 64
.visible .entry trap_now()
{
  trap;
}
.visible .entry harmless(.param .u64 p)
{
  ret;
}
)";
#endif

// What a dead context answers, from the runtime, after the call that first
// reported the fault. `code` is the fault.
static void runtime_after(cudaError_t code) {
  WANT(cudaGetLastError(), code);
  WANT(cudaGetLastError(), cudaSuccess);   // read, so cleared
  WANT(cudaPeekAtLastError(), cudaSuccess);
  void* p = nullptr;
  WANT(cudaMalloc(&p, 16), code);
  WANT(cudaPeekAtLastError(), code);       // a failed call sets it again
  WANT(cudaGetLastError(), code);
  int h = 0;
  WANT(cudaMemcpy(&h, g_dev, sizeof h, cudaMemcpyDeviceToHost), code);
  WANT(cudaDeviceSynchronize(), code);
  WANT(cudaStreamQuery(0), code);
  cudaStream_t s = nullptr;
  WANT(cudaStreamCreate(&s), code);
  cudaEvent_t e = nullptr;
  WANT(cudaEventCreate(&e), code);
  store<<<1, 1>>>(nullptr, 0);
  WANT(cudaGetLastError(), code);          // the next launch fails at once
  cudaGetLastError();
  int dev = -1, sms = 0, count = 0;
  WANT(cudaGetDevice(&dev), cudaSuccess);
  WANT(cudaSetDevice(0), cudaSuccess);
  WANT(cudaGetDeviceCount(&count), cudaSuccess);
  WANT(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), cudaSuccess);
  cudaDeviceProp prop;
  WANT(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  WANT(cudaGetLastError(), cudaSuccess);
}

int main(int argc, char** argv) {
  const char* which = argc > 1 ? argv[1] : "";
  int* d = nullptr;
  if (cudaMalloc(&d, sizeof(int)) != cudaSuccess || !(g_dev = d)) {
    std::printf("FAIL cudaMalloc\n");
    return 1;
  }
  if (!std::strcmp(which, "assert") || !std::strcmp(which, "trap") || !std::strcmp(which, "address")) {
    const cudaError_t code = !std::strcmp(which, "assert") ? cudaErrorAssert
                             : !std::strcmp(which, "trap") ? cudaErrorLaunchFailure
                                                           : cudaErrorIllegalAddress;
    if (code == cudaErrorAssert) fail_assert<<<1, 1>>>(d);
    else if (code == cudaErrorLaunchFailure) fail_trap<<<1, 1>>>();
    else fail_address<<<1, 1>>>(reinterpret_cast<int*>(16));
    WANT(cudaPeekAtLastError(), cudaSuccess);
    WANT(cudaGetLastError(), cudaSuccess);
    WANT(cudaDeviceSynchronize(), code);
    runtime_after(code);
  } else if (!std::strcmp(which, "launch-ex")) {
    // The extended launch, on a stream of its own; its synchronize reports it.
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(1);
    cfg.stream = s;
    const int* arg = d;
    WANT(cudaLaunchKernelEx(&cfg, fail_assert, arg), cudaSuccess);
    WANT(cudaStreamSynchronize(s), cudaErrorAssert);
    runtime_after(cudaErrorAssert);
  } else if (!std::strcmp(which, "graph")) {
    // A graph whose kernel fails: the graph's launch succeeds, and its
    // stream's synchronize reports the fault.
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    cudaGraph_t g = nullptr;
    cudaGraphExec_t x = nullptr;
    WANT(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), cudaSuccess);
    store<<<1, 1, 0, s>>>(d, 1);
    fail_assert<<<1, 1, 0, s>>>(d);
    WANT(cudaStreamEndCapture(s, &g), cudaSuccess);
    WANT(cudaGraphInstantiate(&x, g, 0), cudaSuccess);
    WANT(cudaGraphLaunch(x, s), cudaSuccess);
    WANT(cudaGetLastError(), cudaSuccess);
    WANT(cudaStreamSynchronize(s), cudaErrorAssert);
    WANT(cudaGraphLaunch(x, s), cudaErrorAssert);   // the context is dead
    runtime_after(cudaErrorAssert);
#ifndef DEFERRED_ERRORS_RUNTIME_ONLY
  } else if (!std::strcmp(which, "driver")) {
    // The driver API: cuLaunchKernel succeeds, cuCtxSynchronize reports the
    // trap, and so does the runtime, which shares the context.
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    CUmodule mod = nullptr;
    CUfunction trap_now = nullptr, harmless = nullptr;
    WANT(cuInit(0), CUDA_SUCCESS);
    WANT(cuDeviceGet(&dev, 0), CUDA_SUCCESS);
    WANT(cuDevicePrimaryCtxRetain(&ctx, dev), CUDA_SUCCESS);
    WANT(cuCtxSetCurrent(ctx), CUDA_SUCCESS);
    WANT(cuModuleLoadData(&mod, kPtx), CUDA_SUCCESS);
    WANT(cuModuleGetFunction(&trap_now, mod, "trap_now"), CUDA_SUCCESS);
    WANT(cuModuleGetFunction(&harmless, mod, "harmless"), CUDA_SUCCESS);
    WANT(cuLaunchKernel(trap_now, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr), CUDA_SUCCESS);
    WANT(cuCtxSynchronize(), CUDA_ERROR_LAUNCH_FAILED);
    WANT(cuCtxSynchronize(), CUDA_ERROR_LAUNCH_FAILED);
    CUdeviceptr p = 0;
    WANT(cuMemAlloc(&p, 16), CUDA_ERROR_LAUNCH_FAILED);
    void* params[] = {&p};
    WANT(cuLaunchKernel(harmless, 1, 1, 1, 1, 1, 1, 0, nullptr, params, nullptr), CUDA_ERROR_LAUNCH_FAILED);
    WANT(cuStreamQuery(nullptr), CUDA_ERROR_LAUNCH_FAILED);
    CUstream s = nullptr;
    WANT(cuStreamCreate(&s, 0), CUDA_ERROR_LAUNCH_FAILED);
    CUdevice cur = -1;
    WANT(cuCtxGetDevice(&cur), CUDA_SUCCESS);
    int sms = 0;
    WANT(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), CUDA_SUCCESS);
    CUcontext now = nullptr;
    WANT(cuCtxGetCurrent(&now), CUDA_SUCCESS);
    // The runtime has not seen it yet: its last error is clear until a call
    // that needs the context.
    WANT(cudaGetLastError(), cudaSuccess);
    WANT(cudaDeviceSynchronize(), cudaErrorLaunchFailure);
    runtime_after(cudaErrorLaunchFailure);
#endif
  } else {
    std::printf("FAIL unknown case '%s'\n", which);
    return 1;
  }
  std::printf(fails ? "FAIL\n" : "PASS\n");
  return fails != 0;
}
