// The driver's current context against the runtime's calls, as an RTX 3060 (CUDA 13.0) answers. Each group runs in a
// child process of its own (fork before anything initialises CUDA), since several leave the driver in a state no later
// group could start from.
//
//   - destroyed: cuCtxSetCurrent takes a destroyed context and succeeds (the simulator answered INVALID_CONTEXT); the
//     calls that need a context are then CUDA_ERROR_CONTEXT_IS_DESTROYED (709), and cuCtxGetCurrent hands the handle
//     back. (The card's process can crash at exit after this, so the child leaves with _exit.) cuCtxGetDevice still
//     names the context's device (the simulator answered INVALID_CONTEXT); cuCtxGetId is 709, named or current, and
//     cuCtxGetApiVersion 709 for the current context and INVALID_CONTEXT for the handle (the simulator answered 201 for
//     both). The runtime's calls that use the device are cudaErrorDeviceUninitialized and leave the destroyed context
//     current (the simulator made the primary context current and succeeded); its queries succeed.
//   - reset: after cudaDeviceReset the primary context stays current but is inactive, and the driver's calls in it are
//     709 until the runtime's next call that uses the device (cudaFree(0), cudaSetDevice): the queries
//     (cudaGetDeviceCount, cudaGetDevice, cudaGetLastError) do not count. The simulator answered them all. The same
//     after cuDevicePrimaryCtxReset. Meanwhile cuCtxGetDevice answers, cuCtxGetFlags is 0 with flags 0 (the simulator
//     answered 709), and cuCtxGetId and cuCtxGetApiVersion answer as for a destroyed context.
//   - initdevice: cudaInitDevice makes the primary context active and current on no thread (the simulator did neither);
//     deviceFlags are its flags only with cudaInitDeviceFlagsAreValid, which is the only bit of `flags` looked at (the
//     simulator refused other bits), and then a bad value is cudaErrorInvalidValue with the context active anyway, and
//     current.
//   - unset: after cuCtxSetCurrent(NULL), the runtime's next call that uses the device makes the primary context current
//     again; a query does not, and a context of the program's own that is current stays current. The simulator left
//     the thread with no context.
//   - queries: cudaGetDeviceProperties, cudaDeviceGetAttribute and cudaGetDeviceCount on a fresh process make no
//     context current or active.
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

// cuCtxCreate gained a CUctxCreateParams argument in CUDA 13 (cuCtxCreate_v4); CUDA 12 headers have the 3-argument form.
#if CUDA_VERSION >= 13000
#define CTX_CREATE(pctx, dev) cuCtxCreate((pctx), nullptr, 0, (dev))
#else
#define CTX_CREATE(pctx, dev) cuCtxCreate((pctx), 0, (dev))
#endif

#include <cstdio>
#include <cstdlib>

static int fails = 0;
#define WANT(x, want) do { const long long got_ = static_cast<long long>(x); if (got_ != static_cast<long long>(want)) { \
  std::printf("FAIL %s line %d: %s -> %lld, expected %lld\n", group, __LINE__, #x, got_, static_cast<long long>(want)); \
  ++fails; } } while (0)

static const char* group = "";
static const int kDestroyed = 709;   // CUDA_ERROR_CONTEXT_IS_DESTROYED

static CUcontext current() {
  CUcontext c = nullptr;
  cuCtxGetCurrent(&c);
  return c;
}
static int active() {
  unsigned flags = 0;
  int on = -1;
  cuDevicePrimaryCtxGetState(0, &flags, &on);
  return on;
}

static void destroyed() {
  CUdevice dev;
  WANT(cuInit(0), CUDA_SUCCESS);
  WANT(cuDeviceGet(&dev, 0), CUDA_SUCCESS);
  CUcontext c = nullptr;
  WANT(CTX_CREATE(&c, dev), CUDA_SUCCESS);
  WANT(cuCtxDestroy(c), CUDA_SUCCESS);
  WANT(current() == nullptr, true);
  WANT(cuCtxSetCurrent(c), CUDA_SUCCESS);
  WANT(current() == c, true);
  CUdeviceptr p = 0;
  WANT(cuMemAlloc(&p, 16), kDestroyed);
  WANT(cuCtxSynchronize(), kDestroyed);
  CUstream s;
  WANT(cuStreamCreate(&s, 0), kDestroyed);
  CUevent e;
  WANT(cuEventCreate(&e, 0), kDestroyed);
  size_t fr = 0, to = 0;
  WANT(cuMemGetInfo(&fr, &to), kDestroyed);
  CUdevice d = 77;
  WANT(cuCtxGetDevice(&d), CUDA_SUCCESS);
  WANT(d, 0);
  unsigned long long id = 0;
  unsigned ver = 0;
  WANT(cuCtxGetId(c, &id), kDestroyed);
  WANT(cuCtxGetId(nullptr, &id), kDestroyed);
  WANT(cuCtxGetApiVersion(c, &ver), CUDA_ERROR_INVALID_CONTEXT);
  WANT(cuCtxGetApiVersion(nullptr, &ver), kDestroyed);
}

static void destroyed_runtime() {
  CUdevice dev;
  WANT(cuInit(0), CUDA_SUCCESS);
  WANT(cuDeviceGet(&dev, 0), CUDA_SUCCESS);
  CUcontext c = nullptr;
  WANT(CTX_CREATE(&c, dev), CUDA_SUCCESS);
  WANT(cuCtxDestroy(c), CUDA_SUCCESS);
  WANT(cuCtxSetCurrent(c), CUDA_SUCCESS);
  int n = 0;
  cudaDeviceProp prop;
  WANT(cudaGetDeviceCount(&n), cudaSuccess);
  WANT(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  WANT(cudaFree(nullptr), cudaErrorDeviceUninitialized);
  WANT(cudaGetLastError(), cudaErrorDeviceUninitialized);
  WANT(cudaFree(nullptr), cudaErrorDeviceUninitialized);
  void* p = nullptr;
  WANT(cudaMalloc(&p, 16), cudaErrorDeviceUninitialized);
  WANT(cudaMallocHost(&p, 16), cudaErrorDeviceUninitialized);
  WANT(cudaDeviceSynchronize(), cudaErrorDeviceUninitialized);
  cudaStream_t s;
  WANT(cudaStreamCreate(&s), cudaErrorDeviceUninitialized);
  cudaEvent_t e;
  WANT(cudaEventCreate(&e), cudaErrorDeviceUninitialized);
  size_t fr = 0, to = 0;
  WANT(cudaMemGetInfo(&fr, &to), cudaErrorDeviceUninitialized);
  WANT(cudaGetDevice(&n), cudaSuccess);
  WANT(current() == c, true);
}

static void reset(int how) {
  CUdevice dev;
  WANT(cudaFree(nullptr), cudaSuccess);
  WANT(cuDeviceGet(&dev, 0), CUDA_SUCCESS);
  const CUcontext prim = current();
  WANT(prim != nullptr, true);
  WANT(how == 1 ? static_cast<int>(cuDevicePrimaryCtxReset(dev)) : static_cast<int>(cudaDeviceReset()), 0);
  WANT(current() == prim, true);
  WANT(active(), 0);
  CUdeviceptr p = 0;
  WANT(cuMemAlloc(&p, 16), kDestroyed);
  WANT(cuCtxSynchronize(), kDestroyed);
  CUstream s;
  WANT(cuStreamCreate(&s, 0), kDestroyed);
  size_t fr = 0, to = 0;
  WANT(cuMemGetInfo(&fr, &to), kDestroyed);
  CUdevice d = 77;
  WANT(cuCtxGetDevice(&d), CUDA_SUCCESS);
  WANT(d, 0);
  unsigned f = 99, ver = 0;
  WANT(cuCtxGetFlags(&f), CUDA_SUCCESS);
  WANT(f, 0);
  unsigned long long id = 0;
  WANT(cuCtxGetId(nullptr, &id), kDestroyed);
  WANT(cuCtxGetId(prim, &id), kDestroyed);
  WANT(cuCtxGetApiVersion(prim, &ver), CUDA_ERROR_INVALID_CONTEXT);
  WANT(cuCtxGetApiVersion(nullptr, &ver), kDestroyed);
  int n = 0;
  WANT(cudaGetDeviceCount(&n), cudaSuccess);
  WANT(cuMemAlloc(&p, 16), kDestroyed);
  WANT(cudaGetDevice(&n), cudaSuccess);
  WANT(cudaGetLastError(), cudaSuccess);
  WANT(cuMemAlloc(&p, 16), kDestroyed);
  if (how == 2) {
    WANT(cudaSetDevice(0), cudaSuccess);
  } else {
    WANT(cudaFree(nullptr), cudaSuccess);
  }
  WANT(cuMemAlloc(&p, 16), CUDA_SUCCESS);
  WANT(cuMemFree(p), CUDA_SUCCESS);
  WANT(active(), 1);
  WANT(current() == prim, true);
}

static void unset() {
  WANT(cudaFree(nullptr), cudaSuccess);
  const CUcontext prim = current();
  WANT(prim != nullptr, true);
  int n = 0;
  WANT(cuCtxSetCurrent(nullptr), CUDA_SUCCESS);
  WANT(current() == nullptr, true);
  WANT(cudaGetDeviceCount(&n), cudaSuccess);
  WANT(cudaGetDevice(&n), cudaSuccess);
  WANT(cudaGetLastError(), cudaSuccess);
  WANT(current() == nullptr, true);
  int* q = nullptr;
  WANT(cudaMalloc(&q, 16), cudaSuccess);
  WANT(current() == prim, true);
  WANT(cuCtxSetCurrent(nullptr), CUDA_SUCCESS);
  WANT(cudaDeviceSynchronize(), cudaSuccess);
  WANT(current() == prim, true);
  WANT(cuCtxSetCurrent(nullptr), CUDA_SUCCESS);
  cudaStream_t st;
  WANT(cudaStreamCreate(&st), cudaSuccess);
  WANT(current() == prim, true);
  WANT(cudaStreamDestroy(st), cudaSuccess);
  WANT(cuCtxSetCurrent(nullptr), CUDA_SUCCESS);
  WANT(cudaFree(q), cudaSuccess);
  WANT(current() == prim, true);
  // A context of the program's own stays current under runtime calls; destroying it leaves none, and the next
  // runtime call brings the primary context back.
  CUdevice dev;
  WANT(cuDeviceGet(&dev, 0), CUDA_SUCCESS);
  CUcontext own = nullptr;
  WANT(cuCtxSetCurrent(nullptr), CUDA_SUCCESS);   // so that destroying it leaves nothing below
  WANT(CTX_CREATE(&own, dev), CUDA_SUCCESS);
  WANT(cudaMalloc(&q, 16), cudaSuccess);
  WANT(current() == own, true);
  WANT(cudaFree(q), cudaSuccess);
  WANT(cuCtxDestroy(own), CUDA_SUCCESS);
  WANT(current() == nullptr, true);
  WANT(cudaMalloc(&q, 16), cudaSuccess);
  WANT(current() == prim, true);
  WANT(cudaFree(q), cudaSuccess);
}

static unsigned primary_flags() {
  unsigned flags = 99;
  int on = -1;
  cuDevicePrimaryCtxGetState(0, &flags, &on);
  return flags;
}

// how: 0 no flags, 1 blocking sync applied, 2 a bad value applied, 3 a bad value with cudaInitDeviceFlagsAreValid
// not given (other bits of `flags` set), 4 a device that does not exist.
static void initdevice(int how) {
  const unsigned dev_flags[] = {0, cudaDeviceScheduleBlockingSync | cudaDeviceMapHost, 0x100, 0x3, 0};
  const unsigned flags[] = {0, cudaInitDeviceFlagsAreValid, cudaInitDeviceFlagsAreValid, 0x80000002u, 0};
  const cudaError_t want[] = {cudaSuccess, cudaSuccess, cudaErrorInvalidValue, cudaSuccess, cudaErrorInvalidDevice};
  WANT(cudaInitDevice(how == 4 ? 5 : 0, dev_flags[how], flags[how]), want[how]);
  WANT(cuInit(0), CUDA_SUCCESS);
  WANT(current() == nullptr, how != 2);   // (refused flags leave the primary context current)
  WANT(active(), how == 4 ? 0 : 1);
  WANT(primary_flags(), how == 1 ? 4 : 0);
  if (how == 4) return;
  int d = -1;
  WANT(cudaGetDevice(&d), cudaSuccess);
  WANT(current() == nullptr, how != 2);
  WANT(cudaFree(nullptr), cudaSuccess);
  WANT(current() != nullptr, true);
  unsigned f = 0;
  WANT(cuCtxGetFlags(&f), CUDA_SUCCESS);
  WANT(f, how == 1 ? 0xc : 0x8);
}

static void queries() {
  cudaDeviceProp p;
  int v = 0;
  WANT(cudaGetDeviceProperties(&p, 0), cudaSuccess);
  WANT(cudaDeviceGetAttribute(&v, cudaDevAttrMaxThreadsPerBlock, 0), cudaSuccess);
  WANT(cudaGetDeviceCount(&v), cudaSuccess);
  WANT(cuInit(0), CUDA_SUCCESS);
  WANT(current() == nullptr, true);
  WANT(active(), 0);
}

int main() {
  struct { const char* name; void (*run)(); } groups[] = {
      {"destroyed", destroyed},
      {"destroyed, runtime", destroyed_runtime},
      {"cudaDeviceReset", [] { reset(0); }},
      {"cuDevicePrimaryCtxReset", [] { reset(1); }},
      {"cudaDeviceReset then cudaSetDevice", [] { reset(2); }},
      {"unset", unset},
      {"queries", queries},
      {"cudaInitDevice", [] { initdevice(0); }},
      {"cudaInitDevice blocking sync", [] { initdevice(1); }},
      {"cudaInitDevice bad flags", [] { initdevice(2); }},
      {"cudaInitDevice flags not valid", [] { initdevice(3); }},
      {"cudaInitDevice bad device", [] { initdevice(4); }},
  };
  int failed = 0;
  for (const auto& g : groups) {
    std::fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
      group = g.name;
      g.run();
      std::fflush(stdout);
      _exit(fails ? 1 : 0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::printf("FAIL group %s (status %d)\n", g.name, status);
      ++failed;
    }
  }
  std::printf("%s\n", failed ? "FAILED" : "PASS");
  return failed ? 1 : 0;
}
