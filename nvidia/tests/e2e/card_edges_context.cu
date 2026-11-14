// The driver's current context against the runtime's calls, as an RTX 3060 (CUDA 13.0) answers. Each group runs in a
// child process of its own (fork before anything initialises CUDA), since several leave the driver in a state no later
// group could start from.
//
//   - destroyed: cuCtxSetCurrent takes a destroyed context and succeeds (the simulator answered INVALID_CONTEXT); the
//     calls that need a context are then CUDA_ERROR_CONTEXT_IS_DESTROYED (709), and cuCtxGetCurrent hands the handle
//     back. (The card's process can crash at exit after this, so the child leaves with _exit.)
//   - reset: after cudaDeviceReset the primary context stays current but is inactive, and the driver's calls in it are
//     709 until the runtime's next call that uses the device (cudaFree(0), cudaSetDevice): the queries
//     (cudaGetDeviceCount, cudaGetDevice, cudaGetLastError) do not count. The simulator answered them all. The same
//     after cuDevicePrimaryCtxReset.
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
      {"cudaDeviceReset", [] { reset(0); }},
      {"cuDevicePrimaryCtxReset", [] { reset(1); }},
      {"cudaDeviceReset then cudaSetDevice", [] { reset(2); }},
      {"unset", unset},
      {"queries", queries},
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
