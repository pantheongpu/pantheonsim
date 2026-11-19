// The resource check (VGPU_LEAK_CHECK, vgpu/leak_check.hpp) must say "clean" for a program that gives everything back, through every
// kind of handle this program uses, and name the resource for a program that does not. Case "clean" uses the runtime, the driver and
// cuBLAS (when built with -DWITH_CUBLAS) the way a program does, and frees all of it; each "leak-*" case keeps exactly one thing.
// run_leak_probe.sh runs one process per case and reads the line the simulator wrote.
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef WITH_CUBLAS
#include <cublas_v2.h>
#endif

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>

static int fails = 0;
#define WANT_OK(x) do { const int e_ = static_cast<int>(x); if (e_ != 0) { std::printf("FAIL %s -> %d\n", #x, e_); ++fails; } } while (0)

__global__ void bump(int* p) { p[threadIdx.x] += 1; }

static const char* kPtx = R"(
.version 7.0
.target sm_75
.address_size 64
.visible .entry harmless(.param .u64 p)
{
  ret;
}
)";

static void everything_and_give_it_back() {
  // Runtime: device, pinned, managed and stream-ordered memory; streams and events; kernels; a graph.
  int *d = nullptr, *h = nullptr, *m = nullptr, *a = nullptr;
  WANT_OK(cudaMalloc(&d, 1 << 20));
  WANT_OK(cudaMemset(d, 0, 1 << 20));
  WANT_OK(cudaMallocHost(&h, 1 << 16));
  WANT_OK(cudaMallocManaged(&m, 1 << 16));
  cudaStream_t s;
  cudaEvent_t e0, e1;
  WANT_OK(cudaStreamCreate(&s));
  WANT_OK(cudaEventCreate(&e0));
  WANT_OK(cudaEventCreate(&e1));
  WANT_OK(cudaMallocAsync(reinterpret_cast<void**>(&a), 1 << 16, s));
  WANT_OK(cudaEventRecord(e0, s));
  bump<<<2, 32, 0, s>>>(d);
  WANT_OK(cudaMemsetAsync(a, 0, 1 << 16, s));
  WANT_OK(cudaEventRecord(e1, s));
  WANT_OK(cudaFreeAsync(a, s));
  WANT_OK(cudaStreamSynchronize(s));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  WANT_OK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  bump<<<1, 32, 0, s>>>(d);
  WANT_OK(cudaStreamEndCapture(s, &g));
  WANT_OK(cudaGraphInstantiate(&ge, g, 0));
  WANT_OK(cudaGraphLaunch(ge, s));
  WANT_OK(cudaStreamSynchronize(s));
  WANT_OK(cudaGraphExecDestroy(ge));
  WANT_OK(cudaGraphDestroy(g));
  WANT_OK(cudaEventDestroy(e0));
  WANT_OK(cudaEventDestroy(e1));
  WANT_OK(cudaStreamDestroy(s));
  WANT_OK(cudaFree(m));
  WANT_OK(cudaFreeHost(h));
  WANT_OK(cudaFree(d));
#ifdef WITH_CUBLAS
  {
    cublasHandle_t hb;
    float *x = nullptr, *y = nullptr, *c = nullptr;
    WANT_OK(cublasCreate(&hb));
    WANT_OK(cudaMalloc(&x, 64 * 64 * 4));
    WANT_OK(cudaMalloc(&y, 64 * 64 * 4));
    WANT_OK(cudaMalloc(&c, 64 * 64 * 4));
    WANT_OK(cudaMemset(x, 0, 64 * 64 * 4));
    WANT_OK(cudaMemset(y, 0, 64 * 64 * 4));
    const float one = 1.f, zero = 0.f;
    WANT_OK(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, 64, 64, 64, &one, x, 64, y, 64, &zero, c, 64));
    WANT_OK(cudaDeviceSynchronize());
    WANT_OK(cublasDestroy(hb));
    WANT_OK(cudaFree(x));
    WANT_OK(cudaFree(y));
    WANT_OK(cudaFree(c));
  }
#endif
  // Driver: a module, a function, memory.
  CUmodule mod;
  CUfunction f;
  CUdeviceptr dp;
  WANT_OK(cuModuleLoadData(&mod, kPtx));
  WANT_OK(cuModuleGetFunction(&f, mod, "harmless"));
  WANT_OK(cuMemAlloc(&dp, 4096));
  void* args[] = {&dp};
  WANT_OK(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  WANT_OK(cuCtxSynchronize());
  WANT_OK(cuMemFree(dp));
  WANT_OK(cuModuleUnload(mod));
  // Threads and descriptors of the program's own, joined and closed.
  std::thread t([] { usleep(1000); });
  t.join();
  const int fd = open("/dev/null", O_RDONLY);
  close(fd);
}

int main(int argc, char** argv) {
  const char* which = argc > 1 ? argv[1] : "clean";
  WANT_OK(cuInit(0));
  everything_and_give_it_back();
  if (!std::strcmp(which, "clean")) {
    // nothing kept
  } else if (!std::strcmp(which, "leak-device")) {
    void* p = nullptr;
    WANT_OK(cudaMalloc(&p, 1 << 20));
  } else if (!std::strcmp(which, "leak-driver")) {
    CUdeviceptr p;
    WANT_OK(cuMemAlloc(&p, 4096));
  } else if (!std::strcmp(which, "leak-pinned")) {
    void* p = nullptr;
    WANT_OK(cudaMallocHost(&p, 4096));
  } else if (!std::strcmp(which, "leak-fd")) {
    if (open("/dev/null", O_RDONLY) < 0) ++fails;
  } else if (!std::strcmp(which, "leak-thread")) {
    static std::atomic<bool> stop{false};
    static std::thread* t = new std::thread([] { while (!stop) usleep(1000); });
    (void)t;
  } else {
    std::printf("FAIL: unknown case %s\n", which);
    return 2;
  }
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  std::fflush(stdout);
  return fails ? 1 : 0;
}
