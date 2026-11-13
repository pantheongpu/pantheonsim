// Driver functions a program takes from cudaGetDriverEntryPointByVersion without linking libcuda (pantheonsim #604),
// as gputools does (src/cudaUtils.cpp). The simulator's runtime loads its libcuda for the first such call, and the
// functions it hands out must see what the runtime has done, as an RTX 3060 shows (CUDA 13.0 runtime):
//   - after runtime calls, the calling thread has the runtime's primary context current, and a module loads and runs;
//   - so has another thread the runtime used before the driver was loaded;
//   - a thread the runtime never used has no context (cuCtxGetCurrent succeeds with NULL);
//   - with no runtime call before it, cudaGetDriverEntryPoint still initialises the driver (cuDeviceGetCount works,
//     no context is current), and the first runtime call then makes the primary context current.
// The last case runs in a child process (this program with the argument "first"), since the driver is loaded once
// per process.
//
// Build: nvcc -cudart shared driver_entry_late.cu   (no -lcuda). Passes on an RTX 3060.
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

const char kPtx[] =
    ".version 7.0\n.target sm_80\n.address_size 64\n"
    ".visible .entry k(.param .u64 p) {\n"
    "  .reg .b64 %rd<3>; .reg .b32 %r<2>;\n"
    "  ld.param.u64 %rd1, [p]; cvta.to.global.u64 %rd2, %rd1;\n"
    "  mov.u32 %r1, 42; st.global.u32 [%rd2], %r1; ret;\n}\n";

template <typename F>
F entry(const char* name) {
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
  const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &fn, CUDART_VERSION, cudaEnableDefault, &q);
  if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || !fn) {
    std::printf("FAIL entry point %s: %d, found %d\n", name, static_cast<int>(e), static_cast<int>(q));
    ++failures;
  }
  return reinterpret_cast<F>(fn);
}

int first() {
  auto device_count = entry<decltype(&cuDeviceGetCount)>("cuDeviceGetCount");
  auto get_current = entry<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent");
  if (failures) return 1;
  int n = -1;
  const CUresult rn = device_count(&n);
  check(rn == CUDA_SUCCESS && n >= 1, "first: cudaGetDriverEntryPoint initialised the driver (cuDeviceGetCount)");
  CUcontext c = reinterpret_cast<CUcontext>(1);
  const CUresult rc = get_current(&c);
  check(rc == CUDA_SUCCESS && c == nullptr, "first: no context current before any runtime call");
  int* d = nullptr;
  check(cudaMalloc(&d, 4) == cudaSuccess, "first: cudaMalloc");
  c = nullptr;
  check(get_current(&c) == CUDA_SUCCESS && c != nullptr, "first: the runtime's call made its primary context current");
  cudaFree(d);
  return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "first") == 0) return first();

  // A thread the runtime uses before the driver is loaded, which calls the driver only afterwards.
  std::mutex mu;
  std::condition_variable cv;
  bool loaded = false;
  CUcontext worker_ctx = nullptr;
  CUresult worker_rc = CUDA_ERROR_UNKNOWN;
  decltype(&cuCtxGetCurrent) get_current = nullptr;
  std::thread worker([&] {
    cudaFree(nullptr);
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return loaded; });
    if (get_current) worker_rc = get_current(&worker_ctx);
  });

  int* d = nullptr;
  check(cudaMalloc(&d, 4) == cudaSuccess, "cudaMalloc");
  check(cudaFree(nullptr) == cudaSuccess, "cudaFree(0)");
  get_current = entry<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent");
  auto load_data = entry<decltype(&cuModuleLoadData)>("cuModuleLoadData");
  auto get_function = entry<decltype(&cuModuleGetFunction)>("cuModuleGetFunction");
  auto launch = entry<decltype(&cuLaunchKernel)>("cuLaunchKernel");
  auto unload = entry<decltype(&cuModuleUnload)>("cuModuleUnload");
  {
    std::lock_guard<std::mutex> lock(mu);
    loaded = true;
  }
  cv.notify_all();
  if (failures) {
    worker.join();
    std::printf("FAIL\n");
    return 1;
  }

  CUcontext cur = nullptr;
  const CUresult rc = get_current(&cur);
  check(rc == CUDA_SUCCESS && cur != nullptr, "cuCtxGetCurrent: the runtime's primary context is current");
  CUmodule m = nullptr;
  const CUresult rl = load_data(&m, kPtx);
  check(rl == CUDA_SUCCESS, "cuModuleLoadData");
  if (rl == CUDA_SUCCESS) {
    CUfunction f = nullptr;
    check(get_function(&f, m, "k") == CUDA_SUCCESS, "cuModuleGetFunction");
    void* args[] = {&d};
    check(launch(f, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS, "cuLaunchKernel");
    int h = 0;
    check(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost) == cudaSuccess && h == 42, "the kernel wrote 42");
    unload(m);
  }

  worker.join();
  check(worker_rc == CUDA_SUCCESS && worker_ctx == cur, "a thread the runtime used earlier has the same context");
  CUcontext fresh = reinterpret_cast<CUcontext>(1);
  CUresult fresh_rc = CUDA_ERROR_UNKNOWN;
  std::thread([&] { fresh_rc = get_current(&fresh); }).join();
  check(fresh_rc == CUDA_SUCCESS && fresh == nullptr, "a thread the runtime never used has no context");
  cudaFree(d);

  // The driver loaded before any runtime call: in a process of its own.
  std::fflush(stdout);
  const pid_t pid = fork();
  if (pid == 0) {
    execl("/proc/self/exe", argv[0], "first", static_cast<char*>(nullptr));
    _exit(127);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "the driver loaded before any runtime call (child process)");

  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
