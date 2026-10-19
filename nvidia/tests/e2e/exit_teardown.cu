// The driver API called after the calling thread's thread_local destructors have run: from a thread_local
// object's destructor that is destroyed after the driver library's per-thread state, and, on the main thread,
// from a static object's destructor at exit() (glibc destroys the exiting thread's thread_local objects before
// it runs the static destructors). StringZilla's device scope does the latter: it makes the context current and
// destroys its stream when the process exits. The context a thread made current must still be there, and the
// shim must not touch freed memory to find it -- it did, through a destroyed std::vector, which corrupted the
// heap and ended such a process with "malloc_consolidate(): unaligned fastbin chunk detected".
//
// Prints PASS before the process exits and "exit-time: context kept" from the static destructor after it.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unistd.h>

static CUcontext g_ctx = nullptr;
static CUcontext g_seen_in_worker = reinterpret_cast<CUcontext>(~uintptr_t{0});

// Constructed (and so registered for destruction) before the driver library's own thread-local state in that
// thread, so it is destroyed after it.
struct Sentinel {
  void touch() {}
  ~Sentinel() {
    CUcontext c = nullptr;
    cuCtxGetCurrent(&c);
    g_seen_in_worker = c;
  }
};

static void worker() {
  thread_local Sentinel sentinel;
  sentinel.touch();
  cuCtxSetCurrent(g_ctx);
}

struct AtExit {
  ~AtExit() {
    CUcontext c = nullptr;
    cuCtxGetCurrent(&c);   // what main() made current, before anything is written here
    cuCtxSetCurrent(g_ctx);   // and what StringZilla's destructor does next
    const char* msg = c == g_ctx ? "exit-time: context kept\n" : "exit-time: context LOST\n";
    const ssize_t n = write(1, msg, std::char_traits<char>::length(msg));
    (void)n;
    if (c != g_ctx) _exit(2);
  }
};

int main() {
  if (cuInit(0) != CUDA_SUCCESS) { std::printf("FAIL: cuInit\n"); return 1; }
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  if (cuDevicePrimaryCtxRetain(&g_ctx, dev) != CUDA_SUCCESS || !g_ctx) { std::printf("FAIL: no context\n"); return 1; }
  int fails = 0;

  std::thread t(worker);
  t.join();
  if (g_seen_in_worker != g_ctx) {
    std::printf("FAIL: a thread_local destroyed after the driver's per-thread state saw context %p, not %p\n",
                static_cast<void*>(g_seen_in_worker), static_cast<void*>(g_ctx));
    ++fails;
  }

  // The main thread: make the context current (the driver's per-thread state now exists), and leave a static object
  // whose destructor uses it at exit.
  cuCtxSetCurrent(g_ctx);
  static AtExit at_exit;
  (void)at_exit;
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  std::fflush(stdout);
  return fails ? 1 : 0;
}
