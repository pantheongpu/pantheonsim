// HIP called from a static destructor, after main: what hipBLASLt does at
// exit, when it unloads its code objects. exit() destroys the thread's
// thread_local objects first, so state the HIP runtime or its profiler kept
// in one (the profiler's stack of calls in progress, the launch
// configurations) was written after it was freed -- and glibc found the heap
// damaged on the next free. Linked with the rocprofiler library, whose hooks
// see every HIP call, as they do in PyTorch. Under the sanitizer build, a
// use after free here is reported; elsewhere the program must exit cleanly.
#include <cstdio>
#include <cstdlib>

#include "vgpu_hip.h"

namespace {
struct CallsHipAtExit {
  ~CallsHipAtExit() {
    int n = 0;
    void* p = nullptr;
    const bool ok = hipGetDeviceCount(&n) == hipSuccess && n > 0 && hipMalloc(&p, 256) == hipSuccess &&
                    hipFree(p) == hipSuccess && hipDeviceSynchronize() == hipSuccess;
    std::printf("%s HIP from a static destructor, after thread_local objects are gone\n", ok ? "ok" : "FAIL");
    std::fflush(stdout);
    if (!ok) std::_Exit(1);
  }
} calls_hip_at_exit;
}  // namespace

int main() {
  // The first calls make the thread's state -- and the profiler's.
  int n = 0;
  void* p = nullptr;
  if (hipGetDeviceCount(&n) != hipSuccess || hipMalloc(&p, 256) != hipSuccess || hipFree(p) != hipSuccess) {
    std::printf("FAIL HIP in main\n");
    return 1;
  }
  std::printf("ok HIP in main\n");
  return 0;
}
