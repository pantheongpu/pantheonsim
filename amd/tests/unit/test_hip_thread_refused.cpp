// A stream's work runs on a host thread of its own, started with the stream's first item. Where the host refuses the
// thread (EAGAIN at a process or user limit) the call must fail with an error code, leave the stream usable, and not
// throw out of the C entry point (an uncaught std::system_error is a std::terminate). RLIMIT_NPROC is the limit; a
// process allowed past it (root, for one) cannot make the host refuse, and the program says so and passes.
#include <sys/resource.h>

#include <cstdio>
#include <system_error>
#include <thread>

#include "vgpu_hip.h"

int main() {
  void* p = nullptr;
  hipStream_t stream = nullptr;
  if (hipMalloc(&p, 256) != hipSuccess || hipStreamCreate(&stream) != hipSuccess) {
    std::printf("FAIL setup\n");
    return 1;
  }
  rlimit saved{};
  getrlimit(RLIMIT_NPROC, &saved);
  rlimit low = saved;
  low.rlim_cur = 1;
  bool effective = false;
  if (setrlimit(RLIMIT_NPROC, &low) == 0) {
    try {
      std::thread([] {}).join();
    } catch (const std::system_error&) {
      effective = true;
    }
  }
  if (!effective) {
    setrlimit(RLIMIT_NPROC, &saved);
    std::printf("SKIP: this process is allowed to start threads past RLIMIT_NPROC\n");
    return 0;
  }
  const hipError_t refused = hipMemsetAsync(p, 0, 256, stream);   // (the stream's first item: it would start the thread)
  setrlimit(RLIMIT_NPROC, &saved);
  if (refused == hipSuccess) {
    std::printf("FAIL the call succeeded without a thread to run it\n");
    return 1;
  }
  // The stream is as it was: the same call now works, and the stream finishes it.
  if (hipMemsetAsync(p, 0, 256, stream) != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) {
    std::printf("FAIL the stream is unusable after the refusal\n");
    return 1;
  }
  std::printf("ok a refused stream thread is an error code (%d), and the stream works afterwards\n", static_cast<int>(refused));
  return 0;
}
