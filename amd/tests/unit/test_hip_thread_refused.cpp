// A stream's work runs on a host thread of its own, started with the stream's first item. Where the host refuses the
// thread (EAGAIN at a process or user limit) the call must fail with an error code, leave the stream usable, and not
// throw out of the C entry point (an uncaught std::system_error is a std::terminate). RLIMIT_NPROC is the limit; a
// process allowed past it (root, for one) cannot make the host refuse, and the program says so and passes.
#include <sys/resource.h>

#include <cstdio>
#include <system_error>
#include <thread>

#include "vgpu_hip.h"

extern "C" hipError_t hipLaunchHostFunc(hipStream_t stream, void (*fn)(void*), void* data);

namespace {
int g_ran = 0;
void count(void*) { ++g_ran; }
}  // namespace

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
  // (a host function is the stream's first item that cannot be done on the caller's thread: it starts the thread; a
  // small fill on a stream with nothing in it is done where it is called, and starts none)
  const hipError_t refused = hipLaunchHostFunc(stream, count, nullptr);
  setrlimit(RLIMIT_NPROC, &saved);
  if (refused == hipSuccess) {
    std::printf("FAIL the call succeeded without a thread to run it\n");
    return 1;
  }
  // The stream is as it was: the same call now works, and the stream finishes it.
  if (hipLaunchHostFunc(stream, count, nullptr) != hipSuccess || hipStreamSynchronize(stream) != hipSuccess ||
      g_ran != 1) {
    std::printf("FAIL the stream is unusable after the refusal\n");
    return 1;
  }
  std::printf("ok a refused stream thread is an error code (%d), and the stream works afterwards\n", static_cast<int>(refused));
  return 0;
}
