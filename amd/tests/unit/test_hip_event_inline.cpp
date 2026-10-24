// hipEventRecord on a stream with nothing in it takes its timestamp on the calling thread (hip_api.cpp): the event is
// complete when the call returns, as a card's is on an idle stream, and the stream's thread is not started for it. On a
// stream with work ahead of it, or behind work the legacy null stream has queued, the record still waits its turn.
#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <system_error>
#include <thread>

#include "vgpu_hip.h"

extern "C" hipError_t hipLaunchHostFunc(hipStream_t stream, void (*fn)(void*), void* data);

namespace {
int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++failures;
  }
}
void hold(void* flag) {
  auto* release = static_cast<std::atomic<bool>*>(flag);
  while (!release->load()) std::this_thread::sleep_for(std::chrono::microseconds(100));
}
}  // namespace

int main() {
  // On a stream nothing was queued on, a record is complete at once, every time (a fresh stream each time, so that its
  // thread has certainly not started).
  for (int i = 0; i < 100; ++i) {
    hipStream_t s = nullptr;
    hipEvent_t e0 = nullptr, e1 = nullptr;
    check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking) == hipSuccess, "stream");
    check(hipEventCreate(&e0) == hipSuccess && hipEventCreate(&e1) == hipSuccess, "events");
    check(hipEventRecord(e0, s) == hipSuccess, "record e0");
    check(hipEventQuery(e0) == hipSuccess, "an event recorded on an idle stream is complete when the call returns");
    check(hipEventRecord(e1, s) == hipSuccess, "record e1");
    float ms = -1;
    check(hipEventElapsedTime(&ms, e0, e1) == hipSuccess && ms >= 0, "the time between two records on an idle stream");
    check(hipEventSynchronize(e1) == hipSuccess, "synchronize");
    check(hipStreamSynchronize(s) == hipSuccess, "stream synchronize");
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    hipStreamDestroy(s);
  }

  // Behind work the stream has not finished, a record waits for it.
  {
    hipStream_t s = nullptr;
    hipEvent_t e = nullptr;
    std::atomic<bool> release{false};
    hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    hipEventCreate(&e);
    check(hipLaunchHostFunc(s, hold, &release) == hipSuccess, "host function");
    check(hipEventRecord(e, s) == hipSuccess, "record behind work");
    check(hipEventQuery(e) == hipErrorNotReady, "a record behind running work is not complete");
    release = true;
    check(hipEventSynchronize(e) == hipSuccess, "the record completes after the work");
    check(hipEventQuery(e) == hipSuccess, "complete");
    hipEventDestroy(e);
    hipStreamDestroy(s);
  }

  // On a blocking stream a record also follows what the legacy null stream was given.
  {
    hipStream_t s = nullptr;
    hipEvent_t e = nullptr;
    std::atomic<bool> release{false};
    hipStreamCreate(&s);
    hipEventCreate(&e);
    check(hipLaunchHostFunc(nullptr, hold, &release) == hipSuccess, "host function on the null stream");
    check(hipEventRecord(e, s) == hipSuccess, "record behind the null stream");
    check(hipEventQuery(e) == hipErrorNotReady, "a record on a blocking stream waits for the null stream's work");
    release = true;
    check(hipEventSynchronize(e) == hipSuccess, "complete after the null stream's work");
    hipEventDestroy(e);
    hipStreamDestroy(s);
  }

  // With the host refusing threads, a record on an idle stream still works: it needs none. (Where this process may
  // start threads past the limit, the above is all that can be said.)
  rlimit saved{};
  getrlimit(RLIMIT_NPROC, &saved);
  rlimit low = saved;
  low.rlim_cur = 1;
  bool effective = false;
  hipStream_t fresh = nullptr;
  hipEvent_t a = nullptr, b = nullptr;
  hipStreamCreateWithFlags(&fresh, hipStreamNonBlocking);
  hipEventCreate(&a);
  hipEventCreate(&b);
  if (setrlimit(RLIMIT_NPROC, &low) == 0) {
    try {
      std::thread([] {}).join();
    } catch (const std::system_error&) {
      effective = true;
    }
  }
  if (effective) {
    const hipError_t r0 = hipEventRecord(a, fresh), r1 = hipEventRecord(b, fresh);
    setrlimit(RLIMIT_NPROC, &saved);
    check(r0 == hipSuccess && r1 == hipSuccess, "records on an idle stream need no thread");
    check(hipEventQuery(b) == hipSuccess, "complete without a thread");
  } else {
    setrlimit(RLIMIT_NPROC, &saved);
  }
  hipEventDestroy(a);
  hipEventDestroy(b);
  hipStreamDestroy(fresh);

  if (failures) return 1;
  std::printf("ok events recorded on an idle stream are taken on the caller%s\n", effective ? "" : " (threads could not be refused here)");
  return 0;
}
