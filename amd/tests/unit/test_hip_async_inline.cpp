// A small asynchronous copy or fill on a stream with nothing in it is done on the calling thread (in_order in
// hip_api.cpp), not handed to the stream's thread: the stream's thread is not started for it, and the work is done
// when the call returns. Behind work the stream has not finished, and for a copy past the size limit, the work is
// queued as before; a kernel or a host function always is.
#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <system_error>
#include <thread>
#include <vector>

#include "vgpu_hip.h"

extern "C" hipError_t hipLaunchHostFunc(hipStream_t stream, void (*fn)(void*), void* data);
extern "C" hipError_t hipStreamQuery(hipStream_t stream);

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
bool all(const std::vector<unsigned char>& v, unsigned char b) {
  for (unsigned char c : v)
    if (c != b) return false;
  return true;
}
}  // namespace

int main() {
  constexpr size_t kSmall = 4096, kBig = 1 << 20;   // (the size limit is between: 256 KiB)
  void *d = nullptr, *big = nullptr;
  check(hipMalloc(&d, kSmall) == hipSuccess && hipMalloc(&big, kBig) == hipSuccess, "allocate");

  // Behind work the stream has not finished, a fill waits for it (a stream that does not order against the null
  // stream, so that the check below can read the memory meanwhile).
  {
    hipStream_t s = nullptr;
    hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    std::atomic<bool> release{false};
    check(hipMemset(d, 0xAA, kSmall) == hipSuccess, "fill");
    check(hipLaunchHostFunc(s, hold, &release) == hipSuccess, "host function");
    check(hipMemsetAsync(d, 0x55, kSmall, s) == hipSuccess, "async fill behind it");
    std::vector<unsigned char> seen(kSmall);
    check(hipMemcpy(seen.data(), d, kSmall, hipMemcpyDeviceToHost) == hipSuccess, "read");
    check(all(seen, 0xAA), "a fill queued behind running work has not been done");
    release = true;
    check(hipStreamSynchronize(s) == hipSuccess, "synchronize");
    check(hipMemcpy(seen.data(), d, kSmall, hipMemcpyDeviceToHost) == hipSuccess, "read again");
    check(all(seen, 0x55), "the fill is done after the work ahead of it");
    hipStreamDestroy(s);
  }

  // On a stream with nothing in it the work is done by the time the call returns.
  {
    hipStream_t s = nullptr;
    hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    check(hipMemsetAsync(d, 0x33, kSmall, s) == hipSuccess, "async fill");
    check(hipStreamQuery(s) == hipSuccess, "the stream is idle when a small async fill returns");
    std::vector<unsigned char> seen(kSmall, 0);
    check(hipMemcpyAsync(seen.data(), d, kSmall, hipMemcpyDeviceToHost, s) == hipSuccess, "async copy");
    check(all(seen, 0x33), "the copy is done when the call returns");
    check(hipStreamSynchronize(s) == hipSuccess, "synchronize");
    hipStreamDestroy(s);
  }

  // With the host refusing threads: a small copy or fill needs none, a large one and a host function do.
  rlimit saved{};
  getrlimit(RLIMIT_NPROC, &saved);
  rlimit low = saved;
  low.rlim_cur = 1;
  hipStream_t fresh = nullptr, fresh2 = nullptr;
  hipStreamCreateWithFlags(&fresh, hipStreamNonBlocking);
  hipStreamCreateWithFlags(&fresh2, hipStreamNonBlocking);
  bool effective = false;
  if (setrlimit(RLIMIT_NPROC, &low) == 0) {
    try {
      std::thread([] {}).join();
    } catch (const std::system_error&) {
      effective = true;
    }
  }
  if (effective) {
    std::vector<unsigned char> host(kSmall, 0);
    const hipError_t fill = hipMemsetAsync(d, 0x77, kSmall, fresh);
    const hipError_t copy = hipMemcpyAsync(host.data(), d, kSmall, hipMemcpyDeviceToHost, fresh);
    const hipError_t large = hipMemsetAsync(big, 1, kBig, fresh2);
    setrlimit(RLIMIT_NPROC, &saved);
    check(fill == hipSuccess && copy == hipSuccess, "a small fill and copy need no thread");
    check(all(host, 0x77), "and are done");
    check(large != hipSuccess, "a fill past the size limit is queued, and the host would not start the thread");
  } else {
    setrlimit(RLIMIT_NPROC, &saved);
  }
  hipStreamDestroy(fresh);
  hipStreamDestroy(fresh2);
  hipFree(d);
  hipFree(big);

  if (failures) return 1;
  std::printf("ok small async copies and fills on an idle stream are done on the caller%s\n",
              effective ? "" : " (threads could not be refused here)");
  return 0;
}
