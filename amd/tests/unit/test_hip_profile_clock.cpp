// A HIP call a profiler may time (hipMalloc, hipFree, hipMemcpy, hipMemset, hipHostMalloc) reads the clock for its start only when a profiler is
// attached (profile_clock in hip_api.cpp). Counted here by standing in for clock_gettime: the same calls, with and without a profiler, differ by
// the four reads of a malloc and a free (start and end of each) -- and a call nobody times reads none of its own. The profiler is told what it was
// told before: a start no later than the end, for every call it hears of.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <sys/syscall.h>
#include <unistd.h>

#include "vgpu/hip_profiler.hpp"
#include "vgpu_hip.h"

extern "C" int vgpu_hip_profiler_attach(const vgpu::amd::hipprof::Hooks* hooks);

namespace {
std::atomic<uint64_t> g_clock_reads{0};
int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++failures;
  }
}
std::atomic<int> g_allocs{0}, g_copies{0}, g_fills{0}, g_bad_order{0};
void allocated(int, uint64_t, uint64_t, bool, uint64_t start, uint64_t end) {
  ++g_allocs;
  if (start == 0 || start > end) ++g_bad_order;
}
void copied(vgpu::amd::hipprof::Copy, int, int, uint64_t, uint64_t, uint64_t, uint64_t start, uint64_t end) {
  ++g_copies;
  if (start == 0 || start > end) ++g_bad_order;
}
void filled(int, uint64_t, uint64_t, uint64_t start, uint64_t end) {
  ++g_fills;
  if (start == 0 || start > end) ++g_bad_order;
}

constexpr int kRounds = 40;
// What a round of calls reads from the clock.
uint64_t clock_reads_of_rounds() {
  const uint64_t before = g_clock_reads.load();
  for (int i = 0; i < kRounds; ++i) {
    void* p = nullptr;
    check(hipMalloc(&p, 4096) == hipSuccess && p, "hipMalloc");
    check(hipFree(p) == hipSuccess, "hipFree");
  }
  return g_clock_reads.load() - before;
}
}  // namespace

// The runtime's clock reads go through this (the executable's symbol comes before libc's).
extern "C" int clock_gettime(clockid_t id, struct timespec* ts) {
  g_clock_reads.fetch_add(1, std::memory_order_relaxed);
  return static_cast<int>(syscall(SYS_clock_gettime, id, ts));
}

int main() {
  void* warm = nullptr;
  hipMalloc(&warm, 64);   // (the runtime and the telemetry segment start)
  hipFree(warm);
  clock_reads_of_rounds();

  const uint64_t unwatched = clock_reads_of_rounds();

  vgpu::amd::hipprof::Hooks hooks;
  hooks.allocated = allocated;
  hooks.copied = copied;
  hooks.filled = filled;
  check(vgpu_hip_profiler_attach(&hooks) >= 0, "attach");
  const uint64_t watched = clock_reads_of_rounds();
  check(g_allocs.load() >= 2 * kRounds, "the profiler hears of every allocation and free");

  // A copy and a fill, timed for the profiler.
  void* d = nullptr;
  static char host[256];
  check(hipMalloc(&d, 256) == hipSuccess, "hipMalloc");
  check(hipMemset(d, 7, 256) == hipSuccess, "hipMemset");
  check(hipMemcpy(host, d, 256, hipMemcpyDeviceToHost) == hipSuccess && host[0] == 7, "hipMemcpy");
  check(hipFree(d) == hipSuccess, "hipFree");
  check(g_copies.load() >= 1 && g_fills.load() >= 1, "the profiler hears of a copy and a fill");
  check(g_bad_order.load() == 0, "every start is set and no later than its end");

  // Four reads for a malloc and free the profiler is told of; the same pair unwatched reads none beyond what the telemetry does.
  check(watched >= unwatched + 3 * kRounds, "a call nobody times does not read the clock for its start");
  std::printf("%s clock reads: unwatched %llu, watched %llu\n", failures ? "FAILED" : "ok", static_cast<unsigned long long>(unwatched),
              static_cast<unsigned long long>(watched));
  return failures ? 1 : 0;
}
