#include "vgpu/profiling.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <chrono>
#include <mutex>

#include <sys/syscall.h>
#include <unistd.h>

namespace vgpu::profiling {
namespace {

std::atomic<bool> g_on{false};
std::mutex g_mu;
std::vector<Event> g_events;
std::atomic<uint32_t> g_correlation{1};
thread_local uint32_t t_api_correlation = 0;   // the API call this thread is inside

// A profiler that never drains must not grow the process without bound. The
// oldest events go first, which is the right end to lose: a timeline is read
// from the recent end, and dropping silently would be worse than dropping
// visibly, so the count is kept.
constexpr size_t kMaxBuffered = 1u << 20;

}  // namespace

bool enabled() { return g_on.load(std::memory_order_relaxed); }
void set_enabled(bool on) { g_on.store(on, std::memory_order_relaxed); }

void record(Event&& e) {
  if (!enabled()) return;
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_events.size() >= kMaxBuffered) g_events.erase(g_events.begin());
  g_events.push_back(std::move(e));
}

std::vector<Event> drain() {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<Event> out;
  out.swap(g_events);
  return out;
}

uint64_t now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

uint32_t next_correlation() { return g_correlation.fetch_add(1, std::memory_order_relaxed); }

uint32_t work_correlation() { return t_api_correlation ? t_api_correlation : next_correlation(); }

ApiCall::ApiCall(const char* name) {
  if (!enabled()) return;
  name_ = name;
  correlation_ = next_correlation();
  outer_ = t_api_correlation;
  t_api_correlation = correlation_;
  start_ = now_ns();
}

ApiCall::~ApiCall() {
  if (!name_) return;
  t_api_correlation = outer_;
  Event e;
  e.kind = EventKind::Api;
  e.start_ns = start_;
  e.end_ns = now_ns();
  e.correlation = correlation_;
  e.name = name_;
  e.process_id = static_cast<uint32_t>(::getpid());
  e.thread_id = static_cast<uint32_t>(::syscall(SYS_gettid));
  e.result = result_;
  record(std::move(e));
}

}  // namespace vgpu::profiling

namespace vgpu {

void load_injection_library() {
  static std::once_flag once;
  std::call_once(once, [] {
    const char* path = std::getenv("CUDA_INJECTION64_PATH");
    if (!path || !path[0] || std::strcmp(path, "none") == 0) return;
    const bool quiet = [] {
      const char* q = std::getenv("VGPU_QUIET");
      return q && q[0] == '1';
    }();
    void* handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      if (!quiet)
        std::fprintf(stderr,
                     "[vgpu] CUDA_INJECTION64_PATH is '%s' but it could not be loaded: %s\n",
                     path, dlerror());
      return;
    }
    using InitFn = int (*)(void);
    auto init = reinterpret_cast<InitFn>(dlsym(handle, "InitializeInjection"));
    if (!init) {
      if (!quiet)
        std::fprintf(stderr, "[vgpu] '%s' has no InitializeInjection entry point\n", path);
      return;
    }
    const int ok = init();
    if (!quiet)
      std::fprintf(stderr, "[vgpu] profiler injection library '%s' initialized (returned %d)\n",
                   path, ok);
  });
}

}  // namespace vgpu
