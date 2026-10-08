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
thread_local int t_api_depth = 0;              // public calls this thread is inside
thread_local int t_silence = 0;                // Silence scopes this thread is inside
thread_local int t_sync_depth = 0;             // waits this thread is inside
thread_local const void* t_args[16];           // the next call's arguments (note_args)
thread_local int t_nargs = 0;
thread_local const char* t_symbol = nullptr;

std::mutex g_hook_mu;
Hooks g_hooks;
std::atomic<bool> g_hooked{false};

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

void set_hooks(const Hooks& h) {
  std::lock_guard<std::mutex> lock(g_hook_mu);
  g_hooks = h;
  g_hooked.store(h.api || h.resource || h.sync, std::memory_order_release);
}
bool hooked() { return g_hooked.load(std::memory_order_acquire); }

void notify_resource(Resource what, uint64_t handle, uint32_t device) {
  if (!hooked()) return;
  Hooks h;
  {
    std::lock_guard<std::mutex> lock(g_hook_mu);
    h = g_hooks;
  }
  if (h.resource) h.resource(what, handle, device);
}
void notify_sync(SyncKind what, uint64_t stream) {
  if (!hooked()) return;
  Hooks h;
  {
    std::lock_guard<std::mutex> lock(g_hook_mu);
    h = g_hooks;
  }
  if (h.sync) h.sync(what, stream);
}

void note_args(const void* const* args, int n) {
  if (!enabled() && !hooked()) return;
  t_nargs = n > 16 ? 16 : n;
  for (int i = 0; i < t_nargs; ++i) t_args[i] = args[i];
}
void note_symbol(const char* name) {
  if (enabled() || hooked()) t_symbol = name;
}

Silence::Silence() {
  ++t_api_depth;
  ++t_silence;
}
Silence::~Silence() {
  --t_api_depth;
  --t_silence;
}
bool silenced() { return t_silence > 0; }

void notify_init_finished() {
  static std::atomic<bool> told{false};
  if (!enabled() && !hooked()) return;
  if (!told.exchange(true)) notify_resource(Resource::CuInitFinished, 0, 0);
}

SyncScope::SyncScope(SyncKind kind, uint64_t stream, uint64_t event, uint32_t device)
    : kind_(kind), stream_(stream), event_(event), device_(device), outermost_(t_sync_depth++ == 0) {
  if (outermost_) t0_ = now_ns();
}
SyncScope::~SyncScope() {
  --t_sync_depth;
  if (!outermost_) return;
  if (enabled()) {
    Event ev;
    ev.kind = EventKind::Sync;
    ev.sync_kind = static_cast<uint8_t>(kind_);
    ev.start_ns = t0_;
    ev.end_ns = now_ns();
    ev.device = device_;
    ev.correlation = work_correlation();
    ev.stream = stream_;
    ev.handle = event_;
    record(std::move(ev));
  }
  if (kind_ == SyncKind::Stream || kind_ == SyncKind::Context) notify_sync(kind_, stream_);
}

ApiCall::ApiCall(const char* name, Domain domain) {
  // The arguments noted for this call are this call's, whether or not it is
  // the one that reports them.
  const int nargs = t_nargs;
  const char* symbol = t_symbol;
  t_nargs = 0;
  t_symbol = nullptr;
  outermost_ = t_api_depth++ == 0;
  if (!outermost_) return;
  hooked_ = hooked();
  if (!enabled() && !hooked_) return;
  name_ = name;
  domain_ = domain;
  correlation_ = next_correlation();
  outer_ = t_api_correlation;
  t_api_correlation = correlation_;
  nargs_ = nargs;
  // The arguments live in the caller's frame; keep the pointers only for this
  // call, in storage this object owns.
  for (int i = 0; i < nargs; ++i) saved_[i] = t_args[i];
  args_ = nargs ? saved_ : nullptr;
  symbol_ = symbol;
  start_ = now_ns();
  if (hooked_) {
    Hooks h;
    {
      std::lock_guard<std::mutex> lock(g_hook_mu);
      h = g_hooks;
    }
    if (h.api) {
      ApiInfo info;
      info.domain = domain_;
      info.enter = true;
      info.name = name_;
      info.correlation = correlation_;
      info.args = args_;
      info.nargs = nargs_;
      info.symbol = symbol_;
      h.api(info);
    }
  }
}

ApiCall::~ApiCall() {
  --t_api_depth;
  if (!name_) return;
  t_api_correlation = outer_;
  const uint64_t end = now_ns();
  if (hooked_) {
    Hooks h;
    {
      std::lock_guard<std::mutex> lock(g_hook_mu);
      h = g_hooks;
    }
    if (h.api) {
      ApiInfo info;
      info.domain = domain_;
      info.enter = false;
      info.name = name_;
      info.correlation = correlation_;
      info.args = args_;
      info.nargs = nargs_;
      info.symbol = symbol_;
      info.result = result_;
      h.api(info);
    }
  }
  if (!enabled()) return;
  Event e;
  e.kind = EventKind::Api;
  e.start_ns = start_;
  e.end_ns = end;
  e.correlation = correlation_;
  e.name = name_;
  e.process_id = static_cast<uint32_t>(::getpid());
  e.thread_id = static_cast<uint32_t>(::syscall(SYS_gettid));
  e.result = result_;
  e.domain_driver = domain_ == Domain::Driver;
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
