#include "vgpu/profiling.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <chrono>
#include <mutex>
#include <unordered_map>

#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace vgpu::profiling {
namespace {

std::mutex g_mu;
std::vector<Event> g_events;
std::atomic<uint32_t> g_correlation{1};
// What every API call of a thread reads and writes, in one thread-local: a thread-local of a shim library is
// reached through __tls_get_addr, and a call that touched several paid for each (the calls are made by the
// million). `t` below is that one lookup, made once in a function.
struct ThreadApi {
  int depth = 0;                  // public calls this thread is inside
  int silence = 0;                // Silence scopes this thread is inside
  int sync_depth = 0;             // waits this thread is inside
  int nargs = 0;                  // the next call's arguments (note_args)
  uint32_t correlation = 0;       // the API call this thread is inside
  bool have_sizes = false;
  const char* symbol = nullptr;
  const void* args[16];
  uint16_t sizes[16];
};
thread_local ThreadApi t_api;

thread_local std::vector<uint64_t> t_external[kExternalKinds];
thread_local uint32_t t_graph_id = 0;          // the graph launch this thread is running (GraphWork)
thread_local uint32_t t_graph_node = 0;

std::mutex g_hook_mu;
Hooks g_hooks;

// A profiler that never drains must not grow the process without bound. The
// oldest events go first, which is the right end to lose: a timeline is read
// from the recent end, and dropping silently would be worse than dropping
// visibly, so the count is kept.
constexpr size_t kMaxBuffered = 1u << 20;

}  // namespace

std::atomic<bool> detail::g_on{false};
std::atomic<bool> detail::g_hooked{false};
void set_enabled(bool on) { detail::g_on.store(on, std::memory_order_relaxed); }

namespace {
std::mutex g_graph_mu;
std::atomic<uint32_t> g_graph_ids{1};
std::unordered_map<uint64_t, uint32_t> g_graph_by_handle;
std::unordered_map<uint64_t, uint64_t> g_node_by_handle;
}  // namespace

uint32_t next_graph_id() { return g_graph_ids.fetch_add(1, std::memory_order_relaxed); }
namespace {
std::mutex g_module_mu;
std::unordered_map<uint32_t, uint32_t> g_module_ids;   // device -> the next number
std::atomic<uint32_t> g_function_ids{1};
}
uint32_t next_module_id(uint32_t device) {
  std::lock_guard<std::mutex> lock(g_module_mu);
  auto it = g_module_ids.find(device);
  if (it == g_module_ids.end()) it = g_module_ids.emplace(device, 22u).first;
  return it->second++;
}
void reset_module_ids(uint32_t device) {
  std::lock_guard<std::mutex> lock(g_module_mu);
  g_module_ids.erase(device);
}
uint32_t next_function_id() { return g_function_ids.fetch_add(1, std::memory_order_relaxed); }
void register_graph(uint64_t handle, uint32_t id) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  g_graph_by_handle[handle] = id;
}
void register_graph_node(uint64_t handle, uint64_t id) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  g_node_by_handle[handle] = id;
}
void forget_graph_object(uint64_t handle) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  g_graph_by_handle.erase(handle);
  g_node_by_handle.erase(handle);
}
bool graph_id_of(uint64_t handle, uint32_t* id) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  const auto it = g_graph_by_handle.find(handle);
  if (it == g_graph_by_handle.end()) return false;
  *id = it->second;
  return true;
}
bool graph_node_id_of(uint64_t handle, uint64_t* id) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  const auto it = g_node_by_handle.find(handle);
  if (it == g_node_by_handle.end()) return false;
  *id = it->second;
  return true;
}

GraphWork::GraphWork(uint32_t graph_id, uint32_t node_index)
    : saved_graph_(t_graph_id), saved_node_(t_graph_node) {
  t_graph_id = graph_id;
  t_graph_node = node_index;
}
GraphWork::~GraphWork() {
  t_graph_id = saved_graph_;
  t_graph_node = saved_node_;
}

namespace {
std::atomic<void (*)(Event&)> g_record_hook{nullptr};
}
void set_record_hook(void (*fn)(Event&)) { g_record_hook.store(fn, std::memory_order_release); }

void record(Event&& e) {
  if (!enabled()) return;
  if (t_api.silence > 0) return;   // not the program's: see Silence
  if (t_graph_id && (e.kind == EventKind::Kernel || e.kind == EventKind::Memcpy ||
                     e.kind == EventKind::Memset || e.kind == EventKind::Memcpy2)) {
    e.graph_id = t_graph_id;
    e.graph_node_id = (uint64_t{t_graph_id} << 32) | t_graph_node;
  }
  if (const auto hook = g_record_hook.load(std::memory_order_acquire)) hook(e);
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

uint32_t work_correlation() { return t_api.correlation ? t_api.correlation : next_correlation(); }

bool push_external(int kind, uint64_t id) {
  if (kind < 0 || kind >= kExternalKinds) return false;
  t_external[kind].push_back(id);
  return true;
}
bool pop_external(int kind, uint64_t* last) {
  if (kind < 0 || kind >= kExternalKinds || t_external[kind].empty()) return false;
  if (last) *last = t_external[kind].back();
  t_external[kind].pop_back();
  return true;
}

void set_hooks(const Hooks& h) {
  std::lock_guard<std::mutex> lock(g_hook_mu);
  g_hooks = h;
  detail::g_hooked.store(h.api || h.resource || h.sync, std::memory_order_release);
}

void notify_resource(const ResourceInfo& info) {
  if (!hooked()) return;
  Hooks h;
  {
    std::lock_guard<std::mutex> lock(g_hook_mu);
    h = g_hooks;
  }
  if (h.resource) h.resource(info);
}
void notify_resource(Resource what, uint64_t handle, uint32_t device) {
  ResourceInfo info;
  info.what = what;
  info.handle = handle;
  info.device = device;
  notify_resource(info);
}

namespace {
std::atomic<uint64_t (*)()> g_host_clock{nullptr};
}
void set_host_clock(uint64_t (*fn)()) { g_host_clock.store(fn, std::memory_order_release); }
namespace {
std::atomic<bool> g_thread_id_system{false};
}
void set_thread_id_system(bool system) { g_thread_id_system.store(system, std::memory_order_relaxed); }
bool thread_id_system() { return g_thread_id_system.load(std::memory_order_relaxed); }
uint32_t thread_id() {
  // cupti_activity.h, CUpti_ActivityThreadIdType: DEFAULT is pthread_self() on
  // Linux, SYSTEM is syscall SYS_gettid. The record field is 32 bits wide.
  if (thread_id_system()) return static_cast<uint32_t>(::syscall(SYS_gettid));
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(::pthread_self()));
}

uint64_t host_ns() {
  const auto fn = g_host_clock.load(std::memory_order_acquire);
  return fn ? fn() : now_ns();
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

void note_args(const void* const* args, int n, const uint16_t* sizes) {
  if (!enabled() && !hooked()) return;
  ThreadApi& t = t_api;
  t.nargs = n > 16 ? 16 : n;
  for (int i = 0; i < t.nargs; ++i) t.args[i] = args[i];
  t.have_sizes = sizes != nullptr;
  if (sizes)
    for (int i = 0; i < t.nargs; ++i) t.sizes[i] = sizes[i];
}
void note_symbol(const char* name) {
  if (enabled() || hooked()) t_api.symbol = name;
}

Silence::Silence() {
  ThreadApi& t = t_api;
  ++t.depth;
  ++t.silence;
}
Silence::~Silence() {
  ThreadApi& t = t_api;
  --t.depth;
  --t.silence;
}
bool silenced() { return t_api.silence > 0; }

namespace {
std::atomic<bool> g_context_made{false};
std::atomic<void (*)()> g_context_hook{nullptr};
}  // namespace
void set_context_hook(void (*fn)()) { g_context_hook.store(fn, std::memory_order_release); }
bool context_made() { return g_context_made.load(std::memory_order_acquire); }
void note_context_made() {
  if (g_context_made.load(std::memory_order_acquire)) return;   // (every API call asks; the exchange is a locked instruction)
  if (g_context_made.exchange(true, std::memory_order_acq_rel)) return;
  if (const auto fn = g_context_hook.load(std::memory_order_acquire)) fn();
}

void notify_init_finished() {
  static std::atomic<bool> told{false};
  if (!enabled() && !hooked()) return;
  if (!told.exchange(true)) notify_resource(Resource::CuInitFinished, 0, 0);
}

SyncScope::SyncScope(SyncKind kind, uint64_t stream, uint64_t event, uint32_t device)
    : kind_(kind), stream_(stream), event_(event), device_(device), outermost_(t_api.sync_depth++ == 0) {
  // The start is only a record's; nobody recording means nobody reads it (the destructor reads it only when
  // recording is on at the end, and a record that was switched on during the wait has no start to give).
  if (outermost_ && enabled()) t0_ = host_ns();
}
SyncScope::~SyncScope() {
  --t_api.sync_depth;
  if (!outermost_) return;
  if (enabled()) {
    Event ev;
    ev.kind = EventKind::Sync;
    ev.sync_kind = static_cast<uint8_t>(kind_);
    ev.end_ns = host_ns();
    ev.start_ns = t0_ ? t0_ : ev.end_ns;   // recording came on during the wait: no start was taken
    ev.device = device_;
    ev.correlation = work_correlation();
    ev.stream = stream_;
    ev.handle = event_;
    record(std::move(ev));
  }
  if (kind_ == SyncKind::Stream || kind_ == SyncKind::Context) notify_sync(kind_, stream_);
}

ApiCall::ApiCall(const char* name, Domain domain) {
  ThreadApi& t = t_api;
  // The arguments noted for this call are this call's, whether or not it is
  // the one that reports them. Nothing is noted unless somebody listens.
  int nargs = 0;
  const char* symbol = nullptr;
  bool have_sizes = false;
  uint16_t sizes[16];
  if (__builtin_expect(t.nargs != 0 || t.have_sizes || t.symbol != nullptr, 0)) {
    nargs = t.nargs;
    symbol = t.symbol;
    have_sizes = t.have_sizes;
    if (have_sizes)
      for (int i = 0; i < nargs; ++i) sizes[i] = t.sizes[i];
    t.have_sizes = false;
    t.nargs = 0;
    t.symbol = nullptr;
  }
  outermost_ = t.depth++ == 0;
  if (!outermost_) return;
  hooked_ = hooked();
  if (!enabled() && !hooked_) return;
  name_ = name;
  domain_ = domain;
  correlation_ = next_correlation();
  outer_ = t.correlation;
  t.correlation = correlation_;
  nargs_ = nargs;
  // The arguments live in the caller's frame; keep the pointers only for this
  // call, in storage this object owns.
  for (int i = 0; i < nargs; ++i) saved_[i] = t.args[i];
  args_ = nargs ? saved_ : nullptr;
  if (have_sizes && nargs) {
    for (int i = 0; i < nargs; ++i) saved_sizes_[i] = sizes[i];
    sizes_ = saved_sizes_;
  }
  symbol_ = symbol;
  start_ = host_ns();
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
      info.arg_sizes = sizes_;
      info.symbol = symbol_;
      h.api(info);
    }
  }
}

ApiCall::~ApiCall() {
  ThreadApi& t = t_api;
  --t.depth;
  if (!name_) return;
  t.correlation = outer_;
  const uint64_t end = host_ns();
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
      info.arg_sizes = sizes_;
      info.symbol = symbol_;
      info.result = result_;
      info.return_value = return_value_;
      h.api(info);
    }
  }
  if (!enabled()) return;
  // The tags in force when the call was made, one per kind, ahead of the call.
  for (int k = 0; k < kExternalKinds; ++k) {
    if (t_external[k].empty()) continue;
    Event x;
    x.kind = EventKind::ExternalCorrelation;
    x.start_ns = x.end_ns = start_;
    x.correlation = correlation_;
    x.flags = static_cast<uint32_t>(k);
    x.handle = t_external[k].back();
    record(std::move(x));
  }
  Event e;
  e.kind = EventKind::Api;
  e.start_ns = start_;
  e.end_ns = end;
  e.correlation = correlation_;
  e.name = name_;
  e.process_id = static_cast<uint32_t>(::getpid());
  e.thread_id = thread_id();
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
