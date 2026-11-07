// Stream capture, as the driver API sees it (included into driver_api.cpp's anonymous namespace).
//
// The driver API has no capture engine of its own. Capture is the runtime's (cudaStreamBeginCapture
// and the graph it builds); the two libraries share one simulated machine, and a driver stream is a
// handle the runtime's capture state accepts as a key. So, as cuLaunchKernel does, a driver call on
// a capturing stream asks the runtime by name:
//   - a stream-ordered copy, fill or host function becomes a node of that kind whose launch is the
//     call, repeated (vgpu_record_driver_node_v1; the node does not give its parameters back through
//     cuGraph*NodeGetParams);
//   - an event record or wait goes through a runtime event that stands for the driver's (the driver
//     keeps its own events), so a fork and a join across streams are the runtime's;
//   - a call CUDA does not allow while capturing (allocation, synchronization, ...) is refused with
//     CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED and invalidates the capture, as it does on the card.
// All of these are measured on an RTX 3060 (nvidia/tests/e2e/graph_capture_driver.cu).

constexpr CUresult kCaptureUnsupported = static_cast<CUresult>(900);   // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED

// A runtime export (libcudart, or libvgpucudart), found by name: in the global scope if the program
// loaded the runtime, else by loading the one that sits beside this library. Null if there is none,
// in which case nothing can be capturing.
void* runtime_symbol(const char* name) {
  if (void* p = dlsym(RTLD_DEFAULT, name)) return p;
  static void* lib = [] {
    Dl_info here{};
    std::string dir;
    if (dladdr(reinterpret_cast<void*>(&runtime_symbol), &here) && here.dli_fname) {
      dir = here.dli_fname;
      dir = dir.substr(0, dir.find_last_of('/') + 1);
    }
    for (const char* so : {"libcudart.so.13", "libcudart.so.12", "libcudart.so"}) {
      if (!dir.empty())
        if (void* h = dlopen((dir + so).c_str(), RTLD_NOW | RTLD_GLOBAL)) return h;
      if (void* h = dlopen(so, RTLD_NOW | RTLD_GLOBAL)) return h;
    }
    return static_cast<void*>(nullptr);
  }();
  return lib ? dlsym(lib, name) : nullptr;
}

template <class Fn>
Fn runtime_fn(const char* name) {
  return reinterpret_cast<Fn>(runtime_symbol(name));
}

// Whether the stream is capturing right now (cudaStreamCaptureStatusActive).
bool stream_capturing(CUstream st) {
  using Fn = int (*)(void*, int*);
  static const Fn fn = runtime_fn<Fn>("cudaStreamIsCapturing");
  int status = 0;
  return st && fn && fn(st, &status) == 0 && status == 1;
}

// Records `op` as a node of `type` (cudaGraphNodeType) when the stream is capturing.
bool capture_node(CUstream st, int type, const std::function<void()>& op) {
  if (!stream_capturing(st)) return false;
  using Fn = bool (*)(void*, int, const std::function<void()>*);
  static const Fn fn = runtime_fn<Fn>("vgpu_record_driver_node_v1");
  return fn && fn(st, type, &op);
}

// For a call that is not allowed on a capturing stream: invalidates the capture and says so.
bool capture_forbidden(CUstream st, const char* what) {
  using Fn = int (*)(void*, const char*);
  static const Fn fn = runtime_fn<Fn>("vgpu_capture_refuse_v1");
  return st && fn && stream_capturing(st) && fn(st, what) != 0;
}

// The "potentially unsafe" calls (allocation, freeing, host registration, pool and graph management,
// ...): refused while the calling thread's capture mode forbids them, which also invalidates the
// capture. True if refused.
bool capture_unsafe(const char* api) {
  using Fn = int (*)(const char*);
  static const Fn fn = runtime_fn<Fn>("vgpu_capture_unsafe_gate_v1");
  return fn && fn(api) != 0;
}

// cuCtxSynchronize and the like: refused if any capture is going on.
bool capture_forbidden_anywhere(const char* api) {
  using Fn = int (*)(const char*);
  static const Fn fn = runtime_fn<Fn>("vgpu_capture_refuse_any_v1");
  return fn && fn(api) != 0;
}

#define VGPU_CAPTURE_NODE(stream, node_type, ...)                                          \
  do {                                                                                     \
    if (capture_node((stream), (node_type), [=]() { (void)(__VA_ARGS__); })) return CUDA_SUCCESS; \
  } while (0)

#define VGPU_CAPTURE_UNSAFE(api)                                           \
  do {                                                                     \
    if (capture_unsafe(api)) return kCaptureUnsupported;                   \
  } while (0)

// cudaGraphNodeType values the driver's captures use.
constexpr int kNodeKernel = 0, kNodeMemcpy = 1, kNodeMemset = 2, kNodeHost = 3;

// The runtime event that stands for a driver event in a capture (events the runtime made are their
// own). Made on first use; a driver event never used in a capture has none.
void* shadow_event(CUevent ev, bool create) {
  ShimState& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  auto it = s.events.find(reinterpret_cast<uintptr_t>(ev));
  if (it == s.events.end()) return reinterpret_cast<void*>(ev);   // the runtime's own
  if (!it->second.rt_event && create) {
    using Fn = int (*)(void**, unsigned);
    static const Fn make = runtime_fn<Fn>("cudaEventCreateWithFlags");
    void* e = nullptr;
    if (make && make(&e, 0x2 /* cudaEventDisableTiming */) == 0) it->second.rt_event = e;
  }
  return it->second.rt_event;
}

// ---- Host functions ----------------------------------------------------------------------------------------------------
// cuLaunchHostFunc queues the function on the stream and returns, as the card does: the function runs on a worker thread
// after the stream's earlier work, and what the stream is given afterwards waits for it. The stream lanes that do this are
// the runtime's (runtime_api.cpp, "Stream lanes": the two libraries share one machine and one set of stream handles), and
// libcuda reaches them by name (vgpu_async_*_v1), as it does capture. Every hook acts only once a host function has been
// queued by either API, so a program that never queues one is not changed by any of this: the checks below are one call
// that reads an atomic. What needs the lanes:
//   - the work a stream is given (kernels, copies, fills, memory operations, callbacks) first waits for the stream's
//     earlier host functions (StreamOrder, before the API lock is taken);
//   - cuStreamQuery is CUDA_ERROR_NOT_READY, and cuStreamSynchronize and cuCtxSynchronize wait, while they run;
//   - a driver event recorded from then on is also recorded on a runtime event (the one that stands for it in a capture),
//     which stays pending until the lane reaches it: cuEventQuery is NOT_READY, cuEventSynchronize waits,
//     cuEventElapsedTime is NOT_READY and then measures to the moment the lane got there, and cuStreamWaitEvent holds
//     the stream back.
// cuLaunchHostFunc loads the runtime library if the program has not (as the capture check beside it already does); every other
// call only uses one that is already there, since a program without it has no host function to wait for.
struct AsyncHooks {
  int (*used)();
  void (*host_fn)(void*, CUhostFn, void*);
  void (*order)(void*);
  int (*busy)(void*);
  void (*wait_stream)(void*);
  void (*wait_device)();
  void (*event_record)(void*, void*);
  int (*event_state)(void*, long long*);
  void (*event_wait)(void*);
  void (*stream_wait_event)(void*, void*);
  void (*stream_flags)(void*, int, int);
};
// The hooks, or null. With `load` false only a runtime already in the process is used: a program on the driver API alone
// (no libcudart) must not have one loaded by a copy or a launch (two copies of the simulator in a process are what a sanitizer
// build refuses), and one that never loads it has no host function to wait for -- cuLaunchHostFunc itself loads it. A miss is not
// cached for good (a program may load libcudart later), only for a second.
const AsyncHooks* async_hooks(bool load = false) {
  static std::atomic<const AsyncHooks*> found{nullptr};
  static std::atomic<long long> retry_ns{0};
  if (const AsyncHooks* h = found.load(std::memory_order_acquire)) return h;
  struct timespec now {};
  clock_gettime(CLOCK_MONOTONIC, &now);
  const long long t = now.tv_sec * 1000000000LL + now.tv_nsec;
  if (!load && t < retry_ns.load(std::memory_order_relaxed)) return nullptr;
  static std::mutex probe_mu;
  std::lock_guard<std::mutex> lock(probe_mu);
  if (const AsyncHooks* h = found.load(std::memory_order_acquire)) return h;
  auto get = [&](const char* name) { return load ? runtime_symbol(name) : dlsym(RTLD_DEFAULT, name); };
  const AsyncHooks h{reinterpret_cast<int (*)()>(get("vgpu_async_host_used_v1")),
                     reinterpret_cast<void (*)(void*, CUhostFn, void*)>(get("vgpu_async_host_fn_v1")),
                     reinterpret_cast<void (*)(void*)>(get("vgpu_async_order_v1")),
                     reinterpret_cast<int (*)(void*)>(get("vgpu_async_stream_busy_v1")),
                     reinterpret_cast<void (*)(void*)>(get("vgpu_async_wait_stream_v1")),
                     reinterpret_cast<void (*)()>(get("vgpu_async_wait_device_v1")),
                     reinterpret_cast<void (*)(void*, void*)>(get("vgpu_async_event_record_v1")),
                     reinterpret_cast<int (*)(void*, long long*)>(get("vgpu_async_event_state_v1")),
                     reinterpret_cast<void (*)(void*)>(get("vgpu_async_wait_event_v1")),
                     reinterpret_cast<void (*)(void*, void*)>(get("vgpu_async_stream_wait_event_v1")),
                     reinterpret_cast<void (*)(void*, int, int)>(get("vgpu_async_stream_flags_v1"))};
  const bool all = h.used && h.host_fn && h.order && h.busy && h.wait_stream && h.wait_device && h.event_record &&
                   h.event_state && h.event_wait && h.stream_wait_event && h.stream_flags;
  if (!all) {
    retry_ns.store(t + 1000000000LL, std::memory_order_relaxed);
    return nullptr;
  }
  static AsyncHooks stored;
  stored = h;
  found.store(&stored, std::memory_order_release);
  return &stored;
}
// Whether any host function has been queued (the runtime keeps the flag), so the lanes matter.
bool host_async() {
  const AsyncHooks* h = async_hooks();
  return h && h->used();
}
void async_order(CUstream st) {
  if (const AsyncHooks* h = async_hooks()) h->order(st);
}
void async_wait_stream(CUstream st) {
  if (const AsyncHooks* h = async_hooks()) h->wait_stream(st);
}
void async_wait_device() {
  if (const AsyncHooks* h = async_hooks()) h->wait_device();
}
bool async_stream_busy(CUstream st) {
  const AsyncHooks* h = async_hooks();
  return h && h->busy(st);
}
// The runtime event standing for `ev`, if it has one, and whether it is a driver event at all.
void* async_shadow(void* ev) {
  void* shadow = shadow_event(static_cast<CUevent>(ev), false);
  return shadow && shadow != ev ? shadow : nullptr;
}
