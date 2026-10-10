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
