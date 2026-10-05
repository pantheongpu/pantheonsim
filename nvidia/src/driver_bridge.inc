// The driver side of the bridge to libcudart (vgpu_bridge.h): how libcuda finds
// the runtime, calls into it, and records into a capturing stream's graph.
// Included early in driver_api.cpp, ahead of the calls that must be recorded
// when their stream is capturing.

// libcuda's graph, stream-capture and user-object entry points. Each is a
// thin call into libcudart, which owns the graph machinery (vgpu_bridge.h):
// the driver and the runtime see one set of graphs, nodes and executable
// graphs, as NVIDIA's do. Included by driver_api.cpp.

namespace {

// libcudart, as this library finds it. The scope the program loaded it into
// first; failing that, one already loaded by name (a private load, which the
// global scope cannot see); and only when `load` is set, the one beside this
// library, which is how a program that links libcuda alone gets graphs. Every
// candidate must export vgpu_runtime_marker_v1, so NVIDIA's own libcudart is
// never mistaken for the simulator's.
struct RuntimeScope {
  bool ok = false;
  void* handle = nullptr;   // RTLD_DEFAULT is null, so `ok` says whether there is one
};
RuntimeScope runtime_scope(bool load) {
  static RuntimeScope found;
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  if (found.ok) return found;
  if (dlsym(RTLD_DEFAULT, "vgpu_runtime_marker_v1")) return found = RuntimeScope{true, RTLD_DEFAULT};
  for (const char* name : {"libcudart.so.13", "libcudart.so.12"}) {
    if (void* h = dlopen(name, RTLD_LAZY | RTLD_NOLOAD)) {
      if (dlsym(h, "vgpu_runtime_marker_v1")) return found = RuntimeScope{true, h};
      dlclose(h);
    }
  }
  if (!load) return RuntimeScope{};
  Dl_info self{};
  if (!dladdr(reinterpret_cast<void*>(&runtime_scope), &self) || !self.dli_fname) return RuntimeScope{};
  std::string dir = self.dli_fname;
  dir = dir.substr(0, dir.find_last_of('/') + 1);
  for (const char* name : {"libcudart.so.13", "libcudart.so.12"}) {
    void* h = dlopen((dir + name).c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!h) continue;
    if (dlsym(h, "vgpu_runtime_marker_v1")) return found = RuntimeScope{true, h};
    dlclose(h);
  }
  return RuntimeScope{};
}

void* runtime_symbol(const char* name, bool load) {
  const RuntimeScope scope = runtime_scope(load);
  return scope.ok ? dlsym(scope.handle, name) : nullptr;
}

// A call into the runtime with the exact signature the bridge header declares
// for it; CUDA_ERROR_NOT_SUPPORTED when there is no simulator runtime to ask.
template <class F> struct Bridge;
template <class R, class... A>
struct Bridge<R (*)(A...)> {
  static CUresult call(const char* symbol, A... a) {
    void* p = runtime_symbol(symbol, /*load=*/true);
    if (!p) return CUDA_ERROR_NOT_SUPPORTED;
    return static_cast<CUresult>(reinterpret_cast<R (*)(A...)>(p)(a...));
  }
};
#define VGPU_EXPAND(...) __VA_ARGS__

}  // namespace

VGPU_GRAPH_BRIDGE_LIST(VGPU_DECLARE_BRIDGE)

// ---- operations on a capturing stream ------------------------------------------
//
// Work sent to a capturing stream is recorded into the capture's graph rather
// than run. The runtime holds the capture; here each operation is added as the
// node the equivalent graph call makes -- after the nodes the stream last
// captured -- and the stream moves on to it, which is how a capture adds
// nodes by hand with cuStreamGetCaptureInfo and
// cuStreamUpdateCaptureDependencies.
//
// Every hook returns kNotCapturing when the stream is not capturing, and the
// caller carries on and runs the work; anything else is the call's result.

constexpr int kNotCapturing = -1;

// Whether `stream` is being captured: the machine counts captures in progress,
// so a program that captures nothing pays one atomic load.
bool stream_capturing(ShimState& s, CUstream stream, int* status = nullptr) {
  if (!s.rt || s.rt->captures_in_progress() == 0) return false;
  using Fn = int (*)(void*, int*);
  static Fn fn = nullptr;
  if (!fn) fn = reinterpret_cast<Fn>(runtime_symbol("vgpu_cuStreamIsCapturing", false));
  if (!fn) return false;
  int st = 0;
  fn(stream, &st);
  if (status) *status = st;
  return st != 0;
}

// Adds a node at the stream's position in its capture. `add` gets the graph,
// the dependencies and the count, and adds the node.
template <class Add>
int capture_op(ShimState& s, CUstream stream, Add&& add) {
  int status = 0;
  if (!stream_capturing(s, stream, &status)) return kNotCapturing;
  if (status == 2) return 901;   // CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
  unsigned long long id = 0;
  void* graph = nullptr;
  const void* const* deps = nullptr;
  size_t n = 0;
  CUresult r = Bridge<decltype(&vgpu_cuStreamGetCaptureInfo_v3)>::call("vgpu_cuStreamGetCaptureInfo_v3", stream, &status,
                                                                       &id, &graph, &deps, nullptr, &n);
  if (r != CUDA_SUCCESS) return r;
  void* node = nullptr;
  r = add(graph, const_cast<void* const*>(deps), n, &node);
  if (r != CUDA_SUCCESS) return r;
  return Bridge<decltype(&vgpu_cuStreamUpdateCaptureDependencies)>::call("vgpu_cuStreamUpdateCaptureDependencies", stream,
                                                                         &node, 1, 1u /* set */);
}

template <class Fn, class... A>
CUresult bridge_call(const char* symbol, A... a) {
  return Bridge<Fn>::call(symbol, a...);
}
#define VGPU_BRIDGE(fn, ...) Bridge<decltype(&vgpu_##fn)>::call("vgpu_" #fn, __VA_ARGS__)

// A kernel launch into the capture.
int capture_kernel(ShimState& s, CUstream stream, CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                   unsigned by, unsigned bz, unsigned shared, void** params, void** extra, bool cooperative = false) {
  return capture_op(s, stream, [&](void* g, void* const* deps, size_t n, void** node) {
    vcu::KernelNodeParamsV3 k{};
    k.func = f;
    k.gridDimX = gx; k.gridDimY = gy; k.gridDimZ = gz;
    k.blockDimX = bx; k.blockDimY = by; k.blockDimZ = bz;
    k.sharedMemBytes = shared;
    k.kernelParams = params;
    k.extra = extra;
    const CUresult r = VGPU_BRIDGE(cuGraphAddKernelNode_v2, node, g, deps, n, &k);
    if (r == CUDA_SUCCESS && cooperative) {   // the node keeps how it was launched
      vcu::LaunchAttrValue v{};
      const int one = 1;
      std::memcpy(v.bytes, &one, sizeof one);
      return VGPU_BRIDGE(cuGraphKernelNodeSetAttribute, *node, 2 /* cooperative */, &v);
    }
    return r;
  });
}

// A copy between plain memory, either side host or device: the unified memory
// type lets the node tell by the address.
int capture_copy(ShimState& s, CUstream stream, const void* dst, const void* src, size_t bytes) {
  return capture_op(s, stream, [&](void* g, void* const* deps, size_t n, void** node) {
    vcu::Memcpy3D c{};
    c.srcMemoryType = vcu::kMemUnified;
    c.srcDevice = reinterpret_cast<vcu::DevPtr>(src);
    c.dstMemoryType = vcu::kMemUnified;
    c.dstDevice = reinterpret_cast<vcu::DevPtr>(dst);
    c.WidthInBytes = bytes;
    c.Height = 1;
    c.Depth = 1;
    return VGPU_BRIDGE(cuGraphAddMemcpyNode, node, g, deps, n, &c, nullptr);
  });
}

static_assert(sizeof(CUDA_MEMCPY3D) == sizeof(vcu::Memcpy3D), "CUDA_MEMCPY3D and the graph ABI's copy differ");
int capture_copy3d(ShimState& s, CUstream stream, const CUDA_MEMCPY3D& p) {
  return capture_op(s, stream, [&](void* g, void* const* deps, size_t n, void** node) {
    return VGPU_BRIDGE(cuGraphAddMemcpyNode, node, g, deps, n, &p, nullptr);
  });
}

int capture_fill(ShimState& s, CUstream stream, CUdeviceptr dst, size_t pitch, unsigned value, unsigned elem, size_t width,
                 size_t height) {
  return capture_op(s, stream, [&](void* g, void* const* deps, size_t n, void** node) {
    vcu::MemsetNodeParams m{dst, pitch, value, elem, width, height};
    return VGPU_BRIDGE(cuGraphAddMemsetNode, node, g, deps, n, &m, nullptr);
  });
}

int capture_host_fn(ShimState& s, CUstream stream, void (*fn)(void*), void* user) {
  return capture_op(s, stream, [&](void* g, void* const* deps, size_t n, void** node) {
    vcu::HostNodeParams h{fn, user};
    return VGPU_BRIDGE(cuGraphAddHostNode, node, g, deps, n, &h);
  });
}

// cuMemAllocAsync and cuMemFreeAsync on a capturing stream make allocation and
// free nodes of the capture's graph.
int capture_alloc(ShimState& s, CUstream stream, size_t bytes, CUdeviceptr* out) {
  return capture_op(s, stream, [&](void* g, void* const* deps, size_t n, void** node) {
    vcu::MemAllocNodeParams a{};
    a.poolProps.allocType = 1;   // CU_MEM_ALLOCATION_TYPE_PINNED
    a.poolProps.location.type = 1;   // CU_MEM_LOCATION_TYPE_DEVICE
    a.poolProps.location.id = ctx_stack().empty() ? 0 : current_device(s);
    a.bytesize = bytes;
    const CUresult r = VGPU_BRIDGE(cuGraphAddMemAllocNode, node, g, deps, n, &a);
    if (r == CUDA_SUCCESS) *out = a.dptr;
    return r;
  });
}
int capture_free(ShimState& s, CUstream stream, CUdeviceptr ptr) {
  return capture_op(s, stream, [&](void* g, void* const* deps, size_t n, void** node) {
    return VGPU_BRIDGE(cuGraphAddMemFreeNode, node, g, deps, n, ptr);
  });
}

// Waiting for the stream, which a capture cannot do: the capture is
// invalidated, and the call says why (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED).
int capture_refuse_sync(ShimState& s, CUstream stream) {
  int status = 0;
  if (!stream_capturing(s, stream, &status)) return kNotCapturing;
  using Fn = int (*)(void*, const char*);
  static Fn fn = reinterpret_cast<Fn>(runtime_symbol("vgpu_cuCaptureRefuse", false));
  if (fn) fn(stream, "a synchronize or query of a capturing stream");
  return 900;   // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
}

// For the entry points that do not go through api(): the lock, and nothing to
// do before the first cuInit.
template <class F>
int with_capture(F&& f) {
  ShimState& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (!s.initialized) return kNotCapturing;
  return f(s);
}
