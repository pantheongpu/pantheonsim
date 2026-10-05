// The driver API's view of the CUDA graph machinery above: every cuGraph*,
// cuStream*Capture* and cuUserObject* entry point of libcuda lands here, as
// vgpu_<name> (vgpu_bridge.h), and works on the very records the runtime's
// cudaGraph* functions use -- a CUgraph is a cudaGraph_t, a CUgraphNode a
// cudaGraphNode_t.
//
// Included at the end of runtime_api.cpp, because the records and the locks
// are private to it. Everything here follows the file's one lock order: the
// runtime's lock (State::mu) first, then g_graph_mu.
//
// What an RTX 3060's driver (596.36, CUDA 13.2) answers for each call was
// measured and is noted where it decided something; the structures are the
// documented ones (vgpu_cu_abi.h). A driver call must not leave the runtime's
// "last error" behind, which the runtime's own functions do (KeepLastError).

VGPU_GRAPH_BRIDGE_LIST(VGPU_DECLARE_BRIDGE)

namespace {

// CUresult values this file returns itself. The rest come from cudaError_t,
// whose codes the driver's share for everything a graph call can say.
constexpr int kCuOk = 0, kCuInvalidContext = 201, kCuInvalidHandle = 400, kCuIllegalState = 401,
              kCuLaunchTimeout = 702;
// kCuInvalidValue (1) and kCuNotSupported (801) are defined above, with the tensor-map entry points.

// The runtime's thread-local last error is the runtime's: restore it when a
// driver call leaves, whatever the shared machinery did to it on the way.
struct KeepLastError {
  cudaError_t saved = g_last_error;
  ~KeepLastError() { g_last_error = saved; }
};

// cudaError_t to CUresult. The two enumerations agree on every code a graph
// call returns; these are the ones that differ.
int to_cu(cudaError_t e) {
  switch (static_cast<int>(e)) {
    case 7: return 701;     // cudaErrorLaunchOutOfResources
    case 9: return kCuInvalidValue;    // cudaErrorInvalidConfiguration
    case 13: return kCuInvalidValue;   // cudaErrorInvalidSymbol
    case 17: return kCuInvalidValue;   // cudaErrorInvalidDevicePointer
    case 21: return kCuInvalidValue;   // cudaErrorInvalidMemcpyDirection
    case 52: return kCuInvalidValue;   // cudaErrorMissingConfiguration
    case 98: return kCuInvalidHandle;  // cudaErrorInvalidDeviceFunction
    default: return static_cast<int>(e);
  }
}

// The body of a driver-view entry point: the runtime's lock, initialisation,
// and errors thrown inside turned into codes. The graph machinery does not
// need a live context, so a dead one is not asked about (libcuda's own api()
// wrapper has already answered for it).
template <class F>
int drv_call(const char* api, F&& body) {
  KeepLastError keep;
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  try {
    ensure_init(s);
    return body(s);
  } catch (const vgpu::Error& e) {
    if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", api, e.what());
    return to_cu(cudaErrorInvalidValue);
  } catch (const std::exception& e) {
    if (!quiet()) std::fprintf(stderr, "[vgpu] %s: %s\n", api, e.what());
    return 999;
  }
}

// ---- looking handles up ---------------------------------------------------

// A node handle that belongs to a graph, or nullptr. Caller holds g_graph_mu.
GraphNodeRec* node_of(void* h) {
  auto* n = static_cast<GraphNodeRec*>(h);
  return n && owner_of(n) ? n : nullptr;
}

GraphRec* exec_of(void* h) {   // caller holds g_graph_mu
  const auto it = g_graph_execs.find(h);
  return it == g_graph_execs.end() ? nullptr : it->second.get();
}

// The graph and the dependencies a driver call named: the same checks the
// runtime's own Add*Node calls make. Caller holds g_graph_mu.
bool graph_and_deps(void* graph, void* const* deps, size_t n, GraphRec** g, std::vector<GraphNodeRec*>* pred) {
  *g = graph_from(static_cast<cudaGraph_t>(graph));
  if (!*g) return false;
  if (n && !deps) return false;
  return deps_ok(**g, reinterpret_cast<const cudaGraphNode_t*>(deps), n, pred);
}

// Edge data: only default edges are modelled, as for the runtime's.
bool no_edge_data(const void* edge, size_t n) {
  if (!edge) return true;
  const auto* e = static_cast<const vcu::GraphEdgeData*>(edge);
  for (size_t i = 0; i < n; ++i)
    if (e[i].from_port || e[i].to_port || e[i].type) return false;
  return true;
}

// ---- kernel nodes ---------------------------------------------------------

// A kernel node's work from the driver's parameters: a CUfunction (or the
// function of a CUkernel when `func` is null), its launch shape, and the
// parameter values copied out -- from kernelParams, or from the packed block
// `extra` describes. Checked as an RTX 3060 checks it: no function at all is
// CUDA_ERROR_INVALID_HANDLE, a launch shape the device cannot run is
// CUDA_ERROR_INVALID_VALUE. Takes the runtime's lock; the caller must not
// hold g_graph_mu.
int fill_drv_kernel_work(RecordedLaunch* w, void* func, void* kern, unsigned gx, unsigned gy, unsigned gz,
                         unsigned bx, unsigned by, unsigned bz, unsigned shared, void** kernelParams,
                         void** extra) {
  void* const handle = func ? func : kern;
  VgpuDriverFuncInfo info;
  if (!handle || !driver_function_info(handle, &info)) return kCuInvalidHandle;
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  ensure_init(s);
  const vgpu::DeviceProfile& p = s.rt->device(info.device).profile();
  if (!gx || !gy || !gz || !bx || !by || !bz) return kCuInvalidValue;
  if (gx > p.limits.max_grid_dim[0] || gy > p.limits.max_grid_dim[1] || gz > p.limits.max_grid_dim[2] ||
      bx > p.limits.max_block_dim[0] || by > p.limits.max_block_dim[1] || bz > p.limits.max_block_dim[2] ||
      uint64_t{bx} * by * bz > p.limits.max_threads_per_block || shared > p.limits.shared_mem_per_block_optin)
    return kCuInvalidValue;
  const auto& params = info.fn->params;
  w->arg_bytes.assign(params.size(), {});
  if (kernelParams) {
    for (size_t i = 0; i < params.size(); ++i) {
      if (!kernelParams[i]) return kCuInvalidValue;
      w->arg_bytes[i].resize(params[i].size);
      std::memcpy(w->arg_bytes[i].data(), kernelParams[i], params[i].size);
    }
  } else if (extra) {
    const void* block = nullptr;
    size_t size = 0;
    if (!vgpu_launch::read_extra(extra, &block, &size)) return kCuInvalidValue;
    w->arg_bytes = vgpu_launch::unpack(*info.fn, block, size);
  } else if (!params.empty()) {
    return kCuInvalidValue;
  }
  w->kind = RecordedLaunch::Kind::Kernel;
  w->func = nullptr;
  w->drv_func = info.function;
  w->drv_kernel = kern ? kern : info.kernel;
  w->grid = dim3(gx, gy, gz);
  w->block = dim3(bx, by, bz);
  w->shared = shared;
  return kCuOk;
}

// The CUfunction of a kernel the runtime registered, as the driver sees it:
// minted by libcuda on first use and the same each time. Null when there is
// none.
void* function_for_stub(const void* stub) {
  using Fn = int (*)(int, uint64_t, const char*, void**);
  static const Fn fn = reinterpret_cast<Fn>(driver_symbol("vgpu_driver_function_for_v1"));
  if (!fn) return nullptr;
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  const auto it = s.kernels.find(stub);
  if (it == s.kernels.end() || !it->second.mod || !registered_ptx(s, *it->second.mod)) return nullptr;
  const uint64_t mid = module_on_current(s, *it->second.mod);
  void* out = nullptr;
  return fn(t_current_device, mid, it->second.entry_name.c_str(), &out) == 0 ? out : nullptr;
}

// What a kernel node reads back as: `v3` adds the CUkernel and the context,
// which the original CUDA_KERNEL_NODE_PARAMS does not have room for.
void write_kernel_params(GraphNodeRec* n, void* out, bool v3) {
  RecordedLaunch& w = n->work;
  auto* p = static_cast<vcu::KernelNodeParamsV3*>(out);
  p->func = w.drv_func ? w.drv_func : function_for_stub(w.func);
  p->gridDimX = w.grid.x; p->gridDimY = w.grid.y; p->gridDimZ = w.grid.z;
  p->blockDimX = w.block.x; p->blockDimY = w.block.y; p->blockDimZ = w.block.z;
  p->sharedMemBytes = static_cast<unsigned>(w.shared);
  // Pointers into the node's own copies, which is what the driver hands back too.
  w.arg_ptrs.resize(w.arg_bytes.size());
  for (size_t i = 0; i < w.arg_bytes.size(); ++i) w.arg_ptrs[i] = w.arg_bytes[i].data();
  p->kernelParams = w.arg_ptrs.data();
  p->extra = nullptr;
  if (v3) {
    p->kern = w.drv_kernel;
    VgpuDriverFuncInfo info;
    int device = 0;
    if (p->func && driver_function_info(p->func, &info)) device = info.device;
    using Ctx = void* (*)(int);
    static const Ctx primary = reinterpret_cast<Ctx>(driver_symbol("vgpu_driver_primary_ctx_v1"));
    p->ctx = primary ? primary(device) : nullptr;
  }
}


// ---- copy and fill nodes --------------------------------------------------

// A CUDA_MEMCPY3D between plain memory as a copy node's work. Pitches of zero
// mean tightly packed; a copy that names a CUDA array is kept as given and run
// through the driver's cuMemcpy3D when the graph runs.
int fill_drv_memcpy_work(RecordedLaunch* w, const vcu::Memcpy3D* c) {
  if (!c) return kCuInvalidValue;
  if (!c->WidthInBytes || !c->Height || !c->Depth) return kCuInvalidValue;
  const bool src_array = c->srcMemoryType == vcu::kMemArray, dst_array = c->dstMemoryType == vcu::kMemArray;
  const auto known = [](unsigned t) { return t >= vcu::kMemHost && t <= vcu::kMemUnified; };
  if (!known(c->srcMemoryType) || !known(c->dstMemoryType)) return kCuInvalidValue;
  const std::vector<uint8_t> raw(reinterpret_cast<const uint8_t*>(c), reinterpret_cast<const uint8_t*>(c) + sizeof *c);
  if (src_array || dst_array) {
    *w = RecordedLaunch{};
    w->kind = RecordedLaunch::Kind::Memcpy;
    w->blob_a = raw;
    const vcu::Memcpy3D copy = *c;
    w->host_op = [copy]() {
      using Fn = int (*)(const void*);
      static const Fn fn = reinterpret_cast<Fn>(driver_symbol("vgpu_driver_copy3d_v1"));
      if (fn) fn(&copy);
    };
    w->bytes = c->WidthInBytes;
    return kCuOk;
  }
  cudaMemcpy3DParms p{};
  const void* src = c->srcMemoryType == vcu::kMemHost ? c->srcHost : reinterpret_cast<const void*>(c->srcDevice);
  void* dst = c->dstMemoryType == vcu::kMemHost ? c->dstHost : reinterpret_cast<void*>(c->dstDevice);
  const size_t spitch = c->srcPitch ? c->srcPitch : c->WidthInBytes, dpitch = c->dstPitch ? c->dstPitch : c->WidthInBytes;
  p.srcPtr = cudaPitchedPtr{const_cast<void*>(src), spitch, spitch, c->srcHeight ? c->srcHeight : c->Height};
  p.dstPtr = cudaPitchedPtr{dst, dpitch, dpitch, c->dstHeight ? c->dstHeight : c->Height};
  p.srcPos = cudaPos{c->srcXInBytes, c->srcY, c->srcZ};
  p.dstPos = cudaPos{c->dstXInBytes, c->dstY, c->dstZ};
  p.extent = cudaExtent{c->WidthInBytes, c->Height, c->Depth};
  p.kind = cudaMemcpyDefault;
  if (const cudaError_t rc = fill_memcpy_work(w, &p); rc != cudaSuccess) return to_cu(rc);
  w->blob_a = raw;   // read back exactly as given
  return kCuOk;
}

int fill_drv_memset_work(RecordedLaunch* w, const vcu::MemsetNodeParams* m) {
  if (!m) return kCuInvalidValue;
  cudaMemsetParams p{};
  p.dst = reinterpret_cast<void*>(m->dst);
  p.pitch = m->pitch;
  p.value = m->value;
  p.elementSize = m->elementSize;
  p.width = m->width;
  p.height = m->height;
  return to_cu(fill_memset_work(w, &p));
}

// Adds a node the way every driver Add*Node call does: validates the graph and
// the dependencies, then adds. Caller holds g_graph_mu.
int add_node(void** out, void* graph, void* const* deps, size_t n, cudaGraphNodeType type, RecordedLaunch&& w,
             GraphNodeRec** made = nullptr) {
  GraphRec* g = nullptr;
  std::vector<GraphNodeRec*> pred;
  if (!out || !graph_and_deps(graph, deps, n, &g, &pred)) return kCuInvalidValue;
  GraphNodeRec* node = g->add(type, std::move(w), pred);
  *out = node;
  if (made) *made = node;
  return kCuOk;
}

}  // namespace

// ---- graphs ---------------------------------------------------------------

extern "C" VGPU_EXPORT int vgpu_runtime_marker_v1() { return 1; }

extern "C" VGPU_EXPORT int vgpu_cuGraphCreate(void** g, unsigned flags) {
  return drv_call("cuGraphCreate", [&](State&) { return to_cu(cudaGraphCreate(reinterpret_cast<cudaGraph_t*>(g), flags)); });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphDestroy(void* g) {
  return drv_call("cuGraphDestroy", [&](State&) { return to_cu(cudaGraphDestroy(static_cast<cudaGraph_t>(g))); });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphClone(void** out, void* g) {
  return drv_call("cuGraphClone", [&](State&) {
    return to_cu(cudaGraphClone(reinterpret_cast<cudaGraph_t*>(out), static_cast<cudaGraph_t>(g)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphNodeFindInClone(void** out, void* node, void* clone) {
  return drv_call("cuGraphNodeFindInClone", [&](State&) {
    return to_cu(cudaGraphNodeFindInClone(reinterpret_cast<cudaGraphNode_t*>(out), static_cast<cudaGraphNode_t>(node),
                                          static_cast<cudaGraph_t>(clone)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphGetNodes(void* g, void** nodes, size_t* n) {
  return drv_call("cuGraphGetNodes", [&](State&) {
    return to_cu(cudaGraphGetNodes(static_cast<cudaGraph_t>(g), reinterpret_cast<cudaGraphNode_t*>(nodes), n));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphGetRootNodes(void* g, void** nodes, size_t* n) {
  return drv_call("cuGraphGetRootNodes", [&](State&) {
    return to_cu(cudaGraphGetRootNodes(static_cast<cudaGraph_t>(g), reinterpret_cast<cudaGraphNode_t*>(nodes), n));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphGetEdges(void* g, void** from, void** to, size_t* n) {
  return drv_call("cuGraphGetEdges", [&](State&) {
    return to_cu(graph_get_edges(static_cast<cudaGraph_t>(g), reinterpret_cast<cudaGraphNode_t*>(from),
                                 reinterpret_cast<cudaGraphNode_t*>(to), n));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphGetEdges_v2(void* g, void** from, void** to, void* edge, size_t* n) {
  const int rc = vgpu_cuGraphGetEdges(g, from, to, n);
  if (rc == kCuOk && edge && n) std::memset(edge, 0, *n * sizeof(vcu::GraphEdgeData));   // every edge is a plain dependency
  return rc;
}
extern "C" VGPU_EXPORT int vgpu_cuGraphNodeGetType(void* node, int* type) {
  return drv_call("cuGraphNodeGetType", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || !type) return kCuInvalidValue;
    *type = static_cast<int>(n->type);
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphNodeGetDependencies(void* node, void** deps, size_t* n) {
  return drv_call("cuGraphNodeGetDependencies", [&](State&) {
    return to_cu(node_get_deps(static_cast<cudaGraphNode_t>(node), reinterpret_cast<cudaGraphNode_t*>(deps), n));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphNodeGetDependencies_v2(void* node, void** deps, void* edge, size_t* n) {
  const int rc = vgpu_cuGraphNodeGetDependencies(node, deps, n);
  if (rc == kCuOk && edge && n) std::memset(edge, 0, *n * sizeof(vcu::GraphEdgeData));
  return rc;
}
extern "C" VGPU_EXPORT int vgpu_cuGraphNodeGetDependentNodes(void* node, void** deps, size_t* n) {
  return drv_call("cuGraphNodeGetDependentNodes", [&](State&) {
    return to_cu(node_get_dependents(static_cast<cudaGraphNode_t>(node), reinterpret_cast<cudaGraphNode_t*>(deps), n));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphNodeGetDependentNodes_v2(void* node, void** deps, void* edge, size_t* n) {
  const int rc = vgpu_cuGraphNodeGetDependentNodes(node, deps, n);
  if (rc == kCuOk && edge && n) std::memset(edge, 0, *n * sizeof(vcu::GraphEdgeData));
  return rc;
}
extern "C" VGPU_EXPORT int vgpu_cuGraphAddDependencies(void* g, void* const* from, void* const* to, size_t n) {
  return drv_call("cuGraphAddDependencies", [&](State&) {
    return to_cu(graph_add_deps(static_cast<cudaGraph_t>(g), reinterpret_cast<const cudaGraphNode_t*>(from),
                                reinterpret_cast<const cudaGraphNode_t*>(to), n));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphAddDependencies_v2(void* g, void* const* from, void* const* to, const void* edge,
                                                          size_t n) {
  if (!no_edge_data(edge, n)) return kCuNotSupported;
  return vgpu_cuGraphAddDependencies(g, from, to, n);
}
extern "C" VGPU_EXPORT int vgpu_cuGraphRemoveDependencies(void* g, void* const* from, void* const* to, size_t n) {
  return drv_call("cuGraphRemoveDependencies", [&](State&) {
    return to_cu(graph_remove_deps(static_cast<cudaGraph_t>(g), reinterpret_cast<const cudaGraphNode_t*>(from),
                                   reinterpret_cast<const cudaGraphNode_t*>(to), n));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphRemoveDependencies_v2(void* g, void* const* from, void* const* to,
                                                             const void* edge, size_t n) {
  if (!no_edge_data(edge, n)) return kCuNotSupported;
  return vgpu_cuGraphRemoveDependencies(g, from, to, n);
}
extern "C" VGPU_EXPORT int vgpu_cuGraphDestroyNode(void* node) {
  return drv_call("cuGraphDestroyNode", [&](State&) { return to_cu(cudaGraphDestroyNode(static_cast<cudaGraphNode_t>(node))); });
}

// ---- adding nodes ---------------------------------------------------------

extern "C" VGPU_EXPORT int vgpu_cuGraphAddEmptyNode(void** node, void* g, void* const* deps, size_t n) {
  return drv_call("cuGraphAddEmptyNode", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    return add_node(node, g, deps, n, cudaGraphNodeTypeEmpty, RecordedLaunch{});
  });
}

// The original structure has the first ten fields of the later ones: reading
// it as the longer one would read past what the caller has.
static int add_kernel_node(void** node, void* g, void* const* deps, size_t n, const void* p, bool v2) {
  if (!p) return kCuInvalidValue;
  const auto* k = static_cast<const vcu::KernelNodeParamsV3*>(p);
  RecordedLaunch w;
  const int rc = fill_drv_kernel_work(&w, k->func, v2 ? k->kern : nullptr, k->gridDimX, k->gridDimY, k->gridDimZ,
                                      k->blockDimX, k->blockDimY, k->blockDimZ, k->sharedMemBytes, k->kernelParams,
                                      k->extra);
  if (rc != kCuOk) return rc;
  return drv_call("cuGraphAddKernelNode", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    return add_node(node, g, deps, n, cudaGraphNodeTypeKernel, std::move(w));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphAddKernelNode(void** node, void* g, void* const* deps, size_t n, const void* p) {
  return add_kernel_node(node, g, deps, n, p, false);
}
extern "C" VGPU_EXPORT int vgpu_cuGraphAddKernelNode_v2(void** node, void* g, void* const* deps, size_t n, const void* p) {
  return add_kernel_node(node, g, deps, n, p, true);
}

static int get_kernel_params(void* node, void* p, bool v3) {
  return drv_call("cuGraphKernelNodeGetParams", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || !p || n->type != cudaGraphNodeTypeKernel) return kCuInvalidValue;
    write_kernel_params(n, p, v3);
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphKernelNodeGetParams(void* node, void* p) { return get_kernel_params(node, p, false); }
extern "C" VGPU_EXPORT int vgpu_cuGraphKernelNodeGetParams_v2(void* node, void* p) { return get_kernel_params(node, p, true); }

// The kernel a node runs may change too (measured), within the device.
static int set_kernel_params(void* node, const void* p, bool v2) {
  if (!p) return kCuInvalidValue;
  const auto* k = static_cast<const vcu::KernelNodeParamsV3*>(p);
  RecordedLaunch w;
  const int rc = fill_drv_kernel_work(&w, k->func, v2 ? k->kern : nullptr, k->gridDimX, k->gridDimY, k->gridDimZ,
                                      k->blockDimX, k->blockDimY, k->blockDimZ, k->sharedMemBytes, k->kernelParams,
                                      k->extra);
  if (rc != kCuOk) return rc;
  return drv_call("cuGraphKernelNodeSetParams", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || n->type != cudaGraphNodeTypeKernel) return kCuInvalidValue;
    w.attrs = n->work.attrs;
    n->work = std::move(w);
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphKernelNodeSetParams(void* node, const void* p) { return set_kernel_params(node, p, false); }
extern "C" VGPU_EXPORT int vgpu_cuGraphKernelNodeSetParams_v2(void* node, const void* p) { return set_kernel_params(node, p, true); }

extern "C" VGPU_EXPORT int vgpu_cuGraphAddMemcpyNode(void** node, void* g, void* const* deps, size_t n, const void* p,
                                                      void* ctx) {
  (void)ctx;
  if (!p) return kCuInvalidValue;
  RecordedLaunch w;
  if (const int rc = fill_drv_memcpy_work(&w, static_cast<const vcu::Memcpy3D*>(p)); rc != kCuOk) return rc;
  return drv_call("cuGraphAddMemcpyNode", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    return add_node(node, g, deps, n, cudaGraphNodeTypeMemcpy, std::move(w));
  });
}

// A copy node as the driver reads it back: exactly what was given, or, for a
// node the runtime made, the CUDA_MEMCPY3D its parameters describe.
extern "C" VGPU_EXPORT int vgpu_cuGraphMemcpyNodeGetParams(void* node, void* p) {
  return drv_call("cuGraphMemcpyNodeGetParams", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || !p || n->type != cudaGraphNodeTypeMemcpy) return kCuInvalidValue;
    if (n->work.blob_a.size() == sizeof(vcu::Memcpy3D)) {
      std::memcpy(p, n->work.blob_a.data(), sizeof(vcu::Memcpy3D));
      return kCuOk;
    }
    const cudaMemcpy3DParms& c = n->work.copy_params;
    vcu::Memcpy3D out{};
    const auto side = [](void* ptr, unsigned* type, const void** host, vcu::DevPtr* dev) {
      if (is_device_ptr(ptr)) {
        *type = vcu::kMemDevice;
        *dev = reinterpret_cast<vcu::DevPtr>(ptr);
      } else {
        *type = vcu::kMemHost;
        *host = ptr;
      }
    };
    const void* dst_host = nullptr;
    side(c.srcPtr.ptr, &out.srcMemoryType, &out.srcHost, &out.srcDevice);
    side(c.dstPtr.ptr, &out.dstMemoryType, &dst_host, &out.dstDevice);
    out.dstHost = const_cast<void*>(dst_host);
    out.srcPitch = c.srcPtr.pitch;
    out.srcHeight = c.srcPtr.ysize;
    out.dstPitch = c.dstPtr.pitch;
    out.dstHeight = c.dstPtr.ysize;
    out.srcXInBytes = c.srcPos.x; out.srcY = c.srcPos.y; out.srcZ = c.srcPos.z;
    out.dstXInBytes = c.dstPos.x; out.dstY = c.dstPos.y; out.dstZ = c.dstPos.z;
    out.WidthInBytes = c.extent.width; out.Height = c.extent.height; out.Depth = c.extent.depth;
    std::memcpy(p, &out, sizeof out);
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphMemcpyNodeSetParams(void* node, const void* p) {
  return drv_call("cuGraphMemcpyNodeSetParams", [&](State&) {
    RecordedLaunch w;
    if (const int rc = fill_drv_memcpy_work(&w, static_cast<const vcu::Memcpy3D*>(p)); rc != kCuOk) return rc;
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || n->type != cudaGraphNodeTypeMemcpy) return kCuInvalidValue;
    n->work = std::move(w);
    return kCuOk;
  });
}

extern "C" VGPU_EXPORT int vgpu_cuGraphAddMemsetNode(void** node, void* g, void* const* deps, size_t n, const void* p,
                                                      void* ctx) {
  (void)ctx;   // measured: a null context is accepted
  return drv_call("cuGraphAddMemsetNode", [&](State&) {
    RecordedLaunch w;
    if (const int rc = fill_drv_memset_work(&w, static_cast<const vcu::MemsetNodeParams*>(p)); rc != kCuOk) return rc;
    std::lock_guard<std::mutex> lock(g_graph_mu);
    return add_node(node, g, deps, n, cudaGraphNodeTypeMemset, std::move(w));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphMemsetNodeGetParams(void* node, void* p) {
  return drv_call("cuGraphMemsetNodeGetParams", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || !p || n->type != cudaGraphNodeTypeMemset) return kCuInvalidValue;
    const cudaMemsetParams& f = n->work.fill_params;   // as given
    auto* out = static_cast<vcu::MemsetNodeParams*>(p);
    out->dst = reinterpret_cast<vcu::DevPtr>(f.dst);
    out->pitch = f.pitch;
    out->value = f.value;
    out->elementSize = f.elementSize;
    out->width = f.width;
    out->height = f.height;
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphMemsetNodeSetParams(void* node, const void* p) {
  return drv_call("cuGraphMemsetNodeSetParams", [&](State&) {
    RecordedLaunch w;
    if (const int rc = fill_drv_memset_work(&w, static_cast<const vcu::MemsetNodeParams*>(p)); rc != kCuOk) return rc;
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || n->type != cudaGraphNodeTypeMemset) return kCuInvalidValue;
    n->work = std::move(w);
    return kCuOk;
  });
}

extern "C" VGPU_EXPORT int vgpu_cuGraphAddHostNode(void** node, void* g, void* const* deps, size_t n, const void* p) {
  return drv_call("cuGraphAddHostNode", [&](State&) {
    if (!p) return kCuInvalidValue;
    const auto* h = static_cast<const vcu::HostNodeParams*>(p);
    RecordedLaunch w;
    w.kind = RecordedLaunch::Kind::Host;
    w.host_fn = h->fn;
    w.host_user = h->userData;
    std::lock_guard<std::mutex> lock(g_graph_mu);
    return add_node(node, g, deps, n, cudaGraphNodeTypeHost, std::move(w));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphHostNodeGetParams(void* node, void* p) {
  return drv_call("cuGraphHostNodeGetParams", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || !p || n->type != cudaGraphNodeTypeHost) return kCuInvalidValue;
    auto* out = static_cast<vcu::HostNodeParams*>(p);
    out->fn = n->work.host_fn;
    out->userData = n->work.host_user;
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphHostNodeSetParams(void* node, const void* p) {
  return drv_call("cuGraphHostNodeSetParams", [&](State&) {
    const auto* h = static_cast<const vcu::HostNodeParams*>(p);
    if (!h || !h->fn) return kCuInvalidValue;
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || n->type != cudaGraphNodeTypeHost) return kCuInvalidValue;
    n->work.host_fn = h->fn;
    n->work.host_user = h->userData;
    n->work.host_op = nullptr;
    return kCuOk;
  });
}

extern "C" VGPU_EXPORT int vgpu_cuGraphAddChildGraphNode(void** node, void* g, void* const* deps, size_t n, void* child) {
  return drv_call("cuGraphAddChildGraphNode", [&](State&) {
    return to_cu(cudaGraphAddChildGraphNode(reinterpret_cast<cudaGraphNode_t*>(node), static_cast<cudaGraph_t>(g),
                                            reinterpret_cast<const cudaGraphNode_t*>(deps), n,
                                            static_cast<cudaGraph_t>(child)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphChildGraphNodeGetGraph(void* node, void** g) {
  return drv_call("cuGraphChildGraphNodeGetGraph", [&](State&) {
    return to_cu(cudaGraphChildGraphNodeGetGraph(static_cast<cudaGraphNode_t>(node), reinterpret_cast<cudaGraph_t*>(g)));
  });
}

extern "C" VGPU_EXPORT int vgpu_cuGraphAddEventRecordNode(void** node, void* g, void* const* deps, size_t n, void* ev) {
  return drv_call("cuGraphAddEventRecordNode", [&](State&) {
    return to_cu(cudaGraphAddEventRecordNode(reinterpret_cast<cudaGraphNode_t*>(node), static_cast<cudaGraph_t>(g),
                                             reinterpret_cast<const cudaGraphNode_t*>(deps), n,
                                             static_cast<cudaEvent_t>(ev)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphEventRecordNodeGetEvent(void* node, void** ev) {
  return drv_call("cuGraphEventRecordNodeGetEvent", [&](State&) {
    return to_cu(event_node_get(static_cast<cudaGraphNode_t>(node), reinterpret_cast<cudaEvent_t*>(ev),
                                cudaGraphNodeTypeEventRecord));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphEventRecordNodeSetEvent(void* node, void* ev) {
  return drv_call("cuGraphEventRecordNodeSetEvent", [&](State&) {
    return to_cu(event_node_set(static_cast<cudaGraphNode_t>(node), static_cast<cudaEvent_t>(ev), cudaGraphNodeTypeEventRecord));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphAddEventWaitNode(void** node, void* g, void* const* deps, size_t n, void* ev) {
  return drv_call("cuGraphAddEventWaitNode", [&](State&) {
    return to_cu(cudaGraphAddEventWaitNode(reinterpret_cast<cudaGraphNode_t*>(node), static_cast<cudaGraph_t>(g),
                                           reinterpret_cast<const cudaGraphNode_t*>(deps), n,
                                           static_cast<cudaEvent_t>(ev)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphEventWaitNodeGetEvent(void* node, void** ev) {
  return drv_call("cuGraphEventWaitNodeGetEvent", [&](State&) {
    return to_cu(event_node_get(static_cast<cudaGraphNode_t>(node), reinterpret_cast<cudaEvent_t*>(ev),
                                cudaGraphNodeTypeWaitEvent));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphEventWaitNodeSetEvent(void* node, void* ev) {
  return drv_call("cuGraphEventWaitNodeSetEvent", [&](State&) {
    return to_cu(event_node_set(static_cast<cudaGraphNode_t>(node), static_cast<cudaEvent_t>(ev), cudaGraphNodeTypeWaitEvent));
  });
}

// ---- memory allocation nodes ----------------------------------------------

static_assert(sizeof(vcu::MemPoolProps) == sizeof(cudaMemPoolProps), "CUmemPoolProps and cudaMemPoolProps differ");
extern "C" VGPU_EXPORT int vgpu_cuGraphAddMemAllocNode(void** node, void* g, void* const* deps, size_t n, void* p) {
  return drv_call("cuGraphAddMemAllocNode", [&](State&) {
    if (!p) return kCuInvalidValue;
    auto* in = static_cast<vcu::MemAllocNodeParams*>(p);
    cudaMemAllocNodeParams a{};
    std::memcpy(&a.poolProps, &in->poolProps, sizeof a.poolProps);
    a.accessDescs = reinterpret_cast<const cudaMemAccessDesc*>(in->accessDescs);
    a.accessDescCount = in->accessDescCount;
    a.bytesize = in->bytesize;
    const cudaError_t rc = cudaGraphAddMemAllocNode(reinterpret_cast<cudaGraphNode_t*>(node), static_cast<cudaGraph_t>(g),
                                                    reinterpret_cast<const cudaGraphNode_t*>(deps), n, &a);
    if (rc == cudaSuccess) in->dptr = reinterpret_cast<vcu::DevPtr>(a.dptr);   // the one output
    return to_cu(rc);
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphMemAllocNodeGetParams(void* node, void* p) {
  return drv_call("cuGraphMemAllocNodeGetParams", [&](State&) {
    if (!p) return kCuInvalidValue;
    cudaMemAllocNodeParams a{};
    const cudaError_t rc = cudaGraphMemAllocNodeGetParams(static_cast<cudaGraphNode_t>(node), &a);
    if (rc != cudaSuccess) return to_cu(rc);
    auto* out = static_cast<vcu::MemAllocNodeParams*>(p);
    std::memcpy(&out->poolProps, &a.poolProps, sizeof out->poolProps);
    out->accessDescs = nullptr;
    out->accessDescCount = 0;
    out->bytesize = a.bytesize;
    out->dptr = reinterpret_cast<vcu::DevPtr>(a.dptr);
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphAddMemFreeNode(void** node, void* g, void* const* deps, size_t n,
                                                       unsigned long long dptr) {
  return drv_call("cuGraphAddMemFreeNode", [&](State&) {
    return to_cu(cudaGraphAddMemFreeNode(reinterpret_cast<cudaGraphNode_t*>(node), static_cast<cudaGraph_t>(g),
                                         reinterpret_cast<const cudaGraphNode_t*>(deps), n,
                                         reinterpret_cast<void*>(dptr)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphMemFreeNodeGetParams(void* node, unsigned long long* dptr) {
  return drv_call("cuGraphMemFreeNodeGetParams", [&](State&) {
    void* out = nullptr;
    if (!dptr) return kCuInvalidValue;
    const cudaError_t rc = cudaGraphMemFreeNodeGetParams(static_cast<cudaGraphNode_t>(node), &out);
    if (rc == cudaSuccess) *dptr = reinterpret_cast<unsigned long long>(out);
    return to_cu(rc);
  });
}

// ---- external semaphore and batch memory-operation nodes -------------------
//
// Kept as the bytes the program gave and read back as given. An external
// semaphore node holds the handles (blob_a) and the per-semaphore parameters
// (blob_b); a batch node holds its operations (blob_a). What running one does
// is in the functions below and in the external-resource section.

namespace {

constexpr unsigned kStreamMemOpWait32 = 1, kStreamMemOpWrite32 = 2, kStreamMemOpFlush = 3, kStreamMemOpWait64 = 4,
                   kStreamMemOpWrite64 = 5, kStreamMemOpBarrier = 6;

int fill_ext_sem_work_impl(RecordedLaunch* w, vcu::ExtSem* sems, const void* params, size_t param_size,
                           unsigned count) {
  if (count && (!sems || !params)) return kCuInvalidValue;
  w->kind = RecordedLaunch::Kind::Host;
  w->count = count;
  w->blob_a.assign(reinterpret_cast<const uint8_t*>(sems), reinterpret_cast<const uint8_t*>(sems) + count * sizeof(void*));
  w->blob_b.assign(static_cast<const uint8_t*>(params), static_cast<const uint8_t*>(params) + count * param_size);
  return kCuOk;
}

int fill_batch_work(RecordedLaunch* w, const vcu::BatchMemOpNodeParams* p) {
  if (!p || (p->count && !p->paramArray) || p->flags) return kCuInvalidValue;
  for (unsigned i = 0; i < p->count; ++i) {
    const unsigned op = static_cast<unsigned>(p->paramArray[i].pad[0] & 0xffffffffu);
    if (op < kStreamMemOpWait32 || op > kStreamMemOpBarrier) return kCuInvalidValue;
  }
  w->kind = RecordedLaunch::Kind::Host;
  w->count = p->count;
  w->blob_a.assign(reinterpret_cast<const uint8_t*>(p->paramArray),
                   reinterpret_cast<const uint8_t*>(p->paramArray) + p->count * sizeof(vcu::BatchMemOpParams));
  return kCuOk;
}

}  // namespace

// Stream memory operations (cuStreamWriteValue32, cuStreamWaitValue32,
// cuStreamBatchMemOp and a graph's batch node), on memory the machine holds.
// Everything ahead of one has finished -- one synchronous engine -- so a wait
// whose condition is false can never become true: the real device would
// block, and this reports it instead of hanging.
static cudaError_t run_batch_memops(const std::vector<uint8_t>& blob, unsigned count) {
  return guard("cuStreamBatchMemOp", [&](State& s) -> cudaError_t {
    const auto* ops = reinterpret_cast<const vcu::BatchMemOpParams*>(blob.data());
    for (unsigned i = 0; i < count; ++i) {
      uint32_t w32[12];
      std::memcpy(w32, &ops[i], sizeof w32);
      uint64_t w64[6];
      std::memcpy(w64, &ops[i], sizeof w64);
      const unsigned op = w32[0];
      if (op == kStreamMemOpFlush || op == kStreamMemOpBarrier) continue;
      const uint64_t addr = w64[1];
      const bool wide = op == kStreamMemOpWait64 || op == kStreamMemOpWrite64;
      const size_t bytes = wide ? 8 : 4;
      uint64_t val = w64[2];
      if (!wide) val &= 0xffffffffu;
      uint8_t* host = !is_device_ptr(reinterpret_cast<void*>(addr)) ? reinterpret_cast<uint8_t*>(addr) : nullptr;
      if (op == kStreamMemOpWrite32 || op == kStreamMemOpWrite64) {
        if (host) std::memcpy(host, &val, bytes);
        else owner_memory(s, reinterpret_cast<void*>(addr)).write(addr, &val, bytes);
        continue;
      }
      uint64_t cur = 0;
      if (host) std::memcpy(&cur, host, bytes);
      else owner_memory(s, reinterpret_cast<void*>(addr)).read(addr, &cur, bytes);
      const unsigned cond = w32[6] & 0xf;   // flags: GEQ 0, EQ 1, AND 2, NOR 3 (bit 30 flush)
      bool ok = false;
      switch (cond) {
        case 0: ok = wide ? static_cast<int64_t>(cur - val) >= 0 : static_cast<int32_t>(static_cast<uint32_t>(cur) - static_cast<uint32_t>(val)) >= 0; break;
        case 1: ok = cur == val; break;
        case 2: ok = (cur & val) != 0; break;
        case 3: ok = (~(cur | val) & (wide ? ~0ull : 0xffffffffull)) != 0; break;
        default: return cudaErrorInvalidValue;
      }
      if (!ok) {
        if (!quiet())
          std::fprintf(stderr, "[vgpu] stream memory wait on %p can never be satisfied: this simulator runs one stream at a time, "
                               "so nothing could write the value while the wait blocked\n", reinterpret_cast<void*>(addr));
        return cudaErrorLaunchTimeout;
      }
    }
    return cudaSuccess;
  });
}

static cudaError_t run_batch_memop_node(const RecordedLaunch& rl) { return run_batch_memops(rl.blob_a, rl.count); }

// Defined with the external resources, below: signalling and waiting on the
// semaphores an external-semaphore node names.
static cudaError_t run_ext_sem_node(const RecordedLaunch& rl, bool signal);

namespace {
int add_ext_sem(void** node, void* g, void* const* deps, size_t n, vcu::ExtSem* sems, const void* params, size_t psz,
                unsigned count, bool signal) {
  RecordedLaunch w;
  if (const int rc = fill_ext_sem_work_impl(&w, sems, params, psz, count); rc != kCuOk) return rc;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  return add_node(node, g, deps, n, signal ? cudaGraphNodeTypeExtSemaphoreSignal : cudaGraphNodeTypeExtSemaphoreWait,
                  std::move(w));
}
}  // namespace

extern "C" VGPU_EXPORT int vgpu_cuGraphAddExternalSemaphoresSignalNode(void** node, void* g, void* const* deps, size_t n,
                                                                        const void* p) {
  return drv_call("cuGraphAddExternalSemaphoresSignalNode", [&](State&) {
    if (!p) return kCuInvalidValue;
    const auto* e = static_cast<const vcu::ExtSemSignalNodeParams*>(p);
    return add_ext_sem(node, g, deps, n, e->extSemArray, e->paramsArray, sizeof(vcu::ExtSemSignalParams), e->numExtSems, true);
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphAddExternalSemaphoresWaitNode(void** node, void* g, void* const* deps, size_t n,
                                                                      const void* p) {
  return drv_call("cuGraphAddExternalSemaphoresWaitNode", [&](State&) {
    if (!p) return kCuInvalidValue;
    const auto* e = static_cast<const vcu::ExtSemWaitNodeParams*>(p);
    return add_ext_sem(node, g, deps, n, e->extSemArray, e->paramsArray, sizeof(vcu::ExtSemWaitParams), e->numExtSems, false);
  });
}

namespace {
// Get/set for both external semaphore node types.
int get_ext_sem(void* node, void* p, cudaGraphNodeType type) {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphNodeRec* n = node_of(node);
  if (!n || !p || n->type != type) return kCuInvalidValue;
  struct Out { vcu::ExtSem* sems; const void* params; unsigned count; };
  auto* out = static_cast<Out*>(p);
  out->sems = n->work.count ? reinterpret_cast<vcu::ExtSem*>(n->work.blob_a.data()) : nullptr;
  out->params = n->work.count ? n->work.blob_b.data() : nullptr;
  out->count = n->work.count;
  return kCuOk;
}
int set_ext_sem(void* node, vcu::ExtSem* sems, const void* params, size_t psz, unsigned count, cudaGraphNodeType type,
                void* exec) {
  RecordedLaunch w;
  if (const int rc = fill_ext_sem_work_impl(&w, sems, params, psz, count); rc != kCuOk) return rc;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphNodeRec* n = node_of(node);
  if (!n || n->type != type) return kCuInvalidValue;
  if (exec) {
    const auto it = g_graph_execs.find(exec);
    if (it == g_graph_execs.end()) return kCuInvalidValue;
    GraphNodeRec* target = exec_twin(*it->second, n);
    if (!target || target->type != type) return kCuInvalidValue;
    n = target;
  }
  n->work = std::move(w);
  return kCuOk;
}
}  // namespace

extern "C" VGPU_EXPORT int vgpu_cuGraphExternalSemaphoresSignalNodeGetParams(void* node, void* p) {
  return drv_call("cuGraphExternalSemaphoresSignalNodeGetParams", [&](State&) { return get_ext_sem(node, p, cudaGraphNodeTypeExtSemaphoreSignal); });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExternalSemaphoresWaitNodeGetParams(void* node, void* p) {
  return drv_call("cuGraphExternalSemaphoresWaitNodeGetParams", [&](State&) { return get_ext_sem(node, p, cudaGraphNodeTypeExtSemaphoreWait); });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExternalSemaphoresSignalNodeSetParams(void* node, const void* p) {
  return drv_call("cuGraphExternalSemaphoresSignalNodeSetParams", [&](State&) {
    if (!p) return kCuInvalidValue;
    const auto* e = static_cast<const vcu::ExtSemSignalNodeParams*>(p);
    return set_ext_sem(node, e->extSemArray, e->paramsArray, sizeof(vcu::ExtSemSignalParams), e->numExtSems,
                       cudaGraphNodeTypeExtSemaphoreSignal, nullptr);
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExternalSemaphoresWaitNodeSetParams(void* node, const void* p) {
  return drv_call("cuGraphExternalSemaphoresWaitNodeSetParams", [&](State&) {
    if (!p) return kCuInvalidValue;
    const auto* e = static_cast<const vcu::ExtSemWaitNodeParams*>(p);
    return set_ext_sem(node, e->extSemArray, e->paramsArray, sizeof(vcu::ExtSemWaitParams), e->numExtSems,
                       cudaGraphNodeTypeExtSemaphoreWait, nullptr);
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExecExternalSemaphoresSignalNodeSetParams(void* exec, void* node, const void* p) {
  return drv_call("cuGraphExecExternalSemaphoresSignalNodeSetParams", [&](State&) {
    if (!p) return kCuInvalidValue;
    const auto* e = static_cast<const vcu::ExtSemSignalNodeParams*>(p);
    return set_ext_sem(node, e->extSemArray, e->paramsArray, sizeof(vcu::ExtSemSignalParams), e->numExtSems,
                       cudaGraphNodeTypeExtSemaphoreSignal, exec);
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExecExternalSemaphoresWaitNodeSetParams(void* exec, void* node, const void* p) {
  return drv_call("cuGraphExecExternalSemaphoresWaitNodeSetParams", [&](State&) {
    if (!p) return kCuInvalidValue;
    const auto* e = static_cast<const vcu::ExtSemWaitNodeParams*>(p);
    return set_ext_sem(node, e->extSemArray, e->paramsArray, sizeof(vcu::ExtSemWaitParams), e->numExtSems,
                       cudaGraphNodeTypeExtSemaphoreWait, exec);
  });
}

extern "C" VGPU_EXPORT int vgpu_cuGraphAddBatchMemOpNode(void** node, void* g, void* const* deps, size_t n, const void* p) {
  return drv_call("cuGraphAddBatchMemOpNode", [&](State&) {
    RecordedLaunch w;
    if (const int rc = fill_batch_work(&w, static_cast<const vcu::BatchMemOpNodeParams*>(p)); rc != kCuOk) return rc;
    w.flags = 0;
    std::lock_guard<std::mutex> lock(g_graph_mu);
    return add_node(node, g, deps, n, static_cast<cudaGraphNodeType>(vcu::kNodeBatchMemOp), std::move(w));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphBatchMemOpNodeGetParams(void* node, void* p) {
  return drv_call("cuGraphBatchMemOpNodeGetParams", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || !p || static_cast<int>(n->type) != vcu::kNodeBatchMemOp) return kCuInvalidValue;
    auto* out = static_cast<vcu::BatchMemOpNodeParams*>(p);
    using Ctx = void* (*)(int);
    static const Ctx primary = reinterpret_cast<Ctx>(driver_symbol("vgpu_driver_primary_ctx_v1"));
    out->ctx = primary ? primary(t_current_device) : nullptr;
    out->count = n->work.count;
    out->paramArray = n->work.count ? reinterpret_cast<vcu::BatchMemOpParams*>(n->work.blob_a.data()) : nullptr;
    out->flags = 0;
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphBatchMemOpNodeSetParams(void* node, const void* p) {
  return drv_call("cuGraphBatchMemOpNodeSetParams", [&](State&) {
    RecordedLaunch w;
    if (const int rc = fill_batch_work(&w, static_cast<const vcu::BatchMemOpNodeParams*>(p)); rc != kCuOk) return rc;
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || static_cast<int>(n->type) != vcu::kNodeBatchMemOp) return kCuInvalidValue;
    n->work = std::move(w);
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExecBatchMemOpNodeSetParams(void* exec, void* node, const void* p) {
  return drv_call("cuGraphExecBatchMemOpNodeSetParams", [&](State&) {
    RecordedLaunch w;
    if (const int rc = fill_batch_work(&w, static_cast<const vcu::BatchMemOpNodeParams*>(p)); rc != kCuOk) return rc;
    std::lock_guard<std::mutex> lock(g_graph_mu);
    const auto it = g_graph_execs.find(exec);
    GraphNodeRec* n = node_of(node);
    if (it == g_graph_execs.end() || !n) return kCuInvalidValue;
    GraphNodeRec* target = exec_twin(*it->second, n);
    if (!target || static_cast<int>(target->type) != vcu::kNodeBatchMemOp) return kCuInvalidValue;
    // The update cannot change how many operations there are, as documented.
    if (target->work.count != w.count) return kCuInvalidValue;
    target->work = std::move(w);
    return kCuOk;
  });
}

// ---- kernel node attributes -------------------------------------------------
//
// CUlaunchAttributeID 1..15 are the ones a kernel node takes (measured: 0, the
// ignore id, and anything past the last are CUDA_ERROR_INVALID_VALUE). A
// cooperative launch and a cluster shape change how the node runs, so those two
// live in the node's launch description; the rest are kept and read back, as
// scheduling hints have nothing to act on here.
namespace {
constexpr int kAttrCooperative = 2, kAttrClusterDimension = 4;
bool kernel_attr_id(int id) { return id >= 1 && id <= 15; }

void read_attr(const GraphNodeRec& n, int id, uint8_t out[64]) {
  std::memset(out, 0, 64);
  if (id == kAttrCooperative) {
    const int v = n.work.cooperative ? 1 : 0;
    std::memcpy(out, &v, sizeof v);
  } else if (id == kAttrClusterDimension) {
    std::memcpy(out, n.work.cluster.data(), 12);
  } else if (const auto it = n.work.attrs.find(id); it != n.work.attrs.end()) {
    std::memcpy(out, it->second.data(), 64);
  }
}
void write_attr(GraphNodeRec* n, int id, const uint8_t in[64]) {
  if (id == kAttrCooperative) {
    int v = 0;
    std::memcpy(&v, in, sizeof v);
    n->work.cooperative = v != 0;
  } else if (id == kAttrClusterDimension) {
    std::memcpy(n->work.cluster.data(), in, 12);
  } else {
    std::memcpy(n->work.attrs[id].data(), in, 64);
  }
}
}  // namespace

extern "C" VGPU_EXPORT int vgpu_cuGraphKernelNodeGetAttribute(void* node, int attr, void* value) {
  return drv_call("cuGraphKernelNodeGetAttribute", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || !value || !kernel_attr_id(attr) || n->type != cudaGraphNodeTypeKernel) return kCuInvalidValue;
    read_attr(*n, attr, static_cast<uint8_t*>(value));
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphKernelNodeSetAttribute(void* node, int attr, const void* value) {
  return drv_call("cuGraphKernelNodeSetAttribute", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || !value || !kernel_attr_id(attr) || n->type != cudaGraphNodeTypeKernel) return kCuInvalidValue;
    write_attr(n, attr, static_cast<const uint8_t*>(value));
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphKernelNodeCopyAttributes(void* dst, void* src) {
  return drv_call("cuGraphKernelNodeCopyAttributes", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* d = node_of(dst);
    GraphNodeRec* s = node_of(src);
    if (!d || !s || d->type != cudaGraphNodeTypeKernel || s->type != cudaGraphNodeTypeKernel) return kCuInvalidValue;
    d->work.attrs = s->work.attrs;
    d->work.cooperative = s->work.cooperative;
    d->work.cluster = s->work.cluster;
    return kCuOk;
  });
}

// ---- conditional nodes ------------------------------------------------------

extern "C" VGPU_EXPORT int vgpu_cuGraphConditionalHandleCreate(unsigned long long* h, void* g, void* ctx, unsigned def,
                                                                unsigned flags) {
#if CUDART_VERSION >= 12030
  return drv_call("cuGraphConditionalHandleCreate", [&](State&) {
    // Measured: a null context is CUDA_ERROR_INVALID_CONTEXT, not a bad value.
    using Fn = int (*)(void*, int*);
    static const Fn ctx_device = reinterpret_cast<Fn>(driver_symbol("vgpu_driver_ctx_device_v1"));
    int device = 0;
    if (!ctx || (ctx_device && ctx_device(ctx, &device) != 0)) return kCuInvalidContext;
    return to_cu(cudaGraphConditionalHandleCreate(reinterpret_cast<cudaGraphConditionalHandle*>(h),
                                                  static_cast<cudaGraph_t>(g), def, flags));
  });
#else
  (void)h; (void)g; (void)ctx; (void)def; (void)flags;
  return kCuNotSupported;
#endif
}

// ---- the generic node calls -------------------------------------------------
//
// cuGraphAddNode, cuGraphNodeGetParams, cuGraphNodeSetParams and
// cuGraphExecNodeSetParams take every node type through one structure; each is
// the per-type call on the member that type uses. The reserved words must be
// zero (measured: a nonzero one is CUDA_ERROR_INVALID_VALUE, as is a type
// outside the enumeration).

namespace {
using NodeParams = vcu::GraphNodeParams;

bool reserved_zero(const NodeParams* p) {
  return !p->reserved0[0] && !p->reserved0[1] && !p->reserved0[2] && !p->reserved2;
}
}  // namespace

static int add_generic(void** node, void* g, void* const* deps, size_t n, NodeParams* p) {
  if (!p || !node || !reserved_zero(p)) return kCuInvalidValue;
  switch (p->type) {
    case vcu::kNodeKernel: {
      const auto& k = p->kernel;
      return vgpu_cuGraphAddKernelNode_v2(node, g, deps, n, &k);
    }
    case vcu::kNodeMemcpy:
      if (p->memcpy.flags || p->memcpy.reserved) return kCuInvalidValue;
      return vgpu_cuGraphAddMemcpyNode(node, g, deps, n, &p->memcpy.copyParams, p->memcpy.copyCtx);
    case vcu::kNodeMemset: {
      const vcu::MemsetNodeParams m{p->memset.dst, p->memset.pitch, p->memset.value, p->memset.elementSize,
                                    p->memset.width, p->memset.height};
      return vgpu_cuGraphAddMemsetNode(node, g, deps, n, &m, p->memset.ctx);
    }
    case vcu::kNodeHost: {
      const vcu::HostNodeParams h{p->host.fn, p->host.userData};
      return vgpu_cuGraphAddHostNode(node, g, deps, n, &h);
    }
    case vcu::kNodeGraph: {
      if (p->graph.ownership == 1) {   // CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE
        return drv_call("cuGraphAddNode", [&](State&) {
          std::lock_guard<std::mutex> lock(g_graph_mu);
          GraphRec* parent = nullptr;
          std::vector<GraphNodeRec*> pred;
          if (!graph_and_deps(g, deps, n, &parent, &pred)) return kCuInvalidValue;
          const auto it = g_graphs.find(p->graph.graph);
          if (it == g_graphs.end() || it->second.get() == parent) return kCuInvalidValue;
          if (holds_conditional(*it->second)) return kCuNotSupported;
          // The parent takes the graph itself, handle and all: the handle stays
          // valid, read-only, and goes when the parent does.
          GraphRec* child = it->second.get();
          parent->children.push_back(std::move(it->second));
          g_graphs.erase(it);
          g_borrowed_graphs[child] = child;
          GraphNodeRec* made = parent->add(cudaGraphNodeTypeGraph, RecordedLaunch{}, pred);
          made->child = child;
          *node = made;
          return kCuOk;
        });
      }
      if (p->graph.ownership != 0) return kCuInvalidValue;
      return vgpu_cuGraphAddChildGraphNode(node, g, deps, n, p->graph.graph);
    }
    case vcu::kNodeEmpty:
      return vgpu_cuGraphAddEmptyNode(node, g, deps, n);
    case vcu::kNodeWaitEvent:
      return vgpu_cuGraphAddEventWaitNode(node, g, deps, n, p->eventWait.event);
    case vcu::kNodeEventRecord:
      return vgpu_cuGraphAddEventRecordNode(node, g, deps, n, p->eventRecord.event);
    case vcu::kNodeExtSemSignal:
      return vgpu_cuGraphAddExternalSemaphoresSignalNode(node, g, deps, n, &p->extSemSignal);
    case vcu::kNodeExtSemWait:
      return vgpu_cuGraphAddExternalSemaphoresWaitNode(node, g, deps, n, &p->extSemWait);
    case vcu::kNodeMemAlloc:
      return vgpu_cuGraphAddMemAllocNode(node, g, deps, n, &p->alloc);
    case vcu::kNodeMemFree:
      return vgpu_cuGraphAddMemFreeNode(node, g, deps, n, p->free.dptr);
    case vcu::kNodeBatchMemOp:
      return vgpu_cuGraphAddBatchMemOpNode(node, g, deps, n, &p->memOp);
    case vcu::kNodeConditional:
#if CUDART_VERSION >= 12030
      return drv_call("cuGraphAddNode", [&](State&) {
        // Measured: a null context is CUDA_ERROR_INVALID_CONTEXT here too.
        if (!p->conditional.ctx) return kCuInvalidContext;
        cudaConditionalNodeParams c{};
        c.handle = p->conditional.handle;
        c.type = static_cast<cudaGraphConditionalNodeType>(p->conditional.type);
        c.size = p->conditional.size;
        const cudaError_t rc = add_conditional_node(reinterpret_cast<cudaGraphNode_t*>(node), static_cast<cudaGraph_t>(g),
                                                    reinterpret_cast<const cudaGraphNode_t*>(deps), n, &c);
        if (rc == cudaSuccess) p->conditional.phGraph_out = reinterpret_cast<vcu::Graph*>(c.phGraph_out);
        return to_cu(rc);
      });
#else
      return kCuNotSupported;
#endif
    default:
      return kCuInvalidValue;
  }
}

extern "C" VGPU_EXPORT int vgpu_cuGraphAddNode(void** node, void* g, void* const* deps, size_t n, void* p) {
  return add_generic(node, g, deps, n, static_cast<NodeParams*>(p));
}
extern "C" VGPU_EXPORT int vgpu_cuGraphAddNode_v2(void** node, void* g, void* const* deps, const void* edge, size_t n,
                                                   void* p) {
  if (!no_edge_data(edge, n)) return kCuNotSupported;
  return add_generic(node, g, deps, n, static_cast<NodeParams*>(p));
}

extern "C" VGPU_EXPORT int vgpu_cuGraphNodeGetParams(void* node, void* out) {
  if (!out) return kCuInvalidValue;
  int type = 0;
  if (const int rc = vgpu_cuGraphNodeGetType(node, &type); rc != kCuOk) return rc;
  auto* p = static_cast<NodeParams*>(out);
  std::memset(p, 0, sizeof *p);
  p->type = type;
  switch (type) {
    case vcu::kNodeKernel: return vgpu_cuGraphKernelNodeGetParams_v2(node, &p->kernel);
    case vcu::kNodeMemcpy: {
      using Ctx = void* (*)(int);
      static const Ctx primary = reinterpret_cast<Ctx>(driver_symbol("vgpu_driver_primary_ctx_v1"));
      p->memcpy.copyCtx = primary ? primary(t_current_device) : nullptr;
      return vgpu_cuGraphMemcpyNodeGetParams(node, &p->memcpy.copyParams);
    }
    case vcu::kNodeMemset: {
      vcu::MemsetNodeParams m{};
      const int rc = vgpu_cuGraphMemsetNodeGetParams(node, &m);
      p->memset = {m.dst, m.pitch, m.value, m.elementSize, m.width, m.height, nullptr};
      return rc;
    }
    case vcu::kNodeHost: {
      vcu::HostNodeParams h{};
      const int rc = vgpu_cuGraphHostNodeGetParams(node, &h);
      p->host = {h.fn, h.userData, 0};
      return rc;
    }
    case vcu::kNodeGraph: {
      const int rc = vgpu_cuGraphChildGraphNodeGetGraph(node, &p->graph.graph);
      p->graph.ownership = -1;   // CU_GRAPH_CHILD_GRAPH_OWNERSHIP_INVALID, as measured
      return rc;
    }
    case vcu::kNodeEmpty: return kCuOk;
    case vcu::kNodeWaitEvent: return vgpu_cuGraphEventWaitNodeGetEvent(node, &p->eventWait.event);
    case vcu::kNodeEventRecord: return vgpu_cuGraphEventRecordNodeGetEvent(node, &p->eventRecord.event);
    case vcu::kNodeExtSemSignal: return vgpu_cuGraphExternalSemaphoresSignalNodeGetParams(node, &p->extSemSignal);
    case vcu::kNodeExtSemWait: return vgpu_cuGraphExternalSemaphoresWaitNodeGetParams(node, &p->extSemWait);
    case vcu::kNodeMemAlloc: return vgpu_cuGraphMemAllocNodeGetParams(node, &p->alloc);
    case vcu::kNodeMemFree: return vgpu_cuGraphMemFreeNodeGetParams(node, &p->free.dptr);
    case vcu::kNodeBatchMemOp: return vgpu_cuGraphBatchMemOpNodeGetParams(node, &p->memOp);
    case vcu::kNodeConditional:
#if CUDART_VERSION >= 12030
      return drv_call("cuGraphNodeGetParams", [&](State&) {
        std::lock_guard<std::mutex> lock(g_graph_mu);
        GraphNodeRec* n = node_of(node);
        if (!n) return kCuInvalidValue;
        p->conditional.handle = n->cond.handle;
        p->conditional.type = n->cond.kind;
        p->conditional.size = static_cast<unsigned>(n->cond.bodies.size());
        p->conditional.phGraph_out = reinterpret_cast<vcu::Graph*>(n->cond.body_handles.data());
        using Ctx = void* (*)(int);
        static const Ctx primary = reinterpret_cast<Ctx>(driver_symbol("vgpu_driver_primary_ctx_v1"));
        p->conditional.ctx = primary ? primary(t_current_device) : nullptr;
        return kCuOk;
      });
#else
      return kCuNotSupported;
#endif
    default: return kCuInvalidValue;
  }
}

// The type has to be the node's own: this changes a node's parameters, not what
// kind of node it is.
extern "C" VGPU_EXPORT int vgpu_cuGraphNodeSetParams(void* node, void* in) {
  if (!in) return kCuInvalidValue;
  auto* p = static_cast<NodeParams*>(in);
  int type = 0;
  if (const int rc = vgpu_cuGraphNodeGetType(node, &type); rc != kCuOk) return rc;
  if (p->type != type || !reserved_zero(p)) return kCuInvalidValue;
  switch (type) {
    case vcu::kNodeKernel: return vgpu_cuGraphKernelNodeSetParams_v2(node, &p->kernel);
    case vcu::kNodeMemcpy: return vgpu_cuGraphMemcpyNodeSetParams(node, &p->memcpy.copyParams);
    case vcu::kNodeMemset: {
      const vcu::MemsetNodeParams m{p->memset.dst, p->memset.pitch, p->memset.value, p->memset.elementSize,
                                    p->memset.width, p->memset.height};
      return vgpu_cuGraphMemsetNodeSetParams(node, &m);
    }
    case vcu::kNodeHost: {
      const vcu::HostNodeParams h{p->host.fn, p->host.userData};
      return vgpu_cuGraphHostNodeSetParams(node, &h);
    }
    case vcu::kNodeEmpty: return kCuOk;
    case vcu::kNodeWaitEvent: return vgpu_cuGraphEventWaitNodeSetEvent(node, p->eventWait.event);
    case vcu::kNodeEventRecord: return vgpu_cuGraphEventRecordNodeSetEvent(node, p->eventRecord.event);
    case vcu::kNodeExtSemSignal: return vgpu_cuGraphExternalSemaphoresSignalNodeSetParams(node, &p->extSemSignal);
    case vcu::kNodeExtSemWait: return vgpu_cuGraphExternalSemaphoresWaitNodeSetParams(node, &p->extSemWait);
    case vcu::kNodeBatchMemOp: return vgpu_cuGraphBatchMemOpNodeSetParams(node, &p->memOp);
    default: return kCuInvalidValue;   // child graph, allocation, free and conditional nodes have none to set
  }
}

extern "C" VGPU_EXPORT int vgpu_cuGraphExecNodeSetParams(void* exec, void* node, void* in) {
  if (!in) return kCuInvalidValue;
  auto* p = static_cast<NodeParams*>(in);
  int type = 0;
  if (const int rc = vgpu_cuGraphNodeGetType(node, &type); rc != kCuOk) return rc;
  if (p->type != type || !reserved_zero(p)) return kCuInvalidValue;
  switch (type) {
    case vcu::kNodeKernel: return vgpu_cuGraphExecKernelNodeSetParams_v2(exec, node, &p->kernel);
    case vcu::kNodeMemcpy: return vgpu_cuGraphExecMemcpyNodeSetParams(exec, node, &p->memcpy.copyParams, p->memcpy.copyCtx);
    case vcu::kNodeMemset: {
      const vcu::MemsetNodeParams m{p->memset.dst, p->memset.pitch, p->memset.value, p->memset.elementSize,
                                    p->memset.width, p->memset.height};
      return vgpu_cuGraphExecMemsetNodeSetParams(exec, node, &m, p->memset.ctx);
    }
    case vcu::kNodeHost: {
      const vcu::HostNodeParams h{p->host.fn, p->host.userData};
      return vgpu_cuGraphExecHostNodeSetParams(exec, node, &h);
    }
    case vcu::kNodeGraph: return vgpu_cuGraphExecChildGraphNodeSetParams(exec, node, p->graph.graph);
    case vcu::kNodeEmpty: return kCuOk;
    case vcu::kNodeWaitEvent: return vgpu_cuGraphExecEventWaitNodeSetEvent(exec, node, p->eventWait.event);
    case vcu::kNodeEventRecord: return vgpu_cuGraphExecEventRecordNodeSetEvent(exec, node, p->eventRecord.event);
    case vcu::kNodeExtSemSignal: return vgpu_cuGraphExecExternalSemaphoresSignalNodeSetParams(exec, node, &p->extSemSignal);
    case vcu::kNodeExtSemWait: return vgpu_cuGraphExecExternalSemaphoresWaitNodeSetParams(exec, node, &p->extSemWait);
    case vcu::kNodeBatchMemOp: return vgpu_cuGraphExecBatchMemOpNodeSetParams(exec, node, &p->memOp);
    default: return kCuInvalidValue;
  }
}

// ---- instantiating, launching and updating -----------------------------------

// Measured (RTX 3060, driver 596.36): cuGraphInstantiateWithFlags takes the
// auto-free, device-launch and node-priority flags; upload (2) belongs to
// cuGraphInstantiateWithParams, and anything else, is CUDA_ERROR_INVALID_VALUE.
extern "C" VGPU_EXPORT int vgpu_cuGraphInstantiateWithFlags(void** exec, void* g, unsigned long long flags) {
  return drv_call("cuGraphInstantiateWithFlags", [&](State&) {
    if (flags & ~(1ull | 4ull | 8ull)) return kCuInvalidValue;
    return to_cu(cudaGraphInstantiate(reinterpret_cast<cudaGraphExec_t*>(exec), static_cast<cudaGraph_t>(g), flags));
  });
}
// The original, pre-flags spelling (cuGraphInstantiate before CUDA 11.0): the
// node that made it fail and a log. Neither is ever set: a failure here is
// reported by its code, and measured, a success leaves both untouched.
extern "C" VGPU_EXPORT int vgpu_cuGraphInstantiate_v2(void** exec, void* g, void** errNode, char* log, size_t logSize) {
  const int rc = vgpu_cuGraphInstantiateWithFlags(exec, g, 0);
  if (rc != kCuOk) {
    if (errNode) *errNode = nullptr;
    if (log && logSize) log[0] = '\0';
  }
  return rc;
}
static_assert(sizeof(vcu::GraphInstantiateParams) == sizeof(cudaGraphInstantiateParams),
              "CUDA_GRAPH_INSTANTIATE_PARAMS and cudaGraphInstantiateParams differ");
extern "C" VGPU_EXPORT int vgpu_cuGraphInstantiateWithParams(void** exec, void* g, void* params) {
  return drv_call("cuGraphInstantiateWithParams", [&](State&) {
    return to_cu(cudaGraphInstantiateWithParams(reinterpret_cast<cudaGraphExec_t*>(exec), static_cast<cudaGraph_t>(g),
                                                static_cast<cudaGraphInstantiateParams*>(params)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExecDestroy(void* exec) {
  return drv_call("cuGraphExecDestroy", [&](State&) { return to_cu(cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec))); });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphLaunch(void* exec, void* stream) {
  return drv_call("cuGraphLaunch", [&](State&) {
    return to_cu(cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec), static_cast<cudaStream_t>(stream)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphUpload(void* exec, void* stream) {
  return drv_call("cuGraphUpload", [&](State&) {
    return to_cu(cudaGraphUpload(static_cast<cudaGraphExec_t>(exec), static_cast<cudaStream_t>(stream)));
  });
}

static_assert(sizeof(vcu::GraphExecUpdateResultInfo) == sizeof(cudaGraphExecUpdateResultInfo),
              "CUgraphExecUpdateResultInfo and cudaGraphExecUpdateResultInfo differ");
extern "C" VGPU_EXPORT int vgpu_cuGraphExecUpdate_v2(void* exec, void* g, void* info) {
  return drv_call("cuGraphExecUpdate", [&](State&) {
    return to_cu(cudaGraphExecUpdate(static_cast<cudaGraphExec_t>(exec), static_cast<cudaGraph_t>(g),
                                     static_cast<cudaGraphExecUpdateResultInfo*>(info)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExecUpdate(void* exec, void* g, void** errNode, int* result) {
  cudaGraphExecUpdateResultInfo info{};
  const int rc = vgpu_cuGraphExecUpdate_v2(exec, g, &info);
  if (errNode) *errNode = info.errorNode;
  if (result) *result = static_cast<int>(info.result);
  return rc;
}

namespace {
// The node in an instantiated graph that answers for `original`, checked to be
// of `type`. Caller holds g_graph_mu.
GraphNodeRec* exec_target(void* exec, void* original, cudaGraphNodeType type) {
  const auto it = g_graph_execs.find(exec);
  GraphNodeRec* n = node_of(original);
  if (it == g_graph_execs.end() || !n) return nullptr;
  GraphNodeRec* target = exec_twin(*it->second, n);
  return target && target->type == type ? target : nullptr;
}
}  // namespace

static int exec_set_kernel(void* exec, void* node, const void* p, bool v2) {
  if (!p) return kCuInvalidValue;
  const auto* k = static_cast<const vcu::KernelNodeParamsV3*>(p);
  RecordedLaunch w;
  const int rc = fill_drv_kernel_work(&w, k->func, v2 ? k->kern : nullptr, k->gridDimX, k->gridDimY, k->gridDimZ,
                                      k->blockDimX, k->blockDimY, k->blockDimZ, k->sharedMemBytes, k->kernelParams,
                                      k->extra);
  if (rc != kCuOk) return rc;
  return drv_call("cuGraphExecKernelNodeSetParams", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* target = exec_target(exec, node, cudaGraphNodeTypeKernel);
    if (!target) return kCuInvalidValue;
    w.attrs = target->work.attrs;
    target->work = std::move(w);
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExecKernelNodeSetParams(void* exec, void* node, const void* p) {
  return exec_set_kernel(exec, node, p, false);
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExecKernelNodeSetParams_v2(void* exec, void* node, const void* p) {
  return exec_set_kernel(exec, node, p, true);
}
// An update of a copy or fill node keeps its shape: both the node and the new
// parameters have to be one-dimensional, as the runtime's calls document.
extern "C" VGPU_EXPORT int vgpu_cuGraphExecMemcpyNodeSetParams(void* exec, void* node, const void* p, void* ctx) {
  (void)ctx;
  return drv_call("cuGraphExecMemcpyNodeSetParams", [&](State&) {
    RecordedLaunch w;
    if (const int rc = fill_drv_memcpy_work(&w, static_cast<const vcu::Memcpy3D*>(p)); rc != kCuOk) return rc;
    if (w.height > 1 || w.depth > 1) return kCuInvalidValue;
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* target = exec_target(exec, node, cudaGraphNodeTypeMemcpy);
    if (!target || target->work.height > 1 || target->work.depth > 1) return kCuInvalidValue;
    target->work = std::move(w);
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExecMemsetNodeSetParams(void* exec, void* node, const void* p, void* ctx) {
  (void)ctx;
  return drv_call("cuGraphExecMemsetNodeSetParams", [&](State&) {
    const auto* m = static_cast<const vcu::MemsetNodeParams*>(p);
    if (!m || m->height > 1 || m->width == 0) return kCuInvalidValue;
    RecordedLaunch w;
    if (const int rc = fill_drv_memset_work(&w, m); rc != kCuOk) return rc;
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* target = exec_target(exec, node, cudaGraphNodeTypeMemset);
    if (!target || target->work.height > 1) return kCuInvalidValue;
    target->work = std::move(w);
    return kCuOk;
  });
}
// Measured: a null function is refused here, though it is accepted when the
// node is added.
extern "C" VGPU_EXPORT int vgpu_cuGraphExecHostNodeSetParams(void* exec, void* node, const void* p) {
  return drv_call("cuGraphExecHostNodeSetParams", [&](State&) {
    const auto* h = static_cast<const vcu::HostNodeParams*>(p);
    if (!h || !h->fn) return kCuInvalidValue;
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* target = exec_target(exec, node, cudaGraphNodeTypeHost);
    if (!target) return kCuInvalidValue;
    target->work.host_fn = h->fn;
    target->work.host_user = h->userData;
    target->work.host_op = nullptr;
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExecChildGraphNodeSetParams(void* exec, void* node, void* child) {
  return drv_call("cuGraphExecChildGraphNodeSetParams", [&](State&) {
    return to_cu(cudaGraphExecChildGraphNodeSetParams(static_cast<cudaGraphExec_t>(exec), static_cast<cudaGraphNode_t>(node),
                                                      static_cast<cudaGraph_t>(child)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExecEventRecordNodeSetEvent(void* exec, void* node, void* ev) {
  return drv_call("cuGraphExecEventRecordNodeSetEvent", [&](State&) {
    return to_cu(cudaGraphExecEventRecordNodeSetEvent(static_cast<cudaGraphExec_t>(exec), static_cast<cudaGraphNode_t>(node),
                                                      static_cast<cudaEvent_t>(ev)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExecEventWaitNodeSetEvent(void* exec, void* node, void* ev) {
  return drv_call("cuGraphExecEventWaitNodeSetEvent", [&](State&) {
    return to_cu(cudaGraphExecEventWaitNodeSetEvent(static_cast<cudaGraphExec_t>(exec), static_cast<cudaGraphNode_t>(node),
                                                    static_cast<cudaEvent_t>(ev)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphNodeSetEnabled(void* exec, void* node, unsigned enabled) {
  return drv_call("cuGraphNodeSetEnabled", [&](State&) {
    return to_cu(cudaGraphNodeSetEnabled(static_cast<cudaGraphExec_t>(exec), static_cast<cudaGraphNode_t>(node), enabled));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphNodeGetEnabled(void* exec, void* node, unsigned* enabled) {
  return drv_call("cuGraphNodeGetEnabled", [&](State&) {
    return to_cu(cudaGraphNodeGetEnabled(static_cast<cudaGraphExec_t>(exec), static_cast<cudaGraphNode_t>(node), enabled));
  });
}

// ---- ids --------------------------------------------------------------------

extern "C" VGPU_EXPORT int vgpu_cuGraphExecGetFlags(void* exec, unsigned long long* flags) {
  return drv_call("cuGraphExecGetFlags", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    const auto it = g_exec_flags.find(exec);
    if (it == g_exec_flags.end() || !flags) return kCuInvalidValue;
    *flags = it->second;
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphExecGetId(void* exec, unsigned* id) {
  return drv_call("cuGraphExecGetId", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    const auto it = g_exec_ids.find(exec);
    if (it == g_exec_ids.end() || !id) return kCuInvalidValue;
    *id = it->second;
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphGetId(void* g, unsigned* id) {
  return drv_call("cuGraphGetId", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    const GraphRec* graph = graph_from(static_cast<cudaGraph_t>(g));
    if (!graph || !id) return kCuInvalidValue;
    *id = graph->id;
    return kCuOk;
  });
}
// A node's id within its graph is its position (measured: the first node is 0),
// and its tools id puts the graph's id above it.
namespace {
bool node_position(void* node, const GraphRec** owner, unsigned* index) {   // caller holds g_graph_mu
  GraphNodeRec* n = node_of(node);
  if (!n) return false;
  *owner = owner_of(n);
  for (size_t i = 0; i < (*owner)->nodes.size(); ++i)
    if ((*owner)->nodes[i].get() == n) {
      *index = static_cast<unsigned>(i);
      return true;
    }
  return false;
}
}  // namespace
extern "C" VGPU_EXPORT int vgpu_cuGraphNodeGetLocalId(void* node, unsigned* id) {
  return drv_call("cuGraphNodeGetLocalId", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    const GraphRec* owner = nullptr;
    unsigned index = 0;
    if (!id || !node_position(node, &owner, &index)) return kCuInvalidValue;
    *id = index;
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphNodeGetToolsId(void* node, unsigned long long* id) {
  return drv_call("cuGraphNodeGetToolsId", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    const GraphRec* owner = nullptr;
    unsigned index = 0;
    if (!id || !node_position(node, &owner, &index)) return kCuInvalidValue;
    *id = (static_cast<unsigned long long>(owner->id) << 32) | index;
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphNodeGetContainingGraph(void* node, void** g) {
  return drv_call("cuGraphNodeGetContainingGraph", [&](State&) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphNodeRec* n = node_of(node);
    if (!n || !g) return kCuInvalidValue;
    *g = owner_of(n);
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphDebugDotPrint(void* g, const char* path, unsigned flags) {
  return drv_call("cuGraphDebugDotPrint", [&](State&) {
    return to_cu(cudaGraphDebugDotPrint(static_cast<cudaGraph_t>(g), path, flags));
  });
}

// ---- user objects -----------------------------------------------------------
//
// Measured on an RTX 3060: no destructor, no references, a flag other than
// "no destructor sync" (1), and retaining zero references are each
// CUDA_ERROR_INVALID_VALUE. A graph retaining without the move flag takes new
// references; with it, the program's references become the graph's.

namespace {
int uo_create(void** out, void* ptr, void (*destroy)(void*), unsigned count, unsigned flags) {
  if (!out || !destroy || count == 0 || (flags & ~1u)) return kCuInvalidValue;
  auto obj = std::make_unique<UserObj>();
  obj->ptr = ptr;
  obj->destroy = destroy;
  obj->refs = count;
  void* handle = obj.get();
  std::lock_guard<std::mutex> lock(g_uo_mu);
  g_user_objects[handle] = std::move(obj);
  *out = handle;
  return kCuOk;
}
int uo_graph_retain(void* graph, void* obj, unsigned count, unsigned flags) {
  if (!count || (flags & ~1u)) return kCuInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(static_cast<cudaGraph_t>(graph));
  if (!g) return kCuInvalidValue;
  {
    std::lock_guard<std::mutex> uo(g_uo_mu);
    if (!g_user_objects.count(obj)) return kCuInvalidValue;
    if (!(flags & 1u)) g_user_objects[obj]->refs += count;
  }
  for (auto& [o, c] : g->user_refs)
    if (o == obj) {
      c += count;
      return kCuOk;
    }
  g->user_refs.emplace_back(obj, count);
  return kCuOk;
}
int uo_graph_release(void* graph, void* obj, unsigned count) {
  if (!count) return kCuInvalidValue;
  std::lock_guard<std::mutex> lock(g_graph_mu);
  GraphRec* g = graph_from(static_cast<cudaGraph_t>(graph));
  if (!g) return kCuInvalidValue;
  for (auto it = g->user_refs.begin(); it != g->user_refs.end(); ++it) {
    if (it->first != obj) continue;
    const unsigned drop = std::min(count, it->second);
    it->second -= drop;
    if (it->second == 0) g->user_refs.erase(it);
    uo_release(obj, drop);
    return kCuOk;
  }
  return kCuInvalidValue;
}
}  // namespace

extern "C" VGPU_EXPORT int vgpu_cuUserObjectCreate(void** obj, void* ptr, void (*destroy)(void*), unsigned count,
                                                    unsigned flags) {
  return drv_call("cuUserObjectCreate", [&](State&) { return uo_create(obj, ptr, destroy, count, flags); });
}
extern "C" VGPU_EXPORT int vgpu_cuUserObjectRetain(void* obj, unsigned count) {
  return drv_call("cuUserObjectRetain", [&](State&) {
    if (!count) return kCuInvalidValue;
    std::lock_guard<std::mutex> lock(g_uo_mu);
    const auto it = g_user_objects.find(obj);
    if (it == g_user_objects.end()) return kCuInvalidValue;
    it->second->refs += count;
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuUserObjectRelease(void* obj, unsigned count) {
  DrainUserObjects drain;
  return drv_call("cuUserObjectRelease", [&](State&) {
    if (!count) return kCuInvalidValue;
    return uo_release(obj, count) ? kCuOk : kCuInvalidValue;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphRetainUserObject(void* g, void* obj, unsigned count, unsigned flags) {
  return drv_call("cuGraphRetainUserObject", [&](State&) { return uo_graph_retain(g, obj, count, flags); });
}
extern "C" VGPU_EXPORT int vgpu_cuGraphReleaseUserObject(void* g, void* obj, unsigned count) {
  DrainUserObjects drain;
  return drv_call("cuGraphReleaseUserObject", [&](State&) { return uo_graph_release(g, obj, count); });
}

// The runtime's own spellings of the same five.
VGPU_EXPORT cudaError_t cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy,
                                             unsigned int initialRefcount, unsigned int flags) {
  return static_cast<cudaError_t>(uo_create(reinterpret_cast<void**>(object_out), ptr, destroy, initialRefcount, flags));
}
VGPU_EXPORT cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) {
  return static_cast<cudaError_t>(vgpu_cuUserObjectRetain(object, count));
}
VGPU_EXPORT cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) {
  DrainUserObjects drain;
  if (!count) return cudaErrorInvalidValue;
  return uo_release(object, count) ? cudaSuccess : cudaErrorInvalidValue;
}
VGPU_EXPORT cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count,
                                                  unsigned int flags) {
  return static_cast<cudaError_t>(uo_graph_retain(graph, object, count, flags));
}
VGPU_EXPORT cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) {
  DrainUserObjects drain;
  return static_cast<cudaError_t>(uo_graph_release(graph, object, count));
}

// ---- stream capture ---------------------------------------------------------

namespace {
thread_local int t_capture_mode = 0;   // cudaStreamCaptureModeGlobal
}

// A thread's capture mode: what it exchanges. Measured: it starts as global (0),
// each call returns the previous mode, and a value outside 0..2 is accepted and
// changes nothing.
VGPU_EXPORT cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
  if (!mode) return cudaErrorInvalidValue;
  const int want = static_cast<int>(*mode);
  *mode = static_cast<cudaStreamCaptureMode>(t_capture_mode);
  if (want >= 0 && want <= 2) t_capture_mode = want;
  return cudaSuccess;
}
extern "C" VGPU_EXPORT int vgpu_cuThreadExchangeStreamCaptureMode(int* mode) {
  cudaStreamCaptureMode m = static_cast<cudaStreamCaptureMode>(mode ? *mode : 0);
  if (!mode) return kCuInvalidValue;
  cudaThreadExchangeStreamCaptureMode(&m);
  *mode = static_cast<int>(m);
  return kCuOk;
}

extern "C" VGPU_EXPORT int vgpu_cuStreamBeginCapture_v2(void* stream, int mode) {
  return drv_call("cuStreamBeginCapture", [&](State&) {
    if (mode < 0 || mode > 2) return kCuInvalidValue;
    return to_cu(cudaStreamBeginCapture(static_cast<cudaStream_t>(stream), static_cast<cudaStreamCaptureMode>(mode)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuStreamBeginCaptureToGraph(void* stream, void* g, void* const* deps, const void* edge,
                                                             size_t n, int mode) {
#if CUDART_VERSION >= 12030
  return drv_call("cuStreamBeginCaptureToGraph", [&](State&) {
    if (mode < 0 || mode > 2) return kCuInvalidValue;
    return to_cu(cudaStreamBeginCaptureToGraph(static_cast<cudaStream_t>(stream), static_cast<cudaGraph_t>(g),
                                               reinterpret_cast<const cudaGraphNode_t*>(deps),
                                               static_cast<const cudaGraphEdgeData*>(edge), n,
                                               static_cast<cudaStreamCaptureMode>(mode)));
  });
#else
  (void)stream; (void)g; (void)deps; (void)edge; (void)n; (void)mode;
  return kCuNotSupported;
#endif
}
extern "C" VGPU_EXPORT int vgpu_cuStreamEndCapture(void* stream, void** g) {
  return drv_call("cuStreamEndCapture", [&](State&) {
    return to_cu(cudaStreamEndCapture(static_cast<cudaStream_t>(stream), reinterpret_cast<cudaGraph_t*>(g)));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuStreamGetCaptureInfo_v3(void* stream, int* status, unsigned long long* id, void** g,
                                                        const void* const** deps, const void** edge, size_t* n) {
  return drv_call("cuStreamGetCaptureInfo", [&](State&) {
    if (!status) return kCuInvalidValue;
    cudaStreamCaptureStatus st{};
    size_t count = 0;
    const cudaGraphNode_t* d = nullptr;
    const cudaError_t rc = capture_info(static_cast<cudaStream_t>(stream), &st, id, reinterpret_cast<cudaGraph_t*>(g), &d, &count);
    if (rc != cudaSuccess) return to_cu(rc);
    if (status) *status = static_cast<int>(st);
    if (st == cudaStreamCaptureStatusNone) return kCuOk;   // measured: nothing else is written
    if (deps) *deps = reinterpret_cast<const void* const*>(d);
    if (n) *n = count;
    if (edge) {
      static thread_local std::vector<vcu::GraphEdgeData> edges;
      edges.assign(count, vcu::GraphEdgeData{});
      *edge = count ? edges.data() : nullptr;
    }
    return kCuOk;
  });
}
extern "C" VGPU_EXPORT int vgpu_cuStreamUpdateCaptureDependencies(void* stream, void** deps, size_t n, unsigned flags) {
  return vgpu_cuStreamUpdateCaptureDependencies_v2(stream, deps, nullptr, n, flags);
}
extern "C" VGPU_EXPORT int vgpu_cuStreamUpdateCaptureDependencies_v2(void* stream, void** deps, const void* edge, size_t n,
                                                                      unsigned flags) {
  return drv_call("cuStreamUpdateCaptureDependencies", [&](State&) {
    if (!no_edge_data(edge, n)) return kCuNotSupported;
    return to_cu(update_capture_deps(static_cast<cudaStream_t>(stream), reinterpret_cast<cudaGraphNode_t*>(deps), n, flags));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuStreamIsCapturing(void* stream, int* status) {
  return drv_call("cuStreamIsCapturing", [&](State&) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    const cudaError_t rc = cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &st);
    if (status) *status = static_cast<int>(st);
    return to_cu(rc);
  });
}

// ---- graph memory -----------------------------------------------------------

extern "C" VGPU_EXPORT int vgpu_cuDeviceGraphMemTrim(int device) {
  return drv_call("cuDeviceGraphMemTrim", [&](State&) { return to_cu(cudaDeviceGraphMemTrim(device)); });
}
extern "C" VGPU_EXPORT int vgpu_cuDeviceGetGraphMemAttribute(int device, int attr, void* value) {
  return drv_call("cuDeviceGetGraphMemAttribute", [&](State&) {
    return to_cu(cudaDeviceGetGraphMemAttribute(device, static_cast<cudaGraphMemAttributeType>(attr), value));
  });
}
extern "C" VGPU_EXPORT int vgpu_cuDeviceSetGraphMemAttribute(int device, int attr, void* value) {
  return drv_call("cuDeviceSetGraphMemAttribute", [&](State&) {
    return to_cu(cudaDeviceSetGraphMemAttribute(device, static_cast<cudaGraphMemAttributeType>(attr), value));
  });
}

// ---- external semaphores, run ------------------------------------------------

static cudaError_t run_ext_sem_node(const RecordedLaunch& rl, bool signal) {
  (void)rl;
  (void)signal;
  return cudaSuccess;
}

// ---- what libcuda needs of a capture beyond adding nodes ---------------------

// An operation a capture cannot hold: the capture is invalidated and says why.
extern "C" VGPU_EXPORT int vgpu_cuCaptureRefuse(void* stream, const char* what) {
  return capture_refuse(static_cast<cudaStream_t>(stream), what) ? 0 : 1;
}

// A driver event recorded on a stream: where the capture has got to, which a
// wait on the event depends on. `external` makes it a node, which is recorded
// each time the graph runs. 0 when the stream was capturing (and the event now
// stands for the capture), 1 when it was not.
extern "C" VGPU_EXPORT int vgpu_cuCaptureEventRecord(void* stream, void* event, int external, unsigned long long* id,
                                                      std::vector<void*>* deps) {
  KeepLastError keep;
  if (external) return capture_event_node(static_cast<cudaStream_t>(stream), true, static_cast<cudaEvent_t>(event), id, deps) ? 0 : 1;
  return capture_position(static_cast<cudaStream_t>(stream), id, deps) ? 0 : 1;
}

// A wait on an event: inside a capture it joins streams (the waiting stream's
// next operation depends on where the event's stream had got to, and a stream
// not yet capturing joins). Returns the CUresult.
extern "C" VGPU_EXPORT int vgpu_cuCaptureEventWait(void* stream, void* event, int captured, unsigned long long id,
                                                    const std::vector<void*>* deps, unsigned flags) {
  KeepLastError keep;
  if (flags & ~1u) return kCuInvalidValue;   // CU_EVENT_WAIT_EXTERNAL
  if (flags & 1u) {
    capture_event_node(static_cast<cudaStream_t>(stream), false, static_cast<cudaEvent_t>(event), nullptr, nullptr);
    return kCuOk;
  }
  return to_cu(capture_wait(static_cast<cudaStream_t>(stream), captured != 0, id, deps ? *deps : std::vector<void*>{}));
}

extern "C" VGPU_EXPORT int vgpu_cuCaptureInvalidateId(unsigned long long id, const char* what) {
  capture_invalidate_id(id, what);
  return 0;
}
