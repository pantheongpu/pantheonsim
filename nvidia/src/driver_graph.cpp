// libvgpucuda's graphs and stream capture: the driver API's cuStreamBeginCapture / cuStreamEndCapture,
// cuGraph*, cuStream*Value* and cuUserObject*.
//
// The graph engine is the runtime's (libvgpucudart: capture, the graph structure, instantiation and
// launch), and the two libraries share one simulated machine, so each call here is a call to the runtime
// by name with the driver's types turned into the runtime's -- CUgraph, CUgraphNode, CUgraphExec and
// CUstream are the same handles in both (as they are in NVIDIA's libraries). A program that never
// loaded libcudart gets the one beside this library the first time it captures.
//
// What differs between the two APIs:
//   - events: the driver keeps its own, so an event node names the runtime event that stands for the
//     driver's (runtime_event_for);
//   - kernel, memcpy, memset and batch memory operation nodes hold the driver's own parameters
//     (CUfunction, CUDA_MEMCPY3D, ...), which the runtime's nodes cannot: the node is a closure over a
//     copy of them (the runtime's vgpu_graph_add_closure_node_v1), kept here by node for the Get calls.
//     Nodes made by a capture are closures too, without parameters to give back, and so are the nodes
//     of a graph the runtime made from cudaGraph* calls;
//   - errors: CUresult and cudaError_t share their numbers for everything a graph can return.
//
// Written from NVIDIA's documentation; checked against the card by nvidia/tests/e2e/graph_driver.cu.
#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "driver_graph.hpp"
#include "kept_args.hpp"
#include "vgpu/profiling.hpp"
#include "vgpu_cuda_graph.h"

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

extern "C" {
// Exports of this library that the nodes run (declared by vgpu_cuda.h only in part).
CUresult cuMemcpyDtoH_v2(void* dst, CUdeviceptr src, size_t n);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void* src, size_t n);
CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D* p);
CUresult cuMemsetD8_v2(CUdeviceptr d, unsigned char v, size_t n);
CUresult cuMemsetD16_v2(CUdeviceptr d, unsigned short v, size_t n);
CUresult cuMemsetD32_v2(CUdeviceptr d, unsigned int v, size_t n);
CUresult cuMemsetD2D8_v2(CUdeviceptr d, size_t pitch, unsigned char v, size_t w, size_t h);
CUresult cuMemsetD2D16_v2(CUdeviceptr d, size_t pitch, unsigned short v, size_t w, size_t h);
CUresult cuMemsetD2D32_v2(CUdeviceptr d, size_t pitch, unsigned int v, size_t w, size_t h);
}

namespace {
// Reports one driver call to a profiler, as driver_api.cpp's traced() does (a profiler is told of the
// entry points a program calls itself; see vgpu/profiling.hpp). Only the calls that driver_api.cpp used
// to define and report are wrapped here.
template <class Body, class... A>
CUresult traced(const char* name, Body body, A... a) {
  const vgpu_traced::KeptArgs<A...> kept(a...);   // in this frame for the whole call: read again at its exit
  if (vgpu::profiling::enabled() || vgpu::profiling::hooked())
    vgpu::profiling::note_args(kept.argv(), kept.count(), kept.sizes());
  vgpu::profiling::ApiCall call(name, vgpu::profiling::Domain::Driver);
  const CUresult rc = body(a...);
  call.set_result(static_cast<int32_t>(rc));
  return rc;
}
}  // namespace

namespace {

constexpr CUresult kNotSupported = CUDA_ERROR_NOT_SUPPORTED;
constexpr CUresult kInvalidValue = CUDA_ERROR_INVALID_VALUE;
constexpr CUresult kInvalidHandle = CUDA_ERROR_INVALID_HANDLE;

/* ---- the runtime ------------------------------------------------------------------------------ */

void* rt_symbol(const char* name) {
  if (void* p = dlsym(RTLD_DEFAULT, name)) return p;
  static void* lib = [] {
    Dl_info here{};
    std::string dir;
    if (dladdr(reinterpret_cast<void*>(&rt_symbol), &here) && here.dli_fname) {
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

template <class F>
F rt(const char* name) {
  return reinterpret_cast<F>(rt_symbol(name));
}

// A runtime function by name.
#define RT(NAME, SIG) rt<SIG>(NAME)

CUresult res(int rc) { return static_cast<CUresult>(rc); }

// Forwards to a runtime function, whose result is the driver's.
#define FORWARD(NAME, SIG, ...)                \
  do {                                         \
    const auto f_ = rt<SIG>(NAME);             \
    if (!f_) return kNotSupported;             \
    return res(f_(__VA_ARGS__));               \
  } while (0)

int runtime_version() {
  static const int v = [] {
    int x = 0;
    if (auto f = rt<int (*)(int*)>("cudaRuntimeGetVersion")) f(&x);
    return x;
  }();
  return v;
}

using Node = void*;
using Graph = void*;
using Exec = void*;

// The calls whose runtime signatures grew an edge-data argument (12.3) and then lost the old spelling
// (13.0): each is called with the newest form and adapts to the one this runtime has.
int rt_get_edges(Graph g, CUgraphNode* from_n, CUgraphNode* to_n, CUgraphEdgeData* edges, size_t* n) {
  Node* from = reinterpret_cast<Node*>(from_n);
  Node* to = reinterpret_cast<Node*>(to_n);
  using New = int (*)(Graph, Node*, Node*, CUgraphEdgeData*, size_t*);
  using Old = int (*)(Graph, Node*, Node*, size_t*);
  if (runtime_version() >= 13000) {
    if (auto f = rt<New>("cudaGraphGetEdges")) return f(g, from, to, edges, n);
  } else {
    if (auto f = rt<New>("cudaGraphGetEdges_v2")) return f(g, from, to, edges, n);
    if (auto f = rt<Old>("cudaGraphGetEdges")) return f(g, from, to, n);
  }
  return kNotSupported;
}

int rt_node_edges(const char* base, Node node, CUgraphNode* list_n, CUgraphEdgeData* edges, size_t* n) {
  Node* list = reinterpret_cast<Node*>(list_n);
  using New = int (*)(Node, Node*, CUgraphEdgeData*, size_t*);
  using Old = int (*)(Node, Node*, size_t*);
  if (runtime_version() >= 13000) {
    if (auto f = rt<New>(base)) return f(node, list, edges, n);
  } else {
    if (auto f = rt<New>((std::string(base) + "_v2").c_str())) return f(node, list, edges, n);
    if (auto f = rt<Old>(base)) return f(node, list, n);
  }
  return kNotSupported;
}

int rt_edit_dependencies(const char* base, Graph g, const CUgraphNode* from_n, const CUgraphNode* to_n,
                         const CUgraphEdgeData* edges, size_t n) {
  const Node* from = reinterpret_cast<const Node*>(from_n);
  const Node* to = reinterpret_cast<const Node*>(to_n);
  using New = int (*)(Graph, const Node*, const Node*, const CUgraphEdgeData*, size_t);
  using Old = int (*)(Graph, const Node*, const Node*, size_t);
  if (runtime_version() >= 13000) {
    if (auto f = rt<New>(base)) return f(g, from, to, edges, n);
  } else {
    if (auto f = rt<New>((std::string(base) + "_v2").c_str())) return f(g, from, to, edges, n);
    if (auto f = rt<Old>(base)) return f(g, from, to, n);
  }
  return kNotSupported;
}

/* ---- parameters of the nodes the driver describes ---------------------------------------------- */

enum class Kind { Kernel, Memcpy, Memset, BatchMemOp };

// A copy of what a kernel, copy, fill or batch memory operation node was given. Immutable once made:
// the node's closure and the table share it, and a change makes a new one (an instantiated graph
// must keep the parameters it was instantiated with).
struct NodeParams {
  Kind kind = Kind::Kernel;
  // kernel
  CUfunction func = nullptr;
  CUkernel kern = nullptr;
  unsigned grid[3] = {1, 1, 1}, block[3] = {1, 1, 1}, shared = 0;
  std::vector<std::vector<uint8_t>> args;
  std::vector<void*> arg_ptrs;      // one per parameter, into args: what GetParams gives back as kernelParams
  void** extra = nullptr;
  CUcontext ctx = nullptr;
  // memcpy
  CUDA_MEMCPY3D copy{};
  // memset
  CUDA_MEMSET_NODE_PARAMS_v2 set{};
  // batch memory operations
  std::vector<CUstreamBatchMemOpParams> ops;
  unsigned flags = 0;
};

std::mutex g_mu;
std::map<Node, std::shared_ptr<const NodeParams>> g_params;   // by runtime node
std::map<Node, CUevent> g_event_nodes;                         // an event node's driver event

/* ---- memory operations ------------------------------------------------------------------------ */

// Waits for a value in memory another stream, thread or device writes. Everything on a stream is
// synchronous here, so a condition not already true is waited for by polling, up to
// VGPU_STREAM_WAIT_TIMEOUT_MS (default 60000), then CUDA_ERROR_LAUNCH_TIMEOUT.
bool satisfied(unsigned long long have, unsigned long long want, unsigned flags, bool wide) {
  const unsigned long long mask = wide ? ~0ull : 0xffffffffull;
  have &= mask, want &= mask;
  switch (flags & 3u) {
    case 0: {   // CU_STREAM_WAIT_VALUE_GEQ: a wrap-around comparison
      if (wide) return static_cast<long long>(have - want) >= 0;
      return static_cast<int>(static_cast<unsigned>(have) - static_cast<unsigned>(want)) >= 0;
    }
    case 1: return have == want;
    case 2: return (have & want) != 0;
    default: return (~(have | want) & mask) != 0;   // CU_STREAM_WAIT_VALUE_NOR
  }
}

CUresult read_word(CUdeviceptr addr, bool wide, unsigned long long* out) {
  unsigned long long v = 0;
  const CUresult r = cuMemcpyDtoH_v2(&v, addr, wide ? 8 : 4);
  *out = v;
  return r;
}

CUresult wait_value(CUdeviceptr addr, unsigned long long value, unsigned flags, bool wide) {
  using namespace std::chrono;
  static const long limit_ms = [] {
    const char* e = std::getenv("VGPU_STREAM_WAIT_TIMEOUT_MS");
    return e && e[0] ? std::atol(e) : 60000L;
  }();
  const auto deadline = steady_clock::now() + milliseconds(limit_ms);
  for (;;) {
    unsigned long long have = 0;
    if (const CUresult r = read_word(addr, wide, &have)) return r;
    if (satisfied(have, value, flags, wide)) return CUDA_SUCCESS;
    if (steady_clock::now() > deadline) return CUDA_ERROR_LAUNCH_TIMEOUT;
    std::this_thread::sleep_for(microseconds(50));
  }
}

CUresult write_value(CUdeviceptr addr, unsigned long long value, bool wide) {
  return cuMemcpyHtoD_v2(addr, &value, wide ? 8 : 4);
}

CUresult run_one_mem_op(const CUstreamBatchMemOpParams& op) {
  switch (op.operation) {
    case CU_STREAM_MEM_OP_WAIT_VALUE_32: return wait_value(op.waitValue.address, op.waitValue.value, op.waitValue.flags, false);
    case CU_STREAM_MEM_OP_WAIT_VALUE_64: return wait_value(op.waitValue.address, op.waitValue.value64, op.waitValue.flags, true);
    case CU_STREAM_MEM_OP_WRITE_VALUE_32: return write_value(op.writeValue.address, op.writeValue.value, false);
    case CU_STREAM_MEM_OP_WRITE_VALUE_64: return write_value(op.writeValue.address, op.writeValue.value64, true);
    case CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES:
    case CU_STREAM_MEM_OP_BARRIER: return CUDA_SUCCESS;   // memory is coherent here
    default: return kInvalidValue;
  }
}

CUresult run_mem_ops(const std::vector<CUstreamBatchMemOpParams>& ops) {
  for (const auto& op : ops)
    if (const CUresult r = run_one_mem_op(op)) return r;
  return CUDA_SUCCESS;
}

// Whether an operation is one this implementation knows, and has what it needs.
CUresult check_mem_op(const CUstreamBatchMemOpParams& op, bool allow_barrier) {
  switch (op.operation) {
    case CU_STREAM_MEM_OP_WAIT_VALUE_32:
    case CU_STREAM_MEM_OP_WAIT_VALUE_64:
      return op.waitValue.address ? CUDA_SUCCESS : kInvalidValue;
    case CU_STREAM_MEM_OP_WRITE_VALUE_32:
    case CU_STREAM_MEM_OP_WRITE_VALUE_64:
      return op.writeValue.address ? CUDA_SUCCESS : kInvalidValue;
    case CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES: return CUDA_SUCCESS;
    case CU_STREAM_MEM_OP_BARRIER: return allow_barrier ? CUDA_SUCCESS : kInvalidValue;
    default: return kInvalidValue;
  }
}

bool capturing(CUstream st) {
  using Fn = int (*)(void*, int*);
  static const Fn fn = rt<Fn>("cudaStreamIsCapturing");
  int status = 0;
  return st && fn && fn(st, &status) == 0 && status == 1;
}

// A stream-ordered operation on a capturing stream becomes a node whose launch is `op`.
bool record_node(CUstream st, int type, const std::function<void()>& op) {
  if (!capturing(st)) return false;
  using Fn = bool (*)(void*, int, const std::function<void()>*);
  static const Fn fn = rt<Fn>("vgpu_record_driver_node_v1");
  return fn && fn(st, type, &op);
}

CUresult stream_mem_op(CUstream st, std::vector<CUstreamBatchMemOpParams> ops) {
  if (record_node(st, 0 /* a kernel node, as the runtime's API shows it */, [ops] { run_mem_ops(ops); }))
    return CUDA_SUCCESS;
  return run_mem_ops(ops);
}

}  // namespace

/* ======================================================================================== */
/* stream memory operations                                                                  */
/* ======================================================================================== */

#define MEMOP_WRITE(NAME, WIDE, TY)                                                                              \
  VGPU_EXPORT CUresult NAME(CUstream st, CUdeviceptr addr, TY value, unsigned int flags) {                       \
    if (flags & ~1u) return kInvalidValue;   /* CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER */                       \
    if (!addr) return kInvalidValue;                                                                            \
    CUstreamBatchMemOpParams p{};                                                                               \
    p.operation = WIDE ? CU_STREAM_MEM_OP_WRITE_VALUE_64 : CU_STREAM_MEM_OP_WRITE_VALUE_32;                     \
    p.writeValue.address = addr;                                                                                \
    if (WIDE) p.writeValue.value64 = value; else p.writeValue.value = static_cast<cuuint32_t>(value);           \
    p.writeValue.flags = flags;                                                                                 \
    return stream_mem_op(st, {p});                                                                              \
  }
#define MEMOP_WAIT(NAME, WIDE, TY)                                                                               \
  VGPU_EXPORT CUresult NAME(CUstream st, CUdeviceptr addr, TY value, unsigned int flags) {                       \
    if ((flags & ~((1u << 30) | 3u)) || !addr) return kInvalidValue;                                            \
    CUstreamBatchMemOpParams p{};                                                                               \
    p.operation = WIDE ? CU_STREAM_MEM_OP_WAIT_VALUE_64 : CU_STREAM_MEM_OP_WAIT_VALUE_32;                       \
    p.waitValue.address = addr;                                                                                 \
    if (WIDE) p.waitValue.value64 = value; else p.waitValue.value = static_cast<cuuint32_t>(value);             \
    p.waitValue.flags = flags;                                                                                  \
    return stream_mem_op(st, {p});                                                                              \
  }

static CUresult cuStreamWriteValue32_impl(CUstream st, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
  if (flags & ~1u) return kInvalidValue;   /* CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER */
  if (!addr) return kInvalidValue;
  CUstreamBatchMemOpParams p{};
  p.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
  p.writeValue.address = addr;
  p.writeValue.value = value;
  p.writeValue.flags = flags;
  return stream_mem_op(st, {p});
}
VGPU_EXPORT CUresult cuStreamWriteValue32(CUstream st, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
  return traced("cuStreamWriteValue32", cuStreamWriteValue32_impl, st, addr, value, flags);
}
MEMOP_WRITE(cuStreamWriteValue32_v2, false, cuuint32_t)
MEMOP_WRITE(cuStreamWriteValue64, true, cuuint64_t)
MEMOP_WRITE(cuStreamWriteValue64_v2, true, cuuint64_t)
MEMOP_WAIT(cuStreamWaitValue32, false, cuuint32_t)
MEMOP_WAIT(cuStreamWaitValue32_v2, false, cuuint32_t)
MEMOP_WAIT(cuStreamWaitValue64, true, cuuint64_t)
MEMOP_WAIT(cuStreamWaitValue64_v2, true, cuuint64_t)

namespace {
CUresult batch_mem_op(CUstream st, unsigned count, CUstreamBatchMemOpParams* params, unsigned flags, bool v2) {
  if (flags || !count || !params) return kInvalidValue;
  std::vector<CUstreamBatchMemOpParams> ops(params, params + count);
  for (const auto& op : ops)
    if (const CUresult r = check_mem_op(op, v2)) return r;
  return stream_mem_op(st, std::move(ops));
}
}  // namespace

VGPU_EXPORT CUresult cuStreamBatchMemOp(CUstream st, unsigned int count, CUstreamBatchMemOpParams* params, unsigned int flags) {
  return batch_mem_op(st, count, params, flags, false);
}
VGPU_EXPORT CUresult cuStreamBatchMemOp_v2(CUstream st, unsigned int count, CUstreamBatchMemOpParams* params, unsigned int flags) {
  return batch_mem_op(st, count, params, flags, true);
}

/* ======================================================================================== */
/* stream capture                                                                            */
/* ======================================================================================== */

VGPU_EXPORT CUresult cuStreamBeginCapture_v2(CUstream st, CUstreamCaptureMode mode) {
  FORWARD("cudaStreamBeginCapture", int (*)(void*, int), st, static_cast<int>(mode));
}
VGPU_EXPORT CUresult cuStreamBeginCapture(CUstream st) { return cuStreamBeginCapture_v2(st, CU_STREAM_CAPTURE_MODE_GLOBAL); }

VGPU_EXPORT CUresult cuStreamBeginCaptureToGraph(CUstream st, CUgraph graph, const CUgraphNode* deps,
                                                 const CUgraphEdgeData* edges, size_t num_deps, CUstreamCaptureMode mode) {
  using Fn = int (*)(void*, void*, const void* const*, const CUgraphEdgeData*, size_t, int);
  FORWARD("cudaStreamBeginCaptureToGraph", Fn, st, graph, reinterpret_cast<const void* const*>(deps), edges, num_deps,
          static_cast<int>(mode));
}

VGPU_EXPORT CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) {
  if (!mode) return kInvalidValue;
  int m = static_cast<int>(*mode);
  const auto f = RT("cudaThreadExchangeStreamCaptureMode", int (*)(int*));
  if (!f) return kNotSupported;
  const int rc = f(&m);
  *mode = static_cast<CUstreamCaptureMode>(m);
  return res(rc);
}

VGPU_EXPORT CUresult cuStreamEndCapture(CUstream st, CUgraph* graph) {
  FORWARD("cudaStreamEndCapture", int (*)(void*, void**), st, reinterpret_cast<void**>(graph));
}

namespace {
// The capture information in the newest form. With no runtime loaded nothing can be capturing.
CUresult capture_info(CUstream st, CUstreamCaptureStatus* status, cuuint64_t* id, CUgraph* graph, const CUgraphNode** deps,
                      const CUgraphEdgeData** edges, size_t* n) {
  using V3 = int (*)(void*, int*, cuuint64_t*, void**, const void***, const CUgraphEdgeData**, size_t*);
  using V2 = int (*)(void*, int*, cuuint64_t*, void**, const void***, size_t*);
  using V1 = int (*)(void*, int*, cuuint64_t*);
  int s = 0;
  cuuint64_t i = 0;
  void* g = nullptr;
  const void** d = nullptr;
  const CUgraphEdgeData* e = nullptr;
  size_t count = 0;
  int rc;
  if (auto f = rt<V3>(runtime_version() >= 13000 ? "cudaStreamGetCaptureInfo" : "cudaStreamGetCaptureInfo_v3")) {
    rc = f(st, &s, &i, &g, &d, &e, &count);
  } else if (auto f2 = rt<V2>("cudaStreamGetCaptureInfo_v2")) {
    rc = f2(st, &s, &i, &g, &d, &count);
  } else if (auto f1 = rt<V1>("cudaStreamGetCaptureInfo")) {
    rc = f1(st, &s, &i);
  } else {
    s = 0, i = 0, rc = 0;   // no runtime: nothing is capturing
  }
  if (rc) return res(rc);
  if (status) *status = static_cast<CUstreamCaptureStatus>(s);
  if (id) *id = i;
  if (graph) *graph = static_cast<CUgraph>(g);
  if (deps) *deps = (const CUgraphNode*)d;
  if (edges) *edges = e;
  if (n) *n = count;
  return CUDA_SUCCESS;
}
}  // namespace

VGPU_EXPORT CUresult cuStreamGetCaptureInfo_v3(CUstream st, CUstreamCaptureStatus* status, cuuint64_t* id, CUgraph* graph,
                                               const CUgraphNode** deps, const CUgraphEdgeData** edges, size_t* n) {
  return capture_info(st, status, id, graph, deps, edges, n);
}
static CUresult cuStreamGetCaptureInfo_v2_impl(CUstream st, CUstreamCaptureStatus* status, cuuint64_t* id, CUgraph* graph,
                                               const CUgraphNode** deps, size_t* n) {
  return capture_info(st, status, id, graph, deps, nullptr, n);
}
VGPU_EXPORT CUresult cuStreamGetCaptureInfo_v2(CUstream st, CUstreamCaptureStatus* status, cuuint64_t* id, CUgraph* graph,
                                               const CUgraphNode** deps, size_t* n) {
  return traced("cuStreamGetCaptureInfo_v2", cuStreamGetCaptureInfo_v2_impl, st, status, id, graph, deps, n);
}
VGPU_EXPORT CUresult cuStreamGetCaptureInfo(CUstream st, CUstreamCaptureStatus* status, cuuint64_t* id) {
  return capture_info(st, status, id, nullptr, nullptr, nullptr, nullptr);
}

namespace {
CUresult update_capture_dependencies(CUstream st, CUgraphNode* deps, const CUgraphEdgeData* edges, size_t n, unsigned flags) {
  using New = int (*)(void*, void**, const CUgraphEdgeData*, size_t, unsigned);
  using Old = int (*)(void*, void**, size_t, unsigned);
  if (runtime_version() >= 13000) {
    if (auto f = rt<New>("cudaStreamUpdateCaptureDependencies")) return res(f(st, reinterpret_cast<void**>(deps), edges, n, flags));
  } else {
    if (auto f = rt<New>("cudaStreamUpdateCaptureDependencies_v2")) return res(f(st, reinterpret_cast<void**>(deps), edges, n, flags));
    if (auto f = rt<Old>("cudaStreamUpdateCaptureDependencies")) return res(f(st, reinterpret_cast<void**>(deps), n, flags));
  }
  return kNotSupported;
}
}  // namespace

VGPU_EXPORT CUresult cuStreamUpdateCaptureDependencies_v2(CUstream st, CUgraphNode* deps, const CUgraphEdgeData* edges,
                                                          size_t n, unsigned int flags) {
  return update_capture_dependencies(st, deps, edges, n, flags);
}
VGPU_EXPORT CUresult cuStreamUpdateCaptureDependencies(CUstream st, CUgraphNode* deps, size_t n, unsigned int flags) {
  return update_capture_dependencies(st, deps, nullptr, n, flags);
}

/* ======================================================================================== */
/* graphs                                                                                    */
/* ======================================================================================== */

namespace {
void forget_nodes(Graph g) {
  using Nodes = int (*)(void*, void**, size_t*);
  const Nodes nodes = RT("cudaGraphGetNodes", Nodes);
  if (!nodes) return;
  size_t n = 0;
  if (nodes(g, nullptr, &n) != 0 || !n) return;
  std::vector<Node> list(n);
  if (nodes(g, list.data(), &n) != 0) return;
  std::lock_guard<std::mutex> lock(g_mu);
  for (size_t i = 0; i < n; ++i) {
    g_params.erase(list[i]);
    g_event_nodes.erase(list[i]);
  }
}
}  // namespace

VGPU_EXPORT CUresult cuGraphCreate(CUgraph* graph, unsigned int flags) {
  FORWARD("cudaGraphCreate", int (*)(void**, unsigned), reinterpret_cast<void**>(graph), flags);
}

VGPU_EXPORT CUresult cuGraphDestroy(CUgraph graph) {
  forget_nodes(graph);
  FORWARD("cudaGraphDestroy", int (*)(void*), graph);
}

VGPU_EXPORT CUresult cuGraphClone(CUgraph* clone, CUgraph original) {
  const auto get_nodes = RT("cudaGraphGetNodes", int (*)(void*, void**, size_t*));
  const auto find = RT("cudaGraphNodeFindInClone", int (*)(void**, void*, void*));
  const auto make = RT("cudaGraphClone", int (*)(void**, void*));
  if (!make || !get_nodes || !find) return kNotSupported;
  if (const int rc = make(reinterpret_cast<void**>(clone), original)) return res(rc);
  // The clone's nodes have the original's parameters.
  size_t n = 0;
  if (get_nodes(original, nullptr, &n) == 0 && n) {
    std::vector<Node> list(n);
    if (get_nodes(original, list.data(), &n) == 0) {
      std::lock_guard<std::mutex> lock(g_mu);
      for (size_t i = 0; i < n; ++i) {
        Node twin = nullptr;
        if (find(&twin, list[i], *clone) != 0 || !twin) continue;
        if (auto it = g_params.find(list[i]); it != g_params.end()) g_params[twin] = it->second;
        if (auto it = g_event_nodes.find(list[i]); it != g_event_nodes.end()) g_event_nodes[twin] = it->second;
      }
    }
  }
  return CUDA_SUCCESS;
}

VGPU_EXPORT CUresult cuGraphNodeFindInClone(CUgraphNode* node, CUgraphNode original, CUgraph clone) {
  FORWARD("cudaGraphNodeFindInClone", int (*)(void**, void*, void*), reinterpret_cast<void**>(node), original, clone);
}

VGPU_EXPORT CUresult cuGraphGetNodes(CUgraph graph, CUgraphNode* nodes, size_t* n) {
  FORWARD("cudaGraphGetNodes", int (*)(void*, void**, size_t*), graph, reinterpret_cast<void**>(nodes), n);
}
VGPU_EXPORT CUresult cuGraphGetRootNodes(CUgraph graph, CUgraphNode* nodes, size_t* n) {
  FORWARD("cudaGraphGetRootNodes", int (*)(void*, void**, size_t*), graph, reinterpret_cast<void**>(nodes), n);
}
VGPU_EXPORT CUresult cuGraphGetEdges_v2(CUgraph graph, CUgraphNode* from, CUgraphNode* to, CUgraphEdgeData* edges, size_t* n) {
  return res(rt_get_edges(graph, from, to, edges, n));
}
VGPU_EXPORT CUresult cuGraphGetEdges(CUgraph graph, CUgraphNode* from, CUgraphNode* to, size_t* n) {
  return res(rt_get_edges(graph, from, to, nullptr, n));
}
VGPU_EXPORT CUresult cuGraphNodeGetDependencies_v2(CUgraphNode node, CUgraphNode* deps, CUgraphEdgeData* edges, size_t* n) {
  return res(rt_node_edges("cudaGraphNodeGetDependencies", node, deps, edges, n));
}
VGPU_EXPORT CUresult cuGraphNodeGetDependencies(CUgraphNode node, CUgraphNode* deps, size_t* n) {
  return res(rt_node_edges("cudaGraphNodeGetDependencies", node, deps, nullptr, n));
}
VGPU_EXPORT CUresult cuGraphNodeGetDependentNodes_v2(CUgraphNode node, CUgraphNode* deps, CUgraphEdgeData* edges, size_t* n) {
  return res(rt_node_edges("cudaGraphNodeGetDependentNodes", node, deps, edges, n));
}
VGPU_EXPORT CUresult cuGraphNodeGetDependentNodes(CUgraphNode node, CUgraphNode* deps, size_t* n) {
  return res(rt_node_edges("cudaGraphNodeGetDependentNodes", node, deps, nullptr, n));
}
VGPU_EXPORT CUresult cuGraphAddDependencies_v2(CUgraph graph, const CUgraphNode* from, const CUgraphNode* to,
                                               const CUgraphEdgeData* edges, size_t n) {
  return res(rt_edit_dependencies("cudaGraphAddDependencies", graph, from, to, edges, n));
}
VGPU_EXPORT CUresult cuGraphAddDependencies(CUgraph graph, const CUgraphNode* from, const CUgraphNode* to, size_t n) {
  return res(rt_edit_dependencies("cudaGraphAddDependencies", graph, from, to, nullptr, n));
}
VGPU_EXPORT CUresult cuGraphRemoveDependencies_v2(CUgraph graph, const CUgraphNode* from, const CUgraphNode* to,
                                                  const CUgraphEdgeData* edges, size_t n) {
  return res(rt_edit_dependencies("cudaGraphRemoveDependencies", graph, from, to, edges, n));
}
VGPU_EXPORT CUresult cuGraphRemoveDependencies(CUgraph graph, const CUgraphNode* from, const CUgraphNode* to, size_t n) {
  return res(rt_edit_dependencies("cudaGraphRemoveDependencies", graph, from, to, nullptr, n));
}

VGPU_EXPORT CUresult cuGraphDestroyNode(CUgraphNode node) {
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_params.erase(node);
    g_event_nodes.erase(node);
  }
  FORWARD("cudaGraphDestroyNode", int (*)(void*), node);
}

VGPU_EXPORT CUresult cuGraphNodeGetType(CUgraphNode node, CUgraphNodeType* type) {
  if (!type) return kInvalidValue;
  const auto f = RT("cudaGraphNodeGetType", int (*)(void*, int*));
  if (!f) return kNotSupported;
  int t = 0;
  if (const int rc = f(node, &t)) return res(rc);
  *type = static_cast<CUgraphNodeType>(t);
  {   // the runtime's view of a batch memory operation is a kernel node
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_params.find(node); it != g_params.end() && it->second->kind == Kind::BatchMemOp)
      *type = CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
  }
  return CUDA_SUCCESS;
}

VGPU_EXPORT CUresult cuGraphNodeGetEnabled(CUgraphExec exec, CUgraphNode node, unsigned int* enabled) {
  FORWARD("cudaGraphNodeGetEnabled", int (*)(void*, void*, unsigned*), exec, node, enabled);
}
VGPU_EXPORT CUresult cuGraphNodeSetEnabled(CUgraphExec exec, CUgraphNode node, unsigned int enabled) {
  FORWARD("cudaGraphNodeSetEnabled", int (*)(void*, void*, unsigned), exec, node, enabled);
}

VGPU_EXPORT CUresult cuGraphDebugDotPrint(CUgraph graph, const char* path, unsigned int flags) {
  FORWARD("cudaGraphDebugDotPrint", int (*)(void*, const char*, unsigned), graph, path, flags);
}

/* ---- instantiation and launch ------------------------------------------------------------------ */

VGPU_EXPORT CUresult cuGraphInstantiateWithFlags(CUgraphExec* exec, CUgraph graph, unsigned long long flags) {
  FORWARD("cudaGraphInstantiateWithFlags", int (*)(void**, void*, unsigned long long), reinterpret_cast<void**>(exec), graph, flags);
}

VGPU_EXPORT CUresult cuGraphInstantiate_v2(CUgraphExec* exec, CUgraph graph, CUgraphNode* error_node, char* log, size_t log_size) {
  if (error_node) *error_node = nullptr;
  if (log && log_size) log[0] = '\0';
  return cuGraphInstantiateWithFlags(exec, graph, 0);
}
VGPU_EXPORT CUresult cuGraphInstantiate(CUgraphExec* exec, CUgraph graph, CUgraphNode* error_node, char* log, size_t log_size) {
  return cuGraphInstantiate_v2(exec, graph, error_node, log, log_size);
}

VGPU_EXPORT CUresult cuGraphInstantiateWithParams(CUgraphExec* exec, CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS* params) {
  if (!params) return kInvalidValue;
  // The runtime's cudaGraphInstantiateParams has the same fields.
  FORWARD("cudaGraphInstantiateWithParams", int (*)(void**, void*, void*), reinterpret_cast<void**>(exec), graph, params);
}

VGPU_EXPORT CUresult cuGraphLaunch(CUgraphExec exec, CUstream st) {
  FORWARD("cudaGraphLaunch", int (*)(void*, void*), exec, st);
}
VGPU_EXPORT CUresult cuGraphUpload(CUgraphExec exec, CUstream st) {
  FORWARD("cudaGraphUpload", int (*)(void*, void*), exec, st);
}
VGPU_EXPORT CUresult cuGraphExecDestroy(CUgraphExec exec) {
  FORWARD("cudaGraphExecDestroy", int (*)(void*), exec);
}
VGPU_EXPORT CUresult cuGraphExecGetFlags(CUgraphExec exec, cuuint64_t* flags) {
  FORWARD("cudaGraphExecGetFlags", int (*)(void*, unsigned long long*), exec, flags);
}

VGPU_EXPORT CUresult cuGraphExecUpdate_v2(CUgraphExec exec, CUgraph graph, CUgraphExecUpdateResultInfo* info) {
  FORWARD("cudaGraphExecUpdate", int (*)(void*, void*, void*), exec, graph, info);
}
VGPU_EXPORT CUresult cuGraphExecUpdate(CUgraphExec exec, CUgraph graph, CUgraphNode* error_node, int* result) {
  CUgraphExecUpdateResultInfo info{};
  const CUresult r = cuGraphExecUpdate_v2(exec, graph, &info);
  if (error_node) *error_node = info.errorNode;
  if (result) *result = info.result;
  return r;
}

/* ---- graph memory ------------------------------------------------------------------------------ */

VGPU_EXPORT CUresult cuDeviceGetGraphMemAttribute(CUdevice dev, int attr, void* value) {
  FORWARD("cudaDeviceGetGraphMemAttribute", int (*)(int, int, void*), dev, attr, value);
}
VGPU_EXPORT CUresult cuDeviceSetGraphMemAttribute(CUdevice dev, int attr, void* value) {
  FORWARD("cudaDeviceSetGraphMemAttribute", int (*)(int, int, void*), dev, attr, value);
}
VGPU_EXPORT CUresult cuDeviceGraphMemTrim(CUdevice dev) {
  FORWARD("cudaDeviceGraphMemTrim", int (*)(int), dev);
}

/* ---- user objects ------------------------------------------------------------------------------ */

VGPU_EXPORT CUresult cuUserObjectCreate(CUuserObject* object, void* ptr, CUhostFn destroy, unsigned int initial, unsigned int flags) {
  FORWARD("cudaUserObjectCreate", int (*)(void**, void*, CUhostFn, unsigned, unsigned), reinterpret_cast<void**>(object), ptr,
          destroy, initial, flags);
}
VGPU_EXPORT CUresult cuUserObjectRetain(CUuserObject object, unsigned int count) {
  FORWARD("cudaUserObjectRetain", int (*)(void*, unsigned), object, count);
}
VGPU_EXPORT CUresult cuUserObjectRelease(CUuserObject object, unsigned int count) {
  FORWARD("cudaUserObjectRelease", int (*)(void*, unsigned), object, count);
}
VGPU_EXPORT CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) {
  FORWARD("cudaGraphRetainUserObject", int (*)(void*, void*, unsigned, unsigned), graph, object, count, flags);
}
VGPU_EXPORT CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) {
  FORWARD("cudaGraphReleaseUserObject", int (*)(void*, void*, unsigned), graph, object, count);
}

/* ======================================================================================== */
/* nodes                                                                                     */
/* ======================================================================================== */

namespace {

// Adds a closure node (the runtime's vgpu_graph_add_closure_node_v1) that runs `p`, and remembers `p`.
CUresult add_closure_node(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t num_deps,
                          std::shared_ptr<const NodeParams> p, int node_type, const std::function<void()>& op) {
  using Fn = int (*)(void*, int, const void* const*, size_t, const std::function<void()>*, const unsigned*, void**);
  const Fn add = RT("vgpu_graph_add_closure_node_v1", Fn);
  if (!add || !out) return add ? kInvalidValue : kNotSupported;
  unsigned shape[7] = {p->grid[0], p->grid[1], p->grid[2], p->block[0], p->block[1], p->block[2], p->shared};
  void* node = nullptr;
  if (const int rc = add(graph, node_type, reinterpret_cast<const void* const*>(deps), num_deps, &op,
                         p->kind == Kind::Kernel ? shape : nullptr, &node))
    return res(rc);
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_params[node] = std::move(p);
  }
  *out = static_cast<CUgraphNode>(node);
  return CUDA_SUCCESS;
}

std::shared_ptr<const NodeParams> params_of(Node node) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_params.find(node);
  return it == g_params.end() ? nullptr : it->second;
}

/* ---- kernel ---- */

CUresult make_kernel_params(const CUDA_KERNEL_NODE_PARAMS_v3& in, std::shared_ptr<NodeParams>* out) {
  auto p = std::make_shared<NodeParams>();
  p->kind = Kind::Kernel;
  p->func = in.func;
  p->kern = in.kern;
  p->grid[0] = in.gridDimX, p->grid[1] = in.gridDimY, p->grid[2] = in.gridDimZ;
  p->block[0] = in.blockDimX, p->block[1] = in.blockDimY, p->block[2] = in.blockDimZ;
  p->shared = in.sharedMemBytes;
  p->extra = in.extra;
  p->ctx = in.ctx;
  void* handle = in.func ? static_cast<void*>(in.func) : static_cast<void*>(in.kern);
  if (!handle) return kInvalidHandle;   // measured: a kernel node with no function
  if (!p->grid[0] || !p->grid[1] || !p->grid[2] || !p->block[0] || !p->block[1] || !p->block[2]) return kInvalidValue;
  if (const int rc = vgpu_driver::copy_kernel_args(handle, in.kernelParams, in.extra, &p->args)) return res(rc);
  for (auto& a : p->args) p->arg_ptrs.push_back(a.data());
  *out = std::move(p);
  return CUDA_SUCCESS;
}

std::function<void()> kernel_closure(std::shared_ptr<const NodeParams> p) {
  return [p] {
    void* handle = p->func ? static_cast<void*>(p->func) : static_cast<void*>(p->kern);
    vgpu_driver::run_kernel(handle, p->grid, p->block, p->shared, p->args);
  };
}

/* ---- memcpy / memset ---- */

std::function<void()> memcpy_closure(std::shared_ptr<const NodeParams> p) {
  return [p] {
    CUDA_MEMCPY3D c = p->copy;
    cuMemcpy3D_v2(&c);
  };
}

CUresult run_memset(const CUDA_MEMSET_NODE_PARAMS_v2& s) {
  const bool flat = s.height <= 1;
  switch (s.elementSize) {
    case 1:
      return flat ? cuMemsetD8_v2(s.dst, static_cast<unsigned char>(s.value), s.width)
                  : cuMemsetD2D8_v2(s.dst, s.pitch, static_cast<unsigned char>(s.value), s.width, s.height);
    case 2:
      return flat ? cuMemsetD16_v2(s.dst, static_cast<unsigned short>(s.value), s.width)
                  : cuMemsetD2D16_v2(s.dst, s.pitch, static_cast<unsigned short>(s.value), s.width, s.height);
    case 4:
      return flat ? cuMemsetD32_v2(s.dst, s.value, s.width) : cuMemsetD2D32_v2(s.dst, s.pitch, s.value, s.width, s.height);
    default: return kInvalidValue;
  }
}

std::function<void()> memset_closure(std::shared_ptr<const NodeParams> p) {
  return [p] { run_memset(p->set); };
}

std::function<void()> memops_closure(std::shared_ptr<const NodeParams> p) {
  return [p] { run_mem_ops(p->ops); };
}

CUresult make_memset_params(const CUDA_MEMSET_NODE_PARAMS_v2& in, std::shared_ptr<NodeParams>* out) {
  if (in.elementSize != 1 && in.elementSize != 2 && in.elementSize != 4) return kInvalidValue;
  if (!in.dst || !in.width || !in.height) return kInvalidValue;
  auto p = std::make_shared<NodeParams>();
  p->kind = Kind::Memset;
  p->set = in;
  *out = std::move(p);
  return CUDA_SUCCESS;
}

CUresult make_memcpy_params(const CUDA_MEMCPY3D& in, CUcontext ctx, std::shared_ptr<NodeParams>* out) {
  // Measured: a zero descriptor is CUDA_ERROR_INVALID_VALUE.
  const auto valid_type = [](int m) { return m >= CU_MEMORYTYPE_HOST && m <= CU_MEMORYTYPE_UNIFIED; };
  if (!valid_type(in.srcMemoryType) || !valid_type(in.dstMemoryType) || !in.WidthInBytes) return kInvalidValue;
  auto p = std::make_shared<NodeParams>();
  p->kind = Kind::Memcpy;
  p->copy = in;
  p->ctx = ctx;
  *out = std::move(p);
  return CUDA_SUCCESS;
}

CUresult make_memop_params(const CUDA_BATCH_MEM_OP_NODE_PARAMS_v2& in, std::shared_ptr<NodeParams>* out) {
  if (!in.count || !in.paramArray || in.flags) return kInvalidValue;
  auto p = std::make_shared<NodeParams>();
  p->kind = Kind::BatchMemOp;
  p->ops.assign(in.paramArray, in.paramArray + in.count);
  p->ctx = in.ctx;
  for (const auto& op : p->ops)
    if (const CUresult r = check_mem_op(op, true)) return r;
  *out = std::move(p);
  return CUDA_SUCCESS;
}

// Replaces what a graph node runs.
CUresult replace_closure(Node node, std::shared_ptr<const NodeParams> p, const std::function<void()>& op) {
  using Fn = int (*)(void*, const std::function<void()>*, const unsigned*);
  const Fn set = RT("vgpu_graph_set_closure_v1", Fn);
  if (!set) return kNotSupported;
  unsigned shape[7] = {p->grid[0], p->grid[1], p->grid[2], p->block[0], p->block[1], p->block[2], p->shared};
  if (const int rc = set(node, &op, p->kind == Kind::Kernel ? shape : nullptr)) return res(rc);
  std::lock_guard<std::mutex> lock(g_mu);
  g_params[node] = std::move(p);
  return CUDA_SUCCESS;
}

// Replaces what an instantiated graph runs for `node` (the node of the graph it was made from).
CUresult replace_exec_closure(Exec exec, Node node, const std::function<void()>& op) {
  using Fn = int (*)(void*, void*, const std::function<void()>*);
  const Fn set = RT("vgpu_graph_exec_set_closure_v1", Fn);
  return set ? res(set(exec, node, &op)) : kNotSupported;
}

CUresult need_kind(Node node, Kind k, std::shared_ptr<const NodeParams>* out) {
  auto p = params_of(node);
  if (!p) return kNotSupported;   // a node the runtime or a capture made: no driver parameters to give
  if (p->kind != k) return kInvalidValue;
  *out = std::move(p);
  return CUDA_SUCCESS;
}

void fill_kernel_out(const NodeParams& p, CUDA_KERNEL_NODE_PARAMS_v3* out) {
  out->func = p.func;
  out->gridDimX = p.grid[0], out->gridDimY = p.grid[1], out->gridDimZ = p.grid[2];
  out->blockDimX = p.block[0], out->blockDimY = p.block[1], out->blockDimZ = p.block[2];
  out->sharedMemBytes = p.shared;
  out->kernelParams = const_cast<void**>(p.arg_ptrs.data());   // the node's own copy, as NVIDIA's gives
  out->extra = p.extra;
  out->kern = p.kern;
  out->ctx = p.ctx;
}

CUDA_KERNEL_NODE_PARAMS_v3 from_v1(const CUDA_KERNEL_NODE_PARAMS_v1& v) {
  CUDA_KERNEL_NODE_PARAMS_v3 n{};
  n.func = v.func;
  n.gridDimX = v.gridDimX, n.gridDimY = v.gridDimY, n.gridDimZ = v.gridDimZ;
  n.blockDimX = v.blockDimX, n.blockDimY = v.blockDimY, n.blockDimZ = v.blockDimZ;
  n.sharedMemBytes = v.sharedMemBytes;
  n.kernelParams = v.kernelParams;
  n.extra = v.extra;
  return n;
}

CUresult add_kernel(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, const CUDA_KERNEL_NODE_PARAMS_v3* in) {
  if (!in || !out) return kInvalidValue;
  std::shared_ptr<NodeParams> p;
  if (const CUresult r = make_kernel_params(*in, &p)) return r;
  return add_closure_node(out, graph, deps, n, p, 0, kernel_closure(p));
}

CUresult set_kernel(Node node, const CUDA_KERNEL_NODE_PARAMS_v3* in) {
  if (!in) return kInvalidValue;
  std::shared_ptr<const NodeParams> old;
  if (const CUresult r = need_kind(node, Kind::Kernel, &old)) return r;
  std::shared_ptr<NodeParams> p;
  if (const CUresult r = make_kernel_params(*in, &p)) return r;
  return replace_closure(node, p, kernel_closure(p));
}

CUresult exec_set_kernel(Exec exec, Node node, const CUDA_KERNEL_NODE_PARAMS_v3* in) {
  if (!in) return kInvalidValue;
  std::shared_ptr<NodeParams> p;
  if (const CUresult r = make_kernel_params(*in, &p)) return r;
  return replace_exec_closure(exec, node, kernel_closure(p));
}

CUresult add_memcpy(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, const CUDA_MEMCPY3D* in, CUcontext ctx) {
  if (!in || !out) return kInvalidValue;
  std::shared_ptr<NodeParams> p;
  if (const CUresult r = make_memcpy_params(*in, ctx, &p)) return r;
  return add_closure_node(out, graph, deps, n, p, 1, memcpy_closure(p));
}

CUresult add_memset(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, const CUDA_MEMSET_NODE_PARAMS_v2* in) {
  if (!in || !out) return kInvalidValue;
  std::shared_ptr<NodeParams> p;
  if (const CUresult r = make_memset_params(*in, &p)) return r;
  return add_closure_node(out, graph, deps, n, p, 2, memset_closure(p));
}

CUresult add_memops(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, const CUDA_BATCH_MEM_OP_NODE_PARAMS_v2* in) {
  if (!in || !out) return kInvalidValue;
  std::shared_ptr<NodeParams> p;
  if (const CUresult r = make_memop_params(*in, &p)) return r;
  return add_closure_node(out, graph, deps, n, p, 0, memops_closure(p));
}

/* ---- host, child, empty ---- */

CUresult add_host(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, const CUDA_HOST_NODE_PARAMS_v2* in) {
  if (!in) return kInvalidValue;
  FORWARD("cudaGraphAddHostNode", int (*)(void**, void*, const void* const*, size_t, const void*), reinterpret_cast<void**>(out), graph,
          reinterpret_cast<const void* const*>(deps), n, in);
}

CUresult add_child(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUgraph child) {
  FORWARD("cudaGraphAddChildGraphNode", int (*)(void**, void*, const void* const*, size_t, void*), reinterpret_cast<void**>(out), graph,
          reinterpret_cast<const void* const*>(deps), n, child);
}

CUresult add_empty(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n) {
  FORWARD("cudaGraphAddEmptyNode", int (*)(void**, void*, const void* const*, size_t), reinterpret_cast<void**>(out), graph,
          reinterpret_cast<const void* const*>(deps), n);
}

/* ---- events ---- */

CUresult add_event_node(const char* rt_name, CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUevent event) {
  using Fn = int (*)(void**, void*, const void* const*, size_t, void*);
  const Fn add = rt<Fn>(rt_name);
  if (!add || !out || !event) return add ? kInvalidValue : kNotSupported;
  void* rt_event = vgpu_driver::runtime_event_for(event);
  if (!rt_event) return kInvalidHandle;
  if (const int rc = add(reinterpret_cast<void**>(out), graph, reinterpret_cast<const void* const*>(deps), n, rt_event)) return res(rc);
  std::lock_guard<std::mutex> lock(g_mu);
  g_event_nodes[*out] = event;
  return CUDA_SUCCESS;
}

CUresult get_event_node(const char* rt_name, Node node, CUevent* out) {
  using Fn = int (*)(void*, void**);
  const Fn get = rt<Fn>(rt_name);
  if (!get || !out) return get ? kInvalidValue : kNotSupported;
  void* rt_event = nullptr;
  if (const int rc = get(node, &rt_event)) return res(rc);
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_event_nodes.find(node); it != g_event_nodes.end()) {
      *out = it->second;
      return CUDA_SUCCESS;
    }
  }
  void* mine = vgpu_driver::driver_event_of(rt_event);
  *out = static_cast<CUevent>(mine ? mine : rt_event);
  return CUDA_SUCCESS;
}

CUresult set_event_node(const char* rt_name, Node node, CUevent event) {
  using Fn = int (*)(void*, void*);
  const Fn set = rt<Fn>(rt_name);
  if (!set || !event) return set ? kInvalidValue : kNotSupported;
  void* rt_event = vgpu_driver::runtime_event_for(event);
  if (!rt_event) return kInvalidHandle;
  if (const int rc = set(node, rt_event)) return res(rc);
  std::lock_guard<std::mutex> lock(g_mu);
  g_event_nodes[node] = event;
  return CUDA_SUCCESS;
}

CUresult exec_set_event_node(const char* rt_name, Exec exec, Node node, CUevent event) {
  using Fn = int (*)(void*, void*, void*);
  const Fn set = rt<Fn>(rt_name);
  if (!set || !event) return set ? kInvalidValue : kNotSupported;
  void* rt_event = vgpu_driver::runtime_event_for(event);
  return rt_event ? res(set(exec, node, rt_event)) : kInvalidHandle;
}

/* ---- graph memory ---- */

CUresult add_mem_alloc(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUDA_MEM_ALLOC_NODE_PARAMS_v2* params) {
  if (!params || !out) return kInvalidValue;
  // The runtime's cudaMemAllocNodeParams has the same layout; dptr comes back through it.
  FORWARD("cudaGraphAddMemAllocNode", int (*)(void**, void*, const void* const*, size_t, void*), reinterpret_cast<void**>(out), graph,
          reinterpret_cast<const void* const*>(deps), n, params);
}

}  // namespace

/* ---- exported node calls ----------------------------------------------------------------------- */

VGPU_EXPORT CUresult cuGraphAddEmptyNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n) {
  return add_empty(out, graph, deps, n);
}
VGPU_EXPORT CUresult cuGraphAddHostNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n,
                                        const CUDA_HOST_NODE_PARAMS_v1* params) {
  return add_host(out, graph, deps, n, params);
}
VGPU_EXPORT CUresult cuGraphHostNodeGetParams(CUgraphNode node, CUDA_HOST_NODE_PARAMS_v1* params) {
  FORWARD("cudaGraphHostNodeGetParams", int (*)(void*, void*), node, params);
}
VGPU_EXPORT CUresult cuGraphHostNodeSetParams(CUgraphNode node, const CUDA_HOST_NODE_PARAMS_v1* params) {
  FORWARD("cudaGraphHostNodeSetParams", int (*)(void*, const void*), node, params);
}
VGPU_EXPORT CUresult cuGraphExecHostNodeSetParams(CUgraphExec exec, CUgraphNode node, const CUDA_HOST_NODE_PARAMS_v1* params) {
  FORWARD("cudaGraphExecHostNodeSetParams", int (*)(void*, void*, const void*), exec, node, params);
}
VGPU_EXPORT CUresult cuGraphAddChildGraphNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUgraph child) {
  return add_child(out, graph, deps, n, child);
}
VGPU_EXPORT CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode node, CUgraph* graph) {
  FORWARD("cudaGraphChildGraphNodeGetGraph", int (*)(void*, void**), node, reinterpret_cast<void**>(graph));
}
VGPU_EXPORT CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec exec, CUgraphNode node, CUgraph child) {
  FORWARD("cudaGraphExecChildGraphNodeSetParams", int (*)(void*, void*, void*), exec, node, child);
}

VGPU_EXPORT CUresult cuGraphAddEventRecordNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUevent event) {
  return add_event_node("cudaGraphAddEventRecordNode", out, graph, deps, n, event);
}
VGPU_EXPORT CUresult cuGraphAddEventWaitNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUevent event) {
  return add_event_node("cudaGraphAddEventWaitNode", out, graph, deps, n, event);
}
VGPU_EXPORT CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode node, CUevent* event) {
  return get_event_node("cudaGraphEventRecordNodeGetEvent", node, event);
}
VGPU_EXPORT CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode node, CUevent* event) {
  return get_event_node("cudaGraphEventWaitNodeGetEvent", node, event);
}
VGPU_EXPORT CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode node, CUevent event) {
  return set_event_node("cudaGraphEventRecordNodeSetEvent", node, event);
}
VGPU_EXPORT CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode node, CUevent event) {
  return set_event_node("cudaGraphEventWaitNodeSetEvent", node, event);
}
VGPU_EXPORT CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec exec, CUgraphNode node, CUevent event) {
  return exec_set_event_node("cudaGraphExecEventRecordNodeSetEvent", exec, node, event);
}
VGPU_EXPORT CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec exec, CUgraphNode node, CUevent event) {
  return exec_set_event_node("cudaGraphExecEventWaitNodeSetEvent", exec, node, event);
}

VGPU_EXPORT CUresult cuGraphAddKernelNode_v2(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n,
                                             const CUDA_KERNEL_NODE_PARAMS_v3* params) {
  return add_kernel(out, graph, deps, n, params);
}
VGPU_EXPORT CUresult cuGraphAddKernelNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n,
                                          const CUDA_KERNEL_NODE_PARAMS_v1* params) {
  if (!params) return kInvalidValue;
  const CUDA_KERNEL_NODE_PARAMS_v3 v3 = from_v1(*params);
  return add_kernel(out, graph, deps, n, &v3);
}
VGPU_EXPORT CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode node, CUDA_KERNEL_NODE_PARAMS_v3* params) {
  if (!params) return kInvalidValue;
  std::shared_ptr<const NodeParams> p;
  if (const CUresult r = need_kind(node, Kind::Kernel, &p)) return r;
  fill_kernel_out(*p, params);
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuGraphKernelNodeGetParams(CUgraphNode node, CUDA_KERNEL_NODE_PARAMS_v1* params) {
  if (!params) return kInvalidValue;
  CUDA_KERNEL_NODE_PARAMS_v3 v3{};
  if (const CUresult r = cuGraphKernelNodeGetParams_v2(node, &v3)) return r;
  std::memcpy(params, &v3, sizeof(CUDA_KERNEL_NODE_PARAMS_v1));   // v1 is v3's first ten fields
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS_v3* params) {
  return set_kernel(node, params);
}
VGPU_EXPORT CUresult cuGraphKernelNodeSetParams(CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS_v1* params) {
  if (!params) return kInvalidValue;
  const CUDA_KERNEL_NODE_PARAMS_v3 v3 = from_v1(*params);
  return set_kernel(node, &v3);
}
VGPU_EXPORT CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec exec, CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS_v3* params) {
  return exec_set_kernel(exec, node, params);
}
VGPU_EXPORT CUresult cuGraphExecKernelNodeSetParams(CUgraphExec exec, CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS_v1* params) {
  if (!params) return kInvalidValue;
  const CUDA_KERNEL_NODE_PARAMS_v3 v3 = from_v1(*params);
  return exec_set_kernel(exec, node, &v3);
}
VGPU_EXPORT CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) {
  FORWARD("cudaGraphKernelNodeCopyAttributes", int (*)(void*, void*), dst, src);
}
VGPU_EXPORT CUresult cuGraphKernelNodeGetAttribute(CUgraphNode node, int attr, void* value) {
  FORWARD("cudaGraphKernelNodeGetAttribute", int (*)(void*, int, void*), node, attr, value);
}
VGPU_EXPORT CUresult cuGraphKernelNodeSetAttribute(CUgraphNode node, int attr, const void* value) {
  FORWARD("cudaGraphKernelNodeSetAttribute", int (*)(void*, int, const void*), node, attr, value);
}

VGPU_EXPORT CUresult cuGraphAddMemcpyNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n,
                                          const CUDA_MEMCPY3D* copy, CUcontext ctx) {
  return add_memcpy(out, graph, deps, n, copy, ctx);
}
VGPU_EXPORT CUresult cuGraphMemcpyNodeGetParams(CUgraphNode node, CUDA_MEMCPY3D* params) {
  if (!params) return kInvalidValue;
  std::shared_ptr<const NodeParams> p;
  if (const CUresult r = need_kind(node, Kind::Memcpy, &p)) return r;
  *params = p->copy;
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuGraphMemcpyNodeSetParams(CUgraphNode node, const CUDA_MEMCPY3D* params) {
  if (!params) return kInvalidValue;
  std::shared_ptr<const NodeParams> old;
  if (const CUresult r = need_kind(node, Kind::Memcpy, &old)) return r;
  std::shared_ptr<NodeParams> p;
  if (const CUresult r = make_memcpy_params(*params, old->ctx, &p)) return r;
  return replace_closure(node, p, memcpy_closure(p));
}
VGPU_EXPORT CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec exec, CUgraphNode node, const CUDA_MEMCPY3D* params, CUcontext ctx) {
  if (!params) return kInvalidValue;
  std::shared_ptr<NodeParams> p;
  if (const CUresult r = make_memcpy_params(*params, ctx, &p)) return r;
  return replace_exec_closure(exec, node, memcpy_closure(p));
}

VGPU_EXPORT CUresult cuGraphAddMemsetNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n,
                                          const CUDA_MEMSET_NODE_PARAMS_v1* params, CUcontext ctx) {
  if (!params) return kInvalidValue;
  CUDA_MEMSET_NODE_PARAMS_v2 v2{};
  std::memcpy(&v2, params, sizeof *params);
  v2.ctx = ctx;
  return add_memset(out, graph, deps, n, &v2);
}
VGPU_EXPORT CUresult cuGraphMemsetNodeGetParams(CUgraphNode node, CUDA_MEMSET_NODE_PARAMS_v1* params) {
  if (!params) return kInvalidValue;
  std::shared_ptr<const NodeParams> p;
  if (const CUresult r = need_kind(node, Kind::Memset, &p)) return r;
  std::memcpy(params, &p->set, sizeof *params);
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuGraphMemsetNodeSetParams(CUgraphNode node, const CUDA_MEMSET_NODE_PARAMS_v1* params) {
  if (!params) return kInvalidValue;
  std::shared_ptr<const NodeParams> old;
  if (const CUresult r = need_kind(node, Kind::Memset, &old)) return r;
  CUDA_MEMSET_NODE_PARAMS_v2 v2{};
  std::memcpy(&v2, params, sizeof *params);
  v2.ctx = old->set.ctx;
  std::shared_ptr<NodeParams> p;
  if (const CUresult r = make_memset_params(v2, &p)) return r;
  return replace_closure(node, p, memset_closure(p));
}
VGPU_EXPORT CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec exec, CUgraphNode node, const CUDA_MEMSET_NODE_PARAMS_v1* params,
                                                    CUcontext ctx) {
  if (!params) return kInvalidValue;
  CUDA_MEMSET_NODE_PARAMS_v2 v2{};
  std::memcpy(&v2, params, sizeof *params);
  v2.ctx = ctx;
  std::shared_ptr<NodeParams> p;
  if (const CUresult r = make_memset_params(v2, &p)) return r;
  return replace_exec_closure(exec, node, memset_closure(p));
}

VGPU_EXPORT CUresult cuGraphAddBatchMemOpNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n,
                                              const CUDA_BATCH_MEM_OP_NODE_PARAMS_v1* params) {
  return add_memops(out, graph, deps, n, params);
}
VGPU_EXPORT CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode node, CUDA_BATCH_MEM_OP_NODE_PARAMS_v1* params) {
  if (!params) return kInvalidValue;
  std::shared_ptr<const NodeParams> p;
  if (const CUresult r = need_kind(node, Kind::BatchMemOp, &p)) return r;
  // The array is the caller's to size (count says how many it holds).
  if (params->paramArray && params->count >= p->ops.size())
    std::memcpy(params->paramArray, p->ops.data(), p->ops.size() * sizeof(CUstreamBatchMemOpParams));
  params->count = static_cast<unsigned>(p->ops.size());
  params->ctx = p->ctx;
  params->flags = p->flags;
  return CUDA_SUCCESS;
}
VGPU_EXPORT CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode node, const CUDA_BATCH_MEM_OP_NODE_PARAMS_v1* params) {
  if (!params) return kInvalidValue;
  std::shared_ptr<const NodeParams> old;
  if (const CUresult r = need_kind(node, Kind::BatchMemOp, &old)) return r;
  std::shared_ptr<NodeParams> p;
  if (const CUresult r = make_memop_params(*params, &p)) return r;
  return replace_closure(node, p, memops_closure(p));
}
VGPU_EXPORT CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec exec, CUgraphNode node, const CUDA_BATCH_MEM_OP_NODE_PARAMS_v1* params) {
  if (!params) return kInvalidValue;
  std::shared_ptr<NodeParams> p;
  if (const CUresult r = make_memop_params(*params, &p)) return r;
  return replace_exec_closure(exec, node, memops_closure(p));
}

VGPU_EXPORT CUresult cuGraphAddMemAllocNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n,
                                            CUDA_MEM_ALLOC_NODE_PARAMS_v1* params) {
  return add_mem_alloc(out, graph, deps, n, params);
}
VGPU_EXPORT CUresult cuGraphMemAllocNodeGetParams(CUgraphNode node, CUDA_MEM_ALLOC_NODE_PARAMS_v1* params) {
  FORWARD("cudaGraphMemAllocNodeGetParams", int (*)(void*, void*), node, params);
}
VGPU_EXPORT CUresult cuGraphAddMemFreeNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUdeviceptr dptr) {
  FORWARD("cudaGraphAddMemFreeNode", int (*)(void**, void*, const void* const*, size_t, void*), reinterpret_cast<void**>(out), graph,
          reinterpret_cast<const void* const*>(deps), n, reinterpret_cast<void*>(dptr));
}
VGPU_EXPORT CUresult cuGraphMemFreeNodeGetParams(CUgraphNode node, CUdeviceptr* dptr) {
  FORWARD("cudaGraphMemFreeNodeGetParams", int (*)(void*, void*), node, dptr);
}

/* ---- the tagged-union forms -------------------------------------------------------------------- */

VGPU_EXPORT CUresult cuGraphAddNode_v2(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, const CUgraphEdgeData* edges,
                                       size_t n, CUgraphNodeParams* params) {
  if (!params || !out) return kInvalidValue;
  if (edges)
    for (size_t i = 0; i < n; ++i)
      if (edges[i].from_port || edges[i].to_port || edges[i].type) return kNotSupported;   // only plain edges here
  switch (params->type) {
    case CU_GRAPH_NODE_TYPE_KERNEL: return add_kernel(out, graph, deps, n, &params->kernel);
    case CU_GRAPH_NODE_TYPE_MEMCPY: return add_memcpy(out, graph, deps, n, &params->memcpy.copyParams, params->memcpy.copyCtx);
    case CU_GRAPH_NODE_TYPE_MEMSET: return add_memset(out, graph, deps, n, &params->memset);
    case CU_GRAPH_NODE_TYPE_HOST: return add_host(out, graph, deps, n, &params->host);
    case CU_GRAPH_NODE_TYPE_GRAPH: return add_child(out, graph, deps, n, params->graph.graph);
    case CU_GRAPH_NODE_TYPE_EMPTY: return add_empty(out, graph, deps, n);
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT: return add_event_node("cudaGraphAddEventWaitNode", out, graph, deps, n, params->eventWait.event);
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD:
      return add_event_node("cudaGraphAddEventRecordNode", out, graph, deps, n, params->eventRecord.event);
    case CU_GRAPH_NODE_TYPE_MEM_ALLOC: return add_mem_alloc(out, graph, deps, n, &params->alloc);
    case CU_GRAPH_NODE_TYPE_MEM_FREE: return cuGraphAddMemFreeNode(out, graph, deps, n, params->free.dptr);
    case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: return add_memops(out, graph, deps, n, &params->memOp);
    default: return kNotSupported;   // external semaphores and conditional nodes
  }
}
VGPU_EXPORT CUresult cuGraphAddNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUgraphNodeParams* params) {
  return cuGraphAddNode_v2(out, graph, deps, nullptr, n, params);
}

VGPU_EXPORT CUresult cuGraphNodeSetParams(CUgraphNode node, CUgraphNodeParams* params) {
  if (!params) return kInvalidValue;
  switch (params->type) {
    case CU_GRAPH_NODE_TYPE_KERNEL: return set_kernel(node, &params->kernel);
    case CU_GRAPH_NODE_TYPE_MEMCPY: return cuGraphMemcpyNodeSetParams(node, &params->memcpy.copyParams);
    case CU_GRAPH_NODE_TYPE_MEMSET: {
      CUDA_MEMSET_NODE_PARAMS_v1 v1{};
      std::memcpy(&v1, &params->memset, sizeof v1);
      return cuGraphMemsetNodeSetParams(node, &v1);
    }
    case CU_GRAPH_NODE_TYPE_HOST: return cuGraphHostNodeSetParams(node, &params->host);
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT: return cuGraphEventWaitNodeSetEvent(node, params->eventWait.event);
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD: return cuGraphEventRecordNodeSetEvent(node, params->eventRecord.event);
    case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: return cuGraphBatchMemOpNodeSetParams(node, &params->memOp);
    default: return kNotSupported;
  }
}

VGPU_EXPORT CUresult cuGraphExecNodeSetParams(CUgraphExec exec, CUgraphNode node, CUgraphNodeParams* params) {
  if (!params) return kInvalidValue;
  switch (params->type) {
    case CU_GRAPH_NODE_TYPE_KERNEL: return exec_set_kernel(exec, node, &params->kernel);
    case CU_GRAPH_NODE_TYPE_MEMCPY:
      return cuGraphExecMemcpyNodeSetParams(exec, node, &params->memcpy.copyParams, params->memcpy.copyCtx);
    case CU_GRAPH_NODE_TYPE_MEMSET: {
      CUDA_MEMSET_NODE_PARAMS_v1 v1{};
      std::memcpy(&v1, &params->memset, sizeof v1);
      return cuGraphExecMemsetNodeSetParams(exec, node, &v1, params->memset.ctx);
    }
    case CU_GRAPH_NODE_TYPE_HOST: return cuGraphExecHostNodeSetParams(exec, node, &params->host);
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT: return cuGraphExecEventWaitNodeSetEvent(exec, node, params->eventWait.event);
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD: return cuGraphExecEventRecordNodeSetEvent(exec, node, params->eventRecord.event);
    case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: return cuGraphExecBatchMemOpNodeSetParams(exec, node, &params->memOp);
    default: return kNotSupported;
  }
}

/* ---- ids and parameters of graphs and nodes (CUDA 13.2) ---------------------------------------- */
// The ids come from the runtime (vgpu_graph_id_v1 and the rest: runtime_api.cpp has the numbering and the card it was
// measured on); a null graph, executable or node, a null result and a handle that is none are CUDA_ERROR_INVALID_VALUE.

VGPU_EXPORT CUresult cuGraphGetId(CUgraph graph, unsigned int* graphId) {
  return traced("cuGraphGetId", [](CUgraph g, unsigned int* id) -> CUresult {
    FORWARD("vgpu_graph_id_v1", int (*)(void*, unsigned int*), g, id);
  }, graph, graphId);
}
VGPU_EXPORT CUresult cuGraphExecGetId(CUgraphExec exec, unsigned int* graphId) {
  return traced("cuGraphExecGetId", [](CUgraphExec e, unsigned int* id) -> CUresult {
    FORWARD("vgpu_graph_exec_id_v1", int (*)(void*, unsigned int*), e, id);
  }, exec, graphId);
}
VGPU_EXPORT CUresult cuGraphNodeGetLocalId(CUgraphNode node, unsigned int* nodeId) {
  return traced("cuGraphNodeGetLocalId", [](CUgraphNode n, unsigned int* id) -> CUresult {
    if (!id) return kInvalidValue;
    FORWARD("vgpu_graph_node_ids_v1", int (*)(void*, unsigned int*, unsigned long long*, void**), n, id, nullptr, nullptr);
  }, node, nodeId);
}
VGPU_EXPORT CUresult cuGraphNodeGetToolsId(CUgraphNode node, unsigned long long* toolsNodeId) {
  return traced("cuGraphNodeGetToolsId", [](CUgraphNode n, unsigned long long* id) -> CUresult {
    if (!id) return kInvalidValue;
    FORWARD("vgpu_graph_node_ids_v1", int (*)(void*, unsigned int*, unsigned long long*, void**), n, nullptr, id, nullptr);
  }, node, toolsNodeId);
}
VGPU_EXPORT CUresult cuGraphNodeGetContainingGraph(CUgraphNode node, CUgraph* phGraph) {
  return traced("cuGraphNodeGetContainingGraph", [](CUgraphNode n, CUgraph* g) -> CUresult {
    if (!g) return kInvalidValue;
    FORWARD("vgpu_graph_node_ids_v1", int (*)(void*, unsigned int*, unsigned long long*, void**), n, nullptr, nullptr,
            reinterpret_cast<void**>(g));
  }, node, phGraph);
}

// The parameters of a node of any kind, as the tagged union cuGraphAddNode takes: the answer of the kind's own
// getter. A node of a kind this graph engine cannot make (memory allocation and free, batch memory operations,
// external semaphores, conditional nodes) is CUDA_ERROR_NOT_SUPPORTED here; an empty node is the type and nothing else.
VGPU_EXPORT CUresult cuGraphNodeGetParams(CUgraphNode node, CUgraphNodeParams* nodeParams) {
  return traced("cuGraphNodeGetParams", [](CUgraphNode n, CUgraphNodeParams* out) -> CUresult {
    if (!n || !out) return kInvalidValue;
    CUgraphNodeType type;
    if (const CUresult r = cuGraphNodeGetType(n, &type)) return r;
    std::memset(out, 0, sizeof *out);
    out->type = type;
    switch (type) {
      case CU_GRAPH_NODE_TYPE_KERNEL: return cuGraphKernelNodeGetParams_v2(n, &out->kernel);
      case CU_GRAPH_NODE_TYPE_MEMCPY: return cuGraphMemcpyNodeGetParams(n, &out->memcpy.copyParams);
      case CU_GRAPH_NODE_TYPE_MEMSET: {
        CUDA_MEMSET_NODE_PARAMS_v1 v1{};
        if (const CUresult r = cuGraphMemsetNodeGetParams(n, &v1)) return r;
        std::memcpy(&out->memset, &v1, sizeof v1);
        return CUDA_SUCCESS;
      }
      case CU_GRAPH_NODE_TYPE_HOST: return cuGraphHostNodeGetParams(n, &out->host);
      case CU_GRAPH_NODE_TYPE_GRAPH: return cuGraphChildGraphNodeGetGraph(n, &out->graph.graph);
      case CU_GRAPH_NODE_TYPE_EMPTY: return CUDA_SUCCESS;
      case CU_GRAPH_NODE_TYPE_WAIT_EVENT: return cuGraphEventWaitNodeGetEvent(n, &out->eventWait.event);
      case CU_GRAPH_NODE_TYPE_EVENT_RECORD: return cuGraphEventRecordNodeGetEvent(n, &out->eventRecord.event);
      default: return kNotSupported;
    }
  }, node, nodeParams);
}

/* External semaphores are imported through cuImportExternalSemaphore, which this library does not have;
 * a node that names one has nothing to name. */
VGPU_EXPORT CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const void*) {
  return kNotSupported;
}
VGPU_EXPORT CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const void*) {
  return kNotSupported;
}

/* ===================================================================== */
/* Per-thread default stream                                             */
/* ===================================================================== */

// A program built with nvcc --default-stream per-thread (or with CUDA_API_PER_THREAD_DEFAULT_STREAM defined)
// calls the driver's functions under the names cuMemcpyHtoD_v2_ptds, cuLaunchKernel_ptsz and the rest, where
// stream 0 is the calling thread's own default stream rather than the legacy one. Every stream here is
// synchronous, so the two defaults behave alike and each name is the plain function under another symbol
// (the runtime does the same: runtime_api.cpp).

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattribute-alias"
#endif
#define VGPU_PT_ALIAS(name, target) \
  extern "C" __attribute__((visibility("default"))) void name() __attribute__((alias(#target)));
VGPU_PT_ALIAS(cuGraphInstantiateWithParams_ptsz, cuGraphInstantiateWithParams)
VGPU_PT_ALIAS(cuGraphLaunch_ptsz, cuGraphLaunch)
VGPU_PT_ALIAS(cuGraphUpload_ptsz, cuGraphUpload)
VGPU_PT_ALIAS(cuStreamBatchMemOp_v2_ptsz, cuStreamBatchMemOp_v2)
VGPU_PT_ALIAS(cuStreamBeginCaptureToGraph_ptsz, cuStreamBeginCaptureToGraph)
VGPU_PT_ALIAS(cuStreamBeginCapture_v2_ptsz, cuStreamBeginCapture_v2)
VGPU_PT_ALIAS(cuStreamEndCapture_ptsz, cuStreamEndCapture)
VGPU_PT_ALIAS(cuStreamGetCaptureInfo_v3_ptsz, cuStreamGetCaptureInfo_v3)
VGPU_PT_ALIAS(cuStreamUpdateCaptureDependencies_v2_ptsz, cuStreamUpdateCaptureDependencies_v2)
VGPU_PT_ALIAS(cuStreamUpdateCaptureDependencies_ptsz, cuStreamUpdateCaptureDependencies)
VGPU_PT_ALIAS(cuStreamGetCaptureInfo_v2_ptsz, cuStreamGetCaptureInfo_v2)
VGPU_PT_ALIAS(cuStreamWaitValue32_v2_ptsz, cuStreamWaitValue32_v2)
VGPU_PT_ALIAS(cuStreamWaitValue64_v2_ptsz, cuStreamWaitValue64_v2)
VGPU_PT_ALIAS(cuStreamWriteValue32_v2_ptsz, cuStreamWriteValue32_v2)
VGPU_PT_ALIAS(cuStreamWriteValue64_v2_ptsz, cuStreamWriteValue64_v2)
#undef VGPU_PT_ALIAS
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
