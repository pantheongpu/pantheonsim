// Graph edges with data, copies to and from CUDA arrays in graphs (built and
// captured), child graphs the parent owns, external semaphore nodes and the
// external memory and semaphore calls, and kernel node attributes. Every check
// passes on an RTX 3060's runtime (driver 13.2, under WSL) as well as on this
// simulator.
//
// The edge-data calls exist from CUDA 12.3 (as _v2 names until 13), moving a child graph
// into its parent from 13; the checks for them are compiled out below that.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

__global__ void inc(int* p) { atomicAdd(p, 1); }

static int* g_dev = nullptr;

static cudaGraphNode_t kernel_node(cudaGraph_t g) {
  cudaGraphNode_t n = nullptr;
  cudaKernelNodeParams kp = {};
  int* d = g_dev;
  void* args[] = {&d};
  kp.func = reinterpret_cast<void*>(inc);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  kp.kernelParams = args;
  cudaGraphAddKernelNode(&n, g, nullptr, 0, &kp);
  return n;
}
static cudaGraphNode_t empty_node(cudaGraph_t g) {
  cudaGraphNode_t n = nullptr;
  cudaGraphAddEmptyNode(&n, g, nullptr, 0);
  return n;
}
static cudaGraphNode_t memset_node(cudaGraph_t g) {
  cudaGraphNode_t n = nullptr;
  cudaMemsetParams mp = {};
  mp.dst = g_dev;
  mp.elementSize = 4;
  mp.width = 1;
  mp.height = 1;
  cudaGraphAddMemsetNode(&n, g, nullptr, 0, &mp);
  return n;
}
static int cc_major() {
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrComputeCapabilityMajor, 0);
  return v;
}

// ---- edge data ----
#if CUDART_VERSION >= 12030
#if CUDART_VERSION >= 13000
#define ADD_DEPS(g, f, t, e, n) cudaGraphAddDependencies(g, f, t, e, n)
#define REMOVE_DEPS(g, f, t, e, n) cudaGraphRemoveDependencies(g, f, t, e, n)
#define GET_EDGES(g, f, t, e, n) cudaGraphGetEdges(g, f, t, e, n)
#define NODE_DEPS(node, d, e, n) cudaGraphNodeGetDependencies(node, d, e, n)
#define NODE_DEPENDENTS(node, d, e, n) cudaGraphNodeGetDependentNodes(node, d, e, n)
#else
#define ADD_DEPS(g, f, t, e, n) cudaGraphAddDependencies_v2(g, f, t, e, n)
#define REMOVE_DEPS(g, f, t, e, n) cudaGraphRemoveDependencies_v2(g, f, t, e, n)
#define GET_EDGES(g, f, t, e, n) cudaGraphGetEdges_v2(g, f, t, e, n)
#define NODE_DEPS(node, d, e, n) cudaGraphNodeGetDependencies_v2(node, d, e, n)
#define NODE_DEPENDENTS(node, d, e, n) cudaGraphNodeGetDependentNodes_v2(node, d, e, n)
#endif

static cudaGraphEdgeData E(int from_port, int to_port, int type) {
  cudaGraphEdgeData e;
  std::memset(&e, 0, sizeof e);
  e.from_port = static_cast<unsigned char>(from_port);
  e.to_port = static_cast<unsigned char>(to_port);
  e.type = static_cast<unsigned char>(type);
  return e;
}
static bool edge_is(cudaGraph_t g, cudaGraphNode_t from, cudaGraphNode_t to, int fp, int tp, int ty) {
  size_t n = 0;
  GET_EDGES(g, nullptr, nullptr, nullptr, &n);
  std::vector<cudaGraphNode_t> f(n), t(n);
  std::vector<cudaGraphEdgeData> e(n);
  GET_EDGES(g, f.data(), t.data(), e.data(), &n);
  for (size_t i = 0; i < n; ++i)
    if (f[i] == from && t[i] == to) return e[i].from_port == fp && e[i].to_port == tp && e[i].type == ty;
  return false;
}
static size_t edge_count(cudaGraph_t g) {
  size_t n = 0;
  GET_EDGES(g, nullptr, nullptr, nullptr, &n);
  return n;
}

static void edge_data() {
  cudaGraph_t g;
  cudaGraphCreate(&g, 0);
  // The edges that carry data a compute capability 8 device takes: the plain one, and the launch-completion
  // port with the default type.
  struct Case {
    const char* name;
    cudaGraphEdgeData e;
    bool ok;
  } cases[] = {
      {"(0,0,0) the default", E(0, 0, 0), true},
      {"(2,0,0) the launch-completion port", E(2, 0, 0), true},
      {"(1,0,0) the programmatic port, default type", E(1, 0, 0), false},
      {"(0,0,1) programmatic type from the full-completion port", E(0, 0, 1), false},
      {"(1,0,1) programmatic", E(1, 0, 1), cc_major() >= 9},
      {"(2,0,1) launch completion, programmatic type", E(2, 0, 1), cc_major() >= 9},
      {"(0,1,0) a destination port", E(0, 1, 0), false},
      {"(3,0,1) a source port that does not exist", E(3, 0, 1), false},
      {"(0,0,2) a type that does not exist", E(0, 0, 2), false},
  };
  for (const Case& c : cases) {
    cudaGraphNode_t a = kernel_node(g), b = kernel_node(g);
    const cudaError_t rc = ADD_DEPS(g, &a, &b, &c.e, 1);
    char what[160];
    std::snprintf(what, sizeof what, "edge %s kernel to kernel is %s", c.name, c.ok ? "taken" : "refused");
    check(rc == (c.ok ? cudaSuccess : cudaErrorInvalidValue), what);
    if (!c.ok) {
      check(edge_count(g) == edge_count(g) && !edge_is(g, a, b, c.e.from_port, c.e.to_port, c.e.type),
            "a refused edge adds nothing");
    } else {
      std::snprintf(what, sizeof what, "edge %s reads back as given", c.name);
      check(edge_is(g, a, b, c.e.from_port, c.e.to_port, c.e.type), what);
      cudaGraphNode_t deps[2];
      cudaGraphEdgeData de[2];
      size_t nd = 2;
      IS(NODE_DEPS(b, deps, de, &nd), cudaSuccess);
      check(nd == 1 && deps[0] == a && de[0].from_port == c.e.from_port && de[0].type == c.e.type,
            "cudaGraphNodeGetDependencies hands the edge data back");
      cudaGraphNode_t after[2];
      cudaGraphEdgeData ae[2];
      size_t na = 2;
      IS(NODE_DEPENDENTS(a, after, ae, &na), cudaSuccess);
      check(na == 1 && after[0] == b && ae[0].from_port == c.e.from_port && ae[0].type == c.e.type,
            "and so does cudaGraphNodeGetDependentNodes");
      // The same edge again is invalid, with whatever data.
      IS(ADD_DEPS(g, &a, &b, &c.e, 1), cudaErrorInvalidValue);
      cudaGraphEdgeData plain = E(0, 0, 0);
      IS(ADD_DEPS(g, &a, &b, &plain, 1), cudaErrorInvalidValue);
      // Removing an edge does not look at the data given: any data names the edge.
      IS(REMOVE_DEPS(g, &a, &b, &plain, 1), cudaSuccess);
      check(!edge_is(g, a, b, c.e.from_port, c.e.to_port, c.e.type), "the edge is gone");
      IS(REMOVE_DEPS(g, &a, &b, &c.e, 1), cudaErrorInvalidValue);
    }
  }
  // A source port other than the full completion needs a kernel upstream.
  {
    cudaGraphNode_t k = kernel_node(g), m = memset_node(g), em = empty_node(g), k2 = kernel_node(g), k3 = kernel_node(g);
    cudaGraphEdgeData lc = E(2, 0, 0);
    IS(ADD_DEPS(g, &m, &k2, &lc, 1), cudaErrorInvalidValue);
    IS(ADD_DEPS(g, &em, &k2, &lc, 1), cudaErrorInvalidValue);
    IS(ADD_DEPS(g, &k, &m, &lc, 1), cudaSuccess);
    IS(ADD_DEPS(g, &k, &em, &lc, 1), cudaSuccess);
    // Several edges at once: one invalid and the call is refused.
    cudaGraphNode_t from[2] = {k3, m}, to[2] = {k2, k2};
    cudaGraphEdgeData two[2] = {E(0, 0, 0), E(2, 0, 0)};
    IS(ADD_DEPS(g, from, to, two, 2), cudaErrorInvalidValue);
    check(!edge_is(g, k3, k2, 0, 0, 0), "a list with an invalid edge adds none of its edges");
    // Without any edge data, edges are plain ones.
    IS(ADD_DEPS(g, from, to, nullptr, 1), cudaSuccess);
    check(edge_is(g, k3, k2, 0, 0, 0), "a null edge-data array means plain edges");
  }
  // cudaGraphAddNode takes edge data, and refuses a node with an edge it cannot take.
#if CUDART_VERSION >= 13000
  {
    cudaGraphNode_t k = kernel_node(g), n = nullptr;
    alignas(cudaGraphNodeParams) unsigned char buf[sizeof(cudaGraphNodeParams)];
    std::memset(buf, 0, sizeof buf);
    cudaGraphNodeParams* np = reinterpret_cast<cudaGraphNodeParams*>(buf);
    np->type = cudaGraphNodeTypeEmpty;
    cudaGraphEdgeData lc = E(2, 0, 0), bad = E(0, 0, 2);
    IS(cudaGraphAddNode(&n, g, &k, &lc, 1, np), cudaSuccess);
    check(edge_is(g, k, n, 2, 0, 0), "an edge given to cudaGraphAddNode is kept");
    IS(cudaGraphAddNode(&n, g, &k, &bad, 1, np), cudaErrorInvalidValue);
    IS(cudaGraphAddNode(&n, g, &k, nullptr, 1, np), cudaSuccess);
  }
#endif
  // Clones and child graphs keep the edge data.
  {
    cudaGraph_t h;
    cudaGraphCreate(&h, 0);
    cudaGraphNode_t a = kernel_node(h), b = kernel_node(h);
    cudaGraphEdgeData lc = E(2, 0, 0);
    IS(ADD_DEPS(h, &a, &b, &lc, 1), cudaSuccess);
    cudaGraph_t clone;
    IS(cudaGraphClone(&clone, h), cudaSuccess);
    size_t n = 0;
    GET_EDGES(clone, nullptr, nullptr, nullptr, &n);
    cudaGraphNode_t f[1], t[1];
    cudaGraphEdgeData e[1];
    n = 1;
    IS(GET_EDGES(clone, f, t, e, &n), cudaSuccess);
    check(n == 1 && e[0].from_port == 2 && e[0].type == 0, "a clone keeps the edge data");
    cudaGraph_t parent;
    cudaGraphCreate(&parent, 0);
    cudaGraphNode_t child;
    IS(cudaGraphAddChildGraphNode(&child, parent, nullptr, 0, h), cudaSuccess);
    cudaGraph_t inner;
    IS(cudaGraphChildGraphNodeGetGraph(child, &inner), cudaSuccess);
    n = 1;
    IS(GET_EDGES(inner, f, t, e, &n), cudaSuccess);
    check(n == 1 && e[0].from_port == 2, "a child graph keeps it too");
    cudaGraphDestroy(clone);
    cudaGraphDestroy(parent);
    cudaGraphDestroy(h);
  }
  cudaGraphDestroy(g);
}

// A launch with programmatic stream serialization is captured with a programmatic edge, though the
// edge cannot be added by hand on a device below compute capability 9.
static void captured_edges() {
  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  inc<<<1, 1, 0, s>>>(g_dev);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1);
  cfg.blockDim = dim3(1);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  IS(cudaLaunchKernelEx(&cfg, inc, g_dev), cudaSuccess);
  inc<<<1, 1, 0, s>>>(g_dev);
  cudaGraph_t g = nullptr;
  IS(cudaStreamEndCapture(s, &g), cudaSuccess);
  size_t n = 0;
  GET_EDGES(g, nullptr, nullptr, nullptr, &n);
  std::vector<cudaGraphNode_t> f(n), t(n);
  std::vector<cudaGraphEdgeData> e(n);
  GET_EDGES(g, f.data(), t.data(), e.data(), &n);
  check(n == 2, "three captured kernels make two edges");
  int programmatic = 0, plain = 0;
  for (size_t i = 0; i < n; ++i) {
    if (e[i].from_port == 1 && e[i].type == 1) ++programmatic;
    if (e[i].from_port == 0 && e[i].type == 0) ++plain;
  }
  check(programmatic == 1 && plain == 1, "the launch that asked for it has a programmatic edge, the other a plain one");
  // The graph runs.
  cudaMemset(g_dev, 0, 4);
  cudaGraphExec_t ge;
  IS(cudaGraphInstantiate(&ge, g, 0), cudaSuccess);
  IS(cudaGraphLaunch(ge, s), cudaSuccess);
  IS(cudaStreamSynchronize(s), cudaSuccess);
  int v = 0;
  cudaMemcpy(&v, g_dev, 4, cudaMemcpyDeviceToHost);
  check(v == 3, "and runs all three kernels");
  cudaGraphExecDestroy(ge);
  // What a capture can be told about its edges is checked like a hand-made edge.
  cudaGraph_t into;
  cudaGraphCreate(&into, 0);
  cudaGraphNode_t seed = kernel_node(into);
  cudaGraphEdgeData bad = E(0, 0, 1), lc = E(2, 0, 0);
  IS(cudaStreamBeginCaptureToGraph(s, into, &seed, &bad, 1, cudaStreamCaptureModeGlobal), cc_major() >= 9 ? cudaSuccess : cudaErrorInvalidValue);
  if (cc_major() >= 9) {
    cudaGraph_t out;
    cudaStreamEndCapture(s, &out);
  }
  IS(cudaStreamBeginCaptureToGraph(s, into, &seed, &lc, 1, cudaStreamCaptureModeGlobal), cudaSuccess);
#if CUDART_VERSION >= 13000
  IS(cudaStreamUpdateCaptureDependencies(s, &seed, &bad, 1, cudaStreamSetCaptureDependencies), cc_major() >= 9 ? cudaSuccess : cudaErrorInvalidValue);
  IS(cudaStreamUpdateCaptureDependencies(s, &seed, &lc, 1, cudaStreamSetCaptureDependencies), cudaSuccess);
#else
  IS(cudaStreamUpdateCaptureDependencies_v2(s, &seed, &bad, 1, cudaStreamSetCaptureDependencies), cc_major() >= 9 ? cudaSuccess : cudaErrorInvalidValue);
  IS(cudaStreamUpdateCaptureDependencies_v2(s, &seed, &lc, 1, cudaStreamSetCaptureDependencies), cudaSuccess);
#endif
  inc<<<1, 1, 0, s>>>(g_dev);
  cudaGraph_t out = nullptr;
  IS(cudaStreamEndCapture(s, &out), cudaSuccess);
  cudaGraphNode_t nodes[4];
  size_t nn = 4;
  cudaGraphGetNodes(out, nodes, &nn);
  size_t k = 0;
  GET_EDGES(out, nullptr, nullptr, nullptr, &k);
  std::vector<cudaGraphNode_t> kf(k), kt(k);
  std::vector<cudaGraphEdgeData> ke(k);
  GET_EDGES(out, kf.data(), kt.data(), ke.data(), &k);
  check(k == 1 && ke[0].from_port == 2 && ke[0].type == 0, "the edge a capture was told to use is the one it makes");
  cudaGraphDestroy(g);
  cudaGraphDestroy(into);
  cudaStreamDestroy(s);
}
#endif   // CUDART_VERSION >= 12030

// ---- CUDA arrays in copy nodes ----
static void array_copies() {
  const cudaChannelFormatDesc fd = cudaCreateChannelDesc<float>();
  cudaArray_t arr;
  cudaMallocArray(&arr, &fd, 8, 4);
  float* d = nullptr;
  cudaMalloc(&d, 1024);
  float h[32], r[32];
  for (int i = 0; i < 32; ++i) {
    h[i] = static_cast<float>(i);
    r[i] = -1;
  }
  cudaStream_t s;
  cudaStreamCreate(&s);
  struct Op {
    const char* name;
    bool to_array;
    cudaError_t (*fn)(cudaStream_t, cudaArray_t, float*);
  };
  const Op ops[] = {
      {"cudaMemcpy2DToArrayAsync", true,
       [](cudaStream_t s, cudaArray_t a, float* d) { return cudaMemcpy2DToArrayAsync(a, 0, 0, d, 32, 32, 4, cudaMemcpyDeviceToDevice, s); }},
      {"cudaMemcpyToArrayAsync", true,
       [](cudaStream_t s, cudaArray_t a, float* d) { return cudaMemcpyToArrayAsync(a, 0, 0, d, 128, cudaMemcpyDeviceToDevice, s); }},
      {"cudaMemcpy2DFromArrayAsync", false,
       [](cudaStream_t s, cudaArray_t a, float* d) { return cudaMemcpy2DFromArrayAsync(d, 32, a, 0, 0, 32, 4, cudaMemcpyDeviceToDevice, s); }},
      {"cudaMemcpyFromArrayAsync", false,
       [](cudaStream_t s, cudaArray_t a, float* d) { return cudaMemcpyFromArrayAsync(d, a, 0, 0, 128, cudaMemcpyDeviceToDevice, s); }},
      {"cudaMemcpy3DAsync into an array", true,
       [](cudaStream_t s, cudaArray_t a, float* d) {
         cudaMemcpy3DParms p = {};
         p.srcPtr = make_cudaPitchedPtr(d, 32, 8, 4);
         p.dstArray = a;
         p.extent = make_cudaExtent(8, 4, 1);
         p.kind = cudaMemcpyDeviceToDevice;
         return cudaMemcpy3DAsync(&p, s);
       }},
      {"cudaMemcpy3DAsync out of an array", false,
       [](cudaStream_t s, cudaArray_t a, float* d) {
         cudaMemcpy3DParms p = {};
         p.dstPtr = make_cudaPitchedPtr(d, 32, 8, 4);
         p.srcArray = a;
         p.extent = make_cudaExtent(8, 4, 1);
         p.kind = cudaMemcpyDeviceToDevice;
         return cudaMemcpy3DAsync(&p, s);
       }},
  };
  for (const Op& op : ops) {
    cudaGraph_t g = nullptr;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
    const cudaError_t rc = op.fn(s, arr, d);
    const cudaError_t end = cudaStreamEndCapture(s, &g);
    char what[160];
    std::snprintf(what, sizeof what, "%s is captured as one node", op.name);
    size_t n = 0;
    if (g) cudaGraphGetNodes(g, nullptr, &n);
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    if (g && n == 1) {
      cudaGraphNode_t node;
      cudaGraphGetNodes(g, &node, &n);
      cudaGraphNodeGetType(node, &type);
    }
    check(rc == cudaSuccess && end == cudaSuccess && n == 1 && type == cudaGraphNodeTypeMemcpy, what);
    if (!g) continue;
    // Replay with other contents than at capture: the array and the buffer are read when the graph runs.
    cudaGraphExec_t ge;
    cudaGraphInstantiate(&ge, g, 0);
    float fresh[32];
    for (int i = 0; i < 32; ++i) fresh[i] = 100.0f + i;
    if (op.to_array) {
      cudaMemcpy(d, fresh, 128, cudaMemcpyHostToDevice);
      cudaMemset(d + 64, 0, 4);
      IS(cudaGraphLaunch(ge, s), cudaSuccess);
      cudaStreamSynchronize(s);
      cudaMemcpy2DFromArray(r, 32, arr, 0, 0, 32, 4, cudaMemcpyDeviceToHost);
      std::snprintf(what, sizeof what, "replaying %s copies what the buffer holds now", op.name);
      check(r[0] == 100.0f && r[7] == 107.0f && r[31] == 131.0f, what);
    } else {
      cudaMemcpy2DToArray(arr, 0, 0, fresh, 32, 32, 4, cudaMemcpyHostToDevice);
      cudaMemset(d, 0, 128);
      IS(cudaGraphLaunch(ge, s), cudaSuccess);
      cudaStreamSynchronize(s);
      cudaMemcpy(r, d, 128, cudaMemcpyDeviceToHost);
      std::snprintf(what, sizeof what, "replaying %s copies what the array holds now", op.name);
      check(r[0] == 100.0f && r[8] == 108.0f && r[31] == 131.0f, what);
    }
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaGetLastError();
  }
  // Nodes built by hand.
  {
    cudaGraph_t g;
    cudaGraphCreate(&g, 0);
    cudaGraphNode_t n;
    cudaMemcpy3DParms p = {};
    p.srcPtr = make_cudaPitchedPtr(d, 32, 8, 4);
    p.dstArray = arr;
    p.extent = make_cudaExtent(8, 4, 1);
    p.kind = cudaMemcpyDeviceToDevice;
    IS(cudaGraphAddMemcpyNode(&n, g, nullptr, 0, &p), cudaSuccess);
    cudaMemcpy3DParms got;
    std::memset(&got, 0xff, sizeof got);
    IS(cudaGraphMemcpyNodeGetParams(n, &got), cudaSuccess);
    check(got.dstArray == arr && got.srcPtr.ptr == d && got.extent.width == 8 && got.extent.height == 4,
          "the node hands its parameters back, the array included");
    cudaGraphExec_t ge;
    IS(cudaGraphInstantiate(&ge, g, 0), cudaSuccess);
    cudaMemcpy(d, h, 128, cudaMemcpyHostToDevice);
    IS(cudaGraphLaunch(ge, s), cudaSuccess);
    cudaStreamSynchronize(s);
    cudaMemcpy2DFromArray(r, 32, arr, 0, 0, 32, 4, cudaMemcpyDeviceToHost);
    check(r[0] == 0 && r[9] == 9 && r[31] == 31, "the graph copies into the array");
    IS(cudaGraphExecMemcpyNodeSetParams(ge, n, &p), cudaSuccess);
    IS(cudaGraphMemcpyNodeSetParams(n, &p), cudaSuccess);
    cudaMemcpy3DParms both = p;
    both.srcArray = arr;   // a pointer and an array on one side
    IS(cudaGraphAddMemcpyNode(&n, g, nullptr, 0, &both), cudaErrorInvalidValue);
    cudaMemcpy3DParms aa = {};
    aa.srcArray = arr;
    aa.dstArray = arr;
    aa.extent = make_cudaExtent(8, 4, 1);
    aa.kind = cudaMemcpyDeviceToDevice;
    IS(cudaGraphAddMemcpyNode(&n, g, nullptr, 0, &aa), cudaSuccess);
    cudaMemcpy3DParms off = {};
    off.srcPtr = make_cudaPitchedPtr(d, 32, 8, 4);
    off.dstArray = arr;
    off.dstPos = make_cudaPos(4, 1, 0);
    off.extent = make_cudaExtent(4, 2, 1);
    off.kind = cudaMemcpyDeviceToDevice;
    IS(cudaGraphAddMemcpyNode(&n, g, nullptr, 0, &off), cudaSuccess);
    off.dstPos = make_cudaPos(6, 3, 0);
    IS(cudaGraphAddMemcpyNode(&n, g, nullptr, 0, &off), cudaErrorInvalidValue);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
  }
  cudaStreamDestroy(s);
  cudaFree(d);
  cudaFreeArray(arr);
}

// ---- a child graph moved into its parent ----
#if CUDART_VERSION >= 13000
static void child_ownership() {
  alignas(cudaGraphNodeParams) unsigned char buf[sizeof(cudaGraphNodeParams)];
  std::memset(buf, 0, sizeof buf);
  cudaGraphNodeParams* np = reinterpret_cast<cudaGraphNodeParams*>(buf);
  np->type = cudaGraphNodeTypeGraph;
  np->graph.ownership = cudaGraphChildGraphOwnershipMove;

  cudaGraph_t child, parent;
  cudaGraphCreate(&child, 0);
  empty_node(child);
  cudaGraphCreate(&parent, 0);
  np->graph.graph = child;
  cudaGraphNode_t node;
  IS(cudaGraphAddNode(&node, parent, nullptr, nullptr, 0, np), cudaSuccess);
  cudaGraph_t seen = nullptr;
  IS(cudaGraphChildGraphNodeGetGraph(node, &seen), cudaSuccess);
  check(seen == child, "the graph that was moved is the node's graph: same handle");
  size_t n = 0;
  IS(cudaGraphGetNodes(child, nullptr, &n), cudaSuccess);
  check(n == 1, "and still readable through it");
  IS(cudaGraphDestroy(child), cudaErrorInvalidValue);   // the node owns it now
  // Changes through the handle are changes to the parent's child.
  empty_node(child);
  IS(cudaGraphGetNodes(child, nullptr, &n), cudaSuccess);
  check(n == 2, "a node added through the handle is in the child");
  cudaGraphExec_t ge;
  IS(cudaGraphInstantiate(&ge, parent, 0), cudaSuccess);
  IS(cudaGraphInstantiate(&ge, child, 0), cudaErrorNotSupported);   // a child cannot be instantiated alone
  cudaGraphExecDestroy(ge);
  // It cannot go into another parent, cloned or moved.
  cudaGraph_t other;
  cudaGraphCreate(&other, 0);
  cudaGraphNode_t n2;
  IS(cudaGraphAddChildGraphNode(&n2, other, nullptr, 0, child), cudaErrorNotSupported);
  np->graph.graph = child;
  IS(cudaGraphAddNode(&n2, other, nullptr, nullptr, 0, np), cudaErrorNotSupported);
  cudaGraph_t cl;
  IS(cudaGraphClone(&cl, child), cudaSuccess);   // but it can be cloned
  cudaGraphDestroy(cl);
  // A graph that is a child by cloning is no different.
  cudaGraph_t source, cloned_parent;
  cudaGraphCreate(&source, 0);
  empty_node(source);
  cudaGraphCreate(&cloned_parent, 0);
  IS(cudaGraphAddChildGraphNode(&n2, cloned_parent, nullptr, 0, source), cudaSuccess);
  cudaGraph_t inner;
  IS(cudaGraphChildGraphNodeGetGraph(n2, &inner), cudaSuccess);
  check(inner != source, "a cloned child is a graph of its own");
  IS(cudaGraphAddChildGraphNode(&n2, other, nullptr, 0, inner), cudaErrorNotSupported);
  // The parent's end is the child's (the handle is not used again: what a freed graph answers is not defined).
  IS(cudaGraphDestroy(parent), cudaSuccess);
  cudaGraphDestroy(cloned_parent);
  cudaGraphDestroy(source);
  cudaGraphDestroy(other);
}
#endif

// ---- external memory and semaphores ----
static void external_interop() {
  cudaExternalSemaphore_t sem = nullptr;
  cudaExternalSemaphoreHandleDesc sd;
  std::memset(&sd, 0, sizeof sd);
  sd.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  sd.handle.fd = -1;
  // A valid description of a semaphore no other API has made: not supported, not an invalid value.
  IS(cudaImportExternalSemaphore(&sem, &sd), cudaErrorNotSupported);
  sd.handle.fd = 99;
  IS(cudaImportExternalSemaphore(&sem, &sd), cudaErrorNotSupported);
  sd.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  IS(cudaImportExternalSemaphore(&sem, &sd), cudaErrorNotSupported);
  sd.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
  IS(cudaImportExternalSemaphore(&sem, &sd), cudaErrorNotSupported);
  sd.type = cudaExternalSemaphoreHandleTypeKeyedMutex;
  IS(cudaImportExternalSemaphore(&sem, &sd), cudaErrorNotSupported);
  sd.type = static_cast<cudaExternalSemaphoreHandleType>(99);
  IS(cudaImportExternalSemaphore(&sem, &sd), cudaErrorInvalidValue);
  sd.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  sd.flags = 1;
  IS(cudaImportExternalSemaphore(&sem, &sd), cudaErrorInvalidValue);
  sd.flags = 0;
  IS(cudaImportExternalSemaphore(nullptr, &sd), cudaErrorInvalidValue);
  IS(cudaImportExternalSemaphore(&sem, nullptr), cudaErrorInvalidValue);
  check(sem == nullptr, "no import made a semaphore");

  cudaExternalMemory_t em = nullptr;
  cudaExternalMemoryHandleDesc md;
  std::memset(&md, 0, sizeof md);
  md.type = cudaExternalMemoryHandleTypeOpaqueFd;
  md.handle.fd = -1;
  md.size = 4096;
  IS(cudaImportExternalMemory(&em, &md), cudaErrorNotSupported);
  md.size = 0;
  md.handle.fd = 3;
  IS(cudaImportExternalMemory(&em, &md), cudaErrorInvalidValue);
  md.size = 4096;
  md.type = static_cast<cudaExternalMemoryHandleType>(99);
  IS(cudaImportExternalMemory(&em, &md), cudaErrorInvalidValue);
  IS(cudaImportExternalMemory(nullptr, &md), cudaErrorInvalidValue);
  check(em == nullptr, "no import made memory");

  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaExternalSemaphore_t none[1] = {nullptr};
  cudaExternalSemaphoreSignalParams sp;
  std::memset(&sp, 0, sizeof sp);
  IS(cudaSignalExternalSemaphoresAsync(none, &sp, 0, s), cudaErrorInvalidValue);
  IS(cudaSignalExternalSemaphoresAsync(nullptr, &sp, 1, s), cudaErrorInvalidValue);
  IS(cudaSignalExternalSemaphoresAsync(none, nullptr, 1, s), cudaErrorInvalidValue);
  cudaExternalSemaphoreWaitParams wp;
  std::memset(&wp, 0, sizeof wp);
  IS(cudaWaitExternalSemaphoresAsync(none, &wp, 0, s), cudaErrorInvalidValue);
  IS(cudaDestroyExternalSemaphore(nullptr), cudaErrorInvalidValue);
  IS(cudaDestroyExternalMemory(nullptr), cudaErrorInvalidValue);
  void* dp = nullptr;
  cudaExternalMemoryBufferDesc bd;
  std::memset(&bd, 0, sizeof bd);
  IS(cudaExternalMemoryGetMappedBuffer(&dp, nullptr, &bd), cudaErrorInvalidValue);
  cudaMipmappedArray_t mm = nullptr;
  cudaExternalMemoryMipmappedArrayDesc ad;
  std::memset(&ad, 0, sizeof ad);
  IS(cudaExternalMemoryGetMappedMipmappedArray(&mm, nullptr, &ad), cudaErrorInvalidChannelDescriptor);
  // GPUDirect RDMA is not there.
  IS(cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice, cudaFlushGPUDirectRDMAWritesToOwner),
    cudaErrorNotSupported);

  // The nodes: made with the arrays given, handed back from the node's own copy.
  cudaGraph_t g;
  cudaGraphCreate(&g, 0);
  cudaExternalSemaphore_t arr[2] = {nullptr, nullptr};
  cudaExternalSemaphoreSignalParams ps[2];
  std::memset(ps, 0, sizeof ps);
  ps[0].params.fence.value = 5;
  cudaExternalSemaphoreSignalNodeParams snp;
  std::memset(&snp, 0, sizeof snp);
  snp.extSemArray = arr;
  snp.paramsArray = ps;
  snp.numExtSems = 2;
  cudaGraphNode_t sn, wn;
  IS(cudaGraphAddExternalSemaphoresSignalNode(&sn, g, nullptr, 0, &snp), cudaSuccess);
  cudaGraphNodeType type;
  IS(cudaGraphNodeGetType(sn, &type), cudaSuccess);
  check(type == cudaGraphNodeTypeExtSemaphoreSignal, "a signal node has its own type");
  cudaExternalSemaphoreSignalNodeParams got;
  std::memset(&got, 0xff, sizeof got);
  IS(cudaGraphExternalSemaphoresSignalNodeGetParams(sn, &got), cudaSuccess);
  check(got.numExtSems == 2 && got.paramsArray[0].params.fence.value == 5 && got.extSemArray[0] == nullptr,
        "the node hands the parameters back");
  cudaExternalSemaphoreWaitParams pw[1];
  std::memset(pw, 0, sizeof pw);
  cudaExternalSemaphore_t aw[1] = {nullptr};
  cudaExternalSemaphoreWaitNodeParams wnp;
  std::memset(&wnp, 0, sizeof wnp);
  wnp.extSemArray = aw;
  wnp.paramsArray = pw;
  wnp.numExtSems = 1;
  IS(cudaGraphAddExternalSemaphoresWaitNode(&wn, g, &sn, 1, &wnp), cudaSuccess);
  IS(cudaGraphExternalSemaphoresWaitNodeSetParams(wn, &wnp), cudaSuccess);
  IS(cudaGraphExternalSemaphoresSignalNodeSetParams(sn, &snp), cudaSuccess);
  IS(cudaGraphExternalSemaphoresSignalNodeSetParams(wn, &snp), cudaErrorInvalidValue);   // a wait node is not a signal node
  cudaExternalSemaphoreWaitNodeParams gw;
  IS(cudaGraphExternalSemaphoresWaitNodeGetParams(sn, &gw), cudaErrorInvalidValue);
  IS(cudaGraphAddExternalSemaphoresSignalNode(&sn, g, nullptr, 0, nullptr), cudaErrorInvalidValue);
  IS(cudaGraphAddExternalSemaphoresSignalNode(nullptr, g, nullptr, 0, &snp), cudaErrorInvalidValue);
  cudaGraphExec_t ge;
  IS(cudaGraphInstantiate(&ge, g, 0), cudaSuccess);
  IS(cudaGraphExecExternalSemaphoresSignalNodeSetParams(ge, sn, &snp), cudaSuccess);
  IS(cudaGraphExecExternalSemaphoresWaitNodeSetParams(ge, wn, &wnp), cudaSuccess);
  cudaGraph_t cl;
  IS(cudaGraphClone(&cl, g), cudaSuccess);
  size_t n = 0;
  cudaGraphGetNodes(cl, nullptr, &n);
  check(n == 2, "a clone has the nodes");
  cudaGraphDestroy(cl);
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(g);
  cudaStreamDestroy(s);
}

// ---- kernel node attributes ----
static void kernel_attributes() {
  cudaGraph_t g;
  cudaGraphCreate(&g, 0);
  cudaGraphNode_t a = kernel_node(g), b = kernel_node(g), e = empty_node(g);
  cudaKernelNodeAttrValue v;
  std::memset(&v, 0, sizeof v);
  // Reading an attribute that was never set: zeros, for the attributes a node has.
  std::memset(&v, 0xee, sizeof v);
  IS(cudaGraphKernelNodeGetAttribute(a, cudaLaunchAttributeCooperative, &v), cudaSuccess);
  check(v.cooperative == 0, "an attribute never set reads as zero");
  v.cooperative = 1;
  IS(cudaGraphKernelNodeSetAttribute(a, cudaLaunchAttributeCooperative, &v), cudaSuccess);
  std::memset(&v, 0xee, sizeof v);
  IS(cudaGraphKernelNodeGetAttribute(a, cudaLaunchAttributeCooperative, &v), cudaSuccess);
  check(v.cooperative == 1, "and as set once it is");
  std::memset(&v, 0, sizeof v);
  v.priority = 3;
  IS(cudaGraphKernelNodeSetAttribute(a, cudaLaunchAttributePriority, &v), cudaSuccess);
  IS(cudaGraphKernelNodeCopyAttributes(b, a), cudaSuccess);
  std::memset(&v, 0xee, sizeof v);
  IS(cudaGraphKernelNodeGetAttribute(b, cudaLaunchAttributeCooperative, &v), cudaSuccess);
  check(v.cooperative == 1, "cudaGraphKernelNodeCopyAttributes copies them");
  // Other nodes have none.
  IS(cudaGraphKernelNodeSetAttribute(e, cudaLaunchAttributeCooperative, &v), cudaErrorInvalidValue);
  IS(cudaGraphKernelNodeGetAttribute(e, cudaLaunchAttributeCooperative, &v), cudaErrorInvalidValue);
  IS(cudaGraphKernelNodeCopyAttributes(e, a), cudaErrorInvalidValue);
  IS(cudaGraphKernelNodeSetAttribute(nullptr, cudaLaunchAttributeCooperative, &v), cudaErrorInvalidValue);
  if (cc_major() < 9) {
    // Programmatic launch is Hopper's: below it these cannot be set or read.
    std::memset(&v, 0, sizeof v);
    v.programmaticStreamSerializationAllowed = 1;
    IS(cudaGraphKernelNodeSetAttribute(a, cudaLaunchAttributeProgrammaticStreamSerialization, &v), cudaErrorInvalidValue);
    IS(cudaGraphKernelNodeGetAttribute(a, cudaLaunchAttributeProgrammaticStreamSerialization, &v), cudaErrorInvalidValue);
    std::memset(&v, 0, sizeof v);
    v.clusterDim.x = v.clusterDim.y = v.clusterDim.z = 1;
    IS(cudaGraphKernelNodeSetAttribute(a, cudaLaunchAttributeClusterDimension, &v), cudaErrorInvalidClusterSize);
    IS(cudaGraphKernelNodeGetAttribute(a, cudaLaunchAttributeClusterDimension, &v), cudaSuccess);
  }
  cudaGraphDestroy(g);
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (cudaFree(nullptr) != cudaSuccess) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  cudaMalloc(&g_dev, 64);
  cudaMemset(g_dev, 0, 64);
#if CUDART_VERSION >= 12030
  edge_data();
  captured_edges();
#endif
  array_copies();
#if CUDART_VERSION >= 13000
  child_ownership();
#endif
  external_interop();
  kernel_attributes();
  cudaFree(g_dev);
  std::printf(failures ? "FAIL: %d runtime checks\n" : "PASS: every runtime check\n", failures);
  return failures ? 1 : 0;
}
