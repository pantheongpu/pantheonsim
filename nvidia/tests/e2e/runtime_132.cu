// The runtime functions only the CUDA 13.2 header declares (runtime_132.inc), called through the loader so the
// program builds with every toolkit and SKIPs when the runtime shim was built with an older one (they exist only in a
// shim built with 13.2 headers). NVIDIA's CUDA 13.2 libcudart crashes on its first call on the RTX 3060 under WSL
// the other cases were measured on, so these have no card oracle: each is checked against its driver twin, whose answers
// the [13.2] cases of exports_sweep_driver.cpp took from the card, and against what the header documents.
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstring>

__global__ void k_none() {}
__global__ void k_three(int* a, float b, double* c) { a[0] = static_cast<int>(b); c[0] = 1.0; }
__global__ void k_one(int* a) { a[0] = 7; }

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("%s: %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++g_fail;
}
template <class F>
static F sym(const char* n) {
  return reinterpret_cast<F>(dlsym(RTLD_DEFAULT, n));
}
struct NodeParams {   // cudaGraphNodeParams: a type and a union of 232 bytes
  int type;
  int reserved0[3];
  unsigned char u[232];
  long long reserved2;
  template <class T> T get(size_t off) const { T v; std::memcpy(&v, u + off, sizeof v); return v; }
};

int main() {
  cudaFree(0);
  auto param_count = sym<cudaError_t (*)(const void*, size_t*)>("cudaFuncGetParamCount");
  if (!param_count) {
    std::printf("SKIP: the runtime was built with a toolkit older than 13.2\n");
    return 0;
  }
  // ---- parameter counts
  size_t n = 99;
  check(param_count((const void*)k_three, &n) == cudaSuccess && n == 3, "cudaFuncGetParamCount: three");
  check(param_count((const void*)k_none, &n) == cudaSuccess && n == 0, "cudaFuncGetParamCount: none");
  check(param_count((const void*)k_one, &n) == cudaSuccess && n == 1, "cudaFuncGetParamCount: one");
  check(param_count((const void*)k_one, nullptr) == cudaErrorInvalidValue, "cudaFuncGetParamCount: null result");
  check(param_count(nullptr, &n) != cudaSuccess, "cudaFuncGetParamCount: null function");

  // ---- a host function with a synchronization mode
  using HostV2 = cudaError_t (*)(cudaStream_t, cudaHostFn_t, void*, unsigned);
  auto host_v2 = sym<HostV2>("cudaLaunchHostFunc_v2");
  static int ran;
  cudaStream_t s;
  cudaStreamCreate(&s);
  ran = 0;
  auto bump = [](void* u) { ++*static_cast<int*>(u); };
  check(host_v2 && host_v2(s, bump, &ran, 0) == cudaSuccess && (cudaStreamSynchronize(s), ran == 1), "cudaLaunchHostFunc_v2: blocking");
  check(host_v2(s, bump, &ran, 1) == cudaSuccess && (cudaStreamSynchronize(s), ran == 2), "cudaLaunchHostFunc_v2: spin wait");
  check(host_v2(s, bump, &ran, 2) == cudaErrorInvalidValue && ran == 2, "cudaLaunchHostFunc_v2: mode 2 is refused");
  check(host_v2(s, nullptr, nullptr, 0) == cudaSuccess, "cudaLaunchHostFunc_v2: a null function is accepted");

  // ---- one copy, with attributes
  struct Attr { int order; int a, b, c, d; unsigned flags; } at{};
  using CopyFn = cudaError_t (*)(void*, const void*, size_t, void*, cudaStream_t);
  auto copy = sym<CopyFn>("cudaMemcpyWithAttributesAsync");
  char *d1, *d2;
  char h[256], o[256];
  for (int i = 0; i < 256; ++i) h[i] = static_cast<char>(i * 3);
  cudaMalloc(&d1, 256);
  cudaMalloc(&d2, 256);
  cudaMemcpy(d1, h, 256, cudaMemcpyHostToDevice);
  at.order = 1;
  check(copy && copy(d2, d1, 256, &at, s) == cudaSuccess, "cudaMemcpyWithAttributesAsync");
  cudaStreamSynchronize(s);
  cudaMemcpy(o, d2, 256, cudaMemcpyDeviceToHost);
  check(std::memcmp(h, o, 256) == 0, "cudaMemcpyWithAttributesAsync: copied right");
  check(copy(d2, d1, 256, nullptr, s) == cudaErrorInvalidValue, "cudaMemcpyWithAttributesAsync: null attributes");
  check(copy(d2, d1, 0, &at, s) == cudaErrorInvalidValue, "cudaMemcpyWithAttributesAsync: size 0");
  at.order = 9;
  check(copy(d2, d1, 16, &at, s) == cudaErrorInvalidValue, "cudaMemcpyWithAttributesAsync: access order 9");
  at.order = 1;
  check(copy(d2, d1, 16, &at, nullptr) == cudaErrorInvalidValue, "cudaMemcpyWithAttributesAsync: legacy stream");

  // ---- ids of graphs, executables and nodes, against the driver's
  using GraphId = cudaError_t (*)(cudaGraph_t, unsigned*);
  using ExecId = cudaError_t (*)(cudaGraphExec_t, unsigned*);
  using NodeId = cudaError_t (*)(cudaGraphNode_t, unsigned*);
  using ToolsId = cudaError_t (*)(cudaGraphNode_t, unsigned long long*);
  using Containing = cudaError_t (*)(cudaGraphNode_t, cudaGraph_t*);
  auto graph_id = sym<GraphId>("cudaGraphGetId");
  auto exec_id = sym<ExecId>("cudaGraphExecGetId");
  auto local_id = sym<NodeId>("cudaGraphNodeGetLocalId");
  auto tools_id = sym<ToolsId>("cudaGraphNodeGetToolsId");
  auto containing = sym<Containing>("cudaGraphNodeGetContainingGraph");
  cudaGraph_t g1, g2, child;
  cudaGraphCreate(&g1, 0);
  cudaGraphCreate(&g2, 0);
  unsigned a = 9999, b = 9999, via_driver = 9999;
  check(graph_id(g1, &a) == cudaSuccess && graph_id(g2, &b) == cudaSuccess && b == a + 1, "cudaGraphGetId: one id for each graph made");
  auto drv_graph_id = sym<CUresult (*)(CUgraph, unsigned*)>("cuGraphGetId");
  check(drv_graph_id && drv_graph_id(reinterpret_cast<CUgraph>(g1), &via_driver) == CUDA_SUCCESS && via_driver == a, "cudaGraphGetId: the driver's cuGraphGetId says the same");
  check(graph_id(g1, nullptr) == cudaErrorInvalidValue && graph_id(nullptr, &a) == cudaErrorInvalidValue, "cudaGraphGetId: null arguments");

  cudaGraphNode_t n1, n2, n3, in;
  cudaGraphAddEmptyNode(&n1, g1, nullptr, 0);
  cudaGraphAddEmptyNode(&n2, g1, &n1, 1);
  cudaGraphAddEmptyNode(&n3, g1, &n2, 1);
  unsigned l1 = 9999, l2 = 9999, l3 = 9999;
  check(local_id(n1, &l1) == cudaSuccess && local_id(n2, &l2) == cudaSuccess && local_id(n3, &l3) == cudaSuccess && l1 == 0 && l2 == 1 && l3 == 2,
        "cudaGraphNodeGetLocalId: the index in the graph");
  check(local_id(nullptr, &l1) == cudaErrorInvalidValue && local_id(n1, nullptr) == cudaErrorInvalidValue, "cudaGraphNodeGetLocalId: null arguments");
  unsigned long long t1 = 0, t2 = 0;
  check(tools_id(n1, &t1) == cudaSuccess && tools_id(n2, &t2) == cudaSuccess && t1 != t2 && (t1 >> 32) == a && (t1 & 0xffffffffu) == 0 && (t2 & 0xffffffffu) == 1,
        "cudaGraphNodeGetToolsId: the graph's id over the node's index");
  cudaGraph_t got = nullptr;
  check(containing(n3, &got) == cudaSuccess && got == g1, "cudaGraphNodeGetContainingGraph: the graph");
  cudaGraphCreate(&child, 0);
  cudaGraphAddEmptyNode(&in, child, nullptr, 0);
  cudaGraphNode_t cn;
  cudaGraphAddChildGraphNode(&cn, g2, nullptr, 0, child);
  cudaGraph_t got_in = nullptr, got_cn = nullptr;
  check(containing(in, &got_in) == cudaSuccess && got_in == child && containing(cn, &got_cn) == cudaSuccess && got_cn == g2,
        "cudaGraphNodeGetContainingGraph: a child graph's node and the node holding it");
  cudaGraphExec_t e1, e2;
  cudaGraphInstantiateWithFlags(&e1, g1, 0);
  cudaGraphInstantiateWithFlags(&e2, g1, 0);
  unsigned x1 = 9999, x2 = 9999;
  check(exec_id(e1, &x1) == cudaSuccess && exec_id(e2, &x2) == cudaSuccess && x1 != x2 && x2 == x1 + 2, "cudaGraphExecGetId: two ids for each instantiation");
  check(exec_id(nullptr, &x1) == cudaErrorInvalidValue, "cudaGraphExecGetId: null executable");

  // ---- the parameters of a node
  using GetParams = cudaError_t (*)(cudaGraphNode_t, NodeParams*);
  auto get_params = sym<GetParams>("cudaGraphNodeGetParams");
  NodeParams p;
  std::memset(&p, 0, sizeof p);
  check(get_params(n1, &p) == cudaSuccess && p.type == cudaGraphNodeTypeEmpty, "cudaGraphNodeGetParams: an empty node");
  int* dev;
  cudaMalloc(&dev, 1024);
  cudaMemsetParams mp{};
  mp.dst = dev;
  mp.value = 7;
  mp.elementSize = 1;
  mp.width = 256;
  mp.height = 1;
  cudaGraphNode_t mn;
  cudaGraphAddMemsetNode(&mn, g1, nullptr, 0, &mp);
  std::memset(&p, 0, sizeof p);
  check(get_params(mn, &p) == cudaSuccess && p.type == cudaGraphNodeTypeMemset && p.get<void*>(0) == dev && p.get<unsigned>(16) == 7 && p.get<size_t>(24) == 256,
        "cudaGraphNodeGetParams: a memset node");
  cudaKernelNodeParams kp{};
  void* args[1] = {&dev};
  kp.func = (void*)k_one;
  kp.gridDim = dim3(2);
  kp.blockDim = dim3(32);
  kp.kernelParams = args;
  cudaGraphNode_t kn;
  cudaGraphAddKernelNode(&kn, g1, nullptr, 0, &kp);
  std::memset(&p, 0, sizeof p);
  check(get_params(kn, &p) == cudaSuccess && p.type == cudaGraphNodeTypeKernel && p.get<void*>(0) == (void*)k_one && p.get<unsigned>(8) == 2 && p.get<unsigned>(20) == 32,
        "cudaGraphNodeGetParams: a kernel node");
  static int x;
  cudaHostNodeParams hp{};
  hp.fn = [](void*) {};
  hp.userData = &x;
  cudaGraphNode_t hn;
  cudaGraphAddHostNode(&hn, g1, nullptr, 0, &hp);
  std::memset(&p, 0, sizeof p);
  check(get_params(hn, &p) == cudaSuccess && p.type == cudaGraphNodeTypeHost && p.get<void*>(8) == &x, "cudaGraphNodeGetParams: a host node");
  std::memset(&p, 0, sizeof p);
  check(get_params(cn, &p) == cudaSuccess && p.type == cudaGraphNodeTypeGraph && p.get<cudaGraph_t>(0) != nullptr && p.get<cudaGraph_t>(0) != child,
        "cudaGraphNodeGetParams: a child graph node holds a clone");
  check(get_params(nullptr, &p) == cudaErrorInvalidValue && get_params(n1, nullptr) == cudaErrorInvalidValue, "cudaGraphNodeGetParams: null arguments");

  // ---- the same conditional handle as the first form, for the current context only
  using CondV2 = cudaError_t (*)(unsigned long long*, cudaGraph_t, void*, unsigned, unsigned);
  auto cond_v2 = sym<CondV2>("cudaGraphConditionalHandleCreate_v2");
  unsigned long long handle = 0;
  check(cond_v2 && cond_v2(&handle, g1, reinterpret_cast<void*>(0x1234), 0, 0) == cudaErrorInvalidResourceHandle, "cudaGraphConditionalHandleCreate_v2: a context that is none");

  // ---- green contexts and device resources are not simulated
  using GetCtx = cudaError_t (*)(void**, int);
  auto get_ctx = sym<GetCtx>("cudaDeviceGetExecutionCtx");
  void* ctx = nullptr;
  check(get_ctx && get_ctx(&ctx, 0) == cudaErrorNotSupported, "cudaDeviceGetExecutionCtx: refused");
  using StreamRes = cudaError_t (*)(cudaStream_t, void*, int);
  auto stream_res = sym<StreamRes>("cudaStreamGetDevResource");
  char res[512];
  check(stream_res && stream_res(s, res, 1) == cudaErrorNotSupported, "cudaStreamGetDevResource: refused");
  using DevRes = cudaError_t (*)(int, void*, int);
  auto dev_res = sym<DevRes>("cudaDeviceGetDevResource");
  check(dev_res && dev_res(0, res, 1) == cudaErrorNotSupported, "cudaDeviceGetDevResource: refused");

  std::printf("%s\n", g_fail ? "FAIL" : "PASS");
  return g_fail != 0;
}
