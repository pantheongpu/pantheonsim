// The driver API's graphs: every cuGraph* call a program makes to build, read
// back, instantiate, launch and update a graph, stream capture through the
// driver, user objects and graph memory, with what an RTX 3060's driver
// answers for each error. The program passes against NVIDIA's libcuda on the
// card as well; where the simulator differs by design (a user object's
// destructor runs when the last reference goes, rather than some time after)
// the check says so.
//
// Built against whichever cuda.h the toolkit has: the calls newer than CUDA
// 12.0 are looked up with cuGetProcAddress and skipped when the driver
// does not have them.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <vector>

static int failures = 0;
static void check(bool ok, const char* what, int line) {
  if (!ok) {
    std::printf("FAIL line %d: %s\n", line, what);
    ++failures;
  }
}
#define CHECK(c) check((c), #c, __LINE__)
#define IS(call, want) do { const CUresult r_ = (call); if (r_ != (want)) { \
  std::printf("FAIL line %d: %s -> %d, expected %s (%d)\n", __LINE__, #call, (int)r_, #want, (int)(want)); ++failures; } } while (0)
#define OK(call) IS(call, CUDA_SUCCESS)

static const char* kPtx = R"(
.version 7.0
.target sm_80
.address_size 64
.visible .entry addp(.param .u64 p, .param .u32 v)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r1, [v];
  mov.u32 %r2, %tid.x;
  mul.wide.u32 %rd2, %r2, 4;
  add.u64 %rd3, %rd1, %rd2;
  ld.global.u32 %r3, [%rd3];
  add.u32 %r3, %r3, %r1;
  st.global.u32 [%rd3], %r3;
  ret;
}
.visible .entry mulp(.param .u64 p, .param .u32 v)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r1, [v];
  mov.u32 %r2, %tid.x;
  mul.wide.u32 %rd2, %r2, 4;
  add.u64 %rd3, %rd1, %rd2;
  ld.global.u32 %r3, [%rd3];
  mul.lo.u32 %r3, %r3, %r1;
  st.global.u32 [%rd3], %r3;
  ret;
}
)";

static void CUDA_CB bump(void* user) { ++*static_cast<int*>(user); }
static int destroyed = 0;
static void CUDA_CB note_destroyed(void*) { ++destroyed; }

constexpr int kN = 64;

static CUfunction add_fn, mul_fn;
static CUdeviceptr dbuf;

static std::vector<int> read_back() {
  std::vector<int> h(kN);
  cuMemcpyDtoH(h.data(), dbuf, kN * sizeof(int));
  return h;
}
static void fill(int v) {
  std::vector<int> h(kN, v);
  cuMemcpyHtoD(dbuf, h.data(), kN * sizeof(int));
}

static CUDA_KERNEL_NODE_PARAMS kernel_params(CUfunction f, void** args, unsigned block = kN) {
  CUDA_KERNEL_NODE_PARAMS p;
  std::memset(&p, 0, sizeof p);
  p.func = f;
  p.gridDimX = p.gridDimY = p.gridDimZ = 1;
  p.blockDimX = block;
  p.blockDimY = p.blockDimZ = 1;
  p.kernelParams = args;
  return p;
}

static size_t node_count(CUgraph g) {
  size_t n = 0;
  cuGraphGetNodes(g, nullptr, &n);
  return n;
}
static size_t edge_count(CUgraph g) {
  size_t n = 0;
#if CUDA_VERSION >= 13000
  cuGraphGetEdges(g, nullptr, nullptr, nullptr, &n);
#else
  cuGraphGetEdges(g, nullptr, nullptr, &n);
#endif
  return n;
}
static CUresult add_dep(CUgraph g, CUgraphNode* from, CUgraphNode* to, size_t n) {
#if CUDA_VERSION >= 13000
  return cuGraphAddDependencies(g, from, to, nullptr, n);
#else
  return cuGraphAddDependencies(g, from, to, n);
#endif
}
static CUresult remove_dep(CUgraph g, CUgraphNode* from, CUgraphNode* to, size_t n) {
#if CUDA_VERSION >= 13000
  return cuGraphRemoveDependencies(g, from, to, nullptr, n);
#else
  return cuGraphRemoveDependencies(g, from, to, n);
#endif
}
static size_t dep_count(CUgraphNode node, bool dependents) {
  size_t n = 0;
#if CUDA_VERSION >= 13000
  if (dependents) cuGraphNodeGetDependentNodes(node, nullptr, nullptr, &n);
  else cuGraphNodeGetDependencies(node, nullptr, nullptr, &n);
#else
  if (dependents) cuGraphNodeGetDependentNodes(node, nullptr, &n);
  else cuGraphNodeGetDependencies(node, nullptr, &n);
#endif
  return n;
}

// ---- building, reading back, launching --------------------------------------

static void build_and_run() {
  int v3 = 3, v10 = 10;
  void* a3[] = {&dbuf, &v3};
  void* a10[] = {&dbuf, &v10};

  CUgraph g = nullptr;
  OK(cuGraphCreate(&g, 0));
  CUgraph refused = nullptr;
  IS(cuGraphCreate(&refused, 1), CUDA_ERROR_INVALID_VALUE);
  IS(cuGraphCreate(nullptr, 0), CUDA_ERROR_INVALID_VALUE);

  // memset -> add 3 -> add 3 (a second one), -> copy to host -> host function
  CUDA_MEMSET_NODE_PARAMS ms;
  std::memset(&ms, 0, sizeof ms);
  ms.dst = dbuf;
  ms.elementSize = 4;
  ms.width = kN;
  ms.height = 1;
  ms.value = 1;
  CUcontext ctx = nullptr;
  OK(cuCtxGetCurrent(&ctx));
  CUgraphNode nset = nullptr, nadd = nullptr, nadd2 = nullptr, ncopy = nullptr, nhost = nullptr, nempty = nullptr;
  OK(cuGraphAddMemsetNode(&nset, g, nullptr, 0, &ms, ctx));
  CUDA_KERNEL_NODE_PARAMS kp = kernel_params(add_fn, a3);
  OK(cuGraphAddKernelNode(&nadd, g, &nset, 1, &kp));
  OK(cuGraphAddKernelNode(&nadd2, g, &nadd, 1, &kp));
  std::vector<int> host(kN, -1);
  CUDA_MEMCPY3D cp;
  std::memset(&cp, 0, sizeof cp);
  cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  cp.srcDevice = dbuf;
  cp.dstMemoryType = CU_MEMORYTYPE_HOST;
  cp.dstHost = host.data();
  cp.WidthInBytes = kN * sizeof(int);
  cp.Height = 1;
  cp.Depth = 1;
  OK(cuGraphAddMemcpyNode(&ncopy, g, &nadd2, 1, &cp, ctx));
  int hits = 0;
  CUDA_HOST_NODE_PARAMS hp;
  hp.fn = bump;
  hp.userData = &hits;
  OK(cuGraphAddHostNode(&nhost, g, &ncopy, 1, &hp));
  OK(cuGraphAddEmptyNode(&nempty, g, &nhost, 1));

  // The shape, and each node's type.
  CHECK(node_count(g) == 6);
  CHECK(edge_count(g) == 5);
  size_t roots = 0;
  OK(cuGraphGetRootNodes(g, nullptr, &roots));
  CHECK(roots == 1);
  CUgraphNodeType type;
  OK(cuGraphNodeGetType(nset, &type));
  CHECK(type == CU_GRAPH_NODE_TYPE_MEMSET);
  OK(cuGraphNodeGetType(nadd, &type));
  CHECK(type == CU_GRAPH_NODE_TYPE_KERNEL);
  OK(cuGraphNodeGetType(ncopy, &type));
  CHECK(type == CU_GRAPH_NODE_TYPE_MEMCPY);
  OK(cuGraphNodeGetType(nhost, &type));
  CHECK(type == CU_GRAPH_NODE_TYPE_HOST);
  OK(cuGraphNodeGetType(nempty, &type));
  CHECK(type == CU_GRAPH_NODE_TYPE_EMPTY);
  IS(cuGraphNodeGetType(nullptr, &type), CUDA_ERROR_INVALID_VALUE);
  CHECK(dep_count(nadd, false) == 1 && dep_count(nadd, true) == 1);

  // Arrays larger than the count are zero-filled past it.
  CUgraphNode many[10];
  for (auto& n : many) n = reinterpret_cast<CUgraphNode>(0x77);
  size_t want = 10;
  OK(cuGraphGetNodes(g, many, &want));
  CHECK(want == 6 && many[6] == nullptr && many[0] == nset);

  // What each node reads back as.
  CUDA_KERNEL_NODE_PARAMS kg;
  std::memset(&kg, 0xcd, sizeof kg);
  OK(cuGraphKernelNodeGetParams(nadd, &kg));
  CHECK(kg.func == add_fn && kg.blockDimX == kN && kg.gridDimX == 1 && kg.sharedMemBytes == 0 && kg.kernelParams);
  CHECK(kg.kernelParams && kg.kernelParams[0] != a3[0]);   // the node owns copies
  CHECK(kg.kernelParams && *static_cast<int*>(kg.kernelParams[1]) == 3);
  CUDA_MEMSET_NODE_PARAMS mg;
  std::memset(&mg, 0xcd, sizeof mg);
  OK(cuGraphMemsetNodeGetParams(nset, &mg));
  CHECK(mg.dst == dbuf && mg.elementSize == 4 && mg.width == kN && mg.height == 1 && mg.value == 1);
  CUDA_MEMCPY3D cg;
  std::memset(&cg, 0xcd, sizeof cg);
  OK(cuGraphMemcpyNodeGetParams(ncopy, &cg));
  CHECK(cg.srcMemoryType == CU_MEMORYTYPE_DEVICE && cg.dstMemoryType == CU_MEMORYTYPE_HOST && cg.dstHost == host.data() &&
        cg.WidthInBytes == kN * sizeof(int) && cg.Height == 1 && cg.Depth == 1);
  CUDA_HOST_NODE_PARAMS hg;
  std::memset(&hg, 0xcd, sizeof hg);
  OK(cuGraphHostNodeGetParams(nhost, &hg));
  CHECK(hg.fn == bump && hg.userData == &hits);
  IS(cuGraphKernelNodeGetParams(nset, &kg), CUDA_ERROR_INVALID_VALUE);

  // Launch it: 1 + 3 + 3 = 7.
  CUgraphExec exec = nullptr;
  OK(cuGraphInstantiate(&exec, g, 0));
  OK(cuGraphLaunch(exec, 0));
  OK(cuCtxSynchronize());
  CHECK(host[0] == 7 && host[kN - 1] == 7);
  CHECK(hits == 1);

  // The same executable graph again: it starts from the memset each time.
  OK(cuGraphLaunch(exec, 0));
  OK(cuCtxSynchronize());
  CHECK(host[0] == 7 && hits == 2);

  // New parameters, on the executable graph only: the first add now adds 10.
  CUDA_KERNEL_NODE_PARAMS k10 = kernel_params(add_fn, a10);
  OK(cuGraphExecKernelNodeSetParams(exec, nadd, &k10));
  OK(cuGraphLaunch(exec, 0));
  OK(cuCtxSynchronize());
  CHECK(host[0] == 14);
  // ... and the graph itself is unchanged.
  CUgraphExec fresh = nullptr;
  OK(cuGraphInstantiate(&fresh, g, 0));
  OK(cuGraphLaunch(fresh, 0));
  OK(cuCtxSynchronize());
  CHECK(host[0] == 7);
  // A different function on an executable graph's kernel node is allowed (an
  // RTX 3060 accepts it): the first node now multiplies by 3, so 1 * 3 + 3.
  CUDA_KERNEL_NODE_PARAMS kmul = kernel_params(mul_fn, a3);
  OK(cuGraphExecKernelNodeSetParams(fresh, nadd, &kmul));
  OK(cuGraphLaunch(fresh, 0));
  OK(cuCtxSynchronize());
  CHECK(host[0] == 6);
  // A node switched off does nothing and orders its neighbours.
  OK(cuGraphNodeSetEnabled(fresh, nadd2, 0));
  unsigned enabled = 7;
  OK(cuGraphNodeGetEnabled(fresh, nadd2, &enabled));
  CHECK(enabled == 0);
  OK(cuGraphLaunch(fresh, 0));
  OK(cuCtxSynchronize());
  CHECK(host[0] == 3);
  IS(cuGraphNodeSetEnabled(fresh, nhost, 0), CUDA_ERROR_INVALID_VALUE);   // only kernel, copy and fill nodes
  IS(cuGraphNodeSetEnabled(fresh, nempty, 0), CUDA_ERROR_INVALID_VALUE);

  // cuGraphExecUpdate takes new parameters from a graph of the same shape.
  OK(cuGraphKernelNodeSetParams(nadd, &k10));
  CUgraphExecUpdateResultInfo info;
  std::memset(&info, 0xcd, sizeof info);
  OK(cuGraphExecUpdate(exec, g, &info));
  CHECK(info.result == CU_GRAPH_EXEC_UPDATE_SUCCESS);
  OK(cuGraphLaunch(exec, 0));
  OK(cuCtxSynchronize());
  CHECK(host[0] == 14);
  OK(cuGraphKernelNodeSetParams(nadd, &kp));

  // A graph with a node more is a different shape.
  CUgraph bigger = nullptr;
  OK(cuGraphClone(&bigger, g));
  CUgraphNode extra = nullptr;
  OK(cuGraphAddEmptyNode(&extra, bigger, nullptr, 0));
  std::memset(&info, 0xcd, sizeof info);
  IS(cuGraphExecUpdate(exec, bigger, &info), CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE);
  CHECK(info.result == CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED);

  // Clones, and finding a node in one.
  CUgraph clone = nullptr;
  OK(cuGraphClone(&clone, g));
  CHECK(clone != g && node_count(clone) == 6);
  CUgraphNode twin = nullptr;
  OK(cuGraphNodeFindInClone(&twin, nadd, clone));
  CHECK(twin && twin != nadd);
  CUgraphNodeType twin_type;
  OK(cuGraphNodeGetType(twin, &twin_type));
  CHECK(twin_type == CU_GRAPH_NODE_TYPE_KERNEL);
  IS(cuGraphNodeFindInClone(&twin, nadd, g), CUDA_ERROR_INVALID_VALUE);   // g is not a clone of its own graph

  // Edges: one that exists, one to itself (both refused), and a longer cycle,
  // which is accepted and fails the instantiation.
  CUgraphNode a = nset, b = nadd2;
  IS(add_dep(g, &nset, &nadd, 1), CUDA_ERROR_INVALID_VALUE);
  IS(add_dep(g, &a, &a, 1), CUDA_ERROR_INVALID_VALUE);
  OK(add_dep(g, &b, &a, 1));
  CUgraphExec cyclic = nullptr;
  IS(cuGraphInstantiate(&cyclic, g, 0), CUDA_ERROR_INVALID_VALUE);
  OK(remove_dep(g, &b, &a, 1));
  IS(remove_dep(g, &b, &a, 1), CUDA_ERROR_INVALID_VALUE);   // not an edge now
  OK(cuGraphInstantiate(&cyclic, g, 0));
  OK(cuGraphExecDestroy(cyclic));

  // Instantiation's flags, and what an executable graph says of itself.
  CUgraphExec flagged = nullptr;
  IS(cuGraphInstantiateWithFlags(&flagged, g, 2), CUDA_ERROR_INVALID_VALUE);   // upload is for ...WithParams
  IS(cuGraphInstantiateWithFlags(&flagged, g, 0x10), CUDA_ERROR_INVALID_VALUE);
  IS(cuGraphInstantiateWithFlags(nullptr, g, 0), CUDA_ERROR_INVALID_VALUE);
  OK(cuGraphInstantiateWithFlags(&flagged, g, CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH));
  CUDA_GRAPH_INSTANTIATE_PARAMS ip;
  std::memset(&ip, 0, sizeof ip);
  ip.flags = CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD;
  CUgraphExec withparams = nullptr;
  OK(cuGraphInstantiateWithParams(&withparams, g, &ip));
  CHECK(ip.result_out == CUDA_GRAPH_INSTANTIATE_SUCCESS);
  OK(cuGraphUpload(withparams, 0));
  IS(cuGraphUpload(nullptr, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuGraphLaunch(nullptr, 0), CUDA_ERROR_INVALID_VALUE);

  // Child graphs, and events.
  CUgraph child = nullptr;
  OK(cuGraphCreate(&child, 0));
  CUgraphNode cnode = nullptr;
  OK(cuGraphAddKernelNode(&cnode, child, nullptr, 0, &kp));
  CUgraph outer = nullptr;
  OK(cuGraphCreate(&outer, 0));
  CUgraphNode nchild = nullptr;
  OK(cuGraphAddChildGraphNode(&nchild, outer, nullptr, 0, child));
  CUgraph inner = nullptr;
  OK(cuGraphChildGraphNodeGetGraph(nchild, &inner));
  CHECK(inner && node_count(inner) == 1);
  IS(cuGraphDestroy(inner), CUDA_ERROR_INVALID_VALUE);   // the node owns it
  CUevent ev = nullptr;
  OK(cuEventCreate(&ev, 0));
  CUgraphNode nrec = nullptr, nwait = nullptr;
  OK(cuGraphAddEventRecordNode(&nrec, outer, &nchild, 1, ev));
  OK(cuGraphAddEventWaitNode(&nwait, outer, &nrec, 1, ev));
  CUevent got = nullptr;
  OK(cuGraphEventRecordNodeGetEvent(nrec, &got));
  CHECK(got == ev);
  IS(cuGraphEventRecordNodeGetEvent(nwait, &got), CUDA_ERROR_INVALID_VALUE);   // the other kind
  IS(cuGraphAddEventRecordNode(&nrec, outer, nullptr, 0, nullptr), CUDA_ERROR_INVALID_VALUE);
  fill(0);
  CUgraphExec oexec = nullptr;
  OK(cuGraphInstantiate(&oexec, outer, 0));
  OK(cuGraphLaunch(oexec, 0));
  OK(cuCtxSynchronize());
  CHECK(read_back()[0] == 3);
  // The event the graph recorded can be timed against another.
  OK(cuGraphExecDestroy(oexec));

  // A graph is gone once destroyed, and a second destroy says so.
  OK(cuGraphDestroy(outer));
  IS(cuGraphDestroy(outer), CUDA_ERROR_INVALID_VALUE);
  IS(cuGraphDestroy(nullptr), CUDA_ERROR_INVALID_VALUE);
  OK(cuGraphExecDestroy(exec));
  IS(cuGraphExecDestroy(nullptr), CUDA_ERROR_INVALID_VALUE);

  // Bad node parameters, as the card refuses them.
  CUgraphNode junk = nullptr;
  CUDA_KERNEL_NODE_PARAMS bad = kernel_params(add_fn, a3);
  bad.func = nullptr;
  IS(cuGraphAddKernelNode(&junk, g, nullptr, 0, &bad), CUDA_ERROR_INVALID_HANDLE);
  bad = kernel_params(add_fn, a3, 0);
  IS(cuGraphAddKernelNode(&junk, g, nullptr, 0, &bad), CUDA_ERROR_INVALID_VALUE);
  bad = kernel_params(add_fn, a3, 100000);
  IS(cuGraphAddKernelNode(&junk, g, nullptr, 0, &bad), CUDA_ERROR_INVALID_VALUE);
  bad = kernel_params(add_fn, nullptr);
  IS(cuGraphAddKernelNode(&junk, g, nullptr, 0, &bad), CUDA_ERROR_INVALID_VALUE);
  IS(cuGraphAddKernelNode(&junk, g, nullptr, 0, nullptr), CUDA_ERROR_INVALID_VALUE);
  IS(cuGraphAddKernelNode(nullptr, g, nullptr, 0, &kp), CUDA_ERROR_INVALID_VALUE);
  IS(cuGraphAddKernelNode(&junk, g, nullptr, 3, &kp), CUDA_ERROR_INVALID_VALUE);   // three dependencies, none given
  CUDA_MEMSET_NODE_PARAMS badm = ms;
  badm.elementSize = 3;
  IS(cuGraphAddMemsetNode(&junk, g, nullptr, 0, &badm, ctx), CUDA_ERROR_INVALID_VALUE);
  badm = ms;
  badm.dst = 0;
  IS(cuGraphAddMemsetNode(&junk, g, nullptr, 0, &badm, ctx), CUDA_ERROR_INVALID_VALUE);
  CUDA_MEMCPY3D badc = cp;
  badc.Depth = 0;
  IS(cuGraphAddMemcpyNode(&junk, g, nullptr, 0, &badc, ctx), CUDA_ERROR_INVALID_VALUE);
  // A null host function is accepted when the node is added, refused when set.
  CUDA_HOST_NODE_PARAMS nofn;
  nofn.fn = nullptr;
  nofn.userData = nullptr;
  OK(cuGraphAddHostNode(&junk, g, nullptr, 0, &nofn));
  IS(cuGraphHostNodeSetParams(junk, &nofn), CUDA_ERROR_INVALID_VALUE);
  OK(cuGraphDestroyNode(junk));

  // The text drawing of a graph.
  OK(cuGraphDebugDotPrint(g, "/tmp/vgpu-driver-graphs.dot", 0));
  std::remove("/tmp/vgpu-driver-graphs.dot");
  IS(cuGraphDebugDotPrint(g, nullptr, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuGraphDebugDotPrint(g, "/nonexistent-dir/g.dot", 0), CUDA_ERROR_OPERATING_SYSTEM);

  cuGraphExecDestroy(fresh);
  cuGraphExecDestroy(flagged);
  cuGraphExecDestroy(withparams);
  cuGraphDestroy(g);
  cuGraphDestroy(clone);
  cuGraphDestroy(bigger);
  cuGraphDestroy(child);
  cuEventDestroy(ev);
}

// ---- kernel node attributes and the CUkernel path ------------------------------

static void attributes_and_kernels() {
  int v = 5;
  void* args[] = {&dbuf, &v};
  CUgraph g = nullptr;
  OK(cuGraphCreate(&g, 0));
  CUDA_KERNEL_NODE_PARAMS kp = kernel_params(add_fn, args);
  CUgraphNode k1 = nullptr, k2 = nullptr;
  OK(cuGraphAddKernelNode(&k1, g, nullptr, 0, &kp));
  CUkernelNodeAttrValue av;
  std::memset(&av, 0xcd, sizeof av);
  OK(cuGraphKernelNodeGetAttribute(k1, CU_LAUNCH_ATTRIBUTE_PRIORITY, &av));
  CHECK(av.priority == 0);
  std::memset(&av, 0, sizeof av);
  av.priority = -1;
  OK(cuGraphKernelNodeSetAttribute(k1, CU_LAUNCH_ATTRIBUTE_PRIORITY, &av));
  std::memset(&av, 0xcd, sizeof av);
  OK(cuGraphKernelNodeGetAttribute(k1, CU_LAUNCH_ATTRIBUTE_PRIORITY, &av));
  CHECK(av.priority == -1);
  std::memset(&av, 0xcd, sizeof av);
  OK(cuGraphKernelNodeGetAttribute(k1, CU_LAUNCH_ATTRIBUTE_COOPERATIVE, &av));
  CHECK(av.cooperative == 0);
  IS(cuGraphKernelNodeSetAttribute(k1, static_cast<CUlaunchAttributeID>(999), &av), CUDA_ERROR_INVALID_VALUE);
  IS(cuGraphKernelNodeGetAttribute(k1, static_cast<CUlaunchAttributeID>(999), &av), CUDA_ERROR_INVALID_VALUE);
  IS(cuGraphKernelNodeSetAttribute(k1, CU_LAUNCH_ATTRIBUTE_IGNORE, &av), CUDA_ERROR_INVALID_VALUE);
  OK(cuGraphAddKernelNode(&k2, g, nullptr, 0, &kp));
  OK(cuGraphKernelNodeCopyAttributes(k2, k1));
  std::memset(&av, 0xcd, sizeof av);
  OK(cuGraphKernelNodeGetAttribute(k2, CU_LAUNCH_ATTRIBUTE_PRIORITY, &av));
  CHECK(av.priority == -1);
  IS(cuGraphKernelNodeCopyAttributes(nullptr, k1), CUDA_ERROR_INVALID_VALUE);

  // A node made from a CUkernel (the function is null): it reads back the kernel
  // and the function that kernel gives.
  CUlibrary lib = nullptr;
  CUkernel kern = nullptr;
  OK(cuLibraryLoadData(&lib, kPtx, nullptr, nullptr, 0, nullptr, nullptr, 0));
  OK(cuLibraryGetKernel(&kern, lib, "addp"));
  CUDA_KERNEL_NODE_PARAMS kk = kp;
  kk.func = nullptr;
#if CUDA_VERSION >= 12000
  CUDA_KERNEL_NODE_PARAMS k3;   // since CUDA 12 the structure carries the kernel and the context
  std::memset(&k3, 0, sizeof k3);
  k3.kern = kern;
  k3.gridDimX = k3.gridDimY = k3.gridDimZ = 1;
  k3.blockDimX = kN;
  k3.blockDimY = k3.blockDimZ = 1;
  k3.kernelParams = args;
  CUgraphNode k3n = nullptr;
  OK(cuGraphAddKernelNode(&k3n, g, nullptr, 0, &k3));
  CUDA_KERNEL_NODE_PARAMS got;
  std::memset(&got, 0xcd, sizeof got);
  OK(cuGraphKernelNodeGetParams(k3n, &got));
  CUfunction kf = nullptr;
  OK(cuKernelGetFunction(&kf, kern));
  CHECK(got.func == kf && got.kern == kern);
#endif
  fill(0);
  CUgraphExec ex = nullptr;
  OK(cuGraphInstantiate(&ex, g, 0));
  OK(cuGraphLaunch(ex, 0));
  OK(cuCtxSynchronize());
  CHECK(read_back()[0] == 15);   // three nodes, each adding 5
  OK(cuGraphExecDestroy(ex));
  OK(cuGraphDestroy(g));
  OK(cuLibraryUnload(lib));
}

// ---- stream capture through the driver ------------------------------------------

static void capture() {
  CUstream s = nullptr, s2 = nullptr;
  OK(cuStreamCreate(&s, 0));
  OK(cuStreamCreate(&s2, 0));
  int v = 4;
  void* args[] = {&dbuf, &v};
  CUstreamCaptureStatus st;
  OK(cuStreamIsCapturing(s, &st));
  CHECK(st == CU_STREAM_CAPTURE_STATUS_NONE);
  CUgraph g = nullptr;
  IS(cuStreamEndCapture(s, &g), CUDA_ERROR_ILLEGAL_STATE);
  IS(cuStreamBeginCapture(0, CU_STREAM_CAPTURE_MODE_GLOBAL), CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);   // not the legacy stream
  IS(cuStreamBeginCapture(s, static_cast<CUstreamCaptureMode>(9)), CUDA_ERROR_INVALID_VALUE);

  fill(1);
  OK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
  IS(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL), CUDA_ERROR_ILLEGAL_STATE);
  OK(cuStreamIsCapturing(s, &st));
  CHECK(st == CU_STREAM_CAPTURE_STATUS_ACTIVE);
  OK(cuLaunchKernel(add_fn, 1, 1, 1, kN, 1, 1, 0, s, args, nullptr));
  int hits = 0;
  OK(cuLaunchHostFunc(s, bump, &hits));
  std::vector<int> host(kN, -1);
  OK(cuMemcpyDtoHAsync(host.data(), dbuf, kN * sizeof(int), s));
  OK(cuMemsetD32Async(dbuf, 9, kN, s));
  CHECK(hits == 0 && host[0] == -1);                   // nothing ran: it is all in the graph
  CUgraph building = nullptr;
  cuuint64_t id = 0;
  const CUgraphNode* deps = nullptr;
  size_t ndeps = 0;
#if CUDA_VERSION >= 13000
  OK(cuStreamGetCaptureInfo(s, &st, &id, &building, &deps, nullptr, &ndeps));
#else
  OK(cuStreamGetCaptureInfo(s, &st, &id, &building, &deps, &ndeps));
#endif
  CHECK(st == CU_STREAM_CAPTURE_STATUS_ACTIVE && id != 0 && building && ndeps == 1);
  CHECK(node_count(building) == 4);
  CUgraph captured = nullptr;
  OK(cuStreamEndCapture(s, &captured));
  CHECK(captured == building && node_count(captured) == 4);
  CHECK(read_back()[0] == 1);   // nothing of it ran
  CUgraphExec ex = nullptr;
  OK(cuGraphInstantiate(&ex, captured, 0));
  OK(cuGraphLaunch(ex, s));
  OK(cuStreamSynchronize(s));
  CHECK(host[0] == 5 && hits == 1);   // the copy ran before the memset, after the add
  CHECK(read_back()[0] == 9);
  OK(cuStreamIsCapturing(s, &st));
  CHECK(st == CU_STREAM_CAPTURE_STATUS_NONE);

  // Two streams, joined: a fork on an event, the second stream's work, and a join.
  CUevent fork = nullptr, join = nullptr;
  OK(cuEventCreate(&fork, CU_EVENT_DISABLE_TIMING));
  OK(cuEventCreate(&join, CU_EVENT_DISABLE_TIMING));
  fill(0);
  OK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
  OK(cuLaunchKernel(add_fn, 1, 1, 1, kN, 1, 1, 0, s, args, nullptr));
  OK(cuEventRecord(fork, s));
  OK(cuStreamWaitEvent(s2, fork, 0));
  OK(cuStreamIsCapturing(s2, &st));
  CHECK(st == CU_STREAM_CAPTURE_STATUS_ACTIVE);       // it joined the capture
  OK(cuLaunchKernel(add_fn, 1, 1, 1, kN, 1, 1, 0, s2, args, nullptr));
  OK(cuEventRecord(join, s2));
  OK(cuStreamWaitEvent(s, join, 0));
  CUgraph forked = nullptr;
  OK(cuStreamEndCapture(s, &forked));
  CHECK(node_count(forked) == 2 && edge_count(forked) == 1);
  OK(cuStreamIsCapturing(s2, &st));
  CHECK(st == CU_STREAM_CAPTURE_STATUS_NONE);
  CUgraphExec fx = nullptr;
  OK(cuGraphInstantiate(&fx, forked, 0));
  OK(cuGraphLaunch(fx, s));
  OK(cuStreamSynchronize(s));
  CHECK(read_back()[0] == 8);

  // An event recorded in a capture cannot be asked about (CUDA_ERROR_CAPTURED_EVENT),
  // and asking invalidates the capture.
  CUevent captured_event = nullptr;
  OK(cuEventCreate(&captured_event, 0));
  OK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
  OK(cuEventRecord(captured_event, s));
  IS(cuEventQuery(captured_event), static_cast<CUresult>(907));
  CUgraph asked = reinterpret_cast<CUgraph>(0x1);
  IS(cuStreamEndCapture(s, &asked), CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);

  // A stream that joined and was never joined back: the capture cannot end.
  OK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
  OK(cuEventRecord(fork, s));
  OK(cuStreamWaitEvent(s2, fork, 0));
  OK(cuLaunchKernel(add_fn, 1, 1, 1, kN, 1, 1, 0, s2, args, nullptr));
  CUgraph lost = reinterpret_cast<CUgraph>(0x1);
  IS(cuStreamEndCapture(s, &lost), CUDA_ERROR_STREAM_CAPTURE_UNJOINED);
  CHECK(lost == nullptr);
  OK(cuStreamIsCapturing(s2, &st));
  CHECK(st == CU_STREAM_CAPTURE_STATUS_NONE);

  // Waiting for a capturing stream invalidates the capture, which says so.
  OK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
  IS(cuStreamSynchronize(s), CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
  CUgraph inv = reinterpret_cast<CUgraph>(0x1);
  IS(cuStreamEndCapture(s, &inv), CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
  CHECK(inv == nullptr);

  // Into a graph the program made, after a node it names.
#if CUDA_VERSION >= 12030
  CUgraph target = nullptr;
  OK(cuGraphCreate(&target, 0));
  CUgraphNode root = nullptr;
  OK(cuGraphAddEmptyNode(&root, target, nullptr, 0));
  OK(cuStreamBeginCaptureToGraph(s, target, &root, nullptr, 1, CU_STREAM_CAPTURE_MODE_GLOBAL));
  OK(cuLaunchKernel(add_fn, 1, 1, 1, kN, 1, 1, 0, s, args, nullptr));
  CUgraph seen = nullptr;
#if CUDA_VERSION >= 13000
  OK(cuStreamGetCaptureInfo(s, &st, &id, &seen, &deps, nullptr, &ndeps));
#else
  OK(cuStreamGetCaptureInfo(s, &st, &id, &seen, &deps, &ndeps));
#endif
  CHECK(seen == target);
  IS(cuGraphDestroy(target), CUDA_ERROR_ILLEGAL_STATE);   // not while a capture is adding to it
  CUgraph same = nullptr;
  OK(cuStreamEndCapture(s, &same));
  CHECK(same == target && node_count(target) == 2 && edge_count(target) == 1);
  OK(cuGraphDestroy(target));
#endif

  // Dependencies, set by hand.
  OK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CUgraph hand = nullptr;
#if CUDA_VERSION >= 13000
  OK(cuStreamGetCaptureInfo(s, &st, &id, &hand, &deps, nullptr, &ndeps));
#else
  OK(cuStreamGetCaptureInfo(s, &st, &id, &hand, &deps, &ndeps));
#endif
  CHECK(ndeps == 0);
  CUgraphNode en = nullptr;
  OK(cuGraphAddEmptyNode(&en, hand, nullptr, 0));
#if CUDA_VERSION >= 13000
  OK(cuStreamUpdateCaptureDependencies(s, &en, nullptr, 1, CU_STREAM_SET_CAPTURE_DEPENDENCIES));
  OK(cuStreamGetCaptureInfo(s, &st, &id, &hand, &deps, nullptr, &ndeps));
  IS(cuStreamUpdateCaptureDependencies(s, &en, nullptr, 1, 7), CUDA_ERROR_INVALID_VALUE);
#else
  OK(cuStreamUpdateCaptureDependencies(s, &en, 1, CU_STREAM_SET_CAPTURE_DEPENDENCIES));
  OK(cuStreamGetCaptureInfo(s, &st, &id, &hand, &deps, &ndeps));
  IS(cuStreamUpdateCaptureDependencies(s, &en, 1, 7), CUDA_ERROR_INVALID_VALUE);
#endif
  CHECK(ndeps == 1 && deps[0] == en);
  OK(cuStreamEndCapture(s, &hand));
  OK(cuGraphDestroy(hand));

  // A thread's capture mode: it starts global, and each exchange returns the last.
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
  OK(cuThreadExchangeStreamCaptureMode(&mode));
  CHECK(mode == CU_STREAM_CAPTURE_MODE_GLOBAL);
  mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
  OK(cuThreadExchangeStreamCaptureMode(&mode));
  CHECK(mode == CU_STREAM_CAPTURE_MODE_RELAXED);

  OK(cuGraphExecDestroy(fx));
  OK(cuGraphExecDestroy(ex));
  OK(cuGraphDestroy(forked));
  OK(cuGraphDestroy(captured));
  cuEventDestroy(fork);
  cuEventDestroy(join);
  cuEventDestroy(captured_event);
  cuStreamDestroy(s);
  cuStreamDestroy(s2);
}

// ---- user objects and graph memory ------------------------------------------------

static void user_objects() {
  CUuserObject obj = nullptr;
  int dummy = 0;
  IS(cuUserObjectCreate(&obj, &dummy, nullptr, 1, 0), CUDA_ERROR_INVALID_VALUE);                               // no destructor
  IS(cuUserObjectCreate(&obj, &dummy, note_destroyed, 0, 0), CUDA_ERROR_INVALID_VALUE);                       // no references
  IS(cuUserObjectCreate(&obj, &dummy, note_destroyed, 1, 7), CUDA_ERROR_INVALID_VALUE);                       // unknown flags
  OK(cuUserObjectCreate(&obj, &dummy, note_destroyed, 2, CU_USER_OBJECT_NO_DESTRUCTOR_SYNC));
  IS(cuUserObjectRetain(obj, 0), CUDA_ERROR_INVALID_VALUE);
  OK(cuUserObjectRetain(obj, 1));                                  // three now
  CUgraph g = nullptr;
  OK(cuGraphCreate(&g, 0));
  OK(cuGraphRetainUserObject(g, obj, 1, CU_GRAPH_USER_OBJECT_MOVE));   // one of the three is the graph's now
  OK(cuUserObjectRelease(obj, 2));                                 // the program's two
  // The graph's reference keeps it alive; releasing it is what ends it. (On the
  // card the destructor runs some time after, from a thread of the driver's.)
  OK(cuGraphReleaseUserObject(g, obj, 1));
  for (int i = 0; i < 100 && destroyed == 0; ++i) usleep(10000);
  CHECK(destroyed == 1);
  OK(cuGraphDestroy(g));

  // A graph's references go when it does, and an instantiation keeps its own.
  destroyed = 0;
  CUuserObject o2 = nullptr;
  OK(cuUserObjectCreate(&o2, &dummy, note_destroyed, 1, CU_USER_OBJECT_NO_DESTRUCTOR_SYNC));
  CUgraph g2 = nullptr;
  OK(cuGraphCreate(&g2, 0));
  CUgraphNode n = nullptr;
  OK(cuGraphAddEmptyNode(&n, g2, nullptr, 0));
  OK(cuGraphRetainUserObject(g2, o2, 1, CU_GRAPH_USER_OBJECT_MOVE));
  CUgraphExec e2 = nullptr;
  OK(cuGraphInstantiate(&e2, g2, 0));
  OK(cuGraphDestroy(g2));
  usleep(50000);
  CHECK(destroyed == 0);   // the executable graph still holds one
  OK(cuGraphExecDestroy(e2));
  for (int i = 0; i < 100 && destroyed == 0; ++i) usleep(10000);
  CHECK(destroyed == 1);
}

static void graph_memory(CUdevice dev) {
  OK(cuDeviceGraphMemTrim(dev));
  IS(cuDeviceGraphMemTrim(77), CUDA_ERROR_INVALID_DEVICE);
  cuuint64_t val = 99;
  OK(cuDeviceGetGraphMemAttribute(dev, CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT, &val));
  CHECK(val == 0);
  IS(cuDeviceGetGraphMemAttribute(dev, static_cast<CUgraphMem_attribute>(9), &val), CUDA_ERROR_INVALID_VALUE);
  val = 0;
  OK(cuDeviceSetGraphMemAttribute(dev, CU_GRAPH_MEM_ATTR_USED_MEM_HIGH, &val));
  val = 5;
  IS(cuDeviceSetGraphMemAttribute(dev, CU_GRAPH_MEM_ATTR_USED_MEM_HIGH, &val), CUDA_ERROR_INVALID_VALUE);   // only zero resets
  val = 0;
  IS(cuDeviceSetGraphMemAttribute(dev, CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT, &val), CUDA_ERROR_INVALID_VALUE);   // a current total is not settable
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (cuInit(0) != CUDA_SUCCESS) {
    std::printf("SKIP: no driver\n");
    return 0;
  }
  CUdevice dev;
  OK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  OK(cuDevicePrimaryCtxRetain(&ctx, dev));
  OK(cuCtxSetCurrent(ctx));
  CUmodule mod;
  OK(cuModuleLoadData(&mod, kPtx));
  OK(cuModuleGetFunction(&add_fn, mod, "addp"));
  OK(cuModuleGetFunction(&mul_fn, mod, "mulp"));
  OK(cuMemAlloc(&dbuf, kN * sizeof(int)));
  build_and_run();
  attributes_and_kernels();
  capture();
  user_objects();
  graph_memory(dev);
  std::printf(failures ? "FAIL: %d driver graph checks\n" : "PASS: every driver graph check\n", failures);
  return failures ? 1 : 0;
}
