// The driver API's explicit graphs: cuGraphCreate, the cuGraphAdd*Node calls and their Get/SetParams,
// dependencies, instantiation, launch, executable-graph updates, cloning, user objects, and capture into
// an existing graph. Every expectation was written from what NVIDIA's driver does on an RTX 3060
// (run_graph_driver.sh --card runs this program on it).
#include <cuda.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
static void expect(const std::string& what, bool ok, double detail = 0) {
  std::printf("%s %s", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok && detail != 0) std::printf(" (%g)", detail);
  std::printf("\n");
  fails += !ok;
}
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
  std::printf("FAIL %s -> %d\n", #x, (int)r_); ++fails; return 1; } } while (0)
#define OKRC(x) ((x) == CUDA_SUCCESS)

static const char* kPtx = R"(
.version 7.0
.target sm_50
.address_size 64
// y[i] = x[i] + v
.visible .entry addv(.param .u64 py, .param .u64 px, .param .f32 pv, .param .u32 pn)
{
  .reg .pred %p<2>;
  .reg .b32 %r<8>;
  .reg .f32 %f<5>;
  .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [py];
  ld.param.u64 %rd2, [px];
  ld.param.f32 %f1, [pv];
  ld.param.u32 %r1, [pn];
  mov.u32 %r2, %tid.x;
  mov.u32 %r3, %ctaid.x;
  mov.u32 %r4, %ntid.x;
  mad.lo.s32 %r5, %r3, %r4, %r2;
  setp.ge.s32 %p1, %r5, %r1;
  @%p1 bra DONE;
  cvta.to.global.u64 %rd1, %rd1;
  cvta.to.global.u64 %rd2, %rd2;
  mul.wide.s32 %rd4, %r5, 4;
  add.s64 %rd5, %rd2, %rd4;
  ld.global.f32 %f2, [%rd5];
  add.f32 %f3, %f2, %f1;
  add.s64 %rd6, %rd1, %rd4;
  st.global.f32 [%rd6], %f3;
DONE:
  ret;
}
)";

static const int N = 64;
static std::atomic<int> g_host_calls{0};
static void host_fn(void* user) { g_host_calls += *static_cast<int*>(user); }

static std::vector<float> get(CUdeviceptr p, int n = N) {
  std::vector<float> h(n);
  cuMemcpyDtoH(h.data(), p, n * 4);
  return h;
}
static bool all_equal(const std::vector<float>& v, float x) {
  for (float e : v) if (e != x) return false;
  return true;
}

int main() {
#if CUDA_VERSION < 13000
  // The 13.0 spellings of the edge-data calls (cuGraphGetEdges and the others take the edge data) are what
  // this program is written in.
  std::puts("SKIP: needs the CUDA 13 headers");
  return 0;
#else
  cuInit(0);
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext ctx;
  cuDevicePrimaryCtxRetain(&ctx, dev);
  cuCtxSetCurrent(ctx);
  CUmodule mod;
  CUfunction addv;
  if (cuModuleLoadData(&mod, kPtx) != CUDA_SUCCESS || cuModuleGetFunction(&addv, mod, "addv") != CUDA_SUCCESS) {
    std::puts("FAIL module");
    return 1;
  }
  CUstream stream;
  cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
  CUdeviceptr x, y, z, w;
  cuMemAlloc(&x, N * 4), cuMemAlloc(&y, N * 4), cuMemAlloc(&z, N * 4), cuMemAlloc(&w, N * 4);

  /* ---- a graph of every kind of node, built by hand ---------------------------------- */
  {
    CUgraph g;
    CK(cuGraphCreate(&g, 0));
    // x = 0x3f800000 (1.0f) -> y = x + 2 (kernel) -> z = y (copy) -> host function ; w = 7.0 (second root)
    CUDA_MEMSET_NODE_PARAMS ms{};
    ms.dst = x, ms.pitch = 0, ms.value = 0x3f800000u, ms.elementSize = 4, ms.width = N, ms.height = 1;
    CUgraphNode nset, nkern, ncopy, nhost, nw, nempty, nrec;
    CK(cuGraphAddMemsetNode(&nset, g, nullptr, 0, &ms, ctx));
    float v = 2.0f;
    int n = N;
    void* args[] = {&y, &x, &v, &n};
    CUDA_KERNEL_NODE_PARAMS kp{};
    kp.func = addv, kp.gridDimX = 2, kp.gridDimY = 1, kp.gridDimZ = 1, kp.blockDimX = 32, kp.blockDimY = 1, kp.blockDimZ = 1;
    kp.kernelParams = args;
    CK(cuGraphAddKernelNode(&nkern, g, &nset, 1, &kp));
    v = 1000.0f;   // changing the caller's copy after the call changes nothing in the graph
    CUDA_MEMCPY3D c;
    std::memset(&c, 0, sizeof c);
    c.srcMemoryType = CU_MEMORYTYPE_DEVICE, c.srcDevice = y, c.srcPitch = N * 4, c.srcHeight = 1;
    c.dstMemoryType = CU_MEMORYTYPE_DEVICE, c.dstDevice = z, c.dstPitch = N * 4, c.dstHeight = 1;
    c.WidthInBytes = N * 4, c.Height = 1, c.Depth = 1;
    CK(cuGraphAddMemcpyNode(&ncopy, g, &nkern, 1, &c, ctx));
    static int hv = 5;
    CUDA_HOST_NODE_PARAMS hp{host_fn, &hv};
    CK(cuGraphAddHostNode(&nhost, g, &ncopy, 1, &hp));
    CUDA_MEMSET_NODE_PARAMS mw{};
    mw.dst = w, mw.value = 0x40e00000u, mw.elementSize = 4, mw.width = N, mw.height = 1;   // 7.0f
    CK(cuGraphAddMemsetNode(&nw, g, nullptr, 0, &mw, ctx));
    CUgraphNode both[2] = {nhost, nw};
    CK(cuGraphAddEmptyNode(&nempty, g, both, 2));
    CUevent ev;
    CK(cuEventCreate(&ev, 0));
    CK(cuGraphAddEventRecordNode(&nrec, g, &nempty, 1, ev));

    size_t count = 0;
    CK(cuGraphGetNodes(g, nullptr, &count));
    expect("cuGraphGetNodes counts the seven nodes", count == 7, (double)count);
    count = 0;
    CK(cuGraphGetRootNodes(g, nullptr, &count));
    expect("two root nodes", count == 2, (double)count);
    count = 0;
    CK(cuGraphGetEdges(g, nullptr, nullptr, nullptr, &count));
    expect("six edges", count == 6, (double)count);
    CUgraphNodeType t;
    bool types = true;
    const CUgraphNode order[7] = {nset, nkern, ncopy, nhost, nw, nempty, nrec};
    const CUgraphNodeType want[7] = {CU_GRAPH_NODE_TYPE_MEMSET, CU_GRAPH_NODE_TYPE_KERNEL, CU_GRAPH_NODE_TYPE_MEMCPY,
                                     CU_GRAPH_NODE_TYPE_HOST, CU_GRAPH_NODE_TYPE_MEMSET, CU_GRAPH_NODE_TYPE_EMPTY,
                                     CU_GRAPH_NODE_TYPE_EVENT_RECORD};
    for (int i = 0; i < 7; ++i) types = types && OKRC(cuGraphNodeGetType(order[i], &t)) && t == want[i];
    expect("each node reports its type", types);
    count = 0;
    CK(cuGraphNodeGetDependencies(nempty, nullptr, nullptr, &count));
    expect("the empty node depends on two", count == 2, (double)count);
    count = 0;
    CK(cuGraphNodeGetDependentNodes(nset, nullptr, nullptr, &count));
    expect("the first memset has one dependent", count == 1, (double)count);

    // Get*Params give back what was set.
    CUDA_KERNEL_NODE_PARAMS got{};
    CK(cuGraphKernelNodeGetParams(nkern, &got));
    // The node keeps its own copy of the parameters, and gives back pointers into it.
    expect("cuGraphKernelNodeGetParams", got.func == addv && got.gridDimX == 2 && got.blockDimX == 32 && got.kernelParams &&
                                            got.kernelParams != args && *static_cast<float*>(got.kernelParams[2]) == 2.0f);
    CUDA_MEMCPY3D gc;
    CK(cuGraphMemcpyNodeGetParams(ncopy, &gc));
    expect("cuGraphMemcpyNodeGetParams", gc.srcDevice == y && gc.dstDevice == z && gc.WidthInBytes == N * 4);
    CUDA_MEMSET_NODE_PARAMS gm{};
    CK(cuGraphMemsetNodeGetParams(nset, &gm));
    expect("cuGraphMemsetNodeGetParams", gm.dst == x && gm.value == 0x3f800000u && gm.elementSize == 4 && gm.width == N);
    CUDA_HOST_NODE_PARAMS gh{};
    CK(cuGraphHostNodeGetParams(nhost, &gh));
    expect("cuGraphHostNodeGetParams", gh.fn == host_fn && gh.userData == &hv);
    CUevent gev = nullptr;
    CK(cuGraphEventRecordNodeGetEvent(nrec, &gev));
    expect("cuGraphEventRecordNodeGetEvent", gev == ev);

    CUgraphExec exec;
    CK(cuGraphInstantiateWithFlags(&exec, g, 0));
    const int before = g_host_calls;
    CK(cuMemsetD32(x, 0, N));
    CK(cuMemsetD32(y, 0, N));
    CK(cuMemsetD32(z, 0, N));
    CK(cuMemsetD32(w, 0, N));
    CK(cuGraphLaunch(exec, stream));
    CK(cuStreamSynchronize(stream));
    expect("a launch runs every node, in order", all_equal(get(x), 1.0f) && all_equal(get(y), 3.0f) && all_equal(get(z), 3.0f) &&
                                                    all_equal(get(w), 7.0f) && g_host_calls == before + 5);
    // The kernel's parameters were copied when the node was made (1000 was set afterwards).

    // A change to the graph's node does not reach the graph already instantiated; a launch of an exec
    // updated in place does see it, and the source graph is untouched.
    float v2 = 10.0f;
    void* args2[] = {&y, &x, &v2, &n};
    CUDA_KERNEL_NODE_PARAMS kp2 = kp;
    kp2.kernelParams = args2;
    CK(cuGraphKernelNodeSetParams(nkern, &kp2));
    CUgraphExec exec2;
    CK(cuGraphInstantiateWithFlags(&exec2, g, 0));
    CK(cuGraphLaunch(exec, stream));
    CK(cuStreamSynchronize(stream));
    expect("an executable graph keeps the parameters it was instantiated with", all_equal(get(z), 3.0f));
    CK(cuGraphLaunch(exec2, stream));
    CK(cuStreamSynchronize(stream));
    expect("cuGraphKernelNodeSetParams before instantiating", all_equal(get(z), 11.0f));
    float v3 = 20.0f;
    void* args3[] = {&y, &x, &v3, &n};
    CUDA_KERNEL_NODE_PARAMS kp3 = kp;
    kp3.kernelParams = args3;
    CK(cuGraphExecKernelNodeSetParams(exec2, nkern, &kp3));
    CK(cuGraphLaunch(exec2, stream));
    CK(cuStreamSynchronize(stream));
    expect("cuGraphExecKernelNodeSetParams", all_equal(get(z), 21.0f));
    CK(cuGraphKernelNodeGetParams(nkern, &got));
    expect("... leaves the graph's own node alone", *static_cast<float*>(got.kernelParams[2]) == 10.0f);

    // A copy and a fill, updated on the executable graph.
    CUDA_MEMSET_NODE_PARAMS mx = ms;
    mx.value = 0x40000000u;   // 2.0f
    CK(cuGraphExecMemsetNodeSetParams(exec2, nset, &mx, ctx));
    CK(cuGraphLaunch(exec2, stream));
    CK(cuStreamSynchronize(stream));
    expect("cuGraphExecMemsetNodeSetParams", all_equal(get(x), 2.0f) && all_equal(get(z), 22.0f));

    // Enabled and disabled nodes of an executable graph.
    unsigned en = 0;
    CK(cuGraphNodeGetEnabled(exec2, nkern, &en));
    expect("a node is enabled by default", en == 1);
    CK(cuGraphNodeSetEnabled(exec2, ncopy, 0));
    CK(cuMemsetD32(z, 0, N));
    CK(cuGraphLaunch(exec2, stream));
    CK(cuStreamSynchronize(stream));
    expect("a disabled copy node does not run", all_equal(get(z), 0.0f));
    CK(cuGraphNodeSetEnabled(exec2, ncopy, 1));

    // Clones.
    CUgraph clone;
    CK(cuGraphClone(&clone, g));
    CUgraphNode twin = nullptr;
    CK(cuGraphNodeFindInClone(&twin, nkern, clone));
    CUDA_KERNEL_NODE_PARAMS cgot{};
    CK(cuGraphKernelNodeGetParams(twin, &cgot));
    expect("a clone's node has the original's parameters", twin && twin != nkern && cgot.func == addv && cgot.gridDimX == 2);
    size_t cn = 0;
    CK(cuGraphGetNodes(clone, nullptr, &cn));
    expect("a clone has the same nodes", cn == 7);

    // cuGraphExecUpdate takes the parameters of a graph with the same shape.
    CUgraphExecUpdateResultInfo info{};
    CUresult ur = cuGraphExecUpdate(exec, clone, &info);
    expect("cuGraphExecUpdate with a graph of the same shape", ur == CUDA_SUCCESS && info.result == 0, (double)info.result);
    CK(cuGraphLaunch(exec, stream));
    CK(cuStreamSynchronize(stream));
    expect("... runs the new parameters", all_equal(get(z), 11.0f));

    // Removing an edge and adding it back.
    CK(cuGraphRemoveDependencies(g, &nkern, &ncopy, nullptr, 1));
    count = 0;
    CK(cuGraphGetEdges(g, nullptr, nullptr, nullptr, &count));
    expect("cuGraphRemoveDependencies", count == 5, (double)count);
    CK(cuGraphAddDependencies(g, &nkern, &ncopy, nullptr, 1));
    count = 0;
    CK(cuGraphGetEdges(g, nullptr, nullptr, nullptr, &count));
    expect("cuGraphAddDependencies", count == 6, (double)count);

    cuGraphExecDestroy(exec2);
    cuGraphExecDestroy(exec);
    cuGraphDestroy(clone);
    cuGraphDestroy(g);
    cuEventDestroy(ev);
  }

  /* ---- child graphs, memory nodes, the tagged-union calls ------------------------------ */
  {
    CUgraph child, g;
    CK(cuGraphCreate(&child, 0));
    CK(cuGraphCreate(&g, 0));
    CUgraphNode cm, nchild;
    CUDA_MEMSET_NODE_PARAMS ms{};
    ms.dst = w, ms.value = 0x41200000u, ms.elementSize = 4, ms.width = N, ms.height = 1;   // 10.0f
    CK(cuGraphAddMemsetNode(&cm, child, nullptr, 0, &ms, ctx));
    CK(cuGraphAddChildGraphNode(&nchild, g, nullptr, 0, child));
    CUgraph back = nullptr;
    CK(cuGraphChildGraphNodeGetGraph(nchild, &back));
    expect("cuGraphChildGraphNodeGetGraph", back != nullptr);

    // A graph allocation: filled by a memset node, copied out, then freed.
    CUDA_MEM_ALLOC_NODE_PARAMS ap{};
    ap.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.poolProps.location.id = 0;
    ap.bytesize = N * 4;
    CUgraphNode nalloc, nfill, ncopy, nfree;
    CK(cuGraphAddMemAllocNode(&nalloc, g, &nchild, 1, &ap));
    expect("a graph allocation names its address", ap.dptr != 0);
    CUDA_MEMSET_NODE_PARAMS fill{};
    fill.dst = ap.dptr, fill.value = 0x41a00000u, fill.elementSize = 4, fill.width = N, fill.height = 1;   // 20.0f
    CK(cuGraphAddMemsetNode(&nfill, g, &nalloc, 1, &fill, ctx));
    CUDA_MEMCPY3D c;
    std::memset(&c, 0, sizeof c);
    c.srcMemoryType = CU_MEMORYTYPE_DEVICE, c.srcDevice = ap.dptr, c.srcPitch = N * 4, c.srcHeight = 1;
    c.dstMemoryType = CU_MEMORYTYPE_DEVICE, c.dstDevice = z, c.dstPitch = N * 4, c.dstHeight = 1;
    c.WidthInBytes = N * 4, c.Height = 1, c.Depth = 1;
    CK(cuGraphAddMemcpyNode(&ncopy, g, &nfill, 1, &c, ctx));
    CK(cuGraphAddMemFreeNode(&nfree, g, &ncopy, 1, ap.dptr));
    CK(cuMemsetD32(w, 0, N));
    CK(cuMemsetD32(z, 0, N));
    CUgraphExec exec;
    CK(cuGraphInstantiateWithFlags(&exec, g, 0));
    CK(cuGraphLaunch(exec, stream));
    CK(cuStreamSynchronize(stream));
    expect("a child graph and graph memory run", all_equal(get(w), 10.0f) && all_equal(get(z), 20.0f));
    CUdeviceptr freed = 0;
    CK(cuGraphMemFreeNodeGetParams(nfree, &freed));
    expect("cuGraphMemFreeNodeGetParams", freed == ap.dptr);
    cuGraphExecDestroy(exec);
    cuGraphDestroy(g);
    cuGraphDestroy(child);
  }
  {
    // cuGraphAddNode with the tagged union, and cuGraphNodeSetParams.
    CUgraph g;
    CK(cuGraphCreate(&g, 0));
    CUgraphNodeParams np;
    std::memset(&np, 0, sizeof np);
    np.type = CU_GRAPH_NODE_TYPE_MEMSET;
    np.memset.dst = w, np.memset.value = 0x40a00000u, np.memset.elementSize = 4, np.memset.width = N, np.memset.height = 1;   // 5.0f
    np.memset.ctx = ctx;
    CUgraphNode n1, n2;
    CK(cuGraphAddNode(&n1, g, nullptr, nullptr, 0, &np));
    float v = 3.0f;
    int n = N;
    void* args[] = {&z, &w, &v, &n};
    std::memset(&np, 0, sizeof np);
    np.type = CU_GRAPH_NODE_TYPE_KERNEL;
    np.kernel.func = addv, np.kernel.gridDimX = 2, np.kernel.gridDimY = 1, np.kernel.gridDimZ = 1;
    np.kernel.blockDimX = 32, np.kernel.blockDimY = 1, np.kernel.blockDimZ = 1, np.kernel.kernelParams = args;
    np.kernel.ctx = ctx;
    CK(cuGraphAddNode(&n2, g, &n1, nullptr, 1, &np));
    CUgraphExec exec;
    CK(cuGraphInstantiateWithFlags(&exec, g, 0));
    CK(cuMemsetD32(z, 0, N));
    CK(cuGraphLaunch(exec, stream));
    CK(cuStreamSynchronize(stream));
    expect("cuGraphAddNode (memset, kernel)", all_equal(get(z), 8.0f));
    std::memset(&np, 0, sizeof np);
    np.type = CU_GRAPH_NODE_TYPE_MEMSET;
    np.memset.dst = w, np.memset.value = 0x40e00000u, np.memset.elementSize = 4, np.memset.width = N, np.memset.height = 1;   // 7.0f
    np.memset.ctx = ctx;
    CK(cuGraphExecNodeSetParams(exec, n1, &np));
    CK(cuGraphLaunch(exec, stream));
    CK(cuStreamSynchronize(stream));
    expect("cuGraphExecNodeSetParams", all_equal(get(z), 10.0f));
    cuGraphExecDestroy(exec);
    cuGraphDestroy(g);
  }

  /* ---- capture into a graph, dependencies, capture info -------------------------------- */
  {
    CUgraph g;
    CK(cuGraphCreate(&g, 0));
    CUgraphNode root;
    CUDA_MEMSET_NODE_PARAMS ms{};
    ms.dst = x, ms.value = 0x3f800000u, ms.elementSize = 4, ms.width = N, ms.height = 1;
    CK(cuGraphAddMemsetNode(&root, g, nullptr, 0, &ms, ctx));
    CK(cuStreamBeginCaptureToGraph(stream, g, &root, nullptr, 1, CU_STREAM_CAPTURE_MODE_GLOBAL));
    CUstreamCaptureStatus status;
    cuuint64_t id = 0;
    CUgraph cg = nullptr;
    const CUgraphNode* deps = nullptr;
    const CUgraphEdgeData* edges = nullptr;
    size_t ndeps = 0;
    CK(cuStreamGetCaptureInfo(stream, &status, &id, &cg, &deps, &edges, &ndeps));
    expect("cuStreamGetCaptureInfo: capturing into the graph, after its one dependency",
           status == CU_STREAM_CAPTURE_STATUS_ACTIVE && id != 0 && cg == g && ndeps == 1 && deps && deps[0] == root);
    float v = 4.0f;
    int n = N;
    void* args[] = {&y, &x, &v, &n};
    CK(cuLaunchKernel(addv, 2, 1, 1, 32, 1, 1, 0, stream, args, nullptr));
    CK(cuStreamGetCaptureInfo(stream, &status, &id, &cg, &deps, &edges, &ndeps));
    expect("... and after a launch, on the new node", ndeps == 1 && deps && deps[0] != root);
    CUgraphNode kernel_node = deps[0];
    CUgraph out = nullptr;
    CK(cuStreamEndCapture(stream, &out));
    expect("capturing into a graph ends in that graph", out == g);
    size_t cnt = 0;
    CK(cuGraphGetNodes(g, nullptr, &cnt));
    expect("the graph has the root and the captured kernel", cnt == 2);
    CUgraphNodeType t;
    CK(cuGraphNodeGetType(kernel_node, &t));
    expect("the captured launch is a kernel node", t == CU_GRAPH_NODE_TYPE_KERNEL);

    // Capture dependencies set by hand: a second node that does not follow the first.
    CK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    CK(cuMemsetD32Async(z, 1, 4, stream));
    CUgraph cg2 = nullptr;
    CK(cuStreamGetCaptureInfo(stream, &status, nullptr, &cg2, &deps, nullptr, &ndeps));
    CUgraphNode first = deps[0];
    CK(cuStreamUpdateCaptureDependencies(stream, nullptr, nullptr, 0, CU_STREAM_SET_CAPTURE_DEPENDENCIES));
    CK(cuMemsetD32Async(z + 64, 2, 4, stream));
    CK(cuStreamGetCaptureInfo(stream, &status, nullptr, &cg2, &deps, nullptr, &ndeps));
    CUgraphNode second = deps[0];
    CUgraphNode both[2] = {first, second};
    CK(cuStreamUpdateCaptureDependencies(stream, both, nullptr, 2, CU_STREAM_SET_CAPTURE_DEPENDENCIES));
    CK(cuMemsetD32Async(z + 128, 3, 4, stream));
    CUgraph cap = nullptr;
    CK(cuStreamEndCapture(stream, &cap));
    size_t nroots = 0, nnodes = 0, nedges = 0;
    CK(cuGraphGetRootNodes(cap, nullptr, &nroots));
    CK(cuGraphGetNodes(cap, nullptr, &nnodes));
    CK(cuGraphGetEdges(cap, nullptr, nullptr, nullptr, &nedges));
    expect("cuStreamUpdateCaptureDependencies: two roots and a join", nroots == 2 && nnodes == 3 && nedges == 2,
           (double)(nroots * 100 + nnodes * 10 + nedges));

    // Instantiate with the parameters form.
    CUDA_GRAPH_INSTANTIATE_PARAMS ip{};
    ip.flags = 0;
    CUgraphExec exec;
    CK(cuGraphInstantiateWithParams(&exec, g, &ip));
    cuuint64_t flags = 99;
    CK(cuGraphExecGetFlags(exec, &flags));
    expect("cuGraphInstantiateWithParams, cuGraphExecGetFlags", flags == 0, (double)flags);
    CK(cuMemsetD32(y, 0, N));
    CK(cuGraphLaunch(exec, stream));
    CK(cuStreamSynchronize(stream));
    expect("the graph with a captured kernel runs", all_equal(get(y), 5.0f));
    cuGraphExecDestroy(exec);
    cuGraphDestroy(cap);
    cuGraphDestroy(g);
  }

  /* ---- user objects --------------------------------------------------------------------- */
  {
    static std::atomic<int> destroyed{0};
    destroyed = 0;
    CUuserObject obj;
    CK(cuUserObjectCreate(&obj, &destroyed, [](void* p) { ++*static_cast<std::atomic<int>*>(p); }, 1, CU_USER_OBJECT_NO_DESTRUCTOR_SYNC));
    CUgraph g;
    CK(cuGraphCreate(&g, 0));
    CK(cuGraphRetainUserObject(g, obj, 1, CU_GRAPH_USER_OBJECT_MOVE));
    expect("a user object the graph owns is alive", destroyed == 0);
    cuGraphDestroy(g);
    cuCtxSynchronize();
    expect("destroying the graph releases it", destroyed == 1, (double)destroyed.load());
  }

  /* ---- what the driver answers for bad calls --------------------------------------------- */
  {
    CUgraph g;
    CK(cuGraphCreate(&g, 0));
    CUgraphNode node;
    CUDA_MEMSET_NODE_PARAMS bad{};
    bad.dst = x, bad.elementSize = 3, bad.width = N, bad.height = 1;
    expect("a memset node of element size 3 is refused", cuGraphAddMemsetNode(&node, g, nullptr, 0, &bad, ctx) == CUDA_ERROR_INVALID_VALUE);
    CUDA_KERNEL_NODE_PARAMS k{};
    expect("a kernel node with no function is refused", cuGraphAddKernelNode(&node, g, nullptr, 0, &k) == CUDA_ERROR_INVALID_HANDLE);
    CUDA_MEMCPY3D zero;
    std::memset(&zero, 0, sizeof zero);
    expect("a zero copy descriptor is refused", cuGraphAddMemcpyNode(&node, g, nullptr, 0, &zero, ctx) == CUDA_ERROR_INVALID_VALUE);
    expect("a null graph is refused", cuGraphAddEmptyNode(&node, nullptr, nullptr, 0) != CUDA_SUCCESS);
    expect("destroying a graph", cuGraphDestroy(g) == CUDA_SUCCESS);
    expect("destroying it again is refused", cuGraphDestroy(g) == CUDA_ERROR_INVALID_VALUE);
  }

  std::puts(fails ? "FAIL" : "PASS");
  return fails != 0;
#endif
}
