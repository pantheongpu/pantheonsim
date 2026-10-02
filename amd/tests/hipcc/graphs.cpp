// HIP graphs and stream capture, answered as ROCm's HIP answers them on an
// MI300-class device: what AMD's own tests (hip-tests) hold it to. Each check
// prints "ok <what>" or "FAIL <what>: <why>", and the last line counts them.
// Built by build.sh with hipcc; run on two simulated MI300Xs by
// amd/tests/e2e/run_hipcc.sh.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static int checks = 0, failures = 0;
static void check(bool ok, const char* what, const std::string& why = "") {
  ++checks;
  if (ok) {
    std::printf("ok    %s\n", what);
  } else {
    ++failures;
    std::printf("FAIL  %s%s%s\n", what, why.empty() ? "" : ": ", why.c_str());
  }
}
static std::string err(hipError_t e) { return hipGetErrorName(e); }
#define EXPECT(call, want, what) \
  do { \
    const hipError_t got_ = (call); \
    check(got_ == (want), what, "got " + err(got_) + ", want " + err(want)); \
  } while (0)

__global__ void add(int* p, int v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += v;
}
__global__ void twice(int* p, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] *= 2;
}
__global__ void nothing() {}

__device__ int symbol_data[64];

static std::atomic<int> host_calls{0};
static void count_call(void*) { ++host_calls; }
static std::atomic<int> destroyed{0};
static void destroy_object(void*) { ++destroyed; }

static bool all_equal(const std::vector<int>& v, int want) {
  for (int x : v)
    if (x != want) return false;
  return true;
}
static size_t count_of(const std::string& text, const std::string& what) {
  size_t n = 0;
  for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + what.size())) ++n;
  return n;
}

int main() {
  constexpr int N = 256;
  constexpr size_t bytes = N * sizeof(int);
  int* d = nullptr;
  (void)hipMalloc(&d, bytes);
  std::vector<int> host(N, 3), back(N, 0);
  hipStream_t stream;
  (void)hipStreamCreate(&stream);

  // ---- A graph built node by node: copy in, two kernels, copy out, a host node
  hipGraph_t g = nullptr;
  EXPECT(hipGraphCreate(&g, 1), hipErrorInvalidValue, "a graph is made with no flags");
  (void)hipGraphCreate(&g, 0);
  hipGraphNode_t in, k1, k2, out, h;
  EXPECT(hipGraphAddMemcpyNode1D(&in, g, nullptr, 0, d, host.data(), bytes, hipMemcpyHostToDevice), hipSuccess,
         "a copy node");
  int v = 5, n = N;
  void* add_args[] = {&d, &v, &n};
  hipKernelNodeParams kp{};
  kp.func = reinterpret_cast<void*>(add);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(N);
  kp.kernelParams = add_args;
  EXPECT(hipGraphAddKernelNode(&k1, g, &in, 1, &kp), hipSuccess, "a kernel node after it");
  v = 1000;   // the node took its arguments when it was made
  void* twice_args[] = {&d, &n};
  hipKernelNodeParams tp = kp;
  tp.func = reinterpret_cast<void*>(twice);
  tp.kernelParams = twice_args;
  (void)hipGraphAddKernelNode(&k2, g, &k1, 1, &tp);
  (void)hipGraphAddMemcpyNode1D(&out, g, &k2, 1, back.data(), d, bytes, hipMemcpyDeviceToHost);
  hipHostNodeParams hp{count_call, nullptr};
  (void)hipGraphAddHostNode(&h, g, &out, 1, &hp);
  size_t count = 0;
  (void)hipGraphGetNodes(g, nullptr, &count);
  check(count == 5, "the graph has its five nodes");
  hipGraphNode_t nodes[8];
  count = 8;
  (void)hipGraphGetNodes(g, nodes, &count);
  check(count == 5 && nodes[0] == in && nodes[4] == h && nodes[5] == nullptr,
        "in the order they were added, the rest of the array null");
  size_t edges = 0;
  (void)hipGraphGetEdges(g, nullptr, nullptr, &edges);
  check(edges == 4, "and four edges");
  hipGraphNodeType type;
  (void)hipGraphNodeGetType(k1, &type);
  check(type == hipGraphNodeTypeKernel, "a kernel node says so");
  hipKernelNodeParams got{};
  (void)hipGraphKernelNodeGetParams(k1, &got);
  check(got.func == kp.func && *static_cast<int*>(got.kernelParams[1]) == 5,
        "and gives back its function and the arguments it took");

  hipGraphExec_t x = nullptr;
  EXPECT(hipGraphInstantiate(&x, g, nullptr, nullptr, 0), hipSuccess, "it instantiates");
  EXPECT(hipGraphLaunch(x, stream), hipSuccess, "and launches");
  (void)hipStreamSynchronize(stream);
  check(all_equal(back, (3 + 5) * 2) && host_calls == 1, "the launch ran every node in order",
        std::to_string(back[0]) + ", host node ran " + std::to_string(host_calls) + " times");
  std::fill(host.begin(), host.end(), 10);
  (void)hipGraphLaunch(x, stream);
  (void)hipGraphLaunch(x, stream);
  (void)hipStreamSynchronize(stream);
  check(all_equal(back, (10 + 5) * 2) && host_calls == 3, "each launch reads the host memory again");

  // ---- What changes an executable graph, and what does not
  hipGraphNode_t extra;
  (void)hipGraphAddEmptyNode(&extra, g, &h, 1);
  unsigned enabled = 7;
  EXPECT(hipGraphNodeGetEnabled(x, extra, &enabled), hipErrorInvalidValue,
         "a node added after instantiation is not the executable graph's");
  (void)hipGraphNodeGetEnabled(x, k2, &enabled);
  check(enabled == 1, "a kernel node starts enabled");
  EXPECT(hipGraphNodeSetEnabled(x, h, 0), hipErrorInvalidValue, "a host node cannot be disabled");
  (void)hipGraphNodeSetEnabled(x, k2, 0);
  (void)hipGraphLaunch(x, stream);
  (void)hipStreamSynchronize(stream);
  check(all_equal(back, 10 + 5), "a disabled kernel node does not run", std::to_string(back[0]));
  (void)hipGraphNodeSetEnabled(x, k2, 1);
  v = 7;
  kp.kernelParams = add_args;
  (void)hipGraphExecKernelNodeSetParams(x, k1, &kp);
  (void)hipGraphLaunch(x, stream);
  (void)hipStreamSynchronize(stream);
  check(all_equal(back, (10 + 7) * 2), "the executable graph takes new kernel arguments", std::to_string(back[0]));
  EXPECT(hipGraphExecMemcpyNodeSetParams1D(x, in, d, host.data(), bytes, hipMemcpyDeviceToHost), hipErrorInvalidValue,
         "but not a copy in another direction");

  // ---- Topology: cycles, a copy, a child graph, the update of an executable graph
  hipGraph_t cyc;
  (void)hipGraphCreate(&cyc, 0);
  hipGraphNode_t a, b;
  (void)hipGraphAddEmptyNode(&a, cyc, nullptr, 0);
  (void)hipGraphAddEmptyNode(&b, cyc, &a, 1);
  EXPECT(hipGraphAddDependencies(cyc, &a, &b, 1), hipErrorInvalidValue, "an edge twice is refused");
  EXPECT(hipGraphAddDependencies(cyc, &b, &a, 1), hipSuccess, "an edge making a cycle is taken");
  hipGraphExec_t cx = nullptr;
  EXPECT(hipGraphInstantiate(&cx, cyc, nullptr, nullptr, 0), hipErrorInvalidValue, "and refused at instantiation");
  (void)hipGraphRemoveDependencies(cyc, &b, &a, 1);
  EXPECT(hipGraphInstantiate(&cx, cyc, nullptr, nullptr, 0), hipSuccess, "until the cycle is broken");
  (void)hipGraphExecDestroy(cx);
  (void)hipGraphDestroy(cyc);

  hipGraph_t clone;
  (void)hipGraphClone(&clone, g);
  hipGraphNode_t found = nullptr;
  (void)hipGraphNodeFindInClone(&found, k1, clone);
  (void)hipGraphNodeGetType(found, &type);
  check(found && found != k1 && type == hipGraphNodeTypeKernel, "a node's copy is found in the clone");
  hipGraph_t parent;
  (void)hipGraphCreate(&parent, 0);
  hipGraphNode_t child;
  (void)hipGraphAddChildGraphNode(&child, parent, nullptr, 0, clone);
  (void)hipGraphDestroy(clone);   // the node has a copy of its own
  hipGraphExec_t px;
  std::fill(host.begin(), host.end(), 1);
  (void)hipGraphInstantiate(&px, parent, nullptr, nullptr, 0);
  (void)hipGraphLaunch(px, stream);
  (void)hipStreamSynchronize(stream);
  check(all_equal(back, (1 + 5) * 2), "a child graph runs in its node's place, from its own copy",
        std::to_string(back[0]));
  hipGraphExecUpdateResult result;
  hipGraphNode_t bad = nullptr;
  EXPECT(hipGraphExecUpdate(px, g, &bad, &result), hipErrorGraphExecUpdateFailure,
         "an executable graph is not updated to a graph of another shape");
  check(result == hipGraphExecUpdateErrorTopologyChanged, "which says the topology changed");
  (void)hipGraphExecDestroy(px);
  (void)hipGraphDestroy(parent);

  // ---- Capture: a fork and a join through events, and what comes out
  hipStream_t side;
  (void)hipStreamCreate(&side);
  hipEvent_t fork, join;
  (void)hipEventCreate(&fork);
  (void)hipEventCreate(&join);
  std::fill(host.begin(), host.end(), 2);
  EXPECT(hipStreamBeginCapture(nullptr, hipStreamCaptureModeGlobal), hipErrorStreamCaptureUnsupported,
         "the null stream is not captured");
  EXPECT(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess, "a capture begins");
  EXPECT(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipErrorIllegalState, "only once");
  (void)hipMemcpyAsync(d, host.data(), bytes, hipMemcpyHostToDevice, stream);
  (void)hipEventRecord(fork, stream);
  (void)hipStreamWaitEvent(side, fork, 0);
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  unsigned long long id = 0, side_id = 0;
  (void)hipStreamGetCaptureInfo(stream, &status, &id);
  (void)hipStreamGetCaptureInfo(side, &status, &side_id);
  check(status == hipStreamCaptureStatusActive && id == side_id && id, "a stream that waits joins the capture");
  add<<<1, N, 0, side>>>(d, 4, N);
  (void)hipEventRecord(join, side);
  EXPECT(hipStreamEndCapture(side, &clone), hipErrorStreamCaptureUnmatched, "a joined stream cannot end it");
  (void)hipStreamWaitEvent(stream, join, 0);
  twice<<<1, N, 0, stream>>>(d, N);
  (void)hipMemcpyAsync(back.data(), d, bytes, hipMemcpyDeviceToHost, stream);
  (void)hipLaunchHostFunc(stream, count_call, nullptr);
  const hipGraphNode_t* deps = nullptr;
  size_t ndeps = 0;
  hipGraph_t live = nullptr;
  (void)hipStreamGetCaptureInfo_v2(stream, &status, &id, &live, &deps, &ndeps);
  check(live && ndeps == 1, "the capture's graph so far, and the one node the stream is at");
  EXPECT(hipMemcpy(back.data(), d, 4, hipMemcpyDeviceToHost), hipErrorStreamCaptureImplicit,
         "a copy on the legacy stream is refused while a blocking stream captures");
  hipGraph_t captured = nullptr;
  EXPECT(hipStreamEndCapture(stream, &captured), hipErrorStreamCaptureInvalidated, "and gives the capture up");
  check(captured == nullptr, "which hands back no graph");
  (void)hipStreamIsCapturing(side, &status);
  check(status == hipStreamCaptureStatusNone, "and every stream it took in is its own again");

  (void)hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
  (void)hipMemcpyAsync(d, host.data(), bytes, hipMemcpyHostToDevice, stream);
  (void)hipEventRecord(fork, stream);
  (void)hipStreamWaitEvent(side, fork, 0);
  add<<<1, N, 0, side>>>(d, 4, N);
  (void)hipEventRecord(join, side);
  (void)hipStreamWaitEvent(stream, join, 0);
  twice<<<1, N, 0, stream>>>(d, N);
  (void)hipMemcpyAsync(back.data(), d, bytes, hipMemcpyDeviceToHost, stream);
  (void)hipLaunchHostFunc(stream, count_call, nullptr);
  void* p = nullptr;
  EXPECT(hipMalloc(&p, 16), hipErrorStreamCaptureUnsupported, "hipMalloc is unsafe during a global capture");
  check(p == nullptr, "and leaves its pointer alone");
  (void)hipStreamEndCapture(stream, &captured);
  check(captured == nullptr, "which gives the capture up too");

  (void)hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
  (void)hipMemcpyAsync(d, host.data(), bytes, hipMemcpyHostToDevice, stream);
  (void)hipEventRecord(fork, stream);
  (void)hipStreamWaitEvent(side, fork, 0);
  add<<<1, N, 0, side>>>(d, 4, N);
  EXPECT(hipStreamEndCapture(stream, &captured), hipErrorStreamCaptureUnjoined,
         "a capture whose fork was never joined does not end well");
  (void)hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
  (void)hipMemcpyAsync(d, host.data(), bytes, hipMemcpyHostToDevice, stream);
  (void)hipEventRecord(fork, stream);
  (void)hipStreamWaitEvent(side, fork, 0);
  add<<<1, N, 0, side>>>(d, 4, N);
  (void)hipEventRecord(join, side);
  (void)hipStreamWaitEvent(stream, join, 0);
  twice<<<1, N, 0, stream>>>(d, N);
  (void)hipMemcpyAsync(back.data(), d, bytes, hipMemcpyDeviceToHost, stream);
  (void)hipLaunchHostFunc(stream, count_call, nullptr);
  EXPECT(hipStreamEndCapture(stream, &captured), hipSuccess, "a capture joined back ends");
  (void)hipGraphGetNodes(captured, nullptr, &count);
  (void)hipGraphGetEdges(captured, nullptr, nullptr, &edges);
  // The stream after the join follows both its own copy and the fork's kernel.
  check(count == 5 && edges == 5, "with a node for each operation and none for the events",
        std::to_string(count) + " nodes, " + std::to_string(edges) + " edges");
  std::fill(back.begin(), back.end(), 0);
  check(all_equal(back, 0), "nothing ran while it was captured");
  const int calls_before = host_calls;
  (void)hipGraphInstantiate(&x, captured, nullptr, nullptr, 0);
  (void)hipGraphLaunch(x, side);
  (void)hipStreamSynchronize(side);
  check(all_equal(back, (2 + 4) * 2) && host_calls == calls_before + 1, "the captured graph runs on another stream",
        std::to_string(back[0]));
  (void)hipGraphExecDestroy(x);
  (void)hipGraphDestroy(captured);

  // ---- Capture modes: relaxed allows what global refuses
  hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
  (void)hipThreadExchangeStreamCaptureMode(&mode);
  check(mode == hipStreamCaptureModeGlobal, "a thread starts in global mode");
  (void)hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed);
  EXPECT(hipMalloc(&p, 16), hipSuccess, "hipMalloc is allowed in a relaxed capture");
  (void)hipFree(p);
  EXPECT(hipStreamEndCapture(stream, &captured), hipSuccess, "which stays whole");
  (void)hipGraphDestroy(captured);
  (void)hipThreadExchangeStreamCaptureMode(&mode);
  int bogus = 3;
  EXPECT(hipThreadExchangeStreamCaptureMode(reinterpret_cast<hipStreamCaptureMode*>(&bogus)), hipErrorInvalidValue,
         "a mode that is not one is refused");

  // ---- Event nodes: the times between records, and a wait on another graph's record
  hipGraph_t eg;
  (void)hipGraphCreate(&eg, 0);
  hipEvent_t start, stop;
  (void)hipEventCreate(&start);
  (void)hipEventCreate(&stop);
  hipGraphNode_t r1, fill, r2;
  (void)hipGraphAddEventRecordNode(&r1, eg, nullptr, 0, start);
  hipMemsetParams mp{};
  mp.dst = d;
  mp.elementSize = 4;
  mp.width = N;
  mp.height = 1;
  mp.value = 0x11;
  (void)hipGraphAddMemsetNode(&fill, eg, &r1, 1, &mp);
  (void)hipGraphAddEventRecordNode(&r2, eg, &fill, 1, stop);
  (void)hipGraphInstantiate(&x, eg, nullptr, nullptr, 0);
  (void)hipGraphLaunch(x, stream);
  (void)hipEventSynchronize(stop);
  float ms = 0;
  EXPECT(hipEventElapsedTime(&ms, start, stop), hipSuccess, "record nodes time what runs between them");
  check(ms > 0, "and the time is never zero");
  (void)hipMemcpy(back.data(), d, bytes, hipMemcpyDeviceToHost);
  check(all_equal(back, 0x11), "a memset node fills 4-byte elements");
  (void)hipGraphExecDestroy(x);
  (void)hipGraphDestroy(eg);

  // ---- A 2D memset node in an executable graph takes new parameters of
  // the same shape (another allocation), as ROCm's HIP allows, and refuses
  // another shape.
  {
    char *p1 = nullptr, *p2 = nullptr;
    size_t pitch = 0;
    (void)hipMallocPitch(reinterpret_cast<void**>(&p1), &pitch, 64, 8);
    (void)hipMallocPitch(reinterpret_cast<void**>(&p2), &pitch, 64, 8);
    hipMemsetParams m2{};
    m2.dst = p1;
    m2.elementSize = 1;
    m2.width = 64;
    m2.height = 8;
    m2.pitch = pitch;
    m2.value = 1;
    hipGraph_t g2;
    hipGraphNode_t n2;
    hipGraphExec_t x2;
    (void)hipGraphCreate(&g2, 0);
    (void)hipGraphAddMemsetNode(&n2, g2, nullptr, 0, &m2);
    (void)hipGraphInstantiate(&x2, g2, nullptr, nullptr, 0);
    m2.dst = p2;
    m2.value = 0x5a;
    EXPECT(hipGraphExecMemsetNodeSetParams(x2, n2, &m2), hipSuccess,
           "an executable 2D memset node takes another allocation of its shape");
    (void)hipGraphLaunch(x2, stream);
    (void)hipStreamSynchronize(stream);
    std::vector<char> rows(64 * 8, 0);
    (void)hipMemcpy2D(rows.data(), 64, p2, pitch, 64, 8, hipMemcpyDeviceToHost);
    bool filled = true;
    for (char c : rows) filled &= c == 0x5a;
    check(filled, "and fills it");
    m2.height = 4;
    EXPECT(hipGraphExecMemsetNodeSetParams(x2, n2, &m2), hipErrorInvalidValue, "but not one of another shape");
    (void)hipGraphExecDestroy(x2);
    (void)hipGraphDestroy(g2);
    (void)hipFree(p1);
    (void)hipFree(p2);
  }

  // ---- Graph memory: an allocation node's address, live beyond the graph
  hipGraph_t mg;
  (void)hipGraphCreate(&mg, 0);
  hipMemAllocNodeParams ap{};
  ap.poolProps.allocType = hipMemAllocationTypePinned;
  ap.poolProps.location = {hipMemLocationTypeDevice, 0};
  ap.bytesize = bytes;
  hipGraphNode_t alloc_node, set_node;
  EXPECT(hipGraphAddMemAllocNode(&alloc_node, mg, nullptr, 0, &ap), hipSuccess, "an allocation node");
  check(ap.dptr != nullptr, "hands back its address when it is made");
  mp.dst = ap.dptr;
  mp.value = 0x22;
  (void)hipGraphAddMemsetNode(&set_node, mg, &alloc_node, 1, &mp);
  hipGraphExec_t mx, mx2;
  (void)hipGraphInstantiate(&mx, mg, nullptr, nullptr, 0);
  EXPECT(hipGraphInstantiate(&mx2, mg, nullptr, nullptr, 0), hipErrorNotSupported,
         "a graph that allocates is instantiated once at a time");
  // A graph that frees is held to the same, as ROCm's HIP holds it; this one
  // is never launched, so the memory stays.
  hipGraph_t fg;
  (void)hipGraphCreate(&fg, 0);
  hipGraphNode_t free_node;
  EXPECT(hipGraphAddMemFreeNode(&free_node, fg, nullptr, 0, ap.dptr), hipSuccess, "a free node");
  hipGraphExec_t fx, fx2;
  (void)hipGraphInstantiate(&fx, fg, nullptr, nullptr, 0);
  EXPECT(hipGraphInstantiate(&fx2, fg, nullptr, nullptr, 0), hipErrorNotSupported,
         "and a graph that frees is instantiated once at a time too");
  (void)hipGraphExecDestroy(fx);
  (void)hipGraphDestroy(fg);
  hipGraph_t mclone;
  EXPECT(hipGraphClone(&mclone, mg), hipErrorNotSupported, "nor cloned");
  EXPECT(hipGraphDestroyNode(alloc_node), hipErrorNotSupported, "and its allocation node stays");
  (void)hipGraphLaunch(mx, stream);
  (void)hipStreamSynchronize(stream);
  uint64_t used = 0;
  (void)hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrUsedMemCurrent, &used);
  check(used == bytes, "the device counts what the graph allocated", std::to_string(used));
  (void)hipGraphExecDestroy(mx);
  (void)hipGraphDestroy(mg);
  (void)hipMemcpy(back.data(), ap.dptr, bytes, hipMemcpyDeviceToHost);
  check(all_equal(back, 0x22), "graph memory outlives its graph");
  size_t info = 0;
  (void)hipMemPtrGetInfo(ap.dptr, &info);
  check(info == bytes, "and has its size");
  EXPECT(hipFree(ap.dptr), hipSuccess, "and is freed with hipFree");
  (void)hipDeviceGraphMemTrim(0);
  uint64_t reserved = 1, high = 0;
  (void)hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrReservedMemCurrent, &reserved);
  (void)hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrUsedMemHigh, &high);
  check(reserved == 0 && high == bytes, "a trim gives it back; the high-water mark stays");

  // ---- Symbols, user objects and a drawing
  hipGraph_t sg;
  (void)hipGraphCreate(&sg, 0);
  hipGraphNode_t to_sym, from_sym;
  std::fill(host.begin(), host.end(), 9);
  EXPECT(hipGraphAddMemcpyNodeToSymbol(&to_sym, sg, nullptr, 0, nullptr, host.data(), 64 * sizeof(int), 0,
                                       hipMemcpyHostToDevice),
         hipErrorInvalidSymbol, "a copy to no symbol");
  EXPECT(hipGraphAddMemcpyNodeToSymbol(&to_sym, sg, nullptr, 0, HIP_SYMBOL(symbol_data), host.data(),
                                       65 * sizeof(int), 0, hipMemcpyHostToDevice),
         hipErrorInvalidValue, "a copy past the symbol's end");
  (void)hipGraphAddMemcpyNodeToSymbol(&to_sym, sg, nullptr, 0, HIP_SYMBOL(symbol_data), host.data(), 64 * sizeof(int),
                                      0, hipMemcpyHostToDevice);
  (void)hipGraphAddMemcpyNodeFromSymbol(&from_sym, sg, &to_sym, 1, back.data(), HIP_SYMBOL(symbol_data),
                                        64 * sizeof(int), 0, hipMemcpyDeviceToHost);
  hipUserObject_t object;
  (void)hipUserObjectCreate(&object, nullptr, destroy_object, 1, hipUserObjectNoDestructorSync);
  (void)hipGraphRetainUserObject(sg, object, 1, hipGraphUserObjectMove);
  (void)hipGraphInstantiate(&x, sg, nullptr, nullptr, 0);
  std::fill(back.begin(), back.end(), 0);
  (void)hipGraphLaunch(x, stream);
  (void)hipStreamSynchronize(stream);
  check(back[0] == 9 && back[63] == 9 && back[64] == 0, "copy nodes to and from a __device__ variable");
  const char* dot = "graphs-test.dot";
  (void)hipGraphDebugDotPrint(sg, dot, 0);
  std::ifstream f(dot);
  std::stringstream text;
  text << f.rdbuf();
  check(count_of(text.str(), "->") == 1 && count_of(text.str(), "MEMCPY") == 2 && count_of(text.str(), "HtoD") == 1,
        "a graph is drawn, node and edge");
  std::remove(dot);
  EXPECT(hipGraphDebugDotPrint(sg, "", 0), hipErrorOperatingSystem, "into no file");
  (void)hipGraphDestroy(sg);
  check(destroyed == 0, "a user object lives while an executable graph holds it");
  (void)hipGraphExecDestroy(x);
  check(destroyed == 1, "and is destroyed once with the last reference");
  EXPECT(hipUserObjectRelease(object, 1), hipSuccess, "after which releasing it changes nothing");

  // ---- Handles that are not
  EXPECT(hipGraphLaunch(x, stream), hipErrorInvalidValue, "a destroyed executable graph");
  EXPECT(hipGraphDestroyNode(reinterpret_cast<hipGraphNode_t>(0x1234)), hipErrorInvalidValue, "a node that never was");
  EXPECT(hipStreamEndCapture(stream, &captured), hipErrorIllegalState, "ending a capture that is not going on");
  (void)hipGetLastError();

  (void)hipGraphExecDestroy(x);
  (void)hipGraphDestroy(g);
  (void)hipStreamDestroy(side);
  (void)hipStreamDestroy(stream);
  (void)hipFree(d);
  std::printf("graphs: %d checks, %d failed\n", checks, failures);
  return failures ? 1 : 0;
}
