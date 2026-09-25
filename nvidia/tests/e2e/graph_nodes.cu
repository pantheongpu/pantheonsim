// The graph node types that are not work on the device: a host function, an
// event recorded, an event waited on -- and a node switched off without
// rebuilding the graph around it.
//
// These are what a program reaches for once a graph is more than a list of
// kernels: bookkeeping that has to happen between two steps, timing a graph
// from outside it, and skipping one step this launch.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
#define WANT(x, want) do { cudaError_t e_ = (x); if (e_ != (want)) { \
  printf("FAIL %s -> %s, expected %s\n", #x, cudaGetErrorString(e_), #want); return 1; } } while (0)
#define CHECK(c) do { if (!(c)) { printf("FAIL %s (line %d)\n", #c, __LINE__); return 1; } } while (0)

// CUDA 13 added an edge-data argument to the dependency and query calls. The
// test speaks whichever spelling the toolkit it is compiled with declares.
static cudaError_t add_dep(cudaGraph_t g, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t n) {
#if CUDART_VERSION >= 13000
  return cudaGraphAddDependencies(g, from, to, nullptr, n);
#else
  return cudaGraphAddDependencies(g, from, to, n);
#endif
}
static cudaError_t count_edges(cudaGraph_t g, size_t* n) {
#if CUDART_VERSION >= 13000
  return cudaGraphGetEdges(g, nullptr, nullptr, nullptr, n);
#else
  return cudaGraphGetEdges(g, nullptr, nullptr, n);
#endif
}

static const int kInts = 256;

__global__ void add_one(int* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1;
}

// What the host nodes do. Plain counters, so a launch is observable from the
// host side without reading device memory at all.
static int g_first = 0, g_second = 0;
static void CUDART_CB bump_first(void* user) { *static_cast<int*>(user) += 1; }
static void CUDART_CB bump_second(void* user) { *static_cast<int*>(user) += 10; }

int main() {
  int* buf = nullptr;
  CK(cudaMalloc(reinterpret_cast<void**>(&buf), kInts * sizeof(int)));
  int host[kInts];

  // ---- a host function as a node -------------------------------------------
  //
  // fill -> kernel -> host function. The host node runs when the graph reaches
  // it, every launch, and it is the last thing the graph does.
  cudaGraph_t g = nullptr;
  CK(cudaGraphCreate(&g, 0));

  cudaGraphNode_t fill = nullptr;
  cudaMemsetParams mp{};
  mp.dst = buf;
  mp.value = 0;
  mp.elementSize = 4;
  mp.width = kInts;
  mp.height = 1;
  CK(cudaGraphAddMemsetNode(&fill, g, nullptr, 0, &mp));

  cudaGraphNode_t kern = nullptr;
  int n = kInts;
  void* args[] = {&buf, &n};
  cudaKernelNodeParams kp{};
  kp.func = reinterpret_cast<void*>(add_one);
  kp.gridDim = dim3((kInts + 63) / 64);
  kp.blockDim = dim3(64);
  kp.kernelParams = args;
  CK(cudaGraphAddKernelNode(&kern, g, &fill, 1, &kp));

  cudaGraphNode_t host_node = nullptr;
  cudaHostNodeParams hp{};
  hp.fn = bump_first;
  hp.userData = &g_first;
  CK(cudaGraphAddHostNode(&host_node, g, &kern, 1, &hp));

  // Read back what the node holds, which is what the program handed it.
  cudaHostNodeParams got{};
  CK(cudaGraphHostNodeGetParams(host_node, &got));
  CHECK(got.fn == bump_first && got.userData == &g_first);
  // A host node is not a kernel node, and the wrong query says so.
  cudaKernelNodeParams wrong{};
  WANT(cudaGraphKernelNodeGetParams(host_node, &wrong), cudaErrorInvalidValue);

  // ---- events as nodes -----------------------------------------------------
  //
  // A wait node before the fill and a record node after the host function: the
  // shape a program uses to join a graph to work outside it.
  cudaEvent_t started = nullptr, finished = nullptr;
  CK(cudaEventCreate(&started));
  CK(cudaEventCreate(&finished));

  cudaGraphNode_t wait_node = nullptr, record_node = nullptr;
  CK(cudaGraphAddEventWaitNode(&wait_node, g, nullptr, 0, started));
  CK(cudaGraphAddEventRecordNode(&record_node, g, &host_node, 1, finished));
  CK(add_dep(g, &wait_node, &fill, 1));

  cudaEvent_t back = nullptr;
  CK(cudaGraphEventRecordNodeGetEvent(record_node, &back));
  CHECK(back == finished);
  CK(cudaGraphEventWaitNodeGetEvent(wait_node, &back));
  CHECK(back == started);
  // Each query answers for its own node type only.
  WANT(cudaGraphEventRecordNodeGetEvent(wait_node, &back), cudaErrorInvalidValue);
  WANT(cudaGraphEventWaitNodeGetEvent(record_node, &back), cudaErrorInvalidValue);
  // An event that no longer exists cannot be put in a node.
  cudaEvent_t gone = nullptr;
  CK(cudaEventCreate(&gone));
  CK(cudaEventDestroy(gone));
  cudaGraphNode_t dead = nullptr;
  WANT(cudaGraphAddEventRecordNode(&dead, g, nullptr, 0, gone), cudaErrorInvalidResourceHandle);

  // The types read back as what they are.
  cudaGraphNodeType type{};
  CK(cudaGraphNodeGetType(host_node, &type));
  CHECK(type == cudaGraphNodeTypeHost);
  CK(cudaGraphNodeGetType(record_node, &type));
  CHECK(type == cudaGraphNodeTypeEventRecord);
  CK(cudaGraphNodeGetType(wait_node, &type));
  CHECK(type == cudaGraphNodeTypeWaitEvent);

  size_t count = 0;
  CK(cudaGraphGetNodes(g, nullptr, &count));
  CHECK(count == 5);

  // ---- launching it --------------------------------------------------------
  cudaGraphExec_t exec = nullptr;
  CK(cudaGraphInstantiate(&exec, g, 0));
  // Uploading before the first launch is what a program does to get the cost
  // out of the way; here there is nothing to upload, and it succeeds.
  CK(cudaGraphUpload(exec, 0));
  CK(cudaGraphLaunch(exec, 0));
  CK(cudaDeviceSynchronize());

  CK(cudaMemcpy(host, buf, sizeof host, cudaMemcpyDeviceToHost));
  CHECK(host[0] == 1 && host[kInts - 1] == 1);
  CHECK(g_first == 1);                  // the host node ran, once
  CK(cudaEventQuery(finished));         // the record node recorded it

  CK(cudaGraphLaunch(exec, 0));
  CK(cudaDeviceSynchronize());
  CHECK(g_first == 2);                  // and again on the next launch

  // ---- a node switched off -------------------------------------------------
  //
  // The kernel is disabled in the instantiated graph. The fill still runs, so
  // the buffer comes back zero: the node is an empty one until it is turned
  // back on, and nothing else about the graph changes.
  unsigned int on = 7;
  CK(cudaGraphNodeGetEnabled(exec, kern, &on));
  CHECK(on == 1);
  CK(cudaGraphNodeSetEnabled(exec, kern, 0));
  CK(cudaGraphNodeGetEnabled(exec, kern, &on));
  CHECK(on == 0);
  CK(cudaGraphLaunch(exec, 0));
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(host, buf, sizeof host, cudaMemcpyDeviceToHost));
  CHECK(host[0] == 0 && host[kInts - 1] == 0);
  CHECK(g_first == 3);                  // the host node was not disabled

  CK(cudaGraphNodeSetEnabled(exec, kern, 1));
  CK(cudaGraphLaunch(exec, 0));
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(host, buf, sizeof host, cudaMemcpyDeviceToHost));
  CHECK(host[0] == 1);

  // Only kernel, memset and memcpy nodes can be switched off, which is what the
  // API documents; a host node cannot.
  WANT(cudaGraphNodeSetEnabled(exec, host_node, 0), cudaErrorInvalidValue);
  // And the node has to belong to the graph this was instantiated from.
  cudaGraph_t other = nullptr;
  CK(cudaGraphCreate(&other, 0));
  cudaGraphNode_t foreign = nullptr;
  CK(cudaGraphAddEmptyNode(&foreign, other, nullptr, 0));
  WANT(cudaGraphNodeSetEnabled(exec, foreign, 0), cudaErrorInvalidValue);

  // ---- new parameters for one node, without rebuilding ---------------------
  //
  // The fill writes 2 instead of zero, set on the instantiated graph. Its
  // elements are 4 bytes, so 2 goes into every int -- not into every byte,
  // which is what this test used to expect, wrongly -- and the kernel then adds
  // one, so each int is 3.
  cudaMemsetParams two = mp;
  two.value = 2;
  CK(cudaGraphExecMemsetNodeSetParams(exec, fill, &two));
  CK(cudaGraphLaunch(exec, 0));
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(host, buf, sizeof host, cudaMemcpyDeviceToHost));
  CHECK(host[0] == 3 && host[kInts - 1] == 3);
  // A 2D fill is refused by this call as invalid, which is how it documents it.
  cudaMemsetParams two_d = two;
  two_d.height = 2;
  WANT(cudaGraphExecMemsetNodeSetParams(exec, fill, &two_d), cudaErrorInvalidValue);
  // And the node named has to be a fill.
  WANT(cudaGraphExecMemsetNodeSetParams(exec, kern, &two), cudaErrorInvalidValue);

  // A different host function for the same node, again only in the instantiated
  // graph: the graph it came from still names the first one.
  cudaHostNodeParams second{};
  second.fn = bump_second;
  second.userData = &g_second;
  CK(cudaGraphExecHostNodeSetParams(exec, host_node, &second));
  CK(cudaGraphLaunch(exec, 0));
  CK(cudaDeviceSynchronize());
  CHECK(g_second == 10);
  CHECK(g_first == 5);                  // unchanged by this launch
  CK(cudaGraphHostNodeGetParams(host_node, &got));
  CHECK(got.fn == bump_first);          // the original graph was not touched

  // A different event for the record node, and it is the one that gets recorded.
  cudaEvent_t other_event = nullptr;
  CK(cudaEventCreate(&other_event));
  CK(cudaGraphExecEventRecordNodeSetEvent(exec, record_node, other_event));
  CK(cudaGraphLaunch(exec, 0));
  CK(cudaDeviceSynchronize());
  CK(cudaEventQuery(other_event));
  WANT(cudaGraphExecEventRecordNodeSetEvent(exec, wait_node, other_event), cudaErrorInvalidValue);

  // ---- a rewired graph is not the same graph -------------------------------
  //
  // Two graphs with the same nodes and the same number of edges, wired
  // differently: one where both kernels wait on the fill, one where the second
  // waits on the first. cudaGraphExecUpdate has to refuse the second as a
  // topology change -- counting nodes and edges cannot tell them apart.
  const auto build = [&](bool chain, cudaGraph_t* out, cudaGraphNode_t* second_kernel) {
    cudaGraph_t h = nullptr;
    if (cudaGraphCreate(&h, 0) != cudaSuccess) return false;
    cudaGraphNode_t f = nullptr, k1 = nullptr, k2 = nullptr;
    if (cudaGraphAddMemsetNode(&f, h, nullptr, 0, &mp) != cudaSuccess) return false;
    if (cudaGraphAddKernelNode(&k1, h, &f, 1, &kp) != cudaSuccess) return false;
    const cudaGraphNode_t dep = chain ? k1 : f;
    if (cudaGraphAddKernelNode(&k2, h, &dep, 1, &kp) != cudaSuccess) return false;
    *out = h;
    if (second_kernel) *second_kernel = k2;
    return true;
  };
  cudaGraph_t forked = nullptr, chained = nullptr;
  CHECK(build(false, &forked, nullptr));
  CHECK(build(true, &chained, nullptr));
  size_t edges = 0;
  CK(count_edges(forked, &edges));
  CHECK(edges == 2);
  CK(count_edges(chained, &edges));
  CHECK(edges == 2);                    // the same count, a different graph

  cudaGraphExec_t fork_exec = nullptr;
  CK(cudaGraphInstantiate(&fork_exec, forked, 0));
  cudaGraphExecUpdateResultInfo info{};
  WANT(cudaGraphExecUpdate(fork_exec, chained, &info), cudaErrorGraphExecUpdateFailure);
  CHECK(info.result == cudaGraphExecUpdateErrorTopologyChanged);
  // Updating from a graph of the same shape is taken.
  cudaGraph_t forked_again = nullptr;
  CHECK(build(false, &forked_again, nullptr));
  std::memset(&info, 0, sizeof info);
  CK(cudaGraphExecUpdate(fork_exec, forked_again, &info));

  // ---- a child graph's parameters --------------------------------------------
  cudaGraph_t parent = nullptr, child = nullptr, child_new = nullptr;
  CK(cudaGraphCreate(&parent, 0));
  CHECK(build(false, &child, nullptr));
  cudaGraphNode_t child_node = nullptr;
  CK(cudaGraphAddChildGraphNode(&child_node, parent, nullptr, 0, child));
  cudaGraphExec_t parent_exec = nullptr;
  CK(cudaGraphInstantiate(&parent_exec, parent, 0));
  CHECK(build(false, &child_new, nullptr));
  CK(cudaGraphExecChildGraphNodeSetParams(parent_exec, child_node, child_new));
  // A child of a different shape is refused.
  cudaGraph_t child_chained = nullptr;
  CHECK(build(true, &child_chained, nullptr));
  WANT(cudaGraphExecChildGraphNodeSetParams(parent_exec, child_node, child_chained),
       cudaErrorInvalidValue);
  CK(cudaGraphLaunch(parent_exec, 0));
  CK(cudaDeviceSynchronize());

  // ---- the drawing names the new node types ---------------------------------
  const char* dot = "/tmp/vgpu-graph-nodes.dot";
  CK(cudaGraphDebugDotPrint(g, dot, 0));
  FILE* f = fopen(dot, "r");
  CHECK(f != nullptr);
  char text[8192];
  const size_t got_bytes = fread(text, 1, sizeof text - 1, f);
  fclose(f);
  text[got_bytes] = '\0';
  CHECK(strstr(text, "HOST") != nullptr);
  CHECK(strstr(text, "EVENT_RECORD") != nullptr);
  CHECK(strstr(text, "WAIT_EVENT") != nullptr);
  remove(dot);

  // A destroyed executable graph is no longer a handle anything accepts.
  CK(cudaGraphExecDestroy(exec));
  WANT(cudaGraphUpload(exec, 0), cudaErrorInvalidValue);

  CK(cudaGraphExecDestroy(fork_exec));
  CK(cudaGraphExecDestroy(parent_exec));
  CK(cudaGraphDestroy(g));
  CK(cudaGraphDestroy(other));
  CK(cudaGraphDestroy(forked));
  CK(cudaGraphDestroy(forked_again));
  CK(cudaGraphDestroy(chained));
  CK(cudaGraphDestroy(child));
  CK(cudaGraphDestroy(child_new));
  CK(cudaGraphDestroy(child_chained));
  CK(cudaGraphDestroy(parent));
  CK(cudaEventDestroy(started));
  CK(cudaEventDestroy(finished));
  CK(cudaEventDestroy(other_event));
  CK(cudaFree(buf));
  printf("PASS\n");
  return 0;
}
