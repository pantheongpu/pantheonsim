// CUDA graphs through CUPTI: the ids a profiler gives graphs, executable graphs
// and their nodes, the callbacks its resource domain raises as a graph is
// built, cloned, instantiated, changed and destroyed, and what a launch
// reports -- one graph-trace record while that kind is wanted, and otherwise
// each node's kernel, copy or fill with the graph and node it belongs to.
// Run against NVIDIA's libcupti on a card it gives
// nvidia/tests/data/cupti_graph.expected (an RTX 3060, CUDA 13.0).
//
// What the card does, and the shim reproduces:
//  * a graph, a capture, a clone and each executable graph take the next graph
//    id; instantiating takes two (the executable graph is a copy of the graph,
//    and the driver makes a second, lowered one it launches from, which is
//    announced node by node without their completion);
//  * a node's id is its graph's in the high half and its place in the graph in
//    the low half, and the launch records carry those of the executable graph;
//  * the callbacks are ordered as the driver orders them, and name the context
//    only for executable graphs.
// Not reproduced: the lowered graph's shape for graphs of kinds not measured
// (here only kernel, fill, copy and empty nodes), and an empty node with
// dependencies being kept or dropped (an isolated one is dropped).
#include "cupti_test_util.h"

using cupti_test::fmt;
using cupti_test::out;

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_) {                                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

#if CUPTI_API_VERSION >= 130000
using KernelRecord = CUpti_ActivityKernel10;
#else
using KernelRecord = CUpti_ActivityKernel9;
#endif
#if CUPTI_API_VERSION >= 22
using GraphTraceRecord = CUpti_ActivityGraphTrace2;
#else
using GraphTraceRecord = CUpti_ActivityGraphTrace;
#endif

namespace {

bool g_report_resources = false;

void CUPTIAPI on_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* data) {
  if (domain != CUPTI_CB_DOMAIN_RESOURCE || !g_report_resources || !data) return;
  if (cbid < CUPTI_CBID_RESOURCE_GRAPH_CREATED || cbid > 23) return;
  const auto* r = static_cast<const CUpti_ResourceData*>(data);
  const auto* g = static_cast<const CUpti_GraphData*>(r->resourceDescriptor);
  auto h = [](const void* p, char tag) {
    return p ? fmt("#%c%llu", tag, (unsigned long long)reinterpret_cast<uintptr_t>(p)) : std::string("-");
  };
  out(fmt("RES %d context=%s graph=%s original=%s node=%s originalNode=%s type=%d dependency=%s exec=%s", (int)cbid,
          r->context ? "set" : "none", h(g->graph, 'g').c_str(), h(g->originalGraph, 'g').c_str(),
          h(g->node, 'n').c_str(), h(g->originalNode, 'n').c_str(), (int)g->nodeType, h(g->dependency, 'n').c_str(),
          h(g->graphExec, 'g').c_str()));
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &r) == CUPTI_SUCCESS) {
    switch (static_cast<int>(r->kind)) {
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
        const auto* k = reinterpret_cast<const KernelRecord*>(r);
        out(fmt("REC kernel name=%s graph=%u node=%llx corr=#c%u", k->name, k->graphId,
                (unsigned long long)k->graphNodeId, k->correlationId));
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMCPY: {
        const auto* m = reinterpret_cast<const CUpti_ActivityMemcpy6*>(r);
        out(fmt("REC memcpy bytes=%llu graph=%u node=%llx corr=#c%u", (unsigned long long)m->bytes, m->graphId,
                (unsigned long long)m->graphNodeId, m->correlationId));
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMSET: {
        const auto* m = reinterpret_cast<const CUpti_ActivityMemset4*>(r);
        out(fmt("REC memset bytes=%llu graph=%u node=%llx corr=#c%u", (unsigned long long)m->bytes, m->graphId,
                (unsigned long long)m->graphNodeId, m->correlationId));
        break;
      }
      case 51: {   // CUPTI_ACTIVITY_KIND_GRAPH_TRACE
        const auto* t = reinterpret_cast<const GraphTraceRecord*>(r);
        out(fmt("REC graph-trace graph=%u start<=end=%d corr=#c%u", t->graphId, t->end >= t->start ? 1 : 0,
                t->correlationId));
        break;
      }
      case CUPTI_ACTIVITY_KIND_RUNTIME: {
        const auto* a = reinterpret_cast<const CUpti_ActivityAPI*>(r);
        const char* name = nullptr;
        cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, a->cbid, &name);
        out(fmt("REC runtime %s corr=#c%u", name ? name : "?", a->correlationId));
        break;
      }
      default: break;
    }
  }
}

void section(const char* name) {
  out(std::string("# ") + name);
  cuptiActivityFlushAll(0);
}

}  // namespace

__global__ void first(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1.f;
}
__global__ void second(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] *= 2.f;
}

static int run(bool with_trace) {
  out(with_trace ? "######## graph trace wanted" : "######## graph trace not wanted");
  const int trace_kind = 51;
  if (with_trace) cuptiActivityEnable(static_cast<CUpti_ActivityKind>(trace_kind));
  else cuptiActivityDisable(static_cast<CUpti_ActivityKind>(trace_kind));

  float *d = nullptr, *e = nullptr, *h = nullptr;
  CK(cudaMalloc(&d, 4096));
  CK(cudaMalloc(&e, 4096));
  CK(cudaMallocHost(&h, 4096));
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  int n = 32;
  void* args[] = {&d, &n};

  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  first<<<1, 32, 0, s>>>(d, n);
  second<<<1, 32, 0, s>>>(d, n);
  CK(cudaMemcpyAsync(e, d, 512, cudaMemcpyDeviceToDevice, s));
  CK(cudaMemsetAsync(e, 0, 256, s));
  CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  section("captured, instantiated");
  CK(cudaGraphLaunch(ge, s));
  CK(cudaGraphLaunch(ge, s));
  CK(cudaStreamSynchronize(s));
  section("launched twice");

  cudaGraph_t g2;
  CK(cudaGraphCreate(&g2, 0));
  cudaGraphNode_t n1, n2;
  cudaKernelNodeParams kp;
  std::memset(&kp, 0, sizeof kp);
  kp.func = reinterpret_cast<void*>(first);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(32);
  kp.kernelParams = args;
  CK(cudaGraphAddKernelNode(&n1, g2, nullptr, 0, &kp));
  CK(cudaGraphAddKernelNode(&n2, g2, &n1, 1, &kp));
  cudaGraphExec_t ge2;
  CK(cudaGraphInstantiate(&ge2, g2, 0));
  CK(cudaGraphLaunch(ge2, s));
  CK(cudaStreamSynchronize(s));
  section("explicit graph");

  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphExecDestroy(ge2));
  CK(cudaGraphDestroy(g));
  CK(cudaGraphDestroy(g2));
  CK(cudaStreamDestroy(s));
  CK(cudaFreeHost(h));
  CK(cudaFree(d));
  CK(cudaFree(e));
  return 0;
}

int main() {
  CUpti_SubscriberHandle sub;
  cuptiSubscribe(&sub, on_callback, nullptr);
  cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RESOURCE);
  cuptiActivityRegisterCallbacks(cupti_test::request_buffer, buffer_completed);
  for (int kind : {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET,
                   CUPTI_ACTIVITY_KIND_RUNTIME})
    cuptiActivityEnable(static_cast<CUpti_ActivityKind>(kind));
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));
  cuptiActivityFlushAll(0);
  cupti_test::lines().clear();

  // ---- the graph ids a profiler reads back ----
  {
    float* d = nullptr;
    CK(cudaMalloc(&d, 4096));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaGraph_t g, g2, g3;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    first<<<1, 32, 0, s>>>(d, 32);
    second<<<1, 32, 0, s>>>(d, 32);
    CK(cudaStreamEndCapture(s, &g));
    uint32_t id = 0;
    cuptiGetGraphId(reinterpret_cast<CUgraph>(g), &id);
    out(fmt("id: captured graph %u", id));
    size_t count = 0;
    CK(cudaGraphGetNodes(g, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    CK(cudaGraphGetNodes(g, nodes.data(), &count));
    for (size_t i = 0; i < count; ++i) {
      uint64_t node_id = 0;
      cuptiGetGraphNodeId(reinterpret_cast<CUgraphNode>(nodes[i]), &node_id);
      out(fmt("id: its node %zu is %llx", i, (unsigned long long)node_id));
    }
    CK(cudaGraphInstantiate(&ge, g, 0));
    cuptiGetGraphExecId(reinterpret_cast<CUgraphExec>(ge), &id);
    out(fmt("id: its executable graph %u", id));
    CK(cudaGraphCreate(&g2, 0));
    cuptiGetGraphId(reinterpret_cast<CUgraph>(g2), &id);
    out(fmt("id: a created graph %u", id));
    CK(cudaGraphClone(&g3, g2));
    cuptiGetGraphId(reinterpret_cast<CUgraph>(g3), &id);
    out(fmt("id: its clone %u", id));
    cudaGraphExec_t ge2;
    CK(cudaGraphInstantiate(&ge2, g2, 0));
    cuptiGetGraphExecId(reinterpret_cast<CUgraphExec>(ge2), &id);
    out(fmt("id: an executable graph of it %u", id));
    cudaGraph_t g4;
    CK(cudaGraphCreate(&g4, 0));
    cuptiGetGraphId(reinterpret_cast<CUgraph>(g4), &id);
    out(fmt("id: the next graph %u", id));
    out(fmt("id: none -> %d %d %d", (int)cuptiGetGraphId(nullptr, &id), (int)cuptiGetGraphExecId(nullptr, &id),
            (int)cuptiGetGraphNodeId(nullptr, reinterpret_cast<uint64_t*>(&id))));
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphExecDestroy(ge2));
    CK(cudaGraphDestroy(g));
    CK(cudaGraphDestroy(g2));
    CK(cudaGraphDestroy(g3));
    CK(cudaGraphDestroy(g4));
    CK(cudaStreamDestroy(s));
    CK(cudaFree(d));
    cuptiActivityFlushAll(0);
    cupti_test::lines().resize(cupti_test::lines().size());   // kept: these ids are compared as they are
  }

  // ---- what a launch reports, with and without the graph-trace kind ----
  if (run(true)) return 1;
  if (run(false)) return 1;

  // ---- the callbacks of the resource domain ----
  out("######## resource callbacks");
  g_report_resources = true;
  {
    float* d = nullptr;
    CK(cudaMalloc(&d, 4096));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    int n = 32;
    void* args[] = {&d, &n};
    out("# capture");
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    first<<<1, 32, 0, s>>>(d, n);
    second<<<1, 32, 0, s>>>(d, n);
    CK(cudaStreamEndCapture(s, &g));
    out("# instantiate");
    CK(cudaGraphInstantiate(&ge, g, 0));
    out("# explicit graph, empty node, dependencies");
    cudaGraph_t g2;
    CK(cudaGraphCreate(&g2, 0));
    cudaGraphNode_t n1, n2, n3;
    cudaKernelNodeParams kp;
    std::memset(&kp, 0, sizeof kp);
    kp.func = reinterpret_cast<void*>(first);
    kp.gridDim = dim3(1);
    kp.blockDim = dim3(32);
    kp.kernelParams = args;
    CK(cudaGraphAddKernelNode(&n1, g2, nullptr, 0, &kp));
    kp.func = reinterpret_cast<void*>(second);
    CK(cudaGraphAddKernelNode(&n2, g2, &n1, 1, &kp));
    CK(cudaGraphAddEmptyNode(&n3, g2, nullptr, 0));
#if CUDART_VERSION >= 13000
    CK(cudaGraphAddDependencies(g2, &n2, &n3, nullptr, 1));
    out("# remove dependency");
    CK(cudaGraphRemoveDependencies(g2, &n2, &n3, nullptr, 1));
#else
    CK(cudaGraphAddDependencies(g2, &n2, &n3, 1));
    out("# remove dependency");
    CK(cudaGraphRemoveDependencies(g2, &n2, &n3, 1));
#endif
    out("# clone");
    cudaGraph_t g3;
    CK(cudaGraphClone(&g3, g2));
    out("# instantiate the explicit graph");
    cudaGraphExec_t ge2;
    CK(cudaGraphInstantiate(&ge2, g2, 0));
    out("# change a node of the executable graph");
    kp.gridDim = dim3(2);
    CK(cudaGraphExecKernelNodeSetParams(ge2, n2, &kp));
    out("# destroy a node");
    CK(cudaGraphDestroyNode(n3));
    out("# destroy");
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphExecDestroy(ge2));
    CK(cudaGraphDestroy(g));
    CK(cudaGraphDestroy(g2));
    CK(cudaGraphDestroy(g3));
    CK(cudaStreamDestroy(s));
    CK(cudaFree(d));
  }
  g_report_resources = false;

  cuptiUnsubscribe(sub);
  cupti_test::print_all();
  std::printf("# end\n");
  return 0;
}
