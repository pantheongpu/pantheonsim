// Stream capture across streams, and every kind of operation a captured
// stream can be given.
//
// A multi-stream program is captured by forking: the stream that began the
// capture records an event, another stream waits on it and so joins the
// capture, does its own work, records an event of its own, and the first stream
// waits on that to join it back. The graph that comes out has two branches.
// Before this, a wait did nothing during capture, a host function ran at capture
// time and never again, and a captured cudaMallocAsync/cudaFreeAsync pair freed
// the memory before the graph ever ran.
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
#define WANT(x, want) do { cudaError_t e_ = (x); if (e_ != (want)) { \
  printf("FAIL %s -> %s, expected %s\n", #x, cudaGetErrorString(e_), #want); return 1; } } while (0)
#define CHECK(c) do { if (!(c)) { printf("FAIL %s (line %d)\n", #c, __LINE__); return 1; } } while (0)

static cudaError_t count_edges(cudaGraph_t g, size_t* n) {
#if CUDART_VERSION >= 13000
  return cudaGraphGetEdges(g, nullptr, nullptr, nullptr, n);
#else
  return cudaGraphGetEdges(g, nullptr, nullptr, n);
#endif
}
static cudaError_t capture_info(cudaStream_t s, cudaStreamCaptureStatus* st,
                                unsigned long long* id, const cudaGraphNode_t** deps,
                                size_t* n) {
#if CUDART_VERSION >= 13000
  return cudaStreamGetCaptureInfo(s, st, id, nullptr, deps, nullptr, n);
#else
  return cudaStreamGetCaptureInfo(s, st, id, nullptr, deps, n);
#endif
}

static const int kN = 128;
__global__ void set(int* p, int v) {
  const int i = threadIdx.x;
  if (i < kN) p[i] = v + i;
}
__global__ void add(const int* a, const int* b, int* out) {
  const int i = threadIdx.x;
  if (i < kN) out[i] = a[i] + b[i];
}

// Every block writes its slot, the grid synchronises, then every block reads
// its neighbour's -- which needs a cooperative launch to mean anything.
__global__ void neighbours(int* slots, int* seen) {
  cg::grid_group grid = cg::this_grid();
  if (threadIdx.x == 0) slots[blockIdx.x] = 50 + static_cast<int>(blockIdx.x);
  grid.sync();
  if (threadIdx.x == 0) seen[blockIdx.x] = slots[(blockIdx.x + 1) % gridDim.x];
}

static int g_host_calls = 0;
static void CUDART_CB count(void*) { ++g_host_calls; }
static void CUDART_CB never(cudaStream_t, cudaError_t, void*) {}

int main() {
  int *a = nullptr, *b = nullptr, *c = nullptr, *out = nullptr;
  CK(cudaMalloc(&a, kN * sizeof(int)));
  CK(cudaMalloc(&b, kN * sizeof(int)));
  CK(cudaMalloc(&c, kN * sizeof(int)));
  CK(cudaMalloc(&out, kN * sizeof(int)));
  cudaStream_t s1, s2, s3;
  CK(cudaStreamCreate(&s1));
  CK(cudaStreamCreate(&s2));
  CK(cudaStreamCreate(&s3));
  cudaEvent_t fork, join;
  CK(cudaEventCreate(&fork));
  CK(cudaEventCreate(&join));
  int h[kN];

  // ---- a fork and a join ------------------------------------------------------
  {
    CK(cudaMemset(out, 0, kN * sizeof(int)));
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeRelaxed));
    set<<<1, kN, 0, s1>>>(a, 1);                   // A
    CK(cudaEventRecord(fork, s1));
    CK(cudaStreamWaitEvent(s2, fork, 0));          // s2 joins the capture here
    cudaStreamCaptureStatus st1, st2;
    unsigned long long id1 = 0, id2 = 0;
    CK(capture_info(s1, &st1, &id1, nullptr, nullptr));
    CK(capture_info(s2, &st2, &id2, nullptr, nullptr));
    CHECK(st2 == cudaStreamCaptureStatusActive && id2 == id1);   // one capture, two streams
    // The event stands for captured work nothing has run: it cannot be queried.
    WANT(cudaEventQuery(fork), cudaErrorCapturedEvent);
    set<<<1, kN, 0, s2>>>(b, 100);                 // B, on the forked stream
    set<<<1, kN, 0, s1>>>(c, 1000);                // C, on the first
    const cudaGraphNode_t* pos = nullptr;
    size_t npos = 0;
    CK(capture_info(s2, &st2, nullptr, &pos, &npos));
    CHECK(npos == 1);                               // s2's own position: B
    CK(cudaEventRecord(join, s2));
    CK(cudaStreamWaitEvent(s1, join, 0));          // the join
    add<<<1, kN, 0, s1>>>(b, c, out);              // D, after both branches
    // A stream that joined cannot end the capture.
    cudaGraph_t wrong = nullptr;
    WANT(cudaStreamEndCapture(s2, &wrong), cudaErrorStreamCaptureUnmatched);
    cudaGraph_t g = nullptr;
    CK(cudaStreamEndCapture(s1, &g));
    CK(cudaStreamIsCapturing(s2, &st2));
    CHECK(st2 == cudaStreamCaptureStatusNone);     // the capture's end released it too
    size_t nodes = 0, edges = 0;
    CK(cudaGraphGetNodes(g, nullptr, &nodes));
    CK(count_edges(g, &edges));
    CHECK(nodes == 4 && edges == 4);               // A -> B, A -> C, B -> D, C -> D
    CK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
    CHECK(h[0] == 0 && h[kN - 1] == 0);            // nothing ran during capture
    cudaGraphExec_t e;
    CK(cudaGraphInstantiate(&e, g, 0));
    CK(cudaGraphLaunch(e, s1));
    CK(cudaStreamSynchronize(s1));
    CK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
    for (int i = 0; i < kN; ++i) CHECK(h[i] == (100 + i) + (1000 + i));
    CK(cudaGraphExecDestroy(e));
    CK(cudaGraphDestroy(g));
    // Recorded again outside a capture, the event is an ordinary one.
    CK(cudaEventRecord(fork, s1));
    CK(cudaEventQuery(fork));
  }

  // ---- what the rules refuse --------------------------------------------------
  {
    // A fork that is never joined back.
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeRelaxed));
    CK(cudaEventRecord(fork, s1));
    CK(cudaStreamWaitEvent(s2, fork, 0));
    set<<<1, kN, 0, s2>>>(b, 5);
    cudaGraph_t g = reinterpret_cast<cudaGraph_t>(0x1);
    WANT(cudaStreamEndCapture(s1, &g), cudaErrorStreamCaptureUnjoined);
    CHECK(g == nullptr);
    cudaStreamCaptureStatus st;
    CK(cudaStreamIsCapturing(s2, &st));
    CHECK(st == cudaStreamCaptureStatusNone);

    // A wait on an event recorded outside the capture crosses its boundary.
    cudaEvent_t outside;
    CK(cudaEventCreate(&outside));
    CK(cudaEventRecord(outside, s3));
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeRelaxed));
    WANT(cudaStreamWaitEvent(s1, outside, 0), cudaErrorStreamCaptureIsolation);
    WANT(cudaStreamEndCapture(s1, &g), cudaErrorStreamCaptureInvalidated);

    // Two captures cannot be merged by a wait.
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeRelaxed));
    CK(cudaStreamBeginCapture(s3, cudaStreamCaptureModeRelaxed));
    CK(cudaEventRecord(fork, s1));
    WANT(cudaStreamWaitEvent(s3, fork, 0), cudaErrorStreamCaptureMerge);
    WANT(cudaStreamEndCapture(s3, &g), cudaErrorStreamCaptureInvalidated);
    // The legacy stream cannot be forked into a capture.
    WANT(cudaStreamWaitEvent(0, fork, 0), cudaErrorStreamCaptureImplicit);
    // A stream already capturing cannot begin again.
    WANT(cudaStreamBeginCapture(s1, cudaStreamCaptureModeRelaxed), cudaErrorIllegalState);
    CK(cudaStreamEndCapture(s1, &g));
    CK(cudaGraphDestroy(g));
    // The legacy stream cannot capture at all.
    WANT(cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed), cudaErrorStreamCaptureUnsupported);

    // A callback is not supported under capture, and neither is synchronizing
    // the capturing stream; each invalidates the capture.
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeRelaxed));
    WANT(cudaStreamAddCallback(s1, never, nullptr, 0), cudaErrorStreamCaptureUnsupported);
    WANT(cudaStreamEndCapture(s1, &g), cudaErrorStreamCaptureInvalidated);
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeRelaxed));
    WANT(cudaStreamSynchronize(s1), cudaErrorStreamCaptureUnsupported);
    WANT(cudaStreamEndCapture(s1, &g), cudaErrorStreamCaptureInvalidated);
    CK(cudaEventDestroy(outside));
  }

  // ---- host functions, external events, and memory the graph owns -------------
  {
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeRelaxed));
    CK(cudaEventRecordWithFlags(t0, s1, cudaEventRecordExternal));   // a node of its own
    int* scratch = nullptr;
    CK(cudaMallocAsync(reinterpret_cast<void**>(&scratch), kN * sizeof(int), s1));
    set<<<1, kN, 0, s1>>>(scratch, 7);
    CK(cudaMemcpyAsync(out, scratch, kN * sizeof(int), cudaMemcpyDeviceToDevice, s1));
    CK(cudaFreeAsync(scratch, s1));
    // Only a graph allocation can be freed by a captured cudaFreeAsync.
    WANT(cudaFreeAsync(a, s1), cudaErrorInvalidValue);
    CK(cudaLaunchHostFunc(s1, count, nullptr));
    CK(cudaEventRecordWithFlags(t1, s1, cudaEventRecordExternal));
    cudaGraph_t g;
    CK(cudaStreamEndCapture(s1, &g));
    CHECK(g_host_calls == 0);                       // not run at capture
    size_t nodes = 0;
    CK(cudaGraphGetNodes(g, nullptr, &nodes));
    CHECK(nodes == 7);   // record, alloc, kernel, copy, free, host, record
    cudaGraphNode_t list[7];
    CK(cudaGraphGetNodes(g, list, &nodes));
    int allocs = 0, frees = 0, hosts = 0, records = 0;
    for (size_t i = 0; i < nodes; ++i) {
      cudaGraphNodeType t;
      CK(cudaGraphNodeGetType(list[i], &t));
      allocs += t == cudaGraphNodeTypeMemAlloc;
      frees += t == cudaGraphNodeTypeMemFree;
      hosts += t == cudaGraphNodeTypeHost;
      records += t == cudaGraphNodeTypeEventRecord;
    }
    CHECK(allocs == 1 && frees == 1 && hosts == 1 && records == 2);

    cudaGraphExec_t e;
    CK(cudaGraphInstantiate(&e, g, 0));
    for (int launch = 1; launch <= 2; ++launch) {
      CK(cudaMemset(out, 0, kN * sizeof(int)));
      CK(cudaGraphLaunch(e, s1));
      CK(cudaStreamSynchronize(s1));
      CK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
      for (int i = 0; i < kN; ++i) CHECK(h[i] == 7 + i);
      CHECK(g_host_calls == launch);                // once per launch
    }
    // The external records ran with the graph, so they time it.
    float ms = -1.0f;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    CHECK(ms >= 0.0f);
    CK(cudaGraphExecDestroy(e));
    CK(cudaGraphDestroy(g));
    CK(cudaEventDestroy(t0));
    CK(cudaEventDestroy(t1));
  }

  // ---- a captured launch keeps how it was launched, and a graph launched on a
  // capturing stream becomes a child of the graph being captured -------------
  {
    const int blocks = 4;
    int *slots = nullptr, *seen = nullptr;
    CK(cudaMalloc(&slots, blocks * sizeof(int)));
    CK(cudaMalloc(&seen, blocks * sizeof(int)));
    CK(cudaMemset(seen, 0, blocks * sizeof(int)));
    // An inner graph of its own, instantiated.
    cudaGraph_t inner_g;
    CK(cudaStreamBeginCapture(s2, cudaStreamCaptureModeRelaxed));
    set<<<1, kN, 0, s2>>>(c, 40);
    cudaGraph_t tmp;
    CK(cudaStreamEndCapture(s2, &tmp));
    inner_g = tmp;
    cudaGraphExec_t inner;
    CK(cudaGraphInstantiate(&inner, inner_g, 0));
    CK(cudaMemset(c, 0, kN * sizeof(int)));

    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeRelaxed));
    void* args[] = {&slots, &seen};
    CK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(neighbours), dim3(blocks), dim3(32),
                                   args, 0, s1));
    CK(cudaGraphLaunch(inner, s1));                 // a child node, not run now
    cudaGraph_t g;
    CK(cudaStreamEndCapture(s1, &g));
    CK(cudaMemcpy(h, c, sizeof h, cudaMemcpyDeviceToHost));
    CHECK(h[0] == 0);                               // the inner graph did not run
    size_t nodes = 0;
    CK(cudaGraphGetNodes(g, nullptr, &nodes));
    CHECK(nodes == 2);
    cudaGraphNode_t list[2];
    CK(cudaGraphGetNodes(g, list, &nodes));
    int children = 0;
    for (size_t i = 0; i < nodes; ++i) {
      cudaGraphNodeType t;
      CK(cudaGraphNodeGetType(list[i], &t));
      children += t == cudaGraphNodeTypeGraph;
    }
    CHECK(children == 1);
    cudaGraphExec_t e;
    CK(cudaGraphInstantiate(&e, g, 0));
    CK(cudaGraphLaunch(e, s1));
    CK(cudaStreamSynchronize(s1));
    // The replay was cooperative: every block saw its neighbour's write, which a
    // plain launch cannot promise -- cooperative_groups would have trapped.
    int got[blocks];
    CK(cudaMemcpy(got, seen, sizeof got, cudaMemcpyDeviceToHost));
    for (int b2 = 0; b2 < blocks; ++b2) CHECK(got[b2] == 50 + (b2 + 1) % blocks);
    CK(cudaMemcpy(h, c, sizeof h, cudaMemcpyDeviceToHost));
    for (int i = 0; i < kN; ++i) CHECK(h[i] == 40 + i);   // and the child ran with it
    CK(cudaGraphExecDestroy(e));
    CK(cudaGraphDestroy(g));
    CK(cudaGraphExecDestroy(inner));
    CK(cudaGraphDestroy(inner_g));
    CK(cudaFree(slots));
    CK(cudaFree(seen));
  }

  CK(cudaEventDestroy(fork));
  CK(cudaEventDestroy(join));
  CK(cudaStreamDestroy(s1));
  CK(cudaStreamDestroy(s2));
  CK(cudaStreamDestroy(s3));
  CK(cudaFree(a));
  CK(cudaFree(b));
  CK(cudaFree(c));
  CK(cudaFree(out));
  printf("PASS\n");
  return 0;
}
