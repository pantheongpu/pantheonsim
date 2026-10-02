// Conditional graph nodes (CUDA 12.3; IF/ELSE and SWITCH 12.8): a node whose
// body graphs run or not on a value a kernel in the graph sets with
// cudaGraphSetConditional -- once if it is nonzero (IF, with an optional ELSE
// body), for as long as it is nonzero (WHILE), or the body it numbers
// (SWITCH). And cudaStreamBeginCaptureToGraph, which fills a body by capture.
// Every expected value is an RTX 3060's:
//
//   - with cudaGraphCondAssignDefault each launch sets a handle to its default
//     as the launch begins -- once, not again each time a body runs; without
//     the flag the default is never applied, and the value carries over from
//     one launch to the next (and is undefined before anything sets it: on the
//     card, whatever a destroyed graph's handle left behind);
//   - a graph with a conditional node may have one instantiation at a time,
//     cannot be cloned or made a child (cudaErrorNotSupported), and fails to
//     instantiate (cudaErrorInvalidValue) when one of its handles decides no
//     node, or a body holds a host node;
//   - a body belongs to its node: cudaGraphDestroy refuses it, and
//     instantiating it alone is cudaErrorNotSupported;
//   - a kernel outside a graph that calls cudaGraphSetConditional faults with
//     an illegal address.
#include <cuda_runtime.h>

#include <cstdio>

#if CUDART_VERSION >= 12030

static int fails = 0;
#define WANT(x, want) do { const int got_ = static_cast<int>(x); if (got_ != static_cast<int>(want)) { \
  std::printf("FAIL line %d: %s -> %d, expected %s (%d)\n", __LINE__, #x, got_, #want, static_cast<int>(want)); \
  ++fails; } } while (0)
#define CK(x) WANT(x, cudaSuccess)

__global__ void bump(int* c) { atomicAdd(c, 1); }
__global__ void bump_by(int* c, int by) { atomicAdd(c, by); }
// The condition from memory the host sets before each launch.
__global__ void set_from(const int* v, cudaGraphConditionalHandle h) { cudaGraphSetConditional(h, *v); }
__global__ void set_to(cudaGraphConditionalHandle h, unsigned v) { cudaGraphSetConditional(h, v); }
// A loop body: counts its runs in c[1], and stops the loop after `limit`.
__global__ void count_down(int* c, int limit, cudaGraphConditionalHandle h) {
  if (atomicAdd(c + 1, 1) + 1 >= limit) cudaGraphSetConditional(h, 0);
}

static int* g_count = nullptr;   // [0]: body runs, [1]: loop iterations
static int* g_value = nullptr;   // what set_from reads

static int take(int i = 0) {
  int v = 0;
  cudaMemcpy(&v, g_count + i, sizeof v, cudaMemcpyDeviceToHost);
  cudaMemset(g_count + i, 0, sizeof v);
  return v;
}
static void set_value(int v) { cudaMemcpy(g_value, &v, sizeof v, cudaMemcpyHostToDevice); }

static cudaError_t add_kernel(cudaGraphNode_t* n, cudaGraph_t g, cudaGraphNode_t* deps, size_t nd,
                              void* func, void** args) {
  cudaKernelNodeParams p = {};
  p.func = func;
  p.gridDim = dim3(1);
  p.blockDim = dim3(1);
  p.kernelParams = args;
  return cudaGraphAddKernelNode(n, g, deps, nd, &p);
}

// A conditional node; its bodies land in `bodies`.
static cudaError_t add_conditional(cudaGraphNode_t* n, cudaGraph_t g, cudaGraphNode_t* deps, size_t nd,
                                   cudaGraphConditionalHandle h, int type, unsigned size,
                                   cudaGraph_t* bodies) {
  cudaGraphNodeParams p = {};
  p.type = cudaGraphNodeTypeConditional;
  p.conditional.handle = h;
  p.conditional.type = static_cast<cudaGraphConditionalNodeType>(type);
  p.conditional.size = size;
#if CUDART_VERSION >= 13000
  const cudaError_t e = cudaGraphAddNode(n, g, deps, nullptr, nd, &p);
#else
  const cudaError_t e = cudaGraphAddNode(n, g, deps, nd, &p);
#endif
  if (e == cudaSuccess)
    for (unsigned i = 0; i < size; ++i) bodies[i] = p.conditional.phGraph_out[i];
  return e;
}
static const int kIf = 0, kWhile = 1, kSwitch = 2;   // cudaGraphCondTypeIf, While, Switch

static cudaError_t capture_info(cudaStream_t s, cudaGraph_t* g, const cudaGraphNode_t** deps, size_t* n) {
  cudaStreamCaptureStatus st;
#if CUDART_VERSION >= 13000
  return cudaStreamGetCaptureInfo(s, &st, nullptr, g, deps, nullptr, n);
#else
  return cudaStreamGetCaptureInfo_v2(s, &st, nullptr, g, deps, n);
#endif
}

int main() {
  CK(cudaMalloc(&g_count, 2 * sizeof(int)));
  CK(cudaMalloc(&g_value, sizeof(int)));
  cudaMemset(g_count, 0, 2 * sizeof(int));
  void* bump_args[] = {&g_count};

  // ---- handles ---------------------------------------------------------------
  {
    cudaGraph_t g;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle h;
    WANT(cudaGraphConditionalHandleCreate(&h, g, 1, 2), cudaErrorInvalidValue);   // flags: 0 or AssignDefault
    WANT(cudaGraphConditionalHandleCreate(&h, nullptr, 1, 0), cudaErrorInvalidValue);
    WANT(cudaGraphConditionalHandleCreate(nullptr, g, 1, 0), cudaErrorInvalidValue);
    CK(cudaGraphConditionalHandleCreate(&h, g, 0, 0));
    WANT(h != 0, true);
    cudaGraphNode_t n;
    cudaGraph_t b[4];
    // The sizes each kind takes, and the kinds there are.
    WANT(add_conditional(&n, g, nullptr, 0, h, kWhile, 2, b), cudaErrorInvalidValue);
    WANT(add_conditional(&n, g, nullptr, 0, h, kIf, 3, b), cudaErrorInvalidValue);
    WANT(add_conditional(&n, g, nullptr, 0, h, kSwitch, 0, b), cudaErrorInvalidValue);
    WANT(add_conditional(&n, g, nullptr, 0, h, 3, 1, b), cudaErrorInvalidValue);
    WANT(add_conditional(&n, g, nullptr, 0, 0, kIf, 1, b), cudaErrorInvalidValue);   // no such handle
    CK(add_conditional(&n, g, nullptr, 0, h, kIf, 1, b));
    cudaGraphNodeType t;
    CK(cudaGraphNodeGetType(n, &t));
    WANT(t, cudaGraphNodeTypeConditional);
    // A handle decides one node.
    WANT(add_conditional(&n, g, nullptr, 0, h, kIf, 1, b), cudaErrorInvalidValue);
    // And only in the graph it was made for, or a body inside it.
    cudaGraph_t other;
    CK(cudaGraphCreate(&other, 0));
    cudaGraphConditionalHandle ho;
    CK(cudaGraphConditionalHandleCreate(&ho, other, 0, 0));
    WANT(add_conditional(&n, g, nullptr, 0, ho, kIf, 1, b), cudaErrorInvalidValue);
    CK(cudaGraphDestroy(other));
    CK(cudaGraphDestroy(g));
  }

  // ---- IF: the body runs when a kernel before it set a nonzero value --------
  {
    cudaGraph_t g;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle h;
    CK(cudaGraphConditionalHandleCreate(&h, g, 0, 0));
    void* args[] = {&g_value, &h};
    cudaGraphNode_t k, n, unused;
    cudaGraph_t b[1];
    CK(add_kernel(&k, g, nullptr, 0, reinterpret_cast<void*>(set_from), args));
    CK(add_conditional(&n, g, &k, 1, h, kIf, 1, b));
    CK(add_kernel(&unused, b[0], nullptr, 0, reinterpret_cast<void*>(bump), bump_args));
    size_t count = 0;
    CK(cudaGraphGetNodes(g, nullptr, &count));
    WANT(count, 2);   // the body's node is the body's
    cudaGraphExec_t x;
    CK(cudaGraphInstantiate(&x, g, 0));
    for (int v : {0, 1, 0, 7}) {
      set_value(v);
      CK(cudaGraphLaunch(x, 0));
      CK(cudaDeviceSynchronize());
      WANT(take(), v ? 1 : 0);
    }
    // One instantiation at a time; no clone; not a child; the body is not the
    // program's to destroy or to instantiate.
    cudaGraphExec_t x2;
    WANT(cudaGraphInstantiate(&x2, g, 0), cudaErrorNotSupported);
    cudaGraph_t copy;
    WANT(cudaGraphClone(&copy, g), cudaErrorNotSupported);
    cudaGraph_t outer;
    CK(cudaGraphCreate(&outer, 0));
    cudaGraphNode_t child;
    WANT(cudaGraphAddChildGraphNode(&child, outer, nullptr, 0, g), cudaErrorNotSupported);
    CK(cudaGraphDestroy(outer));
    WANT(cudaGraphDestroy(b[0]), cudaErrorInvalidValue);
    WANT(cudaGraphInstantiate(&x2, b[0], 0), cudaErrorNotSupported);
    CK(cudaGraphExecDestroy(x));
    CK(cudaGraphInstantiate(&x, g, 0));   // once the first is gone
    CK(cudaGraphExecDestroy(x));
    CK(cudaGraphDestroy(g));
  }

  // ---- the default, with and without cudaGraphCondAssignDefault ------------
  {
    // Without the flag the default is never applied, and what a launch set is
    // what the next launch starts with. The first launch's value is undefined.
    cudaGraph_t g;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle h;
    CK(cudaGraphConditionalHandleCreate(&h, g, 5, 0));
    unsigned one = 1;
    void* args[] = {&h, &one};
    cudaGraphNode_t n, k, unused;
    cudaGraph_t b[1];
    CK(add_conditional(&n, g, nullptr, 0, h, kIf, 1, b));
    CK(add_kernel(&unused, b[0], nullptr, 0, reinterpret_cast<void*>(bump), bump_args));
    CK(add_kernel(&k, g, &n, 1, reinterpret_cast<void*>(set_to), args));   // after the conditional
    cudaGraphExec_t x;
    CK(cudaGraphInstantiate(&x, g, 0));
    CK(cudaGraphLaunch(x, 0));
    take();
    CK(cudaGraphLaunch(x, 0));
    WANT(take(), 1);   // the previous launch's 1, not the default 5's
    CK(cudaGraphExecDestroy(x));
    CK(cudaGraphInstantiate(&x, g, 0));
    CK(cudaGraphLaunch(x, 0));
    WANT(take(), 1);   // and across instantiations
    CK(cudaGraphExecDestroy(x));
    CK(cudaGraphDestroy(g));
  }
  {
    // WHILE, defaulted to 1: the body runs until it clears the value, and the
    // next launch starts at 1 again.
    cudaGraph_t g;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle h;
    CK(cudaGraphConditionalHandleCreate(&h, g, 1, cudaGraphCondAssignDefault));
    int limit = 3;
    void* args[] = {&g_count, &limit, &h};
    cudaGraphNode_t n, unused;
    cudaGraph_t b[1];
    CK(add_conditional(&n, g, nullptr, 0, h, kWhile, 1, b));
    CK(add_kernel(&unused, b[0], nullptr, 0, reinterpret_cast<void*>(count_down), args));
    cudaGraphExec_t x;
    CK(cudaGraphInstantiate(&x, g, 0));
    for (int launch = 0; launch < 2; ++launch) {
      CK(cudaGraphLaunch(x, 0));
      CK(cudaDeviceSynchronize());
      WANT(take(1), 3);
    }
    CK(cudaGraphExecDestroy(x));
    CK(cudaGraphDestroy(g));
  }
  {
    // WHILE starting at 0 runs nothing: it is a while, not a do-while.
    cudaGraph_t g;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle h;
    CK(cudaGraphConditionalHandleCreate(&h, g, 0, cudaGraphCondAssignDefault));
    cudaGraphNode_t n, unused;
    cudaGraph_t b[1];
    CK(add_conditional(&n, g, nullptr, 0, h, kWhile, 1, b));
    CK(add_kernel(&unused, b[0], nullptr, 0, reinterpret_cast<void*>(bump), bump_args));
    cudaGraphExec_t x;
    CK(cudaGraphInstantiate(&x, g, 0));
    CK(cudaGraphLaunch(x, 0));
    WANT(take(), 0);
    CK(cudaGraphExecDestroy(x));
    CK(cudaGraphDestroy(g));
  }

  // ---- nested: an IF inside a WHILE's body ----------------------------------
  //
  // The IF's handle, made for the WHILE's body, is defaulted to 1 and cleared
  // by its own body: the default is applied once per launch, so the IF's body
  // runs on the loop's first iteration and not again. A handle made for the
  // outer graph decides a node in a body too.
  {
    cudaGraph_t g;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle hw, hi, hout;
    CK(cudaGraphConditionalHandleCreate(&hw, g, 1, cudaGraphCondAssignDefault));
    CK(cudaGraphConditionalHandleCreate(&hout, g, 1, cudaGraphCondAssignDefault));
    int limit = 3, ten = 10;
    unsigned zero = 0;
    void* loop_args[] = {&g_count, &limit, &hw};
    void* clear_args[] = {&hi, &zero};
    void* ten_args[] = {&g_count, &ten};
    cudaGraphNode_t n, k, in, k1, k2, kout;
    cudaGraph_t wb[1], ib[1], ob[1];
    CK(add_conditional(&n, g, nullptr, 0, hw, kWhile, 1, wb));
    CK(add_kernel(&k, wb[0], nullptr, 0, reinterpret_cast<void*>(count_down), loop_args));
    CK(cudaGraphConditionalHandleCreate(&hi, wb[0], 1, cudaGraphCondAssignDefault));
    CK(add_conditional(&in, wb[0], &k, 1, hi, kIf, 1, ib));
    CK(add_kernel(&k1, ib[0], nullptr, 0, reinterpret_cast<void*>(bump), bump_args));
    CK(add_kernel(&k2, ib[0], &k1, 1, reinterpret_cast<void*>(set_to), clear_args));
    CK(add_conditional(&in, wb[0], nullptr, 0, hout, kIf, 1, ob));
    CK(add_kernel(&kout, ob[0], nullptr, 0, reinterpret_cast<void*>(bump_by), ten_args));
    cudaGraphExec_t x;
    CK(cudaGraphInstantiate(&x, g, 0));
    CK(cudaGraphLaunch(x, 0));
    CK(cudaDeviceSynchronize());
    WANT(take(1), 3);        // iterations
    WANT(take(), 1 + 3 * 10);   // the inner IF once; the outer handle's IF every time
    CK(cudaGraphExecDestroy(x));
    CK(cudaGraphDestroy(g));
  }

#if CUDART_VERSION >= 12080
  // ---- IF/ELSE and SWITCH (CUDA 12.8) ---------------------------------------
  {
    cudaGraph_t g;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle h;
    CK(cudaGraphConditionalHandleCreate(&h, g, 0, 0));
    void* args[] = {&g_value, &h};
    int ten = 10;
    void* else_args[] = {&g_count, &ten};
    cudaGraphNode_t k, n, unused;
    cudaGraph_t b[2];
    CK(add_kernel(&k, g, nullptr, 0, reinterpret_cast<void*>(set_from), args));
    CK(add_conditional(&n, g, &k, 1, h, cudaGraphCondTypeIf, 2, b));
    CK(add_kernel(&unused, b[0], nullptr, 0, reinterpret_cast<void*>(bump), bump_args));
    CK(add_kernel(&unused, b[1], nullptr, 0, reinterpret_cast<void*>(bump_by), else_args));
    cudaGraphExec_t x;
    CK(cudaGraphInstantiate(&x, g, 0));
    for (int v : {0, 1, 2}) {
      set_value(v);
      CK(cudaGraphLaunch(x, 0));
      WANT(take(), v ? 1 : 10);
    }
    CK(cudaGraphExecDestroy(x));
    CK(cudaGraphDestroy(g));
  }
  {
    cudaGraph_t g;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle h;
    CK(cudaGraphConditionalHandleCreate(&h, g, 0, 0));
    void* args[] = {&g_value, &h};
    int by[3] = {1, 10, 100};
    cudaGraphNode_t k, n, unused;
    cudaGraph_t b[3];
    CK(add_kernel(&k, g, nullptr, 0, reinterpret_cast<void*>(set_from), args));
    CK(add_conditional(&n, g, &k, 1, h, cudaGraphCondTypeSwitch, 3, b));
    for (int i = 0; i < 3; ++i) {
      void* body_args[] = {&g_count, &by[i]};
      CK(add_kernel(&unused, b[i], nullptr, 0, reinterpret_cast<void*>(bump_by), body_args));
    }
    cudaGraphExec_t x;
    CK(cudaGraphInstantiate(&x, g, 0));
    const int want[4] = {1, 10, 100, 0};   // past the last body, none runs
    for (int v = 0; v < 4; ++v) {
      set_value(v);
      CK(cudaGraphLaunch(x, 0));
      WANT(take(), want[v]);
    }
    CK(cudaGraphExecDestroy(x));
    CK(cudaGraphDestroy(g));
  }
#endif

  // ---- what instantiation refuses --------------------------------------------
  {
    // A handle that decides no node: cudaGraphInstantiateConditionalHandleUnused.
    cudaGraph_t g;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle h, spare;
    CK(cudaGraphConditionalHandleCreate(&h, g, 1, cudaGraphCondAssignDefault));
    CK(cudaGraphConditionalHandleCreate(&spare, g, 1, cudaGraphCondAssignDefault));
    cudaGraphNode_t n, hn;
    cudaGraph_t b[1];
    CK(add_conditional(&n, g, nullptr, 0, h, kIf, 1, b));
    cudaGraphExec_t x;
    cudaGraphInstantiateParams ip = {};
    WANT(cudaGraphInstantiateWithParams(&x, g, &ip), cudaErrorInvalidValue);
    WANT(ip.result_out, 5);   // cudaGraphInstantiateConditionalHandleUnused
    CK(add_conditional(&hn, g, nullptr, 0, spare, kIf, 1, b + 0));
    // A host node in a body is accepted when added and refused at instantiation.
    cudaHostNodeParams hp = {[](void*) {}, nullptr};
    CK(cudaGraphAddHostNode(&hn, b[0], nullptr, 0, &hp));
    WANT(cudaGraphInstantiate(&x, g, 0), cudaErrorInvalidValue);
    CK(cudaGraphDestroyNode(hn));
    CK(cudaGraphInstantiate(&x, g, 0));
    CK(cudaGraphExecDestroy(x));
    // A conditional node destroyed leaves its handle deciding nothing.
    CK(cudaGraphDestroyNode(n));
    WANT(cudaGraphInstantiate(&x, g, 0), cudaErrorInvalidValue);
    CK(cudaGraphDestroy(g));
  }

  // ---- capturing into a graph: cudaStreamBeginCaptureToGraph -----------------
  {
    cudaGraph_t g;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphNode_t first;
    CK(add_kernel(&first, g, nullptr, 0, reinterpret_cast<void*>(bump), bump_args));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    WANT(cudaStreamBeginCaptureToGraph(s, nullptr, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal),
         cudaErrorInvalidValue);
    WANT(cudaStreamBeginCaptureToGraph(0, g, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal),
         cudaErrorStreamCaptureUnsupported);
    CK(cudaStreamBeginCaptureToGraph(s, g, &first, nullptr, 1, cudaStreamCaptureModeGlobal));
    cudaGraph_t during = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    size_t nd = 0;
    CK(capture_info(s, &during, &deps, &nd));
    WANT(during == g, true);
    WANT(nd == 1 && deps[0] == first, true);
    bump<<<1, 1, 0, s>>>(g_count);
    bump<<<1, 1, 0, s>>>(g_count);
    WANT(cudaGraphDestroy(g), cudaErrorIllegalState);
    cudaGraph_t out = nullptr;
    CK(cudaStreamEndCapture(s, &out));
    WANT(out == g, true);
    size_t count = 0, edges = 0;
    CK(cudaGraphGetNodes(g, nullptr, &count));
    WANT(count, 3);
#if CUDART_VERSION >= 13000
    CK(cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &edges));
#else
    CK(cudaGraphGetEdges(g, nullptr, nullptr, &edges));
#endif
    WANT(edges, 2);
    cudaGraphExec_t x;
    CK(cudaGraphInstantiate(&x, g, 0));
    CK(cudaGraphLaunch(x, 0));
    WANT(take(), 3);
    CK(cudaGraphExecDestroy(x));
    // A capture that fails takes the graph with it: its handle is no graph
    // afterwards, as on the card.
    CK(cudaStreamBeginCaptureToGraph(s, g, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
    bump<<<1, 1, 0, s>>>(g_count);
    WANT(cudaStreamSynchronize(s), cudaErrorStreamCaptureUnsupported);
    WANT(cudaStreamEndCapture(s, &out), cudaErrorStreamCaptureInvalidated);
    WANT(out == nullptr, true);
    cudaGetLastError();
    WANT(cudaGraphGetNodes(g, nullptr, &count), cudaErrorInvalidValue);
    cudaGetLastError();

    // As CUDA Samples' graphConditionalNodes uses it: a loop whose condition
    // is set in a captured graph, and whose body is captured into it.
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    cudaGraph_t cg = nullptr;
    CK(capture_info(s, &cg, &deps, &nd));
    cudaGraphConditionalHandle h;
    CK(cudaGraphConditionalHandleCreate(&h, cg, 0, 0));
    set_to<<<1, 1, 0, s>>>(h, 1);
    CK(capture_info(s, &cg, &deps, &nd));
    cudaGraphNode_t loop;
    cudaGraph_t b[1];
    CK(add_conditional(&loop, cg, const_cast<cudaGraphNode_t*>(deps), nd, h, kWhile, 1, b));
#if CUDART_VERSION >= 13000
    CK(cudaStreamUpdateCaptureDependencies(s, &loop, nullptr, 1, cudaStreamSetCaptureDependencies));
#else
    CK(cudaStreamUpdateCaptureDependencies(s, &loop, 1, cudaStreamSetCaptureDependencies));
#endif
    bump_by<<<1, 1, 0, s>>>(g_count, 100);
    CK(cudaStreamEndCapture(s, &cg));
    cudaStream_t bs;
    CK(cudaStreamCreate(&bs));
    CK(cudaStreamBeginCaptureToGraph(bs, b[0], nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
    int limit = 4;
    count_down<<<1, 1, 0, bs>>>(g_count, limit, h);
    CK(cudaStreamEndCapture(bs, nullptr));
    CK(cudaGraphInstantiate(&x, cg, 0));
    CK(cudaGraphLaunch(x, s));
    CK(cudaStreamSynchronize(s));
    WANT(take(1), 4);
    WANT(take(), 100);
    CK(cudaGraphExecDestroy(x));
    CK(cudaGraphDestroy(cg));
    CK(cudaStreamDestroy(bs));
    CK(cudaStreamDestroy(s));
  }

  // ---- outside a graph -------------------------------------------------------
  //
  // Last, because it kills the context: a kernel launched on its own has no
  // conditional to set, and on the card the call is an illegal address,
  // reported -- as any kernel's fault is -- by the next synchronize.
  {
    cudaGraph_t g;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle h;
    CK(cudaGraphConditionalHandleCreate(&h, g, 0, 0));
    set_to<<<1, 1>>>(h, 1);
    CK(cudaGetLastError());
    WANT(cudaDeviceSynchronize(), cudaErrorIllegalAddress);
  }

  std::printf(fails ? "FAIL\n" : "PASS\n");
  return fails != 0;
}

#else
int main() {
  std::printf("SKIP: conditional graph nodes need CUDA 12.3 or later (built with %d)\n", CUDART_VERSION);
  return 0;
}
#endif
