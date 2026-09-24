// Two things a library does that the plain API paths do not exercise.
//
// A cooperative launch through the extended launch form: cudaLaunchKernelEx with
// cudaLaunchAttributeCooperative is the same request as
// cudaLaunchCooperativeKernel, and a kernel that calls grid.sync() needs it to be
// honoured -- without the grid-barrier workspace, cooperative_groups traps.
//
// Splicing work into someone else's stream capture: a library that finds the
// stream it was handed is capturing asks cudaStreamGetCaptureInfo for the graph
// and the nodes the next operation will depend on, adds its own nodes to that
// graph, and then says with cudaStreamUpdateCaptureDependencies that the capture
// continues from them. Here it forks two nodes off the captured kernel and the
// next captured kernel joins them -- a diamond out of a single stream.
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
#define WANT(x, want) do { cudaError_t e_ = (x); if (e_ != (want)) { \
  printf("FAIL %s -> %s, expected %s\n", #x, cudaGetErrorString(e_), #want); return 1; } } while (0)
#define CHECK(c) do { if (!(c)) { printf("FAIL %s (line %d)\n", #c, __LINE__); return 1; } } while (0)

// CUDA 13 gave the capture queries and the edge query an edge-data argument.
static cudaError_t capture_info(cudaStream_t s, cudaStreamCaptureStatus* st,
                                unsigned long long* id, cudaGraph_t* g,
                                const cudaGraphNode_t** deps, size_t* n) {
#if CUDART_VERSION >= 13000
  return cudaStreamGetCaptureInfo(s, st, id, g, deps, nullptr, n);
#else
  return cudaStreamGetCaptureInfo(s, st, id, g, deps, n);
#endif
}
static cudaError_t set_capture_deps(cudaStream_t s, cudaGraphNode_t* deps, size_t n,
                                    unsigned flags) {
#if CUDART_VERSION >= 13000
  return cudaStreamUpdateCaptureDependencies(s, deps, nullptr, n, flags);
#else
  return cudaStreamUpdateCaptureDependencies(s, deps, n, flags);
#endif
}
static cudaError_t count_edges(cudaGraph_t g, size_t* n) {
#if CUDART_VERSION >= 13000
  return cudaGraphGetEdges(g, nullptr, nullptr, nullptr, n);
#else
  return cudaGraphGetEdges(g, nullptr, nullptr, n);
#endif
}

static const int kN = 256;

// ---- the cooperative kernel ------------------------------------------------
//
// Every block writes its own slot, the grid synchronises, then every block reads
// the slot of the block after it. Without a real grid barrier the read can see
// the slot before it was written.
__global__ void neighbours(int* slots, int* seen) {
  cg::grid_group grid = cg::this_grid();
  if (threadIdx.x == 0) slots[blockIdx.x] = 100 + static_cast<int>(blockIdx.x);
  grid.sync();
  if (threadIdx.x == 0) seen[blockIdx.x] = slots[(blockIdx.x + 1) % gridDim.x];
}

// ---- the kernels the capture is made of -------------------------------------
__global__ void seed(int* a) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kN) a[i] = i;
}
__global__ void scale(const int* a, int* out, int by) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kN) out[i] = a[i] * by;
}
__global__ void join(const int* b, const int* c, int* d) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kN) d[i] = b[i] + c[i];
}

int main() {
  // ---- cooperative launch through cudaLaunchKernelEx --------------------------
  {
    const int blocks = 4;
    int *slots = nullptr, *seen = nullptr;
    CK(cudaMalloc(&slots, blocks * sizeof(int)));
    CK(cudaMalloc(&seen, blocks * sizeof(int)));
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(32);
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeCooperative;
    attr.val.cooperative = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, neighbours, slots, seen));
    CK(cudaDeviceSynchronize());
    int host[blocks];
    CK(cudaMemcpy(host, seen, sizeof host, cudaMemcpyDeviceToHost));
    for (int b = 0; b < blocks; ++b)
      if (host[b] != 100 + (b + 1) % blocks) {
        printf("FAIL block %d saw %d, expected %d\n", b, host[b], 100 + (b + 1) % blocks);
        return 1;
      }
    CK(cudaFree(slots));
    CK(cudaFree(seen));
  }

  // ---- splicing into a capture -----------------------------------------------
  int *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
  CK(cudaMalloc(&a, kN * sizeof(int)));
  CK(cudaMalloc(&b, kN * sizeof(int)));
  CK(cudaMalloc(&c, kN * sizeof(int)));
  CK(cudaMalloc(&d, kN * sizeof(int)));
  cudaStream_t stream = nullptr;
  CK(cudaStreamCreate(&stream));

  // Not capturing: that is all it says, and the dependency set cannot be set.
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusActive;
  CK(capture_info(stream, &status, nullptr, nullptr, nullptr, nullptr));
  CHECK(status == cudaStreamCaptureStatusNone);
  WANT(set_capture_deps(stream, nullptr, 0, cudaStreamSetCaptureDependencies),
       cudaErrorIllegalState);

  CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  seed<<<kN / 64, 64, 0, stream>>>(a);

  unsigned long long id = 0;
  cudaGraph_t capturing = nullptr;
  const cudaGraphNode_t* tail = nullptr;
  size_t tail_n = 0;
  CK(capture_info(stream, &status, &id, &capturing, &tail, &tail_n));
  CHECK(status == cudaStreamCaptureStatusActive);
  CHECK(id != 0);
  CHECK(capturing != nullptr);           // used to be null while "active"
  CHECK(tail_n == 1 && tail != nullptr); // the seed kernel
  const cudaGraphNode_t seed_node = tail[0];
  cudaGraphNodeType type{};
  CK(cudaGraphNodeGetType(seed_node, &type));
  CHECK(type == cudaGraphNodeTypeKernel);

  // The capture owns its graph: destroying it from outside is refused.
  WANT(cudaGraphDestroy(capturing), cudaErrorInvalidValue);

  // The library's part: two nodes added to the capture graph by hand, both
  // depending on what the capture had reached.
  int by2 = 2, by3 = 3;
  void* args_b[] = {&a, &b, &by2};
  void* args_c[] = {&a, &c, &by3};
  cudaKernelNodeParams kp{};
  kp.func = reinterpret_cast<void*>(scale);
  kp.gridDim = dim3(kN / 64);
  kp.blockDim = dim3(64);
  cudaGraphNode_t fork[2] = {nullptr, nullptr};
  kp.kernelParams = args_b;
  CK(cudaGraphAddKernelNode(&fork[0], capturing, &seed_node, 1, &kp));
  kp.kernelParams = args_c;
  CK(cudaGraphAddKernelNode(&fork[1], capturing, &seed_node, 1, &kp));

  // Refusals first: a node from another graph, and a flag that is neither.
  cudaGraph_t other = nullptr;
  CK(cudaGraphCreate(&other, 0));
  cudaGraphNode_t stranger = nullptr;
  CK(cudaGraphAddEmptyNode(&stranger, other, nullptr, 0));
  WANT(set_capture_deps(stream, &stranger, 1, cudaStreamSetCaptureDependencies),
       cudaErrorInvalidValue);
  WANT(set_capture_deps(stream, fork, 2, 7u), cudaErrorInvalidValue);

  // The capture continues from both: the next captured kernel joins them. Set
  // with one, then add the other, so both flags are used.
  CK(set_capture_deps(stream, &fork[0], 1, cudaStreamSetCaptureDependencies));
  CK(set_capture_deps(stream, &fork[1], 1, cudaStreamAddCaptureDependencies));
  CK(capture_info(stream, &status, nullptr, nullptr, &tail, &tail_n));
  CHECK(tail_n == 2);
  CHECK((tail[0] == fork[0] && tail[1] == fork[1]) || (tail[0] == fork[1] && tail[1] == fork[0]));

  join<<<kN / 64, 64, 0, stream>>>(b, c, d);

  cudaGraph_t graph = nullptr;
  CK(cudaStreamEndCapture(stream, &graph));
  // The graph handed out during the capture is the graph the capture returns.
  CHECK(graph == capturing);

  // A diamond: seed -> {x2, x3} -> join.
  size_t nodes = 0, edges = 0;
  CK(cudaGraphGetNodes(graph, nullptr, &nodes));
  CK(count_edges(graph, &edges));
  CHECK(nodes == 4);
  CHECK(edges == 4);

  cudaGraphExec_t exec = nullptr;
  CK(cudaGraphInstantiate(&exec, graph, 0));
  CK(cudaGraphLaunch(exec, stream));
  CK(cudaStreamSynchronize(stream));
  int host[kN];
  CK(cudaMemcpy(host, d, sizeof host, cudaMemcpyDeviceToHost));
  for (int i = 0; i < kN; ++i)
    if (host[i] != 5 * i) {
      printf("FAIL d[%d] = %d, expected %d\n", i, host[i], 5 * i);
      return 1;
    }

  // A second capture on the same stream is a different capture sequence.
  CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  unsigned long long id2 = 0;
  CK(capture_info(stream, &status, &id2, nullptr, nullptr, nullptr));
  CHECK(id2 != 0 && id2 != id);
  cudaGraph_t empty = nullptr;
  CK(cudaStreamEndCapture(stream, &empty));

  CK(cudaGraphExecDestroy(exec));
  CK(cudaGraphDestroy(graph));
  CK(cudaGraphDestroy(empty));
  CK(cudaGraphDestroy(other));
  CK(cudaStreamDestroy(stream));
  CK(cudaFree(a));
  CK(cudaFree(b));
  CK(cudaFree(c));
  CK(cudaFree(d));
  printf("PASS\n");
  return 0;
}
