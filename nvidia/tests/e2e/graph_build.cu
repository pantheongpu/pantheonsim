// A graph built node by node, rather than captured from a stream. This is how a
// framework that knows its own dependency structure makes one: create an empty
// graph, add nodes, say which depend on which, instantiate once and launch many
// times -- changing parameters between launches without rebuilding.
//
// The graph here is a diamond, so the order is not a straight line:
//
//        memset(a, 0)
//        /          \
//   add(a, 3)     add(a, 4)     (both read and write a, so they are ordered
//        \          /            against the memset but not against each
//         copy a -> host         other; either order gives 7)
//
// and the test checks the result, the shape read back, what refuses, and that
// an instantiated graph takes new parameters.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
#define WANT(x, want) do { cudaError_t e_ = (x); if (e_ != (want)) { \
  printf("FAIL %s -> %s, expected %s\n", #x, cudaGetErrorString(e_), #want); return 1; } } while (0)

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
static cudaError_t count_deps(cudaGraphNode_t node, size_t* n) {
#if CUDART_VERSION >= 13000
  return cudaGraphNodeGetDependencies(node, nullptr, nullptr, n);
#else
  return cudaGraphNodeGetDependencies(node, nullptr, n);
#endif
}
static cudaError_t count_dependents(cudaGraphNode_t node, size_t* n) {
#if CUDART_VERSION >= 13000
  return cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, n);
#else
  return cudaGraphNodeGetDependentNodes(node, nullptr, n);
#endif
}

__global__ void add_to(int* p, int n, int v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&p[i], v);
}

int main() {
  const int n = 256;
  int* a = nullptr;
  int* out = nullptr;
  CK(cudaMalloc(reinterpret_cast<void**>(&a), n * sizeof(int)));
  CK(cudaMallocHost(reinterpret_cast<void**>(&out), n * sizeof(int)));

  cudaGraph_t graph = nullptr;
  CK(cudaGraphCreate(&graph, 0));

  // A fill, two kernels that depend on it, and a copy that depends on both.
  cudaMemsetParams zero{};
  zero.dst = a;
  zero.value = 0;
  zero.elementSize = 4;
  zero.width = n;
  zero.height = 1;
  cudaGraphNode_t fill = nullptr;
  CK(cudaGraphAddMemsetNode(&fill, graph, nullptr, 0, &zero));

  int three = 3, four = 4, count = n;
  void* args3[] = {&a, &count, &three};
  void* args4[] = {&a, &count, &four};
  cudaKernelNodeParams k{};
  k.func = reinterpret_cast<void*>(add_to);
  k.gridDim = dim3((n + 63) / 64);
  k.blockDim = dim3(64);
  k.sharedMemBytes = 0;
  k.kernelParams = args3;
  cudaGraphNode_t add3 = nullptr;
  CK(cudaGraphAddKernelNode(&add3, graph, &fill, 1, &k));
  k.kernelParams = args4;
  cudaGraphNode_t add4 = nullptr;
  CK(cudaGraphAddKernelNode(&add4, graph, &fill, 1, &k));

  const cudaGraphNode_t both[] = {add3, add4};
  cudaGraphNode_t copy = nullptr;
  CK(cudaGraphAddMemcpyNode1D(&copy, graph, both, 2, out, a, n * sizeof(int),
                              cudaMemcpyDeviceToHost));

  // The shape, read back.
  size_t nodes = 0, edges = 0, roots = 0;
  CK(cudaGraphGetNodes(graph, nullptr, &nodes));
  CK(count_edges(graph, &edges));
  CK(cudaGraphGetRootNodes(graph, nullptr, &roots));
  if (nodes != 4 || edges != 4 || roots != 1) {
    printf("FAIL shape: %zu nodes, %zu edges, %zu roots (want 4, 4, 1)\n", nodes, edges, roots);
    return 1;
  }
  cudaGraphNodeType type{};
  CK(cudaGraphNodeGetType(fill, &type));
  if (type != cudaGraphNodeTypeMemset) { printf("FAIL fill is type %d\n", (int)type); return 1; }
  CK(cudaGraphNodeGetType(copy, &type));
  if (type != cudaGraphNodeTypeMemcpy) { printf("FAIL copy is type %d\n", (int)type); return 1; }
  size_t deps = 0;
  CK(count_deps(copy, &deps));
  if (deps != 2) { printf("FAIL the copy has %zu dependencies, want 2\n", deps); return 1; }
  size_t dependents = 0;
  CK(count_dependents(fill, &dependents));
  if (dependents != 2) { printf("FAIL the fill has %zu dependents, want 2\n", dependents); return 1; }

  // A dependency that would close a cycle is refused: the copy already runs
  // after the fill, so the fill cannot also run after the copy.
  WANT(add_dep(graph, &copy, &fill, 1), cudaErrorInvalidValue);
  // And a node from another graph is not a dependency in this one.
  cudaGraph_t other = nullptr;
  cudaGraphNode_t stray = nullptr;
  CK(cudaGraphCreate(&other, 0));
  CK(cudaGraphAddEmptyNode(&stray, other, nullptr, 0));
  WANT(add_dep(graph, &stray, &fill, 1), cudaErrorInvalidValue);
  CK(cudaGraphDestroy(other));

  cudaGraphExec_t exec = nullptr;
  CK(cudaGraphInstantiate(&exec, graph, 0));
  CK(cudaGraphLaunch(exec, 0));
  CK(cudaStreamSynchronize(0));
  for (int i = 0; i < n; ++i) {
    if (out[i] != 7) {   // 0, then +3 and +4 in some order
      printf("FAIL out[%d] = %d, expected 7\n", i, out[i]);
      return 1;
    }
  }

  // Launching the same executable graph again re-runs the whole thing, fill
  // included, so the answer is the same rather than accumulating.
  CK(cudaGraphLaunch(exec, 0));
  CK(cudaStreamSynchronize(0));
  if (out[0] != 7) { printf("FAIL a second launch gave %d\n", out[0]); return 1; }

  // New parameters without rebuilding: the node that added 3 now adds 10.
  int ten = 10;
  void* args10[] = {&a, &count, &ten};
  k.kernelParams = args10;
  CK(cudaGraphExecKernelNodeSetParams(exec, add3, &k));
  CK(cudaGraphLaunch(exec, 0));
  CK(cudaStreamSynchronize(0));
  if (out[0] != 14) { printf("FAIL after new parameters, out[0] = %d, want 14\n", out[0]); return 1; }
  // The graph itself is unchanged: what was set on the executable graph stays
  // there, so re-instantiating gives the original numbers again.
  cudaGraphExec_t fresh = nullptr;
  CK(cudaGraphInstantiate(&fresh, graph, 0));
  CK(cudaGraphLaunch(fresh, 0));
  CK(cudaStreamSynchronize(0));
  if (out[0] != 7) { printf("FAIL a fresh instantiation gave %d, want 7\n", out[0]); return 1; }
  CK(cudaGraphExecDestroy(fresh));

  // cudaGraphExecUpdate: same shape, new parameters, no rebuild.
  int twenty = 20;
  void* args20[] = {&a, &count, &twenty};
  k.kernelParams = args20;
  CK(cudaGraphKernelNodeSetParams(add3, &k));
  cudaGraphExecUpdateResultInfo info{};
  CK(cudaGraphExecUpdate(exec, graph, &info));
  CK(cudaGraphLaunch(exec, 0));
  CK(cudaStreamSynchronize(0));
  if (out[0] != 24) { printf("FAIL after an update, out[0] = %d, want 24\n", out[0]); return 1; }
  // A graph with a different shape cannot update an instantiated one.
  cudaGraphNode_t extra = nullptr;
  CK(cudaGraphAddEmptyNode(&extra, graph, nullptr, 0));
  WANT(cudaGraphExecUpdate(exec, graph, &info), cudaErrorGraphExecUpdateFailure);
  if (info.result != cudaGraphExecUpdateErrorTopologyChanged) {
    printf("FAIL the update reported %d\n", (int)info.result);
    return 1;
  }
  CK(cudaGraphDestroyNode(extra));

  // A clone is its own graph: a node found in it is not the original.
  cudaGraph_t clone = nullptr;
  CK(cudaGraphClone(&clone, graph));
  cudaGraphNode_t cloned_fill = nullptr;
  CK(cudaGraphNodeFindInClone(&cloned_fill, fill, clone));
  if (cloned_fill == fill) { printf("FAIL the clone shares its nodes\n"); return 1; }
  size_t clone_nodes = 0;
  CK(cudaGraphGetNodes(clone, nullptr, &clone_nodes));
  if (clone_nodes != nodes) { printf("FAIL the clone has %zu nodes\n", clone_nodes); return 1; }
  CK(cudaGraphDestroy(clone));

  // A child graph runs inside its parent.
  cudaGraph_t parent = nullptr;
  CK(cudaGraphCreate(&parent, 0));
  cudaGraphNode_t child_node = nullptr;
  CK(cudaGraphAddChildGraphNode(&child_node, parent, nullptr, 0, graph));
  CK(cudaGraphNodeGetType(child_node, &type));
  if (type != cudaGraphNodeTypeGraph) { printf("FAIL child node type %d\n", (int)type); return 1; }
  cudaGraphExec_t parent_exec = nullptr;
  CK(cudaGraphInstantiate(&parent_exec, parent, 0));
  CK(cudaGraphLaunch(parent_exec, 0));
  CK(cudaStreamSynchronize(0));
  if (out[0] != 24) { printf("FAIL the child graph gave %d, want 24\n", out[0]); return 1; }
  CK(cudaGraphExecDestroy(parent_exec));
  CK(cudaGraphDestroy(parent));

  // What the API refuses.
  cudaGraphNode_t bad = nullptr;
  cudaKernelNodeParams no_func = k;
  no_func.func = nullptr;
  WANT(cudaGraphAddKernelNode(&bad, graph, nullptr, 0, &no_func), cudaErrorInvalidValue);
  cudaMemsetParams tall = zero;
  tall.height = 4;   // a 2D fill needs a pitch this engine does not carry
  WANT(cudaGraphAddMemsetNode(&bad, graph, nullptr, 0, &tall), cudaErrorNotSupported);

  // The drawing names every node and every dependency.
  const char* dot = "/tmp/vgpu-graph-build.dot";
  CK(cudaGraphDebugDotPrint(graph, dot, 0));
  FILE* df = fopen(dot, "r");
  if (!df) { printf("FAIL no DOT file\n"); return 1; }
  char text[4096] = {0};
  const size_t got = fread(text, 1, sizeof text - 1, df);
  fclose(df);
  remove(dot);
  int arrows = 0, kernels = 0;
  for (const char* p = text; (p = strstr(p, "->")) != nullptr; ++p) ++arrows;
  for (const char* p = text; (p = strstr(p, "KERNEL")) != nullptr; ++p) ++kernels;
  if (got == 0 || arrows != 4 || kernels != 2) {
    printf("FAIL the DOT drawing has %d edges and %d kernels (want 4 and 2)\n", arrows, kernels);
    return 1;
  }

  CK(cudaGraphExecDestroy(exec));
  CK(cudaGraphDestroy(graph));
  CK(cudaFreeHost(out));
  CK(cudaFree(a));
  printf("PASS\n");
  return 0;
}
