// Memory a graph owns: an allocation node allocates when the graph reaches it
// and a free node frees it, so scratch space belongs to the graph rather than to
// the program for the graph's whole life.
//
// The address is the thing to get right. CUDA documents it as fixed across every
// instantiation and every launch, so a kernel built into the graph can hold it,
// and this test checks that it really does not move -- including across a trim,
// which gives the memory back but not the address.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
#define WANT(x, want) do { cudaError_t e_ = (x); if (e_ != (want)) { \
  printf("FAIL %s -> %s, expected %s\n", #x, cudaGetErrorString(e_), #want); return 1; } } while (0)
#define CHECK(c) do { if (!(c)) { printf("FAIL %s (line %d)\n", #c, __LINE__); return 1; } } while (0)

static cudaError_t add_dep(cudaGraph_t g, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t n) {
#if CUDART_VERSION >= 13000
  return cudaGraphAddDependencies(g, from, to, nullptr, n);
#else
  return cudaGraphAddDependencies(g, from, to, n);
#endif
}

static cudaError_t remove_dep(cudaGraph_t g, cudaGraphNode_t* from, cudaGraphNode_t* to,
                              size_t n) {
#if CUDART_VERSION >= 13000
  return cudaGraphRemoveDependencies(g, from, to, nullptr, n);
#else
  return cudaGraphRemoveDependencies(g, from, to, n);
#endif
}

static const int kInts = 1024;
static const size_t kBytes = kInts * sizeof(int);

// Writes i + base at every element, so the test can tell which launch wrote.
__global__ void stamp(int* p, int n, int base) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = i + base;
}

static uint64_t attr(int device, cudaGraphMemAttributeType which) {
  uint64_t v = 0;
  if (cudaDeviceGetGraphMemAttribute(device, which, &v) != cudaSuccess) return ~0ull;
  return v;
}

int main() {
  const int device = 0;
  CK(cudaSetDevice(device));
  // Nothing is charged to graph memory before there is a graph.
  CHECK(attr(device, cudaGraphMemAttrUsedMemCurrent) == 0);
  CHECK(attr(device, cudaGraphMemAttrReservedMemCurrent) == 0);

  // ---- a graph that allocates, uses and frees its own buffer ----------------
  cudaGraph_t g = nullptr;
  CK(cudaGraphCreate(&g, 0));

  cudaMemAllocNodeParams ap{};
  std::memset(&ap, 0, sizeof ap);
  ap.poolProps.allocType = cudaMemAllocationTypePinned;
  ap.poolProps.location.type = cudaMemLocationTypeDevice;
  ap.poolProps.location.id = device;
  ap.poolProps.handleTypes = cudaMemHandleTypeNone;
  ap.bytesize = kBytes;

  cudaGraphNode_t alloc = nullptr;
  CK(cudaGraphAddMemAllocNode(&alloc, g, nullptr, 0, &ap));
  // The address comes back from the call that built the node, before any launch.
  int* buf = static_cast<int*>(ap.dptr);
  CHECK(buf != nullptr);

  // What the node holds reads back as what was asked for.
  cudaMemAllocNodeParams back{};
  CK(cudaGraphMemAllocNodeGetParams(alloc, &back));
  CHECK(back.bytesize == kBytes && back.dptr == ap.dptr);

  cudaGraphNode_t kern = nullptr;
  int n = kInts, base = 100;
  void* args[] = {&buf, &n, &base};
  cudaKernelNodeParams kp{};
  kp.func = reinterpret_cast<void*>(stamp);
  kp.gridDim = dim3((kInts + 63) / 64);
  kp.blockDim = dim3(64);
  kp.kernelParams = args;
  CK(cudaGraphAddKernelNode(&kern, g, &alloc, 1, &kp));

  // A copy out of the graph's own buffer, so the host can see what it held while
  // the graph was running -- after the free there is nothing to read.
  int* out = nullptr;
  CK(cudaMalloc(reinterpret_cast<void**>(&out), kBytes));
  cudaGraphNode_t copy = nullptr;
  CK(cudaGraphAddMemcpyNode1D(&copy, g, &kern, 1, out, buf, kBytes, cudaMemcpyDeviceToDevice));

  cudaGraphNode_t freed = nullptr;
  CK(cudaGraphAddMemFreeNode(&freed, g, &copy, 1, buf));
  void* freed_ptr = nullptr;
  CK(cudaGraphMemFreeNodeGetParams(freed, &freed_ptr));
  CHECK(freed_ptr == buf);

  // An allocation is freed once, and in one graph. A second free node, here or
  // in another graph, is refused.
  cudaGraphNode_t twice = nullptr;
  WANT(cudaGraphAddMemFreeNode(&twice, g, &freed, 1, buf), cudaErrorInvalidValue);
  cudaGraph_t elsewhere = nullptr;
  CK(cudaGraphCreate(&elsewhere, 0));
  WANT(cudaGraphAddMemFreeNode(&twice, elsewhere, nullptr, 0, buf), cudaErrorInvalidValue);
  // And a pointer that is not a graph allocation at all cannot be freed by a node.
  WANT(cudaGraphAddMemFreeNode(&twice, g, nullptr, 0, out), cudaErrorInvalidValue);

  // ---- what such a graph refuses -------------------------------------------
  //
  // Every one of these would leave an allocation whose address nothing owns,
  // and CUDA documents each as refused.
  cudaGraph_t clone = nullptr;
  WANT(cudaGraphClone(&clone, g), cudaErrorInvalidValue);
  cudaGraph_t parent = nullptr;
  CK(cudaGraphCreate(&parent, 0));
  cudaGraphNode_t as_child = nullptr;
  WANT(cudaGraphAddChildGraphNode(&as_child, parent, nullptr, 0, g), cudaErrorInvalidValue);
  WANT(cudaGraphDestroyNode(kern), cudaErrorInvalidValue);
  WANT(remove_dep(g, &alloc, &kern, 1), cudaErrorInvalidValue);

  // ---- launching it --------------------------------------------------------
  cudaGraphExec_t exec = nullptr;
  CK(cudaGraphInstantiate(&exec, g, 0));
  // One instantiation of such a graph at a time: two would each believe they own
  // the allocation.
  cudaGraphExec_t second = nullptr;
  WANT(cudaGraphInstantiate(&second, g, 0), cudaErrorInvalidValue);

  CK(cudaGraphLaunch(exec, 0));
  CK(cudaDeviceSynchronize());

  int host[kInts];
  CK(cudaMemcpy(host, out, kBytes, cudaMemcpyDeviceToHost));
  CHECK(host[0] == 100 && host[kInts - 1] == kInts - 1 + 100);

  // The free node ran, so nothing is held now -- but the device still has the
  // memory, kept for the next launch. That is the difference between the two
  // pairs of attributes.
  CHECK(attr(device, cudaGraphMemAttrUsedMemCurrent) == 0);
  CHECK(attr(device, cudaGraphMemAttrUsedMemHigh) == kBytes);
  CHECK(attr(device, cudaGraphMemAttrReservedMemCurrent) >= kBytes);
  const uint64_t reserved_after_first = attr(device, cudaGraphMemAttrReservedMemCurrent);

  // The address does not move: the next launch allocates at the same place, so
  // the kernel's argument -- captured when the node was built -- still points at
  // the graph's buffer.
  base = 500;
  CK(cudaGraphExecKernelNodeSetParams(exec, kern, &kp));
  CK(cudaGraphLaunch(exec, 0));
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(host, out, kBytes, cudaMemcpyDeviceToHost));
  CHECK(host[0] == 500);
  CHECK(attr(device, cudaGraphMemAttrReservedMemCurrent) == reserved_after_first);

  // A high-water mark resets by writing zero to it, and nothing else may be
  // written.
  uint64_t zero = 0, nonzero = 4096;
  CK(cudaDeviceSetGraphMemAttribute(device, cudaGraphMemAttrUsedMemHigh, &zero));
  CHECK(attr(device, cudaGraphMemAttrUsedMemHigh) == 0);
  WANT(cudaDeviceSetGraphMemAttribute(device, cudaGraphMemAttrUsedMemHigh, &nonzero),
       cudaErrorInvalidValue);
  WANT(cudaDeviceSetGraphMemAttribute(device, cudaGraphMemAttrUsedMemCurrent, &zero),
       cudaErrorInvalidValue);

  // ---- trimming ------------------------------------------------------------
  //
  // The memory the graph is not holding goes back to the device. The address
  // stays, so a launch after the trim still works and still writes the same
  // place.
  CK(cudaDeviceGraphMemTrim(device));
  CHECK(attr(device, cudaGraphMemAttrReservedMemCurrent) == 0);
  CHECK(attr(device, cudaGraphMemAttrReservedMemHigh) >= kBytes);

  base = 900;
  CK(cudaGraphExecKernelNodeSetParams(exec, kern, &kp));
  CK(cudaGraphLaunch(exec, 0));
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(host, out, kBytes, cudaMemcpyDeviceToHost));
  CHECK(host[0] == 900);
  CHECK(attr(device, cudaGraphMemAttrReservedMemCurrent) >= kBytes);

  // An allocation its own graph frees cannot be freed from outside it.
  WANT(cudaFree(buf), cudaErrorInvalidValue);

  // ---- an allocation the graph does not free -------------------------------
  //
  // Without a free node the buffer outlives the launch, which is how a graph
  // hands a result to the work after it. Then the program frees it.
  cudaGraph_t keeper = nullptr;
  CK(cudaGraphCreate(&keeper, 0));
  cudaMemAllocNodeParams kap = ap;
  kap.dptr = nullptr;
  cudaGraphNode_t keep_alloc = nullptr;
  CK(cudaGraphAddMemAllocNode(&keep_alloc, keeper, nullptr, 0, &kap));
  int* kept = static_cast<int*>(kap.dptr);
  CHECK(kept != nullptr && kept != buf);
  int keep_base = 7;
  void* keep_args[] = {&kept, &n, &keep_base};
  cudaKernelNodeParams kkp = kp;
  kkp.kernelParams = keep_args;
  cudaGraphNode_t keep_kern = nullptr;
  CK(cudaGraphAddKernelNode(&keep_kern, keeper, &keep_alloc, 1, &kkp));
  CK(add_dep(keeper, &keep_alloc, &keep_kern, 1));

  cudaGraphExec_t keep_exec = nullptr;
  CK(cudaGraphInstantiate(&keep_exec, keeper, 0));
  CK(cudaGraphLaunch(keep_exec, 0));
  CK(cudaDeviceSynchronize());
  // Still allocated after the graph finished, and readable from outside it.
  CHECK(attr(device, cudaGraphMemAttrUsedMemCurrent) == kBytes);
  CK(cudaMemcpy(host, kept, kBytes, cudaMemcpyDeviceToHost));
  CHECK(host[0] == 7 && host[kInts - 1] == kInts - 1 + 7);
  // A trim does not take memory the graph is still holding.
  CK(cudaDeviceGraphMemTrim(device));
  CHECK(attr(device, cudaGraphMemAttrUsedMemCurrent) == kBytes);
  CK(cudaMemcpy(host, kept, kBytes, cudaMemcpyDeviceToHost));
  CHECK(host[0] == 7);
  // This one the program frees itself, because its graph never does.
  CK(cudaFree(kept));
  CHECK(attr(device, cudaGraphMemAttrUsedMemCurrent) == 0);

  // ---- freeing before relaunching, on request -------------------------------
  //
  // A graph with an allocation node and no free node can still be relaunched if
  // it was instantiated to free what the last launch left.
  CK(cudaGraphExecDestroy(keep_exec));
  cudaGraphExec_t auto_exec = nullptr;
  CK(cudaGraphInstantiate(&auto_exec, keeper, cudaGraphInstantiateFlagAutoFreeOnLaunch));
  CK(cudaGraphLaunch(auto_exec, 0));
  CK(cudaDeviceSynchronize());
  CHECK(attr(device, cudaGraphMemAttrUsedMemCurrent) == kBytes);
  CK(cudaGraphLaunch(auto_exec, 0));
  CK(cudaDeviceSynchronize());
  CHECK(attr(device, cudaGraphMemAttrUsedMemCurrent) == kBytes);   // freed, then allocated again
  // The kernel in the graph holds the address from when its node was built, and
  // it wrote through it again -- so the allocation really did come back at the
  // same place, after the program itself had freed it.
  CK(cudaMemcpy(host, kept, kBytes, cudaMemcpyDeviceToHost));
  CHECK(host[0] == 7 && host[kInts - 1] == kInts - 1 + 7);

  // Sharing a graph allocation between processes would need a handle type this
  // does not offer, and the API documents IPC as unsupported for these too.
  cudaGraph_t shared = nullptr;
  CK(cudaGraphCreate(&shared, 0));
  cudaMemAllocNodeParams sap = ap;
  sap.poolProps.handleTypes = cudaMemHandleTypePosixFileDescriptor;
  cudaGraphNode_t shared_node = nullptr;
  WANT(cudaGraphAddMemAllocNode(&shared_node, shared, nullptr, 0, &sap), cudaErrorNotSupported);

  CK(cudaGraphExecDestroy(exec));
  CK(cudaGraphExecDestroy(auto_exec));
  CK(cudaGraphDestroy(g));
  CK(cudaGraphDestroy(keeper));
  CK(cudaGraphDestroy(elsewhere));
  CK(cudaGraphDestroy(parent));
  CK(cudaGraphDestroy(shared));
  CK(cudaFree(out));
  printf("PASS\n");
  return 0;
}
