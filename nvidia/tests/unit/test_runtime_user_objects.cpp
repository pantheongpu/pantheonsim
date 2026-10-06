// CUDA 12's user objects and the runtime's library API, which PyTorch 2.14
// binds at load time (libtorch_cuda is linked with -z now, so a missing name
// fails `import torch`):
//   - a user object's destructor runs once, when the last reference goes;
//     a graph holds references, and so does each executable graph made from
//     it, so the object outlives a destroyed graph while an exec remains;
//   - cudaLibraryLoadData / GetKernel / Unload reach the driver's libraries.
#include <cuda_runtime_api.h>

#include <cstring>

#include "vtest.hpp"

namespace {
int g_destroyed = 0;
void* g_last = nullptr;
void CUDART_CB on_destroy(void* p) {
  ++g_destroyed;
  g_last = p;
}
}  // namespace

VTEST(a_user_object_is_destroyed_once_when_its_last_reference_goes) {
  g_destroyed = 0;
  int payload = 7;
  cudaUserObject_t obj = nullptr;
  VCHECK_EQ(cudaUserObjectCreate(&obj, &payload, on_destroy, 2, cudaUserObjectNoDestructorSync), cudaSuccess);
  VCHECK_EQ(cudaUserObjectRetain(obj, 3), cudaSuccess);   // 5
  VCHECK_EQ(cudaUserObjectRelease(obj, 4), cudaSuccess);  // 1
  VCHECK_EQ(g_destroyed, 0);
  VCHECK_EQ(cudaUserObjectRelease(obj, 2), cudaErrorInvalidValue);   // more than it holds
  VCHECK_EQ(cudaUserObjectRelease(obj, 1), cudaSuccess);
  VCHECK_EQ(g_destroyed, 1);
  VCHECK(g_last == &payload);
  VCHECK_EQ(cudaUserObjectRelease(obj, 1), cudaErrorInvalidValue);   // gone
}

VTEST(arguments_are_checked) {
  cudaUserObject_t obj = nullptr;
  VCHECK_EQ(cudaUserObjectCreate(nullptr, nullptr, on_destroy, 1, 0), cudaErrorInvalidValue);
  VCHECK_EQ(cudaUserObjectCreate(&obj, nullptr, nullptr, 1, 0), cudaErrorInvalidValue);
  VCHECK_EQ(cudaUserObjectCreate(&obj, nullptr, on_destroy, 0, 0), cudaErrorInvalidValue);   // no references
  VCHECK_EQ(cudaUserObjectCreate(&obj, nullptr, on_destroy, 1, 0x80), cudaErrorInvalidValue);   // unknown flag
}

VTEST(a_graph_and_the_execs_made_from_it_hold_references) {
  g_destroyed = 0;
  cudaUserObject_t obj = nullptr;
  VCHECK_EQ(cudaUserObjectCreate(&obj, nullptr, on_destroy, 1, 0), cudaSuccess);
  cudaGraph_t graph = nullptr;
  VCHECK_EQ(cudaGraphCreate(&graph, 0), cudaSuccess);
  VCHECK_EQ(cudaGraphRetainUserObject(graph, obj, 1, 0), cudaSuccess);   // graph: +1, caller: 1
  VCHECK_EQ(cudaUserObjectRelease(obj, 1), cudaSuccess);                  // caller lets go; the graph holds it
  VCHECK_EQ(g_destroyed, 0);
  cudaGraphExec_t exec = nullptr;
  VCHECK_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);        // the exec takes its own
  VCHECK_EQ(cudaGraphDestroy(graph), cudaSuccess);
  VCHECK_EQ(g_destroyed, 0);                                              // the exec still needs it
  VCHECK_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
  VCHECK_EQ(g_destroyed, 1);
}

VTEST(moving_references_to_a_graph_and_releasing_them_from_it) {
  g_destroyed = 0;
  cudaUserObject_t obj = nullptr;
  VCHECK_EQ(cudaUserObjectCreate(&obj, nullptr, on_destroy, 2, 0), cudaSuccess);
  cudaGraph_t graph = nullptr;
  VCHECK_EQ(cudaGraphCreate(&graph, 0), cudaSuccess);
  VCHECK_EQ(cudaGraphRetainUserObject(graph, obj, 3, cudaGraphUserObjectMove), cudaErrorInvalidValue);   // only 2 to move
  VCHECK_EQ(cudaGraphRetainUserObject(graph, obj, 2, cudaGraphUserObjectMove), cudaSuccess);
  VCHECK_EQ(cudaGraphReleaseUserObject(graph, obj, 3), cudaErrorInvalidValue);   // it holds 2
  VCHECK_EQ(cudaGraphReleaseUserObject(graph, obj, 1), cudaSuccess);
  VCHECK_EQ(g_destroyed, 0);
  VCHECK_EQ(cudaGraphDestroy(graph), cudaSuccess);   // the last one goes with the graph
  VCHECK_EQ(g_destroyed, 1);
}

VTEST(the_runtime_loads_a_library_and_finds_its_kernel) {
  const char* ptx =
      ".version 7.0\n.target sm_75\n.address_size 64\n"
      ".visible .entry noop() { ret; }\n";
  cudaLibrary_t lib = nullptr;
  VCHECK_EQ(cudaFree(nullptr), cudaSuccess);   // starts the runtime and its context
  VCHECK_EQ(cudaLibraryLoadData(&lib, ptx, nullptr, nullptr, 0, nullptr, nullptr, 0), cudaSuccess);
  VCHECK(lib != nullptr);
  cudaKernel_t kernel = nullptr;
  VCHECK_EQ(cudaLibraryGetKernel(&kernel, lib, "noop"), cudaSuccess);
  VCHECK(kernel != nullptr);
  VCHECK_EQ(cudaKernelSetAttributeForDevice(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024, 0),
            cudaSuccess);
  VCHECK_EQ(cudaLibraryUnload(lib), cudaSuccess);
  VCHECK(cudaLibraryUnload(lib) != cudaSuccess);   // already gone
}

VTEST_MAIN
