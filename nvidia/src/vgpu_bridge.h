// How the two CUDA shims reach each other.
//
// NVIDIA's runtime is a layer over its driver, so a CUgraph and a cudaGraph_t
// are one object, as are a CUstream and a cudaStream_t, a CUevent and a
// cudaEvent_t, a CUmemoryPool and a cudaMemPool_t. Here the two APIs are two
// libraries carrying one simulated machine (src/runtime/shared_runtime.cpp),
// and the larger half of the graph machinery -- graphs, nodes, executable
// graphs, stream capture, memory pools -- lives in libcudart, because the
// runtime API was written first. libcuda's entry points for the same objects
// are therefore thin: each decodes nothing and calls the function of the same
// name here, prefixed vgpu_, which libcudart exports. Both views then act on
// the same records, so a graph built with cuGraph* launches with
// cudaGraphLaunch and the other way round.
//
// libcuda finds libcudart without being linked to it (a program that uses only
// the driver API must not load a runtime it never asked for): through
// whichever copy is already loaded, or else the one beside it, loaded when the
// first graph function is called. Without one, those entry points return
// CUDA_ERROR_NOT_SUPPORTED.
//
// The other direction -- libcudart reaching libcuda for what only the driver
// holds: a CUfunction's kernel, a driver event -- goes through the
// vgpu_driver_*_v1 functions libcuda exports.
#ifndef VGPU_BRIDGE_H_
#define VGPU_BRIDGE_H_

#include <stddef.h>
#include <stdint.h>

#include "vgpu/exec/launch.hpp"
#include "vgpu/ptx/ast.hpp"

// What libcuda says about one of its functions, for libcudart to launch it:
// the device it belongs to, the code to run and the module's symbols.
struct VgpuDriverFuncInfo {
  void* function = nullptr;   // the CUfunction (what a CUkernel resolves to)
  void* kernel = nullptr;     // the CUkernel it was made from, or null
  int device = 0;
  const vgpu::ptx::EntryFn* fn = nullptr;
  const vgpu::exec::SymbolTable* syms = nullptr;
  bool nonportable_cluster = false;
  const char* name = nullptr;
};

// The driver graph and capture entry points that libcudart implements. Each
// has the signature of the CUDA function it backs with every handle a void*
// and every structure a const void* (decoded in runtime_driver_graphs.inc), and
// returns a CUresult. X(name, (parameters), (arguments)).
#define VGPU_GRAPH_BRIDGE_LIST(X) \
  X(cuGraphCreate, (void** g, unsigned flags), (g, flags)) \
  X(cuGraphDestroy, (void* g), (g)) \
  X(cuGraphClone, (void** out, void* g), (out, g)) \
  X(cuGraphNodeFindInClone, (void** out, void* node, void* clone), (out, node, clone)) \
  X(cuGraphGetNodes, (void* g, void** nodes, size_t* n), (g, nodes, n)) \
  X(cuGraphGetRootNodes, (void* g, void** nodes, size_t* n), (g, nodes, n)) \
  X(cuGraphGetEdges, (void* g, void** from, void** to, size_t* n), (g, from, to, n)) \
  X(cuGraphGetEdges_v2, (void* g, void** from, void** to, void* edge, size_t* n), (g, from, to, edge, n)) \
  X(cuGraphNodeGetType, (void* node, int* type), (node, type)) \
  X(cuGraphNodeGetDependencies, (void* node, void** deps, size_t* n), (node, deps, n)) \
  X(cuGraphNodeGetDependencies_v2, (void* node, void** deps, void* edge, size_t* n), (node, deps, edge, n)) \
  X(cuGraphNodeGetDependentNodes, (void* node, void** deps, size_t* n), (node, deps, n)) \
  X(cuGraphNodeGetDependentNodes_v2, (void* node, void** deps, void* edge, size_t* n), (node, deps, edge, n)) \
  X(cuGraphAddDependencies, (void* g, void* const* from, void* const* to, size_t n), (g, from, to, n)) \
  X(cuGraphAddDependencies_v2, (void* g, void* const* from, void* const* to, const void* edge, size_t n), \
    (g, from, to, edge, n)) \
  X(cuGraphRemoveDependencies, (void* g, void* const* from, void* const* to, size_t n), (g, from, to, n)) \
  X(cuGraphRemoveDependencies_v2, (void* g, void* const* from, void* const* to, const void* edge, size_t n), \
    (g, from, to, edge, n)) \
  X(cuGraphDestroyNode, (void* node), (node)) \
  X(cuGraphAddEmptyNode, (void** node, void* g, void* const* deps, size_t n), (node, g, deps, n)) \
  X(cuGraphAddKernelNode, (void** node, void* g, void* const* deps, size_t n, const void* p), (node, g, deps, n, p)) \
  X(cuGraphAddKernelNode_v2, (void** node, void* g, void* const* deps, size_t n, const void* p), (node, g, deps, n, p)) \
  X(cuGraphKernelNodeGetParams, (void* node, void* p), (node, p)) \
  X(cuGraphKernelNodeGetParams_v2, (void* node, void* p), (node, p)) \
  X(cuGraphKernelNodeSetParams, (void* node, const void* p), (node, p)) \
  X(cuGraphKernelNodeSetParams_v2, (void* node, const void* p), (node, p)) \
  X(cuGraphAddMemcpyNode, (void** node, void* g, void* const* deps, size_t n, const void* p, void* ctx), \
    (node, g, deps, n, p, ctx)) \
  X(cuGraphMemcpyNodeGetParams, (void* node, void* p), (node, p)) \
  X(cuGraphMemcpyNodeSetParams, (void* node, const void* p), (node, p)) \
  X(cuGraphAddMemsetNode, (void** node, void* g, void* const* deps, size_t n, const void* p, void* ctx), \
    (node, g, deps, n, p, ctx)) \
  X(cuGraphMemsetNodeGetParams, (void* node, void* p), (node, p)) \
  X(cuGraphMemsetNodeSetParams, (void* node, const void* p), (node, p)) \
  X(cuGraphAddHostNode, (void** node, void* g, void* const* deps, size_t n, const void* p), (node, g, deps, n, p)) \
  X(cuGraphHostNodeGetParams, (void* node, void* p), (node, p)) \
  X(cuGraphHostNodeSetParams, (void* node, const void* p), (node, p)) \
  X(cuGraphAddChildGraphNode, (void** node, void* g, void* const* deps, size_t n, void* child), \
    (node, g, deps, n, child)) \
  X(cuGraphChildGraphNodeGetGraph, (void* node, void** g), (node, g)) \
  X(cuGraphAddEventRecordNode, (void** node, void* g, void* const* deps, size_t n, void* ev), (node, g, deps, n, ev)) \
  X(cuGraphEventRecordNodeGetEvent, (void* node, void** ev), (node, ev)) \
  X(cuGraphEventRecordNodeSetEvent, (void* node, void* ev), (node, ev)) \
  X(cuGraphAddEventWaitNode, (void** node, void* g, void* const* deps, size_t n, void* ev), (node, g, deps, n, ev)) \
  X(cuGraphEventWaitNodeGetEvent, (void* node, void** ev), (node, ev)) \
  X(cuGraphEventWaitNodeSetEvent, (void* node, void* ev), (node, ev)) \
  X(cuGraphAddExternalSemaphoresSignalNode, (void** node, void* g, void* const* deps, size_t n, const void* p), \
    (node, g, deps, n, p)) \
  X(cuGraphExternalSemaphoresSignalNodeGetParams, (void* node, void* p), (node, p)) \
  X(cuGraphExternalSemaphoresSignalNodeSetParams, (void* node, const void* p), (node, p)) \
  X(cuGraphAddExternalSemaphoresWaitNode, (void** node, void* g, void* const* deps, size_t n, const void* p), \
    (node, g, deps, n, p)) \
  X(cuGraphExternalSemaphoresWaitNodeGetParams, (void* node, void* p), (node, p)) \
  X(cuGraphExternalSemaphoresWaitNodeSetParams, (void* node, const void* p), (node, p)) \
  X(cuGraphAddBatchMemOpNode, (void** node, void* g, void* const* deps, size_t n, const void* p), \
    (node, g, deps, n, p)) \
  X(cuGraphBatchMemOpNodeGetParams, (void* node, void* p), (node, p)) \
  X(cuGraphBatchMemOpNodeSetParams, (void* node, const void* p), (node, p)) \
  X(cuGraphAddMemAllocNode, (void** node, void* g, void* const* deps, size_t n, void* p), (node, g, deps, n, p)) \
  X(cuGraphMemAllocNodeGetParams, (void* node, void* p), (node, p)) \
  X(cuGraphAddMemFreeNode, (void** node, void* g, void* const* deps, size_t n, unsigned long long dptr), \
    (node, g, deps, n, dptr)) \
  X(cuGraphMemFreeNodeGetParams, (void* node, unsigned long long* dptr), (node, dptr)) \
  X(cuGraphAddNode, (void** node, void* g, void* const* deps, size_t n, void* p), (node, g, deps, n, p)) \
  X(cuGraphAddNode_v2, (void** node, void* g, void* const* deps, const void* edge, size_t n, void* p), \
    (node, g, deps, edge, n, p)) \
  X(cuGraphNodeGetParams, (void* node, void* p), (node, p)) \
  X(cuGraphNodeSetParams, (void* node, void* p), (node, p)) \
  X(cuGraphExecNodeSetParams, (void* exec, void* node, void* p), (exec, node, p)) \
  X(cuGraphInstantiateWithFlags, (void** exec, void* g, unsigned long long flags), (exec, g, flags)) \
  X(cuGraphInstantiate_v2, (void** exec, void* g, void** errNode, char* log, size_t logSize), \
    (exec, g, errNode, log, logSize)) \
  X(cuGraphInstantiateWithParams, (void** exec, void* g, void* params), (exec, g, params)) \
  X(cuGraphExecDestroy, (void* exec), (exec)) \
  X(cuGraphLaunch, (void* exec, void* stream), (exec, stream)) \
  X(cuGraphUpload, (void* exec, void* stream), (exec, stream)) \
  X(cuGraphExecUpdate, (void* exec, void* g, void** errNode, int* result), (exec, g, errNode, result)) \
  X(cuGraphExecUpdate_v2, (void* exec, void* g, void* info), (exec, g, info)) \
  X(cuGraphExecKernelNodeSetParams, (void* exec, void* node, const void* p), (exec, node, p)) \
  X(cuGraphExecKernelNodeSetParams_v2, (void* exec, void* node, const void* p), (exec, node, p)) \
  X(cuGraphExecMemcpyNodeSetParams, (void* exec, void* node, const void* p, void* ctx), (exec, node, p, ctx)) \
  X(cuGraphExecMemsetNodeSetParams, (void* exec, void* node, const void* p, void* ctx), (exec, node, p, ctx)) \
  X(cuGraphExecHostNodeSetParams, (void* exec, void* node, const void* p), (exec, node, p)) \
  X(cuGraphExecChildGraphNodeSetParams, (void* exec, void* node, void* child), (exec, node, child)) \
  X(cuGraphExecEventRecordNodeSetEvent, (void* exec, void* node, void* ev), (exec, node, ev)) \
  X(cuGraphExecEventWaitNodeSetEvent, (void* exec, void* node, void* ev), (exec, node, ev)) \
  X(cuGraphExecExternalSemaphoresSignalNodeSetParams, (void* exec, void* node, const void* p), (exec, node, p)) \
  X(cuGraphExecExternalSemaphoresWaitNodeSetParams, (void* exec, void* node, const void* p), (exec, node, p)) \
  X(cuGraphExecBatchMemOpNodeSetParams, (void* exec, void* node, const void* p), (exec, node, p)) \
  X(cuGraphNodeSetEnabled, (void* exec, void* node, unsigned enabled), (exec, node, enabled)) \
  X(cuGraphNodeGetEnabled, (void* exec, void* node, unsigned* enabled), (exec, node, enabled)) \
  X(cuGraphExecGetFlags, (void* exec, unsigned long long* flags), (exec, flags)) \
  X(cuGraphExecGetId, (void* exec, unsigned* id), (exec, id)) \
  X(cuGraphGetId, (void* g, unsigned* id), (g, id)) \
  X(cuGraphNodeGetLocalId, (void* node, unsigned* id), (node, id)) \
  X(cuGraphNodeGetToolsId, (void* node, unsigned long long* id), (node, id)) \
  X(cuGraphNodeGetContainingGraph, (void* node, void** g), (node, g)) \
  X(cuGraphDebugDotPrint, (void* g, const char* path, unsigned flags), (g, path, flags)) \
  X(cuGraphConditionalHandleCreate, (unsigned long long* h, void* g, void* ctx, unsigned def, unsigned flags), \
    (h, g, ctx, def, flags)) \
  X(cuGraphKernelNodeGetAttribute, (void* node, int attr, void* value), (node, attr, value)) \
  X(cuGraphKernelNodeSetAttribute, (void* node, int attr, const void* value), (node, attr, value)) \
  X(cuGraphKernelNodeCopyAttributes, (void* dst, void* src), (dst, src)) \
  X(cuUserObjectCreate, (void** obj, void* ptr, void (*destroy)(void*), unsigned count, unsigned flags), \
    (obj, ptr, destroy, count, flags)) \
  X(cuUserObjectRetain, (void* obj, unsigned count), (obj, count)) \
  X(cuUserObjectRelease, (void* obj, unsigned count), (obj, count)) \
  X(cuGraphRetainUserObject, (void* g, void* obj, unsigned count, unsigned flags), (g, obj, count, flags)) \
  X(cuGraphReleaseUserObject, (void* g, void* obj, unsigned count), (g, obj, count)) \
  X(cuStreamBeginCapture_v2, (void* stream, int mode), (stream, mode)) \
  X(cuStreamBeginCaptureToGraph, (void* stream, void* g, void* const* deps, const void* edge, size_t n, int mode), \
    (stream, g, deps, edge, n, mode)) \
  X(cuStreamEndCapture, (void* stream, void** g), (stream, g)) \
  X(cuStreamGetCaptureInfo_v3, (void* stream, int* status, unsigned long long* id, void** g, \
                              const void* const** deps, const void** edge, size_t* n), \
    (stream, status, id, g, deps, edge, n)) \
  X(cuStreamUpdateCaptureDependencies, (void* stream, void** deps, size_t n, unsigned flags), (stream, deps, n, flags)) \
  X(cuStreamUpdateCaptureDependencies_v2, (void* stream, void** deps, const void* edge, size_t n, unsigned flags), \
    (stream, deps, edge, n, flags)) \
  X(cuStreamIsCapturing, (void* stream, int* status), (stream, status)) \
  X(cuThreadExchangeStreamCaptureMode, (int* mode), (mode)) \
  X(cuDeviceGraphMemTrim, (int device), (device)) \
  X(cuDeviceGetGraphMemAttribute, (int device, int attr, void* value), (device, attr, value)) \
  X(cuDeviceSetGraphMemAttribute, (int device, int attr, void* value), (device, attr, value))

// libcudart's side: one extern "C" function per entry above.
#define VGPU_DECLARE_BRIDGE(name, params, args) extern "C" int vgpu_##name params;

// Exported by libcudart so libcuda can tell it is the simulator's runtime.
extern "C" int vgpu_runtime_marker_v1();

// Exported by libcuda for libcudart.
extern "C" {
// The function behind a CUfunction or a CUkernel; 0 on success.
int vgpu_driver_function_info_v1(void* function_or_kernel, VgpuDriverFuncInfo* out);
// A CUfunction for an entry of a module the runtime loaded (device, module id
// as Device::load_module returned it), minted on first use.
int vgpu_driver_function_for_v1(int device, uint64_t module_id, const char* name, void** out);
// Driver events: whether a handle is one, and the record a graph's event-record
// node does when it runs.
int vgpu_driver_event_exists_v1(void* event);
int vgpu_driver_event_record_now_v1(void* event);
// A driver stream's creation flags, priority and device; 0 if it is not one.
int vgpu_driver_stream_info_v1(void* stream, unsigned* flags, int* priority, int* device);
// The device the calling thread's current context is on, or -1.
int vgpu_driver_current_device_v1();
// A context handle's device (0 if it is not a live context), and a device's
// primary context handle (creating nothing: null when none is retained).
int vgpu_driver_ctx_device_v1(void* ctx, int* device);
void* vgpu_driver_primary_ctx_v1(int device);
// A graph memcpy node that touches a CUDA array runs through cuMemcpy3D.
int vgpu_driver_copy3d_v1(const void* copy3d);
}

#endif  // VGPU_BRIDGE_H_
