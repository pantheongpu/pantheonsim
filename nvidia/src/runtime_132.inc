// The runtime functions only the CUDA 13.2 cuda_runtime_api.h declares (tests/lint/check_header_exports.py against a
// 13.2 toolkit). Included by runtime_api.cpp after runtime_sweep.inc, whose helpers they use, and compiled only
// where the toolkit has them: a 12.0 or a 13.0 build neither declares nor exports these (CUDART_VERSION < 13020).
//
// NVIDIA's CUDA 13.2 runtime library does not start on the RTX 3060 under WSL that the other measurements were made on
// (its first call, cudaGetDeviceCount, crashes), so none of these could be checked against the runtime of that
// toolkit. Where the driver has the same function (the parameter count, the host function, the copies with
// attributes, the graph and node ids and parameters) the driver's [13.2] lines of
// nvidia/tests/data/exports_sweep_driver.rtx3060.expected were measured, and these answer the way the driver does,
// which is what the runtime documents (it hands them to the driver). Each is marked here as derived from
// that measurement and the header, not checked against a card.
#if CUDART_VERSION >= 13020

// ---- how many parameters a kernel takes ------------------------------------------------------------------------------
// Derived from the driver's cuFuncGetParamCount: a null result is cudaErrorInvalidValue; the function is looked up as
// cudaFuncGetParamInfo does.
static cudaError_t cudaFuncGetParamCount_traced(const void* func, size_t* paramCount) {
  return guard("cudaFuncGetParamCount", [&](State& s) -> cudaError_t {
    if (!paramCount) return cudaErrorInvalidValue;
    const vgpu::ptx::EntryFn* fn = nullptr;
    if (const cudaError_t e = kernel_function(s, func, "cudaFuncGetParamCount", &fn); e != cudaSuccess) return e;
    *paramCount = fn->params.size();
    return cudaSuccess;
  });
}

VGPU_EXPORT cudaError_t cudaFuncGetParamCount(const void* func, size_t* paramCount) {
  return traced_call("cudaFuncGetParamCount", cudaFuncGetParamCount_traced, func, paramCount);
}

// ---- a host function, with its synchronization mode -------------------------------------------------------------------------
// As the driver's cuLaunchHostFunc_v2 (measured there): mode 0 or 1, anything else cudaErrorInvalidValue first, then a
// null function is accepted and nothing runs. On a capturing stream it is a host node, as cudaLaunchHostFunc.
static cudaError_t cudaLaunchHostFunc_v2_traced(cudaStream_t stream, cudaHostFn_t fn, void* user, unsigned int syncMode) {
  if (const cudaError_t dead = dead_context()) return dead;
  if (syncMode > 1) return cudaErrorInvalidValue;
  if (!fn) return cudaSuccess;
  if (capture_host_fn(stream, fn, user)) return cudaSuccess;
  fn(user);
  return cudaSuccess;
}

VGPU_EXPORT cudaError_t cudaLaunchHostFunc_v2(cudaStream_t stream, cudaHostFn_t fn, void* userData, unsigned int syncMode) {
  return traced_call("cudaLaunchHostFunc_v2", cudaLaunchHostFunc_v2_traced, stream, fn, userData, syncMode);
}

// ---- one copy, with attributes -----------------------------------------------------------------------------------------------
// A batch of one (the driver's cuMemcpyWithAttributesAsync and cuMemcpy3DWithAttributesAsync, measured, are the
// batch calls' checks): a null attribute, a size or extent of 0, a bad access order or flags and the legacy default stream
// are cudaErrorInvalidValue.
static cudaError_t cudaMemcpyWithAttributesAsync_traced(void* dst, const void* src, size_t size, cudaMemcpyAttributes* attr, cudaStream_t stream) {
  void* d = dst;
  void* s = const_cast<void*>(src);
  size_t z = size, idx = 0;
  return memcpy_batch("cudaMemcpyWithAttributesAsync", &d, &s, &z, 1, attr, &idx, 1, nullptr, stream);
}

VGPU_EXPORT cudaError_t cudaMemcpyWithAttributesAsync(void* dst, const void* src, size_t size, cudaMemcpyAttributes* attr, cudaStream_t stream) {
  return traced_call("cudaMemcpyWithAttributesAsync", cudaMemcpyWithAttributesAsync_traced, dst, src, size, attr, stream);
}

static cudaError_t cudaMemcpy3DWithAttributesAsync_traced(cudaMemcpy3DBatchOp* op, unsigned long long flags, cudaStream_t stream) {
  return memcpy3d_batch(1, op, nullptr, flags, stream);
}

VGPU_EXPORT cudaError_t cudaMemcpy3DWithAttributesAsync(cudaMemcpy3DBatchOp* op, unsigned long long flags, cudaStream_t stream) {
  return traced_call("cudaMemcpy3DWithAttributesAsync", cudaMemcpy3DWithAttributesAsync_traced, op, flags, stream);
}

// ---- ids and parameters of graphs, executables and nodes ---------------------------------------------------------------------
// The numbering is the runtime's own hooks' (runtime_api.cpp, "graph, graph exec and node ids"), measured through
// the driver's cuGraphGetId and the rest. A null graph, executable or node and a null result are cudaErrorInvalidValue.
extern "C" int vgpu_graph_id_v1(void* graph, unsigned int* id);
extern "C" int vgpu_graph_exec_id_v1(void* exec, unsigned int* id);
extern "C" int vgpu_graph_node_ids_v1(void* node, unsigned int* local_id, unsigned long long* tools_id, void** graph);

static cudaError_t cudaGraphGetId_traced(cudaGraph_t graph, unsigned int* graphId) {
  return static_cast<cudaError_t>(vgpu_graph_id_v1(graph, graphId));
}
VGPU_EXPORT cudaError_t cudaGraphGetId(cudaGraph_t hGraph, unsigned int* graphID) {
  return traced_call("cudaGraphGetId", cudaGraphGetId_traced, hGraph, graphID);
}

static cudaError_t cudaGraphExecGetId_traced(cudaGraphExec_t exec, unsigned int* graphId) {
  return static_cast<cudaError_t>(vgpu_graph_exec_id_v1(exec, graphId));
}
VGPU_EXPORT cudaError_t cudaGraphExecGetId(cudaGraphExec_t hGraphExec, unsigned int* graphID) {
  return traced_call("cudaGraphExecGetId", cudaGraphExecGetId_traced, hGraphExec, graphID);
}

static cudaError_t cudaGraphNodeGetLocalId_traced(cudaGraphNode_t node, unsigned int* nodeId) {
  if (!nodeId) return cudaErrorInvalidValue;
  return static_cast<cudaError_t>(vgpu_graph_node_ids_v1(node, nodeId, nullptr, nullptr));
}
VGPU_EXPORT cudaError_t cudaGraphNodeGetLocalId(cudaGraphNode_t hNode, unsigned int* nodeId) {
  return traced_call("cudaGraphNodeGetLocalId", cudaGraphNodeGetLocalId_traced, hNode, nodeId);
}

static cudaError_t cudaGraphNodeGetToolsId_traced(cudaGraphNode_t node, unsigned long long* toolsNodeId) {
  if (!toolsNodeId) return cudaErrorInvalidValue;
  return static_cast<cudaError_t>(vgpu_graph_node_ids_v1(node, nullptr, toolsNodeId, nullptr));
}
VGPU_EXPORT cudaError_t cudaGraphNodeGetToolsId(cudaGraphNode_t hNode, unsigned long long* toolsNodeId) {
  return traced_call("cudaGraphNodeGetToolsId", cudaGraphNodeGetToolsId_traced, hNode, toolsNodeId);
}

static cudaError_t cudaGraphNodeGetContainingGraph_traced(cudaGraphNode_t node, cudaGraph_t* graph) {
  if (!graph) return cudaErrorInvalidValue;
  void* g = nullptr;
  const cudaError_t e = static_cast<cudaError_t>(vgpu_graph_node_ids_v1(node, nullptr, nullptr, &g));
  if (e == cudaSuccess) *graph = static_cast<cudaGraph_t>(g);
  return e;
}
VGPU_EXPORT cudaError_t cudaGraphNodeGetContainingGraph(cudaGraphNode_t hNode, cudaGraph_t* phGraph) {
  return traced_call("cudaGraphNodeGetContainingGraph", cudaGraphNodeGetContainingGraph_traced, hNode, phGraph);
}

// The parameters of a node of any kind, as the tagged union cudaGraphAddNode takes: the answer of the kind's own getter.
// A node of a kind this engine cannot make (allocation and free, external semaphores, conditional nodes) is
// cudaErrorNotSupported; an empty node is the type and nothing else (the driver's cuGraphNodeGetParams, measured).
static cudaError_t cudaGraphNodeGetParams_traced(cudaGraphNode_t node, cudaGraphNodeParams* out) {
  if (!node || !out) return cudaErrorInvalidValue;
  cudaGraphNodeType type;
  if (const cudaError_t e = cudaGraphNodeGetType(node, &type); e != cudaSuccess) return e;
  std::memset(out, 0, sizeof *out);
  out->type = type;
  switch (type) {
    case cudaGraphNodeTypeKernel: {
      cudaKernelNodeParams k{};
      if (const cudaError_t e = cudaGraphKernelNodeGetParams(node, &k); e != cudaSuccess) return e;
      out->kernel.func = k.func;
      out->kernel.gridDim = k.gridDim;
      out->kernel.blockDim = k.blockDim;
      out->kernel.sharedMemBytes = k.sharedMemBytes;
      out->kernel.kernelParams = k.kernelParams;
      out->kernel.extra = k.extra;
      out->kernel.functionType = cudaKernelFunctionTypeDeviceEntry;
      return cudaSuccess;
    }
    case cudaGraphNodeTypeMemcpy: return cudaGraphMemcpyNodeGetParams(node, &out->memcpy.copyParams);
    case cudaGraphNodeTypeMemset: {
      cudaMemsetParams m{};
      if (const cudaError_t e = cudaGraphMemsetNodeGetParams(node, &m); e != cudaSuccess) return e;
      out->memset.dst = m.dst;
      out->memset.pitch = m.pitch;
      out->memset.value = m.value;
      out->memset.elementSize = m.elementSize;
      out->memset.width = m.width;
      out->memset.height = m.height;
      return cudaSuccess;
    }
    case cudaGraphNodeTypeHost: {
      cudaHostNodeParams h{};
      if (const cudaError_t e = cudaGraphHostNodeGetParams(node, &h); e != cudaSuccess) return e;
      out->host.fn = h.fn;
      out->host.userData = h.userData;
      return cudaSuccess;
    }
    case cudaGraphNodeTypeGraph: return cudaGraphChildGraphNodeGetGraph(node, &out->graph.graph);
    case cudaGraphNodeTypeEmpty: return cudaSuccess;
    case cudaGraphNodeTypeWaitEvent: return cudaGraphEventWaitNodeGetEvent(node, &out->eventWait.event);
    case cudaGraphNodeTypeEventRecord: return cudaGraphEventRecordNodeGetEvent(node, &out->eventRecord.event);
    default: return refuse("cudaGraphNodeGetParams", "this node's kind is not simulated");
  }
}
VGPU_EXPORT cudaError_t cudaGraphNodeGetParams(cudaGraphNode_t node, struct cudaGraphNodeParams* nodeParams) {
  return traced_call("cudaGraphNodeGetParams", cudaGraphNodeGetParams_traced, node, nodeParams);
}

// ---- a conditional handle in an execution context --------------------------------------------------------------------------
// The same handle as cudaGraphConditionalHandleCreate, with the execution context it is for. A null context is the
// current one; any other is one this library never made (green contexts are not simulated), cudaErrorInvalidResourceHandle.
// Derived from the documentation, not checked against a card.
static cudaError_t cudaGraphConditionalHandleCreate_v2_traced(cudaGraphConditionalHandle* pHandle, cudaGraph_t graph, cudaExecutionContext_t ctx, unsigned int defaultLaunchValue, unsigned int flags) {
  if (ctx) return cudaErrorInvalidResourceHandle;
  return cudaGraphConditionalHandleCreate(pHandle, graph, defaultLaunchValue, flags);
}
VGPU_EXPORT cudaError_t cudaGraphConditionalHandleCreate_v2(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, cudaExecutionContext_t ctx, unsigned int defaultLaunchValue, unsigned int flags) {
  return traced_call("cudaGraphConditionalHandleCreate_v2", cudaGraphConditionalHandleCreate_v2_traced, pHandle_out, graph, ctx, defaultLaunchValue, flags);
}

// ---- green contexts and the resources of a device -------------------------------------------------------------------------------
// An execution context is a green context (or the primary one) and a device resource is a set of SMs to give one. This
// simulator has neither, as the driver shim has none (driver_sweep.inc): every one of these says so once on stderr and
// answers cudaErrorNotSupported. The card answers them; they are not among the card-checked cases of the sweep for that reason.
#define VGPU_EXEC_CTX(NAME, SIG, ARGS)                                                                       \
  static cudaError_t NAME##_traced SIG {                                                                     \
    return refuse(#NAME, "green contexts and device resources are not simulated");                           \
  }
VGPU_EXEC_CTX(cudaDevResourceGenerateDesc, (cudaDevResourceDesc_t*, cudaDevResource*, unsigned int), 0)
VGPU_EXEC_CTX(cudaDevSmResourceSplit, (cudaDevResource*, unsigned int, const cudaDevResource*, cudaDevResource*, unsigned int, cudaDevSmResourceGroupParams*), 0)
VGPU_EXEC_CTX(cudaDevSmResourceSplitByCount, (cudaDevResource*, unsigned int*, const cudaDevResource*, cudaDevResource*, unsigned int, unsigned int), 0)
VGPU_EXEC_CTX(cudaDeviceGetDevResource, (int, cudaDevResource*, enum cudaDevResourceType), 0)
VGPU_EXEC_CTX(cudaDeviceGetExecutionCtx, (cudaExecutionContext_t*, int), 0)
VGPU_EXEC_CTX(cudaExecutionCtxDestroy, (cudaExecutionContext_t), 0)
VGPU_EXEC_CTX(cudaExecutionCtxGetDevResource, (cudaExecutionContext_t, cudaDevResource*, enum cudaDevResourceType), 0)
VGPU_EXEC_CTX(cudaExecutionCtxGetDevice, (int*, cudaExecutionContext_t), 0)
VGPU_EXEC_CTX(cudaExecutionCtxGetId, (cudaExecutionContext_t, unsigned long long*), 0)
VGPU_EXEC_CTX(cudaExecutionCtxRecordEvent, (cudaExecutionContext_t, cudaEvent_t), 0)
VGPU_EXEC_CTX(cudaExecutionCtxStreamCreate, (cudaStream_t*, cudaExecutionContext_t, unsigned int, int), 0)
VGPU_EXEC_CTX(cudaExecutionCtxSynchronize, (cudaExecutionContext_t), 0)
VGPU_EXEC_CTX(cudaExecutionCtxWaitEvent, (cudaExecutionContext_t, cudaEvent_t), 0)
VGPU_EXEC_CTX(cudaGreenCtxCreate, (cudaExecutionContext_t*, cudaDevResourceDesc_t, int, unsigned int), 0)
VGPU_EXEC_CTX(cudaStreamGetDevResource, (cudaStream_t, cudaDevResource*, enum cudaDevResourceType), 0)
#undef VGPU_EXEC_CTX

VGPU_EXPORT cudaError_t cudaDevResourceGenerateDesc(cudaDevResourceDesc_t* phDesc, cudaDevResource* resources, unsigned int nbResources) {
  return traced_call("cudaDevResourceGenerateDesc", cudaDevResourceGenerateDesc_traced, phDesc, resources, nbResources);
}
VGPU_EXPORT cudaError_t cudaDevSmResourceSplit(cudaDevResource* result, unsigned int nbGroups, const cudaDevResource* input, cudaDevResource* remainder, unsigned int flags, cudaDevSmResourceGroupParams* groupParams) {
  return traced_call("cudaDevSmResourceSplit", cudaDevSmResourceSplit_traced, result, nbGroups, input, remainder, flags, groupParams);
}
VGPU_EXPORT cudaError_t cudaDevSmResourceSplitByCount(cudaDevResource* result, unsigned int* nbGroups, const cudaDevResource* input, cudaDevResource* remaining, unsigned int flags, unsigned int minCount) {
  return traced_call("cudaDevSmResourceSplitByCount", cudaDevSmResourceSplitByCount_traced, result, nbGroups, input, remaining, flags, minCount);
}
VGPU_EXPORT cudaError_t cudaDeviceGetDevResource(int device, cudaDevResource* resource, enum cudaDevResourceType type) {
  return traced_call("cudaDeviceGetDevResource", cudaDeviceGetDevResource_traced, device, resource, type);
}
VGPU_EXPORT cudaError_t cudaDeviceGetExecutionCtx(cudaExecutionContext_t* ctx, int device) {
  return traced_call("cudaDeviceGetExecutionCtx", cudaDeviceGetExecutionCtx_traced, ctx, device);
}
VGPU_EXPORT cudaError_t cudaExecutionCtxDestroy(cudaExecutionContext_t ctx) {
  return traced_call("cudaExecutionCtxDestroy", cudaExecutionCtxDestroy_traced, ctx);
}
VGPU_EXPORT cudaError_t cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx, cudaDevResource* resource, enum cudaDevResourceType type) {
  return traced_call("cudaExecutionCtxGetDevResource", cudaExecutionCtxGetDevResource_traced, ctx, resource, type);
}
VGPU_EXPORT cudaError_t cudaExecutionCtxGetDevice(int* device, cudaExecutionContext_t ctx) {
  return traced_call("cudaExecutionCtxGetDevice", cudaExecutionCtxGetDevice_traced, device, ctx);
}
VGPU_EXPORT cudaError_t cudaExecutionCtxGetId(cudaExecutionContext_t ctx, unsigned long long* ctxId) {
  return traced_call("cudaExecutionCtxGetId", cudaExecutionCtxGetId_traced, ctx, ctxId);
}
VGPU_EXPORT cudaError_t cudaExecutionCtxRecordEvent(cudaExecutionContext_t ctx, cudaEvent_t event) {
  return traced_call("cudaExecutionCtxRecordEvent", cudaExecutionCtxRecordEvent_traced, ctx, event);
}
VGPU_EXPORT cudaError_t cudaExecutionCtxStreamCreate(cudaStream_t* phStream, cudaExecutionContext_t ctx, unsigned int flags, int priority) {
  return traced_call("cudaExecutionCtxStreamCreate", cudaExecutionCtxStreamCreate_traced, phStream, ctx, flags, priority);
}
VGPU_EXPORT cudaError_t cudaExecutionCtxSynchronize(cudaExecutionContext_t ctx) {
  return traced_call("cudaExecutionCtxSynchronize", cudaExecutionCtxSynchronize_traced, ctx);
}
VGPU_EXPORT cudaError_t cudaExecutionCtxWaitEvent(cudaExecutionContext_t ctx, cudaEvent_t event) {
  return traced_call("cudaExecutionCtxWaitEvent", cudaExecutionCtxWaitEvent_traced, ctx, event);
}
VGPU_EXPORT cudaError_t cudaGreenCtxCreate(cudaExecutionContext_t* phCtx, cudaDevResourceDesc_t desc, int device, unsigned int flags) {
  return traced_call("cudaGreenCtxCreate", cudaGreenCtxCreate_traced, phCtx, desc, device, flags);
}
VGPU_EXPORT cudaError_t cudaStreamGetDevResource(cudaStream_t hStream, cudaDevResource* resource, enum cudaDevResourceType type) {
  return traced_call("cudaStreamGetDevResource", cudaStreamGetDevResource_traced, hStream, resource, type);
}

// The per-thread default stream's names of the 13.2 functions that take a stream.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattribute-alias"
#endif
#define VGPU_PT_ALIAS_132(name, target) \
  extern "C" __attribute__((visibility("default"))) void name() __attribute__((alias(#target)));
VGPU_PT_ALIAS_132(cudaLaunchHostFunc_v2_ptsz, cudaLaunchHostFunc_v2)
VGPU_PT_ALIAS_132(cudaMemcpyWithAttributesAsync_ptsz, cudaMemcpyWithAttributesAsync)
VGPU_PT_ALIAS_132(cudaMemcpy3DWithAttributesAsync_ptsz, cudaMemcpy3DWithAttributesAsync)
VGPU_PT_ALIAS_132(cudaStreamGetDevResource_ptsz, cudaStreamGetDevResource)
#undef VGPU_PT_ALIAS_132
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif  // CUDART_VERSION >= 13020
