/* The driver API's graphs and stream capture: the types and layouts of NVIDIA's cuda.h (CUDA 13.0)
 * that libvgpucuda's graph API takes, declared here so the library builds with no toolkit headers.
 *
 * The struct layouts and enumerator values are the documented ones, so a program compiled against
 * NVIDIA's cuda.h passes the same bytes. Include after vgpu_cuda.h. */
#ifndef VGPU_CUDA_GRAPH_H_
#define VGPU_CUDA_GRAPH_H_

#include "vgpu_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CUgraph_st* CUgraph;
typedef struct CUgraphNode_st* CUgraphNode;
typedef struct CUgraphExec_st* CUgraphExec;
typedef struct CUkern_st* CUkernel;
typedef struct CUuserObject_st* CUuserObject;
typedef unsigned long long cuuint64_t;
typedef unsigned int cuuint32_t;
typedef unsigned long long CUgraphConditionalHandle;

typedef enum CUstreamCaptureMode_enum {
  CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
  CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
  CU_STREAM_CAPTURE_MODE_RELAXED = 2
} CUstreamCaptureMode;

typedef enum CUstreamCaptureStatus_enum {
  CU_STREAM_CAPTURE_STATUS_NONE = 0,
  CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
  CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2
} CUstreamCaptureStatus;

typedef enum CUgraphNodeType_enum {
  CU_GRAPH_NODE_TYPE_KERNEL = 0,
  CU_GRAPH_NODE_TYPE_MEMCPY = 1,
  CU_GRAPH_NODE_TYPE_MEMSET = 2,
  CU_GRAPH_NODE_TYPE_HOST = 3,
  CU_GRAPH_NODE_TYPE_GRAPH = 4,
  CU_GRAPH_NODE_TYPE_EMPTY = 5,
  CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
  CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
  CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
  CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
  CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
  CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
  CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12,
  CU_GRAPH_NODE_TYPE_CONDITIONAL = 13
} CUgraphNodeType;

typedef struct CUgraphEdgeData_st {
  unsigned char from_port;
  unsigned char to_port;
  unsigned char type;
  unsigned char reserved[5];
} CUgraphEdgeData;

/* Kernel nodes: v1 (CUDA 10 to 11), v2 and v3 add the CUkernel and the context. */
typedef struct CUDA_KERNEL_NODE_PARAMS_v1_st {
  CUfunction func;
  unsigned int gridDimX, gridDimY, gridDimZ;
  unsigned int blockDimX, blockDimY, blockDimZ;
  unsigned int sharedMemBytes;
  void** kernelParams;
  void** extra;
} CUDA_KERNEL_NODE_PARAMS_v1;
typedef struct CUDA_KERNEL_NODE_PARAMS_v3_st {
  CUfunction func;
  unsigned int gridDimX, gridDimY, gridDimZ;
  unsigned int blockDimX, blockDimY, blockDimZ;
  unsigned int sharedMemBytes;
  void** kernelParams;
  void** extra;
  CUkernel kern;
  CUcontext ctx;
} CUDA_KERNEL_NODE_PARAMS_v3;
typedef CUDA_KERNEL_NODE_PARAMS_v3 CUDA_KERNEL_NODE_PARAMS_v2;

typedef struct CUDA_MEMSET_NODE_PARAMS_v1_st {
  CUdeviceptr dst;
  size_t pitch;
  unsigned int value;
  unsigned int elementSize;
  size_t width;
  size_t height;
} CUDA_MEMSET_NODE_PARAMS_v1;
typedef struct CUDA_MEMSET_NODE_PARAMS_v2_st {
  CUdeviceptr dst;
  size_t pitch;
  unsigned int value;
  unsigned int elementSize;
  size_t width;
  size_t height;
  CUcontext ctx;
} CUDA_MEMSET_NODE_PARAMS_v2;

typedef struct CUDA_HOST_NODE_PARAMS_v1_st {
  CUhostFn fn;
  void* userData;
} CUDA_HOST_NODE_PARAMS_v1;
typedef CUDA_HOST_NODE_PARAMS_v1 CUDA_HOST_NODE_PARAMS_v2;

typedef struct CUmemPoolProps_st {
  CUmemAllocationType allocType;
  int handleTypes;
  CUmemLocation location;
  void* win32SecurityAttributes;
  size_t maxSize;
  unsigned short usage;
  unsigned char reserved[54];
} CUmemPoolProps;
typedef struct CUDA_MEM_ALLOC_NODE_PARAMS_v1_st {
  CUmemPoolProps poolProps;
  const CUmemAccessDesc* accessDescs;
  size_t accessDescCount;
  size_t bytesize;
  CUdeviceptr dptr;
} CUDA_MEM_ALLOC_NODE_PARAMS_v1;
typedef CUDA_MEM_ALLOC_NODE_PARAMS_v1 CUDA_MEM_ALLOC_NODE_PARAMS_v2;

typedef struct CUDA_GRAPH_INSTANTIATE_PARAMS_st {
  cuuint64_t flags;
  CUstream hUploadStream;
  CUgraphNode hErrNode_out;
  int result_out; /* CUgraphInstantiateResult */
} CUDA_GRAPH_INSTANTIATE_PARAMS;

typedef struct CUgraphExecUpdateResultInfo_st {
  int result; /* CUgraphExecUpdateResult */
  CUgraphNode errorNode;
  CUgraphNode errorFromNode;
} CUgraphExecUpdateResultInfo;

typedef enum CUstreamBatchMemOpType_enum {
  CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
  CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
  CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
  CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,
  CU_STREAM_MEM_OP_BARRIER = 6,
  CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3
} CUstreamBatchMemOpType;

typedef union CUstreamBatchMemOpParams_union {
  CUstreamBatchMemOpType operation;
  struct CUstreamMemOpWaitValueParams_st {
    CUstreamBatchMemOpType operation;
    CUdeviceptr address;
    union {
      cuuint32_t value;
      cuuint64_t value64;
    };
    unsigned int flags;
    CUdeviceptr alias;
  } waitValue;
  struct CUstreamMemOpWriteValueParams_st {
    CUstreamBatchMemOpType operation;
    CUdeviceptr address;
    union {
      cuuint32_t value;
      cuuint64_t value64;
    };
    unsigned int flags;
    CUdeviceptr alias;
  } writeValue;
  struct CUstreamMemOpFlushRemoteWritesParams_st {
    CUstreamBatchMemOpType operation;
    unsigned int flags;
  } flushRemoteWrites;
  struct CUstreamMemOpMemoryBarrierParams_st {
    CUstreamBatchMemOpType operation;
    unsigned int flags;
  } memoryBarrier;
  cuuint64_t pad[6];
} CUstreamBatchMemOpParams;

typedef struct CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st {
  CUcontext ctx;
  unsigned int count;
  CUstreamBatchMemOpParams* paramArray;
  unsigned int flags;
} CUDA_BATCH_MEM_OP_NODE_PARAMS_v1;
typedef CUDA_BATCH_MEM_OP_NODE_PARAMS_v1 CUDA_BATCH_MEM_OP_NODE_PARAMS_v2;

/* cuGraphAddNode and cuGraphNodeSetParams take one tagged union. */
typedef struct CUDA_MEMCPY_NODE_PARAMS_st {
  int flags;
  int reserved;
  CUcontext copyCtx;
  CUDA_MEMCPY3D copyParams;
} CUDA_MEMCPY_NODE_PARAMS;
typedef struct CUDA_CHILD_GRAPH_NODE_PARAMS_st {
  CUgraph graph;
  int ownership; /* CUgraphChildGraphNodeOwnership */
} CUDA_CHILD_GRAPH_NODE_PARAMS;
typedef struct CUDA_EVENT_WAIT_NODE_PARAMS_st {
  CUevent event;
} CUDA_EVENT_WAIT_NODE_PARAMS;
typedef struct CUDA_EVENT_RECORD_NODE_PARAMS_st {
  CUevent event;
} CUDA_EVENT_RECORD_NODE_PARAMS;
typedef struct CUDA_MEM_FREE_NODE_PARAMS_st {
  CUdeviceptr dptr;
} CUDA_MEM_FREE_NODE_PARAMS;

typedef struct CUgraphNodeParams_st {
  CUgraphNodeType type;
  int reserved0[3];
  union {
    long long reserved1[29];
    CUDA_KERNEL_NODE_PARAMS_v3 kernel;
    CUDA_MEMCPY_NODE_PARAMS memcpy;
    CUDA_MEMSET_NODE_PARAMS_v2 memset;
    CUDA_HOST_NODE_PARAMS_v2 host;
    CUDA_CHILD_GRAPH_NODE_PARAMS graph;
    CUDA_EVENT_WAIT_NODE_PARAMS eventWait;
    CUDA_EVENT_RECORD_NODE_PARAMS eventRecord;
    CUDA_MEM_ALLOC_NODE_PARAMS_v2 alloc;
    CUDA_MEM_FREE_NODE_PARAMS free;
    CUDA_BATCH_MEM_OP_NODE_PARAMS_v2 memOp;
  };
  long long reserved2;
} CUgraphNodeParams;

/* Graphs and stream capture (nvidia/src/driver_graph.cpp). */
CUresult cuStreamBatchMemOp(CUstream st, unsigned int count, CUstreamBatchMemOpParams* params, unsigned int flags);
CUresult cuStreamBatchMemOp_v2(CUstream st, unsigned int count, CUstreamBatchMemOpParams* params, unsigned int flags);
CUresult cuStreamBeginCapture_v2(CUstream st, CUstreamCaptureMode mode);
CUresult cuStreamBeginCapture(CUstream st);
CUresult cuStreamBeginCaptureToGraph(CUstream st, CUgraph graph, const CUgraphNode* deps, const CUgraphEdgeData* edges, size_t num_deps, CUstreamCaptureMode mode);
CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode);
CUresult cuStreamEndCapture(CUstream st, CUgraph* graph);
CUresult cuStreamGetCaptureInfo_v3(CUstream st, CUstreamCaptureStatus* status, cuuint64_t* id, CUgraph* graph, const CUgraphNode** deps, const CUgraphEdgeData** edges, size_t* n);
CUresult cuStreamGetCaptureInfo_v2(CUstream st, CUstreamCaptureStatus* status, cuuint64_t* id, CUgraph* graph, const CUgraphNode** deps, size_t* n);
CUresult cuStreamGetCaptureInfo(CUstream st, CUstreamCaptureStatus* status, cuuint64_t* id);
CUresult cuStreamUpdateCaptureDependencies_v2(CUstream st, CUgraphNode* deps, const CUgraphEdgeData* edges, size_t n, unsigned int flags);
CUresult cuStreamUpdateCaptureDependencies(CUstream st, CUgraphNode* deps, size_t n, unsigned int flags);
CUresult cuGraphCreate(CUgraph* graph, unsigned int flags);
CUresult cuGraphDestroy(CUgraph graph);
CUresult cuGraphClone(CUgraph* clone, CUgraph original);
CUresult cuGraphNodeFindInClone(CUgraphNode* node, CUgraphNode original, CUgraph clone);
CUresult cuGraphGetNodes(CUgraph graph, CUgraphNode* nodes, size_t* n);
CUresult cuGraphGetRootNodes(CUgraph graph, CUgraphNode* nodes, size_t* n);
CUresult cuGraphGetEdges_v2(CUgraph graph, CUgraphNode* from, CUgraphNode* to, CUgraphEdgeData* edges, size_t* n);
CUresult cuGraphGetEdges(CUgraph graph, CUgraphNode* from, CUgraphNode* to, size_t* n);
CUresult cuGraphNodeGetDependencies_v2(CUgraphNode node, CUgraphNode* deps, CUgraphEdgeData* edges, size_t* n);
CUresult cuGraphNodeGetDependencies(CUgraphNode node, CUgraphNode* deps, size_t* n);
CUresult cuGraphNodeGetDependentNodes_v2(CUgraphNode node, CUgraphNode* deps, CUgraphEdgeData* edges, size_t* n);
CUresult cuGraphNodeGetDependentNodes(CUgraphNode node, CUgraphNode* deps, size_t* n);
CUresult cuGraphAddDependencies_v2(CUgraph graph, const CUgraphNode* from, const CUgraphNode* to, const CUgraphEdgeData* edges, size_t n);
CUresult cuGraphAddDependencies(CUgraph graph, const CUgraphNode* from, const CUgraphNode* to, size_t n);
CUresult cuGraphRemoveDependencies_v2(CUgraph graph, const CUgraphNode* from, const CUgraphNode* to, const CUgraphEdgeData* edges, size_t n);
CUresult cuGraphRemoveDependencies(CUgraph graph, const CUgraphNode* from, const CUgraphNode* to, size_t n);
CUresult cuGraphDestroyNode(CUgraphNode node);
CUresult cuGraphNodeGetType(CUgraphNode node, CUgraphNodeType* type);
CUresult cuGraphNodeGetEnabled(CUgraphExec exec, CUgraphNode node, unsigned int* enabled);
CUresult cuGraphNodeSetEnabled(CUgraphExec exec, CUgraphNode node, unsigned int enabled);
CUresult cuGraphDebugDotPrint(CUgraph graph, const char* path, unsigned int flags);
CUresult cuGraphInstantiateWithFlags(CUgraphExec* exec, CUgraph graph, unsigned long long flags);
CUresult cuGraphInstantiate_v2(CUgraphExec* exec, CUgraph graph, CUgraphNode* error_node, char* log, size_t log_size);
CUresult cuGraphInstantiate(CUgraphExec* exec, CUgraph graph, CUgraphNode* error_node, char* log, size_t log_size);
CUresult cuGraphInstantiateWithParams(CUgraphExec* exec, CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS* params);
CUresult cuGraphLaunch(CUgraphExec exec, CUstream st);
CUresult cuGraphUpload(CUgraphExec exec, CUstream st);
CUresult cuGraphExecDestroy(CUgraphExec exec);
CUresult cuGraphExecGetFlags(CUgraphExec exec, cuuint64_t* flags);
CUresult cuGraphExecUpdate_v2(CUgraphExec exec, CUgraph graph, CUgraphExecUpdateResultInfo* info);
CUresult cuGraphExecUpdate(CUgraphExec exec, CUgraph graph, CUgraphNode* error_node, int* result);
CUresult cuDeviceGetGraphMemAttribute(CUdevice dev, int attr, void* value);
CUresult cuDeviceSetGraphMemAttribute(CUdevice dev, int attr, void* value);
CUresult cuDeviceGraphMemTrim(CUdevice dev);
CUresult cuUserObjectCreate(CUuserObject* object, void* ptr, CUhostFn destroy, unsigned int initial, unsigned int flags);
CUresult cuUserObjectRetain(CUuserObject object, unsigned int count);
CUresult cuUserObjectRelease(CUuserObject object, unsigned int count);
CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags);
CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count);
CUresult cuGraphAddEmptyNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n);
CUresult cuGraphAddHostNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, const CUDA_HOST_NODE_PARAMS_v1* params);
CUresult cuGraphHostNodeGetParams(CUgraphNode node, CUDA_HOST_NODE_PARAMS_v1* params);
CUresult cuGraphHostNodeSetParams(CUgraphNode node, const CUDA_HOST_NODE_PARAMS_v1* params);
CUresult cuGraphExecHostNodeSetParams(CUgraphExec exec, CUgraphNode node, const CUDA_HOST_NODE_PARAMS_v1* params);
CUresult cuGraphAddChildGraphNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUgraph child);
CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode node, CUgraph* graph);
CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec exec, CUgraphNode node, CUgraph child);
CUresult cuGraphAddEventRecordNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUevent event);
CUresult cuGraphAddEventWaitNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUevent event);
CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode node, CUevent* event);
CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode node, CUevent* event);
CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode node, CUevent event);
CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode node, CUevent event);
CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec exec, CUgraphNode node, CUevent event);
CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec exec, CUgraphNode node, CUevent event);
CUresult cuGraphAddKernelNode_v2(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, const CUDA_KERNEL_NODE_PARAMS_v3* params);
CUresult cuGraphAddKernelNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, const CUDA_KERNEL_NODE_PARAMS_v1* params);
CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode node, CUDA_KERNEL_NODE_PARAMS_v3* params);
CUresult cuGraphKernelNodeGetParams(CUgraphNode node, CUDA_KERNEL_NODE_PARAMS_v1* params);
CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS_v3* params);
CUresult cuGraphKernelNodeSetParams(CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS_v1* params);
CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec exec, CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS_v3* params);
CUresult cuGraphExecKernelNodeSetParams(CUgraphExec exec, CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS_v1* params);
CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src);
CUresult cuGraphKernelNodeGetAttribute(CUgraphNode node, int attr, void* value);
CUresult cuGraphKernelNodeSetAttribute(CUgraphNode node, int attr, const void* value);
CUresult cuGraphAddMemcpyNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, const CUDA_MEMCPY3D* copy, CUcontext ctx);
CUresult cuGraphMemcpyNodeGetParams(CUgraphNode node, CUDA_MEMCPY3D* params);
CUresult cuGraphMemcpyNodeSetParams(CUgraphNode node, const CUDA_MEMCPY3D* params);
CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec exec, CUgraphNode node, const CUDA_MEMCPY3D* params, CUcontext ctx);
CUresult cuGraphAddMemsetNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, const CUDA_MEMSET_NODE_PARAMS_v1* params, CUcontext ctx);
CUresult cuGraphMemsetNodeGetParams(CUgraphNode node, CUDA_MEMSET_NODE_PARAMS_v1* params);
CUresult cuGraphMemsetNodeSetParams(CUgraphNode node, const CUDA_MEMSET_NODE_PARAMS_v1* params);
CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec exec, CUgraphNode node, const CUDA_MEMSET_NODE_PARAMS_v1* params, CUcontext ctx);
CUresult cuGraphAddBatchMemOpNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, const CUDA_BATCH_MEM_OP_NODE_PARAMS_v1* params);
CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode node, CUDA_BATCH_MEM_OP_NODE_PARAMS_v1* params);
CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode node, const CUDA_BATCH_MEM_OP_NODE_PARAMS_v1* params);
CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec exec, CUgraphNode node, const CUDA_BATCH_MEM_OP_NODE_PARAMS_v1* params);
CUresult cuGraphAddMemAllocNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUDA_MEM_ALLOC_NODE_PARAMS_v1* params);
CUresult cuGraphMemAllocNodeGetParams(CUgraphNode node, CUDA_MEM_ALLOC_NODE_PARAMS_v1* params);
CUresult cuGraphAddMemFreeNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUdeviceptr dptr);
CUresult cuGraphMemFreeNodeGetParams(CUgraphNode node, CUdeviceptr* dptr);
CUresult cuGraphAddNode_v2(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, const CUgraphEdgeData* edges, size_t n, CUgraphNodeParams* params);
CUresult cuGraphAddNode(CUgraphNode* out, CUgraph graph, const CUgraphNode* deps, size_t n, CUgraphNodeParams* params);
CUresult cuGraphNodeSetParams(CUgraphNode node, CUgraphNodeParams* params);
CUresult cuGraphExecNodeSetParams(CUgraphExec exec, CUgraphNode node, CUgraphNodeParams* params);
CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const void*);
CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const void*);
CUresult cuStreamWriteValue32(CUstream st, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
CUresult cuStreamWriteValue32_v2(CUstream st, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
CUresult cuStreamWriteValue64(CUstream st, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
CUresult cuStreamWriteValue64_v2(CUstream st, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
CUresult cuStreamWaitValue32(CUstream st, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
CUresult cuStreamWaitValue32_v2(CUstream st, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
CUresult cuStreamWaitValue64(CUstream st, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
CUresult cuStreamWaitValue64_v2(CUstream st, CUdeviceptr addr, cuuint64_t value, unsigned int flags);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_CUDA_GRAPH_H_ */
