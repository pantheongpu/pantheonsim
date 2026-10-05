// The driver API's structures that its graph, memory-pool, green-context and
// external-resource entry points pass, as NVIDIA's public cuda.h documents
// them, written out here so that neither CUDA shim needs the toolkit's headers
// for them. libvgpucuda is built without any, and libvgpucudart is built
// against whichever toolkit is installed: CUDA 12.0's cuda.h has none of the
// newer structures, so naming the toolkit's would make the two libraries
// differ by build host. Each layout below is the one the headers give, which
// nvidia/tests/e2e/driver_abi_layouts.cpp checks against the installed toolkit
// at compile time.
//
// Handles are void*, as every one of them is a pointer in the real API:
// CUgraph and cudaGraph_t are the same handle, as are CUstream and
// cudaStream_t, CUevent and cudaEvent_t, CUgraphExec and cudaGraphExec_t,
// CUgraphNode and cudaGraphNode_t, CUmemoryPool and cudaMemPool_t.
#ifndef VGPU_CU_ABI_H_
#define VGPU_CU_ABI_H_

#include <stddef.h>
#include <stdint.h>

namespace vcu {

typedef void* Ctx;
typedef void* Func;
typedef void* Kernel;
typedef void* Stream;
typedef void* Event;
typedef void* Graph;
typedef void* GraphNode;
typedef void* GraphExec;
typedef void* Array;
typedef void* MemPool;
typedef void* ExtSem;
typedef void* ExtMem;
typedef void* UserObject;
typedef unsigned long long DevPtr;

// CUgraphNodeType.
enum NodeType {
  kNodeKernel = 0, kNodeMemcpy = 1, kNodeMemset = 2, kNodeHost = 3, kNodeGraph = 4, kNodeEmpty = 5,
  kNodeWaitEvent = 6, kNodeEventRecord = 7, kNodeExtSemSignal = 8, kNodeExtSemWait = 9,
  kNodeMemAlloc = 10, kNodeMemFree = 11, kNodeBatchMemOp = 12, kNodeConditional = 13,
};

struct KernelNodeParams {   // CUDA_KERNEL_NODE_PARAMS_v1: the first ten fields of v2 and v3
  Func func;
  unsigned gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes;
  void** kernelParams;
  void** extra;
};
struct KernelNodeParamsV3 {   // CUDA_KERNEL_NODE_PARAMS_v2 and v3, which have one layout
  Func func;
  unsigned gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes;
  void** kernelParams;
  void** extra;
  Kernel kern;
  Ctx ctx;
};
struct MemsetNodeParams {   // CUDA_MEMSET_NODE_PARAMS_v1
  DevPtr dst;
  size_t pitch;
  unsigned value, elementSize;
  size_t width, height;
};
struct MemsetNodeParamsV2 {
  DevPtr dst;
  size_t pitch;
  unsigned value, elementSize;
  size_t width, height;
  Ctx ctx;
};
struct HostNodeParams {   // CUDA_HOST_NODE_PARAMS_v1
  void (*fn)(void*);
  void* userData;
};
struct HostNodeParamsV2 {
  void (*fn)(void*);
  void* userData;
  unsigned syncMode;
};

// CUDA_MEMCPY3D, with the memory types of CUmemorytype.
enum MemType { kMemHost = 1, kMemDevice = 2, kMemArray = 3, kMemUnified = 4 };
struct Memcpy3D {
  size_t srcXInBytes, srcY, srcZ, srcLOD;
  unsigned srcMemoryType;
  const void* srcHost;
  DevPtr srcDevice;
  Array srcArray;
  void* reserved0;
  size_t srcPitch, srcHeight;
  size_t dstXInBytes, dstY, dstZ, dstLOD;
  unsigned dstMemoryType;
  void* dstHost;
  DevPtr dstDevice;
  Array dstArray;
  void* reserved1;
  size_t dstPitch, dstHeight;
  size_t WidthInBytes, Height, Depth;
};
struct MemcpyNodeParams {   // CUDA_MEMCPY_NODE_PARAMS
  int flags, reserved;
  Ctx copyCtx;
  Memcpy3D copyParams;
};

struct EventNodeParams { Event event; };   // CUDA_EVENT_RECORD_NODE_PARAMS and _WAIT_

struct ChildGraphNodeParams {   // CUDA_CHILD_GRAPH_NODE_PARAMS
  Graph graph;
  int ownership;   // CU_GRAPH_CHILD_GRAPH_OWNERSHIP_CLONE 0, _MOVE 1, _INVALID -1
};

struct MemLocation { int type; int id; };   // CUmemLocation
struct MemPoolProps {   // CUmemPoolProps
  int allocType;
  int handleTypes;
  MemLocation location;
  void* win32SecurityAttributes;
  size_t maxSize;
  unsigned short usage;
  unsigned char reserved[54];
};
struct MemAccessDesc { MemLocation location; int flags; };   // CUmemAccessDesc
struct MemAllocNodeParams {   // CUDA_MEM_ALLOC_NODE_PARAMS_v1 and v2
  MemPoolProps poolProps;
  const MemAccessDesc* accessDescs;
  size_t accessDescCount;
  size_t bytesize;
  DevPtr dptr;
};
struct MemFreeNodeParams { DevPtr dptr; };

// CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS and _WAIT_PARAMS (v1). Both are 144
// bytes: a 72-byte `params` (fence value, an NvSciSync pointer, keyed mutex
// key and timeout), flags, and 16 reserved words.
struct ExtSemSignalParams {
  struct {
    struct { unsigned long long value; } fence;
    union { void* fence; unsigned long long reserved; } nvSciSync;
    struct { unsigned long long key; } keyedMutex;
    unsigned reserved[12];
  } params;
  unsigned flags;
  unsigned reserved[16];
};
struct ExtSemWaitParams {
  struct {
    struct { unsigned long long value; } fence;
    union { void* fence; unsigned long long reserved; } nvSciSync;
    struct { unsigned long long key; unsigned timeoutMs; } keyedMutex;
    unsigned reserved[10];
  } params;
  unsigned flags;
  unsigned reserved[16];
};
struct ExtSemSignalNodeParams {   // CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 and v2
  ExtSem* extSemArray;
  const ExtSemSignalParams* paramsArray;
  unsigned numExtSems;
};
struct ExtSemWaitNodeParams {
  ExtSem* extSemArray;
  const ExtSemWaitParams* paramsArray;
  unsigned numExtSems;
};

// CUstreamBatchMemOpParams: a 48-byte union whose first word is the operation.
struct BatchMemOpParams { unsigned long long pad[6]; };
struct BatchMemOpNodeParams {   // CUDA_BATCH_MEM_OP_NODE_PARAMS_v1 and v2
  Ctx ctx;
  unsigned count;
  BatchMemOpParams* paramArray;
  unsigned flags;
};

struct ConditionalNodeParams {   // CUDA_CONDITIONAL_NODE_PARAMS
  unsigned long long handle;
  int type;     // CU_GRAPH_COND_TYPE_IF 0, _WHILE 1, _SWITCH 2
  unsigned size;
  Graph* phGraph_out;
  Ctx ctx;
};

// CUgraphNodeParams: a type, three reserved words, a 232-byte union (29
// 64-bit words) and one more reserved word.
struct GraphNodeParams {
  int type;
  int reserved0[3];
  union {
    long long reserved1[29];
    KernelNodeParamsV3 kernel;
    MemcpyNodeParams memcpy;
    MemsetNodeParamsV2 memset;
    HostNodeParamsV2 host;
    ChildGraphNodeParams graph;
    EventNodeParams eventWait;
    EventNodeParams eventRecord;
    ExtSemSignalNodeParams extSemSignal;
    ExtSemWaitNodeParams extSemWait;
    MemAllocNodeParams alloc;
    MemFreeNodeParams free;
    BatchMemOpNodeParams memOp;
    ConditionalNodeParams conditional;
  };
  long long reserved2;
};

struct GraphEdgeData {   // CUgraphEdgeData
  unsigned char from_port, to_port, type, reserved[5];
};

struct GraphInstantiateParams {   // CUDA_GRAPH_INSTANTIATE_PARAMS
  unsigned long long flags;
  Stream hUploadStream;
  GraphNode hErrNode_out;
  int result_out;
};

struct GraphExecUpdateResultInfo {   // CUgraphExecUpdateResultInfo
  int result;
  GraphNode errorNode;
  GraphNode errorFromNode;
};

// CUkernelNodeAttrValue and CUlaunchAttributeValue: 64 bytes, a union.
struct LaunchAttrValue { unsigned char bytes[64]; };

}  // namespace vcu

#endif  // VGPU_CU_ABI_H_
