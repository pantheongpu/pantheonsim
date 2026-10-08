/*
 * vgpu_cutensornet.h -- VirtualGPU's clean-room declarations of the cuTensorNet API.
 *
 * Written from NVIDIA's publicly documented cuTensorNet API
 * (docs.nvidia.com/cuda/cuquantum/latest/cutensornet), version 2.14, so that
 * programs built against cuQuantum's tensor-network library can run on
 * VirtualGPU's libcutensornet. It contains no NVIDIA code. Type layouts and
 * numeric values (status codes, enums, attribute numbers) follow the
 * documented ABI, so a program compiled with NVIDIA's cutensornet.h and one
 * compiled with this header call the same library the same way.
 *
 * Why the simulator needs its own: NVIDIA's libcutensornet carries a
 * statically linked CUDA runtime, which reaches the driver through NVIDIA's
 * undocumented internal interface, so it cannot run on a simulated driver.
 *
 * Declared is the subset VirtualGPU implements: tensor network contraction
 * (the network API and the older descriptor/plan API), the contraction
 * optimizer, slicing, workspace management, the tensor decompositions
 * (QR, SVD, gate split), and the high-level state API (states, tensor
 * operators, network operators, accessors, expectations, marginals, samplers,
 * MPS). Distributed execution and the undocumented exports are exported by
 * the library but answer NOT_SUPPORTED, so they are not declared here.
 */
#ifndef VGPU_CUTENSORNET_H_
#define VGPU_CUTENSORNET_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h> /* FILE, for cutensornetLoggerSetFile */

#include <driver_types.h>  /* cudaStream_t */
#include <library_types.h> /* cudaDataType_t */
#include <cuComplex.h>    /* cuDoubleComplex, a network operator's coefficients */

#define CUTENSORNET_MAJOR 2
#define CUTENSORNET_MINOR 14
#define CUTENSORNET_PATCH 0
#define CUTENSORNET_VERSION (CUTENSORNET_MAJOR * 10000 + CUTENSORNET_MINOR * 100 + CUTENSORNET_PATCH)

#define CUTENSORNET_ALLOCATOR_NAME_LEN 64

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CUTENSORNET_STATUS_SUCCESS = 0,
  CUTENSORNET_STATUS_NOT_INITIALIZED = 1,
  CUTENSORNET_STATUS_ALLOC_FAILED = 3,
  CUTENSORNET_STATUS_INVALID_VALUE = 7,
  CUTENSORNET_STATUS_ARCH_MISMATCH = 8,
  CUTENSORNET_STATUS_MAPPING_ERROR = 11,
  CUTENSORNET_STATUS_EXECUTION_FAILED = 13,
  CUTENSORNET_STATUS_INTERNAL_ERROR = 14,
  CUTENSORNET_STATUS_NOT_SUPPORTED = 15,
  CUTENSORNET_STATUS_LICENSE_ERROR = 16,
  CUTENSORNET_STATUS_CUBLAS_ERROR = 17,
  CUTENSORNET_STATUS_CUDA_ERROR = 18,
  CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE = 19,
  CUTENSORNET_STATUS_INSUFFICIENT_DRIVER = 20,
  CUTENSORNET_STATUS_IO_ERROR = 21,
  CUTENSORNET_STATUS_CUTENSOR_VERSION_MISMATCH = 22,
  CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR = 23,
  CUTENSORNET_STATUS_ALL_HYPER_SAMPLES_FAILED = 24,
  CUTENSORNET_STATUS_CUSOLVER_ERROR = 25,
  CUTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR = 26,
  CUTENSORNET_STATUS_DISTRIBUTED_FAILURE = 27,
  CUTENSORNET_STATUS_INTERRUPTED = 28,
  CUTENSORNET_STATUS_CUTENSOR_ERROR = 29
} cutensornetStatus_t;

typedef enum {
  CUTENSORNET_COMPUTE_16F = (1U << 0U),
  CUTENSORNET_COMPUTE_16BF = (1U << 10U),
  CUTENSORNET_COMPUTE_TF32 = (1U << 12U),
  CUTENSORNET_COMPUTE_3XTF32 = (1U << 13U),
  CUTENSORNET_COMPUTE_32F = (1U << 2U),
  CUTENSORNET_COMPUTE_64F = (1U << 4U),
  CUTENSORNET_COMPUTE_8U = (1U << 6U),
  CUTENSORNET_COMPUTE_8I = (1U << 8U),
  CUTENSORNET_COMPUTE_32U = (1U << 7U),
  CUTENSORNET_COMPUTE_32I = (1U << 9U)
} cutensornetComputeType_t;

typedef enum {
  CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONSTANT = 0,
  CUTENSORNET_NETWORK_INPUT_TENSORS_CONSTANT = 1,
  CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONJUGATED = 10,
  CUTENSORNET_NETWORK_INPUT_TENSORS_CONJUGATED = 11,
  CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_REQUIRE_GRAD = 20,
  CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD = 21,
  CUTENSORNET_NETWORK_COMPUTE_TYPE = 30
} cutensornetNetworkAttributes_t;

typedef enum { CUTENSORNET_GRAPH_ALGO_RB, CUTENSORNET_GRAPH_ALGO_KWAY } cutensornetGraphAlgo_t;
typedef enum { CUTENSORNET_MEMORY_MODEL_HEURISTIC, CUTENSORNET_MEMORY_MODEL_CUTENSOR } cutensornetMemoryModel_t;
typedef enum { CUTENSORNET_OPTIMIZER_COST_FLOPS, CUTENSORNET_OPTIMIZER_COST_TIME } cutensornetOptimizerCost_t;
typedef enum {
  CUTENSORNET_SMART_OPTION_DISABLED = 0,
  CUTENSORNET_SMART_OPTION_ENABLED = 1
} cutensornetSmartOption_t;

typedef enum {
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_PARTITIONS = 0,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_CUTOFF_SIZE = 1,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM = 2,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR = 3,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_ITERATIONS = 4,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_CUTS = 5,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS = 10,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES = 11,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING = 20,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL = 21,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR = 22,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES = 23,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_SLICE_FACTOR = 24,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES = 30,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS = 31,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR = 40,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED = 60,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE = 61,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS = 62,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION = 63,
  CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GPU_ARCH = 64
} cutensornetContractionOptimizerConfigAttributes_t;

typedef enum {
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH = 0,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES = 10,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES = 11,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE = 12,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT = 13,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_CONFIG = 14,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_OVERHEAD = 15,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PHASE1_FLOP_COUNT = 20,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT = 21,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS_EST = 22,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_RUNTIME_EST = 23,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR = 24,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES = 30,
  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_INTERMEDIATE_MODES = 31
} cutensornetContractionOptimizerInfoAttributes_t;

typedef enum {
  CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
  CUTENSORNET_CONTRACTION_AUTOTUNE_INTERMEDIATE_MODES
} cutensornetContractionAutotunePreferenceAttributes_t;

typedef enum {
  CUTENSORNET_NETWORK_AUTOTUNE_MAX_ITERATIONS,
  CUTENSORNET_NETWORK_AUTOTUNE_INTERMEDIATE_MODES
} cutensornetNetworkAutotunePreferenceAttributes_t;

typedef void* cutensornetNetworkDescriptor_t;
typedef void* cutensornetContractionPlan_t;
typedef void* cutensornetHandle_t;
typedef void* cutensornetWorkspaceDescriptor_t;

typedef enum {
  CUTENSORNET_WORKSIZE_PREF_MIN = 0,
  CUTENSORNET_WORKSIZE_PREF_RECOMMENDED = 1,
  CUTENSORNET_WORKSIZE_PREF_MAX = 2
} cutensornetWorksizePref_t;

typedef enum { CUTENSORNET_MEMSPACE_DEVICE = 0, CUTENSORNET_MEMSPACE_HOST = 1 } cutensornetMemspace_t;

typedef enum { CUTENSORNET_WORKSPACE_SCRATCH = 0, CUTENSORNET_WORKSPACE_CACHE = 1 } cutensornetWorkspaceKind_t;

typedef struct {
  int32_t numTensors;
  int32_t* data;
} cutensornetTensorIDList_t;

typedef struct __attribute__((aligned(4), packed)) {
  int32_t first;
  int32_t second;
} cutensornetNodePair_t;

typedef struct {
  int32_t numContractions;
  cutensornetNodePair_t* data;
} cutensornetContractionPath_t;

typedef struct {
  int32_t slicedMode;
  int64_t slicedExtent;
} cutensornetSliceInfoPair_t;

typedef struct {
  uint32_t numSlicedModes;
  cutensornetSliceInfoPair_t* data;
} cutensornetSlicingConfig_t;

typedef void* cutensornetContractionOptimizerConfig_t;
typedef void* cutensornetContractionOptimizerInfo_t;
typedef void* cutensornetContractionAutotunePreference_t;
typedef void* cutensornetNetworkAutotunePreference_t;
typedef void* cutensornetSliceGroup_t;

typedef struct {
  int32_t isConjugate;
  int32_t isConstant;
  int32_t requiresGradient;
} cutensornetTensorQualifiers_t;

typedef void* cutensornetTensorDescriptor_t;

typedef enum {
  CUTENSORNET_TENSOR_DESCRIPTOR_IS_DISTRIBUTED = 0,
  CUTENSORNET_TENSOR_DESCRIPTOR_ELEMENT_STRIDES = 1,
  CUTENSORNET_TENSOR_DESCRIPTOR_BLOCK_SIZES = 2,
  CUTENSORNET_TENSOR_DESCRIPTOR_BLOCK_STRIDES = 3,
  CUTENSORNET_TENSOR_DESCRIPTOR_NRANKS_PER_MODE = 4,
  CUTENSORNET_TENSOR_DESCRIPTOR_LOCAL_DATA_SIZE = 5,
  CUTENSORNET_TENSOR_DESCRIPTOR_LOCAL_EXTENTS = 6
} cutensornetTensorDescriptorAttributes_t;

typedef void* cutensornetTensorSVDConfig_t;

typedef enum {
  CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF,
  CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF,
  CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION,
  CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION,
  CUTENSORNET_TENSOR_SVD_CONFIG_ALGO,
  CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS,
  CUTENSORNET_TENSOR_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF
} cutensornetTensorSVDConfigAttributes_t;

typedef enum {
  CUTENSORNET_TENSOR_SVD_ALGO_GESVD,
  CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ,
  CUTENSORNET_TENSOR_SVD_ALGO_GESVDP,
  CUTENSORNET_TENSOR_SVD_ALGO_GESVDR
} cutensornetTensorSVDAlgo_t;

typedef struct {
  double tol;
  int32_t maxSweeps;
} cutensornetGesvdjParams_t;

typedef struct {
  int64_t oversampling;
  int64_t niters;
} cutensornetGesvdrParams_t;

typedef enum {
  CUTENSORNET_TENSOR_SVD_PARTITION_NONE,
  CUTENSORNET_TENSOR_SVD_PARTITION_US,
  CUTENSORNET_TENSOR_SVD_PARTITION_SV,
  CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL
} cutensornetTensorSVDPartition_t;

typedef enum {
  CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE,
  CUTENSORNET_TENSOR_SVD_NORMALIZATION_L1,
  CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2,
  CUTENSORNET_TENSOR_SVD_NORMALIZATION_LINF
} cutensornetTensorSVDNormalization_t;

typedef void* cutensornetTensorSVDInfo_t;

typedef enum {
  CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT,
  CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT,
  CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT,
  CUTENSORNET_TENSOR_SVD_INFO_ALGO,
  CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS
} cutensornetTensorSVDInfoAttributes_t;

typedef struct {
  double residual;
  int32_t sweeps;
} cutensornetGesvdjStatus_t;

typedef struct {
  double errSigma;
} cutensornetGesvdpStatus_t;

typedef enum {
  CUTENSORNET_GATE_SPLIT_ALGO_DIRECT,
  CUTENSORNET_GATE_SPLIT_ALGO_REDUCED
} cutensornetGateSplitAlgo_t;

typedef struct {
  void* ctx;
  int (*device_alloc)(void* ctx, void** ptr, size_t size, cudaStream_t stream);
  int (*device_free)(void* ctx, void* ptr, size_t size, cudaStream_t stream);
  char name[CUTENSORNET_ALLOCATOR_NAME_LEN];
} cutensornetDeviceMemHandler_t;

typedef void (*cutensornetLoggerCallback_t)(int32_t logLevel, const char* functionName, const char* message);
typedef void (*cutensornetLoggerCallbackData_t)(int32_t logLevel, const char* functionName, const char* message,
                                                void* userData);

/* Library management */
cutensornetStatus_t cutensornetCreate(cutensornetHandle_t* handle);
cutensornetStatus_t cutensornetDestroy(cutensornetHandle_t handle);
size_t cutensornetGetVersion(void);
size_t cutensornetGetCudartVersion(void);
const char* cutensornetGetErrorString(cutensornetStatus_t error);
const char* cutensornetGetLastError(void);
cutensornetStatus_t cutensornetGetDeviceMemHandler(const cutensornetHandle_t handle,
                                                   cutensornetDeviceMemHandler_t* devMemHandler);
cutensornetStatus_t cutensornetSetDeviceMemHandler(cutensornetHandle_t handle,
                                                   const cutensornetDeviceMemHandler_t* devMemHandler);
cutensornetStatus_t cutensornetLoggerSetCallback(cutensornetLoggerCallback_t callback);
cutensornetStatus_t cutensornetLoggerSetCallbackData(cutensornetLoggerCallbackData_t callback, void* userData);
cutensornetStatus_t cutensornetLoggerSetFile(FILE* file);
cutensornetStatus_t cutensornetLoggerOpenFile(const char* logFile);
cutensornetStatus_t cutensornetLoggerSetLevel(int32_t level);
cutensornetStatus_t cutensornetLoggerSetMask(int32_t mask);
cutensornetStatus_t cutensornetLoggerForceDisable(void);

/* Networks */
cutensornetStatus_t cutensornetCreateNetworkDescriptor(
    const cutensornetHandle_t handle, int32_t numInputs, const int32_t numModesIn[],
    const int64_t* const extentsIn[], const int64_t* const stridesIn[], const int32_t* const modesIn[],
    const cutensornetTensorQualifiers_t qualifiersIn[], int32_t numModesOut, const int64_t extentsOut[],
    const int64_t stridesOut[], const int32_t modesOut[], cudaDataType_t dataType,
    cutensornetComputeType_t computeType, cutensornetNetworkDescriptor_t* networkDesc);
cutensornetStatus_t cutensornetCreateNetwork(const cutensornetHandle_t handle,
                                             cutensornetNetworkDescriptor_t* networkDesc);
cutensornetStatus_t cutensornetDestroyNetworkDescriptor(cutensornetNetworkDescriptor_t networkDesc);
cutensornetStatus_t cutensornetDestroyNetwork(cutensornetNetworkDescriptor_t networkDesc);
cutensornetStatus_t cutensornetNetworkAppendTensor(const cutensornetHandle_t handle,
                                                   cutensornetNetworkDescriptor_t networkDesc, int32_t numModes,
                                                   const int64_t extents[], const int32_t modeLabels[],
                                                   const cutensornetTensorQualifiers_t* const qualifiers,
                                                   cudaDataType_t dataType, int64_t* tensorId);
cutensornetStatus_t cutensornetNetworkSetOutputTensor(const cutensornetHandle_t handle,
                                                      cutensornetNetworkDescriptor_t networkDesc, int32_t numModes,
                                                      const int32_t modeLabels[], cudaDataType_t dataType);
cutensornetStatus_t cutensornetNetworkGetAttribute(const cutensornetHandle_t handle,
                                                   const cutensornetNetworkDescriptor_t networkDesc,
                                                   cutensornetNetworkAttributes_t attr, void* buffer,
                                                   size_t sizeInBytes);
cutensornetStatus_t cutensornetNetworkSetAttribute(const cutensornetHandle_t handle,
                                                   cutensornetNetworkDescriptor_t networkDesc,
                                                   cutensornetNetworkAttributes_t attr, const void* const buffer,
                                                   size_t sizeInBytes);
cutensornetStatus_t cutensornetGetOutputTensorDetails(const cutensornetHandle_t handle,
                                                      const cutensornetNetworkDescriptor_t networkDesc,
                                                      int32_t* numModes, size_t* dataSize, int32_t* modeLabels,
                                                      int64_t* extents, int64_t* strides);
cutensornetStatus_t cutensornetGetOutputTensorDescriptor(const cutensornetHandle_t handle,
                                                         const cutensornetNetworkDescriptor_t networkDesc,
                                                         cutensornetTensorDescriptor_t* outputTensorDesc);
cutensornetStatus_t cutensornetNetworkSetInputTensorMemory(const cutensornetHandle_t handle,
                                                           cutensornetNetworkDescriptor_t networkDesc,
                                                           int64_t tensorId, const void* const buffer,
                                                           const int64_t strides[]);
cutensornetStatus_t cutensornetNetworkSetOutputTensorMemory(const cutensornetHandle_t handle,
                                                            cutensornetNetworkDescriptor_t networkDesc,
                                                            void* const buffer, const int64_t strides[]);

/* Tensor descriptors */
cutensornetStatus_t cutensornetCreateTensorDescriptor(const cutensornetHandle_t handle, int32_t numModes,
                                                      const int64_t extents[], const int64_t strides[],
                                                      const int32_t modeLabels[], cudaDataType_t dataType,
                                                      cutensornetTensorDescriptor_t* tensorDesc);
cutensornetStatus_t cutensornetGetTensorDetails(const cutensornetHandle_t handle,
                                                const cutensornetTensorDescriptor_t tensorDesc, int32_t* numModes,
                                                size_t* dataSize, int32_t* modeLabels, int64_t* extents,
                                                int64_t* strides);
cutensornetStatus_t cutensornetTensorDescriptorGetAttribute(const cutensornetHandle_t handle,
                                                            const cutensornetTensorDescriptor_t tensorDesc,
                                                            cutensornetTensorDescriptorAttributes_t attr,
                                                            void* buffer, size_t sizeInBytes);
cutensornetStatus_t cutensornetDestroyTensorDescriptor(cutensornetTensorDescriptor_t tensorDesc);

/* Workspaces */
cutensornetStatus_t cutensornetCreateWorkspaceDescriptor(const cutensornetHandle_t handle,
                                                         cutensornetWorkspaceDescriptor_t* workDesc);
cutensornetStatus_t cutensornetWorkspaceComputeSizes(const cutensornetHandle_t handle,
                                                     const cutensornetNetworkDescriptor_t networkDesc,
                                                     const cutensornetContractionOptimizerInfo_t optimizerInfo,
                                                     cutensornetWorkspaceDescriptor_t workDesc);
cutensornetStatus_t cutensornetWorkspaceComputeContractionSizes(
    const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc,
    const cutensornetContractionOptimizerInfo_t optimizerInfo, cutensornetWorkspaceDescriptor_t workDesc);
cutensornetStatus_t cutensornetWorkspaceGetSize(const cutensornetHandle_t handle,
                                                const cutensornetWorkspaceDescriptor_t workDesc,
                                                cutensornetWorksizePref_t workPref, cutensornetMemspace_t memSpace,
                                                uint64_t* workspaceSize);
cutensornetStatus_t cutensornetWorkspaceGetMemorySize(const cutensornetHandle_t handle,
                                                      const cutensornetWorkspaceDescriptor_t workDesc,
                                                      cutensornetWorksizePref_t workPref,
                                                      cutensornetMemspace_t memSpace,
                                                      cutensornetWorkspaceKind_t workKind, int64_t* memorySize);
cutensornetStatus_t cutensornetWorkspaceSet(const cutensornetHandle_t handle,
                                            cutensornetWorkspaceDescriptor_t workDesc,
                                            cutensornetMemspace_t memSpace, void* const workspacePtr,
                                            uint64_t workspaceSize);
cutensornetStatus_t cutensornetWorkspaceSetMemory(const cutensornetHandle_t handle,
                                                  cutensornetWorkspaceDescriptor_t workDesc,
                                                  cutensornetMemspace_t memSpace,
                                                  cutensornetWorkspaceKind_t workKind, void* const memoryPtr,
                                                  int64_t memorySize);
cutensornetStatus_t cutensornetWorkspaceGet(const cutensornetHandle_t handle,
                                            const cutensornetWorkspaceDescriptor_t workDesc,
                                            cutensornetMemspace_t memSpace, void** workspacePtr,
                                            uint64_t* workspaceSize);
cutensornetStatus_t cutensornetWorkspaceGetMemory(const cutensornetHandle_t handle,
                                                  const cutensornetWorkspaceDescriptor_t workDesc,
                                                  cutensornetMemspace_t memSpace,
                                                  cutensornetWorkspaceKind_t workKind, void** memoryPtr,
                                                  int64_t* memorySize);
cutensornetStatus_t cutensornetWorkspacePurgeCache(const cutensornetHandle_t handle,
                                                   cutensornetWorkspaceDescriptor_t workDesc,
                                                   cutensornetMemspace_t memSpace);
cutensornetStatus_t cutensornetDestroyWorkspaceDescriptor(cutensornetWorkspaceDescriptor_t workDesc);

/* The contraction optimizer */
cutensornetStatus_t cutensornetCreateContractionOptimizerConfig(
    const cutensornetHandle_t handle, cutensornetContractionOptimizerConfig_t* optimizerConfig);
cutensornetStatus_t cutensornetDestroyContractionOptimizerConfig(
    cutensornetContractionOptimizerConfig_t optimizerConfig);
cutensornetStatus_t cutensornetContractionOptimizerConfigGetAttribute(
    const cutensornetHandle_t handle, const cutensornetContractionOptimizerConfig_t optimizerConfig,
    cutensornetContractionOptimizerConfigAttributes_t attr, void* buffer, size_t sizeInBytes);
cutensornetStatus_t cutensornetContractionOptimizerConfigSetAttribute(
    const cutensornetHandle_t handle, cutensornetContractionOptimizerConfig_t optimizerConfig,
    cutensornetContractionOptimizerConfigAttributes_t attr, const void* buffer, size_t sizeInBytes);
cutensornetStatus_t cutensornetDestroyContractionOptimizerInfo(cutensornetContractionOptimizerInfo_t optimizerInfo);
cutensornetStatus_t cutensornetCreateContractionOptimizerInfo(const cutensornetHandle_t handle,
                                                              const cutensornetNetworkDescriptor_t networkDesc,
                                                              cutensornetContractionOptimizerInfo_t* optimizerInfo);
cutensornetStatus_t cutensornetContractionOptimize(const cutensornetHandle_t handle,
                                                   const cutensornetNetworkDescriptor_t networkDesc,
                                                   const cutensornetContractionOptimizerConfig_t optimizerConfig,
                                                   uint64_t workspaceSizeConstraint,
                                                   cutensornetContractionOptimizerInfo_t optimizerInfo);
cutensornetStatus_t cutensornetNetworkSetOptimizerInfo(const cutensornetHandle_t handle,
                                                       cutensornetNetworkDescriptor_t networkDesc,
                                                       const cutensornetContractionOptimizerInfo_t optimizerInfo);
cutensornetStatus_t cutensornetContractionOptimizerInfoGetAttribute(
    const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo,
    cutensornetContractionOptimizerInfoAttributes_t attr, void* buffer, size_t sizeInBytes);
cutensornetStatus_t cutensornetContractionOptimizerInfoSetAttribute(
    const cutensornetHandle_t handle, cutensornetContractionOptimizerInfo_t optimizerInfo,
    cutensornetContractionOptimizerInfoAttributes_t attr, const void* buffer, size_t sizeInBytes);
cutensornetStatus_t cutensornetContractionOptimizerInfoGetPackedSize(
    const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo,
    size_t* sizeInBytes);
cutensornetStatus_t cutensornetContractionOptimizerInfoPackData(
    const cutensornetHandle_t handle, const cutensornetContractionOptimizerInfo_t optimizerInfo, void* buffer,
    size_t sizeInBytes);
cutensornetStatus_t cutensornetCreateContractionOptimizerInfoFromPackedData(
    const cutensornetHandle_t handle, const cutensornetNetworkDescriptor_t networkDesc, const void* buffer,
    size_t sizeInBytes, cutensornetContractionOptimizerInfo_t* optimizerInfo);
cutensornetStatus_t cutensornetUpdateContractionOptimizerInfoFromPackedData(
    const cutensornetHandle_t handle, const void* buffer, size_t sizeInBytes,
    cutensornetContractionOptimizerInfo_t optimizerInfo);

/* Plans and contraction */
cutensornetStatus_t cutensornetCreateContractionPlan(const cutensornetHandle_t handle,
                                                     const cutensornetNetworkDescriptor_t networkDesc,
                                                     const cutensornetContractionOptimizerInfo_t optimizerInfo,
                                                     const cutensornetWorkspaceDescriptor_t workDesc,
                                                     cutensornetContractionPlan_t* plan);
cutensornetStatus_t cutensornetNetworkPrepareContraction(const cutensornetHandle_t handle,
                                                         cutensornetNetworkDescriptor_t networkDesc,
                                                         const cutensornetWorkspaceDescriptor_t workDesc);
cutensornetStatus_t cutensornetDestroyContractionPlan(cutensornetContractionPlan_t plan);
cutensornetStatus_t cutensornetContractionAutotune(const cutensornetHandle_t handle,
                                                   cutensornetContractionPlan_t plan,
                                                   const void* const rawDataIn[], void* rawDataOut,
                                                   cutensornetWorkspaceDescriptor_t workDesc,
                                                   const cutensornetContractionAutotunePreference_t pref,
                                                   cudaStream_t stream);
cutensornetStatus_t cutensornetNetworkAutotuneContraction(const cutensornetHandle_t handle,
                                                          cutensornetNetworkDescriptor_t networkDesc,
                                                          const cutensornetWorkspaceDescriptor_t workDesc,
                                                          const cutensornetNetworkAutotunePreference_t pref,
                                                          cudaStream_t stream);
cutensornetStatus_t cutensornetCreateContractionAutotunePreference(
    const cutensornetHandle_t handle, cutensornetContractionAutotunePreference_t* autotunePreference);
cutensornetStatus_t cutensornetCreateNetworkAutotunePreference(
    const cutensornetHandle_t handle, cutensornetNetworkAutotunePreference_t* autotunePreference);
cutensornetStatus_t cutensornetContractionAutotunePreferenceGetAttribute(
    const cutensornetHandle_t handle, const cutensornetContractionAutotunePreference_t autotunePreference,
    cutensornetContractionAutotunePreferenceAttributes_t attr, void* buffer, size_t sizeInBytes);
cutensornetStatus_t cutensornetNetworkAutotunePreferenceGetAttribute(
    const cutensornetHandle_t handle, const cutensornetNetworkAutotunePreference_t autotunePreference,
    cutensornetNetworkAutotunePreferenceAttributes_t attr, void* buffer, size_t sizeInBytes);
cutensornetStatus_t cutensornetContractionAutotunePreferenceSetAttribute(
    const cutensornetHandle_t handle, cutensornetContractionAutotunePreference_t autotunePreference,
    cutensornetContractionAutotunePreferenceAttributes_t attr, const void* buffer, size_t sizeInBytes);
cutensornetStatus_t cutensornetNetworkAutotunePreferenceSetAttribute(
    const cutensornetHandle_t handle, cutensornetNetworkAutotunePreference_t autotunePreference,
    cutensornetNetworkAutotunePreferenceAttributes_t attr, const void* buf, size_t sizeInBytes);
cutensornetStatus_t cutensornetDestroyContractionAutotunePreference(
    cutensornetContractionAutotunePreference_t autotunePreference);
cutensornetStatus_t cutensornetDestroyNetworkAutotunePreference(
    cutensornetNetworkAutotunePreference_t autotunePreference);
cutensornetStatus_t cutensornetContraction(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan,
                                           const void* const rawDataIn[], void* rawDataOut,
                                           cutensornetWorkspaceDescriptor_t workDesc, int64_t sliceId,
                                           cudaStream_t stream);
cutensornetStatus_t cutensornetCreateSliceGroupFromIDRange(const cutensornetHandle_t handle, int64_t sliceIdStart,
                                                           int64_t sliceIdStop, int64_t sliceIdStep,
                                                           cutensornetSliceGroup_t* sliceGroup);
cutensornetStatus_t cutensornetCreateSliceGroupFromIDs(const cutensornetHandle_t handle,
                                                       const int64_t* beginIDSequence, const int64_t* endIDSequence,
                                                       cutensornetSliceGroup_t* sliceGroup);
cutensornetStatus_t cutensornetDestroySliceGroup(cutensornetSliceGroup_t sliceGroup);
cutensornetStatus_t cutensornetContractSlices(const cutensornetHandle_t handle, cutensornetContractionPlan_t plan,
                                              const void* const rawDataIn[], void* rawDataOut,
                                              int32_t accumulateOutput, cutensornetWorkspaceDescriptor_t workDesc,
                                              const cutensornetSliceGroup_t sliceGroup, cudaStream_t stream);
cutensornetStatus_t cutensornetNetworkContract(const cutensornetHandle_t handle,
                                               cutensornetNetworkDescriptor_t networkDesc, int32_t accumulateOutput,
                                               const cutensornetWorkspaceDescriptor_t workDesc,
                                               const cutensornetSliceGroup_t sliceGroup, cudaStream_t stream);

/* Decompositions */
cutensornetStatus_t cutensornetCreateTensorSVDConfig(const cutensornetHandle_t handle,
                                                     cutensornetTensorSVDConfig_t* svdConfig);
cutensornetStatus_t cutensornetDestroyTensorSVDConfig(cutensornetTensorSVDConfig_t svdConfig);
cutensornetStatus_t cutensornetTensorSVDConfigGetAttribute(const cutensornetHandle_t handle,
                                                           const cutensornetTensorSVDConfig_t svdConfig,
                                                           cutensornetTensorSVDConfigAttributes_t attr,
                                                           void* buffer, size_t sizeInBytes);
cutensornetStatus_t cutensornetTensorSVDConfigSetAttribute(const cutensornetHandle_t handle,
                                                           cutensornetTensorSVDConfig_t svdConfig,
                                                           cutensornetTensorSVDConfigAttributes_t attr,
                                                           const void* buffer, size_t sizeInBytes);
cutensornetStatus_t cutensornetWorkspaceComputeSVDSizes(const cutensornetHandle_t handle,
                                                        const cutensornetTensorDescriptor_t descTensorIn,
                                                        const cutensornetTensorDescriptor_t descTensorU,
                                                        const cutensornetTensorDescriptor_t descTensorV,
                                                        const cutensornetTensorSVDConfig_t svdConfig,
                                                        cutensornetWorkspaceDescriptor_t workDesc);
cutensornetStatus_t cutensornetWorkspaceComputeQRSizes(const cutensornetHandle_t handle,
                                                       const cutensornetTensorDescriptor_t descTensorIn,
                                                       const cutensornetTensorDescriptor_t descTensorQ,
                                                       const cutensornetTensorDescriptor_t descTensorR,
                                                       cutensornetWorkspaceDescriptor_t workDesc);
cutensornetStatus_t cutensornetCreateTensorSVDInfo(const cutensornetHandle_t handle,
                                                   cutensornetTensorSVDInfo_t* svdInfo);
cutensornetStatus_t cutensornetTensorSVDInfoGetAttribute(const cutensornetHandle_t handle,
                                                         const cutensornetTensorSVDInfo_t svdInfo,
                                                         cutensornetTensorSVDInfoAttributes_t attr, void* buffer,
                                                         size_t sizeInBytes);
cutensornetStatus_t cutensornetDestroyTensorSVDInfo(cutensornetTensorSVDInfo_t svdInfo);
cutensornetStatus_t cutensornetTensorSVD(const cutensornetHandle_t handle,
                                         const cutensornetTensorDescriptor_t descTensorIn,
                                         const void* const rawDataIn, cutensornetTensorDescriptor_t descTensorU,
                                         void* u, void* s, cutensornetTensorDescriptor_t descTensorV, void* v,
                                         const cutensornetTensorSVDConfig_t svdConfig,
                                         cutensornetTensorSVDInfo_t svdInfo,
                                         const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream);
cutensornetStatus_t cutensornetTensorQR(const cutensornetHandle_t handle,
                                        const cutensornetTensorDescriptor_t descTensorIn,
                                        const void* const rawDataIn, const cutensornetTensorDescriptor_t descTensorQ,
                                        void* q, const cutensornetTensorDescriptor_t descTensorR, void* r,
                                        const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream);
cutensornetStatus_t cutensornetWorkspaceComputeGateSplitSizes(
    const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorInA,
    const cutensornetTensorDescriptor_t descTensorInB, const cutensornetTensorDescriptor_t descTensorInG,
    const cutensornetTensorDescriptor_t descTensorU, const cutensornetTensorDescriptor_t descTensorV,
    const cutensornetGateSplitAlgo_t gateAlgo, const cutensornetTensorSVDConfig_t svdConfig,
    cutensornetComputeType_t computeType, cutensornetWorkspaceDescriptor_t workDesc);
cutensornetStatus_t cutensornetGateSplit(
    const cutensornetHandle_t handle, const cutensornetTensorDescriptor_t descTensorInA, const void* rawDataInA,
    const cutensornetTensorDescriptor_t descTensorInB, const void* rawDataInB,
    const cutensornetTensorDescriptor_t descTensorInG, const void* rawDataInG,
    cutensornetTensorDescriptor_t descTensorU, void* u, void* s, cutensornetTensorDescriptor_t descTensorV, void* v,
    const cutensornetGateSplitAlgo_t gateAlgo, const cutensornetTensorSVDConfig_t svdConfig,
    cutensornetComputeType_t computeType, cutensornetTensorSVDInfo_t svdInfo,
    const cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t stream);

/* ---- Gradients of a network ---- */

cutensornetStatus_t cutensornetNetworkSetGradientTensorMemory(const cutensornetHandle_t handle,
                                                              cutensornetNetworkDescriptor_t networkDesc,
                                                              int64_t correspondingTensorId, void* const buffer,
                                                              const int64_t strides[]);
cutensornetStatus_t cutensornetNetworkSetAdjointTensorMemory(const cutensornetHandle_t handle,
                                                             cutensornetNetworkDescriptor_t networkDesc,
                                                             const void* const buffer, const int64_t strides[]);
cutensornetStatus_t cutensornetNetworkPrepareGradientsBackward(const cutensornetHandle_t handle,
                                                               cutensornetNetworkDescriptor_t networkDesc,
                                                               const cutensornetWorkspaceDescriptor_t workDesc);
cutensornetStatus_t cutensornetNetworkComputeGradientsBackward(const cutensornetHandle_t handle,
                                                               cutensornetNetworkDescriptor_t networkDesc,
                                                               int32_t accumulateOutput,
                                                               const cutensornetWorkspaceDescriptor_t workDesc,
                                                               const cutensornetSliceGroup_t sliceGroup,
                                                               cudaStream_t stream);

/* ---- The state API: tensor network states, operators, and what is computed from them ---- */

typedef void* cutensornetState_t;
typedef void* cutensornetStateAccessor_t;
typedef void* cutensornetStateExpectation_t;
typedef void* cutensornetStateMarginal_t;
typedef void* cutensornetStateSampler_t;
typedef void* cutensornetStateProjectionMPS_t;
typedef void* cutensornetNetworkOperator_t;

typedef enum {
  CUTENSORNET_STATE_PURITY_PURE,
  CUTENSORNET_STATE_PURITY_MIXED
} cutensornetStatePurity_t;

typedef enum {
  CUTENSORNET_BOUNDARY_CONDITION_OPEN
} cutensornetBoundaryCondition_t;

typedef enum {
  CUTENSORNET_STATE_MPS_CANONICAL_CENTER = 0,
  CUTENSORNET_STATE_MPS_SVD_CONFIG_ABS_CUTOFF = 1,
  CUTENSORNET_STATE_MPS_SVD_CONFIG_REL_CUTOFF = 2,
  CUTENSORNET_STATE_MPS_SVD_CONFIG_S_NORMALIZATION = 3,
  CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO = 4,
  CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO_PARAMS = 5,
  CUTENSORNET_STATE_MPS_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF = 6,
  CUTENSORNET_STATE_NUM_HYPER_SAMPLES = 7,
  CUTENSORNET_STATE_CONFIG_MPS_CANONICAL_CENTER = 16,
  CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF = 17,
  CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF = 18,
  CUTENSORNET_STATE_CONFIG_MPS_SVD_S_NORMALIZATION = 19,
  CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO = 20,
  CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO_PARAMS = 21,
  CUTENSORNET_STATE_CONFIG_MPS_SVD_DISCARDED_WEIGHT_CUTOFF = 22,
  CUTENSORNET_STATE_CONFIG_MPS_MPO_APPLICATION = 23,
  CUTENSORNET_STATE_CONFIG_MPS_GAUGE_OPTION = 24,
  CUTENSORNET_STATE_CONFIG_NUM_HYPER_SAMPLES = 30,
  CUTENSORNET_STATE_INFO_FLOPS = 64
} cutensornetStateAttributes_t;

typedef enum {
  CUTENSORNET_STATE_MPO_APPLICATION_INEXACT,
  CUTENSORNET_STATE_MPO_APPLICATION_EXACT
} cutensornetStateMPOApplication_t;

typedef enum {
  CUTENSORNET_STATE_MPS_GAUGE_FREE = 0,
  CUTENSORNET_STATE_MPS_GAUGE_SIMPLE = 1
} cutensornetStateMPSGaugeOption_t;

typedef enum {
  CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES = 0,
  CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES = 1,
  CUTENSORNET_ACCESSOR_INFO_FLOPS = 64
} cutensornetAccessorAttributes_t;

typedef enum {
  CUTENSORNET_EXPECTATION_OPT_NUM_HYPER_SAMPLES = 0,
  CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES = 1,
  CUTENSORNET_EXPECTATION_INFO_FLOPS = 64
} cutensornetExpectationAttributes_t;

typedef enum {
  CUTENSORNET_MARGINAL_KIND_FULL = 0,
  CUTENSORNET_MARGINAL_KIND_DIAGONAL = 1
} cutensornetMarginalKind_t;

typedef enum {
  CUTENSORNET_MARGINAL_OPT_NUM_HYPER_SAMPLES = 0,
  CUTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES = 1,
  CUTENSORNET_MARGINAL_INFO_FLOPS = 64,
  CUTENSORNET_MARGINAL_INFO_KIND = 65
} cutensornetMarginalAttributes_t;

typedef enum {
  CUTENSORNET_SAMPLER_OPT_NUM_HYPER_SAMPLES = 0,
  CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES = 1,
  CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC = 2,
  CUTENSORNET_SAMPLER_INFO_FLOPS = 64
} cutensornetSamplerAttributes_t;

cutensornetStatus_t cutensornetCreateState(const cutensornetHandle_t handle, cutensornetStatePurity_t purity,
                                           int32_t numStateModes, const int64_t* stateModeExtents,
                                           cudaDataType_t dataType, cutensornetState_t* tensorNetworkState);
cutensornetStatus_t cutensornetDestroyState(cutensornetState_t tensorNetworkState);
cutensornetStatus_t cutensornetStateApplyTensor(const cutensornetHandle_t handle,
                                                cutensornetState_t tensorNetworkState, int32_t numStateModes,
                                                const int32_t* stateModes, void* tensorData,
                                                const int64_t* tensorModeStrides, const int32_t immutable,
                                                const int32_t adjoint, const int32_t unitary, int64_t* tensorId);
cutensornetStatus_t cutensornetStateApplyTensorOperator(const cutensornetHandle_t handle,
                                                        cutensornetState_t tensorNetworkState,
                                                        int32_t numStateModes, const int32_t* stateModes,
                                                        void* tensorData, const int64_t* tensorModeStrides,
                                                        const int32_t immutable, const int32_t adjoint,
                                                        const int32_t unitary, int64_t* tensorId);
cutensornetStatus_t cutensornetStateApplyTensorOperatorWithGradient(
    const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes,
    const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable,
    const int32_t adjoint, const int32_t unitary, void* gradientData, const int64_t* gradientModeStrides,
    int64_t* tensorId);
cutensornetStatus_t cutensornetStateUpdateTensorOperatorGradient(const cutensornetHandle_t handle,
                                                                 cutensornetState_t tensorNetworkState,
                                                                 int64_t tensorId, void* gradientData);
cutensornetStatus_t cutensornetStateApplyDiagonalTensorOperator(
    const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numStateModes,
    const int32_t* stateModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable,
    const int32_t adjoint, const int32_t unitary, int64_t* tensorId);
cutensornetStatus_t cutensornetStateApplyControlledTensorOperator(
    const cutensornetHandle_t handle, cutensornetState_t tensorNetworkState, int32_t numControlModes,
    const int32_t* stateControlModes, const int64_t* stateControlValues, int32_t numTargetModes,
    const int32_t* stateTargetModes, void* tensorData, const int64_t* tensorModeStrides, const int32_t immutable,
    const int32_t adjoint, const int32_t unitary, int64_t* tensorId);
cutensornetStatus_t cutensornetStateUpdateTensor(const cutensornetHandle_t handle,
                                                 cutensornetState_t tensorNetworkState, int64_t tensorId,
                                                 void* tensorData, int32_t unitary);
cutensornetStatus_t cutensornetStateUpdateTensorOperator(const cutensornetHandle_t handle,
                                                         cutensornetState_t tensorNetworkState, int64_t tensorId,
                                                         void* tensorData, int32_t unitary);
cutensornetStatus_t cutensornetStateApplyNetworkOperator(const cutensornetHandle_t handle,
                                                         cutensornetState_t tensorNetworkState,
                                                         const cutensornetNetworkOperator_t tensorNetworkOperator,
                                                         const int32_t immutable, const int32_t adjoint,
                                                         const int32_t unitary, int64_t* operatorId);
cutensornetStatus_t cutensornetStateApplyUnitaryChannel(const cutensornetHandle_t handle,
                                                        cutensornetState_t tensorNetworkState,
                                                        int32_t numStateModes, const int32_t* stateModes,
                                                        int32_t numTensors, void* tensorData[],
                                                        const int64_t* tensorModeStrides,
                                                        const double probabilities[], int64_t* channelId);
cutensornetStatus_t cutensornetStateApplyGeneralChannel(const cutensornetHandle_t handle,
                                                        cutensornetState_t tensorNetworkState,
                                                        int32_t numStateModes, const int32_t* stateModes,
                                                        int32_t numTensors, void* tensorData[],
                                                        const int64_t* tensorModeStrides, int64_t* channelId);
cutensornetStatus_t cutensornetStateInitializeMPS(const cutensornetHandle_t handle,
                                                  cutensornetState_t tensorNetworkState,
                                                  cutensornetBoundaryCondition_t boundaryCondition,
                                                  const int64_t* const extentsIn[],
                                                  const int64_t* const stridesIn[], void* stateTensorsIn[]);
cutensornetStatus_t cutensornetStateFinalizeMPS(const cutensornetHandle_t handle,
                                                cutensornetState_t tensorNetworkState,
                                                cutensornetBoundaryCondition_t boundaryCondition,
                                                const int64_t* const extentsOut[],
                                                const int64_t* const stridesOut[]);
cutensornetStatus_t cutensornetStateCaptureMPS(const cutensornetHandle_t handle,
                                               cutensornetState_t tensorNetworkState);
cutensornetStatus_t cutensornetStateConfigure(const cutensornetHandle_t handle,
                                              cutensornetState_t tensorNetworkState,
                                              cutensornetStateAttributes_t attribute, const void* attributeValue,
                                              size_t attributeSize);
cutensornetStatus_t cutensornetStatePrepare(const cutensornetHandle_t handle,
                                            cutensornetState_t tensorNetworkState, size_t maxWorkspaceSizeDevice,
                                            cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream);
cutensornetStatus_t cutensornetStateGetInfo(const cutensornetHandle_t handle,
                                            const cutensornetState_t tensorNetworkState,
                                            cutensornetStateAttributes_t attribute, void* attributeValue,
                                            size_t attributeSize);
cutensornetStatus_t cutensornetStateCompute(const cutensornetHandle_t handle,
                                            cutensornetState_t tensorNetworkState,
                                            cutensornetWorkspaceDescriptor_t workDesc, int64_t* extentsOut[],
                                            int64_t* stridesOut[], void* stateTensorsOut[],
                                            cudaStream_t cudaStream);
cutensornetStatus_t cutensornetGetOutputStateDetails(const cutensornetHandle_t handle,
                                                     const cutensornetState_t tensorNetworkState,
                                                     int32_t* numTensorsOut, int32_t numModesOut[],
                                                     int64_t* extentsOut[], int64_t* stridesOut[]);

/* Network operators: sums of products and matrix product operators */
cutensornetStatus_t cutensornetCreateNetworkOperator(const cutensornetHandle_t handle, int32_t numStateModes,
                                                     const int64_t stateModeExtents[], cudaDataType_t dataType,
                                                     cutensornetNetworkOperator_t* tensorNetworkOperator);
cutensornetStatus_t cutensornetNetworkOperatorAppendProduct(
    const cutensornetHandle_t handle, cutensornetNetworkOperator_t tensorNetworkOperator,
    cuDoubleComplex coefficient, int32_t numTensors, const int32_t numStateModes[], const int32_t* stateModes[],
    const int64_t* tensorModeStrides[], const void* tensorData[], int64_t* componentId);
cutensornetStatus_t cutensornetNetworkOperatorAppendMPO(
    const cutensornetHandle_t handle, cutensornetNetworkOperator_t tensorNetworkOperator,
    cuDoubleComplex coefficient, int32_t numStateModes, const int32_t stateModes[],
    const int64_t* tensorModeExtents[], const int64_t* tensorModeStrides[], const void* tensorData[],
    cutensornetBoundaryCondition_t boundaryCondition, int64_t* componentId);
cutensornetStatus_t cutensornetDestroyNetworkOperator(cutensornetNetworkOperator_t tensorNetworkOperator);

/* Amplitudes */
cutensornetStatus_t cutensornetCreateAccessor(const cutensornetHandle_t handle,
                                              cutensornetState_t tensorNetworkState, int32_t numProjectedModes,
                                              const int32_t* projectedModes,
                                              const int64_t* amplitudesTensorStrides,
                                              cutensornetStateAccessor_t* tensorNetworkAccessor);
cutensornetStatus_t cutensornetAccessorConfigure(const cutensornetHandle_t handle,
                                                 cutensornetStateAccessor_t tensorNetworkAccessor,
                                                 cutensornetAccessorAttributes_t attribute,
                                                 const void* attributeValue, size_t attributeSize);
cutensornetStatus_t cutensornetAccessorPrepare(const cutensornetHandle_t handle,
                                               cutensornetStateAccessor_t tensorNetworkAccessor,
                                               size_t maxWorkspaceSizeDevice,
                                               cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream);
cutensornetStatus_t cutensornetAccessorGetInfo(const cutensornetHandle_t handle,
                                               const cutensornetStateAccessor_t tensorNetworkAccessor,
                                               cutensornetAccessorAttributes_t attribute, void* attributeValue,
                                               size_t attributeSize);
cutensornetStatus_t cutensornetAccessorCompute(const cutensornetHandle_t handle,
                                               cutensornetStateAccessor_t tensorNetworkAccessor,
                                               const int64_t* projectedModeValues,
                                               cutensornetWorkspaceDescriptor_t workDesc, void* amplitudesTensor,
                                               void* stateNorm, cudaStream_t cudaStream);
cutensornetStatus_t cutensornetDestroyAccessor(cutensornetStateAccessor_t tensorNetworkAccessor);

/* Expectation values */
cutensornetStatus_t cutensornetCreateExpectation(const cutensornetHandle_t handle,
                                                 cutensornetState_t tensorNetworkState,
                                                 cutensornetNetworkOperator_t tensorNetworkOperator,
                                                 cutensornetStateExpectation_t* tensorNetworkExpectation);
cutensornetStatus_t cutensornetExpectationConfigure(const cutensornetHandle_t handle,
                                                    cutensornetStateExpectation_t tensorNetworkExpectation,
                                                    cutensornetExpectationAttributes_t attribute,
                                                    const void* attributeValue, size_t attributeSize);
cutensornetStatus_t cutensornetExpectationPrepare(const cutensornetHandle_t handle,
                                                  cutensornetStateExpectation_t tensorNetworkExpectation,
                                                  size_t maxWorkspaceSizeDevice,
                                                  cutensornetWorkspaceDescriptor_t workDesc,
                                                  cudaStream_t cudaStream);
cutensornetStatus_t cutensornetExpectationGetInfo(const cutensornetHandle_t handle,
                                                  const cutensornetStateExpectation_t tensorNetworkExpectation,
                                                  cutensornetExpectationAttributes_t attribute,
                                                  void* attributeValue, size_t attributeSize);
cutensornetStatus_t cutensornetExpectationCompute(const cutensornetHandle_t handle,
                                                  cutensornetStateExpectation_t tensorNetworkExpectation,
                                                  cutensornetWorkspaceDescriptor_t workDesc,
                                                  void* expectationValue, void* stateNorm,
                                                  cudaStream_t cudaStream);
cutensornetStatus_t cutensornetExpectationComputeWithGradientsBackward(
    const cutensornetHandle_t handle, cutensornetStateExpectation_t tensorNetworkExpectation,
    int32_t accumulateGradients, const void* expectationValueAdjoint, const void* stateNormAdjoint,
    cutensornetWorkspaceDescriptor_t workDesc, void* expectationValue, void* stateNorm, cudaStream_t cudaStream);
cutensornetStatus_t cutensornetDestroyExpectation(cutensornetStateExpectation_t tensorNetworkExpectation);

/* Marginals (reduced density matrices) and their diagonals */
cutensornetStatus_t cutensornetCreateMarginal(const cutensornetHandle_t handle,
                                              cutensornetState_t tensorNetworkState, int32_t numMarginalModes,
                                              const int32_t* marginalModes, int32_t numProjectedModes,
                                              const int32_t* projectedModes, const int64_t* marginalTensorStrides,
                                              cutensornetStateMarginal_t* tensorNetworkMarginal);
cutensornetStatus_t cutensornetCreateMarginalDiagonal(const cutensornetHandle_t handle,
                                                      cutensornetState_t tensorNetworkState,
                                                      int32_t numMarginalModes, const int32_t* marginalModes,
                                                      int32_t numProjectedModes, const int32_t* projectedModes,
                                                      const int64_t* marginalDiagonalTensorStrides,
                                                      cutensornetStateMarginal_t* tensorNetworkMarginal);
cutensornetStatus_t cutensornetMarginalConfigure(const cutensornetHandle_t handle,
                                                 cutensornetStateMarginal_t tensorNetworkMarginal,
                                                 cutensornetMarginalAttributes_t attribute,
                                                 const void* attributeValue, size_t attributeSize);
cutensornetStatus_t cutensornetMarginalPrepare(const cutensornetHandle_t handle,
                                               cutensornetStateMarginal_t tensorNetworkMarginal,
                                               size_t maxWorkspaceSizeDevice,
                                               cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream);
cutensornetStatus_t cutensornetMarginalGetInfo(const cutensornetHandle_t handle,
                                               const cutensornetStateMarginal_t tensorNetworkMarginal,
                                               cutensornetMarginalAttributes_t attribute, void* attributeValue,
                                               size_t attributeSize);
cutensornetStatus_t cutensornetMarginalCompute(const cutensornetHandle_t handle,
                                               cutensornetStateMarginal_t tensorNetworkMarginal,
                                               const int64_t* projectedModeValues,
                                               cutensornetWorkspaceDescriptor_t workDesc, void* marginalTensor,
                                               cudaStream_t cudaStream);
cutensornetStatus_t cutensornetDestroyMarginal(cutensornetStateMarginal_t tensorNetworkMarginal);

/* Samplers */
cutensornetStatus_t cutensornetCreateSampler(const cutensornetHandle_t handle,
                                             cutensornetState_t tensorNetworkState, int32_t numModesToSample,
                                             const int32_t* modesToSample,
                                             cutensornetStateSampler_t* tensorNetworkSampler);
cutensornetStatus_t cutensornetSamplerConfigure(const cutensornetHandle_t handle,
                                                cutensornetStateSampler_t tensorNetworkSampler,
                                                cutensornetSamplerAttributes_t attribute,
                                                const void* attributeValue, size_t attributeSize);
cutensornetStatus_t cutensornetSamplerPrepare(const cutensornetHandle_t handle,
                                              cutensornetStateSampler_t tensorNetworkSampler,
                                              size_t maxWorkspaceSizeDevice,
                                              cutensornetWorkspaceDescriptor_t workDesc, cudaStream_t cudaStream);
cutensornetStatus_t cutensornetSamplerGetInfo(const cutensornetHandle_t handle,
                                              const cutensornetStateSampler_t tensorNetworkSampler,
                                              cutensornetSamplerAttributes_t attribute, void* attributeValue,
                                              size_t attributeSize);
cutensornetStatus_t cutensornetSamplerSample(const cutensornetHandle_t handle,
                                             cutensornetStateSampler_t tensorNetworkSampler, int64_t numShots,
                                             cutensornetWorkspaceDescriptor_t workDesc, int64_t* samples,
                                             cudaStream_t cudaStream);
cutensornetStatus_t cutensornetDestroySampler(cutensornetStateSampler_t tensorNetworkSampler);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_CUTENSORNET_H_ */
