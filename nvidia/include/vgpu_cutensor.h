/*
 * vgpu_cutensor.h -- VirtualGPU's clean-room declarations of the cuTENSOR 2 API.
 *
 * Written from NVIDIA's publicly documented cuTENSOR API
 * (docs.nvidia.com/cuda/cutensor), version 2.8, so that programs built against
 * cuTENSOR -- tensor contractions, reductions, permutations and elementwise
 * operations -- can run on VirtualGPU's libcutensor. It contains no NVIDIA
 * code. Type layouts and numeric values (status codes, operators, attributes)
 * follow the documented ABI, so a program compiled with NVIDIA's cutensor.h
 * and one compiled with this header call the same library the same way.
 *
 * Why the simulator needs its own: NVIDIA's libcutensor carries a statically
 * linked CUDA runtime, which reaches the driver through NVIDIA's undocumented
 * internal interface, so it cannot run on a simulated driver.
 *
 * Every documented entry point is declared: the handle and its plan cache,
 * tensor and block-sparse tensor descriptors, the six kinds of operation,
 * plan preferences, plans, workspace estimation and the logger.
 */
#ifndef VGPU_CUTENSOR_H_
#define VGPU_CUTENSOR_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h> /* FILE, for cutensorLoggerSetFile */

#include <driver_types.h>  /* cudaStream_t */
#include <library_types.h> /* cudaDataType_t */

#define CUTENSOR_MAJOR 2
#define CUTENSOR_MINOR 8
#define CUTENSOR_PATCH 0
#define CUTENSOR_VERSION (CUTENSOR_MAJOR * 10000 + CUTENSOR_MINOR * 100 + CUTENSOR_PATCH)

#ifdef __cplusplus
extern "C" {
#endif

/* Element types are the runtime's cudaDataType_t. */
typedef cudaDataType_t cutensorDataType_t;
#define CUTENSOR_R_16F CUDA_R_16F
#define CUTENSOR_C_16F CUDA_C_16F
#define CUTENSOR_R_16BF CUDA_R_16BF
#define CUTENSOR_C_16BF CUDA_C_16BF
#define CUTENSOR_R_32F CUDA_R_32F
#define CUTENSOR_C_32F CUDA_C_32F
#define CUTENSOR_R_64F CUDA_R_64F
#define CUTENSOR_C_64F CUDA_C_64F
#define CUTENSOR_R_8I CUDA_R_8I
#define CUTENSOR_C_8I CUDA_C_8I
#define CUTENSOR_R_8U CUDA_R_8U
#define CUTENSOR_C_8U CUDA_C_8U
#define CUTENSOR_R_32I CUDA_R_32I
#define CUTENSOR_C_32I CUDA_C_32I
#define CUTENSOR_R_32U CUDA_R_32U
#define CUTENSOR_C_32U CUDA_C_32U

typedef enum {
  /* unary */
  CUTENSOR_OP_IDENTITY = 1,
  CUTENSOR_OP_SQRT = 2,
  CUTENSOR_OP_RELU = 8,
  CUTENSOR_OP_CONJ = 9,
  CUTENSOR_OP_RCP = 10,
  CUTENSOR_OP_SIGMOID = 11,
  CUTENSOR_OP_TANH = 12,
  CUTENSOR_OP_EXP = 22,
  CUTENSOR_OP_LOG = 23,
  CUTENSOR_OP_ABS = 24,
  CUTENSOR_OP_NEG = 25,
  CUTENSOR_OP_SIN = 26,
  CUTENSOR_OP_COS = 27,
  CUTENSOR_OP_TAN = 28,
  CUTENSOR_OP_SINH = 29,
  CUTENSOR_OP_COSH = 30,
  CUTENSOR_OP_ASIN = 31,
  CUTENSOR_OP_ACOS = 32,
  CUTENSOR_OP_ATAN = 33,
  CUTENSOR_OP_ASINH = 34,
  CUTENSOR_OP_ACOSH = 35,
  CUTENSOR_OP_ATANH = 36,
  CUTENSOR_OP_CEIL = 37,
  CUTENSOR_OP_FLOOR = 38,
  CUTENSOR_OP_MISH = 39,
  CUTENSOR_OP_SWISH = 40,
  CUTENSOR_OP_SOFT_PLUS = 41,
  CUTENSOR_OP_SOFT_SIGN = 42,
  /* binary */
  CUTENSOR_OP_ADD = 3,
  CUTENSOR_OP_MUL = 5,
  CUTENSOR_OP_MAX = 6,
  CUTENSOR_OP_MIN = 7,
  CUTENSOR_OP_UNKNOWN = 126
} cutensorOperator_t;

typedef enum {
  CUTENSOR_STATUS_SUCCESS = 0,
  CUTENSOR_STATUS_NOT_INITIALIZED = 1,
  CUTENSOR_STATUS_ALLOC_FAILED = 3,
  CUTENSOR_STATUS_INVALID_VALUE = 7,
  CUTENSOR_STATUS_ARCH_MISMATCH = 8,
  CUTENSOR_STATUS_MAPPING_ERROR = 11,
  CUTENSOR_STATUS_EXECUTION_FAILED = 13,
  CUTENSOR_STATUS_INTERNAL_ERROR = 14,
  CUTENSOR_STATUS_NOT_SUPPORTED = 15,
  CUTENSOR_STATUS_LICENSE_ERROR = 16,
  CUTENSOR_STATUS_CUBLAS_ERROR = 17,
  CUTENSOR_STATUS_CUDA_ERROR = 18,
  CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
  CUTENSOR_STATUS_INSUFFICIENT_DRIVER = 20,
  CUTENSOR_STATUS_IO_ERROR = 21
} cutensorStatus_t;

typedef enum {
  CUTENSOR_ALGO_DEFAULT_PATIENT = -6,
  CUTENSOR_ALGO_GETT = -4,
  CUTENSOR_ALGO_TGETT = -3,
  CUTENSOR_ALGO_TTGT = -2,
  CUTENSOR_ALGO_DEFAULT = -1
} cutensorAlgo_t;

typedef enum {
  CUTENSOR_WORKSPACE_MIN = 1,
  CUTENSOR_WORKSPACE_DEFAULT = 2,
  CUTENSOR_WORKSPACE_MAX = 3
} cutensorWorksizePreference_t;

typedef struct cutensorComputeDescriptor* cutensorComputeDescriptor_t;

extern const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_16F;
extern const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_16BF;
extern const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_TF32;
extern const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_3XTF32;
extern const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_32F;
extern const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_64F;
extern const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_9X16BF;
extern const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_8XINT8;
extern const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_4X16F;

typedef enum {
  CUTENSOR_OPERATION_DESCRIPTOR_TAG = 0,         /* int32_t */
  CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE = 1, /* cudaDataType_t */
  CUTENSOR_OPERATION_DESCRIPTOR_FLOPS = 2,       /* float */
  CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES = 3, /* float */
  CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT = 4,  /* uint32_t[numModes of the output] */
  CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT = 5, /* uint32_t[numModes of the output] */
  CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE = 6, /* one element of the output's type */
  CUTENSOR_OPERATION_DESCRIPTOR_BLOCKSPARSE_REPRODUCIBLE = 14 /* int32_t */
} cutensorOperationDescriptorAttribute_t;

typedef enum {
  CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE = 0,
  CUTENSOR_PLAN_PREFERENCE_CACHE_MODE = 1,
  CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT = 2,
  CUTENSOR_PLAN_PREFERENCE_ALGO = 3,
  CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK = 4,
  CUTENSOR_PLAN_PREFERENCE_JIT = 5,
  CUTENSOR_PLAN_PREFERENCE_GPU_ARCH = 6
} cutensorPlanPreferenceAttribute_t;

typedef enum {
  CUTENSOR_AUTOTUNE_MODE_NONE = 0,
  CUTENSOR_AUTOTUNE_MODE_INCREMENTAL = 1
} cutensorAutotuneMode_t;

typedef enum {
  CUTENSOR_JIT_MODE_NONE = 0,
  CUTENSOR_JIT_MODE_DEFAULT = 1
} cutensorJitMode_t;

typedef enum {
  CUTENSOR_CACHE_MODE_NONE = 0,
  CUTENSOR_CACHE_MODE_PEDANTIC = 1
} cutensorCacheMode_t;

typedef enum {
  CUTENSOR_PLAN_REQUIRED_WORKSPACE = 0 /* uint64_t */
} cutensorPlanAttribute_t;

typedef struct cutensorOperationDescriptor* cutensorOperationDescriptor_t;
typedef struct cutensorPlan* cutensorPlan_t;
typedef struct cutensorPlanPreference* cutensorPlanPreference_t;
typedef struct cutensorHandle* cutensorHandle_t;
typedef struct cutensorTensorDescriptor* cutensorTensorDescriptor_t;
typedef struct cutensorBlockSparseTensorDescriptor* cutensorBlockSparseTensorDescriptor_t;

typedef void (*cutensorLoggerCallback_t)(int32_t logLevel, const char* functionName,
                                         const char* message);

/* Handle and caches */
cutensorStatus_t cutensorCreate(cutensorHandle_t* handle);
cutensorStatus_t cutensorDestroy(cutensorHandle_t handle);
cutensorStatus_t cutensorHandleResizePlanCache(cutensorHandle_t handle, const uint32_t numEntries);
cutensorStatus_t cutensorHandleWritePlanCacheToFile(const cutensorHandle_t handle, const char filename[]);
cutensorStatus_t cutensorHandleReadPlanCacheFromFile(cutensorHandle_t handle, const char filename[],
                                                     uint32_t* numCachelinesRead);
cutensorStatus_t cutensorWriteKernelCacheToFile(const cutensorHandle_t handle, const char filename[]);
cutensorStatus_t cutensorReadKernelCacheFromFile(cutensorHandle_t handle, const char filename[]);

/* Tensor descriptors */
cutensorStatus_t cutensorCreateTensorDescriptor(const cutensorHandle_t handle,
                                                cutensorTensorDescriptor_t* desc, const uint32_t numModes,
                                                const int64_t extent[], const int64_t stride[],
                                                cudaDataType_t dataType, uint32_t alignmentRequirement);
cutensorStatus_t cutensorDestroyTensorDescriptor(cutensorTensorDescriptor_t desc);

/* Elementwise operations */
cutensorStatus_t cutensorCreateElementwiseTrinary(
    const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc,
    const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA,
    const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB,
    const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC,
    const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAB,
    cutensorOperator_t opABC, const cutensorComputeDescriptor_t descCompute);
cutensorStatus_t cutensorElementwiseTrinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan,
                                                   const void* alpha, const void* A, const void* beta,
                                                   const void* B, const void* gamma, const void* C, void* D,
                                                   cudaStream_t stream);
cutensorStatus_t cutensorCreateElementwiseBinary(
    const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc,
    const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA,
    const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC,
    const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAC,
    const cutensorComputeDescriptor_t descCompute);
cutensorStatus_t cutensorElementwiseBinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan,
                                                  const void* alpha, const void* A, const void* gamma,
                                                  const void* C, void* D, cudaStream_t stream);
cutensorStatus_t cutensorCreatePermutation(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc,
                                           const cutensorTensorDescriptor_t descA, const int32_t modeA[],
                                           cutensorOperator_t opA, const cutensorTensorDescriptor_t descB,
                                           const int32_t modeB[], const cutensorComputeDescriptor_t descCompute);
cutensorStatus_t cutensorPermute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha,
                                 const void* A, void* B, const cudaStream_t stream);

/* Contractions and reductions */
cutensorStatus_t cutensorCreateContraction(
    const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc,
    const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA,
    const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB,
    const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC,
    const cutensorTensorDescriptor_t descD, const int32_t modeD[],
    const cutensorComputeDescriptor_t descCompute);
cutensorStatus_t cutensorDestroyOperationDescriptor(cutensorOperationDescriptor_t desc);
cutensorStatus_t cutensorOperationDescriptorSetAttribute(const cutensorHandle_t handle,
                                                         cutensorOperationDescriptor_t desc,
                                                         cutensorOperationDescriptorAttribute_t attr,
                                                         const void* buf, size_t sizeInBytes);
cutensorStatus_t cutensorOperationDescriptorGetAttribute(const cutensorHandle_t handle,
                                                         cutensorOperationDescriptor_t desc,
                                                         cutensorOperationDescriptorAttribute_t attr, void* buf,
                                                         size_t sizeInBytes);
cutensorStatus_t cutensorCreatePlanPreference(const cutensorHandle_t handle, cutensorPlanPreference_t* pref,
                                              cutensorAlgo_t algo, cutensorJitMode_t jitMode);
cutensorStatus_t cutensorDestroyPlanPreference(cutensorPlanPreference_t pref);
cutensorStatus_t cutensorPlanPreferenceGetAttribute(const cutensorHandle_t handle, cutensorPlanPreference_t pref,
                                                    cutensorPlanPreferenceAttribute_t attr, void* buf,
                                                    size_t sizeInBytes);
cutensorStatus_t cutensorPlanPreferenceSetAttribute(const cutensorHandle_t handle, cutensorPlanPreference_t pref,
                                                    cutensorPlanPreferenceAttribute_t attr, const void* buf,
                                                    size_t sizeInBytes);
cutensorStatus_t cutensorPlanGetAttribute(const cutensorHandle_t handle, const cutensorPlan_t plan,
                                          cutensorPlanAttribute_t attr, void* buf, size_t sizeInBytes);
cutensorStatus_t cutensorEstimateWorkspaceSize(const cutensorHandle_t handle,
                                               const cutensorOperationDescriptor_t desc,
                                               const cutensorPlanPreference_t planPref,
                                               const cutensorWorksizePreference_t workspacePref,
                                               uint64_t* workspaceSizeEstimate);
cutensorStatus_t cutensorCreatePlan(const cutensorHandle_t handle, cutensorPlan_t* plan,
                                    const cutensorOperationDescriptor_t desc, const cutensorPlanPreference_t pref,
                                    uint64_t workspaceSizeLimit);
cutensorStatus_t cutensorDestroyPlan(cutensorPlan_t plan);
cutensorStatus_t cutensorContract(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha,
                                  const void* A, const void* B, const void* beta, const void* C, void* D,
                                  void* workspace, uint64_t workspaceSize, cudaStream_t stream);
cutensorStatus_t cutensorCreateReduction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc,
                                         const cutensorTensorDescriptor_t descA, const int32_t modeA[],
                                         cutensorOperator_t opA, const cutensorTensorDescriptor_t descC,
                                         const int32_t modeC[], cutensorOperator_t opC,
                                         const cutensorTensorDescriptor_t descD, const int32_t modeD[],
                                         cutensorOperator_t opReduce, const cutensorComputeDescriptor_t descCompute);
cutensorStatus_t cutensorReduce(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha,
                                const void* A, const void* beta, const void* C, void* D, void* workspace,
                                uint64_t workspaceSize, cudaStream_t stream);
cutensorStatus_t cutensorCreateContractionTrinary(
    const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc,
    const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA,
    const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB,
    const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC,
    const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opD,
    const cutensorTensorDescriptor_t descE, const int32_t modeE[],
    const cutensorComputeDescriptor_t descCompute);
cutensorStatus_t cutensorContractTrinary(const cutensorHandle_t handle, const cutensorPlan_t plan,
                                         const void* alpha, const void* A, const void* B, const void* C,
                                         const void* beta, const void* D, void* E, void* workspace,
                                         uint64_t workspaceSize, cudaStream_t stream);

/* Block-sparse contractions */
cutensorStatus_t cutensorCreateBlockSparseTensorDescriptor(
    cutensorHandle_t handle, cutensorBlockSparseTensorDescriptor_t* desc, const uint32_t numModes,
    const uint64_t numNonZeroBlocks, const uint32_t numSectionsPerMode[], const int64_t extent[],
    const int32_t nonZeroCoordinates[], const int64_t stride[], cudaDataType_t dataType);
cutensorStatus_t cutensorDestroyBlockSparseTensorDescriptor(cutensorBlockSparseTensorDescriptor_t desc);
cutensorStatus_t cutensorCreateBlockSparseContraction(
    const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc,
    const cutensorBlockSparseTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA,
    const cutensorBlockSparseTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB,
    const cutensorBlockSparseTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC,
    const cutensorBlockSparseTensorDescriptor_t descD, const int32_t modeD[],
    const cutensorComputeDescriptor_t descCompute);
cutensorStatus_t cutensorBlockSparseContract(const cutensorHandle_t handle, const cutensorPlan_t plan,
                                             const void* alpha, const void* const A[], const void* const B[],
                                             const void* beta, const void* const C[], void* const D[],
                                             void* workspace, uint64_t workspaceSize, cudaStream_t stream);

/* Version, errors and logging */
const char* cutensorGetErrorString(const cutensorStatus_t error);
size_t cutensorGetVersion(void);
size_t cutensorGetCudartVersion(void);
cutensorStatus_t cutensorLoggerSetCallback(cutensorLoggerCallback_t callback);
cutensorStatus_t cutensorLoggerSetFile(FILE* file);
cutensorStatus_t cutensorLoggerOpenFile(const char* logFile);
cutensorStatus_t cutensorLoggerSetLevel(int32_t level);
cutensorStatus_t cutensorLoggerSetMask(int32_t mask);
cutensorStatus_t cutensorLoggerForceDisable(void);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_CUTENSOR_H_ */
