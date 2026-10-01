/*
 * vgpu_custatevec.h -- VirtualGPU's clean-room subset of the cuStateVec API.
 *
 * Written from NVIDIA's publicly documented cuStateVec API
 * (docs.nvidia.com/cuda/cuquantum/latest/custatevec) so that programs built
 * against cuQuantum can run on VirtualGPU's libcustatevec. It contains no
 * NVIDIA code. Type layouts and numeric values (status codes, enums) follow
 * the documented ABI, so a program compiled with NVIDIA's custatevec.h and
 * one compiled with this header call the same library the same way.
 *
 * Why the simulator needs its own: NVIDIA's libcustatevec carries a
 * statically linked CUDA runtime, which reaches the driver through NVIDIA's
 * undocumented internal interface, so it cannot run on a simulated driver.
 *
 * Only the subset VirtualGPU implements is declared: what QuEST's cuQuantum
 * backend calls, and the small utility calls around it.
 */
#ifndef VGPU_CUSTATEVEC_H_
#define VGPU_CUSTATEVEC_H_

#include <stddef.h>
#include <stdint.h>

#include <driver_types.h>   /* cudaStream_t */
#include <library_types.h>  /* cudaDataType_t, libraryPropertyType */
#include <vector_types.h>   /* int2 */

#ifdef __cplusplus
extern "C" {
#endif

#define CUSTATEVEC_VER_MAJOR 1
#define CUSTATEVEC_VER_MINOR 15
#define CUSTATEVEC_VER_PATCH 0
#define CUSTATEVEC_VERSION \
  (CUSTATEVEC_VER_MAJOR * 10000 + CUSTATEVEC_VER_MINOR * 100 + CUSTATEVEC_VER_PATCH)

#define CUSTATEVEC_ALLOCATOR_NAME_LEN 64

typedef int64_t custatevecIndex_t;
typedef struct custatevecContext* custatevecHandle_t;

typedef enum custatevecStatus_t {
  CUSTATEVEC_STATUS_SUCCESS = 0,
  CUSTATEVEC_STATUS_NOT_INITIALIZED = 1,
  CUSTATEVEC_STATUS_ALLOC_FAILED = 2,
  CUSTATEVEC_STATUS_INVALID_VALUE = 3,
  CUSTATEVEC_STATUS_ARCH_MISMATCH = 4,
  CUSTATEVEC_STATUS_EXECUTION_FAILED = 5,
  CUSTATEVEC_STATUS_INTERNAL_ERROR = 6,
  CUSTATEVEC_STATUS_NOT_SUPPORTED = 7,
  CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE = 8,
  CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED = 9,
  CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR = 10,
  CUSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR = 11,
  CUSTATEVEC_STATUS_COMMUNICATOR_ERROR = 12,
  CUSTATEVEC_STATUS_LOADING_LIBRARY_FAILED = 13,
  CUSTATEVEC_STATUS_INVALID_CONFIGURATION = 14,
  CUSTATEVEC_STATUS_ALREADY_INITIALIZED = 15,
  CUSTATEVEC_STATUS_INVALID_WIRE = 16,
  CUSTATEVEC_STATUS_SYSTEM_ERROR = 17,
  CUSTATEVEC_STATUS_CUDA_ERROR = 18,
  CUSTATEVEC_STATUS_NUMERICAL_ERROR = 19,
  CUSTATEVEC_STATUS_RESOURCES_NOT_ACCESSIBLE = 20,
  CUSTATEVEC_STATUS_MAX_VALUE = 21
} custatevecStatus_t;

typedef enum custatevecPauli_t {
  CUSTATEVEC_PAULI_I = 0,
  CUSTATEVEC_PAULI_X = 1,
  CUSTATEVEC_PAULI_Y = 2,
  CUSTATEVEC_PAULI_Z = 3
} custatevecPauli_t;

typedef enum custatevecMatrixLayout_t {
  CUSTATEVEC_MATRIX_LAYOUT_COL = 0,
  CUSTATEVEC_MATRIX_LAYOUT_ROW = 1
} custatevecMatrixLayout_t;

typedef enum custatevecComputeType_t {
  CUSTATEVEC_COMPUTE_DEFAULT = 0,
  CUSTATEVEC_COMPUTE_32F = (1U << 2U),
  CUSTATEVEC_COMPUTE_64F = (1U << 4U),
  CUSTATEVEC_COMPUTE_TF32 = (1U << 12U)
} custatevecComputeType_t;

typedef struct {
  void* ctx;
  int (*device_alloc)(void* ctx, void** ptr, size_t size, cudaStream_t stream);
  int (*device_free)(void* ctx, void* ptr, size_t size, cudaStream_t stream);
  char name[CUSTATEVEC_ALLOCATOR_NAME_LEN];
} custatevecDeviceMemHandler_t;

/* Library management */
custatevecStatus_t custatevecCreate(custatevecHandle_t* handle);
custatevecStatus_t custatevecDestroy(custatevecHandle_t handle);
const char* custatevecGetErrorName(custatevecStatus_t status);
const char* custatevecGetErrorString(custatevecStatus_t status);
custatevecStatus_t custatevecGetProperty(libraryPropertyType type, int32_t* value);
size_t custatevecGetVersion(void);
custatevecStatus_t custatevecSetStream(custatevecHandle_t handle, cudaStream_t streamId);
custatevecStatus_t custatevecGetStream(custatevecHandle_t handle, cudaStream_t* streamId);
custatevecStatus_t custatevecGetDeviceMemHandler(custatevecHandle_t handle,
                                                 custatevecDeviceMemHandler_t* handler);
custatevecStatus_t custatevecSetDeviceMemHandler(custatevecHandle_t handle,
                                                 const custatevecDeviceMemHandler_t* handler);

/* Gates */
custatevecStatus_t custatevecApplyMatrixGetWorkspaceSize(
    custatevecHandle_t handle, cudaDataType_t svDataType, const uint32_t nIndexBits,
    const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout,
    const int32_t adjoint, const uint32_t nTargets, const uint32_t nControls,
    custatevecComputeType_t computeType, size_t* extraWorkspaceSizeInBytes);

custatevecStatus_t custatevecApplyMatrix(
    custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits,
    const void* matrix, cudaDataType_t matrixDataType, custatevecMatrixLayout_t layout,
    const int32_t adjoint, const int32_t* targets, const uint32_t nTargets,
    const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls,
    custatevecComputeType_t computeType, void* extraWorkspace, size_t extraWorkspaceSizeInBytes);

custatevecStatus_t custatevecApplyGeneralizedPermutationMatrix(
    custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits,
    custatevecIndex_t* permutation, const void* diagonals, cudaDataType_t diagonalsDataType,
    const int32_t adjoint, const int32_t* targets, const uint32_t nTargets,
    const int32_t* controls, const int32_t* controlBitValues, const uint32_t nControls,
    void* extraWorkspace, size_t extraWorkspaceSizeInBytes);

custatevecStatus_t custatevecSwapIndexBits(
    custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits,
    const int2* bitSwaps, const uint32_t nBitSwaps, const int32_t* maskBitString,
    const int32_t* maskOrdering, const uint32_t maskLen);

/* Measurement and probabilities */
custatevecStatus_t custatevecAbs2SumArray(
    custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType,
    const uint32_t nIndexBits, double* abs2sum, const int32_t* bitOrdering,
    const uint32_t bitOrderingLen, const int32_t* maskBitString, const int32_t* maskOrdering,
    const uint32_t maskLen);

custatevecStatus_t custatevecAbs2SumOnZBasis(
    custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType,
    const uint32_t nIndexBits, double* abs2sum0, double* abs2sum1, const int32_t* basisBits,
    const uint32_t nBasisBits);

custatevecStatus_t custatevecCollapseByBitString(
    custatevecHandle_t handle, void* sv, cudaDataType_t svDataType, const uint32_t nIndexBits,
    const int32_t* bitString, const int32_t* bitOrdering, const uint32_t bitStringLen,
    double norm);

/* Expectation values */
custatevecStatus_t custatevecComputeExpectationsOnPauliBasis(
    custatevecHandle_t handle, const void* sv, cudaDataType_t svDataType,
    const uint32_t nIndexBits, double* expectationValues,
    const custatevecPauli_t** pauliOperatorsArray, const uint32_t nPauliOperatorArrays,
    const int32_t** basisBitsArray, const uint32_t* nBasisBitsArray);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_CUSTATEVEC_H_ */
