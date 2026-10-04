/*
 * vgpu_cusparselt.h -- VirtualGPU's clean-room declarations of the cuSPARSELt
 * API.
 *
 * Written from NVIDIA's publicly documented cuSPARSELt API
 * (docs.nvidia.com/cuda/cusparselt), version 0.10, so that programs built
 * against cuSPARSELt -- matrix multiplication with a 2:4 structured sparse
 * operand -- can run on VirtualGPU's libcusparseLt. It contains no NVIDIA
 * code. Type layouts and numeric values (the 512-byte opaque objects aligned
 * to 16 bytes, enum values, signatures) follow the documented ABI, so a
 * program compiled with NVIDIA's cusparseLt.h and one compiled with this
 * header call the same library the same way. The status, operation and order
 * types are cuSPARSE's, from the toolkit's cusparse.h.
 *
 * Why the simulator needs its own: NVIDIA's libcusparseLt carries a
 * statically linked CUDA runtime, which reaches the driver through NVIDIA's
 * undocumented internal interface, so it cannot run on a simulated driver.
 *
 * The whole 0.10 API is declared.
 */
#ifndef VGPU_CUSPARSELT_H_
#define VGPU_CUSPARSELT_H_

#include <stddef.h>
#include <stdint.h>

#include <cusparse.h>      /* cusparseStatus_t, cusparseOperation_t, cusparseOrder_t */
#include <driver_types.h>  /* cudaStream_t */
#include <library_types.h> /* cudaDataType, libraryPropertyType */

#define CUSPARSELT_VER_MAJOR 0
#define CUSPARSELT_VER_MINOR 10
#define CUSPARSELT_VER_PATCH 0
#define CUSPARSELT_VER_BUILD 0
/* 1000 for 0.10.0: major in the millions, minor in the hundreds, patch in the
 * units. For every 0.x.y release this is the same number the older
 * major*1000 encoding gave (0.8.1 is 801). */
#define CUSPARSELT_VERSION \
  (CUSPARSELT_VER_MAJOR * 1000000 + CUSPARSELT_VER_MINOR * 100 + CUSPARSELT_VER_PATCH)

#ifndef CUSPARSELT_API
#define CUSPARSELT_API
#endif

/* The opaque objects are caller-allocated: 512 bytes each, aligned to 16. */
#if defined(__cplusplus)
#define VGPU_CUSPARSELT_OPAQUE(name) \
  typedef struct { alignas(16) uint8_t data[512]; } name
#else
#define VGPU_CUSPARSELT_OPAQUE(name) \
  typedef struct { _Alignas(16) uint8_t data[512]; } name
#endif

#ifdef __cplusplus
extern "C" {
#endif

VGPU_CUSPARSELT_OPAQUE(cusparseLtHandle_t);
VGPU_CUSPARSELT_OPAQUE(cusparseLtMatDescriptor_t);
VGPU_CUSPARSELT_OPAQUE(cusparseLtMatmulDescriptor_t);
VGPU_CUSPARSELT_OPAQUE(cusparseLtMatmulAlgSelection_t);
VGPU_CUSPARSELT_OPAQUE(cusparseLtMatmulPlan_t);

typedef enum {
  CUSPARSELT_SPARSITY_50_PERCENT = 0 /* 2:4 (1:2 for 32-bit values) */
} cusparseLtSparsity_t;

typedef enum {
  CUSPARSELT_MAT_NUM_BATCHES = 0, /* int */
  CUSPARSELT_MAT_BATCH_STRIDE = 1 /* int64_t, in elements; 0 broadcasts one matrix */
} cusparseLtMatDescAttribute_t;

typedef enum {
  CUSPARSE_COMPUTE_32I = 0,
  CUSPARSE_COMPUTE_16F = 1,
  CUSPARSE_COMPUTE_32F = 2
} cusparseComputeType;

typedef enum {
  CUSPARSELT_MATMUL_ACTIVATION_RELU = 0,            /* int, 0 or 1 */
  CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND = 1, /* float */
  CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD = 2,  /* float */
  CUSPARSELT_MATMUL_ACTIVATION_GELU = 3,            /* int, 0 or 1 */
  CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING = 4,    /* float */
  CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING = 5,       /* int, 0 or 1 */
  CUSPARSELT_MATMUL_BETA_VECTOR_SCALING = 6,        /* int, 0 or 1 */
  CUSPARSELT_MATMUL_BIAS_STRIDE = 7,                /* int64_t */
  CUSPARSELT_MATMUL_BIAS_POINTER = 8,               /* void* (device) */
  CUSPARSELT_MATMUL_SPARSE_MAT_POINTER = 9,         /* void* (device) */
  CUSPARSELT_MATMUL_A_SCALE_MODE = 10,              /* block scaling (fp8/fp4) */
  CUSPARSELT_MATMUL_B_SCALE_MODE = 11,
  CUSPARSELT_MATMUL_C_SCALE_MODE = 12,
  CUSPARSELT_MATMUL_D_SCALE_MODE = 13,
  CUSPARSELT_MATMUL_D_OUT_SCALE_MODE = 14,
  CUSPARSELT_MATMUL_A_SCALE_POINTER = 15,
  CUSPARSELT_MATMUL_B_SCALE_POINTER = 16,
  CUSPARSELT_MATMUL_C_SCALE_POINTER = 17,
  CUSPARSELT_MATMUL_D_SCALE_POINTER = 18,
  CUSPARSELT_MATMUL_D_OUT_SCALE_POINTER = 19
} cusparseLtMatmulDescAttribute_t;

typedef enum {
  CUSPARSELT_MATMUL_SCALE_NONE = 0,
  CUSPARSELT_MATMUL_MATRIX_SCALE_SCALAR_32F = 1,
  CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3 = 2,
  CUSPARSELT_MATMUL_MATRIX_SCALE_VEC64_UE8M0 = 3
} cusparseLtMatmulMatrixScale_t;

typedef enum {
  CUSPARSELT_MATMUL_ALG_DEFAULT = 0
} cusparseLtMatmulAlg_t;

typedef enum {
  CUSPARSELT_MATMUL_ALG_CONFIG_ID = 0,     /* int */
  CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID = 1, /* int, read-only */
  CUSPARSELT_MATMUL_SEARCH_ITERATIONS = 2, /* int */
  CUSPARSELT_MATMUL_SPLIT_K = 3,           /* int */
  CUSPARSELT_MATMUL_SPLIT_K_MODE = 4,      /* int (cusparseLtSplitKMode_t) */
  CUSPARSELT_MATMUL_SPLIT_K_BUFFERS = 5    /* int */
} cusparseLtMatmulAlgAttribute_t;

typedef enum {
  CUSPARSELT_INVALID_MODE = 0,
  CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL = 1,
  CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS = 2,
  CUSPARSELT_HEURISTIC = 3,
  CUSPARSELT_DATAPARALLEL = 4,
  CUSPARSELT_SPLITK = 5,
  CUSPARSELT_STREAMK = 6
} cusparseLtSplitKMode_t;

typedef enum {
  CUSPARSELT_PRUNE_SPMMA_TILE = 0, /* 2:4 along both dimensions of each 4x4 tile */
  CUSPARSELT_PRUNE_SPMMA_STRIP = 1 /* 2:4 along the reduction dimension only */
} cusparseLtPruneAlg_t;

/* Library */
cusparseStatus_t CUSPARSELT_API cusparseLtInit(cusparseLtHandle_t* handle);
cusparseStatus_t CUSPARSELT_API cusparseLtDestroy(const cusparseLtHandle_t* handle);
cusparseStatus_t CUSPARSELT_API cusparseLtGetVersion(const cusparseLtHandle_t* handle, int* version);
cusparseStatus_t CUSPARSELT_API cusparseLtGetProperty(libraryPropertyType propertyType, int* value);
const char* CUSPARSELT_API cusparseLtGetErrorName(cusparseStatus_t status);
const char* CUSPARSELT_API cusparseLtGetErrorString(cusparseStatus_t status);

/* Matrix descriptors */
cusparseStatus_t CUSPARSELT_API cusparseLtDenseDescriptorInit(
    const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols,
    int64_t ld, uint32_t alignment, cudaDataType valueType, cusparseOrder_t order);
cusparseStatus_t CUSPARSELT_API cusparseLtStructuredDescriptorInit(
    const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols,
    int64_t ld, uint32_t alignment, cudaDataType valueType, cusparseOrder_t order,
    cusparseLtSparsity_t sparsity);
cusparseStatus_t CUSPARSELT_API cusparseLtMatDescriptorDestroy(const cusparseLtMatDescriptor_t* matDescr);
cusparseStatus_t CUSPARSELT_API cusparseLtMatDescSetAttribute(
    const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matmulDescr,
    cusparseLtMatDescAttribute_t matAttribute, const void* data, size_t dataSize);
cusparseStatus_t CUSPARSELT_API cusparseLtMatDescGetAttribute(
    const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* matmulDescr,
    cusparseLtMatDescAttribute_t matAttribute, void* data, size_t dataSize);

/* Matmul descriptor */
cusparseStatus_t CUSPARSELT_API cusparseLtMatmulDescriptorInit(
    const cusparseLtHandle_t* handle, cusparseLtMatmulDescriptor_t* matmulDescr, cusparseOperation_t opA,
    cusparseOperation_t opB, const cusparseLtMatDescriptor_t* matA, const cusparseLtMatDescriptor_t* matB,
    const cusparseLtMatDescriptor_t* matC, const cusparseLtMatDescriptor_t* matD,
    cusparseComputeType computeType);
cusparseStatus_t CUSPARSELT_API cusparseLtMatmulDescSetAttribute(
    const cusparseLtHandle_t* handle, cusparseLtMatmulDescriptor_t* matmulDescr,
    cusparseLtMatmulDescAttribute_t matmulAttribute, const void* data, size_t dataSize);
cusparseStatus_t CUSPARSELT_API cusparseLtMatmulDescGetAttribute(
    const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr,
    cusparseLtMatmulDescAttribute_t matmulAttribute, void* data, size_t dataSize);

/* Algorithm selection */
cusparseStatus_t CUSPARSELT_API cusparseLtMatmulAlgSelectionInit(
    const cusparseLtHandle_t* handle, cusparseLtMatmulAlgSelection_t* algSelection,
    const cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulAlg_t alg);
cusparseStatus_t CUSPARSELT_API cusparseLtMatmulAlgSelectionDestroy(
    const cusparseLtMatmulAlgSelection_t* algSelection);
cusparseStatus_t CUSPARSELT_API cusparseLtMatmulAlgSetAttribute(
    const cusparseLtHandle_t* handle, cusparseLtMatmulAlgSelection_t* algSelection,
    cusparseLtMatmulAlgAttribute_t attribute, const void* data, size_t dataSize);
cusparseStatus_t CUSPARSELT_API cusparseLtMatmulAlgGetAttribute(
    const cusparseLtHandle_t* handle, const cusparseLtMatmulAlgSelection_t* algSelection,
    cusparseLtMatmulAlgAttribute_t attribute, void* data, size_t dataSize);

/* Plan */
cusparseStatus_t CUSPARSELT_API cusparseLtMatmulGetWorkspace(const cusparseLtHandle_t* handle,
                                                             const cusparseLtMatmulPlan_t* plan,
                                                             size_t* workspaceSize);
cusparseStatus_t CUSPARSELT_API cusparseLtMatmulPlanInit(const cusparseLtHandle_t* handle,
                                                         cusparseLtMatmulPlan_t* plan,
                                                         const cusparseLtMatmulDescriptor_t* matmulDescr,
                                                         const cusparseLtMatmulAlgSelection_t* algSelection);
cusparseStatus_t CUSPARSELT_API cusparseLtMatmulPlanDestroy(const cusparseLtMatmulPlan_t* plan);

/* Execution: D = activation(alpha * op(A) op(B) + beta * C + bias) */
cusparseStatus_t CUSPARSELT_API cusparseLtMatmul(const cusparseLtHandle_t* handle,
                                                 const cusparseLtMatmulPlan_t* plan, const void* alpha,
                                                 const void* d_A, const void* d_B, const void* beta,
                                                 const void* d_C, void* d_D, void* workspace,
                                                 cudaStream_t* streams, int32_t numStreams);
cusparseStatus_t CUSPARSELT_API cusparseLtMatmulSearch(const cusparseLtHandle_t* handle,
                                                       cusparseLtMatmulPlan_t* plan, const void* alpha,
                                                       const void* d_A, const void* d_B, const void* beta,
                                                       const void* d_C, void* d_D, void* workspace,
                                                       cudaStream_t* streams, int32_t numStreams);

/* Pruning */
cusparseStatus_t CUSPARSELT_API cusparseLtSpMMAPrune(const cusparseLtHandle_t* handle,
                                                     const cusparseLtMatmulDescriptor_t* matmulDescr,
                                                     const void* d_in, void* d_out,
                                                     cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream);
cusparseStatus_t CUSPARSELT_API cusparseLtSpMMAPruneCheck(const cusparseLtHandle_t* handle,
                                                          const cusparseLtMatmulDescriptor_t* matmulDescr,
                                                          const void* d_in, int* valid, cudaStream_t stream);
cusparseStatus_t CUSPARSELT_API cusparseLtSpMMAPrune2(const cusparseLtHandle_t* handle,
                                                      const cusparseLtMatDescriptor_t* sparseMatDescr,
                                                      int isSparseA, cusparseOperation_t op, const void* d_in,
                                                      void* d_out, cusparseLtPruneAlg_t pruneAlg,
                                                      cudaStream_t stream);
cusparseStatus_t CUSPARSELT_API cusparseLtSpMMAPruneCheck2(const cusparseLtHandle_t* handle,
                                                           const cusparseLtMatDescriptor_t* sparseMatDescr,
                                                           int isSparseA, cusparseOperation_t op,
                                                           const void* d_in, int* d_valid, cudaStream_t stream);

/* Compression */
cusparseStatus_t CUSPARSELT_API cusparseLtSpMMACompressedSize(const cusparseLtHandle_t* handle,
                                                              const cusparseLtMatmulPlan_t* plan,
                                                              size_t* compressedSize,
                                                              size_t* compressedBufferSize);
cusparseStatus_t CUSPARSELT_API cusparseLtSpMMACompress(const cusparseLtHandle_t* handle,
                                                        const cusparseLtMatmulPlan_t* plan, const void* d_dense,
                                                        void* d_compressed, void* d_compressed_buffer,
                                                        cudaStream_t stream);
cusparseStatus_t CUSPARSELT_API cusparseLtSpMMACompressedSize2(const cusparseLtHandle_t* handle,
                                                               const cusparseLtMatDescriptor_t* sparseMatDescr,
                                                               size_t* compressedSize,
                                                               size_t* compressedBufferSize);
cusparseStatus_t CUSPARSELT_API cusparseLtSpMMACompress2(const cusparseLtHandle_t* handle,
                                                         const cusparseLtMatDescriptor_t* sparseMatDescr,
                                                         int isSparseA, cusparseOperation_t op,
                                                         const void* d_dense, void* d_compressed,
                                                         void* d_compressed_buffer, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#undef VGPU_CUSPARSELT_OPAQUE

#endif /* VGPU_CUSPARSELT_H_ */
