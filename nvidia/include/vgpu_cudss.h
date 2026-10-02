/*
 * vgpu_cudss.h -- VirtualGPU's clean-room declarations of the cuDSS API.
 *
 * Written from NVIDIA's publicly documented cuDSS API
 * (docs.nvidia.com/cuda/cudss), version 0.8, so that programs built against
 * cuDSS -- the sparse direct solver -- can run on VirtualGPU's libcudss. It
 * contains no NVIDIA code. Type layouts and numeric values (status codes,
 * enums, phase bits) follow the documented ABI, so a program compiled with
 * NVIDIA's cudss.h and one compiled with this header call the same library
 * the same way.
 *
 * Why the simulator needs its own: NVIDIA's libcudss carries a statically
 * linked CUDA runtime, which reaches the driver through NVIDIA's undocumented
 * internal interface, so it cannot run on a simulated driver (cudssCreate
 * fails).
 *
 * The whole 0.8 API is declared. The communication and threading layer
 * interfaces (the structs a cuDSS comm or threading library fills in) are
 * not: VirtualGPU's cuDSS loads neither library.
 */
#ifndef VGPU_CUDSS_H_
#define VGPU_CUDSS_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h> /* FILE, for cudssLoggerSetFile */

#include <driver_types.h>  /* cudaStream_t */
#include <library_types.h> /* cudaDataType_t, libraryPropertyType */

#define CUDSS_VERSION_MAJOR 0
#define CUDSS_VERSION_MINOR 8
#define CUDSS_VERSION_PATCH 0
/* 800 for 0.8.0. Programs test it (>= 800) to pick the 0.8 signatures, in
 * which cudssMatrixCreateCsr takes the offset type and index type apart. */
#define CUDSS_VERSION \
  (CUDSS_VERSION_MAJOR * 10000 + CUDSS_VERSION_MINOR * 100 + CUDSS_VERSION_PATCH)

#define CUDSS_ALLOCATOR_NAME_LEN 64

#ifndef CUDSSAPI
#define CUDSSAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cudssContext* cudssHandle_t;
typedef struct cudssMatrix* cudssMatrix_t; /* dense or sparse, single or batch */
typedef struct cudssData* cudssData_t;     /* factors and per-system results */
typedef struct cudssConfig* cudssConfig_t; /* solver settings */

/* Value and index types. The floating-point ones are cudaDataType_t's values. */
typedef enum cudssDataType_t {
  CUDSS_DATA_TYPE_UNSET = 1024,
  CUDSS_R_32F = CUDA_R_32F,
  CUDSS_R_64F = CUDA_R_64F,
  CUDSS_C_32F = CUDA_C_32F,
  CUDSS_C_64F = CUDA_C_64F,
  CUDSS_R_64F_64F = 1025 + CUDA_R_64F, /* double-double */
  CUDSS_R_32I = CUDA_R_32I,
  CUDSS_R_64I = CUDA_R_64I
} cudssDataType_t;

typedef enum cudssConfigParam_t {
  CUDSS_CONFIG_REORDERING_ALG = 0,       /* int (cudssReorderingAlg_t) */
  CUDSS_CONFIG_FACTORIZATION_ALG = 1,    /* int (cudssFactorizationAlg_t) */
  CUDSS_CONFIG_SOLVE_ALG = 2,            /* int (cudssSolveAlg_t) */
  CUDSS_CONFIG_MATCHING_ALG = 3,         /* int (cudssMatchingAlg_t) */
  CUDSS_CONFIG_SOLVE_MODE = 4,           /* int */
  CUDSS_CONFIG_IR_N_STEPS = 5,           /* int */
  CUDSS_CONFIG_IR_TOL = 6,               /* double */
  CUDSS_CONFIG_PIVOT_TYPE = 7,           /* int (cudssPivotType_t) */
  CUDSS_CONFIG_PIVOT_THRESHOLD = 8,      /* double */
  CUDSS_CONFIG_PIVOT_EPSILON = 9,        /* double */
  CUDSS_CONFIG_MAX_LU_NNZ = 10,          /* int64_t */
  CUDSS_CONFIG_HYBRID_MEMORY_MODE = 11,  /* int */
  CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT = 12, /* int64_t */
  CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY = 13,   /* int */
  CUDSS_CONFIG_HOST_NTHREADS = 14,       /* int */
  CUDSS_CONFIG_HYBRID_EXECUTE_MODE = 15, /* int */
  CUDSS_CONFIG_PIVOT_EPSILON_ALG = 16,   /* int (cudssPivotEpsilonAlg_t) */
  CUDSS_CONFIG_ND_NLEVELS = 17,          /* int */
  CUDSS_CONFIG_UBATCH_SIZE = 18,         /* int */
  CUDSS_CONFIG_UBATCH_INDEX = 19,        /* int */
  CUDSS_CONFIG_USE_SUPERPANELS = 20,     /* int */
  CUDSS_CONFIG_DEVICE_COUNT = 21,        /* int */
  CUDSS_CONFIG_DEVICE_INDICES = 22,      /* int[device count] */
  CUDSS_CONFIG_SCHUR_MODE = 23,          /* int */
  CUDSS_CONFIG_DETERMINISTIC_MODE = 24,  /* int */
  CUDSS_CONFIG_ND_UBFACTOR = 25          /* int */
} cudssConfigParam_t;

/* "index" below is the system matrix's index type; "value" its value type. */
typedef enum cudssDataParam_t {
  CUDSS_DATA_INFO = 0,                   /* out/in: int */
  CUDSS_DATA_LU_NNZ = 1,                 /* out: int64_t */
  CUDSS_DATA_NPIVOTS = 2,                /* out: index */
  CUDSS_DATA_INERTIA = 3,                /* out: index[2], positive then negative */
  CUDSS_DATA_PERM_REORDER_ROW = 4,       /* out: index[n] */
  CUDSS_DATA_PERM_REORDER_COL = 5,       /* out: index[n] */
  CUDSS_DATA_PERM_ROW = 6,               /* out: index[n] */
  CUDSS_DATA_PERM_COL = 7,               /* out: index[n] */
  CUDSS_DATA_DIAG = 8,                   /* out: value[n] */
  CUDSS_DATA_USER_PERM = 9,              /* in/out: index[n] */
  CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN = 10, /* out: int64_t */
  CUDSS_DATA_COMM_DEVICE = 11,           /* in: void* */
  CUDSS_DATA_COMM_HOST = 12,             /* in: void* */
  CUDSS_DATA_MEMORY_ESTIMATES = 13,      /* out: int64_t[16] */
  CUDSS_DATA_PERM_MATCHING = 14,         /* out: index[n] */
  CUDSS_DATA_SCALE_ROW = 15,             /* out: real[n] */
  CUDSS_DATA_SCALE_COL = 16,             /* out: real[n] */
  CUDSS_DATA_NSUPERPANELS = 17,          /* out: index */
  CUDSS_DATA_USER_SCHUR_INDICES = 18,    /* in: index[n] */
  CUDSS_DATA_SCHUR_SHAPE = 19,           /* out: int64_t[3] */
  CUDSS_DATA_SCHUR_MATRIX = 20,          /* in: cudssMatrix_t */
  CUDSS_DATA_USER_ND_PARTITION_TREE = 21, /* in/out: index[2^levels - 1] */
  CUDSS_DATA_ND_PARTITION_TREE = 22,     /* out: index[2^levels - 1] */
  CUDSS_DATA_USER_HOST_INTERRUPT = 23,   /* in: int* */
  CUDSS_DATA_IR_N_STEPS = 24,            /* out: int */
  CUDSS_DATA_UBATCH_MASK = 25,           /* in/out: int64_t */
  CUDSS_DATA_FLOPS = 26                  /* out: int64_t */
} cudssDataParam_t;

/* Phases are bits; cudssExecute takes several OR'd together, in order. */
typedef enum cudssPhase_t {
  CUDSS_PHASE_REORDERING = 1 << 0,
  CUDSS_PHASE_SYMBOLIC_FACTORIZATION = 1 << 1,
  CUDSS_PHASE_ANALYSIS = (1 << 0) | (1 << 1),
  CUDSS_PHASE_FACTORIZATION = 1 << 2,
  CUDSS_PHASE_REFACTORIZATION = 1 << 3,
  CUDSS_PHASE_SOLVE_FWD_PERM = 1 << 4,
  CUDSS_PHASE_SOLVE_FWD = 1 << 5,
  CUDSS_PHASE_SOLVE_DIAG = 1 << 6,
  CUDSS_PHASE_SOLVE_BWD = 1 << 7,
  CUDSS_PHASE_SOLVE_BWD_PERM = 1 << 8,
  CUDSS_PHASE_SOLVE_REFINEMENT = 1 << 9,
  CUDSS_PHASE_SOLVE = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)
} cudssPhase_t;

typedef enum cudssStatus_t {
  CUDSS_STATUS_SUCCESS = 0,
  CUDSS_STATUS_NOT_INITIALIZED = 1,
  CUDSS_STATUS_ALLOC_FAILED = 2,
  CUDSS_STATUS_INVALID_VALUE = 3,
  CUDSS_STATUS_NOT_SUPPORTED = 4,
  CUDSS_STATUS_EXECUTION_FAILED = 5,
  CUDSS_STATUS_INTERNAL_ERROR = 6,
  CUDSS_STATUS_IR_FAILED = 7
} cudssStatus_t;

typedef enum cudssMatrixType_t {
  CUDSS_MTYPE_GENERAL = 0,   /* LU */
  CUDSS_MTYPE_SYMMETRIC = 1, /* LDL^T */
  CUDSS_MTYPE_HERMITIAN = 2, /* LDL^H */
  CUDSS_MTYPE_SPD = 3,       /* Cholesky */
  CUDSS_MTYPE_HPD = 4        /* complex Cholesky */
} cudssMatrixType_t;

typedef enum cudssMatrixViewType_t {
  CUDSS_MVIEW_FULL = 0,
  CUDSS_MVIEW_LOWER = 1, /* lower triangle and diagonal */
  CUDSS_MVIEW_UPPER = 2  /* upper triangle and diagonal */
} cudssMatrixViewType_t;

typedef enum cudssIndexBase_t {
  CUDSS_BASE_ZERO = 0,
  CUDSS_BASE_ONE = 1
} cudssIndexBase_t;

typedef enum cudssLayout_t {
  CUDSS_LAYOUT_COL_MAJOR = 0,
  CUDSS_LAYOUT_ROW_MAJOR = 1 /* refused: cuDSS 0.8 supports column-major only */
} cudssLayout_t;

typedef enum cudssReorderingAlg_t {
  CUDSS_REORDERING_ALG_DEFAULT = 0,
  CUDSS_REORDERING_ALG_BTF_COLAMD = 1,
  CUDSS_REORDERING_ALG_COLAMD = 2,
  CUDSS_REORDERING_ALG_AMD = 3,
  CUDSS_REORDERING_ALG_NESTED_DISSECTION = 4,
  CUDSS_REORDERING_ALG_NONE = 5
} cudssReorderingAlg_t;

typedef enum cudssFactorizationAlg_t {
  CUDSS_FACTORIZATION_ALG_DEFAULT = 0,
  CUDSS_FACTORIZATION_ALG_MULTIBLOCK = 1,
  CUDSS_FACTORIZATION_ALG_GENERAL = 2
} cudssFactorizationAlg_t;

typedef enum cudssPivotEpsilonAlg_t {
  CUDSS_PIVOT_EPSILON_ALG_DEFAULT = 0,
  CUDSS_PIVOT_EPSILON_ALG_SCALED = 1,
  CUDSS_PIVOT_EPSILON_ALG_STATIC = 2
} cudssPivotEpsilonAlg_t;

typedef enum cudssSolveAlg_t {
  CUDSS_SOLVE_ALG_DEFAULT = 0,
  CUDSS_SOLVE_ALG_GENERAL = 1
} cudssSolveAlg_t;

typedef enum cudssMatchingAlg_t {
  CUDSS_MATCHING_ALG_NONE = 0,
  CUDSS_MATCHING_ALG_MAX_DIAG_COUNT = 1,
  CUDSS_MATCHING_ALG_MAX_MIN_DIAG = 2,
  CUDSS_MATCHING_ALG_MAX_MIN_DIAG_ALT = 3,
  CUDSS_MATCHING_ALG_MAX_DIAG_SUM = 4,
  CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT = 5,
  CUDSS_MATCHING_ALG_AUTO = 6
} cudssMatchingAlg_t;

typedef enum cudssPivotType_t {
  CUDSS_PIVOT_AUTO = 0,
  CUDSS_PIVOT_NONE = 1,
  CUDSS_PIVOT_GLOBAL_COL = 2,
  CUDSS_PIVOT_GLOBAL_ROW = 3,
  CUDSS_PIVOT_DIAGONAL = 4,
  CUDSS_PIVOT_LOCAL_BLOCK = 5,
  CUDSS_PIVOT_BUNCH_KAUFMAN = 6 /* reserved by cuDSS 0.8: NOT_SUPPORTED */
} cudssPivotType_t;

/* Bit flags; a batch of CSR matrices reports CSR | BATCH. */
typedef enum cudssMatrixFormat_t {
  CUDSS_MFORMAT_DENSE = 1,
  CUDSS_MFORMAT_CSR = 2,
  CUDSS_MFORMAT_BATCH = 4,
  CUDSS_MFORMAT_DISTRIBUTED = 8
} cudssMatrixFormat_t;

/* A stream-ordered device memory pool the library may allocate from. */
typedef struct {
  void* ctx;
  int (*device_alloc)(void* ctx, void** ptr, size_t size, cudaStream_t stream);
  int (*device_free)(void* ctx, void* ptr, size_t size, cudaStream_t stream);
  char name[CUDSS_ALLOCATOR_NAME_LEN];
} cudssDeviceMemHandler_t;

/* The CUDSS_R_64F_64F element: an unevaluated sum hi + lo, 16-byte aligned. */
#ifdef __cplusplus
struct alignas(16) cudss_fp64mp2_t {
  double hi;
  double lo;
};
#else
typedef struct {
  _Alignas(16) double hi;
  double lo;
} cudss_fp64mp2_t;
#endif

typedef void (*cudssLoggerCallback_t)(int logLevel, const char* functionName, const char* message);

/* Library handle */
cudssStatus_t CUDSSAPI cudssCreate(cudssHandle_t* handle);
cudssStatus_t CUDSSAPI cudssCreateMg(cudssHandle_t* handle, int device_count, const int* device_indices);
cudssStatus_t CUDSSAPI cudssDestroy(cudssHandle_t handle);
cudssStatus_t CUDSSAPI cudssGetProperty(libraryPropertyType propertyType, int* value);
cudssStatus_t CUDSSAPI cudssSetStream(cudssHandle_t handle, cudaStream_t stream);
cudssStatus_t CUDSSAPI cudssSetMgStreams(cudssHandle_t handle, const cudaStream_t* streams, int stream_count);
cudssStatus_t CUDSSAPI cudssSetCommLayer(cudssHandle_t handle, const char* commLibFileName);
cudssStatus_t CUDSSAPI cudssSetThreadingLayer(cudssHandle_t handle, const char* thrLibFileName);
cudssStatus_t CUDSSAPI cudssGetDeviceMemHandler(const cudssHandle_t handle, cudssDeviceMemHandler_t* handler);
cudssStatus_t CUDSSAPI cudssSetDeviceMemHandler(cudssHandle_t handle, const cudssDeviceMemHandler_t* handler);

/* Settings and solver data */
cudssStatus_t CUDSSAPI cudssConfigCreate(cudssConfig_t* config);
cudssStatus_t CUDSSAPI cudssConfigDestroy(cudssConfig_t config);
cudssStatus_t CUDSSAPI cudssConfigSet(cudssConfig_t config, cudssConfigParam_t param, const void* value,
                                      size_t sizeInBytes);
cudssStatus_t CUDSSAPI cudssConfigGet(const cudssConfig_t config, cudssConfigParam_t param, void* value,
                                      size_t sizeInBytes, size_t* sizeWritten);
cudssStatus_t CUDSSAPI cudssDataCreate(const cudssHandle_t handle, cudssData_t* data);
cudssStatus_t CUDSSAPI cudssDataDestroy(cudssHandle_t handle, cudssData_t data);
cudssStatus_t CUDSSAPI cudssDataSet(const cudssHandle_t handle, cudssData_t data, cudssDataParam_t param,
                                    const void* value, size_t sizeInBytes);
cudssStatus_t CUDSSAPI cudssDataGet(const cudssHandle_t handle, const cudssData_t data, cudssDataParam_t param,
                                    void* value, size_t sizeInBytes, size_t* sizeWritten);

/* The solver: one or more phases, in order */
cudssStatus_t CUDSSAPI cudssExecute(cudssHandle_t handle, int phase, const cudssConfig_t config,
                                    cudssData_t data, const cudssMatrix_t matrix, cudssMatrix_t solution,
                                    const cudssMatrix_t rhs);

/* Matrix wrappers: thin descriptors over the caller's buffers */
cudssStatus_t CUDSSAPI cudssMatrixCreateDn(cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t ld,
                                           const void* values, cudssDataType_t valueType, cudssLayout_t layout);
cudssStatus_t CUDSSAPI cudssMatrixCreateCsr(cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t nnz,
                                            const void* rowStart, const void* rowEnd, const void* colIndices,
                                            const void* values, cudssDataType_t offsetType,
                                            cudssDataType_t indexType, cudssDataType_t valueType,
                                            cudssMatrixType_t mtype, cudssMatrixViewType_t mview,
                                            cudssIndexBase_t indexBase);
cudssStatus_t CUDSSAPI cudssMatrixCreateBatchDn(cudssMatrix_t* matrix, int64_t batchCount, const void* nrows,
                                                const void* ncols, const void* ld, const void* const* values,
                                                cudssDataType_t integerType, cudssDataType_t valueType,
                                                cudssLayout_t layout);
cudssStatus_t CUDSSAPI cudssMatrixCreateBatchCsr(cudssMatrix_t* matrix, int64_t batchCount, const void* nrows,
                                                 const void* ncols, const void* nnz,
                                                 const void* const* rowStart, const void* const* rowEnd,
                                                 const void* const* colIndices, const void* const* values,
                                                 cudssDataType_t offsetType, cudssDataType_t indexType,
                                                 cudssDataType_t valueType, cudssMatrixType_t mtype,
                                                 cudssMatrixViewType_t mview, cudssIndexBase_t indexBase);
cudssStatus_t CUDSSAPI cudssMatrixDestroy(cudssMatrix_t matrix);
cudssStatus_t CUDSSAPI cudssMatrixGetDn(const cudssMatrix_t matrix, int64_t* nrows, int64_t* ncols, int64_t* ld,
                                        void** values, cudssDataType_t* type, cudssLayout_t* layout);
cudssStatus_t CUDSSAPI cudssMatrixGetCsr(const cudssMatrix_t matrix, int64_t* nrows, int64_t* ncols,
                                         int64_t* nnz, void** rowStart, void** rowEnd, void** colIndices,
                                         void** values, cudssDataType_t* offsetType, cudssDataType_t* indexType,
                                         cudssDataType_t* valueType, cudssMatrixType_t* mtype,
                                         cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase);
cudssStatus_t CUDSSAPI cudssMatrixSetValues(cudssMatrix_t matrix, const void* values);
cudssStatus_t CUDSSAPI cudssMatrixSetCsrPointers(cudssMatrix_t matrix, const void* rowOffsets, const void* rowEnd,
                                                 const void* colIndices, const void* values);
cudssStatus_t CUDSSAPI cudssMatrixGetBatchDn(const cudssMatrix_t matrix, int64_t* batchCount, void** nrows,
                                             void** ncols, void** ld, void*** values, cudssDataType_t* indexType,
                                             cudssDataType_t* valueType, cudssLayout_t* layout);
cudssStatus_t CUDSSAPI cudssMatrixGetBatchCsr(const cudssMatrix_t matrix, int64_t* batchCount, void** nrows,
                                              void** ncols, void** nnz, void*** rowStart, void*** rowEnd,
                                              void*** colIndices, void*** values, cudssDataType_t* offsetType,
                                              cudssDataType_t* indexType, cudssDataType_t* valueType,
                                              cudssMatrixType_t* mtype, cudssMatrixViewType_t* mview,
                                              cudssIndexBase_t* indexBase);
cudssStatus_t CUDSSAPI cudssMatrixSetBatchValues(cudssMatrix_t matrix, const void* const* values);
cudssStatus_t CUDSSAPI cudssMatrixSetBatchCsrPointers(cudssMatrix_t matrix, const void* const* rowOffsets,
                                                      const void* const* rowEnd, const void* const* colIndices,
                                                      const void* const* values);
cudssStatus_t CUDSSAPI cudssMatrixGetFormat(const cudssMatrix_t matrix, int* format);
cudssStatus_t CUDSSAPI cudssMatrixSetDistributionRow1d(cudssMatrix_t matrix, int64_t first_row, int64_t last_row);
cudssStatus_t CUDSSAPI cudssMatrixGetDistributionRow1d(const cudssMatrix_t matrix, int64_t* first_row,
                                                       int64_t* last_row);

/* Logging */
cudssStatus_t CUDSSAPI cudssLoggerSetCallback(cudssLoggerCallback_t callback);
cudssStatus_t CUDSSAPI cudssLoggerSetFile(FILE* file);
cudssStatus_t CUDSSAPI cudssLoggerOpenFile(const char* logFile);
cudssStatus_t CUDSSAPI cudssLoggerSetLevel(int level);
cudssStatus_t CUDSSAPI cudssLoggerSetMask(int mask);
cudssStatus_t CUDSSAPI cudssLoggerForceDisable(void);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_CUDSS_H_ */
