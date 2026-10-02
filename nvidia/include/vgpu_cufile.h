/*
 * vgpu_cufile.h -- VirtualGPU's clean-room declarations of the cuFile API
 * (GPUDirect Storage).
 *
 * Written from NVIDIA's publicly documented cuFile API reference
 * (docs.nvidia.com/gpudirect-storage/api-reference-guide), the version that
 * ships with CUDA 13.0, so that programs that read and write files straight
 * into GPU memory can run on VirtualGPU's libcufile. It contains no NVIDIA
 * code. Type layouts and numeric values (the CUfileOpError codes, the enums,
 * the stream flags) follow the documented ABI, so a program compiled with
 * NVIDIA's cufile.h and one compiled with this header call the same library
 * the same way.
 *
 * The simulator's library is cuFile's "compatibility mode": the transfer is
 * an ordinary POSIX read or write staged through host memory, which is what
 * NVIDIA's library does on a machine without the nvidia-fs driver.
 */
#ifndef VGPU_CUFILE_H_
#define VGPU_CUFILE_H_

// Every enum here takes the values a program may store in it, valid or not:
// the entry points check them and answer with the library's own status. In
// C++ an enum without a fixed underlying type cannot even be loaded holding
// an unlisted value (UBSan's -fsanitize=enum), so the C++ view fixes it to
// int -- the same size, and the same ABI, as the C view.
#ifndef VGPU_ENUM_INT
#ifdef __cplusplus
#define VGPU_ENUM_INT : int
#else
#define VGPU_ENUM_INT
#endif
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/socket.h> /* struct sockaddr, for the file system operation table */
#include <sys/types.h>  /* off_t, ssize_t, loff_t */
#include <time.h>       /* struct timespec */

#include <cuda.h> /* CUresult, CUstream */

#ifdef __cplusplus
extern "C" {
#endif

#define CUFILEOP_BASE_ERR 5000

/* The cuFile status codes: 0 for success, otherwise 5000 plus a small number.
 * Data path calls (cuFileRead, cuFileWrite) return their negation, or -1 with
 * errno for a file system error. */
typedef enum CUfileOpError VGPU_ENUM_INT {
  CU_FILE_SUCCESS = 0,
  CU_FILE_DRIVER_NOT_INITIALIZED = CUFILEOP_BASE_ERR + 1,
  CU_FILE_DRIVER_INVALID_PROPS = CUFILEOP_BASE_ERR + 2,
  CU_FILE_DRIVER_UNSUPPORTED_LIMIT = CUFILEOP_BASE_ERR + 3,
  CU_FILE_DRIVER_VERSION_MISMATCH = CUFILEOP_BASE_ERR + 4,
  CU_FILE_DRIVER_VERSION_READ_ERROR = CUFILEOP_BASE_ERR + 5,
  CU_FILE_DRIVER_CLOSING = CUFILEOP_BASE_ERR + 6,
  CU_FILE_PLATFORM_NOT_SUPPORTED = CUFILEOP_BASE_ERR + 7,
  CU_FILE_IO_NOT_SUPPORTED = CUFILEOP_BASE_ERR + 8,
  CU_FILE_DEVICE_NOT_SUPPORTED = CUFILEOP_BASE_ERR + 9,
  CU_FILE_NVFS_DRIVER_ERROR = CUFILEOP_BASE_ERR + 10,
  CU_FILE_CUDA_DRIVER_ERROR = CUFILEOP_BASE_ERR + 11,
  CU_FILE_CUDA_POINTER_INVALID = CUFILEOP_BASE_ERR + 12,
  CU_FILE_CUDA_MEMORY_TYPE_INVALID = CUFILEOP_BASE_ERR + 13,
  CU_FILE_CUDA_POINTER_RANGE_ERROR = CUFILEOP_BASE_ERR + 14,
  CU_FILE_CUDA_CONTEXT_MISMATCH = CUFILEOP_BASE_ERR + 15,
  CU_FILE_INVALID_MAPPING_SIZE = CUFILEOP_BASE_ERR + 16,
  CU_FILE_INVALID_MAPPING_RANGE = CUFILEOP_BASE_ERR + 17,
  CU_FILE_INVALID_FILE_TYPE = CUFILEOP_BASE_ERR + 18,
  CU_FILE_INVALID_FILE_OPEN_FLAG = CUFILEOP_BASE_ERR + 19,
  CU_FILE_DIO_NOT_SET = CUFILEOP_BASE_ERR + 20,
  CU_FILE_INVALID_VALUE = CUFILEOP_BASE_ERR + 22,
  CU_FILE_MEMORY_ALREADY_REGISTERED = CUFILEOP_BASE_ERR + 23,
  CU_FILE_MEMORY_NOT_REGISTERED = CUFILEOP_BASE_ERR + 24,
  CU_FILE_PERMISSION_DENIED = CUFILEOP_BASE_ERR + 25,
  CU_FILE_DRIVER_ALREADY_OPEN = CUFILEOP_BASE_ERR + 26,
  CU_FILE_HANDLE_NOT_REGISTERED = CUFILEOP_BASE_ERR + 27,
  CU_FILE_HANDLE_ALREADY_REGISTERED = CUFILEOP_BASE_ERR + 28,
  CU_FILE_DEVICE_NOT_FOUND = CUFILEOP_BASE_ERR + 29,
  CU_FILE_INTERNAL_ERROR = CUFILEOP_BASE_ERR + 30,
  CU_FILE_GETNEWFD_FAILED = CUFILEOP_BASE_ERR + 31,
  CU_FILE_NVFS_SETUP_ERROR = CUFILEOP_BASE_ERR + 33,
  CU_FILE_IO_DISABLED = CUFILEOP_BASE_ERR + 34,
  CU_FILE_BATCH_SUBMIT_FAILED = CUFILEOP_BASE_ERR + 35,
  CU_FILE_GPU_MEMORY_PINNING_FAILED = CUFILEOP_BASE_ERR + 36,
  CU_FILE_BATCH_FULL = CUFILEOP_BASE_ERR + 37,
  CU_FILE_ASYNC_NOT_SUPPORTED = CUFILEOP_BASE_ERR + 38,
  CU_FILE_INTERNAL_BATCH_SETUP_ERROR = CUFILEOP_BASE_ERR + 39,
  CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR = CUFILEOP_BASE_ERR + 40,
  CU_FILE_INTERNAL_BATCH_GETSTATUS_ERROR = CUFILEOP_BASE_ERR + 41,
  CU_FILE_INTERNAL_BATCH_CANCEL_ERROR = CUFILEOP_BASE_ERR + 42,
  CU_FILE_NOMEM_ERROR = CUFILEOP_BASE_ERR + 43,
  CU_FILE_IO_ERROR = CUFILEOP_BASE_ERR + 44,
  CU_FILE_INTERNAL_BUF_REGISTER_ERROR = CUFILEOP_BASE_ERR + 45,
  CU_FILE_HASH_OPR_ERROR = CUFILEOP_BASE_ERR + 46,
  CU_FILE_INVALID_CONTEXT_ERROR = CUFILEOP_BASE_ERR + 47,
  CU_FILE_NVFS_INTERNAL_DRIVER_ERROR = CUFILEOP_BASE_ERR + 48,
  CU_FILE_BATCH_NOCOMPAT_ERROR = CUFILEOP_BASE_ERR + 49,
  CU_FILE_IO_MAX_ERROR = CUFILEOP_BASE_ERR + 50
} CUfileOpError;

/* A status: the cuFile code, and the CUDA driver's result when the code is
 * CU_FILE_CUDA_DRIVER_ERROR. */
typedef struct CUfileError {
  CUfileOpError err;
  CUresult cu_err;
} CUfileError_t;

#define IS_CUFILE_ERR(err) (abs((err)) > CUFILEOP_BASE_ERR)
#define IS_CUDA_ERR(status) ((status).err == CU_FILE_CUDA_DRIVER_ERROR)
#define CU_FILE_CUDA_ERR(status) ((status).cu_err)

/* Bit positions in CUfileDrvProps_t.nvfs.dstatusflags. */
typedef enum CUfileDriverStatusFlags VGPU_ENUM_INT {
  CU_FILE_LUSTRE_SUPPORTED = 0,
  CU_FILE_WEKAFS_SUPPORTED = 1,
  CU_FILE_NFS_SUPPORTED = 2,
  CU_FILE_GPFS_SUPPORTED = 3,
  CU_FILE_NVME_SUPPORTED = 4,
  CU_FILE_NVMEOF_SUPPORTED = 5,
  CU_FILE_SCSI_SUPPORTED = 6,
  CU_FILE_SCALEFLUX_CSD_SUPPORTED = 7,
  CU_FILE_NVMESH_SUPPORTED = 8,
  CU_FILE_BEEGFS_SUPPORTED = 9,
  CU_FILE_NVME_P2P_SUPPORTED = 11,
  CU_FILE_SCATEFS_SUPPORTED = 12
} CUfileDriverStatusFlags_t;

/* Bit positions in CUfileDrvProps_t.nvfs.dcontrolflags. */
typedef enum CUfileDriverControlFlags VGPU_ENUM_INT {
  CU_FILE_USE_POLL_MODE = 0,
  CU_FILE_ALLOW_COMPAT_MODE = 1
} CUfileDriverControlFlags_t;

/* Bit positions in CUfileDrvProps_t.fflags. */
typedef enum CUfileFeatureFlags VGPU_ENUM_INT {
  CU_FILE_DYN_ROUTING_SUPPORTED = 0,
  CU_FILE_BATCH_IO_SUPPORTED = 1,
  CU_FILE_STREAMS_SUPPORTED = 2,
  CU_FILE_PARALLEL_IO_SUPPORTED = 3
} CUfileFeatureFlags_t;

typedef struct CUfileDrvProps {
  struct {
    unsigned int major_version;
    unsigned int minor_version;
    size_t poll_thresh_size;    /* KiB */
    size_t max_direct_io_size;  /* KiB */
    unsigned int dstatusflags;
    unsigned int dcontrolflags;
  } nvfs;
  unsigned int fflags;
  unsigned int max_device_cache_size;     /* KiB */
  unsigned int per_buffer_cache_size;     /* KiB */
  unsigned int max_device_pinned_mem_size; /* KiB */
  unsigned int max_batch_io_size;
  unsigned int max_batch_io_timeout_msecs;
} CUfileDrvProps_t;

typedef struct sockaddr sockaddr_t;

typedef struct cufileRDMAInfo {
  int version;
  int desc_len;
  const char* desc_str;
} cufileRDMAInfo_t;

#define CU_FILE_RDMA_REGISTER 1
#define CU_FILE_RDMA_RELAXED_ORDERING (1 << 1)

/* A user-space file system's operation table (CU_FILE_HANDLE_TYPE_USERSPACE_FS). */
typedef struct CUfileFSOps {
  const char* (*fs_type)(void* handle);
  int (*getRDMADeviceList)(void* handle, sockaddr_t** hostaddrs);
  int (*getRDMADevicePriority)(void* handle, char*, size_t, loff_t, sockaddr_t* hostaddr);
  ssize_t (*read)(void* handle, char*, size_t, loff_t, cufileRDMAInfo_t*);
  ssize_t (*write)(void* handle, const char*, size_t, loff_t, cufileRDMAInfo_t*);
} CUfileFSOps_t;

enum CUfileFileHandleType VGPU_ENUM_INT {
  CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1,
  CU_FILE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
  CU_FILE_HANDLE_TYPE_USERSPACE_FS = 3
};

typedef struct CUfileDescr_t {
  enum CUfileFileHandleType type;
  union {
    int fd;
    void* handle;
  } handle;
  const CUfileFSOps_t* fs_ops;
} CUfileDescr_t;

typedef void* CUfileHandle_t;

/* ---- driver ---- */
CUfileError_t cuFileDriverOpen(void);
/* The header's cuFileDriverClose is the _v2 entry point; both are exported. */
CUfileError_t cuFileDriverClose(void);
CUfileError_t cuFileDriverClose_v2(void);
#define cuFileDriverClose cuFileDriverClose_v2
long cuFileUseCount(void);
CUfileError_t cuFileDriverGetProperties(CUfileDrvProps_t* props);
CUfileError_t cuFileDriverSetPollMode(bool poll, size_t poll_threshold_size);
CUfileError_t cuFileDriverSetMaxDirectIOSize(size_t max_direct_io_size);
CUfileError_t cuFileDriverSetMaxCacheSize(size_t max_cache_size);
CUfileError_t cuFileDriverSetMaxPinnedMemSize(size_t max_pinned_size);
CUfileError_t cuFileGetVersion(int* version);

/* ---- files and buffers ---- */
CUfileError_t cuFileHandleRegister(CUfileHandle_t* fh, CUfileDescr_t* descr);
void cuFileHandleDeregister(CUfileHandle_t fh);
CUfileError_t cuFileBufRegister(const void* bufPtr_base, size_t length, int flags);
CUfileError_t cuFileBufDeregister(const void* bufPtr_base);

/* ---- synchronous I/O ---- */
ssize_t cuFileRead(CUfileHandle_t fh, void* bufPtr_base, size_t size, off_t file_offset,
                   off_t bufPtr_offset);
ssize_t cuFileWrite(CUfileHandle_t fh, const void* bufPtr_base, size_t size, off_t file_offset,
                    off_t bufPtr_offset);

/* ---- batch I/O ---- */
typedef enum CUfileOpcode VGPU_ENUM_INT { CUFILE_READ = 0, CUFILE_WRITE } CUfileOpcode_t;

typedef enum CUFILEStatus_enum VGPU_ENUM_INT {
  CUFILE_WAITING = 0x000001,
  CUFILE_PENDING = 0x000002,
  CUFILE_INVALID = 0x000004,
  CUFILE_CANCELED = 0x000008,
  CUFILE_COMPLETE = 0x000010,
  CUFILE_TIMEOUT = 0x000020,
  CUFILE_FAILED = 0x000040
} CUfileStatus_t;

typedef enum cufileBatchMode VGPU_ENUM_INT { CUFILE_BATCH = 1 } CUfileBatchMode_t;

typedef struct CUfileIOParams {
  CUfileBatchMode_t mode; /* first, always */
  union {
    struct {
      void* devPtr_base; /* device or host memory */
      off_t file_offset;
      off_t devPtr_offset;
      size_t size;
    } batch;
  } u;
  CUfileHandle_t fh;
  CUfileOpcode_t opcode;
  void* cookie;
} CUfileIOParams_t;

typedef struct CUfileIOEvents {
  void* cookie;
  CUfileStatus_t status;
  size_t ret; /* bytes transferred, or a negative error */
} CUfileIOEvents_t;

typedef void* CUfileBatchHandle_t;

CUfileError_t cuFileBatchIOSetUp(CUfileBatchHandle_t* batch_idp, unsigned nr);
CUfileError_t cuFileBatchIOSubmit(CUfileBatchHandle_t batch_idp, unsigned nr, CUfileIOParams_t* iocbp,
                                  unsigned int flags);
CUfileError_t cuFileBatchIOGetStatus(CUfileBatchHandle_t batch_idp, unsigned min_nr, unsigned* nr,
                                     CUfileIOEvents_t* iocbp, struct timespec* timeout);
CUfileError_t cuFileBatchIOCancel(CUfileBatchHandle_t batch_idp);
void cuFileBatchIODestroy(CUfileBatchHandle_t batch_idp);

/* ---- stream-ordered I/O ---- */
#define CU_FILE_STREAM_FIXED_BUF_OFFSET 1
#define CU_FILE_STREAM_FIXED_FILE_OFFSET 2
#define CU_FILE_STREAM_FIXED_FILE_SIZE 4
#define CU_FILE_STREAM_PAGE_ALIGNED_INPUTS 8

CUfileError_t cuFileReadAsync(CUfileHandle_t fh, void* bufPtr_base, size_t* size_p, off_t* file_offset_p,
                              off_t* bufPtr_offset_p, ssize_t* bytes_read_p, CUstream stream);
CUfileError_t cuFileWriteAsync(CUfileHandle_t fh, void* bufPtr_base, size_t* size_p, off_t* file_offset_p,
                               off_t* bufPtr_offset_p, ssize_t* bytes_written_p, CUstream stream);
CUfileError_t cuFileStreamRegister(CUstream stream, unsigned flags);
CUfileError_t cuFileStreamDeregister(CUstream stream);

/* ---- configuration ---- */
typedef enum CUFileSizeTConfigParameter_t VGPU_ENUM_INT {
  CUFILE_PARAM_PROFILE_STATS,
  CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH,
  CUFILE_PARAM_EXECUTION_MAX_IO_THREADS,
  CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB,
  CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM,
  CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB,
  CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB,
  CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB,
  CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB,
  CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE,
  CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB,
  CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS
} CUFileSizeTConfigParameter_t;

typedef enum CUFileBoolConfigParameter_t VGPU_ENUM_INT {
  CUFILE_PARAM_PROPERTIES_USE_POLL_MODE,
  CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE,
  CUFILE_PARAM_FORCE_COMPAT_MODE,
  CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE,
  CUFILE_PARAM_EXECUTION_PARALLEL_IO,
  CUFILE_PARAM_PROFILE_NVTX,
  CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY,
  CUFILE_PARAM_USE_PCIP2PDMA,
  CUFILE_PARAM_PREFER_IO_URING,
  CUFILE_PARAM_FORCE_ODIRECT_MODE,
  CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION,
  CUFILE_PARAM_STREAM_MEMOPS_BYPASS
} CUFileBoolConfigParameter_t;

typedef enum CUFileStringConfigParameter_t VGPU_ENUM_INT {
  CUFILE_PARAM_LOGGING_LEVEL,
  CUFILE_PARAM_ENV_LOGFILE_PATH,
  CUFILE_PARAM_LOG_DIR
} CUFileStringConfigParameter_t;

typedef enum CUFileArrayConfigParameter_t VGPU_ENUM_INT {
  CUFILE_PARAM_POSIX_POOL_SLAB_SIZE_KB,
  CUFILE_PARAM_POSIX_POOL_SLAB_COUNT
} CUFileArrayConfigParameter_t;

CUfileError_t cuFileGetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t* value);
CUfileError_t cuFileGetParameterBool(CUFileBoolConfigParameter_t param, bool* value);
CUfileError_t cuFileGetParameterString(CUFileStringConfigParameter_t param, char* desc_str, int len);
CUfileError_t cuFileGetParameterMinMaxValue(CUFileSizeTConfigParameter_t param, size_t* min_value,
                                            size_t* max_value);
CUfileError_t cuFileSetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t value);
CUfileError_t cuFileSetParameterBool(CUFileBoolConfigParameter_t param, bool value);
CUfileError_t cuFileSetParameterString(CUFileStringConfigParameter_t param, const char* desc_str);
CUfileError_t cuFileSetParameterPosixPoolSlabArray(const size_t* size_values, const size_t* count_values,
                                                   int len);
CUfileError_t cuFileGetParameterPosixPoolSlabArray(size_t* size_values, size_t* count_values, int len);

/* ---- statistics ---- */
CUfileError_t cuFileSetStatsLevel(int level);
CUfileError_t cuFileGetStatsLevel(int* level);

#define CUFILE_GPU_UUID_LEN 16

typedef struct CUfileOpCounter {
  uint64_t ok;
  uint64_t err;
} CUfileOpCounter_t;

typedef struct CUfileStatsLevel1 {
  CUfileOpCounter_t read_ops;
  CUfileOpCounter_t write_ops;
  CUfileOpCounter_t hdl_register_ops;
  CUfileOpCounter_t hdl_deregister_ops;
  CUfileOpCounter_t buf_register_ops;
  CUfileOpCounter_t buf_deregister_ops;
  uint64_t read_bytes;
  uint64_t write_bytes;
  uint64_t read_bw_bytes_per_sec;
  uint64_t write_bw_bytes_per_sec;
  uint64_t read_lat_avg_us;
  uint64_t write_lat_avg_us;
  uint64_t read_ops_per_sec;
  uint64_t write_ops_per_sec;
  uint64_t read_lat_sum_us;
  uint64_t write_lat_sum_us;
  CUfileOpCounter_t batch_submit_ops;
  CUfileOpCounter_t batch_complete_ops;
  CUfileOpCounter_t batch_setup_ops;
  CUfileOpCounter_t batch_cancel_ops;
  CUfileOpCounter_t batch_destroy_ops;
  CUfileOpCounter_t batch_enqueued_ops;
  CUfileOpCounter_t batch_posix_enqueued_ops;
  CUfileOpCounter_t batch_processed_ops;
  CUfileOpCounter_t batch_posix_processed_ops;
  CUfileOpCounter_t batch_nvfs_submit_ops;
  CUfileOpCounter_t batch_p2p_submit_ops;
  CUfileOpCounter_t batch_aio_submit_ops;
  CUfileOpCounter_t batch_iouring_submit_ops;
  CUfileOpCounter_t batch_mixed_io_submit_ops;
  CUfileOpCounter_t batch_total_submit_ops;
  uint64_t batch_read_bytes;
  uint64_t batch_write_bytes;
  uint64_t batch_read_bw_bytes;
  uint64_t batch_write_bw_bytes;
  uint64_t batch_submit_lat_avg_us;
  uint64_t batch_completion_lat_avg_us;
  uint64_t batch_submit_ops_per_sec;
  uint64_t batch_complete_ops_per_sec;
  uint64_t batch_submit_lat_sum_us;
  uint64_t batch_completion_lat_sum_us;
  uint64_t last_batch_read_bytes;
  uint64_t last_batch_write_bytes;
} CUfileStatsLevel1_t;

typedef struct CUfileStatsLevel2 {
  CUfileStatsLevel1_t basic;
  uint64_t read_size_kb_hist[32];
  uint64_t write_size_kb_hist[32];
} CUfileStatsLevel2_t;

typedef struct CUfilePerGpuStats {
  char uuid[CUFILE_GPU_UUID_LEN];
  uint64_t read_bytes;
  uint64_t read_bw_bytes_per_sec;
  uint64_t read_utilization;
  uint64_t read_duration_us;
  uint64_t n_total_reads;
  uint64_t n_p2p_reads;
  uint64_t n_nvfs_reads;
  uint64_t n_posix_reads;
  uint64_t n_unaligned_reads;
  uint64_t n_dr_reads;
  uint64_t n_sparse_regions;
  uint64_t n_inline_regions;
  uint64_t n_reads_err;
  uint64_t writes_bytes;
  uint64_t write_bw_bytes_per_sec;
  uint64_t write_utilization;
  uint64_t write_duration_us;
  uint64_t n_total_writes;
  uint64_t n_p2p_writes;
  uint64_t n_nvfs_writes;
  uint64_t n_posix_writes;
  uint64_t n_unaligned_writes;
  uint64_t n_dr_writes;
  uint64_t n_writes_err;
  uint64_t n_mmap;
  uint64_t n_mmap_ok;
  uint64_t n_mmap_err;
  uint64_t n_mmap_free;
  uint64_t reg_bytes;
} CUfilePerGpuStats_t;

typedef struct CUfileStatsLevel3 {
  CUfileStatsLevel2_t detailed;
  uint32_t num_gpus;
  CUfilePerGpuStats_t per_gpu_stats[16];
} CUfileStatsLevel3_t;

CUfileError_t cuFileStatsStart(void);
CUfileError_t cuFileStatsStop(void);
CUfileError_t cuFileStatsReset(void);
CUfileError_t cuFileGetStatsL1(CUfileStatsLevel1_t* stats);
CUfileError_t cuFileGetStatsL2(CUfileStatsLevel2_t* stats);
CUfileError_t cuFileGetStatsL3(CUfileStatsLevel3_t* stats);
/* The BAR1 aperture of a GPU, in KiB. */
CUfileError_t cuFileGetBARSizeInKB(int gpuIndex, size_t* barSize);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_CUFILE_H_ */
