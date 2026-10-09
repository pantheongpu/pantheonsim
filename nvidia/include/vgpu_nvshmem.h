/*
 * vgpu_nvshmem.h -- VirtualGPU's clean-room declarations of NVSHMEM's host API.
 *
 * Written from NVIDIA's public NVSHMEM documentation and the declarations of
 * its public headers (version 3.8), so that host code built against NVSHMEM
 * runs on VirtualGPU's libnvshmem_host. It contains no NVIDIA code. Type
 * layouts and numeric values (the init attribute and unique-ID structs, team
 * and status constants, signal and comparison operators) follow the
 * documented ABI, and so do the C++ names a program links against:
 * nvshmem_init() is inline and calls nvshmemi_init_thread(), as NVIDIA's is.
 *
 * Device code -- the nvshmem_* calls inside kernels -- is NVIDIA's own headers'
 * business: VirtualGPU's host library sets up the state those inline functions
 * read, for programs built with NVIDIA's headers and device library. This
 * header declares the host side only.
 */
#ifndef VGPU_NVSHMEM_H_
#define VGPU_NVSHMEM_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <driver_types.h> /* cudaStream_t */
#include <vector_types.h> /* dim3 */

#define NVSHMEM_VENDOR_MAJOR_VERSION 3
#define NVSHMEM_VENDOR_MINOR_VERSION 8
#define NVSHMEM_VENDOR_PATCH_VERSION 0
/* The OpenSHMEM specification version NVSHMEM implements. */
#define NVSHMEM_MAJOR_VERSION 1
#define NVSHMEM_MINOR_VERSION 3
#define NVSHMEM_MAX_NAME_LEN 256

/* ---- the init attribute and unique ID ---- */
typedef struct {
  int version;
  char internal[124];
} nvshmemx_uniqueid_v1;
typedef nvshmemx_uniqueid_v1 nvshmemx_uniqueid_t;

typedef struct {
  int version;
  nvshmemx_uniqueid_v1* id;
  int myrank;
  int nranks;
} nvshmemx_uniqueid_args_v1;
typedef nvshmemx_uniqueid_args_v1 nvshmemx_uniqueid_args_t;

typedef struct {
  int version;
  nvshmemx_uniqueid_args_t uid_args;
  int cuda_device_id; /* the device to use; -1 for the current one */
  char content[92];
} nvshmemx_init_args_v2;
typedef nvshmemx_init_args_v2 nvshmemx_init_args_t;

typedef struct {
  int version;
  void* mpi_comm;
  nvshmemx_init_args_t args;
} nvshmemx_init_attr_v2;
typedef nvshmemx_init_attr_v2 nvshmemx_init_attr_t;

typedef struct {
  int major;
  int minor;
  int patch;
} nvshmemi_version_t;

#ifdef __cplusplus
static_assert(sizeof(nvshmemx_uniqueid_t) == 128, "nvshmemx_uniqueid_t is 128 bytes");
static_assert(sizeof(nvshmemx_uniqueid_args_t) == 24, "nvshmemx_uniqueid_args_t is 24 bytes");
static_assert(sizeof(nvshmemx_init_args_t) == 128, "nvshmemx_init_args_t is 128 bytes");
static_assert(sizeof(nvshmemx_init_attr_t) == 144, "nvshmemx_init_attr_t is 144 bytes");
#endif

#define NVSHMEMX_UNIQUEID_INITIALIZER {(1 << 16) + (int)sizeof(nvshmemx_uniqueid_t), {0}}
#define NVSHMEMX_UNIQUEID_ARGS_INITIALIZER {(1 << 16) + (int)sizeof(nvshmemx_uniqueid_args_t), NULL, -1, -1}
#define NVSHMEM_INIT_ARGS_V2_IDENTIFIER ((2 << 16) + (int)sizeof(nvshmemx_init_args_t))
#define NVSHMEM_INIT_ATTR_V2_IDENTIFIER ((2 << 16) + (int)sizeof(nvshmemx_init_attr_t))
#define NVSHMEM_INIT_ATTR_V1_IDENTIFIER ((1 << 16) + (int)sizeof(nvshmemx_init_attr_t))
#define NVSHMEMX_INIT_ATTR_INITIALIZER \
  {NVSHMEM_INIT_ATTR_V2_IDENTIFIER, NULL, {NVSHMEM_INIT_ARGS_V2_IDENTIFIER, NVSHMEMX_UNIQUEID_ARGS_INITIALIZER, -1, {0}}}

/* ---- teams ---- */
typedef int32_t nvshmem_team_t;
typedef nvshmem_team_t nvshmemx_team_t;
typedef uint64_t nvshmemx_team_uniqueid_t;

enum {
  NVSHMEM_TEAM_INVALID = -1,
  NVSHMEM_TEAM_WORLD = 0,
  NVSHMEM_TEAM_SHARED = 1,
  NVSHMEMX_TEAM_NODE = 2,
  NVSHMEMX_TEAM_SAME_MYPE_NODE = 3,
  NVSHMEMI_TEAM_SAME_GPU = 4,
  NVSHMEMI_TEAM_GPU_LEADERS = 5,
  NVSHMEM_TEAM_MC_SHARED = 6,
  NVSHMEM_TEAMS_MIN = 7
};

typedef struct {
  int version;
  int num_contexts;
  nvshmemx_team_uniqueid_t uniqueid;
  char padding[48];
} nvshmem_team_config_v2;
typedef nvshmem_team_config_v2 nvshmem_team_config_t;
#define NVSHMEM_TEAM_CONFIG_INITIALIZER {(2 << 16) + (int)sizeof(nvshmem_team_config_t), -1, ~0ull, {0}}
#define NVSHMEM_TEAM_CONFIG_MASK_NUM_CONTEXTS 0x0000000000000001
#define NVSHMEM_TEAM_CONFIG_MASK_UNIQUEID 0x0000000000000002

/* ---- constants ---- */
enum {
  NVSHMEM_THREAD_SINGLE = 0,
  NVSHMEM_THREAD_FUNNELED,
  NVSHMEM_THREAD_SERIALIZED,
  NVSHMEM_THREAD_MULTIPLE
};
enum {
  NVSHMEM_STATUS_NOT_INITIALIZED = 0,
  NVSHMEM_STATUS_IS_BOOTSTRAPPED,
  NVSHMEM_STATUS_IS_INITIALIZED,
  NVSHMEM_STATUS_LIMITED_MPG,
  NVSHMEM_STATUS_FULL_MPG
};
enum { NVSHMEM_CMP_EQ = 0, NVSHMEM_CMP_NE, NVSHMEM_CMP_GT, NVSHMEM_CMP_LE, NVSHMEM_CMP_LT, NVSHMEM_CMP_GE };
enum { NVSHMEM_SIGNAL_SET = 9, NVSHMEM_SIGNAL_ADD = 10 };
enum flags {
  NVSHMEMX_INIT_THREAD_PES = 1,
  NVSHMEMX_INIT_WITH_MPI_COMM = 1 << 1,
  NVSHMEMX_INIT_WITH_SHMEM = 1 << 2,
  NVSHMEMX_INIT_WITH_UNIQUEID = 1 << 3
};

/* ---- initialization ----
 * NVSHMEM's own entry point is C++: the inline functions below call it. */
int nvshmemi_init_thread(int requested_thread_support, int* provided_thread_support, unsigned int bootstrap_flags,
                         nvshmemx_init_attr_t* bootstrap_attr, nvshmemi_version_t);

#ifdef __cplusplus
extern "C" {
#endif

int nvshmemx_init_status(void);
void nvshmem_query_thread(int* provided);
void nvshmem_global_exit(int status);
void nvshmemi_finalize(void);
int nvshmemx_hostlib_init_attr(unsigned int flags, nvshmemx_init_attr_t* attr);
void nvshmemx_hostlib_finalize(void);
int nvshmemx_get_uniqueid(nvshmemx_uniqueid_t* uniqueid);
int nvshmemx_set_attr_uniqueid_args(const int myrank, const int nranks, const nvshmemx_uniqueid_t* uniqueid,
                                    nvshmemx_init_attr_t* attr);
int nvshmemx_set_attr_mpi_comm_args(void* mpi_comm, nvshmemx_init_attr_t* nvshmem_attr);

/* ---- who and where ---- */
int nvshmem_my_pe(void);
int nvshmem_n_pes(void);
void nvshmem_info_get_version(int* major, int* minor);
void nvshmem_info_get_name(char* name);
void nvshmemx_vendor_get_version_info(int* major, int* minor, int* patch);

/* ---- the symmetric heap ---- */
void* nvshmem_malloc(size_t size);
void* nvshmem_calloc(size_t count, size_t size);
void* nvshmem_align(size_t alignment, size_t size);
void nvshmem_free(void* ptr);
void* nvshmem_ptr(const void* dest, int pe);
void* nvshmemx_mc_ptr(nvshmem_team_t team, const void* ptr);
int nvshmemx_buffer_register(void* addr, size_t length);
int nvshmemx_buffer_unregister(void* addr);
void nvshmemx_buffer_unregister_all(void);

/* ---- remote memory access, blocking ---- */
void nvshmem_putmem(void* dest, const void* source, size_t bytes, int pe);
void nvshmem_getmem(void* dest, const void* source, size_t bytes, int pe);
void nvshmem_putmem_nbi(void* dest, const void* source, size_t bytes, int pe);
void nvshmem_getmem_nbi(void* dest, const void* source, size_t bytes, int pe);
#define VGPU_NVSHMEM_SIZED(BITS)                                                                         \
  void nvshmem_put##BITS(void* dest, const void* source, size_t nelems, int pe);                        \
  void nvshmem_get##BITS(void* dest, const void* source, size_t nelems, int pe);                        \
  void nvshmem_put##BITS##_nbi(void* dest, const void* source, size_t nelems, int pe);                  \
  void nvshmem_get##BITS##_nbi(void* dest, const void* source, size_t nelems, int pe);                  \
  void nvshmem_iput##BITS(void* dest, const void* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe); \
  void nvshmem_iget##BITS(void* dest, const void* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe); \
  void nvshmemx_put##BITS##_on_stream(void* dest, const void* source, size_t nelems, int pe, cudaStream_t s); \
  void nvshmemx_get##BITS##_on_stream(void* dest, const void* source, size_t nelems, int pe, cudaStream_t s);
VGPU_NVSHMEM_SIZED(8)
VGPU_NVSHMEM_SIZED(16)
VGPU_NVSHMEM_SIZED(32)
VGPU_NVSHMEM_SIZED(64)
VGPU_NVSHMEM_SIZED(128)
#undef VGPU_NVSHMEM_SIZED

/* Typed host RMA for the standard integer and floating types. */
#define VGPU_NVSHMEM_TYPED(NAME, TYPE)                                                                    \
  void nvshmem_##NAME##_put(TYPE* dest, const TYPE* source, size_t nelems, int pe);                      \
  void nvshmem_##NAME##_get(TYPE* dest, const TYPE* source, size_t nelems, int pe);                      \
  void nvshmem_##NAME##_p(TYPE* dest, const TYPE value, int pe);                                         \
  TYPE nvshmem_##NAME##_g(const TYPE* source, int pe);                                                   \
  void nvshmemx_##NAME##_put_on_stream(TYPE* dest, const TYPE* source, size_t nelems, int pe, cudaStream_t s); \
  void nvshmemx_##NAME##_get_on_stream(TYPE* dest, const TYPE* source, size_t nelems, int pe, cudaStream_t s); \
  void nvshmemx_##NAME##_p_on_stream(TYPE* dest, const TYPE value, int pe, cudaStream_t s);
#define VGPU_NVSHMEM_FOR_TYPES(X)                                                                         \
  X(float, float) X(double, double) X(char, char) X(short, short) X(schar, signed char) X(int, int)       \
  X(long, long) X(longlong, long long) X(uchar, unsigned char) X(ushort, unsigned short)                 \
  X(uint, unsigned int) X(ulong, unsigned long) X(ulonglong, unsigned long long) X(int8, int8_t)         \
  X(int16, int16_t) X(int32, int32_t) X(int64, int64_t) X(uint8, uint8_t) X(uint16, uint16_t)            \
  X(uint32, uint32_t) X(uint64, uint64_t) X(size, size_t) X(ptrdiff, ptrdiff_t)
VGPU_NVSHMEM_FOR_TYPES(VGPU_NVSHMEM_TYPED)
#undef VGPU_NVSHMEM_TYPED

/* ---- remote memory access in stream order ---- */
void nvshmemx_putmem_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t stream);
void nvshmemx_getmem_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t stream);
void nvshmemx_putmem_nbi_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t stream);
void nvshmemx_getmem_nbi_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t stream);
void nvshmemx_putmem_signal_on_stream(void* dest, const void* source, size_t bytes, uint64_t* sig_addr,
                                      uint64_t signal, int sig_op, int pe, cudaStream_t stream);
void nvshmemx_putmem_signal_nbi_on_stream(void* dest, const void* source, size_t bytes, uint64_t* sig_addr,
                                          uint64_t signal, int sig_op, int pe, cudaStream_t stream);
void nvshmemx_signal_op_on_stream(uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, cudaStream_t stream);
void nvshmemx_signal_wait_until_on_stream(uint64_t* sig_addr, int cmp, uint64_t cmp_value, cudaStream_t stream);
uint64_t nvshmem_signal_fetch(uint64_t* sig_addr);

/* ---- ordering and synchronization ---- */
void nvshmem_quiet(void);
void nvshmem_fence(void);
void nvshmemx_quiet_on_stream(cudaStream_t stream);
void nvshmemx_flush(void);
void nvshmemx_flush_on_stream(cudaStream_t stream);
void nvshmemx_signal_counted_reset(uint64_t* signal_addr);
void nvshmem_barrier_all(void);
void nvshmem_sync_all(void);
int nvshmem_barrier(nvshmem_team_t team);
int nvshmem_team_sync(nvshmem_team_t team);
void nvshmemx_barrier_all_on_stream(cudaStream_t stream);
void nvshmemx_sync_all_on_stream(cudaStream_t stream);
int nvshmemx_barrier_on_stream(nvshmem_team_t team, cudaStream_t stream);
int nvshmemx_team_sync_on_stream(nvshmem_team_t team, cudaStream_t stream);

/* ---- teams ---- */
int nvshmem_team_my_pe(nvshmem_team_t team);
int nvshmem_team_n_pes(nvshmem_team_t team);
int nvshmem_team_translate_pe(nvshmem_team_t src_team, int src_pe, nvshmem_team_t dest_team);
int nvshmem_team_split_strided(nvshmem_team_t parent_team, int start, int stride, int size,
                               const nvshmem_team_config_t* config, long config_mask, nvshmem_team_t* new_team);
int nvshmem_team_split_2d(nvshmem_team_t parent_team, int xrange, const nvshmem_team_config_t* xaxis_config,
                          long xaxis_mask, nvshmem_team_t* xaxis_team, const nvshmem_team_config_t* yaxis_config,
                          long yaxis_mask, nvshmem_team_t* yaxis_team);
void nvshmem_team_get_config(nvshmem_team_t team, nvshmem_team_config_t* config);
void nvshmem_team_destroy(nvshmem_team_t team);
int nvshmemx_team_get_uniqueid(nvshmemx_team_uniqueid_t* uniqueid);
int nvshmemx_team_init(nvshmem_team_t* team, nvshmem_team_config_t* config, long config_mask, int npes,
                       int pe_idx_in_team);

/* ---- collectives over the symmetric heap ---- */
int nvshmem_broadcastmem(nvshmem_team_t team, void* dest, const void* source, size_t nelems, int PE_root);
int nvshmem_fcollectmem(nvshmem_team_t team, void* dest, const void* source, size_t nelems);
int nvshmem_alltoallmem(nvshmem_team_t team, void* dest, const void* source, size_t nelems);
int nvshmemx_broadcastmem_on_stream(nvshmem_team_t team, void* dest, const void* source, size_t nelems,
                                    int PE_root, cudaStream_t stream);
int nvshmemx_fcollectmem_on_stream(nvshmem_team_t team, void* dest, const void* source, size_t nelems,
                                   cudaStream_t stream);
int nvshmemx_alltoallmem_on_stream(nvshmem_team_t team, void* dest, const void* source, size_t nelems,
                                   cudaStream_t stream);

/* ---- the rest of the typed host API (C++: half and bfloat16 are the CUDA types) ----
 * Nonblocking and strided RMA, signals, collectives, reductions, atomics and
 * waits in stream order, over NVIDIA's lists of types. */
#ifdef __cplusplus
}
#include <cuda_bf16.h>
#include <cuda_fp16.h>
extern "C" {

#define VGPU_NVSHMEM_RMA_TYPES(X) \
  VGPU_NVSHMEM_FOR_TYPES(X) X(bfloat16, __nv_bfloat16) X(half, __half)
#define VGPU_NVSHMEM_RMA_DECLS(NAME, TYPE)                                                                  \
  void nvshmem_##NAME##_put_nbi(TYPE* dest, const TYPE* source, size_t nelems, int pe);                    \
  void nvshmem_##NAME##_get_nbi(TYPE* dest, const TYPE* source, size_t nelems, int pe);                    \
  void nvshmem_##NAME##_iput(TYPE* dest, const TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems,   \
                             int pe);                                                                       \
  void nvshmem_##NAME##_iget(TYPE* dest, const TYPE* source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems,   \
                             int pe);                                                                       \
  void nvshmemx_##NAME##_put_nbi_on_stream(TYPE* dest, const TYPE* source, size_t nelems, int pe,          \
                                           cudaStream_t s);                                                 \
  void nvshmemx_##NAME##_get_nbi_on_stream(TYPE* dest, const TYPE* source, size_t nelems, int pe,          \
                                           cudaStream_t s);                                                 \
  void nvshmemx_##NAME##_iput_on_stream(TYPE* dest, const TYPE* source, ptrdiff_t dst, ptrdiff_t sst,       \
                                        size_t nelems, int pe, cudaStream_t s);                             \
  void nvshmemx_##NAME##_iget_on_stream(TYPE* dest, const TYPE* source, ptrdiff_t dst, ptrdiff_t sst,       \
                                        size_t nelems, int pe, cudaStream_t s);                             \
  TYPE nvshmemx_##NAME##_g_on_stream(const TYPE* source, int pe, cudaStream_t s);                          \
  void nvshmemx_##NAME##_put_signal_on_stream(TYPE* dest, const TYPE* source, size_t nelems,                \
                                              uint64_t* sig_addr, uint64_t signal, int sig_op, int pe,      \
                                              cudaStream_t s);                                              \
  void nvshmemx_##NAME##_put_signal_nbi_on_stream(TYPE* dest, const TYPE* source, size_t nelems,            \
                                                  uint64_t* sig_addr, uint64_t signal, int sig_op, int pe,  \
                                                  cudaStream_t s);                                          \
  int nvshmem_##NAME##_broadcast(nvshmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems,        \
                                 int PE_root);                                                              \
  int nvshmem_##NAME##_fcollect(nvshmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems);        \
  int nvshmem_##NAME##_alltoall(nvshmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems);        \
  int nvshmemx_##NAME##_broadcast_on_stream(nvshmem_team_t team, TYPE* dest, const TYPE* source,            \
                                            size_t nelems, int PE_root, cudaStream_t s);                    \
  int nvshmemx_##NAME##_fcollect_on_stream(nvshmem_team_t team, TYPE* dest, const TYPE* source,             \
                                           size_t nelems, cudaStream_t s);                                  \
  int nvshmemx_##NAME##_alltoall_on_stream(nvshmem_team_t team, TYPE* dest, const TYPE* source,             \
                                           size_t nelems, cudaStream_t s);
VGPU_NVSHMEM_RMA_TYPES(VGPU_NVSHMEM_RMA_DECLS)
#undef VGPU_NVSHMEM_RMA_DECLS

/* half and bfloat16 also have the base forms. */
#define VGPU_NVSHMEM_TYPED(NAME, TYPE)                                                                    \
  void nvshmem_##NAME##_put(TYPE* dest, const TYPE* source, size_t nelems, int pe);                      \
  void nvshmem_##NAME##_get(TYPE* dest, const TYPE* source, size_t nelems, int pe);                      \
  void nvshmem_##NAME##_p(TYPE* dest, const TYPE value, int pe);                                         \
  TYPE nvshmem_##NAME##_g(const TYPE* source, int pe);                                                   \
  void nvshmemx_##NAME##_put_on_stream(TYPE* dest, const TYPE* source, size_t nelems, int pe, cudaStream_t s); \
  void nvshmemx_##NAME##_get_on_stream(TYPE* dest, const TYPE* source, size_t nelems, int pe, cudaStream_t s); \
  void nvshmemx_##NAME##_p_on_stream(TYPE* dest, const TYPE value, int pe, cudaStream_t s);
VGPU_NVSHMEM_TYPED(bfloat16, __nv_bfloat16)
VGPU_NVSHMEM_TYPED(half, __half)
#undef VGPU_NVSHMEM_TYPED

#define VGPU_NVSHMEM_SIZED_MORE(BITS)                                                                      \
  void nvshmemx_put##BITS##_nbi_on_stream(void* dest, const void* source, size_t nelems, int pe,          \
                                          cudaStream_t s);                                                 \
  void nvshmemx_get##BITS##_nbi_on_stream(void* dest, const void* source, size_t nelems, int pe,          \
                                          cudaStream_t s);                                                 \
  void nvshmemx_iput##BITS##_on_stream(void* dest, const void* source, ptrdiff_t dst, ptrdiff_t sst,       \
                                       size_t nelems, int pe, cudaStream_t s);                             \
  void nvshmemx_iget##BITS##_on_stream(void* dest, const void* source, ptrdiff_t dst, ptrdiff_t sst,       \
                                       size_t nelems, int pe, cudaStream_t s);                             \
  void nvshmemx_put##BITS##_signal_on_stream(void* dest, const void* source, size_t nelems,                \
                                             uint64_t* sig_addr, uint64_t signal, int sig_op, int pe,      \
                                             cudaStream_t s);                                              \
  void nvshmemx_put##BITS##_signal_nbi_on_stream(void* dest, const void* source, size_t nelems,            \
                                                 uint64_t* sig_addr, uint64_t signal, int sig_op, int pe,  \
                                                 cudaStream_t s);
VGPU_NVSHMEM_SIZED_MORE(8)
VGPU_NVSHMEM_SIZED_MORE(16)
VGPU_NVSHMEM_SIZED_MORE(32)
VGPU_NVSHMEM_SIZED_MORE(64)
VGPU_NVSHMEM_SIZED_MORE(128)
#undef VGPU_NVSHMEM_SIZED_MORE

#define VGPU_NVSHMEM_BITWISE_TYPES(X, OP)                                                                  \
  X(uchar, unsigned char, OP) X(ushort, unsigned short, OP) X(uint, unsigned int, OP)                      \
  X(ulong, unsigned long, OP) X(ulonglong, unsigned long long, OP) X(int8, int8_t, OP) X(int16, int16_t, OP) \
  X(int32, int32_t, OP) X(int64, int64_t, OP) X(uint8, uint8_t, OP) X(uint16, uint16_t, OP)                \
  X(uint32, uint32_t, OP) X(uint64, uint64_t, OP) X(size, size_t, OP)
#define VGPU_NVSHMEM_STANDARD_TYPES(X, OP)                                                                 \
  VGPU_NVSHMEM_BITWISE_TYPES(X, OP)                                                                        \
  X(char, char, OP) X(schar, signed char, OP) X(short, short, OP) X(int, int, OP) X(long, long, OP)        \
  X(longlong, long long, OP) X(bfloat16, __nv_bfloat16, OP) X(half, __half, OP) X(float, float, OP)        \
  X(double, double, OP)
#define VGPU_NVSHMEM_REDUCE_DECLS(NAME, TYPE, OP)                                                          \
  int nvshmem_##NAME##_##OP##_reduce(nvshmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce); \
  int nvshmem_##NAME##_##OP##_reducescatter(nvshmem_team_t team, TYPE* dest, const TYPE* source,           \
                                            size_t nreduce);                                               \
  int nvshmemx_##NAME##_##OP##_reduce_on_stream(nvshmem_team_t team, TYPE* dest, const TYPE* source,       \
                                                size_t nreduce, cudaStream_t s);                           \
  int nvshmemx_##NAME##_##OP##_reducescatter_on_stream(nvshmem_team_t team, TYPE* dest, const TYPE* source, \
                                                       size_t nreduce, cudaStream_t s);
VGPU_NVSHMEM_BITWISE_TYPES(VGPU_NVSHMEM_REDUCE_DECLS, and)
VGPU_NVSHMEM_BITWISE_TYPES(VGPU_NVSHMEM_REDUCE_DECLS, or)
VGPU_NVSHMEM_BITWISE_TYPES(VGPU_NVSHMEM_REDUCE_DECLS, xor)
VGPU_NVSHMEM_STANDARD_TYPES(VGPU_NVSHMEM_REDUCE_DECLS, max)
VGPU_NVSHMEM_STANDARD_TYPES(VGPU_NVSHMEM_REDUCE_DECLS, min)
VGPU_NVSHMEM_STANDARD_TYPES(VGPU_NVSHMEM_REDUCE_DECLS, sum)
VGPU_NVSHMEM_STANDARD_TYPES(VGPU_NVSHMEM_REDUCE_DECLS, prod)
#undef VGPU_NVSHMEM_REDUCE_DECLS

#define VGPU_NVSHMEM_AMO_BITWISE_TYPES(X) \
  X(uint, unsigned int) X(ulong, unsigned long) X(ulonglong, unsigned long long) X(int32, int32_t) \
  X(uint32, uint32_t) X(int64, int64_t) X(uint64, uint64_t)
#define VGPU_NVSHMEM_AMO_STANDARD_TYPES(X) \
  X(int, int) X(long, long) X(longlong, long long) X(size, size_t) X(ptrdiff, ptrdiff_t)
#define VGPU_NVSHMEM_AMO_EXTENDED_TYPES(X) X(half, __half) X(float, float) X(double, double)
#define VGPU_NVSHMEM_AMO_INT_DECLS(NAME, TYPE)                                                             \
  void nvshmem_##NAME##_atomic_inc(TYPE* dest, int pe);                                                    \
  TYPE nvshmem_##NAME##_atomic_fetch_inc(TYPE* dest, int pe);                                              \
  void nvshmem_##NAME##_atomic_add(TYPE* dest, TYPE value, int pe);                                        \
  TYPE nvshmem_##NAME##_atomic_fetch_add(TYPE* dest, TYPE value, int pe);                                  \
  TYPE nvshmem_##NAME##_atomic_compare_swap(TYPE* dest, TYPE cond, TYPE value, int pe);
#define VGPU_NVSHMEM_AMO_ALL_DECLS(NAME, TYPE)                                                             \
  TYPE nvshmem_##NAME##_atomic_fetch(const TYPE* dest, int pe);                                            \
  void nvshmem_##NAME##_atomic_set(TYPE* dest, TYPE value, int pe);                                        \
  TYPE nvshmem_##NAME##_atomic_swap(TYPE* dest, TYPE value, int pe);
#define VGPU_NVSHMEM_AMO_BIT_DECLS(NAME, TYPE)                                                             \
  void nvshmem_##NAME##_atomic_and(TYPE* dest, TYPE value, int pe);                                        \
  void nvshmem_##NAME##_atomic_or(TYPE* dest, TYPE value, int pe);                                         \
  void nvshmem_##NAME##_atomic_xor(TYPE* dest, TYPE value, int pe);                                        \
  TYPE nvshmem_##NAME##_atomic_fetch_and(TYPE* dest, TYPE value, int pe);                                  \
  TYPE nvshmem_##NAME##_atomic_fetch_or(TYPE* dest, TYPE value, int pe);                                   \
  TYPE nvshmem_##NAME##_atomic_fetch_xor(TYPE* dest, TYPE value, int pe);
#define VGPU_NVSHMEM_AMO_XADD_DECLS(NAME, TYPE)                                                            \
  void nvshmemx_##NAME##_atomic_add(TYPE* dest, TYPE value, int pe);                                       \
  TYPE nvshmemx_##NAME##_atomic_fetch_add(TYPE* dest, TYPE value, int pe);
VGPU_NVSHMEM_AMO_BITWISE_TYPES(VGPU_NVSHMEM_AMO_INT_DECLS)
VGPU_NVSHMEM_AMO_STANDARD_TYPES(VGPU_NVSHMEM_AMO_INT_DECLS)
VGPU_NVSHMEM_AMO_BITWISE_TYPES(VGPU_NVSHMEM_AMO_ALL_DECLS)
VGPU_NVSHMEM_AMO_STANDARD_TYPES(VGPU_NVSHMEM_AMO_ALL_DECLS)
VGPU_NVSHMEM_AMO_EXTENDED_TYPES(VGPU_NVSHMEM_AMO_ALL_DECLS)
VGPU_NVSHMEM_AMO_BITWISE_TYPES(VGPU_NVSHMEM_AMO_BIT_DECLS)
VGPU_NVSHMEM_AMO_EXTENDED_TYPES(VGPU_NVSHMEM_AMO_XADD_DECLS)
#undef VGPU_NVSHMEM_AMO_INT_DECLS
#undef VGPU_NVSHMEM_AMO_ALL_DECLS
#undef VGPU_NVSHMEM_AMO_BIT_DECLS
#undef VGPU_NVSHMEM_AMO_XADD_DECLS

#define VGPU_NVSHMEM_WAIT_TYPES(X)                                                                         \
  X(short, short) X(int, int) X(long, long) X(longlong, long long) X(ushort, unsigned short)               \
  X(uint, unsigned int) X(ulong, unsigned long) X(ulonglong, unsigned long long) X(int32, int32_t)        \
  X(int64, int64_t) X(uint32, uint32_t) X(uint64, uint64_t) X(size, size_t) X(ptrdiff, ptrdiff_t)
#define VGPU_NVSHMEM_WAIT_DECLS(NAME, TYPE)                                                                \
  void nvshmemx_##NAME##_wait_until_on_stream(TYPE* ivar, int cmp, TYPE cmp_value, cudaStream_t s);        \
  void nvshmemx_##NAME##_wait_until_all_on_stream(TYPE* ivars, size_t nelems, const int* status, int cmp,  \
                                                  TYPE cmp_value, cudaStream_t s);                         \
  void nvshmemx_##NAME##_wait_until_all_vector_on_stream(TYPE* ivars, size_t nelems, const int* status,    \
                                                         int cmp, TYPE* cmp_values, cudaStream_t s);
VGPU_NVSHMEM_WAIT_TYPES(VGPU_NVSHMEM_WAIT_DECLS)
#undef VGPU_NVSHMEM_WAIT_DECLS
#endif /* __cplusplus */

/* ---- kernels that use NVSHMEM's device API ---- */
int nvshmemx_collective_launch(const void* func, dim3 gridDims, dim3 blockDims, void** args, size_t sharedMem,
                               cudaStream_t stream);
typedef struct nvshmemx_collective_launch_attr {
  cudaLaunchConfig_t cuda_config;
} nvshmemx_collective_launch_attr_t;
int nvshmemx_collective_launch_attr(const nvshmemx_collective_launch_attr_t* attr, const void* func, void** args);
int nvshmemx_collective_launch_query_gridsize(const void* func, dim3 blockDims, void** args, size_t sharedMem,
                                              int* gridsize);

#ifdef __cplusplus
}
#endif

#define VGPU_NVSHMEM_VERSION_ARG                                                                \
  nvshmemi_version_t app_version = {NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, \
                                    NVSHMEM_VENDOR_PATCH_VERSION}

static inline void nvshmem_init(void) {
  int provided = 0;
  VGPU_NVSHMEM_VERSION_ARG;
  if (nvshmemi_init_thread(NVSHMEM_THREAD_SERIALIZED, &provided, 0, NULL, app_version) != 0) {
    fprintf(stderr, "nvshmem_init failed\n");
    exit(-1);
  }
}

static inline int nvshmem_init_thread(int requested, int* provided) {
  VGPU_NVSHMEM_VERSION_ARG;
  return nvshmemi_init_thread(requested, provided, 0, NULL, app_version);
}

static inline int nvshmemx_init_attr(unsigned int flags, nvshmemx_init_attr_t* attributes) {
  int provided = 0;
  VGPU_NVSHMEM_VERSION_ARG;
  return nvshmemi_init_thread(NVSHMEM_THREAD_SERIALIZED, &provided, flags, attributes, app_version);
}

static inline void nvshmem_finalize(void) { nvshmemi_finalize(); }

#endif /* VGPU_NVSHMEM_H_ */
