/*
 * A cuTensorNet distributed communication library over MPI, for the tests of
 * distributed execution (cutensornet_mpi_paths.cpp).
 *
 * cuTensorNet (NVIDIA's, and VirtualGPU's) reads $CUTENSORNET_COMM_LIB, opens
 * that shared library and uses the table `cutensornetCommInterface` it
 * exports: the wrappers below are those of the sample cuQuantum ships
 * (distributed_interfaces/cutensornet_distributed_interface_mpi.c,
 * BSD-3-Clause, Copyright NVIDIA CORPORATION & AFFILIATES), the interface
 * itself documented in cutensornet/typesDistributed.h. What this file adds is
 * a call log: every call the library makes through the table is recorded
 * (name, element count, data type) and `cutn_comm_log_get` returns the record,
 * so a test can say which primitive a library call used and with what.
 * The log is the only thing a test can observe of what the library does
 * between its ranks, and it is what NVIDIA's library does that the simulator's
 * follows.
 *
 * Build: mpicc -shared -fPIC -std=gnu99 cutn_comm_mpi.c -o libcutn_comm_mpi.so
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/vgpu_cutensornet.h"

#define LOG_LINES 4096
static char g_log[LOG_LINES][64];
static int g_log_n = 0;

static void note(const char* name, int count, int dtype, int extra) {
  if (g_log_n < LOG_LINES) snprintf(g_log[g_log_n++], 64, "%s count=%d dtype=%d x=%d", name, count, dtype, extra);
}

/*
 * cuTensorNet hands the primitives device pointers (an MPI that is not
 * CUDA-aware, such as the Open MPI of a distribution, faults on them inside
 * MPI_Allreduce), so a buffer in device memory is staged through the host.
 * The CUDA runtime is reached through dlsym, as the process already has one.
 */
typedef int (*ptr_attr_t)(void*, const void*);
typedef int (*memcpy_t)(void*, const void*, size_t, int);

static int on_device(const void* p) {
  static ptr_attr_t attr;
  static int bound;
  if (!bound) {
    attr = (ptr_attr_t)dlsym(RTLD_DEFAULT, "cudaPointerGetAttributes");
    bound = 1;
  }
  if (!attr || !p) return 0;
  struct {  /* cudaPointerAttributes: type first (CUDA 11 and later) */
    int type;
    int device;
    void* devicePointer;
    void* hostPointer;
    long reserved[8];
  } a;
  memset(&a, 0, sizeof a);
  if (attr(&a, p) != 0) return 0;
  return a.type == 2; /* cudaMemoryTypeDevice */
}

static int copy(void* dst, const void* src, size_t n, int kind) {
  static memcpy_t mc;
  if (!mc) mc = (memcpy_t)dlsym(RTLD_DEFAULT, "cudaMemcpy");
  return mc ? mc(dst, src, n, kind) : 1;
}

static size_t type_bytes(cudaDataType_t t) {
  switch (t) {
    case CUDA_R_8I: return 1;
    case CUDA_R_16I: return 2;
    case CUDA_R_32I: case CUDA_R_32F: return 4;
    case CUDA_R_64I: case CUDA_R_64F: case CUDA_C_32F: return 8;
    case CUDA_C_64F: return 16;
    default: return 0;
  }
}

/*
 * Test hooks. cutn_comm_fail("<name>") makes the primitive of that name (as in
 * the log: Barrier, Bcast, AllreduceInPlace, getProcRank, ...) return 1
 * without communicating, until cutn_comm_fail(NULL); CUTN_COMM_VERSION=<n>
 * sets the version field of the table.
 */
static const char* g_fail = NULL;
void cutn_comm_fail(const char* name) { g_fail = name; }
static int failing(const char* name) { return g_fail && strcmp(g_fail, name) == 0; }

int cutn_comm_log_size(void) { return g_log_n; }
const char* cutn_comm_log_line(int i) { return i >= 0 && i < g_log_n ? g_log[i] : ""; }
void cutn_comm_log_clear(void) { g_log_n = 0; }

static MPI_Datatype to_mpi(const cudaDataType_t t) {
  switch (t) {
    case CUDA_R_8I: return MPI_INT8_T;
    case CUDA_R_16I: return MPI_INT16_T;
    case CUDA_R_32I: return MPI_INT32_T;
    case CUDA_R_64I: return MPI_INT64_T;
    case CUDA_R_32F: return MPI_FLOAT;
    case CUDA_R_64F: return MPI_DOUBLE;
    case CUDA_C_32F: return MPI_C_FLOAT_COMPLEX;
    case CUDA_C_64F: return MPI_C_DOUBLE_COMPLEX;
    default:
      fprintf(stderr, "cutn_comm_mpi: unknown CUDA data type %d\n", (int)t);
      exit(EXIT_FAILURE);
  }
}

static MPI_Comm unpack(const cutensornetDistributedCommunicator_t* c) {
  if (c->commPtr == NULL) return MPI_COMM_NULL;
  if (sizeof(MPI_Comm) != c->commSize) {
    fprintf(stderr, "cutn_comm_mpi: MPI_Comm has an unexpected size %zu\n", c->commSize);
    exit(EXIT_FAILURE);
  }
  return *((MPI_Comm*)(c->commPtr));
}

static int comm_size(const cutensornetDistributedCommunicator_t* c, int32_t* n) {
  note("getNumRanks", 0, 0, 0);
  if (failing("getNumRanks")) return 1;
  int k = 0;
  const int e = MPI_Comm_size(unpack(c), &k);
  *n = k;
  return e;
}

static int comm_size_shared(const cutensornetDistributedCommunicator_t* c, int32_t* n) {
  note("getNumRanksShared", 0, 0, 0);
  if (failing("getNumRanksShared")) return 1;
  *n = 0;
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "mpi_hw_resource_type", "mpi_shared_memory");
  int r = -1;
  int e = MPI_Comm_rank(unpack(c), &r);
  if (e == MPI_SUCCESS) {
    MPI_Comm local;
    e = MPI_Comm_split_type(unpack(c), MPI_COMM_TYPE_SHARED, r, info, &local);
    if (e == MPI_SUCCESS) {
      int k = 0;
      MPI_Comm_size(local, &k);
      *n = k;
      MPI_Comm_free(&local);
    }
  }
  MPI_Info_free(&info);
  return e;
}

static int comm_rank(const cutensornetDistributedCommunicator_t* c, int32_t* r) {
  note("getProcRank", 0, 0, 0);
  if (failing("getProcRank")) return 1;
  int k = -1;
  const int e = MPI_Comm_rank(unpack(c), &k);
  *r = k;
  return e;
}

static int comm_barrier(const cutensornetDistributedCommunicator_t* c) {
  note("Barrier", 0, 0, 0);
  if (failing("Barrier")) return 1;
  return MPI_Barrier(unpack(c));
}

/* A buffer as MPI sees it: the caller's, or a host copy of device memory. */
typedef struct {
  void* user;
  void* mpi;
  size_t bytes;
  int staged;
} buf_t;

static buf_t take(const void* p, size_t bytes, int copy_in) {
  buf_t b = {(void*)p, (void*)p, bytes, 0};
  if (bytes && on_device(p)) {
    b.mpi = malloc(bytes);
    b.staged = 1;
    if (copy_in) copy(b.mpi, p, bytes, 2 /* cudaMemcpyDeviceToHost */);
  }
  return b;
}

static void give(buf_t* b, int copy_out) {
  if (!b->staged) return;
  if (copy_out) copy(b->user, b->mpi, b->bytes, 1 /* cudaMemcpyHostToDevice */);
  free(b->mpi);
}

static int comm_bcast(const cutensornetDistributedCommunicator_t* c, void* p, int32_t n, cudaDataType_t t, int32_t root) {
  if (failing("Bcast")) return note("Bcast", n, (int)t, 0), 1;
  const MPI_Datatype mt = to_mpi(t);
  buf_t b = take(p, (size_t)n * type_bytes(t), 1);
  note("Bcast", n, (int)t, b.staged);
  const int e = MPI_Bcast(b.mpi, (int)n, mt, (int)root, unpack(c));
  give(&b, 1);
  return e;
}

static int comm_allreduce(const cutensornetDistributedCommunicator_t* c, const void* in, void* out, int32_t n, cudaDataType_t t) {
  if (failing("Allreduce")) return note("Allreduce", n, (int)t, 0), 1;
  const MPI_Datatype mt = to_mpi(t);
  buf_t i = take(in, (size_t)n * type_bytes(t), 1), o = take(out, (size_t)n * type_bytes(t), 0);
  note("Allreduce", n, (int)t, i.staged);
  const int e = MPI_Allreduce(i.mpi, o.mpi, (int)n, mt, MPI_SUM, unpack(c));
  give(&i, 0);
  give(&o, 1);
  return e;
}

static int allreduce_in_place(const cutensornetDistributedCommunicator_t* c, void* p, int32_t n, cudaDataType_t t,
                              MPI_Op op, const char* name) {
  if (failing(name)) return note(name, n, (int)t, 0), 1;
  const MPI_Datatype mt = to_mpi(t);
  buf_t b = take(p, (size_t)n * type_bytes(t), 1);
  note(name, n, (int)t, b.staged);
  const int e = MPI_Allreduce(MPI_IN_PLACE, b.mpi, (int)n, mt, op, unpack(c));
  give(&b, 1);
  return e;
}

static int comm_allreduce_inplace(const cutensornetDistributedCommunicator_t* c, void* p, int32_t n, cudaDataType_t t) {
  return allreduce_in_place(c, p, n, t, MPI_SUM, "AllreduceInPlace");
}

static int comm_allreduce_inplace_min(const cutensornetDistributedCommunicator_t* c, void* p, int32_t n, cudaDataType_t t) {
  return allreduce_in_place(c, p, n, t, MPI_MIN, "AllreduceInPlaceMin");
}

static int comm_allreduce_minloc(const cutensornetDistributedCommunicator_t* c, const void* in, void* out) {
  if (failing("AllreduceDoubleIntMinloc")) return note("AllreduceDoubleIntMinloc", 1, 0, 0), 1;
  buf_t i = take(in, sizeof(struct { double d; int i; }), 1), o = take(out, sizeof(struct { double d; int i; }), 0);
  note("AllreduceDoubleIntMinloc", 1, 0, i.staged);
  const int e = MPI_Allreduce(i.mpi, o.mpi, 1, MPI_DOUBLE_INT, MPI_MINLOC, unpack(c));
  give(&i, 0);
  give(&o, 1);
  return e;
}

static int comm_allgather(const cutensornetDistributedCommunicator_t* c, const void* in, void* out, int32_t n, cudaDataType_t t) {
  const MPI_Datatype mt = to_mpi(t);
  if (failing("Allgather")) return note("Allgather", n, (int)t, 0), 1;
  int np = 1;
  MPI_Comm_size(unpack(c), &np);
  buf_t i = take(in, (size_t)n * type_bytes(t), 1), o = take(out, (size_t)n * type_bytes(t) * (size_t)np, 0);
  note("Allgather", n, (int)t, i.staged);
  const int e = MPI_Allgather(i.mpi, (int)n, mt, o.mpi, (int)n, mt, unpack(c));
  give(&i, 0);
  give(&o, 1);
  return e;
}

cutensornetDistributedInterface_t cutensornetCommInterface;

__attribute__((constructor)) static void init_table(void) {
  const char* v = getenv("CUTN_COMM_VERSION");
  cutensornetCommInterface = (cutensornetDistributedInterface_t){
      v ? atoi(v) : CUTENSORNET_DISTRIBUTED_INTERFACE_VERSION,
      comm_size,
      comm_size_shared,
      comm_rank,
      comm_barrier,
      comm_bcast,
      comm_allreduce,
      comm_allreduce_inplace,
      comm_allreduce_inplace_min,
      comm_allreduce_minloc,
      comm_allgather};
}
