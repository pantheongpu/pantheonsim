// The typed host API of NVSHMEM that nvshmem_api.cpp does not spell out: every
// type of the standard RMA list for nonblocking, strided and stream-ordered
// puts and gets, signals, collectives and reductions; the atomics; and waiting
// in stream order. Included inside extern "C" in nvshmem_api.cpp, after
// nvshmem_typed_impl.inc. The type lists and the names are those of NVIDIA's
// public headers (host/nvshmem_api.h, nvshmem_coll_api.h, nvshmemx_api.h).
//
// Derived from the documentation: a job with a symmetric heap does not run on
// the RTX 3060 pair, so none of this is compared with NVIDIA's library.

// The standard RMA types (25) -- the shim's list of 23, plus half and bfloat16.
#define VGPU_NV_RMA_TYPES(X) VGPU_NVSHMEM_FOR_TYPES(X) X(bfloat16, __nv_bfloat16) X(half, __half)

// ---- the two types the base list leaves out ----
VGPU_TYPED(bfloat16, __nv_bfloat16)
VGPU_TYPED(half, __half)

// ---- nonblocking, strided, stream-ordered ----
#define VGPU_TYPED_MORE(NAME, TYPE)                                                                            \
  NVSHMEM_EXPORT void nvshmem_##NAME##_put_nbi(TYPE* d, const TYPE* src, size_t n, int pe) {                  \
    put(d, src, n * sizeof(TYPE), pe, "nvshmem_" #NAME "_put_nbi");                                           \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_##NAME##_get_nbi(TYPE* d, const TYPE* src, size_t n, int pe) {                  \
    get(d, src, n * sizeof(TYPE), pe, "nvshmem_" #NAME "_get_nbi");                                           \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_##NAME##_iput(TYPE* d, const TYPE* src, ptrdiff_t dst, ptrdiff_t sst, size_t n,  \
                                            int pe) {                                                          \
    iput(d, src, dst, sst, n, sizeof(TYPE), pe, true);                                                         \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_##NAME##_iget(TYPE* d, const TYPE* src, ptrdiff_t dst, ptrdiff_t sst, size_t n,  \
                                            int pe) {                                                          \
    iput(d, src, dst, sst, n, sizeof(TYPE), pe, false);                                                        \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_put_nbi_on_stream(TYPE* d, const TYPE* src, size_t n, int pe,         \
                                                          cudaStream_t s) {                                    \
    nvshmemx_putmem_on_stream(d, src, n * sizeof(TYPE), pe, s);                                                \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_get_nbi_on_stream(TYPE* d, const TYPE* src, size_t n, int pe,         \
                                                          cudaStream_t s) {                                    \
    nvshmemx_getmem_on_stream(d, src, n * sizeof(TYPE), pe, s);                                                \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_iput_on_stream(TYPE* d, const TYPE* src, ptrdiff_t dst, ptrdiff_t sst,  \
                                                       size_t n, int pe, cudaStream_t s) {                     \
    in_stream_order(s, [=] { iput(d, src, dst, sst, n, sizeof(TYPE), pe, true); });                            \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_iget_on_stream(TYPE* d, const TYPE* src, ptrdiff_t dst, ptrdiff_t sst,  \
                                                       size_t n, int pe, cudaStream_t s) {                     \
    in_stream_order(s, [=] { iput(d, src, dst, sst, n, sizeof(TYPE), pe, false); });                           \
  }                                                                                                            \
  NVSHMEM_EXPORT TYPE nvshmemx_##NAME##_g_on_stream(const TYPE* src, int pe, cudaStream_t s) {                 \
    TYPE v = zero_of<TYPE>();                                                                                  \
    /* the value is the return value: the stream is waited for */                                              \
    cuda_ok(cudaStreamSynchronize(s), "cudaStreamSynchronize");                                                \
    get(&v, src, sizeof(TYPE), pe, "nvshmemx_" #NAME "_g_on_stream");                                          \
    return v;                                                                                                  \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_put_signal_on_stream(TYPE* d, const TYPE* src, size_t n,              \
                                                             uint64_t* sig_addr, uint64_t signal, int sig_op,  \
                                                             int pe, cudaStream_t s) {                         \
    nvshmemx_putmem_signal_on_stream(d, src, n * sizeof(TYPE), sig_addr, signal, sig_op, pe, s);               \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_put_signal_nbi_on_stream(TYPE* d, const TYPE* src, size_t n,          \
                                                                 uint64_t* sig_addr, uint64_t signal,          \
                                                                 int sig_op, int pe, cudaStream_t s) {         \
    nvshmemx_putmem_signal_on_stream(d, src, n * sizeof(TYPE), sig_addr, signal, sig_op, pe, s);               \
  }
VGPU_NV_RMA_TYPES(VGPU_TYPED_MORE)
#undef VGPU_TYPED_MORE

// Sized: the stream-ordered forms the base list lacks.
#define VGPU_SIZED_MORE(BITS)                                                                                  \
  NVSHMEM_EXPORT void nvshmemx_put##BITS##_nbi_on_stream(void* d, const void* src, size_t n, int pe,          \
                                                         cudaStream_t s) {                                     \
    nvshmemx_putmem_on_stream(d, src, n * (BITS / 8), pe, s);                                                  \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_get##BITS##_nbi_on_stream(void* d, const void* src, size_t n, int pe,          \
                                                         cudaStream_t s) {                                     \
    nvshmemx_getmem_on_stream(d, src, n * (BITS / 8), pe, s);                                                  \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_iput##BITS##_on_stream(void* d, const void* src, ptrdiff_t dst, ptrdiff_t sst,  \
                                                      size_t n, int pe, cudaStream_t s) {                      \
    in_stream_order(s, [=] { iput(d, src, dst, sst, n, BITS / 8, pe, true); });                                \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_iget##BITS##_on_stream(void* d, const void* src, ptrdiff_t dst, ptrdiff_t sst,  \
                                                      size_t n, int pe, cudaStream_t s) {                      \
    in_stream_order(s, [=] { iput(d, src, dst, sst, n, BITS / 8, pe, false); });                               \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_put##BITS##_signal_on_stream(void* d, const void* src, size_t n,               \
                                                            uint64_t* sig_addr, uint64_t signal, int sig_op,   \
                                                            int pe, cudaStream_t s) {                          \
    nvshmemx_putmem_signal_on_stream(d, src, n * (BITS / 8), sig_addr, signal, sig_op, pe, s);                 \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_put##BITS##_signal_nbi_on_stream(void* d, const void* src, size_t n,           \
                                                                uint64_t* sig_addr, uint64_t signal,           \
                                                                int sig_op, int pe, cudaStream_t s) {          \
    nvshmemx_putmem_signal_on_stream(d, src, n * (BITS / 8), sig_addr, signal, sig_op, pe, s);                 \
  }
VGPU_SIZED_MORE(8)
VGPU_SIZED_MORE(16)
VGPU_SIZED_MORE(32)
VGPU_SIZED_MORE(64)
VGPU_SIZED_MORE(128)
#undef VGPU_SIZED_MORE

// ---- collectives, one function per type over the byte-wise ones ----
#define VGPU_COLL(NAME, TYPE)                                                                                  \
  NVSHMEM_EXPORT int nvshmem_##NAME##_broadcast(nvshmem_team_t team, TYPE* d, const TYPE* src, size_t n,      \
                                                int root) {                                                    \
    return nvshmem_broadcastmem(team, d, src, n * sizeof(TYPE), root);                                         \
  }                                                                                                            \
  NVSHMEM_EXPORT int nvshmem_##NAME##_fcollect(nvshmem_team_t team, TYPE* d, const TYPE* src, size_t n) {      \
    return nvshmem_fcollectmem(team, d, src, n * sizeof(TYPE));                                                \
  }                                                                                                            \
  NVSHMEM_EXPORT int nvshmem_##NAME##_alltoall(nvshmem_team_t team, TYPE* d, const TYPE* src, size_t n) {      \
    return nvshmem_alltoallmem(team, d, src, n * sizeof(TYPE));                                                \
  }                                                                                                            \
  NVSHMEM_EXPORT int nvshmemx_##NAME##_broadcast_on_stream(nvshmem_team_t team, TYPE* d, const TYPE* src,     \
                                                           size_t n, int root, cudaStream_t s) {               \
    return nvshmemx_broadcastmem_on_stream(team, d, src, n * sizeof(TYPE), root, s);                           \
  }                                                                                                            \
  NVSHMEM_EXPORT int nvshmemx_##NAME##_fcollect_on_stream(nvshmem_team_t team, TYPE* d, const TYPE* src,      \
                                                          size_t n, cudaStream_t s) {                          \
    return nvshmemx_fcollectmem_on_stream(team, d, src, n * sizeof(TYPE), s);                                  \
  }                                                                                                            \
  NVSHMEM_EXPORT int nvshmemx_##NAME##_alltoall_on_stream(nvshmem_team_t team, TYPE* d, const TYPE* src,      \
                                                          size_t n, cudaStream_t s) {                          \
    return nvshmemx_alltoallmem_on_stream(team, d, src, n * sizeof(TYPE), s);                                  \
  }
VGPU_NV_RMA_TYPES(VGPU_COLL)
#undef VGPU_COLL

// ---- reductions ----
//
// Bitwise types: the unsigned and fixed-width ones. The standard list adds the
// signed C types and the floating-point ones; sum and prod use the standard list
// (the complex types are not built, as in NVIDIA's headers).
#define VGPU_NV_BITWISE_TYPES(X, OP)                                                                           \
  X(uchar, unsigned char, OP) X(ushort, unsigned short, OP) X(uint, unsigned int, OP)                          \
  X(ulong, unsigned long, OP) X(ulonglong, unsigned long long, OP) X(int8, int8_t, OP) X(int16, int16_t, OP)  \
  X(int32, int32_t, OP) X(int64, int64_t, OP) X(uint8, uint8_t, OP) X(uint16, uint16_t, OP)                    \
  X(uint32, uint32_t, OP) X(uint64, uint64_t, OP) X(size, size_t, OP)
#define VGPU_NV_STANDARD_TYPES(X, OP)                                                                          \
  VGPU_NV_BITWISE_TYPES(X, OP)                                                                                 \
  X(char, char, OP) X(schar, signed char, OP) X(short, short, OP) X(int, int, OP) X(long, long, OP)            \
  X(longlong, long long, OP) X(bfloat16, __nv_bfloat16, OP) X(half, __half, OP) X(float, float, OP)            \
  X(double, double, OP)

#define VGPU_REDUCE(NAME, TYPE, OP)                                                                            \
  NVSHMEM_EXPORT int nvshmem_##NAME##_##OP##_reduce(nvshmem_team_t team, TYPE* d, const TYPE* src,             \
                                                    size_t n) {                                                \
    return typed_reduce<TYPE>(team, d, src, n, false, r_##OP<TYPE>, "nvshmem_" #NAME "_" #OP "_reduce");      \
  }                                                                                                            \
  NVSHMEM_EXPORT int nvshmem_##NAME##_##OP##_reducescatter(nvshmem_team_t team, TYPE* d, const TYPE* src,      \
                                                           size_t n) {                                         \
    return typed_reduce<TYPE>(team, d, src, n, true, r_##OP<TYPE>,                                             \
                              "nvshmem_" #NAME "_" #OP "_reducescatter");                                      \
  }                                                                                                            \
  NVSHMEM_EXPORT int nvshmemx_##NAME##_##OP##_reduce_on_stream(nvshmem_team_t team, TYPE* d, const TYPE* src,  \
                                                               size_t n, cudaStream_t s) {                     \
    in_stream_order(s, [=] {                                                                                   \
      typed_reduce<TYPE>(team, d, src, n, false, r_##OP<TYPE>, "nvshmemx_" #NAME "_" #OP "_reduce_on_stream"); \
    });                                                                                                        \
    return 0;                                                                                                  \
  }                                                                                                            \
  NVSHMEM_EXPORT int nvshmemx_##NAME##_##OP##_reducescatter_on_stream(nvshmem_team_t team, TYPE* d,            \
                                                                      const TYPE* src, size_t n,               \
                                                                      cudaStream_t s) {                        \
    in_stream_order(s, [=] {                                                                                   \
      typed_reduce<TYPE>(team, d, src, n, true, r_##OP<TYPE>,                                                  \
                         "nvshmemx_" #NAME "_" #OP "_reducescatter_on_stream");                                \
    });                                                                                                        \
    return 0;                                                                                                  \
  }
VGPU_NV_BITWISE_TYPES(VGPU_REDUCE, and)
VGPU_NV_BITWISE_TYPES(VGPU_REDUCE, or)
VGPU_NV_BITWISE_TYPES(VGPU_REDUCE, xor)
VGPU_NV_STANDARD_TYPES(VGPU_REDUCE, max)
VGPU_NV_STANDARD_TYPES(VGPU_REDUCE, min)
VGPU_NV_STANDARD_TYPES(VGPU_REDUCE, sum)
VGPU_NV_STANDARD_TYPES(VGPU_REDUCE, prod)
#undef VGPU_REDUCE

// ---- atomics ----
#define VGPU_AMO_BITWISE_TYPES(X) \
  X(uint, unsigned int) X(ulong, unsigned long) X(ulonglong, unsigned long long) X(int32, int32_t) \
  X(uint32, uint32_t) X(int64, int64_t) X(uint64, uint64_t)
#define VGPU_AMO_STANDARD_TYPES(X) \
  X(int, int) X(long, long) X(longlong, long long) X(size, size_t) X(ptrdiff, ptrdiff_t)
#define VGPU_AMO_EXTENDED_TYPES(X) X(half, __half) X(float, float) X(double, double)

#define VGPU_AMO_INT(NAME, TYPE)                                                                               \
  NVSHMEM_EXPORT void nvshmem_##NAME##_atomic_inc(TYPE* d, int pe) {                                           \
    amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_inc", [](TYPE o) { return v_add(o, one_of<TYPE>()); });    \
  }                                                                                                            \
  NVSHMEM_EXPORT TYPE nvshmem_##NAME##_atomic_fetch_inc(TYPE* d, int pe) {                                     \
    return amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_fetch_inc",                                          \
                         [](TYPE o) { return v_add(o, one_of<TYPE>()); });                                     \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_##NAME##_atomic_add(TYPE* d, TYPE v, int pe) {                                   \
    amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_add", [=](TYPE o) { return v_add(o, v); });                \
  }                                                                                                            \
  NVSHMEM_EXPORT TYPE nvshmem_##NAME##_atomic_fetch_add(TYPE* d, TYPE v, int pe) {                             \
    return amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_fetch_add", [=](TYPE o) { return v_add(o, v); });    \
  }                                                                                                            \
  NVSHMEM_EXPORT TYPE nvshmem_##NAME##_atomic_compare_swap(TYPE* d, TYPE cond, TYPE v, int pe) {               \
    return amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_compare_swap",                                       \
                         [=](TYPE o) { return o == cond ? v : o; });                                           \
  }
VGPU_AMO_BITWISE_TYPES(VGPU_AMO_INT)
VGPU_AMO_STANDARD_TYPES(VGPU_AMO_INT)
#undef VGPU_AMO_INT

// fetch, set and swap: every type of all three lists.
#define VGPU_AMO_ALL(NAME, TYPE)                                                                               \
  NVSHMEM_EXPORT TYPE nvshmem_##NAME##_atomic_fetch(const TYPE* d, int pe) {                                   \
    return amo_fetch<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_fetch");                                           \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_##NAME##_atomic_set(TYPE* d, TYPE v, int pe) {                                   \
    amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_set", [=](TYPE) { return v; });                             \
  }                                                                                                            \
  NVSHMEM_EXPORT TYPE nvshmem_##NAME##_atomic_swap(TYPE* d, TYPE v, int pe) {                                  \
    return amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_swap", [=](TYPE) { return v; });                     \
  }
VGPU_AMO_BITWISE_TYPES(VGPU_AMO_ALL)
VGPU_AMO_STANDARD_TYPES(VGPU_AMO_ALL)
VGPU_AMO_EXTENDED_TYPES(VGPU_AMO_ALL)
#undef VGPU_AMO_ALL

// and, or, xor and their fetching forms.
#define VGPU_AMO_BIT(NAME, TYPE)                                                                               \
  NVSHMEM_EXPORT void nvshmem_##NAME##_atomic_and(TYPE* d, TYPE v, int pe) {                                   \
    amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_and", [=](TYPE o) { return v_and(o, v); });                \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_##NAME##_atomic_or(TYPE* d, TYPE v, int pe) {                                    \
    amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_or", [=](TYPE o) { return v_or(o, v); });                   \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmem_##NAME##_atomic_xor(TYPE* d, TYPE v, int pe) {                                   \
    amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_xor", [=](TYPE o) { return v_xor(o, v); });                 \
  }                                                                                                            \
  NVSHMEM_EXPORT TYPE nvshmem_##NAME##_atomic_fetch_and(TYPE* d, TYPE v, int pe) {                             \
    return amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_fetch_and", [=](TYPE o) { return v_and(o, v); });    \
  }                                                                                                            \
  NVSHMEM_EXPORT TYPE nvshmem_##NAME##_atomic_fetch_or(TYPE* d, TYPE v, int pe) {                              \
    return amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_fetch_or", [=](TYPE o) { return v_or(o, v); });      \
  }                                                                                                            \
  NVSHMEM_EXPORT TYPE nvshmem_##NAME##_atomic_fetch_xor(TYPE* d, TYPE v, int pe) {                             \
    return amo_rmw<TYPE>(d, pe, "nvshmem_" #NAME "_atomic_fetch_xor", [=](TYPE o) { return v_xor(o, v); });    \
  }
VGPU_AMO_BITWISE_TYPES(VGPU_AMO_BIT)
#undef VGPU_AMO_BIT

// The extension: add on half, float and double (nvshmemx_, not standard).
#define VGPU_AMO_XADD(NAME, TYPE)                                                                              \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_atomic_add(TYPE* d, TYPE v, int pe) {                                  \
    amo_rmw<TYPE>(d, pe, "nvshmemx_" #NAME "_atomic_add", [=](TYPE o) { return v_add(o, v); });                \
  }                                                                                                            \
  NVSHMEM_EXPORT TYPE nvshmemx_##NAME##_atomic_fetch_add(TYPE* d, TYPE v, int pe) {                            \
    return amo_rmw<TYPE>(d, pe, "nvshmemx_" #NAME "_atomic_fetch_add", [=](TYPE o) { return v_add(o, v); });   \
  }
VGPU_AMO_EXTENDED_TYPES(VGPU_AMO_XADD)
#undef VGPU_AMO_XADD

// ---- waiting in stream order ----
#define VGPU_NV_WAIT_TYPES(X)                                                                                  \
  X(short, short) X(int, int) X(long, long) X(longlong, long long) X(ushort, unsigned short)                   \
  X(uint, unsigned int) X(ulong, unsigned long) X(ulonglong, unsigned long long) X(int32, int32_t)            \
  X(int64, int64_t) X(uint32, uint32_t) X(uint64, uint64_t) X(size, size_t) X(ptrdiff, ptrdiff_t)

#define VGPU_WAIT(NAME, TYPE)                                                                                  \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_wait_until_on_stream(TYPE* ivar, int cmp, TYPE value, cudaStream_t s) { \
    in_stream_order(s, [=] {                                                                                   \
      wait_for([&] {                                                                                           \
        TYPE v = zero_of<TYPE>();                                                                              \
        copy(&v, ivar, sizeof v, "nvshmemx_" #NAME "_wait_until_on_stream");                                   \
        return cmp_ok<TYPE>(v, cmp, value);                                                                    \
      }, "a variable");                                                                                        \
    });                                                                                                        \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_wait_until_all_on_stream(TYPE* ivars, size_t n, const int* status,     \
                                                                 int cmp, TYPE value, cudaStream_t s) {        \
    in_stream_order(s, [=] {                                                                                   \
      wait_for([&] {                                                                                           \
        std::vector<TYPE> v;                                                                                   \
        std::vector<int> mask;                                                                                 \
        read_n<TYPE>(v, ivars, n, "nvshmemx_" #NAME "_wait_until_all_on_stream");                              \
        if (status) read_n<int>(mask, status, n, "nvshmemx_" #NAME "_wait_until_all_on_stream");               \
        for (size_t i = 0; i < n; ++i)                                                                         \
          if ((!status || !mask[i]) && !cmp_ok<TYPE>(v[i], cmp, value)) return false;                          \
        return true;                                                                                           \
      }, "variables");                                                                                         \
    });                                                                                                        \
  }                                                                                                            \
  NVSHMEM_EXPORT void nvshmemx_##NAME##_wait_until_all_vector_on_stream(TYPE* ivars, size_t n,                 \
                                                                        const int* status, int cmp,            \
                                                                        TYPE* values, cudaStream_t s) {        \
    in_stream_order(s, [=] {                                                                                   \
      wait_for([&] {                                                                                           \
        std::vector<TYPE> v, w;                                                                                \
        std::vector<int> mask;                                                                                 \
        read_n<TYPE>(v, ivars, n, "nvshmemx_" #NAME "_wait_until_all_vector_on_stream");                       \
        read_n<TYPE>(w, values, n, "nvshmemx_" #NAME "_wait_until_all_vector_on_stream");                      \
        if (status) read_n<int>(mask, status, n, "nvshmemx_" #NAME "_wait_until_all_vector_on_stream");        \
        for (size_t i = 0; i < n; ++i)                                                                         \
          if ((!status || !mask[i]) && !cmp_ok<TYPE>(v[i], cmp, w[i])) return false;                           \
        return true;                                                                                           \
      }, "variables");                                                                                         \
    });                                                                                                        \
  }
VGPU_NV_WAIT_TYPES(VGPU_WAIT)
#undef VGPU_WAIT
