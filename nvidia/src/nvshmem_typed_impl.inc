// The typed host API of NVSHMEM, the half that nvshmem_api.cpp does not spell
// out one function at a time: atomics, reductions and the arithmetic they share.
// Included inside the anonymous namespace of nvshmem_api.cpp, after the
// collectives, so it can use put, get, copy, remote, team_barrier and the rest.
//
// Derived from NVSHMEM's documentation and its public headers; no NVIDIA
// library behind a symmetric heap was available to compare against (the RTX 3060
// pair gives a job with no heap), so none of this is checked against a card.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>

// ---- arithmetic -------------------------------------------------------------
//
// Integers wrap (computed unsigned, so that an overflowing sum is not undefined
// behaviour); half and bfloat16 compute in float and round once, as the device
// code does.

template <class T>
T zero_of() {
  T z;
  std::memset(static_cast<void*>(&z), 0, sizeof z);
  return z;
}

template <class T>
inline T v_add(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    using U = std::make_unsigned_t<T>;
    using W = std::conditional_t<(sizeof(U) < sizeof(unsigned)), unsigned, U>;
    return static_cast<T>(static_cast<U>(static_cast<W>(static_cast<U>(a)) + static_cast<W>(static_cast<U>(b))));
  } else {
    return a + b;
  }
}
inline __half v_add(__half a, __half b) { return __float2half(__half2float(a) + __half2float(b)); }
inline __nv_bfloat16 v_add(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

template <class T>
inline T v_mul(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    using U = std::make_unsigned_t<T>;
    using W = std::conditional_t<(sizeof(U) < sizeof(unsigned)), unsigned, U>;
    return static_cast<T>(static_cast<U>(static_cast<W>(static_cast<U>(a)) * static_cast<W>(static_cast<U>(b))));
  } else {
    return a * b;
  }
}
inline __half v_mul(__half a, __half b) { return __float2half(__half2float(a) * __half2float(b)); }
inline __nv_bfloat16 v_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
}

template <class T>
inline T v_max(T a, T b) { return a < b ? b : a; }
inline __half v_max(__half a, __half b) { return __half2float(a) < __half2float(b) ? b : a; }
inline __nv_bfloat16 v_max(__nv_bfloat16 a, __nv_bfloat16 b) { return __bfloat162float(a) < __bfloat162float(b) ? b : a; }
template <class T>
inline T v_min(T a, T b) { return b < a ? b : a; }
inline __half v_min(__half a, __half b) { return __half2float(b) < __half2float(a) ? b : a; }
inline __nv_bfloat16 v_min(__nv_bfloat16 a, __nv_bfloat16 b) { return __bfloat162float(b) < __bfloat162float(a) ? b : a; }

template <class T> inline T v_and(T a, T b) { return static_cast<T>(a & b); }
template <class T> inline T v_or(T a, T b) { return static_cast<T>(a | b); }
template <class T> inline T v_xor(T a, T b) { return static_cast<T>(a ^ b); }

// The reduction operators, by name, so that a macro can paste one.
template <class T> inline T r_sum(T a, T b) { return v_add(a, b); }
template <class T> inline T r_prod(T a, T b) { return v_mul(a, b); }
template <class T> inline T r_max(T a, T b) { return v_max(a, b); }
template <class T> inline T r_min(T a, T b) { return v_min(a, b); }
template <class T> inline T r_and(T a, T b) { return v_and(a, b); }
template <class T> inline T r_or(T a, T b) { return v_or(a, b); }
template <class T> inline T r_xor(T a, T b) { return v_xor(a, b); }

// ---- atomics -----------------------------------------------------------------
//
// An atomic on a peer's heap is a read-modify-write under the job's lock, the
// same lock the host-side signal operations use: atomic against the other PEs'
// host-side atomics and signals, not against a kernel's atomics in flight.

template <class T, class F>
T amo_rmw(T* dest, int pe, const char* api, F f) {
  State& s = st();
  T old = zero_of<T>();
  if (!initialized(s)) {
    say("%s before nvshmem_init", api);
    return old;
  }
  if (!check_pe(s, pe, api)) return old;
  if (!symmetric(s, dest, sizeof(T))) {
    say("%s: the target is not symmetric memory", api);
    return old;
  }
  T* where = reinterpret_cast<T*>(remote(s, dest, pe));
  seg_lock(s);
  if (copy(&old, where, sizeof old, api)) {
    const T nv = f(old);
    copy(where, &nv, sizeof nv, api);
  }
  seg_unlock(s);
  return old;
}

template <class T>
T amo_fetch(const T* src, int pe, const char* api) {
  State& s = st();
  T v = zero_of<T>();
  if (!initialized(s)) {
    say("%s before nvshmem_init", api);
    return v;
  }
  if (!check_pe(s, pe, api)) return v;
  if (!symmetric(s, src, sizeof(T))) {
    say("%s: the target is not symmetric memory", api);
    return v;
  }
  seg_lock(s);
  copy(&v, remote(s, src, pe), sizeof v, api);
  seg_unlock(s);
  return v;
}

template <class T> T one_of() { return v_add(zero_of<T>(), static_cast<T>(1)); }
template <> inline __half one_of<__half>() { return __float2half(1.0f); }
template <> inline __nv_bfloat16 one_of<__nv_bfloat16>() { return __float2bfloat16(1.0f); }

// ---- collectives with a type --------------------------------------------------

// Reduction over a team, in team order (so a floating-point sum is the same on
// every PE). `scatter` is the reduce-scatter form: the source holds one block of
// n elements per PE, and PE i keeps the reduction of block i.
template <class T>
int typed_reduce(nvshmem_team_t team, T* dest, const T* src, size_t n, bool scatter, T (*op)(T, T), const char* api) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  const int me = team_rank_ok(s, team, api);
  if (me < 0) return 1;
  const HostTeam& t = s.teams[team];
  const size_t span = scatter ? n * size_t(t.size) : n;
  if (n && (!symmetric(s, src, span * sizeof(T)) || !symmetric(s, dest, n * sizeof(T)))) {
    say("%s: the source and the destination must be symmetric memory", api);
    return 1;
  }
  cudaDeviceSynchronize();
  cudaGetLastError();
  team_barrier(s, team);  // every PE's source is final
  std::vector<T> acc(n), tmp(n);
  const size_t off = scatter ? size_t(me) * n : 0;
  bool ok = true;
  for (int i = 0; i < t.size && n; ++i) {
    T* dst = i == 0 ? acc.data() : tmp.data();
    ok = copy(dst, remote(s, src + off, t.members[size_t(i)]), n * sizeof(T), api) && ok;
    if (i > 0)
      for (size_t k = 0; k < n; ++k) acc[k] = op(acc[k], tmp[k]);
  }
  team_barrier(s, team);  // every PE has read: a destination that is the source may be overwritten
  if (n) ok = copy(dest, acc.data(), n * sizeof(T), api) && ok;
  return ok ? 0 : 1;
}

// ---- waiting on a variable ------------------------------------------------------

template <class T>
bool cmp_ok(T a, int cmp, T b) {
  switch (cmp) {
    case NVSHMEM_CMP_EQ: return a == b;
    case NVSHMEM_CMP_NE: return a != b;
    case NVSHMEM_CMP_GT: return a > b;
    case NVSHMEM_CMP_LE: return a <= b;
    case NVSHMEM_CMP_LT: return a < b;
    case NVSHMEM_CMP_GE: return a >= b;
  }
  return false;
}

// Reads `n` elements of caller memory, host or device, wherever it lives.
template <class T>
bool read_n(std::vector<T>& out, const T* p, size_t n, const char* api) {
  out.assign(n, T{});
  return n == 0 || copy(out.data(), p, n * sizeof(T), api);
}
