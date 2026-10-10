// NVSHMEM's host API across PEs: a job of PEs, each a process with a GPU of
// its own, bootstrapped with a unique ID (no MPI). The symmetric heap,
// blocking, strided, typed and stream-ordered remote memory access, signals,
// barriers, teams (strided and 2-D splits, translation, destruction) and the
// broadcast, fcollect and alltoall collectives -- every result checked on
// every PE.
//
//   nvshmem_host             a job of one PE (nvshmem_init(), no bootstrap), then one of three
//   nvshmem_host npes        a job of npes PEs, forked from here
//   nvshmem_host single      the job of one only
//   nvshmem_host peerless    a job of two PEs on two devices that have no peer path (the rtx3060 profile): refused,
//                            as NVIDIA's NVSHMEM refuses two GeForce GPUs ("Peer GPU 1 is not accessible",
//                            NVSHMEMX_ERROR_NOT_SUPPORTED, measured on two RTX 3060s)
//
// Written against VirtualGPU's declarations (nvidia/include/vgpu_nvshmem.h).
// NVIDIA's library cannot run this on an RTX 3060 under WSL beyond its first
// steps -- nvshmem_init succeeds as one PE and nvshmem_malloc returns NULL --
// so what a multi-PE job does is the documentation's (docs.nvidia.com/nvshmem).
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../include/vgpu_nvshmem.h"

static int failures = 0;
static int me = -1;

static void check(bool ok, const std::string& what) {
  if (!ok || me == 0) std::printf("%-4s PE %d: %s\n", ok ? "ok" : "FAIL", me, what.c_str());
  if (!ok) ++failures;
}

template <class T>
static std::vector<T> fetch(const T* p, size_t n) {
  std::vector<T> v(n);
  cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
  return v;
}

static int pe_main(int rank, int npes, nvshmemx_uniqueid_t* id) {
  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  cudaSetDevice(rank % (ndev ? ndev : 1));
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  if (nvshmemx_set_attr_uniqueid_args(rank, npes, id, &attr) != 0) return 1;
  if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) != 0) return 1;
  me = nvshmem_my_pe();
  const int n = nvshmem_n_pes();
  check(me == rank && n == npes, "my_pe and n_pes are the bootstrap's");
  // PEs on one GPU make NVSHMEM's multiple-processes-per-GPU mode (measured on an RTX 3060: status 3).
  check(nvshmemx_init_status() == (npes > (ndev ? ndev : 1) ? NVSHMEM_STATUS_LIMITED_MPG : NVSHMEM_STATUS_IS_INITIALIZED),
        "initialized (in the multiple-processes-per-GPU mode when the PEs share a GPU)");
  int major = 0, minor = 0, patch = 0;
  nvshmem_info_get_version(&major, &minor);
  nvshmemx_vendor_get_version_info(nullptr, nullptr, &patch);
  check(major == 1 && minor == 3, "the OpenSHMEM version is 1.3");
  char name[NVSHMEM_MAX_NAME_LEN];
  nvshmem_info_get_name(name);
  check(std::string(name).rfind("NVSHMEM v3.", 0) == 0, "the vendor name");
  const int next = (me + 1) % n, prev = (me + n - 1) % n;

  // ---- the symmetric heap ----
  int* a = static_cast<int*>(nvshmem_malloc(1024 * sizeof(int)));
  int* z = static_cast<int*>(nvshmem_calloc(256, sizeof(int)));
  double* al = static_cast<double*>(nvshmem_align(4096, 100 * sizeof(double)));
  check(a && z && al, "malloc, calloc and align return memory");
  check(reinterpret_cast<uintptr_t>(al) % 4096 == 0, "align honours the alignment");
  const std::vector<int> zeros = fetch(z, 256);
  bool all_zero = true;
  for (int v : zeros) all_zero = all_zero && v == 0;
  check(all_zero, "calloc's memory is zero");
  check(nvshmem_ptr(a, me) == a, "nvshmem_ptr of this PE is the pointer itself");
  check(nvshmem_ptr(a, next) != nullptr, "every peer is reachable by load and store");
  check(nvshmem_ptr(&me, next) == nullptr, "nvshmem_ptr of non-symmetric memory is NULL");
  check(nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, a) == nullptr, "no multicast address");

  // ---- blocking puts and gets ----
  std::vector<int> mine(1024);
  for (int i = 0; i < 1024; ++i) mine[i] = me * 10000 + i;
  nvshmem_int_put(a, mine.data(), 1024, next);  // from host memory
  nvshmem_barrier_all();
  std::vector<int> got = fetch(a, 1024);
  bool ring = true;
  for (int i = 0; i < 1024; ++i) ring = ring && got[i] == prev * 10000 + i;
  check(ring, "nvshmem_int_put from host memory reaches the next PE");
  std::vector<int> back(1024, -1);
  nvshmem_getmem(back.data(), a, 1024 * sizeof(int), next);
  bool g = true;
  for (int i = 0; i < 1024; ++i) g = g && back[i] == me * 10000 + i;
  check(g, "nvshmem_getmem reads the next PE's copy");
  nvshmem_barrier_all();
  nvshmem_int_p(z + me, 1000 + me, next);
  nvshmem_quiet();
  nvshmem_barrier_all();
  check(fetch(z + prev, 1)[0] == 1000 + prev, "nvshmem_int_p");
  check(nvshmem_int_g(z + me, next) == 1000 + me, "nvshmem_int_g");
  // From device memory, with the sized and strided forms.
  int* src = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&src), 64 * sizeof(int));
  cudaMemcpy(src, mine.data(), 64 * sizeof(int), cudaMemcpyHostToDevice);
  nvshmem_barrier_all();
  nvshmem_put32(a + 512, src, 64, next);
  nvshmem_iput32(a + 600, src, 2, 3, 10, next);  // a[600 + 2i] = src[3i]
  nvshmem_barrier_all();
  got = fetch(a, 1024);
  bool sized = true, strided = true;
  for (int i = 0; i < 64; ++i) sized = sized && got[512 + i] == prev * 10000 + i;
  for (int i = 0; i < 10; ++i) strided = strided && got[600 + 2 * i] == prev * 10000 + 3 * i;
  check(sized, "nvshmem_put32 from device memory");
  check(strided, "nvshmem_iput32 with strides 2 and 3");
  std::vector<int> st(10, -1);
  nvshmem_iget32(st.data(), a + 600, 1, 2, 10, next);
  bool sg = true;
  for (int i = 0; i < 10; ++i) sg = sg && st[i] == me * 10000 + 3 * i;
  check(sg, "nvshmem_iget32");
  nvshmem_double_p(al + me, 0.5 * me, 0);
  nvshmem_barrier_all();
  if (me == 0) {
    const std::vector<double> d = fetch(al, size_t(n));
    bool ok = true;
    for (int i = 0; i < n; ++i) ok = ok && d[i] == 0.5 * i;
    check(ok, "every PE's nvshmem_double_p lands on PE 0");
  }

  // ---- stream order ----
  cudaStream_t s;
  cudaStreamCreate(&s);
  nvshmem_barrier_all();
  std::vector<int> pattern(32);
  for (int i = 0; i < 32; ++i) pattern[i] = -(me * 100 + i);
  cudaMemcpyAsync(src, pattern.data(), 32 * sizeof(int), cudaMemcpyHostToDevice, s);
  nvshmemx_putmem_on_stream(a, src, 32 * sizeof(int), next, s);  // after the copy before it
  nvshmemx_barrier_all_on_stream(s);
  cudaStreamSynchronize(s);
  got = fetch(a, 32);
  bool on_stream = true;
  for (int i = 0; i < 32; ++i) on_stream = on_stream && got[i] == -(prev * 100 + i);
  check(on_stream, "nvshmemx_putmem_on_stream follows the copy queued before it");

  // ---- signals ----
  uint64_t* sig = static_cast<uint64_t*>(nvshmem_calloc(4, sizeof(uint64_t)));
  nvshmemx_putmem_signal_on_stream(a + 100, src, 4 * sizeof(int), sig, 7, NVSHMEM_SIGNAL_SET, next, s);
  nvshmemx_signal_wait_until_on_stream(sig, NVSHMEM_CMP_EQ, 7, s);
  cudaStreamSynchronize(s);
  check(nvshmem_signal_fetch(sig) == 7, "a put with a signal sets the signal");
  check(fetch(a + 100, 1)[0] == -(prev * 100), "and the data is there before the signal");
  nvshmem_barrier_all();
  nvshmemx_signal_op_on_stream(sig + 1, uint64_t(me + 1), NVSHMEM_SIGNAL_ADD, 0, s);
  cudaStreamSynchronize(s);
  nvshmem_barrier_all();
  if (me == 0) check(nvshmem_signal_fetch(sig + 1) == uint64_t(n * (n + 1) / 2), "every PE's signal add on PE 0");

  // ---- teams ----
  check(nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD) == me && nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD) == n,
        "the world team");
  check(nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE) == n, "one node holds every PE");
  check(nvshmem_team_n_pes(NVSHMEMX_TEAM_SAME_MYPE_NODE) == 1, "a team of PEs with this node rank: this PE");
  nvshmem_team_t even = NVSHMEM_TEAM_INVALID;
  const int evens = (n + 1) / 2;
  check(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 2, evens, nullptr, 0, &even) == 0, "split strided");
  if (me % 2 == 0) {
    check(even != NVSHMEM_TEAM_INVALID && nvshmem_team_my_pe(even) == me / 2 && nvshmem_team_n_pes(even) == evens,
          "an even PE is in the even team at half its rank");
    check(nvshmem_team_translate_pe(even, 1, NVSHMEM_TEAM_WORLD) == (evens > 1 ? 2 : -1),
          "the even team's PE 1 is world PE 2");
    check(nvshmem_team_translate_pe(NVSHMEM_TEAM_WORLD, me, even) == me / 2, "and back");
    check(nvshmem_team_sync(even) == 0, "the even team synchronizes");
    nvshmem_team_config_t cfg;
    nvshmem_team_get_config(even, &cfg);
    check(cfg.num_contexts >= 0, "the team has a configuration");
  } else {
    check(even == NVSHMEM_TEAM_INVALID, "an odd PE is not in the even team");
  }
  nvshmem_team_t x = NVSHMEM_TEAM_INVALID, y = NVSHMEM_TEAM_INVALID;
  check(nvshmem_team_split_2d(NVSHMEM_TEAM_WORLD, 2, nullptr, 0, &x, nullptr, 0, &y) == 0, "split 2-D");
  check(nvshmem_team_n_pes(x) == ((me / 2) * 2 + 2 <= n ? 2 : n - (me / 2) * 2) && nvshmem_team_my_pe(x) == me % 2,
        "the x team is this PE's row of two");
  check(nvshmem_team_my_pe(y) == me / 2, "the y team is this PE's column");
  nvshmem_barrier_all();
  if (even != NVSHMEM_TEAM_INVALID) nvshmem_team_destroy(even);
  nvshmem_team_destroy(x);
  nvshmem_team_destroy(y);
  nvshmem_barrier_all();
  nvshmem_team_t again = NVSHMEM_TEAM_INVALID;
  check(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 1, n, nullptr, 0, &again) == 0 &&
            nvshmem_team_n_pes(again) == n,
        "a team index is reused after destruction");
  nvshmem_team_destroy(again);

  // ---- collectives ----
  int* bsrc = static_cast<int*>(nvshmem_malloc(16 * sizeof(int)));
  int* bdst = static_cast<int*>(nvshmem_malloc(16 * size_t(n) * sizeof(int)));
  std::vector<int> mine16(16 * size_t(n));
  for (size_t i = 0; i < mine16.size(); ++i) mine16[i] = me * 1000 + int(i);
  cudaMemcpy(bsrc, mine16.data(), 16 * sizeof(int), cudaMemcpyHostToDevice);
  check(nvshmem_broadcastmem(NVSHMEM_TEAM_WORLD, bdst, bsrc, 16 * sizeof(int), n - 1) == 0, "broadcast");
  got = fetch(bdst, 16);
  bool bc = true;
  for (int i = 0; i < 16; ++i) bc = bc && got[i] == (n - 1) * 1000 + i;
  check(bc, "broadcast delivers the root's block to every PE");
  check(nvshmem_fcollectmem(NVSHMEM_TEAM_WORLD, bdst, bsrc, 16 * sizeof(int)) == 0, "fcollect");
  got = fetch(bdst, 16 * size_t(n));
  bool fc = true;
  for (int p = 0; p < n; ++p)
    for (int i = 0; i < 16; ++i) fc = fc && got[size_t(p) * 16 + i] == p * 1000 + i;
  check(fc, "fcollect concatenates every PE's block in PE order");
  int* asrc = static_cast<int*>(nvshmem_malloc(16 * size_t(n) * sizeof(int)));
  cudaMemcpy(asrc, mine16.data(), mine16.size() * sizeof(int), cudaMemcpyHostToDevice);
  check(nvshmem_alltoallmem(NVSHMEM_TEAM_WORLD, bdst, asrc, 4 * sizeof(int)) == 0, "alltoall");
  got = fetch(bdst, 4 * size_t(n));
  bool at = true;
  for (int p = 0; p < n; ++p)
    for (int i = 0; i < 4; ++i) at = at && got[size_t(p) * 4 + i] == p * 1000 + me * 4 + i;
  check(at, "alltoall sends block j to PE j");

  // ---- atomics ----
  {
    long* ctr = static_cast<long*>(nvshmem_calloc(8, sizeof(long)));
    unsigned* bits = static_cast<unsigned*>(nvshmem_calloc(8, sizeof(unsigned)));
    double* dv = static_cast<double*>(nvshmem_calloc(4, sizeof(double)));
    __half* hv = static_cast<__half*>(nvshmem_calloc(4, sizeof(__half)));
    long* seen = static_cast<long*>(nvshmem_calloc(size_t(n), sizeof(long)));
    nvshmem_barrier_all();
    nvshmem_long_atomic_add(ctr, me + 1, 0);
    const long old = nvshmem_long_atomic_fetch_inc(ctr + 1, 0);
    nvshmem_long_p(seen + me, old, 0);  // every PE's fetched value lands on PE 0
    nvshmem_barrier_all();
    if (me == 0) {
      check(fetch(ctr, 1)[0] == long(n) * (n + 1) / 2, "atomic add from every PE");
      check(fetch(ctr + 1, 1)[0] == n, "atomic inc from every PE");
      std::vector<long> v = fetch(seen, size_t(n));
      std::vector<bool> hit(size_t(n), false);
      bool perm = true;
      for (long x : v) {
        if (x < 0 || x >= n || hit[size_t(x)]) perm = false;
        else hit[size_t(x)] = true;
      }
      check(perm, "fetch_inc hands every PE a different value");
    }
    nvshmem_barrier_all();
    // set, fetch, swap and compare-and-swap, all against PE 0's word.
    if (me == 0) nvshmem_long_atomic_set(ctr + 2, 41, 0);
    nvshmem_barrier_all();
    check(nvshmem_long_atomic_fetch(ctr + 2, 0) == 41, "atomic set and fetch");
    nvshmem_barrier_all();
    if (me == n - 1) {
      check(nvshmem_long_atomic_swap(ctr + 2, 7, 0) == 41, "atomic swap returns the old value");
      check(nvshmem_long_atomic_compare_swap(ctr + 2, 8, 99, 0) == 7, "a failing compare-and-swap returns the old value");
      check(nvshmem_long_atomic_fetch(ctr + 2, 0) == 7, "and leaves it");
      check(nvshmem_long_atomic_compare_swap(ctr + 2, 7, 99, 0) == 7, "a matching compare-and-swap returns the old value");
      check(nvshmem_long_atomic_fetch(ctr + 2, 0) == 99, "and stores the new one");
    }
    nvshmem_barrier_all();
    // bitwise
    if (me == 0) nvshmem_uint_atomic_set(bits, 0xff00ff00u, 0);
    nvshmem_barrier_all();
    if (me == n - 1) {
      check(nvshmem_uint_atomic_fetch_and(bits, 0x0ff00ff0u, 0) == 0xff00ff00u, "fetch_and returns the old value");
      nvshmem_uint_atomic_or(bits, 0x1u, 0);
      nvshmem_uint_atomic_xor(bits, 0x8u, 0);
      check(nvshmem_uint_atomic_fetch(bits, 0) == ((((0xff00ff00u & 0x0ff00ff0u) | 1u) ^ 8u)), "and, or, xor");
    }
    // floating point and half precision
    nvshmem_barrier_all();
    nvshmemx_double_atomic_add(dv, 0.5, 0);
    nvshmemx_half_atomic_add(hv, __float2half(1.0f), 0);
    nvshmem_barrier_all();
    if (me == 0) {
      check(fetch(dv, 1)[0] == 0.5 * n, "double atomic add");
      __half h;
      cudaMemcpy(&h, hv, sizeof h, cudaMemcpyDeviceToHost);
      check(__half2float(h) == float(n), "half atomic add");
    }
    nvshmem_barrier_all();
    nvshmem_free(seen);
    nvshmem_free(hv);
    nvshmem_free(dv);
    nvshmem_free(bits);
    nvshmem_free(ctr);
  }

  // ---- nonblocking, strided and stream-ordered typed RMA ----
  {
    float* f = static_cast<float*>(nvshmem_calloc(64, sizeof(float)));
    uint64_t* flag = static_cast<uint64_t*>(nvshmem_calloc(2, sizeof(uint64_t)));
    std::vector<float> hostf(64);
    for (int i = 0; i < 64; ++i) hostf[size_t(i)] = float(me * 100 + i);
    float* devf = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&devf), 64 * sizeof(float));
    cudaMemcpy(devf, hostf.data(), 64 * sizeof(float), cudaMemcpyHostToDevice);
    nvshmem_barrier_all();
    nvshmem_float_put_nbi(f, devf, 16, next);
    nvshmem_float_iput(f + 20, devf, 2, 3, 8, next);
    nvshmem_quiet();
    nvshmem_barrier_all();
    std::vector<float> gf = fetch(f, 64);
    bool ok = true;
    for (int i = 0; i < 16; ++i) ok = ok && gf[size_t(i)] == float(prev * 100 + i);
    check(ok, "nvshmem_float_put_nbi");
    ok = true;
    for (int i = 0; i < 8; ++i) ok = ok && gf[size_t(20 + 2 * i)] == float(prev * 100 + 3 * i);
    check(ok, "nvshmem_float_iput");
    std::vector<float> back(16, -1.0f);
    nvshmem_float_get_nbi(back.data(), f, 16, next);
    ok = true;
    for (int i = 0; i < 16; ++i) ok = ok && back[size_t(i)] == float(me * 100 + i);
    check(ok, "nvshmem_float_get_nbi");
    nvshmem_barrier_all();
    cudaStream_t s2;
    cudaStreamCreate(&s2);
    nvshmemx_float_put_signal_on_stream(f + 32, devf, 8, flag, 5, NVSHMEM_SIGNAL_SET, next, s2);
    nvshmemx_signal_wait_until_on_stream(flag, NVSHMEM_CMP_EQ, 5, s2);
    nvshmemx_float_iget_on_stream(back.data(), f + 32, 1, 1, 8, next, s2);
    cudaStreamSynchronize(s2);
    ok = true;
    for (int i = 0; i < 8; ++i) ok = ok && back[size_t(i)] == float(me * 100 + i);
    check(ok, "typed put with a signal, then an iget, in stream order");
    check(nvshmemx_float_g_on_stream(f + 33, next, s2) == float(me * 100 + 1), "nvshmemx_float_g_on_stream");
    nvshmem_barrier_all();
    // Waiting for a variable in stream order.
    int* w = static_cast<int*>(nvshmem_calloc(4, sizeof(int)));
    nvshmem_barrier_all();
    // Streams here are synchronous (work is done when it is enqueued), so the writes
    // come first: a wait enqueued before them would hold its PE and every other.
    nvshmem_int_p(w, 3 + me, next);
    nvshmem_int_p(w + 1, 1, next);
    nvshmem_int_p(w + 2, 2, next);
    nvshmem_barrier_all();
    nvshmemx_int_wait_until_on_stream(w, NVSHMEM_CMP_GE, 3, s2);
    cudaStreamSynchronize(s2);
    check(fetch(w, 1)[0] == 3 + prev, "nvshmemx_int_wait_until_on_stream returns once the variable is written");
    int cmpv[2] = {1, 2};
    int* d_cmp = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_cmp), sizeof cmpv);
    cudaMemcpy(d_cmp, cmpv, sizeof cmpv, cudaMemcpyHostToDevice);
    nvshmemx_int_wait_until_all_vector_on_stream(w + 1, 2, nullptr, NVSHMEM_CMP_EQ, d_cmp, s2);
    nvshmemx_int_wait_until_all_on_stream(w + 1, 2, nullptr, NVSHMEM_CMP_GT, 0, s2);
    cudaStreamSynchronize(s2);
    check(true, "wait_until_all and its vector form finish");
    nvshmem_barrier_all();
    cudaFree(d_cmp);
    nvshmem_free(w);
    cudaStreamDestroy(s2);
    cudaFree(devf);
    nvshmem_free(flag);
    nvshmem_free(f);
  }

  // ---- typed collectives and reductions ----
  {
    const size_t cnt = 5;
    float* fs = static_cast<float*>(nvshmem_malloc(cnt * size_t(n) * sizeof(float)));
    float* fd = static_cast<float*>(nvshmem_malloc(cnt * size_t(n) * sizeof(float)));
    std::vector<float> in(cnt * size_t(n));
    for (size_t i = 0; i < in.size(); ++i) in[i] = float(me + 1) * float(i + 1);
    cudaMemcpy(fs, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
    nvshmem_barrier_all();
    check(nvshmem_float_broadcast(NVSHMEM_TEAM_WORLD, fd, fs, cnt, 1 % n) == 0, "nvshmem_float_broadcast");
    std::vector<float> o = fetch(fd, cnt);
    bool ok = true;
    for (size_t i = 0; i < cnt; ++i) ok = ok && o[i] == float((1 % n) + 1) * float(i + 1);
    check(ok, "the float broadcast delivers the root's block");
    check(nvshmem_float_fcollect(NVSHMEM_TEAM_WORLD, fd, fs, cnt) == 0, "nvshmem_float_fcollect");
    o = fetch(fd, cnt * size_t(n));
    ok = true;
    for (int p = 0; p < n; ++p)
      for (size_t i = 0; i < cnt; ++i) ok = ok && o[size_t(p) * cnt + i] == float(p + 1) * float(i + 1);
    check(ok, "the float fcollect concatenates the blocks");
    check(nvshmem_float_alltoall(NVSHMEM_TEAM_WORLD, fd, fs, 2) == 0, "nvshmem_float_alltoall");
    o = fetch(fd, 2 * size_t(n));
    ok = true;
    for (int p = 0; p < n; ++p)
      for (size_t i = 0; i < 2; ++i) ok = ok && o[size_t(p) * 2 + i] == float(p + 1) * float(size_t(me) * 2 + i + 1);
    check(ok, "the float alltoall sends block j to PE j");

    // Reductions over the world team: sum, prod, max, min.
    int* is = static_cast<int*>(nvshmem_malloc(cnt * sizeof(int)));
    int* id2 = static_cast<int*>(nvshmem_malloc(cnt * sizeof(int)));
    std::vector<int> iv(cnt);
    for (size_t i = 0; i < cnt; ++i) iv[i] = (me + 1) * int(i + 1);
    cudaMemcpy(is, iv.data(), cnt * sizeof(int), cudaMemcpyHostToDevice);
    nvshmem_barrier_all();
    check(nvshmem_int_sum_reduce(NVSHMEM_TEAM_WORLD, id2, is, cnt) == 0, "nvshmem_int_sum_reduce");
    std::vector<int> ri = fetch(id2, cnt);
    ok = true;
    for (size_t i = 0; i < cnt; ++i) ok = ok && ri[i] == (n * (n + 1) / 2) * int(i + 1);
    check(ok, "the sum of every PE's block, on every PE");
    check(nvshmem_int_sum_reduce(NVSHMEM_TEAM_WORLD, is, is, cnt) == 0, "an in-place reduction");
    ri = fetch(is, cnt);
    ok = true;
    for (size_t i = 0; i < cnt; ++i) ok = ok && ri[i] == (n * (n + 1) / 2) * int(i + 1);
    check(ok, "and its result");
    cudaMemcpy(is, iv.data(), cnt * sizeof(int), cudaMemcpyHostToDevice);
    nvshmem_barrier_all();
    check(nvshmem_int_max_reduce(NVSHMEM_TEAM_WORLD, id2, is, cnt) == 0, "nvshmem_int_max_reduce");
    ri = fetch(id2, cnt);
    ok = true;
    for (size_t i = 0; i < cnt; ++i) ok = ok && ri[i] == n * int(i + 1);
    check(ok, "the maximum is the last PE's");
    check(nvshmem_int_min_reduce(NVSHMEM_TEAM_WORLD, id2, is, cnt) == 0, "nvshmem_int_min_reduce");
    ri = fetch(id2, cnt);
    ok = true;
    for (size_t i = 0; i < cnt; ++i) ok = ok && ri[i] == int(i + 1);
    check(ok, "the minimum is the first PE's");
    check(nvshmem_int_prod_reduce(NVSHMEM_TEAM_WORLD, id2, is, cnt) == 0, "nvshmem_int_prod_reduce");
    ri = fetch(id2, cnt);
    int fact = 1;
    for (int p = 1; p <= n; ++p) fact *= p;
    ok = true;
    for (size_t i = 0; i < cnt; ++i) {
      int want = 1;
      for (int p = 1; p <= n; ++p) want *= p * int(i + 1);
      ok = ok && ri[i] == want;
    }
    check(ok && fact > 0, "the product");
    // Reduce-scatter: PE i keeps block i of the sum.
    int* rs = static_cast<int*>(nvshmem_malloc(cnt * size_t(n) * sizeof(int)));
    int* rd = static_cast<int*>(nvshmem_malloc(cnt * sizeof(int)));
    std::vector<int> big(cnt * size_t(n));
    for (size_t i = 0; i < big.size(); ++i) big[i] = (me + 1) * int(i + 1);
    cudaMemcpy(rs, big.data(), big.size() * sizeof(int), cudaMemcpyHostToDevice);
    nvshmem_barrier_all();
    check(nvshmem_int_sum_reducescatter(NVSHMEM_TEAM_WORLD, rd, rs, cnt) == 0, "nvshmem_int_sum_reducescatter");
    ri = fetch(rd, cnt);
    ok = true;
    for (size_t i = 0; i < cnt; ++i) ok = ok && ri[i] == (n * (n + 1) / 2) * int(size_t(me) * cnt + i + 1);
    check(ok, "each PE keeps its own block of the sum");
    // Bitwise, on unsigned types.
    uint32_t* bs = static_cast<uint32_t*>(nvshmem_malloc(2 * sizeof(uint32_t)));
    uint32_t* bd = static_cast<uint32_t*>(nvshmem_malloc(2 * sizeof(uint32_t)));
    const uint32_t bv[2] = {~(1u << (me % 32)), 1u << (me % 32)};
    cudaMemcpy(bs, bv, sizeof bv, cudaMemcpyHostToDevice);
    nvshmem_barrier_all();
    check(nvshmem_uint32_and_reduce(NVSHMEM_TEAM_WORLD, bd, bs, 2) == 0, "nvshmem_uint32_and_reduce");
    uint32_t wa[2] = {~0u, ~0u}, wo[2] = {0, 0}, wx[2] = {0, 0};
    for (int p = 0; p < n; ++p) {
      const uint32_t v0 = ~(1u << (p % 32)), v1 = 1u << (p % 32);
      wa[0] &= v0; wa[1] &= v1;
      wo[0] |= v0; wo[1] |= v1;
      wx[0] ^= v0; wx[1] ^= v1;
    }
    uint32_t gb[2] = {0, 0};
    cudaMemcpy(gb, bd, sizeof gb, cudaMemcpyDeviceToHost);
    check(gb[0] == wa[0] && gb[1] == wa[1], "the AND of every PE's words");
    check(nvshmem_uint32_or_reduce(NVSHMEM_TEAM_WORLD, bd, bs, 2) == 0, "nvshmem_uint32_or_reduce");
    cudaMemcpy(gb, bd, sizeof gb, cudaMemcpyDeviceToHost);
    check(gb[0] == wo[0] && gb[1] == wo[1], "the OR");
    check(nvshmem_uint32_xor_reduce(NVSHMEM_TEAM_WORLD, bd, bs, 2) == 0, "nvshmem_uint32_xor_reduce");
    cudaMemcpy(gb, bd, sizeof gb, cudaMemcpyDeviceToHost);
    check(gb[0] == wx[0] && gb[1] == wx[1], "the XOR");
    // Half precision, and the stream-ordered form.
    __half* hs = static_cast<__half*>(nvshmem_malloc(4 * sizeof(__half)));
    __half* hd = static_cast<__half*>(nvshmem_malloc(4 * sizeof(__half)));
    __half hin[4];
    for (int i = 0; i < 4; ++i) hin[i] = __float2half(0.5f * float(me + 1) * float(i + 1));
    cudaMemcpy(hs, hin, sizeof hin, cudaMemcpyHostToDevice);
    cudaStream_t s3;
    cudaStreamCreate(&s3);
    nvshmem_barrier_all();
    check(nvshmemx_half_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, hd, hs, 4, s3) == 0, "nvshmemx_half_sum_reduce_on_stream");
    cudaStreamSynchronize(s3);
    __half hout[4];
    cudaMemcpy(hout, hd, sizeof hout, cudaMemcpyDeviceToHost);
    ok = true;
    for (int i = 0; i < 4; ++i) ok = ok && __half2float(hout[i]) == 0.5f * float(n * (n + 1) / 2) * float(i + 1);
    check(ok, "the half-precision sum, in stream order");
    cudaStreamDestroy(s3);
    nvshmem_barrier_all();
    nvshmem_free(hd);
    nvshmem_free(hs);
    nvshmem_free(bd);
    nvshmem_free(bs);
    nvshmem_free(rd);
    nvshmem_free(rs);
    nvshmem_free(id2);
    nvshmem_free(is);
    nvshmem_free(fd);
    nvshmem_free(fs);
  }

  // ---- a team made from a unique ID ----
  {
    nvshmemx_team_uniqueid_t tid = 0;
    // One PE makes the ID; the others get it out of band (here: the symmetric heap).
    uint64_t* shared_id = static_cast<uint64_t*>(nvshmem_calloc(1, sizeof(uint64_t)));
    if (me == 0) {
      check(nvshmemx_team_get_uniqueid(&tid) == 0 && tid != 0, "a team unique ID");
      for (int p = 0; p < n; ++p) nvshmem_uint64_p(shared_id, tid, p);
    }
    nvshmem_barrier_all();
    cudaMemcpy(&tid, shared_id, sizeof tid, cudaMemcpyDeviceToHost);
    nvshmem_team_config_t cfg = NVSHMEM_TEAM_CONFIG_INITIALIZER;
    cfg.uniqueid = tid;
    nvshmem_team_t t = NVSHMEM_TEAM_INVALID;
    // The team has every PE, in reverse order.
    check(nvshmemx_team_init(&t, &cfg, NVSHMEM_TEAM_CONFIG_MASK_UNIQUEID, n, n - 1 - me) == 0,
          "nvshmemx_team_init");
    check(t >= NVSHMEM_TEAMS_MIN && nvshmem_team_n_pes(t) == n && nvshmem_team_my_pe(t) == n - 1 - me,
          "the new team has the sizes and indices the members gave");
    check(nvshmem_team_translate_pe(t, 0, NVSHMEM_TEAM_WORLD) == n - 1, "and its PE 0 is the last world PE");
    check(nvshmem_team_sync(t) == 0, "the team synchronizes");
    nvshmem_team_destroy(t);
    nvshmem_barrier_all();
    nvshmem_team_t bad = NVSHMEM_TEAM_INVALID;
    nvshmem_team_config_t none = NVSHMEM_TEAM_CONFIG_INITIALIZER;
    check(nvshmemx_team_init(&bad, &none, NVSHMEM_TEAM_CONFIG_MASK_UNIQUEID, n, me) != 0 && bad == NVSHMEM_TEAM_INVALID,
          "a team without a unique ID is refused");
    nvshmem_free(shared_id);
  }

  // ---- the allocator ----
  nvshmem_free(bsrc);
  nvshmem_free(bdst);
  nvshmem_free(asrc);
  int* reuse = static_cast<int*>(nvshmem_malloc(16 * sizeof(int)));
  check(reuse == bsrc, "a freed block is handed out again");
  nvshmem_free(reuse);
  check(nvshmem_malloc(size_t(1) << 40) == nullptr, "an allocation larger than the heap is NULL");

  nvshmem_free(a);
  nvshmem_free(z);
  nvshmem_free(al);
  nvshmem_free(sig);
  cudaFree(src);
  cudaStreamDestroy(s);
  nvshmem_finalize();
  check(nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED, "finalized");
  return failures ? 1 : 0;
}

static int single_main() {
  {
    nvshmem_init();
    me = nvshmem_my_pe();
    check(me == 0 && nvshmem_n_pes() == 1, "nvshmem_init without a bootstrap is a job of one PE");
    int* p = static_cast<int*>(nvshmem_malloc(64));
    check(p != nullptr, "a one-PE job has a heap");
    nvshmem_int_p(p, 5, 0);
    check(nvshmem_int_g(p, 0) == 5, "and reaches it");
    nvshmem_free(p);
    nvshmem_barrier_all();
    nvshmem_finalize();
    return failures ? 1 : 0;
  }
}

// One PE of a job on devices that cannot reach each other: the init must be refused, and the PE must not be up.
static int refused_main(int rank, int npes, nvshmemx_uniqueid_t* id) {
  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  if (ndev < npes) {
    std::printf("SKIP: the job needs %d devices (set VGPU_DEVICE_COUNT)\n", npes);
    return 0;
  }
  cudaSetDevice(rank);
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  if (nvshmemx_set_attr_uniqueid_args(rank, npes, id, &attr) != 0) return 1;
  const int rc = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  const bool up = nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED;
  std::printf("%-4s PE %d: init on devices without a peer path -> %d, %s\n", rc != 0 && !up ? "ok" : "FAIL", rank, rc,
              up ? "the PE is up" : "the PE is not up");
  return rc != 0 && !up ? 0 : 1;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc > 1 && std::string(argv[1]) == "single") return single_main();
  if (argc > 1 && std::string(argv[1]) == "peerless") {
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    if (nvshmemx_get_uniqueid(&id) != 0) return 1;
    const pid_t k = fork();
    if (k == 0) std::exit(refused_main(1, 2, &id));
    int rc = refused_main(0, 2, &id);
    int status = 0;
    waitpid(k, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) rc = 1;
    std::printf("%s: two PEs on devices without a peer path are refused\n", rc ? "FAIL" : "PASS");
    return rc;
  }
  // Without arguments: a one-PE job, then a job of three, each in processes
  // of their own (a process is one PE for its lifetime).
  if (argc == 1) {
    const pid_t k = fork();
    if (k == 0) std::exit(single_main());
    int status = 0;
    waitpid(k, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::printf("FAIL the one-PE job ended with status %d\n", status);
      return 1;
    }
  }
  const int npes = argc > 1 ? std::atoi(argv[1]) : 3;
  // The unique ID is made before any PE touches CUDA, then shared by fork.
  nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
  if (nvshmemx_get_uniqueid(&id) != 0) return 1;
  std::vector<pid_t> kids;
  for (int r = 1; r < npes; ++r) {
    const pid_t k = fork();
    if (k == 0) std::exit(pe_main(r, npes, &id));  // exit, not _exit: the PE tears down its GPU
    kids.push_back(k);
  }
  int rc = pe_main(0, npes, &id);
  for (pid_t k : kids) {
    int status = 0;
    waitpid(k, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::printf("FAIL a PE's process ended with status %d\n", status);
      rc = 1;
    }
  }
  std::printf("%s: %d PEs%s\n", rc ? "FAIL" : "PASS", npes, rc ? ", see above" : "");
  return rc;
}
