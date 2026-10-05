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
//
// Written against VirtualGPU's declarations (nvidia/include/vgpu_nvshmem.h).
// NVIDIA's library cannot run this on an RTX 3060 under WSL beyond its first
// steps -- nvshmem_init succeeds as one PE and nvshmem_malloc returns NULL --
// so what a multi-PE job does is the documentation's (docs.nvidia.com/nvshmem).
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
  check(nvshmemx_init_status() == NVSHMEM_STATUS_IS_INITIALIZED, "initialized");
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

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc > 1 && std::string(argv[1]) == "single") return single_main();
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
