// NVSHMEM's device API, built with NVIDIA's NVSHMEM headers and device library
// (libnvshmem_device.a, linked with -rdc), on VirtualGPU's host library: a job
// of PEs, each a process with a GPU, whose kernels put, get, signal, wait,
// fence, quiet and synchronize across PEs.
//
//   nvshmem_device [npes]    (default 3; run_nvshmem_device.sh builds it)
//
// The device calls are NVIDIA's inline code; what this checks is that the
// state VirtualGPU's host library hands it -- the heap bases, the peers' heaps,
// the team table and the collectives' arrays -- is what that code expects, so
// every PE's kernels reach every other PE's memory by load and store. It also
// checks the host library's copy of the state's layout (nvshmem_abi.hpp)
// against NVIDIA's headers, field by field.
//
// An RTX 3060 under WSL cannot run this: NVIDIA's host library initializes one
// PE and then has no symmetric heap (nvshmem_malloc returns NULL), and two
// 3060s have no peer-to-peer path. What a multi-PE job computes here follows
// NVSHMEM's documentation.
#include <nvshmem.h>
#include <nvshmemx.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "nvshmem_abi.hpp"

static int failures = 0;
static int me = -1;

static void check(bool ok, const std::string& what) {
  if (!ok || me == 0) std::printf("%-4s PE %d: %s\n", ok ? "ok" : "FAIL", me, what.c_str());
  if (!ok) ++failures;
}

// ---- the layout the host library writes, against NVIDIA's ----
#define SAME(OURS, THEIRS, FIELD)                                                                     \
  if (offsetof(vgpu::nvshmem_abi::OURS, FIELD) != offsetof(THEIRS, FIELD)) {                          \
    std::printf("FAIL %s.%s is at %zu, NVIDIA's at %zu\n", #OURS, #FIELD,                             \
                offsetof(vgpu::nvshmem_abi::OURS, FIELD), offsetof(THEIRS, FIELD));                   \
    ++bad;                                                                                            \
  }
static int layout_mismatches() {
  int bad = 0;
  if (sizeof(vgpu::nvshmem_abi::DeviceState) != sizeof(nvshmemi_device_host_state_t)) ++bad;
  if (sizeof(vgpu::nvshmem_abi::Team) != sizeof(nvshmemi_team_t)) ++bad;
  using S = nvshmemi_device_host_state_t;
  SAME(DeviceState, S, mype) SAME(DeviceState, S, npes) SAME(DeviceState, S, job_connectivity)
  SAME(DeviceState, S, heap_base) SAME(DeviceState, S, heap_size) SAME(DeviceState, S, peer_heap_base_p2p)
  SAME(DeviceState, S, peer_heap_base_remote) SAME(DeviceState, S, timeout)
  SAME(DeviceState, S, test_wait_any_start_idx_ptr) SAME(DeviceState, S, team_pool) SAME(DeviceState, S, psync_pool)
  SAME(DeviceState, S, sync_counter) SAME(DeviceState, S, gpu_coll_env_params_var)
  SAME(DeviceState, S, proxy_channels_buf) SAME(DeviceState, S, global_exit_request_state)
  SAME(DeviceState, S, nvshmemi_is_nvshmem_initialized) SAME(DeviceState, S, selected_device_transport)
  SAME(DeviceState, S, tma_policy) SAME(DeviceState, S, unicast_le_ids_) SAME(DeviceState, S, region_slots)
  SAME(DeviceState, S, region_slot_probe_limit)
  using T = nvshmemi_team_t;
  SAME(Team, T, my_pe) SAME(Team, T, team_idx) SAME(Team, T, config) SAME(Team, T, reduce_recexch)
  SAME(Team, T, rdxn_count) SAME(Team, T, alltoall_pwrk) SAME(Team, T, fcollect_ll_flag)
  SAME(Team, T, are_gpus_p2p_connected) SAME(Team, T, team_node) SAME(Team, T, team_dups) SAME(Team, T, pe_mapping)
  SAME(Team, T, mc_leid_with_flag)
  using C = gpu_coll_env_params_t;
  SAME(CollParams, C, barrier_dissem_kval) SAME(CollParams, C, fcollect_ll_threshold)
  SAME(CollParams, C, reduce_scratch_size) SAME(CollParams, C, fcollect_ll128_threshold)
  SAME(CollParams, C, fcollect_ll_supported)
  return bad;
}

// ---- kernels ----

struct Results {
  int p_got, g_got, get_ok, signal_ok, add_got, fetch_old, team_pe, team_n, fence_ok, ptr_got;
};

__global__ void thread_scope(int* data, int* box, uint64_t* sig, int* counter, int* ordered, int* scratch,
                             nvshmem_team_t even, Results* r) {
  const int me = nvshmem_my_pe(), n = nvshmem_n_pes();
  const int next = (me + 1) % n, prev = (me + n - 1) % n;
  // p and g.
  nvshmem_int_p(box, 100 + me, next);
  nvshmem_quiet();
  nvshmem_barrier_all();
  r->p_got = *box;
  r->g_got = nvshmem_int_g(box, next);
  // A block put from local device memory, and a get back.
  for (int i = 0; i < 64; ++i) scratch[i] = me * 1000 + i;
  nvshmem_int_put(data, scratch, 64, next);
  nvshmem_barrier_all();
  nvshmem_int_get(scratch + 64, data, 64, next);
  int ok = 1;
  for (int i = 0; i < 64; ++i) ok &= scratch[64 + i] == me * 1000 + i && data[i] == prev * 1000 + i;
  r->get_ok = ok;
  nvshmem_barrier_all();
  // A put with a signal, and the next PE waiting for the signal.
  nvshmem_int_put_signal(data + 100, scratch, 8, sig, 5, NVSHMEM_SIGNAL_SET, next);
  nvshmem_signal_wait_until(sig, NVSHMEM_CMP_EQ, 5);
  ok = 1;
  for (int i = 0; i < 8; ++i) ok &= data[100 + i] == prev * 1000 + i;
  r->signal_ok = ok;
  // Signal add and a fetching atomic, each with one writer per target.
  nvshmemx_signal_op(sig + 1, 3, NVSHMEM_SIGNAL_ADD, next);
  nvshmemx_signal_op(sig + 1, 4, NVSHMEM_SIGNAL_ADD, next);
  r->fetch_old = nvshmem_int_atomic_fetch_add(counter, 10 + me, next);
  nvshmem_barrier_all();
  r->add_got = (int)nvshmem_signal_fetch(sig + 1) * 1000 + *counter;
  // Two puts in order: with a fence between them, the flag is never seen
  // before the value.
  nvshmem_int_p(ordered, 77, next);
  nvshmem_fence();
  nvshmem_int_p(ordered + 1, 1, next);
  nvshmem_int_wait_until(ordered + 1, NVSHMEM_CMP_EQ, 1);
  r->fence_ok = ordered[0] == 77;
  // A team made on the host, used in the kernel.
  if (even != NVSHMEM_TEAM_INVALID) {
    r->team_pe = nvshmem_team_my_pe(even);
    r->team_n = nvshmem_team_n_pes(even);
    nvshmem_team_sync(even);
  }
  nvshmem_barrier_all();
  // A peer's memory through nvshmem_ptr, written directly.
  int* peer = (int*)nvshmem_ptr(box, next);
  if (peer) *peer = 500 + me;
  nvshmem_barrier_all();
  r->ptr_got = *box;
}

__global__ void block_scope(int* data, int* scratch, int* out) {
  const int me = nvshmem_my_pe(), n = nvshmem_n_pes();
  const int next = (me + 1) % n, prev = (me + n - 1) % n;
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) scratch[i] = me * 100000 + i;
  __syncthreads();
  nvshmemx_int_put_block(data, scratch, 1024, next);
  nvshmemx_barrier_all_block();
  int ok = 1;
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) ok &= data[i] == prev * 100000 + i;
  atomicAnd(out, ok);
}

__global__ void collective(int* box) {
  nvshmem_barrier_all();
  if (threadIdx.x == 0 && blockIdx.x == 0) *box = nvshmem_n_pes();
  nvshmem_barrier_all();
}

static int pe_main(int rank, int npes, nvshmemx_uniqueid_t* id) {
  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  cudaSetDevice(rank % (ndev ? ndev : 1));
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_set_attr_uniqueid_args(rank, npes, id, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  me = nvshmem_my_pe();
  const int n = nvshmem_n_pes();
  const int prev = (me + n - 1) % n;
  check(me == rank && n == npes, "the job");
  if (me == 0) check(layout_mismatches() == 0, "the host library's state layout is NVIDIA's");

  int* data = (int*)nvshmem_calloc(2048, sizeof(int));
  int* box = (int*)nvshmem_calloc(1, sizeof(int));
  uint64_t* sig = (uint64_t*)nvshmem_calloc(2, sizeof(uint64_t));
  int* counter = (int*)nvshmem_calloc(1, sizeof(int));
  int* ordered = (int*)nvshmem_calloc(2, sizeof(int));
  int *scratch = nullptr, *flag = nullptr;
  Results* r = nullptr;
  cudaMalloc((void**)&scratch, 2048 * sizeof(int));
  cudaMalloc((void**)&flag, sizeof(int));
  cudaMallocManaged((void**)&r, sizeof(Results));
  nvshmem_team_t even = NVSHMEM_TEAM_INVALID;
  nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 2, (n + 1) / 2, nullptr, 0, &even);

  thread_scope<<<1, 1>>>(data, box, sig, counter, ordered, scratch, even, r);
  cudaError_t e = cudaDeviceSynchronize();
  check(e == cudaSuccess, std::string("the thread-scope kernel ran: ") + cudaGetErrorString(e));
  check(r->p_got == 100 + prev, "nvshmem_int_p from the previous PE");
  check(r->g_got == 100 + me, "nvshmem_int_g of the next PE");
  check(r->get_ok, "nvshmem_int_put and nvshmem_int_get");
  check(r->signal_ok, "nvshmem_int_put_signal and nvshmem_signal_wait_until");
  check(r->fetch_old == 0, "nvshmem_int_atomic_fetch_add returns the old value");
  check(r->add_got == 7 * 1000 + 10 + prev, "signal adds and the atomic add land");
  check(r->fence_ok, "nvshmem_fence orders two puts");
  if (me % 2 == 0) check(r->team_pe == me / 2 && r->team_n == (n + 1) / 2, "a team made on the host, in a kernel");
  check(r->ptr_got == 500 + prev, "a store through nvshmem_ptr");

  cudaMemset(data, 0, 2048 * sizeof(int));
  const int one = 1;
  cudaMemcpy(flag, &one, sizeof(int), cudaMemcpyHostToDevice);
  nvshmem_barrier_all();
  block_scope<<<1, 128>>>(data, scratch, flag);
  e = cudaDeviceSynchronize();
  int ok = 0;
  cudaMemcpy(&ok, flag, sizeof ok, cudaMemcpyDeviceToHost);
  check(e == cudaSuccess && ok == 1, "nvshmemx_int_put_block and nvshmemx_barrier_all_block");

  void* args[] = {&box};
  check(nvshmemx_collective_launch((const void*)collective, dim3(2), dim3(32), args, 0, 0) == 0,
        "nvshmemx_collective_launch");
  cudaDeviceSynchronize();
  int b = 0;
  cudaMemcpy(&b, box, sizeof b, cudaMemcpyDeviceToHost);
  check(b == n, "a collectively launched kernel's barriers");

  if (even != NVSHMEM_TEAM_INVALID) nvshmem_team_destroy(even);
  nvshmem_free(data);
  nvshmem_free(box);
  nvshmem_free(sig);
  nvshmem_free(counter);
  nvshmem_free(ordered);
  cudaFree(scratch);
  cudaFree(flag);
  cudaFree(r);
  nvshmem_finalize();
  return failures ? 1 : 0;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const int npes = argc > 1 ? std::atoi(argv[1]) : 3;
  nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
  nvshmemx_get_uniqueid(&id);
  std::vector<pid_t> kids;
  for (int k = 1; k < npes; ++k) {
    const pid_t p = fork();
    if (p == 0) std::exit(pe_main(k, npes, &id));  // exit, not _exit: the PE tears down its GPU
    kids.push_back(p);
  }
  int rc = pe_main(0, npes, &id);
  for (pid_t p : kids) {
    int status = 0;
    waitpid(p, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::printf("FAIL a PE's process ended with status %d\n", status);
      rc = 1;
    }
  }
  std::printf("%s: %d PEs\n", rc ? "FAIL" : "PASS", npes);
  return rc;
}
