// The state NVSHMEM's device code reads, as the simulator's libnvshmem_host
// lays it out for programs built with NVIDIA's NVSHMEM headers and device
// library (libnvshmem_device.a).
//
// NVSHMEM's device API is inline code in its public headers. It reads one
// __constant__ struct, nvshmemi_device_state_d, which the device library
// defines and the host library fills in at initialization: through it a
// kernel finds the symmetric heap, each peer's copy of it, the team table and
// the collectives' synchronization arrays. These are the layouts the public
// headers declare (NVSHMEM 3.8: the state is 848 bytes, a team 816), written
// out field for field so the host library can build them; the sizes are
// checked here and every offset against NVIDIA's headers in
// nvidia/tests/e2e/nvshmem_device.cu.
#ifndef VGPU_NVSHMEM_ABI_HPP
#define VGPU_NVSHMEM_ABI_HPP

#include <cstddef>
#include <cstdint>

namespace vgpu::nvshmem_abi {

constexpr int kTeamDupCount = 128;
// The synchronization arrays' sizes, in longs (nvshmem_common.cuh).
constexpr size_t kSyncSize = 27648;
constexpr size_t kSyncSizeX2 = 2 * kSyncSize;     // NVSHMEMI_SYNC_SIZE
constexpr size_t kBcastSyncSize = 10 * kSyncSize;  // NVSHMEMI_BCAST_SYNC_SIZE
constexpr size_t kAlltoallSyncSize = kSyncSize;    // NVSHMEMI_ALLTOALL_SYNC_SIZE

enum JobConnectivity : int {
  kJobGpuLdstAtomics = 1,
  kJobGpuLdst = 1 << 1,
  kJobGpuLdstRemoteAtomics = 1 << 2,
  kJobGpuProxy = 1 << 3,
  kJobGpuProxyCst = 1 << 4,
};

struct ReduceRecexch {
  int version;
  int step1_sendto;
  int* step1_recvfrom;
  int** step2_nbrs;
  int step1_nrecvs;
  int step2_nphases;
  char padding[32];
};
static_assert(sizeof(ReduceRecexch) == 64, "reduce_recexch_v1");

struct TeamConfig {
  int version;
  int num_contexts;
  uint64_t uniqueid;
  char padding[48];
};
static_assert(sizeof(TeamConfig) == 64, "team_config_v2");

struct Team {  // nvshmemi_team_v4
  int version;
  int my_pe;
  int start, stride, size;
  int team_idx;
  TeamConfig config;
  long config_mask;
  void* nccl_comm;
  ReduceRecexch reduce_recexch;
  size_t rdxn_count;
  uint32_t ll_flag;
  uint64_t alltoall_pwrk[2];
  uint64_t alltoall_count;
  uint64_t bcast_count;
  uint64_t bcast_sync_offset;
  uint64_t fcollect_count;
  uint32_t fcollect_ll_flag;
  bool are_gpus_p2p_connected;
  bool is_team_node;
  int32_t team_node;
  bool is_team_same_mype_node;
  int32_t team_same_mype_node;
  void* nvls_rsc;
  void* nvls_rsc_base_ptr;
  int32_t team_dups[kTeamDupCount];
  int* pe_mapping;  // size entries team -> world, then npes entries world -> team (-1 if absent)
  uint64_t p2p_sync_on_stream_count;
  bool are_gpus_nvls_connected;
  uint64_t mc_leid_with_flag;
};
static_assert(sizeof(Team) == 816, "team_v4 is 816 bytes");

struct CollParams {  // gpu_coll_env_params_v2
  int version;
  int barrier_dissem_kval;
  int barrier_tg_dissem_kval;
  int reduce_recexch_kval;
  int bcast_tree_kval;
  int bcast_algo;
  int reduce_algo;
  size_t fcollect_ll_threshold;
  size_t fcollect_nvls_threshold;
  size_t reduce_scratch_size;
  int fcollect_algo;
  size_t reducescatter_nvls_threshold;
  int reducescatter_algo;
  int reduce_maxloc_algo;
  size_t fcollect_ll128_threshold;
  size_t reduce_nvls_threshold;
  bool fcollect_ll_supported;
  char padding[415];
};
static_assert(sizeof(CollParams) == 512, "gpu_coll_env_params_v2 is 512 bytes");

struct Timeout {
  int version;
  uint64_t signal;
  uint64_t caller;
  uint64_t signal_addr;
  uint64_t signal_val_found;
  uint64_t signal_val_expected;
  char padding[16];
};
static_assert(sizeof(Timeout) == 64, "timeout_v1 is 64 bytes");

struct DeviceState {  // nvshmemi_device_host_state_v1
  int version;
  int mype;
  int npes;
  int node_mype;
  int node_npes;
  int pe_dist;
  int proxy;
  int atomics_sync;
  int job_connectivity;
  bool proxy_ops_are_ordered;
  bool atomics_complete_on_quiet;
  void* heap_base;
  size_t heap_size;
  void** peer_heap_base_p2p;
  void** peer_heap_base_remote;
  bool symmetric_heap_kind;
  bool enable_rail_opt;
  uint32_t atomics_le_min_size;
  Timeout* timeout;
  unsigned long long* test_wait_any_start_idx_ptr;
  Team** team_pool;
  long* psync_pool;
  long* sync_counter;
  CollParams gpu_coll_env_params_var;
  void* proxy_channels_buf;
  char* proxy_channel_g_buf;
  char* proxy_channel_g_coalescing_buf;
  uint64_t* proxy_channel_g_buf_head_ptr;
  uint64_t proxy_channel_g_buf_size;
  uint64_t proxy_channel_g_buf_log_size;
  uint64_t* proxy_channels_issue;
  uint64_t* proxy_channels_complete;
  uint64_t* proxy_channels_complete_local_ptr;
  uint64_t* proxy_channels_quiet_issue;
  uint64_t* proxy_channels_quiet_ack;
  uint64_t* proxy_channels_cst_issue;
  uint64_t* proxy_channels_cst_ack;
  uint64_t proxy_channel_buf_size;
  uint32_t proxy_channel_buf_logsize;
  int* global_exit_request_state;
  int* global_exit_code;
  bool ibgda_is_initialized;
  bool nvshmemi_is_nvshmem_initialized;
  bool nvshmemi_is_nvshmem_bootstrapped;
  int selected_device_transport;
  int tma_policy;
  uintptr_t* tma_smem_bases;
  size_t tma_smem_bases_len;
  size_t* tma_smem_size;
  void* unicast_le_ids_;
  bool counted_operations_available;
  void* region_slots;
  uint32_t* region_active_count;
  uint32_t region_slots_len;
  uint32_t region_slot_probe_limit;
};
static_assert(sizeof(DeviceState) == 848, "device_host_state_v1 is 848 bytes");

// The length of one team's synchronization arrays, in longs, as the device
// code computes it (get_psync_len_per_team), rounded up to an even count.
inline size_t psync_len_per_team(const CollParams& p, int npes) {
  const size_t fcollect = (2 * 2 * size_t(npes) * p.fcollect_ll_threshold) / sizeof(long);
  auto round_up = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
  // NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(threshold, char): the threshold
  // rounded up to 120-byte packets plus 8 bytes per packet.
  const size_t t = p.fcollect_ll128_threshold;
  const size_t ll128 = (round_up(t, 120) + 8 * ((t + 119) / 120)) * 2 * size_t(npes) / sizeof(long);
  const size_t ans = 4 * kSyncSizeX2 + p.reduce_scratch_size / sizeof(long) + kBcastSyncSize + fcollect +
                     2 * kAlltoallSyncSize + ll128 + size_t(npes);
  return round_up(ans, 2);
}

}  // namespace vgpu::nvshmem_abi

#endif  // VGPU_NVSHMEM_ABI_HPP
