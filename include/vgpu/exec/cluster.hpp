// What a thread-block cluster can be given, for the occupancy queries
// (cudaOccupancyMaxActiveClusters, cudaOccupancyMaxPotentialClusterSize and the
// driver's cuOccupancy twins). Header-only so the rules can be tested on their own.
//
// The rules are the ones NVIDIA documents: a cluster is co-scheduled on one GPC
// (CUDA programming guide, "Thread Block Clusters"); a cluster of up to 8 blocks
// is portable; a kernel that opts in with cudaFuncAttributeNonPortableClusterSizeAllowed
// may have up to 16 on Hopper (the H100's table of limits) -- the launch check in
// validate_launch applies the same 16 to every compute capability from 9.0 -- and
// the occupancy API respects the kernel's launch bounds and its required cluster
// size. How many clusters then fit is the sum over the GPCs of the blocks a GPC holds
// divided by the cluster size, and depends on the SMs in each GPC, which NVIDIA does
// not publish for a part with TPCs disabled: GpuLayout (vgpu/profile.hpp) has the
// counts a profile has a source for and spreads them evenly, so an answer that
// needs more than one SM's worth of blocks per GPC is derived, and says so.
//
// What a card without clusters answers was measured on an RTX 3060 (sm_86):
// a configuration with no cluster dimension is cudaSuccess with 0, one that names
// a cluster dimension (even 1x1x1) is cudaErrorInvalidClusterSize
// (CUDA_ERROR_INVALID_CLUSTER_SIZE), for both queries and both APIs. The
// answers for a part with clusters are NOT checked against a card: there is none
// at hand, so they rest on the documentation above.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "vgpu/profile.hpp"

namespace vgpu::exec {

constexpr uint32_t kPortableClusterBlocks = 8;

inline bool supports_clusters(const DeviceProfile& p) { return p.vendor == "nvidia" && p.cc_major >= 9; }
inline uint32_t max_nonportable_cluster(const DeviceProfile& p) { return p.cc_major >= 9 ? 16 : kPortableClusterBlocks; }

struct ClusterAnswer {
  enum class Status { Ok, InvalidValue, InvalidClusterSize, NotSupported } status = Status::Ok;
  int64_t value = 0;
  bool uses_layout = false;   // the answer rests on the SMs-per-GPC spread of the profile (derived)
  const char* why = "";       // for NotSupported: what is missing
};

inline uint64_t cluster_blocks(const std::array<uint32_t, 3>& d) { return uint64_t{d[0]} * d[1] * d[2]; }

// `requested`: the cluster dimension of the launch configuration, `has_requested` whether it names one;
// `required`: the kernel's own (all zeros when it has none); `blocks_per_sm`: how many blocks of the
// configuration an SM holds (0 when one does not fit).
inline ClusterAnswer cluster_occupancy(const DeviceProfile& p, bool want_active_clusters,
                                       bool has_requested, const std::array<uint32_t, 3>& requested,
                                       const std::array<uint32_t, 3>& required, bool nonportable_allowed,
                                       uint32_t blocks_per_sm) {
  using S = ClusterAnswer::Status;
  ClusterAnswer a;
  if (!supports_clusters(p)) {
    if (has_requested) a.status = S::InvalidClusterSize;
    return a;   // otherwise Ok with 0
  }
  const bool has_required = cluster_blocks(required) != 0;
  if (!want_active_clusters) {
    // The cluster dimensions of the configuration are ignored; a kernel with a required size reports it.
    if (has_required) {
      a.value = static_cast<int64_t>(cluster_blocks(required));
      return a;
    }
  } else {
    if (has_required && has_requested && requested != required) {
      a.status = S::InvalidClusterSize;
      return a;
    }
    if (!has_required && !has_requested) {
      a.status = S::InvalidValue;   // "the cluster size must be specified in config, else the function will return an error"
      return a;
    }
  }
  const uint32_t limit = nonportable_allowed ? max_nonportable_cluster(p) : kPortableClusterBlocks;
  if (want_active_clusters) {
    const uint64_t size = cluster_blocks(has_required ? required : requested);
    if (size == 0 || size > limit || size > max_nonportable_cluster(p)) {
      a.status = S::InvalidClusterSize;
      return a;
    }
    if (blocks_per_sm == 0) return a;   // not even one block fits: nothing is active
    if (size == 1) {   // every block alone: no GPC to know
      a.value = int64_t{blocks_per_sm} * p.limits.multiprocessors;
      return a;
    }
    if (!p.layout.known()) {
      a.status = S::NotSupported;
      a.why = "the GPC layout of this part (how many SMs each GPC has) is not published, and a cluster has to fit in one GPC";
      return a;
    }
    int64_t total = 0;
    for (uint32_t sms : p.layout.gpc_sms()) total += int64_t{sms} * blocks_per_sm / static_cast<int64_t>(size);
    a.value = total;
    a.uses_layout = true;
    return a;
  }
  // The largest cluster that can launch.
  if (blocks_per_sm == 0) return a;
  if (limit <= kPortableClusterBlocks) {   // 8 is guaranteed on every part with clusters
    a.value = limit;
    return a;
  }
  if (!p.layout.known()) {
    a.status = S::NotSupported;
    a.why = "the GPC layout of this part (how many SMs each GPC has) is not published, and a cluster has to fit in one GPC";
    return a;
  }
  uint64_t biggest = 0;
  for (uint32_t sms : p.layout.gpc_sms()) biggest = std::max<uint64_t>(biggest, uint64_t{sms} * blocks_per_sm);
  a.value = static_cast<int64_t>(std::min<uint64_t>(limit, biggest));
  a.uses_layout = true;
  return a;
}

}  // namespace vgpu::exec
