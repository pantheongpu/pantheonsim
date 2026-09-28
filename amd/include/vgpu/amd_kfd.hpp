// The AMD kernel driver's (amdkfd's) topology, as it publishes it under
// /sys/class/kfd/kfd/topology: a node per processor -- the CPU is node 0, the
// GPUs follow -- each with a properties file of "name value" lines, its memory
// banks and its links. Tools read it to find AMD GPUs without starting a
// runtime: rocm_agent_enumerator, Ollama, RCCL and ROCm SMI's own library.
//
// The layout and property names are the kernel's sysfs interface
// (drivers/gpu/drm/amd/amdkfd/kfd_topology.c); the values are the simulated
// devices', the same ones HIP, rocminfo, rocm-smi and amd-smi report.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vgpu/profile.hpp"

namespace vgpu::amd {

struct KfdFile {
  std::string path;      // relative to /sys/class/kfd/kfd/topology
  std::string contents;
};

// The topology of `count` devices of profile `p`, on a host of `cpu_cores`
// cores and `system_memory` bytes.
std::vector<KfdFile> kfd_topology(const DeviceProfile& p, int count, uint32_t cpu_cores, uint64_t system_memory);

// KFD's gfx_target_version for an ISA target: major * 10000 + minor * 100 +
// stepping, the last two digits of the name being hexadecimal ("gfx942" is
// 90402, "gfx90a" is 90010, "gfx1100" is 110000). 0 for anything else.
uint32_t kfd_gfx_target_version(const std::string& gfx);

// KFD's id for a GPU: a hash, stable per device. rocm-smi shows it as the
// GUID and amd-smi as kfd_id.
uint32_t kfd_gpu_id(const char* uuid);

}  // namespace vgpu::amd
