// libamd_smi: AMD SMI's library, answered for the simulated GPUs -- what AMD's
// Python package (amdsmi), and through it vLLM and PyTorch, ask it: which
// GPUs there are, what each one is (its target, IDs, compute units, UUID,
// PCI address, KFD node), its memory, and how the GPUs are linked.
//
// The answers come from the machine's state, as ROCm SMI's library here
// (rocm_smi.cpp) and VirtualGPU's amd-smi read it, so the three agree, and
// asking does not start a runtime. The rest of the library's functions answer
// AMDSMI_STATUS_NOT_SUPPORTED (amd_smi_stubs.cpp).
//
// Each GPU is a socket of its own with one processor in it, as AMD SMI
// groups discrete GPUs. Handles are tokens, one per socket and per GPU.
//
// The declarations follow AMD SMI's documented interface (amd_smi/amdsmi.h,
// AMD, MIT license); nothing here is AMD's code.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "vgpu/amd_chip.hpp"
#include "vgpu/amd_kfd.hpp"
#include "vgpu/machine.hpp"
#include "vgpu/regs.hpp"
#include "vgpu/telemetry.hpp"

namespace {

// amdsmi_status_t.
enum Status : int {
  kSuccess = 0,
  kInval = 1,
  kNotSupported = 2,
  kInputOutOfBounds = 17,
  kNotFound = 31,
  kNotInit = 32,
  kInsufficientSize = 41,
};

enum ProcessorType : int { kAmdGpu = 1 };                  // processor_type_t
enum LinkType : int { kLinkInternal = 0, kLinkPcie = 1, kLinkXgmi = 2 };  // amdsmi_link_type_t
enum MemoryType : int { kMemVram = 0, kMemVisVram = 1 };   // amdsmi_memory_type_t
constexpr uint64_t kInitAmdGpus = 1u << 1;                 // AMDSMI_INIT_AMD_GPUS
constexpr uint32_t kUuidSize = 38;                          // AMDSMI_GPU_UUID_SIZE
constexpr uint32_t kNone32 = 0xFFFFFFFF;                    // what the package shows as N/A

struct Version {  // amdsmi_version_t
  uint32_t major, minor, release;
  const char* build;
};
struct AsicInfo {  // amdsmi_asic_info_t
  char market_name[256];
  uint32_t vendor_id;
  char vendor_name[256];
  uint32_t subvendor_id;
  uint64_t device_id;
  uint32_t rev_id;
  char asic_serial[256];
  uint32_t oam_id;
  uint32_t num_of_compute_units;
  uint64_t target_graphics_version;
  uint32_t subsystem_id;
  uint32_t reserved[21];
};
struct KfdInfo {  // amdsmi_kfd_info_t
  uint64_t kfd_id;
  uint32_t node_id;
  uint32_t current_partition_id;
  uint32_t reserved[12];
};
struct DriverInfo {  // amdsmi_driver_info_t
  char driver_version[256], driver_date[256], driver_name[256];
};
struct EnumerationInfo {  // amdsmi_enumeration_info_t
  uint32_t drm_render, drm_card, hsa_id, hip_id;
  char hip_uuid[256];
};
struct EngineUsage {  // amdsmi_engine_usage_t
  uint32_t gfx_activity, umc_activity, mm_activity;
  uint32_t reserved[13];
};
// amdsmi_bdf_t: function 3 bits, device 5, bus 8, domain 48.
using Bdf = uint64_t;
static_assert(sizeof(AsicInfo) == 896 && sizeof(KfdInfo) == 64 && sizeof(DriverInfo) == 768);
static_assert(sizeof(EnumerationInfo) == 272 && sizeof(Version) == 24 && sizeof(EngineUsage) == 64);

using Sample = vgpu::telemetry::DeviceSample;

bool g_init = false;

// The AMD GPUs of the machine, as `vgpu smi` and rocm_smi.cpp read them.
std::vector<Sample> machine() {
  std::vector<Sample> out;
  vgpu::telemetry::Shared snap{};
  if (!vgpu::read_machine(&snap)) return out;
  vgpu::drop_lost_amd(&snap);
  for (uint32_t i = 0; i < snap.device_count; ++i)
    if (std::strcmp(snap.devices[i].vendor, "amd") == 0) out.push_back(snap.devices[i]);
  return out;
}

// Handles: a socket's and a GPU's token for each device ordinal.
constexpr uintptr_t kSocketBase = 0x56475300, kGpuBase = 0x56475000;
void* socket_handle(uint32_t i) { return reinterpret_cast<void*>(kSocketBase + i); }
void* gpu_handle(uint32_t i) { return reinterpret_cast<void*>(kGpuBase + i); }
int ordinal(void* h, uintptr_t base) {
  const uintptr_t v = reinterpret_cast<uintptr_t>(h);
  if (v < base || v >= base + 256) return -1;
  const int i = static_cast<int>(v - base);
  return i < static_cast<int>(machine().size()) ? i : -1;
}

int put_string(char* buf, size_t len, const std::string& s) {
  if (!buf || !len) return kInval;
  std::snprintf(buf, len, "%s", s.c_str());
  return s.size() + 1 > len ? kInsufficientSize : kSuccess;
}

Bdf bdf_of(const Sample& s) {
  unsigned domain = 0, bus = 0, device = 0, fn = 0;
  std::sscanf(s.bus_id, "%x:%x:%x.%x", &domain, &bus, &device, &fn);
  return (uint64_t{domain} << 16) | (uint64_t{bus & 0xff} << 8) | (uint64_t{device & 0x1f} << 3) | (fn & 7);
}

bool instinct(const Sample& s) { return s.architecture[0] == 'c'; }

// The UUID amd-smi list prints: the device's, without NVML's "GPU-" prefix.
std::string uuid_of(const Sample& s) {
  std::string u = s.uuid;
  return u.rfind("GPU-", 0) == 0 ? u.substr(4) : u;
}

}  // namespace

#define AMDSMI_API extern "C" __attribute__((visibility("default")))

// Looks a GPU up: initialized, the handle one of the machine's, the pointer
// given.
#define GPU(h, s, p)                                  \
  if (!g_init) return kNotInit;                       \
  if (!(p)) return kInval;                            \
  const int s##_i = ordinal((h), kGpuBase);           \
  if (s##_i < 0) return kNotFound;                    \
  const Sample s = machine()[static_cast<size_t>(s##_i)]

AMDSMI_API int amdsmi_init(uint64_t flags) {
  // CPUs and non-AMD devices are not simulated; a caller asking only for
  // those finds none, as AMD SMI's library finds none on a machine without.
  (void)flags;
  g_init = true;
  (void)kInitAmdGpus;
  return kSuccess;
}
AMDSMI_API int amdsmi_shut_down() {
  g_init = false;
  return kSuccess;
}

AMDSMI_API int amdsmi_get_lib_version(Version* v) {
  if (!v) return kInval;
  *v = {26, 2, 2, "VirtualGPU"};
  return kSuccess;
}

AMDSMI_API int amdsmi_status_code_to_string(int status, const char** out) {
  if (!out) return kInval;
  switch (status) {
    case kSuccess: *out = "AMDSMI_STATUS_SUCCESS: Call succeeded"; break;
    case kInval: *out = "AMDSMI_STATUS_INVAL: Invalid parameters"; break;
    case kNotSupported: *out = "AMDSMI_STATUS_NOT_SUPPORTED: Command not supported"; break;
    case kInputOutOfBounds: *out = "AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS: Input out of bounds"; break;
    case kNotFound: *out = "AMDSMI_STATUS_NOT_FOUND: Device not found"; break;
    case kNotInit: *out = "AMDSMI_STATUS_NOT_INIT: Device not initialized"; break;
    case kInsufficientSize: *out = "AMDSMI_STATUS_INSUFFICIENT_SIZE: Not enough resources were available for the operation"; break;
    default: *out = "AMDSMI_STATUS_UNKNOWN_ERROR: An unknown error occurred"; break;
  }
  return kSuccess;
}

// ---- Sockets and processors ------------------------------------------------------

// With no array, the count; with one, as many handles as it holds.
AMDSMI_API int amdsmi_get_socket_handles(uint32_t* count, void** handles) {
  if (!g_init) return kNotInit;
  if (!count) return kInval;
  const uint32_t n = static_cast<uint32_t>(machine().size());
  if (handles) {
    const uint32_t k = *count < n ? *count : n;
    for (uint32_t i = 0; i < k; ++i) handles[i] = socket_handle(i);
    *count = k;
  } else {
    *count = n;
  }
  return kSuccess;
}
// A socket's name: its GPU's PCI address, without the function.
AMDSMI_API int amdsmi_get_socket_info(void* socket, size_t len, char* name) {
  if (!g_init) return kNotInit;
  const int i = ordinal(socket, kSocketBase);
  if (i < 0) return kNotFound;
  std::string b = machine()[static_cast<size_t>(i)].bus_id;
  if (b.size() > 12 && b.compare(0, 4, "0000") == 0) b = b.substr(4);
  if (const size_t dot = b.rfind('.'); dot != std::string::npos) b = b.substr(0, dot);
  return put_string(name, len, b);
}
AMDSMI_API int amdsmi_get_processor_handles(void* socket, uint32_t* count, void** handles) {
  if (!g_init) return kNotInit;
  if (!count) return kInval;
  const int i = ordinal(socket, kSocketBase);
  if (i < 0) return kNotFound;
  if (handles && *count >= 1) handles[0] = gpu_handle(static_cast<uint32_t>(i));
  *count = handles && *count == 0 ? 0 : 1;
  return kSuccess;
}
AMDSMI_API int amdsmi_get_processor_handles_by_type(void* socket, int type, void** handles, uint32_t* count) {
  if (type != kAmdGpu) {
    if (!count) return kInval;
    *count = 0;
    return kSuccess;
  }
  return amdsmi_get_processor_handles(socket, count, handles);
}
AMDSMI_API int amdsmi_get_processor_type(void* h, int* type) {
  GPU(h, s, type);
  (void)s;
  *type = kAmdGpu;
  return kSuccess;
}
AMDSMI_API int amdsmi_get_processor_count_from_handles(void** handles, uint32_t* count, uint32_t* cpu_sockets,
                                                       uint32_t* cpu_cores, uint32_t* gpus) {
  if (!handles || !count || !cpu_sockets || !cpu_cores || !gpus) return kInval;
  *cpu_sockets = *cpu_cores = *gpus = 0;
  for (uint32_t i = 0; i < *count; ++i)
    if (ordinal(handles[i], kGpuBase) >= 0) ++*gpus;
  return kSuccess;
}
AMDSMI_API int amdsmi_get_processor_handle_from_bdf(Bdf bdf, void** h) {
  if (!g_init) return kNotInit;
  if (!h) return kInval;
  const std::vector<Sample> m = machine();
  for (size_t i = 0; i < m.size(); ++i)
    if (bdf_of(m[i]) == bdf) {
      *h = gpu_handle(static_cast<uint32_t>(i));
      return kSuccess;
    }
  return kNotFound;
}

// ---- Identity ---------------------------------------------------------------------

AMDSMI_API int amdsmi_get_gpu_device_bdf(void* h, Bdf* bdf) {
  GPU(h, s, bdf);
  *bdf = bdf_of(s);
  return kSuccess;
}
// ROCm SMI's form of the address: domain << 32 | bus << 8 | device << 3 | function.
AMDSMI_API int amdsmi_get_gpu_bdf_id(void* h, uint64_t* id) {
  GPU(h, s, id);
  const Bdf b = bdf_of(s);
  *id = ((b >> 16) << 32) | (b & 0xffff);
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_device_uuid(void* h, unsigned int* len, char* uuid) {
  GPU(h, s, len);
  if (!uuid) return kInval;
  const std::string u = uuid_of(s);
  if (*len < kUuidSize) {
    *len = kUuidSize;
    return kInsufficientSize;
  }
  const int r = put_string(uuid, *len, u);
  *len = static_cast<unsigned>(u.size() + 1);
  return r;
}
AMDSMI_API int amdsmi_get_gpu_id(void* h, uint16_t* id) {
  GPU(h, s, id);
  *id = static_cast<uint16_t>(s.pci_device_id >> 16);
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_revision(void* h, uint16_t* rev) {
  GPU(h, s, rev);
  vgpu::regs::ConfigSpace cs(s);
  *rev = static_cast<uint16_t>(cs.image(16)[0x08]);
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_vendor_name(void* h, char* name, size_t len) {
  GPU(h, s, name);
  (void)s;
  return put_string(name, len, "Advanced Micro Devices Inc. [AMD/ATI]");
}
AMDSMI_API int amdsmi_get_gpu_subsystem_id(void* h, uint16_t* id) {
  GPU(h, s, id);
  *id = static_cast<uint16_t>(s.pci_subsystem_id >> 16);
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_subsystem_name(void* h, char* name, size_t len) {
  GPU(h, s, name);
  return put_string(name, len, s.name);
}
AMDSMI_API int amdsmi_get_gpu_asic_info(void* h, AsicInfo* info) {
  GPU(h, s, info);
  const vgpu::amd::Chip c = vgpu::amd::chip(s.architecture);
  std::memset(info, 0, sizeof *info);
  std::snprintf(info->market_name, sizeof info->market_name, "%s", s.name);
  info->vendor_id = 0x1002;
  std::snprintf(info->vendor_name, sizeof info->vendor_name, "Advanced Micro Devices Inc. [AMD/ATI]");
  info->subvendor_id = 0x1002;
  info->device_id = s.pci_device_id >> 16;
  vgpu::regs::ConfigSpace cs(s);
  info->rev_id = cs.image(16)[0x08];
  // No serial number: the package shows N/A.
  // An OAM module id only for Instinct GPUs, which come on OAM boards.
  info->oam_id = instinct(s) ? static_cast<uint32_t>(s_i) : kNone32;
  info->num_of_compute_units = s.multiprocessors * c.cus_per_mp;
  // The target's digits read as hexadecimal: gfx942 is 0x942, as the package
  // prints "gfx" + hex(value).
  info->target_graphics_version = std::strtoull(c.gfx + 3, nullptr, 16);
  info->subsystem_id = s.pci_subsystem_id >> 16;
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_kfd_info(void* h, KfdInfo* info) {
  GPU(h, s, info);
  std::memset(info, 0, sizeof *info);
  info->kfd_id = vgpu::amd::kfd_gpu_id(s.uuid);
  info->node_id = static_cast<uint32_t>(s_i + 1);
  info->current_partition_id = 0;
  return kSuccess;
}
// Where the GPU is to each runtime: its DRM card and render node, its HSA
// agent (the CPU is 0), its HIP ordinal.
AMDSMI_API int amdsmi_get_gpu_enumeration_info(void* h, EnumerationInfo* info) {
  GPU(h, s, info);
  (void)s;
  std::memset(info, 0, sizeof *info);
  info->drm_card = static_cast<uint32_t>(s_i);
  info->drm_render = static_cast<uint32_t>(128 + s_i);
  info->hsa_id = static_cast<uint32_t>(s_i + 1);
  info->hip_id = static_cast<uint32_t>(s_i);
  std::snprintf(info->hip_uuid, sizeof info->hip_uuid, "GPU-%016llx", 0x5647505500000000ull + static_cast<unsigned>(s_i));
  return kSuccess;
}
// amdgpu's version, or the kernel's for an in-tree driver, as ROCm SMI's
// library here reports it.
AMDSMI_API int amdsmi_get_gpu_driver_info(void* h, DriverInfo* info) {
  GPU(h, s, info);
  (void)s;
  std::memset(info, 0, sizeof *info);
  std::string v;
  if (std::ifstream f("/sys/module/amdgpu/version"); f) std::getline(f, v);
  if (v.empty())
    if (std::ifstream f("/proc/sys/kernel/osrelease"); f) std::getline(f, v);
  std::snprintf(info->driver_version, sizeof info->driver_version, "%s", v.empty() ? "N/A" : v.c_str());
  std::snprintf(info->driver_name, sizeof info->driver_name, "amdgpu");
  return kSuccess;
}

// ---- Memory and activity ------------------------------------------------------------

AMDSMI_API int amdsmi_get_gpu_memory_total(void* h, int type, uint64_t* total) {
  GPU(h, s, total);
  if (type != kMemVram && type != kMemVisVram) return kInval;
  *total = s.vram_total_bytes;
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_memory_usage(void* h, int type, uint64_t* used) {
  GPU(h, s, used);
  if (type != kMemVram && type != kMemVisVram) return kInval;
  *used = s.vram_used_bytes;
  return kSuccess;
}
// Graphics and memory busy, as the machine counts them; no media engines.
AMDSMI_API int amdsmi_get_gpu_activity(void* h, EngineUsage* u) {
  GPU(h, s, u);
  std::memset(u, 0, sizeof *u);
  u->gfx_activity = s.utilization_gpu;
  u->umc_activity = s.utilization_mem;
  u->mm_activity = kNone32;
  return kSuccess;
}

// ---- Topology -------------------------------------------------------------------------
//
// As ROCm SMI's library and KFD's topology here give it: Instinct GPUs one
// XGMI hop apart, weight 15; Radeon GPUs two PCI Express hops apart, through
// the host, weight 40.

namespace {
int pair(void* a, void* b, Sample* sa, Sample* sb) {
  if (!g_init) return kNotInit;
  const int i = ordinal(a, kGpuBase), j = ordinal(b, kGpuBase);
  if (i < 0 || j < 0) return kNotFound;
  const std::vector<Sample> m = machine();
  *sa = m[static_cast<size_t>(i)];
  *sb = m[static_cast<size_t>(j)];
  return i == j ? -1 : kSuccess;
}
}  // namespace

AMDSMI_API int amdsmi_topo_get_link_type(void* a, void* b, uint64_t* hops, int* type) {
  if (!hops || !type) return kInval;
  Sample sa{}, sb{};
  const int r = pair(a, b, &sa, &sb);
  if (r > 0) return r;
  if (r < 0) {
    *hops = 0;
    *type = kLinkInternal;
    return kSuccess;
  }
  const bool xgmi = instinct(sa) && instinct(sb);
  *hops = xgmi ? 1 : 2;
  *type = xgmi ? kLinkXgmi : kLinkPcie;
  return kSuccess;
}
AMDSMI_API int amdsmi_topo_get_link_weight(void* a, void* b, uint64_t* weight) {
  if (!weight) return kInval;
  Sample sa{}, sb{};
  const int r = pair(a, b, &sa, &sb);
  if (r > 0) return r;
  *weight = r < 0 ? 0 : instinct(sa) && instinct(sb) ? 15 : 40;
  return kSuccess;
}
AMDSMI_API int amdsmi_is_P2P_accessible(void* a, void* b, bool* ok) {
  if (!ok) return kInval;
  Sample sa{}, sb{};
  const int r = pair(a, b, &sa, &sb);
  if (r > 0) return r;
  *ok = true;
  return kSuccess;
}
AMDSMI_API int amdsmi_topo_get_numa_node_number(void* h, uint32_t* node) {
  GPU(h, s, node);
  (void)s;
  *node = 0;
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_topo_numa_affinity(void* h, int32_t* node) {
  GPU(h, s, node);
  (void)s;
  *node = 0;
  return kSuccess;
}
