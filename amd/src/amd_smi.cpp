// libamd_smi: AMD SMI's library, answered for the simulated GPUs -- what AMD's
// Python package (amdsmi), and through it vLLM and PyTorch, ask it: which
// GPUs there are, what each one is (its target, IDs, compute units, UUID,
// PCI address, KFD node), its memory, and how the GPUs are linked.
//
// The answers come from the machine's state, as ROCm SMI's library here
// (rocm_smi.cpp) and VirtualGPU's amd-smi read it, so the three agree, and
// asking does not start a runtime. The rest of the library's functions answer
// AMDSMI_STATUS_NOT_SUPPORTED (amd_smi_stubs.cpp). The health queries (ECC,
// bad pages, XGMI, PCIe, sensors, partitions, processes) are at the end.
//
// Each GPU is a socket of its own with one processor in it, as AMD SMI
// groups discrete GPUs. Handles are tokens, one per socket and per GPU.
//
// The declarations follow AMD SMI's documented interface (amd_smi/amdsmi.h,
// AMD, MIT license); nothing here is AMD's code.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "vgpu/amd_chip.hpp"
#include "vgpu/amd_kfd.hpp"
#include "vgpu/amd_metrics.hpp"
#include "vgpu/machine.hpp"
#include "vgpu/ras.hpp"
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

// ---- Health and diagnostics ---------------------------------------------------------
//
// ECC and RAS, retired pages, XGMI, PCIe, sensors, partitions and processes,
// from the same state ROCm SMI's library (rocm_smi.cpp) and `amd-smi` read, so
// the three agree. What a profile's card does not have is answered
// AMDSMI_STATUS_NOT_SUPPORTED, as the driver answers where the sysfs file is
// missing. The profiles decide:
//   * ECC / RAS / bad pages: only where the profile has `ecc: true` (the
//     Instinct parts, HBM). Radeon cards (GDDR6, no ECC) have no RAS sysfs
//     and no bad-page record, so they are refused.
//   * XGMI: Instinct only; Radeon cards reach each other over PCIe.
//   * Fans: Radeon only; Instinct boards are passively cooled.
//   * Compute and memory partitions: CDNA3 and CDNA4 (MI300X, MI325X, MI350X).
// Assumptions with no profile datum behind them are marked where they are made.
// Layouts are those of amdsmi.h 26.2 (checked by size and offset below).
// Not modelled, still NOT_SUPPORTED: violation status,
// PCIe throughput, bad-page threshold, every setter and reset. The getters
// ROCm SMI's library answers (perf level, overdrive, busy percent, reserved
// pages, the compute processes) are answered here the same way.

namespace {

constexpr uint64_t kNone64 = ~uint64_t{0};

struct ErrorCount {  // amdsmi_error_count_t
  uint64_t correctable_count, uncorrectable_count, deferred_count, reserved[5];
};
struct RetiredPage {  // amdsmi_retired_page_record_t
  uint64_t page_address, page_size;
  int status;  // 0 reserved, 1 pending, 2 unreservable
};
struct XgmiInfo {  // amdsmi_xgmi_info_t
  uint8_t xgmi_lanes;
  uint64_t xgmi_hive_id, xgmi_node_id;
  uint32_t index, reserved[9];
};
struct XgmiLinks {  // amdsmi_xgmi_link_status_t
  uint32_t total_links;
  int status[8];  // 0 down, 1 up, 2 disabled
  uint64_t reserved[7];
};
struct PcieInfo {  // amdsmi_pcie_info_t
  struct {
    uint16_t max_pcie_width;
    uint32_t max_pcie_speed, pcie_interface_version;
    int slot_type;  // 0 PCIe, 1 OAM, 2 CEM, 3 unknown
    uint32_t max_pcie_interface_version;
    uint64_t reserved[9];
  } pcie_static;
  struct {
    uint16_t pcie_width;
    uint32_t pcie_speed, pcie_bandwidth;
    uint64_t pcie_replay_count, pcie_l0_to_recovery_count, pcie_replay_roll_over_count, pcie_nak_sent_count,
        pcie_nak_received_count;
    uint32_t pcie_lc_perf_other_end_recovery_count;
    uint64_t reserved[12];
  } pcie_metric;
  uint64_t reserved[32];
};
struct Frequencies {  // amdsmi_frequencies_t
  bool has_deep_sleep;
  uint32_t num_supported, current;
  uint64_t frequency[33];
};
struct PcieBandwidth {  // amdsmi_pcie_bandwidth_t
  Frequencies transfer_rate;
  uint32_t lanes[33];
};
struct PowerInfo {  // amdsmi_power_info_t
  uint64_t socket_power;
  uint32_t current_socket_power, average_socket_power;
  uint64_t gfx_voltage, soc_voltage, mem_voltage;
  uint32_t power_limit;
  uint64_t reserved[18];
};
struct PowerCapInfo {  // amdsmi_power_cap_info_t
  uint64_t power_cap, default_power_cap, dpm_cap, min_power_cap, max_power_cap, reserved[3];
};
struct ClkInfo {  // amdsmi_clk_info_t
  uint32_t clk, min_clk, max_clk;
  uint8_t clk_locked, clk_deep_sleep;
  uint32_t reserved[4];
};
struct ProcInfo {  // amdsmi_proc_info_t
  char name[256];
  uint32_t pid;
  uint64_t mem;
  struct {
    uint64_t gfx, enc;
    uint32_t reserved[12];
  } engine_usage;
  struct {
    uint64_t gtt_mem, cpu_mem, vram_mem;
    uint32_t reserved[10];
  } memory_usage;
  char container_name[256];
  uint32_t cu_occupancy, evicted_time, reserved[10];
};
static_assert(sizeof(ErrorCount) == 64 && sizeof(RetiredPage) == 24 && sizeof(XgmiInfo) == 64 &&
              sizeof(XgmiLinks) == 96 && sizeof(PcieInfo) == 512 && sizeof(PcieBandwidth) == 416 &&
              sizeof(PowerInfo) == 192 && sizeof(PowerCapInfo) == 64 && sizeof(ClkInfo) == 32 &&
              sizeof(ProcInfo) == 704);
static_assert(offsetof(XgmiInfo, xgmi_hive_id) == 8 && offsetof(XgmiInfo, index) == 24 &&
              offsetof(PcieInfo, pcie_static.slot_type) == 12 && offsetof(PcieInfo, pcie_metric.pcie_width) == 96 &&
              offsetof(PcieInfo, pcie_metric.pcie_replay_count) == 112 &&
              offsetof(PcieInfo, pcie_metric.pcie_lc_perf_other_end_recovery_count) == 152 &&
              offsetof(PowerInfo, power_limit) == 40 && offsetof(ProcInfo, pid) == 256 &&
              offsetof(ProcInfo, memory_usage) == 336 && offsetof(ProcInfo, container_name) == 400 &&
              offsetof(ProcInfo, cu_occupancy) == 656);

struct XcpMetrics {  // amdsmi_gpu_xcp_metrics_t
  uint32_t gfx_busy_inst[8];
  uint16_t jpeg_busy[40];
  uint16_t vcn_busy[4];
  uint64_t gfx_busy_acc[8];
  uint64_t gfx_below_host_limit_acc[8];
  uint64_t gfx_below_host_limit_ppt_acc[8];
  uint64_t gfx_below_host_limit_thm_acc[8];
  uint64_t gfx_low_utilization_acc[8];
  uint64_t gfx_below_host_limit_total_acc[8];
};
struct GpuMetrics {  // amdsmi_gpu_metrics_t, ROCm 7.2.0's (content revision 1.7)
  struct {
    uint16_t structure_size;
    uint8_t format_revision, content_revision;
  } common_header;
  uint16_t temperature_edge, temperature_hotspot, temperature_mem, temperature_vrgfx, temperature_vrsoc,
      temperature_vrmem;
  uint16_t average_gfx_activity, average_umc_activity, average_mm_activity;
  uint16_t average_socket_power;
  uint64_t energy_accumulator, system_clock_counter;
  uint16_t average_gfxclk_frequency, average_socclk_frequency, average_uclk_frequency, average_vclk0_frequency,
      average_dclk0_frequency, average_vclk1_frequency, average_dclk1_frequency;
  uint16_t current_gfxclk, current_socclk, current_uclk, current_vclk0, current_dclk0, current_vclk1,
      current_dclk1;
  uint32_t throttle_status;
  uint16_t current_fan_speed, pcie_link_width, pcie_link_speed;
  uint32_t gfx_activity_acc, mem_activity_acc;
  uint16_t temperature_hbm[4];
  uint64_t firmware_timestamp;
  uint16_t voltage_soc, voltage_gfx, voltage_mem;
  uint64_t indep_throttle_status;
  uint16_t current_socket_power;
  uint16_t vcn_activity[4];
  uint32_t gfxclk_lock_status;
  uint16_t xgmi_link_width, xgmi_link_speed;
  uint64_t pcie_bandwidth_acc, pcie_bandwidth_inst, pcie_l0_to_recov_count_acc, pcie_replay_count_acc,
      pcie_replay_rover_count_acc;
  uint64_t xgmi_read_data_acc[8], xgmi_write_data_acc[8];
  uint16_t current_gfxclks[8], current_socclks[4], current_vclk0s[4], current_dclk0s[4];
  uint16_t jpeg_activity[32];
  uint32_t pcie_nak_sent_count_acc, pcie_nak_rcvd_count_acc;
  uint64_t accumulation_counter, prochot_residency_acc, ppt_residency_acc, socket_thm_residency_acc,
      vr_thm_residency_acc, hbm_thm_residency_acc;
  uint16_t num_partition;
  XcpMetrics xcp_stats[8];
  uint32_t pcie_lc_perf_other_end_recovery;
  uint64_t vram_max_bandwidth;
  uint16_t xgmi_link_status[8];
};
// Sizes and offsets of amdsmi_gpu_metrics_t as compiled from rocm-7.2.0's amdsmi.h.
static_assert(sizeof(XcpMetrics) == 504 && sizeof(GpuMetrics) == 4544);
static_assert(offsetof(GpuMetrics, temperature_hotspot) == 6 && offsetof(GpuMetrics, average_gfx_activity) == 16 &&
              offsetof(GpuMetrics, energy_accumulator) == 24 && offsetof(GpuMetrics, system_clock_counter) == 32 &&
              offsetof(GpuMetrics, current_gfxclk) == 54 && offsetof(GpuMetrics, throttle_status) == 68 &&
              offsetof(GpuMetrics, pcie_link_width) == 74 && offsetof(GpuMetrics, gfx_activity_acc) == 80 &&
              offsetof(GpuMetrics, temperature_hbm) == 88 && offsetof(GpuMetrics, firmware_timestamp) == 96 &&
              offsetof(GpuMetrics, indep_throttle_status) == 112 && offsetof(GpuMetrics, current_socket_power) == 120 &&
              offsetof(GpuMetrics, vcn_activity) == 122 && offsetof(GpuMetrics, gfxclk_lock_status) == 132 &&
              offsetof(GpuMetrics, pcie_bandwidth_acc) == 144 && offsetof(GpuMetrics, pcie_replay_count_acc) == 168 &&
              offsetof(GpuMetrics, xgmi_read_data_acc) == 184 && offsetof(GpuMetrics, current_gfxclks) == 312 &&
              offsetof(GpuMetrics, jpeg_activity) == 352 && offsetof(GpuMetrics, pcie_nak_sent_count_acc) == 416 &&
              offsetof(GpuMetrics, accumulation_counter) == 424 && offsetof(GpuMetrics, num_partition) == 472 &&
              offsetof(GpuMetrics, xcp_stats) == 480 && offsetof(GpuMetrics, pcie_lc_perf_other_end_recovery) == 4512 &&
              offsetof(GpuMetrics, vram_max_bandwidth) == 4520 && offsetof(GpuMetrics, xgmi_link_status) == 4528);

// amdsmi_gpu_block_t: one bit each. The blocks an ECC card reports are the
// ones ROCm SMI's library reports (rocm_smi.cpp).
constexpr uint64_t kBlockUmc = 1, kBlockSdma = 2, kBlockGfx = 4, kBlockMmhub = 8, kBlockPcieBif = 0x20,
                   kBlockHdp = 0x40, kBlockXgmiWafl = 0x80;
constexpr uint64_t kEccBlocks = kBlockUmc | kBlockSdma | kBlockGfx | kBlockMmhub | kBlockPcieBif | kBlockHdp |
                                kBlockXgmiWafl;
enum RasState : int { kRasDisabled = 1, kRasEnabled = 6 };  // amdsmi_ras_err_state_t
enum TempType : int { kTempEdge = 0, kTempHotspot = 1, kTempVram = 2, kTempHbm0 = 3, kTempHbm3 = 6 };
enum TempMetric : int { kTempCurrent = 0, kTempMax = 1, kTempCritical = 5, kTempEmergency = 7 };
enum ClkType : int { kClkGfx = 0, kClkMem = 4 };   // amdsmi_clk_type_t (GFX is SYS)
constexpr int kOutOfResources = 15;

vgpu::ras::Counters ras_counters(const Sample& s, bool lifetime) {
  try {
    const vgpu::ras::State st = vgpu::ras::read(s.uuid);
    return lifetime ? st.lifetime : st.since_load;
  } catch (const std::exception&) {
    return {};
  }
}

uint64_t fnv(const char* s) {
  uint64_t h = 1469598103934665603ull;
  for (; *s; ++s) h = (h ^ static_cast<unsigned char>(*s)) * 1099511628211ull;
  return h;
}

// ECC by block: device memory is the UMC block, the on-chip memories GFX; the
// other blocks have no model, so they read zero where ECC is on. Block 0 is
// every block.
ErrorCount ecc_of(const Sample& s, uint64_t block) {
  using vgpu::ras::Location;
  using vgpu::ras::Severity;
  const vgpu::ras::Counters c = ras_counters(s, false);
  const auto memory = [&](Severity v) {
    return c.ecc[static_cast<uint32_t>(v)][static_cast<uint32_t>(Location::DeviceMemory)];
  };
  const auto count = [&](Severity v) -> uint64_t {
    if (block == kBlockUmc) return memory(v);
    if (block == kBlockGfx) return c.ecc_total(v) - memory(v);
    if (block == 0) return c.ecc_total(v);
    return 0;
  };
  ErrorCount e{};
  e.correctable_count = count(Severity::Corrected);
  e.uncorrectable_count = count(Severity::Uncorrected);
  // Assumption: no deferred errors are modelled (poison consumption records on
  // MI300-class cards), so the count is zero rather than N/A.
  return e;
}

bool partitionable(const Sample& d) {
  return std::strcmp(d.architecture, "cdna3") == 0 || std::strcmp(d.architecture, "cdna4") == 0;
}
uint32_t nbio(const Sample& d, const char* name) {
  vgpu::regs::RegisterSpace mmio(vgpu::regs::Space::AmdMmio, d);
  const vgpu::regs::Register* r = vgpu::regs::find(vgpu::regs::Space::AmdMmio, name);
  return r ? mmio.value(*r) : 0;
}

// GT/s and MT/s of each PCIe generation, 1 to 6 (index 0 pads). Gen 1's 2.5 GT/s
// rounds down in the GT/s the static info gives.
constexpr uint32_t kGts[] = {0, 2, 5, 8, 16, 32, 64};
constexpr uint32_t kMts[] = {0, 2500, 5000, 8000, 16000, 32000, 64000};

}  // namespace

// ---- ECC and RAS ------------------------------------------------------------------

// Totals over every block. A card without ECC has no count to give.
AMDSMI_API int amdsmi_get_gpu_total_ecc_count(void* h, ErrorCount* ec) {
  GPU(h, s, ec);
  if (!s.ecc_enabled) return kNotSupported;
  *ec = ecc_of(s, 0);
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_ecc_enabled(void* h, uint64_t* mask) {
  GPU(h, s, mask);
  *mask = s.ecc_enabled ? kEccBlocks : 0;
  return kSuccess;
}
// One block (a single bit of amdsmi_gpu_block_t); several is invalid.
AMDSMI_API int amdsmi_get_gpu_ecc_count(void* h, uint64_t block, ErrorCount* ec) {
  GPU(h, s, ec);
  if (block == 0 || (block & (block - 1))) return kInval;
  if (!s.ecc_enabled || !(block & kEccBlocks)) return kNotSupported;
  *ec = ecc_of(s, block);
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_ecc_status(void* h, uint64_t block, int* state) {
  GPU(h, s, state);
  if (block == 0 || (block & (block - 1))) return kInval;
  if (!(block & kEccBlocks)) return kNotSupported;
  *state = s.ecc_enabled ? kRasEnabled : kRasDisabled;
  return kSuccess;
}

// The pages taken out of service (reserved) and those marked and waiting for
// the next window (pending). The simulator keeps no addresses, so each page
// is numbered by its place. With no array, the count; with one, as many as it
// holds, and INSUFFICIENT_SIZE if there are more. No ECC, no record: the
// driver has no bad-page file on such a card.
AMDSMI_API int amdsmi_get_gpu_bad_page_info(void* h, uint32_t* num, RetiredPage* records) {
  GPU(h, s, num);
  if (!s.ecc_enabled) return kNotSupported;
  // Assumption: the profiles' HBM "row remapping" is how this simulator takes
  // memory out of service, where amdgpu retires pages (its RAS EEPROM's bad
  // pages), so a remapped row is listed as a page. The retired and remapped
  // counts are aggregate (lifetime) state, kept across driver loads. A pending
  // retirement (a flag until the next driver load) is the newest page.
  const vgpu::ras::Counters c = ras_counters(s, true);
  const uint32_t total = static_cast<uint32_t>(c.retired_sbe + c.retired_dbe + c.rows_correctable + c.rows_uncorrectable);
  const uint32_t pending = (c.retired_pending || c.rows_pending) && total ? 1 : 0;
  const uint32_t reserved = total - pending;
  const uint32_t k = records ? std::min(*num, total) : total;
  for (uint32_t i = 0; records && i < k; ++i) records[i] = {uint64_t{i} << 12, 4096, i < reserved ? 0 : 1};
  const bool fits = !records || *num >= total;
  *num = k;
  return fits ? kSuccess : kInsufficientSize;
}

// ---- XGMI -----------------------------------------------------------------------------

// Instinct GPUs only. Every Instinct GPU of the machine is in one hive, whose
// id is KFD's (topology `hive_id`: a hash of the first GPU's UUID). Assumption:
// x16 XGMI links (the lane count of an Infinity Fabric link on MI300-class).
AMDSMI_API int amdsmi_get_xgmi_info(void* h, XgmiInfo* info) {
  GPU(h, s, info);
  if (!instinct(s)) return kNotSupported;
  std::memset(info, 0, sizeof *info);
  info->xgmi_lanes = 16;
  info->xgmi_hive_id = fnv(machine()[0].uuid);
  info->xgmi_node_id = fnv(s.uuid);  // ROCm SMI's unique id
  info->index = static_cast<uint32_t>(s_i);
  return kSuccess;
}
AMDSMI_API int amdsmi_gpu_xgmi_error_status(void* h, int* status) {
  GPU(h, s, status);
  if (!instinct(s)) return kNotSupported;
  *status = 0;  // AMDSMI_XGMI_STATUS_NO_ERRORS: no XGMI error is modelled
  return kSuccess;
}
// Assumption: one link up to each other Instinct GPU of the machine (a fully
// connected MI300X node has seven), capped at the structure's eight.
AMDSMI_API int amdsmi_get_gpu_xgmi_link_status(void* h, XgmiLinks* links) {
  GPU(h, s, links);
  if (!instinct(s)) return kNotSupported;
  std::memset(links, 0, sizeof *links);
  uint32_t peers = 0;
  for (const Sample& m : machine()) peers += instinct(m);
  peers = peers ? peers - 1 : 0;
  links->total_links = std::min<uint32_t>(peers, 8);
  for (uint32_t i = 0; i < links->total_links; ++i) links->status[i] = 1;  // AMDSMI_XGMI_LINK_UP
  return kSuccess;
}

// ---- PCI Express ----------------------------------------------------------------------

AMDSMI_API int amdsmi_get_pcie_info(void* h, PcieInfo* info) {
  GPU(h, s, info);
  std::memset(info, 0, sizeof *info);
  const vgpu::ras::Counters c = ras_counters(s, false);
  using vgpu::ras::Pcie;
  info->pcie_static.max_pcie_width = static_cast<uint16_t>(s.pcie_width_max);
  info->pcie_static.max_pcie_speed = kGts[std::min<uint32_t>(s.pcie_gen_max, 6)];
  info->pcie_static.pcie_interface_version = s.pcie_gen_max;
  info->pcie_static.max_pcie_interface_version = s.pcie_gen_max;
  info->pcie_static.slot_type = instinct(s) ? 1 : 0;  // OAM boards (as oam_id says) or PCIe cards
  info->pcie_metric.pcie_width = static_cast<uint16_t>(s.pcie_width);
  info->pcie_metric.pcie_speed = kMts[std::min<uint32_t>(s.pcie_gen, 6)];
  info->pcie_metric.pcie_bandwidth = kNone32;  // no traffic is metered, as ROCm SMI says
  info->pcie_metric.pcie_replay_count = c.pcie[static_cast<uint32_t>(Pcie::Replay)];
  info->pcie_metric.pcie_l0_to_recovery_count = c.pcie[static_cast<uint32_t>(Pcie::L0ToRecovery)];
  info->pcie_metric.pcie_replay_roll_over_count = c.pcie[static_cast<uint32_t>(Pcie::ReplayRollover)];
  info->pcie_metric.pcie_nak_sent_count = c.pcie[static_cast<uint32_t>(Pcie::NaksSent)];
  info->pcie_metric.pcie_nak_received_count = c.pcie[static_cast<uint32_t>(Pcie::NaksReceived)];
  info->pcie_metric.pcie_lc_perf_other_end_recovery_count = 0;
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_pci_replay_counter(void* h, uint64_t* counter) {
  GPU(h, s, counter);
  *counter = ras_counters(s, false).pcie[static_cast<uint32_t>(vgpu::ras::Pcie::Replay)];
  return kSuccess;
}
// Each generation up to the card's, in transfers a second, the one trained at
// marked, as ROCm SMI's library gives it.
AMDSMI_API int amdsmi_get_gpu_pci_bandwidth(void* h, PcieBandwidth* bw) {
  GPU(h, s, bw);
  std::memset(bw, 0, sizeof *bw);
  const uint32_t gens = std::min<uint32_t>(std::max<uint32_t>(s.pcie_gen_max, 1), 6);
  bw->transfer_rate.num_supported = gens;
  for (uint32_t g = 0; g < gens; ++g) {
    bw->transfer_rate.frequency[g] = uint64_t{kMts[g + 1]} * 1000000;
    bw->lanes[g] = s.pcie_width;
  }
  bw->transfer_rate.current = std::min<uint32_t>(std::max<uint32_t>(s.pcie_gen, 1), gens) - 1;
  return kSuccess;
}

// ---- Sensors --------------------------------------------------------------------------

// Degrees Celsius (ROCm SMI's are millidegrees). An Instinct card has no edge
// sensor; a Radeon's edge and junction are the one die temperature the model
// keeps. Memory is where the profile says real cards report it. Assumption:
// an Instinct card's four HBM stack sensors all read that one memory sensor.
// Limits: the slowdown threshold, only for the hotspot, and only where the
// profile has one (mi325x's is 0, unknown, so it is refused).
AMDSMI_API int amdsmi_get_temp_metric(void* h, int type, int metric, int64_t* t) {
  GPU(h, s, t);
  if (type < 0 || type > 249 || metric < 0 || metric > 14) return kInval;   // amdsmi_temperature_type_t, _metric_t
  uint32_t c = 0;
  if (type == kTempHotspot) c = s.temperature_c;
  else if (type == kTempEdge && !instinct(s)) c = s.temperature_c;
  else if (type == kTempVram && s.has_memory_temperature) c = s.temperature_mem_c ? s.temperature_mem_c : s.temperature_c;
  else if (type >= kTempHbm0 && type <= kTempHbm3 && instinct(s) && s.has_memory_temperature)
    c = s.temperature_mem_c ? s.temperature_mem_c : s.temperature_c;
  else return kNotSupported;
  if (metric == kTempCurrent) *t = c;
  else if ((metric == kTempMax || metric == kTempCritical || metric == kTempEmergency) && type == kTempHotspot && s.temperature_max_c)
    *t = s.temperature_max_c;
  else return kNotSupported;
  return kSuccess;
}

// Watts. A CDNA3 or newer card gives its current socket power, the rest (Radeon,
// MI200) an average, as the header's field comments say; the other is
// UINT32_MAX. No SoC or memory voltage is modelled.
AMDSMI_API int amdsmi_get_power_info(void* h, PowerInfo* info) {
  GPU(h, s, info);
  std::memset(info, 0, sizeof *info);
  const uint32_t w = s.power_mw / 1000;
  const bool current = partitionable(s);
  info->socket_power = w;
  info->current_socket_power = current ? w : kNone32;
  info->average_socket_power = current ? kNone32 : w;
  info->gfx_voltage = s.voltage_mv;
  info->soc_voltage = kNone64;
  info->mem_voltage = kNone64;
  info->power_limit = s.power_limit_mw / 1000;
  return kSuccess;
}
// Microwatts. The minimum is 0 and no DPM cap is modelled, as in ROCm SMI's.
AMDSMI_API int amdsmi_get_power_cap_info(void* h, uint32_t sensor, PowerCapInfo* info) {
  GPU(h, s, info);
  if (sensor != 0) return kNotSupported;
  std::memset(info, 0, sizeof *info);
  info->power_cap = info->default_power_cap = info->max_power_cap = uint64_t{s.power_limit_mw} * 1000;
  return kSuccess;
}
// Fans on amdgpu's 0-255 scale; Instinct boards are passively cooled. No tach.
AMDSMI_API int amdsmi_get_gpu_fan_speed(void* h, uint32_t sensor, int64_t* speed) {
  GPU(h, s, speed);
  if (instinct(s) || sensor != 0) return kNotSupported;
  *speed = int64_t{s.fan_percent} * 255 / 100;
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_fan_speed_max(void* h, uint32_t sensor, uint64_t* max) {
  GPU(h, s, max);
  if (instinct(s) || sensor != 0) return kNotSupported;
  *max = 255;
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_fan_rpms(void* h, uint32_t, int64_t* rpm) {
  GPU(h, s, rpm);
  return kNotSupported;
}
// The graphics clock and the memory clock; the others are not modelled. The
// minimum is the clock model's idle level, as ROCm SMI's levels and amd-smi.
AMDSMI_API int amdsmi_get_clock_info(void* h, int type, ClkInfo* info) {
  GPU(h, s, info);
  std::memset(info, 0, sizeof *info);
  if (type == kClkGfx) {
    info->clk = s.sm_clock_mhz;
    info->max_clk = s.sm_clock_max_mhz;
    info->min_clk = s.sm_clock_max_mhz / 6;
    info->clk_deep_sleep = s.utilization_gpu == 0;
  } else if (type == kClkMem) {
    info->clk = s.mem_clock_mhz;
    info->max_clk = s.mem_clock_max_mhz;
    info->min_clk = s.mem_clock_max_mhz / 5;
  } else {
    return kNotSupported;
  }
  return kSuccess;
}
// The clock levels: two, as ROCm SMI's library gives them -- the clock the card
// runs at below its most (or its idle where it is at the most), and the most.
// Frequencies are in Hz: the header's field comment says MHz, but its function
// documentation says Hz, and AMD's implementation passes ROCm SMI's levels
// (Hz) through, which the CLI divides by 1e6 (amd-smi's fclk).
namespace {
// The levels a caller restricted a clock to with amdsmi_set_clk_freq: a mask
// for each GPU and clock, held by this process (the machine's sample is
// shared; what a process asks of the driver is its own request).
std::mutex g_clk_mu;
uint64_t g_clk_mask[256][2];   // [GPU][0: graphics, 1: memory]; 0 = unrestricted

int clock_index(int type) { return type == kClkGfx ? 0 : type == kClkMem ? 1 : -1; }
}  // namespace

AMDSMI_API int amdsmi_get_clk_freq(void* h, int type, Frequencies* f) {
  GPU(h, s, f);
  const int idx = clock_index(type);
  if (idx < 0) return kNotSupported;   // fabric, SoC, video, display: not modelled
  const uint32_t max_mhz = idx == 0 ? s.sm_clock_max_mhz : s.mem_clock_max_mhz;
  const uint32_t now_mhz = idx == 0 ? s.sm_clock_mhz : s.mem_clock_mhz;
  const uint32_t idle_mhz = max_mhz / (idx == 0 ? 6 : 5);
  std::memset(f, 0, sizeof *f);
  const bool at_max = now_mhz >= max_mhz;
  f->num_supported = 2;
  f->frequency[0] = uint64_t{at_max ? idle_mhz : now_mhz} * 1000000;
  f->frequency[1] = uint64_t{max_mhz} * 1000000;
  f->current = at_max ? 1 : 0;
  // A request for one level pins the clock there.
  std::lock_guard<std::mutex> lock(g_clk_mu);
  const uint64_t m = g_clk_mask[s_i][idx];
  if (m == 1 || m == 2) f->current = m == 1 ? 0 : 1;
  return kSuccess;
}
// Limits the clock to the levels whose bits are set. A mask with a level the
// clock does not have is out of bounds, as is none at all.
AMDSMI_API int amdsmi_set_clk_freq(void* h, int type, uint64_t mask) {
  if (!g_init) return kNotInit;
  const int i = ordinal(h, kGpuBase);
  if (i < 0) return kNotFound;
  const int idx = clock_index(type);
  if (idx < 0) return kNotSupported;
  if (mask == 0 || (mask & ~uint64_t{3})) return kInputOutOfBounds;
  std::lock_guard<std::mutex> lock(g_clk_mu);
  g_clk_mask[i][idx] = mask == 3 ? 0 : mask;
  return kSuccess;
}

// The energy accumulator: ticks of `resolution` micro joules (15.3), as ROCm
// SMI's rsmi_dev_energy_count_get gives it, from the one accumulator both
// libraries share (vgpu/amd_metrics.hpp). Only Instinct GPUs have it: a
// Radeon's driver has no energy file.
AMDSMI_API int amdsmi_get_energy_count(void* h, uint64_t* energy, float* resolution, uint64_t* timestamp) {
  GPU(h, s, energy);
  if (!timestamp) return kInval;
  if (!instinct(s)) return kNotSupported;
  const vgpu::amd::EnergyReading e = vgpu::amd::energy_counter(s);
  *energy = e.ticks;
  *timestamp = e.timestamp_ns;
  if (resolution) *resolution = vgpu::amd::kEnergyTickUj;
  return kSuccess;
}
// The driver's gpu_metrics table (v1.5, as the simulated driver publishes it in
// sysfs) in AMD SMI's public structure, as the library copies it: header from
// the table, each field the table has copied across, every other member all
// ones (N/A). The single-value clocks are the first of the table's arrays, as
// the library's v1.5 compatibility copy does. Where the table carries no energy
// accumulator the Instinct GPU's own counter is put in, the one
// amdsmi_get_energy_count gives. A Radeon's table is the same v1.5 layout.
AMDSMI_API int amdsmi_get_gpu_metrics_info(void* h, GpuMetrics* out) {
  GPU(h, s, out);
  vgpu::amd::MetricsV15 t;
  const std::string bytes = vgpu::amd::gpu_metrics(s, ras_counters(s, false));
  if (bytes.size() != sizeof t) return kNotSupported;
  std::memcpy(&t, bytes.data(), sizeof t);
  std::memset(out, 0xFF, sizeof *out);
  out->common_header = {t.structure_size, t.format_revision, t.content_revision};
  out->temperature_hotspot = t.temperature_hotspot;
  out->temperature_mem = t.temperature_mem;
  out->temperature_vrsoc = t.temperature_vrsoc;
  out->current_socket_power = t.curr_socket_power;
  out->average_gfx_activity = t.average_gfx_activity;
  out->average_umc_activity = t.average_umc_activity;
  std::memcpy(out->vcn_activity, t.vcn_activity, sizeof t.vcn_activity);
  std::memcpy(out->jpeg_activity, t.jpeg_activity, sizeof t.jpeg_activity);
  out->energy_accumulator = t.energy_accumulator;
  out->system_clock_counter = t.system_clock_counter;
  if (instinct(s) && t.energy_accumulator == kNone64) {
    const vgpu::amd::EnergyReading e = vgpu::amd::energy_counter(s);
    out->energy_accumulator = e.ticks;
    out->system_clock_counter = e.timestamp_ns;
  }
  out->throttle_status = t.throttle_status;
  out->gfxclk_lock_status = t.gfxclk_lock_status;
  out->pcie_link_width = t.pcie_link_width;
  out->pcie_link_speed = t.pcie_link_speed;
  out->xgmi_link_width = t.xgmi_link_width;
  out->xgmi_link_speed = t.xgmi_link_speed;
  out->gfx_activity_acc = t.gfx_activity_acc;
  out->mem_activity_acc = t.mem_activity_acc;
  out->pcie_bandwidth_acc = t.pcie_bandwidth_acc;
  out->pcie_bandwidth_inst = t.pcie_bandwidth_inst;
  out->pcie_l0_to_recov_count_acc = t.pcie_l0_to_recov_count_acc;
  out->pcie_replay_count_acc = t.pcie_replay_count_acc;
  out->pcie_replay_rover_count_acc = t.pcie_replay_rover_count_acc;
  out->pcie_nak_sent_count_acc = t.pcie_nak_sent_count_acc;
  out->pcie_nak_rcvd_count_acc = t.pcie_nak_rcvd_count_acc;
  std::memcpy(out->xgmi_read_data_acc, t.xgmi_read_data_acc, sizeof t.xgmi_read_data_acc);
  std::memcpy(out->xgmi_write_data_acc, t.xgmi_write_data_acc, sizeof t.xgmi_write_data_acc);
  out->firmware_timestamp = t.firmware_timestamp;
  std::memcpy(out->current_gfxclks, t.current_gfxclk, sizeof t.current_gfxclk);
  std::memcpy(out->current_socclks, t.current_socclk, sizeof t.current_socclk);
  std::memcpy(out->current_vclk0s, t.current_vclk0, sizeof t.current_vclk0);
  std::memcpy(out->current_dclk0s, t.current_dclk0, sizeof t.current_dclk0);
  out->current_uclk = t.current_uclk;
  out->current_gfxclk = out->current_gfxclks[0];
  out->current_socclk = out->current_socclks[0];
  out->current_vclk0 = out->current_vclk0s[0];
  out->current_vclk1 = out->current_vclk0s[1];
  out->current_dclk0 = out->current_dclk0s[0];
  out->current_dclk1 = out->current_dclk0s[1];
  return kSuccess;
}

// The graphics core's voltage now, in millivolts.
AMDSMI_API int amdsmi_get_gpu_volt_metric(void* h, int type, int metric, int64_t* mv) {
  GPU(h, s, mv);
  if (type != 0 || metric != 0) return kNotSupported;  // VDDGFX, current
  *mv = s.voltage_mv;
  return kSuccess;
}

// ---- Partitions -----------------------------------------------------------------------

// As the device's NBIO registers hold them; only CDNA3 and CDNA4 have any.
AMDSMI_API int amdsmi_get_gpu_compute_partition(void* h, char* out, uint32_t len) {
  GPU(h, s, out);
  if (!partitionable(s)) return kNotSupported;
  static const char* const kCompute[] = {"SPX", "DPX", "TPX", "QPX", "CPX"};
  const uint32_t px = (nbio(s, "nbio_partition_compute_status") >> 4) & 0xF;
  return put_string(out, len, px < 5 ? kCompute[px] : "UNKNOWN");
}
AMDSMI_API int amdsmi_get_gpu_memory_partition(void* h, char* out, uint32_t len) {
  GPU(h, s, out);
  if (!partitionable(s)) return kNotSupported;
  const int nps = __builtin_ffs(static_cast<int>((nbio(s, "nbio_partition_mem_status") >> 4) & 0xFF));
  return put_string(out, len, nps ? "NPS" + std::to_string(nps) : "UNKNOWN");
}

// ---- Processes ------------------------------------------------------------------------

// The processes using the GPU, as the machine's telemetry lists them. With
// *max 0 the count comes back; where the list is too short it holds as many as
// fit, *max is the count, and OUT_OF_RESOURCES says so, as the header does.
AMDSMI_API int amdsmi_get_gpu_process_list(void* h, uint32_t* max, ProcInfo* list) {
  GPU(h, s, max);
  const uint32_t n = std::min<uint32_t>(s.proc_count, vgpu::telemetry::kMaxProcs);
  if (*max == 0 || !list) {
    *max = n;
    return kSuccess;
  }
  const uint32_t k = std::min(*max, n);
  for (uint32_t i = 0; i < k; ++i) {
    ProcInfo& p = list[i];
    std::memset(&p, 0, sizeof p);
    std::snprintf(p.name, sizeof p.name, "%s", s.procs[i].name);
    p.pid = s.procs[i].pid;
    p.mem = s.procs[i].used_bytes;
    p.memory_usage.vram_mem = s.procs[i].used_bytes;
    p.cu_occupancy = kNone32;  // not modelled, as ROCm SMI's compute process list says
  }
  const bool fits = *max >= n;
  *max = n;
  return fits ? kSuccess : kOutOfResources;
}

// ---- What ROCm SMI's library answers, in AMD SMI's names ---------------------------------
//
// The same machine state, the same answers as rocm_smi.cpp gives them, so
// `amd-smi` and `rocm-smi` agree.

struct VramUsage {  // amdsmi_vram_usage_t
  uint32_t vram_total, vram_used, reserved[2];   // in MB
};
struct ComputeProcess {  // amdsmi_process_info_t
  uint32_t process_id;
  uint64_t vram_usage;    // MB
  uint64_t sdma_usage;    // microseconds
  uint32_t cu_occupancy;  // percent
  uint32_t evicted_time;  // ms
};
struct MetricsHeader {  // amd_metrics_table_header_t
  uint16_t structure_size;
  uint8_t format_revision, content_revision;
};
static_assert(sizeof(VramUsage) == 16 && sizeof(ComputeProcess) == 32 && sizeof(MetricsHeader) == 4);
static_assert(offsetof(ComputeProcess, vram_usage) == 8 && offsetof(ComputeProcess, sdma_usage) == 16 &&
              offsetof(ComputeProcess, cu_occupancy) == 24 && offsetof(ComputeProcess, evicted_time) == 28);

// Memory in megabytes, as the structure says (the other queries give bytes).
AMDSMI_API int amdsmi_get_gpu_vram_usage(void* h, VramUsage* u) {
  GPU(h, s, u);
  std::memset(u, 0, sizeof *u);
  u->vram_total = static_cast<uint32_t>(s.vram_total_bytes >> 20);
  u->vram_used = static_cast<uint32_t>(s.vram_used_bytes >> 20);
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_busy_percent(void* h, uint32_t* pct) {
  GPU(h, s, pct);
  *pct = s.utilization_gpu;
  return kSuccess;
}
// No knob is modelled, so the card is as the driver leaves it: the performance
// level "auto" (0) and no overdrive.
AMDSMI_API int amdsmi_get_gpu_perf_level(void* h, int* level) {
  GPU(h, s, level);
  *level = 0;   // AMDSMI_DEV_PERF_LEVEL_AUTO
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_overdrive_level(void* h, uint32_t* od) {
  GPU(h, s, od);
  *od = 0;
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_mem_overdrive_level(void* h, uint32_t* od) {
  GPU(h, s, od);
  *od = 0;
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_metrics_header_info(void* h, MetricsHeader* header) {
  GPU(h, s, header);
  vgpu::amd::MetricsV15 t;
  const std::string bytes = vgpu::amd::gpu_metrics(s, ras_counters(s, false));
  if (bytes.size() != sizeof t) return kNotSupported;
  std::memcpy(&t, bytes.data(), sizeof t);
  *header = {t.structure_size, t.format_revision, t.content_revision};
  return kSuccess;
}
// The pages retired for uncorrectable errors, the list amdsmi_get_gpu_bad_page_info
// gives: the driver keeps one record of them.
AMDSMI_API int amdsmi_get_gpu_memory_reserved_pages(void* h, uint32_t* num, RetiredPage* records) {
  return amdsmi_get_gpu_bad_page_info(h, num, records);
}

namespace {
// Every process using an AMD GPU: the GPUs it uses (by ordinal) and its VRAM.
struct ComputeUse {
  uint64_t vram = 0;
  std::vector<uint32_t> gpus;
};
std::map<uint32_t, ComputeUse> compute_processes() {
  std::map<uint32_t, ComputeUse> out;
  const std::vector<Sample> m = machine();
  for (uint32_t k = 0; k < m.size(); ++k)
    for (uint32_t p = 0; p < m[k].proc_count && p < vgpu::telemetry::kMaxProcs; ++p) {
      ComputeUse& e = out[m[k].procs[p].pid];
      e.vram += m[k].procs[p].used_bytes;
      e.gpus.push_back(k);
    }
  return out;
}
ComputeProcess compute_info(uint32_t pid, const ComputeUse& u) {
  // SDMA time is not kept and the eviction time is not modelled; the CU
  // occupancy is N/A, as ROCm SMI's compute process list says.
  return {pid, u.vram >> 20, 0, kNone32, kNone32};
}
}  // namespace

// Where `procs` is null the count comes back; where it is too short it holds
// as many as fit, and *num_items says how many were written.
AMDSMI_API int amdsmi_get_gpu_compute_process_info(ComputeProcess* procs, uint32_t* num_items) {
  if (!g_init) return kNotInit;
  if (!num_items) return kInval;
  const auto all = compute_processes();
  uint32_t k = 0;
  if (procs)
    for (const auto& [pid, u] : all) {
      if (k >= *num_items) break;
      procs[k++] = compute_info(pid, u);
    }
  const bool fits = !procs || *num_items >= all.size();
  *num_items = procs ? k : static_cast<uint32_t>(all.size());
  return fits ? kSuccess : kInsufficientSize;
}
AMDSMI_API int amdsmi_get_gpu_compute_process_info_by_pid(uint32_t pid, ComputeProcess* proc) {
  if (!g_init) return kNotInit;
  if (!proc) return kInval;
  const auto all = compute_processes();
  const auto it = all.find(pid);
  if (it == all.end()) return kNotFound;
  *proc = compute_info(pid, it->second);
  return kSuccess;
}
AMDSMI_API int amdsmi_get_gpu_compute_process_gpus(uint32_t pid, uint32_t* indices, uint32_t* num_devices) {
  if (!g_init) return kNotInit;
  if (!num_devices) return kInval;
  const auto all = compute_processes();
  const auto it = all.find(pid);
  if (it == all.end()) return kNotFound;
  const auto& g = it->second.gpus;
  if (indices)
    for (uint32_t k = 0; k < std::min<size_t>(*num_devices, g.size()); ++k) indices[k] = g[k];
  const bool fits = !indices || *num_devices >= g.size();
  *num_devices = indices ? std::min<uint32_t>(*num_devices, static_cast<uint32_t>(g.size()))
                         : static_cast<uint32_t>(g.size());
  return fits ? kSuccess : kInsufficientSize;
}
