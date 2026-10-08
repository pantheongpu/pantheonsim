// librocm_smi64: ROCm SMI's library, answered for the simulated GPUs -- what
// ROCm's own rocm-smi, RCCL and AMD's test clients ask it: the devices and
// where each sits on the PCI bus, what links them, and each one's identity,
// temperatures, power, clocks, memory, fans, voltage, PCIe link, ECC counts,
// partitions and processes.
//
// The answers come from the same state `vgpu smi` reads (the machine's
// telemetry, its RAS counters and registers), so ROCm's rocm-smi over this
// library and VirtualGPU's rocm-smi agree; the device list and bus addresses
// are the HIP runtime's, so HIP and ROCm SMI agree too. What a simulated GPU
// has no value for -- a VBIOS, firmware, serial number --
// is RSMI_STATUS_NOT_SUPPORTED, as on a card without it,
// and setting anything is refused the same way: nothing here is a knob.
//
// RCCL learns the topology one of two ways: by reading the AMD kernel
// driver's topology under /sys/class/kfd (its default), or through this
// library (RCCL_USE_ROCM_SMI_LIB=1). Under WSL it does neither. On any other
// Linux machine there is no /sys/class/kfd without an AMD GPU, and RCCL's
// initialization fails ("internal error"), so this library, once loaded,
// asks RCCL for the second way unless the environment already says which --
// as it does in a vgpu session, which has a /sys/class/kfd of its own.
// RCCL reads that setting when it first initializes, after the libraries it
// links -- this one among them -- are loaded.
//
// The declarations follow ROCm SMI's documented interface
// (rocm_smi/rocm_smi.h, AMD, NCSA license); nothing here is AMD's code.
#include <sys/stat.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "vgpu/amd_chip.hpp"
#include "vgpu/amd_metrics.hpp"
#include "vgpu/machine.hpp"
#include "vgpu/ras.hpp"
#include "vgpu/regs.hpp"
#include "vgpu/telemetry.hpp"

namespace {

// rsmi_status_t, in rocm_smi.h's order.
enum Status : int {
  kSuccess = 0,
  kInvalidArgs = 1,
  kNotSupported = 2,
  kFileError = 3,
  kPermission = 4,
  kInputOutOfBounds = 7,
  kInitError = 8,
  kNotFound = 10,
  kInsufficientSize = 11,
};

// RSMI_IO_LINK_TYPE.
enum LinkType : int {
  kLinkUndefined = 0,
  kLinkPcie = 1,
  kLinkXgmi = 2,
};

struct Version {  // rsmi_version_t
  uint32_t major, minor, patch;
  const char* build;
};

constexpr uint32_t kMaxFrequencies = 33;   // RSMI_MAX_NUM_FREQUENCIES
struct Frequencies {                        // rsmi_frequencies_t
  bool has_deep_sleep;
  uint32_t num_supported;
  uint32_t current;
  uint64_t frequency[kMaxFrequencies];
};
struct PcieBandwidth {                      // rsmi_pcie_bandwidth_t
  Frequencies transfer_rate;
  uint32_t lanes[kMaxFrequencies];
};
struct ErrorCount {                         // rsmi_error_count_t
  uint64_t correctable_err, uncorrectable_err;
};
struct ProcessInfo {                        // rsmi_process_info_t
  uint32_t process_id, pasid;
  uint64_t vram_usage, sdma_usage;
  uint32_t cu_occupancy;
};
struct RetiredPage {                        // rsmi_retired_page_record_t
  uint64_t page_address, page_size;
  int status;                               // RSMI_MEM_PAGE_STATUS_RETIRED is 1
};
constexpr uint32_t kCuOccupancyInvalid = 0xFFFFFFFF;

// The enums the queries take.
enum TempType { kTempEdge = 0, kTempJunction = 1, kTempMemory = 2 };
enum TempMetric { kTempCurrent = 0, kTempMax = 1, kTempCritical = 5 };
enum ClkType { kClkSys = 0, kClkMem = 4 };
enum MemType { kMemVram = 0, kMemVisVram = 1, kMemGtt = 2 };
enum VoltMetric { kVoltCurrent = 0 };
enum VoltType { kVoltVddgfx = 0 };
enum PowerType { kPowerAverage = 0, kPowerCurrent = 1 };
enum RasState { kRasDisabled = 1, kRasEnabled = 6 };
enum XgmiStatus { kXgmiNoErrors = 0 };
// rsmi_gpu_block_t: one bit a block.
constexpr uint64_t kBlockUmc = 1, kBlockSdma = 2, kBlockGfx = 4, kBlockMmhub = 8, kBlockPcieBif = 0x20,
                   kBlockHdp = 0x40, kBlockXgmiWafl = 0x80;
constexpr uint64_t kBlocks = kBlockUmc | kBlockSdma | kBlockGfx | kBlockMmhub | kBlockPcieBif | kBlockHdp | kBlockXgmiWafl;

using Sample = vgpu::telemetry::DeviceSample;

}  // namespace

#define RSMI_API extern "C" __attribute__((visibility("default")))

namespace {

// The AMD GPUs of the machine, as `vgpu smi` reads it: the devices HIP counts,
// in HIP's order, on the buses HIP reports. Read without starting the HIP
// runtime, which would make the tool asking one of the machine's workloads
// (its clocks and power up, its GPU busy).
std::vector<Sample> machine() {
  std::vector<Sample> out;
  vgpu::telemetry::Shared snap{};
  if (!vgpu::read_machine(&snap)) return out;
  vgpu::drop_lost_amd(&snap);
  for (uint32_t i = 0; i < snap.device_count; ++i)
    if (std::strcmp(snap.devices[i].vendor, "amd") == 0) out.push_back(snap.devices[i]);
  return out;
}

int count() { return static_cast<int>(machine().size()); }

bool valid(uint32_t d) { return d < static_cast<uint32_t>(count()); }

bool sample(uint32_t d, Sample* out) {
  const std::vector<Sample> m = machine();
  if (d >= m.size()) return false;
  *out = m[d];
  return true;
}

// Instinct GPUs (CDNA) are joined by Infinity Fabric in the machines they
// come in; Radeon ones (RDNA) by PCI Express alone.
bool instinct(uint32_t d) {
  Sample s{};
  return sample(d, &s) && s.architecture[0] == 'c';
}

// Fills a buffer of `len` bytes with a string, as ROCm SMI does: truncated
// and terminated, or INSUFFICIENT_SIZE where it does not fit.
int put_string(char* buf, size_t len, const std::string& s) {
  if (!buf || !len) return kInvalidArgs;
  std::snprintf(buf, len, "%s", s.c_str());
  return s.size() + 1 > len ? kInsufficientSize : kSuccess;
}

uint64_t fnv(const char* s) {
  uint64_t h = 1469598103934665603ull;
  for (; *s; ++s) h = (h ^ static_cast<unsigned char>(*s)) * 1099511628211ull;
  return h;
}

// Two clock levels, in Hz: the clock the card runs at below its most (the
// clock model's now, or its idle where it is at the most) and the most, the
// current one marked -- so the level a tool reads as current is the clock
// `vgpu smi` reports.
void two_levels(Frequencies* f, uint32_t idle_mhz, uint32_t max_mhz, uint32_t now_mhz) {
  std::memset(f, 0, sizeof *f);
  const bool at_max = now_mhz >= max_mhz;
  f->num_supported = 2;
  f->frequency[0] = uint64_t{at_max ? idle_mhz : now_mhz} * 1000000;
  f->frequency[1] = uint64_t{max_mhz} * 1000000;
  f->current = at_max ? 1 : 0;
}

// The partition modes amdgpu reads from NBIO, as the device's registers hold
// them; only CDNA3 and CDNA4 have any.
bool partitionable(const Sample& d) {
  return std::strcmp(d.architecture, "cdna3") == 0 || std::strcmp(d.architecture, "cdna4") == 0;
}
uint32_t nbio(const Sample& d, const char* name) {
  vgpu::regs::RegisterSpace mmio(vgpu::regs::Space::AmdMmio, d);
  const vgpu::regs::Register* r = vgpu::regs::find(vgpu::regs::Space::AmdMmio, name);
  return r ? mmio.value(*r) : 0;
}

// Tells RCCL to ask this library rather than read /sys/class/kfd: where
// there is none to read, and inside a vgpu session, where the topology there
// is either the session's own copy of what this library says or the host's
// real GPUs, not the simulated ones. An environment that already chose is
// left alone.
__attribute__((constructor)) void prefer_this_library() {
  struct stat st{};
  if (std::getenv("VGPU_SESSION") || stat("/sys/class/kfd/kfd/topology/nodes", &st) != 0)
    setenv("RCCL_USE_ROCM_SMI_LIB", "1", 0);
}

// Everything that looks a device up: the arguments checked, then the reading.
#define DEVICE(d, s)                        \
  Sample s{};                               \
  if (!valid(d)) return kInputOutOfBounds;  \
  if (!sample(d, &s)) return kNotFound

}  // namespace

// ---- The library and the devices -----------------------------------------

RSMI_API int rsmi_init(uint64_t) { return kSuccess; }
RSMI_API int rsmi_shut_down() { return kSuccess; }

RSMI_API int rsmi_num_monitor_devices(uint32_t* n) {
  if (!n) return kInvalidArgs;
  *n = static_cast<uint32_t>(count());
  return kSuccess;
}

RSMI_API int rsmi_version_get(Version* v) {
  if (!v) return kInvalidArgs;
  *v = {7, 8, 0, "VirtualGPU"};
  return kSuccess;
}

// The driver's version (RSMI_SW_COMP_DRIVER, the only component): amdgpu's,
// or the kernel's for an in-tree driver, as ROCm SMI reports it.
RSMI_API int rsmi_version_str_get(int component, char* out, uint32_t len) {
  if (component != 0) return kInvalidArgs;
  std::string v;
  if (std::ifstream f("/sys/module/amdgpu/version"); f) std::getline(f, v);
  if (v.empty())
    if (std::ifstream f("/proc/sys/kernel/osrelease"); f) std::getline(f, v);
  return put_string(out, len, v.empty() ? "N/A" : v);
}

RSMI_API int rsmi_status_string(int status, const char** out) {
  if (!out) return kInvalidArgs;
  switch (status) {
    case kSuccess: *out = "RSMI_STATUS_SUCCESS: The function has been executed successfully."; break;
    case kInvalidArgs: *out = "RSMI_STATUS_INVALID_ARGS: The provided arguments do not meet the preconditions required for the input."; break;
    case kNotSupported: *out = "RSMI_STATUS_NOT_SUPPORTED: The requested information or action is not available for the given input, on the given system"; break;
    case kPermission: *out = "RSMI_STATUS_PERMISSION: The user does not have permission to perform the requested operation"; break;
    case kInputOutOfBounds: *out = "RSMI_STATUS_INPUT_OUT_OF_BOUNDS: The provided input is out of allowable or safe range"; break;
    case kNotFound: *out = "RSMI_STATUS_NOT_FOUND: An item was searched for but not found"; break;
    case kInsufficientSize: *out = "RSMI_STATUS_INSUFFICIENT_SIZE: Not enough resources were available for the operation"; break;
    default: *out = "An unknown error occurred"; break;
  }
  return kSuccess;
}

// ---- Identity ----------------------------------------------------------------

// BDFID: domain in bits 32-63, bus 8-15, device 3-7, function 0-2 -- the
// address hipDeviceGetPCIBusId prints.
RSMI_API int rsmi_dev_pci_id_get(uint32_t d, uint64_t* bdfid) {
  if (!bdfid) return kInvalidArgs;
  DEVICE(d, s);
  unsigned domain = 0, bus = 0, device = 0, fn = 0;
  std::sscanf(s.bus_id, "%x:%x:%x.%x", &domain, &bus, &device, &fn);
  *bdfid = (uint64_t{domain} << 32) | (uint64_t{bus & 0xff} << 8) | (uint64_t{device & 0x1f} << 3) | (fn & 7);
  return kSuccess;
}

RSMI_API int rsmi_dev_id_get(uint32_t d, uint16_t* id) {
  if (!id) return kInvalidArgs;
  DEVICE(d, s);
  *id = static_cast<uint16_t>(s.pci_device_id >> 16);
  return kSuccess;
}
RSMI_API int rsmi_dev_subsystem_id_get(uint32_t d, uint16_t* id) { return rsmi_dev_id_get(d, id); }
RSMI_API int rsmi_dev_vendor_id_get(uint32_t d, uint16_t* id) {
  if (!id) return kInvalidArgs;
  DEVICE(d, s);
  *id = static_cast<uint16_t>(s.pci_device_id & 0xFFFF);
  return kSuccess;
}
RSMI_API int rsmi_dev_subsystem_vendor_id_get(uint32_t d, uint16_t* id) { return rsmi_dev_vendor_id_get(d, id); }
// The revision the card's config space reports (vgpu/regs: Instinct 00, a
// Radeon its chip's).
RSMI_API int rsmi_dev_revision_get(uint32_t d, uint16_t* rev) {
  if (!rev) return kInvalidArgs;
  DEVICE(d, s);
  vgpu::regs::ConfigSpace cs(s);
  *rev = static_cast<uint16_t>(cs.image(16)[0x08]);
  return kSuccess;
}
RSMI_API int rsmi_dev_market_name_get(uint32_t d, char* name, uint32_t len) {
  DEVICE(d, s);
  return put_string(name, len, s.name);
}
RSMI_API int rsmi_dev_name_get(uint32_t d, char* name, size_t len) {
  DEVICE(d, s);
  return put_string(name, len, s.name);
}
RSMI_API int rsmi_dev_brand_get(uint32_t d, char* name, uint32_t len) {
  DEVICE(d, s);
  return put_string(name, len, s.name);
}
RSMI_API int rsmi_dev_vendor_name_get(uint32_t d, char* name, size_t len) {
  DEVICE(d, s);
  return put_string(name, len, "Advanced Micro Devices, Inc. [AMD/ATI]");
}
// No profile records whose memory a card carries.
RSMI_API int rsmi_dev_vram_vendor_get(uint32_t d, char* brand, uint32_t len) {
  DEVICE(d, s);
  return put_string(brand, len, "unknown");
}
RSMI_API int rsmi_dev_serial_number_get(uint32_t d, char*, uint32_t) {
  DEVICE(d, s);
  return kNotSupported;
}
RSMI_API int rsmi_dev_vbios_version_get(uint32_t d, char*, uint32_t) {
  DEVICE(d, s);
  return kNotSupported;
}
// The compute microengine's (MEC) is the one firmware anything here asks for: RCCL 2.27 reads it to
// know whether a card needs HSA_NO_SCRATCH_RECLAIM, and refuses to start without the variable when it
// is missing. An MI300-class card reports the version RCCL accepts (amd_chip.hpp); the other blocks,
// and the other chips, are not modelled.
RSMI_API int rsmi_dev_firmware_version_get(uint32_t d, int block, uint64_t* fw_version) {
  if (!fw_version) return kInvalidArgs;
  DEVICE(d, s);
  const uint32_t mec = vgpu::amd::chip(s.architecture).mec_fw;
  constexpr int kBlockMec = 5;   // RSMI_FW_BLOCK_MEC
  if (block != kBlockMec || mec == 0) return kNotSupported;
  *fw_version = mec;
  return kSuccess;
}
// Stable per device: from the UUID NVML and amd-smi report, as `vgpu smi`'s.
RSMI_API int rsmi_dev_unique_id_get(uint32_t d, uint64_t* id) {
  if (!id) return kInvalidArgs;
  DEVICE(d, s);
  *id = fnv(s.uuid);
  return kSuccess;
}
// KFD's GPU id: a hash, stable per device, as amd-smi list's kfd_id.
RSMI_API int rsmi_dev_guid_get(uint32_t d, uint64_t* guid) {
  if (!guid) return kInvalidArgs;
  DEVICE(d, s);
  std::string uuid = s.uuid;
  if (uuid.rfind("GPU-", 0) == 0) uuid = uuid.substr(4);
  *guid = fnv(uuid.c_str()) % 65536;
  return kSuccess;
}
// KFD's node: the CPU is node 0, so the GPUs start at 1.
RSMI_API int rsmi_dev_node_id_get(uint32_t d, uint32_t* node) {
  if (!node) return kInvalidArgs;
  DEVICE(d, s);
  *node = d + 1;
  return kSuccess;
}
RSMI_API int rsmi_dev_partition_id_get(uint32_t d, uint32_t* id) {
  if (!id) return kInvalidArgs;
  DEVICE(d, s);
  *id = 0;
  return kSuccess;
}
// The gfx target, its digits as hex (gfx942 is 0x942, gfx1100 0x1100), which
// is how rocm-smi prints it back.
RSMI_API int rsmi_dev_target_graphics_version_get(uint32_t d, uint64_t* v) {
  if (!v) return kInvalidArgs;
  DEVICE(d, s);
  *v = std::strtoull(vgpu::amd::chip(s.architecture).gfx + 3, nullptr, 16);
  return kSuccess;
}

// ---- Sensors -------------------------------------------------------------

// Temperatures in millidegrees. An Instinct card has no edge sensor; a
// Radeon's edge and junction are the one die temperature the model keeps.
// Memory is there where the profile says real cards report it.
RSMI_API int rsmi_dev_temp_metric_get(uint32_t d, uint32_t sensor, int metric, int64_t* t) {
  if (!t) return kInvalidArgs;
  DEVICE(d, s);
  uint32_t c = 0;
  if (sensor == kTempJunction) c = s.temperature_c;
  else if (sensor == kTempEdge && !instinct(d)) c = s.temperature_c;
  else if (sensor == kTempMemory && s.has_memory_temperature) c = s.temperature_mem_c ? s.temperature_mem_c : s.temperature_c;
  else return kNotSupported;
  if (metric == kTempCurrent) *t = int64_t{c} * 1000;
  else if ((metric == kTempMax || metric == kTempCritical) && sensor == kTempJunction) *t = int64_t{s.temperature_max_c} * 1000;
  else return kNotSupported;
  return kSuccess;
}

// Socket power now, in microwatts.
RSMI_API int rsmi_dev_power_get(uint32_t d, uint64_t* power, int* type) {
  if (!power) return kInvalidArgs;
  DEVICE(d, s);
  *power = uint64_t{s.power_mw} * 1000;
  if (type) *type = kPowerCurrent;
  return kSuccess;
}
RSMI_API int rsmi_dev_current_socket_power_get(uint32_t d, uint64_t* power) { return rsmi_dev_power_get(d, power, nullptr); }
RSMI_API int rsmi_dev_power_ave_get(uint32_t d, uint32_t, uint64_t* power) { return rsmi_dev_power_get(d, power, nullptr); }
RSMI_API int rsmi_dev_power_cap_get(uint32_t d, uint32_t, uint64_t* cap) {
  if (!cap) return kInvalidArgs;
  DEVICE(d, s);
  *cap = uint64_t{s.power_limit_mw} * 1000;
  return kSuccess;
}
RSMI_API int rsmi_dev_power_cap_default_get(uint32_t d, uint64_t* cap) { return rsmi_dev_power_cap_get(d, 0, cap); }
RSMI_API int rsmi_dev_power_cap_range_get(uint32_t d, uint32_t, uint64_t* max, uint64_t* min) {
  if (!max || !min) return kInvalidArgs;
  DEVICE(d, s);
  *max = uint64_t{s.power_limit_mw} * 1000;
  *min = 0;
  return kSuccess;
}
// The energy accumulator: `power` is a count of `counter_resolution` micro
// joules (15.3, as the library's kEnergyCounterResolution), so the energy is
// their product in uJ; `timestamp` is the metrics table's system_clock_counter
// (ns). Only Instinct GPUs have one (a Radeon's driver has no energy file), and
// the value is amdsmi_get_energy_count's: one accumulator serves both libraries.
RSMI_API int rsmi_dev_energy_count_get(uint32_t d, uint64_t* power, float* counter_resolution, uint64_t* timestamp) {
  if (!power || !timestamp) return kInvalidArgs;
  DEVICE(d, s);
  if (s.architecture[0] != 'c') return kNotSupported;
  const vgpu::amd::EnergyReading e = vgpu::amd::energy_counter(s);
  *power = e.ticks;
  *timestamp = e.timestamp_ns;
  if (counter_resolution) *counter_resolution = vgpu::amd::kEnergyTickUj;
  return kSuccess;
}

RSMI_API int rsmi_dev_busy_percent_get(uint32_t d, uint32_t* pct) {
  if (!pct) return kInvalidArgs;
  DEVICE(d, s);
  *pct = s.utilization_gpu;
  return kSuccess;
}
RSMI_API int rsmi_dev_memory_busy_percent_get(uint32_t d, uint32_t* pct) {
  if (!pct) return kInvalidArgs;
  DEVICE(d, s);
  *pct = s.utilization_mem;
  return kSuccess;
}

// VRAM, and the CPU-visible part of it (all of it, with a resizable BAR).
// GTT, system memory the GPU maps, is not in any profile.
RSMI_API int rsmi_dev_memory_total_get(uint32_t d, int type, uint64_t* total) {
  if (!total) return kInvalidArgs;
  DEVICE(d, s);
  if (type != kMemVram && type != kMemVisVram) return kNotSupported;
  *total = s.vram_total_bytes;
  return kSuccess;
}
RSMI_API int rsmi_dev_memory_usage_get(uint32_t d, int type, uint64_t* used) {
  if (!used) return kInvalidArgs;
  DEVICE(d, s);
  if (type != kMemVram && type != kMemVisVram) return kNotSupported;
  *used = s.vram_used_bytes;
  return kSuccess;
}

// The system (sclk) and memory (mclk) clocks; the others (fabric, SoC,
// display) are not modelled.
RSMI_API int rsmi_dev_gpu_clk_freq_get(uint32_t d, int type, Frequencies* f) {
  if (!f) return kInvalidArgs;
  DEVICE(d, s);
  if (type == kClkSys) two_levels(f, s.sm_clock_max_mhz / 6, s.sm_clock_max_mhz, s.sm_clock_mhz);
  else if (type == kClkMem) two_levels(f, s.mem_clock_max_mhz / 5, s.mem_clock_max_mhz, s.mem_clock_mhz);
  else return kNotSupported;
  return kSuccess;
}

RSMI_API int rsmi_dev_perf_level_get(uint32_t d, int* level) {
  if (!level) return kInvalidArgs;
  DEVICE(d, s);
  *level = 0;   // RSMI_DEV_PERF_LEVEL_AUTO
  return kSuccess;
}
RSMI_API int rsmi_dev_overdrive_level_get(uint32_t d, uint32_t* od) {
  if (!od) return kInvalidArgs;
  DEVICE(d, s);
  *od = 0;
  return kSuccess;
}
RSMI_API int rsmi_dev_mem_overdrive_level_get(uint32_t d, uint32_t* od) { return rsmi_dev_overdrive_level_get(d, od); }

// Instinct cards are passively cooled: no fan. A Radeon's speed is on
// amdgpu's 0-255 scale.
RSMI_API int rsmi_dev_fan_speed_get(uint32_t d, uint32_t, int64_t* speed) {
  if (!speed) return kInvalidArgs;
  DEVICE(d, s);
  if (instinct(d)) return kNotSupported;
  *speed = int64_t{s.fan_percent} * 255 / 100;
  return kSuccess;
}
RSMI_API int rsmi_dev_fan_speed_max_get(uint32_t d, uint32_t, uint64_t* max) {
  if (!max) return kInvalidArgs;
  DEVICE(d, s);
  if (instinct(d)) return kNotSupported;
  *max = 255;
  return kSuccess;
}
RSMI_API int rsmi_dev_fan_rpms_get(uint32_t d, uint32_t, int64_t*) {
  DEVICE(d, s);
  return kNotSupported;
}

// The graphics core's voltage, in millivolts.
RSMI_API int rsmi_dev_volt_metric_get(uint32_t d, int type, int metric, int64_t* mv) {
  if (!mv) return kInvalidArgs;
  DEVICE(d, s);
  if (type != kVoltVddgfx || metric != kVoltCurrent) return kNotSupported;
  *mv = s.voltage_mv;
  return kSuccess;
}

// The PCIe link: each generation up to the card's, in transfers a second,
// the one it trained at marked, and its width.
RSMI_API int rsmi_dev_pci_bandwidth_get(uint32_t d, PcieBandwidth* bw) {
  if (!bw) return kInvalidArgs;
  DEVICE(d, s);
  static const uint64_t kRate[] = {2500000000ull, 5000000000ull, 8000000000ull, 16000000000ull, 32000000000ull,
                                   64000000000ull};
  std::memset(bw, 0, sizeof *bw);
  const uint32_t gens = std::min<uint32_t>(std::max<uint32_t>(s.pcie_gen_max, 1), 6);
  bw->transfer_rate.num_supported = gens;
  for (uint32_t g = 0; g < gens; ++g) {
    bw->transfer_rate.frequency[g] = kRate[g];
    bw->lanes[g] = s.pcie_width;
  }
  bw->transfer_rate.current = std::min<uint32_t>(std::max<uint32_t>(s.pcie_gen, 1), gens) - 1;
  return kSuccess;
}
RSMI_API int rsmi_dev_pci_throughput_get(uint32_t d, uint64_t*, uint64_t*, uint64_t*) {
  DEVICE(d, s);
  return kNotSupported;
}
RSMI_API int rsmi_dev_pci_replay_counter_get(uint32_t d, uint64_t* counter) {
  if (!counter) return kInvalidArgs;
  DEVICE(d, s);
  vgpu::ras::Counters c{};
  try {
    c = vgpu::ras::read(s.uuid).since_load;
  } catch (const std::exception&) {
  }
  *counter = c.pcie[0];
  return kSuccess;
}

// ---- Reliability -----------------------------------------------------------

// ECC by RAS block, from the same counts `vgpu smi` reports: device memory is
// the UMC (memory controller) block, the on-chip memories GFX.
RSMI_API int rsmi_dev_ecc_enabled_get(uint32_t d, uint64_t* mask) {
  if (!mask) return kInvalidArgs;
  DEVICE(d, s);
  *mask = s.ecc_enabled ? kBlocks : 0;
  return kSuccess;
}
RSMI_API int rsmi_dev_ecc_status_get(uint32_t d, uint64_t block, int* state) {
  if (!state) return kInvalidArgs;
  DEVICE(d, s);
  if (!(block & kBlocks)) return kNotSupported;
  *state = s.ecc_enabled ? kRasEnabled : kRasDisabled;
  return kSuccess;
}
RSMI_API int rsmi_dev_ecc_count_get(uint32_t d, uint64_t block, ErrorCount* ec) {
  if (!ec) return kInvalidArgs;
  DEVICE(d, s);
  if (!s.ecc_enabled || !(block & kBlocks)) return kNotSupported;
  vgpu::ras::Counters c{};
  try {
    c = vgpu::ras::read(s.uuid).since_load;
  } catch (const std::exception&) {
  }
  using vgpu::ras::Location;
  using vgpu::ras::Severity;
  const auto memory = [&](Severity v) {
    return c.ecc[static_cast<uint32_t>(v)][static_cast<uint32_t>(Location::DeviceMemory)];
  };
  const auto count = [&](Severity v) -> uint64_t {
    if (block == kBlockUmc) return memory(v);
    if (block == kBlockGfx) return c.ecc_total(v) - memory(v);
    return 0;
  };
  ec->correctable_err = count(Severity::Corrected);
  ec->uncorrectable_err = count(Severity::Uncorrected);
  return kSuccess;
}
// The pages retired for uncorrectable errors, as the RAS model counts them.
// Their addresses are not kept, so each is numbered by the page it is.
RSMI_API int rsmi_dev_memory_reserved_pages_get(uint32_t d, uint32_t* num, RetiredPage* records) {
  if (!num) return kInvalidArgs;
  DEVICE(d, s);
  // A card without ECC (the Radeon profiles) has no record of bad pages.
  if (!s.ecc_enabled) return kNotSupported;
  vgpu::ras::Counters c{};
  try {
    c = vgpu::ras::read(s.uuid).lifetime;
  } catch (const std::exception&) {
  }
  // As amdsmi_get_gpu_bad_page_info (amd_smi.cpp) lists them: a remapped row
  // counts as a page, reserved (status 0) but for the newest while a
  // retirement is pending (status 1).
  const uint32_t pages = static_cast<uint32_t>(c.retired_sbe + c.retired_dbe + c.rows_correctable + c.rows_uncorrectable);
  const uint32_t reserved = pages - ((c.retired_pending || c.rows_pending) && pages ? 1 : 0);
  if (records)
    for (uint32_t k = 0; k < std::min(*num, pages); ++k) records[k] = {uint64_t{k} << 12, 4096, k < reserved ? 0 : 1};
  const bool fits = !records || *num >= pages;
  *num = records ? std::min(*num, pages) : pages;
  return fits ? kSuccess : kInsufficientSize;
}
RSMI_API int rsmi_dev_xgmi_error_status(uint32_t d, int* status) {
  if (!status) return kInvalidArgs;
  DEVICE(d, s);
  if (!instinct(d)) return kNotSupported;
  *status = kXgmiNoErrors;
  return kSuccess;
}

// ---- Partitions ------------------------------------------------------------

RSMI_API int rsmi_dev_compute_partition_get(uint32_t d, char* out, uint32_t len) {
  DEVICE(d, s);
  if (!partitionable(s)) return kNotSupported;
  static const char* const kCompute[] = {"SPX", "DPX", "TPX", "QPX", "CPX"};
  const uint32_t px = (nbio(s, "nbio_partition_compute_status") >> 4) & 0xF;
  return put_string(out, len, px < 5 ? kCompute[px] : "UNKNOWN");
}
RSMI_API int rsmi_dev_memory_partition_get(uint32_t d, char* out, uint32_t len) {
  DEVICE(d, s);
  if (!partitionable(s)) return kNotSupported;
  const int nps = __builtin_ffs(static_cast<int>((nbio(s, "nbio_partition_mem_status") >> 4) & 0xFF));
  return put_string(out, len, nps ? "NPS" + std::to_string(nps) : "UNKNOWN");
}
RSMI_API int rsmi_dev_memory_partition_capabilities_get(uint32_t d, char* out, uint32_t len) {
  DEVICE(d, s);
  if (!partitionable(s)) return kNotSupported;
  std::string caps, sep;
  for (uint32_t cap = nbio(s, "nbio_partition_mem_cap"); cap; cap &= cap - 1) {
    caps += sep + "NPS" + std::to_string(__builtin_ffs(static_cast<int>(cap)));
    sep = ",";
  }
  return put_string(out, len, caps);
}

// ---- Topology ----------------------------------------------------------------

// Instinct GPUs are one Infinity Fabric hop apart, weight 15, as the AMD
// kernel driver gives such a link. Radeon GPUs reach each other over PCIe
// through the host's root complex: two hops, weight 40 (20 each way to the
// CPU) -- what rocm-smi --showtopo and amd-smi topology print.
RSMI_API int rsmi_topo_get_link_type(uint32_t src, uint32_t dst, uint64_t* hops, int* type) {
  if (!hops || !type) return kInvalidArgs;
  if (!valid(src) || !valid(dst)) return kInputOutOfBounds;
  const bool xgmi = instinct(src) && instinct(dst);
  *hops = src == dst ? 0 : xgmi ? 1 : 2;
  *type = xgmi ? kLinkXgmi : kLinkPcie;
  return kSuccess;
}
RSMI_API int rsmi_topo_get_link_weight(uint32_t src, uint32_t dst, uint64_t* weight) {
  if (!weight) return kInvalidArgs;
  if (!valid(src) || !valid(dst)) return kInputOutOfBounds;
  *weight = src == dst ? 0 : instinct(src) && instinct(dst) ? 15 : 40;
  return kSuccess;
}

// An XGMI link's bandwidth, in MB/s: MI300X's datasheet gives each of its
// Infinity Fabric links 64 GB/s in each direction. Placeholder, like the
// profiles' numbers; the simulator's copies take no modelled time. ROCm SMI
// answers this only for XGMI.
RSMI_API int rsmi_minmax_bandwidth_get(uint32_t src, uint32_t dst, uint64_t* min_bw, uint64_t* max_bw) {
  if (!min_bw || !max_bw) return kInvalidArgs;
  if (!valid(src) || !valid(dst) || src == dst) return kInputOutOfBounds;
  if (!instinct(src) || !instinct(dst)) return kNotSupported;
  *min_bw = *max_bw = 64000;
  return kSuccess;
}
RSMI_API int rsmi_is_P2P_accessible(uint32_t src, uint32_t dst, bool* ok) {
  if (!ok) return kInvalidArgs;
  if (!valid(src) || !valid(dst)) return kInputOutOfBounds;
  *ok = true;
  return kSuccess;
}
RSMI_API int rsmi_topo_get_numa_node_number(uint32_t d, uint32_t* node) {
  if (!node) return kInvalidArgs;
  if (!valid(d)) return kInputOutOfBounds;
  *node = 0;
  return kSuccess;
}
RSMI_API int rsmi_topo_numa_affinity_get(uint32_t d, int32_t* node) {
  if (!node) return kInvalidArgs;
  if (!valid(d)) return kInputOutOfBounds;
  *node = 0;
  return kSuccess;
}

// ---- Processes -------------------------------------------------------------

namespace {
// Every process using an AMD GPU: its id, the GPUs it uses, and its VRAM.
struct Proc {
  uint64_t vram = 0;
  std::vector<uint32_t> gpus;
};
std::map<uint32_t, Proc> processes() {
  std::map<uint32_t, Proc> out;
  vgpu::telemetry::Shared snap{};
  if (count() == 0 || !vgpu::read_machine(&snap)) return out;
  vgpu::drop_lost_amd(&snap);
  uint32_t k = 0;
  for (uint32_t i = 0; i < snap.device_count; ++i) {
    const Sample& d = snap.devices[i];
    if (std::strcmp(d.vendor, "amd") != 0) continue;
    for (uint32_t p = 0; p < d.proc_count && p < vgpu::telemetry::kMaxProcs; ++p) {
      Proc& e = out[d.procs[p].pid];
      e.vram += d.procs[p].used_bytes;
      e.gpus.push_back(k);
    }
    ++k;
  }
  return out;
}
ProcessInfo info(uint32_t pid, const Proc& p) { return {pid, 0, p.vram, 0, kCuOccupancyInvalid}; }
}  // namespace

RSMI_API int rsmi_compute_process_info_get(ProcessInfo* procs, uint32_t* num) {
  if (!num) return kInvalidArgs;
  const auto all = processes();
  uint32_t k = 0;
  if (procs)
    for (const auto& [pid, p] : all) {
      if (k >= *num) break;
      procs[k++] = info(pid, p);
    }
  const bool fits = !procs || *num >= all.size();
  *num = procs ? k : static_cast<uint32_t>(all.size());
  return fits ? kSuccess : kInsufficientSize;
}
RSMI_API int rsmi_compute_process_info_by_pid_get(uint32_t pid, ProcessInfo* proc) {
  if (!proc) return kInvalidArgs;
  const auto all = processes();
  const auto it = all.find(pid);
  if (it == all.end()) return kNotFound;
  *proc = info(pid, it->second);
  return kSuccess;
}
RSMI_API int rsmi_compute_process_info_by_device_get(uint32_t pid, uint32_t d, ProcessInfo* proc) {
  if (!proc) return kInvalidArgs;
  if (!valid(d)) return kInputOutOfBounds;
  const auto all = processes();
  const auto it = all.find(pid);
  if (it == all.end() || std::find(it->second.gpus.begin(), it->second.gpus.end(), d) == it->second.gpus.end())
    return kNotFound;
  *proc = info(pid, it->second);
  return kSuccess;
}
RSMI_API int rsmi_compute_process_gpus_get(uint32_t pid, uint32_t* indices, uint32_t* num) {
  if (!num) return kInvalidArgs;
  const auto all = processes();
  const auto it = all.find(pid);
  if (it == all.end()) return kNotFound;
  const auto& g = it->second.gpus;
  if (indices)
    for (uint32_t k = 0; k < std::min<size_t>(*num, g.size()); ++k) indices[k] = g[k];
  const bool fits = !indices || *num >= g.size();
  *num = indices ? std::min<uint32_t>(*num, static_cast<uint32_t>(g.size())) : static_cast<uint32_t>(g.size());
  return fits ? kSuccess : kInsufficientSize;
}

// ---- The metrics table and the identity queries the rest of the header has ----

namespace {

struct MetricsHeader {  // metrics_table_header_t
  uint16_t structure_size;
  uint8_t format_revision, content_revision;
};
struct XcpMetrics {  // amdgpu_xcp_metrics_t
  uint32_t gfx_busy_inst[8];
  uint16_t jpeg_busy[40];
  uint16_t vcn_busy[4];
  uint64_t gfx_busy_acc[8], gfx_below_host_limit_acc[8], gfx_below_host_limit_ppt_acc[8],
      gfx_below_host_limit_thm_acc[8], gfx_low_utilization_acc[8], gfx_below_host_limit_total_acc[8];
};
// rsmi_gpu_metrics_t, ROCm 7.2.0's (content revision 1.7): the members of
// amdsmi_gpu_metrics_t, in the same order.
struct GpuMetrics {
  MetricsHeader common_header;
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
// Sizes and offsets as compiled from rocm_smi.h of ROCm 7.2.0 (amdsmi 26.2).
static_assert(sizeof(MetricsHeader) == 4 && sizeof(XcpMetrics) == 504 && sizeof(GpuMetrics) == 4544);
static_assert(offsetof(GpuMetrics, temperature_hotspot) == 6 && offsetof(GpuMetrics, energy_accumulator) == 24 &&
              offsetof(GpuMetrics, current_gfxclk) == 54 && offsetof(GpuMetrics, throttle_status) == 68 &&
              offsetof(GpuMetrics, temperature_hbm) == 88 && offsetof(GpuMetrics, firmware_timestamp) == 96 &&
              offsetof(GpuMetrics, indep_throttle_status) == 112 && offsetof(GpuMetrics, current_socket_power) == 120 &&
              offsetof(GpuMetrics, gfxclk_lock_status) == 132 && offsetof(GpuMetrics, pcie_bandwidth_acc) == 144 &&
              offsetof(GpuMetrics, xgmi_read_data_acc) == 184 && offsetof(GpuMetrics, current_gfxclks) == 312 &&
              offsetof(GpuMetrics, jpeg_activity) == 352 && offsetof(GpuMetrics, pcie_nak_sent_count_acc) == 416 &&
              offsetof(GpuMetrics, accumulation_counter) == 424 && offsetof(GpuMetrics, num_partition) == 472 &&
              offsetof(GpuMetrics, xcp_stats) == 480 && offsetof(GpuMetrics, pcie_lc_perf_other_end_recovery) == 4512 &&
              offsetof(GpuMetrics, vram_max_bandwidth) == 4520 && offsetof(GpuMetrics, xgmi_link_status) == 4528);

// The driver's table for a device, as it publishes it in sysfs (v1.5).
bool metrics_table(const Sample& s, vgpu::amd::MetricsV15* t) {
  vgpu::ras::Counters c{};
  try {
    c = vgpu::ras::read(s.uuid).since_load;
  } catch (const std::exception&) {
  }
  const std::string bytes = vgpu::amd::gpu_metrics(s, c);
  if (bytes.size() != sizeof *t) return false;
  std::memcpy(t, bytes.data(), sizeof *t);
  return true;
}

}  // namespace

// The table in ROCm SMI's public structure: what the v1.5 table has, copied
// across (the same copy amdsmi_get_gpu_metrics_info makes), every other member
// all ones, N/A. The energy accumulator of an Instinct GPU is the shared
// counter rsmi_dev_energy_count_get counts.
RSMI_API int rsmi_dev_gpu_metrics_info_get(uint32_t d, GpuMetrics* out) {
  if (!valid(d)) return kInputOutOfBounds;
  if (!out) return kInvalidArgs;
  Sample s{};
  vgpu::amd::MetricsV15 t;
  if (!sample(d, &s)) return kNotFound;
  if (!metrics_table(s, &t)) return kNotSupported;
  std::memset(out, 0xFF, sizeof *out);
  vgpu::amd::fill_public_metrics(out, t);
  if (s.architecture[0] == 'c' && t.energy_accumulator == ~uint64_t{0}) {
    const vgpu::amd::EnergyReading e = vgpu::amd::energy_counter(s);
    out->energy_accumulator = e.ticks;
    out->system_clock_counter = e.timestamp_ns;
  }
  return kSuccess;
}
RSMI_API int rsmi_dev_metrics_header_info_get(uint32_t d, MetricsHeader* out) {
  if (!valid(d)) return kInputOutOfBounds;
  if (!out) return kInvalidArgs;
  Sample s{};
  vgpu::amd::MetricsV15 t;
  if (!sample(d, &s)) return kNotFound;
  if (!metrics_table(s, &t)) return kNotSupported;
  *out = {t.structure_size, t.format_revision, t.content_revision};
  return kSuccess;
}

// The DRM render node's minor: the first render node is 128, and a GPU has
// the one the kernel driver's topology gives it (kfd.cpp).
RSMI_API int rsmi_dev_drm_render_minor_get(uint32_t d, uint32_t* minor) {
  if (!minor) return kInvalidArgs;
  DEVICE(d, s);
  *minor = 128 + d;
  return kSuccess;
}
// The hive the Instinct GPUs of the machine share: KFD's topology `hive_id`
// (a hash of the first GPU's UUID), as amdsmi_get_xgmi_info gives it. A
// Radeon GPU has no hive.
RSMI_API int rsmi_dev_xgmi_hive_id_get(uint32_t d, uint64_t* hive) {
  if (!hive) return kInvalidArgs;
  DEVICE(d, s);
  if (s.architecture[0] != 'c') return kNotSupported;
  *hive = fnv(machine()[0].uuid);
  return kSuccess;
}
// rsmi_pcie_slot_type_t: Instinct GPUs sit in OAM modules, Radeon cards in a
// PCIe slot (amdsmi_get_pcie_info says the same).
RSMI_API int rsmi_dev_pcie_slot_type_get(uint32_t d, int* type) {
  if (!type) return kInvalidArgs;
  DEVICE(d, s);
  *type = s.architecture[0] == 'c' ? 2 : 0;
  return kSuccess;
}

// ---- Not modelled ------------------------------------------------------------
//
// Counters and tables the simulator does not keep, and every setting: a
// simulated card has no knob to turn. Each is refused as a card without it
// refuses it, so a tool reports N/A (or that it cannot set it) and goes on.

#define NOT_MODELLED(name, ...)                          \
  RSMI_API int name(uint32_t d, __VA_ARGS__) {           \
    return valid(d) ? kNotSupported : kInputOutOfBounds; \
  }
NOT_MODELLED(rsmi_dev_metrics_xcd_counter_get, uint16_t*)
NOT_MODELLED(rsmi_dev_activity_avg_mm_get, uint16_t*)
NOT_MODELLED(rsmi_utilization_count_get, void*, uint32_t, uint64_t*)
NOT_MODELLED(rsmi_dev_od_volt_info_get, void*)
NOT_MODELLED(rsmi_dev_od_volt_curve_regions_get, uint32_t*, void*)
NOT_MODELLED(rsmi_dev_power_profile_presets_get, uint32_t, void*)
NOT_MODELLED(rsmi_dev_perf_level_set, int)
NOT_MODELLED(rsmi_dev_overdrive_level_set, uint32_t)
NOT_MODELLED(rsmi_dev_gpu_clk_freq_set, int, uint64_t)
NOT_MODELLED(rsmi_dev_clk_range_set, uint64_t, uint64_t, int)
NOT_MODELLED(rsmi_dev_clk_extremum_set, int, uint64_t, int)
NOT_MODELLED(rsmi_dev_od_clk_info_set, int, uint64_t, int)
NOT_MODELLED(rsmi_dev_od_volt_info_set, uint32_t, uint64_t, uint64_t)
NOT_MODELLED(rsmi_dev_power_cap_set, uint32_t, uint64_t)
NOT_MODELLED(rsmi_dev_power_profile_set, uint32_t, uint64_t)
NOT_MODELLED(rsmi_dev_fan_speed_set, uint32_t, uint64_t)
NOT_MODELLED(rsmi_dev_fan_reset, uint32_t)
NOT_MODELLED(rsmi_dev_pci_bandwidth_set, uint64_t)
NOT_MODELLED(rsmi_dev_compute_partition_set, int)
NOT_MODELLED(rsmi_dev_memory_partition_set, int)
NOT_MODELLED(rsmi_perf_determinism_mode_set, uint64_t)
NOT_MODELLED(rsmi_event_notification_mask_set, uint64_t)
RSMI_API int rsmi_dev_gpu_reset(uint32_t d) { return valid(d) ? kNotSupported : kInputOutOfBounds; }
RSMI_API int rsmi_dev_xgmi_error_reset(uint32_t d) { return valid(d) ? kNotSupported : kInputOutOfBounds; }
RSMI_API int rsmi_event_notification_init(uint32_t d) { return valid(d) ? kNotSupported : kInputOutOfBounds; }
RSMI_API int rsmi_event_notification_stop(uint32_t d) { return valid(d) ? kNotSupported : kInputOutOfBounds; }
RSMI_API int rsmi_event_notification_get(int, uint32_t* num, void*) {
  if (num) *num = 0;
  return kNotSupported;
}

// The rest of the header's functions, which a simulated card has nothing to
// answer: a VBIOS build, SKU, XGMI physical id and port numbers, the PM and
// register tables, cache info, performance counters, the power-throttle
// limit and the like. The library exports every one, as the package that binds
// them expects, and refuses each as a card without the feature does.
// (The device index is checked; the other arguments are not read.)
#define NOT_MODELLED_ANY(name)                           \
  RSMI_API int name(uint32_t d, ...) {                   \
    return valid(d) ? kNotSupported : kInputOutOfBounds; \
  }
NOT_MODELLED_ANY(rsmi_dev_activity_metric_get)
NOT_MODELLED_ANY(rsmi_dev_cache_info_get)
NOT_MODELLED_ANY(rsmi_dev_compute_partition_capabilities_get)
NOT_MODELLED_ANY(rsmi_dev_compute_partition_resource_profile_get)
NOT_MODELLED_ANY(rsmi_dev_compute_partition_supported_nps_configs_get)
NOT_MODELLED_ANY(rsmi_dev_compute_partition_supported_xcp_configs_get)
NOT_MODELLED_ANY(rsmi_dev_compute_partition_xcp_config_set)
NOT_MODELLED_ANY(rsmi_dev_current_compute_xcp_config_get)
NOT_MODELLED_ANY(rsmi_dev_device_identifiers_get)
NOT_MODELLED_ANY(rsmi_dev_gpu_partition_metrics_info_get)
NOT_MODELLED_ANY(rsmi_dev_gpu_run_cleaner_shader)
NOT_MODELLED_ANY(rsmi_dev_metrics_log_get)
NOT_MODELLED_ANY(rsmi_dev_npm_info_get)
NOT_MODELLED_ANY(rsmi_dev_overdrive_level_set_v1)
NOT_MODELLED_ANY(rsmi_dev_pcie_vendor_name_get)
NOT_MODELLED_ANY(rsmi_dev_perf_level_set_v1)
NOT_MODELLED_ANY(rsmi_dev_pm_metrics_info_get)
NOT_MODELLED_ANY(rsmi_dev_process_isolation_get)
NOT_MODELLED_ANY(rsmi_dev_process_isolation_set)
NOT_MODELLED_ANY(rsmi_dev_reg_table_info_get)
NOT_MODELLED_ANY(rsmi_dev_sku_get)
NOT_MODELLED_ANY(rsmi_dev_soc_pstate_get)
NOT_MODELLED_ANY(rsmi_dev_soc_pstate_set)
NOT_MODELLED_ANY(rsmi_dev_subsystem_name_get)
NOT_MODELLED_ANY(rsmi_dev_supported_power_cap_get)
NOT_MODELLED_ANY(rsmi_dev_vbios_build_number_get)
NOT_MODELLED_ANY(rsmi_dev_xgmi_physical_id_get)
NOT_MODELLED_ANY(rsmi_dev_xgmi_plpd_get)
NOT_MODELLED_ANY(rsmi_dev_xgmi_plpd_set)
NOT_MODELLED_ANY(rsmi_dev_xgmi_port_num_get)
NOT_MODELLED_ANY(rsmi_get_gpu_ptl_formats)
NOT_MODELLED_ANY(rsmi_get_gpu_ptl_state)
NOT_MODELLED_ANY(rsmi_ras_feature_info_get)
NOT_MODELLED_ANY(rsmi_read_supported_ptl_formats)
NOT_MODELLED_ANY(rsmi_set_gpu_ptl_enable_with_formats)
NOT_MODELLED_ANY(rsmi_set_gpu_ptl_formats)
NOT_MODELLED_ANY(rsmi_set_gpu_ptl_state)
NOT_MODELLED_ANY(rsmi_counter_available_counters_get)
NOT_MODELLED_ANY(rsmi_dev_counter_create)
NOT_MODELLED_ANY(rsmi_dev_counter_group_supported)
NOT_MODELLED_ANY(rsmi_dev_supported_func_iterator_open)
// These take no device index.
#define NOT_MODELLED_NO_DEVICE(name) \
  RSMI_API int name(...) { return kNotSupported; }
NOT_MODELLED_NO_DEVICE(rsmi_dev_amdgpu_driver_reload)
NOT_MODELLED_NO_DEVICE(rsmi_driver_status)
NOT_MODELLED_NO_DEVICE(rsmi_counter_control)
NOT_MODELLED_NO_DEVICE(rsmi_counter_read)
NOT_MODELLED_NO_DEVICE(rsmi_dev_counter_destroy)
NOT_MODELLED_NO_DEVICE(rsmi_dev_supported_func_iterator_close)
NOT_MODELLED_NO_DEVICE(rsmi_dev_supported_variant_iterator_open)
NOT_MODELLED_NO_DEVICE(rsmi_func_iter_next)
NOT_MODELLED_NO_DEVICE(rsmi_func_iter_value_get)
NOT_MODELLED_NO_DEVICE(rsmi_topo_get_p2p_status)
