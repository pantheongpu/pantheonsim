// libvgpunvml — VirtualGPU's implementation of NVML (NVIDIA Management Library).
//
// nvidia-smi does not link NVML: it dlopens "libnvidia-ml.so.1" and resolves
// entry points by name. That soname is searched along LD_LIBRARY_PATH first,
// so placing this library in the shim directory makes the *real* nvidia-smi
// binary report VirtualGPU's virtual devices.
//
// Data comes from the shared telemetry segment published by a running
// VirtualGPU process (see include/vgpu/telemetry.hpp). Memory and utilization
// are real simulator state; power, temperature, clocks, voltage and fan are the
// documented synthetic model. Queries VirtualGPU has no answer for return
// NVML_ERROR_NOT_SUPPORTED, which nvidia-smi renders as "N/A" — the honest
// result, rather than an invented number.
// Keep the header from aliasing base names onto _v2/_v3 entries: we export
// both spellings ourselves, because callers may look up either.
#define NVML_NO_UNVERSIONED_FUNC_DEFS 1
#include <nvml.h>

#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <thread>

#include "vgpu/host_cpus.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/ras.hpp"
#include "vgpu/regs.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/telemetry.hpp"

namespace {

// Recursive, because entry points forward to one another (the unversioned
// names call their _v2/_v3 forms) and every one of them holds it.
std::recursive_mutex g_mu;
// NVML is reference counted: every nvmlInit needs its own nvmlShutdown, and
// with the count at zero every query answers NVML_ERROR_UNINITIALIZED. This
// was a flag that nothing read, so a tool that shut NVML down and kept calling
// was still answered -- the bug a tool's own tests exist to catch.
int g_init_count = 0;
// Takes the lock for the rest of the entry point and then checks the count.
// The check used to happen before the lock, so a query racing the last
// nvmlShutdown on another thread could pass it and read state being torn down.
#define REQUIRE_INIT()                                        \
  std::lock_guard<std::recursive_mutex> init_lock_(g_mu);     \
  if (g_init_count == 0) return NVML_ERROR_UNINITIALIZED
vgpu::telemetry::Shared g_snap{};
// Set when the machine is described by VGPU_GPU / VGPU_DEVICE_COUNT rather
// than by a live publisher; see nvmlInit_v2.
bool g_have_profile = false;
vgpu::telemetry::Shared g_idle{};
// GPUs that have fallen off the bus (`vgpu fault lose`), as of the last refresh.
bool g_lost[vgpu::telemetry::kMaxDevices] = {};

// Handles are 1-based indices encoded as pointers, so they are never null.
nvmlDevice_t handle_for(unsigned int index) {
  return reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(index) + 1);
}
bool index_of(nvmlDevice_t dev, unsigned int* out) {
  uintptr_t v = reinterpret_cast<uintptr_t>(dev);
  if (v == 0) return false;
  unsigned int idx = static_cast<unsigned int>(v - 1);
  if (idx >= g_snap.device_count || g_lost[idx]) return false;
  *out = idx;
  return true;
}

// What a query about a device it could not answer returns: the handle is a
// GPU that has fallen off the bus, or not a device at all.
nvmlReturn_t bad(nvmlDevice_t dev) {
  const uintptr_t v = reinterpret_cast<uintptr_t>(dev);
  return v && v - 1 < g_snap.device_count && g_lost[v - 1] ? NVML_ERROR_GPU_IS_LOST
                                                          : NVML_ERROR_INVALID_ARGUMENT;
}

// Re-reads telemetry so repeated queries (nvidia-smi -l) show live values.
// With nothing publishing, a machine described by the environment answers as
// an idle rack -- see nvmlInit_v2.
bool refresh() {
  if (!vgpu::telemetry::read_snapshot(&g_snap)) {
    if (!g_have_profile) return false;
    g_snap = g_idle;
  }
  // NVML is NVIDIA's driver's: it sees NVIDIA GPUs only. A machine whose GPUs
  // are all AMD's has no NVIDIA driver, and NVML answers there as it does on
  // one (DRIVER_NOT_LOADED, from nvmlInit). It counted every simulated GPU,
  // so a simulated MI300X was an NVIDIA GPU to pynvml too, and vLLM, which
  // asks both vendors' libraries, found a CUDA and a ROCm machine at once.
  uint32_t kept = 0;
  for (uint32_t i = 0; i < g_snap.device_count; ++i)
    if (std::strcmp(g_snap.devices[i].vendor, "amd") != 0) g_snap.devices[kept++] = g_snap.devices[i];
  const bool had = g_snap.device_count > 0;
  g_snap.device_count = kept;
  if (had && kept == 0) return false;
  // Injected clock-event reasons and the readings they imply, applied once
  // here so every getter agrees with every other, and with nvidia-smi.
  for (uint32_t i = 0; i < g_snap.device_count; ++i) {
    try {
      vgpu::ras::apply_throttle(g_snap.devices[i]);
      vgpu::ras::apply_link(g_snap.devices[i]);
      g_lost[i] = vgpu::ras::is_lost(g_snap.devices[i].uuid);
    } catch (const std::exception&) {
      g_lost[i] = false;
    }
  }
  return true;
}

// The machine VGPU_GPU / VGPU_DEVICE_COUNT describe, as an idle snapshot.
// `vgpu run` and the GitHub Action set both for the program they start, and a
// program that asks NVML before (or without) touching CUDA -- pynvml, gpustat,
// a framework probing for GPUs -- runs before any simulated process has
// published telemetry. Answering DRIVER_NOT_LOADED there told it the machine
// had no GPU at all, which is the one answer that is certainly wrong: the
// profile says what the devices are, and `vgpu smi` already answers from it.
// Memory reads as unused because nothing is using it.
bool load_described_machine() {
  if (g_have_profile) return true;
  const char* gpu = std::getenv("VGPU_GPU");
  const char* count = std::getenv("VGPU_DEVICE_COUNT");
  const bool described = (gpu && gpu[0]) || (count && count[0]);
  if (!described) return false;
  try {
    vgpu::DeviceProfile p = vgpu::load_gpu(gpu && gpu[0] ? gpu : "nvidia/h100");
    vgpu::apply_vram_override(p);  // the card the program's CUDA calls will see
    g_idle = vgpu::telemetry::idle_snapshot(p, count && count[0] ? std::atoi(count) : 1);
    g_have_profile = true;
  } catch (const std::exception&) {
    return false;  // an unknown profile id: there is no machine to describe
  }
  return true;
}

const vgpu::telemetry::DeviceSample* sample(nvmlDevice_t dev) {
  unsigned int idx;
  if (!index_of(dev, &idx)) return nullptr;
  return &g_snap.devices[idx];
}

// NVML documents NVML_ERROR_INSUFFICIENT_SIZE for a buffer too small for the
// answer. This used to truncate into it and report success, so a tool that
// sized a UUID buffer wrong got a prefix it believed was the whole UUID -- and
// a lookup by that UUID then failed somewhere far from the cause.
nvmlReturn_t copy_string(const char* src, char* dst, unsigned int len) {
  if (!dst) return NVML_ERROR_INVALID_ARGUMENT;
  const size_t need = std::strlen(src) + 1;
  if (need > len) return NVML_ERROR_INSUFFICIENT_SIZE;
  std::memcpy(dst, src, need);
  return NVML_SUCCESS;
}


// Microseconds the device has spent in any of the clock-event reasons in
// `mask`, from the windows `vgpu fault throttle` opens.
uint64_t reason_time_us(const std::string& uuid, uint64_t mask) {
  uint64_t us = 0;
  try {
    for (uint64_t bit = 1; bit && bit <= mask; bit <<= 1)
      if (mask & bit) us += vgpu::ras::throttle_time_us(uuid, bit);
  } catch (const std::exception&) {
  }
  return us;
}

// The clock-event reasons a performance policy counts, by the NVML field id
// that reports it (NVML_FI_DEV_PERF_POLICY_* and the CLOCKS_EVENT_REASON_*
// counters). Sync boost, low utilization, reliability and the application and
// base clock totals are reasons VirtualGPU never raises, so they read zero.
uint64_t policy_reasons(unsigned field) {
  switch (field) {
    case 74: return vgpu::ras::kSwPowerCap;
    case 75: return vgpu::ras::kSwThermalSlowdown | vgpu::ras::kHwThermalSlowdown;
    case 77: return vgpu::ras::kHwPowerBrakeSlowdown | vgpu::ras::kHwSlowdown;
    case 269: return vgpu::ras::kSwThermalSlowdown;
    case 270: return vgpu::ras::kHwThermalSlowdown;
    case 271: return vgpu::ras::kHwPowerBrakeSlowdown;
    default: return 0;
  }
}

// The energy counter, in millijoules: the model's power integrated over the
// time between readings. NVML's counts from driver load, which here is the
// process's first reading, so the figure is monotonic within a process (DCGM's
// hostengine, a monitoring loop) and starts again from zero in the next.
unsigned long long energy_mj(unsigned int idx, const vgpu::telemetry::DeviceSample& d) {
  static double mj[vgpu::telemetry::kMaxDevices] = {};
  static std::chrono::steady_clock::time_point last[vgpu::telemetry::kMaxDevices] = {};
  const auto now = std::chrono::steady_clock::now();
  if (last[idx].time_since_epoch().count() != 0)
    mj[idx] += d.power_mw * std::chrono::duration<double>(now - last[idx]).count();
  last[idx] = now;
  return static_cast<unsigned long long>(mj[idx]);
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- lifecycle ---- */

VGPU_EXPORT nvmlReturn_t nvmlInit_v2(void) {
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  if (!refresh() && !(load_described_machine() && refresh())) {
    // No live VirtualGPU and no machine described: report "driver not
    // loaded", exactly as NVML does on a machine with no NVIDIA driver.
    return NVML_ERROR_DRIVER_NOT_LOADED;
  }
  ++g_init_count;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlInit(void) { return nvmlInit_v2(); }
VGPU_EXPORT nvmlReturn_t nvmlInitWithFlags(unsigned int) { return nvmlInit_v2(); }
VGPU_EXPORT nvmlReturn_t nvmlShutdown(void) {
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  if (g_init_count == 0) return NVML_ERROR_UNINITIALIZED;
  --g_init_count;
  return NVML_SUCCESS;
}

// Every code nvml.h declares. Numbers rather than enumerators, because the
// shim also builds against CUDA 12.0's nvml.h, which predates the last few.
// Previously eight codes had strings and the rest -- NO_PERMISSION, GPU_IS_LOST,
// RESET_REQUIRED, the ones a monitoring tool most needs to explain -- all read
// "Unknown Error".
VGPU_EXPORT const char* nvmlErrorString(nvmlReturn_t result) {
  switch (static_cast<int>(result)) {
    case 0: return "The operation was successful";
    case 1: return "Uninitialized";
    case 2: return "Invalid Argument";
    case 3: return "Not Supported";
    case 4: return "Insufficient Permissions";
    case 5: return "Already Initialized";
    case 6: return "Not Found";
    case 7: return "Insufficient Size";
    case 8: return "Insufficient External Power";
    case 9: return "Driver Not Loaded";
    case 10: return "Timeout";
    case 11: return "Interrupt Request Issue";
    case 12: return "NVML Shared Library Not Found";
    case 13: return "Function Not Found";
    case 14: return "Corrupted infoROM";
    case 15: return "GPU is lost";
    case 16: return "GPU requires restart";
    case 17: return "The operating system has blocked the request";
    case 18: return "RM has detected an NVML/RM version mismatch";
    case 19: return "In use by another client";
    case 20: return "Insufficient Memory";
    case 21: return "No data";
    case 22: return "The requested vgpu operation is not available on target device, because ECC is enabled";
    case 23: return "Insufficient Resources";
    case 24: return "Frequency not supported";
    case 25: return "Argument version mismatch";
    case 26: return "Deprecated";
    case 27: return "Not Ready";
    case 28: return "GPU not found";
    case 29: return "Invalid state";
    case 30: return "Reset type not supported";
    default: return "Unknown Error";
  }
}

/* ---- system ---- */

VGPU_EXPORT nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  return copy_string(g_snap.driver_version, version, length);
}
VGPU_EXPORT nvmlReturn_t nvmlSystemGetNVMLVersion(char* version, unsigned int length) { REQUIRE_INIT();
  // nvidia-smi validates this against the version it was built with and exits
  // if it disagrees, so it is overridable: set VGPU_NVML_VERSION to the string
  // `nvidia-smi --version` reports (as "NVML version", e.g. 595.71 ->
  // "13.595.71.01"). `vgpu smi` needs none of this.
  const char* v = std::getenv("VGPU_NVML_VERSION");
  return copy_string(v && v[0] ? v : "13.580.00.00", version, length);
}
// The session's CUDA version, as nvidia-smi's header prints it ("12.4" ->
// 12040). It was 13000 whatever the session was.
VGPU_EXPORT nvmlReturn_t nvmlSystemGetCudaDriverVersion(int* version) { REQUIRE_INIT();
  if (!version) return NVML_ERROR_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  int major = 13, minor = 0;
  if (g_snap.cuda_version[0]) std::sscanf(g_snap.cuda_version, "%d.%d", &major, &minor);
  *version = major * 1000 + minor * 10;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int* version) { REQUIRE_INIT();
  return nvmlSystemGetCudaDriverVersion(version);
}

/* ---- enumeration and identity ---- */

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* count) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  if (!count) return NVML_ERROR_INVALID_ARGUMENT;
  refresh();
  *count = g_snap.device_count;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCount(unsigned int* count) { REQUIRE_INIT();
  return nvmlDeviceGetCount_v2(count);
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  if (!device) return NVML_ERROR_INVALID_ARGUMENT;
  refresh();
  if (index >= g_snap.device_count) return NVML_ERROR_INVALID_ARGUMENT;
  if (g_lost[index]) return NVML_ERROR_GPU_IS_LOST;
  *device = handle_for(index);
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) { REQUIRE_INIT();
  return nvmlDeviceGetHandleByIndex_v2(index, device);
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  if (!index) return NVML_ERROR_INVALID_ARGUMENT;
  return index_of(device, index) ? NVML_SUCCESS : bad(device);
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  return d ? copy_string(d->name, name, length) : bad(device);
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  return d ? copy_string(d->uuid, uuid, length) : bad(device);
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t, char*, unsigned int) { REQUIRE_INIT();
  return NVML_ERROR_NOT_SUPPORTED;  // virtual devices have no serial number
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !pci) return bad(device);
  std::memset(pci, 0, sizeof *pci);
  unsigned int domain = 0, bus = 0, dev_id = 0;
  std::sscanf(d->bus_id, "%x:%x:%x", &domain, &bus, &dev_id);
  // The two fields are different formats, not one string copied twice: busId
  // carries an eight-digit domain in a 32-byte buffer, busIdLegacy a four-digit
  // one in a 16-byte buffer. Copying the full id into the legacy field
  // truncated it, which is what NVML_DEVICE_PCI_BUS_ID_LEGACY_FMT exists to
  // prevent.
  std::snprintf(pci->busId, sizeof pci->busId, NVML_DEVICE_PCI_BUS_ID_FMT, domain, bus, dev_id);
  std::snprintf(pci->busIdLegacy, sizeof pci->busIdLegacy, NVML_DEVICE_PCI_BUS_ID_LEGACY_FMT,
                domain, bus, dev_id);
  pci->domain = domain;
  pci->bus = bus;
  pci->device = dev_id;
  pci->pciDeviceId = d->pci_device_id;
  pci->pciSubSystemId = d->pci_subsystem_id;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t device, nvmlPciInfo_t* pci) { REQUIRE_INIT();
  return nvmlDeviceGetPciInfo_v3(device, pci);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t* pci) { REQUIRE_INIT();
  return nvmlDeviceGetPciInfo_v3(device, pci);
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major,
                                                            int* minor) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !major || !minor) return bad(device);
  *major = d->cc_major;
  *minor = d->cc_minor;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* cores) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !cores) return bad(device);
  *cores = d->multiprocessors;
  return NVML_SUCCESS;
}

/* ---- live measurements (memory + utilization are REAL) ---- */

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !memory) return bad(device);
  memory->total = d->vram_total_bytes;
  memory->used = d->vram_used_bytes;
  memory->free = d->vram_total_bytes - d->vram_used_bytes;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !memory) return bad(device);
  memory->version = nvmlMemory_v2;
  memory->total = d->vram_total_bytes;
  memory->used = d->vram_used_bytes;
  memory->free = d->vram_total_bytes - d->vram_used_bytes;
  memory->reserved = 0;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device,
                                                       nvmlUtilization_t* utilization) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !utilization) return bad(device);
  utilization->gpu = d->utilization_gpu;
  utilization->memory = d->utilization_mem;
  return NVML_SUCCESS;
}

/* ---- synthetic model: power / thermals / clocks ---- */

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device,
                                                  nvmlTemperatureSensors_t sensorType,
                                                  unsigned int* temp) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !temp) return bad(device);
  if (sensorType != NVML_TEMPERATURE_GPU) return NVML_ERROR_NOT_SUPPORTED;
  *temp = d->temperature_c;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device,
                                                           nvmlTemperatureThresholds_t which,
                                                           unsigned int* temp) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !temp) return bad(device);
  // A profile characterized from a device whose driver reports no threshold
  // carries zero here. Saying "not supported" is what that driver said, and is
  // better than inventing a number a monitoring tool would then act on.
  if (d->temperature_max_c == 0) return NVML_ERROR_NOT_SUPPORTED;
  switch (which) {
    case NVML_TEMPERATURE_THRESHOLD_SHUTDOWN: *temp = d->temperature_max_c + 5; break;
    case NVML_TEMPERATURE_THRESHOLD_SLOWDOWN: *temp = d->temperature_max_c; break;
    case NVML_TEMPERATURE_THRESHOLD_GPU_MAX: *temp = d->temperature_max_c; break;
    default: return NVML_ERROR_NOT_SUPPORTED;
  }
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* milliwatts) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !milliwatts) return bad(device);
  *milliwatts = d->power_mw;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !limit) return bad(device);
  *limit = d->power_limit_mw;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device,
                                                           unsigned int* limit) { REQUIRE_INIT();
  return nvmlDeviceGetEnforcedPowerLimit(device, limit);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device,
                                                                  unsigned int* limit) { REQUIRE_INIT();
  return nvmlDeviceGetEnforcedPowerLimit(device, limit);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device,
                                                                      unsigned int* minLimit,
                                                                      unsigned int* maxLimit) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !minLimit || !maxLimit) return bad(device);
  *minLimit = d->power_limit_mw / 2;
  *maxLimit = d->power_limit_mw;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type,
                                                unsigned int* clock) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !clock) return bad(device);
  switch (type) {
    case NVML_CLOCK_GRAPHICS:
    case NVML_CLOCK_SM: *clock = d->sm_clock_mhz; break;
    case NVML_CLOCK_MEM: *clock = d->mem_clock_mhz; break;
    case NVML_CLOCK_VIDEO: *clock = d->sm_clock_mhz / 2; break;
    default: return NVML_ERROR_NOT_SUPPORTED;
  }
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type,
                                                   unsigned int* clock) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !clock) return bad(device);
  switch (type) {
    case NVML_CLOCK_GRAPHICS:
    case NVML_CLOCK_SM: *clock = d->sm_clock_max_mhz; break;
    case NVML_CLOCK_MEM: *clock = d->mem_clock_max_mhz; break;
    case NVML_CLOCK_VIDEO: *clock = d->sm_clock_max_mhz / 2; break;
    default: return NVML_ERROR_NOT_SUPPORTED;
  }
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !speed) return bad(device);
  *speed = d->fan_percent;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t* state) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !state) return bad(device);
  *state = static_cast<nvmlPstates_t>(d->perf_state);
  return NVML_SUCCESS;
}

/* ---- modes and processes ---- */

// The mode queries answer constants, and used to answer them for any handle at
// all -- NULL, a stale one, a number from nowhere -- so a tool iterating past
// the last device got plausible modes for GPUs that do not exist. The handle is
// checked like every other device query.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t* mode) { REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !mode) return bad(device);
  *mode = NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t* mode) { REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !mode) return bad(device);
  *mode = NVML_COMPUTEMODE_DEFAULT;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t* current,
                                              nvmlEnableState_t* pending) { REQUIRE_INIT();
  // On for a card that ships with ECC, which the profile records; absent on
  // one without it, as on a GeForce card.
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !current || !pending) return bad(device);
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  *current = *pending = NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t, unsigned int*, unsigned int*) { REQUIRE_INIT();
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t* mode) { REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !mode) return bad(device);
  *mode = NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t* mode) { REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !mode) return bad(device);
  *mode = NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device,
                                                                 unsigned int* infoCount,
                                                                 nvmlProcessInfo_t* infos) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !infoCount) return bad(device);
  unsigned int have = d->proc_count;
  if (!infos || *infoCount < have) {
    *infoCount = have;
    return have ? NVML_ERROR_INSUFFICIENT_SIZE : NVML_SUCCESS;
  }
  for (unsigned int i = 0; i < have; ++i) {
    std::memset(&infos[i], 0, sizeof infos[i]);
    infos[i].pid = d->procs[i].pid;
    infos[i].usedGpuMemory = d->procs[i].used_bytes;
  }
  *infoCount = have;
  return NVML_SUCCESS;
}
// _v2 takes nvmlProcessInfo_v2_t, which is a distinct struct from the
// nvmlProcessInfo_t the _v3 entry point uses -- it has no confidential-compute
// memory field. On toolkits where the two happen to be layout-compatible,
// declaring this with nvmlProcessInfo_t builds; on CUDA 12.0 the headers
// disagree and it does not. Fill the v2 struct on its own terms.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t device,
                                                                 unsigned int* infoCount,
                                                                 nvmlProcessInfo_v2_t* infos) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !infoCount) return bad(device);
  unsigned int have = d->proc_count;
  if (!infos || *infoCount < have) {
    *infoCount = have;
    return have ? NVML_ERROR_INSUFFICIENT_SIZE : NVML_SUCCESS;
  }
  for (unsigned int i = 0; i < have; ++i) {
    std::memset(&infos[i], 0, sizeof infos[i]);
    infos[i].pid = d->procs[i].pid;
    infos[i].usedGpuMemory = d->procs[i].used_bytes;
  }
  *infoCount = have;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t, unsigned int* n,
                                                                  nvmlProcessInfo_t*) { REQUIRE_INIT();
  if (n) *n = 0;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t, unsigned int* n,
                                                                    nvmlProcessInfo_t*) { REQUIRE_INIT();
  if (n) *n = 0;
  return NVML_SUCCESS;
}

/* ---- what monitoring tools ask for ----
 *
 * nvitop, gpustat and pynvml-based dashboards call these. A missing symbol is
 * not "not supported": pynvml raises FunctionNotFound, which nvitop does not
 * catch, so the whole tool exited. Where the simulator has an answer it gives
 * it; where it has none it says NOT_SUPPORTED, which every one of these tools
 * renders as N/A -- what they already do on hardware that lacks the feature.
 * Only entry points present in the CUDA 12.0 header are here, so the shim
 * still builds against every toolkit CI uses.
 */

namespace {
bool same_bus_id(const char* a, const char* b) {
  // "00000000:01:00.0" and "0000:01:00.0" name one device: compare from the right.
  const size_t la = std::strlen(a), lb = std::strlen(b);
  const size_t n = la < lb ? la : lb;
  for (size_t i = 1; i <= n; ++i)
    if (std::tolower(static_cast<unsigned char>(a[la - i])) !=
        std::tolower(static_cast<unsigned char>(b[lb - i])))
      return false;
  return n >= 7;   // at least bus:device.function
}
}  // namespace

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  if (!uuid || !device) return NVML_ERROR_INVALID_ARGUMENT;
  refresh();
  for (unsigned int i = 0; i < g_snap.device_count; ++i)
    if (std::strcmp(g_snap.devices[i].uuid, uuid) == 0) {
      if (g_lost[i]) return NVML_ERROR_GPU_IS_LOST;
      *device = handle_for(i);
      return NVML_SUCCESS;
    }
  return NVML_ERROR_NOT_FOUND;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char* busId, nvmlDevice_t* device) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  if (!busId || !device) return NVML_ERROR_INVALID_ARGUMENT;
  refresh();
  for (unsigned int i = 0; i < g_snap.device_count; ++i)
    if (same_bus_id(g_snap.devices[i].bus_id, busId)) {
      if (g_lost[i]) return NVML_ERROR_GPU_IS_LOST;
      *device = handle_for(i);
      return NVML_SUCCESS;
    }
  return NVML_ERROR_NOT_FOUND;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char* busId, nvmlDevice_t* device) {
  return nvmlDeviceGetHandleByPciBusId_v2(busId, device);
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minor) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  if (!minor) return NVML_ERROR_INVALID_ARGUMENT;
  return index_of(device, minor) ? NVML_SUCCESS : bad(device);  // /dev/nvidiaN
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device,
                                                   nvmlDeviceArchitecture_t* arch) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !arch) return bad(device);
  const std::string a = d->architecture;
  *arch = a == "turing"   ? NVML_DEVICE_ARCH_TURING
        : a == "ampere"   ? NVML_DEVICE_ARCH_AMPERE
        : a == "ada" || a == "ada_lovelace" ? NVML_DEVICE_ARCH_ADA
        : a == "hopper"   ? NVML_DEVICE_ARCH_HOPPER
#ifdef NVML_DEVICE_ARCH_BLACKWELL
        : a == "blackwell" ? NVML_DEVICE_ARCH_BLACKWELL
#endif
        : NVML_DEVICE_ARCH_UNKNOWN;
  return NVML_SUCCESS;
}

// Nothing throttles a simulated clock; the empty mask is the true answer, and
// it is the one `nvidia-smi --query-gpu=clocks_throttle_reasons.active` gives.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device,
                                                                   unsigned long long* reasons) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !reasons) return bad(device);
  // GPU idle (bit 0) whenever the device is, as a real idle card reports it,
  // and whatever was injected with `vgpu fault throttle`.
  *reasons = (d->utilization_gpu == 0 ? vgpu::ras::kGpuIdle : 0) | d->clock_event_reasons;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char* name, unsigned int length) {
  REQUIRE_INIT();
  if (!name || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
  char path[64];
  std::snprintf(path, sizeof path, "/proc/%u/comm", pid);
  FILE* f = std::fopen(path, "r");
  if (!f) return NVML_ERROR_NOT_FOUND;
  char buf[256] = {0};
  const bool got = std::fgets(buf, sizeof buf, f) != nullptr;
  std::fclose(f);
  if (!got) return NVML_ERROR_NOT_FOUND;
  buf[std::strcspn(buf, "\n")] = 0;
  return copy_string(buf, name, length);
}

// MIG: these devices are not partitioned, which is an answer, not a gap.
VGPU_EXPORT nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int* isMig) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  if (!isMig) return NVML_ERROR_INVALID_ARGUMENT;
  unsigned int idx;
  if (!index_of(device, &idx)) return bad(device);
  *isMig = 0;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t, unsigned int*) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t, unsigned int, nvmlDevice_t*) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t, unsigned int*) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t, unsigned int*) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t, nvmlDevice_t*) {
  REQUIRE_INIT(); return NVML_ERROR_INVALID_ARGUMENT;   // no handle here is a MIG handle
}

// No data the simulator keeps. N/A in every tool, as on hardware without them.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t, unsigned int*, unsigned int*) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t, unsigned int*, unsigned int*) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t, nvmlPcieUtilCounter_t, unsigned int*) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}
// The PCIe link, read from the device's PCI Express capability as a driver
// reads it -- Link Capabilities for the maximum, Link Status for the link as
// trained -- so the read shows in the register access log (vgpu regs log).
// Without register state to read, the reading's own link.
static nvmlReturn_t pcie_link(nvmlDevice_t device, unsigned int* out, bool generation, bool current) {
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !out) return bad(device);
  vgpu::regs::Link l{d->pcie_gen, d->pcie_width, d->pcie_gen_max, d->pcie_width_max};
  try {
    vgpu::regs::ConfigSpace cs(*d);
    l = vgpu::regs::link(cs);
  } catch (const std::exception&) {
  }
  const unsigned v = generation ? (current ? l.gen : l.max_gen) : (current ? l.width : l.max_width);
  if (!v) return NVML_ERROR_NOT_SUPPORTED;
  *out = v;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* gen) {
  REQUIRE_INIT(); return pcie_link(device, gen, true, true);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* gen) {
  REQUIRE_INIT(); return pcie_link(device, gen, true, false);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* width) {
  REQUIRE_INIT(); return pcie_link(device, width, false, true);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int* width) {
  REQUIRE_INIT(); return pcie_link(device, width, false, false);
}
// ECC counts come from the machine's reliability state (vgpu/ras.hpp), the one
// nvidia-smi reads: zero until something is injected with `vgpu fault`.
// errorType 0 is corrected, 1 uncorrected; counterType 0 volatile, 1 aggregate.
static const vgpu::ras::Counters& ecc_counts(const vgpu::ras::State& st, int counter_type) {
  return counter_type == 1 ? st.lifetime : st.since_load;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t error_type,
                                                     nvmlEccCounterType_t counter_type,
                                                     unsigned long long* count) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !count || static_cast<int>(error_type) > 1 || static_cast<int>(counter_type) > 1)
    return bad(device);
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;   // matches nvmlDeviceGetEccMode
  const vgpu::ras::State st = vgpu::ras::read(d->uuid);
  *count = ecc_counts(st, static_cast<int>(counter_type))
               .ecc_total(error_type == 0 ? vgpu::ras::Severity::Corrected : vgpu::ras::Severity::Uncorrected);
  return NVML_SUCCESS;
}
// The same counts by where they happened. NVML's locations, by number: L1 0,
// L2 1, device memory 2, register file 3, texture memory 4, texture shared 5,
// CBU 6, SRAM 7. Texture shared memory is not a location VirtualGPU tracks.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device,
                                                         nvmlMemoryErrorType_t error_type,
                                                         nvmlEccCounterType_t counter_type,
                                                         nvmlMemoryLocation_t location,
                                                         unsigned long long* count) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  const int loc = static_cast<int>(location);
  if (!d || !count || static_cast<int>(error_type) > 1 || static_cast<int>(counter_type) > 1 ||
      loc < 0 || loc > 7)
    return bad(device);
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  using L = vgpu::ras::Location;
  static const int kMap[8] = {static_cast<int>(L::L1Cache),       static_cast<int>(L::L2Cache),
                              static_cast<int>(L::DeviceMemory),  static_cast<int>(L::RegisterFile),
                              static_cast<int>(L::TextureMemory), -1,
                              static_cast<int>(L::Cbu),           static_cast<int>(L::Sram)};
  if (kMap[loc] < 0) {
    *count = 0;
    return NVML_SUCCESS;
  }
  const vgpu::ras::State st = vgpu::ras::read(d->uuid);
  const auto sev = error_type == 0 ? 0u : 1u;
  *count = ecc_counts(st, static_cast<int>(counter_type)).ecc[sev][kMap[loc]];
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t, nvmlDriverModel_t*, nvmlDriverModel_t*) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;   // a Windows-only query
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t, nvmlBAR1Memory_t*) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t, char*, unsigned int) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;   // a virtual device has no VBIOS
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device,
                                                             unsigned long long* energy) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  unsigned int idx;
  if (!d || !energy || !index_of(device, &idx)) return bad(device);
  *energy = energy_mj(idx, *d);
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t, nvmlProcessUtilizationSample_t*,
                                                         unsigned int* count, unsigned long long) {
  REQUIRE_INIT();
  if (count) *count = 0;
  return NVML_ERROR_NOT_FOUND;   // the documented answer when there are no samples
}
// Each field carries its own status, and the call itself succeeds.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int count,
                                                  nvmlFieldValue_t* values) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  unsigned int idx;
  if (!index_of(device, &idx) || count < 0 || (count > 0 && !values))
    return bad(device);
  refresh();
  const auto* d = sample(device);
  // NVML's documented field ids, by number: they are ABI, and the oldest
  // toolkit headers this builds against predate some of the names.
  enum : unsigned {
    kMemoryTemp = 82, kPcieReplay = 94, kPcieReplayRollover = 95, kPcieL0ToRecovery = 169,
    kPcieCorrectable = 173, kPcieNaksReceived = 174, kPcieBadTlp = 176, kPcieNaksSent = 177,
    kPcieBadDllp = 178, kPcieNonFatal = 179, kPcieFatal = 180, kPcieLcrc = 182, kPcieLane = 183,
  };
  const long long now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
  const vgpu::ras::Counters pcie = d ? vgpu::ras::read(d->uuid).since_load : vgpu::ras::Counters{};
  auto pcie_count = [&](unsigned field) -> unsigned long long {
    using P = vgpu::ras::Pcie;
    P c;
    switch (field) {
      case kPcieReplay: c = P::Replay; break;
      case kPcieReplayRollover: c = P::ReplayRollover; break;
      case kPcieL0ToRecovery: c = P::L0ToRecovery; break;
      case kPcieCorrectable: c = P::Correctable; break;
      case kPcieNaksReceived: c = P::NaksReceived; break;
      case kPcieBadTlp: c = P::BadTlp; break;
      case kPcieNaksSent: c = P::NaksSent; break;
      case kPcieBadDllp: c = P::BadDllp; break;
      case kPcieNonFatal: c = P::NonFatal; break;
      case kPcieFatal: c = P::Fatal; break;
      case kPcieLcrc: c = P::Lcrc; break;
      default: c = P::Lane; break;
    }
    return pcie.pcie[static_cast<uint32_t>(c)];
  };
  // What DCGM's cache manager reads through this call: ECC by location, page
  // retirement and row remapping, the time spent in each clock-event reason, and
  // the energy counter. All from the same state the dedicated entry points read.
  const vgpu::ras::State ras_state = d ? vgpu::ras::read(d->uuid) : vgpu::ras::State{};
  auto set_ull = [](nvmlFieldValue_t& v, unsigned long long n) {
    v.valueType = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
    v.value.ullVal = n;
    v.nvmlReturn = NVML_SUCCESS;
  };
  auto set_uint = [](nvmlFieldValue_t& v, unsigned int n) {
    v.valueType = NVML_VALUE_TYPE_UNSIGNED_INT;
    v.value.uiVal = n;
    v.nvmlReturn = NVML_SUCCESS;
  };
  for (int i = 0; i < count; ++i) {
    nvmlFieldValue_t& v = values[i];
    v.nvmlReturn = NVML_ERROR_NOT_SUPPORTED;
    v.timestamp = now_us;
    v.latencyUsec = 0;
    switch (v.fieldId) {
      // PCIe transport error counters. A real card answers these whatever its
      // class -- an RTX 3060 does. A simulated link errs only when an error is
      // injected (`vgpu fault inject --pcie`), so they are zero until then.
      case kPcieReplay: case kPcieReplayRollover: case kPcieL0ToRecovery:
      case kPcieCorrectable: case kPcieNaksReceived: case kPcieBadTlp: case kPcieNaksSent:
      case kPcieBadDllp: case kPcieNonFatal: case kPcieFatal: case kPcieLcrc: case kPcieLane:
        v.valueType = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
        v.value.ullVal = pcie_count(v.fieldId);
        v.nvmlReturn = NVML_SUCCESS;
        break;
      // The memory sensor, only where real cards of the model report one.
      case kMemoryTemp:
        if (d && d->has_memory_temperature) {
          v.valueType = NVML_VALUE_TYPE_UNSIGNED_INT;
          v.value.uiVal = d->temperature_mem_c ? d->temperature_mem_c : d->temperature_c;
          v.nvmlReturn = NVML_SUCCESS;
        }
        break;
      default:
        if (!d) break;
        // ECC: fields 1 and 2 are the mode, 3..6 the totals (single and double
        // bit, volatile then aggregate), 7..16 the volatile and 18..27 the
        // aggregate counts by location in (single, double) pairs: L1, L2,
        // device memory, register file, texture memory.
        if (v.fieldId >= 1 && v.fieldId <= 27 && v.fieldId != 17) {
          if (!d->ecc_enabled) break;
          using S = vgpu::ras::Severity;
          using L = vgpu::ras::Location;
          if (v.fieldId <= 2) { set_uint(v, 1); break; }
          if (v.fieldId <= 6) {
            const auto& c = v.fieldId <= 4 ? ras_state.since_load : ras_state.lifetime;
            set_ull(v, c.ecc_total(v.fieldId % 2 ? S::Corrected : S::Uncorrected));
            break;
          }
          static const L kWhere[5] = {L::L1Cache, L::L2Cache, L::DeviceMemory, L::RegisterFile,
                                      L::TextureMemory};
          const unsigned base = v.fieldId >= 18 ? 18 : 7;
          const unsigned at = (v.fieldId - base) / 2, double_bit = (v.fieldId - base) % 2;
          if (at >= 5) break;
          const auto& c = base == 7 ? ras_state.since_load : ras_state.lifetime;
          set_ull(v, c.ecc[double_bit][static_cast<unsigned>(kWhere[at])]);
          break;
        }
        // Retired pages are a GDDR card's; row remapping an HBM card's.
        if (v.fieldId >= 29 && v.fieldId <= 31 && d->memory_retirement == 1) {
          const auto& c = ras_state.lifetime;
          set_ull(v, v.fieldId == 29 ? c.retired_sbe : v.fieldId == 30 ? c.retired_dbe : c.retired_pending);
          break;
        }
        if (v.fieldId >= 142 && v.fieldId <= 145 && d->memory_retirement == 2) {
          const auto& c = ras_state.lifetime;
          set_uint(v, static_cast<unsigned>(v.fieldId == 142   ? c.rows_correctable
                                            : v.fieldId == 143 ? c.rows_uncorrectable
                                            : v.fieldId == 144 ? c.rows_pending
                                                               : c.rows_failure));
          break;
        }
        // Time spent in each performance policy, in nanoseconds; the clock-event
        // reason counters are the same measure by reason.
        if ((v.fieldId >= 74 && v.fieldId <= 81) || (v.fieldId >= 269 && v.fieldId <= 271)) {
          set_ull(v, reason_time_us(d->uuid, policy_reasons(v.fieldId)) * 1000);
          break;
        }
        if (v.fieldId == 83) {   // total energy
          unsigned int at;
          if (index_of(device, &at)) set_ull(v, energy_mj(at, *d));
          break;
        }
        if (v.fieldId == 230) set_uint(v, 0);   // GPU recovery action: none needed
        break;
    }
  }
  return NVML_SUCCESS;
}
/* ---- events ---- */
// An event set watches devices for the event types each was registered for.
// Events come from the machine's reliability state (vgpu/ras.hpp), where
// `vgpu fault` and faults delivered to running kernels record them, so a health
// daemon waiting here sees what happens in any process on the machine.
struct nvmlEventSet_st {
  struct Watch {
    unsigned int index;
    std::string uuid;
    unsigned long long types;
    uint64_t after;   // the last event this watch has looked at
  };
  std::mutex mu;
  std::vector<Watch> watches;
};

namespace {
// Xid on every card; ECC events only where there is ECC to report them.
unsigned long long supported_events(const vgpu::telemetry::DeviceSample& d) {
  return vgpu::ras::kEventXid |
         (d.ecc_enabled ? vgpu::ras::kEventSingleBitEcc | vgpu::ras::kEventDoubleBitEcc : 0);
}

// Polls the watches until one has an event or the time runs out. The set's
// lock is released while sleeping, so another thread may register meanwhile.
nvmlReturn_t wait_event(nvmlEventSet_t set, unsigned int timeout_ms, nvmlDevice_t* device,
                        unsigned long long* type, unsigned long long* data) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  for (;;) {
    {
      std::lock_guard<std::mutex> lock(set->mu);
      for (auto& w : set->watches) {
        vgpu::ras::Event e{};
        bool got = false;
        try {
          got = vgpu::ras::next_event(w.uuid, w.types, &w.after, &e);
        } catch (const std::exception&) {
        }
        if (got) {
          *device = handle_for(w.index);
          *type = e.type;
          *data = e.data;
          return NVML_SUCCESS;
        }
      }
    }
    if (std::chrono::steady_clock::now() >= deadline) return NVML_ERROR_TIMEOUT;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
}  // namespace

VGPU_EXPORT nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t* set) {
  REQUIRE_INIT();
  if (!set) return NVML_ERROR_INVALID_ARGUMENT;
  *set = new nvmlEventSet_st();
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set) {
  REQUIRE_INIT();
  if (!set) return NVML_ERROR_INVALID_ARGUMENT;
  delete set;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device,
                                                          unsigned long long* types) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !types) return bad(device);
  *types = supported_events(*d);
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long types,
                                                  nvmlEventSet_t set) {
  REQUIRE_INIT();
  std::string uuid;
  unsigned int index = 0;
  unsigned long long supported = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(g_mu);
    refresh();
    const auto* d = sample(device);
    if (!d || !set || !index_of(device, &index)) return bad(device);
    supported = supported_events(*d);
    uuid = d->uuid;
  }
  if (types & ~supported) return NVML_ERROR_NOT_SUPPORTED;
  uint64_t head = 0;
  try {
    head = vgpu::ras::event_head(uuid);   // only what happens from now on
  } catch (const std::exception&) {
  }
  std::lock_guard<std::mutex> lock(set->mu);
  set->watches.push_back({index, uuid, types, head});
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t* data,
                                             unsigned int timeout_ms) {
  REQUIRE_INIT();
  if (!set || !data) return NVML_ERROR_INVALID_ARGUMENT;
  nvmlDevice_t device = nullptr;
  unsigned long long type = 0, value = 0;
  if (const nvmlReturn_t rc = wait_event(set, timeout_ms, &device, &type, &value); rc != NVML_SUCCESS)
    return rc;
  data->device = device;
  data->eventType = type;
  data->eventData = value;
  data->gpuInstanceId = 0xFFFFFFFFu;       // not attributable to a MIG instance
  data->computeInstanceId = 0xFFFFFFFFu;
  return NVML_SUCCESS;
}
// The first version, from before MIG: its event data ends after eventData, so
// writing the instance ids would run past the caller's structure.
VGPU_EXPORT nvmlReturn_t nvmlEventSetWait(nvmlEventSet_t set, nvmlEventData_t* data,
                                          unsigned int timeout_ms) {
  REQUIRE_INIT();
  if (!set || !data) return NVML_ERROR_INVALID_ARGUMENT;
  nvmlDevice_t device = nullptr;
  unsigned long long type = 0, value = 0;
  if (const nvmlReturn_t rc = wait_event(set, timeout_ms, &device, &type, &value); rc != NVML_SUCCESS)
    return rc;
  data->device = device;
  data->eventType = type;
  data->eventData = value;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(nvmlDevice_t, unsigned int* n,
                                                                  nvmlProcessInfo_v2_t*) {
  REQUIRE_INIT();
  if (n) *n = 0;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v2(nvmlDevice_t, unsigned int* n,
                                                                    nvmlProcessInfo_v2_t*) {
  REQUIRE_INIT();
  if (n) *n = 0;
  return NVML_SUCCESS;
}

/* ---- what DCGM and NVVS ask for ----
 *
 * NVIDIA's DCGM (`dcgmi diag`, NVVS and its plugins) reads the GPU through
 * NVML: its hostengine resolves every entry point by name, treats a missing one
 * as FUNCTION_NOT_FOUND and a NOT_SUPPORTED answer as "this part has no such
 * thing", and runs a diagnostic only on what it could read. These are the
 * queries behind the fields the diagnostics and the health watches use -- clock
 * events, thermal and power violations, ECC, retired pages and remapped rows,
 * the PCIe link and its replay count, topology -- answered from the same
 * profile and reliability state as everything above, so an injected fault
 * (`vgpu fault`) shows up in DCGM as it does in nvidia-smi. Entry points that
 * postdate CUDA 12.0's nvml.h are declared here by their plain-C types, so the
 * shim still builds against every toolkit CI uses.
 */

extern "C" {
nvmlReturn_t nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t, unsigned long long*);
nvmlReturn_t nvmlDeviceGetSupportedClocksEventReasons(nvmlDevice_t, unsigned long long*);
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t device,
                                                                unsigned long long* reasons) {
  REQUIRE_INIT();
  return nvmlDeviceGetCurrentClocksThrottleReasons(device, reasons);   // the renamed query
}
// The reasons `vgpu fault throttle` can raise, plus GPU idle: the ones the
// model can say are active.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device,
                                                                     unsigned long long* reasons) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  unsigned int idx;
  if (!index_of(device, &idx) || !reasons) return bad(device);
  *reasons = vgpu::ras::kGpuIdle | vgpu::ras::kSwPowerCap | vgpu::ras::kHwSlowdown |
             vgpu::ras::kSwThermalSlowdown | vgpu::ras::kHwThermalSlowdown |
             vgpu::ras::kHwPowerBrakeSlowdown;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSupportedClocksEventReasons(nvmlDevice_t device,
                                                                  unsigned long long* reasons) {
  REQUIRE_INIT();
  return nvmlDeviceGetSupportedClocksThrottleReasons(device, reasons);
}

// How long the device has spent held back by a performance policy. referenceTime
// is the host clock in microseconds, violationTime the time in the policy in
// nanoseconds; NVVS compares two readings and fails a run whose thermal or power
// violation time grew.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device,
                                                      nvmlPerfPolicyType_t policy,
                                                      nvmlViolationTime_t* viol) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !viol) return bad(device);
  // Policy numbers are NVML's: power 0, thermal 1, sync boost 2, board limit 3,
  // low utilization 4, reliability 5, total app clocks 10, total base clocks 11.
  static const int kField[12] = {74, 75, 76, 77, 78, 79, -1, -1, -1, -1, 80, 81};
  const int p = static_cast<int>(policy);
  if (p < 0 || p > 11 || kField[p] < 0) return NVML_ERROR_INVALID_ARGUMENT;
  viol->referenceTime = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
  viol->violationTime = reason_time_us(d->uuid, policy_reasons(kField[p])) * 1000;
  return NVML_SUCCESS;
}

/* identity */

// Datacenter cards are Tesla as NVML brands them; the profile's name says the rest.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t* type) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !type) return bad(device);
  const std::string name = d->name;
  auto has = [&](const char* s) { return name.find(s) != std::string::npos; };
  *type = has("GeForce") ? NVML_BRAND_GEFORCE
          : has("TITAN") ? NVML_BRAND_TITAN
          : has("Quadro") ? NVML_BRAND_QUADRO
          : has("RTX")   ? NVML_BRAND_NVIDIA_RTX
                         : NVML_BRAND_TESLA;
  return NVML_SUCCESS;
}
// One board per GPU, told apart by where it sits on the bus.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int* id) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !id) return bad(device);
  unsigned int domain = 0, bus = 0, dev_id = 0;
  std::sscanf(d->bus_id, "%x:%x:%x", &domain, &bus, &dev_id);
  *id = bus << 8 | dev_id;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int* multi) {
  REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !multi) return bad(device);
  *multi = 0;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t* type) {
  REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !type) return bad(device);
  *type = NVML_BUS_TYPE_PCIE;
  return NVML_SUCCESS;
}
// A virtual device has no infoROM, as it has no VBIOS and no serial number.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t, nvmlInforomObject_t, char*,
                                                     unsigned int) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t, char*, unsigned int) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device,
                                                          nvmlEnableState_t* mode) {
  REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !mode) return bad(device);
  *mode = NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t* state) {
  REQUIRE_INIT();
  return nvmlDeviceGetPerformanceState(device, state);
}

/* clocks */

// The simulated memory clock does not step: one supported memory clock, the
// profile's, and graphics clocks from the profile's maximum down in the 15 MHz
// steps real boards use, to the 210 MHz floor they idle at.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device,
                                                            unsigned int* count,
                                                            unsigned int* clocks) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !count) return bad(device);
  if (!d->mem_clock_max_mhz) return NVML_ERROR_NOT_SUPPORTED;
  const unsigned int have = *count;
  *count = 1;
  if (!clocks || have < 1) return NVML_ERROR_INSUFFICIENT_SIZE;
  clocks[0] = d->mem_clock_max_mhz;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device,
                                                              unsigned int memory_mhz,
                                                              unsigned int* count,
                                                              unsigned int* clocks) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !count) return bad(device);
  if (!d->sm_clock_max_mhz) return NVML_ERROR_NOT_SUPPORTED;
  if (memory_mhz != d->mem_clock_max_mhz) return NVML_ERROR_INVALID_ARGUMENT;
  constexpr unsigned int kFloor = 210, kStep = 15;
  const unsigned int top = d->sm_clock_max_mhz;
  const unsigned int n = top > kFloor ? (top - kFloor) / kStep + 1 : 1;
  const unsigned int have = *count;
  *count = n;
  if (!clocks || have < n) return NVML_ERROR_INSUFFICIENT_SIZE;
  for (unsigned int i = 0; i < n; ++i) clocks[i] = top - i * kStep;
  return NVML_SUCCESS;
}
// The application clocks are the maximum clocks, which is what a card nobody
// has set application clocks on runs at; so are the customer boost maximum.
static nvmlReturn_t max_clock_of(nvmlDevice_t device, nvmlClockType_t type, unsigned int* mhz) {
  return nvmlDeviceGetMaxClockInfo(device, type, mhz);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t type,
                                                        unsigned int* mhz) {
  REQUIRE_INIT(); return max_clock_of(device, type, mhz);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device,
                                                               nvmlClockType_t type,
                                                               unsigned int* mhz) {
  REQUIRE_INIT(); return max_clock_of(device, type, mhz);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device,
                                                            nvmlClockType_t type,
                                                            unsigned int* mhz) {
  REQUIRE_INIT(); return max_clock_of(device, type, mhz);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t type,
                                            nvmlClockId_t id, unsigned int* mhz) {
  REQUIRE_INIT();
  return id == NVML_CLOCK_ID_CURRENT ? nvmlDeviceGetClockInfo(device, type, mhz)
                                     : max_clock_of(device, type, mhz);
}

/* ECC, retired pages, remapped rows */

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device,
                                                     nvmlEnableState_t* mode) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !mode) return bad(device);
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  *mode = NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device,
                                                        nvmlMemoryErrorType_t error_type,
                                                        nvmlEccCounterType_t counter_type,
                                                        nvmlEccErrorCounts_t* counts) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !counts || static_cast<int>(error_type) > 1 || static_cast<int>(counter_type) > 1)
    return bad(device);
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  using L = vgpu::ras::Location;
  const vgpu::ras::State st = vgpu::ras::read(d->uuid);
  const auto& c = ecc_counts(st, static_cast<int>(counter_type));
  const unsigned sev = error_type == 0 ? 0u : 1u;
  counts->l1Cache = c.ecc[sev][static_cast<unsigned>(L::L1Cache)];
  counts->l2Cache = c.ecc[sev][static_cast<unsigned>(L::L2Cache)];
  counts->deviceMemory = c.ecc[sev][static_cast<unsigned>(L::DeviceMemory)];
  counts->registerFile = c.ecc[sev][static_cast<unsigned>(L::RegisterFile)];
  return NVML_SUCCESS;
}
// `nvidia-smi -p`: zeroes the volatile counts (0) or the aggregate ones (1).
// Retired pages and remapped rows stay, as on the card.
VGPU_EXPORT nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device,
                                                       nvmlEccCounterType_t counter_type) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || static_cast<int>(counter_type) > 1) return bad(device);
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  try {
    if (counter_type == NVML_VOLATILE_ECC) vgpu::ras::reset_volatile(d->uuid, false);
    else vgpu::ras::reset_aggregate(d->uuid);
  } catch (const std::exception&) {
    return NVML_ERROR_UNKNOWN;
  }
  return NVML_SUCCESS;
}

// Pages retired for `cause` (0 single-bit, 1 double-bit), on the cards that
// retire pages (GDDR with ECC). The addresses are made up, one 64 KiB frame
// apart: the simulator retires a count, not a particular frame.
static nvmlReturn_t retired_pages(nvmlDevice_t device, nvmlPageRetirementCause_t cause,
                                  unsigned int* count, unsigned long long* addresses,
                                  unsigned long long* timestamps) {
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !count || static_cast<int>(cause) < 0 || static_cast<int>(cause) > 1) return bad(device);
  if (d->memory_retirement != 1) return NVML_ERROR_NOT_SUPPORTED;
  const auto& c = vgpu::ras::read(d->uuid).lifetime;
  const unsigned int have = *count;
  const unsigned int n = static_cast<unsigned int>(cause == 0 ? c.retired_sbe : c.retired_dbe);
  *count = n;
  if (n > have || (n && !addresses)) return NVML_ERROR_INSUFFICIENT_SIZE;
  for (unsigned int i = 0; i < n; ++i) {
    addresses[i] = 0x10000000ull + (cause == 0 ? 0 : 0x1000000ull) + i * 0x10000ull;
    if (timestamps) timestamps[i] = 0;
  }
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device,
                                                   nvmlPageRetirementCause_t cause,
                                                   unsigned int* count,
                                                   unsigned long long* addresses) {
  REQUIRE_INIT(); return retired_pages(device, cause, count, addresses, nullptr);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device,
                                                      nvmlPageRetirementCause_t cause,
                                                      unsigned int* count,
                                                      unsigned long long* addresses,
                                                      unsigned long long* timestamps) {
  REQUIRE_INIT(); return retired_pages(device, cause, count, addresses, timestamps);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device,
                                                                nvmlEnableState_t* pending) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !pending) return bad(device);
  if (d->memory_retirement != 1) return NVML_ERROR_NOT_SUPPORTED;
  *pending = vgpu::ras::read(d->uuid).lifetime.retired_pending ? NVML_FEATURE_ENABLED
                                                              : NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}
// Row remapping, on the HBM cards that do it.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int* corrected,
                                                   unsigned int* uncorrected, unsigned int* pending,
                                                   unsigned int* failure) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !corrected || !uncorrected || !pending || !failure) return bad(device);
  if (d->memory_retirement != 2) return NVML_ERROR_NOT_SUPPORTED;
  const auto& c = vgpu::ras::read(d->uuid).lifetime;
  *corrected = static_cast<unsigned int>(c.rows_correctable);
  *uncorrected = static_cast<unsigned int>(c.rows_uncorrectable);
  *pending = c.rows_pending ? 1 : 0;
  *failure = c.rows_failure ? 1 : 0;
  return NVML_SUCCESS;
}
// The row-remapper's bank histogram is not modelled: banks are not.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t, nvmlRowRemapperHistogramValues_t*) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}

/* PCIe, topology, affinity */

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int* value) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !value) return bad(device);
  *value = static_cast<unsigned int>(
      vgpu::ras::read(d->uuid).since_load.pcie[static_cast<unsigned>(vgpu::ras::Pcie::Replay)]);
  return NVML_SUCCESS;
}
// Per-lane speed of the link as trained, in MB/s, by generation.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int* mbps) {
  REQUIRE_INIT();
  static const unsigned int kMbps[7] = {0, 2500, 5000, 8000, 16000, 32000, 64000};
  unsigned int gen = 0;
  if (!mbps) return NVML_ERROR_INVALID_ARGUMENT;
  if (const nvmlReturn_t rc = pcie_link(device, &gen, true, true); rc != NVML_SUCCESS) return rc;
  if (gen < 1 || gen > 6) return NVML_ERROR_NOT_SUPPORTED;
  *mbps = kMbps[gen];
  return NVML_SUCCESS;
}
// The maximum link speed as NVML_PCIE_LINK_MAX_SPEED_*: 1 is 2500 MB/s, and so
// on up through the generations.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int* speed) {
  REQUIRE_INIT();
  unsigned int gen = 0;
  if (!speed) return NVML_ERROR_INVALID_ARGUMENT;
  if (const nvmlReturn_t rc = pcie_link(device, &gen, true, false); rc != NVML_SUCCESS) return rc;
  if (gen < 1 || gen > 6) return NVML_ERROR_NOT_SUPPORTED;
  *speed = gen;
  return NVML_SUCCESS;
}

// Every GPU sits behind the host bridge: the topology `vgpu smi topo` prints
// (PHB), since no profile records NVLink.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t a, nvmlDevice_t b,
                                                             nvmlGpuTopologyLevel_t* level) {
  REQUIRE_INIT();
  unsigned int ia, ib;
  if (!index_of(a, &ia)) return bad(a);
  if (!index_of(b, &ib)) return bad(b);
  if (!level || ia == ib) return NVML_ERROR_INVALID_ARGUMENT;
  *level = NVML_TOPOLOGY_HOSTBRIDGE;
  return NVML_SUCCESS;
}
// Peer reads, writes and atomics work between any two devices (cuDeviceCanAccessPeer
// says so), over PCIe; there is no NVLink to report.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t a, nvmlDevice_t b,
                                                nvmlGpuP2PCapsIndex_t cap,
                                                nvmlGpuP2PStatus_t* status) {
  REQUIRE_INIT();
  unsigned int ia, ib;
  if (!index_of(a, &ia)) return bad(a);
  if (!index_of(b, &ib)) return bad(b);
  if (!status || ia == ib) return NVML_ERROR_INVALID_ARGUMENT;
  switch (static_cast<int>(cap)) {
    case NVML_P2P_CAPS_INDEX_READ:
    case NVML_P2P_CAPS_INDEX_WRITE:
    case NVML_P2P_CAPS_INDEX_ATOMICS:
    case NVML_P2P_CAPS_INDEX_PCI: *status = NVML_P2P_STATUS_OK; break;
    case NVML_P2P_CAPS_INDEX_NVLINK: *status = NVML_P2P_STATUS_NOT_SUPPORTED; break;
    default: return NVML_ERROR_INVALID_ARGUMENT;
  }
  return NVML_SUCCESS;
}
// The host CPUs, as the bit mask NVML returns, and one memory node.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int words,
                                                  unsigned long* set) {
  REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !set || words == 0) return bad(device);
  constexpr unsigned int kBits = 8 * sizeof(unsigned long);
  const unsigned int cpus = vgpu::host_cpus();
  for (unsigned int w = 0; w < words; ++w) {
    const unsigned int lo = w * kBits;
    set[w] = lo >= cpus ? 0ul : cpus - lo >= kBits ? ~0ul : (1ul << (cpus - lo)) - 1;
  }
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int words,
                                                     unsigned long* set, nvmlAffinityScope_t) {
  REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !set || words == 0) return bad(device);
  for (unsigned int w = 0; w < words; ++w) set[w] = w == 0 ? 1ul : 0ul;
  return NVML_SUCCESS;
}

// No profile records NVLink: these are PCIe parts as far as the simulator knows,
// and a PCIe part answers NOT_SUPPORTED, which DCGM takes to mean zero links.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int,
                                                  nvmlEnableState_t*) {
  REQUIRE_INIT();
  unsigned int idx;
  return index_of(device, &idx) ? NVML_ERROR_NOT_SUPPORTED : bad(device);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int,
                                                    unsigned int*) {
  REQUIRE_INIT();
  unsigned int idx;
  return index_of(device, &idx) ? NVML_ERROR_NOT_SUPPORTED : bad(device);
}

/* ---- undocumented internal handshake ----
 *
 * nvidia-smi gates startup on nvmlInternalGetExportTable, an undocumented
 * vtable query (the NVML analogue of cuGetExportTable). Its contents are not
 * published, so VirtualGPU reports it as unsupported rather than fabricating a
 * table nvidia-smi would then call through and crash on. Consequence: the
 * stock nvidia-smi binary refuses to run against this NVML. Use `vgpu smi`
 * (or the drop-in nvidia-smi in build/bin) instead; both read the same
 * telemetry. See docs/telemetry.md.
 */
VGPU_EXPORT nvmlReturn_t nvmlInternalGetExportTable(const void** table, void* uuid) {
  (void)uuid;
  if (table) *table = nullptr;
  return NVML_ERROR_NOT_SUPPORTED;
}
