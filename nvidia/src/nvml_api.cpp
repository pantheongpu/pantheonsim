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

#include "vgpu/profile.hpp"
#include "vgpu/ras.hpp"
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

// Handles are 1-based indices encoded as pointers, so they are never null.
nvmlDevice_t handle_for(unsigned int index) {
  return reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(index) + 1);
}
bool index_of(nvmlDevice_t dev, unsigned int* out) {
  uintptr_t v = reinterpret_cast<uintptr_t>(dev);
  if (v == 0) return false;
  unsigned int idx = static_cast<unsigned int>(v - 1);
  if (idx >= g_snap.device_count) return false;
  *out = idx;
  return true;
}

// Re-reads telemetry so repeated queries (nvidia-smi -l) show live values.
// With nothing publishing, a machine described by the environment answers as
// an idle rack -- see nvmlInit_v2.
bool refresh() {
  if (!vgpu::telemetry::read_snapshot(&g_snap)) {
    if (!g_have_profile) return false;
    g_snap = g_idle;
  }
  // Injected clock-event reasons and the readings they imply, applied once
  // here so every getter agrees with every other, and with nvidia-smi.
  for (uint32_t i = 0; i < g_snap.device_count; ++i) {
    try {
      vgpu::ras::apply_throttle(g_snap.devices[i]);
    } catch (const std::exception&) {
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
  *device = handle_for(index);
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) { REQUIRE_INIT();
  return nvmlDeviceGetHandleByIndex_v2(index, device);
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  if (!index) return NVML_ERROR_INVALID_ARGUMENT;
  return index_of(device, index) ? NVML_SUCCESS : NVML_ERROR_INVALID_ARGUMENT;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  return d ? copy_string(d->name, name, length) : NVML_ERROR_INVALID_ARGUMENT;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  return d ? copy_string(d->uuid, uuid, length) : NVML_ERROR_INVALID_ARGUMENT;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t, char*, unsigned int) { REQUIRE_INIT();
  return NVML_ERROR_NOT_SUPPORTED;  // virtual devices have no serial number
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !pci) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!d || !major || !minor) return NVML_ERROR_INVALID_ARGUMENT;
  *major = d->cc_major;
  *minor = d->cc_minor;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* cores) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !cores) return NVML_ERROR_INVALID_ARGUMENT;
  *cores = d->multiprocessors;
  return NVML_SUCCESS;
}

/* ---- live measurements (memory + utilization are REAL) ---- */

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !memory) return NVML_ERROR_INVALID_ARGUMENT;
  memory->total = d->vram_total_bytes;
  memory->used = d->vram_used_bytes;
  memory->free = d->vram_total_bytes - d->vram_used_bytes;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !memory) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!d || !utilization) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!d || !temp) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!d || !temp) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!d || !milliwatts) return NVML_ERROR_INVALID_ARGUMENT;
  *milliwatts = d->power_mw;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !limit) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!d || !minLimit || !maxLimit) return NVML_ERROR_INVALID_ARGUMENT;
  *minLimit = d->power_limit_mw / 2;
  *maxLimit = d->power_limit_mw;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type,
                                                unsigned int* clock) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !clock) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!d || !clock) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!d || !speed) return NVML_ERROR_INVALID_ARGUMENT;
  *speed = d->fan_percent;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t* state) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !state) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!index_of(device, &idx) || !mode) return NVML_ERROR_INVALID_ARGUMENT;
  *mode = NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t* mode) { REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !mode) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!d || !current || !pending) return NVML_ERROR_INVALID_ARGUMENT;
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  *current = *pending = NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t, unsigned int*, unsigned int*) { REQUIRE_INIT();
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t* mode) { REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !mode) return NVML_ERROR_INVALID_ARGUMENT;
  *mode = NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t* mode) { REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !mode) return NVML_ERROR_INVALID_ARGUMENT;
  *mode = NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device,
                                                                 unsigned int* infoCount,
                                                                 nvmlProcessInfo_t* infos) { REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !infoCount) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!d || !infoCount) return NVML_ERROR_INVALID_ARGUMENT;
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
  return index_of(device, minor) ? NVML_SUCCESS : NVML_ERROR_INVALID_ARGUMENT;  // /dev/nvidiaN
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device,
                                                   nvmlDeviceArchitecture_t* arch) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !arch) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!d || !reasons) return NVML_ERROR_INVALID_ARGUMENT;
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
  if (!index_of(device, &idx)) return NVML_ERROR_INVALID_ARGUMENT;
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
// The PCIe link the profile records: the one real cards of the model most often
// run at. A simulated link does not retrain, so current and maximum agree.
static nvmlReturn_t pcie_link(nvmlDevice_t device, unsigned int* out, bool generation) {
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !out) return NVML_ERROR_INVALID_ARGUMENT;
  const unsigned v = generation ? d->pcie_gen : d->pcie_width;
  if (!v) return NVML_ERROR_NOT_SUPPORTED;
  *out = v;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* gen) {
  REQUIRE_INIT(); return pcie_link(device, gen, true);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* gen) {
  REQUIRE_INIT(); return pcie_link(device, gen, true);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* width) {
  REQUIRE_INIT(); return pcie_link(device, width, false);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int* width) {
  REQUIRE_INIT(); return pcie_link(device, width, false);
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
    return NVML_ERROR_INVALID_ARGUMENT;
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
    return NVML_ERROR_INVALID_ARGUMENT;
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
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t, unsigned long long*) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
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
    return NVML_ERROR_INVALID_ARGUMENT;
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
        break;
    }
  }
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
