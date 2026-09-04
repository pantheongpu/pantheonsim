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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "vgpu/telemetry.hpp"

namespace {

std::mutex g_mu;
bool g_initialized = false;
vgpu::telemetry::Shared g_snap{};

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
bool refresh() {
  return vgpu::telemetry::read_snapshot(&g_snap);
}

const vgpu::telemetry::DeviceSample* sample(nvmlDevice_t dev) {
  unsigned int idx;
  if (!index_of(dev, &idx)) return nullptr;
  return &g_snap.devices[idx];
}

nvmlReturn_t copy_string(const char* src, char* dst, unsigned int len) {
  if (!dst || len == 0) return NVML_ERROR_INVALID_ARGUMENT;
  std::snprintf(dst, len, "%s", src);
  return NVML_SUCCESS;
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- lifecycle ---- */

VGPU_EXPORT nvmlReturn_t nvmlInit_v2(void) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!refresh()) {
    // No live VirtualGPU: report "driver not loaded", exactly as NVML does on
    // a machine with no NVIDIA driver.
    return NVML_ERROR_DRIVER_NOT_LOADED;
  }
  g_initialized = true;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlInit(void) { return nvmlInit_v2(); }
VGPU_EXPORT nvmlReturn_t nvmlInitWithFlags(unsigned int) { return nvmlInit_v2(); }
VGPU_EXPORT nvmlReturn_t nvmlShutdown(void) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_initialized = false;
  return NVML_SUCCESS;
}

VGPU_EXPORT const char* nvmlErrorString(nvmlReturn_t result) {
  switch (result) {
    case NVML_SUCCESS: return "The operation was successful";
    case NVML_ERROR_UNINITIALIZED: return "Uninitialized";
    case NVML_ERROR_INVALID_ARGUMENT: return "Invalid Argument";
    case NVML_ERROR_NOT_SUPPORTED: return "Not Supported";
    case NVML_ERROR_NOT_FOUND: return "Not Found";
    case NVML_ERROR_INSUFFICIENT_SIZE: return "Insufficient Size";
    case NVML_ERROR_DRIVER_NOT_LOADED: return "Driver Not Loaded";
    default: return "Unknown Error";
  }
}

/* ---- system ---- */

VGPU_EXPORT nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length) {
  std::lock_guard<std::mutex> lock(g_mu);
  refresh();
  return copy_string(g_snap.driver_version, version, length);
}
VGPU_EXPORT nvmlReturn_t nvmlSystemGetNVMLVersion(char* version, unsigned int length) {
  // nvidia-smi validates this against the version it was built with and exits
  // if it disagrees, so it is overridable: set VGPU_NVML_VERSION to the string
  // `nvidia-smi --version` reports (as "NVML version", e.g. 595.71 ->
  // "13.595.71.01"). `vgpu smi` needs none of this.
  const char* v = std::getenv("VGPU_NVML_VERSION");
  return copy_string(v && v[0] ? v : "13.580.00.00", version, length);
}
VGPU_EXPORT nvmlReturn_t nvmlSystemGetCudaDriverVersion(int* version) {
  if (!version) return NVML_ERROR_INVALID_ARGUMENT;
  *version = 13000;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int* version) {
  return nvmlSystemGetCudaDriverVersion(version);
}

/* ---- enumeration and identity ---- */

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* count) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!count) return NVML_ERROR_INVALID_ARGUMENT;
  refresh();
  *count = g_snap.device_count;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCount(unsigned int* count) {
  return nvmlDeviceGetCount_v2(count);
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!device) return NVML_ERROR_INVALID_ARGUMENT;
  refresh();
  if (index >= g_snap.device_count) return NVML_ERROR_INVALID_ARGUMENT;
  *device = handle_for(index);
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
  return nvmlDeviceGetHandleByIndex_v2(index, device);
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!index) return NVML_ERROR_INVALID_ARGUMENT;
  return index_of(device, index) ? NVML_SUCCESS : NVML_ERROR_INVALID_ARGUMENT;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) {
  std::lock_guard<std::mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  return d ? copy_string(d->name, name, length) : NVML_ERROR_INVALID_ARGUMENT;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) {
  std::lock_guard<std::mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  return d ? copy_string(d->uuid, uuid, length) : NVML_ERROR_INVALID_ARGUMENT;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t, char*, unsigned int) {
  return NVML_ERROR_NOT_SUPPORTED;  // virtual devices have no serial number
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci) {
  std::lock_guard<std::mutex> lock(g_mu);
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
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t device, nvmlPciInfo_t* pci) {
  return nvmlDeviceGetPciInfo_v3(device, pci);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t* pci) {
  return nvmlDeviceGetPciInfo_v3(device, pci);
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major,
                                                            int* minor) {
  std::lock_guard<std::mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !major || !minor) return NVML_ERROR_INVALID_ARGUMENT;
  *major = d->cc_major;
  *minor = d->cc_minor;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* cores) {
  std::lock_guard<std::mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !cores) return NVML_ERROR_INVALID_ARGUMENT;
  *cores = d->multiprocessors;
  return NVML_SUCCESS;
}

/* ---- live measurements (memory + utilization are REAL) ---- */

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
  std::lock_guard<std::mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !memory) return NVML_ERROR_INVALID_ARGUMENT;
  memory->total = d->vram_total_bytes;
  memory->used = d->vram_used_bytes;
  memory->free = d->vram_total_bytes - d->vram_used_bytes;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) {
  std::lock_guard<std::mutex> lock(g_mu);
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
                                                       nvmlUtilization_t* utilization) {
  std::lock_guard<std::mutex> lock(g_mu);
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
                                                  unsigned int* temp) {
  std::lock_guard<std::mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !temp) return NVML_ERROR_INVALID_ARGUMENT;
  if (sensorType != NVML_TEMPERATURE_GPU) return NVML_ERROR_NOT_SUPPORTED;
  *temp = d->temperature_c;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device,
                                                           nvmlTemperatureThresholds_t which,
                                                           unsigned int* temp) {
  std::lock_guard<std::mutex> lock(g_mu);
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

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* milliwatts) {
  std::lock_guard<std::mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !milliwatts) return NVML_ERROR_INVALID_ARGUMENT;
  *milliwatts = d->power_mw;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit) {
  std::lock_guard<std::mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !limit) return NVML_ERROR_INVALID_ARGUMENT;
  *limit = d->power_limit_mw;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device,
                                                           unsigned int* limit) {
  return nvmlDeviceGetEnforcedPowerLimit(device, limit);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device,
                                                                  unsigned int* limit) {
  return nvmlDeviceGetEnforcedPowerLimit(device, limit);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device,
                                                                      unsigned int* minLimit,
                                                                      unsigned int* maxLimit) {
  std::lock_guard<std::mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !minLimit || !maxLimit) return NVML_ERROR_INVALID_ARGUMENT;
  *minLimit = d->power_limit_mw / 2;
  *maxLimit = d->power_limit_mw;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type,
                                                unsigned int* clock) {
  std::lock_guard<std::mutex> lock(g_mu);
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
                                                   unsigned int* clock) {
  std::lock_guard<std::mutex> lock(g_mu);
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

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed) {
  std::lock_guard<std::mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !speed) return NVML_ERROR_INVALID_ARGUMENT;
  *speed = d->fan_percent;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t* state) {
  std::lock_guard<std::mutex> lock(g_mu);
  refresh();
  const auto* d = sample(device);
  if (!d || !state) return NVML_ERROR_INVALID_ARGUMENT;
  *state = static_cast<nvmlPstates_t>(d->perf_state);
  return NVML_SUCCESS;
}

/* ---- modes and processes ---- */

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t, nvmlEnableState_t* mode) {
  if (!mode) return NVML_ERROR_INVALID_ARGUMENT;
  *mode = NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t, nvmlComputeMode_t* mode) {
  if (!mode) return NVML_ERROR_INVALID_ARGUMENT;
  *mode = NVML_COMPUTEMODE_DEFAULT;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t, nvmlEnableState_t* current,
                                              nvmlEnableState_t* pending) {
  // Virtual memory has no ECC to model; report the feature as absent rather
  // than claiming a clean ECC state that means nothing here.
  (void)current;
  (void)pending;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t, unsigned int*, unsigned int*) {
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t, nvmlEnableState_t* mode) {
  if (!mode) return NVML_ERROR_INVALID_ARGUMENT;
  *mode = NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t, nvmlEnableState_t* mode) {
  if (!mode) return NVML_ERROR_INVALID_ARGUMENT;
  *mode = NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device,
                                                                 unsigned int* infoCount,
                                                                 nvmlProcessInfo_t* infos) {
  std::lock_guard<std::mutex> lock(g_mu);
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
                                                                 nvmlProcessInfo_v2_t* infos) {
  std::lock_guard<std::mutex> lock(g_mu);
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
                                                                  nvmlProcessInfo_t*) {
  if (n) *n = 0;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t, unsigned int* n,
                                                                    nvmlProcessInfo_t*) {
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
