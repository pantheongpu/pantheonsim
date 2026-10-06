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
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t, unsigned long long*) {
  REQUIRE_INIT(); return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t, nvmlProcessUtilizationSample_t*,
                                                         unsigned int* count, unsigned long long) {
  REQUIRE_INIT();
  if (count) *count = 0;
  return NVML_ERROR_NOT_FOUND;   // the documented answer when there are no samples
}
// ---- Health and diagnostic fields -----------------------------------------
//
// The field ids a health tool asks nvmlDeviceGetFieldValues for -- ECC counts,
// retired pages and remapped rows, NVLink errors, the power limits and the
// time spent throttled. Each is answered from the same state the dedicated
// entry points below read, with NOT_SUPPORTED where the card of the profile
// has no such thing:
//
//   ECC fields              profile ecc: false (GeForce) answers NOT_SUPPORTED
//   retired pages           only a card that retires pages (profile: Turing
//                           and earlier with ECC), as telemetry.cpp decides
//   remapped rows           only a card that remaps rows (Ampere and later
//                           with ECC, and every HBM card)
//   NVLink                  only a profile that names its links (telemetry.nvlink)
//
// Ids are numbers, as above: they are ABI and older headers lack some names.
namespace {
using Dev = vgpu::telemetry::DeviceSample;

// What the card's driver offers as clock-event reasons: bits 0-8 of NVML's
// mask (idle, application clocks, SW power cap, HW slowdown, sync boost, SW
// thermal, HW thermal, HW power brake, display clock). An RTX 3060's, which is
// what nvidia-smi's clocks_event_reasons.supported answers too. Board-limit
// and reliability (bits 9 and 10) are newer than any measured card.
constexpr unsigned long long kSupportedClockEvents = 0x1FF;

// The reasons active now: GPU idle whenever the device is, as a real idle card
// reports it, plus whatever `vgpu fault throttle` made active.
unsigned long long current_clock_events(const Dev& d) {
  return (d.utilization_gpu == 0 ? vgpu::ras::kGpuIdle : 0) | d.clock_event_reasons;
}

// Nanoseconds the clocks were held down by any of these reasons, summed over
// every window `vgpu fault throttle` has opened. Zero until something is
// injected: nothing throttles a simulated clock on its own.
unsigned long long throttle_ns(const Dev& d, unsigned long long reasons) {
  unsigned long long us = 0;
  for (unsigned long long bit = 1; bit <= reasons; bit <<= 1)
    if (reasons & bit) us += vgpu::ras::throttle_time_us(d.uuid, bit);
  return us * 1000;
}
constexpr unsigned long long kPowerReasons =
    vgpu::ras::kSwPowerCap | vgpu::ras::kHwPowerBrakeSlowdown;
constexpr unsigned long long kThermalReasons =
    vgpu::ras::kSwThermalSlowdown | vgpu::ras::kHwThermalSlowdown;

// The minimum a power limit can be set to. The profile carries only the
// limit, so this is the same half of it nvmlDeviceGetPowerManagementLimitConstraints
// has always answered -- not a measured figure.
unsigned int min_power_limit_mw(const Dev& d) { return d.power_limit_mw / 2; }

// NVML's ECC field ids: 0 volatile or 1 aggregate counts, corrected (single
// bit) or uncorrected (double bit), by location (-1 is the total).
struct EccField {
  unsigned id;
  int counter, severity;
  int location;   // a vgpu::ras::Location, or -1 for the total
};
const EccField kEccFields[] = {
    {3, 0, 0, -1}, {4, 0, 1, -1}, {5, 1, 0, -1}, {6, 1, 1, -1},
    {7, 0, 0, static_cast<int>(vgpu::ras::Location::L1Cache)},
    {8, 0, 1, static_cast<int>(vgpu::ras::Location::L1Cache)},
    {9, 0, 0, static_cast<int>(vgpu::ras::Location::L2Cache)},
    {10, 0, 1, static_cast<int>(vgpu::ras::Location::L2Cache)},
    {11, 0, 0, static_cast<int>(vgpu::ras::Location::DeviceMemory)},
    {12, 0, 1, static_cast<int>(vgpu::ras::Location::DeviceMemory)},
    {13, 0, 0, static_cast<int>(vgpu::ras::Location::RegisterFile)},
    {14, 0, 1, static_cast<int>(vgpu::ras::Location::RegisterFile)},
    {15, 0, 0, static_cast<int>(vgpu::ras::Location::TextureMemory)},
    {16, 0, 1, static_cast<int>(vgpu::ras::Location::TextureMemory)},
    {17, 0, 1, static_cast<int>(vgpu::ras::Location::Cbu)},
    {18, 1, 0, static_cast<int>(vgpu::ras::Location::L1Cache)},
    {19, 1, 1, static_cast<int>(vgpu::ras::Location::L1Cache)},
    {20, 1, 0, static_cast<int>(vgpu::ras::Location::L2Cache)},
    {21, 1, 1, static_cast<int>(vgpu::ras::Location::L2Cache)},
    {22, 1, 0, static_cast<int>(vgpu::ras::Location::DeviceMemory)},
    {23, 1, 1, static_cast<int>(vgpu::ras::Location::DeviceMemory)},
    {24, 1, 0, static_cast<int>(vgpu::ras::Location::RegisterFile)},
    {25, 1, 1, static_cast<int>(vgpu::ras::Location::RegisterFile)},
    {26, 1, 0, static_cast<int>(vgpu::ras::Location::TextureMemory)},
    {27, 1, 1, static_cast<int>(vgpu::ras::Location::TextureMemory)},
    {28, 1, 1, static_cast<int>(vgpu::ras::Location::Cbu)},
};

// An NVLink error-counter field: 0 flow-control CRC, 1 data CRC, 2 replay,
// 3 recovery. Lanes 0-5 and 6-11 are two id ranges; the total has an id of
// its own. Returns the link, -1 for the total, or -2 when the id is not one.
int nvlink_error_field(unsigned id, int* kind) {
  static const unsigned lo[4] = {32, 39, 46, 53}, hi[4] = {96, 102, 108, 114}, total[4] = {38, 45, 52, 59};
  for (int k = 0; k < 4; ++k) {
    *kind = k;
    if (id == total[k]) return -1;
    if (id >= lo[k] && id < lo[k] + 6) return static_cast<int>(id - lo[k]);
    if (id >= hi[k] && id < hi[k] + 6) return 6 + static_cast<int>(id - hi[k]);
  }
  return -2;
}

void health_field(const Dev& d, const vgpu::ras::State& rs, nvmlFieldValue_t* v) {
  const auto ull = [&](unsigned long long x) {
    v->valueType = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
    v->value.ullVal = x;
    v->nvmlReturn = NVML_SUCCESS;
  };
  const auto ui = [&](unsigned x) {
    v->valueType = NVML_VALUE_TYPE_UNSIGNED_INT;
    v->value.uiVal = x;
    v->nvmlReturn = NVML_SUCCESS;
  };
  const unsigned id = v->fieldId;
  const vgpu::ras::Counters& life = rs.lifetime;
  // ECC mode and counts.
  if (id == 1 || id == 2) {
    if (d.ecc_enabled) ui(1);
    return;
  }
  for (const EccField& f : kEccFields) {
    if (f.id != id) continue;
    if (!d.ecc_enabled) return;
    const vgpu::ras::Counters& c = f.counter ? rs.lifetime : rs.since_load;
    if (f.location < 0) {
      ull(c.ecc_total(f.severity ? vgpu::ras::Severity::Uncorrected : vgpu::ras::Severity::Corrected));
    } else {
      ull(c.ecc[f.severity][f.location]);
    }
    return;
  }
  // Page retirement: counts and what is pending, on a card that retires pages.
  // Only a double-bit error retires a page in this model (ras.cpp), so none is
  // ever pending for a single-bit one.
  if (id == 29 || id == 30 || id == 31 || id == 92 || id == 93) {
    if (d.memory_retirement != 1) return;
    ui(id == 29 ? life.retired_sbe : id == 30 ? life.retired_dbe
       : id == 92 ? 0u : life.retired_pending ? 1u : 0u);
    return;
  }
  // Row remapping, on a card that remaps rows.
  if (id >= 142 && id <= 145) {
    if (d.memory_retirement != 2) return;
    ui(id == 142 ? life.rows_correctable : id == 143 ? life.rows_uncorrectable
       : id == 144 ? (life.rows_pending ? 1u : 0u) : (life.rows_failure ? 1u : 0u));
    return;
  }
  // Time spent below the application clocks, by policy, in nanoseconds. Only
  // power and thermal are modelled; the other policies need state a simulated
  // card does not have.
  if (id == 74) { ull(throttle_ns(d, kPowerReasons)); return; }
  if (id == 75) { ull(throttle_ns(d, kThermalReasons)); return; }
  // Power, in milliwatts. The average over a second is documented as Ampere
  // (except GA100) and newer only; the rest are on every card.
  switch (id) {
    case 185:
      if (d.cc_major >= 8 && !(d.cc_major == 8 && d.cc_minor == 0)) ui(d.power_mw);
      return;
    case 186: ui(d.power_mw); return;
    case 187: ui(min_power_limit_mw(d)); return;
    case 188: case 189: case 190: case 192: ui(d.power_limit_mw); return;
    default: break;
  }
  // NVLink: how many links the card has, and their error counts, all zero
  // because nothing injects NVLink errors yet.
  if (id == 91) {
    if (d.nvlink_count) ui(d.nvlink_count);
    return;
  }
  int kind = 0;
  const int link = nvlink_error_field(id, &kind);
  if (link != -2 && d.nvlink_count && link < static_cast<int>(d.nvlink_count)) ull(0);
}
}  // namespace

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
  const vgpu::ras::State rs = d ? vgpu::ras::read(d->uuid) : vgpu::ras::State{};
  const vgpu::ras::Counters& pcie = rs.since_load;
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
        if (d) health_field(*d, rs, &v);
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

/* ---- reliability and health: what DCGM-style diagnostics ask ----
 *
 * Memory error management, clock-event reasons and violation times, PCIe
 * replays, NVLink, accounting. Every answer follows the device's profile and
 * the machine's reliability state (vgpu/ras.hpp), and a card that has no such
 * thing answers NVML_ERROR_NOT_SUPPORTED, as a real one does:
 *
 *   T4 (Turing, ECC)        retires pages; no row remapping
 *   A10/L4/L40S, A100, H100 remap rows; no page retirement
 *   RTX 3060, 3080 Ti, 5090 no ECC: every ECC, retirement and remap query
 *                           is NOT_SUPPORTED
 *   NVLink                  only the SXM profiles that name their links
 *
 * Counts come only from injection (`vgpu fault`); nothing here faults on its
 * own. Public references: the NVML API reference, nvml.h from the CUDA
 * toolkit, and NVIDIA's GPU memory error management guide.
 */

namespace {
// The device a health query is about, under the lock, or the error to return.
#define HEALTH_DEVICE(d_)                                    \
  std::lock_guard<std::recursive_mutex> lock(g_mu);          \
  refresh();                                                 \
  const auto* d_ = sample(device);                           \
  if (!d_) return bad(device)

// Retired-page addresses are not kept -- the state counts pages, not where
// they were. A tool needs distinct, stable addresses to list, so these are
// derived from the device UUID and the page's index: aligned to 64 KiB and
// inside the framebuffer, and the same on every query. They are placeholders,
// not hardware addresses.
unsigned long long synthetic_page(const Dev& d, unsigned cause, unsigned index) {
  uint64_t h = 1469598103934665603ull;
  for (const char* c = d.uuid; *c; ++c) h = (h ^ static_cast<unsigned char>(*c)) * 1099511628211ull;
  h = (h ^ (cause * 2654435761ull + index)) * 1099511628211ull;
  h ^= h >> 29;
  const uint64_t pages = d.vram_total_bytes >> 16;
  return pages ? (h % pages) << 16 : 0;
}

// Accounting mode is the driver's, not a process's: it outlives nvmlShutdown,
// as it outlives the program that set it. Kept for the life of this process,
// since the simulator has no driver to hold it; a second process sees it off.
bool g_accounting[vgpu::telemetry::kMaxDevices] = {};

long long now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch()).count();
}
}  // namespace

/* -- ECC -- */

// NVML's older per-location form. Single-bit is corrected, double-bit is
// uncorrected. Deprecated in favour of nvmlDeviceGetMemoryErrorCounter, which
// this agrees with.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device,
                                                        nvmlMemoryErrorType_t error_type,
                                                        nvmlEccCounterType_t counter_type,
                                                        nvmlEccErrorCounts_t* counts) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (!counts || static_cast<int>(error_type) > 1 || static_cast<int>(counter_type) > 1)
    return NVML_ERROR_INVALID_ARGUMENT;
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  const vgpu::ras::State st = vgpu::ras::read(d->uuid);
  const vgpu::ras::Counters& c = ecc_counts(st, static_cast<int>(counter_type));
  const auto sev = error_type == 0 ? 0u : 1u;
  using L = vgpu::ras::Location;
  counts->l1Cache = c.ecc[sev][static_cast<int>(L::L1Cache)];
  counts->l2Cache = c.ecc[sev][static_cast<int>(L::L2Cache)];
  counts->deviceMemory = c.ecc[sev][static_cast<int>(L::DeviceMemory)];
  counts->registerFile = c.ecc[sev][static_cast<int>(L::RegisterFile)];
  return NVML_SUCCESS;
}

// What the card boots with: on for every profile that has ECC.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t* mode) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (!mode) return NVML_ERROR_INVALID_ARGUMENT;
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  *mode = NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}

// Zeroes the counts, as `nvidia-smi -p` does: volatile (0) or aggregate (1).
// Retired pages and remapped rows stay, being memory taken out of service.
VGPU_EXPORT nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device,
                                                       nvmlEccCounterType_t counter_type) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (static_cast<int>(counter_type) > 1) return NVML_ERROR_INVALID_ARGUMENT;
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  try {
    if (counter_type == NVML_VOLATILE_ECC) vgpu::ras::reset_volatile(d->uuid, /*driver_reload=*/false);
    else vgpu::ras::reset_aggregate(d->uuid);
  } catch (const std::exception&) {
    return NVML_ERROR_UNKNOWN;
  }
  return NVML_SUCCESS;
}

// Changing the mode takes a reset, and the simulator keeps no pending mode to
// apply one to, so only what is already true is accepted: asking for ECC on a
// card that has it on is a no-op. Turning it off, or on for a card without it,
// is NOT_SUPPORTED rather than a change nothing would honour.
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (ecc != NVML_FEATURE_ENABLED && ecc != NVML_FEATURE_DISABLED) return NVML_ERROR_INVALID_ARGUMENT;
  if (!d->ecc_enabled || ecc != NVML_FEATURE_ENABLED) return NVML_ERROR_NOT_SUPPORTED;
  return NVML_SUCCESS;
}

/* -- retired pages and row remapping -- */

// Pages retired for a cause: 0 multiple single-bit errors, 1 a double-bit
// error. Only a card that retires pages (ECC, Turing and earlier) has any to
// report. The count includes pages pending retirement, as nvidia-smi's
// retired_pages.* fields do, and each address is a placeholder (see above).
static nvmlReturn_t retired_pages(nvmlDevice_t device, nvmlPageRetirementCause_t cause,
                                  unsigned int* page_count, unsigned long long* addresses,
                                  unsigned long long* timestamps) {
  HEALTH_DEVICE(d);
  if (!page_count || static_cast<int>(cause) < 0 || static_cast<int>(cause) > 1)
    return NVML_ERROR_INVALID_ARGUMENT;
  if (d->memory_retirement != 1) return NVML_ERROR_NOT_SUPPORTED;
  const vgpu::ras::State st = vgpu::ras::read(d->uuid);
  const unsigned int n = static_cast<unsigned int>(cause == 0 ? st.lifetime.retired_sbe
                                                              : st.lifetime.retired_dbe);
  if (*page_count < n || (n && !addresses)) {
    *page_count = n;
    return NVML_ERROR_INSUFFICIENT_SIZE;
  }
  for (unsigned int i = 0; i < n; ++i) {
    addresses[i] = synthetic_page(*d, static_cast<unsigned>(cause), i);
    if (timestamps) timestamps[i] = 0;   // when a page was retired is not recorded
  }
  *page_count = n;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause,
                                                   unsigned int* page_count,
                                                   unsigned long long* addresses) {
  REQUIRE_INIT();
  return retired_pages(device, cause, page_count, addresses, nullptr);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device,
                                                      nvmlPageRetirementCause_t cause,
                                                      unsigned int* page_count,
                                                      unsigned long long* addresses,
                                                      unsigned long long* timestamps) {
  REQUIRE_INIT();
  return retired_pages(device, cause, page_count, addresses, timestamps);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device,
                                                                nvmlEnableState_t* pending) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (!pending) return NVML_ERROR_INVALID_ARGUMENT;
  if (d->memory_retirement != 1) return NVML_ERROR_NOT_SUPPORTED;
  *pending = vgpu::ras::read(d->uuid).lifetime.retired_pending ? NVML_FEATURE_ENABLED
                                                                : NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}

// Rows remapped for correctable and uncorrectable errors, whether a remap is
// pending (it takes effect at the next driver load or reset), and whether one
// failed. Only a card that remaps rows has them.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int* corrected,
                                                   unsigned int* uncorrected, unsigned int* pending,
                                                   unsigned int* failure) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (!corrected || !uncorrected || !pending || !failure) return NVML_ERROR_INVALID_ARGUMENT;
  if (d->memory_retirement != 2) return NVML_ERROR_NOT_SUPPORTED;
  const vgpu::ras::Counters life = vgpu::ras::read(d->uuid).lifetime;
  *corrected = static_cast<unsigned int>(life.rows_correctable);
  *uncorrected = static_cast<unsigned int>(life.rows_uncorrectable);
  *pending = life.rows_pending ? 1 : 0;
  *failure = life.rows_failure ? 1 : 0;
  return NVML_SUCCESS;
}
// Banks by how many spare rows each has left. That needs the card's bank
// count and per-bank spare rows, which no profile carries and NVIDIA's public
// documentation does not give per model, so a card that remaps rows says
// NOT_SUPPORTED here -- as nvidia-smi's remapped_rows.histogram.* read [N/A] --
// rather than a count of banks made up for it.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device,
                                                           nvmlRowRemapperHistogramValues_t* values) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (!values) return NVML_ERROR_INVALID_ARGUMENT;
  (void)d;
  return NVML_ERROR_NOT_SUPPORTED;
}

/* -- clock-event reasons, performance state, power and thermal limits -- */

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device,
                                                                     unsigned long long* reasons) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (!reasons) return NVML_ERROR_INVALID_ARGUMENT;
  (void)d;
  *reasons = kSupportedClockEvents;
  return NVML_SUCCESS;
}
// Toolkit 13 renamed throttle reasons to clock-event reasons and kept both.
#ifdef nvmlClocksEventReasonGpuIdle
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSupportedClocksEventReasons(nvmlDevice_t device,
                                                                  unsigned long long* reasons) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (!reasons) return NVML_ERROR_INVALID_ARGUMENT;
  (void)d;
  *reasons = kSupportedClockEvents;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t device,
                                                                unsigned long long* reasons) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (!reasons) return NVML_ERROR_INVALID_ARGUMENT;
  *reasons = current_clock_events(*d);
  return NVML_SUCCESS;
}
#endif

// How long the clocks were held below the application clocks by a policy.
// referenceTime is a CPU timestamp in microseconds and violationTime is in
// nanoseconds, as nvml.h says. Power and thermal are modelled, from the
// windows `vgpu fault throttle` opens; the other policies (sync boost, board
// limit, low utilization, reliability, the totals) need state a simulated
// card does not have and are NOT_SUPPORTED.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device,
                                                      nvmlPerfPolicyType_t policy,
                                                      nvmlViolationTime_t* violation) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (!violation) return NVML_ERROR_INVALID_ARGUMENT;
  if (static_cast<int>(policy) < 0 || static_cast<int>(policy) >= static_cast<int>(NVML_PERF_POLICY_COUNT))
    return NVML_ERROR_INVALID_ARGUMENT;
  if (policy != NVML_PERF_POLICY_POWER && policy != NVML_PERF_POLICY_THERMAL)
    return NVML_ERROR_NOT_SUPPORTED;
  violation->referenceTime = static_cast<unsigned long long>(now_us());
  violation->violationTime =
      throttle_ns(*d, policy == NVML_PERF_POLICY_POWER ? kPowerReasons : kThermalReasons);
  return NVML_SUCCESS;
}

// Deprecated spellings of the performance state and of power management,
// which a card with power limits always has.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t* state) {
  REQUIRE_INIT();
  return nvmlDeviceGetPerformanceState(device, state);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device,
                                                          nvmlEnableState_t* mode) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (!mode) return NVML_ERROR_INVALID_ARGUMENT;
  if (!d->power_limit_mw) return NVML_ERROR_NOT_SUPPORTED;
  *mode = NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}

/* -- PCIe -- */

// Replays since the driver loaded -- the same count as the replay field and
// nvidia-smi's "Replays Since Reset". Zero until `vgpu fault inject --pcie replay`.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int* value) {
  REQUIRE_INIT();
  HEALTH_DEVICE(d);
  if (!value) return NVML_ERROR_INVALID_ARGUMENT;
  *value = static_cast<unsigned int>(
      vgpu::ras::read(d->uuid).since_load.pcie[static_cast<uint32_t>(vgpu::ras::Pcie::Replay)]);
  return NVML_SUCCESS;
}

/* -- NVLink --
 *
 * Present on the SXM profiles that name their links (telemetry.nvlink, from
 * the data sheet). Every other card answers NOT_SUPPORTED, as a T4 or a PCIe
 * card without a bridge does. Assumptions, since no profile measures them:
 * all links are up (a baseboard wires them), the capabilities are those of an
 * x86 host (peer access and peer atomics, no system-memory access), and error
 * counters are zero because nothing injects NVLink errors yet. The remote end
 * of a link is not modelled, so its type and PCI address are NOT_SUPPORTED
 * ("nvidia-smi topo -m" shows no NVLink either, for the same reason).
 */
namespace {
// The device, checked for NVLink and for a link in range; the error to return
// otherwise, in *rc.
const Dev* nvlink_device(nvmlDevice_t device, unsigned int link, nvmlReturn_t* rc) {
  const auto* d = sample(device);
  if (!d) { *rc = bad(device); return nullptr; }
  if (!d->nvlink_count) { *rc = NVML_ERROR_NOT_SUPPORTED; return nullptr; }
  if (link >= d->nvlink_count) { *rc = NVML_ERROR_INVALID_ARGUMENT; return nullptr; }
  return d;
}
}  // namespace

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link,
                                                  nvmlEnableState_t* active) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  nvmlReturn_t rc = NVML_SUCCESS;
  if (!active) return NVML_ERROR_INVALID_ARGUMENT;
  if (!nvlink_device(device, link, &rc)) return rc;
  *active = NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link,
                                                    unsigned int* version) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  nvmlReturn_t rc = NVML_SUCCESS;
  if (!version) return NVML_ERROR_INVALID_ARGUMENT;
  const auto* d = nvlink_device(device, link, &rc);
  if (!d) return rc;
  *version = d->nvlink_version;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
                                                       nvmlNvLinkCapability_t capability,
                                                       unsigned int* result) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  nvmlReturn_t rc = NVML_SUCCESS;
  if (!result) return NVML_ERROR_INVALID_ARGUMENT;
  if (!nvlink_device(device, link, &rc)) return rc;
  switch (capability) {
    case NVML_NVLINK_CAP_P2P_SUPPORTED: *result = 1; break;
    case NVML_NVLINK_CAP_SYSMEM_ACCESS: *result = 0; break;
    case NVML_NVLINK_CAP_P2P_ATOMICS: *result = 1; break;
    case NVML_NVLINK_CAP_SYSMEM_ATOMICS: *result = 0; break;
    case NVML_NVLINK_CAP_SLI_BRIDGE: *result = 0; break;
    case NVML_NVLINK_CAP_VALID: *result = 1; break;
    default: return NVML_ERROR_INVALID_ARGUMENT;
  }
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link,
                                                         nvmlNvLinkErrorCounter_t counter,
                                                         unsigned long long* value) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  nvmlReturn_t rc = NVML_SUCCESS;
  if (!value || static_cast<int>(counter) < 0 || static_cast<int>(counter) >= static_cast<int>(NVML_NVLINK_ERROR_COUNT))
    return NVML_ERROR_INVALID_ARGUMENT;
  if (!nvlink_device(device, link, &rc)) return rc;
  *value = 0;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  nvmlReturn_t rc = NVML_SUCCESS;
  if (!nvlink_device(device, link, &rc)) return rc;
  return NVML_SUCCESS;   // they are zero already
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link,
                                                             nvmlIntNvLinkDeviceType_t* type) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  nvmlReturn_t rc = NVML_SUCCESS;
  if (!type) return NVML_ERROR_INVALID_ARGUMENT;
  if (!nvlink_device(device, link, &rc)) return rc;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link,
                                                             nvmlPciInfo_t* pci) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  nvmlReturn_t rc = NVML_SUCCESS;
  if (!pci) return NVML_ERROR_INVALID_ARGUMENT;
  if (!nvlink_device(device, link, &rc)) return rc;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link,
                                                          nvmlPciInfo_t* pci) {
  REQUIRE_INIT();
  return nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, pci);
}

/* -- accounting and processes --
 *
 * Accounting keeps statistics for processes that have used the GPU. The
 * simulator knows the processes using a device now and the memory each holds,
 * and keeps none after they exit, so it reports running processes only, with
 * their memory and "not available" for the utilization it does not measure
 * per process.
 */
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t* mode) {
  REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !mode) return bad(device);
  *mode = g_accounting[idx] ? NVML_FEATURE_ENABLED : NVML_FEATURE_DISABLED;   // off by default
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) {
  REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx)) return bad(device);
  if (mode != NVML_FEATURE_ENABLED && mode != NVML_FEATURE_DISABLED) return NVML_ERROR_INVALID_ARGUMENT;
  g_accounting[idx] = mode == NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int* size) {
  REQUIRE_INIT();
  unsigned int idx;
  if (!index_of(device, &idx) || !size) return bad(device);
  *size = 4000;   // the driver's default, which no query of the simulator can outgrow
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int* count,
                                                     unsigned int* pids) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  unsigned int idx;
  const auto* d = sample(device);
  if (!d || !count || !index_of(device, &idx)) return bad(device);
  if (!g_accounting[idx]) return NVML_ERROR_NOT_SUPPORTED;   // "or accounting mode is disabled"
  const unsigned int have = d->proc_count;
  if (!pids || *count < have) {
    *count = have;
    return have ? NVML_ERROR_INSUFFICIENT_SIZE : NVML_SUCCESS;
  }
  for (unsigned int i = 0; i < have; ++i) pids[i] = d->procs[i].pid;
  *count = have;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid,
                                                      nvmlAccountingStats_t* stats) {
  REQUIRE_INIT();
  std::lock_guard<std::recursive_mutex> lock(g_mu);
  refresh();
  unsigned int idx;
  const auto* d = sample(device);
  if (!d || !stats || !index_of(device, &idx)) return bad(device);
  if (!g_accounting[idx]) return NVML_ERROR_NOT_SUPPORTED;
  for (unsigned int i = 0; i < d->proc_count; ++i) {
    if (d->procs[i].pid != pid) continue;
    std::memset(stats, 0, sizeof *stats);
    stats->gpuUtilization = static_cast<unsigned int>(NVML_VALUE_NOT_AVAILABLE);
    stats->memoryUtilization = static_cast<unsigned int>(NVML_VALUE_NOT_AVAILABLE);
    stats->maxMemoryUsage = d->procs[i].used_bytes;   // what it holds now, not its peak
    stats->isRunning = 1;
    return NVML_SUCCESS;
  }
  return NVML_ERROR_NOT_FOUND;
}
#undef HEALTH_DEVICE

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
