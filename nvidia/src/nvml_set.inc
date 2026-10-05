// NVML: setters, clearers and the rest that write. See nvml_common.inc.
//
// The real driver demands root for every one of these. Here a setter takes effect
// when setters_allowed() says the process is root (VGPU_NVML_ROOT overrides, for
// tests) and is NVML_ERROR_NO_PERMISSION otherwise. The order of the checks is
// the one nvml.h's return-code lists suggest: the handle, the arguments
// (INVALID_ARGUMENT), whether the GPU has the feature at all (NOT_SUPPORTED),
// then permission, then the change. That order is a choice: nvml.h documents the
// codes, not which wins, and no setter was called on the card (nothing here was
// measured on hardware -- the card's getters decide which features a GeForce has).
//
// What a setter changes is kept in the device's settings overlay (nvml_common.inc)
// and read back by the getters: persistence mode, compute mode, power limit, clock
// locks and offsets, fan policy and speed, accounting mode, the pending ECC mode.
// Settings the real driver forgets at a driver reload are volatile (they live as
// long as the session); the pending ECC mode, which the inforom keeps, persists.
//
// Features the card lacks (application clocks, auto boost, API restriction, ECC,
// the driver model, the GPU operation mode) are NOT_SUPPORTED here too, and so is
// everything that needs NVLink, a fabric, or a hostname.

namespace {

// The clock-limit symbols nvmlDeviceSetGpuLockedClocks takes in place of MHz.
bool clock_symbolic(unsigned v) { return v >= 0xffffff00u; }

}  // namespace

/* ---- modes ---- */

VGPU_EXPORT nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode) {
  NVML_DEV(device);
  const int m = static_cast<int>(mode);
  if (m != 0 && m != 1) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_SETTER_PERMISSION();
  overlay_set(d->uuid, Life::Volatile, "persistence", m ? "1" : "0");
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode) {
  NVML_DEV(device);
  const int m = static_cast<int>(mode);
  if (m < 0 || m >= NVML_COMPUTEMODE_COUNT) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_SETTER_PERMISSION();
  overlay_set(d->uuid, Life::Volatile, "compute_mode", std::to_string(m));
  return NVML_SUCCESS;
}
// The ECC mode takes effect after the next reboot or GPU reset, neither of which
// is simulated: a change shows as the pending mode (nvmlDeviceGetEccMode), the
// current one stays.
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) {
  NVML_DEV(device);
  const int e = static_cast<int>(ecc);
  if (e != 0 && e != 1) return NVML_ERROR_INVALID_ARGUMENT;
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  NVML_SETTER_PERMISSION();
  overlay_set(d->uuid, Life::Persistent, "ecc_pending", e ? "1" : "0");
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) {
  NVML_DEV(device);
  const int m = static_cast<int>(mode);
  if (m != 0 && m != 1) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_SETTER_PERMISSION();
  const bool was = accounting_on(*d);
  overlay_set(d->uuid, Life::Volatile, "accounting", m ? "1" : "0");
  if (was && !m) accounting_clear(*d, false);   // disabling clears every record
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device) {
  NVML_DEV(device);
  NVML_SETTER_PERMISSION();
  if (accounting_on(*d)) accounting_clear(*d, true);   // the finished processes only
  return NVML_SUCCESS;
}
// The deprecated families the card does not have.
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType,
                                                     nvmlEnableState_t isRestricted) {
  NVML_DEV(device);
  const int t = static_cast<int>(apiType), r = static_cast<int>(isRestricted);
  if (t < 0 || t > 1 || r < 0 || r > 1) return NVML_ERROR_INVALID_ARGUMENT;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz,
                                                         unsigned int graphicsClockMHz) {
  NVML_DEV(device);
  (void)memClockMHz;
  (void)graphicsClockMHz;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceResetApplicationsClocks(nvmlDevice_t device) {
  NVML_DEV(device);
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled) {
  NVML_DEV(device);
  const int e = static_cast<int>(enabled);
  if (e != 0 && e != 1) return NVML_ERROR_INVALID_ARGUMENT;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled,
                                                                      unsigned int flags) {
  NVML_DEV(device);
  const int e = static_cast<int>(enabled);
  if (e != 0 && e != 1) return NVML_ERROR_INVALID_ARGUMENT;
  (void)flags;
  return NVML_ERROR_NOT_SUPPORTED;
}
// Windows-only, and the GPU operation mode is a Kepler Tesla feature.
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel,
                                                  unsigned int flags) {
  NVML_DEV(device);
  const int m = static_cast<int>(driverModel);
  if (m < 0 || m > 1) return NVML_ERROR_INVALID_ARGUMENT;
  (void)flags;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode) {
  NVML_DEV(device);
  const int m = static_cast<int>(mode);
  if (m < 0 || m > 2) return NVML_ERROR_INVALID_ARGUMENT;
  return NVML_ERROR_NOT_SUPPORTED;
}
#if defined(VGPU_NVML_12_8)
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetDramEncryptionMode(nvmlDevice_t device, const nvmlDramEncryptionInfo_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  NVML_VERSION(info, nvmlDramEncryptionInfo_v1);
  return NVML_ERROR_NOT_SUPPORTED;
}
#endif
#if defined(VGPU_NVML_13_0)
// A hostname is written by a fabric manager, which a simulated machine lacks.
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetHostname_v1(nvmlDevice_t device, nvmlHostname_v1_t* hostname) {
  NVML_DEV(device);
  NVML_ARG(hostname);
  return NVML_ERROR_NOT_SUPPORTED;
}
// PowerMizer (a GeForce driver feature the card has): the mode must be one the
// device supports (mask 0x7: adaptive, prefer-maximum-performance, auto).
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetPowerMizerMode_v1(nvmlDevice_t device, nvmlDevicePowerMizerModes_v1_t* mode) {
  NVML_DEV(device);
  NVML_ARG(mode);
  if (!consumer(*d)) return NVML_ERROR_NOT_SUPPORTED;
  if (mode->mode > 31 || !((0x7u >> mode->mode) & 1u)) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_SETTER_PERMISSION();
  overlay_set(d->uuid, Life::Volatile, "powermizer_mode", std::to_string(mode->mode));
  return NVML_SUCCESS;
}
#endif
#if defined(VGPU_NVML_13_2)
// Read-only user shared data: a kernel-driver feature the simulated machine does not have.
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetRusdSettings_v1(nvmlDevice_t device, nvmlRusdSettings_v1_t* settings) {
  NVML_DEV(device);
  NVML_ARG(settings);
  NVML_VERSION(settings, nvmlRusdSettings_v1);
  return NVML_ERROR_NOT_SUPPORTED;
}
#endif

/* ---- power ---- */

namespace {
// The range nvmlDeviceGetPowerManagementLimitConstraints reports, which a limit
// must fall in.
bool power_limit_valid(const vgpu::telemetry::DeviceSample& d, unsigned mw) {
  return mw >= d.power_limit_mw / 2 && mw <= d.power_limit_mw;
}
}  // namespace
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int limit) {
  NVML_DEV(device);
  if (!power_limit_valid(*d, limit)) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_SETTER_PERMISSION();
  overlay_set(d->uuid, Life::Volatile, "power_limit_mw", std::to_string(limit));
  return NVML_SUCCESS;
}
#if defined(VGPU_NVML_12_4)
// The scope is the GPU (0); a total-module limit belongs to a superchip.
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetPowerManagementLimit_v2(nvmlDevice_t device, nvmlPowerValue_v2_t* powerValue) {
  NVML_DEV(device);
  NVML_ARG(powerValue);
  NVML_VERSION(powerValue, nvmlPowerValue_v2);
  if (powerValue->powerScope != 0 || !power_limit_valid(*d, powerValue->powerValueMw)) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_SETTER_PERMISSION();
  overlay_set(d->uuid, Life::Volatile, "power_limit_mw", std::to_string(powerValue->powerValueMw));
  return NVML_SUCCESS;
}
#endif
// The only settable threshold is the acoustic one, which this model does not
// report; the other types are fixed.
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device,
                                                           nvmlTemperatureThresholds_t thresholdType, int* temp) {
  NVML_DEV(device);
  NVML_ARG(temp);
  const int t = static_cast<int>(thresholdType);
  if (t < 0 || t >= NVML_TEMPERATURE_THRESHOLD_COUNT) return NVML_ERROR_INVALID_ARGUMENT;
  return NVML_ERROR_NOT_SUPPORTED;
}

/* ---- clocks ---- */

namespace {
// Locked clocks need the profile's clock tables; a lock is a range, both ends
// numbers in the card's range or both "unlimited" (a reset). The ends the table
// does not bound are invalid; "TDP" locks are not modelled.
nvmlReturn_t lock_clocks(nvmlDevice_t device, bool memory, unsigned lo, unsigned hi, bool reset_only) {
  NVML_DEV(device);
  const vgpu::NvmlClass& n = nvml_facts(*d);
  const char* key = memory ? "mem_locked_clocks" : "gpu_locked_clocks";
  if (n.memory_clocks_mhz.empty() || !n.graphics_floor_mhz) return NVML_ERROR_NOT_SUPPORTED;
  if (memory && d->cc_major < 8) return NVML_ERROR_NOT_SUPPORTED;   // Ampere and newer
  bool reset = reset_only;
  if (!reset_only) {
    if (clock_symbolic(lo) != clock_symbolic(hi)) return NVML_ERROR_INVALID_ARGUMENT;   // mixed
    if (clock_symbolic(lo)) {
      if (lo == NVML_CLOCK_LIMIT_ID_UNLIMITED && hi == NVML_CLOCK_LIMIT_ID_UNLIMITED) reset = true;
      else if (lo == NVML_CLOCK_LIMIT_ID_TDP || hi == NVML_CLOCK_LIMIT_ID_TDP) return NVML_ERROR_NOT_SUPPORTED;
      else return NVML_ERROR_INVALID_ARGUMENT;
    } else {
      const unsigned floor = memory ? *std::min_element(n.memory_clocks_mhz.begin(), n.memory_clocks_mhz.end())
                                    : n.graphics_floor_mhz;
      const unsigned top = memory ? *std::max_element(n.memory_clocks_mhz.begin(), n.memory_clocks_mhz.end())
                                  : d->sm_clock_max_mhz;
      if (lo > hi || lo < floor || hi > top) return NVML_ERROR_INVALID_ARGUMENT;
    }
  }
  NVML_SETTER_PERMISSION();
  overlay_set(d->uuid, Life::Volatile, key, reset ? "" : std::to_string(lo) + "," + std::to_string(hi));
  return NVML_SUCCESS;
}
}  // namespace
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz,
                                                      unsigned int maxGpuClockMHz) {
  return lock_clocks(device, false, minGpuClockMHz, maxGpuClockMHz, false);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) {
  return lock_clocks(device, false, 0, 0, true);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz,
                                                         unsigned int maxMemClockMHz) {
  return lock_clocks(device, true, minMemClockMHz, maxMemClockMHz, false);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device) {
  return lock_clocks(device, true, 0, 0, true);
}

namespace {
nvmlReturn_t set_vf_offset(nvmlDevice_t device, bool gpc, int offset) {
  const char* key = nullptr;
  std::string uuid;
  int lo = 0, hi = 0;
  const nvmlReturn_t rc = vf_offset_range(device, gpc, &lo, &hi, &key, &uuid);
  if (rc != NVML_SUCCESS) return rc;
  if (offset < lo || offset > hi) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_SETTER_PERMISSION();
  overlay_set(uuid, Life::Volatile, key, std::to_string(offset));
  return NVML_SUCCESS;
}
}  // namespace
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetGpcClkVfOffset(nvmlDevice_t device, int offset) {
  return set_vf_offset(device, true, offset);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetMemClkVfOffset(nvmlDevice_t device, int offset) {
  return set_vf_offset(device, false, offset);
}
#if defined(VGPU_NVML_12_8)
// The offset applies to every P-state: a per-state offset is not modelled.
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t* info) {
  REQUIRE_INIT();
  NVML_ARG(info);
  NVML_VERSION(info, nvmlClockOffset_v1);
  const int type = enum_value(info->type), pstate = enum_value(info->pstate);
  if ((type != NVML_CLOCK_GRAPHICS && type != NVML_CLOCK_MEM) || pstate < 0 || pstate > 15)
    return NVML_ERROR_INVALID_ARGUMENT;
  return set_vf_offset(device, type == NVML_CLOCK_GRAPHICS, info->clockOffsetMHz);
}
#endif

/* ---- fans ---- */

VGPU_EXPORT nvmlReturn_t nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan,
                                                       nvmlFanControlPolicy_t policy) {
  NVML_DEV(device);
  if (const nvmlReturn_t rc = fan_checked(*d, fan); rc != NVML_SUCCESS) return rc;
  if (policy > 1) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_SETTER_PERMISSION();
  overlay_update(d->uuid, Life::Volatile,
                 {{"fan_policy_" + std::to_string(fan), policy ? "1" : ""},
                  {"fan_speed_" + std::to_string(fan), policy ? setting(d->uuid, Life::Volatile,
                                                                       ("fan_speed_" + std::to_string(fan)).c_str(), "")
                                                            : ""}});
  return NVML_SUCCESS;
}
// Setting a speed puts the fan under manual control (the header warns of it).
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed) {
  NVML_DEV(device);
  if (const nvmlReturn_t rc = fan_checked(*d, fan); rc != NVML_SUCCESS) return rc;
  const vgpu::NvmlClass& n = nvml_facts(*d);
  if (speed < n.fan_min_percent || speed > n.fan_max_percent) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_SETTER_PERMISSION();
  overlay_update(d->uuid, Life::Volatile,
                 {{"fan_policy_" + std::to_string(fan), "1"}, {"fan_speed_" + std::to_string(fan), std::to_string(speed)}});
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan) {
  NVML_DEV(device);
  if (const nvmlReturn_t rc = fan_checked(*d, fan); rc != NVML_SUCCESS) return rc;
  NVML_SETTER_PERMISSION();
  overlay_update(d->uuid, Life::Volatile,
                 {{"fan_policy_" + std::to_string(fan), ""}, {"fan_speed_" + std::to_string(fan), ""}});
  return NVML_SUCCESS;
}

/* ---- ECC counters, fields, inforom ---- */

// VOLATILE counts since the driver loaded, AGGREGATE for the life of the card
// (vgpu/ras.hpp); the retired pages and remapped rows stay, as memory taken out
// of service.
VGPU_EXPORT nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType) {
  NVML_DEV(device);
  const int c = static_cast<int>(counterType);
  if (c < 0 || c > 1) return NVML_ERROR_INVALID_ARGUMENT;
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  NVML_SETTER_PERMISSION();
  try {
    if (c == 0) vgpu::ras::reset_volatile(d->uuid, false);
    else vgpu::ras::reset_aggregate(d->uuid);
  } catch (const std::exception&) {
    return NVML_ERROR_UNKNOWN;
  }
  return NVML_SUCCESS;
}
// The call succeeds and each field says what became of it. The fields that count
// something VirtualGPU keeps (the PCIe and ECC counters) are cleared by
// nvmlDeviceClearEccErrorCounts and nvidia-smi's reset, not one at a time.
VGPU_EXPORT nvmlReturn_t nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) {
  NVML_DEV(device);
  if (valuesCount < 0 || (valuesCount > 0 && !values)) return NVML_ERROR_INVALID_ARGUMENT;
  for (int i = 0; i < valuesCount; ++i) values[i].nvmlReturn = NVML_ERROR_NOT_SUPPORTED;
  return NVML_SUCCESS;
}
// The inforom that was read from the card verifies; a profile with none recorded
// has nothing to verify.
VGPU_EXPORT nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device) {
  NVML_DEV(device);
  if (nvml_facts(*d).inforom_image.empty()) return NVML_ERROR_NOT_SUPPORTED;
  return NVML_SUCCESS;
}

/* ---- CPU affinity ---- */

// The affinity of the calling thread: bound to the CPUs NVML reports as ideal
// (the ones the process may use), and unbound to every CPU the system has.
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device) {
  NVML_DEV(device);
  std::vector<unsigned long> words;
  if (!allowed_cpus(&words)) return NVML_ERROR_NOT_SUPPORTED;
  cpu_set_t set;
  CPU_ZERO(&set);
  const unsigned bits = 8 * sizeof(unsigned long);
  for (unsigned c = 0; c < words.size() * bits && c < CPU_SETSIZE; ++c)
    if (words[c / bits] >> (c % bits) & 1ul) CPU_SET(c, &set);
  return ::sched_setaffinity(0, sizeof set, &set) == 0 ? NVML_SUCCESS : NVML_ERROR_UNKNOWN;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device) {
  NVML_DEV(device);
  cpu_set_t set;
  CPU_ZERO(&set);
  const long n = ::sysconf(_SC_NPROCESSORS_CONF);
  for (long c = 0; c < n && c < CPU_SETSIZE; ++c) CPU_SET(static_cast<int>(c), &set);
  return ::sched_setaffinity(0, sizeof set, &set) == 0 ? NVML_SUCCESS : NVML_ERROR_UNKNOWN;
}

/* ---- NVLink ---- */

// Nothing here is connected: the same NOT_SUPPORTED as the NVLink getters, after
// the same checks.
VGPU_EXPORT nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link) {
  NVML_DEV(device);
  if (const nvmlReturn_t rc = link_checked(*d, link); rc != NVML_SUCCESS) return rc;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceResetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link,
                                                                 unsigned int counter) {
  NVML_DEV(device);
  if (counter > 1) return NVML_ERROR_INVALID_ARGUMENT;
  if (const nvmlReturn_t rc = link_checked(*d, link); rc != NVML_SUCCESS) return rc;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link,
                                                               unsigned int counter,
                                                               nvmlNvLinkUtilizationControl_t* control,
                                                               unsigned int reset) {
  NVML_DEV(device);
  if (!control || counter > 1 || reset > 1) return NVML_ERROR_INVALID_ARGUMENT;
  if (const nvmlReturn_t rc = link_checked(*d, link); rc != NVML_SUCCESS) return rc;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  return NVML_ERROR_NOT_SUPPORTED;
}
#if defined(VGPU_NVML_12_8)
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkSetBwMode_t* mode) {
  NVML_DEV(device);
  NVML_ARG(mode);
  NVML_VERSION(mode, nvmlNvlinkSetBwMode_v1);
  return NVML_ERROR_NOT_SUPPORTED;
}
#endif
#if defined(VGPU_NVML_12_4)
VGPU_EXPORT nvmlReturn_t nvmlSystemSetNvlinkBwMode(unsigned int nvlinkBwMode) {
  REQUIRE_INIT();
  (void)nvlinkBwMode;
  return NVML_ERROR_NOT_SUPPORTED;
}
#endif

/* ---- hot removal ----
 *
 * nvmlDeviceRemoveGpu takes a GPU out of the driver's view and nvmlDeviceDiscoverGpus
 * brings it back. The removal is a settings-overlay flag that refresh() honours (it
 * leaves the GPU out of the snapshot, so it is not counted, found or listed) and
 * lasts as long as the session, as a driver reload would end it. A GPU with
 * processes on it is IN_USE and stays.
 */
VGPU_EXPORT nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t newState) {
  REQUIRE_INIT();
  refresh();
  const int s = static_cast<int>(newState);
  if (!pciInfo || (s != 0 && s != 1)) return NVML_ERROR_INVALID_ARGUMENT;
  const auto* d = by_pci(pciInfo, nullptr);
  if (!d) return NVML_ERROR_NOT_FOUND;
  NVML_SETTER_PERMISSION();
  overlay_set(d->uuid, Life::Volatile, "drain", s ? "1" : "");
  return NVML_SUCCESS;
}
namespace {
nvmlReturn_t remove_gpu(nvmlPciInfo_t* pciInfo) {
  REQUIRE_INIT();
  refresh();
  if (!pciInfo) return NVML_ERROR_INVALID_ARGUMENT;
  const auto* d = by_pci(pciInfo, nullptr);
  if (!d) return NVML_ERROR_NOT_FOUND;
  NVML_SETTER_PERMISSION();
  if (d->proc_count) return NVML_ERROR_IN_USE;
  overlay_set(d->uuid, Life::Volatile, "removed", "1");
  return NVML_SUCCESS;
}
}  // namespace
VGPU_EXPORT nvmlReturn_t nvmlDeviceRemoveGpu(nvmlPciInfo_t* pciInfo) { return remove_gpu(pciInfo); }
VGPU_EXPORT nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t* pciInfo, nvmlDetachGpuState_t gpuState,
                                                nvmlPcieLinkState_t linkState) {
  const int g = static_cast<int>(gpuState), l = static_cast<int>(linkState);
  if (g < 0 || g > 1 || l < 0 || l > 1) {
    REQUIRE_INIT();
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  return remove_gpu(pciInfo);
}
// Only a GPU that was removed can be discovered; anything else is not found.
VGPU_EXPORT nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t* pciInfo) {
  REQUIRE_INIT();
  refresh();
  if (!pciInfo) return NVML_ERROR_INVALID_ARGUMENT;
  char text[40];
  std::snprintf(text, sizeof text, "%08x:%02x:%02x.0", pciInfo->domain, pciInfo->bus, pciInfo->device);
  for (const auto& r : g_removed) {
    if (!same_bus_id(r.second.c_str(), text)) continue;
    NVML_SETTER_PERMISSION();
    overlay_set(r.first, Life::Volatile, "removed", "");
    return NVML_SUCCESS;
  }
  return NVML_ERROR_NOT_FOUND;
}
