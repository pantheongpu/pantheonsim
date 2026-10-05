// NVML: identity, clocks, fans, power, PCIe, thermals, affinity. See nvml_common.inc.
//
// Where an answer comes from, in the order the file uses them:
//   * the profile's nvml: section (datasheets, and for the RTX 3060 values read from the card);
//   * the telemetry sample and the reliability state the rest of NVML already uses;
//   * the host, for CPU and NUMA affinity (the one part of a topology a simulator
//     does not have to model -- `vgpu smi topo` reads it the same way).
// Everything else is NVML_ERROR_NOT_SUPPORTED, which is what the real driver
// answers for a feature a GPU of that class lacks and what every tool renders as
// N/A. Measurements on the card were taken through NVIDIA's NVML 13.0 on the WSL
// driver 596.36, GPU index 1, getters only. That driver is a Windows-hosted one:
// a few of its answers belong to WSL (no /dev/nvidia, no CPU or NUMA affinity,
// WDDM driver model) and are not copied; the comments say where.

namespace {

// The graphics clocks nvmlDeviceGetSupportedGraphicsClocks lists for a memory
// clock, highest first: from the profile's maximum in the card's step down to its
// floor, and at the lowest memory clock only the idle range. Empty where the
// profile has no table.
std::vector<unsigned> graphics_clock_list(const vgpu::telemetry::DeviceSample& d, unsigned mem_clock) {
  const vgpu::NvmlClass& n = nvml_facts(d);
  std::vector<unsigned> out;
  if (n.memory_clocks_mhz.empty() || !n.graphics_step_tenths_mhz || !d.sm_clock_max_mhz) return out;
  const unsigned lowest = *std::min_element(n.memory_clocks_mhz.begin(), n.memory_clocks_mhz.end());
  const unsigned long top10 = static_cast<unsigned long>(d.sm_clock_max_mhz) * 10;
  for (unsigned long k = 0; top10 >= n.graphics_step_tenths_mhz * k; ++k) {
    const unsigned mhz = static_cast<unsigned>((top10 - n.graphics_step_tenths_mhz * k) / 10);
    if (mhz < n.graphics_list_floor_mhz) break;
    if (mem_clock == lowest && n.idle_graphics_max_mhz && mhz > n.idle_graphics_max_mhz) continue;
    out.push_back(mhz);
  }
  return out;
}

// The clock lock a setter recorded: {min, max} in MHz, or {0, 0}.
struct Lock {
  unsigned lo = 0, hi = 0;
};
Lock locked_clocks(const std::string& uuid, const char* key) {
  Lock l;
  const std::string v = setting(uuid, Life::Volatile, key, "");
  std::sscanf(v.c_str(), "%u,%u", &l.lo, &l.hi);
  return l;
}

}  // namespace

/* ---- identity ---- */

// Brand: the profile's. A GeForce board is NVML_BRAND_GEFORCE; every datacenter
// part since Ampere is NVML_BRAND_NVIDIA (the card reports 5 for the RTX 3060;
// the datacenter value is NVIDIA's documented one and has no card behind it).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t* type) {
  NVML_DEV(device);
  NVML_ARG(type);
  const std::string& b = nvml_facts(*d).brand;
  *type = b == "geforce" ? NVML_BRAND_GEFORCE : b == "nvidia" ? NVML_BRAND_NVIDIA : NVML_BRAND_UNKNOWN;
  return NVML_SUCCESS;
}
// Every simulated GPU is on PCIe (NVML_BUS_TYPE_PCIE = 2, as the card answers).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t* type) {
  NVML_DEV(device);
  NVML_ARG(type);
  *type = NVML_BUS_TYPE_PCIE;
  return NVML_SUCCESS;
}
// The board id groups GPUs behind one PLX switch; the card answers bus << 8
// (0x600 for 06:00.0). Each simulated device has its own bus, hence its own id.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int* boardId) {
  NVML_DEV(device);
  NVML_ARG(boardId);
  unsigned int domain = 0, bus = 0, dev_id = 0;
  std::sscanf(d->bus_id, "%x:%x:%x", &domain, &bus, &dev_id);
  *boardId = (bus << 8) | (dev_id << 3);
  return NVML_SUCCESS;
}
// One GPU per board: not a multi-GPU board (the card answers 0).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int* multiGpuBool) {
  NVML_DEV(device);
  NVML_ARG(multiGpuBool);
  *multiGpuBool = 0;
  return NVML_SUCCESS;
}
// The card answers NOT_SUPPORTED for a device compared with itself: the query is
// for multi-GPU boards. Both handles are validated first.
VGPU_EXPORT nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int* onSameBoard) {
  REQUIRE_INIT();
  refresh();
  if (!sample(device1)) return bad(device1);
  if (!sample(device2)) return bad(device2);
  NVML_ARG(onSameBoard);
  return NVML_ERROR_NOT_SUPPORTED;
}
// The module id is the GPU's place on its board/node; the card (GPU index 1)
// answers 1, taken here as the index (not verified for index 0).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetModuleId(nvmlDevice_t device, unsigned int* moduleId) {
  NVML_DEV(device);
  NVML_ARG(moduleId);
  *moduleId = device_index(d);
  return NVML_SUCCESS;
}
// A part number is programmed into the board's inforom; a simulated board has none.
VGPU_NVML_NS_DEV(nvmlDeviceGetBoardPartNumber, device, (nvmlDevice_t device, char*, unsigned int))
// No device has a serial number (nvmlDeviceGetSerial says so), so none is found.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetHandleBySerial(const char* serial, nvmlDevice_t* device) {
  REQUIRE_INIT();
  if (!serial || !device) return NVML_ERROR_INVALID_ARGUMENT;
  return NVML_ERROR_NOT_FOUND;
}
// The per-device identifier is unique per simulated device; derived from the UUID.
#if defined(VGPU_NVML_13_0)
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPdi(nvmlDevice_t device, nvmlPdi_t* pdi) {
  NVML_DEV(device);
  NVML_ARG(pdi);
  NVML_VERSION(pdi, nvmlPdi_v1);
  unsigned long long h = 1469598103934665603ull;
  for (const char* c = d->uuid; *c; ++c) h = (h ^ static_cast<unsigned char>(*c)) * 1099511628211ull;
  pdi->value = h;
  return NVML_SUCCESS;
}
#endif
#if defined(VGPU_NVML_12_9)
// By UUID in the versioned form: type ASCII (1) or binary (2). The card's NVML
// refuses a zeroed type (NONE) as an invalid argument. Binary UUIDs are 16 raw
// bytes; ours are text, so only the ASCII form can match.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetHandleByUUIDV(const nvmlUUID_t* uuid, nvmlDevice_t* device) {
  REQUIRE_INIT();
  if (!uuid || !device) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_VERSION(uuid, nvmlUUID_v1);
  const int type = enum_value(uuid->type);
  if (type != NVML_UUID_TYPE_ASCII)
    return type == NVML_UUID_TYPE_BINARY ? NVML_ERROR_NOT_FOUND : NVML_ERROR_INVALID_ARGUMENT;
  char text[NVML_DEVICE_UUID_ASCII_LEN + 1] = {0};
  std::memcpy(text, uuid->value.str, NVML_DEVICE_UUID_ASCII_LEN);
  return nvmlDeviceGetHandleByUUID(text, device);
}
#endif
// MIG device handles are the only ones with attributes; the card answers
// NOT_SUPPORTED for a physical GPU.
VGPU_NVML_NS_DEV(nvmlDeviceGetAttributes, device, (nvmlDevice_t device, nvmlDeviceAttributes_t*))
VGPU_NVML_NS_DEV(nvmlDeviceGetAttributes_v2, device, (nvmlDevice_t device, nvmlDeviceAttributes_t*))

// The inforom versions, from the profile where a card was read: object 0 (OEM) and
// the image version; the ECC, power and DEN objects the card lacks.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object,
                                                     char* version, unsigned int length) {
  NVML_DEV(device);
  NVML_ARG(version);
  const int o = static_cast<int>(object);
  if (o < 0 || o >= 4) return NVML_ERROR_INVALID_ARGUMENT;   // the card: 4 is invalid, 1..3 not supported
  const vgpu::NvmlClass& n = nvml_facts(*d);
  if (o != 0 || n.inforom_oem.empty()) return NVML_ERROR_NOT_SUPPORTED;
  return copy_string(n.inforom_oem.c_str(), version, length);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char* version,
                                                          unsigned int length) {
  NVML_DEV(device);
  NVML_ARG(version);
  const vgpu::NvmlClass& n = nvml_facts(*d);
  if (n.inforom_image.empty()) return NVML_ERROR_NOT_SUPPORTED;
  return copy_string(n.inforom_image.c_str(), version, length);
}
// The checksum of the inforom's configuration: the card answers 0 (a board
// whose configuration section is empty).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device,
                                                                   unsigned int* checksum) {
  NVML_DEV(device);
  NVML_ARG(checksum);
  if (nvml_facts(*d).inforom_image.empty()) return NVML_ERROR_NOT_SUPPORTED;
  *checksum = 0;
  return NVML_SUCCESS;
}
// GSP firmware: the card (WSL) answers NOT_SUPPORTED to both queries, and no
// profile records a GSP firmware version.
VGPU_NVML_NS_DEV(nvmlDeviceGetGspFirmwareMode, device, (nvmlDevice_t device, unsigned int*, unsigned int*))
VGPU_NVML_NS_DEV(nvmlDeviceGetGspFirmwareVersion, device, (nvmlDevice_t device, char*))
// Interrupt line: a simulated device has none; the card (WSL) answers 0.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int* irqNum) {
  NVML_DEV(device);
  NVML_ARG(irqNum);
  *irqNum = 0;
  return NVML_SUCCESS;
}
#if defined(VGPU_NVML_13_0)
// A hostname is written by a fabric manager, which a simulated machine lacks
// (the card answers NOT_SUPPORTED).
VGPU_NVML_NS_DEV(nvmlDeviceGetHostname_v1, device, (nvmlDevice_t device, nvmlHostname_v1_t*))
#endif
#if defined(VGPU_NVML_12_8)
// The card answers host id 1 and its module id, with no chassis, slot or tray
// (a PCIe board is in no chassis).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPlatformInfo(nvmlDevice_t device, nvmlPlatformInfo_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
#if defined(nvmlPlatformInfo_v2)
  if (info->version != nvmlPlatformInfo_v1 && info->version != nvmlPlatformInfo_v2)
#else
  if (info->version != nvmlPlatformInfo_v1)
#endif
    return NVML_ERROR_ARGUMENT_VERSION_MISMATCH;
  if (!consumer(*d)) return NVML_ERROR_NOT_SUPPORTED;   // an SXM board's platform is not simulated
  const unsigned version = info->version;
  std::memset(info, 0, sizeof *info);
  info->version = version;
  info->hostId = 1;
  info->moduleId = static_cast<unsigned char>(device_index(d));
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCapabilities(nvmlDevice_t device, nvmlDeviceCapabilities_t* caps) {
  NVML_DEV(device);
  NVML_ARG(caps);
  NVML_VERSION(caps, nvmlDeviceCapabilities_v1);
  // The card's mask is 0. A datacenter part's capabilities (EGM, fabric) depend
  // on hardware state VirtualGPU does not model.
  if (!consumer(*d)) return NVML_ERROR_NOT_SUPPORTED;
  caps->capMask = 0;
  return NVML_SUCCESS;
}
#endif
#if defined(VGPU_NVML_12_4)
// The Grace link: one NVLink-C2C on a GH200, enabled; none elsewhere.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetC2cModeInfoV(nvmlDevice_t device, nvmlC2cModeInfo_v1_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  if (!nvml_facts(*d).c2c_links) return NVML_ERROR_NOT_SUPPORTED;
  info->isC2cEnabled = 1;
  return NVML_SUCCESS;
}
// PCI info with the class code. A GPU is a display controller (base class 3):
// VGA-compatible (sub class 0) for a GeForce board, a 3D controller (sub class 2)
// for a datacenter one. The card (WSL) reads 0/0 from its virtual PCI device.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPciInfoExt(nvmlDevice_t device, nvmlPciInfoExt_t* pci) {
  NVML_DEV(device);
  NVML_ARG(pci);
  NVML_VERSION(pci, nvmlPciInfoExt_v1);
  nvmlPciInfo_t legacy;
  const nvmlReturn_t rc = nvmlDeviceGetPciInfo_v3(device, &legacy);
  if (rc != NVML_SUCCESS) return rc;
  pci->domain = legacy.domain;
  pci->bus = legacy.bus;
  pci->device = legacy.device;
  pci->pciDeviceId = legacy.pciDeviceId;
  pci->pciSubSystemId = legacy.pciSubSystemId;
  pci->baseClass = 3;
  pci->subClass = consumer(*d) ? 0 : 2;
  const size_t n = sizeof pci->busId < sizeof legacy.busId ? sizeof pci->busId : sizeof legacy.busId;
  std::memcpy(pci->busId, legacy.busId, n);
  return NVML_SUCCESS;
}
#endif

/* ---- memory, cores, engines ---- */

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int* busWidth) {
  NVML_DEV(device);
  NVML_ARG(busWidth);
  if (!memory_bus_width(*d)) return NVML_ERROR_NOT_SUPPORTED;
  *busWidth = memory_bus_width(*d);
  return NVML_SUCCESS;
}
// Adaptive clocking: enabled on the card (1). A datacenter part's status has no
// card behind it.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int* status) {
  NVML_DEV(device);
  NVML_ARG(status);
  if (!consumer(*d)) return NVML_ERROR_NOT_SUPPORTED;
  *status = 1;   // NVML_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED
  return NVML_SUCCESS;
}

/* ---- clocks ---- */

// Clock domains: 0 graphics, 1 SM, 2 memory, 3 video. The card answers the
// current clock (id 0) for all four and NOT_SUPPORTED for the application,
// default-application and customer-boost ids (applications clocks are deprecated
// and VirtualGPU keeps none), and for the counting enumerator (4).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType,
                                            nvmlClockId_t clockId, unsigned int* clockMHz) {
  NVML_DEV(device);
  NVML_ARG(clockMHz);
  const int type = static_cast<int>(clockType), id = static_cast<int>(clockId);
  if (type < 0 || type > 4 || id < 0 || id > 3) return NVML_ERROR_INVALID_ARGUMENT;
  if (type == 4 || id != 0) return NVML_ERROR_NOT_SUPPORTED;
  return nvmlDeviceGetClockInfo(device, clockType, clockMHz);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType,
                                                            unsigned int* clockMHz) {
  NVML_DEV(device);
  NVML_ARG(clockMHz);
  return static_cast<int>(clockType) < 0 || static_cast<int>(clockType) > 4 ? NVML_ERROR_INVALID_ARGUMENT
                                                                           : NVML_ERROR_NOT_SUPPORTED;
}
// Applications clocks and auto boost: deprecated, absent on the card, not simulated.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType,
                                                        unsigned int* clockMHz) {
  NVML_DEV(device);
  NVML_ARG(clockMHz);
  return static_cast<int>(clockType) < 0 || static_cast<int>(clockType) > 4 ? NVML_ERROR_INVALID_ARGUMENT
                                                                           : NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType,
                                                               unsigned int* clockMHz) {
  return nvmlDeviceGetApplicationsClock(device, clockType, clockMHz);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t* isEnabled,
                                                               nvmlEnableState_t* defaultIsEnabled) {
  NVML_DEV(device);
  NVML_ARG(isEnabled);
  (void)defaultIsEnabled;
  return NVML_ERROR_NOT_SUPPORTED;
}
// Which API calls are root-only: the card answers NOT_SUPPORTED for both.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType,
                                                     nvmlEnableState_t* isRestricted) {
  NVML_DEV(device);
  NVML_ARG(isRestricted);
  const int t = static_cast<int>(apiType);
  if (t < 0 || t > 1) return NVML_ERROR_INVALID_ARGUMENT;
  return NVML_ERROR_NOT_SUPPORTED;
}

// The memory clocks and the graphics clocks at each: from the profile's table
// (the card's, for the RTX 3060); NOT_SUPPORTED where none was read. The count
// is in/out: INSUFFICIENT_SIZE with the needed count when the buffer is small.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int* count,
                                                            unsigned int* clocksMHz) {
  NVML_DEV(device);
  NVML_ARG(count);
  const vgpu::NvmlClass& n = nvml_facts(*d);
  if (n.memory_clocks_mhz.empty()) return NVML_ERROR_NOT_SUPPORTED;
  const unsigned need = static_cast<unsigned>(n.memory_clocks_mhz.size());
  if (!clocksMHz) {
    // Measured: a NULL buffer with a count of 0 is INSUFFICIENT_SIZE with the
    // needed count; a NULL buffer with a count that claims room is INVALID_ARGUMENT.
    if (*count != 0) return NVML_ERROR_INVALID_ARGUMENT;
    *count = need;
    return NVML_ERROR_INSUFFICIENT_SIZE;
  }
  if (*count < need) {
    *count = need;
    return NVML_ERROR_INSUFFICIENT_SIZE;
  }
  std::copy(n.memory_clocks_mhz.begin(), n.memory_clocks_mhz.end(), clocksMHz);
  *count = need;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz,
                                                              unsigned int* count, unsigned int* clocksMHz) {
  NVML_DEV(device);
  NVML_ARG(count);
  const vgpu::NvmlClass& n = nvml_facts(*d);
  if (n.memory_clocks_mhz.empty()) return NVML_ERROR_NOT_SUPPORTED;
  // Measured: a memory clock the GPU does not list is INVALID_ARGUMENT for 0 and
  // NOT_FOUND for any other number.
  if (std::find(n.memory_clocks_mhz.begin(), n.memory_clocks_mhz.end(), memoryClockMHz) == n.memory_clocks_mhz.end())
    return memoryClockMHz == 0 ? NVML_ERROR_INVALID_ARGUMENT : NVML_ERROR_NOT_FOUND;
  const std::vector<unsigned> list = graphics_clock_list(*d, memoryClockMHz);
  const unsigned need = static_cast<unsigned>(list.size());
  if (!clocksMHz || *count < need) {
    *count = need;
    return NVML_ERROR_INSUFFICIENT_SIZE;
  }
  std::copy(list.begin(), list.end(), clocksMHz);
  *count = need;
  return NVML_SUCCESS;
}
// The performance states the GPU runs in. The card requires room for all of
// NVML_MAX_GPU_PERF_PSTATES (16): a smaller non-zero size is INSUFFICIENT_SIZE and
// a zero size INVALID_ARGUMENT.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t* pstates,
                                                                 unsigned int size) {
  NVML_DEV(device);
  NVML_ARG(pstates);
  if (size == 0) return NVML_ERROR_INVALID_ARGUMENT;
  const vgpu::NvmlClass& n = nvml_facts(*d);
  if (n.pstates.empty()) return NVML_ERROR_NOT_SUPPORTED;
  if (size < NVML_MAX_GPU_PERF_PSTATES) return NVML_ERROR_INSUFFICIENT_SIZE;
  for (unsigned i = 0; i < size; ++i)
    pstates[i] = i < n.pstates.size() ? static_cast<nvmlPstates_t>(n.pstates[i]) : NVML_PSTATE_UNKNOWN;
  return NVML_SUCCESS;
}
// The range of a clock in a performance state. Measured on the card: graphics
// and SM from 210 MHz to the maximum (to the idle maximum in P8), memory exactly
// the memory clock of that state (the lowest in P8, the highest in P0), video 555
// to its maximum; a state the GPU does not run in answers UNKNOWN (999), an
// out-of-range state or a clock type that is no domain INVALID_ARGUMENT.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type,
                                                          nvmlPstates_t pstate, unsigned int* minClockMHz,
                                                          unsigned int* maxClockMHz) {
  NVML_DEV(device);
  NVML_ARG(minClockMHz);
  NVML_ARG(maxClockMHz);
  const int t = static_cast<int>(type), p = static_cast<int>(pstate);
  if (t < 0 || t > 3 || p < 0 || p > 15) return NVML_ERROR_INVALID_ARGUMENT;
  const vgpu::NvmlClass& n = nvml_facts(*d);
  if (n.pstates.empty() || n.memory_clocks_mhz.empty()) return NVML_ERROR_NOT_SUPPORTED;
  const auto at = std::find(n.pstates.begin(), n.pstates.end(), static_cast<unsigned>(p));
  if (at == n.pstates.end()) return NVML_ERROR_UNKNOWN;
  // The states are listed lowest first (P8 ... P0), and so are the memory clocks
  // sorted ascending: the first state has the lowest memory clock.
  const size_t rank = static_cast<size_t>(at - n.pstates.begin());
  std::vector<unsigned> mem = n.memory_clocks_mhz;
  std::sort(mem.begin(), mem.end());
  const unsigned mem_clock = rank < mem.size() ? mem[rank] : mem.back();
  const bool idle = rank == 0;
  if (t == NVML_CLOCK_MEM) {
    *minClockMHz = *maxClockMHz = mem_clock;
  } else if (t == NVML_CLOCK_VIDEO) {
    if (!n.video_clock_max_mhz) return NVML_ERROR_NOT_SUPPORTED;
    *minClockMHz = n.video_clock_floor_mhz;
    *maxClockMHz = idle ? n.video_clock_floor_mhz : n.video_clock_max_mhz;
  } else {
    *minClockMHz = n.graphics_floor_mhz;
    *maxClockMHz = idle ? n.idle_graphics_max_mhz : d->sm_clock_max_mhz;
  }
  return NVML_SUCCESS;
}

#if defined(VGPU_NVML_12_8)
namespace {
// The strings nvmlDeviceGetCurrentClockFreqs and nvmlDeviceGetPerformanceModes
// return, in the format the card produces ("key=value, ..."): a domain's current
// value, range and whether it can be edited.
std::string clock_domain_text(const char* name, unsigned cur, unsigned lo, unsigned hi, bool with_cur) {
  char buf[200];
  if (with_cur)
    std::snprintf(buf, sizeof buf, "%s=%u, %smin=%u, %smax=%u, %seditable=1", name, cur, name, lo, name, hi, name);
  else
    std::snprintf(buf, sizeof buf, "%smin=%u, %smax=%u, %seditable=1", name, lo, name, hi, name);
  return buf;
}
}  // namespace

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCurrentClockFreqs(nvmlDevice_t device, nvmlDeviceCurrentClockFreqs_t* freqs) {
  NVML_DEV(device);
  NVML_ARG(freqs);
  NVML_VERSION(freqs, nvmlDeviceCurrentClockFreqs_v1);
  const vgpu::NvmlClass& n = nvml_facts(*d);
  if (n.memory_clocks_mhz.empty() || !n.graphics_floor_mhz) return NVML_ERROR_NOT_SUPPORTED;
  const Lock lock = locked_clocks(d->uuid, "gpu_locked_clocks");
  const unsigned lo = lock.hi ? lock.lo : n.graphics_floor_mhz, hi = lock.hi ? lock.hi : d->sm_clock_max_mhz;
  const std::string text =
      clock_domain_text("nvclock", d->sm_clock_mhz, lo, hi, true) + ", " +
      clock_domain_text("memclock", d->mem_clock_mhz, d->mem_clock_max_mhz, d->mem_clock_max_mhz, true) + ", " +
      clock_domain_text("memTransferRate", d->mem_clock_mhz * 2, d->mem_clock_max_mhz * 2,
                        d->mem_clock_max_mhz * 2, true);
  return copy_string(text.c_str(), freqs->str, sizeof freqs->str);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPerformanceModes(nvmlDevice_t device, nvmlDevicePerfModes_t* modes) {
  NVML_DEV(device);
  NVML_ARG(modes);
  NVML_VERSION(modes, nvmlDevicePerfModes_v1);
  const vgpu::NvmlClass& n = nvml_facts(*d);
  if (n.memory_clocks_mhz.empty() || !n.graphics_floor_mhz) return NVML_ERROR_NOT_SUPPORTED;
  std::vector<unsigned> mem = n.memory_clocks_mhz;
  std::sort(mem.begin(), mem.end());
  std::string text;
  for (size_t i = 0; i < mem.size(); ++i) {
    const unsigned hi = i == 0 ? n.idle_graphics_max_mhz : d->sm_clock_max_mhz;
    char head[40];
    std::snprintf(head, sizeof head, "perf=%zu, nvclock=%u, ", i, n.graphics_floor_mhz);
    text += (i ? " ; " : "") + std::string(head) + clock_domain_text("nvclock", 0, n.graphics_floor_mhz, hi, false) +
            ", " + clock_domain_text("memclock", mem[i], mem[i], mem[i], true) + ", " +
            clock_domain_text("memTransferRate", mem[i] * 2, mem[i] * 2, mem[i] * 2, true);
  }
  return copy_string(text.c_str(), modes->str, sizeof modes->str);
}
// Overclocking offsets for one clock domain in one P-state. The card answers the
// graphics (type 0) and memory (type 2) domains with the ranges the profile
// records, and the SM and video domains INVALID_ARGUMENT. The offset is what
// nvmlDeviceSetGpcClkVfOffset / SetMemClkVfOffset recorded.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetClockOffsets(nvmlDevice_t device, nvmlClockOffset_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  NVML_VERSION(info, nvmlClockOffset_v1);
  const int type = enum_value(info->type), pstate = enum_value(info->pstate);
  const vgpu::NvmlClass& n = nvml_facts(*d);
  if (type != NVML_CLOCK_GRAPHICS && type != NVML_CLOCK_MEM) return NVML_ERROR_INVALID_ARGUMENT;
  if (pstate < 0 || pstate > 15) return NVML_ERROR_INVALID_ARGUMENT;
  const bool gpc = type == NVML_CLOCK_GRAPHICS;
  if (gpc ? n.gpc_offset_max_mhz == 0 : n.mem_offset_max_mhz == 0) return NVML_ERROR_NOT_SUPPORTED;
  info->minClockOffsetMHz = gpc ? n.gpc_offset_min_mhz : n.mem_offset_min_mhz;
  info->maxClockOffsetMHz = gpc ? n.gpc_offset_max_mhz : n.mem_offset_max_mhz;
  info->clockOffsetMHz = static_cast<int>(setting_int(d->uuid, Life::Volatile, gpc ? "gpc_offset" : "mem_offset", 0));
  return NVML_SUCCESS;
}
#endif

namespace {
nvmlReturn_t vf_offset_range(nvmlDevice_t device, bool gpc, int* lo, int* hi, const char** key, std::string* uuid) {
  NVML_DEV(device);
  const vgpu::NvmlClass& n = nvml_facts(*d);
  if (gpc ? n.gpc_offset_max_mhz == 0 : n.mem_offset_max_mhz == 0) return NVML_ERROR_NOT_SUPPORTED;
  if (lo) *lo = gpc ? n.gpc_offset_min_mhz : n.mem_offset_min_mhz;
  if (hi) *hi = gpc ? n.gpc_offset_max_mhz : n.mem_offset_max_mhz;
  if (key) *key = gpc ? "gpc_offset" : "mem_offset";
  if (uuid) *uuid = d->uuid;
  return NVML_SUCCESS;
}
}  // namespace
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) {
  if (!minOffset || !maxOffset) { REQUIRE_INIT(); return NVML_ERROR_INVALID_ARGUMENT; }
  return vf_offset_range(device, true, minOffset, maxOffset, nullptr, nullptr);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) {
  if (!minOffset || !maxOffset) { REQUIRE_INIT(); return NVML_ERROR_INVALID_ARGUMENT; }
  return vf_offset_range(device, false, minOffset, maxOffset, nullptr, nullptr);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int* offset) {
  if (!offset) { REQUIRE_INIT(); return NVML_ERROR_INVALID_ARGUMENT; }
  const char* key = nullptr;
  std::string uuid;
  const nvmlReturn_t rc = vf_offset_range(device, true, nullptr, nullptr, &key, &uuid);
  if (rc != NVML_SUCCESS) return rc;
  *offset = static_cast<int>(setting_int(uuid, Life::Volatile, key, 0));
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int* offset) {
  if (!offset) { REQUIRE_INIT(); return NVML_ERROR_INVALID_ARGUMENT; }
  const char* key = nullptr;
  std::string uuid;
  const nvmlReturn_t rc = vf_offset_range(device, false, nullptr, nullptr, &key, &uuid);
  if (rc != NVML_SUCCESS) return rc;
  *offset = static_cast<int>(setting_int(uuid, Life::Volatile, key, 0));
  return NVML_SUCCESS;
}

// Clock-event reasons: every reason NVML defines is one a GPU can report (the
// card answers 0x1ff, bits 0 to 8). The current reasons are what
// nvmlDeviceGetCurrentClocksThrottleReasons reports, under its newer name.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device,
                                                                     unsigned long long* reasons) {
  NVML_DEV(device);
  NVML_ARG(reasons);
  *reasons = 0x1ff;
  return NVML_SUCCESS;
}
#if defined(VGPU_NVML_12_4)
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSupportedClocksEventReasons(nvmlDevice_t device,
                                                                  unsigned long long* reasons) {
  return nvmlDeviceGetSupportedClocksThrottleReasons(device, reasons);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t device,
                                                                unsigned long long* reasons) {
  return nvmlDeviceGetCurrentClocksThrottleReasons(device, reasons);
}
#endif
// How long the clocks were held below their application clocks, by policy, in
// nanoseconds, against a CPU timestamp in microseconds. What VirtualGPU counts is
// what `vgpu fault throttle` injected (vgpu/ras.hpp): power, thermal and board
// limits from the matching reasons; nothing it simulates ever idles a clock down
// on its own account, so the low-utilization and reliability policies read 0, as
// does sync boost. Policies 0 to 5 are the ones the card answers; 6 and above
// are invalid.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType,
                                                      nvmlViolationTime_t* violTime) {
  NVML_DEV(device);
  NVML_ARG(violTime);
  const int p = static_cast<int>(perfPolicyType);
  if (p < 0 || p > 5) return NVML_ERROR_INVALID_ARGUMENT;
  unsigned long long us = 0;
  try {
    using namespace vgpu::ras;
    switch (p) {
      case 0: us = throttle_time_us(d->uuid, kSwPowerCap); break;
      case 1: us = throttle_time_us(d->uuid, kSwThermalSlowdown) + throttle_time_us(d->uuid, kHwThermalSlowdown); break;
      case 3: us = throttle_time_us(d->uuid, kHwSlowdown) + throttle_time_us(d->uuid, kHwPowerBrakeSlowdown); break;
      default: break;
    }
  } catch (const std::exception&) {
  }
  violTime->referenceTime = static_cast<unsigned long long>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
  violTime->violationTime = us * 1000ull;
  return NVML_SUCCESS;
}

/* ---- fans, coolers ---- */

// Fans: the profile records a count where a card was read (the RTX 3060 has 2);
// a passive datacenter board has none and the others are unrecorded, which NVML
// renders as N/A. A fan index past the count is INVALID_ARGUMENT, as measured.
namespace {
nvmlReturn_t fan_checked(const vgpu::telemetry::DeviceSample& d, unsigned fan) {
  const vgpu::NvmlClass& n = nvml_facts(d);
  if (!n.fans) return NVML_ERROR_NOT_SUPPORTED;
  return fan < n.fans ? NVML_SUCCESS : NVML_ERROR_INVALID_ARGUMENT;
}
long long manual_fan_speed(const vgpu::telemetry::DeviceSample& d, unsigned fan) {
  return setting_int(d.uuid, Life::Volatile, ("fan_speed_" + std::to_string(fan)).c_str(), -1);
}
}  // namespace
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int* numFans) {
  NVML_DEV(device);
  NVML_ARG(numFans);
  if (!nvml_facts(*d).fans) return NVML_ERROR_NOT_SUPPORTED;
  *numFans = nvml_facts(*d).fans;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int* speed) {
  NVML_DEV(device);
  NVML_ARG(speed);
  if (const nvmlReturn_t rc = fan_checked(*d, fan); rc != NVML_SUCCESS) return rc;
  const long long manual = manual_fan_speed(*d, fan);
  *speed = manual >= 0 ? static_cast<unsigned>(manual) : d->fan_percent;
  return NVML_SUCCESS;
}
// The target is what the fan controller aims for: the minimum the fan runs at
// while the GPU is up (30 percent on the card), or the manual setting.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int* target) {
  NVML_DEV(device);
  NVML_ARG(target);
  if (const nvmlReturn_t rc = fan_checked(*d, fan); rc != NVML_SUCCESS) return rc;
  const long long manual = manual_fan_speed(*d, fan);
  *target = manual >= 0 ? static_cast<unsigned>(manual) : std::max(nvml_facts(*d).fan_min_percent, d->fan_percent);
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int* minSpeed,
                                                     unsigned int* maxSpeed) {
  NVML_DEV(device);
  NVML_ARG(minSpeed);
  NVML_ARG(maxSpeed);
  const vgpu::NvmlClass& n = nvml_facts(*d);
  if (!n.fans || !n.fan_max_percent) return NVML_ERROR_NOT_SUPPORTED;
  *minSpeed = n.fan_min_percent;
  *maxSpeed = n.fan_max_percent;
  return NVML_SUCCESS;
}
// 0 is NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW (the default), 1 manual.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan,
                                                          nvmlFanControlPolicy_t* policy) {
  NVML_DEV(device);
  NVML_ARG(policy);
  if (const nvmlReturn_t rc = fan_checked(*d, fan); rc != NVML_SUCCESS) return rc;
  *policy = static_cast<nvmlFanControlPolicy_t>(
      setting_int(d->uuid, Life::Volatile, ("fan_policy_" + std::to_string(fan)).c_str(), 0));
  return NVML_SUCCESS;
}
#if defined(VGPU_NVML_12_8)
// RPM: a stopped fan (0 percent) is 0 RPM, a fact; a running fan's speed in RPM
// needs the fan's tachometer curve, which no profile records, so it is N/A.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetFanSpeedRPM(nvmlDevice_t device, nvmlFanSpeedInfo_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  NVML_VERSION(info, nvmlFanSpeedInfo_v1);
  if (const nvmlReturn_t rc = fan_checked(*d, info->fan); rc != NVML_SUCCESS) return rc;
  const long long manual = manual_fan_speed(*d, info->fan);
  if ((manual >= 0 ? static_cast<unsigned>(manual) : d->fan_percent) != 0) return NVML_ERROR_NOT_SUPPORTED;
  info->speed = 0;
  return NVML_SUCCESS;
}
// Each fan is a variable-speed cooler cooling the GPU, its memory and the power
// supply (signal type 2, target 14), as measured.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCoolerInfo(nvmlDevice_t device, nvmlCoolerInfo_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  NVML_VERSION(info, nvmlCoolerInfo_v1);
  if (const nvmlReturn_t rc = fan_checked(*d, info->index); rc != NVML_SUCCESS) return rc;
  info->signalType = NVML_THERMAL_COOLER_SIGNAL_VARIABLE;
  info->target = NVML_THERMAL_COOLER_TARGET_GPU_RELATED;
  return NVML_SUCCESS;
}
#endif

/* ---- thermals, power, performance state ---- */

#if defined(VGPU_NVML_12_8)
// The versioned temperature: the GPU sensor (the only one, NVML_TEMPERATURE_GPU
// = 0); the card refuses any other sensor type as an invalid argument.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetTemperatureV(nvmlDevice_t device, nvmlTemperature_t* temperature) {
  NVML_DEV(device);
  NVML_ARG(temperature);
  NVML_VERSION(temperature, nvmlTemperature_v1);
  if (enum_value(temperature->sensorType) != NVML_TEMPERATURE_GPU) return NVML_ERROR_INVALID_ARGUMENT;
  temperature->temperature = static_cast<int>(d->temperature_c);
  return NVML_SUCCESS;
}
// Margin to the slowdown threshold; Hopper-generation and newer datacenter GPUs
// report it (the card, a GeForce, says NOT_SUPPORTED).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMarginTemperature(nvmlDevice_t device, nvmlMarginTemperature_t* margin) {
  NVML_DEV(device);
  NVML_ARG(margin);
  NVML_VERSION(margin, nvmlMarginTemperature_v1);
  if (consumer(*d) || d->cc_major < 9 || d->temperature_max_c == 0) return NVML_ERROR_NOT_SUPPORTED;
  margin->marginTemperature = static_cast<int>(d->temperature_max_c) - static_cast<int>(d->temperature_c);
  return NVML_SUCCESS;
}
#endif
// Thermal controllers: the GPU's internal sensor with the range the card reports
// (0 to 127 C), for a GeForce board. Sensor index 0 and 15 (every target) answer
// it; any other index answers no sensors.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex,
                                                      nvmlGpuThermalSettings_t* settings) {
  NVML_DEV(device);
  NVML_ARG(settings);
  if (!consumer(*d)) return NVML_ERROR_NOT_SUPPORTED;
  std::memset(settings, 0, sizeof *settings);
  if (sensorIndex != 0 && sensorIndex != 15) return NVML_SUCCESS;
  settings->count = 1;
  settings->sensor[0].controller = NVML_THERMAL_CONTROLLER_GPU_INTERNAL;
  settings->sensor[0].defaultMinTemp = 0;
  settings->sensor[0].defaultMaxTemp = 127;
  settings->sensor[0].currentTemp = static_cast<int>(d->temperature_c);
  settings->sensor[0].target = NVML_THERMAL_TARGET_GPU;
  return NVML_SUCCESS;
}
// The utilization domains the dynamic P-state controller watches (graphics,
// frame buffer, video, bus): the card marks the first four present, with no
// thresholds. The model has the first two (the real utilization); video and bus
// carry no activity.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  if (!consumer(*d)) return NVML_ERROR_NOT_SUPPORTED;
  std::memset(info, 0, sizeof *info);
  for (int i = 0; i < 4; ++i) info->utilization[i].bIsPresent = 1;
  info->utilization[0].percentage = d->utilization_gpu;
  info->utilization[1].percentage = d->utilization_mem;
  return NVML_SUCCESS;
}
// The deprecated names of nvmlDeviceGetPerformanceState and of the power
// management query; the card answers both.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t* pState) {
  return nvmlDeviceGetPerformanceState(device, pState);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, nvmlEnableState_t* mode) {
  NVML_DEV(device);
  NVML_ARG(mode);
  *mode = NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}
// AC power: a simulated board is on the wall (NVML_POWER_SOURCE_AC = 0, the card's answer).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t* powerSource) {
  NVML_DEV(device);
  NVML_ARG(powerSource);
  *powerSource = static_cast<nvmlPowerSource_t>(0);
  return NVML_SUCCESS;
}
#if defined(VGPU_NVML_13_0)
// PowerMizer, the GeForce driver's power management preference: the card supports
// adaptive, prefer-maximum-performance and auto (mask 0x7) and runs adaptive (0).
// The mode is the one nvmlDeviceSetPowerMizerMode_v1 recorded.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPowerMizerMode_v1(nvmlDevice_t device, nvmlDevicePowerMizerModes_v1_t* mode) {
  NVML_DEV(device);
  NVML_ARG(mode);
  if (!consumer(*d)) return NVML_ERROR_NOT_SUPPORTED;
  mode->currentMode = static_cast<unsigned>(setting_int(d->uuid, Life::Volatile, "powermizer_mode", 0));
  mode->supportedPowerMizerModes = 0x7;
  return NVML_SUCCESS;
}
#endif

/* ---- PCIe ---- */

namespace {
// A PCIe generation as NVML's link speed, in Mb/s of raw rate.
unsigned pcie_mbps(unsigned gen) {
  static const unsigned kRate[] = {0, 2500, 5000, 8000, 16000, 32000, 64000};
  return gen < 7 ? kRate[gen] : 0;
}
}  // namespace
// The link's current speed in Mb/s (the card, at idle on a Gen3 link: 8000).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int* pcieSpeed) {
  REQUIRE_INIT();
  NVML_ARG(pcieSpeed);
  unsigned int gen = 0;
  const nvmlReturn_t rc = pcie_link(device, &gen, true, true);
  if (rc != NVML_SUCCESS) return rc;
  *pcieSpeed = pcie_mbps(gen);
  return *pcieSpeed ? NVML_SUCCESS : NVML_ERROR_NOT_SUPPORTED;
}
// The enumerator for the link's maximum speed: 1 for 2.5 GT/s up to 6 for 64.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int* maxSpeed) {
  REQUIRE_INIT();
  NVML_ARG(maxSpeed);
  return pcie_link(device, maxSpeed, true, false);
}
// The generation the GPU itself supports; the same here as the link's maximum
// (the card, in a Gen3 slot, answers 4 here and 3 for the link).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* gen) {
  REQUIRE_INIT();
  NVML_ARG(gen);
  return pcie_link(device, gen, true, false);
}
// The replay counter is the PCIe transport counter nvmlDeviceGetFieldValues
// reports (vgpu/ras.hpp), zero until a replay is injected.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int* value) {
  NVML_DEV(device);
  NVML_ARG(value);
  uint64_t n = 0;
  try {
    n = vgpu::ras::read(d->uuid).since_load.pcie[static_cast<uint32_t>(vgpu::ras::Pcie::Replay)];
  } catch (const std::exception&) {
  }
  *value = static_cast<unsigned>(n);
  return NVML_SUCCESS;
}

/* ---- the host's CPUs and memory nodes ---- */

namespace {
// The CPUs this process may run on, the way `vgpu smi topo` prints them: the part
// of a topology the simulator does not have to model.
bool allowed_cpus(std::vector<unsigned long>* words) {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof set, &set) != 0) return false;
  const unsigned bits = 8 * sizeof(unsigned long);
  words->assign(CPU_SETSIZE / bits, 0ul);
  for (int c = 0; c < CPU_SETSIZE; ++c)
    if (CPU_ISSET(c, &set)) (*words)[c / bits] |= 1ul << (c % bits);
  return true;
}
int numa_node_count() {
  int n = 0;
  std::error_code ec;
  for (const auto& e : std::filesystem::directory_iterator("/sys/devices/system/node", ec)) {
    const std::string name = e.path().filename().string();
    if (name.rfind("node", 0) == 0 && name.size() > 4 && name.find_first_not_of("0123456789", 4) == std::string::npos)
      ++n;
  }
  return n;
}
}  // namespace
// The ideal CPUs for a GPU: all the CPUs the process may use. The card (WSL) has
// no affinity to report; on Linux a GPU's NUMA node holds them.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize,
                                                  unsigned long* cpuSet) {
  NVML_DEV(device);
  if (cpuSetSize == 0 || !cpuSet) return NVML_ERROR_INVALID_ARGUMENT;
  std::vector<unsigned long> words;
  if (!allowed_cpus(&words)) return NVML_ERROR_NOT_SUPPORTED;
  for (unsigned i = 0; i < cpuSetSize; ++i) cpuSet[i] = i < words.size() ? words[i] : 0ul;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize,
                                                             unsigned long* cpuSet, nvmlAffinityScope_t scope) {
  if (scope != NVML_AFFINITY_SCOPE_NODE && scope != NVML_AFFINITY_SCOPE_SOCKET) {
    REQUIRE_INIT();
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  return nvmlDeviceGetCpuAffinity(device, cpuSetSize, cpuSet);
}
// One NUMA node (node 0) where the host has exactly one; with several, which one
// holds a simulated GPU is not something a simulator can say, as `vgpu smi topo`.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize,
                                                     unsigned long* nodeSet, nvmlAffinityScope_t scope) {
  NVML_DEV(device);
  if (nodeSetSize == 0 || !nodeSet || (scope != NVML_AFFINITY_SCOPE_NODE && scope != NVML_AFFINITY_SCOPE_SOCKET))
    return NVML_ERROR_INVALID_ARGUMENT;
  if (numa_node_count() != 1) return NVML_ERROR_NOT_SUPPORTED;
  for (unsigned i = 0; i < nodeSetSize; ++i) nodeSet[i] = i == 0 ? 1ul : 0ul;
  return NVML_SUCCESS;
}
#if defined(VGPU_NVML_12_4)
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNumaNodeId(nvmlDevice_t device, unsigned int* node) {
  NVML_DEV(device);
  NVML_ARG(node);
  if (numa_node_count() != 1) return NVML_ERROR_NOT_SUPPORTED;
  *node = 0;
  return NVML_SUCCESS;
}
#endif

/* ---- queries the card and the model have no answer for ---- */

// The card answers NOT_SUPPORTED to each of these: the frequency monitor needs
// root and Ampere driver internals, the bridge chip is for multi-GPU boards, the
// GPU operation mode for Kepler Tesla boards, the VBIOS flush time for the black
// box recorder.
VGPU_NVML_NS_DEV(nvmlDeviceGetClkMonStatus, device, (nvmlDevice_t device, nvmlClkMonStatus_t*))
VGPU_NVML_NS_DEV(nvmlDeviceGetBridgeChipInfo, device, (nvmlDevice_t device, nvmlBridgeChipHierarchy_t*))
VGPU_NVML_NS_DEV(nvmlDeviceGetGpuOperationMode, device,
                 (nvmlDevice_t device, nvmlGpuOperationMode_t*, nvmlGpuOperationMode_t*))
#if defined(VGPU_NVML_12_4)
VGPU_NVML_NS_DEV(nvmlDeviceGetLastBBXFlushTime, device,
                 (nvmlDevice_t device, unsigned long long*, unsigned long*))
#endif
#if defined(VGPU_NVML_12_8)
// The Windows driver model: a Linux-only machine has none (nvmlDeviceGetDriverModel
// says so). The card is WSL and answers WDDM.
VGPU_NVML_NS_DEV(nvmlDeviceGetDriverModel_v2, device,
                 (nvmlDevice_t device, nvmlDriverModel_t*, nvmlDriverModel_t*))
VGPU_NVML_NS_DEV(nvmlDeviceGetDramEncryptionMode, device,
                 (nvmlDevice_t device, nvmlDramEncryptionInfo_t*, nvmlDramEncryptionInfo_t*))
#endif
#if defined(VGPU_NVML_13_0)
// Addressing mode (HMM or ATS): the version word is checked first.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetAddressingMode(nvmlDevice_t device, nvmlDeviceAddressingMode_t* mode) {
  NVML_DEV(device);
  NVML_ARG(mode);
  NVML_VERSION(mode, nvmlDeviceAddressingMode_v1);
  return NVML_ERROR_NOT_SUPPORTED;
}
#endif
