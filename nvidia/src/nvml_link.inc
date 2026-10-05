// NVML: NVLink, fabric, C2C, topology, P2P, drain state, system events. See nvml_common.inc.
//
// What a simulated machine is: every GPU on the one host bridge (`vgpu smi topo`
// prints PHB for every pair, and no profile measures an NVLink that is cabled),
// so
//   * NVLink ports exist where the part has them (the profile's nvlink_links and
//     nvlink_version, datasheet) but none is connected: every link is inactive,
//     and what only a live link can tell (remote end, error and utilization
//     counters) is NOT_SUPPORTED, not a made-up zero;
//   * the topology between two different GPUs is the host bridge, between a GPU
//     and itself nothing (the card, asked about itself, says NOT_SUPPORTED);
//   * there is no fabric manager and no NVSwitch, so the fabric state is
//     "not supported", the state a GPU outside an NVLink fabric reports.
// The card, a GeForce RTX 3060, has no NVLink at all: every NVLink query on it is
// NOT_SUPPORTED, which is what a profile with no ports answers here.

namespace {

// NVML's NVLink version enumerator for a generation (3 is 3.0, 4 is 4.0, 5 is 5.0).
unsigned nvlink_version_enum(unsigned generation) {
  return generation == 3 ? 4 : generation == 4 ? 6 : generation == 5 ? 7 : 0;
}

// Device and link validation shared by the per-link queries: no ports at all is
// NOT_SUPPORTED, a port number past the last INVALID_ARGUMENT.
nvmlReturn_t link_checked(const vgpu::telemetry::DeviceSample& d, unsigned link) {
  const unsigned ports = nvml_facts(d).nvlink_links;
  if (!ports) return NVML_ERROR_NOT_SUPPORTED;
  return link < ports ? NVML_SUCCESS : NVML_ERROR_INVALID_ARGUMENT;
}

// The topology between two devices, by index; -1 for a device and itself.
int topology_between(unsigned a, unsigned b) { return a == b ? -1 : NVML_TOPOLOGY_HOSTBRIDGE; }

}  // namespace

/* ---- NVLink ---- */

// No link is connected: inactive (NVML_FEATURE_DISABLED).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive) {
  NVML_DEV(device);
  NVML_ARG(isActive);
  if (const nvmlReturn_t rc = link_checked(*d, link); rc != NVML_SUCCESS) return rc;
  *isActive = NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int* version) {
  NVML_DEV(device);
  NVML_ARG(version);
  if (const nvmlReturn_t rc = link_checked(*d, link); rc != NVML_SUCCESS) return rc;
  *version = nvlink_version_enum(nvml_facts(*d).nvlink_version);
  return NVML_SUCCESS;
}
// What the port can do: peer-to-peer and peer atomics over NVLink on every part
// that has the port; access to system memory over it only where the CPU is
// reached over NVLink (the GH200's C2C is a different link: its ports are the
// GPU-to-GPU ones); no SLI bridge. The port is valid.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
                                                       nvmlNvLinkCapability_t capability, unsigned int* capResult) {
  NVML_DEV(device);
  NVML_ARG(capResult);
  const int cap = static_cast<int>(capability);
  if (cap < 0 || cap >= NVML_NVLINK_CAP_COUNT) return NVML_ERROR_INVALID_ARGUMENT;
  if (const nvmlReturn_t rc = link_checked(*d, link); rc != NVML_SUCCESS) return rc;
  *capResult = (cap == NVML_NVLINK_CAP_P2P_SUPPORTED || cap == NVML_NVLINK_CAP_P2P_ATOMICS || cap == NVML_NVLINK_CAP_VALID) ? 1 : 0;
  return NVML_SUCCESS;
}
// Error counters, the remote end and the (deprecated, "no longer supported")
// utilization counters need a connected link; none is.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link,
                                                         nvmlNvLinkErrorCounter_t counter,
                                                         unsigned long long* counterValue) {
  NVML_DEV(device);
  NVML_ARG(counterValue);
  const int c = static_cast<int>(counter);
  if (c < 0 || c >= NVML_NVLINK_ERROR_COUNT) return NVML_ERROR_INVALID_ARGUMENT;
  if (const nvmlReturn_t rc = link_checked(*d, link); rc != NVML_SUCCESS) return rc;
  return NVML_ERROR_NOT_SUPPORTED;
}
// The remote end of an unconnected link is unknown (NVML_NVLINK_DEVICE_TYPE_UNKNOWN).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link,
                                                             nvmlIntNvLinkDeviceType_t* type) {
  NVML_DEV(device);
  NVML_ARG(type);
  if (const nvmlReturn_t rc = link_checked(*d, link); rc != NVML_SUCCESS) return rc;
  *type = NVML_NVLINK_DEVICE_TYPE_UNKNOWN;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) {
  NVML_DEV(device);
  NVML_ARG(pci);
  if (const nvmlReturn_t rc = link_checked(*d, link); rc != NVML_SUCCESS) return rc;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) {
  return nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, pci);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link,
                                                               unsigned int counter, unsigned long long* rxcounter,
                                                               unsigned long long* txcounter) {
  NVML_DEV(device);
  if (!rxcounter || !txcounter || counter > 1) return NVML_ERROR_INVALID_ARGUMENT;
  if (const nvmlReturn_t rc = link_checked(*d, link); rc != NVML_SUCCESS) return rc;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link,
                                                               unsigned int counter,
                                                               nvmlNvLinkUtilizationControl_t* control) {
  NVML_DEV(device);
  if (!control || counter > 1) return NVML_ERROR_INVALID_ARGUMENT;
  if (const nvmlReturn_t rc = link_checked(*d, link); rc != NVML_SUCCESS) return rc;
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceFreezeNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link,
                                                                  unsigned int counter, nvmlEnableState_t freeze) {
  NVML_DEV(device);
  const int f = static_cast<int>(freeze);
  if (counter > 1 || f < 0 || f > 1) return NVML_ERROR_INVALID_ARGUMENT;
  if (const nvmlReturn_t rc = link_checked(*d, link); rc != NVML_SUCCESS) return rc;
  return NVML_ERROR_NOT_SUPPORTED;   // deprecated: "no longer supported"
}
#if defined(VGPU_NVML_13_0)
// NVLink encryption and the NVLink firmware: neither is simulated.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkInfo(nvmlDevice_t device, nvmlNvLinkInfo_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  if (info->version != nvmlNvLinkInfo_v1 && info->version != nvmlNvLinkInfo_v2)
    return NVML_ERROR_ARGUMENT_VERSION_MISMATCH;
  if (!nvml_facts(*d).nvlink_links) return NVML_ERROR_NOT_SUPPORTED;
  const unsigned version = info->version;
  std::memset(info, 0, version == nvmlNvLinkInfo_v1 ? sizeof(nvmlNvLinkInfo_v1_t) : sizeof(nvmlNvLinkInfo_v2_t));
  info->version = version;
  return NVML_SUCCESS;
}
#endif
#if defined(VGPU_NVML_12_8)
// The reduced-bandwidth mode belongs to a fabric that is not simulated.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvlinkBwMode(nvmlDevice_t device, nvmlNvlinkGetBwMode_t* mode) {
  NVML_DEV(device);
  NVML_ARG(mode);
  NVML_VERSION(mode, nvmlNvlinkGetBwMode_v1);
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetNvlinkSupportedBwModes(nvmlDevice_t device, nvmlNvlinkSupportedBwModes_t* modes) {
  NVML_DEV(device);
  NVML_ARG(modes);
  NVML_VERSION(modes, nvmlNvlinkSupportedBwModes_v1);
  return NVML_ERROR_NOT_SUPPORTED;
}
#endif
#if defined(VGPU_NVML_12_4)
// The global NVLink bandwidth mode is the fabric manager's.
VGPU_EXPORT nvmlReturn_t nvmlSystemGetNvlinkBwMode(unsigned int* nvlinkBwMode) {
  REQUIRE_INIT();
  NVML_ARG(nvlinkBwMode);
  return NVML_ERROR_NOT_SUPPORTED;
}
#endif

/* ---- the fabric ---- */

// No fabric: state NVML_GPU_FABRIC_STATE_NOT_SUPPORTED (0), no cluster, no clique. The
// card answers success with exactly this.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpuFabricInfo(nvmlDevice_t device, nvmlGpuFabricInfo_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  std::memset(info, 0, sizeof *info);
  return NVML_SUCCESS;
}
#if defined(VGPU_NVML_12_4)
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  bool ok = false;
#if defined(nvmlGpuFabricInfo_v2)
  ok = ok || info->version == nvmlGpuFabricInfo_v2;
#endif
#if defined(nvmlGpuFabricInfo_v3)
  ok = ok || info->version == nvmlGpuFabricInfo_v3;
#endif
  if (!ok) return NVML_ERROR_ARGUMENT_VERSION_MISMATCH;
  const unsigned version = info->version;
  // Zero what the caller's version of the struct holds, not what the newest one does.
  size_t size = sizeof *info;
#if defined(nvmlGpuFabricInfo_v2)
  if (version == nvmlGpuFabricInfo_v2) size = sizeof(nvmlGpuFabricInfo_v2_t);
#endif
  std::memset(info, 0, size);
  info->version = version;
  return NVML_SUCCESS;
}
#endif

/* ---- topology and peer-to-peer ---- */

// Two GPUs of the simulated machine share a host bridge. A GPU and itself: the
// card says NOT_SUPPORTED.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2,
                                                             nvmlGpuTopologyLevel_t* pathInfo) {
  REQUIRE_INIT();
  refresh();
  unsigned a = 0, b = 0;
  if (!index_of(device1, &a)) return bad(device1);
  if (!index_of(device2, &b)) return bad(device2);
  NVML_ARG(pathInfo);
  const int level = topology_between(a, b);
  if (level < 0) return NVML_ERROR_NOT_SUPPORTED;
  *pathInfo = static_cast<nvmlGpuTopologyLevel_t>(level);
  return NVML_SUCCESS;
}
// The GPUs at a level or closer: with the host bridge between every pair, every
// other GPU from the host-bridge level up, none below it. A count of 0 asks how
// many; otherwise the array is filled up to the count. The levels are NVML's
// numbers: 0, 10, 20, 30, 40, 50.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level,
                                                          unsigned int* count, nvmlDevice_t* deviceArray) {
  REQUIRE_INIT();
  refresh();
  unsigned self = 0;
  if (!index_of(device, &self)) return bad(device);
  const int l = static_cast<int>(level);
  if (!count || l < 0 || l > 50 || l % 10 != 0 || (*count != 0 && !deviceArray)) return NVML_ERROR_INVALID_ARGUMENT;
  std::vector<nvmlDevice_t> found;
  if (l >= NVML_TOPOLOGY_HOSTBRIDGE)
    for (unsigned i = 0; i < g_snap.device_count; ++i)
      if (i != self && !g_lost[i]) found.push_back(handle_for(i));
  if (*count == 0) {
    *count = static_cast<unsigned>(found.size());
    return NVML_SUCCESS;
  }
  const unsigned n = std::min<unsigned>(*count, static_cast<unsigned>(found.size()));
  for (unsigned i = 0; i < n; ++i) deviceArray[i] = found[i];
  *count = n;
  return NVML_SUCCESS;
}
// The GPUs near a CPU: all of them, for a CPU the process may run on.
VGPU_EXPORT nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int* count,
                                                     nvmlDevice_t* deviceArray) {
  REQUIRE_INIT();
  refresh();
  if (!count || (*count != 0 && !deviceArray)) return NVML_ERROR_INVALID_ARGUMENT;
  std::vector<unsigned long> words;
  if (!allowed_cpus(&words)) return NVML_ERROR_NOT_SUPPORTED;
  const unsigned bits = 8 * sizeof(unsigned long);
  if (cpuNumber >= words.size() * bits) return NVML_ERROR_INVALID_ARGUMENT;
  std::vector<nvmlDevice_t> found;
  if (words[cpuNumber / bits] >> (cpuNumber % bits) & 1ul)
    for (unsigned i = 0; i < g_snap.device_count; ++i)
      if (!g_lost[i]) found.push_back(handle_for(i));
  if (*count == 0) {
    *count = static_cast<unsigned>(found.size());
    return NVML_SUCCESS;
  }
  const unsigned n = std::min<unsigned>(*count, static_cast<unsigned>(found.size()));
  for (unsigned i = 0; i < n; ++i) deviceArray[i] = found[i];
  *count = n;
  return NVML_SUCCESS;
}
// Peer-to-peer capabilities. A GeForce card's answers are the card's own
// (measured with the card asked about itself: read and write CHIPSET_NOT_SUPPORTED,
// NVLink, atomics and PCI NOT_SUPPORTED, the unknown index OK -- consumer
// boards do not do P2P over PCIe); a datacenter part does it over PCIe (read, write,
// atomics, PCI OK) and not over NVLink, which is inactive.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2,
                                                nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t* p2pStatus) {
  REQUIRE_INIT();
  refresh();
  const auto* a = sample(device1);
  if (!a) return bad(device1);
  if (!sample(device2)) return bad(device2);
  NVML_ARG(p2pStatus);
  const int idx = static_cast<int>(p2pIndex);
  if (idx < 0 || idx > 5) return NVML_ERROR_INVALID_ARGUMENT;
  static const int kConsumer[6] = {1, 1, 5, 5, 5, 0};
  static const int kDatacenter[6] = {0, 0, 5, 0, 0, 0};
  *p2pStatus = static_cast<nvmlGpuP2PStatus_t>(consumer(*a) ? kConsumer[idx] : kDatacenter[idx]);
  return NVML_SUCCESS;
}

/* ---- drain state, hot removal ---- */

namespace {
// A device named by its PCI address, among the ones NVML lists.
const vgpu::telemetry::DeviceSample* by_pci(const nvmlPciInfo_t* pci, unsigned* index) {
  if (!pci) return nullptr;
  char text[40];
  std::snprintf(text, sizeof text, "%08x:%02x:%02x.0", pci->domain, pci->bus, pci->device);
  for (unsigned i = 0; i < g_snap.device_count; ++i)
    if (same_bus_id(g_snap.devices[i].bus_id, text)) {
      if (index) *index = i;
      return &g_snap.devices[i];
    }
  return nullptr;
}
}  // namespace
// A GPU drains (stops taking new work, ahead of removal) when nvmlDeviceModifyDrainState
// says so; it does not by default.
VGPU_EXPORT nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t* currentState) {
  REQUIRE_INIT();
  refresh();
  if (!pciInfo || !currentState) return NVML_ERROR_INVALID_ARGUMENT;
  const auto* d = by_pci(pciInfo, nullptr);
  if (!d) return NVML_ERROR_NOT_FOUND;   // measured: an address that is no GPU
  *currentState = setting_int(d->uuid, Life::Volatile, "drain", 0) ? NVML_FEATURE_ENABLED : NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}

/* ---- system events ----
 *
 * What a system event set watches for is the driver binding and unbinding GPUs. A
 * simulated machine's driver never does either on its own, so a wait finds
 * nothing and times out, as it does on a machine where nothing happens.
 */
#if defined(VGPU_NVML_12_9)
struct nvmlSystemEventSet_st {
  std::mutex mu;
  unsigned long long types = 0;
};

VGPU_EXPORT nvmlReturn_t nvmlSystemEventSetCreate(nvmlSystemEventSetCreateRequest_t* request) {
  REQUIRE_INIT();
  NVML_ARG(request);
  NVML_VERSION(request, nvmlSystemEventSetCreateRequest_v1);
  request->set = new nvmlSystemEventSet_st();
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlSystemEventSetFree(nvmlSystemEventSetFreeRequest_t* request) {
  REQUIRE_INIT();
  NVML_ARG(request);
  NVML_VERSION(request, nvmlSystemEventSetFreeRequest_v1);
  if (!request->set) return NVML_ERROR_INVALID_ARGUMENT;
  delete request->set;
  request->set = nullptr;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlSystemRegisterEvents(nvmlSystemRegisterEventRequest_t* request) {
  REQUIRE_INIT();
  NVML_ARG(request);
  NVML_VERSION(request, nvmlSystemRegisterEventRequest_v1);
  if (!request->set) return NVML_ERROR_INVALID_ARGUMENT;
  const unsigned long long all = nvmlSystemEventTypeGpuDriverUnbind | nvmlSystemEventTypeGpuDriverBind;
  if (request->eventTypes & ~all) return NVML_ERROR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lock(request->set->mu);
  request->set->types |= request->eventTypes;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlSystemEventSetWait(nvmlSystemEventSetWaitRequest_t* request) {
  REQUIRE_INIT();
  NVML_ARG(request);
  NVML_VERSION(request, nvmlSystemEventSetWaitRequest_v1);
  if (!request->set || (request->dataSize && !request->data)) return NVML_ERROR_INVALID_ARGUMENT;
  request->numEvent = 0;
  if (request->timeoutms) std::this_thread::sleep_for(std::chrono::milliseconds(request->timeoutms));
  return NVML_ERROR_TIMEOUT;
}
#endif

/* ---- the driver ---- */

#if defined(VGPU_NVML_12_8)
// The driver's branch, "r<major>_00" as NVIDIA names a release branch (r550_00),
// from the session's driver version. The card (a development driver) says r596_25.
VGPU_EXPORT nvmlReturn_t nvmlSystemGetDriverBranch(nvmlSystemDriverBranchInfo_t* branchInfo, unsigned int length) {
  REQUIRE_INIT();
  NVML_ARG(branchInfo);
  NVML_VERSION(branchInfo, nvmlSystemDriverBranchInfo_v1);
  refresh();
  unsigned major = 0;
  std::sscanf(g_snap.driver_version, "%u", &major);
  char text[32];
  std::snprintf(text, sizeof text, "r%u_00", major);
  const size_t room = length < sizeof branchInfo->branch ? length : sizeof branchInfo->branch;
  if (std::strlen(text) + 1 > room) return NVML_ERROR_INSUFFICIENT_SIZE;
  std::memcpy(branchInfo->branch, text, std::strlen(text) + 1);
  return NVML_SUCCESS;
}
#endif
