// NVML: vGPU, units, GPM, confidential computing, PRM, power profiles. See nvml_common.inc for the conventions.
//
// Every GPU VirtualGPU simulates is a bare-metal GPU in a machine with no vGPU
// manager, no S-class unit, no confidential-computing hardware, no GPM counter
// model (the simulator is functional, it has no timing model to count) and no
// PRM access. This file answers the whole of that API the way NVIDIA's NVML
// answers on such a GPU. The sources of the answers, in order:
//
//   * measurements on the RTX 3060 through NVIDIA's own NVML 13.0 (WSL driver
//     596.36), noted where each is used. The probe passed the real library a
//     valid device and correctly versioned structs, then each pointer null in
//     turn, and a wrong version word. It matters because the real library is
//     not uniform: some entry points say NOT_SUPPORTED before they look at a
//     null pointer, some after, and a few validate a version word before the
//     feature, a few after. Each function below checks in the order the card did.
//   * where a function takes something the card could not be asked about (a
//     handle that does not exist there, or a setter, which must never be run on
//     the card), the nvml.h comment of the function and the order its measured
//     sibling uses. Those are marked "doc".
//
// The card is a GeForce. The profiles here include datacenter parts (T4, A100,
// H100) for which no oracle exists; they answer exactly like the GeForce,
// because without a vGPU manager, a CC-enabled platform or an S-class chassis
// the real driver does too. GPM and the power-profile queries are the exception:
// Hopper and later report them supported on real hardware. The functional
// simulator has no counters or power-management model to put behind them, so
// they say not supported, and never a plausible wrong number.
//
// Where the real library crashes on a null handle (the vGPU entry points
// dereference it unchecked) this one answers INVALID_ARGUMENT, as nvml.h
// documents.

#include <new>
#include <set>

namespace {

// ---- helpers (prefixed vgpu_ so no other group's names collide) ----------------

// vGPU instance ids are handed out from 1 and 0 is "none". Nothing on this
// machine creates one, so 0 is invalid and any other id is unknown.
// Measured: every nvmlVgpuInstance* query with id 0 is INVALID_ARGUMENT and with
// id 1 is NOT_FOUND on the card.
nvmlReturn_t vgpu_unknown_instance(nvmlVgpuInstance_t id) {
  return id == 0 ? NVML_ERROR_INVALID_ARGUMENT : NVML_ERROR_NOT_FOUND;
}

// Sizes the real library insists on, measured on the card: a smaller buffer is
// INSUFFICIENT_SIZE before the id is even looked at.
constexpr unsigned vgpu_kTypeNameSize = 64;      // nvmlVgpuTypeGetName / GetClass (NVML_DEVICE_NAME_BUFFER_SIZE)
constexpr unsigned vgpu_kLicenseSize = 128;      // nvmlVgpuTypeGetLicense (NVML_GRID_LICENSE_BUFFER_SIZE)
constexpr unsigned vgpu_kVmIdSize = 80;          // nvmlVgpuInstanceGetVmID (NVML_DEVICE_UUID_BUFFER_SIZE)
constexpr unsigned vgpu_kMetadataSize = 468;     // nvmlVgpuInstanceGetMetadata: 0x1d4 bytes, written to *bufferSize
constexpr unsigned vgpu_kAccountingPids = 4000;  // nvmlVgpuInstanceGetAccountingPids: 0xfa0, the capacity it reports

// GPM samples. nvmlGpmSampleAlloc hands out an opaque buffer; the registry lets
// nvmlGpmSampleFree and the sample readers tell one of ours from a stray pointer.
std::mutex vgpu_gpm_mu;
std::set<const void*>& vgpu_gpm_samples() {
  static std::set<const void*> live;
  return live;
}
bool vgpu_gpm_known(const void* sample) {
  std::lock_guard<std::mutex> lock(vgpu_gpm_mu);
  return sample && vgpu_gpm_samples().count(sample) != 0;
}

}  // namespace

// The opaque type behind nvmlGpmSample_t. NVML's callers never look inside it.
struct nvmlGpmSample_st {
  unsigned int filled;  // would become 1 on a successful nvmlGpmSampleGet; that never succeeds here
};

/* ======================================================================== *
 * vGPU instances. No vGPU manager, so no instance exists.
 *
 * Order measured on the card for the plain getters: a null output pointer is
 * INVALID_ARGUMENT, before the id is looked at; then id 0 is INVALID_ARGUMENT
 * and any other id NOT_FOUND.
 * ======================================================================== */

#define VGPU_INSTANCE_GET(name, type)                                         \
  VGPU_EXPORT nvmlReturn_t name(nvmlVgpuInstance_t vgpuInstance, type* out) { \
    REQUIRE_INIT();                                                           \
    NVML_ARG(out);                                                            \
    return vgpu_unknown_instance(vgpuInstance);                               \
  }

VGPU_INSTANCE_GET(nvmlVgpuInstanceGetAccountingMode, nvmlEnableState_t)
VGPU_INSTANCE_GET(nvmlVgpuInstanceGetEccMode, nvmlEnableState_t)
VGPU_INSTANCE_GET(nvmlVgpuInstanceGetEncoderCapacity, unsigned int)
VGPU_INSTANCE_GET(nvmlVgpuInstanceGetFBCStats, nvmlFBCStats_t)
VGPU_INSTANCE_GET(nvmlVgpuInstanceGetFbUsage, unsigned long long)
VGPU_INSTANCE_GET(nvmlVgpuInstanceGetFrameRateLimit, unsigned int)
VGPU_INSTANCE_GET(nvmlVgpuInstanceGetGpuInstanceId, unsigned int)
VGPU_INSTANCE_GET(nvmlVgpuInstanceGetLicenseInfo, nvmlVgpuLicenseInfo_t)
VGPU_INSTANCE_GET(nvmlVgpuInstanceGetLicenseInfo_v2, nvmlVgpuLicenseInfo_t)
VGPU_INSTANCE_GET(nvmlVgpuInstanceGetLicenseStatus, unsigned int)
VGPU_INSTANCE_GET(nvmlVgpuInstanceGetType, nvmlVgpuTypeId_t)

VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int,
                                                            nvmlAccountingStats_t* stats) {
  REQUIRE_INIT();
  NVML_ARG(stats);
  return vgpu_unknown_instance(vgpuInstance);
}

// Measured: with the pids array null and *count 0 the real library answers
// INSUFFICIENT_SIZE and reports its capacity (4000) without looking at the id,
// even for id 0; with the array null and *count non-zero it is INVALID_ARGUMENT.
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int* count,
                                                           unsigned int* pids) {
  REQUIRE_INIT();
  NVML_ARG(count);
  if (!pids) {
    if (*count != 0) return NVML_ERROR_INVALID_ARGUMENT;
    *count = vgpu_kAccountingPids;
    return NVML_ERROR_INSUFFICIENT_SIZE;
  }
  return vgpu_unknown_instance(vgpuInstance);
}

// Session lists: the count is mandatory, the session array may be null (a
// count-only query); measured on the card.
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount,
                                                            nvmlEncoderSessionInfo_t*) {
  REQUIRE_INIT();
  NVML_ARG(sessionCount);
  return vgpu_unknown_instance(vgpuInstance);
}
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount,
                                                        nvmlFBCSessionInfo_t*) {
  REQUIRE_INIT();
  NVML_ARG(sessionCount);
  return vgpu_unknown_instance(vgpuInstance);
}
// All three pointers are mandatory (measured).
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount,
                                                         unsigned int* averageFps, unsigned int* averageLatency) {
  REQUIRE_INIT();
  if (!sessionCount || !averageFps || !averageLatency) return NVML_ERROR_INVALID_ARGUMENT;
  return vgpu_unknown_instance(vgpuInstance);
}

// Strings. The id is looked up after the null check; there is no size check for
// these three (measured: id 1 with a 1-byte buffer is NOT_FOUND, not
// INSUFFICIENT_SIZE).
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char* uuid, unsigned int) {
  REQUIRE_INIT();
  NVML_ARG(uuid);
  return vgpu_unknown_instance(vgpuInstance);
}
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char* mdevUuid, unsigned int) {
  REQUIRE_INIT();
  NVML_ARG(mdevUuid);
  return vgpu_unknown_instance(vgpuInstance);
}
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char* version,
                                                            unsigned int) {
  REQUIRE_INIT();
  NVML_ARG(version);
  return vgpu_unknown_instance(vgpuInstance);
}
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char* vgpuPciId,
                                                     unsigned int* length) {
  REQUIRE_INIT();
  if (!vgpuPciId || !length) return NVML_ERROR_INVALID_ARGUMENT;
  return vgpu_unknown_instance(vgpuInstance);
}
// Measured: unlike the other strings this one is INSUFFICIENT_SIZE when the
// buffer is under 80 bytes, even for id 0, before the pointers and the id.
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char* vmId, unsigned int size,
                                                 nvmlVgpuVmIdType_t* vmIdType) {
  REQUIRE_INIT();
  if (size < vgpu_kVmIdSize) return NVML_ERROR_INSUFFICIENT_SIZE;
  if (!vmId || !vmIdType) return NVML_ERROR_INVALID_ARGUMENT;
  return vgpu_unknown_instance(vgpuInstance);
}
// Measured: the size the real library wants (468 bytes) is reported without
// looking at the id; a null buffer is only an error when *bufferSize is not 0
// (the sizing call is metadata == NULL, *bufferSize == 0).
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t* vgpuMetadata,
                                                     unsigned int* bufferSize) {
  REQUIRE_INIT();
  NVML_ARG(bufferSize);
  if (!vgpuMetadata && *bufferSize != 0) return NVML_ERROR_INVALID_ARGUMENT;
  if (*bufferSize < vgpu_kMetadataSize) {
    *bufferSize = vgpu_kMetadataSize;
    return NVML_ERROR_INSUFFICIENT_SIZE;
  }
  return vgpu_unknown_instance(vgpuInstance);
}

#if VGPU_NVML_12_8
// Measured: the id is checked and nothing else, not even a null pointer or the
// version word (id 1 with a bad version and a null struct is still NOT_FOUND).
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetRuntimeStateSize(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuRuntimeState_t*) {
  REQUIRE_INIT();
  return vgpu_unknown_instance(vgpuInstance);
}
#endif
#if VGPU_NVML_12_4
// Measured: a null struct is INVALID_ARGUMENT, then the version word (25 even
// for id 0), then the id.
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceGetPlacementId(nvmlVgpuInstance_t vgpuInstance,
                                                        nvmlVgpuPlacementId_t* pPlacement) {
  REQUIRE_INIT();
  NVML_ARG(pPlacement);
  NVML_VERSION(pPlacement, nvmlVgpuPlacementId_v1);
  return vgpu_unknown_instance(vgpuInstance);
}
#endif

// Setters on an instance that cannot exist: the id is the handle, and a handle
// is validated first (doc; setters are never run on the card).
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int) {
  REQUIRE_INIT();
  return vgpu_unknown_instance(vgpuInstance);
}
VGPU_EXPORT nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance) {
  REQUIRE_INIT();
  return vgpu_unknown_instance(vgpuInstance);
}

/* ======================================================================== *
 * vGPU types. No vGPU manager, so no type id is valid (the card says
 * INVALID_ARGUMENT for ids 0 and 1 alike).
 * ======================================================================== */

#define VGPU_TYPE_INVALID(name, params) \
  VGPU_EXPORT nvmlReturn_t name params { \
    REQUIRE_INIT();                      \
    return NVML_ERROR_INVALID_ARGUMENT;  \
  }

VGPU_TYPE_INVALID(nvmlVgpuTypeGetCapabilities, (nvmlVgpuTypeId_t, nvmlVgpuCapability_t, unsigned int*))
VGPU_TYPE_INVALID(nvmlVgpuTypeGetDeviceID, (nvmlVgpuTypeId_t, unsigned long long*, unsigned long long*))
VGPU_TYPE_INVALID(nvmlVgpuTypeGetFrameRateLimit, (nvmlVgpuTypeId_t, unsigned int*))
VGPU_TYPE_INVALID(nvmlVgpuTypeGetFramebufferSize, (nvmlVgpuTypeId_t, unsigned long long*))
VGPU_TYPE_INVALID(nvmlVgpuTypeGetGpuInstanceProfileId, (nvmlVgpuTypeId_t, unsigned int*))
VGPU_TYPE_INVALID(nvmlVgpuTypeGetMaxInstancesPerVm, (nvmlVgpuTypeId_t, unsigned int*))
VGPU_TYPE_INVALID(nvmlVgpuTypeGetNumDisplayHeads, (nvmlVgpuTypeId_t, unsigned int*))
VGPU_TYPE_INVALID(nvmlVgpuTypeGetResolution, (nvmlVgpuTypeId_t, unsigned int, unsigned int*, unsigned int*))
#if VGPU_NVML_12_4
VGPU_TYPE_INVALID(nvmlVgpuTypeGetFbReservation, (nvmlVgpuTypeId_t, unsigned long long*))
VGPU_TYPE_INVALID(nvmlVgpuTypeGetGspHeapSize, (nvmlVgpuTypeId_t, unsigned long long*))
#endif

// Name and class (measured): id 0 first, then a null size pointer, then a
// buffer shorter than 64 bytes (INSUFFICIENT_SIZE with *size set to 64, before
// the buffer pointer is even looked at), then the buffer pointer, then the type.
static nvmlReturn_t vgpu_type_string(nvmlVgpuTypeId_t vgpuTypeId, char* out, unsigned int* size) {
  REQUIRE_INIT();
  if (vgpuTypeId == 0 || !size) return NVML_ERROR_INVALID_ARGUMENT;
  if (*size < vgpu_kTypeNameSize) {
    *size = vgpu_kTypeNameSize;
    return NVML_ERROR_INSUFFICIENT_SIZE;
  }
  (void)out;  // a null buffer and an unknown type are the same answer
  return NVML_ERROR_INVALID_ARGUMENT;
}
VGPU_EXPORT nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeName, unsigned int* size) {
  return vgpu_type_string(vgpuTypeId, vgpuTypeName, size);
}
VGPU_EXPORT nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeClass, unsigned int* size) {
  return vgpu_type_string(vgpuTypeId, vgpuTypeClass, size);
}
// Measured: null buffer, then id 0, then a buffer shorter than 128 (the size is
// a plain argument here, nothing is written back), then the unknown type.
VGPU_EXPORT nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeLicenseString,
                                                unsigned int size) {
  REQUIRE_INIT();
  if (!vgpuTypeLicenseString || vgpuTypeId == 0) return NVML_ERROR_INVALID_ARGUMENT;
  if (size < vgpu_kLicenseSize) return NVML_ERROR_INSUFFICIENT_SIZE;
  return NVML_ERROR_INVALID_ARGUMENT;
}

#if VGPU_NVML_12_8
// Measured: null struct, then type id 0, then the version word, then the unknown type.
VGPU_EXPORT nvmlReturn_t nvmlVgpuTypeGetBAR1Info(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuTypeBar1Info_t* bar1Info) {
  REQUIRE_INIT();
  NVML_ARG(bar1Info);
  if (vgpuTypeId == 0) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_VERSION(bar1Info, nvmlVgpuTypeBar1Info_v1);
  return NVML_ERROR_INVALID_ARGUMENT;
}
#endif
#if VGPU_NVML_12_9
// Same order, the type id being a field of the struct (measured).
VGPU_EXPORT nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerGpuInstance(nvmlVgpuTypeMaxInstance_t* pMaxInstance) {
  REQUIRE_INIT();
  NVML_ARG(pMaxInstance);
  if (pMaxInstance->vgpuTypeId == 0) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_VERSION(pMaxInstance, nvmlVgpuTypeMaxInstance_v1);
  return NVML_ERROR_INVALID_ARGUMENT;
}
#endif
// Measured: the device is validated, then NOT_SUPPORTED, whatever the type id
// and the pointer (this one asks the device, not the type table).
VGPU_NVML_NS_DEV(nvmlVgpuTypeGetMaxInstances, device, (nvmlDevice_t device, nvmlVgpuTypeId_t, unsigned int*))

/* ======================================================================== *
 * Device-level vGPU. A vGPU host needs the vGPU manager; without it every
 * one of these is NOT_SUPPORTED (measured), in the order of checks the card
 * used, which the comments name when a null pointer or a version word is
 * looked at first.
 * ======================================================================== */

// NOT_SUPPORTED, whatever the pointers (measured: null pointers too).
VGPU_NVML_NS_DEV(nvmlDeviceGetActiveVgpus, device, (nvmlDevice_t device, unsigned int*, nvmlVgpuInstance_t*))
VGPU_NVML_NS_DEV(nvmlDeviceGetCreatableVgpus, device, (nvmlDevice_t device, unsigned int*, nvmlVgpuTypeId_t*))
VGPU_NVML_NS_DEV(nvmlDeviceGetSupportedVgpus, device, (nvmlDevice_t device, unsigned int*, nvmlVgpuTypeId_t*))
VGPU_NVML_NS_DEV(nvmlDeviceGetVgpuProcessUtilization, device,
                 (nvmlDevice_t device, unsigned long long, unsigned int*, nvmlVgpuProcessUtilizationSample_t*))
VGPU_NVML_NS_DEV(nvmlDeviceGetVgpuUtilization, device,
                 (nvmlDevice_t device, unsigned long long, nvmlValueType_t*, unsigned int*,
                  nvmlVgpuInstanceUtilizationSample_t*))
#if VGPU_NVML_12_4
VGPU_NVML_NS_DEV(nvmlDeviceGetVgpuHeterogeneousMode, device, (nvmlDevice_t device, nvmlVgpuHeterogeneousMode_t*))
// Measured: NOT_SUPPORTED before the type id, the version word and the pointer.
VGPU_NVML_NS_DEV(nvmlDeviceGetVgpuTypeCreatablePlacements, device,
                 (nvmlDevice_t device, nvmlVgpuTypeId_t, nvmlVgpuPlacementList_t*))
VGPU_NVML_NS_DEV(nvmlDeviceGetVgpuTypeSupportedPlacements, device,
                 (nvmlDevice_t device, nvmlVgpuTypeId_t, nvmlVgpuPlacementList_t*))
// Measured: a null struct is INVALID_ARGUMENT, then NOT_SUPPORTED even for a
// bad version word (this one does not look at the version).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetVgpuProcessesUtilizationInfo(nvmlDevice_t device,
                                                                   nvmlVgpuProcessesUtilizationInfo_t* vgpuProcUtilInfo) {
  NVML_DEV(device);
  NVML_ARG(vgpuProcUtilInfo);
  return NVML_ERROR_NOT_SUPPORTED;
}
// Measured: here the version word IS checked first (25), then NOT_SUPPORTED.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetVgpuInstancesUtilizationInfo(nvmlDevice_t device,
                                                                   nvmlVgpuInstancesUtilizationInfo_t* vgpuUtilInfo) {
  NVML_DEV(device);
  NVML_ARG(vgpuUtilInfo);
  NVML_VERSION(vgpuUtilInfo, nvmlVgpuInstancesUtilizationInfo_v1);
  return NVML_ERROR_NOT_SUPPORTED;
}
#endif

// A null result pointer is INVALID_ARGUMENT first, then NOT_SUPPORTED (measured).
#define VGPU_NS_AFTER_ARG(name, device, ptrtype, ptr) \
  VGPU_EXPORT nvmlReturn_t name(nvmlDevice_t device, ptrtype* ptr) { \
    NVML_DEV(device);                                                \
    NVML_ARG(ptr);                                                   \
    return NVML_ERROR_NOT_SUPPORTED;                                 \
  }
VGPU_NS_AFTER_ARG(nvmlDeviceGetHostVgpuMode, device, nvmlHostVgpuMode_t, pHostVgpuMode)
VGPU_NS_AFTER_ARG(nvmlDeviceGetVgpuSchedulerCapabilities, device, nvmlVgpuSchedulerCapabilities_t, pCapabilities)
VGPU_NS_AFTER_ARG(nvmlDeviceGetVgpuSchedulerLog, device, nvmlVgpuSchedulerLog_t, pSchedulerLog)
VGPU_NS_AFTER_ARG(nvmlDeviceGetVgpuSchedulerState, device, nvmlVgpuSchedulerGetState_t, pSchedulerState)
#if VGPU_NVML_13_2
VGPU_NS_AFTER_ARG(nvmlDeviceGetVgpuSchedulerLog_v2, device, nvmlVgpuSchedulerLogInfo_v2_t, pSchedulerLogInfo)
VGPU_NS_AFTER_ARG(nvmlDeviceGetVgpuSchedulerState_v2, device, nvmlVgpuSchedulerStateInfo_v2_t, pSchedulerStateInfo)
#endif

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t,
                                                       unsigned int* capResult) {
  NVML_DEV(device);
  NVML_ARG(capResult);
  return NVML_ERROR_NOT_SUPPORTED;  // measured: also for a capability the enum does not have
}

// Both strings: the size pointer is mandatory (INVALID_ARGUMENT), the buffer may
// be null, then NOT_SUPPORTED; nothing is written (measured).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char*, unsigned int* bufferSize) {
  NVML_DEV(device);
  NVML_ARG(bufferSize);
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t*,
                                                   unsigned int* bufferSize) {
  NVML_DEV(device);
  NVML_ARG(bufferSize);
  return NVML_ERROR_NOT_SUPPORTED;
}

// The one that succeeds: bare metal (measured: SUCCESS, NONE, on the RTX 3060).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t* pVirtualMode) {
  NVML_DEV(device);
  NVML_ARG(pVirtualMode);
  *pVirtualMode = NVML_GPU_VIRTUALIZATION_MODE_NONE;
  return NVML_SUCCESS;
}

// vGPU Software licensing: measured SUCCESS on the card with nothing licensable
// (not supported, no features), in all four spellings. Only the two scalar
// fields are written; the feature array is past the count.
static nvmlReturn_t vgpu_grid_features(nvmlDevice_t device, nvmlGridLicensableFeatures_t* features) {
  NVML_DEV(device);
  NVML_ARG(features);
  features->isGridLicenseSupported = 0;
  features->licensableFeaturesCount = 0;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGridLicensableFeatures(nvmlDevice_t device, nvmlGridLicensableFeatures_t* f) {
  return vgpu_grid_features(device, f);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v2(nvmlDevice_t device, nvmlGridLicensableFeatures_t* f) {
  return vgpu_grid_features(device, f);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v3(nvmlDevice_t device, nvmlGridLicensableFeatures_t* f) {
  return vgpu_grid_features(device, f);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t* f) {
  return vgpu_grid_features(device, f);
}

// Setters. The feature is absent, so NOT_SUPPORTED, root or not (policy 3: a
// permission failure is for features that would otherwise be present). The
// handle is validated first; the getter of each pair is NOT_SUPPORTED for a null
// pointer too, so nothing else is looked at (doc order, siblings measured).
VGPU_NVML_NS_DEV(nvmlDeviceSetVirtualizationMode, device, (nvmlDevice_t device, nvmlGpuVirtualizationMode_t))
#if VGPU_NVML_12_4
VGPU_NVML_NS_DEV(nvmlDeviceSetVgpuCapabilities, device,
                 (nvmlDevice_t device, nvmlDeviceVgpuCapability_t, nvmlEnableState_t))
VGPU_NVML_NS_DEV(nvmlDeviceSetVgpuHeterogeneousMode, device,
                 (nvmlDevice_t device, const nvmlVgpuHeterogeneousMode_t*))
// The getters of the scheduler are INVALID_ARGUMENT for a null struct and then
// NOT_SUPPORTED, so the setters are too.
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetVgpuSchedulerState(nvmlDevice_t device,
                                                         nvmlVgpuSchedulerSetState_t* pSchedulerState) {
  NVML_DEV(device);
  NVML_ARG(pSchedulerState);
  return NVML_ERROR_NOT_SUPPORTED;
}
#endif
#if VGPU_NVML_13_2
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetVgpuSchedulerState_v2(nvmlDevice_t device,
                                                            nvmlVgpuSchedulerState_v2_t* pSchedulerState) {
  NVML_DEV(device);
  NVML_ARG(pSchedulerState);
  return NVML_ERROR_NOT_SUPPORTED;
}
VGPU_NVML_NS_DEV(nvmlDeviceVgpuForceGspUnload, device, (nvmlDevice_t device))
#endif

/* ======================================================================== *
 * vGPU inside a GPU instance (MIG-backed vGPU). The handle is the GPU
 * instance's; this group cannot see MIG's table, so it rejects the null handle
 * (INVALID_ARGUMENT, as the card does for 0) and treats any other as one that
 * exists but, with no vGPU host, answers NOT_SUPPORTED. (The card crashes on a
 * made-up non-null handle in half of these, so there is nothing to mirror.)
 * Order: handle, null struct, version word where the header documents one,
 * NOT_SUPPORTED (doc). The heterogeneous-mode and placement entries answer
 * NOT_SUPPORTED before the version word, as their device-level twins measured
 * on the card do.
 * ======================================================================== */

#if VGPU_NVML_12_9
#define VGPU_GI_GET(name, type, vermacro)                                \
  VGPU_EXPORT nvmlReturn_t name(nvmlGpuInstance_t gpuInstance, type* p) { \
    REQUIRE_INIT();                                                      \
    if (!gpuInstance) return NVML_ERROR_INVALID_ARGUMENT;                \
    NVML_ARG(p);                                                         \
    NVML_VERSION(p, vermacro);                                           \
    return NVML_ERROR_NOT_SUPPORTED;                                     \
  }
#define VGPU_GI_GET_NS_FIRST(name, type)                                 \
  VGPU_EXPORT nvmlReturn_t name(nvmlGpuInstance_t gpuInstance, type* p) { \
    REQUIRE_INIT();                                                      \
    if (!gpuInstance) return NVML_ERROR_INVALID_ARGUMENT;                \
    NVML_ARG(p);                                                         \
    return NVML_ERROR_NOT_SUPPORTED;                                     \
  }
VGPU_GI_GET(nvmlGpuInstanceGetActiveVgpus, nvmlActiveVgpuInstanceInfo_t, nvmlActiveVgpuInstanceInfo_v1)
VGPU_GI_GET(nvmlGpuInstanceGetCreatableVgpus, nvmlVgpuTypeIdInfo_t, nvmlVgpuTypeIdInfo_v1)
VGPU_GI_GET(nvmlGpuInstanceGetVgpuSchedulerLog, nvmlVgpuSchedulerLogInfo_t, nvmlVgpuSchedulerLogInfo_v1)
VGPU_GI_GET(nvmlGpuInstanceGetVgpuSchedulerState, nvmlVgpuSchedulerStateInfo_t, nvmlVgpuSchedulerStateInfo_v1)
VGPU_GI_GET(nvmlGpuInstanceSetVgpuSchedulerState, nvmlVgpuSchedulerState_t, nvmlVgpuSchedulerState_v1)
VGPU_GI_GET_NS_FIRST(nvmlGpuInstanceGetVgpuHeterogeneousMode, nvmlVgpuHeterogeneousMode_t)
VGPU_GI_GET_NS_FIRST(nvmlGpuInstanceGetVgpuTypeCreatablePlacements, nvmlVgpuCreatablePlacementInfo_t)
// Setter of the heterogeneous mode: NOT_SUPPORTED right after the handle, like its getter.
VGPU_EXPORT nvmlReturn_t nvmlGpuInstanceSetVgpuHeterogeneousMode(nvmlGpuInstance_t gpuInstance,
                                                                 const nvmlVgpuHeterogeneousMode_t*) {
  REQUIRE_INIT();
  if (!gpuInstance) return NVML_ERROR_INVALID_ARGUMENT;
  return NVML_ERROR_NOT_SUPPORTED;
}
#endif
#if VGPU_NVML_13_2
// The _v2 structs carry no version word.
VGPU_GI_GET_NS_FIRST(nvmlGpuInstanceGetVgpuSchedulerLog_v2, nvmlVgpuSchedulerLogInfo_v2_t)
VGPU_GI_GET_NS_FIRST(nvmlGpuInstanceGetVgpuSchedulerState_v2, nvmlVgpuSchedulerStateInfo_v2_t)
VGPU_GI_GET_NS_FIRST(nvmlGpuInstanceSetVgpuSchedulerState_v2, nvmlVgpuSchedulerState_v2_t)
#endif

/* ======================================================================== *
 * System-level vGPU. No vGPU manager: the version range NVML itself carries is
 * still reported (measured), and the rest is NOT_SUPPORTED.
 * ======================================================================== */

// Measured on the RTX 3060: SUCCESS, supported == current == [0x00180001, 0x001c0001],
// and a null pointer is skipped (SUCCESS with only the other one filled).
// 596.36's range; it is a fact of NVML's version, not of any GPU.
VGPU_EXPORT nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t* supported, nvmlVgpuVersion_t* current) {
  REQUIRE_INIT();
  if (supported) {
    supported->minVersion = 0x00180001u;
    supported->maxVersion = 0x001c0001u;
  }
  if (current) {
    current->minVersion = 0x00180001u;
    current->maxVersion = 0x001c0001u;
  }
  return NVML_SUCCESS;
}
// No manager to override: a null range is INVALID_ARGUMENT, anything else NOT_SUPPORTED (doc).
VGPU_EXPORT nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t* vgpuVersion) {
  REQUIRE_INIT();
  NVML_ARG(vgpuVersion);
  return NVML_ERROR_NOT_SUPPORTED;
}
// Measured: SUCCESS with vgpuVmCompatibility NONE and the limit code HOST_DRIVER
// (a host driver that is not a vGPU manager limits everything); all three
// pointers are mandatory.
VGPU_EXPORT nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t* vgpuMetadata, nvmlVgpuPgpuMetadata_t* pgpuMetadata,
                                                  nvmlVgpuPgpuCompatibility_t* compatibilityInfo) {
  REQUIRE_INIT();
  if (!vgpuMetadata || !pgpuMetadata || !compatibilityInfo) return NVML_ERROR_INVALID_ARGUMENT;
  compatibilityInfo->vgpuVmCompatibility = NVML_VGPU_VM_COMPATIBILITY_NONE;
  compatibilityInfo->compatibilityLimitCode = NVML_VGPU_COMPATIBILITY_LIMIT_HOST_DRIVER;
  return NVML_SUCCESS;
}
// Measured: null pointer INVALID_ARGUMENT, then NOT_SUPPORTED even for a capability the enum does not have.
VGPU_EXPORT nvmlReturn_t nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t, unsigned int* capResult) {
  REQUIRE_INIT();
  NVML_ARG(capResult);
  return NVML_ERROR_NOT_SUPPORTED;
}
// Excluded GPUs (ones the driver found and will not use): measured, none.
VGPU_EXPORT nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int* deviceCount) {
  REQUIRE_INIT();
  NVML_ARG(deviceCount);
  *deviceCount = 0;
  return NVML_SUCCESS;
}
// No index is valid when the count is 0 (measured: INVALID_ARGUMENT for 0 and 1).
VGPU_EXPORT nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int, nvmlExcludedDeviceInfo_t*) {
  REQUIRE_INIT();
  return NVML_ERROR_INVALID_ARGUMENT;
}

/* ======================================================================== *
 * S-class units and Host Interface Cards. None exists on a GPU that is not in
 * an S-class chassis: the count is 0 (measured) and every unit handle is
 * invalid (measured for the values 0 and 1; NVML documents INVALID_ARGUMENT
 * for a unit that is not one). Nothing here can be set either.
 * ======================================================================== */

VGPU_EXPORT nvmlReturn_t nvmlUnitGetCount(unsigned int* unitCount) {
  REQUIRE_INIT();
  NVML_ARG(unitCount);
  *unitCount = 0;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int, nvmlUnit_t*) {
  REQUIRE_INIT();
  return NVML_ERROR_INVALID_ARGUMENT;  // the count is 0, so no index is valid
}
VGPU_EXPORT nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t, nvmlUnitInfo_t*) {
  REQUIRE_INIT();
  return NVML_ERROR_INVALID_ARGUMENT;
}
VGPU_EXPORT nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t, nvmlLedState_t*) {
  REQUIRE_INIT();
  return NVML_ERROR_INVALID_ARGUMENT;
}
VGPU_EXPORT nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t, nvmlPSUInfo_t*) {
  REQUIRE_INIT();
  return NVML_ERROR_INVALID_ARGUMENT;
}
VGPU_EXPORT nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t, unsigned int, unsigned int*) {
  REQUIRE_INIT();
  return NVML_ERROR_INVALID_ARGUMENT;
}
VGPU_EXPORT nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t, nvmlUnitFanSpeeds_t*) {
  REQUIRE_INIT();
  return NVML_ERROR_INVALID_ARGUMENT;
}
VGPU_EXPORT nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t, unsigned int*, nvmlDevice_t*) {
  REQUIRE_INIT();
  return NVML_ERROR_INVALID_ARGUMENT;
}
// A setter of a unit that does not exist: the unit handle is what is wrong.
VGPU_EXPORT nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t, nvmlLedColor_t) {
  REQUIRE_INIT();
  return NVML_ERROR_INVALID_ARGUMENT;
}
// Measured: SUCCESS with no HICs: *hwbcCount becomes 0, the entry array may be
// null, a null count is INVALID_ARGUMENT.
VGPU_EXPORT nvmlReturn_t nvmlSystemGetHicVersion(unsigned int* hwbcCount, nvmlHwbcEntry_t*) {
  REQUIRE_INIT();
  NVML_ARG(hwbcCount);
  *hwbcCount = 0;
  return NVML_SUCCESS;
}

/* ======================================================================== *
 * GPM (GPU Performance Monitoring). The simulator has no timing model, hence no
 * counters to sample: a device is "not a GPM device" the way every GeForce is
 * (measured: nvmlGpmQueryDeviceSupport is SUCCESS with isSupportedDevice 0,
 * nvmlGpmSampleGet is NOT_SUPPORTED). Allocating and freeing a sample is
 * client-side and works (measured), and computing metrics from two samples that
 * were never filled reports each metric UNINITIALIZED, as the card does.
 * ======================================================================== */

VGPU_EXPORT nvmlReturn_t nvmlGpmQueryDeviceSupport(nvmlDevice_t device, nvmlGpmSupport_t* gpmSupport) {
  NVML_DEV(device);
  NVML_ARG(gpmSupport);
  NVML_VERSION(gpmSupport, NVML_GPM_SUPPORT_VERSION);
  gpmSupport->isSupportedDevice = 0;
  return NVML_SUCCESS;
}

// Measured: a null pointer is INVALID_ARGUMENT. Needs no nvmlInit (client memory).
VGPU_EXPORT nvmlReturn_t nvmlGpmSampleAlloc(nvmlGpmSample_t* gpmSample) {
  NVML_ARG(gpmSample);
  nvmlGpmSample_st* s = new (std::nothrow) nvmlGpmSample_st{0};
  if (!s) return NVML_ERROR_MEMORY;
  {
    std::lock_guard<std::mutex> lock(vgpu_gpm_mu);
    vgpu_gpm_samples().insert(s);
  }
  *gpmSample = s;
  return NVML_SUCCESS;
}
// Measured: freeing NULL is SUCCESS. A pointer this library did not hand out
// (or already freed) is INVALID_ARGUMENT, as the header documents.
VGPU_EXPORT nvmlReturn_t nvmlGpmSampleFree(nvmlGpmSample_t gpmSample) {
  if (!gpmSample) return NVML_SUCCESS;
  {
    std::lock_guard<std::mutex> lock(vgpu_gpm_mu);
    if (vgpu_gpm_samples().erase(gpmSample) == 0) return NVML_ERROR_INVALID_ARGUMENT;
  }
  delete gpmSample;
  return NVML_SUCCESS;
}
// Measured: NOT_SUPPORTED with a valid sample (the card crashes on a null one).
VGPU_EXPORT nvmlReturn_t nvmlGpmSampleGet(nvmlDevice_t device, nvmlGpmSample_t gpmSample) {
  NVML_DEV(device);
  if (!vgpu_gpm_known(gpmSample)) return NVML_ERROR_INVALID_ARGUMENT;
  return NVML_ERROR_NOT_SUPPORTED;
}
// Measured: NOT_SUPPORTED for GPU instance ids 0 and 1, INVALID_ARGUMENT for a null sample.
VGPU_EXPORT nvmlReturn_t nvmlGpmMigSampleGet(nvmlDevice_t device, unsigned int, nvmlGpmSample_t gpmSample) {
  NVML_DEV(device);
  if (!vgpu_gpm_known(gpmSample)) return NVML_ERROR_INVALID_ARGUMENT;
  return NVML_ERROR_NOT_SUPPORTED;
}
// Measured on the card, in this order: a null struct is INVALID_ARGUMENT; the
// version word (25, even for an all-zero struct); a metric count of 0 or above
// the array's size (333 on the card) INVALID_ARGUMENT; two distinct samples the
// library allocated, else INVALID_ARGUMENT; then SUCCESS with the per-metric
// status filled in: ids 1..332 are real metrics whose samples were never read
// (UNINITIALIZED), anything else INVALID_ARGUMENT. `value` is not written.
VGPU_EXPORT nvmlReturn_t nvmlGpmMetricsGet(nvmlGpmMetricsGet_t* metricsGet) {
  REQUIRE_INIT();
  NVML_ARG(metricsGet);
  NVML_VERSION(metricsGet, NVML_GPM_METRICS_GET_VERSION);
  const unsigned int capacity = sizeof metricsGet->metrics / sizeof metricsGet->metrics[0];
  if (metricsGet->numMetrics == 0 || metricsGet->numMetrics > capacity) return NVML_ERROR_INVALID_ARGUMENT;
  if (!vgpu_gpm_known(metricsGet->sample1) || !vgpu_gpm_known(metricsGet->sample2) ||
      metricsGet->sample1 == metricsGet->sample2)
    return NVML_ERROR_INVALID_ARGUMENT;
  for (unsigned int i = 0; i < metricsGet->numMetrics; ++i) {
    const unsigned int id = metricsGet->metrics[i].metricId;
    metricsGet->metrics[i].nvmlReturn = (id >= 1 && id <= 332) ? NVML_ERROR_UNINITIALIZED : NVML_ERROR_INVALID_ARGUMENT;
  }
  return NVML_SUCCESS;
}
#if VGPU_NVML_12_4
// Streaming GPM: NOT_SUPPORTED, a null pointer first for the getter (measured).
VGPU_NS_AFTER_ARG(nvmlGpmQueryIfStreamingEnabled, device, unsigned int, state)
VGPU_NVML_NS_DEV(nvmlGpmSetStreamingEnabled, device, (nvmlDevice_t device, unsigned int))
#endif

/* ======================================================================== *
 * Confidential computing. The simulated platforms are not CC-enabled and the
 * GPUs report it the way a GeForce does (measured). Every setter is
 * NOT_SUPPORTED: there is nothing to configure.
 * ======================================================================== */

#if VGPU_NVML_12_4
// Measured: the capabilities query succeeds and reports none (cpu 0, gpus 0).
VGPU_EXPORT nvmlReturn_t nvmlSystemGetConfComputeCapabilities(nvmlConfComputeSystemCaps_t* capabilities) {
  REQUIRE_INIT();
  NVML_ARG(capabilities);
  capabilities->cpuCaps = 0;
  capabilities->gpusCaps = 0;
  return NVML_SUCCESS;
}
// Measured: NOT_SUPPORTED regardless of the pointer.
VGPU_NVML_NS(nvmlSystemGetConfComputeState, (nvmlConfComputeSystemState_t*))
VGPU_NVML_NS(nvmlSystemGetConfComputeGpusReadyState, (unsigned int*))
VGPU_NVML_NS(nvmlSystemSetConfComputeGpusReadyState, (unsigned int))
// Measured: NOT_SUPPORTED, and the version word is not looked at.
VGPU_NVML_NS(nvmlSystemGetConfComputeSettings, (nvmlSystemConfComputeSettings_t*))
// Measured: null struct INVALID_ARGUMENT, then the version word, then NOT_SUPPORTED.
VGPU_EXPORT nvmlReturn_t nvmlSystemGetConfComputeKeyRotationThresholdInfo(
    nvmlConfComputeGetKeyRotationThresholdInfo_t* pKeyRotationThrInfo) {
  REQUIRE_INIT();
  NVML_ARG(pKeyRotationThrInfo);
  NVML_VERSION(pKeyRotationThrInfo, nvmlConfComputeGetKeyRotationThresholdInfo_v1);
  return NVML_ERROR_NOT_SUPPORTED;
}
// The setter checks the same things in the same order as the getter (doc).
VGPU_EXPORT nvmlReturn_t nvmlSystemSetConfComputeKeyRotationThresholdInfo(
    nvmlConfComputeSetKeyRotationThresholdInfo_t* pKeyRotationThrInfo) {
  REQUIRE_INIT();
  NVML_ARG(pKeyRotationThrInfo);
  NVML_VERSION(pKeyRotationThrInfo, nvmlConfComputeSetKeyRotationThresholdInfo_v1);
  return NVML_ERROR_NOT_SUPPORTED;
}
// Report and certificate: null struct INVALID_ARGUMENT, then NOT_SUPPORTED (measured).
VGPU_NS_AFTER_ARG(nvmlDeviceGetConfComputeGpuAttestationReport, device, nvmlConfComputeGpuAttestationReport_t,
                  gpuAtstReport)
VGPU_NS_AFTER_ARG(nvmlDeviceGetConfComputeGpuCertificate, device, nvmlConfComputeGpuCertificate_t, gpuCert)
// Memory sizes: NOT_SUPPORTED, null pointer or not (measured).
VGPU_NVML_NS_DEV(nvmlDeviceGetConfComputeMemSizeInfo, device, (nvmlDevice_t device, nvmlConfComputeMemSizeInfo_t*))
VGPU_NVML_NS_DEV(nvmlDeviceGetConfComputeProtectedMemoryUsage, device, (nvmlDevice_t device, nvmlMemory_t*))
VGPU_NVML_NS_DEV(nvmlDeviceSetConfComputeUnprotectedMemSize, device, (nvmlDevice_t device, unsigned long long))
#endif

/* ======================================================================== *
 * PRM, power smoothing and workload power profiles. Not modelled: the PRM
 * register interface needs the device's firmware, and power smoothing and
 * power profiles exist on Hopper/Blackwell hardware that the functional
 * simulator has no power-management model for. NOT_SUPPORTED, in the order the
 * card measured for the profile getters (null struct INVALID_ARGUMENT, version
 * word, NOT_SUPPORTED) and, for the others, handle, null struct, version word
 * where the header documents one (doc).
 * ======================================================================== */

#if VGPU_NVML_13_0
VGPU_NS_AFTER_ARG(nvmlDeviceReadWritePRM_v1, device, nvmlPRMTLV_v1_t, buffer)
#endif
#if VGPU_NVML_13_2
VGPU_NS_AFTER_ARG(nvmlDeviceReadPRMCounters_v1, device, nvmlPRMCounterList_v1_t, counterList)
VGPU_NS_AFTER_ARG(nvmlDeviceWorkloadPowerProfileUpdateProfiles_v1, device, nvmlWorkloadPowerProfileUpdateProfiles_v1_t,
                  updateProfiles)
#endif
#if VGPU_NVML_12_8
#define VGPU_VERSIONED_NS(name, type, vermacro)                \
  VGPU_EXPORT nvmlReturn_t name(nvmlDevice_t device, type* p) { \
    NVML_DEV(device);                                          \
    NVML_ARG(p);                                               \
    NVML_VERSION(p, vermacro);                                 \
    return NVML_ERROR_NOT_SUPPORTED;                           \
  }
VGPU_VERSIONED_NS(nvmlDeviceWorkloadPowerProfileGetProfilesInfo, nvmlWorkloadPowerProfileProfilesInfo_t,
                  nvmlWorkloadPowerProfileProfilesInfo_v1)
VGPU_VERSIONED_NS(nvmlDeviceWorkloadPowerProfileGetCurrentProfiles, nvmlWorkloadPowerProfileCurrentProfiles_t,
                  nvmlWorkloadPowerProfileCurrentProfiles_v1)
VGPU_VERSIONED_NS(nvmlDeviceWorkloadPowerProfileSetRequestedProfiles, nvmlWorkloadPowerProfileRequestedProfiles_t,
                  nvmlWorkloadPowerProfileRequestedProfiles_v1)
VGPU_VERSIONED_NS(nvmlDeviceWorkloadPowerProfileClearRequestedProfiles, nvmlWorkloadPowerProfileRequestedProfiles_t,
                  nvmlWorkloadPowerProfileRequestedProfiles_v1)
VGPU_VERSIONED_NS(nvmlDevicePowerSmoothingUpdatePresetProfileParam, nvmlPowerSmoothingProfile_t,
                  nvmlPowerSmoothingProfile_v1)
// Their headers document no version check for these two.
VGPU_NS_AFTER_ARG(nvmlDevicePowerSmoothingActivatePresetProfile, device, nvmlPowerSmoothingProfile_t, profile)
VGPU_NS_AFTER_ARG(nvmlDevicePowerSmoothingSetState, device, nvmlPowerSmoothingState_t, state)
#endif
