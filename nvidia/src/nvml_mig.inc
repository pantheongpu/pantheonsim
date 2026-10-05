// NVML: MIG, GPU instances, compute instances. See nvml_common.inc for the conventions.
//
// Multi-Instance GPU partitioning is NOT simulated: there is no instance model and
// no MIG device handle, so no GPU instance or compute instance ever exists. What
// this file answers is therefore the state of a device that has MIG switched off:
//
//  * A part with no MIG at all (nvml_facts(*d).mig false: the RTX 3060, T4 ...) and
//    a MIG part with MIG off both have no instances. Where the two differ, the
//    comment says which measurement or which documentation line decides.
//  * nvmlDeviceSetMigMode keeps the pending mode in the settings overlay, in the
//    Persistent lifetime like the real driver (the inforom survives a reset).
//    Disabling an already disabled device is real; enabling is refused (below).
//    nvmlDeviceGetMigMode, in nvml_api.cpp, reads it back.
//  * nvmlGpuInstance_t and nvmlComputeInstance_t are pointers this library never
//    hands out, so every one a caller passes is invalid: the documented
//    INVALID_ARGUMENT. They are never dereferenced.
//
// The RTX 3060 oracle: measured through NVIDIA's NVML 13.0 (WSL, driver 596.36), GPU 1,
// getters only, with a valid handle (the creators, destroyers and nvmlDeviceSetMigMode
// were never called on the card; their answers come from the documentation in nvml.h):
//     GetGpuInstanceProfileInfo       profile 0..17 NOT_SUPPORTED, 18 and above INVALID_ARGUMENT
//     GetGpuInstanceProfileInfoV      the version word must be the v2 or the v3 one, else
//                                     INVALID_ARGUMENT (not ARGUMENT_VERSION_MISMATCH: the
//                                     documentation says "info->version are invalid"); then as above
//     GetGpuInstanceProfileInfoByIdV  right version: NOT_SUPPORTED for every id; else INVALID_ARGUMENT
//     PossiblePlacements(_v2), RemainingCapacity, GetGpuInstances
//                                     profile 0 NOT_SUPPORTED, any other profile INVALID_ARGUMENT
//     GetGpuInstanceById              NOT_FOUND for every id
//     a NULL output pointer           INVALID_ARGUMENT, before the profile is looked at
//     a NULL GPU or compute instance  INVALID_ARGUMENT, whatever else is passed

namespace {

// The last profile value the card's driver recognises. NVML_GPU_INSTANCE_PROFILE_COUNT is
// 0x11 in CUDA 13.2's nvml.h, and the card answers NOT_SUPPORTED for 0x11 itself, so its
// driver knows a profile the header does not name yet. Above it: INVALID_ARGUMENT, which is
// how a tool that enumerates "from 0 until INVALID_ARGUMENT" knows to stop.
constexpr unsigned int kMigLastProfile = 0x11;

// Whether this device's profile has MIG hardware.
bool mig_capable(const vgpu::telemetry::DeviceSample& d) { return nvml_facts(d).mig; }

// The profile-information queries enumerate alike on every device (measured on the RTX 3060).
nvmlReturn_t mig_info_profile(unsigned int profile) {
  return profile <= kMigLastProfile ? NVML_ERROR_NOT_SUPPORTED : NVML_ERROR_INVALID_ARGUMENT;
}

// What a profile value in a placements, capacity, instances or create call answers on a
// device with no instances. Never SUCCESS: with MIG off there is nothing to place or count.
//   MIG part, MIG off  every profile value up to the last the driver knows is NOT_SUPPORTED
//                      ("device doesn't have MIG mode enabled"), above it INVALID_ARGUMENT.
//   no-MIG part        measured on the RTX 3060: only profile 0 is NOT_SUPPORTED, every
//                      other value INVALID_ARGUMENT.
nvmlReturn_t mig_profile_status(const vgpu::telemetry::DeviceSample& d, unsigned int profile) {
  if (mig_capable(d)) return mig_info_profile(profile);
  return profile == 0 ? NVML_ERROR_NOT_SUPPORTED : NVML_ERROR_INVALID_ARGUMENT;
}

// The version words of nvmlGpuInstanceProfileInfo_v2_t and _v3_t. The card accepts both
// (the two have the same size; the v3 type exists only from CUDA 12.4's header on).
constexpr unsigned int kMigInfoV2 = static_cast<unsigned int>(sizeof(nvmlGpuInstanceProfileInfo_v2_t)) | (2u << 24);
constexpr unsigned int kMigInfoV3 = static_cast<unsigned int>(sizeof(nvmlGpuInstanceProfileInfo_v2_t)) | (3u << 24);
static_assert(kMigInfoV2 == nvmlGpuInstanceProfileInfo_v2, "the v2 version word");
#if defined(nvmlGpuInstanceProfileInfo_v3)
static_assert(kMigInfoV3 == nvmlGpuInstanceProfileInfo_v3, "the v3 version word");
#endif

// The versioned GPU instance profile queries (by profile value, or by profile ID).
nvmlReturn_t mig_profile_info_v(nvmlDevice_t device, unsigned int profile, bool by_id,
                                nvmlGpuInstanceProfileInfo_v2_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  if (info->version != kMigInfoV2 && info->version != kMigInfoV3) return NVML_ERROR_INVALID_ARGUMENT;
  // By ID the card answers NOT_SUPPORTED for every ID; by profile value it enumerates.
  return by_id ? NVML_ERROR_NOT_SUPPORTED : mig_info_profile(profile);
}

// Placements of a profile. The documentation says the call "returns all the possible
// placements for the given profile regardless of whether MIG is enabled or not", which
// would make a MIG part answer SUCCESS. But the profile IDs a caller passes come from the
// profile enumeration, which has nothing to report on a device with MIG off, and there
// is no placement table to give (the slice layout depends on the part and on its memory
// size, and would be invented), so the documented "profileId isn't supported" is the answer.
nvmlReturn_t mig_placements(nvmlDevice_t device, unsigned int profileId, unsigned int* count) {
  NVML_DEV(device);
  NVML_ARG(count);
  return mig_profile_status(*d, profileId);
}

// Creating a GPU instance. Order: handle, arguments, whether MIG is enabled (it never is:
// nvmlDeviceSetMigMode refuses to enable it, so NOT_SUPPORTED comes before the privilege
// check, as for any feature the device does not have).
nvmlReturn_t mig_create(nvmlDevice_t device, unsigned int profileId, bool placed,
                        const nvmlGpuInstancePlacement_t* placement, nvmlGpuInstance_t* gpuInstance) {
  NVML_DEV(device);
  NVML_ARG(gpuInstance);
  if (placed && !placement) return NVML_ERROR_INVALID_ARGUMENT;
  return mig_profile_status(*d, profileId);
}

}  // namespace

// ---- MIG mode ---------------------------------------------------------------------

// Order of checks: handle, arguments (INVALID_ARGUMENT), whether the part has MIG
// (NOT_SUPPORTED), the privilege (NO_PERMISSION), then the act.
//
// ENABLE is refused with NOT_SUPPORTED on a MIG part: switching the mode on is a promise
// that GPU instances can be created, and this machine cannot create them (partitioning
// is not simulated). Accepting it would leave nvidia-smi and every orchestrator believing
// they can carve the GPU up. activationStatus carries the same NOT_SUPPORTED and the
// pending mode stays disabled. DISABLE of a device that is already disabled succeeds and
// is recorded as the pending mode.
VGPU_EXPORT nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode,
                                              nvmlReturn_t* activationStatus) {
  NVML_DEV(device);
  if (mode != NVML_DEVICE_MIG_DISABLE && mode != NVML_DEVICE_MIG_ENABLE) return NVML_ERROR_INVALID_ARGUMENT;
  NVML_ARG(activationStatus);
  if (!mig_capable(*d)) return NVML_ERROR_NOT_SUPPORTED;
  NVML_SETTER_PERMISSION();
  if (mode == NVML_DEVICE_MIG_ENABLE) {
    *activationStatus = NVML_ERROR_NOT_SUPPORTED;
    return NVML_ERROR_NOT_SUPPORTED;
  }
  overlay_set(d->uuid, Life::Persistent, "mig_mode", "0");
  *activationStatus = NVML_SUCCESS;
  return NVML_SUCCESS;
}

// ---- GPU instance profiles, placements, capacity, instances ---------------------------

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile,
                                                             nvmlGpuInstanceProfileInfo_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  return mig_info_profile(profile);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile,
                                                              nvmlGpuInstanceProfileInfo_v2_t* info) {
  return mig_profile_info_v(device, profile, false, info);
}
#if VGPU_NVML_13_0
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoByIdV(nvmlDevice_t device, unsigned int profileId,
                                                                  nvmlGpuInstanceProfileInfo_v2_t* info) {
  return mig_profile_info_v(device, profileId, true, info);
}
#endif
// Both spellings: the header's own unversioned name is an alias of _v2.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId,
                                                                       nvmlGpuInstancePlacement_t*,
                                                                       unsigned int* count) {
  return mig_placements(device, profileId, count);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements(nvmlDevice_t device, unsigned int profileId,
                                                                    nvmlGpuInstancePlacement_t*,
                                                                    unsigned int* count) {
  return mig_placements(device, profileId, count);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId,
                                                                   unsigned int* count) {
  NVML_DEV(device);
  NVML_ARG(count);
  return mig_profile_status(*d, profileId);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId,
                                                   nvmlGpuInstance_t* gpuInstances, unsigned int* count) {
  NVML_DEV(device);
  NVML_ARG(gpuInstances);
  NVML_ARG(count);
  return mig_profile_status(*d, profileId);
}
// Documented NOT_SUPPORTED "if device doesn't have MIG mode enabled", which is what a MIG
// part with MIG off answers. A part with no MIG: NOT_FOUND, measured on the RTX 3060 for
// every ID (its driver searches an empty instance list without asking about MIG).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int,
                                                      nvmlGpuInstance_t* gpuInstance) {
  NVML_DEV(device);
  NVML_ARG(gpuInstance);
  return mig_capable(*d) ? NVML_ERROR_NOT_SUPPORTED : NVML_ERROR_NOT_FOUND;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId,
                                                     nvmlGpuInstance_t* gpuInstance) {
  return mig_create(device, profileId, false, nullptr, gpuInstance);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId,
                                                                  const nvmlGpuInstancePlacement_t* placement,
                                                                  nvmlGpuInstance_t* gpuInstance) {
  return mig_create(device, profileId, true, placement, gpuInstance);
}

// ---- GPU instance and compute instance handles ----------------------------------------
//
// No instance handle exists: each entry point validates the handle first (the documented
// INVALID_ARGUMENT; measured so on the RTX 3060 for NULL), so none gets further. The
// remaining arguments, including a version word, are not looked at: a handle that is not
// valid decides the answer, as it does when the real driver cannot find the instance.

#define VGPU_MIG_NO_INSTANCE() \
  REQUIRE_INIT();              \
  return NVML_ERROR_INVALID_ARGUMENT

VGPU_EXPORT nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t) { VGPU_MIG_NO_INSTANCE(); }
VGPU_EXPORT nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t, nvmlGpuInstanceInfo_t*) {
  VGPU_MIG_NO_INSTANCE();
}
VGPU_EXPORT nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t, unsigned int,
                                                                      unsigned int,
                                                                      nvmlComputeInstanceProfileInfo_t*) {
  VGPU_MIG_NO_INSTANCE();
}
VGPU_EXPORT nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t, unsigned int,
                                                                       unsigned int,
                                                                       nvmlComputeInstanceProfileInfo_v2_t*) {
  VGPU_MIG_NO_INSTANCE();
}
VGPU_EXPORT nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t, unsigned int,
                                                                            unsigned int*) {
  VGPU_MIG_NO_INSTANCE();
}
VGPU_EXPORT nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t, unsigned int,
                                                                             nvmlComputeInstancePlacement_t*,
                                                                             unsigned int*) {
  VGPU_MIG_NO_INSTANCE();
}
VGPU_EXPORT nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t, unsigned int,
                                                              nvmlComputeInstance_t*) {
  VGPU_MIG_NO_INSTANCE();
}
VGPU_EXPORT nvmlReturn_t nvmlGpuInstanceCreateComputeInstanceWithPlacement(
    nvmlGpuInstance_t, unsigned int, const nvmlComputeInstancePlacement_t*, nvmlComputeInstance_t*) {
  VGPU_MIG_NO_INSTANCE();
}
VGPU_EXPORT nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t, unsigned int,
                                                            nvmlComputeInstance_t*, unsigned int*) {
  VGPU_MIG_NO_INSTANCE();
}
VGPU_EXPORT nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t, unsigned int,
                                                               nvmlComputeInstance_t*) {
  VGPU_MIG_NO_INSTANCE();
}
VGPU_EXPORT nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t) { VGPU_MIG_NO_INSTANCE(); }
// Both spellings: the header's own unversioned name is an alias of _v2.
VGPU_EXPORT nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t, nvmlComputeInstanceInfo_t*) {
  VGPU_MIG_NO_INSTANCE();
}
VGPU_EXPORT nvmlReturn_t nvmlComputeInstanceGetInfo(nvmlComputeInstance_t, nvmlComputeInstanceInfo_t*) {
  VGPU_MIG_NO_INSTANCE();
}

#undef VGPU_MIG_NO_INSTANCE
