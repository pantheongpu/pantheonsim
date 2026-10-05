"""NVML's vGPU, unit, GPM, confidential-computing, PRM and power-profile entry points, through ctypes against the shim.

Run inside `vgpu shell --gpu <profile> --count 2` (nvidia/t4 and nvidia/rtx3060), with the shim directory and the
profile id as arguments. The machine is bare metal: no vGPU manager, no S-class unit, no confidential computing, no GPM
counters, no PRM. What each call answers there was measured on a real RTX 3060 through NVIDIA's NVML (the order in which it
looks at a null pointer, a version word, a handle and the feature itself differs call by call, and is mirrored); the
comments say where a check follows the nvml.h documentation instead, because the real library could not be asked.

Symbols newer than the header the shim was built with are skipped, not failed (hosted CI builds with CUDA 12.0's).
"""
import ctypes, os, struct, sys

SUCCESS, UNINITIALIZED, INVALID, NOT_SUPPORTED, NO_PERMISSION, NOT_FOUND, INSUFFICIENT = 0, 1, 2, 3, 4, 6, 7
MEMORY, VERSION_MISMATCH = 20, 25
lib = ctypes.CDLL(os.path.join(sys.argv[1], "libnvidia-ml.so.1"))
profile = sys.argv[2] if len(sys.argv) > 2 else ""
fails = 0
skipped = 0
U, ULL, P = ctypes.c_uint, ctypes.c_ulonglong, ctypes.c_void_p
BAD = 0xDEAD  # a version word no struct has

# NVML_STRUCT_VERSION(struct, 1) = sizeof | 1 << 24 (the struct sizes are ABI).
VER = dict(placement_id=0x01000008, bar1=0x01000010, runtime=0x01000010, hetero=0x01000008, inst_util=0x01000020,
           proc_util=0x01000018, current_profiles=0x01000064, profiles_info=0x01002BF8, requested=0x01000024,
           key_get=0x01000010, key_set=0x01000010, max_instance=0x0100000C, active=0x01000010, type_ids=0x01000010,
           sched_log=0x010025A0, sched_state_info=0x01000018, sched_state=0x01000018, creatable=0x01000020,
           smoothing_profile=0x01000018, smoothing_state=0x01000008)


def has(name):
    return hasattr(lib, name)


def check(name, ok, got=""):
    global fails
    print(("ok    " if ok else "FAIL  ") + name + ("" if ok else f"  -> {got}"))
    fails += 0 if ok else 1


def call(name, *args):
    """The status of lib.<name>(*args), or None when the shim was built with a header that lacks the symbol."""
    global skipped
    if not has(name):
        skipped += 1
        print(f"skip  {name} (not in this build's header)")
        return None
    return getattr(lib, name)(*args)


def t(name, want, *args, label=""):
    rc = call(name, *args)
    if rc is None:
        return None
    check(f"{name}{(' ' + label) if label else ''} -> {want}", rc == want, rc)
    return rc


def buf(n=70000, fill=0):
    b = ctypes.create_string_buffer(n)
    if fill:
        ctypes.memset(b, fill, n)
    return b


def versioned(version, n=70000):
    b = buf(n)
    struct.pack_into("<I", b, 0, version)
    return b


def u32(b, off=0):
    return struct.unpack_from("<I", b, off)[0]


def ptr_u():
    return ctypes.byref(U())


check("nvmlInit_v2", lib.nvmlInit_v2() == SUCCESS)
count = U()
lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
check("two devices", count.value == 2, count.value)
devs = []
for i in range(count.value):
    h = P()
    check(f"handle {i}", lib.nvmlDeviceGetHandleByIndex_v2(U(i), ctypes.byref(h)) == SUCCESS)
    devs.append(h.value)
D = [P(d) for d in devs]
NULL_DEV, BAD_DEV = P(0), P(999)

# ---- S-class units and host interface cards: none -------------------------------
n = U(77)
t("nvmlUnitGetCount", SUCCESS, ctypes.byref(n))
if has("nvmlUnitGetCount"):
    check("no S-class unit: the count is 0", n.value == 0, n.value)
t("nvmlUnitGetCount", INVALID, None, label="null pointer")
unit = P()
t("nvmlUnitGetHandleByIndex", INVALID, U(0), ctypes.byref(unit), label="index 0 of 0 units")
for u in (P(0), P(1)):
    s = f"unit {u.value}"
    t("nvmlUnitGetUnitInfo", INVALID, u, buf(), label=s)
    t("nvmlUnitGetLedState", INVALID, u, buf(), label=s)
    t("nvmlUnitGetPsuInfo", INVALID, u, buf(), label=s)
    t("nvmlUnitGetTemperature", INVALID, u, U(0), ptr_u(), label=s)
    t("nvmlUnitGetFanSpeedInfo", INVALID, u, buf(), label=s)
    t("nvmlUnitGetDevices", INVALID, u, ctypes.byref(U(4)), buf(), label=s)
    t("nvmlUnitSetLedState", INVALID, u, U(0), label=s)  # NVML_LED_COLOR_GREEN; the unit is what is wrong
hwbc = U(3)
t("nvmlSystemGetHicVersion", SUCCESS, ctypes.byref(hwbc), buf())
if has("nvmlSystemGetHicVersion"):
    check("no HIC: the count is written as 0", hwbc.value == 0, hwbc.value)
t("nvmlSystemGetHicVersion", SUCCESS, ctypes.byref(U(3)), None, label="no entry array")
t("nvmlSystemGetHicVersion", INVALID, None, buf(), label="null count")

# ---- excluded devices -------------------------------------------------------------
n = U(5)
t("nvmlGetExcludedDeviceCount", SUCCESS, ctypes.byref(n))
if has("nvmlGetExcludedDeviceCount"):
    check("no excluded GPU", n.value == 0, n.value)
t("nvmlGetExcludedDeviceCount", INVALID, None, label="null pointer")
t("nvmlGetExcludedDeviceInfoByIndex", INVALID, U(0), buf(), label="index 0 of 0")

# ---- the vGPU manager's version range and compatibility -------------------------------
sup, cur = buf(8), buf(8)
t("nvmlGetVgpuVersion", SUCCESS, sup, cur)
if has("nvmlGetVgpuVersion"):
    # Measured on the RTX 3060 through NVIDIA's NVML 13.0 (596.36): both ranges are [0x00180001, 0x001c0001].
    check("vGPU version: supported == current == the range NVML carries",
          (u32(sup), u32(sup, 4), u32(cur), u32(cur, 4)) == (0x00180001, 0x001C0001, 0x00180001, 0x001C0001),
          (hex(u32(sup)), hex(u32(sup, 4)), hex(u32(cur)), hex(u32(cur, 4))))
only = buf(8)
t("nvmlGetVgpuVersion", SUCCESS, only, None, label="one null: the other is still filled")
t("nvmlGetVgpuVersion", SUCCESS, None, None, label="both null (as the real library)")
t("nvmlSetVgpuVersion", NOT_SUPPORTED, buf(8), label="no vGPU manager to override")
t("nvmlSetVgpuVersion", INVALID, None, label="null")
comp = buf(8, 0xFF)
t("nvmlGetVgpuCompatibility", SUCCESS, buf(), buf(), comp)
if has("nvmlGetVgpuCompatibility"):
    check("compatibility: NONE, limited by the host driver (measured)", (u32(comp), u32(comp, 4)) == (0, 1),
          (u32(comp), u32(comp, 4)))
t("nvmlGetVgpuCompatibility", INVALID, None, buf(), buf(8), label="null vgpu metadata")
t("nvmlGetVgpuCompatibility", INVALID, buf(), None, buf(8), label="null pgpu metadata")
t("nvmlGetVgpuCompatibility", INVALID, buf(), buf(), None, label="null result")
for cap in (0, 1, 99):
    t("nvmlGetVgpuDriverCapabilities", NOT_SUPPORTED, U(cap), ptr_u(), label=f"capability {cap}")
t("nvmlGetVgpuDriverCapabilities", INVALID, U(0), None, label="null result")

# ---- vGPU instances: none exists -----------------------------------------------------
# Measured: the plain getters check a null pointer first (INVALID_ARGUMENT), then id 0 is INVALID_ARGUMENT and any
# other id NOT_FOUND.
SIMPLE = ["nvmlVgpuInstanceGetAccountingMode", "nvmlVgpuInstanceGetEccMode", "nvmlVgpuInstanceGetEncoderCapacity",
          "nvmlVgpuInstanceGetFBCStats", "nvmlVgpuInstanceGetFbUsage", "nvmlVgpuInstanceGetFrameRateLimit",
          "nvmlVgpuInstanceGetGpuInstanceId", "nvmlVgpuInstanceGetLicenseInfo", "nvmlVgpuInstanceGetLicenseInfo_v2",
          "nvmlVgpuInstanceGetLicenseStatus", "nvmlVgpuInstanceGetType"]
for name in SIMPLE:
    t(name, INVALID, U(0), buf(), label="id 0")
    t(name, NOT_FOUND, U(1), buf(), label="id 1")
    t(name, NOT_FOUND, U(0xFFFFFFFF), buf(), label="id 0xffffffff")
    t(name, INVALID, U(1), None, label="id 1, null pointer")
    t(name, INVALID, U(0), None, label="id 0, null pointer")
t("nvmlVgpuInstanceGetAccountingStats", INVALID, U(0), U(1), buf(), label="id 0")
t("nvmlVgpuInstanceGetAccountingStats", NOT_FOUND, U(1), U(1), buf(), label="id 1")
t("nvmlVgpuInstanceGetAccountingStats", INVALID, U(1), U(1), None, label="null stats")

# Accounting pids: a null array with *count 0 asks for the capacity (INSUFFICIENT_SIZE, 4000, the id is not looked at).
c = U(0)
t("nvmlVgpuInstanceGetAccountingPids", INSUFFICIENT, U(0), ctypes.byref(c), None, label="sizing call, id 0")
if has("nvmlVgpuInstanceGetAccountingPids"):
    check("accounting pids capacity is reported", c.value == 4000, c.value)
t("nvmlVgpuInstanceGetAccountingPids", INSUFFICIENT, U(1), ctypes.byref(U(0)), None, label="sizing call, id 1")
t("nvmlVgpuInstanceGetAccountingPids", INVALID, U(1), ctypes.byref(U(64)), None, label="null array, count 64")
t("nvmlVgpuInstanceGetAccountingPids", INVALID, U(0), ctypes.byref(U(64)), buf(), label="id 0")
t("nvmlVgpuInstanceGetAccountingPids", NOT_FOUND, U(1), ctypes.byref(U(64)), buf(), label="id 1")
t("nvmlVgpuInstanceGetAccountingPids", INVALID, U(1), None, buf(), label="null count")

for name in ("nvmlVgpuInstanceGetEncoderSessions", "nvmlVgpuInstanceGetFBCSessions"):
    t(name, INVALID, U(0), ctypes.byref(U(0)), buf(), label="id 0")
    t(name, NOT_FOUND, U(1), ctypes.byref(U(0)), buf(), label="id 1")
    t(name, NOT_FOUND, U(1), ctypes.byref(U(0)), None, label="a count-only query is allowed")
    t(name, INVALID, U(1), None, buf(), label="null count")
t("nvmlVgpuInstanceGetEncoderStats", INVALID, U(0), ptr_u(), ptr_u(), ptr_u(), label="id 0")
t("nvmlVgpuInstanceGetEncoderStats", NOT_FOUND, U(1), ptr_u(), ptr_u(), ptr_u(), label="id 1")
t("nvmlVgpuInstanceGetEncoderStats", INVALID, U(1), None, ptr_u(), ptr_u(), label="null session count")
t("nvmlVgpuInstanceGetEncoderStats", INVALID, U(1), ptr_u(), None, ptr_u(), label="null fps")
t("nvmlVgpuInstanceGetEncoderStats", INVALID, U(1), ptr_u(), ptr_u(), None, label="null latency")

# Strings: no size check for these three, the id decides (measured: id 1 with a 1-byte buffer is NOT_FOUND).
for name in ("nvmlVgpuInstanceGetUUID", "nvmlVgpuInstanceGetMdevUUID", "nvmlVgpuInstanceGetVmDriverVersion"):
    t(name, INVALID, U(0), buf(), U(80), label="id 0")
    t(name, NOT_FOUND, U(1), buf(), U(80), label="id 1")
    t(name, NOT_FOUND, U(1), buf(), U(1), label="id 1, a 1-byte buffer")
    t(name, INVALID, U(1), None, U(80), label="null buffer")
t("nvmlVgpuInstanceGetGpuPciId", INVALID, U(0), buf(), ctypes.byref(U(64)), label="id 0")
t("nvmlVgpuInstanceGetGpuPciId", NOT_FOUND, U(1), buf(), ctypes.byref(U(64)), label="id 1")
t("nvmlVgpuInstanceGetGpuPciId", NOT_FOUND, U(1), buf(), ctypes.byref(U(1)), label="id 1, length 1 (no size check)")
t("nvmlVgpuInstanceGetGpuPciId", INVALID, U(1), None, ctypes.byref(U(64)), label="null buffer")
t("nvmlVgpuInstanceGetGpuPciId", INVALID, U(1), buf(), None, label="null length")
# The VM id is the exception: under 80 bytes is INSUFFICIENT_SIZE first, even for id 0 (measured).
for sz in (0, 1, 79):
    t("nvmlVgpuInstanceGetVmID", INSUFFICIENT, U(0), buf(), U(sz), ptr_u(), label=f"id 0, size {sz}")
    t("nvmlVgpuInstanceGetVmID", INSUFFICIENT, U(1), buf(), U(sz), ptr_u(), label=f"id 1, size {sz}")
t("nvmlVgpuInstanceGetVmID", INSUFFICIENT, U(1), None, U(1), ptr_u(), label="size beats the null buffer")
t("nvmlVgpuInstanceGetVmID", INVALID, U(0), buf(), U(80), ptr_u(), label="id 0, size 80")
t("nvmlVgpuInstanceGetVmID", NOT_FOUND, U(1), buf(), U(80), ptr_u(), label="id 1, size 80")
t("nvmlVgpuInstanceGetVmID", INVALID, U(1), None, U(80), ptr_u(), label="null buffer")
t("nvmlVgpuInstanceGetVmID", INVALID, U(1), buf(), U(80), None, label="null id type")
# Metadata: the size NVML asks for (468) is reported before the id is looked at (measured).
sz = U(0)
t("nvmlVgpuInstanceGetMetadata", INSUFFICIENT, U(1), None, ctypes.byref(sz), label="sizing call")
if has("nvmlVgpuInstanceGetMetadata"):
    check("metadata size is reported", sz.value == 468, sz.value)
sz = U(0)
t("nvmlVgpuInstanceGetMetadata", INSUFFICIENT, U(0), buf(), ctypes.byref(sz), label="id 0 too, buffer, size 0")
sz = U(467)
t("nvmlVgpuInstanceGetMetadata", INSUFFICIENT, U(1), buf(), ctypes.byref(sz), label="467 is too small")
if has("nvmlVgpuInstanceGetMetadata"):
    check("... and says 468", sz.value == 468, sz.value)
sz = U(468)
t("nvmlVgpuInstanceGetMetadata", INVALID, U(0), buf(), ctypes.byref(sz), label="id 0, big enough")
t("nvmlVgpuInstanceGetMetadata", NOT_FOUND, U(1), buf(), ctypes.byref(sz), label="id 1, big enough")
sz = U(1000)
t("nvmlVgpuInstanceGetMetadata", NOT_FOUND, U(1), buf(), ctypes.byref(sz), label="id 1, 1000 bytes")
if has("nvmlVgpuInstanceGetMetadata"):
    check("... size left alone", sz.value == 1000, sz.value)
t("nvmlVgpuInstanceGetMetadata", INVALID, U(1), None, ctypes.byref(U(1000)), label="null buffer with a size")
t("nvmlVgpuInstanceGetMetadata", INVALID, U(1), buf(), None, label="null size")
# Newer: the runtime state size looks at the id only; the placement id looks at the pointer and version first.
t("nvmlVgpuInstanceGetRuntimeStateSize", INVALID, U(0), versioned(VER["runtime"]), label="id 0")
t("nvmlVgpuInstanceGetRuntimeStateSize", NOT_FOUND, U(1), versioned(VER["runtime"]), label="id 1")
t("nvmlVgpuInstanceGetRuntimeStateSize", NOT_FOUND, U(1), versioned(BAD), label="id 1, bad version (not looked at)")
t("nvmlVgpuInstanceGetRuntimeStateSize", NOT_FOUND, U(1), None, label="id 1, null (not looked at)")
t("nvmlVgpuInstanceGetPlacementId", INVALID, U(0), versioned(VER["placement_id"]), label="id 0")
t("nvmlVgpuInstanceGetPlacementId", NOT_FOUND, U(1), versioned(VER["placement_id"]), label="id 1")
t("nvmlVgpuInstanceGetPlacementId", VERSION_MISMATCH, U(0), versioned(BAD), label="bad version beats the id")
t("nvmlVgpuInstanceGetPlacementId", INVALID, U(1), None, label="null struct")
# Setters on an instance: the id is the handle.
for cap in (0, 50, 101):
    t("nvmlVgpuInstanceSetEncoderCapacity", INVALID, U(0), U(cap), label=f"id 0, capacity {cap}")
    t("nvmlVgpuInstanceSetEncoderCapacity", NOT_FOUND, U(1), U(cap), label=f"id 1, capacity {cap}")
t("nvmlVgpuInstanceClearAccountingPids", INVALID, U(0), label="id 0")
t("nvmlVgpuInstanceClearAccountingPids", NOT_FOUND, U(1), label="id 1")

# ---- vGPU types: no id is valid (the real library says INVALID_ARGUMENT for 0 and 1 alike) ---------
for tid in (0, 1):
    s = f"type {tid}"
    t("nvmlVgpuTypeGetCapabilities", INVALID, U(tid), U(0), ptr_u(), label=s)
    t("nvmlVgpuTypeGetDeviceID", INVALID, U(tid), ctypes.byref(ULL()), ctypes.byref(ULL()), label=s)
    t("nvmlVgpuTypeGetFbReservation", INVALID, U(tid), ctypes.byref(ULL()), label=s)
    t("nvmlVgpuTypeGetFrameRateLimit", INVALID, U(tid), ptr_u(), label=s)
    t("nvmlVgpuTypeGetFramebufferSize", INVALID, U(tid), ctypes.byref(ULL()), label=s)
    t("nvmlVgpuTypeGetGpuInstanceProfileId", INVALID, U(tid), ptr_u(), label=s)
    t("nvmlVgpuTypeGetGspHeapSize", INVALID, U(tid), ctypes.byref(ULL()), label=s)
    t("nvmlVgpuTypeGetMaxInstancesPerVm", INVALID, U(tid), ptr_u(), label=s)
    t("nvmlVgpuTypeGetNumDisplayHeads", INVALID, U(tid), ptr_u(), label=s)
    t("nvmlVgpuTypeGetResolution", INVALID, U(tid), U(0), ptr_u(), ptr_u(), label=s)
    t("nvmlVgpuTypeGetCapabilities", INVALID, U(tid), U(0), None, label=s + ", null")
# Name and class: id 0, null size, short buffer (INSUFFICIENT_SIZE, *size = 64, before the buffer pointer), null
# buffer, then the unknown type (measured).
for name in ("nvmlVgpuTypeGetName", "nvmlVgpuTypeGetClass"):
    sz = U(0)
    t(name, INVALID, U(0), buf(), ctypes.byref(sz), label="id 0")
    t(name, INSUFFICIENT, U(1), buf(), ctypes.byref(sz), label="*size 0")
    if has(name):
        check(f"{name}: the size it wants is 64", sz.value == 64, sz.value)
    sz = U(63)
    t(name, INSUFFICIENT, U(1), None, ctypes.byref(sz), label="*size 63 beats the null buffer")
    t(name, INVALID, U(1), buf(), ctypes.byref(U(64)), label="*size 64: the type does not exist")
    t(name, INVALID, U(1), None, ctypes.byref(U(64)), label="null buffer")
    t(name, INVALID, U(1), buf(), None, label="null size")
# License: null buffer, id 0, short buffer, unknown type.
t("nvmlVgpuTypeGetLicense", INVALID, U(1), None, U(0), label="null buffer")
t("nvmlVgpuTypeGetLicense", INVALID, U(0), buf(), U(0), label="id 0")
t("nvmlVgpuTypeGetLicense", INSUFFICIENT, U(1), buf(), U(127), label="127 bytes")
t("nvmlVgpuTypeGetLicense", INVALID, U(1), buf(), U(128), label="128 bytes: the type does not exist")
# BAR1: null struct, id 0, version, unknown type. Max instances per GPU instance: the same, the id in the struct.
t("nvmlVgpuTypeGetBAR1Info", INVALID, U(1), None, label="null struct")
t("nvmlVgpuTypeGetBAR1Info", INVALID, U(0), versioned(BAD), label="id 0 beats the version")
t("nvmlVgpuTypeGetBAR1Info", VERSION_MISMATCH, U(1), versioned(BAD), label="bad version")
t("nvmlVgpuTypeGetBAR1Info", INVALID, U(1), versioned(VER["bar1"]), label="good version, unknown type")
mi = versioned(VER["max_instance"])
struct.pack_into("<I", mi, 4, 0)
t("nvmlVgpuTypeGetMaxInstancesPerGpuInstance", INVALID, mi, label="type 0")
mi = versioned(BAD)
struct.pack_into("<I", mi, 4, 0)
t("nvmlVgpuTypeGetMaxInstancesPerGpuInstance", INVALID, mi, label="type 0, bad version")
mi = versioned(BAD)
struct.pack_into("<I", mi, 4, 1)
t("nvmlVgpuTypeGetMaxInstancesPerGpuInstance", VERSION_MISMATCH, mi, label="type 1, bad version")
mi = versioned(VER["max_instance"])
struct.pack_into("<I", mi, 4, 1)
t("nvmlVgpuTypeGetMaxInstancesPerGpuInstance", INVALID, mi, label="type 1, good version: unknown type")
t("nvmlVgpuTypeGetMaxInstancesPerGpuInstance", INVALID, None, label="null struct")
for d in D:
    for tid in (0, 1):
        t("nvmlVgpuTypeGetMaxInstances", NOT_SUPPORTED, d, U(tid), ptr_u(), label=f"type {tid}")
    t("nvmlVgpuTypeGetMaxInstances", NOT_SUPPORTED, d, U(1), None, label="null pointer (looked at after)")
t("nvmlVgpuTypeGetMaxInstances", INVALID, BAD_DEV, U(1), ptr_u(), label="bad handle")

# ---- device-level vGPU: no vGPU manager ------------------------------------------------
for i, d in enumerate(D):
    s = f"device {i}"
    # NOT_SUPPORTED whatever the pointers (measured: null pointers too).
    t("nvmlDeviceGetActiveVgpus", NOT_SUPPORTED, d, ptr_u(), buf(), label=s)
    t("nvmlDeviceGetActiveVgpus", NOT_SUPPORTED, d, None, None, label=s + ", null pointers")
    t("nvmlDeviceGetCreatableVgpus", NOT_SUPPORTED, d, ptr_u(), buf(), label=s)
    t("nvmlDeviceGetCreatableVgpus", NOT_SUPPORTED, d, None, None, label=s + ", null pointers")
    t("nvmlDeviceGetSupportedVgpus", NOT_SUPPORTED, d, ptr_u(), buf(), label=s)
    t("nvmlDeviceGetSupportedVgpus", NOT_SUPPORTED, d, None, None, label=s + ", null pointers")
    t("nvmlDeviceGetVgpuProcessUtilization", NOT_SUPPORTED, d, ULL(0), ptr_u(), buf(), label=s)
    t("nvmlDeviceGetVgpuUtilization", NOT_SUPPORTED, d, ULL(0), ptr_u(), ptr_u(), buf(), label=s)
    t("nvmlDeviceGetVgpuHeterogeneousMode", NOT_SUPPORTED, d, versioned(VER["hetero"]), label=s)
    t("nvmlDeviceGetVgpuHeterogeneousMode", NOT_SUPPORTED, d, versioned(BAD), label=s + ", bad version (not looked at)")
    t("nvmlDeviceGetVgpuHeterogeneousMode", NOT_SUPPORTED, d, None, label=s + ", null struct (not looked at)")
    for fn in ("nvmlDeviceGetVgpuTypeCreatablePlacements", "nvmlDeviceGetVgpuTypeSupportedPlacements"):
        for tid in (0, 1):
            t(fn, NOT_SUPPORTED, d, U(tid), versioned(0x01000020), label=s + f", type {tid}")
            t(fn, NOT_SUPPORTED, d, U(tid), versioned(BAD), label=s + f", type {tid}, bad version")
            t(fn, NOT_SUPPORTED, d, U(tid), None, label=s + f", type {tid}, null struct")
    # Measured: the process utilization info is not looked at for a version, the instance one is.
    t("nvmlDeviceGetVgpuProcessesUtilizationInfo", NOT_SUPPORTED, d, versioned(VER["proc_util"]), label=s)
    t("nvmlDeviceGetVgpuProcessesUtilizationInfo", NOT_SUPPORTED, d, versioned(BAD), label=s + ", bad version")
    t("nvmlDeviceGetVgpuProcessesUtilizationInfo", INVALID, d, None, label=s + ", null struct")
    t("nvmlDeviceGetVgpuInstancesUtilizationInfo", NOT_SUPPORTED, d, versioned(VER["inst_util"]), label=s)
    t("nvmlDeviceGetVgpuInstancesUtilizationInfo", VERSION_MISMATCH, d, versioned(BAD), label=s + ", bad version")
    t("nvmlDeviceGetVgpuInstancesUtilizationInfo", INVALID, d, None, label=s + ", null struct")
    # A null result is INVALID_ARGUMENT first, then NOT_SUPPORTED.
    for fn, size in (("nvmlDeviceGetHostVgpuMode", 4), ("nvmlDeviceGetVgpuSchedulerCapabilities", 64),
                     ("nvmlDeviceGetVgpuSchedulerLog", 9624), ("nvmlDeviceGetVgpuSchedulerState", 64),
                     ("nvmlDeviceGetVgpuSchedulerLog_v2", 9624), ("nvmlDeviceGetVgpuSchedulerState_v2", 64),
                     ("nvmlDeviceGetConfComputeGpuAttestationReport", 9000),
                     ("nvmlDeviceGetConfComputeGpuCertificate", 9000), ("nvmlGpmQueryIfStreamingEnabled", 4)):
        t(fn, NOT_SUPPORTED, d, buf(size), label=s)
        t(fn, INVALID, d, None, label=s + ", null pointer")
    # NOT_SUPPORTED even for a capability the enum does not have, null pointer INVALID_ARGUMENT first.
    t("nvmlDeviceGetVgpuCapabilities", NOT_SUPPORTED, d, U(0), ptr_u(), label=s)
    t("nvmlDeviceGetVgpuCapabilities", NOT_SUPPORTED, d, U(99), ptr_u(), label=s + ", capability 99")
    t("nvmlDeviceGetVgpuCapabilities", INVALID, d, U(0), None, label=s + ", null pointer")
    # The size pointer is mandatory, the buffer is not; nothing is written.
    for fn in ("nvmlDeviceGetPgpuMetadataString", "nvmlDeviceGetVgpuMetadata"):
        t(fn, NOT_SUPPORTED, d, buf(), ctypes.byref(U(64)), label=s)
        t(fn, NOT_SUPPORTED, d, None, ctypes.byref(U(0)), label=s + ", null buffer")
        t(fn, NOT_SUPPORTED, d, None, ctypes.byref(U(64)), label=s + ", null buffer, size 64 (as the real library)")
        t(fn, INVALID, d, buf(), None, label=s + ", null size")
    # Bare metal: SUCCESS and NONE (measured).
    mode = U(99)
    t("nvmlDeviceGetVirtualizationMode", SUCCESS, d, ctypes.byref(mode), label=s)
    if has("nvmlDeviceGetVirtualizationMode"):
        check(f"{s}: virtualization mode is NONE", mode.value == 0, mode.value)
    t("nvmlDeviceGetVirtualizationMode", INVALID, d, None, label=s + ", null pointer")
    # Grid licensing: SUCCESS, nothing licensable (measured), the four spellings.
    for fn in ("nvmlDeviceGetGridLicensableFeatures", "nvmlDeviceGetGridLicensableFeatures_v2",
               "nvmlDeviceGetGridLicensableFeatures_v3", "nvmlDeviceGetGridLicensableFeatures_v4"):
        feat = buf(860, 0xFF)
        t(fn, SUCCESS, d, feat, label=s)
        if has(fn):
            check(f"{fn} {s}: not supported, no features", (u32(feat, 0), u32(feat, 4)) == (0, 0),
                  (u32(feat, 0), u32(feat, 4)))
        t(fn, INVALID, d, None, label=s + ", null pointer")
    # Host mode, GPM support.
    gpm = versioned(1)
    t("nvmlGpmQueryDeviceSupport", SUCCESS, d, gpm, label=s)
    if has("nvmlGpmQueryDeviceSupport"):
        check(f"{s}: GPM is not supported (no counter model)", u32(gpm, 4) == 0, u32(gpm, 4))
    t("nvmlGpmQueryDeviceSupport", VERSION_MISMATCH, d, versioned(BAD), label=s + ", bad version")
    t("nvmlGpmQueryDeviceSupport", VERSION_MISMATCH, d, buf(8), label=s + ", zero version")
    t("nvmlGpmQueryDeviceSupport", INVALID, d, None, label=s + ", null struct")

# A handle that is not a device is INVALID_ARGUMENT, everywhere (the real library crashes on some of these).
for bad in (NULL_DEV, BAD_DEV):
    t("nvmlDeviceGetVirtualizationMode", INVALID, bad, ptr_u(), label="bad handle")
    t("nvmlDeviceGetActiveVgpus", INVALID, bad, ptr_u(), buf(), label="bad handle")
    t("nvmlDeviceGetHostVgpuMode", INVALID, bad, ptr_u(), label="bad handle")
    t("nvmlDeviceGetGridLicensableFeatures_v4", INVALID, bad, buf(860), label="bad handle")
    t("nvmlDeviceGetConfComputeMemSizeInfo", INVALID, bad, buf(), label="bad handle")
    t("nvmlGpmQueryDeviceSupport", INVALID, bad, versioned(1), label="bad handle")
    t("nvmlDeviceWorkloadPowerProfileGetCurrentProfiles", INVALID, bad, versioned(VER["current_profiles"]),
      label="bad handle")
    t("nvmlDeviceReadWritePRM_v1", INVALID, bad, buf(), label="bad handle")
    t("nvmlDeviceSetVirtualizationMode", INVALID, bad, U(0), label="bad handle")
    t("nvmlDeviceVgpuForceGspUnload", INVALID, bad, label="bad handle")

# ---- GPU-instance vGPU: a null handle is invalid (as measured), the rest follows the device-level order ---------
GI = P(1)  # this group cannot see MIG's table: any non-null handle is taken to be a GPU instance
for fn, ver in (("nvmlGpuInstanceGetActiveVgpus", VER["active"]), ("nvmlGpuInstanceGetCreatableVgpus", VER["type_ids"]),
                ("nvmlGpuInstanceGetVgpuHeterogeneousMode", VER["hetero"]),
                ("nvmlGpuInstanceGetVgpuSchedulerLog", VER["sched_log"]),
                ("nvmlGpuInstanceGetVgpuSchedulerState", VER["sched_state_info"]),
                ("nvmlGpuInstanceGetVgpuTypeCreatablePlacements", VER["creatable"]),
                ("nvmlGpuInstanceSetVgpuSchedulerState", VER["sched_state"]),
                ("nvmlGpuInstanceGetVgpuSchedulerLog_v2", None), ("nvmlGpuInstanceGetVgpuSchedulerState_v2", None),
                ("nvmlGpuInstanceSetVgpuSchedulerState_v2", None)):
    arg = versioned(ver) if ver else buf()
    t(fn, INVALID, None, arg, label="null GPU instance")
    t(fn, INVALID, GI, None, label="null struct")
    t(fn, NOT_SUPPORTED, GI, arg, label="no vGPU host")
t("nvmlGpuInstanceGetActiveVgpus", VERSION_MISMATCH, GI, versioned(BAD), label="bad version")
t("nvmlGpuInstanceSetVgpuHeterogeneousMode", INVALID, None, versioned(VER["hetero"]), label="null GPU instance")
t("nvmlGpuInstanceSetVgpuHeterogeneousMode", NOT_SUPPORTED, GI, versioned(VER["hetero"]), label="no vGPU host")

# ---- GPM ---------------------------------------------------------------------------
# Allocating and freeing a sample is client-side: it works, and needs no GPU (measured).
s1, s2 = P(), P()
t("nvmlGpmSampleAlloc", INVALID, None, label="null pointer")
t("nvmlGpmSampleAlloc", SUCCESS, ctypes.byref(s1))
t("nvmlGpmSampleAlloc", SUCCESS, ctypes.byref(s2))
if has("nvmlGpmSampleAlloc"):
    check("samples are distinct, non-null", s1.value and s2.value and s1.value != s2.value, (s1.value, s2.value))
    for d in D:
        t("nvmlGpmSampleGet", NOT_SUPPORTED, d, s1, label="no counters to read")
        t("nvmlGpmMigSampleGet", NOT_SUPPORTED, d, U(0), s1, label="GPU instance 0")
        t("nvmlGpmMigSampleGet", NOT_SUPPORTED, d, U(1), s1, label="GPU instance 1")
        t("nvmlGpmSampleGet", INVALID, d, None, label="null sample")
        t("nvmlGpmMigSampleGet", INVALID, d, U(0), None, label="null sample")
        t("nvmlGpmSampleGet", INVALID, d, P(0x1000), label="a pointer that is not a sample")
    t("nvmlGpmSampleGet", INVALID, BAD_DEV, s1, label="bad handle")


class Metric(ctypes.Structure):
    _fields_ = [("metricId", ctypes.c_uint), ("nvmlReturn", ctypes.c_int), ("value", ctypes.c_double),
                ("shortName", P), ("longName", P), ("unit", P)]


def metrics_get(version=1, num=1, ids=(1,), a=None, b=None, extra=0):
    cap = 333  # NVML_GPM_METRIC_MAX in CUDA 13's header; older headers have fewer entries, which this struct overruns
    m_t = Metric * cap

    class MetricsGet(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint), ("numMetrics", ctypes.c_uint), ("sample1", P), ("sample2", P),
                    ("metrics", m_t)]

    m = MetricsGet()
    m.version, m.numMetrics, m.sample1, m.sample2 = version, num, a, b
    for i, mid in enumerate(ids):
        m.metrics[i].metricId = mid
    return m


if has("nvmlGpmMetricsGet") and has("nvmlGpmSampleAlloc"):
    t("nvmlGpmMetricsGet", INVALID, None, label="null struct")
    t("nvmlGpmMetricsGet", VERSION_MISMATCH, ctypes.byref(metrics_get(version=0, num=0, a=s1, b=s2)), label="zero version")
    t("nvmlGpmMetricsGet", VERSION_MISMATCH, ctypes.byref(metrics_get(version=7, a=s1, b=s2)), label="bad version")
    t("nvmlGpmMetricsGet", INVALID, ctypes.byref(metrics_get(num=0, a=s1, b=s2)), label="no metrics")
    t("nvmlGpmMetricsGet", INVALID, ctypes.byref(metrics_get(a=None, b=None)), label="no samples")
    t("nvmlGpmMetricsGet", INVALID, ctypes.byref(metrics_get(a=s1, b=None)), label="one sample")
    t("nvmlGpmMetricsGet", INVALID, ctypes.byref(metrics_get(a=s1, b=s1)), label="the same sample twice")
    t("nvmlGpmMetricsGet", INVALID, ctypes.byref(metrics_get(a=s1, b=P(0x1000))), label="a stray pointer")
    m = metrics_get(num=4, ids=(0, 1, 332, 333), a=s1, b=s2)
    t("nvmlGpmMetricsGet", SUCCESS, ctypes.byref(m), label="fresh samples")
    # Measured: ids 1..332 are real metrics whose samples were never read; 0 and 333 are not metrics.
    check("per-metric status: 0 invalid, 1 and 332 uninitialized, 333 invalid",
          [m.metrics[i].nvmlReturn for i in range(4)] == [INVALID, UNINITIALIZED, UNINITIALIZED, INVALID],
          [m.metrics[i].nvmlReturn for i in range(4)])
    t("nvmlGpmMetricsGet", INVALID, ctypes.byref(metrics_get(num=334, a=s1, b=s2)), label="more metrics than the array holds")
    # A shim built with an older header has a smaller metrics array and refuses the full set; skip then.
    rc = call("nvmlGpmMetricsGet", ctypes.byref(metrics_get(num=333, ids=tuple(range(1, 334)), a=s1, b=s2)))
    if rc == INVALID:
        print("skip  nvmlGpmMetricsGet 333 metrics (this build's header has a smaller array)")
    else:
        check("nvmlGpmMetricsGet 333 metrics -> 0", rc == SUCCESS, rc)

# Free: NULL is fine (measured), a stray pointer or a double free is INVALID_ARGUMENT.
t("nvmlGpmSampleFree", SUCCESS, None, label="NULL")
t("nvmlGpmSampleFree", INVALID, P(0x1000), label="not a sample")
t("nvmlGpmSampleFree", SUCCESS, s1)
t("nvmlGpmSampleFree", INVALID, s1, label="freed twice")
t("nvmlGpmSampleFree", SUCCESS, s2)
t("nvmlGpmSampleGet", INVALID, D[0], s1, label="a freed sample")

# ---- confidential computing -------------------------------------------------------------
caps = buf(8, 0xFF)
t("nvmlSystemGetConfComputeCapabilities", SUCCESS, caps)
if has("nvmlSystemGetConfComputeCapabilities"):
    check("no CC capabilities (cpu 0, gpus 0)", (u32(caps), u32(caps, 4)) == (0, 0), (u32(caps), u32(caps, 4)))
t("nvmlSystemGetConfComputeCapabilities", INVALID, None, label="null")
# NOT_SUPPORTED whatever the pointer (measured).
t("nvmlSystemGetConfComputeState", NOT_SUPPORTED, buf(12))
t("nvmlSystemGetConfComputeState", NOT_SUPPORTED, None, label="null pointer")
t("nvmlSystemGetConfComputeGpusReadyState", NOT_SUPPORTED, ptr_u())
t("nvmlSystemGetConfComputeGpusReadyState", NOT_SUPPORTED, None, label="null pointer")
t("nvmlSystemGetConfComputeSettings", NOT_SUPPORTED, versioned(0x01000014))
t("nvmlSystemGetConfComputeSettings", NOT_SUPPORTED, versioned(BAD), label="bad version (not looked at)")
t("nvmlSystemGetConfComputeSettings", NOT_SUPPORTED, None, label="null struct (not looked at)")
# Key rotation: null struct, version, then NOT_SUPPORTED.
t("nvmlSystemGetConfComputeKeyRotationThresholdInfo", NOT_SUPPORTED, versioned(VER["key_get"]))
t("nvmlSystemGetConfComputeKeyRotationThresholdInfo", VERSION_MISMATCH, versioned(BAD), label="bad version")
t("nvmlSystemGetConfComputeKeyRotationThresholdInfo", VERSION_MISMATCH, buf(16), label="zero version")
t("nvmlSystemGetConfComputeKeyRotationThresholdInfo", INVALID, None, label="null struct")
t("nvmlSystemSetConfComputeKeyRotationThresholdInfo", NOT_SUPPORTED, versioned(VER["key_set"]))
t("nvmlSystemSetConfComputeKeyRotationThresholdInfo", VERSION_MISMATCH, versioned(BAD), label="bad version")
t("nvmlSystemSetConfComputeKeyRotationThresholdInfo", INVALID, None, label="null struct")
t("nvmlSystemSetConfComputeGpusReadyState", NOT_SUPPORTED, U(0))
t("nvmlSystemSetConfComputeGpusReadyState", NOT_SUPPORTED, U(1))
for d in D:
    t("nvmlDeviceGetConfComputeMemSizeInfo", NOT_SUPPORTED, d, buf(16))
    t("nvmlDeviceGetConfComputeMemSizeInfo", NOT_SUPPORTED, d, None, label="null pointer (not looked at)")
    t("nvmlDeviceGetConfComputeProtectedMemoryUsage", NOT_SUPPORTED, d, buf(24))
    t("nvmlDeviceGetConfComputeProtectedMemoryUsage", NOT_SUPPORTED, d, None, label="null pointer (not looked at)")
    t("nvmlDeviceSetConfComputeUnprotectedMemSize", NOT_SUPPORTED, d, ULL(0))

# ---- PRM, power smoothing, workload power profiles ------------------------------------
for d in D:
    t("nvmlDeviceReadWritePRM_v1", NOT_SUPPORTED, d, buf(600))
    t("nvmlDeviceReadWritePRM_v1", INVALID, d, None, label="null buffer")
    t("nvmlDeviceReadPRMCounters_v1", NOT_SUPPORTED, d, buf(16))
    t("nvmlDeviceReadPRMCounters_v1", INVALID, d, None, label="null list")
    # Measured for the two getters: null struct, then version, then NOT_SUPPORTED.
    for fn, ver in (("nvmlDeviceWorkloadPowerProfileGetCurrentProfiles", VER["current_profiles"]),
                    ("nvmlDeviceWorkloadPowerProfileGetProfilesInfo", VER["profiles_info"])):
        t(fn, NOT_SUPPORTED, d, versioned(ver))
        t(fn, VERSION_MISMATCH, d, versioned(BAD), label="bad version")
        t(fn, VERSION_MISMATCH, d, buf(), label="zero version")
        t(fn, INVALID, d, None, label="null struct")
    for fn in ("nvmlDeviceWorkloadPowerProfileSetRequestedProfiles", "nvmlDeviceWorkloadPowerProfileClearRequestedProfiles"):
        t(fn, NOT_SUPPORTED, d, versioned(VER["requested"]))
        t(fn, VERSION_MISMATCH, d, versioned(BAD), label="bad version")
        t(fn, INVALID, d, None, label="null struct")
    t("nvmlDeviceWorkloadPowerProfileUpdateProfiles_v1", NOT_SUPPORTED, d, buf(40), label="no version word in this struct")
    t("nvmlDeviceWorkloadPowerProfileUpdateProfiles_v1", INVALID, d, None, label="null struct")
    t("nvmlDevicePowerSmoothingActivatePresetProfile", NOT_SUPPORTED, d, versioned(VER["smoothing_profile"]))
    t("nvmlDevicePowerSmoothingActivatePresetProfile", INVALID, d, None, label="null")
    t("nvmlDevicePowerSmoothingSetState", NOT_SUPPORTED, d, versioned(VER["smoothing_state"]))
    t("nvmlDevicePowerSmoothingSetState", INVALID, d, None, label="null")
    t("nvmlDevicePowerSmoothingUpdatePresetProfileParam", NOT_SUPPORTED, d, versioned(VER["smoothing_profile"]))
    t("nvmlDevicePowerSmoothingUpdatePresetProfileParam", VERSION_MISMATCH, d, versioned(BAD), label="bad version")
    t("nvmlDevicePowerSmoothingUpdatePresetProfileParam", INVALID, d, None, label="null")

# ---- setters: the feature is absent, so NOT_SUPPORTED with the root override ... and without it -------
# (policy: NO_PERMISSION is for a feature that would otherwise be present. Nothing in this file is, so a user that is
# not root is still told the feature is not there.)
def all_setters(label):
    for i, d in enumerate(D):
        s = f"{label}, device {i}"
        t("nvmlDeviceSetVirtualizationMode", NOT_SUPPORTED, d, U(0), label=s)
        t("nvmlDeviceSetVgpuCapabilities", NOT_SUPPORTED, d, U(0), U(1), label=s)
        t("nvmlDeviceSetVgpuHeterogeneousMode", NOT_SUPPORTED, d, versioned(VER["hetero"]), label=s)
        t("nvmlDeviceSetVgpuSchedulerState", NOT_SUPPORTED, d, buf(64), label=s)
        t("nvmlDeviceSetVgpuSchedulerState_v2", NOT_SUPPORTED, d, buf(64), label=s)
        t("nvmlDeviceVgpuForceGspUnload", NOT_SUPPORTED, d, label=s)
        t("nvmlGpmSetStreamingEnabled", NOT_SUPPORTED, d, U(1), label=s)
        t("nvmlDeviceSetConfComputeUnprotectedMemSize", NOT_SUPPORTED, d, ULL(1024), label=s)
        t("nvmlDevicePowerSmoothingSetState", NOT_SUPPORTED, d, versioned(VER["smoothing_state"]), label=s)
        t("nvmlDevicePowerSmoothingActivatePresetProfile", NOT_SUPPORTED, d, versioned(VER["smoothing_profile"]), label=s)
        t("nvmlDeviceWorkloadPowerProfileSetRequestedProfiles", NOT_SUPPORTED, d, versioned(VER["requested"]), label=s)
        t("nvmlDeviceWorkloadPowerProfileClearRequestedProfiles", NOT_SUPPORTED, d, versioned(VER["requested"]), label=s)
        t("nvmlDeviceWorkloadPowerProfileUpdateProfiles_v1", NOT_SUPPORTED, d, buf(40), label=s)
        t("nvmlDeviceReadWritePRM_v1", NOT_SUPPORTED, d, buf(600), label=s)
    t("nvmlSetVgpuVersion", NOT_SUPPORTED, buf(8), label=label)
    t("nvmlSystemSetConfComputeGpusReadyState", NOT_SUPPORTED, U(1), label=label)
    t("nvmlSystemSetConfComputeKeyRotationThresholdInfo", NOT_SUPPORTED, versioned(VER["key_set"]), label=label)
    t("nvmlUnitSetLedState", INVALID, P(1), U(1), label=label + " (no unit)")
    t("nvmlVgpuInstanceSetEncoderCapacity", NOT_FOUND, U(1), U(50), label=label + " (no instance)")
    t("nvmlVgpuInstanceClearAccountingPids", NOT_FOUND, U(1), label=label + " (no instance)")


root_before = os.environ.get("VGPU_NVML_ROOT")
for root in ("1", "0"):
    os.environ["VGPU_NVML_ROOT"] = root
    all_setters(f"VGPU_NVML_ROOT={root}")
if root_before is None:
    os.environ.pop("VGPU_NVML_ROOT", None)
else:
    os.environ["VGPU_NVML_ROOT"] = root_before
t("nvmlDeviceSetVirtualizationMode", INVALID, NULL_DEV, U(0), label="null handle")
t("nvmlDeviceSetVgpuCapabilities", INVALID, BAD_DEV, U(0), U(1), label="bad handle")
t("nvmlDeviceSetVgpuSchedulerState", INVALID, D[0], None, label="null struct")
t("nvmlDeviceSetVgpuSchedulerState_v2", INVALID, D[0], None, label="null struct")
t("nvmlSetVgpuVersion", INVALID, None, label="null range")
t("nvmlSystemSetConfComputeKeyRotationThresholdInfo", VERSION_MISMATCH, versioned(BAD), label="bad version")

# ---- after nvmlShutdown ---------------------------------------------------------------------
check("nvmlShutdown", lib.nvmlShutdown() == SUCCESS)
t("nvmlUnitGetCount", UNINITIALIZED, ptr_u(), label="after shutdown")
t("nvmlGetExcludedDeviceCount", UNINITIALIZED, ptr_u(), label="after shutdown")
t("nvmlGetVgpuVersion", UNINITIALIZED, buf(8), buf(8), label="after shutdown")
t("nvmlSystemGetHicVersion", UNINITIALIZED, ptr_u(), None, label="after shutdown")
t("nvmlVgpuInstanceGetType", UNINITIALIZED, U(1), ptr_u(), label="after shutdown")
t("nvmlVgpuTypeGetNumDisplayHeads", UNINITIALIZED, U(1), ptr_u(), label="after shutdown")
t("nvmlDeviceGetVirtualizationMode", UNINITIALIZED, D[0], ptr_u(), label="after shutdown")
t("nvmlDeviceGetActiveVgpus", UNINITIALIZED, D[0], ptr_u(), buf(), label="after shutdown")
t("nvmlGpmQueryDeviceSupport", UNINITIALIZED, D[0], versioned(1), label="after shutdown")
t("nvmlSystemGetConfComputeState", UNINITIALIZED, buf(12), label="after shutdown")
t("nvmlSystemSetConfComputeGpusReadyState", UNINITIALIZED, U(1), label="after shutdown")
t("nvmlDeviceSetVirtualizationMode", UNINITIALIZED, D[0], U(0), label="after shutdown")
# A sample is client memory: it needs no initialised library.
late = P()
t("nvmlGpmSampleAlloc", SUCCESS, ctypes.byref(late), label="after shutdown")
t("nvmlGpmSampleFree", SUCCESS, late, label="after shutdown")

print(f"{fails} failed, {skipped} skipped" + (f" (profile {profile})" if profile else ""))
sys.exit(1 if fails else 0)
