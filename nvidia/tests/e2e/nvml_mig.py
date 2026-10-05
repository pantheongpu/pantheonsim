"""NVML's MIG, GPU-instance and compute-instance entry points, through ctypes against the shim.

    python3 nvml_mig.py <shim dir> <profile id>      (run by run_nvml_category.sh inside `vgpu shell`)

Multi-Instance GPU partitioning is not simulated: no GPU instance, compute instance or MIG device
handle ever exists. So every device answers as one with MIG off, and this test pins down which
documented (and, for the RTX 3060, card-measured) status each entry point gives:

  * a part with MIG hardware (a100, h100 ...): nvmlDeviceGetMigMode succeeds with current = pending =
    disabled, nvmlDeviceSetMigMode(DISABLE) succeeds, ENABLE is NOT_SUPPORTED (it would promise
    instances this machine cannot create), the instance getters say "MIG not enabled";
  * a part without (the RTX 3060): the card's own answers, measured through NVIDIA's NVML 13.0.

Both device classes are run on both profile lists, so the file works whichever profiles
CMakeLists.txt registers; which class the profile is comes from MIG_PROFILES below.
The shim built against CUDA 12.0's header lacks the newest symbols: such a check is skipped.
"""
import ctypes, os, sys

SUCCESS, UNINITIALIZED, INVALID_ARGUMENT, NOT_SUPPORTED, NO_PERMISSION = 0, 1, 2, 3, 4
NOT_FOUND, ARGUMENT_VERSION_MISMATCH = 6, 25
LAST_PROFILE = 0x11   # the last profile value the driver recognises (card measurement)
MIG_PROFILES = {"nvidia/a100", "nvidia/a100-sxm4-40gb", "nvidia/h100", "nvidia/h100-pcie", "nvidia/h200",
                "nvidia/b200", "nvidia/b300", "nvidia/gh200-480gb"}

lib = ctypes.CDLL(os.path.join(sys.argv[1], "libnvidia-ml.so.1"))
profile = sys.argv[2] if len(sys.argv) > 2 else ""
capable = profile in MIG_PROFILES
fails = skipped = 0


def check(name, ok, got=""):
    global fails
    print(("ok    " if ok else "FAIL  ") + name + ("" if ok else f"  -> {got}"))
    fails += 0 if ok else 1


def skip(name, why):
    global skipped
    print(f"skip  {name}: {why}")
    skipped += 1


P, U, UP, H = ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p


def fn(name, *argtypes):
    """The entry point `name` with typed arguments, or None when this build of the shim lacks it."""
    if not hasattr(lib, name):
        skip(name, "not in this build (older nvml.h)")
        return None
    f = getattr(lib, name)
    f.restype = ctypes.c_int
    f.argtypes = list(argtypes)
    return f


class InfoV2(ctypes.Structure):    # nvmlGpuInstanceProfileInfo_v2_t
    _fields_ = [("version", U), ("id", U), ("isP2pSupported", U), ("sliceCount", U), ("instanceCount", U),
                ("multiprocessorCount", U), ("copyEngineCount", U), ("decoderCount", U), ("encoderCount", U),
                ("jpegCount", U), ("ofaCount", U), ("memorySizeMB", ctypes.c_ulonglong), ("name", ctypes.c_char * 96)]


class InfoV1(ctypes.Structure):    # nvmlGpuInstanceProfileInfo_t
    _fields_ = [("id", U), ("isP2pSupported", U), ("sliceCount", U), ("instanceCount", U),
                ("multiprocessorCount", U), ("copyEngineCount", U), ("decoderCount", U), ("encoderCount", U),
                ("jpegCount", U), ("ofaCount", U), ("memorySizeMB", ctypes.c_ulonglong)]


class CInfoV1(ctypes.Structure):   # nvmlComputeInstanceProfileInfo_t
    _fields_ = [("id", U), ("sliceCount", U), ("instanceCount", U), ("multiprocessorCount", U),
                ("sharedCopyEngineCount", U), ("sharedDecoderCount", U), ("sharedEncoderCount", U),
                ("sharedJpegCount", U), ("sharedOfaCount", U)]


class CInfoV2(ctypes.Structure):   # nvmlComputeInstanceProfileInfo_v2_t
    _fields_ = [("version", U), ("id", U), ("sliceCount", U), ("instanceCount", U), ("multiprocessorCount", U),
                ("sharedCopyEngineCount", U), ("sharedDecoderCount", U), ("sharedEncoderCount", U),
                ("sharedJpegCount", U), ("sharedOfaCount", U), ("name", ctypes.c_char * 96)]


class Placement(ctypes.Structure):  # nvmlGpuInstancePlacement_t / nvmlComputeInstancePlacement_t
    _fields_ = [("start", U), ("size", U)]


class GInfo(ctypes.Structure):     # nvmlGpuInstanceInfo_t
    _fields_ = [("device", P), ("id", U), ("profileId", U), ("placement", Placement)]


class CInfo(ctypes.Structure):     # nvmlComputeInstanceInfo_t
    _fields_ = [("device", P), ("gpuInstance", P), ("id", U), ("profileId", U), ("placement", Placement)]


def v2_version(cls, n):
    return ctypes.sizeof(cls) | (n << 24)


# The card accepts the word 0x02000098 / 0x03000098 for the GPU instance profile info.
assert ctypes.sizeof(InfoV2) == 0x98, ctypes.sizeof(InfoV2)

check("nvmlInit_v2", lib.nvmlInit_v2() == SUCCESS)
count = U()
lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
check("two devices", count.value == 2, count.value)
devices = []
for i in range(count.value):
    h = P()
    lib.nvmlDeviceGetHandleByIndex_v2(U(i), ctypes.byref(h))
    devices.append(h.value)
uuids = []
for h in devices:
    buf = ctypes.create_string_buffer(96)
    lib.nvmlDeviceGetUUID(P(h), buf, U(96))
    uuids.append(buf.value.decode())

BAD = [0, 0x7ff0dead0]   # NULL and a pointer that is not any device
NOT_AN_INSTANCE = 0x1234  # a pointer that is not an instance this library handed out (never dereferenced)

get_mig = fn("nvmlDeviceGetMigMode", H, UP, UP)
set_mig = fn("nvmlDeviceSetMigMode", H, U, ctypes.POINTER(ctypes.c_int))
max_mig = fn("nvmlDeviceGetMaxMigDeviceCount", H, UP)
mig_by_index = fn("nvmlDeviceGetMigDeviceHandleByIndex", H, U, ctypes.POINTER(P))
gi_id = fn("nvmlDeviceGetGpuInstanceId", H, UP)
ci_id = fn("nvmlDeviceGetComputeInstanceId", H, UP)
parent = fn("nvmlDeviceGetDeviceHandleFromMigDeviceHandle", H, ctypes.POINTER(P))
is_mig = fn("nvmlDeviceIsMigDeviceHandle", H, UP)
prof_info = fn("nvmlDeviceGetGpuInstanceProfileInfo", H, U, ctypes.POINTER(InfoV1))
prof_info_v = fn("nvmlDeviceGetGpuInstanceProfileInfoV", H, U, ctypes.POINTER(InfoV2))
prof_info_id = fn("nvmlDeviceGetGpuInstanceProfileInfoByIdV", H, U, ctypes.POINTER(InfoV2))
placements_v2 = fn("nvmlDeviceGetGpuInstancePossiblePlacements_v2", H, U, ctypes.POINTER(Placement), UP)
placements_v1 = fn("nvmlDeviceGetGpuInstancePossiblePlacements", H, U, ctypes.POINTER(Placement), UP)
remaining = fn("nvmlDeviceGetGpuInstanceRemainingCapacity", H, U, UP)
instances = fn("nvmlDeviceGetGpuInstances", H, U, ctypes.POINTER(P), UP)
instance_by_id = fn("nvmlDeviceGetGpuInstanceById", H, U, ctypes.POINTER(P))
create = fn("nvmlDeviceCreateGpuInstance", H, U, ctypes.POINTER(P))
create_placed = fn("nvmlDeviceCreateGpuInstanceWithPlacement", H, U, ctypes.POINTER(Placement), ctypes.POINTER(P))
gi_destroy = fn("nvmlGpuInstanceDestroy", P)
gi_info = fn("nvmlGpuInstanceGetInfo", P, ctypes.POINTER(GInfo))
ci_prof = fn("nvmlGpuInstanceGetComputeInstanceProfileInfo", P, U, U, ctypes.POINTER(CInfoV1))
ci_prof_v = fn("nvmlGpuInstanceGetComputeInstanceProfileInfoV", P, U, U, ctypes.POINTER(CInfoV2))
ci_remaining = fn("nvmlGpuInstanceGetComputeInstanceRemainingCapacity", P, U, UP)
ci_placements = fn("nvmlGpuInstanceGetComputeInstancePossiblePlacements", P, U, ctypes.POINTER(Placement), UP)
ci_create = fn("nvmlGpuInstanceCreateComputeInstance", P, U, ctypes.POINTER(P))
ci_create_placed = fn("nvmlGpuInstanceCreateComputeInstanceWithPlacement", P, U, ctypes.POINTER(Placement),
                      ctypes.POINTER(P))
ci_list = fn("nvmlGpuInstanceGetComputeInstances", P, U, ctypes.POINTER(P), UP)
ci_by_id = fn("nvmlGpuInstanceGetComputeInstanceById", P, U, ctypes.POINTER(P))
ci_destroy = fn("nvmlComputeInstanceDestroy", P)
ci_info_v2 = fn("nvmlComputeInstanceGetInfo_v2", P, ctypes.POINTER(CInfo))
ci_info = fn("nvmlComputeInstanceGetInfo", P, ctypes.POINTER(CInfo))


def ref(x):
    return ctypes.byref(x)


# --- MIG mode -------------------------------------------------------------------------------
# What a status function says about the mode on this profile: a part with MIG hardware has a mode
# (disabled), one without says NOT_SUPPORTED, as measured on the RTX 3060 through NVIDIA's NVML.
if get_mig:
    for n, h in enumerate(devices):
        cur, pend = U(7), U(7)
        rc = get_mig(h, ref(cur), ref(pend))
        if capable:
            check(f"GPU {n}: GetMigMode is SUCCESS, current = pending = disabled",
                  (rc, cur.value, pend.value) == (SUCCESS, 0, 0), (rc, cur.value, pend.value))
        else:
            check(f"GPU {n}: GetMigMode is NOT_SUPPORTED and leaves the outputs alone",
                  (rc, cur.value, pend.value) == (NOT_SUPPORTED, 7, 7), (rc, cur.value, pend.value))
    check("GetMigMode: NULL current is INVALID_ARGUMENT", get_mig(devices[0], None, ref(U())) == INVALID_ARGUMENT)
    check("GetMigMode: NULL pending is INVALID_ARGUMENT", get_mig(devices[0], ref(U()), None) == INVALID_ARGUMENT)
    for b in BAD:
        check(f"GetMigMode: bad handle {b:#x} is INVALID_ARGUMENT", get_mig(b, ref(U()), ref(U())) == INVALID_ARGUMENT)

if set_mig:
    act = ctypes.c_int(-99)
    for b in BAD:
        check(f"SetMigMode: bad handle {b:#x} is INVALID_ARGUMENT", set_mig(b, 0, ref(act)) == INVALID_ARGUMENT)
    # Arguments are checked before the feature, so even a part without MIG says INVALID_ARGUMENT.
    for mode in (2, 0xFFFFFFFF):
        check(f"SetMigMode: mode {mode:#x} is INVALID_ARGUMENT", set_mig(devices[0], mode, ref(act)) == INVALID_ARGUMENT)
    check("SetMigMode: NULL activationStatus is INVALID_ARGUMENT", set_mig(devices[0], 0, None) == INVALID_ARGUMENT)
    check("SetMigMode: nothing was written on an argument error", act.value == -99, act.value)

    for n, h in enumerate(devices):
        act = ctypes.c_int(-99)
        rc = set_mig(h, 0, ref(act))
        if capable:
            check(f"GPU {n}: SetMigMode(DISABLE) of a disabled device succeeds, activationStatus SUCCESS",
                  (rc, act.value) == (SUCCESS, SUCCESS), (rc, act.value))
        else:
            check(f"GPU {n}: SetMigMode(DISABLE) on a part without MIG is NOT_SUPPORTED",
                  rc == NOT_SUPPORTED, rc)
        act = ctypes.c_int(-99)
        rc = set_mig(h, 1, ref(act))
        if capable:
            # Enabling would promise instances this machine cannot create: refused, with the same
            # status in activationStatus.
            check(f"GPU {n}: SetMigMode(ENABLE) is NOT_SUPPORTED, activationStatus too",
                  (rc, act.value) == (NOT_SUPPORTED, NOT_SUPPORTED), (rc, act.value))
        else:
            check(f"GPU {n}: SetMigMode(ENABLE) on a part without MIG is NOT_SUPPORTED", rc == NOT_SUPPORTED, rc)
        if get_mig and capable:
            cur, pend = U(7), U(7)
            rc = get_mig(h, ref(cur), ref(pend))
            check(f"GPU {n}: after refusing ENABLE the mode is still disabled / disabled",
                  (rc, cur.value, pend.value) == (SUCCESS, 0, 0), (rc, cur.value, pend.value))

    # The pending mode is state (the Persistent settings overlay, as the inforom is): a value
    # left there is what GetMigMode reports as pending, current stays disabled, and a later
    # SetMigMode(DISABLE) writes the pending mode back.
    state = os.environ.get("VGPU_STATE_DIR", "")
    path = os.path.join(state, f"nvml-{uuids[0]}.conf") if state else ""
    if capable and get_mig and path and os.path.exists(path):
        check("SetMigMode(DISABLE) recorded the pending mode in the settings file",
              "mig_mode=0" in open(path).read().split(), open(path).read())
        lines = [ln for ln in open(path).read().split("\n") if ln and not ln.startswith("mig_mode=")]
        with open(path, "w") as f:
            f.write("\n".join(lines + ["mig_mode=1"]) + "\n")
        cur, pend = U(7), U(7)
        rc = get_mig(devices[0], ref(cur), ref(pend))
        check("a recorded pending mode ENABLE is reported as pending, current stays disabled",
              (rc, cur.value, pend.value) == (SUCCESS, 0, 1), (rc, cur.value, pend.value))
        act = ctypes.c_int(-99)
        rc = set_mig(devices[0], 0, ref(act))
        cur, pend = U(7), U(7)
        get_mig(devices[0], ref(cur), ref(pend))
        check("SetMigMode(DISABLE) clears a pending ENABLE",
              (rc, act.value, cur.value, pend.value) == (SUCCESS, SUCCESS, 0, 0), (rc, act.value, cur.value, pend.value))
    elif capable:
        skip("settings file check", "the settings directory is not reachable from the test (VGPU_STATE_DIR)")

    # Without root (VGPU_NVML_ROOT=0 beats the harness's =1) a setter is NO_PERMISSION wherever the feature
    # is present and the arguments are valid; a part without MIG still says NOT_SUPPORTED first.
    saved = os.environ.get("VGPU_NVML_ROOT")
    os.environ["VGPU_NVML_ROOT"] = "0"
    try:
        for mode in (0, 1):
            act = ctypes.c_int(-99)
            rc = set_mig(devices[0], mode, ref(act))
            if capable:
                check(f"without root: SetMigMode({mode}) is NO_PERMISSION and writes nothing",
                      (rc, act.value) == (NO_PERMISSION, -99), (rc, act.value))
            else:
                check(f"without root: SetMigMode({mode}) on a part without MIG is NOT_SUPPORTED", rc == NOT_SUPPORTED, rc)
        check("without root: an invalid mode is still INVALID_ARGUMENT", set_mig(devices[0], 5, ref(act)) == INVALID_ARGUMENT)
        check("without root: a bad handle is still INVALID_ARGUMENT", set_mig(0, 0, ref(act)) == INVALID_ARGUMENT)
        if get_mig and capable:
            cur, pend = U(7), U(7)
            rc = get_mig(devices[0], ref(cur), ref(pend))
            check("the getter needs no privilege", (rc, cur.value, pend.value) == (SUCCESS, 0, 0), (rc, cur.value, pend.value))
    finally:
        if saved is None:
            os.environ.pop("VGPU_NVML_ROOT", None)
        else:
            os.environ["VGPU_NVML_ROOT"] = saved

# --- no MIG devices exist ---------------------------------------------------------------------
# Card-measured (RTX 3060, NVIDIA's NVML 13.0): the count is SUCCESS 0, no index is found, the
# physical device is not a MIG handle.
for n, h in enumerate(devices):
    if max_mig:
        c = U(9)
        rc = max_mig(h, ref(c))
        check(f"GPU {n}: GetMaxMigDeviceCount is SUCCESS with 0", (rc, c.value) == (SUCCESS, 0), (rc, c.value))
    if mig_by_index:
        out = P(0x55)
        rc = mig_by_index(h, 0, ref(out))
        check(f"GPU {n}: GetMigDeviceHandleByIndex(0) is NOT_FOUND, output untouched",
              (rc, out.value) == (NOT_FOUND, 0x55), (rc, out.value))
    if gi_id:
        check(f"GPU {n}: GetGpuInstanceId of a physical device is INVALID_ARGUMENT", gi_id(h, ref(U())) == INVALID_ARGUMENT)
    if ci_id:
        check(f"GPU {n}: GetComputeInstanceId of a physical device is INVALID_ARGUMENT", ci_id(h, ref(U())) == INVALID_ARGUMENT)
    if parent:
        check(f"GPU {n}: GetDeviceHandleFromMigDeviceHandle of a physical device is INVALID_ARGUMENT",
              parent(h, ref(P())) == INVALID_ARGUMENT)
    if is_mig:
        v = U(99)
        rc = is_mig(h, ref(v))
        check(f"GPU {n}: IsMigDeviceHandle says no", (rc, v.value) == (SUCCESS, 0), (rc, v.value))
if max_mig:
    check("GetMaxMigDeviceCount: NULL count is INVALID_ARGUMENT", max_mig(devices[0], None) == INVALID_ARGUMENT)
    check("GetMaxMigDeviceCount: bad handle is INVALID_ARGUMENT", max_mig(0, ref(U())) == INVALID_ARGUMENT)
if mig_by_index:
    check("GetMigDeviceHandleByIndex: NULL output is INVALID_ARGUMENT", mig_by_index(devices[0], 0, None) == INVALID_ARGUMENT)
    check("GetMigDeviceHandleByIndex: bad handle is INVALID_ARGUMENT", mig_by_index(0, 0, ref(P())) == INVALID_ARGUMENT)
if gi_id:
    check("GetGpuInstanceId: NULL id and bad handle are INVALID_ARGUMENT",
          gi_id(devices[0], None) == INVALID_ARGUMENT and gi_id(0, ref(U())) == INVALID_ARGUMENT)

# --- GPU instance profiles (MIG off) ----------------------------------------------------------
# Profile values 0 .. LAST_PROFILE are NOT_SUPPORTED, above that INVALID_ARGUMENT: the answer a
# tool enumerating "from 0 until INVALID_ARGUMENT" (the documented way) stops at. Measured on the card.
TOO_BIG = (LAST_PROFILE + 1, 100, 0xFFFFFFFF)
for n, h in enumerate(devices):
    if prof_info:
        info = InfoV1()
        ok = all(prof_info(h, p, ref(info)) == NOT_SUPPORTED for p in range(LAST_PROFILE + 1))
        check(f"GPU {n}: GetGpuInstanceProfileInfo(0..{LAST_PROFILE}) is NOT_SUPPORTED", ok)
        check(f"GPU {n}: GetGpuInstanceProfileInfo(>{LAST_PROFILE}) is INVALID_ARGUMENT",
              all(prof_info(h, p, ref(info)) == INVALID_ARGUMENT for p in TOO_BIG))
        check(f"GPU {n}: ...and writes nothing", bytes(info) == bytes(InfoV1()))
    if prof_info_v:
        for ver, label in ((v2_version(InfoV2, 2), "v2"), (v2_version(InfoV2, 3), "v3")):
            info = InfoV2()
            info.version = ver
            check(f"GPU {n}: GetGpuInstanceProfileInfoV {label}: profile 0 NOT_SUPPORTED, {LAST_PROFILE + 1} INVALID_ARGUMENT",
                  prof_info_v(h, 0, ref(info)) == NOT_SUPPORTED and prof_info_v(h, LAST_PROFILE + 1, ref(info)) == INVALID_ARGUMENT)
    if prof_info_id:
        for ver, label in ((v2_version(InfoV2, 2), "v2"), (v2_version(InfoV2, 3), "v3")):
            info = InfoV2()
            info.version = ver
            check(f"GPU {n}: GetGpuInstanceProfileInfoByIdV {label}: every id is NOT_SUPPORTED",
                  all(prof_info_id(h, i, ref(info)) == NOT_SUPPORTED for i in (0, 5, LAST_PROFILE, 100, 0xFFFFFFFF)))

# A version word that is not the v2 / v3 one is INVALID_ARGUMENT, not ARGUMENT_VERSION_MISMATCH: the
# documentation says "info->version are invalid", and the card answers so.
for name, f in (("GetGpuInstanceProfileInfoV", prof_info_v), ("GetGpuInstanceProfileInfoByIdV", prof_info_id)):
    if not f:
        continue
    for ver in (0, 1, (1 << 24) | ctypes.sizeof(InfoV2), (2 << 24) | 4, (4 << 24) | ctypes.sizeof(InfoV2)):
        info = InfoV2()
        info.version = ver
        rc = f(devices[0], 0, ref(info))
        check(f"{name}: version {ver:#010x} is INVALID_ARGUMENT", rc == INVALID_ARGUMENT, rc)
    info = InfoV2()
    info.version = v2_version(InfoV2, 2)
    check(f"{name}: NULL info is INVALID_ARGUMENT", f(devices[0], 0, None) == INVALID_ARGUMENT)
    for b in BAD:
        check(f"{name}: bad handle {b:#x} is INVALID_ARGUMENT", f(b, 0, ref(info)) == INVALID_ARGUMENT)
if prof_info:
    check("GetGpuInstanceProfileInfo: NULL info is INVALID_ARGUMENT", prof_info(devices[0], 0, None) == INVALID_ARGUMENT)
    check("GetGpuInstanceProfileInfo: bad handle is INVALID_ARGUMENT",
          all(prof_info(b, 0, ref(InfoV1())) == INVALID_ARGUMENT for b in BAD))

# --- placements, capacity, instances of a profile ---------------------------------------------
# MIG part with MIG off: every profile up to LAST_PROFILE is NOT_SUPPORTED. A part without MIG
# (card-measured): profile 0 is NOT_SUPPORTED, every other value INVALID_ARGUMENT.
def profile_status(p):
    if capable:
        return NOT_SUPPORTED if p <= LAST_PROFILE else INVALID_ARGUMENT
    return NOT_SUPPORTED if p == 0 else INVALID_ARGUMENT


PROFILES = (0, 1, 4, LAST_PROFILE, LAST_PROFILE + 1, 100, 0xFFFFFFFF)
for n, h in enumerate(devices):
    for name, f in (("PossiblePlacements_v2", placements_v2), ("PossiblePlacements", placements_v1)):
        if not f:
            continue
        pl = (Placement * 8)()
        ok, got = True, []
        for p in PROFILES:
            c = U(5)
            rc = f(h, p, pl, ref(c))
            got.append((p, rc))
            ok = ok and rc == profile_status(p) and c.value == 5
        check(f"GPU {n}: GetGpuInstance{name} follows the profile rule and leaves count alone", ok, got)
        check(f"GPU {n}: GetGpuInstance{name} with NULL placements (discovery) gives the same",
              f(h, 0, None, ref(U(5))) == profile_status(0))
    if remaining:
        check(f"GPU {n}: GetGpuInstanceRemainingCapacity follows the profile rule",
              all(remaining(h, p, ref(U(5))) == profile_status(p) for p in PROFILES))
    if instances:
        buf = (P * 8)()
        check(f"GPU {n}: GetGpuInstances follows the profile rule",
              all(instances(h, p, buf, ref(U(8))) == profile_status(p) for p in PROFILES))
    if instance_by_id:
        out = P(0x55)
        want = NOT_SUPPORTED if capable else NOT_FOUND
        check(f"GPU {n}: GetGpuInstanceById: {'NOT_SUPPORTED (MIG not enabled)' if capable else 'NOT_FOUND (card)'}",
              all(instance_by_id(h, i, ref(out)) == want for i in (0, 1, 999)) and out.value == 0x55)
    if create:
        out = P(0x55)
        check(f"GPU {n}: CreateGpuInstance follows the profile rule (MIG is never enabled), nothing returned",
              all(create(h, p, ref(out)) == profile_status(p) for p in PROFILES) and out.value == 0x55)
    if create_placed:
        out = P(0x55)
        pl = Placement(0, 1)
        check(f"GPU {n}: CreateGpuInstanceWithPlacement follows the profile rule",
              all(create_placed(h, p, ref(pl), ref(out)) == profile_status(p) for p in PROFILES) and out.value == 0x55)

# Arguments: invalid ones are INVALID_ARGUMENT whatever the profile (checked before it), a bad
# handle likewise.
for name, f, args in (
        ("PossiblePlacements_v2", placements_v2, lambda h, p: (h, p, (Placement * 2)(), None)),
        ("PossiblePlacements", placements_v1, lambda h, p: (h, p, (Placement * 2)(), None)),
        ("RemainingCapacity", remaining, lambda h, p: (h, p, None)),
        ("GetGpuInstances (NULL count)", instances, lambda h, p: (h, p, (P * 2)(), None)),
        ("GetGpuInstances (NULL buffer)", instances, lambda h, p: (h, p, None, ref(U(2)))),
        ("GetGpuInstanceById", instance_by_id, lambda h, p: (h, p, None)),
        ("CreateGpuInstance", create, lambda h, p: (h, p, None)),
        ("CreateGpuInstanceWithPlacement (NULL output)", create_placed, lambda h, p: (h, p, ref(Placement(0, 1)), None)),
        ("CreateGpuInstanceWithPlacement (NULL placement)", create_placed, lambda h, p: (h, p, None, ref(P()))),
):
    if not f:
        continue
    check(f"{name}: a NULL argument is INVALID_ARGUMENT for every profile",
          all(f(*args(devices[0], p)) == INVALID_ARGUMENT for p in (0, 3, LAST_PROFILE, 100)))
    check(f"{name}: a bad device handle is INVALID_ARGUMENT",
          all(f(*args(b, 0)) == INVALID_ARGUMENT for b in BAD))

# --- GPU instance and compute instance handles --------------------------------------------------
# No instance exists, so no handle is valid: NULL (measured on the card) and any other pointer
# are INVALID_ARGUMENT, whatever else is passed. A destroyer is not privileged until it has a handle.
handles = (0, NOT_AN_INSTANCE)
for h in handles:
    tag = f"handle {h:#x}"
    cases = [
        ("nvmlGpuInstanceDestroy", gi_destroy, (h,)),
        ("nvmlGpuInstanceGetInfo", gi_info, (h, ref(GInfo()))),
        ("nvmlGpuInstanceGetComputeInstanceProfileInfo", ci_prof, (h, 0, 0, ref(CInfoV1()))),
        ("nvmlGpuInstanceGetComputeInstanceProfileInfoV", ci_prof_v, (h, 0, 0, ref(CInfoV2()))),
        ("nvmlGpuInstanceGetComputeInstanceRemainingCapacity", ci_remaining, (h, 0, ref(U()))),
        ("nvmlGpuInstanceGetComputeInstancePossiblePlacements", ci_placements, (h, 0, (Placement * 2)(), ref(U()))),
        ("nvmlGpuInstanceCreateComputeInstance", ci_create, (h, 0, ref(P()))),
        ("nvmlGpuInstanceCreateComputeInstanceWithPlacement", ci_create_placed, (h, 0, ref(Placement(0, 1)), ref(P()))),
        ("nvmlGpuInstanceGetComputeInstances", ci_list, (h, 0, (P * 2)(), ref(U()))),
        ("nvmlGpuInstanceGetComputeInstanceById", ci_by_id, (h, 0, ref(P()))),
        ("nvmlComputeInstanceDestroy", ci_destroy, (h,)),
        ("nvmlComputeInstanceGetInfo_v2", ci_info_v2, (h, ref(CInfo()))),
        ("nvmlComputeInstanceGetInfo", ci_info, (h, ref(CInfo()))),
    ]
    for name, f, args in cases:
        if f:
            rc = f(*args)
            check(f"{name}: {tag} is INVALID_ARGUMENT", rc == INVALID_ARGUMENT, rc)
# The same with a NULL output, and a version word, on the handle checks that take them.
if ci_prof_v:
    for ver in (0, v2_version(CInfoV2, 2), v2_version(CInfoV2, 3)):
        info = CInfoV2()
        info.version = ver
        check(f"GetComputeInstanceProfileInfoV: version {ver:#010x} on a NULL handle is INVALID_ARGUMENT",
              ci_prof_v(0, 0, 0, ref(info)) == INVALID_ARGUMENT)
if ci_prof:
    check("GetComputeInstanceProfileInfo: an engine profile past the last and a NULL output are INVALID_ARGUMENT",
          ci_prof(0, 0, 5, ref(CInfoV1())) == INVALID_ARGUMENT and ci_prof(0, 0, 0, None) == INVALID_ARGUMENT)

# Used before nvmlInit / after nvmlShutdown: UNINITIALIZED, for the entry points that take no device too.
check("nvmlShutdown", lib.nvmlShutdown() == SUCCESS)
if get_mig:
    check("after shutdown GetMigMode is UNINITIALIZED", get_mig(devices[0], ref(U()), ref(U())) == UNINITIALIZED)
if gi_destroy:
    check("after shutdown nvmlGpuInstanceDestroy is UNINITIALIZED", gi_destroy(NOT_AN_INSTANCE) == UNINITIALIZED)
if ci_info_v2:
    check("after shutdown nvmlComputeInstanceGetInfo_v2 is UNINITIALIZED", ci_info_v2(0, ref(CInfo())) == UNINITIALIZED)
if set_mig:
    check("after shutdown SetMigMode is UNINITIALIZED", set_mig(devices[0], 0, ref(ctypes.c_int())) == UNINITIALIZED)

print(f"{fails} failed, {skipped} skipped")
sys.exit(1 if fails else 0)
