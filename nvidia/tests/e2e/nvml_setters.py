"""NVML setters, clearers and hot removal, against the shim.

Run by run_nvml_category.sh once per profile (t4, rtx3060, a100) with VGPU_NVML_ROOT=1, the override that
lets a process that is not root change settings. The real driver needs root for every one of these; here
the same call is NO_PERMISSION without the override (VGPU_NVML_ROOT=0, set below). What a setter changes
is kept in a per-device settings file, so the getters read it back -- and so does another process.

Nothing here was run on the card: the card's getters decide which features a GeForce has, and nvml.h's
return-code lists decide the rest. Checks go handle, arguments, feature, permission.
"""
import ctypes, os, subprocess, sys
from nvml_support import *

shim = sys.argv[1]
lib = Lib(shim)
profile = sys.argv[2]
GEFORCE = profile == "nvidia/rtx3060"
ECC = profile in ("nvidia/t4", "nvidia/a100")
check("nvmlInit_v2", lib.lib.nvmlInit_v2() == SUCCESS)
h0, h1 = lib.handle(0), lib.handle(1)
check("the override is on", os.environ.get("VGPU_NVML_ROOT") == "1", os.environ.get("VGPU_NVML_ROOT"))


def val(name, h, ctype=ctypes.c_uint, *extra):
    x = ctype()
    rc = lib(name, h, *extra, ref(x))
    return rc, x.value


def other_process(code):
    """Runs python code in a second process against the same shim and machine; prints its output."""
    r = subprocess.run([sys.executable, "-c",
        "import ctypes,sys\nl=ctypes.CDLL(sys.argv[1]+'/libnvidia-ml.so.1'); l.nvmlInit_v2()\n"
        "h=ctypes.c_void_p(); l.nvmlDeviceGetHandleByIndex_v2(1, ctypes.byref(h))\n" + code, shim],
        capture_output=True, text=True, timeout=60)
    return r.stdout.strip()


# ---- persistence and compute mode --------------------------------------------------------------------------------------
check("persistence mode is on by default", val("nvmlDeviceGetPersistenceMode", h1, ctypes.c_int) == (SUCCESS, 1))
check("turn persistence mode off", lib("nvmlDeviceSetPersistenceMode", h1, ctypes.c_int(0)) == SUCCESS)
check("the getter reads it back", val("nvmlDeviceGetPersistenceMode", h1, ctypes.c_int) == (SUCCESS, 0))
out = other_process("m=ctypes.c_int(9); l.nvmlDeviceGetPersistenceMode(h, ctypes.byref(m)); print(m.value)")
check("and so does another process", out == "0", out)
check("persistence mode 5 is not a mode", lib("nvmlDeviceSetPersistenceMode", h1, ctypes.c_int(5)) == INVALID_ARGUMENT)
check("turn it back on", lib("nvmlDeviceSetPersistenceMode", h1, ctypes.c_int(1)) == SUCCESS)
check("it is per device", val("nvmlDeviceGetPersistenceMode", h0, ctypes.c_int) == (SUCCESS, 1))
check("a bad handle", lib("nvmlDeviceSetPersistenceMode", BAD, ctypes.c_int(1)) == INVALID_ARGUMENT)
check("compute mode: exclusive process", lib("nvmlDeviceSetComputeMode", h1, ctypes.c_int(3)) == SUCCESS
      and val("nvmlDeviceGetComputeMode", h1, ctypes.c_int) == (SUCCESS, 3))
check("compute mode 9 is not a mode", lib("nvmlDeviceSetComputeMode", h1, ctypes.c_int(9)) == INVALID_ARGUMENT)
check("compute mode back to default", lib("nvmlDeviceSetComputeMode", h1, ctypes.c_int(0)) == SUCCESS
      and val("nvmlDeviceGetComputeMode", h1, ctypes.c_int) == (SUCCESS, 0))

# ---- ECC mode (pending until a reset that is not simulated) --------------------------------------------------------------------
cur, pend = ctypes.c_int(), ctypes.c_int()
if ECC:
    check("disable ECC", lib("nvmlDeviceSetEccMode", h1, ctypes.c_int(0)) == SUCCESS)
    rc = lib.lib.nvmlDeviceGetEccMode(h1, ref(cur), ref(pend))
    check("the change is pending, the current mode stays", (rc, cur.value, pend.value) == (SUCCESS, 1, 0), (rc, cur.value, pend.value))
    check("ECC mode 3 is not a mode", lib("nvmlDeviceSetEccMode", h1, ctypes.c_int(3)) == INVALID_ARGUMENT)
    check("enable ECC again", lib("nvmlDeviceSetEccMode", h1, ctypes.c_int(1)) == SUCCESS)
    lib.lib.nvmlDeviceGetEccMode(h1, ref(cur), ref(pend))
    check("pending is back on", pend.value == 1)
else:
    check("no ECC on a GeForce card", lib("nvmlDeviceSetEccMode", h1, ctypes.c_int(1)) == NOT_SUPPORTED)

# ---- power limit ---------------------------------------------------------------------------------------------------------------------
lo, hi = u(), u()
lib.lib.nvmlDeviceGetPowerManagementLimitConstraints(h1, ref(lo), ref(hi))
default = u()
lib.lib.nvmlDeviceGetPowerManagementDefaultLimit(h1, ref(default))
mid = (lo.value + hi.value) // 2
check("set a power limit inside the constraints", lib("nvmlDeviceSetPowerManagementLimit", h1, u(mid)) == SUCCESS)
check("it is the enforced limit", val("nvmlDeviceGetEnforcedPowerLimit", h1) == (SUCCESS, mid))
check("and the management limit", val("nvmlDeviceGetPowerManagementLimit", h1) == (SUCCESS, mid))
check("the default limit is the profile's", val("nvmlDeviceGetPowerManagementDefaultLimit", h1) == (SUCCESS, default.value))
out = other_process("m=ctypes.c_uint(); l.nvmlDeviceGetPowerManagementLimit(h, ctypes.byref(m)); print(m.value)")
check("another process sees the limit", out == str(mid), out)
check("a limit above the maximum", lib("nvmlDeviceSetPowerManagementLimit", h1, u(hi.value + 1000)) == INVALID_ARGUMENT)
check("a limit below the minimum", lib("nvmlDeviceSetPowerManagementLimit", h1, u(lo.value - 1)) == INVALID_ARGUMENT)


class PowerValue(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("scope", ctypes.c_ubyte), ("mw", ctypes.c_uint)]


pv = PowerValue(version(PowerValue, 1), 0, hi.value)
rc = lib("nvmlDeviceSetPowerManagementLimit_v2", h1, ref(pv))
if rc is not None:
    check("the versioned setter", rc == SUCCESS and val("nvmlDeviceGetPowerManagementLimit", h1) == (SUCCESS, hi.value), rc)
    pv.version = 5
    check("with a wrong version", lib("nvmlDeviceSetPowerManagementLimit_v2", h1, ref(pv)) == VERSION_MISMATCH)
    pv.version, pv.scope = version(PowerValue, 1), 1
    check("a scope that does not exist", lib("nvmlDeviceSetPowerManagementLimit_v2", h1, ref(pv)) == INVALID_ARGUMENT)

# ---- clock locks and offsets (the profiles with a clock table: the RTX 3060) -----------------------------------------------------------
sm = u()
if GEFORCE:
    check("lock the graphics clock to 1000..1500", lib("nvmlDeviceSetGpuLockedClocks", h1, u(1000), u(1500)) == SUCCESS)
    rc, v = val("nvmlDeviceGetClockInfo", h1, ctypes.c_uint, ctypes.c_int(1))
    check("the SM clock is held inside the lock", rc == SUCCESS and 1000 <= v <= 1500, (rc, v))
    check("min above max", lib("nvmlDeviceSetGpuLockedClocks", h1, u(1500), u(1000)) == INVALID_ARGUMENT)
    check("below the card's floor", lib("nvmlDeviceSetGpuLockedClocks", h1, u(100), u(200)) == INVALID_ARGUMENT)
    check("a number with a symbol", lib("nvmlDeviceSetGpuLockedClocks", h1, u(1000), u(0xffffff02)) == INVALID_ARGUMENT)
    check("TDP locks are not modelled", lib("nvmlDeviceSetGpuLockedClocks", h1, u(0xffffff01), u(0xffffff01)) == NOT_SUPPORTED)
    check("unlimited and unlimited is a reset", lib("nvmlDeviceSetGpuLockedClocks", h1, u(0xffffff02), u(0xffffff02)) == SUCCESS)
    check("lock again, then reset", lib("nvmlDeviceSetGpuLockedClocks", h1, u(1000), u(1500)) == SUCCESS and lib("nvmlDeviceResetGpuLockedClocks", h1) == SUCCESS)
    check("memory clock lock", lib("nvmlDeviceSetMemoryLockedClocks", h1, u(7501), u(7501)) == SUCCESS
          and val("nvmlDeviceGetClockInfo", h1, ctypes.c_uint, ctypes.c_int(2)) == (SUCCESS, 7501))
    check("and its reset", lib("nvmlDeviceResetMemoryLockedClocks", h1) == SUCCESS)
    check("offset 2000 is out of range", lib("nvmlDeviceSetGpcClkVfOffset", h1, ctypes.c_int(2000)) == INVALID_ARGUMENT)
    check("graphics offset +100", lib("nvmlDeviceSetGpcClkVfOffset", h1, ctypes.c_int(100)) == SUCCESS
          and val("nvmlDeviceGetGpcClkVfOffset", h1, ctypes.c_int) == (SUCCESS, 100))
    check("memory offset +500", lib("nvmlDeviceSetMemClkVfOffset", h1, ctypes.c_int(500)) == SUCCESS
          and val("nvmlDeviceGetMemClkVfOffset", h1, ctypes.c_int) == (SUCCESS, 500))

    class ClockOffset(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint), ("type", ctypes.c_int), ("pstate", ctypes.c_int), ("offset", ctypes.c_int),
                    ("min", ctypes.c_int), ("max", ctypes.c_int)]
    co = ClockOffset(version(ClockOffset, 1), 0, 0, -50, 0, 0)
    rc = lib("nvmlDeviceSetClockOffsets", h1, ref(co))
    if rc is not None:
        co2 = ClockOffset(version(ClockOffset, 1), 0, 0, 0, 0, 0)
        lib("nvmlDeviceGetClockOffsets", h1, ref(co2))
        check("the versioned offset setter, read back by its getter", rc == SUCCESS and co2.offset == -50, (rc, co2.offset))
        co.version = 3
        check("with a wrong version", lib("nvmlDeviceSetClockOffsets", h1, ref(co)) == VERSION_MISMATCH)
    lib("nvmlDeviceSetGpcClkVfOffset", h1, ctypes.c_int(0))
    lib("nvmlDeviceSetMemClkVfOffset", h1, ctypes.c_int(0))
else:
    check("no clock table recorded: locking is N/A", lib("nvmlDeviceSetGpuLockedClocks", h1, u(1000), u(1500)) == NOT_SUPPORTED)
    check("offsets are N/A", lib("nvmlDeviceSetGpcClkVfOffset", h1, ctypes.c_int(10)) == NOT_SUPPORTED)

# ---- fans ------------------------------------------------------------------------------------------------------------------------------------
if GEFORCE:
    check("fan 0 to 60 percent", lib("nvmlDeviceSetFanSpeed_v2", h1, u(0), u(60)) == SUCCESS)
    check("the speed reads back", val("nvmlDeviceGetFanSpeed_v2", h1, ctypes.c_uint, u(0)) == (SUCCESS, 60))
    check("setting a speed makes the policy manual", val("nvmlDeviceGetFanControlPolicy_v2", h1, ctypes.c_uint, u(0)) == (SUCCESS, 1))
    check("and the other fan is not touched", val("nvmlDeviceGetFanControlPolicy_v2", h1, ctypes.c_uint, u(1)) == (SUCCESS, 0))
    check("10 percent is below the minimum", lib("nvmlDeviceSetFanSpeed_v2", h1, u(0), u(10)) == INVALID_ARGUMENT)
    check("fan 2 does not exist", lib("nvmlDeviceSetFanSpeed_v2", h1, u(2), u(60)) == INVALID_ARGUMENT)
    check("restore the default", lib("nvmlDeviceSetDefaultFanSpeed_v2", h1, u(0)) == SUCCESS
          and val("nvmlDeviceGetFanControlPolicy_v2", h1, ctypes.c_uint, u(0)) == (SUCCESS, 0))
    check("a policy of 2 does not exist", lib("nvmlDeviceSetFanControlPolicy", h1, u(0), u(2)) == INVALID_ARGUMENT)
    check("manual policy", lib("nvmlDeviceSetFanControlPolicy", h1, u(1), u(1)) == SUCCESS
          and val("nvmlDeviceGetFanControlPolicy_v2", h1, ctypes.c_uint, u(1)) == (SUCCESS, 1))
    check("and back", lib("nvmlDeviceSetFanControlPolicy", h1, u(1), u(0)) == SUCCESS)

    class PowerMizer(ctypes.Structure):
        _fields_ = [("current", ctypes.c_uint), ("mode", ctypes.c_uint), ("supported", ctypes.c_uint)]
    pm = PowerMizer(0, 2, 0)
    rc = lib("nvmlDeviceSetPowerMizerMode_v1", h1, ref(pm))
    if rc is not None:
        got = PowerMizer()
        lib("nvmlDeviceGetPowerMizerMode_v1", h1, ref(got))
        check("PowerMizer mode auto, read back", rc == SUCCESS and got.current == 2, (rc, got.current))
        pm.mode = 3
        check("a PowerMizer mode the card lacks", lib("nvmlDeviceSetPowerMizerMode_v1", h1, ref(pm)) == INVALID_ARGUMENT)
        pm.mode = 0
        lib("nvmlDeviceSetPowerMizerMode_v1", h1, ref(pm))
else:
    check("a passive board has no fan to set", lib("nvmlDeviceSetFanSpeed_v2", h1, u(0), u(60)) == NOT_SUPPORTED)

# ---- ECC counters, fields, inforom ---------------------------------------------------------------------------------------------------------
total = ctypes.c_ulonglong()
if ECC:
    code, text = run_vgpu(shim, "fault", "inject", "--gpu", "1", "--ecc", "corrected", "--location", "l2_cache", "--count", "4")
    check("inject four corrected errors", code == 0, text)
    lib.lib.nvmlDeviceGetTotalEccErrors(h1, ctypes.c_int(0), ctypes.c_int(0), ref(total))
    check("volatile count 4", total.value == 4, total.value)
    check("clear the volatile counts", lib("nvmlDeviceClearEccErrorCounts", h1, ctypes.c_int(0)) == SUCCESS)
    lib.lib.nvmlDeviceGetTotalEccErrors(h1, ctypes.c_int(0), ctypes.c_int(0), ref(total))
    check("volatile is 0", total.value == 0, total.value)
    lib.lib.nvmlDeviceGetTotalEccErrors(h1, ctypes.c_int(0), ctypes.c_int(1), ref(total))
    check("aggregate is still 4", total.value == 4, total.value)
    check("clear the aggregate counts", lib("nvmlDeviceClearEccErrorCounts", h1, ctypes.c_int(1)) == SUCCESS)
    lib.lib.nvmlDeviceGetTotalEccErrors(h1, ctypes.c_int(0), ctypes.c_int(1), ref(total))
    check("aggregate is 0", total.value == 0, total.value)
    check("a counter type that does not exist", lib("nvmlDeviceClearEccErrorCounts", h1, ctypes.c_int(5)) == INVALID_ARGUMENT)
else:
    check("no ECC counts to clear", lib("nvmlDeviceClearEccErrorCounts", h1, ctypes.c_int(0)) == NOT_SUPPORTED)


class Field(ctypes.Structure):
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_int), ("nvmlReturn", ctypes.c_int), ("value", ctypes.c_ulonglong)]


fields = (Field * 2)()
fields[0].fieldId, fields[1].fieldId = 94, 180
check("clearing fields succeeds and each field says what became of it",
      lib("nvmlDeviceClearFieldValues", h1, ctypes.c_int(2), fields) == SUCCESS and [f.nvmlReturn for f in fields] == [NOT_SUPPORTED] * 2)
check("clearing no fields", lib("nvmlDeviceClearFieldValues", h1, ctypes.c_int(0), fields) == SUCCESS)
check("the inforom validates where one was read", lib("nvmlDeviceValidateInforom", h1) == (SUCCESS if GEFORCE else NOT_SUPPORTED))

# ---- the families the card does not have ----------------------------------------------------------------------------------------------------
check("API restriction", lib("nvmlDeviceSetAPIRestriction", h1, ctypes.c_int(0), ctypes.c_int(1)) == NOT_SUPPORTED)
check("application clocks", lib("nvmlDeviceSetApplicationsClocks", h1, u(7501), u(1500)) == NOT_SUPPORTED)
check("reset application clocks", lib("nvmlDeviceResetApplicationsClocks", h1) == NOT_SUPPORTED)
check("auto boost", lib("nvmlDeviceSetAutoBoostedClocksEnabled", h1, ctypes.c_int(1)) == NOT_SUPPORTED)
check("default auto boost", lib("nvmlDeviceSetDefaultAutoBoostedClocksEnabled", h1, ctypes.c_int(1), u(0)) == NOT_SUPPORTED)
check("driver model", lib("nvmlDeviceSetDriverModel", h1, ctypes.c_int(0), u(0)) == NOT_SUPPORTED)
check("GPU operation mode", lib("nvmlDeviceSetGpuOperationMode", h1, ctypes.c_int(0)) == NOT_SUPPORTED)
check("temperature threshold", lib("nvmlDeviceSetTemperatureThreshold", h1, ctypes.c_int(5), ref(ctypes.c_int(70))) == NOT_SUPPORTED)
check("a temperature threshold that does not exist", lib("nvmlDeviceSetTemperatureThreshold", h1, ctypes.c_int(99), ref(ctypes.c_int(70))) == INVALID_ARGUMENT)
check("NVLink counters", lib("nvmlDeviceResetNvLinkErrorCounters", h1, u(0)) in (NOT_SUPPORTED, INVALID_ARGUMENT))
check("NVLink bandwidth mode, system-wide", lib("nvmlSystemSetNvlinkBwMode", u(0)) in (None, NOT_SUPPORTED))

# ---- CPU affinity -----------------------------------------------------------------------------------------------------------------------------
before = os.sched_getaffinity(0)
check("bind to the GPU's CPUs", lib("nvmlDeviceSetCpuAffinity", h1) == SUCCESS and os.sched_getaffinity(0) == before)
check("clear the binding", lib("nvmlDeviceClearCpuAffinity", h1) == SUCCESS and os.sched_getaffinity(0) >= before)

# ---- accounting --------------------------------------------------------------------------------------------------------------------------------
check("accounting on", lib("nvmlDeviceSetAccountingMode", h1, ctypes.c_int(1)) == SUCCESS and val("nvmlDeviceGetAccountingMode", h1, ctypes.c_int) == (SUCCESS, 1))
check("accounting mode 4 is not a mode", lib("nvmlDeviceSetAccountingMode", h1, ctypes.c_int(4)) == INVALID_ARGUMENT)
check("clear the finished records", lib("nvmlDeviceClearAccountingPids", h1) == SUCCESS)
check("accounting off", lib("nvmlDeviceSetAccountingMode", h1, ctypes.c_int(0)) == SUCCESS)

# ---- drain state, hot removal ---------------------------------------------------------------------------------------------------------------------
class Pci(ctypes.Structure):
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


p1 = Pci()
lib.lib.nvmlDeviceGetPciInfo_v3(h1, ref(p1))
uuid = ctypes.create_string_buffer(96)
lib.lib.nvmlDeviceGetUUID(h1, uuid, u(96))
st = ctypes.c_int(-1)
check("drain GPU 1", lib("nvmlDeviceModifyDrainState", ref(p1), ctypes.c_int(1)) == SUCCESS
      and lib("nvmlDeviceQueryDrainState", ref(p1), ref(st)) == SUCCESS and st.value == 1)
check("undrain it", lib("nvmlDeviceModifyDrainState", ref(p1), ctypes.c_int(0)) == SUCCESS
      and lib("nvmlDeviceQueryDrainState", ref(p1), ref(st)) == SUCCESS and st.value == 0)
check("a drain state that does not exist", lib("nvmlDeviceModifyDrainState", ref(p1), ctypes.c_int(7)) == INVALID_ARGUMENT)
count = u()
check("remove GPU 1", lib("nvmlDeviceRemoveGpu", ref(p1)) == SUCCESS)
lib.lib.nvmlDeviceGetCount_v2(ref(count))
check("the machine has one GPU now", count.value == 1, count.value)
check("it is no longer found by UUID", lib.lib.nvmlDeviceGetHandleByUUID(uuid.value, ref(ctypes.c_void_p())) == NOT_FOUND)
check("nor by index", lib.lib.nvmlDeviceGetHandleByIndex_v2(u(1), ref(ctypes.c_void_p())) == INVALID_ARGUMENT)
check("removing it again", lib("nvmlDeviceRemoveGpu", ref(p1)) == NOT_FOUND)
check("discover it", lib("nvmlDeviceDiscoverGpus", ref(p1)) == SUCCESS)
lib.lib.nvmlDeviceGetCount_v2(ref(count))
check("two GPUs again", count.value == 2, count.value)
check("discovering a GPU that is there", lib("nvmlDeviceDiscoverGpus", ref(p1)) == NOT_FOUND)
check("remove it with the detach states", lib("nvmlDeviceRemoveGpu_v2", ref(p1), ctypes.c_int(1), ctypes.c_int(1)) == SUCCESS
      and lib("nvmlDeviceDiscoverGpus", ref(p1)) == SUCCESS)
check("a detach state that does not exist", lib("nvmlDeviceRemoveGpu_v2", ref(p1), ctypes.c_int(9), ctypes.c_int(0)) == INVALID_ARGUMENT)

# ---- without root ----------------------------------------------------------------------------------------------------------------------------------
os.environ["VGPU_NVML_ROOT"] = "0"
check("persistence mode without root", lib("nvmlDeviceSetPersistenceMode", h1, ctypes.c_int(0)) == NO_PERMISSION)
check("it did not change", val("nvmlDeviceGetPersistenceMode", h1, ctypes.c_int) == (SUCCESS, 1))
check("compute mode without root", lib("nvmlDeviceSetComputeMode", h1, ctypes.c_int(2)) == NO_PERMISSION)
check("power limit without root", lib("nvmlDeviceSetPowerManagementLimit", h1, u(hi.value)) == NO_PERMISSION)
check("accounting mode without root", lib("nvmlDeviceSetAccountingMode", h1, ctypes.c_int(1)) == NO_PERMISSION)
check("clearing accounting without root", lib("nvmlDeviceClearAccountingPids", h1) == NO_PERMISSION)
check("removing a GPU without root", lib("nvmlDeviceRemoveGpu", ref(p1)) == NO_PERMISSION)
check("draining without root", lib("nvmlDeviceModifyDrainState", ref(p1), ctypes.c_int(1)) == NO_PERMISSION)
check("invalid arguments are refused before permission", lib("nvmlDeviceSetPersistenceMode", h1, ctypes.c_int(5)) == INVALID_ARGUMENT)
check("an unsupported feature is refused before permission", lib("nvmlDeviceSetApplicationsClocks", h1, u(1), u(1)) == NOT_SUPPORTED)
if ECC:
    check("ECC mode without root", lib("nvmlDeviceSetEccMode", h1, ctypes.c_int(0)) == NO_PERMISSION)
    check("ECC counts without root", lib("nvmlDeviceClearEccErrorCounts", h1, ctypes.c_int(0)) == NO_PERMISSION)
if GEFORCE:
    check("clock locks without root", lib("nvmlDeviceSetGpuLockedClocks", h1, u(1000), u(1500)) == NO_PERMISSION)
    check("fan speed without root", lib("nvmlDeviceSetFanSpeed_v2", h1, u(0), u(60)) == NO_PERMISSION)
os.environ["VGPU_NVML_ROOT"] = "1"
check("with the override on again", lib("nvmlDeviceSetPersistenceMode", h1, ctypes.c_int(1)) == SUCCESS)
check("shutdown", lib.lib.nvmlShutdown() == SUCCESS)
finish()
