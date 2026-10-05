"""NVML identity, clocks, fans, thermals, PCIe and affinity, through ctypes against the shim.

Run by run_nvml_category.sh once per profile (rtx3060, t4, h100): `python3 nvml_device.py <shim> <profile>`.
What the RTX 3060 answers (clock tables, two fans, 192-bit bus) was read from the card through NVIDIA's own
NVML; a datacenter profile records no clock table and has no fan, and says NOT_SUPPORTED to those queries.
"""
import ctypes, os, sys
from nvml_support import *

lib = Lib(sys.argv[1])
profile = sys.argv[2]
FACTS = {
    "nvidia/rtx3060": dict(bus=192, brand=5, fans=2, gen=4, cores=28 * 128, table=True, geforce=True, sm_max=2145),
    "nvidia/t4": dict(bus=256, brand=14, fans=0, gen=3, cores=40 * 64, table=False, geforce=False, sm_max=1590),
    "nvidia/h100": dict(bus=5120, brand=14, fans=0, gen=5, cores=132 * 128, table=False, geforce=False, sm_max=1980),
}
f = FACTS[profile]

check("nvmlInit_v2", lib.lib.nvmlInit_v2() == SUCCESS)
h0, h1 = lib.handle(0), lib.handle(1)


def val(name, h, ctype=ctypes.c_uint, *extra):
    x = ctype()
    rc = lib(name, h, *extra, ref(x))
    return rc, x.value


# ---- identity -------------------------------------------------------------------------------------
rc, v = val("nvmlDeviceGetBrand", h1, ctypes.c_int)
check("brand follows the profile", (rc, v) == (SUCCESS, f["brand"]), (rc, v))
rc, v = val("nvmlDeviceGetBusType", h1, ctypes.c_int)
check("every GPU is on PCIe (2)", (rc, v) == (SUCCESS, 2), (rc, v))
rc, v = val("nvmlDeviceGetMemoryBusWidth", h1)
check("memory bus width is the datasheet's", (rc, v) == (SUCCESS, f["bus"]), (rc, v))
rc0, b0 = val("nvmlDeviceGetBoardId", h0)
rc1, b1 = val("nvmlDeviceGetBoardId", h1)
check("two GPUs have two board ids", rc0 == rc1 == SUCCESS and b0 != b1, (b0, b1))
rc, v = val("nvmlDeviceGetMultiGpuBoard", h1)
check("not a multi-GPU board", (rc, v) == (SUCCESS, 0), (rc, v))
rc, v = val("nvmlDeviceGetModuleId", h1)
check("module id is the index", (rc, v) == (SUCCESS, 1), (rc, v))
rc, v = val("nvmlDeviceGetNumGpuCores", h1)
check("CUDA cores, not multiprocessors", (rc, v) == (SUCCESS, f["cores"]), (rc, v))
buf = ctypes.create_string_buffer(96)
check("no board part number", lib("nvmlDeviceGetBoardPartNumber", h1, buf, u(96)) == NOT_SUPPORTED)
check("no handle by serial", lib("nvmlDeviceGetHandleBySerial", b"0123", ref(ctypes.c_void_p())) == NOT_FOUND)
check("no GSP firmware", lib("nvmlDeviceGetGspFirmwareVersion", h1, buf) == NOT_SUPPORTED)
check("attributes belong to MIG handles", lib("nvmlDeviceGetAttributes_v2", h1, ctypes.create_string_buffer(64)) == NOT_SUPPORTED)
rc, v = val("nvmlDeviceGetIrqNum", h1)
check("no interrupt line", (rc, v) == (SUCCESS, 0), (rc, v))
x = ctypes.c_int(9)
check("two different GPUs: not on the same board", lib("nvmlDeviceOnSameBoard", h0, h1, ref(x)) == NOT_SUPPORTED)

# ---- the versioned handle lookup and PCI info ----------------------------------------------------------------
class PciExt(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint), ("device", ctypes.c_uint),
                ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint), ("baseClass", ctypes.c_uint),
                ("subClass", ctypes.c_uint), ("busId", ctypes.c_char * 32)]


class Pci(ctypes.Structure):
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


pci = Pci()
lib.lib.nvmlDeviceGetPciInfo_v3(h1, ref(pci))
ext = PciExt()
ext.version = version(PciExt, 1)
rc = lib("nvmlDeviceGetPciInfoExt", h1, ref(ext))
if rc is not None:
    check("PCI info with the class: same address, display controller", rc == SUCCESS and ext.busId == pci.busId
          and ext.baseClass == 3 and ext.subClass == (0 if f["geforce"] else 2), (rc, ext.busId, ext.baseClass, ext.subClass))
    ext.version = 7
    check("PCI info with a wrong version", lib("nvmlDeviceGetPciInfoExt", h1, ref(ext)) == VERSION_MISMATCH)


class UUID(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("type", ctypes.c_uint), ("value", ctypes.c_char * 41)]


uuid = ctypes.create_string_buffer(96)
lib.lib.nvmlDeviceGetUUID(h1, uuid, u(96))
q = UUID()
q.version, q.type, q.value = version(UUID, 1), 1, uuid.value
found = ctypes.c_void_p()
rc = lib("nvmlDeviceGetHandleByUUIDV", ref(q), ref(found))
if rc is not None:
    check("handle by versioned UUID", rc == SUCCESS and found.value == h1.value, (rc, found.value, h1.value))
    q.type = 0
    check("a UUID of type NONE is invalid", lib("nvmlDeviceGetHandleByUUIDV", ref(q), ref(found)) == INVALID_ARGUMENT)
    q.type, q.version = 1, 9
    check("a UUID with a wrong version", lib("nvmlDeviceGetHandleByUUIDV", ref(q), ref(found)) == VERSION_MISMATCH)

# ---- inforom ------------------------------------------------------------------------------------------------------
if f["table"]:
    rc, s = lib("nvmlDeviceGetInforomImageVersion", h1, buf, u(96)), buf.value
    check("inforom image version was read from the card", rc == SUCCESS and s == b"G001.0000.03.03", (rc, s))
    check("inforom OEM object 2.0", lib("nvmlDeviceGetInforomVersion", h1, ctypes.c_int(0), buf, u(96)) == SUCCESS and buf.value == b"2.0")
    check("no ECC inforom object on a GeForce", lib("nvmlDeviceGetInforomVersion", h1, ctypes.c_int(1), buf, u(96)) == NOT_SUPPORTED)
    check("an inforom object that does not exist", lib("nvmlDeviceGetInforomVersion", h1, ctypes.c_int(4), buf, u(96)) == INVALID_ARGUMENT)
    check("inforom image in a short buffer", lib("nvmlDeviceGetInforomImageVersion", h1, buf, u(4)) == INSUFFICIENT_SIZE)
    check("inforom checksum", val("nvmlDeviceGetInforomConfigurationChecksum", h1) == (SUCCESS, 0))
else:
    check("no inforom recorded", lib("nvmlDeviceGetInforomImageVersion", h1, buf, u(96)) == NOT_SUPPORTED)

# ---- clock tables ----------------------------------------------------------------------------------------------------
n = u(0)
rc = lib("nvmlDeviceGetSupportedMemoryClocks", h1, ref(n), None)
if f["table"]:
    check("supported memory clocks: the sizing call", rc == INSUFFICIENT_SIZE and n.value == 5, (rc, n.value))
    mem = (ctypes.c_uint * 8)()
    n = u(8)
    rc = lib("nvmlDeviceGetSupportedMemoryClocks", h1, ref(n), mem)
    check("the card's five memory clocks", rc == SUCCESS and list(mem)[:5] == [7501, 7301, 5001, 810, 405], (rc, list(mem)))
    n = u(3)
    check("a buffer for 3 of 5", lib("nvmlDeviceGetSupportedMemoryClocks", h1, ref(n), mem) == INSUFFICIENT_SIZE and n.value == 5)
    n = u(9)
    check("NULL buffer claiming room is invalid", lib("nvmlDeviceGetSupportedMemoryClocks", h1, ref(n), None) == INVALID_ARGUMENT)
    top = f["sm_max"] * 10
    want = [(top - 75 * k) // 10 for k in range(0, 400) if (top - 75 * k) // 10 >= 405]
    g = (ctypes.c_uint * 400)()
    n = u(400)
    rc = lib("nvmlDeviceGetSupportedGraphicsClocks", h1, u(7501), ref(n), g)
    check("graphics clocks: 7.5 MHz steps from the maximum to 405", rc == SUCCESS and list(g)[:n.value] == want, (rc, n.value, want[:3]))
    n = u(400)
    rc = lib("nvmlDeviceGetSupportedGraphicsClocks", h1, u(405), ref(n), g)
    check("only the idle range at the lowest memory clock", rc == SUCCESS and list(g)[:n.value] == [420, 412, 405], (rc, list(g)[:n.value]))
    n = u(1)
    check("graphics clocks in a short buffer", lib("nvmlDeviceGetSupportedGraphicsClocks", h1, u(7501), ref(n), g) == INSUFFICIENT_SIZE and n.value == len(want))
    n = u(400)
    check("a memory clock the GPU lacks: not found", lib("nvmlDeviceGetSupportedGraphicsClocks", h1, u(12345), ref(n), g) == NOT_FOUND)
    check("memory clock 0: invalid", lib("nvmlDeviceGetSupportedGraphicsClocks", h1, u(0), ref(n), g) == INVALID_ARGUMENT)
    ps = (ctypes.c_int * 16)()
    check("supported P-states are 8 5 3 2 0", lib("nvmlDeviceGetSupportedPerformanceStates", h1, ps, u(16)) == SUCCESS and list(ps)[:5] == [8, 5, 3, 2, 0])
    check("P-states in a short buffer", lib("nvmlDeviceGetSupportedPerformanceStates", h1, ps, u(6)) == INSUFFICIENT_SIZE)
    check("P-states with size 0", lib("nvmlDeviceGetSupportedPerformanceStates", h1, ps, u(0)) == INVALID_ARGUMENT)

    def mm(t, p):
        a, b = u(), u()
        rc = lib("nvmlDeviceGetMinMaxClockOfPState", h1, ctypes.c_int(t), ctypes.c_int(p), ref(a), ref(b))
        return rc, a.value, b.value
    check("graphics range in P0", mm(0, 0) == (SUCCESS, 210, f["sm_max"]), mm(0, 0))
    check("graphics range in P8 is the idle range", mm(0, 8) == (SUCCESS, 210, 420), mm(0, 8))
    check("memory clock of P8 and of P0", mm(2, 8) == (SUCCESS, 405, 405) and mm(2, 0) == (SUCCESS, 7501, 7501), (mm(2, 8), mm(2, 0)))
    check("a state the GPU does not run in", mm(0, 1)[0] == UNKNOWN, mm(0, 1))
    check("a clock type that is no domain", mm(4, 0)[0] == INVALID_ARGUMENT, mm(4, 0))
    check("video range in P8", mm(3, 8) == (SUCCESS, 555, 555), mm(3, 8))
    check("video range in P0", mm(3, 0) == (SUCCESS, 555, 1950), mm(3, 0))
else:
    check("no memory clock table: NOT_SUPPORTED", rc == NOT_SUPPORTED, rc)
    ps = (ctypes.c_int * 16)()
    check("no P-state table: NOT_SUPPORTED", lib("nvmlDeviceGetSupportedPerformanceStates", h1, ps, u(16)) == NOT_SUPPORTED)

# ---- current clocks ---------------------------------------------------------------------------------------------------
cur = {}
for t in range(4):
    rc, c = val("nvmlDeviceGetClock", h1, ctypes.c_uint, ctypes.c_int(t), ctypes.c_int(0))
    rc2, c2 = val("nvmlDeviceGetClockInfo", h1, ctypes.c_uint, ctypes.c_int(t))
    check(f"clock type {t}, id 0 is the clock info", rc == rc2 == SUCCESS and c == c2, (rc, c, rc2, c2))
check("application clock ids are not supported", val("nvmlDeviceGetClock", h1, ctypes.c_uint, ctypes.c_int(0), ctypes.c_int(1))[0] == NOT_SUPPORTED)
check("the counting enumerator is not supported", val("nvmlDeviceGetClock", h1, ctypes.c_uint, ctypes.c_int(4), ctypes.c_int(0))[0] == NOT_SUPPORTED)
check("a clock type past it is invalid", val("nvmlDeviceGetClock", h1, ctypes.c_uint, ctypes.c_int(9), ctypes.c_int(0))[0] == INVALID_ARGUMENT)
check("applications clock: not supported", lib("nvmlDeviceGetApplicationsClock", h1, ctypes.c_int(0), ref(u())) == NOT_SUPPORTED)
check("customer boost clock: not supported", lib("nvmlDeviceGetMaxCustomerBoostClock", h1, ctypes.c_int(0), ref(u())) == NOT_SUPPORTED)
x = ctypes.c_int()
check("auto boost: not supported", lib("nvmlDeviceGetAutoBoostedClocksEnabled", h1, ref(x), ref(ctypes.c_int())) == NOT_SUPPORTED)
check("API restriction: not supported", lib("nvmlDeviceGetAPIRestriction", h1, ctypes.c_int(0), ref(x)) == NOT_SUPPORTED)

class Freqs(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("str", ctypes.c_char * 2048)]


fr = Freqs()
fr.version = version(Freqs, 1)
rc = lib("nvmlDeviceGetCurrentClockFreqs", h1, ref(fr))
if rc is not None:
    if f["table"]:
        s = fr.str.decode()
        check("current clock frequencies string", rc == SUCCESS and s.startswith("nvclock=") and "memclock=" in s and "memTransferRate=" in s, (rc, s[:80]))
        rc = lib("nvmlDeviceGetPerformanceModes", h1, ref(fr))
        s = fr.str.decode()
        check("performance modes: five modes, the lowest the idle clocks", rc == SUCCESS and s.count("perf=") == 5 and s.startswith("perf=0, nvclock=210") and "memclock=405," in s, (rc, s[:100]))
    else:
        check("no frequency table: NOT_SUPPORTED", rc == NOT_SUPPORTED, rc)
    fr.version = 3
    check("clock frequencies with a wrong version", lib("nvmlDeviceGetCurrentClockFreqs", h1, ref(fr)) == VERSION_MISMATCH)


# ---- overclocking offsets ----------------------------------------------------------------------------------------------------
lo, hi = ctypes.c_int(), ctypes.c_int()
rc = lib("nvmlDeviceGetGpcClkMinMaxVfOffset", h1, ref(lo), ref(hi))
if f["table"]:
    check("graphics offset range", (rc, lo.value, hi.value) == (SUCCESS, -1000, 1000), (rc, lo.value, hi.value))
    rc = lib("nvmlDeviceGetMemClkMinMaxVfOffset", h1, ref(lo), ref(hi))
    check("memory offset range", (rc, lo.value, hi.value) == (SUCCESS, -2000, 6000), (rc, lo.value, hi.value))
    check("offset is 0 until set", lib("nvmlDeviceGetGpcClkVfOffset", h1, ref(lo)) == SUCCESS and lo.value == 0)

    class ClockOffset(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint), ("type", ctypes.c_int), ("pstate", ctypes.c_int), ("offset", ctypes.c_int),
                    ("min", ctypes.c_int), ("max", ctypes.c_int)]
    co = ClockOffset(version(ClockOffset, 1), 0, 0, 0, 0, 0)
    rc = lib("nvmlDeviceGetClockOffsets", h1, ref(co))
    if rc is not None:
        check("clock offsets, graphics", (rc, co.min, co.max) == (SUCCESS, -1000, 1000), (rc, co.min, co.max))
        co.type = 1
        check("clock offsets, SM domain: invalid", lib("nvmlDeviceGetClockOffsets", h1, ref(co)) == INVALID_ARGUMENT)
        co.type, co.version = 2, 1
        check("clock offsets with a wrong version", lib("nvmlDeviceGetClockOffsets", h1, ref(co)) == VERSION_MISMATCH)
else:
    check("no offset range recorded: NOT_SUPPORTED", rc == NOT_SUPPORTED, rc)

# ---- clock-event reasons and violations ----------------------------------------------------------------------------------------
rc, v = val("nvmlDeviceGetSupportedClocksThrottleReasons", h1, ctypes.c_ulonglong)
check("every reason is supported (0x1ff)", (rc, v) == (SUCCESS, 0x1ff), (rc, v))
rc, v = val("nvmlDeviceGetCurrentClocksEventReasons", h1, ctypes.c_ulonglong)
if rc is not None:
    check("an idle GPU reports the idle reason", (rc, v) == (SUCCESS, 1), (rc, v))


class Viol(ctypes.Structure):
    _fields_ = [("ref", ctypes.c_ulonglong), ("viol", ctypes.c_ulonglong)]


for p in range(6):
    vt = Viol()
    rc = lib("nvmlDeviceGetViolationStatus", h1, ctypes.c_int(p), ref(vt))
    check(f"violation policy {p}", rc == SUCCESS and vt.ref > 0 and vt.viol == 0, (rc, vt.ref, vt.viol))
check("violation policy 6 is invalid", lib("nvmlDeviceGetViolationStatus", h1, ctypes.c_int(6), ref(Viol())) == INVALID_ARGUMENT)

# ---- fans and coolers --------------------------------------------------------------------------------------------------------------
rc, v = val("nvmlDeviceGetNumFans", h1)
if f["fans"]:
    check("two fans", (rc, v) == (SUCCESS, 2), (rc, v))
    rc, s = val("nvmlDeviceGetFanSpeed_v2", h1, ctypes.c_uint, u(1))
    check("fan 1 speed", rc == SUCCESS and 0 <= s <= 100, (rc, s))
    check("fan 2 does not exist", val("nvmlDeviceGetFanSpeed_v2", h1, ctypes.c_uint, u(2))[0] == INVALID_ARGUMENT)
    lo, hi = u(), u()
    check("fan range 30..100", lib("nvmlDeviceGetMinMaxFanSpeed", h1, ref(lo), ref(hi)) == SUCCESS and (lo.value, hi.value) == (30, 100))
    rc, v = val("nvmlDeviceGetTargetFanSpeed", h1, ctypes.c_uint, u(0))
    check("target fan speed is the minimum at idle", (rc, v) == (SUCCESS, 30), (rc, v))
    rc, v = val("nvmlDeviceGetFanControlPolicy_v2", h1, ctypes.c_uint, u(0))
    check("fan policy defaults to temperature-controlled", (rc, v) == (SUCCESS, 0), (rc, v))

    class Cooler(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint), ("index", ctypes.c_uint), ("signal", ctypes.c_int), ("target", ctypes.c_int)]
    cl = Cooler(version(Cooler, 1), 0, 0, 0)
    rc = lib("nvmlDeviceGetCoolerInfo", h1, ref(cl))
    if rc is not None:
        check("cooler: variable signal, cools the GPU and its surroundings", (rc, cl.signal, cl.target) == (SUCCESS, 2, 14), (rc, cl.signal, cl.target))
        cl.version = 5
        check("cooler with a wrong version", lib("nvmlDeviceGetCoolerInfo", h1, ref(cl)) == VERSION_MISMATCH)

    class FanRpm(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint), ("fan", ctypes.c_uint), ("speed", ctypes.c_uint)]
    fan = FanRpm(version(FanRpm, 1), 0, 0)
    rc = lib("nvmlDeviceGetFanSpeedRPM", h1, ref(fan))
    if rc is not None:
        check("fan RPM: 0 for a stopped fan, N/A for a running one", rc in (SUCCESS, NOT_SUPPORTED), rc)
else:
    check("a passive board has no fans", rc == NOT_SUPPORTED, (rc, v))
    check("no fan speed", val("nvmlDeviceGetFanSpeed_v2", h1, ctypes.c_uint, u(0))[0] == NOT_SUPPORTED)

# ---- thermals ------------------------------------------------------------------------------------------------------------------------
class Temp(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("sensor", ctypes.c_int), ("temperature", ctypes.c_int)]


tv = Temp(version(Temp, 1), 0, 0)
rc = lib("nvmlDeviceGetTemperatureV", h1, ref(tv))
if rc is not None:
    t = u()
    lib.lib.nvmlDeviceGetTemperature(h1, ctypes.c_int(0), ref(t))
    check("versioned temperature is the GPU's", rc == SUCCESS and abs(tv.temperature - t.value) <= 5, (rc, tv.temperature, t.value))
    tv.sensor = 1
    check("a sensor that does not exist is invalid", lib("nvmlDeviceGetTemperatureV", h1, ref(tv)) == INVALID_ARGUMENT)
    tv.sensor, tv.version = 0, 2
    check("temperature with a wrong version", lib("nvmlDeviceGetTemperatureV", h1, ref(tv)) == VERSION_MISMATCH)
check("GetTemperature, sensor 1: invalid", lib.lib.nvmlDeviceGetTemperature(h1, ctypes.c_int(1), ref(u())) == INVALID_ARGUMENT)


class Margin(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("margin", ctypes.c_int)]


mg = Margin(version(Margin, 1), 0)
rc = lib("nvmlDeviceGetMarginTemperature", h1, ref(mg))
if rc is not None:
    check("margin temperature: Hopper reports it, older and GeForce parts do not",
          rc == (SUCCESS if profile == "nvidia/h100" else NOT_SUPPORTED), (rc, mg.margin))


class Thermal(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint), ("sensor", ctypes.c_int * 15)]


ts = Thermal()
rc = lib("nvmlDeviceGetThermalSettings", h1, u(0), ref(ts))
if f["geforce"]:
    check("thermal settings: one sensor, 0..127 C", rc == SUCCESS and ts.count == 1 and ts.sensor[1] == 0 and ts.sensor[2] == 127, (rc, ts.count))
else:
    check("thermal settings: not supported", rc == NOT_SUPPORTED, rc)

# ---- power ---------------------------------------------------------------------------------------------------------------------------
check("power source: AC", val("nvmlDeviceGetPowerSource", h1, ctypes.c_int) == (SUCCESS, 0))
check("power management: enabled", val("nvmlDeviceGetPowerManagementMode", h1, ctypes.c_int) == (SUCCESS, 1))
rc_a, a = val("nvmlDeviceGetPowerState", h1, ctypes.c_int)
rc_b, b = val("nvmlDeviceGetPerformanceState", h1, ctypes.c_int)
check("power state is the performance state", rc_a == rc_b == SUCCESS and a == b, (a, b))
if f["geforce"]:
    class PowerMizer(ctypes.Structure):
        _fields_ = [("current", ctypes.c_uint), ("mode", ctypes.c_uint), ("supported", ctypes.c_uint)]
    pm = PowerMizer()
    rc = lib("nvmlDeviceGetPowerMizerMode_v1", h1, ref(pm))
    if rc is not None:
        check("PowerMizer: adaptive, three modes supported", (rc, pm.current, pm.supported) == (SUCCESS, 0, 7), (rc, pm.current, pm.supported))

# ---- PCIe -------------------------------------------------------------------------------------------------------------------------------
MBPS = {3: 8000, 4: 16000, 5: 32000}
rc, v = val("nvmlDeviceGetPcieSpeed", h1)
check("PCIe speed in Mb/s", (rc, v) == (SUCCESS, MBPS[f["gen"]]), (rc, v))
check("PCIe link max speed enumerator", val("nvmlDeviceGetPcieLinkMaxSpeed", h1) == (SUCCESS, f["gen"]))
check("the GPU's maximum generation", val("nvmlDeviceGetGpuMaxPcieLinkGeneration", h1) == (SUCCESS, f["gen"]))
check("no PCIe replays", val("nvmlDeviceGetPcieReplayCounter", h1) == (SUCCESS, 0))

# ---- CPU and NUMA affinity ------------------------------------------------------------------------------------------------------------------
mask = (ctypes.c_ulong * 16)()
rc = lib("nvmlDeviceGetCpuAffinity", h1, u(16), mask)
check("CPU affinity is the CPUs the process may use", rc == SUCCESS and any(mask), rc)
check("CPU affinity with size 0", lib("nvmlDeviceGetCpuAffinity", h1, u(0), mask) == INVALID_ARGUMENT)
check("CPU affinity with no array", lib("nvmlDeviceGetCpuAffinity", h1, u(16), None) == INVALID_ARGUMENT)
check("CPU affinity, scope 7 is invalid", lib("nvmlDeviceGetCpuAffinityWithinScope", h1, u(16), mask, u(7)) == INVALID_ARGUMENT)
check("CPU affinity within the socket", lib("nvmlDeviceGetCpuAffinityWithinScope", h1, u(16), mask, u(1)) == SUCCESS)
node = u()
rc_n = lib("nvmlDeviceGetNumaNodeId", h1, ref(node))
if rc_n is not None:
    check("NUMA node: node 0 on a one-node host, N/A otherwise", rc_n in (SUCCESS, NOT_SUPPORTED) and (rc_n != SUCCESS or node.value == 0), (rc_n, node.value))
nodes = (ctypes.c_ulong * 4)()
rc = lib("nvmlDeviceGetMemoryAffinity", h1, u(4), nodes, u(0))
check("memory affinity", rc in (SUCCESS, NOT_SUPPORTED) and (rc != SUCCESS or nodes[0] == 1), rc)

# ---- NOT_SUPPORTED by design --------------------------------------------------------------------------------------------------------------------
check("clock monitor: not supported", lib("nvmlDeviceGetClkMonStatus", h1, ref(u())) == NOT_SUPPORTED)
check("bridge chips: not supported", lib("nvmlDeviceGetBridgeChipInfo", h1, ctypes.create_string_buffer(256)) == NOT_SUPPORTED)
check("GPU operation mode: not supported", lib("nvmlDeviceGetGpuOperationMode", h1, ref(ctypes.c_int()), ref(ctypes.c_int())) == NOT_SUPPORTED)
check("driver model v2 (Windows): not supported", lib("nvmlDeviceGetDriverModel_v2", h1, ref(ctypes.c_int()), ref(ctypes.c_int())) == NOT_SUPPORTED)

# ---- handle validation: every one of these refuses a handle that is no device -----------------------------------------------------------------------
for name, args in [("nvmlDeviceGetBrand", [ref(ctypes.c_int())]), ("nvmlDeviceGetMemoryBusWidth", [ref(u())]),
                   ("nvmlDeviceGetNumFans", [ref(u())]), ("nvmlDeviceGetPcieSpeed", [ref(u())]),
                   ("nvmlDeviceGetPcieReplayCounter", [ref(u())]), ("nvmlDeviceGetCpuAffinity", [u(1), mask]),
                   ("nvmlDeviceGetSupportedMemoryClocks", [ref(u()), None]), ("nvmlDeviceGetIrqNum", [ref(u())]),
                   ("nvmlDeviceGetViolationStatus", [ctypes.c_int(0), ref(Viol())])]:
    rc = lib(name, BAD, *args)
    check(f"{name}: a bad handle is invalid", rc in (None, INVALID_ARGUMENT), rc)
check("a NULL pointer where the answer goes", lib("nvmlDeviceGetBrand", h1, None) == INVALID_ARGUMENT)

check("shutdown", lib.lib.nvmlShutdown() == SUCCESS)
check("after shutdown: UNINITIALIZED", lib("nvmlDeviceGetBrand", h1, ref(ctypes.c_int())) == UNINITIALIZED)
finish()
