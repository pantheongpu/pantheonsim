"""NVML as NVIDIA's DCGM (`dcgmi diag`, NVVS) reads it, through ctypes against the shim.

usage: nvml_dcgm.py <shim dir> <vgpu binary> <gpu profile>

DCGM's hostengine resolves NVML entry points by name and builds its watched
fields from them: clock events, thermal and power violation time, ECC by
location, retired pages and remapped rows, PCIe link and replays, topology,
energy. Each check is one of those, read back after the fault that should move
it was injected with `vgpu fault`. The environment (VGPU_GPU, VGPU_DEVICE_COUNT,
VGPU_TELEMETRY_PATH, VGPU_STATE_DIR) is the runner's.
"""
import ctypes, os, subprocess, sys, time

SUCCESS, INVALID_ARGUMENT, NOT_SUPPORTED, INSUFFICIENT_SIZE = 0, 2, 3, 7
shim, vgpu, gpu = sys.argv[1], sys.argv[2], sys.argv[3]
lib = ctypes.CDLL(os.path.join(shim, "libnvidia-ml.so.1"))
fails = 0


def check(name, ok, got=""):
    global fails
    print(("ok    " if ok else "FAIL  ") + f"[{gpu}] " + name + ("" if ok else f"  -> {got}"))
    fails += 0 if ok else 1


def fault(*args):
    subprocess.run([vgpu, "fault", *args], check=True, capture_output=True)


class FieldValue(ctypes.Structure):
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_int), ("nvmlReturn", ctypes.c_int),
                ("value", ctypes.c_ulonglong)]


class Violation(ctypes.Structure):
    _fields_ = [("referenceTime", ctypes.c_ulonglong), ("violationTime", ctypes.c_ulonglong)]


def field(h, fid):
    fv = FieldValue(fieldId=fid)
    rc = lib.nvmlDeviceGetFieldValues(h, 1, ctypes.byref(fv))
    return fv.nvmlReturn if rc == SUCCESS else rc, fv.value


def u(fn, h, *pre):
    v = ctypes.c_uint(0xdead)
    rc = getattr(lib, fn)(h, *pre, ctypes.byref(v))
    return rc, v.value


def ull(fn, h, *pre):
    v = ctypes.c_ulonglong(0xdead)
    rc = getattr(lib, fn)(h, *pre, ctypes.byref(v))
    return rc, v.value


def violation(h, policy):
    v = Violation()
    rc = lib.nvmlDeviceGetViolationStatus(h, ctypes.c_int(policy), ctypes.byref(v))
    return rc, v.violationTime


assert lib.nvmlInit_v2() == SUCCESS
handles = []
for i in range(2):
    h = ctypes.c_void_p()
    assert lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i), ctypes.byref(h)) == SUCCESS
    handles.append(h)
h0, h1 = handles
hbm = gpu in ("nvidia/h100", "nvidia/a100")
has_ecc = gpu != "nvidia/rtx3060"

# The entry points DCGM's cache manager and NVVS reach for are all there.
wanted = """nvmlDeviceGetCurrentClocksEventReasons nvmlDeviceGetSupportedClocksEventReasons
nvmlDeviceGetViolationStatus nvmlDeviceGetBrand nvmlDeviceGetBoardId nvmlDeviceGetBusType
nvmlDeviceGetSupportedMemoryClocks nvmlDeviceGetSupportedGraphicsClocks nvmlDeviceGetApplicationsClock
nvmlDeviceGetClock nvmlDeviceGetDefaultEccMode nvmlDeviceGetDetailedEccErrors nvmlDeviceClearEccErrorCounts
nvmlDeviceGetRetiredPages nvmlDeviceGetRetiredPages_v2 nvmlDeviceGetRetiredPagesPendingStatus
nvmlDeviceGetRemappedRows nvmlDeviceGetPcieReplayCounter nvmlDeviceGetPcieSpeed nvmlDeviceGetPcieLinkMaxSpeed
nvmlDeviceGetTopologyCommonAncestor nvmlDeviceGetP2PStatus nvmlDeviceGetCpuAffinity nvmlDeviceGetMemoryAffinity
nvmlDeviceGetNvLinkState nvmlDeviceGetTotalEnergyConsumption nvmlDeviceGetInforomVersion""".split()
missing = [n for n in wanted if not hasattr(lib, n)]
check("every NVML entry point DCGM uses is exported", not missing, missing)

# Identity.
brand = ctypes.c_int(-1)
rc = lib.nvmlDeviceGetBrand(h0, ctypes.byref(brand))
want_brand = 5 if gpu == "nvidia/rtx3060" else 2   # NVML_BRAND_GEFORCE / NVML_BRAND_TESLA
check("brand follows the model", (rc, brand.value) == (SUCCESS, want_brand), (rc, brand.value))
b0, b1 = u("nvmlDeviceGetBoardId", h0)[1], u("nvmlDeviceGetBoardId", h1)[1]
check("each GPU is its own board", b0 != b1, (b0, b1))
check("the bus is PCIe", u("nvmlDeviceGetBusType", h0) == (SUCCESS, 2), u("nvmlDeviceGetBusType", h0))
buf = ctypes.create_string_buffer(16)
check("a virtual device has no infoROM", lib.nvmlDeviceGetInforomVersion(h0, 0, buf, 16) == NOT_SUPPORTED)

# Clocks.
_, mem_max = u("nvmlDeviceGetMaxClockInfo", h0, ctypes.c_int(2))
_, sm_max = u("nvmlDeviceGetMaxClockInfo", h0, ctypes.c_int(1))
n = ctypes.c_uint(0)
rc = lib.nvmlDeviceGetSupportedMemoryClocks(h0, ctypes.byref(n), None)
check("memory clocks: asking for the size says how many", (rc, n.value) == (INSUFFICIENT_SIZE, 1), (rc, n.value))
mem = (ctypes.c_uint * 1)()
rc = lib.nvmlDeviceGetSupportedMemoryClocks(h0, ctypes.byref(n), mem)
check("the one memory clock is the profile's", (rc, mem[0]) == (SUCCESS, mem_max), (rc, mem[0], mem_max))
n = ctypes.c_uint(0)
rc = lib.nvmlDeviceGetSupportedGraphicsClocks(h0, ctypes.c_uint(mem_max), ctypes.byref(n), None)
count = n.value
check("graphics clocks: the size comes back", rc == INSUFFICIENT_SIZE and count > 1, (rc, count))
gfx = (ctypes.c_uint * count)()
rc = lib.nvmlDeviceGetSupportedGraphicsClocks(h0, ctypes.c_uint(mem_max), ctypes.byref(n), gfx)
steps = {gfx[i] - gfx[i + 1] for i in range(count - 1)}
check("graphics clocks run from the maximum down in 15 MHz steps",
      rc == SUCCESS and gfx[0] == sm_max and steps == {15} and gfx[count - 1] >= 210, (rc, gfx[0], sm_max, steps))
rc = lib.nvmlDeviceGetSupportedGraphicsClocks(h0, ctypes.c_uint(mem_max + 1), ctypes.byref(n), gfx)
check("a memory clock the card does not have is refused", rc == INVALID_ARGUMENT, rc)
check("application clock is the maximum", u("nvmlDeviceGetApplicationsClock", h0, ctypes.c_int(1)) == (SUCCESS, sm_max))
check("a clock by id: current is the live clock",
      u("nvmlDeviceGetClock", h0, ctypes.c_int(1), ctypes.c_int(0)) == u("nvmlDeviceGetClockInfo", h0, ctypes.c_int(1)))
check("a clock by id: customer boost is the maximum",
      u("nvmlDeviceGetClock", h0, ctypes.c_int(1), ctypes.c_int(3)) == (SUCCESS, sm_max))

# Clock events and violation time: nothing before a throttle, time in the right
# policy after one, and only on the throttled GPU.
THERMAL, POWER = 1, 0
check("no thermal violation on a quiet card", violation(h0, THERMAL) == (SUCCESS, 0), violation(h0, THERMAL))
check("a policy NVML has no number for is refused", violation(h0, 7)[0] == INVALID_ARGUMENT)
rc, supported = ull("nvmlDeviceGetSupportedClocksEventReasons", h0)
check("supported reasons include thermal and power", rc == SUCCESS and supported & 0x24 == 0x24, (rc, supported))
fault("throttle", "--gpu", "0", "--reason", "sw_thermal_slowdown")
time.sleep(0.25)
rc, active = ull("nvmlDeviceGetCurrentClocksEventReasons", h0)
check("current event reasons carry the injected one", rc == SUCCESS and active & 0x20, (rc, active))
check("current and supported agree: nothing active is unsupported", active & ~supported == 0, (active, supported))
rc, thermal_ns = violation(h0, THERMAL)
check("thermal violation time accrues", rc == SUCCESS and thermal_ns >= 200_000_000, (rc, thermal_ns))
check("the power policy did not move", violation(h0, POWER) == (SUCCESS, 0), violation(h0, POWER))
check("the other GPU is not throttled", violation(h1, THERMAL) == (SUCCESS, 0), violation(h1, THERMAL))
rc, fld = field(h0, 75)
check("the same time through the field DCGM watches", rc == SUCCESS and fld >= thermal_ns, (rc, fld, thermal_ns))
rc, fld = field(h0, 269)
check("and through the sw thermal reason counter", rc == SUCCESS and fld >= 200_000_000, (rc, fld))
fault("throttle", "--gpu", "0", "--clear")
rc, kept = violation(h0, THERMAL)
time.sleep(0.1)
check("a cleared throttle stops counting", violation(h0, THERMAL) == (SUCCESS, kept), (kept, violation(h0, THERMAL)))

# ECC.
rc, mode = u("nvmlDeviceGetDefaultEccMode", h0)
check("default ECC mode follows the card", (rc == SUCCESS) == has_ecc, rc)
if has_ecc:
    check("ECC fields start at zero", [field(h0, f) for f in (3, 4, 5, 6, 11, 30)][:4] == [(SUCCESS, 0)] * 4)
    fault("inject", "--gpu", "0", "--ecc", "corrected", "--location", "l2_cache", "--count", "3")
    fault("inject", "--gpu", "0", "--ecc", "uncorrected", "--location", "dram")
    check("volatile single-bit total", field(h0, 3) == (SUCCESS, 3), field(h0, 3))
    check("volatile double-bit total", field(h0, 4) == (SUCCESS, 1), field(h0, 4))
    check("aggregate totals", (field(h0, 5), field(h0, 6)) == ((SUCCESS, 3), (SUCCESS, 1)))
    check("L2 single-bit, volatile and aggregate", (field(h0, 9), field(h0, 20)) == ((SUCCESS, 3), (SUCCESS, 3)))
    check("device-memory double-bit, volatile and aggregate", (field(h0, 12), field(h0, 23)) == ((SUCCESS, 1), (SUCCESS, 1)))
    check("the other GPU is clean", field(h1, 3) == (SUCCESS, 0) and field(h1, 4) == (SUCCESS, 0))

    class Counts(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("l1", "l2", "dev", "reg")]
    c = Counts()
    rc = lib.nvmlDeviceGetDetailedEccErrors(h0, 0, 0, ctypes.byref(c))
    check("detailed counts: corrected, volatile", (rc, c.l2, c.dev) == (SUCCESS, 3, 0), (rc, c.l2, c.dev))
    rc = lib.nvmlDeviceGetDetailedEccErrors(h0, 1, 1, ctypes.byref(c))
    check("detailed counts: uncorrected, aggregate", (rc, c.l2, c.dev) == (SUCCESS, 0, 1), (rc, c.l2, c.dev))

    # Memory taken out of service: pages on a GDDR card, rows on an HBM one.
    if hbm:
        corr, unc, pend, fail = (ctypes.c_uint(9) for _ in range(4))
        rc = lib.nvmlDeviceGetRemappedRows(h0, *(ctypes.byref(x) for x in (corr, unc, pend, fail)))
        check("a row is remapped, pending, not failed", (rc, unc.value, pend.value, fail.value) == (SUCCESS, 1, 1, 0),
              (rc, unc.value, pend.value, fail.value))
        check("remap fields: uncorrectable and pending", (field(h0, 143), field(h0, 144)) == ((SUCCESS, 1), (SUCCESS, 1)))
        check("no retired pages on an HBM card", u("nvmlDeviceGetRetiredPagesPendingStatus", h0)[0] == NOT_SUPPORTED
              and field(h0, 30)[0] == NOT_SUPPORTED)
    else:
        npages = ctypes.c_uint(0)
        rc = lib.nvmlDeviceGetRetiredPages(h0, 1, ctypes.byref(npages), None)
        check("a page is retired for the double-bit error", (rc, npages.value) == (INSUFFICIENT_SIZE, 1), (rc, npages.value))
        addrs, stamps = (ctypes.c_ulonglong * 1)(), (ctypes.c_ulonglong * 1)()
        rc = lib.nvmlDeviceGetRetiredPages_v2(h0, 1, ctypes.byref(npages), addrs, stamps)
        check("with an address", (rc, npages.value) == (SUCCESS, 1) and addrs[0] != 0, (rc, npages.value))
        pending = ctypes.c_int(-1)
        rc = lib.nvmlDeviceGetRetiredPagesPendingStatus(h0, ctypes.byref(pending))
        check("pending until the driver reloads", (rc, pending.value) == (SUCCESS, 1), (rc, pending.value))
        check("retired page fields", (field(h0, 30), field(h0, 31)) == ((SUCCESS, 1), (SUCCESS, 1)))
        check("no remapped rows on a GDDR card", field(h0, 143)[0] == NOT_SUPPORTED)
    # nvidia-smi -p: volatile only, aggregate stays.
    check("clear volatile counts", lib.nvmlDeviceClearEccErrorCounts(h0, 0) == SUCCESS)
    check("volatile is zero, aggregate stays", (field(h0, 3), field(h0, 5)) == ((SUCCESS, 0), (SUCCESS, 3)),
          (field(h0, 3), field(h0, 5)))
else:
    check("no ECC fields on a card without ECC", field(h0, 3)[0] == NOT_SUPPORTED and field(h0, 1)[0] == NOT_SUPPORTED)
    check("no detailed ECC counts either", lib.nvmlDeviceGetDetailedEccErrors(h0, 0, 0, ctypes.byref(ctypes.c_ulonglong())) == NOT_SUPPORTED)

# PCIe: the link, its speed, and replays.
_, gen = u("nvmlDeviceGetCurrPcieLinkGeneration", h0)
_, max_gen = u("nvmlDeviceGetMaxPcieLinkGeneration", h0)
per_gen = {1: 2500, 2: 5000, 3: 8000, 4: 16000, 5: 32000, 6: 64000}
check("PCIe speed is the trained generation's", u("nvmlDeviceGetPcieSpeed", h0) == (SUCCESS, per_gen[gen]),
      (u("nvmlDeviceGetPcieSpeed", h0), gen))
check("max link speed is the maximum generation, as NVML's enum",
      u("nvmlDeviceGetPcieLinkMaxSpeed", h0) == (SUCCESS, max_gen), (u("nvmlDeviceGetPcieLinkMaxSpeed", h0), max_gen))
fault("link", "--gpu", "1", "--gen", "1")
check("a degraded link shows in the speed, not the maximum",
      u("nvmlDeviceGetPcieSpeed", h1) == (SUCCESS, 2500) and u("nvmlDeviceGetPcieLinkMaxSpeed", h1) == (SUCCESS, max_gen),
      (u("nvmlDeviceGetPcieSpeed", h1), u("nvmlDeviceGetPcieLinkMaxSpeed", h1)))
fault("link", "--gpu", "1", "--clear")
check("no replays yet", u("nvmlDeviceGetPcieReplayCounter", h0) == (SUCCESS, 0))
fault("inject", "--gpu", "0", "--pcie", "replay", "--count", "4")
check("replays injected are replays counted", u("nvmlDeviceGetPcieReplayCounter", h0) == (SUCCESS, 4),
      u("nvmlDeviceGetPcieReplayCounter", h0))
check("and the other GPU's are not", u("nvmlDeviceGetPcieReplayCounter", h1) == (SUCCESS, 0))

# Topology: PCIe parts behind a host bridge, with peer access over PCIe.
lvl = ctypes.c_int(-1)
rc = lib.nvmlDeviceGetTopologyCommonAncestor(h0, h1, ctypes.byref(lvl))
check("two GPUs share a host bridge", (rc, lvl.value) == (SUCCESS, 30), (rc, lvl.value))
check("a GPU against itself is refused", lib.nvmlDeviceGetTopologyCommonAncestor(h0, h0, ctypes.byref(lvl)) == INVALID_ARGUMENT)
st = ctypes.c_int(-1)
rc = lib.nvmlDeviceGetP2PStatus(h0, h1, 0, ctypes.byref(st))
check("peer reads work", (rc, st.value) == (SUCCESS, 0), (rc, st.value))
rc = lib.nvmlDeviceGetP2PStatus(h0, h1, 2, ctypes.byref(st))
check("NVLink peer access is not offered", rc == SUCCESS and st.value != 0, (rc, st.value))
check("no NVLinks: state is NOT_SUPPORTED", lib.nvmlDeviceGetNvLinkState(h0, 0, ctypes.byref(ctypes.c_int())) == NOT_SUPPORTED)
check("no NVLinks: the link-count field is NOT_SUPPORTED, which DCGM reads as zero",
      field(h0, 91)[0] == NOT_SUPPORTED)

# Affinity.
words = (ctypes.c_ulong * 4)()
rc = lib.nvmlDeviceGetCpuAffinity(h0, 4, words)
bits = sum(bin(w).count("1") for w in words)
check("CPU affinity is some of this host's CPUs", rc == SUCCESS and 1 <= bits <= len(os.sched_getaffinity(0)), (rc, bits))
node = (ctypes.c_ulong * 1)()
rc = lib.nvmlDeviceGetMemoryAffinity(h0, 1, node, 0)
check("one memory node", (rc, node[0]) == (SUCCESS, 1), (rc, node[0]))

# Energy only goes up.
e1 = ull("nvmlDeviceGetTotalEnergyConsumption", h0)
time.sleep(0.3)
e2 = ull("nvmlDeviceGetTotalEnergyConsumption", h0)
f2 = field(h0, 83)
check("energy is monotonic and grows with time", e1[0] == e2[0] == SUCCESS and e2[1] > e1[1] and f2[1] >= e2[1], (e1, e2, f2))
check("GPU recovery action: none needed", field(h0, 230) == (SUCCESS, 0), field(h0, 230))

lib.nvmlShutdown()
print(f"{fails} failed")
sys.exit(1 if fails else 0)
