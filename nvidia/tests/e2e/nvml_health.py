"""NVML's health and diagnostic surface, through ctypes against the shim.

`nvml_health.py <shim dir> <kind>` runs against one simulated card, described
by VGPU_GPU, with nothing publishing telemetry; run_nvml_health.sh sets the
machine up. Errors are injected with `$VGPU_BIN fault ...`, in another process,
as a health tool's own test would, and read back through the NVML entry points
DCGM-style diagnostics use: ECC counts, retired pages, remapped rows,
clock-event reasons and violation times, PCIe replays, NVLink, accounting, and
the same numbers through nvmlDeviceGetFieldValues.

What each kind is, and so what it must answer (telemetry.cpp decides who
retires pages and who remaps rows):

  t4        Turing, GDDR, ECC: retires pages; no row remapping; no NVLink
  l4        Ada, GDDR, ECC: remaps rows (Ampere and later), retires no pages
  h100      Hopper, HBM: remaps rows; NVLink 4 x 18
  a100      Ampere HBM: remaps rows; NVLink 3 x 12; GA100 has no 1 s average power
  rtx3060   GeForce: no ECC, so every ECC, retirement and remap query is
            NOT_SUPPORTED
"""
import ctypes, os, subprocess, sys, time

SUCCESS, INVALID_ARGUMENT, NOT_SUPPORTED, NOT_FOUND, INSUFFICIENT_SIZE = 0, 2, 3, 6, 7
lib = ctypes.CDLL(os.path.join(sys.argv[1], "libnvidia-ml.so.1"))
kind = sys.argv[2]
vgpu = os.environ["VGPU_BIN"]
fails = 0


def check(name, ok, got=""):
    global fails
    print(("ok    " if ok else "FAIL  ") + f"[{kind}] " + name + ("" if ok else f"  -> {got}"))
    fails += 0 if ok else 1


def fault(*args):
    subprocess.run([vgpu, "fault", *args], check=True, stdout=subprocess.DEVNULL)


class FieldValue(ctypes.Structure):
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_int), ("nvmlReturn", ctypes.c_int),
                ("value", ctypes.c_ulonglong)]


class EccCounts(ctypes.Structure):
    _fields_ = [("l1", ctypes.c_ulonglong), ("l2", ctypes.c_ulonglong), ("dev", ctypes.c_ulonglong),
                ("reg", ctypes.c_ulonglong)]


class Violation(ctypes.Structure):
    _fields_ = [("reference", ctypes.c_ulonglong), ("violation", ctypes.c_ulonglong)]


class Histogram(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint) for n in ("max", "high", "partial", "low", "none")]


class AccountingStats(ctypes.Structure):
    _fields_ = [("gpuUtilization", ctypes.c_uint), ("memoryUtilization", ctypes.c_uint),
                ("maxMemoryUsage", ctypes.c_ulonglong), ("time", ctypes.c_ulonglong),
                ("startTime", ctypes.c_ulonglong), ("isRunning", ctypes.c_uint), ("reserved", ctypes.c_uint * 5)]


assert lib.nvmlInit_v2() == SUCCESS
h = ctypes.c_void_p()
assert lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) == SUCCESS
BAD = ctypes.c_void_p(0xdeadbeef)


def field(fid):
    """(nvmlReturn, value) of one field; a 32-bit field's upper bits are not its value."""
    f = (FieldValue * 1)()
    f[0].fieldId = fid
    assert lib.nvmlDeviceGetFieldValues(h, 1, f) == SUCCESS
    v = f[0].value & 0xFFFFFFFF if f[0].valueType == 1 else f[0].value
    return f[0].nvmlReturn, v


def uint(fn, *pre):
    v = ctypes.c_uint(0xAAAAAAAA)
    rc = fn(h, *pre, ctypes.byref(v))
    return rc, v.value


def ull(fn, *pre):
    v = ctypes.c_ulonglong(0xAAAAAAAAAAAAAAAA)
    rc = fn(h, *pre, ctypes.byref(v))
    return rc, v.value


def pages(cause, count=0, with_buffer=True):
    n = ctypes.c_uint(count)
    buf = (ctypes.c_ulonglong * max(count, 1))()
    rc = lib.nvmlDeviceGetRetiredPages(h, cause, ctypes.byref(n), buf if with_buffer else None)
    return rc, n.value, list(buf)[:n.value]


def remapped():
    v = [ctypes.c_uint(0xAAAA) for _ in range(4)]
    rc = lib.nvmlDeviceGetRemappedRows(h, *[ctypes.byref(x) for x in v])
    return rc, tuple(x.value for x in v)


def detailed(err, counter):
    c = EccCounts()
    rc = lib.nvmlDeviceGetDetailedEccErrors(h, err, counter, ctypes.byref(c))
    return rc, (c.l1, c.l2, c.dev, c.reg)


def violation(policy):
    v = Violation()
    rc = lib.nvmlDeviceGetViolationStatus(h, policy, ctypes.byref(v))
    return rc, v


ecc = kind != "rtx3060"
retires = kind == "t4"
remaps = kind in ("l4", "h100", "a100")
nvlinks = {"h100": (4, 18), "a100": (3, 12)}.get(kind)

# ---- ECC ----------------------------------------------------------------
if ecc:
    check("a clean card has no ECC errors, per location",
          detailed(1, 0) == (SUCCESS, (0, 0, 0, 0)) and detailed(0, 1) == (SUCCESS, (0, 0, 0, 0)), detailed(1, 0))
    mode = ctypes.c_int(-1)
    check("the default ECC mode is on",
          lib.nvmlDeviceGetDefaultEccMode(h, ctypes.byref(mode)) == SUCCESS and mode.value == 1, mode.value)
    check("ECC on is accepted (already so); off is not modelled",
          lib.nvmlDeviceSetEccMode(h, 1) == SUCCESS and lib.nvmlDeviceSetEccMode(h, 0) == NOT_SUPPORTED)
    check("ECC mode fields answer 1", field(1) == (SUCCESS, 1) and field(2) == (SUCCESS, 1), (field(1), field(2)))
else:
    check("no ECC: detailed errors are NOT_SUPPORTED", detailed(1, 0)[0] == NOT_SUPPORTED, detailed(1, 0))
    mode = ctypes.c_int(-1)
    check("no ECC: no default ECC mode", lib.nvmlDeviceGetDefaultEccMode(h, ctypes.byref(mode)) == NOT_SUPPORTED)
    check("no ECC: clearing counts and setting the mode are NOT_SUPPORTED",
          lib.nvmlDeviceClearEccErrorCounts(h, 0) == NOT_SUPPORTED and lib.nvmlDeviceSetEccMode(h, 1) == NOT_SUPPORTED)
    check("no ECC: the ECC fields are NOT_SUPPORTED",
          all(field(f)[0] == NOT_SUPPORTED for f in (1, 2, 3, 6, 12, 23, 28)), [field(f) for f in (1, 3, 12)])
    check("no ECC: no retired pages, no remapped rows, no histogram",
          pages(1)[0] == NOT_SUPPORTED and remapped()[0] == NOT_SUPPORTED
          and lib.nvmlDeviceGetRetiredPagesPendingStatus(h, ctypes.byref(mode)) == NOT_SUPPORTED
          and lib.nvmlDeviceGetRowRemapperHistogram(h, ctypes.byref(Histogram())) == NOT_SUPPORTED)
    check("no ECC: the retirement and remap fields are NOT_SUPPORTED",
          all(field(f)[0] == NOT_SUPPORTED for f in (29, 30, 31, 92, 93, 142, 143, 144, 145)))

# ---- page retirement and row remapping, clean ----------------------------
if retires:
    check("T4: no pages retired, none pending",
          pages(0)[:2] == (SUCCESS, 0) and pages(1)[:2] == (SUCCESS, 0)
          and uint(lib.nvmlDeviceGetRetiredPagesPendingStatus) == (SUCCESS, 0))
    check("T4 is Turing: no row remapping, and no histogram",
          remapped()[0] == NOT_SUPPORTED and lib.nvmlDeviceGetRowRemapperHistogram(h, ctypes.byref(Histogram())) == NOT_SUPPORTED)
    check("T4's retirement fields answer, its remap fields do not",
          field(29) == (SUCCESS, 0) and field(30) == (SUCCESS, 0) and field(31) == (SUCCESS, 0)
          and field(142)[0] == NOT_SUPPORTED and field(145)[0] == NOT_SUPPORTED, (field(30), field(142)))
if remaps:
    check(f"{kind}: no rows remapped, none pending, none failed", remapped() == (SUCCESS, (0, 0, 0, 0)), remapped())
    check(f"{kind} remaps rows rather than retiring pages",
          pages(1)[0] == NOT_SUPPORTED
          and lib.nvmlDeviceGetRetiredPagesPendingStatus(h, ctypes.byref(ctypes.c_int())) == NOT_SUPPORTED
          and field(30)[0] == NOT_SUPPORTED and field(31)[0] == NOT_SUPPORTED)
    check(f"{kind}: the remap fields answer zero",
          [field(f) for f in (142, 143, 144, 145)] == [(SUCCESS, 0)] * 4, [field(f) for f in (142, 143, 144, 145)])
    check(f"{kind}: the bank histogram needs data no profile has: NOT_SUPPORTED",
          lib.nvmlDeviceGetRowRemapperHistogram(h, ctypes.byref(Histogram())) == NOT_SUPPORTED)

if ecc:
    # ---- inject, then read it back every way ------------------------------
    fault("inject", "--gpu", "0", "--ecc", "corrected", "--location", "l2_cache", "--count", "2")
    fault("inject", "--gpu", "0", "--ecc", "uncorrected", "--location", "dram")
    check("corrected errors by location (single bit)", detailed(0, 0) == (SUCCESS, (0, 2, 0, 0)), detailed(0, 0))
    check("an uncorrected one in device memory (double bit), volatile and aggregate",
          detailed(1, 0) == (SUCCESS, (0, 0, 1, 0)) and detailed(1, 1) == (SUCCESS, (0, 0, 1, 0)),
          (detailed(1, 0), detailed(1, 1)))
    check("the ECC count fields agree: totals, L2, device memory",
          [field(f) for f in (3, 4, 5, 6, 9, 12, 20, 23)] ==
          [(SUCCESS, 2), (SUCCESS, 1), (SUCCESS, 2), (SUCCESS, 1), (SUCCESS, 2), (SUCCESS, 1), (SUCCESS, 2), (SUCCESS, 1)],
          [field(f) for f in (3, 4, 5, 6, 9, 12, 20, 23)])
    check("a location nothing injected into is zero", field(7) == (SUCCESS, 0) and field(25) == (SUCCESS, 0))
    check("the counts agree with nvmlDeviceGetMemoryErrorCounter",
          ull(lib.nvmlDeviceGetMemoryErrorCounter, 0, 0, 1) == (SUCCESS, 2)
          and ull(lib.nvmlDeviceGetTotalEccErrors, 1, 1) == (SUCCESS, 1))
    if retires:
        check("an uncorrected DRAM error retires a page, pending until a reload",
              uint(lib.nvmlDeviceGetRetiredPagesPendingStatus) == (SUCCESS, 1)
              and field(30) == (SUCCESS, 1) and field(31) == (SUCCESS, 1) and field(93) == (SUCCESS, 1)
              and field(29) == (SUCCESS, 0) and field(92) == (SUCCESS, 0))
        # Too small a buffer is INSUFFICIENT_SIZE with the count filled in.
        rc, n, _ = pages(1, count=0, with_buffer=False)
        check("a page list too small is INSUFFICIENT_SIZE, and says how many", (rc, n) == (INSUFFICIENT_SIZE, 1), (rc, n))
        rc, n, addrs = pages(1, count=4)
        rc2, n2, addrs2 = pages(1, count=4)
        check("the retired page is listed, 64 KiB aligned, and the same on every query",
              rc == SUCCESS and n == 1 and addrs == addrs2 and addrs[0] % 65536 == 0, (rc, n, addrs, addrs2))
        n = ctypes.c_uint(4)
        addr, ts = (ctypes.c_ulonglong * 4)(), (ctypes.c_ulonglong * 4)()
        check("the second form carries a timestamp per page",
              lib.nvmlDeviceGetRetiredPages_v2(h, 1, ctypes.byref(n), addr, ts) == SUCCESS and n.value == 1)
        check("nothing was retired for single-bit errors", pages(0)[:2] == (SUCCESS, 0))
        check("a card that retires pages has no rows to remap", remapped()[0] == NOT_SUPPORTED)
    else:
        check("an uncorrected DRAM error remaps a row, pending until a reload",
              remapped() == (SUCCESS, (0, 1, 1, 0)) and field(143) == (SUCCESS, 1)
              and field(144) == (SUCCESS, 1) and field(145) == (SUCCESS, 0) and field(142) == (SUCCESS, 0),
              (remapped(), field(143), field(144)))
        check("and retires no page", pages(1)[0] == NOT_SUPPORTED)
    # A reload completes what was pending and zeroes the volatile counts.
    fault("reset", "--gpu", "0", "--volatile")
    check("a driver reload: volatile counts zero, aggregate stay",
          detailed(0, 0)[1] == (0, 0, 0, 0) and detailed(1, 1)[1] == (0, 0, 1, 0) and field(3) == (SUCCESS, 0)
          and field(5) == (SUCCESS, 2), (detailed(0, 0), detailed(1, 1)))
    if retires:
        check("the retirement is complete, and the page stays retired",
              uint(lib.nvmlDeviceGetRetiredPagesPendingStatus) == (SUCCESS, 0) and pages(1, count=2)[1] == 1)
    else:
        check("the remap is complete, and the row stays remapped", remapped() == (SUCCESS, (0, 1, 0, 0)), remapped())
    check("clearing the aggregate counts leaves the retired page or remapped row",
          lib.nvmlDeviceClearEccErrorCounts(h, 1) == SUCCESS and detailed(1, 1)[1] == (0, 0, 0, 0)
          and field(5) == (SUCCESS, 0) and (pages(1, count=2)[1] == 1 if retires else remapped()[1][1] == 1))
    fault("inject", "--gpu", "0", "--ecc", "corrected", "--count", "3")
    check("clearing the volatile counts", lib.nvmlDeviceClearEccErrorCounts(h, 0) == SUCCESS and field(3) == (SUCCESS, 0)
          and field(5) == (SUCCESS, 3), (field(3), field(5)))
    check("an unknown counter type is INVALID_ARGUMENT", lib.nvmlDeviceClearEccErrorCounts(h, 7) == INVALID_ARGUMENT)
    check("a bogus handle is INVALID_ARGUMENT",
          lib.nvmlDeviceGetRemappedRows(BAD, *[ctypes.byref(ctypes.c_uint()) for _ in range(4)]) == INVALID_ARGUMENT
          and lib.nvmlDeviceGetDetailedEccErrors(BAD, 0, 0, ctypes.byref(EccCounts())) == INVALID_ARGUMENT)

# ---- performance state, throttle reasons, power and thermal ---------------
rc, p = uint(lib.nvmlDeviceGetPerformanceState)
rc2, p2 = uint(lib.nvmlDeviceGetPowerState)
check("an idle card is in P8, and the older name agrees", (rc, p, rc2, p2) == (SUCCESS, 8, SUCCESS, 8), (rc, p, rc2, p2))
rc, mask = ull(lib.nvmlDeviceGetSupportedClocksThrottleReasons)
check("the supported clock-event reasons are the driver's mask, 0x1FF", (rc, mask) == (SUCCESS, 0x1FF), (rc, hex(mask)))
if hasattr(lib, "nvmlDeviceGetCurrentClocksEventReasons"):
    rc, cur = ull(lib.nvmlDeviceGetCurrentClocksEventReasons)
    check("an idle card reports only the idle reason (new name)", (rc, cur) == (SUCCESS, 1), (rc, cur))
rc, v = violation(0)
check("no power violation time on a card that has not throttled", rc == SUCCESS and v.violation == 0 and v.reference > 0, (rc, v.violation))
check("nor thermal; and the field counters agree",
      violation(1)[1].violation == 0 and field(74) == (SUCCESS, 0) and field(75) == (SUCCESS, 0))
check("policies the model has no state for are NOT_SUPPORTED, bad ones INVALID_ARGUMENT",
      violation(2)[0] == NOT_SUPPORTED and violation(10)[0] == NOT_SUPPORTED and violation(99)[0] == INVALID_ARGUMENT)
fault("throttle", "--gpu", "0", "--reason", "sw_power_cap,sw_thermal_slowdown")
time.sleep(0.25)
rc, cur = ull(lib.nvmlDeviceGetCurrentClocksThrottleReasons)
check("an injected throttle is active: idle, power cap, thermal slowdown", (rc, cur) == (SUCCESS, 0x25), (rc, hex(cur)))
rc, power = violation(0)
rc2, therm = violation(1)
check("violation times run while it is active (nanoseconds), power and thermal apart",
      rc == SUCCESS and rc2 == SUCCESS and power.violation >= 150_000_000 and therm.violation >= 150_000_000,
      (power.violation, therm.violation))
check("the perf-policy fields agree", field(74)[1] >= 150_000_000 and field(75)[1] >= 150_000_000, (field(74), field(75)))
# A card whose driver reported no thermal threshold (an L4's profile) has none
# to read the temperature at; NVML says so rather than inventing one.
rc, slowdown = uint(lib.nvmlDeviceGetTemperatureThreshold, 1)
check("a thermal slowdown reads at the slowdown threshold, where the card has one",
      (rc == SUCCESS and slowdown == uint(lib.nvmlDeviceGetTemperature, 0)[1]) or rc == NOT_SUPPORTED, (rc, slowdown))
fault("throttle", "--gpu", "0", "--clear")
before = violation(0)[1].violation
time.sleep(0.15)
check("clearing keeps the time and stops it growing",
      before >= 150_000_000 and violation(0)[1].violation == before and ull(lib.nvmlDeviceGetCurrentClocksThrottleReasons)[1] == 1)

# Power: the limits and the draw. The profile carries the limit; the minimum is
# half of it, as nvmlDeviceGetPowerManagementLimitConstraints has always said.
limit = uint(lib.nvmlDeviceGetEnforcedPowerLimit)[1]
rc, instant = field(186)
check("instant power is answered, in milliwatts", rc == SUCCESS and 0 < instant <= limit, (rc, instant, limit))
check("power limits: min, max, default, current, requested",
      field(187) == (SUCCESS, limit // 2) and [field(f) for f in (188, 189, 190, 192)] == [(SUCCESS, limit)] * 4,
      [field(f) for f in (187, 188, 189, 190, 192)])
check("the 1 s average is Ampere (not GA100) and newer only",
      (field(185)[0] == SUCCESS) == (kind in ("h100", "l4", "rtx3060")), field(185))
mode = ctypes.c_int(-1)
check("power management is supported", lib.nvmlDeviceGetPowerManagementMode(h, ctypes.byref(mode)) == SUCCESS and mode.value == 1)

# ---- PCIe -----------------------------------------------------------------
check("no PCIe replays", uint(lib.nvmlDeviceGetPcieReplayCounter) == (SUCCESS, 0))
fault("inject", "--gpu", "0", "--pcie", "replay", "--count", "3")
fault("inject", "--gpu", "0", "--pcie", "bad_tlp", "--count", "2")
check("replays are counted, and the replay field and the bad-TLP field agree",
      uint(lib.nvmlDeviceGetPcieReplayCounter) == (SUCCESS, 3) and field(94) == (SUCCESS, 3) and field(176) == (SUCCESS, 2),
      (uint(lib.nvmlDeviceGetPcieReplayCounter), field(94), field(176)))

# ---- NVLink ---------------------------------------------------------------
if nvlinks:
    version, links = nvlinks
    check("NVLink link count field", field(91) == (SUCCESS, links), field(91))
    states = [uint(lib.nvmlDeviceGetNvLinkState, i) for i in range(links)]
    check("every link is up", all(s == (SUCCESS, 1) for s in states), states)
    check("a link past the last is INVALID_ARGUMENT",
          uint(lib.nvmlDeviceGetNvLinkState, links)[0] == INVALID_ARGUMENT)
    check("the NVLink version is the profile's", uint(lib.nvmlDeviceGetNvLinkVersion, 0) == (SUCCESS, version))
    check("peer access and atomics, no system-memory access",
          [uint(lib.nvmlDeviceGetNvLinkCapability, 0, c) for c in range(6)] ==
          [(SUCCESS, 1), (SUCCESS, 0), (SUCCESS, 1), (SUCCESS, 0), (SUCCESS, 0), (SUCCESS, 1)])
    check("the error counters are zero, and an unknown counter is INVALID_ARGUMENT",
          all(ull(lib.nvmlDeviceGetNvLinkErrorCounter, 0, c) == (SUCCESS, 0) for c in range(5))
          and ull(lib.nvmlDeviceGetNvLinkErrorCounter, 0, 9)[0] == INVALID_ARGUMENT
          and ull(lib.nvmlDeviceGetNvLinkErrorCounter, links, 0)[0] == INVALID_ARGUMENT)
    check("resetting a link's counters succeeds, past the last link does not",
          lib.nvmlDeviceResetNvLinkErrorCounters(h, 0) == SUCCESS and lib.nvmlDeviceResetNvLinkErrorCounters(h, links) == INVALID_ARGUMENT)
    check("the error fields: totals, and lanes 0 and 6 (the second id range)",
          [field(f) for f in (38, 45, 52, 59, 32, 96, 114)] == [(SUCCESS, 0)] * 7, [field(f) for f in (38, 96)])
    check("the remote end is not modelled",
          uint(lib.nvmlDeviceGetNvLinkRemoteDeviceType, 0)[0] == NOT_SUPPORTED)
else:
    check("no NVLink on this card: state, version, counters and fields are NOT_SUPPORTED",
          uint(lib.nvmlDeviceGetNvLinkState, 0)[0] == NOT_SUPPORTED and uint(lib.nvmlDeviceGetNvLinkVersion, 0)[0] == NOT_SUPPORTED
          and ull(lib.nvmlDeviceGetNvLinkErrorCounter, 0, 0)[0] == NOT_SUPPORTED
          and field(91)[0] == NOT_SUPPORTED and field(38)[0] == NOT_SUPPORTED)

# ---- accounting -----------------------------------------------------------
mode = ctypes.c_int(-1)
check("accounting is off by default",
      lib.nvmlDeviceGetAccountingMode(h, ctypes.byref(mode)) == SUCCESS and mode.value == 0)
n = ctypes.c_uint(0)
check("with it off, pids and stats are NOT_SUPPORTED",
      lib.nvmlDeviceGetAccountingPids(h, ctypes.byref(n), None) == NOT_SUPPORTED
      and lib.nvmlDeviceGetAccountingStats(h, 1, ctypes.byref(AccountingStats())) == NOT_SUPPORTED)
check("turn it on", lib.nvmlDeviceSetAccountingMode(h, 1) == SUCCESS
      and lib.nvmlDeviceGetAccountingMode(h, ctypes.byref(mode)) == SUCCESS and mode.value == 1)
pids = (ctypes.c_uint * 8)()
n = ctypes.c_uint(8)
check("no process has used the card: no pids, and a pid asked for is NOT_FOUND",
      lib.nvmlDeviceGetAccountingPids(h, ctypes.byref(n), pids) == SUCCESS and n.value == 0
      and lib.nvmlDeviceGetAccountingStats(h, 1, ctypes.byref(AccountingStats())) == NOT_FOUND)
size = ctypes.c_uint(0)
check("the accounting buffer holds 4000 entries",
      lib.nvmlDeviceGetAccountingBufferSize(h, ctypes.byref(size)) == SUCCESS and size.value == 4000)
check("a mode that is neither is INVALID_ARGUMENT", lib.nvmlDeviceSetAccountingMode(h, 5) == INVALID_ARGUMENT)
lib.nvmlDeviceSetAccountingMode(h, 0)

# ---- process queries ------------------------------------------------------
n = ctypes.c_uint(0)
check("no compute, graphics or MPS processes",
      lib.nvmlDeviceGetComputeRunningProcesses_v3(h, ctypes.byref(n), None) == SUCCESS and n.value == 0
      and lib.nvmlDeviceGetGraphicsRunningProcesses_v3(h, ctypes.byref(n), None) == SUCCESS and n.value == 0
      and lib.nvmlDeviceGetMPSComputeRunningProcesses_v3(h, ctypes.byref(n), None) == SUCCESS and n.value == 0)

print(f"{fails} failed")
sys.exit(1 if fails else 0)
