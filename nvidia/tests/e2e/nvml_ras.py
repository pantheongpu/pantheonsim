"""NVML ECC detail, page retirement, row remapping, accounting, engines and samples, against the shim.

Run by run_nvml_category.sh once per profile (t4: pages retire; h100: rows remap; rtx3060: no ECC).
Errors are injected with `vgpu fault`, so the counts, the retired pages and the remapped rows
come from the same reliability state nvidia-smi reads.
"""
import ctypes, os, subprocess, sys, time
from nvml_support import *

shim = sys.argv[1]
lib = Lib(shim)
profile = sys.argv[2]
ECC = profile in ("nvidia/t4", "nvidia/h100")
PAGES = profile == "nvidia/t4"
ROWS = profile == "nvidia/h100"
check("nvmlInit_v2", lib.lib.nvmlInit_v2() == SUCCESS)
h1 = lib.handle(1)


def val(name, h, ctype=ctypes.c_uint, *extra):
    x = ctype()
    rc = lib(name, h, *extra, ref(x))
    return rc, x.value


# ---- ECC ------------------------------------------------------------------------------------------------------------------
rc, v = val("nvmlDeviceGetDefaultEccMode", h1, ctypes.c_int)
check("default ECC mode: on where the card has ECC, N/A where not", (rc, v) == ((SUCCESS, 1) if ECC else (NOT_SUPPORTED, 0)), (rc, v))


class EccCounts(ctypes.Structure):
    _fields_ = [("l1", ctypes.c_ulonglong), ("l2", ctypes.c_ulonglong), ("dev", ctypes.c_ulonglong), ("reg", ctypes.c_ulonglong)]


ec = EccCounts()
rc = lib("nvmlDeviceGetDetailedEccErrors", h1, ctypes.c_int(0), ctypes.c_int(1), ref(ec))
check("detailed ECC errors: zero until something is injected", rc == (SUCCESS if ECC else NOT_SUPPORTED) and ec.l2 == 0, (rc, ec.l2))
check("detailed ECC errors, a bad error type", lib("nvmlDeviceGetDetailedEccErrors", h1, ctypes.c_int(5), ctypes.c_int(1), ref(ec)) == INVALID_ARGUMENT)


class SramStatus(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("aggUncParity", ctypes.c_ulonglong), ("aggUncSecDed", ctypes.c_ulonglong),
                ("aggCor", ctypes.c_ulonglong), ("volUncParity", ctypes.c_ulonglong), ("volUncSecDed", ctypes.c_ulonglong),
                ("volCor", ctypes.c_ulonglong), ("b1", ctypes.c_ulonglong), ("b2", ctypes.c_ulonglong),
                ("b3", ctypes.c_ulonglong), ("b4", ctypes.c_ulonglong), ("b5", ctypes.c_ulonglong),
                ("thresholdExceeded", ctypes.c_uint)]


ss = SramStatus()
ss.version = version(SramStatus, 1)
rc = lib("nvmlDeviceGetSramEccErrorStatus", h1, ref(ss))
if rc is not None:
    check("SRAM ECC status: Ampere and newer", rc == (NOT_SUPPORTED if profile == "nvidia/t4" else SUCCESS), rc)
    ss.version = 3
    check("SRAM ECC status with a wrong version", lib("nvmlDeviceGetSramEccErrorStatus", h1, ref(ss)) == VERSION_MISMATCH)


def inject(*args):
    code, out = run_vgpu(shim, "fault", "inject", "--gpu", "1", *args)
    check("vgpu fault inject " + " ".join(args), code == 0, out)


if ECC:
    inject("--ecc", "corrected", "--location", "l2_cache", "--count", "2")
    ec = EccCounts()
    rc = lib("nvmlDeviceGetDetailedEccErrors", h1, ctypes.c_int(0), ctypes.c_int(1), ref(ec))
    check("an injected corrected L2 error is counted, aggregate", rc == SUCCESS and ec.l2 == 2 and ec.dev == 0, (rc, ec.l2, ec.dev))
    rc = lib("nvmlDeviceGetDetailedEccErrors", h1, ctypes.c_int(0), ctypes.c_int(0), ref(ec))
    check("and volatile", rc == SUCCESS and ec.l2 == 2, (rc, ec.l2))
    if profile == "nvidia/h100":
        inject("--ecc", "corrected", "--location", "sram", "--count", "3")
        ss = SramStatus()
        ss.version = version(SramStatus, 1)
        rc = lib("nvmlDeviceGetSramEccErrorStatus", h1, ref(ss))
        if rc is not None:
            check("SRAM ECC status counts the injected SRAM errors", rc == SUCCESS and ss.aggCor == 3 and ss.volCor == 3, (rc, ss.aggCor, ss.volCor))

# ---- retired pages (GDDR with ECC) and remapped rows (HBM) -------------------------------------------------------------------------
n = u(0)
a = (ctypes.c_ulonglong * 8)()
t = (ctypes.c_ulonglong * 8)()
rc = lib("nvmlDeviceGetRetiredPages", h1, ctypes.c_int(1), ref(n), a)
if PAGES:
    check("no retired pages yet", (rc, n.value) == (SUCCESS, 0), (rc, n.value))
    pend = ctypes.c_int(9)
    check("none pending", lib("nvmlDeviceGetRetiredPagesPendingStatus", h1, ref(pend)) == SUCCESS and pend.value == 0)
    inject("--ecc", "uncorrected", "--location", "dram")
    n = u(0)
    check("the sizing call: INSUFFICIENT_SIZE with the count", lib("nvmlDeviceGetRetiredPages", h1, ctypes.c_int(1), ref(n), a) == INSUFFICIENT_SIZE and n.value == 1)
    n = u(8)
    rc = lib("nvmlDeviceGetRetiredPages_v2", h1, ctypes.c_int(1), ref(n), a, t)
    check("an uncorrectable device-memory error retires a page", rc == SUCCESS and n.value == 1, (rc, n.value))
    first = a[0]
    check("the page address is 4 KiB aligned and inside memory", first % 4096 == 0 and first < 16 * 1024 ** 3, hex(first))
    check("retirement has a time", t[0] > 1_000_000_000_000_000, t[0])
    n = u(8)
    lib("nvmlDeviceGetRetiredPages", h1, ctypes.c_int(1), ref(n), a)
    check("the same page every time it is asked", a[0] == first, (hex(a[0]), hex(first)))
    n = u(8)
    check("none retired for single-bit errors", lib("nvmlDeviceGetRetiredPages", h1, ctypes.c_int(0), ref(n), a) == SUCCESS and n.value == 0)
    inject("--ecc", "uncorrected", "--location", "dram", "--count", "2")
    n = u(8)
    lib("nvmlDeviceGetRetiredPages", h1, ctypes.c_int(1), ref(n), a)
    check("three pages now, the first still the first", n.value == 3 and a[0] == first and len(set(list(a)[:3])) == 3, (n.value, [hex(x) for x in list(a)[:3]]))
    n = u(2)
    check("a buffer for 2 of 3", lib("nvmlDeviceGetRetiredPages", h1, ctypes.c_int(1), ref(n), a) == INSUFFICIENT_SIZE and n.value == 3)
    check("a bad cause", lib("nvmlDeviceGetRetiredPages", h1, ctypes.c_int(2), ref(u(8)), a) == INVALID_ARGUMENT)
    check("a pending retirement", lib("nvmlDeviceGetRetiredPagesPendingStatus", h1, ref(pend)) == SUCCESS and pend.value == 1)
else:
    check("page retirement: N/A (HBM remaps rows, a GeForce has no ECC)", rc == NOT_SUPPORTED, rc)

c, un, pend, fail = u(), u(), u(), u()
rc = lib("nvmlDeviceGetRemappedRows", h1, ref(c), ref(un), ref(pend), ref(fail))
if ROWS:
    check("no remapped rows yet", (rc, c.value, un.value, pend.value, fail.value) == (SUCCESS, 0, 0, 0, 0), (rc, c.value, un.value))
    inject("--ecc", "uncorrected", "--location", "dram", "--count", "2")
    rc = lib("nvmlDeviceGetRemappedRows", h1, ref(c), ref(un), ref(pend), ref(fail))
    check("an uncorrectable error remaps a row, pending", (rc, c.value, un.value, pend.value, fail.value) == (SUCCESS, 0, 2, 1, 0), (rc, c.value, un.value, pend.value, fail.value))
    st = ctypes.create_string_buffer(64)
    check("the remapper histogram is not modelled", lib("nvmlDeviceGetRowRemapperHistogram", h1, st) == NOT_SUPPORTED)
else:
    check("row remapping: N/A on GDDR", rc == NOT_SUPPORTED, rc)
check("remapped rows with a NULL pointer", lib("nvmlDeviceGetRemappedRows", h1, ref(c), None, ref(pend), ref(fail)) == INVALID_ARGUMENT)


class Repair(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("channel", ctypes.c_uint), ("tpc", ctypes.c_uint)]


rs = Repair(version(Repair, 1), 9, 9)
rc = lib("nvmlDeviceGetRepairStatus", h1, ref(rs))
if rc is not None:
    check("repair status: nothing pending (Ampere and newer)", (rc, rs.channel, rs.tpc) == ((NOT_SUPPORTED, 9, 9) if profile == "nvidia/t4" else (SUCCESS, 0, 0)), (rc, rs.channel, rs.tpc))
    rs.version = 2
    check("repair status with a wrong version", lib("nvmlDeviceGetRepairStatus", h1, ref(rs)) == VERSION_MISMATCH)

# ---- processes and accounting -----------------------------------------------------------------------------------------------------------
class Proc(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong)]


np_ = u(0)
check("compute processes (v1 layout): none", lib("nvmlDeviceGetComputeRunningProcesses", h1, ref(np_), None) == SUCCESS and np_.value == 0)
check("graphics processes: none", lib("nvmlDeviceGetGraphicsRunningProcesses", h1, ref(np_), None) == SUCCESS and np_.value == 0)
check("MPS processes: none", lib("nvmlDeviceGetMPSComputeRunningProcesses", h1, ref(np_), None) == SUCCESS and np_.value == 0)
check("accounting buffer: 4000 records", val("nvmlDeviceGetAccountingBufferSize", h1) == (SUCCESS, 4000))
check("accounting is off", val("nvmlDeviceGetAccountingMode", h1, ctypes.c_int) == (SUCCESS, 0))
pids = (ctypes.c_uint * 16)()
n = u(16)
check("with accounting off: an empty list", lib("nvmlDeviceGetAccountingPids", h1, ref(n), pids) == SUCCESS and n.value == 0)
n = u(0)
check("the sizing call is INSUFFICIENT_SIZE", lib("nvmlDeviceGetAccountingPids", h1, ref(n), None) == INSUFFICIENT_SIZE)


class Acct(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("mem", ctypes.c_uint), ("maxMem", ctypes.c_ulonglong), ("time", ctypes.c_ulonglong),
                ("start", ctypes.c_ulonglong), ("running", ctypes.c_uint), ("reserved", ctypes.c_uint * 5)]


st = Acct()
check("with accounting off no pid is found", lib("nvmlDeviceGetAccountingStats", h1, u(1), ref(st)) == NOT_FOUND)
check("enable accounting (the root override)", lib("nvmlDeviceSetAccountingMode", h1, ctypes.c_int(1)) == SUCCESS)
check("accounting is on", val("nvmlDeviceGetAccountingMode", h1, ctypes.c_int) == (SUCCESS, 1))

cudart = None
try:
    cudart = ctypes.CDLL(os.path.join(shim, "libcudart.so"))
except OSError:
    print("skip  no CUDA runtime shim: the process-dependent accounting checks")
if cudart is not None:
    child = subprocess.Popen([sys.executable, "-c",
        "import ctypes,sys,time\n"
        "c=ctypes.CDLL(sys.argv[1]); c.cudaSetDevice(1)\n"
        "p=ctypes.c_void_p(); r=c.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(64<<20))\n"
        "print('ready', r, flush=True); time.sleep(120)\n", os.path.join(shim, "libcudart.so")],
        stdout=subprocess.PIPE, text=True)
    ready = []
    for _ in range(20):                      # past anything the runtime prints first
        line = child.stdout.readline().split()
        if not line or line[0] == "ready":
            ready = line
            break
    check("a program holds 64 MiB on GPU 1", ready[:2] == ["ready", "0"], ready)
    try:
        procs = (Proc * 8)()
        n = u(8)
        rc = lib("nvmlDeviceGetComputeRunningProcesses", h1, ref(n), procs)
        listed = {procs[i].pid: procs[i].used for i in range(n.value)} if rc == SUCCESS else {}
        check("it is listed as a compute process with its memory", child.pid in listed and listed[child.pid] >= 64 << 20, (rc, listed))
        n = u(16)
        rc = lib("nvmlDeviceGetAccountingPids", h1, ref(n), pids)
        check("accounting lists it", rc == SUCCESS and child.pid in list(pids)[:n.value], (rc, list(pids)[:n.value]))
        rc = lib("nvmlDeviceGetAccountingStats", h1, u(child.pid), ref(st))
        check("its stats: running, 64 MiB at least, utilization not attributed",
              rc == SUCCESS and st.running == 1 and st.maxMem >= 64 << 20 and st.gpu == 0xFFFFFFFF and st.time == 0 and st.start > 0,
              (rc, st.running, st.maxMem, st.gpu, st.time))
        child.kill()
        child.wait()
        deadline = time.time() + 15
        while time.time() < deadline:
            rc = lib("nvmlDeviceGetAccountingStats", h1, u(child.pid), ref(st))
            if rc == SUCCESS and st.running == 0:
                break
            time.sleep(0.3)
        check("after it exits: finished, its memory high-water mark kept", rc == SUCCESS and st.running == 0 and st.maxMem >= 64 << 20, (rc, st.running, st.maxMem))
        check("clearing drops the finished process", lib("nvmlDeviceClearAccountingPids", h1) == SUCCESS)
        check("and it is gone", lib("nvmlDeviceGetAccountingStats", h1, u(child.pid), ref(st)) == NOT_FOUND)
    finally:
        child.kill()
check("disable accounting", lib("nvmlDeviceSetAccountingMode", h1, ctypes.c_int(0)) == SUCCESS)
n = u(16)
check("disabling empties the list", lib("nvmlDeviceGetAccountingPids", h1, ref(n), pids) == SUCCESS and n.value == 0)


class PDetail(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("mode", ctypes.c_uint), ("n", ctypes.c_uint), ("arr", ctypes.c_void_p)]


pd = PDetail(version(PDetail, 1), 0, 0, None)
rc = lib("nvmlDeviceGetRunningProcessDetailList", h1, ref(pd))
if rc is not None:
    check("process details: Hopper and newer", rc == (SUCCESS if profile == "nvidia/h100" else NOT_SUPPORTED), rc)
    pd.version = 4
    check("process details with a wrong version", lib("nvmlDeviceGetRunningProcessDetailList", h1, ref(pd)) == VERSION_MISMATCH)


class PUtil(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("count", ctypes.c_uint), ("last", ctypes.c_ulonglong), ("arr", ctypes.c_void_p)]


pu = PUtil(version(PUtil, 1), 0, 0, None)
rc = lib("nvmlDeviceGetProcessesUtilizationInfo", h1, ref(pu))
if rc is not None:
    check("per-process utilization: the sizing call", rc == INSUFFICIENT_SIZE, rc)

# ---- engines ------------------------------------------------------------------------------------------------------------------------------------
NVENC = profile in ("nvidia/rtx3060", "nvidia/t4")
a_, b_ = u(), u()
rc = lib("nvmlDeviceGetEncoderUtilization", h1, ref(a_), ref(b_))
check("encoder utilization: 0 over 200 ms where there is an encoder", (rc, a_.value, b_.value) == ((SUCCESS, 0, 200000) if NVENC else (NOT_SUPPORTED, 0, 0)), (rc, a_.value, b_.value))
check("decoder utilization", lib("nvmlDeviceGetDecoderUtilization", h1, ref(a_), ref(b_)) == SUCCESS)
rc = lib("nvmlDeviceGetJpgUtilization", h1, ref(a_), ref(b_))
if rc is not None:
    check("JPEG engine: Ampere and newer", rc == (NOT_SUPPORTED if profile == "nvidia/t4" else SUCCESS), rc)
rc = lib("nvmlDeviceGetOfaUtilization", h1, ref(a_), ref(b_))
if rc is not None:
    check("optical flow accelerator: Turing and newer", rc == SUCCESS, rc)
rc, v = val("nvmlDeviceGetEncoderCapacity", h1, ctypes.c_uint, ctypes.c_int(0))
check("encoder capacity, H.264", (rc, v) == ((SUCCESS, 100) if NVENC else (NOT_SUPPORTED, 0)), (rc, v))
check("encoder capacity, AV1 (Ada and newer)", val("nvmlDeviceGetEncoderCapacity", h1, ctypes.c_uint, ctypes.c_int(2))[0] == NOT_SUPPORTED)
check("encoder capacity, a codec that does not exist", val("nvmlDeviceGetEncoderCapacity", h1, ctypes.c_uint, ctypes.c_int(3))[0] == INVALID_ARGUMENT)
rc = lib("nvmlDeviceGetEncoderStats", h1, ref(a_), ref(b_), ref(u()))
check("encoder statistics", rc == (SUCCESS if NVENC else NOT_SUPPORTED), rc)
n = u(5)
check("encoder sessions: none", lib("nvmlDeviceGetEncoderSessions", h1, ref(n), None) == (SUCCESS if NVENC else NOT_SUPPORTED))


class Fbc(ctypes.Structure):
    _fields_ = [("sessions", ctypes.c_uint), ("fps", ctypes.c_uint), ("latency", ctypes.c_uint)]


fb = Fbc()
check("frame buffer capture: GeForce only", lib("nvmlDeviceGetFBCStats", h1, ref(fb)) == (SUCCESS if profile == "nvidia/rtx3060" else NOT_SUPPORTED))

# ---- samples ------------------------------------------------------------------------------------------------------------------------------------
class Sample(ctypes.Structure):
    _fields_ = [("ts", ctypes.c_ulonglong), ("value", ctypes.c_ulonglong)]


vt = ctypes.c_int(-1)
cnt = u(0)
rc = lib("nvmlDeviceGetSamples", h1, ctypes.c_int(0), ull(0), ref(vt), ref(cnt), None)
check("samples: the sizing call finds one", rc == SUCCESS and cnt.value == 1 and vt.value == 1, (rc, cnt.value, vt.value))
smp = (Sample * 2)()
cnt = u(2)
rc = lib("nvmlDeviceGetSamples", h1, ctypes.c_int(5), ull(0), ref(vt), ref(cnt), smp)
check("processor clock sample", rc == SUCCESS and cnt.value == 1 and smp[0].ts > 0 and (smp[0].value & 0xFFFFFFFF) > 0, (rc, cnt.value, smp[0].value))
cnt = u(2)
check("samples after the last one: not found", lib("nvmlDeviceGetSamples", h1, ctypes.c_int(0), ull(2 ** 62), ref(vt), ref(cnt), smp) == NOT_FOUND)
cnt = u(0)
check("a buffer of size 0", lib("nvmlDeviceGetSamples", h1, ctypes.c_int(0), ull(0), ref(vt), ref(cnt), smp) == INVALID_ARGUMENT)
check("a sampling type that does not exist", lib("nvmlDeviceGetSamples", h1, ctypes.c_int(42), ull(0), ref(vt), ref(u(1)), smp) == INVALID_ARGUMENT)

# ---- handles ----------------------------------------------------------------------------------------------------------------------------------------
for name, args in [("nvmlDeviceGetDefaultEccMode", [ref(ctypes.c_int())]), ("nvmlDeviceGetAccountingMode", [ref(ctypes.c_int())]),
                   ("nvmlDeviceGetRetiredPagesPendingStatus", [ref(ctypes.c_int())]),
                   ("nvmlDeviceGetEncoderCapacity", [ctypes.c_int(0), ref(u())])]:
    rc = lib(name, BAD, *args)
    check(f"{name}: a bad handle is invalid", rc in (None, INVALID_ARGUMENT), rc)
check("shutdown", lib.lib.nvmlShutdown() == SUCCESS)
finish()
