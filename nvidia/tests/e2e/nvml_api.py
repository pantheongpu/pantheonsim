"""NVML as monitoring tools call it, through ctypes against the shim.

Run inside `vgpu shell` on 2 x T4, which publishes the telemetry NVML reads.
Each check is something a real tool relies on and the shim once got wrong:
queries after nvmlShutdown were answered; the CUDA version was 13.0 whatever
the session said; lookups by UUID and bus id, and the MIG and encoder queries,
were missing symbols -- which pynvml turns into FunctionNotFound, and nvitop
exits on.
"""
import ctypes, os, sys

SUCCESS, UNINITIALIZED, INVALID_ARGUMENT, NOT_SUPPORTED, NOT_FOUND = 0, 1, 2, 3, 6
INSUFFICIENT_SIZE, DRIVER_NOT_LOADED = 7, 9
lib = ctypes.CDLL(os.path.join(sys.argv[1], "libnvidia-ml.so.1"))
lib.nvmlErrorString.restype = ctypes.c_char_p
fails = 0


def check(name, ok, got=""):
    global fails
    print(("ok    " if ok else "FAIL  ") + name + ("" if ok else f"  -> {got}"))
    fails += 0 if ok else 1


class Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


# `--described`: nothing is publishing telemetry, as for a program `vgpu run`
# starts before it has touched CUDA. The machine VGPU_GPU / VGPU_DEVICE_COUNT
# describe must still answer, idle. It used to be DRIVER_NOT_LOADED.
# `--undescribed`: no telemetry and no description, which is a machine with no
# driver.
if len(sys.argv) > 2 and sys.argv[2] == "--undescribed":
    rc = lib.nvmlInit_v2()
    check("no telemetry and no machine described: DRIVER_NOT_LOADED", rc == DRIVER_NOT_LOADED, rc)
    sys.exit(1 if fails else 0)
if len(sys.argv) > 2 and sys.argv[2] == "--described":
    rc = lib.nvmlInit_v2()
    check("no telemetry, machine described: init succeeds", rc == SUCCESS, rc)
    n = ctypes.c_uint()
    lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
    want = int(os.environ.get("VGPU_DEVICE_COUNT", "1"))
    check("device count is the described one", n.value == want, (n.value, want))
    h = ctypes.c_void_p()
    lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(n.value - 1), ctypes.byref(h))
    name = ctypes.create_string_buffer(96)
    rc = lib.nvmlDeviceGetName(h, name, ctypes.c_uint(96))
    check("name comes from the profile", rc == SUCCESS and b"T4" in name.value, (rc, name.value))
    mem = Memory()
    rc = lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem))
    check("memory is total and unused", rc == SUCCESS and mem.total > 0 and mem.used == 0, (rc, mem.total, mem.used))
    check("shutdown", lib.nvmlShutdown() == SUCCESS)
    print(f"{fails} failed")
    sys.exit(1 if fails else 0)


def handle(i):
    h = ctypes.c_void_p()
    rc = lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i), ctypes.byref(h))
    return rc, h.value


def string(fn, h, n=96):
    buf = ctypes.create_string_buffer(n)
    rc = fn(ctypes.c_void_p(h), buf, ctypes.c_uint(n))
    return rc, buf.value.decode()


check("nvmlInit_v2", lib.nvmlInit_v2() == SUCCESS)
count = ctypes.c_uint()
lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
check("two devices", count.value == 2, count.value)

rc, h1 = handle(1)
rc, uuid1 = string(lib.nvmlDeviceGetUUID, h1)
found = ctypes.c_void_p()
rc = lib.nvmlDeviceGetHandleByUUID(uuid1.encode(), ctypes.byref(found))
check("handle by UUID finds the same device", rc == SUCCESS and found.value == h1, (rc, found.value, h1))
rc = lib.nvmlDeviceGetHandleByUUID(b"GPU-00000000-0000-0000-0000-000000000000", ctypes.byref(found))
check("an unknown UUID is NOT_FOUND", rc == NOT_FOUND, rc)


class Pci(ctypes.Structure):
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


pci = Pci()
lib.nvmlDeviceGetPciInfo_v3(ctypes.c_void_p(h1), ctypes.byref(pci))
rc = lib.nvmlDeviceGetHandleByPciBusId_v2(pci.busId, ctypes.byref(found))
check("handle by PCI bus id finds the same device", rc == SUCCESS and found.value == h1, (rc, pci.busId))

minor = ctypes.c_uint(99)
check("minor number is the index", lib.nvmlDeviceGetMinorNumber(ctypes.c_void_p(h1), ctypes.byref(minor)) == SUCCESS
      and minor.value == 1, minor.value)
is_mig = ctypes.c_uint(99)
check("not a MIG handle", lib.nvmlDeviceIsMigDeviceHandle(ctypes.c_void_p(h1), ctypes.byref(is_mig)) == SUCCESS
      and is_mig.value == 0, is_mig.value)
util, period = ctypes.c_uint(), ctypes.c_uint()
check("encoder utilization exists and says NOT_SUPPORTED",
      lib.nvmlDeviceGetEncoderUtilization(ctypes.c_void_p(h1), ctypes.byref(util), ctypes.byref(period)) == NOT_SUPPORTED)
reasons = ctypes.c_ulonglong(99)
check("an idle card reports only the idle reason, as a real one does",
      lib.nvmlDeviceGetCurrentClocksThrottleReasons(ctypes.c_void_p(h1), ctypes.byref(reasons))
      == SUCCESS and reasons.value == 1, reasons.value)

# Reliability and link, from the T4 profile: ECC on, the Gen3 x8 link clouds
# attach it at, PCIe error counters answered (a real RTX 3060 answers them
# too), no memory sensor.
cur, pend = ctypes.c_int(-1), ctypes.c_int(-1)
rc = lib.nvmlDeviceGetEccMode(ctypes.c_void_p(h1), ctypes.byref(cur), ctypes.byref(pend))
check("ECC is on", rc == SUCCESS and (cur.value, pend.value) == (1, 1), (rc, cur.value, pend.value))
errors = ctypes.c_ulonglong(99)
rc = lib.nvmlDeviceGetTotalEccErrors(ctypes.c_void_p(h1), 1, 0, ctypes.byref(errors))
check("no ECC errors", rc == SUCCESS and errors.value == 0, (rc, errors.value))
gen, width = ctypes.c_uint(), ctypes.c_uint()
rc_g = lib.nvmlDeviceGetCurrPcieLinkGeneration(ctypes.c_void_p(h1), ctypes.byref(gen))
rc_w = lib.nvmlDeviceGetCurrPcieLinkWidth(ctypes.c_void_p(h1), ctypes.byref(width))
check("PCIe link is Gen3 x8", (rc_g, rc_w, gen.value, width.value) == (SUCCESS, SUCCESS, 3, 8),
      (rc_g, rc_w, gen.value, width.value))


class FieldValue(ctypes.Structure):
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_int), ("nvmlReturn", ctypes.c_int),
                ("value", ctypes.c_ulonglong)]


fields = (FieldValue * 3)()
fields[0].fieldId, fields[1].fieldId, fields[2].fieldId = 94, 180, 82  # replay, fatal, memory temp
rc = lib.nvmlDeviceGetFieldValues(ctypes.c_void_p(h1), 3, fields)
check("PCIe replay and fatal-error counters answer zero",
      rc == SUCCESS and [(f.nvmlReturn, f.value) for f in fields[:2]] == [(SUCCESS, 0), (SUCCESS, 0)],
      (rc, [(f.nvmlReturn, f.value) for f in fields[:2]]))
check("no memory temperature on a T4", fields[2].nvmlReturn == NOT_SUPPORTED, fields[2].nvmlReturn)

cuda = ctypes.c_int()
lib.nvmlSystemGetCudaDriverVersion_v2(ctypes.byref(cuda))
major, minor_v = (os.environ.get("VGPU_CUDA_VERSION", "13.0").split(".") + ["0"])[:2]
want = int(major) * 1000 + int(minor_v) * 10
check("CUDA driver version is the session's", cuda.value == want, (cuda.value, want))

# A buffer too small is INSUFFICIENT_SIZE. It used to be a truncated string and
# SUCCESS, which a caller cannot tell from the real answer.
rc, short = string(lib.nvmlDeviceGetUUID, h1, 8)
check("a UUID buffer too small is INSUFFICIENT_SIZE", rc == INSUFFICIENT_SIZE, (rc, short))
rc, short = string(lib.nvmlDeviceGetName, h1, 4)
check("a name buffer too small is INSUFFICIENT_SIZE", rc == INSUFFICIENT_SIZE, (rc, short))
rc, full = string(lib.nvmlDeviceGetUUID, h1, len(uuid1) + 1)
check("a buffer of exactly the right size succeeds", rc == SUCCESS and full == uuid1, (rc, full))

# The mode queries check the handle. They used to answer for any handle at all.
mode = ctypes.c_int(-1)
check("persistence mode of a bogus handle is INVALID_ARGUMENT",
      lib.nvmlDeviceGetPersistenceMode(ctypes.c_void_p(0xdeadbeef), ctypes.byref(mode)) == INVALID_ARGUMENT)
check("compute mode of a bogus handle is INVALID_ARGUMENT",
      lib.nvmlDeviceGetComputeMode(ctypes.c_void_p(0xdeadbeef), ctypes.byref(mode)) == INVALID_ARGUMENT)
check("display mode of a NULL handle is INVALID_ARGUMENT",
      lib.nvmlDeviceGetDisplayMode(ctypes.c_void_p(0), ctypes.byref(mode)) == INVALID_ARGUMENT)
check("persistence mode of a real handle still answers",
      lib.nvmlDeviceGetPersistenceMode(ctypes.c_void_p(h1), ctypes.byref(mode)) == SUCCESS)

# Every declared code has a string; these read "Unknown Error".
check("error strings cover the header",
      all(lib.nvmlErrorString(c) != b"Unknown Error" for c in (4, 5, 8, 10, 11, 12, 14, 15, 16, 17)),
      [lib.nvmlErrorString(c) for c in (4, 15)])

# Reference counting: two inits need two shutdowns.
check("a second init", lib.nvmlInit_v2() == SUCCESS)
check("first shutdown", lib.nvmlShutdown() == SUCCESS)
check("still initialized after one of two shutdowns", lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) == SUCCESS)
check("second shutdown", lib.nvmlShutdown() == SUCCESS)
check("queries after the last shutdown are UNINITIALIZED",
      lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) == UNINITIALIZED)
check("a shutdown too many is UNINITIALIZED", lib.nvmlShutdown() == UNINITIALIZED)

print(f"{fails} failed")
sys.exit(1 if fails else 0)
