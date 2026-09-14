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
lib = ctypes.CDLL(os.path.join(sys.argv[1], "libnvidia-ml.so.1"))
fails = 0


def check(name, ok, got=""):
    global fails
    print(("ok    " if ok else "FAIL  ") + name + ("" if ok else f"  -> {got}"))
    fails += 0 if ok else 1


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
check("nothing throttles", lib.nvmlDeviceGetCurrentClocksThrottleReasons(ctypes.c_void_p(h1), ctypes.byref(reasons))
      == SUCCESS and reasons.value == 0, reasons.value)

cuda = ctypes.c_int()
lib.nvmlSystemGetCudaDriverVersion_v2(ctypes.byref(cuda))
major, minor_v = (os.environ.get("VGPU_CUDA_VERSION", "13.0").split(".") + ["0"])[:2]
want = int(major) * 1000 + int(minor_v) * 10
check("CUDA driver version is the session's", cuda.value == want, (cuda.value, want))

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
