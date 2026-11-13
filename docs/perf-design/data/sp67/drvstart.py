#!/usr/bin/env python3
"""Round sp67 (#570 item 2): the driver start the host-GPU mode makes (src/runtime/host_gpu.cpp begin_start), timed call by call in a fresh process:
dlopen of libcuda, cuInit, cuDeviceGet, the attributes, cuDevicePrimaryCtxRetain + cuCtxSetCurrent, and the 1 TiB reservation at the simulator's device-0
window (0x2000'0000'0000). Prints one line per step and the total."""
import ctypes, sys, time
t = [time.monotonic()]
lib = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
t.append(time.monotonic())
def call(name, *a):
    f = getattr(lib, name)
    r = f(*a)
    t.append(time.monotonic())
    if r: print(name, "returned", r); sys.exit(1)
call("cuInit", 0)
dev = ctypes.c_int()
call("cuDeviceGet", ctypes.byref(dev), 0)
v = ctypes.c_int()
for attr in (16, 75, 76):
    lib.cuDeviceGetAttribute(ctypes.byref(v), attr, dev)
t.append(time.monotonic())
ctx = ctypes.c_void_p()
call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev)
call("cuCtxSetCurrent", ctx)
p = ctypes.c_ulonglong()
lib.cuMemAddressReserve.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulonglong, ctypes.c_ulonglong]
call("cuMemAddressReserve", ctypes.byref(p), 1 << 40, 0, 0x200000000000, 0)
names = ["dlopen", "cuInit", "cuDeviceGet", "attributes", "cuDevicePrimaryCtxRetain", "cuCtxSetCurrent", "cuMemAddressReserve(1 TiB)"]
for n, a, b in zip(names, t, t[1:]):
    print(f"{n:28s} {1e3*(b-a):8.2f} ms")
print(f"{'total':28s} {1e3*(t[-1]-t[0]):8.2f} ms  (reserved at 0x{p.value:x})")
