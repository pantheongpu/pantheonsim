"""Helpers shared by the nvml_<category>.py checks (ctypes against the shim).

Each check file is run as `python3 nvml_<category>.py <shim dir> <profile id>` inside
`vgpu shell --gpu <profile> --count 2` with a private VGPU_STATE_DIR (see
run_nvml_category.sh). The hosted CI builds the shim with CUDA 12.0's nvml.h, which
lacks the newer entry points, so a symbol the shim does not export skips its checks.
"""
import ctypes, os, sys

SUCCESS, UNINITIALIZED, INVALID_ARGUMENT, NOT_SUPPORTED, NO_PERMISSION = 0, 1, 2, 3, 4
NOT_FOUND, INSUFFICIENT_SIZE, TIMEOUT, IN_USE, VERSION_MISMATCH, UNKNOWN = 6, 7, 10, 19, 25, 999

fails = 0
skipped = 0


def check(name, ok, got=""):
    global fails
    print(("ok    " if ok else "FAIL  ") + name + ("" if ok else f"  -> {got}"))
    fails += 0 if ok else 1


def finish():
    print(f"{fails} failed, {skipped} skipped")
    sys.exit(1 if fails else 0)


def version(struct_type, v):
    """NVML_STRUCT_VERSION: the struct's size with the version in the top byte."""
    return ctypes.sizeof(struct_type) | (v << 24)


class Lib:
    def __init__(self, shim_dir):
        self.lib = ctypes.CDLL(os.path.join(shim_dir, "libnvidia-ml.so.1"))
        self.lib.nvmlErrorString.restype = ctypes.c_char_p

    def has(self, name):
        return hasattr(self.lib, name)

    def __call__(self, name, *args):
        """The return code of nvml<name>, or None when the shim does not export it."""
        global skipped
        if not self.has(name):
            skipped += 1
            print(f"skip  {name} (not in this build's header)")
            return None
        return getattr(self.lib, name)(*args)

    def handle(self, index):
        h = ctypes.c_void_p()
        rc = self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(h))
        assert rc == SUCCESS, rc
        return h


def u(v=0):
    return ctypes.c_uint(v)


def ull(v=0):
    return ctypes.c_ulonglong(v)


def ref(x):
    return ctypes.byref(x)


BAD = ctypes.c_void_p(0xdeadbeef)    # a handle that is no device


def run_vgpu(shim_dir, *args):
    """Runs `vgpu <args>` in the session this test runs in; returns (exit code, output)."""
    import subprocess
    vgpu = os.path.join(shim_dir, "..", "vgpu")
    r = subprocess.run([vgpu, *args], capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout + r.stderr
