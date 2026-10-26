"""Shared helpers for the round-12 host-GPU measurements (runs on the g5.4xlarge rig HOST, not in the container).

Three ways to start the SAME program binary:
  sim     LD_LIBRARY_PATH=<build>/shim, VGPU_QUIET=1, VGPU_GPU=<profile>   (the simulator's shim libcudart/libcuda)
  native  LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64                       (the host's real libcudart and the A10G)
  ptxjit  native + CUDA_FORCE_PTX_JIT=1 CUDA_CACHE_DISABLE=1               (driver ignores embedded SASS, JITs the PTX)
"""
import os, subprocess, tempfile, time, resource, signal

HOME = os.path.expanduser("~")
BUILD = os.environ.get("HG_BUILD", HOME + "/hg/build")
HOSTCUDA = "/usr/local/cuda-13.2/lib64"


def env_for(mode, profile="nvidia/a10g", extra=None, threads=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_") and k not in ("LD_LIBRARY_PATH", "CUDA_FORCE_PTX_JIT", "CUDA_CACHE_DISABLE")}
    if mode.startswith("sim"):
        e["LD_LIBRARY_PATH"] = BUILD + "/shim"
        e["VGPU_QUIET"] = "1"
        e["VGPU_GPU"] = profile
        if threads:
            e["VGPU_THREADS"] = str(threads)
    else:
        e["LD_LIBRARY_PATH"] = HOSTCUDA
        if mode == "ptxjit":
            e["CUDA_FORCE_PTX_JIT"] = "1"
            e["CUDA_CACHE_DISABLE"] = "1"
    if extra:
        e.update(extra)
    return e


def run(argv, env, cwd=None, timeout=600):
    """Run once; returns dict(rc, timeout, wall, cpu, stdout, stderr)."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        t0 = time.monotonic()
        p = subprocess.Popen(argv, env=env, cwd=cwd, stdout=out, stderr=err, stdin=subprocess.DEVNULL, start_new_session=True)
        to = False
        try:
            _, status, ru = os.wait4(p.pid, 0) if timeout is None else _wait(p, timeout)
        except TimeoutError:
            to = True
            os.killpg(p.pid, signal.SIGKILL)
            _, status, ru = os.wait4(p.pid, 0)
        wall = time.monotonic() - t0
        out.seek(0); err.seek(0)
        return dict(rc=os.waitstatus_to_exitcode(status), timeout=to, wall=wall, cpu=ru.ru_utime + ru.ru_stime,
                    stdout=out.read().decode("utf-8", "replace"), stderr=err.read().decode("utf-8", "replace"))


def _wait(p, timeout):
    end = time.monotonic() + timeout
    while True:
        pid, status, ru = os.wait4(p.pid, os.WNOHANG)
        if pid:
            return pid, status, ru
        if time.monotonic() > end:
            raise TimeoutError
        time.sleep(0.005)
