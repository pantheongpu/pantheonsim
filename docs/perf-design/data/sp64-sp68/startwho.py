#!/usr/bin/env python3
"""Which rule starts the driver: runs corpus programs with VGPU_HOST_GPU_LOG=1 against the stand-in driver (which refuses every module, so
every launch runs in the simulator) and prints, per program and mode, the "starts the driver" line, the wall time and the summary counts.

  startwho.py --build B --corp C --stub libstubcuda.so --programs a,b --modes exact,turbo [--env K=V] [--timeout S]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True)
ap.add_argument("--corp", required=True)
ap.add_argument("--stub", required=True)
ap.add_argument("--real", required=True, help="tools/bench/real directory")
ap.add_argument("--programs", required=True)
ap.add_argument("--modes", default="exact,turbo")
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--timeout", type=float, default=270)
ap.add_argument("--raw", default="")
ap.add_argument("--stop", action="store_true", help="end a run once the driver is started")
a = ap.parse_args()
sys.path.insert(0, a.real)
import real  # noqa: E402

want = a.programs.split(",")
progs = [p for p in real.manifest(a.corp, ("polybench", "rodinia", "hecbench", "samples")) if p[1] in want]
for corpus, name, cwd, argv, copy in progs:
    for mode in a.modes.split(","):
        env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
        env.update(VGPU_GPU="nvidia/a10g", VGPU_THREADS="3", LD_LIBRARY_PATH=os.path.join(a.build, "shim"), VGPU_HOST_GPU=mode,
                   VGPU_HOST_GPU_LOG="1", VGPU_HOST_GPU_LIBCUDA=a.stub, STUB_INIT_MS="150", STUB_SM="86", STUB_MP="72")
        env.update(dict(kv.split("=", 1) for kv in a.env))
        scratch = None
        if copy:
            scratch = tempfile.mkdtemp(prefix="sw-", dir=os.path.dirname(a.build))
            dd = os.path.join(os.path.dirname(argv[0]), "data")
            with open(os.path.join(scratch, "filelist"), "w") as f:
                f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
            cwd = scratch
        t0 = time.monotonic()
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        lines, rc = [], None
        import select
        buf = b""
        while True:
            left = a.timeout - (time.monotonic() - t0)
            if left <= 0:
                os.killpg(p.pid, 9); rc = "timeout"; break
            r, _, _ = select.select([p.stderr], [], [], min(left, 1.0))
            if r:
                chunk = os.read(p.stderr.fileno(), 65536)
                if not chunk:
                    break
                buf += chunk
                if a.stop and b"starts the driver" in buf:
                    os.killpg(p.pid, 9); rc = "stopped"; break
        p.wait()
        if rc is None:
            rc = p.returncode
        err = buf.decode("utf-8", "replace")
        wall = time.monotonic() - t0
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        if a.raw:
            with open(os.path.join(a.raw, f"{name}.{mode}.err"), "w") as f:
                f.write(err)
        starts = [l for l in err.splitlines() if "starts the driver" in l]
        summ = [l for l in err.splitlines() if l.startswith("host_gpu: ") and "launches ran on the card" in l]
        sampled = [l for l in err.splitlines() if l.startswith("host_gpu_sampled:")]
        print(f"{corpus}/{name} {mode} rc={rc} wall={wall:.2f}")
        print("   " + (starts[0] if starts else "driver never started"))
        for l in summ + sampled:
            print("   " + l[:240])
        sys.stdout.flush()
