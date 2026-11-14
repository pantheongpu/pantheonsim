#!/usr/bin/env python3
"""Round sp67 (#569): verify mode against the plain simulator, interleaved, per program. Each side is a (build, environment) pair; the runs of one
program alternate between the sides, each under the machine's build lock (taken before the clock starts) and niced. Kept per run: wall seconds, exit status, a hash of stdout
with its timing lines removed, and the mode's machine-readable summary line (host_gpu_verify: ...).

  verify_ab.py --corp <dir with polybench/bin> --out out.json --runs 5 --side plain=<build> --side verify=<build>:VGPU_HOST_GPU=exact;VGPU_HOST_GPU_VERIFY=1
               [--stub libstubcuda.so] [--progs 2DCONV,3DCONV] [--threads 4] [--gpu nvidia/a10g] [--timeout 300] [--lock /tmp/pantheonsim-builder.lock]

With --stub, every side that sets VGPU_HOST_GPU gets the stand-in driver (VGPU_HOST_GPU_LIBCUDA) in the mode of a card that keeps memory and computes
nothing (STUB_CARD_OK=1 STUB_KEEP_MEMORY=1): verify believes each launch ran on the card, copies every live byte both ways and compares; the program's
result is the simulator's. Without --stub the real driver is used (a card rig).
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import statistics
import subprocess
import time

TIMING = re.compile(r"(time|Time|TIME|seconds|elapsed)")
NUMBER = re.compile(r"^\s*[-+0-9.eE]+\s*$")


def run(build, env_extra, prog, args, gpu, threads, timeout, lock, stub):
    env = dict(os.environ)
    env.update({"VGPU_GPU": gpu, "VGPU_THREADS": str(threads), "LD_LIBRARY_PATH": os.path.join(build, "shim"), "VGPU_QUIET": "1"})
    env.update(env_extra)
    if stub and "VGPU_HOST_GPU" in env_extra:
        env.update({"VGPU_HOST_GPU_LIBCUDA": stub, "STUB_CARD_OK": "1", "STUB_KEEP_MEMORY": "1", "STUB_SM": "86", "STUB_MP": "80", "STUB_INIT_MS": "0"})
    cmd = ["nice", "-n", "19", "timeout", str(timeout), prog] + args
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)   # (the wait for the lock is not timed)
    try:
        t0 = time.monotonic()
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="replace", cwd=os.path.dirname(prog))
        wall = time.monotonic() - t0
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    out = "\n".join(l for l in p.stdout.splitlines() if not TIMING.search(l) and not NUMBER.match(l))   # (PolyBench prints a time on the line after its label)
    summ = [l for l in p.stderr.splitlines() if l.startswith(("host_gpu_verify:", "host_gpu: verify:"))]
    return {"wall": round(wall, 4), "rc": p.returncode, "out": hashlib.sha256(out.encode()).hexdigest()[:16], "summary": summ[-2:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--side", action="append", required=True, help="name=build[:K=V;K=V]")
    ap.add_argument("--stub")
    ap.add_argument("--progs", default="2DCONV,3DCONV,ATAX")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu", default="nvidia/a10g")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    a = ap.parse_args()
    sides = []
    for s in a.side:
        name, rest = s.split("=", 1)
        build, _, envs = rest.partition(":")
        env = dict(kv.split("=", 1) for kv in envs.split(";") if kv)
        sides.append((name, build, env))
    res = {}
    for prog in a.progs.split(","):
        path = os.path.join(a.corp, "polybench", "bin", prog)
        res[prog] = {n: [] for n, _, _ in sides}
        for i in range(a.runs):
            for n, b, e in (sides if i % 2 == 0 else list(reversed(sides))):
                r = run(b, e, path, [], a.gpu, a.threads, a.timeout, a.lock, a.stub)
                res[prog][n].append(r)
                print(prog, n, i, r["wall"], r["rc"], r["out"], *r["summary"], flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f, separators=(",", ":"))
    print()
    print("| program | " + " | ".join(f"{n} median (min-max)" for n, _, _ in sides) + " | outputs equal |")
    print("|---|" + "---:|" * len(sides) + "---|")
    for prog, by in res.items():
        cells = []
        for n, _, _ in sides:
            w = [r["wall"] for r in by[n]]
            cells.append(f"{statistics.median(w):.3f} ({min(w):.3f}-{max(w):.3f})")
        hashes = {r["out"] for rs in by.values() for r in rs}
        print(f"| {prog} | " + " | ".join(cells) + f" | {'yes' if len(hashes) == 1 else 'NO'} |")


if __name__ == "__main__":
    main()
