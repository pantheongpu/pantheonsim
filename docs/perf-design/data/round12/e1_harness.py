#!/usr/bin/env python3
"""E1 for the 16 harness workloads: same bench_kernels binary, simulator (profile a10g) vs the real A10G.
usage: e1_harness.py <bench binary> <out.jsonl> <runs_native> <runs_sim> [workloads,...]"""
import sys, json, re
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *

WL = "int_alu f32_fma f64_fma f64_subnormal wmma mma_ptx mem shared atomics texture local divergent warp launch api startup".split()
ALU = {"int_alu", "f32_fma", "f64_fma", "f64_subnormal"}

binp, outp, nn, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
wls = sys.argv[5].split(",") if len(sys.argv) > 5 else WL

def bench(r):
    m = re.search(r"BENCH .*device_cpu_s=(\S+) device_wall_s=(\S+) units=(\d+) unit=(\S+)", r["stdout"])
    return dict(dev_cpu=float(m[1]), dev_wall=float(m[2]), units=int(m[3]), unit=m[4]) if m else {}

def one(w, side, rep):
    # side: native-sass, native-ptx (driver JIT, cold), sim-sass, sim-ptx, sim-sass-jit
    kind, eng = side.split("-")[0], side.split("-")[1]
    mode = {"native": "native", "sim": "sim"}[kind]
    extra = {}
    if kind == "native" and eng == "ptx":
        mode = "ptxjit"
    if kind == "sim" and eng == "ptx":
        extra["VGPU_SASS"] = "0"
    if side == "sim-sass-jit":
        extra["VGPU_JIT"] = "1"
    if side == "sim-ptx-jit":
        extra["VGPU_JIT"] = "1"; extra["VGPU_SASS"] = "0"
    r = run([binp, w, "full"], env_for(mode, extra=extra), timeout=1800)
    rec = dict(workload=w, side=side, rep=rep, rc=r["rc"], timeout=r["timeout"], wall=r["wall"], cpu=r["cpu"],
               passed=r["stdout"].rstrip().endswith("PASS"), **bench(r), load=open("/proc/loadavg").read().split()[0])
    if not rec["passed"]:
        rec["tail"] = (r["stderr"] + r["stdout"]).strip().splitlines()[-3:]
    with open(outp, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(w, side, rep, "wall %.3f dev %.4f pass=%s" % (rec["wall"], rec.get("dev_wall", -1), rec["passed"]), flush=True)

for w in wls:
    sides_n = ["native-sass", "native-ptx"]
    sides_s = ["sim-sass", "sim-ptx"] + (["sim-sass-jit", "sim-ptx-jit"] if w in ALU else [])
    # one discarded native warm-up (driver/context init is slow on the first process after boot)
    run([binp, "startup", "quick"], env_for("native"), timeout=120) if w == wls[0] else None
    for i in range(nn):
        for s in sides_n: one(w, s, i)
    for i in range(ns):
        for s in sides_s: one(w, s, i)
