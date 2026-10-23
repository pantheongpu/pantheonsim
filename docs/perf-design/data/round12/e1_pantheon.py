#!/usr/bin/env python3
"""E1 for pantheon stress workloads: duration-driven, so compare the program's own 'Throughput:' (work per second).
usage: e1_pantheon.py <bindir> <out.jsonl> <duration_s> <runs> [workloads,...]
sets: 'knobs' = the harness runner's knobs (--kernel_loops 2 --grid_size 4, 1 GB VRAM cap): both sides, comparable to the repo's measurements
      'default' = the program's own default launch geometry (grid auto, kernel_loops default): NATIVE ONLY (the simulator would take hours per launch)"""
import sys, json, re
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
bindir, outp, dur, runs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
wls = sys.argv[5].split(",") if len(sys.argv) > 5 else "int_virus compute_virus fp64_virus mma_virus pulse_virus omni_virus memory_bank_thrash".split()
ALU = {"int_virus", "compute_virus", "fp64_virus"}
KN = ["0", dur, "2", "--kernel_loops", "2", "--grid_size", "4", "--verify"]
DF = ["0", dur, "2", "--verify"]

def one(w, side, kset, i):
    mode = {"native": "native", "sim": "sim"}[side.split("-")[0]]
    extra = {"VGPU_VRAM_MB": "1024", "VGPU_MAX_STEPS": "17179869184"} if mode == "sim" else {}
    if side == "sim-ptx": extra["VGPU_SASS"] = "0"
    if side == "sim-sass-jit": extra["VGPU_JIT"] = "1"
    r = run([bindir + "/" + w] + (KN if kset == "knobs" else DF), env_for(mode, extra=extra), timeout=int(dur) * 40 + 120)
    m = re.search(r"Throughput:\s*(\S+)\s*(\S+)", r["stdout"])
    v = re.search(r"Verification:\s*(\w+)", r["stdout"])
    rec = dict(workload=w, side=side, kset=kset, rep=i, rc=r["rc"], timeout=r["timeout"], wall=r["wall"], cpu=r["cpu"],
               tput=float(m[1]) if m else None, unit=m[2] if m else None, verify=v[1] if v else None,
               load=open("/proc/loadavg").read().split()[0])
    if r["rc"] != 0: rec["tail"] = (r["stderr"] + r["stdout"]).strip().splitlines()[-3:]
    open(outp, "a").write(json.dumps(rec) + "\n")
    print(w, side, kset, i, rec["tput"], rec["unit"], rec["verify"], "rc", r["rc"], "wall %.1f" % r["wall"], flush=True)

for w in wls:
    for i in range(runs):
        one(w, "native", "knobs", i)
        one(w, "native", "default", i)
        one(w, "sim-sass", "knobs", i)
        one(w, "sim-ptx", "knobs", i)
        if w in ALU: one(w, "sim-sass-jit", "knobs", i)
