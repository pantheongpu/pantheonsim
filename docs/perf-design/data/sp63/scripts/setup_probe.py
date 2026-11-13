#!/usr/bin/env python3
"""setup_probe.py --corp C --build B --work W --progs p1,p2 --runs N [--sides a,b,...]: what VGPU_JIT=1 costs a short program, side by side, interleaved.
Sides: a no VGPU_JIT; b VGPU_JIT=1; u VGPU_JIT=1 VGPU_JIT_MIN_REGION=4096 (no section gets a unit: region discovery only); s VGPU_JIT=1
VGPU_JIT_SAMPLE=1000 (the share rule's sample in the hook loop 50 times shorter); p VGPU_JIT=1 VGPU_JIT_PROFILE=0 (no waits).  One thread, the cache
in W/jc (kept: a warm cache).  A side with a leading 2 (2a, 2b) runs on --build2.  Prints per program the median wall of each side in ms and the difference to a."""
import argparse, importlib.util, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sp63", os.path.join(HERE, "sp63.py")); sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
V = {"a": {}, "b": {"VGPU_JIT": "1"}, "u": {"VGPU_JIT": "1", "VGPU_JIT_MIN_REGION": "4096"}, "s": {"VGPU_JIT": "1", "VGPU_JIT_SAMPLE": "1000"},
     "p": {"VGPU_JIT": "1", "VGPU_JIT_PROFILE": "0"}, "B": {"VGPU_JIT": "1", "VGPU_JIT_STATS": "1"}}
ap = argparse.ArgumentParser(); ap.add_argument("--corp"); ap.add_argument("--build"); ap.add_argument("--work"); ap.add_argument("--progs"); ap.add_argument("--runs", type=int, default=21)
ap.add_argument("--build2", default=""); ap.add_argument("--sides", default="a,b,u,s"); ap.add_argument("--cpus", default=""); ap.add_argument("--threads", default="1"); ap.add_argument("--timeout", type=float, default=120)
a = ap.parse_args(); a.envd = {}; a.corp, a.work, a.build = (os.path.abspath(x) for x in (a.corp, a.work, a.build)); os.makedirs(a.work, exist_ok=True)
progs = {p[0] + "/" + p[1]: p for p in sp.real.manifest(a.corp, ["polybench", "rodinia", "hecbench"])}
sides = a.sides.split(","); b1 = a.build
for pid in a.progs.split(","):
    prog = progs[pid]; W = {s: [] for s in sides}
    for i in range(a.runs):
        for s in (sides if i % 2 == 0 else sides[::-1]):
            a.build = os.path.abspath(a.build2) if s[0] == "2" else b1; v = s.lstrip("2")
            W[s].append(sp.run1(a, prog, dict(V[v], VGPU_JIT_CACHE=os.path.join(a.work, "jc" + s[0])) if v != "a" else {}, a.work)["wall"])
    m = {s: statistics.median(W[s]) * 1000 for s in sides}
    print("%-32s " % pid + " ".join("%s=%.2f" % (s, m[s]) for s in sides) + "  | " + " ".join("%s-a=%+.2f" % (s, m[s] - m["a"]) for s in sides if s != "a"), flush=True)
sp.SLOT.release()
