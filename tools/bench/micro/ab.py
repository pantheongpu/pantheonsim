#!/usr/bin/env python3
"""Alternating A/B of one program on two simulator builds: ab.py A_BUILD B_BUILD --runs N --cpus 6 [--cwd D] -- prog args...
Prints user-CPU medians, best, A/B and the Mann-Whitney p of bench.py, plus the load average. Output of the program is
compared between A and B (stdout must be identical)."""
import argparse, importlib.util, os, resource, statistics, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bench", os.path.join(here, "..", "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--runs", type=int, default=9)
ap.add_argument("--cpus", default="6")
ap.add_argument("--cwd", default=".")
ap.add_argument("--env", action="append", default=[])
ap.add_argument("cmd", nargs="+")
o = ap.parse_args()

def once(build):
    env = dict(os.environ, VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", VGPU_THREADS="1",
               LD_LIBRARY_PATH=os.path.join(build, "shim"))
    for kv in o.env:
        k, v = kv.split("=", 1); env[k] = v
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
    r = subprocess.run(["taskset", "-c", o.cpus] + o.cmd, cwd=o.cwd, env=env, capture_output=True, timeout=900)
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime - before, r.returncode, r.stdout

once(o.a); once(o.b)   # warmup
ta, tb, outs = [], [], set()
for _ in range(o.runs):
    for build, t in ((o.a, ta), (o.b, tb)):
        s, rc, out = once(build)
        t.append(s); outs.add(out)
        if rc: print("exit", rc, file=sys.stderr)
ma, mb = statistics.median(ta), statistics.median(tb)
print("A median %.3f best %.3f | B median %.3f best %.3f | A/B median %.3fx best %.3fx | p=%.4f | load %s | stdout identical: %s"
      % (ma, min(ta), mb, min(tb), ma / mb, min(ta) / min(tb), bench.mann_whitney_p(ta, tb),
         " ".join("%.2f" % x for x in os.getloadavg()), len(outs) == 1))
