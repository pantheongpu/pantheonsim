#!/usr/bin/env python3
"""profile.py --build B --corp C --programs polybench/CORR,... [--cpu N] [--secs 60]
perf-records each program on the simulator (one thread, cycles:u, call graphs
on) and prints: share of cycles in the program's own binary (host-side work
such as a CPU reference check), in the simulator libraries, and the top
symbols.  Use a CPU other than the timing CPU."""
import argparse, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real

ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True)
ap.add_argument("--corp", required=True)
ap.add_argument("--programs", required=True)
ap.add_argument("--cpu", type=int, default=14)
ap.add_argument("--secs", type=int, default=60)
ap.add_argument("--top", type=int, default=8)
a = ap.parse_args()
corp = os.path.abspath(a.corp)
want = a.programs.split(",")
for corpus, name, cwd, argv, cp in real.manifest(corp, ["polybench", "rodinia", "hecbench", "samples"]):
    if corpus + "/" + name not in want:
        continue
    env = dict(os.environ, VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", VGPU_THREADS="1",
               LD_LIBRARY_PATH=os.path.join(a.build, "shim"))
    if cp:
        cwd = tempfile.mkdtemp(prefix="prof-")
    data = os.path.join(tempfile.gettempdir(), "prof-%s.data" % name)
    subprocess.run(["timeout", str(a.secs), "perf", "record", "-o", data, "-e", "cycles:u", "-F", "999", "-g",
                    "taskset", "-c", str(a.cpu)] + argv, cwd=cwd, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rep = subprocess.run(["perf", "report", "-i", data, "--no-children", "--sort", "dso,sym", "-g", "none",
                          "--stdio"], capture_output=True, text=True).stdout
    rows = []
    for l in rep.splitlines():
        m = re.match(r"\s*([0-9.]+)%\s+(\S+)\s+\[\.\]\s+(.*)", l)
        if m:
            rows.append((float(m.group(1)), m.group(2), re.sub(r" \[clone[^\]]*\]", "", m.group(3))[:110]))
    own = sum(r[0] for r in rows if not r[1].startswith("lib"))
    print("== %s/%s: %.1f%% in the program's own code, %.1f%% in simulator/libs" % (corpus, name, own, sum(r[0] for r in rows) - own))
    for r in rows[:a.top]:
        print("  %5.1f%% %-18s %s" % r)
    os.remove(data)
