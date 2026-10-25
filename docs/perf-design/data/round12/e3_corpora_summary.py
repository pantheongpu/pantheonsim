#!/usr/bin/env python3
"""E3 on the corpora: native PTX-JIT on the A10G (tag ptxjit1) vs the simulator on other profiles (tags e3-<profile>), IDENT=1 filter.
usage: IDENT=1 e3_corpora_summary.py <rundir> [profiles...]"""
import os, sys, collections
os.environ["IDENT"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import load, classify
d = sys.argv[1]
profs = sys.argv[2:] or ["a10g", "l4", "h100", "rtx5090", "a100", "t4"]
A = load(d, "ptxjit1")
res = {}
for p in profs:
    tag = "sim2" if p == "a10g" else "e3-" + p   # a10g: the same-architecture run, for reference
    B = {**load(d, "sim1"), **load(d, "sim2")} if p == "a10g" else load(d, tag)
    if not B: continue
    res[p] = {k: classify(A.get(k), B.get(k)) for k in A}
def cat(c, why):
    if c in ("identical", "timing-only"): return "identical"
    if c == "numeric-close": return "numeric-close"
    if c == "differs": return "differs"
    if c == "A-fails": return "native cannot run"
    if c == "B-fails": return "simulator timeout" if why == "timeout" else "simulator fails (%s)" % why
    if c.startswith("both-fail"): return "fail on both"
    return c
GROUPS = [("sm_86-only builds (Rodinia 12, PolyBench 20, HeCBench 34)", lambda k: not k.startswith("samples/")),
          ("CUDA Samples (98; multi-architecture fatbin sm_75..sm_120)", lambda k: k.startswith("samples/"))]
ENGINE = {"a10g": ("SASS sm_86", "SASS of the profile's own architecture"), "l4": ("SASS sm_86 (cubin-compatible with sm_89)", "SASS sm_89"),
          "h100": ("PTX compute_86 (JIT in the simulator)", "SASS sm_90"), "rtx5090": ("PTX compute_86", "SASS sm_120"),
          "a100": ("none: sm_86 SASS and compute_86 PTX are refused", "SASS sm_80"), "t4": ("none: refused", "SASS sm_75")}
print("| simulated profile | group | engine the simulator ran | programs | identical | numeric-close | differs | native cannot run | simulator timeout | simulator fails | fail on both | identical / (ran on both) |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for p, r in res.items():
    for gname, f in GROUPS:
        sub = {k: v for k, v in r.items() if f(k)}
        c = collections.Counter(("simulator fails" if cat(*v).startswith("simulator fails") else cat(*v)) for v in sub.values())
        ran = sum(c[x] for x in ("identical", "numeric-close", "differs"))
        eng = ENGINE[p][0 if gname.startswith("sm_86") else 1]
        if p in ("a100", "t4") and gname.startswith("sm_86"):
            print("| %s | %s | %s | %d | n/a | n/a | n/a | 0 | n/a | n/a | n/a | **no kernel ran**: every launch fails with 'module targets sm_86 but this device is compute capability %s; PTX runs on newer architectures, not older ones'; the programs do not check the error, so %d of 66 print output that matches or looks plausible (not a comparison) |" % (p, gname.split(" (")[0], eng, len(sub), "8.0" if p == "a100" else "7.5", c["identical"]))
            continue
        print("| %s | %s | %s | %d | %d | %d | %d | %d | %d | %d | %d | %d / %d (%.0f%%) |" % (p, gname.split(" (")[0], eng, len(sub), c["identical"], c["numeric-close"], c["differs"], c["native cannot run"], c["simulator timeout"], c["simulator fails"], c["fail on both"], c["identical"], ran, 100.0 * c["identical"] / max(1, ran)))
print()
print("Programs that run on both and differ, by profile (first differing line):\n")
for p, r in res.items():
    if p == "a10g": continue
    rows = [(k, v) for k, v in sorted(r.items()) if v[0] in ("differs", "numeric-close") and (p not in ("a100", "t4") or k.startswith("samples/"))]
    if not rows: continue
    print("* **%s** (%d)" % (p, len(rows)))
    for k, v in rows: print("  * `%s` %s: %s" % (k, v[0], v[1][:150].replace("`", "'")))
