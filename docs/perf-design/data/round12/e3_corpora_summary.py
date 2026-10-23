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
cats = ["identical", "numeric-close", "differs", "simulator timeout", "simulator fails", "native cannot run", "fail on both", "missing"]
print("| simulated profile | engine the sim ran | programs | identical | numeric-close | differs | native cannot run | simulator timeout | simulator fails | fail on both | identical / (programs that ran on both) |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for p, r in res.items():
    c = collections.Counter(("simulator fails" if cat(*v).startswith("simulator fails") else cat(*v)) for v in r.values())
    ran = sum(c[x] for x in ("identical", "numeric-close", "differs"))
    print("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d | %d / %d (%.0f%%) |" % (p, "SASS (sm_86 cubin compatible)" if p in ("a10g", "l4", "l40s") else "PTX (compute_86 JIT)", len(r), c["identical"], c["numeric-close"], c["differs"], c["native cannot run"], c["simulator timeout"], c["simulator fails"], c["fail on both"], c["identical"], ran, 100.0 * c["identical"] / max(1, ran)))
print()
print("Programs that run on both and differ, by profile (first differing line):\n")
for p, r in res.items():
    if p == "a10g": continue
    rows = [(k, v) for k, v in sorted(r.items()) if v[0] in ("differs", "numeric-close")]
    if not rows: continue
    print("* **%s** (%d)" % (p, len(rows)))
    for k, v in rows: print("  * `%s` %s: %s" % (k, v[0], v[1][:150].replace("`", "'")))
