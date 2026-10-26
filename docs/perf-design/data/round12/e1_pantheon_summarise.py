#!/usr/bin/env python3
"""Summarise e1_pantheon.jsonl: median [min..max] Throughput per workload and side, ratio native/sim, verification verdicts."""
import json, sys, statistics as st, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
g = collections.defaultdict(list)
order = []
for r in rows:
    if r["workload"] not in order: order.append(r["workload"])
    g[(r["workload"], r["side"], r["kset"])].append(r)
def f(rs):
    xs = [r["tput"] for r in rs if r["tput"] is not None]
    return (st.median(xs), min(xs), max(xs)) if xs else None
def fmt(x): return "%.4g [%.3g..%.3g]" % x if x else "-"
print("| workload | unit | native, harness knobs | native, program defaults | sim sass | sim ptx | sim sass+JIT | verify (all runs) |")
print("|---|---|---|---|---|---|---|---|")
for w in order:
    u = next(r["unit"] for r in rows if r["workload"] == w and r["unit"])
    cells = [fmt(f(g.get((w, s, k), []))) for s, k in (("native","knobs"),("native","default"),("sim-sass","knobs"),("sim-ptx","knobs"),("sim-sass-jit","knobs"))]
    vs = collections.Counter(str(r["verify"]) + ("" if r["rc"] == 0 else "/rc%d" % r["rc"]) for r in rows if r["workload"] == w)
    print("| %s | %s | %s | %s |" % (w, u, " | ".join(cells), ", ".join("%s x%d" % kv for kv in vs.items())))
print()
print("| workload | native(knobs)/sim sass | native(knobs)/sim ptx | native(defaults)/sim sass(knobs) | sim JIT/sim sass |")
print("|---|---:|---:|---:|---:|")
for w in order:
    nk = f(g.get((w,"native","knobs"),[])); nd = f(g.get((w,"native","default"),[]))
    s = f(g.get((w,"sim-sass","knobs"),[])); p = f(g.get((w,"sim-ptx","knobs"),[])); j = f(g.get((w,"sim-sass-jit","knobs"),[]))
    r = lambda a, b: "%.3gx" % (a[0]/b[0]) if a and b and b[0] else "-"
    print("| %s | %s | %s | %s | %s |" % (w, r(nk,s), r(nk,p), r(nd,s), r(j,s)))
