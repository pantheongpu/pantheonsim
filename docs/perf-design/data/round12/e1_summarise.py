#!/usr/bin/env python3
"""Summarise e1_harness.jsonl: median [min..max] of whole-process wall and device-phase wall per workload and side."""
import json, sys, statistics as st, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
g = collections.defaultdict(list)
for r in rows: g[(r["workload"], r["side"])].append(r)
def f(xs): return "%.4g [%.3g..%.3g]" % (st.median(xs), min(xs), max(xs))
order = []
for r in rows:
    if r["workload"] not in order: order.append(r["workload"])
print("| workload | side | n | pass | process wall s | device-phase wall s |")
print("|---|---|---:|---|---|---|")
for w in order:
    for s in ["native-sass","native-ptx","sim-sass","sim-ptx","sim-sass-jit","sim-ptx-jit"]:
        rs = g.get((w, s))
        if not rs: continue
        print("| %s | %s | %d | %s | %s | %s |" % (w, s, len(rs), "%d/%d" % (sum(r["passed"] for r in rs), len(rs)),
              f([r["wall"] for r in rs]), f([r["dev_wall"] for r in rs if "dev_wall" in r]) if any("dev_wall" in r for r in rs) else "-"))
print()
print("| workload | engine | dev-phase speed-up native/sim (median) | process-wall speed-up native/sim (median) | sim JIT / sim no-JIT dev |")
print("|---|---|---:|---:|---:|")
for w in order:
    for e in ("sass","ptx"):
        n, s = g.get((w,"native-"+e)), g.get((w,"sim-"+e))
        if not n or not s: continue
        dn = st.median([r["dev_wall"] for r in n]); ds = st.median([r["dev_wall"] for r in s])
        pn = st.median([r["wall"] for r in n]); ps = st.median([r["wall"] for r in s])
        j = g.get((w,"sim-%s-jit"%e))
        jj = "%.2f" % (ds / st.median([r["dev_wall"] for r in j])) if j else "-"
        print("| %s | %s | %.3gx | %.3gx | %s |" % (w, e, ds/dn, ps/pn, jj))
