#!/usr/bin/env python3
"""Compact E1 table for the report. usage: e1_table.py e1_harness.jsonl"""
import json, sys, statistics as st, collections
g = collections.defaultdict(list); order = []
for l in open(sys.argv[1]):
    r = json.loads(l); g[(r["workload"], r["side"])].append(r)
    if r["workload"] not in order: order.append(r["workload"])
m = lambda w, s, k: st.median([r[k] for r in g[(w, s)]])
rg = lambda w, s, k: (min(r[k] for r in g[(w, s)]), max(r[k] for r in g[(w, s)]))
ok = lambda w, s: "%d/%d" % (sum(r["passed"] for r in g[(w, s)]), len(g[(w, s)]))
print("| workload | native device phase (SASS) | sim device phase (SASS) | native faster by | native process | sim process | PASS native / sim | native PTX-JIT device phase (cold) | sim PTX device phase |")
print("|---|---:|---:|---:|---:|---:|---|---:|---:|")
for w in order:
    nd, sd = m(w, "native-sass", "dev_wall"), m(w, "sim-sass", "dev_wall")
    print("| %s | %.3g ms [%.3g..%.3g] | %.3g ms [%.3g..%.3g] | %.3gx | %.0f ms | %.0f ms | %s / %s | %.0f ms | %.3g ms |" % (
        w, nd * 1e3, rg(w, "native-sass", "dev_wall")[0] * 1e3, rg(w, "native-sass", "dev_wall")[1] * 1e3,
        sd * 1e3, rg(w, "sim-sass", "dev_wall")[0] * 1e3, rg(w, "sim-sass", "dev_wall")[1] * 1e3, sd / nd,
        m(w, "native-sass", "wall") * 1e3, m(w, "sim-sass", "wall") * 1e3, ok(w, "native-sass"), ok(w, "sim-sass"),
        m(w, "native-ptx", "dev_wall") * 1e3, m(w, "sim-ptx", "dev_wall") * 1e3))
