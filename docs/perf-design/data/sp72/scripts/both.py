#!/usr/bin/env python3
"""Rows of an A/B that are low in each of several passes (bench-amd JSON files, one a pass, the same two builds' roles in each).

  python3 -I both.py <bound> <json>...      e.g. both.py 0.97 bench-off1.json bench-offF.json

For every workload and target, B / A of the best run of each pass; prints the rows below the bound in every pass, and the
geometric mean of each pass. (A row that is low in one pass and not in another is the host, not the code.)
"""
import json
import math
import sys

bound = float(sys.argv[1])
passes = []
for path in sys.argv[2:]:
    d = json.load(open(path))
    a, b = d["meta"]["builds"]
    by = {}
    for r in d["rows"]:
        if str(r["threads"]) != "1":
            continue
        by.setdefault((r["workload"], r["target"]), {})[r["build"]] = r["stat"]["values"]
    ratio = {}
    for k, v in by.items():
        if a in v and b in v:
            pick = min if k[0] == "startup" else max
            ratio[k] = pick(v[b]) / pick(v[a])
    passes.append(ratio)
keys = [k for k in passes[0] if all(k in p for p in passes)]
for i, p in enumerate(passes):
    low = sum(1 for k in keys if p[k] < bound)
    print("pass %d: geometric mean %.3f, %d of %d rows below %.2f, lowest %.3f, highest %.3f" %
          (i + 1, math.exp(sum(math.log(p[k]) for k in keys) / len(keys)), low, len(keys), bound, min(p[k] for k in keys), max(p[k] for k in keys)))
both = [k for k in keys if all(p[k] < bound for p in passes)]
print("below %.2f in every pass: %s" % (bound, ", ".join("%s/%s (%s)" % (k[0], k[1], ", ".join("%.3f" % p[k] for p in passes)) for k in both) or "none"))
