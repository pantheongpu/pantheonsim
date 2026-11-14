#!/usr/bin/env python3
"""A table from qall.sh's lines: for each row, the ratio of the first spec's CPU seconds to each other spec's (above 1: that spec is faster).

  python3 -I qtab.py [--md] <q-*.txt> [name=spec ...]     names rename the specs in the heading
"""
import math
import sys

args = sys.argv[1:]
md = "--md" in args
if md:
    args.remove("--md")
names = dict(a.split("=", 1)[::-1] for a in args[1:])
rows = []
for line in open(args[0]):
    p = line.split()
    if len(p) < 4 or p[0] == "DONE":
        continue
    vals = []
    for f in p[2:]:
        spec, v = f.rsplit("=", 1)
        vals.append((spec, float(v)))
    rows.append((p[0], "mi300x" if p[1] == "gfx942" else "rx6800", vals))
specs = [s for s, _ in rows[0][2]]
head = ["row", "target", names.get(specs[0], specs[0]) + " (s)"] + [names.get(s, s) for s in specs[1:]]
if md:
    print("| " + " | ".join(head) + " |")
    print("|---|---|" + "---:|" * (len(head) - 2))
else:
    print("  ".join(head))
prod = [0.0] * (len(specs) - 1)
low = [[] for _ in specs[1:]]
for w, tg, vals in rows:
    base = vals[0][1]
    r = [base / v for _, v in vals[1:]]
    for i, x in enumerate(r):
        prod[i] += math.log(x)
        if x < 0.97:
            low[i].append("%s/%s %.3f" % (w, tg, x))
    cells = [w, tg, "%.4f" % base] + ["%.3f" % x for x in r]
    print(("| " + " | ".join(cells) + " |") if md else "  ".join("%-12s" % c for c in cells))
print()
for i, s in enumerate(specs[1:]):
    print("%s: geometric mean %.3f over %d rows; below 0.97: %s" % (names.get(s, s), math.exp(prod[i] / len(rows)), len(rows), ", ".join(low[i]) or "none"))
