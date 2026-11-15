#!/usr/bin/env python3
"""A table from inq2.sh's lines (out/inq-<tag>.txt): for each test the best run of each of the two specs -- wall seconds and CPU seconds
(user + system) of the whole process -- the second's CPU over the first's, and whether every run passed with the same output checksum.

  python3 -I inq2tab.py <inq-*.txt> [name=spec ...]
"""
import sys

names = dict(a.split("=", 1)[::-1] for a in sys.argv[2:])
runs, order, specs = {}, [], []
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) < 8 or not p[2][0].isdigit():
        continue
    spec = p[0][:-1]          # (the pass number is the label's last character)
    test, wall, cpu = p[1], float(p[2]), float(p[3]) + float(p[4])
    if test not in order:
        order.append(test)
    if spec not in specs:
        specs.append(spec)
    runs.setdefault((test, spec), []).append((cpu, wall, p[5] == "rc=0" and p[6] == "pass=1", p[7]))
a, b = specs
print("| test | %s: wall s / CPU s | %s: wall s / CPU s | CPU, second / first | runs a side | all pass, same output |" % (names.get(a, a), names.get(b, b)))
print("|---|---:|---:|---:|---:|---|")
for t in order:
    ra, rb = runs[(t, a)], runs[(t, b)]
    ba, bb = min(ra), min(rb)
    ok = all(r[2] for r in ra + rb) and len({r[3] for r in ra + rb}) == 1
    print("| %s | %.1f / %.1f | %.1f / %.1f | %.3f | %d | %s |" % (t, min(r[1] for r in ra), ba[0], min(r[1] for r in rb), bb[0], bb[0] / ba[0], len(ra), "yes" if ok else "NO"))
