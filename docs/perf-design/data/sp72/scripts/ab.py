#!/usr/bin/env python3
"""An A/B table from one or more bench-amd JSON files of the same two builds (tools/bench-amd/bench.py --build A --build B --json).

  python3 -I ab.py [--md] [--threads 1] <label A> <label B> <json>...

For each workload and target: the rate of each build (wave-instructions, or the workload's own unit, per CPU second; the best run
of all the runs of all the files, or with --median their median), B / A, and whether every counter of the counting runs is the same
in both builds. Then the geometric mean, and the rows below 0.97 and above 1.05. The best run is the default because the host is
shared: another session's work makes a run slower, never faster, so the fastest of many runs is the one least disturbed.
"""
import json
import math
import statistics
import sys

args = sys.argv[1:]
md = "--md" in args
if md:
    args.remove("--md")
use_median = "--median" in args
if use_median:
    args.remove("--median")
threads = "1"
if "--threads" in args:
    i = args.index("--threads")
    threads = args[i + 1]
    del args[i:i + 2]
la, lb = args[0], args[1]
vals, counts, builds = {}, {}, None
for path in args[2:]:
    d = json.load(open(path))
    builds = d["meta"]["builds"]
    for r in d["rows"]:
        if str(r["threads"]) != threads:
            continue
        vals.setdefault((r["workload"], r["target"]), {}).setdefault(r["build"], []).extend(r["stat"]["values"])
    for k, v in d["counts"].items():
        tg, wl, b = k.split("/", 2)
        counts.setdefault((wl, tg), {})[b.lstrip("/") if False else "/" + b.lstrip("/")] = v
a, b = builds
rows, same_all, counted = [], True, 0
for (wl, tg), by in vals.items():
    if a not in by or b not in by:
        continue
    lower = (wl == "startup")   # (its metric is wall seconds: lower is better)
    pick = statistics.median if use_median else (min if lower else max)
    ma, mb = pick(by[a]), pick(by[b])
    c = counts.get((wl, tg), {})
    same = None
    if a in c and b in c:
        same = c[a] == c[b]
        counted += 1
        same_all = same_all and same
    rows.append((wl, tg, ma, mb, mb / ma, same, len(by[a])))
sep = " | " if md else "  "
if md:
    print("| workload | target | %s | %s | %s / %s | counters |" % (la, lb, lb, la))
    print("|---|---|---:|---:|---:|---|")
for wl, tg, ma, mb, ratio, same, n in rows:
    unit = "M/s" if ma > 1e5 else "/s"
    fa, fb = (ma / 1e6, mb / 1e6) if ma > 1e5 else (ma, mb)
    cs = "-" if same is None else "same" if same else "DIFFER"
    if md:
        print("| %s | %s | %.2f %s | %.2f %s | %.3f | %s |" % (wl, tg, fa, unit, fb, unit, ratio, cs))
    else:
        print("%-14s %-7s %9.2f %9.2f %s  %.3f  %s" % (wl, tg, fa, fb, unit, ratio, cs))
g = math.exp(sum(math.log(r[4]) for r in rows) / len(rows))
print()
print("%s of %d runs a side; " % ("median" if use_median else "best", rows[0][6]), end="")
print("%d rows, %d runs a side each; geometric mean of %s / %s: %.3f; counters the same in %s of the %d rows counted" %
      (len(rows), rows[0][6], lb, la, g, "all" if same_all else "NOT ALL", counted))
print("below 0.97: " + (", ".join("%s/%s %.3f" % (r[0], r[1], r[4]) for r in rows if r[4] < 0.97) or "none"))
print("1.05 or more: " + (", ".join("%s/%s %.3f" % (r[0], r[1], r[4]) for r in sorted(rows, key=lambda r: -r[4]) if r[4] >= 1.05) or "none"))
