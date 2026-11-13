#!/usr/bin/env python3
"""Round sp60a: tables from the run records of ab.py.

  tables.py --runs runs-default.jsonl --json out.json > table.md

Per program and form: the median wall time of the runs that finished, the spread
((max - min) / median), and the ratios  a/b, a/c (and a/m, a/M when those forms
were run).  A ratio above 1 means the default (a) is slower than the other form.
"""
import argparse, json, math, statistics, sys
from collections import defaultdict


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def load(path):
    runs = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            runs[r["prog"]][r["form"]].append(r)
    return runs


def summarise(runs, min_secs):
    out = {}
    for prog, forms in runs.items():
        rec = {}
        for f, rs in forms.items():
            ok = [r["wall"] for r in rs if not r["timeout"]]
            d = {"n": len(ok), "timeouts": sum(1 for r in rs if r["timeout"]), "rc": sorted(set(r["rc"] for r in rs)),
                 "outs": sorted(set(r["out"] for r in rs if not r["timeout"]))}
            if ok:
                d["s"] = round(statistics.median(ok), 4)
                d["spread"] = round((max(ok) - min(ok)) / d["s"], 3) if d["s"] > 0 else 0
            for k in ("final", "counters", "digest"):
                vals = sorted(set(str(r.get(k)) for r in rs if k in r and not r["timeout"]))
                if vals:
                    d[k] = vals
            memo = [r["memo"] for r in rs if r.get("memo")]
            if memo:
                d["memo"] = memo[-1]
            rec[f] = d
        a = rec.get("a", {}).get("s")
        for f in rec:
            if f != "a" and a and rec[f].get("s"):
                rec[f]["a_over"] = round(a / rec[f]["s"], 4)
        outs = set()
        for f in rec:
            outs.update(rec[f]["outs"])
        rec["_same_out"] = len(outs) <= 1
        rec["_timed"] = bool(a and a >= min_secs)
        out[prog] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--json")
    ap.add_argument("--min-secs", type=float, default=0.3, help="programs shorter than this in form a are listed but left out of the geometric means (they time process start-up)")
    ap.add_argument("--title", default="")
    ap.add_argument("--replace", default="", help="a second record file: its programs replace the same programs of --runs")
    ap.add_argument("--drop", default="", help="programs to leave out, by name")
    a = ap.parse_args()
    runs = load(a.runs)
    if a.replace:   # a later, cleaner pass of some programs takes the place of their records
        for prog, forms in load(a.replace).items():
            runs[prog] = forms
    for prog in [p for p in runs if p.split("/")[1] in a.drop.split(",")]:
        del runs[prog]
    s = summarise(runs, a.min_secs)
    forms = sorted(set(f for p in s.values() for f in p if not f.startswith("_")))
    others = [f for f in forms if f != "a"]
    if a.title:
        print("### " + a.title + "\n")
    head = "| program | runs | " + " | ".join("%s s (spread)" % f for f in forms) + " | " + " | ".join("a/%s" % f for f in others) + " | outputs |"
    print(head)
    print("|---|---:|" + "---:|" * (len(forms) + len(others)) + "---|")
    for prog in sorted(s):
        p = s[prog]
        cells = []
        for f in forms:
            d = p.get(f)
            if not d or "s" not in d:
                cells.append("TIMEOUT" if d and d["timeouts"] else "-")
            else:
                cells.append("%.3f (%d%%)" % (d["s"], round(100 * d["spread"])) + (" rc%s" % d["rc"] if d["rc"] != [0] else ""))
        rat = ["%.3f" % p[f]["a_over"] if f in p and "a_over" in p[f] else "-" for f in others]
        n = p.get("a", {}).get("n", 0)
        print("| %s%s | %d | %s | %s | %s |" % (prog, "" if p["_timed"] else " (short)", n, " | ".join(cells), " | ".join(rat), "same" if p["_same_out"] else "DIFFER"))
    print()
    g = {}
    for f in others:
        xs = [p[f]["a_over"] for p in s.values() if p["_timed"] and f in p and "a_over" in p[f]]
        g[f] = {"geomean": round(geomean(xs), 4), "n": len(xs), "min": min(xs) if xs else None, "max": max(xs) if xs else None,
                "over_1.03": sum(1 for x in xs if x > 1.03), "under_0.97": sum(1 for x in xs if x < 0.97)}
        print("a/%s: geometric mean %.4f over %d programs of %.1f s or more (smallest %.3f, largest %.3f; above 1.03: %d, below 0.97: %d)"
              % (f, g[f]["geomean"], g[f]["n"], a.min_secs, g[f]["min"], g[f]["max"], g[f]["over_1.03"], g[f]["under_0.97"]))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"programs": s, "geomeans": g, "min_secs": a.min_secs}, fh, indent=0, sort_keys=True)


if __name__ == "__main__":
    main()
