#!/usr/bin/env python3
# Round sp68: the card A/B's JSON files as a markdown table: per program and side the median wall (min-max), the median "GPU Time" where the program
# prints one, x1 against x0, the distinct output hashes, and the host_gpu summary of the last x1 run.
import json, statistics, sys
rows = []
for f in sys.argv[1:]:
    rows += json.load(open(f))
progs = list(dict.fromkeys(r["prog"] for r in rows))
print("| program | build | side | wall s median (min-max) | GPU Time median | x1 vs x0 (wall) | rc | output hashes |")
print("|---|---|---|---|---:|---:|---|---|")
for f in sys.argv[1:]:
    rs = json.load(open(f))
    tag = f.split("/")[-1].replace(".json", "")
    for p in dict.fromkeys(r["prog"] for r in rs):
        med = {}
        for s in dict.fromkeys(r["side"] for r in rs if r["prog"] == p):
            w = [r["wall"] for r in rs if r["prog"] == p and r["side"] == s]
            g = [r["gpu_time"] for r in rs if r["prog"] == p and r["side"] == s and r["gpu_time"] is not None]
            med[s] = statistics.median(w)
            rc = sorted(set(r["rc"] for r in rs if r["prog"] == p and r["side"] == s))
            hs = sorted(set(r["out_hash"] for r in rs if r["prog"] == p and r["side"] == s))
            ratio = "%.3f" % (med["x0"] / med[s]) if s == "x1" and "x0" in med else ""
            print("| %s | %s | %s | %.3f (%.3f-%.3f) | %s | %s | %s | %s |" % (p, tag, s, med[s], min(w), max(w), "%.3f" % statistics.median(g) if g else "-",
                                                                         ratio, ",".join(map(str, rc)), " ".join(hs)))
print()
for r in rows:
    if r["side"] == "x1" and r["run"] == max(x["run"] for x in rows if x["prog"] == r["prog"] and x["side"] == "x1"):
        print("* %s x1: %s %s" % (r["prog"], "; ".join(r["summary"]), "; ".join(r["idiom"])))
