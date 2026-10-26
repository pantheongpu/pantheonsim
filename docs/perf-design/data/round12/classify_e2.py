#!/usr/bin/env python3
"""E2 table: for every case, native-vs-native, native-vs-sim (two sim runs where the first differed), and the verdict class.
usage: classify_e2.py <rundir> [notes.json] > table.md
notes.json: {"corpus/case": ["class", "reason"]} manual classification for cases that still differ after the filters."""
import sys, json, collections
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from compare import load, classify
d = sys.argv[1]
notes = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
N1, N2, S1, S2 = (load(d, t) for t in ("native1", "native2", "sim1", "sim2"))
rows, tally, raw = [], collections.Counter(), collections.Counter()
for k in sorted(N1):
    nn = classify(N1.get(k), N2.get(k))[0]
    ns1 = classify(N1.get(k), S1.get(k))
    ns2 = classify(N1.get(k), S2.get(k)) if k in S2 else None
    ss = classify(S1.get(k), S2.get(k))[0] if k in S2 else "-"
    best = ns2 if (ns2 and ns2[0] in ("identical", "timing-only") and ns1[0] not in ("identical", "timing-only")) else ns1
    if k in S2 and ns2 and ns1[0] in ("B-fails",) and ns2[0] not in ("B-fails",): best = ns2
    c = best[0]
    if c in ("identical", "timing-only"): v = "bit-identical"; raw[c] += 1
    elif c == "both-fail-same": v = "fails the same way on both"
    elif c == "both-fail-differ": v = "fails on both (differently)"
    elif k in notes: v = notes[k][0]
    elif c in ("B-fails",): v = "sim: " + best[1]
    elif c in ("A-fails",): v = "native fails"
    elif nn not in ("identical", "timing-only"): v = "native not run-to-run stable"
    else: v = "UNCLASSIFIED " + c
    tally[v] += 1
    rows.append((k, nn, ns1[0], ns2[0] if ns2 else "-", ss, v, notes.get(k, ["", best[1]])[1] if v != "bit-identical" else ""))
print("| case | native vs native | native vs sim (run 1) | native vs sim (run 2) | sim vs sim | verdict | note |")
print("|---|---|---|---|---|---|---|")
for r in rows: print("| " + " | ".join(x.replace("|", "/") for x in r) + " |")
print()
for v, n in tally.most_common(): print("* %s: %d" % (v, n))
print("* total:", sum(tally.values()))
print("* of the bit-identical: %d byte-identical, %d identical after the timing filter" % (raw["identical"], raw["timing-only"]))
