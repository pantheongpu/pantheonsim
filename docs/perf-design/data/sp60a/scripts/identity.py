#!/usr/bin/env python3
"""Round sp60a: reads an --identity record file of ab.py (forms a, A, b, c, m; one run each) and says, per program,
whether the normalised output, the final memory hash (VGPU_MEMO_FINAL_HASH), the counters file and the kernel digest
of every form are those of form a.  A is a second default run: a program whose A differs from a does not repeat itself.

  identity.py runs-identity-t1.jsonl
"""
import json, sys
from collections import defaultdict

runs = defaultdict(dict)
for line in open(sys.argv[1]):
    r = json.loads(line)
    runs[r["prog"]][r["form"]] = r
keys = ("out", "final", "counters", "digest")
n = same = unrepeatable = incomplete = differ = 0
print("| program | forms compared | result |")
print("|---|---|---|")
for prog in sorted(runs):
    f = runs[prog]
    a = f.get("a")
    if not a or a["timeout"] or any(r["timeout"] for r in f.values()):
        incomplete += 1
        print("| %s | - | not compared (a run passed the time limit: the digest of every launch makes many-launch programs slow) |" % prog)
        continue
    n += 1
    bad = {}
    for name, r in f.items():
        if name == "a":
            continue
        d = [k for k in keys if r.get(k) != a.get(k)] + (["exit status"] if r["rc"] != a["rc"] else [])
        if d:
            bad[name] = d
    others = ",".join(sorted(x for x in f if x != "a"))
    if not bad:
        same += 1
        print("| %s | %s | same (rc %s) |" % (prog, others, a["rc"]))
    elif "A" in bad:
        unrepeatable += 1
        print("| %s | %s | the default does not repeat itself (%s); others: %s |" % (prog, others, ", ".join(bad["A"]), "; ".join("%s: %s" % (k, ", ".join(v)) for k, v in sorted(bad.items()) if k != "A") or "same"))
    else:
        differ += 1
        print("| %s | %s | DIFFER: %s |" % (prog, others, "; ".join("%s: %s" % (k, ", ".join(v)) for k, v in sorted(bad.items()))))
print("\n%d programs compared: %d the same in every form, %d that do not repeat themselves in the default, %d that differ in a form only; %d not compared" % (n, same, unrepeatable, differ, incomplete))
