#!/usr/bin/env python3
"""Round sp65: how much slower the A10G rig's simulator was than server3's, per program: the simulator seconds Round sp59 recorded for a program
none of whose launches went to the card (`sim_s` of its exact or turbo form with `card` 0: every launch's time) over the sum of the launch times in
server3's trace of the same program. The replay multiplies a trace's times by this to decide as the rig would have (--sim-scale).

  sim_scale.py --bench ../sp59/bench.json --traces DIR [--json out.json]
"""
import argparse, json, os, statistics
ap = argparse.ArgumentParser(); ap.add_argument("--bench", required=True); ap.add_argument("--traces", required=True); ap.add_argument("--json", default="")
a = ap.parse_args()
b = json.load(open(a.bench))
out = {}
for suite in ("polybench", "rodinia", "hecbench"):
    for prog, forms in b[suite]["programs"].items():
        tr = os.path.join(a.traces, prog.replace("/", "_") + ".turbo.trace")
        if not os.path.exists(tr): continue
        tot = sum(float(l.split()[3]) for l in open(tr, errors="replace") if l.startswith("L ") and len(l.split()) > 9)
        for form in ("turbo", "exact"):
            r = forms.get(form)
            if r and r.get("card") == 0 and r.get("sim_s") and tot > 0.01:
                out[prog] = round(r["sim_s"] / tot, 3); break
for k, v in sorted(out.items()): print("%-36s %.2f" % (k, v))
if out:
    v = sorted(out.values()); print("programs %d: min %.2f median %.2f max %.2f" % (len(v), v[0], statistics.median(v), v[-1]))
if a.json: json.dump(out, open(a.json, "w"), indent=0, sort_keys=True)
