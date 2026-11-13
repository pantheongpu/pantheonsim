#!/usr/bin/env python3
"""Round sp60a: what round sp59's own records say about "the plain simulator is slower than a host-GPU run
that sends nothing to the card", and the speed-ups with that removed.

  sp59_baseline.py <sp59 bench.json> [--json out.json]

1. Every program and card form with a simulator time and 0 launches on the card: sim / form.  If something the
   VGPU_HOST_GPU variable switches made the plain simulator slower, this ratio would sit above 1 for all of them.
2. The plain simulator against itself: the main pass against the START_COST=0 pass (same rig, same build, same
   programs, hours apart).
3. The headline numbers again with the speed-up of a form that sent nothing to the card set to 1.00 where it
   read above 1.05 (it cannot have gained from a card it did not use).
"""
import json, math, sys

def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

d = json.load(open(sys.argv[1]))
suites = ("rodinia", "polybench", "hecbench")
forms = ("exact", "exactdiv", "turbo")
out = {"zero_card": {}, "sim_vs_sim": {}, "headline": {}}
print("### 1. sim / form where the form sent 0 launches to the card\n")
print("| program | sim s (runs) | exact | exact + DIV | turbo |")
print("|---|---:|---:|---:|---:|")
allr = {f: [] for f in forms}
for s in suites:
    for k, v in d[s]["programs"].items():
        sim = v.get("sim", {})
        row, any0 = [], False
        for f in forms:
            e = v.get(f, {})
            if e.get("card") == 0 and e.get("x") is not None:
                row.append("%.3f" % e["x"]); allr[f].append((k, e["x"], sim["s"])); any0 = True
            else:
                row.append("-")
        if any0:
            out["zero_card"][k] = {f: (v[f]["x"] if v.get(f, {}).get("card") == 0 else None) for f in forms}
            print("| %s | %.2f (%s) | %s |" % (k, sim["s"], sim.get("runs") or 1, " | ".join(row)))
print()
for f in forms:
    xs = [x for _, x, _ in allr[f]]
    big = [x for _, x, s in allr[f] if s >= 0.3]
    outl = ["%s %.3f" % (k.split("/")[1], x) for k, x, s in allr[f] if s >= 0.3 and abs(x - 1) > 0.05]
    out["zero_card"]["_" + f] = {"n": len(xs), "geomean": round(gm(xs), 4), "n_0.3s": len(big), "geomean_0.3s": round(gm(big), 4),
                                "median_0.3s": sorted(big)[len(big) // 2], "outside_5pct": outl}
    print("%s: %d programs, geometric mean %.4f; of 0.3 s or more: %d, geometric mean %.4f, median %.3f, more than 5 %% from 1: %s"
          % (f, len(xs), gm(xs), len(big), gm(big), sorted(big)[len(big) // 2], ", ".join(outl) or "none"))
print("\n### 2. The plain simulator against itself (main pass / START_COST=0 pass)\n")
print("| program | sim, main pass | sim, START_COST=0 pass (1 run) | ratio |")
print("|---|---:|---:|---:|")
rs = []
for k, v in d["start_cost_0"]["programs"].items():
    s1 = v.get("sim", {}).get("s")
    s0 = next((d[s]["programs"][k]["sim"]["s"] for s in suites if k in d[s]["programs"]), None)
    if s0 and s1 and s0 >= 0.3:
        rs.append(s0 / s1); out["sim_vs_sim"][k] = round(s0 / s1, 3)
        print("| %s | %.2f | %.2f | %.3f |" % (k, s0, s1, s0 / s1))
print("\n%d programs of 0.3 s or more: ratio from %.3f to %.3f; %d of them more than 5 %% from 1" % (len(rs), min(rs), max(rs), sum(1 for r in rs if abs(r - 1) > 0.05)))
print("\n### 3. Headline numbers of sp59, as published and with a form that used no card capped at 1.00\n")
print("| form | programs | geometric mean, published | corrected | summed sim s / form s, published | corrected |")
print("|---|---:|---:|---:|---:|---:|")
for f in forms:
    xs, cx, sim_sum, form_sum, csim_sum, changed = [], [], 0.0, 0.0, 0.0, []
    for s in suites:
        for k, v in d[s]["programs"].items():
            e = v.get(f, {})
            if e.get("x") is None:
                continue
            x, sim_s = e["x"], v["sim"]["s"]
            xs.append(x); sim_sum += sim_s; form_sum += e["s"]
            if e.get("card") == 0 and x > 1.05:
                changed.append("%s %.3f" % (k.split("/")[1], x)); cx.append(1.0); csim_sum += e["s"]
            else:
                cx.append(x); csim_sum += sim_s
    out["headline"][f] = {"n": len(xs), "geomean": round(gm(xs), 4), "geomean_corrected": round(gm(cx), 4), "sum_x": round(sim_sum / form_sum, 4),
                          "sum_x_corrected": round(csim_sum / form_sum, 4), "sim_sum": round(sim_sum, 1), "sim_sum_corrected": round(csim_sum, 1), "changed": changed}
    print("| %s | %d | %.3fx | %.3fx | %.0f / %.0f = %.3fx | %.0f / %.0f = %.3fx |" % (f, len(xs), gm(xs), gm(cx), sim_sum, form_sum, sim_sum / form_sum, csim_sum, form_sum, csim_sum / form_sum))
    print("|  | changed: %s | | | | |" % (", ".join(changed) or "none"))
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=0, sort_keys=True)
