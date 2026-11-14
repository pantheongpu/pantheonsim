#!/usr/bin/env python3
"""Round sp65: the card A/B from the harness's JSON files. Per program and mode: the median seconds with the rule (exact, turbo, exactdiv) and without
it (exactn, turbon, exactdiv with VGPU_HOST_GPU_STREAK=0), their ratio, the launches the rule sent (host_gpu_streak), the outputs; the programs more
than 3 % slower; and a compact JSON.

  summary.py [--md] [--json out.json] [--div0 target-div0.json] file.json...
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+"); ap.add_argument("--md", action="store_true"); ap.add_argument("--json", default=""); ap.add_argument("--div0", default="")
ap.add_argument("--short", action="store_true", help="only the rows where the rule sent a launch, or that are more than 3 % apart; a count of the others")
a = ap.parse_args()
progs = {}
def norm(x):
    return {"s": x.get("median_s"), "card": x.get("launches_card"), "streak": x.get("streak"), "out": (x.get("output") or {}).get("class"),
            "mib": x.get("moved_mib")}


for f in a.files:
    for p, r in json.load(open(f))["programs"].items():
        progs.setdefault(p, {}).update({k: norm(v) for k, v in r.items() if isinstance(v, dict)})
if a.div0:
    for p, r in json.load(open(a.div0))["programs"].items():
        if "exactdiv" in r: progs.setdefault(p, {})["exactdiv0"] = norm(r["exactdiv"])
pairs = [("exact", "exactn"), ("turbo", "turbon"), ("exactdiv", "exactdiv0")]
rows, slow, comp = [], [], {}
for p in sorted(progs):
    r = progs[p]
    c = {"sim": r.get("sim", {}).get("s")}
    for new, old in pairs:
        if new in r and old in r and r[new].get("s") and r[old].get("s"):
            ratio = r[new]["s"] / r[old]["s"]
            st = (r[new].get("streak") or {}).get("sent")
            outs = (r[new].get("out"), r[old].get("out"))
            rows.append((p, new, r[old]["s"], r[new]["s"], ratio, st, r[new].get("card"), r[old].get("card"), outs))
            c[new] = [r[new]["s"], r[old]["s"], st, r[new].get("card"), r[old].get("card")]
            if ratio > 1.03: slow.append((p, new, ratio))
            if new.startswith("exact") and outs[0] != "identical": c[new + "_out"] = outs[0]
    comp[p] = c
sep = " | "
print("| program | mode | without the rule s | with it s | with / without | sent by the rule | card launches with / without | outputs with / without |")
print("|---|---|---:|---:|---:|---:|---:|---|")
hidden = 0
for p, m, o, n, ratio, st, cn, co, outs in rows:
    if a.short and not st and 0.97 <= ratio <= 1.03 and outs[0] == "identical":
        hidden += 1; continue
    print("| %s | %s | %.3f | %.3f | %.3f | %s | %s / %s | %s / %s |" % (p, m, o, n, ratio, "-" if st is None else st, cn, co, outs[0], outs[1]))
if a.short: print("\n%d more rows: nothing sent by the rule, within 3 %%, outputs identical (all rows: --json)" % hidden)
print()
print("more than 3 %% slower with the rule: %s" % (", ".join("%s %s %.3f" % s for s in slow) or "none"))
if a.json: json.dump(comp, open(a.json, "w"), separators=(",", ":"), sort_keys=True)
