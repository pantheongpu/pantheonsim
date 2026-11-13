#!/usr/bin/env python3
"""Round sp66: tables from census.py's JSON (one file per program).

  table.py DIR [DIR2] [--md out.md] [--json out.json]

For every launch the stand-in driver saw (each one ran in the simulator: the stand-in refuses every module), the reason the mode gave, weighted by the
simulator's seconds: "sent" (the mode sent it, the stand-in refused it: on a card it would have run there), the size rule, and every refusal. With DIR2
(a census of another build) the two are put side by side, per reason and per program."""
import argparse
import collections
import glob
import json
import os
import re


def group(reason):
    """-> (group, detail): what kept a launch, in the census's words."""
    if reason == "failed on the card":
        return "sent", ""
    if reason == "not asked":   # (a launch decided after a sample of its first blocks: the size rule's, the sampled path does not say its reason)
        return "size rule", ""
    if reason in ("size rule", "verify", "mode off"):
        return reason, ""
    if reason.startswith("class "):
        return "class: " + reason[6:], ""
    if reason.startswith("safety: "):
        r = reason[8:]
        m = re.match(r"module variables, device calls or device-side launches: module holds (.*); kernel reaches (.*)$", r)
        if m:   # the old rule: refused for the module
            held, reached = m.group(1), m.group(2)
            if reached != "none":
                return "safety: the kernel calls " + reached, "module holds " + held
            return "safety: the module holds " + held + " (the kernel calls none)", ""
        return "safety: " + r, ""
    return reason, ""


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        out[r["program"]] = r
    return out


def by_reason(progs):
    agg = collections.OrderedDict()
    for p, r in progs.items():
        for k in r["kept"]:
            g, _ = group(k["reason"])
            a = agg.setdefault(g, {"launches": 0, "sim_s": 0.0, "programs": set(), "kernels": 0})
            a["launches"] += k["launches"]
            a["sim_s"] += k["sim_s"]
            a["programs"].add(p)
            a["kernels"] += 1
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("--min-s", type=float, default=0.05, help="per-program rows: programs whose kept seconds outside 'sent' and the size rule reach this")
    a = ap.parse_args()
    runs = [load(d) for d in a.dirs]
    names = [os.path.basename(os.path.normpath(d)) for d in a.dirs]
    lines = []
    total = [sum(k["sim_s"] for r in pr.values() for k in r["kept"]) for pr in runs]
    lines.append("Programs: " + ", ".join("%s %d (%d timed out: %s)" % (n, len(pr), sum(r["timeout"] for r in pr.values()),
                                                                          ", ".join(p for p, r in pr.items() if r["timeout"]) or "none")
                                          for n, pr in zip(names, runs)))
    lines.append("")
    lines.append("Simulator seconds of all launches: " + ", ".join("%s %.1f s" % (n, t) for n, t in zip(names, total)))
    lines.append("")
    aggs = [by_reason(pr) for pr in runs]
    keys = []
    for ag in aggs:
        for k in ag:
            if k not in keys:
                keys.append(k)
    keys.sort(key=lambda k: -max(ag.get(k, {"sim_s": 0})["sim_s"] for ag in aggs))
    hdr = "| reason |" + "".join(" %s launches | %s sim s | %s programs |" % (n, n, n) for n in names)
    lines.append(hdr)
    lines.append("|---|" + "---:|---:|---:|" * len(names))
    for k in keys:
        row = "| %s |" % k
        for ag in aggs:
            v = ag.get(k)
            row += " %d | %.2f | %d |" % (v["launches"], v["sim_s"], len(v["programs"])) if v else " 0 | 0 | 0 |"
        lines.append(row)
    lines.append("")
    # per program: what is kept for a reason other than the size rule and "sent"
    lines.append("Per program, the simulator seconds kept for a reason other than the size rule (`sent`: would run on a card):")
    lines.append("")
    lines.append("| program | run | all sim s | sent s | size rule s | refused s | refused for |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    progs = sorted(set().union(*[set(pr) for pr in runs]))
    js = {"runs": names, "by_reason": [{k: {"launches": v["launches"], "sim_s": round(v["sim_s"], 6), "programs": sorted(v["programs"])} for k, v in ag.items()} for ag in aggs],
          "programs": {}}
    for p in progs:
        rows = []
        show = False
        for n, pr in zip(names, runs):
            r = pr.get(p)
            if not r:
                continue
            sums = collections.Counter()
            refused = collections.Counter()
            for k in r["kept"]:
                g, _ = group(k["reason"])
                sums[g if g in ("sent", "size rule") else "refused"] += k["sim_s"]
                if g not in ("sent", "size rule"):
                    refused[g] += k["sim_s"]
            allp = sum(sums.values())
            js["programs"].setdefault(p, {})[n] = {"all_s": round(allp, 6), "sent_s": round(sums["sent"], 6), "size_s": round(sums["size rule"], 6),
                                                   "refused": {g: round(s, 6) for g, s in refused.items()}, "timeout": r["timeout"], "rc": r["rc"]}
            show = show or sums["refused"] >= a.min_s
            why = "; ".join("%s %.2f" % (g, s) for g, s in refused.most_common(3))
            rows.append("| %s | %s | %.2f | %.2f | %.2f | %.2f | %s |" % (p, n, allp, sums["sent"], sums["size rule"], sums["refused"], why or ""))
        if show:
            lines.extend(rows)
    text = "\n".join(lines) + "\n"
    if a.md:
        open(a.md, "w").write(text)
    if a.json:
        json.dump(js, open(a.json, "w"), separators=(",", ":"))
    print(text)


if __name__ == "__main__":
    main()
