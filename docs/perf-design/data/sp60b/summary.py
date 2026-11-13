#!/usr/bin/env python3
"""Round sp60b: the A/B's summary from the harness records (tools/bench/hostgpu/run.py --json).

  summary.py ab-rodinia.json ab-polybench.json ab-hecbench.json [--md] [--compact out.json]

Per program: the simulator's seconds, each form's median seconds, exact against exact0 and turbo against turbo0 (over 1: the sampled form is
faster), the launches sampled and sent, the outputs. Then per suite and over all: the geometric means of the speed-up over the simulator of
each form, and of exact against exact0 and turbo against turbo0; the programs more than 3% slower than their unsampled form, with their three runs;
and what the fork checks counted (samples made after the driver's start, copies still running at exit, copies left).
"""
import json
import math
import sys


def geo(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    md = "--md" in sys.argv
    compact = sys.argv[sys.argv.index("--compact") + 1] if "--compact" in sys.argv else None
    if compact:
        files.remove(compact)
    rows, keep = [], {}
    checks = {"sampled": 0, "sent": 0, "timeouts": 0, "after_driver": 0, "copies_running_at_exit": 0, "copies_left": 0, "runs_with_line": 0}
    for f in files:
        suite = json.load(open(f))
        for prog, rec in suite["programs"].items():
            r = {"program": prog, "suite": prog.split("/")[0]}
            for form in ("sim", "exact", "exact0", "turbo", "turbo0"):
                d = rec.get(form) or {}
                r[form] = None if d.get("timeout") or "median_s" not in d else d["median_s"]
                r[form + "_wall"] = d.get("wall")
                r[form + "_out"] = (d.get("output") or {}).get("class")
                r[form + "_card"] = d.get("launches_card")
                r[form + "_launches"] = None if d.get("launches_card") is None else d["launches_card"] + (d.get("launches_sim") or 0)
                s = d.get("sampled")
                r[form + "_sampled"] = s
                if s:
                    checks["runs_with_line"] += 1
                    for k in checks:
                        if k in s:
                            checks[k] += s[k]
            rows.append(r)
            # compact: the medians, the launches on the card, and what was sampled (sampled, sent, copies that did not answer, cost in ms)
            k = {}
            for form in ("sim", "exact", "exact0", "turbo", "turbo0"):
                if r[form] is not None:
                    k[form] = round(r[form], 3)
                sm = r[form + "_sampled"]
                if sm and sm.get("sampled"):
                    k[form + "_sampled"] = [sm["sampled"], sm["sent"], sm["timeouts"], round(sm["cost_s"] * 1e3, 1)]
                if r[form + "_card"]:
                    k[form + "_card"] = r[form + "_card"]
                if r[form + "_out"] not in (None, "identical"):
                    k[form + "_out"] = r[form + "_out"]
            keep[prog] = k
    sep = " | " if md else "\t"
    head = ["program", "sim s", "exact s", "exact0 s", "exact0 / exact", "turbo s", "turbo0 s", "turbo0 / turbo", "sampled, sent (exact; turbo)",
            "on card of launches (exact; turbo)", "output (exact; turbo)"]
    print(("| " if md else "") + sep.join(head) + (" |" if md else ""))
    if md:
        print("|---|" + "---:|" * (len(head) - 1))

    def t(x):
        return "-" if x is None else "%.2f" % x

    def ratio(a, b):
        return None if not a or not b else b / a

    def smp(r, f):
        s = r[f + "_sampled"]
        return "-" if not s else "%d, %d" % (s.get("sampled", 0), s.get("sent", 0))

    for r in rows:
        re, rt = ratio(r["exact"], r["exact0"]), ratio(r["turbo"], r["turbo0"])
        cells = [r["program"], t(r["sim"]), t(r["exact"]), t(r["exact0"]), "-" if re is None else "%.2fx" % re, t(r["turbo"]), t(r["turbo0"]),
                 "-" if rt is None else "%.2fx" % rt, smp(r, "exact") + "; " + smp(r, "turbo"),
                 "%s of %s; %s of %s" % (r["exact_card"], r["exact_launches"], r["turbo_card"], r["turbo_launches"]),
                 "%s; %s" % (r["exact_out"], r["turbo_out"])]
        print(("| " if md else "") + sep.join(cells) + (" |" if md else ""))
    print()
    for suite in sorted(set(r["suite"] for r in rows)) + ["all"]:
        rs = [r for r in rows if suite == "all" or r["suite"] == suite]
        over = {f: geo([r["sim"] / r[f] for r in rs if r["sim"] and r[f]]) for f in ("exact", "exact0", "turbo", "turbo0")}
        n = {f: sum(1 for r in rs if r["sim"] and r[f]) for f in over}
        ab_e = [ratio(r["exact"], r["exact0"]) for r in rs]
        ab_t = [ratio(r["turbo"], r["turbo0"]) for r in rs]
        print("%s: %d programs. Over the simulator (geometric mean; programs with a simulator time): exact %.3fx (%d), exact0 %.3fx, turbo %.3fx (%d), turbo0 %.3fx. "
              "exact0 / exact %.3fx over %d; turbo0 / turbo %.3fx over %d. Summed seconds: exact %.1f, exact0 %.1f, turbo %.1f, turbo0 %.1f%s" % (
                  suite, len(rs), over["exact"], n["exact"], over["exact0"], over["turbo"], n["turbo"], over["turbo0"], geo(ab_e), sum(1 for x in ab_e if x),
                  geo(ab_t), sum(1 for x in ab_t if x),
                  sum(r["exact"] for r in rs if r["exact"] and r["exact0"]), sum(r["exact0"] for r in rs if r["exact"] and r["exact0"]),
                  sum(r["turbo"] for r in rs if r["turbo"] and r["turbo0"]), sum(r["turbo0"] for r in rs if r["turbo"] and r["turbo0"]),
                  "" if md else ""))
    print()
    print("More than 3% slower than the unsampled form (median), with the runs of both:")
    any_slow = False
    for r in rows:
        for f, g in (("exact", "exact0"), ("turbo", "turbo0")):
            x = ratio(r[f], r[g])
            if x is not None and x < 1 / 1.03:
                any_slow = True
                s = r[f + "_sampled"] or {}
                print("* %s %s: %.3f s against %.3f s (%+.1f%%); runs %s against %s; launches sampled %s, sample cost %s s" % (
                    r["program"], f, r[f], r[g], (r[f] / r[g] - 1) * 100, r[f + "_wall"], r[g + "_wall"], s.get("sampled", "-"), s.get("cost_s", "-")))
    if not any_slow:
        print("* none")
    print()
    print("Fork checks, summed over the %d runs whose summary line was kept: %s" % (checks["runs_with_line"], ", ".join("%s=%s" % (k, v) for k, v in checks.items() if k != "runs_with_line")))
    different = [(r["program"], f) for r in rows for f in ("exact", "exact0", "turbo", "turbo0") if r[f + "_out"] not in (None, "identical")]
    print("Outputs that are not identical to the simulator's: %s" % (different or "none"))
    if compact:
        json.dump(keep, open(compact, "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
