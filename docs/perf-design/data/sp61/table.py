#!/usr/bin/env python3
"""Round sp61: per sampled launch (the host_gpu_sample lines that ../sp60b/samplelog.py keeps), what the sample estimated against what the simulator
then took, for samples of whole blocks and samples inside the blocks alike.

  table.py out.json [--form tlog] [--md] [--inside-only]

Columns: program, kernel, blocks, inside (1: sampled inside its blocks), live MiB, static estimate, the process's age, what the sample cost (fork to
answer), warps followed / not, the turns left read, the steps of a turn, the estimate, what the launch took in the simulator, their ratio (under 1: low,
towards the simulator) and whether the rule would have sent it.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--form", default="tlog")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--inside-only", action="store_true")
    a = ap.parse_args()
    d = json.load(open(a.json))
    head = ["program", "kernel", "blocks", "inside", "live MiB", "static est s", "age s", "affordable", "cost ms", "warps followed/not", "turns left read",
            "steps a turn", "estimate s", "launch took s", "ratio", "would send"]
    rows = []
    for prog, forms in sorted(d.items()):
        rec = forms.get(a.form)
        if not rec:
            continue
        for l in rec["launches"]:
            ins = int(l.get("inside", 0))
            if a.inside_only and not ins:
                continue
            est, took = float(l["estimate_s"]), float(l["launch_s"])
            rows.append([prog, l["kernel"][:36], "%d" % l["blocks"], "%d" % ins, "%.1f" % (l["live_bytes"] / 1048576.0), "%.4f" % l["static_s"], "%.3f" % l["age_s"],
                         "%d" % l["affordable"], "%.1f" % (l["cost_s"] * 1e3), "%d/%d" % (l.get("followed", 0), l.get("unfollowed", 0)) if ins else "-",
                         "%.0f" % l.get("read_turns", 0) if ins else "-", "%.0f" % l.get("per_turn", 0) if ins else "-", "%.4f" % est,
                         "%.4f" % took if took > 0 else "sent", "%.2f" % (est / took) if took > 0 and est > 0 else "-", "%d" % l["would_send"]])
    sep = " | " if a.md else "\t"
    if a.md:
        print("| " + sep.join(head) + " |")
        print("|" + "|".join(["---"] * 2 + ["---:"] * (len(head) - 2)) + "|")
    else:
        print(sep.join(head))
    for r in rows:
        print("| " + sep.join(r) + " |" if a.md else sep.join(r))
    rat = sorted(float(r[14]) for r in rows if r[14] != "-")
    if rat:
        n = len(rat)
        print("\nlaunches %d; ratio min %.2f, median %.2f, max %.2f; over 2: %d; under 0.5: %d" % (n, rat[0], rat[n // 2], rat[-1], sum(x > 2 for x in rat), sum(x < 0.5 for x in rat)))


if __name__ == "__main__":
    main()
