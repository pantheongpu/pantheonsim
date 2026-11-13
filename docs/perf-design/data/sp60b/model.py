#!/usr/bin/env python3
"""Round sp60b: reads what samplelog.py kept and prints, per sampled launch, what the sample said against what the launch took.

  model.py out.json [--form log]  [--md]

Columns: the static estimate (static_sim_seconds), the process's age at the launch, what the sample cost (fork to answer), the time of the copy's first
block and of the timed (second) runs of its blocks, the plain extrapolation from those (per-block time x blocks / host threads: the rule's estimate), what the simulator then took for the launch, and their ratio: under 1 the launch
was estimated low (towards the simulator), over 1 high.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--form", default="log")
    ap.add_argument("--slowdown", type=float, default=1.0)
    ap.add_argument("--min-rest", type=float, default=0.0, help="only launches that took at least this long")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    d = json.load(open(a.json))
    rows = []
    for prog, forms in sorted(d.items()):
        rec = forms.get(a.form)
        if not rec:
            continue
        for l in rec["launches"]:
            timed = l["ran"]
            if not l["answered"] or timed < 1 or l["launch_s"] <= 0 or l["timed_s"] <= 0:
                rows.append((prog, l["kernel"][:40], l["blocks"], l["live_bytes"] / 1048576.0, l["static_s"], l["age_s"], l["cost_s"] * 1e3, l["fork_s"] * 1e3, l["ran"],
                             l["first_block_s"] * 1e3, 0.0, 0.0, l["launch_s"], 0.0, l["launch_s"] / max(l["static_s"], 1e-9), 0.0, l["would_send"]))
                continue
            plain = l["timed_s"] / timed * l["blocks"] / max(l["threads"], 1)
            if l["launch_s"] < a.min_rest:
                continue
            rows.append((prog, l["kernel"][:40], l["blocks"], l["live_bytes"] / 1048576.0, l["static_s"], l["age_s"], l["cost_s"] * 1e3, l["fork_s"] * 1e3, l["ran"],
                         l["first_block_s"] * 1e3, l["timed_s"] * 1e3, plain, l["launch_s"], plain / l["launch_s"], l["launch_s"] / max(l["static_s"], 1e-9),
                         plain / a.slowdown, l["would_send"]))
    head = ["program", "kernel", "blocks", "live MiB", "static est s", "age s", "sample cost ms", "fork ms", "blocks timed", "first block ms", "timed runs ms",
            "plain extrapolation s", "launch took s", "ratio", "launch / static", "estimate s", "would send"]
    fmt = ["%s", "%s", "%d", "%.1f", "%.4f", "%.3f", "%.2f", "%.2f", "%d", "%.3f", "%.3f", "%.4f", "%.4f", "%.2f", "%.1f", "%.4f", "%d"]
    sep = " | " if a.md else "\t"
    if a.md:
        print("| " + sep.join(head) + " |")
        print("|" + "|".join(["---"] * 2 + ["---:"] * (len(head) - 2)) + "|")
    else:
        print(sep.join(head))
    for r in rows:
        line = sep.join(f % v for f, v in zip(fmt, r))
        print("| " + line + " |" if a.md else line)
    ratios = sorted(r[13] for r in rows if r[13] > 0)
    if ratios:
        n = len(ratios)
        print()
        print("launches %d; ratio min %.2f, median %.2f, 90th percentile %.2f, max %.2f; estimated high (ratio over %.1f): %d" %
              (n, ratios[0], ratios[n // 2], ratios[min(n - 1, int(n * 0.9))], ratios[-1], a.slowdown, sum(1 for x in ratios if x > a.slowdown)))


if __name__ == "__main__":
    main()
