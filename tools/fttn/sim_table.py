#!/usr/bin/env python3
"""Turns the lines tools/fttn/run-sim.sh prints (or docs/numerics/fttn/simulator-*.txt) into the markdown table of docs/numerics/fttn-simulators.md:
for each NVIDIA profile, the cell FTTN's summary row gives (subnormals / extra bits / rounding / block width) for the four formats and the
motivating example's D, on both engines, set beside the row of a card of the same compute capability (docs/numerics/fttn/cards-fttn.txt).

  sim_table.py <sim lines file> <cards-fttn.txt>

Our own script. Run it with python3 -I.
"""
import re, sys

LINE = re.compile(r"^(?:nvidia/(?P<p1>\S+) sm_(?P<s1>\d+) (?P<t1>\S+) (?P<e1>ptx|sass)|nvidia\.(?P<p2>[\w-]+)\.sm_(?P<s2>\d+)\.(?P<t2>[\w-]+)\.(?P<e2>ptx|sass)\.txt) \| (?P<row>.*?)(?: rc=\d+)?\s*$")
CARD = re.compile(r"^(?P<c>\S+) sm_(?P<s>\d+)-(?P<t>\S+) \| (?P<row>.*?)\s*$")
FORMATS = [("fp16in", "FP16"), ("bf16in", "BF16"), ("tf-fp32in", "TF32"), ("fp64in", "FP64")]
NO_SOURCE = {"b300", "vr200", "thor", "gb10"}   # compute capabilities 10.3, 10.7, 11.0, 12.1: no card, no paper, no recorded data
CARD_OF = {"75": "t4", "80": "a100", "86": "rtx3060", "89": "l4", "90": "h100", "120": "rtx-pro-6000-server"}

def cell(row, fmt):
    if row.startswith("D="):
        return row.split()[0][2:].rstrip("0").rstrip(".") or "0"
    f = row.split()
    if len(f) < 7 or "Error" in row:
        return "no run"
    sub, bits, rnd, width = ("yes" if f[-7].endswith("Yes") else "no"), f[-6], f[-5], f[-2]    # a long model name runs into "Yes"
    if fmt == "bf16in" and width == "10":      # FTTN prints the width in hex after a hex dump: 10 is 16
        width = "16"
    rnd = {"Trucate": "truncate", "RTN": "RTN", "N.A.": "-"}.get(rnd, rnd)
    return f"{sub} / {bits} / {rnd} / {width}"

def main():
    sim, cards = {}, {}
    for line in open(sys.argv[1]):
        m = LINE.match(line)
        if not m:
            continue
        g = lambda k: m.group(k + "1") or m.group(k + "2")
        sim.setdefault((g("p"), g("s")), {}).setdefault(g("t"), {})[g("e")] = m.group("row")
    for line in open(sys.argv[2]):
        m = CARD.match(line)
        if m:
            cards[(m.group("c"), m.group("s"), m.group("t"))] = m.group("row")
    print("| Profile | Cubin | FP16 | BF16 | TF32 | FP64 | Motivating D | Engines | Same as the card's row |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (prof, sm), tests in sorted(sim.items(), key=lambda kv: (int(kv[0][1]), kv[0][0])):
        ref = CARD_OF.get(sm)
        cols, agree, same, nref = [], True, True, 0
        for t, fmt in FORMATS + [("motivating", "D")]:
            e = tests.get(t)
            if not e:
                cols.append("-")
                continue
            cs = {k: cell(v, t) for k, v in e.items()}
            if len(set(cs.values())) > 1:
                agree = False
            cols.append(cs.get("sass") or cs.get("ptx"))
            cr = cards.get((ref, sm, t)) if ref else None
            if cr is not None:
                nref += 1
                if cell(cr, t) != cols[-1]:
                    same = False
        verdict = "no card, no published model (the sequential sum)" if prof in NO_SOURCE else "no card of ours of that capability" if not nref else ("yes" if same else "NO") + f" ({ref}, {nref} rows)"
        print(f"| {prof} | sm_{sm} | {' | '.join(cols)} | {'same' if agree else 'DIFFER'} | {verdict} |")

main()
