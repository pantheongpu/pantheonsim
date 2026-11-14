#!/usr/bin/env python3
"""Round sp65, question 2: what the card's kernel time is, against the simulator's time for the same launches.

  q2_card_time.py --bench ../sp59/bench.json --traces DIR [--sim-scale X] [--md]

Round sp59 recorded, per program and form, how many launches ran on the card (`card`) and the seconds of their card path (`card_s`: from the
decision to the end of the launch, the pushes of the Host chunks included; with residency nothing is pulled then). It did not record a launch's own
card time. The sp65 trace of the same program (trace_corpus.py, server3) has every launch's simulator time; residency_replay.py's base rule picks the
launches that go to the card, with the same rules sp59's build had (no sampler), at the scale of the rig's simulator against server3's that best
reproduces sp59's counts for the program (residency_replay.fit_scale). Per program this prints the card path per
launch, the pushes the replay says those launches made, what is left per launch once 7 us and the pushes (at round13's calibrated rate) are taken
off (the kernel and its synchronisation), and the ratio of the simulator's time for those launches (scaled to the rig) to that card time.

The pairing is by count, not launch by launch: when the replay sends as many launches as sp59's run did, the two sets are taken to be the same.
Rows where the counts differ by more than 2 % are marked and left out of the fit.
"""
import argparse, glob, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency_replay as rr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--traces", required=True)
    ap.add_argument("--sim-scale", type=float, default=1.0); ap.add_argument("--form", default="turbo")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    b = json.load(open(a.bench))
    rows = []
    for suite in ("polybench", "rodinia", "hecbench"):
        for prog, forms in b[suite]["programs"].items():
            r = forms.get(a.form)
            if not r or not r.get("card") or r.get("card_s") is None: continue
            name = prog.replace("/", "_")
            tr = os.path.join(a.traces, "%s.%s.trace" % (name, a.form))
            if not os.path.exists(tr): continue
            sc, _ = rr.fit_scale(tr, b, a.form)
            sc = sc or a.sim_scale
            rep = rr.run(tr, "base", 1e9, 0.0, sc, sample=False)   # (card kernel time ~0 in the replay: only the copies are counted)
            sent = [e for e in rep.events if e[1] == "card"]
            if not sent:
                rows.append((prog, r["card"], 0, r["card_s"], 0, 0, 0, 0, False, 0)); continue
            push = sum(e[5] for e in sent)
            sim = sum(e[3] for e in sent)
            path = r["card_s"] / r["card"]
            copies = rr.PSCALE * push / rr.PUSH_EFF / len(sent)
            kern = path - rr.LAUNCH - copies
            ok = abs(len(sent) - r["card"]) <= max(1, 0.02 * r["card"])
            rows.append((prog, r["card"], len(sent), r["card_s"], path, copies, kern, (sim / len(sent)) / kern if kern > 0 else float("inf"), ok, sim / len(sent)))
    head = "| program | sp59 card launches | replay card launches | card path s (sp59) | path per launch us | pushes per launch us (model) | kernel per launch us | sim per launch us (rig) | sim / kernel |"
    print(head); print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    good = []
    for p, n, m, cs, path, cp, k, ratio, ok, simpl in rows:
        print("| %s%s | %d | %d | %.4f | %.1f | %.1f | %.1f | %.0f | %.0f |" % (p, "" if ok else " (counts differ)", n, m, cs, path * 1e6, cp * 1e6, k * 1e6, simpl * 1e6, ratio))
        if ok and k > 0: good.append((p, k, ratio))
    if good:
        ks = sorted(k for _, k, _ in good); rs = sorted(r for _, _, r in good)
        print("\nprograms fitted: %d; kernel per launch: min %.1f us, median %.1f us; sim / kernel: min %.0f, median %.0f, max %.0f" %
              (len(good), ks[0] * 1e6, statistics.median(ks) * 1e6, rs[0], statistics.median(rs), rs[-1]))


if __name__ == "__main__":
    main()
