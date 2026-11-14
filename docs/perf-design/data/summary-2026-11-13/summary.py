#!/usr/bin/env python3
"""The summary of the host-GPU benchmark of 2026-11-13: reads the harness JSON of each pass
(tools/bench/hostgpu/run.py), writes bench.json (the same records, compacted: no stdout, no
refused-kernel lists) and summary.md (per suite and overall: geometric-mean speed-up over the
simulator, total wall time per form, gains and losses, the top programs, outputs, timeouts).

  summary.py --out DIR polybench.json rodinia.json hecbench.json samples.json hecheavy.json ..."""
import argparse
import json
import math
import os

FORMS = ("exact", "exacti", "turbo")
NAMES = {"sim": "simulator", "exact": "exact", "exacti": "exact + idioms", "turbo": "turbo"}
SUITES = ("polybench", "rodinia", "hecbench", "samples")
# Programs whose output carries timings the harness's filter does not catch (read from their sources): a "different" of theirs is not a result.
TIMED_OUTPUT = {
    "hecbench/babelstream-cuda": "the bandwidth table's rows (MBytes/sec, min, max, average seconds) are timings; only one simulator run (> 300 s), so no repeat check",
    "hecbench/base64e-cuda": "it repeats the whole test when the measured overlap of two streams is not acceptable (main.cu, RETRIES_ON_FAILURE): the number of lines depends on timing",
}


def compact(rec):
    """One program's record, small: times rounded to 10 ms, the exit status only when not 0, the output's detail only when not identical."""
    out = {}
    for f, d in rec.items():
        c = {"median_s": round(d["median_s"], 3), "wall": [round(x, 2) for x in d["wall"]]}
        for k in ("speedup", "launches_card", "launches_total"):
            if d.get(k) is not None:
                c[k] = d[k]
        if d.get("timeout"):
            c["timeout"] = True
        if any(r != 0 for r in d.get("rc", [])):
            c["rc"] = d["rc"]
        if d.get("output"):
            o = d["output"]
            c["output"] = o["class"] if o["class"] == "identical" else {"class": o["class"], "max_abs": o.get("max_abs"), "max_rel": round(o.get("max_rel") or 0, 4),
                                                                         "detail": o.get("detail", "")}
        if d.get("repeats") is not None:
            c["repeats"] = d["repeats"]
        out[f] = c
    return out


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def failed(d):
    return (not d.get("timeout")) and any(rc != 0 for rc in d.get("rc", []))


def t(d, timeout):
    if d is None:
        return "-"
    if d.get("timeout"):
        return "> %d" % timeout
    s = "%.2f" % d["median_s"]
    return s + (" rc%s" % d["rc"][-1] if failed(d) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("jsons", nargs="+")
    a = ap.parse_args()
    progs, meta = {}, {}
    for j in a.jsons:
        d = json.load(open(j))
        meta[os.path.basename(j)] = d["meta"]
        for k, rec in d["programs"].items():
            progs[k] = compact(rec)
    timeout = int(max(m["timeout"] for m in meta.values()))
    json.dump({"meta": meta, "programs": progs}, open(os.path.join(a.out, "bench.json"), "w"), separators=(",", ":"), sort_keys=True)

    L = []
    w = L.append
    groups = [(s, [k for k in progs if k.startswith(s + "/")]) for s in SUITES]
    groups = [g for g in groups if g[1]] + [("all", list(progs))]
    w("# Host-GPU modes against the simulator, one A10G (2026-11-13)\n")
    w("Speed-up = simulator wall time / mode wall time, median of the runs (README.md has the method). Geometric means are over the programs "
      "where the simulator and that mode both finished; programs whose simulator run passed %d s are counted separately (a lower bound).\n" % timeout)
    w("## Headline\n")
    w("| suite | programs | geomean exact | geomean exact + idioms | geomean turbo | total s simulator | total s exact | total s exact + idioms | total s turbo |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, keys in groups:
        ok = [k for k in keys if not progs[k]["sim"].get("timeout")]
        gm = []
        for f in FORMS:
            xs = [progs[k][f]["speedup"] for k in ok if progs[k].get(f, {}).get("speedup")]
            gm.append("%.2fx (%d)" % (gmean(xs), len(xs)) if xs else "-")
        tot = []
        for f in ("sim",) + FORMS:
            s = sum(min(progs[k][f]["median_s"], timeout) if not progs[k][f].get("timeout") else timeout for k in keys if f in progs[k])
            n_to = sum(1 for k in keys if progs[k].get(f, {}).get("timeout"))
            tot.append("%.0f%s" % (s, " (%d at > %d)" % (n_to, timeout) if n_to else ""))
        w("| %s | %d | %s | %s |" % (name, len(keys), " | ".join(gm), " | ".join(tot)))
    w("\nTotals sum the median wall time of every program in the suite; a timed-out run counts as %d s.\n" % timeout)
    w("The same geometric means over the programs whose simulator run took 1 s or more (the rest mostly time process start-up):\n")
    w("| suite | programs | geomean exact | geomean exact + idioms | geomean turbo |")
    w("|---|---:|---:|---:|---:|")
    for name, keys in groups:
        ok = [k for k in keys if not progs[k]["sim"].get("timeout") and progs[k]["sim"]["median_s"] >= 1]
        gm = []
        for f in FORMS:
            xs = [progs[k][f]["speedup"] for k in ok if progs[k].get(f, {}).get("speedup")]
            gm.append("%.2fx (%d)" % (gmean(xs), len(xs)) if xs else "-")
        w("| %s | %d | %s |" % (name, len(ok), " | ".join(gm)))
    w("")

    w("## Gains and losses (programs where the simulator finished)\n")
    w("| suite | mode | > 2x faster | 1.03x to 2x | within 3 % | > 3 % slower | of those, simulator run >= 0.5 s |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for name, keys in groups:
        for f in FORMS:
            ks = [k for k in keys if not progs[k]["sim"].get("timeout") and progs[k].get(f, {}).get("speedup")]
            xs = [progs[k][f]["speedup"] for k in ks]
            w("| %s | %s | %d | %d | %d | %d | %d |" % (name, NAMES[f], sum(x > 2 for x in xs), sum(1.03 <= x <= 2 for x in xs),
                                                   sum(1 / 1.03 < x < 1.03 for x in xs), sum(x <= 1 / 1.03 for x in xs),
                                                   sum(progs[k][f]["speedup"] <= 1 / 1.03 and progs[k]["sim"]["median_s"] >= 0.5 for k in ks)))
    w("\nA program that runs for a few hundredths of a second times process start-up: most losses are of those (listed below).\n")
    slow = [(k, f, progs[k][f]["speedup"]) for k in progs for f in FORMS if progs[k].get(f, {}).get("speedup") and progs[k][f]["speedup"] <= 1 / 1.03]
    if slow:
        w("Slower by more than 3 %: " + "; ".join("%s %s %.2fx (%.2f s against %.2f s)" % (k, NAMES[f], x, progs[k][f]["median_s"], progs[k]["sim"]["median_s"])
                                                 for k, f, x in sorted(slow)) + ".\n")

    w("## Top 15 by turbo speed-up\n")
    w("| program | runs | simulator s | exact s | x | exact + idioms s | x | turbo s | x | turbo launches on card |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    top = sorted((k for k in progs if progs[k].get("turbo", {}).get("speedup")), key=lambda k: -progs[k]["turbo"]["speedup"])[:15]
    for k in top:
        p = progs[k]
        cells = []
        for f in FORMS:
            cells += [t(p.get(f), timeout), "%.2fx" % p[f]["speedup"] if p.get(f, {}).get("speedup") else "-"]
        lc = p["turbo"].get("launches_card")
        w("| %s | %d | %s | %s | %s |" % (k, len(p["sim"]["wall"]), t(p["sim"], timeout), " | ".join(cells),
                                          "%s / %s" % (lc, p["turbo"].get("launches_total")) if lc is not None else "-"))
    w("")

    tos = [k for k in progs if progs[k]["sim"].get("timeout")]
    if tos:
        w("## Simulator past %d s (speed-up is a lower bound)\n" % timeout)
        w("| program | exact s | exact + idioms s | turbo s | turbo at least |")
        w("|---|---:|---:|---:|---:|")
        for k in sorted(tos):
            p = progs[k]
            lb = ("%.0fx" % (timeout / p["turbo"]["median_s"])) if not p["turbo"].get("timeout") else "-"
            w("| %s | %s | %s | %s | %s |" % (k, t(p.get("exact"), timeout), t(p.get("exacti"), timeout), t(p.get("turbo"), timeout), lb))
        w("")

    w("## Outputs against the simulator's first run\n")
    w("| mode | identical | within tolerance | different | different, timing-dependent output (the simulator's own runs differ, or the output carries timings) | not compared |")
    w("|---|---:|---:|---:|---:|---:|")
    for f in FORMS:
        c = {"identical": 0, "numeric": 0, "different": 0, "noisy": 0, "none": 0}
        for k in progs:
            o = progs[k].get(f, {}).get("output")
            cl = (o if isinstance(o, str) else o["class"]) if o else "none"
            if cl == "different" and (progs[k]["sim"].get("repeats") is False or k in TIMED_OUTPUT):
                cl = "noisy"
            c[cl] += 1
        w("| %s | %d | %d | %d | %d | %d |" % (NAMES[f], c["identical"], c["numeric"], c["different"], c["noisy"], c["none"]))
    w("")
    odd = [(k, f, progs[k][f]["output"]) for k in sorted(progs) for f in FORMS
           if progs[k].get(f, {}).get("output") and progs[k][f]["output"] != "identical"]
    for k, f, o in odd:
        why = TIMED_OUTPUT.get(k) or ("the simulator's own runs differ too" if progs[k]["sim"].get("repeats") is False else "")
        w("- %s, %s: %s (%s; largest abs %.3g, rel %.3g)%s" % (k, NAMES[f], o["class"], o["detail"] or "numbers only", o["max_abs"] or 0, o["max_rel"] or 0,
                                                            ": " + why if why else ""))
    nrep = [k for k in progs if progs[k]["sim"].get("repeats") is False]
    if nrep:
        w("- the simulator does not repeat itself run to run: " + ", ".join(sorted(nrep)))
    w("")
    bad = [(k, f) for k in sorted(progs) for f in ("sim",) + FORMS if f in progs[k] and failed(progs[k][f])]
    if bad:
        w("Non-zero exit status: " + ", ".join("%s %s (rc %s)" % (k, NAMES[f], progs[k][f]["rc"]) for k, f in bad) + ".\n")

    w("## Every program\n")
    w("| program | runs | simulator s | exact s | x | exact + idioms s | x | turbo s | x | output exact / idioms / turbo |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    ab = {"identical": "=", "numeric": "~", "different": "DIFF"}
    for k in sorted(progs, key=lambda k: (SUITES.index(k.split("/")[0]), k.lower())):
        p = progs[k]
        cells = []
        for f in FORMS:
            cells += [t(p.get(f), timeout), "%.2fx" % p[f]["speedup"] if p.get(f, {}).get("speedup") else "-"]
        oc = lambda o: o if isinstance(o, str) else (o or {}).get("class")
        outs = " / ".join(ab.get(oc(p.get(f, {}).get("output")), "-") for f in FORMS)
        w("| %s | %d | %s | %s | %s |" % (k, len(p["sim"]["wall"]), t(p["sim"], timeout), " | ".join(cells), outs))
    w("\n`=` identical, `~` within tolerance (atol 1e-8 + rtol 1e-5), `DIFF` different, `-` not compared (the simulator timed out).")
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
