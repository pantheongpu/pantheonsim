#!/usr/bin/env python3
"""Round sp75: the tables of census.py's records (one JSON a program, forms old and new).

  table.py <census dir> [--old old --new new] > census.md
"""
import argparse, glob, json, os


def best(runs, key, default=0):
    return [r.get(key, default) for r in runs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--old", default="old")
    ap.add_argument("--new", default="new")
    ap.add_argument("--small-rows", type=int, default=24)
    ap.add_argument("--only-decisions", action="store_true", help="the first two tables only")
    a = ap.parse_args()
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.dir, "*.json")))]
    changed, slower, small, lost, reasons = [], [], [], [], {}
    n_timeout = 0
    for r in recs:
        o, n = r["forms"][a.old][0], r["forms"][a.new][0]
        if o["timeout"] or n["timeout"]:
            n_timeout += 1
        sig = lambda d: (d.get("launches_sim"), d.get("too_small"), d.get("failed"), d.get("sampled"), d.get("sample_sent"), d.get("sample_timeouts"), len(d["starts"]))
        # (a run cut at the time limit prints no summary: its decisions are the lines it logged before)
        live = lambda d: (len(d["starts"]), sum(1 for s in d["sampled_lines"] if s.endswith(": sent")), sum(1 for s in d["sampled_lines"] if s.endswith(": kept")))
        if (sig(o) != sig(n)) if not (o["timeout"] or n["timeout"]) else (live(o) != live(n)):
            changed.append((r["program"], o, n))
        if not (o["timeout"] or n["timeout"]) and o["wall_s"] > 0 and n["wall_s"] > 1.03 * o["wall_s"]:
            slower.append((r["program"], o["wall_s"], n["wall_s"]))
        if o.get("sample_timeouts"):
            lost.append((r["program"], o.get("sample_timeouts"), n.get("sample_timeouts"), [s for s in n["sampled_lines"] if "stopped at the limit" in s]))
        for ns in n["nosample"]:
            if "followed" in ns and int(ns["blocks"]) < 256 and int(ns["blocks"]) >= 2 and ns["loop"] == "1" and ns["followed"] == "0":
                small.append((r["program"], ns["kernel"], int(ns["blocks"]), int(ns["instrs"]), float(ns["static_s"]), ns["reason"]))
        for s in n["sampled_lines"]:
            if " ran=" in s and int(r["forms"][a.new][0].get("sampled", 0) or 0) >= 0 and "nosample" not in s:
                pass
        for k in n["kept"]:
            e = reasons.setdefault(k["reason"], [0, 0.0, set()])
            e[0] += k["launches"]
            e[1] += k["sim_s"]
            e[2].add(r["program"])
    print("# Round sp75: the census (server3, the stand-in driver, turbo mode)\n")
    print("Every program once in each form, `VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1`, profile `nvidia/a10g`, `VGPU_THREADS=3`, against the stand-in driver,")
    print("which refuses every module: a launch the mode sends \"fails on the card\" and the simulator runs it. `old`: commit 6cf503d3 (the rules before the")
    print("round). `new`: this round's. A run is cut at the time limit; a cut run has no summary, and its decisions are the lines it logged before the cut.\n")
    print("Programs: %d (%d cut at the time limit in a form).\n" % (len(recs), n_timeout))
    print("## Programs whose decisions differ between old and new\n")
    print("| program | old: kept / too small / sent / sampled / sent on a sample / copies that did not answer | new | new: the sampled launches |")
    print("|---|---|---|---|")
    f = lambda d: "%s / %s / %s / %s / %s / %s%s" % (d.get("launches_sim", "-"), d.get("too_small", "-"), d.get("failed", "-"), d.get("sampled", "-"), d.get("sample_sent", "-"),
                                                     d.get("sample_timeouts", "-"), " (cut)" if d["timeout"] else "")
    for p, o, n in changed:
        print("| %s | %s | %s | %s |" % (p, f(o), f(n), "; ".join(s.split(" ", 1)[1][:150] for s in n["sampled_lines"])))
    if not changed:
        print("| none | | | |")
    print("\n## Samples whose copy did not answer under the old rule (it was ended at the wait's end, or the launch threw in it)\n")
    print("| program | old: copies that did not answer | new | new: blocks stopped at the limit |")
    print("|---|---:|---:|---|")
    for p, a_, b_, s in lost:
        print("| %s | %s | %s | %s |" % (p, a_, b_ if b_ is not None else "- (cut)", "; ".join(x.split(" ", 1)[1][:140] for x in s)))
    if not lost:
        print("| none | | | |")
    if a.only_decisions:
        return
    print("\n## Launches on a grid of 2 to 255 blocks of a kernel with a loop and no followed loop (new: the candidates of the bounded sample of a small grid)\n")
    print("| program | kernel | blocks | instructions | static estimate ms | new |")
    print("|---|---|---:|---:|---:|---|")
    seen = set()
    for p, k, b, i, s, why in sorted(small, key=lambda x: -x[4]):
        if (p, k) in seen:
            continue
        seen.add((p, k))
        if len(seen) <= a.small_rows:
            print("| %s | `%s` | %d | %d | %.2f | not sampled: %s |" % (p, k[:48], b, i, s * 1e3, why))
    if len(seen) > a.small_rows:
        print("\n(%d kernels in all; the %d with the largest static estimate are listed.)" % (len(seen), a.small_rows))
    for r in recs:
        n = r["forms"][a.new][0]
        for s in n["sampled_lines"]:
            pass
    print("\n(A small grid that IS sampled has no `host_gpu_nosample` line: it is in the first table.)\n")
    print("## Wall time, new against old, over 3 % slower in this one run (the stand-in driver: every launch is simulated in both; reruns in README.md)\n")
    print("| program | old s | new s | |")
    print("|---|---:|---:|---:|")
    for p, o, n in sorted(slower, key=lambda x: -x[2] / x[1]):
        print("| %s | %.3f | %.3f | %+.1f %% |" % (p, o, n, 100 * (n / o - 1)))
    if not slower:
        print("| none | | | |")
    tot_o = sum(r["forms"][a.old][0]["wall_s"] for r in recs if not r["forms"][a.old][0]["timeout"] and not r["forms"][a.new][0]["timeout"])
    tot_n = sum(r["forms"][a.new][0]["wall_s"] for r in recs if not r["forms"][a.old][0]["timeout"] and not r["forms"][a.new][0]["timeout"])
    print("\nSum over the programs not cut: old %.1f s, new %.1f s.\n" % (tot_o, tot_n))
    print("## What keeps a launch in the simulator (new; the reasons are this round's `host_gpu_kept` texts; programs not cut)\n")
    print("| reason | launches | sim s | programs |")
    print("|---|---:|---:|---:|")
    for why, (l, s, ps) in sorted(reasons.items(), key=lambda x: -x[1][1]):
        print("| %s | %d | %.2f | %d |" % (why, l, s, len(ps)))


if __name__ == "__main__":
    main()
