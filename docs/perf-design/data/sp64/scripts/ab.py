#!/usr/bin/env python3
"""Round sp64's small A/B driver (server3).
  ab.py --work W --out o.json --progs corpus/name,... --side NAME=BUILD[:K=V;K=V...] ... [--threads 1|default] [--cpus 12-15] [--runs 5]
Each side is a build directory and environment. A side with VGPU_JIT set gets a cache of its own, emptied before every run unless the side sets
KEEPCACHE=1 (then it is emptied once, before the first run of the program; FILL=1 then fills it with one untimed VGPU_JIT_SYNC=1 run: warm). PRE=1 runs the program once untimed before every timed run, on the
emptied cache (round sp63's "second run"). Sides alternate (a round in order, the next reversed); medians are printed. Every run writes
VGPU_COUNTERS_FILE and VGPU_MEMO_FINAL_HASH and is compared with the first side's first run (output, exit status, counters, final hash)."""
import argparse, json, os, shutil, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sp63, real

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--progs", required=True)
    ap.add_argument("--side", action="append", required=True); ap.add_argument("--threads", default="1"); ap.add_argument("--cpus", default="")
    ap.add_argument("--runs", type=int, default=5); ap.add_argument("--long-secs", type=float, default=1.0); ap.add_argument("--long-runs", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300); ap.add_argument("--corp", default=os.path.join(HERE, "../corp"))
    a = ap.parse_args()
    a.envd = {}
    sides = []
    for s in a.side:
        name, rest = s.split("=", 1)
        build, _, envs = rest.partition(":")
        env = dict(x.split("=", 1) for x in envs.split(";") if x)
        sides.append((name, os.path.abspath(os.path.expanduser(build)), env))
    a.work = os.path.abspath(a.work); os.makedirs(a.work, exist_ok=True)
    progs = {p[0] + "/" + p[1]: p for p in real.manifest(os.path.abspath(a.corp), ["polybench", "rodinia", "hecbench"])}
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for pid in a.progs.split(","):
        if pid in res: continue
        prog = progs[pid]
        R = {n: [] for n, _, _ in sides}; P = {n: [] for n, _, _ in sides}; O = {n: [] for n, _, _ in sides}
        def one(n, build, env, i):
            e = {k: v for k, v in env.items() if k not in ("KEEPCACHE", "PRE", "FILL")}
            cache = os.path.join(a.work, "jc-" + n)
            a.build = build
            if "VGPU_JIT" in e:
                e["VGPU_JIT_CACHE"] = cache
                if env.get("KEEPCACHE") != "1" or i == 0: sp63.fresh(cache)
                if env.get("FILL") == "1" and i == 0: P[n].append(sp63.run1(a, prog, dict(e, VGPU_JIT_SYNC="1"), a.work))   # (warm: filled by one untimed synchronous run)
            if env.get("PRE") == "1": P[n].append(sp63.run1(a, prog, e, a.work))
            R[n].append(sp63.run1(a, prog, e, a.work))
            if "VGPU_JIT" in e: O[n].append(sp63.objs(cache))
        first = None
        n_runs = a.runs
        for i in range(a.runs):
            order = sides if i % 2 == 0 else sides[::-1]
            for n, b, env in order: one(n, b, env, i)
            if i == 0:
                w = statistics.median(x["wall"] for v in R.values() for x in v)
                if w > a.long_secs: n_runs = a.long_runs
            if i + 1 >= n_runs: break
        ref = R[sides[0][0]][0]
        allr = [x for v in R.values() for x in v] + [x for v in P.values() for x in v]
        same = all((x["out"], x["cnt"], x["fin"], x["rc"]) == (ref["out"], ref["cnt"], ref["fin"], ref["rc"]) for x in allr)
        diff = sorted(set(n for n in R for x in R[n] + P[n] if (x["out"], x["cnt"], x["fin"], x["rc"]) != (ref["out"], ref["cnt"], ref["fin"], ref["rc"])))
        res[pid] = dict(sides=R, pre=P, objs=O, same=same, differ=diff)
        json.dump(res, open(a.out + ".tmp", "w"), separators=(",", ":")); os.replace(a.out + ".tmp", a.out)
        med = {n: statistics.median(x["wall"] for x in R[n]) for n in R}
        b0 = med[sides[0][0]]
        print("%-32s %s same=%s%s objs=%s" % (pid, " ".join("%s=%.3f(%.3f)" % (n, med[n], b0 / med[n]) for n in R), same, ("" if same else " differ=%s" % diff),
              {n: v for n, v in O.items() if v}), flush=True)
        for n in R:
            if "VGPU_JIT" in sides[[s[0] for s in sides].index(n)][2]: shutil.rmtree(os.path.join(a.work, "jc-" + n), ignore_errors=True)
    sp63.SLOT.release()
    print("lock: %d holds, %.0f s waited" % (sp63.SLOT.holds, sp63.SLOT.waited), flush=True)

if __name__ == "__main__": main()
