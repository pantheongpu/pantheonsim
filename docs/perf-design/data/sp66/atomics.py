#!/usr/bin/env python3
"""Round sp66: the hang risk of turbo mode's float atomics with a value, compare-and-swap and exchange, now admitted by default.

For each program one run with VGPU_HOST_GPU=turbo VGPU_HOST_GPU_VERIFY=all VGPU_HOST_GPU_LOG=1: every launch the card can run runs there and then in
the simulator, so each launch has both times: the card's kernel time ("on the card (verify): kernel N ms") and the simulator's (the host_gpu_kept
lines, reason verify). The kernels of the classes in question are the ones whose "classes:" line names atomic-float or atomic-swap. A loop of
compare-and-swap that spun on the card would show as a card time near or over the simulator's, or as a run the watchdog ended (status 124). Then one
plain run with the defaults (`card`): its status and whether the watchdog spoke.

  atomics.py --build BUILD --corp CORP --out DIR [--programs aop-cuda,atomicPerf-cuda,bh-cuda] [--gpu nvidia/a10g]   # runs, then the table
  atomics.py --out DIR --table-only
"""
import argparse
import collections
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools", "bench", "real"))

KEEP = re.compile(r"^host_gpu(_kept|_refused|_turbo|_refusals|: .*(on the card|failed|classes:))")
ON_CARD = re.compile(r"^host_gpu: (\S+) on the card \(verify\): kernel ([0-9.]+) ms")
KEPT = re.compile(r"^host_gpu_kept: kernel=(\S+) launches=(\d+) sim_s=(\S+) reason=(.*)$")
CLASSES = re.compile(r"^host_gpu: (\S+): classes: ([^;]*);")


def run(a):
    import real  # noqa: E402
    progs = [p for p in real.manifest(a.corp, ["hecbench"]) if p[1] in a.programs.split(",")]
    for corpus, name, cwd, argv, copy_cwd in progs:
        for form, extra in (("verify", {"VGPU_HOST_GPU_VERIFY": "all", "VGPU_HOST_GPU_LOG": "1"}), ("card", {})):
            env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
            env.update(VGPU_GPU=a.gpu, LD_LIBRARY_PATH=os.path.join(a.build, "shim"), VGPU_HOST_GPU="turbo", **extra)
            p = subprocess.run(["timeout", "900"] + argv, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            text = p.stderr.decode("utf-8", "replace")
            lines = [l[:400] for l in text.splitlines() if KEEP.match(l)]
            with open(os.path.join(a.out, "atomics-%s-%s.txt" % (name, form)), "w") as f:
                f.write("rc %d%s\n" % (p.returncode, " watchdog" if "VGPU_HOST_GPU_TIMEOUT" in text else "") + "\n".join(lines) + "\n")


def table(a):
    rows = []
    for fn in sorted(os.listdir(a.out)):
        m = re.match(r"atomics-(.*)-verify\.txt$", fn)
        if not m:
            continue
        prog = m.group(1)
        ver = open(os.path.join(a.out, fn)).read().splitlines()
        cardp = os.path.join(a.out, "atomics-%s-card.txt" % prog)
        card_rc = open(cardp).readline().strip() if os.path.exists(cardp) else "?"
        cls = {}
        for l in ver:
            c = CLASSES.match(l)
            if c and ("atomic-float" in c.group(2) or "atomic-swap" in c.group(2)):
                cls[c.group(1)] = " ".join(x for x in c.group(2).split() if x.startswith("atomic"))
        sim_t = collections.defaultdict(lambda: [0, 0.0])
        for l in ver:
            k = KEPT.match(l)
            if k and k.group(1) in cls:
                sim_t[k.group(1)][0] += int(k.group(2))
                sim_t[k.group(1)][1] += float(k.group(3))
        card_t = collections.defaultdict(list)
        for l in ver:
            c = ON_CARD.match(l)
            if c:
                card_t[c.group(1)].append(float(c.group(2)) / 1e3)
        for k in sorted(cls):
            n_sim, s_sim = sim_t[k]
            ts = card_t.get(k, [])
            per_sim = s_sim / n_sim if n_sim else float("nan")
            worst = max(ts) if ts else float("nan")
            rows.append("| %s | %s | %s | %d | %.6f | %d | %.6f | %.6f | %s | %s / %s |" % (
                prog, k[:56], cls[k], n_sim, per_sim, len(ts), (sum(ts) / len(ts)) if ts else float("nan"), worst,
                "%.4f" % (worst / per_sim) if ts and n_sim else "-", ver[0] if ver else "?", card_rc))
    out = ["| program | kernel | classes | sim launches | sim s a launch | card launches | card kernel s a launch (mean) | (largest) | largest / sim | runs: verify / default |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---|"] + rows
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build")
    ap.add_argument("--corp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--programs", default="aop-cuda,atomicPerf-cuda,bh-cuda")
    ap.add_argument("--gpu", default="nvidia/a10g")
    ap.add_argument("--table-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.table_only:
        run(a)
    table(a)


if __name__ == "__main__":
    main()
