#!/usr/bin/env python3
"""Round sp67 (#570): runs one program with VGPU_HOST_GPU_LOG=1, stamps every stderr line with the seconds since the process started, and prints a
compact summary: per kernel, the launches that stayed (and why) and the ones that went to the card, with their pushed MiB, kernel and card-path times;
the time of the first line that says a launch went to the card; and the mode's summary lines. The raw stamped log goes to --raw.

  logged_run.py --raw out.log [--env K=V ...] -- <program> [args]

The environment of the caller is kept (VGPU_GPU, LD_LIBRARY_PATH, VGPU_HOST_GPU are expected there or in --env).
"""
import argparse
import collections
import os
import re
import subprocess
import sys
import time

CARD = re.compile(r"host_gpu: (\S+) on the card(?: \(resident\))?: kernel ([\d.]+) ms, total ([\d.]+) ms \(sim estimate ([\d.]+) ms\), ([\d.]+) MiB")
STAY = re.compile(r"host_gpu: (\S+) stays in the simulator(?::| \()(.*)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("cmd", nargs="+")
    a = ap.parse_args()
    env = dict(os.environ)
    env.update(dict(kv.split("=", 1) for kv in a.env))
    env["VGPU_HOST_GPU_LOG"] = env.get("VGPU_HOST_GPU_LOG", "1")
    t0 = time.monotonic()
    p = subprocess.Popen(a.cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    lines = []
    for line in p.stderr:
        lines.append((time.monotonic() - t0, line.rstrip("\n")))
    out = p.stdout.read()
    rc = p.wait()
    wall = time.monotonic() - t0
    with open(a.raw, "w") as f:
        for t, l in lines:
            f.write(f"{t:10.4f} {l}\n")
        f.write(f"# rc={rc} wall={wall:.3f}\n# stdout:\n{out}")
    card = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])   # launches, kernel ms, total ms, estimate ms, MiB pushed
    stay = collections.defaultdict(collections.Counter)
    first_card = None
    last_before_card = None
    for t, l in lines:
        m = CARD.search(l)
        if m:
            k = card[m.group(1)]
            k[0] += 1
            for i in range(4):
                k[i + 1] += float(m.group(i + 2))
            if first_card is None:
                first_card = (t, l)
            continue
        m = STAY.search(l)
        if m:
            why = re.sub(r"[\d.]+ ms", "N ms", m.group(2)).strip()
            why = re.sub(r"[\d.]+ MiB", "N MiB", why)
            stay[m.group(1)][why] += 1
        if first_card is None and l.startswith("host_gpu"):
            last_before_card = (t, l)
    print(f"rc={rc} wall={wall:.3f}s stderr_lines={len(lines)}")
    if first_card:
        print(f"first card line at {first_card[0]:.4f}s: {first_card[1][:160]}")
    if last_before_card:
        print(f"last host_gpu line before it at {last_before_card[0]:.4f}s: {last_before_card[1][:160]}")
    for name in sorted(set(card) | set(stay)):
        c = card.get(name)
        if c:
            print(f"CARD {name}: {c[0]} launches, kernel {c[1]:.1f} ms, card path {c[2]:.1f} ms, sim estimate {c[3]:.1f} ms, pushed {c[4]:.1f} MiB")
        for why, n in stay.get(name, {}).items():
            print(f"STAY {name}: {n} x {why[:140]}")
    for t, l in lines:
        if l.startswith(("host_gpu_", "host_gpu: verify:")) or re.match(r"host_gpu: [\d.]+ MiB went|host_gpu: \d+ launches ran", l):
            print(f"{t:9.3f} {l}")
    sys.stdout.write("stdout: " + " | ".join(x for x in out.splitlines() if x.strip())[:400] + "\n")


if __name__ == "__main__":
    main()
