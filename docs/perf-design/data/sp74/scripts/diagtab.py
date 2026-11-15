#!/usr/bin/env python3
"""diagtab.py <lib> <diag file> [top]: the forms by TSC ticks in their handlers, with the handler's and the kernel's names (nm)."""
import subprocess, sys, bisect, collections
lib, f = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
syms = []
for l in subprocess.run(["nm", "-C", "--defined-only", lib], capture_output=True, text=True).stdout.splitlines():
    p = l.split(None, 2)
    if len(p) == 3 and p[1] in "tTwW":
        syms.append((int(p[0], 16), p[2]))
syms.sort()
addrs = [a for a, _ in syms]
def name(x):
    if not x: return "-"
    i = bisect.bisect_right(addrs, x) - 1
    n = syms[i][1]
    n = n.replace("(anonymous namespace)::", "").replace("vgpu::amd::gcn::", "")
    return n.split("(")[0][-60:]
rows = collections.defaultdict(lambda: [0, 0, 0, 0])
for l in open(f):
    p = l.split(None, 4)
    w = p[4].strip()
    r = rows[w]
    for i in range(4): r[i] += int(p[i])
tot = sum(r[0] for r in rows.values())
byh = collections.Counter()
out = []
for w, r in rows.items():
    parts = w.split(" run@")
    run, kern = parts[1].split(" kern@")
    h = name(int(run, 16)); k = name(int(kern, 16))
    byh[h] += r[0]
    out.append((r[0], r[1], r[2], r[3], parts[0], h, k))
print("ticks in tier handlers: %.3g; instructions %d" % (tot, sum(o[1] for o in out)))
print("by handler:")
for h, t in byh.most_common(25): print("  %5.1f%%  %s" % (100.0 * t / tot, h))
print("by form (share of ticks, ticks a call, calls, lanes a call, hand-backs):")
for t, n, la, b, w, h, k in sorted(out, reverse=True)[:top]:
    print("  %5.1f%%  %6.0f  %10d  %4.1f  %8d  %-58s %s / %s" % (100.0 * t / tot, t / max(n, 1), n, la / max(n, 1), b, w, h, k))
