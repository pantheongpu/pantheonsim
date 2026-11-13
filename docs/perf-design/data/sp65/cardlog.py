#!/usr/bin/env python3
"""Round sp65: reads a VGPU_HOST_GPU_LOG=1 run's stderr (and stdout) on stdin and prints, per kernel, what its launches on the card took against
the simulator's estimate (question 2 on the card), how many the streak rule sent and how many stayed for their size; then the mode's summary
lines as they are. Compact: one line a kernel, not one a launch.
"""
import collections, re, statistics, sys

card = collections.defaultdict(list)     # kernel -> [(kernel_ms, total_ms, sim_ms, pushed_mib)]
streak = collections.Counter()
stays = collections.Counter()
summary = []
R_CARD = re.compile(r"^host_gpu: (\S+) on the card \(resident\): kernel ([0-9.]+) ms, total ([0-9.]+) ms \(sim estimate ([0-9.]+) ms\), ([0-9.]+) MiB pushed")
R_STREAK = re.compile(r"^host_gpu: (\S+): sent because the card holds the memory")
R_STAYS = re.compile(r"^host_gpu: (\S+) stays in the simulator \(sim ")
for line in sys.stdin:
    m = R_CARD.match(line)
    if m:
        card[m.group(1)].append(tuple(float(m.group(i)) for i in range(2, 6))); continue
    m = R_STREAK.match(line)
    if m: streak[m.group(1)] += 1; continue
    m = R_STAYS.match(line)
    if m: stays[m.group(1)] += 1; continue
    if re.match(r"^host_gpu(_turbo|_refusals|_sampled|_streak)?: ", line) and ("launches ran" in line or "MiB went" in line or "_turbo:" in line
                                                                             or "_refusals:" in line or "_sampled:" in line or "_streak:" in line):
        summary.append(line.rstrip()[:400])
print("kernel | card launches | kernel ms median (min-max) | card path ms median | sim estimate ms median | estimate / kernel median | MiB pushed median | sent by the streak rule | stayed for size")
for k in sorted(set(card) | set(streak) | set(stays)):
    v = card.get(k, [])
    if v:
        km = [x[0] for x in v]; tm = [x[1] for x in v]; sm = [x[2] for x in v]; pm = [x[3] for x in v]
        ratio = statistics.median(s / kk for s, kk in zip(sm, km) if kk > 0) if any(kk > 0 for kk in km) else float("nan")
        print("%s | %d | %.4f (%.4f-%.4f) | %.4f | %.4f | %.0f | %.2f | %d | %d" % (k[:48], len(v), statistics.median(km), min(km), max(km), statistics.median(tm),
              statistics.median(sm), ratio, statistics.median(pm), streak[k], stays[k]))
    else:
        print("%s | 0 | - | - | - | - | - | %d | %d" % (k[:48], streak[k], stays[k]))
for l in summary: print(l)
