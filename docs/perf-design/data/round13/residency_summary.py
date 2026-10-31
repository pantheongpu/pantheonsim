#!/usr/bin/env python3
import re, sys
rows = []
for l in open(sys.argv[1]):
    m = re.match(r'(\S+)\s+L=\s*(\d+) inset=\s*(\d+) chained=\s*(\d+) \| off\s+([\d.]+) wt\s+([\d.]+) persist\s+([\d.]+) args\s+([\d.]+) s \| offloaded wt\s+(\d+) persist\s+(\d+)', l)
    if m: rows.append((m.group(1), *map(float, m.groups()[1:])))
t = lambda i: sum(r[i] for r in rows)
print('programs', len(rows), 'with launches', sum(1 for r in rows if r[1] > 0))
print('launches', t(1), 'in the exact set', t(2), 'consecutive in-set with no host copy between', t(3), 'offloaded: write-through', t(8), 'persistent', t(9))
print('sum of device time over all programs: off %.1f s, write-through %.1f s, persistent %.1f s, per-allocation reach %.1f s' % (t(4), t(5), t(6), t(7)))
big = [r for r in rows if r[4] > 3 and r[8] > 0]
print('programs with >3 s of simulation that write-through already offloads:', len(big))
for r in big: print('  %-26s off %6.1f  wt %6.1f  persistent %6.1f  gain %.2fx' % (r[0], r[4], r[5], r[6], r[5] / r[6]))
ben = [r for r in rows if r[8] > 0 and r[5] / max(r[6], 1e-9) >= 1.5]
print('programs with persistent/write-through >= 1.5x:', len(ben), [r[0] for r in ben])
print('their launches %d, offloaded by persistent %d; device time off %.1f wt %.1f persistent %.1f' % (sum(r[1] for r in ben), sum(r[9] for r in ben), sum(r[4] for r in ben), sum(r[5] for r in ben), sum(r[6] for r in ben)))
print('programs where persistent offloads something:', sum(1 for r in rows if r[9] > 0), '; write-through:', sum(1 for r in rows if r[8] > 0))
