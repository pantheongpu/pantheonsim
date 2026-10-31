#!/usr/bin/env python3
"""M3: apply the measured transfer costs (raw/m3_xfer.txt) to every launch captured in M2 and say where offload would win.
Per launch: sim = the simulator's wall time for the launch; card = launch+sync 7 us + the kernel's time on the card + the data moved.
Three data scenarios, from the worst for the card to the best:
  cold   : all live memory is pushed (host-resident) and the changed bytes are pulled back (a lone launch between two simulator phases)
  dirty  : only the bytes the launch changes are pushed and pulled (the simulator wrote them since the last launch, the card reads them back)
  chain  : nothing moves (a run of offloaded kernels on card-resident data)
Bandwidths: 64 KiB chunks one call each 4.4 GB/s up, 4.0 down; gather through pinned staging 5.8 GB/s; contiguous pinned 13.4 GB/s."""
import re, sys, collections
rows = []
for f in ['raw/m2_polybench.txt', 'raw/m2_rodinia.txt', 'raw/m2_hecbench.txt']:
    corpus = f.split('_')[1].split('.')[0]
    for line in open(f):
        m = re.match(r'(\d+) (\S+)\s+grid (\d+)x(\d+)x(\d+) block (\d+)x(\d+)x(\d+) mem\s+([\d.]+) MiB changed\s+([\d.]+) MiB chunks (\d+) bad (\d+) words (\d+) maxulp (\d+) \| card kernel\s+([\d.]+) us push\s+([\d.]+) ms pull\s+([\d.]+) ms \| sim\s+([\d.]+) ms \| (AGREE|DIFFER)', line)
        if not m: continue
        g = m.groups()
        rows.append(dict(corpus=corpus, name=g[1], blocks=int(g[2]) * int(g[3]) * int(g[4]), threads=int(g[5]) * int(g[6]) * int(g[7]), mem=float(g[8]), chg=float(g[9]), card_us=float(g[14]), sim_ms=float(g[17]), ok=g[18] == 'AGREE'))
MiB = 1048576.0
def cost(r, scen, up=4.4e9, down=4.0e9):
    base = 0.007 + r['card_us'] / 1000.0   # ms
    if scen == 'cold': return base + r['mem'] * MiB / up * 1e3 + r['chg'] * MiB / down * 1e3
    if scen == 'dirty': return base + r['chg'] * MiB / up * 1e3 + r['chg'] * MiB / down * 1e3
    return base
print('launches captured:', len(rows), ' agree:', sum(r['ok'] for r in rows))
tot = sum(r['sim_ms'] for r in rows)
print('sum of simulator time over these launches: %.1f s' % (tot / 1e3))
for scen in ('cold', 'dirty', 'chain'):
    for bw, (up, down) in (('chunk-at-a-time', (4.4e9, 4.0e9)), ('pinned bulk', (13.4e9, 13.2e9))):
        if scen == 'chain' and bw != 'chunk-at-a-time': continue
        win = [r for r in rows if cost(r, scen, up, down) < r['sim_ms'] / 2]   # the rule: offload only if the card path is under half the simulator's estimate (margin 2x)
        saved = sum(r['sim_ms'] - cost(r, scen, up, down) for r in win)
        lost_if_all = sum(max(0, cost(r, scen, up, down) - r['sim_ms']) for r in rows)
        print('%-6s %-16s: offload wins (margin 2x) on %3d of %d launches, saves %.1f s of %.1f s (%.1f%%); offloading ALL would lose %.1f s on the others' % (scen, bw, len(win), len(rows), saved / 1e3, tot / 1e3, 100 * saved / tot, lost_if_all / 1e3))
# by simulator time bands: which launches are worth it
print('\nby simulator time of the launch (cold scenario, chunk-at-a-time, margin 2x):')
bands = [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 1e9)]
for lo, hi in bands:
    b = [r for r in rows if lo <= r['sim_ms'] < hi]
    if not b: continue
    w = [r for r in b if cost(r, 'cold') < r['sim_ms'] / 2]; w2 = [r for r in b if cost(r, 'dirty') < r['sim_ms'] / 2]; w3 = [r for r in b if cost(r, 'chain') < r['sim_ms'] / 2]
    print('  sim %7g-%-7g ms: %3d launches, offload wins cold %3d, dirty %3d, chain %3d' % (lo, hi, len(b), len(w), len(w2), len(w3)))
# a size proxy: warp-instruction-equivalents is not in the capture; use threads*blocks as the grid size proxy
print('\nby grid size (threads total) -- median sim ms and median card ms (chain), launches:')
for lo, hi in [(0, 1024), (1024, 16384), (16384, 262144), (262144, 1 << 40)]:
    b = [r for r in rows if lo <= r['blocks'] * r['threads'] < hi]
    if not b: continue
    import statistics as st
    print('  %9d-%-11d threads: n=%3d median sim %9.3f ms, median card (launch+sync+kernel) %8.3f ms, wins chain %3d' % (lo, hi, len(b), st.median(r['sim_ms'] for r in b), st.median(0.007 + r['card_us'] / 1e3 for r in b), sum(cost(r, 'chain') < r['sim_ms'] / 2 for r in b)))
# the break-even: sim_ms needed so that cold/dirty cost wins as a function of bytes moved
print('\nbreak-even simulator time by bytes moved (card kernel ~0.1 ms, margin 2x):')
for mb in (0.0625, 0.5, 1, 4, 16, 64, 256, 1024):
    for name, up, down in (('chunk', 4.4e9, 4.0e9), ('pinned', 13.4e9, 13.2e9)):
        t = 0.007 + 0.1 + mb * MiB / up * 1e3 + mb * MiB / down * 1e3
        print('  move %7.3f MiB each way, %-6s: card path %.2f ms -> offload only if the simulator needs > %.1f ms' % (mb, name, t, 2 * t))
