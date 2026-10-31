#!/usr/bin/env python3
"""Step A of the persistent-residency question: replay a VGPU_HOST_GPU_TRACE stream and count what each policy moves and costs.

Trace lines: A base size | F base size | W/R/Z addr len (host copy in / out / fill of device memory) |
             L kernel blocks sim_s in_set card_s | arg-allocation-bases...
Every launch of the program is in the trace (the run was a VERIFY run: each launch ran in the simulator, and in-set ones on the card too).
Policies, per launch (decided by the same rule as the runtime, with the launch's own simulator time as the oracle; a kernel's first two launches stay
in the simulator, as in the default rule):
  off     : every launch in the simulator
  wt      : write-through (today): an offloaded launch pushes all live bytes and reads all live bytes back
  persist : persistent card copy, sound: chunk states Host/Card/Both; an offloaded launch pushes Host chunks and leaves every live chunk Card; a launch the
            simulator runs first pulls every Card chunk and leaves every live chunk Host (it may read or write anything)
  argsonly: as persist, but a launch is assumed to touch only the allocations its arguments name (UNSOUND: pointers stored in memory are not followed);
            an upper bound on what a finer residency (per-allocation reach) could give
A host read pulls Card chunks it covers (they become Both); a host write or fill makes the chunks Host (a Card chunk partly covered is pulled first); a free drops
the allocation; a new allocation is zero (not pushed until a launch needs it: `zskip`, the optimistic reading, or counted as Host, the pessimistic one).
Costs (09 doc, M3): launch+sync 7 us, push 12 GB/s, pull 10 GB/s, 10 us per pull or push call (runs of adjacent chunks are one call).
"""
import sys, collections
K = 64 * 1024
LAUNCH, PUSH, PULL, CALL = 7e-6, 12e9, 10e9, 10e-6          # the link (M3): what the size rule believes
# What the prototype actually pays per byte, measured on FDTD-2D (48 MiB live, 43 ms per write-through launch = 0.85 ns per live byte for the push, the
# read-back, the second read of the simulator's memory and the compare), and derived for a persistent copy, whose push is a gather of the simulator's chunks
# (memcpy ~8 GB/s) plus the copy in, and whose pull is the copy out plus the write into the simulator.
WT_NS_PER_LIVE_BYTE = 0.854e-9
PUSH_EFF, PULL_EFF = 1.0 / (1 / 12e9 + 1 / 8e9), 1.0 / (1 / 10e9 + 1 / 8e9 + 1 / 10e9)
MARGIN, MIN_SIM = 2.0, 200e-6

class Alloc:
    __slots__ = ('base', 'size', 'st')
    def __init__(s, base, size, init):
        s.base, s.size = base, size
        s.st = bytearray([init]) * ((size + K - 1) // K)   # 0 Host, 1 Card, 2 Both, 3 Zero (never written)
def chunk_bytes(a, i): return min(K, a.size - i * K)
def runs(idx):
    n = 0; prev = None
    for i in idx:
        if prev is None or i != prev + 1: n += 1
        prev = i
    return n

class Sim:
    def __init__(self, policy, zskip):
        self.policy, self.zskip = policy, zskip
        self.allocs = {}              # base -> Alloc
        self.t = 0.0                  # seconds spent (device-side work + transfers)
        self.pushed = self.pulled = 0
        self.n_off = self.n_sim = 0
        self.kcount = collections.Counter()
        self.calls = 0
        self.banned = set()
    def find(self, addr):
        # linear scan is fine: few live allocations
        for a in self.allocs.values():
            if a.base <= addr < a.base + a.size: return a
        return None
    def live_bytes(self): return sum(a.size for a in self.allocs.values())
    def pull_chunks(self, a, lo, hi):    # chunks [lo, hi] that are Card -> Both; returns bytes
        b = 0; idx = []
        for i in range(lo, hi + 1):
            if a.st[i] == 1: a.st[i] = 2; b += chunk_bytes(a, i); idx.append(i)
        self.calls += runs(idx)
        return b, runs(idx)
    def host_op(self, op, addr, ln):
        if self.policy in ('off', 'wt'): return
        a = self.find(addr)
        if not a: return
        lo = (addr - a.base) // K; hi = min((addr + ln - 1 - a.base) // K, len(a.st) - 1)
        if op == 'R':
            b, r = self.pull_chunks(a, lo, hi); self.pulled += b; self.t += b / PULL_EFF + r * CALL
        else:
            # W / Z: a partly covered Card chunk is pulled first; then the chunk is the host's
            for i in range(lo, hi + 1):
                whole = addr <= a.base + i * K and addr + ln >= a.base + i * K + chunk_bytes(a, i)
                if a.st[i] == 1 and not whole:
                    b = chunk_bytes(a, i); self.pulled += b; self.t += b / PULL_EFF + CALL
                a.st[i] = 0
    def launch(self, name, sim_s, in_set, card_s, bases):
        self.kcount[name] += 1
        first = self.kcount[name] <= 2
        pol = self.policy
        if pol == 'off' or not in_set or card_s < 0 or first or sim_s < MIN_SIM:
            self.sim_launch(sim_s, bases); return
        if name in self.banned:
            self.sim_launch(sim_s, bases); return
        if pol == 'wt':
            tot = self.live_bytes(); believed = LAUNCH + card_s + tot / PUSH + tot / PULL + 2 * CALL
            actual = LAUNCH + card_s + tot * WT_NS_PER_LIVE_BYTE
            if sim_s > MARGIN * believed:
                self.t += actual; self.pushed += tot; self.pulled += tot; self.n_off += 1; self.calls += 2
                if self.kcount[name] > 3 and actual > sim_s: self.banned.add(name)   # the runtime's ban: one launch slower than the simulator ends it
            else: self.sim_launch(sim_s, bases)
            return
        reach = [a for a in self.allocs.values() if a.base in bases] if pol == 'argsonly' else list(self.allocs.values())
        push = 0; pidx = 0
        for a in reach:
            idx = []
            for i, s in enumerate(a.st):
                if s == 0 or (s == 3 and not self.zskip): idx.append(i); push += chunk_bytes(a, i)
            pidx += runs(idx)
        path = LAUNCH + card_s + push / PUSH_EFF + pidx * CALL
        if sim_s > MARGIN * path:
            for a in reach:
                for i, s in enumerate(a.st): a.st[i] = 1
            self.t += path; self.pushed += push; self.n_off += 1; self.calls += pidx
            if self.kcount[name] > 3 and path > sim_s: self.banned.add(name)
        else: self.sim_launch(sim_s, bases)
    def sim_launch(self, sim_s, bases):
        self.n_sim += 1; self.t += sim_s
        pol = self.policy
        if pol in ('off', 'wt'): return
        reach = [a for a in self.allocs.values() if a.base in bases] if pol == 'argsonly' else list(self.allocs.values())
        for a in reach:
            b, r = self.pull_chunks(a, 0, len(a.st) - 1)
            self.pulled += b; self.t += b / PULL_EFF + r * CALL
            for i in range(len(a.st)): a.st[i] = 0

import gzip
def _open(p): return gzip.open(p, 'rt') if p.endswith('.gz') else open(p)
def run(path, policy, zskip=False):
    s = Sim(policy, zskip)
    tot_sim = 0.0
    for line in _open(path):
        f = line.split()
        if not f or (f[0] == 'L' and ('|' not in f or len(f) < 7)) or (f[0] in 'AFWRZ' and len(f) < 3): continue   # a line cut by the run's timeout
        c = f[0]
        if c == 'A': b = int(f[1], 16); s.allocs[b] = Alloc(b, int(f[2], 16), 3)
        elif c == 'F': s.allocs.pop(int(f[1], 16), None)
        elif c in 'WRZ': s.host_op(c, int(f[1], 16), int(f[2], 16))
        elif c == 'L':
            bar = f.index('|')
            name, blocks, sim_s, in_set, card_s = f[1], int(f[2]), float(f[3]), int(f[4]), float(f[5])
            bases = set(int(x, 16) for x in f[bar + 1:])
            s.launch(name, sim_s, in_set, card_s, bases)
    return s

def summary(path):
    ev = collections.Counter(); chained = prev_off = 0; launches = in_set = 0; sim_total = 0.0; in_set_sim = 0.0
    touched = False; off_flag = False
    for line in _open(path):
        f = line.split()
        if not f or (f[0] == 'L' and ('|' not in f or len(f) < 7)) or (f[0] in 'AFWRZ' and len(f) < 3): continue   # a line cut by the run's timeout
        c = f[0]
        if c in 'WRZ': touched = True
        elif c == 'L':
            launches += 1; sim_total += float(f[3]); ins = int(f[4])
            if ins: in_set += 1; in_set_sim += float(f[3])
            if ins and prev_off and not touched: chained += 1
            prev_off = ins; touched = False
    return launches, in_set, sim_total, in_set_sim, chained

if __name__ == '__main__':
    zskip = '--zskip' in sys.argv
    for p in [a for a in sys.argv[1:] if not a.startswith('--')]:

        launches, in_set, sim_total, in_set_sim, chained = summary(p)
        r = {pol: run(p, pol, zskip) for pol in ('off', 'wt', 'persist', 'argsonly')}
        name = p.split('/')[-1].replace('.gz', '').replace('.trace', '')
        print('%-26s L=%6d inset=%6d chained=%6d | off %8.3f wt %8.3f persist %8.3f args %8.3f s | offloaded wt %5d persist %5d | MiB moved wt %9.1f persist %8.1f | persist/wt %6.2fx' % (
            name, launches, in_set, chained, r['off'].t, r['wt'].t, r['persist'].t, r['argsonly'].t, r['wt'].n_off, r['persist'].n_off,
            (r['wt'].pushed + r['wt'].pulled) / 1048576.0, (r['persist'].pushed + r['persist'].pulled) / 1048576.0, r['wt'].t / max(r['persist'].t, 1e-9)))
