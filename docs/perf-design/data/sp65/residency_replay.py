#!/usr/bin/env python3
"""Round sp65: replay the host-GPU size rule over a launch trace (trace_corpus.py), with the residency of the live memory, and say what a
residency-aware rule for launches in a card streak would change.

  residency_replay.py [options] TRACE...           per program: launches, on the card, too small, the streak counts, modelled seconds
  residency_replay.py --q1 TRACE...                 question 1: "too small" launches that sit between two card launches with every chunk resident
  residency_replay.py --changes TRACE...            per kernel: the launches whose decision the streak rule changes

The rules are the runtime's (include/vgpu/host_gpu.hpp, src/runtime/host_gpu.cpp Impl::launch), replayed with each launch's own simulator time
from the trace as the history:
  * a kernel's estimate is the smallest per-block time of its simulated launches, after two of them (one if the first took 0.25 s or more, for two
    offloads);
  * a kernel with no estimate: the first-launch rule (static count, 8 blocks per multiprocessor, 256 MiB live at most, a loop: always true, see
    Round sp60b A), else the sampler of sp60b/sp61, modelled: a launch of a kernel with a loop, before the driver is started, whose static estimate is
    at least the floor (30 ms and 3 x 0.6 ms per MiB live), sampled by whole blocks on a grid of 256 blocks or more, or inside its blocks on a smaller
    grid when its launch takes a second or more (the floor is then asked of the loop's bound, which the trace does not have: not asked); its estimate is 0.65 x its simulator time (the rule errs low: Round sp60b C), and it goes when
    sampled_offload says so;
  * the size rule with the start (should_offload_with_start) and the ski-rental account; the bans (offload_paid_off after a kernel's first card
    launch, the traffic table);
  * residency: 64 KiB chunks Host / Card / Both. A card launch pushes the Host chunks and leaves every live chunk Card; a simulator launch pulls every
    Card chunk and leaves every chunk Host; a host read pulls the Card chunks it covers, a host write or fill makes its chunks Host (a Card chunk it
    covers in part is pulled first). A new allocation is Host (the runtime pushes its zeros).
The card's kernel time is not in a trace made without a card: it is modelled as max(--kfloor, simulator time / --kratio) (question 2 fits the two
numbers to the card timings of Round sp59). Seconds: the simulator's launches at their trace time, card launches at 7 us + the kernel + the copies at
round13's rates (residency_model.py: 1/(1/12e9 + 1/8e9) per byte pushed, 1/(1/10e9+1/8e9+1/10e9) pulled, 10 us a call; PSCALE below),
the driver start 0.23 s. Host code between launches is not in the trace: the seconds are the device side only.

Streak rules (--rule):
  base     today's rule
  pull     a launch of a kernel with an estimate, while the card holds chunks the simulator does not (the previous launch ran on the card), goes
           to the card when sim + pull > margin x (launch + card kernel + push) and sim + pull >= 200 us: `pull` is what keeping it in the simulator
           pulls back first (every Card chunk) at the size rule's 10 GB/s. The bans compare the card path with sim + pull.
  pullpush as pull, and the simulator's side also pays pushing those bytes again at the next card launch
  pullnf   as pull, without the 200 us floor (to see what the floor keeps)
"""
import argparse, collections, json, math, os, sys

K = 64 * 1024
# the size rule's beliefs (Costs)
LAUNCH, PUSH, PULL, KDEFAULT, MARGIN, MIN_SIM, START = 7e-6, 12e9, 10e9, 100e-6, 2.0, 200e-6, 0.25
SIM_RATE = 100e6
# what is paid (residency_model.py, round13). round13 fitted PSCALE = 1.5 on LU (a copy that alternates with the simulator); Round sp59's GRAMSCHM card
# launches took 10.5 ms each with 48 MiB pushed (card_s / card), which PSCALE = 1.0 gives and 1.5 does not (15.7 ms, and its traffic ban then fires
# in the replay and did not on the rig): 1.0 here.
PSCALE = 1.0
PUSH_EFF, PULL_EFF = 1.0 / (1 / 12e9 + 1 / 8e9), 1.0 / (1 / 10e9 + 1 / 8e9 + 1 / 10e9)
CALL, START_PAID = 10e-6, 0.23
MP = 80                      # nvidia/a10g profile
FL_BLOCKS_PER_SM, FL_MAX = 8, 256 << 20
ONE_SAMPLE_MIN, ONE_SAMPLE_OFF = 0.25, 2
SAMPLE_FLOOR, SAMPLE_PER_BYTE, SAMPLE_SPEEDUP = 0.030, 6e-10, 8.0
TRAFFIC_N, TRAFFIC_PCT, TRAFFIC_MIN, TRAFFIC_SLOW = 2, 25, 12 << 20, 0.5
HOST, CARD, BOTH = 0, 1, 2


class Alloc:
    """The chunks of one allocation: one state for all of them (`uni`), or one each (`st`) once a host copy has touched part of it."""
    __slots__ = ("base", "size", "n", "uni", "st")

    def __init__(s, base, size):
        s.base, s.size, s.n = base, size, (size + K - 1) // K
        s.uni, s.st = HOST, None

    def flat(s):
        if s.st is None:
            s.st, s.uni = bytearray([s.uni]) * s.n, None
        return s.st

    def set_all(s, state): s.uni, s.st = state, None

    def bytes_in(s, state):
        if s.st is None: return s.size if s.uni == state else 0
        return sum(clen(s, i) for i, x in enumerate(s.st) if x == state)

    def runs_in(s, state):
        if s.st is None: return 1 if s.uni == state and s.n else 0
        return nruns([i for i, x in enumerate(s.st) if x == state])


def clen(a, i): return min(K, a.size - i * K)


def nruns(idx):
    n, prev = 0, None
    for i in idx:
        if prev is None or i != prev + 1: n += 1
        prev = i
    return n


class Replay:
    def __init__(s, rule, kratio, kfloor, sim_scale, kdefault_fit=None, sample=True):
        s.rule, s.kratio, s.kfloor, s.sim_scale, s.kdefault_fit, s.sample = rule, kratio, kfloor, sim_scale, kdefault_fit, sample
        s.allocs = {}
        s.t = 0.0
        s.pushed = s.pulled = 0
        s.hist = {}                 # kernel -> [per_block, first_total, samples, one_sample_offloads]
        s.card_k = {}               # kernel -> last card kernel seconds
        s.banned, s.heavy = set(), collections.Counter()
        s.started = False
        s.eligible = 0.0
        s.n = s.n_card = s.too_small = s.streak_sent = 0
        s.events = []               # per launch: (kernel, decision, why, sim_s, card_only_bytes, push_bytes)
        s.prev_card = False
        s.small_why, s.small_why_s = collections.Counter(), collections.Counter()

    def live(self): return sum(a.size for a in self.allocs.values())

    def bytes_in(self, state):
        return sum(a.bytes_in(state) for a in self.allocs.values())

    def find(self, addr):
        for a in self.allocs.values():
            if a.base <= addr < a.base + a.size: return a
        return None

    # ---- host side
    def host_op(self, op, addr, ln):
        a = self.find(addr)
        if not a or ln == 0: return
        lo = (addr - a.base) // K
        hi = min((addr + ln - 1 - a.base) // K, a.n - 1)
        if op == "R":
            if a.st is None and a.uni != CARD: return
            st = a.flat()
            idx = [i for i in range(lo, hi + 1) if st[i] == CARD]
            for i in idx:
                st[i] = BOTH
                b = clen(a, i); self.pulled += b; self.t += PSCALE * b / PULL_EFF
            self.t += nruns(idx) * CALL
        else:
            if a.st is None and a.uni == HOST: return
            st = a.flat()
            for i in range(lo, hi + 1):
                whole = addr <= a.base + i * K and addr + ln >= a.base + i * K + clen(a, i)
                if st[i] == CARD and not whole:
                    b = clen(a, i); self.pulled += b; self.t += PSCALE * b / PULL_EFF + CALL
                st[i] = HOST

    # ---- the rules
    def predict(self, k, blocks):
        h = self.hist.get(k)
        if not h: return None
        ok1 = h[2] == 1 and h[1] >= ONE_SAMPLE_MIN and h[3] < ONE_SAMPLE_OFF
        if h[2] < 2 and not ok1: return None
        return h[0] * max(blocks, 1)

    def record(self, k, blocks, sec):
        h = self.hist.setdefault(k, [0.0, 0.0, 0, 0])
        pb = sec / blocks if blocks else sec
        if not h[2]: h[1] = sec
        h[0] = min(h[0], pb) if h[2] else pb
        h[2] += 1

    def card_kernel_model(self, sim_s):
        return max(self.kfloor, sim_s / self.kratio)

    def card_path(self, sim_s, push, kern, start):
        return start + LAUNCH + (kern if kern is not None else KDEFAULT) + push / PUSH

    def offload(self, sim_s, push, pull, kern, start=0.0):
        return sim_s >= MIN_SIM and sim_s > MARGIN * (self.card_path(sim_s, push, kern, start) + pull / PULL)

    def with_start(self, sim_s, push, pull, kern):
        if not self.offload(sim_s, push, pull, kern): return False
        if self.started: return True
        return self.offload(sim_s, push, pull, kern, START) or self.eligible >= START

    def streak_kernel_estimate(self, k):
        if k in self.card_k: return self.card_k[k]
        return self.kdefault_fit if self.kdefault_fit is not None else KDEFAULT

    # ---- a launch
    def launch(self, k, blocks, sim_s, in_set, tpb, instrs, loop):
        # the rig's simulator against server3's: a launch of fewer blocks than host threads runs on as many threads as it has blocks, where the rig's
        # one-thread speed is what counts; a wider one on all four (the rig's 4 vCPU are 2 cores)
        sim_s *= self.sim_scale[1] if blocks < 4 else self.sim_scale[0]
        self.n += 1
        card_only = self.bytes_in(CARD)
        push = self.bytes_in(HOST)
        live = self.live()
        ev = [k, "sim", "", sim_s, card_only, push, live]
        self.events.append(ev)
        if not in_set:
            ev[2] = "refused"; return self.sim_launch(k, blocks, sim_s, in_set)
        est = self.predict(k, blocks)
        sampled = False
        if est is None and self.hist.get(k, [0, 0, 0])[2] == 0 and k not in self.card_k:
            if blocks >= FL_BLOCKS_PER_SM * MP and live <= FL_MAX:
                st = blocks * ((tpb + 31) // 32) * instrs / SIM_RATE
                if self.with_start(st, live, live, None): est = st
        if est is None and self.sample and not self.started and loop:
            st = blocks * ((tpb + 31) // 32) * instrs / SIM_RATE
            floor = max(SAMPLE_FLOOR, 3 * SAMPLE_PER_BYTE * live)
            whole = None
            h = self.hist.get(k)
            if h and h[2]: whole = h[0] * max(blocks, 1)
            inside = blocks < 256 and sim_s >= 1.0
            if (blocks >= 256 or inside) and (whole is None or whole > MARGIN * (START + LAUNCH + KDEFAULT + 2 * live / PUSH)) and (st >= floor or inside):
                e = 0.65 * sim_s
                kern = max(KDEFAULT, e / SAMPLE_SPEEDUP)
                if e >= MIN_SIM and e > MARGIN * (START + LAUNCH + kern + push / PUSH + live / PULL):
                    est, sampled = e, True
        if est is None:
            ev[2] = "no_estimate"; return self.sim_launch(k, blocks, sim_s, in_set)
        if k in self.banned and not sampled:
            ev[2] = "banned"; return self.sim_launch(k, blocks, sim_s, in_set)
        kern_known = self.card_k.get(k)
        go = sampled or self.with_start(est, push, 0, kern_known)
        streak = False
        ref = est
        if not go and self.rule != "base" and self.started and card_only > 0:
            keep = est + card_only / PULL + (live / PUSH if self.rule == "pullpush" else 0.0)
            path = LAUNCH + self.streak_kernel_estimate(k) + push / PUSH
            if (keep >= MIN_SIM or self.rule == "pullnf") and keep > MARGIN * path:
                go, streak, ref = True, True, keep
        if not go:
            ev[2] = "too_small"; self.too_small += 1
            # why, for question 2: under the floor; else the push of the Host chunks is the larger part of the card path; else the kernel's time
            # there (assumed 100 us before it has run there, measured after); the start, while the driver is not up
            if not self.started and self.offload(est, push, 0, kern_known): why = "start"
            elif est < MIN_SIM: why = "floor"
            elif push / PUSH >= (kern_known if kern_known is not None else KDEFAULT): why = "push"
            elif kern_known is None: why = "kernel_default"
            else: why = "kernel_measured"
            self.small_why[why] += 1
            self.small_why_s[why] += sim_s
            return self.sim_launch(k, blocks, sim_s, in_set)
        # the card path
        if not self.started:
            self.started = True; self.t += START_PAID
        first_card = k not in self.card_k
        kern = self.card_kernel_model(sim_s)
        idx_calls = sum(a.runs_in(HOST) for a in self.allocs.values())
        total = LAUNCH + kern + PSCALE * push / PUSH_EFF + idx_calls * CALL
        self.t += total; self.pushed += push
        for a in self.allocs.values(): a.set_all(CARD)
        if not first_card and total > ref: self.banned.add(k)
        big = push > TRAFFIC_MIN and live > 0 and push * 100 > live * TRAFFIC_PCT
        slow = first_card or total > TRAFFIC_SLOW * ref
        self.heavy[k] = self.heavy[k] + 1 if (big and slow) else 0
        if self.heavy[k] >= TRAFFIC_N: self.banned.add(k)
        self.card_k[k] = kern
        h = self.hist.get(k)
        if h and h[2] < 2: h[3] += 1
        if sampled and h is None:
            self.record(k, blocks, est); self.hist[k][2] = 2
        self.n_card += 1
        if streak: self.streak_sent += 1
        ev[1] = "card"; ev[2] = "streak" if streak else ("sampled" if sampled else "")

    def sim_launch(self, k, blocks, sim_s, in_set):
        idx_calls = 0
        for a in self.allocs.values():
            b = a.bytes_in(CARD)
            self.pulled += b; self.t += PSCALE * b / PULL_EFF
            idx_calls += a.runs_in(CARD)
            a.set_all(HOST)
        self.t += idx_calls * CALL + sim_s
        self.record(k, blocks, sim_s)
        if in_set: self.eligible += sim_s


def parse(path):
    for line in open(path, errors="replace"):
        f = line.split()
        if not f: continue
        c = f[0]
        if c == "L":
            if "|" not in f or f.index("|") < 9: continue          # cut by the timeout, or an old trace
            yield ("L", f[1], int(f[2]), float(f[3]), int(f[4]), int(f[6]), int(f[7]), int(f[8]))
        elif c in "AFWRZ" and len(f) >= 3:
            try: yield (c, int(f[1], 16), int(f[2], 16))
            except ValueError: continue


def run(path, rule="base", kratio=300.0, kfloor=12e-6, sim_scale=1.0, kdefault_fit=None, sample=True):
    if not isinstance(sim_scale, tuple): sim_scale = (sim_scale, sim_scale)
    r = Replay(rule, kratio, kfloor, sim_scale, kdefault_fit, sample)
    for ev in parse(path):
        c = ev[0]
        if c == "A": r.allocs[ev[1]] = Alloc(ev[1], ev[2])
        elif c == "F": r.allocs.pop(ev[1], None)
        elif c in "WRZ": r.host_op(c, ev[1], ev[2])
        else: r.launch(*ev[1:])
    return r


def name_of(p): return os.path.basename(p).split(".")[0]


SCALES = (1.0, 1.25, 1.5, 1.8, 2.2, 2.65, 3.2, 4.0, 5.0)
SCALE_PAIRS = [(m, o) for m in SCALES for o in SCALES if o <= m]


def fmt_scale(sc):
    if not isinstance(sc, tuple): return "%.2f" % sc
    return "%.2f" % sc[0] if sc[0] == sc[1] else "%.2f / %.2f" % sc


def fit_scale(path, bench, mode):
    """The simulator scale (server3 to the rig) at which the base rule's counts come closest to what Round sp59's run of the program recorded on the
    rig (`card` and the `too_small` launches kept): (scale, distance), or (None, None) when sp59 has no record. sp59 had no sampler: replayed without."""
    prog = name_of(path).replace("_", "/", 1)
    rec = None
    for suite in ("polybench", "rodinia", "hecbench"):
        rec = bench.get(suite, {}).get("programs", {}).get(prog, {}).get(mode) or rec
    if not rec or "card" not in rec: return None, None
    want_card, want_small = rec["card"], (rec.get("kept") or {}).get("too_small", 0)
    best = None
    one = any(int(l.split()[2]) < 4 for l in open(path, errors="replace") if l.startswith("L ") and len(l.split()) > 9)
    for sc in (SCALE_PAIRS if one else [(m, m) for m in SCALES]):
        r = run(path, "base", 1e9, 12e-6, sc, sample=False)
        d = abs(r.n_card - want_card) + abs(r.too_small - want_small)
        if best is None or d < best[1] or (d == best[1] and abs(sc[0] - 2.65) < abs(best[0][0] - 2.65)): best = (sc, d)
    return best


def q1(r):
    """too-small launches with the card holding every chunk (Host bytes 0) after a card launch, and a card launch after them before the program ends"""
    ev = r.events
    nxt_card = [False] * (len(ev) + 1)
    for i in range(len(ev) - 1, -1, -1): nxt_card[i] = nxt_card[i + 1] or ev[i][1] == "card"
    rows = collections.defaultdict(lambda: [0, 0, 0, 0.0, 0])   # kernel -> [count, pulled bytes, re-pushed bytes, sim s, partly resident]
    prev_card = False
    for i, e in enumerate(ev):
        k, dec, why, sim_s, card_only, push, live = e
        if dec == "sim" and why == "too_small" and prev_card and nxt_card[i + 1]:
            row = rows[k]
            if push == 0:
                row[0] += 1; row[1] += card_only; row[2] += live; row[3] += sim_s
            else:
                row[4] += 1
        prev_card = dec == "card"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--rule", default="pull")
    ap.add_argument("--kratio", type=float, default=300.0)
    ap.add_argument("--kfloor", type=float, default=12e-6)
    ap.add_argument("--kdefault", type=float, default=None, help="the streak rule's card kernel time for a kernel never measured (default: 100 us)")
    ap.add_argument("--sim-scale", type=float, default=1.0, help="multiply the trace's simulator times (the rig's simulator is slower than server3's)")
    ap.add_argument("--one-scale", type=float, default=None, help="the same for launches of fewer than 4 blocks (default: --sim-scale)")
    ap.add_argument("--fit", default="", help="Round sp59's bench.json: per program, the scale at which the base rule (no sampler) reproduces sp59's counts")
    ap.add_argument("--q1", action="store_true"); ap.add_argument("--why", action="store_true"); ap.add_argument("--changes", action="store_true")
    ap.add_argument("--json", default="", help="write the per-program records here")
    a = ap.parse_args()
    bench = json.load(open(a.fit)) if a.fit else None
    records = {}

    def scale_of(p):
        dflt = (a.sim_scale, a.one_scale if a.one_scale is not None else a.sim_scale)
        if not bench: return dflt, None
        mode = "exact" if ".exact." in os.path.basename(p) else "turbo"
        sc, d = fit_scale(p, bench, mode)
        return (sc, d) if sc is not None else (dflt, None)

    if a.q1:
        print("| program | scale | kernel | too small, all resident | MiB pulled | MiB pushed again | their simulator s | too small, some Host chunks |")
        print("|---|---:|---|---:|---:|---:|---:|---:|")
        tot = [0, 0, 0, 0.0, 0]; progs = 0; total_small = 0
        for p in a.traces:
            sc, _ = scale_of(p)
            r = run(p, "base", a.kratio, a.kfloor, sc)
            total_small += r.too_small
            rows = q1(r)
            if rows: progs += 1
            for k, v in sorted(rows.items(), key=lambda kv: -kv[1][1]):
                print("| %s | %s | %s | %d | %.1f | %.1f | %.4f | %d |" % (name_of(p), fmt_scale(sc), k[:40], v[0], v[1] / 1048576, v[2] / 1048576, v[3], v[4]))
                for j in range(5): tot[j] += v[j]
            records[name_of(p)] = {"scale": sc, "too_small": r.too_small, "q1": {k: v for k, v in rows.items()}}
        print("| all (%d programs) | | | %d | %.1f | %.1f | %.3f | %d |" % (progs, tot[0], tot[1] / 1048576, tot[2] / 1048576, tot[3], tot[4]))
        print("\ntoo small in all: %d" % total_small)
    elif a.why:
        tot, tot_s = collections.Counter(), collections.Counter()
        print("| program | scale | too small | floor (under 200 us) | the push | kernel time, 100 us assumed | kernel time, measured | the start |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for p in a.traces:
            sc, _ = scale_of(p)
            r = run(p, "base", a.kratio, a.kfloor, sc)
            if not r.too_small: continue
            w = r.small_why
            print("| %s | %s | %d | %d | %d | %d | %d | %d |" % (name_of(p), fmt_scale(sc), r.too_small, w["floor"], w["push"], w["kernel_default"], w["kernel_measured"], w["start"]))
            tot.update(w); tot_s.update(r.small_why_s)
            records[name_of(p)] = {"scale": sc, "why": dict(w)}
        print("| all | | %d | %d | %d | %d | %d | %d |" % (sum(tot.values()), tot["floor"], tot["push"], tot["kernel_default"], tot["kernel_measured"], tot["start"]))
        print("| all, simulator s | | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (sum(tot_s.values()), tot_s["floor"], tot_s["push"], tot_s["kernel_default"], tot_s["kernel_measured"], tot_s["start"]))
    elif a.changes:
        print("| program | scale | kernel | launches | today: card | today: too small | %s: card | of them by the streak rule | MiB only on the card at those launches |" % a.rule)
        print("|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for p in a.traces:
            sc, _ = scale_of(p)
            b = run(p, "base", a.kratio, a.kfloor, sc)
            n = run(p, a.rule, a.kratio, a.kfloor, sc, a.kdefault)
            per = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
            for e in b.events:
                v = per[e[0]]; v[0] += 1; v[1] += e[1] == "card"; v[2] += e[2] == "too_small"
            for e in n.events:
                v = per[e[0]]; v[3] += e[1] == "card"; v[4] += e[2] == "streak"
                if e[2] == "streak": v[5] += e[4]
            for k, v in per.items():
                if v[1] != v[3]:
                    print("| %s | %s | %s | %d | %d | %d | %d | %d | %.1f |" % (name_of(p), fmt_scale(sc), k[:40], v[0], v[1], v[2], v[3], v[4], v[5] / 1048576))
                    records.setdefault(name_of(p), {"scale": sc, "kernels": {}})["kernels"][k] = v
    else:
        print("| program | scale (fit distance) | launches | today: card | too small | %s: card | sent by the streak rule | MiB moved today | MiB moved %s | device s today | device s %s | today / %s |" % ((a.rule,) * 4))
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for p in a.traces:
            sc, d = scale_of(p)
            b = run(p, "base", a.kratio, a.kfloor, sc)
            n = run(p, a.rule, a.kratio, a.kfloor, sc, a.kdefault)
            print("| %s | %s%s | %d | %d | %d | %d | %d | %.1f | %.1f | %.3f | %.3f | %.3f |" % (name_of(p), fmt_scale(sc), "" if d is None else " (%d)" % d, b.n, b.n_card, b.too_small,
                  n.n_card, n.streak_sent, (b.pushed + b.pulled) / 1048576, (n.pushed + n.pulled) / 1048576, b.t, n.t, b.t / max(n.t, 1e-12)))
            records[name_of(p)] = {"scale": sc, "fit": d, "launches": b.n, "card": b.n_card, "too_small": b.too_small, "new_card": n.n_card,
                                   "streak": n.streak_sent, "mib_today": round((b.pushed + b.pulled) / 1048576, 1),
                                   "mib_new": round((n.pushed + n.pulled) / 1048576, 1), "s_today": round(b.t, 4), "s_new": round(n.t, 4)}
    if a.json:
        json.dump(records, open(a.json, "w"), indent=0, sort_keys=True, default=lambda o: list(o) if isinstance(o, (set, tuple)) else str(o))


if __name__ == "__main__":
    main()
