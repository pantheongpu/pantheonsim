#!/usr/bin/env python3
"""Round sp76, #662: which rule buys the driver start, replayed over the launches of real programs.

`collect` runs programs of the corpora in turbo mode with a driver start that is never bought (VGPU_HOST_GPU_START_COST=100000, no sampling, no
first-launch rule), so that every launch is the simulator's, and keeps of each launch what the mode knew: the kernel, the grid, the simulator's
time, whether the kernel is admitted (VGPU_HOST_GPU_TRACE, the "L" lines), with the allocations and frees (the live memory) and the host's writes.

`model` replays each record under the start rules: the history estimate (SimTimeTable), the size rule, the account of simulator time spent and
the share a launch must save (include/vgpu/host_gpu.hpp should_offload_with_start, start_share_launches), and prices the program: the launches
kept at the simulator's time, the start at 0.25 s and a card launch at its path. `--scale` multiplies the simulator's times (server3's
simulator runs these programs 2.5 to 3 times as fast as the A10G rig's). The model has no sample and no first-launch rule (programs whose
start a sampled launch buys are not its business), takes a card kernel for 25 us, and charges a push of the live memory to the first card
launch after any launch the simulator ran or any host write.

  start_model.py collect --build BUILD --corp CORP --out DIR [--programs a,b] [--skip a,b] [--threads 3] [--timeout 100]
  start_model.py model --traces DIR [--scale 1,2.75] [--rules cur,regret,...]
"""
import argparse, fcntl, glob, os, shutil, subprocess, sys, tempfile, time

START = 0.25
LAUNCH = 7e-6
KDEF = 100e-6
PUSH = 12e9
PULL = 10e9
MARGIN = 2.0
MINSIM = 200e-6
CARD_KERNEL = 25e-6      # what a short kernel took on the A10G (Round sp75's card log of silu: 21 to 26 us)
MODULE_LOAD = 1.1e-3     # the first card launch of a kernel (the same log: 1.17 ms with 4 MiB pushed)


def collect(a):
    sys.path.insert(0, a.real)
    import real
    os.makedirs(a.out, exist_ok=True)
    want = set(x for x in a.programs.split(",") if x)
    skip = set(x for x in a.skip.split(",") if x)
    fd, since = None, 0.0
    for corpus, name, cwd, argv, copy_cwd in real.manifest(a.corp, a.corpora.split(",")):
        if (want and name not in want) or name in skip:
            continue
        dest = os.path.join(a.out, corpus + "__" + name + ".txt")
        if os.path.exists(dest):
            continue
        if fd is not None and time.monotonic() - since > 150:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            fd = None
            time.sleep(1)
        if fd is None:
            fd = os.open(a.lock, os.O_RDWR | os.O_CREAT, 0o666)
            fcntl.flock(fd, fcntl.LOCK_EX)
            since = time.monotonic()
        raw = tempfile.mktemp(prefix="trace-")
        env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
        env.update(VGPU_GPU=a.gpu, LD_LIBRARY_PATH=os.path.join(a.build, "shim"), VGPU_THREADS=a.threads, VGPU_HOST_GPU="turbo", VGPU_HOST_GPU_SAMPLE="0",
                   VGPU_HOST_GPU_NO_FIRST_LAUNCH="1", VGPU_HOST_GPU_START_COST="100000", VGPU_HOST_GPU_TRACE=raw, VGPU_QUIET="1")
        scratch, rcwd = None, cwd
        if copy_cwd:
            scratch = tempfile.mkdtemp(prefix="trace-cwd-")
            dd = os.path.join(os.path.dirname(argv[0]), "data")
            with open(os.path.join(scratch, "filelist"), "w") as f:
                f.write("".join(os.path.join(dd, "cane4_%d.db\n" % j) for j in range(4)))
            rcwd = scratch
        t0 = time.monotonic()
        try:
            rc = subprocess.run(["nice", "-n", "19"] + argv, cwd=rcwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=a.timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -1
        wall = time.monotonic() - t0
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        n = 0
        if os.path.exists(raw):
            with open(raw) as f, open(dest, "w") as o:
                o.write("# %s/%s rc=%d wall_s=%.3f\n" % (corpus, name, rc, wall))
                wrote = False
                for line in f:
                    c = line[0]
                    if c == "L":
                        o.write(line.split(" |")[0] + "\n")
                        n += 1
                        wrote = False
                    elif c in "AF":
                        o.write(line)
                    elif c == "W" and not wrote:   # (one mark for the host writes between two launches)
                        o.write("W\n")
                        wrote = True
            os.remove(raw)
        print("%-40s rc=%d %.2f s, %d launches" % (corpus + "/" + name, rc, wall, n), flush=True)
    if fd is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def load(path):
    ev = []
    with open(path) as f:
        head = f.readline().strip()
        for line in f:
            p = line.split()
            if p[0] == "L":   # L kernel blocks sim_s in_set card_s threads_per_block static_instrs loop
                ev.append(("L", p[1], int(p[2]), float(p[3]), p[4] == "1"))
            elif p[0] == "A":
                ev.append(("A", int(p[1], 16), int(p[2], 16)))
            elif p[0] == "F":
                ev.append(("F", int(p[1], 16)))
            else:
                ev.append(("W",))
    return head, ev


def card_path(sim, push, kern, start=0.0):
    return start + LAUNCH + (kern if kern is not None else KDEF) + push / PUSH


def should(sim, push, kern, start=0.0):
    return sim >= MINSIM and sim > MARGIN * card_path(sim, push, kern, start)


def share_launches(kept, repeats=200, rmax=500):
    return max(50, min(kept, rmax)) if kept >= repeats else 50


def replay(ev, scale, rule):
    """-> (seconds of the launches and the start, launch index that bought the start or None, launches sent, simulator seconds of all launches)"""
    table = {}   # kernel -> [samples, per_block, first_total, one_sample_offloads]
    card_k = {}
    live = {}
    acct = regret = 0.0
    up = False
    dirty = True   # the simulator holds chunks the card does not
    total = simall = 0.0
    bought = None
    sent = 0
    i = 0
    seen_k = {}    # kernel -> simulator seconds of its launches so far (kept, admitted)
    n_adm = 0      # admitted launches the simulator has run so far
    for e in ev:
        if e[0] == "A":
            live[e[1]] = e[2]
            continue
        if e[0] == "F":
            live.pop(e[1], None)
            continue
        if e[0] == "W":
            dirty = True
            continue
        _, k, blocks, sim, in_set = e
        sim *= scale
        simall += sim
        i += 1
        lb = sum(live.values())
        t = table.get(k)
        est = None
        if in_set and t:
            one = t[0] == 1 and t[2] >= 0.25 and t[3] < 2
            if t[0] >= 2 or one:
                est = t[1] * max(blocks, 1)
        go = False
        push = lb if dirty else 0
        if est is not None:
            kern = card_k.get(k)
            ok_up = should(est, push, kern)
            if up:
                go = ok_up or (not dirty and est + lb / PULL > MARGIN * card_path(est, 0, kern) and est + lb / PULL >= MINSIM)
            elif ok_up:
                saving = est - card_path(est, push, kern)
                kept_n = t[0]
                if should(est, push, kern, START):
                    go = True
                else:
                    share = saving >= START / share_launches(kept_n)
                    if rule == "cur":
                        go = acct >= START and share
                    elif rule == "regret":       # what the launches the size rule would have sent with the driver up would have saved
                        go = regret >= START and share
                    elif rule == "regret+acct":  # ... and the account, as it is
                        go = regret >= START and acct >= START and share
                    elif rule.startswith("acct"):   # the account at a multiple of the start
                        go = acct >= float(rule[4:]) * START and share
                    elif rule == "kernel":       # this kernel alone: as much again as it has taken so far repays the start
                        go = acct >= START and share and seen_k.get(k, 0.0) >= START
                    elif rule == "kcount":       # as many launches again of this kernel as it has had, each saving what this one saves
                        go = acct >= START and share and saving * kept_n >= START
                    elif rule == "pcount":       # as many admitted launches again as the program has had, each saving what this one saves
                        go = acct >= START and share and saving * n_adm >= START
                    else:
                        raise SystemExit("unknown rule " + rule)
                if not go:
                    regret += saving
        if go:
            if not up:
                up = True
                bought = i
                total += START
            first = k not in card_k
            total += LAUNCH + CARD_KERNEL + push / PUSH + (MODULE_LOAD if first else 0)
            card_k[k] = CARD_KERNEL
            if t and t[0] < 2:
                t[3] += 1
            dirty = False
            sent += 1
        else:
            total += sim + (lb / PULL if up and not dirty else 0)
            dirty = True
            if not t:
                table[k] = [1, sim / max(blocks, 1), sim, 0]
            else:
                t[1] = min(t[1], sim / max(blocks, 1))
                t[0] += 1
            if in_set:
                n_adm += 1
                acct += sim
                seen_k[k] = seen_k.get(k, 0.0) + sim
    return total, bought, sent, simall, i


def model(a):
    rules = a.rules.split(",")
    scales = [float(x) for x in a.scale.split(",")]
    for scale in scales:
        print("\nSimulator times x %g. Seconds of the launches and the start under each rule (the launch that buys the start / launches sent); `sim`: all launches simulated.\n" % scale)
        print("| program | launches | sim s | " + " | ".join(rules) + " |")
        print("|---|---:|---:|" + "---|" * len(rules))
        tot = {r: 0.0 for r in rules}
        worse = {r: [] for r in rules}
        for path in sorted(glob.glob(os.path.join(a.traces, "*.txt"))):
            head, ev = load(path)
            name = head.split()[1]
            res = {r: replay(ev, scale, r) for r in rules}
            cur = res[rules[0]]
            if all(res[r][1] is None for r in rules) and not a.all:
                continue
            cells = []
            for r in rules:
                t, b, s, simall, n = res[r]
                tot[r] += t
                cells.append("%.3f (%s / %d)" % (t, b if b is not None else "-", s))
                if t > cur[0] * 1.03 and t - cur[0] > 0.005:
                    worse[r].append("%s %+.0f %%" % (name, (t / cur[0] - 1) * 100))
            print("| %s | %d | %.3f | %s |" % (name, cur[4], cur[3], " | ".join(cells)))
        print("\nSums: " + ", ".join("%s %.2f s" % (r, tot[r]) for r in rules))
        for r in rules[1:]:
            print("More than 3 %% over `%s` under `%s`: %s" % (rules[0], r, ", ".join(worse[r]) or "none"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--build", required=True)
    c.add_argument("--corp", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--real", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools", "bench", "real"))
    c.add_argument("--corpora", default="polybench,rodinia,hecbench,samples")
    c.add_argument("--programs", default="")
    c.add_argument("--skip", default="")
    c.add_argument("--gpu", default="nvidia/a10g")
    c.add_argument("--threads", default="3")
    c.add_argument("--timeout", type=float, default=100)
    c.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    m = sub.add_parser("model")
    m.add_argument("--traces", required=True)
    m.add_argument("--scale", default="1,2.75")
    m.add_argument("--rules", default="cur,regret,acct1.5,acct2,kernel,kcount,pcount")
    m.add_argument("--all", action="store_true")
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else model(a)


if __name__ == "__main__":
    main()
