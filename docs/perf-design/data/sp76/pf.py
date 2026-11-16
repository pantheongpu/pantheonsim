#!/usr/bin/env python3
"""Round sp76: Rodinia particlefilter_float with its clock seed fixed (fixtime.c), the bounded-block switch of Round sp75 on against off.

For each seed (a value of time(0)) the program is run --runs times a form, interleaved, against the stand-in driver in turbo mode, and once a
form with VGPU_HOST_GPU_LOG=1: the wall times per seed and form, and whether the mode's decisions are the same in the two forms: every line
that says what a launch did (sampled and sent or kept, started the driver, stayed, ran on the card or failed there) and the summary's
counts, with what a sample measured taken out (its times, and the counts it reads within a time: the blocks it ran, the warps it found in a
followed loop after a millisecond). The lines that give the REASON a first launch was not sampled are counted and not compared: with
the switch on a grid of 2 to 255 blocks is a candidate for bounded blocks and is refused by the floor or by its earlier launch, with it off
it is refused as a small grid; neither is sampled. A third form runs the program with the mode off. With --runs 0 only the logged runs are made.

  pf.py --build BUILD --corp CORP --stub libstubcuda.so --fixtime libfixtime.so --seeds 1763200000:1763200016 [--runs 5] [--threads 3] [--free]
  --free: the clock is not fixed (what the card run did): --seeds then only counts the runs
  --stub "": the real driver (on a rig with a card); --threads "": the host's own
"""
import argparse, fcntl, hashlib, os, re, statistics, subprocess, sys, time

NUM = re.compile(r"\d+\.\d+")


HELD = {"fd": None, "since": 0.0}


def run(argv, cwd, env, lock):
    # the machine's build lock, kept over the runs of up to 150 s and then given up for a moment (another session builds under it)
    if HELD["fd"] is not None and time.monotonic() - HELD["since"] > 150:
        fcntl.flock(HELD["fd"], fcntl.LOCK_UN)
        os.close(HELD["fd"])
        HELD["fd"] = None
        time.sleep(1.0)
    if HELD["fd"] is None:
        HELD["fd"] = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(HELD["fd"], fcntl.LOCK_EX)
        HELD["since"] = time.monotonic()
    t0 = time.monotonic()
    p = subprocess.run(["nice", "-n", "19"] + argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    return time.monotonic() - t0, p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--corp", required=True)
    ap.add_argument("--stub", required=True)
    ap.add_argument("--fixtime", default="")
    ap.add_argument("--seeds", required=True, help="first:last (last not included), or a comma list")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", default="3")
    ap.add_argument("--free", action="store_true")
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    a = ap.parse_args()
    seeds = list(range(*map(int, a.seeds.split(":")))) if ":" in a.seeds else [int(x) for x in a.seeds.split(",")]
    exe = os.path.join(a.corp, "rodinia/bin/particlefilter_float")
    argv = [exe] + "-x 128 -y 128 -z 10 -np 10000".split()
    base = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_") and not k.startswith("STUB_")}
    base.update(VGPU_GPU="nvidia/a10g", LD_LIBRARY_PATH=os.path.join(a.build, "shim"))
    if a.threads:
        base.update(VGPU_THREADS=a.threads)
    turbo = dict(VGPU_HOST_GPU="turbo")
    if a.stub:
        turbo.update(VGPU_HOST_GPU_LIBCUDA=a.stub, STUB_SM="86", STUB_MP="80")
    forms = [("off", {}), ("turbo", turbo), ("turbob0", dict(turbo, VGPU_HOST_GPU_SAMPLE_BLOCK="0"))]
    if a.runs == 0:
        print("| seed | the mode's decisions, turbo against turbob0 | launches kept / failed on the card |")
        print("|---|---|---|")
    else:
        print("| seed | off s (min to max) | turbo s (min to max) | turbob0 s (min to max) | turbo / turbob0 | outputs seen: off, turbo, turbob0 (all forms) | the mode's lines, turbo against turbob0 | launches kept / failed on the card | longest mean simulated launch of a kernel and reason, ms |")
        print("|---|---:|---:|---:|---:|---|---|---|---:|")
    all_t = {n: [] for n, _ in forms}
    for seed in seeds:
        env0 = dict(base)
        if not a.free:
            env0.update(LD_PRELOAD=a.fixtime, FIXTIME=str(seed))
        walls = {n: [] for n, _ in forms}
        outs = {n: set() for n, _ in forms}
        for i in range(a.runs):
            for n, extra in (forms if i % 2 == 0 else forms[::-1]):
                w, rc, out, err = run(argv, os.path.dirname(exe), dict(env0, **extra), a.lock)
                if rc != 0:
                    print("FAILED %s seed %d rc %d: %s" % (n, seed, rc, err[-300:]), file=sys.stderr)
                walls[n].append(w)
                # what the program computed: the position it found (XE, YE) and the distance (a bare number); every other line is a time
                outs[n].add(hashlib.sha256("\n".join(l for l in out.splitlines() if re.match(r"(XE|YE): |-?[0-9.]+$", l)).encode()).hexdigest()[:8])
        logs, longest, counts, unsampled = {}, 0.0, "", {}
        for n, extra in forms[1:]:
            _, _, _, err = run(argv, os.path.dirname(exe), dict(env0, VGPU_HOST_GPU_LOG="1", **extra), a.lock)
            lines = [l for l in err.splitlines() if l.startswith("host_gpu") and not l.startswith("host_gpu_overhead") and not l.startswith("host_gpu_turbo")]
            # the decisions as text: every measured time taken out (the estimates are measured times too), the names and counts kept
            unsampled[n] = sum(1 for l in lines if l.startswith("host_gpu_nosample: "))
            every = lines
            lines = [l for l in lines if not l.startswith("host_gpu_nosample: ") and not l.startswith("host_gpu_kept: ")]
            logs[n] = [NUM.sub("#", re.sub(r"(age_s|cost_s|fork_s|timed_s|first_block_s|estimate_s|launch_s|measure_s|sim_s|step_ns|per_turn|read_steps|measure_steps|left|read_turns|measure_turns|followed|unfollowed|ran)=\S+", r"\1=#", l)) for l in lines]
            for l in every:
                m = re.match(r"host_gpu_kept: kernel=\S+ launches=(\d+) sim_s=(\S+) reason=", l)
                if m and n == "turbo":
                    longest = max(longest, float(m.group(2)) / int(m.group(1)))
                m = re.match(r"host_gpu: (\d+) launches ran on the card, (\d+) in the simulator \(.*?(\d+) failed on the card\)", l)
                if m and n == "turbo":
                    counts = "%d / %d" % (int(m.group(2)) - int(m.group(3)), int(m.group(3)))
        same = "the same (%d lines; %d and %d launches said not to be sampled)" % (len(logs["turbo"]), unsampled["turbo"], unsampled["turbob0"]) \
            if logs["turbo"] == logs["turbob0"] and unsampled["turbo"] == unsampled["turbob0"] else "DIFFERENT (%d and %d lines): " % (len(logs["turbo"]), len(logs["turbob0"])) + "; ".join(
            x.split()[0] + " " + ", ".join("%s against %s" % (p, q) for p, q in zip(x.split(), y.split()) if p != q) for x, y in zip(logs["turbo"], logs["turbob0"]) if x != y)[:300]
        if a.runs == 0:
            print("| %s | %s | %s |" % ("free" if a.free else seed, same, counts), flush=True)
            continue
        med = {n: statistics.median(walls[n]) for n in walls}
        for n in walls:
            all_t[n] += walls[n]
        cell = lambda n: "%.3f (%.3f to %.3f)" % (med[n], min(walls[n]), max(walls[n]))
        every = set().union(*outs.values())
        print("| %s | %s | %s | %s | %.3f | %d, %d, %d (%d) | %s | %s | %.0f |" % ("free" if a.free else seed, cell("off"), cell("turbo"), cell("turbob0"), med["turbo"] / med["turbob0"],
              len(outs["off"]), len(outs["turbo"]), len(outs["turbob0"]), len(every), same, counts, longest * 1e3), flush=True)
    print()
    for n in (all_t if a.runs else []):
        print("%s: %d runs, median %.3f s, mean %.3f s, min %.3f, max %.3f" % (n, len(all_t[n]), statistics.median(all_t[n]), statistics.mean(all_t[n]), min(all_t[n]), max(all_t[n])))


if __name__ == "__main__":
    main()
