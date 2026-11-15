#!/usr/bin/env python3
"""Round sp76: Rodinia particlefilter_float run by run. Its normalize_weights_kernel has a race between blocks (block 0 writes u[0], every
other block reads it, and nothing orders the two), so with several host threads the program does not compute the same thing twice, clock
seed fixed or not, and the launch after it (find_index_kernel) takes the time the values it is given ask.

Runs the program --runs times a form, the forms interleaved, in turbo mode with VGPU_HOST_GPU_LOG=1, and prints per run the simulator's time
for the first launch of find_index_kernel (launch_s of its sample's line), the program's answer and its wall time; then per form how the
runs fall. Forms: `turbo`; `turbo1`, the same with VGPU_HOST_GPU_SAMPLE_BLOCK=1 (the rule on, the environment the size of turbob0's);
`turbob0`, VGPU_HOST_GPU_SAMPLE_BLOCK=0 (the rule of before Round sp75).

  pf_race.py --build BUILD --corp CORP [--stub libstubcuda.so] [--fixtime libfixtime.so --seed N] [--threads 4] [--runs 20]
  pf_race.py --summary TABLE     the table per form again, from the per-run rows of an earlier run
"""
import argparse, fcntl, os, re, statistics, subprocess, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="")
    ap.add_argument("--corp", default="")
    ap.add_argument("--summary", default="")
    ap.add_argument("--stub", default="")
    ap.add_argument("--fixtime", default="")
    ap.add_argument("--seed", default="1763200000")
    ap.add_argument("--threads", default="4")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    a = ap.parse_args()
    forms = [("turbo", {}), ("turbo1", {"VGPU_HOST_GPU_SAMPLE_BLOCK": "1"}), ("turbob0", {"VGPU_HOST_GPU_SAMPLE_BLOCK": "0"})]
    rows = {n: [] for n, _ in forms}
    if a.summary:
        for line in open(a.summary):
            p = [x.strip() for x in line.split("|")]
            if len(p) > 7 and p[1].isdigit():
                rows[p[2]].append((float(p[3]), float(p[7]), p[6]))
        return summary(forms, rows)
    exe = os.path.join(a.corp, "rodinia/bin/particlefilter_float")
    argv = ["nice", "-n", "19", exe] + "-x 128 -y 128 -z 10 -np 10000".split()
    base = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_") and not k.startswith("STUB_")}
    base.update(VGPU_GPU="nvidia/a10g", LD_LIBRARY_PATH=os.path.join(a.build, "shim"), VGPU_HOST_GPU="turbo", VGPU_HOST_GPU_LOG="1")
    if a.threads:
        base.update(VGPU_THREADS=a.threads)
    if a.stub:
        base.update(VGPU_HOST_GPU_LIBCUDA=a.stub, STUB_SM="86", STUB_MP="80")
    if a.fixtime:
        base.update(LD_PRELOAD=a.fixtime, FIXTIME=a.seed)
    fd, since = None, 0.0
    print("threads %s, clock %s, %d runs a form\n" % (a.threads or "the host's", "fixed at " + a.seed if a.fixtime else "free", a.runs))
    print("| run | form | find_index_kernel, first launch ms | its sample's estimate ms | launches on the card | answer (XE, YE) | wall s |")
    print("|---:|---|---:|---:|---:|---|---:|")
    for i in range(a.runs):
        for n, extra in forms[i % 3:] + forms[:i % 3]:
            if fd is not None and time.monotonic() - since > 150:   # the machine's build lock, given up now and then
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                fd = None
                time.sleep(1)
            if fd is None:
                fd = os.open(a.lock, os.O_RDWR | os.O_CREAT, 0o666)
                fcntl.flock(fd, fcntl.LOCK_EX)
                since = time.monotonic()
            t0 = time.monotonic()
            p = subprocess.run(argv, cwd=os.path.dirname(exe), env=dict(base, **extra), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            wall = time.monotonic() - t0
            err, out = p.stderr.decode("utf-8", "replace"), p.stdout.decode("utf-8", "replace")
            m = re.search(r"^host_gpu_sample: kernel=\S*find_index\S* .* estimate_s=(\S+) .* launch_s=(\S+)", err, re.M)
            first, est = (float(m.group(2)) * 1e3, float(m.group(1)) * 1e3) if m else (-1.0, -1.0)
            c = re.search(r"^host_gpu: (\d+) launches ran on the card", err, re.M)
            ans = " ".join(re.findall(r"^[XY]E: (\S+)", out, re.M))
            rows[n].append((first, wall, ans))
            print("| %d | %s | %.1f | %.1f | %s | %s | %.3f |" % (i + 1, n, first, est, c.group(1) if c else "?", ans, wall), flush=True)
    print()
    summary(forms, rows)


def summary(forms, rows):
    every = sorted(x[0] for v in rows.values() for x in v if x[0] > 0)
    # the two groups: cut at the widest gap (as a ratio) between two first-launch times next to each other
    gap, cut = max((every[i + 1] / every[i], (every[i + 1] * every[i]) ** 0.5) for i in range(len(every) - 1)) if len(every) > 1 else (1.0, 0.0)
    if gap < 1.5:   # no gap to speak of: one group
        cut = 2 * every[-1] if every else 0.0
        print("The first launch of find_index_kernel takes %.1f to %.1f ms: one group (no two times next to each other are a factor of 1.5 apart).\n" % (every[0], every[-1]))
    else:
        print("The first launch of find_index_kernel falls in two groups, cut at %.0f ms (the widest gap between two times next to each other, a factor of %.1f).\n" % (cut, gap))
    print("| form | runs | short first launch | long first launch | short: ms | long: ms | answers seen | most frequent answer, runs | wall s, median | wall s, mean | wall s of the runs with a short / a long first launch, median |")
    print("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|")
    for n, _ in forms:
        v = rows[n]
        if not v:
            continue
        lo, hi = [x for x in v if x[0] < cut], [x for x in v if x[0] >= cut]
        span = lambda g: "%.0f to %.0f" % (min(x[0] for x in g), max(x[0] for x in g)) if g else "-"
        ans = {}
        for x in v:
            ans[x[2]] = ans.get(x[2], 0) + 1
        med = lambda g: "%.3f" % statistics.median(x[1] for x in g) if g else "-"
        print("| %s | %d | %d | %d | %s | %s | %d | %d | %.3f | %.3f | %s / %s |" % (n, len(v), len(lo), len(hi), span(lo), span(hi), len(ans), max(ans.values()),
              statistics.median(x[1] for x in v), statistics.mean(x[1] for x in v), med(lo), med(hi)))


if __name__ == "__main__":
    main()
