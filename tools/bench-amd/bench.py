#!/usr/bin/env python3
"""Benchmark harness for the AMD (GCN/CDNA/RDNA) interpreter.

Run through tools/bench-amd/bench.sh (which uses `python3 -I`); stdlib only, plus
tools/bench/bench.py for the statistics and the process runner it shares.

What it does
  * compiles tools/bench-amd/kernels/bench_amd.hip once per target with hipcc
    (gfx942 for the simulated MI300X, gfx1030 for the RX 6800) and runs one
    self-checking workload per process through VirtualGPU's libamdhip64;
  * takes the instruction counts of a workload from rocprofv3's SQ_INSTS_*
    counters inside `vgpu shell --no-isolate` (exact and deterministic: they come
    from the interpreter's own DispatchStats), then times N runs without any
    profiler: wave-instructions per CPU second of the device phase;
  * alternates the runs over every (build, thread count) configuration, so a slow
    drift of the machine falls on all of them, and reports median, best and
    spread; with two builds it also gives the ratio and a Mann-Whitney p-value;
  * `--check-counters` compares the counts (never the times) with
    tools/bench-amd/expected/ and fails on any drift.

What "wave-instruction" means here.  The interpreter's own total
(DispatchStats::instructions) is not exposed as a profiler counter, so the count
is the sum of the SQ_INSTS_{SALU,SMEM,VALU,VMEM,LDS,BRANCH,SENDMSG} counters:
every instruction the interpreter classifies, which leaves out the SOPP
instructions that are none of those (s_waitcnt, s_nop, s_endpgm, s_barrier ...).
docs/performance.md ("The AMD engine") says by how much that undercounts.
On a simulated Radeon card, which offers only the counters a real one lists, the count is the interpreter's own
instruction total as the two by-wave-size counters (SQ_WAVE32_INSTS + SQ_WAVE64_INSTS).
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KERNEL_SRC = os.path.join(HERE, "kernels", "bench_amd.hip")
EXPECTED_DIR = os.path.join(HERE, "expected")


def _load_common():
    spec = importlib.util.spec_from_file_location("bench_common", os.path.join(ROOT, "tools", "bench", "bench.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_common()

# target -> (offload arch, VGPU_GPU profile)
TARGETS = {
    "mi300x": ("gfx942", "amd/mi300x"),
    "rx6800": ("gfx1030", "amd/rx6800"),
    "rx7900xtx": ("gfx1100", "amd/rx7900xtx"),   # RDNA3: WMMA
    "rx9070xt": ("gfx1201", "amd/rx9070xt"),     # RDNA4: WMMA
}
DEFAULT_TARGETS = ("mi300x", "rx6800")
# name -> (group, what it measures, targets)
BOTH = ("mi300x", "rx6800")
WORKLOADS = {
    "valu_int": ("alu", "32-bit integer ALU chain: mul, xor, shifts, rotate, popcount (bit tricks)", BOTH),
    "valu_f32": ("alu", "f32 fma chains plus unfused mul/add", BOTH),
    "valu_f64": ("alu", "f64 fma chains", BOTH),
    "mfma": ("matrix", "v_mfma_f32_16x16x16_f16 accumulated in a loop (CDNA only; RDNA2 has no matrix unit)", ("mi300x",)),
    "mfma_i8": ("matrix", "v_mfma_i32_16x16x32_i8 accumulated in a loop (CDNA3)", ("mi300x",)),
    "mfma_fp8": ("matrix", "v_mfma_f32_16x16x32_fp8_fp8 accumulated in a loop (CDNA3)", ("mi300x",)),
    "mfma_bf16": ("matrix", "v_mfma_f32_16x16x16_bf16 accumulated in a loop (CDNA3)", ("mi300x",)),
    "mfma_f64": ("matrix", "v_mfma_f64_16x16x4_f64 accumulated in a loop (CDNA3)", ("mi300x",)),
    "smfmac": ("matrix", "v_smfmac_f32_16x16x32_f16 (2:4 sparse A) accumulated in a loop (CDNA3)", ("mi300x",)),
    "wmma": ("matrix", "v_wmma_f32_16x16x16_f16 accumulated in a loop (RDNA3, RDNA4)", ("rx7900xtx", "rx9070xt")),
    "wmma_iu8": ("matrix", "v_wmma_i32_16x16x16_iu8 accumulated in a loop (RDNA3, RDNA4)", ("rx7900xtx", "rx9070xt")),
    "mem": ("memory", "global memory: 16-byte vector copy, scalar scale, indexed gather", BOTH),
    "lds": ("memory", "LDS with barriers: tree reduction, 32x32 tile transpose", BOTH),
    "atomics": ("memory", "global atomic add (one address, 256 bins), max, a CAS loop", BOTH),
    "divergent": ("control", "data-dependent while loop (collatz) and a switch: divergent branches", BOTH),
    "wave": ("control", "shuffles (ds_bpermute / DPP), ballot, any/all", BOTH),
    "launch": ("api", "kernel launches on two streams (cost = the launch path)", BOTH),
    "api": ("api", "hipMalloc/Memcpy/Memset/Free, events, streams in a loop", BOTH),
    "startup": ("api", "one launch of a trivial kernel; read the whole-process wall time", BOTH),
}
# What an A/B verdict and the rate columns use. Default: wave-instructions per CPU second.
PRIMARY = {"launch": "units_per_s", "api": "units_per_s", "startup": "wall_s"}
NO_COUNTS = ("launch", "api", "startup")
NO_KERNELS = ("api",)  # no dispatch at all, so nothing for the profiler to count  # rate is calls per second; no profiler pass at full size
INSTS = ["SQ_INSTS_SALU", "SQ_INSTS_SMEM", "SQ_INSTS_VALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "SQ_INSTS_BRANCH",
         "SQ_INSTS_SENDMSG", "SQ_INSTS_MFMA"]
EXTRA = ["SQ_WAVES", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "TA_FLAT_ATOMIC_WAVEFRONTS"]
# The counters a simulated Radeon card offers are the ones a real RX 6800 lists (rocprofv3-avail), which has no
# per-class branch, vector-memory or message counters: its instruction total is the two by-wave-size counters.
RDNA_INSTS = ["SQ_INSTS_SALU", "SQ_INSTS_SMEM", "SQ_INSTS_VALU", "SQ_INSTS_FLAT", "SQ_INSTS_LDS", "SQ_WAVE32_INSTS",
              "SQ_WAVE64_INSTS", "SQ_WAVES"]


def die(msg, code=2):
    sys.stderr.write("bench-amd: %s\n" % msg)
    sys.exit(code)


def host_threads(cpus):
    if cpus:
        n = 0
        for part in cpus.split(","):
            a, _, b = part.partition("-")
            n += int(b or a) - int(a) + 1
        return n
    return len(os.sched_getaffinity(0))


# --------------------------------------------------------------------------- build the program

def find_hipcc(explicit):
    for c in [explicit, os.environ.get("HIPCC"), "/opt/rocm/bin/hipcc", shutil.which("hipcc")]:
        if c and os.path.exists(c):
            return c
    die("no hipcc found (set --hipcc or HIPCC)")


def hipcc_version(hipcc):
    r = subprocess.run([hipcc, "--version"], capture_output=True, text=True)
    m = re.search(r"HIP version: (\S+)", r.stdout)
    return m.group(1) if m else (r.stdout.splitlines() or ["?"])[0]


def ensure_binary(arch, cache, hipcc, verbose):
    with open(KERNEL_SRC, "rb") as f:
        src = f.read()
    key = hashlib.sha256(src + arch.encode() + hipcc_version(hipcc).encode()).hexdigest()[:12]
    os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, "bench_amd.%s.%s" % (arch, key))
    if not os.path.exists(out):
        cmd = [hipcc, "-O2", "-std=c++17", "--offload-arch=" + arch, "-ffp-contract=off", KERNEL_SRC, "-o", out + ".tmp"]
        if verbose:
            print("+ " + " ".join(cmd), file=sys.stderr)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            die("hipcc failed for %s:\n%s" % (arch, r.stderr[-2000:]))
        os.replace(out + ".tmp", out)
    return out


# --------------------------------------------------------------------------- run

def clean_env(build, gpu, threads, tmp, extra=()):
    """A clean environment: nothing of the caller's VGPU_* reaches the program."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VGPU_", "HIP_", "ROCM_", "HSA_"))}
    env["LD_LIBRARY_PATH"] = os.path.join(build, "shim")
    env["VGPU_GPU"] = gpu
    env["VGPU_QUIET"] = "1"
    env["VGPU_TELEMETRY_PATH"] = os.path.join(tmp, "telemetry")
    env["VGPU_STATE_DIR"] = os.path.join(tmp, "state")
    if threads != "all":
        env["VGPU_THREADS"] = str(threads)
    for e in extra:
        k, v = e.split("=", 1)
        env[k] = v
    return env


def pin(argv, cpus):
    if cpus and shutil.which("taskset"):
        return ["taskset", "-c", cpus] + argv
    return argv


def parse_bench_line(stdout):
    r = C.parse_bench_line(stdout)
    return r


FAILED_RUNS = []
LOADS = []   # the 1-minute load average before each workload's timed runs


def timed_run(binary, wl, size, build, gpu, threads, tmp, args):
    """One timed run. A run that fails its own check is recorded (with everything it printed) and repeated, up to
    three times: it is reported at the end, never silently dropped, because a result that comes out wrong
    sometimes is a finding, not noise."""
    for attempt in range(3):
        res = C.run_argv(pin([binary, wl, size], args.cpus), clean_env(build, gpu, threads, tmp, args.env), args.timeout)
        ok = res["rc"] == 0 and not res["timeout"] and res["stdout"].rstrip().endswith("PASS")
        if ok:
            b = parse_bench_line(res["stdout"])
            if not b:
                die("%s: no BENCH line" % wl, 1)
            res.update(b)
            return res
        FAILED_RUNS.append({"workload": wl, "gpu": gpu, "threads": threads, "attempt": attempt, "rc": res["rc"],
                            "timeout": res["timeout"], "stdout": res["stdout"][-2000:], "stderr": res["stderr"][-2000:],
                            "loadavg": os.getloadavg()})
    r = FAILED_RUNS[-1]
    die("%s/%s threads=%s failed its check three times:\n    %s" % (wl, gpu, threads,
        "\n    ".join((r["stderr"] + r["stdout"]).strip().splitlines()[-6:])), 1)


def counting_run(binary, wl, size, build, gpu, tmp, args):
    """Counts from rocprofv3 inside `vgpu shell`: summed over the workload's dispatches."""
    out = tempfile.mkdtemp(prefix="cnt.", dir=tmp)
    rdna = gpu.startswith("amd/rx")
    cmds = "rocprofv3 --pmc %s --output-format csv -d %s -- %s %s %s" % (
        " ".join(RDNA_INSTS if rdna else INSTS + EXTRA), out, binary, wl, size)
    argv = pin([os.path.join(build, "vgpu"), "shell", "-y", "--gpu", gpu, "--count", "1", "--no-isolate", "-c", cmds],
               args.cpus)
    res = C.run_argv(argv, clean_env(build, gpu, 1, tmp, args.env), max(args.timeout, 600))
    if res["rc"] != 0 or not res["stdout"].rstrip().endswith("PASS"):
        tail = (res["stderr"] + res["stdout"]).strip().splitlines()[-8:]
        return None, "counting run of %s failed (rc=%s):\n    %s" % (wl, res["rc"], "\n    ".join(tail))
    import csv
    files = []
    for dp, _, fs in os.walk(out):
        files += [os.path.join(dp, f) for f in fs if f.endswith("_counter_collection.csv")]
    if not files:
        return None, "rocprofv3 wrote no counter_collection.csv for %s" % wl
    totals, dispatches = {}, set()
    with open(files[0]) as f:
        for row in csv.DictReader(f):
            totals[row["Counter_Name"]] = totals.get(row["Counter_Name"], 0) + int(float(row["Counter_Value"]))
            dispatches.add(row["Dispatch_Id"])
    totals["dispatches"] = len(dispatches)
    if rdna:
        totals["wave_instructions"] = totals.get("SQ_WAVE32_INSTS", 0) + totals.get("SQ_WAVE64_INSTS", 0)
    else:
        totals["wave_instructions"] = sum(totals.get(n, 0) for n in INSTS if n != "SQ_INSTS_MFMA")  # MFMA is inside VALU
    shutil.rmtree(out, ignore_errors=True)
    return totals, ""


# --------------------------------------------------------------------------- flow

def selected(args):
    names = [w for w in WORKLOADS] if not args.workloads else args.workloads.split(",")
    for n in names:
        if n not in WORKLOADS:
            die("unknown workload %s (have: %s)" % (n, ", ".join(WORKLOADS)))
    targets = args.target.split(",") if args.target != "all" else list(DEFAULT_TARGETS)
    for t in targets:
        if t not in TARGETS:
            die("unknown target %s (have: %s)" % (t, ", ".join(TARGETS)))
    return names, targets


def meta_block(args, builds, hipcc):
    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"load1_before_each_workload": LOADS, "date": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "affinity_cpus": len(os.sched_getaffinity(0)), "cpus_arg": args.cpus,
            "hipcc": hipcc, "hipcc_version": hipcc_version(hipcc), "builds": builds,
            "runs": args.runs, "size": args.size, "threads": args.threads, "python": platform.python_version()}


def timing_flow(names, targets, builds, thread_list, counts, bins, args, tmp):
    """results[(wl, target)][(build, threads)] = list of runs, alternating over the configurations."""
    results = {}
    configs = [(b, t) for b in builds for t in thread_list]
    for wl in names:
        for tg in targets:
            if tg not in WORKLOADS[wl][2]:
                continue
            arch, gpu = TARGETS[tg]
            LOADS.append(os.getloadavg()[0])
            runs = {c: [] for c in configs}
            for c in configs:  # one warm-up per configuration (page cache, CPU frequency)
                timed_run(bins[arch], wl, args.size, c[0], gpu, c[1], tmp, args)
            for i in range(args.runs):
                for c in configs:
                    runs[c].append(timed_run(bins[arch], wl, args.size, c[0], gpu, c[1], tmp, args))
            results[(wl, tg)] = runs
            if not args.quiet:
                sys.stderr.write("  timed %s/%s\n" % (wl, tg))
    return results


def metric(wl, run, instr):
    p = PRIMARY.get(wl, "winst")
    if p == "units_per_s":
        return run["units"] / run["device_wall_s"] if run["device_wall_s"] else 0.0
    if p == "wall_s":
        return run["wall"]
    # (without the counting runs the count is taken as 1: a rate in arbitrary units, whose ratio between two builds is the
    # ratio of their speeds, the same work being done by both)
    return (instr or 1) / run["device_cpu_s"] if run["device_cpu_s"] else 0.0


def fmt(st, scale=1e6, nd=2):
    return "%.*f" % (nd, st / scale)


def report(names, targets, builds, thread_list, results, counts, args, out=sys.stdout):
    rows = []
    for wl in names:
        for tg in targets:
            runs = results.get((wl, tg))
            if not runs:
                continue
            cnt = counts.get((wl, tg, builds[0])) or {}
            instr = cnt.get("wave_instructions", 0)
            for c, rs in runs.items():
                vals = [metric(wl, r, instr) for r in rs]
                hi = PRIMARY.get(wl) != "wall_s"
                st = C.stat(vals, hi)
                wallrate = C.stat([instr / r["device_wall_s"] for r in rs if r["device_wall_s"] > 0], True) if instr else None
                cpu = C.stat([r["device_cpu_s"] for r in rs])
                wall = C.stat([r["wall"] for r in rs])
                rows.append({"workload": wl, "target": tg, "build": c[0], "threads": c[1], "instr": instr, "stat": st,
                             "wallrate": wallrate, "cpu": cpu, "wall": wall, "primary": PRIMARY.get(wl, "winst"),
                             "unit": rs[0]["unit"], "dispatches": cnt.get("dispatches", 0)})
    ab = {}
    if len(builds) == 2:
        for (wl, tg), runs in results.items():
            instr = (counts.get((wl, tg, builds[0])) or {}).get("wave_instructions", 0)
            for t in thread_list:
                a = [metric(wl, r, instr) for r in runs[(builds[0], t)]]
                b = [metric(wl, r, instr) for r in runs[(builds[1], t)]]
                p = C.mann_whitney_p(a, b)
                ma, mb = statistics.median(a), statistics.median(b)
                ba, bb = (max(a), max(b)) if PRIMARY.get(wl) != "wall_s" else (min(a), min(b))
                ratio_m, ratio_b = (mb / ma if ma else 0), (bb / ba if ba else 0)
                if PRIMARY.get(wl) == "wall_s":
                    ratio_m, ratio_b = (ma / mb if mb else 0), (ba / bb if bb else 0)  # >1 is faster
                if p <= 0.05 and ratio_m > 1.02 and ratio_b > 1.02:
                    verdict = "faster"
                elif p <= 0.05 and ratio_m < 0.98 and ratio_b < 0.98:
                    verdict = "slower"
                else:
                    verdict = "same"
                same = ((counts.get((wl, tg, builds[0])) or {}) == (counts.get((wl, tg, builds[1])) or {}))
                ab[(wl, tg, t)] = {"p": p, "ratio_median": ratio_m, "ratio_best": ratio_b, "verdict": verdict,
                                   "counters_identical": same}
    hdr = "| workload | target | threads | wave-instr | rate median | best | spread | rate when wall-clocked | cpu s | wall s |"
    print(hdr, file=out)
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", file=out)
    for r in rows:
        if len(builds) == 2 and r["build"] != builds[0]:
            continue
        st = r["stat"]
        unit = {"winst": "Minst/s", "units_per_s": "k%s/s" % r["unit"], "wall_s": "s wall"}[r["primary"]]
        scale = {"winst": 1e6, "units_per_s": 1e3, "wall_s": 1.0}[r["primary"]]
        nd = 3 if r["primary"] == "wall_s" else 2
        wr = ("%.2f" % (r["wallrate"]["median"] / 1e6)) if r["wallrate"] and r["threads"] != 1 else ""
        print("| %s | %s | %s | %s | %s %s | %s | %.0f%% | %s | %.3f | %.3f |" % (
            r["workload"], r["target"], r["threads"], r["instr"] or "", fmt(st["median"], scale, nd), unit,
            fmt(st["best"], scale, nd), st["spread_pct"], wr, r["cpu"]["median"], r["wall"]["median"]), file=out)
    if ab:
        print("\nA = %s, B = %s (ratio > 1 means B is faster)\n" % (builds[0], builds[1]), file=out)
        print("| workload | target | threads | median ratio | best ratio | p | counters identical | verdict |", file=out)
        print("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |", file=out)
        for (wl, tg, t), v in ab.items():
            print("| %s | %s | %s | %.3f | %.3f | %.3f | %s | %s |" % (
                wl, tg, t, v["ratio_median"], v["ratio_best"], v["p"], "yes" if v["counters_identical"] else "NO",
                v["verdict"]), file=out)
    return rows, ab


# --------------------------------------------------------------------------- counters check

def expected_path(hv):
    return os.path.join(EXPECTED_DIR, "amd-hip-%s.json" % re.sub(r"[^0-9.]", "", hv.split("-")[0]))


def compact(counts):
    keep = INSTS + EXTRA + RDNA_INSTS + ["dispatches", "wave_instructions"]
    return {k: counts[k] for k in keep if k in counts}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="append", help="build directory (twice for an A/B comparison)")
    p.add_argument("--target", default="all", help="mi300x, rx6800, rx7900xtx, rx9070xt (comma list), or all (default: mi300x and rx6800)")
    p.add_argument("--workloads", help="comma-separated; default all: " + ",".join(WORKLOADS))
    p.add_argument("--runs", type=int, default=9)
    p.add_argument("--size", default="full", choices=("quick", "full"))
    p.add_argument("--threads", default="1,all", help="comma list of VGPU_THREADS values; 'all' leaves it unset (default 1,all)")
    p.add_argument("--cpus", help="pin every run to this CPU list (taskset -c)")
    p.add_argument("--env", action="append", default=[], help="extra K=V for the program")
    p.add_argument("--hipcc")
    p.add_argument("--cache", default=os.path.expanduser("~/.cache/vgpu-bench-amd"))
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--json", help="write everything to this file")
    p.add_argument("--no-counts", action="store_true", help="skip the rocprofv3 counting runs")
    p.add_argument("--check-counters", action="store_true", help="compare counts with tools/bench-amd/expected (quick size)")
    p.add_argument("--update-expected", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    builds = [os.path.abspath(b) for b in (args.build or [os.path.join(ROOT, "build")])]
    if len(builds) > 2:
        die("at most two --build")
    for b in builds:
        if not (os.path.exists(os.path.join(b, "vgpu")) and os.path.exists(os.path.join(b, "shim", "libamdhip64.so.7"))):
            die("%s has no vgpu / shim/libamdhip64.so.7; build the vgpu and vgpuhip targets" % b)
    hipcc = find_hipcc(args.hipcc)
    names, targets = selected(args)
    check = args.check_counters or args.update_expected
    if check:
        args.size = "quick"
    thread_list = [int(t) if t != "all" else "all" for t in args.threads.split(",")]
    tmp = tempfile.mkdtemp(prefix="bench-amd.")
    try:
        bins = {TARGETS[t][0]: ensure_binary(TARGETS[t][0], args.cache, hipcc, not args.quiet) for t in targets}
        counts = {}
        if not args.no_counts or check:
            for wl in names:
                if (wl in NO_COUNTS and not check) or wl in NO_KERNELS:
                    continue
                for tg in targets:
                    if tg not in WORKLOADS[wl][2]:
                        continue
                    for b in builds:
                        cnt, err = counting_run(bins[TARGETS[tg][0]], wl, args.size, b, TARGETS[tg][1], tmp, args)
                        if cnt is None:
                            die(err, 1)
                        counts[(wl, tg, b)] = cnt
                        if not args.quiet:
                            sys.stderr.write("  counted %s/%s: %d wave-instructions\n" % (wl, tg, cnt["wave_instructions"]))
        if len(builds) == 2:
            for k, v in list(counts.items()):
                if k[2] == builds[0] and counts.get((k[0], k[1], builds[1])) != v:
                    sys.stdout.write("COUNTER DRIFT %s/%s between the two builds\n" % (k[0], k[1]))
        if check:
            path = expected_path(hipcc_version(hipcc))
            hv = hipcc_version(hipcc)
            new = {"hipcc": hv, "size": "quick", "counts": {"%s/%s" % (k[1], k[0]): compact(v) for k, v in counts.items()
                                                            if k[2] == builds[0]}}
            if args.update_expected:
                os.makedirs(EXPECTED_DIR, exist_ok=True)
                with open(path, "w") as f:
                    json.dump(new, f, indent=1, sort_keys=True)
                    f.write("\n")
                print("wrote " + path)
                return 0
            if not os.path.exists(path):
                print("SKIP: no expected file for hipcc %s (have: %s)" % (hv, ", ".join(sorted(os.listdir(EXPECTED_DIR))) if os.path.isdir(EXPECTED_DIR) else "none"))
                return 0
            with open(path) as f:
                exp = json.load(f)
            bad = []
            for key, got in new["counts"].items():
                want = exp["counts"].get(key)
                if want is None:
                    bad.append("%s: no expected entry" % key)
                    continue
                for k in sorted(set(got) | set(want)):
                    if got.get(k) != want.get(k):
                        bad.append("%s: %s expected %s, got %s" % (key, k, want.get(k), got.get(k)))
            if bad:
                print("counter drift:\n  " + "\n  ".join(bad))
                return 1
            print("counters match %s (%d workload/target pairs)" % (os.path.basename(path), len(new["counts"])))
            return 0
        results = timing_flow(names, targets, builds, thread_list, counts, bins, args, tmp)
        meta = meta_block(args, builds, hipcc)
        print("<!-- host %s, %s, hipcc %s, 1-minute load average before each workload: min %.1f median %.1f max %.1f -->" % (
              meta["host"], meta["cpu"], meta["hipcc_version"], min(LOADS), statistics.median(LOADS), max(LOADS)))
        rows, ab = report(names, targets, builds, thread_list, results, counts, args)
        if FAILED_RUNS:
            print("\nNOTE: %d timed run(s) failed their own check and were repeated:" % len(FAILED_RUNS))
            for r in FAILED_RUNS:
                print("  %s on %s, threads=%s, attempt %d, rc=%s, load %s: %s" % (
                    r["workload"], r["gpu"], r["threads"], r["attempt"], r["rc"], "/".join("%.1f" % x for x in r["loadavg"]),
                    " | ".join((r["stderr"] + r["stdout"]).strip().splitlines()[-3:])))
        if args.json:
            dump = {"meta": meta,
                    "counts": {"%s/%s/%s" % (k[1], k[0], k[2]): v for k, v in counts.items()},
                    "rows": [{k: v for k, v in r.items()} for r in rows],
                    "ab": {"%s/%s/%s" % k: v for k, v in ab.items()}, "failed_runs": FAILED_RUNS}
            with open(args.json, "w") as f:
                json.dump(dump, f, indent=1, default=str)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
