#!/usr/bin/env python3
"""Simulator benchmark harness.

Run through tools/bench/bench.sh (which uses `python3 -I`); stdlib only.

What it does
  * builds tools/bench/kernels/bench.cu once (nvcc matching the shim's toolkit
    major) and runs one self-checking workload per process, on the SASS engine
    (the default) and on the PTX interpreter (VGPU_SASS=0);
  * for each workload and engine takes the instruction counters from
    VGPU_COUNTERS (exact, deterministic) and times N runs without counters:
    warp-instructions per CPU second, wall seconds, peak RSS;
  * `--ab A B` alternates the runs A B A B ... of the SAME program against two
    build directories and reports the median ratio, the spread, and whether the
    counters of A and B are identical;
  * `--check-counters` compares the counters (never the times) with the files in
    tools/bench/expected and fails on any drift;
  * `mix` prints the instruction-class mix of the workloads (or of any command).

Every number printed comes from a run made by this script on this machine; the
`meta` block of the JSON records the host, the load and the toolchain so a
number is never separated from where it was measured.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KERNEL_SRC = os.path.join(HERE, "kernels", "bench.cu")
EXPECTED_DIR = os.path.join(HERE, "expected")

# name -> (group, what it measures)
WORKLOADS = {
    "int_alu": ("alu", "32-bit integer ALU: imad, lop3, shifts, popc, rem"),
    "f32_fma": ("alu", "f32 fma chains, normal numbers, plus unfused add/mul"),
    "f64_fma": ("alu", "f64 fma chains, normal numbers"),
    "f64_subnormal": ("alu", "pantheon fp64_virus chain: decays into subnormals"),
    "wmma": ("tensor", "nvcuda::wmma 16x16x16 f16 -> f32"),
    "mma_ptx": ("tensor", "mma.sync.m16n8k16 f16 -> f32, fragments by the PTX ISA layout"),
    "mem": ("memory", "global loads/stores: scalar, 16-byte vector, uncoalesced gather"),
    "shared": ("memory", "shared memory with barriers: tree reduction, tile transpose"),
    "atomics": ("memory", "atomicAdd/Max/CAS, contended and spread, 32/64-bit, f32"),
    "texture": ("memory", "tex2D point + linear, tex1Dfetch"),
    "local": ("memory", "per-thread array in local memory (LDL/STL)"),
    "divergent": ("control", "data-dependent loops and switch: divergent control flow"),
    "warp": ("control", "shuffles, ballot, any/all"),
    "launch": ("api", "thousands of one-warp launches (cost = the launch path)"),
    "api": ("api", "cudaMalloc/Memcpy/Memset/Free, events, streams in a loop"),
    "startup": ("api", "context + module + one launch (process cost excluded from the device phase)"),
}
ENGINES = ("sass", "ptx")
GPU_DEFAULT = "nvidia/rtx3060"  # sm_86: the arch bench.cu is compiled for (SASS and PTX)
NVCC_ARCH = "sm_86"

CLASS_ORDER = ["fp16", "fp32", "fp64", "int", "cvt", "ctrl", "mem", "tensor", "misc"]


def die(msg, code=2):
    print("bench: " + msg, file=sys.stderr)
    sys.exit(code)


# --------------------------------------------------------------------------- toolchain


def shim_major(build):
    """The CUDA major the build's runtime shim was built for (libcudart.so.N), or None."""
    shim = os.path.join(build, "shim")
    try:
        names = os.listdir(shim)
    except OSError:
        return None
    for n in sorted(names):
        m = re.fullmatch(r"libcudart\.so\.(\d+)", n)
        if m:
            return int(m.group(1))
    return None


def nvcc_release(nvcc):
    """('13.0', '13.0.88') from `nvcc --version`, or None."""
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", out)
    return (m.group(1), m.group(2)) if m else None


def find_nvcc(major, explicit=None):
    cands = []
    if explicit:
        cands.append(explicit)
    cands.append(os.environ.get("VGPU_NVCC", ""))
    cands.append(shutil.which("nvcc") or "")
    cands += ["/usr/bin/nvcc", "/usr/local/cuda/bin/nvcc"]
    if os.path.isdir("/usr/local"):
        for d in sorted(os.listdir("/usr/local"), reverse=True):
            if d.startswith("cuda-"):
                cands.append("/usr/local/%s/bin/nvcc" % d)
    seen = set()
    for c in cands:
        if not c or c in seen or not os.access(c, os.X_OK):
            continue
        seen.add(c)
        rel = nvcc_release(c)
        if rel and (major is None or int(rel[0].split(".")[0]) == major):
            return c, rel
    return None, None


def host_probe():
    """What the host lets us count. Hosted runners are VMs: do not assume a PMU."""
    info = {"perf_event_paranoid": None, "perf": None, "hw_instructions": "not probed"}
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            info["perf_event_paranoid"] = int(f.read().strip())
    except (OSError, ValueError):
        pass
    perf = os.environ.get("PERF") or shutil.which("perf")
    info["perf"] = perf
    if perf:
        try:
            r = subprocess.run([perf, "stat", "-x,", "-e", "instructions:u", "true"],
                               capture_output=True, text=True, timeout=30)
            txt = (r.stderr + r.stdout)
            if "not supported" in txt or "not counted" in txt or r.returncode != 0:
                info["hw_instructions"] = "not supported on this host"
            else:
                info["hw_instructions"] = "available"
        except (OSError, subprocess.SubprocessError):
            info["hw_instructions"] = "perf failed"
    else:
        info["hw_instructions"] = "no perf binary"
    return info


def meta_block(args, builds, nvcc, rel):
    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    rev = ""
    try:
        rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node() if args.record_hostname else "(not recorded)",
        "kernel": platform.release(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "loadavg_start": os.getloadavg(),
        "python": platform.python_version(),
        "nvcc": nvcc,
        "toolkit": rel[0] if rel else None,
        "nvcc_version": rel[1] if rel else None,
        "gpu_profile": args.gpu,
        "compiled_arch": NVCC_ARCH,
        "vgpu_threads": args.threads,
        "size": args.size,
        "runs": args.runs,
        "warmup": args.warmup,
        "extra_env": dict(e.split("=", 1) for e in args.env),
        "source_rev": rev,
        "host_counters": host_probe(),
        "builds": {k: v for k, v in builds.items()},
    }


# --------------------------------------------------------------------------- build the program


def ensure_binary(build, args, rel_nvcc):
    nvcc, rel = rel_nvcc
    with open(KERNEL_SRC, "rb") as f:
        src = f.read()
    san = shim_sanitizer_flags(build)
    key = hashlib.sha256(src + (nvcc + rel[1] + NVCC_ARCH + " ".join(san)).encode()).hexdigest()[:16]
    cache = args.cache or os.environ.get("VGPU_BENCH_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "vgpu-bench")
    try:
        os.makedirs(cache, exist_ok=True)
    except OSError:
        cache = tempfile.mkdtemp(prefix="vgpu-bench-")
    out = os.path.join(cache, "bench_kernels.%s.%s" % (rel[1], key))
    if os.path.exists(out):
        return out
    cmd = [nvcc, "-std=c++17", "-O2", "-arch=" + NVCC_ARCH, "-cudart", "shared",
           "-Xcompiler", "-ffp-contract=off", "-Wno-deprecated-gpu-targets"] + san + \
          [KERNEL_SRC, "-o", out + ".tmp"]
    env = dict(os.environ)
    if int(rel[0].split(".")[0]) < 13 and not env.get("NVCC_PREPEND_FLAGS") and shutil.which("g++-12"):
        env["NVCC_PREPEND_FLAGS"] = "-ccbin g++-12"  # same workaround as tests/shim_guard.sh
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        die("nvcc failed:\n" + r.stdout + r.stderr, 1)
    os.replace(out + ".tmp", out)
    print("bench: built %s with nvcc %s in %.1f s" % (os.path.basename(out), rel[1],
                                                       time.monotonic() - t0), file=sys.stderr)
    return out


def shim_sanitizer_flags(build):
    """-fsanitize flags when the shim is sanitizer-built (as tests/shim_guard.sh does)."""
    shim = os.path.join(build, "shim")
    try:
        names = [n for n in os.listdir(shim) if re.fullmatch(r"libcudart\.so\.\d+", n)]
        if not names:
            return []
        out = subprocess.run(["objdump", "-p", os.path.join(shim, names[0])], capture_output=True,
                             text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    if "libtsan" in out:
        return ["-Xcompiler", "-fsanitize=thread", "-g"]
    if "libasan" in out:
        return ["-Xcompiler", "-fsanitize=address", "-Xcompiler", "-fsanitize=undefined",
                "-Xcompiler", "-fno-omit-frame-pointer", "-g"]
    return []


# --------------------------------------------------------------------------- running a workload


def clean_env(build, args, engine, counters):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    shim = os.path.join(os.path.abspath(build), "shim")
    env["LD_LIBRARY_PATH"] = shim + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["VGPU_QUIET"] = "1"
    env["VGPU_GPU"] = args.gpu
    env["VGPU_THREADS"] = str(args.threads)
    if engine == "ptx":
        env["VGPU_SASS"] = "0"
    if counters:
        env["VGPU_COUNTERS"] = "1"
        env["VGPU_SASS_LOG"] = "1"
    for e in args.env:
        k, v = e.split("=", 1)
        env[k] = v
    return env


def _poll_hwm(pid, binary, stop, box):
    """Peak RSS of the child as the kernel keeps it (VmHWM), sampled while it runs.

    wait4's ru_maxrss is not usable here: a child forked from this Python
    process starts with the parent's resident size folded into its maximum
    (measured: every run reported the same 62 MB). VmHWM belongs to the new
    address space only. A run shorter than one poll interval may be missed
    (box stays empty) and is reported from ru_maxrss as an upper bound.
    """
    path = "/proc/%d/status" % pid
    while not stop.is_set():
        try:
            with open("/proc/%d/cmdline" % pid, "rb") as f:
                started = f.read().split(b"\0")[0] == binary.encode()
            if started:
                with open(path) as f:
                    for line in f:
                        if line.startswith("VmHWM:"):
                            kb = int(line.split()[1])
                            if kb > box.get("kb", 0):
                                box["kb"] = kb
                            break
        except (OSError, ValueError):
            pass
        stop.wait(0.002)


def run_process(binary, workload, size, env, timeout):
    """Run once; returns wall/cpu from wait4 (this child only), peak RSS, and its output."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        t0 = time.monotonic()
        p = subprocess.Popen([binary, workload, size], env=env, stdout=out, stderr=err,
                             stdin=subprocess.DEVNULL)
        killed = []

        def kill():
            killed.append(1)
            p.kill()

        timer = threading.Timer(timeout, kill)
        timer.start()
        stop, box = threading.Event(), {}
        poller = threading.Thread(target=_poll_hwm, args=(p.pid, binary, stop, box), daemon=True)
        poller.start()
        _, status, ru = os.wait4(p.pid, 0)
        wall = time.monotonic() - t0
        stop.set()
        poller.join()
        timer.cancel()
        p.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        return {
            "rc": p.returncode, "timeout": bool(killed), "wall": wall,
            "cpu": ru.ru_utime + ru.ru_stime, "user": ru.ru_utime, "sys": ru.ru_stime,
            "rss_kb": box.get("kb", ru.ru_maxrss), "rss_sampled": "kb" in box,
            "stdout": out.read().decode("utf-8", "replace"),
            "stderr": err.read().decode("utf-8", "replace"),
        }


def parse_bench_line(stdout):
    for line in stdout.splitlines():
        if line.startswith("BENCH "):
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            return {"device_cpu_s": float(kv["device_cpu_s"]), "device_wall_s": float(kv["device_wall_s"]),
                    "units": int(kv["units"]), "unit": kv["unit"]}
    return None


def checked(res, name):
    """PASS is printed by the program only when every result matched its host reference."""
    ok = res["rc"] == 0 and not res["timeout"] and res["stdout"].rstrip().endswith("PASS")
    if not ok:
        tail = (res["stderr"] + res["stdout"]).strip().splitlines()[-6:]
        return False, "%s: %s\n    %s" % (name, "timed out" if res["timeout"] else "rc=%s, no PASS" % res["rc"],
                                         "\n    ".join(tail))
    return True, ""


# counters --------------------------------------------------------------------

BLOCK_RE = re.compile(r"^\[vgpu\]\[counters\] (\S+)\s+grid=(\d+)x(\d+)x(\d+) block=(\d+)x(\d+)x(\d+)")


def _kv(line):
    return {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}


def parse_counters(stderr):
    """Per-kernel sums of the VGPU_COUNTERS blocks (format: src/runtime/runtime.cpp report_counters).

    Returns (kernels, engine_log) where kernels maps a kernel name to the sum of
    its launches' counters, plus `launches`. A block this parser cannot read is
    an error, not a silent zero.
    """
    kernels = {}
    engine = []
    cur = None
    for raw in stderr.splitlines():
        line = raw.strip()
        if line.startswith("[vgpu] running "):
            msg = line[len("[vgpu] "):]
            kind = "PTX" if msg.startswith("running PTX") else "SASS"
            if (kind, msg) not in engine and len(engine) < 8:
                engine.append((kind, msg))
            continue
        m = BLOCK_RE.match(line)
        if m:
            cur = kernels.setdefault(m.group(1), {"launches": 0})
            cur["launches"] += 1
            continue
        if cur is None:
            continue

        def add(key, val):
            cur[key] = cur.get(key, 0) + val

        if line.startswith("blocks="):
            for k, v in _kv(line).items():
                add(k, v)
        elif line.startswith("inst_executed="):
            d = _kv(line)
            add("instructions", d["inst_executed"])
            add("thread_instructions", d["thread_inst_executed"])
        elif line.startswith("divergent_branches="):
            d = _kv(line)
            add("divergent_branches", d["divergent_branches"])
            add("barriers", d["barriers"])
            add("atomics", d["atomics"])
            m2 = re.search(r"\((\d+) B\)", line)
            add("atomic_bytes", int(m2.group(1)) if m2 else 0)
        elif re.match(r"(global|shared|local)\s+ld=", line):
            sp = line.split()[0]
            d = _kv(line)
            for k in ("ld", "st", "read", "write"):
                add("%s_%s" % (sp, k if k in ("ld", "st") else k + "_bytes"), d.get(k, 0))
        elif line.startswith("sectors global="):
            m2 = re.match(r"sectors global=(\d+) over (\d+) requests", line)
            add("global_sectors", int(m2.group(1)))
            add("global_requests", int(m2.group(2)))
        elif line.startswith("shared  bank_conflicts=") or line.startswith("shared bank_conflicts="):
            m2 = re.match(r"shared\s+bank_conflicts=(\d+) over (\d+) requests", line)
            add("shared_bank_conflicts", int(m2.group(1)))
            add("shared_requests", int(m2.group(2)))
        elif line.startswith("mix "):
            for k, v in _kv(line).items():
                add("mix_" + k, v)
        elif line.startswith("ctrl="):
            d = _kv(line)  # ctrl= mem= tensor= misc= (tensor issues=N)
            for k in ("ctrl", "mem", "tensor", "misc"):
                add("mix_" + k, d[k])
            add("tensor_issues", d["issues"])
        elif line.startswith("by opcode"):
            # PTX only, truncated to the ten most used: kept per kernel for `mix`, not compared
            ops = cur.setdefault("_opcodes", {})
            for name, n in re.findall(r"(\S+)=(\d+)", line[len("by opcode"):]):
                ops[name] = ops.get(name, 0) + int(n)
    for name, k in kernels.items():
        if "instructions" not in k or "mix_ctrl" not in k:
            raise ValueError("counter block for %s could not be read: the format in "
                             "src/runtime/runtime.cpp report_counters changed?" % name)
    return kernels, engine


def totals(kernels):
    tot = {}
    for k in kernels.values():
        for key, v in k.items():
            if key.startswith("_"):
                continue
            tot[key] = tot.get(key, 0) + v
    return tot


def counting_run(binary, workload, engine, build, args):
    env = clean_env(build, args, engine, True)
    res = run_process(binary, workload, args.size_for_counters, env, args.timeout)
    ok, why = checked(res, "%s/%s" % (workload, engine))
    if not ok:
        return None, why
    try:
        kernels, log = parse_counters(res["stderr"])
    except ValueError as e:
        return None, "%s/%s: %s" % (workload, engine, e)
    bench = parse_bench_line(res["stdout"]) or {}
    fell_back = [m for kind, m in log if kind == "PTX"]
    return {"kernels": kernels, "totals": totals(kernels), "engine_log": [m for _, m in log],
            "fell_back_to_ptx": fell_back, "units": bench.get("units", 0), "unit": bench.get("unit", "none")}, ""


# statistics ------------------------------------------------------------------


def stat(xs):
    if not xs:
        return None
    med = statistics.median(xs)
    lo, hi = min(xs), max(xs)
    return {"median": med, "min": lo, "max": hi, "n": len(xs),
            "spread_pct": (100.0 * (hi - lo) / med) if med else 0.0}


def summarize_runs(runs, instructions, units):
    s = {
        "wall_s": stat([r["wall"] for r in runs]),
        "cpu_s": stat([r["cpu"] for r in runs]),
        "rss_mb": stat([r["rss_kb"] / 1024.0 for r in runs]),
        "device_cpu_s": stat([r["device_cpu_s"] for r in runs]),
        "device_wall_s": stat([r["device_wall_s"] for r in runs]),
    }
    if instructions:
        s["winst_per_cpu_s"] = stat([instructions / r["device_cpu_s"] for r in runs if r["device_cpu_s"] > 0])
    if units:
        s["units_per_s"] = stat([units / r["device_wall_s"] for r in runs if r["device_wall_s"] > 0])
    return s


# --------------------------------------------------------------------------- the main flows


def selected_workloads(args):
    names = []
    if args.workloads:
        for w in args.workloads.split(","):
            w = w.strip()
            if w in WORKLOADS:
                names.append(w)
            elif w in set(g for g, _ in WORKLOADS.values()):
                names += [n for n, (g, _) in WORKLOADS.items() if g == w]
            else:
                die("unknown workload or group '%s' (workloads: %s; groups: alu tensor memory control api)"
                    % (w, ", ".join(WORKLOADS)))
    else:
        names = list(WORKLOADS)
    seen = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen


def setup(args):
    builds = {}
    dirs = [args.build] if not args.ab else list(args.ab)
    labels = ["A", "B"] if args.ab else ["A"]
    majors = set()
    for lab, d in zip(labels, dirs):
        d = os.path.abspath(d)
        if not os.path.isdir(os.path.join(d, "shim")):
            die("%s has no shim/ directory: build the shim first (cmake --build %s --target vgpucudart vgpucuda)"
                % (d, d))
        m = shim_major(d)
        if m is None:
            die("%s/shim has no libcudart.so.N: the runtime shim needs the CUDA toolkit headers at build time" % d)
        majors.add(m)
        builds[lab] = {"dir": d, "cuda_major": m}
    if len(majors) != 1:
        die("the two builds were made for different CUDA majors (%s); A/B needs one program" % sorted(majors))
    nvcc, rel = find_nvcc(majors.pop(), args.nvcc)
    if not nvcc:
        return None, builds, None, None
    return True, builds, nvcc, rel


def count_all(binary, names, engines, builds, args):
    """Counting runs for every (build, workload, engine), in parallel (they are deterministic)."""
    jobs = []
    for lab, b in builds.items():
        for w in names:
            for e in engines:
                jobs.append((lab, w, e, b["dir"]))
    out = {}
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(counting_run, binary, w, e, d, args): (lab, w, e) for lab, w, e, d in jobs}
        for fut in concurrent.futures.as_completed(futs):
            res, why = fut.result()
            if res is None:
                errors.append(why)
            else:
                out[futs[fut]] = res
    return out, errors


def timing_flow(binary, names, engines, builds, counts, args):
    results = []
    failures = []
    labs = list(builds)
    for w in names:
        for e in engines:
            base = {lab: counts[(lab, w, e)] for lab in labs}
            runs = {lab: [] for lab in labs}
            n = args.runs + args.warmup
            for i in range(n):
                for lab in labs:  # A B A B ...
                    env = clean_env(builds[lab]["dir"], args, e, False)
                    res = run_process(binary, w, args.size, env, args.timeout)
                    ok, why = checked(res, "%s/%s[%s]" % (w, e, lab))
                    if not ok:
                        failures.append(why)
                        continue
                    b = parse_bench_line(res["stdout"])
                    if i >= args.warmup:
                        res.update(b)
                        runs[lab].append(res)
            for lab in labs:
                c = base[lab]
                rec = {"workload": w, "engine": e, "group": WORKLOADS[w][0], "build": lab,
                       "counters": c["totals"], "kernels": c["kernels"],
                       "engine_log": c["engine_log"], "fell_back_to_ptx": c["fell_back_to_ptx"],
                       "units": c["units"], "unit": c["unit"],
                       "runs": [{k: r[k] for k in ("wall", "cpu", "rss_kb", "device_cpu_s", "device_wall_s")}
                                for r in runs[lab]],
                       "stats": summarize_runs(runs[lab], c["totals"].get("instructions", 0), c["units"])}
                for r in rec["kernels"].values():
                    r.pop("_opcodes", None)
                results.append(rec)
            if not args.quiet:
                print("  done %-14s %-4s" % (w, e), file=sys.stderr)
    return results, failures


def ab_compare(results):
    by = {}
    for r in results:
        by.setdefault((r["workload"], r["engine"]), {})[r["build"]] = r
    out = []
    for (w, e), d in by.items():
        if "A" not in d or "B" not in d:
            continue
        a, b = d["A"], d["B"]
        diff = {k: (a["counters"].get(k), b["counters"].get(k))
                for k in sorted(set(a["counters"]) | set(b["counters"]))
                if a["counters"].get(k) != b["counters"].get(k)}
        key = "winst_per_cpu_s" if "winst_per_cpu_s" in a["stats"] else "units_per_s"
        if key not in a["stats"] or key not in b["stats"] or not a["stats"][key] or not b["stats"][key]:
            key = "wall_s"
        sa, sb = a["stats"][key], b["stats"][key]
        ratio = sb["median"] / sa["median"] if sa["median"] else float("nan")
        if key == "wall_s":
            ratio = 1.0 / ratio if ratio else float("nan")  # report speed-up either way
        overlap = not (sa["min"] > sb["max"] or sb["min"] > sa["max"])
        if overlap:
            verdict = "noise"
        else:
            verdict = "faster" if ratio > 1 else "slower"
        out.append({"workload": w, "engine": e, "metric": key, "a_median": sa["median"], "b_median": sb["median"],
                    "speedup_b_over_a": ratio, "delta_pct": (ratio - 1.0) * 100.0, "verdict": verdict,
                    "ranges_overlap": overlap, "counters_identical": not diff, "counter_diffs": diff})
    return out


# --------------------------------------------------------------------------- output


def fmt_rate(st, scale=1e6):
    return "%8.2f" % (st["median"] / scale) if st else "       -"


def print_table(results, ab, out=sys.stdout):
    rows = {}
    for r in results:
        rows.setdefault((r["workload"], r["engine"]), {})[r["build"]] = r
    print("", file=out)
    hdr = "%-14s %-4s %12s %9s %7s %9s %8s %8s  %s" % (
        "workload", "eng", "warp-instr", "Minst/s", "spread", "wall s", "cpu s", "RSS MB", "note")
    print(hdr, file=out)
    print("-" * len(hdr), file=out)
    for (w, e), d in rows.items():
        for lab in d:
            r = d[lab]
            st = r["stats"]
            if "winst_per_cpu_s" in st:
                rate = fmt_rate(st["winst_per_cpu_s"])
                spr = "%6.1f%%" % st["winst_per_cpu_s"]["spread_pct"]
            elif "units_per_s" in st:
                rate = "       -"
                spr = "%6.1f%%" % st["units_per_s"]["spread_pct"]
            else:
                rate, spr = "       -", "     -"
            note = []
            if r["fell_back_to_ptx"]:
                note.append("SASS FELL BACK TO PTX")
            if len(d) > 1:
                note.append("build " + lab)
            if "units_per_s" in st:
                note.append("%.1fk %s/s" % (st["units_per_s"]["median"] / 1e3, r["unit"]))
            print("%-14s %-4s %12d %9s %7s %9.3f %8.3f %8.1f  %s" % (
                w, e, r["counters"].get("instructions", 0), rate, spr,
                st["wall_s"]["median"], st["cpu_s"]["median"], st["rss_mb"]["median"],
                ", ".join(note)), file=out)
    if ab:
        print("\nA/B (B relative to A; ratio of medians of %s; 'noise' = the min..max ranges overlap)"
              % "Minst/s, units/s or 1/wall", file=out)
        h2 = "%-14s %-4s %-16s %10s %10s %8s  %-7s %s" % ("workload", "eng", "metric", "A", "B", "B/A", "verdict",
                                                          "counters")
        print(h2, file=out)
        print("-" * len(h2), file=out)
        for x in ab:
            sc = 1e6 if x["metric"] == "winst_per_cpu_s" else (1e3 if x["metric"] == "units_per_s" else 1.0)
            print("%-14s %-4s %-16s %10.3f %10.3f %7.3fx  %-7s %s" % (
                x["workload"], x["engine"], x["metric"], x["a_median"] / sc, x["b_median"] / sc,
                x["speedup_b_over_a"], x["verdict"],
                "identical" if x["counters_identical"] else "DIFFER: " + ", ".join(
                    "%s %s->%s" % (k, v[0], v[1]) for k, v in list(x["counter_diffs"].items())[:4])), file=out)


def markdown_table(results, ab):
    import io
    buf = io.StringIO()
    print("| workload | engine | warp-instr | Minst/s (median) | spread | wall s | RSS MB |", file=buf)
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |", file=buf)
    for r in results:
        if r["build"] != "A" and not ab:
            continue
        st = r["stats"]
        if "winst_per_cpu_s" in st:
            rate, spr = "%.2f" % (st["winst_per_cpu_s"]["median"] / 1e6), "%.1f%%" % st["winst_per_cpu_s"]["spread_pct"]
        elif "units_per_s" in st:
            rate, spr = "%.1f k %s/s" % (st["units_per_s"]["median"] / 1e3, r["unit"]), "%.1f%%" % st["units_per_s"]["spread_pct"]
        else:
            rate, spr = "-", "-"
        print("| %s | %s%s | %d | %s | %s | %.3f | %.1f |" % (
            r["workload"], r["engine"], (" (%s)" % r["build"]) if ab else "", r["counters"].get("instructions", 0),
            rate, spr, st["wall_s"]["median"], st["rss_mb"]["median"]), file=buf)
    return buf.getvalue()


# --------------------------------------------------------------------------- check-counters


def expected_path(rel):
    return os.path.join(EXPECTED_DIR, "cuda-%s.json" % rel[0])


def compact_counters(counts, names, engines):
    out = {}
    for w in names:
        out[w] = {}
        for e in engines:
            c = counts[("A", w, e)]
            out[w][e] = {
                "totals": c["totals"],
                "kernels": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                            for k, v in sorted(c["kernels"].items())},
                "engine": "ptx" if e == "ptx" else ("sass" if not c["fell_back_to_ptx"] else "sass+ptx-fallback"),
            }
    return out


def write_expected(path, rel, args, names, engines, counts):
    doc = {
        "note": "Exact instruction counters of tools/bench/kernels/bench.cu workloads (size 'quick', "
                "VGPU_THREADS=1), written by `tools/bench/bench.sh --update-expected`. Counts, never times. "
                "They depend on the compiler that built the program, hence one file per CUDA release.",
        "toolkit": rel[0], "nvcc_version": rel[1], "gpu_profile": args.gpu, "compiled_arch": NVCC_ARCH,
        "size": "quick", "workloads": compact_counters(counts, names, engines),
    }
    with open(path, "w") as f:
        f.write("{\n")
        keys = [k for k in doc if k != "workloads"]
        for k in keys:
            f.write("  %s: %s,\n" % (json.dumps(k), json.dumps(doc[k])))
        f.write('  "workloads": {\n')
        wl = doc["workloads"]
        for wi, (w, d) in enumerate(wl.items()):
            f.write("    %s: {\n" % json.dumps(w))
            for ei, (e, c) in enumerate(d.items()):
                f.write("      %s: {\n" % json.dumps(e))
                f.write('        "engine": %s,\n' % json.dumps(c["engine"]))
                f.write('        "totals": %s,\n' % json.dumps(c["totals"], sort_keys=True))
                f.write('        "kernels": {\n')
                ks = list(c["kernels"].items())
                for ki, (k, v) in enumerate(ks):
                    f.write("          %s: %s%s\n" % (json.dumps(k), json.dumps(v, sort_keys=True),
                                                       "," if ki + 1 < len(ks) else ""))
                f.write("        }\n")
                f.write("      }%s\n" % ("," if ei + 1 < len(d) else ""))
            f.write("    }%s\n" % ("," if wi + 1 < len(wl) else ""))
        f.write("  }\n}\n")


def check_counters(path, rel, names, engines, counts, args):
    with open(path) as f:
        exp = json.load(f)
    cur = compact_counters(counts, names, engines)
    bad = []
    for w in names:
        for e in engines:
            want = exp.get("workloads", {}).get(w, {}).get(e)
            got = cur[w][e]
            if want is None:
                bad.append("%s/%s: no expected entry (run --update-expected)" % (w, e))
                continue
            if want["engine"] != got["engine"]:
                bad.append("%s/%s: ran on %s, expected %s" % (w, e, got["engine"], want["engine"]))
            wk, gk = want["kernels"], got["kernels"]
            for k in sorted(set(wk) | set(gk)):
                if k not in gk:
                    bad.append("%s/%s: kernel %s no longer launched" % (w, e, k))
                    continue
                if k not in wk:
                    bad.append("%s/%s: new kernel %s" % (w, e, k))
                    continue
                for field in sorted(set(wk[k]) | set(gk[k])):
                    a, b = wk[k].get(field), gk[k].get(field)
                    if a != b:
                        bad.append("%s/%s: %s: %s expected %s, got %s" % (w, e, k, field, a, b))
    extra = [w for w in exp.get("workloads", {}) if w not in names]
    return bad, extra, exp


# --------------------------------------------------------------------------- mix


def mix_table(counts_or_kernels, label_fn, out=sys.stdout):
    """Class mix as a share of thread-instructions, per row."""
    hdr = "%-26s %12s %7s " % ("", "warp-instr", "lanes") + " ".join("%6s" % c for c in CLASS_ORDER)
    print(hdr, file=out)
    print("-" * len(hdr), file=out)
    for label, tot in counts_or_kernels:
        ti = tot.get("thread_instructions", 0)
        wi = tot.get("instructions", 0)
        shares = []
        for c in CLASS_ORDER:
            v = tot.get("mix_" + c, 0)
            shares.append("%5.1f%%" % (100.0 * v / ti) if ti else "     -")
        classes = sum(tot.get("mix_" + c, 0) for c in CLASS_ORDER)
        flag = "" if classes == ti else "  !! classes sum to %d, thread-instructions %d" % (classes, ti)
        print("%-26s %12d %7.2f " % (label, wi, (ti / wi) if wi else 0.0) + " ".join(shares) + flag, file=out)
    print("lanes = thread-instructions / warp-instructions (average active lanes of 32).", file=out)


# --------------------------------------------------------------------------- command line


def build_parser():
    p = argparse.ArgumentParser(prog="bench.sh", description=__doc__.split("\n\n")[0])
    p.add_argument("mode", nargs="?", choices=["run", "mix"], default="run",
                   help="run (default): time the workloads; mix: print instruction-class mix")
    p.add_argument("--build", default=os.path.join(ROOT, "build"), help="build directory (has shim/)")
    p.add_argument("--ab", nargs=2, metavar=("DIR_A", "DIR_B"), help="alternate A B A B ... between two build dirs")
    p.add_argument("--workloads", help="comma list of workloads or groups (alu tensor memory control api)")
    p.add_argument("--engines", default="sass,ptx", help="sass (default engine), ptx (VGPU_SASS=0), or both")
    p.add_argument("--size", choices=["quick", "full"], default=None,
                   help="workload size: full for timing (default), quick for mix and the counter check")
    p.add_argument("--runs", type=int, default=5, help="timed runs per workload/engine/build")
    p.add_argument("--warmup", type=int, default=1, help="untimed runs before the timed ones")
    p.add_argument("--threads", type=int, default=1, help="VGPU_THREADS (1 = serial blocks, the default)")
    p.add_argument("--gpu", default=GPU_DEFAULT, help="VGPU_GPU profile (the program is compiled for " + NVCC_ARCH + ")")
    p.add_argument("--env", action="append", default=[], metavar="K=V", help="extra environment for every run")
    p.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                   help="parallel counting runs (counters are deterministic; timing runs are always serial)")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds before one run is killed")
    p.add_argument("--json", metavar="FILE", help="write the full results as JSON")
    p.add_argument("--markdown", metavar="FILE", help="write the table as Markdown (for $GITHUB_STEP_SUMMARY)")
    p.add_argument("--check-counters", action="store_true",
                   help="quick size, counters only: fail if they differ from tools/bench/expected")
    p.add_argument("--update-expected", action="store_true", help="rewrite tools/bench/expected/cuda-<release>.json")
    p.add_argument("--require-expected", action="store_true", help="with --check-counters: no file for this toolkit is an error")
    p.add_argument("--nvcc", help="nvcc to compile the program with (default: one matching the shim's CUDA major)")
    p.add_argument("--cache", help="directory for the compiled program (default ~/.cache/vgpu-bench)")
    p.add_argument("--record-hostname", action="store_true", help="put the host name in the JSON meta")
    p.add_argument("--build-only", action="store_true", help="compile the program, print its path, and stop")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--cmd", nargs=argparse.REMAINDER, help="mix only: run this command instead of the workloads")
    return p


def main():
    args = build_parser().parse_args()
    if args.mode == "mix" and args.cmd:
        return mix_command(args)
    engines = [e for e in args.engines.split(",") if e]
    for e in engines:
        if e not in ENGINES:
            die("unknown engine '%s' (sass, ptx)" % e)
    if args.size is None:
        args.size = "quick" if args.mode == "mix" else "full"
    names = selected_workloads(args)
    ok, builds, nvcc, rel = setup(args)
    if not ok:
        msg = "no nvcc for the shim's CUDA major (%s); the benchmark programs are compiled from source" % \
              builds["A"]["cuda_major"]
        if args.check_counters:
            print("SKIP: " + msg)
            return 0
        die(msg, 1)
    # The counting runs use the quick size under --check-counters and --update-expected, else the timed size.
    args.size_for_counters = "quick" if (args.check_counters or args.update_expected) else args.size
    if args.check_counters or args.update_expected:
        args.size = "quick"
        args.threads = 1  # atomics' retry loops (CAS) take schedule-dependent paths on more than one host thread
    for lab, b in builds.items():
        sanitized = shim_sanitizer_flags(b["dir"])
        if sanitized and args.check_counters:
            print("SKIP: %s is built with a sanitizer; the counter check runs on the Release job" % b["dir"])
            return 0
    binary = ensure_binary(builds["A"]["dir"], args, (nvcc, rel))
    if args.build_only:
        print(binary)
        return 0
    t_start = time.monotonic()
    counts, errors = count_all(binary, names, engines, builds, args)
    if errors:
        print("FAIL: a workload did not pass its own check:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1

    if args.update_expected or args.check_counters:
        path = expected_path(rel)
        if args.update_expected:
            if not args.workloads and len(builds) == 1:
                os.makedirs(EXPECTED_DIR, exist_ok=True)
                write_expected(path, rel, args, names, engines, counts)
                print("wrote %s (%d workloads x %s)" % (os.path.relpath(path, ROOT), len(names), "+".join(engines)))
                return 0
            die("--update-expected rewrites the whole file: run it with no --workloads/--ab")
        if not os.path.exists(path):
            have = sorted(os.listdir(EXPECTED_DIR)) if os.path.isdir(EXPECTED_DIR) else []
            msg = "no expected counters for CUDA %s (nvcc %s) in tools/bench/expected (have: %s)" % (
                rel[0], rel[1], ", ".join(have) or "none")
            if args.require_expected:
                print("FAIL: " + msg)
                return 1
            print("SKIP: " + msg)
            return 0
        bad, extra, exp = check_counters(path, rel, names, engines, counts, args)
        dt = time.monotonic() - t_start
        if exp.get("nvcc_version") != rel[1]:
            print("note: expected counters were written with nvcc %s, this is nvcc %s (same release; "
                  "a different patch level can change the generated code)" % (exp.get("nvcc_version"), rel[1]))
        if bad:
            print("FAIL: instruction counters drifted from %s (%d differences):" % (os.path.relpath(path, ROOT), len(bad)))
            for line in bad[:60]:
                print("  " + line)
            if len(bad) > 60:
                print("  ... and %d more" % (len(bad) - 60))
            print("If the change is intended (a simulator change must not alter a counter: ARCHITECTURE.md), "
                  "regenerate with tools/bench/bench.sh --update-expected and explain it in the commit.")
            return 1
        print("PASS: counters of %d workloads x %s match %s (%.1f s)" % (
            len(names), "+".join(engines), os.path.relpath(path, ROOT), dt))
        return 0

    if args.mode == "mix":
        rows = []
        for w in names:
            for e in engines:
                rows.append(("%s/%s" % (w, e), counts[("A", w, e)]["totals"]))
        mix_table(rows, None)
        if "ptx" in engines:
            print("\nPTX opcodes (the ten most used per launch, summed over launches; the SASS engine keeps no "
                  "opcode histogram):")
            for w in names:
                ops = {}
                for k in counts[("A", w, "ptx")]["kernels"].values():
                    for n, v in k.get("_opcodes", {}).items():
                        ops[n] = ops.get(n, 0) + v
                top = sorted(ops.items(), key=lambda x: -x[1])[:8]
                print("  %-14s %s" % (w, "  ".join("%s=%d" % t for t in top)))
        return 0

    # timing
    meta = meta_block(args, builds, nvcc, rel)
    results, failures = timing_flow(binary, names, engines, builds, counts, args)
    meta["loadavg_end"] = os.getloadavg()
    meta["elapsed_s"] = time.monotonic() - t_start
    ab = ab_compare(results) if args.ab else []
    print_table(results, ab)
    print("\nhost: %s, %s logical CPUs, load %.2f %.2f %.2f -> %.2f %.2f %.2f; nvcc %s; VGPU_THREADS=%d; GPU profile %s; "
          "%d timed runs after %d warmup" % (
              meta["cpu"], meta["logical_cpus"], *meta["loadavg_start"], *meta["loadavg_end"],
              meta["nvcc_version"], args.threads, args.gpu, args.runs, args.warmup))
    if any(r["fell_back_to_ptx"] for r in results):
        print("WARNING: some SASS rows fell back to PTX for a kernel the SASS engine could not run; "
              "those numbers are PTX numbers.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"schema": 1, "meta": meta, "results": results, "ab": ab}, f, indent=1, sort_keys=True)
            f.write("\n")
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(markdown_table(results, bool(args.ab)))
    if failures:
        print("FAIL: runs that did not pass:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    if args.ab and any(not x["counters_identical"] for x in ab):
        print("NOTE: counters differ between A and B for at least one workload (see the table): the two "
              "builds did not execute the same instruction stream.")
    return 0


def mix_command(args):
    """bench.sh mix --cmd prog args...: class mix of any program's launches under the shim."""
    build = os.path.abspath(args.build)
    if not os.path.isdir(os.path.join(build, "shim")):
        die("%s has no shim/ directory" % build)
    rows = []
    for e in [x for x in args.engines.split(",") if x]:
        env = clean_env(build, args, e, True)
        r = subprocess.run(args.cmd, env=env, capture_output=True, text=True)
        kernels, log = parse_counters(r.stderr)
        for k, v in sorted(kernels.items()):
            rows.append(("%s [%s]" % (k[:20], e), v))
        rows.append(("TOTAL [%s]" % e, totals(kernels)))
    mix_table(rows, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
