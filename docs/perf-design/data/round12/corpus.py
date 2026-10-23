#!/usr/bin/env python3
"""Run every program of the four prebuilt corpora (Rodinia, PolyBench/GPU, HeCBench reduced list, CUDA Samples 13.0) in the modes
of E2/E3 and store stdout+stderr, rc, wall and new/changed files' sha256 per (case, mode).

usage: corpus.py <outdir> <mode> [--profile P] [--threads N] [--jobs J] [--timeout S] [--tag TAG] [--only a,b] [--corpus c1,c2]
  mode: native | ptxjit | sim
  results: <outdir>/<tag>/<case>.json   (tag defaults to the mode, or sim-<profile basename>)
Cases never run concurrently on the GPU unless --jobs > 1 (use --jobs 1 for native runs you want timed).
"""
import sys, os, json, re, hashlib, argparse, shutil, tempfile, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

C = HOME + "/.cache"
ROD = C + "/gpu-rodinia/bin"
D = ROD + "/data"
rodinia = {
    "backprop": ["backprop", "16384"], "gaussian": ["gaussian", "-s", "64"], "lud": ["lud", "-s", "256", "-v"],
    "needle": ["needle", "256", "10"], "pathfinder": ["pathfinder", "1000", "50", "20"],
    "srad_v2": ["srad_v2", "64", "64", "0", "31", "0", "31", "0.5", "2"],
    "particlefilter_float": ["particlefilter_float", "-x", "64", "-y", "64", "-z", "5", "-np", "1000"],
    "sc_gpu": ["sc_gpu", "10", "20", "16", "1024", "1024", "1000", "none", "out.txt", "1"],
    "hotspot": ["hotspot", "64", "2", "20", D + "/temp_64", D + "/power_64", "output.out"],
    "bfs": ["bfs", D + "/graph4096.txt"], "nn": ["nn", D + "/filelist4", "-r", "10", "-lat", "30", "-lng", "90"],
    "lavaMD": ["lavaMD", "-boxes1d", "4"],
}
POLY = "2DCONV 2MM 3DCONV 3MM ADI ATAX BICG CORR COVAR FDTD-2D GEMM GEMVER GESUMMV GRAMSCHM JACOBI1D JACOBI2D LU MVT SYR2K SYRK".split()


def cases(corpora):
    out = []  # (corpus, name, cwd or None for fresh tmp dir, argv)
    if "rodinia" in corpora:
        for n, a in rodinia.items():
            out.append(("rodinia", n, None, [ROD + "/" + a[0]] + a[1:]))
    if "polybench" in corpora:
        for n in POLY:
            out.append(("polybench", n, None, [C + "/polybenchGpu/bin/" + n]))
    if "hecbench" in corpora:
        src = C + "/HeCBench/src"
        for line in open(C + "/HeCBench/reduced.txt"):
            d, _, args = line.rstrip("\n").partition("|")
            mk = open(src + "/" + d + "/Makefile").read()
            m = re.search(r"^program\s*=\s*(\S+)", mk, re.M)
            names = os.listdir(src + "/" + d)
            exe = "main" if "main" in names else (m[1] if m and m[1] in names else None)
            if exe:
                out.append(("hecbench", d, src + "/" + d, ["./" + exe] + args.split()))
    if "samples" in corpora:
        base = C + "/cuda-samples/build"
        for line in open(base + "/exes.txt"):
            p = line.strip()
            if p:
                out.append(("samples", os.path.basename(p), base + "/" + os.path.dirname(p), ["./" + os.path.basename(p)]))
    return out


CENSUS = "cuda*+__cuda*+cu*+nvml*+cublas*+cufft*+curand*+cusparse*+cusolver*+cudnn*+nccl*+nvrtc*+nvjpeg*+npp*+thrust*"


def snapshot(d):
    s = {}
    for n in os.listdir(d):
        p = os.path.join(d, n)
        try:
            st = os.lstat(p)
            if os.path.isfile(p):
                s[n] = (st.st_size, st.st_mtime_ns)
        except OSError:
            pass
    return s


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def do_case(cs, mode, profile, threads, timeout, outdir, extra):
    corpus, name, cwd, argv = cs
    tmp = None
    if cwd is None:
        tmp = tempfile.mkdtemp(prefix="c-", dir="/dev/shm")
        cwd = tmp
    before = snapshot(cwd)
    lt = None
    if mode == "census":  # native run under ltrace -c: how often the program calls each CUDA / library entry point
        lt = tempfile.mktemp(prefix="lt-", dir="/dev/shm")
        argv = ["ltrace", "-f", "-c", "-o", lt, "-e", CENSUS] + argv
    env = env_for("native" if mode == "census" else mode, profile=profile, threads=threads, extra=extra)
    r = run(argv, env, cwd=cwd, timeout=timeout)
    after = snapshot(cwd)
    files = {n: [after[n][0], sha(os.path.join(cwd, n))] for n in after if before.get(n) != after[n] and after[n][0] < 64 << 20}
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    rec = dict(corpus=corpus, case=name, mode=mode, profile=profile if mode == "sim" else None, rc=r["rc"], timeout=r["timeout"],
               wall=round(r["wall"], 3), stdout=r["stdout"][-200000:], stderr=r["stderr"][-20000:], files=files, extra=extra)
    if lt:
        try:
            rec["ltrace"] = open(lt).read()
            os.unlink(lt)
        except OSError:
            rec["ltrace"] = ""
    with open(os.path.join(outdir, corpus + "__" + name + ".json"), "w") as f:
        json.dump(rec, f)
    return name, r["rc"], r["timeout"], r["wall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("mode", choices=["native", "ptxjit", "sim", "census"])
    ap.add_argument("--profile", default="nvidia/a10g"); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--tag"); ap.add_argument("--only"); ap.add_argument("--corpus", default="rodinia,polybench,hecbench,samples")
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    tag = a.tag or (a.mode if a.mode != "sim" else "sim-" + a.profile.split("/")[-1])
    od = os.path.join(a.outdir, tag)
    os.makedirs(od, exist_ok=True)
    extra = dict(e.split("=", 1) for e in a.env)
    cs = cases(a.corpus.split(","))
    if a.only:
        want = set(a.only.split(","))
        cs = [c for c in cs if c[1] in want]
    print("%d cases, mode %s tag %s jobs %d" % (len(cs), a.mode, tag, a.jobs), flush=True)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(do_case, c, a.mode, a.profile, a.threads if a.mode == "sim" else None, a.timeout, od, extra) for c in cs]
        for f in cf.as_completed(futs):
            n, rc, to, w = f.result()
            print("%-34s rc=%s%s %.1fs" % (n, rc, " TIMEOUT" if to else "", w), flush=True)


main()
