#!/usr/bin/env python3
"""Real-program A/B: run prebuilt CUDA programs (PolyBench/GPU, Rodinia,
HeCBench, CUDA Samples) on the simulator with two builds, alternating, pinned
to one CPU, and print median wall seconds, user CPU seconds and the check
results per program.  Simulator side only: no GPU card is ever touched.

  real.py --before /tmp/base0/build --after /tmp/sp5-fin3/build \
          --corp /tmp/sp8-realwork/corp --cpu 0 --runs 5 --threads 1

See README.md for the corpus layout (pack_corpora.sh builds it)."""
import argparse, hashlib, json, math, os, re, shutil, signal, statistics
import subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))

# lines that carry timings or host identity: dropped before comparing outputs
NOISE = re.compile(r"(?i)(\btime\b|\bsec(onds?)?\b|\bms\b|elapsed|throughput|"
                   r"gflop|gb/s|mb/s|bandwidth|latency|\[vgpu\]|speedup|"
                   r"performance|fom|rss|clock|cycles|\bMFlops\b|\busec\b|"
                   r"\bmsec\b|runtime|kernel execution|\bnow\b)")
CHECK = re.compile(r"(?i)(non-matching[^\n]*|\bPASS(ED)?\b|\bFAIL(ED)?\b|"
                   r"result[s]?\s*=?\s*(ok|correct)|test passed|rmse[^\n]*|"
                   r"error[: =]+[0-9.e+-]+)")


def manifest(corp, which):
    """-> list of (corpus, name, cwd, argv, copy_cwd)"""
    out = []
    if "polybench" in which:
        b = os.path.join(corp, "polybench/bin")
        for n in sorted(os.listdir(b)):
            if os.path.isfile(os.path.join(b, n)):
                out.append(("polybench", n, b, [os.path.join(b, n)], False))
    if "rodinia" in which:
        b = os.path.join(corp, "rodinia/bin")
        D = os.path.join(b, "data")
        # sizes scaled up from the card/sim runners (run.sh): those finish in
        # tens of milliseconds, which only times process start-up
        R = [("backprop", ["262144"]), ("gaussian", ["-s", "512"]),
             ("lud", ["-s", "1024", "-v"]), ("needle", ["2048", "10"]),
             ("pathfinder", ["100000", "100", "20"]),
             ("srad_v2", "512 512 0 127 0 127 0.5 2".split()),
             ("particlefilter_float", "-x 128 -y 128 -z 10 -np 10000".split()),
             ("sc_gpu", "10 20 16 4096 4096 1000 none out.txt 1".split()),
             ("hotspot", ["64", "2", "2000", D + "/temp_64", D + "/power_64", "output.out"]),
             ("bfs", [D + "/graph4096.txt"]),
             ("nn", ["filelist", "-r", "10", "-lat", "30", "-lng", "90"]),
             ("lavaMD", ["-boxes1d", "6"])]
        for n, a in R:
            if os.path.exists(os.path.join(b, n)):
                out.append(("rodinia", n, b, [os.path.join(b, n)] + a, True))
    if "hecbench" in which:
        s = os.path.join(corp, "hec/src")
        for line in open(os.path.join(corp, "hec/reduced.txt")):
            line = line.rstrip("\n")
            if not line:
                continue
            d, args = line.split("|", 1)
            exe = "main"
            if not os.path.exists(os.path.join(s, d, exe)):
                m = re.search(r"^program\s*=\s*(\S+)", open(os.path.join(s, d, "Makefile")).read(), re.M)
                exe = m.group(1) if m else exe
            out.append(("hecbench", d, os.path.join(s, d), [os.path.join(s, d, exe)] + args.split(), False))
    if "samples" in which:
        base = os.path.join(corp, "samples")
        for line in open(os.path.join(base, "sim_main.txt")):
            rc, exe = line.split()
            if rc != "0":
                continue
            out.append(("samples", os.path.basename(exe), os.path.join(base, os.path.dirname(exe)),
                        [os.path.join(base, exe)], False))
    return out


def run_one(build, cwd, argv, copy_cwd, cpu, threads, timeout):
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("VGPU_"):
            del env[k]
    env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060",
               LD_LIBRARY_PATH=os.path.join(build, "shim"))
    if threads:
        env["VGPU_THREADS"] = str(threads)
    scratch = None
    if copy_cwd:
        scratch = tempfile.mkdtemp(prefix="real-")
        # nn reads a list of absolute db paths
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        with open(os.path.join(scratch, "filelist"), "w") as f:
            f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    t0 = time.monotonic()
    p = subprocess.Popen(["taskset", "-c", str(cpu)] + argv, cwd=cwd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         start_new_session=True)
    to = False
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        to = True
    wall = time.monotonic() - t0
    text = out.decode("utf-8", "replace")
    extra = b""
    if scratch:
        for f in sorted(x for x in os.listdir(scratch) if x != "filelist"):
            fp = os.path.join(scratch, f)
            if os.path.isfile(fp):
                extra += f.encode() + open(fp, "rb").read()
        shutil.rmtree(scratch, ignore_errors=True)
    return wall, p.returncode, to, text, extra


def normalise(text, extra):
    keep = [l for l in text.splitlines() if not NOISE.search(l)]
    h = hashlib.sha256(("\n".join(keep)).encode() + extra).hexdigest()[:12]
    checks = sorted(set(m.group(0).strip() for m in CHECK.finditer(text)))
    return h, checks


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--corp", required=True)
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", default="1", help="VGPU_THREADS value, or 'default' to leave it unset")
    ap.add_argument("--corpora", default="polybench,rodinia,hecbench,samples")
    ap.add_argument("--programs", default="", help="comma list of corpus/name; default all")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--max-load", type=float, default=8.0)
    ap.add_argument("--only", choices=["before", "after"], help="calibration: run one side only")
    ap.add_argument("--skip", default="", help="comma list of corpus/name to leave out (known timeouts)")
    ap.add_argument("--long-secs", type=float, default=30, help="programs slower than this get --long-runs runs")
    ap.add_argument("--long-runs", type=int, default=3)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    threads = None if a.threads == "default" else a.threads
    a.corp = os.path.abspath(a.corp)
    progs = manifest(a.corp, a.corpora.split(","))
    if a.programs:
        want = set(a.programs.split(","))
        progs = [p for p in progs if p[0] + "/" + p[1] in want or p[1] in want]
    if a.skip:
        sk = set(a.skip.split(","))
        progs = [p for p in progs if p[0] + "/" + p[1] not in sk]
    builds = {"before": a.before, "after": a.after}
    sides = [a.only] if a.only else ["before", "after"]
    res = {}
    for corpus, name, cwd, argv, cp in progs:
        key = corpus + "/" + name
        r = {s: {"wall": [], "user": [], "rc": [], "hash": set(), "checks": set(), "timeout": False} for s in sides}
        nruns = a.runs
        i = -1
        while i + 1 < nruns:
            i += 1
            for s in (sides if i % 2 == 0 else sides[::-1]):
                while os.getloadavg()[0] > a.max_load:
                    time.sleep(15)
                c0 = os.times()
                wall, rc, to, text, extra = run_one(builds[s], cwd, argv, cp, a.cpu, threads, a.timeout)
                c1 = os.times()
                h, ch = normalise(text, extra)
                d = r[s]
                d["wall"].append(wall); d["user"].append(c1.children_user - c0.children_user)
                d["rc"].append(rc); d["hash"].add(h); d["checks"].update(ch); d["timeout"] |= to
                if to:
                    break
            if any(r[s]["timeout"] for s in sides):
                break
            if i == 0 and max(r[s]["wall"][0] for s in sides) > a.long_secs:
                nruns = min(nruns, a.long_runs)
        res[key] = r
        line = key.ljust(40)
        for s in sides:
            d = r[s]
            line += " %s %.2fs" % (s[:1].upper(), statistics.median(d["wall"]))
        print(line, "rc", {s: sorted(set(r[s]["rc"])) for s in sides}, flush=True)
    # table
    ser = {}
    print()
    for corpus in a.corpora.split(","):
        rows = [(k, v) for k, v in res.items() if k.startswith(corpus + "/")]
        if not rows:
            continue
        print("### %s (threads=%s, cpu %d, %d runs, median)\n" % (corpus, a.threads, a.cpu, a.runs))
        if a.only:
            continue
        print("| program | n | BEFORE s | AFTER s | ratio | BEFORE user s | AFTER user s | rc | outputs | checks |")
        print("|---|---:|---:|---:|---:|---:|---:|---|---|---|")
        rs = []
        for k, v in rows:
            b, f = v["before"], v["after"]
            mb, mf = statistics.median(b["wall"]), statistics.median(f["wall"])
            ok = not (b["timeout"] or f["timeout"])
            same = ok and b["hash"] == f["hash"] and len(b["hash"]) == 1
            rc = "ok" if set(b["rc"]) == set(f["rc"]) and len(set(b["rc"])) == 1 else "B%s/A%s" % (sorted(set(b["rc"])), sorted(set(f["rc"])))
            if b["timeout"] or f["timeout"]:
                rc = "TIMEOUT"
            ratio = mb / mf
            if ok and same:
                rs.append(ratio)
            chk = "; ".join(sorted(f["checks"]))[:60]
            print("| %s | %d | %.2f | %.2f | %.2fx | %.2f | %.2f | %s | %s | %s |" % (
                k.split("/", 1)[1], len(b["wall"]), mb, mf, ratio, statistics.median(b["user"]),
                statistics.median(f["user"]), rc, "same" if same else "DIFFER", chk))
        print("\ngeometric mean (programs with identical outputs): %.3fx over %d programs\n" % (geomean(rs), len(rs)))
    if a.json:
        for v in res.values():
            for d in v.values():
                d["hash"] = sorted(d["hash"]); d["checks"] = sorted(d["checks"])
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
