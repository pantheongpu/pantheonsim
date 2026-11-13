#!/usr/bin/env python3
"""timeline.py --build B --work W --progs corpus/name,... [--threads 1|default] [--cpus 12-15] [--env K=V,...] [--warm]
One cold VGPU_JIT=1 run per program under strace (seccomp filter: execve/exit_group/wait4 only), giving when the compiler's
processes start and end against the process's start and end.  The generated sources are kept (VGPU_JIT_KEEP)."""
import argparse, fcntl, json, os, re, shutil, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import real
LOCK = "/tmp/pantheonsim-builder.lock"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True); ap.add_argument("--work", required=True); ap.add_argument("--progs", required=True)
    ap.add_argument("--threads", default="1"); ap.add_argument("--cpus", default=""); ap.add_argument("--env", default="")
    ap.add_argument("--corp", default=os.path.join(HERE, "../corp")); ap.add_argument("--out", default="")
    ap.add_argument("--keep-cache", action="store_true"); ap.add_argument("--nostrace", action="store_true")
    a = ap.parse_args()
    envd = dict(x.split("=", 1) for x in a.env.split(",") if x)
    want = a.progs.split(",")
    progs = {p[0] + "/" + p[1]: p for p in real.manifest(os.path.abspath(a.corp), ["polybench", "rodinia", "hecbench"])}
    os.makedirs(a.work, exist_ok=True)
    res = {}
    lf = open(LOCK, "a"); fcntl.flock(lf, fcntl.LOCK_EX)
    for pid in want:
        corpus, name, cwd, argv, cp = progs[pid]
        cache = os.path.join(a.work, "tl-cache")
        if not a.keep_cache:
            shutil.rmtree(cache, ignore_errors=True)
        os.makedirs(cache, mode=0o700, exist_ok=True)
        sf = os.path.join(a.work, "tl-stats"); open(sf, "w").close()
        env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
        env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", LD_LIBRARY_PATH=a.build + "/shim", VGPU_JIT="1", VGPU_JIT_CACHE=cache,
                   VGPU_JIT_STATS="1", VGPU_JIT_STATS_FILE=sf, VGPU_JIT_KEEP="1")
        if a.threads != "default": env["VGPU_THREADS"] = a.threads
        env.update(envd)
        if cp:
            scratch = os.path.join(a.work, "tl-run"); shutil.rmtree(scratch, ignore_errors=True); os.makedirs(scratch)
            dd = os.path.join(os.path.dirname(argv[0]), "data")
            open(os.path.join(scratch, "filelist"), "w").write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
            cwd = scratch
        st = os.path.join(a.work, "tl-strace")
        pre = [] if a.nostrace else ["strace", "-f", "--seccomp-bpf", "-qq", "-ttt", "-e", "trace=execve,exit_group", "-o", st]
        cmd = (["taskset", "-c", a.cpus] if a.cpus else []) + ["nice", "-n", "19"] + pre + argv
        t0 = time.time()
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t1 = time.time()
        ev = []
        if not a.nostrace:
            first = None
            for line in open(st):
                m = re.match(r"(\d+)\s+([\d.]+)\s+(.*)", line)
                if not m: continue
                pid_, ts, rest = int(m[1]), float(m[2]), m[3]
                if first is None: first = ts
                if rest.startswith("execve("):
                    ex = re.match(r'execve\("([^"]+)"', rest)
                    ev.append((round(ts - first, 3), pid_, "exec", os.path.basename(ex[1]) if ex else rest[:40]))
                elif rest.startswith("exit_group(") or rest.startswith("+++"):
                    ev.append((round(ts - first, 3), pid_, "exit", rest[:30]))
        stats = open(sf).read()
        srcs = [(f, os.path.getsize(os.path.join(cache, f))) for f in os.listdir(cache) if f.endswith(".cpp")]
        objs = [f for f in os.listdir(cache) if f.endswith(".so")]
        r = dict(wall=round(t1 - t0, 3), rc=p.returncode, events=ev, srcs=srcs, objs=len(objs),
                 summary=[l for l in stats.splitlines() if l.startswith("jit: ") and "units" in l or l.startswith("jit-")])
        res[pid] = r
        print("== %s wall=%.3f rc=%d objs=%d srcs=%s" % (pid, r["wall"], p.returncode, len(objs), srcs))
        for e in ev:
            if e[3] in ("cc1plus", "as", "collect2", "ld", "c++", "g++") or e[2] == "exit": print("   ", e)
        for l in r["summary"]: print("   ", l[:300])
        if not a.keep_cache:
            keep = os.path.join(a.work, "src-" + pid.replace("/", "_")); shutil.rmtree(keep, ignore_errors=True); os.makedirs(keep)
            for f, _ in srcs: shutil.copy(os.path.join(cache, f), keep)
            rt = [d for d in os.listdir(cache) if d.startswith("rt-")]
            for d in rt: shutil.copytree(os.path.join(cache, d), os.path.join(keep, d), dirs_exist_ok=True)
    if a.out: json.dump(res, open(a.out, "w"), separators=(",", ":"))

if __name__ == "__main__": main()
