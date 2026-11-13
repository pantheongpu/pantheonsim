#!/usr/bin/env python3
# Round sp68 card A/B: interleaved runs of programs on several sides (build + environment), on the A10G rig. Per run: process wall time, the program's
# "GPU Time" line when it prints one, a hash of stdout without its timing lines, and the host_gpu summary lines (launches on the card, MiB moved, verify).
#   usage: ab.py --out ab.json --runs 3 --prog NAME=DIR:CMD... --side NAME=BUILD[:VAR=V;VAR=V] ...
import argparse, hashlib, json, os, re, subprocess, time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--prog", action="append", required=True)
ap.add_argument("--side", action="append", required=True)
ap.add_argument("--timeout", type=int, default=900)
a = ap.parse_args()
home = os.path.expanduser("~/w")
progs = [(p.split("=", 1)[0], p.split("=", 1)[1].split(":", 1)) for p in a.prog]
sides = []
for s in a.side:
    name, rest = s.split("=", 1)
    build, _, env = rest.partition(":")
    sides.append((name, build, dict(kv.split("=", 1) for kv in env.split(";") if kv)))
timing = re.compile(r"(Time in seconds|time|Time|elapsed|ms\b|seconds)", re.I)
res = json.load(open(a.out)) if os.path.exists(a.out) else []
for r in range(a.runs):
    for pname, (pdir, cmd) in progs:
        for sname, build, env in sides:
            e = dict(os.environ, VGPU_GPU="nvidia/a10g", VGPU_THREADS="4")
            if build != "native":
                e["LD_LIBRARY_PATH"] = f"{home}/{build}/shim"
            e.update(env)
            t0 = time.time()
            try:
                p = subprocess.run(cmd, shell=True, cwd=os.path.join(home, pdir), env=e, capture_output=True, text=True, timeout=a.timeout, errors="replace")
                rc, out, err = p.returncode, p.stdout, p.stderr
            except subprocess.TimeoutExpired as x:
                rc, out, err = 124, (x.stdout or b"").decode(errors="replace") if isinstance(x.stdout, bytes) else (x.stdout or ""), ""
            wall = time.time() - t0
            keep, skip = [], False
            for l in out.splitlines():   # (PolyBench prints the time on the line after "GPU Time in seconds:")
                if skip and re.fullmatch(r"\s*[0-9.eE+-]+\s*", l):
                    skip = False
                    continue
                skip = bool(timing.search(l))
                if not skip and not l.startswith("[vgpu]"):
                    keep.append(l)
            body = "\n".join(keep)
            gpu = re.findall(r"GPU Time in seconds:\s*\n?\s*([0-9.]+)", out)
            rec = dict(prog=pname, side=sname, run=r, rc=rc, wall=round(wall, 6), gpu_time=float(gpu[0]) if gpu else None,
                       out_hash=hashlib.sha256(body.encode()).hexdigest()[:16],
                       summary=[l for l in err.splitlines() if l.startswith("host_gpu_verify:") or (l.startswith("host_gpu: ") and ("launches ran" in l or "MiB went" in l))][-4:],
                       idiom=sorted(set(l for l in err.splitlines() if "square root idiom" in l)))
            res.append(rec)
            json.dump(res, open(a.out, "w"), indent=1)
            print(pname, sname, r, rc, rec["wall"], rec["gpu_time"], rec["out_hash"], flush=True)
