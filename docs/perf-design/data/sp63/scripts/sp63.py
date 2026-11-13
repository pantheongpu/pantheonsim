#!/usr/bin/env python3
"""Round sp63's driver (server3), from round sp57's: two builds side by side, the JIT cold, warm and on its second run.
  sp63.py --corp C --base B0 --build B1 --work W --threads 1|default --out o.json [--runs 5] [--long-secs 1 --long-runs 3]
          [--sides Ba,Bbc,Na,Nbc,Nbw,Nbs] [--progs corpus/name,...] [--skip ...] [--cpus 12-15] [--env K=V,...]
A side is a build (B: --base, N: --build) and one of round sp57's: a = no VGPU_JIT, bc = VGPU_JIT=1 cold (the cache emptied before every run),
bw = warm (the cache filled by one untimed run with VGPU_JIT_SYNC=1), and bs = the second run (before every timed run the cache is emptied and the
program run once, untimed, with the defaults: what a user's second run sees).  Every run writes VGPU_COUNTERS_FILE and VGPU_MEMO_FINAL_HASH.
Round sp57's description follows.
  sp57.py --corp C --build B --work W --threads 1|default --out o.json [--runs 5] [--long-secs 4 --long-runs 3]
          [--sides a,bc,bw,cc,cw,dc,dw] [--progs corpus/name,...] [--skip ...] [--cpus 12-15] [--env K=V,...]
Sides: a = no VGPU_JIT; b = VGPU_JIT=1; c = b + VGPU_JIT_LINK=1; d = c + VGPU_JIT_FUSE=1.  A second letter: c = cold (the side's
cache directory is emptied before every run), w = warm (the cache was filled by one untimed run with VGPU_JIT_SYNC=1 and statistics,
whose numbers are kept as "fill").  Nothing else of the JIT is set: threshold, asynchronous compile and the rest are the defaults.
Every run (timed ones too) writes VGPU_COUNTERS_FILE and VGPU_MEMO_FINAL_HASH; their hashes, the normalised output's and the exit
status are kept per run.  The sides of a program are interleaved (a round in order, the next reversed); the machine lock is held
over consecutive runs and given up every four minutes.  real.py of tools/bench/real must lie beside this file."""
import argparse, fcntl, hashlib, importlib.util, json, os, re, shutil, signal, statistics, subprocess, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("real", os.path.join(HERE, "real.py"))
real = importlib.util.module_from_spec(spec); spec.loader.exec_module(real)
LOCK = "/tmp/pantheonsim-builder.lock"
VAR = {"a": {}, "b": {"VGPU_JIT": "1"}, "c": {"VGPU_JIT": "1", "VGPU_JIT_LINK": "1"},
       "d": {"VGPU_JIT": "1", "VGPU_JIT_LINK": "1", "VGPU_JIT_FUSE": "1"}}

class Slot:
    """The machine lock, held over consecutive runs for at most `span` seconds (other sessions wait for it: one hold a run would queue behind them every time)."""
    def __init__(self, span): self.f = None; self.span = span; self.holds = 0; self.waited = 0.0
    def ensure(self):
        if self.f and time.monotonic() - self.t0 > self.span: self.release()
        if not self.f:
            t = time.monotonic(); self.f = open(LOCK, "a"); fcntl.flock(self.f, fcntl.LOCK_EX); self.t0 = time.monotonic(); self.holds += 1; self.waited += self.t0 - t
    def release(self):
        if self.f:
            fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close(); self.f = None; time.sleep(2)   # (flock is not fair: let a waiter in)
SLOT = Slot(240)

def h(b): return hashlib.sha256(b).hexdigest()[:12]

def run1(a, prog, extra, work):
    SLOT.ensure()
    corpus, name, cwd, argv, cp = prog
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    cf, ff = os.path.join(work, "cnt"), os.path.join(work, "fin")
    for f in (cf, ff):
        if os.path.exists(f): os.unlink(f)
    env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", LD_LIBRARY_PATH=a.build + "/shim", VGPU_COUNTERS_FILE=cf, VGPU_MEMO_FINAL_HASH=ff)
    if a.threads != "default": env["VGPU_THREADS"] = a.threads
    env.update(a.envd); env.update(extra)
    scratch = None
    if cp:
        scratch = tempfile.mkdtemp(prefix="run-", dir=work)
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        with open(os.path.join(scratch, "filelist"), "w") as f:
            f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    cmd = (["taskset", "-c", a.cpus] if a.cpus else []) + ["nice", "-n", "19"] + argv
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    to = False
    try: out, _ = p.communicate(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); out, _ = p.communicate(); to = True
    wall = time.monotonic() - t0
    eb = b""
    if scratch:
        for f in sorted(x for x in os.listdir(scratch) if x != "filelist"):
            fp = os.path.join(scratch, f)
            if os.path.isfile(fp) and f != "output.dat": eb += f.encode() + open(fp, "rb").read()
        shutil.rmtree(scratch, ignore_errors=True)
    text = "\n".join(l for l in out.decode("utf-8", "replace").splitlines() if not l.startswith("jit: "))
    oh, _ = real.normalise(text, eb)
    cnt = re.sub(r'"pid":\d+,', "", open(cf).read()) if os.path.exists(cf) else ""
    fin = open(ff).read() if os.path.exists(ff) else ""
    return dict(wall=round(wall, 4), rc=p.returncode, to=to, out=oh, cnt=h(cnt.encode()), launches=cnt.count("\n"), fin=h(fin.encode()),
                fin_lines=fin.count("\n"), load=os.getloadavg()[0])

def objs(d): return sum(1 for x in os.listdir(d) if x.endswith(".so")) if os.path.isdir(d) else 0
def dsize(d): return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)
def fresh(d):
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d, mode=0o700)

def parse_stats(path):
    d = {}
    if not os.path.exists(path): return d
    t = open(path).read()
    m = re.search(r"(\d+) units \((\d+) regions\), (\d+) ready, (\d+) cache hits, (\d+) compiles \(([\d.]+) ms\), (\d+) failures", t)
    if m: d.update(units=int(m[1]), regions=int(m[2]), ready=int(m[3]), hits=int(m[4]), compiles=int(m[5]), compile_ms=float(m[6]), failures=int(m[7]))
    m = re.search(r"(\d+) region calls ran (\d+) warp instructions \((\d+) computed by generated code\)", t)
    if m: d.update(region_calls=int(m[1]), region_insts=int(m[2]), generated=int(m[3]))
    m = re.search(r"(\d+) declined, (\d+) accepted", t)
    if m: d.update(declined=int(m[1]), accepted=int(m[2]))
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corp", required=True); ap.add_argument("--base", required=True); ap.add_argument("--build", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--threads", default="1"); ap.add_argument("--out", required=True); ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--long-secs", type=float, default=1.0); ap.add_argument("--long-runs", type=int, default=3)
    ap.add_argument("--sides", default="Ba,Bbc,Na,Nbc,Nbw,Nbs"); ap.add_argument("--progs", default=""); ap.add_argument("--skip", default="")
    ap.add_argument("--cpus", default=""); ap.add_argument("--timeout", type=float, default=300); ap.add_argument("--env", default="")
    ap.add_argument("--max-wall", type=float, default=1e9, help="leave out a program whose first run is longer")
    a = ap.parse_args()
    a.envd = dict(x.split("=", 1) for x in a.env.split(",") if x)
    a.corp, a.work = (os.path.abspath(x) for x in (a.corp, a.work))
    builds = {"B": os.path.abspath(a.base), "N": os.path.abspath(a.build)}
    os.makedirs(a.work, exist_ok=True)
    progs = real.manifest(a.corp, ["polybench", "rodinia", "hecbench"])
    want = set(x for x in a.progs.split(",") if x); skip = set(x for x in a.skip.split(",") if x)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    sides = a.sides.split(",")
    def run_on(s, prog, extra):
        a.build = builds[s[0]]
        return run1(a, prog, extra, a.work)
    for prog in progs:
        pid = prog[0] + "/" + prog[1]
        if (want and pid not in want) or pid in skip or pid in res: continue
        R = dict(sides={s: [] for s in sides}, fill={})
        cache = lambda s: os.path.join(a.work, "jc-" + s)
        first = run_on("Na", prog, {})
        if first["to"] or first["wall"] > a.max_wall:
            res[pid] = dict(skipped="timeout" if first["to"] else "long", first=first); print("%-34s skipped (%.1f s)" % (pid, first["wall"]), flush=True); continue
        for s in sides:
            if s[1:] == "bw":
                fresh(cache(s)); sf = os.path.join(a.work, "stats"); open(sf, "w").close()
                r = run_on(s, prog, dict(VAR["b"], VGPU_JIT_CACHE=cache(s), VGPU_JIT_SYNC="1", VGPU_JIT_STATS="1", VGPU_JIT_STATS_FILE=sf))
                r.update(stats=parse_stats(sf), objs=objs(cache(s)), bytes=dsize(cache(s))); R["fill"][s] = r
        n = a.long_runs if first["wall"] > a.long_secs else a.runs
        R["first"] = first; R["runs"] = n; R["objs"] = {s: [] for s in sides}; R["pre"] = {s: [] for s in sides}
        def one(s):
            e = dict(VAR[s[1]])
            if s[1] != "a":
                e["VGPU_JIT_CACHE"] = cache(s)
                if s[1:] in ("bc", "bs"): fresh(cache(s))
                if s[1:] == "bs": R["pre"][s].append(run_on(s, prog, e))
            R["sides"][s].append(run_on(s, prog, e))
            if s[1] != "a": R["objs"][s].append(objs(cache(s)))
        for i in range(n):
            for s in (sides if i % 2 == 0 else sides[::-1]): one(s)
        for s in sides:
            if s[1] != "a": shutil.rmtree(cache(s), ignore_errors=True)
        res[pid] = R
        json.dump(res, open(a.out + ".tmp", "w"), separators=(",", ":")); os.replace(a.out + ".tmp", a.out)
        med = {s: statistics.median(x["wall"] for x in v) for s, v in R["sides"].items()}
        ref = R["sides"][sides[0]][0]
        allruns = [x for v in R["sides"].values() for x in v] + [x for v in R["pre"].values() for x in v] + list(R["fill"].values())
        same = all((x["out"], x["cnt"], x["fin"], x["rc"]) == (ref["out"], ref["cnt"], ref["fin"], ref["rc"]) for x in allruns)
        print("%-34s n=%d %s same=%s objs=%s" % (pid, n, " ".join("%s=%.3f" % (s, med[s]) for s in sides), same,
              {k: v[-1] for k, v in R["objs"].items() if v}), flush=True)
    SLOT.release()
    print("lock: %d holds, %.0f s waited" % (SLOT.holds, SLOT.waited), flush=True)
if __name__ == "__main__": main()
