#!/usr/bin/env python3
"""Round sp60a: interleaved timing of whole programs under several environments
("forms") of ONE build.  Each run is appended to a JSON-lines file; tables.py
reads it.

  ab.py --build B --corp C --src S --forms a,b,c,m --threads default --out runs.jsonl
  ab.py ... --threads 1 --workers 3 --cpus 2,6,10     # three pinned programs side by side

Forms (the only difference between them is the environment):
  a  default                        (nothing set)
  A  the same again                 (the A/A control of the identity runs)
  b  VGPU_MEMO=0                    (the cache's own switch, off)
  c  VGPU_HOST_GPU=exact            (no NVIDIA driver on this box: the mode reports itself off)
  L  VGPU_HOST_GPU=exact VGPU_HOST_GPU_LOG=1   (c with the mode's own account of what it did; its host_gpu lines are kept apart)
  m  VGPU_MEMO=1                    (the cache on, entries in memory)
  w  VGPU_MEMO=1 VGPU_MEMO_DIR=<a directory per program, kept for the pass>   (run 0 is cold, the later runs are warm)
  M  VGPU_MEMO=1 VGPU_MEMO_BUDGET=inf
Runs of one program alternate a,b,c,m / m,c,b,a / ...  The machine-wide builder
lock is held while anything runs and given up between runs after --hold seconds.
With --identity every run also writes VGPU_MEMO_FINAL_HASH, VGPU_COUNTERS_FILE and
VGPU_KERNEL_DIGEST, whose hashes go into the record (such runs are not for timing).
"""
import argparse, fcntl, hashlib, json, os, queue, re, resource, shutil, signal
import subprocess, sys, tempfile, threading, time

FORMS = {
    "a": {},
    "A": {},
    "b": {"VGPU_MEMO": "0"},
    "c": {"VGPU_HOST_GPU": "exact"},
    "L": {"VGPU_HOST_GPU": "exact", "VGPU_HOST_GPU_LOG": "1"},
    "m": {"VGPU_MEMO": "1"},
    "w": {"VGPU_MEMO": "1", "VGPU_MEMO_DIR": "{tmp}/memo-{name}"},
    "M": {"VGPU_MEMO": "1", "VGPU_MEMO_BUDGET": "inf"},
}


class Gate:
    """Holds the builder lock while runs are in flight; gives it up between runs once it has been held --hold seconds."""

    def __init__(self, path, hold):
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        self.hold = hold
        self.cv = threading.Condition()
        self.active = 0
        self.since = None
        self.closing = False

    def enter(self):
        with self.cv:
            while True:
                if self.since is not None and not self.closing and time.monotonic() - self.since > self.hold:
                    self.closing = True
                if self.closing:
                    if self.active == 0:
                        fcntl.flock(self.fd, fcntl.LOCK_UN)
                        self.since = None
                        self.closing = False
                        self.cv.notify_all()
                        time.sleep(1.0)
                        continue
                    self.cv.wait()
                    continue
                if self.since is None:
                    fcntl.flock(self.fd, fcntl.LOCK_EX)
                    self.since = time.monotonic()
                self.active += 1
                return

    def leave(self):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()

    def done(self):
        with self.cv:
            if self.since is not None and self.active == 0:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                self.since = None


def file_hash(path, strip_pid=False):
    try:
        data = open(path, "rb").read()
    except OSError:
        return None
    if strip_pid:
        data = re.sub(rb'"pid"\s*:\s*\d+,?\s*', b"", data)
        data = re.sub(rb"\bpid[= ]\d+", b"", data)
    return hashlib.sha256(data).hexdigest()[:12]


def run_one(real, a, entry, form, cpus, threads):
    corpus, name, cwd, argv, copy_cwd = entry
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    env.update(VGPU_QUIET="1", VGPU_GPU=a.gpu, LD_LIBRARY_PATH=os.path.join(os.path.abspath(a.build), "shim"))
    if threads != "default":
        env["VGPU_THREADS"] = threads
    env.update({k: v.format(tmp=a.tmp, name=name) for k, v in FORMS[form].items()})
    side = tempfile.mkdtemp(prefix="sp60a-", dir=a.tmp)
    if a.identity:
        env["VGPU_MEMO_FINAL_HASH"] = os.path.join(side, "final")
        env["VGPU_COUNTERS_FILE"] = os.path.join(side, "counters")
        env["VGPU_KERNEL_DIGEST"] = os.path.join(side, "digest")
    if form in ("m", "M", "w"):
        env["VGPU_MEMO_STATS_FILE"] = os.path.join(side, "memo")
    scratch = None
    if copy_cwd:
        scratch = tempfile.mkdtemp(prefix="real-", dir=a.tmp)
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        with open(os.path.join(scratch, "filelist"), "w") as f:
            f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    cmd = (["taskset", "-c", cpus] if cpus else []) + argv
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    to = False
    try:
        out, _ = p.communicate(timeout=a.timeout)
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
            if os.path.isfile(fp) and f != "output.dat":
                extra += f.encode() + open(fp, "rb").read()
        shutil.rmtree(scratch, ignore_errors=True)
    hg_lines = [l for l in text.splitlines() if l.startswith("host_gpu:")]
    text = "\n".join(l for l in text.splitlines() if not l.startswith("host_gpu:"))
    h, checks = real.normalise(text, extra)
    rec = {"prog": corpus + "/" + name, "threads": threads, "form": form, "wall": round(wall, 4), "rc": p.returncode,
           "timeout": to, "out": h, "host_gpu": [l for l in hg_lines if "stays in the simulator" not in l][-4:],
           "hg_kept": sum(1 for l in hg_lines if "stays in the simulator" in l)}
    if a.identity:
        rec["final"] = file_hash(os.path.join(side, "final"), True)
        rec["counters"] = file_hash(os.path.join(side, "counters"), True)
        rec["digest"] = file_hash(os.path.join(side, "digest"), True)
    if form in ("m", "M", "w"):
        try:
            rec["memo"] = open(os.path.join(side, "memo")).read().strip()[-1500:]
        except OSError:
            rec["memo"] = None
    shutil.rmtree(side, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--corp", required=True)
    ap.add_argument("--src", required=True, help="the source tree (for tools/bench/real/real.py: the programs and their arguments)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--gpu", default="nvidia/a10g")
    ap.add_argument("--forms", default="a,b,c")
    ap.add_argument("--threads", default="default")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--mid-secs", type=float, default=4.0)
    ap.add_argument("--mid-runs", type=int, default=3)
    ap.add_argument("--long-secs", type=float, default=30.0)
    ap.add_argument("--long-runs", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--corpora", default="polybench,rodinia,hecbench")
    ap.add_argument("--programs", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cpus", default="", help="one taskset list per worker, separated by ';' (empty: not pinned)")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    ap.add_argument("--hold", type=float, default=200.0)
    a = ap.parse_args()
    os.nice(19)
    sys.path.insert(0, os.path.join(a.src, "tools/bench/real"))
    import real
    forms = a.forms.split(",")
    man = real.manifest(a.corp, a.corpora.split(","))
    only = set(x for x in a.programs.split(",") if x)
    skip = set(x for x in a.skip.split(",") if x)
    man = [e for e in man if (not only or e[1] in only) and e[1] not in skip]
    os.makedirs(a.tmp, exist_ok=True)
    gate = Gate(a.lock, a.hold)
    q = queue.Queue()
    for e in man:
        q.put(e)
    wlock = threading.Lock()
    cpus = a.cpus.split(";") if a.cpus else [""] * a.workers

    def worker(w):
        while True:
            try:
                e = q.get_nowait()
            except queue.Empty:
                return
            nruns, i, dead = a.runs, 0, set()
            while i < nruns:
                for f in (forms if i % 2 == 0 else forms[::-1]):
                    if f in dead:
                        continue
                    gate.enter()
                    try:
                        rec = run_one(real, a, e, f, cpus[w], a.threads)
                    finally:
                        gate.leave()
                    rec["i"] = i
                    if rec["timeout"]:
                        dead.add(f)
                    with wlock:
                        with open(a.out, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                    if i == 0 and f == forms[0]:
                        if rec["wall"] > a.long_secs:
                            nruns = min(nruns, a.long_runs)
                        elif rec["wall"] > a.mid_secs:
                            nruns = min(nruns, a.mid_runs)
                i += 1
            print("%s done" % e[1], flush=True)

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    gate.done()


if __name__ == "__main__":
    main()
