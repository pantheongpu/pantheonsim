#!/usr/bin/env python3
"""Round sp65: a launch trace of every corpus program, no card needed.

  trace_corpus.py --build B --corp C --out DIR --mode exact|turbo [--programs a,b] [--skip a,b] [--timeout 270] [--lock /tmp/pantheonsim-builder.lock]

Each program runs once with VGPU_HOST_GPU=<mode>, VGPU_HOST_GPU_MAX_MIB=0 (every launch is refused for its size, so nothing goes to a card, the
driver is never loaded and no launch is sampled) and VGPU_HOST_GPU_TRACE: the trace then holds every allocation, free, host copy and fill and
every launch, with the simulator's own time for it, whether the mode admits its kernel, its block size, static instruction count and loop flag.
residency_replay.py replays the rules on it. Every run is under the machine's lock at nice 19 (one hold per program, cut at --timeout).
"""
import argparse, os, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../tools/bench/real"))
from real import manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True); ap.add_argument("--corp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--mode", required=True); ap.add_argument("--programs", default=""); ap.add_argument("--skip", default="")
ap.add_argument("--timeout", type=int, default=270); ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
ap.add_argument("--corpora", default="polybench,rodinia,hecbench")
a = ap.parse_args()
a.corp, a.out = os.path.abspath(a.corp), os.path.abspath(a.out)   # (the programs run in their own directories)
os.makedirs(a.out, exist_ok=True)
want = set(x for x in a.programs.split(",") if x); skip = set(x for x in a.skip.split(",") if x)
for corpus, name, cwd, argv, copy_cwd in manifest(a.corp, a.corpora.split(",")):
    key = "%s/%s" % (corpus, name)
    if (want and key not in want and name not in want) or key in skip or name in skip:
        continue
    tr = os.path.join(a.out, "%s_%s.%s.trace" % (corpus, name, a.mode))
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/a10g", VGPU_THREADS="4", LD_LIBRARY_PATH=os.path.join(os.path.abspath(a.build), "shim"),
               VGPU_HOST_GPU=a.mode, VGPU_HOST_GPU_MAX_MIB="0", VGPU_HOST_GPU_SAMPLE="0", VGPU_HOST_GPU_TRACE=tr)
    scratch = None
    if copy_cwd:
        scratch = tempfile.mkdtemp(prefix="sp65-", dir=os.path.expanduser("~/.cache/sp65"))
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        with open(os.path.join(scratch, "filelist"), "w") as f:
            f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    t0 = time.monotonic()
    p = subprocess.run(["flock", a.lock, "nice", "-n", "19", "timeout", str(a.timeout)] + argv, cwd=cwd, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wall = time.monotonic() - t0
    if scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    n = sum(1 for l in open(tr) if l.startswith("L ")) if os.path.exists(tr) else -1
    print("%-32s rc=%d wall=%.1f launches=%d" % (key, p.returncode, wall, n), flush=True)
    time.sleep(2)   # let the other user of the lock in between programs
