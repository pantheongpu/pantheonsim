#!/usr/bin/env python3
"""Program history A/B for turbo mode (docs/perf-design/12-host-gpu-turbo.md section 17), on a machine with the card.

Per program and repetition, interleaved: native (the real runtime), sim (the mode off), turbo (VGPU_HOST_GPU=turbo, no history), then the first, second and third run
with VGPU_HOST_GPU_HISTORY pointing at a directory made fresh for the repetition.  Median wall seconds over --runs repetitions.  One untimed run of each history form with
VGPU_HOST_GPU_LOG=1 gives the history line (what the record said before and after, held/released) and the exit status.  Output text is compared with the sim run's
(timing lines dropped, as run.py does).

  history_bench.py --build /path/to/build --corp /path/to/corp --gpu nvidia/a10g --runs 5 \\
      --programs rodinia/bfs,hecbench/atan2-cuda --json out.json"""
import argparse, json, os, re, shutil, statistics, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "real"))
import real  # noqa: E402
import run as hostrun  # noqa: E402


def one(prog, env_extra, base_env, timeout):
    _c, _n, cwd, argv, copy_cwd = prog
    return hostrun.run_program(argv, cwd, copy_cwd, env_extra, base_env, timeout)


def text_of(r):
    return hostrun.program_text(r["stdout"], r["stderr"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--corp", required=True)
    ap.add_argument("--gpu", default="nvidia/a10g")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", default="4")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--programs", required=True)
    ap.add_argument("--corpora", default="polybench,rodinia,hecbench")
    ap.add_argument("--card", default="")   # VGPU_HOST_GPU_HISTORY_CARD for a run that has no sysfs
    ap.add_argument("--native-ld", default="/usr/local/cuda-13.0/lib64")
    ap.add_argument("--forms", default="native,sim,turbo,hist1,hist2,hist3")   # leave sim out for a program the simulator needs minutes for
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    progs = real.manifest(os.path.abspath(a.corp), a.corpora.split(","))
    want = a.programs.split(",")
    progs = [p for p in progs if p[0] + "/" + p[1] in want or p[1] in want]
    base = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    sim_env = dict(base, VGPU_GPU=a.gpu, LD_LIBRARY_PATH=os.path.join(os.path.abspath(a.build), "shim"), VGPU_THREADS=a.threads, VGPU_QUIET="1")
    nat_env = dict(base, LD_LIBRARY_PATH=a.native_ld)
    forms = a.forms.split(",")
    res = {}
    for prog in progs:
        name = prog[0] + "/" + prog[1]
        walls = {f: [] for f in forms}
        rcs = {f: [] for f in forms}
        ref = None
        same = {}
        for rep in range(a.runs):
            hdir = tempfile.mkdtemp(prefix="hist-bench-")
            try:
                for f in forms:
                    env = {}
                    if f == "native":
                        r = one(prog, {}, nat_env, a.timeout)
                    else:
                        if f != "sim":
                            env["VGPU_HOST_GPU"] = "turbo"
                        if f.startswith("hist"):
                            env["VGPU_HOST_GPU_HISTORY"] = hdir
                            if a.card:
                                env["VGPU_HOST_GPU_HISTORY_CARD"] = a.card
                        r = one(prog, env, sim_env, a.timeout)
                    walls[f].append(round(r["wall"], 4))
                    rcs[f].append(r["rc"])
                    if f == "sim" and ref is None:
                        ref = text_of(r)
                    if f not in ("native", "sim") and ref is not None:
                        same[f] = same.get(f, True) and text_of(r) == ref
            finally:
                shutil.rmtree(hdir, ignore_errors=True)
        lines = {}
        hdir = tempfile.mkdtemp(prefix="hist-bench-")
        try:
            for f in ("hist1", "hist2", "hist3"):
                env = {"VGPU_HOST_GPU": "turbo", "VGPU_HOST_GPU_LOG": "1", "VGPU_HOST_GPU_HISTORY": hdir}
                if a.card:
                    env["VGPU_HOST_GPU_HISTORY_CARD"] = a.card
                r = one(prog, env, sim_env, a.timeout)
                keep = [l for l in r["stderr"].splitlines() if l.startswith(("host_gpu_history:", "host_gpu_turbo:", "host_gpu: ")) and ("history" in l or "failed on the card" in l or "host_gpu_turbo" in l)]
                lines[f] = {"rc": r["rc"], "lines": keep[-6:]}
        finally:
            shutil.rmtree(hdir, ignore_errors=True)
        res[name] = {"wall": walls, "rc": rcs, "median": {f: round(statistics.median(walls[f]), 4) for f in forms}, "output_same_as_sim": same, "log": lines}
        print("%-28s %s  same=%s" % (name, " ".join("%s %.3f" % (f, res[name]["median"][f]) for f in forms), all(same.values())), flush=True)
    json.dump({"runs": a.runs, "gpu": a.gpu, "programs": res}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
