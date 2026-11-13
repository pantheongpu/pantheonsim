#!/usr/bin/env python3
"""Round sp66: the census of what keeps turbo mode's launches in the simulator.

Runs every program of the corpora (tools/bench/real/real.py's manifest: PolyBench/GPU, Rodinia, the reduced HeCBench list, the
CUDA Samples that run in the simulator) once with VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 against the stand-in driver
(nvidia/tests/e2e/host_gpu_stub_driver.c, which refuses every module: a launch the mode sends "fails on the card" and the
simulator runs it, so the program runs as it would with the mode off and every launch's simulator seconds are measured), and
keeps the mode's lines: host_gpu_kept (per kernel and reason: launches, simulator seconds), host_gpu_refused, host_gpu_turbo,
host_gpu_refusals. One JSON per program in --out. The runs hold the machine's build lock (flock), several to a hold of at most --hold seconds,
and each is cut at --timeout.

  census.py --shim BUILD/shim --stub libstubcuda.so --corp CORP --out DIR [--corpora polybench,rodinia,hecbench,samples]
            [--programs a,b] [--timeout 280] [--gpu nvidia/a10g --sm 86 --mp 80] [--threads 3] [--lock /tmp/pantheonsim-builder.lock]
"""
import argparse, fcntl, json, os, re, shutil, signal, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools", "bench", "real"))
import real  # noqa: E402

KEEP = re.compile(r"^(host_gpu_kept|host_gpu_refused|host_gpu_turbo|host_gpu_refusals|host_gpu_sampled): ")


def fields(line):
    head, _, rest = line.partition(": ")
    if head == "host_gpu_kept":
        m = re.match(r"kernel=(\S+) launches=(\d+) sim_s=(\S+) reason=(.*)$", rest)
        return head, {"kernel": m.group(1), "launches": int(m.group(2)), "sim_s": float(m.group(3)), "reason": m.group(4)}
    if head == "host_gpu_refused":
        m = re.match(r"kernel=(\S+) class=(\S+) insn=(.*)$", rest)
        return head, {"kernel": m.group(1), "class": m.group(2), "insn": m.group(3)}
    return head, dict(kv.split("=", 1) for kv in rest.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", required=True)
    ap.add_argument("--stub", required=True)
    ap.add_argument("--corp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpora", default="polybench,rodinia,hecbench,samples")
    ap.add_argument("--programs", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--timeout", type=float, default=280)
    ap.add_argument("--gpu", default="nvidia/a10g")
    ap.add_argument("--sm", default="86")
    ap.add_argument("--mp", default="80")
    ap.add_argument("--threads", default="3")
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    ap.add_argument("--env", action="append", default=[], help="extra NAME=VALUE for the run")
    ap.add_argument("--hold", type=float, default=240, help="seconds of runs to a hold of the lock")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fd = os.open(a.lock, os.O_RDWR | os.O_CREAT, 0o666)
    since = None
    want = set(x for x in a.programs.split(",") if x)
    skip = set(x for x in a.skip.split(",") if x)
    for corpus, name, cwd, argv, copy_cwd in real.manifest(a.corp, a.corpora.split(",")):
        key = corpus + "/" + name
        if (want and key not in want and name not in want) or key in skip or name in skip:
            continue
        dest = os.path.join(a.out, corpus + "__" + name.replace("/", "_") + ".json")
        if os.path.exists(dest):
            continue
        env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_") and not k.startswith("STUB_")}
        env.update(VGPU_GPU=a.gpu, LD_LIBRARY_PATH=a.shim, VGPU_THREADS=a.threads, VGPU_HOST_GPU="turbo", VGPU_HOST_GPU_LOG="1",
                   VGPU_HOST_GPU_LIBCUDA=a.stub, STUB_SM=a.sm, STUB_MP=a.mp)
        for kv in a.env:
            k, _, v = kv.partition("=")
            env[k] = v
        scratch = None
        if copy_cwd:
            scratch = tempfile.mkdtemp(prefix="census-")
            dd = os.path.join(os.path.dirname(argv[0]), "data")
            with open(os.path.join(scratch, "filelist"), "w") as f:
                f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
            cwd = scratch
        if since is not None and time.monotonic() - since > a.hold - min(a.timeout, a.hold / 2):   # (a hold is given up before a run could take it past --hold)
            fcntl.flock(fd, fcntl.LOCK_UN)
            since = None
            time.sleep(2)
        if since is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
            since = time.monotonic()
        t0 = time.monotonic()
        p = subprocess.Popen(["timeout", "-s", "KILL", "%g" % a.timeout, "nice", "-n", "19"] + argv, cwd=cwd, env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, start_new_session=True)
        timed_out = False
        try:
            _, err = p.communicate(timeout=a.timeout + 60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            _, err = p.communicate()
            timed_out = True
        wall = time.monotonic() - t0
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        timed_out = timed_out or p.returncode in (137, -9)
        rec = {"program": key, "rc": p.returncode, "timeout": timed_out, "wall_incl_lock_s": round(wall, 3),
               "kept": [], "refused": [], "turbo": {}, "refusals": {}, "sampled": {}}
        for line in err.decode("utf-8", "replace").splitlines():
            if not KEEP.match(line):
                continue
            head, f = fields(line)
            if head == "host_gpu_kept": rec["kept"].append(f)
            elif head == "host_gpu_refused": rec["refused"].append(f)
            elif head == "host_gpu_turbo": rec["turbo"] = f
            elif head == "host_gpu_refusals": rec["refusals"] = f
            elif head == "host_gpu_sampled": rec["sampled"] = f
        json.dump(rec, open(dest, "w"), separators=(",", ":"))
        tot = sum(k["sim_s"] for k in rec["kept"])
        print("%-45s rc=%s%s kept sim_s=%.2f kernels=%d" % (key, p.returncode, " TIMEOUT" if timed_out else "", tot, len(rec["kept"])), flush=True)


if __name__ == "__main__":
    main()
