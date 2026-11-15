#!/usr/bin/env python3
"""Round sp75: the census of what keeps a launch in the simulator, per launch, against the stand-in driver.

Runs programs of the corpora (tools/bench/real/real.py's manifest) with VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 against the stand-in driver
(nvidia/tests/e2e/host_gpu_stub_driver.c, which refuses every module: a launch the mode sends "fails on the card" and the simulator runs it),
once per form, the forms one after another for each program, and keeps what the mode said: the summary counts, the host_gpu_kept reasons, the
samples (taken, sent, lost, their cost), the host_gpu_nosample lines and the decision lines of sampled launches. A form is NAME=BUILD[,ENV=VALUE...].
With --cut N a run is ended once N decisions have been logged (a long launch's decision is made before it runs).

  census.py --corp CORP --stub libstubcuda.so --out DIR --form old=/path/build-old --form new=/path/build --form knob=/path/build,VGPU_HOST_GPU_SAMPLE_BLOCK=0
            [--programs a,b] [--skip a,b] [--timeout 150] [--runs 1] [--cut 0] [--threads 3] [--gpu nvidia/a10g --sm 86 --mp 80]
"""
import argparse, fcntl, json, os, re, shutil, signal, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools", "bench", "real"))
import real  # noqa: E402

DECISION = re.compile(r"host_gpu_nosample:|host_gpu: \S+ sampled|stays in the simulator \(sim|starts the driver")


def run(argv, cwd, env, timeout, cut, lock):
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        t0 = time.monotonic()
        p = subprocess.Popen(["nice", "-n", "19"] + argv, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        state = {"timeout": False}

        def killer():
            if p.wait_done.wait(timeout):
                return
            state["timeout"] = True
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        p.wait_done = threading.Event()
        threading.Thread(target=killer, daemon=True).start()
        lines, n, was_cut = [], 0, False
        for raw in p.stderr:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if "host_gpu" not in line or line.startswith("host_gpu:   "):
                continue
            lines.append(line)
            if DECISION.search(line):
                n += 1
                if cut and n >= cut and not was_cut:
                    was_cut = True
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        rc = p.wait()
        p.wait_done.set()
        return {"rc": rc, "wall_s": round(time.monotonic() - t0, 4), "timeout": state["timeout"], "cut": was_cut, "lines": lines}
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def digest(r):
    """What a run decided, from its lines."""
    d = {"rc": r["rc"], "wall_s": r["wall_s"], "timeout": r["timeout"], "cut": r["cut"], "kept": [], "nosample": [], "sampled_lines": [], "starts": []}
    for line in r["lines"]:
        m = re.match(r"host_gpu: (\d+) launches ran on the card, (\d+) in the simulator \(.*?(\d+) too small, (\d+) unsupported launch shapes, (\d+) failed on the card\)", line)
        if m:
            d["launches_sim"], d["too_small"], d["failed"] = int(m.group(2)), int(m.group(3)), int(m.group(5))
        m = re.match(r"host_gpu_sampled: sampled=(\d+) sent=(\d+) timeouts=(\d+) cost_s=(\S+)", line)
        if m:
            d["sampled"], d["sample_sent"], d["sample_timeouts"], d["sample_cost_s"] = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
        m = re.match(r"host_gpu_kept: kernel=(\S+) launches=(\d+) sim_s=(\S+) reason=(.*)$", line)
        if m:
            d["kept"].append({"kernel": m.group(1), "launches": int(m.group(2)), "sim_s": float(m.group(3)), "reason": m.group(4)})
        if line.startswith("host_gpu_nosample: "):
            d["nosample"].append(dict(kv.split("=", 1) for kv in line[len("host_gpu_nosample: "):].split(" reason=")[0].split()) | {"reason": line.split(" reason=")[1]})
        m = re.match(r"host_gpu: (\S+) sampled(.*)$", line)
        if m:
            d["sampled_lines"].append(m.group(1) + m.group(2))
        m = re.match(r"host_gpu: (\S+) starts the driver: (.*)$", line)
        if m:
            d["starts"].append(m.group(1) + ": " + m.group(2))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corp", required=True)
    ap.add_argument("--stub", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--form", action="append", required=True)
    ap.add_argument("--corpora", default="polybench,rodinia,hecbench,samples")
    ap.add_argument("--programs", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--timeout", type=float, default=150)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--cut", type=int, default=0)
    ap.add_argument("--gpu", default="nvidia/a10g")
    ap.add_argument("--sm", default="86")
    ap.add_argument("--mp", default="80")
    ap.add_argument("--threads", default="3")
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    forms = []
    for f in a.form:
        name, _, rest = f.partition("=")
        parts = rest.split(",")
        forms.append((name, parts[0], dict(kv.split("=", 1) for kv in parts[1:])))
    want = set(x for x in a.programs.split(",") if x)
    skip = set(x for x in a.skip.split(",") if x)
    for corpus, name, cwd, argv, copy_cwd in real.manifest(a.corp, a.corpora.split(",")):
        key = corpus + "/" + name
        if (want and key not in want and name not in want) or key in skip or name in skip:
            continue
        dest = os.path.join(a.out, corpus + "__" + name.replace("/", "_") + ".json")
        if os.path.exists(dest):
            continue
        rec = {"program": key, "forms": {n: [] for n, _, _ in forms}}
        for i in range(a.runs):
            for fname, build, extra in (forms if i % 2 == 0 else forms[::-1]):
                env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_") and not k.startswith("STUB_")}
                env.update(VGPU_GPU=a.gpu, LD_LIBRARY_PATH=os.path.join(build, "shim"), VGPU_THREADS=a.threads, VGPU_HOST_GPU="turbo", VGPU_HOST_GPU_LOG="1",
                           VGPU_HOST_GPU_LIBCUDA=a.stub, STUB_SM=a.sm, STUB_MP=a.mp)
                env.update(extra)
                scratch, rcwd = None, cwd
                if copy_cwd:
                    scratch = tempfile.mkdtemp(prefix="census-")
                    dd = os.path.join(os.path.dirname(argv[0]), "data")
                    with open(os.path.join(scratch, "filelist"), "w") as f:
                        f.write("".join(os.path.join(dd, "cane4_%d.db\n" % j) for j in range(4)))
                    rcwd = scratch
                r = run(argv, rcwd, env, a.timeout, a.cut, a.lock)
                if scratch:
                    shutil.rmtree(scratch, ignore_errors=True)
                rec["forms"][fname].append(digest(r))
        json.dump(rec, open(dest, "w"), separators=(",", ":"))
        print("%-45s %s" % (key, "  ".join("%s %.2fs%s" % (n, min(x["wall_s"] for x in rec["forms"][n]), " T" if any(x["timeout"] for x in rec["forms"][n]) else "") for n, _, _ in forms)), flush=True)


if __name__ == "__main__":
    main()
