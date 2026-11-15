#!/usr/bin/env python3
"""Round sp75: the output of programs under two builds, with the mode off, in exact mode and in turbo mode against the stand-in driver (every
launch is simulated: the output must be the simulator's). Prints, per program and form, whether stdout is byte for byte the same in both builds
and the same as the mode-off output of the old build; lines that carry timings are left out (the benchmark harness's filter).

  same-output.py --corp CORP --stub libstubcuda.so --old BUILD --new BUILD --programs a,b,... [--timeout 120]
"""
import argparse, fcntl, hashlib, os, re, shutil, signal, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools", "bench", "real"))
import real  # noqa: E402



def program_lines(out):
    """stdout without the lines that carry timings: the benchmark harness's filter (tools/bench/hostgpu/run.py program_text, real.py's NOISE)."""
    keep, prev_noise = [], False
    for l in out.decode("utf-8", "replace").splitlines():
        n = bool(real.NOISE.search(l)) or bool(re.search(r"\bs,\s.*%", l))
        if not n and not (prev_noise and re.fullmatch(r"\s*[0-9.eE+-]+\s*(s|ms|sec)?\s*", l)):
            keep.append(l)
        prev_noise = n
    return keep


def out_of(argv, cwd, env, timeout, lock):
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        p = subprocess.Popen(["nice", "-n", "19"] + argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            return None
        kept = program_lines(out)
        return p.returncode, hashlib.sha256("\n".join(kept).encode()).hexdigest()[:16], len(kept)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corp", required=True)
    ap.add_argument("--stub", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--programs", required=True)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    a = ap.parse_args()
    want = set(a.programs.split(","))
    bad = 0
    print("| program | lines compared | off: new = old | exact: new = old = off | turbo: new = old = off |")
    print("|---|---:|---|---|---|")
    for corpus, name, cwd, argv, copy_cwd in real.manifest(a.corp, ["polybench", "rodinia", "hecbench", "samples"]):
        if name not in want and corpus + "/" + name not in want:
            continue
        res = {}
        for build_name, build in (("old", a.old), ("new", a.new)):
            for form in ("off", "exact", "turbo"):
                env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_") and not k.startswith("STUB_")}
                env.update(VGPU_GPU="nvidia/a10g", LD_LIBRARY_PATH=os.path.join(build, "shim"), VGPU_THREADS="3", VGPU_QUIET="1")
                if form != "off":
                    env.update(VGPU_HOST_GPU=form, VGPU_HOST_GPU_LIBCUDA=a.stub, STUB_SM="86", STUB_MP="80")
                scratch, rcwd = None, cwd
                if copy_cwd:
                    scratch = tempfile.mkdtemp(prefix="same-")
                    dd = os.path.join(os.path.dirname(argv[0]), "data")
                    with open(os.path.join(scratch, "filelist"), "w") as f:
                        f.write("".join(os.path.join(dd, "cane4_%d.db\n" % j) for j in range(4)))
                    rcwd = scratch
                res[(build_name, form)] = out_of(argv, rcwd, env, a.timeout, a.lock)
                if scratch:
                    shutil.rmtree(scratch, ignore_errors=True)
        ref = res[("old", "off")]
        cell = lambda form: "timeout" if res[("new", form)] is None or res[("old", form)] is None or ref is None else \
            ("same" if res[("new", form)] == res[("old", form)] and (res[("new", form)] == ref) else "DIFFERENT")
        cells = [cell("off"), cell("exact"), cell("turbo")]
        bad += sum(1 for c in cells if c == "DIFFERENT")
        print("| %s/%s | %s | %s | %s | %s |" % (corpus, name, ref[2] if ref else "-", cells[0], cells[1], cells[2]), flush=True)
    print("\n%d different" % bad)


if __name__ == "__main__":
    main()
