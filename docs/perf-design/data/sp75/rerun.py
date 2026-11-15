#!/usr/bin/env python3
"""Round sp75: wall time of programs in several forms, interleaved, against the stand-in driver (turbo mode, no log), each run under the
machine's build lock. A form is NAME=BUILD[,ENV=VALUE...]. Prints a markdown table: the median of --runs runs a form, and each form against the first.

  rerun.py --corp CORP --stub libstubcuda.so --programs a,b --runs 7 --form old=/path/build-old --form new=/path/build --form knob=/path/build,VGPU_HOST_GPU_SAMPLE_BLOCK=0
"""
import argparse, fcntl, os, shutil, signal, statistics, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools", "bench", "real"))
import real  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corp", required=True)
    ap.add_argument("--stub", required=True)
    ap.add_argument("--programs", required=True)
    ap.add_argument("--form", action="append", required=True)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    a = ap.parse_args()
    forms = []
    for f in a.form:
        name, _, rest = f.partition("=")
        parts = rest.split(",")
        forms.append((name, parts[0], dict(kv.split("=", 1) for kv in parts[1:])))
    want = set(a.programs.split(","))
    fd = os.open(a.lock, os.O_RDWR | os.O_CREAT, 0o666)
    print("| program | " + " | ".join("%s s" % n for n, _, _ in forms) + " | " + " | ".join("%s / %s" % (n, forms[0][0]) for n, _, _ in forms[1:]) + " | spread of %s |" % forms[0][0])
    print("|---|" + "---:|" * (2 * len(forms)))
    for corpus, name, cwd, argv, copy_cwd in real.manifest(a.corp, ["polybench", "rodinia", "hecbench", "samples"]):
        if name not in want and corpus + "/" + name not in want:
            continue
        T = {n: [] for n, _, _ in forms}
        for k in range(a.runs):
            for fname, build, extra in (forms if k % 2 == 0 else forms[::-1]):
                env = {x: y for x, y in os.environ.items() if not x.startswith("VGPU_") and not x.startswith("STUB_")}
                env.update(VGPU_GPU="nvidia/a10g", LD_LIBRARY_PATH=os.path.join(build, "shim"), VGPU_THREADS="3", VGPU_QUIET="1", VGPU_HOST_GPU="turbo",
                           VGPU_HOST_GPU_LIBCUDA=a.stub, STUB_SM="86", STUB_MP="80")
                env.update(extra)
                scratch, rcwd = None, cwd
                if copy_cwd:
                    scratch = tempfile.mkdtemp(prefix="rerun-")
                    dd = os.path.join(os.path.dirname(argv[0]), "data")
                    with open(os.path.join(scratch, "filelist"), "w") as f:
                        f.write("".join(os.path.join(dd, "cane4_%d.db\n" % j) for j in range(4)))
                    rcwd = scratch
                fcntl.flock(fd, fcntl.LOCK_EX)
                t0 = time.monotonic()
                p = subprocess.Popen(["nice", "-n", "19"] + argv, cwd=rcwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                # (a wait with a time limit polls every 50 ms, which is the difference looked for: the limit is a timer's, the wait blocks)
                killer = threading.Timer(a.timeout, lambda: os.killpg(p.pid, signal.SIGKILL))
                killer.start()
                p.wait()
                T[fname].append(time.monotonic() - t0)
                killer.cancel()
                fcntl.flock(fd, fcntl.LOCK_UN)
                if scratch:
                    shutil.rmtree(scratch, ignore_errors=True)
        med = {n: statistics.median(v) for n, v in T.items()}
        base = forms[0][0]
        print("| %s/%s | " % (corpus, name) + " | ".join("%.3f" % med[n] for n, _, _ in forms) + " | " +
              " | ".join("%.3f" % (med[n] / med[base]) for n, _, _ in forms[1:]) + " | %.3f to %.3f |" % (min(T[base]), max(T[base])), flush=True)


if __name__ == "__main__":
    main()
