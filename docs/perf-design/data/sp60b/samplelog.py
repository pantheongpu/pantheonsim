#!/usr/bin/env python3
"""Round sp60b: runs the corpus programs of tools/bench/real/real.py in a few forms of the host-GPU mode and keeps, per program and form, the wall
time, a hash of the output, the mode's machine-readable lines and every host_gpu_sample line. Needs no card: with VGPU_HOST_GPU_SAMPLE=log the mode
samples and decides but sends nothing; with a stand-in driver (--stub) a sampled launch is sent, and the stand-in either refuses it (the simulator
then runs it) or "runs" it by computing nothing (the program then reads the memory as it was before the launch).

  samplelog.py --src <tree> --build <build> --corp <corp> --out out.json [--corpora polybench,rodinia,hecbench] [--forms off,log,...]
               [--programs a,b] [--timeout 120] [--threads 4] [--stub /path/libstub.so] [--lock /tmp/pantheonsim-builder.lock]

Runs are made under the machine's build lock, which is given up between two runs once it has been held --hold seconds; niced.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

STUB = {"VGPU_HOST_GPU_LOG": "1", "VGPU_HOST_GPU_START_COST": "0", "VGPU_HOST_GPU_NO_FIRST_LAUNCH": "1"}
FORMS = {
    "off": {},
    "exact0": {"VGPU_HOST_GPU": "exact", "VGPU_HOST_GPU_SAMPLE": "0", "VGPU_HOST_GPU_LOG": "1"},
    "exact": {"VGPU_HOST_GPU": "exact", "VGPU_HOST_GPU_LOG": "1"},
    "turbo": {"VGPU_HOST_GPU": "turbo", "VGPU_HOST_GPU_LOG": "1"},
    "log": {"VGPU_HOST_GPU": "exact", "VGPU_HOST_GPU_SAMPLE": "log", "VGPU_HOST_GPU_LOG": "1"},
    "tlog": {"VGPU_HOST_GPU": "turbo", "VGPU_HOST_GPU_SAMPLE": "log", "VGPU_HOST_GPU_LOG": "1"},
    # the stand-in driver refuses the module: a sampled launch is sent, refused, and run by the simulator
    "stub": dict(STUB, VGPU_HOST_GPU="exact"),
    "tstub": dict(STUB, VGPU_HOST_GPU="turbo"),
    # a card that holds memory and computes nothing: a sampled launch that is sent leaves the memory as it was before the launch
    "stubnop": dict(STUB, VGPU_HOST_GPU="exact", STUB_CARD_OK="1", STUB_KEEP_MEMORY="1"),
}


class Lock:
    """The machine's build lock, held for a program's forms together and given up after --hold seconds so that others get their turn."""

    def __init__(self, path, hold):
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666) if path else None
        self.hold = hold
        self.since = None

    def take(self):
        if self.fd is None:
            return
        if self.since is not None and time.monotonic() - self.since > self.hold:
            self.drop()
        if self.since is None:
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            self.since = time.monotonic()

    def drop(self):
        if self.fd is not None and self.since is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.since = None
            time.sleep(1)


def run(real, build, cwd, argv, copy_cwd, form_env, threads, timeout, gpu, stub, mp):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_") and not k.startswith("STUB_")}
    env.update(VGPU_QUIET="1", VGPU_GPU=gpu, LD_LIBRARY_PATH=os.path.join(build, "shim"), VGPU_THREADS=str(threads))
    if stub and form_env.get("VGPU_HOST_GPU_START_COST") == "0":
        env.update(VGPU_HOST_GPU_LIBCUDA=stub, STUB_SM="86", STUB_MP=str(mp), STUB_INIT_MS="150")
    env.update(form_env)
    scratch = None
    if copy_cwd:
        scratch = tempfile.mkdtemp(prefix="sp60b-")
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        with open(os.path.join(scratch, "filelist"), "w") as f:
            f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    pre = ["nice", "-n", "19"]
    errf = tempfile.TemporaryFile()
    t0 = time.monotonic()
    p = subprocess.Popen(pre + argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=errf, start_new_session=True)
    to = False
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        to = True
    wall = time.monotonic() - t0
    errf.seek(0)
    err = errf.read().decode("utf-8", "replace")
    extra = b""
    if scratch:
        for f in sorted(x for x in os.listdir(scratch) if x != "filelist"):
            fp = os.path.join(scratch, f)
            if os.path.isfile(fp) and f != "output.dat":
                extra += f.encode() + open(fp, "rb").read()
        shutil.rmtree(scratch, ignore_errors=True)
    h, _ = real.normalise(out.decode("utf-8", "replace"), extra)
    text = out.decode("utf-8", "replace")
    rec = {"wall_s": round(wall, 4), "rc": p.returncode, "timeout": to, "out": h, "launches": [],
           "says": [l[:160] for l in text.splitlines() if re.search(r"Non-Matching|PASS|FAIL|[Cc]hecksum|[Ee]rror|result", l)][:6]}
    for line in err.splitlines():
        if line.startswith("host_gpu_sample:"):
            d = {}
            for kv in line.split()[1:]:
                k, v = kv.split("=", 1)
                d[k] = v if k == "kernel" else float(v) if "." in v else int(v)
            rec["launches"].append(d)

        elif re.match(r"host_gpu_(turbo|refusals|sampled):", line):
            rec[line.split(":")[0]] = dict(kv.split("=", 1) for kv in line.split()[1:])
        elif "mode off" in line or "sent by the time" in line or "did not load the module" in line:
            rec.setdefault("notes", []).append(line[:300])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--corp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpora", default="polybench,rodinia,hecbench")
    ap.add_argument("--forms", default="off,log")
    ap.add_argument("--programs", default="")
    ap.add_argument("--skip", default="fpdc-cuda,zerocopy-cuda")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu", default="nvidia/a10g")
    ap.add_argument("--mp", type=int, default=80, help="the multiprocessors of --gpu, for the stand-in driver")
    ap.add_argument("--stub", default="")
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    ap.add_argument("--hold", type=float, default=150, help="seconds after which the lock is given up before the next run")
    a = ap.parse_args()
    lock = Lock(a.lock, a.hold)
    sys.path.insert(0, os.path.join(a.src, "tools/bench/real"))
    import real
    only = set(x for x in a.programs.split(",") if x)
    skip = set(x for x in a.skip.split(",") if x)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for corpus, name, cwd, argv, copy_cwd in real.manifest(os.path.abspath(a.corp), a.corpora.split(",")):
        if (only and name not in only) or name in skip:
            continue
        key = corpus + "/" + name
        for form in a.forms.split(","):
            lock.take()
            rec = run(real, os.path.abspath(a.build), cwd, argv, copy_cwd, FORMS[form], a.threads, a.timeout, a.gpu, a.stub, a.mp)
            res.setdefault(key, {})[form] = rec
            w = rec.get("host_gpu_sampled", {})
            print("%-34s %-7s %8.2fs rc=%s%s out=%s sampled=%s sent=%s says=%s" % (key, form, rec["wall_s"], rec["rc"], " TIMEOUT" if rec["timeout"] else "", rec["out"],
                                                                             w.get("sampled", "-"), w.get("sent", "-"), "|".join(rec["says"])[-40:]), flush=True)
            json.dump(res, open(a.out, "w"), separators=(",", ":"))
    lock.drop()


if __name__ == "__main__":
    main()
