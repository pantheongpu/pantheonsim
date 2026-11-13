#!/usr/bin/env python3
"""Round sp65: the runtime's own decisions with no card, against the stand-in driver (nvidia/tests/e2e/host_gpu_stub_driver.c) built as a card that
keeps memory and computes nothing (STUB_CARD_OK=1 STUB_KEEP_MEMORY=1). A program's output then changes wherever a launch went to the "card", so only
the decisions are compared, not the outputs: per program and form, the launches on the card, the ones kept for their size, and what host_gpu_streak
counts. Forms: the mode with the rule of Round sp65 and with VGPU_HOST_GPU_STREAK=0. Each run is one hold of the machine's lock, at nice 19.

  stub_check.py --build B --corp C --stub libstubcuda.so --programs a,b [--modes exact,turbo] [--timeout 270] [--json out.json]
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../tools/bench/real"))
from real import manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True); ap.add_argument("--corp", required=True); ap.add_argument("--stub", required=True)
ap.add_argument("--programs", required=True); ap.add_argument("--modes", default="exact,turbo"); ap.add_argument("--timeout", type=int, default=270)
ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock"); ap.add_argument("--json", default="")
a = ap.parse_args()
want = set(a.programs.split(","))
res = {}
for corpus, name, cwd, argv, copy_cwd in manifest(os.path.abspath(a.corp), ["polybench", "rodinia", "hecbench"]):
    if name not in want and "%s/%s" % (corpus, name) not in want: continue
    for mode in a.modes.split(","):
        for streak in ("1", "0"):
            env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_") and not k.startswith("STUB_")}
            env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/a10g", VGPU_THREADS="4", LD_LIBRARY_PATH=os.path.join(os.path.abspath(a.build), "shim"),
                       VGPU_HOST_GPU=mode, VGPU_HOST_GPU_LOG="1", VGPU_HOST_GPU_STREAK=streak, VGPU_HOST_GPU_LIBCUDA=os.path.abspath(a.stub),
                       STUB_CARD_OK="1", STUB_KEEP_MEMORY="1", STUB_SM="86", STUB_MP="80")
            scratch = None
            if copy_cwd:
                scratch = tempfile.mkdtemp(prefix="sp65-stub-", dir=os.path.expanduser("~/.cache/sp65"))
                dd = os.path.join(os.path.dirname(argv[0]), "data")
                with open(os.path.join(scratch, "filelist"), "w") as f:
                    f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
                cwd = scratch
            p = subprocess.run(["flock", a.lock, "nice", "-n", "19", "timeout", str(a.timeout)] + argv, cwd=cwd, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
            if scratch: shutil.rmtree(scratch, ignore_errors=True)
            err = p.stderr
            rec = {"rc": p.returncode}
            m = re.search(r"host_gpu: (\d+) launches ran on the card, (\d+) in the simulator \((\d+) kernels not [^,]*, (\d+) too small", err)
            if m: rec.update(card=int(m.group(1)), sim=int(m.group(2)), refused=int(m.group(3)), too_small=int(m.group(4)))
            m = re.search(r"host_gpu_streak: sent=(\d+) card_only_mib=([0-9.]+) rule=([01])", err)
            if m: rec.update(streak_sent=int(m.group(1)), streak_mib=float(m.group(2)), rule=int(m.group(3)))
            m = re.search(r"host_gpu: ([0-9.]+) MiB went to the card, ([0-9.]+) MiB came back", err)
            if m: rec.update(pushed_mib=float(m.group(1)), pulled_mib=float(m.group(2)))
            rec["banned"] = len(re.findall(r"offloading it ends", err))
            rec["failed_lines"] = len(re.findall(r"failed on the card", err))
            key = "%s/%s" % (corpus, name)
            res.setdefault(key, {})["%s%s" % (mode, "" if streak == "1" else "n")] = rec
            print(key, mode, "streak" if streak == "1" else "STREAK=0", rec, flush=True)
            time.sleep(2)
if a.json:
    json.dump(res, open(a.json, "w"), indent=0, sort_keys=True)
