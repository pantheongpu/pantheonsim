#!/usr/bin/env python3
"""r11 data: one logged FIT=0 run of each program with the rule off (FIRST_BIG=0); keeps the nosample/sample/kept lines.  usage: r11-logs.py out corpora [timeout] [first_big]"""
import os, subprocess, sys, time
sys.path.insert(0, os.path.expanduser("~/a656/tools/bench/real"))
import real
out, corpora = sys.argv[1], sys.argv[2].split(",")
tmo = float(sys.argv[3]) if len(sys.argv) > 3 else 120
fb = sys.argv[4] if len(sys.argv) > 4 else "0"
build = os.path.expanduser("~/a656-build")
os.makedirs(out, exist_ok=True)
base = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
base.update(VGPU_GPU="nvidia/a10g", LD_LIBRARY_PATH=build + "/shim", VGPU_HOST_GPU="turbo", VGPU_HOST_GPU_FIT="0",
            VGPU_HOST_GPU_LOG="1", VGPU_HOST_GPU_FIRST_BIG=fb)
for corpus, name, cwd, argv, copy in real.manifest(os.path.expanduser("~/corp"), corpora):
    import tempfile, shutil
    wd = cwd
    if copy:
        wd = tempfile.mkdtemp(prefix="r11-"); [shutil.copy(os.path.join(cwd, f), wd) for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]
    t0 = time.time()
    try:
        p = subprocess.run(argv, cwd=wd, env=base, capture_output=True, text=True, timeout=tmo)
        err = p.stderr
    except subprocess.TimeoutExpired as e:
        err = (e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr or "") + "\nTIMEOUT\n"
    with open(f"{out}/{corpus}-{name}.txt", "w") as f:
        f.write("".join(l + "\n" for l in err.splitlines() if l.startswith("host_gpu_nosample") or l.startswith("host_gpu_sample") or l.startswith("host_gpu_kept") or l.startswith("host_gpu_turbo") or l == "TIMEOUT"))
    print(f"{corpus}/{name} {time.time()-t0:.1f}s", flush=True)
    if copy: shutil.rmtree(wd, ignore_errors=True)
