#!/usr/bin/env python3
"""E3 on the 16 harness workloads, which carry a bit-exact host reference (PASS means every checked value matched bit for bit).
For each build (bench.cu compiled for sm_80/86/89/90/120, SASS + PTX of that architecture) and each simulated profile, run:
  sim   on the profile (engine: SASS if the build has the profile's architecture, else PTX of the build, as VGPU_SASS_LOG says)
  native on the A10G, SASS as built, and with CUDA_FORCE_PTX_JIT=1 (driver JITs the build's PTX for sm_86)
usage: e3_harness.py <variantdir> <out.jsonl>"""
import sys, json, re, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
vd, outp = sys.argv[1], sys.argv[2]
WL = "int_alu f32_fma f64_fma f64_subnormal wmma mma_ptx mem shared atomics texture local divergent warp launch api startup".split()
# (build arch, simulated profile) pairs: the profile whose architecture the build targets, plus the sm_86 PTX build on every profile
PAIRS = [(80, "nvidia/a100"), (86, "nvidia/a10g"), (89, "nvidia/l4"), (89, "nvidia/l40s"), (90, "nvidia/h100"), (120, "nvidia/rtx5090"),
         (86, "nvidia/l4"), (86, "nvidia/l40s"), (86, "nvidia/h100"), (86, "nvidia/rtx5090"), (86, "nvidia/a100"), (86, "nvidia/t4")]

def go(arch, side, prof, w):
    b = "%s/bench_sm%d" % (vd, arch)
    mode = {"sim": "sim", "native-sass": "native", "native-ptx": "ptxjit"}[side]
    extra = {"VGPU_SASS_LOG": "1"} if side == "sim" else {}
    r = run([b, w, "quick"], env_for(mode, profile=prof or "nvidia/a10g", extra=extra, threads=4), timeout=300)
    passed = r["stdout"].rstrip().endswith("PASS")
    eng = ["SASS" if "running SASS" in r["stderr"] else "PTX"] if side == "sim" else []
    tail = "" if passed else (r["stderr"] + r["stdout"]).strip().splitlines()[-1:] 
    rec = dict(build="sm_%d" % arch, side=side, profile=prof, workload=w, rc=r["rc"], timeout=r["timeout"], passed=passed, engine_log=eng,
               tail=tail, wall=round(r["wall"], 3))
    open(outp, "a").write(json.dumps(rec) + "\n")

done_native = set()
for arch, prof in PAIRS:
    for w in WL:
        for side in ("native-sass", "native-ptx"):
            if (arch, side, w) not in done_native:
                done_native.add((arch, side, w)); go(arch, side, None, w)
        go(arch, "sim", prof, w)
    print("done", arch, prof, flush=True)
