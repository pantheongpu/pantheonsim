#!/usr/bin/env python3
"""E1 add-on: native PTX-JIT with the driver's JIT cache WARM (CUDA_FORCE_PTX_JIT=1, cache enabled, one discarded fill run).
usage: e1_ptxwarm.py <bench binary> <out.jsonl> <runs>"""
import sys, json, re, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
binp, outp, nn = sys.argv[1], sys.argv[2], int(sys.argv[3])
WL = "int_alu f32_fma f64_fma f64_subnormal wmma mma_ptx mem shared atomics texture local divergent warp launch api startup".split()
env = env_for("native"); env["CUDA_FORCE_PTX_JIT"] = "1"; env["CUDA_CACHE_PATH"] = HOME + "/results/jitcache"; env.pop("CUDA_CACHE_DISABLE", None)
for w in WL:
    run([binp, w, "full"], env, timeout=600)
    for i in range(nn):
        r = run([binp, w, "full"], env, timeout=600)
        m = re.search(r"device_cpu_s=(\S+) device_wall_s=(\S+)", r["stdout"])
        open(outp, "a").write(json.dumps(dict(workload=w, side="native-ptx-warm", rep=i, wall=r["wall"], dev_wall=float(m[2]) if m else None,
                                              passed=r["stdout"].rstrip().endswith("PASS"))) + "\n")
    print(w, flush=True)
