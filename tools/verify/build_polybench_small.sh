#!/usr/bin/env bash
# Builds small-size copies of the PolyBench/GPU CUDA programs for tools/verify/run_corpora.sh.
#
#   tools/verify/build_polybench_small.sh <polybenchGpu checkout> <output dir>
#
# Why: in this checkout the MINI_DATASET block of every header holds the full-size numbers
# (2MM 2048^3, ATAX 4096^2, JACOBI1D 10000 steps): one 2MM run takes minutes even on all cores of
# the simulator, and the verification runs every program six times on ONE host thread. This copies
# the CUDA sources, caps every dimension in the MINI_DATASET block (cubic programs 256, 3DCONV 96,
# the rest 512) and the step counts (10), and builds them with the same flags as the existing
# binaries (sm_86, shared runtime, compat.h mapping cudaThreadSynchronize). DOITGEN is skipped: its
# source does not compile. The programs still compare the GPU result with a CPU reference.
set -uo pipefail
src="${1:?polybenchGpu checkout}"; out="${2:?output dir}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out/src" "$out/bin"; out="$(cd "$out" && pwd)"
cp -a "$src/CUDA/." "$out/src/"; cp -a "$src/common" "$out/common"
python3 -I - "$out/src" <<'PY'
import os, re, sys
root = sys.argv[1]
cubic = {"2MM", "3MM", "GEMM", "SYRK", "SYR2K", "CORR", "COVAR", "GRAMSCHM", "LU"}
for prog in sorted(os.listdir(root)):
    d = os.path.join(root, prog)
    if not os.path.isdir(d):
        continue
    cap = 256 if prog in cubic else 96 if prog == "3DCONV" else 512
    for fn in os.listdir(d):
        if not fn.endswith(".cuh"):
            continue
        p = os.path.join(d, fn)
        lines = open(p).read().split("\n")
        inblk = False
        for i, ln in enumerate(lines):
            if re.match(r"\s*#\s*ifdef\s+MINI_DATASET", ln):
                inblk = True
            elif inblk and re.match(r"\s*#\s*endif", ln):
                inblk = False
            elif inblk:
                m = re.match(r"(\s*#\s*define\s+)(\w+)(\s+)(\d+)(.*)$", ln)
                if m:
                    name, val = m.group(2), int(m.group(4))
                    new = min(val, 10) if name in ("TSTEPS", "TMAX") else min(val, cap)
                    lines[i] = f"{m.group(1)}{name}{m.group(3)}{new}{m.group(5)}"
        open(p, "w").write("\n".join(lines))
PY
for d in "$out"/src/*/; do
  p="$(basename "$d")"; [[ "$p" == DOITGEN ]] && continue
  cu="$(ls "$d"*.cu | head -1)"
  "$here/nvcc_locked.sh" -O3 -arch=sm_86 -w -Wno-deprecated-gpu-targets -DMINI_DATASET -include "$here/compat.h" \
      -I"$out/common" "$cu" -o "$out/bin/$p" 2>"$out/bin/$p.build.log" || echo "build failed: $p (see $out/bin/$p.build.log)"
done
ls "$out/bin" | grep -v '\.log$' | tr '\n' ' '; echo
