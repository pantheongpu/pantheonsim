#!/usr/bin/env bash
# The real-card check of the host-GPU hand-back fix (include/vgpu/host_gpu_track.hpp, MemoryManager::free): HeCBench hadamard-cuda aborted with "illegal memory access"
# in most runs of VGPU_HOST_GPU=exact before it, and the programs that use the card must still match the card's own results after it. Run it on a machine with an
# NVIDIA card whose profile exists (an A10G: nvidia/a10g), with nvcc 13.0 on PATH and the simulator built there (`ninja vgpucudart`):
#
#   tools/card-ci/handback-check.sh [build-dir]
#   env: HANDBACK_RUNS      (default 20)  runs of hadamard-cuda per host-thread count
#        HANDBACK_THREADS   (default "1 2 3 4")
#        HANDBACK_GPU       (default nvidia/a10g) the simulated profile; its compute capability must be the card's
#        HANDBACK_CONTROL   a build dir of the simulator WITHOUT the fix (origin/main before it): the same loop runs on it, to show the failure reproduces here
#                           (HANDBACK_EXPECT_CONTROL_FAIL=1: the script fails if the control never fails)
#        HANDBACK_SUBSET    (default 1)   also run the 33 programs that launched on the card in the 2026-11-02 A10G trial under VGPU_HOST_GPU=exact, against the card's
#                           own results in ci/external/*.txt (suites.sh run); 0 skips it
#        HANDBACK_WORK      (default a temp dir, removed at exit) where the suites are fetched and built
#
# Pass: every one of the runs of hadamard-cuda exits 0, prints 15 PASS (what the card prints, ci/external/hecbench.txt) and no "CUDA error", on every thread count; and
# (with the subset) every program of the subset matches its list. Expected before the fix: about 8 failures in 10 runs (1, 3 and 4 threads). Exit 0 on pass.
# The programs, their sources and the expected results are the ones the suites already pin (ci/external/suites.sh); nothing is added to them.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${1:-$root/build}"
runs="${HANDBACK_RUNS:-20}"
threads="${HANDBACK_THREADS:-1 2 3 4}"
gpu="${HANDBACK_GPU:-nvidia/a10g}"
fails=0
say() { echo "== $*"; }
command -v nvidia-smi > /dev/null && nvidia-smi -L || { echo "FAIL: no card on this machine"; exit 1; }
command -v nvcc > /dev/null || { echo "FAIL: no nvcc on PATH (CUDA 13.0 is what the lists were captured with)"; exit 1; }
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
[[ "$cc" == 86 ]] || { echo "FAIL: this check builds the suites for sm_86 (ci/external/suites.sh), the card is sm_$cc"; exit 1; }
shim="$build/shim"
ls "$shim"/libcudart.so.[0-9]* > /dev/null 2>&1 || { echo "FAIL: no simulator shim in $shim (ninja -C $build vgpucudart)"; exit 1; }
work="${HANDBACK_WORK:-$(mktemp -d "${TMPDIR:-/tmp}/handback-check.XXXXXX")}"
[[ -n "${HANDBACK_WORK:-}" ]] || trap 'rm -rf "$work"' EXIT
mkdir -p "$work"
suites="$root/ci/external/suites.sh"

say "hadamard-cuda: fetch and build (sm_86, -cudart shared)"
SUITE_ONLY='^hadamard-cuda$' "$suites" fetch hecbench "$work/hecbench" || { echo "FAIL: fetch"; exit 1; }
(cd "$work/hecbench/src/hadamard-cuda" && make -s CC="nvcc -cudart shared" ARCH=sm_86 EXTRA_CFLAGS="-w" > build.log 2>&1) || { tail -5 "$work/hecbench/src/hadamard-cuda/build.log"; echo "FAIL: build"; exit 1; }
prog="$work/hecbench/src/hadamard-cuda/main"

# loop <label> <shim-dir>: prints "label threads=T pass=P of N", and counts the failures of the runs (the first failure's last lines are kept).
loop() {
  local label=$1 sh=$2 t i p bad=0
  for t in $threads; do
    p=0
    for ((i = 1; i <= runs; i++)); do
      out="$work/$label.t$t.$i.out"
      (cd "$(dirname "$prog")" && env VGPU_GPU="$gpu" LD_LIBRARY_PATH="$sh" VGPU_THREADS=$t VGPU_HOST_GPU=exact VGPU_HOST_GPU_LOG=1 timeout -k 10 600 "$prog" 64 1 > "$out" 2> "$out.err" < /dev/null)
      rc=$?
      if [[ $rc == 0 && "$(grep -c PASS "$out")" == 15 ]] && ! grep -q "CUDA error" "$out"; then p=$((p + 1))
      else bad=$((bad + 1)); [[ -e "$work/$label.first-failure" ]] || { { echo "rc=$rc threads=$t run=$i"; tail -4 "$out"; tail -3 "$out.err"; } > "$work/$label.first-failure"; } ; fi
    done
    echo "$label threads=$t pass=$p of $runs"
  done
  return $bad
}

say "hadamard-cuda with the fix ($build)"
loop fixed "$shim"
bad=$?
if (( bad )); then echo "FAIL: $bad runs failed with the fix; the first:"; cat "$work/fixed.first-failure"; fails=1; else echo "PASS: no run failed"; fi

if [[ -n "${HANDBACK_CONTROL:-}" ]]; then
  say "hadamard-cuda without the fix ($HANDBACK_CONTROL): the failure must reproduce here"
  loop control "$HANDBACK_CONTROL/shim"
  cbad=$?
  echo "control: $cbad runs failed"
  [[ -e "$work/control.first-failure" ]] && { echo "the first:"; cat "$work/control.first-failure"; }
  if (( cbad == 0 )) && [[ -n "${HANDBACK_EXPECT_CONTROL_FAIL:-}" ]]; then echo "FAIL: the control never failed: this machine does not reproduce the bug, so the fix is not shown"; fails=1; fi
fi

if [[ "${HANDBACK_SUBSET:-1}" != 0 ]]; then
  # The programs that launched on the card in the A10G trial (33), by suite.
  only_poly='^(2DCONV|3MM|GEMM|JACOBI2D|LU|SYR2K|SYRK)$'
  only_rod='^(pathfinder|needle|lud|streamcluster)$'
  only_hec='^(axpby-cuda|blockAccess-cuda|boxfilter-cuda|chemv-cuda|compute-score-cuda|concurrentKernels-cuda|epistasis-cuda|f8cast-cuda|fdtd3d-cuda|floydwarshall-cuda|fwt-cuda|ga-cuda|gaussian-cuda|hadamard-cuda|is-cuda|ising-cuda|iso2dfd-cuda|keccaktreehash-cuda|laplace3d-cuda|marchingCubes-cuda|matrixT-cuda|mergeVS-cuda)$'
  for pair in "polybench:$only_poly" "rodinia:$only_rod" "hecbench:$only_hec"; do
    s=${pair%%:*}; o=${pair#*:}
    say "subset: $s under VGPU_HOST_GPU=exact"
    d="$work/subset-$s"
    SUITE_ONLY="$o" "$suites" fetch "$s" "$d" > /dev/null || { echo "FAIL: fetch $s"; fails=1; continue; }
    SUITE_ONLY="$o" "$suites" build "$s" "$d" || true
    # (the programs are compared with the card's results in the lists, as the suites job does: a launch that went to the card must not change a result)
    VGPU_HOST_GPU=exact SUITE_GPU="$gpu" SUITE_TIMEOUT=900 SUITE_ONLY="$o" "$suites" run "$s" "$d" "$shim" "$work/report-$s" || { echo "FAIL: $s: a program differs from its list"; fails=1; }
    cat "$work/report-$s/results.tsv" 2> /dev/null | awk -F'\t' '{print "  " $1 ": " $2}'
  done
fi

if (( fails )); then echo "FAIL: the hand-back check"; exit 1; fi
echo "PASS: the hand-back check"
