#!/usr/bin/env bash
# Build and run the pantheon GPU stress/diagnostics workloads on VirtualGPU.
#
# The workload sources are UNMODIFIED; only two things differ from a physical
# GPU run:
#   1. the CUDA runtime is linked shared (-cudart shared) so VirtualGPU's
#      libcudart.so.13 can be substituted at load time, and
#   2. the workloads' own CLI knobs are set to a CPU-appropriate intensity.
#
# (2) matters because these are saturation tests: a GPU retires ~10^13 ops/s,
# a CPU interpreter ~10^8. Reducing --kernel_loops/--grid_size runs the same
# code over a smaller working set; it does not change what is executed.
#
# Usage: run-pantheon-workloads.sh [pantheon-repo] [outdir]
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="${1:-$here/../pantheongpu}"
out="${2:-${TMPDIR:-/tmp}/vgpu-pantheon}"
shim="$here/build/shim"
bin="$out/bin"; logs="$out/logs"
mkdir -p "$bin" "$logs"

: "${VGPU_WL_GPU:=nvidia/a10}"       # sm_86 profile matches the sm_86 build
: "${VGPU_WL_VRAM_MB:=512}"          # advertised VRAM (sizes %-of-VRAM tests)
: "${VGPU_WL_DURATION:=5}"           # seconds each workload should run
: "${VGPU_WL_TIMEOUT:=180}"          # hard cap per workload
: "${VGPU_WL_LOOPS:=4}"              # --kernel_loops
: "${VGPU_WL_GRID:=8}"               # --grid_size
: "${VGPU_WL_MEMPCT:=5}"             # percent of virtual VRAM to allocate

if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (needed to compile the workloads)"; exit 0
fi
[[ -e "$shim/libcudart.so.13" ]] || { echo "error: build VirtualGPU first (./scripts/build.sh)" >&2; exit 1; }

names=$(ls "$repo/kernels" | grep -v '^common$')
echo "Building $(wc -w <<<"$names") workloads (unmodified sources)..."
for w in $names; do
  ( cd "$repo" && nvcc -O3 -std=c++14 -DNDEBUG -Ikernels/common -x cu \
      --gpu-architecture=sm_86 -Wno-deprecated-gpu-targets \
      -I kernels/common/optix -I kernels/common/nvenc -cudart shared \
      "kernels/$w/$w.cpp" -o "$bin/$w" -L"$shim" -lcuda -ldl ) >"$logs/build_$w.log" 2>&1 &
  while [[ $(jobs -r | wc -l) -ge $(nproc) ]]; do wait -n; done
done
wait

pass=0; fail=0; skip=0
printf '\n%-26s %-8s %s\n' WORKLOAD RESULT DETAIL
printf '%-26s %-8s %s\n' "--------------------------" "------" "------"
for w in $names; do
  if [[ ! -x "$bin/$w" ]]; then
    printf '%-26s %-8s %s\n' "$w" SKIP "does not build in this environment"; ((skip++)); continue
  fi
  start=$SECONDS
  VGPU_QUIET=1 VGPU_GPU="$VGPU_WL_GPU" VGPU_VRAM_MB="$VGPU_WL_VRAM_MB" \
    LD_LIBRARY_PATH="$shim" timeout "$VGPU_WL_TIMEOUT" \
    "$bin/$w" 0 "$VGPU_WL_DURATION" "$VGPU_WL_MEMPCT" \
    --kernel_loops "$VGPU_WL_LOOPS" --warmup_iters 1 --grid_size "$VGPU_WL_GRID" \
    >"$logs/$w.log" 2>&1
  rc=$?; el=$((SECONDS - start))
  detail=$(grep -iE 'verification|throughput' "$logs/$w.log" | head -1 | cut -c1-52)
  if [[ $rc -eq 0 ]]; then
    printf '%-26s %-8s %ss  %s\n' "$w" PASS "$el" "$detail"; ((pass++))
  else
    why=$(grep -iE 'unsupported PTX|error \[|PANTHEON ERROR' "$logs/$w.log" | head -1 | cut -c1-52)
    printf '%-26s %-8s %ss  rc=%s %s\n' "$w" FAIL "$el" "$rc" "$why"; ((fail++))
  fi
done
printf '\n%d passed, %d failed, %d skipped. Logs in %s\n' "$pass" "$fail" "$skip" "$logs"
[[ $fail -eq 0 ]]
