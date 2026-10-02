#!/usr/bin/env bash
# SASS by default, PTX as the fallback (nvidia/docs/sass.md): which code a
# binary's kernels run, by what the binary carries and what the executor can
# run, and VGPU_SASS's overrides. Each run's answer is checked too.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-sass-path.$$"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -f "$out".*' EXIT

read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
src="$root/nvidia/tests/e2e/sass_path.cu"
# Both SASS and PTX for sm_86; SASS only; PTX only.
"$nvcc_bin" -std=c++17 -arch=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" "$src" -o "$out.both" &&
"$nvcc_bin" -std=c++17 -gencode arch=compute_86,code=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" "$src" -o "$out.sass" &&
"$nvcc_bin" -std=c++17 -gencode arch=compute_86,code=compute_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" "$src" -o "$out.ptx" ||
  { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out.both"; then exit 0; fi

fails=0
# run <name> <binary> <gpu> <expected log line (regex), or "none"> [env...]
run() {
  local name=$1 bin=$2 gpu=$3 want=$4
  shift 4
  local log
  log="$(env VGPU_QUIET=1 VGPU_SASS_LOG=1 VGPU_GPU="$gpu" LD_LIBRARY_PATH="$shim" "$@" "$bin" 2>&1)"
  local rc=$?
  echo "$log" | sed "s/^/    $name: /"
  if [[ $rc != 0 ]] || ! grep -q "^wrong: 0$" <<< "$log"; then
    echo "FAIL: $name: wrong answer or failed run"; fails=1
  fi
  if [[ $want == none ]]; then
    grep -q "running" <<< "$log" && { echo "FAIL: $name: expected no SASS decision"; fails=1; }
  elif ! grep -qE "$want" <<< "$log"; then
    echo "FAIL: $name: expected '$want'"; fails=1
  fi
}
run default      "$out.both" nvidia/rtx3060 "running SASS: an sm_86 cubin"
run sass-only    "$out.sass" nvidia/rtx3060 "running SASS: an sm_86 cubin"
run ptx-only     "$out.ptx"  nvidia/rtx3060 none
run a10-sm86     "$out.both" nvidia/a10     "running SASS: an sm_86 cubin"
run ptx-forced   "$out.both" nvidia/rtx3060 none VGPU_SASS=0
run fallback     "$out.both" nvidia/rtx3060 "running PTX instead of SASS: sm_86 .*SHFL" VGPU_SASS_REFUSE=SHFL
run sass-forced  "$out.both" nvidia/rtx3060 "running SASS: an sm_86 cubin" VGPU_SASS=1 VGPU_SASS_REFUSE=SHFL
# An sm_89 GPU runs sm_86 SASS (same major, newer minor); sm_90 does not,
# and falls back to the PTX.
run ada-sm86     "$out.both" nvidia/l4 "running SASS: an sm_86 cubin"
run hopper-ptx   "$out.both" nvidia/h100    none
# SASS for the wrong architecture and no PTX: the launch is refused.
log="$(VGPU_QUIET=1 VGPU_GPU=nvidia/h100 LD_LIBRARY_PATH="$shim" "$out.sass" 2>&1)"
echo "$log" | sed "s/^/    sass-on-h100: /"
grep -qE "launch: cudaErrorNoKernelImageForDevice" <<< "$log" ||
  { echo "FAIL: SASS for another architecture was not refused"; fails=1; }
# A performance-monitor counter is refused by name whichever code runs: the
# fatbin falls back from SASS that reads SR_PM0 to PTX that names %pm0, and
# SASS alone refuses it at the instruction.
pm="$root/nvidia/tests/e2e/pm_counter.cu"
if "$nvcc_bin" -arch=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" "$pm" -o "$out.pm-both" &&
   "$nvcc_bin" -gencode arch=compute_86,code=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" \
     "$pm" -o "$out.pm-sass"; then
  for v in both sass; do
    log="$(VGPU_SASS_LOG=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$out.pm-$v" 2>&1)"
    rc=$?
    echo "$log" | sed "s/^/    pm-counter-$v: /"
    if [[ $rc == 0 ]] || ! grep -q "performance-monitor counter" <<< "$log"; then
      echo "FAIL: pm-counter-$v: %pm0 was not refused by name"; fails=1
    fi
  done
else
  echo "FAIL: pm_counter.cu does not compile"; fails=1
fi
[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: sass path"
exit $fails
