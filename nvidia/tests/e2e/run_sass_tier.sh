#!/usr/bin/env bash
# The SASS executor's pre-decoded tier (src/sass/tier.hpp, VGPU_TIER=1) against the interpreter (VGPU_TIER=0).
#
# sass_tier.cu (random inputs that break arithmetic through every form the tier has a handler for) is built for each
# architecture below, and so is a set of the other SASS test programs (the ones sass_archs.sh runs on SASS, which
# exercise barriers, divergence, warp ops, atomics, textures, MMA and more). Each runs twice on its generation's
# simulated GPU, with one host thread and the performance counters on: with the interpreter, and with the tier. The two
# must pass and print byte for byte the same -- every result and every counter of every launch, and the text of any
# error -- apart from the tier's own statistics lines (VGPU_TIER_STATS=1, lines starting "tier").
#
# The comparison cannot pass by the tier doing nothing: the statistics must show that the tier ran at least
# VGPU_TIER_MIN_PERCENT (default 30) percent of sass_tier's warp instructions.
#
# VGPU_TIER_ARCHS narrows the architectures (default sm_75 sm_80 sm_86 sm_90), VGPU_TIER_PROGRAMS the other programs
# (name:first architecture), VGPU_TIER_JOBS caps the builds run at once.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-sass-tier.XXXXXX")"
trap 'rm -rf "$work"' EXIT

declare -A gpu=([sm_75]=nvidia/t4 [sm_80]=nvidia/a100 [sm_86]=nvidia/rtx3060 [sm_89]=nvidia/l4 [sm_90]=nvidia/h100)
archs=(${VGPU_TIER_ARCHS:-sm_75 sm_80 sm_86 sm_90})
# program:first architecture; run on sm_86 only except sass_tier (every architecture).
others=(${VGPU_TIER_PROGRAMS:-sass_fastpath:75 sass_archs:75 vector_add:75 device_functions:75 device_intrinsics:75 bsync_recursive_abi:75
        waterfall:90 named_barriers:75 alloca_stack:75 shared_atomics64:75 large_params:75 sync_masks:86 mma_fragment_layout:80
        modern_dtypes:80 wmma_gemm:80 wmma_types:80})
min_percent="${VGPU_TIER_MIN_PERCENT:-30}"

supported="$("$nvcc_bin" --list-gpu-code 2>/dev/null)"
read -r rmaj rmin <<< "$("$nvcc_bin" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
# sync_masks' expected values are CUDA 13's ptxas code; the others need nothing special.
jobs=()
for arch in "${archs[@]}"; do
  if ! grep -qx "$arch" <<< "$supported"; then echo "SKIP $arch: this nvcc cannot target it"; continue; fi
  jobs+=("sass_tier $arch")
done
for p in "${others[@]}"; do
  IFS=: read -r name first <<< "$p"
  [[ $name == sync_masks && ${rmaj:-0} -lt 13 ]] && continue
  arch=sm_86
  (( ${arch#sm_} >= first )) || continue
  grep -qx "$arch" <<< "$supported" && jobs+=("$name $arch")
done
(( ${#jobs[@]} )) || { echo "SKIP: no architecture this nvcc can target"; exit 0; }

read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
export nvcc_bin root work
export san="${san_flags[*]}"
printf '%s\n' "${jobs[@]}" | xargs -P "${VGPU_TIER_JOBS:-$("$root/ci/jobs.sh" -p sass)}" -L 1 bash -c '
  "$nvcc_bin" -std=c++17 -arch="$1" -cudart shared -w -Wno-deprecated-gpu-targets -Xlinker --as-needed $san \
    -I"$root/nvidia/tests/e2e" "$root/nvidia/tests/e2e/$0.cu" -o "$work/$0.$1" -lcuda 2> "$work/$0.$1.build" ||
    echo "build failed" >> "$work/$0.$1.build"'

fails=0
check() {
  local p=$1 arch=$2 bin="$work/$1.$2" a b ra rb why=""
  if [[ ! -x $bin ]]; then
    echo "FAIL $arch $p: does not build"; sed 's/^/    /' "$bin.build" | tail -5; fails=1; return
  fi
  require_shim_libs "$shim" "$bin" >/dev/null || return
  local env=(VGPU_QUIET=1 VGPU_GPU=${gpu[$arch]} VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_SASS_LOG=1 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim"))
  a="$(cd "$work" && env "${env[@]}" VGPU_TIER=0 timeout 900 "$bin" 2>&1)"; ra=$?
  b="$(cd "$work" && env "${env[@]}" VGPU_TIER=1 VGPU_TIER_STATS=1 timeout 900 "$bin" 2>&1)"; rb=$?
  local stats bq
  stats="$(grep '^tier: ' <<< "$b" | head -1)"
  bq="$(grep -v '^tier' <<< "$b")"
  grep -q "running SASS" <<< "$a" || why="its SASS did not run"
  grep -q "running PTX instead of SASS" <<< "$a" && why="fell back to PTX: $(grep -m1 'running PTX instead' <<< "$a")"
  [[ $ra == "$rb" ]] || why="${why:+$why; }exit status $ra with the interpreter, $rb with the tier"
  [[ $ra == 0 ]] || why="${why:+$why; }the run failed (exit $ra)"
  [[ "$a" == "$bq" ]] || why="${why:+$why; }the interpreter and the tier print differently"
  if [[ $p == sass_tier ]]; then
    local ran total
    ran="$(sed -n 's/^tier: \([0-9]*\) of \([0-9]*\) warp.*/\1/p' <<< "$stats")"
    total="$(sed -n 's/^tier: \([0-9]*\) of \([0-9]*\) warp.*/\2/p' <<< "$stats")"
    if [[ -z $ran || -z $total ]] || (( total == 0 || ran * 100 < total * min_percent )); then
      why="${why:+$why; }the tier ran ${ran:-?} of ${total:-?} warp instructions (wanted at least $min_percent%)"
    fi
  fi
  if [[ -n $why ]]; then
    echo "FAIL $arch $p: $why"
    diff <(echo "$a") <(echo "$bq") | head -12 | sed 's/^/    /'
    fails=1
  else
    echo "ok   $arch $p${stats:+  ($stats)}"
  fi
}
for j in "${jobs[@]}"; do check $j; done
[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: the tier and the interpreter differ"
exit $fails
