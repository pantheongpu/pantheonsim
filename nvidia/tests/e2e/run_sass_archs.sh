#!/usr/bin/env bash
# SASS on every architecture (nvidia/docs/sass.md). A program built with
# -arch=sm_XX carries that generation's SASS and its PTX, and the simulator
# runs the SASS by default. Each program below, built so for each generation
# from Turing to Blackwell, runs on that generation's simulated GPU twice: by
# default, where the log must say its SASS ran, and on its PTX (VGPU_SASS=0).
# Both runs must pass and print the same.
#
# An nvcc too old for an architecture (sm_100 and sm_120 need CUDA 12.8) skips
# it and says so. VGPU_SASS_ARCHS and VGPU_SASS_PROGRAMS narrow the lists;
# VGPU_SASS_JOBS caps the builds run at once (default: one per CPU).
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
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-sass-archs.XXXXXX")"
trap 'rm -rf "$work"' EXIT

declare -A gpu=([sm_75]=nvidia/t4 [sm_80]=nvidia/a100 [sm_86]=nvidia/rtx3060 [sm_89]=nvidia/l4
                [sm_90]=nvidia/h100 [sm_100]=nvidia/b200 [sm_120]=nvidia/rtx5090)
archs=(${VGPU_SASS_ARCHS:-sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120})
# program:first architecture it builds for (the MMA programs need sm_80)
# [:last one it runs on (runtime_conformance checks a T4's properties)].
progs=(${VGPU_SASS_PROGRAMS:-sass_archs:75 vector_add:75 device_functions:75 device_intrinsics:75 video_forms:75
       runtime_conformance:75:75 symbols:75 surface_oob:75 textures:75 texture_filtering:75 texture_gather:75
       texture_layers:75 texture_mipmaps:75 block_semaphore:75 cooperative_grid:75 mma_forms:80
       mma_fragment_layout:80 modern_dtypes:80 wmma_gemm:80 wmma_types:80})

supported="$("$nvcc_bin" --list-gpu-code 2>/dev/null)"
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
jobs=()
for arch in "${archs[@]}"; do
  if ! grep -qx "$arch" <<< "$supported"; then echo "SKIP $arch: this nvcc cannot target it"; continue; fi
  for p in "${progs[@]}"; do
    IFS=: read -r name first last <<< "$p"
    (( ${arch#sm_} >= first && ${arch#sm_} <= ${last:-999} )) && jobs+=("$name $arch")
  done
done
(( ${#jobs[@]} )) || { echo "SKIP: no architecture this nvcc can target"; exit 0; }

# Builds, several at once.
export nvcc_bin root work
export san="${san_flags[*]}"
printf '%s\n' "${jobs[@]}" | xargs -P "${VGPU_SASS_JOBS:-$(nproc)}" -L 1 bash -c '
  "$nvcc_bin" -std=c++17 -arch="$1" -cudart shared -w -Wno-deprecated-gpu-targets $san \
    -I"$root/nvidia/tests/e2e" "$root/nvidia/tests/e2e/$0.cu" -o "$work/$0.$1" -lcuda 2> "$work/$0.$1.build" ||
    echo "build failed" >> "$work/$0.$1.build"'

fails=0
check() {
  local p=$1 arch=$2 bin="$work/$1.$2" sass ptx rs rp
  if [[ ! -x $bin ]]; then
    echo "FAIL $arch $p: does not build"; sed 's/^/    /' "$bin.build" | tail -5; fails=1; return
  fi
  require_shim_libs "$shim" "$bin" >/dev/null || return
  local env=(VGPU_QUIET=1 VGPU_GPU=${gpu[$arch]} LD_LIBRARY_PATH="$shim")
  [[ $p == runtime_conformance ]] && env+=(VGPU_DEVICE_COUNT=2)   # as run_runtime_conformance.sh runs it
  sass="$(cd "$work" && env "${env[@]}" VGPU_SASS_LOG=1 timeout 600 "$bin" 2>&1)"; rs=$?
  ptx="$(cd "$work" && env "${env[@]}" VGPU_SASS=0 timeout 600 "$bin" 2>&1)"; rp=$?
  local why=""
  grep -q "running SASS" <<< "$sass" || why="its SASS did not run"
  grep -q "running PTX instead of SASS" <<< "$sass" && why="fell back to PTX: $(grep -m1 'running PTX instead' <<< "$sass")"
  [[ $rs == 0 ]] || why="${why:+$why; }the SASS run failed (exit $rs)"
  [[ $rp == 0 ]] || why="${why:+$why; }the PTX run failed (exit $rp)"
  [[ "$(grep -v '^\[vgpu\] running' <<< "$sass")" == "$ptx" ]] || why="${why:+$why; }the SASS and PTX runs print differently"
  if [[ -n $why ]]; then
    echo "FAIL $arch $p: $why"
    diff <(echo "$ptx") <(grep -v '^\[vgpu\] running' <<< "$sass") | head -8 | sed 's/^/    /'
    fails=1
  else
    echo "ok   $arch $p"
  fi
}
for j in "${jobs[@]}"; do check $j; done
[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: SASS on every architecture"
exit $fails
