#!/usr/bin/env bash
# The plain ALU and tensor-core forms the simulator's SIMD kernels run (simd_lane_forms.cu), hashed over values that
# break arithmetic, on both engines at every SIMD level (VGPU_SIMD unset, =base, =off), against what the simulator
# printed before the kernels existed:
#
#   nvidia/tests/data/simd_lane_forms.sass.txt, .ptx.txt   the simulator's own hashes, pinned (a kernel that changes
#                                                           a bit anywhere fails here)
#   nvidia/tests/data/simd_lane_forms.rtx3060.txt          an RTX 3060's hashes. The simulator agrees with it on every line, on both
#                                                           engines: the integer, float, half and double forms, and the f16, bf16 and
#                                                           tf32 tensor core over full-range random data and over the special values
#                                                           (the tensor core sums in chunks with truncation, include/vgpu/exec/
#                                                           tc_ampere.hpp; the sequential f32 sum this simulator used before differed
#                                                           from the card on the random lines). So the pinned files ARE the card's
#                                                           lines, and this script requires it.
#
#   nvidia/tests/data/simd_lane_forms.sass.tcseq.txt, .ptx.tcseq.txt   what VGPU_TC_EXACT=0 prints (the tensor cores' earlier sequential f32
#                                                           sum, on sm_86 too): the sass file is r8-a's pinned file byte for byte; the ptx
#                                                           file is r8-a's with its two wmma tf32 lines replaced (random, edges), because
#                                                           the PTX engine no longer rounds a tf32 operand when wmma.load reads it -- it
#                                                           truncates, as the SASS executor and the card do -- a fix that is not part of
#                                                           the sum model and is not switched. Every other line is the same in both modes.
#
#   run_simd_lane_forms.sh                  simulator vs the pinned files, with the tensor-core model on (the default) and off
#   run_simd_lane_forms.sh --card [--update]   NVIDIA's runtime on a real GPU vs the card file (or rewrite it)
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
data="$root/nvidia/tests/data"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/simd_lane_forms.cu"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
out="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_simd_lane_forms.XXXXXX")"
trap 'rm -rf "$out"' EXIT
build() { nvcc -std=c++17 -O1 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "$@" "$src" -o "$out/app"; }
if [[ "${1:-}" == --card ]]; then
  build
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  LD_LIBRARY_PATH="$cuda_root/lib64" "$out/app" > "$out/card.txt"
  if [[ "${2:-}" == --update ]]; then cp "$out/card.txt" "$data/simd_lane_forms.rtx3060.txt"; echo "wrote simd_lane_forms.rtx3060.txt"; exit 0; fi
  diff "$data/simd_lane_forms.rtx3060.txt" "$out/card.txt" && { echo "PASS (card)"; exit 0; }
  echo "FAIL: the card disagrees with simd_lane_forms.rtx3060.txt"; exit 1
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
build $(shim_sanitizer_nvcc_flags "$shim")
if ! require_shim_libs "$shim" "$out/app"; then exit 0; fi
fails=0
for engine in sass ptx; do
  for level in default base off; do
    case "$level" in default) lv=();; *) lv=(VGPU_SIMD="$level");; esac
    env "${lv[@]}" VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_SASS=$([[ $engine == sass ]] && echo 1 || echo 0) \
        LD_LIBRARY_PATH="$shim" "$out/app" > "$out/sim.txt" 2>&1 || true
    cp "$data/simd_lane_forms.$engine.txt" "$out/want.txt"
    if diff "$out/want.txt" "$out/sim.txt" > "$out/d.txt"; then
      echo "ok   $engine VGPU_SIMD=$level"
    else
      echo "FAIL $engine VGPU_SIMD=$level: differs from the pinned hashes"; head -10 "$out/d.txt"; fails=1
    fi
  done
done
# VGPU_TC_EXACT=0: the earlier sequential sums, every engine and level; and a mutation guard: the two modes must differ on exactly the
# tensor-core lines (a switch an engine ignores would print the card's lines here).
for engine in sass ptx; do
  for level in default base off; do
    case "$level" in default) lv=();; *) lv=(VGPU_SIMD="$level");; esac
    env "${lv[@]}" VGPU_TC_EXACT=0 VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_SASS=$([[ $engine == sass ]] && echo 1 || echo 0) \
        LD_LIBRARY_PATH="$shim" "$out/app" > "$out/sim.txt" 2>&1 || true
    if diff "$data/simd_lane_forms.$engine.tcseq.txt" "$out/sim.txt" > "$out/d.txt"; then
      echo "ok   $engine VGPU_SIMD=$level VGPU_TC_EXACT=0"
    else
      echo "FAIL $engine VGPU_SIMD=$level VGPU_TC_EXACT=0: differs from the sequential-sum pins"; head -10 "$out/d.txt"; fails=1
    fi
  done
  # (the lines that differ between the two modes are the wmma and mma tensor lines, and only those)
  if diff "$data/simd_lane_forms.$engine.tcseq.txt" "$data/simd_lane_forms.$engine.txt" | grep '^[<>]' | grep -v -e 'wmma_' -e 'mma_' > "$out/d.txt"; then
    echo "FAIL: the modes differ on lines that are not tensor-core sums (sass/ptx $engine):"; head -5 "$out/d.txt"; fails=1
  fi
done
# The pinned files are the card's lines: SASS == PTX == card.
for engine in sass ptx; do
  if ! diff "$data/simd_lane_forms.rtx3060.txt" "$data/simd_lane_forms.$engine.txt" > "$out/d.txt"; then
    echo "FAIL: simd_lane_forms.$engine.txt differs from the RTX 3060's lines:"; head -10 "$out/d.txt"; fails=1
  fi
done
(( fails == 0 )) && echo PASS || { echo FAIL; exit 1; }
