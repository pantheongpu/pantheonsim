#!/usr/bin/env bash
# The deliberate round 9 changes, against an RTX 3060 (round9_probe.cu, written by the r9-verify agent): tensor-core sums, float and
# double NaN selection (every NaN pair of dadd, dmul, dfma, dmin, dmax), conversions, -0 under .sat/.ftz, bf16 HFMA2, 64-bit saturation,
# __sad, and the launch error codes. The program prints one line per section (a hash of every result and the first values) and nothing
# that varies from run to run, so the card's lines are pinned:
#
#   nvidia/tests/data/round9_probe.rtx3060.txt     an RTX 3060's lines (CUDA 13.0)
#
#   run_round9_probe.sh                      simulator, both engines at every SIMD level, vs the card file; and VGPU_TC_EXACT=0
#   run_round9_probe.sh --card [--update]    NVIDIA's runtime on a real GPU vs the card file (or rewrite it)
#
# Every line must equal the card's but those of the known gaps listed in `gaps` below, each of which is compared without its hash (the
# first values it prints must still agree). With the tensor-core model switched off (VGPU_TC_EXACT=0) only the three wmma lines may
# differ from the default mode. On the SASS engine nothing may fall back to PTX.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
data="$root/nvidia/tests/data"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/round9_probe.cu"
card_file="$data/round9_probe.rtx3060.txt"
# The known gaps, one section name each (a regular expression on the first column):
#   f32_lg2, f32_sin  MUFU.LG2 and MUFU.SIN are approximations on the card (within a few ulp of the correctly rounded result); the
#                     simulator computes them with libm, so the hash differs. The NaN and infinity results (the printed values) agree.
gaps='^(f32_lg2|f32_sin)$'
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
out="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_round9_probe.XXXXXX")"
trap 'rm -rf "$out"' EXIT
build() { nvcc -std=c++17 -O1 -cudart shared -arch=sm_86 -w -Wno-deprecated-gpu-targets "$@" "$src" -o "$out/app"; }
if [[ "${1:-}" == --card ]]; then
  build
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  LD_LIBRARY_PATH="$cuda_root/lib64:$cuda_root/targets/x86_64-linux/lib" "$out/app" > "$out/card.txt"
  if [[ "${2:-}" == --update ]]; then cp "$out/card.txt" "$card_file"; echo "wrote round9_probe.rtx3060.txt"; exit 0; fi
  diff "$card_file" "$out/card.txt" && { echo "PASS (card)"; exit 0; }
  echo "FAIL: the card disagrees with round9_probe.rtx3060.txt"; exit 1
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
build $(shim_sanitizer_nvcc_flags "$shim")
if ! require_shim_libs "$shim" "$out/app"; then exit 0; fi
# A line without its hash, for the gaps.
unhash() { awk -v gaps="$gaps" '$1 ~ gaps { $2 = "-" } { print }'; }
fails=0
for engine in sass ptx; do
  for level in default base off; do
    case "$level" in default) lv=();; *) lv=(VGPU_SIMD="$level");; esac
    env "${lv[@]}" VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 VGPU_SASS=$([[ $engine == sass ]] && echo 1 || echo 0) VGPU_SASS_LOG=1 \
        LD_LIBRARY_PATH="$shim" "$out/app" > "$out/sim.txt" 2>&1 || true
    if [[ $engine == sass ]] && grep -q 'running PTX instead of SASS' "$out/sim.txt"; then
      echo "FAIL $engine VGPU_SIMD=$level: a kernel fell back to PTX:"; grep 'running PTX instead of SASS' "$out/sim.txt" | head -5; fails=1
    fi
    grep -v '^\[vgpu\]' "$out/sim.txt" > "$out/sim1.txt" || true
    if diff <(unhash < "$card_file") <(unhash < "$out/sim1.txt") > "$out/d.txt"; then
      echo "ok   $engine VGPU_SIMD=$level"
    else
      echo "FAIL $engine VGPU_SIMD=$level: differs from the card"; head -12 "$out/d.txt"; fails=1
    fi
    if [[ $level == default ]]; then cp "$out/sim1.txt" "$out/default.$engine.txt"; fi
  done
  # The tensor-core model off: only the wmma lines move.
  env VGPU_TC_EXACT=0 VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 VGPU_SASS=$([[ $engine == sass ]] && echo 1 || echo 0) \
      LD_LIBRARY_PATH="$shim" "$out/app" > "$out/off.txt" 2>&1 || true
  changed="$( { diff "$out/default.$engine.txt" "$out/off.txt" || true; } | { grep '^>' || true; } | awk '{print $2}' | tr '\n' ' ')"
  if [[ "$changed" == "wmma_f16 wmma_bf16 wmma_tf32 " ]]; then
    echo "ok   $engine VGPU_TC_EXACT=0 changes the three wmma lines and nothing else"
  else
    echo "FAIL $engine VGPU_TC_EXACT=0 changed: ${changed:-nothing} (want exactly wmma_f16 wmma_bf16 wmma_tf32)"; fails=1
  fi
done
(( fails == 0 )) && echo PASS || { echo FAIL; exit 1; }
