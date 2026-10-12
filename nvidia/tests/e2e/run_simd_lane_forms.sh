#!/usr/bin/env bash
# The plain ALU and tensor-core forms the simulator's SIMD kernels run (simd_lane_forms.cu), hashed over values that
# break arithmetic, on both engines at every SIMD level (VGPU_SIMD unset, =base, =off), against what the simulator
# printed before the kernels existed:
#
#   nvidia/tests/data/simd_lane_forms.sass.txt, .ptx.txt   the simulator's own hashes, pinned (a kernel that changes
#                                                           a bit anywhere fails here)
#   nvidia/tests/data/simd_lane_forms.rtx3060.txt          an RTX 3060's hashes. The simulator agrees with it on
#                                                           every integer, float, half and double form; the
#                                                           tensor-core forms over full-range random data differ (the
#                                                           card orders its sums differently, and the pinned files
#                                                           keep what the simulator does). The card-checked tensor
#                                                           forms are mma_forms and wmma_types.
#
#   run_simd_lane_forms.sh                  simulator vs the pinned files
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
# The PTX engine's wmma produces NaNs by adding and multiplying NaNs (and infinities) in its tile, and which NaN comes
# out -- its sign, its payload -- is the one the host instruction returns for two NaN operands: the first source
# operand's. The compiler chooses the order of the operands of every vector add and multiply (they are commutative),
# so it differs between an optimised build and an instrumented -O1 one: the same source gives 0x7fc00000 where the
# other gives 0xffc00000. Measured with a ThreadSanitizer Debug build against the optimised one, in the f32-accumulator
# lines (f16, bf16, tf32; random and edge data): 6879 / 7626, 9259 / 9879 and 27 / 12495 elements differ, every one a
# NaN against a NaN, and not an element that is not a NaN; VGPU_SIMD=off does the same, so it is the PTX tile and not
# the SIMD kernels. simd_lane_forms.cu therefore prints a second line for each, _nonan, with every NaN made the
# canonical one. The raw lines are compared in an uninstrumented build (where they are pinned) and left out under a
# sanitizer; the _nonan lines are compared everywhere.
sanitizer="$(shim_sanitizer "$shim")"
skip_lines='^wmma_(f16|bf16|tf32)_f32acc +(random|edges) '
fails=0
for engine in sass ptx; do
  for level in default base off; do
    case "$level" in default) lv=();; *) lv=(VGPU_SIMD="$level");; esac
    env "${lv[@]}" VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_SASS=$([[ $engine == sass ]] && echo 1 || echo 0) \
        LD_LIBRARY_PATH="$shim" "$out/app" > "$out/sim.txt" 2>&1 || true
    cp "$data/simd_lane_forms.$engine.txt" "$out/want.txt"
    note=""
    if [[ -n "$sanitizer" && $engine == ptx ]]; then
      grep -Ev "$skip_lines" "$data/simd_lane_forms.$engine.txt" > "$out/want.txt"
      grep -Ev "$skip_lines" "$out/sim.txt" > "$out/sim_cmp.txt" || true
      mv "$out/sim_cmp.txt" "$out/sim.txt"
      note=" (the raw wmma f32-accumulator lines are not compared under $sanitizer, their _nonan twins are: see the comment above)"
    fi
    if diff "$out/want.txt" "$out/sim.txt" > "$out/d.txt"; then
      echo "ok   $engine VGPU_SIMD=$level$note"
    else
      echo "FAIL $engine VGPU_SIMD=$level: differs from the pinned hashes"; head -10 "$out/d.txt"; fails=1
    fi
  done
done
echo "(informational) lines where the simulator differs from the RTX 3060: sass $(diff "$data/simd_lane_forms.rtx3060.txt" "$data/simd_lane_forms.sass.txt" | grep -c '^>') ptx $(diff "$data/simd_lane_forms.rtx3060.txt" "$data/simd_lane_forms.ptx.txt" | grep -c '^>')"
(( fails == 0 )) && echo PASS || { echo FAIL; exit 1; }
