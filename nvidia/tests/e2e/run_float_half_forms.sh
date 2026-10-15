#!/usr/bin/env bash
# The f32, f16 and f64 forms with a flush, a saturation, a rounding mode, a NaN rule or a byte permute (float_half_forms.cu), hashed
# per form over values that break arithmetic, against an RTX 3060's hashes (nvidia/tests/data/float_half_forms.rtx3060.txt). Both
# of the simulator's engines -- SASS, the default, and PTX, VGPU_SASS=0 -- at every SIMD level (VGPU_SIMD unset, =base, =off) must
# print exactly the card's lines: SASS == PTX == card, line by line. (Before this test the SASS executor disagreed with the card on
# FFMA/FMUL .SAT of a negative zero, FMNMX of two NaNs, MUFU's NaNs and TANH's subnormals, PRMT.F4E/B4E/ECL/ECR, HADD2.F32 and
# F2FP.PACK_AB of a NaN, F2FP.RELU of -0 and the NaN of DADD/DMUL/DFMA given a negated NaN; the PTX engine on
# cvt.rmi.ftz.s32.f32 of a negative subnormal.)
#
#   run_float_half_forms.sh                    simulator vs the card file
#   run_float_half_forms.sh --card [--update]  NVIDIA's runtime on a real GPU vs the card file (or rewrite it)
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
data="$root/nvidia/tests/data"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/float_half_forms.cu"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
out="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_float_half_forms.XXXXXX")"
trap 'rm -rf "$out"' EXIT
# (-fmad=true, nvcc's default, is what the card file was made with: the kernels' separate multiplies and adds are not contracted
# because they are written as inline PTX or intrinsics.)
build() { nvcc -std=c++17 -O1 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "$@" "$src" -o "$out/app"; }
if [[ "${1:-}" == --card ]]; then
  build
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  LD_LIBRARY_PATH="$cuda_root/lib64" "$out/app" > "$out/card.txt"
  if [[ "${2:-}" == --update ]]; then cp "$out/card.txt" "$data/float_half_forms.rtx3060.txt"; echo "wrote float_half_forms.rtx3060.txt"; exit 0; fi
  diff "$data/float_half_forms.rtx3060.txt" "$out/card.txt" && { echo "PASS (card)"; exit 0; }
  echo "FAIL: the card disagrees with float_half_forms.rtx3060.txt"; exit 1
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
    env "${lv[@]}" VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_SASS=$([[ $engine == sass ]] && echo 1 || echo 0) VGPU_SASS_LOG=1 \
        LD_LIBRARY_PATH="$shim" "$out/app" > "$out/sim.txt" 2>&1 || true
    # On the SASS engine every kernel must run as SASS: an instruction the decoder does not know sends a kernel to the PTX engine,
    # which gives the same answers here and would hide that (VABSDIFF, the instruction __sad and __usad compile to, was one).
    if [[ $engine == sass ]] && grep -q 'running PTX instead of SASS' "$out/sim.txt"; then
      echo "FAIL $engine VGPU_SIMD=$level: a kernel fell back to PTX:"; grep 'running PTX instead of SASS' "$out/sim.txt" | head -5; fails=1
    fi
    grep -v '^\[vgpu\]' "$out/sim.txt" > "$out/sim1.txt" || true; mv "$out/sim1.txt" "$out/sim.txt"
    cp "$data/float_half_forms.rtx3060.txt" "$out/want.txt"
    if [[ $engine == ptx ]]; then   # (the lines that need an operand modifier ptxas folds: see float_half_forms.cu)
      grep -v 'sassonly_' "$out/want.txt" > "$out/want2.txt"; mv "$out/want2.txt" "$out/want.txt"
      grep -v 'sassonly_' "$out/sim.txt" > "$out/sim2.txt"; mv "$out/sim2.txt" "$out/sim.txt"
    fi
    if diff "$out/want.txt" "$out/sim.txt" > "$out/d.txt"; then
      echo "ok   $engine VGPU_SIMD=$level"
    else
      echo "FAIL $engine VGPU_SIMD=$level: differs from the card's hashes"; head -20 "$out/d.txt"; fails=1
    fi
  done
done
(( fails == 0 )) && echo PASS || { echo FAIL; exit 1; }
