#!/usr/bin/env bash
# CUTLASS's own SM100 convolution unit tests, unmodified, on a simulated B200
# (nvidia/b200), each checked against CUTLASS's host reference. Implicit-GEMM
# convolutions: TMA's im2col mode feeding tcgen05.mma, and for dgrad and
# wgrad the transposed walks, which the GEMM tests never take.
#
#   conv2d fprop f16/f32, s8/s32 and tf32; dgrad and wgrad f16/f32.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
# As for run_cutlass_hopper.sh: googletest is built here without a sanitizer.
if [[ -n "$(shim_sanitizer "$shim")" ]]; then
  echo "SKIP: CUTLASS's unit tests run against the uninstrumented build"; exit 0
fi
probe="${TMPDIR:-/tmp}/vgpu_cutlass_sm100_conv_probe_$$.cu"
echo '__global__ void k() {}' > "$probe"
if ! nvcc -arch=compute_100a -code=compute_100a -c "$probe" -o /dev/null 2>/dev/null; then
  rm -f "$probe"
  echo "SKIP: this nvcc cannot target sm_100a (needs CUDA 12.8 or later)"; exit 0
fi
rm -f "$probe"
need_gtest=1
. "$root/nvidia/tests/e2e/cutlass_fetch.sh"
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_cutlass_sm100_conv.XXXXXX")"
trap 'rm -rf "$work"' EXIT

# directory/test file | gtest filter. VGPU_CUTLASS_SM100_CONV_ALL=1 adds the
# conv1d/conv3d, fp8, fusion and stream-K files.
tests=(
  "fprop/sm100_conv2d_fprop_implicit_gemm_f16_f16_f32_tensorop_f32|*"
  "dgrad/sm100_conv2d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f32|*"
  "wgrad/sm100_conv2d_wgrad_implicit_gemm_f16_f16_f32_tensorop_f32|*"
  "fprop/sm100_conv2d_fprop_implicit_gemm_s8_s8_s32_tensorop_s32|*"
  "fprop/sm100_conv2d_fprop_implicit_gemm_tf32_tf32_f32_tensorop_f32|*"
)
if [[ "${VGPU_CUTLASS_SM100_CONV_ALL:-0}" == 1 ]]; then
  shopt -s nullglob
  for f in "$cutlass"/test/unit/conv/device_3x/*/sm100_*.cu; do
    rel="${f#"$cutlass"/test/unit/conv/device_3x/}"
    rel="${rel%.cu}"
    [[ " ${tests[*]} " == *" $rel|*"* ]] || tests+=("$rel|*")
  done
  shopt -u nullglob
fi
u="$cutlass/test/unit"
# Each compile peaks near 10 GB; as many run together as the free memory (or
# the container's limit) holds, and never fewer than one (cutlass_fetch.sh).
jobs=$(cutlass_compile_jobs ${#tests[@]})
compile() {
  local path="$1" name="${1##*/}"
  nvcc -std=c++17 -O1 -cudart shared -arch=compute_100a -code=compute_100a --expt-relaxed-constexpr \
       -DCUTLASS_TARGET_NAME="\"$name\"" \
       -I "$cutlass/include" -I "$cutlass/tools/util/include" -I "$u/common" -I "$u" -I "$cutlass/test" \
       -I "$gtest/googletest/include" "$u/conv/device_3x/$path.cu" "$u/test_unit.cpp" \
       "$u/common/filter_architecture.cpp" "$gtest/libgtest.a" -o "$work/$name" >"$work/$name.log" 2>&1
}
for ((first = 0; first < ${#tests[@]}; first += jobs)); do
  pids=()
  names=()
  for entry in "${tests[@]:first:jobs}"; do
    IFS='|' read -r path _ <<<"$entry"
    compile "$path" & pids+=($!)
    names+=("${path##*/}")
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "FAIL: ${names[$i]} did not compile"; tail -20 "$work/${names[$i]}.log"; exit 1
    fi
  done
done
fail=0
for entry in "${tests[@]}"; do
  IFS='|' read -r path filter <<<"$entry"
  name="${path##*/}"
  if ! require_shim_libs "$shim" "$work/$name"; then exit 0; fi
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/b200 LD_LIBRARY_PATH="$shim" \
            "$work/$name" --gtest_filter="$filter" 2>&1 || true)"
  summary="$(grep -E '^\[  (PASSED|FAILED)  \]' <<<"$result" | head -2 | tr '\n' ' ')"
  echo "$name: $summary"
  if ! grep -q '^\[  PASSED  \]' <<<"$result" || grep -q '^\[  FAILED  \]' <<<"$result"; then
    grep -E 'FAILED|Failure|VirtualGPU|timed out|exception|thrown|Error|error:' <<<"$result" | head -30
    fail=1
  fi
done
exit $fail
