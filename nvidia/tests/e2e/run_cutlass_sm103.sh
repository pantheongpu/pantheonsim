#!/usr/bin/env bash
# CUTLASS's own SM103 GEMM unit tests, unmodified, on a simulated B300
# (nvidia/b300), each checked against CUTLASS's host reference. They run
# Blackwell Ultra's fp4 "ultra" tensor-core MMA: .kind::mxf4nvf4 at K = 96
# (instruction descriptor bit 31) with three or six scale factors a row, and
# shared-memory descriptors whose leading-dimension field is the absolute
# address of the next pipeline buffer (bit 52), where a K block straddles two.
#
#   sm103_gemm_f4_f4_f32_tensor_op_f32_1sm: one CTA a tile.
#   sm103_gemm_f4_f4_f32_tensor_op_f32_2sm: a CTA pair a tile.
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
probe="${TMPDIR:-/tmp}/vgpu_cutlass_sm103_probe_$$.cu"
echo '__global__ void k() {}' > "$probe"
if ! nvcc -arch=compute_103a -code=compute_103a -c "$probe" -o /dev/null 2>/dev/null; then
  rm -f "$probe"
  echo "SKIP: this nvcc cannot target sm_103a (needs CUDA 12.9 or later)"; exit 0
fi
rm -f "$probe"
need_gtest=1
. "$root/nvidia/tests/e2e/cutlass_fetch.sh"
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_cutlass_sm103.XXXXXX")"
trap 'rm -rf "$work"' EXIT

# test file | gtest filter. VGPU_CUTLASS_SM103_ALL=1 adds the grouped,
# ptr-array and stream-K files.
tests=(
  "sm103_gemm_f4_f4_f32_tensor_op_f32_1sm|*"
  "sm103_gemm_f4_f4_f32_tensor_op_f32_2sm|*"
)
if [[ "${VGPU_CUTLASS_SM103_ALL:-0}" == 1 ]]; then
  tests+=("sm103_gemm_f4_f4_f32_tensor_op_f32_group_1sm_128x128|*"
          "sm103_gemm_f4_f4_f32_tensor_op_f32_ptr_array_1sm_128x128|*"
          "sm103_gemm_f4_f4_f32_tensor_op_f32_stream_k|*")
fi
u="$cutlass/test/unit"
# Each compile peaks near 10 GB; as many run together as the free memory (or
# the container's limit) holds, and never fewer than one (cutlass_fetch.sh).
jobs=$(cutlass_compile_jobs ${#tests[@]})
compile() {
  local path="$1" name="${1##*/}"
  nvcc -std=c++17 -O1 -cudart shared -arch=compute_103a -code=compute_103a --expt-relaxed-constexpr \
       -DCUTLASS_TARGET_NAME="\"$name\"" \
       -I "$cutlass/include" -I "$cutlass/tools/util/include" -I "$u/common" -I "$u" -I "$cutlass/test" \
       -I "$gtest/googletest/include" "$u/gemm/device/$path.cu" "$u/test_unit.cpp" \
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
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/b300 LD_LIBRARY_PATH="$shim" \
            "$work/$name" --gtest_filter="$filter" 2>&1 || true)"
  summary="$(grep -E '^\[  (PASSED|FAILED)  \]' <<<"$result" | head -2 | tr '\n' ' ')"
  echo "$name: $summary"
  if ! grep -q '^\[  PASSED  \]' <<<"$result" || grep -q '^\[  FAILED  \]' <<<"$result"; then
    grep -E 'FAILED|Failure|VirtualGPU|timed out' <<<"$result" | head -20
    fail=1
  fi
done
exit $fail
