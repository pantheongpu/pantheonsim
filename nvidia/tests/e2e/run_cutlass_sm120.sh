#!/usr/bin/env bash
# CUTLASS's own SM120 GEMM unit tests, unmodified, on a simulated RTX 5090
# (nvidia/rtx5090), each checked against CUTLASS's host reference. They run
# sm_120's warp-level tensor core: mma.sync's .kind::f8f6f4 with fp4/fp6 in
# 8-bit containers loaded by ldmatrix's .b8x16 expansion, the block-scaled
# .kind::mxf8f6f4/mxf4/mxf4nvf4 forms with their scale selectors, the sparse
# mma.sp::ordered_metadata kinds, and the fp8 scale-factor epilogue.
#
#   sm120_gemm_f4_f6_f32:          e2m1 x e3m2 / e2m3 x e2m1 through ldmatrix.
#   sm120_bs_gemm_mxf4_mxf4:       .kind::mxf4 with UE8M0 factors.
#   sm120_bs_gemm_nvf4_*_epilogue: .kind::mxf4nvf4, UE4M3, e2m1 output.
#   sm120_bssp_gemm_f8t_f8n_f8t:   sparse mxf8f6f4 with an e4m3 + UE8M0 output.
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
probe="${TMPDIR:-/tmp}/vgpu_cutlass_sm120_probe_$$.cu"
echo '__global__ void k() {}' > "$probe"
if ! nvcc -arch=compute_120a -code=compute_120a -c "$probe" -o /dev/null 2>/dev/null; then
  rm -f "$probe"
  echo "SKIP: this nvcc cannot target sm_120a (needs CUDA 12.8 or later)"; exit 0
fi
rm -f "$probe"
need_gtest=1
. "$root/nvidia/tests/e2e/cutlass_fetch.sh"
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_cutlass_sm120.XXXXXX")"
trap 'rm -rf "$work"' EXIT

# directory/test file | gtest filter. Every case of these files passes;
# VGPU_CUTLASS_SM120_ALL=1 adds the rest of the dense, block-scaled and sparse
# files that were checked by hand.
tests=(
  "sm120_tensorop_gemm/sm120_gemm_f4_f6_f32_tensor_op|*"
  "sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_mxf4_mxf4_f32_f32|*"
  "sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_epilogue|*"
  "sm120_blockscaled_sparse_tensorop_gemm/sm120_bssp_gemm_f8t_f8n_f8t_tensor_op|*"
)
if [[ "${VGPU_CUTLASS_SM120_ALL:-0}" == 1 ]]; then
  tests+=("sm120_tensorop_gemm/sm120_gemm_f8_f8_f32_tensor_op|*"
          "sm120_tensorop_gemm/sm120_gemm_f4_f4_f32_tensor_op|*"
          "sm120_tensorop_gemm/sm120_gemm_f6_f6_f16_tensor_op|*"
          "sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_mxf6_mxf8_f32_f32|*"
          "sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_f32|*"
          "sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_bf16|*"
          "sm120_sparse_tensorop_gemm/sm120_sparse_gemm_f4_f4_f32_tensor_op|*"
          "sm120_sparse_tensorop_gemm/sm120_sparse_gemm_f8_f6_f32_tensor_op|*"
          "sm120_blockscaled_sparse_tensorop_gemm/sm120_bssp_gemm_f4_f4_f32_tensor_op|*"
          "sm120_blockscaled_sparse_tensorop_gemm/sm120_bssp_gemm_f8_f6_f32_tensor_op|*"
          "sm120_blockscaled_sparse_tensorop_gemm/sm120_bssp_gemm_f6_f4_f32_tensor_op|*")
fi
u="$cutlass/test/unit"
# Each compile peaks near 10 GB; as many run together as the free memory (or
# the container's limit) holds, and never fewer than one (cutlass_fetch.sh).
jobs=$(cutlass_compile_jobs ${#tests[@]})
compile() {
  local path="$1" name="${1##*/}"
  nvcc -std=c++17 -O1 -cudart shared -arch=compute_120a -code=compute_120a --expt-relaxed-constexpr \
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
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/rtx5090 LD_LIBRARY_PATH="$shim" \
            "$work/$name" --gtest_filter="$filter" 2>&1 || true)"
  summary="$(grep -E '^\[  (PASSED|FAILED)  \]' <<<"$result" | head -2 | tr '\n' ' ')"
  echo "$name: $summary"
  if ! grep -q '^\[  PASSED  \]' <<<"$result" || grep -q '^\[  FAILED  \]' <<<"$result"; then
    grep -E 'FAILED|Failure|VirtualGPU|timed out' <<<"$result" | head -20
    fail=1
  fi
done
exit $fail
