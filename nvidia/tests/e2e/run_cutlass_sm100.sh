#!/usr/bin/env bash
# CUTLASS's own SM100 GEMM unit tests, unmodified, on a simulated B200, each
# checked against CUTLASS's host reference. They run Blackwell's whole path:
# Tensor Memory allocated by one warp and read back by the epilogue warps,
# tcgen05.mma issued by one thread for one CTA or a CTA pair, tcgen05.commit
# multicast to the pair's barriers, TMA loads with .cta_group::2 completing on
# the even CTA's barrier (reached by clearing bit 24 of an odd CTA's own
# address), the 128-byte swizzle in 32-byte atoms for the f32 epilogue, and
# cluster launch control taking over tiles that have not started.
#
#   f16_f16_void_f32: 1-SM and 2-SM kernels, M = 64/128/256, stream-K.
#   f8_f8_void_f32:   .kind::f8f6f4 with e4m3/e5m2.
#   s8_s8_void_s32:   .kind::i8.
#   sm100_sp_gemm_*:  tcgen05.mma.sp, the metadata in Tensor Memory (f16,
#                     tf32, f8, s8).
#   mxf8_mxf6_*:      .kind::mxf8f6f4.block_scale, e4m3 x e2m3 with UE8M0
#                     factors copied in by tcgen05.cp, B loaded by TMA as
#                     .b6x16_p32.
#   *_blockwise:      fp8 with per-block scales applied on the CUDA cores,
#                     loaded by cp.async and arriving with .noinc.
#   *_ptr_array:      grouped fp4 GEMM on a 4x4 cluster, a tensor map per SM
#                     rewritten by tensormap.replace -- %smid must be distinct
#                     among the resident blocks.
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
probe="${TMPDIR:-/tmp}/vgpu_cutlass_sm100_probe_$$.cu"
echo '__global__ void k() { asm volatile("tcgen05.fence::before_thread_sync;"); }' > "$probe"
if ! nvcc -arch=compute_100a -code=compute_100a -c "$probe" -o /dev/null 2>/dev/null; then
  rm -f "$probe"
  echo "SKIP: this nvcc cannot target sm_100a (needs CUDA 12.8 or later)"; exit 0
fi
rm -f "$probe"
need_gtest=1
. "$root/nvidia/tests/e2e/cutlass_fetch.sh"
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_cutlass_sm100.XXXXXX")"
trap 'rm -rf "$work"' EXIT

# directory/test file | gtest filter. All cases of these files pass; the
# stream-K ones take minutes each here, so by default one 1-SM and one 2-SM
# stream-K kernel stand for them, and a sparse and a block-scaled file run
# part of theirs. VGPU_CUTLASS_SM100_ALL=1 runs every case.
tests=(
  "sm100_tensorop_gemm/f16_f16_void_f32|*-*_1x8x1_streamK"
  "sm100_tensorop_gemm/f8_f8_void_f32|*-*streamK"
  "sm100_tensorop_gemm/s8_s8_void_s32|*-*streamK"
  "sm100_sparse_tensorop_gemm/sm100_sp_gemm_f16_f16_f32_f16_f16_hmma|*void*"
  "sm100_blockscaled_tensorop_gemm/mxf8_mxf6_f16_f8_tn_layout|*4x4x1_1sm_auto"
  "sm100_gemm_f8_f8_f8_tensor_op_f32_blockwise|*2sm*64x4x32*"
  "sm100_gemm_f4_f4_f32_tensor_op_f32_ptr_array|*2sm*4x4x1*"
)
if [[ "${VGPU_CUTLASS_SM100_ALL:-0}" == 1 ]]; then
  tests=("sm100_tensorop_gemm/f16_f16_void_f32|*" "sm100_tensorop_gemm/f8_f8_void_f32|*"
         "sm100_tensorop_gemm/s8_s8_void_s32|*")
  for f in f16_f16_f32_f16_f16_hmma f32_f32_f32_f32_f32_tfmma f8_f8_f32_f32_f32_qmma s8_s8_s32_s8_s8_imma; do
    tests+=("sm100_sparse_tensorop_gemm/sm100_sp_gemm_$f|*")
  done
  tests+=("sm100_blockscaled_tensorop_gemm/mxf8_mxf6_f16_f8_tn_layout|*"
          "sm100_gemm_f8_f8_f8_tensor_op_f32_blockwise|*"
          "sm100_gemm_f4_f4_f32_tensor_op_f32_ptr_array|*")
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
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/b200 LD_LIBRARY_PATH="$shim" \
            "$work/$name" --gtest_filter="$filter" 2>&1 || true)"
  summary="$(grep -E '^\[  (PASSED|FAILED)  \]' <<<"$result" | head -2 | tr '\n' ' ')"
  echo "$name: $summary"
  if ! grep -q '^\[  PASSED  \]' <<<"$result" || grep -q '^\[  FAILED  \]' <<<"$result"; then
    grep -E 'FAILED|Failure|VirtualGPU|timed out' <<<"$result" | head -20
    fail=1
  fi
done
exit $fail
