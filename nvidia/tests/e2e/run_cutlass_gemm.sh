#!/usr/bin/env bash
# CUTLASS's own GEMM unit tests, unmodified, on a simulated A100 and H100: the
# cases that have found interpreter bugs, each checked against CUTLASS's host
# reference. They are regression tests for those bugs at the level of real
# nvcc output, where a unit test's hand-written PTX can miss the shape that
# matters.
#
#   SM80 sparse GEMM (all 19 tile shapes, blocks spread over four host
#   threads): mma.sp; empty cp.async groups counting toward wait_group; the
#   split-K semaphore wait -- bar.red in a loop with thread 0 on a higher-pc
#   path, and __syncthreads_and's brace-scoped %p1/%p2.
#   SM90 group GEMM, identity and silu epilogues (2x2x1 clusters): wgmma
#   reading its operands once per warpgroup; cvt.sat and tiny fp16 results in
#   the silu epilogue's expf.
#   SM90 ping-pong GEMM, 2x4x1 and the SIMT epilogue at 2x2x1: the same
#   wgmma operand read, and wgmma.wait_group holding a warp until its whole
#   warpgroup has issued.
#   SM90 s8 GEMM: cvt.pack.sat in the epilogue.
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
probe="${TMPDIR:-/tmp}/vgpu_cutlass_gemm_probe_$$.cu"
echo '__global__ void k() { asm volatile("wgmma.fence.sync.aligned;"); }' > "$probe"
if ! nvcc -arch=compute_90a -code=compute_90a -c "$probe" -o /dev/null 2>/dev/null; then
  rm -f "$probe"
  echo "SKIP: this nvcc cannot target sm_90a (needs CUDA 12.0 or later)"; exit 0
fi
rm -f "$probe"
need_gtest=1
. "$root/nvidia/tests/e2e/cutlass_fetch.sh"
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_cutlass_gemm.XXXXXX")"
trap 'rm -rf "$work"' EXIT

# test file | arch | simulated GPU | gtest filter
tests=(
  "gemm_f16t_f16n_f32t_tensor_op_f32_sparse_sm80|compute_80|nvidia/a100|*"
  "sm90_gemm_f16_f16_f16_tensor_op_f32_group_gemm|compute_90a|nvidia/h100|*.128x128x64_2x2x1:*.128x128x64_2x2x1_silu"
  "sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized_pingpong|compute_90a|nvidia/h100|*f16t_f16t_f32n_tensor_op_gmma_f32_persistent.64x128x64_2x4x1:*f16t_f16n_f16t_tensor_op_gmma_f32_persistent_Epilogue.64x128x64_2x2x1"
  "sm90_gemm_s8_s8_s8_tensor_op_s32|compute_90a|nvidia/h100|*.128x128x128:*cooperative_epilogue*"
)
u="$cutlass/test/unit"
# Each compile peaks near 10 GB, as run_cutlass_hopper.sh measured; as many
# run together as the free memory holds, and never fewer than one.
per_compile_kb=$((${VGPU_NVCC_COMPILE_MB:-10240} * 1024))
avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
jobs=$(( avail_kb / per_compile_kb ))
(( jobs < 1 )) && jobs=1
(( jobs > ${#tests[@]} )) && jobs=${#tests[@]}
compile() {
  local name="$1" arch="$2"
  nvcc -std=c++17 -O1 -cudart shared -arch="$arch" -code="$arch" --expt-relaxed-constexpr \
       -DCUTLASS_TARGET_NAME="\"$name\"" \
       -I "$cutlass/include" -I "$cutlass/tools/util/include" -I "$u/common" -I "$u" -I "$cutlass/test" \
       -I "$gtest/googletest/include" "$u/gemm/device/$name.cu" "$u/test_unit.cpp" \
       "$u/common/filter_architecture.cpp" "$gtest/libgtest.a" -o "$work/$name" >"$work/$name.log" 2>&1
}
for ((first = 0; first < ${#tests[@]}; first += jobs)); do
  pids=()
  names=()
  for entry in "${tests[@]:first:jobs}"; do
    IFS='|' read -r name arch _ _ <<<"$entry"
    compile "$name" "$arch" & pids+=($!)
    names+=("$name")
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "FAIL: ${names[$i]} did not compile"; tail -20 "$work/${names[$i]}.log"; exit 1
    fi
  done
done
fail=0
for entry in "${tests[@]}"; do
  IFS='|' read -r name _ gpu filter <<<"$entry"
  if ! require_shim_libs "$shim" "$work/$name"; then exit 0; fi
  # Four host threads whatever the machine has, so blocks that wait on each
  # other (the split-K semaphore) really do run at the same time.
  result="$(VGPU_QUIET=1 VGPU_THREADS=4 VGPU_GPU="$gpu" LD_LIBRARY_PATH="$shim" \
            "$work/$name" --gtest_filter="$filter" 2>&1 || true)"
  summary="$(grep -E '^\[  (PASSED|FAILED)  \]' <<<"$result" | head -2 | tr '\n' ' ')"
  echo "$name: $summary"
  if ! grep -q '^\[  PASSED  \]' <<<"$result" || grep -q '^\[  FAILED  \]' <<<"$result"; then
    grep -E 'FAILED|Failure|VirtualGPU|timed out' <<<"$result" | head -20
    fail=1
  fi
done
exit $fail
