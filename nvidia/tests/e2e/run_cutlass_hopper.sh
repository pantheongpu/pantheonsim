#!/usr/bin/env bash
# CUTLASS's own Hopper unit tests for TMA and bulk copies, unmodified, on a
# simulated H100: tensor loads and stores in one to five dimensions, every
# swizzle mode, boxes that cross the tensor's edge, internal-type conversion,
# and the plain bulk copies. They are NVIDIA's tests of the same instructions
# on real hardware, so passing them is agreement with the hardware's layouts
# rather than with this simulator's reading of the ISA.
#
# Two are left out, each for a stated reason:
#   *Tma_Load_1D (in both files): its testbed copies a 256-element tile into a
#     128-element buffer. Hardware does not notice, because the write stays in
#     the allocator's padding; the simulator reports it as the out-of-bounds
#     access it is, as compute-sanitizer would.
#   tma_mcast_load: multicast into other blocks of a cluster needs distributed
#     shared memory, which is not implemented and is refused by name.
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
# googletest and CUTLASS's test harness are built here without a sanitizer, and
# a program linked against an instrumented shim needs one; rather than build
# googletest three ways, this runs in the plain build only.
if [[ -n "$(shim_sanitizer "$shim")" ]]; then
  echo "SKIP: CUTLASS's unit tests run against the uninstrumented build"; exit 0
fi
probe="${TMPDIR:-/tmp}/vgpu_cutlass_probe_$$.cu"
echo '__global__ void k() { asm volatile("wgmma.fence.sync.aligned;"); }' > "$probe"
if ! nvcc -arch=compute_90a -code=compute_90a -c "$probe" -o /dev/null 2>/dev/null; then
  rm -f "$probe"
  echo "SKIP: this nvcc cannot target sm_90a (needs CUDA 12.0 or later)"; exit 0
fi
rm -f "$probe"
need_gtest=1
. "$root/nvidia/tests/e2e/cutlass_fetch.sh"
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_cutlass_hopper.XXXXXX")"
trap 'rm -rf "$work"' EXIT
tests=(tma_load tma_store bulk_load bulk_store)
u="$cutlass/test/unit"
# Each compile instantiates most of CuTe and peaks near 10 GB (9.6 GB measured
# for tma_load with CUDA 13). All four at once took a 16 GB GitHub runner past
# its memory, and the runner was shut down in the middle of the job ("The runner
# has received a shutdown signal"), three runs out of three. So as many run
# together as the free memory holds, and never fewer than one.
per_compile_kb=$((${VGPU_NVCC_COMPILE_MB:-10240} * 1024))
avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
jobs=$(( avail_kb / per_compile_kb ))
(( jobs < 1 )) && jobs=1
(( jobs > ${#tests[@]} )) && jobs=${#tests[@]}
compile() {
  nvcc -std=c++17 -O1 -cudart shared -arch=compute_90a -code=compute_90a --expt-relaxed-constexpr \
       -I "$cutlass/include" -I "$cutlass/tools/util/include" -I "$u/common" -I "$u" \
       -I "$gtest/googletest/include" "$u/cute/hopper/$1.cu" "$u/test_unit.cpp" \
       "$u/common/filter_architecture.cpp" "$gtest/libgtest.a" -o "$work/$1" >"$work/$1.log" 2>&1
}
for ((first = 0; first < ${#tests[@]}; first += jobs)); do
  pids=()
  names=("${tests[@]:first:jobs}")
  for t in "${names[@]}"; do compile "$t" & pids+=($!); done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "FAIL: ${names[$i]} did not compile"; tail -20 "$work/${names[$i]}.log"; exit 1
    fi
  done
done
fail=0
for t in "${tests[@]}"; do
  if ! require_shim_libs "$shim" "$work/$t"; then exit 0; fi
  # Kept out of a failing command substitution, so a failure prints its reason.
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/h100 LD_LIBRARY_PATH="$shim" \
            "$work/$t" --gtest_filter='-*Tma_Load_1D' 2>&1 || true)"
  summary="$(grep -E '^\[  (PASSED|FAILED)  \]' <<<"$result" | head -2 | tr '\n' ' ')"
  echo "$t: $summary"
  if ! grep -q '^\[  PASSED  \]' <<<"$result" || grep -q '^\[  FAILED  \]' <<<"$result"; then
    grep -E 'FAILED|Failure|VirtualGPU' <<<"$result" | head -20
    fail=1
  fi
done
exit $fail
