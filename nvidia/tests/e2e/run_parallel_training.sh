#!/usr/bin/env bash
# A network trained across several processes, one simulated GPU each, split four
# ways -- data parallel, tensor parallel, pipeline parallel, sharded
# (all-gather + reduce-scatter) -- against a single process on the whole batch
# (nvidia/tests/e2e/parallel_training.cu). cuBLAS does the matrix products and
# NCCL's file-backed transport carries the collectives between the processes.
#
#   nvidia/tests/e2e/run_parallel_training.sh [nranks]     (default 4)
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
nranks="${1:-4}"
out="${TMPDIR:-/tmp}/vgpu-parallel-training.$$"

command -v nvcc >/dev/null || { echo "SKIP: nvcc not found"; exit 0; }
[[ -e "$shim/libnccl.so.2" ]] || { echo "SKIP: libvgpunccl not built"; exit 0; }
[[ -e "$shim/libcublas.so.13" || -e "$shim/libcublas.so.12" ]] || { echo "SKIP: no cuBLAS shim in $shim"; exit 0; }

mkdir -p "$out"
trap 'rm -rf "$out"' EXIT
nvcc -std=c++14 -arch=sm_86 -Wno-deprecated-gpu-targets -cudart shared \
     $(shim_sanitizer_nvcc_flags "$shim") \
     -I"$root/nvidia/third_party/nccl_include" "$root/nvidia/tests/e2e/parallel_training.cu" \
     -L"$shim" -lnccl -lcublas -o "$out/pt" || { echo "FAIL: compile"; exit 1; }
# A program that needs a library of another CUDA major than the shim's cannot run.
require_shim_libs "$shim" "$out/pt" || exit 0

pids=()
for ((r = 0; r < nranks; ++r)); do
  VGPU_QUIET=1 VGPU_GPU=nvidia/a10 VGPU_VRAM_MB=256 \
  VGPU_NCCL_DIR="$out/rendezvous" VGPU_NCCL_TIMEOUT=120 \
  LD_LIBRARY_PATH="$shim" "$out/pt" "$r" "$nranks" "$out/id" > "$out/rank$r.log" 2>&1 &
  pids+=($!)
done

fail=0
for r in "${!pids[@]}"; do
  wait "${pids[$r]}" || { echo "FAIL  rank $r exited non-zero"; fail=1; }
done
cat "$out"/rank*.log | grep -E '^(ok|FAIL) ' | sed 's/^/      /'
# Every rank's own FAIL line, and anything else it said on stderr.
if cat "$out"/rank*.log | grep -vE '^(ok|FAIL) ' | grep -q .; then
  echo "FAIL  unexpected output:"; cat "$out"/rank*.log | grep -vE '^(ok|FAIL) ' | head -10 | sed 's/^/      /'; fail=1
fi
want=4; [[ $nranks == 2 ]] && want=5   # the loss line, then each mode (pipeline parallel only at two ranks)
got=$(cat "$out"/rank*.log | grep -c '^ok ')
if [[ $got != $want ]]; then echo "FAIL  expected $want ok lines, saw $got"; fail=1; fi
[[ $fail == 0 ]] && echo "$nranks ranks, $nranks processes: data, tensor, pipeline and sharded parallel training agree with one process on the whole batch"
exit $fail
