#!/usr/bin/env bash
# hipSPARSELt's structured-sparse GEMMs on a simulated GPU (hipsparselt/spmm):
# the library and its kernels from PyTorch's ROCm wheel, HIP the simulator's.
# Its kernels prune, compress and multiply with the sparse matrix
# instructions; the program checks each product against the host's.
#
#   amd/tests/e2e/run_hipsparselt.sh [gpu]      (amd/mi300x by default)
#
# Skips where there is no such wheel (VGPU_TORCH_PYTHON names its Python).
set -uo pipefail
gpu="${1:-amd/mi300x}"
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim/libamdhip64.so.7"
[[ -e "$shim" ]] || { echo "SKIP: no HIP shim in $build/shim"; exit 0; }
lib=""
for c in "${VGPU_TORCH_PYTHON:-}" $(ls -d "$HOME"/.local/share/torch-rocm*/bin/python 2>/dev/null); do
  [[ -n "$c" && -x "$c" ]] || continue
  d=$("$c" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)
  [[ -n "$d" && -e "$d/libhipsparselt.so" && -d "$d/hipsparselt/library" ]] && { lib=$d; break; }
done
[[ -n "$lib" ]] || { echo "SKIP: no PyTorch for ROCm wheel with hipSPARSELt (set VGPU_TORCH_PYTHON)"; exit 0; }

# The wheel's libraries, but HIP the simulator's. hipSPARSELt finds its
# kernels beside itself, in hipsparselt/library.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for f in "$lib"/*; do [[ "$(basename "$f")" == libamdhip64.so ]] || ln -s "$f" "$tmp/"; done
ln -s "$lib/libhipsparselt.so" "$tmp/libhipsparselt.so.0"
ln -s "$(readlink -f "$shim")" "$tmp/libamdhip64.so"
ln -s "$(readlink -f "$shim")" "$tmp/libamdhip64.so.7"

cap=()
if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
  cap=(systemd-run --user --scope -q -p "MemoryMax=${VGPU_TORCH_MEMORY_MAX:-10G}" -p MemorySwapMax=0)
fi
out=$(VGPU_QUIET=1 VGPU_GPU="$gpu" LD_LIBRARY_PATH="$tmp" "${cap[@]}" "$root/amd/tests/hipsparselt/spmm" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
fail=0
[[ $status == 0 ]] || { echo "FAIL  the hipSPARSELt program ran to the end (exit $status)"; fail=1; }
right=$(grep -c ': pruned 2:4, 0 runs of four not 2:4, 0 of [0-9]* wrong$' <<< "$out")
if [[ $right == 6 ]]; then echo "ok    each hipSPARSELt sparse GEMM is the host's product on $gpu: 6 of 6"
else echo "FAIL  each hipSPARSELt sparse GEMM is the host's product on $gpu: $right of 6"; fail=1; fi
exit $fail
