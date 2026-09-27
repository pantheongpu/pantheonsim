#!/usr/bin/env bash
# PyTorch's CUDA build, unmodified, on a simulated NVIDIA GPU.
#
#   run_pytorch.sh <checks.py> <count> [gpu]      (nvidia/rtx5090 by default)
#
# <checks.py> prints "ok <name>" or "FAIL <name>: <why>" per check; <count> is
# how many it has. The Python it runs is one with PyTorch's CUDA 13 build
# installed: VGPU_TORCH_CUDA_PYTHON, or a venv under ~/.local/share/torch-cu13*.
# Where there is none, it skips.
#
# The GPU is an sm_120 one because that is what PyTorch's wheels carry PTX for
# (compute_120, beside SASS for older architectures); the simulator runs PTX,
# and a driver JITs PTX only for a device at least as new.
#
# It runs as a user would, through `vgpu run --preload`: PyTorch finds NVIDIA's
# libraries through an RPATH into its nvidia-* packages, which LD_LIBRARY_PATH
# does not override, and python names no CUDA library itself, so every CUDA
# library the shim carries is preloaded -- a library already loaded under a
# soname is the one every later request for that soname gets. NVIDIA's
# cuFile, cuSPARSELt and NVSHMEM load as they are; nothing here calls them.
set -uo pipefail
script="$1" expected="$2" gpu="${3:-nvidia/rtx5090}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
[[ -e "$shim/libcudart.so.13" ]] || { echo "SKIP: no CUDA 13 runtime shim in $shim"; exit 0; }
[[ -e "$script" ]] || { echo "FAIL  no such script of checks: $script"; exit 1; }
python=""
for c in "${VGPU_TORCH_CUDA_PYTHON:-}" $(ls -d "$HOME"/.local/share/torch-cu13*/bin/python 2>/dev/null); do
  [[ -n "$c" && -x "$c" ]] && "$c" -c 'import torch, sys; sys.exit(0 if (torch.version.cuda or "").startswith("13") else 1)' 2>/dev/null &&
    { python=$c; break; }
done
[[ -n "$python" ]] || { echo "SKIP: no Python with PyTorch for CUDA 13 (set VGPU_TORCH_CUDA_PYTHON)"; exit 0; }

vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "SKIP: $vgpu not built"; exit 0; }

# As on AMD: a run held to a memory cap where systemd can hold one, since
# PyTorch on a simulated device grows to gigabytes and the machines are shared.
cap=()
if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
  cap=(systemd-run --user --scope -q -p "MemoryMax=${VGPU_TORCH_MEMORY_MAX:-10G}" -p MemorySwapMax=0)
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
out=$(cd "$tmp" && TRITON_CACHE_DIR="$tmp/triton" TORCHINDUCTOR_CACHE_DIR="$tmp/inductor" \
  "${cap[@]}" "$vgpu" run --gpu "$gpu" --preload "$python" "$script" 2>&1)
status=$?
echo "$out" | grep -E '^(ok|FAIL) ' | sed 's/^/      /'
fail=0
# The simulator's errors are left on: a kernel it could not run (an
# instruction it lacks, a fault) or a library call it refused is reported as
# "VirtualGPU error [...]", and PyTorch may carry on past it with whatever was
# in the output buffer, so a check can still print "ok". Any such line fails
# the test, as on AMD (amd/tests/e2e/run_pytorch.sh).
if grep -q 'VirtualGPU error \[' <<< "$out"; then
  echo "FAIL  the simulator ran every kernel it was given"; grep -m5 'VirtualGPU error \[' <<< "$out" | sed 's/^/      /'; fail=1
fi
[[ $status == 0 ]] || { echo "FAIL  the checks ran to the end (exit $status)"; echo "$out" | tail -5; fail=1; }
passed=$(grep -c '^ok ' <<< "$out")
name=$(basename "$script")
if grep -q '^FAIL ' <<< "$out" || [[ $passed != "$expected" ]]; then
  echo "FAIL  every PyTorch check in $name matches the CPU on $gpu: $passed of $expected"; fail=1
else
  echo "ok    every PyTorch check in $name matches the CPU on $gpu: $expected of $expected"
fi
exit $fail
