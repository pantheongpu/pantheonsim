#!/usr/bin/env bash
# PyTorch's ROCm build, unmodified, on a simulated MI300X (amd/tests/pytorch/check.py).
#
# The Python it runs is one with PyTorch for ROCm installed: VGPU_TORCH_PYTHON,
# or a venv under ~/.local/share/torch-rocm*. PyTorch loads the libamdhip64
# beside its own libraries, so the installation is left as it is and a tree of
# links to it is made in a temporary directory, the same in every file but
# that one, which is VirtualGPU's. Where no such Python is found, it skips.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim/libamdhip64.so.7"
[[ -e "$shim" ]] || { echo "SKIP: no HIP shim in $build/shim"; exit 0; }
python=""
for c in "${VGPU_TORCH_PYTHON:-}" $(ls -d "$HOME"/.local/share/torch-rocm*/bin/python 2>/dev/null); do
  [[ -n "$c" && -x "$c" ]] && "$c" -c 'import torch, sys; sys.exit(0 if torch.version.hip else 1)' 2>/dev/null &&
    { python=$c; break; }
done
[[ -n "$python" ]] || { echo "SKIP: no Python with PyTorch for ROCm (set VGPU_TORCH_PYTHON)"; exit 0; }
package=$("$python" -c 'import os, torch; print(os.path.dirname(torch.__file__))')
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/torch/lib"
for e in "$package"/*; do [[ "$(basename "$e")" == lib ]] || ln -s "$e" "$tmp/torch/"; done
for f in "$package"/lib/*; do [[ "$(basename "$f")" == libamdhip64.so ]] || ln -s "$f" "$tmp/torch/lib/"; done
ln -s "$(readlink -f "$shim")" "$tmp/torch/lib/libamdhip64.so"

# PyTorch on a simulated device can grow to many gigabytes of host memory,
# and the machines this runs on are shared: the simulated device keeps at
# most VGPU_MEMORY_RAM_MB of its memory in RAM (the rest spills to disk), and
# the whole run is held to VGPU_TORCH_MEMORY_MAX where systemd can hold it, so
# that it, and not everything else, is what stops if it grows past that.
# An uncapped run took a shared 23 GB machine down on 2026-09-25.
cap=()
if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
  cap=(systemd-run --user --scope -q -p "MemoryMax=${VGPU_TORCH_MEMORY_MAX:-10G}" -p MemorySwapMax=0)
fi
out=$(cd "$tmp" && VGPU_QUIET=1 VGPU_GPU=amd/mi300x VGPU_MEMORY_RAM_MB="${VGPU_MEMORY_RAM_MB:-4096}" \
  PYTHONPATH="$tmp" "${cap[@]}" "$python" "$root/amd/tests/pytorch/check.py" 2>&1)
status=$?
echo "$out" | grep -E '^(ok|FAIL) ' | sed 's/^/      /'
fail=0
[[ $status == 0 ]] || { echo "FAIL  the checks ran to the end (exit $status)"; echo "$out" | tail -5; fail=1; }
passed=$(grep -c '^ok ' <<< "$out")
if grep -q '^FAIL ' <<< "$out" || [[ $passed != 20 ]]; then
  echo "FAIL  every PyTorch check matches the CPU: $passed of 20"; fail=1
else
  echo "ok    every PyTorch check matches the CPU: 20 of 20"
fi
exit $fail
