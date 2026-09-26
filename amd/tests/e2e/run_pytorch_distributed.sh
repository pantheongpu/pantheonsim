#!/usr/bin/env bash
# torch.distributed across two processes (amd/tests/pytorch/distributed.py),
# each with its own simulated MI300X, launched as torchrun launches them:
# RCCL's collectives and DistributedDataParallel, device memory reaching
# across the processes through HIP's IPC handles. Set up as run_pytorch.sh
# sets up PyTorch; each process is held to its own memory cap.
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
cap=()
if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
  cap=(systemd-run --user --scope -q -p "MemoryMax=${VGPU_TORCH_MEMORY_MAX:-5G}" -p MemorySwapMax=0)
fi
port=$((29500 + RANDOM % 1000))
for r in 0 1; do
  (cd "$tmp" && VGPU_QUIET=1 VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 VGPU_MEMORY_RAM_MB="${VGPU_MEMORY_RAM_MB:-2048}" \
     PYTHONPATH="$tmp" MASTER_ADDR=127.0.0.1 MASTER_PORT=$port RANK=$r WORLD_SIZE=2 NCCL_DEBUG=ERROR \
     timeout 900 "${cap[@]}" "$python" "$root/amd/tests/pytorch/distributed.py" > "$tmp/rank$r.log" 2>&1
   echo "exit $?" >> "$tmp/rank$r.log") &
done
wait
out=$(cat "$tmp"/rank0.log "$tmp"/rank1.log)
echo "$out" | grep -E '^(ok|FAIL) ' | sed 's/^/      /'
fail=0
for r in 0 1; do
  st=$(grep -o '^exit [0-9]*' "$tmp/rank$r.log" | tail -1)
  [[ "$st" == "exit 0" ]] || { echo "FAIL  rank $r ran to the end ($st)"; tail -5 "$tmp/rank$r.log" | sed 's/^/      /'; fail=1; }
done
passed=$(grep -c '^ok ' <<< "$out")
if grep -q '^FAIL ' <<< "$out" || [[ $passed != 8 ]]; then
  echo "FAIL  torch.distributed across two processes: $passed of 8"; fail=1
else
  echo "ok    torch.distributed across two processes: 8 of 8"
fi
exit $fail
