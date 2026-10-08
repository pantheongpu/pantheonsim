#!/usr/bin/env bash
# torch.distributed's NCCL backend across two processes, each its own
# simulated NVIDIA GPU (nvidia/tests/pytorch/dist.py): the ranks start the way
# torchrun starts them and meet through NCCL's file-backed transport. Set up as
# run_pytorch.sh sets up PyTorch (a Python with PyTorch for CUDA 13, run through
# `vgpu run --preload`); a failed rank, an unfinished run or any
# "VirtualGPU error [" line fails the test.
#
#   run_pytorch_dist.sh <expected ok lines per rank> [gpu]     (nvidia/rtx5090 by default)
#   VGPU_SWEEP_TIER=full adds the checks that are not time-bounded.
set -uo pipefail
expected="$1" gpu="${2:-nvidia/rtx5090}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
[[ -e "$build/shim/libcudart.so.13" && -e "$build/shim/libnccl.so.2" ]] || { echo "SKIP: no CUDA 13 runtime or NCCL shim in $build/shim"; exit 0; }
python=""
for c in "${VGPU_TORCH_CUDA_PYTHON:-}" $(ls -d "$HOME"/.local/share/torch-cu13*/bin/python 2>/dev/null); do
  [[ -n "$c" && -x "$c" ]] && "$c" -c 'import torch, sys; sys.exit(0 if (torch.version.cuda or "").startswith("13") else 1)' 2>/dev/null &&
    { python=$c; break; }
done
[[ -n "$python" ]] || { echo "SKIP: no Python with PyTorch for CUDA 13 (set VGPU_TORCH_CUDA_PYTHON)"; exit 0; }
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "SKIP: $vgpu not built"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
port=$((29500 + RANDOM % 1000))
for r in 0 1; do
  (cd "$tmp" && MASTER_ADDR=127.0.0.1 MASTER_PORT=$port RANK=$r WORLD_SIZE=2 NCCL_DEBUG=ERROR \
     VGPU_NCCL_DIR="$tmp/rendezvous" VGPU_NCCL_TIMEOUT=300 VGPU_QUIET=1 \
     TRITON_CACHE_DIR="$tmp/triton$r" TORCHINDUCTOR_CACHE_DIR="$tmp/inductor$r" \
     timeout "${VGPU_SWEEP_RANK_TIMEOUT:-1500}" "$vgpu" run --gpu "$gpu" --count 2 --preload "$python" \
     "$root/nvidia/tests/pytorch/dist.py" > "$tmp/rank$r.log" 2>&1
   echo "exit $?" >> "$tmp/rank$r.log") &
done
wait
out=$(cat "$tmp"/rank0.log "$tmp"/rank1.log)
echo "$out" | grep -E '^(ok|FAIL) ' | sed 's/^/      /'
fail=0
if grep -q 'VirtualGPU error \[' <<< "$out"; then
  echo "FAIL  the simulator ran every kernel it was given"; grep -m5 'VirtualGPU error \[' <<< "$out" | sed 's/^/      /'; fail=1
fi
for r in 0 1; do
  st=$(grep -o '^exit [0-9]*' "$tmp/rank$r.log" | tail -1)
  [[ "$st" == "exit 0" ]] || { echo "FAIL  rank $r ran to the end ($st)"; tail -8 "$tmp/rank$r.log" | sed 's/^/      /'; fail=1; }
done
passed=$(grep -c '^ok ' <<< "$out")
if grep -q '^FAIL ' <<< "$out" || [[ $passed != $((2 * expected)) ]]; then
  echo "FAIL  torch.distributed across two processes: $passed ok lines of $((2 * expected))"; fail=1
else
  echo "ok    torch.distributed across two processes on $gpu: $expected checks on each rank"
fi
exit $fail
