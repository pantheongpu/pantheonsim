#!/usr/bin/env bash
# torch.distributed across two processes (nvidia/tests/pytorch/distributed.py),
# each with its own simulated GPU, launched as torchrun launches them: the
# collectives, then a small network trained with DistributedDataParallel,
# tensor parallelism, pipeline parallelism and FullyShardedDataParallel, each
# against one process training on the whole batch. PyTorch's NCCL backend runs
# on VirtualGPU's libnccl, whose ranks meet through its file-backed transport.
#
#   run_pytorch_distributed.sh [gpu]      (nvidia/rtx5090 by default)
#
# Set up as run_pytorch.sh sets up PyTorch's CUDA build: through `vgpu run
# --preload`, in a Python with PyTorch for CUDA 13 (VGPU_TORCH_CUDA_PYTHON, or a
# venv under ~/.local/share/torch-cu13*), and skipped where there is none. Each
# process is held to its own memory cap.
set -uo pipefail
gpu="${1:-nvidia/rtx5090}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
world=2
checks=11   # per rank, in distributed.py
[[ -e "$shim/libcudart.so.13" ]] || { echo "SKIP: no CUDA 13 runtime shim in $shim"; exit 0; }
[[ -e "$shim/libnccl.so.2" ]] || { echo "SKIP: no NCCL shim in $shim"; exit 0; }
python=""
for c in "${VGPU_TORCH_CUDA_PYTHON:-}" $(ls -d "$HOME"/.local/share/torch-cu13*/bin/python 2>/dev/null); do
  [[ -n "$c" && -x "$c" ]] && "$c" -c 'import torch, sys; sys.exit(0 if (torch.version.cuda or "").startswith("13") else 1)' 2>/dev/null &&
    { python=$c; break; }
done
[[ -n "$python" ]] || { echo "SKIP: no Python with PyTorch for CUDA 13 (set VGPU_TORCH_CUDA_PYTHON)"; exit 0; }
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "SKIP: $vgpu not built"; exit 0; }

cap=()
if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
  cap=(systemd-run --user --scope -q -p "MemoryMax=${VGPU_TORCH_MEMORY_MAX:-6G}" -p MemorySwapMax=0)
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
port=$((29500 + RANDOM % 1000))
for ((r = 0; r < world; ++r)); do
  (cd "$tmp" && TRITON_CACHE_DIR="$tmp/triton" TORCHINDUCTOR_CACHE_DIR="$tmp/inductor" \
     MASTER_ADDR=127.0.0.1 MASTER_PORT=$port RANK=$r WORLD_SIZE=$world NCCL_DEBUG=ERROR \
     VGPU_NCCL_DIR="$tmp/rendezvous" VGPU_NCCL_TIMEOUT=300 \
     timeout 1200 "${cap[@]}" "$vgpu" run --gpu "$gpu" --count "$world" --preload "$python" \
       "$root/nvidia/tests/pytorch/distributed.py" > "$tmp/rank$r.log" 2>&1
   echo "exit $?" >> "$tmp/rank$r.log") &
done
wait
out=$(cat "$tmp"/rank*.log)
echo "$out" | grep -E '^(ok|FAIL) ' | sed 's/^/      /'
fail=0
# The simulator's own errors are left on: a kernel it could not run or a library
# call it refused is reported as "VirtualGPU error [...]", and PyTorch may carry
# on past it with whatever was in the output buffer, so a check can still print
# "ok". Any such line fails the test, as in run_pytorch.sh.
if grep -q 'VirtualGPU error \[' <<< "$out"; then
  echo "FAIL  the simulator ran every kernel it was given"; grep -m5 'VirtualGPU error \[' <<< "$out" | sed 's/^/      /'; fail=1
fi
for ((r = 0; r < world; ++r)); do
  st=$(grep -o '^exit [0-9]*' "$tmp/rank$r.log" | tail -1)
  [[ "$st" == "exit 0" ]] || { echo "FAIL  rank $r ran to the end ($st)"; tail -5 "$tmp/rank$r.log" | sed 's/^/      /'; fail=1; }
done
passed=$(grep -c '^ok ' <<< "$out")
want=$((world * checks))
if grep -q '^FAIL ' <<< "$out" || [[ $passed != "$want" ]]; then
  echo "FAIL  torch.distributed across $world processes on $gpu: $passed of $want"; fail=1
else
  echo "ok    torch.distributed across $world processes on $gpu: $want of $want"
fi
exit $fail
