#!/usr/bin/env bash
# The PyTorch sweep (nvidia/tests/pytorch/sweep.py) on a simulated NVIDIA GPU.
#
#   run_pytorch_sweep.sh <quick|full> [gpu]        (nvidia/rtx5090 by default)
#
# Each group of checks runs in its own `vgpu run --preload` process, so a hang or
# a crash costs one group, not the sweep; the checks print "ok <name>" or
# "FAIL <name>: <why>", each against the CPU. The number of ok lines must be the
# number of checks the sweep lists for this tier (a check that silently stopped
# running is a failure), and "VirtualGPU error [" in any output -- a kernel the
# simulator could not run, which PyTorch may carry on past -- fails the run.
#
# Optional packages (torchvision, torchaudio, transformers, ...) are used where the
# Python has them, or where VGPU_SWEEP_DEPS names directories holding them; with
# VGPU_SWEEP_PIP=1 the missing ones are fetched into a cache first (the nightly
# does that). Checks that need one that is absent are skipped and said so.
set -uo pipefail
tier="${1:-quick}" gpu="${2:-nvidia/rtx5090}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
[[ -e "$shim/libcudart.so.13" ]] || { echo "SKIP: no CUDA 13 runtime shim in $shim"; exit 0; }
python=""
for c in "${VGPU_TORCH_CUDA_PYTHON:-}" $(ls -d "$HOME"/.local/share/torch-cu13*/bin/python 2>/dev/null); do
  [[ -n "$c" && -x "$c" ]] && "$c" -c 'import torch, sys; sys.exit(0 if (torch.version.cuda or "").startswith("13") else 1)' 2>/dev/null &&
    { python=$c; break; }
done
[[ -n "$python" ]] || { echo "SKIP: no Python with PyTorch for CUDA 13 (set VGPU_TORCH_CUDA_PYTHON)"; exit 0; }
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "SKIP: $vgpu not built"; exit 0; }
sweep="$root/nvidia/tests/pytorch/sweep.py"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

deps="${VGPU_SWEEP_DEPS:-}"
if [[ "${VGPU_SWEEP_PIP:-0}" == 1 ]]; then
  cache="${VGPU_SWEEP_PIP_CACHE:-$HOME/.cache/vgpu-sweep-deps}"
  mkdir -p "$cache"
  tv=$("$python" -c 'import torch; print(torch.__version__.split("+")[0])')
  # torchvision/torchaudio are pinned to the torch (0.<minor+15>.x and the same version); a
  # failed install only skips the checks that need them.
  "$python" - "$cache" "$tv" <<'PY' || true
import importlib.util, subprocess, sys, os
cache, tv = sys.argv[1:3]
sys.path.insert(0, os.path.join(cache, 'hf')); sys.path.insert(0, os.path.join(cache, 'tv'))
def have(m): return importlib.util.find_spec(m) is not None
def pip(target, *args):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--target', os.path.join(cache, target), *args], check=False)
if not (have('torchvision') and have('torchaudio')):
    major, minor, _ = tv.split('.')
    vision = f"0.{int(minor) + 15}.0"
    pip('tv', '--no-deps', f'torchvision=={vision}', f'torchaudio=={tv}', '--index-url', 'https://download.pytorch.org/whl/cu130')
if not have('transformers') or not have('PIL'):
    pip('hf', 'transformers', 'pillow')
    for junk in ('numpy', 'numpy.libs', 'bin'):
        subprocess.run(['rm', '-rf', os.path.join(cache, 'hf', junk)])
PY
  deps="$cache/tv:$cache/hf${deps:+:$deps}"
fi
export VGPU_SWEEP_DEPS="$deps" VGPU_SWEEP_TIER="$tier"

# What the sweep lists for this tier, by group: the expected ok lines.
mapfile -t listing < <("$python" "$sweep" --list 2>/dev/null)
[[ ${#listing[@]} -gt 0 ]] || { echo "FAIL  the sweep lists no checks"; exit 1; }
groups=$(printf '%s\n' "${listing[@]}" | cut -f1 | awk '!seen[$0]++')

cap=()
if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
  cap=(systemd-run --user --scope -q -p "MemoryMax=${VGPU_TORCH_MEMORY_MAX:-10G}" -p MemorySwapMax=0)
fi
fail=0 total=0 passed=0
for g in $groups; do
  want=$(printf '%s\n' "${listing[@]}" | cut -f1 | grep -cx "$g")
  count=1
  [[ $g == multi ]] && count=2
  start=$SECONDS
  out=$(cd "$tmp" && VGPU_SWEEP_GROUPS=$g TRITON_CACHE_DIR="$tmp/triton" TORCHINDUCTOR_CACHE_DIR="$tmp/inductor" \
    timeout "${VGPU_SWEEP_GROUP_TIMEOUT:-1500}" "${cap[@]}" "$vgpu" run --gpu "$gpu" --count "$count" --preload "$python" "$sweep" 2>&1)
  status=$?
  secs=$((SECONDS - start))
  echo "$out" | grep -E '^(ok|FAIL|skip) ' | sed "s/^/      [$g] /"
  ok=$(grep -c '^ok ' <<< "$out")
  total=$((total + want)); passed=$((passed + ok))
  if grep -q '^FAIL ' <<< "$out" || [[ $ok != "$want" ]]; then
    echo "FAIL  group $g: $ok of $want checks match the CPU on $gpu (${secs}s)"; fail=1
  else
    echo "ok    group $g: $ok of $want checks match the CPU on $gpu (${secs}s)"
  fi
  if grep -q 'VirtualGPU error \[' <<< "$out"; then
    echo "FAIL  group $g: the simulator ran every kernel it was given"; grep -m5 'VirtualGPU error \[' <<< "$out" | sed 's/^/      /'; fail=1
  fi
  [[ $status == 0 ]] || { echo "FAIL  group $g: the checks ran to the end (exit $status)"; echo "$out" | tail -5 | sed 's/^/      /'; fail=1; }
done
echo "      $passed of $total checks (tier $tier)"
exit $fail
