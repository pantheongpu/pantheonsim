#!/usr/bin/env bash
# Conditional graph nodes (IF, IF/ELSE, WHILE, SWITCH) and capture into an
# existing graph, against the values an RTX 3060 gives (graph_conditional.cu).
# Skips if nvcc is unavailable, and when the toolkit predates CUDA 12.3.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/graph_conditional.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_graph_conditional_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
nvcc -std=c++17 -cudart shared --gpu-architecture=sm_86 -Wno-deprecated-gpu-targets \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
# Kept out of a failing command substitution, so a failure prints its reason.
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$out" 2>&1 || true)"
rm -f "$out"
echo "conditional graph nodes: $result"
last="$(tail -n 1 <<<"$result")"
[[ "$last" == SKIP:* ]] && exit 0
[[ "$last" == "PASS" ]] && ! grep -q '^FAIL' <<<"$result"
