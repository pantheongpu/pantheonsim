#!/usr/bin/env bash
# graph_capture_nccl.cu: NCCL's collectives and point-to-point operations recorded into captured CUDA graphs,
# two ranks in one process, one per device.
#
#   run_graph_capture_nccl.sh            against VirtualGPU's libnccl on two virtual devices
#   run_graph_capture_nccl.sh --card     against NVIDIA's libnccl on two real GPUs: the check that the program's
#                                        expectations are what NVIDIA's NCCL does. SKIPs without two GPUs, nvcc,
#                                        or the library (the pip wheel nvidia-nccl-cu13, or NCCL_LIB_DIR).
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/graph_capture_nccl.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_graph_capture_nccl_$$"
command -v nvcc >/dev/null 2>&1 || { echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0; }
mkdir -p "$out"
trap 'rm -rf "$out"' EXIT
inc="$root/nvidia/third_party/nccl_include"
if [[ "${1:-}" == --card ]]; then
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  blas_lib="$cuda_root/lib64"
  [[ -d "$blas_lib" ]] || blas_lib="$cuda_root/targets/x86_64-linux/lib"
  nccl_lib="${NCCL_LIB_DIR:-}"
  if [[ -z "$nccl_lib" ]]; then
    for d in "$HOME"/.local/share/torch-cu13*/lib/python3*/site-packages/nvidia/nccl/lib /usr/lib/x86_64-linux-gnu; do
      compgen -G "$d/libnccl.so.2*" >/dev/null && nccl_lib="$d" && break
    done
  fi
  [[ -n "$nccl_lib" ]] || { echo "SKIP: no NVIDIA NCCL library (set NCCL_LIB_DIR)"; exit 0; }
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets -I"$inc" "$src" -o "$out/prog" \
       -L"$blas_lib" -L"$nccl_lib" -l:libnccl.so.2
  # NVIDIA's NCCL maps more address space than a build wants to be held to (a limit set with `ulimit -S -v` for the
  # compile above, as the build lock does, makes ncclCommInitRank fail with "out of memory").
  result="$(ulimit -S -v unlimited 2>/dev/null || true; LD_LIBRARY_PATH="$nccl_lib:$blas_lib" "$out/prog" 2>&1 || true)"
else
  [[ -e "$shim/libnccl.so.2" ]] || { echo "SKIP: libvgpunccl not built"; exit 0; }
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") \
       -I"$inc" "$src" -L"$shim" -lnccl -o "$out/prog"
  require_shim_libs "$shim" "$out/prog" || exit 0
  result="$(env VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_DEVICE_COUNT=2 VGPU_NCCL_RANKS=2 VGPU_NCCL_DIR="$out/rendezvous" \
            VGPU_NCCL_TIMEOUT=120 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim") "$out/prog" 2>&1 || true)"
fi
echo "$result"
[[ "$(tail -1 <<< "$result")" == SKIP:* ]] && exit 0
[[ "$(tail -1 <<< "$result")" == "PASS" ]]
