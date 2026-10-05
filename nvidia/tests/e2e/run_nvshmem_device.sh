#!/usr/bin/env bash
# NVSHMEM's device API on the simulator: nvshmem_device.cu, built with
# NVIDIA's NVSHMEM headers and device library (-rdc, libnvshmem_device.a)
# against VirtualGPU's libnvshmem_host, run as a job of PEs.
#
# NVSHMEM is not part of the CUDA toolkit. The test looks for it in
# NVSHMEM_HOME (include/nvshmem.h, lib/libnvshmem_device.a), then in the pip
# package nvidia-nvshmem-cu13 (or -cu12), and skips without it. Its device
# library must match nvcc's CUDA major.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found"; exit 0
fi
if [[ ! -e "$shim/libnvshmem_host.so.3" ]]; then
  echo "SKIP: the NVSHMEM shim is not built"; exit 0
fi
nvs="${NVSHMEM_HOME:-}"
if [[ -z "$nvs" ]]; then
  nvs="$(python3 -c 'import os, nvidia.nvshmem as m; print(os.path.dirname(m.__file__))' 2>/dev/null || true)"
  [[ -n "$nvs" && -d "$nvs" ]] || nvs="$(python3 -c 'import os, nvidia; print(os.path.join(list(nvidia.__path__)[0], "nvshmem"))' 2>/dev/null || true)"
fi
if [[ -z "$nvs" || ! -e "$nvs/include/nvshmem.h" || ! -e "$nvs/lib/libnvshmem_device.a" ]]; then
  echo "SKIP: NVIDIA's NVSHMEM headers and device library are not installed (set NVSHMEM_HOME)"; exit 0
fi
out="${TMPDIR:-/tmp}/vgpu_e2e_nvshmem_device_$$"
log="$out.log"
trap 'rm -f "$out" "$log"' EXIT
if ! nvcc -std=c++17 -rdc=true -cudart shared -arch=sm_80 -Wno-deprecated-gpu-targets \
       -I"$nvs/include" -I"$root/nvidia/src" "$root/nvidia/tests/e2e/nvshmem_device.cu" -o "$out" \
       "$nvs/lib/libnvshmem_device.a" -L"$shim" -lnvshmem_host >"$log" 2>&1; then
  cat "$log"
  echo "SKIP: NVSHMEM's device library at $nvs does not link with this nvcc (a different CUDA major)"; exit 0
fi
require_shim_libs "$shim" "$out" || exit 0
status=0
result="$(VGPU_GPU=nvidia/a100 VGPU_DEVICE_COUNT=4 LD_LIBRARY_PATH="$shim" "$out" 3 2>&1)" || status=$?
echo "$result"
if grep -qE 'VirtualGPU error \[|is not implemented by VirtualGPU' <<< "$result"; then
  echo "FAIL: a kernel was refused or a stub reached"; exit 1
fi
[[ $status == 0 && "$(tail -1 <<< "$result")" == PASS* ]]
