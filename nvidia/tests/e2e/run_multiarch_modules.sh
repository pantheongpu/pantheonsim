#!/usr/bin/env bash
# One template kernel registered by two translation units built for different architectures (the shape of
# StringZilla's CUDA libraries: a unit for sm_80 and sm_90 and another for sm_90 alone, both instantiating the same
# kernels). multiarch_modules/main.cu checks, on an sm_80, an sm_89 and an sm_90 GPU and with either link order,
# that the module the GPU has code for runs the kernel, cudaGetFuncBySymbol agrees, and a kernel that only the
# newer unit holds is cudaErrorNoKernelImageForDevice on the older GPUs.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/multiarch_modules"
out="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-multiarch.XXXXXX")"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -rf "$out"' EXIT
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
n() { "$nvcc_bin" -std=c++17 -w -Wno-deprecated-gpu-targets -cudart shared "${san_flags[@]}" "$@"; }
n -c -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 \
  "$src/base.cu" -o "$out/base.o" &&
n -c -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 "$src/hopper.cu" -o "$out/hopper.o" &&
n -c "$src/main.cu" -o "$out/main.o" -gencode arch=compute_80,code=sm_80 &&
n "$out/main.o" "$out/base.o" "$out/hopper.o" -o "$out/hopper_last" &&
n "$out/main.o" "$out/hopper.o" "$out/base.o" -o "$out/hopper_first" ||
  { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out/hopper_last"; then exit 0; fi
fails=0
# gpu cc
for spec in "nvidia/a100 80" "nvidia/l4 89" "nvidia/h100 90"; do
  read -r gpu cc <<< "$spec"
  for order in hopper_last hopper_first; do
    log="$(VGPU_QUIET=1 VGPU_GPU=$gpu LD_LIBRARY_PATH="$shim" "$out/$order" "$cc" 2>&1)"
    rc=$?
    if [[ $rc != 0 || "$(tail -n 1 <<< "$log")" != PASS ]]; then
      echo "$log" | sed "s/^/    $gpu $order: /"
      echo "FAIL: $gpu, $order"; fails=1
    else
      echo "ok   $gpu $order"
    fi
  done
done
[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: multi-architecture modules"
exit $fails
