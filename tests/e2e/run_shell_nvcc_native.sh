#!/usr/bin/env bash
# `nvcc -arch=native` inside vgpu shell builds for the session's card.
#
# The real nvcc resolves native by asking the driver -- which here answers with
# the simulated card -- and then builds machine code alone for it, no PTX. A
# GPU runs that; the simulator runs PTX, so every launch was refused with "no
# PTX (SASS-only fatbin)". The session's nvcc now resolves native itself, to
# -arch=sm_XX for the card it has, which embeds both.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
[[ -x "$build/vgpu" ]] || { echo "SKIP: vgpu not built"; exit 0; }
san="$(shim_sanitizer "$build/shim")"
if [[ -n "$san" ]]; then echo "SKIP: $san shim (the test program is built without it)"; exit 0; fi
nvcc_bin="$(pick_nvcc_for_shim "$build/shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching the shim's CUDA major"; exit 0; }
nvcc_host_compiler_fix
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_native.XXXXXX")"; trap 'rm -rf "$work"' EXIT

cat > "$work/k.cu" <<'CU'
#include <cstdio>
__global__ void addk(float* x) { x[threadIdx.x] += 1.f; }
int main() {
  float h[32]; for (int i = 0; i < 32; ++i) h[i] = i;
  float* d; cudaMalloc(&d, sizeof h);
  cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice);
  addk<<<1, 32>>>(d);
  cudaError_t e = cudaGetLastError();
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  printf("%s h[7]=%.1f\n", cudaGetErrorString(e), h[7]);
  return e == cudaSuccess && h[7] == 8.f ? 0 : 1;
}
CU

fails=0
for form in "-arch=native" "--gpu-architecture=native" "-arch native"; do
  out=$(cd "$work" && PATH="$(dirname "$nvcc_bin"):$PATH" VGPU_QUIET=1 \
        "$build/vgpu" shell -y --gpu nvidia/t4 --no-isolate \
        -c "nvcc $form -Wno-deprecated-gpu-targets k.cu -o k && ./k" 2>&1)
  if grep -q "no error h\[7\]=8.0" <<< "$out"; then
    echo "ok    nvcc $form builds for the session's T4 and runs"
  else
    echo "FAIL  nvcc $form:"; sed 's/^/    /' <<< "$out" | tail -8; fails=$((fails + 1))
  fi
done
exit $((fails > 0))
