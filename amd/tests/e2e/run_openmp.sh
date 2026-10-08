#!/usr/bin/env bash
# OpenMP target offloading, built by ROCm's amdclang++ and run on every
# simulated AMD GPU.
#
# LLVM's libomptarget has an AMD plugin that talks to the HSA runtime directly
# (queues, signals, memory pools, executables), so this reaches a part of the
# simulator no HIP program does. amd/tests/openmp/omp.cpp is built for each
# GPU's ISA and run on two simulated devices of that model; it checks its own
# answers and prints "N checks, 0 failed".
#
# Needs a ROCm with its LLVM (rocm-llvm and rocm-device-libs): found through
# VGPU_ROCM_PATH, ROCM_PATH or /opt/rocm; skips elsewhere.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim"
[[ -e "$shim/libhsa-runtime64.so.1" ]] || { echo "SKIP: no HSA runtime in $shim"; exit 0; }
rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV); do
  [[ -n "$c" && -x "$c/lib/llvm/bin/amdclang++" ]] && ls "$c"/lib/llvm/lib/libomptarget.so* >/dev/null 2>&1 &&
    { rocm=$c; break; }
done
[[ -n "$rocm" ]] || { echo "SKIP: no ROCm LLVM with OpenMP offload found (set VGPU_ROCM_PATH)"; exit 0; }
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# A sanitizer-instrumented shim needs its runtime loaded first, by the name
# the shim asks for.
preload=""
if command -v objdump >/dev/null; then
  preload=$(objdump -p "$shim/libhsa-runtime64.so.1" 2>/dev/null | awk '/NEEDED/ && /lib(asan|tsan)\.so/ {print $2}')
fi

# The GPUs this ISA runs on: one simulated model per target, and all six.
cases="amd/mi300x:gfx942 amd/mi250x:gfx90a amd/mi350x:gfx950 amd/rx7900xtx:gfx1100 amd/rx9070xt:gfx1201 amd/rx6900xt:gfx1030"
[[ -n "${VGPU_OPENMP_GPUS:-}" ]] && cases="$VGPU_OPENMP_GPUS"
for pair in $cases; do
  gpu=${pair%:*} arch=${pair#*:}
  exe="$tmp/omp.$arch"
  if ! "$rocm/lib/llvm/bin/amdclang++" -O2 -fopenmp --offload-arch="$arch" "$root/amd/tests/openmp/omp.cpp" -o "$exe" 2> "$tmp/cc.log"; then
    echo "FAIL  $gpu: the program did not build"; sed 's/^/      /' "$tmp/cc.log" | head -20; fail=1; continue
  fi
  out=$(LD_PRELOAD="$preload" VGPU_QUIET=1 VGPU_GPU="$gpu" VGPU_DEVICE_COUNT=2 \
        LD_LIBRARY_PATH="$shim:$rocm/lib/llvm/lib" timeout 900 "$exe" 2>&1)
  status=$?
  grep -E '^FAIL' <<< "$out" | sed 's/^/      /'
  expect "$gpu ($arch): it runs to the end" "0" "$status"
  expect "$gpu ($arch): both simulated devices are offload devices" "devices 2 default 0 initial 2" \
    "$(grep -o '^devices .*' <<< "$out")"
  expect "$gpu ($arch): printf from a target region reaches stdout" "printf from the device: 42 2.50" \
    "$(grep -o '^printf from the device: .*' <<< "$out")"
  expect "$gpu ($arch): every check passes" "0" "$(grep -oE ', [0-9]+ failed' <<< "$out" | grep -oE '[0-9]+')"
done
exit $fail
