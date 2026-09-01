#!/usr/bin/env bash
# Differential conformance: compile each test once, run it on a physical GPU
# and on VirtualGPU, and diff the results. Any difference is a semantics bug.
#
#   tests/conformance/run_conformance.sh            # both sides (needs a GPU)
#   VGPU_ONLY=1 tests/conformance/run_conformance.sh  # virtual side only
#
# The physical reference must be built with the default (static) cudart; the
# virtual side uses -cudart shared so the shim can be substituted.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
shim="$root/build/shim"
out="${TMPDIR:-/tmp}/vgpu-conformance"
mkdir -p "$out"
: "${VGPU_CONF_GPU:=nvidia/a10}"   # sm_86 profile matches the sm_86 build
: "${VGPU_CONF_ARCH:=sm_86}"

command -v nvcc >/dev/null || { echo "SKIP: nvcc not found"; exit 0; }
[[ -e "$shim/libcudart.so.13" ]] || { echo "SKIP: build VirtualGPU first"; exit 0; }
have_gpu=0
if [[ -z "${VGPU_ONLY:-}" ]] && nvidia-smi -L >/dev/null 2>&1; then have_gpu=1; fi

fail=0
for src in "$root"/tests/conformance/*.cu; do
  name="$(basename "$src" .cu)"
  # Link whatever vendor libraries the test uses; the shim supplies the
  # VirtualGPU implementations of the same sonames at run time.
  libs=""
  grep -q "cublas" "$src" && libs="$libs -lcublas"
  nvcc -std=c++14 -arch="$VGPU_CONF_ARCH" -Wno-deprecated-gpu-targets "$src" \
       -o "$out/$name.real" $libs 2>/dev/null
  nvcc -std=c++14 -arch="$VGPU_CONF_ARCH" -Wno-deprecated-gpu-targets -cudart shared "$src" \
       -o "$out/$name.virt" $libs 2>/dev/null
  VGPU_QUIET=1 VGPU_GPU="$VGPU_CONF_GPU" VGPU_VRAM_MB=256 LD_LIBRARY_PATH="$shim" \
    timeout 300 "$out/$name.virt" > "$out/$name.virt.txt" 2>&1
  vrc=$?
  if [[ $vrc -ne 0 ]] || grep -q "LAUNCHFAIL\|ALLOCFAIL" "$out/$name.virt.txt"; then
    echo "FAIL  $name: VirtualGPU did not complete"
    grep -m3 -E "error \[|LAUNCHFAIL" "$out/$name.virt.txt" | sed 's/^/        /'
    fail=1; continue
  fi
  if [[ $have_gpu -eq 0 ]]; then
    echo "ok    $name (virtual only; no physical GPU to compare against)"
    continue
  fi
  timeout 300 "$out/$name.real" > "$out/$name.real.txt" 2>&1
  if diff -q "$out/$name.real.txt" "$out/$name.virt.txt" >/dev/null; then
    echo "MATCH $name ($(wc -l < "$out/$name.real.txt") values identical to hardware)"
  elif python3 "$root/tests/conformance/compare_numeric.py" \
         "$out/$name.real.txt" "$out/$name.virt.txt" "${VGPU_CONF_TOL:-1e-5}"; then
    echo "MATCH $name (within floating-point tolerance of hardware)"
  else
    echo "FAIL  $name: differs from hardware"
    diff "$out/$name.real.txt" "$out/$name.virt.txt" | head -10 | sed "s/^/        /"
    fail=1
  fi
done
exit $fail
