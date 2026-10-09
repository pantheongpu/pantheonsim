#!/usr/bin/env bash
# OpenCL on every simulated AMD GPU.
#
# ROCm's OpenCL runtime (libamdocl64, built on the same CLR as HIP) compiles
# each kernel with ROCm's compiler library and runs it through the HSA runtime,
# which here is the simulator's. amd/tests/opencl/cl.c is an ordinary OpenCL
# program built with the system's compiler and ICD loader; it checks its own
# answers and prints "N checks, 0 failed".
#
# Needs ROCm's OpenCL runtime (rocm-opencl, comgr, rocm-device-libs: found
# through VGPU_ROCM_PATH, ROCM_PATH or /opt/rocm), a C compiler and the
# OpenCL headers and loader (ocl-icd-opencl-dev); skips elsewhere.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim"
[[ -e "$shim/libhsa-runtime64.so.1" ]] || { echo "SKIP: no HSA runtime in $shim"; exit 0; }
rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV); do
  [[ -n "$c" && -e "$c/lib/libamdocl64.so" && -e "$c/lib/libamd_comgr.so" ]] && { rocm=$c; break; }
done
[[ -n "$rocm" ]] || { echo "SKIP: no ROCm OpenCL runtime found (set VGPU_ROCM_PATH)"; exit 0; }
command -v gcc >/dev/null || { echo "SKIP: no C compiler"; exit 0; }
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if ! gcc -O1 -Wall "$root/amd/tests/opencl/cl.c" -o "$tmp/cl" -lOpenCL -lm 2> "$tmp/cc.log"; then
  if grep -q 'CL/cl.h\|-lOpenCL' "$tmp/cc.log"; then echo "SKIP: no OpenCL headers or ICD loader (ocl-icd-opencl-dev)"; exit 0; fi
  echo "FAIL  the program did not build"; sed 's/^/      /' "$tmp/cc.log" | head -20; exit 1
fi

# The one platform: the ICD loader reads its vendor from a file naming the
# library, here ROCm's.
mkdir -p "$tmp/vendors"
echo "$rocm/lib/libamdocl64.so" > "$tmp/vendors/amdocl64.icd"

preload=""
if command -v objdump >/dev/null; then
  preload=$(objdump -p "$shim/libhsa-runtime64.so.1" 2>/dev/null | awk '/NEEDED/ && /lib(asan|tsan)\.so/ {print $2}')
fi

cases="amd/mi300x amd/mi250x amd/mi350x amd/rx7900xtx amd/rx9070xt amd/rx6900xt amd/rx6800 amd/rx6700xt"
[[ -n "${VGPU_OPENCL_GPUS:-}" ]] && cases="$VGPU_OPENCL_GPUS"
for gpu in $cases; do
  # MI250X's texture units are real, and the gfx9 image instructions are not
  # modelled yet: its images are the one thing left out.
  noimg=""; [[ $gpu == amd/mi250x ]] && noimg=1
  out=$(VGPU_CL_NO_IMAGES=$noimg LD_PRELOAD="$preload" VGPU_QUIET=1 VGPU_GPU="$gpu" VGPU_DEVICE_COUNT=2 OCL_ICD_VENDORS="$tmp/vendors" \
        ROCM_PATH="$rocm" LD_LIBRARY_PATH="$shim:$rocm/lib" timeout 900 "$tmp/cl" 2>&1)
  status=$?
  grep -E '^(FAIL|build log)' <<< "$out" | sed 's/^/      /'
  expect "$gpu: it runs to the end" "0" "$status"
  expect "$gpu: both simulated devices are OpenCL devices" "devices 2" "$(grep -o '^devices [0-9]*' <<< "$out")"
  expect "$gpu: every check passes" "0" "$(grep -oE ', [0-9]+ failed' <<< "$out" | grep -oE '[0-9]+')"
done
exit $fail
