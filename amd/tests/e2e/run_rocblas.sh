#!/usr/bin/env bash
# AMD's rocBLAS, unmodified, on a simulated MI300X.
#
# A program built by hipcc against rocBLAS calls it as a program on a card
# would: level-1 routines whose kernels are built into the library, and GEMMs
# whose kernels are Tensile's, loaded from the code objects rocBLAS ships.
# Every result is checked against the same arithmetic done on the host.
#
# rocBLAS (with hipBLASLt, which it links) is ROCm's, so this runs wherever
# ROCm with rocBLAS is installed and skips everywhere else.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim"
[[ -e "$shim/libamdhip64.so.7" ]] || { echo "SKIP: no HIP shim in $shim"; exit 0; }
rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV); do
  [[ -n "$c" && -x "$c/bin/hipcc" && -e "$c/include/rocblas/rocblas.h" ]] && ls "$c"/lib/librocblas.so.* >/dev/null 2>&1 &&
    { rocm=$c; break; }
done
[[ -n "$rocm" ]] || { echo "SKIP: no ROCm with rocBLAS found (set VGPU_ROCM_PATH)"; exit 0; }
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
  preload=$(objdump -p "$shim/libamdhip64.so.7" 2>/dev/null | awk '/NEEDED/ && /lib(asan|tsan)\.so/ {print $2}')
fi

export ROCM_PATH=$rocm HIP_PATH=$rocm HIP_CLANG_PATH=$rocm/lib/llvm/bin HIP_DEVICE_LIB_PATH=$rocm/amdgcn/bitcode
if ! "$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 "$root/amd/tests/rocblas/blas.cpp" -o "$tmp/blas" \
     -L"$rocm/lib" -lrocblas 2>"$tmp/err"; then
  expect "hipcc builds the rocBLAS program" "yes" "no: $(grep -m2 error "$tmp/err")"
  exit $fail
fi
# The shim takes libamdhip64's place; rocBLAS, hipBLASLt and the code objects
# rocBLAS loads are ROCm's own.
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x LD_PRELOAD="$preload" LD_LIBRARY_PATH="$shim:$rocm/lib" "$tmp/blas" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "the program runs to the end" "0" "$status"
expect "rocBLAS sees the device HIP names" "yes" \
  "$(grep -q '^rocBLAS .* on gfx942:sramecc+:xnack-$' <<< "$out" && echo yes || echo no)"
expect "saxpy, a kernel built into the library" "saxpy: 1000 of 1000 elements right" "$(grep -o '^saxpy: .*' <<< "$out")"
expect "sdot, reduced on the device" "sdot: right" "$(grep -o '^sdot: .*' <<< "$out")"
expect "sgemm, through Tensile's float matrix instructions" "sgemm 96x80x64: 7680 of 7680 elements right" \
  "$(grep -o '^sgemm .*' <<< "$out")"
expect "dgemm, through its double ones" "dgemm 96x80x64, A transposed: 7680 of 7680 elements right" \
  "$(grep -o '^dgemm .*' <<< "$out")"
expect "ragged GEMMs, where the kernels count on a buffer's bounds at the edges" \
  "ragged GEMMs, every transpose, float and double: 48 of 48 right" "$(grep -o '^ragged GEMMs.*' <<< "$out")"
expect "gemm_ex of halves, bfloat16s and bytes, through their matrix instructions and half-register loads" \
  "gemm_ex, halves, bfloat16s and bytes, every transpose: 100 of 100 right" "$(grep -o '^gemm_ex, .*' <<< "$out")"
expect "complex GEMMs, float and double, each operand transposed, conjugated or neither" \
  "complex GEMMs, every transpose and conjugate: 90 of 90 right" "$(grep -o '^complex GEMMs.*' <<< "$out")"
exit $fail
