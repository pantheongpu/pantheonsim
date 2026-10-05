#!/usr/bin/env bash
# AMD's hipFFT, hipRAND, hipSOLVER and hipSPARSE, unmodified, on a simulated
# MI300X: programs built by hipcc against them (amd/tests/libraries) call
# them as programs on a card would, and check every answer on the host.
#
# The libraries are ROCm's, so this runs wherever ROCm with them is installed,
# or wherever PyTorch's ROCm wheel is, which carries all four (that is how CI
# runs it, with nothing of ROCm installed), and skips elsewhere. The programs
# are built ahead of time (amd/tests/libraries/build.sh).
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim"
dir="$root/amd/tests/libraries"
[[ -e "$shim/libamdhip64.so.7" ]] || { echo "SKIP: no HIP shim in $shim"; exit 0; }
libs=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm $(ls -d /opt/rocm-* 2>/dev/null | sort -rV); do
  [[ -n "$c" ]] || continue
  have=yes
  for l in hipfft hiprand hipsolver hipsparse; do ls "$c"/lib/lib$l.so.* >/dev/null 2>&1 || have=no; done
  [[ $have == yes ]] && { libs="$c/lib"; break; }
done
[[ "${VGPU_LIBRARIES_FROM_WHEEL:-}" == 1 ]] && libs=""
wheel=""
if [[ -z "$libs" ]]; then
  for c in "${VGPU_TORCH_PYTHON:-}" $(ls -d "$HOME"/.local/share/torch-rocm*/bin/python 2>/dev/null); do
    [[ -n "$c" && -x "$c" ]] || continue
    d=$("$c" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)
    [[ -n "$d" && -e "$d/libhipfft.so" && -e "$d/libhipsolver.so" ]] && { wheel=$d; break; }
  done
fi
[[ -n "$libs" || -n "$wheel" ]] || { echo "SKIP: no ROCm with hipFFT, hipRAND, hipSOLVER and hipSPARSE, and no PyTorch ROCm wheel"; exit 0; }
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

if [[ -n "$wheel" ]]; then
  # The wheel's libraries, which it names without a version, by the names the
  # programs ask for; HIP is the simulator's.
  echo "      (the libraries from the PyTorch wheel at $wheel)"
  mkdir -p "$tmp/lib"
  for f in "$wheel"/*; do [[ "$(basename "$f")" == libamdhip64.so ]] || ln -s "$f" "$tmp/lib/"; done
  for p in "$dir"/*.gfx942; do
    for need in $(objdump -p "$p" | awk '/NEEDED/ {print $2}'); do
      [[ -e "$tmp/lib/$need" || ! -e "$wheel/${need%%.so*}.so" ]] || ln -s "$wheel/${need%%.so*}.so" "$tmp/lib/$need"
    done
  done
  ln -sf "$(readlink -f "$shim/libamdhip64.so.7")" "$tmp/lib/libamdhip64.so"
  libs="$tmp/lib"
else
  echo "      (the libraries from $libs)"
fi

run() {  # run <program>: its output, and its status in $status
  out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x LD_PRELOAD="$preload" LD_LIBRARY_PATH="$shim:$libs" timeout 1200 "$dir/$1" 2>&1)
  status=$?
  echo "$out" | sed 's/^/      /'
}

run fft.gfx942
expect "hipFFT: complex, real and double transforms, batched and 2D, forward and back" \
  "hipFFT: 8 of 8 transforms match the host's" "$(grep -o '^hipFFT: .*' <<< "$out")"
expect "the hipFFT program exits cleanly" "0" "$status"

run rand.gfx942
expect "hipRAND: each generator's numbers on the device are the host generator's, and the device API's" \
  "hipRAND: 19 of 19 sequences match the host's" "$(grep -o '^hipRAND: .*' <<< "$out")"
expect "the hipRAND program exits cleanly" "0" "$status"

run solver.gfx942
expect "hipSOLVER: LU and solve, Cholesky, symmetric eigenvalues and SVD" \
  "hipSOLVER: 4 of 4 factorizations hold" "$(grep -o '^hipSOLVER: .*' <<< "$out")"
expect "the hipSOLVER program exits cleanly" "0" "$status"

run sparse.gfx942
expect "hipSPARSE: SpMV (CSR, COO, transposed, double), SpMM and SparseToDense" \
  "6 of 6 products match" "$(grep -o '[0-9]* of [0-9]* products match' <<< "$out")"
expect "the hipSPARSE program exits cleanly" "0" "$status"
exit $fail
