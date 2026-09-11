#!/usr/bin/env bash
# VGPU_QUIET=1 silences diagnostics; any other value, or none, does not. That is
# what include/vgpu_cuda.h documents and what the hand-written shims do, but the
# generated library stubs tested for the variable's *presence* -- so
# VGPU_QUIET=0 silenced "X is not implemented by VirtualGPU" in cuFFT, cuBLAS,
# cuSPARSE, cuSOLVER, cuDNN and NCCL, and nowhere else. That message is the only
# thing telling a user their call was refused rather than answered wrongly.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"

# Loading an instrumented shim into the system Python aborts ("ASan runtime
# does not come first"), so under a sanitizer build this defers to the C++ unit
# tests, which check the same logic with the instrumentation on.
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
san="$(shim_sanitizer "$build/shim")"
if [[ -n "$san" ]]; then echo "SKIP: shim is built with $san, and the Python interpreter is not instrumented"; exit 0; fi
lib="$build/shim/libcufft.so"
command -v python3 >/dev/null || { echo "SKIP: no python3"; exit 0; }
[[ -e "$lib" ]] || { echo "SKIP: no cuFFT shim at $lib"; exit 0; }
fail=0
for case in "unset:yes" "0:yes" "1:no" "true:yes"; do
  q="${case%%:*}"; want="${case##*:}"
  if [[ "$q" == unset ]]; then run=(env -u VGPU_QUIET); else run=(env "VGPU_QUIET=$q"); fi
  out=$("${run[@]}" python3 -c "import ctypes; ctypes.CDLL('$lib').cufftXtExec()" 2>&1)
  if grep -q "not implemented by VirtualGPU" <<< "$out"; then got=yes; else got=no; fi
  if [[ "$got" == "$want" ]]; then echo "ok    VGPU_QUIET=$q -> $([[ $got == yes ]] && echo "diagnostic printed" || echo silent)"
  else echo "FAIL  VGPU_QUIET=$q -> printed=$got, expected $want"; fail=1; fi
done
exit $fail
