#!/usr/bin/env bash
# Numba and Triton, unmodified, compiling Python to GPU kernels that run on the
# simulator.
#
# These are the two JIT frameworks that reach the interpreter, and they reach it
# by different routes: Numba assembles its own PTX through the driver's JIT link
# API, Triton compiles to PTX and is stopped there by tools/vgpu_triton.py
# before it can shell out to ptxas. Neither uses cudart, so neither is covered
# by anything else in this suite.
#
# Skips cleanly when the frameworks are not installed -- they are a heavy
# optional dependency, and a contributor without them still gets a green run.
# Point VGPU_PY at an interpreter that has them.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"

py="${VGPU_PY:-python3}"
command -v "$py" >/dev/null 2>&1 || { echo "SKIP: $py not found"; exit 0; }
[[ -e "$shim/libcuda.so.1" ]] || { echo "SKIP: libvgpucuda not built"; exit 0; }

status=0
for fw in numba triton; do
  if ! "$py" -c "import $fw" >/dev/null 2>&1; then
    echo "SKIP: $fw not installed for $py"
    continue
  fi
  # Triton also has to be new enough to expose the stage hook; older releases
  # can only be driven through ptxas, which produces a cubin nothing here loads.
  if [[ $fw == triton ]] &&
     ! PYTHONPATH="$root/tools" "$py" -c \
       "import vgpu_triton, sys; sys.exit(0 if vgpu_triton.available() else 1)" \
       >/dev/null 2>&1; then
    echo "SKIP: this Triton has no knobs.runtime.add_stages_inspection_hook"
    continue
  fi
  echo "--- $fw ---"
  if VGPU_QUIET=1 VGPU_GPU=nvidia/a10 LD_LIBRARY_PATH="$shim" \
     "$py" "$root/tests/e2e/jit_$fw.py"; then
    :
  else
    echo "FAIL: $fw"
    status=1
  fi
done
exit $status
