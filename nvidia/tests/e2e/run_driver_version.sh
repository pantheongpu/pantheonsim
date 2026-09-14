#!/usr/bin/env bash
# One machine, one CUDA version. On real hardware cuDriverGetVersion,
# cudaDriverGetVersion and nvidia-smi's "CUDA Version" all read the same
# installed driver, so they cannot disagree. Here they came from three places --
# a constant, the runtime's build toolkit, and the session's --cuda flag -- and
# one session answered 13.0, 12.6 and 12.4 depending on who asked. This pins
# them to the session's declaration and to one default.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"

# Loading an instrumented shim into the system Python aborts ("ASan runtime
# does not come first"), so under a sanitizer build this defers to the C++ unit
# tests, which check the same logic with the instrumentation on.
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
san="$(shim_sanitizer "$build/shim")"
if [[ -n "$san" ]]; then echo "SKIP: shim is built with $san, and the Python interpreter is not instrumented"; exit 0; fi
shim="$build/shim"
command -v python3 >/dev/null || { echo "SKIP: no python3"; exit 0; }
[[ -e "$shim/libcuda.so.1" ]] || { echo "no driver shim at $shim"; exit 1; }

probe() {  # prints "<driver-api> <runtime-api> <nvidia-smi>"
  VGPU_QUIET=1 python3 - "$shim" <<'PY'
import ctypes, os, sys
shim = sys.argv[1]
cu = ctypes.CDLL(os.path.join(shim, "libcuda.so.1"))
rts = [f for f in os.listdir(shim) if f.startswith("libcudart.so.")]
d = ctypes.c_int(); cu.cuDriverGetVersion(ctypes.byref(d))
r = ctypes.c_int(-1)
if rts: ctypes.CDLL(os.path.join(shim, rts[0])).cudaDriverGetVersion(ctypes.byref(r))
print(d.value, r.value, end=" ")
PY
  VGPU_GPU=nvidia/h100 VGPU_TELEMETRY_PATH=/nonexistent-so-idle "$build/vgpu" smi 2>/dev/null \
    | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p'
}

fail=0
check() {  # VGPU_CUDA_VERSION value ("" for unset), expected encoding, expected smi text
  local got
  if [[ -n "$1" ]]; then got=$(VGPU_CUDA_VERSION="$1" probe); else got=$(env -u VGPU_CUDA_VERSION bash -c "$(declare -f probe); shim='$shim' build='$build' probe"); fi
  read -r drv rt smi <<< "$got"
  local label="${1:-(unset)}"
  if [[ "$drv" == "$2" && ( "$rt" == "$2" || "$rt" == "-1" ) && "$smi" == "$3" ]]; then
    echo "ok    VGPU_CUDA_VERSION=$label -> driver $drv, runtime $rt, nvidia-smi $smi"
  else
    echo "FAIL  VGPU_CUDA_VERSION=$label -> driver $drv, runtime $rt, nvidia-smi $smi (expected $2 / $3)"
    fail=1
  fi
}

check ""      13000 13.0
check "12.4"  12040 12.4
check "12.6"  12060 12.6
check "13.0"  13000 13.0
check "junk"  13000 13.0   # rejected everywhere alike -- nvidia-smi must not show what the driver refused
exit $fail
