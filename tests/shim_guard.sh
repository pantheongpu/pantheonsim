# Shared guard for tests that compile an app with nvcc and run it against the
# shims in one build directory.
#
# The shim's soname major follows the toolkit it was built against, and nvcc
# stamps the app with the major *it* ships. Those can differ on one host -- a
# CUDA 12 build directory next to a CUDA 13 nvcc -- and then the app asks for a
# library this build does not provide, quietly falls through to the real
# libcudart, and fails against the real driver. That is an unbuildable
# combination rather than a defect, so it skips.
#
# Skipping only matters if it cannot hide a genuine failure, so the check is on
# what the linker actually recorded, not on a guess about versions.

# require_shim_libs <shim_dir> <binary>
# Returns 1 (and prints a SKIP line) when the binary needs a CUDA library this
# shim directory does not supply.
require_shim_libs() {
  local shim="$1" bin="$2" need
  command -v objdump >/dev/null 2>&1 || return 0
  for need in $(objdump -p "$bin" 2>/dev/null | awk '/NEEDED/ {print $2}'); do
    case "$need" in
      libcudart.so.*|libcublas.so.*|libcublasLt.so.*|libnccl.so.*|libcuda.so.*|libcudnn.so.*)
        if [[ ! -e "$shim/$need" ]]; then
          echo "SKIP: app needs $need, which $shim does not provide" \
               "(nvcc's toolkit major differs from this build's)"
          return 1
        fi
        ;;
    esac
  done
  return 0
}

# The sanitizer the shim was built with, if any: "asan", "tsan" or empty.
#
# A sanitizer has to be linked into the *program*, not just into a library it
# loads. An uninstrumented nvcc binary that dlopens an ASan-built shim aborts
# with "ASan runtime does not come first in initial library list", and TSan
# cannot map its shadow at all. So these tests compile the app with whatever
# the shim was built with, and the answer comes from the shim's own linkage
# rather than from a flag someone has to remember to pass.
# Read statically, with objdump rather than ldd. ldd *runs* the object through
# the loader, and doing that to a sanitizer-instrumented library -- which maps
# terabytes of shadow -- fails often enough under load to matter: when it did,
# this reported "no sanitizer", the app was built without one, and the test
# failed with the very link-order error the detection exists to avoid. A
# detection that fails open is worse than none.
shim_sanitizer() {
  local shim="$1" lib needed
  command -v objdump >/dev/null 2>&1 || return 0
  shopt -s nullglob
  local libs=("$shim"/libcudart.so.[0-9]*)
  shopt -u nullglob
  (( ${#libs[@]} )) || return 0
  lib="${libs[0]}"
  needed="$(objdump -p "$lib" 2>/dev/null | awk '/NEEDED/ {print $2}')"
  case "$needed" in
    *libtsan*) echo tsan ;;
    *libasan*) echo asan ;;
  esac
}

# nvcc flags that match the shim's sanitizer. Passed one at a time because
# nvcc reads a comma inside -Xcompiler as an argument separator, which turns
# "-fsanitize=address,undefined" into two compiler invocations and fails.
shim_sanitizer_nvcc_flags() {
  case "$(shim_sanitizer "$1")" in
    asan) printf '%s' "-Xcompiler -fsanitize=address -Xcompiler -fsanitize=undefined -Xcompiler -fno-omit-frame-pointer -g" ;;
    tsan) printf '%s' "-Xcompiler -fsanitize=thread -g" ;;
    *) printf '%s' "" ;;
  esac
}

# pick_nvcc_for_shim <shim_dir>
# Echoes the path of an nvcc whose toolkit major matches this shim's, or nothing.
#
# A machine can have more than one toolkit, and the first nvcc on PATH is often
# not the one this build directory was configured against. Compiling with the
# wrong one produces a binary that asks for a soname the shim does not carry,
# which is a real diagnostic in `vgpu run` and pure noise in a test.
pick_nvcc_for_shim() {
  local shim="$1" cand major
  shopt -s nullglob
  local libs=("$shim"/libcudart.so.[0-9]*)
  shopt -u nullglob
  (( ${#libs[@]} )) || return 0
  local want="${libs[0]##*.}"
  for cand in "${VGPU_NVCC:-}" "$(command -v nvcc 2>/dev/null)" /usr/bin/nvcc \
              /usr/local/cuda/bin/nvcc; do
    [[ -n "$cand" && -x "$cand" ]] || continue
    major="$("$cand" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)[,.].*/\1/p' | head -1)"
    [[ "$major" == "$want" ]] && { printf '%s' "$cand"; return 0; }
  done
  return 0
}

# nvcc_host_compiler_fix
# CUDA 12's nvcc rejects GCC 13's <bits/floatn.h> on _Float128 and fails inside
# <math.h> before it sees any of our code. Pointing it at g++-12 is the standard
# workaround, and the CI workflow sets the same thing for the same reason;
# doing it here too means a developer's shell does not have to.
nvcc_host_compiler_fix() {
  if [[ -z "${NVCC_PREPEND_FLAGS:-}" ]] && command -v g++-12 >/dev/null 2>&1; then
    export NVCC_PREPEND_FLAGS="-ccbin g++-12"
  fi
}
