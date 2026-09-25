#!/usr/bin/env bash
# Every shim library loads with every symbol resolved.
#
# A function declared in an anonymous namespace and defined outside it compiles
# and links into a shared library as an undefined reference that nothing can
# ever satisfy. A program that binds lazily never notices until it calls it; one
# that binds at load -- Python's ctypes does -- cannot load the library at all.
# That happened once in development, and only the Python-driven tests noticed it. This checks it
# directly: no library may import a symbol from an anonymous namespace, and each
# must dlopen with RTLD_NOW.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
shopt -s nullglob
libs=("$shim"/lib*.so*)
shopt -u nullglob
(( ${#libs[@]} )) || { echo "SKIP: no shim libraries built"; exit 0; }

rc=0
declare -A seen=()
checked=0
for lib in "${libs[@]}"; do
  real="$(readlink -f "$lib")"
  [[ -n "${seen[$real]:-}" ]] && continue
  seen[$real]=1
  checked=$((checked + 1))
  internal=$(nm -D --undefined-only "$real" 2>/dev/null | awk '{print $NF}' | grep '_GLOBAL__N' || true)
  if [[ -n "$internal" ]]; then
    echo "FAIL $(basename "$real") imports symbols from an anonymous namespace, which no library can define:"
    echo "$internal" | sed 's/^/       /' | head -5
    rc=1
  fi
  # A sanitizer-instrumented library cannot be loaded into an uninstrumented
  # interpreter; the symbol check above still covers it.
  # Not grep -q: it stops at the first match, nm dies of SIGPIPE, and pipefail
  # turns the match into a failure.
  if nm -D "$real" 2>/dev/null | grep -E '__(asan|tsan|ubsan)_' >/dev/null; then continue; fi
  if ! out=$(LD_LIBRARY_PATH="$shim" python3 -c 'import ctypes, os, sys
ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOW)' "$real" 2>&1); then
    echo "FAIL $(basename "$real") does not load with every symbol bound: ${out##*OSError: }"
    rc=1
  fi
done
[[ $rc -eq 0 ]] && echo "shim symbols: $checked libraries load with every symbol resolved: PASS"
exit $rc
