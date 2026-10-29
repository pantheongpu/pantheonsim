#!/usr/bin/env bash
# The shims must not export, or relocate by name, the std::variant dispatch tables (_ZNSt8__detail9__variant...): their
# mangled names are 1,000 to 2,000 characters long, the loader hashes every name it looks up at every process start, and
# exported they were a quarter of a small program's user cycles (docs/performance.md, "Round sp5"). The version scripts in
# CMakeLists.txt (VGPU_LOCAL_STD_TABLES) make them local. Usage: check_shim_dynsyms.sh <build dir> [lib ...]
set -u
build="${1:-${VGPU_BUILD_DIR:-}}"
[[ -n "$build" ]] || { echo "usage: $0 <build dir> [lib ...]" >&2; exit 2; }
shift || true
command -v nm >/dev/null 2>&1 && command -v readelf >/dev/null 2>&1 || { echo "SKIP: nm or readelf missing"; exit 0; }
libs=("$@")
if [[ ${#libs[@]} -eq 0 ]]; then
  for f in "$build"/lib*.so.*; do
    [[ -f "$f" && ! -L "$f" ]] && libs+=("$f")
  done
fi
[[ ${#libs[@]} -gt 0 ]] || { echo "SKIP: no shared libraries in $build"; exit 0; }
fail=0 checked=0
for lib in "${libs[@]}"; do
  if [[ ! -f "$lib" ]]; then echo "FAIL: $lib does not exist"; fail=1; continue; fi
  checked=$((checked + 1))
  defined=$(nm -D --defined-only "$lib" 2>/dev/null | grep -c '_ZNSt8__detail9__variant' || true)
  relocated=$(readelf -W -r "$lib" 2>/dev/null | grep -c '_ZNSt8__detail9__variant' || true)
  if [[ "$defined" != 0 || "$relocated" != 0 ]]; then
    echo "FAIL: $(basename "$lib") exports $defined and relocates $relocated std::variant dispatch table symbols"
    fail=1
  fi
done
[[ $checked -gt 0 ]] || { echo "SKIP: nothing to check"; exit 0; }
[[ $fail -eq 0 ]] && echo "PASS: $checked shared libraries carry no std::variant dispatch table symbols"
exit $fail
