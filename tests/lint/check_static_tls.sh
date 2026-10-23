#!/usr/bin/env bash
# No shim library takes static TLS.
#
# The simulator core is linked into every shim library. glibc keeps only a small static-TLS surplus
# (about 1.6 KB) for all libraries a process loads after it starts, shared with ROCm, CUDA and Python's
# own; a library with an initial-exec or local-exec thread-local (the DF_STATIC_TLS flag, or a TPOFF
# relocation) takes some of it, and the process that loads enough shims (rocprofv3, a Python program,
# the symbol lint) fails to dlopen with "cannot allocate memory in static TLS block". Thread-locals
# of the core are general-dynamic. This checks every library: no STATIC_TLS flag, no TPOFF relocation,
# and then that all of them load, one after the other, in a process whose static TLS surplus has been
# used up but for a small budget by a library of its own (LD_PRELOAD of tls_hog, built here).
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
shopt -s nullglob
libs=("$shim"/lib*.so*)
shopt -u nullglob
(( ${#libs[@]} )) || { echo "SKIP: no shim libraries built"; exit 0; }
command -v readelf >/dev/null || { echo "SKIP: no readelf"; exit 0; }
rc=0
declare -A seen=()
n=0
for lib in "${libs[@]}"; do
  real="$(readlink -f "$lib")"
  [[ -n "${seen[$real]:-}" ]] && continue
  seen[$real]=1
  n=$((n + 1))
  if readelf -d "$real" 2>/dev/null | grep -q 'STATIC_TLS'; then
    echo "FAIL: $(basename "$real") is flagged STATIC_TLS (an initial-exec or local-exec thread-local)"; rc=1
  fi
  if readelf -r "$real" 2>/dev/null | grep -Eq 'R_X86_64_TPOFF|R_AARCH64_TLS_TPREL'; then
    echo "FAIL: $(basename "$real") has a TPOFF relocation (static TLS):"
    readelf -r "$real" | grep -E 'R_X86_64_TPOFF|R_AARCH64_TLS_TPREL' | head -3; rc=1
  fi
done
# Load them all with the surplus nearly gone.
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-static-tls.XXXXXX")"; trap 'rm -rf "$work"' EXIT
cat > "$work/hog.c" <<'C'
/* An initial-exec thread-local of 1400 bytes: most of what glibc leaves for libraries loaded later. */
__thread char vgpu_tls_hog[1400] __attribute__((tls_model("initial-exec")));
C
cat > "$work/load.c" <<'C'
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char** argv) {
  int bad = 0;
  for (int i = 1; i < argc; ++i)
    if (!dlopen(argv[i], RTLD_LAZY | RTLD_GLOBAL)) { printf("FAIL: %s: %s\n", argv[i], dlerror()); bad = 1; }
  return bad;
}
C
if cc -shared -fPIC -o "$work/libhog.so" "$work/hog.c" && cc -o "$work/load" "$work/load.c" -ldl; then
  out="$(cd "$work" && LD_PRELOAD="$work/libhog.so" VGPU_QUIET=1 LD_LIBRARY_PATH="$shim" ./load "${!seen[@]}" 2>&1)" || { echo "$out"; rc=1; }
else
  echo "SKIP load check: no C compiler"
fi
(( rc == 0 )) && echo "ok: $n libraries take no static TLS and load with the surplus used up"
exit $rc
