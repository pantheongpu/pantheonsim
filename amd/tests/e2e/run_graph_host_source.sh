#!/usr/bin/env bash
# When a captured host-to-device copy takes its bytes: heap memory and a live local at each launch,
# a local of a frame that has gone as it was at capture (graph_host_source.c). rocPRIM's
# device_histogram copies a local of its own to the device from a graph, launched after it returned.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim"
[[ -e "$shim/libamdhip64.so.7" ]] || { echo "SKIP: no HIP shim in $shim"; exit 0; }
command -v gcc >/dev/null || { echo "SKIP: no C compiler"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# A sanitizer-instrumented shim needs its runtime loaded first.
preload=""
if command -v objdump >/dev/null; then
  preload=$(objdump -p "$shim/libamdhip64.so.7" 2>/dev/null | awk '/NEEDED/ && /lib(asan|tsan)\.so/ {print $2}')
fi
for opt in -O0 -O2; do
  gcc $opt -o "$tmp/prog" "$root/amd/tests/e2e/graph_host_source.c" -L"$shim" -l:libamdhip64.so.7 -Wl,-rpath,"$shim" || exit 1
  echo "== $opt"
  LD_PRELOAD="$preload" VGPU_QUIET=1 VGPU_GPU=amd/mi300x "$tmp/prog" || exit 1
done
