#!/usr/bin/env bash
# hipGraphInstantiateWithParams with each flag ROCm takes there, Upload with an uploadStream among them
# (graph_instantiate_params.c, pantheonsim #628). Colvars instantiates its graphs that way.
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
  gcc $opt -o "$tmp/prog" "$root/amd/tests/e2e/graph_instantiate_params.c" -L"$shim" -l:libamdhip64.so.7 -Wl,-rpath,"$shim" || exit 1
  echo "== $opt"
  LD_PRELOAD="$preload" VGPU_QUIET=1 VGPU_GPU=amd/mi300x "$tmp/prog" || exit 1
done
