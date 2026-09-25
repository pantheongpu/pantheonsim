#!/usr/bin/env bash
# The same HIP programs, built by each current ROCm release (6.4, 7.0, 7.1,
# 7.2), run unmodified on a simulated MI300X, each giving what every other
# gives.
#
# Each release links a HIP runtime of its own name (libamdhip64.so.6, then
# .so.7), binds each call to that release's symbol versions, and carries its
# own device library -- the printf and the grid barrier are that release's.
# The binaries are checked in
# (amd/tests/hipcc/rocm/<release>/, rebuilt by its build.sh wherever the
# releases are installed), so this runs without any ROCm.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim"
dir="$root/amd/tests/hipcc/rocm"
[[ -e "$shim/libamdhip64.so.7" ]] || { echo "SKIP: no HIP shim in $shim"; exit 0; }
fail=0
# A sanitizer-instrumented shim needs its runtime loaded first.
preload=""
if command -v objdump >/dev/null; then
  preload=$(objdump -p "$shim/libamdhip64.so.7" 2>/dev/null | awk '/NEEDED/ && /lib(asan|tsan)\.so/ {print $2}')
fi
runs=0
for exe in "$dir"/*/*.gfx942; do
  release=$(basename "$(dirname "$exe")") program=$(basename "$exe" .gfx942)
  if [[ -d "$root/.git" ]] && ! git -C "$root" ls-files --error-unmatch "${exe#$root/}" >/dev/null 2>&1; then
    echo "FAIL  ROCm $release's $program is not in the repository"; fail=1; continue
  fi
  # The shim first: a binary's RUNPATH may name the release's own runtime,
  # which LD_LIBRARY_PATH is searched ahead of.
  out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 LD_PRELOAD="$preload" LD_LIBRARY_PATH="$shim" \
        timeout 300 "$exe" 2>&1)
  status=$?
  runs=$((runs + 1))
  # What every release gives, unless the release's own directory says
  # otherwise.
  expected="$dir/$program.expected"
  [[ -e "$dir/$release/$program.expected" ]] && expected="$dir/$release/$program.expected"
  if [[ $status -eq 0 ]] && diff -q <(echo "$out") "$expected" >/dev/null; then
    echo "ok    ROCm $release's $program"
  else
    echo "FAIL  ROCm $release's $program (exit $status)"
    diff <(echo "$out") "$expected" | head -6 | sed 's/^/      /'
    fail=1
  fi
done
[[ $runs -gt 0 ]] || { echo "FAIL  no programs under $dir"; fail=1; }
exit $fail
