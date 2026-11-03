#!/usr/bin/env bash
# `vgpu run` on programs that need libcudss.so.1 (built against cuDSS 1.0, as SCS's
# run_tests_cudss is) and libcudss.so.0 (cuDSS 0.8): each resolves to the shim's library of that
# major, which reports its own version. Before the 1.0 shim, the first stopped with
# "needs libcudss.so.1, but this build of the simulator has no libcudss.so".
#
#   run_cudss_sonames.sh <probe-for-0.8> <probe-for-1.0>
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
if ! command -v readelf >/dev/null 2>&1; then echo "SKIP: readelf not found"; exit 0; fi
[[ -e "$build/shim/libcudss.so.0" ]] || { echo "SKIP: the cuDSS shim is not built"; exit 0; }
fail=0
check() { # check <label> <probe> <soname> <expected output fragment>
  local label="$1" probe="$2" soname="$3" want="$4"
  if ! readelf -d "$probe" | grep -q "NEEDED.*\[$soname\]"; then
    echo "FAIL  $label: the program does not need $soname"; readelf -d "$probe" | grep NEEDED; fail=1; return
  fi
  if readelf -d "$probe" | grep -qE "RPATH|RUNPATH"; then
    echo "FAIL  $label: the probe carries a library search path, so it would not prove the launcher's"; fail=1; return
  fi
  local out rc
  out=$(env -u LD_LIBRARY_PATH VGPU_GPU=nvidia/t4 "$vgpu" run "$probe" 2>&1); rc=$?
  if [[ $rc -eq 0 && "$out" == *"$want"* && "$out" != *"needs lib"* ]]; then echo "ok    $label"; else
    echo "FAIL  $label (exit $rc)"; echo "      expected to contain: $want"; echo "      actual: $out"; fail=1; fi
}
check "vgpu run: a program needing libcudss.so.0 gets cuDSS 0.8" "$1" libcudss.so.0 "library 0.8, default matching 0"
check "vgpu run: a program needing libcudss.so.1 gets cuDSS 1.0" "$2" libcudss.so.1 "library 1.0, default matching 6"
# Run without the launcher, the shim's directory named by hand: the same answers.
for pair in "$1:libcudss.so.0:library 0.8" "$2:libcudss.so.1:library 1.0"; do
  IFS=: read -r probe soname want <<<"$pair"
  out=$(LD_LIBRARY_PATH="$build/shim" VGPU_GPU=nvidia/t4 "$probe" 2>&1); rc=$?
  if [[ $rc -eq 0 && "$out" == *"$want"* ]]; then echo "ok    LD_LIBRARY_PATH=shim: $soname"; else
    echo "FAIL  LD_LIBRARY_PATH=shim: $soname (exit $rc): $out"; fail=1; fi
done
exit $fail
