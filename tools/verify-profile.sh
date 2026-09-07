#!/usr/bin/env bash
# Check a profile's `verified: true` by running the conformance suite on the
# simulator and diffing it against the same binary's output on the physical
# device.
#
#   tools/verify-profile.sh nvidia/l4 [ref-dir]
#
# characterize-aws.sh and characterize-cloud.sh bring back two reference files
# per device -- the output of ptx_semantics and control_flow run on the real
# card. Nothing compared them: the diff was a step someone had to remember, and
# a flag that depends on a remembered step is a flag that will eventually be
# wrong. This is that step, written down.
#
# It needs an nvcc whose toolkit matches the shim, and it compiles for the
# profile's own architecture -- PTX runs forward onto newer hardware, never
# backward, so a profile has to be checked with code built for it.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
. "$here/tests/shim_guard.sh"

profile="${1:?usage: verify-profile.sh <vendor/model> [ref-dir]}"
refdir="${2:-${VGPU_CLOUD_OUT:-/tmp/vgpu-cloud}}"
build="${VGPU_BUILD_DIR:-$here/build}"
shim="$build/shim"
vgpu="$build/vgpu"

[[ -x "$vgpu" ]] || { echo "SKIP: $vgpu not built"; exit 0; }
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }

nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix

# The architecture to build for comes from the profile, not from a flag: that
# is the whole point of checking a profile rather than a machine.
cc="$("$vgpu" info --gpu "$profile" 2>/dev/null | sed -n 's/.*compute capability: *//p' | tr -d ' .')"
[[ -n "$cc" ]] || { echo "FAIL: no profile '$profile'"; exit 1; }

# Reference files are named after the instance type that produced them, which
# is not the profile id, so accept either and say which was used.
# Derive the stem from the reference files themselves rather than guessing at
# it: globbing the directory and appending the suffix matches nothing, because
# the glob already includes the suffixed names.
found=""
if [[ -e "$refdir/$(basename "$profile").ptx_semantics.ref.txt" ]]; then
  found="$refdir/$(basename "$profile")"
else
  shopt -s nullglob
  for f in "$refdir"/*.ptx_semantics.ref.txt; do found="${f%.ptx_semantics.ref.txt}"; break; done
  shopt -u nullglob
fi
[[ -n "$found" ]] || { echo "SKIP: no *.ptx_semantics.ref.txt under $refdir"; exit 0; }
echo "profile $profile (sm_$cc) against $(basename "$found").*.ref.txt"

work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-verify.XXXXXX")"
trap 'rm -rf "$work"' EXIT

fails=0
total=0
for t in ptx_semantics control_flow; do
  ref="${found}.${t}.ref.txt"
  [[ -s "$ref" ]] || { echo "  $t: no reference output"; fails=$((fails + 1)); continue; }
  # compute_XX rather than sm_XX: the simulator loads PTX, and -arch=sm_XX
  # alone embeds only SASS.
  if ! "$nvcc_bin" -std=c++14 -gencode "arch=compute_${cc},code=compute_${cc}" -cudart shared \
       -Wno-deprecated-gpu-targets "$here/tests/conformance/$t.cu" -o "$work/$t" 2>"$work/$t.build"; then
    echo "  $t: does not compile for sm_$cc"; sed 's/^/      /' "$work/$t.build" | head -5
    fails=$((fails + 1)); continue
  fi
  VGPU_QUIET=1 VGPU_GPU="$profile" LD_LIBRARY_PATH="$shim" "$work/$t" > "$work/$t.sim" 2>"$work/$t.err"
  n=$(wc -l < "$work/$t.sim")
  if diff -q "$work/$t.sim" "$ref" >/dev/null 2>&1; then
    printf "  %-15s %4s values identical to hardware\n" "$t" "$n"
    total=$((total + n))
  else
    bad=$(diff "$work/$t.sim" "$ref" | grep -c '^<')
    printf "  %-15s %s of %s values differ\n" "$t" "$bad" "$n"
    diff "$work/$t.sim" "$ref" | head -10 | sed 's/^/      /'
    fails=$((fails + 1))
  fi
done

if (( fails )); then
  echo "FAIL: $profile does not match hardware; 'verified: true' is not earned"
  exit 1
fi
echo "RESULT: $profile matches hardware on $total conformance values"
