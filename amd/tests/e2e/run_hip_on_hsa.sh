#!/usr/bin/env bash
# ROCm's own HIP runtime on VirtualGPU's HSA runtime: the hipcc-built programs
# in amd/tests/hipcc run with each installed release's libamdhip64 (CLR)
# unmodified, and only libhsa-runtime64 is VirtualGPU's. CLR then does all
# of HIP itself -- its copies and fills are its own kernels in AQL packets,
# device printf comes back through its hostcall listener on a signal the
# kernel rings, a cooperative launch goes to a cooperative queue -- so what
# passes here is the HSA layer doing what ROCm's runtime needs of it.
#
# The releases are those found at VGPU_ROCM_LIBS (one lib directory),
# /opt/rocm/lib, or unpacked under ~/.local/share/rocm-*/opt/rocm-*/lib.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$(cd "$build/shim" 2>/dev/null && pwd)"
[[ -e "$shim/libhsa-runtime64.so.1" ]] || { echo "SKIP: no HSA runtime in $build/shim"; exit 0; }
if objdump -p "$shim/libhsa-runtime64.so.1" 2>/dev/null | grep -q 'NEEDED.*lib[at]san'; then
  echo "SKIP: a sanitizer build, and ROCm's HIP is not built with one"; exit 0
fi
libs=()
if [[ -n "${VGPU_ROCM_LIBS:-}" ]]; then libs=("$VGPU_ROCM_LIBS")
else
  for d in /opt/rocm/lib $(ls -d "$HOME"/.local/share/rocm-7*/opt/rocm-*/lib 2>/dev/null | sort -V); do
    [[ -e "$d/libamdhip64.so.7" ]] && libs+=("$d")
  done
fi
[[ ${#libs[@]} -gt 0 ]] || { echo "SKIP: no ROCm libamdhip64 to run (set VGPU_ROCM_LIBS)"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# Only libhsa-runtime64 is VirtualGPU's; ROCm's HIP comes first for the rest.
ln -s "$shim/libhsa-runtime64.so.1" "$tmp/libhsa-runtime64.so.1"
ln -s "$shim/libhsa-runtime64.so.1" "$tmp/libhsa-runtime64.so"
bin="$root/amd/tests/hipcc"
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$2" == "$3" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
run() {   # run <lib dir> <program>
  VGPU_QUIET=1 VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$tmp:$1" timeout 300 "$bin/$2.gfx942" 2>&1
}
for lib in "${libs[@]}"; do
  release=$(ls "$lib"/libamdhip64.so.7.* 2>/dev/null | head -1 | sed 's/.*libamdhip64.so.//')
  echo "--    ROCm's HIP $release ($lib)"
  out=$(run "$lib" chevron)
  expect "  a chevron launch computes every element" "chevron launch wrong 0 of 3000" \
    "$(grep -o '^chevron launch.*' <<< "$out")"
  expect "  a capture runs nothing and each replay runs the launch once" "graph captured 0 replayed 6" \
    "$(grep -o '^graph captured.*' <<< "$out")"
  expect "  a peer is reachable, and a copy to it arrives intact" \
    "peer can 1, enabling twice says hipErrorPeerAccessAlreadyEnabled, copy intact 1" \
    "$(grep -o '^peer can.*' <<< "$out")"
  out=$(run "$lib" runtime)
  expect "  a kernel on one device reads and writes another's memory" \
    "a kernel on device 0 read and wrote device 1's memory right for 256 of 256 elements" \
    "$(grep -o '^a kernel on device 0.*' <<< "$out")"
  expect "  the last error is kept until it is read" \
    "the last error outlives a call that succeeds: hipErrorInvalidDevice, hipErrorInvalidDevice, then hipSuccess" \
    "$(grep -o '^the last error outlives.*' <<< "$out")"
  out=$(run "$lib" printf)
  expect "  device printf comes back through ROCm's hostcall listener" \
    "printf returned what it printed for 3 of 3 lanes" "$(grep -o '^printf returned.*' <<< "$out")"
  out=$(run "$lib" cooperative)
  expect "  a cooperative launch passes its grid barrier" \
    "after the grid barrier, 16 of 16 groups saw every group's value" "$(grep -o '^after the grid barrier.*' <<< "$out")"
  out=$(run "$lib" fp8)
  expect "  the 8-bit float conversions are right, all 16 kinds" "16" "$(grep -c ': 0 of [0-9]* wrong$' <<< "$out")"
  # The last of streams' checks is a kernel that faults, which ROCm's HIP
  # answers by ending the program, as it does on a card; the six before it
  # are what is checked here.
  out=$(run "$lib" streams)
  expect "  streams run at once, wait for events and keep their order" "6" \
    "$(grep -cE '^(two kernels|a stream is busy|a stream waits|the null stream|a host function|an asynchronous copy).* 1$' <<< "$out")"
done
exit $fail
