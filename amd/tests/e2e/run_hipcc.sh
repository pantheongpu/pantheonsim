#!/usr/bin/env bash
# A program built by hipcc, run unmodified on a simulated AMD GPU.
#
# The executable carries its device code inside itself and reaches the
# runtime the way hipcc arranges: it registers the code before main, launches
# kernels with the chevron syntax, and reads the device through the real HIP
# headers' structures. It was built by amd/tests/hipcc/build.sh, which needs
# ROCm; running it needs only VirtualGPU's libamdhip64.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim"
exe="$root/amd/tests/hipcc/chevron.gfx942"
[[ -e "$shim/libamdhip64.so.7" ]] || { echo "SKIP: no HIP shim in $shim"; exit 0; }
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}

# The executable has to be the one in the repository: a fixture only present
# where it was built passes here and fails for everyone who clones.
if git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
  expect "the hipcc-built executable is in the repository" "yes" \
    "$(git -C "$root" ls-files --error-unmatch amd/tests/hipcc/chevron.gfx942 >/dev/null 2>&1 && echo yes || echo no)"
fi

# A sanitizer-instrumented shim needs its runtime loaded before anything else,
# and this program was built long before, without it -- so the runtime the
# shim was linked against is preloaded, by the exact name the shim asks for.
preload=""
if command -v objdump >/dev/null; then
  preload=$(objdump -p "$shim/libamdhip64.so.7" 2>/dev/null | awk '/NEEDED/ && /lib(asan|tsan)\.so/ {print $2}')
fi
export LD_PRELOAD="$preload"

# The shim is loaded under the name ROCm 7 gives the library, and the program
# binds each call to the version the real library defines it with.
need=$(readelf -d "$exe" | sed -n 's/.*Shared library: \[\(libamdhip64[^]]*\)\].*/\1/p')
expect "the program asks for the ROCm 7 library" "libamdhip64.so.7" "$need"

out=$(VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" "$exe" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "it runs to the end" "0" "$status"
expect "it reads the device through the real headers' layout" \
  "device AMD Instinct MI300X gfx942 warp 64 CUs 304 threads/CU 2048 L2 4194304" \
  "$(grep -o '^device .*' <<< "$out")"
expect "a chevron launch computes every element" "chevron launch wrong 0 of 3000" \
  "$(grep -o 'chevron launch wrong .*' <<< "$out")"
expect "a capture runs nothing and each replay runs the launch once" "graph captured 0 replayed 6" \
  "$(grep -o 'graph captured .*' <<< "$out")"
expect "a peer is reachable, once, and a copy to it arrives intact" \
  "peer can 1, enabling twice says hipErrorPeerAccessAlreadyEnabled, copy intact 1" \
  "$(grep -o 'peer can .*' <<< "$out")"

# Device-side printf: three lanes print through the hostcall buffer -- one of
# them a string that takes several packets -- and a __constant__ table is read
# from the program's own code object. printf's return value is checked by the
# program against the host's own snprintf.
out=$(VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/printf.gfx942" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "a kernel that prints runs to the end" "0" "$status"
expect "each lane's printf comes out whole, in lane order" \
  "lane 0 of 64: short, beef,  0.00, 2.500000e-03, a, 1099511627776, 100%, prime 7|lane 1 of 64: a string long enough that it takes more than one packet to carry it, bef0,  1.50, 2.500000e-03, b, 1099511627776, 100%, prime 11|lane 2 of 64: short, bef1,  3.00, 2.500000e-03, c, 1099511627776, 100%, prime 13" \
  "$(grep '^lane ' <<< "$out" | paste -sd'|')"
expect "and printf returns what it printed" "printf returned what it printed for 3 of 3 lanes" \
  "$(grep -o 'printf returned .*' <<< "$out")"

# The same program on a device of another target is told so by name rather
# than handed code it cannot run.
out=$(VGPU_GPU=amd/mi350x LD_LIBRARY_PATH="$shim" "$exe" 2>&1)
expect "a device of another target is refused by name" "yes" \
  "$(grep -q 'carries device code for gfx942, and this device is gfx950' <<< "$out" && echo yes || echo no)"
exit $fail
