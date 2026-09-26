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
expect "it reads the device through the real headers' layout, named as HIP names it" \
  "device AMD Instinct MI300X gfx942:sramecc+:xnack- warp 64 CUs 304 threads/CU 2048 L2 4194304" \
  "$(grep -o '^device .*' <<< "$out")"
expect "a chevron launch computes every element" "chevron launch wrong 0 of 3000" \
  "$(grep -o 'chevron launch wrong .*' <<< "$out")"
expect "a capture runs nothing and each replay runs the launch once" "graph captured 0 replayed 6" \
  "$(grep -o 'graph captured .*' <<< "$out")"
expect "a peer is reachable, once, and a copy to it arrives intact" \
  "peer can 1, enabling twice says hipErrorPeerAccessAlreadyEnabled, copy intact 1" \
  "$(grep -o 'peer can .*' <<< "$out")"

# The same program built unoptimized (-O0, what CMake builds HIP with when no
# build type is set) computes the same answers. Its device library calls are
# real calls, it spills scalars into lanes and reads them back with every lane
# off, and nearly every vector instruction is in its long form.
out=$(VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/chevron.O0.gfx942" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "unoptimized, it runs to the end" "0" "$status"
expect "unoptimized, a chevron launch computes every element" "chevron launch wrong 0 of 3000" \
  "$(grep -o 'chevron launch wrong .*' <<< "$out")"
expect "unoptimized, a capture runs nothing and each replay runs the launch once" "graph captured 0 replayed 6" \
  "$(grep -o 'graph captured .*' <<< "$out")"
expect "unoptimized, a copy to a peer arrives intact" \
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

# A cooperative launch: sixteen work-groups each publish a value, wait at a
# grid barrier, and sum what all of them published, then wait again before
# clearing it -- right only if no group got past a barrier early. A grid one
# work-group larger than the device holds at once is refused.
out=$(VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/cooperative.gfx942" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "a cooperative kernel runs to the end" "0" "$status"
expect "the device says it launches cooperatively, and how many groups a compute unit holds" \
  "cooperative launch 1, 16 work-groups of 128 a compute unit" "$(grep -o '^cooperative launch .*' <<< "$out")"
expect "no work-group passes the grid barrier before every one reaches it" \
  "after the grid barrier, 16 of 16 groups saw every group's value" "$(grep -o '^after the grid barrier.*' <<< "$out")"
expect "a grid too large to be resident at once is refused" \
  "a grid too large to be resident at once: hipErrorCooperativeLaunchTooLarge" "$(grep -o '^a grid too large.*' <<< "$out")"

# Occupancy, as ROCm's runtime works it out (clr's hip_platform.cpp) from each
# kernel's metadata. A gfx942 SIMD holds 8 waves, 512 vector registers in
# steps of 8, and a compute unit 64 KB of LDS; 4 SIMDs, waves of 64. So:
#   plain (4 VGPRs), 256 threads:   8 waves x 4 x 64 = 2048 threads, 8 groups
#   of 65 threads, a group of 2 waves:                  2048 / 128  = 16
#   with 20000 bytes of LDS:        65536 / 20000                   = 3
#   tiled (24576 bytes of LDS):     65536 / 24576                   = 2
#   wide (129 VGPRs, so 136):       512 / 136 = 3 waves, 768 / 256  = 3
# and the grid that fills the 304 compute units: plain at 1024 threads, 2 a
# compute unit, 608; wide held to 256 threads, 3 a compute unit, 912.
out=$(VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/runtime.gfx942" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "a program asking about its device and kernels runs to the end" "0" "$status"
expect "work-groups a compute unit holds come from the kernel's registers and LDS" \
  "work-groups a compute unit holds: plain 8, of 65 threads 16, with 20000 bytes of LDS 3, tiled 2, wide 3" \
  "$(grep -o '^work-groups a compute unit holds.*' <<< "$out")"
expect "the block size that fills the device" "filling the device: plain 608 of 1024, wide 912 of 256" \
  "$(grep -o '^filling the device.*' <<< "$out")"
expect "each device attribute is the property it names" "30 of 30 attributes agree with the properties" \
  "$(grep -o '^[0-9]* of [0-9]* attributes agree.*' <<< "$out")"
expect "an attribute of a device that is not there is refused" "a device that is not there: hipErrorInvalidDevice" \
  "$(grep -o '^a device that is not there.*' <<< "$out")"
expect "the last error is kept until it is read, past a call that succeeds" \
  "the last error outlives a call that succeeds: hipErrorInvalidDevice, hipErrorInvalidDevice, then hipSuccess" \
  "$(grep -o '^the last error outlives.*' <<< "$out")"
expect "a kernel reads and writes a peer's memory once peer access is enabled" \
  "a kernel on device 0 read and wrote device 1's memory right for 256 of 256 elements" \
  "$(grep -o '^a kernel on device 0.*' <<< "$out")"
expect "peer access disabled twice says it is not enabled" "disabling it again: hipErrorPeerAccessNotEnabled" \
  "$(grep -o '^disabling it again.*' <<< "$out")"

# gfx942's 8-bit floats (fp8 and bf8, without infinities or a negative zero):
# floats narrowed by the device's instructions, rounded to nearest and
# stochastically, and all 256 bytes widened, each checked in the program
# against HIP's software conversion, which is what AMD says the hardware does.
out=$(VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/fp8.gfx942" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "the 8-bit float program runs to the end" "0" "$status"
expect "every float narrows to the fp8 and bf8 HIP's header gives, every way" "12 of 12 ways right" \
  "$(grep -c '^floats to .*: 0 of 4096 wrong$' <<< "$out") of 12 ways right"
expect "every fp8 and bf8 widens to the float HIP's header gives" "4 of 4 ways right" \
  "$(grep -c '^[bf][fp]8 to floats.*: 0 of 256 wrong$' <<< "$out") of 4 ways right"

# gfx950's (MI350X), which are the OCP formats: E4M3 with a negative zero and
# no infinity, E5M2 with both. The same instructions, meaning these.
out=$(VGPU_GPU=amd/mi350x LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/fp8.gfx950" 2>&1)
status=$?
expect "gfx950's 8-bit float program runs to the end" "0" "$status"
expect "every float narrows to the OCP fp8 and bf8 HIP's header gives, every way" "12 of 12 ways right" \
  "$(grep -c '^floats to .*: 0 of 4096 wrong$' <<< "$out") of 12 ways right"
expect "every OCP fp8 and bf8 widens to the float HIP's header gives" "4 of 4 ways right" \
  "$(grep -c '^[bf][fp]8 to floats.*: 0 of 256 wrong$' <<< "$out") of 4 ways right"

# Streams that run at once, as a card's do (hipcc/streams.cpp). Each waiting
# kernel gives up after a bounded time, so a runtime that ran the streams one
# after another fails these rather than hanging.
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$shim" timeout 300 "$(dirname "$exe")/streams.gfx942" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "the streams program runs to the end" "0" "$status"
for line in \
  "two kernels on two streams run at once 1" \
  "a stream is busy while its kernel waits, and done after 1" \
  "a stream waits for another's event 1" \
  "the null stream waits for the blocking streams, not a non-blocking one 1" \
  "a host function runs in stream order 1" \
  "an asynchronous copy takes its bytes when it is called 1" \
  "a kernel's fault is told at the synchronization after it 1"; do
  expect "$line" "$line" "$(grep -Fo "$line" <<< "$out")"
done

# The same program on a device of another target is told so by name rather
# than handed code it cannot run.
out=$(VGPU_GPU=amd/mi350x LD_LIBRARY_PATH="$shim" "$exe" 2>&1)
expect "a device of another target is refused by name" "yes" \
  "$(grep -q 'carries device code for gfx942, and this device is gfx950' <<< "$out" && echo yes || echo no)"
# The texture API on a GPU with no texture units: what ROCm's HIP answers,
# line for line (the same file run_hip_on_hsa.sh holds ROCm's HIP to).
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/textures.gfx942" 2>&1)
expect "the texture API answers as ROCm's HIP does on a GPU without texture units" "same" \
  "$(diff -q <(echo "$out") "$(dirname "$exe")/rocm/textures.expected" >/dev/null && echo same ||
     diff <(echo "$out") "$(dirname "$exe")/rocm/textures.expected" | head -4 | tr '\n' ' ')"
exit $fail
