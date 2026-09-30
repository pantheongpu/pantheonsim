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
expect "a kernel that failed is reported by the blocking copy after it, once" \
  "a failed launch is reported by the copy after it: hipErrorLaunchFailure, then hipSuccess" \
  "$(grep -o '^a failed launch is reported.*' <<< "$out")"

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

# gfx950's block-scaled matrix instructions (fp8, bf8, fp6, bf6 and fp4, each
# lane's scale byte chosen), v_prng_b32 and v_permlane32_swap_b32, against what
# the CDNA4 ISA guide says they compute (hipcc/gfx950.cpp).
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi350x LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/gfx950.gfx950" 2>&1)
status=$?
expect "the gfx950 program runs to the end" "0" "$status"
expect "each block-scaled product, of every format pairing, is what the ISA says" "6 of 6" \
  "$(grep -c '^[0-9x]* formats .*: 0 of [0-9]* wrong$' <<< "$out") of 6"
expect "v_prng_b32 steps the ISA's LFSR" "v_prng_b32: 0 of 64 wrong" "$(grep -o '^v_prng_b32:.*' <<< "$out")"
expect "v_permlane32_swap_b32 trades the halves of two registers" "v_permlane32_swap_b32: 0 of 128 wrong" \
  "$(grep -o '^v_permlane32_swap_b32:.*' <<< "$out")"

# The sparse matrix instructions (v_smfmac_*): A 2:4 sparse, its indices'
# set chosen by CBSZ and ABID, against what the CDNA3 and CDNA4 ISA guides
# say they compute (hipcc/smfmac.cpp) -- gfx942's on an MI300X, gfx950's (and
# some of gfx942's, with the OCP 8-bit floats) on an MI350X.
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/smfmac.gfx942" 2>&1)
status=$?
expect "the gfx942 sparse matrix program runs to the end" "0" "$status"
expect "each gfx942 sparse product is what the ISA says" "12 of 12" \
  "$(grep -c '^[fi]32_.* cbsz:.*: 0 of [0-9]* wrong$' <<< "$out") of 12"
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi350x LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/smfmac.gfx950" 2>&1)
status=$?
expect "the gfx950 sparse matrix program runs to the end" "0" "$status"
expect "each gfx950 sparse product is what the ISA says" "22 of 22" \
  "$(grep -c '^[fi]32_.* cbsz:.*: 0 of [0-9]* wrong$' <<< "$out") of 22"

# MODE's floating-point modes (hipcc/numerics.cpp): what a kernel's
# descriptor starts it with -- denormals kept by default, single-precision
# ones flushed when built with -fgpu-flush-denormals-to-zero -- and each
# round mode a kernel sets, against the host's IEEE arithmetic.
for b in numerics numerics.flush; do
  out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/$b.gfx942" 2>&1)
  status=$?
  expect "the $b program runs to the end" "0" "$status"
  expect "every $b check holds (MODE, denormals, the four round modes)" "23 of 23" \
    "$(grep -c '^ok ' <<< "$out") of 23"
done

# RDNA3 on a simulated Radeon RX 7900 XTX (hipcc/rdna3.cpp): VOPD reading
# both halves' sources first, WMMA in each type against the host, sin() in
# double (whose reduction takes 64-bit literals), and -- built wave64 -- DPP
# with bank masks, row mirrors and output modifiers.
out=$(VGPU_QUIET=1 VGPU_GPU=amd/rx7900xtx LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/rdna3.gfx1100" 2>&1)
status=$?
expect "the RDNA3 wave32 program runs to the end" "0" "$status"
expect "every RDNA3 wave32 check holds (VOPD, WMMA, double literals)" "6 of 6" \
  "$(grep -c ': 0 of [0-9]* wrong$' <<< "$out") of 6"
out=$(VGPU_QUIET=1 VGPU_GPU=amd/rx7900xtx LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/rdna3.w64.gfx1100" 2>&1)
status=$?
expect "the RDNA3 wave64 program runs to the end" "0" "$status"
expect "every RDNA3 wave64 check holds (DPP, output modifiers, double literals)" "2 of 2" \
  "$(grep -c ': 0 of [0-9]* wrong$' <<< "$out") of 2"

# Arrays, texture objects and surfaces on each Radeon generation
# (hipcc/images.cpp): samples, filtering, address modes, gathers, linear,
# pitched, 3D and layered textures, surfaces, the copies to and from arrays
# and the API's answers -- the same program built for gfx1030, gfx1100 and
# gfx1201, whose image resources are laid out differently.
for run in "gfx1030 rx6900xt" "gfx1100 rx7900xtx" "gfx1201 rx9070xt"; do
  set -- $run
  out=$(VGPU_QUIET=1 VGPU_GPU=amd/$2 LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/images.$1" 2>&1)
  status=$?
  expect "the $1 image program runs to the end" "0" "$status"
  expect "every $1 texture and surface check holds" "18 of 18" \
    "$(grep -c ': 0 of [0-9]* wrong$' <<< "$out") of 18"
done

# RDNA4 on a simulated Radeon RX 9070 XT (hipcc/rdna4.cpp): WMMA in gfx12's
# layout (f16, bf16, OCP fp8, iu8), the scalar float unit and its
# transcendentals, and the split barrier.
out=$(VGPU_QUIET=1 VGPU_GPU=amd/rx9070xt LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/rdna4.gfx1201" 2>&1)
status=$?
expect "the RDNA4 program runs to the end" "0" "$status"
expect "every RDNA4 check holds (WMMA, scalar floats, split barrier, SCHED_MODE)" "8 of 8" \
  "$(grep -c ': 0 of [0-9]* wrong$' <<< "$out") of 8"

# RDNA2 on a simulated Radeon RX 6900 XT (hipcc/rdna2.cpp): SDWA's partial
# reads and writes, registers indexed through M0, permlane16 and permlanex16,
# DPP's row_share and row_xmask, and the stack reached through a flat pointer
# (FLAT_SCRATCH set first) -- in wave32 and wave64.
for w in "" .w64; do
  out=$(VGPU_QUIET=1 VGPU_GPU=amd/rx6900xt LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/rdna2$w.gfx1030" 2>&1)
  status=$?
  expect "the RDNA2${w:+ wave64} program runs to the end" "0" "$status"
  expect "every RDNA2${w:+ wave64} check holds (SDWA, M0, permlane16, DPP, FLAT_SCRATCH)" "12 of 12" \
    "$(grep -c ': 0 of [0-9]* wrong$' <<< "$out") of 12"
done

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

# HIP's calls beyond the everyday ones (hipcc/api.cpp): device flags and
# UUIDs, contexts, the legacy and per-thread default streams, callbacks,
# waiting on memory in a stream, every launch form. Each is held to what
# ROCm's HIP answers, and the program counts the checks that did not hold.
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" timeout 300 \
      "$(dirname "$exe")/api.gfx942" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "the API program runs to the end" "0" "$status"
expect "every one of its checks holds" "0 failed" "$(grep -o '[0-9]* failed$' <<< "$out")"

# HIP's copies, fills and allocations of every shape (hipcc/memory.cpp):
# pitched memory, 2D and 3D copies with offsets, the driver API's forms,
# memsets of each width, who waits for what, peer copies, managed memory's
# advice, pools (exported and imported through a file descriptor), virtual
# memory at page granularity and graph memory's counters.
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" timeout 300 \
      "$(dirname "$exe")/memory.gfx942" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "the memory program runs to the end" "0" "$status"
expect "every one of its checks holds" "0 failed" "$(grep -o '[0-9]* failed$' <<< "$out")"

# Graphs and stream capture (hipcc/graphs.cpp): graphs built node by node and
# captured across streams, executable graphs and what changes them, capture
# modes and what they refuse, event and memory nodes, user objects. It
# writes a drawing of a graph into the working directory, so it runs in one
# of its own.
graph_dir=$(mktemp -d)
graph_shim=$(cd "$shim" && pwd)
out=$(cd "$graph_dir" && VGPU_QUIET=1 VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$graph_shim" timeout 300 \
      "$(dirname "$exe")/graphs.gfx942" 2>&1)
status=$?
rm -rf "$graph_dir"
echo "$out" | sed 's/^/      /'
expect "the graphs program runs to the end" "0" "$status"
expect "every one of its graph checks holds" "0 failed" "$(grep -o '[0-9]* failed$' <<< "$out")"

# The rest of what ROCm's library exports (hipcc/exports.cpp): __managed__
# variables, a code object loaded as a library, a fat binary and a link's
# input, HCC's launch by its C and C++ names, half conversions, and what a
# device with no OpenGL and no dma-bufs answers.
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" timeout 300 \
      "$(dirname "$exe")/exports.gfx942" "$(dirname "$exe")/exports_kernel.gfx942.co" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "the exports program runs to the end" "0" "$status"
expect "every one of its export checks holds" "0 failed" "$(grep -o '[0-9]* failed$' <<< "$out")"

# Modules (hipcc/modules.cpp): generic code (gfx9-4-generic) loaded and run on
# gfx942, and the module calls' answers to hip-tests' negative cases.
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" timeout 300 \
      "$(dirname "$exe")/modules.gfx942" "$(dirname "$exe")/modules_kernel.generic.co" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "the modules program runs to the end" "0" "$status"
expect "every one of its module checks holds" "0 failed" "$(grep -o '[0-9]* failed$' <<< "$out")"

# Atomics (hipcc/atomics.cpp): a flat atomic landing in LDS, a kernel's and a
# host thread's atomics on one pinned counter, and a float max.
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$shim" timeout 300 "$(dirname "$exe")/atomics.gfx942" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "the atomics program runs to the end" "0" "$status"
expect "every one of its atomic checks holds" "0 failed" "$(grep -o '[0-9]* failed$' <<< "$out")"

# Events shared between processes (hipcc/ipc.cpp): an interprocess event's
# handle opened in a process it forks, whose wait waits for the record made
# here.
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$shim" timeout 120 "$(dirname "$exe")/ipc.gfx942" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "the interprocess events program runs to the end" "0" "$status"
expect "every one of its interprocess checks holds" "0 failed" "$(grep -o '[0-9]* failed$' <<< "$out")"

# Every error code's name and text, from both the runtime's and the driver's
# forms, as ROCm 7.1's own library gives them, line for line.
out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x LD_LIBRARY_PATH="$shim" "$(dirname "$exe")/errors.gfx942" 2>&1)
expect "every error's name and text are ROCm's" "same" \
  "$(diff -q <(echo "$out") "$(dirname "$exe")/rocm/errors.expected" >/dev/null && echo same ||
     diff <(echo "$out") "$(dirname "$exe")/rocm/errors.expected" | head -4 | tr '\n' ' ')"

# The devices a program is shown: ROCR_VISIBLE_DEVICES picks from four, then
# HIP_VISIBLE_DEVICES from those, each read up to its first bad entry; none
# shown is no device.
visible() { env "$@" VGPU_QUIET=1 VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=4 LD_LIBRARY_PATH="$shim" \
            "$(dirname "$exe")/visible.gfx942" 2>&1; }
expect "HIP_VISIBLE_DEVICES=0,2 shows two devices" "hipSuccess 2" "$(visible HIP_VISIBLE_DEVICES=0,2)"
expect "ROCR_VISIBLE_DEVICES=3 shows one" "hipSuccess 1" "$(visible ROCR_VISIBLE_DEVICES=3)"
expect "a list stops at its first device that is not there" "hipSuccess 1" "$(visible HIP_VISIBLE_DEVICES=1,9,0)"
expect "HIP's list picks from ROCr's" "hipSuccess 1" "$(visible ROCR_VISIBLE_DEVICES=1,2 HIP_VISIBLE_DEVICES=1)"
expect "an empty list shows none" "hipErrorNoDevice 0" "$(visible HIP_VISIBLE_DEVICES=)"
# A device named by its UUID: the HSA agent's, or KFD's unique_id in hex,
# which is how Ollama names a ROCm GPU in ROCR_VISIBLE_DEVICES. unique_id is
# the 64-bit FNV-1a of the device's UUID string (amd/src/kfd.cpp), and that
# string comes from the profile's id (describe_device, src/core/telemetry.cpp).
kfd_id=$(python3 - <<'PY'
h = 2166136261
for c in b"amd/mi300x": h = ((h ^ c) * 16777619) & 0xFFFFFFFF
u = "GPU-%08x-%04x-%04x-%04x-%08x%04x" % (h, (h >> 16) & 0xFFFF, 0x4000 | (h & 0x0FFF),
                                          0x8000 | ((h >> 4) & 0x3FFF), (h * 2654435761) & 0xFFFFFFFF, 2)
f = 1469598103934665603
for c in u.encode(): f = ((f ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
print("GPU-%016x" % f)
PY
)
expect "ROCR_VISIBLE_DEVICES names a device by its HSA UUID" "hipSuccess 1" "$(visible ROCR_VISIBLE_DEVICES=GPU-5647505500000002)"
expect "ROCR_VISIBLE_DEVICES names a device by KFD's unique_id, as Ollama does" "hipSuccess 1" "$(visible ROCR_VISIBLE_DEVICES=$kfd_id)"
expect "a UUID no device has shows none" "hipErrorNoDevice 0" "$(visible ROCR_VISIBLE_DEVICES=GPU-00000000deadbeef)"
# CUDA_VISIBLE_DEVICES stands in for HIP_VISIBLE_DEVICES when that is unset,
# as ROCm's HIP reads it: -1 hides every GPU, unless HIP_VISIBLE_DEVICES says.
expect "CUDA_VISIBLE_DEVICES=-1 hides every GPU from HIP" "hipErrorNoDevice 0" "$(visible CUDA_VISIBLE_DEVICES=-1)"
expect "HIP_VISIBLE_DEVICES comes before CUDA_VISIBLE_DEVICES" "hipSuccess 1" \
  "$(visible HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=-1)"

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
