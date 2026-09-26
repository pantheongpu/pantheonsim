#!/usr/bin/env bash
# A HIP program on a simulated AMD GPU, end to end: the host program links
# against VirtualGPU's libamdhip64 the way it would link against AMD's, loads
# a gfx942 code object, launches the kernel in it, and checks the answer.
#
# Nothing here needs ROCm: the host side is plain C, and the code object was
# built by clang (amd/tests/data/build.sh).
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim"
[[ -e "$shim/libamdhip64.so" ]] || { echo "SKIP: no HIP shim in $shim"; exit 0; }
cc=${CC:-cc}
command -v "$cc" >/dev/null || { echo "SKIP: no C compiler"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state"
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}

# A sanitizer-instrumented shim needs its runtime first in the link order, so
# the program is built with the sanitizer the shim was built with. Read from
# the library's own linkage, so no flag has to be remembered here.
sanitize=""
if command -v objdump >/dev/null; then
  case "$(objdump -p "$shim/libamdhip64.so" 2>/dev/null | awk '/NEEDED/ {print $2}')" in
    *libtsan*) sanitize="-fsanitize=thread -g" ;;
    *libasan*) sanitize="-fsanitize=address,undefined -fno-omit-frame-pointer -g" ;;
  esac
fi
build() {  # build <name>
  "$cc" -O1 $sanitize -I"$root/amd/include" "$root/amd/tests/e2e/$1.c" -o "$tmp/$1" \
    -L"$shim" -lamdhip64 -Wl,-rpath,"$shim" 2>"$tmp/$1.err"
}
build vector_add_hip
expect "a HIP program links against the shim" "yes" \
  "$([[ -x "$tmp/vector_add_hip" ]] && echo yes || echo "no: $(head -3 "$tmp/vector_add_hip.err")")"
[[ -x "$tmp/vector_add_hip" ]] || exit 1

out=$(VGPU_GPU=amd/mi300x "$tmp/vector_add_hip" "$root/amd/tests/data/vector_add.gfx942.o" \
        "$root/amd/tests/data/globals.gfx942.o" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "it runs and every element is right" "0" "$status"
expect "it sees the GPU the profile describes" "device AMD Instinct MI300X gfx942:sramecc+:xnack- warp 64" \
  "$(grep -o '^device AMD Instinct MI300X gfx942:sramecc+:xnack- warp 64' <<< "$out")"
expect "and no element came out wrong" "wrong 0 of 1000" "$(grep -o 'wrong 0 of 1000' <<< "$out")"
expect "a module's variable is where hipModuleGetGlobal says" "scale is 4 bytes" \
  "$(grep -o 'scale is 4 bytes' <<< "$out")"
expect "and what the host writes there is what the kernel reads" "global scale applied 1" \
  "$(grep -o 'global scale applied 1' <<< "$out")"
expect "and the host reads back what the kernel wrote into it" "kernel wrote its own array 1" \
  "$(grep -o 'kernel wrote its own array 1' <<< "$out")"
expect "a kernel the code object does not have is refused" "missing kernel refused 1" \
  "$(grep -o 'missing kernel refused 1' <<< "$out")"
expect "and so is a launch with no work-items" "empty launch refused 1" \
  "$(grep -o 'empty launch refused 1' <<< "$out")"

# What a program asks of the runtime besides a launch: how much memory the
# device has, work issued on a stream, LDS the launch pays for, and a second
# device that keeps its own memory.
build runtime_hip
expect "the runtime program links against the shim" "yes" \
  "$([[ -x "$tmp/runtime_hip" ]] && echo yes || echo "no: $(head -3 "$tmp/runtime_hip.err")")"
if [[ -x "$tmp/runtime_hip" ]]; then
  out=$(VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 "$tmp/runtime_hip" "$root/amd/tests/data/memory.gfx942.o" \
    "$root/amd/tests/data/asm_memory.gfx942.o" 2>&1)
  status=$?
  echo "$out" | sed 's/^/      /'
  expect "it runs" "0" "$status"
  for line in \
    "an allocation costs what it asked for 1" \
    "a kernel reading past its arguments runs 1" \
    "and freeing it gives that back 1" \
    "a copy on a stream lands 1" \
    "a kernel whose LDS the launch paid for 1" \
    "a launch that forgets it is refused 1" \
    "a kernel takes a measurable time 1" \
    "an event with nothing recorded has no time 1" \
    "and neither has one created without timing 1" \
    "an event is queried and destroyed 1" \
    "a stream is created with the flags it asked for 1" \
    "devices 2" \
    "a second device keeps its own memory 1" \
    "and what it holds is not missing from the first 1"; do
    expect "$line" "$line" "$(grep -Fo "$line" <<< "$out")"
  done
fi

# What ROCm's libraries ask of HIP when PyTorch loads them: streams with a
# priority or a compute-unit mask, host memory a kernel reaches, pools,
# virtual memory mapped by hand, and the rest (libraries_hip.c).
build libraries_hip
expect "the libraries program links against the shim" "yes" \
  "$([[ -x "$tmp/libraries_hip" ]] && echo yes || echo "no: $(head -3 "$tmp/libraries_hip.err")")"
if [[ -x "$tmp/libraries_hip" ]]; then
  out=$(VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 "$tmp/libraries_hip" "$root/amd/tests/data/memory.gfx942.o" 2>&1)
  status=$?
  echo "$out" | sed 's/^/      /'
  expect "it runs" "0" "$status"
  for line in \
    "a stream keeps its priority, flags and device 1" \
    "a stream keeps its compute-unit mask, and the null stream has every unit 1" \
    "a destroyed stream is no longer known 1" \
    "a kernel reads and writes registered host memory 1" \
    "registered memory is host memory, at its own address 1" \
    "registering it twice is refused 1" \
    "and unregistering what is not registered 1" \
    "a kernel reads and writes pinned host memory 1" \
    "a kernel reads and writes managed memory, which says it is managed 1" \
    "a pool counts what is in use and its high mark 1" \
    "virtual memory mapped by hand is written and read back 1" \
    "an address knows its allocation and its device 1" \
    "a value written in stream order lands 1" \
    "each device is found by its PCI address 1" \
    "a limit is kept, and one that cannot be set is refused 1" \
    "a kernel is named dyn_lds" \
    "a host function runs in stream order 1" \
    "a capture is active, with an id, until it ends 1" \
    "an IPC handle is made, and its own process may not open it 1"; do
    expect "$line" "$line" "$(grep -Fo "$line" <<< "$out")"
  done
fi

# The same program on an NVIDIA profile: HIP runs on AMD GPUs, and saying so
# is better than running the kernel on a card that could not have run it.
out=$(VGPU_GPU=nvidia/h100 "$tmp/vector_add_hip" "$root/amd/tests/data/vector_add.gfx942.o" 2>&1)
expect "an NVIDIA profile is refused by name" "yes" \
  "$(grep -q "is not an AMD GPU" <<< "$out" && echo yes || echo no)"
exit $fail
