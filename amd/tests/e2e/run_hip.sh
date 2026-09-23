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
"$cc" -O1 $sanitize -I"$root/amd/include" "$root/amd/tests/e2e/vector_add_hip.c" -o "$tmp/vector_add_hip" \
  -L"$shim" -lamdhip64 -Wl,-rpath,"$shim" 2>"$tmp/cc.err"
expect "a HIP program links against the shim" "yes" \
  "$([[ -x "$tmp/vector_add_hip" ]] && echo yes || echo "no: $(head -3 "$tmp/cc.err")")"
[[ -x "$tmp/vector_add_hip" ]] || exit 1

out=$(VGPU_GPU=amd/mi300x "$tmp/vector_add_hip" "$root/amd/tests/data/vector_add.gfx942.o" 2>&1)
status=$?
echo "$out" | sed 's/^/      /'
expect "it runs and every element is right" "0" "$status"
expect "it sees the GPU the profile describes" "device AMD Instinct MI300X gfx942 warp 64" \
  "$(grep -o '^device AMD Instinct MI300X gfx942 warp 64' <<< "$out")"
expect "and no element came out wrong" "wrong 0 of 1000" "$(grep -o 'wrong 0 of 1000' <<< "$out")"
expect "a kernel the code object does not have is refused" "missing kernel refused 1" \
  "$(grep -o 'missing kernel refused 1' <<< "$out")"
expect "and so is a launch with no work-items" "empty launch refused 1" \
  "$(grep -o 'empty launch refused 1' <<< "$out")"

# The same program on an NVIDIA profile: HIP runs on AMD GPUs, and saying so
# is better than running the kernel on a card that could not have run it.
out=$(VGPU_GPU=nvidia/h100 "$tmp/vector_add_hip" "$root/amd/tests/data/vector_add.gfx942.o" 2>&1)
expect "an NVIDIA profile is refused by name" "yes" \
  "$(grep -q "is not an AMD GPU" <<< "$out" && echo yes || echo no)"
exit $fail
