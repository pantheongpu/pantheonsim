#!/usr/bin/env bash
# An HSA program on VirtualGPU's libhsa-runtime64 (amd/tests/hsa/hsa_dispatch.c):
# agents, memory pools, an executable loaded from a code object, and kernels
# dispatched through AQL queues, on two simulated MI300Xs.
#
# It is built against VirtualGPU's own HSA header, and again against ROCm's
# where ROCm's headers are installed (VGPU_ROCM_INCLUDE, /opt/rocm/include, or
# ~/.local/share/rocm-*): both builds must pass, which is what says the header
# and the runtime agree with the ABI ROCm's programs are built for.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim"
[[ -e "$shim/libhsa-runtime64.so.1" ]] || { echo "SKIP: no HSA runtime in $shim"; exit 0; }
cc="${CC:-cc}"
command -v "$cc" >/dev/null || { echo "SKIP: no C compiler"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
# A sanitizer-instrumented runtime needs its sanitizer's runtime first, so
# the program is built with the sanitizer the library was built with.
sanitize=()
if command -v objdump >/dev/null; then
  case "$(objdump -p "$shim/libhsa-runtime64.so.1" 2>/dev/null | awk '/NEEDED/ {print $2}')" in
    *libtsan*) sanitize=(-fsanitize=thread -g) ;;
    *libasan*) sanitize=(-fsanitize=address,undefined -fno-omit-frame-pointer -g) ;;
  esac
fi

run() {   # run <label> <extra cflags...>
  local label=$1; shift
  if ! "$cc" -std=c11 -O1 -Wall -Werror "${sanitize[@]}" "$@" "$root/amd/tests/hsa/hsa_dispatch.c" -o "$tmp/hsa_dispatch" \
       -L"$shim" -l:libhsa-runtime64.so.1 -Wl,-rpath,"$(cd "$shim" && pwd)" 2> "$tmp/cc.log"; then
    echo "FAIL  hsa_dispatch.c builds against $label"; sed 's/^/      /' "$tmp/cc.log" | head -20; fail=1; return
  fi
  local out status
  out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 timeout 300 "$tmp/hsa_dispatch" \
        "$root/amd/tests/data/vector_add.gfx942.hsaco" 2>&1)
  status=$?
  echo "$out" | grep -E '^(ok|FAIL) ' | sed 's/^/      /'
  local passed; passed=$(grep -c '^ok ' <<< "$out")
  if [[ $status != 0 ]] || grep -q '^FAIL' <<< "$out" || [[ $passed != 15 ]]; then
    echo "FAIL  an HSA program built against $label runs: $passed of 15 (exit $status)"
    echo "$out" | grep -v '^ok ' | tail -5 | sed 's/^/      /'
    fail=1
  else
    echo "ok    an HSA program built against $label runs: 15 of 15"
  fi
}

run "VirtualGPU's HSA header" -I"$root/amd/include"
rocm_include="${VGPU_ROCM_INCLUDE:-}"
if [[ -z "$rocm_include" ]]; then
  for d in /opt/rocm/include $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-*/include 2>/dev/null | sort -V | tail -1); do
    [[ -e "$d/hsa/hsa.h" ]] && { rocm_include=$d; break; }
  done
fi
if [[ -n "$rocm_include" ]]; then
  run "ROCm's HSA headers" -DVGPU_REAL_HSA -D__HIP_PLATFORM_AMD__ -I"$rocm_include"
else
  echo "skip  no ROCm headers to build against as well"
fi
exit $fail
