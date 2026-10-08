#!/usr/bin/env bash
# An HSA program on VirtualGPU's libhsa-runtime64 (amd/tests/hsa/hsa_dispatch.c):
# agents, memory pools, an executable loaded from a code object, and kernels
# dispatched through AQL queues, on two simulated MI300Xs; and the images
# extension (hsa_images.c) on each Radeon profile and an MI300X.
#
# It is built against VirtualGPU's own HSA header, and again against ROCm's
# where ROCm's headers are installed (VGPU_ROCM_INCLUDE, /opt/rocm/include, or
# ~/.local/share/rocm-*): both builds must pass, which is what says the header
# and the runtime agree with the ABI ROCm's programs are built for. And where
# ROCm's rocminfo is at hand (VGPU_ROCMINFO, /opt/rocm/bin, or unpacked from
# its package under ~/.local/share/rocm-*-debs), it runs unmodified and must
# describe both devices.
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
  if [[ $status != 0 ]] || grep -q '^FAIL' <<< "$out" || [[ $passed != 23 ]]; then
    echo "FAIL  an HSA program built against $label runs: $passed of 23 (exit $status)"
    echo "$out" | grep -v '^ok ' | tail -5 | sed 's/^/      /'
    fail=1
  else
    echo "ok    an HSA program built against $label runs: 23 of 23"
  fi
}

# The images extension (amd/tests/hsa/hsa_images.c): every check on a
# Radeon profile, which has texture units, and the refusals on an MI300X,
# which has none.
run_images() {   # run_images <label> <extra cflags...>
  local label=$1; shift
  if ! "$cc" -std=c11 -O1 -Wall -Werror "${sanitize[@]}" "$@" "$root/amd/tests/hsa/hsa_images.c" -o "$tmp/hsa_images" \
       -L"$shim" -l:libhsa-runtime64.so.1 -Wl,-rpath,"$(cd "$shim" && pwd)" 2> "$tmp/cc.log"; then
    echo "FAIL  hsa_images.c builds against $label"; sed 's/^/      /' "$tmp/cc.log" | head -20; fail=1; return
  fi
  local gpu want out status passed
  for run in "rx7900xtx 10" "rx9070xt 10" "rx6900xt 10" "mi300x 3"; do
    set -- $run
    gpu=$1 want=$2
    out=$(VGPU_QUIET=1 VGPU_GPU=amd/$gpu timeout 120 "$tmp/hsa_images" 2>&1)
    status=$?
    passed=$(grep -c '^ok ' <<< "$out")
    if [[ $status != 0 ]] || grep -q '^FAIL' <<< "$out" || [[ $passed != "$want" ]]; then
      echo "FAIL  the images extension on $gpu, built against $label: $passed of $want (exit $status)"
      echo "$out" | grep -v '^ok ' | tail -5 | sed 's/^/      /'
      fail=1
    else
      echo "ok    the images extension on $gpu, built against $label: $want of $want"
    fi
  done
}

# Virtual memory, IPC and SVM (amd/tests/hsa/hsa_vmem.c), on an MI300X.
run_vmem() {   # run_vmem <label> <extra cflags...>
  local label=$1; shift
  if ! "$cc" -std=c11 -O1 -Wall -Werror "${sanitize[@]}" "$@" "$root/amd/tests/hsa/hsa_vmem.c" -o "$tmp/hsa_vmem" \
       -L"$shim" -l:libhsa-runtime64.so.1 -Wl,-rpath,"$(cd "$shim" && pwd)" 2> "$tmp/cc.log"; then
    echo "FAIL  hsa_vmem.c builds against $label"; sed 's/^/      /' "$tmp/cc.log" | head -20; fail=1; return
  fi
  local out status passed
  out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x timeout 120 "$tmp/hsa_vmem" 2>&1)
  status=$?
  passed=$(grep -c '^ok ' <<< "$out")
  if [[ $status != 0 ]] || grep -q '^FAIL' <<< "$out" || [[ $passed != 10 ]]; then
    echo "FAIL  virtual memory and IPC, built against $label: $passed of 10 (exit $status)"
    echo "$out" | grep -v '^ok ' | tail -5 | sed 's/^/      /'
    fail=1
  else
    echo "ok    virtual memory and IPC, built against $label: 10 of 10"
  fi
}

# Caches, wavefronts, ISA compatibility and signal groups (amd/tests/hsa/hsa_caches.c).
run_caches() {   # run_caches <label> <extra cflags...>
  local label=$1; shift
  if ! "$cc" -std=c11 -O1 -Wall -Werror "${sanitize[@]}" "$@" -pthread "$root/amd/tests/hsa/hsa_caches.c" -o "$tmp/hsa_caches" \
       -L"$shim" -l:libhsa-runtime64.so.1 -Wl,-rpath,"$(cd "$shim" && pwd)" 2> "$tmp/cc.log"; then
    echo "FAIL  hsa_caches.c builds against $label"; sed 's/^/      /' "$tmp/cc.log" | head -20; fail=1; return
  fi
  local out status passed
  out=$(VGPU_QUIET=1 VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 timeout 120 "$tmp/hsa_caches" 2>&1)
  status=$?
  passed=$(grep -c '^ok ' <<< "$out")
  if [[ $status != 0 ]] || grep -q '^FAIL' <<< "$out" || [[ $passed != 11 ]]; then
    echo "FAIL  caches, wavefronts and signal groups, built against $label: $passed of 11 (exit $status)"
    echo "$out" | grep -v '^ok ' | tail -5 | sed 's/^/      /'
    fail=1
  else
    echo "ok    caches, wavefronts and signal groups, built against $label: 11 of 11"
  fi
}

run "VirtualGPU's HSA header" -I"$root/amd/include"
run_caches "VirtualGPU's HSA header" -I"$root/amd/include"
run_images "VirtualGPU's HSA header" -I"$root/amd/include"
run_vmem "VirtualGPU's HSA header" -I"$root/amd/include"
rocm_include="${VGPU_ROCM_INCLUDE:-}"
if [[ -z "$rocm_include" ]]; then
  for d in /opt/rocm/include $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-*/include 2>/dev/null | sort -V | tail -1); do
    [[ -e "$d/hsa/hsa.h" ]] && { rocm_include=$d; break; }
  done
fi
if [[ -n "$rocm_include" ]]; then
  run "ROCm's HSA headers" -DVGPU_REAL_HSA -D__HIP_PLATFORM_AMD__ -I"$rocm_include"
  run_caches "ROCm's HSA headers" -DVGPU_REAL_HSA -D__HIP_PLATFORM_AMD__ -I"$rocm_include"
  run_images "ROCm's HSA headers" -DVGPU_REAL_HSA -D__HIP_PLATFORM_AMD__ -I"$rocm_include"
  run_vmem "ROCm's HSA headers" -DVGPU_REAL_HSA -D__HIP_PLATFORM_AMD__ -I"$rocm_include"
else
  echo "skip  no ROCm headers to build against as well"
fi
rocminfo="${VGPU_ROCMINFO:-}"
if [[ -z "$rocminfo" ]]; then
  for f in /opt/rocm/bin/rocminfo $(ls "$HOME"/.local/share/rocm-*-debs/rocminfo/opt/rocm-*/bin/rocminfo 2>/dev/null | sort -V | tail -1); do
    [[ -x "$f" ]] && { rocminfo=$f; break; }
  done
fi
# ROCm's rocminfo asks the kernel first: without the amdgpu module loaded
# (/sys/module/amdgpu/initstate) it says "ROCk module is NOT loaded" and stops
# before it calls HSA at all -- except under WSL (/dev/dxg), where there is no
# module to ask. So on a Linux machine with no AMD GPU it tests the host, not
# the simulator, and is not run (vgpu-rocminfo prints the same in its place;
# amd_tools compares the two where both run).
if [[ -n "$rocminfo" && ! -e /sys/module/amdgpu/initstate && ! -e /dev/dxg ]]; then
  echo "skip  ROCm's rocminfo: no amdgpu kernel module here, which it asks for before HSA"
elif [[ -n "$rocminfo" && -z "${sanitize[*]}" ]]; then
  info=$(VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$(cd "$shim" && pwd)" timeout 120 "$rocminfo" 2>&1)
  status=$?
  gpus=$(grep -c '^  Name: *gfx942 *$' <<< "$info")
  if [[ $status == 0 ]] && ! grep -q 'failure\|VirtualGPU HSA' <<< "$info" && [[ $gpus == 2 ]] &&
     grep -q 'Marketing Name: *AMD Instinct MI300X' <<< "$info" && grep -q 'Compute Unit: *304' <<< "$info" &&
     grep -q 'amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-' <<< "$info"; then
    echo "ok    ROCm's rocminfo describes both devices"
  else
    echo "FAIL  ROCm's rocminfo describes both devices (exit $status, $gpus GPU agents)"
    grep -m5 'failure\|Call returned\|VirtualGPU HSA' <<< "$info" | sed 's/^/      /'
    fail=1
  fi
else
  echo "skip  no rocminfo to run (or a sanitizer build, which ROCm's binary is not)"
fi
exit $fail
