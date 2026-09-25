#!/usr/bin/env bash
# VirtualGPU's copy of the HIP structures and numbers against the HIP headers
# themselves, field by field, for every ROCm release installed.
#
# A hipcc-built program reads each field at the offset its own header gives,
# so every field here has to sit where each release's header puts it: the
# device properties in the R0600 layout of ROCm 6 and 7, and in ROCm 5's
# layout (hipDeviceProp_t there, hipDeviceProp_tR0000 after), the pointer
# attributes, and the numbers hipDeviceGetAttribute is asked by. Skips where
# no ROCm headers are found: the unit test pins the fields the workloads read
# everywhere else.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
releases=()
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm "$HOME"/.local/share/rocm-*/opt/rocm-*; do
  [[ -n "$c" && -e "$c/include/hip/hip_runtime_api.h" ]] || continue
  real=$(readlink -f "$c")
  [[ " ${releases[*]} " == *" $real "* ]] || releases+=("$real")
done
[[ ${#releases[@]} -gt 0 ]] || { echo "SKIP: no ROCm headers found (set VGPU_ROCM_PATH)"; exit 0; }
cxx=${CXX:-c++}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
fields() {  # the fields of one of VirtualGPU's structures
  sed -n "/^struct $1 {/,/^};/p" "$root/amd/include/vgpu/hip_abi.hpp" |
    sed -n 's/.*[ *]\([A-Za-z_][A-Za-z0-9_]*\)\(\[[0-9]*\]\)*;.*/\1/p'
}
check() {  # check <name> <source>: compiles the static_asserts in <source>
  if "$cxx" -std=c++17 -D__HIP_PLATFORM_AMD__ -I"$rocm/include" -I"$root/amd/include" -c "$2" -o "$tmp/x.o" \
       2>"$tmp/err"; then
    echo "ok    $1"
  else
    echo "FAIL  $1:"
    grep -oE "static assertion failed.*|error: .*" "$tmp/err" | sed 's/^/      /' | head -20
    fail=1
  fi
}
for rocm in "${releases[@]}"; do
  release=$(basename "$rocm")
  header="$rocm/include/hip/hip_runtime_api.h"
  modern=$(grep -q "hipDeviceProp_tR0600" "$header" && echo 1 || echo 0)
  {
    echo '#include <cstddef>'
    echo '#include <hip/hip_runtime_api.h>'
    # ROCm 6 and 7 keep the old layout, hipDeviceProp_tR0000, in a header of
    # its own.
    [[ -e "$rocm/include/hip/hip_deprecated.h" ]] && echo '#include <hip/hip_deprecated.h>'
    echo '#include "vgpu/hip_abi.hpp"'
    if [[ $modern == 1 ]]; then
      for f in $(fields DevicePropR0600); do
        echo "static_assert(offsetof(vgpu::amd::abi::DevicePropR0600, $f) == offsetof(hipDeviceProp_tR0600, $f), \"R0600 $f is out of place\");"
      done
      echo 'static_assert(sizeof(vgpu::amd::abi::DevicePropR0600) == sizeof(hipDeviceProp_tR0600), "the R0600 sizes differ");'
      old=hipDeviceProp_tR0000 type=type
    else
      old=hipDeviceProp_t type=memoryType   # ROCm 5: one layout, and the field before its renaming
    fi
    for f in $(fields DevicePropR0000); do
      echo "static_assert(offsetof(vgpu::amd::abi::DevicePropR0000, $f) == offsetof($old, $f), \"R0000 $f is out of place\");"
    done
    echo "static_assert(sizeof(vgpu::amd::abi::DevicePropR0000) == sizeof($old), \"the R0000 sizes differ\");"
    echo 'static_assert(sizeof(vgpu::amd::abi::DeviceArch) == sizeof(hipDeviceArch_t), "the arch flags differ");'
    for f in $(fields PointerAttribute); do
      theirs=$f; [[ $f == type ]] && theirs=$type
      echo "static_assert(offsetof(vgpu::amd::abi::PointerAttribute, $f) == offsetof(hipPointerAttribute_t, $theirs), \"pointer attribute $f is out of place\");"
    done
    echo 'static_assert(sizeof(vgpu::amd::abi::PointerAttribute) == sizeof(hipPointerAttribute_t), "the pointer attributes differ");'
    # ROCm 6 renumbered the memory types, and gave memory the runtime knows
    # nothing of a type of its own.
    if [[ $modern == 1 ]]; then
      echo 'static_assert(vgpu::amd::abi::kMemoryUnregistered == hipMemoryTypeUnregistered && vgpu::amd::abi::kMemoryHost == hipMemoryTypeHost && vgpu::amd::abi::kMemoryDevice == hipMemoryTypeDevice, "the memory types differ");'
    else
      echo 'static_assert(vgpu::amd::abi::kMemoryHostRocm5 == hipMemoryTypeHost && vgpu::amd::abi::kMemoryDeviceRocm5 == hipMemoryTypeDevice, "the ROCm 5 memory types differ");'
    fi
  } > "$tmp/abi.cpp"
  check "$release: the device properties$([[ $modern == 1 ]] && echo " (R0600 and R0000)" || echo " (ROCm 5's layout)") and pointer attributes sit where its headers put them" "$tmp/abi.cpp"
  # hipDeviceGetAttribute is asked by number, so each attribute VirtualGPU
  # answers has to carry the number the header gives it. VirtualGPU names each
  # after the header's own, kFoo for hipDeviceAttributeFoo.
  {
    echo '#include <hip/hip_runtime_api.h>'
    echo '#include "vgpu/hip_abi.hpp"'
    echo 'using A = vgpu::amd::abi::DeviceAttribute;'
    sed -n '/^enum class DeviceAttribute/,/^};/p' "$root/amd/include/vgpu/hip_abi.hpp" |
      sed -n 's/^ *k\([A-Za-z0-9]*\) = .*/\1/p' | while read -r a; do
        grep -q "hipDeviceAttribute$a\b" "$header" "$rocm"/include/hip/*.h 2>/dev/null || continue   # not in this release
        echo "static_assert(static_cast<int>(A::k$a) == hipDeviceAttribute$a, \"hipDeviceAttribute$a has another number\");"
      done
  } > "$tmp/attrs.cpp"
  check "$release: every device attribute it names carries the number VirtualGPU answers" "$tmp/attrs.cpp"
done
exit $fail
