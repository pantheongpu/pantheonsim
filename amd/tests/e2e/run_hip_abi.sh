#!/usr/bin/env bash
# VirtualGPU's copy of the HIP structures and numbers against the HIP headers
# themselves, field by field, for every current ROCm release installed (6.4
# and later; older ones are not targeted, and are passed over).
#
# A hipcc-built program reads each field at the offset its own header gives,
# so every field here has to sit where each release's header puts it: the
# device properties in both layouts (R0600, and the older hipDeviceProp_tR0000
# the unsuffixed call keeps), the pointer attributes, and the numbers
# hipDeviceGetAttribute is asked by. Skips where no ROCm headers are found:
# the unit test pins the fields the workloads read everywhere else.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
releases=()
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm "$HOME"/.local/share/rocm-*/opt/rocm-*; do
  [[ -n "$c" && -e "$c/include/hip/hip_runtime_api.h" ]] || continue
  real=$(readlink -f "$c")
  version=$(basename "$real" | sed -n 's/^rocm-\([0-9.]*\).*/\1/p')
  [[ -n "$version" && $(printf '%s\n' "$version" 6.4 | sort -V | head -1) != 6.4 ]] && continue   # older than 6.4
  [[ " ${releases[*]} " == *" $real "* ]] || releases+=("$real")
done
[[ ${#releases[@]} -gt 0 ]] || { echo "SKIP: no ROCm 6.4 or later headers found (set VGPU_ROCM_PATH)"; exit 0; }
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
  {
    echo '#include <cstddef>'
    echo '#include <hip/hip_runtime_api.h>'
    # The older layout, hipDeviceProp_tR0000, is in a header of its own.
    echo '#include <hip/hip_deprecated.h>'
    echo '#include "vgpu/hip_abi.hpp"'
    for f in $(fields DevicePropR0600); do
      echo "static_assert(offsetof(vgpu::amd::abi::DevicePropR0600, $f) == offsetof(hipDeviceProp_tR0600, $f), \"R0600 $f is out of place\");"
    done
    echo 'static_assert(sizeof(vgpu::amd::abi::DevicePropR0600) == sizeof(hipDeviceProp_tR0600), "the R0600 sizes differ");'
    for f in $(fields DevicePropR0000); do
      echo "static_assert(offsetof(vgpu::amd::abi::DevicePropR0000, $f) == offsetof(hipDeviceProp_tR0000, $f), \"R0000 $f is out of place\");"
    done
    echo "static_assert(sizeof(vgpu::amd::abi::DevicePropR0000) == sizeof(hipDeviceProp_tR0000), \"the R0000 sizes differ\");"
    echo 'static_assert(sizeof(vgpu::amd::abi::DeviceArch) == sizeof(hipDeviceArch_t), "the arch flags differ");'
    for f in $(fields PointerAttribute); do
      echo "static_assert(offsetof(vgpu::amd::abi::PointerAttribute, $f) == offsetof(hipPointerAttribute_t, $f), \"pointer attribute $f is out of place\");"
    done
    echo 'static_assert(sizeof(vgpu::amd::abi::PointerAttribute) == sizeof(hipPointerAttribute_t), "the pointer attributes differ");'
    echo 'static_assert(vgpu::amd::abi::kMemoryUnregistered == hipMemoryTypeUnregistered && vgpu::amd::abi::kMemoryHost == hipMemoryTypeHost && vgpu::amd::abi::kMemoryDevice == hipMemoryTypeDevice, "the memory types differ");'
  } > "$tmp/abi.cpp"
  check "$release: the device properties (R0600 and R0000) and pointer attributes sit where its headers put them" "$tmp/abi.cpp"
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
