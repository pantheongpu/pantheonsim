#!/usr/bin/env bash
# VirtualGPU's copy of the HIP structures against the HIP headers themselves,
# field by field, wherever ROCm is installed.
#
# A hipcc-built program reads each field at the offset its own header gives,
# so every field here has to sit where the header puts it. This builds a small
# program that includes both and checks each field's offset, then the sizes.
# Skips where no ROCm headers are found: the unit test pins the fields the
# workloads read everywhere else.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm "$HOME"/.local/share/rocm-*/opt/rocm-*; do
  [[ -n "$c" && -e "$c/include/hip/hip_runtime_api.h" ]] && { rocm=$c; break; }
done
[[ -n "$rocm" ]] || { echo "SKIP: no ROCm headers found (set VGPU_ROCM_PATH)"; exit 0; }
cxx=${CXX:-c++}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Every field of the structure, as VirtualGPU declares it.
fields=$(sed -n '/^struct DevicePropR0600 {/,/^};/p' "$root/amd/include/vgpu/hip_abi.hpp" |
  sed -n 's/.*[ *]\([A-Za-z_][A-Za-z0-9_]*\)\(\[[0-9]*\]\)*;.*/\1/p')
{
  echo '#include <cstddef>'
  echo '#include <hip/hip_runtime_api.h>'
  echo '#include "vgpu/hip_abi.hpp"'
  echo 'using Ours = vgpu::amd::abi::DevicePropR0600;'
  for f in $fields; do
    echo "static_assert(offsetof(Ours, $f) == offsetof(hipDeviceProp_tR0600, $f), \"$f is out of place\");"
  done
  echo 'static_assert(sizeof(Ours) == sizeof(hipDeviceProp_tR0600), "the sizes differ");'
  echo 'static_assert(sizeof(vgpu::amd::abi::DeviceArch) == sizeof(hipDeviceArch_t), "the arch flags differ");'
  # What hipPointerGetAttributes fills in.
  for f in $(sed -n '/^struct PointerAttribute {/,/^};/p' "$root/amd/include/vgpu/hip_abi.hpp" |
             sed -n 's/.*[ *]\([A-Za-z_][A-Za-z0-9_]*\);.*/\1/p'); do
    echo "static_assert(offsetof(vgpu::amd::abi::PointerAttribute, $f) == offsetof(hipPointerAttribute_t, $f), \"pointer attribute $f is out of place\");"
  done
  echo 'static_assert(sizeof(vgpu::amd::abi::PointerAttribute) == sizeof(hipPointerAttribute_t), "the pointer attributes differ");'
  echo 'static_assert(vgpu::amd::abi::kMemoryUnregistered == hipMemoryTypeUnregistered && vgpu::amd::abi::kMemoryHost == hipMemoryTypeHost && vgpu::amd::abi::kMemoryDevice == hipMemoryTypeDevice, "the memory types differ");'
  echo 'int main() { return 0; }'
} > "$tmp/abi.cpp"
count=$(wc -w <<< "$fields")
if "$cxx" -std=c++17 -D__HIP_PLATFORM_AMD__ -I"$rocm/include" -I"$root/amd/include" -c "$tmp/abi.cpp" \
     -o "$tmp/abi.o" 2>"$tmp/err"; then
  echo "ok    all $count fields sit where $rocm's hip_runtime_api.h puts them"
else
  echo "FAIL  the device properties differ from $rocm's hip_runtime_api.h:"
  grep -o 'static assertion failed.*' "$tmp/err" | sed 's/^/      /' | head -20
  exit 1
fi

# hipDeviceGetAttribute is asked by number, so each attribute VirtualGPU
# answers has to carry the number the header gives it. VirtualGPU names each
# after the header's own, kFoo for hipDeviceAttributeFoo.
attrs=$(sed -n '/^enum class DeviceAttribute/,/^};/p' "$root/amd/include/vgpu/hip_abi.hpp" |
  sed -n 's/^ *k\([A-Za-z0-9]*\) = .*/\1/p')
{
  echo '#include <hip/hip_runtime_api.h>'
  echo '#include "vgpu/hip_abi.hpp"'
  echo 'using A = vgpu::amd::abi::DeviceAttribute;'
  for a in $attrs; do
    echo "static_assert(static_cast<int>(A::k$a) == hipDeviceAttribute$a, \"hipDeviceAttribute$a has another number\");"
  done
  echo 'int main() { return 0; }'
} > "$tmp/attrs.cpp"
count=$(wc -w <<< "$attrs")
if "$cxx" -std=c++17 -D__HIP_PLATFORM_AMD__ -I"$rocm/include" -I"$root/amd/include" -c "$tmp/attrs.cpp" \
     -o "$tmp/attrs.o" 2>"$tmp/err"; then
  echo "ok    all $count device attributes carry the numbers $rocm's hip_runtime_api.h gives them"
else
  echo "FAIL  the device attributes differ from $rocm's hip_runtime_api.h:"
  grep -oE "static assertion failed.*|error: .*" "$tmp/err" | sed 's/^/      /' | head -20
  exit 1
fi
