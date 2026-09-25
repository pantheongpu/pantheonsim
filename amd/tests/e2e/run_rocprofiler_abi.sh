#!/usr/bin/env bash
# VirtualGPU's copy of the rocprofiler-sdk structures against rocprofiler-sdk's
# own headers, field by field, wherever they are installed.
#
# AMD's profiler reads every agent, counter and trace record at the offsets its
# headers give, so each field in amd/include/vgpu/rocprofiler_abi.hpp has to
# sit where they put it. This builds a program that includes both and checks
# every field's offset and every structure's size. Skips where the headers are
# not found (they come with ROCm's rocprofiler-sdk and rocprofiler-sdk-roctx
# packages).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
sdk=""; rocm=""
# Newest release first: the SDK and the HSA headers it is checked with belong
# together, and ROCm 5's HSA headers predate the SDK.
for c in "${VGPU_ROCPROFILER_SDK_PATH:-}" "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm \
         $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV) "$HOME"/.local/share/rocprofiler-sdk*/opt/rocm-*; do
  [[ -n "$c" && -e "$c/include/rocprofiler-sdk/rocprofiler.h" && -z "$sdk" ]] && sdk=$c
  [[ -n "$c" && -e "$c/include/hsa/hsa.h" && -z "$rocm" ]] && rocm=$c
done
[[ -n "$sdk" && -n "$rocm" ]] || { echo "SKIP: no rocprofiler-sdk and HSA headers found (set VGPU_ROCPROFILER_SDK_PATH)"; exit 0; }
[[ -e "$sdk/include/rocprofiler-sdk-roctx/api_trace.h" ]] || { echo "SKIP: $sdk has no rocprofiler-sdk-roctx headers"; exit 0; }
cxx=${CXX:-c++}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

header="$root/amd/include/vgpu/rocprofiler_abi.hpp"
structs=$(sed -n 's/^typedef struct \(rocprofiler_[a-z0-9_]*\) {$/\1/p' "$header")
{
  echo '#include <cstddef>'
  echo '#include <rocprofiler-sdk/rocprofiler.h>'
  echo '#include <rocprofiler-sdk/registration.h>'
  echo '#include "vgpu/rocprofiler_abi.hpp"'
  echo 'namespace ours = vgpu::amd::rocprof;'
  for s in $structs; do
    # Every named field of the structure, bit-fields aside (offsetof cannot
    # take one; the size check below covers them).
    fields=$(sed -n "/^typedef struct $s {\$/,/^} $s;\$/p" "$header" | grep -v ':' |
      sed -n 's/^ \{2,\}[A-Za-z_].*[ *]\([A-Za-z_][A-Za-z0-9_]*\)\(\[[0-9]*\]\)*;$/\1/p')
    for f in $fields; do
      echo "static_assert(offsetof(ours::$s, $f) == offsetof(::$s, $f), \"$s.$f is out of place\");"
    done
    echo "static_assert(sizeof(ours::$s) == sizeof(::$s), \"$s differs in size\");"
    echo "static_assert(alignof(ours::$s) == alignof(::$s), \"$s differs in alignment\");"
  done
  echo 'static_assert(ours::kVersion == ROCPROFILER_VERSION, "a different rocprofiler-sdk version");'
  echo 'int main() { return 0; }'
} > "$tmp/abi.cpp"
count=$(grep -c 'offsetof' "$tmp/abi.cpp")
nstructs=$(wc -w <<< "$structs")
if "$cxx" -std=c++17 -D__HIP_PLATFORM_AMD__ -I"$sdk/include" -I"$rocm/include" -I"$root/amd/include" \
     -c "$tmp/abi.cpp" -o "$tmp/abi.o" 2>"$tmp/err"; then
  echo "ok    $count fields of $nstructs structures sit where $sdk's rocprofiler-sdk headers put them"
else
  echo "FAIL  the structures differ from $sdk's rocprofiler-sdk headers:"
  grep -E 'static assertion failed|error' "$tmp/err" | sed 's/^/      /' | head -20
  exit 1
fi
