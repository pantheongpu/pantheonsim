#!/usr/bin/env bash
# Rebuilds the HIP programs in ../ (chevron, printf, runtime, cooperative,
# pointers)
# with every ROCm release installed, one directory each, for
# amd/tests/e2e/run_rocm_versions.sh to run on the shim. A release is found
# at ~/.local/share/rocm-<version>/opt/rocm-* or /opt/rocm-<version>*.
#
# The releases are the current ones: the last of ROCm 6 and every ROCm 7.
set -uo pipefail
cd "$(dirname "$0")"
for v in 6.4 7.0 7.1 7.2; do
  rocm=$(ls -d "$HOME"/.local/share/rocm-$v/opt/rocm-* /opt/rocm-$v* 2>/dev/null | head -1)
  [[ -n "$rocm" && -x "$rocm/bin/hipcc" ]] || { echo "ROCm $v: not installed, kept as it is"; continue; }
  export ROCM_PATH=$rocm HIP_PATH=$rocm
  export HIP_CLANG_PATH=$([[ -d $rocm/lib/llvm/bin ]] && echo "$rocm/lib/llvm/bin" || echo "$rocm/llvm/bin")
  export HIP_DEVICE_LIB_PATH=$([[ -d $rocm/amdgcn/bitcode ]] && echo "$rocm/amdgcn/bitcode" ||
                               ls -d "$rocm"/llvm/lib/clang/*/lib/amdgcn/bitcode 2>/dev/null | head -1)
  mkdir -p "$v"
  for p in chevron printf runtime cooperative pointers; do
    if "$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 "../$p.cpp" -o "$v/$p.gfx942" 2>/dev/null; then
      echo "ROCm $v: built $p"
    else
      rm -f "$v/$p.gfx942"
      echo "ROCm $v: its compiler does not build $p"
    fi
  done
done
