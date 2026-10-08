#!/usr/bin/env bash
# Unpacks ROCm 7.1's OpenCL runtime from AMD's apt repository into a directory,
# with no install and no root: fetch-rocm-opencl.sh DIR
#
# That is libamdocl64 (the OpenCL runtime on the same CLR as HIP), the compiler
# library it builds kernels with (comgr) and the device bitcode comgr links.
# run_opencl.sh finds the result with VGPU_ROCM_PATH=DIR/opt/rocm-7.1.0. The
# HSA runtime it loads is the simulator's, and the ICD loader is the system's.
set -euo pipefail
dir=$1
repo=https://repo.radeon.com/rocm/apt/7.1/pool/main
v=70100-20~24.04_amd64.deb
pkgs=(r/rocm-opencl/rocm-opencl_2.0.0.$v c/comgr/comgr_3.0.0.$v r/rocm-device-libs/rocm-device-libs_1.0.0.$v
      r/rocm-core/rocm-core_7.1.0.$v)
mkdir -p "$dir" "$dir.debs"
for p in "${pkgs[@]}"; do
  f="$dir.debs/$(basename "$p")"
  [[ -s $f ]] || curl -fsSL --retry 3 -o "$f" "$repo/$p"
  dpkg-deb -x "$f" "$dir"
done
rm -rf "$dir.debs"
ls -d "$dir"/opt/rocm-7.1.0
