#!/usr/bin/env bash
# Unpacks ROCm 7.1's LLVM from AMD's apt repository into a directory, with no
# install and no root: fetch-rocm-llvm.sh DIR
#
# That is the compiler (amdclang++), the OpenMP runtime and offload library
# (libomp, libomptarget) and the device bitcode a program built with
# -fopenmp --offload-arch=gfx... links. run_openmp.sh finds the result with
# VGPU_ROCM_PATH=DIR/opt/rocm-7.1.0. The HSA runtime it loads is the
# simulator's.
set -euo pipefail
dir=$1
repo=https://repo.radeon.com/rocm/apt/7.1/pool/main
v=70100-20~24.04_amd64.deb
pkgs=(r/rocm-llvm/rocm-llvm_20.0.0.25425.$v r/rocm-device-libs/rocm-device-libs_1.0.0.$v)
mkdir -p "$dir" "$dir.debs"
for p in "${pkgs[@]}"; do
  f="$dir.debs/$(basename "$p")"
  [[ -s $f ]] || curl -fsSL --retry 3 -o "$f" "$repo/$p"
  dpkg-deb -x "$f" "$dir"
done
rm -rf "$dir.debs"
ls -d "$dir"/opt/rocm-7.1.0
