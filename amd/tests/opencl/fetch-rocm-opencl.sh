#!/usr/bin/env bash
# Unpacks ROCm's OpenCL runtime from AMD's apt repository into a directory,
# with no install and no root: fetch-rocm-opencl.sh DIR [VERSION]
#
# That is libamdocl64 (the OpenCL runtime on the same CLR as HIP), the compiler
# library it builds kernels with (comgr) and the device bitcode comgr links.
# VERSION is a ROCm release in the repository (default 7.1.0; 7.2.4 works too).
# run_opencl.sh finds the result with VGPU_ROCM_PATH=DIR/opt/rocm-VERSION. The
# HSA runtime it loads is the simulator's, and the ICD loader is the system's.
set -euo pipefail
dir=$1
ver=${2:-7.1.0}
repo=https://repo.radeon.com/rocm/apt/${ver%.0}/pool/main
mkdir -p "$dir" "$dir.debs"
for p in r/rocm-opencl c/comgr r/rocm-device-libs r/rocm-core; do
  # The build number differs per release (70100-20, 70204-93), so read the pool's index.
  name=$(curl -fsSL --http1.1 --retry 5 --retry-all-errors --retry-delay 3 "$repo/$p/" |
         grep -o "href=\"$(basename "$p")_[^\"]*~24.04_amd64.deb\"" | sort -V | tail -1 | cut -d'"' -f2)
  [[ -n $name ]] || { echo "no 24.04 package for $p in $repo" >&2; exit 1; }
  f="$dir.debs/$name"
  [[ -s $f ]] || curl -fsSL --http1.1 --retry 5 --retry-all-errors --retry-delay 3 -o "$f" "$repo/$p/$name"
  dpkg-deb -x "$f" "$dir"
done
rm -rf "$dir.debs"
ls -d "$dir"/opt/rocm-"$ver"
