#!/usr/bin/env bash
# Library conformance tests against output captured on a physical GPU.
#
# run_conformance.sh compares the simulator with NVIDIA's libraries, but only
# where a GPU is attached, so it cannot guard anything in CI. This runs the
# same test programs on the simulator alone and compares them with what an
# RTX 3060 printed for them (nvidia/tests/conformance/golden/), to the same
# tolerance: a change that makes a routine disagree with the hardware fails
# here, on a runner with no GPU.
#
# A golden file is refreshed by running run_conformance.sh on a machine with a
# GPU and copying its <name>.real.txt over, minus the library's version line.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
conf="$root/nvidia/tests/conformance"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (the tests are CUDA programs)"; exit 0
fi
command -v python3 >/dev/null 2>&1 || { echo "SKIP: python3 not found (the comparison is a Python script)"; exit 0; }

fail=0
# name:libraries[:relative tolerance], the libraries comma-separated. The
# cuDNN programs build against the vendored headers and link the shim's
# library, since cuDNN is not part of the toolkit; cudnn_types prints
# half-precision results, so it is compared to half precision. npp_imgproc's
# integer results are compared exactly whatever the tolerance; its floats
# (single-precision image arithmetic, summed) to 1e-4. npp_segment is all
# integers (the watershed and marker-label compression).
npp_libs=-lnppc,-lnppial,-lnppicc,-lnppidei,-lnppif,-lnppig,-lnppim,-lnppist,-lnppisu,-lnppitc,-lnpps
for spec in cublas_level1:-lcublas cusparse_ops:-lcusparse cudnn_backward:-lcudnn cudnn_types:-lcudnn:1e-3 \
            npp_imgproc:$npp_libs:1e-4 npp_segment:$npp_libs:1e-4; do
  name="${spec%%:*}" rest="${spec#*:}"
  lib="${rest%%:*}" tol=1e-5 inc=""
  [[ "$rest" == *:* ]] && tol="${rest#*:}"
  lib="${lib//,/ }"
  if [[ "$lib" == -lcudnn ]]; then
    [[ -e "$shim/libcudnn.so" ]] || { echo "skip  $name: libvgpucudnn not built"; continue; }
    cudnn_inc="$(cudnn_include_dir)" || { echo "skip  $name: no cuDNN headers (see scripts/fetch-cudnn-headers.py)"; continue; }
    inc="-I$cudnn_inc -Xcompiler -Wno-deprecated-declarations"
  fi
  if ! nvcc -std=c++17 -cudart shared -arch=compute_75 -code=compute_75 -Wno-deprecated-gpu-targets \
        $(shim_sanitizer_nvcc_flags "$shim") $inc "$conf/$name.cu" -o "$out/$name" -L"$shim" $lib \
        2> "$out/$name.build.log"; then
    echo "FAIL  $name: did not compile"; head -5 "$out/$name.build.log"; fail=1; continue
  fi
  require_shim_libs "$shim" "$out/$name" || exit 0
  VGPU_QUIET=1 VGPU_GPU=nvidia/a10 VGPU_VRAM_MB=256 LD_LIBRARY_PATH="$shim" timeout 300 "$out/$name" \
    2>&1 | grep -v '^cusparse major' > "$out/$name.txt"
  if python3 "$conf/compare_numeric.py" "$conf/golden/$name.rtx3060.txt" "$out/$name.txt" "$tol"; then
    echo "ok    $name: matches the RTX 3060 ($(wc -l < "$out/$name.txt") lines)"
  else
    echo "FAIL  $name: differs from the RTX 3060"; fail=1
  fi
done
exit $fail
