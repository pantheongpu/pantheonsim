#!/usr/bin/env bash
# The driver API's graphs (driver_graphs.cpp), and the two CUDA APIs sharing
# graphs, streams and events (driver_runtime_graphs.cu), built with nvcc and run
# on a simulated GPU. The driver's graph entry points are thin calls into
# libcudart, which libcuda finds beside itself, so both libraries must be built.
# The second program runs as SASS, as SASS only, and as PTX.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-driver-graphs.$$"
command -v nvcc >/dev/null || { echo "SKIP: nvcc not found"; exit 0; }
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#cudart_libs[@]} )) || { echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0; }
[[ -e "$shim/libcuda.so.1" ]] || { echo "SKIP: libvgpucuda not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -f "$out".*' EXIT
# Both libraries carry the simulator's core, and a thread sanitizer cannot map
# two copies; an address sanitizer reports the pair as an ODR violation, which
# they are by design.
case "$(shim_sanitizer "$shim")" in
  tsan) echo "SKIP: a thread-sanitizer build loads two copies of the core"; exit 0 ;;
  asan) export ASAN_OPTIONS="${ASAN_OPTIONS:+$ASAN_OPTIONS:}detect_odr_violation=0" ;;
esac
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"

drv="$root/nvidia/tests/e2e/driver_graphs.cpp"
mix="$root/nvidia/tests/e2e/driver_runtime_graphs.cu"
"$nvcc_bin" -std=c++17 -cudart none -Wno-deprecated-gpu-targets "${san_flags[@]}" "$drv" -o "$out.drv" -L"$shim" -lcuda || { echo "FAIL: $drv does not compile"; exit 1; }
"$nvcc_bin" -std=c++17 -arch=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" "$mix" -o "$out.both" -L"$shim" -lcuda || { echo "FAIL: $mix does not compile"; exit 1; }
"$nvcc_bin" -std=c++17 -gencode arch=compute_86,code=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" "$mix" -o "$out.sass" -L"$shim" -lcuda || { echo "FAIL: $mix (SASS) does not compile"; exit 1; }
"$nvcc_bin" -std=c++17 -gencode arch=compute_86,code=compute_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" "$mix" -o "$out.ptx" -L"$shim" -lcuda || { echo "FAIL: $mix (PTX) does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out.both"; then exit 0; fi

status=0
run() {
  local name=$1 bin=$2 gpu=$3
  shift 3
  local log
  log="$(env VGPU_QUIET=1 VGPU_GPU="$gpu" LD_LIBRARY_PATH="$shim" "$@" "$bin" 2>&1)" || { echo "$log" | sed "s/^/    $name: /"; echo "FAIL: $name"; status=1; return; }
  echo "$log" | tail -1 | sed "s/^/    $name: /"
  [[ "$(tail -1 <<< "$log")" == PASS* ]] || { echo "FAIL: $name"; status=1; }
}
run "driver graphs" "$out.drv" nvidia/rtx3060
run "shared graphs (SASS and PTX)" "$out.both" nvidia/rtx3060
run "shared graphs (SASS only)" "$out.sass" nvidia/rtx3060
run "shared graphs (PTX only)" "$out.ptx" nvidia/rtx3060
run "shared graphs (PTX forced)" "$out.both" nvidia/rtx3060 VGPU_SASS=0
exit $status
