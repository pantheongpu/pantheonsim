#!/usr/bin/env bash
# `vgpu run --preload` and two programs that use the shim as PyTorch does, with no PyTorch, so it runs where the PyTorch tests skip.
# With the driver loaded privately (#597) both broke, and every PyTorch test failed:
#   preload_torch_driver.cpp    reaches the driver through the runtime first, then the already loaded libcuda.so.1, with no
#                               cuInit of its own: the runtime neither initialised that driver nor handed it its primary
#                               context ("CUDA driver error: initialization error" on every PyTorch check);
#   preload_torch_profiler.cpp  the profiling engine switched on around a cuBLAS call: the shims no longer shared one engine,
#                               and the copies cuBLAS makes on the program's behalf were recorded as the program's cudaMemcpy;
#   preload_torch_peers.cpp     CUPTI and NVSHMEM, which call the driver by name, could not find the private one either: CUPTI's
#                               callbacks named a placeholder context, and nvshmemx_cumodule_init / culibrary_init failed.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
vgpu="$build/vgpu"
shim="$build/shim"
[[ -x $vgpu ]] || { echo "SKIP: vgpu not built"; exit 0; }
# The sonames, as PyTorch's libraries name them (libcudart.so.13 for CUDA 13): the shortest name of each.
soname() {
  local f
  for f in "$shim"/"$1".so.[0-9]*; do
    [[ -e $f ]] || continue
    basename "$f"
  done | awk '{ print length, $0 }' | sort -n | head -n1 | cut -d' ' -f2
}
cart=$(soname libcudart) blas=$(soname libcublas) cupti=$(soname libcupti) shmem=$(soname libnvshmem_host)
[[ -e $shim/libcuda.so.1 && -n $cart && -n $blas ]] || { echo "SKIP: the driver, runtime or cuBLAS shim is not built"; exit 0; }
. "$root/tests/shim_guard.sh"
if [[ -n "$(shim_sanitizer "$shim")" ]]; then
  # An instrumented shim preloaded ahead of the program's sanitizer runtime does not start (run_vgpu_run.sh says the same).
  echo "SKIP: sanitizer-instrumented shim (load order)"; exit 0
fi
cxx="${CXX:-c++}"
command -v "$cxx" >/dev/null 2>&1 || { echo "SKIP: no C++ compiler"; exit 0; }
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-preload-torch-like.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for p in driver profiler peers; do
  "$cxx" -std=c++17 -O1 -I"$root/include" "$root/nvidia/tests/e2e/preload_torch_$p.cpp" -o "$work/$p" -ldl -pthread ||
    { echo "FAIL: preload_torch_$p.cpp does not compile"; exit 1; }
done
fails=0
run() {   # <what> <gpu> <program> <args...>
  local what=$1 gpu=$2; shift 2
  local got rc
  got="$(cd "$work" && timeout 120 "$vgpu" run --quiet --gpu "$gpu" --preload "$@" 2>&1)"; rc=$?
  if (( rc != 0 )) || [[ "$(tail -n1 <<< "$got")" != ok ]]; then
    echo "FAIL: $what on $gpu: exit status $rc"; echo "$got" | sed 's/^/    /'; fails=1
  else
    echo "ok   $what on $gpu"
  fi
}
for gpu in nvidia/t4 nvidia/rtx5090; do
  run "the driver, initialised by the runtime" "$gpu" ./driver "$cart"
done
run "one profiling engine for every shim" nvidia/rtx5090 ./profiler "$cart" "$blas"
if [[ -n $cupti || -n $shmem ]]; then
  run "CUPTI and NVSHMEM reach the private driver" nvidia/rtx3060 ./peers "$cart" "${cupti:--}" "${shmem:--}"
else
  echo "skip CUPTI and NVSHMEM: neither shim is built"
fi
(( fails == 0 )) || exit 1
echo "PASS"
