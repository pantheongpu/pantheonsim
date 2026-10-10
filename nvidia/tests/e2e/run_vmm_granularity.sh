#!/usr/bin/env bash
# The virtual memory API's granularity and its size, offset and alignment checks (vmm_granularity.cu).
#
#   run_vmm_granularity.sh                   against the shims, on the rtx3060 profile
#   run_vmm_granularity.sh --card            against NVIDIA's libraries on a real GPU
#   run_vmm_granularity.sh --card --update   rewrite the expected file from the card
#
# nvidia/tests/data/vmm_granularity.rtx3060.expected is what an RTX 3060 printed; the shim's rtx3060 profile
# must print the same. The other profiles are checked for the one thing they share with it here: the granularity
# is 2 MiB, the minimum and the recommended one.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/vmm_granularity.cu"
expected="$root/nvidia/tests/data/vmm_granularity.rtx3060.expected"
out="${TMPDIR:-/tmp}/vgpu-vmm-granularity.$$"
card=0; update=0
for a in "$@"; do
  case "$a" in
    --card) card=1 ;;
    --update) update=1 ;;
    *) echo "usage: $0 [--card [--update]]" >&2; exit 2 ;;
  esac
done
command -v nvcc >/dev/null 2>&1 || { echo "SKIP: nvcc not found"; exit 0; }
trap 'rm -f "$out" "$out".*' EXIT

if (( card )); then
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  libs=""
  for d in "$cuda_root/targets/x86_64-linux/lib" "$cuda_root/lib64"; do
    [[ -e "$d/libcudart.so" || -e "$d"/libcudart.so.[0-9]* ]] && libs="$d" && break
  done
  [[ -n "$libs" && -e "$libs/stubs/libcuda.so" ]] || { echo "SKIP: no CUDA libraries beside nvcc"; exit 0; }
  nvcc -std=c++17 -arch=sm_86 -Wno-deprecated-gpu-targets "$src" -o "$out" -lcuda -L"$libs/stubs" || { echo "FAIL: does not compile"; exit 1; }
  LD_LIBRARY_PATH="$libs" "$out" > "$out.txt" || { echo "FAIL: the program failed on the card"; exit 1; }
  if (( update )); then cp "$out.txt" "$expected"; echo "wrote $expected"; exit 0; fi
  diff -u "$expected" "$out.txt" && echo "the VMM answers match the card's" || { echo "FAIL: differs from what the card printed"; exit 1; }
  exit 0
fi

shopt -s nullglob
cuda_libs=("$shim"/libcuda.so.[0-9]*)
shopt -u nullglob
(( ${#cuda_libs[@]} )) || { echo "SKIP: libvgpucuda not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -arch=sm_86 -Wno-deprecated-gpu-targets "${san_flags[@]}" "$src" -o "$out" -lcuda -L"$shim" \
  || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

run() {   # profile -> output file
  env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU="$1" LD_LIBRARY_PATH="$shim" "$out" > "$2" 2>"$2.err" || { echo "FAIL: $1: the program failed"; cat "$2.err"; exit 1; }
}

run nvidia/rtx3060 "$out.3060"
diff -u "$expected" "$out.3060" || { echo "FAIL: the rtx3060 profile does not print what the card printed"; exit 1; }
echo "rtx3060: the card's answers"
for prof in nvidia/t4 nvidia/h100 nvidia/b200 nvidia/rtx5090; do
  run "$prof" "$out.p"
  grep -q "^granularity minimum  *CUDA_SUCCESS 2097152$" "$out.p" && grep -q "^granularity recommended  *CUDA_SUCCESS 2097152$" "$out.p" \
    || { echo "FAIL: $prof: the granularity is not 2 MiB"; exit 1; }
  echo "$prof: 2 MiB"
done
echo "PASS"
