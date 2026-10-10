#!/usr/bin/env bash
# Peer access and managed memory on a pair of devices (p2p_managed.cu).
#
#   run_p2p_managed.sh                   against the shims: two simulated RTX 3060s
#   run_p2p_managed.sh --card            against NVIDIA's libraries on this machine's first two GPUs
#   run_p2p_managed.sh --card --update   rewrite the expected file from the card
#
# nvidia/tests/data/p2p_managed.rtx3060.expected is what two RTX 3060s printed (no peer path, no concurrent
# managed access); the simulated rtx3060 profile must print the same. Then the same program on two simulated
# A100s must show a peer path that works (the card has no such pair here, so these answers are the documented
# ones, not measured) and managed memory that prefetches.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/p2p_managed.cu"
expected="$root/nvidia/tests/data/p2p_managed.rtx3060.expected"
out="${TMPDIR:-/tmp}/vgpu-p2p-managed.$$"
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
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "$src" -o "$out" -lcuda -L"$libs/stubs" || { echo "FAIL: does not compile"; exit 1; }
  LD_LIBRARY_PATH="$libs" "$out" > "$out.txt" || { echo "FAIL: the program failed on the card"; exit 1; }
  if head -1 "$out.txt" | grep -q "^SKIP"; then head -1 "$out.txt"; exit 0; fi
  if (( update )); then cp "$out.txt" "$expected"; echo "wrote $expected"; exit 0; fi
  diff -u "$expected" "$out.txt" && echo "peer access and managed memory match the card's" || { echo "FAIL: differs from what the card printed"; exit 1; }
  exit 0
fi

shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "${san_flags[@]}" "$src" -o "$out" -lcuda -L"$shim" \
  || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

run() {   # profile -> output file
  env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU="$1" VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" "$out" > "$2" 2>"$2.err" \
    || { echo "FAIL: $1: the program failed"; cat "$2.err"; exit 1; }
}

# The expected file is what CUDA 13 printed: an older toolkit's headers have neither the prefetch's flags nor
# the location-taking runtime calls, so the lines marked [cuda13] are not printed there.
release="$("$nvcc_bin" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
want="$out.want"
if (( ${release:-0} >= 13 )); then cp "$expected" "$want"; else grep -v '\[cuda13\]' "$expected" > "$want"; fi
run nvidia/rtx3060 "$out.3060"
diff -u "$want" "$out.3060" || { echo "FAIL: the rtx3060 profile does not print what the card printed"; exit 1; }
echo "rtx3060: the card's answers"

# A data-centre pair has a peer path.
run nvidia/a100 "$out.a100"
bad=0
expect() {   # <extended regex of a whole line>
  grep -Eq "$1" "$out.a100" || { echo "FAIL: a100: no line matches $1"; bad=1; }
}
expect '^device 0 ConcurrentManagedAccess +cudaSuccess 1$'
expect '^cudaDeviceCanAccessPeer\(0, 1\) +cudaSuccess 1$'
expect '^cudaDeviceCanAccessPeer\(1, 0\) +cudaSuccess 1$'
expect '^cudaDeviceCanAccessPeer\(0, 0\) +cudaSuccess 0$'
expect '^cudaDeviceGetP2PAttribute\(2, 0 -> 1\) +cudaSuccess 1$'
expect '^cudaDeviceGetP2PAttribute\(3, 0 -> 1\) +cudaSuccess 1$'
expect '^cudaDeviceEnablePeerAccess\(1, 0\) +cudaSuccess$'
expect '^cudaDeviceEnablePeerAccess\(1, 0\) again +cudaErrorPeerAccessAlreadyEnabled$'
expect '^cudaDeviceEnablePeerAccess\(1, 1\) +cudaErrorInvalidValue$'
expect '^cudaDeviceEnablePeerAccess\(0, 0\) \(itself\) +cudaErrorInvalidDevice$'
expect '^cudaDeviceDisablePeerAccess\(1\) +cudaSuccess$'
expect '^cudaDeviceDisablePeerAccess\(0\) +cudaErrorPeerAccessNotEnabled$'
expect '^cuDeviceCanAccessPeer\(0, 1\) +CUDA_SUCCESS 1$'
expect '^cuDeviceGetP2PAttribute\(2, 0 -> 1\) +CUDA_SUCCESS 1$'
expect '^cuCtxEnablePeerAccess\(other, 0\) +CUDA_SUCCESS$'
expect '^cuCtxEnablePeerAccess\(other, 0\) again +CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED$'
expect '^cuCtxDisablePeerAccess\(other\) +CUDA_SUCCESS$'
expect '^cuMemPrefetchAsync\(managed, device 0\) +CUDA_SUCCESS$'
expect '^cuMemAdvise preferred location \(device 0\) +CUDA_SUCCESS$'
expect '^the first byte after the peer copy +5a$'
(( bad == 0 )) || exit 1
echo "a100: a peer path that works"
echo "PASS"
