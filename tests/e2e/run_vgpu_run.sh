#!/usr/bin/env bash
# `vgpu run` — the wrapper, and the checks that are the reason it exists.
#
# The happy path is the least interesting part: what this pins down is that the
# two ways a program silently fails to reach the simulator are caught *before*
# it starts, and named with the fix. Both are read out of the binary's dynamic
# section, so both are testable without a GPU.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${VGPU_BUILD_DIR:-$root/build}"
vgpu="$build/vgpu"
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_run_test.XXXXXX")"
trap 'rm -rf "$work"' EXIT

[[ -x "$vgpu" ]] || { echo "SKIP: $vgpu not built"; exit 0; }
[[ -e "$build/shim/libcuda.so.1" ]] || { echo "SKIP: shim not built"; exit 0; }

fails=0
check() {  # check <name> <expected-substring> -- <command...>
  local name="$1" want="$2"; shift 3
  local out; out="$("$@" 2>&1)"
  if [[ "$out" == *"$want"* ]]; then
    echo "PASS $name"
  else
    echo "FAIL $name: expected to see '$want', got:"
    sed 's/^/    /' <<<"$out"
    fails=$((fails + 1))
  fi
}

# --- things that need no compiler ---

check "unknown device is rejected by name" "no device profile named" -- \
  "$vgpu" run --gpu nvidia/nonsuch /bin/true

check "a missing program exits 127" "cannot execute" -- \
  "$vgpu" run --quiet /no/such/program

check "--print-env shows what would be set" "VGPU_RACE=1" -- \
  "$vgpu" run --gpu nvidia/a10 --race --print-env /bin/true

# An ordinary program is passed through with no advice at all: the checks below
# must not fire on everything that happens not to link CUDA.
out="$("$vgpu" run --quiet --gpu nvidia/a10 /bin/sh -c 'exit 42' 2>&1)"
code=$?
if [[ $code -eq 42 && -z "$out" ]]; then
  echo "PASS exit code passes through, non-CUDA program is not lectured"
else
  echo "FAIL exit code/quiet: code=$code out='$out'"
  fails=$((fails + 1))
fi

# A soname the shim does not carry. Built with a stub library rather than a real
# CUDA 12 toolkit, since only the DT_NEEDED string is being checked.
if command -v gcc >/dev/null 2>&1; then
  echo 'void stub(void){}' > "$work/stub.c"
  echo 'int main(void){return 0;}' > "$work/prog.c"
  # Major 99 will never be a real CUDA release, so this stays a mismatch whatever
  # toolkit the shim was built against -- and the message has to name the major
  # the shim actually carries, which differs between build directories.
  shopt -s nullglob
  shim_carts=("$build/shim"/libcudart.so.[0-9]*)
  shopt -u nullglob
  have_cart="$(basename "${shim_carts[0]}")"
  gcc -shared -fPIC -Wl,-soname,libcudart.so.99 "$work/stub.c" -o "$work/libcudart.so.99"
  # --no-as-needed keeps the dependency even though nothing in it is called.
  gcc "$work/prog.c" -Wl,--no-as-needed -L"$work" -l:libcudart.so.99 -o "$work/needs99"
  check "a soname the shim lacks is named, with what it has instead" \
        "this build of the simulator has $have_cart" -- \
    "$vgpu" run --gpu nvidia/a10 "$work/needs99"
else
  echo "SKIP: gcc not found (soname mismatch check)"
fi

# --- things that need nvcc ---

nvcc_bin="$(pick_nvcc_for_shim "$build/shim")"
nvcc_host_compiler_fix
if [[ -n "$nvcc_bin" ]]; then
  cat > "$work/k.cu" <<'CU'
#include <cstdio>
__global__ void addk(float* x, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) x[i] += 1.f;
}
int main() {
  const int n = 256; float* d; float h[n];
  for (int i = 0; i < n; ++i) h[i] = i;
  cudaMalloc(&d, n * sizeof(float));
  cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice);
  addk<<<4, 64>>>(d, n);
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  printf("h[7]=%.1f\n", h[7]);
  cudaFree(d);
  return h[7] == 8.f ? 0 : 1;
}
CU
  # The programs must carry whatever sanitizer the shim was built with, or they
  # abort loading it with "ASan runtime does not come first".
  read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$build/shim")"
  nvcc_flags=(-std=c++17 -arch=sm_86 -Wno-deprecated-gpu-targets "${san_flags[@]}")
  "$nvcc_bin" "${nvcc_flags[@]}" -cudart shared "$work/k.cu" -o "$work/shared" 2>/dev/null
  "$nvcc_bin" "${nvcc_flags[@]}" "$work/k.cu" -o "$work/static" 2>/dev/null
  # The RPATH has to name a directory that really holds the real libcudart, or
  # there is nothing for the check to catch -- and where that is depends on the
  # toolkit layout (a distribution package puts it in the multiarch directory,
  # NVIDIA's installer under targets/). Ask the loader's cache rather than
  # guessing from nvcc's location.
  real_cart="$(ldconfig -p | awk -v n="$have_cart" '$1 == n {print $NF; exit}')"
  real_dir=""
  [[ -n "$real_cart" && -e "$real_cart" ]] && real_dir="$(dirname "$real_cart")"
  # DT_RPATH, not DT_RUNPATH: --disable-new-dtags is what makes the difference,
  # and the difference is the whole point of the check.
  if [[ -n "$real_dir" ]]; then
    "$nvcc_bin" "${nvcc_flags[@]}" -cudart shared \
         -Xlinker --disable-new-dtags -Xlinker -rpath="$real_dir" \
         "$work/k.cu" -o "$work/rpath" 2>/dev/null
  fi

  check "a shared-cudart program runs and computes" "h[7]=8.0" -- \
    "$vgpu" run --quiet --gpu nvidia/a10 "$work/shared"

  out="$("$vgpu" run --quiet --gpu nvidia/a10 "$work/static" 2>&1)"
  if [[ "$out" == *"embeds GPU code but links no CUDA library"* ]]; then
    echo "PASS a static-cudart program is diagnosed before it starts"
  else
    echo "FAIL static-cudart diagnosis: $out"
    fails=$((fails + 1))
  fi

  if [[ -e "$work/rpath" ]] && readelf -d "$work/rpath" 2>/dev/null | grep -q "(RPATH)"; then
    out="$("$vgpu" run --quiet --gpu nvidia/a10 "$work/rpath" 2>&1)"
    if [[ "$out" == *"DT_RPATH"* ]]; then
      echo "PASS a DT_RPATH that shadows the shim is caught"
    else
      echo "FAIL DT_RPATH not caught: $out"
      fails=$((fails + 1))
    fi
    # And --preload gets past it, which is the advice the warning gives.
    #
    # Not under a sanitizer: LD_PRELOAD entries load before the program's own
    # DT_NEEDED, so preloading an instrumented shim puts it ahead of the
    # program's libasan and the ordering check fires. That is a real limitation
    # of --preload against an instrumented build, not a defect in the warning
    # above, and the warning is what this test is for.
    if [[ -z "$(shim_sanitizer "$build/shim")" ]]; then
      check "--preload runs it anyway" "h[7]=8.0" -- \
        "$vgpu" run --quiet --gpu nvidia/a10 --preload "$work/rpath"
    else
      echo "SKIP: --preload against a sanitizer-instrumented shim (load order)"
    fi
  else
    echo "SKIP: no DT_RPATH binary (no real libcudart directory to point at)"
  fi
else
  echo "SKIP: no nvcc matching this shim's toolkit (compiled-program checks)"
fi

if (( fails )); then
  echo "FAIL: $fails vgpu run checks failed"
  exit 1
fi
echo "RESULT: vgpu run wrapper and its pre-flight checks behave"
