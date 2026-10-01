#!/usr/bin/env bash
# CUTLASS's Blackwell fused multi-head attention (examples/77_blackwell_fmha),
# unmodified, on a simulated B200: the forward kernels -- tcgen05.mma for
# Q*K^T and P*V, the softmax between them in registers, TMA loads -- each
# checked by the example's own reference (--verify), which prints [OK] or
# [FAIL] per kernel. The problems are the example's own tests, with
# --iterations=0 so nothing is timed.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
if [[ -n "$(shim_sanitizer "$shim")" ]]; then
  echo "SKIP: CUTLASS's examples run against the uninstrumented build"; exit 0
fi
probe="${TMPDIR:-/tmp}/vgpu_cutlass_fmha_probe_$$.cu"
echo '__global__ void k() {}' > "$probe"
if ! nvcc -arch=compute_100a -code=compute_100a -c "$probe" -o /dev/null 2>/dev/null; then
  rm -f "$probe"
  echo "SKIP: this nvcc cannot target sm_100a (needs CUDA 12.8 or later)"; exit 0
fi
rm -f "$probe"
. "$root/nvidia/tests/e2e/cutlass_fetch.sh"
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_cutlass_fmha.XXXXXX")"
trap 'rm -rf "$work"' EXIT

ex="$cutlass/examples/77_blackwell_fmha"
start=$SECONDS
if ! nvcc -std=c++17 -O1 -cudart shared -arch=compute_100a -code=compute_100a --expt-relaxed-constexpr \
     --use_fast_math -I "$cutlass/include" -I "$cutlass/tools/util/include" -I "$cutlass/examples/common" \
     -I "$ex" "$ex/77_blackwell_fmha.cu" -o "$work/fmha" >"$work/fmha.log" 2>&1; then
  echo "FAIL: 77_blackwell_fmha did not compile"; tail -20 "$work/fmha.log"; exit 1
fi
echo "compiled 77_blackwell_fmha in $((SECONDS - start)) s"
if ! require_shim_libs "$shim" "$work/fmha"; then exit 0; fi

# From the example's CMakeLists.txt: plain, causal, GQA at d = 64, and a
# ragged batch with the causal and residual masks.
cases=(
  "--b=1 --h=4 --q=512 --k=512 --d=128 --verify --mask=no"
  "--b=1 --h=4 --q=512 --k=512 --d=128 --verify --mask=causal"
  "--b=2 --h=4 --h_k=2 --q=512 --k=512 --d=64 --verify"
  "--verify --varlen --mask=causal,residual --d=64 --h=4 --h_k=2 --varlen-q=17:10 --varlen-k=13:10"
)
fail=0
for args in "${cases[@]}"; do
  start=$SECONDS
  # shellcheck disable=SC2086
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/b200 LD_LIBRARY_PATH="$shim" "$work/fmha" $args --iterations=0 2>&1)" \
    && rc=0 || rc=$?
  ok=$(grep -c '^ \[OK\]' <<<"$result" || true)
  bad=$(grep -cE '^\[FAIL\]|^ \[--\]' <<<"$result" || true)
  echo "fmha $args: $ok verified, $bad not (exit $rc, $((SECONDS - start)) s)"
  if (( rc != 0 || ok == 0 || bad != 0 )); then
    grep -E 'FAIL|\[--\]|Reference|VirtualGPU|Failed|error' <<<"$result" | head -20
    fail=1
  fi
done
exit $fail
