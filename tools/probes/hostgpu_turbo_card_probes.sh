#!/usr/bin/env bash
# The card runs behind turbo host-GPU mode (docs/perf-design/12-host-gpu-turbo.md, "What a card has to say"). Needs ONE NVIDIA card and a CUDA toolkit;
# the `tensor` part also needs the simulator built (<build-dir>/shim) and a profile for the card under nvidia/profiles. Nothing here was run when it was
# written (no card): the simulator side of each comparison is unit tested (tests/unit/test_host_gpu_ref.cpp, test_host_gpu_tol.cpp).
#
#   tools/probes/hostgpu_turbo_card_probes.sh sfu     <out-dir> [stride]        sin, cos, tanh over all 2^32 float patterns (stride 1; about 5 minutes on an A10G)
#   tools/probes/hostgpu_turbo_card_probes.sh atomics <out-dir> [runs]          does the card repeat itself on each kind of atomic (default 8 runs)
#   tools/probes/hostgpu_turbo_card_probes.sh tensor  <out-dir> <build-dir>     HMMA / IMMA / DMMA kernels: the card's memory against the simulator's, by tolerance
#   tools/probes/hostgpu_turbo_card_probes.sh all     <out-dir> <build-dir>     the three, with the defaults
#
# Output, on stdout and in <out-dir>:
#   sfu      "RESULT op=sin tested=4294967296 differ=N max_ulp=N first=0x<in>:card=0x<bits>:sim=0x<bits> share=P%" for sin, cos and tanh. Reported, not judged:
#            the three are expected to differ from the simulator (as rcp, rsq, sqrt, ex2 and lg2 do: 4% to 25% of the patterns, 1 to 2 ulp, sp44).
#   atomics  "RESULT kind=<k> part=<sums|seen> runs=N repeats=M ..." and "VERDICT kind=<k> repeats|varies" for imax (control), iadd, fred, fatom, dadd, cas, exch.
#            Exit 1 only when the control varies.
#   tensor   per program (wmma_gemm, wmma_types, mma_forms) and tolerance (rtol=1e-5: the default; rtol=2e-3: one ulp of a binary16 is 1e-3; rtol=f16:
#            2e-3 with VGPU_HOST_GPU_TOL_TYPE=f16, every word read as two halves, which is the only reading that judges half results; float and
#            integer memory of the same launch then shows as outside, so read the three lines together):
#            "TENSOR <program> rtol=<r>: host_gpu_turbo: ... words_different=N words_outside=N ... max_rel=R", the host_gpu_outside and host_gpu_refused
#            lines, and "TENSOR <program> output: same|DIFFERENT" (the program's own output against a run with the mode off; under verify it must be the same).
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
what="${1:?usage: $0 sfu|atomics|tensor|all <out-dir> [...]}"
out="${2:?usage: $0 sfu|atomics|tensor|all <out-dir> [...]}"
mkdir -p "$out"
command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1 || { echo "SKIP: no card"; exit 0; }
nvcc_bin="${NVCC:-$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | tail -1)}"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc"; exit 0; }
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[[ -n "$cap" ]] || { echo "SKIP: cannot read the card's compute capability"; exit 0; }
echo "card: $name (sm_$cap), toolkit: $("$nvcc_bin" --version | tail -1)"
status=0

sfu() {
  local stride="${1:-1}"
  "$nvcc_bin" -std=c++20 -O2 -arch="sm_$cap" -w -I "$root/include" "$root/tools/probes/hostgpu_sfu_exhaustive.cu" -o "$out/hostgpu_sfu_exhaustive" || { echo "FAIL: the SFU probe does not build"; status=1; return; }
  for op in sin cos tanh; do
    "$out/hostgpu_sfu_exhaustive" "$op" "$stride" > "$out/$op.txt" 2>&1
    grep '^RESULT' "$out/$op.txt" || { echo "FAIL: $op printed no RESULT line"; tail -3 "$out/$op.txt"; status=1; }
  done
  (( stride == 1 )) || echo "NOTE: stride $stride is a smoke run, not all 2^32 patterns"
}

atomics() {
  local runs="${1:-8}"
  "$nvcc_bin" -std=c++17 -O2 -arch="sm_$cap" -w "$root/tools/probes/hostgpu_atomic_repeat.cu" -o "$out/hostgpu_atomic_repeat" || { echo "FAIL: the atomics probe does not build"; status=1; return; }
  "$out/hostgpu_atomic_repeat" "$runs" | tee "$out/atomics.txt"
  [[ "${PIPESTATUS[0]}" == 0 ]] || { echo "FAIL: the atomics probe's control varied, or the probe failed"; status=1; }
}

tensor() {
  local build="${1:?the tensor part needs the build directory of the simulator}"
  [[ -d "$build/shim" ]] || { echo "FAIL: no $build/shim"; status=1; return; }
  local id profile
  id="$(echo "$name" | tr 'A-Z' 'a-z' | sed -e 's/^nvidia //' -e 's/^geforce //' -e 's/ //g')"
  profile="${VGPU_HOST_GPU_TEST_PROFILE:-nvidia/$id}"
  [[ -f "$root/nvidia/profiles/${profile#nvidia/}.yaml" ]] || { echo "FAIL: no simulator profile for the card ($name -> $profile)"; status=1; return; }
  (( cap >= 80 )) || { echo "SKIP: the tensor programs need sm_80 or later"; return; }
  local p
  for p in wmma_gemm wmma_types mma_forms; do
    "$nvcc_bin" -std=c++17 -O1 -cudart shared -arch="sm_$cap" -w -Wno-deprecated-gpu-targets "$root/nvidia/tests/e2e/$p.cu" -o "$out/$p" 2> "$out/$p.build.log" \
      || { echo "TENSOR $p: does not build for sm_$cap ($(head -1 "$out/$p.build.log"))"; continue; }
    VGPU_GPU="$profile" LD_LIBRARY_PATH="$build/shim" VGPU_QUIET=1 "$out/$p" > "$out/$p.off.out" 2> "$out/$p.off.err"
    local rtol tol_env
    for rtol in 1e-5 2e-3 f16; do
      tol_env=(VGPU_HOST_GPU_RTOL="$rtol")
      [[ $rtol == f16 ]] && tol_env=(VGPU_HOST_GPU_RTOL=2e-3 VGPU_HOST_GPU_TOL_TYPE=f16)
      env VGPU_GPU="$profile" LD_LIBRARY_PATH="$build/shim" VGPU_HOST_GPU=turbo VGPU_HOST_GPU_VERIFY=all "${tol_env[@]}" "$out/$p" > "$out/$p.$rtol.out" 2> "$out/$p.$rtol.err"
      if grep -q 'mode off' "$out/$p.$rtol.err"; then echo "TENSOR $p rtol=$rtol: $(grep -m1 'mode off' "$out/$p.$rtol.err")"; status=1; continue; fi
      echo "TENSOR $p rtol=$rtol: $(grep '^host_gpu_turbo:' "$out/$p.$rtol.err" | tail -1)"
      echo "TENSOR $p rtol=$rtol: $(grep '^host_gpu_verify:' "$out/$p.$rtol.err" | tail -1)"
      grep '^host_gpu_outside:\|^host_gpu_refused:\|^host_gpu_refusals:' "$out/$p.$rtol.err" | sed "s/^/TENSOR $p rtol=$rtol: /"
      if cmp -s "$out/$p.off.out" "$out/$p.$rtol.out"; then echo "TENSOR $p rtol=$rtol output: same"; else echo "TENSOR $p rtol=$rtol output: DIFFERENT"; status=1; fi
    done
  done
}

case "$what" in
  sfu) sfu "${3:-1}" ;;
  atomics) atomics "${3:-8}" ;;
  tensor) tensor "${3:-}" ;;
  all) sfu 1; atomics 8; tensor "${3:-}" ;;
  *) echo "usage: $0 sfu|atomics|tensor|all <out-dir> [...]"; exit 2 ;;
esac
exit $status
