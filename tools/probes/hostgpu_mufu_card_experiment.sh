#!/usr/bin/env bash
# The card experiment that decides whether more of the MUFU family may join the host-GPU exact set (docs/performance.md "Round sp43", finding 3).
# It needs ONE NVIDIA card (the A10G of the card CI is the reference; any sm_75+ card gives a valid answer for that card's family) and a CUDA toolkit. It was run
# on an A10G at stride 1 in Round sp44 (docs/perf-design/rounds/11-rounds-sp36-sp41.md): div, sqrtrn, rcprn and the harness PASS, the bare MUFU forms differ on
# 4% to 25% of the patterns. The simulator side of every comparison is unit tested without a card (tests/unit/test_host_gpu_ref.cpp). The probes of turbo
# mode (sin, cos, tanh, atomics, tensor instructions) are in tools/probes/hostgpu_turbo_card_probes.sh.
#
#   usage: tools/probes/hostgpu_mufu_card_experiment.sh <out-dir> [stride]       (stride 1 = all 2^32 patterns; the full run takes about 10 minutes on an A10G)
#
# What it runs (tools/probes/hostgpu_sfu_exhaustive.cu, every pattern of 2^32 floats through the card's instruction, compared with include/vgpu/host_gpu_ref.hpp):
#   E1  rcp rsq sqrt ex2 lg2   the .approx.ftz forms (bare MUFU): recorded, EXPECTED to differ from the simulator's correctly rounded result (they are not in the set
#                              and stay out unless one of them reports differ=0 over all 2^32 patterns, which no earlier measurement suggests: rcp differs by one
#                              ulp on about 13% of the mantissas of [1, 2)).
#   E2  div                    IEEE a / b for 12 dividends and every divisor: the division idiom (VGPU_HOST_GPU_DIV=1).
#   E3  sqrtrn, rcprn          IEEE sqrt and IEEE reciprocal of every pattern: the idioms of sqrtf(x) / 1.0f/x written as an exact operation.
#   E4  the harness's own mutation (HOSTGPU_SFU_MUTATE=1, a one-ulp nudge of the card side): each IEEE op must report differ > 0, or the comparison is blind.
#
# Acceptance (the script prints one VERDICT line per item and exits 0 only when every IEEE item passes and the harness sees its mutation):
#   VERDICT div      PASS  <=>  all 12 RESULT lines of op=div have differ=0 and tested=4294967296      -> VGPU_HOST_GPU_DIV may become the default
#   VERDICT sqrtrn   PASS  <=>  differ=0 over all 2^32 patterns                                          -> VGPU_HOST_GPU_SQRT=1 (Round sp68's recogniser) may be trusted on
#                                                                                                          this card: the refinement is NOT exact for every
#                                                                                                          rsqrt one ulp off (test_host_gpu_ref), so each
#                                                                                                          architecture needs its own PASS
#   VERDICT rcprn    PASS  <=>  differ=0 over all 2^32 patterns                                          -> same for rcp.rn
#   VERDICT harness  PASS  <=>  the three mutated runs (stride 257) each report differ > 0
#   INFO bare MUFU lines (rcp rsq sqrt ex2 lg2) are printed with their differ counts and max_ulp and never fail the script.
#
# Expected output on an A10G (from the documents; the differ counts of E1 are what the first full run will record):
#   RESULT op=div a=0x3f800000 tested=4294967296 differ=0 max_ulp=0 first=0x00000000:card=0x00000000:sim=0x00000000     (twelve lines, differ=0 each)
#   RESULT op=sqrtrn tested=4294967296 differ=0 max_ulp=0 ...
#   RESULT op=rcprn  tested=4294967296 differ=0 max_ulp=0 ...
#   INFO   op=rcp   differ=<about 13% of the mantissas, so hundreds of millions> max_ulp=1 ...   (E1 lines differ, max_ulp of 1 or 2)
#   VERDICT div PASS / VERDICT sqrtrn PASS / VERDICT rcprn PASS / VERDICT harness PASS
# A FAIL of div, sqrtrn or rcprn is a finding to report as it is (the first differing pattern is on the RESULT line): the idiom's refinement is NOT bit-exact on that
# card for that input, and the exact set must stay as it is.
#
# After this script, the second half of the evidence (needs the same card and the simulator built, one run each, 1 hour):
#   for every program of the refusal list (Rodinia backprop lud hotspot srad_v2; PolyBench ADI LU GRAMSCHM CORR; HeCBench adam mask md5hash mandelbrot matrixT lud):
#     VGPU_GPU=nvidia/a10g VGPU_HOST_GPU=exact VGPU_HOST_GPU_DIV=1 VGPU_HOST_GPU_VERIFY=all VGPU_HOST_GPU_LOG=1 <program>
#   and read the last two lines "host_gpu_verify: in_set_identical=N in_set_different=N other_identical=N other_different=N": in_set_different must be 0 for
#   the kernels the idiom added. (tools/card-ci/run-card-ci.sh already runs the division check; this adds the programs and the new ops.)
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:?usage: $0 <out-dir> [stride]}"
stride="${2:-1}"
mkdir -p "$out"
command -v nvcc > /dev/null || { echo "SKIP: no nvcc"; exit 0; }
command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1 || { echo "SKIP: no card"; exit 0; }
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
[[ -n "$cap" ]] || { echo "SKIP: cannot read the card's compute capability"; exit 0; }
nvcc -std=c++20 -O2 -arch="sm_$cap" -I "$root/include" "$root/tools/probes/hostgpu_sfu_exhaustive.cu" -o "$out/hostgpu_sfu_exhaustive" || { echo "FAIL: the probe does not build"; exit 1; }
probe="$out/hostgpu_sfu_exhaustive"
nvidia-smi -q > "$out/nvidia-smi.txt" 2>&1
fails=0
differs() { sed -n 's/.* differ=\([0-9]*\) .*/\1/p'; }
for op in rcp rsq sqrt ex2 lg2; do
  "$probe" "$op" "$stride" > "$out/$op.txt" 2>&1
  sed 's/^RESULT/INFO  /' "$out/$op.txt"
done
for op in div sqrtrn rcprn; do
  "$probe" "$op" "$stride" > "$out/$op.txt" 2>&1
  rc=$?
  total_differ=0
  for d in $(differs < "$out/$op.txt"); do total_differ=$(( total_differ + d )); done
  lines="$(grep -c '^RESULT' "$out/$op.txt")"
  want=1
  [[ $op == div ]] && want=12
  if (( rc == 0 && total_differ == 0 && lines == want )); then echo "VERDICT $op PASS ($lines RESULT lines, differ=0, stride $stride)"
  else echo "VERDICT $op FAIL (rc=$rc, $lines of $want RESULT lines, total differ=$total_differ): $(grep -m1 'differ=[1-9]' "$out/$op.txt")"; fails=1; fi
done
blind=0
for op in div sqrtrn rcprn; do
  HOSTGPU_SFU_MUTATE=1 "$probe" "$op" 257 > "$out/mutated_$op.txt" 2>&1
  m=0
  for d in $(differs < "$out/mutated_$op.txt"); do m=$(( m + d )); done
  (( m > 0 )) || { echo "harness: the mutated $op run saw no difference"; blind=1; }
done
if (( blind == 0 )); then echo "VERDICT harness PASS"; else echo "VERDICT harness FAIL"; fails=1; fi
(( stride == 1 )) || echo "NOTE: stride $stride is a smoke run: a PASS here is not the evidence (all 2^32 patterns are)"
exit $fails
