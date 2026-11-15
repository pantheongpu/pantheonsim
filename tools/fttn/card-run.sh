#!/usr/bin/env bash
# Everything a rented NVIDIA card is asked for the tensor-core numerics (docs/matrix-numerics.md), in one run. Needs the card, a CUDA
# toolkit (nvcc) and this repository; no simulator build. Makes no cloud call: launch the instance, run this on it, copy <out-dir> back.
#
#   tools/fttn/card-run.sh <out-dir> [fttn-dir]
#
# What it runs, and what each answers:
#   1. FTTN's four programs and the motivating example, built for the card's own architecture (and for every older one the toolkit
#      has, since a card runs older cubins): <out>/fttn/sm_<N>-<test>.txt. The summary row and D.
#   2. nvidia/tests/e2e/run_simd_lane_forms.sh --card and run_float_half_forms.sh --card: the card's lines of the two probes against
#      the pinned file of its architecture (the RTX 3060's; the H100's for sm_90). <out>/probe/*.txt hold the card's own lines; the
#      exit status of each is recorded. For a Blackwell card (sm_100, sm_120) there is no pinned file yet: expect FAIL, and keep the lines.
#   3. The million-result dump (simd_lane_forms.cu -DTC_DUMP -DTILES=4096) into <out>/dump/t4096/dump: the data a rule is fitted
#      to. Afterwards, anywhere with python3 and numpy:   tools/fttn/fit_tc_model.py <out>/dump/t4096/dump
#      and, with a simulator build:   tc_dump_check <out>/dump/t4096/dump hopper     (tools/fttn/tc_dump_check.cpp)
#   4. The corners of the subnormal range (tools/fttn/tc_corners.cu): the two inputs that are on an RTX 3060 already (S1, S2) and
#      the four that are the published models' word only (R1, R2, R3, R4). <out>/corners.txt.
#
# Cards still wanted (AWS names; nothing here launches them):
#   B200 / RTX PRO 6000 Blackwell (g7e, p6-b200)  sm_100 / sm_120: no card run of ours; the published model (the Hopper numbers) is opt-in
#   Tesla T4 (g4dn)                               float_half_forms -DNO80 has not run on a Turing card (half_form_30: HMMA.1688 beside HMUL2.F32)
#   any Ampere / Ada / Hopper card                step 4 (R1, R3, R4 have no card run; S1, S2 only on the RTX 3060)
#   V100 (p3)                                     sm_70 needs a CUDA 12 toolkit; the simulator has no V100 profile yet
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$(mkdir -p "${1:?usage: $0 <out-dir> [fttn-dir]}" && cd "$1" && pwd)"
fttn="${2:-$out/fttn-src}"
command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1 || { echo "no card"; exit 1; }
nvcc_bin="${NVCC:-$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | tail -1)}"
[[ -n "$nvcc_bin" ]] || { echo "no nvcc"; exit 1; }
export PATH="$(dirname "$nvcc_bin"):$PATH"
cuda_root="$(dirname "$(dirname "$(readlink -f "$nvcc_bin")")")"
export LD_LIBRARY_PATH="$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
{ nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1; "$nvcc_bin" --version | tail -2; date -u +%Y-%m-%d; } > "$out/gpu.txt"
cat "$out/gpu.txt"
mkdir -p "$out/fttn" "$out/probe" "$out/dump/t4096/dump"
# 1. FTTN
"$root/tools/fttn/fetch.sh" "$fttn" > /dev/null && FTTN_NO_ROCWMMA=1 true
archs=(); for a in 75 80 86 89 90 100 120; do (( a <= cap )) && archs+=("$a"); done
"$root/tools/fttn/build.sh" nvidia "$fttn" "${archs[@]}" 2> "$out/fttn/build.err"
for a in "${archs[@]}"; do
  for t in fp16in bf16in tf-fp32in fp64in motivating; do
    [[ -x "$fttn/bin/sm_$a/$t" ]] || continue
    timeout 600 "$fttn/bin/sm_$a/$t" > "$out/fttn/sm_$a-$t.txt" 2> "$out/fttn/sm_$a-$t.err"; echo "sm_$a $t rc=$?" >> "$out/fttn/rc.txt"
    case $t in motivating) grep -a MOTIVATING "$out/fttn/sm_$a-$t.txt" ;; *) echo "sm_$a $t | $(tail -1 "$out/fttn/sm_$a-$t.txt" | tr -s ' ')" ;; esac
  done
done
# 2. the probes against the pinned files
for p in simd_lane_forms float_half_forms; do
  VGPU_CARD_OUT="$out/probe/$p.sm_$cap.txt" bash "$root/nvidia/tests/e2e/run_$p.sh" --card > "$out/probe/$p.sm_$cap.log" 2>&1
  echo "$p sm_$cap rc=$? $(tail -1 "$out/probe/$p.sm_$cap.log")" | tee -a "$out/probe/rc.txt"
done
# 3. the dump
no80=(); (( cap < 80 )) && no80=(-DNO80)
if "$nvcc_bin" -std=c++17 -O1 -cudart shared -arch="sm_$cap" -Wno-deprecated-gpu-targets "${no80[@]}" -DTC_DUMP -DTILES=4096 "$root/nvidia/tests/e2e/simd_lane_forms.cu" -o "$out/dump/slf_dump" 2> "$out/dump/build.err"; then
  ( cd "$out/dump/t4096" && "$out/dump/slf_dump" > lines.txt 2> err.txt ); echo "dump rc=$? ($(ls "$out/dump/t4096/dump" | wc -l) files)"
else
  echo "the dump program does not build: $out/dump/build.err"
fi
# 4. the corners
if "$nvcc_bin" -std=c++17 -O1 -cudart shared -arch="sm_$cap" -Wno-deprecated-gpu-targets "${no80[@]}" "$root/tools/fttn/tc_corners.cu" -o "$out/tc_corners" 2> "$out/corners.build.err"; then
  "$out/tc_corners" | tee "$out/corners.txt"
else
  echo "tc_corners does not build: $out/corners.build.err"
fi
echo "done: $out"
