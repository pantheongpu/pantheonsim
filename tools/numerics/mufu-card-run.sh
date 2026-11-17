#!/usr/bin/env bash
# The approximate-math instructions (MUFU) of the card in this machine against an RTX 3060's, without shipping dumps: every one
# of the ten sub-operations over all 2^32 inputs, as 64 hashes of 2^26 results each (docs/numerics/math-functions.md, "The
# probe"). Needs an NVIDIA card, nvcc and this directory; no simulator build, no cloud call, no network. About a second of GPU
# time an operation; the two compiles take longer.
#
#   tools/numerics/mufu-card-run.sh <out-dir>
#
# What it leaves in <out-dir> (a few hundred KB; copy the whole directory back):
#   gpu.txt            the card, driver and toolkit
#   hashes.txt         "<OP> <first input> <hash>" a chunk, "<OP> all <hash>" an operation: the file to compare
#   sass.txt           the disassembly of the code that ran, per operation: the one MUFU.<OP> instruction of each kernel
#   compare.txt        per operation: the same as the RTX 3060 (tools/numerics/mufu/rtx3060.hashes.txt) or the chunks that differ
#   check.txt          (MUFU_CHECK=1) every result against the simulator's model, include/vgpu/exec/mufu_model.hpp: the number
#                      that differ and the first of them. About a minute an operation on one host core.
#   <OP>.<start>.bin   (MUFU_DUMP="RCP:0x3f800000:0x800000 ...") the raw results of a range, for fitting a card that differs
#   hashes16.txt       the half-precision forms (MUFU.<op>.F16 and the PTX .f16/.f16x2 forms): a hash over all 65,536 patterns each
# Exit status: 0 when every operation that ran has the RTX 3060's hashes, 1 otherwise.
#
# Environment: NVCC (the compiler), MUFU_OPS (default: all ten), MUFU_CHECK, MUFU_DUMP, CUDA_VISIBLE_DEVICES (the card; the
# probe uses the first visible device).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$(mkdir -p "${1:?usage: $0 <out-dir>}" && cd "$1" && pwd)"
nvcc_bin="${NVCC:-$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | tail -1)}"
[[ -n "$nvcc_bin" ]] || { echo "no nvcc"; exit 2; }
cuda_root="$(dirname "$(dirname "$(readlink -f "$nvcc_bin")")")"
export LD_LIBRARY_PATH="$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1 || { echo "no NVIDIA card"; exit 2; }
first="${CUDA_VISIBLE_DEVICES:-0}"; first="${first%%,*}"
cap="$(nvidia-smi -i "$first" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
[[ -n "$cap" ]] || cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
{ nvidia-smi -i "$first" --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -1; "$nvcc_bin" --version | tail -2; date -u +%Y-%m-%d; } > "$out/gpu.txt"
cat "$out/gpu.txt"
"$nvcc_bin" -cubin -arch="sm_$cap" -Wno-deprecated-gpu-targets "$here/mufu_kernel.cu" -o "$out/mufu_kernel.cubin" || { echo "the kernel does not build"; exit 2; }
"$nvcc_bin" -std=c++17 -O2 -I "$here/../../include/vgpu/exec" "$here/mufu_probe.cu" -o "$out/mufu_probe" -lcuda 2> "$out/probe.build.err" || { echo "the probe does not build:"; cat "$out/probe.build.err"; exit 2; }
rm -f "$out/probe.build.err"
read -r -a ops <<< "${MUFU_OPS:-COS SIN EX2 LG2 RCP RSQ RCP64H RSQ64H SQRT TANH}"
ref="$here/mufu/rtx3060.hashes.txt"
: > "$out/hashes.txt"; : > "$out/sass.txt"; : > "$out/compare.txt"
[[ "${MUFU_CHECK:-0}" == 1 ]] && : > "$out/check.txt"
status=0
for op in "${ops[@]}"; do
  # The code that runs: the kernels with their MUFU rewritten to this operation. It must hold MUFU.<op> and no other MUFU.
  "$out/mufu_probe" "$out/mufu_kernel.cubin" "$op" --cubin-out "$out/rewritten.cubin" > /dev/null || { echo "$op: not rewritten" | tee -a "$out/compare.txt"; status=1; continue; }
  if command -v nvdisasm > /dev/null || [[ -x "$cuda_root/bin/nvdisasm" ]]; then
    dis="$(command -v nvdisasm || echo "$cuda_root/bin/nvdisasm")"
    { echo "== $op"; "$dis" "$out/rewritten.cubin" | grep -E '^\.text\.|MUFU' | sed 's/^ *//'; } >> "$out/sass.txt"
    n_op="$("$dis" "$out/rewritten.cubin" | grep -c "MUFU\.$op R[0-9]*, R[0-9]* ;")"; n_all="$("$dis" "$out/rewritten.cubin" | grep -c 'MUFU')"
    if (( n_op == 0 || n_op != n_all )); then echo "$op: the rewritten code is not MUFU.$op alone ($n_op of $n_all MUFU instructions)" | tee -a "$out/compare.txt"; status=1; continue; fi
  fi
  rm -f "$out/rewritten.cubin"
  # One process an operation: a card without the operation (MUFU.TANH before Turing) fails alone.
  if ! "$out/mufu_probe" "$out/mufu_kernel.cubin" "$op" --hash > "$out/$op.hash.tmp" 2> "$out/$op.err"; then
    echo "$op: did not run ($(head -1 "$out/$op.err"))" | tee -a "$out/compare.txt"; status=1; rm -f "$out/$op.hash.tmp"; continue
  fi
  rm -f "$out/$op.err"
  cat "$out/$op.hash.tmp" >> "$out/hashes.txt"; rm -f "$out/$op.hash.tmp"
  if [[ -f "$ref" ]]; then
    if diff <(grep "^$op " "$ref") <(grep "^$op " "$out/hashes.txt") > "$out/d.tmp"; then
      echo "$op: the same as the RTX 3060 on all 2^32 inputs" | tee -a "$out/compare.txt"
    else
      echo "$op: DIFFERS from the RTX 3060 in $(grep -c '^> ' "$out/d.tmp" | awk '{print $1 - 1}') of 64 chunks: $(grep '^> ' "$out/d.tmp" | grep -v ' all ' | cut -d' ' -f3 | head -8 | tr '\n' ' ')" | tee -a "$out/compare.txt"
      status=1
    fi
    rm -f "$out/d.tmp"
  fi
  if [[ "${MUFU_CHECK:-0}" == 1 ]]; then "$out/mufu_probe" "$out/mufu_kernel.cubin" "$op" --check >> "$out/check.txt" 2>&1; tail -1 "$out/check.txt"; fi
done
# The half-precision forms (MUFU.<op>.F16 and the PTX forms): every one of the 65,536 half patterns, hashed, against the RTX 3060's
# (tools/numerics/mufu/rtx3060.f16.hashes.txt). MUFU_CHECK=1 also checks all 2^32 registers of each against the model.
ref16="$here/mufu/rtx3060.f16.hashes.txt"
: > "$out/hashes16.txt"
if [[ -z "${MUFU_NO_F16:-}" ]]; then
  for op in COS SIN EX2 LG2 RCP RSQ SQRT TANH; do
    "$out/mufu_probe" "$out/mufu_kernel.cubin" "$op" --f16 --cubin-out "$out/rewritten.cubin" > /dev/null || { echo "$op.F16: not rewritten" | tee -a "$out/compare.txt"; status=1; continue; }
    if command -v nvdisasm > /dev/null || [[ -x "$cuda_root/bin/nvdisasm" ]]; then
      dis="$(command -v nvdisasm || echo "$cuda_root/bin/nvdisasm")"
      { echo "== $op.F16"; "$dis" "$out/rewritten.cubin" | grep -E '^\.text\.|MUFU' | sed 's/^ *//'; } >> "$out/sass.txt"
    fi
    rm -f "$out/rewritten.cubin"
    if ! "$out/mufu_probe" "$out/mufu_kernel.cubin" "$op" --f16 --hash16 >> "$out/hashes16.txt" 2> "$out/$op.err"; then
      echo "$op.F16: did not run ($(head -1 "$out/$op.err"))" | tee -a "$out/compare.txt"; status=1
    fi
    rm -f "$out/$op.err"
    if [[ "${MUFU_CHECK:-0}" == 1 ]]; then
      for fl in "" "--h1"; do "$out/mufu_probe" "$out/mufu_kernel.cubin" "$op" --f16 $fl --check >> "$out/check.txt" 2>&1; tail -1 "$out/check.txt"; done
    fi
  done
  # The PTX forms: nvcc's own code for ex2.approx.f16, .f16x2 and tanh.approx.f16, .f16x2 (cubin disassembly to sass.txt).
  if "$nvcc_bin" -std=c++17 -arch="sm_$cap" -Wno-deprecated-gpu-targets "$here/mufu_f16_forms.cu" -o "$out/mufu_f16_forms" 2> "$out/f16forms.build.err"; then
    "$out/mufu_f16_forms" >> "$out/hashes16.txt" || { echo "mufu_f16_forms did not run" | tee -a "$out/compare.txt"; status=1; }
    rm -f "$out/f16forms.build.err"
  else
    echo "mufu_f16_forms does not build here:"; head -3 "$out/f16forms.build.err" | tee -a "$out/compare.txt"; status=1
  fi
  if [[ -f "$ref16" ]]; then
    if diff <(grep -v '^#' "$ref16" | sort) <(sort "$out/hashes16.txt") > "$out/d.tmp"; then
      echo "half-precision forms: the same as the RTX 3060 on all 65,536 patterns of each" | tee -a "$out/compare.txt"
    else
      echo "half-precision forms: DIFFER from the RTX 3060: $(grep '^>' "$out/d.tmp" | cut -d' ' -f2 | tr '\n' ' ')" | tee -a "$out/compare.txt"; status=1
    fi
    rm -f "$out/d.tmp"
  fi
fi
for spec in ${MUFU_DUMP:-}; do
  IFS=: read -r op start count <<< "$spec"
  "$out/mufu_probe" "$out/mufu_kernel.cubin" "$op" --dump "$out/$op.$start.bin" --start "$start" --count "$count" && echo "dumped $op $start +$count"
done
echo "done: $out (status $status)"
exit $status
