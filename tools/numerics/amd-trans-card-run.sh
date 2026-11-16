#!/usr/bin/env bash
# The AMD transcendental instructions (v_rcp_f32, v_rsq_f32, v_sqrt_f32, v_exp_f32, v_log_f32, v_sin_f32, v_cos_f32) of the card in
# this machine against an RX 6800's, as 64 hashes of 2^26 results each over all 2^32 inputs (docs/numerics/math-functions.md,
# "AMD"). Needs hipcc and one AMD GPU; no network, no profiler. About a second of GPU time an operation.
#
#   ROCR_VISIBLE_DEVICES=<one device> tools/numerics/amd-trans-card-run.sh <out-dir>
#
# Name ONE device (two at once make a kernel 50x slower on a mixed box). Leaves hashes.txt and compare.txt in <out-dir>;
# exit status 0 when every operation has the RX 6800's hashes, 1 otherwise. Environment: HIPCC, AMD_ARCH (default: the card's).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$(mkdir -p "${1:?usage: $0 <out-dir>}" && cd "$1" && pwd)"
hipcc_bin="${HIPCC:-$(command -v hipcc || ls /opt/rocm*/bin/hipcc 2>/dev/null | tail -1)}"
[[ -n "$hipcc_bin" ]] || { echo "no hipcc"; exit 2; }
arch="${AMD_ARCH:-$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | head -1)}"
[[ -n "$arch" ]] || { echo "no AMD card"; exit 2; }
"$hipcc_bin" -O2 --offload-arch="$arch" "$here/amd_trans_probe.cpp" -o "$out/amd_trans_probe" || { echo "the probe does not build"; exit 2; }
"$hipcc_bin" -O2 -fgpu-flush-denormals-to-zero --offload-arch="$arch" "$here/amd_trans_probe.cpp" -o "$out/amd_trans_probe_ftz" || exit 2
ref="$here/amd/gfx1030.hashes.txt"; status=0
: > "$out/hashes.txt"; : > "$out/compare.txt"
for op in RCP RSQ SQRT EXP LOG SIN COS; do
  timeout 300 "$out/amd_trans_probe" "$op" hash >> "$out/hashes.txt" 2> "$out/$op.err" || { echo "$op: did not run ($(head -1 "$out/$op.err"))" | tee -a "$out/compare.txt"; status=1; break; }
  rm -f "$out/$op.err"
  if diff <(grep "^$op " "$ref") <(grep "^$op " "$out/hashes.txt") > /dev/null; then echo "$op: the same as the RX 6800 on all 2^32 inputs" | tee -a "$out/compare.txt"
  else echo "$op: DIFFERS from the RX 6800" | tee -a "$out/compare.txt"; status=1; fi
done
timeout 300 "$out/amd_trans_probe_ftz" SIN hash 2>/dev/null | sed 's/^SIN /SIN_FTZ /' >> "$out/hashes.txt"
diff <(grep '^SIN_FTZ ' "$ref") <(grep '^SIN_FTZ ' "$out/hashes.txt") > /dev/null && echo "SIN_FTZ: the same as the RX 6800" | tee -a "$out/compare.txt" || { echo "SIN_FTZ: DIFFERS" | tee -a "$out/compare.txt"; status=1; }
exit $status
