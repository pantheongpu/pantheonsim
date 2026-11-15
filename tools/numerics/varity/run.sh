#!/usr/bin/env bash
# Runs every program build.sh built, each over its input sets, and writes one line file a variant: <out-dir>/varity.<variant>.txt,
# lines "<program>.<input set> <result as %.17g> <result bits>" (FAILED, TIMEOUT, CRASHED or EMPTY-KERNEL in place of a result). Two runs are
# compared with tools/numerics/compare.py <out-a> <out-b>.
#
#   tools/numerics/varity/run.sh <gen-dir> <bin-dir> <out-dir> [ENV=value ...]
#
# On a card: no ENV. On the simulator: VGPU_GPU=nvidia/t4 VGPU_SASS=1 LD_LIBRARY_PATH=<build>/shim ... (tools/numerics/run-sim.sh does it).
# JOBS=4 programs at once.
set -uo pipefail
gen="$(cd "${1:?usage: $0 <gen-dir> <bin-dir> <out-dir> [ENV=value ...]}" && pwd)"; bin="$(cd "${2:?}" && pwd)"; mkdir -p "${3:?}"; out="$(cd "$3" && pwd)"; shift 3
tmp="$(mktemp -d "${TMPDIR:-/tmp}/varity_run.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
run_one() {   # <exe name> [ENV=value ...]
  local exe="$1"; shift
  local p="${exe%.*}"
  env "$@" timeout "${VARITY_TIMEOUT:-120}" "$bin/$exe" "$gen/$p.inputs" > "$tmp/$exe.out" 2> /dev/null
  local rc=$?
  if (( rc == 124 )); then echo "$p.all TIMEOUT" >> "$tmp/$exe.out"; elif (( rc != 0 )); then echo "$p.rc CRASHED rc=$rc" >> "$tmp/$exe.out"; fi
}
export -f run_one; export gen bin tmp
ls "$bin" | grep -E '\.(O0|O3|O0_fm|O3_fm)$' | xargs -P "${JOBS:-4}" -I{} bash -c 'run_one "$@"' _ {} "$@"
# (a program build.sh left out because its kernel is empty: one line, so that two runs of the same build list the same programs)
if [[ -s "$bin/empty-kernel.txt" ]]; then while read -r e; do echo "${e%.*}.all EMPTY-KERNEL" > "$tmp/$e.out"; done < "$bin/empty-kernel.txt"; fi
for v in O0 O3 O0_fm O3_fm; do
  for f in $(ls "$tmp" | grep -E "\.$v\.out$" | sort); do cat "$tmp/$f"; done > "$out/varity.$v.txt"
  echo "varity.$v.txt: $(wc -l < "$out/varity.$v.txt") lines, $(grep -c -E 'FAILED|TIMEOUT|CRASHED|EMPTY-KERNEL' "$out/varity.$v.txt") not run"
done
