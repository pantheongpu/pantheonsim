#!/usr/bin/env bash
# Card-oracle diff: a fixed set of kernels (tools/oracle-diff/manifest.txt), outputs compared bit for bit against a card's.
#
#   oracle-diff.sh            simulator vs the pinned card files (no GPU; needs a build and nvcc)
#   oracle-diff.sh --card     NVIDIA's runtime on the real GPU vs the same pinned files (needs nvcc, nvidia-smi, no build)
#
# Both modes diff against one pinned file, so a pass in both is simulator == card, line by line. The card mode is what notices a
# card, driver or library that now answers differently from the file. A script that skips (no nvcc, no shim) counts as a failure:
# a nightly that compared nothing must not be green. Output: a table on stdout, a copy in $GITHUB_STEP_SUMMARY and in
# $ORACLE_DIFF_OUT/summary.md (default: a temp dir), each script's log beside it. Exit 0 only if every script passed.
#
# ORACLE_E2E_DIR and ORACLE_MANIFEST override the script directory and the manifest (the self test uses stubs).
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
e2e="${ORACLE_E2E_DIR:-$root/nvidia/tests/e2e}"
manifest="${ORACLE_MANIFEST:-$root/tools/oracle-diff/manifest.txt}"
mode=sim; args=()
case "${1:-}" in
  "") ;;
  --card) mode=card; args=(--card) ;;
  *) echo "usage: $0 [--card]" >&2; exit 2 ;;
esac
out="${ORACLE_DIFF_OUT:-$(mktemp -d "${TMPDIR:-/tmp}/oracle_diff.XXXXXX")}"
mkdir -p "$out"
if [[ $mode == card ]]; then
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 || { echo "FAIL: --card needs a GPU visible to nvidia-smi"; exit 1; }
  gpu="$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1)"
else
  gpu="simulator"
fi
scripts=()
while IFS= read -r l; do l="${l%%#*}"; l="${l//[[:space:]]/}"; [[ -n $l ]] && scripts+=("$l"); done < "$manifest"
(( ${#scripts[@]} )) || { echo "FAIL: empty manifest $manifest"; exit 1; }
rows=(); failed=(); bad=0
for s in "${scripts[@]}"; do
  log="$out/${s%.sh}.$mode.log"
  if [[ ! -x $e2e/$s ]]; then verdict="FAIL (script missing or not executable)"; bad=1
  else
    timeout "${ORACLE_TIMEOUT:-1800}" "$e2e/$s" "${args[@]}" >"$log" 2>&1; rc=$?
    if (( rc != 0 )); then verdict="FAIL (exit $rc)"; bad=1
    elif grep -q '^SKIP' "$log"; then verdict="FAIL (skipped: nothing was compared)"; bad=1
    elif ! grep -q '^PASS' "$log"; then verdict="FAIL (no PASS line)"; bad=1
    else verdict="PASS ($(grep '^PASS' "$log" | tail -1))"; fi
  fi
  [[ $verdict == FAIL* ]] && failed+=("$s")
  rows+=("| $s | $verdict |")
done
{
  echo "### Card-oracle diff ($mode: $gpu)"; echo
  echo "| script | verdict |"; echo "|---|---|"; printf '%s\n' "${rows[@]}"
} | tee "$out/summary.md" >> "${GITHUB_STEP_SUMMARY:-/dev/stdout}"
[[ -n "${GITHUB_STEP_SUMMARY:-}" ]] && cat "$out/summary.md"
for s in "${failed[@]}"; do tail -n 15 "$out/${s%.sh}.$mode.log" 2>/dev/null | sed "s|^|  [$s] |"; done
exit $bad
