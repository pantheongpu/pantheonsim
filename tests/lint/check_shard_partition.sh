#!/usr/bin/env bash
# Proof that the shards of a sharded test cover the whole test, every item once.
#
#   check_shard_partition.sh ENVVAR SPEC... -- COMMAND [ARG...]
#
# COMMAND lists the items (a program, a corpus line, a stage) its run would take, one a line, and runs nothing, when its environment says so
# (VGPU_TIER_LIST=1, VGPU_JIT_LIST=1: the test's ENVIRONMENT sets it). It takes its shard from the environment variable ENVVAR; none is the whole
# test. This runs it once whole and once per SPEC (the specs CMakeLists.txt gave the shard tests, so what is checked is what ctest runs) and fails
# unless
#   - the whole list is not empty and has no item twice,
#   - every shard's list is not empty,
#   - the shards' lists together, as a multiset, are the whole list: no item missing, none run by two shards, none that the whole does not run.
# Lines that are not items (the test framework's "[ PASS ]" and "tests passed" lines) are ignored.
set -uo pipefail
(( $# >= 4 )) || { echo "usage: $0 ENVVAR SPEC... -- COMMAND [ARG...]" >&2; exit 2; }
var=$1; shift
specs=()
while (( $# )) && [[ $1 != -- ]]; do specs+=("$1"); shift; done
[[ ${1:-} == -- ]] || { echo "$0: no -- before the command" >&2; exit 2; }
shift
(( ${#specs[@]} >= 2 )) || { echo "$0: fewer than two shards is not a partition worth checking" >&2; exit 2; }

tmp="$(mktemp -d "${TMPDIR:-/tmp}/shard-partition.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

run_list() {   # SPEC-or-empty
  local spec=$1 out rc
  if [[ -n $spec ]]; then out="$(env "$var=$spec" "${cmd[@]}" 2>/dev/null)"; else out="$(env -u "$var" "${cmd[@]}" 2>/dev/null)"; fi
  rc=$?
  (( rc == 0 )) || { echo "FAIL: the list run exited $rc (${spec:-whole})" >&2; return 1; }
  grep -v '^\[ \|tests passed$' <<< "$out" | grep . | LC_ALL=C sort
}
cmd=("$@")

run_list "" > "$tmp/whole" || exit 1
total=$(wc -l < "$tmp/whole")
(( total > 0 )) || { echo "FAIL: the whole test lists no items"; exit 1; }
dups=$(uniq -d < "$tmp/whole" | head -3)
[[ -z $dups ]] || { echo "FAIL: the whole list has an item twice: $dups"; exit 1; }

: > "$tmp/union"
for spec in "${specs[@]}"; do
  run_list "$spec" > "$tmp/shard" || exit 1
  n=$(wc -l < "$tmp/shard")
  (( n > 0 )) || { echo "FAIL: shard $spec has no items"; exit 1; }
  echo "shard $spec: $n of $total items"
  cat "$tmp/shard" >> "$tmp/union"
done
LC_ALL=C sort -o "$tmp/union" "$tmp/union"
if ! cmp -s "$tmp/whole" "$tmp/union"; then
  echo "FAIL: the shards do not partition the whole test"
  echo "  in the whole run but in no shard (or in the shards too few times):"; LC_ALL=C comm -23 "$tmp/whole" "$tmp/union" | head -5 | sed 's/^/    /'
  echo "  in the shards but not in the whole run (or in two shards):"; LC_ALL=C comm -13 "$tmp/whole" "$tmp/union" | head -5 | sed 's/^/    /'
  exit 1
fi
echo "OK: ${#specs[@]} shards, $total items, each once"
