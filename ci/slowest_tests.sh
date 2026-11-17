#!/usr/bin/env bash
# The slowest tests of the last ctest run in a build directory, and a budget for one test.
#
#   ci/slowest_tests.sh BUILD_DIR [TOP]
#
# Reads BUILD_DIR/Testing/Temporary/LastTest.log (what ctest wrote for its last run there, so run this right after the ctest step, once per step),
# prints the TOP (default 20) longest tests, and writes the same as a table to the job's summary ($GITHUB_STEP_SUMMARY). The job's wall time is
# the sum of these divided by the -j it ran at, and the longest one is what a job's time cannot go below, so this is where to look first.
#
# Budget (seconds, one test): a test over CI_TEST_WARN_SECONDS (default 1500, 25 minutes) is a warning; over CI_TEST_LIMIT_SECONDS (default 1800,
# 30 minutes) fails the job. A test that long is split into shards (vgpu_shard_tests, docs/ci.md, "Shards and the compile cache"), which
# run side by side. CI_TEST_LIMIT_SECONDS=0 turns the failing off; the report stays.
#
# Never fails for want of a log (ctest killed, the step skipped): it says so and exits 0.
set -uo pipefail
dir=${1:?usage: ci/slowest_tests.sh BUILD_DIR [TOP]}
top=${2:-20}
warn=${CI_TEST_WARN_SECONDS:-1500}
limit=${CI_TEST_LIMIT_SECONDS:-1800}
log="$dir/Testing/Temporary/LastTest.log"
[[ -r $log ]] || { echo "slowest tests: no $log (ctest did not run here)"; exit 0; }

# "N/M Testing: name" opens a test's section and "Test time =  S sec" closes it.
times="$(awk '
  /^[0-9]+\/[0-9]+ Testing: / { name = $3; next }
  /^Test time = / && name != "" { printf "%s\t%s\n", $4, name; name = "" }
' "$log" | LC_ALL=C sort -t$'\t' -k1,1 -rn)"
[[ -n $times ]] || { echo "slowest tests: $log lists no test times"; exit 0; }
count=$(wc -l <<< "$times")
total=$(awk -F'\t' '{ s += $1 } END { printf "%d", s }' <<< "$times")

echo "slowest $top of $count tests in $dir (added up: $((total / 60)) min; ctest -j overlaps them):"
head -n "$top" <<< "$times" | awk -F'\t' '{ printf "  %7.0f s  %5.1f min  %s\n", $1, $1 / 60, $2 }'
if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
  {
    echo "### Slowest tests in \`$dir\`"
    echo
    echo "$count tests, $((total / 60)) min added up (ctest runs them side by side)."
    echo
    echo "| seconds | minutes | test |"
    echo "|---:|---:|---|"
    head -n "$top" <<< "$times" | awk -F'\t' '{ printf "| %.0f | %.1f | `%s` |\n", $1, $1 / 60, $2 }'
    echo
  } >> "$GITHUB_STEP_SUMMARY"
fi

status=0
while IFS=$'\t' read -r secs name; do
  if awk -v s="$secs" -v l="$limit" 'BEGIN { exit !(l > 0 && s > l) }'; then
    echo "::error::$name took $(awk -v s="$secs" 'BEGIN { printf "%.0f", s / 60 }') min, over the $((limit / 60)) min a test may take: split it into shards (docs/ci.md)"
    status=1
  elif awk -v s="$secs" -v w="$warn" 'BEGIN { exit !(s > w) }'; then
    echo "::warning::$name took $(awk -v s="$secs" 'BEGIN { printf "%.0f", s / 60 }') min, over $((warn / 60)) min: it is close to the $((limit / 60)) min a test may take"
  fi
done <<< "$(awk -F'\t' -v w="$warn" '$1 > w' <<< "$times")"
exit $status
