#!/usr/bin/env bash
# ci/slowest_tests.sh on a made-up ctest log: it lists the longest first, warns past the warning mark, fails past the limit, can have the
# limit turned off, and says (without failing) that there is nothing to read.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
script="$root/ci/slowest_tests.sh"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/slowest-tests.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
fail() { echo "FAIL: $*"; exit 1; }

mkdir -p "$tmp/b/Testing/Temporary" "$tmp/empty"
cat > "$tmp/b/Testing/Temporary/LastTest.log" <<'L'
1/4 Testing: quick_one
1/4 Test: quick_one
Command: "/bin/true"
Output:
Test time =   1.20 sec
----------------------------------------------------------
Test Passed.
2/4 Testing: slow_one
2/4 Test: slow_one
Output:
a line that says Test time = nonsense inside the output
Test time = 1600.00 sec
----------------------------------------------------------
3/4 Testing: huge_one
3/4 Test: huge_one
Test time = 2000.50 sec
4/4 Testing: middle_one
4/4 Test: middle_one
Test time =  90.00 sec
L

out="$(CI_TEST_LIMIT_SECONDS=0 "$script" "$tmp/b" 2 2>&1)"; rc=$?
(( rc == 0 )) || fail "the limit turned off still failed ($rc)"
first="$(grep -n 'huge_one' <<< "$out" | head -1 | cut -d: -f1)"; second="$(grep -n 'slow_one' <<< "$out" | head -1 | cut -d: -f1)"
[[ -n $first && -n $second && $first -lt $second ]] || fail "the longest test is not listed first: $out"
grep -q 'quick_one' <<< "$(grep -v '^::' <<< "$out")" && fail "TOP 2 listed a third test"
grep -q '^::warning::huge_one' <<< "$out" || fail "no warning for huge_one with the limit off: $out"

out="$("$script" "$tmp/b" 2>&1)"; rc=$?
(( rc == 1 )) || fail "a test over the limit did not fail the job ($rc)"
grep -q '^::error::huge_one took 33 min' <<< "$out" || fail "no error naming huge_one: $out"
grep -q '^::warning::slow_one took 27 min' <<< "$out" || fail "no warning for slow_one: $out"
grep -q 'middle_one' <<< "$(grep '^::' <<< "$out")" && fail "a 90 s test was flagged"

CI_TEST_LIMIT_SECONDS=3000 "$script" "$tmp/b" >/dev/null 2>&1 || fail "nothing over a 3000 s limit, yet it failed"

GITHUB_STEP_SUMMARY="$tmp/summary.md" CI_TEST_LIMIT_SECONDS=0 "$script" "$tmp/b" 3 >/dev/null 2>&1
grep -q '^| 2000 | 33.3 | `huge_one` |' "$tmp/summary.md" || fail "the summary table has no row for huge_one: $(cat "$tmp/summary.md")"

out="$("$script" "$tmp/empty" 2>&1)"; rc=$?
(( rc == 0 )) || fail "no log failed the job ($rc)"
grep -q 'no .*LastTest.log' <<< "$out" || fail "no note about the missing log: $out"
echo "OK"
