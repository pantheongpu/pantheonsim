#!/usr/bin/env bash
# ci/external/hip-tests.sh `run`, checked against a fake Catch2 suite (a shell script that lists
# three cases) with its own baseline: a case that does not finish in a slow suite is SLOW and
# not a failure, a slow suite also runs its baseline `timeout` cases, and in an ordinary suite
# a timeout outside the baseline is still a failure while a baseline timeout is still skipped.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for v in $(env | sed -n 's/^\(VGPU_[A-Za-z0-9_]*\|HT_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done

# The script looks for its baseline next to itself, so it runs from a copy.
mkdir -p "$tmp/ci/hip-tests" "$tmp/tree/build-gfx942/catch_tests"
cp "$root/ci/external/hip-tests.sh" "$tmp/ci/"
printf 'Unit_baseline_hang\ttimeout\n' > "$tmp/ci/hip-tests/baseline-amd-mi300x.tsv"
: > "$tmp/ci/hip-tests/disabled-gfx942.txt"
for s in FakeSlow FakeFast; do
  cat > "$tmp/tree/build-gfx942/catch_tests/$s" <<'SUITE'
#!/usr/bin/env bash
# A Catch2 stand-in: the case named on the command line is run, `--list-test-names-only` lists them.
if [[ $1 == --list-test-names-only ]]; then printf 'Unit_quick\nUnit_baseline_hang\nUnit_unknown_hang\n'; exit 0; fi
case $1 in
  Unit_quick) echo "All tests passed (1 assertion in 1 test case)"; exit 0 ;;
  *_hang) sleep 4; exit 0 ;;
esac
SUITE
  chmod +x "$tmp/tree/build-gfx942/catch_tests/$s"
done

fail=0
check() { if eval "$2"; then echo "ok    $1"; else echo "FAIL  $1"; fail=1; fi; }
run() {   # run <report> <env...> -- <suites>
  local rep=$1; shift
  env "$@" "$tmp/ci/hip-tests.sh" run "$tmp/tree" amd/mi300x "$tmp" "$tmp/$rep" FakeSlow >"$tmp/$rep.out" 2>&1
}
res() { awk -F'\t' -v t="$2" '$2 == t {print $3}' "$tmp/$1/results.tsv"; }

# A slow suite: nothing is skipped, the cases that do not finish in 2 s are SLOW, and the run passes.
run slow HT_SLOW_SUITES="FakeSlow" HT_SLOW_TIMEOUT=2 HT_TIMEOUT=1; rc=$?
check "slow suite: exit status 0"                  "[[ $rc == 0 ]]"
check "slow suite: quick case passes"              "[[ \$(res slow Unit_quick) == pass ]]"
check "slow suite: baseline timeout case is run"   "[[ \$(res slow Unit_baseline_hang) == slow ]]"
check "slow suite: unknown hang is SLOW"           "[[ \$(res slow Unit_unknown_hang) == slow ]]"
check "slow suite: listed in slow.txt"             "[[ \$(grep -c '^SLOW' $tmp/slow/slow.txt) == 2 ]]"
check "slow suite: no failure recorded"            "[[ ! -s $tmp/slow/new.txt ]]"
check "slow suite: summary counts them"            "grep -q '0 new failures, 2 slow' $tmp/slow/summary.md"

# The same suite, ordinary: the baseline timeout is skipped, the unknown one is a failure.
run plain HT_TIMEOUT=2; rc=$?
check "plain suite: exit status 1"                 "[[ $rc == 1 ]]"
check "plain suite: baseline timeout skipped"      "[[ \$(res plain Unit_baseline_hang) == skipped-slow ]]"
check "plain suite: unknown hang is a timeout"     "[[ \$(res plain Unit_unknown_hang) == timeout ]]"
check "plain suite: reported as NEW"               "grep -q 'NEW  timeout: FakeSlow Unit_unknown_hang' $tmp/plain/new.txt"

# A slow-suite case that finishes inside its limit is judged as usual (here: passes, so good news).
run fin HT_SLOW_SUITES="FakeSlow" HT_SLOW_TIMEOUT=60; rc=$?
check "slow suite, long limit: all pass"           "[[ $rc == 0 && \$(res fin Unit_baseline_hang) == pass ]]"
check "slow suite, long limit: good news printed"  "grep -q 'now passes (baseline: timeout): FakeSlow Unit_baseline_hang' $tmp/fin.out"
exit $fail
