#!/usr/bin/env bash
# Checks how run_pantheon_workloads.sh judges a finished workload (verdict.sh), under both
# bash and the plain sh the runner may be given. Needs no GPU, no simulator and no pantheon.
set -u

here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
failed=0

# check <shell> <name> <exit status> <log text> <wanted RESULT> <wanted detail fragment>
check() {
  shell="$1"; name="$2"; status="$3"; text="$4"; want="$5"; fragment="$6"
  printf '%s' "$text" >"$tmp/log"
  got="$("$shell" -c ". '$here/verdict.sh'; workload_verdict '$status' '$tmp/log'")"
  result="$(printf '%s' "$got" | cut -f1)"
  detail="$(printf '%s' "$got" | cut -f2-)"
  if [ "$result" != "$want" ] || ! printf '%s' "$detail" | grep -qF -- "$fragment"; then
    echo "FAIL [$shell] $name: wanted $want / '$fragment', got '$result' / '$detail'"
    failed=1
  fi
}

for shell in bash sh; do
  command -v "$shell" >/dev/null || continue

  check $shell "a measurement is a pass" 0 \
$'[PANTHEON] GPU 0: running\nThroughput: 12.5 GB/s\nVerification: PASS (0 errors)\n' PASS "Throughput: 12.5 GB/s"

  # What the pantheon contract prints for a deliberate skip: both lines, exit 0, no throughput.
  check $shell "a deliberate skip is a skip, not a pass" 0 \
$'[PANTHEON] GPU 0: Skipping NCCL_ALL_REDUCE (fewer than two GPUs selected).\nSkipping: fewer than two GPUs selected\n' SKIP "fewer than two GPUs selected"

  check $shell "a skip with a verification line is still a skip" 0 \
$'Verification: SKIPPED (only 1 GPU detected)\n[PANTHEON] GPU 0: Skipping P2P_MATRIX (only 1 GPU detected).\nSkipping: only 1 GPU detected\n' SKIP "only 1 GPU detected"

  check $shell "a skip cannot hide an error" 0 \
$'Skipping: no library\n[PANTHEON ERROR] NCCL_ALL_REDUCE: something broke\n' FAIL "PANTHEON ERROR"

  check $shell "a failed verification is a failure" 0 \
$'Throughput: 3 GB/s\nVerification: FAIL (1 errors)\n' FAIL "Verification: FAIL"

  check $shell "a non-zero exit is a failure with its reason" 3 \
$'Throughput: 3 GB/s\nCUDA error: an illegal memory access\n' FAIL "exit 3: CUDA error"

  check $shell "a non-zero exit with no reason is still a failure" 1 $'nothing useful\n' FAIL "exit 1"

  check $shell "exit 0 with nothing measured is a failure" 0 $'hello\n' FAIL "reported no measurement"

  # The word Skipping inside other text is not a skip; only a line that starts with it is.
  check $shell "'Skipping' mid-line does not make a skip" 0 \
$'Throughput: 9 GB/s\n  -> Skipping warm-up, not measured\n' PASS "Throughput: 9 GB/s"
done

if [ "$failed" -eq 0 ]; then
  echo "workload verdicts: all checks passed"
fi
exit "$failed"
