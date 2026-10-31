#!/usr/bin/env bash
# Run ctest and fail if a test skipped that is not on the allowlist.
#
#   ci/ctest-no-skips.sh BUILD_DIR ALLOW_REGEX [ctest arguments...]
#
# ALLOW_REGEX is an extended regular expression matched against whole test
# names (use '^$' for none). A skipped test, or one ctest could not find, that
# it does not match fails the job: a skip that nobody chose is a test that
# silently stopped running. ctest hides a skipped test's output, so the SKIP
# line that says why is printed for each skipped test.
set -uo pipefail
build=${1:?usage: $0 BUILD_DIR ALLOW_REGEX [ctest args...]}
allow=${2:?usage: $0 BUILD_DIR ALLOW_REGEX [ctest args...]}
shift 2
log=$(mktemp)
trap 'rm -f "$log"' EXIT

ctest --test-dir "$build" --output-on-failure "$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

skipped=$(grep -oE 'Test +#[0-9]+: [A-Za-z0-9_]+ \.+ *\*{3}(Skipped|Not Run)' "$log" \
          | sed -E 's/Test +#[0-9]+: ([A-Za-z0-9_]+).*/\1/' | sort -u || true)
if [[ -n "$skipped" ]]; then
  echo "Skipped tests and why:"
  ctest --test-dir "$build" -V -R "^($(paste -sd'|' <<<"$skipped"))\$" 2>&1 \
    | grep -E '^[0-9]+: SKIP:' | sed 's/^/  /' || true
fi
unexpected=$(grep -vE "$allow" <<<"$skipped" | grep . || true)
if [[ -n "$unexpected" ]]; then
  echo "::error::tests skipped that should run here: $(tr '\n' ' ' <<<"$unexpected")"
  exit 1
fi
exit "$status"
