#!/usr/bin/env bash
# Keep the "workloads failing on main" issue in step with the Workloads run.
#
# While main is broken there is one open issue saying where, updated by each
# run that fails and closed by the first run that passes. Run as the last step
# of .github/workflows/workloads.yml, after workload_matrix.py wrote matrix.md.
#
# RESULT is the result of the machine jobs:
#   success    close the open issue (if any), saying the run passed
#   cancelled  do nothing: a newer push superseded this run, so it says nothing
#              about main (issue #376 was opened by a run a later push cancelled)
#   anything else (failure, skipped) open the issue, or comment on the open one
#
# Environment: RESULT, RUN_URL, GITHUB_EVENT_NAME, GITHUB_SHA, and gh's GH_TOKEN.
# WORKLOADS_ISSUE_LABEL overrides the label (for tests); matrix.md is read from
# the current directory.
set -euo pipefail

label="${WORKLOADS_ISSUE_LABEL:-workloads-failing}"
sha="${GITHUB_SHA:-}"

if [[ "${RESULT:-}" == cancelled ]]; then
  echo "The machine jobs were cancelled (a newer run took over): no issue opened, updated or closed."
  exit 0
fi

gh label create "$label" --color B60205 --description "The nightly pantheon workloads run is failing" --force >/dev/null
open=$(gh issue list --label "$label" --state open --json number --jq '.[0].number // empty')
if [[ "${RESULT:-}" == success ]]; then
  [[ -n "$open" ]] && gh issue close "$open" --comment "Passing again on every machine: $RUN_URL"
  exit 0
fi
{ echo "Workloads are failing on main (${GITHUB_EVENT_NAME:-}, ${sha:0:7}): $RUN_URL"; echo; cat matrix.md; } > body.md
if [[ -n "$open" ]]; then
  gh issue comment "$open" --body-file body.md
else
  gh issue create --title "Pantheon workloads failing on main" --label "$label" --body-file body.md
fi
