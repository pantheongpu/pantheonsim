#!/usr/bin/env bash
# Stop a CI job that is testing a pull request that has a merge conflict.
#
# The CI is slow and shared: a heavy job often waits hours for a runner, and by
# the time it starts main has moved and the pull request now conflicts. The whole
# run is wasted, because the author has to rebase and push, which starts the CI
# over. GitHub does not start pull_request workflows on a pull request that
# conflicts when the push arrives, but it does nothing about a conflict that
# appears while jobs are queued or running.
#
# Run this as the first step of a job (after the checkout; see
# .github/actions/pr-guard, which is that step). It asks the API for the pull
# request's mergeable_state. On `dirty` (a merge conflict) it cancels the whole
# run and exits 1. Everything else is left alone: `behind`, `blocked`,
# `unstable`, `clean` and `unknown` are not conflicts.
#
# It never fails a job for any other reason. Not a pull request event, no PR
# number, no token, no gh or curl, an API error or a rate limit, a mergeability
# GitHub has not computed after about 40 s: each of these prints a notice and
# lets the job continue.
#
# Environment (the Actions defaults, plus):
#   GH_TOKEN / GITHUB_TOKEN  needs `pull-requests: read` to read the PR and
#                            `actions: write` to cancel the run
#   PR_GUARD_TRIES=9         reads of the PR while GitHub computes mergeability
#   PR_GUARD_STEP=5          seconds between those reads
#   PR_GUARD_CANCEL_WAIT=10  seconds to wait for the cancel to land
#   PR_GUARD_NO_GH=1         use curl even when gh is installed
set -uo pipefail

say() {   # level message: an annotation under Actions, plain text otherwise
  if [[ -n ${GITHUB_ACTIONS:-} ]]; then echo "::$1::$2"; else echo "$1: $2"; fi
}

case "${GITHUB_EVENT_NAME:-}" in
  pull_request|pull_request_target) ;;
  *) echo "pr-guard: not a pull request event (${GITHUB_EVENT_NAME:-none}), nothing to check"; exit 0 ;;
esac

repo="${GITHUB_REPOSITORY:-}"
run_id="${GITHUB_RUN_ID:-}"
api="${GITHUB_API_URL:-https://api.github.com}"
token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"

# The PR number: refs/pull/N/merge on pull_request; the event payload otherwise.
pr="${PR_NUMBER:-}"
if [[ -z $pr && ${GITHUB_REF:-} =~ ^refs/pull/([0-9]+)/ ]]; then pr="${BASH_REMATCH[1]}"; fi
if [[ -z $pr && -r ${GITHUB_EVENT_PATH:-/nonexistent} ]]; then
  pr="$(sed -n 's/^  "number": *\([0-9][0-9]*\).*/\1/p' "$GITHUB_EVENT_PATH" | head -1)"
fi
if [[ -z $pr || -z $repo ]]; then
  say notice "pr-guard: cannot tell which pull request or repository this is; continuing"
  exit 0
fi
if [[ -z $token ]]; then
  say notice "pr-guard: no token; continuing without the conflict check"
  exit 0
fi
export GH_TOKEN="$token"

if command -v gh >/dev/null 2>&1 && [[ -z ${PR_GUARD_NO_GH:-} ]]; then
  tool=gh
elif command -v curl >/dev/null 2>&1; then
  tool=curl
else
  say notice "pr-guard: neither gh nor curl is installed; continuing without the conflict check"
  exit 0
fi

bounded() {   # a command with a time limit, where timeout(1) exists
  if command -v timeout >/dev/null 2>&1; then timeout 30 "$@"; else "$@"; fi
}

get_pr() {    # the pull request as JSON on stdout
  if [[ $tool == gh ]]; then
    bounded gh api "repos/$repo/pulls/$pr"
  else
    bounded curl -fsS --max-time 20 -H "Authorization: Bearer $token" \
      -H "Accept: application/vnd.github+json" "$api/repos/$repo/pulls/$pr"
  fi
}

cancel_run() {
  if [[ $tool == gh ]]; then
    bounded gh run cancel "$run_id" --repo "$repo"
  else
    bounded curl -fsS --max-time 20 -X POST -H "Authorization: Bearer $token" \
      -H "Accept: application/vnd.github+json" "$api/repos/$repo/actions/runs/$run_id/cancel"
  fi
}

tries="${PR_GUARD_TRIES:-9}"
step="${PR_GUARD_STEP:-5}"
state=""
for ((i = 1; i <= tries; i++)); do
  if ! json="$(get_pr 2>&1)"; then
    say notice "pr-guard: could not read pull request #$pr (${json:0:200}); continuing"
    exit 0
  fi
  mergeable="$(printf '%s\n' "$json" | sed -n 's/.*"mergeable": *\(true\|false\|null\).*/\1/p' | head -1)"
  state="$(printf '%s\n' "$json" | sed -n 's/.*"mergeable_state": *"\([a-z_]*\)".*/\1/p' | head -1)"
  [[ -n $mergeable ]] || { say notice "pr-guard: no mergeable field in the answer for #$pr; continuing"; exit 0; }
  [[ $mergeable != null ]] && break
  (( i < tries )) && sleep "$step"
done

if [[ $mergeable == null ]]; then
  say notice "pr-guard: GitHub has not computed whether #$pr merges; continuing"
  exit 0
fi

if [[ $state != dirty ]]; then
  echo "pr-guard: #$pr is ${state:-unknown} against ${GITHUB_BASE_REF:-its base}: no conflict, continuing"
  exit 0
fi

say error "PR #$pr has a merge conflict with ${GITHUB_BASE_REF:-its base branch}: cancelling this run, rebase and push"
if [[ -z $run_id ]]; then
  say warning "pr-guard: GITHUB_RUN_ID is not set, so the run cannot be cancelled"
elif out="$(cancel_run 2>&1)"; then
  echo "pr-guard: asked GitHub to cancel run $run_id"
else
  say warning "pr-guard: could not cancel run $run_id (${out:0:200}); this job stops itself instead"
fi
# The cancel lands within seconds; do not let the next step start meanwhile.
sleep "${PR_GUARD_CANCEL_WAIT:-10}"
exit 1
