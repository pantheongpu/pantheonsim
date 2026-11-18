#!/usr/bin/env bash
# Cancels the pull_request runs nobody is waiting for: a run whose head commit is not the head of an open
# pull request (the PR merged or closed, or a newer commit was pushed). Run by
# .github/workflows/cancel-conflicting-prs.yml after every push to main and every 20 minutes.
#
# Without it the sanitizer jobs of a merged PR keep a shared runner for hours while the open PRs queue.
# Runs of push, schedule and dispatch events are left alone. Needs gh, GH_TOKEN, GH_REPO, `actions: write`.
# DRY_RUN=true only prints.
set -uo pipefail
repo="${GH_REPO:-${GITHUB_REPOSITORY:-}}"
[[ -n $repo ]] || { echo "GH_REPO is not set"; exit 1; }
dry="${DRY_RUN:-false}"

# The runs first, the open PRs after: a PR opened in between is then in the list that decides.
runs=""
for status in in_progress queued waiting requested pending; do
  more="$(gh run list --repo "$repo" --event pull_request --status "$status" --limit 200 \
            --json databaseId,headBranch,headSha --jq '.[] | "\(.databaseId) \(.headBranch) \(.headSha)"')" || {
    echo "::warning::could not list the $status runs; nothing done"; exit 0; }
  runs+="$more"$'\n'
done
prs="$(gh pr list --repo "$repo" --state open --limit 200 --json headRefName,headRefOid \
         --jq '.[] | "\(.headRefName) \(.headRefOid)"')" || { echo "::warning::could not list the open pull requests; nothing done"; exit 0; }
(( $(printf '%s\n' "$prs" | sed '/^$/d' | wc -l) < 200 )) || { echo "::warning::200 open pull requests: the list may be cut short; nothing done"; exit 0; }

declare -A open
while read -r branch sha; do [[ -n $branch ]] && open["$branch $sha"]=1; done <<< "$prs"

cancelled=0 kept=0
while read -r id branch sha; do
  [[ -n $id ]] || continue
  if [[ -n ${open["$branch $sha"]:-} ]]; then kept=$((kept + 1)); continue; fi
  cancelled=$((cancelled + 1))
  if [[ $dry == true ]]; then echo "would cancel run $id ($branch ${sha:0:8})"; continue; fi
  if gh run cancel "$id" --repo "$repo" >/dev/null 2>&1; then echo "cancelled run $id ($branch ${sha:0:8})"; else echo "could not cancel run $id"; fi
done <<< "$runs"
echo "$cancelled stale pull_request run(s) $([[ $dry == true ]] && echo would be) cancelled, $kept kept"
