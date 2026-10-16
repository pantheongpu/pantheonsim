#!/usr/bin/env bash
# For every open pull request that has a merge conflict (mergeable_state
# `dirty`), cancel its queued, waiting and running pull_request workflow runs
# for the head commit and label it `merge-conflict`; take the label off a pull
# request that no longer conflicts. Run by .github/workflows/cancel-conflicting-prs.yml.
#
# Needs gh, GH_TOKEN and GH_REPO (owner/name), `actions: write` and
# `pull-requests: write`. DRY_RUN=true prints what it would do and changes
# nothing. Only `dirty` is a conflict: `behind`, `blocked`, `unstable`, `clean`
# and `unknown` are left alone, and a PR whose mergeability GitHub has not
# computed after the retries is skipped (the next run looks again).
#
# Tuning, for tests: CC_STEP (seconds between rounds of retries, 8), CC_ROUNDS
# (retry rounds, 7), CC_PAUSE (seconds between two PRs' first reads, 0.3).
set -uo pipefail

label=merge-conflict
dry="${DRY_RUN:-false}"
step="${CC_STEP:-8}"
rounds="${CC_ROUNDS:-7}"
pause="${CC_PAUSE:-0.3}"
repo="${GH_REPO:-${GITHUB_REPOSITORY:-}}"
[[ -n $repo ]] || { echo "GH_REPO is not set"; exit 1; }
[[ $dry == true ]] && echo "dry run: nothing will be cancelled or labelled"

prs="$(gh pr list --repo "$repo" --state open --limit 100 \
        --json number,headRefName,headRefOid,isDraft,labels \
        --jq '.[] | [.number, .headRefOid, (.labels | map(.name) | index("'"$label"'") != null), .headRefName] | @tsv')" || {
  echo "::warning::could not list the open pull requests; nothing done"
  exit 0
}
[[ -n $prs ]] || { echo "no open pull requests"; exit 0; }

declare -A sha has_label branch state
pending=()
while IFS=$'\t' read -r n s l b; do
  sha[$n]=$s; has_label[$n]=$l; branch[$n]=$b; state[$n]=""
  pending+=("$n")
done <<< "$prs"
total=${#pending[@]}

# read_state N: mergeable_state into state[N] when GitHub has computed it, else
# leaves it empty. Returns 1 when the answer is not final yet.
read_state() {
  local n=$1 json mergeable
  if ! json="$(gh api "repos/$repo/pulls/$n" 2>&1)"; then
    state[$n]=error
    return 0
  fi
  mergeable="$(printf '%s\n' "$json" | sed -n 's/.*"mergeable": *\(true\|false\|null\).*/\1/p' | head -1)"
  if [[ -z $mergeable ]]; then state[$n]=error; return 0; fi
  [[ $mergeable == null ]] && return 1
  state[$n]="$(printf '%s\n' "$json" | sed -n 's/.*"mergeable_state": *"\([a-z_]*\)".*/\1/p' | head -1)"
  return 0
}

# Round 0 reads every PR once (asking is what makes GitHub compute a stale
# answer), spaced out; the rounds after it ask again only for those still null.
for ((r = 0; r <= rounds && ${#pending[@]} > 0; r++)); do
  (( r > 0 )) && sleep "$step"
  next=()
  for n in "${pending[@]}"; do
    read_state "$n" || next+=("$n")
    (( r == 0 )) && sleep "$pause"
  done
  pending=("${next[@]}")
done

label_ready=0
ensure_label() {
  (( label_ready )) && return
  gh label create "$label" --repo "$repo" --color d93f0b \
    --description "Needs a rebase; CI is cancelled while it conflicts" >/dev/null 2>&1 || true
  label_ready=1
}

conflicts=0 cancelled=0 labelled=0 unlabelled=0 skipped=0
for n in $(printf '%s\n' "${!sha[@]}" | sort -n); do
  st="${state[$n]}"
  if [[ -z $st ]]; then
    echo "#$n (${branch[$n]}): mergeability not computed yet, skipped"; skipped=$((skipped + 1)); continue
  fi
  if [[ $st == error ]]; then
    echo "#$n (${branch[$n]}): could not read it, skipped"; skipped=$((skipped + 1)); continue
  fi
  if [[ $st != dirty ]]; then
    if [[ ${has_label[$n]} == true ]]; then
      if [[ $dry == true ]]; then
        echo "#$n (${branch[$n]}): $st, would remove the $label label"
      elif gh pr edit "$n" --repo "$repo" --remove-label "$label" >/dev/null 2>&1; then
        echo "#$n (${branch[$n]}): $st, removed the $label label"
      else
        echo "#$n (${branch[$n]}): $st, could not remove the $label label"
      fi
      unlabelled=$((unlabelled + 1))
    else
      echo "#$n (${branch[$n]}): $st"
    fi
    continue
  fi

  conflicts=$((conflicts + 1))
  ids=""
  for status in in_progress queued waiting requested pending; do
    ids+="$(gh run list --repo "$repo" --commit "${sha[$n]}" --status "$status" --limit 100 \
              --json databaseId,event --jq '.[] | select(.event == "pull_request") | .databaseId' 2>/dev/null)"$'\n'
  done
  ids="$(printf '%s\n' "$ids" | sed '/^$/d' | sort -un)"
  count=0
  for id in $ids; do
    count=$((count + 1))
    if [[ $dry == true ]]; then continue; fi
    gh run cancel "$id" --repo "$repo" >/dev/null 2>&1 || echo "  could not cancel run $id"
  done
  cancelled=$((cancelled + count))
  verb=cancelled; [[ $dry == true ]] && verb="would cancel"
  if [[ ${has_label[$n]} == true ]]; then
    lab="already labelled"
  elif [[ $dry == true ]]; then
    lab="would add the $label label"; labelled=$((labelled + 1))
  else
    ensure_label
    if gh pr edit "$n" --repo "$repo" --add-label "$label" >/dev/null 2>&1; then lab="labelled"; else lab="could not label"; fi
    labelled=$((labelled + 1))
  fi
  echo "#$n (${branch[$n]}): CONFLICT, $verb $count run(s), $lab"
done

echo "checked $total open PR(s): $conflicts conflicting, $cancelled run(s) cancelled, $labelled labelled, $unlabelled unlabelled, $skipped skipped"
