#!/usr/bin/env bash
# Is the burst runner pool alive? Read-only: it prints, and changes nothing.
#
# The SASS, ASan, TSan and ollama jobs (and the toolchain-image finder every one
# of them waits for) run on `vars.BURST_RUNS_ON`, a pool whose runners carry the
# label pantheonsim-burst. A job queued for a label with no online runner waits
# until GitHub cancels it after 24 hours. GitHub's token inside a workflow cannot
# ask for the runner list (the API wants repository admin, which GITHUB_TOKEN has
# no permission for), so this is run by a person with `gh` logged in as one.
#
#   scripts/ci-pool.sh            the variable, the runners that carry the label, and what to do
#   scripts/ci-pool.sh --queued   also count the jobs now waiting for the label (one API call per queued run)
#
# docs/ci.md has the recovery steps this points at.
set -uo pipefail
command -v gh >/dev/null || { echo "gh (the GitHub CLI) is needed" >&2; exit 2; }
repo="${GITHUB_REPOSITORY:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}"
label=pantheonsim-burst
want_queued=0
[[ "${1:-}" == "--queued" ]] && want_queued=1

var=$(gh api "repos/$repo/actions/variables/BURST_RUNS_ON" --jq .value 2>/dev/null || true)
if [[ -n "$var" ]]; then echo "BURST_RUNS_ON: $var"; else echo "BURST_RUNS_ON: not set (the jobs run on GitHub-hosted runners)"; fi

runners=$(gh api "repos/$repo/actions/runners" --paginate \
  --jq ".runners[] | select([.labels[].name] | index(\"$label\")) | [.name, .status, (if .busy then \"busy\" else \"idle\" end)] | @tsv")
online=0
if [[ -z "$runners" ]]; then
  echo "runners with the label $label: none registered"
else
  echo "runners with the label $label:"
  while IFS=$'\t' read -r name status busy; do
    printf '  %-20s %-8s %s\n' "$name" "$status" "$busy"
    [[ "$status" == online ]] && online=$((online + 1))
  done <<<"$runners"
fi

if (( want_queued )); then
  now=$(date +%s); waiting=0; oldest=0
  for run in $(gh run list --status queued --limit 40 --json databaseId --jq '.[].databaseId'); do
    while read -r created; do
      [[ -z "$created" ]] && continue
      waiting=$((waiting + 1))
      age=$(( now - $(date -d "$created" +%s) ))
      (( age > oldest )) && oldest=$age
    done < <(gh api "repos/$repo/actions/runs/$run/jobs?per_page=100" \
      --jq ".jobs[] | select(.status==\"queued\" and (.labels | index(\"$label\"))) | .created_at")
  done
  echo "jobs waiting for the label: $waiting, the oldest for $(( oldest / 60 )) minutes"
fi

if [[ -n "$var" && "$online" -eq 0 ]]; then
  cat <<EOF

No runner with the label is online, and BURST_RUNS_ON sends jobs to it. They wait
until GitHub cancels them after 24 hours. To run them on GitHub's hosted runners
instead, see "The burst pool is offline" in docs/ci.md (gh variable delete BURST_RUNS_ON,
then cancel and re-run the runs that are queued).
EOF
  exit 1
fi
