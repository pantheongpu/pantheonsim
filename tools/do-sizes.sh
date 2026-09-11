#!/usr/bin/env bash
# What GPU sizes this account can actually create, and in which regions.
#
#   tools/do-sizes.sh            # every GPU size the API will admit to
#   tools/do-sizes.sh mi300      # filtered
#
# This exists because its absence is expensive. characterize-do.sh could only
# ask "create this", and a refusal came back as "Size is not available in this
# region" -- which does not say whether the size is unknown, sold out, or
# simply somewhere else. Three creates in three regions were spent learning
# what one read-only call answers directly.
#
# Read-only: GET /v2/sizes. It creates nothing and costs nothing.
set -uo pipefail
filter="${1:-gpu}"
token_file="${VGPU_DO_TOKEN_FILE:-$HOME/.ssh/amd_developer_api}"
[[ -r "$token_file" ]] || { echo "no DigitalOcean token at $token_file" >&2; exit 1; }
TOK=$(tr -d ' \n\r' < "$token_file")

resp=$(curl -sS --max-time 30 -H "Authorization: Bearer $TOK" \
  "https://api.digitalocean.com/v2/sizes?per_page=250") || { echo "request failed" >&2; exit 1; }

if ! jq -e '.sizes' >/dev/null 2>&1 <<<"$resp"; then
  # Print the API's own complaint rather than a generic one; it distinguishes
  # an expired token from a scope problem.
  jq -r '.message // .id // "unrecognised response"' <<<"$resp" >&2
  exit 1
fi

printf '%-26s %-9s %-10s %s\n' SLUG AVAILABLE '$/HOUR' REGIONS
jq -r --arg f "$filter" '
  .sizes[]
  | select(.slug | test($f))
  | [ .slug,
      (if .available then "yes" else "no" end),
      (.price_hourly|tostring),
      ((.regions // []) | if length == 0 then "(none)" else join(",") end)
    ] | @tsv' <<<"$resp" \
| sort | awk -F'\t' '{printf "%-26s %-9s %-10s %s\n", $1, $2, $3, $4}'
