#!/usr/bin/env bash
# Poll EC2 for capacity on a GPU shape and characterize the first one that
# launches. The EC2 counterpart of watch-capacity.sh, which does this on Lambda.
#
#   tools/watch-capacity-aws.sh g6e.xlarge,g6e.2xlarge [poll-seconds] [max-hours]
#
# There is no capacity API on EC2. describe-instance-type-offerings says a zone
# *offers* a type, which g6e.xlarge does in all four us-east-1 zones while
# refusing every launch, and RunInstances --dry-run validates permission and
# quota and stops before capacity is consulted. So the only way to ask the
# question is to attempt the launch, and a launch that fails for want of
# capacity is free.
#
# The retry rule is the part worth getting right. This retries on exactly one
# outcome -- "no capacity in any zone" -- and stops on every other, because the
# other failures happen *after* an instance exists. Retrying a run that got as
# far as booting would launch a second billable instance to fail the same way,
# and then a third.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
want="${1:?usage: watch-capacity-aws.sh <type>[,<type>...] [poll-seconds] [max-hours]}"
poll="${2:-300}"
max_h="${3:-12}"
region="${VGPU_AWS_REGION:-us-east-1}"
outdir="${VGPU_CLOUD_OUT:-/tmp/vgpu-cloud}"
mkdir -p "$outdir"
deadline=$(( $(date +%s) + max_h * 3600 ))

IFS=',' read -r -a types <<< "$want"
while [[ $(date +%s) -lt $deadline ]]; do
  for t in "${types[@]}"; do
    log="$outdir/${t}.watch.log"
    "$here/tools/characterize-aws.sh" "$t" "$region" > "$log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
      echo "[watch] $t characterized -- see $log"
      exit 0
    fi
    if grep -q "no capacity in any zone" "$log"; then
      echo "[watch] $t: no capacity ($(date +%H:%M:%S))"
      continue
    fi
    # Anything else means an instance was launched and something later went
    # wrong. Stop: the next attempt would rent another one to fail identically.
    echo "[watch] $t failed for a reason that is not capacity; stopping"
    tail -5 "$log"
    exit 1
  done
  sleep "$poll"
done
echo "[watch] gave up after ${max_h}h with no capacity for $want"
exit 2
