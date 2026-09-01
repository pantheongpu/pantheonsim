#!/usr/bin/env bash
# Poll Lambda for capacity on an instance type and characterize it the moment
# it appears. B200 and H200 capacity comes and goes, so this is the practical
# way to catch it rather than checking by hand.
#
#   tools/watch-capacity.sh gpu_1x_b200_sxm6 [poll-seconds] [max-minutes]
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
want="${1:?usage: watch-capacity.sh <instance-type> [poll-seconds] [max-minutes]}"
poll="${2:-120}"
max_min="${3:-60}"
KEY=$(tr -d ' \n\r' < "${LAMBDA_KEY_FILE:-$HOME/.ssh/lambda_keys}")
deadline=$(( $(date +%s) + max_min * 60 ))

while [[ $(date +%s) -lt $deadline ]]; do
  region=$(curl -s -u "$KEY:" --max-time 30 https://cloud.lambdalabs.com/api/v1/instance-types |
    python3 -c "
import json,sys
d=json.load(sys.stdin).get('data',{})
e=d.get('$want')
regions=[r['name'] for r in (e or {}).get('regions_with_capacity_available',[])]
print(regions[0] if regions else '')" 2>/dev/null)
  if [[ -n "$region" ]]; then
    echo \"[$want] capacity in $region -- characterizing\"
    exec "$here/tools/characterize-cloud.sh" "$want" "$region"
  fi
  echo "[$want] no capacity ($(date +%H:%M:%S)); next check in ${poll}s"
  sleep "$poll"
done
echo "[$want] gave up after ${max_min} minutes with no capacity"
exit 2
