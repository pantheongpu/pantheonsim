#!/usr/bin/env bash
# Poll Lambda for capacity and act on the first type in the list that has any.
# Capacity for the scarcer shapes comes and goes, so this is the practical way
# to catch it rather than checking by hand. The list is tried in order, so put
# the one you would rather pay for first.
#
#   nvidia/tools/watch-capacity.sh gpu_1x_b200_sxm6 [poll-seconds] [max-minutes]
#   nvidia/tools/watch-capacity.sh gpu_4x_a6000,gpu_4x_a100,gpu_4x_h100_sxm5 120 180
#
#   VGPU_CAPACITY_ACTION=nvidia/tools/verify-multigpu-cloud.sh   # default: characterize
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
want="${1:?usage: watch-capacity.sh <type>[,<type>...] [poll-seconds] [max-minutes]}"
poll="${2:-120}"
max_min="${3:-60}"
action="${VGPU_CAPACITY_ACTION:-tools/characterize-cloud.sh}"
KEY=$(tr -d ' \n\r' < "${LAMBDA_KEY_FILE:-$HOME/.ssh/lambda_keys}")
deadline=$(( $(date +%s) + max_min * 60 ))

while [[ $(date +%s) -lt $deadline ]]; do
  read -r found region < <(curl -s -u "$KEY:" --max-time 30 \
      https://cloud.lambdalabs.com/api/v1/instance-types |
    python3 -c "
import json,sys
d = json.load(sys.stdin).get('data', {})
for name in '''$want'''.split(','):
    entry = d.get(name.strip())
    regions = [r['name'] for r in (entry or {}).get('regions_with_capacity_available', [])]
    if regions:
        print(name.strip(), regions[0])
        break
else:
    print('', '')" 2>/dev/null)
  if [[ -n "${found:-}" ]]; then
    echo "[watch] $found has capacity in $region -- running $action"
    exec "$here/$action" "$found" "$region"
  fi
  echo "[watch] no capacity for $want ($(date +%H:%M:%S)); next check in ${poll}s"
  sleep "$poll"
done
echo "[watch] gave up after ${max_min} minutes with no capacity for $want"
exit 2
