#!/usr/bin/env bash
# Run one CUDA program on a rented GPU and print what it printed.
#
#   nvidia/tools/run-on-hardware.sh <file.cu> [instance-type] [region] [nvcc-args...]
#
# For the question "is this the simulator's answer or my expectation that is
# wrong?", which reading the ISA does not settle. Termination is registered on
# EXIT before the instance is launched, as in characterize-cloud.sh.
set -uo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
src="${1:?usage: run-on-hardware.sh <file.cu> [instance-type] [region] [nvcc-args...]}"
type_name="${2:-gpu_1x_a10}"
region="${3:-us-east-1}"
shift 3 2>/dev/null || shift $#
nvcc_args=("$@")

KEY=$(tr -d ' \n\r' < "${LAMBDA_KEY_FILE:-$HOME/.ssh/lambda_keys}") || { echo "no API key"; exit 1; }
api() { curl -s -u "$KEY:" --max-time 60 "$@"; }

INSTANCE_ID=""
cleanup() {
  [[ -z "$INSTANCE_ID" ]] && return
  echo "[hw] terminating $INSTANCE_ID" >&2
  api -X POST -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}" \
      https://cloud.lambdalabs.com/api/v1/instance-operations/terminate >/dev/null
  for _ in $(seq 1 10); do
    st=$(api https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID |
         python3 -c "import json,sys
try: print(json.load(sys.stdin).get('data',{}).get('status','gone'))
except Exception: print('gone')" 2>/dev/null)
    [[ "$st" == "terminated" || "$st" == "gone" || -z "$st" ]] && break
    sleep 3
  done
  echo "[hw] terminated (${st:-gone})" >&2
}
trap cleanup EXIT INT TERM

echo "[hw] launching $type_name in $region" >&2
resp=$(api -X POST -H "Content-Type: application/json" \
  -d "{\"region_name\":\"$region\",\"instance_type_name\":\"$type_name\",\"ssh_key_names\":[\"saqib_WSL\"],\"quantity\":1}" \
  https://cloud.lambdalabs.com/api/v1/instance-operations/launch)
INSTANCE_ID=$(python3 -c "
import json,sys
d=json.loads('''$resp''')
ids=d.get('data',{}).get('instance_ids',[])
print(ids[0] if ids else '')" 2>/dev/null)
[[ -z "$INSTANCE_ID" ]] && { echo "[hw] launch failed: $(head -c 300 <<<"$resp")" >&2; exit 1; }

IP=""
for _ in $(seq 1 60); do
  IP=$(api https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID |
       python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('ip') or '')" 2>/dev/null)
  [[ -n "$IP" && "$IP" != "None" ]] && break
  sleep 10
done
[[ -z "$IP" || "$IP" == "None" ]] && { echo "[hw] no IP" >&2; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR ubuntu@$IP"
for _ in $(seq 1 40); do $SSH true 2>/dev/null && break; sleep 10; done
$SSH true 2>/dev/null || { echo "[hw] ssh never came up" >&2; exit 1; }

scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    "$src" ubuntu@"$IP":/tmp/prog.cu 2>/dev/null
$SSH "set -e
  cd /tmp
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
  ARCH=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  nvcc -std=c++14 -arch=sm_\$ARCH -Wno-deprecated-gpu-targets ${nvcc_args[*]:-} prog.cu -o prog
  ./prog"
