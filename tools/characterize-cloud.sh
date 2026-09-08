#!/usr/bin/env bash
# Characterize a cloud GPU: launch an instance, read the device, run the
# differential conformance suite on it, write a profile, and terminate.
#
#   tools/characterize-cloud.sh <lambda-instance-type> <region> [profile-id]
#
# Termination is the important part -- an instance left running bills by the
# hour -- so it is registered on EXIT before the instance is ever launched, and
# the script also refuses to start if it cannot read the API key.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
type_name="${1:?usage: characterize-cloud.sh <instance-type> <region> [profile-id]}"
region="${2:?missing region}"
outdir="${VGPU_CLOUD_OUT:-/tmp/vgpu-cloud}"
mkdir -p "$outdir"

KEY=$(tr -d ' \n\r' < "${LAMBDA_KEY_FILE:-$HOME/.ssh/lambda_keys}") || { echo "no API key"; exit 1; }
api() { curl -s -u "$KEY:" --max-time 60 "$@"; }

INSTANCE_ID=""
cleanup() {
  if [[ -n "$INSTANCE_ID" ]]; then
    echo "[$type_name] terminating $INSTANCE_ID"
    api -X POST -H "Content-Type: application/json" \
        -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}" \
        https://cloud.lambdalabs.com/api/v1/instance-operations/terminate >/dev/null
    # Confirm it is really going away; a silent failure here costs money.
    for _ in $(seq 1 10); do
      st=$(api https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID |
           python3 -c "import json,sys
try: print(json.load(sys.stdin).get('data',{}).get('status','gone'))
except Exception: print('gone')" 2>/dev/null)
      [[ "$st" == "terminated" || "$st" == "gone" || -z "$st" ]] && break
      sleep 3
    done
    echo "[$type_name] terminated (status: ${st:-gone})"
  fi
}
trap cleanup EXIT INT TERM

echo "[$type_name] launching in $region"
resp=$(api -X POST -H "Content-Type: application/json" \
  -d "{\"region_name\":\"$region\",\"instance_type_name\":\"$type_name\",\"ssh_key_names\":[\"saqib_WSL\"],\"quantity\":1}" \
  https://cloud.lambdalabs.com/api/v1/instance-operations/launch)
INSTANCE_ID=$(python3 -c "
import json,sys
d=json.loads('''$resp''')
ids=d.get('data',{}).get('instance_ids',[])
print(ids[0] if ids else '')" 2>/dev/null)
if [[ -z "$INSTANCE_ID" ]]; then
  echo "[$type_name] launch failed: $(head -c 300 <<<"$resp")"; exit 1
fi
echo "[$type_name] instance $INSTANCE_ID"

# Wait for an IP and for sshd to answer.
IP=""
for _ in $(seq 1 60); do
  IP=$(api https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID |
       python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('ip') or '')" 2>/dev/null)
  [[ -n "$IP" && "$IP" != "None" ]] && break
  sleep 10
done
[[ -z "$IP" || "$IP" == "None" ]] && { echo "[$type_name] no IP"; exit 1; }
echo "[$type_name] ip $IP; waiting for ssh"
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR ubuntu@$IP"
for _ in $(seq 1 40); do $SSH true 2>/dev/null && break; sleep 10; done
$SSH true 2>/dev/null || { echo "[$type_name] ssh never came up"; exit 1; }

echo "[$type_name] uploading and running"
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    "$here/tools/characterize.cu" "$here/tools/characterize-telemetry.sh" \
    "$here/tests/conformance/ptx_semantics.cu" "$here/tests/conformance/control_flow.cu" \
    ubuntu@"$IP":/tmp/ 2>/dev/null

$SSH 'set -e
  cd /tmp
  nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
  nvcc -std=c++14 -Wno-deprecated-gpu-targets characterize.cu -o characterize -lcuda 2>/dev/null
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ".")
  { ./characterize 0 2>/dev/null | sed "/^telemetry:/,\$d"; bash characterize-telemetry.sh 0; } > profile.yaml
  for t in ptx_semantics control_flow; do
    nvcc -std=c++14 -arch=sm_$ARCH -Wno-deprecated-gpu-targets $t.cu -o $t 2>/dev/null && ./$t > $t.ref.txt
  done
  # The metrics this device exposes -- see the note in characterize-aws.sh.
  (ncu --query-metrics 2>/dev/null || true) > metrics.txt
  echo "METRICS=$(wc -l < metrics.txt)"
  echo "SM_ARCH=$ARCH"
' > "$outdir/${type_name}.run.log" 2>&1
cat "$outdir/${type_name}.run.log" | head -3

scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    ubuntu@"$IP":/tmp/profile.yaml "$outdir/${type_name}.yaml" 2>/dev/null
for t in ptx_semantics control_flow; do
  scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
      ubuntu@"$IP":/tmp/$t.ref.txt "$outdir/${type_name}.$t.ref.txt" 2>/dev/null
done
echo "[$type_name] collected -> $outdir/${type_name}.yaml"
