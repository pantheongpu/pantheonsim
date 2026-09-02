#!/usr/bin/env bash
# Verify VirtualGPU's multi-GPU behaviour against a real multi-GPU machine.
#
#   tools/verify-multigpu-cloud.sh [instance-type] [region]
#
# Rents a Lambda instance, builds VirtualGPU on it, and runs the differential
# conformance suite there -- so the NCCL collectives and the multi-device
# semantics are compared against NVIDIA's own libraries on hardware with more
# than one GPU in it, over whatever interconnect that machine has.
#
# Termination is the point of most of this script: an instance left running
# bills by the hour, so the trap is registered before anything is launched and
# the script refuses to start if it cannot read the API key.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
type_name="${1:-gpu_2x_h100_sxm5}"
region="${2:-}"
outdir="${VGPU_CLOUD_OUT:-/tmp/vgpu-multigpu}"
mkdir -p "$outdir"

KEY=$(tr -d ' \n\r' < "${LAMBDA_KEY_FILE:-$HOME/.ssh/lambda_keys}") || { echo "no API key"; exit 1; }
[[ -n "$KEY" ]] || { echo "empty API key"; exit 1; }
api() { curl -s -u "$KEY:" --max-time 60 "$@"; }

INSTANCE_ID=""
cleanup() {
  if [[ -n "$INSTANCE_ID" ]]; then
    echo "[cloud] terminating $INSTANCE_ID"
    api -X POST -H "Content-Type: application/json" \
        -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}" \
        https://cloud.lambdalabs.com/api/v1/instance-operations/terminate >/dev/null
    for _ in $(seq 1 20); do
      st=$(api https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID |
           python3 -c "import json,sys
try: print(json.load(sys.stdin).get('data',{}).get('status','gone'))
except Exception: print('gone')" 2>/dev/null)
      [[ "$st" == "terminated" || "$st" == "gone" || -z "$st" ]] && break
      sleep 3
    done
    echo "[cloud] terminated (status: ${st:-gone})"
  fi
}
trap cleanup EXIT INT TERM

# Pick a region with capacity if the caller did not name one.
if [[ -z "$region" ]]; then
  region=$(api https://cloud.lambdalabs.com/api/v1/instance-types |
    python3 -c "
import json,sys
d=json.load(sys.stdin)['data'].get('$type_name',{})
r=[x['name'] for x in d.get('regions_with_capacity_available',[])]
print(r[0] if r else '')")
fi
[[ -n "$region" ]] || { echo "[cloud] no capacity for $type_name"; exit 1; }

echo "[cloud] launching $type_name in $region"
resp=$(api -X POST -H "Content-Type: application/json" \
  -d "{\"region_name\":\"$region\",\"instance_type_name\":\"$type_name\",\"ssh_key_names\":[\"saqib_WSL\"],\"quantity\":1}" \
  https://cloud.lambdalabs.com/api/v1/instance-operations/launch)
INSTANCE_ID=$(python3 -c "
import json,sys
d=json.loads('''$resp''')
ids=d.get('data',{}).get('instance_ids',[])
print(ids[0] if ids else '')" 2>/dev/null)
[[ -n "$INSTANCE_ID" ]] || { echo "[cloud] launch failed: $(head -c 400 <<<"$resp")"; exit 1; }
echo "[cloud] instance $INSTANCE_ID"

IP=""
for _ in $(seq 1 90); do
  IP=$(api https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID |
       python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('ip') or '')" 2>/dev/null)
  [[ -n "$IP" && "$IP" != "None" ]] && break
  sleep 10
done
[[ -n "$IP" && "$IP" != "None" ]] || { echo "[cloud] no IP"; exit 1; }
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR"
SSH="ssh $SSHOPT ubuntu@$IP"
echo "[cloud] ip $IP; waiting for ssh"
for _ in $(seq 1 60); do $SSH true 2>/dev/null && break; sleep 10; done
$SSH true 2>/dev/null || { echo "[cloud] ssh never came up"; exit 1; }

echo "[cloud] uploading source"
tar -C "$here" --exclude=build --exclude=.git -czf "$outdir/src.tgz" .
scp -q $SSHOPT "$outdir/src.tgz" ubuntu@"$IP":/tmp/ || { echo "[cloud] upload failed"; exit 1; }

echo "[cloud] building and running (this is the part that costs money)"
$SSH 'set -x
  command -v cmake >/dev/null || sudo apt-get -qq update && sudo apt-get -qq install -y cmake
  mkdir -p ~/vgpu && tar -C ~/vgpu -xzf /tmp/src.tgz
  cd ~/vgpu
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader
  nvidia-smi topo -m 2>/dev/null | head -8
  # Compile for the architecture actually present; the differential run needs
  # the physical side to be native, not JIT-ed from a lower target.
  CC_DOT=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  export VGPU_CONF_ARCH="sm_$(echo $CC_DOT | tr -d .)"
  case "$CC_DOT" in
    9.0) export VGPU_CONF_GPU=nvidia/h100 ;;
    8.0) export VGPU_CONF_GPU=nvidia/a100-sxm4-40gb ;;
    8.6) export VGPU_CONF_GPU=nvidia/a10 ;;
    10.0) export VGPU_CONF_GPU=nvidia/b200 ;;
    *)   export VGPU_CONF_GPU=nvidia/h100 ;;
  esac
  echo "arch=$VGPU_CONF_ARCH profile=$VGPU_CONF_GPU"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j"$(nproc)" >/dev/null
  ctest --test-dir build --output-on-failure 2>&1 | tail -5
  echo "===== CONFORMANCE (differential, against this machines GPUs) ====="
  tests/conformance/run_conformance.sh
  echo "===== NCCL ACROSS 8 PROCESSES (virtual rack) ====="
  tests/e2e/run_nccl_multiproc.sh 8
  echo "===== NCCL SINGLE-PROCESS GROUP, 8 VIRTUAL RANKS ====="
  tests/e2e/run_nccl_group.sh 8
' 2>&1 | tee "$outdir/run.log"

echo "[cloud] log -> $outdir/run.log"
