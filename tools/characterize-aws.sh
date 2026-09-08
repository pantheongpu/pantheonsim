#!/usr/bin/env bash
# Characterize an EC2 GPU: launch an instance, read the device, run the
# differential conformance suite on it, write a profile, and terminate.
#
#   tools/characterize-aws.sh <instance-type> [region]
#
# The EC2 counterpart of characterize-cloud.sh, which does the same on Lambda.
# EC2 reaches parts Lambda does not offer -- Turing and Ada Lovelace among them
# -- and those are whole architectures rather than another card.
#
# Termination is the important part, so it is registered on EXIT before the
# instance is ever launched, and confirmed afterwards: an instance left running
# bills by the hour.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
type_name="${1:?usage: characterize-aws.sh <instance-type> [region]}"
region="${2:-us-east-1}"
outdir="${VGPU_CLOUD_OUT:-/tmp/vgpu-cloud}"
key_name="${VGPU_AWS_KEY_NAME:-pantheon-bench-wsl}"
key_file="${VGPU_AWS_KEY_FILE:-$HOME/.ssh/id_rsa}"
sg_name="${VGPU_AWS_SG:-pantheon-bench-sg}"
mkdir -p "$outdir"

# Check the inputs before renting anything. These are resolved relative to the
# script, so a copy run from outside the tree resolves them to nonsense -- and
# the first version of this discovered that only after launching an instance,
# uploading nothing, and failing on the far side.
for f in tools/characterize.cu tools/characterize-telemetry.sh \
         tests/conformance/ptx_semantics.cu tests/conformance/control_flow.cu; do
  [[ -r "$here/$f" ]] || { echo "missing $here/$f (run this from the repository, not a copy)"; exit 1; }
done

command -v aws >/dev/null || { echo "no aws CLI"; exit 1; }
aws sts get-caller-identity >/dev/null 2>&1 || { echo "not authenticated: run 'aws login'"; exit 1; }
[[ -r "$key_file" ]] || { echo "no ssh key at $key_file"; exit 1; }

INSTANCE_ID=""
SG_TEMP=""
KEY_TEMP=""
cleanup() {
  if [[ -n "$INSTANCE_ID" ]]; then
    echo "[$type_name] terminating $INSTANCE_ID"
  aws ec2 terminate-instances --region "$region" --instance-ids "$INSTANCE_ID" >/dev/null 2>&1
  for _ in $(seq 1 20); do
    st=$(aws ec2 describe-instances --region "$region" --instance-ids "$INSTANCE_ID" \
         --query "Reservations[0].Instances[0].State.Name" --output text 2>/dev/null)
    [[ "$st" == "terminated" || "$st" == "shutting-down" || -z "$st" || "$st" == "None" ]] && break
    sleep 5
  done
    echo "[$type_name] terminated (status: ${st:-gone})"
  fi
  # The group cannot go until the instance's interface has released it, which
  # is why this waits rather than trying once.
  if [[ -n "$SG_TEMP" ]]; then
    for _ in $(seq 1 20); do
      aws ec2 delete-security-group --region "$region" --group-id "$SG_TEMP" >/dev/null 2>&1 && break
      sleep 10
    done
    echo "[$type_name] removed temporary security group $SG_TEMP"
  fi
  if [[ -n "$KEY_TEMP" ]]; then
    aws ec2 delete-key-pair --region "$region" --key-name "$KEY_TEMP" >/dev/null 2>&1
    echo "[$type_name] removed temporary key pair $KEY_TEMP"
  fi
}
trap cleanup EXIT INT TERM

# The Deep Learning base AMI carries the driver and toolkit already, and its
# architecture has to match the instance: the Graviton GPU parts are arm64 and
# will not boot the x86 image.
arch=$(aws ec2 describe-instance-types --region "$region" --instance-types "$type_name" \
       --query "InstanceTypes[0].ProcessorInfo.SupportedArchitectures[0]" --output text 2>/dev/null)
[[ "$arch" == "arm64" ]] && ami_name="Deep Learning ARM64 Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" \
                         || ami_name="Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*"
ami=$(aws ec2 describe-images --region "$region" --owners amazon \
      --filters "Name=name,Values=$ami_name" "Name=architecture,Values=${arch:-x86_64}" \
                "Name=state,Values=available" \
      --query "reverse(sort_by(Images,&CreationDate))[:1].ImageId" --output text 2>/dev/null)
[[ -z "$ami" || "$ami" == "None" ]] && { echo "[$type_name] no Deep Learning AMI for $arch"; exit 1; }
# Security groups are per region, so a name that exists in one is absent in the
# next. Make a temporary one where it is missing and take it away again, rather
# than requiring the caller to have prepared every region by hand.
sg=$(aws ec2 describe-security-groups --region "$region" \
     --filters "Name=group-name,Values=$sg_name" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null)
if [[ -z "$sg" || "$sg" == "None" ]]; then
  vpc=$(aws ec2 describe-vpcs --region "$region" --filters Name=isDefault,Values=true \
        --query "Vpcs[0].VpcId" --output text 2>/dev/null)
  [[ -z "$vpc" || "$vpc" == "None" ]] && { echo "[$type_name] no default VPC in $region"; exit 1; }
  sg=$(aws ec2 create-security-group --region "$region" --vpc-id "$vpc" \
       --group-name "vgpu-characterize-$$" --description "temporary; ssh for one characterization run" \
       --query "GroupId" --output text 2>/dev/null)
  [[ -z "$sg" || "$sg" == "None" ]] && { echo "[$type_name] could not create a security group"; exit 1; }
  SG_TEMP="$sg"
  myip=$(curl -s --max-time 20 https://checkip.amazonaws.com | tr -d ' \n\r')
  aws ec2 authorize-security-group-ingress --region "$region" --group-id "$sg" \
      --protocol tcp --port 22 --cidr "${myip:-0.0.0.0}/32" >/dev/null 2>&1
  echo "[$type_name] created temporary security group $sg (ssh from ${myip:-unknown})"
fi

# Capacity is per availability zone, so a GPU type can be dry in the zone AWS
# picks and free in the next one along. Letting it choose and giving up on the
# first InsufficientInstanceCapacity loses the run for no reason.
launch_in() {
  local az="$1" placement=()
  [[ -n "$az" ]] && placement=(--placement "AvailabilityZone=$az")
  aws ec2 run-instances --region "$region" --image-id "$ami" \
    --instance-type "$type_name" --key-name "$key_name" --security-group-ids "$sg" \
    "${placement[@]}" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3,DeleteOnTermination=true}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=vgpu-characterize-$type_name}]" \
    --query "Instances[0].InstanceId" --output text 2>&1
}

mapfile -t zones < <(aws ec2 describe-instance-type-offerings --region "$region" \
  --location-type availability-zone --filters "Name=instance-type,Values=$type_name" \
  --query "InstanceTypeOfferings[].Location" --output text 2>/dev/null | tr '\t' '\n')
# Key pairs are per region too. Import the local public key where the named
# pair is absent, and take it away again with the security group.
if ! aws ec2 describe-key-pairs --region "$region" --key-names "$key_name" >/dev/null 2>&1; then
  [[ -r "${key_file}.pub" ]] || { echo "[$type_name] no key pair '$key_name' in $region and no ${key_file}.pub to import"; exit 1; }
  key_name="vgpu-characterize-$$"
  aws ec2 import-key-pair --region "$region" --key-name "$key_name" \
      --public-key-material "fileb://${key_file}.pub" >/dev/null 2>&1 \
    || { echo "[$type_name] could not import a key pair"; exit 1; }
  KEY_TEMP="$key_name"
  echo "[$type_name] imported temporary key pair $key_name"
fi

echo "[$type_name] launching in $region from $ami (${#zones[@]} zones offer it)"
for az in "" "${zones[@]}"; do
  out=$(launch_in "$az")
  if [[ -n "$out" && "$out" != None* && "$out" != *Error* && "$out" != *error* ]]; then
    INSTANCE_ID="$out"; break
  fi
  if [[ "$out" == *InsufficientInstanceCapacity* ]]; then
    echo "[$type_name] no capacity in ${az:-the zone AWS picked}"
    continue
  fi
  echo "[$type_name] launch failed: $(head -c 200 <<<"$out")"; INSTANCE_ID=""; exit 1
done
[[ -z "$INSTANCE_ID" ]] && { echo "[$type_name] no capacity in any zone"; exit 1; }
echo "[$type_name] instance $INSTANCE_ID"

IP=""
for _ in $(seq 1 60); do
  IP=$(aws ec2 describe-instances --region "$region" --instance-ids "$INSTANCE_ID" \
       --query "Reservations[0].Instances[0].PublicIpAddress" --output text 2>/dev/null)
  [[ -n "$IP" && "$IP" != "None" ]] && break
  sleep 10
done
[[ -z "$IP" || "$IP" == "None" ]] && { echo "[$type_name] no public IP"; exit 1; }
echo "[$type_name] ip $IP; waiting for ssh"
SSH="ssh -i $key_file -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR ubuntu@$IP"
for _ in $(seq 1 60); do $SSH true 2>/dev/null && break; sleep 10; done
$SSH true 2>/dev/null || { echo "[$type_name] ssh never came up"; exit 1; }

echo "[$type_name] uploading and running"
scp -i "$key_file" -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    "$here/tools/characterize.cu" "$here/tools/characterize-telemetry.sh" \
    "$here/tests/conformance/ptx_semantics.cu" "$here/tests/conformance/control_flow.cu" \
    ubuntu@"$IP":/tmp/ 2>/dev/null

$SSH 'set -e
  cd /tmp
  export PATH=$PATH:/usr/local/cuda/bin
  nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
  # Errors here are not hidden: a compile that fails silently leaves an empty
  # profile and the run reports success anyway, which is worse than a red run.
  nvcc -std=c++14 -Wno-deprecated-gpu-targets characterize.cu -o characterize \
       -L/usr/local/cuda/lib64/stubs -lcuda
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ".")
  { ./characterize 0 2>/dev/null | sed "/^telemetry:/,\$d"; bash characterize-telemetry.sh 0; } > profile.yaml
  for t in ptx_semantics control_flow; do
    nvcc -std=c++14 -arch=sm_$ARCH -Wno-deprecated-gpu-targets $t.cu -o $t 2>/dev/null && ./$t > $t.ref.txt
  done
  # The metrics this device actually exposes. Not to implement them -- most are
  # timing-derived and this engine has no timing model -- but so the gap is
  # per-device data rather than a general claim. "An L4 exposes N metrics and
  # VirtualGPU produces these M" is checkable; "we do not model timing" is a
  # footnote someone has to take on trust.
  (ncu --query-metrics 2>/dev/null || nv-nsight-cu-cli --query-metrics 2>/dev/null || true) \
    > metrics.txt
  echo "METRICS=$(wc -l < metrics.txt)"
  echo "SM_ARCH=$ARCH"
' > "$outdir/${type_name}.run.log" 2>&1
head -3 "$outdir/${type_name}.run.log"

for f in profile.yaml ptx_semantics.ref.txt control_flow.ref.txt metrics.txt; do
  scp -i "$key_file" -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
      ubuntu@"$IP":/tmp/$f "$outdir/${type_name}.${f/profile.yaml/yaml}" 2>/dev/null
done

# Say collected only when something was. Claiming it either way is how a run
# that produced nothing reads as a run that produced a profile.
if [[ -s "$outdir/${type_name}.yaml" ]] && grep -q '^id:' "$outdir/${type_name}.yaml"; then
  echo "[$type_name] collected -> $outdir/${type_name}.yaml"
else
  echo "[$type_name] FAILED: no profile came back; the remote log follows"
  sed 's/^/    /' "$outdir/${type_name}.run.log"
  exit 1
fi
