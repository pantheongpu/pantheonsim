#!/usr/bin/env bash
# An ephemeral GPU runner for the card-host-gpu workflow (.github/workflows/card-host-gpu.yml): one g5.xlarge (NVIDIA A10G, sm_86), one job, then it terminates.
#
#   tools/card-ci/card-runner.sh start        launch the instance and wait for its runner to come online; prints  instance=<id>
#   tools/card-ci/card-runner.sh stop <id>    terminate it (the workflow's last job does this whatever happened) and remove its runner
#   tools/card-ci/card-runner.sh status       instances and runners that carry the project tag
#
# Needs the aws CLI (credentials allowed ec2:RunInstances/TerminateInstances/DescribeInstances/DescribeImages/DescribeSubnets/DescribeSecurityGroups and
# CreateTags on this project's resources) and a gh token with administration on the repository (a registration token is made for the instance). AWS_REGION
# defaults to us-east-1; CARD_INSTANCE_TYPE to g5.xlarge; CARD_TTL_MINUTES to 120; CARD_AMI to a Deep Learning Base GPU AMI (Ubuntu 24.04), found by name
# when unset. Every instance gets the tag Project=pantheonsim-ci-card, terminates on shutdown, has a 120 GB gp3 root volume that goes with it, and no
# inbound access. Cost: about $1.01 an hour on demand; a run is 35 to 50 minutes.
set -euo pipefail
cd "$(dirname "$0")"
REPO="${CARD_REPO:-pantheongpu/pantheonsim}"
export AWS_DEFAULT_REGION="${AWS_REGION:-us-east-1}"
TYPE="${CARD_INSTANCE_TYPE:-g5.xlarge}"
TTL="${CARD_TTL_MINUTES:-120}"
TAG=pantheonsim-ci-card

instances() {
  aws ec2 describe-instances --filters "Name=tag:Project,Values=$TAG" "Name=instance-state-name,Values=${1:-pending,running}" \
    --query 'Reservations[].Instances[].InstanceId' --output text
}
runners() { gh api "repos/$REPO/actions/runners" --paginate --jq ".runners[] | select(.name | startswith(\"card-\")) | \"\(.id) \(.name) \(.status)\""; }

case "${1:-}" in
  start)
    [[ -z "$(instances)" ]] || { echo "an instance is already running: $(instances)" >&2; exit 1; }   # one card run at a time
    ami="${CARD_AMI:-$(aws ec2 describe-images --owners amazon --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*' 'Name=state,Values=available' \
      --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)}"
    [[ -n "$ami" && "$ami" != None ]] || { echo "no GPU AMI found; set CARD_AMI" >&2; exit 1; }
    subnet=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true --query 'Subnets[0].SubnetId' --output text)
    vpc=$(aws ec2 describe-subnets --subnet-ids "$subnet" --query 'Subnets[0].VpcId' --output text)
    sg=$(aws ec2 describe-security-groups --filters Name=group-name,Values=pantheonsim-card-runners Name=vpc-id,Values="$vpc" --query 'SecurityGroups[0].GroupId' --output text)
    if [[ "$sg" == None ]]; then   # no inbound rule: the runner only calls out
      sg=$(aws ec2 create-security-group --group-name pantheonsim-card-runners --description "pantheonsim card CI: outbound only" --vpc-id "$vpc" --query GroupId --output text)
    fi
    ud=$(mktemp); trap 'rm -f "$ud"' EXIT; chmod 600 "$ud"
    tok=$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token)
    sed -e "s/__TOKEN__/$tok/" -e "s/__TTL__/$TTL/" userdata.tmpl > "$ud"; unset tok
    id=$(aws ec2 run-instances --image-id "$ami" --instance-type "$TYPE" --count 1 --subnet-id "$subnet" --security-group-ids "$sg" \
      --metadata-options HttpTokens=required,HttpEndpoint=enabled --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3,DeleteOnTermination=true}' --user-data "file://$ud" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=pantheonsim-card-runner},{Key=Project,Value=$TAG}]" "ResourceType=volume,Tags=[{Key=Project,Value=$TAG}]" \
      --query 'Instances[0].InstanceId' --output text)
    echo "instance=$id"
    for _ in $(seq 1 60); do   # up to 20 minutes for the packages and the runner
      runners | grep -q "card-$id online" && { echo "runner card-$id online"; exit 0; }
      sleep 20
    done
    echo "the runner of $id did not come online: terminating it" >&2
    aws ec2 terminate-instances --instance-ids "$id" > /dev/null
    exit 1
    ;;
  stop)
    id="${2:?usage: stop <instance-id>}"
    aws ec2 terminate-instances --instance-ids "$id" --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
    runners | while read -r rid name _; do [[ "$name" == "card-$id" ]] && gh api -X DELETE "repos/$REPO/actions/runners/$rid" && echo "removed runner $name"; done || true
    ;;
  status)
    echo "instances: $(instances pending,running,stopping)"
    runners
    ;;
  *) sed -n 2,13p "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
