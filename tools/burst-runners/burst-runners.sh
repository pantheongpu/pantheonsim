#!/usr/bin/env bash
# A pool of AWS instances running GitHub Actions runners for the long CI jobs
# (SASS, ASan, TSan: BURST_RUNS_ON in .github/workflows/ci.yml), for when the
# pull request backlog is long.
#
#   tools/burst-runners/burst-runners.sh start [instances]    (default 2)
#   tools/burst-runners/burst-runners.sh status
#   tools/burst-runners/burst-runners.sh stop
#
# start   launches the instances (two runner processes each, m7i.2xlarge,
#         on-demand, no inbound access), waits for their runners to come online,
#         then sets the repository variable BURST_RUNS_ON.
# stop    unsets the variable, terminates the instances and removes their
#         runners. The instances also shut themselves down (they terminate on
#         shutdown) when no pull request has been open for 15 minutes, or after
#         24 hours: userdata.tmpl.
#
# Needs the aws CLI (credentials for an account with room for the vCPUs) and an
# authenticated gh with admin on the repository. Set AWS_PROFILE and AWS_REGION
# as usual. The registration token is passed to the instances in their user
# data and expires after an hour.
set -euo pipefail
cd "$(dirname "$0")"
REPO="${BURST_REPO:-pantheongpu/pantheonsim}"
REGION="${AWS_REGION:-us-east-1}"
TYPE="${BURST_INSTANCE_TYPE:-m7i.2xlarge}"
LABEL=pantheonsim-burst
TAG=pantheonsim-ci-burst
export AWS_DEFAULT_REGION="$REGION"

instances() {
  aws ec2 describe-instances --filters "Name=tag:Project,Values=$TAG" \
    "Name=instance-state-name,Values=${1:-pending,running}" --query 'Reservations[].Instances[].InstanceId' --output text
}
runners() { gh api "repos/$REPO/actions/runners" --paginate --jq ".runners[] | select(.name | startswith(\"burst-\")) | \"\(.id) \(.name) \(.status)\""; }

case "${1:-}" in
  start)
    n="${2:-2}"
    [[ -z "$(instances)" ]] || { echo "already running: $(instances)"; exit 1; }
    # The default VPC's subnet and the newest Ubuntu 24.04, as the playground host uses.
    ami=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)
    subnet=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true --query 'Subnets[0].SubnetId' --output text)
    vpc=$(aws ec2 describe-subnets --subnet-ids "$subnet" --query 'Subnets[0].VpcId' --output text)
    sg=$(aws ec2 describe-security-groups --filters Name=group-name,Values=pantheonsim-burst-runners Name=vpc-id,Values="$vpc" --query 'SecurityGroups[0].GroupId' --output text)
    if [[ "$sg" == None ]]; then   # no inbound rule at all: the runners only call out
      sg=$(aws ec2 create-security-group --group-name pantheonsim-burst-runners --description "pantheonsim CI burst runners: outbound only" --vpc-id "$vpc" --query GroupId --output text)
    fi
    ud=$(mktemp); trap 'rm -f "$ud"' EXIT; chmod 600 "$ud"
    tok=$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token)
    sed "s/__TOKEN__/$tok/" userdata.tmpl > "$ud"; unset tok
    ids=$(aws ec2 run-instances --image-id "$ami" --instance-type "$TYPE" --count "$n" --subnet-id "$subnet" \
      --security-group-ids "$sg" --metadata-options HttpTokens=required,HttpEndpoint=enabled \
      --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
      --user-data "file://$ud" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=pantheonsim-burst-runner},{Key=Project,Value=$TAG}]" \
                           "ResourceType=volume,Tags=[{Key=Project,Value=$TAG}]" \
      --query 'Instances[].InstanceId' --output text)
    echo "launched: $ids"
    want=$((n * 2))
    for _ in $(seq 1 60); do   # up to 20 minutes for Docker and the runners
      have=$(runners | grep -c ' online$' || true)
      [[ "$have" -ge "$want" ]] && break
      sleep 20
    done
    runners
    [[ "$have" -ge "$want" ]] || { echo "only $have of $want runners online: not setting BURST_RUNS_ON; see 'aws ec2 get-console-output'"; exit 1; }
    gh variable set BURST_RUNS_ON --repo "$REPO" --body "[\"self-hosted\",\"$LABEL\"]"
    echo "BURST_RUNS_ON set: the SASS, ASan and TSan jobs go to the pool"
    ;;
  status)
    echo "instances: $(instances pending,running,stopping)"
    echo "BURST_RUNS_ON: $(gh variable get BURST_RUNS_ON --repo "$REPO" 2>/dev/null || echo unset)"
    runners
    echo "open pull requests: $(gh pr list --repo "$REPO" --state open --json number --jq length)"
    ;;
  stop)
    gh variable delete BURST_RUNS_ON --repo "$REPO" 2>/dev/null && echo "BURST_RUNS_ON unset" || true
    ids=$(instances pending,running,stopping)
    [[ -z "$ids" ]] || aws ec2 terminate-instances --instance-ids $ids --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
    # Their runners stay listed as offline until removed.
    runners | while read -r id name _; do gh api -X DELETE "repos/$REPO/actions/runners/$id" && echo "removed runner $name"; done
    ;;
  *) sed -n 2,22p "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
