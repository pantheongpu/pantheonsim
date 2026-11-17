#!/usr/bin/env bash
# Use the playground host as a burst runner, from a maintainer's machine (no token on the host).
#   host-burst.sh check | install | start [--no-route] | status | stop
# The host is 3 slots of 8/3 vCPUs; it takes jobs only when 2 are free (see docs/ci.md).
set -euo pipefail
REPO=pantheongpu/pantheonsim
HOST=ubuntu@3.231.61.49
KEY=~/.ssh/pantheonsim-web.pem
LABEL=pantheonsim-hostburst
DIR=/opt/hostburst
SAVED=/tmp/hostburst-saved-vars
host() { ssh -i "$KEY" -o StrictHostKeyChecking=no "$HOST" "$@"; }
runners() { gh api "repos/$REPO/actions/runners" --jq '.runners[] | select(.name | startswith("hostburst-")) | "\(.id) \(.name) \(.status)"'; }

declare -A VARS=(
  [BURST_RUNS_ON]="[\"$LABEL\"]"
  [BURST_CONTAINER_OPTIONS]="--cpus=2.6 --memory=9g --cpuset-cpus=0-3 --cpu-shares=128"
  [BURST_JOBS_MAX]=2
  [BURST_MEM_LIMIT_GIB]=8
)

gate() {
  read -r sessions ml load < <(host 'echo $(curl -s localhost:8080/api/health | jq -r .sessions) \
    $(sudo docker exec gh-runner-pantheonsim pgrep -f Runner.Worker >/dev/null && echo 1 || echo 0) $(cut -d" " -f1 /proc/loadavg)')
  [[ $sessions =~ ^[0-9]+$ ]] || { echo "can't read the playground: not using the host"; return 3; }
  # a session is 1 vCPU, the ML job is 4
  free=$(awk -v s="$sessions" -v m="$ml" -v l="$load" 'BEGIN { u = l; if (m * 4 > u) u = m * 4; if (s > u) u = s; f = int((8 - u) / (8 / 3) + 1e-9); print (f < 0 ? 0 : f) }')
  echo "sessions $sessions, ML job $([[ $ml == 1 ]] && echo running || echo idle), load $load: $free of 3 slots free"
  (( free >= 2 )) || { echo "fewer than 2 slots free: not using the host"; return 3; }
}

case "${1:-}" in
  check) gate ;;
  install)
    host "sudo mkdir -p $DIR && cd $DIR && [ -x r1/config.sh ] || {
      sudo curl -sL --retry 5 --retry-all-errors -o r.tgz https://github.com/actions/runner/releases/download/v2.337.0/actions-runner-linux-x64-2.337.0.tar.gz
      for i in 1 2; do sudo mkdir -p r\$i && sudo tar xzf r.tgz -C r\$i; done; sudo rm r.tgz; }" ;;
  start)
    gate
    [[ -z $(runners) ]] || { echo "already registered"; exit 1; }
    for i in 1 2; do
      tok=$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token)
      # root, like the AWS pool: container jobs leave root-owned files in the workspace
      host "cd $DIR/r$i && echo RUNNER_ALLOW_RUNASROOT=1 | sudo tee .env >/dev/null &&
        sudo RUNNER_ALLOW_RUNASROOT=1 ./config.sh --unattended --url https://github.com/$REPO --token $tok --name hostburst-$i \
          --no-default-labels --labels $LABEL --replace >/dev/null && sudo ./svc.sh install root >/dev/null && sudo ./svc.sh start >/dev/null"
    done
    for _ in {1..30}; do [[ $(runners | grep -c online) == 2 ]] && break; sleep 5; done
    [[ $(runners | grep -c online) == 2 ]] || { echo "runners not online; run stop"; exit 1; }
    runners
    [[ ${2:-} == --no-route ]] && { echo "up, not routed"; exit 0; }
    : > $SAVED
    for v in "${!VARS[@]}"; do
      echo -e "$v\t$(gh variable get "$v" --repo $REPO 2>/dev/null || echo __unset__)" >> $SAVED
      gh variable set "$v" --repo $REPO --body "${VARS[$v]}"
    done
    echo "routed. Jobs already queued keep their label: cancel and re-run them." ;;
  status)
    gate || true
    runners
    for v in "${!VARS[@]}"; do echo "$v=$(gh variable get "$v" --repo $REPO 2>/dev/null || echo unset)"; done ;;
  stop)
    if [[ -r $SAVED ]]; then
      while IFS=$'\t' read -r v old; do
        [[ $old == __unset__ ]] && gh variable delete "$v" --repo $REPO || gh variable set "$v" --repo $REPO --body "$old"
      done < $SAVED
      rm $SAVED
    else   # no record here: at least stop routing to runners that are going away
      gh variable get BURST_RUNS_ON --repo $REPO 2>/dev/null | grep -q $LABEL && gh variable delete BURST_RUNS_ON --repo $REPO
    fi
    # a busy runner refuses to be removed: wait for it
    while [[ -n $(runners) ]]; do
      runners | while read -r id _; do gh api -X DELETE "repos/$REPO/actions/runners/$id" >/dev/null 2>&1 || true; done
      [[ -z $(runners) ]] || sleep 60
    done
    host "for i in 1 2; do cd $DIR/r\$i && sudo ./svc.sh stop; sudo ./svc.sh uninstall; done >/dev/null 2>&1 || true" ;;
  *) sed -n 2,4p "$0" | sed 's/^# *//'; exit 2 ;;
esac
