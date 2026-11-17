#!/usr/bin/env bash
# The playground host (pantheonsim.com) as a burst runner, for when the pull request
# backlog is long and the host is idle. Run from a maintainer's machine: no GitHub
# token is ever stored on the host (registration tokens last an hour and are passed
# over ssh).
#
#   tools/burst-runners/host-burst.sh check            can the host take jobs now? (exit 0 yes, 3 no)
#   tools/burst-runners/host-burst.sh install          the runner software on the host (once; changes nothing else)
#   tools/burst-runners/host-burst.sh start [--no-route]   gate, register 2 runners, route the burst jobs to them
#   tools/burst-runners/host-burst.sh status
#   tools/burst-runners/host-burst.sh stop             unroute (restores the variables), stop and unregister
#
# The host has 8 vCPUs and 30 GB: CPUs 4-7 are the ML runner (pantheonsim-ci, one job at a time), CPUs 0-3 are the
# live playground. The host is counted as 3 slots of 8/3 vCPUs. It takes burst jobs only when at least 2 of the 3
# slots are free (the gate below); with only 1 free it does not, and the jobs stay on the other runners or on
# GitHub's hosted ones. A slot is used by: the ML job running (4 vCPUs), a playground session (1 vCPU each), or load
# for any other reason (the 1-minute load average). The gate is read when `start` runs, not while jobs run: a job
# already running is not stopped when a session arrives; it is limited to CPUs 0-3 at a low CPU weight, so that
# the playground wins any contention.
#
# The runners carry the one label pantheonsim-hostburst and none of the defaults, so no other job can match them
# (docs/ci.md, "The burst pool is offline"). `start` sets BURST_RUNS_ON to that label only, so the jobs do NOT also go to the
# other burst runners (server1-sim, server2-*), which carry pantheonsim-burst. `stop` puts every variable it changed
# back as it was. While BURST_RUNS_ON names this label and the runners are not up, burst jobs wait: do not leave it set.
set -euo pipefail
REPO="${BURST_REPO:-pantheongpu/pantheonsim}"
HOST="${BURST_HOST:-ubuntu@3.231.61.49}"
KEY="${BURST_HOST_KEY:-$HOME/.ssh/pantheonsim-web.pem}"
LABEL=pantheonsim-hostburst
RUNNER_VERSION=2.337.0
NRUNNERS=2
IDS="$(seq -s " " 1 2)"   # "1 2", on one line, for the commands that run on the host
DIR=/opt/hostburst
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$HOST")
saved=/tmp/hostburst-saved-vars   # on THIS machine: the variables as they were before `start`

host() { "${SSH[@]}" "$@"; }

# The gate. Prints what it saw; exit 3 when fewer than 2 of 3 slots are free.
gate() {
  local out
  out=$(host 'sessions=$(curl -s --max-time 10 localhost:8080/api/health | jq -r .sessions 2>/dev/null)
    ml=0; sudo -n docker exec gh-runner-pantheonsim pgrep -f Runner.Worker >/dev/null 2>&1 && ml=1
    burst=0; pgrep -f "'"$DIR"'/.*Runner.Worker" >/dev/null 2>&1 && burst=1
    echo "$sessions $ml $burst $(cut -d" " -f1 /proc/loadavg)"') || { echo "cannot reach the host"; return 3; }
  read -r sessions ml burst load <<< "$out"
  [[ "$sessions" =~ ^[0-9]+$ ]] || { echo "playground health unreadable ('$sessions'): not using the host"; return 3; }
  python3 - "$sessions" "$ml" "$burst" "$load" <<'PY'
import math, sys
sessions, ml, burst, load = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
slot = 8 / 3
# What is already using the host, in vCPUs. A burst job of our own is not counted against the gate (it is what the gate lets in).
used = max(load if not burst else 0.0, 4.0 if ml else 0.0, float(sessions))
free = math.floor((8 - used) / slot + 1e-9)
print(f"playground sessions {sessions}, ML job {'running' if ml else 'idle'}, load {load}: about {used:.1f} of 8 vCPUs in use, {free} of 3 slots free")
if free < 2:
    print("fewer than 2 of 3 slots free: not using the host"); sys.exit(3)
PY
}

variables=(BURST_RUNS_ON BURST_CONTAINER_OPTIONS BURST_JOBS_MAX BURST_MEM_LIMIT_GIB)
# What the burst jobs get on the host: one job's container is limited to CPUs 0-3, 2.6 vCPUs, 9 GB, low CPU weight.
declare -A host_value=(
  [BURST_RUNS_ON]="[\"$LABEL\"]"
  [BURST_CONTAINER_OPTIONS]="--cpus=2.6 --memory=9g --cpuset-cpus=0-3 --cpu-shares=128"
  [BURST_JOBS_MAX]=2
  [BURST_MEM_LIMIT_GIB]=8
)

runners() { gh api "repos/$REPO/actions/runners" --paginate --jq ".runners[] | select(.name | startswith(\"hostburst-\")) | \"\(.id) \(.name) \(.status) busy=\(.busy)\""; }

case "${1:-}" in
  check) gate ;;
  install)
    host "sudo -n true && set -e; sudo mkdir -p $DIR; cd $DIR
      if [ ! -x r1/config.sh ]; then
        sudo curl -sL -o runner.tgz https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
        for i in $IDS; do sudo mkdir -p r\$i; sudo tar xzf runner.tgz -C r\$i; done
        sudo rm -f runner.tgz
      fi
      ls $DIR; df -h / | tail -1"
    echo "installed in $DIR on the host; nothing is registered or running"
    ;;
  start)
    route=1; [[ "${2:-}" == --no-route ]] && route=0
    gate || exit $?
    [[ -z "$(runners)" ]] || { echo "runners already registered: $(runners | tr '\n' ';')"; exit 1; }
    host "test -x $DIR/r1/config.sh" || { echo "run 'install' first"; exit 1; }
    for i in $(seq 1 $NRUNNERS); do
      tok=$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token)
      # As root, as the AWS pool's runners are (userdata.tmpl): container jobs leave root-owned files in the workspace.
      host "cd $DIR/r$i && echo RUNNER_ALLOW_RUNASROOT=1 | sudo tee .env >/dev/null && sudo RUNNER_ALLOW_RUNASROOT=1 ./config.sh --unattended \
        --url https://github.com/$REPO --token '$tok' --name hostburst-$i --no-default-labels --labels $LABEL --replace --work _work >/dev/null \
        && sudo ./svc.sh install root >/dev/null && sudo ./svc.sh start >/dev/null && echo started runner $i"
      unset tok
    done
    for _ in $(seq 1 30); do [[ $(runners | grep -c ' online ') -ge $NRUNNERS ]] && break; sleep 5; done
    runners
    [[ $(runners | grep -c ' online ') -ge $NRUNNERS ]] || { echo "runners did not come online: not routing; run 'stop'"; exit 1; }
    if (( route )); then
      : > "$saved"
      for v in "${variables[@]}"; do
        printf '%s\t%s\n' "$v" "$(gh variable get "$v" --repo "$REPO" 2>/dev/null || echo __unset__)" >> "$saved"
        gh variable set "$v" --repo "$REPO" --body "${host_value[$v]}"
      done
      echo "routed: BURST_RUNS_ON=${host_value[BURST_RUNS_ON]}; variables as they were before are in $saved (stop restores them)"
      echo "jobs already queued keep the label they were queued with: cancel and re-run them (docs/ci.md, 'The burst pool is offline')"
    else
      echo "runners are up and NOT routed (--no-route)"
    fi
    ;;
  status)
    gate || true
    echo "runners: $(runners | tr '\n' ';')"
    for v in "${variables[@]}"; do echo "$v=$(gh variable get "$v" --repo "$REPO" 2>/dev/null || echo unset)"; done
    ;;
  stop)
    if [[ -r "$saved" ]]; then   # restore what start changed
      while IFS=$'\t' read -r v old; do
        if [[ "$old" == __unset__ ]]; then gh variable delete "$v" --repo "$REPO" 2>/dev/null && echo "$v unset" || true
        else gh variable set "$v" --repo "$REPO" --body "$old" && echo "$v restored"; fi
      done < "$saved"
      rm -f "$saved"
    elif [[ "$(gh variable get BURST_RUNS_ON --repo "$REPO" 2>/dev/null || true)" == *"$LABEL"* ]]; then
      # (no saved file on this machine: the variable names the host's label, so it is not left pointing at runners that go away)
      gh variable delete BURST_RUNS_ON --repo "$REPO" && echo "BURST_RUNS_ON unset (no saved values here: check BURST_CONTAINER_OPTIONS, BURST_JOBS_MAX, BURST_MEM_LIMIT_GIB by hand)"
    fi
    # A runner that is running a job refuses to be removed: wait for it (up to 3 hours).
    for _ in $(seq 1 180); do
      left=0
      while read -r id name _; do
        gh api -X DELETE "repos/$REPO/actions/runners/$id" >/dev/null 2>&1 && echo "removed runner $name" || left=1
      done < <(runners)
      (( left == 0 )) && break
      sleep 60
    done
    host "for i in $IDS; do cd $DIR/r\$i 2>/dev/null && sudo ./svc.sh stop >/dev/null 2>&1; sudo ./svc.sh uninstall >/dev/null 2>&1; done; sudo rm -rf $DIR/r*/_work/* 2>/dev/null; sudo -n docker ps --format '{{.Names}}' | head -20 >/dev/null; echo stopped"
    ;;
  *) sed -n 2,28p "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
