#!/usr/bin/env bash
# ci/jobs.sh sizes CI's parallelism to the memory a machine has. Runs it against
# synthetic /proc/meminfo and cgroup files (CI_MEMINFO, CI_CGROUP_ROOT,
# CI_PROC_CGROUP, CI_NPROC) and checks the number of jobs, the early memory
# check, the overrides, and that a hosted runner (4 CPUs and 16 GB, or 2 CPUs
# and 7 GB) gets one job per CPU under every profile, as it did with $(nproc).
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
jobs="$root/ci/jobs.sh"
[[ -x $jobs ]] || { echo "FAIL: $jobs is not executable"; exit 1; }
d="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_ci_jobs.XXXXXX")"
trap 'rm -rf "$d"' EXIT
unset CI_JOBS CI_JOBS_MAX CI_MEM_CHECK CI_MEM_LIMIT_GIB CI_RESERVE CI_RESERVE_CAPACITY_GIB CI_RESERVE_TTL_SECS CI_RESERVE_ID RUNNER_NAME
# The reservation ledger of every run below is a directory of its own, never the host's.
export CI_RESERVE_DIR="$d/reserve" CI_MEM_SETTLE_SECS=0 CI_NOW=1000000

GiB=1073741824
fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }

# fixture NAME MEMAVAILABLE_KIB: a machine whose cgroup is /ci under a root with no limits yet
fixture() {
  mkdir -p "$d/$1/cg/ci"
  printf 'MemTotal: %d kB\nMemFree: 1000 kB\nMemAvailable: %d kB\nCached: 5 kB\n' "$(($2 * 2))" "$2" > "$d/$1/meminfo"
  echo "0::/ci" > "$d/$1/proc_cgroup"
}
# run NAME NPROC args...: ci/jobs.sh on that fixture
run() {
  local n=$1 cpus=$2; shift 2
  CI_MEMINFO="$d/$n/meminfo" CI_CGROUP_ROOT="$d/$n/cg" CI_PROC_CGROUP="$d/$n/proc_cgroup" CI_NPROC="$cpus" "$jobs" "$@"
}
expect() {   # label want got
  [[ "$3" == "$2" ]] || fail "$1: wanted '$2', got '$3'"
}
kib() { echo $(( $1 * 1048576 )); }   # GiB to KiB
frac() { awk -v g="$1" 'BEGIN { printf "%d", g * 1048576 }'; }

# 1. No cgroup limit: memory alone decides.
fixture big "$(kib 100)"
expect "ample memory, 24 CPUs" 24 "$(run big 24)"
fixture mid "$(kib 30)"
expect "30 GiB at 2 GiB per job, 24 CPUs" 15 "$(run mid 24)"
expect "30 GiB at 3 GiB per job" 10 "$(run mid 24 -g 3)"
expect "a decimal per job: 30 GiB at 1.5" 20 "$(run mid 24 -g 1.5)"
expect "the positional form" 10 "$(run mid 24 3)"
fixture tiny "$(frac 0.5)"
expect "less than one job's memory still runs one" 1 "$(run tiny 24)"

# 2. A cgroup v2 cap below the host's memory: 26 GiB cap, 20 GiB in use.
fixture capped "$(kib 28)"
echo $((26 * GiB)) > "$d/capped/cg/ci/memory.max"
echo $((20 * GiB)) > "$d/capped/cg/ci/memory.current"
expect "26 GiB cap with 20 in use, 24 CPUs, 2 GiB per job" 3 "$(run capped 24)"
expect "the same at 3 GiB per job" 2 "$(run capped 24 -g 3)"
# The inactive file cache counts as free: the kernel drops it before it kills anything.
printf 'anon 1\ninactive_file %d\nactive_file 5\n' $((4 * GiB)) > "$d/capped/cg/ci/memory.stat"
expect "4 GiB of inactive cache raises the headroom to 10 GiB" 5 "$(run capped 24)"
rm "$d/capped/cg/ci/memory.stat"
# A parent's cap counts too: 8 GiB left in the parent, the child unlimited.
fixture parent "$(kib 28)"
mkdir -p "$d/parent/cg/ci/inner"
echo "0::/ci/inner" > "$d/parent/proc_cgroup"
echo max > "$d/parent/cg/ci/inner/memory.max"
echo $((1 * GiB)) > "$d/parent/cg/ci/inner/memory.current"
echo $((10 * GiB)) > "$d/parent/cg/ci/memory.max"
echo $((2 * GiB)) > "$d/parent/cg/ci/memory.current"
expect "a parent cgroup's 8 GiB headroom limits its unlimited child" 4 "$(run parent 24)"
# "max" alone means no limit.
fixture unlimited "$(kib 12)"
echo max > "$d/unlimited/cg/ci/memory.max"
echo $((1 * GiB)) > "$d/unlimited/cg/ci/memory.current"
expect "memory.max of max falls back to MemAvailable" 6 "$(run unlimited 24)"
# A cgroup that is over its limit leaves no headroom, and one job still runs.
fixture over "$(kib 28)"
echo $((4 * GiB)) > "$d/over/cg/ci/memory.max"
echo $((5 * GiB)) > "$d/over/cg/ci/memory.current"
expect "over its limit" 1 "$(run over 24)"

# 3. cgroup v1 (a container's own memory controller), limit set and unset.
fixture v1 "$(kib 28)"
mkdir -p "$d/v1/cg/memory"
echo $((9 * GiB)) > "$d/v1/cg/memory/memory.limit_in_bytes"
echo $((3 * GiB)) > "$d/v1/cg/memory/memory.usage_in_bytes"
expect "cgroup v1: 9 GiB limit, 3 in use" 3 "$(run v1 24)"
echo 9223372036854771712 > "$d/v1/cg/memory/memory.limit_in_bytes"
expect "cgroup v1 with no limit set" 14 "$(run v1 24)"

# 4. The overrides.
expect "CI_JOBS forces a number" 7 "$(CI_JOBS=7 run mid 24)"
expect "CI_JOBS may exceed the CPUs" 40 "$(CI_JOBS=40 run tiny 4)"
expect "-m caps the number" 2 "$(run big 24 -m 2)"
expect "CI_JOBS_MAX caps the number" 2 "$(CI_JOBS_MAX=2 run big 24)"
expect "-m does not raise it" 1 "$(run tiny 24 -m 2)"
expect "CI_MEM_LIMIT_GIB is a ceiling" 3 "$(CI_MEM_LIMIT_GIB=6 run big 24)"
expect "CI_MEM_LIMIT_GIB above the memory changes nothing" 15 "$(CI_MEM_LIMIT_GIB=500 run mid 24)"
if CI_JOBS=0 run mid 24 >/dev/null 2>&1; then fail "CI_JOBS=0 was accepted"; fi
if run mid 24 -p nonsense >/dev/null 2>&1; then fail "an unknown profile was accepted"; fi
if run mid 24 -g abc >/dev/null 2>&1; then fail "a non-numeric GiB per job was accepted"; fi

# 5. Memory that cannot be read: the CPU count, as before.
out="$(CI_MEMINFO="$d/nonexistent" CI_CGROUP_ROOT="$d/nonexistent" CI_PROC_CGROUP="$d/nonexistent" CI_NPROC=6 "$jobs")"
expect "no memory information" 6 "$out"

# 6. The early check, on a machine with 8 GiB and one with 3 GiB.
fixture ok "$(kib 8)"
out="$(run ok 24 --check asan 2>&1)"; rc=$?
expect "asan check with 8 GiB: exit" 0 "$rc"
[[ $out == *"8.0 GiB available"*"asan needs 5 GiB"* ]] || fail "the check's report: $out"
fixture low "$(kib 3)"
out="$(run low 24 --check asan 2>&1)"; rc=$?
expect "asan check with 3 GiB: exit" 1 "$rc"
[[ $out == *"only 3.0 GiB"*"needs 5 GiB"* ]] || fail "the check's refusal: $out"
out="$(GITHUB_ACTIONS=true run low 24 --check tsan 2>&1)"
[[ $out == *"::error::"* ]] || fail "under GitHub Actions the refusal is an error annotation: $out"
out="$(CI_MEM_CHECK=0 run low 24 --check asan 2>&1)"; rc=$?
expect "CI_MEM_CHECK=0 skips the check" 0 "$rc"
run low 24 --check build >/dev/null 2>&1 || fail "a profile with no minimum failed the check"
if run low 24 --check >/dev/null 2>&1; then fail "--check without a profile was accepted"; fi
# Waiting: CI_MEM_WAIT_SECS makes --check look again before it refuses.
out="$(CI_MEM_WAIT_SECS=2 CI_MEM_POLL_SECS=1 run low 24 --check asan 2>&1)"; rc=$?
expect "a short wait on 3 GiB still refuses: exit" 1 "$rc"
[[ $out == *"waiting up to 2 s"*"only 3.0 GiB"* ]] || fail "the wait's report: $out"
fixture later "$(kib 3)"
( sleep 1; printf 'MemTotal: %d kB\nMemFree: 1000 kB\nMemAvailable: %d kB\nCached: 5 kB\n' "$(kib 16)" "$(kib 8)" > "$d/later/meminfo" ) &
out="$(CI_MEM_WAIT_SECS=10 CI_MEM_POLL_SECS=1 run later 24 --check asan 2>&1)"; rc=$?
wait
expect "memory that appears while waiting: exit" 0 "$rc"
[[ $out == *"8.0 GiB available after waiting"* ]] || fail "the wait's success: $out"
out="$(CI_MEM_WAIT_SECS=nonsense run low 24 --check asan 2>&1)"; rc=$?
expect "a wait that is not a number is no wait: exit" 1 "$rc"
# A cgroup cap, not the host's memory, is what the check refuses on.
fixture sharedbox "$(kib 28)"
echo $((26 * GiB)) > "$d/sharedbox/cg/ci/memory.max"
echo $((24 * GiB)) > "$d/sharedbox/cg/ci/memory.current"
run sharedbox 24 --check sass >/dev/null 2>&1 && fail "a box with 2 GiB of cgroup headroom passed the sass check"
expect "--available" "2.0" "$(run sharedbox 24 --available)"

# 7. Hosted runners keep their speed: one job per CPU under every profile, and
# every check passes. 4 CPUs and 16 GB has about 14.5 GiB available, 2 CPUs and
# 7 GB about 6 GiB.
fixture hosted4 "$(frac 14.5)"
fixture hosted2 "$(frac 6)"
for p in $("$jobs" --profiles | awk 'NR > 1 { print $1 }'); do
  expect "4 CPUs / 16 GB, profile $p" 4 "$(run hosted4 4 -p "$p")"
  expect "2 CPUs / 7 GB, profile $p" 2 "$(run hosted2 2 -p "$p")"
  run hosted4 4 --check "$p" >/dev/null 2>&1 || fail "the $p check refused a 4 CPU / 16 GB runner"
  run hosted2 2 --check "$p" >/dev/null 2>&1 || fail "the $p check refused a 2 CPU / 7 GB runner"
done
# A 24 CPU, 26 GB shared box is what the profiles are for.
fixture server "$(kib 26)"
expect "24 CPUs / 26 GB, release" 8 "$(run server 24 -p release)"

# 9. The reservation (#575, #576): --check takes a share of a ledger before it samples the memory,
# so two heavy jobs cannot both pass on the same free memory; --release gives it back, and a
# share a killed runner never released expires.
fixture roomy "$(kib 64)"
export CI_RESERVE_DIR="$d/ledger9"
rj() { local id=$1; shift; CI_RESERVE_ID=$id run roomy 24 "$@"; }   # rj JOBID args...
out="$(rj a --check sass 2>&1)"; rc=$?
expect "the first heavy job passes: exit" 0 "$rc"
[[ $out == *"reserved 5 GiB as a "* ]] || fail "the first job's report: $out"
expect "the ledger holds one entry" 1 "$(rj a --reservations | wc -l)"
out="$(rj b --check ollama 2>&1)"; rc=$?
expect "a second heavy job with no wait is refused although 64 GiB are free: exit" 1 "$rc"
[[ $out == *"a=5120MiB"* ]] || fail "the refusal names the holder: $out"
expect "the refused job took no share" 1 "$(rj a --reservations | wc -l)"
# It waits, and gets in when the first releases.
( sleep 2; rj a --release >/dev/null ) &
out="$(CI_MEM_WAIT_SECS=20 CI_MEM_POLL_SECS=1 rj b --check ollama 2>&1)"; rc=$?
wait
expect "the waiting job gets in once the first releases: exit" 0 "$rc"
[[ $out == *"waiting up to 20 s"*"reserved 5 GiB as b"*"after waiting"* ]] || fail "the wait: $out"
expect "only the second job holds a share now" "b" "$(rj a --reservations | awk '{ print $1 }')"
# Release is idempotent and never fails; releasing one job leaves the other's share.
rj a --release >/dev/null || fail "releasing a share that is not held failed"
expect "an unrelated release changed nothing" 1 "$(rj a --reservations | wc -l)"
rj b --release >/dev/null
expect "the ledger is empty after both release" 0 "$(rj a --reservations | wc -l)"
# A share that was never released (the runner was killed) expires after its time to live.
rj dead --check asan >/dev/null 2>&1
out="$(CI_NOW=$((1000000 + 100)) rj live --check asan 2>&1)"; rc=$?
expect "a stale share still holds inside its time to live: exit" 1 "$rc"
out="$(CI_NOW=$((1000000 + 14401)) rj live --check asan 2>&1)"; rc=$?
expect "an expired share is dropped: exit" 0 "$rc"
expect "only the live job is in the ledger" "live" "$(rj a --reservations | awk '{ print $1 }')"
rj live --release >/dev/null
# A share whose run has ended (the runner was killed before --release) is dropped by a waiting --check
# that asks the API (a file tree stands in for it); a run still going, or no answer, keeps its share.
runs="$d/api/repos/o/r/actions/runs"; mkdir -p "$runs/111/attempts" "$runs/222/attempts"
printf '{\n  "id": 111,\n  "status": "completed",\n  "conclusion": "failure"\n}\n' > "$runs/111/attempts/1"
printf '{\n  "id": 222,\n  "status": "in_progress",\n  "jobs": [\n    {\n      "status": "completed"\n    }\n  ]\n}\n' > "$runs/222/attempts/1"
api() { GH_TOKEN=t GITHUB_REPOSITORY=o/r GITHUB_API_URL="file://$d/api" rj "$@"; }
rj 111-1-s-sass --check sass >/dev/null 2>&1 || fail "the share of the finished run was not taken"
out="$(api 999-1-s-asan --check asan 2>&1)"; rc=$?
expect "a job gets in once the share of a finished run is dropped: exit" 0 "$rc"
[[ $out == *"dropped the share of 111-1-s-sass"* ]] || fail "the drop is not said: $out"
expect "only the new job holds a share" "999-1-s-asan" "$(rj a --reservations | awk '{ print $1 }')"
rj 999-1-s-asan --release >/dev/null
rj 222-1-s-sass --check sass >/dev/null 2>&1 || fail "the share of the running run was not taken"
api 999-1-s-asan --check asan >/dev/null 2>&1 && fail "a share of a run still going was dropped"
rj 222-1-s-sass --release >/dev/null
rj 333-1-s-sass --check sass >/dev/null 2>&1 || fail "the share of the unknown run was not taken"
api 999-1-s-asan --check asan >/dev/null 2>&1 && fail "a share whose run could not be asked about was dropped"
rj 333-1-s-sass --release >/dev/null
# A capacity of two jobs lets two in and refuses the third.
export CI_RESERVE_CAPACITY_GIB=10
rj a --check sass >/dev/null 2>&1 || fail "capacity 10: the first was refused"
rj b --check sass >/dev/null 2>&1 || fail "capacity 10: the second was refused"
rj c --check sass >/dev/null 2>&1 && fail "capacity 10: the third was admitted"
rj a --release >/dev/null; rj b --release >/dev/null
unset CI_RESERVE_CAPACITY_GIB
# On an AWS burst runner (32 GiB, two runners) the default capacity is two heavy jobs; elsewhere one.
RUNNER_NAME=burst-i-0123456789abcdef0-1 rj a --check sass >/dev/null 2>&1 || fail "burst default: the first was refused"
RUNNER_NAME=burst-i-0123456789abcdef0-2 rj b --check sass >/dev/null 2>&1 || fail "burst default: the second was refused"
RUNNER_NAME=burst-i-0123456789abcdef0-1 rj c --check sass >/dev/null 2>&1 && fail "burst default: the third was admitted"
RUNNER_NAME=server2-1 rj c --check sass >/dev/null 2>&1 && fail "a non-burst runner took a second job"
rj a --release >/dev/null; rj b --release >/dev/null
# A capacity below the profile's minimum cannot lock a profile out for good.
CI_RESERVE_CAPACITY_GIB=1 rj a --check sass >/dev/null 2>&1 || fail "a capacity of 1 GiB locked the 5 GiB profile out"
rj a --release >/dev/null
# A job that fails the memory check gives its share back at once.
fixture starved "$(kib 3)"
CI_RESERVE_ID=s run starved 24 --check asan >/dev/null 2>&1 && fail "the starved box passed"
expect "a failed memory check keeps no share" 0 "$(rj a --reservations | wc -l)"
# Profiles with no minimum reserve nothing; CI_RESERVE=0 and an unusable ledger directory
# fall back to sampling, with a note, and never fail by themselves.
rj a --check release >/dev/null 2>&1 || fail "release check failed"
rj a --check build >/dev/null 2>&1 || fail "build check failed"
expect "profiles with no minimum take no share" 0 "$(rj a --reservations | wc -l)"
CI_RESERVE=0 rj a --check sass >/dev/null 2>&1 || fail "CI_RESERVE=0 check failed"
expect "CI_RESERVE=0 takes no share" 0 "$(rj a --reservations | wc -l)"
touch "$d/notadir"
out="$(CI_RESERVE_DIR="$d/notadir/x" rj a --check sass 2>&1)"; rc=$?
expect "an unusable ledger directory: exit" 0 "$rc"
[[ $out == *"no reservation"* ]] || fail "an unusable ledger directory is said: $out"
# Many jobs racing for a one-job ledger: exactly one wins.
rm -rf "$d/ledger9"; wins="$d/wins"; : > "$wins"
for i in 1 2 3 4 5 6 7 8; do
  ( rj "r$i" --check sass >/dev/null 2>&1 && echo "r$i" >> "$wins" ) &
done
wait
expect "eight concurrent heavy jobs: exactly one admitted" 1 "$(wc -l < "$wins")"
expect "and one share in the ledger" 1 "$(rj a --reservations | wc -l)"
export CI_RESERVE_DIR="$d/reserve"

# 8. On this machine it prints a positive integer.
out="$("$jobs" -p release)"
[[ $out =~ ^[1-9][0-9]*$ ]] || fail "on this machine: '$out'"

if (( fails )); then echo "FAIL: $fails check(s) failed"; exit 1; fi
echo "PASS: ci/jobs.sh sizes jobs to memory, cgroup limits, overrides and the early check"
