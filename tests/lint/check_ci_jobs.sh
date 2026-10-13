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
unset CI_JOBS CI_JOBS_MAX CI_MEM_CHECK CI_MEM_LIMIT_GIB

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

# 8. On this machine it prints a positive integer.
out="$("$jobs" -p release)"
[[ $out =~ ^[1-9][0-9]*$ ]] || fail "on this machine: '$out'"

if (( fails )); then echo "FAIL: $fails check(s) failed"; exit 1; fi
echo "PASS: ci/jobs.sh sizes jobs to memory, cgroup limits, overrides and the early check"
