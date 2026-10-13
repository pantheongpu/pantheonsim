#!/usr/bin/env bash
# How many jobs to run at once, sized to the memory the machine has, and an
# early check that a heavy job fits at all.
#
# `make -j$(nproc)` assumes memory scales with CPUs. On a shared runner with 24
# CPUs and a memory cap that is false: one C++ compile takes 1 to 2 GB, one LTO
# or sanitizer link 3 to 6 GB, and two such jobs side by side exhaust the box
# (exit 137, and once a machine-wide out-of-memory that killed the host's
# PostgreSQL). This prints
#
#     min(nproc, floor(available memory / GiB per job)), at least 1
#
# where available memory is the lowest of
#   - /proc/meminfo MemAvailable,
#   - the headroom of this process's cgroup and of every cgroup above it
#     (cgroup v2: memory.max minus memory.current plus the inactive file cache,
#     which the kernel drops before it kills anything; "max" is unlimited;
#     cgroup v1: limit_in_bytes minus usage_in_bytes, the container's own),
#   - CI_MEM_LIMIT_GIB, when the runner's owner sets it as a ceiling.
#
# Usage
#   ci/jobs.sh                  jobs for the default 2 GiB per job
#   ci/jobs.sh -p PROFILE       jobs for a profile below (build, release, asan, ...)
#   ci/jobs.sh -g GIB           jobs for GIB GiB per job (a decimal is fine)
#   ci/jobs.sh -m N             never more than N (ctest -j2 stays at most 2)
#   ci/jobs.sh --check PROFILE  print the memory, fail unless the profile's
#                               minimum is available (exit 1)
#   ci/jobs.sh --available      print the available memory in GiB
#   ci/jobs.sh --profiles       print the table
#
# Environment
#   CI_JOBS            force this many jobs (what a person who knows better sets)
#   CI_JOBS_MAX        same as -m
#   CI_MEM_CHECK=0     make --check pass without looking
#   CI_MEM_LIMIT_GIB   ceiling on the memory this job may count on
#   CI_NPROC, CI_MEMINFO, CI_CGROUP_ROOT, CI_PROC_CGROUP
#                      where the numbers come from (/proc/meminfo and so on);
#                      tests/lint/check_ci_jobs.sh points them at synthetic files
#
# Profiles: GiB per parallel job, and the GiB that must be available before a
# heavy job starts. Both are estimates from what the steps do, not
# measurements. The minimum is one LTO or sanitizer link (about 4 GiB, the
# largest step that cannot be split) plus about 1 GiB for the tests' processes;
# the AMD ollama job builds only the HIP runtime and its CLI, so it needs less.
# They are kept low enough that a hosted runner (4 CPUs and 16 GB, or 2 CPUs
# and 7 GB) gets as many jobs as it has CPUs, as before, and passes the check.
#
#   profile     GiB/job  minimum  used for
#   default        2        0     anything not listed: nvcc and make builds of test programs
#   build          2        0     same, named
#   release        3        0     the simulator's Release build (it links with LTO)
#   sass           3        5     SASS on every architecture (release + one nvcc per arch)
#   ollama         3        5     ollama on every NVIDIA GPU
#   ollama-amd     3        4     ollama on one AMD GPU (HIP runtime only)
#   asan           3        5     ASan + UBSan build and tests
#   tsan           3        5     ThreadSanitizer build and tests
set -u

profile_row() {   # name -> "per-job-GiB minimum-GiB", or nothing if unknown
  case "$1" in
    default|build) echo "2 0" ;;
    release)       echo "3 0" ;;
    sass)          echo "3 5" ;;
    ollama)        echo "3 5" ;;
    ollama-amd)    echo "3 4" ;;
    asan)          echo "3 5" ;;
    tsan)          echo "3 5" ;;
  esac
}
profile_names="default build release sass ollama ollama-amd asan tsan"

die() { echo "ci/jobs.sh: $*" >&2; exit 2; }
is_num() { [[ ${1:-} =~ ^[0-9]+(\.[0-9]+)?$ ]]; }
is_pos_int() { [[ ${1:-} =~ ^[1-9][0-9]*$ ]]; }

meminfo=${CI_MEMINFO:-/proc/meminfo}
cg_root=${CI_CGROUP_ROOT:-/sys/fs/cgroup}
proc_cgroup=${CI_PROC_CGROUP:-/proc/self/cgroup}

first_line() { local l=""; [[ -r $1 ]] && IFS= read -r l < "$1"; printf '%s' "$l"; }
stat_kib() {   # file key -> the key's value in KiB (a /proc/meminfo line) or empty
  [[ -r $1 ]] || return 0
  local k v rest
  while read -r k v rest; do [[ $k == "$2:" ]] && { echo "$v"; return 0; }; done < "$1"
}
stat_bytes() { # file key -> the key's value, bytes (a memory.stat line) or empty
  [[ -r $1 ]] || return 0
  local k v
  while read -r k v; do [[ $k == "$2" ]] && { echo "$v"; return 0; }; done < "$1"
}

# Prints "kib source" for the lowest available memory, or "" when unknown.
available() {
  local best="" src="" v

  v=$(stat_kib "$meminfo" MemAvailable)
  if [[ -z $v ]]; then
    # An old kernel: free memory plus the cache.
    local f c; f=$(stat_kib "$meminfo" MemFree); c=$(stat_kib "$meminfo" Cached)
    [[ -n $f ]] && v=$(( f + ${c:-0} ))
  fi
  [[ -n $v ]] && { best=$v; src="MemAvailable"; }

  # cgroup v2: this process's cgroup and every one above it, up to the root.
  local path="" line rel dir lim cur inact head
  if [[ -r $proc_cgroup ]]; then
    while IFS= read -r line; do
      [[ $line == 0::* ]] && { path=${line#0::}; break; }
    done < "$proc_cgroup"
  fi
  rel=${path%%[[:space:]]*}
  [[ $rel == /* ]] || rel=/$rel
  dir=${cg_root%/}$rel
  dir=${dir%/}
  local guard=0
  while (( guard++ < 64 )); do
    lim=$(first_line "$dir/memory.max")
    if [[ $lim =~ ^[0-9]+$ ]]; then
      cur=$(first_line "$dir/memory.current")
      if [[ $cur =~ ^[0-9]+$ ]]; then
        inact=$(stat_bytes "$dir/memory.stat" inactive_file)
        head=$(( (lim - cur + ${inact:-0}) / 1024 ))
        (( head < 0 )) && head=0
        if [[ -z $best ]] || (( head < best )); then best=$head; src="cgroup headroom"; fi
      fi
    fi
    [[ $dir == "${cg_root%/}" || -z $dir ]] && break
    dir=${dir%/*}
    [[ $dir == "${cg_root%/}"* ]] || break
  done

  # cgroup v1 (the memory controller mounted at <root>/memory, or at the root).
  local d1
  for d1 in "$cg_root/memory" "$cg_root"; do
    lim=$(first_line "$d1/memory.limit_in_bytes")
    if [[ $lim =~ ^[0-9]+$ ]] && (( lim < (1 << 60) )); then
      cur=$(first_line "$d1/memory.usage_in_bytes")
      if [[ $cur =~ ^[0-9]+$ ]]; then
        inact=$(stat_bytes "$d1/memory.stat" total_inactive_file)
        head=$(( (lim - cur + ${inact:-0}) / 1024 ))
        (( head < 0 )) && head=0
        if [[ -z $best ]] || (( head < best )); then best=$head; src="cgroup v1 headroom"; fi
      fi
      break
    fi
  done

  if [[ -n ${CI_MEM_LIMIT_GIB:-} ]]; then
    is_num "$CI_MEM_LIMIT_GIB" || die "CI_MEM_LIMIT_GIB is not a number: $CI_MEM_LIMIT_GIB"
    v=$(awk -v g="$CI_MEM_LIMIT_GIB" 'BEGIN { printf "%d", g * 1048576 }')
    if [[ -z $best ]] || (( v < best )); then best=$v; src="CI_MEM_LIMIT_GIB"; fi
  fi

  [[ -n $best ]] && echo "$best $src"
}

gib() { awk -v k="$1" 'BEGIN { printf "%.1f", k / 1048576 }'; }

cpus() {
  if [[ -n ${CI_NPROC:-} ]]; then echo "$CI_NPROC"; return; fi
  nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
}

mode="jobs"; per=""; profile=""; max=${CI_JOBS_MAX:-}
while (( $# )); do
  case "$1" in
    -p|--profile)  profile=${2:-}; shift 2 || die "$1 needs a profile" ;;
    -g|--gib)      per=${2:-}; shift 2 || die "$1 needs a number" ;;
    -m|--max)      max=${2:-}; shift 2 || die "$1 needs a number" ;;
    --check)       mode=check; profile=${2:-}; shift 2 || die "--check needs a profile" ;;
    --available)   mode=available; shift ;;
    --profiles)    mode=profiles; shift ;;
    -h|--help)     sed -n '2,/^set -u/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    [0-9]*)        per=$1; shift ;;
    *)             die "unknown argument: $1" ;;
  esac
done

if [[ $mode == profiles ]]; then
  printf '%-11s %7s %8s\n' profile GiB/job minimum
  for n in $profile_names; do read -r p m <<<"$(profile_row "$n")"; printf '%-11s %7s %8s\n' "$n" "$p" "$m"; done
  exit 0
fi

pmin=0
if [[ -n $profile ]]; then
  row=$(profile_row "$profile"); [[ -n $row ]] || die "unknown profile '$profile' (one of: $profile_names)"
  read -r pper pmin <<<"$row"
  [[ -n $per ]] || per=$pper
fi
[[ -n $per ]] || per=2
if ! is_num "$per" || [[ $per =~ ^0+(\.0+)?$ ]]; then die "GiB per job must be a positive number: $per"; fi
[[ -z $max ]] || is_pos_int "$max" || die "the maximum must be a positive integer: $max"

av=$(available)
kib=${av%% *}; src=${av#* }

if [[ $mode == available ]]; then
  if [[ -z $av ]]; then echo unknown; exit 1; fi
  gib "$kib"; echo
  exit 0
fi

if [[ $mode == check ]]; then
  [[ -n $profile ]] || die "--check needs a profile (one of: $profile_names)"
  if [[ ${CI_MEM_CHECK:-1} == 0 ]]; then echo "memory check skipped (CI_MEM_CHECK=0)"; exit 0; fi
  if [[ -z $av ]]; then echo "memory: cannot tell how much is available; not checking"; exit 0; fi
  n=$("$0" -p "$profile")
  echo "memory: $(gib "$kib") GiB available (lowest of the limits: $src), $profile needs $pmin GiB; $(cpus) CPUs, $n parallel jobs at $per GiB each"
  need_kib=$(awk -v g="$pmin" 'BEGIN { printf "%d", g * 1048576 }')
  if (( kib < need_kib )); then
    msg="only $(gib "$kib") GiB of memory is available, and the $profile job needs $pmin GiB (ci/jobs.sh); it would take the machine's other jobs down with it, so it stops here. Re-run it when the runner is quieter, or set CI_MEM_CHECK=0 to start anyway."
    if [[ ${GITHUB_ACTIONS:-} == true ]]; then echo "::error::$msg"; else echo "error: $msg"; fi
    exit 1
  fi
  exit 0
fi

# jobs
if [[ -n ${CI_JOBS:-} ]]; then
  is_pos_int "$CI_JOBS" || die "CI_JOBS must be a positive integer: $CI_JOBS"
  echo "$CI_JOBS"; exit 0
fi
n=$(cpus)
is_pos_int "$n" || n=1
if [[ -n $av ]]; then
  per_kib=$(awk -v g="$per" 'BEGIN { printf "%d", g * 1048576 }')
  by_mem=$(( kib / per_kib ))
  (( by_mem < n )) && n=$by_mem
fi
[[ -n $max ]] && (( n > max )) && n=$max
(( n < 1 )) && n=1
echo "$n"
