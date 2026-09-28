#!/usr/bin/env bash
# The simulator spreads a grid over as many host threads as the process can
# run at once -- its affinity mask, capped by its cgroup's CPU quota -- not
# over every CPU the host has. In a one-CPU container on a many-core host the
# host's count made the workers take turns being throttled (1.5x slower on
# Ollama). Checked three ways: as is, pinned to one CPU, and in a container
# given one CPU's quota while seeing all of them (where Docker can run).
set -uo pipefail
bin="$1"
fail=0
check() {  # check <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1 ($3)"; else echo "FAIL  $1: expected $2, got $3"; fail=1; fi
}

got=$("$bin")
visible=$(nproc)   # the affinity mask
if (( got >= 1 && got <= visible )); then echo "ok    as is: $got, within the $visible CPUs it may use"
else echo "FAIL  as is: $got, outside 1..$visible"; fail=1; fi

if command -v taskset >/dev/null; then
  check "pinned to one CPU" 1 "$(taskset -c 0 "$bin")"
else
  echo "SKIP  pinned to one CPU: no taskset"
fi

# A quota with every CPU visible: the case the affinity mask cannot see. The
# binary needs only libc and libstdc++, which ubuntu:24.04 has; it runs where
# it was built.
have_docker() {
  command -v docker >/dev/null && docker info >/dev/null 2>&1 || return 1
  docker image inspect ubuntu:24.04 >/dev/null 2>&1 || docker pull -q ubuntu:24.04 >/dev/null 2>&1
}
if (( $(nproc --all) > 2 )) && have_docker; then
  dir=$(cd "$(dirname "$bin")" && pwd)
  in_box() { docker run --rm --cpus "$1" -v "$dir:/b:ro" ubuntu:24.04 "/b/$(basename "$bin")" 2>&1; }
  q=$(in_box 1)
  if [[ "$q" =~ ^[0-9]+$ ]]; then
    check "in a container with a one-CPU quota" 1 "$q"
    check "and a quota of 1.5 CPUs rounds up" 2 "$(in_box 1.5)"
  else
    echo "SKIP  in a container with a CPU quota: the binary did not run there ($q)"
  fi
else
  echo "SKIP  in a container with a CPU quota: no usable Docker, or too few CPUs to tell"
fi
exit $fail
