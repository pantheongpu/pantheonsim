#!/usr/bin/env bash
# `vgpu test --matrix` exit codes, which CI jobs gate on.
#
# A program that never ran used to pass: exit 127 on every profile is the same
# exit 127 everywhere, so a typo in the program name reported "identical" and
# exited 0. And "results differ" shared exit 1 with vgpu failing outright. Also
# pinned: every --gpus name is checked before anything runs -- a bad second name
# used to fail only after the whole program had run on the first.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_matrix.XXXXXX")"
trap 'rm -rf "$work"' EXIT

fail=0
status() {  # status <name> <expected-exit> -- <command...>
  local name="$1" want="$2"; shift 3
  "$@" > "$work/out" 2>&1
  local rc=$?
  if [[ $rc -eq $want ]]; then echo "ok    $name (exit $rc)"; else
    echo "FAIL  $name: exit $rc, expected $want"; sed 's/^/      /' "$work/out"; fail=1; fi
}
m=("$vgpu" test --matrix --gpus nvidia/t4,nvidia/a10)

status "the same output on every profile" 0 -- "${m[@]}" /bin/true
status "a difference has its own code" 3 -- "${m[@]}" sh -c 'echo "$VGPU_GPU"'
status "a program that does not exist did not run" 4 -- "${m[@]}" "$work/no-such-program"
grep -q "nothing to compare" "$work/out" || { echo "FAIL  the not-run message is missing"; fail=1; }
printf '#!/bin/sh\n' > "$work/not-executable"
status "a file that is not executable did not run" 4 -- "${m[@]}" "$work/not-executable"
status "a baseline exiting 127 did not run" 4 -- "${m[@]}" sh -c 'exit 127'
status "a baseline killed by a signal did not run" 4 -- "${m[@]}" sh -c 'kill -9 $$'
status "an ordinary failure on every profile is still identical" 0 -- "${m[@]}" sh -c 'exit 1'

status "an unknown profile name is an error" 1 -- \
  "$vgpu" test --matrix --gpus nvidia/t4,nvidia/bogus sh -c "touch '$work/ran'"
if [[ -e "$work/ran" ]]; then echo "FAIL  the program ran before the bad profile name was noticed"; fail=1
else echo "ok    nothing ran before the bad profile name was rejected"; fi

help=$("$vgpu" test --help)
if grep -q '^  3  results differ' <<< "$help" && grep -q '^  4  the program did not run' <<< "$help"; then
  echo "ok    --help documents exit codes 3 and 4"
else echo "FAIL  --help does not document exit codes 3 and 4"; fail=1; fi

exit $fail
