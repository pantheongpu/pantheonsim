#!/usr/bin/env bash
# ./run end to end, on the real simulator: a settings file describes a machine, and
# `./run FILE -- command` runs the command inside it and returns its status. Where the
# system restricts user namespaces ./run falls back to the unisolated machine, so this
# passes there too; it is the same call a user makes after `make`.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
[[ -x "$build/vgpu" ]] || { echo "no vgpu at $build/vgpu"; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for v in $(env | sed -n 's/^\(VGPU_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
unset XDG_CONFIG_HOME
export HOME="$tmp/home"; mkdir -p "$HOME" "$tmp/tree" "$tmp/work"
# The launcher finds build/vgpu beside itself; point that at the build under test.
cp "$root/run" "$tmp/tree/run"; ln -s "$build" "$tmp/tree/build"
cd "$tmp/work"

fail=0
expect() { if [[ "$3" == "$2" ]]; then echo "ok    $1"; else echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi; }
contains() { if [[ "$3" == *"$2"* ]]; then echo "ok    $1"; else echo "FAIL  $1"; echo "      expected to contain: $2"; echo "      actual: $3"; fail=1; fi; }

printf 'gpu: nvidia/t4\ncount: 2\n' > ci.yaml
out=$(timeout 120 "$tmp/tree/run" ci.yaml -- nvidia-smi -L </dev/null 2>&1); rc=$?
expect "run FILE -- nvidia-smi -L succeeds" "0" "$rc"
expect "the machine has the two T4s the file asks for" "2" "$(grep -c '^GPU [0-9]*: .*T4' <<<"$out")"

out=$(timeout 120 "$tmp/tree/run" ci.yaml -- sh -c 'echo "$VGPU_DEVICE_COUNT $VGPU_GPU"' </dev/null 2>&1); rc=$?
expect "a command sees the file's settings" "2 nvidia/t4" "$(tail -1 <<<"$out")"

timeout 120 "$tmp/tree/run" ci.yaml -- sh -c 'exit 3' </dev/null >/dev/null 2>&1; rc=$?
expect "the command's exit status is the launcher's" "3" "$rc"

printf 'gpu: nvidia/t4\ntread: 4\n' > bad.yaml
out=$(timeout 120 "$tmp/tree/run" bad.yaml -- true </dev/null 2>&1); rc=$?
expect "a bad file stops before the machine starts" "1" "$rc"
contains "with the simulator's own message" "unknown key 'tread'; did you mean 'threads'?" "$out"

# no argument and no file: the starter is written and the machine starts from it
rm -f pantheonsim.yaml
out=$(timeout 120 "$tmp/tree/run" -- nvidia-smi -L </dev/null 2>&1); rc=$?
expect "no file: still runs" "0" "$rc"
[[ -f pantheonsim.yaml ]] && echo "ok    no file: a starter pantheonsim.yaml was written" || { echo "FAIL  no file: a starter pantheonsim.yaml was written"; fail=1; }
exit $fail
