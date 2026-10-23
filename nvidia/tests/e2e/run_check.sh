#!/usr/bin/env bash
# `vgpu check`: the comparison of a program on the card against the simulator, and its exit codes.
#
# Runs on a machine without a GPU: --card names the card, and the "program" is a shell snippet. The native side runs it with the
# VGPU_* variables removed, the simulated side through `vgpu run --gpu nvidia/a10g`, which sets VGPU_GPU, so a snippet that prints
# $VGPU_GPU differs between the two on purpose. A real card is exercised by the card check job, not here.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_check.XXXXXX")"
trap 'rm -rf "$work"' EXIT

fail=0
status() {  # status <name> <expected-exit> -- <command...>
  local name="$1" want="$2"; shift 3
  "$@" > "$work/out" 2>&1
  local rc=$?
  if [[ $rc -eq $want ]]; then echo "ok    $name (exit $rc)"; else
    echo "FAIL  $name: exit $rc, expected $want"; sed 's/^/      /' "$work/out"; fail=1; fi
}
says() {  # says <name> <text>: the last run's output contains the text
  if grep -q -- "$2" "$work/out"; then echo "ok    $1"; else echo "FAIL  $1: no '$2' in"; sed 's/^/      /' "$work/out"; fail=1; fi
}
c=("$vgpu" check --card "NVIDIA A10G" --gpu nvidia/a10g)

status "the same output on both sides agrees" 0 -- "${c[@]}" sh -c 'echo answer 42'
says "it says so" "the simulator agrees with the card"
status "timing lines are ignored" 0 -- "${c[@]}" sh -c 'echo answer 42; echo "took $RANDOM s"'
says "the ignored lines are counted" "timing/blank line"
status "a difference has its own code" 3 -- "${c[@]}" sh -c 'echo "gpu=$VGPU_GPU"'
status "a difference shows with --show-diff" 3 -- "${c[@]}" --show-diff sh -c 'echo "gpu=$VGPU_GPU"'
says "the first differing pair is printed" "first difference"
status "different exit codes are a difference" 3 -- "${c[@]}" sh -c 'if [ -n "$VGPU_GPU" ]; then exit 1; fi'
status "a card that disagrees with itself is masked, not held against the simulator" 0 -- "${c[@]}" sh -c 'echo seed $RANDOM; echo answer 42'
says "the noise is reported" "not reproducible"
status "noise on the answer line is still compared" 0 -- "${c[@]}" --native-runs 1 sh -c 'echo answer 42'
rm -f "$work/cnt"
status "an unlined-up card is inconclusive" 5 -- "${c[@]}" sh -c 'n=$(cat "'"$work"'/cnt" 2>/dev/null || echo 0); echo $((n+1)) > "'"$work"'/cnt"; for i in $(seq 1 $((n*2+1))); do echo line$i; done'
status "a program that does not exist did not run" 4 -- "${c[@]}" "$work/no-such-program"
says "the message says there is nothing to compare" "nothing to compare"
status "a baseline exiting 127 did not run" 4 -- "${c[@]}" sh -c 'exit 127'
status "a baseline killed by a signal did not run" 4 -- "${c[@]}" sh -c 'kill -9 $$'
status "a timeout on the card did not run" 4 -- "${c[@]}" --timeout 1 sh -c 'sleep 30'

# A file the program writes.
status "a file that is the same on both sides agrees" 0 -- "${c[@]}" --file "$work/f.txt" sh -c "echo data > '$work/f.txt'"
status "a file that differs is a difference" 3 -- "${c[@]}" --file "$work/f.txt" sh -c "echo \"\$VGPU_GPU\" > '$work/f.txt'"

# Numbers.
status "numbers within the tolerance agree" 0 -- "${c[@]}" --tolerance 1e-3 sh -c 'if [ -n "$VGPU_GPU" ]; then echo v 1.0001; else echo v 1.0000; fi'
status "numbers outside the tolerance differ" 3 -- "${c[@]}" --tolerance 1e-6 sh -c 'if [ -n "$VGPU_GPU" ]; then echo v 1.0001; else echo v 1.0000; fi'
status "with no tolerance the last digit counts" 3 -- "${c[@]}" sh -c 'if [ -n "$VGPU_GPU" ]; then echo v 1.0001; else echo v 1.0000; fi'

# The profile has to be the card.
status "a profile that is not the card is refused" 6 -- "$vgpu" check --card "NVIDIA A10G" --gpu nvidia/h100 sh -c 'echo x'
status "--allow-cross compares anyway" 0 -- "$vgpu" check --card "NVIDIA A10G" --gpu nvidia/h100 --allow-cross sh -c 'echo x'
says "and warns" "WARNING"
status "a card with no profile asks for --gpu" 6 -- "$vgpu" check --card "NVIDIA Imaginary 9000" sh -c 'echo x'
status "the card is matched to its profile by name" 0 -- "$vgpu" check --card "NVIDIA A10G" sh -c 'echo x'

# Usage.
status "no program is a usage error" 2 -- "$vgpu" check --card "NVIDIA A10G"
status "an unknown option is a usage error" 2 -- "$vgpu" check --bogus sh -c 'echo x'
status "a bad count is a usage error" 2 -- "$vgpu" check --native-runs 0 sh -c 'echo x'
status "an unknown profile is an error" 1 -- "$vgpu" check --card "NVIDIA A10G" --gpu nvidia/bogus sh -c 'echo x'
status "--help works" 0 -- "$vgpu" check --help
help=$("$vgpu" check --help)
for code in 3 4 5 6; do
  if grep -q "^  $code  " <<< "$help"; then echo "ok    --help documents exit code $code"; else echo "FAIL  --help does not document exit code $code"; fail=1; fi
done
exit $fail
