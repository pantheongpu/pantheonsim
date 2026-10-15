#!/usr/bin/env bash
# The Makefile's front door, checked without building or running anything:
# every documented target exists and `make -n` shows the command it would run;
# command-line variables become the flags they promise and nothing else does;
# the failure paths say what to do; and scripts/doctor.sh reports a missing or
# old tool (against fake tools on a PATH of their own) with the right exit status.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
command -v make >/dev/null || { echo "SKIP: no make"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# A caller's own settings must not leak into the targets.
for v in $(env | sed -n 's/^\(VGPU_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
unset GPU COUNT VRAM_MB THREADS MAX_STEPS STRICT RACE COUNTERS QUIET TRACE PRELOAD CONFIG APP ARGS N ASK FORCE \
      CUDA_VERSION ROCM_VERSION DRIVER_VERSION SHELL_OS NICE VGPU

fail=0
ok() { echo "ok    $1"; }
bad() { echo "FAIL  $1"; shift; for l in "$@"; do echo "      $l"; done; fail=1; }
contains() {  # contains <name> <needle> <haystack>
  if [[ "$3" == *"$2"* ]]; then ok "$1"; else bad "$1" "expected to contain: $2" "actual: $3"; fi
}
lacks() {  # lacks <name> <needle> <haystack>
  if [[ "$3" != *"$2"* ]]; then ok "$1"; else bad "$1" "expected not to contain: $2" "actual: $3"; fi
}
fails() { if [[ $2 -ne 0 ]]; then ok "$1"; else bad "$1" "expected a non-zero exit"; fi; }   # make exits 2 when a recipe fails
expect() { if [[ "$3" == "$2" ]]; then ok "$1"; else bad "$1" "expected: $2" "actual:   $3"; fi; }

targets="doctor build demo list-gpus init config shell run test clean"

# ---- every target exists and make -n shows its command ---------------------------
for t in help $targets; do
  out=$(make -n "$t" APP=./x 2>&1); rc=$?
  if [[ $rc -eq 0 && -n $out ]]; then ok "make -n $t works"; else bad "make -n $t works" "exit $rc" "$out"; fi
done

out=$(make -s 2>&1); rc=$?
expect "plain make prints help and succeeds" "0" "$rc"
for t in $targets; do contains "help lists 'make $t'" "make $t" "$out"; done
contains "help documents the variables" "GPU=" "$out"
contains "help names the settings file" "pantheonsim.yaml" "$out"

# ---- each target runs the command it is documented to run ---------------------------
dry() { make -n "$@" 2>&1; }
# The vgpu command line of a dry run: not the checks in front of it, not the build, spaces squeezed.
vl() { grep -v -e '^test -x' -e '^if ' -e '^case ' -e '^ ' -e '^esac' -e 'scripts/' | tr -s ' ' ' '; }
contains "doctor runs scripts/doctor.sh" "./scripts/doctor.sh" "$(dry doctor)"
contains "build runs scripts/build.sh at low priority" "nice -n 10 ./scripts/build.sh" "$(dry build)"
out=$(dry build NICE=)
contains "NICE= overrides it" "./scripts/build.sh" "$out"
lacks "and drops nice" "nice" "$out"
contains "build can be re-niced" "nice -n 19 ./scripts/build.sh" "$(dry build 'NICE=nice -n 19')"
contains "test runs scripts/test.sh" "./scripts/test.sh" "$(dry test)"
contains "clean removes build/" "rm -rf build" "$(dry clean)"
contains "list-gpus runs vgpu list-gpus" "build/vgpu list-gpus" "$(dry list-gpus | vl)"
contains "init runs vgpu config init" "config init" "$(dry init)"
contains "init --force" "--force" "$(dry init FORCE=1)"
contains "config runs vgpu config show" "config show" "$(dry config)"
contains "shell runs vgpu shell without questions" "build/vgpu shell -y" "$(dry shell | vl)"
lacks "ASK=1 turns the answers back on" " -y" "$(dry shell ASK=1 | vl)"
contains "run runs the program" "build/vgpu run ./x" "$(dry run APP=./x | vl)"
contains "run passes ARGS to the program" "./x a b" "$(dry run APP=./x ARGS='a b' | vl)"
contains "demo runs the built-in kernel" "demo vectoradd" "$(dry demo)"

# demo and run build first when there is no build, and not when there is
out=$(dry demo VGPU=/nonexistent/vgpu)
contains "demo builds first when build/vgpu is missing" "scripts/build.sh" "$out"
contains "and then runs" "/nonexistent/vgpu demo vectoradd" "$(vl <<<"$out")"
contains "run builds first when build/vgpu is missing" "scripts/build.sh" "$(dry run APP=./x VGPU=/nonexistent/vgpu)"
lacks "demo does not build when it is there" "scripts/build.sh" "$(dry demo VGPU=/bin/true)"
lacks "run does not build when it is there" "scripts/build.sh" "$(dry run APP=./x VGPU=/bin/true)"

# ---- variables become flags, only from the make command line -----------------------------
out=$(dry run APP=./x GPU=nvidia/t4 COUNT=2 VRAM_MB=1024 THREADS=3 STRICT=1 RACE=1 COUNTERS=1 QUIET=1 PRELOAD=1 | vl)
for f in "--gpu 'nvidia/t4'" "--count '2'" "--vram-mb '1024'" "--threads '3'" "--strict" "--race" "--counters" "--quiet" "--preload"; do
  contains "run: $f" "$f" "$out"
done
lacks "STRICT=0 is not --strict" "--strict" "$(dry run APP=./x STRICT=0 | vl)"
contains "CONFIG= is --config, before the command" "--config 'rack.yaml' run" "$(dry run APP=./x CONFIG=rack.yaml | vl)"
contains "config show takes the same variables" "--count '4'" "$(dry config COUNT=4 | vl)"
contains "shell takes the version variables" "--cuda '12.4'" "$(dry shell CUDA_VERSION=12.4 | vl)"
contains "shell takes the OS" "--os 'rocky'" "$(dry shell SHELL_OS=rocky | vl)"
contains "demo takes GPU and N" "--gpu 'amd/mi300x' -n '100'" "$(dry demo GPU=amd/mi300x N=100 | vl)"
out=$(GPU=zzz COUNT=9 CUDA_VERSION=11 STRICT=1 make -n run APP=./x 2>&1 | vl; GPU=zzz CUDA_VERSION=11 make -n shell 2>&1 | vl)
lacks "a COUNT in the environment does nothing" "--count" "$out"
lacks "a GPU in the environment does nothing" "zzz" "$out"
lacks "a CUDA_VERSION in the environment does nothing" "--cuda" "$out"
lacks "a STRICT in the environment does nothing" "--strict" "$out"

# ---- the failure paths (these run for real; each stops before anything is built) --------------
out=$(make run NICE=false 2>&1); rc=$?
fails "run without APP fails" "$rc"
contains "and says what to write" "APP is empty" "$out"
out=$(make run APP=./no-such-program NICE=false 2>&1); rc=$?
fails "run with a missing program fails" "$rc"
contains "and names it" "'./no-such-program' does not exist" "$out"
out=$(make run APP=no-such-command-anywhere NICE=false 2>&1); rc=$?
fails "run with an unknown command fails" "$rc"
contains "and says it is not on PATH" "is not a command on PATH" "$out"
out=$(make list-gpus VGPU=/nonexistent/vgpu 2>&1); rc=$?
fails "a target that needs the build fails without it" "$rc"
contains "and offers make build" "make build" "$out"

# ---- scripts/doctor.sh, against fake tools --------------------------------------------------
bash -n scripts/doctor.sh && ok "doctor.sh parses" || bad "doctor.sh parses"
fake() {  # fake <dir> <name> <output>: a tool that prints <output> for any arguments
  mkdir -p "$1"; printf '#!/bin/sh\necho "%s"\n' "$3" > "$1/$2"; chmod +x "$1/$2"
}
doctor() { env -i PATH="$1" /bin/bash "$root/scripts/doctor.sh" 2>&1; }

p="$tmp/none"; mkdir -p "$p"
out=$(doctor "$p"); rc=$?
expect "doctor: nothing installed is a failure" "1" "$rc"
contains "doctor: names cmake" "cmake not found" "$out"
contains "doctor: says what to install" "apt-get install cmake" "$out"
contains "doctor: names the compiler" "needs g++ 13+ or clang++ 17+" "$out"

p="$tmp/oldcmake"; fake "$p" cmake "cmake version 3.19.8"; fake "$p" g++ "13.2.0"
out=$(doctor "$p"); rc=$?
expect "doctor: cmake 3.19 is too old" "1" "$rc"
contains "doctor: says so" "cmake 3.19 is too old" "$out"

p="$tmp/oldgcc"; fake "$p" cmake "cmake version 3.28.3"; fake "$p" g++ "12.3.0"
out=$(doctor "$p"); rc=$?
expect "doctor: g++ 12 is too old" "1" "$rc"
contains "doctor: says what it found" "g++ 12, needs 13 or newer" "$out"

p="$tmp/clang"; fake "$p" cmake "cmake version 3.20.0"; fake "$p" clang++ "Ubuntu clang version 17.0.6 (1ubuntu1)"
out=$(doctor "$p"); rc=$?
expect "doctor: cmake 3.20 and clang++ 17 are enough" "0" "$rc"
contains "doctor: reports the compiler" "clang++ 17" "$out"
contains "doctor: nvcc and hipcc are optional" "nvcc not found" "$out"

p="$tmp/full"; fake "$p" cmake "cmake version 3.28.3"; fake "$p" g++ "13.3.0"
fake "$p" nvcc "Cuda compilation tools, release 12.4, V12.4.131"; fake "$p" hipcc "HIP version: 6.4.43484-123eb5128"
out=$(doctor "$p"); rc=$?
expect "doctor: a full toolchain passes" "0" "$rc"
contains "doctor: reports nvcc" "nvcc 12.4" "$out"
contains "doctor: reports hipcc" "hipcc 6.4.43484" "$out"
contains "doctor: reports memory" "memory:" "$out"

exit $fail
