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

targets="deps doctor build demo list-gpus init config shell run test clean"

# ---- every target exists and make -n shows its command ---------------------------
for t in help $targets; do
  out=$(make -n "$t" APP=./x 2>&1); rc=$?
  if [[ $rc -eq 0 && -n $out ]]; then ok "make -n $t works"; else bad "make -n $t works" "exit $rc" "$out"; fi
done

out=$(make -s help 2>&1); rc=$?
expect "make help prints help and succeeds" "0" "$rc"
for t in $targets; do contains "help lists 'make $t'" "make $t" "$out"; done
contains "help lists plain make" "make                 make deps" "$out"
contains "help documents the variables" "GPU=" "$out"
contains "help names the settings file" "pantheonsim.yaml" "$out"

# ---- plain `make`: tools, then build, then what to type next ----------------------------------
out=$(make -n 2>&1); rc=$?
expect "plain make -n works" "0" "$rc"
contains "plain make checks the tools first" "./scripts/deps.sh" "$out"
contains "then builds" "./scripts/build.sh" "$out"
contains "at low priority" "nice -n 10 ./scripts/build.sh" "$out"
deps_at=$(grep -n 'scripts/deps.sh$' <<<"$out" | head -1 | cut -d: -f1)
build_at=$(grep -n 'scripts/build.sh' <<<"$out" | head -1 | cut -d: -f1)
if [[ -n $deps_at && -n $build_at && $deps_at -lt $build_at ]]; then ok "deps comes before build"; else bad "deps comes before build" "$out"; fi
contains "and says what to type next" "./run" "$out"
lacks "plain make does not print the help" "PantheonSim: run CUDA" "$out"
contains "make deps runs scripts/deps.sh" "./scripts/deps.sh" "$(make -n deps)"
lacks "make deps does not build" "build.sh" "$(make -n deps)"
lacks "make build does not install" "deps.sh" "$(make -n build)"
expect "make build is exactly the build script at low priority" "nice -n 10 ./scripts/build.sh" "$(make -n build)"

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

# ---- scripts/deps.sh (make deps), against fake tools and fake package managers -----------------------
# Nothing is installed: apt-get, sudo and friends are fakes that log what they are asked, and
# the fake apt-get "installs" by dropping fake tools on the fake PATH.
bash -n scripts/deps.sh && ok "deps.sh parses" || bad "deps.sh parses"
depsenv() {  # depsenv <bin-dir> <os-release> [VAR=value ...]; stdin is /dev/null unless the caller redirects
  local bin=$1 osr=$2; shift 2
  env -i PATH="$bin" DEPS_OS_RELEASE="$osr" FAKE_BIN="$bin" FAKE_LOG="$tmp/apt.log" "$@" /bin/bash "$root/scripts/deps.sh" 2>&1
}
depsbin() {  # depsbin <name> [have-packages]: a PATH dir with the basics and fake apt tools
  local d="$tmp/deps-$1"; mkdir -p "$d"
  for t in awk sed cat chmod; do ln -sf "$(command -v $t)" "$d/$t"; done
  printf '#!/bin/sh\necho "apt-cache $*" >> "$FAKE_LOG"\nfor p in %s; do [ "$2" = "$p" ] && { echo "  Candidate: 9.9"; exit 0; }; done\necho "  Candidate: (none)"\n' "${2:-cmake g++-13}" > "$d/apt-cache"
  cat > "$d/apt-get" <<'APT'
#!/bin/sh
echo "apt-get $*" >> "$FAKE_LOG"
if [ "$1" = install ]; then
  for a in "$@"; do case $a in
    cmake) printf '#!/bin/sh\necho "cmake version 3.28.3"\n' > "$FAKE_BIN/cmake"; chmod +x "$FAKE_BIN/cmake" ;;
    g++*) printf '#!/bin/sh\necho 13.3.0\n' > "$FAKE_BIN/$a"; chmod +x "$FAKE_BIN/$a" ;;
  esac; done
fi
exit 0
APT
  printf '#!/bin/sh\necho "add-apt-repository $*" >> "$FAKE_LOG"\n' > "$d/add-apt-repository"
  printf '#!/bin/sh\necho "sudo $*" >> "$FAKE_LOG"\nexec "$@"\n' > "$d/sudo"
  chmod +x "$d"/apt-cache "$d"/apt-get "$d"/add-apt-repository "$d"/sudo
  echo "$d"
}
ubuntu24="$tmp/os-ubuntu24"; printf 'ID=ubuntu\nID_LIKE=debian\nVERSION_ID="24.04"\n' > "$ubuntu24"
ubuntu22="$tmp/os-ubuntu22"; printf 'ID=ubuntu\nID_LIKE=debian\nVERSION_ID="22.04"\n' > "$ubuntu22"
fedora="$tmp/os-fedora"; printf 'ID=fedora\nVERSION_ID="41"\n' > "$fedora"
installed() { grep -c '^apt-get install' "$tmp/apt.log" 2>/dev/null || true; }
: > "$tmp/apt.log"

p=$(depsbin have); fake "$p" cmake "cmake version 3.28.3"; fake "$p" c++ "13.3.0"
out=$(depsenv "$p" "$ubuntu24" </dev/null); rc=$?
expect "deps: nothing missing is fine" "0" "$rc"
contains "deps: says so" "Build dependencies are installed" "$out"
expect "deps: and installs nothing" "0" "$(installed)"
expect "deps --cxx: nothing to say when the default compiler is fine" "" "$(env -i PATH="$p" /bin/bash scripts/deps.sh --cxx)"
out=$(env -i PATH="$p" /bin/bash scripts/deps.sh --check </dev/null); rc=$?
expect "deps --check passes when nothing is missing" "0" "$rc"

p=$(depsbin oldcc); fake "$p" cmake "cmake version 3.28.3"; fake "$p" c++ "12.3.0"; fake "$p" g++-13 "13.2.0"
expect "deps --cxx: names a newer compiler when the default is too old" "g++-13" "$(env -i PATH="$p" /bin/bash scripts/deps.sh --cxx)"

# missing, no terminal, no YES: nothing is installed, the command is printed, and it does not wait
p=$(depsbin noyes); : > "$tmp/apt.log"
out=$(depsenv "$p" "$ubuntu24" </dev/null); rc=$?
expect "deps: missing tools without YES and without a terminal exit 1" "1" "$rc"
expect "deps: and install nothing" "0" "$(installed)"
contains "deps: and print the exact command" "apt-get install -y cmake g++-13" "$out"
contains "deps: and the way to say yes" "make deps YES=1" "$out"
out=$(env -i PATH="$p" DEPS_OS_RELEASE="$ubuntu24" FAKE_BIN="$p" FAKE_LOG="$tmp/apt.log" /bin/bash scripts/deps.sh --check </dev/null 2>&1); rc=$?
expect "deps --check reports missing tools as exit 1 and installs nothing" "1" "$rc"
expect "deps --check installs nothing" "0" "$(installed)"

# YES=1 installs (through sudo unless root) and the result is verified
p=$(depsbin yes); : > "$tmp/apt.log"
out=$(depsenv "$p" "$ubuntu24" YES=1 </dev/null); rc=$?
expect "deps: YES=1 installs and succeeds" "0" "$rc"
contains "deps: runs apt-get update" "apt-get update" "$(cat "$tmp/apt.log")"
contains "deps: installs cmake and the compiler" "apt-get install -y cmake g++-13" "$(cat "$tmp/apt.log")"
if [[ $EUID -ne 0 ]]; then contains "deps: not root: through sudo" "sudo apt-get install" "$(cat "$tmp/apt.log")"; fi
lacks "deps: no PPA on 24.04" "add-apt-repository" "$(cat "$tmp/apt.log")"

# Ubuntu 22.04: no g++ 13 in the archive, so a PPA is needed, which only YES=1 PPA=1 adds
p=$(depsbin jammy "cmake"); : > "$tmp/apt.log"
out=$(depsenv "$p" "$ubuntu22" YES=1 </dev/null); rc=$?
expect "deps: 22.04 with YES=1 alone does not add a PPA" "1" "$rc"
contains "deps: it prints the PPA instructions" "ppa:ubuntu-toolchain-r/test" "$out"
contains "deps: and the export" "export CXX=g++-13" "$out"
lacks "deps: and ran nothing" "apt-get" "$(cat "$tmp/apt.log")"
lacks "deps: no PPA was added" "add-apt-repository" "$(cat "$tmp/apt.log")"
out=$(depsenv "$p" "$ubuntu22" YES=1 PPA=1 </dev/null); rc=$?
expect "deps: YES=1 PPA=1 does it" "0" "$rc"
contains "deps: the PPA was added" "add-apt-repository -y ppa:ubuntu-toolchain-r/test" "$(cat "$tmp/apt.log")"
contains "deps: g++-13 was installed" "apt-get install -y cmake g++-13" "$(cat "$tmp/apt.log")"
p=$(depsbin jammy2 "cmake"); : > "$tmp/apt.log"
out=$(depsenv "$p" "$ubuntu22" PPA=1 </dev/null); rc=$?
expect "deps: PPA=1 without YES=1 still adds nothing" "1" "$rc"
lacks "deps: and runs nothing" "add-apt-repository" "$(cat "$tmp/apt.log")"

# other package managers: the command is printed, never run
p="$tmp/deps-dnf"; mkdir -p "$p"; for t in awk sed cat; do ln -sf "$(command -v $t)" "$p/$t"; done
printf '#!/bin/sh\necho "dnf $*" >> "$FAKE_LOG"\n' > "$p/dnf"; chmod +x "$p/dnf"; : > "$tmp/apt.log"
out=$(depsenv "$p" "$fedora" YES=1 </dev/null); rc=$?
expect "deps: dnf systems get the command, exit 1" "1" "$rc"
contains "deps: the dnf command" "dnf install -y cmake gcc-c++ make" "$out"
expect "deps: and dnf is never run" "" "$(cat "$tmp/apt.log")"

# the question, on a terminal: no is the default, y installs
if command -v python3 >/dev/null 2>&1; then
  ask() {  # ask <answer> <bin-dir>
    : > "$tmp/apt.log"
    python3 -I - "$1" "$2" "$ubuntu24" "$root/scripts/deps.sh" "$tmp/apt.log" <<'PY'
import os, pty, sys, select
answer, binp, osr, script, log = sys.argv[1:6]
pid, fd = pty.fork()
if pid == 0:
    os.execve("/bin/bash", ["bash", script], {"PATH": binp, "DEPS_OS_RELEASE": osr, "FAKE_BIN": binp, "FAKE_LOG": log})
out = b""
sent = False
while True:
    r, _, _ = select.select([fd], [], [], 20)
    if not r:
        break
    try:
        data = os.read(fd, 4096)
    except OSError:
        break
    if not data:
        break
    out += data
    if b"[y/N]" in out and not sent:
        os.write(fd, answer.encode()); sent = True
_, status = os.waitpid(pid, 0)
sys.stdout.write(out.decode(errors="replace"))
sys.exit(os.waitstatus_to_exitcode(status))
PY
  }
  p=$(depsbin tty1)
  out=$(ask $'\n' "$p"); rc=$?
  contains "deps: on a terminal it asks" "[y/N]" "$out"
  expect "deps: an empty answer is no" "0" "$(installed)"
  expect "deps: and exits 1" "1" "$rc"
  p=$(depsbin tty2)
  out=$(ask $'y\n' "$p"); rc=$?
  expect "deps: y installs" "1" "$(installed)"
  expect "deps: and succeeds" "0" "$rc"
else
  echo "SKIP  no python3; the on-a-terminal question is not checked"
fi


exit $fail
