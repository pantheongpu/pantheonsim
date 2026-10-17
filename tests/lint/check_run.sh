#!/usr/bin/env bash
# The ./run launcher, checked against a fake build/vgpu, a fake unshare and fake
# package tools, so nothing is built or installed: a starter file is written when
# there is none, a bad file stops with the simulator's own message, a good file
# reaches `vgpu --config FILE shell -y`, a system that restricts user namespaces
# gets --no-isolate and one calm line, `-- command` becomes `shell -c` with the
# command's exit status, and a missing build runs `make build` and never a package manager.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -x $root/run ]] || { echo "FAIL  no executable run at the repository root"; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

for v in $(env | sed -n 's/^\(VGPU_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
unset XDG_CONFIG_HOME
export HOME="$tmp/home"; mkdir -p "$HOME"

fail=0
ok() { echo "ok    $1"; }
bad() { echo "FAIL  $1"; shift; for l in "$@"; do echo "      $l"; done; fail=1; }
contains() { if [[ "$3" == *"$2"* ]]; then ok "$1"; else bad "$1" "expected to contain: $2" "actual: $3"; fi; }
lacks() { if [[ "$3" != *"$2"* ]]; then ok "$1"; else bad "$1" "expected not to contain: $2" "actual: $3"; fi; }
expect() { if [[ "$3" == "$2" ]]; then ok "$1"; else bad "$1" "expected: $2" "actual:   $3"; fi; }

# A copy of the launcher with a fake simulator beside it.
tree="$tmp/tree"; work="$tmp/work"; fake="$tmp/fake"
mkdir -p "$tree/build" "$work" "$fake"
cp "$root/run" "$tree/run"
log="$tmp/log"; apt_log="$tmp/apt.log"

# The fake vgpu: logs its arguments; `config show` answers from the file's gpu, count and
# cuda_version lines (or fails on a file that says BAD); `config init` writes a starter;
# `shell` exits with $FAKE_RC.
cat > "$tmp/vgpu.fake" <<'VGPU'
#!/bin/bash
echo "$*" >> "$FAKE_LOG"
cfg=""
if [[ ${1:-} == --config ]]; then cfg=$2; shift 2; fi
case ${1:-} in
  config)
    case ${2:-} in
      show)
        if grep -q BAD "$cfg" 2>/dev/null; then echo "vgpu: $cfg:1: unknown key 'tread'; did you mean 'threads'?" >&2; exit 1; fi
        gpu=$(sed -n 's/^gpu: *\([^ ]*\).*/\1/p' "$cfg"); count=$(sed -n 's/^count: *\([^ ]*\).*/\1/p' "$cfg")
        cuda=$(sed -n 's/^cuda_version: *\([^ ]*\).*/\1/p' "$cfg")
        tb="this build's"
        printf 'settings file: %s\nprecedence:    flag > environment > file > default\n\n' "$cfg"
        printf '%-14s  %-14s  %s\n' key value from
        printf '%-14s  %-14s  %s\n' gpu "${gpu:-nvidia/h100}" default count "${count:-1}" default \
               cuda_version "${cuda:-$tb}" default rocm_version "$tb" default
        ;;
      init) printf '# starter\ngpu: nvidia/h100\n' > "$3"; echo "wrote $3" ;;
    esac ;;
  shell) exit "${FAKE_RC:-0}" ;;
esac
VGPU
chmod +x "$tmp/vgpu.fake"
cp "$tmp/vgpu.fake" "$tree/build/vgpu"

cat > "$fake/unshare" <<'UNSHARE'
#!/bin/sh
echo "unshare $*" >> "$FAKE_LOG"
exit "${FAKE_UNSHARE_RC:-0}"
UNSHARE
for t in apt-get sudo apt dnf; do
  printf '#!/bin/sh\necho "%s $*" >> "$FAKE_APT_LOG"\nexit 1\n' "$t" > "$fake/$t"
done
chmod +x "$fake"/*
export FAKE_LOG="$log" FAKE_APT_LOG="$apt_log"
export PATH="$fake:$PATH"
cd "$work" || exit 1

run() { : > "$log"; "$tree/run" "$@" 2>&1 </dev/null; }

out=$(run --help); rc=$?
expect "--help succeeds" "0" "$rc"
contains "--help shows the usage" "Usage: ./run [config.yaml] [-- command ...]" "$out"
expect "--help starts nothing" "" "$(cat "$log")"
out=$(run --bogus); rc=$?
expect "an unknown option is a usage error" "2" "$rc"
out=$(run a.yaml b.yaml); rc=$?
expect "two settings files are a usage error" "2" "$rc"
out=$(run --); rc=$?
expect "nothing after -- is a usage error" "2" "$rc"
contains "and says what to write" "nothing to run after --" "$out"

# ---- no argument, no file: a starter is written, then the machine starts ----------------------
rm -f pantheonsim.yaml
out=$(run); rc=$?
expect "no argument: succeeds" "0" "$rc"
if [[ -f pantheonsim.yaml ]]; then ok "no argument: pantheonsim.yaml was written"; else bad "no argument: pantheonsim.yaml was written"; fi
contains "no argument: says so in one line" "wrote pantheonsim.yaml" "$out"
contains "no argument: the starter is written by vgpu config init" "config init pantheonsim.yaml" "$(cat "$log")"
contains "no argument: starts the shell on it, without questions" "--config pantheonsim.yaml shell -y" "$(cat "$log")"
out=$(run); rc=$?
lacks "a second run keeps the file and does not write again" "config init" "$(cat "$log")"

# ---- a file that is wrong stops with the simulator's message ----------------------------------
echo 'tread: 4 BAD' > bad.yaml
out=$(run bad.yaml); rc=$?
if [[ $rc -ne 0 ]]; then ok "a bad file fails"; else bad "a bad file fails"; fi
contains "a bad file: the simulator's message is shown" "unknown key 'tread'; did you mean 'threads'?" "$out"
lacks "a bad file: no machine is started" " shell" "$(cat "$log")"
out=$(run nope.yaml); rc=$?
if [[ $rc -ne 0 ]]; then ok "a missing file fails"; else bad "a missing file fails"; fi
contains "a missing file is named" "no such settings file: nope.yaml" "$out"

# ---- a good file --------------------------------------------------------------------------------
printf 'gpu: nvidia/h100\ncount: 8\ncuda_version: 13.0\n' > good.yaml
out=$(run good.yaml); rc=$?
expect "a good file succeeds" "0" "$rc"
contains "it is validated first" "--config good.yaml config show" "$(cat "$log")"
contains "then the machine starts from it" "--config good.yaml shell -y" "$(cat "$log")"
lacks "isolation is left on" "--no-isolate" "$(cat "$log")"
contains "the banner names the machine" "8 x nvidia/h100, CUDA 13.0" "$out"
contains "the banner offers nvidia-smi" "nvidia-smi" "$out"
contains "the banner offers a venv for pip, and curl and git" "python3 -m venv ~/v && ~/v/bin/pip install torch" "$out"
contains "the banner says how to leave" "exit" "$out"
banner=$(sed -n '/PantheonSim:/,/Leave with/p' <<<"$out" | wc -l)
if (( banner <= 5 )); then ok "the banner is at most 5 lines ($banner)"; else bad "the banner is at most 5 lines" "$banner lines"; fi
lacks "no namespace complaint when they work" "restricts user namespaces" "$out"

printf 'gpu: amd/mi300x\ncount: 2\n' > amd.yaml
out=$(run amd.yaml)
contains "an AMD machine is named" "2 x amd/mi300x" "$out"
contains "and offers rocm-smi" "rocm-smi" "$out"
contains "and hipcc" "hipcc" "$out"

# ---- compilers: said once, and only when both are missing ----------------------------------------
mkdir -p "$tmp/cc"; printf '#!/bin/sh\nexit 0\n' > "$tmp/cc/nvcc"; chmod +x "$tmp/cc/nvcc"
out=$(PATH="$tmp/cc:$PATH" run good.yaml)
lacks "with nvcc there, no compiler note" "nvcc/hipcc not found" "$out"
# A PATH of only the tools the launcher uses, so no nvcc or hipcc of the host's can be on it.
mkdir -p "$tmp/bare"
for t in bash sh env dirname awk sed grep cat wc make rm mkdir cp mktemp; do
  p=$(command -v "$t" 2>/dev/null) && ln -sf "$p" "$tmp/bare/$t"
done
out=$(PATH="$fake:$tmp/bare" run good.yaml)
contains "without nvcc or hipcc: one note" "nvcc/hipcc not found; prebuilt programs and Python work, see docs/toolchains.md" "$out"

# ---- restricted user namespaces --------------------------------------------------------------------
out=$(FAKE_UNSHARE_RC=1 run good.yaml); rc=$?
expect "restricted namespaces: still succeeds" "0" "$rc"
contains "restricted namespaces: the same unshare the shell uses is tried" "unshare -r -m -u true" "$(cat "$log")"
contains "restricted namespaces: --no-isolate is added" "shell -y --no-isolate" "$(cat "$log")"
contains "restricted namespaces: one calm line" "Your system restricts user namespaces, so the machine is not isolated; nvidia-smi, nvcc and your programs work the same" "$out"
lacks "restricted namespaces: it is not an error" "rror" "$out"

# ---- one command -------------------------------------------------------------------------------------
out=$(run good.yaml -- nvidia-smi -L); rc=$?
expect "-- command: succeeds" "0" "$rc"
contains "-- command: maps to shell -c" "--config good.yaml shell -y -c nvidia-smi -L" "$(cat "$log")"
lacks "-- command: no banner" "Leave with" "$out"
out=$(FAKE_RC=7 run good.yaml -- false); rc=$?
expect "-- command: exits with the command's status" "7" "$rc"
out=$(run good.yaml -- echo 'two words' "it's")
contains "-- command: arguments keep their quoting" "-c echo two\\ words it\\'s" "$(cat "$log")"
out=$(FAKE_UNSHARE_RC=1 run good.yaml -- nvidia-smi -L)
contains "-- command: --no-isolate goes before -c" "shell -y --no-isolate -c nvidia-smi -L" "$(cat "$log")"
out=$(run -- nvidia-smi -L)
contains "-- command with no file uses pantheonsim.yaml" "--config pantheonsim.yaml shell -y -c nvidia-smi -L" "$(cat "$log")"

# ---- a missing build -----------------------------------------------------------------------------------
if command -v make >/dev/null 2>&1; then
  rm -f "$tree/build/vgpu"; : > "$apt_log"
  # "building" makes the fake simulator appear, as the real build would.
  printf '.PHONY: build\nbuild:\n\t@echo building >> "$(FAKE_LOG)"; cp "%s" build/vgpu\n' "$tmp/vgpu.fake" > "$tree/Makefile"
  out=$(run good.yaml); rc=$?
  expect "no build: succeeds after building" "0" "$rc"
  contains "no build: says so" "not built yet" "$out"
  contains "no build: runs make build" "building" "$(cat "$log")"
  contains "no build: then starts the machine" "shell -y" "$(cat "$log")"
  expect "no build: no package manager is ever run" "" "$(cat "$apt_log")"
  rm -f "$tree/build/vgpu"
  printf '.PHONY: build\nbuild:\n\t@exit 1\n' > "$tree/Makefile"
  out=$(run good.yaml); rc=$?
  if [[ $rc -ne 0 ]]; then ok "a failed build fails"; else bad "a failed build fails"; fi
  contains "a failed build points at make doctor and make deps" "make doctor" "$out"
  expect "a failed build installs nothing either" "" "$(cat "$apt_log")"
else
  echo "SKIP  no make; the missing-build cases are not checked"
fi

if bash -n "$root/run"; then ok "run parses"; else bad "run parses"; fi
if command -v shellcheck >/dev/null 2>&1; then
  if shellcheck "$root/run" "$root/scripts/deps.sh"; then ok "shellcheck: run and scripts/deps.sh are clean"; else bad "shellcheck"; fi
fi
exit $fail
