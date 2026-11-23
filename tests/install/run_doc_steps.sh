#!/usr/bin/env bash
# Runs the install steps docs/install.md documents, and only those, in the order named:
#
#   tests/install/run_doc_steps.sh prereqs build cuda hello-cuda ...
#
# A step is a fenced block that follows a line `<!-- install-step: NAME -->`. A file is a fenced block that follows
# `<!-- install-file: NAME -->`; it is written to the current directory before the steps run (the documented
# hello-world programs). The steps run in one bash with `set -euo pipefail`, so an unset variable, a failing command
# or a missing file in a documented step fails the run, and one step sees the environment (PATH) of the one before.
# A machine that is a bare container has no sudo and is root: `sudo` then just runs its command, and apt-get goes
# through scripts/apt-retry.sh when the repository has it (a mirror's 503 is not a documentation bug).
#
#   --list   print the step names the document has, in order
#   --print  print the script that would run, and run nothing
#   DOC      the document (default docs/install.md)
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
doc=${DOC:-$root/docs/install.md}
mode=run
case ${1:-} in --list) mode=list; shift ;; --print) mode=print; shift ;; esac

# extract KIND NAME: the first fenced block after `<!-- install-KIND: NAME -->`
extract() {
  awk -v kind="$1" -v name="$2" '
    $0 == "<!-- install-" kind ": " name " -->" { want = 1; next }
    want && /^```/ { if (infence) exit; infence = 1; next }
    want && infence { print }
  ' "$doc"
}
names() { sed -n 's/^<!-- install-\(step\|file\): \(.*\) -->$/\1 \2/p' "$doc"; }

if [[ $mode == list ]]; then names; exit 0; fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
script=$work/steps.sh
{
  echo 'set -euo pipefail'
  echo "cd '$root'"
  # shellcheck disable=SC2016
  echo 'if (( $(id -u) == 0 )); then sudo() { "$@"; }; fi'
  echo "if [[ -x '$root/scripts/apt-retry.sh' ]]; then apt-get() { SUDO= '$root/scripts/apt-retry.sh' \"\$@\"; }; fi"
} > "$script"

for step in "$@"; do
  body=$(extract step "$step")
  if [[ -z $body ]]; then echo "FAIL  $doc has no install-step '$step'" >&2; exit 1; fi
  { echo "echo '::group::$step'"; echo "$body"; echo "echo '::endgroup::'"; } >> "$script"
done

# The documented programs, written next to the steps that compile them.
while read -r kind name; do
  [[ $kind == file ]] || continue
  extract file "$name" > "$root/$name"
  [[ -s $root/$name ]] || { echo "FAIL  install-file '$name' is empty" >&2; exit 1; }
done < <(names)

if [[ $mode == print ]]; then cat "$script"; exit 0; fi
exec bash "$script"
