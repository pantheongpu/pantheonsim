#!/usr/bin/env bash
# scripts/thp-always.sh sets transparent huge pages to `always` only on a GitHub-hosted runner whose setting is
# `madvise`, only when it can write it (directly, or through sudo that does not prompt), says what it did, and always
# exits 0; action.yml runs it in a step that cannot fail on it (#551).
set -u
here=$(cd "$(dirname "$0")/../.." && pwd)
script=$here/scripts/thp-always.sh
fail=0
check() { if [[ $2 == ok ]]; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

tmp=$(mktemp -d); trap 'chmod -R u+w "$tmp" 2>/dev/null; rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"
f=$tmp/enabled
setting() { printf '%s\n' "$1" > "$f"; }
# A sudo that runs its command (dropping -n), and one that would ask for a password.
printf '#!/bin/sh\n[ "$1" = -n ] && shift\nexec "$@"\n' > "$tmp/bin/sudo-ok"
printf '#!/bin/sh\necho "sudo: a password is required" >&2\nexit 1\n' > "$tmp/bin/sudo-no"
chmod +x "$tmp/bin/sudo-ok" "$tmp/bin/sudo-no"
run() { PATH="$tmp/bin:$PATH" THP_ENABLED="$f" "$@" "$script" 2>&1; }

# 1. A hosted runner as it ships, written directly (a root container job): set to always, and said.
setting 'always [madvise] never'
out=$(run env RUNNER_ENVIRONMENT=github-hosted SUDO=); rc=$?
check "madvise on a hosted runner exits 0" "$([[ $rc == 0 ]] && echo ok)"
check "it writes always" "$([[ $(cat "$f") == always ]] && echo ok)"
check "it says so" "$([[ $out == *"'madvise' -> 'always'"* ]] && echo ok)"

# 2. The same through a sudo that does not prompt.
setting 'always [madvise] never'
out=$(run env RUNNER_ENVIRONMENT=github-hosted SUDO=sudo-ok); rc=$?
check "through sudo it writes always" "$([[ $rc == 0 && $(cat "$f") == always && $out == *"-> 'always'"* ]] && echo ok)"

# 3. sudo that wants a password: nothing is written, nothing hangs, exit 0.
setting 'always [madvise] never'
out=$(run env RUNNER_ENVIRONMENT=github-hosted SUDO=sudo-no </dev/null); rc=$?
check "without passwordless sudo it changes nothing and exits 0" \
  "$([[ $rc == 0 && $(cat "$f") == 'always [madvise] never' && $out == *"no passwordless sudo"* ]] && echo ok)"

# 4. A setting it cannot write (a read-only /sys in a container). Root writes through file modes, so only as a user.
if [[ $(id -u) != 0 ]]; then
  setting 'always [madvise] never'; chmod a-w "$f"
  out=$(run env RUNNER_ENVIRONMENT=github-hosted SUDO=); rc=$?
  check "an unwritable setting is left, said, and exits 0" \
    "$([[ $rc == 0 && $(cat "$f") == 'always [madvise] never' && $out == *"could not write"* ]] && echo ok)"
  chmod u+w "$f"
fi

# 5. Already always, or never: left as it is.
for m in '[always] madvise never' 'always madvise [never]'; do
  setting "$m"
  out=$(run env RUNNER_ENVIRONMENT=github-hosted SUDO=sudo-ok); rc=$?
  check "'$m' is left as it is" "$([[ $rc == 0 && $(cat "$f") == "$m" && $out == *"left as it is"* ]] && echo ok)"
done

# 6. A self-hosted runner (or no RUNNER_ENVIRONMENT at all) is never touched.
for e in self-hosted ''; do
  setting 'always [madvise] never'
  out=$(run env RUNNER_ENVIRONMENT="$e" SUDO=sudo-ok); rc=$?
  check "RUNNER_ENVIRONMENT='$e' is left alone" \
    "$([[ $rc == 0 && $(cat "$f") == 'always [madvise] never' && $out == *"not a GitHub-hosted runner"* ]] && echo ok)"
done

# 7. No such file (a kernel without THP).
rm -f "$f"
out=$(run env RUNNER_ENVIRONMENT=github-hosted SUDO=sudo-ok); rc=$?
check "no setting file is a no-op" "$([[ $rc == 0 && ! -e $f ]] && echo ok)"

# 8. action.yml runs it, behind its input, in a step that cannot fail on it.
step=$(cd "$here" && python3 -I - <<'PY'
import re
s = open("action.yml").read()
ok = ("huge-pages:" in s
      and re.search(r'true\) \[ ! -x "\$\{GITHUB_ACTION_PATH\}/scripts/thp-always.sh" \] \|\| "\$\{GITHUB_ACTION_PATH\}/scripts/thp-always.sh" \|\| true ;;', s))
print("ok" if ok else "")
PY
)
check "action.yml runs it behind huge-pages and never fails on it" "$step"

exit $fail
