#!/usr/bin/env bash
# scripts/apt-mirror-fallback.sh rewrites apt's mirror list only when its first host does not resolve, and every
# hosted-runner "apt-get update" in the workflows and action.yml runs it first.
set -u
here=$(cd "$(dirname "$0")/../.." && pwd)
script=$here/scripts/apt-mirror-fallback.sh
fail=0
check() { if [[ $2 == ok ]]; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"
mk_etc() {
  rm -rf "$tmp/etc"; mkdir -p "$tmp/etc/sources.list.d"
  printf 'http://azure.archive.ubuntu.com/ubuntu/\tpriority:1\nhttp://archive.ubuntu.com/ubuntu/\tpriority:2\n' > "$tmp/etc/apt-mirrors.txt"
  printf 'Types: deb\nURIs: mirror+file:%s/etc/apt-mirrors.txt\nSuites: noble noble-updates\n\nTypes: deb\nURIs: mirror+file:%s/etc/apt-mirrors.txt\nSuites: noble-security\n' "$tmp" "$tmp" > "$tmp/etc/sources.list.d/ubuntu.sources"
  printf 'deb mirror+file:%s/etc/apt-mirrors.txt jammy main\n' "$tmp" > "$tmp/etc/sources.list"
}
fake_getent() {  # the hosts the fake resolver answers for (all others fail)
  printf '#!/bin/sh\necho "$2" >> "%s/asked"\nfor h in %s; do [ "$2" = "$h" ] && exit 0; done\nexit 2\n' "$tmp" "$*" > "$tmp/bin/getent"
  chmod +x "$tmp/bin/getent"
}
run() { PATH="$tmp/bin:$PATH" APT_ETC="$tmp/etc" "$script" 2>&1; }
uris() { cat "$tmp/etc/sources.list.d/ubuntu.sources" "$tmp/etc/sources.list"; }

# 1. Only the image's mirror host fails: archive.ubuntu.com is used.
mk_etc; rm -f "$tmp/asked"; fake_getent archive.ubuntu.com
out=$(run); rc=$?
check "an unresolvable mirror exits 0" "$([[ $rc == 0 ]] && echo ok)"
check "it asks about the first mirror host" "$([[ $(head -n1 "$tmp/asked") == azure.archive.ubuntu.com ]] && echo ok)"
check "it says which mirror it chose" "$([[ $out == *"apt uses archive.ubuntu.com instead"* ]] && echo ok)"
check "deb822 sources point at archive.ubuntu.com, security included" \
  "$([[ $(grep -c 'URIs: http://archive.ubuntu.com/ubuntu/' "$tmp/etc/sources.list.d/ubuntu.sources") == 2 ]] && echo ok)"
check "sources.list points at archive.ubuntu.com" "$(grep -q 'deb http://archive.ubuntu.com/ubuntu/ jammy' "$tmp/etc/sources.list" && echo ok)"
check "no mirror+file entry is left" "$(! grep -rq 'mirror+file:' "$tmp/etc" && echo ok)"
run >/dev/null
check "a second run is harmless" "$([[ $? == 0 ]] && echo ok)"

# 2. The whole ubuntu.com domain fails to resolve (the 2026-11-11 outage): a mirror outside it is used.
mk_etc; fake_getent mirrors.edge.kernel.org
out=$(run)
check "with ubuntu.com unresolvable, kernel.org's mirror is chosen" "$([[ $out == *"apt uses mirrors.edge.kernel.org instead"* ]] && echo ok)"
check "sources carry its path" "$([[ $(uris | grep -c 'http://mirrors.edge.kernel.org/ubuntu/') == 3 ]] && echo ok)"
mk_etc; fake_getent mirror.us.leaseweb.net
out=$(run)
check "the last candidate is reachable too" "$([[ $out == *"apt uses mirror.us.leaseweb.net instead"* ]] && echo ok)"

# 3. Nothing resolves: apt is left alone, and the log says so.
mk_etc; before=$(uris); fake_getent
out=$(run); rc=$?
check "when no mirror resolves it exits 0 and changes nothing" "$([[ $rc == 0 && $(uris) == "$before" ]] && echo ok)"
check "and says so" "$([[ $out == *"neither does any fallback mirror"* ]] && echo ok)"

# 4. The image's mirror resolves: nothing is touched, nothing is printed.
mk_etc; before=$(uris); fake_getent azure.archive.ubuntu.com archive.ubuntu.com
out=$(run)
check "a resolvable mirror changes nothing" "$([[ $(uris) == "$before" && -z $out ]] && echo ok)"

# 5. A container image has no mirror list.
rm -rf "$tmp/etc"; mkdir -p "$tmp/etc"; fake_getent
run >/dev/null
check "no mirror list (a container image) is a no-op" "$([[ $? == 0 ]] && echo ok)"

# 6. --next moves every Ubuntu mirror entry (security.ubuntu.com included) to the next candidate that resolves.
mk_etc; printf 'Types: deb\nURIs: http://security.ubuntu.com/ubuntu/\nSuites: noble-security\n' >> "$tmp/etc/sources.list.d/ubuntu.sources"
fake_getent azure.archive.ubuntu.com archive.ubuntu.com mirrors.edge.kernel.org mirror.us.leaseweb.net
export APT_STATE="$tmp/state"; rm -f "$APT_STATE"
out=$(PATH="$tmp/bin:$PATH" APT_ETC="$tmp/etc" "$script" --next 2>&1); rc=$?
check "--next uses the first candidate that resolves" "$([[ $rc == 0 && $out == *"trying archive.ubuntu.com"* ]] && echo ok)"
check "the security entry moves too" "$([[ $(uris | grep -c 'security.ubuntu.com') == 0 ]] && echo ok)"
out=$(PATH="$tmp/bin:$PATH" APT_ETC="$tmp/etc" "$script" --next 2>&1)
check "a second --next moves on to kernel.org's mirror" "$([[ $out == *"trying mirrors.edge.kernel.org"* && $(uris | grep -c 'http://mirrors.edge.kernel.org/ubuntu/') == 4 ]] && echo ok)"
out=$(PATH="$tmp/bin:$PATH" APT_ETC="$tmp/etc" "$script" --next 2>&1)   # princeton and xtom do not resolve
check "an unresolvable candidate is skipped" "$([[ $out == *"trying mirror.us.leaseweb.net"* ]] && echo ok)"
PATH="$tmp/bin:$PATH" APT_ETC="$tmp/etc" "$script" --next >/dev/null 2>&1
check "--next exits 1 when no candidate is left" "$([[ $? == 1 ]] && echo ok)"
unset APT_STATE

# 7. apt-retry.sh: a failing apt-get is retried on another mirror, and gives up after APT_ATTEMPTS.
retry=$here/scripts/apt-retry.sh
mk_etc; fake_getent azure.archive.ubuntu.com archive.ubuntu.com mirrors.edge.kernel.org
cp "$script" "$retry" "$tmp/"; chmod +x "$tmp/apt-retry.sh" "$tmp/apt-mirror-fallback.sh"
printf '#!/bin/sh\nshift; exec "$@"\n' > "$tmp/bin/timeout"
printf '#!/bin/sh\n[ "$1" = -E ] && shift; exec "$@"\n' > "$tmp/bin/sudo"
printf '#!/bin/sh\nexit 0\n' > "$tmp/bin/dpkg"
# fails until the sources point at kernel.org's mirror, as a 503 from the first two would
printf '#!/bin/sh\necho "apt-get $*" >> "%s/calls"\ncase "$*" in *update*) exit 0;; esac\ngrep -rq mirrors.edge.kernel.org "%s/etc" || exit 100\n' "$tmp" "$tmp" > "$tmp/bin/apt-get"
chmod +x "$tmp"/bin/*
rm -f "$tmp/calls" "$tmp/state"
out=$(PATH="$tmp/bin:$PATH" APT_ETC="$tmp/etc" APT_STATE="$tmp/state" SUDO= "$tmp/apt-retry.sh" install -y foo 2>&1); rc=$?
check "apt-retry succeeds once a later mirror works" "$([[ $rc == 0 ]] && echo ok)"
check "it installed three times (the image's mirror, archive, kernel.org)" "$([[ $(grep -c 'install -y foo' "$tmp/calls") == 3 ]] && echo ok)"
check "it refreshed the lists after switching" "$([[ $(grep -c 'update' "$tmp/calls") == 2 ]] && echo ok)"
check "every call carries apt's own retries" "$([[ $(grep -c 'Acquire::Retries=3' "$tmp/calls") == $(wc -l < "$tmp/calls") ]] && echo ok)"

rm -f "$tmp/calls" "$tmp/state"; mk_etc
out=$(PATH="$tmp/bin:$PATH" APT_ETC="$tmp/etc" APT_STATE="$tmp/state" APT_ATTEMPTS=2 SUDO= "$tmp/apt-retry.sh" install -y foo 2>&1); rc=$?
check "apt-retry gives up after APT_ATTEMPTS with exit 1 and an error" "$([[ $rc == 1 && $out == *"::error::apt-get install -y foo did not succeed in 2 attempts"* ]] && echo ok)"
printf '#!/bin/sh\nexit 0\n' > "$tmp/bin/apt-get"; rm -f "$tmp/calls"
PATH="$tmp/bin:$PATH" SUDO= "$tmp/apt-retry.sh" update -qq >/dev/null 2>&1
check "a working apt-get is run once and exits 0" "$([[ $? == 0 ]] && echo ok)"

# Every hosted-runner apt-get in the workflows and action.yml goes through apt-retry.sh (the self-hosted pool steps take a
# flock and are not hosted; cuda-toolkits.yml has its own retry; apt-get download is not an install).
missing=$(cd "$here" && python3 -I - <<'PY'
import glob, re
bad = []
for f in glob.glob(".github/workflows/*.yml") + ["action.yml"]:
    if f.endswith("cuda-toolkits.yml"):
        continue
    for i, l in enumerate(open(f).read().split("\n")):
        if re.match(r"^\s*(sudo |\$SUDO )?apt-get (update|install)\b", l):
            bad.append(f"{f}:{i + 1}")
print("\n".join(bad))
PY
)
check "every hosted apt-get update/install goes through apt-retry.sh${missing:+ (direct: $(echo $missing | tr '\n' ' '))}" "$([[ -z $missing ]] && echo ok)"

exit $fail
