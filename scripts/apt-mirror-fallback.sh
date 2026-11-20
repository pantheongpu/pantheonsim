#!/usr/bin/env bash
# GitHub-hosted Ubuntu runners point apt at a mirror list (mirror+file:/etc/apt/apt-mirrors.txt) whose first host is
# Azure's azure.archive.ubuntu.com. When that name does not resolve, every apt-get update/install in a job fails
# ("Temporary failure resolving 'azure.archive.ubuntu.com'", exit 100), whatever the job does. In that case, and only
# then, point apt at the first of a few full Ubuntu mirrors whose name does resolve (archive.ubuntu.com first, then
# mirrors outside ubuntu.com, in case the runner's resolver is failing for that whole domain, as it did on 2026-11-11);
# each serves every pocket, security included. On a healthy runner, in a container image with no mirror list, or on a
# machine whose mirror resolves, this changes nothing; if none of them resolves either, it changes nothing and says so.
#
# With --next it is for a mirror that resolves but fails (503, stalls): it moves every Ubuntu mirror entry (the
# image's mirror, security.ubuntu.com and any earlier choice) to the next candidate that resolves, after the one used
# last, and exits 1 when it has none left to try. scripts/apt-retry.sh calls it between attempts.
#
#   APT_ETC     the apt configuration directory (default /etc/apt; the test points it at a temporary one)
#   APT_STATE   where --next remembers how many candidates it has used (default /tmp/apt-mirror-index)
set -u
etc=${APT_ETC:-/etc/apt}
list=$etc/apt-mirrors.txt
# host/path of each candidate, in order of preference
candidates=(archive.ubuntu.com/ubuntu mirrors.edge.kernel.org/ubuntu mirror.math.princeton.edu/pub/ubuntu
            mirrors.xtom.com/ubuntu mirror.us.leaseweb.net/ubuntu)

rewrite() {  # every Ubuntu mirror entry (not other repositories) to http://$1/
  local f sudo
  for f in "$etc/sources.list" "$etc"/sources.list.d/*.sources "$etc"/sources.list.d/*.list; do
    [[ -f $f ]] || continue
    sudo=; [[ -w $f ]] || sudo=sudo
    $sudo sed -i -E "s#(mirror\\+file:$list|https?://(azure\\.archive|security|archive)\\.ubuntu\\.com/ubuntu/?|https?://($(IFS='|'; echo "${candidates[*]%%/*}" | sed 's/\./\\./g'))/[a-z/]*ubuntu/?)#http://$1/#g" "$f"
  done
}

if [[ ${1:-} == --next ]]; then
  state=${APT_STATE:-/tmp/apt-mirror-index}
  i=$(cat "$state" 2>/dev/null || echo 0)
  for ((i = i; i < ${#candidates[@]}; i++)); do
    c=${candidates[i]}
    getent hosts "${c%%/*}" >/dev/null 2>&1 || continue
    echo $((i + 1)) > "$state"
    echo "::warning::apt mirror failed; trying ${c%%/*}"
    rewrite "$c"
    exit 0
  done
  echo "::warning::no mirror left to try"
  exit 1
fi

[[ -r $list ]] || exit 0
host=$(head -n 1 "$list" | sed -E 's#^[a-z+]*://##; s#[/[:space:]].*$##')
[[ -n $host ]] || exit 0
getent hosts "$host" >/dev/null 2>&1 && exit 0

chosen=
for c in "${candidates[@]}"; do
  if getent hosts "${c%%/*}" >/dev/null 2>&1; then chosen=$c; break; fi
done
if [[ -n $chosen ]]; then
  for ((k = 0; k < ${#candidates[@]}; k++)); do [[ ${candidates[k]} == "$chosen" ]] && echo $((k + 1)) > "${APT_STATE:-/tmp/apt-mirror-index}"; done
else
  echo "::warning::$host does not resolve, and neither does any fallback mirror (${candidates[*]%%/*}); apt is left as it is"
  exit 0
fi
echo "::warning::$host does not resolve; apt uses ${chosen%%/*} instead"
rewrite "$chosen"
exit 0
