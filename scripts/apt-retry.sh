#!/usr/bin/env bash
# apt-get that survives a bad mirror. Runs `apt-get <args>` with apt's own retries and a time limit; when it fails or
# stalls (a 503 from security.ubuntu.com, a hung unpack), it finishes any half-configured package, moves apt to the
# next full Ubuntu mirror (scripts/apt-mirror-fallback.sh --next), refreshes the package lists and tries again.
#
#   scripts/apt-retry.sh install -y -qq --no-install-recommends build-essential cmake
#   scripts/apt-retry.sh update -qq
#
#   APT_ATTEMPTS  tries before giving up (default 4: the image's mirror, then up to three others)
#   APT_TIMEOUT   seconds one try may take (default 600)
#   SUDO          prefix for the commands that need root (default: sudo unless already root)
set -u
here=$(cd "$(dirname "$0")" && pwd)
if [[ -n ${SUDO+x} ]]; then sudo=$SUDO; elif [[ $(id -u) == 0 ]]; then sudo=; else sudo=sudo; fi
attempts=${APT_ATTEMPTS:-4}
for ((n = 1; n <= attempts; n++)); do
  if $sudo env DEBIAN_FRONTEND=noninteractive INITRD=No timeout "${APT_TIMEOUT:-600}" \
       apt-get -o Acquire::Retries=3 -o DPkg::Lock::Timeout=300 \
       -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 "$@"; then
    exit 0
  fi
  [[ $n == "$attempts" ]] && break
  echo "::warning::apt-get $1 failed or timed out (attempt $n of $attempts); trying another mirror"
  $sudo dpkg --configure -a || true
  "$here/apt-mirror-fallback.sh" --next || echo "::warning::no other mirror; retrying the same one"
  if [[ $1 != update ]]; then
    $sudo env DEBIAN_FRONTEND=noninteractive timeout 300 apt-get -o Acquire::Retries=3 -o DPkg::Lock::Timeout=300 update -qq || true
  fi
done
echo "::error::apt-get $* did not succeed in $attempts attempts"
exit 1
