#!/usr/bin/env bash
# With no `unshare`, vgpu shell must not touch the host's mounts.
#
# It used to print "namespaces unavailable; continuing without /proc isolation"
# and then continue *with* it: `mount --make-rprivate /` and three bind mounts
# over /etc/os-release, /proc/driver and /sys/class/drm, in the host's own
# namespace. Unprivileged, those fail and nobody notices. As root on a machine
# without unshare -- and the simulator is published for people to run on their
# own machines -- they succeed. This runs the shell with unshare hidden and a
# stand-in `mount` that records what it was asked to do, so the test never
# needs root and never mounts anything.
set -uo pipefail
build="$(cd "${VGPU_BUILD_DIR:-build}" && pwd)"
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
fb="$work/bin"; mkdir -p "$fb"
printf '#!/bin/sh\necho "mount $*" >> "%s/calls"\nexit 1\n' "$work" > "$fb/mount"
chmod +x "$fb/mount"
for t in bash sh rm cat mkdir ln cp chmod dirname readlink env id uname hostname ls grep sed; do
  p=$(command -v "$t") && ln -sf "$p" "$fb/$t"
done
[[ ! -e "$fb/unshare" ]] || { echo "setup error: unshare leaked onto PATH"; exit 1; }

out=$(env -i HOME="$HOME" PATH="$fb" VGPU_QUIET=1 "$build/vgpu" shell -y --gpu nvidia/h100 -c true 2>&1)
if ! grep -q "continuing without /proc isolation" <<< "$out"; then
  echo "FAIL: did not take the no-namespace path: $out"; exit 1
fi
if [[ -s "$work/calls" ]]; then
  echo "FAIL: mounted in the host namespace after saying it would not:"; sed 's/^/  /' "$work/calls"; exit 1
fi
echo "ok    no unshare -> no mount calls in the host namespace"
