#!/usr/bin/env bash
# scripts/package-build.sh must leave every shim link resolving inside the package, including one that reaches its
# library through a directory of its own: libcudss.so.1 -> ../cudss1/libcudss.so.1.0.0 stopped the image build at
# "links that lead nowhere". A small fake build tree with the three shapes of link: into the build root, through
# a chain, and through a subdirectory.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
b="$tmp/build"
mkdir -p "$b/bin" "$b/shim" "$b/cudss1"
printf '#!/bin/sh\n' > "$b/vgpu"
chmod +x "$b/vgpu"
echo plain > "$b/libplain.so.1.2"
ln -s ../libplain.so.1.2 "$b/shim/libplain.so.1"
echo chain > "$b/libchain.so.3.0"
ln -s libchain.so.3.0 "$b/libchain.so.3"
ln -s ../libchain.so.3 "$b/shim/libchain.so"
echo sub > "$b/cudss1/libcudss.so.1.0.0"
ln -s ../cudss1/libcudss.so.1.0.0 "$b/shim/libcudss.so.1"
"$here/scripts/package-build.sh" "$b" "$tmp/out" > /dev/null
broken="$(find "$tmp/out" -xtype l)"
[[ -z "$broken" ]] || { echo "dangling links in the package:" >&2; echo "$broken" >&2; exit 1; }
for l in libplain.so.1 libchain.so libcudss.so.1; do
  [[ -e "$tmp/out/shim/$l" ]] || { echo "the package has no shim/$l" >&2; exit 1; }
done
echo "package-build: the shim's links resolve"
