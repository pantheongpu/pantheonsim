#!/usr/bin/env bash
# Copies what a built simulator needs in order to run out of its build tree --
# vgpu, bin/, shim/ and the libraries the shim's links lead to -- into a
# directory of its own, stripped. docker/Dockerfile ships that directory.
#
#   scripts/package-build.sh <build-dir> <dest-dir>
#
# The layout stays the build tree's: vgpu finds its shim beside itself, and
# the shim's links point one level up.
set -euo pipefail
build="${1:?usage: package-build.sh <build-dir> <dest-dir>}"
dest="${2:?usage: package-build.sh <build-dir> <dest-dir>}"
[[ -x "$build/vgpu" && -d "$build/shim" && -d "$build/bin" ]] \
  || { echo "package-build: $build is not a built simulator (no vgpu, bin/ and shim/)" >&2; exit 1; }
build="$(cd "$build" && pwd)"
mkdir -p "$dest/shim"
cp -a "$build/vgpu" "$build/bin" "$dest/"
cp -a "$build"/shim/. "$dest/shim/"

# Each shim entry is a link into the build root, and that may be a link again.
# Copy the file at the end of the chain, and make the first hop's name a link
# to it.
for entry in "$build"/shim/*; do
  [[ -L "$entry" ]] || continue
  real="$(readlink -f "$entry")"
  [[ -f "$real" ]] || { echo "package-build: $entry leads nowhere" >&2; exit 1; }
  cp -an "$real" "$dest/"
  hop="$(basename "$(readlink "$entry")")"
  [[ "$hop" == "$(basename "$real")" ]] || ln -sfn "$(basename "$real")" "$dest/$hop"
done

# Nothing in the image is debugged with symbols, and they are most of the size.
strip --strip-unneeded "$dest/vgpu" 2>/dev/null || true
for f in "$dest"/*.so*; do
  [[ -f "$f" && ! -L "$f" ]] && { strip --strip-unneeded "$f" 2>/dev/null || true; }
done

# Every link must resolve inside the destination, or the image would run only
# where the build tree is.
broken=$(find "$dest" -xtype l)
[[ -z "$broken" ]] || { echo "package-build: links that lead nowhere:" >&2; echo "$broken" >&2; exit 1; }
du -sh "$dest" | awk '{print "package-build: " $1 " in '"$dest"'"}'
