#!/usr/bin/env bash
# Prints the include directory of NVIDIA's cudnn-frontend (MIT licensed,
# header-only), fetching the pinned release once into a cache directory. The
# graph-API tests build their graphs with it, as PyTorch and Transformer
# Engine do, so that what is checked is the operation sequence the frontend
# really emits. Exits nonzero (printing nothing) when it cannot be fetched.
#
#   CUDNN_FRONTEND_INCLUDE=<dir>  use an existing checkout instead
set -euo pipefail
if [[ -n "${CUDNN_FRONTEND_INCLUDE:-}" && -e "$CUDNN_FRONTEND_INCLUDE/cudnn_frontend.h" ]]; then
  echo "$CUDNN_FRONTEND_INCLUDE"; exit 0
fi
version=1.30.0
sha=2e6a28661b91c0f43c2b24809750b480a93975bcf949c9715e540a7ab8575b31
# The test suite overrides these two with a local archive (tests/lint/check_fetch_frontend.sh); nothing else sets them.
url="${VGPU_FRONTEND_URL:-https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v$version.tar.gz}"
sha="${VGPU_FRONTEND_SHA:-$sha}"
cache="${VGPU_FRONTEND_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/vgpu-cudnn-frontend}"
# The tree is unpacked beside its final place and renamed into it, so that a process that finds the directory finds all of it: tests run side by side
# (ctest -j) and the first ones to ask used to see a half-unpacked tree (a header cut off in the middle). A directory is only ever complete or absent;
# the name differs from the one of the unpack-in-place version so that a half-unpacked tree such as that left in a runner's cache is not taken for one.
dir="$cache/frontend-$version"
if [[ ! -e "$dir/include/cudnn_frontend.h" ]]; then
  mkdir -p "$cache"
  tmp="$(mktemp "$cache/fetch.XXXXXX")"
  stage="$(mktemp -d "$cache/stage.XXXXXX")"
  trap 'rm -rf "$tmp" "$stage"' EXIT
  curl -fsSL --http1.1 --retry 5 --retry-all-errors --retry-delay 3 -o "$tmp" "$url" >&2 || exit 1
  echo "$sha  $tmp" | sha256sum -c --quiet - >&2 || { echo "cudnn-frontend $version: checksum mismatch" >&2; exit 1; }
  tar xzf "$tmp" -C "$stage" "cudnn-frontend-$version/include" "cudnn-frontend-$version/LICENSE.txt" >&2 || exit 1
  # rename(2) of a directory onto an existing non-empty one fails: then another process got there first, and its tree is the one to use.
  mv -T "$stage/cudnn-frontend-$version" "$dir" 2>/dev/null || [[ -e "$dir/include/cudnn_frontend.h" ]] || exit 1
fi
echo "$dir/include"
