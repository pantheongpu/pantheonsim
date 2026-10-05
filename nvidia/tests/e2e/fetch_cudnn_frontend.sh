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
cache="${VGPU_FRONTEND_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/vgpu-cudnn-frontend}"
dir="$cache/cudnn-frontend-$version"
if [[ ! -e "$dir/include/cudnn_frontend.h" ]]; then
  mkdir -p "$cache"
  tmp="$(mktemp "$cache/fetch.XXXXXX")"
  trap 'rm -f "$tmp"' EXIT
  curl -fsSL --retry 3 -o "$tmp" "https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v$version.tar.gz" >&2 || exit 1
  echo "$sha  $tmp" | sha256sum -c --quiet - >&2 || { echo "cudnn-frontend $version: checksum mismatch" >&2; exit 1; }
  tar xzf "$tmp" -C "$cache" "cudnn-frontend-$version/include" "cudnn-frontend-$version/LICENSE.txt" >&2 || exit 1
fi
echo "$dir/include"
