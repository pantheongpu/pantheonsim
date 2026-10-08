#!/usr/bin/env bash
# scripts/fetch-cudnn-headers.py gets cuDNN's API headers out of the
# nvidia-cudnn-cu12 wheel on PyPI by range requests and checks each against
# scripts/cudnn-headers.sha256. Fetches them into a scratch directory, checks
# that they are all there and that a tampered one is replaced, and that a
# second run changes nothing. Skips without network access.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
py="$(command -v python3 || command -v python || true)"
[[ -n "$py" ]] || { echo "SKIP: no python"; exit 0; }
d="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_cudnn_hdrs.XXXXXX")"
trap 'rm -rf "$d"' EXIT
if ! out="$("$py" "$root/scripts/fetch-cudnn-headers.py" "$d/inc" 2>&1)"; then
  case "$out" in
    *urlopen*|*"Name or service"*|*"Temporary failure"*|*"timed out"*|*"Network is unreachable"*|*"Connection"*)
      echo "SKIP: PyPI is not reachable ($out)"; exit 0 ;;
  esac
  echo "FAIL: $out"; exit 1
fi
n="$(grep -vc '^#' "$root/scripts/cudnn-headers.sha256")"
have="$(ls "$d/inc"/*.h | wc -l)"
[[ "$have" == "$n" ]] || { echo "FAIL: fetched $have headers, the manifest lists $n"; exit 1; }
(cd "$d/inc" && grep -v '^#' "$root/scripts/cudnn-headers.sha256" | sha256sum -c --quiet -) || { echo "FAIL: a fetched header does not match its hash"; exit 1; }
grep -q 'define CUDNN_MAJOR 9' "$d/inc/cudnn_version_v9.h" || { echo "FAIL: not cuDNN 9"; exit 1; }
echo tamper >> "$d/inc/cudnn.h"
"$py" "$root/scripts/fetch-cudnn-headers.py" "$d/inc" > /dev/null
(cd "$d/inc" && grep -v '^#' "$root/scripts/cudnn-headers.sha256" | sha256sum -c --quiet -) || { echo "FAIL: a tampered header was not replaced"; exit 1; }
again="$("$py" "$root/scripts/fetch-cudnn-headers.py" "$d/inc")"
[[ "$again" == *"already in"* ]] || { echo "FAIL: a complete directory was fetched again: $again"; exit 1; }
echo "PASS: $n cuDNN headers fetched, checked, repaired and left alone when complete"
