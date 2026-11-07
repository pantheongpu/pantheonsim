#!/usr/bin/env bash
# Proof that nvidia/tests/e2e/fetch_cudnn_frontend.sh never hands out a half-unpacked tree: tests run side by side (ctest -j) and the first ones to
# ask used to find the directory while the first one was still unpacking into it (a header cut off in the middle: "unterminated #if" in
# e2e_dnn_fp8_attention_frontend). Many processes ask at once for a cache that does not exist yet, from a local archive of many files of a few
# thousand bytes (no network); each checks every file of what it was given, and all must have got the same, whole tree.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
script="${FETCH_SCRIPT:-$root/nvidia/tests/e2e/fetch_cudnn_frontend.sh}"
work="$(mktemp -d "${TMPDIR:-/tmp}/fetch-frontend.XXXXXX")"
trap 'rm -rf "$work"' EXIT
n=${FILES:-3000}; procs=${PROCS:-16}
src="$work/src/cudnn-frontend-1.30.0"
mkdir -p "$src/include/sub"
for ((i = 0; i < n; i++)); do head -c 4096 /dev/zero | tr '\0' "$((i % 9))" > "$src/include/sub/f$i.h"; done
echo '#define FRONTEND 1' > "$src/include/cudnn_frontend.h"; echo license > "$src/LICENSE.txt"
tar czf "$work/a.tar.gz" -C "$work/src" cudnn-frontend-1.30.0
sha="$(sha256sum "$work/a.tar.gz" | cut -d' ' -f1)"
for round in 1 2 3; do
  cache="$work/cache$round"; rm -rf "$work"/out*
  for ((p = 0; p < procs; p++)); do
    ( inc="$(VGPU_FRONTEND_URL="file://$work/a.tar.gz" VGPU_FRONTEND_SHA="$sha" VGPU_FRONTEND_CACHE="$cache" "$script" 2>"$work/err$p")" || { echo "process $p: script failed: $(cat "$work/err$p")" > "$work/out$p"; exit 0; }
      got=$(find "$inc/sub" -type f -size 4096c | wc -l)   # whole files only (a file being written is shorter)
      (( got == n )) && [[ -e "$inc/cudnn_frontend.h" ]] || echo "process $p: $got of $n files whole in $inc" > "$work/out$p" ) &
  done
  wait
  bad=$(cat "$work"/out* 2>/dev/null | head -3)
  [[ -z $bad ]] || { echo "FAIL (round $round): $bad"; exit 1; }
done
echo "ok: $procs processes asking at once, 3 rounds, every one got the whole tree"
