#!/usr/bin/env bash
# Fetches LLNL's Varity (https://github.com/LLNL/Varity, MIT), the random floating-point program generator of Zahid, Laguna, Le,
# "Testing GPU Numerics" (arXiv 2410.09172), at a pinned commit. It is fetched when a run needs it and is not part of this repository.
#
#   tools/numerics/varity/fetch.sh <dir>
set -euo pipefail
dir="${1:?usage: $0 <dir>}"
commit=f5e04921d7d48ea6ba9f2d4d71ab8f1edbbadef9
if [[ -d "$dir/.git" ]] && [[ "$(git -C "$dir" rev-parse HEAD)" == "$commit" ]]; then echo "$dir"; exit 0; fi
rm -rf "$dir"
git clone -q https://github.com/LLNL/Varity.git "$dir"
git -C "$dir" checkout -q "$commit"
echo "$dir"
