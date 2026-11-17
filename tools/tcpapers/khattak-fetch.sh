#!/usr/bin/env bash
# Fetches the validation vectors of Khattak and Mikaitis, "Accurate Models of NVIDIA Tensor Cores" (arXiv 2512.07004): the
# model_validation/ directory of github.com/north-numerical-computing/MATLAB-tensor-core (BSD-2-Clause) at the commit these scripts
# were written against, and converts it (khattak_convert.py) to <dir>/vectors/<card>.<fmt>.<out>.txt, which khattak_vectors.cu reads.
# The data (3 to 11 MB a card) is not in this repository; a sample of it is: tools/tcpapers/data/ (see its NOTICE).
#
#   tools/tcpapers/khattak-fetch.sh <dir>
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dir="${1:?usage: $0 <dir>}"
commit=79143bd29a4a71dc7c2ed47741b63572498b17f6
mkdir -p "$dir"
if [[ ! -d "$dir/MATLAB-tensor-core/.git" ]]; then git clone -q https://github.com/north-numerical-computing/MATLAB-tensor-core.git "$dir/MATLAB-tensor-core"; fi
git -C "$dir/MATLAB-tensor-core" checkout -q "$commit"
python3 -I "$here/khattak_convert.py" "$dir/MATLAB-tensor-core" "$dir/vectors"
echo "vectors in $dir/vectors"
