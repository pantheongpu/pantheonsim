#!/usr/bin/env bash
# Fetches FTTN (LLNL/FTTN, MIT; Li, Li, Fang, Swirydowicz, Laguna, Gopalakrishnan, "FTTN: Feature-Targeted Testing for Numerical
# Properties of NVIDIA & AMD Matrix Accelerators", CCGrid 2024, arXiv 2403.00232) at the commit these scripts were written against,
# applies fttn.patch and adds motivating.cpp. FTTN's sources are not in this repository.
#
#   tools/fttn/fetch.sh <dir>        -> <dir>/FTTN (patched), and <dir>/rocWMMA for the AMD build (header only, tag rocm-6.4.4)
#
# fttn.patch changes no test: it lets M_TILES / N_TILES / K_TILES be set on the command line (the GEMM of the paper's motivating
# example is 8192 x 8192 x 8192; every element of D is the same sum, so motivating.cpp builds it with one 16 x 16 tile of outputs),
# and -DFTTN_FP16_ONLY leaves out the tf32 and f64 kernels, which mma.h does not have below sm_80 (for an sm_75 build of fp16in).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dir="${1:?usage: $0 <dir>}"
commit=e184c3b3f233eaa711dedcc7a6da335a4aa76fb2
mkdir -p "$dir"
if [[ ! -d "$dir/FTTN/.git" ]]; then git clone -q https://github.com/LLNL/FTTN.git "$dir/FTTN"; fi
git -C "$dir/FTTN" checkout -q "$commit"
git -C "$dir/FTTN" checkout -q -- .
patch -d "$dir/FTTN" -p1 < "$here/fttn.patch"
cp "$here/motivating.cpp" "$dir/FTTN/"
if [[ "${FTTN_NO_ROCWMMA:-0}" != 1 && ! -d "$dir/rocWMMA" ]]; then
  git clone -q --depth 1 -b "${ROCWMMA_TAG:-rocm-6.4.4}" https://github.com/ROCm/rocWMMA.git "$dir/rocWMMA"
fi
echo "FTTN $commit patched in $dir/FTTN"
