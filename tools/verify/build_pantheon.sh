#!/usr/bin/env bash
# Builds the pantheon GPU workloads (https://github.com/pantheongpu/pantheon) once, for tools/verify/run_corpora.sh.
#
#   tools/verify/build_pantheon.sh <pantheon checkout> <output dir>
#
# Both simulator builds run the same binaries. The sources are exported from the checkout's HEAD
# (git archive), so local edits and build directories there are not used. The workloads are built
# for sm_86 (the Makefile's fallback when no nvidia-smi answers; the corpora run on the RTX 3060
# profile) with a shared CUDA runtime. Every nvcc call takes the build lock, one at a time.
set -uo pipefail
src="${1:?pantheon checkout}"; out="${2:?output dir}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out/src" "$out/bin"; out="$(cd "$out" && pwd)"
git -C "$src" archive HEAD | tar -x -C "$out/src"
echo "pantheon $(git -C "$src" rev-parse --short HEAD)" > "$out/VERSION"
cd "$out/src"
# a real nvidia-smi on PATH (a driver, WSL) would pick its own architecture: put one that fails
# in front, so the Makefile takes its documented fallback, sm_86
mkdir -p "$out/nosmi"; printf '#!/bin/sh\nexit 1\n' > "$out/nosmi/nvidia-smi"; chmod +x "$out/nosmi/nvidia-smi"
PATH="$out/nosmi:$PATH" make -k -j1 PLATFORM=CUDA BUILD_DIR="$out/bin" COMPILER="$here/nvcc_locked.sh" > "$out/build.log" 2>&1
ls "$out/bin" | tr '\n' ' '; echo
