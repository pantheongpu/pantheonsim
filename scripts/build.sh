#!/usr/bin/env bash
# Build VirtualGPU. Output lands in build/ (build/vgpu is the CLI).
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -DCMAKE_BUILD_TYPE="${VGPU_BUILD_TYPE:-Release}" >/dev/null
cmake --build build -j"$(nproc)"
