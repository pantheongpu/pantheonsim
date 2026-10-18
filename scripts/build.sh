#!/usr/bin/env bash
# Build VirtualGPU. Output lands in build/ (build/vgpu is the CLI).
set -euo pipefail
cd "$(dirname "$0")/.."
# On a person's machine use half the CPUs (and never more than memory allows, see
# ci/jobs.sh). CI_JOBS=N forces a count, CI_JOBS_MAX=N sets a cap, `make JOBS=N`
# is CI_JOBS=N. CI (CI=true) keeps the old behaviour: every CPU memory allows.
if [[ -z ${CI_JOBS:-} && -z ${CI_JOBS_MAX:-} && -z ${CI:-} ]]; then
  CI_JOBS_MAX=$(( ($(nproc) + 1) / 2 )); export CI_JOBS_MAX
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE="${VGPU_BUILD_TYPE:-Release}" >/dev/null
cmake --build build -j"$(ci/jobs.sh -p release)"
