#!/usr/bin/env bash
# Moved: the benchmark harness is tools/bench/bench.sh (see docs/performance.md).
# This forwarder keeps the old entry point; it used to time only `vgpu demo vectoradd`.
exec "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/bench/bench.sh" "$@"
