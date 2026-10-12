#!/usr/bin/env bash
# Simulator benchmark harness: see tools/bench/bench.py (--help) and docs/performance.md.
#   bench.sh --build build                       time every workload on SASS and PTX
#   bench.sh --ab buildA buildB --runs 5         alternating A/B
#   bench.sh --check-counters --build build      instruction COUNTS against tools/bench/expected (no timing)
#   bench.sh mix --build build                   instruction-class mix per workload
exec python3 -I "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/bench.py" "$@"
