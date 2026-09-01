#!/usr/bin/env bash
# Build (if needed) and run the full test suite. No GPU required.
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/build.sh
ctest --test-dir build --output-on-failure
