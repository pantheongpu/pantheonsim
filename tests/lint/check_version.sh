#!/usr/bin/env bash
# The version is written once, in CMakeLists.txt's project(). This checks that the sources print the
# compiled-in version rather than a literal of their own.
# Checked without building or running anything.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
fail=0
ok()  { echo "ok    $1"; }
bad() { echo "FAIL  $1"; fail=1; }
cm=$(sed -n 's/^project(pantheonsim VERSION \([0-9][0-9.]*\) .*/\1/p' CMakeLists.txt | head -1)
[[ -n $cm ]] && ok "CMakeLists.txt has a version ($cm)" || bad "no version in CMakeLists.txt's project()"
if grep -nE 'kVersion *= *"[0-9]|VirtualGPU [0-9]+\.[0-9]+\.[0-9]+' src/cli/main.cpp src/cli/smi.cpp; then
  bad "src/cli prints a version literal; use VGPU_VERSION"
else ok "src/cli takes the version from VGPU_VERSION"; fi
grep -q 'VGPU_VERSION=' CMakeLists.txt && ok "CMakeLists.txt defines VGPU_VERSION for vgpu_cli" || bad "CMakeLists.txt does not define VGPU_VERSION"
exit $fail
