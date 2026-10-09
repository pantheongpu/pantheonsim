#!/usr/bin/env bash
# No NVIDIA proprietary file in the tree. Headers and sources that NVIDIA ships
# under its proprietary terms ("PROPRIETARY and CONFIDENTIAL ... reproduction or
# disclosure to any third party ... is prohibited", as in cuDNN's headers) are
# not ours to publish; the build fetches them instead (scripts/fetch-cudnn-headers.py).
# Files NVIDIA releases under an open licence (MIT, Apache-2.0, BSD) are fine and
# carry their licence at the top.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
# NVIDIA's header text wraps "PROPRIETARY and / CONFIDENTIAL" over two lines, so
# the heading "NOTICE TO LICENSEE" is what matches those files; this script names
# both phrases, so it is excluded from its own search.
bad="$(git grep -l -I -e 'PROPRIETARY and CONFIDENTIAL' -e 'NOTICE TO LICENSEE' -- . ':!tests/lint/check_no_proprietary_nvidia.sh' || true)"
if [[ -n "$bad" ]]; then
  echo "FAIL: files that carry NVIDIA's proprietary-licence notice:"
  echo "$bad" | sed 's/^/  /'
  echo "Fetch them at build time instead of keeping them in the repository."
  exit 1
fi
echo "PASS: no file carries NVIDIA's proprietary-licence notice"
