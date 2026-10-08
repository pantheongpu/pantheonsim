#!/usr/bin/env bash
# Fetch the public cuDNN API headers into DEST (default: nvidia/third_party/cudnn_include).
#
# The headers are NVIDIA's, under the NVIDIA SDK license that ships in the
# nvidia-cudnn-cu12 wheel, which does not allow redistributing them in source
# form. They are therefore not committed here: they are downloaded from the wheel
# and only used to compile libvgpucudnn against the real ABI.
#
# Usage: fetch-cudnn-headers.sh [DEST]
#   CUDNN_WHEEL_SPEC   pip requirement to download (default: nvidia-cudnn-cu12)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dest="${1:-$here/cudnn_include}"
spec="${CUDNN_WHEEL_SPEC:-nvidia-cudnn-cu12}"

if [[ -f "$dest/cudnn.h" ]]; then
  echo "cuDNN headers already present in $dest"
  exit 0
fi

py="${PYTHON:-python3}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading $spec (headers only are kept; the wheel is several hundred MB)..."
"$py" -m pip download --no-deps --quiet --dest "$tmp" "$spec"

mkdir -p "$dest"
"$py" - "$tmp" "$dest" <<'PY'
import glob, os, sys, zipfile

tmp, dest = sys.argv[1:3]
wheels = glob.glob(os.path.join(tmp, "*.whl"))
if len(wheels) != 1:
    sys.exit(f"expected one wheel in {tmp}, found {len(wheels)}")
n = 0
with zipfile.ZipFile(wheels[0]) as z:
    for name in z.namelist():
        if "/cudnn/include/" in name and name.endswith(".h"):
            with open(os.path.join(dest, os.path.basename(name)), "wb") as f:
                f.write(z.read(name))
            n += 1
if not os.path.exists(os.path.join(dest, "cudnn.h")):
    sys.exit("no cudnn.h in the wheel")
print(f"Extracted {n} cuDNN headers to {dest}")
PY
