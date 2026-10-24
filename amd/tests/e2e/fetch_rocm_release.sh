#!/usr/bin/env bash
# Unpacks ROCm packages, and what they depend on from the same repository,
# from AMD's apt repository into a directory, with no install and no root:
#
#   fetch_rocm_release.sh DIR RELEASE PACKAGE...
#
# RELEASE is the repository's directory (7.2 for ROCm 7.2.0, 7.2.3 for 7.2.3).
# The result is DIR/opt/rocm-<release>, which the AMD tests find with
# VGPU_ROCM_PATH. Dependencies that are not in the repository (libnuma, libdrm,
# ...) are the system's: install them with apt first. The tests that compare
# rocprofv3's output with a Radeon card's want exactly ROCm 7.2.0, which is the
# 7.2 repository (rocprofv3 1.1.0, hipcc 7.2.26015).
set -euo pipefail
dir=$1 release=$2
shift 2
(( $# )) || { echo "usage: $0 DIR RELEASE PACKAGE..." >&2; exit 2; }
repo=https://repo.radeon.com/rocm/apt/$release
mkdir -p "$dir" "$dir.debs"
curl -fsSL --http1.1 --retry 5 --retry-all-errors --retry-delay 3 \
  -o "$dir.debs/Packages.gz" "$repo/dists/noble/main/binary-amd64/Packages.gz"
# One path per line: the files of every wanted package and of its dependencies.
python3 -I - "$dir.debs/Packages.gz" "$@" > "$dir.debs/files" <<'PY'
import gzip, re, sys
index = {}
for stanza in gzip.open(sys.argv[1], "rt").read().split("\n\n"):
    fields = dict(l.split(": ", 1) for l in stanza.split("\n") if ": " in l and not l.startswith(" "))
    if "Package" in fields:
        index[fields["Package"]] = fields
want, seen, order = list(sys.argv[2:]), set(), []
while want:
    name = want.pop()
    if name in seen:
        continue
    if name not in index:
        if name in sys.argv[2:]:
            sys.exit("no such package in the repository: " + name)
        continue
    seen.add(name)
    order.append(index[name]["Filename"])
    for dep in index[name].get("Depends", "").split(","):
        first = dep.split("|")[0].strip()
        want.append(re.split(r"[ (]", first)[0])
print("\n".join(order))
PY
while read -r f; do
  deb="$dir.debs/$(basename "$f")"
  [[ -s $deb ]] || curl -fsSL --http1.1 --retry 5 --retry-all-errors --retry-delay 3 -o "$deb" "https://repo.radeon.com/rocm/apt/$release/$f"
  dpkg-deb -x "$deb" "$dir"
  rm -f "$deb"
done < "$dir.debs/files"
rm -rf "$dir.debs"
ls -d "$dir"/opt/rocm-*
