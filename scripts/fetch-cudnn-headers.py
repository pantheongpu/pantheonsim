#!/usr/bin/env python3
"""Fetch cuDNN's public API headers into a directory.

The headers are NVIDIA's and are not kept in this repository. They are the ones
in the `nvidia-cudnn-cu12` wheel on PyPI; the wheel is large (hundreds of MB), so
this reads only the headers out of it with HTTP range requests (about 1 MB),
then checks each against scripts/cudnn-headers.sha256. Only the standard
library is used.

    scripts/fetch-cudnn-headers.py [DEST]        DEST defaults to build/cudnn_include
    scripts/fetch-cudnn-headers.py --write-manifest

Nothing is downloaded when DEST already holds every header with the right
hash. Exit status 0 means DEST is complete; anything else means it is not.
"""
import hashlib
import io
import json
import os
import platform
import sys
import time
import urllib.request
import zipfile

VERSION = "9.25.1.1"   # the wheel; its headers say 9.25.1
PREFIX = "nvidia/cudnn/include/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "scripts", "cudnn-headers.sha256")


def get(url, headers=None, method="GET"):
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=headers or {}, method=method)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read() if method == "GET" else dict(r.headers)
        except Exception as e:   # network errors are worth a retry
            last = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"{url}: {last}")


class RemoteFile(io.RawIOBase):
    """A read-only seekable view of a URL, one range request per read."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(get(url, method="HEAD")["Content-Length"])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = get(self.url, {"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def wheel_url():
    meta = json.loads(get(f"https://pypi.org/pypi/nvidia-cudnn-cu12/{VERSION}/json"))
    arch = {"x86_64": "x86_64", "AMD64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(platform.machine())
    for f in meta["urls"]:
        if "manylinux" in f["filename"] and f["filename"].endswith(f"{arch}.whl") and not f.get("yanked"):
            return f["url"]
    raise RuntimeError(f"no manylinux {arch} wheel of nvidia-cudnn-cu12 {VERSION} on PyPI")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_manifest():
    out = {}
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                h, name = line.split(None, 1)
                out[name] = h
    return out


def complete(dest, want):
    for name, h in want.items():
        p = os.path.join(dest, name)
        if not os.path.isfile(p) or sha256(open(p, "rb").read()) != h:
            return False
    return True


def main(argv):
    if argv[1:] == ["--write-manifest"]:
        want = None
        dest = None
    else:
        dest = argv[1] if len(argv) > 1 else os.path.join(ROOT, "build", "cudnn_include")
        want = read_manifest()
        if complete(dest, want):
            print(f"cuDNN headers already in {dest}")
            return 0
    z = zipfile.ZipFile(io.BufferedReader(RemoteFile(wheel_url()), buffer_size=1 << 16))
    names = sorted(n for n in z.namelist() if n.startswith(PREFIX) and n.endswith(".h") and "/" not in n[len(PREFIX):])
    if not names:
        raise RuntimeError("the wheel holds no cuDNN headers")
    got = {n[len(PREFIX):]: z.read(n) for n in names}
    if want is None:
        with open(MANIFEST, "w") as f:
            f.write(f"# SHA-256 of the headers in nvidia-cudnn-cu12 {VERSION} (nvidia/cudnn/include).\n"
                    "# The headers are NVIDIA's and are fetched, not kept here; see scripts/fetch-cudnn-headers.py.\n")
            for n in sorted(got):
                f.write(f"{sha256(got[n])}  {n}\n")
        print(f"wrote {MANIFEST} for {len(got)} headers")
        return 0
    for n, h in want.items():
        if n not in got or sha256(got[n]) != h:
            raise RuntimeError(f"{n}: the wheel's copy does not match scripts/cudnn-headers.sha256")
    os.makedirs(dest, exist_ok=True)
    for n in want:
        with open(os.path.join(dest, n), "wb") as f:
            f.write(got[n])
    print(f"fetched {len(want)} cuDNN headers ({VERSION}) into {dest}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as e:
        print(f"fetch-cudnn-headers: {e}", file=sys.stderr)
        sys.exit(1)
