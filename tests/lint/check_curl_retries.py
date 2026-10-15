#!/usr/bin/env python3
"""Every script that downloads a file with curl retries on any error.

`curl --retry N` alone retries only timeouts and 5xx answers. A dropped HTTP/2
stream ("curl: (92) HTTP/2 stream 1 was not closed cleanly: PROTOCOL_ERROR") or
a connection reset is not retried, and one such blip while the rocBLAS nightly
fetched ROCm failed two of its shards (2026-10-14). So a curl that writes to a
file (-o / -O / --output / --remote-name to something other than /dev/null) has
to carry --retry-all-errors. (--http1.1 and --retry go with it in this
repository, but only the flag that changes what is retried is required.)

usage: check_curl_retries.py        (scans every tracked *.sh)
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CURL = re.compile(r"(?:^|[\s;&|(`$])curl(?=\s)")
OUTPUT = re.compile(r"(?:^|\s)(?:-[A-Za-z]*[oO]|--output|--remote-name(?:-all)?)(?=\s|=|$)")


def downloads_without_retry(text):
    """(line number, command) for every file-writing curl lacking --retry-all-errors."""
    out = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        start = i
        cmd = lines[i]
        while cmd.rstrip().endswith("\\") and i + 1 < len(lines):   # a continued command is one command
            i += 1
            cmd = cmd.rstrip()[:-1] + " " + lines[i].strip()
        i += 1
        stripped = cmd.lstrip()
        if stripped.startswith("#") or not CURL.search(cmd):
            continue
        after = cmd[CURL.search(cmd).end():]
        if not OUTPUT.search(after):
            continue
        if re.search(r"(?:-o|--output)[\s=]+/dev/null", after):
            continue
        if "--retry-all-errors" in after:
            continue
        out.append((start + 1, " ".join(stripped.split())[:120]))
    return out


def main():
    tracked = subprocess.run(["git", "ls-files", "*.sh"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    bad = 0
    for name in tracked:
        for number, cmd in downloads_without_retry((ROOT / name).read_text(encoding="utf-8", errors="replace")):
            print(f"FAIL  {name}:{number}: a curl download without --retry-all-errors: {cmd}")
            bad += 1
    if bad:
        print("\nAdd `--http1.1 --retry 5 --retry-all-errors --retry-delay 3` to each.")
        return 1
    print(f"ok    {len(tracked)} shell scripts: every curl download retries on any error")
    return 0


if __name__ == "__main__":
    sys.exit(main())
