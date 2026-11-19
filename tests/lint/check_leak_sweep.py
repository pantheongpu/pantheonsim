#!/usr/bin/env python3
"""The leak sweep: every end-to-end test runs with VGPU_LEAK_CHECK=1 and a log of its own (CMakeLists.txt, the e2e_ tests), and this reads
the logs after the tests have run. A process that still held device memory, pinned memory, a descriptor or a thread when its machine went
away is a LEAK line (include/vgpu/leak_check.hpp).

Most test programs do not free what they allocate (the card gives it back with the context), so a leak line alone is not a defect. The
known list pins each such test to the figures it leaks, the largest over the test's processes: a test that leaks more than it is listed
for fails (a new leak, or a library that now keeps more), one that leaks less fails too (lower the figure: the list only shrinks), and a
test that is listed and no longer leaks fails as stale. A leak in the simulator is fixed, not listed.

usage: check_leak_sweep.py <log directory> <known list>
       check_leak_sweep.py --update <log directory>     prints the known-list lines for what the logs show
  the list: `<test> <resource>=<figure>[,...]  # <reason>` per line; blank lines and # lines are skipped. The resources are the names a
  LEAK line gives: device_bytes, device_allocs, pinned_allocs, fds, threads.
Exit 0: nothing new. Exit 1: a new or changed leak, or a stale entry. A test with no log (it skipped, or did not run) is not judged.
"""
import os
import re
import sys

LINE = re.compile(r"^vgpu-leak-check: (?P<body>.*?) exe=(?P<exe>\S+) -> (?P<verdict>clean|LEAK (?P<what>\S+))\s*$")
FIGURE = re.compile(r"(device_bytes|device_allocs|pinned_allocs|fds|threads)=\+?(\d+)")


def read_known(path):
    known = {}
    with open(path) as f:
        for n, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2 or "#" not in raw:
                sys.exit(f"{path}:{n}: want '<test> <resource>=<figure>,...  # <reason>'")
            figs = {}
            for item in parts[1].split(","):
                k, _, v = item.partition("=")
                if not v.isdigit():
                    sys.exit(f"{path}:{n}: '{item}' is not <resource>=<figure>")
                figs[k] = int(v)
            known[parts[0]] = figs
    return known


def read_logs(logs):
    """test -> {resource: largest figure over its processes}, and the set of tests with a log."""
    seen, judged, example = {}, set(), {}
    if not os.path.isdir(logs):
        return seen, judged, example
    for name in sorted(os.listdir(logs)):
        if not name.endswith(".log"):
            continue
        test = name[:-4]
        judged.add(test)
        with open(os.path.join(logs, name), errors="replace") as f:
            for raw in f:
                m = LINE.match(raw)
                if not m or not m.group("what"):
                    continue
                named = set(m.group("what").split(","))
                figs = {k: int(v) for k, v in FIGURE.findall(m.group("body")) if k in named}
                for k in named:
                    figs.setdefault(k, 0)
                    cur = seen.setdefault(test, {})
                    if figs[k] >= cur.get(k, -1):
                        cur[k] = figs[k]
                        example[(test, k)] = raw.strip()
    return seen, judged, example


def main(argv):
    if len(argv) == 3 and argv[1] == "--update":
        seen, _, _ = read_logs(argv[2])
        for test, figs in sorted(seen.items()):
            print(f"{test} {','.join(f'{k}={v}' for k, v in sorted(figs.items()))}  # the program does not free it before it exits")
        return 0
    if len(argv) != 3:
        sys.exit(__doc__)
    known = read_known(argv[2])
    seen, judged, example = read_logs(argv[1])
    failed = False
    for test, figs in sorted(seen.items()):
        listed = known.get(test, {})
        for k, v in sorted(figs.items()):
            if k not in listed or v > listed[k]:
                failed = True
                was = f"listed for {listed[k]}" if k in listed else "not listed"
                print(f"NEW LEAK in {test}: {k}={v}, {was}\n  {example[(test, k)]}")
            elif v < listed[k]:
                failed = True
                print(f"LEAKS LESS in {test}: {k}={v}, listed for {listed[k]}; lower it in {argv[2]}")
    for test, listed in sorted(known.items()):
        if test not in judged:
            continue
        for k in sorted(set(listed) - set(seen.get(test, {}))):
            failed = True
            print(f"STALE entry for {test}: {k} no longer leaks; remove it from {argv[2]}")
    print(f"leak sweep: {len(judged)} tests with a log, {len(seen)} with a leak line, {len(known)} known")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
