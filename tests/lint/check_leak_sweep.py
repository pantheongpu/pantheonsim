#!/usr/bin/env python3
"""The leak sweep: every end-to-end test runs with VGPU_LEAK_CHECK=1 and its own log (CMakeLists.txt, the e2e_ tests), and this reads
the logs after the tests have run. A process that still held device memory, pinned memory, a descriptor or a thread when its machine went
away is a LEAK line (include/vgpu/leak_check.hpp).

A test may leak on purpose, or because its program does (a test that never frees what it allocates, a thread it leaves running). Those
are named in the known list with the resources and the reason, and the list is a ratchet: a leak that is not in it fails, a resource that
is, and a test that is in it and no longer leaks fails too, so the list only shrinks. A leak that is the simulator's is a bug to fix, not
a line to add.

usage: check_leak_sweep.py <log directory> <known list>
  the list: `<test> <resource>[,<resource>...]  # <reason>` per line; blank lines and # lines are skipped.
Exit 0: nothing new. Exit 1: a new leak or a stale entry. A test with no log (it skipped, or did not run) is not judged.
"""
import os
import re
import sys

LINE = re.compile(r"^vgpu-leak-check: .*? exe=(?P<exe>\S+) -> (?P<verdict>clean|LEAK (?P<what>\S+))\s*$")


def read_known(path):
    known = {}
    with open(path) as f:
        for n, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2 or "#" not in raw:
                sys.exit(f"{path}:{n}: want '<test> <resources>  # <reason>'")
            known[parts[0]] = set(parts[1].split(","))
    return known


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    logs, known_path = argv[1], argv[2]
    known = read_known(known_path)
    seen = {}      # test -> {resource: example line}
    judged = set()
    if os.path.isdir(logs):
        for name in sorted(os.listdir(logs)):
            if not name.endswith(".log"):
                continue
            test = name[:-4]
            judged.add(test)
            with open(os.path.join(logs, name), errors="replace") as f:
                for raw in f:
                    m = LINE.match(raw)
                    if not m:
                        continue
                    if m.group("what"):
                        for r in m.group("what").split(","):
                            seen.setdefault(test, {}).setdefault(r, raw.strip())
    failed = False
    for test, res in sorted(seen.items()):
        allowed = known.get(test, set())
        new = {r: l for r, l in res.items() if r not in allowed}
        for r, line in sorted(new.items()):
            failed = True
            print(f"NEW LEAK in {test}: {r}\n  {line}")
    for test, allowed in sorted(known.items()):
        if test not in judged:
            continue
        gone = allowed - set(seen.get(test, {}))
        if gone:
            failed = True
            print(f"STALE entry for {test}: {','.join(sorted(gone))} no longer leaks; remove it from {known_path}")
    leaky = sum(1 for t in seen if t in judged)
    print(f"leak sweep: {len(judged)} tests with a log, {leaky} with a leak line, {len(known)} known")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
