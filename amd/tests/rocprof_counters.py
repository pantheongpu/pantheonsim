#!/usr/bin/env python3
"""Reduce `rocprofv3-avail info --pmc` to what the simulator and a real Radeon card can be compared on.

    rocprof_counters.py <info.txt> <gfx target> offers            every counter the agent offers, sorted
    rocprof_counters.py <info.txt> <gfx target> blocks <name>...  those counters' entries, as the tool prints them

An entry is the counter's name, description, block and expression; the line of dimensions is left out, because the
card's counters come one per shader engine and the simulator's interpreter does not say which engine ran a wave
(amd/docs/rocprofiler.md).
"""
import re
import sys


def sections(path, target):
    text = open(path, errors="replace").read()
    for s in re.split(r"(?m)^GPU:\d+\n", text)[1:]:
        head, _, body = s.partition("\n")
        if head.strip() == "Name:" + target:
            return [b for b in body.split("\n\n\n") if "Counter_Name" in b]
    sys.exit("rocprof_counters: no agent %s in %s" % (target, path))


def name(block):
    return re.search(r"Counter_Name\s*:\s*(\S+)", block).group(1)


def main():
    path, target, mode = sys.argv[1:4]
    blocks = sections(path, target)
    if mode == "offers":
        print("\n".join(sorted(name(b) for b in blocks)))
        return
    by = {name(b): b for b in blocks}
    for n in sys.argv[4:]:
        if n not in by:
            print("%s: not offered" % n)
            continue
        print("\n".join(l.rstrip() for l in by[n].strip().split("\n") if not l.startswith("Dimensions")))
        print()


if __name__ == "__main__":
    main()
