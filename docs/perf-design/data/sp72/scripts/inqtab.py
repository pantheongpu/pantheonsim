#!/usr/bin/env python3
"""The inq tests with blocks off and on, from the lines inqab.sh writes ("<off|on><pass> <test> <wall s> <user s> <system s> rc=.. pass=.. out=<md5>").

  python3 -I inqtab.py <title> <inq-*.txt>...

For each test: the best wall and CPU (user + system) seconds of each side over the passes, on / off of the CPU seconds, and whether every
run passed with the same output checksum.
"""
import collections
import sys

title = sys.argv[1]
runs = collections.defaultdict(lambda: collections.defaultdict(list))
for path in sys.argv[2:]:
    for line in open(path):
        p = line.split()
        if len(p) < 8 or not (p[0].startswith("off") or p[0].startswith("on")):
            continue
        side = "off" if p[0].startswith("off") else "on"
        runs[p[1]][side].append((float(p[2]), float(p[3]) + float(p[4]), p[5], p[6], p[7]))
print("### %s\n" % title)
print("| test | off: wall s / CPU s (best of n) | on: wall s / CPU s | CPU on / off | runs | all pass, same output |")
print("|---|---:|---:|---:|---:|---|")
for t, by in runs.items():
    if "off" not in by or "on" not in by:
        continue
    bo = (min(r[0] for r in by["off"]), min(r[1] for r in by["off"]))
    bn = (min(r[0] for r in by["on"]), min(r[1] for r in by["on"]))
    allr = by["off"] + by["on"]
    ok = all(r[2] == "rc=0" and r[3] == "pass=1" for r in allr) and len({r[4] for r in allr}) == 1
    print("| %s | %.1f / %.1f | %.1f / %.1f | %.3f | %d + %d | %s |" % (t, bo[0], bo[1], bn[0], bn[1], bn[1] / bo[1], len(by["off"]), len(by["on"]), "yes" if ok else "NO"))
print()
print("Every run:\n\n```")
for path in sys.argv[2:]:
    for line in open(path):
        if line.strip() and line.strip() != "DONE":
            print(line.rstrip())
print("```")
