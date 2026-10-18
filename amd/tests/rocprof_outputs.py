#!/usr/bin/env python3
"""Reduce what rocprofv3 writes besides results.json (csv, otf2, pftrace, rocpd, --stats, --summary,
--sys-trace, --kernel-rename) to text that does not depend on time or the machine, so a real Radeon
card's files and the simulator's can be compared with diff.

    rocprof_outputs.py <label> <output directory> [<stdout log of the run>]

Kept: the files written (without the process id and the host's directory), the columns of every csv, the
function names and call counts of the stats files, the rocpd tables and their columns, the summary
table's columns and rows (name, domain, calls) and, for the output formats that are not text, that the
file is not empty. Left out: every time, duration, id and address, and what the simulator does not
record yet (amd/docs/rocprofiler.md): HSA calls, the runtime's own kernels (__amd_rocclr_*) and
allocations, and the otf2 location HSA takes.
"""
import csv
import glob
import os
import re
import sqlite3
import sys

HSA = re.compile(r"^hsa_|^HSA_|rocclr")


def files(root):
    out = []
    for dirpath, _, names in os.walk(root):
        for n in names:
            rel = os.path.relpath(os.path.join(dirpath, n), root)
            parts = rel.split(os.sep)[1:]          # the host's directory
            parts = [re.sub(r"^\d+_", "", x) for x in parts]
            out.append("/".join(parts))
    return sorted(out)


def gap(path):
    # hsa_api_*.csv: the HSA trace; results/5.*: otf2's location for the HSA thread
    return path.startswith("hsa_") or re.match(r"results/5\.", path) is not None


def main():
    label, root = sys.argv[1], sys.argv[2]
    log = sys.argv[3] if len(sys.argv) > 3 else None
    print("[%s files]" % label)
    kept = [f for f in files(root) if not gap(f)]
    print("\n".join(kept))
    for path in sorted(glob.glob(os.path.join(root, "*", "*"))):
        base = re.sub(r"^\d+_", "", os.path.basename(path))
        if gap(base) or os.path.isdir(path):
            continue
        if base.endswith(".csv"):
            rows = list(csv.reader(open(path)))
            print("[%s %s]" % (label, base))
            print("columns: " + ",".join(rows[0]))
            if base in ("hip_api_stats.csv", "kernel_stats.csv"):
                name, calls = rows[0].index("Name"), rows[0].index("Calls")
                print("\n".join(sorted("%s calls=%s" % (r[name], r[calls]) for r in rows[1:] if not HSA.search(r[name]))))
        elif base.endswith(".db"):
            c = sqlite3.connect(path)
            print("[%s %s]" % (label, base))
            for (t,) in c.execute("select name from sqlite_master where type='table' order by name"):
                cols = ",".join(r[1] for r in c.execute("pragma table_info(%s)" % t))
                print("%s: %s" % (re.sub(r"_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}$", "_<uuid>", t), cols))
        else:
            print("[%s %s] %s" % (label, base, "not empty" if os.path.getsize(path) else "EMPTY"))
    if log:
        print("[%s summary]" % label)
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in open(log, errors="replace")
                if l.strip().startswith("|") and not l.strip().startswith("|--")]
        for r in rows:
            if r[0] == "NAME":
                print("columns: " + " | ".join(r))
        print("\n".join(sorted(" | ".join(r[:3]) for r in rows
                               if r[0] != "NAME" and not HSA.search(r[0]) and r[1] != "HSA_API"
                               and not r[0].startswith("MEMORY_ALLOCATION"))))


if __name__ == "__main__":
    main()
