#!/usr/bin/env python3
"""One table from every machine's workload results.

Each machine in the workloads run writes a report directory holding
results.tsv (workload, result, seconds, detail) and machine.txt. This reads all
of them and prints a Markdown table with a row per workload and a column per
machine, failures first, so the page answers "what broke, and where" before
anything else.

    python3 tests/workloads/workload_matrix.py results/ > matrix.md
"""
import pathlib
import sys

MARK = {"PASS": "✅", "SKIP": "➖", "FAIL": "❌", "TIMEOUT": "⏱️", "MISSING": "🚫"}


def load(root):
    machines = []
    for results in sorted(pathlib.Path(root).glob("**/results.tsv")):
        label_file = results.with_name("machine.txt")
        label = label_file.read_text().strip() if label_file.exists() else results.parent.name
        rows = {}
        for line in results.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                rows[parts[0]] = (parts[1], parts[3] if len(parts) > 3 else "")
        machines.append((label, rows))
    return machines


def render(machines, expected=None):
    if not machines:
        return "No machine wrote results: every job failed before running a workload.\n"
    out = []
    names = sorted({w for _, rows in machines for w in rows})
    broken = [w for w in names if any(rows.get(w, ("MISSING",))[0] not in ("PASS", "SKIP") for _, rows in machines)]
    total_bad = sum(1 for _, rows in machines for r, _ in rows.values() if r not in ("PASS", "SKIP"))
    missing_machines = (expected or 0) - len(machines)

    out.append("## Pantheon workloads")
    out.append("")
    if total_bad == 0 and missing_machines <= 0:
        out.append(f"All {len(names)} workloads pass or are out of scope on all {len(machines)} machines.")
    else:
        out.append(f"{len(broken)} of {len(names)} workloads fail somewhere; {total_bad} failures across {len(machines)} machines.")
    if missing_machines > 0:
        out.append(f"{missing_machines} machine(s) wrote no results at all (see their jobs).")
    out.append("")
    out.append("✅ pass · ❌ fail · ⏱️ timeout · 🚫 did not build · ➖ out of scope")
    out.append("")
    out.append("| Workload | " + " | ".join(label for label, _ in machines) + " |")
    out.append("| --- |" + " :---: |" * len(machines))
    for w in broken + [w for w in names if w not in broken]:
        cells = [MARK.get(rows.get(w, ("MISSING", ""))[0], "?") for _, rows in machines]
        out.append(f"| {'**' + w + '**' if w in broken else w} | " + " | ".join(cells) + " |")
    if broken:
        out.append("")
        out.append("### Why")
        out.append("")
        for w in broken:
            for label, rows in machines:
                result, detail = rows.get(w, ("MISSING", "no result"))
                if result not in ("PASS", "SKIP"):
                    out.append(f"- `{w}` on {label}: {result.lower()}{': ' + detail if detail else ''}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    expected = int(sys.argv[2]) if len(sys.argv) > 2 else None
    sys.stdout.write(render(load(sys.argv[1]), expected))
