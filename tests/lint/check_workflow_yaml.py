#!/usr/bin/env python3
"""Every workflow file is one GitHub accepts: no key appears twice in a mapping.

A YAML parser that builds a dictionary (Python's, and most editors') keeps the
last of two equal keys and says nothing, but GitHub refuses the whole file: a
workflow with a duplicated job (or duplicated step key) is "a workflow file
issue", it runs on every push as a failure with no jobs, and its schedule stops
firing. That happened to sass-ptx.yml when a merge left a second `digest:` job
in it (2026-10-13), and nothing in the repository's checks looked.

This reads every .github/workflows/*.yml and every .github/*/action.yml with a
small indentation-based scan (the standard library only: PyYAML is not in every
container) and fails on a key repeated in one mapping, naming the file and the
lines. Block scalars (`run: |`) and comments are skipped; a `- key:` list item
starts a mapping of its own.

usage: check_workflow_yaml.py [file ...]    (default: all of them)
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
KEY = re.compile(r"^(\s*)(- +)?(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z0-9_.\-/$@{}()* ]+?))\s*:(?:\s|$)")


def duplicates(text):
    """(line, key, first line) for every key repeated in one mapping."""
    found = []
    stack = []        # (indent, {key: line}) of the open mappings, outermost first
    block = None      # indent of the key whose `|` / `>` block scalar we are inside
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if block is not None:
            if indent > block:
                continue
            block = None
        if line.lstrip().startswith("#"):
            continue
        m = KEY.match(line)
        if not m:
            continue
        lead, dash, q1, q2, bare = m.groups()
        key = q1 or q2 or bare
        col = len(lead) + (len(dash) if dash else 0)   # where the key itself starts
        if dash:
            # A new list item: the mappings opened inside the previous item are over.
            while stack and stack[-1][0] >= len(lead) + 1:
                stack.pop()
        else:
            while stack and stack[-1][0] > col:
                stack.pop()
        if not stack or stack[-1][0] < col:
            stack.append((col, {}))
        seen = stack[-1][1]
        if key in seen:
            found.append((number, key, seen[key]))
        else:
            seen[key] = number
        rest = line[m.end():].strip()
        if rest[:1] in ("|", ">") and re.fullmatch(r"[|>][+\-0-9]*(\s*#.*)?", rest):
            block = col       # the scalar's lines are indented past the key that owns it
    return found


def main(argv):
    files = [pathlib.Path(a) for a in argv] or sorted(
        list((ROOT / ".github" / "workflows").glob("*.yml"))
        + list((ROOT / ".github").glob("*/action.yml")))
    bad = 0
    for path in files:
        for number, key, first in duplicates(path.read_text(encoding="utf-8")):
            print(f"FAIL  {path.relative_to(ROOT) if path.is_absolute() and ROOT in path.parents else path}:"
                  f"{number}: key '{key}' repeated (first at line {first}); GitHub rejects the file")
            bad += 1
    if bad:
        return 1
    print(f"ok    {len(files)} workflow files have no repeated key")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
