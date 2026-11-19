#!/usr/bin/env python3
"""Documentation links and file references resolve.

Checks every Markdown file of the repository (README, ARCHITECTURE, CONTRIBUTING, TODO, docs/, nvidia/docs/, amd/README):
  1. each relative link [text](path) or [text](path#anchor) points at a file or directory that exists, and an anchor names a heading of
     the target (GitHub's slug rules, duplicates numbered -1, -2, ...) when the target is a Markdown file;
  2. each in-page link (#anchor) names a heading of the same file;
  3. each backtick-quoted or bare reference of the form docs/<path>.md, nvidia/docs/<path>.md, amd/<path>.md names an existing file.
Links inside fenced code blocks are not checked. Absolute URLs are not checked (no network). Files under third-party or generated trees
(build*, third_party, node_modules, .git) are skipped. Exit status 1 and one line per problem on failure. Python 3 standard library only.
Usage: check_doc_links.py [repo-root]   (default: the directory two levels above this file)
"""
import os, re, sys

root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SKIP = {".git", "node_modules", "third_party", "external", "vendor"}

def md_files():
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP and not x.startswith("build") and not x.startswith(".cache")]
        for f in files:
            if f.endswith(".md"):
                yield os.path.join(d, f)

def slug(h):
    parts = h.split("`")   # raw HTML tags are dropped by the renderer, but not inside code spans
    h = "`".join(re.sub(r"</?[a-zA-Z][^>]*>", "", x) if i % 2 == 0 else x for i, x in enumerate(parts))
    h = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", h)
    h = h.strip().lower().replace("`", "")
    h = re.sub(r"[^\w\- ]", "", h, flags=re.UNICODE)
    return h.replace(" ", "-")

_anchors = {}
def anchors(path):
    if path in _anchors:
        return _anchors[path]
    seen, out, fence = {}, set(), False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.lstrip().startswith(("```", "~~~")):
                fence = not fence
                continue
            if fence:
                continue
            m = re.match(r"^ {0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
            if m:
                s = slug(m.group(1))
                n = seen.get(s, 0)
                seen[s] = n + 1
                out.add(s if n == 0 else "%s-%d" % (s, n))
            for a in re.findall(r'<a (?:name|id)="([^"]+)"', line):
                out.add(a)
    _anchors[path] = out
    return out

LINK = re.compile(r"(?<!\!)\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
REF = re.compile(r"(?<![\w/.-])((?:nvidia/docs|amd/docs|docs|amd)/[\w./-]*?\.md)\b")
problems = []
for path in sorted(md_files()):
    rel = os.path.relpath(path, root)
    fence = False
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
            continue
        if fence:
            continue
        for target in LINK.findall(line):
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            p, _, frag = target.partition("#")
            dest = path if p == "" else os.path.normpath(os.path.join(os.path.dirname(path), p))
            if not os.path.exists(dest):
                problems.append("%s:%d: link target does not exist: %s" % (rel, n, target))
            elif frag and dest.endswith(".md") and os.path.isfile(dest) and frag.lower() not in anchors(dest):
                problems.append("%s:%d: no heading for anchor in %s" % (rel, n, target))
        for ref in REF.findall(line):
            cands = [os.path.join(root, ref), os.path.join(root, "nvidia", ref), os.path.join(root, "amd", ref),
                     os.path.normpath(os.path.join(os.path.dirname(path), ref))]
            if not any(os.path.exists(c) for c in cands):
                problems.append("%s:%d: referenced file does not exist: %s" % (rel, n, ref))
for p in problems:
    print("FAIL  " + p)
if problems:
    print("%d broken documentation reference(s)" % len(problems))
    sys.exit(1)
print("ok    every relative link, anchor and docs/*.md reference in the Markdown files resolves")
