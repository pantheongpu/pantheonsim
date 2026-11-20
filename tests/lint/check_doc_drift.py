#!/usr/bin/env python3
"""Documented environment variables and flags exist in the source, and the source's user-facing ones are documented.

Three checks, no build, Python 3 standard library only:
  1. every VGPU_* / PSIM_* variable the user documentation names is used somewhere outside the Markdown files (a name ending in "_",
     such as VGPU_WORKLOAD_*, must be the start of one that is);
  2. every VGPU_* / PSIM_* variable the product reads (a string literal "VGPU_X" in C/C++ code outside tests, os.environ in Python,
     ${PSIM_X...} in scripts/ci-setup.sh) is named in the user documentation, unless tests/lint/doc_drift_internal.txt lists it with the
     reason it is a developer knob and not an interface;
  3. every `--flag` that docs/cli.md shows is a string in src/cli (or listed in the internal file as another tool's flag), and every
     flag of vgpu's own subcommands (src/cli, not the nvidia-smi, amd-smi, ncu and lspci look-alikes, which take their real tools'
     flags) is documented or listed.
"User documentation" is every Markdown file except docs/perf-design and docs/history, which record what was tried and name switches that
no longer exist. Usage: check_doc_drift.py [repo-root] | selftest
"""
import os, re, sys, tempfile

VAR = re.compile(r"\b((?:VGPU|PSIM)_[A-Z0-9_]*[A-Z0-9_])")
OWN_CLI = {"main.cpp", "run.cpp", "check.cpp", "config_cmd.cpp", "debug.cpp", "demo.cpp", "fault.cpp", "regs.cpp", "serve.cpp",
           "shell.cpp", "test.cpp"}
SKIPDIR = {".git", "third_party", "external", "node_modules", "vendor"}
SRC_EXT = (".cpp", ".cc", ".c", ".h", ".hpp", ".cu", ".py", ".sh", ".inc", ".txt", ".cmake", ".yml", ".yaml")

def walk(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIPDIR and not x.startswith("build") and not x.startswith(".cache")]
        for f in files:
            yield os.path.join(d, f)

def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()

def is_user_doc(rel):
    return rel.endswith(".md") and not rel.startswith(("docs/perf-design/", "docs/history/"))

def is_test(rel):
    return re.search(r"(^|/)tests?(/|$)", rel) is not None

def load_internal(root):
    out = {}
    p = os.path.join(root, "tests", "lint", "doc_drift_internal.txt")
    for n, line in enumerate(read(p).split("\n"), 1):
        if not line.strip() or line.startswith("#"):
            continue
        name, sep, why = line.partition(" | ")
        if not sep or len(why.strip()) < 12:
            raise SystemExit("%s:%d: '%s' needs ' | reason'" % (p, n, name))
        out[name.strip()] = why.strip()
    return out

def run(root):
    internal = load_internal(root)
    doc_vars, doc_text, other_vars, product_vars = set(), "", set(), {}
    for p in walk(root):
        rel = os.path.relpath(p, root)
        if rel.endswith(".md"):
            if is_user_doc(rel):
                t = read(p)
                doc_text += t + "\n"
                doc_vars |= set(VAR.findall(t))
            continue
        if not rel.endswith(SRC_EXT) and os.path.basename(p) not in ("Makefile", "run"):
            continue
        t = read(p)
        other_vars |= set(VAR.findall(t))
        if is_test(rel) or rel.startswith("docs/"):
            continue
        if rel.endswith((".cpp", ".cc", ".c", ".h", ".hpp", ".cu", ".inc")):
            for m in re.finditer(r'"((?:VGPU|PSIM)_[A-Z0-9_]*[A-Z0-9])"', t):
                product_vars.setdefault(m.group(1), rel)
        elif rel.endswith(".py"):
            for m in re.finditer(r'environ(?:\.get)?[\[(]\s*"((?:VGPU|PSIM)_[A-Z0-9_]+)"', t):
                product_vars.setdefault(m.group(1), rel)
        elif rel == "scripts/ci-setup.sh":
            for m in re.finditer(r"\$\{((?:VGPU|PSIM)_[A-Z0-9_]+)[:}-]", t):
                product_vars.setdefault(m.group(1), rel)
    problems = []
    for v in sorted(doc_vars):
        if v.endswith("_"):
            if not any(o.startswith(v) for o in other_vars):
                problems.append("documented prefix %s* matches no variable in the source" % v)
        elif v not in other_vars and v not in internal:
            problems.append("documented variable %s is not used anywhere in the source" % v)
    for v, rel in sorted(product_vars.items()):
        if v in doc_vars or v in internal:
            continue
        if any(d.endswith("_") and v.startswith(d) for d in doc_vars):
            continue
        problems.append("%s (read in %s) is not documented; document it or list it in tests/lint/doc_drift_internal.txt" % (v, rel))
    for v in sorted(internal):
        if v.startswith("--"):
            continue
        if v in doc_vars:
            problems.append("%s is listed as internal but is documented" % v)
        elif v not in other_vars:
            problems.append("%s is listed as internal but is not in the source" % v)
    # flags
    cli = {f: read(os.path.join(root, "src", "cli", f)) for f in os.listdir(os.path.join(root, "src", "cli"))
           if f.endswith((".cpp", ".hpp"))}
    all_cli = "\n".join(cli.values())
    cli_md = read(os.path.join(root, "docs", "cli.md"))
    for fl in sorted(set(re.findall(r"`(--[a-z][a-z0-9-]+)", cli_md))):
        if fl not in all_cli and fl not in internal:
            problems.append("docs/cli.md shows %s, which src/cli does not define" % fl)
    for f, t in sorted(cli.items()):
        if f not in OWN_CLI:
            continue
        for fl in sorted(set(re.findall(r'"(--[a-z][a-z0-9-]+)"', t))):
            if not re.search(re.escape(fl) + r"(?![a-z0-9-])", doc_text) and fl not in internal:
                problems.append("%s defines %s, which no documentation names" % (f, fl))
    return problems

def selftest():
    d = tempfile.mkdtemp()
    for sub in ("docs", "src/cli", "tests/lint", "src/core", "scripts"):
        os.makedirs(os.path.join(d, sub))
    def w(rel, s):
        with open(os.path.join(d, rel), "w") as f:
            f.write(s)
    w("tests/lint/doc_drift_internal.txt", "VGPU_HIDDEN | a developer knob nobody should rely on\n")
    w("src/core/a.cpp", 'auto a = getenv("VGPU_REAL"); auto h = getenv("VGPU_HIDDEN");\n')
    w("src/cli/run.cpp", 'if (a == "--known") {} if (a == "--unknown") {}\n')
    w("docs/cli.md", "`VGPU_REAL` and `--known`\n")
    assert run(d) == [("%s" % x) for x in run(d)]
    p = run(d)
    assert any("--unknown" in x for x in p) and len(p) == 1, p
    w("docs/cli.md", "`VGPU_REAL` `--known` `--unknown` `VGPU_GONE` `--gone`\n")
    p = run(d)
    assert any("VGPU_GONE" in x for x in p) and any("--gone" in x for x in p) and len(p) == 2, p
    w("docs/cli.md", "`--known` `--unknown`\n")
    p = run(d)
    assert len(p) == 1 and "VGPU_REAL" in p[0], p
    print("ok    doc drift selftest")

if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest(); sys.exit(0)
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    probs = run(root)
    for p in probs:
        print("FAIL  " + p)
    if probs:
        print("%d documentation drift problem(s)" % len(probs))
        sys.exit(1)
    print("ok    documented variables and flags exist in the source, and the source's are documented")
