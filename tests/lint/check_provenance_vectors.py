#!/usr/bin/env python3
"""A profile that claims provenance `measured` for a key must name a vectors file that backs it.

Every profile (nvidia/, amd/, intel/ profiles/*.yaml) may carry `provenance:` (class -> list of keys) and `evidence:` (key -> text), see
nvidia/docs/profiles.md and docs/numerics/. `measured` means "read from, or reproduced bit for bit against, a card of this model by this
project". Prose alone does not show that, so for every key listed under `measured`, the evidence that applies to it -- its own entry,
else its section's (`numerics.tensor_sum` -> `numerics`), following a `see <key>` pointer -- must cite at least one file that exists in
the repository, is not empty, and is data a card produced: a path under nvidia/tests/data/, amd/tests/data/, amd/registers/measurements/,
intel/registers/measurements/ or docs/numerics/ that is not Markdown and not source code. Anything cited that does not exist is also an error
(a renamed file must not leave a dangling claim). `documented`, `inferred` and `unknown` keys are not gated here.

Standard library only, no build.   Usage: check_provenance_vectors.py [repo-root]     check_provenance_vectors.py --selftest
"""
import glob, os, re, sys, tempfile

DATA_DIRS = ("nvidia/tests/data/", "amd/tests/data/", "amd/registers/measurements/", "intel/registers/measurements/", "docs/numerics/")
NOT_DATA = (".md", ".cu", ".cpp", ".hpp", ".h", ".py", ".sh", ".yaml", ".yml", ".cmake")
PATH_RE = re.compile(r"(?<![\w/.-])((?:nvidia|amd|intel|docs|tools)/[\w./+-]*[\w+-])")


def parse(text):
    """(provenance {class: [keys]}, evidence {key: text}) from a profile's restricted YAML."""
    sec, prov, evid = None, {}, {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            m = re.match(r"([A-Za-z_]+):\s*(#.*)?$", line)
            sec = m[1] if m else None
            continue
        m = re.match(r"\s+([\w.]+):\s*(.*)$", line)
        if not m or sec not in ("provenance", "evidence"):
            continue
        k, v = m[1], m[2].strip()
        if sec == "provenance":
            lst = re.match(r"\[(.*?)\]", v)
            prov[k] = [x.strip() for x in lst[1].split(",") if x.strip()] if lst else []
        else:
            q = re.match(r'"((?:[^"\\]|\\.)*)"', v) or re.match(r"'([^']*)'", v)
            evid[k] = q[1] if q else re.sub(r"\s+#.*$", "", v)
    return prov, evid


def evidence_for(key, evid, seen=()):
    """The evidence text that applies to key, or None."""
    cand = key
    while True:
        if cand in evid:
            t = evid[cand]
            m = re.match(r"\s*see\s+([\w.]+)\s*$", t)
            if m and m[1] not in seen and m[1] != cand:
                return evidence_for(m[1], evid, seen + (cand,))
            return t
        if "." not in cand:
            return None
        cand = cand.rsplit(".", 1)[0]


def is_vectors(rel):
    return rel.startswith(DATA_DIRS) and not rel.endswith(NOT_DATA)


def check(root):
    problems, n = [], 0
    files = sorted(glob.glob(os.path.join(root, "*", "profiles", "*.yaml")))
    for f in files:
        rel_f = os.path.relpath(f, root)
        with open(f, encoding="utf-8") as fh:
            prov, evid = parse(fh.read())
        for key in prov.get("measured", []):
            n += 1
            text = evidence_for(key, evid)
            if text is None:
                problems.append("%s: `%s` is measured and has no evidence entry" % (rel_f, key))
                continue
            cited = [p.rstrip(".,;:)") for p in PATH_RE.findall(text)]
            # a bare word that is not a file (e.g. "docs/matrix-numerics.md" is a document) is only an error when it is data-shaped
            backing, dangling = [], []
            for p in cited:
                full = os.path.join(root, p)
                if is_vectors(p):
                    (backing if os.path.isfile(full) and os.path.getsize(full) > 0 else dangling).append(p)
                elif not os.path.exists(full) and p.endswith(NOT_DATA + (".txt", ".json", ".csv", ".expected")):
                    dangling.append(p)
            for p in dangling:
                problems.append("%s: `%s` evidence cites %s, which does not exist or is empty" % (rel_f, key, p))
            if not backing and not dangling:
                problems.append("%s: `%s` is claimed measured but its evidence cites no vectors file (a non-empty file under %s)"
                                % (rel_f, key, ", ".join(DATA_DIRS)))
    return problems, len(files), n


def selftest():
    def tree(profile, files):
        d = tempfile.mkdtemp(prefix="provlint_")
        os.makedirs(os.path.join(d, "nvidia", "profiles"))
        with open(os.path.join(d, "nvidia", "profiles", "x.yaml"), "w") as fh:
            fh.write(profile)
        for p, c in files.items():
            os.makedirs(os.path.dirname(os.path.join(d, p)), exist_ok=True)
            with open(os.path.join(d, p), "w") as fh:
                fh.write(c)
        return d
    head = "name: x\nnumerics:\n  tensor_sum: ampere\nprovenance:\n  measured: [numerics.tensor_sum]\nevidence:\n"
    v = "nvidia/tests/data/probe.x.txt"
    cases = [
        ("backed by a vectors file", head + '  numerics: "card run; vectors %s; docs/matrix-numerics.md"\n' % v, {v: "a 1\n", "docs/matrix-numerics.md": "x"}, 0),
        ("no path at all", head + '  numerics: "card run, trust us. docs/matrix-numerics.md"\n', {}, 1),
        ("only a document cited", head + '  numerics: "see docs/matrix-numerics.md"\n', {"docs/matrix-numerics.md": "x"}, 1),
        ("vectors file missing", head + '  numerics: "vectors %s"\n' % v, {}, 1),
        ("vectors file empty", head + '  numerics: "vectors %s"\n' % v, {v: ""}, 1),
        ("no evidence", "name: x\nprovenance:\n  measured: [numerics.tensor_sum]\n", {}, 1),
        ("own entry wins over section", head + '  numerics: "vectors %s"\n  numerics.tensor_sum: "nothing"\n' % v, {v: "a\n"}, 1),
        ("see pointer followed", head + '  numerics: "vectors %s"\n  numerics.tensor_sum: "see numerics"\n' % v, {v: "a\n"}, 0),
        ("inferred is not gated", "name: x\nprovenance:\n  inferred: [numerics.tensor_sum]\nevidence:\n  numerics: \"others\"\n", {}, 0),
    ]
    bad = 0
    for name, prof, files, want in cases:
        got = 1 if check(tree(prof, files))[0] else 0
        print("%s: %s" % ("ok  " if got == want else "FAIL", name))
        bad += got != want
    return bad


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(1 if selftest() else 0)
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    problems, nfiles, nkeys = check(root)
    if problems:
        print("\n".join(problems)); print("FAIL: %d problem(s)" % len(problems)); sys.exit(1)
    print("PASS: %d measured keys in %d profiles, each backed by a vectors file" % (nkeys, nfiles))
