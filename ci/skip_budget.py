#!/usr/bin/env python3
"""The skipped-test budget (ci/skip-budget.txt): a skip must be listed, give a reason, and not outgrow the total.

  skip_budget.py lint [REPO]          the budget file is well formed, every entry has a reason, the total equals the sum, ci.yml's
                                      RUNNER_SKIPS_OK names exactly the budgeted tests, and each "where it runs instead" workflow exists
                                      and (for the weekly one) selects the test
  skip_budget.py ratchet BASE [REPO]  fail when the budget grew against BASE (a copy of the budget file on the base branch): a higher
                                      total or a test not listed there
  skip_budget.py check CTEST_LOG REASONS_LOG [REPO]
                                      a ctest run's skips: each is budgeted, each has a reason (a "N: SKIP: text" line of ctest -V), and
                                      together they do not exceed the total
  skip_budget.py selftest             the rules above on synthetic input
Python 3 standard library only.
"""
import os, re, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

def parse(path):
    total, entries, errs = None, [], []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = re.match(r"^total (\d+)$", line)
            if m:
                total = int(m.group(1))
                continue
            parts = [p.strip() for p in line.split(" | ")]
            if len(parts) != 4:
                errs.append("%s:%d: expected 'test | count | where | reason'" % (path, n)); continue
            name, cnt, where, why = parts
            if not cnt.isdigit() or int(cnt) < 1:
                errs.append("%s:%d: count of %s is not a positive integer" % (path, n, name)); continue
            if len(why) < 12:
                errs.append("%s:%d: skip of %s has no reason" % (path, n, name))
            try:
                re.compile(name)
            except re.error:
                errs.append("%s:%d: %s is not a regular expression" % (path, n, name)); continue
            entries.append((name, int(cnt), where, why))
    if total is None:
        errs.append("%s: no 'total N' line" % path)
    names = [e[0] for e in entries]
    for d in sorted({x for x in names if names.count(x) > 1}):
        errs.append("%s: %s is listed twice" % (path, d))
    if total is not None and total != sum(e[1] for e in entries):
        errs.append("%s: total is %d but the counts add up to %d" % (path, total, sum(e[1] for e in entries)))
    return total, entries, errs

def lint(root):
    total, entries, errs = parse(os.path.join(root, "ci", "skip-budget.txt"))
    wf = os.path.join(root, ".github", "workflows")
    with open(os.path.join(wf, "ci.yml"), encoding="utf-8") as f:
        ci = f.read()
    m = re.search(r"RUNNER_SKIPS_OK: '\^\((.*?)\)\$'", ci)
    if not m:
        errs.append("ci.yml: no RUNNER_SKIPS_OK")
    else:
        in_ci = set()
        for alt in m.group(1).split("|"):
            in_ci |= {alt.replace("(_blocks)?", ""), alt.replace("(_blocks)?", "_blocks")} if "(_blocks)?" in alt else {alt}
        listed = {e[0] for e in entries if e[0] != "amd_rocprof_.*"}
        for t in sorted(in_ci - listed):
            errs.append("ci.yml allows %s but the budget does not list it" % t)
        for t in sorted(listed - in_ci):
            errs.append("the budget lists %s but ci.yml's RUNNER_SKIPS_OK does not allow it" % t)
    if "ci/skip_budget.py" not in ci:
        errs.append("ci.yml does not run ci/skip_budget.py")
    for name, cnt, where, why in entries:
        if where == "-":
            continue
        p = os.path.join(wf, where)
        if not os.path.isfile(p):
            errs.append("%s: runs instead in %s, which does not exist" % (name, where)); continue
        if where == "skipped-tests-weekly.yml":
            with open(p, encoding="utf-8") as f:
                body = f.read()
            probe = "amd_rocprof_" if name == "amd_rocprof_.*" else name
            if probe not in body:
                errs.append("%s: the weekly workflow does not select it" % name)
    return total, errs

def ratchet(base, root):
    bt, be, berrs = parse(base)
    t, e, errs = parse(os.path.join(root, "ci", "skip-budget.txt"))
    errs = list(errs)
    if berrs:
        return []
    if t is not None and bt is not None and t > bt:
        errs.append("the skip budget grew: total %d, was %d" % (t, bt))
    old = {x[0]: x[1] for x in be}
    for name, cnt, *_ in e:
        if name not in old:
            errs.append("new skipped test in the budget: %s" % name)
        elif cnt > old[name]:
            errs.append("%s may skip %d times, was %d" % (name, cnt, old[name]))
    return errs

def check(ctest_log, reasons_log, root):
    total, entries, errs = parse(os.path.join(root, "ci", "skip-budget.txt"))
    errs = list(errs)
    with open(ctest_log, errors="replace") as f:
        skipped = dict((int(a), b) for a, b in re.findall(r"Test +#(\d+): ([A-Za-z0-9_]+) \.+ *\*{3}(?:Skipped|Not Run)", f.read()))
    with open(reasons_log, errors="replace") as f:
        why = {}
        for a, b in re.findall(r"^(\d+): SKIP: ?(.*)$", f.read(), re.M):
            why[int(a)] = b.strip()
    used = {}
    for num, name in sorted(skipped.items()):
        hit = [e for e in entries if re.fullmatch(e[0], name)]
        if not hit:
            errs.append("%s skipped and is not in the skip budget" % name)
        else:
            used[hit[0][0]] = used.get(hit[0][0], 0) + 1
        if not why.get(num):
            errs.append("%s skipped with no reason (no 'SKIP: text' line)" % name)
    for k, v in used.items():
        lim = [e[1] for e in entries if e[0] == k][0]
        if v > lim:
            errs.append("%s skipped %d tests, the budget allows %d" % (k, v, lim))
    if total is not None and len(skipped) > total:
        errs.append("%d tests skipped, the budget allows %d" % (len(skipped), total))
    return len(skipped), errs

def selftest():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "ci"))
    bud = os.path.join(d, "ci", "skip-budget.txt")
    def w(s):
        with open(bud, "w") as f:
            f.write(s)
    good = "total 3\na | 1 | - | because the image lacks a\nb_.* | 2 | - | because the image lacks b\n"
    w(good)
    assert not parse(bud)[2]
    w(good.replace("total 3", "total 4"))
    assert any("add up" in e for e in parse(bud)[2])
    w(good.replace("because the image lacks a", "x"))
    assert any("no reason" in e for e in parse(bud)[2])
    w(good + "a | 1 | - | because the image lacks a\n")
    assert any("twice" in e for e in parse(bud)[2])
    base = os.path.join(d, "base.txt")
    with open(base, "w") as f:
        f.write(good)
    w(good)
    assert not ratchet(base, d)
    w("total 4\na | 1 | - | because the image lacks a\nb_.* | 2 | - | because the image lacks b\nc | 1 | - | because the image lacks c\n")
    r = ratchet(base, d)
    assert any("grew" in e for e in r) and any("new skipped" in e for e in r), r
    w(good)
    log, rs = os.path.join(d, "log"), os.path.join(d, "rs")
    def run(lines, reasons):
        with open(log, "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(rs, "w") as f:
            f.write("\n".join(reasons) + "\n")
        return check(log, rs, d)[1]
    sk = lambda n, name: " 1/9 Test #%d: %s .....***Skipped   0.1 sec" % (n, name)
    assert not run([sk(1, "a"), sk(2, "b_x")], ["1: SKIP: no a", "2: SKIP: no b"])
    assert any("not in the skip budget" in e for e in run([sk(1, "zzz")], ["1: SKIP: why"]))
    assert any("no reason" in e for e in run([sk(1, "a")], ["1: SKIP: "]))
    assert any("allows" in e for e in run([sk(1, "a"), sk(2, "a")], ["1: SKIP: y", "2: SKIP: y"]))
    print("ok    skip budget selftest")

if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    root_default = os.path.join(HERE, "..")
    if cmd == "lint":
        t, errs = lint(os.path.abspath(a[1] if len(a) > 1 else root_default))
    elif cmd == "ratchet":
        errs = ratchet(a[1], os.path.abspath(a[2] if len(a) > 2 else root_default)); t = None
    elif cmd == "check":
        t, errs = check(a[1], a[2], os.path.abspath(a[3] if len(a) > 3 else root_default))
    elif cmd == "selftest":
        selftest(); sys.exit(0)
    else:
        sys.exit(__doc__)
    for e in errs:
        print("FAIL  " + e)
    if errs:
        sys.exit(1)
    print("ok    skip budget (%s)" % cmd)
