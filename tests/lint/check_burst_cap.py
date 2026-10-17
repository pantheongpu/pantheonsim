#!/usr/bin/env python3
"""Every job that can land on the shared burst pool carries the pool owner's resource cap.

The burst pool (the repository variable BURST_RUNS_ON) can include a machine
that also runs something that must not be starved, such as a production
database. The owner limits what a job may take with three repository
variables, which are empty (no limit) until set:

  BURST_CONTAINER_OPTIONS   extra `docker create` options, e.g. `--cpus=12 --memory=16g`
  BURST_MEM_LIMIT_GIB       memory ceiling for a job that runs on the runner itself
  BURST_JOBS_MAX            parallelism ceiling for the same (read by ci/jobs.sh)

A job on BURST_RUNS_ON that runs in a container must put
`${{ vars.BURST_CONTAINER_OPTIONS }}` in its container options. One that runs
on the runner itself must pass CI_MEM_LIMIT_GIB and CI_JOBS_MAX from the other
two variables in its `env:`. A new job added without them would ignore the cap.

EXEMPT lists jobs that are routed to the pool but are light enough not to
need it, with why. An entry that is no longer needed fails too.

Reads the workflow files as text (the standard library only).
usage: check_burst_cap.py [workflow.yml ...]
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

EXEMPT = {
    ("ci-image-ref.yml", "find"):
        "only resolves the toolchain image names (a few API calls), builds nothing",
}


def jobs(text):
    """{job name: its text} for the top-level entries under `jobs:`."""
    out, name, buf, in_jobs = {}, None, [], False
    for line in text.split("\n"):
        if re.match(r"^jobs:\s*$", line):
            in_jobs = True
            continue
        if in_jobs and re.match(r"^\S", line):
            in_jobs = False
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line) if in_jobs else None
        if m:
            if name:
                out[name] = "\n".join(buf)
            name, buf = m.group(1), []
        elif name:
            buf.append(line)
    if name:
        out[name] = "\n".join(buf)
    return out


def check(path):
    problems, used = [], set()
    for job, body in jobs(path.read_text()).items():
        if not re.search(r"^    runs-on:.*BURST_RUNS_ON", body, re.M):
            continue
        key = (path.name, job)
        has_container = re.search(r"^    container:", body, re.M)
        if has_container:
            ok = re.search(r"^      options:.*vars\.BURST_CONTAINER_OPTIONS", body, re.M)
            want = "put ${{ vars.BURST_CONTAINER_OPTIONS }} in its container options"
        else:
            ok = (re.search(r"CI_MEM_LIMIT_GIB:\s*\$\{\{\s*vars\.BURST_MEM_LIMIT_GIB", body)
                  and re.search(r"CI_JOBS_MAX:\s*\$\{\{\s*vars\.BURST_JOBS_MAX", body))
            want = "pass CI_MEM_LIMIT_GIB and CI_JOBS_MAX from BURST_MEM_LIMIT_GIB / BURST_JOBS_MAX in its env"
        if key in EXEMPT:
            used.add(key)
            if ok:
                problems.append(f"{path.name}: {job} is in EXEMPT but already has the cap; remove the entry")
        elif not ok:
            problems.append(f"{path.name}: job {job} runs on BURST_RUNS_ON and must {want}")
    return problems, used


def main(argv):
    files = [pathlib.Path(a) for a in argv] or sorted((ROOT / ".github/workflows").glob("*.yml"))
    problems, used = [], set()
    for f in files:
        p, u = check(f)
        problems += p
        used |= u
    if not argv:
        for key in EXEMPT:
            if key not in used:
                problems.append(f"EXEMPT entry {key} matches no job on BURST_RUNS_ON; remove it")
    for p in problems:
        print("FAIL:", p)
    if not problems:
        print("ok: every job on BURST_RUNS_ON carries the burst cap")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
