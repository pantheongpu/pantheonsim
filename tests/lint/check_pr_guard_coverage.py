#!/usr/bin/env python3
"""Every heavy job of a workflow that runs on pull requests has the merge-conflict guard.

A pull request that conflicts with main is not worth testing: it has to be
rebased and pushed, which starts the CI over. Heavy jobs wait hours for the
shared runners, so a conflict often appears while one is queued. The guard
(.github/actions/pr-guard, which runs ci/pr-guard.sh) is the first step after
the checkout of such a job: it cancels the run when the PR conflicts. A new
heavy job added without it would spend runner time on a conflicting PR again.

A job needs the guard when its workflow has `pull_request` among its triggers
and the job runs on a shared runner (BURST_RUNS_ON, CONTAINER_RUNS_ON, ML_RUNS_ON, SIM_RUNS_ON,
pantheonsim-burst or any self-hosted label), or its timeout-minutes is 20 or
more (or an expression, which can be longer), or it has no timeout and is on a
shared runner. Jobs that only call a reusable workflow, and light jobs on
GitHub's runners (a summary, the image finder), are not checked.

The guard also has to work: it comes after nothing but checkouts, and its job
has `actions: write` (to cancel the run) and `pull-requests: read` (to read the
PR), in the job's own `permissions:` or the workflow's.

EXEMPT lists the jobs that are heavy by this rule and have no guard, with why.
An entry that is no longer needed (the job is gone, or has the guard) fails
too, so the list stays honest.

Reads the workflow files as text (the standard library only: PyYAML is not in
every container). usage: check_pr_guard_coverage.py [workflow.yml ...]
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

EXEMPT = {
    ("sass-ptx.yml", "ptx"):
        "its `if:` runs it on schedule and workflow_dispatch only, never on a pull request",
    ("action.yml", "container"):
        "runs in the bare ubuntu:24.04 image, which has neither gh nor curl, so the guard could not ask the API",
}

SHARED = re.compile(r"BURST_RUNS_ON|CONTAINER_RUNS_ON|ML_RUNS_ON|SIM_RUNS_ON|pantheonsim-burst|self-hosted")
GUARD = re.compile(r"^\s*-?\s*uses:\s*\./(?:[\w.\-/]+/)?\.github/actions/pr-guard\s*(?:#.*)?$")
CHECKOUT = re.compile(r"^\s*-?\s*uses:\s*actions/checkout@")


def indent(line):
    return len(line) - len(line.lstrip(" "))


def top_block(lines, key):
    """The lines under a top-level `key:` (indented deeper), without the key line."""
    for i, line in enumerate(lines):
        if re.match(rf"^[\"']?{key}[\"']?:", line):
            block = [line.split(":", 1)[1]]
            for nxt in lines[i + 1:]:
                if nxt.strip() and indent(nxt) == 0:
                    break
                block.append(nxt)
            return block
    return []


def jobs_of(lines):
    """name -> its lines, for the jobs: mapping (job keys at indent 2)."""
    block = top_block(lines, "jobs")
    jobs, name = {}, None
    for line in block[1:]:
        m = re.match(r"^  ([A-Za-z0-9_\-]+):\s*(?:#.*)?$", line)
        if m:
            name = m.group(1)
            jobs[name] = []
        elif name is not None:
            jobs[name].append(line)
    return jobs


def perms_text(block_lines):
    """The text of a `permissions:` entry at the first line that starts one, or None."""
    for i, line in enumerate(block_lines):
        if re.match(r"^permissions:", line) or re.match(r"^    permissions:", line):
            text = [line.split(":", 1)[1]]
            base = indent(line)
            for nxt in block_lines[i + 1:]:
                if nxt.strip() and indent(nxt) <= base:
                    break
                text.append(nxt)
            return "\n".join(text)
    return None


def job_facts(body):
    facts = {"uses": None, "runs_on": "", "timeout": None, "steps": [], "perms": perms_text(body)}
    in_steps = False
    for line in body:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        ind = indent(line)
        if ind == 4:
            in_steps = line.strip() == "steps:"
            m = re.match(r"^    (uses|runs-on|timeout-minutes):\s*(.*?)\s*(?:#.*)?$", line)
            if m:
                key, val = m.groups()
                if key == "uses":
                    facts["uses"] = val
                elif key == "runs-on":
                    facts["runs_on"] = val
                else:
                    facts["timeout"] = val
        elif in_steps and ind == 6 and line.lstrip().startswith("- "):
            facts["steps"].append([line])
        elif in_steps and facts["steps"]:
            facts["steps"][-1].append(line)
    return facts


def heavy(facts):
    shared = bool(SHARED.search(facts["runs_on"]))
    t = facts["timeout"]
    if t is None:
        return shared
    if t.isdigit():
        return shared or int(t) >= 20
    return True   # an expression: it can be long


def check(path):
    problems, seen_exempt = [], set()
    lines = path.read_text(encoding="utf-8").split("\n")
    on_keys = {m.group(1) for ln in top_block(lines, "on") if (m := re.match(r"^  ([a-z_]+):", ln))}
    if "pull_request" not in on_keys:
        return problems, seen_exempt, {}
    wf_perms = perms_text(top_block_with_key(lines, "permissions"))
    guarded = {}
    for name, body in jobs_of(lines).items():
        facts = job_facts(body)
        if facts["uses"]:
            continue
        guard_at = next((i for i, step in enumerate(facts["steps"]) if any(GUARD.match(s) for s in step[:1])), None)
        key = (path.name, name)
        if guard_at is not None:
            guarded[key] = True
            if key in EXEMPT:
                problems.append(f"{path.name}: job '{name}' has the guard but is in EXEMPT ({EXEMPT[key]}); remove the entry")
            before = facts["steps"][:guard_at]
            if any(not CHECKOUT.match(step[0]) for step in before):
                problems.append(f"{path.name}: job '{name}': the guard must come right after the checkout, before any other step")
            perms = facts["perms"] if facts["perms"] is not None else wf_perms
            if perms is None or not re.search(r"actions:\s*write", perms) or not re.search(r"pull-requests:\s*(read|write)", perms):
                problems.append(f"{path.name}: job '{name}' has the guard but not `actions: write` and `pull-requests: read` permissions")
        elif heavy(facts):
            if key in EXEMPT:
                seen_exempt.add(key)
            else:
                problems.append(f"{path.name}: job '{name}' (runs-on {facts['runs_on'] or '?'}, timeout {facts['timeout']}) "
                                f"runs on pull requests and has no `- uses: ./.github/actions/pr-guard` step after the checkout")
    return problems, seen_exempt, guarded


def top_block_with_key(lines, key):
    """top_block that keeps the key's own line, so perms_text can find it."""
    for i, line in enumerate(lines):
        if re.match(rf"^{key}:", line):
            block = [line]
            for nxt in lines[i + 1:]:
                if nxt.strip() and indent(nxt) == 0:
                    break
                block.append(nxt)
            return block
    return []


def main(argv):
    files = [pathlib.Path(a) for a in argv] or sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    problems, exempt_used, n_guarded = [], set(), 0
    for path in files:
        p, e, g = check(path)
        problems += p
        exempt_used |= e
        n_guarded += len(g)
    if not argv:
        for key in EXEMPT:
            if key not in exempt_used:
                problems.append(f"EXEMPT has {key[0]}:{key[1]} but that job is missing, not heavy, or has the guard; remove the entry")
    for p in problems:
        print("FAIL  " + p)
    if problems:
        return 1
    print(f"ok    {n_guarded} heavy jobs on pull requests have the merge-conflict guard ({len(EXEMPT)} exempt)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
