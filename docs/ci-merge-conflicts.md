# CI and merge conflicts

The CI does not test a pull request that has a merge conflict. Such a PR has to
be rebased and pushed again, which starts the CI over, so a run on the
conflicting code is wasted. The CI is slow and its heavy jobs wait for a shared
runner, so a PR that was clean when it was pushed often conflicts by the time
its jobs start.

## What happens

- GitHub starts no `pull_request` run for a PR that conflicts with its base at
  the moment of the push.
- When someone merges into `main`, [cancel-conflicting-prs.yml](../.github/workflows/cancel-conflicting-prs.yml)
  checks every open PR. It cancels the queued and running `pull_request` runs of
  each PR that now conflicts and adds the `merge-conflict` label to it. It
  also runs every 20 minutes, which catches a conflict that a force-push or a
  changed base caused.
- Each heavy job starts with the `pr-guard` step
  ([.github/actions/pr-guard](../.github/actions/pr-guard/action.yml), which runs
  [ci/pr-guard.sh](../ci/pr-guard.sh)). If its PR conflicts when the job starts, the
  guard cancels the whole run and the job fails with the message
  `PR #N has a merge conflict with main: cancelling this run, rebase and push`.
- Only a real conflict counts (the API's `mergeable_state` is `dirty`). A PR that
  is `behind` main, `blocked` on a review, or has a failing check is tested as
  usual. If GitHub cannot say, or the API fails, the job carries on: the guard
  never fails a job for any other reason.

## What to do

Rebase on `main` (or merge `main` into the branch) and push. The push starts
the CI again by itself; the `merge-conflict` label comes off at the next check
once the PR no longer conflicts.

To see which PRs the canceller would act on without changing anything, run the
workflow by hand with `dry_run` set (Actions, "Cancel CI on conflicting PRs", Run
workflow).

## Limits

- A conflict that appears while a job is already running is not seen by that
  job's guard (it only looks at the start). The canceller catches it after the
  push to `main` that caused it, usually within a minute or two.
- A conflict can also appear on a PR whose run is complete. Nothing is
  cancelled then; the label is the signal.
- Jobs on a bare image with neither `gh` nor `curl` cannot ask the API. The
  `container` job in `action.yml` is one, and is exempt.
- On a PR from a fork the token is read-only, so the guard cannot cancel the
  run; it fails its job instead.

## Adding a job

A job that runs on pull requests and is heavy (on `BURST_RUNS_ON`, `ML_RUNS_ON`,
`SIM_RUNS_ON` or a self-hosted runner, or with a timeout of 20 minutes or more)
needs the guard as the step after `actions/checkout`, and the permissions it
uses:

```yaml
    permissions: { contents: read, packages: read, actions: write, pull-requests: read }
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/pr-guard
```

If the checkout puts the repository in a subdirectory (`path: pantheonsim`), the
action is `./pantheonsim/.github/actions/pr-guard`. The `lint_pr_guard_coverage`
test fails for a heavy job without it, and names the workflow and job; a job
that cannot have the guard goes in the exemption list at the top of
`tests/lint/check_pr_guard_coverage.py`, with the reason.
