# CI: where the jobs run, how long they take, how they fail, and how to recover

Written 2026-10-15 from the logs of CI runs 38424481981 (the last green run of
PR #362), 38612576048 and 38588962078 (main), and a read of the runner list on
that day. Numbers below are those runs' unless stated. Nothing here was measured
by running the suite on a developer machine.

## Where each job runs

| job (ci.yml) | runs on | variable |
|---|---|---|
| build + unit tests, pantheon workloads, AMD code objects | GitHub-hosted `ubuntu-24.04` (4 cores, 16 GB) | `SIM_RUNS_ON` if set |
| PyTorch, ROCm and the rest of the suite (`ml`) | runner `pantheonsim-ci` on the pantheonsim.com host | `ML_RUNS_ON` |
| ollama (NVIDIA and each AMD GPU) | the burst pool, label `pantheonsim-burst` | `BURST_RUNS_ON` |
| SASS, ASan + UBSan, ThreadSanitizer, and `find the toolchain images` | the burst pool too, unless `CONTAINER_RUNS_ON` is set | `CONTAINER_RUNS_ON`, else `BURST_RUNS_ON` |

The burst pool on 2026-10-15 is one runner, `server2-1` (24 CPUs; 15 to 23 GiB
available when the jobs in these logs started; `scripts/ci-pool.sh` lists it). Every container job of a run `needs:` the image
finder, which also runs on the pool, so a pool that does not answer holds up
every pooled job of the run, not only the long ones. When this was
written 36 jobs were waiting for the label, the oldest for 258 minutes, with the
runner busy: queue time, not run time, is most of the wait on a pull request
(time from job creation to start on the pool: TSan 11024 s and SASS 8540 s in run
38588962078; ASan 15549 s, TSan 19141 s, an ollama job 21183 s in 38424481981).

## What the wall time is made of

Slowest jobs (run time, not queue time), run 38424481981:

| job | seconds | where it goes |
|---|---|---|
| build + unit tests | 3937 | install 58, build 654, ctest 3205 |
| ASan + UBSan | 2011 | build 102, ctest 1840 |
| PyTorch/ROCm (`ml`) | 1689 | build 333, ML tests 395, the rest at `-j2` 956 |
| pantheon workloads | 1667 | build 677, workloads 940 |
| ThreadSanitizer | 1605 | build 99, ctest 1436 |
| ollama on every GPU | 350 | |
| SASS on every architecture | 213 | |

Slowest tests in the build job (ctest `Passed N sec`): `e2e_cutlass_hopper`
1173 s (1439 s in 38612576048), `e2e_cutlass_gemm` 995 s (1024 s), then
`e2e_sass_archs` 111 s, `e2e_lowprec_lt` 53 s, `e2e_wgmma_cute` 40 s; nothing
else is over 40 s. The two CUTLASS tests are 2167 of the 3204 s of ctest time
(68%). In the sanitizer jobs no test is over 173 s (`e2e_sass_archs`,
`e2e_lowprec_lt` 156 s, `amd_memtest_patterns` 130 s, `e2e_ipc` 128 s,
`test_memfast` 126 s): 534 tests, 1840 s (ASan) and 1436 s (TSan) one at a time
on a 24-core machine.

What was done about it, each a separate commit:

- **`ctest -j2` in the build job.** The tests that hold the `nvcc_memory`
  resource lock (CUTLASS, CuTe, `e2e_sass_archs`, `e2e_device_runtime`,
  `e2e_sync_masks`) are 2361 s of the 3204 s and still run one at a time; the
  other 843 s run beside them. Floor with two slots: 2361 s, so about 800 s off a
  3200 s step, less what contention for the four cores costs. The two CUTLASS
  tests have a `COST` so ctest starts them first; without it ctest uses test
  number order, they are numbered last, and the lock chain starts after
  everything else has finished (a synthetic project of 16 one-second tests and
  two eight-second locked tests: 24.1 s without the cost, 16.1 s with, ctest
  4.2.3). The count comes from `ci/jobs.sh -g 6 -m 2`: two on a machine with 12
  GiB free, one below that.
- **`ctest -j2` in the ASan and TSan jobs.** Floor with two slots 920 s and 720 s
  for the first run (the total over two). The pool has one runner, so the minutes
  saved are also minutes the queue behind these jobs does not wait.
- **Not done: trimming `amd_memtest_patterns`** (130 to 150 s under a sanitizer).
  The test loops over six GPUs and prints per-GPU lines only with `-V`, so the logs
  say nothing about the share of each. One of the six (`rx6800`) runs the same
  checked-in program as `rx6900xt` (`memtest.gfx1030`), so dropping it under a
  sanitizer saves at most a sixth of 130 to 150 s. With `-j2` the test overlaps
  others and is not what a sanitizer job waits for (the longest test of the job is
  173 s and the total over two slots is over 700 s), so the saving would be about
  20 s of a job for a loss of coverage. Left alone.

Why `-j2` is safe, and what is not proven. The `ml` job runs the same tests at
`-j2` on every run (all but the CUTLASS/CuTe group), so a shared port, shared
fixed temporary name or shared state between two of them would have shown. The
tests that fetch CUTLASS serialise the fetch (`cutlass_fetch.sh`) and each makes
its own `mktemp` directory; the fixed names the tests do use are one per test
(`/tmp/vgpu-graph-build.dot`, `/tmp/vgpu-graph-nodes.dot`) or carry the pid; no
test binds a fixed port. Not proven: peak memory of a sanitized test (nothing in
the logs says), and CUTLASS's 10 GB compile beside another test on the 16 GB
hosted runner (the test sizes its compiles from the memory free when it starts,
so it takes one compile at a time there). If a sanitizer job is killed with exit
137 in its test step, or the build job in a CUTLASS compile, revert the commit.

## Shards and the compile cache

A survey of run 41545325796 on main found about 26,500 runner-seconds per run. Three tests were 35% of the unit job's CPU, and
six jobs each compiled the same library. Two changes, neither of which removes a check:

**Sharded tests.** `vgpu_shard_tests` (CMakeLists.txt) registers a long test as parallel parts, so that `ctest -j` can overlap them:

| was | now | one part |
|---|---|---|
| `test_amd_gcn_tier_same_bits` (679 s plain, about 3,400 s under ASan) | `test_amd_gcn_tier_same_bits_shard_0of8` ... `_7of8` | every eighth corpus shape with a handler (`VGPU_TIER_SHARD=k/n`); a shape keeps its seed and flavours, so the eight run exactly the runs of the whole |
| `test_amd_gcn_tier_same_bits_simd_base` (688 s) | `..._simd_base_shard_0of8` ... `_7of8` | the same, with `VGPU_SIMD=base` |
| `e2e_sass_jit` (1,611 s) | `e2e_sass_jit_shard_tier`, `_steps`, `_steps_gather`, `_steps_shared`, `_random`, `_modes` | `VGPU_JIT_SHARD`: sass_tier's variants, sass_tier_steps' faults and region caps, its two step-limit bisections, the random kernels, the failure modes; each shard builds only the programs it runs |

Each family also has a `_shards_partition` test (`tests/lint/check_shard_partition.sh`). The test or script lists the items a run would
take (`VGPU_TIER_LIST=1`: corpus and line of each shape; `VGPU_JIT_LIST=1`: stage names) without running anything, and the check
fails unless the shards' lists together are the whole list, every item once, none empty. The shard specs it checks are the ones the
shard tests were registered with, so a shard added to one list and not the other cannot slip through.

For local use: the binary still runs the whole test (`build/test_amd_gcn_tier_same_bits`, `nvidia/tests/e2e/run_sass_jit.sh`), `ctest -R
test_amd_gcn_tier_same_bits` and `ctest -R '^e2e_sass_jit_shard'` run all the parts, and `-DVGPU_TEST_SHARDS=OFF` registers the old single
tests under the old names. `e2e_sass_jit_full` (the nightly) is unchanged. In `ci.yml`'s ThreadSanitizer job the exclusion list
names the tier shards, as it named the whole test.

Shards only help when there are ctest slots to run them in, so the unit, ASan, TSan, `ml` and AMD code-object jobs now allow up to four
(`ci/jobs.sh -g 6 -m 4`; was `-m 2`, and no `-j` at all in the AMD code-object job). The memory rule is unchanged: a machine with less
than 6 GiB per slot gets fewer.

**Compile cache.** `.github/actions/ccache` turns on ccache for the library's C and C++ compiles in the jobs that build it (the unit
job under its own key, because it builds with `-Werror`; workloads and AMD code objects under `cuda12-release`; SASS, ollama and each
ollama-amd job under `cuda13-release`; `ml` under `ml-release`, on its runner's disk). An object is looked up by the preprocessed
source, the compiler's contents and the flags, so a hit is what the compiler would have made; `CCACHE_BASEDIR` makes the workspace
path (`/home/runner/work/...` against `/__w/...`) irrelevant. The Actions cache holds it: one entry per main commit, written only by a
push to main (`.github/actions/ccache-save`; a pull request reads the newest main entry and writes nothing), one writer per key (workloads
for `cuda12-release`, SASS for `cuda13-release`, the unit job for its own), the others restore only. Links are not cached: the LTO link of
the shims and the test executables still runs in every job. If ccache is not installed and cannot be (no root, no passwordless sudo), the job
builds as before.

Why not a build artifact shared between jobs: the jobs run in parallel, so one would have to `needs:` another (and wait for its tests, or
a new build job would go in front of all of them); the build directory holds absolute paths, which differ between a container and a
runner; and the toolchains differ (CUDA 12.0 and 13.0). A cache keyed on content needs none of that.

To see what the cache does in a run, read the "ccache statistics" step of each job (hit rate, and misses by cause), and the step times
of "Build" against the previous run. Not measured: this was written without a CI run.

## How CI has failed, and what it looked like

**The pool went away or never answered.** A job whose label no online runner
carries stays queued until GitHub cancels it after 24 hours (documented limit:
"A job can be in the queue for 24 hours before it is automatically cancelled").
`timeout-minutes` does not start until a runner takes the job.

**Exit 137 and "The runner has received a shutdown signal".** Job 115063329672
(ASan, run 38323704166): the build step, halfway through linking, ended with
`Process completed with exit code 137` and then `The runner has received a
shutdown signal`. That is the kernel's out-of-memory killer taking the build, and
the runner service with it. Instrumented builds are the cause: one ASan or TSan
link is 3 to 6 GB, and several such compiles and links at once (`-j7` at 3 GiB
each is 21 GiB, on a host that other jobs share) exhaust it; the log does not say
what else was running on the host at 00:20 UTC, so this is the cause the
exit code and the step point to, not a measured one. `ci/jobs.sh` now sizes the
parallelism to the memory available, and `ci/jobs.sh --check asan` fails a job
that cannot fit before it starts. An earlier instance killed the host's
PostgreSQL (the header of `ci/jobs.sh` records it).

**AddressSanitizer odr-violation.** Job 115115792387 (ASan, run 38340039147):
`ERROR: AddressSanitizer: odr-violation: global 'g_enabled' at
src/simd/simd.cpp:215:6 in libcudart.so.12 / libcuda.so.1`, in every test that
loads both libraries (`c_harness_vector_add`, `test_driver_mempool`, ...). It
also happened with `regs::kSysfsFiles` and the AMD hwmon file table. Cause:
libcuda and libcudart each link `libvgpu.a`, so a data symbol with external
linkage defined there exists twice in a process that loads both. Fix each time:
internal linkage, `inline`, or hidden visibility. `tests/lint/check_odr_globals.py`
(ctest `lint_odr_globals`) now finds one in the plain build; run against the
CUDA 13.0 Release builds in other worktrees it reports exactly the symbols
above (kSysfsFiles, kHwmonFiles, kDriverFiles; and g_kernels, g_enabled on the
round-7 build) and passes on a build after the fixes. A symbol that is shared on
purpose goes in `tests/lint/odr_globals.allow` with its reason.

**A workflow file GitHub refuses.** A repeated key (a second `digest:` job in
`sass-ptx.yml`) makes GitHub treat the whole file as invalid: the workflow runs
on every push as a failure with no jobs and its schedule stops. ctest
`lint_workflow_yaml` (tests/lint/check_workflow_yaml.py) reads every workflow for
one.

**A root-owned workspace on a runner that lasts.** Container jobs run as root in
the runner user's workspace; what they leave is root's, and the next job's
checkout (a job outside a container: the image finder, the AMD ollama jobs)
fails with permission denied, as do all later non-container jobs on that
runner. The container jobs end with a `chown -R` hand-back step for this
(`if: always() && job.container.id`); a new pooled container job needs it too.

**Leftovers after a killed job.** A job killed by exit 137 runs none of its
cleanup: the e2e scripts' `trap ... EXIT` for their `mktemp -d` directories does
not run on SIGKILL. Container jobs lose that with the container; a job outside
a container (the image finder, the AMD ollama jobs) leaves it in the host's
`/tmp`, and its checkout and build tree stay in the runner's `_work`. I have not
looked at the pool host's disk; `df -h /tmp /var/lib/docker`, `docker system df`
and `ls /tmp | wc -l` on it say whether this has become a problem.

## Why there is no automatic fallback from the pool

Looked at and ruled out without a change to repository variables, settings or
secrets:

- **Ask for the runner list from a job.** `GET /repos/{repo}/actions/runners`
  needs repository admin; `GITHUB_TOKEN` has no `administration` permission (the
  `permissions:` scopes are actions, artifact-metadata, attestations, checks,
  code-quality, contents, copilot-requests, deployments, id-token, issues,
  discussions, packages, pages, pull-requests, security-events, statuses,
  vulnerability-alerts). A pre-job that picks the label from the runner list
  needs a personal access token in a secret.
- **A label list as a fallback.** `runs-on: [a, b]` means a runner with both
  labels, not either.
- **A timeout on the queue.** There is none; 24 hours is GitHub's.
- **Infer health from job history** (any job on the label started in the last N
  hours). A pool idle for N hours would then be judged dead, jobs would run
  hosted, the pool would see no job, and it would stay judged dead. Not done.
- **Concurrency groups** already cancel a run superseded by a newer push to the
  same ref (`ci.yml`), which keeps stale pull-request runs from holding the queue.

What would work, each needing a decision that is the maintainer's: a second
runner in the pool (the single runner is the bottleneck, not a fault); a
fine-grained token with read access to administration in a secret, for a pre-job
that sets `runs-on` from the runner list; a scheduled workflow with
`issues: write` that opens an issue when jobs have waited over two hours with no
job in progress on the label.

## The burst pool is offline: recovery

`scripts/ci-pool.sh` (read-only; `--queued` also counts waiting jobs) says
whether a runner carrying `pantheonsim-burst` is online. If none is and jobs
are waiting, as the maintainer:

1. Either bring the runner back (on the host that runs `server2-1`, however
   it was started: its `actions.runner.*` systemd service, or `./run.sh` in its
   directory), or send the jobs to GitHub's hosted runners:

       gh variable delete BURST_RUNS_ON --repo pantheongpu/pantheonsim

   With the variable unset (and `CONTAINER_RUNS_ON` too, if it was ever set),
   every `runs-on: ${{ fromJSON(vars.BURST_RUNS_ON || '"ubuntu-24.04"') }}` takes
   the hosted runner, and the jobs install their
   toolchain with apt (the image finder returns no image only if the registry
   is unreachable; the container jobs run on the hosted runner either way).
2. A job that is already queued keeps the label it was queued with. Cancel the
   waiting runs and start them again: `gh run cancel <id>` then `gh run rerun
   <id>` (a run that is not finished cannot be re-run job by job), or push an empty
   commit to the branch. Whether a re-run reads the changed variable has not
   been tried here; check the first one before cancelling forty.
3. When the runner is back, restore the variable (`gh variable set
   BURST_RUNS_ON --body '["pantheonsim-burst"]'`).

Do not put a runner in the pool without `--no-default-labels`: with the default
labels (`self-hosted`, `Linux`, `X64`) another job's `runs-on` could match it
(the ML jobs were once handed to a pool runner that way and failed with
`cmake: command not found`). `server2-1` also carries the label
`pantheonsim-ci` that the ML job asks for; the ML job asks for `self-hosted` as
well, which that runner does not have, so it does not match, but the label does
not belong on it.

If a job in the pool died with exit 137: check the host's memory
(`ci/jobs.sh --available`, `free -g`), look for a second runner process
building at the same time, and re-run the job once the host is quiet. Raising
the per-job GiB of a profile in `ci/jobs.sh` makes the builds narrower.
