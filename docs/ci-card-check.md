# Card check: the e2e programs on a real GPU (design, not enabled)

Status: a proposal. No runner is registered and no workflow runs it. Turning it
on is the steps under [Enabling it](#enabling-it), all of which are the
maintainer's to take.

## Why

Every e2e test compiles an unmodified CUDA program with nvcc and runs it
against the simulator's libraries (`LD_LIBRARY_PATH=build/shim`). What each
program prints, and the expected values its script checks, were written
against NVIDIA's real libraries and hardware once, by hand, when the test was
added. Nothing checks them again, and both sides move:

- NVIDIA's driver and libraries change behaviour between releases (error
  codes, attribute values, cuBLAS heuristics, printf formatting, which
  `cudaDeviceProp` fields exist). A test written against driver 5xx can keep
  passing on the simulator long after the real stack answers differently.
- A test can be wrong in a way only the hardware shows: an expected value
  copied from the simulator's output instead of from the card, or a program
  with a race the simulator's schedule happens to hide (the HeCBench and
  PolyBench runs found several of each).

The card check runs the same programs, built the same way, on a real GPU with
NVIDIA's own libraries and no shim, and fails when a program that the
simulator passes does not pass on the card.

## Where: server1's RTX 3080 Ti

server1 (192.168.1.221, 24 CPUs, 31 GB RAM) has an RTX 3080 Ti (sm_86,
driver 595), which is also the `nvidia/rtx3080ti` profile the simulator
measured. It also runs **TerraScoutX's production Postgres**, which serves that
site's live address search. On 2026-09-28 a `-j$(nproc)` build with CMake's
LTO (`-flto=auto`, one process per CPU per link) took the load to 140, the
host swapped, and search latency rose to 0.5-0.8 s. Everything below is shaped
by that.

## The job

A scheduled workflow (`card-check.yml`), on a self-hosted runner labelled
`[self-hosted, server1, rtx3080ti]`, nightly at a quiet hour for the search
site (e.g. 09:30 UTC) and by `workflow_dispatch`. Never on `pull_request` or
`push`: see [Security](#security).

Steps:

1. **Gate on load.** Refuse to start (and pass with a notice, so a busy night
   is not a red run) when the 5-minute load average is above 10 or Postgres has
   more than N active connections. `server1`'s `/tmp/simcheck.sh` already
   applies the load-10 rule by hand.
2. **Build the simulator, capped.** `RelWithDebInfo` with `-O2`, not
   `Release`: this tree's `Release` turns on interprocedural optimization, and
   the link fan-out is what overloaded the host. Then
   `nice -n 19 ionice -c3 taskset -c 0-5 cmake --build build -j6`. In numbers:
   at most 6 compile processes on CPUs 0-5, lowest CPU and I/O priority, about
   a minute for a full build. If LTO is ever wanted there, `-flto=2`, never
   `auto`.
3. **Build each card-portable e2e program twice from the same source**,
   with the same nvcc flags its `run_*.sh` uses (`-cudart shared`,
   `-arch=sm_86` or the script's `compute_XX`): one binary is all that is
   needed, since the shim is chosen at run time.
4. **Run it on the card**: no `LD_LIBRARY_PATH` to the shim, no `VGPU_*`
   variables, `CUDA_VISIBLE_DEVICES=0`, each under `timeout`, one at a time
   (the card is one GPU, and the desktop or other users may hold memory).
5. **Run it on the simulator** with `VGPU_GPU=nvidia/rtx3080ti`, exactly as
   its e2e script does.
6. **Compare.** A program fails the card check when
   - its own verdict differs (PASS on the simulator, not on the card), or
   - for programs marked output-exact, the two outputs differ after dropping
     lines listed as volatile (timings, addresses, driver version strings).
   The summary lists both kinds; the run fails on either.
7. **Report** to the step summary and keep both outputs as an artifact. On
   `main`, open or update one issue (`card-check-failing`) the way
   `workloads.yml` does, closed by the first passing night.

Everything runs as the runner's unprivileged user; nothing needs root. The
GPU must be visible to that user (`/dev/nvidia*`, group `video` or the
driver's device permissions).

### Which programs ("card-portable")

A program qualifies when it needs nothing the simulator alone provides:

- no `vgpu` CLI, `VGPU_*` knobs, fault injection (`vgpu fault arm`), RAS or
  telemetry registers, `vgpu smi` output, or multi-GPU topologies the card
  does not have (`VGPU_DEVICE_COUNT=2` tests are out: one 3080 Ti);
- no profile other than an sm_86 one, or code built for sm_86 or lower;
- a deterministic verdict (self-checking PASS/FAIL, or an exact expected
  output its script states).

That is the bulk of `nvidia/tests/e2e`: the PTX form sweeps
(`ptx_forms`, `ptx_half_forms`, `ptx_memory_forms`, `ptx_sweep`,
`ptx_warp_mem`, whose `*_expected.inc` tables are exactly what the card should
reproduce), `device_functions`, `device_intrinsics`, `atomic_cas`,
`textures*`, `border_colour`, `printf_formats`, `mma_*`/`wmma_*` at sm_80/86,
`cooperative_grid`, the graph tests, the library path tests (`blas_*`,
`rand_paths`, `fft_layouts`, `lt_paths`, `dnn_paths` where cuDNN is installed),
`deferred_errors`, `device_last_error`, `driver_abi`. The list should live in
the repository (`nvidia/tests/e2e/card-portable.txt`: program, nvcc flags,
arguments, `verdict` or `exact`, volatile-line patterns) so a new e2e test
opts in with one line, and a test that cannot be portable says why.

The external suites' lists (`ci/external/*.txt`) already hold an RTX 3060's
results; on server1 the same harness (`ci/external/suites.sh`) can rerun them
on the 3080 Ti with a `card` mode, which keeps those expected results honest
as drivers change.

### Toolkit

Two toolkits, so both of the shims' sonames are exercised against the real
libraries: CUDA 12.6 in user space (`~/work/sim-scratch/cuda-12.6` exists
today; Ubuntu's 12.0 nvcc rejects GCC 13) and CUDA 13.0. Each night uses one,
alternating, or both if the time allows. Record the driver version
(`nvidia-smi --query-gpu=driver_version`) in the report: a failure that starts
on the night the driver changed is drift, not a simulator regression.

## Limits on server1

- CPU: build at `-j6` on CPUs 0-5 (`taskset`), `nice -n 19`, `ionice -c3`;
  runs one program at a time. The runner's service unit adds
  `CPUQuota=600%`, `MemoryMax=8G`, `IOWeight=10` (systemd), so a mistake in a
  workflow cannot exceed them.
- Time: `timeout-minutes: 90` on the job; each program `timeout 300`.
- Disk: the workspace is cleaned after each job; build trees are not cached
  across nights (a capped build is a minute).
- No Docker builds, no `-j$(nproc)`, no `-flto=auto`, ever, on this host.

## Security

server1 is on a private network with production data. A self-hosted runner
there must never run code from a fork:

- Register it to a runner group that only allows `pantheongpu/pantheonsim`
  and only the workflow file `card-check.yml` ("selected workflows").
- `card-check.yml` triggers only on `schedule` and `workflow_dispatch`, and
  checks out `main` (or the dispatched ref, which only people with write
  access can dispatch).
- Keep "Require approval for all outside collaborators" on for fork pull
  requests, so no fork can add a `runs-on: [self-hosted, server1]` job and
  have it picked up.
- An ephemeral runner (`--ephemeral`, re-registered by its service after each
  job) leaves nothing from one run to the next.

## Enabling it

1. Create a runner user on server1 with access to the GPU; install the
   runner as a systemd service with the limits above, labels `server1` and
   `rtx3080ti`, in a runner group restricted as under Security.
2. Add `nvidia/tests/e2e/card-portable.txt` and a `card` mode to the run
   harness (a script that builds each listed program once and runs it with and
   without the shim), then `card-check.yml` as described.
3. Run it once by hand (`workflow_dispatch`) while watching server1's load and
   the search site's latency; then let the schedule take over.
