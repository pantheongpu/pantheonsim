# Round sp67: the two A10G observations of #570, from logged card runs

Rig: the A10G of round sp65 (i-052e0df62d297f557, g5.xlarge, AMD EPYC 7R32 4 vCPU, driver 595.112.07 with persistence mode on, Ubuntu 24.04.5).
Simulator: `hostgpu-relaxed` f55c8e65 built on server3 (`Release`, no LTO) and copied (the shim libraries are self-contained). Every run
`VGPU_GPU=nvidia/a10g VGPU_THREADS=4 VGPU_QUIET=1`, `nice -n 19`, through `logged_run.py` (stderr lines stamped with seconds since the start).
**Round sp65 was timing its own runs on the same 4 vCPUs at the same time**: the times below are of a shared machine; the counts and bytes are not
affected. Raw stamped logs are not kept here (800 KB each); the summaries are.

## 1. GRAMSCHM in exact mode (`VGPU_HOST_GPU=exact`; second run with `VGPU_HOST_GPU_DIV=1`)

| | exact | exact + DIV |
|---|---:|---:|
| wall s | 282.3 | 271.9 |
| "GPU Time" (the device part, printed by the program) s | 50.50 | 52.14 |
| "CPU Time" (its CPU reference) s | 231.6 | 219.6 |
| `gramschmidt_kernel1` (2,048 launches) | refused: `MUFU.RSQ R7, R0` (the `sqrtf` idiom; "not the division idiom: a CALL with no FCHK before it") | same |
| `gramschmidt_kernel2` (2,048) | refused: `MUFU.RCP` (the division idiom, needs DIV) | admitted, all kept for their size (sim 0.16 ms against a card path of 4.3 ms) |
| `gramschmidt_kernel3` (2,048) | 2,045 on the card, 3 in the simulator | 2,043 on the card |
| MiB pushed / pulled | 98,160 / 98,128 | 98,064 / 98,032 |
| = per card launch | 48.0 MiB each way: all live memory (A, R, Q, 16 MiB each) | same |
| card path, sum (median per launch) | 25.3 s (11.3 ms) | 26.0 s (13.7 ms) |
| of which the kernel, median per launch | 1.53 ms | 1.58 ms |
| between two card launches, less the card path: the pull of 48 MiB + kernel1 + kernel2 in the simulator, sum (median) | 24.7 s (11.2 ms) | 25.7 s (13.3 ms) |
| simulator time of the launches it ran (4,099 / 4,101) | 1.32 s | 1.38 s |
| so the pulls | about 23.4 s (11.4 ms per 48 MiB, about 4.2 GB/s) | about 24.3 s |
| the size rule's estimate for kernel3, sum | 151 s | 109 s |

Per iteration k: kernel3 runs on the card and leaves all 768 chunks Card; kernel1 is refused for its code, so before the simulator runs it
`reset_for_simulator` pulls every Card chunk (48 MiB) and makes them Host; kernel2 runs in the simulator; kernel3 then pushes all 48 MiB again.
The card path saves kernel3's simulator time (about 22 ms per launch on average: Round sp44's plain device part, 46 s over 2,048 iterations) and
spends about 23 ms on the two copies, so the device part is no faster than the simulator's (Round sp44: 46.06 s plain, 43.64 s exact, one run each on another A10G), and the
CPU reference (about 220 to 230 s) hides the rest. The traffic ban (`kTrafficLaunches`) does not fire: kernel3's card path (11 ms) is under half its
estimate (74 ms per launch on average here, a contended rig), and the pull is not part of its card path at all.

## 2. The driver start on long programs (`VGPU_HOST_GPU=exact`, default start cost against `VGPU_HOST_GPU_START_COST=0`)

`drvstart.py` (the calls `begin_start` makes, timed in a fresh process, three runs): dlopen 1.5 ms, `cuInit` 48.6 to 49.5 ms, primary context
97.8 to 103.4 ms, the 1 TiB reservation 0.08 ms: **148 to 155 ms** in all (persistence mode on; sp42 measured 0.23 s on another rig). The module
load is in the first card path (FDTD-2D's first launch 19.2 ms against 11.5 ms for the next with the same push).

| program | form | first card launch at s | launches card / simulator | MiB pushed / pulled | simulator time of its launches s | wall s |
|---|---|---:|---|---|---:|---:|
| FDTD-2D | default | 0.529 | 1494 / 6 | 144 / 112 | 0.66 | 6.35 |
| FDTD-2D | START_COST=0 | 0.185 | 1494 / 6 | 96 / 64 | 0.46 | 5.19 |
| babelstream | default | 5.112 | 1188 / 26 | 2304 / 1152 | 13.44 | 17.97 |
| babelstream | START_COST=0 | 3.370 | 1188 / 26 | 3072 / 1920 | 11.67 | 16.31 |

(one run each, on the shared rig: the wall and simulator times are not a comparison; the counts are.)

## Commands

```
export VGPU_GPU=nvidia/a10g VGPU_THREADS=4 VGPU_QUIET=1 LD_LIBRARY_PATH=~/sp67-build/shim
python3 logged_run.py --raw gram-exact.log --env VGPU_HOST_GPU=exact -- ./GRAMSCHM
python3 logged_run.py --raw gram-div.log --env VGPU_HOST_GPU=exact --env VGPU_HOST_GPU_DIV=1 -- ./GRAMSCHM
python3 logged_run.py --raw fdtd-default.log --env VGPU_HOST_GPU=exact -- ./FDTD-2D       # and --env VGPU_HOST_GPU_START_COST=0; babelstream the same
python3 drvstart.py
```
