# Round 10b, workstream sweep: the exit abort, the "0.8 s wait" in `nn`, the memo set's memory

Rig: c7i.4xlarge (Sapphire Rapids), Release build of `speed-round10` (52fd1903) in `~/sweep` (candidate) and an unmodified copy of the
same build in `~/sweep-base` (A/B). Raw output: `docs/perf-design/data/round10/sweep/`. Nothing was run on the WSL machine, except that
`tests/` file below was first run once there by mistake (a 20-line pure-Python script, no torch, no simulator).

## 1. The exit abort (`terminate called without an active exception`, exit 134)

Cause (not ours in the simulator): after the last result is printed, `Py_FinalizeEx` runs while PyTorch's autograd worker thread
(`pt_autograd_0`) is still alive and takes the GIL; the process dies in `std::terminate`. It is a race, and it depends on the CPUs the process
may use.

Frequency on the unmodified build (baseline, `sweep-base`):

| how | runs | aborts |
| --- | --- | --- |
| quick sweep via `run_pytorch_sweep.sh`, 4 CPUs (2 cores), 5 sweeps, plus 5 on 4 other CPUs | 10 sweeps, ~13 groups each | 0 |
| groups fft / losses / nn run directly, 4 CPUs (2 cores), 30 runs each, two parallel jobs | 180 | 0 |
| same, `taskset` to ONE CPU (cpu 4), 30 runs each | 90 | 19 (nn 15 of 30, losses 3 of 30, fft 1 of 30) |
| same, ONE CPU (cpu 6), 30 runs each (repeat for the A/B) | 90 | 17 (nn 15, losses 2, fft 0) |
| same, two hyper-thread siblings (cpus 6,14) | 90 | 0 |
| quick sweep via `run_pytorch_sweep.sh`, ONE CPU, 5 sweeps | 5 | 4 sweeps hit it (losses 3, nn 1) |

So on a multi-CPU rig it did not reproduce at all in 190 sweeps-worth of group runs; it reproduces about half the time for `nn` on a single CPU
(and on the busy WSL VM, where it was first seen).

Fix: `nvidia/tests/pytorch/sweep.py` now ends with `os._exit(code)` after flushing stdout and stderr, the exit status being what
`sys.exit(harness.main(...))` gave (0; the returned int; `SystemExit`'s code; 1 plus a traceback for an uncaught exception; 1 for a text exit value).
`harness.main` returns 0 and reports failures through the `FAIL`/`XFAIL`/`XPASS` lines, so the group's status semantics are unchanged. What
finalisation did that is now skipped: nothing the sweep needs. No `atexit` handler of ours writes a file; the sweep does not use the profiler or CUPTI.
The simulator's own `atexit` handler (cuDNN peer-exchange directories, used by the `multi` group) is skipped too: checked with `--count 2`, the
`multi` group (full tier, 10 of 10 ok) leaves the same 4 entries in `/tmp/vgpu-nccl-<uid>` before and after the change, and stale directories are
swept by the next run anyway. The other entry points (ops.py, dist.py, distributed.py, profiler_check.py) were NOT changed: they never showed the abort
here and the profiler check does need its flush at exit.

Result (same 30-run protocol, ONE CPU, pinned to cpu 4 for the candidate and cpu 6 for the base, run at the same time):

| group | base aborts | candidate aborts |
| --- | --- | --- |
| fft | 0 of 30 | 0 of 30 |
| losses | 2 of 30 | 0 of 30 |
| nn | 15 of 30 | 0 of 30 |

And the quick sweep on ONE CPU, 5 times each: base 4 aborted groups (losses 3, nn 1); candidate 0. Every other line is identical in both: the
same 12 groups ok, `165 of 169 checks, 1 known failures` (the `compile` group fails on this rig in both, no Inductor toolchain).
Output of `nn` is identical (the `ok` lines, timings aside).

Test: `nvidia/tests/pytorch/test_sweep_exit.py` (ctest name `sweep_exit`; framework-free, needs neither torch nor a GPU, so it runs on hosted CI).
It copies `sweep.py` next to a fake harness and checks the seven exit-status cases, that buffered output is delivered through a pipe, and that
an `atexit` handler does NOT run. Against the unmodified `sweep.py` all seven cases FAIL ("an atexit handler ran: the interpreter was
finalised"); with the change all pass (also through `ctest -R sweep_exit` in the CI image).

Side effect, measured: skipping finalisation also saves about 0.6 s of CPU per group (`nn` 3.9 s wall against 4.5 s for a base run that does not abort).

## 2. The "~0.8 s wait" in `nn`

Not the simulator and not PyTorch's own code: it is the **core dump of the abort**. Evidence:

* On the rig, `nn` without the abort has wall time equal to user+sys (`3.89 s wall, 3.12 user, 0.77 sys, 99 %` pinned to one CPU; 4 CPUs and 8 CPUs the
  same, 100 to 108 %). Nothing waits. (The brief's "CPU 3.1 to 3.5 s" is the user time; sys adds 0.77 s, which is page-fault work on 1.6 GB of RSS: 388 k minor
  faults.)
* The runs that do wait are exactly the aborted ones: `5.0 s wall, 3.1 user, 0.75 sys, 77 %` (and `Command terminated by signal 6`), against
  `4.5 s, 3.7 user, 99 %` for the others, in 3 aborted runs out of 12 in this sample and half of the single-CPU runs. The user time of an aborted run is
  lower because it never finishes finalising.
* The core pattern on Ubuntu is `|/usr/share/apport/apport`; `/var/log/apport.log` shows it called for each of these processes, spending about 1 s
  per crash before the process is reaped (a `gdbus` call timing out, then writing a 330 MB crash file in `/var/crash`). `ulimit -c 0` does not stop
  apport from being called (the log says `core limit 0`). `strace -f -tt` shows the process already past its last write when the wait starts.
* The 3-of-11 fast runs the brief measured are the ones that did not abort.

With `os._exit` no run aborts, so the wait is gone (`nn`: 3.9 s wall on every one of 30 runs, against 4.4 s without an abort and 8.5 s average over the
single-CPU abort mix, where apport competes for the CPU). Nothing in the simulator sleeps, polls or joins with a delay at exit. The 0.87 s step at the start
of each `nn` run ("start Conv1d stride 2": first CUDA kernel launches, module load and SASS decode) is CPU work, not a wait.
The rig has a 330 MB `/var/crash/_usr_bin_python3.12.0.crash` left by these measurements; I did not delete it.

## 3. `sass::unsupported`'s per-thread set: memory

Method: a temporary probe (not committed) printed every table growth with the thread id; peak RSS from `/usr/bin/time`; the memo switched off with
`VGPU_SASS_REFUSE=NOSUCHOP` (same results, no table) for the A/B. RTX 5090 = sm_120, so one architecture. 12 quick-sweep groups in one process
(nn, ops, optim, attention, amp, sdpa, rnn, losses, fft, linalg, sparse, graphs; 160 checks ok), 4 CPUs, twice per mode:

| mode | wall | peak RSS |
| --- | --- | --- |
| memo on (default) | 13.07 to 13.08 s | 3,551,056 / 3,550,916 KB |
| `VGPU_THREADS=4` | 13.07 to 13.17 s | 3,550,700 / 3,550,876 KB |
| memo off | 17.03 to 17.07 s | 3,501,616 / 3,501,744 KB |

* The memo costs **+49 MB peak RSS (1.4 % of 3.5 GB)** and saves **4.0 s of 17.05 s (23 %)** on this process. 15 table growths, in 2 threads: one thread reached
  the 32 MB cap (2^21 slots, over 524 k distinct words), the other 16 MB. `VGPU_THREADS=4` changes nothing: the simulator's worker threads do not load
  modules; only the threads that call into the CUDA runtime (PyTorch's main and one helper) do.
* A multi-threaded loader (Python threads each launching different kernels, 10 operators per thread, `/tmp/mt.py`-style; peak RSS memo off to on):
  1 thread 604 to 621 MB (+17 MB, biggest table 16 MB), 4 threads 616 to 632 MB (+16 MB, 4 tables, biggest 8 MB), 8 threads 615 to 649 MB (+34 MB, 5 tables,
  biggest 8 MB). Module loading is done once per module, so more threads split the same words rather than each holding all of them.
* Worst case by construction: 32 MB per architecture per loading thread (a process on several architectures with many loading threads); NOT MEASURED
  beyond the above.

Recommendation: **keep it per-thread, as is.** The measured cost is tens of MB against a gigabyte-scale process and the set sits on the hot path of every
module load (about 10^7 lookups in this sweep); a shared set would need a lock or an atomic table on that path, would save at most the duplicated words between
two tables (somewhere under 32 MB here), and its extra cost was NOT MEASURED because per-thread is not a measured problem. If a process ever shows many
loader threads on many architectures, lowering `kMaxSlots` is the cheaper first step.

## What was dropped

Nothing was dropped. The changes not made: no change to ops.py / dist.py / distributed.py / profiler_check.py (no abort seen, the last needs its
flush); no shared memo set (not a measured problem).
