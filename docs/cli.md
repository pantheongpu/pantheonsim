# Command-line reference

Everything the `vgpu` command and the environment variables can do. The
[README](../README.md) has the short path (`make`, then `./run`) and [getting-started.md](getting-started.md) the step-by-step one (`make doctor`, `make build`, `make demo`); this page is the full list.

Build once (`make build`, which runs `./scripts/build.sh`) and the command is `build/vgpu`. The build
also puts it at `build/bin/vgpu`, next to drop-in `nvidia-smi`, `rocm-smi` and
`amd-smi`, so `PATH=$PWD/build/bin:$PATH` makes all of them available.
`vgpu --help` lists the commands. `run`, `test`, `ncu`, `debug`, `fault`, `regs`
and `smi` take `--help` for their own options; `shell` and `serve` do not, so
their options are listed below.

Settings you use every time can live in a file (`pantheonsim.yaml`, written by
`vgpu config init`) instead of flags and variables: see [config.md](config.md).

## Commands

| Command | What it does |
| --- | --- |
| `vgpu list-gpus` | List the built-in virtual GPU models. |
| `vgpu info --gpu <vendor/model> [--json]` | Show one device profile. |
| `vgpu demo vectoradd [--gpu <id>] [-n <elems>]` | Run a built-in `vectorAdd` kernel end to end, as a smoke test. |
| `vgpu run [options] <program> [args...]` | Run a program on a simulated GPU. [Options](#vgpu-run). |
| `vgpu config show\|init\|keys` | The settings file: what is in effect, a commented starter file, the list of keys. See [config.md](config.md). |
| `vgpu shell` | An interactive simulated machine: `nvidia-smi`, `rocm-smi`, `lspci` and `dmesg` all answer for the GPUs you picked. [Options](#vgpu-shell). |
| `vgpu test --matrix <program> [args...]` | Run a program on every device profile and compare the results. [Options](#vgpu-test---matrix). |
| `vgpu check [options] <program> [args...]` | Run a program on this machine's real NVIDIA card and on the simulator, and compare. [Options](#vgpu-check). |
| `vgpu ncu [options] <program> [args...]` | Nsight Compute's command line, answered from the simulator's exact counters. [Options](#vgpu-ncu). |
| `vgpu counters` | What performance counters are reported, and which cannot be and why. |
| `vgpu debug [-x FILE] -- <program>` | Run an AMD program under the kernel debugger. [Commands](#vgpu-debug). |
| `vgpu fault ...` | Inject ECC and PCIe errors, hangs, a lost GPU or a degraded link, to test health tools. [Synopsis](#vgpu-fault). |
| `vgpu regs ...` | Read, write and dump a device's registers, with an access log. [Synopsis](#vgpu-regs). |
| `vgpu smi [options]` | A live view of running virtual GPUs. It is a drop-in `nvidia-smi`, so it takes `nvidia-smi`'s own options (`vgpu smi --help`), plus `--csv`, `--rocm`, `--agents`, `--lspci`, `--lspci-dump` and `--explain`. |
| `vgpu serve --gpu <id> [--count N] [--load 0..1] [--alloc-mb N]` | Present a virtual rack so monitoring tools have something to watch. Runs until stopped, so start it in the background. |
| `vgpu --version`, `vgpu --help` | Version and usage. |

A GPU is named `<vendor>/<model>`, such as `nvidia/h100`, `nvidia/t4`,
`amd/mi300x`. `vgpu list-gpus` prints every id.

## `vgpu run`

```
vgpu run [options] <program> [args...]
```

Puts the simulator's CUDA libraries in front of the real ones and runs the
program in place, so its exit code and signals are its own. Everything after
the program name belongs to the program; `--` ends the options early.

For a CUDA program, build it with the shared runtime first
(`nvcc -cudart shared`), so the loader can substitute the simulator's
`libcudart`. Before it starts the program, `vgpu run` reads its dynamic section
and says so if a CUDA library it needs is not in this build (a CUDA 12 program
against a CUDA 13 build), or if a `DT_RPATH` would hide the simulator (see
`--preload`).

**Device**

| Option | Meaning |
| --- | --- |
| `--gpu <vendor/model>` | The device to present. Default: `$VGPU_GPU`, else `nvidia/h100`. |
| `--count <n>` | How many identical devices. |
| `--vram-mb <n>` | VRAM per device, in MiB, overriding the profile. |

**Execution**

| Option | Meaning |
| --- | --- |
| `--threads <n>` | Host threads that run blocks, at least 1. Default: one per core. |
| `--max-steps <n>` | Per-thread instruction budget, a guard against a kernel that never ends. `0` turns the guard off. |
| `--strict` | Refuse undefined behaviour instead of continuing. See [diagnostics](diagnostics.md). |
| `--race` | Detect shared-memory data races. See [diagnostics](diagnostics.md). |
| `--counters` | Collect exact per-kernel instruction and memory counters. See [diagnostics](diagnostics.md). |

**Output**

| Option | Meaning |
| --- | --- |
| `--trace` | Log driver-level activity. |
| `--quiet` | Silence every simulator message on stderr, errors and warnings included (sets `VGPU_QUIET=1`). The program's own output and exit code are unaffected. |

**Loading**

| Option | Meaning |
| --- | --- |
| `--preload` | Also `LD_PRELOAD` the shim. Needed when the program carries a `DT_RPATH` naming a directory of the real CUDA libraries: the loader reads that old-style tag *before* `LD_LIBRARY_PATH`, so the simulator never gets in front. (A `DT_RUNPATH`, which modern toolchains emit, comes after and is harmless.) `vgpu run` tells you when it applies. For an interpreter such as `python` with PyTorch, which names no CUDA library itself, every CUDA library the shim carries is preloaded; the driver's libraries and NVRTC among them are loaded privately (`RTLD_LOCAL`) as the preloaded runtime starts (`VGPU_PRELOAD_PRIVATE`), so that a library keeping their functions in exported pointers of the same names (ThrustRTC) does not have those pointers bound to the shim's code. |
| `--print-env` | Print the environment and the command, then exit. |

Examples:

```bash
vgpu run ./my_app                               # the default GPU (nvidia/h100)
vgpu run --gpu nvidia/t4 ./my_app               # a T4
vgpu run --gpu nvidia/h100 --count 8 ./my_app   # eight of them
vgpu run --race --strict ./my_app               # look for races and undefined behaviour
vgpu run python train.py                        # PyTorch, unmodified
```

## `vgpu test --matrix`

```
vgpu test --matrix [options] <program> [args...]
```

Runs the program once per device profile and compares the output. The first
profile is the baseline; the others are reported as identical to it or
different from it.

| Option | Meaning |
| --- | --- |
| `--gpus <a,b,c>` | Only these profiles. Default: every verified NVIDIA one. |
| `--all` | Include unverified profiles too. |
| `--show-diff` | Print the output of the first profile that differs. |

Exit status: `0` every profile matched the baseline; `1` `vgpu` itself failed
(an unknown profile name, say); `2` usage error; `3` results differ; `4` the
program did not run on the baseline profile, so there is nothing to compare
against.

## `vgpu check`

```
vgpu check [options] <program> [args...]
```

Runs the program on the real GPU of this machine (NVIDIA's own libraries) and on the simulator, with the profile of that same card,
and compares what it printed. It audits the simulator: nothing in the launch path changes. Build the program with the shared runtime
(`nvcc -cudart shared`) so `vgpu run` can put the simulator's `libcudart` in front of the real one on the simulated side.

The card is run twice first. Lines that differ between those two runs are the card's own noise (a time-seeded random number, an
uninitialised value) and are left out of the comparison, and the report says how many. Timing lines are dropped; lines that name the
machine (device banner, driver and runtime versions) are compared apart, so a `Driver Version` difference is reported as an identity
difference and not as a different answer. A line that states an outcome (`PASSED`, `FAILED`, `error`) is never dropped as a timing line;
its numbers are masked instead. `--show-ignored` lists every dropped and set-aside line.

| Option | Meaning |
| --- | --- |
| `--gpu <vendor/model>` | The profile to simulate. Default: the one whose model is the card's name. |
| `--allow-cross` | Compare even when the profile is not the card. The two are then different machines, and the report says so. |
| `--card <name>` | The card's name, when `nvidia-smi` cannot be used. |
| `--native-runs <n>` | Runs on the card. Default 2; 1 skips the noise check. |
| `--sim-runs <n>` | Runs on the simulator. Default 1; 2 also checks that it repeats. |
| `--timeout <s>` | Per run. Default 600. |
| `--tolerance <rel>` | Accept numbers within this relative difference. Default 0: any different digit is a difference. |
| `--file <path>` | Also compare a file the program writes, byte for byte. Repeatable. |
| `--show-diff` | Print the first differing pair of lines. |
| `--show-ignored` | List the lines dropped as timing and the lines compared apart as identity. |

Exit status: `0` the simulator agrees with the card; `1` `vgpu` itself failed; `2` usage error; `3` the outputs, exit codes or files differ;
`4` the program did not run on the card (so there is nothing to compare); `5` inconclusive, the card's two runs cannot be lined up;
`6` no usable card, or the profile is not that card.

What it cannot tell: a data race or an uninitialised read gives one valid answer on the card and another in the simulator, and a
program that prints only `Test passed` hides every number behind it. A match means "printed the same", not "computed the same bits".

## `vgpu shell`

```bash
vgpu shell                       # asks which GPU, CUDA/driver and OS
vgpu shell -y --gpu amd/mi300x --count 8 --rocm 6.2.0 --driver 6.3.6 \
    --os ubuntu:24.04 --hostname mi300-node --load 0.65
```

| Option | Meaning |
| --- | --- |
| `--gpu`, `--count 1..16`, `--vram-mb` | The rack. |
| `--cuda`, `--rocm`, `--driver` | Reported toolkit and driver versions. The CUDA and driver defaults are the ones this build's runtime carries; the ROCm default is `6.2.0`. |
| `--os ubuntu:22.04\|ubuntu:24.04\|rocky\|debian`, `--kernel`, `--hostname` | Machine identity. |
| `--load 0..1` | Background load, so the telemetry columns move. |
| `-c`, `--command <cmd>` | Run one command in the machine instead of an interactive shell. |
| `-y`, `--no-prompt` | Take the defaults for anything not given, without asking. |
| `--no-isolate` | Skip the namespaces (session tools only). |
| `--rootfs` | Give the session its own root filesystem instead of the host's: the system read-only from the host, everything else empty and thrown away on exit. Needs user namespaces; where the host refuses them the shell exits `1` rather than start on the host's files. See [machine-simulator.md](machine-simulator.md#a-private-root---rootfs). |
| `-v`, `--volume`, `--mount HOST:DEST[:ro]` | With `--rootfs`: mount a host file or directory at `DEST` in the session, as `docker run -v` does. Read-write unless `:ro`. Repeatable. `HOST` must exist and `DEST` must be absolute and not under `/proc` or `/sys`, or the shell exits `2` before starting. Without `--rootfs` it is an error. |
| `--home DIR` | With `--rootfs`: mount the host directory `DIR` as the session's `/root` (its `$HOME`), so it persists between runs. Without it `/root` is empty each time. |

`./run [config.yaml] [-- command]` at the root of the repository starts this shell
for you (see [machine-simulator.md](machine-simulator.md#start-it-with-run)).
More in [machine-simulator.md](machine-simulator.md).

## `vgpu ncu`

```
vgpu ncu [options] <program> [args...]
vgpu ncu --query-metrics [--query-metrics-mode base|suffix|all] [--devices N]
```

| Option | Meaning |
| --- | --- |
| `--csv` | Report as CSV. |
| `--metrics a,b,...` | Metrics to report. Default: every one. |
| `--set NAME`, `--section NAME` | Accepted, but sections need a timing model and are not produced; only the metrics are reported. |
| `-c`, `--launch-count N` | Report the first N matching launches. |
| `-s`, `--launch-skip N` | Skip the first N matching launches. |
| `-k`, `--kernel-name NAME` | Only launches of this kernel (`regex:...` for a regex). |
| `--devices N[,M]` | Only launches on these devices. |
| `--log-file PATH` | Write the report there instead of stdout. |
| `--query-metrics` | List the metrics this can report, with what each is. |
| `-o`, `--export` | Refused: `.ncu-rep` is Nsight Compute's own format. |

Anything that needs cycles, stalls or cache behaviour is not listed, because
there is no timing model. `vgpu counters` says what is counted and why the rest
is not.

## `vgpu debug`

```
vgpu debug [-x FILE] [vgpu run options] -- ./program [args]
```

Runs an AMD program with the kernel debugger on. Commands come from `FILE` (`-x`)
or from a `(vgpu)` prompt on the terminal: `break KERNEL[+OFFSET]`, `continue`,
`step [N]`, `where`, `disas [N]`, `info registers`, `print[/x|/d|/f] REG`,
`lane N`, `x/N ADDR`, `x/N lds:ADDR`, `set REG[LANE] = VALUE`, `delete [N]`,
`quit`. It runs on one host thread.

## `vgpu fault`

```
vgpu fault inject [--gpu N] --ecc corrected|uncorrected [--location LOC] [--count N]
vgpu fault inject [--gpu N] --pcie COUNTER [--count N]
vgpu fault arm    [--gpu N] --ecc corrected|uncorrected|--bitflip [--count N]
                  [--on load|store|shared|alu|copy] [--rate P [--seed S]]
vgpu fault arm    [--gpu N] --hang [--seconds S]
vgpu fault stuck  [--gpu N] --offset BYTES --bit B --value 0|1 | --clear
vgpu fault lose   [--gpu N] [--clear]
vgpu fault link   [--gpu N] [--gen G] [--width W] | --clear
vgpu fault throttle [--gpu N] --reason R[,R...] [--seconds S] | --clear
vgpu fault show   [--gpu N]
vgpu fault reset  [--gpu N] --volatile|--aggregate
```

Injects errors into the machine that `nvidia-smi` and `rocm-smi` describe, so
health tools can be tested against them. Without `--gpu`, every GPU is
affected. `vgpu fault --help` explains each option.

## `vgpu regs`

```
vgpu regs list   [--space S] [--status done|model]
vgpu regs read   [--space S] [--gpu N] REGISTER [--size 1|2|4]
vgpu regs write  [--space S] [--gpu N] REGISTER VALUE [--size 1|2|4]
vgpu regs dump   [--space S] [--gpu N] [--extended]
vgpu regs log    [--space S] [--gpu N] [--last K]
vgpu regs export [PROFILE...] [--out DIR]
```

`--space` is `config` (default), the PCI configuration space, or `mmio`, an AMD
GPU's registers behind BAR5. More in [registers.md](registers.md).

## Environment variables

Many of these can also be set from the [settings file](config.md), which sets
the variable for you unless it is already set. The `vgpu run` options set most of these for you; set them directly when you run
a program without `vgpu run`, such as a driver-API program or a script. Each
variable is unset by default.

**Device**

| Variable | Meaning | Default |
| --- | --- | --- |
| `VGPU_GPU` | Virtual GPU profile id. | `nvidia/h100` |
| `VGPU_DEVICE_COUNT` | Number of identical virtual devices. | `1` |
| `VGPU_VRAM_MB` | Virtual VRAM size, overriding the profile. | profile |
| `VGPU_CUDA_VERSION`, `VGPU_DRIVER_VERSION` | CUDA and driver version strings that `nvidia-smi` and NVML report. | this build's |
| `VGPU_ROCM_VERSION` | ROCm version string that `amd-smi version` reports. | `6.2.0` in `vgpu shell`, else `N/A` |
| `VGPU_SHELL_NO_NAMESPACES` | `1` makes `vgpu shell --rootfs` behave as on a host that refuses user namespaces (exit `1`, no session). A test hook; nothing else reads it. | unset |

**Memory**

| Variable | Meaning | Default |
| --- | --- | --- |
| `VGPU_MEMORY_RAM_MB` | Host RAM that device memory may use before the rest goes to `VGPU_MEMORY_DIR`. | unlimited |
| `VGPU_MEMORY_DIR` | Directory for device memory past `VGPU_MEMORY_RAM_MB`. A card larger than the machine's RAM runs, at the speed of the disk. | never spill |
| `VGPU_MEMORY_HOST_RESERVE_MB` | Host memory the guard leaves to the rest of the machine. When the cgroup's headroom or `MemAvailable` falls below it, device memory spills to `VGPU_MEMORY_DIR` or, with none, the allocation fails with `cudaErrorMemoryAllocation` / `hipErrorOutOfMemory` rather than the kernel's OOM killer firing. | 6% of the memory limit in play, 1 to 4 GiB (at most a quarter of a small limit) |
| `VGPU_MEMORY_HOST_GUARD` | `0` turns the host-memory guard off. | on |

**Execution**

| Variable | Meaning | Default |
| --- | --- | --- |
| `VGPU_THREADS` | Host threads used to run blocks. | one per core (`make run` and `./run` set half the cores unless this or `threads:` is set) |
| `VGPU_ASYNC_LAUNCH` | A kernel launch on a `cudaStreamNonBlocking` stream returns before the kernel has run (`cudaEventQuery` on work behind it is `cudaErrorNotReady`). `all` does the same on streams made with `cudaStreamCreate`; `0` runs every launch inside its call. Never used with a profiler, `VGPU_TRACE`, a managed allocation or a CUDA library shim loaded (docs/perf-design/streams.md). | non-blocking streams |
| `VGPU_EXACT_THREADS` | `N`, beside `VGPU_THREADS=1` only: the blocks of a launch still leave the bytes, counters and errors of running in order, and a launch is run on `N` host threads when a trial under a shadow of device memory shows that no block touched what another wrote; any other launch is undone and run in order ([exact-threads.md](exact-threads.md)). SASS engine; the PTX and AMD engines run in order as before. | unset (in order on one thread) |
| `VGPU_EXACT_LOG` | `<file>`: appends a line for each launch `VGPU_EXACT_THREADS` applied to (kernel, blocks, threads, `committed`, `stopped` or `in order`, the reason, bytes kept for undoing, 4 KiB pages touched); a trial that a note by page stopped and that was tried again by word has a line with `again` before it. | unset |
| `VGPU_EXACT_PAGES` | `0`: a trial of `VGPU_EXACT_THREADS` notes every page of device memory by word from its first touch (the note the mode began with), instead of noting pages that blocks only read as a whole ([exact-threads.md](exact-threads.md)). The results are the same either way; for measurements. | pages noted as a whole |
| `VGPU_EXACT_STATS` | `<file>`: appends a line for each trial with what its notes cost (pages taken by one reader and by several, pages noted by word, words looked at and taken by compare-and-swap, pieces and bytes kept, accesses answered from a worker's memory). | unset |
| `VGPU_LOCKSTEP` | `0` runs a grid's blocks one after another on each host thread, as before; `all` runs every launch with the blocks the device holds resident together taking turns; the default runs kernels with a loop that polls global memory that way (see [cooperative.md](cooperative.md)). | polling kernels |
| `VGPU_LOCKSTEP_LOG` | `1` prints a line on stderr for each launch that runs its blocks together; any other value is a file the lines are appended to (the tests count the kernels that do). | off |
| `VGPU_MAX_STEPS` | Raises (or lowers) the runaway-kernel step budget; `0` removes it. The PTX engine reads it at each launch; the SASS engine reads it once per process. | built-in |
| `VGPU_SASS` | `0` forces the PTX engine; `1` insists on SASS and does not fall back. | SASS when the binary carries it, else PTX |
| `VGPU_SASS_LOG` | `1` says when a fatbin falls back from SASS to PTX. | off |
| `VGPU_SCHEDULER` | Warp scheduling order: `deterministic`, `random` or `adversarial`. Random and adversarial orders find bugs that depend on which warp runs first, and they run blocks one at a time so an order can be replayed. **PTX engine only:** kernels that run from SASS (the default whenever the binary carries it) use one fixed warp order, so set `VGPU_SASS=0` to use this. | deterministic |
| `VGPU_SCHEDULER_SEED` | Seed for `random` and `adversarial`; the same seed replays the same execution. | `0` |
| `VGPU_PTX_CONTRACT` | `0` runs multiply and add as written, rounding each, instead of fusing them. | fused |
| `VGPU_FASTPATH` | `0` sends every instruction down the interpreter's general path, for ruling out its fast paths when a result looks wrong (they are tested to give the same bits). | on |
| `VGPU_LOOKUP_CACHE` | `0` makes every kernel memory access look its address up in the allocation table, for ruling out the per-thread memory of recent lookups when an access looks wrong (tested to give the same values and errors) | unset (on) |
| `VGPU_LAUNCH_CACHE` | `0` makes every launch work out again what it can remember (the table of every loaded kernel that a device-side launch looks its child up in, and which function a kernel handle names), for ruling the remembering out when a result looks wrong; the answers are tested to be the same either way | unset (on) |
| `VGPU_MEMO` | `1` turns on launch-level result caching (opt-in; [design](perf-design/03-result-memoisation.md), [rules and measurements](perf-design/rounds/10-rounds-sp19-sp27.md#round-sp35-launch-level-result-caching-built-tested-measured-branch-sp35-memo-base-originmain-9f80d40e)): a launch whose kernel, grid, arguments, environment and device-memory history have been seen before is not run, its stored write set, counters and kernel digest are applied instead. The history is a hash chain over what the program did to the device (writes with their bytes, fills, copies, allocations, frees, launches); anything the cache cannot describe (virtual memory management, IPC, pinned or managed memory a kernel can reach, device malloc and printf, cooperative and cluster launches, textures, atomics on several threads, a profiler, the race detector, a trace, an armed fault, ...) restarts the chain from a value no other run has. Off, nothing is attached and results are bit for bit what they were | unset (off) |
| `VGPU_MEMO_DIR` | Directory in which entries persist between runs, so a second run of a deterministic program hits from its first launch (one file per entry, written with a rename, least recently used removed past `VGPU_MEMO_MAX_MB`). Without it entries live in memory for the run | unset |
| `VGPU_MEMO_MAX_MB` | Size the entries in `VGPU_MEMO_DIR` may add up to; the least recently used go first | `4096` |
| `VGPU_MEMO_MEM_MB` | Size of the entries kept in memory (least recently used out first) | `512` |
| `VGPU_MEMO_ENTRY_MB` | The largest write set one entry may hold; a launch that writes more runs and is not stored | `256` |
| `VGPU_MEMO_BUDGET` | What capturing write sets (a before and after hash of the allocations a launch can write) may cost, as a share of the kernel time of the launches that ran; past it a launch simply runs, unrecorded. `inf` records every miss whatever it costs (measurement and tests) | `0.03` |
| `VGPU_MEMO_EVENT_CREDIT_MS` | Hashing the bytes of host writes (every `cudaMemcpy` to the device is an event of the history with the hash of its bytes) is part of the same budget: free up to this many milliseconds, then at most the `VGPU_MEMO_BUDGET` share of the kernel time. A write that does not fit is not hashed: the history is poisoned instead (nothing before it can match anything after it; the launches still run and are never wrong, only the hits after it are lost). Past the credit the hashing also eats the pool that recording draws on. `events_skipped` and `event_ms` are in the statistics | `8` |
| `VGPU_MEMO_VERIFY` | `1` runs every hit for real as well and aborts if its write set or counters differ from the stored ones; `2` counts the differences and goes on (see `verify_mismatch` in the statistics) | `0` |
| `VGPU_MEMO_STATS` | `1` prints a line of what the cache did (launches, hits, entries, why launches were not cached, what it cost) to stderr at exit; `VGPU_MEMO_STATS_FILE=<file>` appends it to a file | unset |
| `VGPU_MEMO_FINAL_HASH` | `<file>`: at exit appends a line per device with a hash of all its live memory, with the cache on or off (what the differential runs compare) | unset |
| `VGPU_MEMO_TEST_MUTANT` | A deliberately wrong rule of the cache, named in `tests/unit/test_memo.cpp`; for the tests that must catch it. Never set it otherwise | unset |
| `VGPU_SIMD` | `base` or `avx2` caps the vector instruction set the lane arithmetic, half-precision conversions and tensor-core kernels use (the default is the best the CPU has: the 512-bit tile kernels of the tensor-core routes where it has AVX-512 F/BW/DQ/VL and VNNI, about 8% faster over the forms measured and up to 2x on the integer ones, and the 512-bit kernel of binary64 add, multiply and fused multiply-add, 3 to 39% faster on the programs measured that compute in `double`, the same bits; `avx512` asks for them explicitly); `off` sends the engines back to their scalar code, for ruling the vector kernels out when a result looks wrong (they are tested to give the same bits) | best available |
| `VGPU_SASS_CACHE` | `0` turns off the file in which the SASS engine keeps, between processes, which cubins it found to run in full (the check that decides between SASS and the fatbin's PTX decodes every instruction of a cubin; a process that finds one passes appends a 128-bit hash of it to `$XDG_CACHE_HOME/pantheonsim/sass-ok-v1-<build id>`, else `~/.cache/pantheonsim/...`, a file per build of the simulator, and a later process answers from it). Refusals are never kept, `VGPU_SASS_REFUSE` bypasses the file, and a file or directory that is not yours alone (another owner, group- or world-writable, a symlink, not a regular file) is ignored. There is no file, and nothing is written, when neither `XDG_CACHE_HOME` nor `HOME` is an absolute path, when the directory cannot be made or written, or when the build has no GNU build id; a full disk or a cut file only stops it growing (at most 65,536 hashes, 1 MB). The answers are tested to be the same either way; a PyTorch sweep group that finds the file warm runs 12% to 36% fewer instructions. Old builds' files stay in the directory and can be deleted any time | on |
| `VGPU_TIER` | the SASS engine runs its straight-line instructions through a pre-decoded tier unless this is `0`: each instruction's form and operands are worked out once per code section instead of at every execution (about 1.2 to 2.7 times faster on the benchmark workloads, [performance](performance.md)). Results, counters, errors and the warp scheduling are those of the interpreter, which also runs everything the tier has no handler for; `VGPU_TIER=0` (the interpreter alone, for ruling the tier out when a result looks wrong), `VGPU_FASTPATH=0`, `VGPU_SIMD=off` and `VGPU_SASS_TRACE` turn it off. `VGPU_TIER_STATS=1` prints how many warp instructions it ran and which ones it left to the interpreter when the process exits (`2` adds a few disassembled examples of each; `VGPU_TIER_STATS_FILE=<path>` appends it to that file instead of stderr). `VGPU_TIER_OFF=iadd3,bsync,...` leaves the forms it names (`tier::form_name`) to the interpreter, to find which form a difference comes from. `VGPU_TIER_CHAIN=0` makes it return to the scheduler after every run of straight instructions, as the interpreter does (by default it goes on by itself while the next call would pick the same lanes again: the same instructions run in the same order, and a warp's turn is counted in the same units). | on |
| `VGPU_HOST_GPU` | `exact` (experimental, [spike](perf-design/09-host-gpu-spike.md)): launches whose kernel is in the exact set (integer and IEEE arithmetic, no SFU, no float or compare-and-swap atomics) and that the size rule says are worth it run on the machine's real NVIDIA GPU, which must be the simulated model; the simulator's memory is copied there at the same addresses and back, so the results are the simulator's bits (checked with `VGPU_HOST_GPU_VERIFY=1`, which also runs every such launch in the simulator and compares; `all` includes kernels outside the set). Refused with a message when a diagnostic that needs the simulator is on. `VGPU_HOST_GPU_LOG=1` prints each decision; the exit summary says how many launches ran on the card. Counters, faults and the simulator's checks do not apply to those launches. `turbo` (experimental, [design](perf-design/12-host-gpu-turbo.md)) also sends kernels whose result is the card's own and not the simulator's bits: the approximate functions (every `MUFU`), packed half arithmetic, float reductions, calls inside the kernel, tensor-core instructions and integer atomics that return a value. The program then gets the card's values, usually one or two units in the last place from the simulator's, which a program can amplify (a comparison that goes the other way, an iteration count). Since Round sp66 it also sends float atomics with a returned value, compare-and-swap and exchange (`VGPU_HOST_GPU_ATOMICS=0` keeps them out) and kernels that call `printf` with an immediate absolute call (not yet nvcc 13.0's call through a register; `VGPU_HOST_GPU_TURBO_OFF=printf` keeps them out; the card's lines are printed when the launch is waited for, after what the program printed before it). Texture and surface instructions and instructions nobody named stay in the simulator. In both modes, since Round sp66, a kernel is no longer refused because its module has `__device__` or `__constant__` variables (the card is given a copy of the module whose references to them are the simulator's addresses, and the constant banks are filled from the simulator's before each launch), because its module has a function it does not call itself, because the device has a texture object, or because it is launched by a CUDA graph; a kernel that calls `malloc`, `free`, `assert` or launches kernels itself stays in the simulator, and so does every kernel of a module with `__managed__` variables. Every other safety check and the size rule are the same in both modes. A value that is not `off`, `0`, `exact` or `turbo` is said and read as off | off |
| `VGPU_HOST_GPU_VERIFY` | with `VGPU_HOST_GPU`: `1` runs every launch the mode would send to the card on the card AND in the simulator and compares the device memory; the simulator's result is the one the program gets, so differences do not add up over launches. `all` does it for every kernel the card can run, admitted or not. Every launch is compared, whatever the size rule would say, and the card starts every launch from the simulator's memory: all live device memory is pushed before the first, and before each later one the chunks (64 KiB) whose bytes differed after the last compare and the ones the host wrote since (a chunk that compared equal is already on the card). All live memory comes back from the card after every launch for the compare, and a buffer that size is kept for it, so a run takes at least the simulator's time. In exact mode the compare is bytes, per 64 KiB chunk (`host_gpu_verify: in_set_identical=.. in_set_different=.. other_identical=.. other_different=..`). In turbo mode it is by tolerance, word by word (see `VGPU_HOST_GPU_ATOL`): each launch with a word outside tolerance is reported (the first three per kernel; all with `VGPU_HOST_GPU_LOG`), and the exit summary adds `host_gpu_outside:` lines per kernel | unset |
| `VGPU_HOST_GPU_LOG` | `1` prints each host-GPU decision (on the card, kept in the simulator and why, with the numbers of the size rule), the classes of each kernel in turbo mode, and in exact mode what a refused kernel's `MUFU` forms are; `2` adds the SASS of a kernel exact mode refused for one. In exact mode it also turns on the two lines for scripts that turbo mode always prints at exit: `host_gpu_turbo: mode=.. launches_card=.. launches_sim=.. card_s=.. sim_s=.. words_compared=.. words_different=.. words_outside=.. words_subnormal=.. kernels_outside=.. max_abs=.. max_rel=..` (launches that ran on the card, a verify run's included, and in the simulator; wall seconds of the card path and of the simulator's launches; the verify counts) and `host_gpu_refusals: refused_never=.. refused_atomics=.. refused_off=.. refused_exact=.. refused_safety=.. too_small=.. failed=..` (launches kept in the simulator, by why; `refused_exact` is exact mode's count of launches turbo mode would take) | unset |
| `VGPU_HOST_GPU_ATOMICS` | turbo mode: `0` keeps in the simulator the kernels with a float atomic that returns a value, a compare-and-swap (`CAS`, `CAST.SPIN`) or an exchange, which turbo mode admits by default since Round sp66 (an opt-in, `1`, before). A card does not repeat itself on programs built on these (the order threads arrive in decides the result), so there is no one answer to compare with; `tools/probes/hostgpu_atomic_repeat.cu` measures it on a card; a loop of compare-and-swap that does not end on the card is ended by `VGPU_HOST_GPU_TIMEOUT` | admitted |
| `VGPU_HOST_GPU_TURBO_OFF` | turbo mode: a comma list of instruction classes to keep in the simulator, for finding which class a difference comes from: `mufu`, `half`, `tensor`, `call`, `red` (float reductions), `atomics` (every atomic that returns a value or swaps), `extra` (the instructions turbo mode names beyond exact mode's: `IDP`, `REDUX`, `LDGSTS`, `LDSM`, `MOVM`, `FFMA2`, `B2R`, `NANOSLEEP` and the like), `printf` (kernels that call `printf`, Round sp66). A kernel of the exact set is always admitted. A name that is not a class turns the mode off with a message | unset |
| `VGPU_HOST_GPU_ATOL`, `VGPU_HOST_GPU_RTOL` | turbo mode's verify: a 4-byte word that differs passes when `abs(card - sim) <= ATOL + RTOL * abs(sim)` read as binary32, else as the binary64 of its aligned 8 bytes (only where the memory does not look like an array of binary32), else as two binary16 halves; a NaN matches any NaN (as halves only the same NaN), an infinity only the same infinity. One unit in the last place of a binary16 is 1e-3 of its value, so half-precision results need `VGPU_HOST_GPU_RTOL=2e-3` or more and `VGPU_HOST_GPU_TOL_TYPE=f16`. Memory has no types: [what the compare cannot see](perf-design/12-host-gpu-turbo.md#4-verify-by-tolerance). A kernel that takes `sinf` or `cosf` needs `VGPU_HOST_GPU_ATOL=1e-6`: the card's sine and cosine are accurate to about 3.6e-7 absolute, not relative (an A10G, [Round sp59](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp59-turbo-host-gpu-mode-on-an-a10g)). A value that is not a number zero or more turns the mode off with a message | `1e-8`, `1e-5` |
| `VGPU_HOST_GPU_TOL_TYPE` | turbo mode's verify: `f32`, `f64` or `f16` reads every differing word that way and no other; `auto` guesses as described above. Half-precision results need `f16`: under the guess two halves in a word are first read as one float, which hides most differences between them (halves below about 0.25 make a float under `ATOL`). Memory of other types in the same launch then shows as outside tolerance, so it is a switch for looking at one kernel | `auto` |
| `VGPU_HOST_GPU_TIMEOUT` | seconds a launch may stay on the card before the process says so and ends with status 124 (a kernel on the card cannot be cancelled and its memory cannot be read back, so there is nothing to fall back to); `0` waits forever | turbo: `600`; exact: `0` |
| `VGPU_HOST_GPU_WIDER` | `0`: the safety rules before Round sp66, for A/B runs: any `__device__` or `__constant__` variable, external function or device-side launch in a module keeps all its kernels in the simulator, a live texture object or a CUDA graph keeps every launch there, and turbo mode's float atomics with a value, compare-and-swap and exchange need `VGPU_HOST_GPU_ATOMICS=1` (the benchmark's forms `exactn` and `turbon`). The class scan is not affected: calls into another code section, `B2R`, `NANOSLEEP` and `LEPC` stay admitted | unset |
| `VGPU_HOST_GPU_DIV` | exact mode: `1` also admits kernels whose only instructions outside the exact set are the IEEE float division sequence (`MUFU.RCP`, the refining `FFMA`s, `FCHK`, the call to the slow path). The division is correctly rounded whatever the last bit of the reciprocal, on an A10G for 12 dividends against all 2^32 divisors ([Round sp44](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp44-host-gpu-card-experiments)); it stays opt-in because other cards were not run | unset |
| `VGPU_HOST_GPU_START_COST` | seconds charged for loading the real driver and making its context when the size rule decides the first launch that would go to the card (measured 0.23 s on an A10G); `0` charges nothing, which sends short programs to the card at a loss | `0.25` |
| `VGPU_HOST_GPU_NO_FIRST_LAUNCH` | `1`: a kernel's first launch is always the simulator's (by default a first launch of a looping kernel on a grid that fills the card may go, judged from its code); for A/B runs | unset |
| `VGPU_HOST_GPU_SAMPLE` | a launch of a kernel with a loop that the mode has no estimate for is sampled before the simulator runs it: the process is forked, the copy runs a few of the launch's blocks (up to 32, a few milliseconds), reports their times and ends, and if they say the launch repays the card path and the driver start the card runs the launch ([design](perf-design/12-host-gpu-turbo.md), section 8). A grid of fewer than 256 blocks is sampled inside its blocks instead, when its kernel has a counted loop the copy can follow: the copy reads how many turns of the loop each warp has left and times one turn (Round sp61); the `host_gpu_sample:` line of such a launch says `inside=1`; `whole` samples by whole blocks only (the rule of Round sp60b, for A/B runs). `0`: no launch is sampled (the rule before Round sp60b; also for a program that reaps children it did not make). `log`: launches are sampled and decided but none is sent for it, and each prints a `host_gpu_sample:` line (what was measured, what would have been done, what the launch took), which needs no card. `always`: as the default, without asking for a loop, a static estimate or a process that can afford the sample (for tests) | on |
| `VGPU_HOST_GPU_STREAK` | while the card holds memory the simulator does not (the launch before ran there), a launch of a kernel the mode has an estimate for is weighed with what each side would move: keeping it in the simulator costs its time plus pulling back every chunk only the card holds, sending it costs the launch, the kernel's time on the card and the push of what only the simulator holds; it goes when the first is over twice the second and over 200 us. Only once the driver is up ([design](perf-design/12-host-gpu-turbo.md), section 9; Round sp65). The summary's `host_gpu_streak:` line counts such launches. `0`: such a launch is judged by its simulator time alone (the rule before Round sp65, for A/B runs) | on |
| `VGPU_HOST_GPU_STATIC` | `1`: a kernel the simulator has not run yet is estimated from its instruction count and grid. Not a lower bound (threads that leave at a bounds check run far less: PolyBench 3DCONV was 38 times slower with it); an experiment switch | unset |
| `VGPU_HOST_GPU_WRITE_THROUGH` | `1`: every launch on the card copies all live device memory there and back (the first form of the mode), instead of keeping a card copy between launches and moving only the 64 KiB chunks that changed hands; for ruling the residency protocol out | unset |
| `VGPU_HOST_GPU_MAX_MIB` | launches with more live device memory than this many MiB stay in the simulator | `2048` |
| `VGPU_HOST_GPU_LIBCUDA` | the real driver library the mode loads (the tests point it at a stand-in) | `/usr/lib/x86_64-linux-gnu/libcuda.so.1` |
| `VGPU_HOST_GPU_TRACE` | a file that receives the stream of allocations, frees, host copies and launches (with the simulator's time for each), for the offline model of transfers in `docs/perf-design/data/round13/residency_model.py` and the replay of the size rule in `docs/perf-design/data/sp65/residency_replay.py` (a launch line also gives the block size, the kernel's static instruction count and whether its code has a loop); the launch lines need the mode on | unset |
| `VGPU_JIT` | `1` turns on tier 2 (off by default; `auto` is `1` without the notices that only say the JIT cannot run here, a missing compiler and a cache directory that cannot be used: the JIT as a default-on one would run, [round sp63](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp63-the-jits-cold-run-the-wait-before-the-lookup-history-the-cache-cap-the-notices-branch-sp63-jit-cold); also off with `VGPU_TIER=0`, `VGPU_FASTPATH=0`, `VGPU_SIMD=off`, `VGPU_SASS_TRACE` and in sanitizer builds): the hot stretches of a code section that has run `VGPU_JIT_THRESHOLD` warp instructions (default 100000000) -- straight-line code, loops and the uniform branches in them (a branch every lane of the group takes, or none), S2R and the uniform datapath, stores -- are turned into C++, compiled in the background with the host's compiler (`VGPU_JIT_CXX`, else `c++`, `g++`, `clang++` on `PATH`) into a shared object, kept in `VGPU_JIT_CACHE` (else `$XDG_CACHE_HOME/pantheonsim/jit/<12 hex digits of the library's GNU build id>`, created `0700`, at most `VGPU_JIT_CACHE_MAX_MB` (256; `0`: no cap) over that directory and the other builds' directories beside it, the least recently loaded objects removed first after a compile, and another build's directory none of whose files changed for 14 days removed; every object checked against its checksum and its instruction words before it is loaded) and run in place of the tier for those stretches. Regions are compiled to C++ that includes the simulator's own lane kernels (src/simd/kernels.inc, embedded in the binary and written into the cache directory), so the arithmetic of MOV, IADD3, IMAD (plain, WIDE, HI, carries), LOP3, LEA, SHF, ISETP, FSETP, FSEL, FFMA, FADD, FMUL, PRMT, IMNMX, I2FP, SEL, PLOP3, S2R (not the clocks), CS2R of SRZ, UMOV, ULDC, UIADD3, ULOP3, USEL and UISETP is computed in the generated code and every other op is a call of the tier's handler (a branch that splits the group, BSSY, BSYNC, EXIT, barriers and the rest end a region: the tier runs them); results, counters, faults and errors are the tier's. Only regions that are hot (`VGPU_JIT_HOT_PERMILLE`, default 1: at least this many thousandths of the section's counted instructions; `VGPU_JIT_MAX_OPS`, default 1500 ops) and at least half generated are compiled; regions shorter than `VGPU_JIT_MIN_REGION` (4) ops stay on the tier. The cache is asked after `VGPU_JIT_LOOKUP_AFTER` (1000000) instructions, and only for a section the share rule has kept (once kept, a section waits for its lookup on the plain run loop, where all of its instructions count, and is counted region by region again from the lookup on; `VGPU_JIT_PROFILE=0` turns this wait off with the other): a section is sampled once over `VGPU_JIT_SAMPLE` (50000) of its warp instructions, and if the generated code would compute less than `VGPU_JIT_MIN_SHARE` percent of them (default 40; `0` turns the rule off, as does `VGPU_JIT_HOT_PERMILLE=0`) it goes back to the plain loop for good and asks neither the cache nor the compiler (`VGPU_JIT_STATS=1` prints each decision; [measurements](perf-design/round14-jit.md)). A section the cache has nothing for, and whose threshold is more than twice `VGPU_JIT_PROFILE` (4000000) instructions away, then waits on the plain run loop: only the warp instructions of each call of the loop are added up (all of them count towards the threshold), and `VGPU_JIT_PROFILE` instructions short of it the section is counted region by region again, which is what chooses the regions to compile (`0`: counted region by region all the way, as before [round sp57](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp57-the-jit-as-the-default-measured-not-flipped-the-count-on-the-way-to-the-threshold-moved-to-the-plain-loop-branch-sp57-jit-default), which cost a program that never reaches the threshold 2 to 16 %). The compiler's identity (its version and target) is asked once and kept in the cache directory (`cid-*.txt`). The regions chosen for a section are recorded before its compile, so that a compile still running when the program ends (it is then cancelled, silently) is compiled at the next run's lookup rather than at its threshold (`VGPU_JIT_HISTORY=0`: counted to the threshold again). No compiler, a failed or too slow compile (`VGPU_JIT_TIMEOUT` seconds, default 120; `VGPU_JIT_MEM_MB` address space for the compiler, default 4096), or a cache that cannot be written or mapped (a `noexec` mount) leaves the tier running, with one notice on stderr (silenced by `VGPU_QUIET`; given at most once a day from one cache directory, a `notice-<kind>` file there, and not at all with `VGPU_JIT=auto` for a missing compiler or an unusable cache). `VGPU_JIT_FLAGS` adds compiler flags (part of the cache key), `VGPU_JIT_MAX_REGION` caps a region (default 200 ops; [control flow](perf-design/round15-jit.md)), `VGPU_JIT_MIN_GEN` is the share of a region's ops the generated code must compute for it to be compiled (percent, default 50), `VGPU_JIT_SPECIALIZE` (default `0`: one body per region; `1` generates one for whole warps and one for any group as well, twice the code) and `VGPU_JIT_RESIDENCE` (default `0`: the generated code reads and writes the register file; `1` keeps the rows an op touches in local arrays within a stretch) choose the form of the generated code (the defaults compile 22 to 48 % faster and run as fast: [round sp38](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp38-follow-ups-host-writes-in-the-caching-budget-the-cheap-jit-form-as-the-default-always_inline-for-the-shared-memory-table-branch-sp38-followups-base-speedups-combined-48d3e8df)), `VGPU_JIT_DUMP=<file>` (or `1` for stderr) appends a listing of every section that gets regions, op by op with its region and whether the generated code computes it (G), calls the tier's handler for it (E), or runs it as a branch (B) or register-only op (C), `VGPU_JIT_SYNC=1` compiles on the thread that reaches the threshold (for tests), `VGPU_JIT_STATS=1` prints what it did at exit (`VGPU_JIT_STATS_FILE=<path>` appends it to a file; the waits, the compiles from an earlier run's record and what the cache's cap removed on `jit-wait:`, `jit-prewait:`, `jit-history:` and `jit-cache:` lines after the summary, then a `jit-pch:` line and one `jit-compile:` line a compile: why it was asked for (`threshold` or `history`), its regions, ops and bytes of source, when it was asked for, started and done in milliseconds since the library was loaded, how long it took, and how it ended, `ok`, `cancelled` or the failure, with `precompiled header` when it read one), with one line for each reason the generated code was left (`jit: exit <reason>: n`). While a section's compile runs it is on the plain run loop (since [round sp64](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp64-the-jits-compile-and-the-run-around-it-the-plain-loop-while-a-compile-runs-the-precompiled-runtime-header-measured-at-1-and-16-threads-branch-sp64-jit-threads); before, the counting loop, which cost the program 2 to 8 % for as long as the compile took). With GCC, the runtime header the generated code includes is compiled once into the runtime directory of the cache (`rt-*/vgpu_jit_pch-*.h.gch`, about 53 MB for each compiler, ISA level and set of flags), after the first compile of a process that found none, and every compile after it reads it instead of the text (0.35 to 0.43 s less a compile on server3: half of a small section's compile; the objects are the same bytes); GCC ignores a `.gch` it cannot use and reads the text, and `VGPU_JIT_PCH=0` makes and reads none. `VGPU_JIT_LINK=1` (off by default, an opt-in of [round sp52](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp52-linking-keeping-execution-inside-generated-code-across-region-exits-branch-sp52-region-linking-base-originmain-7b90cee5)) turns on linking: the generated code also goes on past a branch, a register-only op, a BSSY, an EXIT that no lane takes, a CALL.REL and a RET.REL when the lanes of the warp stand at several addresses, as long as the run loop's chain would pick its group again (the group lands strictly below every other runnable lane; the spin rule and the turn's calls are counted as the loop counts them), runs a branch that splits the group and goes on with the half at the lower address, and the selected regions of a section are compiled as one function, so that a branch, call or return from one region into another stays in generated code. A BSYNC stays the run loop's op: the generated code stops before it. Since [round sp53](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp53-linking-the-follow-up-why-syr2k-lost-bsync-inside-generated-code-what-the-linked-compile-costs-branch-sp53-link-followup-base-sp52-region-linking-1fce4fd5) a section that has a loop and no BSSY, BSYNC, CALL or RET is not linked (the loop rule: its warps do not diverge and spend their life in the loop, so linking has one region call a warp to gain and a larger generated function to lose; `VGPU_JIT_LINK_RULE=0` links every section). `VGPU_JIT_LINK_MERGE=0` (one function a region) and `VGPU_JIT_LINK_EXIT=0` (an EXIT stays the loop's) are diagnostic switches. Results and counters are the tier's; the objects have their own cache keys. `VGPU_JIT_FUSE=1` (off by default; rounds sp48 and sp52) writes runs of register-only ops (MOV, IADD3, IMAD, LOP3, LEA, FADD, FMUL, FFMA, unguarded) as one loop over the blocks of eight lanes with the rows in host registers instead of op by op (AVX2 hosts, the cheap form only); the same bits, other cache keys. [Design](perf-design/06-tier2-jit.md), [measurements](perf-design/round11-jit.md). | off |
| `VGPU_AMD_TIER` | the AMD (CDNA and RDNA) interpreter runs its hottest instruction forms through a pre-decoded tier (`amd/src/gcn_tier.cpp`): each instruction's handler and operand classes are worked out when it is decoded, and a handler works on whole register rows (AVX2 kernels where the host has them) instead of lane by lane. Results, counters, errors and the order of effects are the interpreter's, which also runs every form the tier has no handler for and every instruction whose result would be a NaN (the payload a NaN takes depends on the order the compiler gave the operands, so the tier does not decide it). `VGPU_AMD_TIER=0` turns it off (for ruling it out when a result looks wrong), `=1` on. `VGPU_AMD_TIER_STATS=1` prints, when the process exits, which instructions the interpreter ran anyway and how often. | on |
| `VGPU_FASTCOUNT` | `0` counts memory sectors and bank passes with the earlier reference code instead of the allocation-free version, for ruling the new counting out when a counter looks wrong (tested to give the same numbers) | unset (on) |
| `VGPU_TC_EXACT` | `0` turns the RTX 3060 (sm_86) tensor-core sum model off: wmma, `mma.sync` and SASS `HMMA` in f16, bf16 and tf32 then add the products one at a time in f32, as earlier releases did. The model (chunks of 8 products, truncating, measured on a card, `include/vgpu/exec/tc_ampere.hpp`) is what makes random-data results equal the card's, and costs 3.5 to 4.5 times the time of those instructions in the SASS and PTX engines (the bench workloads `wmma` and `mma_ptx`; a program that does little else spends that much more, one that mostly does other work little), except `mma.sync` on the PTX engine, where it is the faster of the two. Only sm_86 devices and cubins are affected; counters never change. Read at every launch, so it can be set between launches. `vgpu run` passes it through. | unset (on) |

**Checks and counters** (see [diagnostics](diagnostics.md))

| Variable | Meaning | Default |
| --- | --- | --- |
| `VGPU_STRICT` | `1` turns on the checks that catch bugs hardware hides: integer division by zero, and storing a register nothing has written. | off |
| `VGPU_RACE` | `1` reports unordered shared-memory access between warps; `2` also reports stores that change nothing. | off |
| `VGPU_COUNTERS` | `1` prints exact per-launch performance counters. | off |
| `VGPU_COUNTERS_FILE` | Append one line of JSON per kernel launch to this file, for a tool that reads counters. `vgpu ncu` sets it and reports from the file. | off |

**Output and tracing**

| Variable | Meaning | Default |
| --- | --- | --- |
| `VGPU_QUIET` | `1` silences stderr diagnostics. | off |
| `VGPU_TRACE` | `1` logs every runtime and driver entry point. | off |
| `VGPU_TRACE_API`, `VGPU_TRACE_LAUNCHES` | AMD: `1` names each HIP call, or what each launch runs, one line to stderr each. | off |
| `VGPU_SASS_TRACE=<warp>` | Print every SASS instruction one warp runs, with the lane group and the first lane's results; after a `BSSY`, `BSYNC`, `BREAK` or `BMOV` it adds the barrier masks. | off |
| `VGPU_SASS_TRACE_KERNEL=<text>` | Restrict the trace to kernels whose name contains the text. | all |
| `VGPU_SASS_TRACE_BLOCK=<x>` | Trace block `(x,0,0)` instead of block `(0,0,0)`. | `0` |
| `VGPU_SASS_TRACE_RAW=1` | Add each traced instruction's 128 bits (high word first) to the trace, for finding where a field is encoded. | off |

The code reads further `VGPU_*` variables that plumb the simulator's own parts
together (NVSHMEM and NCCL rendezvous, the NVRTC and nvJitLink libraries, state
directories). They are not meant to be set by hand.
