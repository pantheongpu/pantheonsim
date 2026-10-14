# Command-line reference

Everything the `vgpu` command and the environment variables can do. The
[README](../README.md) has the short path; this page is the full list.

Build once (`./scripts/build.sh`) and the command is `build/vgpu`. The build
also puts it at `build/bin/vgpu`, next to drop-in `nvidia-smi`, `rocm-smi` and
`amd-smi`, so `PATH=$PWD/build/bin:$PATH` makes all of them available.
`vgpu --help` lists the commands. `run`, `test`, `ncu`, `debug`, `fault`, `regs`
and `smi` take `--help` for their own options; `shell` and `serve` do not, so
their options are listed below.

## Commands

| Command | What it does |
| --- | --- |
| `vgpu list-gpus` | List the built-in virtual GPU models. |
| `vgpu info --gpu <vendor/model> [--json]` | Show one device profile. |
| `vgpu demo vectoradd [--gpu <id>] [-n <elems>]` | Run a built-in `vectorAdd` kernel end to end, as a smoke test. |
| `vgpu run [options] <program> [args...]` | Run a program on a simulated GPU. [Options](#vgpu-run). |
| `vgpu shell` | An interactive simulated machine: `nvidia-smi`, `rocm-smi`, `lspci` and `dmesg` all answer for the GPUs you picked. [Options](#vgpu-shell). |
| `vgpu test --matrix <program> [args...]` | Run a program on every device profile and compare the results. [Options](#vgpu-test---matrix). |
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
| `--preload` | Also `LD_PRELOAD` the shim. Needed when the program carries a `DT_RPATH` naming a directory of the real CUDA libraries: the loader reads that old-style tag *before* `LD_LIBRARY_PATH`, so the simulator never gets in front. (A `DT_RUNPATH`, which modern toolchains emit, comes after and is harmless.) `vgpu run` tells you when it applies. For an interpreter such as `python` with PyTorch, which names no CUDA library itself, every CUDA library the shim carries is preloaded. |
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

## `vgpu shell`

```bash
vgpu shell                       # asks which GPU, CUDA/driver and OS
vgpu shell -y --gpu amd/mi300x --count 8 --rocm 6.2.0 --driver 6.3.6 \
    --os ubuntu:24.04 --hostname mi300-node --load 0.65
```

| Option | Meaning |
| --- | --- |
| `--gpu`, `--count 1..16`, `--vram-mb` | The rack. |
| `--cuda`, `--rocm`, `--driver` | Reported toolkit and driver versions. The defaults are the ones this build's runtime carries. |
| `--os ubuntu:22.04\|ubuntu:24.04\|rocky\|debian`, `--kernel`, `--hostname` | Machine identity. |
| `--load 0..1` | Background load, so the telemetry columns move. |
| `-c`, `--command <cmd>` | Run one command in the machine instead of an interactive shell. |
| `-y`, `--no-prompt` | Take the defaults for anything not given, without asking. |
| `--no-isolate` | Skip the namespaces (session tools only). |

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

The `vgpu run` options set most of these for you; set them directly when you run
a program without `vgpu run`, such as a driver-API program or a script. Each
variable is unset by default.

**Device**

| Variable | Meaning | Default |
| --- | --- | --- |
| `VGPU_GPU` | Virtual GPU profile id. | `nvidia/h100` |
| `VGPU_DEVICE_COUNT` | Number of identical virtual devices. | `1` |
| `VGPU_VRAM_MB` | Virtual VRAM size, overriding the profile. | profile |
| `VGPU_CUDA_VERSION`, `VGPU_DRIVER_VERSION` | CUDA and driver version strings that `nvidia-smi` and NVML report. | this build's |
| `VGPU_ROCM_VERSION` | ROCm version string that the AMD tools report. | this build's |

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
| `VGPU_THREADS` | Host threads used to run blocks. | one per core |
| `VGPU_MAX_STEPS` | Raises the runaway-kernel step budget; `0` removes it. | built-in |
| `VGPU_SASS` | `0` forces the PTX engine; `1` insists on SASS and does not fall back. | SASS when the binary carries it, else PTX |
| `VGPU_SASS_LOG` | `1` says when a fatbin falls back from SASS to PTX. | off |
| `VGPU_SCHEDULER` | Warp scheduling order: `deterministic`, `random` or `adversarial`. Random and adversarial orders find bugs that depend on which warp runs first, and they run blocks one at a time so an order can be replayed. **PTX engine only:** kernels that run from SASS (the default whenever the binary carries it) use one fixed warp order, so set `VGPU_SASS=0` to use this. | deterministic |
| `VGPU_SCHEDULER_SEED` | Seed for `random` and `adversarial`; the same seed replays the same execution. | `0` |
| `VGPU_PTX_CONTRACT` | `0` runs multiply and add as written, rounding each, instead of fusing them. | fused |
| `VGPU_FASTPATH` | `0` sends every instruction down the interpreter's general path, for ruling out its fast paths when a result looks wrong (they are tested to give the same bits). | on |
| `VGPU_LOOKUP_CACHE` | `0` makes every kernel memory access look its address up in the allocation table, for ruling out the per-thread memory of recent lookups when an access looks wrong (tested to give the same values and errors) | unset (on) |
| `VGPU_LAUNCH_CACHE` | `0` makes every launch work out again what it can remember (the table of every loaded kernel that a device-side launch looks its child up in, and which function a kernel handle names), for ruling the remembering out when a result looks wrong; the answers are tested to be the same either way | unset (on) |
| `VGPU_SIMD` | `base` or `avx2` caps the vector instruction set the lane arithmetic, half-precision conversions and tensor-core kernels use (the default is the best the CPU has); `off` sends the engines back to their scalar code, for ruling the vector kernels out when a result looks wrong (they are tested to give the same bits) | best available |
| `VGPU_FASTCOUNT` | `0` counts memory sectors and bank passes with the earlier reference code instead of the allocation-free version, for ruling the new counting out when a counter looks wrong (tested to give the same numbers) | unset (on) |

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

The code reads further `VGPU_*` variables that plumb the simulator's own parts
together (NVSHMEM and NCCL rendezvous, the NVRTC and nvJitLink libraries, state
directories). They are not meant to be set by hand.
