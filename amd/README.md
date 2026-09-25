# AMD

Support for AMD GPUs and ROCm, which is at the start.

**Discovery** works: a simulated AMD machine answers `rocm-smi`, `amd-smi` and
`rocm_agent_enumerator` from its profile, as the real tools would, and its
registers, RAS counts, CPER records and amdgpu sysfs files are modelled
(`docs/registers.md`, `docs/telemetry.md`).

**Execution** is being built, and CDNA kernels now run:

- `src/codeobject.cpp` reads the code object a HIP program hands the driver --
  the ELF, each kernel's descriptor, and the metadata note naming the kernels
  and laying out their arguments.
- `src/gcn_decode.cpp` decodes the CDNA instructions in it, checked
  instruction by instruction against what the assembler wrote, both on the
  object in `tests/data/` and on one built during the test run
  (`tests/e2e/run_gcn_disasm.sh`).
- `src/gcn_exec.cpp` runs them: work-groups of 64-lane wavefronts, each with
  the scalar registers, VCC, SCC and the EXEC mask the ISA exposes, over LDS
  and device memory, with barriers between the waves of a group. Divergence is
  what the compiler writes -- save EXEC, narrow it, put it back -- not a path
  stack.

- `src/hip_api.cpp` is `libamdhip64`: the HIP runtime API, over those pieces.
  A HIP program links against it the way it links against AMD's, asks for
  devices and memory, loads a code object and launches the kernels in it
  (`tests/e2e/run_hip.sh` does exactly that, and checks the answers).

A kernel learns the shape of its grid the way the ABI says: from the
arguments the compiler adds after its own (which the launch fills in), or
from the packet a dispatch is described by (which the launch writes where the
kernel asked for it). A kernel compiled from HIP reads `blockDim` and
`gridDim` through the first of those.

A module brings its own variables -- what a `__device__` global compiles to.
The loader places them on the device and fills in the addresses the code was
left to have (`R_AMDGPU_REL32_LO` and `_HI`, which a kernel adds to the
program counter), and `hipModuleGetGlobal` hands a program the address of one.

The interface is `include/vgpu_hip.h`, a clean-room subset of the documented
HIP API. What is implemented is devices, memory and the module API: how much
memory a device has and how much is left, several devices each keeping their
own, streams and the events a program times its work with, and a launch whose
shared-memory parameter sizes the LDS the kernel did not reserve for itself.
The time between two events is the simulator's own, not what a card would have
taken, which this does not claim to know. Nothing runs behind the program's back, so
the asynchronous calls are the synchronous ones and a stream is done when the
call returns. A program built by `hipcc` runs unmodified too:
its device code is registered from inside the executable before `main`, its
chevron launches go through `hipLaunchKernel`, and it reads the device through
the real headers' `hipDeviceProp_t`, which is laid out here field for field as
ROCm lays it out. Graph capture and replay, peer access between devices, and
pinned host memory used to stage copies are there for the programs that use
them. The library answers to both of ROCm's names for it (`libamdhip64.so.6`
and `.so.7`) and gives each function the symbol version the real one does,
since a program built by `hipcc` asks for `hipMalloc@hip_4.2`, not just
`hipMalloc`.

The instructions implemented are those clang emits for the kernels in
`tests/data/`.

The arithmetic: scalar and vector integers, the logical and shift ops, and
values of every width a kernel uses. A 64-bit add is the compiler's own --
the low halves, a mask of the lanes that carried, and the high halves with it
added back. Bytes and shorts are loaded, computed on and stored back, and the
16-bit arithmetic they get writes the low half of a register and zeroes the
high half, which is why the compiler leaves out the mask a widening would
otherwise need. Single and double precision have their add, multiply, fma,
min and max, the sequence a division compiles to, the roundings, the
comparisons and the conversions between them. Half precision comes both one
value at a time and packed two to a register, and two floats pack into a
register pair. Beside those: the transcendentals, the sine and the cosine
(whose argument is a turn rather than a radian, which is why the compiler
multiplies by one over two pi first), bit counting, the comparisons in both
their forms and the class test, `v_cndmask`, and the lane-counting ops. So
are the idioms a kernel writes out by hand: a clamp (the median of three), a
rotate or a funnel shift, a bitfield insert, frexp, the integer and half dot
products, and two floats packed into halves rounded toward zero. A half dot
product whose sum does not fit a float exactly is rounded once here; a card
may round it differently.

The control flow: scalar and EXEC branches, and a call to a function the
compiler did not inline with the return from it.

The memory: global loads and stores and their atomics -- an add, a subtract,
the logical ones, an exchange and a compare-and-swap, on a 32-bit value, on a
float and on a pair, each able to give back what it replaced -- LDS, a byte,
a half, a word, two or four at a time, and its atomics on integers and
floats, a work-item's private memory (what a kernel spills into when it runs
out of registers) and the accumulation registers (which it spills into
first), a value read from another lane, and a flat access, whose address says
for itself whether it means LDS or the device.

Lanes trade values through the LDS unit without touching LDS, too: a lane
reads the lane an address names, or the lane a swizzle pattern names -- four
lanes choosing among their own four, or a group of 32 swapped, reversed or
broadcast. Both read every lane's value before any lane's result is written,
since the destination may be the register they read.

The matrix instruction `v_mfma_f32_16x16x16_f16` multiplies two 16x16 matrices
of halves spread across the wave's 64 lanes and adds a 16x16 block of floats.
Which element sits in which lane's register is checked rather than assumed: a
GEMM written with rocWMMA -- AMD's library, whose loads and stores put each
element where the hardware expects it -- runs through it and matches the same
product worked out in C, and moving any part of the arrangement makes it fail.
Its sums are formed in double and rounded once; where a sum is not exact, a
card may round it differently. Its broadcast modifiers, and a wave with lanes
switched off, are refused.

A lane can also read another lane's register through the cross-lane form,
within its row of sixteen: the shifts and the rotate, the two mirrors, the
broadcasts that carry a row into the next, and the masks that say which lanes
are written. The forms that reach across the whole wave are decoded and
refused, since nothing available here settles which way they carry.

A memory fence compiles to a write-back and an invalidate of the caches;
every access here reaches memory directly, so both do nothing. A kernel that
calls the host -- device-side `printf` is built on this -- is refused where it
does, by name, rather than left spinning on a reply that will not come.

The source modifiers are applied -- an absolute value, a negation, a clamp of
the result -- and so is the sub-dword form, where an instruction reads a
named byte or half of each source, with its sign or without, and writes its
result into a named part of the destination with the rest zeroed. That form
is how the compiler mixes widths and how it packs two values into one
register; a source of it may be a scalar register rather than a vector one.
Filling the rest of a destination with the sign instead of zeroes, or leaving
it as it was, is refused: nothing here has been seen to ask for either.

Any other instruction is refused by name, and so is anything this does not
model: an output multiplier, a packed operation that shuffles halves. A wrong
guess would run and give a wrong answer, which is worse than a refusal.

Three things are modelled rather than copied, and are marked where they are
written: the reciprocal, square root, exponent and logarithm are the host's
exact results where the hardware's are tables good to about one unit in the
last place; the scope bits on a memory instruction change nothing, since every
access here is already visible to every wave; LDS sits at an address of
this model's choosing, which a kernel reads from `src_shared_base` the way it
reads the hardware's; and the counter a wave reads to time itself counts the
instructions retired by the host thread running it, which is this model's
cycle, where a card's counts at a fixed rate.

Work-groups run on every host core at once, as a GPU runs them in any order
and concurrently; `VGPU_THREADS=1` runs them one after another, in order, which
is what a kernel with a data race needs to give the same answer every time.
Device memory is shared between the threads a word at a time, an atomic takes
a lock striped by address, and a fence (`buffer_wbl2`, `buffer_inv`) is a
fence on the host. `amd_exec_bench` (`tools/exec-bench.cpp`) says how fast a
kernel runs on the interpreter.

## Profiling

AMD's profiler, `rocprofv3`, runs unmodified on a simulated GPU. It is a front
end over a tool library that asks `librocprofiler-sdk` for the machine's agents
and counters and subscribes to what the runtime does; VirtualGPU's
`librocprofiler-sdk` (`src/rocprofiler_sdk.cpp`, built into `build/shim`)
answers from the simulated devices, and the HIP runtime reports to it every
HIP call, every code object placed on a device and its kernels, every copy
and allocation, and every launch with what its waves did. Inside
`vgpu shell --gpu amd/mi300x`, with ROCm and its rocprofiler-sdk installed:

    rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU TA_FLAT_READ_WAVEFRONTS --kernel-trace -d out -- ./app
    rocprofv3 --runtime-trace --output-format csv -d out -- ./app
    rocprofv3-avail list --pmc

The session's `rocprofv3` is ROCm's own, given `--rocm-root` pointing at a copy
of the installation whose `librocprofiler-sdk` is VirtualGPU's (rocprofv3
loads it by path); `rocprofv3-avail` needs nothing, since its library finds
the shim's first.

The counters offered are the ones the interpreter counts exactly, because
every instruction a wave issues passes through it once: `SQ_WAVES` and the
waves by how many lanes they start with, the instructions each unit issues
(`SQ_INSTS_VALU`, `_MFMA`, `_SALU`, `_SMEM`, `_VMEM`, `_FLAT`, `_LDS`,
`_BRANCH`, `_SENDMSG`, `_GDS`), and the flat reads, writes and atomics the
texture addresser takes (`TA_FLAT_*`). Each is the device's total, as one
instance. A counter of cycles, stalls, cache hits or memory traffic would
need a model of the hardware's timing, and there is none, so those are not
offered: rocprofv3 says the device does not have one, as it does for a
counter a real GPU lacks, rather than printing a number that was made up.
One count is a reading of AMD's description rather than a measurement on a
card: a generic flat access that reaches LDS counts toward `SQ_INSTS_LDS` as
well as `SQ_INSTS_FLAT` ("including FLAT").

Times in the traces are the simulation's own, on the clock
`/proc/<pid>/stat` uses: a truthful order and truthful durations of the
simulation, and nothing about how long a card would take. There is no HSA
runtime, so no HSA trace; no PC sampling or thread trace, which need the
hardware's sampling and trace units; and no records of the kernels ROCm's
runtime runs on its own behalf (a device-to-device copy or a memset is not a
kernel here).

`test_amd_rocprofiler` is a profiling tool of its own that runs everywhere;
`amd_rocprofv3` runs AMD's rocprofv3 in a session wherever rocprofiler-sdk is
installed, and `amd_rocprofiler_abi` checks every structure against its
headers.

## The pantheon workloads

`amd/tools/run-pantheon-workloads.sh [pantheongpu-repo]` builds the pantheon
GPU workloads for gfx942 exactly as pantheon's Makefile does for HIP -- hipcc,
the same flags, the sources unmodified -- and runs each on a simulated MI300X
with `--verify`, so a pass means the workload checked its own results. It
needs ROCm's hipcc (set `VGPU_ROCM_PATH`). The workloads' own knobs are set to
a CPU-appropriate intensity, as the NVIDIA runner sets them; that runs the
same code over a smaller working set.

With ROCm 7.1 every workload that builds for gfx942 passes, and every
workload's device code decodes as ROCm's llvm-objdump prints it. Three do not
run, for reasons a card would share:

- `fused_attention` does not build for AMD: it passes a 32-bit mask to
  `__shfl_xor_sync`, which HIP requires to be 64-bit on 64-lane waves (ROCm
  7), and which ROCm 6.4 does not declare without
  `-DHIP_ENABLE_WARP_SYNC_BUILTINS`.
- `rt_virus` and `media_enc_virus` skip themselves: CDNA has no ray-tracing
  units, and the encoder workload is NVIDIA's.

`mma_virus` builds its matrix path only where rocWMMA's headers are installed
(`rocwmma-dev`, part of a full ROCm install); without them it builds a stand-in
and skips itself, as it would on a card.

What `--verify` proves is worth knowing. The memory workloads check patterns
the host decided; most of the rest check that a kernel gives the same answer
every time, or in every thread, which is what catches a failing card. A model
that was consistently wrong could pass those, which is why each instruction is
also checked on its own, against the assembler and against C. Every workload
that verifies was also run with `--inject_error`, with knobs large enough for
the fault to fire, and each one caught it. A workload whose check reports a
fault with device-side `printf` ends with a refused launch rather than its own
"Verification: FAIL" line, since `printf` needs the hostcall path that is not
modelled yet.

| Folder | What |
| --- | --- |
| `profiles/` | MI300X, MI325X and MI350X. MI325X was read from a physical card with `tools/rocminfo-to-profile.py`; the others are placeholders, and each file's header says which |
| `registers/` | the MMIO database, each model's registers and their power-on values, and the amdgpu headers' licence |
| `src/` | the register model, the metrics table and CPER records; code objects and CDNA decoding; the HIP runtime and ROCm libraries once they are written |
| `tools/` | `rocm-smi`, `amd-smi` and `rocm_agent_enumerator` for simulated machines, the generators of the HIP version script and rocprofiler-sdk's HIP call numbers, and the scripts that characterize a card (`characterize-hip.cpp`, the DigitalOcean scripts) |
| `tests/` | the AMD unit and end-to-end tests, and the code object they read (`tests/data/build.sh` rebuilds it with clang; no ROCm needed) |

The libraries follow the runtime, each checked against the real one on
hardware the way the NVIDIA side is.

A profile id is `amd/<name>`, and its file is `profiles/<name>.yaml` here.
