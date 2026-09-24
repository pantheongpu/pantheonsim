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
instructions the dispatch has retired, which is this model's cycle, where a
card's counts at a fixed rate.

| Folder | What |
| --- | --- |
| `profiles/` | MI300X, MI325X and MI350X. MI325X was read from a physical card with `tools/rocminfo-to-profile.py`; the others are placeholders, and each file's header says which |
| `registers/` | the MMIO database, each model's registers and their power-on values, and the amdgpu headers' licence |
| `src/` | the register model, the metrics table and CPER records; code objects and CDNA decoding; the HIP runtime and ROCm libraries once they are written |
| `tools/` | `rocm-smi`, `amd-smi` and `rocm_agent_enumerator` for simulated machines, and the scripts that characterize a card (`characterize-hip.cpp`, the DigitalOcean scripts) |
| `tests/` | the AMD unit and end-to-end tests, and the code object they read (`tests/data/build.sh` rebuilds it with clang; no ROCm needed) |

The libraries follow the runtime, each checked against the real one on
hardware the way the NVIDIA side is.

A profile id is `amd/<name>`, and its file is `profiles/<name>.yaml` here.
