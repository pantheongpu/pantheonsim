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
  stack. A group's waves take turns, each running until it stops (a barrier,
  `s_sleep`, the end) -- except that a wave waiting for a global load lets the
  others run first, as the load's latency does on a card: rocPRIM reads a
  block id from LDS after a barrier and overwrites it once its loads arrive,
  and without the pause the wave that went on to the load overwrote it before
  the others had read (`tests/data/asm_barrier_order.s`).

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
taken, which this does not claim to know. Streams run at once, as a card's
do: each stream's work runs in order on a host thread of its own, so a kernel
on one stream may wait on a flag another stream's kernel sets, an event and a
stream say `hipErrorNotReady` while their work runs, `hipStreamWaitEvent`
orders one stream behind another, and the null stream is the legacy default
stream, ordered against the blocking streams. `hipStreamLegacy` names it too;
`hipStreamPerThread`, and the null stream in a program built with
`-fgpu-default-stream=per-thread` (the `_spt` calls), is a stream of each
thread's own. A stream can also run a host callback, write a value to memory,
or hold its later work until memory holds a value, while every other stream
goes on. Contexts are a stack per thread over each device's primary context,
as in ROCm's HIP. Where HIP's calls go beyond the everyday ones -- device
flags and UUIDs, limits, stream ids and attributes, every launch form
(`hipLaunchKernelExC`, `hipConfigureCall` and `hipLaunchByPtr`, a launch on
several devices at once) -- each answers as ROCm's HIP does, down to the error
for each wrong argument: AMD's own tests (hip-tests) are the reference, and
`tests/hipcc/api.cpp` checks them. Memory is held to the same reference
(`tests/hipcc/memory.cpp`): copies and memsets of every shape (2D, 3D, the
driver API's forms, peer copies) with ROCm's rules for who waits, managed
memory's advice and prefetch, stream-ordered pools that keep freed memory for
reuse by the rules each pool is set to, pools and their pointers shared with
another process through a file descriptor, and virtual memory at HIP's 4 KiB
page granularity: memory the simulator keeps on the host (a file where it may
be shared), mapped in place at the address a reservation set aside on any
device, exported and imported as a file descriptor, with device memory handed
out as one where ROCm hands out a dma-buf. Events made with `hipEventInterprocess` are shared
the same way: another process opens one's handle, and its waits wait for the
records made anywhere (`tests/hipcc/ipc.cpp`). Graphs are the whole of HIP's graph API
(`src/hip_graph.inc`, checked by `tests/hipcc/graphs.cpp`): every node type
built by hand or captured from streams -- across streams joined by events, in
each capture mode, with the calls a capture refuses refused as ROCm refuses
them -- executable graphs and everything that changes one, child graphs,
clones, allocation nodes whose memory outlives the graph, user objects, and a
drawing in Graphviz's dot. A launch is one piece of its stream's work, its
nodes run in an order their edges allow. Every function ROCm 7.1's `libamdhip64`
exports is here, down to `__managed__` variables, modules loaded as CUDA 12's
libraries or as fat binaries, the run-time linker for code objects (`hipLink*`: one input comes back as it
is; several become one module holding them, a kernel defined twice is a link
error, and the binary `hipLinkComplete` writes is this runtime's own
container, which `hipModuleLoadData` reads back, not an ELF; relocatable
objects that call into one another would need a real linker, and linking LLVM
bitcode needs AMD's compiler library, so both are refused), HCC's launch by
its C and C++ names, and what a device with no OpenGL and no dma-bufs answers
(`tests/hipcc/exports.cpp`). A kernel's fault is told at the
next synchronization, as on a card. `VGPU_SYNC_LAUNCHES=1` makes every call
wait for its own work instead, which rules concurrency out when a program
misbehaves. A program built by `hipcc` runs unmodified too:
its device code is registered from inside the executable before `main`, its
chevron launches go through `hipLaunchKernel`, and it reads the device through
the real headers' `hipDeviceProp_t`, which is laid out here field for field as
ROCm lays it out. Peer access between devices and pinned host memory used to
stage copies are there for the programs that use them. The library answers to both of ROCm's names for it (`libamdhip64.so.6`
and `.so.7`) and gives each function the symbol version the real one does,
since a program built by `hipcc` asks for `hipMalloc@hip_4.2`, not just
`hipMalloc`.

Programs built by each current ROCm release -- 6.4, 7.0, 7.1 and 7.2 -- run on
it. The version each function carries is the same in all of them, and each
later release only adds functions. `hipGetDeviceProperties` without a suffix
fills the older layout (hipDeviceProp_tR0000) that every release keeps under
that name for programs built to it; one built with the headers as they are
calls `hipGetDevicePropertiesR0600`. `tests/hipcc/rocm/` holds five programs
built by each release -- each with that release's own device library, so its
printf and grid barrier -- and `tests/e2e/run_rocm_versions.sh` runs them all
against the same expected output. `tests/e2e/run_hip_abi.sh` checks the
structures, the attribute numbers and the memory types against the headers
of each of those releases installed.

`hipDeviceGetAttribute` answers each attribute with the property of the same
name, by the numbers ROCm's header gives them (`tests/e2e/run_hip_abi.sh`
checks every one against the header). The occupancy calls work out how many
work-groups a compute unit holds the way ROCm's runtime does, from the
kernel's registers and LDS. A cooperative launch puts every work-group of its
grid on the device at once, so a grid barrier (`this_grid().sync()`) holds,
and a grid larger than the device holds at once is refused as HIP refuses it.
With peer access enabled, a kernel reads and writes another device's memory
at the address that device gave it.

AMD's own libraries run on it unmodified, starting with rocBLAS (and
hipBLASLt, which it links). The device is named as HIP names it, features
and all (`gfx942:sramecc+:xnack-`), which is how rocBLAS picks, of the code it
ships for each setting of XNACK, the code built for this one. Its device code
comes as compressed offload bundles, dozens of them, inflated with the
system's zstd or zlib when a kernel first needs one rather than as the
library loads (zlib and zstd are the two methods clang defines; any other method number, and a header version other than 2 and 3, is refused by name); Tensile's GEMMs come from code objects hundreds of kernels
were linked into, each keeping its own metadata. The stream-ordered
allocations, pitched copies, pointer attributes and work-item-sized launches
(`hipExtModuleLaunchKernel`) it calls are there. `tests/e2e/run_rocblas.sh`
builds a program against rocBLAS and checks its level-1 routines, its
float and double GEMMs -- ragged sizes, every transpose -- and its double
triangular solves (`trsv` with a backwards stride, `trsm` from either side;
every fill, transpose and diagonal) against the same arithmetic on the host;
both need ROCm with rocBLAS installed and skip elsewhere. AMD's own rocBLAS test suite, built from ROCm 7.1's source, runs
on it as well: all 162,807 of its quick float and double tests pass, each
checked against OpenBLAS, and its half, bfloat16, int8 and FP8 GEMMs run too.

hipSPARSELt, from PyTorch's ROCm wheel, runs its 2:4 structured-sparse GEMMs
on it: its own kernels prune a matrix, compress it into values and indices,
and multiply with the sparse matrix instructions (`tests/e2e/
run_hipsparselt.sh`, ctests `amd_hipsparselt` and `amd_hipsparselt_mi350x`,
each product checked against the host's). Its kernels read a byte past the
end of a buffer, which a card allows because HIP hands out device memory by
the 4 KB page. AMD devices here allocate the same way. A kernel's read of the
rest of its buffer's last page is served, but a copy or a write past the end
is still refused.

The instructions implemented are those clang emits for the kernels in
`tests/data/`, and those rocBLAS's own kernels and Tensile's float and double
GEMMs use: `tests/e2e/run_rocblas_disasm.sh` decodes every one of their
eighteen million instructions and compares it with ROCm's llvm-objdump.

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
for itself whether it means LDS, private memory or the device. And a buffer
resource -- a base, a stride and a count of records in four scalar registers
-- through which a load past the end reads zero and a store past it goes
nowhere, which Tensile's GEMMs count on at the edges of a matrix; the scalar
offset counts toward the bounds, as Tensile's DGEMM needs it to. LDS read far
past the 64 KB a compute unit has reads zero, as a card gives it (Tensile
clears registers that way); an access within it but past what the work-group
reserved is caught, since that is almost always a launch that did not pay for
its kernel's LDS.

Lanes trade values through the LDS unit without touching LDS, too: a lane
reads the lane an address names, sends its value to the lane an address
names, or reads the lane a swizzle pattern names -- four lanes choosing among
their own four, or a group of 32 swapped, reversed, broadcast or rotated. Both read every lane's value before any lane's result is written,
since the destination may be the register they read.

The matrix instructions multiply two matrices spread across the wave's 64
lanes and add a third: halves into floats (`v_mfma_f32_16x16x16_f16`), floats
(16x16x4, 32x32x2, and the multi-block 16x16x1 and 4x4x1) and doubles
(`v_mfma_f64_16x16x4_f64`). Which element sits in which lane's register is
checked rather than assumed: GEMMs written with rocWMMA -- AMD's library,
whose loads and stores put each element where the hardware expects it -- run
through the half, float and double 16x16 forms and match the same products
worked out in C, and rocBLAS's GEMMs, built on the float and double forms,
match a GEMM done on the host. A double's accumulator does not hold its rows
as a float's does -- each lane group every fourth row, not four together --
and moving any part of either arrangement makes them fail. Sums are formed in
double (fused, for doubles) and rounded once; where a sum is not exact, a card
may round it differently. The reduced-precision xf32 forms (`v_mfma_f32_16x16x8_xf32`
and `32x32x4`, gfx940 to gfx942) multiply floats whose mantissa is cut to 10 bits,
as the MI300 ISA guide describes them: the guide says "round" once and
"truncated" in each instruction's own text, and the instruction's is taken --
which hipBLASLt's golden reference for its xf32 kernels bears out (`XFloat32` in
`tensilelite/include/Tensile/DataTypes_XFloat32.hpp`: "truncate lower 13 bits of
IEEE float to convert to XFloat32", and `client/cpu_gemm_driver.cpp`:
"XFloat32 ... truncates float operands to a 10-bit mantissa"). The rounding
`xfloat32` of rocBLAS and rocWMMA (`rocblas_xfloat32.h`, "add 0xFFF plus the lowest
bit") is the host type a program converts its own floats with, not what the
instruction does to the operands it is given (`test_amd_gcn_asm` checks the layout
the guide's general formula gives and the cut). The B broadcast pattern (BLGP) and a wave with lanes switched off are
refused.

A lane can also read another lane's register through the cross-lane form,
within its row of sixteen: the shifts and the rotate, the two mirrors, the
broadcasts that carry a row into the next, and the masks that say which lanes
are written. The forms that reach across the whole wave are decoded and
refused, since nothing available here settles which way they carry.

A memory fence compiles to a write-back and an invalidate of the caches;
every access here reaches memory directly, so both do nothing beyond fencing
the host threads the work-groups run on.

A kernel can call the host while it runs, which is what device-side `printf`
is built on. The protocol is the one ROCm's device library speaks (ockl's
hostcall): the kernel takes a packet from a buffer the runtime gave it,
fills a slot per lane, pushes it onto a ready stack and raises a doorbell,
whose mailbox makes it send an interrupt (`s_sendmsg`); that is where the
host's part is done (`src/hostcall.cpp`), and the kernel, spinning on the
packet, carries on. `printf` is a service implemented: a message per lane,
carried in as many packets as it needs, formatted as C's printf would and
written to the program's stdout, with printf's return value sent back.
Device-side `malloc` and `free` are the second (service 3, ockl's
`__ockl_devmem_request`): the device library's allocator asks the host for
its 2 MiB slabs and for large blocks, and gives them back; the host answers
with device memory placed on the alignment the library finds slabs by, and a
null pointer when the device is full. A kernel built for code object version
5 or later (every current hipcc) keeps the allocator's state in a block the
runtime provides (`hidden_heap_v1`); the runtime makes it, 128 KiB with the
counts of recordable slabs the allocator expects to find, when the first
kernel that has the argument launches (`Hostcall::heap`, the layout is
dm.cl's; ROCm's CLR does the same with the device library's own
initialization kernel); earlier code objects keep it in their own data
(`tests/data/devmalloc.cpp`, run by `test_amd_runtime_gaps`). The address
sanitizer's report (service 4) ends the
kernel with the report as the error (ROCm's runtime prints it and aborts); no
sanitizer-instrumented code was available to run it, so it is checked against
the device library's argument order only. The device function call (service 1,
`__ockl_call_host_function`) calls a host function the kernel names, once for
each lane that makes the call: word 0 of a lane's slot is the function's
address (`void f(uint64_t* output, const uint64_t* input)` -- seven words in,
two out, the signature ROCm's CLR documents), and its two output words come
back. Those are all the services the device library has; the fprintf
interface (`__ockl_fprintf_*`, the stderr stream included, which `assert`
uses) and printf are service 2. `tests/hipcc/hostcall.cpp` runs a host
function, malloc and free, printf's conversions and fprintf from a kernel
on the shim (wave64 and wave32), and the same program's host function,
malloc and fprintf parts run on ROCm's own HIP over the HSA runtime.
Vector loads and stores may be unaligned, as ROCm runs the
hardware; the compiler counts on that when it packs a string.

A program hipcc built carries linked code objects, whose code reaches its
own constants and variables relative to itself. The whole of such an object
goes on the device as one image and runs from there, so a format string or a
`__constant__` table is where the code looks for it; `__hipRegisterVar` and
the symbol calls (`hipMemcpyToSymbol` and the rest) find a program's
variables in it.

The source modifiers are applied -- an absolute value, a negation, a clamp of
the result -- and so is the sub-dword form, where an instruction reads a
named byte or half of each source, with its sign or without, and writes its
result into a named part of the destination with the rest zeroed. That form
is how the compiler mixes widths and how it packs two values into one
register; a source of it may be a scalar register rather than a vector one.
A comparison has a sub-dword form too, writing VCC or the scalar pair it
names, and every comparison has an X form, which writes EXEC as well.
Packed instructions take op_sel and op_sel_hi -- which half, or which
register of a pair, of each source feeds each result -- and negate each half
on its own.
Filling the rest of a destination with the sign instead of zeroes, or leaving
it as it was, is refused: nothing here has been seen to ask for either.

Any other instruction is refused by name, and so is anything this does not
model: an output multiplier, a packed operation that asks for the second half
of a constant. A wrong guess would run and give a wrong answer, which is worse
than a refusal. An error while a kernel runs names the instruction: the
kernel, how far into it, and the instruction as the assembler writes it. And
`VGPU_TRACE_LAUNCHES=1` prints each launch -- the kernel, its grid and
block, its LDS -- which is how to see what a library such as rocBLAS runs.

Three things are modelled rather than copied, and are marked where they are
written: the reciprocal, square root, exponent and logarithm are the host's
exact results where the hardware's are tables good to about one unit in the
last place; the scope bits on a memory instruction change nothing, since every
access here is already visible to every wave; LDS and a work-item's private
memory sit at addresses of this model's choosing, which a kernel reads from
`src_shared_base` and `src_private_base` the way it reads the hardware's; and the counter a wave reads to time itself counts the
instructions retired by the host thread running it, which is this model's
cycle, where a card's counts at a fixed rate.

The MODE register's float modes are the kernel's own. A wave starts with
the round and denormal modes its descriptor gives (COMPUTE_PGM_RSRC1), and
`s_setreg` changes them. Each vector instruction runs under them, the round
mode through the host's rounding and the denormal mode through its flush
controls (x86's DAZ and FTZ). Every kernel in PyTorch's and ROCm's libraries
asks for the defaults: round to nearest even, denormals kept. Code built with
`-fgpu-flush-denormals-to-zero` flushes single-precision denormals, as a card
does. The matrix instructions ignore MODE, as the ISA says. Half-precision
denormals are always kept, whatever MODE says. `tests/hipcc/numerics.cpp`
checks all of it, built both ways.

Work-groups run on every host core at once, as a GPU runs them in any order
and concurrently; `VGPU_THREADS=1` runs them one after another, in order, which
is what a kernel with a data race needs to give the same answer every time.
Device memory is shared between the threads a word at a time, an atomic takes
a lock striped by address, and a fence (`buffer_wbl2`, `buffer_inv`) is a
fence on the host. `amd_exec_bench` (`tools/exec-bench.cpp`) says how fast a
kernel runs on the interpreter.

## gfx950 (MI350X)

`VGPU_GPU=amd/mi350x` runs code built for gfx950. PyTorch's own kernels and
hipBLASLt's, rocBLAS's and MIOpen's gfx950 builds pass the same 20 PyTorch
checks as gfx942 (ctest `amd_pytorch_mi350x`). What gfx950 adds is
modelled:

- **Instructions:**
  - `v_bitop3` (any function of three inputs, by a truth table)
  - the packed float-to-half and float-to-bfloat16 conversions
  - `v_permlane16_swap` and `v_permlane32_swap`
  - `v_cvt_f32_bf16`
  - the bfloat16 dot products
  - the matrix instructions with K doubled (f16, bf16, int8)
  - the `f8f6f4` ones, whose sources are fp8, bf8, fp6, bf6 or fp4, as each one's CBSZ or BLGP says
- **8-bit floats:** the same fp8 instructions mean the OCP formats on gfx950 (E4M3 and E5M2), where gfx942's are FNUZ. Which one applies is read from the code object's target. `tests/hipcc/fp8.cpp`, built for each, checks every conversion against HIP's own software one.
- **Block-scaled matrix instructions:** `v_mfma_scale_*_f8f6f4` scales each lane's row and 32 values along K by an E8M0 byte of its scale register, the byte `{OP_SEL_HI, OP_SEL}` names, as the CDNA4 ISA reference guide says. `v_prng_b32` is its LFSR step. `tests/hipcc/gfx950.cpp` checks both, and the lane swaps, against the guide.
- **Sparse matrix instructions:** `v_smfmac_*`, gfx942's and gfx950's (whose K is twice as long), take A 2:4 sparse along K: two values held of each four, and an index register that says which of the four each is. CBSZ and ABID choose a set of indices in that register. On gfx950 their B is eight registers split into two runs of K, K/2 apart, where A's run is in one piece; the guide's B tables show this, and hipSPARSELt's gfx950 kernels are right only with it. `tests/hipcc/smfmac.cpp`, built for each target, checks every form against the guides, and hipSPARSELt checks them in real GEMMs.
- **Guide semantics beyond the core matrix instructions (gfx942, gfx950):** matrix BLGP/EXEC/MODE/sparse-pruning rules (`tests/hipcc/mfma_modes.cpp`); instructions these chips added over gfx90a, such as the scalar saveexec/bitset/quadmask family, `v_cube*`, `v_sin/cos_f16`, `v_pk_fmac_f16` and gfx950's `v_ashr_pk_*` and 3-input minimum/maximum (`gfx9_added.cpp`); and ALU rules where the CDNA3/CDNA4 guides differ from older chips: float min/max signed zero and signaling NaNs, OMOD ignored under MODE.IEEE, half-precision absolute values, `v_div_fixup` sign, integer and conversion CLAMP, 32-bit literals sign-extended in 64-bit operands, DPP source-1 modifiers, `ttmp`/`src_scc`/`src_vccz` sources (`gfx9_semantics.cpp`). gfx90a and older chips keep their previous behavior (`is_guide_arch`). Each program is built for both chips and checked in `e2e/run_hipcc.sh`.
- **What the guides settle, second pass (gfx942, gfx950):** `v_cvt_pkaccum_u8_f32`, `v_madmk_f16`, `v_madak_f16`, `s_movrels_*` and `s_movreld_*` (an address past s101 is refused: the guide does not say), `v_mfma_f64_4x4x4_4b_f64` (four blocks, in the layout of the guide's section 7.1.4), gfx950's `v_pk_minimum3_f16` and `v_pk_maximum3_f16`, and its widening `v_cvt_scalef32_*` (4-, 6- and 8-bit floats times 2^(scale - 127), the scale the exponent field of the second source). The float `v_dot2` forms flush denormal inputs and outputs whatever MODE says; `v_mad_f16`, `v_mad_u16`, `v_mad_i16`, `v_fma_f16` and `v_div_fixup_f16` keep the destination's high half when OPSEL[3] is clear, where the "legacy" forms zero it; MODE.FP16_OVFL clamps overflowed half results and the F32 to FP8/BF8 conversions' overflow as the guides' table says (`tests/hipcc/gfx9_decode.cpp`). Left refused because the guides are silent or contradict themselves: the stochastic-rounding conversions (`v_cvt_sr_f16_f32`, `v_cvt_sr_bf16_f32`, `v_cvt_scalef32_sr_*`; see the third-pass entry below), `v_div_scale`/`v_div_fmas`'s scaling and `v_exp_legacy_f32`/`v_log_legacy_f32`. (The narrowing `v_cvt_scalef32_*` and `v_cvt_scalef32_f16_fp8`/`_bf8` that this list once held are implemented: see the third pass.)
- **Scaled narrowing conversions (gfx950 and gfx1250), from other public sources:** `v_cvt_scalef32_pk_{fp8,bf8}_{f32,f16,bf16}`, `_pk_fp4_*`, `_2xpk16_{fp6,bf6}_f32` and `_pk32_{fp6,bf6}_{f16,bf16}` (opcodes 565, 566, 573, 576, 577, 580, 581, 588, 589, 594, 595, 600 to 603 of the CDNA4 guide), gfx1250's `_pk8_*` and `_pk16_*`, and gfx950's `v_cvt_scalef32_f16_fp8` / `_bf8` (586, 587). The guides give where every source and result goes and leave `f32_to_fp8_scale` and its kin undefined. The value converted is `S / 2^(scale - 127)` (the scale the exponent field of its operand), rounded to nearest even: Composable Kernel's `cvt_float_to_fp8_scaled` (`include/ck/utility/mxf8_utils.hpp`, `cast_to_f8(f / scale, rng)`; "If fval / scale > max fp8, returns Nan") and its device tests on a gfx950 (`test/data_type/test_mx_fp8.cpp`: scale 2.0 turns {-8, 4} into {-4, 2}; infinity / 2, 256 / 0.5 and -256 / 0.5 are NaN in fp8; `test_mx_fp4.cpp`, `test_mx_fp6.cpp`: they are the largest value in fp4 and fp6, and 4.75 / 4 is 1.25), HIP's host reference (`amd_hip_ocp_host.hpp`, `fcbx::from_float`: `srcExp -= scale_exp`). To 8-bit floats the overflow rule is the guide's table for `v_cvt_pk_fp8_f32` applied after the scaling, with MODE.FP16_OVFL ("Conversion to F4/F6 does not support FP16_OVFL, while to F8 does"); to 4- and 6-bit floats a value past the largest saturates and one under the least subnormal becomes a zero, both keeping the sign (the guide's 7.3). The pair forms write the half (fp8: OPSEL[3]) or byte (fp4: OPSEL[3:2]) they name and keep the rest; `v_cvt_scalef32_f16_fp8` / `_bf8` write the half OPSEL[3] names from the byte OPSEL[1:0] names and keep the other half (the builtin's old-destination argument in clang's `BuiltinsAMDGPU.td`, LLVM's tied `vdst_in`). `tests/hipcc/scalef32_narrow.cpp` and `scalef32_narrow1250.cpp` check every form against HIP's host reference over a sweep of values and against the cases Composable Kernel's tests expect. Refused, with the reason in the message: a NaN into a 4- or 6-bit float (Composable Kernel's tests give the largest value, AMD's emulator tests -- `rocm-systems/emulation/rocjitsu/tests/hip_cvt_narrow_test.cpp`, written for a gfx950 -- zero), a scale of 255 (E8M0's NaN), input modifiers, clamp and omod on these conversions. A value between an 8-bit format's largest and the halfway point to the next step is a NaN here (the guide's "greater than max", as `v_cvt_pk_fp8_f32` has it); HIP's host reference and rocjitsu round first and give the largest value.
- **Stochastic rounding, left refused (gfx950):** `v_cvt_scalef32_sr_*`, `v_cvt_sr_f16_f32` and `v_cvt_sr_bf16_f32` add the seed to the bits the narrowing drops and cut them off; which bits of the seed those are is where the public sources part. The CDNA3 and CDNA4 pseudocode of `V_CVT_SR_FP8_F32` adds `S1[31:12]` (the seed's top 20 bits; `S1[31:11]` for BF8) and its prose says "not use msbs of src1 mantissa" (the low ones); HIP's older `hip_fp8.h` (`cast_to_f8`) and Composable Kernel's `f8_utils.hpp` add the low bits (`rng & drop_mask`), HIP's newer `fcbx::from_float_sr` and rocjitsu (`f32_to_fp8_e4m3_sr`, `seed >> 12`) the top ones; for the half and bfloat16 forms the guide leaves `f32_to_f16_sr` undefined and only rocjitsu defines it. The existing `v_cvt_sr_fp8_f32` / `_bf8_f32` use the low bits (what `tests/hipcc/fp8.cpp` compares with HIP's header); the rule is unsettled, not confirmed. Each of the others says so by name when a kernel reaches it (`test_amd_gcn_cdna_gaps`).
- **Decoder check:** `tests/data/isa_corpus_gfx950.txt` holds every instruction shape PyTorch's, hipBLASLt's and rocBLAS's gfx950 code uses. The decoder must print each one as `llvm-objdump` does.
- **Extracting code objects:** `amd_fatbin_extract` (`tools/fatbin-extract.cpp`) writes out what a library carries for one target, which is how that corpus is gathered.

## gfx90a (MI250X)

`VGPU_GPU=amd/mi250x` is one of an MI250X's two dies, as HIP, rocminfo and rocm-smi each see it: 110 compute units and 64 GB. `VGPU_DEVICE_COUNT=2` gives the whole package. PyTorch's 20 checks pass on it (ctest `amd_pytorch_mi250x`). The fp8 one checks that PyTorch refuses fp8, as it does on the card.

- **Instruction numbering:** gfx90a (CDNA2) numbers some instructions differently from gfx940 and later, so the decoder takes the code object's target (`gcn::Target`, from `e_flags`). The differences were found by asking LLVM's disassembler about every opcode of each encoding on both targets:
  - int8 matrix instructions with K of 8 and 16
  - the bfloat16 ones four a lane (`_1k`), renumbered on gfx940
  - the older bfloat16 ones two a lane
  - `v_mad_f32`, `v_mad_legacy_f32`, `v_mac_f32` and `v_madmk_f32`: the product rounded before the add, and denormals flushed
  - `buffer_wbinvl1` and `buffer_invl2`
  - FLAT_SCRATCH as a register
- **Scratch:** gfx90a has no flat scratch set up by the hardware. A kernel reaches its private memory through the private segment buffer resource it is handed, with buffer loads and stores the card swizzles across the wave's lanes. The executor hands it that resource and a wave offset, and unswizzles its accesses. This is how rocBLAS's gfx90a kernels spill.
- **Decoder check:** `tests/data/isa_corpus_gfx90a.txt` holds 1628 instruction shapes from PyTorch's, hipBLASLt's and rocBLAS's gfx90a code. Each decodes, and prints in gfx90a's names (`glc`, `slc`, `v_mfma_f32_32x32x8f16`), as `llvm-objdump` prints it.

## What the CDNA parts still refuse

A coverage probe keeps this list honest: every instruction `llvm-mc` will disassemble for a target (every opcode of each encoding,
with and without DPP and SDWA, memory forms with and without the LDS bit: 1,894 forms for gfx90a, 1,922 for gfx942 and 2,031 for gfx950,
and 4,400 to 5,300 more a target in modifier variants) goes into a kernel of its own, and each kernel runs in a process of its own; what stops with
"decoded but not implemented", "not decoded" or any other unsupported error is the list. gfx90a, gfx942 and gfx950 had 239, 167
and 211 plain forms in that state before this pass and 105, 66 and 109 after (the rest are a probe's zero resource descriptors, which no
format or image instruction can read, jumps to address 0, and what the list below says is left); gfx1250's corpus of 4,549 forms went from 34 to 10; the CDNA2, CDNA3 and CDNA4 guides settle most of them, and `tests/unit/test_gcn_cdna_gaps.cpp`
(kernels in `tests/data/asm_cdna_*.s`, one source for the three parts, with the expected values written again from the guides'
pseudocode) runs what they say:

- 64-bit and double buffer atomics (`buffer_atomic_*_x2`, `_add_f64`, `_min_f64`, `_max_f64`) and gfx950's `buffer_atomic_pk_add_bf16`; the
  buffer atomics share FLAT's read-modify-write code now.
- Buffer loads into LDS of a byte, a half and `format_x` (the guides' figure: LDS address `M0[15:0] + offset + 4 * lane`, the data a
  dword, as `buffer_load_ubyte` and the others say "zero extend to 32 bits"), and on gfx90a the dword form too. `global_load_lds_{ubyte,sbyte,ushort,sshort}`
  and `scratch_load_lds_*` on gfx942 and gfx950; FLAT's LDS offset is `M0[17:2] * 4`.
- The rest of LDS's atomics: `ds_mskor_*`, `ds_wrxchg2_*` and `ds_wrxchg2st64_*`, `ds_cmpst_b64`, `ds_cmpst_f64` and `ds_cmpst_rtn_f64`
  (the float compare), `ds_wrap_rtn_b32`, `ds_condxchg32_rtn_b64`, `ds_nop`, `ds_append` and `ds_consume` (on LDS), the ADDTID pair
  (`M0[15:0]`), and gfx942's `ds_pk_add_*`.
- Scalar buffer stores and atomics (`s_buffer_store_*`, `s_buffer_atomic_*`), the cache instructions `s_dcache_*_vol`,
  `s_dcache_discard*` and gfx90a's `s_atc_probe*` (nothing a program can read changes), `s_setvskip` (vector instructions are skipped),
  `s_set_gpr_idx_idx` and `_mode`, `s_setreg_b32`, `s_setkill`, `s_cbranch_cdbg*` (not taken: nothing sets the flags), the
  performance and trace instructions, `v_clrexcp`.
- Formatted buffer accesses (`buffer_{load,store}_format_*`, `tbuffer_*`) on gfx942 and gfx950: the guides list them, and the
  executor had only gfx90a's decoder rows for them.
- gfx90a's `v_sin_f16`, `v_cos_f16`, the cube-map instructions, `v_cvt_rpi_i32_f32`, `v_cvt_off_f32_i4`, `v_frexp_*_f16`, `v_ldexp_f16`,
  `v_fma_legacy_f16`, `v_mad_legacy_i16`, `v_pk_fmac_f16` and `v_sat_pk_u8_i16`, which its guide defines (gfx942 had them already). The
  half NaN of a sine or cosine of an infinity is 0xfe00, as the guides' examples have it.

Left refused, with what the guides say (or do not):

- `s_cbranch_g_fork`, `s_cbranch_i_fork`, `s_cbranch_join`: the stack is "six stack entries of 128-bits each, stored in SGPRS" at
  `SGPR[CSP*4]`, and "requires three SGPRs for each fork/join block"; where it starts and what CSP is at the start are not said.
- `s_rfe_b64`, `s_rfe_restore_b64`: "may only be used within a trap handler", and `s_trap` ends the launch here. `s_sethalt 1`
  and `s_sendmsghalt` halt the wave, and nothing resumes a halted wave in this simulator.
- `s_scratch_load_*` and `s_scratch_store_*`: `CalcScalarScratchAddr` is not defined. `buffer_store_lds_dword` (gfx90a): "Store one DWORD
  from LDS memory to system memory without utilizing VGPRs", and no LDS address.
- The image `_pck` forms and LWE (see Textures).
- `v_exp_legacy_f32`, `v_log_legacy_f32` (the guides give `pow(2.0, S0.f)` and `log2(S0.f)` and not how "legacy" differs),
  `v_screen_partition_4se_b32` (only in the opcode table), and on gfx90a `v_mac_f16` (not said whether the product is rounded before the add),
  `v_cvt_norm_i16_f16`, `v_cvt_norm_u16_f16`, `v_cvt_pkaccum_u8_f32` and `v_mad_legacy_f16` (only in the table).
- gfx950: `ds_read_b96_tr_b6` ("Two instruction load a complete matrix. The first loads K values (0..15, 32..47)"), the stochastic `v_cvt_scalef32_sr_*`, `v_cvt_sr_f16_f32` and `v_cvt_sr_bf16_f32` (which bits of the seed are added is where the guide and AMD's
  software part: see "Stochastic rounding, left refused"), `v_mfma_ld_scale_b32` (not in the guide), and an M0 of 64 KiB or more for MUBUF's LDS loads (CDNA4's text says `M0[17:0]`, its
  figure `M0[15:0]`; gfx90a's and gfx942's text and figure agree on 16 bits, which is what they use).
- `global_load_lds_dwordx3` writes 12 bytes a lane: the guide gives only DWORDX1 (4 bytes) and DWORDX4 (16) for FLAT, and 16 for the buffer
  form's "3 or 4 dwords".

## gfx1250 (Instinct MI455X)

`VGPU_GPU=amd/mi455x` is gfx1250 (CDNA 5): wave32, built on gfx12's instruction set with matrix, tensor, cluster and
64-bit additions. **No MI455X was available: nothing here has been checked against a card.** The oracle is AMD's
published CDNA 5 ISA (the machine-readable XML) and the assembler and disassembler of ROCm 7.2.3's LLVM (22.0.0git),
which knows `gfx1250` and `gfx1251`. Where the two disagree this says so. The profile (`profiles/mi455x.yaml`)
separates what is sourced (432 GiB of HBM4, 256 workgroup processors, 192 MB of last-level cache, 320 KB of LDS from
LLVM) from what is a placeholder (clocks, board power, the PCI device id, register-file figures); it is `verified:
false` throughout. gfx1251 (MI430X) is not covered.

- **Decoder:** `src/rdna_ops_cdna5.inc` is generated by `tools/rdna-ops.py` from the CDNA 5 XML. `tools/isa-fuzz.py`
  writes `tests/data/isa_corpus_gfx1250.txt` from the same XML: every instruction's plain encodings with zero, random
  and sparse operand fields, its 32- and 64-bit literal forms, its DPP forms and a few written in assembly, each with the
  text `llvm-mc` prints for it. The decoder reproduces all of them (ctest `test_amd_gcn`). Left out of the corpus on
  purpose, because LLVM's text for them follows rules that do not change what the instruction does: `s_sendmsg*`,
  `s_wait_alu`, `ds_swizzle_b32`, the prefetch and ATC-probe scalar forms, a 32-bit literal in a 64-bit scalar
  operation, and a few 16-bit literal and inline-constant spellings.
- **What the compiler's output needs, and runs:** the 17 unit-kernel suites the other targets run, built for gfx1250
  by ROCm's clang, all pass (ctests `*_gfx1250`), and the HIP programs in `tests/hipcc` built by ROCm 7.2's hipcc run on
  a simulated MI455X (`run_hipcc.sh`). For that the executor learned: SCALE_OFFSET (the offset register times the
  size of the access, signed, in global, flat, scratch and scalar loads); flat accesses with a scalar base; the
  three-word VOPD3 dual-issue pairs (with `v_dual_bitop2_b32`'s truth table in the third source's field); 64-bit
  PC-relative relocations (`R_AMDGPU_REL64`, how a kernel finds a global or a function); `s_get_pc_i64`,
  `s_set_pc_i64`, `s_swap_pc_i64`, `s_add_pc_i64`, `s_call_i64`; `src_vccz` and `src_execz`; the wave's number from
  `HW_REG_IB_STS2`; `v_add_nc_u64`, `v_sub_nc_u64`, `v_mul_u64`, `v_min`/`v_max` of 64-bit integers,
  `v_mad_u32`, `v_mad_nc_u64_u32`, `v_mad_nc_i64_i32`, `v_add_max`/`v_add_min` (clamped add, then the other
  operand); `v_ashr_pk_i8_i32`, `v_sat_pk4_*`, `v_tanh_{f32,f16,bf16}` and the bfloat16 transcendentals, fp8 to half
  conversions; f64 atomics, `*_prefetch_b8` (a no-op) and `*_load_monitor_*` (a load).
- **Matrix instructions, executed from AMD's CDNA 5 ISA document:** `v_wmma_*` (f16, bf16, f32, fp8/bf8, iu8, the
  mixed 8-, 6- and 4-bit `f8f6f4` forms, and the block-scaled forms with E8M0, E5M3 and E4M3 scales), and the sparse
  `v_swmmac_*`. The lane layouts, the sparse index words and the scale selection come from the document's tables and
  pseudocode. Each product is summed exactly and rounded once, so the last bit can differ from hardware that sums in a
  fixed order; `FP16_OVFL` (MODE bit 23) saturates a 16-bit float result (see below). EXEC must be all ones. No card has run them:
  `tests/hipcc/wmma1250.cpp` checks them against the host, and the dense checks hold for any order of K, while the
  sparse, mixed-width and scaled ones depend on the layout read from the document.
- **Arithmetic the compiler seldom emits** (`tests/hipcc/alu1250.cpp`, each against the ISA's pseudocode written out for
  the host): the IEEE `v_minimum_*` and `v_maximum_*` families in f16, f32 and f64 (a signaling NaN comes back quieted,
  -0 is below +0) with their three-operand and combined forms, the `_num` three-operand forms (a number beats a NaN;
  they keep their own names rather than the older instructions' they replaced), 16-bit integer add, subtract, min, max
  and shift, `v_cvt_i32_i16`, `v_cvt_u32_u16`, `v_sat_pk_u8_i16`, half sine, cosine, frexp and ldexp,
  `v_cvt_nearest_i32_f32`, `v_cvt_off_f32_i4`, `v_cvt_norm_*_f16`, the DX9 and lighting multiplies, `v_sqrt_f64`,
  the packed bfloat16, half and 16-bit integer instructions (`v_pk_*_bf16`, `v_pk_fmac_f16`, `v_pk_{minimum,maximum}*_f16`,
  `v_pk_add_{max,min}_*16`, `v_pk_{max,min}3_*16`), and the scalar `s_{minimum,maximum}_f{16,32}`, `s_{ceil,floor,trunc,
  rndne}_f16`, `s_cvt_pk_rtz_f16_f32`, `s_quadmask_*`, `s_bitset{0,1}_b64`, `s_bitreplicate_b64_b32`, `s_cls_i32_i64` and
  the `s_{nand,nor,xnor}_saveexec_*`, `s_and_not{0,1}_wrexec_*` and further EXEC-writing forms.
- **Named barriers:** `s_barrier_init`, `_join`, `_signal` (with the member count in M0), `_signal_isfirst`, `_wait`,
  `_leave`, `s_get_barrier_state` and `s_wakeup_barrier`, from the ISA document's section 5.6; the work-group barrier's
  signal count is kept too, for `isfirst`. The trap and cluster barriers do nothing (a cluster is one work-group here),
  every named barrier counts as allocated, and a wave that ends is not taken out of a barrier's member count. A wait
  that no wave can end is an error naming the barrier, not a hang (`tests/hipcc/barrier1250.cpp`).
- **Cross-lane permutes:** `v_permlane16_var_b32`, `v_permlanex16_var_b32`, `v_permlane_{bcast,up,down,xor}_b32` (lane
  groups of a power-of-two width a scalar gives; any other width is refused, the ISA leaving it undefined),
  `v_permlane_idx_gen_b32`, and `v_permlane16_swap_b32` (writing only the lanes that are on), from the ISA document's
  pseudocode (`tests/hipcc/permlane1250.cpp`).
- **Asynchronous copies:** `global_load_async_to_lds_b{8,32,64,128}` and `global_store_async_from_lds_*` move each
  lane's bytes between its global address and its LDS address, as the ISA document's pseudocode has it; the copy is done
  when the instruction is, so `s_wait_asynccnt` has nothing to wait for (`tests/hipcc/async1250.cpp`).
- **Transposing loads:** `global_load_tr16_b128`, `global_load_tr8_b64`, `ds_load_tr16_b128` and `ds_load_tr8_b64`, which
  load a 16x16 tile held column by column straight into the registers of a WMMA operand (lane map from the ISA
  document's figure; `tests/hipcc/trload1250.cpp` feeds them to a WMMA). EXEC must be all ones.
- **Vector registers past v255:** a wave's register file grows to what a kernel names, up to the 1024 a gfx1250 wave may
  have, and `s_set_vgpr_msb` puts the two high bits on the numbers each following instruction's destination and sources
  name (VALU, FLAT/GLOBAL and DS instructions; the buffer and image forms and `v_movrel*` are not covered), or a write to
  MODE's bits 19:12 does (see below). `tests/hipcc/vgprs1250.cpp` keeps 600 values live at once.
- **DPP8** (all eight selects, with and without `fi`, on the VALU and compare instructions) and DPP on the compares, which
  were refused.
- **LDS atomics and exchanges** (`tests/hipcc/ds1250.cpp`): `ds_cond_sub_*`, `ds_sub_clamp_*`, `ds_pk_add_{f16,bf16}` (with and
  without a return), `ds_mskor_*`, `ds_cmpstore_b64`, `ds_storexchg_2addr_*` (and the stride-64 forms), `ds_condxchg32_rtn_b64`,
  `ds_store_addtid_b32` / `ds_load_addtid_b32` (which named no address register, and crashed the executor before) and
  `ds_bpermute_fi_b32`; and flat pointers to private memory, which gfx1250 builds from `src_flat_scratch_base` with the
  lane's number in bits 56:52 (`tests/hipcc/gfx1250.cpp`).
- **Operand forms** (`tests/hipcc/forms1250.cpp`), found by running every form the assembler accepts through the simulator: the
  CLAMP and OMOD bits on half-precision, packed and single-precision results (a -0 clamps to +0, as the ISA document says), clamped
  16-bit and packed integer sums (the packed signed ones read their halves unsigned before), OPSEL on a scalar or literal source,
  the byte or half of `v_cvt_f32_fp8` / `v_cvt_pk_f32_fp8` that OPSEL picks, DPP on three-source (VOP3) instructions, DPP8 on
  compares, and the VOPD dot products `v_dual_dot2acc_f32_{f16,bf16}` (in ROCm's LLVM but not the XML; they are the VOP2
  accumulates issued as half of a pair). OPSEL on an inline constant is still refused (the document says it must be zero).
  A second reading of the ISA document's pseudocode (every per-lane arithmetic instruction run over edge cases and compared with
  the document) corrected: the "_num" minimum and maximum (a number beats a NaN, -0 is below +0, two NaNs give the first,
  quieted), `v_fmamk_f16` (it computed `a * b + K`), `v_cvt_i32_f64` (out-of-range saturates, a NaN is zero), `v_bfe_i32` and
  `s_bfe_i32/i64` (an arithmetic shift), `v_lshl_add_u64` (a count over 4 is a shift of zero), `v_dot4_i32_iu8` and
  `v_dot8_i32_iu4` (neg_lo says which source is signed), `v_cmp_class` on a signaling NaN, `v_cvt_f16_fp8`'s byte_sel,
  `s_absdiff_i32` (32-bit), `s_max_i32/u32` (SCC set on a tie), `s_bitcmp0/1_b64`, and `v_sin_f32` / `v_cos_f32` over the full range
  of a float. Running rocPRIM, hipCUB and rocThrust on gfx1250 found two more: scratch accesses with both a VGPR and an SGPR
  offset add them as signed numbers (a negative VGPR offset and a positive SGPR one cancel), and `v_add_min_u32` wraps its sum.
  For that last one the ISA document disagrees with itself (the prose says the add is clamped, the pseudocode is
  `v_min_u32(v_add_nc_u32(...))`); the pseudocode is followed, because AMD's compiler emits the instruction for
  `min(a + b, c)` over wrapping unsigned values and rocPRIM's radix sort relies on it. The signed forms and the packed ones
  keep the "clamped" reading, which nothing contradicts. Known differences left alone: `v_div_fixup` does not apply the sign of the
  denominator and numerator to the quotient's magnitude, the SR conversions' rounding (the document contradicts itself), and
  `v_div_fmas`' 2^64 scale (the document says 2^32, which looks like a typo beside `v_div_scale`).
- **`v_fmamk_f64` and `v_fmaak_f64`** (VOP2 35 and 36), which the XML leaves out but the ISA document lists ("imply the use of a 64-bit
  literal") and the compiler emits for double-precision math (found by hip-tests' `Unit_hipTrigDeviceFunc_Double`). They are
  added to the generated table by `tools/rdna-ops.py` as supplementary rows.
- **Kernel argument preloading:** where the kernel descriptor asks for the first dwords of the arguments in the last user SGPRs
  (`kernarg_preload`, bytes 58-59), the wave starts with them there. gfx942's compiler output carries a prologue that loads
  them itself, which is why only gfx1250's kernels (hip-tests' `Unit_KerArgOptimization_Saxpy`) showed the gap
  (`tests/hipcc/preload1250.cpp`, built with 3 and with 16 dwords preloaded).
- **Cube faces and lookup-table permutes:** `v_cubeid_f32`, `v_cubesc_f32`, `v_cubetc_f32`, `v_cubema_f32`, and
  `v_perm_pk16_b{4,6,8}_u4` (sixteen lookups, by 4-bit indices, in a table of sixteen 4-, 6- or 8-bit entries).
- **Conversions:** `v_cvt_pk_{fp8,bf8}_f16` (nearest even, into the half of the destination op_sel names), the stochastic
  `v_cvt_sr_{fp8,bf8}_f16` (the seed's top bits added to the half's mantissa, as the ISA's pseudocode has it),
  `v_cvt_sr_pk_{bf16,f16}_f32` and `v_fma_mix{,lo,hi}_bf16`.
- **Tensor Data Mover:** `tensor_load_to_lds` and `tensor_store_from_lds`, from the ISA document's descriptor tables: tiles of
  one to five dimensions, 1- to 8-byte elements, zero fill (load) or dropped writes (store) outside the tensor, LDS padding
  on loads, gather and scatter by 16- or 32-bit row indices, and iteration (`tests/hipcc/tensor1250.cpp`, which builds
  each descriptor bit range by bit range from the tables). Done when issued. Not modelled: multicast to other work-groups
  (a cluster is one work-group here, so the mask is ignored), and the LDS barrier a descriptor can ask to be signalled
  (`atomic_barrier_enable` is refused, since the ISA's text does not fix that barrier's width).
- **Block loads and stores** (`global_load_block`, `global_store_block`: up to 32 dwords to or from consecutive registers, by the
  mask in M0; a scaled offset is refused) and the **cluster loads** (`cluster_load_b{32,64,128}` and `cluster_load_async_to_lds_*`,
  run as the global loads they downgrade to outside a cluster; `tests/hipcc/block1250.cpp`).
- **The 6- and 4-bit transposed loads** (`global_load_tr6_b96`, `global_load_tr4_b64`, and the `ds_` ones; `tests/hipcc/forms1250.cpp`):
  the document's figure labels each of the 32 K columns of the 16x32 tile with the lane whose address supplies it (16 elements along M, 96 or
  64 bits) and the code follows it. The figure draws no destination for these two kinds, so that half is assumed from the 16- and 8-bit
  kinds': lane = M + 16 * (K / 16), the 16 elements of a lane along K in order from bit 0.
- **MODE and the VGPR MSB bits:** `s_setreg_b32` / `s_setreg_imm32_b32` on MODE bits 19:12 now set the registers the next instructions name,
  as `s_set_vgpr_msb` does, and `s_getreg_b32` of MODE shows what `s_set_vgpr_msb` set. The scalar prefetches (`s_prefetch_inst`,
  `s_prefetch_data`, their PC-relative forms and `s_buffer_prefetch_data`) do nothing (no cache to warm). `v_cvt_pk_i16_f32` and
  `v_cvt_pk_u16_f32` convert toward zero with saturation and a NaN as zero (assumed: the document's helper functions are not defined).
- **The scaled widening conversions** (`v_cvt_scale_pk{8,16}_{f16,bf16,f32}_{fp4,fp6,bf6,fp8,bf8}`; `tests/hipcc/forms1250.cpp`): the value times
  2^(scale - 127), the scale an E8M0 byte of S1. Which byte, of which lane's S1, a lane takes is the document's Table 36 (4- and 6-bit) or
  Table 37 (8-bit): the code follows the tables (the pseudocode's `S1[OPSEL * 8 +: 8]` reads only the lane's own register and agrees with them
  only where S1 is alike in every lane); Table 37's reserved rows (block16 with OPSEL[2] set) are refused. Assumed, not stated: a scale of
  0xFF gives NaN, and the result is exact but for an overflow, then rounded to nearest even (denormals kept).
- **`MODE.FP16_OVFL`** (bit 23, gfx1250): set, an overflow of half-precision `add`, `sub`, `mul`, `fma`, `mad`, `fmac`, `ldexp` and `cvt_f16_f32` (and the packed
  ones) gives the largest finite half, not infinity, and so do WMMA results of 16-bit float type, the scaled widenings to f16 or bf16, and f32 to
  fp8/bf8 conversions (where an infinity too becomes the largest value, as the document's table says). An infinite operand stays infinite. The
  operations whose infinite result may be a divide by zero (`rcp`, `log`) are left alone.
- **The LDS barrier arrives** (`ds_atomic_barrier_arrive_rtn_b64`, `ds_atomic_async_barrier_arrive_b64`, and the Tensor Data Mover's
  `atomic_barrier_enable`; `tests/hipcc/forms1250.cpp`, `tensor1250.cpp`): a 64-bit LDS word, state (pending count below, phase above) with the initial
  count in bits 47:32; an arrive subtracts 1 (or the _RTN form's DATA0), and below zero the pending count reloads and the phase has dropped by the
  borrow. The document leaves the pending count's width (`WIDTH`, 21 to 32 bits) set by nothing, and its two accounts of the roll-under test disagree
  at 32; this takes 31, the largest width with a phase (a program reading the phase as `state >> W` for any smaller W sees it change), and each
  active lane arrives in lane order. The TDM's arrive is made once after the move, to the word at `atomic_barrier_address[18:3]` from the start of
  the LDS. Nothing wakes a sleeping wave. `src_shared_limit` reads as the document defines it.
- **Clusters** (`tests/hipcc/cluster1250.cpp`, its kernel `cluster1250_kernel.cpp` built by `build.sh` into a code object that the program loads as a
  module): a kernel whose code object carries `.cluster_dims` (ROCm 7.2's clang has no source attribute; `build.sh` adds the `amdgpu-cluster-dims`
  function attribute to the kernel's IR) has its work-groups launched in clusters of that shape, up to 16, which run side by side. A
  work-group finds its place in TTMP6, TTMP7, TTMP9 and IB_STS2 as the document says (2.3), the compiler's own `blockIdx` and the
  `__builtin_amdgcn_cluster_*` builtins read them, and the cluster barrier (`s_barrier_signal -3`, `s_barrier_wait -3`) completes when every work-group
  still running has signalled. No ROCm 7.2 runtime consumes `.cluster_dims`, so no public launch interface exists; the simulator takes the cluster
  from the code object, whichever call launches it, and refuses a grid that is not whole clusters. Assumed, not stated: work-groups and clusters are numbered x
  fastest, and CLUSTER_ID is the cluster's number modulo 15, plus 1. The multicast of cluster loads is ignored (it moves the same data to the same
  places).
- **Decoded and printed, but refused when executed:** the stochastic `v_cvt_scalef32_sr*` families (where the stochastic rounding's random bits sit in the
  dropped bits is where the public sources part: see "Stochastic rounding, left refused" above; the rounding-to-nearest `v_cvt_scalef32_pk8_*` and
  `_pk16_*` are implemented, with the scale dividing, from Composable Kernel and HIP's host reference), `v_permlane64_b32` (assembled by LLVM, absent from the document, and gfx1250 is wave32 only), E5M3 (`CLAMP` on the fp8
  conversions: its largest value and infinity are not stated). Each says so by name
  when a kernel reaches it. The
  numerics test (`numerics.cpp`) is not run: the value MODE starts with on this part is unknown.
- **A disagreement worth knowing:** AMD's XML gives the scaled matrix instructions' two-word prefix the opcodes
  0x37/0xBD (32-bit scales) and 0x3B/0xBA (64-bit), which are also ordinary opcodes in the same XML (0x37 is
  `v_pk_maximum3_f16`). LLVM 22 writes 0x35 and 0x3A. The decoder takes LLVM's.

## RDNA3 (Radeon RX 7900 XTX)

`VGPU_GPU=amd/rx7900xtx` is a consumer card: gfx1100 (RDNA3), whose code is
built 32 lanes to a wave. PyTorch's ROCm wheel runs on it unmodified, with its
own gfx1100 kernels and rocBLAS's, hipBLASLt's, MIOpen's and rocFFT's, and
passes the same 20 checks as the Instinct parts (ctest
`amd_pytorch_rx7900xtx`). The fp8 matmul check is refused by PyTorch, as on
the card, because RDNA3 has no 8-bit floats.

- **Decoder:** gfx11 has its own encodings and instruction numbering. They come from AMD's machine-readable ISA specification (MIT), which `tools/rdna-ops.py` turns into `src/rdna_ops_rdna3.inc`. Every one of the 13 million distinct encodings in the wheel's gfx1100 code decodes and prints as `llvm-objdump` does. That includes VOPD pairs, 16-bit register halves (`v1.l`, `v1.h`), DPP8 and the null register. `tests/data/isa_corpus_gfx1100.txt` keeps 1,759 of their shapes for CI.
- **Execution:** an RDNA instruction runs as the gfx9 instruction that does the same. The decoder names it after that instruction, using the specification's own record of renamings. On top of that, gfx11 adds:
  - wave32 (and wave64, where a kernel's descriptor asks for it);
  - VOPD, whose two halves both read their sources before either writes;
  - `v_cmpx` writing EXEC alone;
  - the b32 exec operations;
  - 16-bit halves, and d16 loads that keep the other half (gfx942 with SRAM ECC zeroes it);
  - RDNA's buffer resource, with a 7-bit format and OOB_SELECT's bounds checks;
  - SMEM's offset register;
  - 106 scalar registers;
  - WMMA;
  - the scheduling hints (`s_clause`, `s_delay_alu`).
- **Checks:** the executor's unit kernels are built for gfx1100 too, and pass every check they pass on gfx942 (ctests `*_gfx1100`). `tests/hipcc/rdna3.cpp` checks VOPD, WMMA in each type and 64-bit literals in wave32, and DPP and output modifiers in wave64.

## RDNA4 (Radeon RX 9070 XT)

`VGPU_GPU=amd/rx9070xt` is gfx1201 (RDNA4). PyTorch's 20 checks pass on it,
through its own gfx1201 kernels and the libraries' (ctest
`amd_pytorch_rx9070xt`). That includes the fp8 matmul, which hipBLASLt runs
with gfx12's OCP fp8 WMMA.

- **Decoder:** the same generator reads AMD's RDNA4 specification into `src/rdna_ops_rdna4.inc`. gfx12 keeps gfx11's scalar and vector encodings and replaces the memory ones: three-word VBUFFER and VFLAT/VGLOBAL/VSCRATCH encodings, 24-bit offsets, and a TH/SCOPE cache policy in place of glc/slc/dlc. All 12,671,278 distinct encodings in the wheel's gfx1201 code decode and print as `llvm-objdump` does. `tests/data/isa_corpus_gfx1201.txt` keeps 1,808 of their shapes.
- **Execution:** on top of RDNA3, gfx12 adds:
  - work-group IDs in trap registers (TTMP9, TTMP7);
  - split wait counters and the split barrier (`s_barrier_signal`, `s_barrier_wait`);
  - sub-dword scalar loads;
  - the scalar float unit (`s_add_f32`, `s_fmac_f32`, conversions, comparisons) and 64-bit scalar arithmetic;
  - the vector unit's transcendentals into scalar registers (`v_s_rcp_f32`);
  - `global_inv`/`global_wb`;
  - WMMA's new layout: each half of the wave holds half of K, and there is no replication;
  - OCP 8-bit floats.
- **Checks:** the executor's unit kernels pass on gfx1201 too (ctests `*_gfx1201`), and `tests/hipcc/rdna4.cpp` checks WMMA, the scalar float unit and the split barrier.

## RDNA2 (Radeon RX 6900 XT, RX 6800, RX 6700 XT)

`VGPU_GPU=amd/rx6900xt` is gfx1030 (RDNA2), and so is `amd/rx6800` (Navi 21, 60 CUs). `amd/rx6700xt` is
gfx1031 (Navi 22, 40 CUs), the same instruction set with a smaller chip. The RX 6800 and RX 6700 XT are
profiled from real cards (`registers/measurements/`): their device properties, `rocminfo`, `amd-smi`, KFD
topology and all 4096 bytes of PCI configuration space, which the simulated card replays (revision,
subsystem id, capability chain and BAR layout included). PyTorch's 20 checks pass on the RX 6900 XT
(ctest `amd_pytorch_rx6900xt`). RDNA2 has no matrix instructions, so its
matrix products run on the vector units, and PyTorch refuses fp8 on it, as it
does on the card.

- **Decoder:** the same generator reads AMD's RDNA2 specification into `src/rdna_ops_rdna2.inc`. gfx10 names its instructions as gfx9 does, and differs from gfx11 in a few places:
  - M0 is operand 124 and null is 125, the other way round from gfx11;
  - glc, slc and dlc sit in other bits, and so does FLAT's segment;
  - it has SDWA (sub-dword reads and writes) and no true16 or VOPD.

  All 12,542,606 distinct encodings in the wheel's gfx1030 code decode and print as `llvm-objdump` does. `tests/data/isa_corpus_gfx1030.txt` keeps 1,775 of their shapes.
- **Execution:** RDNA3's paths, plus:
  - a register per work-item id: gfx10 does not pack them into v0, whatever the code object's ABI;
  - SDWA's three ways of filling the rest of a destination (pad, sign-extend, keep);
  - `v_permlane16_b32` and `v_permlanex16_b32`, which RDNA3 and RDNA4 have too;
  - DPP's `row_share` and `row_xmask`, and its FI bit;
  - occupancy from gfx10.3's register file.
- **Checks:** the executor's unit kernels pass on gfx1030 (ctests `*_gfx1030`), and `tests/hipcc/rdna2.cpp` checks SDWA, M0-relative registers, the permlanes and the DPP modes, in wave32 and wave64. The hipcc-built programs for textures, `__smid`, work-group shapes, the memory test and `rdna2.cpp` were also run on a real RX 6800 and RX 6700 XT, and give the same answers there as on the simulated cards.
- **Where a wave runs:** `HW_ID1` carries the SIMD a wave is on (bits 9:8) as well as its workgroup processor. On gfx10.3 a card puts the waves of a group on SIMDs 0, 2, 1, 3 in turn and moves to the next slot after four; the simulator does the same, so HIP's `__smid` (whose lowest bit is the SIMD's) takes two values within a group, as on the card. `HW_ID2` is not modelled and a wave that reads it is refused by name.
- **`HSA_OVERRIDE_GFX_VERSION=10.3.0`:** an RX 6700 XT reports itself as gfx1030, as ROCm's runtime does, so the code PyTorch and the ROCm libraries ship for gfx1030 (they carry no gfx1031 kernels) loads on it; without it that code is refused. Other values are ignored.
- **sRGB textures:** the card's conversion to linear is approximate (within 2e-3 of the exact curve in the middle of the range); the simulator's is exact.

`VGPU_TRACE_WAVE=1` prints each instruction a work-group's first wave runs,
with what its destination holds after it for lane 0 (or the lane
`VGPU_TRACE_LANE` names). That is how the bugs above were found.

## Memory tests

`amd/tests/hipcc/memtest.cpp` is a HIP memory test written for this repository: walking ones and zeros, address in address, checkerboard and its inverse, moving inversions, seeded random data (verified on the device and on the host), block copies and strided sweeps, on `hipMalloc` memory and on mapped pinned host memory. It never looks at the wave size, so it runs the same on wave 64 (MI300X, MI250X) and wave 32 (RX 6900 XT, RX 7900 XTX). Checked-in builds for gfx942, gfx90a, gfx1030 and gfx1100 are run by ctest `amd_memtest_patterns`, which also arms the simulator's faults and requires the test to find them: a host-flipped bit exactly, a `vgpu fault stuck` cell, bit flips armed on stores, an uncorrectable HBM ECC error on a load (the kernel fails and a row is remapped; corrected errors are counted and change nothing; a card without ECC refuses the fault), and a wild pointer (`hipErrorLaunchFailure`). The programs were built with Ubuntu's ROCm 5.7 `hipcc` (the only toolchain installable without AMD's repositories), so they ask for `libamdhip64.so.5`, which the script maps to the shim; gfx950 and gfx1201 need a newer compiler. `ci/external/babelstream.sh` runs UoB-HPC's BabelStream (HIP), whose own validation must pass, and must fail against a stuck cell, in the nightly external suites.

## Textures

The Radeon GPUs (RDNA2, RDNA3 and RDNA4: gfx1030, gfx1100, gfx1201) have
texture units, and the simulator models them from the kernel up:

- **Instructions:** RDNA's image instructions (MIMG on gfx10 and gfx11,
  VIMAGE and VSAMPLE on gfx12) decode from AMD's ISA XML, and the executor runs
  loads, stores and atomics, point and bilinear samples with mip selection and
  the wrap, mirror, clamp and border modes, gathers and `get_resinfo`, and the
  formatted buffer loads and stores (`buffer_load_format_*`, `tbuffer_*`) that
  `tex1Dfetch` compiles to.
- **Resources:** an image resource (T#), a sampler (S#) and a formatted
  buffer's resource (V#) are laid out as ROCm's image runtime writes them, for
  each generation (`include/vgpu/amd_image.hpp`), with RDNA's single format
  codes. HIP's texture functions read some of those fields themselves (the
  height, the sampler's filter and normalization bits, the width after the
  resource), and do point-sampling's rounding in the kernel.
- **HIP:** arrays (1D, 2D, 3D, layered), every copy to and from them, texture
  objects over arrays, linear and pitched memory (runtime and driver API),
  texture references (`texture<T, dim, mode>`, bound with `hipBindTexture*`
  or a module's through `hipTexRef*`), sRGB textures (over arrays and pitched memory; sRGB over linear memory is refused by name: a buffer resource has no sRGB number format and nothing public says what `tex1Dfetch` would return), and surface objects
  (`src/hip_images.inc`), each check and error as ROCm's HIP makes it.
  ROCm's HIP on Linux has no mipmaps or cube arrays, so neither does the shim.
- **HSA:** the images extension (`src/hsa_images.inc`): images and samplers,
  import, export, copy and clear, and each agent's image limits.
- **Checks:** `tests/data/asm_images.s` checks each instruction against the
  host; `tests/hipcc/images.cpp` (21 checks) runs on all three generations, on
  the shim and on ROCm's own HIP over the HSA runtime; `tests/hsa/hsa_images.c`
  checks the extension.

The MI200 family (gfx90a, MI250X) has texture units too, and its instructions are MIMG in gfx9's own
encoding, which has no DIM field (the resource's TYPE and the DA bit say the shape) and, per the MI200
guide (12.15), exactly one sample, `image_sample` (implicit level of detail, no `_l`, `_lz`, `_b`, `_c`,
`_d`, `_o` or `_cl` forms, no `image_gather4`): hipcc drops `tex2DLod` and `tex2Dgather` from the code it
builds for gfx90a, so a program that calls them gets nothing there (that is the compiler's doing, not the
simulator's). The executor runs loads (`_mip` included), stores, atomics (on 32- and 64-bit texels, with the
DMASK values the guide lists), `get_resinfo` and `image_sample` on gfx90a through the same code as RDNA's, with the guide's rules where they differ (a store writes the
whole texel, the channels DMASK leaves out as zero; with d16 DMASK is only a count), and MUBUF's
`buffer_load_format_*` and `buffer_store_format_*` (opcodes 0-15, 38 and 39) that `tex1Dfetch` compiles to, and
MTBUF's `tbuffer_*_format_*`, whose data and number formats are the instruction's own and whose DST_SEL is the
identity (a channel the format lacks reads 0). A store of fewer components than the format has is refused:
the guide does not say what becomes of the rest.
The T# (Table 41 of the guide, ROCR-Runtime's `resource_ai.h`) names the format as GCN does, a data
format and a number format. Left refused, with the guide's words in the error: the `_pck` forms (the
guide says the resource's format is ignored and 8- and 16-bit components are zero- or sign-extended,
and nothing about which channels a DMASK past the texel's returns), LWE (where `LOD_CLAMPED` goes is not
said), 2D MSAA images and buffers behind a T#. `tests/data/asm_images90a.s`
(`test_amd_gcn_asm`), `tests/data/isa_corpus_images_gfx90a.txt` (`test_amd_gcn`, the decoder against
llvm-objdump), `tests/hipcc/images.gfx90a` (19 checks, on the shim and on ROCm's HIP over the HSA runtime)
and OpenCL's images (`tests/opencl/cl.c`) cover it.

The MI300 family (gfx942 and gfx950) has no texture units. hipcc refuses the
texture API in their device code (`__HIP_NO_IMAGE_SUPPORT`), and ROCm's HIP on
them reports image support 0 and answers every call that would make an array,
a texture or a surface with `hipErrorNotSupported`. The shim gives the same
answers, so a program or library that calls them is told what it would be
told on the card instead of failing to load. `tests/hipcc/textures.cpp` prints
each answer; its output must match `tests/hipcc/rocm/textures.expected` on the
shim, for each release's build of it, and on ROCm's own HIP over the HSA
runtime.

## HSA

The same library is also the HSA runtime (`src/hsa_api.cpp`, declared in
`include/vgpu/hsa_abi.h` from the HSA Foundation's specification and AMD's
documented extensions). `build/shim/libhsa-runtime64.so.1` names it, so a
program that uses HSA and HIP together sees one set of devices and one memory.
A program finds a CPU agent and one GPU agent per simulated device, allocates
from their memory pools (or the older regions), loads a code object into an
executable, and dispatches kernels by writing AQL packets into a queue and
ringing its doorbell. Each queue has a packet processor on a host thread of
its own. It runs kernel dispatches in order, holds on barrier-AND and
barrier-OR packets until their signals reach zero, and decrements each
packet's completion signal when it is done. `hsa_amd_memory_async_copy` waits
on its dependency signals the same way. Memory from the CPU's pools
(fine-grained, and kernarg) is reachable from every device's kernels at its
own address. Memory from a GPU's pool belongs to that device, and the host
reaches it by copying. Every function carries ROCm's symbol version
(`ROCR_1`). `tests/hsa/hsa_dispatch.c` is built against this header, and
against ROCm's `hsa.h` where that is installed, and both builds run
(ctest `amd_hsa`). A grid need not be a whole number of work-groups, as HSA
allows: the last group in a dimension runs short, numbered across its own
shape, and the kernel's `hidden_remainder` arguments say by how much.
The HSA 1.1 queries a tool makes of an agent are answered too: `hsa_agent_iterate_caches` and `hsa_cache_get_info` (the levels, L1 to L3 where the chip has one, with the sizes the deprecated `HSA_AGENT_INFO_CACHE_SIZE` gives; the CPU agent reports none), `hsa_isa_iterate_wavefronts` and `hsa_wavefront_get_info`, `hsa_isa_compatible` (the same full target name only), and signal groups (`hsa_signal_group_create`, `_destroy` and both `_wait_any` forms). `tests/hsa/hsa_caches.c` checks them, built against this header and against ROCm's (ctest `amd_hsa`). Not answered: `hsa_amd_signal_wait_any` (the header does not say what a timeout returns), the deprecated code-object and finalizer calls, `hsa_amd_queue_intercept_*`, `hsa_amd_register_deallocation_callback`, `hsa_amd_memory_migrate` and `hsa_amd_spm_*`.

ROCm's own tools and HIP runtime run on it unmodified:

- **`rocminfo`** describes every device.
- **ROCm's `libamdhip64` (CLR)**, from releases 7.0, 7.1 and 7.2, runs hipcc-built programs with only `libhsa-runtime64` replaced (ctest `amd_hip_on_hsa`). CLR then does all of HIP itself:
  - its copies and fills are its own kernels in AQL packets;
  - device `printf` comes back through its hostcall listener;
  - a cooperative launch goes to a cooperative queue;
  - peers are granted through `hsa_amd_agents_allow_access`.

That works because the runtime keeps to what ROCm's does where CLR looks:

- **Signals:** a signal handle is the address of an `amd_signal_t` every device maps, so a kernel can ring it, and a host wait sees what a kernel wrote.
- **Kernel arguments:** a packet's kernarg segment reaches the kernel as the program wrote it, hidden arguments included.
- **Work-group limit:** a packet is held only to the hardware's work-group limit, not the kernel's metadata.
- **Host access:** the host is never given a device's memory directly, so CLR copies instead of writing through it.
- **Supported extras:** AMD's loader extension, barrier-value packets, asynchronous signal handlers, dispatch timestamps and `hsa_amd_pointer_info` all work.
- **Images, virtual memory and IPC:** HSA's images extension (see Textures), its virtual memory (`hsa_amd_vmem_*`), its IPC handles (`hsa_amd_ipc_memory_*`) and dma-buf export are modelled. The last three are HIP's own (`src/hsa_vmem.inc`), reached by hidden names so that ROCm's libamdhip64 over this runtime does not call itself. ROCm's HIP runs `tests/hipcc/memory.cpp`'s virtual memory, pool IPC and dma-buf checks over it.
- **Shared virtual memory:** the runtime reports SVM support, as ROCm's does on a machine with HMM. ROCm's HIP then keeps managed memory the way it does there: host pages made accessible to each GPU, advised and prefetched through `hsa_amd_svm_attributes_set`, `_get` and `hsa_amd_svm_prefetch_async` (`src/hsa_svm.inc`). A range given to a GPU is mapped for the devices where it is. Advice, access and the last prefetch's target are kept page by page and read back, uniform or not. A prefetch moves nothing, because there is no second copy of the pages. It waits for its dependencies and completes its signal.
- **Also answered:** `hsa_amd_signal_wait_any` (the header does not say what a wait that times out returns; this answers `UINT32_MAX`, no index of any list, with the timeout in the 1 GHz system ticks `hsa_signal_wait` takes); `hsa_amd_ipc_signal_create` and `_attach` for a signal made with `HSA_AMD_SIGNAL_IPC`, within the process (a handle from another process is refused: a signal is an object of its process's heap); `hsa_amd_memory_pool_can_migrate` and `hsa_amd_memory_migrate` between the CPU's fine- and coarse-grained pools and from a pool to itself (a buffer keeps its address, and the grain `hsa_amd_pointer_info` reports changes); `hsa_amd_register_deallocation_callback` and `_deregister_` (fired once, from `hsa_amd_memory_pool_free`, the one release the header names); `hsa_amd_async_function`; and the deprecated `hsa_code_object_deserialize`, `_destroy`, `hsa_executable_create`, `_load_code_object`, `_get_symbol` (no module names) and `_validate`, through the reader path (`test_amd_runtime_gaps`).
- **Refused by name (`HSA_STATUS_ERROR_NOT_SUPPORTED`):** `hsa_amd_spm_*` (no counter stream), the finalizer (`hsa_ext_program_*`; AMD's runtime has had none since ROCm 2), the rest of the deprecated code object interface (`hsa_code_object_serialize`, `_get_info`, `_get_symbol*`, `_iterate_symbols`, `hsa_code_symbol_get_info`, `hsa_executable_load_program_code_object`, `_iterate_program_symbols`), `hsa_amd_queue_intercept_*` and `hsa_ven_amd_aqlprofile_*` (packets are not diverted to a tool and there are no PM4 counter packets; the public header available here did not declare the interception calls, so their signatures were not checked), the variable definitions (`hsa_executable_agent_global_variable_define`, `_global_variable_define`, `_readonly_variable_define`: the loader leaves a code object's undefined symbols unrelocated, so a defined variable would bind to nothing), and graphics interop (`hsa_amd_image_create`, `hsa_amd_interop_map_buffer`, `_unmap_buffer`). Still absent, not refused: `hsa_soft_queue_create`, `hsa_queue_inactivate`, `hsa_memory_assign_agent`, `hsa_extension_get_name`, the exception-policy queries, `hsa_amd_image_get_info_max_dim`.

`VGPU_TRACE_HSA=1` logs what memory the program allocates, locks and registers.

## OpenMP offload and OpenCL

Two more ways to program an AMD GPU run on the simulated ones with ROCm's own runtimes, and neither goes through HIP.

- **OpenMP target offloading.** LLVM's `libomptarget` has an AMD plugin that drives the HSA runtime directly: queues, signals, memory pools, executables, kernel dispatch packets. `tests/openmp/omp.cpp` is built by ROCm's `amdclang++ -fopenmp --offload-arch=gfx...` for each GPU's ISA and run on two simulated devices of each model (ctest `amd_openmp`, `tests/e2e/run_openmp.sh`): `target teams distribute parallel for` with map clauses, reductions (built-in and `declare reduction`), `declare mapper`, `target data` with `enter`/`exit`/`update`, `map(present)` and `map(always)`, `declare target` functions and variables, `omp_target_alloc`/`memcpy`/`associate_ptr`, atomics (update and capture), `collapse`, `teams loop`, `num_teams`/`thread_limit`/`num_threads`, `nowait` with `depend`, `device(n)` for every device, the device math library, and `printf` from a target region. All of them pass on the six GPUs.
- **OpenCL.** ROCm's OpenCL runtime (`libamdocl64`, on the same CLR as HIP) builds each kernel with the compiler library and runs it through the HSA runtime. `tests/opencl/cl.c` is an ordinary OpenCL program (built by the system's compiler against the ICD loader) that runs on two devices of each model (ctest `amd_opencl`, `tests/e2e/run_opencl.sh`): vector add, a reduction through local memory and barriers, atomics, doubles, a 2D range, barriers over global memory, buffer fill, copy and map, a global offset with a range that is not a multiple of the work-group, coarse-grained shared virtual memory, and, on the Radeon GPUs and the MI250X, images and samplers. The MI300 family has no texture units, and says so (`CL_DEVICE_IMAGE_SUPPORT` is false).
- **Which GPUs, and where.** Both need only ROCm's LLVM or OpenCL runtime beside the simulator's HSA runtime; the CI job `models` unpacks them from AMD's apt repository (`tests/openmp/fetch-rocm-llvm.sh`, `tests/opencl/fetch-rocm-opencl.sh`) with nothing installed, and the tests skip where neither is found. `offload-arch` (`amdgpu-arch`), which they and CMake's HIP language use to find the GPUs, reads the kernel driver's own directory `/sys/devices/virtual/kfd/kfd`: an isolated session has it, with `/sys/class/kfd/kfd` a link to it as on a machine with the driver.

## System tools

Inside `vgpu shell --gpu amd/...` the tools an AMD machine has answer for
the simulated GPUs (amd/tests/e2e/run_amd_tools.sh, ctest `amd_tools`):

- **`lspci`**: the host's real lspci over the devices' config space. An isolated session gives it the session's `/sys/bus/pci`, so `-k` and `-vv` show the bound `amdgpu` driver and every capability. Its PCI ID database is the host's, plus the simulated cards it predates (the RX 9070 XT, the MI350X). A host without pciutils gets `vgpu smi --lspci`, which prints the same, form by form. An Instinct card is a processing accelerator; a Radeon card is a VGA controller with its chip's revision (c8 for the RX 7900 XTX).
- **`rocminfo`**: ROCm's own where the host has it, reading the simulated agents through the session's HSA runtime; otherwise `vgpu-rocminfo`, which asks the same runtime through the public HSA interface and prints what ROCm's prints, line for line. Each agent's chip ID, compute units (an RDNA workgroup processor is two), SIMDs, shader engines and arrays, compute dies, caches (L3 included) and memory interface are the chip's (`include/vgpu/amd_chip.hpp`).
- **`rocm-smi`** and **`amd-smi`**: the concise table and every `--show*` section, and `amd-smi`'s `list`, `static`, `metric`, `process`, `topology`, `monitor`, `partition`, `xgmi`, `bad-pages`, `firmware`, `ras` and `version`, with `--json`. Values come from the same machine state nvidia-smi reads. What a simulated card has no value for (VBIOS, serials, firmware, energy) is N/A.
- **`rocm_agent_enumerator`**: each card's own target.

## RCCL, and PyTorch across GPUs

RCCL (PyTorch's collectives on ROCm) runs unmodified across simulated GPUs in one process (`tests/pytorch/multi_gpu.py`, ctest `amd_pytorch_multi_gpu`) and across processes (`distributed.py`, ctest `amd_pytorch_distributed`). Three things make that work on any Linux machine:

- **ROCm SMI's library.** VirtualGPU has its own `librocm_smi64` (`src/rocm_smi.cpp`, `build/shim/librocm_smi64.so.7`), which answers ROCm SMI's public interface for the simulated GPUs. Its answers come from the same machine state `rocm-smi` and nvidia-smi read, not from HIP, so asking does not make a card look busy. It covers identity (ids, names, target version, PCI address, serial-like unique id), temperature, power and its caps, the energy counter (Instinct only), busy percentages, memory, clock levels, fans on Radeon cards, voltage, PCIe, ECC, compute and memory partitions, topology and processes. What the simulator does not model is refused with `RSMI_STATUS_NOT_SUPPORTED`, as a card refuses what it lacks: every setter, reset and event, the XCD counter, and the tables and identifiers with no datum behind them (SKU, VBIOS build, XGMI physical id, PM and register tables, cache info). `rsmi_dev_gpu_metrics_info_get` and `rsmi_dev_metrics_header_info_get` answer the same v1.5 table `amdsmi_get_gpu_metrics_info` does, in `rsmi_gpu_metrics_t` (4544 bytes, offsets checked against ROCm 7.2.0's `rocm_smi.h`); the DRM render minor, the XGMI hive id and the PCIe slot type are answered too, and every function of the header is exported. An isolated AMD session exports `ROCM_SMI_LIB_PATH` to it and shows `/sys/module/amdgpu` loaded, so ROCm's own `rocm-smi` runs there unmodified and says what VirtualGPU's says (ctest `amd_tools`, where the host has ROCm SMI and unprivileged user namespaces).
- **AMD SMI's library.** VirtualGPU also has its own `libamd_smi` (`src/amd_smi.cpp`, `build/shim/libamd_smi.so.26`), which AMD's Python package `amdsmi` loads, and through it vLLM. vLLM asks it before anything else whether this is a ROCm machine, and asks each GPU's target, name, memory and UUID, and whether the GPUs are fully connected by XGMI. Each GPU is a socket of its own with one processor, as AMD SMI groups discrete GPUs. The answers are the machine's, as `amd-smi` and ROCm SMI's library give them: the target, device ID, compute units, UUID, PCI address, KFD node and id, memory, activity, and links. The package binds every function of the library when imported, so all of them are exported (the list is AMD's header's, `tools/amdsmi-symbols.py`), and the ones not modelled answer `AMDSMI_STATUS_NOT_SUPPORTED`. The health queries answer from the profile, as ROCm SMI's library and `amd-smi` do: ECC totals and per-block counts and states, bad pages, XGMI info (hive id, links, error status), PCIe info (link, replay and NAK counters, bandwidth levels), temperature, power and cap, clocks, voltage, fans, compute and memory partitions, and the process list. Instinct profiles (HBM, `ecc: true`) have ECC, bad pages and XGMI; the Radeon profiles refuse all three with `NOT_SUPPORTED` (no RAS files, no hive), have a fan where Instinct has none, and have no partitions. Assumptions with no profile datum (x16 XGMI lanes, one link per peer, HBM stack sensors reading the one memory sensor, a remapped row listed as a bad page, no deferred errors) are marked in `amd_smi.cpp`. Clock levels (`amdsmi_get_clk_freq`, in Hz as the library's function documentation says and AMD's implementation passes ROCm SMI's levels through; the header's field comment saying MHz is stale) and `amdsmi_set_clk_freq` (pins a level for the calling process), the energy counter (Instinct only; the same accumulator `rsmi_dev_energy_count_get` reads, 15.3 uJ a tick) and `amdsmi_get_gpu_metrics_info` (the v1.5 table the driver files publish, in the ROCm 7.2.0 structure, whose size and offsets are checked against ROCm 7.2.0's header) are answered too. The getters ROCm SMI's library answers are answered here the same way (VRAM usage in MB, GPU busy percent, performance level "auto", no overdrive, the metrics table's header, reserved pages, and the compute processes by pid and by GPU). Still `NOT_SUPPORTED`: violation status, PCIe throughput, the bad-page threshold, board, VBIOS and firmware info (the profiles carry none), and every other setter and reset. AMD's package, unmodified (26.2.2, the version vLLM's ROCm wheels install), runs on it: ctests `test_amd_amd_smi`, `test_amd_amd_smi_radeon`, `amd_amdsmi_python`. The package looks in `$ROCM_HOME` and `$ROCM_PATH` before the library path, so a shell with those pointing at a ROCm install loads AMD's library instead.
- **The kernel driver's interface.** A session writes out amdkfd's topology (`src/kfd.cpp`): a CPU node, then a node per GPU with its `properties` (target, SIMDs, engines, PCI location, render minor, hive), memory bank and links. An isolated session shows it at `/sys/devices/virtual/kfd/kfd/topology`, where the driver keeps it (`/sys/class/kfd/kfd` is a link to it), along with a `renderD` entry per GPU in `/sys/class/drm`, and `/dev/kfd` and `/dev/dri/{cardN,renderD128+N}`. Tools find AMD GPUs there without a runtime: ROCm's `rocm_agent_enumerator` reads each node's `gfx_target_version`, and Ollama and RCCL match nodes to PCI devices by `location_id`. Every value is the one HIP, rocminfo, rocm-smi and amd-smi report. The node's `gpu_id` is rocm-smi's GUID and amd-smi's `kfd_id`, and the links are the ones ROCm SMI reports. The device files open as `/dev/null` does. A program that only looks for them finds an AMD machine; one that asks the driver something (an ioctl) is refused, as on a machine without the driver. The simulated runtime asks it nothing. `/sys/class`, `/sys/devices/virtual` and `/dev` are overlaid whole, and every other entry still leads to the host's (ctests `test_amd_kfd`, `amd_tools`).
- **Topology.** RCCL learns how its GPUs are linked from the AMD kernel driver's topology under `/sys/class/kfd`, or from ROCm SMI's library. A machine with no AMD GPU has no `/sys/class/kfd`, and RCCL fails to initialize ("internal error"). WSL is the exception: RCCL skips the question there. The simulator's library answers it: the device count, each device's PCI address as HIP reports it, and the link between each pair. That link is XGMI between Instinct GPUs and PCI Express between Radeon ones. Once loaded, the library sets `RCCL_USE_ROCM_SMI_LIB=1` where there is no `/sys/class/kfd` and inside a `vgpu shell` session, whose `/sys/class/kfd` is its own copy of the same answers. It leaves the variable alone elsewhere, or where the environment already set it. The PyTorch tests swap it in beside `libamdhip64` (ctests `test_amd_rocm_smi`, `test_amd_rocm_smi_radeon`).
- **Wide accesses.** A `global_`, `flat_` or `buffer_` load or store of two to four words moves each aligned pair of words as one 8-byte access, as the hardware does. RCCL's LL protocol puts a word of data and the flag that says it arrived in the same eight bytes, and a reader on another device trusts the data once it sees the flag. When a wide access moved one word at a time, a reader could see the new flag beside the old data, and `broadcast` lost values.
- **Opened IPC memory.** Memory from another process's `hipIpcGetMemHandle` is mapped where the exporter's device is numbered, and every other device in the process reaches it without `hipDeviceEnablePeerAccess`, as `hipIpcMemLazyEnablePeerAccess` asks. RCCL's kernels write straight into the other process's buffer.

## vLLM

vLLM for ROCm, unmodified, generates on every simulated AMD GPU it supports -- MI300X, MI325X, MI250X, MI350X, RX 7900 XTX and RX 9070 XT -- what Hugging Face's transformers generates on the CPU: `facebook/opt-125m`, greedy, "The capital of France is" continued as " the capital of the French Republic." in two to eight minutes (`tests/e2e/run_vllm_amd.sh`, ctest `amd_vllm`). It runs every night on a GitHub-hosted runner (`.github/workflows/vllm-nightly.yml`), with ROCm's libraries and vLLM's wheels installed by `tests/vllm/install.sh`.

- vLLM's ROCm wheels (`wheels.vllm.ai/rocm`) bring a PyTorch that is linked against a ROCm installed on the machine: its RUNPATH names `/opt/rocm-7.2.3/lib`. The machine needs that ROCm's libraries (hipBLAS, hipBLASLt, MIOpen, RCCL, rocprofiler-sdk and the rest) and OpenMPI. The simulator's HIP runtime, ROCm SMI and AMD SMI go in front of them on the library path.
- vLLM asks AMD SMI whether this is a ROCm machine, and NVML whether it is a CUDA one. NVML answers only for NVIDIA GPUs, and says a machine of AMD GPUs has no NVIDIA driver, as it does on one.
- Its paged attention for ROCm multiplies with `v_mfma_f32_4x4x4_16b_f16` and CBSZ 4: every block takes block 0's query (A's broadcast, modelled; B's lane patterns, BLGP, are refused by name).
- The default test run is eager (no graphs, no `torch.compile`), with one short sequence and a 3 GB device, so the KV cache fits in RAM. `VGPU_VLLM_MODE=graphs` (ctest `amd_vllm_graphs`, and a second step of the nightly) runs vLLM as it runs by default on a card: Dynamo and Inductor compile the model into Triton kernels, which Triton compiles for the simulated GPU at run time, and the decode step is captured into HIP graphs (piecewise and full) for batch sizes 1 and 2 and replayed per token, two prompts generated together. The test checks the log for the compile and both captures, since vLLM falls back to eager quietly. It took 8 minutes on a simulated MI300X.
- `VGPU_VLLM_TP=2` (ctest `amd_vllm_graphs_tp2`, the nightly's MI300X job) shards the model over two simulated GPUs, one worker process each: RCCL's all-reduces and vLLM's custom all-reduce, which shares the buffers its graphs capture through HIP's IPC handles, run inside the captured graphs, and the text is the CPU's (12 minutes). That needed two fixes in HIP's IPC: asking twice for the handle of one allocation gave a second handle naming a file that was never made (the first had made it), and `hipMemset` on memory opened from another process wrote through a device address as if it were a host pointer and faulted.
- What the graph and compiler paths need is checked below vLLM, in PyTorch's own terms (`tests/pytorch/graphs.py`: stream capture of kernels, copies, events, forked streams, allocations, random numbers, attention, a KV-cache decode step replayed per token, `make_graphed_callables`, kept graphs; `compile.py`: `torch.compile` including `mode="reduce-overhead"`, scans, sorts, dynamic shapes, a training step; `triton_kernels.py`: hand-written Triton on MFMA or WMMA in every input type, attention, atomics, reductions; `distributed_graphs.py`: RCCL's all-reduce and all-gather recorded into graphs across two processes, a tensor-parallel block), on every simulated generation.
- gfx950's Triton kernels needed three things beyond the other generations': `v_minimum3_f32`, `v_maximum3_f32` and the packed half forms (IEEE minimum and maximum of three, a relu in vLLM's compiled model), the scaled conversions of two 4- or 8-bit floats to bfloat16 (`v_cvt_scalef32_pk_bf16_{fp4,fp8,bf8}`, which MX-scaled `tl.dot_scaled` uses with one scale block of K; the other `v_cvt_scalef32_*` forms are still undecoded), and MUBUF loads into LDS (`buffer_load_dword`, `_dwordx3`, `_dwordx4` with the LDS bit, which Triton uses for asynchronous copies, `TRITON_HIP_USE_ASYNC_COPY=1`). The LDS loads are modelled for those three on gfx942 (dword) and gfx950, and the sub-dword and `format_x` forms too (a byte goes to LDS as a zero- or sign-extended dword, as the guides' instruction text and figure give it); an M0 of 64 KiB or more is refused on gfx950, because CDNA4's text gives the LDS offset as `M0[17:0]` and its figure as `M0[15:0]` (gfx90a and gfx942 take `M0[15:0]`, where the guides agree).
- vLLM's process ended by aborting, after its output, on a damaged heap: exit() destroys a thread's `thread_local` objects, and then the libraries' static destructors, one of which (hipBLASLt's) still calls HIP. The profiler's per-thread call stack and its table of HIP call names were used after they were destroyed. Both now outlive static destruction (ctest `test_amd_hip_at_exit`: HIP from a static destructor, clean under AddressSanitizer).

## Debugging

`vgpu debug` runs an AMD program with a kernel debugger:

```
vgpu debug --gpu amd/mi300x -- ./program            # commands at a (vgpu) prompt
vgpu debug -x session.txt --gpu amd/mi300x -- ./program   # or from a file
```

- **Stopping:**
  - `break KERNEL[+OFFSET]` stops every wave of a kernel that reaches OFFSET, in bytes from its first instruction. The kernel is matched by any part of its name. `tbreak` stops once.
  - `continue` goes on; `step [N]` runs N instructions of the stopped wave.
  - `delete [N]` and `info breakpoints` manage breakpoints.
- **Looking:**
  - `where` gives the kernel, offset, work-group, wave and EXEC; `disas [N]` lists instructions from there.
  - `info registers` shows PC, EXEC, VCC, SCC, M0, MODE and the SGPRs.
  - `print[/x|/d|/f] REG` prints a register for every lane (eight to a row, a switched-off lane marked) or for the lane `lane N` picks.
  - `x/N ADDR` and `x/N lds:ADDR` show device memory and the work-group's LDS.
- **Changing:** `set REG[LANE] = VALUE` changes a register; `quit` fails the launch.

While the debugger is on, work-groups run on one host thread. A stopped wave
stops its dispatch, and everything else runs in a repeatable order.
`VGPU_DEBUG=/dev/tty` or `VGPU_DEBUG=FILE` turns it on without the command.
`VGPU_TRACE_WAVE=1` instead prints every instruction a work-group's first wave
runs, with its results. `amd/tests/e2e/run_debugger.sh` (ctest `amd_debugger`)
runs a session.

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
texture addresser takes (`TA_FLAT_*`). There is also the instruction mix, each
instruction classed once as it is decoded:
- `SQ_INSTS_VALU_{ADD,MUL,FMA,TRANS}_{F16,F32,F64}`, `_CVT`, `_INT32` and `_INT64`;
- matrix work in 512-operation units, `SQ_INSTS_VALU_MFMA_MOPS_{I8,F16,BF16,F32,F64,F8}`, with RDNA's WMMA included;
- vector memory reads and writes, `SQ_INSTS_VMEM_RD` and `_WR`. Each is the device's total, as one
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
runtime, so no HSA trace; no thread trace, or the stochastic method of PC
sampling, which need the hardware's trace and sampling units; and no records of
the kernels ROCm's runtime runs on its own behalf (a device-to-device copy or a
memset is not a kernel here).

PC sampling (`rocprofiler_configure_pc_sampling_service`) is offered on the
Instinct GPUs that have it (gfx90a, gfx942, gfx950) by the host-trap method
only, and **by instruction count, not time**: the simulator has no timing
model, so there is no clock to sample on. The one configuration an agent lists
is the host-trap method with the unit `ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS`
and any interval from 1. Each wave is sampled when it is about to issue its
1st, (1 + interval)th, (1 + 2 * interval)th ... instruction, counting every
instruction it issues; the program counters are therefore exact and repeatable
for a kernel, and say nothing of where a card would spend its time. Samples are
delivered to the buffer as `rocprofiler_pc_sampling_record_host_trap_v0_t`
records when the dispatch completes: the code object and the offset of the
instruction, EXEC, the work-group and the wave in it, where the work-group ran
(compute die, shader engine and array, compute unit; the SIMD, pipe, VM id,
queue and microengine of the hardware ID are not modelled and read 0), the
dispatch and correlation ids, and the time the sample was taken on the clock
the traces use. A tool that asks for the method in time or cycles
(`rocprofv3 --pc-sampling-unit time`) is refused with
`ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT` rather than given invented times;
the stochastic method answers `ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE`, as for a
GPU without it. `test_amd_rocprofiler_pcsampling_1` and `_5` check the samples
against llvm-objdump's addresses for a kernel, every instruction and every
fifth.

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
the fault to fire, and each one caught it, reporting the fault the way it
would on a card: its kernel's own `printf` (`[SDC FAULT] ...`), then
"Verification: FAIL".

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

## ROCm Validation Suite

AMD's open-source [ROCm Validation Suite](https://github.com/ROCm/ROCmValidationSuite)
(RVS, MIT) is the public counterpart of NVIDIA's DCGM/NVVS. Built from its
`rocm-7.2.3` source (and run from ROCm's package too), it runs unmodified inside
`vgpu shell --gpu amd/mi300x`: `rvs -c <conf>` finds the simulated GPUs
through the KFD topology (`src/kfd.cpp`), ties each to its AMD SMI handle and
its PCI device, and runs its modules. What each does on the simulator:

| Module | Result | Why |
| --- | --- | --- |
| `gpup` | pass | the KFD node properties (`simd_count 1216`, `wave_front_size 64`, `gfx_target_version 90402`, `num_xcc 8`, ids). `caches_count`, `cpu_cores_count`, `local_mem_size`, `fw_version` read 0 (no `caches/` directory) |
| `peqt` | pass, 17 of 17 | link speed and width, slot, vendor and device ids, driver `amdgpu`, atomics routing and completer bits, from the PCI configuration space and capability chains (`docs/registers.md`) |
| `pesm` | pass | PCIe link-state monitoring: no change in speed or power |
| `smqt` | pass with a conf for the MI300X's BARs (256 GB, 2 MB, 512 KB); the shipped `smqt_single.conf` expects an older part's 256 KB register BAR and fails | BARs and the expansion ROM through libpci |
| `gm` | pass | `amdsmi_get_temp_metric`, `_power_info`, `_clk_freq`, ... (`src/amd_smi.cpp`) |
| `pebb`, `pbqt` | pass | the link info of `hsa_amd_agent_memory_pool_get_info` (`distance`, `PCIe:20`, `xGMI:15`) and async copies; the GB/s are the interpreter's, not a link's |
| `babel` | pass | five HIP bandwidth kernels (copy, mul, add, triad, dot) |
| `gst` | pass with a target any run meets; correctness checks work | with `self_check`, `accuracy_check` and `error_inject`, every rocBLAS SGEMM is compared with the host's result and with the last run, and a wrong element made on purpose is caught by both. The shipped targets (`target_stress` in GFLOPS) are far above what an interpreter does: the module reports GFLOPS 0 and fails |
| `mem` | pass, all 11 tests, slowly | with `mem_blocks: 1` and `num_passes: 1`: six minutes a GPU (test 7, the 32-bit moving inversions, takes four of them), so it is not in the test |
| `perf`, `iet`, `tst` | fail, by design | `perf`'s pass mark is a GFLOPS target; `iet`'s is the power a GEMM load draws (60 W, the idle floor, is all it ever sees) and `tst`'s the temperature it reaches (a junction sensor at 32 C, and the edge sensor an Instinct lacks reads 0). The telemetry follows how long the simulated work would have run on the card (the instructions retired over the clock), not the host time the interpreter takes, so a load of seconds of host time is microseconds of device time. There is no timing model to give them anything else |
| `rcqt` | runs | it asks `dpkg` about ROCm's meta-packages: the host's answer, not the simulator's |

The `rvs` summary header prints "ROCm version N/A" and "amdgpu version N/A": it
reads `/opt/rocm/.info/version-rocm` and the output of `dkms status`, both of
the host. Without AMD SMI's library in front of ROCm's, RVS's `gpulist` finds no
processors and `peqt` and `pesm` crash in `gpu2gpuindex` (an RVS bug: it indexes
an empty vector); in a session the simulator's library is always first. The
same set (with `gst` at a target of 0.001 GFLOPS, which the Radeon GPUs need
where 0.05 is out of reach) runs and passes on MI250X, MI325X, MI350X, RX 7900 XTX,
RX 9070 XT and RX 6900 XT. `tests/e2e/run_rvs.sh` (ctest `amd_rvs`, which skips without an
`rvs` binary or an isolated session) runs `gpup`, `peqt`, `pesm`, `smqt`, `babel`
and `gst` in half a minute.

What RVS asks, module by module:

| Module | What it reads | State |
| --- | --- | --- |
| `gpup` | `/sys/class/kfd/kfd/topology/nodes/*/{gpu_id,properties,io_links}` | answered by the KFD topology (`src/kfd.cpp`); `caches_count` is 0 and there is no `caches/` directory |
| every module | `amdsmi_get_gpu_kfd_info`, `amdsmi_get_gpu_bdf_id`, `hipDeviceGetPCIBusId` to tie a HIP device to its KFD node and SMI handle | consistent: same bus, same `kfd_id` |
| `gm`, `tst`, `pulse`, `iet`, `gst` | `amdsmi_get_temp_metric`, `amdsmi_get_power_info`, `amdsmi_get_power_cap_info`, `amdsmi_get_clk_freq`, `amdsmi_set_clk_freq`, `amdsmi_get_energy_count` | added (`src/amd_smi.cpp`); they were NOT_SUPPORTED stubs |
| `pebb`, `pbqt` | `hsa_amd_agent_memory_pool_get_info` `NUM_LINK_HOPS` and `LINK_INFO`, async copy and its profiling | `LINK_INFO` added: system memory is one PCI Express hop, an Instinct peer one XGMI hop, a Radeon peer two PCI Express hops, as KFD lists them |
| `mem`, `babel`, `gst` | HIP allocation, copies, kernels; rocBLAS GEMM | RVS's own kernels, compiled for the card, run |
| `peqt`, `pcie` | PCI configuration space through libpci | the configuration space and its capability chains are modelled (`docs/registers.md`) and RVS reads them |

The AMD SMI calls above are the ones described under "AMD SMI's library" in
"RCCL, and PyTorch across GPUs"; the sensors follow each profile's class. Temperatures are
degrees (AMD SMI converts the driver's millidegrees); an Instinct GPU has a
junction and a memory sensor (VRAM and HBM) but no edge sensor, a Radeon has
edge, junction and VRAM, and the slowdown limit is refused where the profile
has none (MI325X). Power is watts, the capability in microwatts; MI300 and newer
fill `current_socket_power`, the earlier Instinct and Radeon
`average_socket_power`, the other reading is UINT32_MAX. Clocks have two
levels, as ROCm SMI's library gives them, in Hz, and `amdsmi_set_clk_freq` pins
one for the calling process (a mask of a level the clock lacks is out of
bounds). Only Instinct GPUs have the energy accumulator (15.3 uJ a tick,
counting up from an offset; `rsmi_dev_energy_count_get` and
`amdsmi_get_energy_count` read the one counter) and a Radeon has ECC disabled
or `NOT_SUPPORTED`, from its profile. `tests/unit/test_amd_smi.cpp` checks all of
it on every profile (`test_amd_amd_smi_rvs_*`), `tests/unit/test_rocm_smi.cpp`
that the two libraries agree on the energy count, and
`tests/hsa/hsa_dispatch.c` the link info.
