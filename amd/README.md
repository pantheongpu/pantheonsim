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
taken, which this does not claim to know. Streams run at once, as a card's
do: each stream's work runs in order on a host thread of its own, so a kernel
on one stream may wait on a flag another stream's kernel sets, an event and a
stream say `hipErrorNotReady` while their work runs, `hipStreamWaitEvent`
orders one stream behind another, and the null stream is the legacy default
stream, ordered against the blocking streams. A kernel's fault is told at the
next synchronization, as on a card. `VGPU_SYNC_LAUNCHES=1` makes every call
wait for its own work instead, which rules concurrency out when a program
misbehaves. A program built by `hipcc` runs unmodified too:
its device code is registered from inside the executable before `main`, its
chevron launches go through `hipLaunchKernel`, and it reads the device through
the real headers' `hipDeviceProp_t`, which is laid out here field for field as
ROCm lays it out. Graph capture and replay, peer access between devices, and
pinned host memory used to stage copies are there for the programs that use
them. The library answers to both of ROCm's names for it (`libamdhip64.so.6`
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
library loads; Tensile's GEMMs come from code objects hundreds of kernels
were linked into, each keeping its own metadata. The stream-ordered
allocations, pitched copies, pointer attributes and work-item-sized launches
(`hipExtModuleLaunchKernel`) it calls are there. `tests/e2e/run_rocblas.sh`
builds a program against rocBLAS and checks its level-1 routines and its
float and double GEMMs -- ragged sizes, every transpose -- against the same
arithmetic on the host; both need ROCm with rocBLAS installed and skip
elsewhere. AMD's own rocBLAS test suite, built from ROCm 7.1's source, runs
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
may round it differently. The reduced-precision xf32 forms, the broadcast
modifiers, and a wave with lanes switched off are refused.

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
packet, carries on. `printf` is the service implemented: a message per lane,
carried in as many packets as it needs, formatted as C's printf would and
written to the program's stdout, with printf's return value sent back. A
hostcall for another service (device `malloc`, the address sanitizer) is
refused by name. Vector loads and stores may be unaligned, as ROCm runs the
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

## RDNA2 (Radeon RX 6900 XT)

`VGPU_GPU=amd/rx6900xt` is gfx1030 (RDNA2). PyTorch's 20 checks pass on it
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
- **Checks:** the executor's unit kernels pass on gfx1030 (ctests `*_gfx1030`), and `tests/hipcc/rdna2.cpp` checks SDWA, M0-relative registers, the permlanes and the DPP modes, in wave32 and wave64.

`VGPU_TRACE_WAVE=1` prints each instruction a work-group's first wave runs,
with what its destination holds after it for lane 0 (or the lane
`VGPU_TRACE_LANE` names). That is how the bugs above were found.

## Textures

The GPUs modelled here, the MI300 family (gfx942 and gfx950), have no texture
units. hipcc refuses the texture API in their device code
(`__HIP_NO_IMAGE_SUPPORT`), and ROCm's HIP on them reports image support 0
and answers every call that would make an array, a texture or a surface with
`hipErrorNotSupported`. The shim gives the same answers, so a program or
library that calls them is told what it would be told on the card instead of
failing to load. `tests/hipcc/textures.cpp` prints each answer; its output must
match `tests/hipcc/rocm/textures.expected` on the shim, for each release's
build of it, and on ROCm's own HIP over the HSA runtime.

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
- **Not modelled:** images, virtual memory, HSA's IPC and SVM are refused by name. HIP's IPC is modelled (see RCCL below).

`VGPU_TRACE_HSA=1` logs what memory the program allocates, locks and registers.

## System tools

Inside `vgpu shell --gpu amd/...` the tools an AMD machine has answer for
the simulated GPUs (amd/tests/e2e/run_amd_tools.sh, ctest `amd_tools`):

- **`lspci`**: the host's real lspci over the devices' config space. An isolated session gives it the session's `/sys/bus/pci`, so `-k` and `-vv` show the bound `amdgpu` driver and every capability. Its PCI ID database is the host's, plus the simulated cards it predates (the RX 9070 XT, the MI350X). A host without pciutils gets `vgpu smi --lspci`, which prints the same, form by form. An Instinct card is a processing accelerator; a Radeon card is a VGA controller with its chip's revision (c8 for the RX 7900 XTX).
- **`rocminfo`**: ROCm's own where the host has it, reading the simulated agents through the session's HSA runtime; otherwise `vgpu-rocminfo`, which asks the same runtime through the public HSA interface and prints what ROCm's prints, line for line. Each agent's chip ID, compute units (an RDNA workgroup processor is two), SIMDs, shader engines and arrays, compute dies, caches (L3 included) and memory interface are the chip's (`include/vgpu/amd_chip.hpp`).
- **`rocm-smi`** and **`amd-smi`**: the concise table and every `--show*` section, and `amd-smi`'s `list`, `static`, `metric`, `process`, `topology`, `monitor`, `partition`, `xgmi`, `bad-pages`, `firmware`, `ras` and `version`, with `--json`. Values come from the same machine state nvidia-smi reads. What a simulated card has no value for (VBIOS, serials, firmware, energy) is N/A.
- **`rocm_agent_enumerator`**: each card's own target.

## RCCL, and PyTorch across GPUs

RCCL (PyTorch's collectives on ROCm) runs unmodified across simulated GPUs in one process (`tests/pytorch/multi_gpu.py`, ctest `amd_pytorch_multi_gpu`) and across processes (`distributed.py`, ctest `amd_pytorch_distributed`). Three things make that work on any Linux machine:

- **ROCm SMI's library.** VirtualGPU has its own `librocm_smi64` (`src/rocm_smi.cpp`, `build/shim/librocm_smi64.so.7`), which answers ROCm SMI's public interface for the simulated GPUs. Its answers come from the same machine state `rocm-smi` and nvidia-smi read, not from HIP, so asking does not make a card look busy. It covers identity (ids, names, target version, PCI address, serial-like unique id), temperature, power and its caps, busy percentages, memory, clock levels, fans on Radeon cards, voltage, PCIe, ECC, compute and memory partitions, topology and processes. What the simulator does not model is refused with `RSMI_STATUS_NOT_SUPPORTED`, as a card refuses what it lacks: `gpu_metrics`, every setter, reset and events. An isolated AMD session exports `ROCM_SMI_LIB_PATH` to it and shows `/sys/module/amdgpu` loaded, so ROCm's own `rocm-smi` runs there unmodified and says what VirtualGPU's says (ctest `amd_tools`, where the host has ROCm SMI and unprivileged user namespaces).
- **AMD SMI's library.** VirtualGPU also has its own `libamd_smi` (`src/amd_smi.cpp`, `build/shim/libamd_smi.so.26`), which AMD's Python package `amdsmi` loads, and through it vLLM. vLLM asks it before anything else whether this is a ROCm machine, and asks each GPU's target, name, memory and UUID, and whether the GPUs are fully connected by XGMI. Each GPU is a socket of its own with one processor, as AMD SMI groups discrete GPUs. The answers are the machine's, as `amd-smi` and ROCm SMI's library give them: the target, device ID, compute units, UUID, PCI address, KFD node and id, memory, activity, and links. The package binds every function of the library when imported, so all of them are exported (the list is AMD's header's, `tools/amdsmi-symbols.py`), and the ones not modelled answer `AMDSMI_STATUS_NOT_SUPPORTED`. AMD's package, unmodified (26.2.2, the version vLLM's ROCm wheels install), runs on it: ctests `test_amd_amd_smi`, `test_amd_amd_smi_radeon`, `amd_amdsmi_python`. The package looks in `$ROCM_HOME` and `$ROCM_PATH` before the library path, so a shell with those pointing at a ROCm install loads AMD's library instead.
- **The kernel driver's interface.** A session writes out amdkfd's topology (`src/kfd.cpp`): a CPU node, then a node per GPU with its `properties` (target, SIMDs, engines, PCI location, render minor, hive), memory bank and links. An isolated session shows it at `/sys/class/kfd/kfd/topology`, along with a `renderD` entry per GPU in `/sys/class/drm`, and `/dev/kfd` and `/dev/dri/{cardN,renderD128+N}`. Tools find AMD GPUs there without a runtime: ROCm's `rocm_agent_enumerator` reads each node's `gfx_target_version`, and Ollama and RCCL match nodes to PCI devices by `location_id`. Every value is the one HIP, rocminfo, rocm-smi and amd-smi report. The node's `gpu_id` is rocm-smi's GUID and amd-smi's `kfd_id`, and the links are the ones ROCm SMI reports. The device files open as `/dev/null` does. A program that only looks for them finds an AMD machine; one that asks the driver something (an ioctl) is refused, as on a machine without the driver. The simulated runtime asks it nothing. `/sys/class` and `/dev` are overlaid whole, and every other entry still leads to the host's (ctests `test_amd_kfd`, `amd_tools`).
- **Topology.** RCCL learns how its GPUs are linked from the AMD kernel driver's topology under `/sys/class/kfd`, or from ROCm SMI's library. A machine with no AMD GPU has no `/sys/class/kfd`, and RCCL fails to initialize ("internal error"). WSL is the exception: RCCL skips the question there. The simulator's library answers it: the device count, each device's PCI address as HIP reports it, and the link between each pair. That link is XGMI between Instinct GPUs and PCI Express between Radeon ones. Once loaded, the library sets `RCCL_USE_ROCM_SMI_LIB=1` where there is no `/sys/class/kfd` and inside a `vgpu shell` session, whose `/sys/class/kfd` is its own copy of the same answers. It leaves the variable alone elsewhere, or where the environment already set it. The PyTorch tests swap it in beside `libamdhip64` (ctests `test_amd_rocm_smi`, `test_amd_rocm_smi_radeon`).
- **Wide accesses.** A `global_`, `flat_` or `buffer_` load or store of two to four words moves each aligned pair of words as one 8-byte access, as the hardware does. RCCL's LL protocol puts a word of data and the flag that says it arrived in the same eight bytes, and a reader on another device trusts the data once it sees the flag. When a wide access moved one word at a time, a reader could see the new flag beside the old data, and `broadcast` lost values.
- **Opened IPC memory.** Memory from another process's `hipIpcGetMemHandle` is mapped where the exporter's device is numbered, and every other device in the process reaches it without `hipDeviceEnablePeerAccess`, as `hipIpcMemLazyEnablePeerAccess` asks. RCCL's kernels write straight into the other process's buffer.

## vLLM

vLLM for ROCm, unmodified, generates on a simulated MI300X what Hugging Face's transformers generates on the CPU: `facebook/opt-125m`, greedy, "The capital of France is" continued as " the capital of the French Republic." in about two minutes (`tests/e2e/run_vllm_amd.sh`, ctest `amd_vllm`). It runs every night on a GitHub-hosted runner (`.github/workflows/vllm-nightly.yml`), with ROCm's libraries and vLLM's wheels installed by `tests/vllm/install.sh`.

- vLLM's ROCm wheels (`wheels.vllm.ai/rocm`) bring a PyTorch that is linked against a ROCm installed on the machine: its RUNPATH names `/opt/rocm-7.2.3/lib`. The machine needs that ROCm's libraries (hipBLAS, hipBLASLt, MIOpen, RCCL, rocprofiler-sdk and the rest) and OpenMPI. The simulator's HIP runtime, ROCm SMI and AMD SMI go in front of them on the library path.
- vLLM asks AMD SMI whether this is a ROCm machine, and NVML whether it is a CUDA one. NVML answers only for NVIDIA GPUs, and says a machine of AMD GPUs has no NVIDIA driver, as it does on one.
- Its paged attention for ROCm multiplies with `v_mfma_f32_4x4x4_16b_f16` and CBSZ 4: every block takes block 0's query (A's broadcast, modelled; B's lane patterns, BLGP, are refused by name).
- The run is eager (no graphs, no `torch.compile`), with one short sequence and a 3 GB device, so the KV cache fits in RAM.
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
