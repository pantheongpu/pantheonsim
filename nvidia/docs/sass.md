# SASS: NVIDIA machine code

VirtualGPU runs a kernel's SASS -- the machine code the GPU itself executes --
whenever the binary carries SASS the simulated GPU can run, as the real driver
does. PTX is the fallback: it is JIT-compiled (here, interpreted) when there is
no suitable SASS, and it stays the path for programs that ship only PTX.

This replaces an earlier policy of executing PTX only. Running SASS is what
puts the simulator at the level the hardware works at: 255 physical registers
per thread and their allocation, predicate and uniform registers, convergence
barriers, the constant banks the driver fills, and the exact instruction
sequence ptxas chose -- including whatever ptxas got wrong. It is also what
lets SASS-only programs run at all.

## Sources

SASS is not documented by NVIDIA. Everything here comes from public material
and measurement, never from disassembling NVIDIA's driver or runtime:

- **Encodings**: Mesa's NAK compiler (`src/nouveau/compiler/nak`, MIT), which
  emits SASS for Volta and later and is itself checked against nvdisasm; and,
  for instructions NAK does not emit, NVIDIA's public disassembler.
- **Decode oracle**: `nvdisasm` / `cuobjdump -sass`, shipped with every CUDA
  toolkit. The decoder is checked instruction by instruction against it over a
  corpus (below), the way the AMD decoder is checked against llvm-objdump.
- **Execution oracles**: the simulator's own PTX interpreter, already checked
  against hardware, running the PTX of the same kernel -- a binary carrying
  both must produce the same results either way; and a physical RTX 3060
  (sm_86) for instruction semantics, bit for bit.

## Architecture

```
fatbin -> cubin (ELF) loader -> decoder (per-arch tables) -> SASS warp executor
            .nv.info: params, regs,          |                    |
            shared, EXIT offsets        disassembler         memory, textures,
            relocations, constants     (checked vs nvdisasm)  runtime: shared with
                                                               the PTX interpreter
```

- **Selection** (`driver_api.cpp`, `runtime_api.cpp`): a fatbin's ELF image for
  the device's architecture wins over its PTX, by the driver's rules: `sm_XY`
  SASS runs on the same major at minor Y or newer; `sm_XYa` only on XY.
- **Loader**: kernel sections `.text.<name>`, per-kernel attributes from
  `.nv.info.<name>` (parameter layout, register count, shared memory, barrier
  count), constant sections, globals and relocations.
- **Kernel handle**: a SASS kernel is an ordinary `ptx::EntryFn` (name,
  parameter sizes, attributes) with the decoded code attached, so everything
  above the launch -- argument marshalling, occupancy, attributes -- is shared.
- **Executor**: a warp is 32 lanes; state is the architectural state: R0-R254
  (RZ), P0-P6 (PT), UR0-UR62 (URZ), UP0-UP6, the convergence barriers of
  BSSY/BSYNC, special registers, and the constant banks (c[0x0] holds the
  launch: grid and block sizes, the stack, and the parameters). Scoreboards and
  stall counts in the control bits are timing, which VirtualGPU does not model;
  every instruction completes before the next.

## Coverage plan

One family of 128-bit encodings covers Volta onward; each generation adds
instructions over it. Every generation is done before this ships:

| Target | Profiles | Adds |
| --- | --- | --- |
| sm_75 | T4 | the base: integer/float ALU, memory, control, HMMA/IMMA |
| sm_80, sm_86, sm_89 | A100, A10, RTX 3060/3080 Ti, L4, L40S | LDGSTS, LDSM, bf16/tf32 MMA, REDUX |
| sm_90 | H100, H200, GH200 | clusters, TMA (UTMA*), wgmma (HGMMA etc.), setmaxnreg |
| sm_100, sm_103 | B200, B300 | tcgen05 (UTC*), tensor memory |
| sm_120 | RTX 5090 | sm_120's MMA kinds |

## Tests

- `sass_decode_<arch>`: every instruction in `nvidia/tests/data/sass/<arch>.txt`
  (encoding and nvdisasm's text, gathered by `nvidia/tools/sass-corpus.py` from
  this repository's CUDA sources, CUTLASS and PyTorch's CUDA wheel) decodes and
  prints back exactly as nvdisasm printed it.
- Every existing CUDA end-to-end test runs on SASS, and again forced to PTX; the
  two must agree, and on sm_86 both must match the RTX 3060.
- Instruction semantics: `nvidia/tests/sass/` kernels checked against hashes
  taken on the RTX 3060.
