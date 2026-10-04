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

## Selection and fallback

A binary's fatbin is searched for an ELF image the GPU can run (above). The
SASS executor then checks every instruction of it: an image with one the
executor does not run is passed over for the fatbin's PTX when there is PTX,
so a program keeps working while the SASS path grows; with no PTX the image
runs and the kernel reports the instruction it stopped at. A binary with SASS
only for another architecture and no PTX gets `cudaErrorNoKernelImageForDevice`,
as on the hardware.

| Variable | Effect |
| --- | --- |
| `VGPU_SASS=0` | PTX whenever the binary has it |
| `VGPU_SASS=1` | the SASS even where the executor lacks something (to find what) |
| `VGPU_SASS_LOG=1` | say which code each module runs, and why a fallback happened |
| `VGPU_SASS_REFUSE=<op>` | treat an op as unsupported (tests the fallback) |
| `VGPU_SASS_TRACE=<warp>` | every instruction that warp of block (0,0,0) runs, with its results |
| `VGPU_SASS_TRACE_KERNEL=<text>` | only in kernels whose name holds the text |
| `VGPU_KERNEL_DIGEST=<file>` | per launch, hashes (and NaN/Inf counts) of the memory its arguments reach: run once with `VGPU_SASS=0` and once without, and `diff` names the first kernel that differs |

## Coverage

Every generation from Turing to Blackwell runs, sm_75 through sm_120a:

| Target | Profiles | What its SASS adds, and runs here |
| --- | --- | --- |
| sm_75 | T4 | the base: integer/float ALU, memory, control, textures and surfaces, HMMA/IMMA |
| sm_80, sm_86, sm_89 | A100, A10, RTX 3060/3080 Ti, L4, L40S | LDGSTS, LDSM, REDUX, bf16/tf32/fp8/sparse MMA, sm_80's mbarriers (ATOMS.ARRIVE) |
| sm_90 | H100, H200, GH200 | clusters (UCGABAR, distributed shared memory, st.async, red.async), mbarriers (SYNCS), TMA (UTMA*, UBLK*: tile, im2col, multicast, reductions), warpgroup MMA (HGMMA/IGMMA/QGMMA/BGMMA), stmatrix, setmaxnreg, collectives |
| sm_100, sm_103 | B200, B300 | the uniform float datapath, tcgen05 (LDTM/STTM, UTC*MMA of every kind, UTCCP, UTCSHIFT, the Tensor Memory allocator), TMA gather4/scatter4 and CTA pairs, cluster launch control |
| sm_120 | RTX 5090 | sm_120's integer and float forms, block-scaled MMA |

Instructions the executor does not run, and so leave a kernel to its PTX:
`LDGMC` (multimem; the PTX engine has no multicast memory either), TMA's `im2col::w` modes (nor does the
PTX engine), and the texture forms with a LOD clamp, a LOD bias, offsets or a
depth compare. A `WARPSYNC.COLLECTIVE` reached from different code paths of
one warp is refused when it happens.

Dynamic parallelism runs on SASS as on PTX. A program using it links CUDA's
device runtime library into its cubin; the loader resolves the relocations
that brings (`R_CUDA_G64`, function descriptors: a kernel's descriptor is its
code address), and the device runtime's public entry points
(`cuda_device_runtime_api.h`: `cudaGetParameterBufferV2`, `cudaLaunchDeviceV2`,
the device-side last error, `cudaGetDevice`, device streams and events) run as
builtins in place of the library's code, whose own calls into the driver
(`__cuda_syscall_*`) are left to fail, by name, if anything reaches them. Child
grids run after their parent, in launch order, as the PTX engine runs them.

Kernel parameters past 4 KiB (CUDA 12.1 and later, up to 32764 bytes) come
with `KPARAM_INFO_V2` records and sit further into bank 0, past 0x8000, which
the 16-bit bank offsets hold as negative numbers: a bank address wraps in the
bank's 64 KiB.

The tensor map (`cuTensorMapEncodeTiled`) keeps its tile-mode fields where
NVIDIA's descriptor has them -- found with ptxas, one `tensormap.replace` field
at a time -- because SASS rewrites a map in place with plain stores.

## Tests

- `test_sass_decode`: every instruction in `nvidia/tests/data/sass/<arch>.txt`
  (sm_75 to sm_120a; encoding and nvdisasm's text, gathered by
  `nvidia/tools/sass-corpora.sh` from this repository's CUDA sources,
  PyTorch's CUDA code and the probes in `nvidia/tests/data/sass/probes`, which
  cover each instruction family's forms) decodes and prints back exactly as
  nvdisasm printed it.
- `e2e_sass_path`: which code a binary runs -- SASS by default, PTX as the
  fallback, the overrides -- and that both give the right answer.
- `e2e_sass_archs`: twenty programs built for each generation's SASS, sm_75 to
  sm_120, plus Hopper's (wgmma, TMA, tensor maps, stmatrix) for sm_90a and
  Blackwell's tensor core for sm_100a, each run by default -- checked to be
  running its SASS -- and on its PTX; the two must agree.
  `nvidia/tests/e2e/sass_archs.cu` keeps the forms that once ran wrong. The
  dynamic-parallelism programs among them (`dynamic_parallelism`,
  `cdp_device_api`, `rdc_device_api`) are built `-rdc=true` with cudadevrt.
- Every other CUDA end-to-end test runs on SASS wherever its binary carries
  it, which is the default now.
