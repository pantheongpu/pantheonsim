# Python JIT frameworks

Numba, Triton and CuPy all end up producing GPU code at runtime, and all three
produce PTX along the way. VirtualGPU consumes PTX, so the only question for
each is whether it will hand the PTX over — and the three answers are
different.

| Framework | Works | How it reaches the interpreter |
| --- | --- | --- |
| Numba | yes, unmodified | its own PTX, assembled through the driver's JIT link API |
| Triton | yes, with `nvidia/tools/vgpu_triton.py` | its own PTX, handed over before `ptxas` runs |
| CuPy | no | statically links cudart; never reaches either path |

## Numba

Nothing to configure. Point `LD_LIBRARY_PATH` at the shim and run:

```bash
VGPU_GPU=nvidia/a10 LD_LIBRARY_PATH=/path/to/build/shim python your_program.py
```

Numba talks to `libcuda` directly and loads no CUDA runtime at all, so it
touches none of the fatbin or host-registration machinery the rest of the shim
exists for. Two things had to be true for it to work:

- **Every `cuIpc*` symbol must exist.** Numba resolves the IPC family during
  `cuInit` and fails at import if a symbol is missing — before it will report a
  device, let alone launch anything. VirtualGPU's per-process virtual VRAM makes
  a cross-process memory handle meaningless, so each of these refuses with
  `CUDA_ERROR_NOT_SUPPORTED`; they exist to be found, not to be called.
- **The JIT link API must produce something loadable.** Numba compiles Python to
  PTX itself and then calls `cuLinkCreate` / `cuLinkAddData` / `cuLinkComplete`
  to turn it into a module image. On hardware that image is a cubin. Here the
  module loader consumes PTX, so `cuLinkComplete` merges its PTX inputs and
  returns the merged text — everything the caller does with the result still
  works, because the result is genuinely what the loader wants.

Numba's PTX also needed three things the parser did not have: `.common` linkage
(a tentative definition — zero-initialised, merged at link time — which Numba
emits once per kernel), decimal integer literals above `INT64_MAX` (ptxas prints
the float sign-bit mask that way), and the extended-precision carry family
(`add.cc`/`addc`, `sub.cc`/`subc`, `mad.lo.cc`/`madc.hi`), which is how Numba
builds 64-bit index arithmetic.

## Triton

Triton needs one line, because it runs `ptxas` itself:

```python
import vgpu_triton; vgpu_triton.install()
```

with `nvidia/tools/` on `sys.path`. Everything after that is ordinary Triton.

`vgpu_triton.available()` says whether the installed Triton is new enough;
releases predating `knobs.runtime.add_stages_inspection_hook` can only be driven
through `ptxas`, and there is nothing to do with the cubin that produces.

The hook ends Triton's compilation pipeline at PTX instead of letting it
assemble a cubin. It uses `knobs.runtime.add_stages_inspection_hook`, Triton's
own documented extension point for rewriting stages, so it depends on one public
name and nothing else. Refusing the cubin rather than guessing at it is
deliberate: SASS is undocumented, and a decoder for it is not something this
project will contain.

`tl.dot` works, which means Triton's tensor-core path works: it emits
`ldmatrix.sync.aligned.m8n8.x4.shared.b16`, the form that names the shared space
in the instruction rather than converting the address first, and `mma.sync`
underneath.

## CuPy

CuPy does not work, and the reason is not fixable from the PTX side. It links
the CUDA runtime **statically** rather than loading `libcudart.so`, so
`LD_LIBRARY_PATH` never reaches it. Its embedded runtime asks the driver for the
undocumented export tables (`cuGetExportTable`) during startup and fails at
`cudaGetDeviceCount` with `cudaErrorSoftwareValidityNotEstablished` — before it
has compiled a single kernel.

Building CuPy from source does not change this: its build links
`cudart_static` either way, so a source-built CuPy has no `libcudart` in its
`DT_NEEDED` and fails in exactly the same place. (Checked, because it is the
obvious thing to try.)

Unblocking CuPy means implementing enough of the dark API to satisfy a static
cudart, which is the same work Nsight Systems needs. That has now been
investigated in detail and stopped for a specific reason rather than a general
one -- see [dark-api.md](dark-api.md). The short version: the runtime's validity
self-test never asks the driver to compute anything, so passing it is not a
matter of being functionally correct, and the values it does want have no
specification a clean-room project can read.

## Running the tests

```bash
VGPU_PY=/path/to/python-with-numba-and-triton ctest -R jit_frameworks
```

The test skips cleanly when neither framework is installed.
