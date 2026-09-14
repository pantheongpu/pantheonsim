# The driver export tables ("dark API")

NVIDIA's CUDA runtime comes in two forms. The **shared** runtime
(`nvcc -cudart shared`) resolves `libcudart.so` at load time, so VirtualGPU's
own `libcudart` answers and everything works — this is the supported path, and
everything else in this project is built on it. The **static** runtime, which is
nvcc's default, puts NVIDIA's runtime *inside the binary*; it then reaches the
driver not through the documented driver API but through `cuGetExportTable`, a
set of undocumented vtables looked up by UUID.

This page records what a static runtime actually asks for, because that question
comes up every time someone tries to run a default-built binary, a profiler that
bundles its own CUPTI, or CuPy.

Everything here was obtained by **observing what the runtime asks us** — the
UUIDs it requests, the arguments it passes to slots we serve it, the order it
does things in. Nothing was obtained by disassembling NVIDIA's runtime, and no
NVIDIA code appears in this project.

## What happens today

A static-cudart binary gets as far as failing cleanly:

```
$ vgpu run --gpu nvidia/a10 ./prog_built_with_default_nvcc
cudaGetDeviceCount(&n)  -> 103 cudaErrorSoftwareValidityNotEstablished
```

Error 103 is documented in `driver_types.h` as:

> By default, the CUDA runtime may perform a minimal set of self-tests, as well
> as CUDA driver tests, to establish the validity of both. [...] this error
> return indicates that at least one of these tests has failed and the validity
> of either the runtime or the driver could not be established.

Every subsequent CUDA call returns 103 as well: the runtime latches the failure
during its own initialization and never reaches a device.

## The bootstrap, in order

With `VGPU_TRACE=2` (every driver call) and `VGPU_TRACE=1` (table lookups and
slot calls with arguments), the whole sequence is visible. For CUDA 13.0:

| # | UUID | Name | Required? |
| --- | --- | --- | --- |
| 1 | `f8cff951-2146-8b4e-b9e2-fb469e7c0dd9` | unknown; fetched before `cuInit` | no — refusing it changes nothing |
| 2 | `6bd5fb6c-5bf4-e74a-8987-d93912fd9df9` | cudart interface | yes — refusing gives a clean `cudaErrorNotSupported` |
| 3 | `a094798c-2e74-2e74-93f2-0800200c0a66` | tools runtime hooks | yes — same |
| 4 | `42d85a81-23f6-cb47-8298-f6e78a3aecdc` | unknown | **mandatory** — the process aborts before `main` without it |
| 5 | `c693336e-1121-df11-a8c3-68f355d89593` | context-local storage | **mandatory** — same |
| 6 | `263e8860-7cd2-6143-92f6-bbd5006dfa7e` | unknown | **mandatory** — same |
| 7 | `d4082055-bde6-704b-8d34-ba123c66e1f2` | telemetry | no |

Interleaved with those, through the ordinary documented driver API:
`cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`,
`cuDeviceTotalMem`, `cuDeviceGetUuid`, and ~73 `cuDeviceGetAttribute` queries.

Then the runtime gives up. It never creates a context, never allocates, never
launches anything.

### Table 7 is telemetry

Its slot 1 is called three times with `(id, magic, record, 0, code, 0)`:

```
slot 1(0x32c8, 0x6a9c79c9, 0x...d60, 0, 0x0a, 0)
slot 1(0x32c9, 0x6a9c79c9, 0x...d70, 0, 0x65, 0)
slot 1(0x32ca, 0x6a9c79c9, 0x...d80, 0, 0x66, 0)
```

The first argument increments per call and the third advances by 16 bytes, so it
is walking an array of records. The second argument looks like a magic number
until you run the program twice: between two runs it went from `0x6a9c79c9` to
`0x6a9c7a74`, a difference of exactly the seconds that elapsed. It is a Unix
timestamp, and these three calls are the runtime recording that its
initialization failed.

## What the failure is not

Three hypotheses, each tested and eliminated:

- **Not a missing table.** All seven are served. Withholding them one at a time
  (`VGPU_DARK_DENY=<uuid prefix>`) only ever makes things worse, never
  different.
- **Not unfilled out-parameters.** `VGPU_DARK_FILL=1` makes every stub zero the
  memory at each argument that looks like a writable pointer, so the runtime
  reads a definite value rather than whatever was on its stack. No change.
- **Not the device model.** The ~73 attribute queries are now all answered from
  documented per-compute-capability values rather than falling through to zero
  (fixing, along the way, ten entries that were filed under the wrong attribute
  number). No change.

And, decisively: **the runtime never runs a functional test.** It does not
allocate memory, launch a kernel, or compare a result against an expected value.
Whatever "validity" it establishes, it establishes from the table interactions
alone — before it has asked the driver to compute anything.

## Why this is where it stops

Getting past error 103 would mean producing the exact values some slot is
expected to return, for a slot with no specification. The only ways to obtain
that specification are NVIDIA's internal headers or disassembly of their
runtime, and this project uses neither: `nvidia/include/vgpu_cuda.h` is clean-room from
public documentation, and that constraint is worth more than static-cudart
support.

So the position is unchanged, but now for a precise reason rather than a general
sense of brittleness: **build against the shared runtime.**

```bash
nvcc -cudart shared my_app.cu -o my_app     # or build inside `vgpu shell`
```

`vgpu run` detects the static case up front — a `.nv_fatbin` section with no
CUDA library in `DT_NEEDED` — and says so before the program starts, rather than
letting it fail later with an error that points nowhere.

## Consequences elsewhere

- **CuPy** (as shipped in its wheels) links the runtime statically and dies
  here, at `cudaGetDeviceCount`, before it compiles a kernel. See
  [jit.md](jit.md). Numba and Triton are unaffected, because neither loads a
  CUDA runtime at all.
- **Nsight Systems** collects through its own bundled CUPTI, loaded by absolute
  path, which reaches the driver the same way — so `nsys` produces a report with
  OS runtime traces and no CUDA data. `nvprof` works, because its path goes
  through the public CUPTI that VirtualGPU does implement. See
  [cupti.md](cupti.md).

## The knobs used here

Both are diagnostics for this question, not features:

| Variable | Effect |
| --- | --- |
| `VGPU_TRACE=1` | Table lookups, and every stub call with its six arguments |
| `VGPU_TRACE=2` | Also every driver call and its result |
| `VGPU_DARK_DENY=<hex prefix>,...` or `all` | Refuse the named tables |
| `VGPU_DARK_FILL=1` | Zero the memory at pointer-shaped stub arguments |
