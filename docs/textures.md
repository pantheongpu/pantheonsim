# Textures and surfaces

Texture and surface objects work for point sampling, which is what the
overwhelming majority of CUDA code uses:

```cpp
cudaResourceDesc rd{};
rd.resType = cudaResourceTypeLinear;
rd.res.linear.devPtr = ptr;
rd.res.linear.desc = cudaCreateChannelDesc<float>();
rd.res.linear.sizeInBytes = bytes;

cudaTextureObject_t tex = 0;
cudaCreateTextureObject(&tex, &rd, &td, nullptr);
// ... kernel: tex1Dfetch<float>(tex, i)
```

## What a texture fetch is here

A texture object is a 64-bit handle the host creates and the kernel receives as
an ordinary parameter. The PTX that reads it carries only the handle and the
coordinates:

```
tex.1d.v4.f32.s32  {%f1,%f2,%f3,%f4}, [%rd1, {%r1}];
tex.2d.v4.f32.f32  {%f5,%f6,%f7,%f8}, [%rd1, {%f2, %f4}];
suld.b.2d.b32.trap {%r12}, [%rd1, {%r11, %r2}];
sust.b.2d.b32.trap [%rd1, {%r11, %r2}], {%r13};
```

Everything else about the fetch — what memory backs it, how wide a texel is,
what happens off the edge — comes from a table the launch consults, built by
`cudaCreateTextureObject`. The device owns that table, because a texture object
outlives any particular module.

**No texture cache is modelled.** VirtualGPU has no memory-hierarchy model, so a
fetch reads the same bytes an ordinary load would. What *is* modelled is the
addressing and the format conversion, because those change results rather than
timing.

## Implemented

| | |
| --- | --- |
| Instructions | `tex.1d`, `tex.2d`, `tex.3d`, `suld.b`, `sust.b` |
| Backing memory | linear (`cudaResourceTypeLinear`), pitched 2D, `cudaArray` |
| Addressing | clamp, wrap, mirror, border |
| Coordinates | unnormalized and normalized, integer and float |
| Channels | 1–4, signed / unsigned / float, any width the format gives |
| Read mode | `cudaReadModeElementType`, `cudaReadModeNormalizedFloat` |
| Runtime API | `cudaCreateTextureObject`, `cudaCreateSurfaceObject`, their destroys, `cudaMallocArray`, `cudaFreeArray`, `cudaMemcpy2DToArray`, `cudaMemcpy2DFromArray`, `cudaGetChannelDesc`, `cudaCreateChannelDesc` |

Two details worth naming because getting them wrong is invisible:

- **An absent channel reads as 0, except `w`, which reads as 1.** A kernel
  reading `.w` of a one-channel texture expects 1. Returning 0 there produces a
  black image rather than an error.
- **A `cudaArray` is dense row-major here.** Hardware stores arrays in an
  opaque swizzled layout only the texture units can address, and nothing outside
  those units is allowed to depend on it — so a plain buffer serves, and
  `cudaMemcpy2DToArray` is an ordinary strided copy.

## Refused, and why

`cudaFilterModeLinear` is **not** implemented, and this is a deliberate refusal
rather than a gap waiting to be filled. Interpolation between texels is a
documented weighted average, but hardware computes the weights in a fixed-point
format with 8 fractional bits. A straightforward float implementation would
agree to about three decimal places and differ in the low bits — which is
exactly the class of divergence the differential testing in this project exists
to catch. Producing a plausible-looking wrong number is worse than refusing.

Point sampling has no such problem: it selects a texel, and the answer is exact.

Also refused, each with its own message: mipmapped fetches (`tex.level`,
`tex.grad`), layered and cubemap textures, resource views, sRGB, anisotropic
filtering, and the `.clamp` / `.zero` out-of-range policies on `suld`/`sust`.

## Surfaces fault rather than clamp

`suld` and `sust` carry a `.trap` out-of-range policy, which is what a surface
access compiles to by default, and it means what it says:

```
[vgpu] surface access at byte x=64, y=0 is outside the 4x1 surface
       (16 bytes per row). The instruction's '.trap' policy is what makes
       this a fault rather than a clamp.
```

Clamping instead would turn an indexing bug into a plausible picture.

## Handles are not interchangeable

A surface object and a texture object have the same shape of handle. Using one
where the other belongs is caught and named, as is a handle that was never
created — reading through an uncreated handle would otherwise produce plausible
garbage from wherever it happened to point.
