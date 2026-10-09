# Textures and surfaces

Texture and surface objects work -- point and linear filtering, layered and
cubemap textures, mipmaps with an explicit level of detail, and gather -- and
every rule a result depends on was measured on an RTX 3060 (sm_86) and
matched bit for bit:

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
| Instructions | `tex.{1d,2d,3d,a1d,a2d,cube,acube}`, `tex.level`, `tld4.{r,g,b,a}.2d`, `suld.b`/`sust.b` in `.1d`/`.2d`/`.3d`/`.a1d`/`.a2d` |
| Backing memory | linear (`cudaResourceTypeLinear`), pitched 2D, `cudaArray` (1D/2D/3D, layered, cubemap, layered cubemap), mipmapped arrays |
| Filtering | point and linear, within and between mip levels |
| Addressing | clamp, wrap, mirror, border (any colour) |
| Coordinates | unnormalized and normalized, integer and float |
| Channels | 1–4, signed / unsigned / float, any width the format gives |
| Read mode | `cudaReadModeElementType`, `cudaReadModeNormalizedFloat` |
| Runtime API | `cudaCreateTextureObject`, `cudaCreateSurfaceObject`, their destroys, `cudaMallocArray`, `cudaMalloc3DArray`, `cudaFreeArray`, `cudaArrayGetInfo`, `cudaMallocMipmappedArray`, `cudaGetMipmappedArrayLevel`, `cudaFreeMipmappedArray`, `cudaMemcpy2DToArray`, `cudaMemcpy2DFromArray`, `cudaMemcpy3D`/`Async`, `cudaGetChannelDesc`, `cudaCreateChannelDesc` |

Two details worth naming because getting them wrong is invisible:

- **An absent channel reads as 0, `w` included.** The graphics APIs return 1
  for a missing alpha, and this simulator used to; an RTX 3060 returns 0 for
  every format, read mode, filter and resource type, so a kernel that reads
  `.w` of a one-channel texture gets 0.
- **A `cudaArray` is dense row-major here.** Hardware stores arrays in an
  opaque swizzled layout only the texture units can address, and nothing outside
  those units is allowed to depend on it — so a plain buffer serves, and
  `cudaMemcpy2DToArray` is an ordinary strided copy.

## Measured, not assumed

Linear filtering was refused here for a long time, because the programming
guide gives the formula -- `tex(x) = (1-a)T[i] + aT[i+1]`, `a` in 9-bit fixed
point with 8 fractional bits -- but not the arithmetic around it, and a float
implementation would differ from the device in the low bits. The arithmetic
was then measured, sample by sample, until every result matched:

- **The weight** rounds to the nearest 1/256, halves up; 256 carries into
  the texel index. Normalized coordinates scale by the size in f32. In clamp
  mode the *coordinate* is clamped to [0.5, size - 0.5] before the half is
  subtracted (it only shows in 3D).
- **A 1D texture is 2D, height 1, sampled at y = 0**, so under border
  addressing every result blends in half a border row. A layer of a 1D
  layered texture filters as true 1D.
- **The weights are integers summing to 256**, split one axis at a time --
  z, then x, then y -- each split rounding half up; the y split rounds the
  upper part on the x = 1 side and the lower part on the x = 0 side (in 2D:
  `w11 = round(a*b/256)`, `w10 = a - w11`, `w01 = b - w11`).
- **The sum** of weight x texel is rounded once: to f32, ties away from
  zero, for float texels; to half for half texels. It is not quite exact.
  Within each 2x2 footprint (a 3D fetch's z-slice, a mip blend's level) every
  value is first truncated toward zero to 2^(E - 27) -- 2^(E - 14) for a half
  -- where E is the exponent of the footprint's largest value with a
  non-zero weight, so 2.75 at weight 1 blended with 0.1 at weight 255 gives
  0x3de207ff where the exact sum rounds to 0x3de20800. Each footprint's sum
  is floored to 2^(Emax - 28) and the footprints added exactly. A subnormal
  float texel reads as zero, a subnormal result is flushed keeping its sign,
  an all-(-0) blend gives -0, and a NaN comes out as all ones (0x7fffffff,
  or 0x7fff for a half). This was found with texels spanning 2^-10..2^11;
  over 15,360 such trilinear fetches, 20 still differ by 1-4 ulp, all where
  the two slices' magnitudes differ widely and the result cancels far below
  both. (The earlier tests' random texels, drawn as lo + x, never carried
  bits that fine.) 8- and 16-bit
  unsigned normalized texels filter as 16-bit integers (an 8-bit `u` is
  `u*257`) and 16-bit signed ones as themselves, rounded half up, read out
  as `K/65535` or `K/32767` (signed clamped to -32767 after the blend).
- **Wrap and mirror** apply only to normalized coordinates; with
  unnormalized ones the hardware clamps.
- **Layers**: the index is unsigned, and past the end (a negative index
  included) reads the last layer.
- **Cubemaps**: the face of the direction's largest-magnitude axis, ties
  going to z, then y, then x; the guide's table for the minor axes;
  `(s/m + 1)/2` as the face coordinate. Point sampling clamps to the face
  whatever the address mode; linear filtering applies the mode inside the
  face. No filtering across faces.
- **Mipmaps**: the level of detail is held in 1/256ths of a level -- an
  explicit lod truncated toward zero, the bias truncated likewise and added
  (a plain fetch is level 0, no bias), then the texture's level clamps, then
  the levels that exist. Point filtering between levels takes the nearer
  one, halves up; linear splits the weight between two levels first, then
  each level's share over its own footprint (in 3D the z split rounding its
  lower part).
- **Gather** (`tld4`) returns the footprint's four texels counter-clockwise
  from the lower left -- `(i, j+1), (i+1, j+1), (i+1, j), (i, j)` -- with the
  weight rounding's carry, and clamp applied to each index. A NaN comes out
  as all ones, a subnormal float as zero, and a signed 8-bit normalized
  texel as its 16-bit form over 32767 (`|k| * 258`, plus one from 64 up --
  65 when negative), not `k/127`.
- **Mipmapped textures** take normalized coordinates whether or not the
  descriptor asks for them; without `normalizedCoords`, wrap and mirror
  still act as clamp. Layered and cubemap ones have every layer (and face)
  in every level, each of that level's size. Point sampling a *mipmapped*
  cubemap applies the address mode, where a plain cubemap clamps to the
  face. One thing is not modelled: the card projects a cube direction by
  multiplying by its approximate reciprocal, which for some magnitudes is an
  ulp low, so a direction that ties two axes -- a face coordinate of exactly
  1.0 -- can land inside the face on the card and on its edge here.
  `e2e_texture_mip_layers` hashes 240 cases against the card.
- **A half NaN** widens to float bit for bit, payload and all.

## sRGB

`cudaTextureDesc::sRGB` decodes an 8-bit unsigned normalized texture read as
normalized float from sRGB to linear -- x, y and z of a four-channel texture,
x of a one- or two-channel one. Alpha is left alone, and so is every other
format (signed, 16-bit, integer-read, float). Measured on an RTX 3060:

- **The decode is a table, not the sRGB formula.** Its 256 entries, in
  1/65536ths, have at most 8 significant bits, and the low codes run at
  20/65536 a step where `1/(255 * 12.92)` gives 19.9. The simulator carries
  the measured table.
- **A linear blend** blends the table values, but the table holds them in
  blocks of eight codes sharing an exponent (that of the block's last
  entry): within each 2x2 footprint every value is truncated to 2^(E - 7),
  E the largest block exponent among the texels with weight, and the exact
  sum is rounded once to a half's precision, ties away. So code 11 counts as
  zero next to code 184, and code 50 as exactly 1/32 next to 200. Fitted to
  134,316 two-texel blends, all matched.
- **A border colour** is encoded to an sRGB code by the sRGB formula --
  rounding a shade early, a fraction of 0.4989 already going up (7,100
  colours) -- and decoded like a texel.

`e2e_texture_srgb` hashes 228 cases (formats, point, linear, gather, 1D, 3D,
layered, cubemap, mipmapped, border colours) against the card.

## Border colours

With border addressing, a fetch outside the texture reads
`cudaTextureDesc::borderColor` -- converted to the texture's format first,
which is where the rules are (measured over 280,000 colours):

- a 32-bit float channel takes the float as it is, NaN payloads included;
- a half channel takes it rounded toward zero (65520 gives 65504; a NaN keeps
  the top of its payload, and stays a NaN);
- an m-bit (magnitude) normalized channel clamps it to [0, 1] or [-1, 1] (NaN
  is 0), truncates it toward zero to m + 4 fractional bits, v, and takes
  `(|v| * (2^m - 1) + 2^(m+3) - 1) >> (m + 4)` with v's sign -- so 0.5 on an
  8-bit channel is 127/255, and the tie-break drifts with the code rather
  than sitting at k + 0.5;
- an integer channel read as an integer takes the float's low bits (1e30,
  0x7149f2ca, reads as 0xca from an 8-bit unsigned channel, 0xffffffca from
  a signed one);
- a channel the format lacks reads 0.

The converted colour then stands in for each outside texel wherever one is
read: point sampling, the linear blend, gather, 1D (whose filter blends in a
border row), 3D, layered, cubemap and mipmapped fetches. `e2e_border_colour`
hashes 705 cases of these against the card's results.

The tests carry this as data from the hardware: `e2e_texture_filtering`,
`e2e_texture_layers`, `e2e_texture_mipmaps` and `e2e_texture_gather` hash
tens of thousands of results against the hashes the same programs produced
on the RTX 3060 (and pass there too), and test_exec3 has a hardware table
for each rule's corners.

## Integer coordinates

A fetch with integer coordinates (`tex1Dfetch`, and `tex.*.s32` on an array)
names a texel, and an RTX 3060 takes that texel whatever the texture's
settings say:

- the filter mode does not apply: a texture set up for linear filtering is
  point-sampled (CUDA Samples' convolutionFFT2D binds linear memory that way
  and fetches it with `tex1Dfetch`);
- outside the extent the result is zero in every channel, under clamp and
  border addressing alike and whatever the border colour (a colour of 7
  still gives 0), while float coordinates on the same texture clamp or take
  the border colour as usual.

This used to be refused ("linear filtering with integer coordinates"), and
out-of-range integer coordinates were clamped. e2e_texture_int_coords checks
both against the card.

## Normalized, block-compressed and packed formats

An array may be made from any channel descriptor (`cudaChannelFormatKind*`) or driver format
(`CU_AD_FORMAT_*`) the card accepts, through one table (`nvidia/src/texture_formats.hpp`) that
the runtime and the driver share: unsigned and signed normalized 8- and 16-bit formats, BC1 to
BC5 (each with its sRGB variant where there is one), BC6H and BC7 (arrays only), and 10:10:10:2.
What a descriptor makes, which read modes, filters and sRGB flags a format takes, the errors
(`cudaErrorInvalidChannelDescriptor`, `cudaErrorInvalidNormSetting` for a read mode the format
cannot be read in, `cudaErrorInvalidFilterSetting`) and the driver's own answers (a format fixes
its channel count; linear and pitched memory take the plain formats and 10:10:10:2 only; block
data cannot be a surface or have the surface flag) are the card's, recorded in
`runtime_texture_gaps_expected.inc` and `driver_texture_gaps_expected.inc`.

The values are the card's. The block decoders (`include/vgpu/exec/block_compression.hpp`) are not
the format descriptions' integer formulas: every number was measured with point-sampled blocks
covering every endpoint pair of every channel, and the decoders reproduce them value for value.

| Format | What the texture unit delivers |
| --- | --- |
| BC1 | 8-bit. Red and blue widen to 16 bits (`round(q * 65535 / 31)`), blend as (2a + b) / 3 and (a + 2b) / 3 and keep the high byte; green widens by bit replication and blends with weights of 321/1024 and 703/1024 on the second endpoint, rounded. Three-colour mode: red/blue `(33 (q0 + q1)) >> 3`, green weight 513/1024, the fourth colour transparent black |
| BC2, BC3 colour | as BC1's four-colour mode; BC2 alpha is `a4 * 17` |
| BC3 alpha | 8-bit: `floor((a0 * 2048 + (a1 - a0) * N + 1024) / 2048)`, N = 289, 578, 892, 1156, 1470, 1759 (eight values) or 385, 770, 1278, 1663 (six values and the two ends) |
| BC4, BC5 unsigned | 16-bit: `a0 * (257 - W) + a1 * W` with W = 36, 72, 113, 144, 185, 221 (eight values) or 48, 96, 161, 209 (six values, then 0 and 65535) |
| BC4, BC5 signed | 16-bit signed: endpoints widen by 32767/127 (-128 reads as -127) and blend with the same steps in that scale, rounded to nearest |
| 10:10:10:2 | normalized floats; filtering widens the codes by bit replication to 16 bits (`(u << 6) \| (u >> 4)`), as the card does |

Linear filtering, wrapped and mirrored addressing of all of these return exactly the card's
floats (`runtime_texture_gaps` hashes them; the same program passes against NVIDIA's runtime).
`maxAnisotropy` of any value is accepted and changes nothing for a fetch with no derivatives, as
on the card. A resource view reinterprets the same bytes as another format of the same texel
size (a `uint2` array as BC1 blocks, with four times the extent); the read mode is checked
against the array's own format, then the view.

## Refused, and why

Each with its own message, rather than a plausible wrong number:

- **`tex.grad`.** The level of detail it needs comes out of the GPU's
  approximate log2 and length units -- axis-aligned gradients land within
  1/256 of log2, others on no textbook formula -- which are not documented.
- **Linear filtering of signed 8-bit normalized texels.** The result is a
  function of the blended sum alone, but not one reproduced here. (A later
  attempt with the texels' measured 16-bit forms, and with fixed-point
  texels and the border colour's output rounding, peaked at 96.7% of 18,176
  two-texel blends.)
- `tld4` on layered or cubemap textures (the runtime refuses a gather array
  that is layered or a cubemap, so there is nothing to measure them on).
- **BC6H and BC7.** Arrays of these formats can be made, filled and copied, but
  creating a texture object over one answers `cudaErrorNotSupported` /
  `CUDA_ERROR_NOT_SUPPORTED` with a message on stderr (once). Their decoders need
  the mode and partition tables, which were not measured (they can be, one
  partition at a time, from the card).

## Surfaces out of range: .trap, .clamp and .zero

`suld` and `sust` carry a `.trap` out-of-range policy, which is what a surface
access compiles to by default, and it means what it says:

```
[vgpu] surface access at byte x=64, y=0 is outside the 4x1 surface
       (16 bytes per row). The instruction's '.trap' policy is what makes
       this a fault rather than a clamp.
```

Clamping instead would turn an indexing bug into a plausible picture.

The other two policies do what an RTX 3060 does with them. The ISA's
descriptions ("the nearest surface location, sized appropriately") left
open how, so these were measured (`nvidia/tests/e2e/surface_oob.cu`, which
compares 245 results with the card's):

- **`.clamp`** moves each coordinate to the nearest place in the surface:
  - x to the last position, aligned to the access, where the whole access
    fits. That is by the byte, not the texel: an 8-bit load at x = 32 on a
    32-byte row reads byte 31, and a 16-byte `.v4` on a 24-byte row reads
    from byte 0, not 8.
  - y and z into their range, and the layer to the last layer (the index is
    unsigned, so -1 is past the end).
  - A store lands where a load would read.
- **`.zero`**: if any byte of the access is out of range, a load reads zero
  and a store is dropped, the whole access even when only part of a vector
  is outside.
- **A misaligned x** faults under every policy, as it does on the card; the
  ISA leaves it undefined.

## Handles are not interchangeable

A surface object and a texture object have the same shape of handle. Using one
where the other belongs is caught and named, as is a handle that was never
created — reading through an uncreated handle would otherwise produce plausible
garbage from wherever it happened to point.
