# Texture and surface kernels on the card: a design note (#661, not built)

Turbo mode (`12-host-gpu-turbo.md`) keeps every kernel with a texture or surface instruction (`TEX`, `TLD`, `TLD4`, `TXD`, `TXQ`, `TMML`, `SULD`,
`SUST`, `SUATOM`, `SURED`) in the simulator: they are class `never`. The census programs are `dct8x8`, `simpleCubemapTexture`,
`simpleLayeredTexture` and `simpleSurfaceWrite` (about 1 s of the issue's 4 s). Tags: measured, read (a statement of this tree's code),
inferred, unknown.

## What the card would have to hold

* **A texture object per simulator texture object (read).** The simulator's table maps a handle to a `TextureDesc` (`include/vgpu/exec/texture.hpp`):
  base address, size, pitch, channel format, filter, address modes, border colour, mip levels, layered or cube. It is a view over device memory:
  a cudaArray is a dense row-major buffer in the simulator's window (`nvidia/src/runtime_api.cpp` `ArrayRec`), not the swizzled layout of a card.
  The card has no such buffer. For a 1D or 2D linear or pitched resource the card can make a texture object over the window's own address
  (`CU_RESOURCE_TYPE_LINEAR` / `PITCH2D`, no copy). For a 3D, layered, cubemap, mipmapped or surface array it needs a `CUarray` made with
  `cuArray3DCreate` (format, size, flags), the texel data copied in with `cuMemcpy3D` from the window, and, for a surface a kernel writes, copied back
  the same way. All four census programs use arrays (cubemap, layered, surface write, and dct8x8's 2D array).
* **The handles the kernel sees (read).** A kernel gets the simulator's handle as a 64-bit argument, or reads it from memory (a struct, a
  bindless table). The card's handle is another number. Either the launch path rewrites every argument that is a handle (parameter types do not say
  which 8-byte words are handles; the module's own `.param` list does not either) and the card's memory contents holding handles are left wrong, or the
  card's objects are made with the simulator's handle values, which `cuTexObjectCreate` does not allow (the driver picks them). Legacy texture
  references (`cuModuleGetTexRef`, a module-level `.texref`) are bound by symbol in the module and would be rebuilt per module load.
* **Coherence in both directions (read).** Every runtime call that moves array bytes (`cudaMemcpy2DToArray`, `cudaMemcpyToArray`, 3D copies,
  `cudaMemcpyArrayToArray`, mipmap level handles, frees) writes the dense buffer; the card's `CUarray` must see the same write before the next
  launch. A surface write (`simpleSurfaceWrite`) changes the card's array, which the simulator reads back through `cudaMemcpy2DFromArray`. The
  residency protocol (`10-host-gpu-hooks.md` section 8, 64 KiB chunks of allocations) tracks allocations by base address; an array's chunks would
  need an entry that also names the `CUarray`, and its copies go through `cuMemcpy3D`, not `cuMemcpyHtoD`.
* **Semantics (unknown).** Turbo does not promise the simulator's bits, so the card's own filtering would be accepted; what is unknown is how the
  census programs' outputs compare (cubemap faces, layered sampling, sRGB, border colours): the simulator models these from measurements on an
  RTX 3060 (`tex_border_raw`, `tex_srgb`, `tex_mip_lod`), and nothing here has run a texture kernel on the A10G.

## What the launch path would have to create

Per launch that reaches a kernel with a texture instruction: the set of live objects (the table, `Runtime::textures()`), a card object for each
one that is new or whose backing changed, the data of each array (a dirty flag per array, kept by the copy and free paths), the argument rewrite, and
after the launch the copy back of arrays a kernel could write (the surface objects). Creation and destruction hooks in both libraries
(`cudaCreateTextureObject`, `cudaCreateSurfaceObject`, `cuTexObjectCreate`, `cuSurfObjectCreate`, the destroy calls, `cudaFreeArray`,
`cudaFreeMipmappedArray`), a mirror table keyed by array, and the driver entry points `cuArray3DCreate`, `cuMemcpy3D`, `cuTexObjectCreate`,
`cuSurfObjectCreate`, `cuTexObjectDestroy`, `cuSurfObjectDestroy`, `cuArrayDestroy` in the loader. Estimate (inferred, by counting the hooks above):
about 600 to 900 lines, a third of them in `nvidia/src`, which the mode does not touch today, and a test that needs a card (the stand-in driver cannot
tell whether a texel was sampled correctly).

## Failure modes

* A handle in memory (not an argument) names nothing on the card: an illegal-address-class error that ends the mode (`im.dead`), and when the card
  holds data the simulator lacks the process ends. The mode could only prove "no handle in memory" by refusing every kernel of a module that
  passes a texture object through a struct, which is most of them.
* An array the program rewrites between launches while the mirror is stale: wrong pixels, silent. Every copy path must dirty the mirror.
* A surface kernel and a simulator launch alternating: each launch copies the array both ways; for imaging loops (`dct8x8` launches twice) the copies
  cost more than the kernel (inferred).
* An array that does not fit the card's limits or a format the card lacks (the simulator accepts block-compressed and packed formats): refused by
  the fit check, with a new `FitRefusal`.

## Decision

Not built: more than 300 lines, in two libraries the mode does not touch, with no way to test the results without a card, and about 1 s of gain in
the census. First step when someone builds it: a card-only probe (`tools/probes/`) that makes a 2D array texture and a surface on the A10G, runs
the `dct8x8` kernel on it, and compares with the simulator, to learn whether the outputs agree before any mirroring is written.
