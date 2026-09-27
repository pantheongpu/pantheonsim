# Blackwell (sm_100a): Tensor Memory and tcgen05

A simulated B200 (`VGPU_GPU=nvidia/b200`, compute capability 10.0) runs code
built for `sm_100a` or `sm_100f` that uses the fifth-generation tensor core:
Tensor Memory, `tcgen05.*`, and what CUTLASS's SM100 kernels need around them
(cluster launch control, TMA's `.cta_group::2` and the wider swizzle atoms).
Everything here follows the PTX ISA's text and figures (section 9.7.18); none
of it comes from looking at how NVIDIA's hardware or libraries do it inside.

## Tensor Memory

Each CTA has 128 lanes by 512 columns of 32-bit cells. `tcgen05.alloc` hands
out a power-of-two run of at least 32 columns (any multiple of 32 with
`.exclusive`) and writes its address -- lane 0, the first column -- to shared
memory. Where in the 512 columns a run goes is not documented; here it is the
lowest free run aligned to its size, and a kernel uses the address it is given.
Contents start at zero, as shared memory does here; on hardware they are
undefined.

With `.cta_group::2` the two CTAs of a pair (cluster ranks differing in the
last bit) allocate together: the first warp to arrive allocates the same
columns in both, and the peer's warp takes that allocation when it gets there.
Deallocation pairs the same way.

Checked, because each is a bug the hardware would not report clearly:

- a block that exits with columns still allocated (the ISA requires a
  `tcgen05.dealloc` first);
- a `tcgen05.ld`/`st` outside the warp's quarter of the lanes (warp `w` of a
  warpgroup reaches lanes `32(w%4)` to `32(w%4)+31`), or into columns no
  allocation covers;
- an allocation after `tcgen05.relinquish_alloc_permit`;
- a `.cta_group::2` operation in a CTA with no peer.

An allocation that does not fit would block until another warp frees columns;
waiting for that is not implemented, so it fails with a message instead.

## tcgen05.ld and tcgen05.st

All five shapes -- `.32x32b`, `.16x64b`, `.16x128b`, `.16x256b` and
`.16x32bx2` -- with `.x1` to `.x128`, and `.pack::16b`/`.unpack::16b`. Each
thread's registers go where figures 186-190 put them. For `.16x32bx2` the
figure draws the second half-warp at lanes 16-31, but the text says the shape
accesses 16 lanes and CUTLASS's copy traits agree: threads 16-31 use lanes
0-15 at `immHalfSplitoff` columns further on.

## tcgen05.mma

One thread issues a whole `M x N x K` product; every active thread that
reaches the instruction issues its own, which is what single-thread semantics
means (CUTLASS elects one).

- Kinds: `.kind::f16` (f16 and bf16 into f16 or f32), `.kind::tf32`,
  `.kind::f8f6f4` with the 8-bit types (e4m3, e5m2), and `.kind::i8` (s8/u8
  into s32, with the saturate bit).
- Shapes: `.cta_group::1` with M = 64 or 128, and `.cta_group::2` with M = 128
  or 256; N from the instruction descriptor.
- D's placement is the data-path layout of 9.7.18.10.5: layout D (M = 128,
  lane = row), F (M = 64, rows in lanes 0-15 or 16-31 of each quarter), A
  (M = 256 over a pair, 128 rows each) and B (M = 128 over a pair, 64 rows
  each, the upper half of N in lanes 64-127 at the same columns). An f16 D is
  one half in the low 16 bits of its cell.
- Over a pair, each CTA supplies half of A's rows and half of B's columns
  from its own shared memory at the descriptor's offsets.
- A can come from Tensor Memory (`[a-tmem]`), packed 32 bits to a column. It
  is read from the lane each D row is written to, so layout B's duplicated A
  must really be in both halves, as on the hardware.
- `enable-input-d`, `scale-input-d`, `disable-output-lane` (one CTA), and the
  instruction descriptor's negate and transpose bits.
- Shared-memory descriptors: the canonical layouts of 9.7.18.3.3 with no
  swizzle, or 32-, 64- or 128-byte swizzles, including the 128-byte swizzle in
  32-byte atoms (mode 1, CuTe's `Swizzle<2,5,2>`). A nonzero base offset and
  the absolute leading-dimension mode (sm_103a) are refused.

Numerics are wgmma's: every product of these input types is exact in f32 and
the sum is kept in f32, in K order. The ISA does not say how tf32 inputs are
narrowed for tcgen05; they are truncated to tf32 as it says for wgmma.

### When things complete

The asynchronous operations -- `mma`, `ld`, `st` -- complete when they are
issued. The ISA lets them complete any time up to their `tcgen05.commit` or
`tcgen05.wait`, and at once is one of the allowed orders; the fences and waits
then have nothing left to order. `tcgen05.commit` arrives on its mbarrier
straight away, on every CTA its mask names with `.multicast::cluster`. The
consequence is the same as for wgmma: a kernel that reads an accumulator
before waiting for it is not caught here.

## Around the tensor core

- **Cluster launch control.** `clusterlaunchcontrol.try_cancel` takes the next
  cluster (or block, without clusters) that has not started in the range the
  running host thread is working through; it then never launches, and the
  canceller gets its first CTA's coordinates. With nothing left to take, the
  request fails, as it does on a device with nothing pending. This exercises
  the persistent loop of CUTLASS's SM100 scheduler rather than letting every
  tile run as its own cluster. The 16-byte response is this engine's own
  encoding; the ISA calls it opaque.
- **`.b128` registers.** A `.reg .b128` is held as two 64-bit registers;
  `ld`/`st` `.b128` move both halves, and `query_cancel` reads them.
- **TMA `.cta_group::2`.** A copy may complete on the barrier in its
  destination's peer CTA, and a multicast one signals, for each destination,
  whichever CTA of its pair has the barrier's rank parity.
- **TMA `.tile::gather4` / `.tile::scatter4`.** Four rows of a 2D tensor at
  one x (`{x, row0, row1, row2, row3}`, as CUTLASS issues them), one row-high
  box each, packed one after another in shared memory.
- **Swizzle atoms.** Tensor maps accept `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B`
  and `_ATOM_64B` ("swizzle 32B/64B chunks within 128B span"), and
  `tensormap.replace` their atomicity field. The `_FLIP_8B` variant is refused.

## Refused by name

`tcgen05.mma.sp` (sparse A), `.ws` (weight-stationary), block-scaled kinds
(`.kind::mxf8f6f4`, `mxf4`, `mxf4nvf4`), the 4- and 6-bit types of
`.kind::f8f6f4`, `.ashift`, `tcgen05.cp`, `tcgen05.shift`, `tcgen05.ld.red`
(sm_103/sm_110), the sm_107 additions (`kind::ti16`, `decompress::lut`), and
TMA's `.im2col::w` modes: the ISA shows their halo walk only in figures that
leave open where `::w::128`'s halos come from and whether a halo crosses into
the next image, and nothing to check against (CUTLASS included) uses them.

## How it is checked

- `tests/unit/test_blackwell.cpp`: the ld/st fragments against figures 186-190
  written as tables; the MMA against a host GEMM with operands laid out from
  the ISA's CuTe canonical layouts, for each kind, both majors, the swizzles,
  A from Tensor Memory, the accumulate/scale/negate/mask options, and both
  pair layouts.
- `nvidia/tests/e2e/run_cutlass_sm100.sh`: CUTLASS's own SM100 GEMM unit tests,
  unmodified, on a simulated B200, checked against CUTLASS's host reference.
