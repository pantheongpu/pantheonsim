# LDS bank-conflict probe

What `amd/tests/data/lds_bank_conflicts.tsv` was recorded with: one wave of N lanes issuing one LDS instruction 1000 times, lane `L` at LDS
byte `(L * stride) % 65536`, while the card's own `SQC_LDS_BANK_CONFLICT` counter is read. The model it checks is in
`amd/include/vgpu/amd_memtraffic.hpp`.

- `probe.cpp`: `ds_read_b32/b64/b128` and `ds_write_b32/b64/b128` (inline assembly, so the instruction is exactly that one).
  `./probe <stride bytes> <width bytes> <write 0|1> <threads>`.
- `probe2.cpp`: the other forms (`ds_read_u8`, `ds_read_b96`, atomics, `ds_read2`, `ds_write2`). `./probe2 <stride> <form> <threads>`;
  the form numbers are in the file.
- `sweep.sh`, `sweep2.sh`: run a probe over the strides under `rocprofv2` and print `conflicts idx_active insts` for each. They were
  written for the machine the cards were in (paths under `/tmp/ldsprobe-saqib`, a `ROCR_VISIBLE_DEVICES` mask): read them as a recipe.

Build: `hipcc --offload-arch=gfx1030 -O2 -o probe probe.cpp` (add `-mwavefrontsize64` for a wave64 program).

Three things to know before running it on a card:

1. **`rocprofv3 --pmc` aborts** on a machine with an iGPU in the agent list (`std::out_of_range`, then a hang), as
   `amd/docs/rocprofiler.md` says. `rocprofv2 -i <file with "pmc: SQC_LDS_BANK_CONFLICT SQ_INSTS_LDS">` works. The counter is per dispatch;
   the 512 stores that initialise LDS add no conflicts.
2. **`ds_read_b96` (form 12 in `probe2`) under `rocprofv2` on an RX 6700 XT (gfx1031) took a page fault and wedged the card**
   (`amdgpu: [gfxhub] page fault`, then `SMU: No response`), which stopped every HIP program on that machine until it was reset. The same
   instruction ran fine on the RX 6800 (gfx1030). Do not run form 12 on gfx1031; the rows in the data file for it come from the RX 6800 only.
3. Only these were measured: RDNA2 (gfx1030, gfx1031), wave32 and wave64. CDNA (MI200 to MI350) and RDNA3 and later were not,
   so the model is applied to them unchecked.
