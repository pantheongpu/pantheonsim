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

What is missing is the HIP runtime above it, so a HIP program still cannot
reach any of this on its own; the tests dispatch kernels directly. The
instructions implemented are those the fixture's kernels use, and any other is
refused by name rather than guessed.

| Folder | What |
| --- | --- |
| `profiles/` | MI300X, MI325X and MI350X. MI325X was read from a physical card with `tools/rocminfo-to-profile.py`; the others are placeholders, and each file's header says which |
| `registers/` | the MMIO database, each model's registers and their power-on values, and the amdgpu headers' licence |
| `src/` | the register model, the metrics table and CPER records; code objects and CDNA decoding; the HIP runtime and ROCm libraries once they are written |
| `tools/` | `rocm-smi`, `amd-smi` and `rocm_agent_enumerator` for simulated machines, and the scripts that characterize a card (`characterize-hip.cpp`, the DigitalOcean scripts) |
| `tests/` | the AMD unit and end-to-end tests, and the code object they read (`tests/data/build.sh` rebuilds it with clang; no ROCm needed) |

The libraries follow the runtime, each checked against the real one on
hardware the way the NVIDIA side is.

A profile id is `amd/<name>`, and its file is `profiles/<name>.yaml` here.
