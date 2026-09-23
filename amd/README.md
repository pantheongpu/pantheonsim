# AMD

Support for AMD GPUs and ROCm, which is at the start.

**Discovery** works: a simulated AMD machine answers `rocm-smi`, `amd-smi` and
`rocm_agent_enumerator` from its profile, as the real tools would, and its
registers, RAS counts, CPER records and amdgpu sysfs files are modelled
(`docs/registers.md`, `docs/telemetry.md`).

**Execution** is being built, and a program cannot run on an AMD device yet.
The front of it is here: a HIP program hands the driver a code object, so
`src/codeobject.cpp` reads one (the ELF, each kernel's descriptor and the
metadata note: kernels, kernarg layout, LDS, registers) and `src/gcn_decode.cpp`
decodes the CDNA instructions in it. The decoder is checked instruction by
instruction against what the assembler wrote, on the object in `tests/data/`
and on one built during the test run (`tests/e2e/run_gcn_disasm.sh`). What is
missing is the wavefront itself -- scalar registers and the EXEC mask -- and
then the HIP runtime.

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
