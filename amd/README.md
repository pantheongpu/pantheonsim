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

- `src/hip_api.cpp` is `libamdhip64`: the HIP runtime API, over those pieces.
  A HIP program links against it the way it links against AMD's, asks for
  devices and memory, loads a code object and launches the kernels in it
  (`tests/e2e/run_hip.sh` does exactly that, and checks the answers).

The interface is `include/vgpu_hip.h`, a clean-room subset of the documented
HIP API. What is implemented is devices, memory and the module API; the
kernel-launch syntax `hipcc` compiles (`__hipRegisterFatBinary` and
`hipLaunchKernel`) is not there yet, so a program uses the module API.

The instructions implemented are those clang emits for the kernels in
`tests/data/`: scalar and vector integer arithmetic, the logical and shift
ops, 32- and 64-bit values, single and double precision (add, multiply, fma,
min, max and the sequence a division compiles to), packed half precision,
conversions, the transcendentals, bit counting, comparisons in both their
forms and the class test, `v_cndmask`, the lane-counting ops, scalar and EXEC
branches, and the memory a kernel uses: global loads and stores and their
atomics, LDS and its atomics, a work-item's private memory (what a kernel
spills into when it runs out of registers), a value read from another lane,
and a flat access, whose address says for itself whether it means LDS or the
device. The source modifiers are applied -- an absolute value, a negation, a
clamp of the result.

Any other instruction is refused by name, and so is anything this does not
model: an output multiplier, a packed operation that shuffles halves. A wrong
guess would run and give a wrong answer, which is worse than a refusal.

Three things are modelled rather than copied, and are marked where they are
written: the reciprocal, square root, exponent and logarithm are the host's
exact results where the hardware's are tables good to about one unit in the
last place; the scope bits on a memory instruction change nothing, since every
access here is already visible to every wave; and LDS sits at an address of
this model's choosing, which a kernel reads from `src_shared_base` the way it
reads the hardware's.

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
