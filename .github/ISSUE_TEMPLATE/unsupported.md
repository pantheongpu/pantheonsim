---
name: Something is not supported
about: The simulator stopped with an "unsupported" error, or a library, API or instruction you need is missing
title: ''
labels: enhancement
---

**The message**

Paste the error exactly. The simulator names what is missing (instruction, PTX line, API call, kernel, profile).

```
```

**What you were running**

The program, library or framework, and its version (for example PyTorch 2.x for CUDA 13, a CUTLASS test, an application).

**How to reproduce**

```bash
```

**Environment**

| | |
| --- | --- |
| PantheonSim | `vgpu --version`, or the commit |
| Simulated GPU | the id from `vgpu list-gpus` |
| Toolchain | CUDA or ROCm version |

**Is it listed in `docs/status.md` or `TODO.md` as a known gap?** Link it if so. Several gaps are deliberate (no timing model, no statically linked CUDA runtime, no Nsight or OptiX), and the rest are worked through in the order the people who ask for them need them.
