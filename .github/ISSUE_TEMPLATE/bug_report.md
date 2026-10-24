---
name: Bug report
about: A program gives a wrong result, crashes, or behaves differently from a real GPU
title: ''
labels: bug
---

**What happened**

**What you expected** (and how you know: a real GPU, NVIDIA's or AMD's documentation, the program's own check)

**How to reproduce**

The smallest program or command that shows it. Run it with `VGPU_THREADS=1` and say whether it still happens.

```bash
# for example
VGPU_GPU=nvidia/h100 vgpu run ./my_app
```

**Output**

Paste the full output, including any line that starts with `[vgpu]`. The error names the instruction, API or kernel; that text is the most useful part of the report.

```
```

**Environment**

| | |
| --- | --- |
| PantheonSim | `vgpu --version`, or the commit |
| Simulated GPU | the id from `vgpu list-gpus`, and how many |
| How you run it | `./run`, `vgpu run`, the container, the GitHub Action |
| Toolchain | CUDA or ROCm version, and the compiler that built your program |
| Host | OS, CPU, and whether it is WSL |
| Settings | `vgpu config show` |

**Does it happen on a real GPU?** If you can, say what the card printed.
