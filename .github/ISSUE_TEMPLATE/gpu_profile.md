---
name: A GPU profile is wrong or missing
about: A simulated GPU reports different values from the real one, or a card you use has no profile
title: ''
---

**Which profile**

The id from `vgpu list-gpus`, or the card that has none.

**What differs**

What `vgpu info --gpu <id>`, `nvidia-smi`, `rocm-smi` or a CUDA/HIP device query shows on the simulator, and what the real card shows. Paste both.

**Where the real values come from**

Output from the physical card is what makes a profile `verified: true`. Documents are fine too, but say which. The tools that read a card are in `nvidia/tools` and `amd/tools`; if you can run one, attach its output.

**Card and driver**

| | |
| --- | --- |
| GPU | model and memory size |
| Driver / ROCm | version |
| Host | OS |
