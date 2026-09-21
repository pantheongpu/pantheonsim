# AMD

Support for AMD GPUs and ROCm, which is at the start.

What exists today is **discovery**: a simulated AMD machine answers `rocm-smi`
and `rocm_agent_enumerator` from its profile, as the real tools would. Programs
do not run on AMD devices yet -- there is no HIP runtime or ROCm library layer.

| Folder | What |
| --- | --- |
| `profiles/` | MI300X, MI325X and MI350X. MI325X was read from a physical card with `tools/rocminfo-to-profile.py`; the others are placeholders, and each file's header says which |
| `tools/` | `rocm-smi`, `amd-smi` and `rocm_agent_enumerator` for simulated machines, and the scripts that characterize a card (`characterize-hip.cpp`, the DigitalOcean scripts) |
| `src/` | The HIP runtime and ROCm libraries, once they are written |

Execution follows once there is a card to measure against: the HIP runtime
first, then the libraries, each checked against the real one on hardware the
way the NVIDIA side is.

A profile id is `amd/<name>`, and its file is `profiles/<name>.yaml` here.
