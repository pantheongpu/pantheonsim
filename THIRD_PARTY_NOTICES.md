# Third-party notices

PantheonSim is Apache-2.0 (see `LICENSE`). It includes or fetches the following
third-party material.

| Component | Where | License |
|---|---|---|
| NVIDIA NCCL public headers | `nvidia/third_party/nccl_include/` | Apache-2.0 / BSD-3-Clause (SPDX tags in each file; text in `nccl_include/LICENSE.txt`). Some files under `nccl_device/` are Amazon.com, Inc. (GPL-2.0 OR BSD-2-Clause OR Apache-2.0) |
| NVIDIA Video Codec SDK `nvEncodeAPI.h` | `nvidia/third_party/nvenc_include/` | MIT (text at the top of the file) |
| Joe-Kuo Sobol' direction numbers | `nvidia/third_party/joe_kuo/` | BSD-style (text at the top of the file) |
| AMD GPU register headers | `amd/registers/` | see `amd/registers/LICENSES/` |
| NVIDIA open-gpu-kernel-modules register headers | `nvidia/registers/` | see `nvidia/registers/LICENSES/` |
| NVIDIA cuDNN headers | **not included**; downloaded at build time by `scripts/fetch-cudnn-headers.py` from the `nvidia-cudnn-cu12` wheel into the build directory, each file checked against `scripts/cudnn-headers.sha256` | NVIDIA SDK license (in the wheel); not redistributed |
| METIS 5.1.0 (graph ordering, used by `cusolverSpXcsrmetisndHost`) | `nvidia/third_party/metis/` | Apache-2.0 (`LICENSE.txt` there); Copyright 1995-2013 Regents of the University of Minnesota |
