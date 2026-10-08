# Vendored headers

`nccl_include/` holds the public API headers from NVIDIA's NCCL, which is open
source (Apache-2.0 / BSD-3-Clause, per the SPDX tags in each file; the license
text is in `nccl_include/LICENSE.txt`). They are here so the shims are compiled
against the *real* ABI rather than a hand-written approximation -- writing struct
layouts by hand has silently produced garbage twice in this project
(`cudaDeviceProp`, `cudaFuncAttributes`). They are build-time only, and the shims
implement the documented APIs, not any NVIDIA code.

The cuDNN headers are **not** in this tree. They are NVIDIA's, under the NVIDIA
SDK license shipped in the `nvidia-cudnn-cu12` wheel, which does not allow
redistributing them as source. `fetch-cudnn-headers.sh` downloads them from that
wheel into `cudnn_include/` (git-ignored); CMake runs it automatically when
`libvgpucudnn` is configured and the headers are missing. To use headers you
already have, set `CUDNN_HEADER_DIR`; to skip the download, configure with
`-DVGPU_FETCH_CUDNN_HEADERS=OFF` (then `libvgpucudnn` is not built). The wheel is
several hundred MB; only the headers are kept.

`nvenc_include/nvEncodeAPI.h` is the public NVENC API header from NVIDIA's
Video Codec SDK, which NVIDIA licenses under the MIT terms printed at the top of
the file. It lets `libvgpunvenc` (presented as `libnvidia-encode.so.1`) build
from this tree alone. Refresh it from the SDK's `Interface/` directory.

`joe_kuo/sobol_directions.inc` packs S. Joe and F. Y. Kuo's Sobol' direction
numbers (`new-joe-kuo-6.21201`, the first 20,000 dimensions), which cuRAND
documents as the source of its direction vectors. Unlike the headers above it
is compiled into `libvgpucurand`, under the BSD-style licence reproduced at the
top of the file. `joe_kuo/gen_sobol_directions.py` regenerates it from the file
published at https://web.maths.unsw.edu.au/~fkuo/sobol/.
