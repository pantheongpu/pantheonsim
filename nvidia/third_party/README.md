# Vendored headers

`nccl_include/` is the public API headers of NCCL (Apache-2.0, as its headers
say) and `nvenc_include/nvEncodeAPI.h` is the public NVENC API header from
NVIDIA's Video Codec SDK, which NVIDIA licenses under the MIT terms printed at
the top of the file; `nvdec_include/` is the public NVDECODE API (`cuviddec.h`, `nvcuvid.h`)
from the same SDK under the same MIT terms, copied from a published MIT-licensed
redistribution of it. They are here so the shims are compiled against the *real*
ABI rather than a hand-written approximation -- writing struct layouts by hand
has silently produced garbage twice in this project (`cudaDeviceProp`,
`cudaFuncAttributes`) -- and so `libvgpunvenc` (presented as
`libnvidia-encode.so.1`) builds from this tree alone.

They are build-time only: nothing here is redistributed in a binary, and the
shims implement the documented APIs, not any NVIDIA code.

Refresh NCCL with:

    pip download --no-deps --dest /tmp/nvpkg nvidia-nccl-cu12

and NVENC from the SDK's `Interface/` directory.

## cuDNN is not here

cuDNN's API headers carry NVIDIA's proprietary licence notice, which does not
allow publishing them, so they are not kept in this repository. The same headers
are built against all the same: the CMake configure fetches them into
`build/cudnn_include` from the `nvidia-cudnn-cu12` wheel on PyPI with
`scripts/fetch-cudnn-headers.py` (about 1 MB, by range requests, each file checked
against `scripts/cudnn-headers.sha256`). To use headers you already have, set
`CUDNN_HEADER_DIR`; to build without cuDNN, configure with
`-DVGPU_FETCH_CUDNN_HEADERS=OFF`. `tests/lint/check_no_proprietary_nvidia.sh` fails
if a file with that notice is ever added back.

## METIS is here, in part

`metis/` is METIS 5.1.0 (Apache-2.0, the same licence as this repository; the
text is in `metis/LICENSE.txt`), cut down to what `METIS_NodeND` reaches and
changed in three marked places -- see `metis/README.md`. It is compiled into
`libcusolver`'s shim, with hidden symbols, because NVIDIA documents
`cusolverSpXcsrmetisndHost` as a wrapper of that function and its
permutation is METIS's. It is vendored rather than fetched at build time so a
build needs no network.
