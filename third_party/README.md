# Vendored headers

`cudnn_include/` and `nccl_include/` are the public API headers from the
redistributable `nvidia-cudnn-cu12` and `nvidia-nccl-cu12` wheels. They are
here so the shims are compiled against the *real* ABI rather than a
hand-written approximation — writing struct layouts by hand has silently
produced garbage twice in this project (`cudaDeviceProp`, `cudaFuncAttributes`),
and there is no reason to risk it a third time.

They are build-time only: nothing here is redistributed in a binary, and the
shims implement the documented APIs, not any NVIDIA code.

Refresh with:

    pip download --no-deps --dest /tmp/nvpkg nvidia-cudnn-cu12 nvidia-nccl-cu12
