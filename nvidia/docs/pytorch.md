# PyTorch on a simulated NVIDIA GPU

PyTorch's official CUDA build runs unmodified on a simulated
**GeForce RTX 5090** (`nvidia/rtx5090`, compute capability 12.0):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
vgpu run --gpu nvidia/rtx5090 --preload python train.py
```

`amd/tests/pytorch/models.py` passes all 12 of its whole-model checks there,
each against the CPU:
- a GPT trained with a one-cycle schedule, and generating with a KV cache;
- a ResNet, an LSTM, a U-Net denoiser and a vision transformer;
- mixture of experts and a DLRM-style recommender;
- half and bfloat16 autocast, activation checkpointing, and a checkpoint read
  back on the host.

The whole suite takes about 40 seconds and 2.4 GB (ctest
`e2e_pytorch_models_rtx5090` where a PyTorch for CUDA 13 is installed).

## Why the RTX 5090

The simulator runs PTX. PyTorch's CUDA 13 wheels carry machine code (SASS)
for sm_75 to sm_120, and PTX only for compute_120: all 456 fatbins in
`libtorch_cuda.so`. A driver compiles PTX only for a device at least as new as
the PTX, so the simulated GPU has to be an sm_120 part for PyTorch's own
kernels to run. On an older simulated GPU, PyTorch finds the device but each
kernel launch fails, naming the missing image.

## Why `--preload`

PyTorch reaches NVIDIA's libraries through an RPATH into its `nvidia-*`
packages, which `LD_LIBRARY_PATH` does not override, and `python` itself
names no CUDA library. `--preload` preloads every CUDA library the simulator
carries in that case, and a library already loaded under a soname is the one
every later request for that soname gets. Without it, NVIDIA's real runtime
loads, finds no NVIDIA driver, and PyTorch reports "integrity checks failed"
(error 103). NVIDIA's cuFile, cuSPARSELt and NVSHMEM load as they are; the
paths tested here do not call them.

## What PyTorch uses, and where it is

- **Its own kernels:** the PTX interpreter. A large module (2 MB or more of
  PTX) is parsed one kernel at a time, as each is first launched
  (`VGPU_LAZY_MODULE_BYTES` sets the threshold, `VGPU_PARSE_WHOLE=1` turns it
  off), and a registered fatbin's PTX is only extracted when first needed.
  Without both, a GPT training step took more than 6 GB.
- **Convolutions:** cuDNN's backend (graph) API (`nvidia/src/cudnn_backend.cpp`):
  forward, backward-data and backward-filter, 1-3 spatial dimensions, groups,
  dilation, any strides; float, half, bfloat16, double. Computed on the host.
- **BatchNorm:** cuDNN's training forward and backward (the Ex forms).
- **RNNs:** cuDNN's RNN API (`nvidia/src/cudnn_rnn.cpp`): LSTM, GRU and
  ReLU/tanh RNNs, one or two directions, padded or packed sequences.
- **Linear layers:** cuBLAS and cuBLASLt's fused matmul + bias; PyTorch passes
  its cuBLAS handle as the cuBLASLt handle, which is accepted.
- **Both CUDA APIs:** PyTorch calls the driver API too. `libcudart` and
  `libcuda` share one simulated machine and one API lock
  (`src/runtime/shared_runtime.cpp`), and the runtime makes its device's
  primary context current in the driver, as NVIDIA's runtime does.

## Not supported

- Dropout between the layers of a cuDNN RNN in training, LSTM projections,
  and non-float RNNs: refused by name.
- Graphs cuDNN's backend would run with fused operations (pointwise, norms,
  matmul inside a graph): refused when the graph is finalized, so PyTorch
  falls back or reports it.
- Anything a kernel does that the PTX interpreter does not implement: the
  launch fails naming the instruction, and the PyTorch test scripts fail on any
  such line, even if a check prints `ok`.

## Tests

- `e2e_dnn_paths`: cuDNN's graph API, BatchNorm, RNNs and cuBLASLt called
  directly, against host references and finite differences. Needs no PyTorch.
- `e2e_mixed_apis`: the runtime and driver APIs on one machine.
- `test_runtime`, `test_exec`: lazy parsing, and the 256-bit `.v8` accesses
  PyTorch's elementwise kernels use.
- `e2e_pytorch_models_rtx5090`: `models.py`, where PyTorch is installed.
