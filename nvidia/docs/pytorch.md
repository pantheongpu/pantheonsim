# PyTorch on a simulated NVIDIA GPU

PyTorch's official CUDA build runs unmodified on the simulated NVIDIA GPUs,
from the T4 to the B200 and the RTX 5090:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
vgpu run --gpu nvidia/h100 --preload python train.py
```

`amd/tests/pytorch/models.py` passes all 12 of its whole-model checks on each
of them (checked on the T4, A100, L4, RTX 3060, H100, B200 and RTX 5090), each
against the CPU:
- a GPT trained with a one-cycle schedule, and generating with a KV cache;
- a ResNet, an LSTM, a U-Net denoiser and a vision transformer;
- mixture of experts and a DLRM-style recommender;
- half and bfloat16 autocast, activation checkpointing, and a checkpoint read
  back on the host.

The whole suite takes about 20 seconds on the RTX 5090 (ctest
`e2e_pytorch_models_<gpu>` where a PyTorch for CUDA 13 is installed).

## Which code runs

PyTorch's CUDA 13 wheels carry machine code (SASS) for sm_75, 80, 86, 90, 100
and 120, and PTX only for compute_120. Its kernels run as the SASS image the
simulated GPU's architecture selects, as on the real driver
([sass.md](sass.md)); the PTX is only usable on an sm_120 part (a driver
compiles PTX for a device at least as new as it), where `VGPU_SASS=0` runs it
instead.

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

- **Its own kernels:** their SASS, each kernel's code decoded when it first
  runs. On the PTX path, a large module (2 MB or more of PTX) is parsed one
  kernel at a time, as each is first launched (`VGPU_LAZY_MODULE_BYTES` sets
  the threshold, `VGPU_PARSE_WHOLE=1` turns it off), and a registered
  fatbin's PTX is only extracted when first needed. Without the lazy steps, a
  GPT training step took more than 6 GB.
- **Convolutions:** cuDNN's backend (graph) API (`nvidia/src/cudnn_backend.cpp`):
  forward, backward-data and backward-filter, 1-3 spatial dimensions, groups,
  dilation, any strides; float, half, bfloat16, double; and the graphs that
  fuse a convolution with pointwise operations, such as conv + bias + ReLU.
  Computed on the host. The classic API (`nvidia/src/cudnn_api.cpp`) covers
  the same convolutions and every other layer's backward pass, for frameworks
  that call it.
- **Attention:** `scaled_dot_product_attention` with the cuDNN backend:
  PyTorch's bundled cudnn-frontend hands cuDNN's graph API a single SDPA
  operation forward (with its softmax statistics) and a composite graph
  backward, both computed on the host (`nvidia/src/cudnn_backend.cpp`).
- **BatchNorm:** cuDNN's training forward and backward (the Ex forms).
- **RNNs:** cuDNN's RNN API (`nvidia/src/cudnn_rnn.cpp`): LSTM, GRU and
  ReLU/tanh RNNs, one or two directions, padded or packed sequences, dropout
  between layers, LSTM projections (`proj_size`) and cell clipping; float,
  half (autocast), bfloat16 and double. Dropout between an RNN's layers draws
  the masks cuDNN draws, from the dropout descriptor's states, so a seed
  gives NVIDIA's masks (measured on an RTX 3060; padded batches of sequences
  shorter than the longest were not).
- **Linear layers:** cuBLAS and cuBLASLt's fused matmul + bias; PyTorch passes
  its cuBLAS handle as the cuBLASLt handle, which is accepted.
- **Both CUDA APIs:** PyTorch calls the driver API too. `libcudart` and
  `libcuda` share one simulated machine and one API lock
  (`src/runtime/shared_runtime.cpp`), and the runtime makes its device's
  primary context current in the driver, as NVIDIA's runtime does.

## Not supported

- cuDNN graph operations beyond what the graph API list in
  `nvidia/docs/libraries.md` computes -- convolution, matmul, pointwise,
  reduction, normalization (multi-GPU batch normalization among it), pooling,
  attention and the data-movement operations -- are refused when the graph
  is finalized, naming the operation; and the ones the card has no engine for
  (nearest and bilinear resampling, the MoE backward pass, a rotary embedding
  outside a matmul or attention graph, ...) finalize and then fail at plan
  creation with cudnn-frontend's "No valid engine configs", as they do on
  the RTX 3060 they were measured on, so PyTorch falls back or reports it.
  PyTorch itself emits convolutions (with bias, add and activation),
  matmul-based linear layers and scaled dot-product attention, all of which run.
- Anything a kernel does that the simulator does not implement: the launch
  fails naming the instruction, and the PyTorch test scripts fail on any such
  line, even if a check prints `ok`.

## Tests

- `e2e_dnn_paths`: cuDNN's graph API, BatchNorm, RNNs and cuBLASLt called
  directly, against host references and finite differences. Needs no PyTorch.
- `e2e_dnn_backward`, `e2e_dnn_graph`, `e2e_dnn_attention`: the classic
  API's backward passes, multi-operation graphs and attention as
  cudnn-frontend builds it, likewise (`nvidia/docs/libraries.md`).
- `e2e_dnn_classic_paths`: the rest of the classic API, LSTM projections and
  multi-head attention among it, likewise.
- `e2e_mixed_apis`: the runtime and driver APIs on one machine.
- `test_runtime`, `test_exec`: lazy parsing, and the 256-bit `.v8` accesses
  PyTorch's elementwise kernels use.
- `e2e_pytorch_models_<gpu>`: `models.py` on one GPU per SASS image (T4,
  A100, RTX 3060, H100, B200, RTX 5090), where PyTorch is installed.
- `e2e_pytorch_distributed`: `torch.distributed` across two processes, each its
  own simulated GPU, launched as torchrun launches them
  (`nvidia/tests/pytorch/distributed.py`). The collectives (all-reduce,
  all-gather, broadcast, reduce-scatter, all-to-all, send/recv), then a small
  network trained with `DistributedDataParallel`, tensor parallelism, pipeline
  parallelism and `FullyShardedDataParallel`, each ending where one process on
  the whole batch ends. PyTorch's NCCL backend runs on the simulator's
  `libnccl`, whose ranks meet through its file-backed transport. Where PyTorch
  is installed.
- `e2e_parallel_training_<n>` (two, three and four ranks): the same four ways
  of splitting a network, written directly against NCCL and cuBLAS
  (`nvidia/tests/e2e/parallel_training.cu`), one process per rank. Needs no
  PyTorch, so hosted CI runs it; the pipeline split is for two ranks.

## The sweep

`nvidia/tests/pytorch/sweep.py` runs many small checks (nn layers, losses, optimizers, graphs, `torch.compile`,
each SDPA backend, ...) on the simulated GPU and on the CPU from the same inputs. `ctest -R e2e_pytorch_sweep`
runs the quick tier through `nvidia/tests/e2e/run_pytorch_sweep.sh` (`e2e_pytorch_sweep_full` the rest; the
nightly workflow runs it). A check that does not match yet is listed with its reason in
`nvidia/tests/pytorch/sweep/known_failures.txt`: it prints `XFAIL` with its numbers and does not fail the run,
a listed check that starts passing prints `XPASS` and does, and a new failure fails. The gaps are in `TODO.md`
("PyTorch sweep: known failures"). Tolerances are never loosened to hide one.
