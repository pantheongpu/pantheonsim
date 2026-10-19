# Round 10, step 0: where the time goes on real workloads

Measured 2026-10-18 on a dedicated AWS c7i.4xlarge (Xeon Platinum 8488C, 16 vCPU, AVX-512, 30 GB, Ubuntu 24.04), nothing else running,
`origin/main` 89d6bcd5 built Release in the CI toolchain image (CUDA 13.0), PyTorch 2.14.1+cu130. Run-to-run spread on the synthetic
workloads was 0.2 to 1.4% (the shared WSL machine of the earlier rounds had 3 to 60%). Raw files are in `data/round10/`.

## 1. The synthetic workloads, tier off and on (`bench.sh --runs 5 --cpus 2-5`, SASS engine, `VGPU_THREADS=1`)

Warp instructions per device-CPU second, in millions; "x" is tier on over tier off.

| workload | tier off | tier on | x |
| --- | ---: | ---: | ---: |
| int_alu | 37.65 | 99.90 | 2.65 |
| f32_fma | 41.74 | 109.41 | 2.62 |
| f64_fma | 29.63 | 46.41 | 1.57 |
| f64_subnormal | 5.58 | 5.96 | 1.07 |
| wmma | 1.26 | 1.40 | 1.11 |
| mma_ptx | 2.44 | 2.72 | 1.12 |
| mem | 11.86 | 16.76 | 1.41 |
| shared | 13.35 | 19.71 | 1.48 |
| atomics | 6.01 | 7.87 | 1.31 |
| texture | 3.19 | 3.41 | 1.07 |
| local | 19.00 | 37.26 | 1.96 |
| divergent | 16.28 | 30.86 | 1.89 |
| warp | 19.35 | 64.59 | 3.34 |

These agree with the numbers in docs/performance.md for the tier, with one difference: `wmma` and `mma_ptx` gain 1.1x here, not 1.4 to 1.6x.

## 2. The PyTorch sweep (quick tier, 169 checks, 13 processes, `nvidia/rtx5090`)

| | wall |
| --- | ---: |
| `VGPU_TIER=0` | 100.6 s |
| `VGPU_TIER=1` | 101.6 s |

**The tier changes nothing here.** Every group has the same ok / known-failure counts in both runs (the `compile` group fails in both: this rig has
no Inductor toolchain set up, which is not a simulator result). With `VGPU_TIER_STATS=1` the whole sweep executes 51.1 M warp
instructions, 88.9% of them in the tier. At the 20 to 100 M/s of section 1 that is under 5 seconds of the 100.

## 3. Where a PyTorch process spends its time

A process that does `import torch`, creates a tensor and runs four small elementwise kernels takes 2.2 s: `import torch` 0.94 s, the first
CUDA call 0.06 s, a 256x256 matmul 0.02 s (host-computed cuBLAS), and the four kernels 0.59 s.

The first launch of each distinct kernel in a process costs 0.05 to 0.2 s, and later launches cost about nothing (`data/round10/mini2.py`):

| step | rtx5090 | rtx3060 |
| --- | ---: | ---: |
| add (first elementwise kernel) | 0.145 s | 0.128 s |
| add again (same kernel) | 0.000 s | 0.001 s |
| mul (new kernel, same file) | 0.091 s | 0.081 s |
| relu (new kernel) | 0.124 s | 0.103 s |
| sum reduction (new kernel) | 0.217 s | 0.172 s |
| softmax (new kernel) | 0.161 s | 0.140 s |
| layer_norm (new kernel) | 0.055 s | 0.046 s |
| layer_norm again | 0.003 s | 0.003 s |

`perf` on the `nn` sweep group (9 s wall, BLAS threads off so idle spinning does not hide it): the simulator library is 29% of the samples,
libc 40% (mostly `malloc`/`free`), zstd 5%. Under the library the inclusive time is in SASS decoding: `vgpu::sass::decode`,
`std::vector<Operand>::emplace_back` (6.9% inclusive), the decoder lookup `std::_Hashtable<unsigned, void(*)(Instr&, const Word&)>` (3.3%), and
text formatting at decode time (`float_text` 2.3% inclusive, `signed_hex`). zstd is the fatbin decompression. Together with the allocator work
they drive, loading and decoding kernels is roughly a third of that group.

Not measured here: CUTLASS GEMM/FMHA run time, the pantheon stress workloads at full size, other architectures' profiles.
