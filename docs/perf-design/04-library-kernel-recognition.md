# Study 4: recognising library kernels (GEMM, reductions, copies) and running them natively

**Question.** Kernels emitted by cuBLAS, CUTLASS or PyTorch follow known patterns. Could the simulator recognise
one and execute it natively, with identical results?

**Recommendation.** **DO NOT** for GEMM and for floating-point reductions. **DO NOT NOW** for copy/fill patterns
(the gain is available more safely by speeding up the generic memory path, study 1); revisit only if tier 1-2
leave a measured gap on a specific, frequently launched kernel. The project already draws the useful boundary
elsewhere, and the "identical results" requirement is what rules this out.

## 1. The premise is partly already true, one level up

`nvidia/src/cublas_api.cpp:3-14` states the design: "WHY THIS IS NOT INTERPRETED ... cuBLAS is a library, not user code
... these entry points read their operands out of virtual device memory, do the arithmetic on the host CPU at
native speed (~10^10 FLOP/s), and write the result back ... application kernels are simulated, vendor library calls
are implemented." The same holds for cuDNN, cuFFT, cuSOLVER, cuSPARSE, NCCL, NPP, the codecs and the rest
(nvidia/docs/libraries.md). So **no cuBLAS-emitted SASS is ever interpreted**: there is nothing for pattern
recognition to find in a program that calls cuBLAS.

What does reach the interpreter are matrix kernels that live in *user code*: CUTLASS compiled into PyTorch
extensions and flash-attention-style kernels, Triton output, hand-written CUDA (the pantheon `tensor_virus`,
`mma_virus`, `fused_attention`, `transformer_*` workloads, TensorRT-style plugins, llama.cpp's ggml kernels). Those
are the only candidates.

## 2. Why the identical-results requirement is the obstacle

* The simulator's HMMA/IMMA/QMMA semantics are defined per instruction, several of them measured bit for bit on
  cards (`exec::mma_narrow_sum` in include/vgpu/exec/numerics.hpp documents an exact-sum-rounded-once toward
  zero model measured on an RTX PRO 6000; the f16 paths on the RTX 3060). A kernel's result is the composition of
  those per-instruction roundings in the **order the kernel issues them**: the K-loop tile order, any split-K
  or stream-K, the accumulator layout, the epilogue (beta scaling, bias, activation, cast order). A native
  replacement produces the same bits only if it reproduces that order exactly; a faster algebraically-equal
  order (blocked, FMA-contracted by the host compiler) gives different bits in f32 and f16.
* "Recognised" must mean *proved equal* for this module's actual SASS, not "looks like CUTLASS". Template
  parameters (tile shape, stages, swizzle, epilogue visitor) change the order, and the same source compiled by
  CUDA 12.0 and 13.0 gives different SASS. A fingerprint of the instruction stream would need one entry per
  (template instantiation, toolkit version, architecture): the open-ended maintenance that route (c) of study 1
  avoids, with silent wrong answers on a miss (ARCHITECTURE.md D5: "Silent wrong answers are the one unforgivable
  bug class").
* Race-free reductions through atomics or tree reductions have order-dependent float results. With the default
  multi-threaded grid the *current* engine already gives any of several valid answers for float atomics
  (launch.hpp:288-294); a native version would pick one, a behaviour change.
* Evidence a recogniser needs before it ships: for each pattern, the card oracle (RTX 3060 and an sm_90/sm_100 card
  for those architectures, via AWS under the round-3 rules) comparing the native result against both NVIDIA's
  kernel and the interpreted one over arithmetic-breaking inputs (zeros, signed zeros, subnormals, NaN payloads,
  infinities, extremes), every dtype pair, transposes, odd shapes, alpha/beta special cases; plus a run-time
  verification mode (execute both for every Nth launch and compare, aborting loudly).

## 3. What it would buy, after study 1

Prototype figure (study 1, `ffma_loop`, 8 independent FFMA chains): generated AVX2 code runs at **22.6 GFLOP/s on
one core**, against 0.63 GFLOP/s for the interpreter today. A host GEMM at the library boundary is quoted at
~1e10 FLOP/s (cublas_api.cpp:12) and a tuned single-core AVX2 GEMM would be several times that (NOT MEASURED;
nominal AVX2 FMA peak of this part, about 150 GFLOP/s, is arithmetic, not measurement). So after tiers 1-2 the
remaining gap to a native GEMM on a compute-bound tile loop is plausibly 5x or less for FFMA kernels; for tensor-core
kernels it is unmeasured (HMMA is not in the prototype and HMMA is 1.7% of the pantheon corpus' static
instructions and 31% of mma_virus's loop-body instructions: 320 of 1,032). That is the gap the recogniser would close, at the correctness price
above. Tier 1-2 apply to every kernel; a recogniser applies to the ones someone wrote a recogniser for.

## 4. Copies and fills

These are the safe cases (no rounding; the result is the bytes), and memory-bound kernels are where the prototype
saw its smallest gains (about 3x) until the bulk memory calls were used (12-21x, study 1 table). The right fix is in
the engine: contiguous-warp loads/stores through `load_run`/`store_run` with the fault hook handled, and the cheap
exact `count_access`. That gives the speed-up for *every* kernel that copies, with no fingerprinting. A
fingerprint-based `memcpy` replacement would also have to reproduce the counters (global sectors, requests, bytes)
exactly, which the generic path does from the addresses anyway.

## 5. If it is ever done

Per-pattern allow-list, default off (`VGPU_NATIVE_KERNELS=gemm_f32,...`), matched on the **relocated instruction
words** (hash) plus the architecture, never on names; always accompanied by a card-oracle test in the CI-wired set
and a check mode. No pattern before the card evidence exists for it. NOT MEASURED: any actual native replacement; this
study is a code reading plus the prototype's FLOP rates.
