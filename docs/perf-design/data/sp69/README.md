# Round sp69: the driver start of a program of many short launches (data)

The text is *Round sp69* in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md). server3 (GCC 13.3, `Release`, LTO), every
build under the machine's build lock, `nice -n 19`, `-j3`. server3 has no card and no CUDA toolkit: for the stand-in test the compiler part of
the dev machine's CUDA 13.0 toolkit (nvcc, the headers, `libcudart`, `libcudadevrt`) was copied into the round's scratch directory, named with
`VGPU_NVCC`, and deleted with it. The corpus programs are the sm_86 binaries of `tools/bench/real/pack_corpora.sh`.

| file | what |
|---|---|
| `card.md` | the A10G run: hotspot, sc_gpu and the four Samples in six forms; the other programs of 200 launches or more; the integration's keep list |
| `start-who.md` | which launch starts the driver under the new rule, at the default start cost and at 0.08 s, with `VGPU_HOST_GPU_START_REPEATS=0` beside it; the launches per kernel of the programs that matter |

## Tests on server3

```
./build/test_host_gpu_logic                                   # 109/109 tests passed
VGPU_NVCC=$PWD/cuda/bin/nvcc VGPU_BUILD_DIR=$PWD/build bash src/nvidia/tests/e2e/run_host_gpu_stub.sh
```

The stand-in test, steps 1 to 37:

```
PASS: first launch 11 ms against a driver start of 2000 ms; driver calls in the big run:  1 cuDeviceGet; 1 cuDevicePrimaryCtxRetain; 1 cuInit; 1 cuMemAddressReserve; 2 cuModuleLoadData; 2 relocs_to_variables 0;; many launches of 2227 us started the driver after 200 of them
```

With the rule taken out (in `start_share_launches`, the line that returns the kernel's count replaced by nothing, so the share is always the
start over 50), same commands:

```
[ FAIL ] the_start_is_repaid_within_fifty_launches_until_a_kernel_has_run_two_hundred_times
tests/unit/test_host_gpu_logic.cpp:791: VCHECK(start_share_launches(c, 200) == 200) failed
[ FAIL ] many_short_launches_of_one_kernel_buy_the_start_when_as_many_again_repay_it
tests/unit/test_host_gpu_logic.cpp:822: VCHECK(should_offload_with_start(hs, c, false, 0.30, 268)) failed
107/109 tests passed

FAIL: 400 launches of one kernel, of 2383 us each, did not start a driver that costs 0.238300 s: ; 398 launches stayed
FAIL: many started the driver after ? launches of its kernel, not after 200 to 399:
```

## The card run (one A10G; results in `card.md`)

Build as in `../summary-2026-11-13/README.md` (targets `vgpucudart`, `vgpucuda`), corpora from `tools/bench/real/pack_corpora.sh` in `~/corp`.
`turbor0` and `exactr0` are the rule before the round (`VGPU_HOST_GPU_START_REPEATS=0`), `turbol0` and `exactl0` the account alone
(`VGPU_HOST_GPU_START_LAUNCHES=0`), interleaved with the new rule by the harness.

```
R="python3 tools/bench/hostgpu/run.py --build build --corp $HOME/corp --gpu nvidia/a10g --timeout 600"
# 1. hotspot, sc_gpu and the four Samples the share protects (10 runs)
$R --corpora rodinia,samples --programs hotspot,sc_gpu,LargeKernelParameter,threadFenceReduction,dct8x8,MC_EstimatePiInlineP \
   --forms sim,exact,exactr0,turbo,turbor0,turbol0 --runs 10 --json four.json --md four.md
# 2. the other programs with 200 launches or more that do not start the driver today, and asyncAllocation (5 runs)
$R --corpora polybench,rodinia,hecbench,samples --programs 3DCONV,ADI,JACOBI1D,needle,cudaGraphsPerfScaling,asyncAllocation-cuda \
   --forms sim,exact,exactr0,turbo,turbor0 --runs 5 --json many.json --md many.md
# 3. the keep list of the integration (3 runs; programs over 20 s one)
$R --corpora polybench,rodinia,hecbench --programs 3MM,FDTD-2D,LU,SYR2K,SYRK,JACOBI2D,gaussian,lud,particlefilter_float,atomicAggregate-cuda,atomicPerf-cuda,black-scholes-cuda \
   --forms sim,exact,exactr0,turbo,turbor0 --runs 3 --long-secs 20 --json keep.json --md keep.md
# 4. card CI
tools/card-ci/run-card-ci.sh
```

What to read: hotspot turbo against turbor0 (1.00x before) and turbol0 (the 2.27x of Round sp66); the four Samples at the simulator's time in
every form but `turbol0`/`exactl0`, with 0 launches on the card; no program of 2 and 3 more than 3 % slower than its `r0` form; exact outputs
identical to the simulator's.
