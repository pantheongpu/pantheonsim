# Which launch starts the driver under the rule of Round sp69 (server3, stand-in driver)

`../sp64-sp68/startwho.py --stop` on server3 against this round's build: `VGPU_THREADS=3`, profile `nvidia/a10g`, the stand-in driver
(`STUB_INIT_MS=150`), which refuses every module, so every launch runs in the simulator. "sim" is the estimate of the launch that started the
driver, "spent" the account, "seen" the launches of that kernel the simulator had run. server3's simulator runs these programs 2 to 3 times as
fast as the A10G rig's; `VGPU_HOST_GPU_START_COST=0.08` stands in for the rig, as in the integration of sp64 to sp68.

## Default start cost (0.25 s)

| program | mode | rule | kernel | sim ms | spent s | seen |
|---|---|---|---|---:|---:|---:|
| rodinia/hotspot | exact, turbo | never started (0.39 ms a launch: 0.29 ms saved, under the 0.5 ms floor) | | | | |
| rodinia/sc_gpu | exact, turbo | never started (0.21 ms a launch) | | | | |
| rodinia/needle | exact, turbo | never started | | | | |
| rodinia/gaussian | turbo | size rule | Fan2 | 5.569 | 0.254 | 41 |
| rodinia/gaussian | exact, runs 1 and 4 | size rule | Fan2 | 3.039, 2.887 | 0.835, 0.818 | 200 |
| rodinia/gaussian | exact, runs 2 and 3 | size rule | Fan2 | 5.567, 5.527 | 0.254 | 43, 42 |
| rodinia/lud | exact | size rule | lud_internal | 11.165 | 0.262 | 16 |
| rodinia/lud | turbo | size rule | lud_internal | 10.445 | 0.259 | 15 |
| hecbench/asyncAllocation-cuda | exact | never started | | | | |
| hecbench/asyncAllocation-cuda | turbo | size rule | process | 1.924 | 0.622 | 412 |
| samples/dct8x8 | exact, turbo | never started | | | | |
| samples/threadFenceReduction | exact, turbo | never started | | | | |
| samples/MC_EstimatePiInlineP | exact, turbo | never started | | | | |
| samples/LargeKernelParameter | exact, turbo | never started | | | | |

With `VGPU_HOST_GPU_START_REPEATS=0` (the rule before): asyncAllocation turbo never started; gaussian exact started in two runs of three
(`Fan2`, 5.6 ms, 41 and 43 seen) and never in the third: its launch at the moment the account fills is at the 5 ms bar on this machine.

## `VGPU_HOST_GPU_START_COST=0.08`

| program | mode | rule | kernel | sim ms | spent s | seen |
|---|---|---|---|---:|---:|---:|
| rodinia/hotspot | exact | never started | | | | |
| rodinia/hotspot | turbo | size rule | calculate_temp | 0.394 | 0.122 | 283 |
| rodinia/hotspot | turbo, `START_REPEATS=0` | never started | | | | |
| rodinia/sc_gpu | exact, turbo | never started | | | | |
| rodinia/needle | exact, turbo | never started | | | | |
| rodinia/gaussian | exact | size rule | Fan2 | 5.204 | 0.081 | 14 |
| rodinia/gaussian | turbo | size rule | Fan2 | 5.621 | 0.081 | 13 |
| rodinia/lud | exact | size rule | lud_internal | 16.751 | 0.097 | 5 |
| rodinia/lud | turbo | size rule | lud_internal | 17.294 | 0.090 | 4 |
| hecbench/asyncAllocation-cuda | turbo | size rule | process | 1.933 | 0.612 | 412 |
| hecbench/asyncAllocation-cuda | turbo, `START_REPEATS=0` | never started | | | | |
| samples/dct8x8 | exact | never started | | | | |
| samples/dct8x8 | turbo | first-launch rule (as before the round: `data/sp64-sp68/start-who.md`) | CUDAkernelQuantizationShort | 8.520 static | 0.129 | 0 |
| samples/threadFenceReduction | exact, turbo | never started | | | | |
| samples/MC_EstimatePiInlineP | exact, turbo | never started | | | | |
| samples/LargeKernelParameter | exact, turbo | never started | | | | |

## Launches per kernel (`host_gpu_kept`, turbo, the build before the round)

| program | kernel | launches | sim s |
|---|---|---:|---:|
| rodinia/hotspot | calculate_temp | 1000 | 0.426 |
| rodinia/sc_gpu | kernel_compute_cost | 1432 | 0.305 |
| hecbench/asyncAllocation-cuda | process | 800 | 1.221 |
| samples/threadFenceReduction | reduceSinglePass | 100 | 0.132 |
| samples/threadFenceReduction | reduceMultiPass | 1 | 0.002 |
| samples/dct8x8 | CUDAkernel2DCT | 101 | 0.151 |
| samples/dct8x8 | CUDAkernel1DCT (class never) | 10 | 0.227 |
| samples/dct8x8 | six others | 1 or 2 each | 0.047 |
| samples/LargeKernelParameter | kernelDefault | 1001 | 0.147 |
| samples/LargeKernelParameter | kernelLargeParam | 1001 | 0.101 |
