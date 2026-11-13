# Round sp59: turbo host-GPU mode on an A10G (data)

The text is *Round sp59* in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md).

## Rig

AWS g5.xlarge, us-east-1a, instance i-018af809900b2ab58, AMI ami-05414055cdd6e338f (Ubuntu 24.04.5, kernel 7.0.0-1014-aws), NVIDIA A10G 23028 MiB,
driver 595.112.07, 4 vCPU AMD EPYC 7R32, 15 GiB, g++ 13.3.0, cmake 3.28.3, Python 3.12.3. Toolkit: nvcc 13.0 (`/usr/local/cuda-13.0`, PATH forced; the
image's default is 13.2). Simulator: branch `hostgpu-turbo`, `RelWithDebInfo`, `-DVGPU_LTO=OFF` (what `tools/card-ci/run-card-ci.sh` configures), profile
`nvidia/a10g`. Programs: the sm_86 binaries of PolyBench/GPU, Rodinia and the reduced HeCBench list, built earlier with nvcc 13.0 and `-cudart shared`,
copied to the rig in the layout of `tools/bench/real/pack_corpora.sh` (not rebuilt), with the arguments of `tools/bench/real/real.py`.

## Commands

```
export PATH=/usr/local/cuda-13.0/bin:$PATH NVCC=/usr/local/cuda-13.0/bin/nvcc
# A
CARD_CI_JOBS=4 tools/card-ci/run-card-ci.sh ~/hgr/build
# B
tools/probes/hostgpu_turbo_card_probes.sh atomics out/B 8
tools/probes/hostgpu_turbo_card_probes.sh tensor  out/B ~/hgr/build
tools/probes/hostgpu_turbo_card_probes.sh sfu     out/B 1
# C, once per suite (rodinia, polybench, hecbench)
python3 tools/bench/hostgpu/run.py --build build --corp corp --gpu nvidia/a10g --corpora $suite --runs 3 --long-secs 30 --long-runs 1 \
    --forms sim,exact,exactdiv,turbo,verify --timeout 1200 --verify-timeout 600 --json $suite.json --md $suite.md
```

## Files

| file | what |
|---|---|
| `card-ci.txt` | stage A: the output of `run-card-ci.sh`, first run (its `sin` and `cos` lines are of the probe's first model, corrected in `probes.txt`) |
| `probes.txt` | stage B: the RESULT, VERDICT and TENSOR lines |
| `rodinia.md`, `polybench.md`, `hecbench.md` | stage C: the harness tables. Their verify columns are of the pass before the verify fix |
| `atomics-opt-in.md` | the three programs with launches refused for atomics, with `--env VGPU_HOST_GPU_ATOMICS=1` (after the fix) |
| `start-cost-0.md` | PolyBench and Rodinia without 2MM, CORR, COVAR, GRAMSCHM, one run, `--env VGPU_HOST_GPU_START_COST=0` |
| `bench.json` | the harness records of every pass in one file, compact: `rodinia`, `polybench`, `hecbench`, `verify_after_fix` (52 programs, verify form only), `atomics_opt_in`, `start_cost_0`. Per form: `s` median seconds, `x` speed-up, `card` / `launches`, `card_s`, `sim_s`, `kept` (launches kept in the simulator, by reason), `out`, `verify` |
| `summary.txt` | the per-suite numbers the round text quotes |

HeCBench was run with a 300 s limit per run, and `fpdc-cuda` and `zerocopy-cuda` were skipped. Follow-up passes:

```
H="python3 tools/bench/hostgpu/run.py --build build --corp corp --gpu nvidia/a10g"
$H --skip <the programs over 30 s, atomicCost-cuda, fpdc-cuda, zerocopy-cuda> --runs 1 --forms sim,turbo,verify --timeout 300 --verify-timeout 300
$H --programs aop-cuda,atomicPerf-cuda,bh-cuda --runs 3 --forms sim,turbo,verify --env VGPU_HOST_GPU_ATOMICS=1 --timeout 300
$H --corpora polybench,rodinia --skip 2MM,CORR,COVAR,GRAMSCHM --runs 1 --forms sim,exact,turbo --env VGPU_HOST_GPU_START_COST=0 --timeout 600
```
