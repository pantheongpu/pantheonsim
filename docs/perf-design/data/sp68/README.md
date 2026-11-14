# Round sp68: the IEEE square root idiom in host-GPU exact mode (data)

The text is *Round sp68* in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md); the recogniser is
`include/vgpu/host_gpu_idiom.hpp`.

## Machines

* server3 (AMD Ryzen 7 PRO 8840U, GCC 13.3 and 12, `Release`, `-DVGPU_LTO=OFF`; shared, every build and run under the machine's build lock,
  `nice -n 19`, `-j3`). No CUDA toolkit and no card: unit tests, mutants, the static scan, the model of the refinement, the runtime verdicts with no
  driver (`VGPU_HOST_GPU_MAX_MIB=0`) and a stand-in card that keeps memory and computes nothing (`nvidia/tests/e2e/host_gpu_stub_driver.c`).
* One A10G (AWS g5.xlarge, driver 595.112.07, 4 vCPU, Ubuntu, g++ 13, `ninja-build` installed), launched and terminated by the coordinator; nvcc 13.0.88 at
  `/usr/local/cuda-13.0` for the probe and the e2e programs; the corpus programs are the cached sm_86 binaries of earlier rounds (nvcc 13.0).
  Two builds from `git archive`: `b68` = `sp68-sqrt-idiom`, `b65` = the scratch branch `sp68-with-sp65` (this branch merged with
  `sp65-resident-streaks`, for GRAMSCHM only). The rig ran nothing else.

## Files

| file | what |
|---|---|
| `gen_sass_inc.py` | turns Round sp67's `cuobjdump -sass` listings (nvcc 13.0 and 13.2, own kernels) into `tests/unit/host_gpu_idiom_sass.inc`, checking the two toolkits' words are equal |
| `sqrt_model.cpp`, `sqrt_model.txt` | the refinement of nvcc's sqrt.rn as scalar code over every positive float, with the reciprocal square root k ulp from the simulator's |
| `mutants.py`, `mutants.txt` | 24 mutants of the recogniser against `test_host_gpu_idiom`: 19 killed, the 5 survivors are checks another check covers |
| `scan-summary.txt` | `tools/probes/hostgpu_idiom_scan` over the 539 cached corpus programs (2,613 sm_86 kernels): the kernels newly admitted, the verdict counts, the reasons sites fail |
| `offline.sh`, `offline.md` | the newly admitted kernels run on server3 with no card: the runtime's verdicts and the simulator's times |
| `gram_stub.sh`, `gram_stub.txt` | GRAMSCHM's launch decisions against the stand-in card, on this branch and on the merge with sp65 |
| `ab.py`, `card_ab.sh`, `card_verify.sh`, `card_short.sh`, `summarise.py` | the card A/B (3 interleaved runs per side), the verify runs, the short programs again (20 runs a side), the table |
| `card.md`, `card/` | the card's results: probe, e2e, A/B table, verify counts; the runs' JSON and the probe's and e2e scripts' output |

## Commands

```
# server3
python3 mutants.py src build mut
hostgpu_idiom_scan $(cat progs.txt)            # build line in tools/probes/hostgpu_idiom_scan.cpp; SCAN_SITES=1 adds a line per MUFU.RSQ
./offline.sh ; ./gram_stub.sh build ; ./gram_stub.sh build65
# rig (~/w: s68, s65 sources, b68, b65 builds, corp)
./sfu sqrtrn 1 ; HOSTGPU_SFU_MUTATE=1 ./sfu sqrtrn 257
VGPU_BUILD_DIR=~/w/b68 bash nvidia/tests/e2e/run_host_gpu.sh ; VGPU_BUILD_DIR=~/w/b68 bash nvidia/tests/e2e/run_host_gpu_stub.sh   # PATH with cuda-13.0
bash card_ab.sh ; bash card_verify.sh
```
