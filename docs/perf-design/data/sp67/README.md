# Round sp67: verify mode's copies (#569) and two A10G observations plus a toolkit question (#570) (data)

The text is *Round sp67* in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md); the gate argument of the change is section 8 of
[../../10-host-gpu-hooks.md](../../10-host-gpu-hooks.md).

## Machines

* server3: AMD Ryzen 7 PRO 8840U, GCC 13.3, `Release`, `-DVGPU_LTO=OFF`, shared with other sessions; every build and timed run under the machine's build
  lock, `nice -n 19`, `-j3`. No CUDA toolkit and no card: verify runs use the stand-in driver (`nvidia/tests/e2e/host_gpu_stub_driver.c`) as a card that
  keeps memory and computes nothing (`STUB_CARD_OK=1 STUB_KEEP_MEMORY=1`), so every compare finds the chunks a kernel wrote different and the others equal.
* The A10G rig of round sp65 (see `card-570.md`), shared with that round's timed runs; nvcc 13.0.88 and 13.2.51 at `/usr/local/cuda-13.0` and
  `/usr/local/cuda-13.2`, used by explicit path (`cuda-cuobjdump` and `cuda-nvdisasm` 13.0 and 13.2 were installed from NVIDIA's apt repository for the
  SASS dumps).

## Files

| file | what |
|---|---|
| `verify_ab.py` | #569: interleaved runs of PolyBench programs on several (build, environment) sides, with the stand-in card or a real one; wall time (the wait for the lock is not timed), exit status, a hash of stdout without its timing lines, the verify summary |
| `ab-server3.md`, `ab-server3.json`, `ab-server3-modes.json` | #569 on server3: base and new build, plain and verify, 8 programs, 5 runs a side; then exact, turbo and turbo with verify on 3 programs |
| `card-ab.md`, `card-ab.json` | #569 on the A10G, rig idle: 8 programs, new plain, base and new verify, 3 runs a side |
| `card-ab-part1.txt` | #569 on the A10G: the partial runs (2DCONV, two of 3DCONV) made while round sp65 was building on the rig; see the round text |
| `mutants.py`, `mutants.txt` | 10 mutants of `compare_with_card` and `keep_verified_chunks` against `test_host_gpu_track`: 10 killed |
| `fakenvcc.sh` | server3 has no nvcc: stands in for nvcc 13.0 in `run_host_gpu_stub.sh` by copying `host_gpu_lazy` built on the rig, so the whole stand-in-driver test (steps 1 to 23 and 27) ran on server3 against both builds |
| `logged_run.py` | #570: one program with `VGPU_HOST_GPU_LOG=1`, stderr stamped, summarised per kernel |
| `card-570.md` | #570 items 1 and 2: GRAMSCHM in exact mode, the driver start, from logged runs on the A10G |
| `drvstart.py` | #570 item 2: the driver calls of the mode's start, timed one by one |
| `tk.sh`, `sass_compare.py`, `sass-13.0-vs-13.2.txt` | #570 item 3: the division probe, the stand-in test's kernels and six PolyBench programs built with both toolkits, SASS compared per kernel |
| `divrec.sh`, `divrec-13.0-vs-13.2.txt` | #570 item 3: the division idiom recogniser (`VGPU_HOST_GPU_DIV=1`) on both builds of each program, no card (`VGPU_HOST_GPU_MAX_MIB=0`) |

## Commands (server3)

```
python3 verify_ab.py --corp corp --out ab2.json --runs 5 --stub libstubcuda.so --progs 2DCONV,3DCONV,ATAX,BICG,GESUMMV,MVT,GEMVER,JACOBI1D \
  --side base_plain=build-base --side new_plain=build \
  --side "base_verify=build-base:VGPU_HOST_GPU=exact;VGPU_HOST_GPU_VERIFY=1" --side "new_verify=build:VGPU_HOST_GPU=exact;VGPU_HOST_GPU_VERIFY=1"
python3 mutants.py --src src --build build --work mut
env TMPDIR=tmp VGPU_NVCC=fakenvcc/nvcc VGPU_BUILD_DIR=build bash src/nvidia/tests/e2e/run_host_gpu_stub.sh        # and VGPU_BUILD_DIR=build-base
```
