# Integration of rounds sp64 to sp68: the driver start of short programs (data)

The text is *Integration of rounds sp64 to sp68* in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md). server3 (GCC 13.3,
`Release`, `-DVGPU_LTO=OFF`, `-Werror`), every build and run under the machine's build lock, `nice -n 19`, `-j3`, holds under 5 minutes. No card and no
nvcc: the corpus is the sm_86 binaries of `tools/bench/real/pack_corpora.sh` (PolyBench, Rodinia, the reduced HeCBench list, the CUDA Samples).

| file | what |
|---|---|
| `startwho.py` | runs corpus programs with `VGPU_HOST_GPU_LOG=1` against the stand-in driver (`nvidia/tests/e2e/host_gpu_stub_driver.c`, which refuses every module, so every launch runs in the simulator) and prints the line that names the rule which started the driver; `--stop` ends a run there |
| `start-who.md` | its output before the change (most of the corpus, both modes) and for the programs that matter after it |
| `card.md`, `card/` | the A10G run: card CI (`card-ci2.txt`), the four programs and the keep list with `exactl0`/`turbol0` (the old rule) interleaved, reruns, each round's headline |
| `mutants.py`, `mutants.txt` | mutants of the start-share rule in `include/vgpu/host_gpu.hpp`; `tests/unit/test_host_gpu_logic.cpp` must fail against each (the runner is `data/sp66/mutants.py` with this list) |

Commands (on server3, from a directory holding `src/`, `build/`, `corp/` and `stub.so`, the stand-in driver built with `gcc -shared -fPIC -O1`):

```
python3 startwho.py --build $PWD/build --corp $PWD/corp --stub $PWD/stub.so --real $PWD/src/tools/bench/real --programs FDTD-2D,dct8x8 --stop \
  [--env VGPU_HOST_GPU_START_COST=0.08] [--env VGPU_HOST_GPU_START_LAUNCHES=0]
python3 mutants.py --src src --work mut
```
