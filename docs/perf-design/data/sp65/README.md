# Round sp65: a launch while the card holds the memory (data)

The design is section 9 of [../../12-host-gpu-turbo.md](../../12-host-gpu-turbo.md) ("A launch while the card holds the memory").
The round's text is appended to [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md) after the card part.

## Before the card (server3: GCC 13.3, `Release`, `-Werror`, no LTO, `VGPU_THREADS=4`, profile `nvidia/a10g`, no driver)

Every build and run under the machine's build lock, `nice 19`, `-j3`. The corpus is the sm_86 binaries of PolyBench/GPU, Rodinia and the
reduced HeCBench list (`tools/bench/real/pack_corpora.sh`), with the arguments of `tools/bench/real/real.py`.

| file | what |
|---|---|
| `trace_corpus.py` | runs programs with `VGPU_HOST_GPU_MAX_MIB=0` (every launch refused for its size: no card, no driver, no sample) and `VGPU_HOST_GPU_TRACE`: every allocation, free, host copy and launch, with the launch's own simulator time, whether the mode admits its kernel, its block size, static count and loop flag |
| `residency_replay.py` | replays the runtime's rules over a trace, with the residency of every 64 KiB chunk: today's rule and the streak rule (`pull`, the one built), and variants (`pullpush`: also charge the push back; `pullnf`: no 200 us floor; `--kdefault`: another card time before a kernel has run there). `--fit` takes, per program, the factor of the rig's simulator over server3's at which today's rule reproduces Round sp59's counts on the rig. `--q1`, `--why`, `--changes`: the tables below |
| `sim_scale.py` | the rig's simulator against server3's, from the programs that ran wholly in the simulator in sp59 |
| `q2_card_time.py` | question 2: sp59's card path per launch, less the modelled pushes, against the replayed launches' simulator time |
| `analyze.sh` | all of the tables from a directory of traces |
| `stub_check.py` | the runtime's own decisions against the stand-in driver as a card that keeps memory and computes nothing (outputs change; only the decisions are compared): launches on the card, kept for size, sent by the streak rule, with and without `VGPU_HOST_GPU_STREAK=0` |
| `mutants.py`, `mutants.txt` | mutants of `streak_offload` and its two sides in `include/vgpu/host_gpu.hpp`, each a copy of the header outside the tree, against `test_host_gpu_logic` |
| `card-run.sh`, `cardlog.py` | the card part, by stage (`build`, `ci`, `target`, `ab`, `log`, `repeat`), and the per-kernel summary of a `VGPU_HOST_GPU_LOG=1` run |
