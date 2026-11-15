# Round sp70: the AMD engine on inq (data)

The text is *Round sp70* in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md). server3 (Ryzen 7 PRO 8840U, 8 cores and 16
hardware threads, GCC 13.3, `Release` with the default link-time optimization), every build under the machine's build lock, `nice -n 19`, `-j3`;
timings only while nothing else of this round ran. No card.

The workload: inq f939e63 (https://gitlab.com/npneq/inq) configured as its user's CI job does (`-DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx942
-DENABLE_GPU_INTEGRATED_MEMORY=OFF -DINQ_MPI_DEVICE_POINTERS_RUN=0`, ROCm 6.4.4), target `all_unit_tests`, run as
`VGPU_GPU=amd/mi300x LD_LIBRARY_PATH=<build>/shim:... all_unit_tests "[dir][test]"` with `PSEUDOPOD_SHARE_PATH` and `INQ_SHARE_PATH` pointing
into its source tree. ROCm and inq's other dependencies were extracted from their debs into a directory (no root); inq's `python` subdirectory was
left out of the build. A "build" below is a directory with `shim/` and `libamdhip64.so*` of one commit.

| file | what |
|---|---|
| `inq.md` | the eight tests build by build, the one-thread runs, four copies at once, the two halves of the first form of change 3 |
| `inq-passes.txt` | every run behind it: `<test> <wall s> <user s> <system s> rc pass out=<md5 of the output without timestamps>` |
| `one-thread-and-four-at-once.txt` | the raw lines of those two measurements |
| `inq8.sh`, `one.sh`, `one_thread.sh`, `four_at_once.sh` | the runners (paths are server3's) |
| `bench.sh` | `tools/bench-amd/bench.py` with this round's hipcc and cache: `bench.sh <tag> <buildA> <buildB> [bench options]` |
| `bench-base-vs-final.md` | its A/B of the base against the last commit: both targets, 1 thread and "all", 7 runs, counters identical in 164 rows |
| `variants.md` | the bisect of the one-thread rows over the commits, and the variants: a function nothing calls added to `gcn_tier.cpp` and to `gcn_exec.cpp`, `Pool::kMax` alone, fmin/fmax alone, the SDWA comparison handler alone |
| `mutants.py`, `mutants.txt` | one edit at a time in a build tree, the named test rebuilt and expected to fail |
| `ctest.txt` | the AMD ctest set on the last commit |

Commands (from the directory that holds the trees and builds):

```
./inq8.sh <label> <dir whose build/shim is the shim>          # -> runs/<label>.tsv
./one.sh <label> <dir> "[dir][test]" VGPU_THREADS=1
./bench.sh final shims/base/build shims/final/build --runs 7
./bench.sh var shims/c2/build shims/v_nop/build --runs 9 --threads 1 --workloads p_dpp,p_dpp_arch,p_dpp_scan,atomics,p_sbuf,p_f64x,smfmac,valu_f64,p_i64
python3 mutants.py --src <tree> --build <tree>/build
ROCM_PATH=<rocm> VGPU_ROCM_PATH=<rocm> ctest -R "amd|hip|gcn|lockstep|rocm|roc|stream|host_cores|host_cpus" -j4 --timeout 2400
```

The profile that chose the changes (a `SIGPROF` sampler loaded with `LD_PRELOAD`, its report script and the per-test reports) is not in the
repository: `perf` is closed to unprivileged users on server3, and the sampler is a tool of the round, not of the simulator.
