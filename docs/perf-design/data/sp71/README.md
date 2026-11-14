# Round sp71: the hiprtc cache (data)

The text is *Round sp71* in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md). server3 (Ryzen 7 PRO 8840U, GCC 13.3,
`Release`), every build under the machine's build lock, `nice -n 19`, `-j3`. Another round was building and measuring on the machine throughout:
wall times are noisy. ROCm 6.4.4 extracted from its debs into a directory (`VGPU_ROCM_PATH`); ROCm 7.1.1's `hip-runtime-amd` was extracted only to
read its `libhiprtc` (`nm -D`, `strings`, `readelf -d`). inq f939e63 built as in Round sp70 (`data/sp70/README.md`).

| file | what |
|---|---|
| `inq.md` | the six inq tests with the cache off, cold and warm (median and each pass), where the compiles came from, four tests at once, ROCm's own cache |
| `inq-runs.txt` | every run behind it |
| `inq_one_mode.sh`, `passes.sh`, `pass3_and_four_at_once.sh` | the runners (paths are server3's) |
| `mutants.py`, `mutants.txt` | 53 single edits of `amd/src/hiprtc_cache.cpp`; `test_hiprtc_cache` is rebuilt and must fail |
| `tests.txt` | the two tests and the configuration tests on the last commit |

Commands (from the build directory unless a path says otherwise):

```
./test_hiprtc_cache $PWD/hiprtc-fakes                                   # ctest hiprtc_cache: the stand-in, no ROCm
VGPU_ROCM_PATH=<rocm> VGPU_BUILD_DIR=$PWD ../amd/tests/e2e/run_hiprtc_cache.sh $PWD/hiprtc_probe   # ctest amd_hiprtc_cache
python3 mutants.py --src <tree> --build <tree>/build
./inq_one_mode.sh <label> off|cache <cache dir> laplacian gradient ...   # -> <label>.tsv
strace -f -e trace=openat <a program that calls hiprtc>                  # what a real compile reads: how the key was found
```
