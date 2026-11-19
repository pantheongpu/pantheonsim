# Fuzzing

Untrusted bytes enter the simulator in three places, and each has a libFuzzer target in `tests/fuzz/`. Input that is not something the simulator runs must come back as an error (a `vgpu::Error`, or a `CUresult` from the driver API): never a crash, a sanitizer report, a hang or another kind of exception.

| Target | What it feeds |
| --- | --- |
| `fuzz_ptx` | `ptx::parse` and the register analysis of every entry it returns |
| `fuzz_sass` | `sass::decode` and `to_text` for 16-byte words (byte 0 picks the architecture), and `parse_cubin` with the decoding of each kernel's text |
| `fuzz_driver` | `libcuda`'s `cuModuleLoadData` with PTX, cubin and fatbin images and the launch of what loads; streams of memory calls with arbitrary device pointers, sizes and offsets |

The simulator has no ioctl layer to fuzz; the driver API is its entry point from a program.

## Seeds

`tests/fuzz/make_seeds.py` writes a small corpus into `<build>/fuzz-seeds/{ptx,sass,driver}` when CMake configures, from files the repository already has: PTX files and the PTX raw strings of the unit tests, the nvdisasm corpus in `nvidia/tests/data/sass`, and the cubin in `nvidia/tests/data`. `tests/fuzz/ptx.dict` is the PTX dictionary.

## Replay tests

In an ordinary build each target has `standalone_main.cpp` for a `main()` and is a ctest test, `fuzz_replay_ptx`, `fuzz_replay_sass` and `fuzz_replay_driver`. Each replays the seeds and `tests/fuzz/regress/<target>/`. The ASan job runs them under the sanitizers.

## Fuzzing

`.github/workflows/fuzz.yml` builds with clang (`-DVGPU_FUZZ=ON`, `-fsanitize=address,undefined,fuzzer-no-link`) and runs `tests/fuzz/run_fuzz.sh <build> <target> <seconds>` for each target: 60 s on a pull request, 15 minutes a night (the corpus carries over from night to night), or the seconds given to a manual run. A failing run uploads the crashing input as the `fuzz-artifacts` artifact.

To run one by hand, with clang and libFuzzer installed:

    CC=clang CXX=clang++ cmake -S . -B build-fuzz -G Ninja -DCMAKE_BUILD_TYPE=Debug -DVGPU_FUZZ=ON \
      -DCMAKE_CXX_FLAGS="-O1 -g -fsanitize=address,undefined,fuzzer-no-link" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined"
    cmake --build build-fuzz --target fuzz_ptx fuzz_sass fuzz_driver
    tests/fuzz/run_fuzz.sh build-fuzz ptx 300

## When a fuzzer finds something

Minimise the input (`-minimize_crash=1`), put it in `tests/fuzz/regress/<target>/` under a name that says what it was, and fix the library in the same change. The replay test then guards the fix.
