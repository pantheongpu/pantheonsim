# Round sp64's probes, as printed (server3, nice 19, under the machine lock). Speed-ups in brackets are the first side's median over the side's.

## 1. What a compile that is running costs the run (scripts/ab.py, 3 runs a side, an empty cache before every run)
a = no JIT; Bfail = base build, VGPU_JIT=1 with a compiler that fails every compile at once (the section is Failed: the plain loop);
Bsleep = base build with a compiler that never finishes (the section stays Queued: the counting loop until the end); Nsleep = the same on the changed build (Queued on the plain loop).
### one thread (taskset 12-15)
polybench/SYRK                   a=5.496(1.000) Bfail=5.766(0.953) Bsleep=5.712(0.962) Nsleep=5.770(0.953) same=True objs={'Bfail': [0, 0, 0], 'Bsleep': [0, 0, 0], 'Nsleep': [0, 0, 0]}
rodinia/gaussian                 a=3.694(1.000) Bfail=3.751(0.985) Bsleep=3.886(0.951) Nsleep=3.702(0.998) same=True objs={'Bfail': [0, 0, 0], 'Bsleep': [0, 0, 0], 'Nsleep': [0, 0, 0]}
hecbench/asyncAllocation-cuda    a=3.256(1.000) Bfail=3.296(0.988) Bsleep=3.561(0.914) Nsleep=3.347(0.973) same=True objs={'Bfail': [0, 0, 0], 'Bsleep': [0, 0, 0], 'Nsleep': [0, 0, 0]}
hecbench/atomicAggregate-cuda    a=3.718(1.000) Bfail=3.739(0.994) Bsleep=3.832(0.970) Nsleep=3.755(0.990) same=True objs={'Bfail': [0, 0, 0], 'Bsleep': [0, 0, 0], 'Nsleep': [0, 0, 0]}
lock: 1 holds, 146 s waited
### 16 threads
polybench/SYRK                   a=1.804(1.000) Bfail=1.819(0.992) Bsleep=1.833(0.984) Nsleep=1.818(0.992) same=True objs={'Bfail': [0, 0, 0], 'Bsleep': [0, 0, 0], 'Nsleep': [0, 0, 0]}
rodinia/gaussian                 a=0.903(1.000) Bfail=0.944(0.956) Bsleep=0.983(0.918) Nsleep=0.932(0.968) same=True objs={'Bfail': [0, 0, 0], 'Bsleep': [0, 0, 0], 'Nsleep': [0, 0, 0]}
hecbench/asyncAllocation-cuda    a=0.989(1.000) Bfail=1.069(0.925) Bsleep=1.091(0.906) Nsleep=1.023(0.967) same=True objs={'Bfail': [0, 0, 0], 'Bsleep': [0, 0, 0], 'Nsleep': [0, 0, 0]}
hecbench/atomicAggregate-cuda    a=0.900(1.000) Bfail=0.905(0.994) Bsleep=0.929(0.968) Nsleep=0.908(0.990) same=True objs={'Bfail': [0, 0, 0], 'Bsleep': [0, 0, 0], 'Nsleep': [0, 0, 0]}
lock: 1 holds, 61 s waited

## 2. The code an -O level makes, warm (scripts/ab.py, changed build, one thread, 3 runs; w2 = the JIT's -O2, w1 = VGPU_JIT_FLAGS=-O1, wnu = VGPU_JIT_FLAGS=-fno-unroll-loops; each side's cache filled by one untimed synchronous run)
polybench/SYRK                   a=5.456(1.000) w2=4.804(1.136) w1=6.298(0.866) wnu=5.311(1.027) same=True objs={'w2': [1, 1, 1], 'w1': [1, 1, 1], 'wnu': [1, 1, 1]}
rodinia/gaussian                 a=3.651(1.000) w2=3.492(1.046) w1=4.288(0.851) wnu=3.643(1.002) same=True objs={'w2': [1, 1, 1], 'w1': [1, 1, 1], 'wnu': [1, 1, 1]}
hecbench/asyncAllocation-cuda    a=3.257(1.000) w2=2.587(1.259) w1=4.574(0.712) wnu=2.988(1.090) same=True objs={'w2': [1, 1, 1], 'w1': [1, 1, 1], 'wnu': [1, 1, 1]}
hecbench/atomicAggregate-cuda    a=3.691(1.000) w2=3.260(1.132) w1=4.777(0.773) wnu=3.517(1.050) same=True objs={'w2': [1, 1, 1], 'w1': [1, 1, 1], 'wnu': [1, 1, 1]}
polybench/SYR2K                  a=10.297(1.000) w2=8.803(1.170) w1=10.449(0.985) wnu=9.743(1.057) same=True objs={'w2': [1, 1, 1], 'w1': [1, 1, 1], 'wnu': [1, 1, 1]}
lock: 2 holds, 333 s waited

## 3. The compiler on the kept sources (scripts/ccost.sh, one CPU, seconds; the first line of each is a file that includes only the runtime header, and has no [hdr-only-O2] column)
## src-hecbench_asyncAllocation-cuda
file bytes version [hdr-only-O2] E syntax O0 O1 O2 O2-no-unroll
empty.cpp 63 0.004 0.063 0.349 0.433 0.426 0.431 0.433
857ddd99db45 167225 0.004 0.428 0.066 0.376 2.289 1.524 2.373 1.847
## src-hecbench_atomicAggregate-cuda
file bytes version [hdr-only-O2] E syntax O0 O1 O2 O2-no-unroll
empty.cpp 63 0.005 0.063 0.359 0.393 0.437 0.434 0.431
35a9c9f02aeb 40626 0.005 0.462 0.070 0.362 0.812 0.642 0.724 0.691
## src-polybench_SYRK
file bytes version [hdr-only-O2] E syntax O0 O1 O2 O2-no-unroll
empty.cpp 63 0.003 0.063 0.354 0.393 0.431 0.547 0.425
8ab93f1be365 162539 0.004 0.432 0.065 0.393 1.989 1.590 2.179 2.204
## src-rodinia_gaussian
file bytes version [hdr-only-O2] E syntax O0 O1 O2 O2-no-unroll
empty.cpp 63 0.004 0.063 0.357 0.391 0.428 0.566 0.443
7c90e323cd0c 43469 0.004 0.600 0.062 0.358 0.750 0.666 0.772 0.795
### other -O forms (scripts/ccost2.sh)
857ddd99db45 [-O2]=2.284 [-O1 -ftree-vectorize]=1.731 [-O1 -ftree-vectorize -finline-small-functions]=1.691 [-O2 -fno-gcse -fno-tree-pre]=1.919 [-O2 -fno-schedule-insns2]=2.073
35a9c9f02aeb [-O2]=0.815 [-O1 -ftree-vectorize]=0.646 [-O1 -ftree-vectorize -finline-small-functions]=0.657 [-O2 -fno-gcse -fno-tree-pre]=0.784 [-O2 -fno-schedule-insns2]=0.824
8ab93f1be365 [-O2]=2.208 [-O1 -ftree-vectorize]=1.779 [-O1 -ftree-vectorize -finline-small-functions]=1.807 [-O2 -fno-gcse -fno-tree-pre]=2.283 [-O2 -fno-schedule-insns2]=2.485
7c90e323cd0c [-O2]=0.777 [-O1 -ftree-vectorize]=0.683 [-O1 -ftree-vectorize -finline-small-functions]=0.684 [-O2 -fno-gcse -fno-tree-pre]=0.753 [-O2 -fno-schedule-insns2]=0.770
### the runtime header's parts, -fsyntax-only, ms, median of 5 (scripts/hdrcost.sh)
nothing 7 / cstdint 12 / immintrin 286 / algorithm 85 / cmath 90 / bit+cstring 22 / all five 402 / the runtime header 424
### a precompiled runtime header (scripts/pchprobe.sh; the header made once, 1.03 to 1.11 s, 53 MB; then each compile twice with it)
857ddd99db45 bytes=167225 plain=2.036 make-pch=1.031 gch-bytes=53200515 with-pch=1.666 with-pch-again=2.072 pch-used=1
35a9c9f02aeb bytes=40626 plain=0.725 make-pch=1.044 gch-bytes=53200687 with-pch=0.356 with-pch-again=0.347 pch-used=1
8ab93f1be365 bytes=162539 plain=2.190 make-pch=1.106 gch-bytes=53204475 with-pch=1.838 with-pch-again=1.805 pch-used=1
7c90e323cd0c bytes=43469 plain=0.767 make-pch=1.030 gch-bytes=53200753 with-pch=0.752 with-pch-again=0.526 pch-used=1
with -include of the precompiled header and the source unchanged (atomicAggregate): make 1.03 s, compile 0.386 s against 0.720 s, the object the same bytes (cmp)

## 4. Sixteen threads or fifteen while a compile runs (scripts/ab.py, changed build, default threads, 5 runs (3 for SYRK); s = round sp63's second run: an empty cache, one untimed run, the timed run)
polybench/SYRK                   a16=1.803(1.000) a15=1.806(0.998) s16=1.953(0.923) s15=1.956(0.922) same=True objs={'s16': [0, 0, 0], 's15': [0, 0, 0]}
rodinia/gaussian                 a16=0.893(1.000) a15=0.889(1.005) s16=0.978(0.914) s15=0.970(0.921) same=True objs={'s16': [0, 0, 0, 0, 0], 's15': [0, 0, 0, 0, 0]}
hecbench/asyncAllocation-cuda    a16=0.985(1.000) a15=0.971(1.014) s16=1.078(0.913) s15=1.072(0.918) same=True objs={'s16': [0, 0, 0], 's15': [0, 0, 0]}
hecbench/atomicAggregate-cuda    a16=0.899(1.000) a15=0.909(0.990) s16=0.950(0.947) s15=0.962(0.935) same=True objs={'s16': [0, 0, 0, 0, 0], 's15': [0, 0, 0, 0, 0]}
lock: 1 holds, 115 s waited
