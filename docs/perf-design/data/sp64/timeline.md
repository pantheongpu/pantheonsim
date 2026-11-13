# Where the compile goes (base build f55c8e65, VGPU_JIT=1, an empty cache, one run under strace -f --seccomp-bpf -e trace=execve,exit_group; scripts/timeline.py)
# seconds from the program's start: the threshold's compile asked for (cc1plus started), cc1plus done, the object linked (ld done), the program's end
## 1 thread, taskset 12-15
program | wall s | source bytes | cc1plus start | cc1plus end | ld end | end | object | jit summary
polybench/SYRK | 6.036 | [162539] | 2.691 | 5.011 | 5.053 | 6.03 | 1 | 1 units (4 regions), 1 ready, 0 cache hits, 1 compiles (2445.3 ms), 0 failures, 0 damaged objects replaced; 4590
rodinia/gaussian | 3.886 | [43469] | 2.613 | 3.441 | 3.473 | 3.881 | 1 | 2 units (10 regions), 1 ready, 0 cache hits, 1 compiles (864.7 ms), 0 failures, 0 damaged objects replaced; 2200967
hecbench/asyncAllocation-cuda | 3.697 | [167225] | 1.305 | 3.6 | 3.649 | 3.692 | 1 | 1 units (10 regions), 1 ready, 0 cache hits, 1 compiles (2352.9 ms), 0 failures, 0 damaged objects replaced; 241056
hecbench/atomicAggregate-cuda | 3.783 | [40626] | 2.327 | 3.106 | 3.138 | 3.778 | 1 | 1 units (3 regions), 1 ready, 0 cache hits, 1 compiles (814.5 ms), 0 failures, 0 damaged objects replaced; 2971742
## 16 threads
program | wall s | source bytes | cc1plus start | cc1plus end | ld end | end | object | jit summary
polybench/SYRK | 1.966 | [162539] | 0.63 | 1.962 | - | 1.962 | 0 | 1 units (4 regions), 0 ready, 0 cache hits, 1 compiles (1335.4 ms), 1 failures, 0 damaged objects replaced; 0
rodinia/gaussian | 1.2 | [43469] | 0.709 | 1.196 | - | 1.196 | 0 | 2 units (10 regions), 0 ready, 0 cache hits, 1 compiles (484.7 ms), 1 failures, 0 damaged objects replaced; 0
hecbench/asyncAllocation-cuda | 1.519 | [167225] | 0.533 | 1.514 | - | 1.514 | 0 | 1 units (10 regions), 0 ready, 0 cache hits, 1 compiles (984.2 ms), 1 failures, 0 damaged objects replaced; 0
hecbench/atomicAggregate-cuda | 0.959 | [40626] | 0.632 | 0.954 | - | 0.954 | 0 | 1 units (3 regions), 0 ready, 0 cache hits, 1 compiles (335.7 ms), 1 failures, 0 damaged objects replaced; 0
