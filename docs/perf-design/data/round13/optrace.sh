#!/bin/bash
# opcode set of one kernel's warp 0 of block 0: optrace.sh <program> <kernel substring> [args]
prog=$1; k=$2; shift 2
cd $(mktemp -d)
VGPU_QUIET=1 VGPU_GPU=nvidia/a10g VGPU_THREADS=2 VGPU_SASS_TRACE=0 VGPU_SASS_TRACE_KERNEL=$k VGPU_HOSTGPU_CAPTURE_STOP=0 LD_LIBRARY_PATH=$HOME/sim/build/shim timeout 100 $prog "$@" 2>&1 | head -c 40000000 | grep -oE "\b(MUFU[.A-Z0-9]*|FADD[.A-Z0-9]*|FMUL[.A-Z0-9]*|FFMA[.A-Z0-9]*|RED[.A-Z0-9]*|ATOM[.A-Z0-9]*|ATOMG[.A-Z0-9]*|DADD|DMUL|DFMA|F2F[.A-Z0-9]*|HFMA2[.A-Z0-9]*|HMMA[.A-Z0-9]*|SHFL[.A-Z0-9]*|VOTE[.A-Z0-9]*|MATCH[.A-Z0-9]*|REDUX[.A-Z0-9]*|FSETP[.A-Z0-9]*|FMNMX|FRND[.A-Z0-9]*|I2F[.A-Z0-9]*|F2I[.A-Z0-9]*|FCHK|S2R|CS2R|BAR[.A-Z0-9]*|ERRBAR|MEMBAR[.A-Z0-9]*)\b" | sort | uniq -c | sort -rn | head -30
