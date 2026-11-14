#!/bin/bash
cd /tmp/sp69-inq/runs
SH=/tmp/sp69-inq/shims
KSH="[hamiltonian][ks_hamiltonian]"; SD="[eigensolvers][steepest_descent]"; TR="[operations][transform]"; POI="[solvers][poisson]"
for r in 1 2 3; do for v in base c3 c4; do ./one.sh t1_ksh_$v $SH/$v "$KSH" VGPU_THREADS=1; done; done
for r in 1 2; do for v in base c3 c4; do ./one.sh t1_sd_$v $SH/$v "$SD" VGPU_THREADS=1; ./one.sh t1_tr_$v $SH/$v "$TR" VGPU_THREADS=1; done; done
for v in base c3 c4; do ./one.sh t1_poi_$v $SH/$v "$POI" VGPU_THREADS=1; done
echo BATCHDONE
cd /tmp/sp69-inq && ./bench70.sh full shims/base/build shims/c3/build --runs 7
