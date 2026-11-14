#!/bin/bash
cd /tmp/sp69-inq/runs
SH=/tmp/sp69-inq/shims
SD="[eigensolvers][steepest_descent]"; KSH="[hamiltonian][ks_hamiltonian]"; IG="[ground_state][initial_guess]"; POI="[solvers][poisson]"
echo "## c4 against c3 on one thread"
for r in 1 2; do for v in c3 c4; do ./one.sh t1_sd_$v $SH/$v "$SD" VGPU_THREADS=1; ./one.sh t1_ksh_$v $SH/$v "$KSH" VGPU_THREADS=1; done; done
echo "## four at once (ctest -j4): c1 against c2"
for t in "sd:$SD" "ig:$IG" "poi:$POI"; do n=${t%%:*}; tags=${t#*:}
  for v in c1 c2 c1 c2; do
    t0=$(date +%s.%N)
    for i in 1 2 3 4; do ./one.sh par_${n}_${v}_$i $SH/$v "$tags" > par_${n}_${v}_$i.out & done
    wait
    t1=$(date +%s.%N)
    echo "par4 $n $v all done in $(echo "$t1 - $t0" | bc) s: $(cat par_${n}_${v}_?.out | awk "{w+=\$2; c+=\$3+\$4; ok+=(\$5==\"rc=0\")} END{printf \"mean wall %.1f s, total CPU %.0f s, %d ok\", w/4, c, ok}")"
    rm -f par_${n}_${v}_?.out
  done
done
echo BATCHDONE
