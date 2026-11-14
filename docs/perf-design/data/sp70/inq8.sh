#!/bin/bash
# usage: inq8.sh <label> <simdir (holds build/shim)> [ENV=...]   -> one line per test: name wall user+sys
cd /tmp/sp69-inq/runs
label=$1; sim=$2; shift 2
. /tmp/sp69-inq/env.sh
export PSEUDOPOD_SHARE_PATH=/tmp/sp69-inq/inq-src/external_libs/pseudopod/share/ INQ_SHARE_PATH=/tmp/sp69-inq/inq-src/share/
out=/tmp/sp69-inq/runs/$label.tsv; : > $out
for t in "laplacian:[operations][laplacian]" "gradient:[operations][gradient]" "steepest_descent:[eigensolvers][steepest_descent]" "ks_hamiltonian:[hamiltonian][ks_hamiltonian]" "initial_guess:[ground_state][initial_guess]" "transform:[operations][transform]" "electrons:[systems][electrons]" "poisson:[solvers][poisson]"; do
  n=${t%%:*}; d=/tmp/sp69-inq/runs/w_$label; rm -rf $d; mkdir -p $d; cd $d
  /usr/bin/time -f "%e %U %S" -o time.txt env "$@" VGPU_GPU=amd/mi300x LD_LIBRARY_PATH=$sim/build/shim:$LD_LIBRARY_PATH /tmp/sp69-inq/build-hip/src/all_unit_tests "${t#*:}" > out.txt 2> err.txt
  rc=$?
  ok=$(grep -c "All tests passed" out.txt)
  md=$(grep -v -E "^\[20|Randomness|seeded" out.txt | md5sum | cut -c1-8)
  echo "$n $(tail -1 time.txt) rc=$rc pass=$ok out=$md" >> $out
  cd /tmp/sp69-inq/runs; rm -rf $d
done
echo DONE >> $out
