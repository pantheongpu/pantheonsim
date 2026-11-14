#!/bin/bash
# usage: inq.sh <label> <builddir (holds shim)> <name> [ENV=...]  -> "label name wall user sys rc pass md5"; stderr of the run kept in out/inq-<label>-<name>.err when KEEPERR=1
label=$1; b=$(readlink -f $2); n=$3; shift 3
case $n in
  laplacian) tags="[operations][laplacian]";; gradient) tags="[operations][gradient]";; steepest_descent) tags="[eigensolvers][steepest_descent]";;
  ks_hamiltonian) tags="[hamiltonian][ks_hamiltonian]";; initial_guess) tags="[ground_state][initial_guess]";; transform) tags="[operations][transform]";;
  electrons) tags="[systems][electrons]";; poisson) tags="[solvers][poisson]";; *) echo "test?"; exit 2;;
esac
. /tmp/sp69-inq/env.sh
export PSEUDOPOD_SHARE_PATH=/tmp/sp69-inq/inq-src/external_libs/pseudopod/share/ INQ_SHARE_PATH=/tmp/sp69-inq/inq-src/share/
d=/tmp/sp72-amd-tier/inq/w_$label-$n; rm -rf $d; mkdir -p $d; cd $d
/usr/bin/time -f "%e %U %S" -o time.txt env "$@" VGPU_GPU=amd/mi300x LD_LIBRARY_PATH=$b/shim:$LD_LIBRARY_PATH /tmp/sp69-inq/build-hip/src/all_unit_tests "$tags" > out.txt 2> err.txt
rc=$?
ok=$(grep -c "All tests passed" out.txt)
md=$(grep -v -E "^\[20|Randomness|seeded" out.txt | md5sum | cut -c1-8)
[ -n "$KEEPERR" ] && cp err.txt /tmp/sp72-amd-tier/out/inq-$label-$n.err
echo "$label $n $(tail -1 time.txt) rc=$rc pass=$ok out=$md"
[ -n "$KEEPDIR" ] || { cd /tmp/sp72-amd-tier; rm -rf $d; }
