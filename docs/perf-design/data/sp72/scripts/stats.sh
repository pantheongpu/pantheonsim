#!/bin/bash
# VGPU_AMD_BLOCKS_STATS of bench rows and inq tests -> out/blocks-stats.txt (the two hottest blocks of each, listings cut at 40 instructions)
cd /tmp/sp72-amd-tier; o=out/blocks-stats.txt; : > $o
for w in valu_int valu_f32 valu_f64 p_i64 p_cmp divergent wave mem narrow lds smem; do
  echo "== bench $w (MI300X)" >> $o
  ./run1.sh work/build $w VGPU_AMD_BLOCKS=1 VGPU_AMD_BLOCKS_STATS=2 2>&1 | grep -A45 "^AMD blocks" | grep -v "^BENCH\|^PASS" | cut -c1-150 >> $o
done
for t in steepest_descent ks_hamiltonian initial_guess poisson; do
  echo "== inq $t (MI300X, one host thread)" >> $o
  KEEPERR=1 taskset -c ${CPUS:-10,11} ./inq.sh st work/build $t VGPU_THREADS=1 VGPU_AMD_BLOCKS=1 VGPU_AMD_BLOCKS_STATS=2 >> out/stats.log 2>&1
  grep -A200 "^AMD blocks" out/inq-st-$t.err | awk '/^AMD blocks/{print; next} /^ +[0-9.]+%/{n=0; print; next} {if (++n <= 40) print; else if (n == 41) print "             ..."}' | cut -c1-150 >> $o
  rm -f out/inq-st-$t.err
done
echo DONE >> out/stats.log
