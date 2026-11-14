#!/bin/bash
cd /tmp/sp69-inq/runs
T="laplacian gradient transform ks_hamiltonian initial_guess electrons"
C=/tmp/sp69-inq/t71/suite-cache
./inq_one_mode.sh h3_off off $C $T
rm -rf $C; ./inq_one_mode.sh h3_cold cache $C $T
./inq_one_mode.sh h3_warm cache $C $T
# four tests at once, as ctest -j4 runs them: until all four are done
P=/tmp/sp69-inq/t71/par-cache
four() {  # four <label> <mode>
  t0=$(date +%s.%N)
  for t in laplacian gradient transform ks_hamiltonian; do ./inq_one_mode.sh par_$1_$t $2 $P $t & done
  wait
  t1=$(date +%s.%N)
  echo "four-at-once $1: $(echo "$t1 - $t0" | bc) s until all are done; $(cat par_$1_*.tsv | awk "{c+=\$3+\$4; real+=substr(\$10,6); ok+=(\$5==\"rc=0\" && \$6==\"pass=1\")} END{printf \"%.0f CPU-s, %d real compiles, %d of 4 pass\", c, real, ok}"); outputs $(cat par_$1_*.tsv | awk "{print \$7}" | sort | tr "\n" " ")" >> four71.txt
}
: > four71.txt
for r in 1 2; do
  four off$r off
  rm -rf $P; four cold$r cache
  four warm$r cache
done
echo DONE > m71b.done
