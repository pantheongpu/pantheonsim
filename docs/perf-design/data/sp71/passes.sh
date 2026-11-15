#!/bin/bash
cd /tmp/sp69-inq/runs
T="laplacian gradient transform ks_hamiltonian initial_guess electrons"
C=/tmp/sp69-inq/t71/suite-cache
for pass in 1 2; do
  ./inq_one_mode.sh h${pass}_off off $C $T
  rm -rf $C; ./inq_one_mode.sh h${pass}_cold cache $C $T
  ./inq_one_mode.sh h${pass}_warm cache $C $T
done
echo DONE > m71.done
