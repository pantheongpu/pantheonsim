#!/bin/bash
# usage: h71.sh <label> <mode: off|cache> <cache dir> [tests...]   -> runs/<label>.tsv: test wall user sys rc pass out-md5 stats
cd /tmp/sp69-inq/runs
label=$1; mode=$2; C=$3; shift 3
. /tmp/sp69-inq/env.sh
export PSEUDOPOD_SHARE_PATH=/tmp/sp69-inq/inq-src/external_libs/pseudopod/share/ INQ_SHARE_PATH=/tmp/sp69-inq/inq-src/share/
out=/tmp/sp69-inq/runs/$label.tsv; : > $out
if [ $mode = off ]; then E="VGPU_HIPRTC_CACHE=0"; else E="VGPU_HIPRTC_CACHE_DIR=$C"; fi
for t in "$@"; do
  case $t in laplacian) tag="[operations][laplacian]";; gradient) tag="[operations][gradient]";; transform) tag="[operations][transform]";; ks_hamiltonian) tag="[hamiltonian][ks_hamiltonian]";; initial_guess) tag="[ground_state][initial_guess]";; electrons) tag="[systems][electrons]";; esac
  d=/tmp/sp69-inq/runs/w_$label.$t; rm -rf $d; mkdir -p $d; cd $d
  /usr/bin/time -f "%e %U %S" -o time.txt env $E VGPU_HIPRTC_CACHE_STATS=1 VGPU_GPU=amd/mi300x LD_LIBRARY_PATH=/tmp/sp69-inq/sim71/build/shim:$LD_LIBRARY_PATH /tmp/sp69-inq/build-hip/src/all_unit_tests "$tag" > out.txt 2> err.txt
  rc=$?
  st=$(sed -n "s/.*hiprtc-cache: \([0-9]*\) compiles: \([0-9]*\) from memory, \([0-9]*\) from [^,]*, \([0-9]*\) by the real library (\([0-9.]*\) s; \([0-9]*\) not cacheable.*/c=\1 mem=\2 disk=\3 real=\4 real_s=\5 nocache=\6/p" err.txt)
  echo "$t $(tail -1 time.txt) rc=$rc pass=$(grep -c "All tests passed" out.txt) out=$(grep -v -E "^\[20|Randomness|seeded" out.txt | md5sum | cut -c1-8) $st" >> $out
  cd /tmp/sp69-inq/runs; rm -rf $d
done
