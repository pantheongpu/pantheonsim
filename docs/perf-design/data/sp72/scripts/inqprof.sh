#!/bin/bash
# usage: inqprof.sh <builddir> <tag> <test>...   samples of an inq test on one host thread (sprof.c), classified -> out/prof-<tag>-<test>.txt
b=$1; tag=$2; shift 2
cd /tmp/sp72-amd-tier; mkdir -p out/samples
for t in "$@"; do
  rm -f out/samples/$tag-$t.*
  taskset -c ${CPUS:-12,13} ./inq.sh p$tag $b $t VGPU_THREADS=1 LD_PRELOAD=/tmp/sp72-amd-tier/libsprof.so SP69_PROF_OUT=/tmp/sp72-amd-tier/out/samples/$tag-$t SP69_PROF_ONLY=all_unit_tests SP69_PROF_HZ=1000 $EXTRA >> out/inqprof-$tag.log 2>&1
  p=$(ls out/samples/$tag-$t.*.samples | head -1); p=${p%.samples}
  python3 -I sprof2sig.py $p > out/samples/$tag-$t.txt
  python3 -I classify.py $b/libamdhip64.so.6.4 out/samples/$tag-$t.txt > out/prof-$tag-$t.txt 2>&1
  rm -f out/samples/$tag-$t.*
done
echo DONE >> out/inqprof-$tag.log
