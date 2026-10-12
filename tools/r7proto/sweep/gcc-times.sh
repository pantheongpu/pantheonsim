#!/usr/bin/env bash
# times g++ on the synthetic generated sources: wall s, user s, max RSS MB
for f in mono_100 multi_100 mono_1000 multi_1000 mono_5000 multi_5000 mono_20000 multi_20000; do
  for o in O0 O1 O2 O3; do
    for isa in avx2; do
      /usr/bin/time -f "$f $o $isa wall=%e user=%U rss_mb=%M" -o t.txt bash -c "ulimit -v 12582912; timeout 900 g++ -$o -mavx2 -mfma -ffp-contract=off -fPIC -shared -o /tmp/claude-1002/r7d/synth/$f.$o.so $f.cpp" 2>/dev/null
      cat t.txt
    done
  done
done
echo finished
