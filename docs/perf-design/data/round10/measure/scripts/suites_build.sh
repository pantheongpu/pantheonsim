#!/bin/bash
cd /w; export SUITE_JOBS=3
for s in polybench rodinia hecbench cuda-samples; do
  t=$SECONDS
  git config --global --add safe.directory '*' 2>/dev/null
  ./ci/external/suites.sh fetch $s /w/m/suites/$s > /w/m/suites/$s.fetch.log 2>&1
  ./ci/external/suites.sh build $s /w/m/suites/$s > /w/m/suites/$s.build.log 2>&1
  echo "$s fetch+build $((SECONDS-t)) s" >> /w/m/suites/times.txt
done
echo ALL_BUILT >> /w/m/suites/times.txt
