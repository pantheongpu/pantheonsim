#!/bin/bash
# bld.sh <builddir> <targets...>: builds under the machine lock in holds of at most 280 s until ninja succeeds.
b=$1; shift
for i in $(seq 1 30); do
  flock /tmp/pantheonsim-builder.lock nice -n 19 timeout 280 ninja -C "$b" -j3 "$@" > "$b.log" 2>&1
  rc=$?
  [ $rc = 0 ] && { echo "built $b: $* (holds: $i)"; tail -1 "$b.log"; exit 0; }
  [ $rc != 124 ] && { echo "FAILED $b rc=$rc"; grep -m5 -B2 -A8 "error" "$b.log"; exit 1; }
  sleep 3
done
echo "gave up $b"; exit 1
