#!/bin/bash
# usage: stackshift.sh <workload> <target gfx942|gfx1030> <gpu> <build>...   the workload's rate (knob off, one thread) with the stack moved by 0..4080 bytes
cd /tmp/sp72-amd-tier; wl=$1; tg=$2; gpu=$3; shift 3
mkdir -p pads
for n in 16 272 528 784 1040 1296 1552 1808 2064 2320 2576 2832 3088 3344 3600 3856; do [ -f pads/pad$n.so ] || gcc -O1 -shared -fPIC -DPAD=$n -o pads/pad$n.so tlspad.c; done
printf "%-6s" pad; for b in "$@"; do printf "%14s" $b; done; echo
for n in 16 272 528 784 1040 1296 1552 1808 2064 2320 2576 2832 3088 3344 3600 3856; do
  printf "%-6s" $n
  for b in "$@"; do
    best=999
    for r in 1 2 3 4 5; do
      t=$(TG=$tg GPU=$gpu taskset -c 14,15 ./run1.sh $b/build $wl LD_PRELOAD=/tmp/sp72-amd-tier/pads/pad$n.so $EXTRA 2>/dev/null | grep BENCH | sed "s/.*device_cpu_s=\([0-9.]*\).*/\1/")
      best=$(python3 -c "print(min($best, ${t:-999}))")
    done
    printf "%14s" $best
  done
  echo
done
