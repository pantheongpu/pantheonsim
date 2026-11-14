#!/bin/bash
# usage: run1.sh <builddir> <workload> [ENV=..]...   one bench workload, full size, 1 thread, mi300x (GPU=... TG=gfx1030 for others)
b=$1; wl=$2; shift 2
bin=$(ls /tmp/sp72-amd-tier/bench-cache/bench_amd.${TG:-gfx942}.* | grep -v "\.tmp" | head -1)
case $wl in p_*) bin=$(ls /tmp/sp72-amd-tier/bench-cache/bench_amd_probe.${TG:-gfx942}.* | grep -v "\.tmp" | head -1); wl=${wl#p_};; esac
t=$(mktemp -d /tmp/sp72-amd-tier/tmp.XXXXXX)
env -i PATH=/usr/bin:/bin VGPU_GPU=${GPU:-amd/mi300x} LD_LIBRARY_PATH=$b/shim VGPU_THREADS=${THREADS:-1} VGPU_QUIET=1 VGPU_TELEMETRY_PATH=$t/t VGPU_STATE_DIR=$t/s "$@" $bin $wl ${SIZE:-full}
rm -rf $t
