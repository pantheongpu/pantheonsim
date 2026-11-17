#!/usr/bin/env bash
# Multi-card turbo timings (docs/perf-design/12-host-gpu-turbo.md section 17). Usage: bench.sh <shim dir> <multibench> <section> ...
#   a <iters>        the plain simulator, turbo with a card for device 0 only (VGPU_HOST_GPU_DEVICES=0: the behaviour before the several-card work) and turbo with a card for
#                    each of 4 devices, one host thread visiting the devices and one thread per device
#   s <iters> <reps> turbo only, 1 to 4 devices (each with its card), both thread patterns: how the time grows with the devices
shim=$1 bin=$2 sec=$3
shift 3
run() {   # <devices> <env...> -- <args>
  local devs=$1 t0 t1
  shift
  t0=$(date +%s.%N)
  env VGPU_GPU=nvidia/t4 VGPU_DEVICE_COUNT=$devs VGPU_QUIET=1 LD_LIBRARY_PATH="$shim" "$@" > /tmp/mb.out 2> /tmp/mb.err
  t1=$(date +%s.%N)
  printf '%7.2f s  out=%s  devices=%s %s\n' "$(echo "$t1 - $t0" | bc -l)" "$(md5sum < /tmp/mb.out | cut -c1-8)" "$devs" "${*/$bin/}"
}
n=1048576
case $sec in
  a)
    it=$1
    for th in 0 1 2; do
      run 4 "$bin" 4 $th $n "$it" 2
      run 4 VGPU_HOST_GPU=turbo VGPU_HOST_GPU_DEVICES=0 "$bin" 4 $th $n "$it" 2
      run 4 VGPU_HOST_GPU=turbo "$bin" 4 $th $n "$it" 2
    done ;;
  s)
    it=$1 reps=$2
    for th in 0 1 2; do
      for d in 1 2 3 4; do
        for ((r = 0; r < reps; ++r)); do run 4 VGPU_HOST_GPU=turbo "$bin" $d $th $n "$it" 2; done
      done
    done ;;
esac
