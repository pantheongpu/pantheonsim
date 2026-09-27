#!/usr/bin/env bash
# Runs AMD's rocblas-test on a simulated GPU, a range of shards of its quick
# tests, two at a time:
#
#   run-shards.sh GPU FILTER FIRST LAST TOTAL LOGDIR
#
# GPU is a profile (amd/mi300x); FILTER "fp" for the float and double tests,
# "mixed" for half, bfloat16 and int8. The environment names the pieces:
# VGPU_BUILD_DIR (its shim/), ROCM_RUN (fetch-rocm.sh's "run" packages) and
# CLIENT (build-client.sh's output). Exits non-zero if any test fails or a
# shard does not finish.
set -uo pipefail
gpu=$1 filter=$2 first=$3 last=$4 total=$5 logs=$6
mkdir -p "$logs"
case $filter in
  fp) f='*/quick_*f32_r*:*/quick_*f64_r*-*f16*:*bf16*:*_c_*:*f32_c*:*f64_c*:*i8*:*f8*:*bf8*' ;;
  mixed) f='*/quick_*f16_r*:*/quick_*bf16_r*:*/quick_*i8_r*-*_c_*:*f8*:*bf8*' ;;
  *) echo "run-shards.sh: FILTER is fp or mixed, not $filter" >&2; exit 2 ;;
esac
# A copy of the shim, its links resolved: it points into the build tree.
shim=$(mktemp -d)
cp -L "$VGPU_BUILD_DIR"/shim/* "$shim"/
export LD_LIBRARY_PATH="$shim:$ROCM_RUN/opt/rocm-7.1.0/lib:$CLIENT"
export VGPU_GPU=$gpu VGPU_QUIET=1 GTEST_TOTAL_SHARDS=$total
cd "$CLIENT"
run() {
  GTEST_SHARD_INDEX=$1 timeout 3600 ./rocblas-test --gtest_filter="$f" --gtest_brief=1 > "$logs/$1.log" 2>&1
  echo "shard $1: exit $? $(grep -h ' ran\. (' "$logs/$1.log" | sed 's/^\[=*\] //')"
}
for ((i = first; i <= last; i += 2)); do
  run "$i" &
  (( i + 1 <= last )) && run $((i + 1)) &
  wait
done
passed=$(grep -h '^\[  PASSED  \]' "$logs"/*.log | awk '{s += $4} END {print s + 0}')
failed=$(grep -h '^\[  FAILED  \] [0-9]* test' "$logs"/*.log | awk '{s += $4} END {print s + 0}')
unfinished=$(grep -L ' ran\. (' "$logs"/*.log | wc -l)
echo "$gpu $filter shards $first-$last of $total: $passed passed, $failed failed, $unfinished unfinished"
grep -h '^\[  FAILED  \] .*\.' "$logs"/*.log | sort -u | head -50
[[ $failed == 0 && $unfinished == 0 && $passed -gt 0 ]]
