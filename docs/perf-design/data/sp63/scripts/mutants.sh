#!/bin/bash
# mutants.sh <build> <work>: round sp63's tests on the pristine build and under each of its mutants (needs -DVGPU_JIT_MUTANTS=ON).
# cold = test_jit_cold; wait = test_jit_wait (round sp57's, which the changed wait_over must still pass). rc 0: the test passed.
b=$1; w=$2; mkdir -p "$w"
for m in none prewait_never prewait_stuck prewait_to_counting wait_no_credit wait_stuck history_never history_after trim_no_age trim_no_cap trim_no_touch \
         cachedir_date notice_every_run auto_loud cancel_notice; do
  [ "$m" = none ] && mm= || mm=$m
  VGPU_JIT_MUTANT=$mm TMPDIR=$w timeout 300 "$b/test_jit_cold" > "$w/o1" 2>&1; rc1=$?
  VGPU_JIT_MUTANT=$mm TMPDIR=$w timeout 300 "$b/test_jit_wait" > "$w/o2" 2>&1; rc2=$?
  echo "mutant=$m cold rc=$rc1 wait rc=$rc2 | $(grep -h -m1 -E 'VCHECK[_A-Z]*\(' "$w/o1" | cut -c1-140)"
done
rm -f "$w/o1" "$w/o2"
